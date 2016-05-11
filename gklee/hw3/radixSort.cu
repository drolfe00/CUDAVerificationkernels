
//#include <helper_cuda.h>

#include <algorithm>
#include <time.h>
#include <limits.h>

//#define RADIX 4294967296
//#define RADIX 2147483658
#define RADIX 65536

//#define numElements 1048576
#define numElements 30000

#define numIterations 10
#define BLOCKSIZE 128




// countlength/threadsperblock
void __global__ d_doPrefix(int *d_count, int countLength, int *d_prefix, int prefixLength)
{
 // printf("do prefix = %d \n", threadIdx.x);
  
  int sum = 0;
  int index = threadIdx.x + blockIdx.x * blockDim.x;


  if(index < prefixLength)
  {
    d_prefix[index] = 0;
  }
  __syncthreads();

  for(int i=index; i>=0; i--)
  {
    sum += d_count[i];
  }

  if(index < prefixLength) 
    atomicAdd(d_prefix +index+1, sum);
  
  //printf("finished doPrefix \n");
}

void __global__ d_doCount(int *d_unsorted, int unsortedLength, int *d_count, int countLength, int offset)
{
  //printf("do count \n");
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  //printf("index = %d \n", index);
  if(index <countLength)
  {
    d_count[index] = 0;
  }
  __syncthreads();
  if(index < unsortedLength)
  {
    int numToSort = d_unsorted[index];
    numToSort = numToSort >> offset;
    numToSort = (countLength-1)&(numToSort); 
    //printf("num = %d \n", numToSort);
    atomicAdd(d_count + numToSort, 1);
  }   
  
  //printf("finished count \n");

}

/*
* d_doReorder:
* leftover from an attempt to find a parallel reorder strategy
* did not get this working
*/
void __global__ d_doReorder(int* d_unsorted, int unsortedLength, int *d_sorted, int sortedLength, int *d_prefix, int prefixLength, int offset)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if( index <unsortedLength)
  {

    int currentNum;
    int newIndex;
    int prefix;
    //printf(" doReorder index %d \n", index);

    // shifting and masking
    currentNum = d_unsorted[index];
    currentNum = currentNum >> offset;
    currentNum = (prefixLength -1) & currentNum;
    
    if (currentNum < prefixLength)
      prefix = d_prefix[currentNum];
    //else
      //prefix = sortedLength;
    
    newIndex = index % prefix;
    //printf("prefix check: prefix = %d masked number = %d,  real number = %d, index = %d, newIndex = %d \n", prefix, currentNum, d_unsorted[index], index, newIndex); 
    
    d_sorted[newIndex] = d_unsorted[index];
    //d_unsorted = d_sorted;

  }
}


/*
* d_lazyReorder:
* sequential reordering done on the GPU,
*/
void __global__ d_lazyReorder(int* d_unsorted, int unsortedLength, int *d_sorted, int sortedLength, int *d_prefix, int prefixLength, int offset, int threadCount)
{
 
  //printf("lazy sort prefixlength %d, offset %d \n", prefixLength, offset);
  
  //int index = threadIdx.x + blockIdx.x * blockDim.x;
  int loopMax = ceil((float)unsortedLength/(float)threadCount);
  int currentNum;
  int newIndex;
  if(threadIdx.x < 1)
  {
    for (int i =0; i<unsortedLength; i++)
    {

      // shifting and masking
      currentNum = d_unsorted[i];
      currentNum = currentNum >> offset;
      currentNum = (prefixLength -1) & currentNum;

      newIndex = d_prefix[currentNum];
      d_prefix[currentNum]++;
   
      d_sorted[newIndex] = d_unsorted[i];
    

      //d_unsorted = d_sorted;    

    }
  }
  __syncthreads();
  for (int i =0; i<loopMax; i++)
  {
    int index = threadIdx.x*loopMax + i;
    if( index < sortedLength)
      d_unsorted[index] = d_sorted[index];
  }
}

/*
* d_lazyReorderorig:
* sequential reordering done on the GPU,
*/
void __global__ d_lazyReorderorig(int* d_unsorted, int unsortedLength, int *d_sorted, int sortedLength, int *d_prefix, int prefixLength, int offset)
{
 
  //printf("lazy sort prefixlength %d, offset %d \n", prefixLength, offset);
  
  int currentNum;
  int newIndex;
  for (int i =0; i<unsortedLength; i++)
  {

    // shifting and masking
    currentNum = d_unsorted[i];
    currentNum = currentNum >> offset;
    currentNum = (prefixLength -1) & currentNum;

    newIndex = d_prefix[currentNum];
    d_prefix[currentNum]++;
   
    d_sorted[newIndex] = d_unsorted[i];
    

    //d_unsorted = d_sorted;    

  }
  for (int i =0; i<unsortedLength; i++)
  {
    d_unsorted[i] = d_sorted[i];
  }
}

// allocate space
// copy from host to dev
// run kernel
// copay from dev to host
// free space

/*
* cudaRadix:
* master function for the cuda implementation
* sets up the resources and starts the kernels
*/
void cudaRadix(int *h_unsorted, int *h_sorted)
{
  //printf("started cudaRadix \n");

  int sortBits, countLength, prefixLength;
  int countSize,  unsortedSize, sortedSize, prefixSize;
  //int *h_count, *d_count, *d_unsorted, *d_sorted, *h_prefix, *d_prefix;
  int *d_count, *d_unsorted, *d_sorted, *d_prefix;
  //int *zeros;
  //int zerosSize, zerosLength; 
 
  //sortBits = 4;
  sortBits = 11;
  countLength = 1 << sortBits;
  //zerosLength = 1 << sortBits;
  prefixLength = countLength;

  countSize = (1 << sortBits)*sizeof(int);
  //zerosSize = (1 << sortBits)*sizeof(int);
  prefixSize = countSize;
  unsortedSize = numElements*sizeof(int);
  sortedSize = unsortedSize; 
  
  //printf("count size is = %d \n", countSize);

  //zeros = (int *) malloc (zerosSize);

  //for (unsigned int i=0; i<zerosLength; i++) zeros[i]=0;
  
  //printArray(h_count, countLength);

  // allocate device space
  cudaMalloc((void**) &d_unsorted, unsortedSize);
  if (! d_unsorted)
  {
    //printf("stdErr: unable to cuda malloc %d bytes for unsorted", unsortedSize);
    exit(-1); 
  }
  cudaMalloc((void**) &d_sorted, sortedSize);
  if (! d_sorted)
  {
    //printf("stdErr: unable to cuda malloc %d bytes for sorted", sortedSize);
    exit(-1); 
  }
  cudaMalloc((void**) &d_count, countSize);
  if (! d_count)
  {
    //printf("stdErr: unable to cuda malloc %d bytes for count", countSize);
    exit(-1); 
  }
  cudaMalloc((void**) &d_prefix, prefixSize);
  if (! d_prefix)
  {
    //printf("stdErr: unable to cuda malloc %d bytes for prefix", prefixSize);
    exit(-1); 
  }
  

  //printf("passed mallocs \n");
  

// copy from host to dev
  // count
  cudaMemcpy(d_unsorted, h_unsorted, unsortedSize, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_count, h_count, countSize, cudaMemcpyHostToDevice);

  //cudaMemcpy(d_count, zeros, countSize, cudaMemcpyHostToDevice);
  // prefix
  //cudaMemcpy(d_prefix, h_prefix, prefixSize, cudaMemcpyHostToDevice);

  //cudaMemcpy(d_prefix, zeros, prefixSize, cudaMemcpyHostToDevice);
  // reorder
  cudaMemcpy(d_sorted, h_sorted, sortedSize, cudaMemcpyHostToDevice);
  //printf("passed copy to dev \n");

  // timing without mem copy
  cudaEvent_t cuda_start_event, cuda_stop_event;
  //checkCudaErrors(cudaEventCreate(&cuda_start_event));
  //checkCudaErrors(cudaEventCreate(&cuda_stop_event));
  //checkCudaErrors(cudaEventRecord(cuda_start_event, 0));


  // setup kernel count
  dim3 threadsPerBlock(BLOCKSIZE);
  dim3 numBlocks(ceil((float)numElements/(float)threadsPerBlock.x));

  dim3 numPrefixBlocks(ceil((float)countLength/(float)threadsPerBlock.x));
  for(int sortLoop=0; sortLoop<32; sortLoop += sortBits)
  {
  // run kernel count
  //cudaMemcpy(d_count, zeros, countSize, cudaMemcpyHostToDevice);
  d_doCount <<< numBlocks, threadsPerBlock >>> (d_unsorted, numElements, d_count, countLength, sortLoop);
  
  // setup and run prefix kernel
  //cudaMemcpy(d_prefix, zeros, countSize, cudaMemcpyHostToDevice);
  d_doPrefix <<< numPrefixBlocks, threadsPerBlock >>> (d_count, countLength, d_prefix, prefixLength);

  //run the reorder kernel
  //void __global__ d_doReorder(int* d_unsorted, int unsortedLength, int *d_sorted, int sortedLength, int *d_prefix, int prefixLength, int offset)
  //d_doReorderorig <<< numBlocks, threadsPerBlock >>> (d_unsorted, numElements, d_sorted, numElements, d_prefix, prefixLength, 0);
  d_lazyReorder <<< 1, 1024 >>>(d_unsorted, numElements, d_sorted, numElements, d_prefix, prefixLength, sortLoop, 1024);
  //d_lazyReorderorig <<< 1, 1 >>>(d_unsorted, numElements, d_sorted, numElements, d_prefix, prefixLength, sortLoop);
  //printf("passed kernels \n")
  }
// timing
  //checkCudaErrors(cudaEventRecord(cuda_stop_event, 0));
  //checkCudaErrors(cudaEventSynchronize(cuda_stop_event));

  float cuda_time = 0;
  //checkCudaErrors(cudaEventElapsedTime(&cuda_time, cuda_start_event, cuda_stop_event));
  cuda_time /= 1.0e3f;



  // copy from dev to host
  cudaMemcpy(h_unsorted, d_unsorted, unsortedSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sorted, d_sorted, sortedSize, cudaMemcpyDeviceToHost);
  //cudaMemcpy(h_count, d_count, countSize, cudaMemcpyDeviceToHost);
  //cudaMemcpy(h_prefix, d_prefix, prefixSize, cudaMemcpyDeviceToHost);
  //printf("passed host to dev cpy \n");
  

  // free space
  cudaFree(d_unsorted);
  cudaFree(d_count);
  cudaFree(d_prefix);
  cudaFree(d_sorted);
  //printf("passed cuda free \n");

  //printf("finished cudaRadix \n");

  /* 
  printf("cuda count array \n"); 
  printArray(h_count, countLength);
  

  printf("cuda prefix array \n");
  //sequentialPrefixSum(h_prefix, h_count);
  printArray(h_prefix, prefixLength);
  */
}



int
main(int argc, char **argv)
{
  int *unsorted, *sorted;
  //int *count, *prefix;

  // initialize list.  Value in range 0..RADIX
  unsorted = (int *) malloc (numElements*sizeof(int));
  sorted = (int *) malloc (numElements*sizeof(int));
  //count = (int *) malloc (RADIX*sizeof(int));
  //prefix = (int *) malloc (RADIX*sizeof(int));
  for (int i=0; i<numElements; i++) {
    unsorted[i] = (int) (rand() % RADIX);
  }
/*
  //initialize list for Thrust
  thrust::host_vector<int> h_keys(numElements);
  thrust::host_vector<int> h_keysSorted(numElements);
  for (int i = 0; i < (int)numElements; i++)
     h_keys[i] = unsorted[i];
*/
  // initialize items for cuda
  int *h_unsorted, *h_sorted;
  //int *h_count, *h_prefix;

  h_unsorted = (int *) malloc (numElements*sizeof(int));
  h_sorted = (int *) malloc (numElements*sizeof(int));

  for(int i=0; i<numElements; i++)
  {
    h_unsorted[i] = unsorted[i];
  }


  // CUDA RUN
/*
// start global timing
  cudaEvent_t cuda_start_event, cuda_stop_event;
  checkCudaErrors(cudaEventCreate(&cuda_start_event));
  checkCudaErrors(cudaEventCreate(&cuda_stop_event));
  checkCudaErrors(cudaEventRecord(cuda_start_event, 0));
*/
   
  //(void) sequentialSort(unsorted,sorted);
  cudaRadix(h_unsorted, h_sorted);

}

