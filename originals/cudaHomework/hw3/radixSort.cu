#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>

#include <helper_cuda.h>

#include <algorithm>
#include <time.h>
#include <limits.h>

//#define RADIX 4294967296
#define RADIX 2147483658
//#define RADIX 65536
#define numElements 1048576
//#define numElements 30000
#define numIterations 10
#define BLOCKSIZE 128

//helpers

/*
*compare:
* compares two arrays of equal length for equality
*/
int compare (int *first, int *second, long count)
{
  for( int  i=0; i<count; i++)
  {
    if((int)first[i] != (int)second[i])
    {
      printf("items in compare are not equal at index %d, first: %d, second: %d", i, first[i], second[i]);
      return -1;
    }
  }
  return 1;
}


/*
*printArray:
*prints the contents of an array in hex format
*/
void printArray(int * array, int length)
{
  for( int i =0; i<length; i+=10)
  {
    for( int j=i; j<i+10; j++)
    {
      if(j >= length)
        break;
      printf(" %#010x ", array[j]);
    }
    printf(" \n");
  }

}

// end helpers

/*
*This section is the original seqential algorithm
*split into the 3 main stages, I was hoping
* this would make it easier to compare the sequential
* and cuda outputs
*
*
*/

/*
* the sort of the seqential algorithm
*/
void 
sequentialSort(int *unsorted, int *sorted)
{
   int *count, *prefix;

  // count number of entries for each value
  count = (int *) malloc (RADIX*sizeof(int));
  for (unsigned int i=0; i<RADIX; i++) count[i]=0;
  for (int i=0; i<numElements; i++) {
    count[unsorted[i]]++;
  }

  // prefix sum of count
  prefix = (int *) malloc (RADIX*sizeof(int));
  prefix[0] = 0;
  for (unsigned int i=1; i<RADIX; i++) {
    prefix[i] = prefix[i-1] + count[i];
  }
  
  // generate result

  int curr = 0;
  for (unsigned int i=0; i<RADIX; i++) {
    for (int j=0; j<count[i]; j++) {
      sorted[curr++] = i;
    }
  }

}


/*
sequential count
*/
void sequentialCount(int *unsorted, int *count)
{
  //int *count;

  // count number of entries for each value
  //count = (int *) malloc (RADIX*sizeof(int));
  for (unsigned int i=0; i<RADIX; i++) count[i]=0;
  for (int i=0; i<numElements; i++) {
    count[unsorted[i]]++;
  }
  //return count;

}

/*
* sequential prefix sum
*/
void sequentialPrefixSum(int *count, int *prefix)
{
  //int * prefix;
  

  // prefix sum of count
  //prefix = (int *) malloc (RADIX*sizeof(int));
  prefix[0] = 0;
  for (unsigned int i=1; i<RADIX; i++) {
    prefix[i] = prefix[i-1] + count[i];
  }
  //return prefix;

}

void sequentialRadix(int *count, int *sorted)
{
  // generate result

  int curr = 0;
  for (unsigned int i=0; i<RADIX; i++) {
    for (int j=0; j<count[i]; j++) {
      sorted[curr++] = i;
    }
  }
  //return sorted;
} 



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
    printf("prefix check: prefix = %d masked number = %d,  real number = %d, index = %d, newIndex = %d \n", prefix, currentNum, d_unsorted[index], index, newIndex); 
    
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
    printf("stdErr: unable to cuda malloc %d bytes for unsorted", unsortedSize);
    exit(-1); 
  }
  cudaMalloc((void**) &d_sorted, sortedSize);
  if (! d_sorted)
  {
    printf("stdErr: unable to cuda malloc %d bytes for sorted", sortedSize);
    exit(-1); 
  }
  cudaMalloc((void**) &d_count, countSize);
  if (! d_count)
  {
    printf("stdErr: unable to cuda malloc %d bytes for count", countSize);
    exit(-1); 
  }
  cudaMalloc((void**) &d_prefix, prefixSize);
  if (! d_prefix)
  {
    printf("stdErr: unable to cuda malloc %d bytes for prefix", prefixSize);
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
  checkCudaErrors(cudaEventCreate(&cuda_start_event));
  checkCudaErrors(cudaEventCreate(&cuda_stop_event));
  checkCudaErrors(cudaEventRecord(cuda_start_event, 0));


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
  checkCudaErrors(cudaEventRecord(cuda_stop_event, 0));
  checkCudaErrors(cudaEventSynchronize(cuda_stop_event));

  float cuda_time = 0;
  checkCudaErrors(cudaEventElapsedTime(&cuda_time, cuda_start_event, cuda_stop_event));
  cuda_time /= 1.0e3f;
  printf("radixSort (CUDA) within kernel, Throughput = %.4f KElements/s, Time = %.5f s, Size = %u elements\n",
           1.0e-3f * numElements / cuda_time, cuda_time, numElements);


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

  //initialize list for Thrust
  thrust::host_vector<int> h_keys(numElements);
  thrust::host_vector<int> h_keysSorted(numElements);
  for (int i = 0; i < (int)numElements; i++)
     h_keys[i] = unsorted[i];

  // initialize items for cuda
  int *h_unsorted, *h_sorted;
  //int *h_count, *h_prefix;

  h_unsorted = (int *) malloc (numElements*sizeof(int));
  h_sorted = (int *) malloc (numElements*sizeof(int));

  for(int i=0; i<numElements; i++)
  {
    h_unsorted[i] = unsorted[i];
  }
  

  // SEQUENTIAL RUN
  cudaEvent_t seq_start_event, seq_stop_event;
  checkCudaErrors(cudaEventCreate(&seq_start_event));
  checkCudaErrors(cudaEventCreate(&seq_stop_event));
  checkCudaErrors(cudaEventRecord(seq_start_event, 0));

  // TODO: THIS TAKES A FEW MINUTES AND SHOULD BE COMMENTED OUT FOR TESTING

  //count
  /*
  sequentialCount(unsorted, count);
  printf("sequential count array \n"); 
  printArray(count, RADIX);

  //compare(count, count, RADIX*sizeof(int));
  

  //prefix
  
  sequentialPrefixSum(count, prefix);
  printf("sequential prefix array \n"); 
  printArray(prefix, RADIX);
  

  //radix
  sequentialRadix(count, sorted);
*/
   
  (void) sequentialSort(unsorted,sorted);

  checkCudaErrors(cudaEventRecord(seq_stop_event, 0));
  checkCudaErrors(cudaEventSynchronize(seq_stop_event));

  float seq_time = 0;
  checkCudaErrors(cudaEventElapsedTime(&seq_time, seq_start_event, seq_stop_event));
  seq_time /= 1.0e3f;
  printf("radixSort (SEQ), Throughput = %.4f KElements/s, Time = %.5f s, Size = %u elements\n",
           1.0e-3f * numElements / seq_time, seq_time, numElements);




  printf ("starting thrust \n");
  // THRUST IMPLEMENTATION
  // copy onto GPU
  thrust::device_vector<int> d_keys;
    
  cudaEvent_t start_event, stop_event;
  checkCudaErrors(cudaEventCreate(&start_event));
  checkCudaErrors(cudaEventCreate(&stop_event));

  float totalTime = 0;
  // run multiple iterations to compute an average sort time
  for (int i = 0; i < numIterations; i++) {
        // reset data before sort
        d_keys= h_keys;

        checkCudaErrors(cudaEventRecord(start_event, 0));

        thrust::sort(d_keys.begin(), d_keys.end());

        checkCudaErrors(cudaEventRecord(stop_event, 0));
        checkCudaErrors(cudaEventSynchronize(stop_event));

        float time = 0;
        checkCudaErrors(cudaEventElapsedTime(&time, start_event, stop_event));
        totalTime += time;
    }

    totalTime /= (1.0e3f * numIterations);
    printf("radixSort in THRUST, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements\n",
           1.0e-6f * numElements / totalTime, totalTime, numElements);

    getLastCudaError("after radixsort");

    // Get results back to host for correctness checking
    thrust::copy(d_keys.begin(), d_keys.end(), h_keysSorted.begin());

    getLastCudaError("copying results to host memory");



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
/*
  checkCudaErrors(cudaEventRecord(cuda_stop_event, 0));
  checkCudaErrors(cudaEventSynchronize(cuda_stop_event));

  float cuda_time = 0;
  checkCudaErrors(cudaEventElapsedTime(&cuda_time, cuda_start_event, cuda_stop_event));
  cuda_time /= 1.0e3f;
  printf("radixSort (CUDA), Throughput = %.4f KElements/s, Time = %.5f s, Size = %u elements\n",
           1.0e-3f * numElements / cuda_time, cuda_time, numElements);

//end global timing
*/
/*
  // output compare
  printf("output compare \n sequential \n");
  printArray(sorted, numElements);

  printf("cuda \n");
  printArray(h_sorted, numElements);

  printf("\n \n \n \n unsorted inputs \n");
  printArray(unsorted, numElements);
*/


/*
  printf("\n \n \n \n unsorted inputs");
  if(compare(h_unsorted, unsorted, numElements))
  {
    printf("the unsorted starts are equal \n");
  }
  printArray(h_unsorted, numElements);
  printf("\n ");
  printArray(unsorted, numElements);
*/



    // Check results
    bool bTestResult = thrust::is_sorted(h_keysSorted.begin(), h_keysSorted.end());

    checkCudaErrors(cudaEventDestroy(start_event));
    checkCudaErrors(cudaEventDestroy(stop_event));

    if (bTestResult) printf("THRUST: VALID!\n");

    // COMPARE SEQUENTIAL WITH THRUST
   bTestResult = true;
   for (int i = 0; i < (int)numElements; i++) {
     if (h_keysSorted[i] != sorted[i]) {
       bTestResult = false;
       break;
     }
   }
   if (bTestResult) printf("SEQ: VALID!\n");


    // COMPARE SEQUENTIAL WITH CUDA
   bTestResult = true;
   for (int i = 0; i < (int)numElements; i++) {
     if (h_sorted[i] != sorted[i]) {
       bTestResult = false;
       break;
     }
   }
   if (bTestResult) printf("CUDA: VALID!\n");

}

