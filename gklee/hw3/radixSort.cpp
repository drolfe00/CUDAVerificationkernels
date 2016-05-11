
 

#include <algorithm>
#include <time.h>
#include <limits.h>

 
 
#define RADIX 65536

 
#define numElements 30000

#define numIterations 10
#define BLOCKSIZE 128




 
void __global__ d_doPrefix(int *d_count, int countLength, int *d_prefix, int prefixLength)
{
  
  
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
  
   
}

void __global__ d_doCount(int *d_unsorted, int unsortedLength, int *d_count, int countLength, int offset)
{
   
  int index = threadIdx.x + blockIdx.x * blockDim.x;
   
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
     
    atomicAdd(d_count + numToSort, 1);
  }   
  
   

}

 
void __global__ d_doReorder(int* d_unsorted, int unsortedLength, int *d_sorted, int sortedLength, int *d_prefix, int prefixLength, int offset)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if( index <unsortedLength)
  {

    int currentNum;
    int newIndex;
    int prefix;
     

     
    currentNum = d_unsorted[index];
    currentNum = currentNum >> offset;
    currentNum = (prefixLength -1) & currentNum;
    
    if (currentNum < prefixLength)
      prefix = d_prefix[currentNum];
     
       
    
    newIndex = index % prefix;
     
    
    d_sorted[newIndex] = d_unsorted[index];
     

  }
}


 
void __global__ d_lazyReorder(int* d_unsorted, int unsortedLength, int *d_sorted, int sortedLength, int *d_prefix, int prefixLength, int offset, int threadCount)
{
 
   
  
   
  int loopMax = ceil((float)unsortedLength/(float)threadCount);
  int currentNum;
  int newIndex;
  if(threadIdx.x < 1)
  {
    for (int i =0; i<unsortedLength; i++)
    {

       
      currentNum = d_unsorted[i];
      currentNum = currentNum >> offset;
      currentNum = (prefixLength -1) & currentNum;

      newIndex = d_prefix[currentNum];
      d_prefix[currentNum]++;
   
      d_sorted[newIndex] = d_unsorted[i];
    

       

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

 
void __global__ d_lazyReorderorig(int* d_unsorted, int unsortedLength, int *d_sorted, int sortedLength, int *d_prefix, int prefixLength, int offset)
{
 
   
  
  int currentNum;
  int newIndex;
  for (int i =0; i<unsortedLength; i++)
  {

     
    currentNum = d_unsorted[i];
    currentNum = currentNum >> offset;
    currentNum = (prefixLength -1) & currentNum;

    newIndex = d_prefix[currentNum];
    d_prefix[currentNum]++;
   
    d_sorted[newIndex] = d_unsorted[i];
    

     

  }
  for (int i =0; i<unsortedLength; i++)
  {
    d_unsorted[i] = d_sorted[i];
  }
}

 
 
 
 
 

 
void cudaRadix(int *h_unsorted, int *h_sorted)
{
   

  int sortBits, countLength, prefixLength;
  int countSize,  unsortedSize, sortedSize, prefixSize;
   
  int *d_count, *d_unsorted, *d_sorted, *d_prefix;
   
   
 
   
  sortBits = 11;
  countLength = 1 << sortBits;
   
  prefixLength = countLength;

  countSize = (1 << sortBits)*sizeof(int);
   
  prefixSize = countSize;
  unsortedSize = numElements*sizeof(int);
  sortedSize = unsortedSize; 
  
   

   

   
  
   

   
  cudaMalloc((void**) &d_unsorted, unsortedSize);
  if (! d_unsorted)
  {
     
    exit(-1); 
  }
  cudaMalloc((void**) &d_sorted, sortedSize);
  if (! d_sorted)
  {
     
    exit(-1); 
  }
  cudaMalloc((void**) &d_count, countSize);
  if (! d_count)
  {
     
    exit(-1); 
  }
  cudaMalloc((void**) &d_prefix, prefixSize);
  if (! d_prefix)
  {
     
    exit(-1); 
  }
  

   
  

 
   
  cudaMemcpy(d_unsorted, h_unsorted, unsortedSize, cudaMemcpyHostToDevice);
   

   
   
   

   
   
  cudaMemcpy(d_sorted, h_sorted, sortedSize, cudaMemcpyHostToDevice);
   

   
  cudaEvent_t cuda_start_event, cuda_stop_event;
   
   
   


   
  dim3 threadsPerBlock(BLOCKSIZE);
  dim3 numBlocks(ceil((float)numElements/(float)threadsPerBlock.x));

  dim3 numPrefixBlocks(ceil((float)countLength/(float)threadsPerBlock.x));
  for(int sortLoop=0; sortLoop<32; sortLoop += sortBits)
  {
{ __set_CUDAConfig(numBlocks, threadsPerBlock ); 
          
 d_doCount (d_unsorted, numElements, d_count, countLength, sortLoop);}
          
{ __set_CUDAConfig(numPrefixBlocks, threadsPerBlock ); 
          
 d_doPrefix (d_count, countLength, d_prefix, prefixLength);}
          
{ __set_CUDAConfig(1, 1024 ); 
          
 d_lazyReorder (d_unsorted, numElements, d_sorted, numElements, d_prefix, prefixLength, sortLoop, 1024);}
          
   
   
  }
 
   
   

  float cuda_time = 0;
   
  cuda_time /= 1.0e3f;



   
  cudaMemcpy(h_unsorted, d_unsorted, unsortedSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sorted, d_sorted, sortedSize, cudaMemcpyDeviceToHost);
   
   
   
  

   
  cudaFree(d_unsorted);
  cudaFree(d_count);
  cudaFree(d_prefix);
  cudaFree(d_sorted);
   

   

   
}



int
main(int argc, char **argv)
{
  int *unsorted, *sorted;
   

   
  unsorted = (int *) malloc (numElements*sizeof(int));
  sorted = (int *) malloc (numElements*sizeof(int));
   
   
  for (int i=0; i<numElements; i++) {
    unsorted[i] = (int) (rand() % RADIX);
  }
 
   
  int *h_unsorted, *h_sorted;
   

  h_unsorted = (int *) malloc (numElements*sizeof(int));
  h_sorted = (int *) malloc (numElements*sizeof(int));

  for(int i=0; i<numElements; i++)
  {
    h_unsorted[i] = unsorted[i];
  }


   
 
   
   
  cudaRadix(h_unsorted, h_sorted);

}

