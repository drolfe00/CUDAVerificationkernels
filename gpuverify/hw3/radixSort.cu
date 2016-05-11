
//#include <helper_cuda.h>

//#include <algorithm>
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
void __global__ d_doPrefix(int* __restrict__ d_count, int countLength, int* __restrict__ d_prefix, int prefixLength)
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

void __global__ d_doCount(int* __restrict__ d_unsorted, int unsortedLength, int* __restrict__ d_count, int countLength, int offset)
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
/*
void __global__ d_doReorder(int* __restrict__ d_unsorted, int unsortedLength, int* __restrict__ d_sorted, int sortedLength, int* __restrict__ d_prefix, int prefixLength, int offset)
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
*/

/*
* d_lazyReorder:
* sequential reordering done on the GPU,
*/
void __global__ d_lazyReorder(int* __restrict__ d_unsorted, int unsortedLength, int* __restrict__ d_sorted, int sortedLength, int* __restrict__ d_prefix, int prefixLength, int offset, int threadCount)
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
void __global__ d_lazyReorderorig(int* __restrict__ d_unsorted, int unsortedLength, int* __restrict__ d_sorted, int sortedLength, int* __restrict__ d_prefix, int prefixLength, int offset)
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


