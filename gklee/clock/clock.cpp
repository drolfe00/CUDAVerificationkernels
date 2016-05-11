 

 
 
 
 
 
 

 
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

 
#include <cuda_runtime.h>

 
 
 

 
 
 
__global__ static void timedReduction(const float *input, float *output, float *shared, clock_t *timer)
{
     
     

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (tid == 0) timer[bid] = clock();

     
    shared[tid] = input[tid];
    shared[tid + blockDim.x] = input[tid + blockDim.x];

     
    for (int d = blockDim.x; d > 0; d /= 2)
    {
        __syncthreads();

        if (tid < d)
        {
            float f0 = shared[tid];
            float f1 = shared[tid + d];

            if (f1 < f0)
            {
                shared[tid] = f1;
            }
        }
    }

     
    if (tid == 0) output[bid] = shared[0];

    __syncthreads();

    if (tid == 0) timer[bid+gridDim.x] = clock();
}


 
 
 
 
 
 

#define NUM_BLOCKS    64
#define NUM_THREADS   256

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

 
int main(int argc, char **argv)
{
    printf("CUDA Clock sample\n");

     
     

    float *dinput = NULL;
    float *doutput = NULL;
    clock_t *dtimer = NULL;

    clock_t timer[NUM_BLOCKS * 2];
    float input[NUM_THREADS * 2];
    __shared__ float shared[NUM_THREADS * 2];

    for (int i = 0; i < NUM_THREADS * 2; i++)
    {
        input[i] = (float)i;
    }
{

     
     
     

     

    __set_CUDAConfig(NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 *NUM_THREADS); 
          


     
     
     

     

    timedReduction(dinput, doutput, shared, dtimer);}
         

     

     
     
     


     
    clock_t minStart = timer[0];
    clock_t maxEnd = timer[NUM_BLOCKS];

    for (int i = 1; i < NUM_BLOCKS; i++)
    {
        minStart = timer[i] < minStart ? timer[i] : minStart;
        maxEnd = timer[NUM_BLOCKS+i] > maxEnd ? timer[NUM_BLOCKS+i] : maxEnd;
    }

    printf("Total clocks = %Lf\n", (long double)(maxEnd - minStart));


     
     
     
     
     
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
