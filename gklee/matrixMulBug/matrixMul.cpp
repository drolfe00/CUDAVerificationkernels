 

 

 
#include <stdio.h>
#include <assert.h>

 
#include <cuda_runtime.h>

 
 

 
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(int *C, int *A, int *B, int wA, int wB)
{
     
    int bx = blockIdx.x;
    int by = blockIdx.y;

     
    int tx = threadIdx.x;
    int ty = threadIdx.y;

     
    int aBegin = wA * BLOCK_SIZE * by;

     
    int aEnd   = aBegin + wA - 1;

     
    int aStep  = BLOCK_SIZE;

     
    int bBegin = BLOCK_SIZE * bx;

     
    int bStep  = BLOCK_SIZE * wB;

     
     
    float Csub = 0;

     
     
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

         
         
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

         
         
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

         
         
         
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

         
         

         
         
         
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

         
         
         
        __syncthreads();
    }

     
     
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void constantInit(int *data, int size, int val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

 
int main(int argc, char **argv)
{
    int block_size = 32;

    dim3 dimsA(1*1*block_size, 1*1*block_size, 1);
    dim3 dimsB(1*2*block_size, 1*1*block_size, 1);

     
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(int) * size_A;
    int *h_A = (int *)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(int) * size_B;
    int *h_B = (int *)malloc(mem_size_B);
    const float valB = 0.01f;

     
    constantInit(h_A, size_A, 1);
    constantInit(h_B, size_B, 1);

     
    int *d_A, *d_B, *d_C;

     
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(int);
    int *h_C = (int *) malloc(mem_size_C);

    if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    cudaError_t error;

    error = cudaMalloc((void **) &d_A, mem_size_A);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_B, mem_size_B);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_C, mem_size_C);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

     
    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

     
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

     
    printf("Computing result using CUDA Kernel...\n");

     
    if (block_size == 16)
    {
{ __set_CUDAConfig(grid, threads ); 
          
 matrixMulCUDA<16>(d_C, d_A, d_B, dimsA.x, dimsB.x);}
          
    }
    else
    {
{ __set_CUDAConfig(grid, threads ); 
          
 matrixMulCUDA<32>(d_C, d_A, d_B, dimsA.x, dimsB.x);}
          
    }

    printf("done\n");

    cudaDeviceSynchronize();

     
    cudaEvent_t start;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

     
    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

     
    int nIter = 300;

    for (int j = 0; j < nIter; j++)
    {
        if (block_size == 16)
        {
{ __set_CUDAConfig(grid, threads ); 
          
 matrixMulCUDA<16>(d_C, d_A, d_B, dimsA.x, dimsB.x);}
          
        }
        else
        {
{ __set_CUDAConfig(grid, threads ); 
          
 matrixMulCUDA<32>(d_C, d_A, d_B, dimsA.x, dimsB.x);}
          
        }
    }

     
    error = cudaEventRecord(stop, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

     
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

     
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);

     
    error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_C,d_C) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("Checking computed result for correctness: ");
    bool correct = true;

     
     
    double eps = 1.e-6 ;  
    for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
    {
        double abs_err = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err/abs_val/dot_length ;
        if (rel_err > eps)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x*valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

     
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\nNote: For peak performance, please refer to the matrixMulCUBLAS example.\n");

    cudaDeviceReset();

    if (correct)
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
}

