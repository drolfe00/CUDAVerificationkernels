 

 



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 
 


 
 
 
extern"C" void fwtCPU(float *h_Output, float *h_Input, int log2N);
extern"C" void slowWTcpu(float *h_Output, float *h_Input, int log2N);
extern "C" void dyadicConvolutionCPU(
    float *h_Result,
    float *h_Data,
    float *h_Kernel,
    int log2dataN,
    int log2kernelN
);


 
 
 
 

#define ELEMENTARY_LOG2SIZE 11

__extern__shared__ float s_data[];
__global__ void fwtBatch1Kernel(int *d_Output, int *d_Input, int log2N)
{
    const int    N = 1 << log2N;
    const int base = blockIdx.x << log2N;

     
    int *d_Src = d_Input  + base;
    int *d_Dst = d_Output + base;

    for (int pos = threadIdx.x; pos < N; pos += blockDim.x)
    {
        s_data[pos] = d_Src[pos];
    }

     
    const int pos = threadIdx.x;

    for (int stride = N >> 2; stride > 0; stride >>= 2)
    {
        int lo = pos & (stride - 1);
        int i0 = ((pos - lo) << 2) + lo;
        int i1 = i0 + stride;
        int i2 = i1 + stride;
        int i3 = i2 + stride;

        __syncthreads();
        float D0 = s_data[i0];
        float D1 = s_data[i1];
        float D2 = s_data[i2];
        float D3 = s_data[i3];

        float T;
        T = D0;
        D0         = D0 + D2;
        D2         = T - D2;
        T = D1;
        D1         = D1 + D3;
        D3         = T - D3;
        T = D0;
        s_data[i0] = D0 + D1;
        s_data[i1] = T - D1;
        T = D2;
        s_data[i2] = D2 + D3;
        s_data[i3] = T - D3;
    }

     
    if (log2N & 1)
    {
        __syncthreads();

        for (int pos = threadIdx.x; pos < N / 2; pos += blockDim.x)
        {
            int i0 = pos << 1;
            int i1 = i0 + 1;

            float D0 = s_data[i0];
            float D1 = s_data[i1];
            s_data[i0] = D0 + D1;
            s_data[i1] = D0 - D1;
        }
    }

    __syncthreads();

    for (int pos = threadIdx.x; pos < N; pos += blockDim.x)
    {
        d_Dst[pos] = s_data[pos];
    }
}

 
 
 
 
__global__ void fwtBatch2Kernel(
    int *d_Output,
    int *d_Input,
    int stride
)
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int   N = blockDim.x *  gridDim.x * 4;

    int *d_Src = d_Input  + blockIdx.y * N;
    int *d_Dst = d_Output + blockIdx.y * N;

    int lo = pos & (stride - 1);
    int i0 = ((pos - lo) << 2) + lo;
    int i1 = i0 + stride;
    int i2 = i1 + stride;
    int i3 = i2 + stride;

    int D0 = d_Src[i0];
    int D1 = d_Src[i1];
    int D2 = d_Src[i2];
    int D3 = d_Src[i3];

    float T;
    T = D0;
    D0        = D0 + D2;
    D2        = T - D2;
    T = D1;
    D1        = D1 + D3;
    D3        = T - D3;
    T = D0;
    d_Dst[i0] = D0 + D1;
    d_Dst[i1] = T - D1;
    T = D2;
    d_Dst[i2] = D2 + D3;
    d_Dst[i3] = T - D3;
}

 
 
 
void fwtBatchGPU(int *d_Data, int M, int log2N)
{
    const int THREAD_N = 256;

    int N = 1 << log2N;
    dim3 grid((1 << log2N) / (4 * THREAD_N), M, 1);

    for (; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2)
    {
{ __set_CUDAConfig(grid, THREAD_N); 
          
 fwtBatch2Kernel(d_Data, d_Data, N / 4);}
          
         
        printf("fwtBatch2Kernel() execution failed\n");
    }
{

    __set_CUDAConfig(M, N / 4, N *sizeof(int)); 
          


    fwtBatch1Kernel(
        d_Data,
        d_Data,
        log2N
    );}
         
     
    printf("fwtBatch1Kernel() execution failed\n");
}



 
 
 
__global__ void modulateKernel(int *d_A, int *d_B, int N)
{
    int        tid = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * gridDim.x;
    float     rcpN = 1.0f / (float)N;

    for (int pos = tid; pos < N; pos += numThreads)
    {
        d_A[pos] *= d_B[pos] * rcpN;
    }
}

 
void modulateGPU(int *d_A, int *d_B, int N)
{
{ __set_CUDAConfig(128, 256); 
          
 modulateKernel(d_A, d_B, N);}
          
}


 
 
 
 
 
const int log2Kernel = 6;
const   int log2Data = 12;

const int   dataN = 1 << log2Data;
const int kernelN = 1 << log2Kernel;

const int   DATA_SIZE = dataN   * sizeof(float);
const int KERNEL_SIZE = kernelN * sizeof(float);

const long NOPS = 3.0 * (double)dataN * (double)log2Data / 2.0;
 



 
 
 
int main(int argc, char *argv[])
{
    float *h_Data,
          *h_Kernel,
          *h_ResultCPU,
          *h_ResultGPU;

    int   *d_Data,
          *d_Kernel;

    double delta, ref, sum_delta2, sum_ref2, L2norm, gpuTime;

     
    int i;

    printf("%s Starting...\n\n", argv[0]);

     
     

     

    printf("Initializing data...\n");
    printf("...allocating CPU memory\n");
    h_Kernel    = (float *)malloc(KERNEL_SIZE);
    h_Data      = (float *)malloc(DATA_SIZE);
    h_ResultCPU = (float *)malloc(DATA_SIZE);
    h_ResultGPU = (float *)malloc(DATA_SIZE);
    printf("...allocating GPU memory\n");
     
     
    cudaMalloc((void **)&d_Kernel, DATA_SIZE);
    cudaMalloc((void **)&d_Data,   DATA_SIZE);

    printf("...generating data\n");
    printf("Data length: %i; kernel length: %i\n", dataN, kernelN);
#ifdef _SYM
    klee_make_symbolic(h_Kernel, KERNEL_SIZE, "h_Kernel_input");
    klee_make_symbolic(h_Data, DATA_SIZE, "h_Data_input");
#else
    srand(2007);

    for (i = 0; i < kernelN; i++)
    {
        h_Kernel[i] = (float)rand() / (float)RAND_MAX;
    }

    for (i = 0; i < dataN; i++)
    {
        h_Data[i] = (float)rand() / (float)RAND_MAX;
    }
#endif

     
     
     
    cudaMemset(d_Kernel, 0, DATA_SIZE);
    cudaMemcpy(d_Kernel, h_Kernel, KERNEL_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Data,   h_Data,     DATA_SIZE, cudaMemcpyHostToDevice);

    printf("Running GPU dyadic convolution using Fast Walsh Transform...\n");
     
     
     
    fwtBatchGPU(d_Data, 1, log2Data);
    fwtBatchGPU(d_Kernel, 1, log2Data);
    modulateGPU(d_Data, d_Kernel, dataN);
    fwtBatchGPU(d_Data, 1, log2Data);
     
     
     
     
     

    printf("Reading back GPU results...\n");
     
    cudaMemcpy(h_ResultGPU, d_Data, DATA_SIZE, cudaMemcpyDeviceToHost);

    printf("Running straightforward CPU dyadic convolution...\n");
     

    printf("Comparing the results...\n");
    sum_delta2 = 0;
    sum_ref2   = 0;

     

    L2norm = sqrt(sum_delta2 / sum_ref2);

    printf("Shutting down...\n");
     
     
     
    cudaFree(d_Data);
    cudaFree(d_Kernel);
    free(h_ResultGPU);
    free(h_ResultCPU);
    free(h_Data);
    free(h_Kernel);

     
    printf("L2 norm: %E\n", L2norm);
    printf(L2norm < 1e-6 ? "Test passed\n" : "Test failed!\n");
}
