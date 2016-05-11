#define N 128
#define B 2

__global__ void k(int* in)
{
  in[threadIdx.x] = blockIdx.x;
}

int main()
{
  int* in = (int*) malloc(N * sizeof(int));
  int* din;
  cudaMalloc((void**) &din, N*sizeof(int));
{ __set_CUDAConfig(B, N/B); 
          
 k(din);}
          
}