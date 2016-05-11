#include <stdio.h>

__global__
void saxpy(int n, int a, int *x, int *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<5;
  int *x, *y, *d_x, *d_y;
  x = (int*)malloc(N*sizeof(int));
  y = (int*)malloc(N*sizeof(int));

  cudaMalloc(&d_x, N*sizeof(int)); 
  cudaMalloc(&d_y, N*sizeof(int));

  for (int i = 0; i < N; i++) {
    x[i] = 1;
    y[i] = 2;
  }

  cudaMemcpy(d_x, x, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(int), cudaMemcpyHostToDevice);
{ __set_CUDAConfig((N+255)/256, 256); 
          
 saxpy(N, 2.0f, d_x, d_y);}
          

  cudaMemcpy(y, d_y, N*sizeof(int), cudaMemcpyDeviceToHost);
 
}
