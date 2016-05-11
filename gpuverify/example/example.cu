#include <stdio.h>

__global__
void saxpy(int n, float a, float* __restrict__ x, float* __restrict__ y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}
