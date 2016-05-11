// Dan Rolfe 
#define BLOCKSIZE  32


/**
* cuda vector add function
**/


// there is a problem here, running this ruins the add
__global__ void d_add( float* __restrict__ x, float* __restrict__ y, float* __restrict__ z, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size)
		z[index] = x[index] + y[index];
}

/**
* mul:
* cuda vector multiply function
**/


__global__ void d_mul( float* __restrict__ x, float* __restrict__ y, float* __restrict__ z, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size)
		z[index] = x[index] * y[index];

}

