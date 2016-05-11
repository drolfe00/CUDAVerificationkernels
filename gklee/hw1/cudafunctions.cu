// Dan Rolfe 
#define BLOCKSIZE  32


/**
* cuda vector add function
**/


// there is a problem here, running this ruins the add
__global__ void d_add( float *x, float *y, float *z, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size)
		z[index] = x[index] + y[index];
}

void add( float *x, float *y, int length)
{

	float *d_x, *d_y, *d_z;  // device copies of x and y and a result z

	int size = length * sizeof(float);  // need space for total number of floats

	// allocate device space
	cudaMalloc( (void**)&d_x, size);	 
	cudaMalloc( (void**)&d_y, size);	 
	cudaMalloc( (void**)&d_z, size);

	// copy vector from host to device
	cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);	 
	cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

	// launch the kernel, eat some chicken
	d_add<<< ceil((float)length/(float)BLOCKSIZE), BLOCKSIZE >>>(d_x, d_y, d_z, size);

	// copy the result back to the host
	cudaMemcpy(x, d_z, size, cudaMemcpyDeviceToHost);

	// free device mem
	cudaFree(d_x);	 
	cudaFree(d_y);	 
	cudaFree(d_z);

	// hope for the best	 
}



/**
* mul:
* cuda vector multiply function
**/


__global__ void d_mul( float *x, float *y, float *z, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size)
		z[index] = x[index] * y[index];

}


void mul( float *x, float *y, int length)
{

	float *d_x, *d_y, *d_z;  // device copies of x and y and a result z

	int size = length * sizeof(float);  // need space for total number of floats

	// allocate device space
	cudaMalloc( (void**)&d_x, size);	 
	cudaMalloc( (void**)&d_y, size);	 
	cudaMalloc( (void**)&d_z, size);

	// copy vector from host to device
	cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);	 
	cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

	// launch the kernel, eat some chicken
	d_mul<<< ceil((float)length/(float)BLOCKSIZE), BLOCKSIZE >>>(d_x, d_y, d_z, size);

	// copy the result back to the host
	cudaMemcpy(x, d_z, size, cudaMemcpyDeviceToHost);

	// free device mem
	cudaFree(d_x);	 
	cudaFree(d_y);	 
	cudaFree(d_z);

}
