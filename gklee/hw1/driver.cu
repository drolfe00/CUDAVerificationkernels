#include<stdio.h>
#include<stdlib.h>
#include<string.h>
//include the header file for your library here
//#include "cudafunctions.cu"

#define BLOCKSIZE  32

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






int func_add ( float *x, float *y, int sz)
{
	int i;
	float *a;
	a = ( float *)malloc(sizeof(float)*sz);
	if (!a){
		printf("memory allocation error\n");
		exit(-1);
	}
	memcpy(a,x,sz*(sizeof(float)));

	/* replace the code to add
         * with a cuda call which you will
	 * implement as a interface to your cuda enabled library
	 */
	/*
	for ( i=0; i<sz; i++)
		x[i]+=y[i];
	*/
	// replace with cuda enabled call
	add(x, y, sz);
		
	for ( i=0; i<sz; i++){
		if (x[i]!= a[i] + y[i]){
			printf("x = %f, a = %f, y = %f, i = %d, size = %d  ", x[i], a[i], y[i], i, sz);
			return 0;
			}
	}
		
	free(a);
	return 1;
}

	 	
int func_mul ( float *x, float *y, int sz)
{
	int i;
	float *a;
	a = ( float *)malloc(sizeof(float)*sz);
	if (!a){
		printf("memory allocation error\n");
		exit(-1);
	}
	memcpy(a,x,sz*(sizeof(float)));

	/* replace the code to multiply
         * with a cuda call which you will
	 * implement as a interface to your cuda enabled library
	 */
	/*
	for ( i=0; i<sz; i++)
		x[i]*=y[i];
	*/
	// cuda call
	mul(x, y, sz);	
	for ( i=0; i<sz; i++){
		if (x[i]!= a[i] * y[i]){
			printf("x = %f, a = %f, y = %f, i = %d, size = %d  ", x[i], a[i], y[i], i, sz);
			return 0;
			}
	}
	
	free(a);
	return 1;
}

int main()
{
	
	float *a,*b;
	int j;
	int i;

	for ( j=10; j<1000000; j*=10){
		a =( float *) malloc(sizeof(float)*j);
		b =( float *) malloc(sizeof(float)*j);

		for (i=0; i<j; i++){
			a[i] = 2;
			b[i] = 3;
		}

		if(!func_add(a,b,j)){
			printf("failed to add\n");
			}
		else{
			printf("add operation completed\n");
			}
		
		if(!func_mul(a,b,j)){
			printf("failed to mul\n");
			}
		else{
			printf("mul operation completed\n");
			}	
		
		free(a);
		free(b);
	}

		return 0;
}
