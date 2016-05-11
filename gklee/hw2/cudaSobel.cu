// Dan Rolfe





// general structure for version 1 requested by prof
// section 5.3 of the cuda pogramming guide
/**
* first load from device mem to shared mem
* sync threads after read
* do the processing from shared mem
* sync threads after processing
* write the results back to device mem
**/


// gpu version 1 of the sobel code
__global__ void d_sobel1(int *result, unsigned int *pic, int xsize, int ysize, int thresh)
{

	int sum1, sum2, magnitude;

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
	if (i>0 && j>0 && i<479 &&j<383)
	{
		//if (j > 470)
		
		int index = i * xsize + j; 
	 
		sum1 =  pic[ xsize * (i-1) + j+1 ] -     pic[ xsize*(i-1) + j-1 ] 
		+ 2 * pic[ xsize * (i)   + j+1 ] - 2 * pic[ xsize*(i)   + j-1 ]
		+     pic[ xsize * (i+1) + j+1 ] -     pic[ xsize*(i+1) + j-1 ];
	      

		sum2 = pic[ xsize * (i-1) + j-1 ] + 2 * pic[ xsize * (i-1) + j ]  + pic[ xsize * (i-1) + j+1 ]
		    - pic[xsize * (i+1) + j-1 ] - 2 * pic[ xsize * (i+1) + j ] - pic[ xsize * (i+1) + j+1 ];

		magnitude =  sum1*sum1 + sum2*sum2;

		if (magnitude > thresh)
			result[index] = 255;
		else 
			result[index] = 0;
	//	if (i >=383 && j >=479 )
	}
}

// gpu version 1 of the parallel sobel code
void sobel1(int *h_result, unsigned int *h_pic, int xsize, int ysize, int thresh)
{

	
	int *d_result;
	unsigned int *d_pic;
	
	// space for device result and pic
	int resultSize = xsize * ysize  * 3 * sizeof(int);
	int picSize = xsize * ysize * sizeof(int);

	// allocate device space
	cudaMalloc( (void**)&d_result, resultSize);
	if( !d_result) {
		exit(-1);
	}
	cudaMalloc( (void**)&d_pic, picSize);
	if( !d_pic) {
		exit(-1);
	}

	// copy from the host to device
	cudaMemcpy(d_result, h_result, resultSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pic, h_pic, picSize, cudaMemcpyHostToDevice);
	
	// launch the kernel
	
	dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 numBlocks(ceil((float)ysize/(float)threadsPerBlock.x), ceil((float)xsize/(float)threadsPerBlock.y));
	
	// setup event recording
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	d_sobel1 <<< numBlocks, threadsPerBlock >>> (d_result, d_pic, xsize, ysize, thresh);
	
	// finish recording time
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//d_sobel1 <<< ceil((float)(xsize+ysize-2)/(float)BLOCKSIZE), BLOCKSIZE >>> (d_result, d_pic, xsize);
	// copy the result back to the host
	cudaMemcpy(h_result, d_result, resultSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_pic, d_pic, picSize, cudaMemcpyDeviceToHost);

	// free device space
	cudaFree(d_result);
	cudaFree(d_pic);
	
}

__global__ void d_sobel2(int *result, unsigned int *pic, int width, int height, int thresh)
{
	int sum1, sum2, magnitude;

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y; 
	int index = x * width +y ;
	//if (x>0 && y>0 && x<height -1 &&y<width -1)
	 __shared__ unsigned int pic_s[BLOCKSIZE+2][BLOCKSIZE+2];

	

	int threadX = threadIdx.x;
	int threadY = threadIdx.y;

	
	pic_s[threadX+1][threadY+1] = pic[index]; 
	
	// top left corner
	if(threadX <1 && threadY <1)
		pic_s[threadX][threadY] = pic[(x-1)*width + y-1];
	// top right corner
	if(threadX <1 && threadY > BLOCKSIZE -2)
		pic_s[threadX][threadY+2] = pic[(x-1)*width + y+1];
	// bottom left corner
	if(threadX > BLOCKSIZE -2 && threadY <1)
		pic_s[threadX+2][threadY] = pic[(x+1)*width + y-1];
	// bottom right corner
	if(threadX > BLOCKSIZE -2 && threadY > BLOCKSIZE -2)
		pic_s[threadX+2][threadY+2] = pic[(x+1)*width + y+1];
	// top edge
	if (threadX < 1)
		pic_s[threadX][threadY+1] = pic[(x-1)*width + y];
	// bottom edge
	if (threadX > BLOCKSIZE -2)
		pic_s[threadX+2][threadY+1] = pic[(x+1)*width + y];
	// left edge
	if (threadY < 1)
		pic_s[threadX+1][threadY] = pic[(x)*width + y-1];
	// right edge
	if (threadY > BLOCKSIZE -2)
		pic_s[threadX+1][threadY+2] = pic[(x)*width + y+1];
		
		
	__syncthreads();

	
	sum1 =  pic_s[threadX][threadY+2] -     pic_s[threadX][threadY] 
	+ 2 * pic_s[threadX+1][threadY+2] - 2 * pic_s[threadX+1][threadY]
	+     pic_s[threadX+2][threadY+2] -     pic_s[threadX+2][threadY];
	

	sum2 = pic_s[threadX][threadY] + 2 * pic_s[threadX][threadY+1]  + pic_s[threadX][threadY+2]
	- pic_s[threadX+2][threadY] - 2 * pic_s[threadX+2][threadY+1] - pic_s[threadX+2][threadY+2];
	
	magnitude =  sum1*sum1 + sum2*sum2;
	__syncthreads();


	
	
	if (magnitude > thresh)
		result[index] = 255;
	else 
		result[index] = 0;
	
	if (x ==0 || y ==0 || x==height-1 || y == width-1)
		result[index] = 0;	

}

// gpu version 2 of the parallel sobel code
void sobel2(int *h_result, unsigned int *h_pic, int width, int height, int thresh)
{

	
	int *d_result;
	unsigned int *d_pic;
	
	// space for device result and pic
	int resultSize = width * height  * 3 * sizeof(int);
	int picSize = width * height * sizeof(int);

	// allocate device space
	cudaMalloc( (void**)&d_result, resultSize);
	if( !d_result) {
		exit(-1);
	}
	cudaMalloc( (void**)&d_pic, picSize);
	if( !d_pic) {
		exit(-1);
	}

	// copy from the host to device
	cudaMemcpy(d_result, h_result, resultSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pic, h_pic, picSize, cudaMemcpyHostToDevice);
	
	// launch the kernel
	
	dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 numBlocks(ceil((float)height/(float)threadsPerBlock.x), ceil((float)width/(float)threadsPerBlock.y));
	
	// setup event recording
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	d_sobel2 <<< numBlocks, threadsPerBlock >>> (d_result, d_pic, width, height, thresh);
	
	// finish recording time
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	// copy the result back to the host
	cudaMemcpy(h_result, d_result, resultSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_pic, d_pic, picSize, cudaMemcpyDeviceToHost);

	// free device space
	cudaFree(d_result);
	cudaFree(d_pic);

}
