 





 
 
 


 
__global__ void d_sobel1(int *result, unsigned int *pic, int xsize, int ysize, int thresh)
{

	int sum1, sum2, magnitude;

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
	if (i>0 && j>0 && i<479 &&j<383)
	{
		 
		
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
	 
	}
}

 
void sobel1(int *h_result, unsigned int *h_pic, int xsize, int ysize, int thresh)
{

	
	int *d_result;
	unsigned int *d_pic;
	
	 
	int resultSize = xsize * ysize  * 3 * sizeof(int);
	int picSize = xsize * ysize * sizeof(int);

	 
	cudaMalloc( (void**)&d_result, resultSize);
	if( !d_result) {
		exit(-1);
	}
	cudaMalloc( (void**)&d_pic, picSize);
	if( !d_pic) {
		exit(-1);
	}

	 
	cudaMemcpy(d_result, h_result, resultSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pic, h_pic, picSize, cudaMemcpyHostToDevice);
	
	 
	
	dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 numBlocks(ceil((float)ysize/(float)threadsPerBlock.x), ceil((float)xsize/(float)threadsPerBlock.y));
	
	 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
{	__set_CUDAConfig(numBlocks, threadsPerBlock ); 
          
	d_sobel1 (d_result, d_pic, xsize, ysize, thresh);}
          
	
	 
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	 
	 
	cudaMemcpy(h_result, d_result, resultSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_pic, d_pic, picSize, cudaMemcpyDeviceToHost);

	 
	cudaFree(d_result);
	cudaFree(d_pic);
	
}

__global__ void d_sobel2(int *result, unsigned int *pic, int width, int height, int thresh)
{
	int sum1, sum2, magnitude;

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y; 
	int index = x * width +y ;
	 
	 __shared__ unsigned int pic_s[BLOCKSIZE+2][BLOCKSIZE+2];

	

	int threadX = threadIdx.x;
	int threadY = threadIdx.y;

	
	pic_s[threadX+1][threadY+1] = pic[index]; 
	
	 
	if(threadX <1 && threadY <1)
		pic_s[threadX][threadY] = pic[(x-1)*width + y-1];
	 
	if(threadX <1 && threadY > BLOCKSIZE -2)
		pic_s[threadX][threadY+2] = pic[(x-1)*width + y+1];
	 
	if(threadX > BLOCKSIZE -2 && threadY <1)
		pic_s[threadX+2][threadY] = pic[(x+1)*width + y-1];
	 
	if(threadX > BLOCKSIZE -2 && threadY > BLOCKSIZE -2)
		pic_s[threadX+2][threadY+2] = pic[(x+1)*width + y+1];
	 
	if (threadX < 1)
		pic_s[threadX][threadY+1] = pic[(x-1)*width + y];
	 
	if (threadX > BLOCKSIZE -2)
		pic_s[threadX+2][threadY+1] = pic[(x+1)*width + y];
	 
	if (threadY < 1)
		pic_s[threadX+1][threadY] = pic[(x)*width + y-1];
	 
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

 
void sobel2(int *h_result, unsigned int *h_pic, int width, int height, int thresh)
{

	
	int *d_result;
	unsigned int *d_pic;
	
	 
	int resultSize = width * height  * 3 * sizeof(int);
	int picSize = width * height * sizeof(int);

	 
	cudaMalloc( (void**)&d_result, resultSize);
	if( !d_result) {
		exit(-1);
	}
	cudaMalloc( (void**)&d_pic, picSize);
	if( !d_pic) {
		exit(-1);
	}

	 
	cudaMemcpy(d_result, h_result, resultSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pic, h_pic, picSize, cudaMemcpyHostToDevice);
	
	 
	
	dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 numBlocks(ceil((float)height/(float)threadsPerBlock.x), ceil((float)width/(float)threadsPerBlock.y));
	
	 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
{	__set_CUDAConfig(numBlocks, threadsPerBlock ); 
          
	d_sobel2 (d_result, d_pic, width, height, thresh);}
          
	
	 
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	 
	cudaMemcpy(h_result, d_result, resultSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_pic, d_pic, picSize, cudaMemcpyDeviceToHost);

	 
	cudaFree(d_result);
	cudaFree(d_pic);

}
