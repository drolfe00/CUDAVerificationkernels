// Dan Rolfe


#define BLOCKSIZE 32


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
__global__ void d_sobel1(int* __restrict__ result, unsigned int* __restrict__ pic, int xsize, int ysize, int thresh)
{

	int sum1, sum2, magnitude;

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
	//printf("xindex = %d, yindex = %d \n", i, j);
	if (i>0 && j>0 && i<479 &&j<383)
	{
		//if (j > 470)
		//	printf("xindex = %d, yindex = %d, result length = %d, pic length = %d \n", i, j, sizeof(result), sizeof(pic));
		
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
	//		printf("result value at i = %d, j = %d, is %d ... index is %d magnitude = %d, thresh = %d\n", i, j, result[index], index, magnitude, thresh);
	}
}


__global__ void d_sobel2(int* __restrict__ result, unsigned int* __restrict__ pic, int width, int height, int thresh)
{
	int sum1, sum2, magnitude;

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y; 
	int index = x * width +y ;
	//printf("x = %d, y = %d \n", x, y);
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
		
		
	//printf("after pics \n");
	__syncthreads();

	
	sum1 =  pic_s[threadX][threadY+2] -     pic_s[threadX][threadY] 
	+ 2 * pic_s[threadX+1][threadY+2] - 2 * pic_s[threadX+1][threadY]
	+     pic_s[threadX+2][threadY+2] -     pic_s[threadX+2][threadY];
	

	sum2 = pic_s[threadX][threadY] + 2 * pic_s[threadX][threadY+1]  + pic_s[threadX][threadY+2]
	- pic_s[threadX+2][threadY] - 2 * pic_s[threadX+2][threadY+1] - pic_s[threadX+2][threadY+2];
	
	magnitude =  sum1*sum1 + sum2*sum2;
	__syncthreads();


	//printf(" index = %d, sum1 = %d, sum2 = %d, magnitude = %d \n", index, sum1, sum2, magnitude);	
	
	
	if (magnitude > thresh)
		result[index] = 255;
	else 
		result[index] = 0;
	
	if (x ==0 || y ==0 || x==height-1 || y == width-1)
		result[index] = 0;	

}


