 /****************************************************************************
 *  Roy Wong
 *  Dan Rolfe
 *  Keri Anderson 
 *
 *  CS6235  CUDA Final Project
 *  Due April 2014
 *
 *
 *  This file runs the CUDA parallel version of fMRI CLEAN
 *
 *  Steps:  (called from fMRI_Main.c)
 *      
 *       1)  Create Point Vector Data (NFITI data transpose)
 *       2)  Clean Data
 * 
 *  
 ***************************************************************************/


//#include <stdio.h>
//#include <stdlib.h> 
#include "fMRI_Clean.h"

//for testing:  print out the calculated matrix
#define PRINT 0  //0 for "off"  1 for "on"
#define BETA_SEQ 0

//pre-declare function calls
void printMatrixClean(float* matrix, int iDim, int jDim);


/*
 *  Error Checking Macro - used to check errors in runtime API code
 *
 *  From stackoverflow.com:  The best way to check for errors in 
 *  runtime API code is to define an assert style handler function and wrapper macro.
 *  Each API call can be wrapped with the gpuErrorchceck macro, which will process 
 *  the return status of the API call it wraps.  If there is an error in a call, a 
 *  textual message describing the error and the file and line in your code where the 
 *  error occurred will be emitted to stderr and the application will exit. You could 
 *  conceivably modify gpuAssert to raise an exception rather than call exit() in a 
 *  more sophisticated application if it were required.
 *
 *  A second related question is how to check for errors in kernel launches, which 
 *  can't be directly wrapped in a macro call like standard runtime API calls. For 
 *  kernels, something like this:
 *
 *       kernel<<<1,1>>>(a);
 *       gpuErrorcheck( cudaPeekAtLastError() );
 *       gpuErrorcheck( cudaDeviceSynchronize() );
 *
 *  will firstly check for invalid launch argument, then force the host to wait 
 *  until the kernel stops and checks for an execution error. The synchronisation 
 *  can be eliminated if you have a subsequent blocking API call like this:
 *
 *       kernel<<<1,1>>>(a_d);
 *       gpuErrorcheck( cudaPeekAtLastError() );
 *       gpuErrorcheck( cudaMemcpy(a_h, a_d, size * sizeof(int), cudaMemcpyDeviceToHost) );
 *
 *  in which case the cudaMemcpy call can return either errors which occurred during 
 *  the kernel execution or those from the memory copy itself. This can be confusing for 
 *  the beginner, and I would recommend using explicit synchronisation after a kernel 
 *  launch during debugging to make it easier to understand where problems might be arising.
 */

//wrap each API call with the gpuErrorCheck macro
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
	fprintf(stderr, "GPUassert:  %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(code);
    }
}//end error checking macro


/*####################################################################################
 *#                                                                                  #
 *#                    CUDA FUNCTIONS                                                #
 *#                                                                                  #
 *####################################################################################*/

/************************************************************************
 *                                                                      *
 *            DEVICE FUNCTIONS                                          *
 *                                                                      *
 ************************************************************************/
__device__ void d_matrixMultiply(float *A, float *B, float *C, int hA, int wAhB, int wB) {
	int xIndex = threadIdx.x;  //# of threads == # of covariates
	
	int iRowStart = xIndex * wAhB;  //floor((float)xIndex / (float)wAhB) * wAhB;
	int j = 0;// xIndex + (floor((float)xIndex / (float)wB) * wB);
	
	int sum = 0;
	for(int i = iRowStart; i < iRowStart + wAhB; i++, j+=wB){
		sum += A[i] * B[j];
	} 

	C[xIndex] = sum;
	__syncthreads();

}//end d_matrixMultiply

/************************************************************************
 *                                                                      *
 *            KERNEL FUNCTIONS                                          *
 *                                                                      *
 ************************************************************************/
/***
 * Main Transpose function
 *
 */
__global__ void twist(float *original, float *transpose, int width, int height){
	
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int xStart = blockIdx.x * TILE_DIM;
    int yStart = blockIdx.y * TILE_DIM;

    int xIndex = xStart + threadIdx.x;
    int yIndex = yStart + threadIdx.y;
    int index_in = xIndex + (yIndex * width);

    xIndex = yStart + threadIdx.x;
    yIndex = xStart + threadIdx.y;
    int index_out = xIndex + (yIndex * height);
	
    tile[threadIdx.y][threadIdx.x] =  original[index_in];
    __syncthreads();

    transpose[index_out] = tile[threadIdx.x][threadIdx.y];

}//end twist

/***
 *  Clean data kernels
 *
 */
__global__ void buildBetas(float *covTranspose, float *pointData, float *matrixInverse,  int numCovariates, int numFiles, float *betas, float *xTransposeY) 
{
    int pointIndex = blockIdx.x * numFiles;  // index into the beginning of the vector for the current point
    int betaIndex = blockIdx.x * numCovariates;
	
	
    //Step 1:  Calculate xTranspose * Y      : will result in a 27 X 1 vector
    d_matrixMultiply(covTranspose, &pointData[pointIndex],  /*&betas[betaIndex],*/ &xTransposeY[betaIndex], numCovariates, numFiles, 1); 

    //tried moving matrixInverse to const mem & shared mem, no speedup
    //Step 2:  Calculate betas = in(Xinverse 27x27) * (xTranspose * Y)1X27    : results betas_out_ in a 27 X 1 vector
    d_matrixMultiply(matrixInverse, &xTransposeY[betaIndex], &betas[betaIndex], numCovariates, numCovariates, 1); 

}//end buildBetas

__global__ void cleanPoint(float *betas, float *covTranspose, int numCovariates, float *pointData, int numFiles, float *cleaned)
{
    int betaStart = blockIdx.x * numCovariates;
    int pointIndex = blockIdx.x * numFiles;
	
    int pointLocation = blockIdx.y * TILE_DIM + threadIdx.x;
	
    float sumOfBetas = 0.0;
    if(pointLocation < numFiles){	
	//Step 3:  Calculate U = Y - (b1X1 + b2X2 + b3X3) Results in a 1190 X 1 vector
	//first calculate b1x1 + b2X2 + b3X3
	for (int k = 0; k < numCovariates; k++){
	    sumOfBetas += betas[betaStart + k] * covTranspose[(k * numFiles) + pointLocation];
	}//end for
	
	//next calculate U = Y - sumOfBetas
	cleaned[pointIndex + pointLocation] = pointData[pointIndex + pointLocation] - sumOfBetas;
    }//end if

}//end cleanPoint



/*####################################################################################
 *#                                                                                  #
 *#                    CALLED FROM fMRI_Main.cu                                      #
 *#                                                                                  #
 *####################################################################################*/

/************************************************************************
 *                                                                      *
 *            TRANSPOSE NIFTI DATA  (CREATE POINT VECTOR DATA)          *
 *                                                                      *
 ************************************************************************/
/***
 *  This function will be called twice:  once to create point-vector data for 
 *      calculation, and once to re-transpose data to be written out to 
 *      new "cleaned" nifti data format. 
 *   
 *   iDim, jDim == iDim, jDim of the origianlMatrix
 *   version:  1 =>  transposing original NIFTI DATA to point-vector form
 *   version:  2 =>  trasnposing Cleaned point-vector data back to NIFTI format
 *
 *
 *
 *  First time called:  create point-vector data
 *
 *  niftiData:  organized by time shots:  niftiVolume X numFiles
 *  pointData:  organized by points in the brain over time:  numFiles X niftiVolume
 *
 *  incoming nifti matrix:  niftiVolume X numFiles matrix  ( (91*109*91)  X 1190 )
 *
 *                         -- 'j'  dim  --  
 *
 *                   Point0  Point1  Point2  .......  Point(niftiVol-1) 
 *   |    Time 0:     val     val     val              val
 *        Time 1:     val     val     val              val
 *  'i'   Time 2:     val     val     val              val
 *         .
 *  dim    .
 *         .
 *   |     Time 1189: val     val     val              val
 *
 *
 *
 *   Transposed matrix (point data):  27 X 1190 matrix
 *
 *                         -- 'j'  dim  --  
 *
 *                          Time 0  Time 1  Time 2 .......  Time 1189
 *   |    Point0:            val     val     val             val
 *        Point1:            val     val     val             val
 *  'i'   Point2:            val     val     val             val
 *         .
 *  dim    .
 *         .
 *   |    Point(niftiVol-1): val     val     val             val
 *
 */
extern "C" int transposeNiftiDataPar(float* originalMatrix, float* transposedMatrix, int iDim, int jDim, float* runTime)
{
    *runTime = 0.0;  // clear out the value
    float* subRunTime;

    //set the device to optimize amount of shared or global memory space
    // options are:
    //    cudaFuncCachePreferNone  :  default config
    //    cudaFuncCachePreferShared:  prefer larger shared memory and smaller L1
    //    cudaFuncCachePreferL1    :  prefer larger L1 cache
    //    cudaFuncCachePreferEqual :  equal L1 and shared
    gpuErrorCheck( cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) );


    int width = jDim;
    int height = iDim;

    dim3 blocks(BLOCK_SETS, BLOCK_SETS),
	threads(TILE_DIM, TILE_DIM);
		
    float *subset = (float *) malloc(BLOCK_SETS * TILE_DIM * BLOCK_SETS * TILE_DIM * sizeof(float));
    float *transpose1 = (float *) malloc(BLOCK_SETS * TILE_DIM * BLOCK_SETS * TILE_DIM * sizeof(float));
    float *transpose2 = (float *) malloc(BLOCK_SETS * TILE_DIM * BLOCK_SETS * TILE_DIM * sizeof(float));
    int subsetLength = BLOCK_SETS * TILE_DIM * BLOCK_SETS * TILE_DIM;
    int subsetSize = subsetLength * sizeof(float);
	
    int iSetCount = ceil((float)width / (BLOCK_SETS * (float)TILE_DIM));
    int jSetCount = ceil((float)height / (BLOCK_SETS * (float)TILE_DIM));
	
    for(int i = 0; i < iSetCount; i++){
	for(int j = 0; j < jSetCount; j+=2){
			
	    int subsetWidth = i * BLOCK_SETS * TILE_DIM + BLOCK_SETS * TILE_DIM < width ?
		BLOCK_SETS * TILE_DIM : width - i * BLOCK_SETS * TILE_DIM;
	    int subsetHeight = j * BLOCK_SETS * TILE_DIM + BLOCK_SETS * TILE_DIM < height ?
		BLOCK_SETS * TILE_DIM : height - j * BLOCK_SETS * TILE_DIM;
			
	    split(i, j, originalMatrix, width, height, subset, subsetWidth);
	    *runTime += sendTwist(blocks, threads, subset, subsetSize, transpose1);
			
			
	    if((j + 1) < jSetCount){
		split(i, j + 1, originalMatrix, width, height, subset, subsetWidth);
		*runTime += sendTwist(blocks, threads, subset, subsetSize, transpose2);
		unsplit(i, j + 1, transposedMatrix, height, width, transpose2, subsetHeight);
	    }
			
	    unsplit(i, j, transposedMatrix, height, width, transpose1, subsetHeight);
	}
    }

    free(subset);
    free(transpose1);
    free(transpose2);


    if (PRINT){  //for testing
	printf("          Original NIFTI Data:  \n");
	printMatrixClean(originalMatrix, iDim, jDim);
	printf("          Result Point Data Parallel:  \n");
	printMatrixClean(transposedMatrix, jDim, iDim);
    }//end if PRINT


    return 0; //success

}//end transposeNiftiDataPar

void split(int iteration, int blockSet, float *values, int width, int height, float *subset, int subsetWidth)
{
    int rowStart = blockSet * BLOCK_SETS * TILE_DIM;
    int rowEnd = rowStart + BLOCK_SETS * TILE_DIM < height ?  rowStart + BLOCK_SETS * TILE_DIM : height;

    int columnStart = iteration * BLOCK_SETS * TILE_DIM;
    int columnEnd =  columnStart + BLOCK_SETS * TILE_DIM < width ? columnStart + BLOCK_SETS * TILE_DIM : width;

    int ii = 0; //subset position
    for(int i = rowStart; i < rowEnd; i+=2){
	memcpy(&subset[ii], &values[i * width + columnStart], (columnEnd - columnStart) * sizeof(float));
	ii += BLOCK_SETS * TILE_DIM;

	if((i + 1) < rowEnd){
	    memcpy(&subset[ii], &values[(i + 1) * width + columnStart], (columnEnd - columnStart) * sizeof(float));
	    ii += BLOCK_SETS * TILE_DIM;
	}
    }
}//end split

void unsplit(int iteration, int blockSet, float *values, int width, int height, float *subset, int subsetWidth)
{
    int rowStart = iteration * BLOCK_SETS * TILE_DIM;
    int rowEnd = rowStart + BLOCK_SETS * TILE_DIM < height ?  rowStart + BLOCK_SETS * TILE_DIM : height;

    int columnStart = blockSet * BLOCK_SETS * TILE_DIM;
    int columnEnd = columnStart + BLOCK_SETS * TILE_DIM < width ?
	columnStart + BLOCK_SETS * TILE_DIM : width;
	
    int ii = 0;
    for(int i = rowStart; i < rowEnd; i+=2){
	memcpy(&values[i * width + columnStart], &subset[ii], (columnEnd - columnStart) * sizeof(float));
	ii +=  BLOCK_SETS * TILE_DIM;

	if((i + 1) < rowEnd){
	    memcpy(&values[(i + 1) * width + columnStart], &subset[ii], (columnEnd - columnStart) * sizeof(float));
	    ii +=  BLOCK_SETS * TILE_DIM;
	}
    }
}//end unsplit

float sendTwist(dim3 blocks, dim3 threads, float *subset, int subsetSize, float *transpose)
{
    float subRunTime = 0.0;
    float *d_subset, *d_pointData;
    gpuErrorCheck( cudaMalloc((void**)&d_subset, subsetSize) );
    gpuErrorCheck( cudaMalloc((void**)&d_pointData, subsetSize) );
    //cudaMalloc((void**)&d_subset, subsetSize);
    //cudaMalloc((void**)&d_pointData, subsetSize);

    gpuErrorCheck( cudaMemcpyAsync(d_subset, subset, subsetSize, cudaMemcpyHostToDevice) );

    cudaEvent_t start_event, stop_event;  
    gpuErrorCheck(cudaEventCreate(&start_event));
    gpuErrorCheck(cudaEventCreate(&stop_event));
    gpuErrorCheck(cudaEventRecord(start_event, 0));

    twist<<<blocks, threads>>>(d_subset, d_pointData, BLOCK_SETS * TILE_DIM, BLOCK_SETS * TILE_DIM);

    gpuErrorCheck(cudaEventRecord(stop_event, 0));
    gpuErrorCheck(cudaEventSynchronize(stop_event));

    gpuErrorCheck(cudaEventElapsedTime(&subRunTime, start_event, stop_event));
    subRunTime /= 1.0e3f;
   
			
    gpuErrorCheck( cudaMemcpyAsync(transpose, d_pointData, subsetSize, cudaMemcpyDeviceToHost) );
			
    gpuErrorCheck( cudaFree(d_subset) );
    gpuErrorCheck( cudaFree(d_pointData) );

    return subRunTime;

}//end sendTwist


/************************************************************************
 *                                                                      *
 *            CLEAN NIFTI DATA                                          *
 *                                                                      *
 ************************************************************************/
/***
 *
 *  This file runs the CUDA parallel version of fMRI CLEAN
 *
 *  This calculation is time intensive.  For ease in testing a smaller portion of data, 
 *     'pointBeg' and 'pointEnd' parameters have been added.
 *
 *  Example:  suppose we only want to test 5 of the (91*109*91) points in the brain.
 *            We could set pointBeg = '0', and pointEnd = '4'
 * 
 *
 *  Recall that 'pointData' will be in the form 'niftiVolume X numFiles'
 *
 *
 *
 *  DETAILED EXPLANATION OF THE ALGORITHM
 *  For simplicity, this example will use very small numbers and systematic float values.
 *
 *  fMRI takes 3D images of the brain, recording a float value at each point in the brain.
 *  This can be repeated many times over a time series, say 1 image of the brain per second.
 *
 *  Suppose that a simple fMRI image records 8 places in the brain (think of a 2*2*2 3D matrix).
 *
 *  Suppose for the first fMRI image (time = 0 ), we have the following float values: 
 *  (note that it is hard to represent 3D in this 2dimensional document, so this 
 *   is done by representing 2D matrices at z = 0 and z = 1)
 *  Time = 0:
 *  
 *     z = 0:                  z = 1:                (i.e. image[0][1][1] = 6.00, for example)
 *     _   y0       y1 _        _   y0       y1 _
 *  x0|   1.00 |   2.00 |    x0|   5.00 |   6.00 |
 *  x1|_  3.00 |   4.00_|    x1|_  7.00 |   8.00_|
 *
 *    We can also represent this image data as one long vector at time t = 0:
 *    image(0) = {1.00, 2.00, 3.00. 4.00, 5.00, 6.00, 7.00, 8.00}
 *
 *  Now, suppose that we are taking images at 5 different times:  time = 0, 1, 2, 3, 4 and we have:
 *    image(0) = {  1.00,   2.00,   3.00.   4.00,   5.00,   6.00,   7.00,   8.00}
 *    image(1) = {101.00, 102.00, 103.00. 104.00, 105.00, 106.00, 107.00, 108.00}
 *    image(2) = {201.00, 202.00, 203.00. 204.00, 205.00, 206.00, 207.00, 208.00}
 *    image(3) = {301.00, 302.00, 303.00. 304.00, 305.00, 306.00, 307.00, 308.00}
 *    image(4) = {401.00, 402.00, 403.00. 404.00, 405.00, 406.00, 407.00, 408.00}
 *
 * The challenge is, that this 'raw' fMRI data contains 'noise' created by the
 * patient's heartbeat, head motion, breathing, etc, and we need to clean
 * out that noise to get a more accurate fMRI scan.  
 *
 * To do this, a 'covariate file' is produced containing calculate estimates for
 * how much each element such as heartbeat, etc, has influenced the fMRI values  
 * at a given time stamp.  
 *
 *  Suppose we have a covariate file as follows: 
 *                   heartRate     head Motion    Respiration
 *  for time(0) = {   0.10,          0.20,          0.30   }
 *  for time(1) = {   0.20,          0.30,          0.40   }
 *  for time(2) = {   0.30,          0.40,          0.50   }
 *  for time(3) = {   0.40,          0.50,          0.10   }
 *  for time(4) = {   0.50,          0.10,          0.20   }
 * 
 *
 *  Now consider a position of the brain that is measured, say image[0][1][1]. 
 *  Recall that this position is imaged 5 times in our case.  We want to make
 *  a vector for position image[0][1][1] over time, so we have a vector of 
 *  all the values for image[0][1][1] at each of the time stamps, and we need
 *  to do this for each position:  
 *                                                              time0   time1   time2   time3   time4
 *  position [0][0][0]  (or [0] in the long vecotor) we have:    1.00  101.00  201.00  301.00  401.00
 *  position [0][1][0]  (or [1] in the long vecotor) we have:    2.00  102.00  202.00  302.00  402.00
 *  position [1][0][0]  (or [2] in the long vecotor) we have:    3.00  103.00  203.00  303.00  403.00
 *  position [1][1][0]  (or [3] in the long vecotor) we have:    4.00  104.00  204.00  304.00  404.00
 *  position [0][0][1]  (or [4] in the long vecotor) we have:    5.00  105.00  205.00  305.00  405.00
 *  position [0][1][1]  (or [5] in the long vecotor) we have:    6.00  106.00  206.00  306.00  406.00
 *  position [1][0][1]  (or [6] in the long vecotor) we have:    7.00  107.00  207.00  307.00  407.00
 *  position [1][1][1]  (or [7] in the long vecotor) we have:    8.00  108.00  208.00  308.00  408.00
 *
 *
 *  Once we have these vectors, we can use them with the covariate data to find an estimate for
 *  how much the extraneous elements such as heart rate, etc, influenced the value at each point
 *  in the brain over time.  We want to solve:
 *
 *         Y = b1*X1 + b2*X2 + b3*X3 + U   // where b1, b2, b3 are scalars
 *
 *  Where Yt is a vector over time for a given point (such as position [0][0][0] above:  {1.00 101.00... 401.00})
 *  X1 is the vector of covariates over time for heart rate:   {0.10, 0.20, 0.30, 0.40, 0.50}
 *  X2 is the vector of covariates over time for head motion:  {0.20, 0.30, 0.40, 0.50, 0.10}  
 *  X3 is the vector of covariates over time for respiration:  {0.30, 0.40, 0.50, 0.10, 0.20}
 *
 *           consider example for position [0][1][1] (5 in long vector)
 * 
 *                   Y         b1   x1        b2     X2        b3     X3     U
 *                  6.00  =  (b1* 0.10)  +  (b2 * 0.20)  +  (b3 * 0.30) + u0
 *                106.00  =  (b1* 0.20)  +  (b2 * 0.30)  +  (b3 * 0.40) + u1
 *                206.00  =  (b1* 0.30)  +  (b2 * 0.40)  +  (b3 * 0.50) + u2
 *                306.00  =  (b1* 0.40)  +  (b2 * 0.50)  +  (b3 * 0.10) + u3
 *                406.00  =  (b1* 0.50)  +  (b2 * 0.10)  +  (b3 * 0.20) + u4
 *                
 *
 *   We need a way to estimate the vectors B1, B2, and B3, so that we can figure out what the remaining
 *  'U' vector data is, and this new vector will be our 'cleaned' data. 
 *
 *  To solve/estimate the Beta b1, b2, b3 values, we will use the Ordinary Least Squares (OLS) method:
 *
 *              B = (X^T * X)^-1  *  X^T * Y
 *
 *       B     _          X^T                            X   _^-1   _           X^T                Y  _ 
 *      b1    |0.10 0.20 0.30 0.40 0.50    0.10   0.20   0.30 |    |0.10 0.20 0.30 0.40 0.50      6.00 |
 *      b2    |0.20 0.30 0.40 0.50 0.10    0.20   0.30   0.40 |    |0.20 0.30 0.40 0.50 0.10  * 106.00 |
 *      b3  = |0.30 0.40 0.50 0.10 0.20 *  0.30   0.40   0.50 |  * |0.30 0.40 0.50 0.10 0.20    206.00 |
 *            |                            0.40   0.50   0.10 |    |                            306.00 |
 *            |_                           0.50   0.10   0.20_|    |_                           406.00_|
 *
 *
 *      Calculating everything out:                                     
 *                 _                _                            _      (X^T * X)^-1        _
 *      X^T * X = | 0.55  0.45  0.40 |     and the inverse is:  |  5.6738  -3.8298   -0.9929 |
 *                | 0.45  0.55  0.45 |                          | -3.8298   8.0851   -3.8298 |
 *                |_0.40  0.45  0.55_|                          |_-0.9929  -3.8298    5.6738_|
 *
 *      NOTE:  We can use this same calulated inverse over ALL of the Y's (the vectors of a given positionin the brain over time)
 *             We only need to calculate it once, and then use it over and over again. 
 *
 *      **This is the point at which we would parallelize.  We will just show one calculation here for simplicity.**
 *
 *      Continuing with our specific Y vector, we have
 *                 _                                 _       _   _
 *                |0.10 0.20 0.30 0.40 0.50      6.00 |     | 409 |
 *      X^T * Y = |0.20 0.30 0.40 0.50 0.10  * 106.00 |  =  | 309 |
 *                |0.30 0.40 0.50 0.10 0.20    206.00 |     |_259_|
 *                |                            306.00 |
 *                |_                           406.00_|
 *
 *      Then B = 
 *       B     _   (X^T * X)^-1           _     _   _           _      _
 *      b1    |  5.6738  -3.8298   -0.9929 |   | 409 |         |  880.0 |
 *      b2  = | -3.8298   8.0851   -3.8298 | * | 309 |    =    | - 60.0 |
 *      b3    |_-0.9929  -3.8298    5.6738_|   |_259_|         |_-120.0_|
 *             
 *
 *
 *   Once we have B = {b1, b2, b3} = {880.0, -60.0, -120.0}, we can turn around and solve
 *
 *                   U = Y - b1X1 + b2X2 + b3X3
 *
 *     U         Y          b1      x1         b2     X2        b3     X3               _  U  _    This is our
 *     u0       6.00      (880.0 * 0.10)  +  (-60.0 * 0.20)  +  (-120.0 * 0.30)        | -34.0 |   cleaned data for a
 *     u1     106.00      (880.0 * 0.20)  +  (-60.0 * 0.30)  +  (-120.0 * 0.40)        | - 4.0 |   specific point in the
 *     u2  =  206.00   -  (880.0 * 0.30)  +  (-60.0 * 0.40)  +  (-120.0 * 0.50)   =    |  26.0 |   brain across time.
 *     u3     306.00      (880.0 * 0.40)  +  (-60.0 * 0.50)  +  (-120.0 * 0.10)        | - 4.0 |
 *     u4     406.00      (880.0 * 0.50)  +  (-60.0 * 0.10)  +  (-120.0 * 0.20)        |_- 4.0_|
 * 
 *  We need to put this data back into nifti data format, so that we have a new cleaned
 *  image of the entire brain for each time shot.
 *  
 *  Then, putting this new U calculated vector for position [0][1][1]  (or [5] in the long vecotor) we have: 
 *
 *                                                    Y5
 *    cleanedImage(0) = {  **,  **,  **,  **,  **,  -34.00,  **,  ** }
 *    cleanedImage(1) = {  **,  **,  **,  **,  **,  - 4.00,  **,  ** }   The '**' represent data that would be 
 *    cleanedImage(2) = {  **,  **,  **,  **,  **,   26.00,  **,  ** }   cacluated from the other Y vectors.
 *    cleanedImage(3) = {  **,  **,  **,  **,  **,  - 4.00,  **,  ** }
 *    cleanedImage(4) = {  **,  **,  **,  **,  **,  - 4.00,  **,  ** }
 *   
 *    
 *
 *
 * 
 */
extern "C" int cleanPar(float* pointData, float* cleanedData, float* covTranspose, float* matrixInverse, int numCovariates, int numFiles, int niftiVolume, float* runTime)
{

    *runTime = 0.0; //clear out the variable
    
    //set the device to optimize amount of shared or global memory space
    // options are:
    //    cudaFuncCachePreferNone:  default config
    //    cudaFuncCachePreferShared:  prefer larger shared memory and smaller L1
    //    cudaFuncCachePreferL1  :    prefer larger L1 cache
    //    cudaFuncCachePreferEqual:  equal L1 and shared
    gpuErrorCheck( cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) );

    int iStepSet = ceil((float)niftiVolume / (float)TILE_DIM);
    int hCleanBlockCount = ceil((float)numFiles / (float)TILE_DIM);
		
    //printf("iStepSet: %d - nv: %d - nf: %d - nc: %d\n", iStepSet, niftiVolume, numFiles, numCovariates);
    float *betas = (float *)calloc(TILE_DIM * numCovariates, sizeof(float));
  
    for (int i = 0; i < iStepSet; i++){
	//figure out how many points to send off
	int rowStart = i * TILE_DIM;
	int rowCount = rowStart + TILE_DIM < niftiVolume ?  TILE_DIM : niftiVolume - rowStart;
	dim3 cleanBlocks(rowCount, hCleanBlockCount);
		
	float *d_covTranspose, *d_pointData, *d_matrixInverse, *d_betas, *d_xTranposeY;
		
	gpuErrorCheck( cudaMalloc((void**)&d_covTranspose, numFiles * numCovariates * sizeof(float)) );
	gpuErrorCheck( cudaMalloc((void**)&d_pointData, rowCount * numFiles * sizeof(float)) );
	gpuErrorCheck( cudaMalloc((void**)&d_matrixInverse, numCovariates * numCovariates * sizeof(float)) );
	gpuErrorCheck( cudaMalloc((void**)&d_betas, rowCount * numCovariates * sizeof(float)) );
	gpuErrorCheck( cudaMalloc((void**)&d_xTranposeY, rowCount * numCovariates * sizeof(float)) );
				
	gpuErrorCheck( cudaMemcpy(d_covTranspose, covTranspose, numFiles * numCovariates * sizeof(float), cudaMemcpyHostToDevice) );
 	gpuErrorCheck( cudaMemcpy(d_pointData, &pointData[rowStart * numFiles], rowCount *numFiles * sizeof(float), cudaMemcpyHostToDevice) );
	gpuErrorCheck( cudaMemcpy(d_matrixInverse, matrixInverse, numCovariates * numCovariates * sizeof(float), cudaMemcpyHostToDevice) );
		

	cudaEvent_t start_event, stop_event;  
	float subTime = 0.0;
	gpuErrorCheck(cudaEventCreate(&start_event));
	gpuErrorCheck(cudaEventCreate(&stop_event));
	gpuErrorCheck(cudaEventRecord(start_event, 0));

	buildBetas<<<rowCount, numCovariates>>>(d_covTranspose, d_pointData, d_matrixInverse, numCovariates, numFiles, d_betas, d_xTranposeY);

	gpuErrorCheck(cudaEventRecord(stop_event, 0));
	gpuErrorCheck(cudaEventSynchronize(stop_event));

	gpuErrorCheck(cudaEventElapsedTime(&subTime, start_event, stop_event));
	subTime /= 1.0e3f;
	*runTime += subTime;


	if(BETA_SEQ){
	    gpuErrorCheck( cudaDeviceSynchronize() );
	    gpuErrorCheck( cudaMemcpy(betas, d_betas, rowCount * numCovariates * sizeof(float), cudaMemcpyDeviceToHost) );
	}
				
	gpuErrorCheck( cudaFree(d_matrixInverse) );
	gpuErrorCheck( cudaFree(d_xTranposeY) );

	if(!BETA_SEQ){
	    float *d_cleanedData;
	    gpuErrorCheck( cudaMalloc((void**)&d_cleanedData, rowCount * numFiles * sizeof(float)) );

	    subTime = 0.0;
	    gpuErrorCheck(cudaEventCreate(&start_event));
	    gpuErrorCheck(cudaEventCreate(&stop_event));
	    gpuErrorCheck(cudaEventRecord(start_event, 0));

	    cleanPoint<<<cleanBlocks, TILE_DIM>>>(d_betas, d_covTranspose, numCovariates, d_pointData, numFiles,  d_cleanedData);

	    gpuErrorCheck(cudaEventRecord(stop_event, 0));
	    gpuErrorCheck(cudaEventSynchronize(stop_event));
	    gpuErrorCheck(cudaEventElapsedTime(&subTime, start_event, stop_event));
	    subTime /= 1.0e3f;
	    *runTime += subTime;

	    gpuErrorCheck( cudaMemcpy(&cleanedData[rowStart * numFiles], d_cleanedData, rowCount * numFiles * sizeof(float), cudaMemcpyDeviceToHost) );
	    gpuErrorCheck( cudaFree(d_cleanedData) );
	}else{
	    //temp loop while I test out matrix multiply
	    for(int ii = 0; ii < rowCount; ii++){
		int betaStart = ii * numCovariates;
		int pointIndex = (rowStart * numFiles)+ (ii * numFiles);
	
		//Step 3:  Calculate U = Y - (b1X1 + b2X2 + b3X3) Results in a 1190 X 1 vector
		float sumOfBetas = 0;
		for (int j= 0; j < numFiles; j++){
		    //first calculate b1x1 + b2X2 + b3X3
		    sumOfBetas = 0.0;
		    for (int k = 0; k < numCovariates; k++){
			sumOfBetas += betas[betaStart + k] * covTranspose[k*numFiles + j];
		    }//end for k
			
		    //next calculate U = Y - sumOfBetas
		    cleanedData[pointIndex + j] = pointData[pointIndex + j] - sumOfBetas;
		}//end for j
	    }
	}

	gpuErrorCheck( cudaFree(d_pointData) );
	gpuErrorCheck( cudaFree(d_covTranspose) );
	gpuErrorCheck( cudaFree(d_betas) );

    }//end for i
	

    if (PRINT){//for testing
	printf("          Result Cleaned Data Parallel:  \n");
	printMatrixClean(cleanedData, niftiVolume, numFiles);

    }//end if TEST

    //free
    free(betas);

    return 0; //success

}//end cleanPar


/************************************************************************
 *                                                                      *
 *            HELPER FUNCTIONS:  FOR TESTING                            *
 *                                                                      *
 ************************************************************************/
void printMatrixClean(float* matrix, int iDim, int jDim)
{
    int i, j;
    for (i = 0; i < iDim; i++){
	printf("\n          ");
	for (j = 0; j < jDim; j++){
	    printf("     %.3f,   ", matrix[i*jDim + j]);   
	    
	}//end for j
    }//end for i

    printf("\n\n");
	    
}//end printMatrix




