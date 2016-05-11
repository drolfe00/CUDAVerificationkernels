 /****************************************************************************
 *  Roy Wong
 *  Dan Rolfe
 *  Keri Anderson 
 *
 *  CS6235  CUDA Final Project
 *  Due April 2014
 *
 *
 *  This file runs the CUDA parallel version of fMRI Connectivity.
 *  This code calculates how connected a point (the "seed") in the brain
 *  is to the other points in the brain.
 *
 *  Steps:  (called from fMRI_Main.c)
 *    
 *       1)  Normalize the data
 *       2)  Calculate Connectivity
 * 
 *  
 ***************************************************************************/


/*******************************************
 *  TODO:
 *      *)  write Normalize Data Kernel
 *      *)  write Connectivity Kernel
 *      *)  use "preferred shared" / L1
 *      *)  
 *      *)    
 *
 *
 *
 *
 *******************************************/

#include <stdio.h>
#include <stdlib.h>
#include "fMRI_Connectivity.h"

//for testing:  print out the calculated matrix
#define PRINT 0  //0 for "off"  1 for "on"

//Recall we have 91 * 109 * 91 = 902,629 points in the brain
//Global memory is 1,073,020,928  bytes
//For a single point vector, we have 1190 * 4 bytes = 4760 bytes
//Then 1,073,020,928 / 4760 bytes = 225424.56, meaning we can send no more than
//   225,424 points over at a time
//902,629 points / 225,424 points = 4.004... meaning we would have to repeat the
// calls to the kernel about 4 times

// Final distribution:
//
// We will handle 180,526 points in a kernel call:  180,526 * 1190 * 4 bytes = 859,303,760, and is within 1,073,020,928 byte range
// We will have to loop through the kernel call 5 times:  5 * 180,526 = 902,630 points (we only have 902,629 points - 1 extra)
// We will have 180,526 blocks  - this is within accepted range
// Each block will have 64 threads (64*19 = 1216, so essentially each thread will handle 19 floats)


//#define NUMBLOCKS 180526
#define NUMBLOCKS 150439
#define NUMTHREADS 64  //64 * 19 = 1216,  1216-1190 = 26, so 26 "extraneous" threads per block


//pre-declare function calls
void printMatrixConnectivity(float* matrix, int iDim, int jDim);


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
/************************************************************************
 *                                                                      *
 *            KERNEL FUNCTIONS                                          *
 *                                                                      *
 ************************************************************************/
/***
 * 64 threads in a block;  each thread is responsible for 19 elements
 *
 *  64 * 19 = 1216 - 1190 = 26 threads left over
 *
 *
 *        Consider example for position [0][1][1] (5 in long vector) with newly "cleaned" data
 *                time0   time1   time2   time3   time4
 *         Y5 = [ -34.0,   -4.0,   26.0,   -4.0,   -4.0 ]
 *
 *           step 1:  add up all of the values, and divide by the number to create the mean:
 *
 *                      mean = (-34.0 + -4.0 + 26.0 + -4.0 + -4.0)/5.0 = -20/5.0 = -4.0
 *
 *           step 2:  subtract the mean from each of the time points:
 *
 *                  Y5 = [-34.0 - (-4), -4.0 - (-4), 26 - (-4), -4.0 - (-4), -4.0 - (-4)]
 *                     = [-30.0, 0, 30, 0, 0]  //notice the mean is now '0'
 *
 *           step 3:  calculate the standard diviation - steps 1 and 2 were already parts of the process, so start with
 *                    squaring each number, calculate the mean (using n-1) of these new values and take the square root
 *
 *                   (-30 * -30) + (0*0) + (30.0 * 30.0) + (0*0) + (0*0) = 1800
 *                   1800 / 4.0 = 450  //there are 5 elements, so divide by 5-1 = 4 to get new mean
 *                   sqrt(450) = 21.21 = standard deviation
 *
 *          step 4:  take data from step 2 and divide by standard deviation
 *
 *                   [-30.0/21.21,  0/21.21, 30/21.21, 0/21.21, 0/21.21] 
 *                 = [-1.41,  0,  1.41,  0,  0 ]
 *
 *
 *              --repeat this for every point in the brain
 *
 */
void __global__ normalizeKernel(float *d_cleanedData, int numFiles)
{
    //get *this* blocks starting point in cleaned data
    int blockStart = blockIdx.x * numFiles;

    //holds *this* thread's partial mean sum
    double myMeanSum = 0.0;

    //holds *this* thread's partial StdDev sum
    double myStdDevSum = 0.0;

    int mySharedElm;
    int myGlobalElm;

    __shared__ float pointVector[NUMTHREADS * 19];
    __shared__ double workspace[NUMTHREADS];
    __shared__ double mean;
    __shared__ double stdDev;

    if (threadIdx.x == 0){  // clear out the values
	mean = 0.0;
	stdDev = 0.0;
    }

    int i;

    for (i = 0; i < 18; i++){
	mySharedElm = threadIdx.x + i*NUMTHREADS;
	myGlobalElm = blockStart + mySharedElm;

	float temp = d_cleanedData[myGlobalElm];
	pointVector[mySharedElm] = temp;
    }
    //get the last values
    mySharedElm = threadIdx.x + 18 *NUMTHREADS;
    myGlobalElm = blockStart + mySharedElm;
    if (mySharedElm < numFiles)
	pointVector[mySharedElm] = d_cleanedData[myGlobalElm];
    else
	pointVector[mySharedElm] = 0.0; //just load with 0

    __syncthreads();  //make sure all the copying has been done



    //Step 1:  calculate the mean - need a reduction here
    for (i = 0; i < 19; i++){
	mySharedElm = threadIdx.x + i *NUMTHREADS;
	myMeanSum += (double)pointVector[mySharedElm];
    }

    //add sum to shared data structure
    workspace[threadIdx.x] = myMeanSum;
     __syncthreads();

     // now perform a reduction
     // reduce to the first 32 spaces
     if (threadIdx.x < 32)
	 workspace[threadIdx.x] += workspace[threadIdx.x + 32];
     __syncthreads();
     //reduce to the first 16 spaces
     if (threadIdx.x < 16)
	 workspace[threadIdx.x] += workspace[threadIdx.x + 16];
     __syncthreads();
     //reduce to the first 8 spaces
     if (threadIdx.x < 8)
	 workspace[threadIdx.x] += workspace[threadIdx.x + 8];
     __syncthreads();
     //reduce to the frist 4 spaces
     if (threadIdx.x < 4)
	 workspace[threadIdx.x] += workspace[threadIdx.x + 4];
     __syncthreads();
     //reduce to the first 2 spaces
     if (threadIdx.x < 2)
	 workspace[threadIdx.x] += workspace[threadIdx.x + 2];
     if (threadIdx.x == 0)
	 mean = (workspace[0] + workspace[1]) / (numFiles * 1.0);
    __syncthreads();

    float myMean = (float)mean;
    //Step 2:  subtract mean from each element
    for (i = 0; i < 18; i++){
	mySharedElm = threadIdx.x + i *NUMTHREADS;
	pointVector[mySharedElm] = pointVector[mySharedElm] - myMean;
    }

    mySharedElm = threadIdx.x + 18 * NUMTHREADS;
    if (mySharedElm < numFiles)
	pointVector[mySharedElm] = pointVector[mySharedElm] - myMean;

    __syncthreads();

    //Step 3:  calculate the standard deviation
    for (i = 0; i < 19; i++){
	mySharedElm = threadIdx.x + i*NUMTHREADS;
	myStdDevSum += (double)(pointVector[mySharedElm] * pointVector[mySharedElm]);
    }
    workspace[threadIdx.x] = myStdDevSum;
    __syncthreads();

     // now perform a reduction
     // reduce to the first 32 spaces
     if (threadIdx.x < 32)
	 workspace[threadIdx.x] += workspace[threadIdx.x + 32];
     __syncthreads();
     //reduce to the first 16 spaces
     if (threadIdx.x < 16)
	 workspace[threadIdx.x] += workspace[threadIdx.x + 16];
     __syncthreads();
     //reduce to the first 8 spaces
     if (threadIdx.x < 8)
	 workspace[threadIdx.x] += workspace[threadIdx.x + 8];
     __syncthreads();
     //reduce to the frist 4 spaces
     if (threadIdx.x < 4)
	 workspace[threadIdx.x] += workspace[threadIdx.x + 4];
     __syncthreads();
     //reduce to the first 2 spaces
     if (threadIdx.x < 2)
	 workspace[threadIdx.x] += workspace[threadIdx.x + 2];
     if (threadIdx.x == 0)
	 stdDev = (workspace[0] + workspace[1]) / ((numFiles-1) * 1.0);
    __syncthreads();

    //step 4:  divide each element from step 2 by the standard deviation and write to global
    float myStdDev = (float)stdDev;
    for (i = 0; i < 18; i++){
	mySharedElm = threadIdx.x + i*NUMTHREADS;
	myGlobalElm = blockStart + mySharedElm;
	d_cleanedData[myGlobalElm] = pointVector[mySharedElm] / myStdDev;
    }
    mySharedElm = threadIdx.x + 18 * NUMTHREADS;
    myGlobalElm = blockStart + mySharedElm;
    if (mySharedElm < numFiles){
	d_cleanedData[myGlobalElm] = pointVector[mySharedElm] / myStdDev;
    }
    
}//end normalize





void __global__ d_dotProduct( float *d_normalizedData, float *d_connectivityData, float *d_seedData, int *d_numFiles, int *d_niftiVolume,  int *d_iteration)
{

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int numFiles = *d_numFiles;
    int niftiVolume = *d_niftiVolume;
    int iteration = *d_iteration ; 
    
    __syncthreads();
    if (index <  niftiVolume)
    {
       for ( int i = 0; i < numFiles; i++)
       {

        float seedData =  d_seedData[i]; //  d_normalizedData[index % numFiles] ;
        float normalData = d_normalizedData[index * numFiles + i];

        float newData = seedData * normalData;
        newData = float(newData)/float(numFiles -1);
        
        __syncthreads();
        d_connectivityData[(index + ( iteration * niftiVolume))] +=  newData;
        

       }
    }

    
 //d_connectivityData[index % numFiles] = d_seedData[index %numFiles] + d_normalizedData[index];
}//end dotProduct


/*####################################################################################
 *#                                                                                  #
 *#                    CALLED FROM fMRI_Main.cu                                      #
 *#                                                                                  #
 *####################################################################################*/

/************************************************************************
 *                                                                      *
 *            NORMALIZE DATA                                            *
 *                                                                      *
 ************************************************************************/
 /* 
 *   This step is computationally and time-wise intensive.  In order to 
 *   test smaller sets of data, the parameters "pointBeg" and "pointEnd"
 *   have been added.  Instead of caclulating for all niftiVolume = 91*109*91
 *   points, we can just calculate pointBeg == 0, and pointEnd == 5, so
 *   6 points total.  
 *
 *
 *   Alogrithm Description:
 *
 *        We need to "normalize the data" so that the mean is '0' and standard deviation 
 *           is '1' for each pt in the brain across all time points.  
 *
 *        Consider example for position [0][1][1] (5 in long vector) with newly "cleaned" data
 *                time0   time1   time2   time3   time4
 *         Y5 = [ -34.0,   -4.0,   26.0,   -4.0,   -4.0 ]
 *
 *           step 1:  add up all of the values, and divide by the number to create the mean:
 *
 *                      mean = (-34.0 + -4.0 + 26.0 + -4.0 + -4.0)/5.0 = -20/5.0 = -4.0
 *
 *           step 2:  subtract the mean from each of the time points:
 *
 *                  Y5 = [-34.0 - (-4), -4.0 - (-4), 26 - (-4), -4.0 - (-4), -4.0 - (-4)]
 *                     = [-30.0, 0, 30, 0, 0]  //notice the mean is now '0'
 *
 *           step 3:  calculate the standard diviation - steps 1 and 2 were already parts of the process, so start with
 *                    squaring each number, calculate the mean (using n-1) of these new values and take the square root
 *
 *                   (-30 * -30) + (0*0) + (30.0 * 30.0) + (0*0) + (0*0) = 1800
 *                   1800 / 4.0 = 450  //there are 5 elements, so divide by 5-1 = 4 to get new mean
 *                   sqrt(450) = 21.21 = standard deviation
 *
 *          step 4:  take data from step 2 and divide by standard deviation
 *
 *                   [-30.0/21.21,  0/21.21, 30/21.21, 0/21.21, 0/21.21] 
 *                 = [-1.41,  0,  1.41,  0,  0 ]
 *
 *
 *              --repeat this for every point in the brain
 *
 */
extern "C" int normalizeDataPar(float* cleanedData, float* normalizedData, int numFiles, int niftiVolume, float* runTime)
{ 
       
    *runTime = 0.0;  //clear out the value
    float subTime = 0.0;
    int i;

    //set the device to optimize amount of shared or global memory space
    // options are:
    //    cudaFuncCachePreferNone:  default config
    //    cudaFuncCachePreferShared:  prefer larger shared memory and smaller L1
    //    cudaFuncCachePreferL1  :    prefer larger L1 cache
    //    cudaFuncCachePreferEqual:  equal L1 and shared
    gpuErrorCheck( cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) );
    
    int gpuSize = NUMBLOCKS * numFiles;
    int numTimesToCallKernel = (int)(ceil(niftiVolume/ (NUMBLOCKS*1.0) ));
    //int numTimesToCallKernel = 6;
    int unNeededEdgeBlocks = (numTimesToCallKernel * NUMBLOCKS) - niftiVolume;
    //printf("numTimesToCallKernel is %d\n", (int)( ceil(niftiVolume/(NUMBLOCKS*1.0)) )); //for testing
    //printf("numunNeededEdgeBloks is %d\n", unNeededEdgeBlocks);

    //memory on the GPU
    float* d_cleanedData;
    gpuErrorCheck( cudaMalloc((void **) &d_cleanedData,  gpuSize*sizeof(float)) );
    

    //call the kernel 6 times, processing chunks of data at a time
    for (i = 0; i < numTimesToCallKernel; i++){
	int numToCopy;
	int startLocation = i*(NUMBLOCKS*numFiles);
	int numBlocks;

	//copy the data to the GPU
	if (i < numTimesToCallKernel - 1){
	    numToCopy = NUMBLOCKS * numFiles;
	    numBlocks = NUMBLOCKS;
	}
	else{ //last time through, edge case
	    numToCopy = (NUMBLOCKS - unNeededEdgeBlocks) * numFiles;
	    numBlocks = NUMBLOCKS - unNeededEdgeBlocks;
	}

	gpuErrorCheck( cudaMemcpy(d_cleanedData, cleanedData+startLocation, numToCopy*sizeof(float), cudaMemcpyHostToDevice) );

	//set up the grid
	dim3 dimGrid(numBlocks, 1, 1);  // '1' means dimension is not used
	dim3 dimBlock(NUMTHREADS, 1, 1);

	cudaEvent_t start_event, stop_event;  
	subTime = 0.0;
	gpuErrorCheck(cudaEventCreate(&start_event));
	gpuErrorCheck(cudaEventCreate(&stop_event));
	gpuErrorCheck(cudaEventRecord(start_event, 0));

	normalizeKernel<<<dimGrid, dimBlock>>>(d_cleanedData, numFiles);

	gpuErrorCheck(cudaEventRecord(stop_event, 0));
	gpuErrorCheck(cudaEventSynchronize(stop_event));

	gpuErrorCheck(cudaEventElapsedTime(&subTime, start_event, stop_event));
	subTime /= 1.0e3f;
	*runTime += subTime;
   

	//copy data back to host
	gpuErrorCheck( cudaMemcpy(normalizedData+startLocation,  d_cleanedData, numToCopy*sizeof(float), cudaMemcpyDeviceToHost) );
	
	
    }//end for i

    //GPU free
    gpuErrorCheck(cudaFree(d_cleanedData));

    if (PRINT){  //for testing
	printf("          Result Normalized Data Parallel:  \n");
	printMatrixConnectivity(normalizedData, niftiVolume, numFiles);
    }//end if PRINT

    return 0; //success


}//end normalizeDataPar


/************************************************************************
 *                                                                      *
 *            CALCULATE CONNECTIVITY                                    *
 *                                                                      *
 ************************************************************************/
/*
 *   This method cacluates connectivity for one point (seed in the brain).
 *   Result is a 1 x niftivolume vector.
 *
 *   Because many of the earlier required computations are time-intesive, 
 *   this function adds parameter "pointBeg" and "pointEnd" for testing.  
 *
 *   If all points are computed, the resulting data structure will contain
 *         niftiData x niftiData computations:  (91*109*91) x (91*109*91)
 *
 *   A Pearson Correlation Coefficient 'r' must be calculated for EACH PAIR of pixels, 
 *            potentially hundreds of millions of operations
 *
 *                    
 *                r =    sum i = 1-> n :  (Xi - Xbar)*(Yi - Ybar)
 *                       _______________________________________
 *
 *                       sqrt(sum (Xi-Xbar)^2) * sqrt(sum (Yi-Ybar)^2)
 *
 *     Example:
 *
 *        For simplicity, suppose that xDim = yDim = zDim = 2, so 8 data points in the brain
 *        and suppose that we have 5 time frames taken.  
 *
 *        As input, we have normalized dataPoint (Y) vectors:
 *
 *        (dummy data - may not actually be normallized but pretend it is)
 *        normalizedData = 
 *              Time0   Time1   Time2  Time3  Time4
 *        Y0 = [01.00   02.00   -1.00  00.00  -2.00]
 *        Y1 = [02.00   03.00   -2.00  -3.00  00.00]
 *        Y2 = [03.00   07.00   -7.00  -2.00  -1.00]
 *        Y3 = [04.00   -4.00   01.00  -1.00  00.00]
 *        Y4 = [05.00   -2.00   01.00  01.00  -5.00]
 *        Y5 = [06.00   -2.00   -4.00  00.00  00.00]
 *        Y6 = [07.00   00.00   00.00  -7.00  00.00]
 *        Y7 = [08.00   -3.00   -5.00  04.00  -4.00]
 *
 *
 *      Let Y2 be the seed.  We need to calculate a correlation coeficient for (Y2, Y0), (Y2, Y1), (Y2, Y2), (Y2, Y3), (Y2, Y4), (Y2, Y5), (Y2, Y6), (Y2, Y7)
 *
 *      To calculate the correlation coeficient between two Yi vectors, take the dot product and divde the result by number of data points - 1.
 *
 *      Example:  calculate the correlation coeficient for pair (Y2, Y5)
 *
 *      1st step:  Dot product:  = (03.00* 06.00) + (07.00 * -2.00) + (-7.00 * -4.00) + (-2.00 * 0.00) + (-1.00 * 0.00)  = 4.00  
 *      2nd step:  Divde by number of time data points - 1,  in our case number of time data points = 5, so divide by 4 = 1.00
 *
 *                   *note:  the result should be between 1 and - 1
 *                           when the correlation coefficient is calculated with itself, the result will be 1.  
 *   
 *                                    
 *
 *        
 */
extern "C" int connectivityPar(int seed, float* normalizedData, float* connectivityData, int numFiles, int niftiVolume, float* runTime)
{
    *runTime = 0.0; //clear out
    float subTime = 0.0;

    //'seed' is the chosen point in the brain for which we caculate connectivity
    
    int threadsPerBlock = 32;

    //printf("hi %d \n ", 0);
    float * d_normalizedData, *d_connectivityData, *d_seedData;
    int *d_numFiles, *d_niftiVolume, *d_seed, *d_iteration;

    int niftiChunkLength = 91 * 91 * 10;
    if ( niftiChunkLength > niftiVolume)
        niftiChunkLength = niftiVolume;
    
    int threadCount = niftiChunkLength ;
    long int remainingSize = niftiVolume * numFiles * sizeof(float); 
    int totalChunks = ceil(float(niftiVolume)/float(niftiChunkLength));
    int connectivityDataSize = niftiVolume  * sizeof(float);
    int seedDataSize = numFiles * sizeof(float);
    long int offset = niftiChunkLength * numFiles;


    int chunkDataSize = niftiChunkLength * numFiles  * sizeof(float);

    //set the device to optimize amount of shared or global memory space
    // options are:
    //    cudaFuncCachePreferNone:  default config
    //    cudaFuncCachePreferShared:  prefer larger shared memory and smaller L1
    //    cudaFuncCachePreferL1  :    prefer larger L1 cache
    //    cudaFuncCachePreferEqual:  equal L1 and shared
    gpuErrorCheck( cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) );
    
    // zero out the connectivitydata before copy
    for ( int i = 0; i < niftiVolume; i++)
    {
        connectivityData[i] = 0.0;
    }


    // allocate device space
    gpuErrorCheck( cudaMalloc((void**) &d_normalizedData, chunkDataSize));
    gpuErrorCheck( cudaMalloc((void**) &d_connectivityData, connectivityDataSize));
    gpuErrorCheck( cudaMalloc((void**) &d_seedData, seedDataSize));
    gpuErrorCheck( cudaMalloc((void**) &d_numFiles, sizeof(int)));
    gpuErrorCheck( cudaMalloc((void**) &d_niftiVolume, sizeof(int)));
    gpuErrorCheck( cudaMalloc((void**) &d_seed, sizeof(int)));
    gpuErrorCheck( cudaMalloc((void**) &d_iteration, sizeof(int)));



    // copy from host to dev
    //gpuErrorCheck( cudaMemcpy(d_normalizedData, normalizedData, chunkDataSize, cudaMemcpyHostToDevice));
    gpuErrorCheck( cudaMemcpy(d_connectivityData, connectivityData, connectivityDataSize, cudaMemcpyHostToDevice));
    gpuErrorCheck( cudaMemcpy(d_seedData, normalizedData + seed * numFiles, seedDataSize, cudaMemcpyHostToDevice));

    gpuErrorCheck( cudaMemcpy(d_numFiles, &numFiles, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck( cudaMemcpy(d_niftiVolume, &niftiChunkLength, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck( cudaMemcpy(d_seed, &seed, sizeof(int), cudaMemcpyHostToDevice));
    //setup the kernel

    int numberBlocks = ceil(float(threadCount)/float(threadsPerBlock));

    //printf("totalChunks: %d \n", totalChunks);
    for ( int i =0; i< totalChunks; i ++)
    {
        if ( chunkDataSize > remainingSize)
          chunkDataSize = remainingSize;
        gpuErrorCheck( cudaMemcpy(d_normalizedData, normalizedData + offset *i, chunkDataSize, cudaMemcpyHostToDevice));
        gpuErrorCheck( cudaMemcpy(d_iteration, &i, sizeof(int), cudaMemcpyHostToDevice));
    

	cudaEvent_t start_event, stop_event;  
	subTime = 0.0;
	gpuErrorCheck(cudaEventCreate(&start_event));
	gpuErrorCheck(cudaEventCreate(&stop_event));
	gpuErrorCheck(cudaEventRecord(start_event, 0));

	d_dotProduct <<< numberBlocks, threadsPerBlock >>>(d_normalizedData, d_connectivityData, d_seedData, d_numFiles, d_niftiVolume, d_iteration);
        remainingSize -= chunkDataSize;

	gpuErrorCheck(cudaEventRecord(stop_event, 0));
	gpuErrorCheck(cudaEventSynchronize(stop_event));

	gpuErrorCheck(cudaEventElapsedTime(&subTime, start_event, stop_event));
	subTime /= 1.0e3f;
	*runTime += subTime;
    
    }//end for i

    // copy from dev to host
    gpuErrorCheck( cudaMemcpy(connectivityData, d_connectivityData, connectivityDataSize, cudaMemcpyDeviceToHost));
    //gpuErrorCheck( cudaMemcpy(normalizedData, d_normalizedData, fullDataSize, cudaMemcpyDeviceToHost));


    // free data sets
    gpuErrorCheck( cudaFree(d_normalizedData));
    gpuErrorCheck( cudaFree(d_connectivityData));
    gpuErrorCheck( cudaFree(d_seedData));
    gpuErrorCheck( cudaFree(d_numFiles));
    gpuErrorCheck( cudaFree(d_niftiVolume));
    gpuErrorCheck( cudaFree(d_seed));
    gpuErrorCheck( cudaFree(d_iteration));



    if (PRINT){  //for testing
	printf("          Result Connectivity Data Parallel   for SEED = %d:  \n", seed);
	printMatrixConnectivity(connectivityData, 1, niftiVolume);
    }//end if PRINT
    return 0; //success

}//end connectivityPar


/************************************************************************
 *                                                                      *
 *            HELPER FUNCTIONS:  FOR TESTING                            *
 *                                                                      *
 ************************************************************************/
void printMatrixConnectivity(float* matrix, int iDim, int jDim)
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


