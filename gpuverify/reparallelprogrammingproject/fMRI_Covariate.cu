/****************************************************************************
 *  Roy Wong
 *  Dan Rolfe
 *  Keri Anderson 
 *
 *  CS6235  CUDA Final Project
 *  Due April 2014
 *
 *
 *  This file runs the CUDA parallel version of cacluating the Covariates
 *  matrix Inverse.
 *
 *  Steps:  (called from fMRI_Main.c)
 *    
 *       1)Create X Transpose (X = Covariate matrix)  
 *       2)Calculate XTranspose * X
 *       3)Calculate (XTranspse * X) inverse  
 * 
 *  
 ***************************************************************************/


/*******************************************
 *  TODO:
 *      *)  X Transpose
 *      *)  XTranspose * X
 *      *)  (XTranspose *X) inverse
 *      *)  
 *      *)    
 *
 *
 *
 *
 *******************************************/
#include <stdio.h>
#include <stdlib.h>
#include "fMRI_Covariate.h"

//for testing:  print out the calculated matrix
#define PRINT 0  //0 for "off"  1 for "on"

#define CHUNKSIZE 32  //for shared memory transpose - assumes num Covariates will not be > 32

//pre-declare function calls
void printMatrixCovariate(float* matrix, int iDim, int jDim);

/***
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
 *  This kernel multiplies CovariatesTranspose * Covariates
 *
 *
 *
 *                                     Covariates Transpose                                              X                            Covariates
 * 
 *   #################################################        #####################################                      #####################################
 *   #   0 |  27 | *** | 702 # 729 | 756 | *** |1431 #        #31347|31374| *** |32049#37076|32103#             blk(0,0) #   0 |   1 |   2 | ... |  25 |  26 # 
 *   #***********************#***********************#        #***********************#***********#                      #***********************************#
 *   #   1 |  28 | *** | 703 # 730 | 757 | *** |1432 #        #31348|31375| *** |32050#32077|32104#                      #  27 |  28 |  29 | ... |  52 |  53 #
 *   #***********************#***********************#  * * * #***********************#***********#                      #***********************************#
 *   #   2 |  29 | *** | 704 # 731 | 758 | *** |1433 #        #31349|31376| *** |32041#32078|32105#                      #               *                   #
 *   #***********************#***********************#        #***********************#***********#                      #               *                   #
 *   #          ***          #          ***          #        #          ***          #    ***    #                      #               *                   #
 *   #***********************#***********************#        #***********************#***********#                      #               *                   #
 *   #  25 |  52 | *** | 727 # 754 | 781 | *** |1456 #        #31372|31399| *** |32074#32101|32138#                      #               *                   #
 *   #***********************#***********************#        #***********************#***********#                      #***********************************#
 *   #  26 |  53 | *** | 728 # 755 | 782 | *** |1457 #        #31373|31400| *** |32075#32102|32129#                      # 702 | 703 | 704 | ... | 727 | 728 #
 *   #################################################        #####################################                      ##################################### 
 *                                                                                                              blk(1,0) # 729 | 730 | 731 | ... | 754 | 755 #
 *                                                                                                                       #***********************************#  
 *        Result is a 27 x 27 matrix                                                                                     # 756 | 757 | 758 | ... | 781 | 782 #           
 *                                                                                                                       #***********************************#           
 *                                                                                                                       #               *                   #             
 *        Use blocks of 27x27 elements, each summing to a shared memory 27 x 27 matrix                                   #               *                   #            
 *                                                                                                                       #               *                   #
 *        myTransposedElement = (threadIdx.y*numFiles) + blockIdx.x*numCovariates + threadIdx.x;                         #               *                   #            
 *        myOriginalElement = (((blockIdx.x*numCovariates) + threadIdx.y)*numCovariates) + threadIdx.x;                  #               *                   #
 *                                                                                                                       #***********************************#            
 *                                                                                                                       #1431 |1432 |1433 | ... |1456 |1457 #
 *        myResultElement = threadIdx.y * numCovariates + threadIdx.x;                                                   #####################################            
 *                                        
 *                                                                                                                                   *  *  *                              
 *
 *                                                                                                                       #####################################        
 *                                                                                                              blk(43,0)#31347|31348|31349| ... |31372|31373#
 *                                                                                                                       #***********************************#
 *       UPDATE:  switched to 32*32 blocks                                                                               #31374|31375|31376| ... |31399|31400#
 *                                                                                                                       #***********************************# 
 *                                                                                                                       #               *                   #
 *                                                                                                                       #               *                   #
 *                                                                                                                       #               *                   #
 *                                                                                                                       #               *                   #
 *                                                                                                                       #               *                   #
 *                                                                                                                       #***********************************#
 *                                                                                                                       #32049|32050|32041| ... |32074|32075#
 *                                                                                                                       #####################################
 *                                                                                                              blk(44,0)#32076|32077|32078| ... |32101|32102#
 *                                                                                                                       #***********************************#
 *                                                                                                                       #32103|32104|32105| ... |32138|32129#
 *                                                                                                                       ##################################### 
 *
 *
 */
__global__ void covariateMultiplyKernel(float* d_transposedMatrix, float* result, int numFiles, int numCovariates)
{
    int myTransposedElement = (threadIdx.y*numFiles) + blockIdx.x*CHUNKSIZE + threadIdx.x;
    int myResultElement = threadIdx.y * numCovariates + threadIdx.x;
    int i;

    //create a subMatrix in shared memory  32 x 33
    __shared__ float transposeChunk[CHUNKSIZE][CHUNKSIZE+1];//+1 offsets bank conflicts
    __shared__ float subMatrix[CHUNKSIZE][CHUNKSIZE];
    
    //first zero everthing out  - recall there will be 32 x 32 threads in a block
    transposeChunk[threadIdx.y][threadIdx.x] = 0.0;
    subMatrix[threadIdx.y][threadIdx.x] = 0.0;

    // use threads that will bring in appropriate data (i.e. within the 27 x 1190 data range)
    //leave 0's in the rest
                // .y < 27                   blockIdx.x * CHUNCKSIZE + threadIdx.x < numFiles
    if ( (threadIdx.y < numCovariates) && (blockIdx.x * CHUNKSIZE + threadIdx.x < numFiles) ){
	transposeChunk[threadIdx.y][threadIdx.x] = d_transposedMatrix[myTransposedElement];
    }

    __syncthreads();// make sure all the copying is done

    double temp = 0.0;
    // have *this* thread calculate it's element
    for (i = 0; i < CHUNKSIZE; i++){
	//subMatrix[threadIdx.y][threadIdx.x] += transposeChunk[threadIdx.y][i] * transposeChunk[threadIdx.x][i];
	temp += (double)(transposeChunk[threadIdx.y][i] * transposeChunk[threadIdx.x][i]);
    }

    //update *this* threads place in shared memory
    subMatrix[threadIdx.y][threadIdx.x] = (float)temp;

    //have *this* thread update the global memory atomically - there will be ~37 threads updating
    //the same location
    if (threadIdx.y < numCovariates && threadIdx.x < numCovariates)
	atomicAdd(&(result[myResultElement]), subMatrix[threadIdx.y][threadIdx.x]);

}//end covariateMultiplyKernel


/*####################################################################################
 *#                                                                                  #
 *#                    CALLED FROM fMRI_Main.cu                                      #
 *#                                                                                  #
 *####################################################################################*/

/************************************************************************
 *                                                                      *
 *            CACULATE (XTRANSPOSE*X) INVERSE                           *
 *                                                                      *
 ************************************************************************/
extern "C" int covariatesTransCovariatesPar(float* covariates, float* covTranspose, float* covTransXCov, int numCovariates, int numFiles, float* runTime)
{
    //set the device to optimize amount of shared or global memory space
    // options are:
    //    cudaFuncCachePreferNone:  default config
    //    cudaFuncCachePreferShared:  prefer larger shared memory and smaller L1
    //    cudaFuncCachePreferL1  :    prefer larger L1 cache
    //    cudaFuncCachePreferEqual:  equal L1 and shared
    gpuErrorCheck( cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) );


    //allocate memory on the GPU
    float* d_covTranspose;
    float* d_result;

    //make sure the result has been zero'd out
    int i;
    for (i = 0; i < numCovariates*numCovariates; i++)
	covTransXCov[i] = 0.0;
    

    gpuErrorCheck( cudaMalloc((void **) &d_covTranspose,  (numCovariates*numFiles)*sizeof(float)) );
    gpuErrorCheck( cudaMalloc((void **) &d_result,        (numCovariates*numCovariates)*sizeof(float)) );

    //copy the data to the GPU
    gpuErrorCheck( cudaMemcpy(d_covTranspose, covTranspose, (numCovariates*numFiles)*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrorCheck( cudaMemcpy(d_result, covTransXCov, (numCovariates*numCovariates)*sizeof(float), cudaMemcpyHostToDevice) );

    //set up the grid 
    int gridDimX = ceil(numFiles/(CHUNKSIZE*1.0));
    
    dim3 dimGrid(gridDimX, 1, 1);  // '1' means dimension is not used
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE, 1);

    cudaEvent_t start_event, stop_event;  
    gpuErrorCheck(cudaEventCreate(&start_event));
    gpuErrorCheck(cudaEventCreate(&stop_event));
    gpuErrorCheck(cudaEventRecord(start_event, 0));

    covariateMultiplyKernel<<<dimGrid, dimBlock>>>(d_covTranspose, d_result, numFiles, numCovariates);

    gpuErrorCheck(cudaEventRecord(stop_event, 0));
    gpuErrorCheck(cudaEventSynchronize(stop_event));

    *runTime = 0;
    gpuErrorCheck(cudaEventElapsedTime(runTime, start_event, stop_event));
    *runTime /= 1.0e3f;
   

    //for testing:
    printf("For testing:   time is %.4f \n", *runTime);

    //copy the data back to the Host
    gpuErrorCheck( cudaMemcpy(covTransXCov,  d_result, (numCovariates*numCovariates)*sizeof(float), cudaMemcpyDeviceToHost) );

    //GPU free
    gpuErrorCheck(cudaFree(d_covTranspose));
    gpuErrorCheck(cudaFree(d_result));
	    

    if (PRINT){
	printf("          Result MatrixMultiply Parallel:  \n");
	printMatrixCovariate(covTransXCov, numCovariates, numCovariates);
    }//end if TEST
    
    return 0; //success

}//end covariateTransCovariatePar

/************************************************************************
 *
 *        HELPER FUNCTIONS:  FOR TESTING
 *
 ***********************************************************************/
void printMatrixCovariate(float* matrix, int iDim, int jDim)
{
    //set the device to optimize amount of shared or global memory space
    // options are:
    //    cudaFuncCachePreferNone:  default config
    //    cudaFuncCachePreferShared:  prefer larger shared memory and smaller L1
    //    cudaFuncCachePreferL1  :    prefer larger L1 cache
    //    cudaFuncCachePreferEqual:  equal L1 and shared
    gpuErrorCheck( cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) );

    int i, j;
    for (i = 0; i < iDim; i++){
	printf("\n          ");
	for (j = 0; j < jDim; j++){
	    printf("     %.3f,   ", matrix[i*jDim + j]);   
	    
	}//end for j
    }//end for i

    printf("\n\n");
	    
}//end printMatrix

  
