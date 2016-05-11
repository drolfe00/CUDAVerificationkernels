 /****************************************************************************
 *  Roy Wong
 *  Dan Rolfe
 *  Keri Anderson 
 *
 *  CS6235  CUDA Final Project
 *  Due April 2014
 *
 *
 *  To Run:  ./connectivity
 *  To Compile:  use "make" command  (use makefile provided)
 *  Use gradlabX.eng.utah.edu, where X can be 1, 2, 3, 4, .... 13
 *  (not all extensions have CUDA installed)
 *
 *                 
 *
 *  EMACS  notes:
 *  To get syntax highlighting in Emacs:  alt-x  c-mode
 *  To get line numbers:  alt-x global-linum-mode
 *
 *  Needed files:  
 *
 *
 *  PROGRAM DESCRIPTION: 
 *  This program reads in NIFTI data, "cleans" the noise from it, and
 *  calculates how "connected" each point in the brain is to every other
 *  point in the brain.  
 * 
 *  Steps:
 *      1)  Query Cuda Device and read in NIFTI files
 *      2)  Run and Time covariate inverse calculation:  Sequential and Parallel
 *      3)  Run and Time nifti Data tranpose into point data:  Sequential and Parallel
 *      4)  Run and Time data cleaning:  Sequential and Parallel
 *      5)  Optional:  write cleaned data to disc
 *      6)  Run and Time data normalization: Sequential and Parallel
 *      7)  Run and Time data connectivity for Seed:  Sequential and Parallel  - 'seed' = one point in the brain
 *      8)  Pring final runtime statistics
 *  
 ***************************************************************************/

/*******************************************
 *  TODO:
 *      *)   
 *      *)  
 *      *)  
 *      *)  
 *      *)  
 *
 *
 *
 *
 *******************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "GPUDeviceQuery.h"  
#include "nifti1_Read_Write.h"     

//number of times to run the code for runtime stats
#define TIMESTORUN 20

//for testing  **** THESE NUMBERS CAN CHANGE FOR TESTING
#define TEST 0  //1 for testing, 0 to turn it off
#define TESTX 2 // dim X
#define TESTY 2 // dim Y
#define TESTZ 2 // dim Z
#define TESTNUMFILES 4  //number of scans taken  
#define TESTNUMCOVARIATES 3 //number of covariates to use

#define TESTSEED 4  // Seed is the point in the brain that will be used to calulate connectivity
                    // with all other points

//FIXED DATA  **** DO NOT CHANGE
#define NUMCOVARIATES 15 //changed from 27 - we are now using covariates 0..11,  24, 25, 26
#define NIFTIDIMX  91   //the x, y, z dimensions from the NIFTI files
#define NIFTIDIMY  109
#define NIFTIDIMZ  91
#define NIFTIBASESTRING  "nifti1Data/rfMRI_REST1_LR_"
#define NIFTIEXTENSION   ".nii"
#define COVARIATESFILE   "nifti1Data/covariates.txt"


#define SEED 4 // Seed is the point in the brain that will be used to calulate connectivity
               // with all other points

#define ERRORMARGIN .01 //margin of error when comparing result


//this function queries the CUDA device and returns info
extern "C" GPUDetails* queryCUDADevice(int verbose);

//these function reads in the NIFTI files or writes to new NIFTI files
extern "C" int read_nifti_file(char* data_file, float* dataArray, int verbose);
extern "C" int write_nifti_file(char* hdr_file, char* data_file, int berbose);

//Sequential functions  - in fMRI_Sequential.c
//for whatever reason, "C" IS required in this file even though it is not used in the fMRI_Sequential.h and .c files
extern "C" int covariatesTranspose(float* covariates, float* covTranspose, int numCovariates, int numFiles);
extern "C" int covariatesTransCovariatesSeq(float* covariates, float* covTranspose, float* covTransXCov, int numCovariates, int numFiles);
extern "C" int covariatesInverse(float* matrixA, float* matrixAInverse, int dim);

extern "C" int transposeNiftiDataSeq(float* originalMatrix, float* transposedMatrix, int iDim, int jDim);
extern "C" int cleanSeq(float* pointData, float* cleanedData, float* covTranspose, float* matrixInverse, int numCovariates, int numFiles, int niftiVolume);

extern "C" int normalizeDataSeq(float* cleanedData, float* normalizedData, int numFiles, int niftiVolume);
extern "C" int connectivitySeq(int seed, float* normalizedData, float* connectivityData, int numFiles, int niftiVolume);
   

//CUDA Parallel functions  - in fMRI_Covariate.cu  (Keri)
extern "C" int covariatesTransCovariatesPar(float* covariates, float* covTranspose, float* covTransXCov, int numCovariates, int numFiles, float* runTime);

//CUDA Parallel functions  - in fMRI_Clean.cu  (Roy)
extern "C" int transposeNiftiDataPar(float* originalMatrix, float* transposedMatrix, int iDim, int jDim, float* runTime);
extern "C" int cleanPar(float* pointData, float* cleanedData, float* covTranspose, float* matrixInverse, int numCovariates, int numFiles, int niftiVolume, float* runTime);

//CUDA Parallel functions  - in fMRI_Connectivity.cu  (Dan)
extern "C" int normalizeDataPar(float* cleanedData, float* normalizedData, int numFiles, int niftiVolume, float* runTime);
extern "C" int connectivityPar(int seed, float* normalizedData, float* connectivityData, int numFiles, int niftiVolume, float* runTime);

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

//pre-declare functions
void getNiftiFileName(int i, char* nameString);
int readCovariateFile(float* covariates, int numFiles);
int readNiftiFiles(float* niftiFiles, int numFiles, int fileVolume, int begFileNumber);
int writeNiftiFiles();
void checkInverse(float* covTransXCov, float* matrixInverse, int numCovariates);
void compareTransXCovariates(float* covTransXCovSeq, float* covTransXCovPar, int numCovariates);
void comparePointData(float* pointDataSeq, float* pointDataPar, int numFiles, int niftiVolume);
void compareCleanData(float* cleanedDataSeq, float* cleanedDataPar, int numFiles, int niftiVolume);
void compareNormalizedData(float* normalizedDataSeq, float* normalizedDataPar, int numFiles, int niftiVolume);
void compareConnectivityData(float* connectivityDataSeq, float* connectivityDataPar, int niftiVolume, int seed);



/************************************************************************
 *                                                                      *
 *            MAIN                                                      *
 *                                                                      *
 ************************************************************************/
int main(int argc, char **argv)
{
    int niftiVolume = 0;
    int begFileNumber = 0;
    int endFileNumber = 0;  
    int numFiles = 0;
    int numCovariates = 0;
    int dimX = 0;
    int dimY = 0;
    int dimZ = 0;
    int seed = 0;

    if (TEST){
	numFiles = TESTNUMFILES;
	dimX = TESTX;
	dimY = TESTY;
	dimZ = TESTZ;
	niftiVolume = dimX * dimY *dimZ;
	numCovariates = TESTNUMCOVARIATES;
	seed = TESTSEED;

    }else{

	begFileNumber = 11;
	endFileNumber = 1200;  

	numFiles = endFileNumber - begFileNumber + 1;
	dimX = NIFTIDIMX;
	dimY = NIFTIDIMY;
	dimZ = NIFTIDIMZ;
	niftiVolume = dimX*dimY*dimZ;
	numCovariates = NUMCOVARIATES;
	seed = SEED;

    }//end if TEST

    //data structures for holding timing stats
    float seqTimeInverse[TIMESTORUN + 1];
   
    float seqTimePointData[TIMESTORUN + 1];
    
    float seqTimeClean[TIMESTORUN + 1];
    
    float seqTimeNormalize[TIMESTORUN + 1];
    
    float seqTimeConnectivity[TIMESTORUN + 1];
    
    float seqTimeTotal[TIMESTORUN + 1];
    

    //these timings do not count the copy time from CPU to GPU and back
    float parTimeInverseNoCopy[TIMESTORUN + 1];
    float parTimePointDataNoCopy[TIMESTORUN + 1];
    float parTimeCleanNoCopy[TIMESTORUN + 1];
    float parTimeNormalizeNoCopy[TIMESTORUN + 1];
    float parTimeConnectivityNoCopy[TIMESTORUN + 1];
    float parTimeTotalNoCopy[TIMESTORUN + 1];

    //Begin TIMES TO RUN
    for (int runTime = 0; runTime < TIMESTORUN; runTime++){
	printf("STARTING RUN #%d OUT OF %d\n\n", runTime+1, TIMESTORUN);

	printf("\n\n#########################################################################################################\n");
	printf("#       STEP 1:  QUERY CUDA DEVICE AND READ IN NIFTI FILES                                              #\n");
	printf("#########################################################################################################\n");
	GPUDetails* gpuDets = queryCUDADevice(1);   //1 = verbose/print

	float* covariates = (float*) malloc ( (numFiles * numCovariates ) * sizeof (float*) );
	float* niftiData  = (float*) malloc ( (numFiles * niftiVolume   ) * sizeof (float*) );
    
	if (TEST){
	    printf("     *** RUNNING WITH TEST DATA:  ***\n");
	    printf("         niftiVolume = %dx%dx%d = %d,  numTimeFiles = %d,  numCovariates = %d\n\n", dimX, dimY, dimZ, niftiVolume, numFiles, numCovariates);
	    //create "dummy" covariate data
	    for (int i = 0; i < numFiles * TESTNUMCOVARIATES; i++ ){
		covariates[i]= (rand()%10 * 1.0); // numbers 0...9
	    }//end covariate data

	    //create "dummy nifti data
	    for (int i = 0; i < numFiles * niftiVolume; i++ ){
		niftiData[i] = (rand()%10 * 1.0);  // numbers 0...9
	    }//end niftidata
	}
	else{
	    printf("     *** RUNNING WITH NIFTI DATA:  ***\n");
	    printf("         niftiVolume = %dx%dx%d = %d,  numTimeFiles = %d,  numCovariates = %d\n\n", dimX, dimY, dimZ, niftiVolume, numFiles, numCovariates);
	    //read in Covariate File:  
	    //   This specific covariate file has '27' covariates.  There will be a set of
	    //   covariates for each time t:  so 27 elements for each of the 1200 - 11 + 1 = 1190 files.
	    //   Matrix has 1190 * 27 = 32130 elements   1190 x 27 matrix
	    //   UPDATE:  we will only use covariates 0..11, 24, 25, 26  - so 15 covariates
	    int errorCov = readCovariateFile(covariates, numFiles);
	    if (errorCov){exit(1);}
  
	    //read NIFTI files
	    // There are 1190 files, each with 91 * 109 * 91 elements.  This will be stored in 
	    // one long array for ease in passing to the GPU
	    int errorNifti = readNiftiFiles(niftiData, numFiles, niftiVolume, begFileNumber);
	    if (errorNifti){exit(1);}  

	}//end if TEST



	printf("\n#########################################################################################################\n");
	printf("#       STEP 2:  RUN AND TIME COVARIATE INVERSE CALCULATION:  SEQUENTIAL AND PARALLEL                   #\n");
	printf("#########################################################################################################\n");

	// code setup - get the covariates transpose matrix
	float* covTranspose  = (float*) malloc ( numFiles * numCovariates * sizeof(float) ); //holds  Xtrans
	int errorTranspose = covariatesTranspose(covariates, covTranspose, numCovariates, numFiles);
	if (errorTranspose){exit(1);}

	/*   SEQUENTIAL CODE   - only times covariatesTranspose * covarites */
	float* covTransXCovSeq  = (float*) malloc ( numCovariates * numCovariates * sizeof(float) ); //holds (Xtrans * X)

	printf("\n     ...RUNNING COVARIATES TRANSPOSE X COVARIATES  SEQUENTIAL...\n");
	cudaEvent_t seq_start_event, seq_stop_event;  //begin timing sequential
	gpuErrorCheck(cudaEventCreate(&seq_start_event));
	gpuErrorCheck(cudaEventCreate(&seq_stop_event));
	gpuErrorCheck(cudaEventRecord(seq_start_event, 0));
    
	int errorSeqCovTxCov = covariatesTransCovariatesSeq(covariates, covTranspose, covTransXCovSeq, numCovariates, numFiles);
	if (errorSeqCovTxCov){exit(1);}

	gpuErrorCheck(cudaEventRecord(seq_stop_event, 0));
	gpuErrorCheck(cudaEventSynchronize(seq_stop_event));

	float seq_time_inverse = 0;
	gpuErrorCheck(cudaEventElapsedTime(&seq_time_inverse, seq_start_event, seq_stop_event));
	seq_time_inverse /= 1.0e3f;
	seqTimeInverse[runTime] = seq_time_inverse;


	/*   CUDA   CODE   - only times covariatesTranspose * covariates */   
	float* covTransXCovPar  = (float*) malloc ( numCovariates * numCovariates * sizeof(float) ); //holds (Xtrans * X)
    
	printf("\n     ...RUNNING COVARIATES TRANSPOSE X COVARIATES  PARALLEL  ...\n");
 
	int errorParCovTxCov = covariatesTransCovariatesPar(covariates, covTranspose, covTransXCovPar, numCovariates, numFiles, &parTimeInverseNoCopy[runTime]);
	if (errorParCovTxCov){exit(1);}
    
    

	//more set up - calculate the inverse
	float* matrixInverse = (float*) malloc ( numCovariates * numCovariates * sizeof(float) ); //holds (Xtrans * X)inverse 
	int errorInverse = covariatesInverse(covTransXCovSeq, matrixInverse, numCovariates);
	if (errorInverse){exit(1);}
	checkInverse(covTransXCovSeq, matrixInverse, numCovariates);


	/*   RESULTS          */
	printf("\n     **** RESULTS COVARIATES INVERSE: ****\n\n");
	//compare results
	compareTransXCovariates(covTransXCovSeq, covTransXCovPar, numCovariates);


	printf("\n          SEQ COVARIATES INVERSE RESULTS:  Time = %.5f s,  Throughput = %.4f KElements/s, Size = %u elements\n",
	       seq_time_inverse, 1.0e-3f * (numFiles*numCovariates) / seq_time_inverse, (numFiles*numCovariates) );

	printf("          PAR COVARIATES INVERSE RESULTS:  Time = %.5f s,  Throughput = %.4f KElements/s, Size = %u elements\n\n",
	       parTimeInverseNoCopy[runTime], 1.0e-3f * (numFiles*numCovariates) / parTimeInverseNoCopy[runTime], (numFiles*numCovariates) );

	//speedup
	printf("\n     **** SPEEDUP COVARIATES INVERSE compared to Sequential:  %2f  ****\n\n", seq_time_inverse/parTimeInverseNoCopy[runTime]);

	//free un-needed data structures
	free(covariates);
	free(covTransXCovSeq);
	free(covTransXCovPar);



	printf("\n#########################################################################################################\n");
	printf("#       STEP 3:  RUN AND TIME NIFTI DATA TRANSPOSE INTO POINT DATA:  SEQUENTIAL AND PARALLELL           #\n");
	printf("#########################################################################################################\n");

	/*   SEQUENTIAL CODE  */
	float* pointDataSeq  = (float*) malloc ( numFiles * niftiVolume * sizeof(float) );      

	printf("\n     ...RUNNING POINT DATA SEQUENTIAL...\n");
	gpuErrorCheck(cudaEventCreate(&seq_start_event));
	gpuErrorCheck(cudaEventCreate(&seq_stop_event));
	gpuErrorCheck(cudaEventRecord(seq_start_event, 0));
    
	int errorSeqPointData = transposeNiftiDataSeq(niftiData, pointDataSeq, numFiles, niftiVolume); 
	if (errorSeqPointData){exit(1);}

	gpuErrorCheck(cudaEventRecord(seq_stop_event, 0));
	gpuErrorCheck(cudaEventSynchronize(seq_stop_event));

	float seq_time_pointData = 0;
	gpuErrorCheck(cudaEventElapsedTime(&seq_time_pointData, seq_start_event, seq_stop_event));
	seq_time_pointData /= 1.0e3f;
	seqTimePointData[runTime] = seq_time_pointData;


	/*   CUDA   CODE  */   
	float* pointDataPar  = (float*) malloc ( numFiles * niftiVolume * sizeof(float) );     
    
	printf("\n     ...RUNNING POINT DATA PARALLEL  ...\n");


	int errorParPointData = transposeNiftiDataPar(niftiData, pointDataPar, numFiles, niftiVolume, &parTimePointDataNoCopy[runTime]); 
	if (errorParPointData){exit(1);}
  

	/*   RESULTS          */
	printf("\n     **** RESULTS POINT DATA: ****\n\n");
	//compare results
	comparePointData(pointDataSeq, pointDataPar, numCovariates, numFiles); 

	printf("\n          SEQ POINT DATA RESULTS:  Time = %.5f s,  Throughput = %.4f KElements/s, Size = %u elements\n",
	       seq_time_pointData, 1.0e-3f * (numFiles*niftiVolume) / seq_time_pointData, (numFiles*niftiVolume) );
	printf("          PAR POINT DATA RESULTS:  Time = %.5f s,  Throughput = %.4f KElements/s, Size = %u elements\n\n",
	       parTimePointDataNoCopy[runTime], 1.0e-3f * (numFiles*niftiVolume) / parTimePointDataNoCopy[runTime], (numFiles*niftiVolume) );

	//speedup
	printf("\n     **** SPEEDUP POINT DATA compared to Sequential:  %2f  ****\n\n", seq_time_pointData/parTimePointDataNoCopy[runTime]);

	//free un-needed data structures
	free(niftiData);
	//free(pointDataSeq);  change this!!!!
	free(pointDataPar);

 
	printf("\n#########################################################################################################\n");
	printf("#       STEP 4:  RUN AND TIME DATA CLEANING:  SEQUENTIAL AND PARALLEL                                   #\n");
	printf("#########################################################################################################\n");

	/*   SEQUENTIAL CODE  */
	float* cleanedDataSeq = (float*) malloc ( numFiles * niftiVolume * sizeof(float) ); //holds cleaned data values    
        
	printf("\n     ...RUNNING CLEAN DATA SEQUENTIAL...\n");
	gpuErrorCheck(cudaEventCreate(&seq_start_event));
	gpuErrorCheck(cudaEventCreate(&seq_stop_event));
	gpuErrorCheck(cudaEventRecord(seq_start_event, 0));
    
	int errorSeqClean = cleanSeq(pointDataSeq, cleanedDataSeq, covTranspose, matrixInverse, numCovariates, numFiles, niftiVolume);
	if (errorSeqClean){exit(1);}

	gpuErrorCheck(cudaEventRecord(seq_stop_event, 0));
	gpuErrorCheck(cudaEventSynchronize(seq_stop_event));

	float seq_time_clean = 0;
	gpuErrorCheck(cudaEventElapsedTime(&seq_time_clean, seq_start_event, seq_stop_event));
	seq_time_clean /= 1.0e3f;
	seqTimeClean[runTime] = seq_time_clean;

	/*   CUDA   CODE  */   
	float* cleanedDataPar = (float*) malloc ( numFiles * niftiVolume * sizeof(float) ); //holds cleaned data values     
    
	printf("\n     ...RUNNING CLEAN DATA PARALLEL  ...\n");

	int errorParClean = cleanPar(pointDataSeq, cleanedDataPar, covTranspose, matrixInverse, numCovariates, numFiles, niftiVolume, &parTimeCleanNoCopy[runTime]);
	if (errorParClean){exit(1);}
    

	/*   RESULTS          */
	printf("\n     **** RESULTS CLEAN DATA: ****\n\n");
	//compare results
	compareCleanData(cleanedDataSeq, cleanedDataPar, numFiles, niftiVolume); 

	printf("\n          SEQ CLEAN RESULTS:  Time = %.5f s,  Throughput = %.4f KElements/s, Size = %u elements\n",
	       seq_time_clean, 1.0e-3f * (numFiles*niftiVolume) / seq_time_clean, (numFiles*niftiVolume) );
	printf("          PAR CLEAN RESULTS:  Time = %.5f s,  Throughput = %.4f KElements/s, Size = %u elements\n\n",
	       parTimeCleanNoCopy[runTime], 1.0e-3f * (numFiles*niftiVolume) / parTimeCleanNoCopy[runTime], (numFiles*niftiVolume) );

	//speedup
	printf("\n     **** SPEEDUP CLEAN DATA compared to Sequential:  %2f  ****\n\n", seq_time_clean/parTimeCleanNoCopy[runTime]);

 
	//free un-needed data structures
	//free(cleanedDataSeq);
	free(cleanedDataPar);
	free(pointDataSeq);
	free(covTranspose);
	free(matrixInverse);

    

	//#########################################################################################################
	//#       STEP 5:  OPTIONAL:  WRITE CLEANED DATA TO DISC                                                  #
	//#########################################################################################################

	//first need to transpose back to NIFTI order???
	if (!TEST){ //skip this step if testing
	    printf("\n#########################################################################################################\n");
	    printf("#       STEP 5:  WRITE CLEANED DATA TO DISC                                                             #\n");
	    printf("#########################################################################################################\n\n");

	    int errorWrite =  writeNiftiFiles();
	    if (errorWrite){exit(1);}
	    printf("     ...finished writing to clean NIFTI files...\n");
	}else{//running test data
	    printf("\n#########################################################################################################\n");
	    printf("#       STEP 5:  STEP 5 SKIPPED - USING TEST DATA                                                       #\n");
	    printf("#########################################################################################################\n\n");
	}


	printf("\n#########################################################################################################\n");
	printf("#       STEP 6:  RUN AND TIME DATA NORMALIZATION:  SEQUENTIAL AND PARALLEL                              #\n");
	printf("#########################################################################################################\n");
	/*   SEQUENTIAL CODE  */
	float* normalizedDataSeq   = (float*) malloc ( numFiles * niftiVolume * sizeof(float) ); //holds normalized values
	float* normalizedDataPar   = (float*) malloc ( numFiles * niftiVolume * sizeof(float) ); //holds normalized values

	printf("\n     ...RUNNING DATA NORMALIZATION SEQUENTIAL...\n");
	gpuErrorCheck(cudaEventCreate(&seq_start_event));
	gpuErrorCheck(cudaEventCreate(&seq_stop_event));
	gpuErrorCheck(cudaEventRecord(seq_start_event, 0));

	int errorNormalizeSeq = normalizeDataSeq(cleanedDataSeq, normalizedDataSeq, numFiles, niftiVolume);
	if (errorNormalizeSeq){exit(1);}
    
	gpuErrorCheck(cudaEventRecord(seq_stop_event, 0));
	gpuErrorCheck(cudaEventSynchronize(seq_stop_event));

	float seq_time_normalize = 0;
	gpuErrorCheck(cudaEventElapsedTime(&seq_time_normalize, seq_start_event, seq_stop_event));
	seq_time_normalize /= 1.0e3f;
	seqTimeNormalize[runTime] = seq_time_normalize;

    
	/*   CUDA   CODE  */ 
	printf("\n     ...RUNNING DATA NORMALIZATION PARALLEL  ...\n");

	int errorNormalizePar = normalizeDataPar(cleanedDataSeq, normalizedDataPar, numFiles, niftiVolume, &parTimeNormalizeNoCopy[runTime]);
	if (errorNormalizePar){exit(1);}


	/*   RESULTS          */
	printf("\n     **** RESULTS DATA NORMALIZATION: ****\n\n");
	//compare results
	compareNormalizedData(normalizedDataSeq, normalizedDataPar, numFiles, niftiVolume);

	printf("\n          SEQ NORMALIZE RESULTS:  Time = %.5f s,  Throughput = %.4f KElements/s, Size = %u elements\n",
	       seq_time_normalize, 1.0e-3f * (numFiles*niftiVolume) / seq_time_normalize, (numFiles*niftiVolume) );
	printf("          PAR NORMALIZE RESULTS:  Time = %.5f s,  Throughput = %.4f KElements/s, Size = %u elements\n\n",
	       parTimeNormalizeNoCopy[runTime], 1.0e-3f * (numFiles*niftiVolume) / parTimeNormalizeNoCopy[runTime], (numFiles*niftiVolume) );

	//speedup
	printf("\n     **** SPEEDUP NORMALIZE compared to Sequential:  %2f  ****\n\n", seq_time_normalize/parTimeNormalizeNoCopy[runTime]);

	//free un-needed data structures
	free(cleanedDataSeq);
	//free(normalizedDataSeq);
	free(normalizedDataPar);


	printf("\n#########################################################################################################\n");
	printf("#       STEP 7:  RUN AND TIME DATA CONNECTIVITY FOR SEED:  SEQUENTIAL AND PARALLEL                      #\n");
	printf("#########################################################################################################\n");
 
	/*   SEQUENTIAL CODE  */
	float* connectivityDataSeq   = (float*) malloc ( niftiVolume * sizeof(float) ); //holds normalized values
	float* connectivityDataPar   = (float*) malloc ( niftiVolume * sizeof(float) ); //holds normalized values

	printf("\n     ...RUNNING CONNECTIVITY SEQUENTIAL FOR SEED = %d...\n", seed);
	gpuErrorCheck(cudaEventCreate(&seq_start_event));
	gpuErrorCheck(cudaEventCreate(&seq_stop_event));
	gpuErrorCheck(cudaEventRecord(seq_start_event, 0));

	int errorConnectivitySeq = connectivitySeq(seed, normalizedDataSeq, connectivityDataSeq, numFiles, niftiVolume);
	if (errorConnectivitySeq){exit(1);}
    
	gpuErrorCheck(cudaEventRecord(seq_stop_event, 0));
	gpuErrorCheck(cudaEventSynchronize(seq_stop_event));

	float seq_time_connectivity = 0;
	gpuErrorCheck(cudaEventElapsedTime(&seq_time_connectivity, seq_start_event, seq_stop_event));
	seq_time_connectivity /= 1.0e3f;
	seqTimeConnectivity[runTime] = seq_time_connectivity;

    
	/*   CUDA   CODE  */ 
	printf("\n     ...RUNNING CONNECTIVITY PARALLEL   FOR SEED = %d...\n", seed);

	int errorConnectivityPar = connectivityPar(seed, normalizedDataSeq, connectivityDataPar, numFiles, niftiVolume, &parTimeConnectivityNoCopy[runTime]);
	if (errorConnectivityPar){exit(1);}

	

	/*   RESULTS          */
	printf("\n     **** RESULTS CONNECTIVITY: ****\n\n");
	//compare results
	compareConnectivityData(connectivityDataSeq, connectivityDataPar, niftiVolume, seed);
  
	printf("\n          SEQ CONNECTIVITY RESULTS:  Time = %.5f s,  Throughput = %.4f KElements/s, Size = %u elements\n",
	       seq_time_connectivity, 1.0e-3f * (niftiVolume) / seq_time_connectivity, (niftiVolume) );
	printf("          PAR CONNECTIVITY RESULTS:  Time = %.5f s,  Throughput = %.4f KElements/s, Size = %u elements\n\n",
	       parTimeConnectivityNoCopy[runTime], 1.0e-3f * (niftiVolume) / parTimeConnectivityNoCopy[runTime], (niftiVolume) );

	//speedup
	printf("\n     **** SPEEDUP CONNECTIVITY compared to Sequential:  %2f  ****\n\n", seq_time_connectivity/parTimeConnectivityNoCopy[runTime]);


	//free un-needed data structures
	free(normalizedDataSeq);
	free(connectivityDataSeq);
	free(connectivityDataPar);



	printf("\n#########################################################################################################\n");
	printf("#       STEP 8:  PRINT FINAL RUNTIME STATISTICS                                                         #\n");
	printf("#########################################################################################################\n");

	float totalTimeSeq = seq_time_inverse + seq_time_pointData + seq_time_clean + seq_time_normalize + seq_time_connectivity;
	float totalTimePar = parTimeInverseNoCopy[runTime] + parTimePointDataNoCopy[runTime] + parTimeCleanNoCopy[runTime] + parTimeNormalizeNoCopy[runTime] + parTimeConnectivityNoCopy[runTime];

	seqTimeTotal[runTime] = totalTimeSeq;
	parTimeTotalNoCopy[runTime] = totalTimePar;



	printf("\n     ***      FINAL SPEEDUP FOR RUN %d out of %d (compared to Sequential): %4.4f      ***\n\n\n\n", runTime, TIMESTORUN, totalTimeSeq/totalTimePar);

    }//end times to run
    
    //print out final stats

    printf("\n\n\n\n");
    printf("\n#########################################################################################################\n");
    printf("#       FINAL RUNTIME STATISTICS                                                                        #\n");
    printf("#########################################################################################################\n");
 
    
    printf("INVERSE STATS\n");
    seqTimeInverse[TIMESTORUN] = 0.0; //for averages
    parTimeInverseNoCopy[TIMESTORUN] = 0.0;
    for (int runTime = 0; runTime < TIMESTORUN; runTime++){	
	float seqTime = seqTimeInverse[runTime];
	float parTime = parTimeInverseNoCopy[runTime];
	seqTimeInverse[TIMESTORUN] += seqTime;
	parTimeInverseNoCopy[TIMESTORUN] += parTime;
	printf("     Run:  %d:  Inverse Seq:   %.5f    Inverse Par:   %.5f    Speedup:  %.5f\n", runTime, seqTime, parTime,  seqTime/parTime);      
    }//end runTime

    printf("\n\nPOINT DATA STATS\n");
    seqTimePointData[TIMESTORUN] = 0.0; //for averages
    parTimePointDataNoCopy[TIMESTORUN] = 0.0;
    for (int runTime = 0; runTime < TIMESTORUN; runTime++){	
	float seqTime = seqTimePointData[runTime];
	float parTime = parTimePointDataNoCopy[runTime];
	seqTimePointData[TIMESTORUN] += seqTime;
	parTimePointDataNoCopy[TIMESTORUN] += parTime;
	printf("     Run:  %d:  PointData Seq:   %.5f    PointData Par:   %.5f    Speedup:  %.5f\n", runTime, seqTime, parTime,  seqTime/parTime);      
    }//end runTime

    printf("\n\nCLEAN STATS\n");
    seqTimeClean[TIMESTORUN] = 0.0; //for averages
    parTimeCleanNoCopy[TIMESTORUN] = 0.0;
    for (int runTime = 0; runTime < TIMESTORUN; runTime++){	
	float seqTime = seqTimeClean[runTime];
	float parTime = parTimeCleanNoCopy[runTime];
	seqTimeClean[TIMESTORUN] += seqTime;
	parTimeCleanNoCopy[TIMESTORUN] += parTime;
	printf("     Run:  %d:  Clean Seq:   %.5f    Clean Par:   %.5f    Speedup:  %.5f\n", runTime, seqTime, parTime,  seqTime/parTime);      
    }//end runTime

    printf("\n\nNORMALIZE STATS\n");
    seqTimeNormalize[TIMESTORUN] = 0.0; //for averages
    parTimeNormalizeNoCopy[TIMESTORUN] = 0.0;
    for (int runTime = 0; runTime < TIMESTORUN; runTime++){	
	float seqTime = seqTimeNormalize[runTime];
	float parTime = parTimeNormalizeNoCopy[runTime];
	seqTimeNormalize[TIMESTORUN] += seqTime;
	parTimeNormalizeNoCopy[TIMESTORUN] += parTime;
	printf("     Run:  %d:  Normalize Seq:   %.5f    Normalize Par:   %.5f    Speedup:  %.5f\n", runTime, seqTime, parTime,  seqTime/parTime);      
    }//end runTime

    printf("\n\nCONNECTIVITY STATS\n");
    seqTimeConnectivity[TIMESTORUN] = 0.0; //for averages
    parTimeConnectivityNoCopy[TIMESTORUN] = 0.0;
    for (int runTime = 0; runTime < TIMESTORUN; runTime++){	
	float seqTime = seqTimeConnectivity[runTime];
	float parTime = parTimeConnectivityNoCopy[runTime];
	seqTimeConnectivity[TIMESTORUN] += seqTime;
	parTimeConnectivityNoCopy[TIMESTORUN] += parTime;
	printf("     Run:  %d:  Connectivity Seq:   %.5f    Connectivity Par:   %.5f    Speedup:  %.5f\n", runTime, seqTime, parTime,  seqTime/parTime);      
    }//end runTime

    printf("\n\nTOTAL TIME STATS\n");
    seqTimeTotal[TIMESTORUN] = 0.0; //for averages
    parTimeTotalNoCopy[TIMESTORUN] = 0.0;
    for (int runTime = 0; runTime < TIMESTORUN; runTime++){	
	float seqTime = seqTimeTotal[runTime];
	float parTime = parTimeTotalNoCopy[runTime];
	seqTimeTotal[TIMESTORUN] += seqTime;
	parTimeTotalNoCopy[TIMESTORUN] += parTime;
	printf("     Run:  %d:  Total Time Seq:   %.5f    Total Time Par:   %.5f    Speedup:  %.5f\n", runTime, seqTime, parTime,  seqTime/parTime);      
    }//end runTime



    printf("*****  AVERAGES  *****\n\n");
    float aveSeqInverse = seqTimeInverse[TIMESTORUN]/ (TIMESTORUN*1.0);
    float aveParInverse = parTimeInverseNoCopy[TIMESTORUN]/ (TIMESTORUN*1.0);
    float aveInvSpeedup = aveSeqInverse/aveParInverse;
    printf("INVERSE       AVERAGES:     Ave Seq Time:  %.5f,   Ave Par Time:  %.5f,  Ave Speedup:  %.5f\n",  aveSeqInverse, aveParInverse, aveInvSpeedup);

    float aveSeqPointData = seqTimePointData[TIMESTORUN]/ (TIMESTORUN*1.0);
    float aveParPointData = parTimePointDataNoCopy[TIMESTORUN]/ (TIMESTORUN*1.0);
    float avePDSpeedup    = aveSeqPointData/aveParPointData;
    printf("POINT DATA    AVERAGES:     Ave Seq Time:  %.5f,   Ave Par Time:  %.5f,  Ave Speedup:  %.5f\n",  aveSeqPointData, aveParPointData, avePDSpeedup);

    float aveSeqClean     = seqTimeClean[TIMESTORUN]/ (TIMESTORUN*1.0);
    float aveParClean     = parTimeCleanNoCopy[TIMESTORUN]/ (TIMESTORUN*1.0);
    float aveCleanSpeedup = aveSeqClean/aveParClean;
    printf("CLEAN         AVERAGES:     Ave Seq Time:  %.5f,   Ave Par Time:  %.5f,  Ave Speedup:  %.5f\n",  aveSeqClean, aveParClean, aveCleanSpeedup);

    float aveSeqNorm     = seqTimeNormalize[TIMESTORUN]/ (TIMESTORUN*1.0);
    float aveParNorm     = parTimeNormalizeNoCopy[TIMESTORUN]/ (TIMESTORUN*1.0);
    float aveNormSpeedup = aveSeqNorm/aveParNorm;
    printf("NORMALIZE     AVERAGES:     Ave Seq Time:  %.5f,   Ave Par Time:  %.5f,  Ave Speedup:  %.5f\n",  aveSeqNorm, aveParNorm, aveNormSpeedup);

    float aveSeqConn     = seqTimeConnectivity[TIMESTORUN]/ (TIMESTORUN*1.0);
    float aveParConn     = parTimeConnectivityNoCopy[TIMESTORUN]/ (TIMESTORUN*1.0);
    float aveConnSpeedup = aveSeqNorm/aveParNorm;
    printf("CONNECTIVITY  AVERAGES:     Ave Seq Time:  %.5f,   Ave Par Time:  %.5f,  Ave Speedup:  %.5f\n",  aveSeqConn, aveParConn, aveConnSpeedup);

    float aveSeqTotal     = seqTimeTotal[TIMESTORUN]/ (TIMESTORUN*1.0);
    float aveParTotal     = parTimeTotalNoCopy[TIMESTORUN]/ (TIMESTORUN*1.0);
    float aveTotalSpeedup = aveSeqTotal/aveParTotal;
    printf("TOTAL         AVERAGES:     Ave Seq Time:  %.5f,   Ave Par Time:  %.5f,  Ave Speedup:  %.5f\n",  aveSeqTotal, aveParTotal, aveTotalSpeedup);


}// end main


/************************************************************************
 *                                                                      *
 *            READ / WRITE NIFTI FILES                                  *
 *                                                                      *
 ************************************************************************/
/***
 * Converts an integer to a string
 *
 * Specifically, we need the string in the form
 * of a 5 character long number:  such as "00011"
 * to create a final string of
 *
 *  "nifti1Data/rfMRI_REST1_LR_00011.nii"
 *  
 *
 */
void getNiftiFileName(int i, char* nameString)
{
    char const digit[] = "0123456789";
    char temp[6];
    char* tempString;

    temp[5] = '\0'; //null terminate
   
    //walk backwards through the integer loading the string
    for (int j = 4; j >=0; j--){
        temp[j] = digit[i%10];
	i = i /10;
    }

    tempString = temp;

    strcpy(nameString, NIFTIBASESTRING);
    strcat(nameString, tempString);
    strcat(nameString, NIFTIEXTENSION);


}//end intToString


/***
 * Reads in the file of covariates.
 */
int readCovariateFile(float* covariates, int numFiles)
{
     printf("     ...READING COVARIATES FILE....\n\n");
    
    FILE* fp;
    char oneWord[50];
    char c;
    int iCov = 0;

    //temporary data structure to hold 27 covariates
    int numCovariateElements = 27 * numFiles;
    float* temp = (float*) calloc ( numCovariateElements, sizeof (float*) ); //calloc initializes bits to zero


    fp = fopen(COVARIATESFILE, "r");  //open in "read" mode
    if (fp == NULL){
	printf("Covariates File Read Error:  %s   Program Abort.\n", COVARIATESFILE);
	return(1); //error
    }//end if
 
    c = fscanf(fp, "%s", oneWord);
    while(c!= EOF)    /* repeat until EOF           */
    {
	if (iCov >= numCovariateElements){
	    printf("Error Reading Covariates File:  number of elements: %d, number expected: %d.  Program Abort.\n", iCov, numCovariateElements);
	    return(1);//error
	}//end if

	temp[iCov] = atof(oneWord);                   /* convert to float and store */
	iCov++;
	c = fscanf(fp,"%s",oneWord);                        /* get next word from the file */	
    }//end while
                              
    fclose(fp);

    if (iCov != numCovariateElements){
	printf("Error Reading Covariates File:  Expected %d elements, but read %d elements.  Program Abort.\n", numCovariateElements, iCov);
	return(1);
    }
        
    // at this point, we really only want to keep covariates 0..11,  24, 25, 26  out of 1 .. 26
    for (int i = 0; i < 12*numFiles; i++){
	covariates[i] = temp[i];
    }// end for i    

    for (int i = 24; i < 27; i++){
	for (int j = 0; j < numFiles; j++){
	    covariates[(i-12)*numFiles + j] = temp[i*numFiles + j];
	}
    }

    free(temp);

    return 0;  //success 
    
}//end readCovariateFile

/***
 * Reads in the data from the 1190 NIFITI files
 */
int readNiftiFiles(float* niftiFiles, int numFiles, int fileVolume, int begFileNumber)
{

    printf("     ...READING NIFTI FILES....\n");

    char* niftiFileName = (char *) malloc ( 80 * sizeof(char));

    int errorCount = 0;
    // read in one file at a time
    for (int i = 0; i <  numFiles; i++){
	//get the file name associated with this number
	int fileNumber = i + begFileNumber;
	getNiftiFileName(fileNumber, niftiFileName);

	//read in the file to the appropriate place in the data structure
	int error = read_nifti_file(niftiFileName, niftiFiles + (i*fileVolume), 0);       //'1' for verbose
	if (error){    
	    errorCount++; 
	    printf("File Error:  %s\n", niftiFileName);
	}

    }//end for i

    free(niftiFileName);

    if (errorCount == 0){
	//printf("       Finished READING NIFTI files.\n\n");
	return 0; //success
    }else{
	printf("       Finished reading NIFTI files.  %d files had read errors.  Program Abort.\n\n", errorCount);
	return(1); //error
    }
    
}//end readNifitiFiles

/***
 * Writes cleaned data to new 1190 NIFITI files
 */
int writeNiftiFiles()
{
    printf("     ...WRITING NIFTI FILES....\n\n");

    return 0; //success
    
}//end writeNiftiFiles


/************************************************************************
 *                                                                      *
 *            HELPER FUNCTIONS FOR TESTING                              *
 *                                                                      *
 ************************************************************************/
void checkInverse(float* covTransXCov, float* matrixInverse, int numCovariates)
{
    int i, j, k;

    for (i = 0; i < numCovariates; i++){
	for (j = 0; j < numCovariates; j++){
	    float temp = 0.0;
	    for (k = 0; k < numCovariates; k++)
		temp += covTransXCov[i*numCovariates+k] * matrixInverse[k*numCovariates+j];
	    //test the result
	    temp = abs(temp);
	    float desiredValue = 0.0;
	    if (i == j)
		desiredValue = 1.0;

	    
	    if ( abs(desiredValue - temp) > ERRORMARGIN ){
		printf("          ERROR: matrix inverse test (identity matrix) not valid at [%d][%d]. Value:  %5.7f, should be: %5.7f\n", i, j, temp, desiredValue);
		return;
		
	    }//end if
	    
	}//end for j
    }//end for i

    //if we get here it's all good
    printf("          Matrix inverse valid for Sequential Version\n");

}//end checkInverse

void compareTransXCovariates(float* covTransXCovSeq, float* covTransXCovPar, int numCovariates)
{
    for (int i = 0; i < numCovariates; i++){
	for (int j = 0; j < numCovariates; j++){
	    int currentElement = i*numCovariates + j;
	    float seqElement = covTransXCovSeq[currentElement];
	    float parElement = covTransXCovPar[currentElement];
	    if ( abs(seqElement - parElement) > ERRORMARGIN){
		printf("          INVALID!!!!!  Transpose X Covariates not equal at [%d][%d]:  Should be:  %.5f,  Actual:  %.5f\n", i, j, seqElement, parElement);
		return;
	    }//end if
	}//end for j
    }//end for i

    printf("          TRANSPOSE X COVARIATES VALID!\n");

}//end compareTransXCovariates

void comparePointData(float* pointDataSeq, float* pointDataPar, int numFiles, int niftiVolume)
{
    for (int i = 0; i < niftiVolume; i++){
	for (int j = 0; j < numFiles; j++){
	    int currentElement = i*numFiles + j;
	    if ( abs(pointDataSeq[currentElement] - pointDataPar[currentElement]) > ERRORMARGIN){	
		printf("          INVALID!!!!!  Point Data Matrix not equal at [%d][%d]\n", i, j);
		return;
	    }//end if !=
	}//end for j
    }//end for i

    // if we get here, data is correct
    printf("          POINT DATA MATRIX VALID!\n");


}//end comparePointData

void compareCleanData(float* cleanedDataSeq, float* cleanedDataPar, int numFiles, int niftiVolume)
{
    int i, j;
   
    
    for (i = 0; i < niftiVolume; i++){
	for (j = 0; j < numFiles; j++){
	    int currentElement = i*numFiles + j;
	    float seqVal = cleanedDataSeq[currentElement];
	    float parVal = cleanedDataPar[currentElement];
	    if ( abs(seqVal - parVal) > ERRORMARGIN){
		printf("          INVALID!!!!!  Clean Data not equal at [%d][%d]:  Should be:  %.5f,  actual:  %.5f\n", i, j, seqVal, parVal );
		return;
	    }//end if
	}//end for j
    }//end for i

    printf("          CLEAN DATA VALID!\n");

}//end compareCleanData 

/***
 * Incoming data will be in point-vector form:
 *   
 *  rows = niftiVolume;
 *  cols = numFiles;
 *
 */
void compareNormalizedData(float* normalizedDataSeq, float* normalizedDataPar, int numFiles, int niftiVolume)
{
    int i, j;
    
    for (i = 0; i < niftiVolume; i++){
	int currentRowStartElement = i * numFiles;
	for (j = 0; j < numFiles; j++){
	    int currentElement = currentRowStartElement + j;
	    float normSeq = normalizedDataSeq[currentElement];
	    float normPar = normalizedDataPar[currentElement];
	    if (abs(normSeq - normPar) > ERRORMARGIN){
		printf("          INVALID!!!!!  Normalized Matrix not equal at [%d][%d]:  Should be:  %.5f,  actual:  %.5f\n", i, j, normSeq, normPar);
		return;
	    }//end if
	}//end for j
    }//end for i

    printf("          NORMALIZED MATRIX VALID!\n");

}//end comparNormalizedData


void compareConnectivityData(float* connectivityDataSeq, float* connectivityDataPar, int niftiVolume, int seed)
{
    int i;

    for (i = 0; i <= niftiVolume; i++){
	float conSeq = connectivityDataSeq[i];
	float conPar = connectivityDataPar[i];
	if ( abs(conSeq - conPar) > ERRORMARGIN){
	    printf("          INVALID!!!!!  Connectivity Vector not equal at [%d] for seed %d:  Should be:  %.5f,  actual:  %.5f\n", i, seed, conSeq, conPar);
		return;
	}//end if	
    }//end for i

    printf("          CONNECTIVITY VECTOR VALID FOR SEED %d!\n", seed);

}//end compareConnectivitydata


