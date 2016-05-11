 /****************************************************************************
 *  Roy Wong
 *  Dan Rolfe
 *  Keri Anderson
 *
 *  CS6235  CUDA  Final Assignment  
 *   
 *  This file runs the sequential version of fMRI Connectivity.
 *
 *  Steps:  (called from fMRI_Main.c)
 *    
 *       1)  Create Covariates Matrix Inverse
 *       2)  Create Point Vector Data
 *       3)  Clean Data
 *       4)  Normalize Data
 *       5)  Calculate Connectivity
 *
 *  
 *****************************************************************************/

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


#include "fMRI_Sequential.h"
#include <math.h>
//#include <float.h> // used to test whether or not calculations exceed float capacity

#define PRINT 0  //0 for off, 1 for on

//pre-declare functions
void printMatrixSequential(float* matrix, int iDim, int jDim);
void matrixMultiply(float* matrixA, float* matrixB, float* matrixC, int iDimA, int jDimAiDimB, int jDimB);

/************************************************************************
 *                                                                      *
 *            COVARIATES TRANSPOSE                                      *
 *                                                                      *
 ************************************************************************/
/***
 *  This method will only be done sequentially.  No gain on GPU.
 */
//for whatever reason, no "C" is required after 'extern' here - something to do with this being a .c file?
extern int covariatesTranspose(float* covariates, float* covTranspose, int numCovariates, int numFiles)
{
    int i, j;
   
    for (i = 0; i < numFiles; i++){
	for (j = 0; j < numCovariates; j++){
	    covTranspose[j*numFiles + i] = covariates[i*numCovariates + j];
	}//end for j
    }//end for i

    if (PRINT){//for testing
	printf("          Covariate Matrix:  \n");
	printMatrixSequential(covariates, numFiles, numCovariates);
	printf("          Covariates Transpose:  \n");
	printMatrixSequential(covTranspose, numCovariates, numFiles);
    }//end if PRINT

    return 0; //success

}//end covariatesTranspose

/************************************************************************
 *                                                                      *
 *            COVARIATESTRANS X COVARIATES                              *
 *                                                                      *
 ************************************************************************/
/***
 *  calulates matrix matrix multiplication
 */
extern int covariatesTransCovariatesSeq(float* covariates, float* covTranspose, float* covTransXCov, int numCovariates, int numFiles)
{
    int i, j, k;
       
    for (i = 0; i < numCovariates; i++){
	for (j = 0; j < numCovariates; j++){
	    float temp = 0.0;
	    for (k = 0; k < numFiles; k++){
		temp += covTranspose[i*numFiles + k] * covariates[k*numCovariates + j];
	    }//end for k
	    covTransXCov[i*numCovariates + j] = temp;	    
	}//end for j
    }//end for i

    if (PRINT){//for testing
	printf("          Result MatrixMultiply Sequential:  \n");
	printMatrixSequential(covTransXCov, numCovariates, numCovariates);
    }//end if TEST

    return 0; //success

}//end covariatesTransCovariatesSeq

/************************************************************************
 *                                                                      *
 *            COVARIATES INVERSE                                        *
 *                                                                      *
 ************************************************************************/
/***
 *  This method will only be done sequentially.  No gain on GPU.
 */
//for whatever reason, no "C" is required after 'extern' here - something to do with this being a .c file?
extern int covariatesInverse(float* matrixA, float* matrixAInverse, int dim)
{	
    int i, j, k;
    double* workspace  = (double*) malloc ( dim * (2*dim) * sizeof(double) ); //holds A and Identity

    //copy matrixA and the identity into workspace  (augment the matrix)
    // example:  
    //           |  3   2   1  |  1   0   0  |
    //           |  1   5   0  |  0   1   0  |
    //           |  4   3   1  |  0   0   1  |

    int numCols = 2*dim;
    for (i = 0; i < dim; i++){
	for (j = 0; j < numCols; j++){
	    if (j < dim){
		workspace[i*numCols + j] = (double)matrixA[i*dim+j];
	    }else{//place the identity matrix
		if (j == i + dim){
		    workspace[i*numCols+j] = 1.0;
		}else{
		    workspace[i*numCols+j] = 0.0;
		}///end if j ==2*i
	    }//end if j < dim
	}//end for j
    }//end for i
    
 

    //perform matrix operations until we have the identity on the left side    
    // example:  
    //           |  1   0   0  |  -5/4   -1/4     5/4  |
    //           |  0   1   0  |   1/4    1/4    -1/4  |
    //           |  0   0   1  |  17/4    1/4   -13/4  |

    for (i = 0; i < dim; i++){  // 'i' will walk through the rows, 'zero-ing' out everything in the 'ith' column
	//find the first row with a non-zero element in the 'ith' column
	int firstNonZeroRow = -1;
	for (k = i; k < dim; k++){
	    double element = workspace[k*numCols + i];
	    //printf(" i is %d  k is %d and element is %3.3f\n", i, k, element ); //for testing
	    if (element != 0.0){//found it
		firstNonZeroRow = k;
		break;  
	    }//end if element != 0.0
	}//end for k
	if (firstNonZeroRow == -1){//could not find nonzero row ==> singular matrix
	    printf("ERROR MATRIX INVERSION:  SINGULAR MATRIX.  PROGRAM ABORT\n\n");
	    return 1;
	}//end if

	if (k != i){// need to swap rows
	    for (j = 0; j < numCols; j++){
		double temp = workspace[i*numCols + j];
		workspace[i*numCols + j] = workspace[k*numCols + j];
		workspace[k*numCols + j] = temp;
	    }//end for j
	}//end if k!= i

	//when we get here, good to go with "zero-ing" out everthing in the column
	//first, place a '1' in this diagonal element
	double divisor = workspace[i*numCols + i];
	for (k = i; k < numCols; k++)
	    workspace[i*numCols + k] = workspace[i*numCols + k] / divisor;

	//next, zero-out everything int the current column
	for (k = 0; k< dim; k++){
	    if (k != i){
		double factor = workspace[k*numCols + i] * (-1.0);

		for (j = i; j < numCols; j ++){
		    workspace[k*numCols + j] = workspace[k*numCols + j] + (factor * workspace[i*numCols + j]);
		}//end for j
	    }//end if k!= i
	}//end for k
    }//end for i
		

 
    //copy result into matrixAInverse
    for (i = 0; i < dim; i++)
	for (j = 0; j < dim; j++)
	    matrixAInverse[i*dim + j] = (float)workspace[i*numCols + j + dim];

    free(workspace);

    
    if (PRINT){//for testing
	printf("          Result MatrixInverse Sequential:  \n");
	printMatrixSequential(matrixAInverse, dim, dim);

	//also calclulate and show the result for matrix * inverse
	float* identity  = (float*) malloc ( dim * (2*dim) * sizeof(float) ); //holds A and Identity
	for (i = 0; i < dim; i++){
	    for (j = 0; j < dim; j++){
		float temp = 0.0;
		for (k = 0; k < dim; k++){
		    temp += matrixA[(i*dim) + k] * matrixAInverse[(k*dim) + j];
		}//end for k
		//take the absolute value for simplicity - (removes a lot of -0.0000000)
		if (temp < 0.0)
		    temp = temp * -1.0;

		identity[i*dim+j] = temp;
	    }//end for j
	}//end for i

	printf("          Resulting Identity Matrix Sequential:  \n");
	printMatrixSequential(identity, dim, dim);

	free(identity);
    }//end if PRINT
    

    return (0); //success

}//end sequentialCovariatesInverse



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
//for whatever reason, no "C" is required after 'extern' here - something to do with this being a .c file?
extern int transposeNiftiDataSeq(float* originalMatrix, float* transposedMatrix, int iDim, int jDim)
{
    int i, j;

    // straight forward, naive implementation
    for (i = 0; i < iDim; i++){
	for (j = 0; j < jDim; j++){
	    transposedMatrix[j*iDim + i] = originalMatrix[i*jDim+j];	    
	}//end for j
    }//end for i

    if (PRINT){//for testing	
	printf("          Original NIFTI Data:  \n");
	printMatrixSequential(originalMatrix, iDim, jDim);
	printf("          Result Point Data Sequential:  \n");
	printMatrixSequential(transposedMatrix, jDim, iDim);
    }//end if PRINT

    return 0; //success 

}//end sequentialPointData 


/************************************************************************
 *                                                                      *
 *            CLEAN NIFTI DATA                                          *
 *                                                                      *
 ************************************************************************/
/***
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
 *  Once we have these point vectors, we can use them with the covariate data to find an estimate for
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
 */
//for whatever reason, no "C" is required after 'extern' here - something to do with this being a .c file?
extern int cleanSeq(float* pointData, float* cleanedData, float* covTranspose, float* matrixInverse, int numCovariates, int numFiles, int niftiVolume)
{
    int i, j, k, pointDataBeg;   

    float* xTransposeY = (float*) malloc ( numCovariates * sizeof(float) );
    float* betas       = (float*) malloc ( numCovariates * sizeof(float) );  
   
    //double maxFloatValue = (double)FLT_MAX;
    //float maxValue = 0.0;    

    for (i = 0; i < niftiVolume; i++){	
	//printf("  current point is:  %d\n",  i);
	int pointIndex = i*numFiles;  // index into the beginning of the vector for the current point
	
	//Step 1:  Calculate xTranspose * Y      : will result in a 27 X 1 vector
	matrixMultiply(covTranspose, pointData+(pointIndex),  xTransposeY, numCovariates, numFiles, 1);

	//Step 2:  Calculate betas = (Xinverse) * (xTranspose * Y)    : results in a 27 X 1 vector
	matrixMultiply(matrixInverse, xTransposeY, betas, numCovariates, numCovariates, 1);


	//Step 3:  Calculate U = Y - b1X1 + b2X2 + b3X3       Results in a 1190 X 1 vector
	double sumOfBetas = 0;

	for (j= 0; j < numFiles; j++){
	    //first calculate b1x1 + b2X2 + b3X3
	    sumOfBetas = 0.0;
	    for (k = 0; k < numCovariates; k++){
		sumOfBetas += (double)(betas[k] * covTranspose[k*numFiles + j]);
	    }//end for k
	    
	    // This code is for testing whether or not the float capacity is exceeded
	    //float betaFloat = (float)sumOfBetas;
	    //if (betaFloat> maxValue)
	    //	maxValue = betaFloat;

	    //if (sumOfBetas > maxFloatValue){
	    //	printf("Data Cleaning:  Exceeded float capacity i = %d, program abort. \n", i);
	    //	return 1; //error
	    //}
	    
	    //next calculate U = Y - sumOfBetas
	    cleanedData[pointIndex + j] = (float)(pointData[pointIndex + j] - sumOfBetas);
	}//end for j

	
    }//end for i
    //printf("max clean float value is %f\n", maxValue);
    //printf("max float       value is %f\n", (float)maxFloatValue);

    if (PRINT){//for testing
	printf("          Result Cleaned Data Sequential:  \n");
	printMatrixSequential(cleanedData, niftiVolume, numFiles);

    }//end if TEST

    //free data structures
    free(xTransposeY);
    free(betas);

    return 0;  //success

}//end cleanSeq


/************************************************************************
 *                                                                      *
 *            NORMALIZE DATA                                            *
 *                                                                      *
 ************************************************************************/
 /* 
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
//for whatever reason, no "C" is required after 'extern' here - something to do with this being a .c file?
extern int normalizeDataSeq(float* cleanedData, float* normalizedData, int numFiles, int niftiVolume)
{
    int i, j, currentPointBeg;
    float mean, stdDev;

    for (i = 0; i < niftiVolume; i++){
	
	//get the location in the cleanedData structure where the current point begins
	currentPointBeg = i*numFiles;

	//Step 1:  get the mean for the current point vector
	mean = 0.0;  
	for (j = 0; j < numFiles; j++)
	    mean += cleanedData[currentPointBeg + j];
	mean = mean / (numFiles * 1.0);


	//Step 2:  subtract the mean from all of the points in the vector
	//   also caluculate part of step 3
	stdDev = 0.0;
	for (j = 0; j < numFiles; j++){
	    float temp = cleanedData[currentPointBeg + j] - mean;
	    normalizedData[currentPointBeg+j] = temp;
            //part of step 3 - accumulate squared numbers into stDev total
	    stdDev += temp * temp;
	}// end for j

	//Step 3:  calculate the standard diviation
	stdDev = stdDev / ((numFiles - 1)*1.0);
	stdDev = sqrt(stdDev);

        //for testing  - comment out
        //printf("for point %d,  mean = %2.4f,  stdev = %2.4f\n\n", pointBeg+i, mean, stdDev); 

	
	//Step 4:  take data from step 2 and divide by std dev
        for (j = 0; j < numFiles; j++)
	    normalizedData[currentPointBeg+j] = normalizedData[currentPointBeg+j]/ stdDev;
	
    }//end for i

    if (PRINT){  //for testing
	printf("          Original Clean Data Sequential:  \n");
	printMatrixSequential(cleanedData, niftiVolume, numFiles);
	printf("          Result Normalized Data Sequential:  \n");
	printMatrixSequential(normalizedData, niftiVolume, numFiles);
    }//end if PRINT

    return 0; //success

}//end normalizeDataSeq

/************************************************************************
 *                                                                      *
 *            CALCULATE CONNECTIVITY                                    *
 *                                                                      *
 ************************************************************************/
/*
 *   This step is computationally and time-wise intensive.  In order to 
 *   test smaller sets of data, the parameters "pointBeg" and "pointEnd"
 *   have been added.  Instead of caclulating for all niftiVolume = 91*109*91
 *   points, we can just calculate pointBeg == 0, and pointEnd == 5, so
 *   6 points total.  
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
extern int connectivitySeq(int seed, float* normalizedData, float* connectivityData, int numFiles, int niftiVolume)
{
    //'seed' is the chosen point in the brain for which we caculate connectivity

    int i, j;

    if (PRINT){
	printf("          Original Normalized Data");
	printMatrixSequential(normalizedData, niftiVolume, numFiles);
    }//end if PRINT
   

    for (i = 0; i < niftiVolume; i++){
	//calculate the dot product between the 'seed' and point 'i'
	float dotProduct = 0;
	for (j = 0; j < numFiles; j++){
	    dotProduct += normalizedData[i*numFiles + j] * normalizedData[seed*numFiles + j];
	}//end for j

       
	//printf("dot product for i = %d is  %5.5f\n", i, dotProduct);//for testing

	//divide dot product by the number of time points - 1 and store the result
	connectivityData[i] = dotProduct / (numFiles - 1);
	
    }//end for i


    if (PRINT){  //for testing
	printf("          Result Connectivity Data Sequential for SEED = %d:  \n", seed);
	printMatrixSequential(connectivityData, 1, niftiVolume);
    }//end if PRINT


    return 0; //success

}//end connectivitySeq


/************************************************************************
 *                                                                      *
 *            HELPER FUNCTIONS:  FOR TESTING                            *
 *                                                                      *
 ************************************************************************/

void matrixMultiply(float* matrixA, float* matrixB, float* matrixC, int iDimA, int jDimAiDimB, int jDimB)
{
    int i, j, k;
    //commented out code is used to test whether or not the float capacity is exceeded
    //double maxFloatValue = (double)FLT_MAX;


    for (i = 0; i < iDimA; i++){
	for (j = 0; j < jDimB; j++){
	    double temp = 0;
	    //matrixC[i*jDimB + j] = 0;
	    for (k = 0; k < jDimAiDimB; k++){
		temp += (double)(matrixA[i*jDimAiDimB + k] * matrixB[k*jDimB + j]);
	    }//end for k
	    //if (temp > maxFloatValue){
	    //	printf("in matrix multiply:  exceeded float capacity\n");
	    //}else
	    //{
	    matrixC[i*jDimB + j] = (float)temp;
		//}
	}//end for j
    }//end for i

}//end matrixMultiply

void printMatrixSequential(float* matrix, int iDim, int jDim)
{
    int i, j;
    for (i = 0; i < iDim; i++){
	printf("\n          ");
	for (j = 0; j < jDim; j++){
	    printf("     %.7f,   ", matrix[i*jDim + j]);   
	    
	}//end for j
    }//end for i

    printf("\n\n");
	    
}//end printMatrix
