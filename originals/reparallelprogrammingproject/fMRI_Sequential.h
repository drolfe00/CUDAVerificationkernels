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
#ifndef FMRI_SEQUENTIAL_H_INCLUDED
#define FMRI_SEQUENTIAL_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>

// functions
//for whatever reason, no "C" is required after 'extern' here - something to do with this being a .c file?
extern int covariatesTranspose(float* covariates, float* covTranspose, int numCovariates, int numFiles);
extern int covariatesTransCovariatesSeq(float* covariates, float* covTranspose, float* covTransXCov, int numCovariates, int numFiles);
extern int covariatesInverse(float* matrixA, float* matrixAInverse, int dim);

extern int transposeNiftiDataSeq(float* originalMatrix, float* transposedMatrix, int iDim, int jDim);
extern int cleanSeq(float* pointData, float* cleanedData, float* covTranspose, float* matrixInverse, int numCovariates, int numFiles, int niftiVolume);

extern int normalizeDataSeq(float* cleanedData, float* normalizedData, int numFiles, int niftiVolume);
extern int connectivitySeq(int seed, float* normalizedData, float* connectivityData, int numFiles, int niftiVolume);

#endif
