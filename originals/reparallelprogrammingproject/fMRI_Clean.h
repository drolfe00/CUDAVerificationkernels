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
#ifndef FMRI_CLEAN_H_INCLUDED
#define FMRI_CLEAN_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>

// cuda dimensions
#define BLOCK_SETS    32
#define TILE_DIM    	32
#define POINT_THREADS 512

// functions called from fMRI_Main.cu
extern "C" int transposeNiftiDataPar(float* originalMatrix, float* transposedMatrix, int iDim, int jDim, float* runTime);
extern "C" int cleanPar(float* pointData, float* cleanedData, float* covTranspose, float* matrixInverse, int numCovariates, int numFiles, int niftiVolume, float* runTime);


//internal functions
void split(int iteration, int blockSet, float *values, int width, int height, float *subset, int subsetWidth);
void unsplit(int iteration, int blockSet, float *values, int width, int height, float *subset, int subsetWidth);

float sendTwist(dim3 blocks, dim3 threads, float *subset, int subsetSize, float *transpose);
__global__ void twist(float *original, float *transpose, int width, int height);

void sendBetas(int rowCount, int numFiles, int numCovariates, float *covTranspose, float *pointData, float *matrixInverse, float *betas);
__global__ void buildBetas(float *covTranspose, float *pointData, float *matrixInverse, int numCovariates, int numFiles, float *betas, float * xTransposeY);
__device__ void d_matrixMultiply(float *A, float *B, float *C, int hA, int wAhB, int wB);
__global__ void cleanPoint(float *betas, float *covTranspose, int numCovariates, float *pointData, int numFiles, float *cleaned);

#endif
