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
#ifndef FMRI_COVARIATE_H_INCLUDED
#define FMRI_COVARIATE_H_INCLUDED


// functions
extern "C" int covariatesTransCovariatesPar(float* covariates, float* covTranspose, float* covTransXCov, int numCovariates, int numFiles, float* runTime);

#endif
