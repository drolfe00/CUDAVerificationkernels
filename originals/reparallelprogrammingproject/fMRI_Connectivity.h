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
#ifndef FMRI_CONNECTIVITY_H_INCLUDED
#define FMRI_CONNECTIVITY_H_INCLUDED


// functions
extern "C" int normalizeDataPar(float* cleanedData, float* normalizedData, int numFiles, int niftiVolume, float* runTime);
extern "C" int connectivityPar(int seed, float* normalizedData, float* connectivityData, int numFiles, int niftiVolume, float* runTime);

#endif
