 /****************************************************************************
 *  Keri Anderson
 *  CS6235  CUDA  
 *   
 *  GPU Device Query.h 
 *
 *
 *
 *  This file contains functions to query the GPU device and return 
 *  specs on the device.
 *
 * 
 *  
 ***************************************************************************/
#ifndef DEVICEQUERY_H_INCLUDED
#define DEVICEQUERY_H_INCLUDED

//struct to hold details about the cuda device
//  note- structs declared outside of main will be on the heap
typedef struct {     // using typedef allows declaration of 'GPUDetails' without 'struct' threadArguments
    int totalMultiProcessors;
    int cudaCoresPerMP;
    int totalCudaCores;
    long globalMemoryInBytes;     
    int sharedMemoryPerBlockInBytes;  
    int registersPerBlock;
    int constantMemoryInBytes;  
    int warpSize;
    int maxThreadsPerMP;
    int maxThreadsPerBlock;
    int maxXDimBlock;
    int maxYDimBlock;
    int maxZDimBlock;
    int maxXDimGrid;
    int maxYDimGrid;
    int maxZDimGrid;
}GPUDetails;


// functions
extern "C" GPUDetails* queryCUDADevice(int verbose);

#endif
