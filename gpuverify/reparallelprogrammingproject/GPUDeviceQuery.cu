 /****************************************************************************
 *  Keri Anderson
 *  CS6235  CUDA  
 *   
 *  GPU Device Query
 *
 *
 *
 *  This file contains functions to query the GPU device and return 
 *  specs on the device.
 *
 * 
 *  
 *****************************************************************************/

#include "GPUDeviceQuery.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>

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
}




// Beginning of GPU Architecture definitions
int convertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);
    return nGpuArchCoresPerSM[7].Cores;

}//end computeSMVerToCores


/***
 * This function queries and stores info about CUDA devices
 * and returns a pointer to a struct of CPU details.
 *
 *  similar to deviceQuery
 *
 */
extern "C" GPUDetails* queryCUDADevice(int verbose)
{
    GPUDetails* gpuDetails;
    int deviceCount = 0;
    gpuErrorCheck( cudaGetDeviceCount(&deviceCount) );

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0){
        printf("There are no available device(s) that support CUDA\n");
	return NULL;
    }
    else 
    {
	gpuDetails = (GPUDetails*)malloc(sizeof(GPUDetails));

        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	int dev;

	//iterate through the listed devices
	//for (dev = 0; dev < deviceCount; ++dev)
	for (dev = 0; dev < 1; ++dev) //hard code to one device for now
	{
	    gpuErrorCheck( cudaSetDevice(dev) );
	    cudaDeviceProp deviceProp;
	    gpuErrorCheck( cudaGetDeviceProperties(&deviceProp, dev) );

	    //load the data structure
	    gpuDetails->totalMultiProcessors        = deviceProp.multiProcessorCount;
	    gpuDetails->cudaCoresPerMP              = convertSMVer2Cores(deviceProp.major, deviceProp.minor);
	    gpuDetails->totalCudaCores              = gpuDetails->totalMultiProcessors * gpuDetails->cudaCoresPerMP;
            gpuDetails->globalMemoryInBytes         = (unsigned long long)deviceProp.totalGlobalMem;
	    gpuDetails->sharedMemoryPerBlockInBytes = deviceProp.sharedMemPerBlock;
	    gpuDetails->registersPerBlock           = deviceProp.regsPerBlock;
	    gpuDetails->constantMemoryInBytes       = deviceProp.totalConstMem;
	    gpuDetails->warpSize                    = deviceProp.warpSize;
	    gpuDetails->maxThreadsPerMP             = deviceProp.maxThreadsPerMultiProcessor;
	    gpuDetails->maxThreadsPerBlock          = deviceProp.maxThreadsPerBlock;
	    gpuDetails->maxXDimBlock                = deviceProp.maxThreadsDim[0];
	    gpuDetails->maxYDimBlock                = deviceProp.maxThreadsDim[1];
	    gpuDetails->maxZDimBlock                = deviceProp.maxThreadsDim[2];
	    gpuDetails->maxXDimGrid                 = deviceProp.maxGridSize[0];
	    gpuDetails->maxYDimGrid                 = deviceProp.maxGridSize[1];
	    gpuDetails->maxZDimGrid                 = deviceProp.maxGridSize[2];
    

	    if (verbose)
	    {
		printf("\nDevice %d: \"%s\"\n\n", dev, deviceProp.name); 
		printf("     Total Multi Processors  :         %d\n",   gpuDetails->totalMultiProcessors);
		printf("     Cuda Cores per MP       :         %d\n",   gpuDetails->cudaCoresPerMP);
		printf("     Total Cuda Cores        :         %d\n",   gpuDetails->totalCudaCores);
		printf("     Global Memory in Bytes  :         %llu\n", (unsigned long long)gpuDetails->globalMemoryInBytes);
		printf("     Shared Memory Per Block :         %d\n",   gpuDetails->sharedMemoryPerBlockInBytes);
		printf("     Registers Per Block     :         %d\n",   gpuDetails->registersPerBlock);
		printf("     Total Constant Memory   :         %d\n",   gpuDetails->constantMemoryInBytes);
		printf("     Warp Size               :         %d\n",   gpuDetails->warpSize);
		printf("     Max Threads per MP      :         %d\n",   gpuDetails->maxThreadsPerMP);
		printf("     Max Threads per Block   :         %d\n",   gpuDetails->maxThreadsPerBlock);
		printf("     Max X Dim Block         :         %d\n",   gpuDetails->maxXDimBlock);
		printf("     Max Y Dim Block         :         %d\n",   gpuDetails->maxYDimBlock);
		printf("     Max Z Dim Block         :         %d\n",   gpuDetails->maxZDimBlock);
		printf("     Max X Dim Grid          :         %d\n",   gpuDetails->maxXDimGrid);
		printf("     Max Y Dim Grid          :         %d\n",   gpuDetails->maxYDimGrid);
		printf("     Max Z Dim Grid          :         %d\n",   gpuDetails->maxZDimGrid);

		printf("\n\n\n\n");
 
	    }//end verbose
	    
	}//end for dev = 0...
    }//end else

    return gpuDetails;

}//end queryCUDADevice

