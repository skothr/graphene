#ifndef CUDA_TOOLS_H
#define CUDA_TOOLS_H

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>

// CUDA initialization
extern bool gCudaInitialized;
extern int  gDevId;
bool initCudaDevice(int devId=-1);


#endif // CUDA_TOOLS_H
