#include "mex.h"
#include "mex_includes.h"
#include <signal.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int number_of_devices = 0;
	
    cudasafe(cudaGetDeviceCount(&number_of_devices), "getDeviceCount");
    mexPrintf("Number of Cuda Devices: %u\n", number_of_devices);
    for(int i = 0; i < number_of_devices; i++) {
        cudaSetDevice(i);
		cudaDeviceReset();
        mexPrintf("Reset Cuda Device %u\n",i);
    }
        
    return;
}