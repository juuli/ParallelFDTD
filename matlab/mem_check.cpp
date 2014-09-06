#include "mex.h"
#include "mex_includes.h"
#include <signal.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int number_of_devices = 0;
    
    cudaGetDeviceCount(&number_of_devices);
    std::vector<unsigned int> mem(number_of_devices, 0);
    plhs[0] = mxCreateNumericMatrix(number_of_devices, 1, mxDOUBLE_CLASS, mxREAL);
    double* ret_ptr = (double*)mxGetData(plhs[0]);
    
    for(int i = 0; i < number_of_devices; i++) {
        cudaSetDevice(i);
        size_t free_mem = 0;
        size_t total_mem = 0;
        cudasafe(cudaMemGetInfo (&free_mem, &total_mem), "Cuda meminfo");
        mexPrintf("Memory on device %u, %u MB\n",i,(unsigned int)(free_mem/1e6f));
        ret_ptr[i] = (double)free_mem;
  }

  return;
}