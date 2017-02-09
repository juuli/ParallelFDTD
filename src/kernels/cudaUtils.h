#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include "../logger.h"
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

#define IGNORE_CHECKSUMS 1

// Struct which contains the device properties
typedef struct CudaDeviceSpecs_t {
  unsigned int compute_capability;
  unsigned int cores_per_sm;

} CudaDeviceSpecs;



//////////////////
// Additional

// C wrapper for the logger

extern "C" {
  void c_log_msg(log_level level, const char* msg, ...);
}

/// \return the best GPU (with maximum GFLOPS)
int gpuGetMaxGflopsDeviceId();

/// \return the best GPU (with maximum free memory)
int gpuGetMaxFreeMemoryDeviceId();

/// \param major The major revision number of the Cuda device
/// \param minor The minor revision number of the Cuda device
/// \return The number of cores in the device
int _ConvertSMVer2Cores(int major, int minor); 

/// \brief Cuda error handling function
/// \param error The cuda error returned by a cuda function
/// \param message user defined message to specify the context of the error
inline void cudasafe( cudaError_t error, const char* message) {
  const char * errorStr = cudaGetErrorString(error);
  if(error!=cudaSuccess) { c_log_msg(LOG_ERROR,"ERROR: %s : %s\n", message, errorStr); throw(-1);
  }
};

// Forward declaration
class Node;

/////////////
// Allocation helpers

template < typename T >
T* toDevice(unsigned int mem_size, unsigned int device)
{
  T* h_P = (T*)calloc(mem_size, sizeof(T));
  for(unsigned int i = 0; i <mem_size; i++) {
    h_P[i] = (T)0;
  }
  cudasafe(cudaSetDevice(device), "T ToDevice-zero: cudaSetDevice");
  T* P;
  cudasafe(cudaMalloc((void**)&P, mem_size*sizeof(T)), "T to device: Malloc");
  cudasafe(cudaMemcpy((void*)P, h_P,  mem_size*sizeof(T), cudaMemcpyHostToDevice), "Memcopy");
  
  //printCheckSum(P, mem_size, "floats initialize to 0: ");
  
  free(h_P);
  return P;
};

template < typename T >
T* toDevice(unsigned int mem_size, const T* h_data, unsigned int device)
{
  c_log_msg(LOG_DEBUG, "cudaUtils.cu: T ToDevice(data) - mem_size %u, device %d", mem_size, device);
  cudasafe(cudaSetDevice(device), "T ToDevice: cudaSetDevice");
  T* d_data;
  cudasafe(cudaMalloc((void**)&d_data, mem_size*sizeof(T)), "T to device: Malloc");
  cudasafe(cudaMemcpy((void*)d_data, h_data,  mem_size*sizeof(T), cudaMemcpyHostToDevice), "Memcopy");
  
  return d_data;
};

// This is defined to avoid mixup with the overloading
template < typename T >
T* valueToDevice(unsigned int mem_size, T val, unsigned int device)
{
  T* h_P = (T*)calloc(mem_size, sizeof(T));
  c_log_msg(LOG_DEBUG, "cudaUtils.cu: t ToDevice - mem_size %u, init val: %d, device %d", mem_size, val, device);

  for(unsigned int i = 0; i < mem_size; i++) {
    h_P[i] = val;
  }
  cudasafe(cudaSetDevice(device), "unsignedToDevice-zero: cudaSetDevice");
  T* P;
  cudasafe(cudaMalloc((void**) &P, mem_size*sizeof(T)), "unsigned char to device: Malloc");
  cudasafe(cudaMemcpy(P, h_P,  mem_size*sizeof(T), cudaMemcpyHostToDevice), "Memcopy");
  
  free(h_P);
  return P;
};

template < typename T >
T* fromDevice(unsigned int mem_size, const T* d_data, unsigned int device)
{
  T* h_data = (T*)calloc(mem_size, sizeof(T));

  c_log_msg(LOG_DEBUG, "cudaUtils.cu: T fromDevice(data) - mem_size %u, device %d", mem_size, device);
  cudasafe(cudaSetDevice(device), "T fromDevice: cudaSetDevice");
  cudasafe(cudaMemcpy(h_data, d_data,  mem_size*sizeof(T), cudaMemcpyDeviceToHost), "T fromDevice: Memcopy");
   
  return h_data;
};

template < typename T >
void resetData(unsigned int mem_size, T* d_data, unsigned int device);

template <typename T>
T getSample(unsigned int element_idx, T* P) {
  T sample = (T)0;
  cudasafe(cudaMemcpy(&sample, &P[element_idx], sizeof(T), cudaMemcpyDeviceToHost), "getSample : - Memcopy");
  return sample;
}

// Copy helper
template < typename T>
void copyHostToDevice(unsigned int mem_size, T* d_dest, T* h_data, unsigned int device) {
    cudasafe(cudaSetDevice(device), " copyHostToDevice: cudaSetDevice");
    c_log_msg(LOG_DEBUG, "cudaUtils.cu: T copyHostToDevice - mem_size %u, device %u", mem_size, device);
    cudasafe(cudaMemcpy(d_dest, h_data,  mem_size*sizeof(T), cudaMemcpyHostToDevice), "Memcopy");
}

// Same goes with this
template < typename T>
void copyDeviceToHost(unsigned int mem_size, T* h_dest, T* d_src, unsigned int device) {
    cudasafe(cudaSetDevice(device), " copyDeviceToHost: cudaSetDevice");
    c_log_msg(LOG_DEBUG, "cudaUtils.cu: T copyDeviceToHost - mem_size %u, device %u", mem_size, device);
    cudasafe(cudaMemcpy(h_dest, d_src,  mem_size*sizeof(T), cudaMemcpyDeviceToHost), "Memcopy");
}

int getCurrentDevice();

void printMemInfo(const char* message, int device);

/////////////////////
// Checkers

float printCheckSum(float* d_data, size_t mem_size, char* message);

void printCheckSum(unsigned char* d_data, size_t mem_size, char* message);

void printMax(float* d_data, size_t mem_size, char* message);

/////////////////////
/// Destroy helpers

template <typename T>
void destroyMem(T* d_data) {
  cudasafe(cudaFree(d_data), "Destroy memory (T)");
};

template <typename T>
void destroyMem(T* d_data, unsigned int device) {
  cudasafe(cudaSetDevice(device), "destroyMem: cudaSetDevice");
  cudasafe(cudaFree(d_data), "Destroy memory (T)");
};

template <typename T>
__global__ void resetKernel(T* d_data, unsigned int mem_size); 


#endif
