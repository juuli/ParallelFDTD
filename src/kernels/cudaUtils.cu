#include "cudaUtils.h"

#include <stdio.h>
#include <stdarg.h>

extern "C" {
void c_log_msg(log_level level, const char* msg, ...) {
  wchar_t wbuffer[512];
  char buffer[512];

  va_list argptr;
  va_start(argptr, msg);
  vsnprintf(buffer, 512, msg, argptr);
  mbstowcs (wbuffer, buffer, 512);
  va_end(argptr);

  Logger c_logger(level, wbuffer);
  }
}

int getCurrentDevice() {
  int dev_num;
  cudasafe(cudaGetDevice(&dev_num), "cudaGetDevice");
  return dev_num;
}

void printMemInfo(const char* message, int device) {
  size_t total_mem = 0;
  size_t free_mem = 0;
  int current_device = getCurrentDevice();

  cudaSetDevice(device);
  cudasafe(cudaMemGetInfo (&free_mem, &total_mem), "Cuda meminfo");

  c_log_msg(LOG_DEBUG, "%s - device %u, mem_size %u MB, free %u MB",message, device, total_mem/1000000, free_mem/1000000);

  cudaSetDevice(current_device);
}

/////////////////////
// Checkkers

float printCheckSum(float* d_data, size_t mem_size, char* message) {
  float sum = 0.f;
  if(!IGNORE_CHECKSUMS) {
    float* h_data = (float*)calloc(mem_size, sizeof(float));
    sum = 0.f;

    cudasafe(cudaMemcpy(h_data, d_data, mem_size*sizeof(float), cudaMemcpyDeviceToHost), "Memcopy");

    for(size_t i = 0; i < mem_size; i++)
      sum += h_data[i];

    c_log_msg(LOG_INFO, "kernels3d.cu: printChecksum float - %s checksum: %f", message, sum);

    free(h_data);
  }

  return sum;
}

void printCheckSum(unsigned char* d_data, size_t mem_size, char* message) {
  if(!IGNORE_CHECKSUMS) {
    unsigned char* h_data = (unsigned char*)calloc(mem_size, sizeof(unsigned char));
    unsigned int sum = 0;

    cudasafe(cudaMemcpy(h_data, d_data, mem_size*sizeof(unsigned char), cudaMemcpyDeviceToHost), "Memcopy");

    for(size_t i = 0; i < mem_size; i++)
      sum += (unsigned int)h_data[i];

    c_log_msg(LOG_INFO, "kernels3d.cu: printChecksum unsigned char - %s checksum: %u", message, sum);

    free(h_data);
  }
}

void printMax(float* d_data, size_t mem_size, char* message) {
  if(IGNORE_CHECKSUMS) {
    float* h_data = (float*)calloc(mem_size, sizeof(float));
    float max_val = -999999999999.f;

    cudasafe(cudaMemcpy(h_data, d_data, mem_size*sizeof(float), cudaMemcpyDeviceToHost), "Memcopy");

    for(size_t i = 0; i < mem_size; i++){
      if(h_data[i] > max_val)
        max_val = h_data[i];
    }

    c_log_msg(LOG_INFO, "kernels3d.cu: printMax - %s Maximum value: %f", message, max_val);

    free(h_data);
  }
}

template <>
void resetData(unsigned int mem_size, float* d_data, unsigned int device) {
  c_log_msg(LOG_DEBUG, "cudaUtils.cu: resetFloats(data) - mem_size %u, device %d", mem_size, device);
  cudasafe(cudaSetDevice(device), "floatsToDevice: cudaSetDevice");

  dim3 block(128);
  dim3 grid(mem_size/block.x+1);
  resetKernel< float ><<<grid, block>>>(d_data, mem_size);
}

template <>
void resetData(unsigned int mem_size, double* d_data, unsigned int device) {
  c_log_msg(LOG_DEBUG, "cudaUtils.cu: resetDouble(data) - mem_size %u, device %d", mem_size, device);

  dim3 block(128);
  dim3 grid(mem_size/block.x+1);
  resetKernel< double ><<<grid, block>>>(d_data, mem_size);
}

template <typename T>
__global__ void resetKernel(T* d_data, unsigned int mem_size) {
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  if(idx < mem_size)
    d_data[idx] = (T)0;
}

// This function returns the best GPU (with maximum GFLOPS)
// Taken from helpers_CUDA h file
int gpuGetMaxGflopsDeviceId() {
  int current_device     = 0, sm_per_multiproc  = 0;
  int max_compute_perf   = 0, max_perf_device   = 0;
  int device_count       = 0, best_SM_arch      = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceCount(&device_count);

  // Find the best major SM Architecture GPU device
  while (current_device < device_count)
  {
      cudaGetDeviceProperties(&deviceProp, current_device);
      // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
      if (deviceProp.computeMode != cudaComputeModeProhibited) {
          if (deviceProp.major > 0 && deviceProp.major < 9999) {
              best_SM_arch = MAX(best_SM_arch, deviceProp.major);
          }
      }
      current_device++;
  }

  // Find the best CUDA capable GPU device
  current_device = 0;

  while (current_device < device_count)
  {
      cudaGetDeviceProperties(&deviceProp, current_device);

      // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
      if (deviceProp.computeMode != cudaComputeModeProhibited) {
          if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
              sm_per_multiproc = 1;
          }
          else {
              sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
          }

          int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;

          if (compute_perf  > max_compute_perf) {
              // If we find GPU with SM major > 2, search only these
              if (best_SM_arch > 2) {
                  // If our device==dest_SM_arch, choose this, or else pass
                  if (deviceProp.major == best_SM_arch) {
                      max_compute_perf  = compute_perf;
                      max_perf_device   = current_device;
                  }
              }
              else {
                  max_compute_perf  = compute_perf;
                  max_perf_device   = current_device;
              }
          }
      }
      ++current_device;
  }
  return max_perf_device;
}

// This function returns the best GPU (with maximum FREE MEM)
int gpuGetMaxFreeMemoryDeviceId(){
	int device_count       = 0, best_FREE_mem_dev_id    = 0;
	size_t free_mem 	   = 0, total_mem 		  		= 0, free_mem_max 		  = 0;
	cudaGetDeviceCount(&device_count);
	if (device_count <= 0)
		return -1;
	cudasafe(cudaSetDevice(0), "cudaSetDevice");
	cudasafe(cudaMemGetInfo(&free_mem, &total_mem), "Cuda meminfo");
	free_mem_max = free_mem;
	best_FREE_mem_dev_id = 0;
	for (unsigned int i = 1; i < device_count; i++) {
		cudasafe(cudaSetDevice(i), "cudaSetDevice");
		cudasafe(cudaMemGetInfo(&free_mem, &total_mem), "Cuda meminfo");
		if (free_mem>free_mem_max){
			free_mem_max = free_mem;
			best_FREE_mem_dev_id = i;
		}
	}
	return best_FREE_mem_dev_id;
};

// Beginning of GPU Architecture definitions
// Taken from helpers_CUDA h file
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
        { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
        { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
        { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
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
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}
