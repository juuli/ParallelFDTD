///////////////////////////////////////////////////////////////////////////////
//
// This file is a part of the PadallelFDTD Finite-Difference Time-Domain
// simulation library. It is released under the MIT License. You should have
// received a copy of the MIT License along with ParallelFDTD.  If not, see
// http://www.opensource.org/licenses/mit-license.php
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// For details, see the LICENSE file
//
// (C) 2013-2014 Jukka Saarelma
// Aalto University School of Science
//
///////////////////////////////////////////////////////////////////////////////

#include "../global_includes.h"
#include "kernels3d.h"
#include "cudaUtils.h"

#include <math.h>
#include <stdio.h>
#include <stdarg.h>

float launchFDTD3d(CudaMesh* d_mesh,
                   SimulationParameters* sp,
                   float* h_return_ptr,
                   bool (*interruptCallback)(void),
                   void (*progressCallback)(int, int, float)) {
  clock_t start_t;
  clock_t end_t;
  start_t = clock();

  c_log_msg(LOG_INFO, "launchFDTD3d - begin");

  dim3 block(d_mesh->getBlockX(), d_mesh->getBlockY(), 1);
  dim3 grid(d_mesh->getGridDimX(),
            d_mesh->getGridDimY(),
            d_mesh->getPartitionSize()-1);

  c_log_msg(LOG_DEBUG, "kernels3d.cu: launchFDTD3d - Mesh dim: x %d y %d z %d",
            d_mesh->getDimX(), d_mesh->getDimY(), d_mesh->getDimZ());
  c_log_msg(LOG_DEBUG, "kernels3d.cu: launchFDTD3d - Block dim: x %d y %d z %d",
            block.x, block.y, block.z);
  c_log_msg(LOG_DEBUG, "kernels3d.cu: launchFDTD3d - Grid dim: x %u, y %u z %u",
            grid.x, grid.y, grid.z);

  c_log_msg(LOG_DEBUG, "kernels3d.cu: launchFDTD3d - Number of partitions: %u", d_mesh->getNumberOfPartitions());
  c_log_msg(LOG_INFO, "kernels3d.cu: launchFDTD3d - Number of steps: %u", sp->getNumSteps());


  /////////////////////////////////////////////////////////////////////////////
  // Allocate return data on the device

  std::vector< std::pair <float*, std::pair<mesh_size_t, int> > > d_receiver_data;
  for(unsigned int i = 0; i < sp->getNumReceivers(); i++) {
    nv::Vec3i pos = sp->getReceiverElementCoordinates(i);
    mesh_size_t element_idx;
    int device_idx;
    d_mesh->getElementIdxAndDevice(pos.x, pos.y, pos.z, &device_idx, &element_idx);
    float* d_return_ptr = (float*)NULL;
    if(device_idx != -1)
      d_return_ptr = toDevice<float>(sp->getNumSteps(), d_mesh->getDeviceAt(device_idx));
    std::pair<mesh_size_t, int> temp_i(element_idx, device_idx);
    std::pair<float*, std::pair<mesh_size_t,int> > temp_d(d_return_ptr, temp_i);
    d_receiver_data.push_back(temp_d);
  }

  unsigned int step;
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  ///////// Step loop
  for(step = 0; step < sp->getNumSteps(); step++)
  {
    clock_t step_start, step_end;
    step_start = clock();
    if(interruptCallback()) {
      c_log_msg(LOG_INFO, "kernels3d.cu: launchFDTD3d interrupted at step %d", step);
      break;
    }

    ///////// Source Loop

    for(unsigned int i = 0; i < sp->getNumSources(); i++) {
      float sample = sp->getSourceSample(i, step);
      nv::Vec3i pos = sp->getSourceElementCoordinates(i);

      if(sp->getSource(i).getSourceType() == SRC_HARD) {
        d_mesh->setSample<float>(sample, pos.x, pos.y, pos.z);
      }
      else {
        d_mesh->addSample<float>(sample, pos.x, pos.y, pos.z);
      }
    }

    cudasafe(cudaDeviceSynchronize(), "kernels3d.cu: launchFDTD3d - cudaDeviceSynchronize after Source insert");

    /////// Receiver loop
    for(unsigned int i = 0; i < sp->getNumReceivers(); i++) {
      nv::Vec3i pos = sp->getReceiverElementCoordinates(i);
      int d_idx = d_receiver_data.at(i).second.second;
      mesh_size_t e_idx = d_receiver_data.at(i).second.first;
      if(d_receiver_data.at(i).second.second == -1) continue;
      float* dest = (d_receiver_data.at(i).first)+step;
      float* src = d_mesh->getPressurePtrAt(d_idx)+e_idx;
      cudaSetDevice(d_mesh->getDeviceAt(d_idx));
      cudasafe(cudaMemcpy(dest, src,  sizeof(float), cudaMemcpyDeviceToDevice), "Memcopy");
    } // End receiver Loop

    ////// FDTD partition loop
    for(unsigned int i = 0; i < d_mesh->getNumberOfPartitions(); i++) {
      cudasafe(cudaSetDevice(d_mesh->getDeviceAt(i)), "kernels3d.cu: launchFDTD3d - set device");

      grid.z = d_mesh->getPartitionSize(i)-1;

      if(sp->getUpdateType() == SRL_FORWARD) {
      fdtd3dStdMaterials<float><<<grid, block>>>(d_mesh->getPositionIdxPtrAt(i),
                                                 d_mesh->getMaterialIdxPtrAt(i),
                                                 d_mesh->getPressurePtrAt(i),
                                                 d_mesh->getPastPressurePtrAt(i),
                                                 d_mesh->getParameterPtrAt(i),
                                                 d_mesh->getMaterialPtrAt(i),
                                                 d_mesh->getDimXY(),
                                                 d_mesh->getDimX());
      }


      if(sp->getUpdateType() == SRL) {
      fdtd3dStdKowalczykMaterials<float><<<grid, block>>>(d_mesh->getPositionIdxPtrAt(i),
                                                          d_mesh->getMaterialIdxPtrAt(i),
                                                          d_mesh->getPressurePtrAt(i),
                                                          d_mesh->getPastPressurePtrAt(i),
                                                          d_mesh->getParameterPtrAt(i),
                                                          d_mesh->getMaterialPtrAt(i),
                                                          d_mesh->getDimXY(),
                                                          d_mesh->getDimX());
      }


      if(sp->getUpdateType() == SHARED) {
      block.x = 32; block.y = 4; block.z = 1;
      mesh_size_t dim_z = grid.z;
      grid.z = 1; // the z-dimension is gone through in the kernel
      fdtd3dSliced<float,32,4><<<grid, block>>>(d_mesh->getPositionIdxPtrAt(i),
                                                d_mesh->getMaterialIdxPtrAt(i),
                                                d_mesh->getPressurePtrAt(i),
                                                d_mesh->getPastPressurePtrAt(i),
                                                d_mesh->getParameterPtrAt(i),
                                                d_mesh->getMaterialPtrAt(i),
                                                d_mesh->getDimXY(),
                                                d_mesh->getDimX(),
                                                dim_z);
      }
    } // End FDTD partition loop

    cudasafe(cudaDeviceSynchronize(), "kernels3d.cu: launchFDTD3d - cudaDeviceSynchronize after FDTD");
    cudasafe(cudaPeekAtLastError(), "kernels3d.cu: launchFDTD3d - Peek after launch");

    ////// TODO boundary loop

    d_mesh->flipPressurePointers();
    d_mesh->switchHalos();

    cudasafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize after step");
    if((step%PROGRESS_MOD) == 0) {
      step_end = clock()-step_start;
      progressCallback(step, sp->getNumSteps(), ((((float)step_end/CLOCKS_PER_SEC))));
    }

  }// End step loop

  /////// Copy device return data to host
  for(unsigned int i = 0; i < sp->getNumReceivers(); i++) {
    float* dest = h_return_ptr+i*sp->getNumSteps();
    float* src = d_receiver_data.at(i).first;
    if(d_receiver_data.at(i).second.second == -1) continue;
    int dev = d_mesh->getDeviceAt(d_receiver_data.at(i).second.second);
    copyDeviceToHost(sp->getNumSteps(), dest, src, dev);
    destroyMem(d_receiver_data.at(i).first, dev);
  } // End receiver Loop


  cudasafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize before return");

  end_t = clock()-start_t;
  step++; // add the first step
  c_log_msg(LOG_INFO, "LaunchFDTD3d - time: %f seconds, per step: %f",
            ((float)end_t/CLOCKS_PER_SEC), (((float)end_t/CLOCKS_PER_SEC)/step));

  c_log_msg(LOG_DEBUG, "LaunchFDTD3d return");
  return (((float)end_t/CLOCKS_PER_SEC)/step);
}

float launchFDTD3dDouble(CudaMesh* d_mesh,
                         SimulationParameters* sp,
                         double* h_return_ptr,
                         bool (*interruptCallback)(void),
                         void (*progressCallback)(int, int, float)) {
  clock_t start_t;
  clock_t end_t;
  start_t = clock();
  c_log_msg(LOG_INFO, "launchFDTD3dDouble - begin");

  dim3 block(d_mesh->getBlockX(), d_mesh->getBlockY(), 1);
  dim3 grid(d_mesh->getGridDimX(),
            d_mesh->getGridDimY(),
            d_mesh->getPartitionSize());

  c_log_msg(LOG_DEBUG, "kernels3d.cu: launchFDTD3dDouble - Mesh dim: x %d y %d z %d",
            d_mesh->getDimX(), d_mesh->getDimY(), d_mesh->getDimZ());
  c_log_msg(LOG_DEBUG, "kernels3d.cu: launchFDTD3dDouble - Block dim: x %d y %d z %d",
            block.x, block.y, block.z);
  c_log_msg(LOG_DEBUG, "kernels3d.cu: launchFDTD3dDouble - Grid dim: x %u, y %u z %u",
            grid.x, grid.y, grid.z);

  c_log_msg(LOG_DEBUG, "kernels3d.cu: launchFDTD3dDouble - Number of partitions: %u", d_mesh->getNumberOfPartitions());
  c_log_msg(LOG_INFO, "kernels3d.cu: launchFDTD3dDouble  - Number of steps: %u", sp->getNumSteps());


  /////////////////////////////////////////////////////////////////////////////
  // Allocate return data on the device

  std::vector< std::pair <double*, std::pair<mesh_size_t, int> > > d_receiver_data;
  for(unsigned int i = 0; i < sp->getNumReceivers(); i++) {
    c_log_msg(LOG_INFO, "kernel3d.cu: launchFDTD3dDouble - allocating receiver %u", i);
    nv::Vec3i pos = sp->getReceiverElementCoordinates(i);
    mesh_size_t element_idx;
    int device_idx;
    d_mesh->getElementIdxAndDevice(pos.x, pos.y, pos.z, &device_idx, &element_idx);
    double* d_return_ptr = (double*)NULL;
    if(device_idx != -1)
      d_return_ptr = toDevice<double>(sp->getNumSteps(), d_mesh->getDeviceAt(device_idx));
    std::pair<mesh_size_t, int> temp_i(element_idx, device_idx);
    std::pair<double*, std::pair<mesh_size_t,int> > temp_d(d_return_ptr, temp_i);
    d_receiver_data.push_back(temp_d);
  }
  c_log_msg(LOG_INFO, "kernel3d.cu: launchFDTD3dDouble - after recevier allocation");

  unsigned int step;
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  ///////// Step loop
  for(step = 0; step < sp->getNumSteps(); step++) {
    clock_t step_start, step_end;
    step_start = clock();
    if(interruptCallback()) {
      c_log_msg(LOG_INFO, "kernels3d.cu: launchFDTD3dDouble interrupted at step %d" ,step);
      break;
    }

    ///////// Source Loop
    for(unsigned int i = 0; i < sp->getNumSources(); i++) {
      double sample = sp->getSourceSampleDouble(i, step);
      nv::Vec3i pos = sp->getSourceElementCoordinates(i);
      if(sp->getSource(i).getSourceType() == SRC_HARD) {
        d_mesh->setSample<double>(sample, pos.x, pos.y, pos.z);
      }
      else {
        d_mesh->addSample<double>(sample, pos.x, pos.y, pos.z);
      }
    }

    cudasafe(cudaDeviceSynchronize(), "kernels3d.cu: launchFDTD3dDouble - cudaDeviceSynchronize after Source insert");

    /////// Receiver loop
    for(unsigned int i = 0; i < sp->getNumReceivers(); i++) {
      nv::Vec3i pos = sp->getReceiverElementCoordinates(i);
      int d_idx = d_receiver_data.at(i).second.second;
      mesh_size_t e_idx = d_receiver_data.at(i).second.first;
      if(d_idx == -1) continue;
      double* dest = (d_receiver_data.at(i).first)+step;
      double* src = d_mesh->getPressureDoublePtrAt(d_idx)+e_idx;
      cudaSetDevice(d_mesh->getDeviceAt(d_idx));
      cudasafe(cudaMemcpy(dest, src,  sizeof(double), cudaMemcpyDeviceToDevice), "Memcopy");
    } // End receiver Loop

    ////// FDTD partition loop
    for(unsigned int i = 0; i < d_mesh->getNumberOfPartitions(); i++) {
      cudasafe(cudaSetDevice(d_mesh->getDeviceAt(i)), "kernels3d.cu: launchFDTD3dDouble - set device");

      grid.z = d_mesh->getPartitionSize(i)-1;

      if(sp->getUpdateType() == SRL_FORWARD)
      fdtd3dStdMaterials<double><<<grid, block>>>(d_mesh->getPositionIdxPtrAt(i),
                                                  d_mesh->getMaterialIdxPtrAt(i),
                                                  d_mesh->getPressureDoublePtrAt(i),
                                                  d_mesh->getPastPressureDoublePtrAt(i),
                                                  d_mesh->getParameterPtrDoubleAt(i),
                                                  d_mesh->getMaterialPtrDoubleAt(i),
                                                  d_mesh->getDimXY(),
                                                  d_mesh->getDimX());



      if(sp->getUpdateType() == SRL)
      fdtd3dStdKowalczykMaterials<double><<<grid, block>>>(d_mesh->getPositionIdxPtrAt(i),
                                                           d_mesh->getMaterialIdxPtrAt(i),
                                                           d_mesh->getPressureDoublePtrAt(i),
                                                           d_mesh->getPastPressureDoublePtrAt(i),
                                                           d_mesh->getParameterPtrDoubleAt(i),
                                                           d_mesh->getMaterialPtrDoubleAt(i),
                                                           d_mesh->getDimXY(),
                                                           d_mesh->getDimX());


      if(sp->getUpdateType() == SHARED) {
      block.x = 32; block.y = 4; block.z = 1;
      mesh_size_t dim_z = grid.z;
      grid.z = 1; // the z-dimension is gone through in the kernel
      fdtd3dSliced<double, 32,4><<<grid, block>>>(d_mesh->getPositionIdxPtrAt(i),
                                                  d_mesh->getMaterialIdxPtrAt(i),
                                                  d_mesh->getPressureDoublePtrAt(i),
                                                  d_mesh->getPastPressureDoublePtrAt(i),
                                                  d_mesh->getParameterPtrDoubleAt(i),
                                                  d_mesh->getMaterialPtrDoubleAt(i),
                                                  d_mesh->getDimXY(),
                                                  d_mesh->getDimX(),
                                                  dim_z);
      }


    } // End FDTD partition loop

    cudasafe(cudaDeviceSynchronize(), "kernels3d.cu: launchFDTD3dDouble - cudaDeviceSynchronize after FDTD");
    cudasafe(cudaPeekAtLastError(), "kernels3d.cu: launchFDTD3dDouble - Peek after launch");

    ////// TODO boundary loop

    d_mesh->flipPressurePointers();
    d_mesh->switchHalos();

    cudasafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize after step");

    if((step%PROGRESS_MOD) == 0) {
      step_end = clock()-step_start;
      progressCallback(step, sp->getNumSteps(), ((((float)step_end/CLOCKS_PER_SEC))));
    }
  }// End step loop

   /////// Copy device return data to host
  for(unsigned int i = 0; i < sp->getNumReceivers(); i++) {
    double* dest = h_return_ptr+i*sp->getNumSteps();
    double* src = d_receiver_data.at(i).first;
    if(d_receiver_data.at(i).second.second == -1) continue;
    int dev = d_mesh->getDeviceAt(d_receiver_data.at(i).second.second);
    copyDeviceToHost(sp->getNumSteps(), dest, src, dev);
    destroyMem(d_receiver_data.at(i).first, dev);
  } // End receiver Loop


  cudasafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize before return");

  end_t = clock()-start_t;
  step++;

  c_log_msg(LOG_INFO, "LaunchFDTD3dDouble - time: %f seconds, per step: %f",
            ((float)end_t/CLOCKS_PER_SEC), (((float)end_t/CLOCKS_PER_SEC)));

  c_log_msg(LOG_DEBUG, "LaunchFDTD3dDouble return");

  return (((float)end_t/CLOCKS_PER_SEC)/step);
}

void launchFDTD3dStep(CudaMesh* d_mesh,
                      SimulationParameters* sp,
                      float* h_return_ptr,
                      unsigned int step,
                      int step_direction,
                      void (*progressCallback)(int, int, float)) {

    clock_t step_start, step_end;
    step_start = clock();

    static int past_step_directon = 1;
    dim3 block(d_mesh->getBlockX(), d_mesh->getBlockY(), 1);
    dim3 grid(d_mesh->getGridDimX(),
              d_mesh->getGridDimY(),
              d_mesh->getPartitionSize()-1);

    c_log_msg(LOG_VERBOSE, "kernels3d.cu: launchFDTD3dStep - Mesh dim: x %d y %d z %d",
              d_mesh->getDimX(), d_mesh->getDimY(), d_mesh->getDimZ());
    c_log_msg(LOG_VERBOSE, "kernels3d.cu: launchFDTD3dStep - Block dim: x %d y %d z %d",
              block.x, block.y, block.z);
    c_log_msg(LOG_VERBOSE, "kernels3d.cu: launchFDTD3dStep - Grid dim: x %u, y %u z %u",
              grid.x, grid.y, grid.z);

    ///////// Source Loop
    for(unsigned int i = 0; i < sp->getNumSources(); i++) {
      float sample = sp->getSourceSample(i, step);
      nv::Vec3i pos = sp->getSourceElementCoordinates(i);
      if(sp->getSource(i).getSourceType() == SRC_HARD) {
        d_mesh->setSample<float>(sample, pos.x, pos.y, pos.z);
      }
      else {
        d_mesh->addSample<float>(sample, pos.x, pos.y, pos.z);
      }
    } // End Source loop

    /////// Receiver loop
    if(h_return_ptr){
      for(unsigned int i = 0; i < sp->getNumReceivers(); i++) {
        nv::Vec3i pos = sp->getReceiverElementCoordinates(i);
        h_return_ptr[i*sp->getNumSteps()+step] = d_mesh->getSample<float>(pos.x, pos.y, pos.z);
      } // End receiver Loop

    }

    ////// FDTD update loop
    for(unsigned int i = 0; i < d_mesh->getNumberOfPartitions(); i++) {
      cudasafe(cudaSetDevice(d_mesh->getDeviceAt(i)), "kernels3d.cu: launchFDTD3d - set device");
      grid.z = d_mesh->getPartitionSize(i);
      grid.z = grid.z-1;

      if(sp->getUpdateType() == SRL_FORWARD) {
      fdtd3dStdMaterials<float><<<grid, block>>> (d_mesh->getPositionIdxPtrAt(i),
                                                  d_mesh->getMaterialIdxPtrAt(i),
                                                  d_mesh->getPressurePtrAt(i),
                                                  d_mesh->getPastPressurePtrAt(i),
                                                  d_mesh->getParameterPtrAt(i),
                                                  d_mesh->getMaterialPtrAt(i),
                                                  d_mesh->getDimXY(),
                                                  d_mesh->getDimX());
      }

      if(sp->getUpdateType() == SRL) {
      fdtd3dStdKowalczykMaterials<float><<<grid, block>>> (d_mesh->getPositionIdxPtrAt(i),
                                                           d_mesh->getMaterialIdxPtrAt(i),
                                                           d_mesh->getPressurePtrAt(i),
                                                           d_mesh->getPastPressurePtrAt(i),
                                                           d_mesh->getParameterPtrAt(i),
                                                           d_mesh->getMaterialPtrAt(i),
                                                           d_mesh->getDimXY(),
                                                           d_mesh->getDimX());
      }


      if(sp->getUpdateType() == SHARED) {
      block.x = 32; block.y = 4; block.z = 1;
      mesh_size_t dim_z = grid.z;
      grid.z = 1; // the z-dimension is gone through in the kernel
      fdtd3dSliced<float,32,4><<<grid, block>>> (d_mesh->getPositionIdxPtrAt(i),
                                                 d_mesh->getMaterialIdxPtrAt(i),
                                                 d_mesh->getPressurePtrAt(i),
                                                 d_mesh->getPastPressurePtrAt(i),
                                                 d_mesh->getParameterPtrAt(i),
                                                 d_mesh->getMaterialPtrAt(i),
                                                 d_mesh->getDimXY(),
                                                 d_mesh->getDimX(),
                                                 dim_z);
      }

    } // End FDTD loop

    cudasafe(cudaDeviceSynchronize(), "kernels3d.cu: launchFDTD3d - cudaDeviceSynchronize after FDTD");
    cudasafe(cudaPeekAtLastError(), "kernels3d.cu: launchFDTD3dStep - Peek after launch");

    ////// TODO boundary loop

    if(past_step_directon == step_direction)
      d_mesh->flipPressurePointers();

    past_step_directon = step_direction;

    d_mesh->switchHalos();

    if((step%PROGRESS_MOD) == 0) {
      step_end = clock()-step_start;
      progressCallback(step, sp->getNumSteps(), ((((float)step_end/CLOCKS_PER_SEC))));
    }
    cudasafe(cudaDeviceSynchronize(), "launchFDTD3dStep: synchDevices at the end");

}

float launchFDTD3dStep_single(CudaMesh* d_mesh,
                              SimulationParameters* sp,
                              unsigned int step,
                              std::vector< std::pair <float*, std::pair<mesh_size_t, int> > >* d_receiver_data,
                              bool (*interruptCallback)(void),
                              void (*progressCallback)(int, int, float)) {
  clock_t start_t;
  clock_t end_t;
  start_t = clock();

  dim3 block(d_mesh->getBlockX(), d_mesh->getBlockY(), 1);
  dim3 grid(d_mesh->getGridDimX(),
            d_mesh->getGridDimY(),
            d_mesh->getPartitionSize()-1);

  c_log_msg(LOG_VERBOSE, "kernels3d.cu: launchFDTD3dStep_single STEP %04d - Mesh dim: x %d y %d z %d ; "
            "Block dim: x %d y %d z %d ; Grid dim: x %u, y %u z %u; Number of partitions: %u: ",
            step, d_mesh->getDimX(), d_mesh->getDimY(), d_mesh->getDimZ(),
      block.x, block.y, block.z, grid.x, grid.y, grid.z,
      d_mesh->getNumberOfPartitions());

  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  clock_t step_start, step_end;
  step_start = clock();
  if(interruptCallback()) {
    c_log_msg(LOG_INFO, "kernels3d.cu: launchFDTD3dStep_single interrupted at step %d", step);
    return 0.0;
  }

  ///////// Source Loop
  for(unsigned int i = 0; i < sp->getNumSources(); i++) {
    float sample = sp->getSourceSample(i, step);
    nv::Vec3i pos = sp->getSourceElementCoordinates(i);

    if(sp->getSource(i).getSourceType() == SRC_HARD) {
      d_mesh->setSample<float>(sample, pos.x, pos.y, pos.z);
    }
    else {
      d_mesh->addSample<float>(sample, pos.x, pos.y, pos.z);
    }
  }

  /////// Receiver loop
  for(unsigned int i = 0; i < sp->getNumReceivers(); i++) {
    nv::Vec3i pos = sp->getReceiverElementCoordinates(i);
    int d_idx = d_receiver_data->at(i).second.second;
    mesh_size_t e_idx = d_receiver_data->at(i).second.first;
    if(d_receiver_data->at(i).second.second == -1) continue;
    float* dest = (d_receiver_data->at(i).first)+step;
    float* src = d_mesh->getPressurePtrAt(d_idx)+e_idx;
    cudaSetDevice(d_mesh->getDeviceAt(d_idx));
    cudasafe(cudaMemcpy(dest, src,  sizeof(float), cudaMemcpyDeviceToDevice), "Memcopy");
  } // End receiver Loop

  cudasafe(cudaDeviceSynchronize(), "kernels3d.cu: launchFDTD3dStep_single - cudaDeviceSynchronize after Source insert");

  ////// FDTD partition loop
  for(unsigned int i = 0; i < d_mesh->getNumberOfPartitions(); i++)
  {
    cudasafe(cudaSetDevice(d_mesh->getDeviceAt(i)), "kernels3d.cu: launchFDTD3dStep_single - set device");

    grid.z = d_mesh->getPartitionSize(i)-1;

    if(sp->getUpdateType() == SRL_FORWARD) {
      fdtd3dStdMaterials<float><<<grid, block>>>(d_mesh->getPositionIdxPtrAt(i),
                                                 d_mesh->getMaterialIdxPtrAt(i),
                                                 d_mesh->getPressurePtrAt(i),
                                                 d_mesh->getPastPressurePtrAt(i),
                                                 d_mesh->getParameterPtrAt(i),
                                                 d_mesh->getMaterialPtrAt(i),
                                                 d_mesh->getDimXY(),
                                                 d_mesh->getDimX());
    }

    if(sp->getUpdateType() == SRL) {
      fdtd3dStdKowalczykMaterials<float><<<grid, block>>>(d_mesh->getPositionIdxPtrAt(i),
                                                          d_mesh->getMaterialIdxPtrAt(i),
                                                          d_mesh->getPressurePtrAt(i),
                                                          d_mesh->getPastPressurePtrAt(i),
                                                          d_mesh->getParameterPtrAt(i),
                                                          d_mesh->getMaterialPtrAt(i),
                                                          d_mesh->getDimXY(),
                                                          d_mesh->getDimX());
    }

    if(sp->getUpdateType() == SHARED) {
      block.x = 32; block.y = 4; block.z = 1;
      mesh_size_t dim_z = grid.z;
      grid.z = 1; // the z-dimension is gone through in the kernel
      fdtd3dSliced<float,32,4><<<grid, block>>>(d_mesh->getPositionIdxPtrAt(i),
                                                d_mesh->getMaterialIdxPtrAt(i),
                                                d_mesh->getPressurePtrAt(i),
                                                d_mesh->getPastPressurePtrAt(i),
                                                d_mesh->getParameterPtrAt(i),
                                                d_mesh->getMaterialPtrAt(i),
                                                d_mesh->getDimXY(),
                                                d_mesh->getDimX(),
                                                dim_z);
    }
  } // End FDTD partition loop

  cudasafe(cudaDeviceSynchronize(), "kernels3d.cu: launchFDTD3dStep_single - cudaDeviceSynchronize after FDTD");
  cudasafe(cudaPeekAtLastError(), "kernels3d.cu: launchFDTD3dStep_single - Peek after launch");

  ////// TODO boundary loop

  d_mesh->flipPressurePointers();
  d_mesh->switchHalos();

  cudasafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize after step");
  if((step%PROGRESS_MOD) == 0) {
    step_end = clock()-step_start;
    progressCallback(step, sp->getNumSteps(), ((((float)step_end/CLOCKS_PER_SEC))));
  }

  cudasafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize before return");

  end_t = clock()-start_t;
  //step++; // add the first step
  //c_log_msg(LOG_VERBOSE, "LaunchFDTD3d - time: %f seconds, per step: %f",
  //          ((float)end_t/CLOCKS_PER_SEC), (((float)end_t/CLOCKS_PER_SEC)/step));
  return ((float)end_t/CLOCKS_PER_SEC);
}

float launchFDTD3dStep_double(CudaMesh* d_mesh,
                              SimulationParameters* sp,
                              unsigned int step,
                              std::vector< std::pair <double*, std::pair<mesh_size_t, int> > >* d_receiver_data,
                              bool (*interruptCallback)(void),
                              void (*progressCallback)(int, int, float)) {
  clock_t start_t;
  clock_t end_t;
  start_t = clock();

  dim3 block(d_mesh->getBlockX(), d_mesh->getBlockY(), 1);
  dim3 grid(d_mesh->getGridDimX(),
            d_mesh->getGridDimY(),
            d_mesh->getPartitionSize());

  c_log_msg(LOG_VERBOSE, "kernels3d.cu: launchFDTD3dStep_double STEP %04d - Mesh dim: x %d y %d z %d ; "
            "Block dim: x %d y %d z %d ; Grid dim: x %u, y %u z %u; Number of partitions: %u: ",
            step, d_mesh->getDimX(), d_mesh->getDimY(), d_mesh->getDimZ(),
            block.x, block.y, block.z, grid.x, grid.y, grid.z,
            d_mesh->getNumberOfPartitions());

  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  clock_t step_start, step_end;
  step_start = clock();
  if(interruptCallback()) {
    c_log_msg(LOG_INFO, "kernels3d.cu: launchFDTD3dStep_double interrupted at step %d" ,step);
    return 0.0;
  }

  ///////// Source Loop
  for(unsigned int i = 0; i < sp->getNumSources(); i++) {
    double sample = sp->getSourceSampleDouble(i, step);
    nv::Vec3i pos = sp->getSourceElementCoordinates(i);
    if(sp->getSource(i).getSourceType() == SRC_HARD) {
      d_mesh->setSample<double>(sample, pos.x, pos.y, pos.z);
    }
    else {
      d_mesh->addSample<double>(sample, pos.x, pos.y, pos.z);
    }
  }

  /////// Receiver loop
  for(unsigned int i = 0; i < sp->getNumReceivers(); i++)
  {
    nv::Vec3i pos = sp->getReceiverElementCoordinates(i);
    int d_idx = d_receiver_data->at(i).second.second;
    mesh_size_t e_idx = d_receiver_data->at(i).second.first;
    if(d_idx == -1) continue;
    double* dest = (d_receiver_data->at(i).first)+step;
    double* src = d_mesh->getPressureDoublePtrAt(d_idx)+e_idx;
    cudaSetDevice(d_mesh->getDeviceAt(d_idx));
    cudasafe(cudaMemcpy(dest, src,  sizeof(double), cudaMemcpyDeviceToDevice), "Memcopy");
  } // End receiver Loop

  cudasafe(cudaDeviceSynchronize(), "kernels3d.cu: launchFDTD3dStep_double - cudaDeviceSynchronize after Source insert");

  ////// FDTD partition loop
  for(unsigned int i = 0; i < d_mesh->getNumberOfPartitions(); i++)
  {
    cudasafe(cudaSetDevice(d_mesh->getDeviceAt(i)), "kernels3d.cu: launchFDTD3dStep_double - set device");
    grid.z = d_mesh->getPartitionSize(i)-1;

    if(sp->getUpdateType() == SRL_FORWARD)
    {
      fdtd3dStdMaterials<double><<<grid, block>>>(d_mesh->getPositionIdxPtrAt(i),
                                                  d_mesh->getMaterialIdxPtrAt(i),
                                                  d_mesh->getPressureDoublePtrAt(i),
                                                  d_mesh->getPastPressureDoublePtrAt(i),
                                                  d_mesh->getParameterPtrDoubleAt(i),
                                                  d_mesh->getMaterialPtrDoubleAt(i),
                                                  d_mesh->getDimXY(),
                                                  d_mesh->getDimX());
    }

    if(sp->getUpdateType() == SRL)
    {
      fdtd3dStdKowalczykMaterials<double><<<grid, block>>>(d_mesh->getPositionIdxPtrAt(i),
                                                           d_mesh->getMaterialIdxPtrAt(i),
                                                           d_mesh->getPressureDoublePtrAt(i),
                                                           d_mesh->getPastPressureDoublePtrAt(i),
                                                           d_mesh->getParameterPtrDoubleAt(i),
                                                           d_mesh->getMaterialPtrDoubleAt(i),
                                                           d_mesh->getDimXY(),
                                                           d_mesh->getDimX());
    }

    if(sp->getUpdateType() == SHARED)
    {
      block.x = 32; block.y = 4; block.z = 1;
      mesh_size_t dim_z = grid.z;
      grid.z = 1; // the z-dimension is gone through in the kernel
      fdtd3dSliced<double, 32,4><<<grid, block>>>(d_mesh->getPositionIdxPtrAt(i),
                                                  d_mesh->getMaterialIdxPtrAt(i),
                                                  d_mesh->getPressureDoublePtrAt(i),
                                                  d_mesh->getPastPressureDoublePtrAt(i),
                                                  d_mesh->getParameterPtrDoubleAt(i),
                                                  d_mesh->getMaterialPtrDoubleAt(i),
                                                  d_mesh->getDimXY(),
                                                  d_mesh->getDimX(),
                                                  dim_z);
    }
  } // End FDTD partition loop

  cudasafe(cudaDeviceSynchronize(), "kernels3d.cu: launchFDTD3dStep_double - cudaDeviceSynchronize after FDTD");
  cudasafe(cudaPeekAtLastError(), "kernels3d.cu: launchFDTD3dStep_double - Peek after launch");

  ////// TODO boundary loop

  d_mesh->flipPressurePointers();
  d_mesh->switchHalos();

  cudasafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize after step");

  if((step%PROGRESS_MOD) == 0) {
    step_end = clock()-step_start;
    progressCallback(step, sp->getNumSteps(), ((((float)step_end/CLOCKS_PER_SEC))));
  }

  cudasafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize before return");

  end_t = clock()-start_t;
  //step++;
  //c_log_msg(LOG_INFO, "LaunchFDTD3dDouble - time: %f seconds, per step: %f",
  //          ((float)end_t/CLOCKS_PER_SEC), (((float)end_t/CLOCKS_PER_SEC)));

  return (((float)end_t/CLOCKS_PER_SEC)/step);
}


template <typename T>
__global__ void fdtd3dStdMaterials(const unsigned char* __restrict d_position_ptr,
                                   const unsigned char* __restrict d_material_idx_ptr,
                                   const T* __restrict P, T* P_past,
                                   const T* d_params_ptr,
                                   const T* d_material_ptr,
                                   mesh_size_t d_dim_xy,
                                   mesh_size_t d_dim_x) {

  mesh_size_t x =  blockIdx.x*blockDim.x +threadIdx.x;
  mesh_size_t y =  blockIdx.y*blockDim.y + threadIdx.y;
  mesh_size_t z =  blockIdx.z*blockDim.z;
  if(z > 0) {
    mesh_size_t current = z*d_dim_xy +d_dim_x*y+x;

    unsigned char pos = d_position_ptr[current];
    T position = (T)(pos&FORWARD_POSITION_MASK);
    T switchBit = (T)(pos>>INSIDE_SWITCH);

    unsigned int mat_idx = ((unsigned int)d_material_idx_ptr[current])*20+d_params_ptr[3];
    T beta =  0.5*((T)d_material_ptr[mat_idx]*(6.f-position)*d_params_ptr[0]);

    T p_z_ = P[(current-d_dim_xy)];
    T _p_z = P[(current+d_dim_xy)];
    T p_y_ = P[(current-d_dim_x)];
    T _p_y = P[(current+d_dim_x)];
    T p_x_ = P[(current-1)];
    T p = P[current];
    T _p_x = P[(current+1)];

    T _p = P_past[current];

    T S = _p_z+p_z_+_p_y+p_y_+_p_x+p_x_;
    T ret = switchBit*(1.f/(1.f+beta))*((2.f-position*d_params_ptr[1])*p+d_params_ptr[1]*S-(1.f-beta)*_p);

    P_past[current] = ret;
  }
}

template <typename T, int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void fdtd3dSliced(const unsigned char* __restrict d_position_ptr,
                             const unsigned char* __restrict d_material_idx_ptr,
                             const T* __restrict P, T* P_past,
                             const T* d_params_ptr,
                             const T* d_material_ptr,
                             mesh_size_t dim_xy,
                             mesh_size_t dim_x,
                             mesh_size_t dim_z) {

  mesh_size_t x =  blockIdx.x*blockDim.x +threadIdx.x;
  mesh_size_t y =  blockIdx.y*blockDim.y + threadIdx.y;

  mesh_size_t current_y = y*dim_x;
  mesh_size_t current = current_y+x;

  __shared__ T P_CUR[BLOCK_SIZE_X][BLOCK_SIZE_Y];
  __shared__ T P_DOWN[BLOCK_SIZE_X][BLOCK_SIZE_Y];

  P_CUR[threadIdx.x][threadIdx.y] = P[current];
  P_DOWN[threadIdx.x][threadIdx.y]  = P[current-dim_xy];

  __syncthreads();

  for(int i = 1; i < dim_z; i++) {
    current = i*dim_xy+current_y+x;

    unsigned char pos = d_position_ptr[current];

    T switchBit = (T)(pos>>INSIDE_SWITCH);
    T position = (T)pos;

    unsigned int mat_idx = ((unsigned int)d_material_idx_ptr[current])*20+d_params_ptr[3];
    T beta = 0.5*(d_material_ptr[mat_idx]*(6.f-position)*d_params_ptr[0]);

    T p_z_ = P[current+dim_xy];
    T p_y_ = 0;
    T _p_y = 0;
    T p_x_ = 0;
    T _p_x = 0;

    if(threadIdx.y == 0) {
      _p_y = P[current-dim_x];
      p_y_ = P_CUR[threadIdx.x][threadIdx.y+1];
    }
    else if(threadIdx.y == BLOCK_SIZE_Y-1) {
      _p_y = P_CUR[threadIdx.x][threadIdx.y-1];
      p_y_ = P[current+dim_x];
    }
    else {
      _p_y = P_CUR[threadIdx.x][threadIdx.y-1];
      p_y_ = P_CUR[threadIdx.x][threadIdx.y+1];
    }

    if(threadIdx.x == 0) {
      _p_x = P[current-1];
      p_x_ = P_CUR[threadIdx.x+1][threadIdx.y];
    }
    else if(threadIdx.x == BLOCK_SIZE_X-1) {
      _p_x = P_CUR[threadIdx.x-1][threadIdx.y];
      p_x_ = P[current+1];
    }
    else {
      _p_x = P_CUR[threadIdx.x-1][threadIdx.y];
      p_x_ = P_CUR[threadIdx.x+1][threadIdx.y];
    }

    T p = P_CUR[threadIdx.x][threadIdx.y];
    T _p = P_past[current];

    P_past[current] = switchBit*(1.f/(1.f+beta))*((2.f-position*d_params_ptr[1])*p+d_params_ptr[1]*
                      (P_DOWN[threadIdx.x][threadIdx.y]+p_z_+_p_y+p_y_+_p_x+p_x_)-(1.f-beta)*_p);
    P_DOWN[threadIdx.x][threadIdx.y]  = p;
    P_CUR[threadIdx.x][threadIdx.y] = p_z_;
    __syncthreads();
  }

}

template <typename T>
__global__ void fdtd3dStdKowalczykMaterials(unsigned char* d_position_ptr, unsigned char* d_material_idx_ptr,
                                            const T* __restrict P, T* P_past,
                                            const T* d_params_ptr,
                                            const T* d_material_ptr,
                                            mesh_size_t d_dim_xy,
                                            mesh_size_t d_dim_x) {

    mesh_size_t x =  blockIdx.x*blockDim.x +threadIdx.x;
    mesh_size_t y =  blockIdx.y*blockDim.y + threadIdx.y;

  // Slice index is the same for a single block
  __shared__ mesh_size_t z;
  __shared__ mesh_size_t _z;
  __shared__ mesh_size_t z_;

  z = blockIdx.z*d_dim_xy;
  _z = z-d_dim_xy;
  z_ = z+d_dim_xy;

  mesh_size_t current_y = y*d_dim_x;
  mesh_size_t _current_y = current_y-d_dim_x;
  mesh_size_t current_y_ = current_y+d_dim_x;
  mesh_size_t current = z+current_y+x;

  unsigned char pos = d_position_ptr[current];

  T switchBit = (T)(pos>>INSIDE_SWITCH);
  T dir_x =  (T)(pos&DIR_X);
  T dir_y =  (T)((pos&DIR_Y)>>1);
  T dir_z =  (T)((pos&DIR_Z)>>2);

  unsigned int mat_idx = ((unsigned int)d_material_idx_ptr[current])*20+d_params_ptr[3];
  T beta = d_material_ptr[mat_idx]*d_params_ptr[0]*(dir_x+dir_y+dir_z);

  T p_z[2], p_y[2], p_x[2];
  p_z[1] = P[_z+current_y+x]; // SIGN_Z is down, hence the indexing is inverted
  p_z[0] = P[z_+current_y+x];
  p_y[0] = P[z+_current_y+x];
  p_y[1] = P[z+current_y_+x]; // SIGN_Y is right
  p_x[0] = P[current-1];
  p_x[1] = P[current+1]; // SIGN_X is out

  T p = P[current];
  T _p = P_past[current];

  int sign_x = (pos&SIGN_X)>>4;
  int sign_y = (pos&SIGN_Y)>>5;
  int sign_z = (pos&SIGN_Z)>>6;

  T S_boundary = p_x[sign_x]*dir_x+
                 p_y[sign_y]*dir_y+
                 p_z[sign_z]*dir_z;

  T S = (p_x[0]+p_x[1]+p_y[0]+p_y[1]+p_z[0]+p_z[1]+S_boundary)*d_params_ptr[1];

  P_past[current] = (S-(2-6*d_params_ptr[1])*p+(beta-1.f)*_p)*switchBit*(1.f/(1.f+beta));

}
