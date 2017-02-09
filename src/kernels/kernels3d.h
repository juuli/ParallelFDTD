#ifndef FDTD_KERNEL
#define FDTD_KERNEL

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

#include "cudaUtils.h"
#include "cudaMesh.h"

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../base/SimulationParameters.h"

#define PROGRESS_MOD 100

///////////////////////////////////////////////////////////////////////////////
/// \brief Launch an predefined number of FDTD steps in single precision
/// \param[in] d_mesh CudaMesh containing the simulation domain
/// \param[in] sp The simulation parameters of the simulation
/// \param[in, out] h_retur_ptr A memory allocation for the return values of
/// the simulation in the host memory
/// \param interruptCallback A callback function which is called between each
/// step to check if the simulation is interrupted by the user
/// \param prgressCallback A callback function that is called between each
/// PROGRESS_MOD number of steps to print progress information
///////////////////////////////////////////////////////////////////////////////
float launchFDTD3d(CudaMesh* d_mesh,
                   SimulationParameters* sp,
                   float* h_return_ptr,
                   bool (*interruptCallback)(void),
                   void (*progressCallback)(int, int, float));

///////////////////////////////////////////////////////////////////////////////
/// \brief Launch an predefined number of FDTD steps in double precision
/// \param[in] d_mesh CudaMesh containing the simulation domain
/// \param[in] sp The simulation parameters of the simulation
/// \param[in, out] h_retur_ptr A memory allocation for the return values of
/// the simulation in the host memory
/// \param interruptCallback A callback function which is called between each
/// step to check if the simulation is interrupted by the user
/// \param prgressCallback A callback function that is called between each
/// PROGRESS_MOD number of steps to print progress information
///////////////////////////////////////////////////////////////////////////////
float launchFDTD3dDouble(CudaMesh* d_mesh,
                         SimulationParameters* sp,
                         double* h_return_ptr,
                         bool (*interruptCallback)(void),
                         void (*progressCallback)(int, int, float));

///////////////////////////////////////////////////////////////////////////////
/// \brief Launch a FDTD step in single precision. Used with the visualization
/// and captures
/// \param[in] d_mesh CudaMesh containing the simulation domain
/// \param[in] sp The simulation parameters of the simulation
/// \param[in, out] h_retur_ptr A memory allocation for the return values of
/// the simulation in the host memory
/// \param[in] step The index of the current step beeing executed
/// \param[in] direction The direction of simulation step. -1 swithces
/// the direction of the particle velocity
/// \param prgressCallback A callback function that is called between each
/// PROGRESS_MOD number of steps to print progress information
///////////////////////////////////////////////////////////////////////////////
void launchFDTD3dStep(CudaMesh* d_mesh,
                      SimulationParameters* sp,
                      float* h_return_ptr,
                      unsigned int step,
                      int step_direction,
                      void (*progressCallback)(int, int, float));

///////////////////////////////////////////////////////////////////////////////
/// \brief Function allocates on the CUDA devices the memory to write receiver
/// data. Taken from launchFDTD3d() function.
///
///    To be used with:
///      -) get_receiver_data_from_device()
///      -) launchFDTD3dStep_single()
///      -) launchFDTD3dStep_double()
///
/// \param[in] d_mesh CudaMesh containing the simulation domain
/// \param[in] sp The simulation parameters of the simulation
/// \returns The vector of pointers and corresponding data. Feed it to step
///  functions.
///    T should be float or double
///                    ??? Maybe move to cudaMesh ???
///////////////////////////////////////////////////////////////////////////////
template <typename T>
std::vector< std::pair <T*, std::pair<mesh_size_t, int> > >
  prepare_receiver_data_on_device(CudaMesh* d_mesh,
                  SimulationParameters* sp)
{
  std::vector< std::pair <T*, std::pair<mesh_size_t, int> > > d_receiver_data;
  for(unsigned int i = 0; i < sp->getNumReceivers(); i++) {
    nv::Vec3i pos = sp->getReceiverElementCoordinates(i);
    mesh_size_t element_idx;
    int device_idx;
    d_mesh->getElementIdxAndDevice(pos.x, pos.y, pos.z, &device_idx, &element_idx);
    T* d_return_ptr = (T*)NULL;

    if(device_idx != -1)
      d_return_ptr = toDevice<T>(sp->getNumSteps(), d_mesh->getDeviceAt(device_idx));
    std::pair<mesh_size_t, int> temp_i(element_idx, device_idx);
    std::pair<T*, std::pair<mesh_size_t,int> > temp_d(d_return_ptr, temp_i);
    d_receiver_data.push_back(temp_d);
  }
  return d_receiver_data;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Function retrieves the allocated data on device by prepare_receiver_
// data_on_device() inside host h_return_ptr. Taken from launchFDTD3d().
///
///    To be used with:
///      -) prepare_receiver_data_on_device()
///      -) launchFDTD3dStep_single()
///      -) launchFDTD3dStep_double()
///
/// \param[in] d_mesh CudaMesh containing the simulation domain
/// \param[in] sp The simulation parameters of the simulation
/// \param[in] d_receiver_data The device data where receiver responses were
///    stored. Comes from prepare_receiver_data_on_device().
/// \returns The vector of pointers and corresponding data. Feed it to step
///  functions.
///    T should be float or double
///                    ??? Maybe move to cudaMesh ???
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void get_receiver_data_from_device(CudaMesh* d_mesh,
                                   SimulationParameters* sp,
                                   T* h_return_ptr,
                                   std::vector< std::pair <T*, std::pair<mesh_size_t, int> > >* d_receiver_data) {
  /////// Copy device return data to host
  for(unsigned int i = 0; i < sp->getNumReceivers(); i++) {
    T* dest = h_return_ptr+i*sp->getNumSteps();
    T* src = d_receiver_data->at(i).first;
    if(d_receiver_data->at(i).second.second == -1) continue;
    int dev = d_mesh->getDeviceAt(d_receiver_data->at(i).second.second);
    copyDeviceToHost(sp->getNumSteps(), dest, src, dev);
    destroyMem(d_receiver_data->at(i).first, dev);
  } // End receiver Loop
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Launch a FDTD step in single precision. Is similar to launchFDTD3dS
/// tep but that did not work (only zeros were recorded on two devices).
///    Function roots in launchFDTD3d() function.
///
///   It is easy to be used with functions:
///    - prepare_receiver_data_on_device();
///    - get_receiver_data_from_device();
///
/// \param[in] d_mesh CudaMesh containing the simulation domain
/// \param[in] sp The simulation parameters of the simulation
/// \param[in, out] h_retur_ptr A memory allocation for the return values of
/// the simulation in the host memory
/// \param[in] step The index of the current step being executed
/// \param[in] d_receiver_data The receiver data on the device. See function
/// launchFDTD3d() or prepare_receiver_data_on_device() for details.
/// \param interruptCallback A callback function which is called between each
/// step to check if the simulation is interrupted by the user
/// \param prgressCallback A callback function that is called between each
///////////////////////////////////////////////////////////////////////////////
float launchFDTD3dStep_single(CudaMesh* d_mesh,
                              SimulationParameters* sp,
                              unsigned int step,
                              std::vector< std::pair <float*, std::pair<mesh_size_t, int> > >* d_receiver_data,
                              bool (*interruptCallback)(void),
                              void (*progressCallback)(int, int, float));

///////////////////////////////////////////////////////////////////////////////
/// \brief Launch a FDTD step in double precision. Is similar to function
/// launchFDTD3dStep_single().
///    Function roots in launchFDTD3dDouble() function.
///
///   It is easy to be used with functions:
///    - prepare_receiver_data_on_device();
///    - get_receiver_data_from_device();
///
/// \param[in] d_mesh CudaMesh containing the simulation domain
/// \param[in] sp The simulation parameters of the simulation
/// \param[in, out] h_retur_ptr A memory allocation for the return values of
/// the simulation in the host memory
/// \param[in] step The index of the current step being executed
/// \param[in] d_receiver_data The receiver data on the device. See function
/// launchFDTD3dDouble() or prepare_receiver_data_on_device() for details.
/// \param interruptCallback A callback function which is called between each
/// step to check if the simulation is interrupted by the user
/// \param prgressCallback A callback function that is called between each
///////////////////////////////////////////////////////////////////////////////
float launchFDTD3dStep_double(CudaMesh* d_mesh,
                              SimulationParameters* sp,
                              unsigned int step,
                              std::vector< std::pair <double*, std::pair<mesh_size_t, int> > >* d_receiver_data,
                              bool (*interruptCallback)(void),
                              void (*progressCallback)(int, int, float));

//// Kernels

///////////////////////////////////////////////////////////////////////////////
/// \brief Kernel for FDTD step using the forward difference boundary
/// \tparam T defines the precision used in the calculation (float / double)
/// \param[in] d_positions A device pointer to the beginning of the orientation
///  mesh
/// \param[in] d_materials A device pointer to the beginning of the material
/// index mesh
/// \param[in] P A device pointer to the current pressure value mesh
/// \param[in] P_past A device pointer to the pressure values mesh of the past step
/// \param[in] d_params_ptr A device pointer to simulation parameters
/// \param[in] d_params_ptr A device pointer to the material coefficients
/// \param[in] dim_xy The size of the xy slice of the mesh
/// \param[in] dim_x The length of the x dimension of the mesh
///////////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ void fdtd3dStdMaterials(const unsigned char* __restrict d_positions,
                                   const unsigned char* __restrict d_materials,
                                   const T* __restrict P, T* P_past,
                                   const T* d_params_ptr,
                                   const T* d_material_ptr,
                                   mesh_size_t dim_xy,
                                   mesh_size_t dim_x);


///////////////////////////////////////////////////////////////////////////////
/// \brief Kernel for FDTD step using the forward difference boundary
/// and the sliced method
/// \tparam T defines the precision used in the calculation (float / double)
/// \tparam BLOCK_SIZE_X the size of a thread block x-dimension
/// \tparam BLOCK_SIZE_Y the size of a thread block y-dimension
/// \param[in] d_positions A device pointer to the beginning of the orientation
///  mesh
/// \param[in] d_materials A device pointer to the beginning of the material
/// index mesh
/// \param[in] P A device pointer to the current pressure value mesh
/// \param[in] P_past A device pointer to the pressure values mesh of the past step
/// \param[in] d_params_ptr A device pointer to simulation parameters
/// \param[in] d_params_ptr A device pointer to the material coefficients
/// \param[in] dim_xy The size of the xy slice of the mesh
/// \param[in] dim_x The length of the x dimension of the mesh
/// \param[in] dim_x The length of the z dimension of the mesh
///////////////////////////////////////////////////////////////////////////////
template <typename T, int BLOCK_SIZE_X, int BLOCK_SIZE_Y >
__global__ void fdtd3dSliced(const unsigned char* __restrict d_positions,
                             const unsigned char* __restrict d_materials,
                             const T* __restrict P, T* P_past,
                             const T* d_params_ptr,
                             const T* d_material_ptr,
                             mesh_size_t d_dim_xy,
                             mesh_size_t d_dim_x,
                             mesh_size_t d_dim_z);

///////////////////////////////////////////////////////////////////////////////
/// \brief Kernel for FDTD step using the centered-difference boundary
/// \tparam T defines the precision used in the calculation (float / double)
/// \param[in] d_positions A device pointer to the beginning of the orientation
///  mesh
/// \param[in] d_materials A device pointer to the beginning of the material
/// index mesh
/// \param[in] P A device pointer to the current pressure value mesh
/// \param[in] P_past A device pointer to the pressure values mesh of the past step
/// \param[in] d_params_ptr A device pointer to simulation parameters
/// \param[in] d_params_ptr A device pointer to the material coefficients
/// \param[in] dim_xy The size of the xy slice of the mesh
/// \param[in] dim_x The length of the x dimension of the mesh
///////////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ void fdtd3dStdKowalczykMaterials(unsigned char* d_position,
                                            unsigned char* d_materials,
                                            const T* __restrict P, T* P_past,
                                            const T* d_params_ptr,
                                            const T* d_material_ptr,
                                            mesh_size_t d_dim_xy,
                                            mesh_size_t d_dim_x);

#endif
