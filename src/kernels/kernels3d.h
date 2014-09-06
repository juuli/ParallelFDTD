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
                                   int dim_xy, int dim_x);


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
                             int d_dim_xy, int d_dim_x, int d_dim_z);

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
                                            int d_dim_xy, int d_dim_x);

#endif
