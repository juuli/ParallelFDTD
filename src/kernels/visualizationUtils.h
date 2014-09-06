#ifndef VISUALIZATION_UTILS_H
#define VISUALIZATION_UTILS_H

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

#include "cudaMesh.h"

// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

/// \brief Register a pixel/vertex buffer to a Cuda graphics resource
void registerGLtoCuda(struct cudaGraphicsResource **resource, 
                      GLuint buffer, unsigned int vbo_res_flags);

///////////////////////////////////////////////////////////////////////////////
/// \brief update a pixel buffer with the data of the CudaMesh in its current
/// state
/// \param pbo_resouce A Cuda graphics resource
/// \param d_mesh A CudaMesh instance used in the simulation
/// \param current_slice the index of the slice to be drawn on the pixel buffer
/// \param orientation the orientation of the pixel buffer<br> 0: xy <br> 1: xz
/// <br> 2: yz
/// \param selector Select what data is to be drawn <br> 0: pressure 
/// <br> 1: orientation <br> 2: inside/outisde switch
/// \param scheme Used scheme, 0: forward-difference / Bilbao <br>
/// 1: centered-difference / Kowalczyk
/// \param dB Dynamic range of the visualization in decivbels
///////////////////////////////////////////////////////////////////////////////
void updatePixelBuffer(struct cudaGraphicsResource **pbo_resource, 
                       CudaMesh* d_mesh,
                       unsigned int current_slice,
                       unsigned int orientation,
                       unsigned int selector,
                       unsigned int scheme,
                       float db);

///////////////////////////////////////////////////////////////////////////////
/// \brief update a vertex buffer with the data of the CudaMesh in its current
/// state
/// \param vbo_resouce A Cuda graphics resource for the vertex coordinates
/// \param color_resource A Cuda graphics resouce for the color values
/// \param d_positions A pointer to the orientation data of the mesh
/// \param d_materials A pointer to the material index data of the mesh
/// \param dim_x The x dimension of the mesh
/// \param dim_y The y dimension of the mesh
/// \param dim_z The z dimension of the mesh
/// \param dx The size of the voxel edge
///////////////////////////////////////////////////////////////////////////////
void renderBoundariesVBO(struct cudaGraphicsResource **vbo_resource, 
                         struct cudaGraphicsResource **color_resource,
                         unsigned char* d_positions, 
                         unsigned char* d_materials, 
                         unsigned int dim_x,  unsigned int dim_y, 
                         unsigned int dim_z, float dx);                       
                       
                                          
///////////////////////////////////////////////////////////////////////////////
/// \brief Function to capture pressure data from a slice of the mesh
/// \param step_to_capture A contained holding the step indices which are to be
/// capture
/// \param slice_to_capture A container holding the slice indices which are to be
/// captured
/// \param slice_orientation A container holding the slice orienation which are to be
/// captured.<br> 0: xy, <br> 1: xz, <br> 2: yz
/// \param current_step Current step of the simulation
/// \param captureCallback A callback called after the data has been fetched. Can
/// be used for example to write the data into a file
///////////////////////////////////////////////////////////////////////////////
void captureSliceFast(CudaMesh* d_mesh,
                      std::vector<unsigned int> &step_to_capture,
                      std::vector<unsigned int> &slice_to_capture,
                      std::vector<unsigned int> &slice_orienation,
                      unsigned int current_step,
                      void (*captureCallback)(float*, unsigned char*, unsigned int, unsigned int, 
                                              unsigned int, unsigned int, unsigned int));

///////////////////////////////////////////////////////////////////////////////
/// \brief Function to capture pressure data of the whole mesh at time instance.
///         The function is currently working only with a single device
/// \param mesh_to_capture A contained holding the step indices which are to be
/// capture
/// \param mesh_captures A container holding the captured mesh instances
/// \param current_step Current step of the simulation
///////////////////////////////////////////////////////////////////////////////
void captureMesh(CudaMesh* d_mesh,
                 std::vector<unsigned int> &mesh_to_capture,
                 std::vector<float*> &mesh_captures,
                 unsigned int current_step);

                 
///////////////////////////////////////////////////////////////////////////////
/// \brief A kernel to write pressure values to a pixel buffer
/// \param pixels A mapped pointer to the pixel buffer data
/// \param P A pointer to the pressure mesh
/// \param dim_x Size of the mesh, x-dimension
/// \param dim_xy Size of a xy- mesh slice
/// \param dim_y Size of the mesh, y-dimension
/// \param num_elems Total number of elements in the mesh
/// \param offset Offset of the slice being extracted
/// \param orientation Orientation of the slice. 0: xy, 1:xz, 2: yz
/// \param dB The dynamic range of the rendering in decibels
///////////////////////////////////////////////////////////////////////////////
__global__ void renderPressuresPBO(uchar4* pixels, float* P, unsigned int dim_x,
                                   unsigned int dim_xy, unsigned int dim_y, 
                                   unsigned int num_elems,
                                   uint3 offset, unsigned int orientation, 
                                   float dB);

///////////////////////////////////////////////////////////////////////////////
/// \brief A kernel to write position values to a pixel buffer
/// \param pixels A mapped pointer to the pixel buffer data
/// \param positions A pointer to the orientation mesh
/// \param dim_x Size of the mesh, x-dimension
/// \param dim_xy Size of a xy- mesh slice
/// \param dim_y Size of the mesh, y-dimension
/// \param num_elems Total number of elements in the mesh
/// \param offset Offset of the slice being extracted
/// \param orientation Orientation of the slice.<br>0: xy,<br>1:xz,<br>2: yz
/// \param scheme Scheme used in the simulation,<br>0: forward-difference<br>
/// 1: cenetered-difference
///////////////////////////////////////////////////////////////////////////////
__global__ void renderPositionsPBO(uchar4* pixels, unsigned char* positions, 
                                   unsigned int dim_x, unsigned int dim_xy, 
                                   unsigned int dim_y, unsigned int num_elems,
                                   uint3 offset, unsigned int orientation,
                                   unsigned int scheme);

                                   
///////////////////////////////////////////////////////////////////////////////
/// \brief A kernel to write material index values to a pixel buffer
/// \param pixels A mapped pointer to the pixel buffer data
/// \param material_idx A pointer to the material index mesh
/// \param dim_x Size of the mesh, x-dimension
/// \param dim_xy Size of a xy- mesh slice
/// \param dim_y Size of the mesh, y-dimension
/// \param num_elems Total number of elements in the mesh
/// \param offset Offset of the slice being extracted
/// \param orientation Orientation of the slice.<br>0: xy,<br>1:xz,<br>2: yz
/// \param scheme Scheme used in the simulation,<br>0: forward-difference<br>
/// 1: cenetered-difference
///////////////////////////////////////////////////////////////////////////////
__global__ void renderMaterialsPBO(uchar4* pixels, unsigned char* material_idx, 
                                   unsigned int dim_x, unsigned int dim_xy, 
                                   unsigned int dim_y, unsigned int num_elems,
                                   uint3 offset, unsigned int orientation,
                                   unsigned int scheme);                                   
                                   
///////////////////////////////////////////////////////////////////////////////
/// \brief A kernel to write inside/outside switch values to a pixel buffer
/// \tparam T single/double precision mesh
/// \param pixels A mapped pointer to the pixel buffer data
/// \param positions A pointer to the orientation mesh
/// \param dim_x Size of the mesh, x-dimension
/// \param dim_xy Size of a xy- mesh slice
/// \param dim_y Size of the mesh, y-dimension
/// \param num_elems Total number of elements in the mesh
/// \param offset Offset of the slice being extracted
/// \param orientation Orientation of the slice. <br>0: xy,<br> 1:xz,<br> 2: yz
/// \param scheme Scheme used in the simulation,<br>0: forward-difference<br>
/// 1: cenetered-difference
///////////////////////////////////////////////////////////////////////////////
__global__ void renderSwitchPBO(uchar4* pixels, unsigned char* positions, 
                                unsigned int dim_x,
                                unsigned int dim_xy, unsigned int dim_y, 
                                unsigned int num_elems,
                                uint3 offset, unsigned int orientation,
                                unsigned int scheme);
 
///////////////////////////////////////////////////////////////////////////////
/// \brief A kernel to write pressure values to a vertex buffer
/// \param pos A mapped pointer to the vertex coordinates of the VBO
/// \param color A mapped pointer to the color coordinates of the VBO
/// \param P The pressure values of the mesh
/// \param dim_x Size of the mesh, x-dimension
/// \param dim_y Size of the mesh, y-dimension
/// \param dim_z Size of the mesh, z-dimension
/// \param dx The size of the voxel edge
///////////////////////////////////////////////////////////////////////////////
__global__ void renderPressuresVBO(float4* pos, float4* color, 
                                   float* P, 
                                   unsigned int dim_x, 
                                   unsigned int dim_y, 
                                   unsigned int dim_z, 
                                   float dx);

///////////////////////////////////////////////////////////////////////////////
/// \brief A kernel to write orientation values to a vertex buffer
/// \param pixels A mapped pointer to the pixel buffer data
/// \param positions A pointer to the orientation mesh
/// \param dim_x Size of the mesh, x-dimension
/// \param dim_y Size of the mesh, y-dimension
/// \param d_positions A pointer to the orientation mesh 
/// \param d_materials A pointer to the material index mesh
/// \param slice current index in z-dimension
/// \param dx The size of the voxel edge
///////////////////////////////////////////////////////////////////////////////
__global__ void boundaryRenderKernelVBO(float4 *pos, float4* color, 
                                        unsigned int dim_x, 
                                        unsigned int dim_y,
                                        unsigned char* d_positions, 
                                        unsigned char* d_materials, 
                                        unsigned int slice, 
                                        float dx);

///////////////////////////
// Capture kernels

__global__ void captureSliceKernel(const float* __restrict P,
                                   const unsigned char* __restrict d_positions,
                                   float* capture_P,
                                   unsigned char* capture_K, const unsigned int num_elems,
                                   const int dim_x, const int dim_y, const int dim_xy, 
                                   uint3 offset, const unsigned int limit, 
                                   const int orientation);

#endif
