#ifndef VOXEL_UTILS_H
#define VOXEL_UTILS_H

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

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "cudaUtils.h"

class Node;

//////////////////////////////////////////////////////
// A wrapper for the voxelizer library
// parses the nodes to position and material vectors

void voxelizeGeometry(float* vertices, 
                      unsigned int* indices, 
                      unsigned char* materials,
                      unsigned int number_of_triangles, 
                      unsigned int number_of_vertices,
                      unsigned int number_of_unique_materials,
                      double voxel_edge,
                      unsigned char** d_postition_idx,
                      unsigned char** d_materials_idx,
                      uint3* voxelization_dim);

template<class Node>
__global__ void nodes2VectorsKernel(Node* nodes, 
                                    unsigned char* d_position_idx_ptr, 
                                    unsigned char* d_material_idx_ptr, 
                                    unsigned int num_elems);
#endif
