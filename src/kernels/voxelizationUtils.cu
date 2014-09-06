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
#include "voxelizationUtils.h"

#include "../Voxelizer/include/helper_math.h"
#include "../Voxelizer/include/voxelizer.h"

void nodes2Vectors(vox::LongNode* nodes, 
                   unsigned char** d_pos, 
                   unsigned char** d_mat,
                   uint3 dim) {

  c_log_msg(LOG_INFO, "voxelizationUtils.cu: nodes2Vectors -  begin");
  unsigned int num_elems = dim.x*dim.y*dim.z;
  
  int threadsPerBlock = 512;
  dim3 block_dim(threadsPerBlock);

  int numBlocks = (num_elems + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid_dim(ceil(sqrt(numBlocks)),ceil(sqrt(numBlocks))); 

  c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: nodes2Vectors - Block x: %u y: %u z: %u", 
            block_dim.x, block_dim.y, block_dim.z);
  c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: nodes2Vectors - Grid x: %u y: %u z: %u", 
            grid_dim.x, grid_dim.y, grid_dim.z);
  
  nodes2VectorsKernel< vox::LongNode ><<<grid_dim, block_dim>>>(nodes, 
                                                               *d_pos, 
                                                               *d_mat, 
                                                               num_elems);
}


void voxelizeGeometry(float* vertices, 
                      unsigned int* indices, 
                      unsigned char* materials,
                      unsigned int number_of_triangles, 
                      unsigned int number_of_vertices,
                      unsigned int number_of_unique_materials,
                      double voxel_edge,
                      unsigned char** d_position_idx,
                      unsigned char** d_material_idx,
                      uint3* voxelization_dim) {
  clock_t start_t;
  clock_t end_t;
  start_t = clock();

  c_log_msg(LOG_INFO, 
        "voxelizationUtils: voxelizeGeometryToDevice - voxelizeGeometryToDevice - begin");          
  
  vox::Voxelizer<vox::LongNode> voxelizer(vertices, 
                                          indices, 
                                          number_of_vertices, 
                                          number_of_triangles);
  if(materials) {
    c_log_msg(LOG_DEBUG, "voxelizationUtils: assigning materials");
    voxelizer.setMaterials(materials, number_of_unique_materials);
    voxelizer.setMaterialOutput(true);
  }

  cudasafe(cudaPeekAtLastError(),
           "voxelizationUtils: voxelizeGeometryToDevice - peek after voxelizer initialization");

  ////////// Voxelize to nodes
  c_log_msg(LOG_DEBUG, "voxelizationUtils: voxelizeGeometryToDevice - toNodes ");
  vox::LongNode* nodes = NULL;
  std::vector< vox::NodePointer< vox::LongNode > > node_ptr;
  node_ptr = voxelizer.voxelizeToNodes(voxel_edge);
  nodes = node_ptr.at(0).ptr;

  cudasafe(cudaDeviceSynchronize(), 
           "voxelizationUtils: voxelizeGeometryToDevice - cudaDeviceSynchronize after voxelization");

  *voxelization_dim = node_ptr.at(0).dim;
  unsigned int num_elements = (*voxelization_dim).x*(*voxelization_dim).y*(*voxelization_dim).z;
  
  end_t = clock()-start_t;
  c_log_msg(LOG_INFO, 
            "voxelizationUtils.cu: voxelizeGeometryToDevice- voxelization time: %f seconds", 
            ((float)end_t/CLOCKS_PER_SEC));

  // Allocate an additional buffer for indexing out of bounds
  unsigned int buffer =  (*voxelization_dim).x*(*voxelization_dim).y+(*voxelization_dim).x+1;
  (*d_position_idx) = valueToDevice<unsigned char>(num_elements+buffer, (unsigned char)0, 0);
  (*d_material_idx) = valueToDevice<unsigned char>(num_elements+buffer, (unsigned char)0, 0);

  //////////// Translate the node data to vectors
  start_t = clock();

  nodes2Vectors(nodes, d_position_idx, d_material_idx, *voxelization_dim);

  end_t = clock()-start_t;

  c_log_msg(LOG_INFO,"voxelizationUtils.cu: nodes2Vectors  - time: %f seconds",
                     ((float)end_t/CLOCKS_PER_SEC));

  cudasafe(cudaPeekAtLastError(), "voxelizationUtils.cu: voxelizeGeometryToDevice" 
                                  "- peek before return");

  cudasafe(cudaDeviceSynchronize(), "voxelizationUtils.cu: voxelizeGeometryToDevice" 
                                    "- cudaDeviceSynchronize at before return");

  c_log_msg(LOG_INFO, "voxelizationUtils.cu: voxelizeGeometryToDevice - voxelization done");


  c_log_msg(LOG_INFO, "voxelizationUtils.cu: voxelizeGeometryToDevice\n" 
                      "- dim x: %d y: %d z: %d num elements %d",
                      (*voxelization_dim).x, 
                      (*voxelization_dim).y, 
                      (*voxelization_dim).z, 
                      num_elements);

  cudasafe(cudaFree(nodes), "cudaFree nodes");  
  printMemInfo("voxelizeGeometryDevice memory before return", getCurrentDevice());
}

template<class Node>
__global__ void nodes2VectorsKernel(Node* nodes, 
                                     unsigned char* d_position_idx_ptr, 
                                     unsigned char* d_material_idx_ptr, 
                                     unsigned int num_elems) {
  int idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < num_elems) {
    d_position_idx_ptr[idx] = nodes[idx].bid();
    d_material_idx_ptr[idx] = nodes[idx].mat();
  }
}

