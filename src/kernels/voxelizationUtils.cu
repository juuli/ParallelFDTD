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
#include "cudaUtils.h"
#include "voxelizationUtils.h"

#include <algorithm>
#include "../Voxelizer/include/helper_math.h"
#include "../Voxelizer/include/voxelizer.h"

#ifndef FLT_EPSILON
    #define FLT_EPSILON __FLT_EPSILON__
#endif

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

unsigned char nodes2Vectors_surface_only(vox::SurfaceNode* nodes,
                                vox::HashMap surface_nodes_HM,
                                unsigned char** d_pos,
                                unsigned char** d_mat,
                                uint3 dim,
                                const unsigned char bit_mask,
                                const unsigned char * unique_materials_ids,
                                unsigned int number_of_unique_materials) {

  c_log_msg(LOG_INFO, "voxelizationUtils.cu: nodes2Vectors_surface_only -  begin");
  unsigned int num_elems = dim.x*dim.y*dim.z;

  unsigned int threadsPerBlock = 512;
  dim3 block_dim(threadsPerBlock); // block_dim.x = threadsPerBlock; block_dim.y = 1; block_dim.z = 1
  unsigned int numBlocks = (num_elems + threadsPerBlock - 1) / threadsPerBlock;
  // grid_dim.x = grid_dim.y = ceil(sqrt(numBlocks); block_dim.z = 1
  dim3 grid_dim(ceil(sqrt(numBlocks)),ceil(sqrt(numBlocks)));

  c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: nodes2Vectors_surface_only - Block x: %u y: %u z: %u",
            block_dim.x, block_dim.y, block_dim.z);
  c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: nodes2Vectors_surface_only - Grid x: %u y: %u z: %u",
            grid_dim.x, grid_dim.y, grid_dim.z);


  // Get result in a temporary position_matrix and material matrix on device:
  unsigned char* d_pos_temp = (unsigned char*)NULL;
  unsigned char* d_mat_temp = (unsigned char*)NULL;
  d_pos_temp = toDevice<unsigned char>(num_elems + 2*dim.x*dim.y, 0);
  d_mat_temp = toDevice<unsigned char>(num_elems + 2*dim.x*dim.y, 0);
  cudaMemset(	d_pos_temp, 1, dim.x*dim.y*sizeof(unsigned char));
  cudaMemset(	d_pos_temp + num_elems + dim.x*dim.y, 1, dim.x*dim.y*sizeof(unsigned char));
  nodes2Vectors_prepare_surface_data_Kernel<<<grid_dim, block_dim>>>(nodes,
                                                                     surface_nodes_HM,
                                                                     d_pos_temp + (int)dim.x*dim.y,
                                                                     d_mat_temp + (int)dim.x*dim.y,
                                                                     num_elems);

  // Now, all air nodes have a d_pos_temp = 1 (d_mat_temp = 0) and all boundary
  // nodes have a d_pos_temp = 0 with a corresponding material_id in d_mat_temp.
  // Keep the same of the domain Prepare new Kernel call for Count_Air_Neighboring_Nodes_Kernel()

  c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: Count_Air_Neighboring_Nodes - Block x: %u y: %u z: %u",
      block_dim.x, block_dim.y, block_dim.z);
  c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: Count_Air_Neighboring_Nodes - Grid x: %u y: %u z: %u",
      grid_dim.x, grid_dim.y, grid_dim.z);

  // Prepare data:
  unsigned int dim_xy = dim.x*dim.y;
  unsigned int dim_x = dim.x;
  double one_over_dim_xy = 1.0 / (double)dim_xy;
  double one_over_dim_x = 1.0 / (double)dim.x;
  // Build a hash: maps material_id to its index: (vox::HashMap needs a .fromHosttoDevice() method)
  // first get the maximum:
  unsigned char max_ = 0;
  for(size_t index = 0; index < number_of_unique_materials; index++){
    if (max_<unique_materials_ids[index])
      max_=unique_materials_ids[index];
  }
  unsigned char* d_error = (unsigned char*)NULL;
  d_error = valueToDevice<unsigned char>(1, 0, 0);
  c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: Count_Air_Neighboring_Nodes - "
        "Max_material_ID = %u, dim_xy = %u, dim_x = %u", max_, dim_xy, dim_x);
  // Call it:
  Count_Air_Neighboring_Nodes_Kernel<<<grid_dim, block_dim>>>(*d_pos + (int)dim.x*dim.y,
                                                              d_pos_temp,
                                *d_mat + (int)dim.x*dim.y,
                                max_,
                                                              d_mat_temp,
                                                              num_elems,
                                                              dim_xy,
                                                              dim_x,
                                                              one_over_dim_xy,
                                                              one_over_dim_x,
                                                              bit_mask,
                                d_error);

  cudasafe(cudaDeviceSynchronize(), "voxelizationUtils.cu: nodes2Vectors_surface_only - cudaDeviceSynchronize after air counts");
  // Check errors:
  unsigned char h_error = 0;
  copyDeviceToHost(1, &h_error, d_error, 0);
  // Free the temporary memory:
  destroyMem(d_error, 0);
  destroyMem(d_pos_temp, 0);
  destroyMem(d_mat_temp, 0);
  return h_error;
}

unsigned char to_ParallelFDTD_surface_voxelization(
                                unsigned char** d_pos,
                                unsigned char** d_mat,
                                uint3 dim,
                                const unsigned char bit_mask,
                                const unsigned char * unique_materials_ids,
                                unsigned int number_of_unique_materials,
                                unsigned char dev_idx) {
  // Now, all air nodes have a d_pos_temp = 1 (d_mat_temp = 0) and all boundary
  // nodes have a d_pos_temp = 0 with a corresponding material_id in d_mat_temp.
  // Keep the same of the domain Prepare new Kernel call for Count_Air_Neighboring_Nodes_Kernel()
  c_log_msg(LOG_INFO, "voxelizationUtils.cu: to_ParallelFDTD_surface_voxelization() - begin");
  unsigned int num_elems = dim.x*dim.y*dim.z;

  unsigned int threadsPerBlock = 512;
  dim3 block_dim(threadsPerBlock); // block_dim.x = threadsPerBlock; block_dim.y = 1; block_dim.z = 1

  unsigned int numBlocks = (num_elems + threadsPerBlock - 1) / threadsPerBlock;

  // grid_dim.x = grid_dim.y = ceil(sqrt(numBlocks); block_dim.z = 1
  dim3 grid_dim(ceil(sqrt(numBlocks)),ceil(sqrt(numBlocks)));

  c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: Count_Air_Neighboring_Nodes - Block x: %u y: %u z: %u",
      block_dim.x, block_dim.y, block_dim.z);
  c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: Count_Air_Neighboring_Nodes - Grid x: %u y: %u z: %u "
      "(numBlocks = %u; num_elems = %u)",
      grid_dim.x, grid_dim.y, grid_dim.z, numBlocks, num_elems);

  // Prepare data:
  // Get result in a temporary position_matrix and material matrix on device:
  unsigned char* d_pos_temp = (unsigned char*)NULL;
  unsigned char* d_mat_temp = (unsigned char*)NULL;
  d_pos_temp = toDevice<unsigned char>(num_elems, dev_idx);
  d_mat_temp = toDevice<unsigned char>(num_elems, dev_idx);

  unsigned int dim_xy = dim.x*dim.y;
  unsigned int dim_x = dim.x;
  double one_over_dim_xy = 1.0 / (double)dim_xy;
  double one_over_dim_x = 1.0 / (double)dim.x;
  // Build a hash: maps material_id to its index: (vox::HashMap needs a .fromHosttoDevice() method)
  // first get the maximum:
  unsigned char max_ = 0;
  for(size_t index = 0; index < number_of_unique_materials; index++){
    if (max_<unique_materials_ids[index])
      max_=unique_materials_ids[index];
  }
  // Error messages:
  unsigned char* d_error = (unsigned char*)NULL;
  d_error = valueToDevice<unsigned char>(1, 0, dev_idx);
  c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: Count_Air_Neighboring_Nodes - "
      "Max_material_ID = %u, dim_xy = %u, dim_x = %u", max_, dim_xy, dim_x);

  Count_Air_Neighboring_Nodes_Kernel<<<grid_dim, block_dim>>>(*d_pos + (int)dim.x*dim.y,
                                                              d_pos_temp,
                                                              *d_mat + (int)dim.x*dim.y,
                                                              max_,
                                                                                            d_mat_temp,
                                                                                            num_elems,
                                                                                            dim_xy,
                                                                                            dim_x,
                                                                                            one_over_dim_xy,
                                                                                            one_over_dim_x,
                                                                                            bit_mask,
                                                              d_error);

  cudasafe(cudaDeviceSynchronize(), "voxelizationUtils.cu: to_ParallelFDTD_surface_voxelization() - cudaDeviceSynchronize after air counts");
  // Now, put the results into the right place (a slightly faster way is to do d_pos = d_pos_temp; after destroymem(d_pos) ):
  cudasafe(cudaMemcpy(*d_pos, d_pos_temp,  num_elems*sizeof(unsigned char), cudaMemcpyDeviceToDevice), "Memcopy");
  cudasafe(cudaMemcpy(*d_mat, d_mat_temp,  num_elems*sizeof(unsigned char), cudaMemcpyDeviceToDevice), "Memcopy");
  unsigned char h_error = 0;
  copyDeviceToHost(1, &h_error, d_error, dev_idx);
  // Free the temporary memory:
  destroyMem(d_error, dev_idx);
  destroyMem(d_pos_temp, dev_idx);
  destroyMem(d_mat_temp, dev_idx);
  return h_error;
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
  voxelizer.setOrientationsOutput(true);
  voxelizer.setDisplace_VoxSpace_dX_2(false);

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

unsigned char voxelizeGeometry_solid(float* vertices,
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
  unsigned char bit_mask = 0x80;

  ///////////
  /// STEP 1) Do the solid voxelization:
  ///////////////////////////////////////////////////////////////////////////////////////////
  start_t = clock();

  c_log_msg(LOG_INFO,
        "voxelizationUtils: voxelizeGeometryToDevice Solid - voxelizeGeometryToDevice - begin");

  vox::Voxelizer<vox::LongNode> voxelizer(vertices,
                                          indices,
                                          number_of_vertices,
                                          number_of_triangles);
  // No need for these steps that will alter the result of a Solid voxelization:
  voxelizer.setOrientationsOutput(false);
  voxelizer.setMaterialOutput(false);
  voxelizer.setDisplace_VoxSpace_dX_2(true);

  ////////// Voxelize to nodes
  c_log_msg(LOG_DEBUG, "voxelizationUtils: voxelizeGeometryToDevice Solid - toNodes ");
  vox::LongNode* nodes = NULL;
  std::vector< vox::NodePointer< vox::LongNode > > node_ptr;
  node_ptr = voxelizer.voxelizeToNodes(voxel_edge);
  nodes = node_ptr.at(0).ptr;

  cudasafe(cudaDeviceSynchronize(),
           "voxelizationUtils: voxelizeGeometryToDevice Solid - cudaDeviceSynchronize after voxelization");

  *voxelization_dim = node_ptr.at(0).dim;
  unsigned int num_elements = (*voxelization_dim).x*(*voxelization_dim).y*(*voxelization_dim).z;

  end_t = clock()-start_t;
  c_log_msg(LOG_INFO,
            "voxelizationUtils.cu: voxelizeGeometryToDevice Solid, voxelization time: %f seconds",
            ((float)end_t/CLOCKS_PER_SEC));
  // Allocate an additional buffer for indexing out of bounds (somehow the zero-padding fails without this one)
  unsigned int buffer =  (*voxelization_dim).x*(*voxelization_dim).y+(*voxelization_dim).x+1;
  (*d_position_idx) = valueToDevice<unsigned char>(num_elements+buffer, (unsigned char)0, 0);
  (*d_material_idx) = valueToDevice<unsigned char>(num_elements+buffer, (unsigned char)0, 0);

  //////////// Translate the node data to vectors:
  // Note that the voxelizer zero-padds the domain with 2 voxels in Y and Z (see host_side.cpp @lines 970-971)
  //  Since removing those lines did not solve the issue, the conversion of nodes to vector will ignore one extra
  //  layer of voxel in Z and Y direction. The regular alternative is to call:
  // nodes2Vectors(nodes, d_position_idx, d_material_idx, *voxelization_dim);
  start_t = clock();

  int cut_voxelized_shapes_by = 1; // In voxels; 1 is normal since we only need one extra layer of voxels surrounding the conservative bounding box of the domain and the voxelizer pads the domain by 2 voxels.
  c_log_msg(LOG_INFO, "voxelizationUtils.cu: nodes2Vectors_new_shape -  begin");
  // Check for overflow:
  if ( ((*voxelization_dim).y > UINT_MAX / (*voxelization_dim).z ) ||
    (*voxelization_dim).x > UINT_MAX / (*voxelization_dim).y*(*voxelization_dim).z ){
    c_log_msg(LOG_ERROR, "voxelizationUtils.cu: Host surface voxelization: "
     "Overflow of number of nodes! Kernels designed to support up to %u nodes", UINT_MAX);
    throw 3;
  }
  unsigned int num_elems_new = (*voxelization_dim).x*((*voxelization_dim).y-1)*((*voxelization_dim).z-1);

  int threadsPerBlock_ = 512; // the _ postfix for variables threadsPerBlock_, block_dim_numBlocks_, grid_dim_ creates useless variables that can be reused afterwards (without the _ at the end)
  dim3 block_dim_(threadsPerBlock_);

  int numBlocks_ = (num_elems_new + threadsPerBlock_ - 1) / threadsPerBlock_;
  dim3 grid_dim_(ceil(sqrt(numBlocks_)),ceil(sqrt(numBlocks_)));

  c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: nodes2Vectors_new_shape - Block x: %u y: %u z: %u",
    block_dim_.x, block_dim_.y, block_dim_.z);
  c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: nodes2Vectors_new_shape - Grid x: %u y: %u z: %u",
    grid_dim_.x, grid_dim_.y, grid_dim_.z);
  int dim_xy_old = (*voxelization_dim).x*(*voxelization_dim).y;
  int dim_x_old = (*voxelization_dim).x;
  int dim_xy_new = (*voxelization_dim).x*((*voxelization_dim).y-cut_voxelized_shapes_by);
  int dim_x_new = (*voxelization_dim).x;
  double one_over_dim_xy_new = 1.0 / (double)dim_xy_new;
  double one_over_dim_x_new = 1.0 / (double)dim_x_new;

  nodes2Vectors_new_shape_Kernel< vox::LongNode ><<<grid_dim_, block_dim_>>>(
                               nodes,
                               *d_position_idx,
                               *d_material_idx,
                               dim_xy_old,
                               dim_x_old,
                               dim_xy_new,
                               dim_x_new,
                               one_over_dim_xy_new,
                               one_over_dim_x_new,
                               num_elems_new);

  (*voxelization_dim).y -= cut_voxelized_shapes_by;
  (*voxelization_dim).z -= cut_voxelized_shapes_by;
  num_elements = (*voxelization_dim).x*(*voxelization_dim).y*(*voxelization_dim).z; // recalculate this:
  end_t = clock()-start_t;

  c_log_msg(LOG_INFO,"voxelizationUtils.cu: nodes2Vectors_new_shape Solid - time: %f seconds",
                     ((float)end_t/CLOCKS_PER_SEC));

  cudasafe(cudaPeekAtLastError(), "voxelizationUtils.cu: voxelizeGeometryToDevice Solid (new shape)"
                                  "- peek before return");

  cudasafe(cudaDeviceSynchronize(), "voxelizationUtils.cu: voxelizeGeometryToDevice Solid (new shape) "
                                    "- cudaDeviceSynchronize at before return");

  c_log_msg(LOG_INFO, "voxelizationUtils.cu: voxelizeGeometryToDevice Solid - voxelization to vectors done. Voxelization dimensions: [%u,%u,%u]", (*voxelization_dim).x, (*voxelization_dim).y, (*voxelization_dim).z);

  // We don't need the nodes anymore - they just occupy memory and 'out of memory might be triggered'
  cudasafe(cudaFree(nodes), "cudaFree nodes");

  ///////////
  /// STEP 2) Zero padd the dimensions for ParallelFDTD - [32, 4, 1]:
  ///////////////////////////////////////////////////////////////////////////////////////////
  uint3 voxelize_for_block_size = make_uint3(32, 4, 1);
  padWithZeros(d_position_idx, d_material_idx, voxelization_dim,
               voxelize_for_block_size.x, voxelize_for_block_size.y,
               voxelize_for_block_size.z);

  c_log_msg(LOG_INFO, "voxelizationUtils.cu: voxelizeGeometryToDevice Solid - zero padding done");
  num_elements = (*voxelization_dim).x*(*voxelization_dim).y*(*voxelization_dim).z;
  // Start the additional required processing on top of a plain solid voxelization:
  ///////////
  /// STEP 3) Invert 0 to 1 and 1 to 0
  ///////////////////////////////////////////////////////////////////////////////////////////
  start_t = clock();
  unsigned int threadsPerBlock = 512;
  dim3 block_dim(threadsPerBlock); // block_dim.x = threadsPerBlock; block_dim.y = 1; block_dim.z = 1
  unsigned int numBlocks = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid_dim(ceil(sqrt(numBlocks)),ceil(sqrt(numBlocks)));

  c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: Solid, prepare call to invert bid() - Block x: %u y: %u z: %u ; Grid x: %u y: %u z: %u",
            block_dim.x, block_dim.y, block_dim.z, grid_dim.x, grid_dim.y, grid_dim.z);
  invert_bid_output_Kernel<<<grid_dim, block_dim>>>(*d_position_idx, num_elements);
  cudasafe(cudaPeekAtLastError(), "voxelizationUtils.cu: Solid invert bid() - peek before return");
  cudasafe(cudaDeviceSynchronize(), "voxelizationUtils.cu: Solid - cudaDeviceSynchronize after bid() invert");

  ///////////
  /// STEP 4) Get materials from conservative surface Voxelization
  ///////////////////////////////////////////////////////////////////////////////////////////
  unsigned int displace_mesh_voxels = 1;
  // Fetch unique material ids from the mesh containting the material indices of each node
  std::vector<unsigned char> unique_material_ids;
  unique_material_ids.assign(materials, materials+number_of_triangles);
  std::sort(unique_material_ids.begin(), unique_material_ids.end());
  std::vector<unsigned char>::iterator it;
  it = std::unique(unique_material_ids.begin(), unique_material_ids.end());
  unique_material_ids.resize(std::distance(unique_material_ids.begin(), it));

  // Some prerequisites:
  if (unique_material_ids.at(0) == 0){
    c_log_msg(LOG_ERROR,
      "voxelizationUtils: voxelizeGeometry_solid - Please don't use material index 0! Reserved for air.");
    throw 10;
  }
  // Add the zero (air material idx):
  unique_material_ids.insert(unique_material_ids.begin(),0);
  number_of_unique_materials += 1;

  /// Get the materials:
  vox::Bounds<uint3>* triangles_BB =  new vox::Bounds<uint3>[number_of_triangles];
  double3* triangles_Normal =  new double3[number_of_triangles];
  double* vertices_local = new double[3*number_of_vertices];
  vox::Bounds<uint3> space_BB_vox;
  space_BB_vox = get_mesh_BB_and_translate_vertices(vertices_local, vertices,
                                                    indices, number_of_triangles,
                                                    number_of_vertices,
                                                    voxel_edge, triangles_BB,
                                                    triangles_Normal);

  unsigned char* h_position_idx = new unsigned char[num_elements]; // this is wasted memory and memory assignment speed ; Sebastian was groggy enough that he could not make the code inside intersect_triangles_surface_Host() work properly without a h_position_idx and had to keep the function here - this should be easy to do.
  unsigned char* h_material_idx = new unsigned char[num_elements];
  memset(h_material_idx, 0, num_elements);

  intersect_triangles_surface_Host(voxel_edge,
                                   vertices_local,
                                   indices,
                                   materials,
                                   triangles_BB,
                                   triangles_Normal,
                                   h_position_idx,
                                   h_material_idx,
                                   number_of_triangles,
                                   (*voxelization_dim).x*(*voxelization_dim).y,
                                   (*voxelization_dim).x,
                                   space_BB_vox,
                                   displace_mesh_voxels,
                                   VOX_CONSERVATIVE);

  cudasafe(cudaMemcpy(*d_material_idx, h_material_idx,
                      num_elements*sizeof(unsigned char),
                      cudaMemcpyHostToDevice), "Memcopy");

  c_log_msg(LOG_DEBUG,"voxelizationUtils.cu: Done surface intersection.");

  // Free host data:
  delete[] vertices_local;
  delete[] triangles_BB;
  delete[] triangles_Normal;
  delete[] h_position_idx;
  delete[] h_material_idx;

  ///////////
  /// STEP 5) Count air nodes (with bit_mask) and set materials
  ///////////////////////////////////////////////////////////////////////////////////////////
  // Start with code from to_parallelFDTD_surface_voxelization
  // Prepare data: put data directly into d_position_idx & d_material_idx. For this,
  // a temporary matrix will be created with two additional Z slices at the bottom
  // and at the top to aid counting in the surrounding layer of air voxels (same trick
  // as in function voxelizeGeometry_surface_Host() )
  buffer =  (*voxelization_dim).x*(*voxelization_dim).y;
  unsigned char * unique_materials_ids = &(unique_material_ids[0]);// easier to work with buffers
  unsigned char* d_pos_temp = (unsigned char*)NULL;
  unsigned char* d_mat_temp = (unsigned char*)NULL;
  d_pos_temp = valueToDevice<unsigned char>(num_elements + 2*buffer, 1, 0);
  d_mat_temp = valueToDevice<unsigned char>(num_elements + 2*buffer, 0, 0);
  // now put the data inside d_pos_temp from d_position_idx at +(int)buffer location:
  cudasafe(cudaMemcpy(d_pos_temp + (int)buffer, *d_position_idx,  num_elements*sizeof(unsigned char), cudaMemcpyDeviceToDevice), "Memcopy");
  cudasafe(cudaMemcpy(d_mat_temp + (int)buffer, *d_material_idx,  num_elements*sizeof(unsigned char), cudaMemcpyDeviceToDevice), "Memcopy");
  // reset d_material_idx (should not affect the simulation but the result of material is clearer!):
  cudasafe(cudaMemset(*d_material_idx, 0, num_elements), "Cuda MemSet");

  unsigned int dim_xy = (*voxelization_dim).x*(*voxelization_dim).y;
  unsigned int dim_x = (*voxelization_dim).x;
  double one_over_dim_xy = 1.0 / (double)dim_xy;
  double one_over_dim_x = 1.0 / (double)(*voxelization_dim).x;
  // Build a HASH: maps material_id to its index: (vox::HashMap needs a .fromHosttoDevice() method) first get the maximum:
  unsigned char max_ = 0;
  for(size_t index = 0; index < number_of_unique_materials; index++)
  {
     if (max_<unique_materials_ids[index])
        max_=unique_materials_ids[index];
  }
  c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: Solid, prepare call to count air nodes () - Block x: %u y: %u z: %u ; Grid x: %u y: %u z: %u",
              block_dim.x, block_dim.y, block_dim.z, grid_dim.x, grid_dim.y, grid_dim.z);
  // KERNEL CALL:
  unsigned char* d_error = (unsigned char*)NULL;
  d_error = valueToDevice<unsigned char>(1, 0, 0);
  c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: Count_Air_Neighboring_Nodes - "
      "Max_material_ID = %u, dim_xy = %u, dim_x = %u", max_, dim_xy, dim_x);
  Count_Air_Neighboring_Nodes_Kernel<<<grid_dim, block_dim>>>(
                                                      d_pos_temp + (int)buffer,
                                                      *d_position_idx,
                                                      d_mat_temp + (int)buffer,
                                                      max_,
                                                      *d_material_idx,
                                                      num_elements,
                                                      dim_xy,
                                                      dim_x,
                                                      one_over_dim_xy,
                                                      one_over_dim_x,
                                                      bit_mask,
                                                      d_error);

  cudasafe(cudaPeekAtLastError(), "voxelizationUtils.cu: Solid count air nodes () - peek before return");
  cudasafe(cudaDeviceSynchronize(), "voxelizationUtils.cu: voxelizeGeometry_solid()  - cudaDeviceSynchronize after air counts");
  // Check errors:
  unsigned char h_error = 0;
  copyDeviceToHost(1, &h_error, d_error, 0);
  if (h_error != 0 ){
    //~d_bid[thread_idx] & 0x1;
    if (h_error & ERR_NO_MATERIAL_FOUND == ERR_NO_MATERIAL_FOUND){
      c_log_msg(LOG_ERROR,"voxelizationUtils.cu: !ERROR %u! Material not found! Solid voxelization differs too much than the surface voxelization! Also a low possibility that a material of a voxel was not found in the hashmap.", h_error);
    }
  }
  // Free the temporary memory:
  destroyMem(d_error, 0);
  destroyMem(d_pos_temp, 0);
  destroyMem(d_mat_temp, 0);
  end_t = clock() - start_t;
  c_log_msg(LOG_INFO,"voxelizationUtils.cu: Solid voxelization post processing, time: %f seconds", ((float)end_t/CLOCKS_PER_SEC));

  c_log_msg(LOG_INFO, "voxelizationUtils.cu: voxelizeGeometryToDevice Solid \n"
                      "- dim x: %d y: %d z: %d num elements %d",
                      (*voxelization_dim).x,
                      (*voxelization_dim).y,
                      (*voxelization_dim).z,
                      num_elements);

  printMemInfo("voxelizeGeometryDevice Solid memory before return", getCurrentDevice());
  return h_error;
}

unsigned char voxelizeGeometry_surface_6_separating(
                                        float* vertices,
                                        unsigned int* indices,
                                        unsigned char* materials,
                                        unsigned int number_of_triangles,
                                        unsigned int number_of_vertices,
                                        unsigned int number_of_unique_materials,
                                        double voxel_edge,
                                        unsigned char** d_position_idx,
                                        unsigned char** d_material_idx,
                                        uint3* voxelization_dim,
                                        long partial_vox_Z_start,
                                        long partial_vox_Z_end,
                                        unsigned char dev_idx) {
  unsigned char bit_mask = 0x80;
  unsigned int displace_mesh_voxels = 1;// Add a layer of air cells around
  enum Surface_Voxelization_Type voxType = VOX_6_SEPARATING;
  uint3 voxelize_for_block_size = make_uint3(32, 4, 1);

  return voxelizeGeometry_surface_Host(vertices,
                                       indices,
                                       materials,
                                       number_of_triangles,
                                       number_of_vertices,
                                       number_of_unique_materials,
                                       voxel_edge,
                                       d_position_idx,
                                       d_material_idx,
                                       voxelization_dim,
                                       bit_mask,
                                       displace_mesh_voxels,
                                       voxType,
                                       voxelize_for_block_size,
                                       partial_vox_Z_start,
                                       partial_vox_Z_end,
                                       dev_idx);
}

unsigned char voxelizeGeometry_surface_conservative(float* vertices,
                                                    unsigned int* indices,
                                                    unsigned char* materials,
                                                    unsigned int number_of_triangles,
                                                    unsigned int number_of_vertices,
                                                    unsigned int number_of_unique_materials,
                                                    double voxel_edge,
                                                    unsigned char** d_position_idx,
                                                    unsigned char** d_material_idx,
                                                    uint3* voxelization_dim,
                                                    long partial_vox_Z_start,
                                                    long partial_vox_Z_end,
                                                    unsigned char dev_idx) {
  unsigned char bit_mask = 0x80;
  unsigned int displace_mesh_voxels = 1;// Add a layer of air cells around
  enum Surface_Voxelization_Type voxType = VOX_CONSERVATIVE;
  uint3 voxelize_for_block_size = make_uint3(32, 4, 1);

  return voxelizeGeometry_surface_Host(vertices,
                                       indices,
                                       materials,
                                       number_of_triangles,
                                       number_of_vertices,
                                       number_of_unique_materials,
                                       voxel_edge,
                                       d_position_idx,
                                       d_material_idx,
                                       voxelization_dim,
                                       bit_mask,
                                       displace_mesh_voxels,
                                       voxType,
                                       voxelize_for_block_size,
                                       partial_vox_Z_start,
                                       partial_vox_Z_end,
                                       dev_idx);
}

uint3 get_Geometry_surface_Voxelization_dims(float* vertices,
                                             unsigned int* indices,
                                             unsigned int number_of_triangles,
                                             unsigned int number_of_vertices,
                                             double voxel_edge) {
  uint3 voxelization_dim;
  // Config, as in voxelizeGeometry_surface_XXX() functions;
  unsigned int displace_mesh_voxels = 1;// Add a layer of air cells around
  uint3 voxelize_for_block_size = make_uint3(32, 4, 1);
  // Make local copy of vertices (otherwise it will translate the mesh)
  double* vertices_local = new double[3*number_of_vertices];
  // Re-use code: safer, although wastes some compuation on triangles_BB
  // and triangles_Normal
  vox::Bounds<uint3>* triangles_BB =  new vox::Bounds<uint3>[number_of_triangles];
  double3* triangles_Normal =  new double3[number_of_triangles];
  vox::Bounds<uint3> space_BB_vox = get_mesh_BB_and_translate_vertices(vertices_local,
      vertices, indices, number_of_triangles, number_of_vertices,
      voxel_edge, triangles_BB, triangles_Normal);
  // Padd the voxelization domain:
  uint3 pad_voxelization_dim = make_uint3(0, 0, 0);
  // "1 +": the geometry is inside BOTH space_BB_vox.min and space_BB_vox.max
  uint3 minimal_voxelization_dim = make_uint3(
      (1 + space_BB_vox.max.x - space_BB_vox.min.x)+ 2*displace_mesh_voxels,
      (1 + space_BB_vox.max.y - space_BB_vox.min.y)+ 2*displace_mesh_voxels,
      (1 + space_BB_vox.max.z - space_BB_vox.min.z)+ 2*displace_mesh_voxels
                 );
  if((minimal_voxelization_dim.x)%voxelize_for_block_size.x != 0)
    pad_voxelization_dim.x = (voxelize_for_block_size.x-((minimal_voxelization_dim.x)%voxelize_for_block_size.x));

  if((minimal_voxelization_dim.y)%voxelize_for_block_size.y != 0)
    pad_voxelization_dim.y = (voxelize_for_block_size.y-((minimal_voxelization_dim.y)%voxelize_for_block_size.y));

  if((minimal_voxelization_dim.z)%voxelize_for_block_size.z != 0)
    pad_voxelization_dim.z = (voxelize_for_block_size.z-((minimal_voxelization_dim.z)%voxelize_for_block_size.z));

  voxelization_dim.x = minimal_voxelization_dim.x + pad_voxelization_dim.x;
  voxelization_dim.y = minimal_voxelization_dim.y + pad_voxelization_dim.y;
  voxelization_dim.z = minimal_voxelization_dim.z + pad_voxelization_dim.z;
  // Free host data:
  delete[] vertices_local;
  delete[] triangles_BB;
  delete[] triangles_Normal;
  return voxelization_dim;
}

unsigned char voxelizeGeometry_surface_Host(float* vertices,
                                            unsigned int* indices,
                                            unsigned char* materials,
                                            unsigned int number_of_triangles,
                                            unsigned int number_of_vertices,
                                            unsigned int number_of_unique_materials,
                                            double voxel_edge,
                                            unsigned char** d_position_idx,
                                            unsigned char** d_material_idx,
                                            uint3* voxelization_dim,
                                            unsigned char bit_mask,
                                            unsigned int displace_mesh_voxels,
                                            enum Surface_Voxelization_Type voxType,
                                            uint3 voxelize_for_block_size,
                                            long partial_vox_Z_start,
                                            long partial_vox_Z_end,
                                            unsigned char dev_idx
                                            ) {
  c_log_msg(LOG_INFO,
  "voxelizationUtils: voxelizeGeometry_surface_Host - voxelizeGeometryToDevice"
      " [%s] - begin", (voxType == 0 ? "CONSERVATIVE" : "6-SEPARATING" ) );
  //{VOX_CONSERVATIVE, VOX_6_SEPARATING};
  if (number_of_triangles <1) {
    c_log_msg(LOG_INFO,
            "voxelizationUtils: voxelizeGeometry_surface_Host - No triangles"
            " to voxelize. Returning.");
    return 0;
  }

  // Make local copy of vertices (otherwise it will translate the mesh)
  double* vertices_local = new double[3*number_of_vertices];
  //memcpy(vertices_local, vertices, 3*number_of_vertices * sizeof(float));

  // Fetch unique material ids from the mesh containting the material indices
  // of each node
  std::vector<unsigned char> unique_material_ids;
  unique_material_ids.assign(materials, materials+number_of_triangles);
  std::sort(unique_material_ids.begin(), unique_material_ids.end());
  std::vector<unsigned char>::iterator it;
  it = std::unique(unique_material_ids.begin(), unique_material_ids.end());
  unique_material_ids.resize(std::distance(unique_material_ids.begin(), it));

  // Some prerequisites:
  if (unique_material_ids.at(0) == 0){
    c_log_msg(LOG_ERROR,
    "voxelizationUtils: voxelizeGeometry_surface_Host - Please don't use"
    " material index 0! Reserved for air.");
    throw 10;
  }
  // Add the zero (air material idx):
  unique_material_ids.insert(unique_material_ids.begin(),0);
  number_of_unique_materials += 1;

  c_log_msg(LOG_DEBUG,
  "voxelizationUtils: voxelizeGeometry_surface_Host - voxelizeGeometryToDevice"
  " - Retrieved unique material ids. number_of_unique_materials = %d; "
  " size of unique_material_ids = %d", number_of_unique_materials,
  unique_material_ids.size());

  // Get bounding box of mesh:
  vox::Bounds<uint3>* triangles_BB =  new vox::Bounds<uint3>[number_of_triangles];
  double3* triangles_Normal =  new double3[number_of_triangles];
  clock_t start_t;
  clock_t end_t;
  start_t = clock();
  // Some useful information:
  // Note that all triangles_BB lie in the non-padded domain, i.e. the first voxel
  // starts at [0,0,0]. However, this voxel will be [displace_mesh_voxels,
  // displace_mesh_voxels,displace_mesh_voxels] in the final d_position_idx or
  // d_material_idx !
  //
  // If you want to manipulate displace_mesh_voxels, do it only before calling
  // voxelizeGeometry_surface_Host(), otherwise things will likely go wrong!
  // 	There will also be two Z layers added at the bottom of the domain and
  //	the top of the domain but this will not affect the aforementioned indexing
  //	(it is used just to count air cells - see some comments below ).
  vox::Bounds<uint3> space_BB_vox = get_mesh_BB_and_translate_vertices(vertices_local,
              vertices, indices, number_of_triangles, number_of_vertices,
              voxel_edge, triangles_BB, triangles_Normal);
  // Add some debugging here:
  c_log_msg(LOG_DEBUG, "voxelizationUtils: voxelizeGeometry_surface_"
         "Host - World BoundingBox: min [%d,%d,%d], max [%d,%d,%d].",
        space_BB_vox.min.x,space_BB_vox.min.y,space_BB_vox.min.z,
        space_BB_vox.max.x,space_BB_vox.max.y,space_BB_vox.max.z);
  ////////////////////////////////////////////////// Partial voxelization:
  bool do_partial_vox = false, remove_first_air_slice = false,
      remove_last_air_slice = false;
  if (partial_vox_Z_start > -1 || partial_vox_Z_end > -1 ){
    do_partial_vox = true;
    // Sanity check:
    if (partial_vox_Z_start > partial_vox_Z_end){
      c_log_msg(LOG_ERROR, "voxelizeGeometry_surface_Host : partial_vox_Z_start"
          " is > partial_vox_Z_end!");
      throw 10;
    }
    if (partial_vox_Z_start > -1){
      remove_first_air_slice = true;
      // Sanity check:
      if (partial_vox_Z_start > space_BB_vox.max.z)
        c_log_msg(LOG_WARNING, "voxelizationUtils: voxelizeGeometry_surface_"
        "Host - given starting Z slice (%d) is bigger than domain(%d)! "
        "Using %d",partial_vox_Z_start,space_BB_vox.max.z,space_BB_vox.max.z);
      space_BB_vox.min.z = min(max((int)(partial_vox_Z_start - displace_mesh_voxels),0),space_BB_vox.max.z);
      if (space_BB_vox.min.z == 0)
        remove_first_air_slice = false;
    }
    if (partial_vox_Z_end > -1){
      remove_last_air_slice = true;
      // Sanity check:
      if (partial_vox_Z_end - displace_mesh_voxels > space_BB_vox.max.z){
        c_log_msg(LOG_WARNING, "voxelizationUtils: voxelizeGeometry_surface_"
        "Host - given ending Z slice (%d) is bigger than domain(%d)! "
        "Using %d",partial_vox_Z_end,space_BB_vox.max.z,space_BB_vox.max.z);
        remove_last_air_slice = false; // We want until end, so no point in removing it
      }
      else
      space_BB_vox.max.z = partial_vox_Z_end - displace_mesh_voxels;

    }
    c_log_msg(LOG_INFO,
            "voxelizationUtils: voxelizeGeometry_surface_Host - Doing a "
            "partial voxelization between Z slices: %d - %d.",
        space_BB_vox.min.z, space_BB_vox.max.z);
  }
  /////////////////////////////////////////////////////////////////////////////
  end_t = clock() - start_t;
  // Padd the voxelization domain:
  uint3 pad_voxelization_dim = make_uint3(0, 0, 0);
  uint3 minimal_voxelization_dim = make_uint3 // "1 +" is because the geometry is
         ( // inside BOTH space_BB_vox.min and space_BB_vox.max
        (1 + space_BB_vox.max.x - space_BB_vox.min.x) + 2*displace_mesh_voxels,
        (1 + space_BB_vox.max.y - space_BB_vox.min.y) + 2*displace_mesh_voxels,
        (1 + space_BB_vox.max.z - space_BB_vox.min.z) + 2*displace_mesh_voxels
       );
  if((minimal_voxelization_dim.x)%voxelize_for_block_size.x != 0)
    pad_voxelization_dim.x = (voxelize_for_block_size.x-((minimal_voxelization_dim.x)%voxelize_for_block_size.x));

  if((minimal_voxelization_dim.y)%voxelize_for_block_size.y != 0)
    pad_voxelization_dim.y = (voxelize_for_block_size.y-((minimal_voxelization_dim.y)%voxelize_for_block_size.y));

  if((minimal_voxelization_dim.z)%voxelize_for_block_size.z != 0)
    pad_voxelization_dim.z = (voxelize_for_block_size.z-((minimal_voxelization_dim.z)%voxelize_for_block_size.z));

  (*voxelization_dim).x = minimal_voxelization_dim.x + pad_voxelization_dim.x;
  (*voxelization_dim).y = minimal_voxelization_dim.y + pad_voxelization_dim.y;
  (*voxelization_dim).z = minimal_voxelization_dim.z + pad_voxelization_dim.z;
  c_log_msg(LOG_DEBUG,
  "voxelizationUtils: voxelizeGeometry_surface -  Processed triangles (time: "
  "%f seconds). Voxelization dimensions: [%d,%d,%d]. Padded by: [%d,%d,%d]",
  (float)end_t/CLOCKS_PER_SEC, (*voxelization_dim).x, (*voxelization_dim).y,
  (*voxelization_dim).z, pad_voxelization_dim.x, pad_voxelization_dim.y,
  pad_voxelization_dim.z );
  // Check for overflow:
  if ( ((*voxelization_dim).y > UINT_MAX / (*voxelization_dim).z ) ||
    (*voxelization_dim).x > UINT_MAX / (*voxelization_dim).y*(*voxelization_dim).z ){
    c_log_msg(LOG_ERROR, "voxelizationUtils.cu: Host surface voxelization - "
     "Overflow of number of nodes! Kernels designed to support up to %u nodes", UINT_MAX);
    throw 3;
  }
  unsigned int num_elements = (*voxelization_dim).x*(*voxelization_dim).y*(*voxelization_dim).z;

  ////////////////////////////////////////////////////////// Host voxelization:
  // For CUDA voxelization, launch CUDA_launch_6_separating_surface_voxelization()
  /// but is slower.
  start_t = clock();
  // Allocate am additional buffer before and after the domain.
  // This trick is used so that when the #_neighboring_air_cells are counted
  // inside Count_Air_Neighboring_Nodes_Kernel(), the domain is surrounded
  // by air cells and no pointer out of bounds (over/under) will occur.
  //   Note the calls with displaced pointers ( +(int)buffer ) when calling
  //   intersect_triangles_6_separating_Host() and later ( +(int)dim.x*dim.y )
  //   when calling Count_Air_Neighboring_Nodes_Kernel() from to_ParallelFDTD_
  //   surface_voxelization().
  unsigned int buffer =  (*voxelization_dim).x*(*voxelization_dim).y;
  unsigned char* h_position_idx = new unsigned char[num_elements + 2*buffer];
  // Set to 1 (boundary has a 0):
  memset(h_position_idx, 1, num_elements + 2*buffer);
  unsigned char* h_material_idx = new unsigned char[num_elements + 2*buffer];
  memset(h_material_idx, 0, num_elements + 2*buffer);// Mat 0 is for air!
  intersect_triangles_surface_Host(
                voxel_edge,
                vertices_local,
                indices,
                materials,
                triangles_BB,
                triangles_Normal,
                h_position_idx + (int)buffer,
                h_material_idx + (int)buffer,
                number_of_triangles,
                (*voxelization_dim).x*(*voxelization_dim).y,
                (*voxelization_dim).x,
                space_BB_vox,
                displace_mesh_voxels,
                voxType);

  // Copy them to device:
  (*d_position_idx) = toDevice(num_elements + 2*buffer, h_position_idx , dev_idx);
  (*d_material_idx) = toDevice(num_elements + 2*buffer, h_material_idx , dev_idx);

  end_t = clock()-start_t;
  c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: Host surface voxelization time:"
      " %f seconds", (float)end_t/CLOCKS_PER_SEC);
  // Free host data:
  delete[] vertices_local;
  delete[] triangles_BB;
  delete[] triangles_Normal;
  delete[] h_position_idx;
  delete[] h_material_idx;
  //////////// Convert data to ParallelFDTD format:
  start_t = clock();

  unsigned char ret = to_ParallelFDTD_surface_voxelization(
                     d_position_idx,
                     d_material_idx,
                     *voxelization_dim,
                     bit_mask,
                     &(unique_material_ids[0]),
                     number_of_unique_materials,
                     dev_idx);

  end_t = clock()-start_t;
  c_log_msg(LOG_INFO,"voxelizationUtils.cu: to_ParallelFDTD_surface_voxelizat"
      "ion()  - time: %f seconds", ((float)end_t/CLOCKS_PER_SEC));

  cudasafe(cudaPeekAtLastError(),
    "voxelizationUtils.cu: to_ParallelFDTD_surface_voxelization - peek "
    "before return");
  cudasafe(cudaDeviceSynchronize(),
    "voxelizationUtils.cu: to_ParallelFDTD_surface_voxelization - "
    "cudaDeviceSynchronize at before return");

  c_log_msg(LOG_INFO, "voxelizationUtils.cu: voxelizeGeometry surface - "
      "voxelization done");

  ////////////////////////////////////////////////// Partial voxelization:
  // The first and last Z air layer has to dissapear:
  unsigned int start_idx = 0, last_idx = 0;
  if (do_partial_vox && remove_first_air_slice){// First slice:
    start_idx = buffer;
    (*voxelization_dim).z -= 1;
    c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: partial voxelizeGeometry "
        "surface - remove first slice!");
  }
  if (do_partial_vox && remove_last_air_slice){// Last slice:
    last_idx = buffer;
    (*voxelization_dim).z -= 1;
    c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: partial voxelizeGeometry "
            "surface - remove last slice!");
  }
  if (do_partial_vox){
  unsigned int mem_size = (num_elements - start_idx - last_idx)*sizeof(unsigned char);
  unsigned char* d_pos_final = (unsigned char*)NULL;
  unsigned char* d_mat_final = (unsigned char*)NULL;
  cudasafe(cudaSetDevice(dev_idx), "voxelizationUtils.cu Partial voxelization"
      " set device before alloc!");
  // BID matrix:
  cudasafe(cudaMalloc((void**)&d_pos_final, mem_size),
      "voxelizationUtils.cu: Allocate BID for partial voxelization!");
  cudasafe(cudaMemcpy(d_pos_final, *d_position_idx + start_idx,  mem_size,
      cudaMemcpyDeviceToDevice), "Memcopy");
  destroyMem(*d_position_idx, dev_idx);
  // MID matrix:
  cudasafe(cudaMalloc((void**)&d_mat_final, mem_size),
        "voxelizationUtils.cu: Allocate MID for partial voxelization!");
  cudasafe(cudaMemcpy(d_mat_final, *d_material_idx + start_idx,  mem_size,
      cudaMemcpyDeviceToDevice), "Memcopy");
  destroyMem(*d_material_idx, dev_idx);
  // Get the correct pointers now:
  *d_position_idx =d_pos_final;
  *d_material_idx = d_mat_final;
  }
  printMemInfo("voxelizationUtils.cu: voxelizeGeometry_surface_Host "
        "Device memory before return", getCurrentDevice());
  c_log_msg(LOG_INFO,
           "voxelizationUtils.cu: voxelizeGeometry_surface_Host - dim x: %u y: "
           "%u z: %u num elements %u", (*voxelization_dim).x,
       (*voxelization_dim).y, (*voxelization_dim).z, num_elements);
  return ret;
}

unsigned char voxelizeGeometrySurfToHost(float* vertices,
                                         unsigned int* indices,
                                         unsigned char* materials,
                                         unsigned int num_triangles,
                                         unsigned int num_vertices,
                                         unsigned int num_unique_materials,
                                         double voxel_edge,
                                         unsigned char** h_position_idx,
                                         unsigned char** h_material_idx,
                                         uint3* voxelization_dim,
                                         unsigned int displace_mesh_voxels,
                                         enum Surface_Voxelization_Type voxType,
                                         uint3 voxelize_for_block_size,
                                         long partial_vox_Z_start,
                                         long partial_vox_Z_end){
  c_log_msg(LOG_INFO, "voxelizationUtils: voxelizeGeometry_surface_Host - "
                      "voxelizeGeometryToDevice [%s] - begin",
                      (voxType == 0 ? "CONSERVATIVE" : "6-SEPARATING" ) );

  if (num_triangles <1) {
   c_log_msg(LOG_INFO,
           "voxelizationUtils: voxelizeGeometry_surface_Host - No triangles"
           " to voxelize. Returning.");
   return 0;
  }

  // Make local copy of vertices (otherwise it will translate the mesh)
  double* vertices_local = new double[3*num_vertices];

  // Fetch unique material ids from the mesh containting the material indices
  // of each node
  std::vector<unsigned char> unique_material_ids;
  unique_material_ids.assign(materials, materials+num_triangles);
  std::sort(unique_material_ids.begin(), unique_material_ids.end());
  std::vector<unsigned char>::iterator it;
  it = std::unique(unique_material_ids.begin(), unique_material_ids.end());
  unique_material_ids.resize(std::distance(unique_material_ids.begin(), it));

  // Some prerequisites:
  if (unique_material_ids.at(0) == 0) {
   c_log_msg(LOG_ERROR,
   "voxelizationUtils: voxelizeGeometry_surface_Host - Please don't use"
   " material index 0! Reserved for air.");
   throw 10;
  }

  // Add the zero (air material idx):
  unique_material_ids.insert(unique_material_ids.begin(),0);
  num_unique_materials += 1;

  c_log_msg(LOG_DEBUG,
  "voxelizationUtils: voxelizeGeometry_surface_Host - voxelizeGeometryToDevice"
  " - Retrieved unique material ids. num_unique_materials = %d; "
  " size of unique_material_ids = %d", num_unique_materials,
  unique_material_ids.size());

  // Get bounding box of mesh:
  vox::Bounds<uint3>* triangles_BB =  new vox::Bounds<uint3>[num_triangles];
  double3* triangles_Normal =  new double3[num_triangles];
  clock_t start_t;
  clock_t end_t;
  start_t = clock();

  vox::Bounds<uint3> space_BB_vox;
  space_BB_vox = get_mesh_BB_and_translate_vertices(vertices_local,
                                                    vertices, indices,
                                                    num_triangles, num_vertices,
                                                    voxel_edge, triangles_BB,
                                                    triangles_Normal);

  c_log_msg(LOG_DEBUG, "voxelizationUtils: voxelizeGeometry_surface_"
        "Host - World BoundingBox: min [%d,%d,%d], max [%d,%d,%d].",
       space_BB_vox.min.x,space_BB_vox.min.y,space_BB_vox.min.z,
       space_BB_vox.max.x,space_BB_vox.max.y,space_BB_vox.max.z);

  /// Partial voxelization:
  bool do_partial_vox = false;
  bool remove_first_air_slice = false;
  bool remove_last_air_slice = false;

  if (partial_vox_Z_start > -1 || partial_vox_Z_end > -1 ) {
   do_partial_vox = true;
   // Sanity check:
   if (partial_vox_Z_start > partial_vox_Z_end){
     c_log_msg(LOG_ERROR, "voxelizeGeometry_surface_Host : partial_vox_Z_start"
         " is > partial_vox_Z_end!");
     throw 10;
   }
   if (partial_vox_Z_start > -1){
     remove_first_air_slice = true;
     // Sanity check:
     if (partial_vox_Z_start > space_BB_vox.max.z)
       c_log_msg(LOG_WARNING, "voxelizationUtils: voxelizeGeometry_surface_"
       "Host - given starting Z slice (%d) is bigger than domain(%d)! "
       "Using %d",partial_vox_Z_start,space_BB_vox.max.z,space_BB_vox.max.z);
     space_BB_vox.min.z = min(max((int)(partial_vox_Z_start - displace_mesh_voxels),0),space_BB_vox.max.z);
     if (space_BB_vox.min.z == 0)
       remove_first_air_slice = false;
   }
   if (partial_vox_Z_end > -1){
     remove_last_air_slice = true;
     // Sanity check:
     if (partial_vox_Z_end - displace_mesh_voxels > space_BB_vox.max.z){
       c_log_msg(LOG_WARNING, "voxelizationUtils: voxelizeGeometry_surface_"
       "Host - given ending Z slice (%d) is bigger than domain(%d)! "
       "Using %d",partial_vox_Z_end,space_BB_vox.max.z,space_BB_vox.max.z);
       remove_last_air_slice = false; // We want until end, so no point in removing it
     }
     else
     space_BB_vox.max.z = partial_vox_Z_end - displace_mesh_voxels;

   }
   c_log_msg(LOG_INFO,
           "voxelizationUtils: voxelizeGeometry_surface_Host - Doing a "
           "partial voxelization between Z slices: %d - %d.",
       space_BB_vox.min.z, space_BB_vox.max.z);
  }
  /////////////////////////////////////////////////////////////////////////////
  end_t = clock() - start_t;
  // Padd the voxelization domain:
  uint3 pad_voxelization_dim = make_uint3(0, 0, 0);

  // "1 +" is because the geometry is inside BOTH
  //space_BB_vox.min and space_BB_vox.max
  uint3 minimal_voxelization_dim = make_uint3(
       (1 + space_BB_vox.max.x - space_BB_vox.min.x) + 2*displace_mesh_voxels,
       (1 + space_BB_vox.max.y - space_BB_vox.min.y) + 2*displace_mesh_voxels,
       (1 + space_BB_vox.max.z - space_BB_vox.min.z) + 2*displace_mesh_voxels);

  if((minimal_voxelization_dim.x)%voxelize_for_block_size.x != 0)
   pad_voxelization_dim.x = (voxelize_for_block_size.x-((minimal_voxelization_dim.x)%voxelize_for_block_size.x));

  if((minimal_voxelization_dim.y)%voxelize_for_block_size.y != 0)
   pad_voxelization_dim.y = (voxelize_for_block_size.y-((minimal_voxelization_dim.y)%voxelize_for_block_size.y));

  if((minimal_voxelization_dim.z)%voxelize_for_block_size.z != 0)
   pad_voxelization_dim.z = (voxelize_for_block_size.z-((minimal_voxelization_dim.z)%voxelize_for_block_size.z));

  (*voxelization_dim).x = minimal_voxelization_dim.x + pad_voxelization_dim.x;
  (*voxelization_dim).y = minimal_voxelization_dim.y + pad_voxelization_dim.y;
  (*voxelization_dim).z = minimal_voxelization_dim.z + pad_voxelization_dim.z;
  c_log_msg(LOG_DEBUG,
  "voxelizationUtils: voxelizeGeometry_surface -  Processed triangles (time: "
  "%f seconds). Voxelization dimensions: [%d,%d,%d]. Padded by: [%d,%d,%d]",
  (float)end_t/CLOCKS_PER_SEC, (*voxelization_dim).x, (*voxelization_dim).y,
  (*voxelization_dim).z, pad_voxelization_dim.x, pad_voxelization_dim.y,
  pad_voxelization_dim.z );

  // Check for overflow:
  if ( ((*voxelization_dim).y > UINT_MAX / (*voxelization_dim).z ) ||
   (*voxelization_dim).x > UINT_MAX / (*voxelization_dim).y*(*voxelization_dim).z ){
   c_log_msg(LOG_ERROR, "voxelizationUtils.cu: Host surface voxelization - "
    "Overflow of number of nodes! Kernels designed to support up to %u nodes", UINT_MAX);
   throw 3;
  }

  mesh_size_t dim_x = voxelization_dim->x;
  mesh_size_t dim_y = voxelization_dim->y;
  mesh_size_t dim_z = voxelization_dim->z;
  mesh_size_t dim_xy = dim_x*dim_y;
  mesh_size_t num_elements = dim_x*dim_y*dim_z;

  c_log_msg(LOG_INFO, "VoxelizationUtils: voxelizeGeometry - "
                      "Allocate mesh domain, domain size: %llu, %llu, %llu",
                      dim_x, dim_y, dim_z);
  start_t = clock();
  // Allocate am additional buffer before and after the domain.
  // This trick is used so that when the #_neighboring_air_cells are counted
  // inside Count_Air_Neighboring_Nodes_Kernel(), the domain is surrounded
  // by air cells and no pointer out of bounds (over/under) will occur.
  //   Note the calls with displaced pointers ( +(int)buffer ) when calling
  //   intersect_triangles_6_separating_Host() and later ( +(int)dim.x*dim.y )
  //   when calling Count_Air_Neighboring_Nodes_Kernel() from to_ParallelFDTD_
  //   surface_voxelization().
  unsigned int buffer =  dim_xy;
  *h_position_idx = (unsigned char*)calloc(num_elements+2*buffer,
                                           sizeof(unsigned char));

  // Set to 0X80 in order to acommodate inplace converstion to boundary
  // values ("Outside geometry" / on the surface will have 0):
  memset(*h_position_idx, 0X80, num_elements + 2*buffer);

  *h_material_idx = (unsigned char*)calloc(num_elements+2*buffer,
                                           sizeof(unsigned char));

  memset(*h_material_idx, 0, num_elements + 2*buffer);

  intersect_triangles_surface_Host(voxel_edge,
                                   vertices_local,
                                   indices,
                                   materials,
                                   triangles_BB,
                                   triangles_Normal,
                                   (*h_position_idx) + buffer,
                                   (*h_material_idx) + buffer,
                                   num_triangles,
                                   dim_xy,
                                   dim_x,
                                   space_BB_vox,
                                   displace_mesh_voxels,
                                   voxType);

  end_t = clock()-start_t;
  c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: Host surface voxelization time:"
     " %f seconds", (float)end_t/CLOCKS_PER_SEC);
  // Free host data:
  delete[] vertices_local;
  delete[] triangles_BB;
  delete[] triangles_Normal;

  unsigned int start_idx = 0, last_idx = 0;
  if (do_partial_vox && remove_first_air_slice){// First slice:
    start_idx = buffer;
    (*voxelization_dim).z -= 1;
    c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: partial voxelizeGeometry "
        "surface - remove first slice!");
  }
  if (do_partial_vox && remove_last_air_slice){// Last slice:
    last_idx = buffer;
    (*voxelization_dim).z -= 1;
    c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: partial voxelizeGeometry "
            "surface - remove last slice!");
  }
  if (do_partial_vox){
    unsigned int mem_size = (num_elements - start_idx - last_idx)*sizeof(unsigned char);
    memcpy(*h_position_idx, *h_position_idx+start_idx, mem_size);
    memcpy(*h_material_idx, *h_material_idx+start_idx, mem_size);
    realloc(*h_position_idx, mem_size);
    realloc(*h_material_idx, mem_size);
  }
  return 0;
}

vox::Bounds<uint3> get_mesh_BB_and_translate_vertices(
                          double* h_local_vertices,
                          const float* h_vertices,
                          unsigned int* h_indices,
                          unsigned int number_of_triangles,
                          unsigned int number_of_vertices,
                          double voxel_edge,
                          vox::Bounds<uint3>* triangles_BB,
                          double3* triangles_Normal
                          ) {
  uint3 WORLD_min_axis_aligned_BB = make_uint3(0, 0, 0); // where the domain starts - will be useful if the domains are partitioned
  double3 WORLD_min_corner = make_double3(WORLD_min_axis_aligned_BB.x * voxel_edge, WORLD_min_axis_aligned_BB.x * voxel_edge, WORLD_min_axis_aligned_BB.x * voxel_edge);
  double3 translate_mesh = make_double3(h_vertices[0], h_vertices[1], h_vertices[2]);
  uint3 tri_indices;
  double3* tri_verts = new double3[3];
  vox::Bounds<uint3> tri_BB_vox, space_BB_vox, voxelization_limits;
  voxelization_limits.min = make_uint3(0,0,0); // no negative voxel numbers please!
  voxelization_limits.max = make_uint3(UINT_MAX, UINT_MAX, UINT_MAX);// we don't know yet the limits :)
  vox::Bounds<double3> tri_BB, space_BB;
  // Go through all triangles and get minimum of vertices:
  for (unsigned int vert_index = 1; vert_index < number_of_vertices; vert_index ++){
    if (h_vertices[vert_index*3] < translate_mesh.x)
      translate_mesh.x = h_vertices[vert_index*3];
    if (h_vertices[vert_index*3+1] < translate_mesh.y)
      translate_mesh.y = h_vertices[vert_index*3+1];
    if (h_vertices[vert_index*3+2] < translate_mesh.z)
      translate_mesh.z = h_vertices[vert_index*3+2];
  }
  // Now translate:
  for (unsigned int vert_index = 0; vert_index < number_of_vertices; vert_index ++){
    h_local_vertices[vert_index*3] = (double)FLT_EPSILON + (double)h_vertices[vert_index*3] - translate_mesh.x;
    h_local_vertices[vert_index*3 + 1] = (double)FLT_EPSILON + (double)h_vertices[vert_index*3 + 1] - translate_mesh.y;
    h_local_vertices[vert_index*3 + 2] = (double)FLT_EPSILON + (double)h_vertices[vert_index*3 + 2] - translate_mesh.z;
  }
  // Assign space bounding box to first triangle:
  tri_indices = make_uint3( h_indices[0], h_indices[1], h_indices[2]);
  tri_verts[0] = make_double3(h_local_vertices[ tri_indices.x*3 ], h_local_vertices[tri_indices.x*3 + 1], h_local_vertices[tri_indices.x*3 + 2]);
  tri_verts[1] = make_double3(h_local_vertices[ tri_indices.y*3 ], h_local_vertices[tri_indices.y*3 + 1], h_local_vertices[tri_indices.y*3 + 2]);
  tri_verts[2] = make_double3(h_local_vertices[ tri_indices.z*3 ], h_local_vertices[tri_indices.z*3 + 1], h_local_vertices[tri_indices.z*3 + 2]);
  vox::getTriangleBounds(tri_verts, space_BB);
  vox::getVoxelBounds_v2(space_BB, WORLD_min_corner, tri_BB_vox, voxel_edge, voxelization_limits);
  triangles_BB[0] = tri_BB_vox;
  triangles_Normal[0] = cross( tri_verts[1] - tri_verts[0], tri_verts[2] - tri_verts[0] );// e0.cross(-e2)
  // Now go through the rest of triangles and get mesh BB:
  for (unsigned int tri_index = 1; tri_index < number_of_triangles; tri_index ++){
    // Get triangle vert indexes:
    tri_indices = make_uint3( h_indices[3*tri_index], h_indices[3*tri_index + 1], h_indices[3*tri_index + 2]);
    // Get triangle vertices:
    tri_verts[0] = make_double3(h_local_vertices[ tri_indices.x*3 ], h_local_vertices[tri_indices.x*3 + 1], h_local_vertices[tri_indices.x*3 + 2]);
    tri_verts[1] = make_double3(h_local_vertices[ tri_indices.y*3 ], h_local_vertices[tri_indices.y*3 + 1], h_local_vertices[tri_indices.y*3 + 2]);
    tri_verts[2] = make_double3(h_local_vertices[ tri_indices.z*3 ], h_local_vertices[tri_indices.z*3 + 1], h_local_vertices[tri_indices.z*3 + 2]);
    // Get BB:
    vox::getTriangleBounds(tri_verts, tri_BB);
    vox::getVoxelBounds_v2(tri_BB, WORLD_min_corner, tri_BB_vox, voxel_edge, voxelization_limits);
    triangles_BB[tri_index] = tri_BB_vox;
    triangles_Normal[tri_index] = cross( tri_verts[1] - tri_verts[0], tri_verts[2] - tri_verts[0] );// e0.cross(-e2)
    // Compare BB_boxes:
    if (tri_BB.min.x < space_BB.min.x) // minimum BB
    space_BB.min.x = tri_BB.min.x;
    if (tri_BB.min.y < space_BB.min.y)
    space_BB.min.y = tri_BB.min.y;
    if (tri_BB.min.z < space_BB.min.z)
    space_BB.min.z = tri_BB.min.z;
    if (tri_BB.max.x > space_BB.max.x) // maximum BB
    space_BB.max.x = tri_BB.max.x;
    if (tri_BB.max.y > space_BB.max.y)
    space_BB.max.y = tri_BB.max.y;
    if (tri_BB.max.z > space_BB.max.z)
    space_BB.max.z = tri_BB.max.z;
  }
  delete[] tri_verts;
  // Convert to voxels:
  vox::getVoxelBounds_v2(space_BB, WORLD_min_corner, space_BB_vox, voxel_edge, voxelization_limits);
  c_log_msg(LOG_DEBUG,
            "voxelizationUtils: voxelizeGeometry_surface - Translated vertices by [%f,%f,%f] to a BB_min[%f,%f,%f] and BB_max[%f,%f,%f].", -translate_mesh.x, -translate_mesh.y, -translate_mesh.z, space_BB.min.x, space_BB.min.y, space_BB.min.z, space_BB.max.x, space_BB.max.y, space_BB.max.z );
  return space_BB_vox;
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

template<class Node>
__global__ void nodes2Vectors_new_shape_Kernel(Node* nodes,
                                               unsigned char* d_position_idx_ptr,
                                               unsigned char* d_material_idx_ptr,
                                               const int dim_xy_old,
                                               const int dim_x_old,
                                               const int dim_xy_new,
                                               const int dim_x_new,
                                               const double one_over_dim_xy_new,
                                               const double one_over_dim_x_new,
                                               unsigned int num_elems_new) {
  int block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
  int thread_idx = block_idx * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

  if(thread_idx < num_elems_new) {
  // Get current cell x,y,z in the new dimensions of the mesh
  std::div_t div_for_z = double_modulo(thread_idx, dim_xy_new, one_over_dim_xy_new);
  int z_new = div_for_z.quot;
  std::div_t div_for_y = double_modulo(div_for_z.rem, dim_x_new, one_over_dim_x_new);
  int y_new = div_for_y.quot;
  int x_new = div_for_y.rem;

  // For the nodes, convert the current position to old coordinates:
    d_position_idx_ptr[thread_idx] = nodes[z_new*dim_xy_old + y_new*dim_x_old + x_new].bid();
    d_material_idx_ptr[thread_idx] = nodes[z_new*dim_xy_old + y_new*dim_x_old + x_new].mat();
  }
}

__global__ void nodes2Vectors_surface_only_Kernel(vox::SurfaceNode* nodes,
                                                  vox::HashMap surface_nodes_HM,
                                                  unsigned char* d_K, // boundary_matrix
                                                  unsigned char* d_B, // material_matrix
                                                  unsigned int num_elems) {
  int idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int surface_node_idx;

  if(idx < num_elems) {
  surface_node_idx = surface_nodes_HM.get(idx);
  if (surface_node_idx != UINT32_MAX) // from tests/src/main.cpp @ l.667
    d_K[idx] = nodes[surface_node_idx].bid(); // Get boundary_id from the SurfaceNode
    // Apparently there are air nodes caught in the HashMap and these don't
    // have a material (allocated?) so an 'unspecified launch failure' is thrown
    if ( d_K[idx] != 0 )
      d_B[idx] = nodes[surface_node_idx].material;
      // from tests/src/main.cpp @ l.676 [get material_id from the
      //SurfaceNode/Volume node always returns 0 - node_types.h @ line 917
  }
}

__global__ void nodes2Vectors_prepare_surface_data_Kernel(
                                                vox::SurfaceNode* nodes,
                                                vox::HashMap surface_nodes_HM,
                                                unsigned char* d_K,
                                                unsigned char* d_B,
                                                unsigned int num_elems) {
  int idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int surface_node_idx;

  if(idx < num_elems) {
    surface_node_idx = surface_nodes_HM.get(idx);
    if (surface_node_idx != UINT32_MAX){ // from tests/src/main.cpp @ l.667
    // Leave the b_id to 0 for surfaces, but put a 1 ni b_id where air is
    if ( nodes[surface_node_idx].bid() != 0 ) {
      // Apparently there are air nodes caught in the HashMap and these don't
      // have a material (allocated?) so an 'unspecified launch failure' is thrown

      d_B[idx] = nodes[surface_node_idx].material;
      // from tests/src/main.cpp @ l.676 [get material_id from the
      //SurfaceNode/Volume node always returns 0 - node_types.h @ line 917
    }
    else {
      d_K[idx] = 1;
    }
    }
    else {
    d_K[idx] = 1;
    }
    }
}

__global__ void invert_bid_output_Kernel(unsigned char* d_bid,
                                         unsigned int num_elems) {
  int block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
  int thread_idx = block_idx * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

  // Just invert the last bit:
  if(thread_idx < num_elems)
    d_bid[thread_idx] = ~d_bid[thread_idx] & 0x1;
}

__global__ void Count_Air_Neighboring_Nodes_Kernel(
                                    const unsigned char* __restrict d_Bid_voxelizer_in,
                                    unsigned char* d_Bid_out,
                                    const unsigned char* __restrict d_Mat_voxelizer_in,
                                    const unsigned char max_mat_id,
                                    unsigned char* d_Mat_out,
                                    const unsigned int num_elems,
                                    const unsigned int dim_xy,
                                    const unsigned int dim_x,
                                    const double one_over_dim_xy,
                                    const double one_over_dim_x,
                                    const unsigned char bit_mask,
                                    unsigned char* d_error) {
  unsigned int block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
  unsigned int thread_idx = block_idx * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

  //if( (thread_idx >= dim_xy) && (thread_idx< (num_elems - dim_xy)) && (d_Bid_voxelizer_in[thread_idx] != 0)) {
  if( (thread_idx< num_elems ) && (d_Bid_voxelizer_in[thread_idx] != 0) ) {
    // Get current cell x,y,z in the dimensions of the mesh (not the CUDA grid block)
    std::ldiv_t div_for_z = udouble_modulo(thread_idx, dim_xy, one_over_dim_xy);
    long z = div_for_z.quot;
    std::ldiv_t div_for_y = udouble_modulo(div_for_z.rem, dim_x, one_over_dim_x);
    long y = div_for_y.quot;
    long x = div_for_y.rem;
    // Sum 6 neighbors:
    d_Bid_out[z*dim_xy + y*dim_x + x] = (d_Bid_voxelizer_in[z*dim_xy + y*dim_x + (x + 1) ] +
                                    d_Bid_voxelizer_in[z*dim_xy + y*dim_x + (x - 1) ] +
                      d_Bid_voxelizer_in[z*dim_xy + (y + 1)*dim_x + x ] +
                      d_Bid_voxelizer_in[z*dim_xy + (y - 1)*dim_x + x ] +
                      d_Bid_voxelizer_in[(z + 1)*dim_xy + y*dim_x + x ] +
                      d_Bid_voxelizer_in[(z - 1)*dim_xy + y*dim_x + x ]) | bit_mask;

    //////////////////////////////////////////////
    // Materials: Rule, set the dominant material.
    //////////////////////////////////////////////
    if (d_Bid_out[z*dim_xy + y*dim_x + x] < (6 | bit_mask))// Don't need to do this for air cells:
    {
      // If a material is set at the current index, then leave it like that:
      // (this rule is for solid voxelization; for surface voxelization,
      // d_Mat_voxelizer_in should be 0 by construction)
      if (d_Mat_voxelizer_in[thread_idx] != 0)
        d_Mat_out[thread_idx] = d_Mat_voxelizer_in[thread_idx];
      else
      {
      // New memory optimized code:
      // Can be further optimized: when +6 was added quit the loop!
        //  or to check if mat_id_idx is in the unique mat vector
      //printf("%d ## KERNEL DBG: new code start...\n", thread_idx);
      bool found_mat_ = false;
      unsigned char max_ = 0, cnt_id = 0, dominant_mat_id_ = 255;
      // Start at 1 since 0 is reserved for air:
      for (unsigned char mat_id_idx = 1; mat_id_idx<=max_mat_id; mat_id_idx++){
        cnt_id = 0;
        if ( mat_id_idx == d_Mat_voxelizer_in[z*dim_xy + y*dim_x + (x + 1)] )
          cnt_id += 1;
        if ( mat_id_idx == d_Mat_voxelizer_in[z*dim_xy + y*dim_x + (x - 1)] )
          cnt_id += 1;
        if ( mat_id_idx == d_Mat_voxelizer_in[z*dim_xy + (y + 1)*dim_x + x] )
          cnt_id += 1;
        if ( mat_id_idx == d_Mat_voxelizer_in[z*dim_xy + (y - 1)*dim_x + x] )
          cnt_id += 1;
        if ( mat_id_idx == d_Mat_voxelizer_in[(z + 1)*dim_xy + y*dim_x + x] )
          cnt_id += 1;
        if ( mat_id_idx == d_Mat_voxelizer_in[(z - 1)*dim_xy + y*dim_x + x] )
          cnt_id += 1;
        if (cnt_id>max_){// Check max:
          max_ = cnt_id;
          dominant_mat_id_ = mat_id_idx;
          found_mat_ = true;
        }
      }
      d_Mat_out[z*dim_xy + y*dim_x + x] = dominant_mat_id_;
      if (!found_mat_)
        *d_error = *d_error | ERR_NO_MATERIAL_FOUND; // Error
      }
    }
  }
}

__device__ std::div_t double_modulo(const int &x,const int &n, const double &one_over_n) {
  double tmp1;
  std::div_t r;
  tmp1 = __int2double_rn( x );
  tmp1 = tmp1 * one_over_n ;
  r.quot = __double2int_rz( tmp1 ) ;
  r.rem = x - n*r.quot;
  return r;
}

__device__ std::ldiv_t udouble_modulo(const unsigned int &x,const unsigned int &n, const double &one_over_n) {
  double tmp1;
  std::ldiv_t r;
  tmp1 = __uint2double_rn( x );
  tmp1 = tmp1 * one_over_n ;
  r.quot = __double2uint_rz( tmp1 ) ;
  r.rem = x - n*r.quot;
  return r;
}

__device__ std::div_t double_modulo(const int &x, const int &n) {
  double one_over_n = 1.0 / (double)n;
  return double_modulo(x,n, one_over_n);
}

__device__ std::ldiv_t udouble_modulo(const unsigned int &x, const unsigned int &n) {
  double one_over_n = 1.0 / (double)n;
  return udouble_modulo(x,n, one_over_n);
}

void intersect_triangles_surface_Host(const double dX,
                                      const double* h_vertices,
                                      const unsigned int* h_indices,
                                      const unsigned char* h_materials,
                                      const vox::Bounds<uint3>* h_triangles_BB,
                                      const double3* h_triangles_Normal,
                                      unsigned char*  h_Bid_out,
                                      unsigned char* h_Mat_out,
                                      const unsigned int num_triangles,
                                      const int dim_xy,
                                      const int dim_x,
                                      vox::Bounds<uint3> space_BB_vox,
                                      unsigned int displace_mesh_voxels,
                                      enum Surface_Voxelization_Type voxType ){
  // TODO: ?Maybe? One optimization method is to precalculate all the edge normals for each triangle and pass to the test
  uint3 tri_indices;
  double3* tri_verts = new double3[3];
  double2 edge_2D_normal;
  double edge_2D_distance;
  double dX_over_2 = dX * 0.5;
  double3 max_point = make_double3(dX, dX,dX);

  // looks more efficient to put the if outside and make two loops:
  if (voxType == VOX_CONSERVATIVE)
  {
    for(unsigned int tri_idx=0; tri_idx<num_triangles; tri_idx++)
      {
        // Get triangle vert indexes:
        tri_indices = make_uint3( h_indices[3*tri_idx], h_indices[3*tri_idx + 1], h_indices[3*tri_idx + 2]);

        // Get triangle vertices:
        tri_verts[0] = make_double3(h_vertices[ tri_indices.x*3 ], h_vertices[tri_indices.x*3 + 1], h_vertices[tri_indices.x*3 + 2]);
        tri_verts[1] = make_double3(h_vertices[ tri_indices.y*3 ], h_vertices[tri_indices.y*3 + 1], h_vertices[tri_indices.y*3 + 2]);
        tri_verts[2] = make_double3(h_vertices[ tri_indices.z*3 ], h_vertices[tri_indices.z*3 + 1], h_vertices[tri_indices.z*3 + 2]);
        // Iterate through all the voxels in the tri_BB:
        for (unsigned int voxel_x = h_triangles_BB[tri_idx].min.x; voxel_x <= h_triangles_BB[tri_idx].max.x; voxel_x++)
          for (unsigned int voxel_y = h_triangles_BB[tri_idx].min.y; voxel_y <= h_triangles_BB[tri_idx].max.y; voxel_y++)
            for (unsigned int voxel_z = h_triangles_BB[tri_idx].min.z; voxel_z <= h_triangles_BB[tri_idx].max.z; voxel_z++)
            {
              if ( test_triangle_box_intersection_double_opt(voxel_x, voxel_y, voxel_z,
                  tri_verts, h_triangles_Normal[tri_idx], dX, edge_2D_normal,
                  edge_2D_distance, max_point) && ((voxel_z >= space_BB_vox.min.z)&&
                      voxel_z <= space_BB_vox.max.z))
              {
                h_Bid_out[(voxel_z+displace_mesh_voxels-space_BB_vox.min.z)*dim_xy+
                    (voxel_y+displace_mesh_voxels)*dim_x +
                    (voxel_x+displace_mesh_voxels)] = 0;
                // ! WARNING: cannot guarantee a rule if many triangles with different materials intersect a voxel
                h_Mat_out[(voxel_z+displace_mesh_voxels-space_BB_vox.min.z)*dim_xy+
                    (voxel_y+displace_mesh_voxels)*dim_x +
                    (voxel_x+displace_mesh_voxels)] = h_materials[tri_idx];
              }
           }

      }
  } else if (voxType == VOX_6_SEPARATING)
  {
    for(unsigned int tri_idx=0; tri_idx<num_triangles; tri_idx++)
      {
        // Get triangle vert indexes:
        tri_indices = make_uint3( h_indices[3*tri_idx], h_indices[3*tri_idx + 1], h_indices[3*tri_idx + 2]);

        // Get triangle vertices:
        tri_verts[0] = make_double3(h_vertices[ tri_indices.x*3 ], h_vertices[tri_indices.x*3 + 1], h_vertices[tri_indices.x*3 + 2]);
        tri_verts[1] = make_double3(h_vertices[ tri_indices.y*3 ], h_vertices[tri_indices.y*3 + 1], h_vertices[tri_indices.y*3 + 2]);
        tri_verts[2] = make_double3(h_vertices[ tri_indices.z*3 ], h_vertices[tri_indices.z*3 + 1], h_vertices[tri_indices.z*3 + 2]);
        // Iterate through all the voxels in the tri_BB:
        for (unsigned int voxel_x = h_triangles_BB[tri_idx].min.x; voxel_x <= h_triangles_BB[tri_idx].max.x; voxel_x++)
          for (unsigned int voxel_y = h_triangles_BB[tri_idx].min.y; voxel_y <= h_triangles_BB[tri_idx].max.y; voxel_y++)
            for (unsigned int voxel_z = h_triangles_BB[tri_idx].min.z; voxel_z <= h_triangles_BB[tri_idx].max.z; voxel_z++)
            {
              if ( test_triangle_box_intersection_6_sep_double_opt(voxel_x, voxel_y, voxel_z,
                  tri_verts, h_triangles_Normal[tri_idx], dX, dX_over_2, edge_2D_normal,
                  edge_2D_distance, max_point) && ((voxel_z >= space_BB_vox.min.z)&&
                      voxel_z <= space_BB_vox.max.z))
              {
                h_Bid_out[(voxel_z+displace_mesh_voxels-space_BB_vox.min.z)*dim_xy+
                    (voxel_y+displace_mesh_voxels)*dim_x+
                    (voxel_x+displace_mesh_voxels)] = 0;
                // ! WARNING: cannot guarantee a rule if many triangles with different materials intersect a voxel
                h_Mat_out[(voxel_z+displace_mesh_voxels-space_BB_vox.min.z)*dim_xy+
                    (voxel_y+displace_mesh_voxels)*dim_x +
                    (voxel_x+displace_mesh_voxels)] = h_materials[tri_idx];
              }
           }
      }
  }
  // Free data:
  delete[] tri_verts;
}

__host__ bool test_triangle_box_intersection_6_sep_double_opt(
                                                    const int &voxel_index_X,
                                                    const int &voxel_index_Y,
                                                    const int &voxel_index_Z,
                                                    const double3* tri_verts,
                                                    const double3 &tri_normal,
                                                    const double &dX,
                                                    const double &dX_over_2,
                                                    double2 &edge_2D_normal,
                                                    double &edge_2D_distance,
                                                    double3 &max_point
                                                    ) {
    double3 voxel_min_corner_WORLD = make_double3(dX*voxel_index_X, dX*voxel_index_Y, dX*voxel_index_Z);
    ///////////// Modified plane-box test:  //////
    // Argmax:
    double normal_dominant_value = fmaxf((double)fmaxf(abs(tri_normal.x), abs(tri_normal.y)), abs(tri_normal.z));
    double d_1 = dot( tri_normal, 0.5 * max_point - tri_verts[0]) + dX_over_2 * normal_dominant_value;
    double d_2 = dot( tri_normal, 0.5 * max_point - tri_verts[0]) - dX_over_2 * normal_dominant_value;
    double n_dot_p = dot( tri_normal, voxel_min_corner_WORLD);

    bool result = (n_dot_p + d_1)*(n_dot_p + d_2) <= 0.0;
    ///////////// 2D projection test: //////
    for ( int i = 0; i < 3; i++ ){
        // 1] COMPUTE Edge normals:
    // XY:
        edge_2D_normal = make_double2( tri_verts[i].y - tri_verts[(i + 1)%3].y, tri_verts[(i + 1)%3].x - tri_verts[i].x ) ;
        edge_2D_normal = copysign( 1.0 , tri_normal.z ) * edge_2D_normal; // copysign exists in CUDA also
    edge_2D_distance = (edge_2D_normal.x * (dX_over_2 - tri_verts[i].x) + edge_2D_normal.y * (dX_over_2 - tri_verts[i].y)) + dX_over_2 * fmaxf( abs(edge_2D_normal.x), abs(edge_2D_normal.y) ) ;
    // Test:
    result &= ( dot(edge_2D_normal, make_double2(voxel_min_corner_WORLD.x, voxel_min_corner_WORLD.y)) + edge_2D_distance) >= 0.0;

    // ZX:
        edge_2D_normal = make_double2( tri_verts[i].x - tri_verts[(i + 1)%3].x, tri_verts[(i + 1)%3].z - tri_verts[i].z ) ;
        edge_2D_normal = copysign( 1.0 , tri_normal.y ) * edge_2D_normal;
    edge_2D_distance = (edge_2D_normal.x * (dX_over_2 - tri_verts[i].z) + edge_2D_normal.y * (dX_over_2 - tri_verts[i].x)) + dX_over_2 * fmaxf( abs(edge_2D_normal.x), abs(edge_2D_normal.y) ) ;
    // Test:
    result &= ( dot(edge_2D_normal, make_double2(voxel_min_corner_WORLD.z, voxel_min_corner_WORLD.x)) + edge_2D_distance) >= 0.0;

    // YZ:
        edge_2D_normal = make_double2( tri_verts[i].z - tri_verts[(i + 1)%3].z, tri_verts[(i + 1)%3].y - tri_verts[i].y ) ;
        edge_2D_normal = copysign( 1.0 , tri_normal.x ) * edge_2D_normal;
    edge_2D_distance = (edge_2D_normal.x * (dX_over_2 - tri_verts[i].y) + edge_2D_normal.y * (dX_over_2 - tri_verts[i].z)) + dX_over_2 * fmaxf( abs(edge_2D_normal.x), abs(edge_2D_normal.y) ) ;
        // Test:
    result &= ( dot(edge_2D_normal, make_double2(voxel_min_corner_WORLD.y, voxel_min_corner_WORLD.z)) + edge_2D_distance) >= 0.0;
    }
    // Exit:
    //return tri_plane_intersects_voxel & testResult_XY & testResult_ZX & testResult_YZ;
  return result;
}

__host__ bool test_triangle_box_intersection_double_opt(
                                                    const int &voxel_index_X,
                                                    const int &voxel_index_Y,
                                                    const int &voxel_index_Z,
                                                    const double3* tri_verts,
                                                    const double3 &tri_normal,
                                                    const double &dX,
                                                    double2 &edge_2D_normal,
                                                    double &edge_2D_distance,
                                                    double3 &max_point) {
    double3 voxel_min_corner_WORLD = make_double3(voxel_index_X*dX, voxel_index_Y*dX, voxel_index_Z*dX);

    //////////////////////////////////// Calculate stuff for the test: (float2 * float2 -> element-wise)
    ///////////// Plane-box test:  //////
    double3 triangle_normal_mask = make_double3( fmaxf(0.0, copysign( 1.0,tri_normal.x) ), fmaxf(0.0, copysign( 1.0,tri_normal.y) ), fmaxf(0.0, copysign( 1.0,tri_normal.z) ) ); // for std::copysign, there is CUDA copysign
    double3 critical_point = triangle_normal_mask * max_point;
  //double3 critical_point = make_double3( (tri_normal.x >= 0.0 ? 1.0 : 0.0) * max_point.x, (tri_normal.y >= 0.0 ? 1.0 : 0.0) * max_point.y, (tri_normal.z >= 0.0 ? 1.0 : 0.0) * max_point.z);
    double d_1 = dot(tri_normal, critical_point - tri_verts[0]);
    double d_2 = dot(tri_normal, max_point - critical_point - tri_verts[0]);
    double n_dot_p = dot(tri_normal, voxel_min_corner_WORLD);

  double eps = 0.0; // 1E-9
    bool result = (n_dot_p + d_1)*(n_dot_p + d_2) <= eps;// 0.0
    ///////////// 2D projection test: //////
    for ( int i = 0; i < 3; i++ ){
        // 1] COMPUTE Edge normals:
      // XY:
      edge_2D_normal = make_double2( tri_verts[i].y - tri_verts[(i + 1)%3].y, tri_verts[(i + 1)%3].x - tri_verts[i].x ) ;
      edge_2D_normal = copysign( 1.0 , tri_normal.z ) * edge_2D_normal;
      edge_2D_distance =  -(edge_2D_normal.x * tri_verts[i].x + edge_2D_normal.y * tri_verts[i].y) + fmaxf( 0.0, dX * edge_2D_normal.x ) + fmaxf( 0.0, dX * edge_2D_normal.y ) ;
      result &= ( dot(edge_2D_normal, make_double2(voxel_min_corner_WORLD.x, voxel_min_corner_WORLD.y)) + edge_2D_distance) >= -eps;// 0.0

      // ZX:
      edge_2D_normal = make_double2( tri_verts[i].x - tri_verts[(i + 1)%3].x, tri_verts[(i + 1)%3].z - tri_verts[i].z ) ;
      edge_2D_normal = copysign( 1.0 , tri_normal.y ) * edge_2D_normal;
      edge_2D_distance = -(edge_2D_normal.x * tri_verts[i].z + edge_2D_normal.y * tri_verts[i].x) + fmaxf( 0.0, dX * edge_2D_normal.x ) + fmaxf( 0.0, dX * edge_2D_normal.y ) ;
      result &= ( dot(edge_2D_normal, make_double2(voxel_min_corner_WORLD.z, voxel_min_corner_WORLD.x)) + edge_2D_distance) >= -eps;

      // YZ:
      edge_2D_normal = make_double2( tri_verts[i].z - tri_verts[(i + 1)%3].z, tri_verts[(i + 1)%3].y - tri_verts[i].y ) ;
      edge_2D_normal = copysign( 1.0 , tri_normal.x ) * edge_2D_normal;
      edge_2D_distance = -(edge_2D_normal.x * tri_verts[i].y + edge_2D_normal.y * tri_verts[i].z) + fmaxf( 0.0, dX * edge_2D_normal.x ) + fmaxf( 0.0, dX * edge_2D_normal.y ) ;
      result &= ( dot(edge_2D_normal, make_double2(voxel_min_corner_WORLD.y, voxel_min_corner_WORLD.z)) + edge_2D_distance) >= -eps;
    }
    return result;
}

void CUDA_launch_6_separating_surface_voxelization(
                                  uint3* h_voxelization_dim,
                                  unsigned char** d_position_idx,
                                  unsigned char** d_material_idx,
                                  float* h_vertices,
                                  unsigned int* h_indices,
                                  unsigned char* h_materials,
                                  const unsigned int number_of_triangles,
                                  unsigned int number_of_vertices,
                                  unsigned int num_elements,
                                  double voxel_edge,
                                  vox::Bounds<uint3>* h_triangles_BB,
                                  double3* h_triangles_Normal) {
    clock_t start_t;
    clock_t end_t;
    // Allocate am additional buffer for indexing out of bounds
    unsigned int buffer =  (*h_voxelization_dim).x*(*h_voxelization_dim).y+(*h_voxelization_dim).x+1;
    (*d_position_idx) = valueToDevice<unsigned char>(num_elements+buffer, (unsigned char)1, 0);
    (*d_material_idx) = valueToDevice<unsigned char>(num_elements+buffer, (unsigned char)0, 0);

    // Mesh information to device:
    float* d_vertices = (float*)NULL;
    unsigned int* d_indices = (unsigned int*)NULL;
    unsigned char* d_materials = (unsigned char*)NULL;
    vox::Bounds<uint3>* d_triangles_BB = (vox::Bounds<uint3>*)NULL;
    double3* d_triangles_Normal = (double3*)NULL;
    d_vertices = toDevice(number_of_vertices * 3, h_vertices, 0);// 0 is the device_id
    d_indices = toDevice(number_of_triangles * 3 , h_indices, 0);
    d_materials = toDevice(number_of_triangles , h_materials, 0);
    d_triangles_BB = toDevice(number_of_triangles , h_triangles_BB, 0);
    d_triangles_Normal = toDevice(number_of_triangles , h_triangles_Normal, 0);
    /////////////// Voxelize 6-separating:
    start_t = clock();

    int threadsPerBlock = 512;
    dim3 block_dim(threadsPerBlock); // block_dim.x = threadsPerBlock; block_dim.y = 1; block_dim.z = 1
    int numBlocks = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    // grid_dim.x = grid_dim.y = ceil(sqrt(numBlocks); block_dim.z = 1
    dim3 grid_dim(ceil(sqrt(numBlocks)),ceil(sqrt(numBlocks)));
    c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: prepare call for 6-separating surface voxelization - Block x: %u y: %u z: %u",
        block_dim.x, block_dim.y, block_dim.z);
    c_log_msg(LOG_DEBUG, "voxelizationUtils.cu: prepare call for 6-separating surface voxelization - Grid x: %u y: %u z: %u",
        grid_dim.x, grid_dim.y, grid_dim.z);

    // For each triangle, voxelize:
    intersect_triangle_6_separating<<<grid_dim, block_dim>>>(voxel_edge,
        d_vertices,
        d_indices,
        d_materials,
        d_triangles_BB,
        d_triangles_Normal,
        *d_position_idx,
        *d_material_idx,
        number_of_triangles,
        num_elements,
        (*h_voxelization_dim).x*(*h_voxelization_dim).y,
        (*h_voxelization_dim).x);

    cudasafe(cudaPeekAtLastError(),
      "voxelizationUtils.cu: voxelizeGeometry_surface_6_separating - peek before return");
    cudasafe(cudaDeviceSynchronize(), "voxelizationUtils.cu: voxelizeGeometry_surface_6_separating - cudaDeviceSynchronize after surface voxelization.");
    end_t = clock()-start_t;

    c_log_msg(LOG_INFO,"voxelizationUtils.cu: Voxelized surface 6-separating  - time: %f seconds",
            ((float)end_t/CLOCKS_PER_SEC));
    // Free pointers from :
    destroyMem(d_triangles_BB);
    destroyMem(d_triangles_Normal);
}

__global__ void intersect_triangle_6_separating(
                    const double dX,
                    const float* __restrict d_vertices,
                    const unsigned int* __restrict d_indices,
                    const unsigned char* __restrict d_materials,
                    const vox::Bounds<uint3>* __restrict d_triangles_BB,
                    const double3* __restrict d_triangles_Normal,
                    unsigned char*  d_Bid_out,
                    unsigned char* d_Mat_out,
                    const unsigned int num_triangles,
                    const unsigned int num_elements,
                    const int dim_xy,
                    const int dim_x) {
  int block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
  int thread_idx = block_idx * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
  if( thread_idx < num_triangles) {
    //printf("[%d] - num_triangles %d\n", thread_idx, num_triangles);
    // Get triangle vert indexes:
    uint3 tri_indices = make_uint3( d_indices[3*thread_idx], d_indices[3*thread_idx + 1], d_indices[3*thread_idx + 2]);
    // Get triangle vertices:
    float3* tri_verts = new float3[3];
    tri_verts[0] = make_float3(d_vertices[ tri_indices.x*3 ], d_vertices[tri_indices.x*3 + 1], d_vertices[tri_indices.x*3 + 2]);
    tri_verts[1] = make_float3(d_vertices[ tri_indices.y*3 ], d_vertices[tri_indices.y*3 + 1], d_vertices[tri_indices.y*3 + 2]);
    tri_verts[2] = make_float3(d_vertices[ tri_indices.z*3 ], d_vertices[tri_indices.z*3 + 1], d_vertices[tri_indices.z*3 + 2]);

    // Get the dimension of the box of voxels (will be useful in the future if sub-kernel will be called!).
    //uint3 voxelDiff = vox_BB.max - vox_BB.min;

    // Iterate through all the voxels (this should be wrong... it should start at minimum or something like adding it later):
    for (unsigned int voxel_x = d_triangles_BB[thread_idx].min.x; voxel_x <= d_triangles_BB[thread_idx].max.x; voxel_x++)
      for (unsigned int voxel_y = d_triangles_BB[thread_idx].min.y; voxel_y <= d_triangles_BB[thread_idx].max.y; voxel_y++)
        for (unsigned int voxel_z = d_triangles_BB[thread_idx].min.z; voxel_z <= d_triangles_BB[thread_idx].max.z; voxel_z++){
          if ( test_triangle_box_intersection_6_sep_double(voxel_x, voxel_y, voxel_z, tri_verts, d_triangles_Normal[thread_idx], dX) ){
            d_Bid_out[voxel_z*dim_xy + voxel_y*dim_x + voxel_x] = 0;
            d_Mat_out[voxel_z*dim_xy + voxel_y*dim_x + voxel_x] = d_materials[thread_idx];
          }
       }
    // Free data:
    delete[] tri_verts;
  }
}

__device__ __host__ bool test_triangle_box_intersection_6_sep(
                                                    const int voxel_index_X,
                                                    const int voxel_index_Y,
                                                    const int voxel_index_Z,
                                                    const float3* tri_verts,
                                                    const float3 tri_normal,
                                                    const float dX) {
  float epsilon = -1.0e-07;
  //////////////////////////////////// Allocate needed data:
  float2* edge_XY_normals =  new float2[3];
  float * edge_XY_distances = new float[3];
  float2* edge_ZX_normals =  new float2[3];
  float * edge_ZX_distances = new float[3];
  float2* edge_YZ_normals =  new float2[3];
  float * edge_YZ_distances = new float[3];

  float3 max_point = make_float3(dX, dX, dX); // A better name would better be delta_p since max_point would be voxel_min_corner_WORLD + make_float3(dX, dX, dX)
  float3 voxel_min_corner_WORLD = make_float3(voxel_index_X*dX, voxel_index_Y*dX, voxel_index_Z*dX);
  float dX_over_2 = 0.5f * dX;

  bool testResult_XY = true, testResult_ZX = true, testResult_YZ = true;
  //////////////////////////////////// Calculate stuff for the test: (float2 * float2 -> element-wise)
  ///////////// Modified plane-box test:  //////
  // Argmax:
  float3 tri_normal_abs = make_float3( abs(tri_normal.x), abs(tri_normal.y), abs(tri_normal.z) );
  float normal_dominant_value = fmaxf(fmaxf(tri_normal_abs.x, tri_normal_abs.y), tri_normal_abs.z);
  float d_1 = dot(tri_normal, 0.5f * max_point - tri_verts[0]) + dX_over_2 * normal_dominant_value;
  float d_2 = dot(tri_normal, 0.5f * max_point - tri_verts[0]) - dX_over_2 * normal_dominant_value;
  float n_dot_p = dot(tri_normal, voxel_min_corner_WORLD);

  bool tri_plane_intersects_voxel = (n_dot_p + d_1)*(n_dot_p + d_2) <= 0.0;
  ///////////// 2D projection test: //////
  for ( int i = 0; i < 3; i++ ){
      // 1] COMPUTE Edge normals:
      edge_XY_normals[i] = make_float2( tri_verts[i].y - tri_verts[(i + 1)%3].y, tri_verts[(i + 1)%3].x - tri_verts[i].x ) ;
      edge_XY_normals[i] = copysign( 1.0 , tri_normal.z ) * edge_XY_normals[i]; // copysign exists in CUDA also
      edge_ZX_normals[i] = make_float2( tri_verts[i].x - tri_verts[(i + 1)%3].x, tri_verts[(i + 1)%3].z - tri_verts[i].z ) ;
      edge_ZX_normals[i] = copysign( 1.0 , tri_normal.y ) * edge_ZX_normals[i];
      edge_YZ_normals[i] = make_float2( tri_verts[i].z - tri_verts[(i + 1)%3].z, tri_verts[(i + 1)%3].y - tri_verts[i].y ) ;
      edge_YZ_normals[i] = copysign( 1.0 , tri_normal.x ) * edge_YZ_normals[i];
      // Normalize edge normals (NO impact):
      //edge_XY_normals[i] = edge_XY_normals[i] / sqrt( dot( edge_XY_normals[i], edge_XY_normals[i] ) );
      //edge_ZX_normals[i] = edge_ZX_normals[i] / sqrt( dot( edge_ZX_normals[i], edge_ZX_normals[i] ) );
      //edge_YZ_normals[i] = edge_YZ_normals[i] / sqrt( dot( edge_YZ_normals[i], edge_YZ_normals[i] ) );

      // 2] COMPUTE ModfieD Edge distances:
      edge_XY_distances[i] = (edge_XY_normals[i].x * (dX_over_2 - tri_verts[i].x) + edge_XY_normals[i].y * (dX_over_2 - tri_verts[i].y)) + dX_over_2 * fmaxf( abs(edge_XY_normals[i].x), abs(edge_XY_normals[i].y) ) ;
      edge_ZX_distances[i] = (edge_ZX_normals[i].x * (dX_over_2 - tri_verts[i].z) + edge_ZX_normals[i].y * (dX_over_2 - tri_verts[i].x)) + dX_over_2 * fmaxf( abs(edge_ZX_normals[i].x), abs(edge_ZX_normals[i].y) ) ;
      edge_YZ_distances[i] = (edge_YZ_normals[i].x * (dX_over_2 - tri_verts[i].y) + edge_YZ_normals[i].y * (dX_over_2 - tri_verts[i].z)) + dX_over_2 * fmaxf( abs(edge_YZ_normals[i].x), abs(edge_YZ_normals[i].y) ) ;

      // 3] TEST 2D intersections:
      testResult_XY &= ( dot(edge_XY_normals[i], make_float2(voxel_min_corner_WORLD.x, voxel_min_corner_WORLD.y)) + edge_XY_distances[i]) >= epsilon;
      testResult_ZX &= ( dot(edge_ZX_normals[i], make_float2(voxel_min_corner_WORLD.z, voxel_min_corner_WORLD.x)) + edge_ZX_distances[i]) >= epsilon;
      testResult_YZ &= ( dot(edge_YZ_normals[i], make_float2(voxel_min_corner_WORLD.y, voxel_min_corner_WORLD.z)) + edge_YZ_distances[i]) >= epsilon;
  }
  //////////////////////////////////// Free memory:
  delete[] edge_XY_normals;
  delete[] edge_XY_distances;
  delete[] edge_ZX_normals;
  delete[] edge_ZX_distances;
  delete[] edge_YZ_normals;
  delete[] edge_YZ_distances;

  // Exit:
  return tri_plane_intersects_voxel & testResult_XY & testResult_ZX & testResult_YZ;
}

__device__ __host__ bool test_triangle_box_intersection_6_sep_double(
                                                      const int voxel_index_X,
                                                      const int voxel_index_Y,
                                                      const int voxel_index_Z,
                                                      const float3* tri_verts,
                                                      const double3 tri_normal,
                                                      const double dX) {
  /////////////////////////////////// Allocate needed data:
  double2* edge_XY_normals =  new double2[3];
  double * edge_XY_distances = new double[3];
  double2* edge_ZX_normals =  new double2[3];
  double * edge_ZX_distances = new double[3];
  double2* edge_YZ_normals =  new double2[3];
  double * edge_YZ_distances = new double[3];

  double3 max_point = make_double3(dX, dX, dX); // A better name would better be delta_p since max_point would be voxel_min_corner_WORLD + make_float3(dX, dX, dX)
  double3 voxel_min_corner_WORLD = make_double3(dX*voxel_index_X, dX*voxel_index_Y, dX*voxel_index_Z);
  double dX_over_2 = 0.5 * dX;
  ///////////// Modified plane-box test:  //////
  // Argmax:
  double normal_dominant_value = fmaxf(fmaxf(abs(tri_normal.x), abs(tri_normal.y)), abs(tri_normal.z));
  double d_1 = dot( tri_normal, 0.5 * max_point - make_double3(tri_verts[0].x, tri_verts[0].y, tri_verts[0].z)) + dX_over_2 * normal_dominant_value;
  double d_2 = dot( tri_normal, 0.5 * max_point - make_double3(tri_verts[0].x, tri_verts[0].y, tri_verts[0].z)) - dX_over_2 * normal_dominant_value;
  double n_dot_p = dot( tri_normal, voxel_min_corner_WORLD);

  bool result = (n_dot_p + d_1)*(n_dot_p + d_2) <= 0.0;
  ///////////// 2D projection test: //////
  for ( int i = 0; i < 3; i++ ){
      // 1] COMPUTE Edge normals:
      edge_XY_normals[i] = make_double2( tri_verts[i].y - tri_verts[(i + 1)%3].y, tri_verts[(i + 1)%3].x - tri_verts[i].x ) ;
      edge_XY_normals[i] = copysign( 1.0 , tri_normal.z ) * edge_XY_normals[i]; // copysign exists in CUDA also
  edge_XY_distances[i] = (edge_XY_normals[i].x * (dX_over_2 - tri_verts[i].x) + edge_XY_normals[i].y * (dX_over_2 - tri_verts[i].y)) + dX_over_2 * fmaxf( abs(edge_XY_normals[i].x), abs(edge_XY_normals[i].y) ) ;
  // Test:
  result &= ( dot(edge_XY_normals[i], make_double2(voxel_min_corner_WORLD.x, voxel_min_corner_WORLD.y)) + edge_XY_distances[i]) >= 0.0;

      edge_ZX_normals[i] = make_double2( tri_verts[i].x - tri_verts[(i + 1)%3].x, tri_verts[(i + 1)%3].z - tri_verts[i].z ) ;
      edge_ZX_normals[i] = copysign( 1.0 , tri_normal.y ) * edge_ZX_normals[i];
  edge_ZX_distances[i] = (edge_ZX_normals[i].x * (dX_over_2 - tri_verts[i].z) + edge_ZX_normals[i].y * (dX_over_2 - tri_verts[i].x)) + dX_over_2 * fmaxf( abs(edge_ZX_normals[i].x), abs(edge_ZX_normals[i].y) ) ;
  // Test:
  result &= ( dot(edge_ZX_normals[i], make_double2(voxel_min_corner_WORLD.z, voxel_min_corner_WORLD.x)) + edge_ZX_distances[i]) >= 0.0;

      edge_YZ_normals[i] = make_double2( tri_verts[i].z - tri_verts[(i + 1)%3].z, tri_verts[(i + 1)%3].y - tri_verts[i].y ) ;
      edge_YZ_normals[i] = copysign( 1.0 , tri_normal.x ) * edge_YZ_normals[i];
  edge_YZ_distances[i] = (edge_YZ_normals[i].x * (dX_over_2 - tri_verts[i].y) + edge_YZ_normals[i].y * (dX_over_2 - tri_verts[i].z)) + dX_over_2 * fmaxf( abs(edge_YZ_normals[i].x), abs(edge_YZ_normals[i].y) ) ;
      // Test:
  result &= ( dot(edge_YZ_normals[i], make_double2(voxel_min_corner_WORLD.y, voxel_min_corner_WORLD.z)) + edge_YZ_distances[i]) >= 0.0;
  }
  //////////////////////////////////// Free memory:
  delete[] edge_XY_normals;
  delete[] edge_XY_distances;
  delete[] edge_ZX_normals;
  delete[] edge_ZX_distances;
  delete[] edge_YZ_normals;
  delete[] edge_YZ_distances;

  //return tri_plane_intersects_voxel & testResult_XY & testResult_ZX & testResult_YZ;
  return result;
}

void calcBoundaryValuesInplace(unsigned char* h_position_idx,
                               unsigned long long dim_x,
                               unsigned long long dim_y,
                               unsigned long long dim_z) {
  mesh_size_t dim_xy = dim_x*dim_y;
  mesh_size_t num_elements = dim_xy*dim_z;

  for(unsigned int i = 0; i < num_elements; i++) {
    if(h_position_idx[i] == 0)
      continue;

    if(i < dim_xy || i > num_elements-dim_xy) {
      h_position_idx[i] = 0x86;
      continue;
    }
    unsigned char count = 0;
    count +=  h_position_idx[i+1]>>7;
    count +=  h_position_idx[i-1]>>7;
    count +=  h_position_idx[i+dim_x]>>7;
    count +=  h_position_idx[i-dim_x]>>7;
    count +=  h_position_idx[i+dim_xy]>>7;
    count +=  h_position_idx[i-dim_xy]>>7;
    h_position_idx[i] |= count;
  }
}

void calcMaterialIndicesInplace(unsigned char* h_material_idx,
                                const unsigned char* h_position_idx,
                                unsigned long long dim_x,
                                unsigned long long dim_y,
                                unsigned long long dim_z) {
  mesh_size_t dim_xy = dim_x*dim_y;
  mesh_size_t num_elements = dim_xy*dim_z;
  unsigned char* h_mat = h_material_idx; // Abreviation

  for(unsigned int i = dim_xy; i < num_elements-dim_xy; i++) {
    if(h_position_idx[i] == 0)
      continue;

    unsigned char c_pos = h_position_idx[i];
    if(c_pos > 0 && c_pos < (6+128)) {
      unsigned char val = 0;
      val = val>(h_mat[i-1]&0x0F) ? val : h_mat[i-1]&0x0F;
      val = val>(h_mat[i+1]&0x0F) ? val : h_mat[i+1]&0x0F;
      val = val>(h_mat[i-dim_x]&0x0F)? val : h_mat[i-dim_x]&0x0F;
      val = val>(h_mat[i+dim_x]&0x0F)? val : h_mat[i+dim_x]&0x0F;
      val = val>(h_mat[i-dim_xy]&0x0F)? val : h_mat[i-dim_xy]&0x0F;
      val = val>(h_mat[i+dim_xy]&0x0F)? val : h_mat[i+dim_xy]&0x0F;
      h_mat[i] |= val<<4;
    }
  }

  for(unsigned int i = dim_xy; i < num_elements-dim_xy; i++) {
    h_mat[i] = h_mat[i]>>4;
  }
}


__global__ void calcBoundaryValuesInplaceKernel(unsigned char* d_position_idx,
                                                unsigned long long dim_x,
                                                unsigned long long dim_y,
                                                unsigned long long dim_z) {


}
