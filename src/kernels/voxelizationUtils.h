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
//
// Aalto University School of Science
//
// Update 12/2014:
// Surface Voxelization wrapping, Sebastian Prepelita
// Update 01/2015:
//  *)Surface Voxelization - conservative and 6-separating, Sebastian Prepelita
//  *)Solid voxelization - voxelizer result as in Schwarz paper (needs branch
//    'next_2' of the voxelizer)
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// Possible error messages:
//
// Material not found! For solid voxelization, solid voxelization differs too
// much than the surface voxelization! Most likely one failed near an edge of
// the mesh/triangles. Also a low possibility that a material of a voxel was
// not found in the hashmap.
#define ERR_NO_MATERIAL_FOUND 0x1

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "cudaUtils.h"
#include "../Voxelizer/include/common.h"
#include "../Voxelizer/include/voxelizer.h"
#include "cudaMesh.h"

class Node;

enum Surface_Voxelization_Type {VOX_CONSERVATIVE, VOX_6_SEPARATING};

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

///////////////////////////////////////////////////////////////////////////////
/// \brief Performs a solid voxelization as defined in the reference. Note that
/// a solid voxelization does not deal with surface materials - it would need
/// waterproof (sub-)objects with unique material id to work. One workaround is
/// to also do a surface (conservative voxelization) and get the material indexes
/// from there (originally done in the voxelizer). The output can directly be
/// fed to the ParallelFDTD solver: it has the bit_mask, materials, and zero
/// padding done properly.
///
///	 Reference: Schwarz M., Seidel H-P, "Fast Parallel Surface and Solid Voxeli
///  zation on GPUs", ACM Transactions on Graphics, Vol. 29, No. 6, Article 179
///  , (2010)
///
/// The voxelized geometry is surrounded by at least 1 extra air layer:
///   - the lower corner of the mesh starts from the lower corner of voxel[1,1,1]
///   - one extra voxel layer is added in Z direction on top of the conservative
///     bounding box of the mes
///   - [1-3] extra voxel layers are added in the Y direction so that the final
///   Y dimension is divisible with 4.
///   - The X dimension will be an integer multiplier of 32 by default - from the
///   bit processing insinde the voxelizer.
///
///
///   ! NOTE: cannot guarantee a rule if many triangles with different
///   materials intersect a voxel.
/// The final material indexes in the air voxels close to a solid boundary voxel
/// will adhere to this rules (after setting the triangle-to-voxel material):
///   - If the air voxel has a material from a surface voxelization, then it is
///     left like that
///   - otherwise, the dominant material (i.e., the material found in most
/// neighbours) is chosen from its neigboring voxels. If there is no dominant
/// material, the first one in the unique_materials_ids is chosen (except air).
///   ! NOTE: Some cases were found where the solid voxelization missed a triangle
/// (from a surrounding box mesh) and the result differed too much from a surface
/// voxelization. This translates into a next-to-solid air voxel that has no
/// material and it has 0 neighbors with any material. In such cases, see error
/// value(s) returned.
///
/// \param[in] vertices Mesh data
/// \param[in] indices Mesh data
/// \param[in] materials Mesh data
/// \param[in] number_of_triangles Mesh data
/// \param[in] number_of_vertices, Mesh data
/// \param[in] number_of_unique_materials, Number of materials
/// \param[in]voxel_edge, The voxel edge dX in m
/// \param[out] d_postition_idx, the boundar_id (number of boundary neighbors +
///				bit_mask)
/// \param[out] d_materials_idx, the material pointer
/// \param[out] voxelization_dim, the dimensions of the voxelized surface
/// \param[out] no_of_surface_nodes, number of surface nodes found
///
/// Returns 0 if no logical errors encountered inside. Check ERR_* #defines for
/// bit-defined error messages.
///////////////////////////////////////////////////////////////////////////////
unsigned char voxelizeGeometry_solid(float* vertices,
                                     unsigned int* indices,
                                     unsigned char* materials,
                                     unsigned int number_of_triangles,
                                     unsigned int number_of_vertices,
                                     unsigned int number_of_unique_materials,
                                     double voxel_edge,
                                     unsigned char** d_position_idx,
                                     unsigned char** d_material_idx,
                                     uint3* voxelization_dim);

///////////////////////////////////////////////////////////////////////////////
/// \brief A wrapper for the voxelizer library that results in a conservative
/// voxelization.
///  Wraps the functionality of voxelizeToSurfaceNodes().
/// It is not required that the domain be surrounded by a border mesh, but the
/// result will be truncated to the bounding box of the mesh.
///
/// It outputs the domain voxelized and just needs to be fed in the Voxelizer.
///
/// For materials, the dominant material (i.e., the material found in most
/// neighbours) is chosen for all boundary neighbors.
/// If there is no dominant material, the first one in the unique_materials_ids
/// is chosen (except air).
///
/// For example, if one air cell has 2 boundaries with mat_id = 2 and 2 with
/// mat_id = 10, then if unique_mat=[0,10,2,3], 10 will be chosen.
///
/// Notes:
/// Do not use the material id = 0 (otherwise error will be thrown). Reserved
/// for air.
/// Function does not check that values in unique_materials_ids are unique -
/// dominance rule will break down in this case.
///
/// \param[in] vertices Mesh data
/// \param[in] indices Mesh data
/// \param[in] materials Mesh data
/// \param[in] number_of_triangles Mesh data
/// \param[in] number_of_vertices, Mesh data
/// \param[in] number_of_unique_materials, Number of materials
/// \param[in]voxel_edge, The voxel edge dX in m
/// \param[out]  d_postition_idx, the boundar_id (number of boundary neighbors + bit_mask)
/// \param[out] d_materials_idx, the material pointer
/// \param[out] voxelization_dim, the dimensions of the voxelized surface
/// \param[out] no_of_surface_nodes, number of surface nodes found
///
/// Returns 0 if no logical errors encountered inside. Check defines for bit-
/// defined error messages.
///////////////////////////////////////////////////////////////////////////////
unsigned char voxelizeGeometry_surface(float* vertices,
                                       unsigned int* indices,
                                       unsigned char* materials,
                                       unsigned int number_of_triangles,
                                       unsigned int number_of_vertices,
                                       unsigned int number_of_unique_materials,
                                       double voxel_edge,
                                       unsigned char** d_postition_idx,
                                       unsigned char** d_materials_idx,
                                       uint3* voxelization_dim,
                                       unsigned int* no_of_surface_nodes);

///////////////////////////////////////////////////////////////////////////////
/// \brief Function converts a surface voxelization to input understood by the
/// parallelFDTD library. It expects that each boundary cell has a bid = 0 and
/// a material_index != 0 and it expects each air node to have a bid = 1.
///
///  Please make sure the d_pos and d_mat have allocated memory before and
///  after the domain (used to count the number of neighboring air_cells).
///  Otherwise it will under/over flow.
///   Concretely, the domain should have one extra voxel in +-x, +-y, +-z.
///    Due to memory alignment, d_pos will be in memory like [x*y, d_p, x*y],
///    and d_p with the corresponding interleaved extra voxels frin d_pos.
///
/// \param[in/out] d_pos the position/boundary_id matrix
/// \param[in/out] d_mat the material matrix
/// \param[in] dim domain dimensions
/// \param[in] bit_mask the bit mask to be applied (bitwise OR) to the air
///    cells.
/// \param[in] unique_materials_ids array, containing all the unique id-s in
/// \param[in] dev_idx The GPU device where all is done
///
/// Returns 0 if no logical errors encountered inside. Check defines for bit-
/// defined error messages.
///////////////////////////////////////////////////////////////////////////////
unsigned char to_ParallelFDTD_surface_voxelization(
                                unsigned char** d_pos,
                                unsigned char** d_mat,
                                uint3 dim,
                                const unsigned char bit_mask,
                                const unsigned char * unique_materials_ids,
                                unsigned int number_of_unique_materials,
                                unsigned char dev_idx = 0);

///////////////////////////////////////////////////////////////////////////////
/// \brief Function that gets the b_id and the corresponding material_id for
/// ONLY the surface nodes coming from a voxelizeToSurfaceNodes()
///
/// \param[in] nodes surface nodes from a voxelizeToSurfaceNodes() call
/// \param[in] surface_nodes_HM, // [in] HashMap from a voxelizeToSurfaceNodes() call
/// \param[out] d_pos the position/boundary_id matrix
/// \param[out] d_mat the material matrix
/// \param[in] dim domain dimensions
/// \param[in] bit_mask the bit mask to be applied (bitwise OR) to the air cells.
/// \param[in] unique_materials_ids array, containing all the unique id-s in
///
/// Returns 0 if no logical errors encountered inside. Check defines for bit-
/// defined error messages.
///////////////////////////////////////////////////////////////////////////////
unsigned char nodes2Vectors_surface_only(vox::SurfaceNode* nodes,
                                vox::HashMap surface_nodes_HM,
                                unsigned char** d_pos,
                                unsigned char** d_mat,
                                uint3 dim,
                                const unsigned char bit_mask,
                                const unsigned char * unique_materials_ids,
                                unsigned int number_of_unique_materials);

///////////////////////////////////////////////////////////////////////////////
/// \brief This function returns the mesh bounding box in number of voxels and
/// calculates the axis-aligned bounding box and normal for each triangle.
///   Function also translates the mesh domain so that its minimum corner of
/// its bounding box is: origin + [FLT_EPSILON,FLT_EPSILON,FLT_EPSILON].
///
/// \param[in] h_vertices Mesh vertices [x,y,z] in world coordinates.
/// \param[in] h_indices Mesh triangle indices (locates d_vertices)
/// \param[in] num_triangles Total number of triangles
/// \param[in] voxel_edge The voxel grid spacing
/// \param[out] triangles_BB An array with the triangle bounding box in voxel
///   coordinates.
/// \param[out] triangles_Normal An array with the normal of the triangle in
///   double precision (it affects the correctness of voxelization).
///////////////////////////////////////////////////////////////////////////////
vox::Bounds<uint3> get_mesh_BB_and_translate_vertices(
                          double* h_local_vertices,
                          const float* h_vertices,
                          unsigned int* h_indices,
                          unsigned int number_of_triangles,
                          unsigned int number_of_vertices,
                          double voxel_edge,
                          vox::Bounds<uint3>* triangles_BB,
                          double3* triangles_Normal
                          );

template<class Node>
__global__ void nodes2VectorsKernel(Node* nodes,
                                    unsigned char* d_position_idx_ptr,
                                    unsigned char* d_material_idx_ptr,
                                    unsigned int num_elems);

///////////////////////////////////////////////////////////////////////////////
/// \brief Kernel that does almost the same job as nodes2VectorsKernel, but it
/// cuts the domain to a new size. It was conceived to remove one extra voxel
/// layer in the Y and Z direction of the result coming from the solid voxeli
/// zer which does this.
///
///    It can be used to cut even more of the domain but not to zero padd
///  since the nodes index will overflow. But it can easily be updated for
///  padding also with an if statement and correct allocation of position and
///  material arrays. For padding, use padWithZeros in cudaMesh afterwards.
///
/// \param[in] nodes The node list coming from the voxelizer.
/// \param[out] d_position_idx_ptr The resulting position matrix (needs to be
///				allocated)
/// \param[out] d_material_idx_ptr The resulting position matrix (needs to be
///				allocated)
/// \param[in] dim_xy_old The dim_x*dim_y coming from the voxelizer.
/// \param[in] dim_x_old The dim_x coming from the voxelizer.
/// \param[in] dim_xy_new The new dim_x*dim_y that the domain will be cut.
/// \param[in] dim_x_new The new dim_x that the domain will be cut.
/// \param[in] one_over_dim_xy_new Equals 1.0/dim_xy_new.
/// \param[in] one_over_dim_x_new Equals 1.0/dim_x_new.
/// \param[in] num_elems_new The new dim_x*dim_y*dim_z of the cut domain.
///////////////////////////////////////////////////////////////////////////////
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
                                     unsigned int num_elems_new);

/// \brief This is a standard Kernel that outputs b_id as conceived by the Voxelizer,
/// i.e. not very useful for Parallel FDTD update. Deprecated - Use to check what
/// surface voxelization //returns.

__global__ void nodes2Vectors_surface_only_Kernel(vox::SurfaceNode* nodes,
                                                  vox::HashMap surface_nodes_HM,
                                                  unsigned char* d_K,
                                                  unsigned char* d_B,
                                                  unsigned int num_elems);

///////////////////////////////////////////////////////////////////////////////
/// \brief This is an intermediate step towards boundary_id that makes sense for the
/// Parallel FDTD update
/// \param[in] nodes nodes of the mesh
/// \param[in] surface_nodes_HM boundary_matrix
/// \param[out] d_K, boundary_matrix
/// \param[out] d_B, material_matrix
/// \param[in] num_elems number of elements in position/material matrix
///////////////////////////////////////////////////////////////////////////////
__global__ void nodes2Vectors_prepare_surface_data_Kernel(vox::SurfaceNode* nodes,
                                                          vox::HashMap ,
                                                          unsigned char* d_K,
                                                          unsigned char* d_B,
                                                          unsigned int num_elems);

///////////////////////////////////////////////////////////////////////////////
/// \brief Simple Kernel that inverts the LSB of a matrix. This is used during
/// the solid voxelization, where a solid voxel will be 1 and an air voxel 0.
/// To count the number of neighboring air nodes, this needs to be inverted.
///
/// \param[in,out] d_bid The matrix to be processed (should be a boundary_id)
/// \param[in] num_elems number of elements in the matrix
///////////////////////////////////////////////////////////////////////////////
__global__ void invert_bid_output_Kernel(unsigned char* d_bid,
                                         unsigned int num_elems);

///////////////////////////////////////////////////////////////////////////////
/// \brief This one transforms the boundary_id coming from
/// nodes2Vectors_prepare_surface_data_Kernel() to a boundary_id as used by the FDTD update
/// \param[in] d_Bid_voxelizer_in  position matrix from nodes2Vectors_prepare_surface_data_Kernel()
/// \param[out]  d_Bid_out the boundary_id as required by Parallel FDTD
/// \param[in] d_Mat_voxelizer_in material matrix from nodes2Vectors_prepare_surface_data_Kernel()
/// \param[in] max_mat_id The maximum ID in the materials.
/// \param[out] d_Mat_out the new material matrix
/// \param[in] num_elems
/// \param[in] dim_xy  (dim.x * dim.y) - slice size of the resulting
/// \param[in] dim_x, dim.x
/// \param[in] one_over_dim_xy 1.0 / (dim.x * dim.y)
/// \param[in] one_over_dim_x 1.0 / dim.x
/// \param[in] bit_mask the bit mask to be applied (bitwise OR) to the air cells.
/// \param[out] d_error Any error from the kernel (done with bit OR - for bit
///				information, see defines in voxelizationUtils.cu
///////////////////////////////////////////////////////////////////////////////
__global__ void Count_Air_Neighboring_Nodes_Kernel(
                                    const unsigned char* __restrict d_Bid_voxelizer_in,
                                    unsigned char* d_Bid_out,
                                    const unsigned char* __restrict d_Mat_voxelizer_in,
                                    const unsigned char max_mat_id,
                                    unsigned char* d_Mat_out,
                                    const long long num_elems,
                                    const long long dim_xy,
                                    const long long dim_x,
                                    const double one_over_dim_xy,
                                    const double one_over_dim_x,
                                    const unsigned char bit_mask,
                                    unsigned char* d_error);

///////////////////////////////////////////////////////////////////////////////
/// \brief Modulo function, as described in [1]
/// It does a division of x by n
/// Chose the double one that can work for bigger meshes. In case speed is an
/// issue, for a 20-fold speed increase, use a float modulo see [1]
/// References: [1] https://devtalk.nvidia.com/default/topic/416969/cuda-programming-and-performance/error-in-modulo-operation/#entry589954
/// \param x : the numerator
/// \param n : the denominator
/// \param one_over_n : it equals 1.0 / (double)n; Useful to pass as arguments
///to kernel when n is fixed across kernel calls (again, see discussion in [1]).
/// \return a structure containing the quotient and remainder of the division of x to n.
///////////////////////////////////////////////////////////////////////////////
__device__ std::div_t double_modulo(const int &x,const int &n, const double &one_over_n);
/// Unsigned int version:
__device__ std::ldiv_t udouble_modulo(const unsigned int &x,const unsigned int &n, const double &one_over_n);

///////////////////////////////////////////////////////////////////////////////
/// \brief Wraps double_modulo(int, int) in case one_over_n is not used - see
/// double_modulo(int, int) function for more information.
/// \param x : the numerator
/// \param n : the denominator
/// \return a structure containing the quotient and remainder of the division of x to n.
///////////////////////////////////////////////////////////////////////////////
__device__ std::div_t double_modulo(const int &x, const int &n);
/// Unsigned int version:
__device__ std::div_t double_modulo(const unsigned int &x, const unsigned int &n);

///////////////////////////////////////////////////////////////////////////////
/// \brief 6-separating surface voxelization of a mesh. The output can be
/// plugged in the ParallelFDTD update code.
///
/// Function wraps voxelizeGeometry_surface_Host(). Reference: Schwarz M.,
/// Seidel H-P, "Fast Parallel Surface and Solid Voxelization on GPUs", ACM
/// Transactions on Graphics, Vol. 29, No. 6, Article 179, (2010)
///
///		Update: Added possibility to restrict the voxelization space between
/// some Z slices. Useful to partially voxelize the geometry. Note that the
/// first and last slice are not correct in such case (if left to -1 they are).
///
///   NOTES:
///		1) The mesh miminum corner of axis-aligned bounding-box will be
/// translated to point [FLT_EPSILON,FLT_EPSILON,FLT_EPSILON]. Also, the center
/// of voxel[0,0,0] is [dX/2,dX/2,dX/2] in world coordinates.
///     2) The voxelized geometry will be surrounded by an 1-voxel thick air
/// domain. Because of this,the actual positions of the vertices are translated
/// by [dX, dX, dX]. This can be controlled by displace_mesh_voxels variable
/// inside the function. If displace_mesh_voxels is set to 0, results might be
/// wrong since the outputs on the border are calculated based on wrapped-around
/// values in the domain.
///     2.1) The air domain might not result in 100% accurate voxelization at
/// the border: if the mesh contains triangles inside the bounding box of the
/// mesh domain, then also voxels from the surrounding air nodes will have to
/// be solid. However, these surrounding voxels will be air nodes no matter
/// what.
///     3)
///     4) Function does not check that values in unique_materials_ids are unique
/// - dominance rule will break down in this case.
///     5) Domain (including the 1-layer air) is padded with air so that the
/// dimensions satisfy X_dim%32=0, Y_dim%4=0, Z_dim%1=0. See internal variable
/// voxelize_for_block_size for details.
/// 	6) The conversion to ParallelFDTD format is done on the GPU. Due to
/// the rule to write the dominant material, a vector of available materials
/// is created in each node, basically multiplying the required memory and if
/// the GPU card runs out of memory, "an illegal memory access" is thrown.
///  Just do partial voxelizations in such case (on different gpu cards).
///
///   ! WARNING: cannot guarantee a rule if many triangles with different
///   materials intersect a voxel.
/// However, once the solid voxel materials are set, the dominant material (i.e.,
/// the material found in most neighbours) is chosen for all air voxels close to
/// a solid boundary voxels. If there is no dominant material, the first one in
/// the unique_materials_ids is chosen (except air).
///
/// For parameter list, see voxelizeGeometry_surface_Host().
///
/// Returns 0 if no logical errors encountered inside. Check defines for bit-
/// defined error messages.
///////////////////////////////////////////////////////////////////////////////
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
                                        long partial_vox_Z_start = -1,
                                        long partial_vox_Z_end = -1,
                                        unsigned char dev_idx = 0);

///////////////////////////////////////////////////////////////////////////////
/// \brief Conservative surface voxelization of a mesh. The output can be
/// plugged in the ParallelFDTD update code.
///
/// Function wraps voxelizeGeometry_surface_Host(). Reference: Schwarz M.,
/// Seidel H-P, "Fast Parallel Surface and Solid Voxelization on GPUs", ACM
/// Transactions on Graphics, Vol. 29, No. 6, Article 179, (2010)
///
///		Update: Added possibility to restrict the voxelization space between
/// some Z slices. Useful to partially voxelize the geometry. Note that the
/// first and last slice are not correct in such case (if left to -1 they are).
///
///   NOTES:
///		1) The mesh miminum corner of axis-aligned bounding-box will be
/// translated to point [FLT_EPSILON,FLT_EPSILON,FLT_EPSILON]. Also, the center
/// of voxel[0,0,0] is [dX/2,dX/2,dX/2] in world coordinates.
///     2) The voxelized geometry will be surrounded by an 1-voxel thick air
/// domain. Because of this,the actual positions of the vertices are translated
/// by [dX, dX, dX]. This can be controlled by displace_mesh_voxels variable
/// inside the function. If displace_mesh_voxels is set to 0, results might be
/// wrong since the outputs on the border are calculated based on wrapped-around
/// values in the domain.
///     2.1) The air domain might not result in 100% accurate voxelization at
/// the border: if the mesh contains triangles inside the bounding box of the
/// mesh domain, then also voxels from the surrounding air nodes will have to
/// be solid. However, these surrounding voxels will be air nodes no matter
/// what. Tests show that some triangles on the border of the mesh bounding box
/// do 'bleed' some solid voxels in the sorrounding 1-voxel air layer.
///     3)
///		4) Function does not check that values in unique_materials_ids are unique
/// - dominance rule will break down in this case.
///     5) Domain (including the 1-layer air) is padded with air so that the
/// dimensions satisfy X_dim%32=0, Y_dim%4=0, Z_dim%1=0. See internal variable
/// voxelize_for_block_size for details.
/// 	6) The conversion to ParallelFDTD format is done on the GPU. Due to
/// the rule to write the dominant material, a vector of available materials
/// is created in each node, basically multiplying the required memory and if
/// the GPU card runs out of memory, "an illegal memory access" is thrown.
///  Just do partial voxelizations in such case (on different gpu cards).
///
///   ! WARNING: cannot guarantee a rule if many triangles with different
///   materials intersect a voxel.
/// However, once the solid voxel materials are set, the dominant material (i.e.,
/// the material found in most neighbours) is chosen for all air voxels close to
/// a solid boundary voxels. If there is no dominant material, the first one in
/// the unique_materials_ids is chosen (except air).
///
/// For parameter list, see voxelizeGeometry_surface_Host().
///
/// Returns 0 if no logical errors encountered inside. Check defines for bit-
/// defined error messages.
///////////////////////////////////////////////////////////////////////////////
unsigned char voxelizeGeometry_surface_conservative(
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
                              long partial_vox_Z_start = -1,
                              long partial_vox_Z_end = -1,
                              unsigned char dev_idx = 0);

///////////////////////////////////////////////////////////////////////////////
/// \brief Function returns the voxelization dimensions as returned by
/// voxelizeGeometry_surface_6_separating() or voxelizeGeometry_surface_con
/// servative(). Note that it does not voxelize anything, it merely determines
/// the FULL voxelization domain.
///
///  Note that if you call voxelizeGeometry_surface_Host() with parameter
/// displace_mesh_voxels != 1 or voxelize_for_block_size != {32,4,1}, then the
/// result returned by this function will not match the voxelization dimensions
/// of voxelizeGeometry_surface_Host() function.
///
/// \param[in] vertices Mesh data: Mesh vertices [x,y,z] in world coordinates.
/// \param[in] indices Mesh data: Mesh triangle indices (locates vertices)
/// \param[in] number_of_triangles Mesh data: The total number of triangles
/// \param[in] number_of_vertices Mesh data: The total number of vertices
/// \param[in] voxel_edge the voxel grid spacing
///
/// Returns the voxelization dimensions for full geometry.
///////////////////////////////////////////////////////////////////////////////
uint3 get_Geometry_surface_Voxelization_dims(float* vertices,
                    unsigned int* indices,
                    unsigned int number_of_triangles,
                    unsigned int number_of_vertices,
                    double voxel_edge);

///////////////////////////////////////////////////////////////////////////////
/// \brief This function processes ALL triangles and intersects all the voxels
/// inside each triangle's bounding box with the triangle creating a
/// conservative or 6-separating voxelization. Then, it processes the output so
/// that it can be plugged in the ParallelFDTD update code.
///
///		Update: Added possibility to restrict the voxelization space between
/// some Z slices. Useful to partially voxelize the geometry. Note that the
/// first and last slice are not correct in such case (if left to -1 they are).
///
/// \param[in] vertices Mesh data: Mesh vertices [x,y,z] in world coordinates.
/// \param[in] indices Mesh data: Mesh triangle indices (locates vertices)
/// \param[in] materials Mesh data: Material of each triangle (note mat_id = 0
//             is reserved for air.
/// \param[in] number_of_triangles Mesh data: The total number of triangles
/// \param[in] number_of_vertices Mesh data: The total number of vertices
/// \param[in] number_of_unique_materials Mesh data: The total number of
///            materials in the mesh (not counting 0)
/// \param[in] voxel_edge the voxel grid spacing
/// \param[out] d_postition_idx The boundar_id (number of boundary neighbors +
///             bit_mask)
/// \param[out] d_materials_idx The material pointer
/// \param[out] voxelization_dim The dimensions of the voxelized domain ( note
///             the mesh is translated to [0,0,0] ).
/// \param[in] bit_mask The bit mask applied by bit-OR to the air nodes.
/// \param[in] displace_mesh_voxels The thickness in voxels of the surrounding
///            layer of air nodes. If 0, the voxelization might be innacurate
///            at the borders.
/// \param[in] voxType The type of voxelization: either 6-separating of
///			   conservative.
/// \param[in] voxelize_for_block_size Pads the domain with zeros so that the
///            final domain size is a multiple of voxelize_for_block_size.x,
///			   .y, .z. Useful when the output is plugged in directly to
///			   ParallelFDTD simulation.
/// \param[in] partial_vox_Z_start Useful in partial voxelization of the geome
///            try: If != -1, then the voxelization is bounded in z dimension
///			   between partial_vox_Z_start and last slice or partial_vox_Z_end
///			   if that is != -1. This has to account for the displace_mesh_vox
///			   els air nodes in the initial voxelization.
///			   ! Note that the first slice is not correct! Just call function
///			   with another extra slice down to overpass this.
/// \param[in] partial_vox_Z_end Useful in partial voxelization of the geome
///            try: If != -1, then the voxelization is bounded in z dimension
///			   between 0 or partial_vox_Z_start and  partial_vox_Z_end. This
///			   has to account for the displace_mesh_voxels air nodes in the
///			   initial voxelization.
///			   ! Note that the last slice is not correct! Just call function
///			   with another extra slice up to overpass this.
/// \param[in] unsigned char dev_idx The GPU device where the counting of boun
///			   daries happens (to match the ParallelFDTD format). Use this
///			   together with partial voxelization Z indices in case you run out
///			   of memory.
///
///
/// Returns 0 if no logical errors encountered inside. Check defines for bit-
/// defined error messages.
///////////////////////////////////////////////////////////////////////////////
unsigned char voxelizeGeometry_surface_Host(
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
                                      unsigned char bit_mask,
                                      unsigned int displace_mesh_voxels,
                                      enum Surface_Voxelization_Type voxType,
                                      uint3 voxelize_for_block_size,
                                      long partial_vox_Z_start = -1,
                                      long partial_vox_Z_end = -1,
                                      unsigned char dev_idx = 0);

///////////////////////////////////////////////////////////////////////////////
/// \brief Function to voxelize a given geometry using either 6 separating
///        or conserative surface voxelization to the HOST memory.
///
/// \param[in] vertices Mesh data: Mesh vertices [x,y,z] in world coordinates.
/// \param[in] indices Mesh data: Mesh triangle indices (locates vertices)
/// \param[in] materials Mesh data: Material index of each triangle
///                                   (note mat_id = 0 is reserved for air.)
/// \param[in] num_triangles Mesh data: The total number of triangles
/// \param[in] num_vertices Mesh data: The total number of vertices
/// \param[in] num_unique_materials Mesh data: The total number of
///            materials in the mesh (not counting 0)
/// \param[in] voxel_edge the voxel grid spacing
/// \param[out] h_postition_idx The boundar_id (number of boundary neighbors +
///             bit_mask) WILL BE ALLOCATED AND NEEDS TO BE FREED
/// \param[out] h_materials_idx The material pointer
///             WILL BE ALLOCATED AND NEEDS TO BE FREED
/// \param[out] voxelization_dim The dimensions of the voxelized domain ( note
///             the mesh is translated to [0,0,0] ).
/// \param[in] displace_mesh_voxels The thickness in voxels of the surrounding
///            layer of air nodes. If 0, the voxelization might be innacurate
///            at the borders.
/// \param[in] voxType The type of voxelization: either 6-separating of
///			   conservative.
/// \param[in] voxelize_for_block_size Pads the domain with zeros so that the
///            final domain size is a multiple of voxelize_for_block_size.x,
///			   .y, .z. Useful when the output is plugged in directly to
///			   ParallelFDTD simulation.
/// \param[in] partial_vox_Z_start Useful in partial voxelization of the geome
///            try: If != -1, then the voxelization is bounded in z dimension
///			   between partial_vox_Z_start and last slice or partial_vox_Z_end
///			   if that is != -1. This has to account for the displace_mesh_vox
///			   els air nodes in the initial voxelization.
///			   ! Note that the first slice is not correct! Just call function
///			   with another extra slice down to overpass this.
/// \param[in] partial_vox_Z_end Useful in partial voxelization of the geome
///            try: If != -1, then the voxelization is bounded in z dimension
///			   between 0 or partial_vox_Z_start and  partial_vox_Z_end. This
///			   has to account for the displace_mesh_voxels air nodes in the
///			   initial voxelization.
///			   ! Note that the last slice is not correct! Just call function
///			   with another extra slice up to overpass this.
///
///
/// Returns 0 if no logical errors encountered inside. Check defines for bit-
/// defined error messages.
///////////////////////////////////////////////////////////////////////////////
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
                                         long partial_vox_Z_start = -1,
                                         long partial_vox_Z_end = -1);

///////////////////////////////////////////////////////////////////////////////
/// \brief This function processes ALL triangles and intersects all the voxels
/// inside each triangle's bounding box with the triangle. The surface voxeli-
/// zation can either be conservative or 6-separating.
///
///   NOTES:
///		1) The function assumes that the voxels start at [0,0,0], so
///   the center of voxel[0,0,0] is [dX/2,dX/2,dX/2] in world coordinates!
///		2) The function sets the bid of the intersected voxels to 0 (set all
///   values of d_Bid_out to 1 beforehand)
///
///   ! WARNING: cannot guarantee a rule if many triangles with different
///   materials intersect a voxel - the final result is a random material.
///
/// \param[in] dX  the voxel grid spacing
/// \param[in] d_vertices Mesh vertices [x,y,z] in world coordinates.
/// \param[in] d_indices Mesh triangle indices (locates d_vertices)
/// \param[in] d_materials Mesh triangle materials for each triangle
/// \param[in] d_triangles_BB Bounding-boxes for each triangle in voxels
/// \param[in] d_triangles_Normal Triangle normal in double precision
/// \param[out] d_Bid_out The boundary_id. Will have 1 if a triangle intersect
///				the voxel and 0 otherwise.
/// \param[out] d_Mat_out The materials. Will have the triangle's material_id
///				if a triangle intersect the voxel and 0 otherwise.
/// \param[in] num_triangles Total number of triangles
/// \param[in] dim_xy The number of XY elements in d_Bid_out (one Z slice size)
/// \param[in] dim_x The number of X elements in d_Bid_out
/// \param[in] space_BB_vox The Bounding Box of the voxelization world. This is
///				used to limit the output of the function between Z slices for
/// 			example. This is useful for partial voxelization.
/// \param[in] displace_mesh_voxels Number of air elements surrounding the
///				domain. Usually 1.
/// \param[in] Surface_Voxelization_Type The type of voxelization: either
/// VOX_CONSERVATIVE or VOX_6_SEPARATING.
///////////////////////////////////////////////////////////////////////////////
void intersect_triangles_surface_Host(
                  const double dX,
                  const double* h_vertices,
                  const unsigned int* h_indices,
                  const unsigned char* h_materials,
                  const vox::Bounds<uint3>* h_triangles_BB,
                  const double3* h_triangles_Normal,
                  unsigned char*  h_Bid_out,
                  unsigned char* h_Mat_out,
                  const unsigned int num_triangles,
                  const mesh_size_t dim_xy,
                  const int dim_x,
                  vox::Bounds<uint3> space_BB_vox,
                  unsigned int displace_mesh_voxels,
                  enum Surface_Voxelization_Type voxType);

///////////////////////////////////////////////////////////////////////////////
/// Host optimized function version of traingle-cube intersection
/// 	(6-separating). Things can be optimized even further.
///
///		For details, see:
///		* Schwarz M., Seidel H-P, "Fast Parallel Surface and Solid Voxelization
///		on GPUs", ACM Transactions on Graphics, Vol. 29, No. 6, Article 179,
///		(2010)
///
/// \param[in] voxel_index_X The index of the cube base (i.e., minimum corner)
///				in voxels (the equivalent minimal world coordinate is x =
///				voxel_index_X*dX)
/// \param[in] voxel_index_Y The index of the cube base (i.e., minimum corner)
///				in voxels (the equivalent minimal world coordinate is y =
///				voxel_index_Y*dX)
/// \param[in] voxel_index_Z The index of the cube base (i.e., minimum corner)
///				in voxels (the equivalent minimal world coordinate is z =
///				voxel_index_Z*dX)
/// \param[in] tri_verts  A vector containing 3 float3 elements. Each float3
///				element represents the (x,y,z) coordinate of each vertex of
///				the triangle.
/// \param[in] tri_normal A float3 number representing the normal vector of the
///				triangle. Note that it need not be normalized.
/// \param[in] dX The dimension of the cube in world coordinates (usually in m).
///
/// \param[out] Returns true if the triangle intersects the cube and false
///				otherwise.
///////////////////////////////////////////////////////////////////////////////
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
                                                double3 &max_point);

///////////////////////////////////////////////////////////////////////////////
/// Host optimized function version of traingle-cube intersection
/// 	(conservative). Things can be optimized even further.
///
///		For details, see:
///		* Schwarz M., Seidel H-P, "Fast Parallel Surface and Solid Voxelization
///		on GPUs", ACM Transactions on Graphics, Vol. 29, No. 6, Article 179,
///		(2010)
///
/// \param[in] voxel_index_X The index of the cube base (i.e., minimum corner)
///				in voxels (the equivalent minimal world coordinate is x =
///				voxel_index_X*dX)
/// \param[in] voxel_index_Y The index of the cube base (i.e., minimum corner)
///				in voxels (the equivalent minimal world coordinate is y =
///				voxel_index_Y*dX)
/// \param[in] voxel_index_Z The index of the cube base (i.e., minimum corner)
///				in voxels (the equivalent minimal world coordinate is z =
///				voxel_index_Z*dX)
/// \param[in] tri_verts  A vector containing 3 float3 elements. Each float3
///				element represents the (x,y,z) coordinate of each vertex of
///				the triangle.
/// \param[in] tri_normal A float3 number representing the normal vector of the
///				triangle. Note that it need not be normalized.
/// \param[in] dX The dimension of the cube in world coordinates (usually in m).
///
/// \param[out] Returns true if the triangle intersects the cube and false
///				otherwise.
///////////////////////////////////////////////////////////////////////////////
__host__ bool test_triangle_box_intersection_double_opt(
                                                      const int &voxel_index_X,
                                                      const int &voxel_index_Y,
                                                      const int &voxel_index_Z,
                                                      const double3* tri_verts,
                                                      const double3 &tri_normal,
                                                      const double &dX,
                                                      double2 &edge_2D_normal,
                                                      double &edge_2D_distance,
                                                      double3 &max_point);

////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// CUDA code: ///////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Dummy function that launches a CUDA Kernel for 6-separating surface
/// voxelization. It is slower than a CPU implementation due to high number of
/// register usage ( <20% occupancy in CUDA )
///////////////////////////////////////////////////////////////////////////////
void CUDA_launch_6_separating_surface_voxelization(uint3* h_voxelization_dim,
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
                                  double3* h_triangles_Normal);

///////////////////////////////////////////////////////////////////////////////
/// \brief This function processes each triangle and intersects all the voxels
/// inside its bounding box with the triangle - 6 separating surface
/// voxelization.
///
///   NOTES:
///		1) The function assumes that the voxels start at [0,0,0], so
///   the center of voxel[0,0,0] is [dX/2,dX/2,dX/2] in world coordinates!
///		2) The function sets the bid of the intersected voxels to 0 (set all
///   values of d_Bid_out to 1 beforehand)
///
///   ! WARNING: cannot guarantee a rule if many triangles with different
///   materials intersect a voxel - the final result is a random material.
///
/// \param[in] dX  the voxel grid spacing
/// \param[in] d_vertices Mesh vertices [x,y,z] in world coordinates.
/// \param[in] d_indices Mesh triangle indices (locates d_vertices)
/// \param[in] d_materials Mesh triangle materials for each triangle
/// \param[in] d_triangles_BB Bounding-boxes for each triangle in voxels
/// \param[in] d_triangles_Normal Triangle normal in double precision
/// \param[out] d_Bid_out The boundary_id. Will have 1 if a triangle intersect
///				the voxel and 0 otherwise.
/// \param[out] d_Mat_out The materials. Will have the triangle's material_id
///				if a triangle intersect the voxel and 0 otherwise.
/// \param[in] num_triangles Total number of triangles
/// \param[in] num_elements Total number of elements in d_Bid_out/d_Mat_out
/// \param[in] dim_xy The number of XY elements in d_Bid_out (one Z slice size)
/// \param[in] dim_x The number of X elements in d_Bid_out
///////////////////////////////////////////////////////////////////////////////
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
                            const int dim_x);

///////////////////////////////////////////////////////////////////////////////
/// Function that does a triangle-cube (axis-aligned) intersection test (6-
///	separating voxelization).
///		For details, see:
///		* Schwarz M., Seidel H-P, "Fast Parallel Surface and Solid Voxelization
///		on GPUs", ACM Transactions on Graphics, Vol. 29, No. 6, Article 179,
///		(2010)
///
/// \param[in] voxel_index_X The index of the cube base (i.e., minimum corner)
///				in voxels (the equivalent minimal world coordinate is x =
///				voxel_index_X*dX)
/// \param[in] voxel_index_Y The index of the cube base (i.e., minimum corner)
///				in voxels (the equivalent minimal world coordinate is y =
///				voxel_index_Y*dX)
/// \param[in] voxel_index_Z The index of the cube base (i.e., minimum corner)
///				in voxels (the equivalent minimal world coordinate is z =
///				voxel_index_Z*dX)
/// \param[in] tri_verts  A vector containing 3 float3 elements. Each float3
///				element represents the (x,y,z) coordinate of each vertex of
///				the triangle.
/// \param[in] tri_normal A float3 number representing the normal vector of the
///				triangle. Note that it need not be normalized.
/// \param[in] dX The dimension of the cube in world coordinates (usually in m).
///
/// \param[out] Returns true if the triangle intersects the cube and false
///				otherwise.
///////////////////////////////////////////////////////////////////////////////
__device__ __host__ bool test_triangle_box_intersection_6_sep(
                          const int voxel_index_X,
                          const int voxel_index_Y,
                          const int voxel_index_Z,
                          const float3* tri_verts,
                          const float3 tri_normal,
                          const float dX);

///////////////////////////////////////////////////////////////////////////////
/// The same function as test_triangle_box_intersection_6_sep(), but everything
///	is done in double precision - CUDA failed some cases in single precision
///	compared to C/C++.
///
/// See test_triangle_box_intersection_6_sep() for argument list and details.
///////////////////////////////////////////////////////////////////////////////
__device__ __host__ bool test_triangle_box_intersection_6_sep_double(
                          const int voxel_index_X,
                          const int voxel_index_Y,
                          const int voxel_index_Z,
                          const float3* tri_verts,
                          const double3 tri_normal,
                          const double dX);

void calcBoundaryValuesInplace(unsigned char* h_position_idx,
                               unsigned long long dim_x,
                               unsigned long long dim_y,
                               unsigned long long dim_z);

void calcMaterialIndicesInplace(unsigned char* h_material_idx,
                                const unsigned char* h_position_idx,
                                unsigned long long dim_x,
                                unsigned long long dim_y,
                                unsigned long long dim_z);

__global__ void calcBoundaryValuesInplaceKernel(unsigned char* h_position_idx,
                                                unsigned long long dim_x,
                                                unsigned long long dim_y,
                                                unsigned long long dim_z);


#endif
