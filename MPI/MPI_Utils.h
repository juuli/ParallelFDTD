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
// (C) Sebastian Prepelita, Nov. 2015
// Aalto University School of Science
//
// Functions used to split and compute with MPI using the PadallelFDTD
// Finite-Difference Time-Domain simulation library.
//
//  For test scenarios, see the end of this file.
//
//  Needs a GPU-aware MPI library, for example MVAPICH at least v1.8.
//       Best with GPUDirect-RDMA libraries, like MVAPICH2.0-GDR.
//
//  Functions created also for regular MPI libraries for testing (less
// optimized).
//
///////////////////////////////////////////////////////////////////////////////
#include <mpi.h>
#include <string>
#include "sha256.h"

// ParallelFDTD libs:
#include "../src/kernels/cudaMesh.h"

/*/////////////////////////////////////////////////////////////////////////////
 * Function switches halos between MPI domains - single precision.
 *
 *  NOTE 1: The function assumes the partitioning is done in Z direction based
 *  on function cudaMesh.getPartitionIndexing().
 *     This means the device[0] holds the lowest (in Z direction)
 *     DEV_partition and device[No_of_used_dev-1] holds the highest
 *     (in Z direction) DEV_partition.
 *
 *   NOTE 2: Function needs to be compiled at least with a GPU-aware MPI library
 *     For example, MVAPICH at least v1.8.
 *       Best with GPUDirect-RDMA libraries, like MVAPICH2.0-GDR.
 *
 *   NOTE 3: To avoid if's, the function does not check that the mesh is single!
 *
 *   NOTE 4: Although recommended by the MVAPICH2.0-GDR user guide, code crashes
 *   if cudaSetDevice() is done before sending receiving. Thus, lines commented.
 *   However, it is unclear if this is safe - emailed MVAPICH and said is fine.
 *     Error: (cudaEventRecord failed @ ibv_cuda_rndv.c:636). This is due to
 *     sending from one device and receiving from the other
 *
 *     UPDATE June 2016: for receiving from below, the device is set, otherwise
 *     an error is thrown - [g46:mpi_rank_0][dreg_register] [src/mpid/ch3/
 *     channels/common/src/reg_cache/dreg.c:1024] cuda failed with 500
 *
 *Use this at least after pointers are flipped through a flipPressurePointers()
 * function for example.
 * It can also be used before or after inner (device) halos are switched.
 *
 * Easiest to use with function launchFDTD3dStep_single() in kernels3d.h
 *
 *      e.g.: for(int step=0;step<N;step++){
 *          time = launchFDTD3dStep_single(..., step,...);
 *        time += MPI_switch_Halos_single(..., step,...);
 *      }
 *
 *  Functioning: first down sends up (first, up listens to down), then up sends
 *  down (first, down listens to up).
 *
 *  Function logs at LOG_TRACE level.
 *
 * \param d_mesh [IN]: CudaMesh containing the simulation domain
 * \param step [IN]: The current step. Used in logging.
 * \param MPI_rank [IN]: The current MPI_rank.
 * \param MPI_rank_neigbor_down [IN]: The MPI rank of MPI domain below.
 *       Function expects -2 if MPI domain does not have such a neighbor.
 *       Comes from MPI_domain_get_useful_data() function.
 * \param MPI_rank_neigbor_up [IN]: The MPI rank of MPI domain above.
 *       Function expects -2 if MPI domain does not have such a neighbor.
 *       Comes from MPI_domain_get_useful_data() function.
 *
 * \returns: The time for MPI halo switching.
 */
float MPI_switch_Halos_single(CudaMesh* d_mesh, unsigned int step,
            int MPI_rank,
            int MPI_rank_neigbor_down,
            int MPI_rank_neigbor_up);

/*/////////////////////////////////////////////////////////////////////////////
 * Function switches halos between MPI domains - double precision.
 *
 *  NOTE 1: The function assumes the partitioning is done in Z direction based
 *  on function cudaMesh.getPartitionIndexing().
 *     This means the device[0] holds the lowest (in Z direction)
 *     DEV_partition and device[No_of_used_dev-1] holds the highest
 *     (in Z direction) DEV_partition.
 *
 *   NOTE 2: Function needs to be compiled at least with a GPU-aware MPI library
 *     For example, MVAPICH at least v1.8.
 *       Best with GPUDirect-RDMA libraries, like MVAPICH2.0-GDR.
 *
 *   NOTE 3: To avoid if's, the function does not check that the mesh is double!
 *
 *   NOTE 4: Although recommended by the MVAPICH2.0-GDR user guide, code crashes
 *   if cudaSetDevice() is done before sending receiving. Thus, lines commented.
 *   However, it is unclear if this is safe - emailed MVAPICH but no answer.
 *     Error: (cudaEventRecord failed @ ibv_cuda_rndv.c:636). This is due to
 *     sending from one device and receiving from the other
 *
 *Use this at least after pointers are flipped through a flipPressurePointers()
 * function for example.
 * It can also be used or after inner halos are switched.
 *
 * Easiest to use with function launchFDTD3dStep_double() in kernels3d.h
 *
 *      e.g.: for(int step=0;step<N;step++){
 *          time = launchFDTD3dStep_double(..., step, ...);
 *        time += MPI_switch_Halos_double(..., step, ...);
 *      }
 *
 *  Functioning: first down sends up (first, up listens to down), then up sends
 *  down (first, down listens to up):
 *
 *   Function logs at LOG_TRACE level.
 *
 * \param d_mesh [IN]: CudaMesh containing the simulation domain
 * \param step [IN]: The current step. Used in logging.
 * \param MPI_rank [IN]: The current MPI_rank.
 * \param MPI_rank_neigbor_down [IN]: The MPI rank of MPI domain below.
 *       Function expects -2 if MPI domain does not have such a neighbor.
 *       Comes from MPI_domain_get_useful_data() function.
 * \param MPI_rank_neigbor_up [IN]: The MPI rank of MPI domain above.
 *       Function expects -2 if MPI domain does not have such a neighbor.
 *       Comes from MPI_domain_get_useful_data() function.
 *
 * \returns: The time for MPI halo switching.
 */
float MPI_switch_Halos_double(CudaMesh* d_mesh, unsigned int step,
            int MPI_rank,
            int MPI_rank_neigbor_down,
            int MPI_rank_neigbor_up);

/*/////////////////////////////////////////////////////////////////////////////
Function extracts the boundary_id and material_id data for the current
MPI node. Also updates the Z dimension.
The node is left only with its data inside the pointers.
  Function logs at LOG_TRACE level.
param *voxelization_dim [IN/OUT]:The dimensions resulting from the voxelization
process.
      Z dimension will be updated according to MPI slice index!
param MPI_rank [IN]: The rank of the MPI node (comes from MPI_Comm_rank(...)).
param device_boundary_idx [IN]: Boundary matrix (comes from the voxelization
                process)
param device_material_idx [IN]: Material matrix (comes from the voxelization
                process)
param partition_indexing [IN]: Partitioning of domain in Z direction (comes
                from getPartitionIndexing() in cudaMesh.h)
*/
void MPI_extract_idx_data(uint3* voxelization_dim, int MPI_rank,
  unsigned char*& device_boundary_idx,
  unsigned char*& device_material_idx,
  const std::vector< std::vector< unsigned int> > &MPI_partition_indexing);

/*/////////////////////////////////////////////////////////////////////////////
Function extracts some useful and simple data on the current MPI domain.
/param MPI_partition_indexing [IN]: Partitioning of domain in Z direction
          Comes from getPartitionIndexing() in cudaMesh.h
/param MPI_rank [IN]: The current MPI rank of MPI domain.
/param MPI_size [IN]: The current size of MPI domains.
/param MPI_partition_Z_start_idx [OUT]: The first Z valid (i.e., non-halo)
    index in the MPI domain.
/param MPI_partition_Z_end_idx [OUT]: The last Z valid (i.e., non-halo)
    index in the MPI domain.
/param MPI_rank_neigbor_down [OUT]: The MPI_rank of the below MPI domain.
    -2 is used if no neighbor exists below.
/param MPI_rank_neigbor_up [OUT]: The MPI_rank of the above MPI domain.
    -2 is used if no neighbor exists above.
*/
void MPI_domain_get_useful_data(
  const std::vector< std::vector< mesh_size_t> > &MPI_partition_indexing,
  int MPI_rank, int MPI_size,
  mesh_size_t* MPI_partition_Z_start_idx,
  mesh_size_t* MPI_partition_Z_end_idx,
  int* MPI_rank_neigbor_down, int* MPI_rank_neigbor_up );

/*/////////////////////////////////////////////////////////////////////////////
 Function checks whether receiver is in the current MPI domain. This is done
 based solely on the z index of the receiver.
 In case this is true, the z index of receiver is updated accordingly.
/param z_index [IN/OUT]: The z index of receiver.
/param MPI_partition_indexing [IN]: Partitioning of domain in Z direction
          Comes from getPartitionIndexing() in cudaMesh.h
/param MPI_partition_Z_start_idx [IN]: The first Z valid (i.e., non-halo)
      index in the MPI domain. Comes from MPI_domain_get_useful_data().
/param MPI_partition_Z_end_idx [IN]: The last Z valid (i.e., non-halo)
      index in the MPI domain. Comes from MPI_domain_get_useful_data().
/param MPI_rank [IN]: The current MPI rank of MPI domain.
/returns: true if receiver is inside MPI domain. false otherwise.
 */
bool MPI_check_receiver(int* z_index,
  const std::vector< std::vector< mesh_size_t > > &MPI_partition_indexing,
  unsigned int MPI_partition_Z_start_idx,
  unsigned int MPI_partition_Z_end_idx, int MPI_rank);

/*/////////////////////////////////////////////////////////////////////////////
 Function checks whether source is in the current MPI domain. This is done
 based solely on the z index of the source.
 In case this is true, the z index of source is updated accordingly.
/param z_index [IN/OUT]: The z index of source.
/param MPI_partition_indexing [IN]: Partitioning of domain in Z direction
          Comes from getPartitionIndexing() in cudaMesh.h
/param MPI_rank [IN]: The current MPI rank of MPI domain.
/returns: true if source is inside MPI domain. false otherwise.
*/
bool MPI_check_source(int* z_index,
  const std::vector< std::vector< mesh_size_t > > &MPI_partition_indexing,
  int MPI_rank);

/*/////////////////////////////////////////////////////////////////////////////
Function with various sanity checks on MPI domains so that things don't go
wrong.
Call this before computing - after all things are set up properly.
Feel free to add any important checks here.
param d_mesh [IN]: CudaMesh containing the simulation domain
param MPI_rank [IN]: The rank of current MPI domain.
param MPI_partition_indexing [IN]: Partitioning of domain in Z direction
          Comes from getPartitionIndexing() in cudaMesh.h.
 */
void MPI_checks(CudaMesh* d_mesh, int MPI_rank,
  const std::vector< std::vector< mesh_size_t> > &MPI_partition_indexing);

///////////////////////////////////////////////////////////////////////////////
//      Debugging functions if needed:
///////////////////////////////////////////////////////////////////////////////
/*/////////////////////////////////////////////////////////////////////////////
 * Function generates a SHA256 hash on the entire slice. Useful to compare
 * slices.
 *
 *  ! Note that the input pointer must be on a device.
 *   For host slices, see debugging_SHA256_slice_host() function.
 *
 *  ! Function does not know of dimensions so a seg fault might be thrown.
 *
/param d_pointer [IN]: The device pointer matrix.Default Fortran alignment used
            in ParallelFDTD lib.
/param dimXY [IN]: The dimension of a z slice ( = X*Y)
/param Z_slice_index [IN]: The slice that has to be printed
*/
template <typename T>
std::string debugging_SHA256_slice_device(T* d_pointer, unsigned int dimXY,
      unsigned int Z_slice_index, unsigned int dev_idx) {
  // First, get slice on host:
  T *host_slice = new T[dimXY];
  copyDeviceToHost( dimXY, host_slice, d_pointer + Z_slice_index * dimXY,
            dev_idx);
  // Get SHA256 hash:
  std::string slice_sha256 = sha256( (char *)host_slice,
      dimXY * (sizeof(T)/sizeof(char)) );
  delete[] host_slice;
  return slice_sha256;
}

/*/////////////////////////////////////////////////////////////////////////////
 * Function generates a SHA256 hash on the entire slice. Useful to compare
 * slices.
 *
 *
 *  ! Note that the input pointer must be on a host.
 *   For device slices, see debugging_SHA256_slice_device() function.
 *
 *  ! Function does not care of dimensions so a seg fault might be thrown.
 *
/param d_pointer [IN]: The device pointer matrix.Default Fortran alignment used
            in ParallelFDTD lib.
/param dimXY [IN]: The dimension of a z slice ( = X*Y)
/param Z_slice_index [IN]: The slice that has to be printed
*/
template <typename T>
std::string debugging_SHA256_slice_host(T* h_pointer, unsigned int dimXY,
    unsigned int Z_slice_index) {
  T *host_slice = h_pointer + Z_slice_index * dimXY;
  // Get SHA256 hash:
  return sha256( (char *)host_slice, dimXY * (sizeof(T)/sizeof(char)) );
}

/*/////////////////////////////////////////////////////////////////////////////
Function logs the partition indexing for debugging purposes.
  Log level is LOG_DEBUG.
/param partition_indexing_[IN]: the partition to be printed.
/param msg_[IN]: Some extra message in the logger print.
*/
void debugging_print_partion_indexing(
    std::vector< unsigned int> partition_indexing_, const char* msg_ = "");

/*/////////////////////////////////////////////////////////////////////////////
 Function used to log the first two and last two slices of current pressure for
 each SUB DOMAIN (the domains from different CUDA devices) within a MPI domain.
  Log level is LOG_DEBUG.
/param first_device_SubDomain_idx [IN]: The device index of first subdomain
                    (usually 0)
/param last_device_SubDomain_idx [IN]: The device index of last subdomain
     (usually 1 for 2 CUDA devices and 0 for 1 CUDA device on the MPI node)
/param MPI_halo_size [IN]: The size of a slice (usually dim_X*dim_Y)
/param d_mesh [IN]: The cuda mesh pointer (has the devices and subdomain
           partitioning)
/param step [IN]: The current step when the function is called
/param MPI_rank [IN]: The MPI rank of this node (domain index !)
*/
void debugging_log_pressure_slices_hash_single(
    int& first_device_SubDomain_idx,
    int& last_device_SubDomain_idx,
    int& MPI_halo_size, CudaMesh* d_mesh,
    unsigned int& step, int& MPI_rank);

/*/////////////////////////////////////////////////////////////////////////////
* Function switches halos between MPI domains - done for a non-CUDA-aware
* library and single precision.
*	Same function as MPI_switch_Halos_single(), but it uses the host as an
* intermediate step when sending/receiving data from device (pressure) data.
*
*  Should work with non-CUDA aware MPI libraries. Does blocking sends!
*
*   Additional parameter:
* \param h_buffer_blockingSR: a temporary buffer with size = dimXY already
* allocated on the host!
*/
float MPI_switch_Halos_single_host(CudaMesh* d_mesh, unsigned int step,
	float* h_buffer_blockingSR,
	int MPI_rank,
	int MPI_rank_neigbor_down,
	int MPI_rank_neigbor_up);

/*/////////////////////////////////////////////////////////////////////////////
* Function switches halos between MPI domains - done for a non-CUDA-aware
* library and double precision.
*	Same function as MPI_switch_Halos_double(), but it uses the host as an
* intermediate step when sending/receiving data from device (pressure) data.
*
*  Should work with non-CUDA aware MPI libraries. Does blocking sends!
*
*   Additional parameter:
* \param h_buffer_blockingSR: a temporary buffer with size = dimXY already
* allocated on the host!
*/
float MPI_switch_Halos_double_host(CudaMesh* d_mesh, unsigned int step,
	double* h_buffer_blockingSR,
	int MPI_rank,
	int MPI_rank_neigbor_down,
	int MPI_rank_neigbor_up);

///////////////////////////////////////////////////////////////////////////////
/*
 * Tests completed OK (i.e., same receiver responses for 1 node and (1 or 2)
 *     CUDA subdomains) for MPI update and HRTF simulations:
 *   2MPI nodes (500 samples):
 *     -OK) 2xMPI nodes, 2xCUDA subdomains, 2xGPU cards, [Nx,Ny,Nz] =
 *         [1024,1012,1012], soft source on halo, SINGLE precision,
 *           1250 receivers
 *     -OK) 2xMPI nodes, 2xCUDA subdomains, 2xGPU cards, [Nx,Ny,Nz] =
 *         [1024,1012,1012], soft source on halo, DOUBLE precision,
 *           1250 receivers
 *     -OK) 2xMPI nodes, 2xCUDA subdomains, 2xGPU cards, [Nx,Ny,Nz] =
 *         [1280,1280,1279], soft source on halo, SINGLE precision,
 *           1250 receivers
 *     -OK) 2xMPI nodes, 2xCUDA subdomains, 2xGPU cards, [Nx,Ny,Nz] =
 *         [1280,1280,1279], soft source on halo, DOUBLE precision,
 *           1250 receivers
 *
 *     -OK) 2xMPI nodes, 1xCUDA subdomains, 1xGPU cards, [Nx,Ny,Nz] =
 *         [1024,1012,1012], soft source on halo, SINGLE precision,
 *           1250 receivers
 *     -OK) 2xMPI nodes, 1xCUDA subdomains, 1xGPU cards, [Nx,Ny,Nz] =
 *         [1024,1012,1012], soft source on halo, DOUBLE precision,
 *           1250 receivers
 *     -OK) 2xMPI nodes, 1xCUDA subdomains, 1xGPU cards, [Nx,Ny,Nz] =
 *         [1280,1280,1279], soft source on halo, SINGLE precision,
 *           1250 receivers
 *
 *   3MPI nodes (500 samples):
 *    -OK) 3xMPI nodes, 2xCUDA subdomains, 2xGPU cards, [Nx,Ny,Nz] =
 *        [1024,1012,1012], soft source mid dom, DOUBLE precision,
 *          1250 receivers
 *    -OK) 3xMPI nodes, 2xCUDA subdomains, 2xGPU cards, [Nx,Ny,Nz] =
 *        [1280,1280,1279], soft source mid dom , SINGLE precision,
 *          1250 receivers
 *
 *    \ More tests: /
 *     -------------
 *   6MPI nodes (500 samples):
 *    -OK) 6xMPI nodes, 2xCUDA subdomains, 2xGPU cards, [Nx,Ny,Nz] =
 *        [1024,1012,1012], soft source dom 3 (z=2) , DOUBLE precision,
 *          1250 receivers
 *    -OK) 6xMPI nodes, 2xCUDA subdomains, 2xGPU cards, [Nx,Ny,Nz] =
 *        [1280,1280,1279], soft source on halo R3 , SINGLE precision,
 *          1250 receivers
 *
 *   7MPI nodes (500 samples):
 *    -OK) 7xMPI nodes, 2xCUDA subdomains, 2xGPU cards, [Nx,Ny,Nz] =
 *        [1024,1012,1012], soft source dom 3 , DOUBLE precision,
 *          1250 receivers
 *    -OK) 7xMPI nodes, 2xCUDA subdomains, 2xGPU cards, [Nx,Ny,Nz] =
 *        [1280,1280,1279], soft source mid dom 3 , SINGLE precision,
 *          1250 receivers
 *
 * MPI Message sizes:
 *     - [1280,1280,1279] SINGLE --> MPI message size of ~ 6MB
 *     - [1024,1012,1012] SINGLE --> MPI message size of ~ 4MB
 *     - [1024,1012,1012] DOUBLE --> MPI message size of ~ 8MB
 */
/////////////////////////////////////////////////////////////////////////////
