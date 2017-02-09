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
///////////////////////////////////////////////////////////////////////////////

#include "MPI_Utils.h"

#include "sha256.h"
#include <stdio.h>
#include <cmath>
#include "../src/logger.h"

///////////////////////////////////////////////////////////////////////////////
float MPI_switch_Halos_single(CudaMesh* d_mesh, unsigned int step,
            int MPI_rank,
            int MPI_rank_neigbor_down,
            int MPI_rank_neigbor_up) {
  clock_t start_t;
  clock_t end_t;

  start_t = clock();

  // MPI message tagging:
  const int MPI_HALOTAG = 1;
  /// Prerequisite variables:
  int MPI_halo_size = d_mesh->getDimXY();
  int first_device_SubDomain_idx = 0,
      last_device_SubDomain_idx = d_mesh->getNumberOfPartitions() - 1;
  int size_lastSubDomain =
      d_mesh->getNumberOfElementsAt(last_device_SubDomain_idx);
  // Use this if halo debugging is needed (works on any domain):
  //debugging_log_pressure_slices_hash_single(first_device_SubDomain_idx,
  //    last_device_SubDomain_idx, MPI_halo_size, d_mesh, step, MPI_rank);

  // Receive status info:
  MPI_Status MPI_rec_status_from_DOWN, MPI_rec_status_from_UP;
/// Step 0: UP listens to DOWN for >> DOWN->UP << message if
/// there is a neighbor down:
  if (MPI_rank_neigbor_down != -2){
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - [STEP %04d (0)] RANK_%d"
        " waits data from UNDER neighbor RANK_%d\nMPI_halo_size = %d")
        %step %MPI_rank %MPI_rank_neigbor_down %MPI_halo_size;
    // MPI_partition receives on the lowest DEVICE_SUBDOMAIN:
    cudaSetDevice(d_mesh->getDeviceAt(first_device_SubDomain_idx));
    // Put data on slice 0 of DEVICE_SUBDOMAIN 0:
    MPI_Recv((float *)d_mesh->getPressurePtrAt(first_device_SubDomain_idx),
        MPI_halo_size, MPI_FLOAT, MPI_rank_neigbor_down, MPI_HALOTAG,
        MPI_COMM_WORLD, &MPI_rec_status_from_DOWN);
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - (0) Receive DOWN->UP"
        " command done!");
  } else {
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - [STEP %04d (0)] RANK_%d"
        " doesn't wait DOWN data since it has no UNDER neighbor.")
        %step %MPI_rank;
  }
/// Step 1: DOWN sends MPI_HALO data to UP for >> DOWN->UP << message if
/// there is a neighbor up:
  if (MPI_rank_neigbor_up != -2){
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - [STEP %04d  (1)] RANK_%d"
        " sends data to TOP neighbor RANK_%d\nMPI_halo_size = %d")
        %step %MPI_rank %MPI_rank_neigbor_up %MPI_halo_size;
    // MPI_partition sends data from the greatest DEVICE_SUBDOMAIN:
    //cudaSetDevice(d_mesh->getDeviceAt(last_device_SubDomain_idx));
    // Send data from penultimate slice of last DEVICE_SUBDOMAIN:
    MPI_Send((float *)d_mesh->getPressurePtrAt(last_device_SubDomain_idx)+
                    size_lastSubDomain - 2*MPI_halo_size,
        MPI_halo_size, MPI_FLOAT, MPI_rank_neigbor_up, MPI_HALOTAG,
        MPI_COMM_WORLD);
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - (1) Send DOWN->UP"
        " command done!");
  } else {
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - [STEP %04d (1)] RANK_%d"
        " doesn't send data UP since it has no TOP neighbor.")
        %step %MPI_rank;
  }
/// Step 2: DOWN listens to UP for >> UP->DOWN << message if
/// there is a neighbor up:
  if (MPI_rank_neigbor_up != -2){
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - [STEP %04d (2)] RANK_%d"
        " waits data from TOP neighbor RANK_%d\nMPI_halo_size = %d")
            %step %MPI_rank %MPI_rank_neigbor_up %MPI_halo_size;
    // MPI_partition receives data to the greatest DEV_partition:
    //cudaSetDevice(d_mesh->getDeviceAt(last_device_SubDomain_idx));
    // Put data on last slice of last DEVICE_SUBDOMAIN:
    MPI_Recv((float *)d_mesh->getPressurePtrAt(last_device_SubDomain_idx)+
                    size_lastSubDomain - 1*MPI_halo_size,
        MPI_halo_size, MPI_FLOAT, MPI_rank_neigbor_up, MPI_HALOTAG,
        MPI_COMM_WORLD, &MPI_rec_status_from_UP);
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - (2) Receive UP->DOWN"
        " command done!");
  } else {
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - [STEP %04d (2)] RANK_%d"
        " doesn't wait UP data since it has no TOP neighbor.")
            %step %MPI_rank;
  }
/// Step 3: UP sends MPI_HALO data to DOWN for >> UP->DOWN << message if
/// there is a neighbor down:
  if (MPI_rank_neigbor_down != -2){
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - [STEP %04d (3)] RANK_%d"
        " sends data to UNDER neighbor RANK_%d\nMPI_halo_size = %d")
            %step %MPI_rank %MPI_rank_neigbor_up %MPI_halo_size;
    //cudaSetDevice(d_mesh->getDeviceAt(first_device_SubDomain_idx));
    // Send data from second slice of first DEVICE_SUBDOMAIN:
    MPI_Send((float *)d_mesh->getPressurePtrAt(first_device_SubDomain_idx)+
                      MPI_halo_size, MPI_halo_size,
       MPI_FLOAT, MPI_rank_neigbor_down, MPI_HALOTAG, MPI_COMM_WORLD);
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - (3) Send DOWN->UP"
        " command done!");
  } else {
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - [STEP %04d (3)] RANK_%d"
        " doesn't send data DOWN since it has no UNDER neighbor.")
            %step %MPI_rank;
  }
  log_msg<LOG_TRACE>(L"\tMPI_switch_Halos_single - [STEP %04d (4)] Waiting"
      " barrier to complete data sending.") %step;
  // Wait for ALL MPI data transfers to end:
  MPI_Barrier(MPI_COMM_WORLD);
  end_t = clock()-start_t;
  // Use this if halo debugging is needed (works on any domain):
  //debugging_log_pressure_slices_hash_single(first_device_SubDomain_idx,
  //    last_device_SubDomain_idx, MPI_halo_size, d_mesh, step, MPI_rank);
  return ((float)end_t/CLOCKS_PER_SEC);
}

///////////////////////////////////////////////////////////////////////////////
float MPI_switch_Halos_double(CudaMesh* d_mesh, unsigned int step,
            int MPI_rank,
            int MPI_rank_neigbor_down,
            int MPI_rank_neigbor_up) {
  clock_t start_t;
  clock_t end_t;
  start_t = clock();

  // MPI message tagging:
  const int MPI_HALOTAG = 1;
  /// Prerequisite variables:
  int MPI_halo_size = d_mesh->getDimXY();
  int first_device_SubDomain_idx = 0,
      last_device_SubDomain_idx = d_mesh->getNumberOfPartitions() - 1;
  int size_lastSubDomain =
      d_mesh->getNumberOfElementsAt(last_device_SubDomain_idx);
  // If halo debugging is needed (works on any domain), have a look at
  // debugging_log_pressure_slices_hash_single();
  // Receive status info"
  MPI_Status MPI_rec_status_from_DOWN, MPI_rec_status_from_UP;
/// Step 0: UP listens to DOWN for >> DOWN->UP << message if
//  there is a neighbor down:
  if (MPI_rank_neigbor_down != -2){
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - [STEP %04d (0)] RANK_%d"
        " waits data from UNDER neighbor RANK_%d")
            %step %MPI_rank %MPI_rank_neigbor_down;
    // MPI_partition receives on the lowest DEVICE_SUBDOMAIN:
    //cudaSetDevice(d_mesh->getDeviceAt(first_device_SubDomain_idx));
    // Put data on slice 0 of DEVICE_SUBDOMAIN 0:
    MPI_Recv((double *)d_mesh->getPressureDoublePtrAt(first_device_SubDomain_idx),
        MPI_halo_size, MPI_DOUBLE, MPI_rank_neigbor_down,
        MPI_HALOTAG, MPI_COMM_WORLD, &MPI_rec_status_from_DOWN);
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - (0) Receive DOWN->UP"
        " command done!");
  } else {
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - [STEP %04d (0)] RANK_%d"
        " doesn't wait DOWN data since it has no UNDER neighbor.")
            %step %MPI_rank;
  }
/// Step 1: DOWN sends MPI_HALO data to UP for >> DOWN->UP << message if
/// there is a neighbor up:
  if (MPI_rank_neigbor_up != -2){
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - [STEP %04d  (1)] RANK_%d"
        " sends data to TOP neighbor RANK_%d")
        %step %MPI_rank %MPI_rank_neigbor_up;
    // MPI_partition sends data from the greatest DEVICE_SUBDOMAIN:
    //cudaSetDevice(d_mesh->getDeviceAt(last_device_SubDomain_idx));
    // Send data from penultimate slice of last DEVICE_SUBDOMAIN:
    MPI_Send((double *)d_mesh->getPressureDoublePtrAt(last_device_SubDomain_idx)+
        size_lastSubDomain - 2*MPI_halo_size, MPI_halo_size,
        MPI_DOUBLE, MPI_rank_neigbor_up, MPI_HALOTAG, MPI_COMM_WORLD);
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - (1) Send DOWN->UP"
        " command done!");
  } else {
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - [STEP %04d (1)] RANK_%d"
        " doesn't send data UP since it has no TOP neighbor.")
        %step %MPI_rank;
  }
/// Step 2: DOWN listens to UP for >> UP->DOWN << message if
/// there is a neighbor up:
  if (MPI_rank_neigbor_up != -2){
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - [STEP %04d (2)] RANK_%d"
        " waits data from TOP neighbor RANK_%d")
        %step %MPI_rank %MPI_rank_neigbor_up;
    // MPI_partition receives data to the greatest DEV_partition:
    //cudaSetDevice(d_mesh->getDeviceAt(last_device_SubDomain_idx));
    // Put data on last slice of last DEVICE_SUBDOMAIN:
    MPI_Recv((double *)d_mesh->getPressureDoublePtrAt(last_device_SubDomain_idx)+
                    size_lastSubDomain - 1*MPI_halo_size,
        MPI_halo_size, MPI_DOUBLE, MPI_rank_neigbor_up, MPI_HALOTAG,
        MPI_COMM_WORLD, &MPI_rec_status_from_UP);
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - (2) Receive UP->DOWN"
        " command done!");
  } else {
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - [STEP %04d (2)] RANK_%d"
        " doesn't wait UP data since it has no TOP neighbor.")
        %step %MPI_rank;
  }
/// Step 3: UP sends MPI_HALO data to DOWN for >> UP->DOWN << message if
/// there is a neighbor down:
  if (MPI_rank_neigbor_down != -2){
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - [STEP %04d (3)] RANK_%d"
        " sends data to UNDER neighbor RANK_%d")
        %step %MPI_rank %MPI_rank_neigbor_down;
    //cudaSetDevice(d_mesh->getDeviceAt(first_device_SubDomain_idx));
    // Send data from second slice of first DEVICE_SUBDOMAIN:
    MPI_Send((double *)d_mesh->getPressureDoublePtrAt(first_device_SubDomain_idx)+
         MPI_halo_size, MPI_halo_size, MPI_DOUBLE,
         MPI_rank_neigbor_down, MPI_HALOTAG, MPI_COMM_WORLD);
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - (3) Send DOWN->UP"
        " command done!");
  } else {
    log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - [STEP %04d (3)] RANK_%d"
        " doesn't send data DOWN since it has no UNDER neighbor.")
            %step %MPI_rank;
  }
  log_msg<LOG_TRACE>(L"\tMPI_switch_Halos_double - [STEP %04d (4)] Waiting"
      " barrier to complete data sending.") %step;
  // Wait for ALL MPI data transfers to end:
  MPI_Barrier(MPI_COMM_WORLD);
  end_t = clock()-start_t;
  // If halo debugging is needed (works on any domain), have a look at
  // debugging_log_pressure_slices_hash_single();
  return ((float)end_t/CLOCKS_PER_SEC);
}

///////////////////////////////////////////////////////////////////////////////
void MPI_extract_idx_data(uint3* voxelization_dim, int MPI_rank,
  unsigned char*& device_boundary_idx,
  unsigned char*& device_material_idx,
  const std::vector< std::vector< mesh_size_t > > &MPI_partition_indexing) {
  unsigned int dimXY = (int)(*voxelization_dim).x*(int)(*voxelization_dim).y;
  // Extract boundary matrix for this process: get the partin indices first:
  std::vector<mesh_size_t> curent_slice_indices =
      MPI_partition_indexing.at(MPI_rank);
  // The boundary matrix is on GPU ( d_postition_idx_ )
  int MPI_offset = dimXY * curent_slice_indices.at(0);
  int MPI_mem_size = dimXY * curent_slice_indices.size();
  // Move data to CPU to save memory on the GPU:
  unsigned char* h_idx_MPI_chunk = new unsigned char[MPI_mem_size];
  copyDeviceToHost(MPI_mem_size, h_idx_MPI_chunk, device_boundary_idx+
                            MPI_offset, 0);

  log_msg<LOG_TRACE>(L"MPI_extract_idx_data - RANK%d : Extracted data from "
  "%d to %d equivalent to %d memsize (dimXY = %d). "
      "Total size of idx_matrix is %d") %MPI_rank %MPI_offset
    %(MPI_offset+MPI_mem_size) %MPI_mem_size %dimXY
  %((int)(*voxelization_dim).x*(int)(*voxelization_dim).y*(int)(*voxelization_dim).z);
  // Now destroy old data on the GPU to make room:
  destroyMem(device_boundary_idx);
  // Put it back:
  device_boundary_idx = toDevice(MPI_mem_size, h_idx_MPI_chunk , 0);
  // Do the same for material matrix (should be enough memory this time to
  // do it on the device):
  copyDeviceToHost(MPI_mem_size, h_idx_MPI_chunk, device_material_idx+
                            MPI_offset, 0);
  destroyMem(device_material_idx);
  device_material_idx = toDevice(MPI_mem_size, h_idx_MPI_chunk , 0);
  // Update z dimension:
  (*voxelization_dim).z = curent_slice_indices.size();
  // Clear host data:
  delete[] h_idx_MPI_chunk;
}

///////////////////////////////////////////////////////////////////////////////
void MPI_domain_get_useful_data(
  const std::vector< std::vector< mesh_size_t> > &MPI_partition_indexing,
  int MPI_rank, int MPI_size, mesh_size_t* MPI_partition_Z_start_idx,
  mesh_size_t* MPI_partition_Z_end_idx, int* MPI_rank_neigbor_down,
  int* MPI_rank_neigbor_up ) {
  // Start with dedicated values:
  // The MPI Z-slice indexes without guard/ghost slices:
  *MPI_partition_Z_start_idx = -1;
  *MPI_partition_Z_end_idx = -1;
  // The MPI_neighbors:
  *MPI_rank_neigbor_down = -2;
  *MPI_rank_neigbor_up = -2; // -1 might be an erroneous value!
  // First subdomain:
  if(MPI_rank == 0){
    *MPI_partition_Z_start_idx= MPI_partition_indexing.at(MPI_rank).at(0);
  } else{
    *MPI_partition_Z_start_idx= MPI_partition_indexing.at(MPI_rank).at(1);
    *MPI_rank_neigbor_down = MPI_rank - 1;
    if (*MPI_rank_neigbor_down < 0) // sanity check...
    {
      log_msg<LOG_ERROR>(L"MPI_domain_get_useful_data - The below MPI "
        "domain index is negative(%d)! Something went wrong!!")
            %(*MPI_rank_neigbor_down) %MPI_size;
      throw 4;
    }
  }
  // Last subdomain:
  if(MPI_rank == MPI_size - 1){
    *MPI_partition_Z_end_idx = MPI_partition_indexing.at(MPI_rank).back();
  } else{
    *MPI_partition_Z_end_idx = MPI_partition_indexing.at(MPI_rank).at(
              MPI_partition_indexing.at(MPI_rank).size() - 2
                                     );
    *MPI_rank_neigbor_up = MPI_rank + 1;
    if (*MPI_rank_neigbor_up >= MPI_size) // sanity check...
    {
      log_msg<LOG_ERROR>(L"MPI_domain_get_useful_data - The above MPI "
        "domain index is out of range (%d, MPI_size = %d)! "
        "Something went wrong!!") %(*MPI_rank_neigbor_up) %MPI_size;
      throw 4;
    }
  }
  // Sanity checks again:
  if ((*MPI_partition_Z_start_idx < 0) || (*MPI_partition_Z_end_idx < 0)){
    log_msg<LOG_ERROR>(L"MPI_domain_get_useful_data - The start index "
      "(%d) or end index (%d) of current domain is negative! "
        "Something went wrong!!")
          %(*MPI_partition_Z_start_idx) %(*MPI_partition_Z_end_idx);
    throw 4;
  }
}

///////////////////////////////////////////////////////////////////////////////
bool MPI_check_receiver(int* z_index,
  const std::vector< std::vector< mesh_size_t> > &MPI_partition_indexing,
  unsigned int MPI_partition_Z_start_idx,
  unsigned int MPI_partition_Z_end_idx, int MPI_rank) {
  if ( ((*z_index) >= MPI_partition_Z_start_idx) &&
      ((*z_index) <= MPI_partition_Z_end_idx) )
  {
    // Adjust receiver for MPI subdomain:
    *z_index = *z_index - MPI_partition_indexing.at(MPI_rank).at(0);
    return true;
  }
  return false;
}

///////////////////////////////////////////////////////////////////////////////
bool MPI_check_source(int* z_index,
  const std::vector< std::vector< mesh_size_t> > &MPI_partition_indexing,
  int MPI_rank)
{
  if ( ((*z_index) >= MPI_partition_indexing.at(MPI_rank).at(0)) &&
      ((*z_index) <= MPI_partition_indexing.at(MPI_rank).back()) )
  {
    // Adjust source for MPI subdomain:
    *z_index = *z_index - MPI_partition_indexing.at(MPI_rank).at(0);
    return true;
  }
  return false;
}

///////////////////////////////////////////////////////////////////////////////
void MPI_checks(CudaMesh* d_mesh, int MPI_rank,
  const std::vector< std::vector< mesh_size_t> > &MPI_partition_indexing) {
  //Check size (just raise warning now-might work for larger message sizes):
  int MPI_halo_size_MB = d_mesh->getDimXY() *
      (d_mesh->isDouble() ? sizeof(double) : sizeof(float)) * 1e-6;
  if (MPI_halo_size_MB >= 2000){
    log_msg<LOG_WARNING>(L"FDTD_MPI - !! Halo size is greater than 2 GB"
      " (size_MB = %d)! This should be unsupported by the MVAPICH2-GDR"
      " implementation so the program should crash-boom-bang. TODO: "
      "split sending into smaller messages of 2GB!") %MPI_halo_size_MB;
  }
  // Check that there is enough slices for inner device subdomains
  // (the ones on the same MPI node but on different CUDA cards):
  if ((d_mesh->getNumberOfPartitions() ==2) &&
      (MPI_partition_indexing.at(MPI_rank).size()<4)){
    log_msg<LOG_ERROR>(L"FDTD_MPI - The size (in Z slices) of the current "
        "MPI_domain is too small (%d) so that two CUDA sub-domains "
        "will fit in!") %MPI_partition_indexing.at(MPI_rank).size();
  }
}

///////////////////////////////////////////////////////////////////////////////
void debugging_print_partion_indexing(
    std::vector< unsigned int> partition_indexing_,
    const char* msg_) {
  int partition_size = partition_indexing_.size();
  // Build message:
  char partition_printout[partition_size*10 + 20], number_[10];
  partition_printout[0] = '\0'; number_[0] = '\0';
  for(int idx = 0; idx< partition_size; idx++ ){
    sprintf(number_, ",%d", partition_indexing_.at(idx));
    strcat(partition_printout, number_);
  }
  log_msg<LOG_DEBUG>(L"\t@ DEBUG_PRINTOUT_PARTION Indexing %s "
      "(size %d):\n[%s]") %msg_ %partition_size %partition_printout;
}

///////////////////////////////////////////////////////////////////////////////
void debugging_log_pressure_slices_hash_single(int& first_device_SubDomain_idx,
                         int& last_device_SubDomain_idx,
                         int& MPI_halo_size,
                         CudaMesh* d_mesh,
                         unsigned int& step,
                         int& MPI_rank) {
  ///// DEBUG ///////// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG //
  // This debugging should help see where pressure slices reach (maybe
  // they reach the wrong devices...)
  int slice_idx_;
  // CURRENT PRESSURE (first 2 slices):
  cudaSetDevice(d_mesh->getDeviceAt(first_device_SubDomain_idx));
  slice_idx_ = 0;
  log_msg<LOG_DEBUG>(L"   STEP %04d R%d_DEV_%d Sw.Hl. : current_P[#%d]:"
    " %s") %step %MPI_rank %first_device_SubDomain_idx %slice_idx_
    %debugging_SHA256_slice_device( (float *)d_mesh->getPressurePtrAt(first_device_SubDomain_idx),
      MPI_halo_size, slice_idx_, first_device_SubDomain_idx).c_str();
  slice_idx_ = 1;
  log_msg<LOG_DEBUG>(L"   STEP %04d R%d_DEV_%d Sw.Hl. : current_P[#%d]:"
    " %s") %step %MPI_rank %first_device_SubDomain_idx %slice_idx_
    %debugging_SHA256_slice_device( (float *)d_mesh->getPressurePtrAt(first_device_SubDomain_idx),
      MPI_halo_size, slice_idx_, first_device_SubDomain_idx).c_str();
  cudaSetDevice(d_mesh->getDeviceAt(last_device_SubDomain_idx));
  slice_idx_ = 0;
  log_msg<LOG_DEBUG>(L"   STEP %04d R%d_DEV_%d Sw.Hl. : current_P[#%d]:"
    " %s") %step %MPI_rank %last_device_SubDomain_idx %slice_idx_
    %debugging_SHA256_slice_device( (float *)d_mesh->getPressurePtrAt(last_device_SubDomain_idx),
      MPI_halo_size, slice_idx_, last_device_SubDomain_idx).c_str();
  slice_idx_ = 1;
  log_msg<LOG_DEBUG>(L"   STEP %04d R%d_DEV_%d Sw.Hl. : current_P[#%d]:"
    " %s") %step %MPI_rank %last_device_SubDomain_idx %slice_idx_
    %debugging_SHA256_slice_device( (float *)d_mesh->getPressurePtrAt(last_device_SubDomain_idx),
      MPI_halo_size, slice_idx_, last_device_SubDomain_idx).c_str();
  // CURRENT PRESSURE (last 2 slices):
  std::vector< std::vector< mesh_size_t > > mesh_partitioning_i =
      d_mesh->partition_indexing_;
  cudaSetDevice(d_mesh->getDeviceAt(first_device_SubDomain_idx));
  slice_idx_=mesh_partitioning_i.at(first_device_SubDomain_idx).at(mesh_partitioning_i.at(first_device_SubDomain_idx).size()-1);
  log_msg<LOG_DEBUG>(L"   STEP %04d R%d_DEV_%d Sw.Hl. : current_P[#%d]: %s")
    %step %MPI_rank %first_device_SubDomain_idx %slice_idx_
    %debugging_SHA256_slice_device( (float *)d_mesh->getPressurePtrAt(first_device_SubDomain_idx),
      MPI_halo_size, slice_idx_, first_device_SubDomain_idx).c_str();
  slice_idx_=mesh_partitioning_i.at(first_device_SubDomain_idx).at(mesh_partitioning_i.at(first_device_SubDomain_idx).size()-2);
  log_msg<LOG_DEBUG>(L"   STEP %04d R%d_DEV_%d Sw.Hl. : current_P[#%d]: %s")
    %step %MPI_rank %first_device_SubDomain_idx %slice_idx_
    %debugging_SHA256_slice_device( (float *)d_mesh->getPressurePtrAt(first_device_SubDomain_idx),
      MPI_halo_size, slice_idx_, first_device_SubDomain_idx).c_str();
  cudaSetDevice(d_mesh->getDeviceAt(last_device_SubDomain_idx));
  slice_idx_=mesh_partitioning_i.at(last_device_SubDomain_idx).at(mesh_partitioning_i.at(last_device_SubDomain_idx).size()-1);
  log_msg<LOG_DEBUG>(L"   STEP %04d R%d_DEV_%d Sw.Hl. : current_P[#%d]: %s")
    %step %MPI_rank %last_device_SubDomain_idx %slice_idx_
    %debugging_SHA256_slice_device( (float *)d_mesh->getPressurePtrAt(last_device_SubDomain_idx),
      MPI_halo_size, slice_idx_, last_device_SubDomain_idx).c_str();
  slice_idx_=mesh_partitioning_i.at(last_device_SubDomain_idx).at(mesh_partitioning_i.at(last_device_SubDomain_idx).size()-2);
  log_msg<LOG_DEBUG>(L"   STEP %04d R%d_DEV_%d Sw.Hl. : current_P[#%d]: %s")
    %step %MPI_rank %last_device_SubDomain_idx %slice_idx_
    %debugging_SHA256_slice_device( (float *)d_mesh->getPressurePtrAt(last_device_SubDomain_idx),
      MPI_halo_size, slice_idx_, last_device_SubDomain_idx).c_str();
  ///// DEBUG ///////// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG //
}

///////////////////////////////////////////////////////////////////////////////
float MPI_switch_Halos_single_host(CudaMesh* d_mesh, unsigned int step,
	float* h_buffer_blockingSR, // allocated pointer on host (size = dimXY) to send/receive data from
	//float* h_receive, // allocated pointer to receive data from
	int MPI_rank,
	int MPI_rank_neigbor_down,
	int MPI_rank_neigbor_up)
{
	clock_t start_t;
	clock_t end_t;

	start_t = clock();

	// MPI message tagging:
	const int MPI_HALOTAG = 1;
	/// Prerequisite variables:
	int MPI_halo_size = d_mesh->getDimXY();
	int first_device_SubDomain_idx = 0,
		last_device_SubDomain_idx = d_mesh->getNumberOfPartitions() - 1;
	int size_lastSubDomain =
		d_mesh->getNumberOfElementsAt(last_device_SubDomain_idx);
	// Use this if halo debugging is needed (works on any domain):
	//debugging_log_pressure_slices_hash_single(first_device_SubDomain_idx,
	//    last_device_SubDomain_idx, MPI_halo_size, d_mesh, step, MPI_rank);

	// Receive status info:
	MPI_Status MPI_rec_status_from_DOWN, MPI_rec_status_from_UP;
	/// Step 0: UP listens to DOWN for >> DOWN->UP << message if
	/// there is a neighbor down:
	if (MPI_rank_neigbor_down != -2) {
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - [STEP %04d (0)] RANK_%d"
			" waits data from UNDER neighbor RANK_%d\nMPI_halo_size = %d")
			% step %MPI_rank %MPI_rank_neigbor_down %MPI_halo_size;
		//log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - [STEP %04d (0)] RANK_%d"
		//				" Sets device %d")
		//				%step %MPI_rank %first_device_SubDomain_idx;
		// MPI_partition receives on the lowest DEVICE_SUBDOMAIN:
		//cudaSetDevice(d_mesh->getDeviceAt(first_device_SubDomain_idx));
		// Recieve:
		MPI_Recv(h_buffer_blockingSR,
			MPI_halo_size, MPI_FLOAT, MPI_rank_neigbor_down, MPI_HALOTAG,
			MPI_COMM_WORLD, &MPI_rec_status_from_DOWN);
		// Put data on slice 0 of DEVICE_SUBDOMAIN 0:
		copyHostToDevice(MPI_halo_size,
			(float *)d_mesh->getPressurePtrAt(first_device_SubDomain_idx),
			h_buffer_blockingSR, first_device_SubDomain_idx);
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - (0) Receive DOWN->UP"
			" command done!");
	}
	else {
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - [STEP %04d (0)] RANK_%d"
			" doesn't wait DOWN data since it has no UNDER neighbor.")
			% step %MPI_rank;
	}
	/// Step 1: DOWN sends MPI_HALO data to UP for >> DOWN->UP << message if
	/// there is a neighbor up:
	if (MPI_rank_neigbor_up != -2) {
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - [STEP %04d  (1)] RANK_%d"
			" sends data to TOP neighbor RANK_%d\nMPI_halo_size = %d")
			% step %MPI_rank %MPI_rank_neigbor_up %MPI_halo_size;
		// MPI_partition sends data from the greatest DEVICE_SUBDOMAIN:
		//cudaSetDevice(d_mesh->getDeviceAt(last_device_SubDomain_idx));
		// Get data from penultimate slice of last DEVICE_SUBDOMAIN:
		copyDeviceToHost(MPI_halo_size, h_buffer_blockingSR,
			(float *)d_mesh->getPressurePtrAt(last_device_SubDomain_idx) +
			size_lastSubDomain - 2 * MPI_halo_size, last_device_SubDomain_idx);
		// Do blocking send
		MPI_Ssend(h_buffer_blockingSR,
			MPI_halo_size, MPI_FLOAT, MPI_rank_neigbor_up, MPI_HALOTAG,
			MPI_COMM_WORLD);
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - (1) Send DOWN->UP"
			" command done!");
	}
	else {
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - [STEP %04d (1)] RANK_%d"
			" doesn't send data UP since it has no TOP neighbor.")
			% step %MPI_rank;
	}
	/// Step 2: DOWN listens to UP for >> UP->DOWN << message if
	/// there is a neighbor up:
	if (MPI_rank_neigbor_up != -2) {
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - [STEP %04d (2)] RANK_%d"
			" waits data from TOP neighbor RANK_%d\nMPI_halo_size = %d")
			% step %MPI_rank %MPI_rank_neigbor_up %MPI_halo_size;
		// MPI_partition receives data to the greatest DEV_partition:
		//cudaSetDevice(d_mesh->getDeviceAt(last_device_SubDomain_idx));
		// Recieve:
		MPI_Recv(h_buffer_blockingSR,
			MPI_halo_size, MPI_FLOAT, MPI_rank_neigbor_up, MPI_HALOTAG,
			MPI_COMM_WORLD, &MPI_rec_status_from_UP);
		// Put data on last slice of last DEVICE_SUBDOMAIN:
		copyHostToDevice(MPI_halo_size,
			(float *)d_mesh->getPressurePtrAt(last_device_SubDomain_idx) +
			size_lastSubDomain - 1 * MPI_halo_size,
			h_buffer_blockingSR, last_device_SubDomain_idx);
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - (2) Receive UP->DOWN"
			" command done!");
	}
	else {
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - [STEP %04d (2)] RANK_%d"
			" doesn't wait UP data since it has no TOP neighbor.")
			% step %MPI_rank;
	}
	/// Step 3: UP sends MPI_HALO data to DOWN for >> UP->DOWN << message if
	/// there is a neighbor down:
	if (MPI_rank_neigbor_down != -2) {
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - [STEP %04d (3)] RANK_%d"
			" sends data to UNDER neighbor RANK_%d\nMPI_halo_size = %d")
			% step %MPI_rank %MPI_rank_neigbor_up %MPI_halo_size;
		//cudaSetDevice(d_mesh->getDeviceAt(first_device_SubDomain_idx));
		// Send data from second slice of first DEVICE_SUBDOMAIN:
		copyDeviceToHost(MPI_halo_size, h_buffer_blockingSR,
			(float *)d_mesh->getPressurePtrAt(first_device_SubDomain_idx) +
			MPI_halo_size, first_device_SubDomain_idx);
		// Do blocking send
		MPI_Ssend(h_buffer_blockingSR,
			MPI_halo_size, MPI_FLOAT, MPI_rank_neigbor_down, MPI_HALOTAG,
			MPI_COMM_WORLD);
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - (3) Send DOWN->UP"
			" command done!");
	}
	else {
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_single - [STEP %04d (3)] RANK_%d"
			" doesn't send data DOWN since it has no UNDER neighbor.")
			% step %MPI_rank;
	}
	log_msg<LOG_TRACE>(L"\tMPI_switch_Halos_single - [STEP %04d (4)] Waiting"
		" barrier to complete data sending.") % step;
	// Wait for ALL MPI data transfers to end:
	MPI_Barrier(MPI_COMM_WORLD);
	end_t = clock() - start_t;
	// Use this if halo debugging is needed (works on any domain):
	//debugging_log_pressure_slices_hash_single(first_device_SubDomain_idx,
	//    last_device_SubDomain_idx, MPI_halo_size, d_mesh, step, MPI_rank);
	return ((float)end_t / CLOCKS_PER_SEC);
}

///////////////////////////////////////////////////////////////////////////////
float MPI_switch_Halos_double_host(CudaMesh* d_mesh, unsigned int step,
	double* h_buffer_blockingSR, // allocated pointer on host (size = dimXY) to send/receive data from
	int MPI_rank,
	int MPI_rank_neigbor_down,
	int MPI_rank_neigbor_up)
{
	clock_t start_t;
	clock_t end_t;
	start_t = clock();

	// MPI message tagging:
	const int MPI_HALOTAG = 1;
	/// Prerequisite variables:
	int MPI_halo_size = d_mesh->getDimXY();
	int first_device_SubDomain_idx = 0,
		last_device_SubDomain_idx = d_mesh->getNumberOfPartitions() - 1;
	int size_lastSubDomain =
		d_mesh->getNumberOfElementsAt(last_device_SubDomain_idx);
	// If halo debugging is needed (works on any domain), have a look at
	// debugging_log_pressure_slices_hash_single();
	// Receive status info"
	MPI_Status MPI_rec_status_from_DOWN, MPI_rec_status_from_UP;
	/// Step 0: UP listens to DOWN for >> DOWN->UP << message if
	//  there is a neighbor down:
	if (MPI_rank_neigbor_down != -2) {
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - [STEP %04d (0)] RANK_%d"
			" waits data from UNDER neighbor RANK_%d")
			% step %MPI_rank %MPI_rank_neigbor_down;
		// MPI_partition receives on the lowest DEVICE_SUBDOMAIN:
		//cudaSetDevice(d_mesh->getDeviceAt(first_device_SubDomain_idx));

		// MPI_partition receives on the lowest DEVICE_SUBDOMAIN:
		// cudaSetDevice(d_mesh->getDeviceAt(first_device_SubDomain_idx));
		// Recieve:
		MPI_Recv(h_buffer_blockingSR,
			MPI_halo_size, MPI_DOUBLE, MPI_rank_neigbor_down, MPI_HALOTAG,
			MPI_COMM_WORLD, &MPI_rec_status_from_DOWN);
		// Put data on slice 0 of DEVICE_SUBDOMAIN 0:
		copyHostToDevice(MPI_halo_size,
			(double *)d_mesh->getPressureDoublePtrAt(first_device_SubDomain_idx),
			h_buffer_blockingSR, first_device_SubDomain_idx);
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - (0) Receive DOWN->UP"
			" command done!");
	}
	else {
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - [STEP %04d (0)] RANK_%d"
			" doesn't wait DOWN data since it has no UNDER neighbor.")
			% step %MPI_rank;
	}
	/// Step 1: DOWN sends MPI_HALO data to UP for >> DOWN->UP << message if
	/// there is a neighbor up:
	if (MPI_rank_neigbor_up != -2) {
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - [STEP %04d  (1)] RANK_%d"
			" sends data to TOP neighbor RANK_%d")
			% step %MPI_rank %MPI_rank_neigbor_up;
		// MPI_partition sends data from the greatest DEVICE_SUBDOMAIN:
		//cudaSetDevice(d_mesh->getDeviceAt(last_device_SubDomain_idx));
		// Send data from penultimate slice of last DEVICE_SUBDOMAIN:
		copyDeviceToHost(MPI_halo_size, h_buffer_blockingSR,
			(double *)d_mesh->getPressureDoublePtrAt(last_device_SubDomain_idx) +
			size_lastSubDomain - 2 * MPI_halo_size, last_device_SubDomain_idx);
		// Do blocking send
		MPI_Ssend(h_buffer_blockingSR,
			MPI_halo_size, MPI_DOUBLE, MPI_rank_neigbor_up, MPI_HALOTAG,
			MPI_COMM_WORLD);
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - (1) Send DOWN->UP"
			" command done!");
	}
	else {
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - [STEP %04d (1)] RANK_%d"
			" doesn't send data UP since it has no TOP neighbor.")
			% step %MPI_rank;
	}
	/// Step 2: DOWN listens to UP for >> UP->DOWN << message if
	/// there is a neighbor up:
	if (MPI_rank_neigbor_up != -2) {
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - [STEP %04d (2)] RANK_%d"
			" waits data from TOP neighbor RANK_%d")
			% step %MPI_rank %MPI_rank_neigbor_up;
		// MPI_partition receives data to the greatest DEV_partition:
		//cudaSetDevice(d_mesh->getDeviceAt(last_device_SubDomain_idx));
		// Recieve:
		MPI_Recv(h_buffer_blockingSR,
			MPI_halo_size, MPI_DOUBLE, MPI_rank_neigbor_up, MPI_HALOTAG,
			MPI_COMM_WORLD, &MPI_rec_status_from_UP);
		// Put data on last slice of last DEVICE_SUBDOMAIN:
		copyHostToDevice(MPI_halo_size,
			(double *)d_mesh->getPressureDoublePtrAt(last_device_SubDomain_idx) +
			size_lastSubDomain - 1 * MPI_halo_size,
			h_buffer_blockingSR, last_device_SubDomain_idx);
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - (2) Receive UP->DOWN"
			" command done!");
	}
	else {
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - [STEP %04d (2)] RANK_%d"
			" doesn't wait UP data since it has no TOP neighbor.")
			% step %MPI_rank;
	}
	/// Step 3: UP sends MPI_HALO data to DOWN for >> UP->DOWN << message if
	/// there is a neighbor down:
	if (MPI_rank_neigbor_down != -2) {
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - [STEP %04d (3)] RANK_%d"
			" sends data to UNDER neighbor RANK_%d")
			% step %MPI_rank %MPI_rank_neigbor_down;
		//cudaSetDevice(d_mesh->getDeviceAt(first_device_SubDomain_idx));
		// Send data from second slice of first DEVICE_SUBDOMAIN:
		copyDeviceToHost(MPI_halo_size, h_buffer_blockingSR,
			(double *)d_mesh->getPressureDoublePtrAt(first_device_SubDomain_idx) +
			MPI_halo_size, first_device_SubDomain_idx);
		// Do blocking send
		MPI_Ssend(h_buffer_blockingSR,
			MPI_halo_size, MPI_DOUBLE, MPI_rank_neigbor_down, MPI_HALOTAG,
			MPI_COMM_WORLD);
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - (3) Send DOWN->UP"
			" command done!");
	}
	else {
		log_msg<LOG_TRACE>(L"MPI_switch_Halos_double - [STEP %04d (3)] RANK_%d"
			" doesn't send data DOWN since it has no UNDER neighbor.")
			% step %MPI_rank;
	}
	log_msg<LOG_TRACE>(L"\tMPI_switch_Halos_double - [STEP %04d (4)] Waiting"
		" barrier to complete data sending.") % step;
	// Wait for ALL MPI data transfers to end:
	MPI_Barrier(MPI_COMM_WORLD);
	end_t = clock() - start_t;
	// If halo debugging is needed (works on any domain), have a look at
	// debugging_log_pressure_slices_hash_single();
	return ((float)end_t / CLOCKS_PER_SEC);
}

///////////////////////////////////////////////////////////////////////////////
//////// Other debugging and slow functions... May be cleaned !  //////////////
//////////////////       If used, use with care !      ////////////////////////
///////////////////////////////////////////////////////////////////////////////
/*
 * Function prints the partition indexing for debugging purposes.
 *
 * //param partition_indexing_[IN]: the partition to be printed.
 * //param msg_[IN]: Some extra message in the logger print.
 */
/*
void debugging_print_partion_indexing(
    std::vector< unsigned int> partition_indexing_,
    const char* msg_ = "")
{
  int partition_size = partition_indexing_.size();
  // Build message:
  char partition_printout[partition_size*10 + 20], number_[10];
  partition_printout[0] = '\0'; number_[0] = '\0';
  for(int idx = 0; idx< partition_size; idx++ ){
    sprintf(number_, ",%d", partition_indexing_.at(idx));
    strcat(partition_printout, number_);
  }
  log_msg<LOG_DEBUG>(L"\t@ DEBUG_PRINTOUT_PARTION Indexing %s "
      "(size %d):\n[%s]") %msg_ %partition_size %partition_printout;
}
*/
/////////////////////////////////////////////////////
/*
// Create a function for debugging:

!!!!!!!  DON'T USE THIS -> very very slow!!!!!!!!!!

  Log level is LOG_DEBUG.

! A segmentation fault might be thrown if Z_slice_index is bigger
than the domain in d_postition_idx.

// param d_postition_idx [IN]: position matrix on device
// param dimXY [IN]: The dimension of a z slice ( = X*Y)
// param Z_slice_index [IN]: The slice that has to be printed
*/
/*
void debugging_print_voxelization_slice(
    unsigned char* d_postition_idx, unsigned int dimXY,
    unsigned int Z_slice_index, unsigned int dev_idx)
{
  // First, get slice on host:
  unsigned char *host_slice = new unsigned char[dimXY];
  copyDeviceToHost( dimXY, host_slice,
      d_postition_idx + Z_slice_index * dimXY, dev_idx);
  // Print it in logger:
  char slice_printout[dimXY*10 + 20], number_[10];
  slice_printout[0] = '\0'; number_[0] = '\0';
  for(int idx = 0; idx< dimXY; idx++ ){
    sprintf(number_, ",%d", host_slice[idx]);
    strcat(slice_printout, number_);
  }
  log_msg<LOG_DEBUG>(L"\t@ DEBUG_PRINTOUT_SLICE %d: [%s]")
      %Z_slice_index %host_slice;
  delete[] host_slice;
}
*/
/////////////////////////////////////////////////////
/*
 * Function prints a vector residing on host memory.
 *
!!!!!!!  DON'T USE THIS -> very very slow!!!!!!!!!!

  Log level is LOG_DEBUG.
 *
 * ! Note that function does not know of how big
 * the vector is in memory --> might get a seg_fault!
 *
 * //param data_[IN]: the data that needs printing
 * //param size_[IN]: How much of data is printed.
 * //param msg_[IN]: Some extra message in the logger print.
 */
/*
template <typename T>
void debugging_print_array_data(T* data_, unsigned int size_,
    const char* msg_ = "")
{
  // Build message:
  char data_printout[size_*10 + 20], number_[10];
  data_printout[0] = '\0'; number_[0] = '\0';
  for(int idx = 0; idx< size_; idx++ ){
    if ( (typeid(* data_) == typeid(float)) ||
        (typeid(* data_) == typeid(double)) )
      sprintf(number_, ",%f", data_[idx]);
    else
      sprintf(number_, ",%d", data_[idx]);
    strcat(data_printout, number_);
  }
  log_msg<LOG_DEBUG>(L"\t@ DEBUG_PRINTOUT_DATA %s (size %d):\n[%s]")
      %msg_ %size_ %data_printout;
}
*/
