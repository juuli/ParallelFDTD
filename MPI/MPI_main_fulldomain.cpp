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
// (C) Sebastian Prepelita, Nov. 2015, Jukka Saarelma, Dec. 2016
// Aalto University School of Science
//
///////////////////////////////////////////////////////////////////////////////

#include "MPI_Utils.h"

#include "hdf5.h"
#include "hdf5_hl.h"

// ParallelFDTD lib:
#include "../src/App.h"
#include "../src/kernels/kernels3d.h"
#include "../src/kernels/voxelizationUtils.h"
#include "../src/kernels/cudaMesh.h"

// The functions needed for launch_ParallelFDTD code:
extern "C" {
  bool interruptCallback_(){
    return false;
  }
}

extern "C" {
  void progressCallback_(int step, int max_step, float t_per_step ){
    //float estimate = t_per_step*(float)(max_step-step);
    //printf("Step %d/%d, time per step %f, estimated time left %f s \\n",
    //    step, max_step, t_per_step, estimate);
    return;
  }
}

void readHDF5(const char *file_name,
              int* fs, double *dX, double* c_sound, double* lambda_sim,
              int* CUDA_steps, int* N_x_CUDA, int* N_y_CUDA, int* N_z_CUDA,
              int* GPU_partitions, int* double_precision, int* num_rec,
              int* num_src, long* num_mat_coefs, long* mat_coefs_vect_len,
              int* source_type,
              // arrays:
              float*& material_coefficients, float*& receiver_vector,
              float*& source_vector, float*& src_data_vector,
              unsigned char*& h_pos,
              unsigned char*& h_mat,
              unsigned long long int* dim_x,
              unsigned long long int* dim_y,
              unsigned long long int* dim_z) {
  // Load up hdf5 file:
  hid_t       file_id;   // file identifier
  herr_t      status;
  file_id = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);

  // Simulation single properties:
  H5LTread_dataset_int(file_id, "/fs", fs);
  H5LTread_dataset_double(file_id, "/dX", dX);
  H5LTread_dataset_double(file_id, "/c_sound", c_sound);
  H5LTread_dataset_double(file_id, "/lambda_sim", lambda_sim);
  H5LTread_dataset_int(file_id, "/CUDA_steps", CUDA_steps);
  H5LTread_dataset_int(file_id, "/N_x_CUDA", N_x_CUDA);
  H5LTread_dataset_int(file_id, "/N_y_CUDA", N_y_CUDA);
  H5LTread_dataset_int(file_id, "/N_z_CUDA", N_z_CUDA);
  H5LTread_dataset_int(file_id, "/GPU_partitions", GPU_partitions);
  H5LTread_dataset_int(file_id, "/num_rec", num_rec);
  H5LTread_dataset_int(file_id, "/num_src", num_src);
  H5LTread_dataset_int(file_id, "/source_type", source_type);
  H5LTread_dataset_int(file_id, "/double_precision", double_precision);

  hid_t dataset_id = H5Dopen2(file_id, "/dim_x", H5P_DEFAULT);
  H5Dread(dataset_id, H5T_NATIVE_ULLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, dim_x);
  dataset_id = H5Dopen2(file_id, "/dim_y", H5P_DEFAULT);
  H5Dread(dataset_id, H5T_NATIVE_ULLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, dim_y);
  dataset_id = H5Dopen2(file_id, "/dim_z", H5P_DEFAULT);
  H5Dread(dataset_id, H5T_NATIVE_ULLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, dim_z);

  long num_src_coords = 3*(*num_src);
  long num_rec_coords = 3*(*num_rec);

  // Array sizes:
  H5LTread_dataset_long(file_id, "/num_mat_coefs", num_mat_coefs);
  H5LTread_dataset_long(file_id, "/mat_coef_vect_len", mat_coefs_vect_len);
  // Allocationg and reading arrays:
  material_coefficients = new float[(*mat_coefs_vect_len)];
  receiver_vector = new float[num_rec_coords];
  source_vector = new float[num_src_coords];
  src_data_vector = new float[*CUDA_steps];
  unsigned long long int num_elements = (*dim_x)*(*dim_y)*(*dim_z);
  h_pos = new unsigned char[num_elements];
  h_mat = new unsigned char[num_elements];

  dataset_id = H5Dopen2(file_id, "/h_pos", H5P_DEFAULT);
  H5Dread(dataset_id, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, h_pos);
  dataset_id = H5Dopen2(file_id, "/h_mat", H5P_DEFAULT);
  H5Dread(dataset_id, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, h_mat);

  H5LTread_dataset_float(file_id, "/material_coefficients", material_coefficients);
  H5LTread_dataset_float(file_id, "/rec_coords", receiver_vector);
  H5LTread_dataset_float(file_id, "/src_coords", source_vector);
  H5LTread_dataset_float(file_id, "/src_data", src_data_vector);

  status = H5Fclose(file_id);
}

void writeResponseHDF5(const char* name, int fs, double dX,
                       double c_sound, double lambda_sim,
                       int CUDA_steps,
                       float* src_data_vector,
                       int num_rec, int rank,
                       const std::vector<int>& orig_rec_indices,
                       double* return_data,
                       bool use_double_FDTD,
                       int src_type) {
  // Writing data:
  hid_t       file_id;
  herr_t      status;
  char file_name[1024];
  char a_str[10];
  char source_type[48];
  int use_double_FDTD_int = (int)use_double_FDTD;

  switch(src_type){
  case 1:
    strcpy(source_type,"SS");
    break;
  case 0:
    strcpy(source_type,"HS");
    break;
  case 2:
    strcpy(source_type,"TS");
    break;
  }

  sprintf(a_str, "%02d", rank);
  strcpy(file_name, "R");
  strcat(file_name, a_str);
  strcat(file_name, "_");
  strcat(file_name, name);
  strcat(file_name, a_str);
  strcat(file_name, (use_double_FDTD ? "_fl64_" : "_fl32_"));
  strcat(file_name, source_type);
  strcat(file_name, ".hdf5");
  // Create a new file using default properties.
  // H5F_ACC_TRUNC means the file will be overwritten
  file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hsize_t dims[1]={1};
  H5LTmake_dataset(file_id, "/fs", 1, dims, H5T_NATIVE_INT, &fs);
  H5LTmake_dataset(file_id, "/dX", 1, dims, H5T_NATIVE_DOUBLE, &dX);
  H5LTmake_dataset(file_id,"/c_sound", 1, dims, H5T_NATIVE_DOUBLE, &c_sound);
  H5LTmake_dataset(file_id, "/lambda_sim", 1, dims, H5T_NATIVE_DOUBLE,
                   &lambda_sim);
  H5LTmake_dataset(file_id, "/CUDA_steps", 1, dims, H5T_NATIVE_INT, &CUDA_steps);
  H5LTmake_dataset(file_id, "/num_rec", 1, dims, H5T_NATIVE_INT, &num_rec);

  H5LTmake_dataset(file_id, "/rank", 1, dims, H5T_NATIVE_INT, &rank);

  H5LTmake_dataset(file_id, "/use_double_FDTD", 1, dims, H5T_NATIVE_INT,
                   &use_double_FDTD_int);
  H5LTmake_dataset_string(file_id, "/source_type", source_type);

  dims[0] = (hsize_t)orig_rec_indices.size();
  H5LTmake_dataset(file_id, "/original_rec_indices", 1, dims, H5T_NATIVE_INT,
                   &orig_rec_indices[0]); // original receiver indices

  dims[0] = (hsize_t)CUDA_steps;
  H5LTmake_dataset(file_id, "/src_data_vector", 1, dims, H5T_NATIVE_FLOAT,
                   src_data_vector);

  // More MPI data here [...]
  // The packed data 1D vector:
  dims[0]=CUDA_steps*num_rec;
  H5LTmake_dataset(file_id, "/return_data", 1, dims,
                   H5T_NATIVE_DOUBLE, return_data);
  status = H5Fclose(file_id);
}

bool partitionReceivers(SimulationParameters* sp, int MPI_rank,
                        std::vector< std::vector< mesh_size_t> > MPI_partition_indexing,
                        std::vector<int>& MPI_receiver_orig_index,
                        float* rec_vector, int n_steps, int n_rec,
                        int& n_rec_MPI,
                        unsigned int MPI_partition_Z_start_idx,
                        unsigned int MPI_partition_Z_end_idx) {
  int temp_z_idx;
  for(int i=0; i<n_rec; i++ ) {
    float r_x = rec_vector[i*3];
    float r_y = rec_vector[i*3+1];
    float r_z = rec_vector[i*3+2];
    float dx = sp->getDx();
    temp_z_idx = (int)r_z;
    // Test that receiver is inside this MPI domain:

    if(MPI_check_receiver(&temp_z_idx, MPI_partition_indexing,
      MPI_partition_Z_start_idx, MPI_partition_Z_end_idx, MPI_rank)) {
      sp->addReceiver(r_x*dx, r_y*dx, ((float)temp_z_idx)*dx);
      log_msg<LOG_INFO> (
        L"FDTD_MPI RANK: %d - Added receiver at [%f,%f,%f]. "
        "Original receiver at: [%f,%f,%f]")
        % MPI_rank % r_x % r_y %((float)temp_z_idx)
        % r_x % r_y % r_z;

      if (sp->getReceiverElementCoordinates(n_rec_MPI).z !=
          r_z-MPI_partition_indexing.at(MPI_rank).at(0) ) {
        log_msg<LOG_ERROR>(L"FDTD_MPI - For global receiver %d (local MPI receiver %d), the z coordinate ended up wrong: intended %f, reached %d.") %i %n_rec_MPI % (rec_vector[i*3+2] - MPI_partition_indexing.at(MPI_rank).at(0) ) %sp->getReceiverElementCoordinates(i).z;
        return false;
      }
      MPI_receiver_orig_index.push_back(i);
      n_rec_MPI += 1;
    }
  }

  // Yet another sanity check:
  if (MPI_receiver_orig_index.size() != n_rec_MPI) {
    log_msg<LOG_ERROR>(L"FDTD_MPI - The number of indices in the "
    "MPI_receiver_orig_index vector (%d) is different than the counted"
      " number of receivers (%d) on this MPI node (%d).")
      %MPI_receiver_orig_index.size() %n_rec_MPI %MPI_rank;
    return false;
  }
  return true;
}

bool paritionSources(SimulationParameters* sp, int MPI_rank,
                     std::vector< std::vector< mesh_size_t> > MPI_partition_indexing,
                     float* src_vector, float* src_data_vector,
                     int n_steps, int n_src, int src_type,
                     int& n_src_MPI, bool use_double_FDTD,
                     unsigned int MPI_partition_Z_start_idx,
                     unsigned int MPI_partition_Z_end_idx) {

  int temp_z_idx;
  for(int i=0; i<n_src; i++ ) {
    float s_x = src_vector[3*i];
    float s_y = src_vector[3*i+1];
    float s_z = src_vector[3*i+2];
    float dx  = sp->getDx();
    temp_z_idx = (int)s_z;
    // Test that receiver is inside this MPI domain:
    if ( MPI_check_source(&temp_z_idx, MPI_partition_indexing, MPI_rank) ) {
      // Log halo source:
      if (temp_z_idx == 0 || temp_z_idx == MPI_partition_indexing.at(MPI_rank).back()){
        // Warning to easily find it.
        log_msg<LOG_WARNING>(L"FDTD_MPI - Source on halo! Added source"
                              " at [%f,%f,%f]. Original source receiver at: [%f,%f,%f].")
                              % src_vector[3*i] % src_vector[3*i+1]
                              % ((float)temp_z_idx) %src_vector[3*i]
                              % src_vector[3*i+1] % src_vector[3*i+2];
      }

      Source s( s_x*dx, s_y*dx, ((float)temp_z_idx)*dx, // x [m]
                (enum SrcType)src_type,
                (enum InputType)DATA,
                0);

      s.setInputDataIdx(n_src_MPI);
      sp->addSource(s);
      log_msg<LOG_INFO> (
        L"FDTD_MPI - Added source at [%f,%f,%f]. "
        "Original source at: [%f,%f,%f]")
        % src_vector[3*i] % src_vector[3*i+1] %((float)temp_z_idx)
        % src_vector[3*i] % src_vector[3*i+1] % src_vector[3*i+2];

      if(sp->getSourceElementCoordinates(n_src_MPI).z !=
        s_z - MPI_partition_indexing.at(MPI_rank).at(0) ) {
        log_msg<LOG_ERROR>(L"FDTD_MPI - For source %d (local MPI source %d), the "
                           L"z coordinate ended up wrong: intended %f, reached %d.")
                           %i %n_src_MPI % temp_z_idx
                           %sp->getSourceElementCoordinates(i).z;
        return false;
      }

      if (use_double_FDTD) {
        std::vector<double> source_d_vector(n_steps);
        for( int j = 0; j<n_steps; j++) {
          source_d_vector.at(j) = (double)(src_data_vector[j]);
        }
        sp->addInputDataDouble(source_d_vector);
      } else {
        std::vector<float> source_d_vector(n_steps);
        for( int j = 0; j<n_steps; j++) {
          source_d_vector.at(j) = (float)(src_data_vector[j]);
        }
        sp->addInputData(source_d_vector);
      }
      n_src_MPI += 1;
    }
  }

  log_msg<LOG_DEBUG>(L"FDTD_MPI - Added %d #SOURCES out of a total of %d"
      ".... Calling set-up mesh...") %n_src_MPI %n_src;
  return true;
}

int main(int argc, char **argv) {
  if(argc == 1) {std::cout<<"HDF5 dataset file needs "
                "to be given as an inp5 dataset file needs "
                "to be given as an input argument"<<std::endl;
    return 0;
  }
  ///////// Init app class (this is first for the logger):
  FDTD::App app;
  app.initializeDevices();

  ///////////////////
  // MPI variables:
  ///////////////////////
  int MPI_size, MPI_rank, MPI_name_length;
  int MPI_error_var;
  char MPI_name[BUFSIZ]; // BUFSIZ should be from <stdio.h>

  // Init the MPI:
  MPI_error_var = MPI_Init(&argc, &argv);
  // Get size and rank:
  MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
  MPI_Get_processor_name(MPI_name, &MPI_name_length);

  log_msg<LOG_INFO>(L"FDTD_MPI - Starting program. This is node %s: "
                     "MPI_SIZE = %d MPI_RANK = %d ") %MPI_name %MPI_size %MPI_rank;

  char* filename = new char[1024];
  strcpy(filename, argv[1]);

  int fs = 0, CUDA_steps = 0, N_x_CUDA = 0, N_y_CUDA = 0, N_z_CUDA = 0;
  int GPU_partitions = 0, n_rec = 0, n_src = 0;
  int double_precision = 0;
  long num_mat_coefs = 0;
  long mat_coef_vect_len = 0;
  double dX = 0.0, c_sound = 0.0, lambda_sim = 0.0;
  float* mat_coefs = 0;
  float* rec_coords = 0;
  float* src_coords = 0;
  float* src_data_vector = 0;
  int src_type;
  unsigned char* h_pos = NULL;
  unsigned char* h_mat = NULL;
  unsigned long long int dim_x = 0;
  unsigned long long int dim_y = 0;
  unsigned long long int dim_z = 0;
  log_msg<LOG_DEBUG>(L"FDTD_MPI - Loading pythond data...");

  // Load all data from a HDF5 file
  readHDF5(filename,
           &fs, &dX, &c_sound, &lambda_sim,
           &CUDA_steps, &N_x_CUDA, &N_y_CUDA,
           &N_z_CUDA, &GPU_partitions, &double_precision, &n_rec,
           &n_src, &num_mat_coefs, &mat_coef_vect_len,
           &src_type,
           mat_coefs, rec_coords, src_coords, src_data_vector,
           h_pos, h_mat, &dim_x, &dim_y, &dim_z);

  log_msg<LOG_INFO>(L"Read done");

  bool use_double_FDTD = (bool)double_precision;

  log_msg<LOG_CONFIG>(L"FDTD_MPI - Starting code. Config:\nprecision = %s ; python data file = %s") %(use_double_FDTD ? "'DOUBLE'" : "'SINGLE'") %filename;

  MaterialHandler mh;
  mh.addMaterials(mat_coefs, 1, 20);
  int no_of_unique_materials = mh.getNumberOfUniqueMaterials();
  unsigned char* material_indices_ptr = mh.getMaterialIdxPtr();
  SimulationParameters sp;
  sp.setSpatialFs(fs);
  sp.setLambda(lambda_sim);
  sp.setC(c_sound);
  sp.setNumSteps(CUDA_steps);

  log_msg<LOG_CONFIG>
  (L" dX = %f\n fs = %d\n"
    " c_sound = %f\n lambda_sim = %f\n "
    "CUDA_steps = %d\n N_x_CUDA = %d\n N_y_CUDA = %d\n N_z_CUDA = %d\n "
    "GPU_partitions = %d\n n_rec = %d\n n_src = %d\n source_type = %d\n "
    "material coef vector size: %d\n num mat coefs: %d\n num materials:%d \n")
    %dX %fs %c_sound %lambda_sim
    %CUDA_steps %N_x_CUDA %N_y_CUDA %N_z_CUDA %GPU_partitions %n_rec
    %n_src  %src_type %mat_coef_vect_len
    %num_mat_coefs %no_of_unique_materials;


  int device_count = 0;
  cudaGetDeviceCount(&device_count);

  if (GPU_partitions < 1) {
    log_msg<LOG_ERROR>(L"FDTD_MPI - No partitions assigned in variable "
                        L"GPU_partitions, exitting");
    throw 5;
  }
  if (GPU_partitions > 2) {
  log_msg<LOG_WARNING>(L"FDTD_MPI - Script tested OK only for 1 or 2 "
      "partitions. Code might be buggy for more than 2 partitions!");
  }

  ///////////////////////////////////////////////////////////////////////////
  //      Split domain for this MPI node:
  ///////////////////////////////////////////////////////////////////////////

  uint3 voxelization_dim_ = make_uint3(dim_x, dim_y, dim_z);

  log_msg<LOG_DEBUG>(L"FDTD_MPI - Full voxelization dimensions = [%d,%d,%d]")
    %voxelization_dim_.x %voxelization_dim_.y
    %voxelization_dim_.z;

  CudaMesh* the_mesh = new CudaMesh();

  //Recycle existing code of domain splitting (done in the Z direction):
  std::vector< std::vector< mesh_size_t> > MPI_partition_indexing_;
  MPI_partition_indexing_ =  the_mesh->getPartitionIndexing(MPI_size, (int)voxelization_dim_.z);

  // // GET MPI domain data:
  // MPI_extract_idx_data(&voxelization_dim_, MPI_rank, d_postition_idx_,
  //                      d_materials_idx_, MPI_partition_indexing_);

  // The MPI Z-slice indexes without guard/ghost slices:
  mesh_size_t MPI_partition_Z_start_idx = -1, MPI_partition_Z_end_idx = -1;
  int MPI_rank_neigbor_down = -2, MPI_rank_neigbor_up = -2;
  MPI_domain_get_useful_data( MPI_partition_indexing_, MPI_rank, MPI_size,
          &MPI_partition_Z_start_idx, &MPI_partition_Z_end_idx,
          &MPI_rank_neigbor_down, &MPI_rank_neigbor_up );

  unsigned long long int mem_size = (unsigned long long int)voxelization_dim_.x *
                                    (unsigned long long int)voxelization_dim_.y *
                                    (unsigned long long int)voxelization_dim_.z;

  unsigned int dimXY = voxelization_dim_.x * voxelization_dim_.y;
  unsigned int MPI_halo_size_MB = dimXY * (use_double_FDTD?sizeof(double):sizeof(float))*1e-6;
  log_msg<LOG_DEBUG>(L"Splitting domain for domain (%s: SIZE "
                    "= %d RANK = %d )") %MPI_name %MPI_size %MPI_rank;


  log_msg<LOG_INFO>(
    L"FDTD_MPI - Domain extracted for rank %d (rank_DOWN "
    "= %d , rank_UP = %d). New MPI dimension are: [%d, %d, %d]. MPI "
    "message size is %d [MB].\nMPI_partition_Z_start_idx = %d ; "
    "MPI_partition_Z_end_idx = %d") %MPI_rank %MPI_rank_neigbor_down
    %MPI_rank_neigbor_up %voxelization_dim_.x %voxelization_dim_.y
    %voxelization_dim_.z %MPI_halo_size_MB %MPI_partition_Z_start_idx
    %MPI_partition_Z_end_idx;

  log_msg<LOG_DEBUG>(L"FDTD_MPI - Setting up %d receivers between z-slices: "
  "%d - %d.") %n_rec %MPI_partition_Z_start_idx
      %MPI_partition_Z_end_idx;
  sp.setAddPaddingToElementIdx(false); // No border
  int n_rec_MPI = 0;
  int n_src_MPI = 0;

  // This is used to know the original receiver index to retrieve the azimuth,
  // elevation and location from original data when packing...

  std::vector<int> MPI_receiver_orig_index;

  //////////////////////////////////////////////////////////////////////////////
  // Set-up RECEIVERS
  //////////////////////////////////////////////////////////////////////////////

  partitionReceivers(&sp, MPI_rank,
                     MPI_partition_indexing_,
                     MPI_receiver_orig_index,
                     rec_coords, CUDA_steps, n_rec,
                     n_rec_MPI,
                     MPI_partition_Z_start_idx,
                     MPI_partition_Z_end_idx);

  log_msg<LOG_DEBUG>(L"FDTD_MPI - Added %d #RECEIVERS out of a total of %d. "
                      "Setting up sources. Setting up sources between z-slices: %d - %d")
                      %n_rec_MPI %n_rec
                      %MPI_partition_indexing_.at(MPI_rank).at(0)
                      %MPI_partition_indexing_.at(MPI_rank).back();

  paritionSources(&sp, MPI_rank,
                  MPI_partition_indexing_,
                  src_coords, src_data_vector,
                  CUDA_steps, n_src, src_type,
                  n_src_MPI, use_double_FDTD,
                  MPI_partition_Z_start_idx,
                  MPI_partition_Z_end_idx);

  uint3 block_size__ = {N_x_CUDA, N_y_CUDA, N_z_CUDA}; // cuda structure
  unsigned int element_type__ = 0;// 0: forward-dif ,1: centered-dif

  the_mesh->setDouble(use_double_FDTD);

  if (use_double_FDTD) {
    the_mesh->setupMesh<double>( no_of_unique_materials,
                                 mh.getMaterialCoefficientPtrDouble(),
                                 sp.getParameterPtrDouble(),
                                 voxelization_dim_,
                                 block_size__,
                                 element_type__);
  } else {
    the_mesh->setupMesh<float>(no_of_unique_materials,
                               mh.getMaterialCoefficientPtr(),
                               sp.getParameterPtr(),
                               voxelization_dim_,
                               block_size__,
                               element_type__);
  }

  log_msg<LOG_TRACE>(L"FDTD_MPI - Mesh setup completed. "
      "Setting up node (internal) partitions...");

  ////// Set-up SUBDOMAINS

  unsigned int dev_idx_0 = 0, dev_idx_1 = 1;
  if ( device_count == 1 && GPU_partitions == 2) {
    dev_idx_1 = 0;
    std::vector<unsigned int> v_;
    v_.push_back( dev_idx_0 );
    v_.push_back( dev_idx_1 );
    the_mesh->makePartitionFromHost(GPU_partitions, h_pos,
                                    h_mat, v_);
  }
  else {
      the_mesh->makePartitionFromHost(GPU_partitions, h_pos,
                                      h_mat);
  }

  log_msg<LOG_INFO>(L"FDTD_MPI - Parallel FDTD library setup done!");
  // MPI SANITY CHECKS:
  MPI_checks(the_mesh, MPI_rank, MPI_partition_indexing_);
  log_msg<LOG_INFO>(L"FDTD_MPI - Checks done ok. Rank: %d") % MPI_rank;

  ///////////////////////////////////////////////////////////////////////////
  //      Compute FDTD in MPI:
  ///////////////////////////////////////////////////////////////////////////

  float time;
  if (use_double_FDTD) {
    log_msg<LOG_INFO>(L"IN, double, rank %d") % MPI_rank;
    log_msg<LOG_DEBUG> (
      L"FDTD_MPI RANK: %d - Start computing double precision for %d "
      L"device partitions... Number of receivers = %d x %d[steps]")
      %MPI_rank
      %the_mesh->getNumberOfPartitions()
      %sp.getNumReceivers()
      %sp.getNumSteps();

    double * h_return_values = new double[sp.getNumSteps()*sp.getNumReceivers()];
    unsigned int step;

    log_msg<LOG_DEBUG>(L"FDTD_MPI - Prepare receiver data on device...");

    std::vector<std::pair<double*, std::pair<mesh_size_t, int> > > d_receiver_data=
    prepare_receiver_data_on_device<double>(the_mesh, &sp);

    for(step = 0; step < sp.getNumSteps(); step++) {
      time = launchFDTD3dStep_double(the_mesh, &sp, step, &d_receiver_data,
                                     interruptCallback_, progressCallback_);

      time += MPI_switch_Halos_double(the_mesh, step, MPI_rank,
                                      MPI_rank_neigbor_down, MPI_rank_neigbor_up);

      log_msg<LOG_TRACE>(L"FDTD_MPI - [STEP %04d] Step done. Time per "
                "step = %f") %step %time;
    }

    log_msg<LOG_DEBUG>(L"FDTD_MPI - Copy receiver data from device to host...");
    get_receiver_data_from_device<double>(the_mesh, &sp,
                                          h_return_values, &d_receiver_data);

    log_msg<LOG_DEBUG>(L"FDTD_MPI - FDTD launch complete. Packing  returned data...");

    std::vector<double> ret_double(sp.getNumReceivers()*CUDA_steps, 0.0);
    for(int i = 0; i < ret_double.size(); i++)
      ret_double.at(i) = (double)h_return_values[i];

    writeResponseHDF5("TEST", fs, dX,
                      c_sound, lambda_sim,
                      CUDA_steps,
                      src_data_vector,
                      sp.getNumReceivers(), MPI_rank,
                      MPI_receiver_orig_index,
                      &ret_double[0],
                      use_double_FDTD,
                      src_type);

    log_msg<LOG_DEBUG>(L"FDTD_MPI - Data packing done. Writing packed data.");

    delete[] h_return_values;

  } else {
    log_msg<LOG_INFO>(L"IN, rank %d")% MPI_rank;
    log_msg<LOG_DEBUG>(L"FDTD_MPI - Start computing single precision for"
    " %d device partitions... Number of receivers = %d x %d[steps]")
      %the_mesh->getNumberOfPartitions()
      %sp.getNumReceivers()
      %sp.getNumSteps();
    // Allocate the receiver pointer (on host)
    float * h_return_values = new float[sp.getNumSteps()*sp.getNumReceivers()];
    unsigned int step;
    log_msg<LOG_DEBUG>(L"FDTD_MPI - Prepare receiver data on device...");
    std::vector<std::pair <float*, std::pair<mesh_size_t, int> > > d_receiver_data=
    prepare_receiver_data_on_device<float>(the_mesh, &sp);

    for(step = 0; step < sp.getNumSteps(); step++) {
      time = launchFDTD3dStep_single(the_mesh, &sp, step, &d_receiver_data,
                                     interruptCallback_, progressCallback_);

      time += MPI_switch_Halos_single(the_mesh, step, MPI_rank,
                                      MPI_rank_neigbor_down, MPI_rank_neigbor_up);

      log_msg<LOG_TRACE>(L"FDTD_MPI - [STEP %04d] Step done. Time per "
                          "step = %f") %step %time;
    }

    log_msg<LOG_DEBUG>(L"FDTD_MPI - Copy receiver data from device to "
                        "host...");

    get_receiver_data_from_device<float>(the_mesh, &sp,
                                         h_return_values,
                                         &d_receiver_data);

    std::vector<double> ret_double(sp.getNumReceivers()*CUDA_steps, 0.0);
    for(int i = 0; i < ret_double.size(); i++)
      ret_double.at(i) = (double)h_return_values[i];

    writeResponseHDF5("TEST", fs, dX,
                      c_sound, lambda_sim,
                      CUDA_steps,
                      src_data_vector,
                      sp.getNumReceivers(), MPI_rank,
                      MPI_receiver_orig_index,
                      &ret_double[0],
                      use_double_FDTD,
                      src_type);

    delete[] h_return_values;
  }

  log_msg<LOG_DEBUG>(L"FDTD_MPI - Done!");

  // Clear MPI stuff:
  MPI_Finalize();

  //////////////////////////////////// Cleanup:
  // Data from hdf5:
  delete[] filename;
  delete[] mat_coefs;
  delete[] rec_coords;
  delete[] src_coords;
  delete[] src_data_vector;
  ///////////////////////////////////
  delete the_mesh;

  // log_msg<LOG_INFO>(L" ---=== FDTD CUDA && MPI DONE OK ===---");
  app.close();
  return(0);
}
