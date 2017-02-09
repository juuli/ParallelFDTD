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
#include "App.h"
#include "./kernels/cudaUtils.h"
#include "./kernels/kernels3d.h"
#ifdef COMPILE_VISUALIZATION
#include "./kernels/visualizationUtils.h"
#endif
#include "./kernels/voxelizationUtils.h"
#include "./kernels/cudaMesh.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#ifdef COMPILE_VISUALIZATION
#include <cuda_gl_interop.h>
#endif

namespace GLOBAL {
  static FDTD::App* currentApp;

  // Global pointers for rendering
  // Slight hack to keep AppWindow class clean of CUDA
  struct cudaGraphicsResource* vertex = NULL;
  struct cudaGraphicsResource* color = NULL;
  struct cudaGraphicsResource* pbo = NULL;

  struct cudaGraphicsResource* pbo_xy = NULL;
  struct cudaGraphicsResource* pbo_xz = NULL;
  struct cudaGraphicsResource* pbo_yz = NULL;
  static volatile bool interrupt = false;

}

using namespace FDTD;

extern "C" {
  void captureBitmap(float* data, unsigned char* position_data, unsigned int dim_x, unsigned int dim_y,
                     unsigned int slice, unsigned int orientation, unsigned int step) {
    GLOBAL::currentApp->saveBitmap(data, position_data, dim_x, dim_y, slice, orientation, step);
  }
}

extern "C" {
  bool interruptCallback(){
    if(GLOBAL::interrupt)
      log_msg<LOG_INFO>(L"main: Execution interrupted");

    return GLOBAL::interrupt;
  }
}

extern "C" {
  void progressCallback(int step, int max_step, float t_per_step ){
    float estimate = t_per_step*(float)(max_step-step);
    printf("Step %d/%d, time per step %f, estimated time left %f s \n",
           step, max_step, t_per_step, estimate);

    return;
  }
}

void App::setupDefaultCallbacks() {
  this->m_interrupt = (InterruptCallback)interruptCallback;
  this->m_progress = (ProgressCallback)progressCallback;
}

///////////////////////////////////////////////////////////////////////////////
// Device reset
///////////////////////////////////////////////////////////////////////////////
void App::queryDevices() {
  int number_of_devices = 0;
  cudasafe(cudaGetDeviceCount(&number_of_devices), "getDeviceCount");
  this->number_of_devices_ = number_of_devices;
  unsigned int best_device = gpuGetMaxGflopsDeviceId();

  log_msg<LOG_INFO>(L"App::queryDevices - number of devices %s") %number_of_devices;
  log_msg<LOG_INFO>(L"App::queryDevices - best device %s") %best_device_;

  this->best_device_ = best_device;

  for(int i = 0; i < this->number_of_devices_; i++) {
	cudasafe(cudaSetDevice(i), "cudaSetDevice");
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudasafe(cudaMemGetInfo (&free_mem, &total_mem), "Cuda meminfo");
    this->device_mem_sizes_.push_back((int)(free_mem/1e6f));
    log_msg<LOG_INFO>(L"App::queryDevices - memory size dev %d: %d MB")
                      %i %(int)(free_mem/1e6f);
  }

}

void App::resetDevices() {
  for(int i = 0; i < this->number_of_devices_; i++) {
    log_msg<LOG_INFO>(L"App::resetDevices - reseting device %d") %i;
    cudasafe(cudaSetDevice(i), "cudaSetDevice");
    cudasafe(cudaDeviceReset(), "cudaDeviceReset");
  }
}

///////////////////////////////////////////////////////////////////////////////
// Initialization functions
///////////////////////////////////////////////////////////////////////////////
void App::initializeDevices() {
  this->queryDevices();
  this->resetDevices();
  cudaSetDevice(this->best_device_);
  cudasafe(cudaPeekAtLastError(), "App::initialize - peek error after initalization");
  GLOBAL::currentApp = this;
}

void App::initializeGeometryFromFile(std::string geometry_fp) {
  log_msg<LOG_DEBUG>(L"App::initializeGeometryFromFile - filename: %d") %geometry_fp.c_str();
  if(!m_file_reader.readVTK(&m_geometry, geometry_fp)) {
    log_msg<LOG_ERROR>(L"App::initializeGeometryFromFile - invalid file: %d") % geometry_fp.c_str();
    throw(-1);
   }
  this->has_geometry_ = true;
}

void App::initializeGeometry(unsigned int* indices, float* vertices,
                             unsigned int number_of_indices,
                             unsigned int number_of_vertices) {
  m_geometry.initialize(indices, vertices, number_of_indices, number_of_vertices);
  this->has_geometry_ = true;
}

void App::initializeMesh(unsigned int number_of_partitions) {
  float dx = this->m_parameters.getDx();
  mesh_size_t est_nodes;
  est_nodes= (mesh_size_t)((this->m_geometry.getBoundingBox()).x/dx*
                                      (this->m_geometry.getBoundingBox()).y/dx*
                                      (this->m_geometry.getBoundingBox()).z/dx);

  unsigned int mesh_size_in_MB = est_nodes*10.00/1000000.0;
  unsigned int mesh_size_in_MB_double = est_nodes*18.0/1000000.0;

  log_msg<LOG_INFO>(L"App::initializeMesh - Estimated size: %u nodes,"
                    L"dx: %f, size float: %u MB, double: %u MB")
                    %est_nodes %dx %mesh_size_in_MB % mesh_size_in_MB_double;

  int total_mem = 0;
  for(int i = 0; i < this->number_of_devices_; i++) {
    total_mem += device_mem_sizes_.at(i);
  }

  if(total_mem < mesh_size_in_MB) {
    log_msg<LOG_INFO>(L"App::initializeMesh - Estimated size: %u nodes "
                      L"too large, exiting") %est_nodes;
    this->close();
    throw(-1);
  }
  if(this->m_mesh.isDouble() && total_mem < mesh_size_in_MB_double){
    log_msg<LOG_INFO>(L"App::initializeMesh - Estimated size: %u nodes "
                      L"too large, exiting") %est_nodes;
    this->close();
    throw(-1);
  }

  unsigned char* d_position_idx = (unsigned char*)NULL;
  unsigned char* d_material_idx = (unsigned char*)NULL;
  uint3 voxelization_dim = make_uint3(0,0,0);

  // Currently the voxeliztaion functions give right position indices. In the
  // case of GPU voxelization, a conversion is needed. Then the
  // control variable convert_position_ptr need to be set
  bool convert_position_ptr = false;
  uint3 vox_dim = get_Geometry_surface_Voxelization_dims(
                                 m_geometry.getVerticePtr(),
                                 m_geometry.getIndexPtr(),
                                 m_geometry.getNumberOfTriangles(),
                                 m_geometry.getNumberOfVertices(),
                                 (double) m_parameters.getDx());
  // Voxelize the geometry
  if(this->m_parameters.getVoxelizationType() == SOLID) {
    convert_position_ptr = true;
    voxelizeGeometry(m_geometry.getVerticePtr(),
                     m_geometry.getIndexPtr(),
                     m_materials.getMaterialIdxPtr(),
                     m_geometry.getNumberOfTriangles(),
                     m_geometry.getNumberOfVertices(),
                     m_materials.getNumberOfUniqueMaterials(),
                     (double) m_parameters.getDx(),
                     &d_position_idx,
                     &d_material_idx,
                     &voxelization_dim);
  }
  if(this->m_parameters.getVoxelizationType() == SURFACE) {
    voxelizeGeometry_surface_conservative(m_geometry.getVerticePtr(),
                                          m_geometry.getIndexPtr(),
                                          m_materials.getMaterialIdxPtr(),
                                          m_geometry.getNumberOfTriangles(),
                                          m_geometry.getNumberOfVertices(),
                                          m_materials.getNumberOfUniqueMaterials(),
                                          (double) m_parameters.getDx(),
                                          &d_position_idx,
                                          &d_material_idx,
                                          &voxelization_dim,
                                          0,vox_dim.z);
  }
  if(this->m_parameters.getVoxelizationType() == SURFACE_6) {
    voxelizeGeometry_surface_6_separating(m_geometry.getVerticePtr(),
                                         m_geometry.getIndexPtr(),
                                         m_materials.getMaterialIdxPtr(),
                                         m_geometry.getNumberOfTriangles(),
                                         m_geometry.getNumberOfVertices(),
                                         m_materials.getNumberOfUniqueMaterials(),
                                         (double) m_parameters.getDx(),
                                         &d_position_idx,
                                         &d_material_idx,
                                         &voxelization_dim,
                                         0,vox_dim.z);
  }
  // Initialize the mesh
  uint3 block_size = make_uint3(32,4,1); // this is a default block size used by the voxelizer

  // Pad the mesh with zeros to match the block size
  padWithZeros(&d_position_idx,
               &d_material_idx,
               &voxelization_dim,
               block_size.x,
               block_size.y,
               block_size.z);

  if(this->m_mesh.isDouble()) {
    this->m_mesh.setupMesh<double>(m_materials.getNumberOfUniqueMaterials(),
                                   m_materials.getMaterialCoefficientPtrDouble(),
                                   m_parameters.getParameterPtrDouble(),
                                   voxelization_dim,
                                   block_size,
                                   (unsigned int)this->m_parameters.getUpdateType());
  }
  else {
    this->m_mesh.setupMesh<float>(m_materials.getNumberOfUniqueMaterials(),
                                  m_materials.getMaterialCoefficientPtr(),
                                  m_parameters.getParameterPtr(),
                                  voxelization_dim,
                                  block_size,
                                  (unsigned int)this->m_parameters.getUpdateType());
  }

  if(convert_position_ptr) {
    unsigned int element_type = (unsigned int)this->m_parameters.getUpdateType();
    if(element_type == 0 || element_type == 1 || element_type == 3)
      this->m_mesh.toBilbaoScheme(d_position_idx, d_material_idx);
    else
      this->m_mesh.toKowalczykScheme(d_position_idx, d_material_idx);
  }

  this->num_elements_ = this->m_mesh.getNumberOfElements();

  if(this->force_partition_to_ != -1 && this->force_partition_to_ <= this->number_of_devices_) {
    this->m_mesh.makePartition(this->force_partition_to_,
                               d_position_idx, d_material_idx);
    log_msg<LOG_DEBUG>(L"App::initializeMesh - force partition to %d") %this->force_partition_to_;
  }
  else {
    log_msg<LOG_DEBUG>(L"App::initializeMesh - no forced number of partitions, 1 parititons");
    this->m_mesh.makePartition(1, d_position_idx, d_material_idx);
  }
}

void App::initializeMeshHost() {
  float dx = this->m_parameters.getDx();
  mesh_size_t est_nodes;
  est_nodes= (mesh_size_t)((this->m_geometry.getBoundingBox()).x/dx*
                                      (this->m_geometry.getBoundingBox()).y/dx*
                                      (this->m_geometry.getBoundingBox()).z/dx);

  unsigned int mesh_size_in_MB = est_nodes*10.00/1000000.0;
  unsigned int mesh_size_in_MB_double = est_nodes*18.0/1000000.0;

  log_msg<LOG_INFO>(L"App::initializeMesh - Estimated size: %u nodes,"
                    L"dx: %f, size float: %u MB, double: %u MB")
                    %est_nodes %dx %mesh_size_in_MB % mesh_size_in_MB_double;

  int total_mem = 0;
  for(int i = 0; i < this->number_of_devices_; i++) {
    total_mem += device_mem_sizes_.at(i);
  }

  if(total_mem < mesh_size_in_MB) {
    log_msg<LOG_INFO>(L"App::initializeMesh - Estimated size: %u nodes "
                      L"too large, exiting") %est_nodes;
    this->close();
    throw(-1);
  }
  if(this->m_mesh.isDouble() && total_mem < mesh_size_in_MB_double){
    log_msg<LOG_INFO>(L"App::initializeMesh - Estimated size: %u nodes "
                      L"too large, exiting") %est_nodes;
    this->close();
    throw(-1);
  }

    // Currently the voxeliztaion functions give right position indices. In the
  // case of GPU voxelization, a conversion is needed. Then the
  // control variable convert_position_ptr need to be set
  bool convert_position_ptr = false;
  uint3 vox_dim = get_Geometry_surface_Voxelization_dims(
                                 m_geometry.getVerticePtr(),
                                 m_geometry.getIndexPtr(),
                                 m_geometry.getNumberOfTriangles(),
                                 m_geometry.getNumberOfVertices(),
                                 (double) m_parameters.getDx());
  // Voxelize the geometry

    unsigned char* h_pos = NULL;
    unsigned char* h_mat = NULL;

    voxelizeGeometrySurfToHost(m_geometry.getVerticePtr(),
                               m_geometry.getIndexPtr(),
                               m_materials.getMaterialIdxPtr(),
                               m_geometry.getNumberOfTriangles(),
                               m_geometry.getNumberOfVertices(),
                               m_materials.getNumberOfUniqueMaterials(),
                               (double) m_parameters.getDx(),
                               &h_pos,
                               &h_mat,
                               &vox_dim,
                               1,
                               VOX_6_SEPARATING,
                               make_uint3(32, 4, 1));

  calcBoundaryValuesInplace(h_pos, vox_dim.x, vox_dim.y, vox_dim.z);
  calcMaterialIndicesInplace(h_mat, h_pos, vox_dim.x, vox_dim.y, vox_dim.z);
  // Initialize the mesh
  uint3 block_size = make_uint3(32,4,1); // this is a default block size used by the voxelizer

  if(this->m_mesh.isDouble()) {
    this->m_mesh.setupMesh<double>(m_materials.getNumberOfUniqueMaterials(),
                                  m_materials.getMaterialCoefficientPtrDouble(),
                                  m_parameters.getParameterPtrDouble(),
                                  vox_dim,
                                  block_size,
                                  (unsigned int)this->m_parameters.getUpdateType());
  }
  else {
    this->m_mesh.setupMesh<float>(m_materials.getNumberOfUniqueMaterials(),
                                  m_materials.getMaterialCoefficientPtr(),
                                  m_parameters.getParameterPtr(),
                                  vox_dim,
                                  block_size,
                                  (unsigned int)this->m_parameters.getUpdateType());
  }

  this->num_elements_ = this->m_mesh.getNumberOfElements();

  if(this->force_partition_to_ != -1 && this->force_partition_to_ <= this->number_of_devices_) {
    this->m_mesh.makePartitionFromHost(this->force_partition_to_,
                                       h_pos, h_mat);
    log_msg<LOG_DEBUG>(L"App::initializeMesh - force partition to %d") %this->force_partition_to_;
  }
  else {
    log_msg<LOG_DEBUG>(L"App::initializeMesh - no forced number of partitions, 1 parititons");
    this->m_mesh.makePartitionFromHost(1, h_pos, h_mat);
  }

  free(h_pos);
  free(h_mat);
}

void App::initializeDomain(unsigned char* position_idx_ptr,
                           unsigned char* material_idx_ptr,
                           unsigned int dim_x, unsigned int dim_y,
                           unsigned int dim_z) {
  log_msg<LOG_INFO>(L"App::initializeDomain - begin");
  this->m_parameters.setAddPaddingToElementIdx(false);
  this->has_geometry_ = false;
  uint3 voxelization_dim = make_uint3(dim_x, dim_y, dim_z);

  // this is a default block size used by the voxelizer
  uint3 block_size = make_uint3(32,4,1);

  if(this->m_mesh.isDouble()) {
    this->m_mesh.setupMesh<double>(m_materials.getNumberOfUniqueMaterials(),
                                   m_materials.getMaterialCoefficientPtrDouble(),
                                   m_parameters.getParameterPtrDouble(),
                                   voxelization_dim,
                                   block_size,
                                   (unsigned int)this->m_parameters.getUpdateType());
  }
  else {
    this->m_mesh.setupMesh<float>(m_materials.getNumberOfUniqueMaterials(),
                                  m_materials.getMaterialCoefficientPtr(),
                                  m_parameters.getParameterPtr(),
                                  voxelization_dim,
                                  block_size,
                                  (unsigned int)this->m_parameters.getUpdateType());
  }

  clock_t start_t;
  clock_t end_t;
  start_t = clock();
  log_msg<LOG_INFO>(L"App::initializeDomain - setup done, to partition");
  if(this->force_partition_to_ != -1 && this->force_partition_to_ <= this->number_of_devices_)
    this->m_mesh.makePartitionFromHost(this->force_partition_to_,
                                       position_idx_ptr,
                                       material_idx_ptr);
  else
    this->m_mesh.makePartitionFromHost(1,
                                       position_idx_ptr,
                                       material_idx_ptr);

  end_t = clock()-start_t;
  log_msg<LOG_INFO>(L"App::initializeDomain - make partition done time: %f seconds")
                    % ((float)end_t/CLOCKS_PER_SEC);

  log_msg<LOG_INFO>(L"App::initializeDomain - done");
}

#ifdef COMPILE_VISUALIZATION
void App::initializeWindow(int argc, char** argv) {
    log_msg<LOG_DEBUG>(L"App::initializeWindow - initialize GL");
    m_window->initializeGL(argc, argv);
    m_window->initializeWindow(1200, 800);

    if(this->has_geometry_)
      m_window->geometryToVbo((void*)&m_geometry);

    // Initialize pixel and vertex buffers

    // The dimensions of the pbo slice are set
    // according to the mesh
    float dx = this->m_parameters.getDx();
    nv::Vec3f pbo_dim((m_mesh.getDimX()*dx),
                      (m_mesh.getDimY()*dx),
                      (m_mesh.getDimZ()*dx));

    // XY slice
    m_window->addPixelBuffer(m_mesh.getDimX(), m_mesh.getDimY(),
                            nv::Vec3f(pbo_dim.x, 0.f, 0.f),
                            nv::Vec3f(0.f, pbo_dim.y, 0.f),
                            nv::Vec3f(0.f, 0.f, dx));

    // XZ slice
    m_window->addPixelBuffer(m_mesh.getDimX(), m_mesh.getDimZ(),
                            nv::Vec3f(pbo_dim.x, 0.f, 0.f),
                            nv::Vec3f(0.f, 0.f, pbo_dim.z),
                            nv::Vec3f(0.f, dx, 0.f));

    // YZ slice
    m_window->addPixelBuffer(m_mesh.getDimY(), m_mesh.getDimZ(),
                            nv::Vec3f(0.f, pbo_dim.y, 0.f),
                            nv::Vec3f(0.f, 0.f, pbo_dim.z),
                            nv::Vec3f(dx, 0.f, 0.f));

    cudasafe(cudaDeviceSynchronize(), "App::initializeWindow - device synch after GL init");
    registerGLtoCuda(&(GLOBAL::pbo_xy), m_window->getPboIdAt(0), cudaGraphicsRegisterFlagsWriteDiscard);
    registerGLtoCuda(&(GLOBAL::pbo_xz), m_window->getPboIdAt(1), cudaGraphicsRegisterFlagsWriteDiscard);
    registerGLtoCuda(&(GLOBAL::pbo_yz), m_window->getPboIdAt(2), cudaGraphicsRegisterFlagsWriteDiscard);
}

///////////////////////////////////////////////////////////////////////////////
// Execution functions
///////////////////////////////////////////////////////////////////////////////
void App::runVisualization() {
  // Mock arguments for GL
  int argc = 0;
  char** argv = NULL;

  this->m_window = new AppWindow();

  // Visualization is run in single precision and on a single device
  m_mesh.setDouble(false);
  this->m_parameters.setNumSteps(this->m_parameters.getSpatialFs()*2);

  this->force_partition_to_ = 1;

  if(this->has_geometry_ && this->m_mesh.getNumberOfPartitions() == 0)
    this->initializeMeshHost();

  this->initializeWindow(argc, argv);
  log_msg<LOG_INFO>(L"App::runVisualization - after initWindow");
  this->updateVisualization(0, 0, 0, 8.f);
  this->updateVisualization(0, 1, 0, 8.f);
  this->updateVisualization(0, 2, 0, 8.f);
  this->responses_.assign(m_parameters.getNumSteps()*m_parameters.getNumReceivers(), 0.f);

  log_msg<LOG_INFO>(L"App::runVisualization - Volume: %f") %this->getVolume();
  log_msg<LOG_INFO>(L"App::runVisualization - TotalAbsorptionArea: %f, octave: %u")
                    %this->getTotalAborptionArea(0) %this->m_parameters.getOctave();

  log_msg<LOG_INFO>(L"App::runVisualization - Sabine RT: %f") %this->getSabine(0);
  log_msg<LOG_INFO>(L"App::runVisualization - Eyrting RT: %f") %this->getEyring(0);

  m_window->startMainLoop(this);
}
#endif

void App::runSimulation() {
  clock_t start_t;
  clock_t end_t;
  start_t = clock();

  if(this->has_geometry_ && this->m_mesh.getNumberOfPartitions() == 0)
    this->initializeMesh(1);

  unsigned int oct = this->m_parameters.getOctave();
  log_msg<LOG_INFO>(L"App::runSimulation - Volume: %f") %this->getVolume();
  log_msg<LOG_INFO>(L"App::runSimulation - Surface Area: %f") %this->m_geometry.getTotalSurfaceArea();
  log_msg<LOG_INFO>(L"App::runSimulation - TotalAbsorptionArea: %f, octave: %u") %this->getTotalAborptionArea(oct)% oct;
  log_msg<LOG_INFO>(L"App::runSimulation - Sabine RT: %f") %this->getSabine(oct);
  log_msg<LOG_INFO>(L"App::runSimulation - Eyrting RT: %f") %this->getEyring(oct);

  // Execute simulation
  if(this->m_mesh.isDouble()) {
    this->responses_double_.assign(m_parameters.getNumSteps()*m_parameters.getNumReceivers(), 0);
    this->time_per_step_ = launchFDTD3dDouble(&this->m_mesh,
                                              &(this->m_parameters),
                                              &this->responses_double_[0],
                                              this->m_interrupt,
                                              this->m_progress);

  }
  else {
    this->responses_.assign(m_parameters.getNumSteps()*m_parameters.getNumReceivers(), 0.f);
    this->time_per_step_ = launchFDTD3d(&this->m_mesh,
                                        &(this->m_parameters),
                                        &this->responses_[0],
                                        this->m_interrupt,
                                        this->m_progress);

  }

  end_t = clock()-start_t;
  log_msg<LOG_INFO>(L"App::runMex - time: %f seconds")
                    % ((float)end_t/CLOCKS_PER_SEC);
  log_msg<LOG_INFO>(L"App::runMex - Performance Mvox/sec: %f ")
                    % ((1.f/this->time_per_step_*this->m_mesh.getNumberOfElements())/1e6);

}

void App::runCapture() {
  clock_t start_t;
  clock_t end_t;
  start_t = clock();
  GLOBAL::currentApp = this;

  if(this->has_geometry_)
    this->initializeMesh(2);

  unsigned int step = this->m_parameters.getNumSteps();
  this->responses_.assign(m_parameters.getNumSteps()*m_parameters.getNumReceivers(), 0.f);

  // Run steps
  for(unsigned int i = 0; i < step; i++) {
    this->executeStep();
    if(this->m_interrupt())
      break;
  }

  end_t = clock()-start_t;
  log_msg<LOG_INFO>(L"LaunchFDTD3d - time: %f seconds, per step: %f")
              % ((float)end_t/CLOCKS_PER_SEC) % (((float)end_t/CLOCKS_PER_SEC)/step);


  this->time_per_step_ = (((float)end_t/CLOCKS_PER_SEC)/step);

  end_t = clock()-start_t;
  log_msg<LOG_INFO>(L"App::runCapture - time: %f seconds")
            % ((float)end_t/CLOCKS_PER_SEC);

}

////////////////////////////////////////////////
// Visualization controls
///////////////////////////////////////////////////////////////////////////////
#ifdef COMPILE_VISUALIZATION
void App::updateVisualization(unsigned int current_slice,
                              unsigned int orientation,
                              unsigned int selector,
                              float dB) {
  unsigned int scheme = (unsigned int)this->m_parameters.getUpdateType();
  if(orientation == 0)
    updatePixelBuffer(&(GLOBAL::pbo_xy), &(this->m_mesh), current_slice, orientation, selector, scheme, dB);
  if(orientation == 1)
    updatePixelBuffer(&(GLOBAL::pbo_xz), &(this->m_mesh), current_slice, orientation, selector, scheme, dB);
  if(orientation == 2)
    updatePixelBuffer(&(GLOBAL::pbo_yz), &(this->m_mesh), current_slice, orientation, selector, scheme, dB);
}
#endif

void App::resetPressureMesh() {
  log_msg<LOG_INFO>(L"App::resetPressureMesh - reseting pressure mesh");
  this->m_mesh.resetPressures();
  this->current_step_ = 0;
}

void App::invertTime() {
  this->step_direction_ *= -1;
}

void App::executeStep() {
  clock_t start_t;
  clock_t end_t;
  start_t = clock();

  launchFDTD3dStep(&(this->m_mesh),
                   &(this->m_parameters),
                   &this->responses_[0],
                   this->current_step_,
                   this->step_direction_,
                   this->m_progress);

  this->current_step_ += this->step_direction_;

#ifdef COMPILE_VISUALIZATION
  captureSliceFast(&(this->m_mesh),
                   this->step_to_capture_,
                   this->slice_to_capture_,
                   this->slice_orientation_,
                   this->current_step_,
                   captureBitmap);

  captureMesh(&(this->m_mesh),
              this->mesh_to_capture_,
              this->mesh_captures_,
              this->current_step_);
#endif
  end_t = clock()-start_t;
  this->time_per_step_ = (this->time_per_step_+((float)end_t/CLOCKS_PER_SEC))/2.f;

}

void App::close() {
  log_msg<LOG_INFO>(L"App::close");
  cudaSetDevice(0);
  this->resetDevices();
  #ifdef COMPILE_VISUALIZATION
  delete this->m_window;
  #endif
}

///////////////////////////////////////////////////////////////////////////////
// Functions to calculate a theoretical reverberation times
// NOTE: the material coefficients are not random incidence coefficients
// and therefore the reverberation times can vary
///////////////////////////////////////////////////////////////////////////////
float App::getVolume() {
  float volume = 0.f;
  mesh_size_t number_of_elements = this->m_mesh.getNumberOfAirElements();
  number_of_elements += this->m_mesh.getNumberOfBoundaryElements();
  float dx = this->m_parameters.getDx();
  log_msg<LOG_INFO>(L"App::getVolume - number_of_elements %lu, dx %f") % number_of_elements %dx;
  volume = dx*dx*dx*number_of_elements;
  return volume;
}

float App::getTotalAborptionArea(unsigned int octave) {
  float total_absorption_area = 0.f;
  for(unsigned int i = 0; i < this->m_geometry.getNumberOfTriangles(); i++) {
    float coef = admitance2Reflection(this->m_materials.getSurfaceCoefAt(i,octave));
    coef = 1.0-coef*coef;
    log_msg<LOG_VERBOSE>(L"App::getTotalAborptionArea - Abs coef in %u, %f") %i %coef;
    float surface_area = this->m_geometry.getSurfaceAreaAt(i);
    total_absorption_area += surface_area*coef;
  }
  return total_absorption_area;
}

float App::getSabine(unsigned int octave) {
  float rt = 0.f;
  float volume = this->getVolume();
  float total_absorption_area = this->getTotalAborptionArea(octave);
  rt = 0.1611f*volume/total_absorption_area;
  log_msg<LOG_INFO>(L"App::getSabine - total absorption area %f, octave %u,"
                    L"Volume: %f, RT: %f")
                    % total_absorption_area % octave % volume  % rt;

  return rt;
}

float App::getEyring(unsigned int octave) {
  float rt = 0.f;
  float volume = this->getVolume();
  float total_surface_area = this->m_geometry.getTotalSurfaceArea();
  float total_absorption_area = this->getTotalAborptionArea(octave);
  float mean_absorption = total_absorption_area/total_surface_area;
  rt = 0.1611f*volume/(-1.f*total_surface_area*log(1.0-mean_absorption));

  log_msg<LOG_INFO>(L"App::getEyring - mean absorption %f, octave %u,"
                    L"Total surface area: %f, RT: %f") % mean_absorption
                    % octave %total_surface_area %rt;
  return rt;
}

void App::saveBitmap(float* data,
                     unsigned char* position_data,
                     unsigned int dim_x,
                     unsigned int dim_y,
                     unsigned int slice,
                     unsigned int orientation,
                     unsigned int step) {
  TGAImage *img = new TGAImage((short)dim_x, (short)dim_y);
  TGAImage::Colour co;

  float dB = this->capture_db_/10.f;
  for(unsigned int i = 0; i < dim_y; i++) {
    for(unsigned int j = 0; j < dim_x; j++) {
      float c = (data[i*dim_x+j]);
      unsigned char c_pos = position_data[i*dim_x+j];

      if((c_pos>>7) == 0X01 && (c_pos&0X7F)!=0x06) {
        co.r = 255;
        co.b = 255;
        co.g = 255;
        co.a = 255;
        img->setPixel(co,i,j);

        continue;
      }
      else {
        co.r = 0;
        co.b = 0;
        co.g = 0;
        co.a = 255;
        img->setPixel(co,i,j);
      }

      float c_ = 0.f;
      float _c = 0.f;
      float sign = c> 0.f ? 1.f : -1.f;

      c = ((log10(c*c)+dB)/dB);
      c = c>0?c:0.f;

      _c = sign>=0?c:0.f;
      c_ = sign<0?c:0.f;

      co.r = (unsigned char)0;
      co.g = (unsigned char)(_c*255);
      co.b = (unsigned char)(c_*255);
      co.a = 255;
      img->setPixel(co, i,j);
    }
  }


  for(unsigned int k = 0; k < this->m_parameters.getNumSources(); k++) {

    nv::Vec3i source_element_idx = this->m_parameters.getSourceElementCoordinates(k);
    int x, y;
    if(orientation == 0) {x = source_element_idx.x; y = source_element_idx.y;};
    if(orientation == 1) {x = source_element_idx.x; y = source_element_idx.z;};
    if(orientation == 2) {x = source_element_idx.y; y = source_element_idx.z;};

  }

  for(unsigned int k = 0; k < this->m_parameters.getNumReceivers(); k++) {

    nv::Vec3i source_element_idx = this->m_parameters.getReceiverElementCoordinates(k);
    int x, y;
    if(orientation == 0) {x = source_element_idx.x; y = source_element_idx.y;};
    if(orientation == 1) {x = source_element_idx.x; y = source_element_idx.z;};
    if(orientation == 2) {x = source_element_idx.y; y = source_element_idx.z;};

  }

  std::stringstream ss;
  if(this->capture_id_ == "")
    ss << "capture_"<<orientation<<"_"<<slice<<"_"<<setfill('0')<<setw(5)<<step<<".tga";
  else
    ss << this->capture_id_<<"_"<<orientation<<"_"<<slice<<"_"<<setfill('0')<<setw(5)<<step<<".tga";
  img->WriteImage(ss.str());
}
