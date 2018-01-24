#if !defined(WIN32)
  #define BOOST_TEST_DYN_LINK
#endif

#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include "../src/kernels/cudaMesh.h"
#include "../src/kernels/voxelizationUtils.h"

#include "../src/kernels/cudaUtils.h"
#include "../src/kernels/kernels3d.h"
#include "../src/base/GeometryHandler.h"
#include "../src/io/FileReader.h"

SimulationParameters parameters;
MaterialHandler materials;
GeometryHandler geometry;

int count = 0;

#define NUM_STEPS  100
#define PING(x) std::cout<<"Ping "<<x<<"! ("<<count<<")"<<std::endl;count++;

extern "C" {
  bool interruptCallbackLocal(){
    if(false)
      log_msg<LOG_INFO>(L"main: Execution interrupted");

    return false;
  }
}

extern "C" {
  void progressCallbackLocal(int step, int max_step, float t_per_step ){
    float estimate = t_per_step*(float)(max_step-step);
    printf("Step %d/%d, time per step %f, estimated time left %f s \n",
           step, max_step, t_per_step, estimate);

    return;
  }
}

std::vector<unsigned int> getDebugDevices(int num_devices) {
 std::vector<unsigned int> debug_devices;
  for(int i = 0; i < num_devices; i++) {
    debug_devices.push_back(0);
    //std::cout<<"Debug device: "<<(i%2)<<std::endl;
  }
  return debug_devices;
}

CudaMesh* getTestMesh(unsigned int number_of_partitions) {
  int number_of_devices = 0;
  cudasafe(cudaGetDeviceCount(&number_of_devices), "getDeviceCount");
  std::cout<<"getTestMesh - number of  devices: "<<number_of_devices<<std::endl;
  for (int i = 0; i<number_of_devices; i++) {
    cudasafe(cudaSetDevice(i), "cudaSetDevice");
    cudasafe(cudaDeviceReset(), "cudaDeviceReset");
  }
  cudasafe(cudaSetDevice(0), "cudaSetDevice");

  FileReader fr;

  if(!fr.readVTK(&geometry, "./Data/hytti.vtk"))
    throw(-1);

  materials.setGlobalMaterial(geometry.getNumberOfTriangles(), 0.5f);
  parameters.setSpatialFs(10000);
  parameters.setNumSteps(NUM_STEPS);
  parameters.resetSourcesAndReceivers();

  float dx = parameters.getDx();
  // Add three sources and receivers, bottom, halo and top
  // Mesh dim: x 128 y 96 z 49

  for(int i = 0; i < 45; i++)
    parameters.addSource(Source( 2.f, 1.f, i*dx, SRC_SOFT));
  // parameters.addSource(Source( 2.f, 1.f, 3.f*dx, SRC_SOFT));
  // parameters.addSource(Source( 2.f, 1.f, 23.f*dx, SRC_SOFT));
  // parameters.addSource(Source( 2.f, 1.f, 40.f*dx, SRC_SOFT));

  parameters.addReceiver(Receiver( 3.f, 1.f, 5.f*dx));
  parameters.addReceiver(Receiver( 3.f, 1.f, 24.f*dx));
  parameters.addReceiver(Receiver( 3.f, 1.f, 42.f*dx));

  parameters.setUpdateType(SRL_FORWARD);

  unsigned char* d_position_idx = (unsigned char*)NULL;
  unsigned char* d_material_idx = (unsigned char*)NULL;
  uint3 voxelization_dim = make_uint3(0,0,0);
  uint3 vox_dim = get_Geometry_surface_Voxelization_dims(
                    geometry.getVerticePtr(),
                    geometry.getIndexPtr(),
                    geometry.getNumberOfTriangles(),
                    geometry.getNumberOfVertices(),
                    (double)parameters.getDx());

  // Voxelize the geometry
  voxelizeGeometry_surface_6_separating(geometry.getVerticePtr(),
                                       geometry.getIndexPtr(),
                                       materials.getMaterialIdxPtr(),
                                       geometry.getNumberOfTriangles(),
                                       geometry.getNumberOfVertices(),
                                       materials.getNumberOfUniqueMaterials(),
                                       parameters.getDx(),
                                       &d_position_idx,
                                       &d_material_idx,
                                       &voxelization_dim,
                                       0, vox_dim.z);

  CudaMesh* cuda_mesh = new CudaMesh();
    // Initialize the mesh
  uint3 block_size = make_uint3(32,4,1); // this is a default block size used by the voxelizer

  padWithZeros(&d_position_idx,
               &d_material_idx,
               &voxelization_dim,
               block_size.x,
               block_size.y,
               block_size.z);

  cuda_mesh->setupMesh<float>(materials.getNumberOfUniqueMaterials(),
                             materials.getMaterialCoefficientPtr(),
                             parameters.getParameterPtr(),
                             voxelization_dim,
                             block_size,
                             (unsigned int)parameters.getUpdateType());

  unsigned int element_type = (unsigned int)parameters.getUpdateType();
  if(element_type == 0 || element_type == 1 || element_type == 3)
    cuda_mesh->toBilbaoScheme(d_position_idx, d_material_idx);
  else
    cuda_mesh->toKowalczykScheme(d_position_idx, d_material_idx);

  cuda_mesh->makePartition(number_of_partitions,
                           d_position_idx,  d_material_idx,
                           getDebugDevices(number_of_partitions));


  return cuda_mesh;
}

CudaMesh* getTestMeshDouble(unsigned int number_of_partitions) {
  int number_of_devices = 0;
  cudasafe(cudaGetDeviceCount(&number_of_devices), "getDeviceCount");
  std::cout<<"getTestMeshDouble - number of  devices: "<<number_of_devices<<std::endl;
  for (int i = 0; i<number_of_devices; i++) {
    cudasafe(cudaSetDevice(i), "cudaSetDevice");
    cudasafe(cudaDeviceReset(), "cudaDeviceReset");
  }

  cudasafe(cudaSetDevice(0), "cudaSetDevice");
  
  FileReader fr;

  fr.readVTK(&geometry, "./Data/hytti.vtk");
  materials.setGlobalMaterial(geometry.getNumberOfTriangles(), 0.5f);

  parameters.setSpatialFs(10000);
  parameters.setNumSteps(NUM_STEPS);
  parameters.resetSourcesAndReceivers();
  float dx = parameters.getDx();
  // Add three sources and receivers, bottom, halo and top
  // Mesh dim: x 128 y 96 z 49

  parameters.addSource(Source( 2.f, 1.f, 3.f*dx, SRC_SOFT));
  parameters.addSource(Source( 2.f, 1.f, 23.f*dx, SRC_SOFT));
  parameters.addSource(Source( 2.f, 1.f, 40.f*dx, SRC_SOFT));

  parameters.addReceiver(Receiver( 3.f, 1.f, 5.f*dx));
  parameters.addReceiver(Receiver( 3.f, 1.f, 24.f*dx));
  parameters.addReceiver(Receiver( 3.f, 1.f, 42.f*dx));


  parameters.setUpdateType(SRL_FORWARD);

  unsigned char* d_position_idx = (unsigned char*)NULL;
  unsigned char* d_material_idx = (unsigned char*)NULL;
  uint3 voxelization_dim = make_uint3(0,0,0);
  // Voxelize the geometry
  //voxelizeGeometry_solid(geometry.getVerticePtr(),
  //                 geometry.getIndexPtr(),
  //                 materials.getMaterialIdxPtr(),
  //                 geometry.getNumberOfTriangles(),
  //                 geometry.getNumberOfVertices(),
  //                 materials.getNumberOfUniqueMaterials(),
  //                 parameters.getDx(),
  //                 &d_position_idx,
  //                 &d_material_idx,
  //                 &voxelization_dim);

  uint3 vox_dim = get_Geometry_surface_Voxelization_dims(
    geometry.getVerticePtr(),
    geometry.getIndexPtr(),
    geometry.getNumberOfTriangles(),
    geometry.getNumberOfVertices(),
    (double)parameters.getDx());

  // Voxelize the geometry
  voxelizeGeometry_surface_6_separating(geometry.getVerticePtr(),
                                        geometry.getIndexPtr(),
                                        materials.getMaterialIdxPtr(),
                                        geometry.getNumberOfTriangles(),
                                        geometry.getNumberOfVertices(),
                                        materials.getNumberOfUniqueMaterials(),
                                        parameters.getDx(),
                                        &d_position_idx,
                                        &d_material_idx,
                                        &voxelization_dim,
                                        0, vox_dim.z);
  CudaMesh* cuda_mesh = new CudaMesh();
  cuda_mesh->setDouble(true);
    // Initialize the mesh
  uint3 block_size = make_uint3(32,4,1); // this is a default block size used by the voxelizer

  padWithZeros(&d_position_idx,
               &d_material_idx,
               &voxelization_dim,
               block_size.x,
               block_size.y,
               block_size.z);

  cuda_mesh->setupMesh<double>(materials.getNumberOfUniqueMaterials(),
                               materials.getMaterialCoefficientPtrDouble(),
                               parameters.getParameterPtrDouble(),
                               voxelization_dim,
                               block_size,
                              (unsigned int)parameters.getUpdateType());

  unsigned int element_type = (unsigned int)parameters.getUpdateType();
  if(element_type == 0 || element_type == 1 || element_type == 3)
    cuda_mesh->toBilbaoScheme(d_position_idx, d_material_idx);
  else
    cuda_mesh->toKowalczykScheme(d_position_idx, d_material_idx);

  cuda_mesh->makePartition(number_of_partitions,
                           d_position_idx, d_material_idx,
                           getDebugDevices(number_of_partitions));

  return cuda_mesh;
}

BOOST_AUTO_TEST_SUITE(CudaMeshTest)

BOOST_AUTO_TEST_CASE(CudaMesh_partition_idx) {
  CudaMesh mesh;
  std::vector< std::vector<unsigned long long int> > partitions;
  std::vector< std::vector<unsigned long long int> > partitions1;
  int dim_z = 100;
  int num_partitions = 13;
  int partition_size = dim_z/num_partitions;

  partitions = mesh.getPartitionIndexing(num_partitions,dim_z);
  partitions1 = mesh.getPartitionIndexing(1, dim_z);
  BOOST_CHECK_EQUAL(partitions.size(), num_partitions);

  // With more than 1 partitions, slices have to overlap
  for(int i = 0; i < num_partitions; i++) {
    int end_inc = 0;
    int start_inc = 0;
    if(i == 0) { end_inc = 1;}
    if(i == (num_partitions-1)) { start_inc = 1;}
    if(i!=0 && i!=(num_partitions-1)) { start_inc = 1; end_inc = 1;}
    if(i == (num_partitions-1) && num_partitions>1) { end_inc = dim_z-(i+1)*partition_size;}

    BOOST_CHECK_EQUAL(partitions.at(i).size(), dim_z/num_partitions+end_inc+start_inc);
    //std::cout<<"Partition "<<i<<std::endl;
    for(int j = 0; j < partitions.at(i).size(); j++) {
      unsigned int slice = i*partition_size-start_inc+j;
      BOOST_CHECK_EQUAL(partitions.at(i).at(j), slice);
      //std::cout<<slice<<" ,";
    }
    //std::cout<<std::endl;
  }

  // If one partition, slices are from 0 to dim
  BOOST_CHECK_EQUAL(partitions1.size(), 1);
  for(int j = 0; j < partitions1.at(0).size(); j++) {
    BOOST_CHECK_EQUAL(partitions1.at(0).at(j), j);
  }
}

BOOST_AUTO_TEST_CASE(CudaMesh_set_get_utils) {
  CudaMesh mesh;
  uint3 voxelization_dim = make_uint3(20, 20, 20);
  uint3 block_size = make_uint3(32,4,1);
  unsigned int number_of_unique_materials(1);
  std::vector<float> material_coefficients(20, 0.f);
  std::vector<float> parameter_ptr(10, 0.f);
  unsigned int element_type = 0;

  unsigned char* d_position_ptr = valueToDevice<unsigned char>(20*20*20, (unsigned char)0, 0);
  unsigned char* d_material_ptr = valueToDevice<unsigned char>(20*20*20, (unsigned char)0, 0);

  // Pad the mesh with zeros to match the block size
  padWithZeros(&d_position_ptr,
               &d_material_ptr,
               &voxelization_dim,
               block_size.x,
               block_size.y,
               block_size.z);

  mesh.setupMesh(number_of_unique_materials,
                 &material_coefficients[0],
                 &parameter_ptr[0],
                 voxelization_dim,
                 block_size,
                 element_type);

  int num_devices = 1;

  std::vector<unsigned int> debug_devices = getDebugDevices(num_devices);
  mesh.makePartition(num_devices,
                     d_position_ptr,
                     d_material_ptr,
                     debug_devices);

  BOOST_CHECK_EQUAL(mesh.pressures_.size(), num_devices);

  int dev_i;
  unsigned long long int elem;
  mesh.getElementIdxAndDevice(10, 10, 10, &dev_i, &elem);

  BOOST_CHECK_EQUAL(dev_i, 0);
  BOOST_CHECK_EQUAL(elem, 10*20*32+10*32+10);
  mesh.getElementIdxAndDevice(40, 10, 50, &dev_i, &elem);
  BOOST_CHECK_EQUAL(elem, -1);
  BOOST_CHECK_EQUAL(dev_i, -1);
  mesh.destroyPartitions();
}

BOOST_AUTO_TEST_CASE(CudaMesh_get_set_multi) {
  CudaMesh mesh;
  int dim = 50;
  uint3 voxelization_dim = make_uint3(dim, dim, dim);
  uint3 block_size = make_uint3(1,1,1);
  unsigned int number_of_unique_materials(1);
  std::vector<float> material_coefficients(20, 0.f);
  std::vector<float> parameter_ptr(10, 0.f);
  unsigned int element_type = 0;

  unsigned char* d_position_ptr = valueToDevice<unsigned char>(dim*dim*dim, (unsigned char)0, 0);
  unsigned char* d_material_ptr = valueToDevice<unsigned char>(dim*dim*dim, (unsigned char)0, 0);

  // Pad the mesh with zeros to match the block size
  padWithZeros(&d_position_ptr,
               &d_material_ptr,
               &voxelization_dim,
               block_size.x,
               block_size.x,
               block_size.z);

  mesh.setupMesh(number_of_unique_materials,
                 &material_coefficients[0],
                 &parameter_ptr[0],
                 voxelization_dim,
                 block_size,
                 element_type);



  BOOST_CHECK_EQUAL(mesh.getDimX(), dim);
  BOOST_CHECK_EQUAL(mesh.getDimY(), dim);
  BOOST_CHECK_EQUAL(mesh.getDimZ(), dim);

  int num_devices = 5;
  std::vector<unsigned int> debug_devices = getDebugDevices(num_devices);

  mesh.makePartition(num_devices,
                     d_position_ptr,
                     d_material_ptr,
                     debug_devices);

  BOOST_CHECK_EQUAL(mesh.pressures_.size(), num_devices);

  int dev_i;
  unsigned long long int elem;
  mesh.getElementIdxAndDevice(10, 10, 11, &dev_i, &elem);

  BOOST_CHECK_EQUAL(dev_i, 1);
  BOOST_CHECK_EQUAL(elem, 2*dim*dim+10*dim+10);

  mesh.destroyPartitions();
}


BOOST_AUTO_TEST_CASE(CudaMesh_test_1_partition) {
  cudasafe(cudaPeekAtLastError(),
           "CudaMesh_test_1_partition_begin");
  CudaMesh* mesh = getTestMesh(1);
  mesh->setSample(3.0, 0,0,0);
  float sample = mesh->getSample<float>(0,0,0);
  BOOST_CHECK_EQUAL(sample, 3.0);
  mesh->addSample(3.f, 0,0,0);
  sample = mesh->getSample<float>(0,0,0);

  BOOST_CHECK_EQUAL(sample, 6.f);

  mesh->setSample(1.f, 27, 27, 27);
  sample = mesh->getSample<float>(27,27, 27);
  BOOST_CHECK_EQUAL(sample, 1.f);


  mesh->destroyPartitions();
  delete mesh;
}


BOOST_AUTO_TEST_CASE(CudaMesh_test_2_partitions) {

  CudaMesh* mesh = getTestMesh(2);

  BOOST_CHECK_EQUAL(mesh->getNumberOfPartitions(), 2);

  unsigned int num_elems = mesh->getNumberOfElements();

  mesh->setSample<float>(3.f, 0,0,0);
  float sample = mesh->getSample<float>(0,0,0);
  BOOST_CHECK_EQUAL(sample, 3.f);

  mesh->addSample<float>(3.f, 0,0,0);
  mesh->addSample<float>(3.f, 0,0,0);
  mesh->addSample<float>(3.f, 0,0,0);
  sample = mesh->getSample<float>(0,0,0);
  BOOST_CHECK_EQUAL(sample, 12.f);

  mesh->setSample<float>(1.f, 27, 27, 27);
  sample = mesh->getSample<float>(27,27, 27);
  BOOST_CHECK_EQUAL(sample, 1.f);

  // Samples to halos
  mesh->setSample<float>(1.f, 2, 0, 23);
  mesh->setSample<float>(89.f, 6, 0, 24);
  mesh->addSample<float>(90.f, 12, 0, 24);

  // Fetch the samples from 1st partition
  sample = mesh->getSampleAt<float>(2,0,23,0);
  BOOST_CHECK_EQUAL(sample, 1.f);
  sample = mesh->getSampleAt<float>(6,0,24,0);
  BOOST_CHECK_EQUAL(sample, 89.f);
  sample = mesh->getSampleAt<float>(12,0,24,0);
  BOOST_CHECK_EQUAL(sample, 90.f);

  // Fetch the samples from 2nd partition
  sample = mesh->getSampleAt<float>(2,0,0,1);
  BOOST_CHECK_EQUAL(sample, 1.f);
  sample = mesh->getSampleAt<float>(6,0,1,1);
  BOOST_CHECK_EQUAL(sample, 89.f);
  sample = mesh->getSampleAt<float>(12,0,1,1);
  BOOST_CHECK_EQUAL(sample, 90.f);
  // Test halo switch
  sample = 0.f;
  // set sample to the second slice of second partition
  mesh->setSampleAt<float>(8.f, 2,0,1,1);
  // set sample to the second last slice of first partition
  mesh->setSampleAt<float>(45.f, 3,0,23,0);
  mesh->switchHalos();

  // last slice of the first should have the sample of second partition
  sample = mesh->getSampleAt<float>(2,0,24,0);
  BOOST_CHECK_EQUAL(sample, 8.f);
  // first slice of second partition should have the sample from first partition
  sample = mesh->getSampleAt<float>(3,0,0,1);
  BOOST_CHECK_EQUAL(sample, 45.f);

  // Check the consistency of z dim
  for(int i = 0; i < 49; i++) {
   mesh->addSample<float>((float)i, 7, 0, i);
  }

  // Check first partition
  for(int i = 0; i < mesh->getPartitionSize(0); i++) {
    BOOST_CHECK_EQUAL(mesh->getSampleAt<float>(7, 0, i, 0), (float)i);
  }

  // Check second partition
  for(int i = 0; i < mesh->getPartitionSize(1); i++) {
    int local_idx = mesh->partition_indexing_.at(1).at(0)+i;
    BOOST_CHECK_EQUAL(mesh->getSampleAt<float>(7, 0, i, 1), (float)local_idx);
  }

  // Switching, Samples should match again
  mesh->switchHalos();
  // Check first partition
  for(int i = 0; i < mesh->getPartitionSize(0); i++) {
    BOOST_CHECK_EQUAL(mesh->getSampleAt<float>(7, 0, i, 0), (float)i);
  }

  // Check second partition
  for(int i = 0; i < mesh->getPartitionSize(1); i++) {
    int local_idx = mesh->partition_indexing_.at(1).at(0)+i;
    BOOST_CHECK_EQUAL(mesh->getSampleAt<float>(7, 0, i, 1), (float)local_idx);
  }
  float sum = 0.f;


  if(!IGNORE_CHECKSUMS) {
    sum = printCheckSum(mesh->getPressurePtrAt(0), mesh->getPartitionSize(), "mesh test partition 0");
    BOOST_CHECK_EQUAL(sum, 6.f);
    sum = printCheckSum(mesh->getPressurePtrAt(1), mesh->getPartitionSize(), "mesh test partition 1");
    BOOST_CHECK_EQUAL(sum, 1.f);

    mesh->flipPressurePointers();
    mesh->switchHalos();
    sum = printCheckSum(mesh->getPressurePtrAt(0), mesh->getPartitionSize(), "mesh test partition 0");
    BOOST_CHECK_EQUAL(sum, 0.f);
    sum = printCheckSum(mesh->getPressurePtrAt(1), mesh->getPartitionSize(), "mesh test partition 1");
    BOOST_CHECK_EQUAL(sum, 0.f);
  }


  mesh->destroyPartitions();
  delete mesh;
}


BOOST_AUTO_TEST_CASE(CudaMesh_test_2_partitions_double) {

  CudaMesh* mesh = getTestMeshDouble(2);

  BOOST_CHECK_EQUAL(mesh->getNumberOfPartitions(), 2);

  unsigned int num_elems = mesh->getNumberOfElements();

  mesh->setSample<double>(3, 0,0,0);
  double sample = mesh->getSample<double>(0,0,0);
  BOOST_CHECK_EQUAL(sample, 3);

  mesh->addSample<double>(3, 0,0,0);
  sample = mesh->getSample<double>(0,0,0);
  BOOST_CHECK_EQUAL(sample, 6);

  mesh->setSample<double>(1, 27, 27, 27);
  sample = mesh->getSample<double>(27,27, 27);
  BOOST_CHECK_EQUAL(sample, 1);

  // Samples to halos
  for(int i = 0; i < mesh->getDimX(); i ++) {
    for(int j = 0; j < mesh->getDimY(); j++) {
      mesh->setSample<double>(1, i, j, 23);
      mesh->setSample<double>(89, i, j, 24);
    }
  }
  // Fetch the samples from 1st partition
  for(int i = 0; i < mesh->getDimX(); i ++) {
    for(int j = 0; j < mesh->getDimY(); j++) {
      sample = mesh->getSampleAt<double>(i,j,23,0);
      BOOST_CHECK_EQUAL(sample, 1.f);
      sample = mesh->getSampleAt<double>(i,j,24,0);
      BOOST_CHECK_EQUAL(sample, 89.f);

      // Fetch the samples from 2nd partition
      sample = mesh->getSampleAt<double>(i,j,0,1);
      BOOST_CHECK_EQUAL(sample, 1.f);
      sample = mesh->getSampleAt<double>(i,j,1,1);
      BOOST_CHECK_EQUAL(sample, 89.f);
    }
  }
  // Test halo switch
  sample = 0;
  // set sample to the second slice of second partition
  mesh->setSampleAt<double>(8, 2,0,1,1);
  mesh->addSampleAt<double>(8, 2,0,1,1);
  // set sample to the second last slice of first partition
  mesh->setSampleAt<double>(45, 3,0,23,0);
  mesh->switchHalos();

  // last slice of the first should have the sample of second partition
  sample = mesh->getSampleAt<double>(2,0,24,0);
  BOOST_CHECK_EQUAL(sample, 16.f);
  // first slice of second partition should have the sample from first partition
  sample = mesh->getSampleAt<double>(3,0,0,1);
  BOOST_CHECK_EQUAL(sample, 45.f);

  float sum = 0.f;


  if(!IGNORE_CHECKSUMS) {
    sum = printCheckSum(mesh->getPressurePtrAt(0), mesh->getPartitionSize(), "mesh test partition 0");
    BOOST_CHECK_EQUAL(sum, 6.f);
    sum = printCheckSum(mesh->getPressurePtrAt(1), mesh->getPartitionSize(), "mesh test partition 1");
    BOOST_CHECK_EQUAL(sum, 1.f);

    mesh->flipPressurePointers();
    mesh->switchHalos();
    sum = printCheckSum(mesh->getPressurePtrAt(0), mesh->getPartitionSize(), "mesh test partition 0");
    BOOST_CHECK_EQUAL(sum, 0.f);
    sum = printCheckSum(mesh->getPressurePtrAt(1), mesh->getPartitionSize(), "mesh test partition 1");
    BOOST_CHECK_EQUAL(sum, 0.f);
  }

  mesh->destroyPartitions();
  delete mesh;
}


BOOST_AUTO_TEST_CASE(CudaMesh_Run_single) {

  // Run single partition simulation
  CudaMesh* mesh1 = getTestMesh(1);

  float* ret1 = new float[parameters.getNumSteps()*parameters.getNumReceivers()];
  launchFDTD3d(mesh1, &parameters, ret1, interruptCallbackLocal, progressCallbackLocal);
  mesh1->destroyPartitions();

  delete mesh1;

  // Run two partition simulations
  CudaMesh* mesh2 = getTestMesh(2);

  float* ret2 = new float[parameters.getNumSteps()*parameters.getNumReceivers()];

  launchFDTD3d(mesh2, &parameters, ret2, interruptCallbackLocal, progressCallbackLocal);
  mesh2->destroyPartitions();

  delete mesh2;

  // Simulate 5 partitio simulation
  CudaMesh* mesh3 = getTestMesh(5);

  float* ret3 = new float[parameters.getNumSteps()*parameters.getNumReceivers()];

  launchFDTD3d(mesh3, &parameters, ret3, interruptCallbackLocal, progressCallbackLocal);

  mesh3->destroyPartitions();
  delete mesh3;

  // Check results
  bool check = true;
  float sum1 = 0;
  float sum2 = 0;
  float sum3 = 0;
  for(unsigned int i = 0; i < parameters.getNumSteps(); i++) {
    sum1 += ret1[i];    sum2 += ret2[i];    sum3 += ret3[i];
    //if(ret1[i]!= 0)
    //	printf("data  found step %u r1 %f r2 %f r3 %f\n",i, ret1[i], ret2[i], ret3[i]);
    if(ret1[i] != ret2[i] )
      check = false;
    if(ret1[i] != ret3[i])
      check = false;
  }

  BOOST_CHECK_EQUAL(check, true);

  delete[] ret1;
  delete[] ret2;
  delete[] ret3;
}

BOOST_AUTO_TEST_CASE(CudaMesh_Run_single_double_core_double) {

  // Run single partition simulation
  CudaMesh* mesh1 = getTestMeshDouble(1);

  double* ret1 = new double[parameters.getNumSteps()*parameters.getNumReceivers()];
  launchFDTD3dDouble(mesh1, &parameters, ret1, interruptCallbackLocal, progressCallbackLocal);

  mesh1->destroyPartitions();
  delete mesh1;

  // Run two partition simulations
  CudaMesh* mesh2 = getTestMeshDouble(2);

  double* ret2 = new double[parameters.getNumSteps()*parameters.getNumReceivers()];

  launchFDTD3dDouble(mesh2, &parameters, ret2, interruptCallbackLocal, progressCallbackLocal);

  mesh2->destroyPartitions();
  delete mesh2;

  // Simulate 5 partitio simulation
  CudaMesh* mesh3 = getTestMeshDouble(8);

  double* ret3 = new double[parameters.getNumSteps()*parameters.getNumReceivers()];

  launchFDTD3dDouble(mesh3, &parameters, ret3, interruptCallbackLocal, progressCallbackLocal);

  mesh3->destroyPartitions();
  delete mesh3;
  double sum1 = 0;
  double sum2 = 0;
  double sum3 = 0;
  // Check results
  bool check = true;
  for(unsigned int i = 0; i < parameters.getNumSteps(); i++) {
    sum1 += ret1[i];    sum2 += ret2[i];    sum3 += ret3[i];
    if(ret1[i] != ret2[i])
      check = false;
    if(ret1[i] != ret3[i])
      check = false;
  }

  std::cout<<"sum1 "<<sum1<<" sum2 "<<sum2<<" sum 3 " <<sum3<<std::endl;
  BOOST_CHECK_EQUAL(check, true);

  delete[] ret1;
  delete[] ret2;
  delete[] ret3;
}

BOOST_AUTO_TEST_CASE(CudaMesh_partition_on_host) {
  CudaMesh mesh;
  SimulationParameters sp;
  sp.setSpatialFs(21000);
  sp.setNumSteps(300);
  sp.setUpdateType(SRL_FORWARD);
  float dx = sp.getDx();
  sp.addSource(Source(256*dx,256*dx,256*dx, SRC_SOFT, IMPULSE, 0));
  sp.addReceiver(Receiver(246*dx, 256*dx, 256*dx));

  unsigned long long int dim = 512;
  unsigned long long int num_elem = dim*dim*dim;
  std::vector<unsigned char> h_pos(num_elem, 6+128);
  std::vector<unsigned char> h_mat(num_elem, 0);
  std::vector<float> materials(20, 0.0f);
  uint3 vox_dim = make_uint3(dim,dim,dim);
  uint3 block_size = make_uint3(32,4,1);

  mesh.setupMesh<float>(1, &materials[0], sp.getParameterPtr(),
                        vox_dim, block_size, 0);

  mesh.makePartitionFromHost(1, &h_pos[0], &h_mat[0]);

  std::vector<float> ret(sp.getNumSteps(), 0.0f);
  launchFDTD3d(&mesh, &parameters, &ret[0],
               interruptCallbackLocal,
               progressCallbackLocal);

}

BOOST_AUTO_TEST_SUITE_END()
