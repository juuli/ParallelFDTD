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

#include <algorithm>
#include <Python.h>
#include <boost/cstdint.hpp>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "App.h"
#include "kernels/visualizationUtils.h"
#include "kernels/voxelizationUtils.h"

namespace GLOBAL {
  static FDTD::App* currentApp;
  static volatile bool interrupt = false;
}


using namespace boost::python;

boost::python::object toNumpyArray(unsigned int* data, long int size) {

    // create a PyObject * from pointer and data
      PyObject * pyObj = PyArray_SimpleNewFromData( 1, &size, NPY_UINT32, data );
      boost::python::handle<> handle( pyObj );
      boost::python::numeric::array arr( handle );

    /* The problem of returning arr is twofold: firstly the user can modify
      the data which will betray the const-correctness
      Secondly the lifetime of the data is managed by the C++ API and not the
      lifetime of the numpy array whatsoever. But we have a simple solution..
     */

       return arr.copy(); // copy the object. numpy owns the copy now.
}

boost::python::object toNumpyArray(float* data, long int size) {

    // create a PyObject * from pointer and data
      PyObject * pyObj = PyArray_SimpleNewFromData( 1, &size, NPY_FLOAT, data );
      boost::python::handle<> handle( pyObj );
      boost::python::numeric::array arr( handle );
      return arr.copy(); // copy the object. numpy owns the copy now.
}

boost::python::object toNumpyArray(double* data, long int size) {
    // create a PyObject * from pointer and data
      PyObject * pyObj = PyArray_SimpleNewFromData( 1, &size, NPY_DOUBLE, data );
      boost::python::handle<> handle( pyObj );
      boost::python::numeric::array arr( handle );
      return arr.copy(); // copy the object. numpy owns the copy now.
}

boost::python::object toNumpyArray(unsigned char* data, long int size) {

    // create a PyObject * from pointer and data
      PyObject * pyObj = PyArray_SimpleNewFromData( 1, &size, NPY_UINT8, data );
      boost::python::handle<> handle( pyObj );
      boost::python::numeric::array arr( handle );
      return arr.copy(); // copy the object. numpy owns the copy now.
}

void FDTD::App::initializeGeometryPy(boost::python::list indices,
                                     boost::python::list vertices) {
  int v_len = (int)boost::python::len(vertices);
  int i_len = (int)boost::python::len(indices);
  std::vector<float> std_vertices(v_len, 0.f);
  std::vector<unsigned int> std_indices(i_len, 0);

  for(int i = 0; i < v_len; i++)
    std_vertices.at(i) = boost::python::extract<float>(vertices[i]);

  for(int i = 0; i < i_len; i++)
    std_indices.at(i) = boost::python::extract<unsigned int>(indices[i]);

  this->m_geometry.initialize(std_indices, std_vertices);
  this->has_geometry_ = true;
}

boost::python::object FDTD::App::getVoxelization(boost::python::list indices,
                                                 boost::python::list vertices,
                                                 boost::python::list material_coefficients,
                                                 int voxelization_type,
                                                 float dx) {
  unsigned char* d_position_idx = (unsigned char*)NULL;
  unsigned char* d_material_idx = (unsigned char*)NULL;
  uint3 voxelization_dim = make_uint3(0,0,0);

  this->initializeGeometryPy(indices, vertices);

  unsigned int number_of_triangles = this->m_geometry.getNumberOfTriangles();

  this->addSurfaceMaterials(material_coefficients,
                            number_of_triangles,
                            20);

  if(voxelization_type == 0)
    voxelizeGeometry_solid(m_geometry.getVerticePtr(),
                           m_geometry.getIndexPtr(),
                           m_materials.getMaterialIdxPtr(),
                           m_geometry.getNumberOfTriangles(),
                           m_geometry.getNumberOfVertices(),
                           m_materials.getNumberOfUniqueMaterials(),
                           (double)dx,
                           &d_position_idx,
                           &d_material_idx,
                           &voxelization_dim);


  if(voxelization_type == 1) {
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
                                         0,voxelization_dim.z);
  }

  unsigned int num_elems = voxelization_dim.x*voxelization_dim.y*voxelization_dim.z;
  unsigned char* h_position_idx;
  unsigned char* h_material_idx;
  boost::python::list result;
  h_position_idx = fromDevice<unsigned char>(num_elems, d_position_idx, 0);
  h_material_idx = fromDevice<unsigned char>(num_elems, d_material_idx, 0);
  result.append(toNumpyArray(h_position_idx, num_elems));
  result.append(toNumpyArray(h_material_idx, num_elems));
  result.append(toNumpyArray((unsigned int*)&voxelization_dim, 3));
  free(h_position_idx);
  free(h_material_idx);
  destroyMem(d_position_idx);
  destroyMem(d_material_idx);
  return result;
}

void FDTD::App::initializeDomainPy(boost::python::numeric::array position_idx,
                                   boost::python::numeric::array material_idx,
                                   unsigned int dim_x,
                                   unsigned int dim_y,
                                   unsigned int dim_z) {
  log_msg<LOG_INFO>(L"App::initializeDomainPy - begin");
  mesh_size_t p_len = (mesh_size_t)boost::python::len(position_idx);
  mesh_size_t m_len = (mesh_size_t)boost::python::len(material_idx);


  if(p_len != dim_x*dim_y*dim_z || m_len !=dim_x*dim_y*dim_z) {
    log_msg<LOG_WARNING>(L"App::initializeGeometryPy, dimensions do not match"
                         L"with the size of the domain, x: %u y:%u z:%u, "
                         L"num pos: %u num mat: %u")
                         % dim_x % dim_y % dim_z % p_len % m_len;
   }
  clock_t start_t;
  clock_t end_t;
  start_t = clock();
  std::vector<unsigned char> std_pos(p_len, 0);
  unsigned char* ps = (unsigned char*)PyArray_GETPTR1(((PyArrayObject*)position_idx.ptr()), 0);

  for(mesh_size_t i = 0; i < p_len; i++)
    std_pos.at(i) = ps[i];

  end_t = clock()-start_t;
  log_msg<LOG_INFO>(L"App::initializeDomainPy - time copy pos: %f seconds")
          % ((float)end_t/CLOCKS_PER_SEC);

  start_t = clock();
  std::vector<unsigned char> std_mat(m_len, 0);
  unsigned char* ms = (unsigned char*)PyArray_GETPTR1(((PyArrayObject*)material_idx.ptr()), 0);

  for(mesh_size_t i = 0; i < m_len; i++)
    std_mat.at(i) = ms[i];

  end_t = clock()-start_t;
  log_msg<LOG_INFO>(L"App::initializeDomainPy - time copy mat: %f seconds")
            % ((float)end_t/CLOCKS_PER_SEC);

  this->initializeDomain(&std_pos[0], &std_mat[0], dim_x, dim_y, dim_z);
}

void FDTD::App::setSliceXY(boost::python::numeric::array slice,
                           unsigned int dim_x, unsigned int dim_y,
                           int slice_idx) {
  int s_len = (int)boost::python::len(slice);
  if(s_len != dim_x*dim_y) {
    std::cout<<"FDTD::App::setSliceXY: slice size does not match "
               "the given dimensions"<<std::endl;
    return;
  }

  if(this->m_mesh.isDouble()) {
    int dev_i = 0;
    mesh_size_t element = 0;
    this->m_mesh.getElementIdxAndDevice(0,0,slice_idx, &dev_i, &element);
    double* d_dest = this->m_mesh.getPressureDoublePtrAt(dev_i)+element;
    //double* h_src = (double*)slice.ptr()->data;
    double* h_src = (double*)PyArray_DATA((PyArrayObject*)slice.ptr());
    copyHostToDevice<double>(s_len, d_dest, h_src, dev_i);
  }
  else {
    int dev_i = 0;
    mesh_size_t element = 0;
    this->m_mesh.getElementIdxAndDevice(0,0,slice_idx, &dev_i, &element);
    //float* h_src = (float*)((PyArrayObject*)slice.ptr())->data;
    float* h_src = (float*)PyArray_DATA((PyArrayObject*)slice.ptr());
    float* d_dest = this->m_mesh.getPressurePtrAt(dev_i)+element;
    copyHostToDevice<float>(s_len, d_dest, h_src, dev_i);
  }
}

void FDTD::App::initializeSimulationPy() {
  GLOBAL::currentApp = this;

  if(this->has_geometry_)
    this->initializeMesh(2);

  unsigned int step = this->m_parameters.getNumSteps();
  this->responses_.assign(m_parameters.getNumSteps()*m_parameters.getNumReceivers(), 0.f);
}

void FDTD::App::setLayerIndicesPy(boost::python::list indices,
                                  std::string name) {
  int i_len = (int)boost::python::len(indices);
  std::vector<int> std_indices(i_len, 0.f);
  for(int i = 0; i < i_len; i++)
    std_indices.at(i) = boost::python::extract<int>(indices[i]);

  this->m_geometry.setLayerIndices(std_indices, name);
}

void FDTD::App::addSurfaceMaterials(boost::python::list material_coefficients,
                                    unsigned int number_of_surfaces,
                                    unsigned int number_of_coefficients) {
  int m_len = (int)boost::python::len(material_coefficients);
  std::vector<float> std_mat(m_len, 0.f);
  for(int i = 0; i < m_len; i++) {
    std_mat.at(i) = boost::python::extract<float>(material_coefficients[i]);
  }
  this->m_materials.addMaterials(&std_mat[0],
                                 number_of_surfaces,
                                 number_of_coefficients);
}

void FDTD::App::addSourceDataFloat(boost::python::list src_data,
                                   int num_steps,
                                   int num_sources) {
  std::vector<float> std_src_data(num_steps*num_sources, 0.0);
  for(int j = 0; j < num_sources; j++) {
    for(int i = 0; i < num_steps; i++) {
      int idx =j*num_steps+i;
      std_src_data.at(idx) = boost::python::extract<float>(src_data[idx]);
    }
  }
  this->m_parameters.addInputData(std_src_data);
}

void FDTD::App::addSourceDataDouble(boost::python::list src_data,
                              int num_steps,
                              int num_sources) {
  std::vector<double> std_src_data(num_steps*num_sources, 0.0);
  for(int j = 0; j < num_sources; j++) {
    for(int i = 0; i < num_steps; i++) {
      int idx =j*num_steps+i;
      std_src_data.at(idx) = boost::python::extract<double>(src_data[idx]);
    }
  }
  this->m_parameters.addInputDataDouble(std_src_data);
}

boost::python::object FDTD::App::getSlice(int orientation,
                                          int slice_idx) {
  float* pressure_data = NULL;
  unsigned char* position_data = NULL;
  unsigned int d_x = 0;
  unsigned int d_y = 0;
  if(orientation == 0) {d_x = this->m_mesh.getDimX(); d_y = this->m_mesh.getDimY();};
  if(orientation == 1) {d_x = this->m_mesh.getDimX(); d_y = this->m_mesh.getDimZ();};
  if(orientation == 2) {d_x = this->m_mesh.getDimY(); d_y = this->m_mesh.getDimZ();};

  captureCurrentSlice(&this->m_mesh, &pressure_data, &position_data,
                      slice_idx, orientation);

  boost::python::object ret = toNumpyArray(pressure_data, d_x*d_y);
  free(pressure_data);
  free(position_data);
  return ret;
}

boost::python::list FDTD::App::getMeshDimensionsPy() {
  boost::python::list result;
  mesh_size_t dim_x = this->m_mesh.getDimX();
  mesh_size_t dim_y = this->m_mesh.getDimY();
  mesh_size_t dim_z = this->m_mesh.getDimZ();
  result.append(dim_x);
  result.append(dim_y);
  result.append(dim_z);
  return result;
}

/// Following convertion code taken from:
/// http://stackoverflow.com/questions/26595350/extracting-unsigned-char-from-array-of-numpy-uint8
/// @brief Converter type that enables automatic conversions between NumPy
///        scalars and C++ types.
template <typename T, NPY_TYPES NumPyScalarType>
struct enable_numpy_scalar_converter
{
  enable_numpy_scalar_converter() {
    // Required NumPy call in order to use the NumPy C API within another
    // extension module.
    import_array();

    boost::python::converter::registry::push_back(
      &convertible,
      &construct,
      boost::python::type_id<T>());
  }

  static void* convertible(PyObject* object) {
    // The object is convertible if all of the following are true:
    // - is a valid object.
    // - is a numpy array scalar.
    // - its descriptor type matches the type for this converter.
    return (
      object &&                                                    // Valid
      PyArray_CheckScalar(object) &&                               // Scalar
      PyArray_DescrFromScalar(object)->type_num == NumPyScalarType // Match
    )
      ? object // The Python object can be converted.
      : NULL;
  }

  static void construct(
    PyObject* object,
    boost::python::converter::rvalue_from_python_stage1_data* data) {
    // Obtain a handle to the memory block that the converter has allocated
    // for the C++ type.
    namespace python = boost::python;
    typedef python::converter::rvalue_from_python_storage<T> storage_type;
    void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

    // Extract the array scalar type directly into the storage.
    PyArray_ScalarAsCtype(object, storage);

    // Set convertible to indicate success.
    data->convertible = storage;
  }
};

BOOST_PYTHON_MODULE(libPyFDTD) {
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

  enable_numpy_scalar_converter<boost::uint8_t, NPY_UBYTE>();

  class_< std::vector<float> >("std_vec_float")
    .def(vector_indexing_suite< std::vector<float> >() )
    ;

  class_< std::vector<double> >("std_vec_double")
    .def(vector_indexing_suite< std::vector<double> >() )
    ;

  class_<FDTD::App>("App")
    .def("initializeDevices", &FDTD::App::initializeDevices)
    .def("initializeGeometryFromFile", &FDTD::App::initializeGeometryFromFile)
    .def("initializeGeometryPy", &FDTD::App::initializeGeometryPy)
    .def("initializeSimulationPy", &FDTD::App::initializeSimulationPy)
    .def("initializeDomainPy", &FDTD::App::initializeDomainPy)
    .def("setLayerIndices", &FDTD::App::setLayerIndicesPy)
    .def("setVoxelizationType", &FDTD::App::setVoxelizationType)
    .def("setDouble", &FDTD::App::setDouble)
    .def("setCapturedB", &FDTD::App::setCapturedB)
    .def("setCaptureId", &FDTD::App::setCaptureId)
    .def("setSpatialFs", &FDTD::App::setSpatialFs)
    .def("setNumSteps", &FDTD::App::setNumSteps )
    .def("setUpdateType", &FDTD::App::setUpdateType)
    .def("setUniformMaterial", &FDTD::App::setUniformMaterial)
    .def("setSliceXY", &FDTD::App::setSliceXY)
    .def("addSource", &FDTD::App::addSource)
    .def("addSourceDataFloat", &FDTD::App::addSourceDataFloat)
    .def("addSourceDataDouble", &FDTD::App::addSourceDataDouble)
    .def("addReceiver", &FDTD::App::addReceiver)
    .def("addSurfaceMaterials", &FDTD::App::addSurfaceMaterials)
    .def("addSliceToCapture", &FDTD::App::addSliceToCapture)
    .def("getSlice", &FDTD::App::getSlice)
    .def("getResponse", &FDTD::App::getResponse)
    .def("getResponseDouble", &FDTD::App::getResponseDouble)
    .def("getMvox", &FDTD::App::getMvoxPerSec)
    .def("getNumElems", &FDTD::App::getNumElements)
    .def("getMeshDimensions", &FDTD::App::getMeshDimensionsPy)
    .def("getEyring", &FDTD::App::getEyring)
    .def("getSabine", &FDTD::App::getSabine)
    .def("getVolume", &FDTD::App::getVolume)
    .def("getVoxelization", &FDTD::App::getVoxelization)
    .def("forcePartitionTo", &FDTD::App::setForcePartitionTo)
    .def("runSimulation", &FDTD::App::runSimulation)
    .def("runCapture", &FDTD::App::runCapture)
    .def("executeStep", &FDTD::App::executeStep)
    .def("close", &FDTD::App::close)
    #ifdef COMPILE_VISUALIZATION
    .def("runVisualization", &FDTD::App::runVisualization)
    #endif
    ;
}
