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

#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "App.h"

using namespace boost::python;

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
  for(int i = 0; i < m_len; i++)
    std_mat.at(i) = boost::python::extract<float>(material_coefficients[i]);

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


BOOST_PYTHON_MODULE(libPyFDTD) {

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
    .def("setLayerIndices", &FDTD::App::setLayerIndicesPy)
    .def("addSource", &FDTD::App::addSource)
    .def("addSourceDataFloat", &FDTD::App::addSourceDataFloat)
    .def("addSourceDataDouble", &FDTD::App::addSourceDataDouble)
    .def("addReceiver", &FDTD::App::addReceiver)
    .def("addSurfaceMaterials", &FDTD::App::addSurfaceMaterials)
    .def("setSpatialFs", &FDTD::App::setSpatialFs)
    .def("setNumSteps", &FDTD::App::setNumSteps )
    .def("setUpdateType", &FDTD::App::setUpdateType)
    .def("setUniform", &FDTD::App::setUniformMaterial)
    .def("runVisualization", &FDTD::App::runVisualization)
    .def("runSimulation", &FDTD::App::runSimulation)
    .def("runCapture", &FDTD::App::runCapture)
    .def("setUniformMaterial", &FDTD::App::setUniformMaterial)
    .def("getResponse", &FDTD::App::getResponse)
    .def("getResponseDouble", &FDTD::App::getResponseDouble)
    .def("forcePartitionTo", &FDTD::App::setForcePartitionTo)
    .def("addSliceToCapture", &FDTD::App::addSliceToCapture)
    .def("setDouble", &FDTD::App::setDouble)
    .def("setCapturedB", &FDTD::App::setCapturedB)
    .def("close", &FDTD::App::close)
    .def("getMvox", &FDTD::App::getMvoxPerSec)
    .def("getNumElems", &FDTD::App::getNumElements)
    ;
}
