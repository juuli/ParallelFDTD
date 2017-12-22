#ifndef APP_H
#define APP_H
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

///////////////////////////////////////////////////////////////////////////////
/// \mainpage ParallelFDTD documentation
///
/// \author Jukka Saarelma
/// \version 0.1
/// \date 16.12.2013
///
///
/// \section mainpage_intro Introduction
///
///////////////////////////////////////////////////////////////////////////////

#include "io/Image.h"
#include "base/SimulationParameters.h"
#include "base/MaterialHandler.h"
#include "base/GeometryHandler.h"
#include "io/FileReader.h"
#include "kernels/cudaMesh.h"

#ifdef COMPILE_VISUALIZATION
#include "gl/AppWindow.h"
#endif

typedef bool (*InterruptCallback)(void);
typedef void (*ProgressCallback)(int, int, float);

// forward declaration
namespace boost {
  namespace python {
    class list;
    class object;
    namespace numeric {
      class array;
    }
  }
}

namespace FDTD {
class App {
public:
  /// Default constructor
  App()
  : m_file_reader(FileReader()),
    m_parameters(SimulationParameters()),
    m_geometry(GeometryHandler()),
    m_materials(MaterialHandler()),
    number_of_devices_(0),
    best_device_(0),
    force_partition_to_(-1),
    capture_db_(60),
    interrupt_(false),
    has_geometry_(true),
    time_per_step_(0.f),
    num_elements_(0),
    current_step_(0),
    step_direction_(1),
    capture_id_("")
  {loggerInit();
   this->setupDefaultCallbacks();
   #ifdef COMPILE_VISUALIZATION
    this->m_window = NULL;
   #endif
 };


  /// Default destructor
  /// Destroy dynamically allocated memory
  ~App() {
    std::vector<float*>::iterator it;
    for(it = this->mesh_captures_.begin();
      it!=this->mesh_captures_.end();
      it++)
      free((*it));
  };

  #ifdef COMPILE_VISUALIZATION
  AppWindow* m_window;
  #endif

  FileReader m_file_reader;
  SimulationParameters m_parameters;
  GeometryHandler m_geometry;
  MaterialHandler m_materials;
  CudaMesh m_mesh;
  InterruptCallback m_interrupt;
  ProgressCallback m_progress;


  void queryDevices();
  void resetDevices();
  void initializeDevices();
  void initializeMaterials(std::string material_fp);
  void initializeGeometryFromFile(std::string geometry_fp);

  ///////////////////////////////////////////////////////////////////////////////
  /// Initializes the geometry from basic datatypes
  ///
  /// \param[in] indices Pointer to the first member of a list of triangle indices.
  ///              Indices refer to the vertex list
  /// \param[in] vertices
  /// \param[in] number_of_indices the length of the indeces array
  /// \param[in] number_of_vertices length of the vertices array
  ///////////////////////////////////////////////////////////////////////////////
  void initializeGeometry(unsigned int* indices, float* vertices,
                          unsigned int number_of_indices,
                          unsigned int number_of_vertices);


  bool hasGeometry() {return this->has_geometry_;}

  ///////////////////////////////////////////////////////////////////////////////
  /// Initializes the domain directily without voxelizer
  ///
  /// \param[in] position_idx_ptr The position indices of each voxel
  /// \param[in] material_idx_ptr The material indices for each voxel
  /// \param[in] dim_x x dimension of the domain
  /// \param[in] dim_y y dimension of the domain
  /// \param[in] dim_z z dimension of the domain
  ///////////////////////////////////////////////////////////////////////////////
  void initializeDomain(unsigned char* position_idx_ptr,
                        unsigned char* material_idx_ptr,
                        unsigned int dim_x, unsigned int dim_y,
                        unsigned int dim_z);


  void initializeSimulationPy();

   ///////////////////////////////////////////////////////////////////////////////
  /// Setup default callback for both progress and intrupt of the solver
  /// Default innterrupt ignores signals, default progress callback prints
  /// the progress of the solver every 100th step to std::cout
  void setupDefaultCallbacks();

  ///////////////////////////////////////////////////////////////////////////////
  /// Initializes the CudaMesh m_mesh field of the app class
  /// \param[in] number_of_partitions On how many devices the mesh is divided
  ///////////////////////////////////////////////////////////////////////////////
  void initializeMesh(unsigned int number_of_partitions);

  ///////////////////////////////////////////////////////////////////////////////
  /// Initializes the CudaMesh m_mesh field of the app class with a host
  /// voxelization. This is desireable when many GPUs are available, and
  /// the initial voxelization domain is too large for a single GPUs memory
  ///////////////////////////////////////////////////////////////////////////////
  void initializeMeshHost();

  ///////////////////////////////////////////////////////////////////////////////
  /// Initializes an Open GL window for visualization
  /// \param[in] arc number of command line arguments
  /// \param[in] arv command line arguments
  ///////////////////////////////////////////////////////////////////////////////
  void initializeWindow(int argc, char** argv);

  // run the application

  ///////////////////////////////////////////////////////////////////////////////
  /// Run the simulation without visualization and without captures.
  /// Simulation is run for predetermined number of steps. No captures are made
  /// even if assigned
  ///////////////////////////////////////////////////////////////////////////////
  void runSimulation();

  ///////////////////////////////////////////////////////////////////////////////
  /// Run the simulation using the OpenGl visualization scheme.
  /// Function initializes the needed data structures for the simulation and
  /// run the simulation step by step basis. Captures are made if assigned.
  ///////////////////////////////////////////////////////////////////////////////
  void runVisualization();

  ///////////////////////////////////////////////////////////////////////////
  /// Run the simulation without visualization step by step. Captures are made
  /// if assigned
  ///////////////////////////////////////////////////////////////////////////
  void runCapture();

  ///////////////////////////////////////////////////////////////////////////
  /// \brief Cleanup function for App clas
  ///////////////////////////////////////////////////////////////////////////
  void close();

  ///////////////////////////////////////////////////////////////////////////
  /// \brief Set the dynamic range of captured image. The dB value under wich
  /// pressure values are cut off under the level of 0 dB and shown
  /// black in the captured
  ///////////////////////////////////////////////////////////////////////////
  void setCapturedB(float db) {this->capture_db_ = db;}


  ////////// Runtime methods

  ///////////////////////////////////////////////////////////////////////////////
  /// Executes a single FDTD step
  ///
  /// \param[in,out] h_return_ptr data form the receiver positions
  ///////////////////////////////////////////////////////////////////////////////
  void executeStep();

  ///////////////////////////////////////////////////////////////////////////////
  /// Update the OpenGL visualization using the data of the this->m_mesh
  ///
  /// \param[in] current_slice the slice index which is to be visualized
  /// \param[in] orientation Orientation of the slice which is updated. 0 = xy,
  ///  1 = xz, 2 = yz
  /// \param[in] selector Which data of CudaMesh is visualized. 0 = pressure, 1 =
  /// position idx, 2 = inside outside switch (white: in, black: out)
  /// \param[in] dB the used dynamic range of the visualization in decibels
  ///////////////////////////////////////////////////////////////////////////////
  void updateVisualization(unsigned int current_z,
                           unsigned int orientation,
                           unsigned int selector, float dB);

  ///////////////////////////////////////////////////////////////////////////
  /// Reset the pressure mesh of CudaMesh class. Sets the current step to 0
  ///////////////////////////////////////////////////////////////////////////
  void resetPressureMesh();

  ///////////////////////////////////////////////////////////////////////////
  /// Flips the current and past pressure pointers.
  ///////////////////////////////////////////////////////////////////////////
  void invertTime();

  ///////////////////////////////////////////////////////////////////////////
  /// Returns the time taken for a single step
  ///////////////////////////////////////////////////////////////////////////
  float getTimePerStep() {return this->time_per_step_;}

  ///////////////////////////////////////////////////////////////////////////
  /// Get number of elements in the simulation domain
  ///////////////////////////////////////////////////////////////////////////
  unsigned int getNumElements() {return this->num_elements_;}

  ///////////////////////////////////////////////////////////////////////////
  /// Get the number of samples in the response
  ///////////////////////////////////////////////////////////////////////////
  unsigned int getResponseSize() {
    if(this->m_mesh.isDouble())
      return (unsigned int)this->responses_double_.size();
    else
      return (unsigned int)this->responses_.size();
  }

  ///////////////////////////////////////////////////////////////////////////
  /// Return the performance of the run in Megavoxels / s
  ///////////////////////////////////////////////////////////////////////////
  float getMvoxPerSec() {return (float)((1.f/this->time_per_step_*this->m_mesh.getNumberOfElements())/1e6);}

  ///////////////////////////////////////////////////////////////////////////
  /// \brief Returns a pointer to the beginning of the response data
  /// \return pointer to first index of the response vector
  ///////////////////////////////////////////////////////////////////////////
  float* getResponsePointer() {return &(this->responses_[0]);}

  ///////////////////////////////////////////////////////////////////////////
  /// \return A single precision pressure sample of receiver rec at time index step
  ///////////////////////////////////////////////////////////////////////////
  float getResponseSampleAt(unsigned int step, unsigned int rec) {
    return this->responses_.at(this->m_parameters.getNumSteps()*rec+step);}

  ///////////////////////////////////////////////////////////////////////////
  /// \return A double precision pressure sample of receiver rec at time index step.
  ///////////////////////////////////////////////////////////////////////////
  double getResponseDoubleSampleAt(unsigned int step, unsigned int rec) {
    return this->responses_double_.at(this->m_parameters.getNumSteps()*rec+step);
  }

  ///////////////////////////////////////////////////////////////////////////
  /// \return A pointer to a mesh capture at index i
  ///////////////////////////////////////////////////////////////////////////
  float* getMeshCaptureAt(unsigned int i) {return this->mesh_captures_.at(i);}

  ///////////////////////////////////////////////////////////////////////////
  /// \return The number of mesh captures done during the simulation
  ///////////////////////////////////////////////////////////////////////////
  unsigned int getNumberOfMeshCaptures() {return (unsigned int)this->mesh_captures_.size();
  }

  ///////////////////////////////////////////////////////////////////////////
  /// \return Add a slice to capture during the simulation. Captures are
  /// utilized when running the simulation with runCapture() or
  ///  runVisualization() functions
  ///////////////////////////////////////////////////////////////////////////
  void addSliceToCapture(unsigned int slice, unsigned int step, unsigned int orientation) {
    this->step_to_capture_.push_back(step);
    this->slice_to_capture_.push_back(slice);
    this->slice_orientation_.push_back(orientation);
  }

  ///////////////////////////////////////////////////////////////////////////
  /// add a time step when a mesh is captured and saved in mesh_captures_
  ///////////////////////////////////////////////////////////////////////////
  void addMeshToCapture(unsigned int step) { this->mesh_to_capture_.push_back(step);}

  ///////////////////////////////////////////////////////////////////////////////
  /// Save pressure data to a bitmap file from a slice of mesh
  /// \param[in] data Pressure data sized dim_x*dim_y
  /// \param[in] position_data Position index data indicating the boundaries
  ///             of the geometry
  /// \params[in] dim_x the x dimension of the mesh slice
  /// \params[in] dim_y the y dimension of the mesh slice
  /// \params[in] slice the slice index which is captured
  /// \params[in] orientation the orientation of the slice 0: xy, 1: xz, 2: yz
  /// \params[in] step The number of the step of the capture
  ///////////////////////////////////////////////////////////////////////////////
  void saveBitmap(float* data, unsigned char* position_data,
                  unsigned int dim_x,
                  unsigned int dim_y,
                  unsigned int slice,
                  unsigned int orientation,
                  unsigned int step);

  ///////////////////////////////////////////////////////////////////////////
  /// Calculates the volume of the model using the voxelization
  ///////////////////////////////////////////////////////////////////////////
  float getVolume();

  ///////////////////////////////////////////////////////////////////////////////
  /// Calculates the total absoption are of the model at a given octave band
  ///
  /// \param[in] octave octave band 0-9 in which the absorption area is calculated
  ///////////////////////////////////////////////////////////////////////////////
  float getTotalAborptionArea(unsigned int octave);

  ///////////////////////////////////////////////////////////////////////////////
  /// Calculates the theoretical reverberation time of the model using the
  ///  Sabines formula
  ///
  /// \param[in] octave octave band 0-9 in which the reverberation time is
  /// calculated
  ///////////////////////////////////////////////////////////////////////////////
  float getSabine(unsigned int octave);

  ///////////////////////////////////////////////////////////////////////////////
  /// Calculates the theoretical reverberation time of the model using the
  ///  Eyrings formula
  ///
  /// \param[in] octave octave band 0-9 in which the absorption area is calculated
  ///////////////////////////////////////////////////////////////////////////////
  float getEyring(unsigned int octave);

  unsigned int current_step_;                  ///< Current step of the simulation
  int step_direction_;                        ///< Direction of step 1 forward, -1 backward

private:
  std::vector<unsigned int> step_to_capture_;   ///< Current step of the simulation
  std::vector<unsigned int> slice_to_capture_;  ///< Current step of the simulation
  std::vector<unsigned int> slice_orientation_;  ///< Current step of the simulation
  std::vector<unsigned int> mesh_to_capture_;    ///< Current step of the simulation

  int number_of_devices_;                     ///< Number of devices available
  int best_device_;                            ///< Device index of the most suitable device
  std::vector<int> device_mem_sizes_;         ///< Amount of memory in MB in the available devices
  int force_partition_to_;                    ///< Force the solver to use specific number of partitions
  float capture_db_;                          ///< The dynamic range of the captured image
  bool interrupt_;                            ///< Indicating if interrupt has been called
  bool has_geometry_;                          ///< Indicating that domain is directly
                                               ///  initialized without voxelization
  // Result variables
  std::vector< float > responses_;            ///< Response values at receivers
  std::vector< double > responses_double_;    ///< Response values when using double precision
  std::vector< float* > slice_captures_;      ///< Slice captures, NOT IMPLEMENTED
  std::vector< float* > mesh_captures_;        ///< Captures of the whole mesh
  std::string capture_id_;                     ///< An Id that is added to the captured image file name
  // Return values to Matlab
  float time_per_step_;                        ///< Average time taken for a simulation step
  mesh_size_t num_elements_;                  ///< Number of elements


public:
  ///////////////////////////////////////////////////////////////////////////////
  // Functions used to bind the FDTD::App class with python via Boost Python

  void initializeGeometryPy(boost::python::list indices,
                            boost::python::list vertices);

  void initializeDomainPy(boost::python::numeric::array position_idx,
                          boost::python::numeric::array material_idx,
                          unsigned int dim_x,
                          unsigned int dim_y,
                          unsigned int dim_z);

  void setLayerIndicesPy(boost::python::list indices,
                         std::string name);

  void setSliceXY(boost::python::numeric::array slice,
                  unsigned int dim_x, unsigned int dim_y,
                  int slice_idx);

  void addSource(float x, float y, float z, int type, int signal, int input_signal_idx) {
    this->m_parameters.addSource(Source(x, y, z, (enum SrcType)type, (enum  InputType)signal, input_signal_idx));
  };

  void addReceiver(float x, float y, float z) {
    this->m_parameters.addReceiver(x, y, z);
  };

  void setVoxelizationType(unsigned int type) {this->m_parameters.setVoxelizationType((enum VoxelizationType)type);}
  void setSpatialFs(unsigned int fs) {this->m_parameters.setSpatialFs(fs);};
  void setNumSteps(unsigned int num) {this->m_parameters.setNumSteps(num);};
  void setUpdateType(int i) {this->m_parameters.setUpdateType((enum UpdateType)i);};
  void setUniformMaterial(float R) {
    this->m_materials.setGlobalMaterial(this->m_geometry.getNumberOfTriangles(),
                                        reflection2Admitance(R));
  }

  std::vector<float> getResponse(unsigned int rec) {
    std::vector<float> ret(this->m_parameters.getNumSteps(), 0.f);
    for(unsigned int i = 0; i < this->m_parameters.getNumSteps(); i++) {
      ret.at(i) = this->getResponseSampleAt(i, rec);
    }
    return ret;
  }

  std::vector<double> getResponseDouble(unsigned int rec) {
    std::vector<double> ret(this->m_parameters.getNumSteps(), 0.f);
    for(unsigned int i = 0; i < this->m_parameters.getNumSteps(); i++) {
      ret.at(i) = this->getResponseDoubleSampleAt(i, rec);
    }
    return ret;
  }

  void setCaptureId(std::string id) {this->capture_id_ = id;}

  void setDouble(bool set_to) {this->m_mesh.setDouble(set_to);}

  void setForcePartitionTo(int num_partitions) {this->force_partition_to_ = num_partitions;}

  void addSurfaceMaterials(boost::python::list material_coefficients,
                           unsigned int number_of_surfaces,
                           unsigned int number_of_coefficients);

  int addMaterialToMaterialList(boost::python::list coeffs);

  void addSourceDataFloat(boost::python::list src_data, int num_steps, int num_sources);

  void addSourceDataDouble(boost::python::list src_data, int num_steps, int num_sources);

  boost::python::object getSlice(int orientation, int slice_idx);

  boost::python::list getMeshDimensionsPy();

  boost::python::object getVoxelization(boost::python::list indices,
                                        boost::python::list vertices,
                                        boost::python::list material_indices,
                                        int voxelization_type,
                                        float dx);

};
}

#endif
