#ifndef SIMPARAMETERS_H
#define SIMPARAMETERS_H

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

/*
Parameters class

Handels the simulation parameters, materials and source/receiver definition
Inside App class
*/

#include "../kernels/cudaUtils.h"
#include "../math/geomMath.h"
#include "SrcRec.h"
#include <string>
#include <vector>

enum UpdateType {SRL_FORWARD, SHARED, SRL};
enum VoxelizationType {SOLID, SURFACE, SURFACE_6};

class SimulationParameters {
public:
  SimulationParameters()
  :
    update_type_(SRL_FORWARD),
    voxelization_type_(SOLID),
    c_(344.f),
    lambda_(((double)1/sqrt((double)3))),
    octave_(0),
    num_steps_(1),
    spatial_fs_(7000),
    bounding_box_min_(nv::Vec3f(0.f, 0.f, 0.f)),
    bounding_box_max_(nv::Vec3f(0.f, 0.f, 0.f)),
    add_padding_to_element_idx_(true),
    sources_(),
    receivers_(),
    source_input_data_(),
    parameter_vec_(),
    parameter_vec_double_(),
    grid_ir_()
  {};

  ~SimulationParameters() {};


private:
  UpdateType update_type_;
  VoxelizationType voxelization_type_;
  float c_;                       ///< Speed of sound
  double lambda_;                 ///< Courant number used in the simulation
  unsigned int octave_;           ///< Ocateve band to simulate
  unsigned int num_steps_;        ///< Number of steps to simulate
  unsigned int spatial_fs_;       ///< Spatial sampling frequency

  nv::Vec3f bounding_box_min_;    ///< Bounding box min coordintae
  nv::Vec3f bounding_box_max_;    ///< Bounding box max coordinate

  // add +1 to element indexes to get indexes inside the geometry
  bool add_padding_to_element_idx_;

  std::vector<Source> sources_;                           ///< List of sources
  std::vector<Receiver> receivers_;                       ///< List of receivers
  std::vector< std::vector<float> > source_input_data_;   ///< input data for each source
  std::vector< std::vector<double> > source_input_data_double_;   ///< input data for each source
  std::vector< std::vector<float> > source_output_data_;  ///<
  std::vector<float*> d_source_output_data_;              ///< Input data for each source
                                                          /// on the device

  // "Pointers" are saved here so they are convenient to clean up
  // These are used when interfacing with cuda code
  std::vector<float> parameter_vec_;
  std::vector<double> parameter_vec_double_;
  std::vector<float> grid_ir_;

  /// \brief Get a raw source sample
  /// \param source_idx source index
  /// \param step the time instance of the sample
  float getRegularSourceSample(unsigned int source_idx, unsigned int step);
  double getRegularSourceSampleDouble(unsigned int source_idx, unsigned int step);
  /// \brief Get a transparent source sample; a regular source sample convolved
  /// the impulse response of the mesh
  /// \param source_idx source index
  /// \param step the
  float getTransparentSourceSample(unsigned int source_idx, unsigned int step);
  double getTransparentSourceSampleDouble(unsigned int source_idx, unsigned int step);
public:
  // Setters
  void readGridIr(std::string ir_fp);
  void setUpdateType(enum UpdateType update_type);
  void setVoxelizationType(enum VoxelizationType voxelization_type);
  void setC(float c) {this->c_ = c;}
  void setLambda(double lambda) {this->lambda_ = lambda;}
  void setOctave(unsigned int octave) {this->octave_ = octave;}
  void setNumSteps(unsigned int num_steps) {this->num_steps_ = num_steps;}
  void setSpatialFs(unsigned int spatial_fs) {this->spatial_fs_ = spatial_fs;}

  void setBoundingBox(nv::Vec3f bounding_box_min, nv::Vec3f bounding_box_max)
    {bounding_box_min_ = bounding_box_min; bounding_box_max_ = bounding_box_max;}

  void setAddPaddingToElementIdx(bool add_padding_to_element_idx)
    {this->add_padding_to_element_idx_ = add_padding_to_element_idx;}

  // Getters
  enum UpdateType getUpdateType() const {return this->update_type_;}
  enum VoxelizationType getVoxelizationType() const {return this->voxelization_type_;}
  float getC() const {return this->c_;};
  double getLambda() const {return this->lambda_;}
  float getDx() const;

  unsigned int getOctave() const {return this->octave_;}
  unsigned int getNumSteps() const {return this->num_steps_;}
  unsigned int getSpatialFs() const {return this->spatial_fs_;}
  // Returns the step number at the given time at current spatial sampling rate
  unsigned int getStepAtTime(float t) {return (unsigned int)(this->spatial_fs_*t);}

  // Source/Receiver functions
  void addSource(float x, float y, float z);
  void addSource(Source src);
  void addSource_no_logging(Source src);
  void addReceiver(float x, float y, float z) {receivers_.push_back(Receiver(x,y,z));};
  void addReceiver(Receiver rec) {receivers_.push_back(rec);};
  // Add device pointer which contain the source input data
  void addSourceDData(float* d_vector);
  void removeSource(unsigned int i);
  void removeReceiver(unsigned int i);
  void updateSourceAt(unsigned int i, Source src);
  void updateReceiverAt(unsigned int i, Receiver rec);
  void updateInputDataAt(unsigned int i, std::vector<float> data);
  void updateInputDataDoubleAt(unsigned int i, std::vector<double> data);
  void resetSourcesAndReceivers();
  void resetReceivers();
  void resetSources();
  void resetInputData(); // both single and double!

  void addInputData(float* data, unsigned int number_of_samples);
  void addInputData(std::vector<float> data) {this->source_input_data_.push_back(data);}
  void addInputDataDouble(std::vector<double> data) {this->source_input_data_double_.push_back(data);}


  float getSourceSample(unsigned int source_idx, unsigned int step);
  double getSourceSampleDouble(unsigned int source_idx, unsigned int step);
  float* getSourceDData(unsigned int source_idx) {return this->d_source_output_data_.at(source_idx);};
  float getInputDataSample(unsigned int idx, unsigned int sample);
  double getInputDataSampleDouble(unsigned int idx, unsigned int sample);
  float getGridIrDataSample(unsigned int sample);

  nv::Vec3i getSourceElementCoordinates(unsigned int source_idx);
  nv::Vec3i getReceiverElementCoordinates(unsigned int receiver_idx);

  unsigned int getSourceElementIdx(unsigned int source_idx,
                                   unsigned int dim_x,
                                   unsigned int dim_y);

  unsigned int getReceiverElementIdx(unsigned int receiver_idx,
                                     unsigned int dim_x,
                                     unsigned int dim_y);

  unsigned int getNumSources() const {return (unsigned int)sources_.size();};
  unsigned int getNumReceivers() const {return (unsigned int)receivers_.size();};
  float* getSourceVectorAt(unsigned int source_idx);

  Source getSource(unsigned int i) const {return sources_.at(i);};
  Receiver getReceiver(unsigned int i) const {return receivers_.at(i);};

  float* getParameterPtr();
  double* getParameterPtrDouble();
};

#endif
