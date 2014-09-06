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

#include "../global_includes.h"
#include "../io/FileReader.h"
#include "SimulationParameters.h"
#include <stdexcept>
#include <iostream>


void SimulationParameters::readGridIr(std::string ir_fp) {
  FileReader fr;
  this->grid_ir_ = fr.readFloat(ir_fp);
}

void SimulationParameters::addSource(float x, float y, float z) {
    Source new_source = Source(x,y,z);
    nv::Vec3i element_idx = new_source.getElementIdx(this->getSpatialFs(), this->getC(), (float)this->getLambda());
    log_msg<LOG_DEBUG>(L"SimulationParameters::addSource - Source added, element idx x: %u y: %u z: %u")
                       %element_idx.x % element_idx.y %element_idx.z;

    sources_.push_back(new_source);
}

void SimulationParameters::addSource(Source src) {
    nv::Vec3i element_idx = src.getElementIdx(this->getSpatialFs(), this->getC(), (float)this->getLambda());
    log_msg<LOG_DEBUG>(L"SimulationParameters::addSource - Source added, element idx x: %u y: %u z: %u")
                       %element_idx.x % element_idx.y %element_idx.z;
    sources_.push_back(src);
}

void SimulationParameters::updateSourceAt(unsigned int i, Source src) {
  unsigned int vector_size = (unsigned int)this->sources_.size();
  nv::Vec3f p = src.getP();
  log_msg<LOG_INFO>(L"SimulationParameters::updateSourceAt - adding source x: %f.1 y: %f.1 z:%f.1 to index : %d") % p.x %p.y %p.z %i;

  if(i >= vector_size)
    log_msg<LOG_ERROR>(L"SimulationParameters:updateSourceAt : index %d out of range : %d") %i %(vector_size-1);

  this->sources_.at(i) = src;
}

void SimulationParameters::updateReceiverAt(unsigned int i, Receiver rec) {
  unsigned int vector_size = (unsigned int)this->sources_.size();

  if(i >= vector_size)
    log_msg<LOG_ERROR>(L"SimulationParameters:updateReceiverAt : index %d out of range : %d") %i %(vector_size-1);


  this->receivers_.at(i) = rec;
}

void SimulationParameters::removeSource(unsigned int i) {
  unsigned int vector_size = (unsigned int)this->sources_.size();

  if(i >= vector_size) {
    log_msg<LOG_ERROR>(L"SimulationParameters::removeSource : index %d out of range : %d") %i %(vector_size-1);
    throw std::out_of_range("SimulationParameters::removeSource: idx out of range");
  }

  this->sources_.erase(sources_.begin()+i);
}

void SimulationParameters::removeReceiver(unsigned int i) {
  unsigned int vector_size = (unsigned int)this->receivers_.size();

  if(i >= vector_size) {
    log_msg<LOG_ERROR>(L"SimulationParameters::removeReceiver : index %d out of range : %d") %i %(vector_size-1);
    throw std::out_of_range("SimulationParameters::removeReceiver: idx out of range");
  }

  this->receivers_.erase(receivers_.begin()+i);
}

void SimulationParameters::resetSourcesAndReceivers() {
  this->receivers_.clear();
  this->sources_.clear();
}

void SimulationParameters::addInputData(float* data, unsigned int number_of_samples) {
  if(!data) {
    log_msg<LOG_WARNING>(L"SimulationParameters::addInputData : invalid input data (NULL)");  
  }

  std::vector<float> new_data;
  new_data.assign(number_of_samples, 0.f);
  
  if(data){ 
    for(unsigned int i = 0; i < number_of_samples; i ++) {
      if(data+i)
        new_data.at(i) = data[i];
    }
  }

  this->source_input_data_.push_back(new_data);
}

// Add device pointer which contain the source data
void SimulationParameters::addSourceDData(float* d_vector)  {
  if(this->d_source_output_data_.size() < this->getNumSources())
    d_source_output_data_.push_back(d_vector);
}

void SimulationParameters::setUpdateType(enum UpdateType update_type) {
  this->update_type_ = update_type;
  if(update_type == SRL) {
    log_msg<LOG_INFO>(L"SimulationParameters::setUpdateType - update type Standard Leapfrog Kowalczyk");
    this->lambda_ = sqrt((double)1/3);
  }
  if(update_type == SRL_FORWARD) {
    log_msg<LOG_INFO>(L"SimulationParameters::setUpdateType - update type Standard Leapfrog Bilbao");
    this->lambda_ = sqrt((double)1/3);
  }
  if(update_type == SHARED) {
    log_msg<LOG_INFO>(L"SimulationParameters::setUpdateType - update type SHARED");
    this->lambda_ = sqrt((double)1/3);
  }
}

float SimulationParameters::getSourceSample(unsigned int source_idx, unsigned int step) {
  float sample = 0.f;

  if(this->getSource(source_idx).getSourceType() == SRC_TRANSPARENT)
    sample += this->getTransparentSourceSample(source_idx, step);
  else
    sample += this->getRegularSourceSample(source_idx, step);

  return sample;
}

double SimulationParameters::getSourceSampleDouble(unsigned int source_idx, unsigned int step) {
  double sample = 0.f;

  if(this->getSource(source_idx).getSourceType() == SRC_TRANSPARENT)
    sample += this->getTransparentSourceSampleDouble(source_idx, step);
  else
    sample += this->getRegularSourceSampleDouble(source_idx, step);

  return sample;
}

float SimulationParameters::getInputDataSample(unsigned int idx, unsigned int sample) {
  unsigned int data_vector_size = (unsigned int)this->source_input_data_.size();
  if(idx >= data_vector_size) {
    log_msg<LOG_ERROR>
    (L"SimulationParameters::getInputDataSample : invalid data index: %d, max idx is %d ")
    %idx % (data_vector_size-1);}

  unsigned int sample_vector_size = (unsigned int)this->source_input_data_.at(idx).size();

  if(sample >= sample_vector_size)
    return 0.f;

  return (this->source_input_data_.at(idx)).at(sample);
}

double SimulationParameters::getInputDataSampleDouble(unsigned int idx, unsigned int sample) {
  unsigned int data_vector_size = (unsigned int)this->source_input_data_double_.size();
  if(idx >= data_vector_size) {
    log_msg<LOG_ERROR>
    (L"SimulationParameters::getInputDataSample : invalid data index: %d, max idx is %d ")
    %idx % (data_vector_size-1);}

  unsigned int sample_vector_size = (unsigned int)this->source_input_data_double_.at(idx).size();

  if(sample >= sample_vector_size)
    return 0.0;

  return (this->source_input_data_double_.at(idx)).at(sample);
}

float SimulationParameters::getGridIrDataSample(unsigned int sample) {
  unsigned int sample_vector_size = (unsigned int)this->grid_ir_.size();

  if(sample >= sample_vector_size)
    return 0.f;

  return (this->grid_ir_.at(sample));
}

nv::Vec3i SimulationParameters::getSourceElementCoordinates(unsigned int source_idx) {
  nv::Vec3i ret = this->getSource(source_idx).getElementIdx(this->getSpatialFs(), 
                                  this->getC(),
                                  (float)this->getLambda());
  if(this->add_padding_to_element_idx_) {
    ret.x = ret.x+1;
    ret.y = ret.y+1;
    ret.z = ret.z+1;
  }

  return ret;
}

nv::Vec3i SimulationParameters::getReceiverElementCoordinates(unsigned int receiver_idx) {
  nv::Vec3i ret = this->getReceiver(receiver_idx).getElementIdx(this->getSpatialFs(), 
                                    this->getC(),
                                    (float)this->getLambda());
  if(this->add_padding_to_element_idx_) {
    ret.x = ret.x+1;
    ret.y = ret.y+1;
    ret.z = ret.z+1;
  }

  return ret;
}

unsigned int SimulationParameters::getSourceElementIdx(unsigned int source_idx, 
                                                       unsigned int dim_x,
                                                       unsigned int dim_y) {
  nv::Vec3i pos = this->getSourceElementCoordinates(source_idx);

  unsigned int padding_increment = 0;

  if(this->add_padding_to_element_idx_)
    padding_increment++;

  pos.x += padding_increment;
  pos.y += padding_increment;
  pos.z += padding_increment;

  return pos.z*dim_x*dim_y+pos.y*dim_x+pos.x;
}

unsigned int SimulationParameters::getReceiverElementIdx(unsigned int receiver_idx, 
                                                         unsigned int dim_x,
                                                         unsigned int dim_y) {

  nv::Vec3i pos = this->getReceiver(receiver_idx).getElementIdx(this->getSpatialFs(), 
                                this->getC(),
                                (float)this->getLambda());

  unsigned int padding_increment = 0;
  if(this->add_padding_to_element_idx_)
    padding_increment++;

  pos.x += padding_increment;
  pos.y += padding_increment;
  pos.z += padding_increment;

  return pos.z*dim_x*dim_y+pos.y*dim_x+pos.x;
}

float SimulationParameters::getRegularSourceSample(unsigned int source_idx, unsigned int step) {
  float sample = 0.f;
  
  switch(sources_.at(source_idx).getInputType()) {
    case IMPULSE: 
      {
      if(step == 1)
        sample += 1.f;
      else
        sample += 0.f;
      break;
      }
    case GAUSSIAN: 
      {
      float t0 = 40;
      float width = 4;
      float exponent_ = ((float)(step-t0)/width);
      sample += expf(-0.5f*(exponent_*exponent_));
      break;
      }

    case DATA:
      {
      unsigned int data_index = sources_.at(source_idx).getInputDataIdx();
      sample += this->getInputDataSample(data_index, step);
      break;
      }

    case SINE:
      {
      float freq = 120;
      float t=(float)step/(float)this->getSpatialFs();
      sample+= sinf(2.f*(float)PI*freq*t);
      }
  }
  return sample;
}

float SimulationParameters::getTransparentSourceSample(unsigned int source_idx, 
                                                       unsigned int step) {
  float sample = 0.f;

  for(unsigned int i = 0; i < step; i++) {
    //printf("step %u ir idx %d, data idx %d \n", step, (step-i), i);
    sample += this->getGridIrDataSample(step-i)*this->getRegularSourceSample(source_idx, i);
  }

  return (this->getRegularSourceSample(source_idx, step))-sample;
}

double SimulationParameters::getRegularSourceSampleDouble(unsigned int source_idx, unsigned int step) {
  double sample = 0.0;
  
  switch(sources_.at(source_idx).getInputType()) {
    case IMPULSE: 
      {
      if(step == 1)
        sample += 1.0;
      else
        sample += 0.0;
      break;
      }
    case GAUSSIAN: 
      {
      double t0 = 40;
      double width = 4;
      double exponent_ = ((float)(step-t0)/width);
      sample += exp(-0.5f*(exponent_*exponent_));
      break;
      }

    case DATA:
      {
      unsigned int data_index = sources_.at(source_idx).getInputDataIdx();
      sample += this->getInputDataSampleDouble(data_index, step);
      break;
      }

    case SINE:
      {
      double freq = 120;
      double t=(double)step/(double)this->getSpatialFs();
      sample+= sin(2.f*(double)PI*freq*t);
      }
  }
  return sample;
}

double SimulationParameters::getTransparentSourceSampleDouble(unsigned int source_idx, 
                                                              unsigned int step) {
  double sample = 0.f;

  for(unsigned int i = 0; i < step; i++) {
    //printf("step %u ir idx %d, data idx %d \n", step, (step-i), i);
    sample += this->getGridIrDataSample(step-i)*this->getRegularSourceSampleDouble(source_idx, i);
  }

  return (this->getRegularSourceSampleDouble(source_idx, step))-sample;
}

float SimulationParameters::getDx() const {
  return (float)((double)this->getC()/((double)this->getSpatialFs()*this->getLambda()));
}

float* SimulationParameters::getSourceVectorAt(unsigned int source_idx) {
  std::vector<float> sample_vector;
  sample_vector.assign(this->getNumSteps(), 0.f);
  this->source_output_data_.resize(this->getNumSources());

  for(unsigned int i = 0; i<this->getNumSteps(); i++) {
    sample_vector.at(i) = this->getSourceSample(source_idx, i);
  }

  this->source_output_data_.at(source_idx) = sample_vector; 
  return &((this->source_output_data_.at(source_idx))[0]);
}

float* SimulationParameters::getParameterPtr()  {
  this->parameter_vec_.assign(4, 0.f);

  this->parameter_vec_.at(0) = (float)this->getLambda();
  this->parameter_vec_.at(1) = (float)(this->getLambda()*this->getLambda());
  this->parameter_vec_.at(2) = 1.f/3.f;
  this->parameter_vec_.at(3) = (float)this->getOctave();
  return &(this->parameter_vec_[0]);
}

double* SimulationParameters::getParameterPtrDouble()  {
  this->parameter_vec_double_.assign(4, 0.f);
  this->parameter_vec_double_.at(0) = this->getLambda();
  this->parameter_vec_double_.at(1) = (this->getLambda()*this->getLambda());
  this->parameter_vec_double_.at(2) = (double)1/(double)3;
  this->parameter_vec_double_.at(3) = (double)this->getOctave();
  return &(this->parameter_vec_double_[0]);
}
