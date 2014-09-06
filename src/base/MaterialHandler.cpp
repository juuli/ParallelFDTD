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

#include "MaterialHandler.h"
#include "../global_includes.h"


void MaterialHandler::addMaterials(float* material_ptr, 
                                   unsigned int number_of_surfaces, 
                                   unsigned int number_of_coefficients) {
  std::vector<float> temp_coefficients;
  log_msg<LOG_INFO>(L"MaterialHandler::addMaterials - num surfaces %d num coef %d")
                    % number_of_surfaces % number_of_coefficients;

  // Allocate the default number of coefficients
  temp_coefficients.assign(MATERIAL_COEF_NUM, 0.f);

  for(unsigned int i = 0; i < number_of_surfaces; i++) {
    // Save the number of coefficients defined
    for(unsigned int j = 0; j < number_of_coefficients; j++) {
      temp_coefficients.at(j) = material_ptr[i*number_of_coefficients+j];
      
    }
    this->addSurfaceMaterial(temp_coefficients);
  }
}

unsigned int MaterialHandler::addSurfaceMaterial(std::vector<float> material_coefficients) {
  unsigned int vec_size = (unsigned int)material_coefficients.size();
  unsigned int coef_num = this->number_of_coefficients_;
  
  if(coef_num != vec_size)
    log_msg<LOG_ERROR>(L"MaterialHandler::addSurfaceMaterial - invalid number of coefficients %d, sould be %d") 
                       %coef_num %MATERIAL_COEF_NUM;

  while(material_coefficients.size() <= MATERIAL_COEF_NUM)
    material_coefficients.push_back(0.f);

  // Check if the material exists
  unsigned int material_idx = 0;
  material_idx = this->checkIfUnique(material_coefficients);

  this->material_indices_.push_back((unsigned char)material_idx);
  this->number_of_surfaces_++;
  return material_idx;
}

float MaterialHandler::getUniqueCoefAt(unsigned int material, unsigned int coef_idx) {
  unsigned int coef_num = (unsigned int)MATERIAL_COEF_NUM;
  if(coef_num <= coef_idx) {
    log_msg<LOG_ERROR>(
      L"MaterialHandler::getUniqueCoefAt - invalid index %d, should be less than %d") 
      %coef_idx %coef_num;
    throw std::out_of_range("Coef idx out of range");
  }
  
  material_t mat = this->unique_coefficients_.at(material);
  return mat.coefs[coef_idx];
}

float MaterialHandler::getSurfaceCoefAt(unsigned int surface, unsigned int coef_idx) {
  unsigned int coef_num = (unsigned int)MATERIAL_COEF_NUM;
  
  if(coef_num <= coef_idx) {
    log_msg<LOG_ERROR>(L"MaterialHandler::getSurfaceCoefAt - invalid index %d, should be less than %d") 
    %coef_idx %coef_num;
    throw std::out_of_range("Coef idx out of range");
  }
  
  return this->getUniqueCoefAt(this->material_indices_.at(surface), coef_idx);
}

unsigned char* MaterialHandler::getMaterialIdxPtr() {
  if(this->material_indices_.size() == 0) {
    log_msg<LOG_INFO>(L"MaterialHandler::getMaterialIdxPtr() - No materials assigned");
    return (unsigned char*)NULL;
  }
  
  return &(this->material_indices_[0]);}

float* MaterialHandler::getMaterialCoefficientPtr() {
  this->coefficient_vector_.clear();
  unsigned int coefs = this->getNumberOfCoefficients();
  unsigned int unique_mat = this->getNumberOfUniqueMaterials();

  this->coefficient_vector_.assign(coefs*unique_mat, 0.f);
  
  float mean = 0.f;
  
  for(unsigned int i = 0; i < unique_mat; i++) {
    for(unsigned int j = 0; j < coefs; j++) {
      if(!this->admitance_){
        float coef =  this->getUniqueCoefAt(i,j);
        coef = reflection2Admitance(coef);
        coefficient_vector_.at(i*coefs +j) = coef;
        mean+=coef;
      }
      else {
        float coef =  this->getUniqueCoefAt(i,j);
        coefficient_vector_.at(i*coefs +j) = coef;
        mean+=coef;
      }
    }
  }
  mean /=coefs*unique_mat;

  log_msg<LOG_INFO>
  (L"MaterialHanlder::getMaterialCoefficientPtr - %d unique materials, %d coefficients, mean value %f")
  % unique_mat %coefs %mean;
  return &(this->coefficient_vector_[0]);
}

double* MaterialHandler::getMaterialCoefficientPtrDouble() {
  this->coefficient_vector_.clear();
  unsigned int coefs = this->getNumberOfCoefficients();
  unsigned int unique_mat = this->getNumberOfUniqueMaterials();

  this->coefficient_vector_double_.assign(coefs*unique_mat, 0.0);
  
  double mean = 0.f;
  
  for(unsigned int i = 0; i < unique_mat; i++) {
    for(unsigned int j = 0; j < coefs; j++) {
      if(!this->admitance_){
        double coef =  (double)this->getUniqueCoefAt(i,j);
        coef = (double)reflection2Admitance((double)coef);
        coefficient_vector_double_.at(i*coefs +j) = coef;
        mean+=coef;
      }
      else {
        double coef =  (double)this->getUniqueCoefAt(i,j);
        coefficient_vector_double_.at(i*coefs +j) = coef;
        mean+=coef;
      }
      //coefficient_vector_.at(i*coefs +j) = this->getUniqueCoefAt(i,j);
    }
  }
  mean /=coefs*unique_mat;

  log_msg<LOG_INFO>
  (L"MaterialHanlder::getMaterialCoefficientPtr - %d unique materials, %d coefficients, mean value %f")
  % unique_mat %coefs %mean;
  return &(this->coefficient_vector_double_[0]);

}

float MaterialHandler::getMeanAbsorption(unsigned int octave) {
  float mean_absorption = 0.f;
  for(unsigned int i = 0; i < this->number_of_surfaces_; i++) {
    float coef = admitance2Reflection(this->getSurfaceCoefAt(i,octave));
    coef = 1-coef*coef;
    mean_absorption += coef;
  }
  return mean_absorption/this->number_of_surfaces_;
}

bool checkCoefficients(std::vector<float> material_coefficient, MaterialHandler::material_t current_coefficient) {
  bool ret = true;
  for(unsigned int i = 0; i < MATERIAL_COEF_NUM; i++) {
    if(material_coefficient.at(i) != current_coefficient.coefs[i])
      ret = false;
  }

  return ret;
}

unsigned int MaterialHandler::checkIfUnique(std::vector<float> material_coefficients) {
  unsigned int ret = 0;

  std::vector<material_t>::iterator material_iterator;

  for(material_iterator = unique_coefficients_.begin(); material_iterator != unique_coefficients_.end(); material_iterator++) {
    if(checkCoefficients(material_coefficients, *material_iterator))
      return ret;
    ret++;
  }

  material_t new_material;
  for(unsigned int i = 0; i < MATERIAL_COEF_NUM; i++) {
    new_material.coefs[i] = material_coefficients.at(i);
  }
  this->number_of_unique_materials_++;
  this->unique_coefficients_.push_back(new_material);
  return ret;
}

void MaterialHandler::setGlobalMaterial(unsigned int number_of_surfaces, float coef) {
  log_msg<LOG_DEBUG>
    (L"MaterialHandler::setGlobalMaterial - assigning %d surfaces with coefficient %f")
    %number_of_surfaces %coef;

  std::vector<float> mat;
  mat.assign(MATERIAL_COEF_NUM, coef);
  for(unsigned int i = 0; i < number_of_surfaces; i++) {
    this->addSurfaceMaterial(mat);
  }
}

void MaterialHandler::setMaterialIndexAt(unsigned int surface_idx, unsigned char material_idx) {
  if(surface_idx >= this->number_of_surfaces_) {
    log_msg<LOG_INFO>(L"MaterialHandler::setMaterialIndexAt - surface index %d out of bounds % d, returning")
                      % surface_idx %this->number_of_surfaces_;
    return;
  }
  this->material_indices_.at(surface_idx) = material_idx;
}

