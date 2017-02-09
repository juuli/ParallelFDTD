#ifndef MATERIAL_HANDLER_H
#define MATERIAL_HANDLER_H

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

#include <vector>
#include <string>

#define MATERIAL_COEF_NUM 20

///////////////////////////////////////////////////////////////////////////////
/// \brief Class that manages the material parameters of the model
///////////////////////////////////////////////////////////////////////////////
class MaterialHandler {
public:
  MaterialHandler()
  : admitance_(true),
    number_of_coefficients_(MATERIAL_COEF_NUM),
    number_of_surfaces_(0),
    number_of_unique_materials_(0),
    material_index_increment_(1)
  {};

  ~MaterialHandler() {};

  /////////////////////////////////////////////////////////////////////////////
  /// \brief Add materials for each surface of the model
  /// \param material_ptr pointer to the list of material coefficients of the
  /// model
  /// \param number_of_surfaces number of surfaces in the model
  /// \param number_of_coefficients number of coefficients for each surface
  /////////////////////////////////////////////////////////////////////////////
  void addMaterials(float* material_ptr, unsigned int number_of_surfaces,
                    unsigned int number_of_coefficients);

  /////////////////////////////////////////////////////////////////////////////
  /// \brief Add a single material to the material list and add a material index
  /// to the material index list. If the material exists, the index of the
  /// previous instance of the material is added to the index list. This function
  /// should be used when the surface materials are added sequientially to the
  /// handler in the same order as the surfaces are presented in the geometry
  /// handler to retain right order of indices.
  ///
  /// \param material_coefficients any number of coefficients to be added
  /// to the material list. If the number of coefficients is less than
  /// MATERIAL_COEF_NUM, rest of the coefficients are set to zero
  /////////////////////////////////////////////////////////////////////////////
  unsigned int addSurfaceMaterial(std::vector<float> material_coefficients);

  /////////////////////////////////////////////////////////////////////////////
  /// \brief Indicate that the material list holds admittance values
  /////////////////////////////////////////////////////////////////////////////
  void coefsAreAdmittances() {this->admitance_ = true;}

  /////////////////////////////////////////////////////////////////////////////
  /// \brief Indicate that the material list holds reflection coefficient values
  /////////////////////////////////////////////////////////////////////////////
  void coefsAreReflectances() {this->admitance_ = false;}

  unsigned int getNumberOfCoefficients() {return this->number_of_coefficients_;}
  unsigned int getNumberOfSurfaces() {return this->number_of_surfaces_;}
  unsigned int getNumberOfUniqueMaterials() {
    return this->number_of_unique_materials_+this->material_index_increment_;}
  unsigned int getMaterialIdxAt(unsigned int idx) {return this->material_indices_.at(idx);}

  /////////////////////////////////////////////////////////////////////////////
  ///\return A pointer to a list of material indices of each surface of the model
  /////////////////////////////////////////////////////////////////////////////
  unsigned char* getMaterialIdxPtr();

  /////////////////////////////////////////////////////////////////////////////
  ///\return A pointer to a list of all material coefficients used in the model,
  /// single precision
  /////////////////////////////////////////////////////////////////////////////
  float* getMaterialCoefficientPtr();

  /////////////////////////////////////////////////////////////////////////////
  ///\return A pointer to a list of all material coefficients used in the model,
  /// double precision
  /////////////////////////////////////////////////////////////////////////////
  double* getMaterialCoefficientPtrDouble();

  /////////////////////////////////////////////////////////////////////////////
  /// \brief Get an unique material from the list
  /// \param material the index of the material at the material list
  /// \param coef_idx the index of the coefficient, [0 - MATERIAL_COEF_NUM-1]
  /// \return a material coefficient at the given material and coefficient index
  /////////////////////////////////////////////////////////////////////////////
  float getUniqueCoefAt(unsigned int material, unsigned int coef_idx);

  /////////////////////////////////////////////////////////////////////////////
  /// \brief Get a material coefficient of a surface of the model
  /// \param surface the index of the surface of the model
  /// \param coef_idx the index of the coefficient, [0 - MATERIAL_COEF_NUM-1]
  /// \return a material coefficient at the given surface and coefficient index
  /////////////////////////////////////////////////////////////////////////////
  float getSurfaceCoefAt(unsigned int surface, unsigned int coef_idx);

  /////////////////////////////////////////////////////////////////////////////
  /// \param octave the octave band in which the mean is calculated
  /// \return the mean absorption of the model
  /////////////////////////////////////////////////////////////////////////////
  float getMeanAbsorption(unsigned int octave);

  void setNumberOfCoefficients(unsigned int number_of_coefficients) {
    this->number_of_coefficients_ = number_of_coefficients;}

  /////////////////////////////////////////////////////////////////////////////
  /// \brief Set a global material for the whole model
  /// \param number_of_surfaces the number of surfaces in the model
  /// \param coef the global coefficient added to the model. Depending if
  /// the coefsAreAdmittances() or coefsAreReflectances() is called
  /// the value is interpreted as Admittance or Reflectance
  /// \return the mean absorption of the model
  /////////////////////////////////////////////////////////////////////////////
  void setGlobalMaterial(unsigned int number_of_surfaces, float coef);

  /////////////////////////////////////////////////////////////////////////////
  /// \brief Set the material index of a given surface to arbitrary value
  /// \param sufrace_idx The index of the surface (triangle) in the geometry
  /// \param material_idx The index of the material in the material list
  /////////////////////////////////////////////////////////////////////////////
  void setMaterialIndexAt(unsigned int surface_idx, unsigned char material_idx);

  /////////////////////////////////////////////////////////////////////////////
  /// \brief Set the number of unique materials list to have N extra materials
  ///        in the beginning of the list. Increments the material indices
  ///        accordignly
  /// \param increment The increment to be added
  /////////////////////////////////////////////////////////////////////////////
  void setMaterialIndexIncrement(unsigned int increment) {
    this->material_index_increment_ = increment;}

  struct material_t {
    float coefs[MATERIAL_COEF_NUM];
  };

private:
  bool admitance_; ///< indicates if the values are admitance values
  unsigned int number_of_coefficients_; ///<number of coefficients used
  unsigned int number_of_surfaces_;     ///<number of surfaces in the model
  unsigned int number_of_unique_materials_; ///<number of unique materials in the material handler
  unsigned int material_index_increment_; ///<increment to add a material to the fron to the list

  /// Indexes of the materials of the polygons
  std::vector<unsigned char> material_indices_;
  /// Unique material coefficients in the model
  std::vector<material_t> unique_coefficients_;

  /// Material coefficients in a 1-D vector, single precision mode
  std::vector<float> coefficient_vector_;
  /// Material coefficients in a 1-D vector, double precision mode
  std::vector<double> coefficient_vector_double_;

  /// Check if the material is already in the unique list
  /// \return The material index of the added material
  unsigned int checkIfUnique(std::vector<float> material_coefficients);

};

#endif
