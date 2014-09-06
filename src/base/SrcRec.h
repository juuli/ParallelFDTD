#ifndef SRCREC_H
#define SRCREC_H

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

#include <string>
#include <vector>

// NVIDIA vector templates
#include "../math/geomMath.h"
#include "../global_includes.h"

enum SrcType {SRC_HARD, SRC_SOFT, SRC_TRANSPARENT};
enum InputType {IMPULSE, GAUSSIAN, SINE, DATA};

///////////////////////////////////////////////////////////////////////////////
/// A base class for source and receiver positions
///////////////////////////////////////////////////////////////////////////////
class Position{
public:
  Position() 
  : p_(nv::Vec3f())
    {};

  Position(float x, float y, float z)
  : p_(nv::Vec3f(x,y,z))
    {};

  Position(float x, float y)
  : p_(nv::Vec3f(x,y,0.f))
    {};


  virtual ~Position() {};

protected:
  nv::Vec3f p_; ///< The position of in 3-D 

public:

  nv::Vec3f getP() {return this->p_;};
  
  ///////////////////////////////////////////////////////////////////////////////
  /// \brief Returns the element index in a finite difference grid
  /// \param spatial_fs The spatial sampling rate
  /// \param c The speed of sound used in the simulation
  /// \param lambda The Courant number of the scheme used
  ///////////////////////////////////////////////////////////////////////////////
  nv::Vec3i getElementIdx(unsigned spatial_fs, float c, float lambda);
};

///////////////////////////////////////////////////////////////////////////////
/// A Class defining a point source which is used in the simulation <br>
/// Srouce type is definend with enum SrcType and the input type is definend 
/// with enum InputType
///////////////////////////////////////////////////////////////////////////////
class Source : public Position {
public:
  Source() 
  : Position(),
    source_type_(SRC_HARD),
    input_type_(IMPULSE),
    group_(0),
    input_data_idx_(0)
  {};

  Source(float x, float y, float z) 
  : Position(x, y, z),
    source_type_(SRC_HARD),
    input_type_(IMPULSE),
    group_(0),
    input_data_idx_(0)
  {};

  Source(float x, float y, float z, enum SrcType source_type) 
  : Position(x, y, z),
    source_type_(source_type),
    input_type_(IMPULSE),
    group_(0),
    input_data_idx_(0)
  {};
  
  Source(float x, float y, float z, enum SrcType source_type, unsigned int group)
  : Position(x, y, z),
    source_type_(source_type),
    input_type_(IMPULSE),
    group_(group),
    input_data_idx_(0)
  {};

  Source(float x, float y, float z, enum SrcType source_type, 
         enum InputType input_type,  unsigned int input_data_idx) 
  : Position(x, y, z),
    source_type_(source_type),
    input_type_(input_type),
    group_(0),
    input_data_idx_(input_data_idx)
  {};

  Source(float x, float y) 
  : Position(x, y),
    source_type_(SRC_HARD),
    input_type_(IMPULSE),
    group_(0),
    input_data_idx_(0)
  {};

  // Setters
  void setSourceType(enum SrcType source_type) {this->source_type_ = source_type;};
  void setInputType(enum InputType input_type) {this->input_type_ = input_type;};
  void setGroup(unsigned int group) {this->group_ = group;};
  
  void setInputDataIdx(unsigned int input_data_idx) 
    {this->input_data_idx_ = input_data_idx;};
  
  unsigned int getInputDataIdx() const {return this->input_data_idx_;};
  
  // Getters
  enum SrcType getSourceType() const {return this->source_type_;};
  enum InputType getInputType() const {return this->input_type_;};
  unsigned int getGroup() const {return this->group_;};

private:
  SrcType source_type_;
  InputType input_type_;
  unsigned int group_;  ///< The source group of the source, NOT IN USE
  unsigned int input_data_idx_; ///< Which custom data is used
};

class Receiver : public Position {
public:
  Receiver() 
  : Position()
  {};

  Receiver(float x, float y, float z) 
  : Position(x, y, z)
  {};

  Receiver(float x, float y) 
  : Position(x, y)
  {};

  void setOutputFp(std::string fp);

private:
  std::string output_fp;
};

#endif
