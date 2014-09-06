#ifndef APP_PBO_H
#define APP_PBO_H

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

#include "glHelpers.h"
#include "../math/geomMath.h"

class AppPbo {
public:
  AppPbo() 
  : pbo_(0),
    texture_(0),
    pbo_w_(0),
    pbo_h_(0),
    offset_(),
    u_(),
    v_()
  {};

  ~AppPbo() {};
  
private:
  // One pixelbuffer for each slice orientation
  GLuint pbo_;
  GLuint texture_;
  unsigned int pbo_w_;
  unsigned int pbo_h_;
  nv::Vec3f offset_;
  nv::Vec3f u_;
  nv::Vec3f v_;

public:

  GLuint getPboId() {return this->pbo_;};

  void initPbo(unsigned int dim_x, unsigned int dim_y, 
               nv::Vec3f u, nv::Vec3f v, nv::Vec3f offset);

  void clear();
  
  void draw(float slice);

};

#endif
