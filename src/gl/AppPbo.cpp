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

#include "AppPbo.h"


void AppPbo::initPbo(unsigned int dim_x, unsigned int dim_y, 
           nv::Vec3f u, nv::Vec3f v, nv::Vec3f offset) {
    // Create a Texture object which handles the pressure data in 3d
  glGenTextures(1, &this->texture_);
  glBindTexture(GL_TEXTURE_2D, this->texture_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, dim_x, dim_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glBindTexture(GL_TEXTURE_2D, 0);

  // Allocate pixel buffer which is updated by CUDA
  glGenBuffers(1, &this->pbo_);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, this->pbo_);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, dim_x*dim_y*sizeof(GL_UNSIGNED_BYTE)*4, 0, GL_STREAM_DRAW_ARB);
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  this->u_ = u;
  this->v_ = v;
  this->pbo_w_ = dim_x;
  this->pbo_h_ = dim_y;
  this->offset_ = offset;
}


void AppPbo::clear() {
  glDeleteBuffers(1, &this->pbo_);
  glDeleteTextures(1, &this->texture_);
}

void AppPbo::draw(float slice) {
   // draw image from PBO
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, this->texture_);
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, this->pbo_);
  glTexSubImage2D(GL_TEXTURE_2D, 0,0,0, this->pbo_w_, this->pbo_h_,
                  GL_RGBA, GL_UNSIGNED_BYTE, 0);

  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  glBindTexture(GL_TEXTURE_2D, this->texture_);

  nv::Vec3f offset = this->offset_*nv::Vec3f(slice,slice,slice);
  nv::Vec3f slice_pos = this->u_+this->v_+ offset;
  nv::Vec3f u = this->u_+offset;
  nv::Vec3f v = this->v_+offset;

  glColor4f(1.f, 1.f, 1.f, 1.f);
  glBegin(GL_QUADS);
  glNormal3f(0.f, 0.f, 1.f);
  glTexCoord2f(0.0f, 0.0f);   glVertex3f(offset.x, offset.y, offset.z);
  glTexCoord2f(1.0f, 0.0f);   glVertex3f(u.x, u.y, u.z);
  glTexCoord2f(1.0f, 1.0f);   glVertex3f(slice_pos.x, slice_pos.y, slice_pos.z);
  glTexCoord2f(0.0f, 1.0f);   glVertex3f(v.x, v.y, v.z);
  glEnd();

    // unbind texture
  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_2D);
  
  glLineWidth(2);
  glColor4f(0.1f, 0.0f, 0.3f, 0.7f);
  glBegin(GL_LINE_STRIP);
  glVertex3f(offset.x, offset.y, offset.z);
  glVertex3f(u.x, u.y, u.z);
  glVertex3f(slice_pos.x, slice_pos.y,  slice_pos.z);
  glVertex3f(v.x, v.y, v.z);
  glVertex3f(offset.x, offset.y, offset.z);
  glEnd();
}

