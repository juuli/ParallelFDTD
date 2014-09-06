#ifndef CAMERA_PROTO_H
#define CAMERA_PROTO_H
#define _USE_MATH_DEFINES

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

#include "../math/geomMath.h"
#include <iostream>

/*
A class for different affine transformations. 
*/

using namespace nv;

static int num = 0;

class cameraProto
{

public:
  cameraProto(void) 
  : look_at_(Vec4f()), 
    orientation_(Vec4f()), 
    current_dist_(5.f),
    prev_mouse_x_(0), 
    prev_mouse_y_(0),
    axis_x_(Vec4f(1.f, 0.f, 0.f, 0.f)), 
    axis_y_(Vec4f(0.f, -1.f, 0.f, 0.f)), 
    axis_z_(Vec4f(0.f, 0.f, 1.f, 0.f)),
    current_axis_()
    {num++;};

  cameraProto(float current_dist, Vec4f look_at, Vec4f orientation);

  ~cameraProto(void) {};

  void cpOrbitX(float deg);
  void cpOrbitY(float deg);
  void cpOrbitZ(float deg);
  void cpScale(float amount); // Means "move away from look_at_ point" in this case
  void cpRotate(Vec4f& point, Vec4f& axis, float deg);
  void cpArchBall(int x, int y);
  void cpTranslateLookAt(int x, int y);
  Vec3f pointOnSphere(const Vec2f& point);
  void setPosition(Vec3f pos);
  void setLookAt(Vec3f look_at);
  void setOrientation(Vec3f look_at);
  void setPerspective(float fov, float nClip, float fClip);
  
  Vec3f getLookAt() {return this->look_at_.getXYZ();};
  Vec3f getOrientation() {return this->orientation_.getXYZ();};
  
  void mousePressed(int x, int y);
  void mouseMoved(int x, int y);
  void mouseDragged(int x, int y, int key, int shift);
  
  void setViewPort(float vP[4]);
  void setViewPort(float x, float y, float w, float h);
  void setGl();

private:  
  Vec4f position_;
  Vec4f look_at_;
  Vec4f orientation_;
  
  float view_port_[4];
  float near_clip_;
  float far_clip_;
  float fov_;
  
  Vec2f initial_mouse_pos_;
  Quatf initial_quat_;
  Mat4f initial_rotation_;
  Quatf current_quat_;
  
  float current_dist_;
  
  int prev_mouse_x_;
  int prev_mouse_y_;
  Vec4f axis_x_, axis_y_, axis_z_;
  Mat4f current_axis_;

};

#endif
