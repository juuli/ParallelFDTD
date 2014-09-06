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

#include "cameraProto.hpp"
#include <GL/glew.h>
#include "../gl/glHelpers.h"


using namespace nv;

cameraProto::cameraProto(float current_dist, Vec4f look_at, Vec4f orientation){
  this->current_dist_ = current_dist;
  this->look_at_ = look_at;
  this->orientation_ = orientation;
  
  near_clip_ = 0.1;
  far_clip_ = 500;
  fov_ = 50;

  this->prev_mouse_x_ = 0;
  this->prev_mouse_y_ = 0;
  this->axis_x_ = Vec4f(1.f, 0.f, 0.f, 0.f); 
  this->axis_y_ = Vec4f(0.f, -1.f, 0.f, 0.f);
  this->axis_z_ = Vec4f(0.f, 0.f, 1.f, 0.f);

  current_axis_.setRow(0, this->axis_x_);
  current_axis_.setRow(1, this->axis_y_);
  current_axis_.setRow(2, this->axis_z_);
  current_axis_.setRow(3, Vec4f(0.f, 0.f, 0.f, 1.f));

  /*Bit oldschool, lets get on with quaternions eh? */
  initial_quat_.set(0,0,0,1);
  current_quat_.set(0.3f, 0.1f, 0.f ,1.f);  
  num++;
}

void cameraProto::cpOrbitX(float deg){
  cpRotate(position_, axis_x_, deg);
  // Rotate the local axes to make the orbit work from all directions
  cpRotate(axis_y_, axis_x_, deg);
  cpRotate(axis_z_, axis_x_, deg);
  // We have to rotate our orientation_ as well to get to the "top" in the right angle
  // while looking at the origo
  cpRotate(orientation_, axis_x_, deg);
  orientation_.normalize();
}

void cameraProto::cpOrbitY(float deg){
  cpRotate(position_, axis_y_, deg);
  cpRotate(axis_x_, axis_y_, deg);
  cpRotate(axis_z_, axis_y_, deg);
}

void cameraProto::cpOrbitZ(float deg){
  // Not used here
  cpRotate(position_, axis_z_, deg);
  
}

void cameraProto::cpScale(float amount){
  // Uniform scale here, always something around 1
  float scale = 1+amount;
  /* 
  Lets make a low limit, negative scale is not usefull,
  zero is useless since we should be on an orbit
  */
  if(scale < 0.0001)
    scale = 0.0001f;
  
  current_dist_ *= scale;
}

//---------------------------------------------------------------------
// Rotation according to Rodrigue's formula
// found from: http://mathworld.wolfram.com/RodriguesRotationFormula.html
void cameraProto::cpRotate(Vec4f& point, Vec4f& axis, float deg){
  // Clamp the rotation incement to have sensible results
  deg = CLAMP(deg, -89, 89);

  // Normalize the axis to get legitimate rotations after rotating around for a while.
  // Semi hack since I quessed the normalization would not work as I wanted with the Vec4f
  Vec3f ax3 = axis.getXYZ();
  ax3.normalize();

  float rad = deg/180*(float)M_PI;
  float c = cos(rad);
  float s = sin(rad);
  float x = ax3.x;
  float y = ax3.y;
  float z = ax3.z;

  Mat4f mat;
  mat.m00 = x*x*(1-c)+c;    mat.m01 = y*x*(1-c)-z*s;  mat.m02 = x*z*(1-c)+y*s;  mat.m03 = 0;
  mat.m10 = y*x*(1-c)+z*s;  mat.m11 = y*y*(1-c)+c;    mat.m12 = y*z*(1-c)-x*s;  mat.m13 = 0;
  mat.m20 = z*x*(1-c)-y*s;  mat.m21 = z*y*(1-c)+x*s;  mat.m22 = z*z*(1-c)+c;    mat.m23 = 0;
  mat.m30 = 0;        mat.m31 = 0;        mat.m32 = 0;        mat.m33 = 1;

  point = mat*point;
}

Vec3f cameraProto::pointOnSphere(const Vec2f& point) {
  Vec3f ret;
  
  /* Radius of the sphere here is the minumum dimension of view_port_ */
  float r = MIN(view_port_[2],view_port_[3]);

  /* The Vector from center of the screen to point */
  ret.x = (point.x - view_port_[2]/2.f);
  ret.y = -(point.y - view_port_[3]/2.f);

  /* If length outside radius, z ought to be 0 */
  if((ret.x*ret.x + ret.y*ret.y) >= r*r){
    ret.z = 0.f;
  }
  else{ // Else the z component will such that the vector is on a sphere radius of r
    ret.z = sqrt(r*r-(ret.x*ret.x + ret.y*ret.y));    
  }

  ret.normalize();
  return ret;
}

void cameraProto::cpArchBall(int x, int y){
  Vec3f currentPos = pointOnSphere(Vec2f((float)x, (float)y));
  Vec3f startPos = pointOnSphere(initial_mouse_pos_);
  Vec3f axis = crossed(startPos, currentPos);

  current_quat_ = Quatf(axis.x, axis.y, axis.z, (dot_(startPos, currentPos)))*initial_quat_;
  current_quat_ = normalize(current_quat_);
  
  Mat4f modelView;
  current_quat_.get(modelView);
  current_axis_ = current_axis_*modelView;
}

void cameraProto::cpTranslateLookAt(int dx, int dy) {
  Vec3f trans((float)dx, -1.f*(float)dy, 0.0f);
  Mat4f modelView;
  current_quat_.get(modelView);
  
  this->look_at_ += (Vec4f(((trans/view_port_[3])*current_dist_),0))*modelView;
}

void cameraProto::mousePressed(int x, int y){
  initial_mouse_pos_.set((float)x ,(float)y);
  initial_quat_ = current_quat_;
}

void cameraProto::mouseMoved(int x, int y){
  prev_mouse_x_ = x;
  prev_mouse_y_ = y;
}

void cameraProto::mouseDragged(int x, int y, int key, int shift){
  int dx = x-prev_mouse_x_;
  int dy = y-prev_mouse_y_;

  if(key == 0){
    if(shift == 1){
      cpTranslateLookAt(dx,dy);
    }
    else{
      cpArchBall(x, y);
    }
  }
  
  if(key == 2){
    cpScale(float(dy)*0.001f);  
  }  

  prev_mouse_x_ = x;
  prev_mouse_y_ = y;

}

void cameraProto::setGl() {
  //// Set up a perspective view, with square aspect ratio
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
  // extern void gluPerspective (GLdouble fov_y, GLdouble aspect, GLdouble zNear, GLdouble zFar);
    gluPerspective((GLdouble)fov_, (GLdouble)1.0, (GLdouble)near_clip_, (GLdouble)far_clip_);
    // Rotate the image
    glMatrixMode( GL_MODELVIEW );  // Current matrix affects objects position_s
    glLoadIdentity();              // Initialize to the identity
    
  Mat4f modelView;
  current_quat_.get(modelView);

  gluLookAt(0,  0,  current_dist_,
            0,  0,  0,
            0,  1,  0);
  
  glMultMatrixf(modelView.getPtr());
  glTranslatef(this->look_at_.x, this->look_at_.y,this->look_at_.z);
}

void cameraProto::setPerspective(float fov__, float nClip, float fClip){
  fov_ = fov_;
  near_clip_ = nClip;
  far_clip_ = fClip;
  
}

void cameraProto::setViewPort(float vP[4]){
  view_port_[0] = vP[0];
  view_port_[1] = vP[1];
  view_port_[2] = vP[2];
  view_port_[3] = vP[3];
}

void cameraProto::setViewPort(float x, float y, float w, float h){
  view_port_[0] = x; 
  view_port_[1] = y;
  view_port_[2] = w;
  view_port_[3] = h;
}

