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

#include "AppWindow.h"
#include "glHelpers.h"

#include "../App.h"
#include "../base/GeometryHandler.h"
#include "../global_includes.h"

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <sstream>

//

AppWindow* currentInstance;
static FDTD::App* currentApp;
static const char* schemes[] = {"Standard Leapfrog Forward", "Standard Leapfrog Forward Shared memory", "Standard Leapfrog Centred", };
static const char* boundaries[] = {"Frequency independent", "DIF"};

////// Hack to make member functions callbacks
// Done because Open GL needs C style callback functions


extern "C" {
static void drawCallback() {
  currentInstance->draw();
}}

extern "C" {
static void mouseCallback(int button, int state, int x, int y) {
  currentInstance->mouse(button, state, x, y);
}}

extern "C" {
static void keyboardDownCallback(unsigned char key, int x, int y) {
  currentInstance->keyboardDown(key, x, y);
}}

extern "C" {
static void keyboardUpCallback(unsigned char key, int x, int y) {
  currentInstance->keyboardUp(key, x, y);
}}

extern "C" {
static void motionCallback(int x,int y) {
  currentInstance->motion(x,y);
}}

extern "C" {
static void closeCallback() {
  currentInstance->close();
}}

extern "C" {
static void initErrorCallback() {
  c_log_msg(LOG_ERROR, "AppWindow - error during glInit");
}
}

extern "C" {
static void initWarningCallback() {
  c_log_msg(LOG_ERROR, "AppWindow - warning during glInit");
}
}


//////////// Callbacks
void AppWindow::draw() {
  float dB = 10.f;

  if(this->window_params_.invert_time)
    currentApp->invertTime(); 

  if(this->window_params_.mesh_reset)
    currentApp->resetPressureMesh();
  
  // Execute step
  if(!this->window_params_.execution_pause)
    currentApp->executeStep();

  if(this->window_params_.draw_xy)
    currentApp->updateVisualization(this->window_params_.current_z, 0, 
                    this->window_params_.selection, dB);
  if(this->window_params_.draw_xz)
    currentApp->updateVisualization(this->window_params_.current_y, 1, 
                    this->window_params_.selection, dB);
  if(this->window_params_.draw_yz)
    currentApp->updateVisualization(this->window_params_.current_x, 2, 
                    this->window_params_.selection, dB);

  glClearColor (1.0f, 1.0f, 1.0f, 1.0f); 
  glClearDepth (1.0f); 
  glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  
  this->drawStatusText();
  if(this->window_params_.help)
    this->drawHelp();

  camera_.setGl();

  this->drawAxes();
  this->drawPbos();

  if(this->window_params_.draw_geometry)
    this->drawGeometry();

  drawSourcesAndReceivers();
  glutSwapBuffers ();
  glFlush (); 

  if(this->window_params_.step)
    this->window_params_.execution_pause = true;

  this->window_params_.mesh_reset = false;
  this->window_params_.invert_time = false;
  
  glutPostRedisplay();

}

void AppWindow::mouse(int button, int state, int x, int y) {

  /*
   The button parameter is one of GLUT_LEFT_BUTTON, GLUT_MIDDLE_BUTTON, or GLUT_RIGHT_BUTTON. 
   For systems with only two mouse buttons, it may not be possible to generate GLUT_MIDDLE_BUTTON callback.
   
   state parameter is either GLUT_UP or GLUT_DOWN indicating whether the callback was due to a release or press respectively
   */
  
  if(GLUT_ACTIVE_SHIFT == glutGetModifiers()){
    this->window_params_.shift = true;
  }

  if(state == GLUT_DOWN){
    this->window_params_.m_down = true;
    this->window_params_.m_button = button;
    camera_.mousePressed(x, y);
  }

  if(state == GLUT_UP){
    this->window_params_.m_down = false;
    this->window_params_.m_button = -1;
    this->window_params_.shift = false;
  }

  
}

void AppWindow::keyboardDown(unsigned char key, int x, int y) {
  this->window_params_.key = key;
  this->window_params_.k_down = true;
  (void)x;
  (void)y;

  if(key == 'p')
    this->window_params_.execution_pause = !this->window_params_.execution_pause;

  if(key == 'r')
    this->window_params_.mesh_reset = true;

  if(key == 'f')
    this->window_params_.invert_time = true;

  if(key == 's')
    this->window_params_.step = !this->window_params_.step;

  if(key == 'z') {
    this->window_params_.draw_xy = !this->window_params_.draw_xy;
  }

  if(key == 'x') {
    this->window_params_.draw_yz = !this->window_params_.draw_yz;
  }
  if(key == 'y') {
    this->window_params_.draw_xz = !this->window_params_.draw_xz;
  }
  if(key == 'g') {
    this->window_params_.draw_geometry = !this->window_params_.draw_geometry;
  }

  if(key == 'q') {
    this->window_params_.selection = this->window_params_.selection++;
    this->window_params_.selection = this->window_params_.selection%3;
  }

  if(key == 'h') {
    this->window_params_.help = !this->window_params_.help;
  }

  if(key == 56) {
    unsigned int dim = currentApp->m_mesh.getDimZ();
    this->window_params_.current_z++;
    this->window_params_.current_z = (window_params_.current_z < dim) ? window_params_.current_z : 0;

  }
  if(key == 50) {
    unsigned int dim = currentApp->m_mesh.getDimZ();
    this->window_params_.current_z--;
    this->window_params_.current_z = (window_params_.current_z < dim) ? window_params_.current_z : 0;
  }

  if(key == 54) {
    unsigned int dim = currentApp->m_mesh.getDimX();
    this->window_params_.current_x++;
    this->window_params_.current_x = (window_params_.current_x < dim) ? window_params_.current_x : 0;

  }

  if(key == 52) {
    unsigned int dim = currentApp->m_mesh.getDimX();
    this->window_params_.current_x--;
    this->window_params_.current_x = (window_params_.current_x < dim) ? window_params_.current_x : 0;

  }

  if(key == 57) {
    unsigned int dim = currentApp->m_mesh.getDimY();
    this->window_params_.current_y++;
    this->window_params_.current_y = (window_params_.current_y < dim) ? window_params_.current_y : 0;

  }

  if(key == 49) {
    unsigned int dim = currentApp->m_mesh.getDimY();
    this->window_params_.current_y--;
    this->window_params_.current_y = (window_params_.current_y < dim) ? window_params_.current_y : 0;
  }
}

void AppWindow::keyboardUp(unsigned char key, int x, int y){
  this->window_params_.key = key;
  this->window_params_.k_down = false;
  (void)x;
  (void)y;
  
  
}

void AppWindow::motion(int x, int y) {

  if(GLUT_ACTIVE_SHIFT == glutGetModifiers()){
    this->window_params_.shift = true;
  }
  else {
    this->window_params_.shift = false;
  }

  if(this->window_params_.mouse_x != x) {
    this->window_params_.p_mouse_x = this->window_params_.mouse_x;
    this->window_params_.mouse_x = x;
  }

  if(this->window_params_.mouse_y != y) {
    this->window_params_.p_mouse_y = this->window_params_.mouse_y;
    this->window_params_.mouse_y = y;
  }

  if(window_params_.m_down){
    camera_.mouseDragged(x,y, window_params_.m_button, window_params_.shift);
    glutPostRedisplay();
  }
  else{
    camera_.mouseMoved(x,y);
  }

}

void AppWindow::close() {
  for(unsigned int i = 0; i < this->vbo_vector_.size(); i++)
    this->deleteVbo(i);
  for(unsigned int i = 0; i < this->pbo_vector_.size(); i++)
    this->pbo_vector_.at(i).clear();
    
  glutLeaveMainLoop();
}

///////// Initialization

void AppWindow::initializeGL(int argc, char **argv) {
  glutInit(&argc, argv);
}

void AppWindow::initializeWindow(int w, int h) {
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
  log_msg<LOG_DEBUG>(L"AppWindow::initializeWindow - init window size");
  glutInitWindowSize(w,h);
  this->window_id_ = glutCreateWindow("Parallel FDTD");
  log_msg<LOG_DEBUG>(L"AppWindow::initializeWindow - create window");
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

  // viewport
  glViewport(0, 0, w, h);
  camera_.setViewPort(0.f,0.f, (float)w, (float)h);

  // projection
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (GLfloat)w / (GLfloat) h, 0.1, 10.0);

  // Initialize GLEW
  glewsafe(glewInit(), "AppWindow: glewinit");

  // Window controls
  this->resetParameters();
  this->setupCallbacks();
    
  glsafe(glGetError(), "AppWindow::initializeWindow - before return");
}

void AppWindow::startMainLoop(FDTD::App* currentApp_) {
  currentApp = currentApp_;
  try {glutMainLoop();}
  catch (...){
    log_msg<LOG_ERROR>(L"AppWindow::startMainLoop - Main loop failed");
    throw(-1);
  }
}

////// Utilities
GLuint AppWindow::getVboVerticeId(unsigned int idx) {
  log_msg<LOG_INFO>(L"AppWindow::getVboVerticeId - index %d") %idx;

  if(idx>=this->vbo_vector_.size())
    log_msg<LOG_INFO>(L"getVboVerticeId: index %d invalid, size is %d") %idx %vbo_vector_.size();

  return vbo_vector_.at(idx).getVertId();
}

GLuint AppWindow::getVboColorId(unsigned int idx) {
  log_msg<LOG_INFO>(L"AppWindow::getVboColorId - index %d") %idx;

  if(idx>=this->vbo_vector_.size())
    log_msg<LOG_INFO>(L"getVboColorId: index %d invalid, size is %d") %idx %vbo_vector_.size();

  return vbo_vector_.at(idx).getColorId();
}

GLuint AppWindow::getPboIdAt(unsigned int idx) {
  log_msg<LOG_INFO>(L"AppWindow::getPboIdAt - index %d") %idx;

  if(idx>=this->pbo_vector_.size())
    log_msg<LOG_INFO>(L"getPboColorId: index %d invalid, size is %d") %idx %pbo_vector_.size();

  return pbo_vector_.at(idx).getPboId();
}

void AppWindow::setupCallbacks() {
  currentInstance = this;
  glutMouseFunc(mouseCallback);
  glutKeyboardFunc(keyboardDownCallback);
  glutKeyboardUpFunc(keyboardUpCallback);
  glutMotionFunc(motionCallback);
  glutPassiveMotionFunc(motionCallback);
  glutCloseFunc(closeCallback);
  glutDisplayFunc(drawCallback);
  //glutInitErrorFunc(initErrorCallback);
  //glutInitWarningFunc(initWarningCallback);
}

void AppWindow::deleteVbo(unsigned int idx) {
  if(idx < vbo_vector_.size())
    vbo_vector_[idx].clear();
}

void AppWindow::setData(void* data) {
  this->data_ = data;
}

void AppWindow::geometryToVbo(void* data) {
  this->data_ = data;
  GeometryHandler* gh = (GeometryHandler*)this->data_;
  // Push an empty Vertex Buffer object
  this->vbo_vector_.push_back(AppVbo());
  
  unsigned int number_of_coordinates = 4*gh->getNumberOfVertices();
  GLfloat* vertices = new GLfloat[number_of_coordinates];

  unsigned int count = 0;
  for(unsigned int i = 0; i < gh->getNumberOfVertices(); i++) {
    vertices[count] = *(gh->getVertexAt(i));
    vertices[count+1] = *(gh->getVertexAt(i)+1);
    vertices[count+2] = *(gh->getVertexAt(i)+2);
    vertices[count+3] = 1.f;
    count+=4;
  }

  // Select the added VBO
  unsigned int idx = (unsigned int)this->vbo_vector_.size()-1;

  log_msg<LOG_INFO>(L"AppWindow::geometryToVbo - vertex buffer size %d") % number_of_coordinates;
  this->vbo_vector_[idx].setVertexData(vertices, 4, gh->getNumberOfVertices(), GL_STATIC_DRAW);
  log_msg<LOG_INFO>(L"AppWindow::geometryToVbo - index buffer size %d") % (3*gh->getNumberOfTriangles());
  this->vbo_vector_[idx].setIndexData(gh->getIndexPtr(), 3*gh->getNumberOfTriangles(), GL_STATIC_DRAW);
  
  delete [] vertices;
}

void AppWindow::createEmptyVbo(unsigned int num_coords, unsigned int num_verts) {
  log_msg<LOG_INFO>(L"AppWindow::createEmptyVbo - num_coords %d, num_verts %d, buffer size %d") %num_coords %num_verts %(num_verts*num_coords);
  this->vbo_vector_.push_back(AppVbo());
  unsigned int idx = (unsigned int)this->vbo_vector_.size()-1;
  
  this->vbo_vector_[idx].setVertexData((float*)NULL, num_coords, num_verts, GL_DYNAMIC_DRAW);
  this->vbo_vector_[idx].setColorData((float*)NULL, GL_DYNAMIC_DRAW);
}

void AppWindow::addPixelBuffer(unsigned int dim_x, unsigned int dim_y,
                               nv::Vec3f u, nv::Vec3f v, nv::Vec3f offset) {
  log_msg<LOG_INFO>(L"AppWindow::addPixelBuffer - dim x %d, dim_y %d") %dim_x %dim_y;
  this->pbo_vector_.push_back(AppPbo());
  unsigned int idx = (unsigned int)this->pbo_vector_.size()-1;
  this->pbo_vector_.at(idx).initPbo(dim_x, dim_y, u, v, offset);
}

//////////////////
//// Draw functions

void AppWindow::drawString3D(const char *str, nv::Vec3f pos, nv::Vec3f color, void *font)
{
  glPushAttrib(GL_LIGHTING_BIT | GL_CURRENT_BIT); // lighting and color mask
  glDisable(GL_LIGHTING);     // need to disable lighting for proper text color
  glDisable(GL_TEXTURE_2D);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  gluOrtho2D(0.0, 1200, 0.0, 800);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glColor4f(color.x,color.y, color.z, 1.f);          // set text color
  glRasterPos3f(pos.x, pos.y, pos.z);        // place text position

  if(!font)
    font =  GLUT_BITMAP_8_BY_13;
    // loop all characters in the string
    while(*str)
    {
      glutBitmapCharacter(font, *str);
      ++str;
    }

  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
    
  glEnable(GL_TEXTURE_2D);
  glEnable(GL_LIGHTING);
  glPopAttrib();
}

void AppWindow::drawAxes(void) {
  // Save current state of OpenGL
  glPushAttrib(GL_ALL_ATTRIB_BITS);

  glDisable(GL_LIGHTING);
  glLineWidth(3);
  glPushMatrix();
  glScaled(5.0,5.0,5.0);
  glBegin(GL_LINES);
  glColor4f(1.f,0.5f,0.5f,1.f); glVertex3d(0.f,0.f,0.f); glVertex3d(1.f,0.f,0.f);
  glColor4f(0.5f,1.f,0.5f,1.f); glVertex3d(0.f,0.f,0.f); glVertex3d(0.f,1.f,0.f);
  glColor4f(0.5f,0.5f,1.f,1.f); glVertex3d(0.f,0.f,0.f); glVertex3d(0.f,0.f,1.f);

  glColor4f(0.5f,0.5f,0.5f,1.f);
  glVertex3d(0,0,0); glVertex3d(-1,0,0);
  glVertex3d(0,0,0); glVertex3d(0,-1,0);
  glVertex3d(0,0,0); glVertex3d(0,0,-1);

  glEnd();
  glPopMatrix();

  glPopAttrib();
}

void AppWindow::drawSourcesAndReceivers() {
  // Save current state of OpenGL
  glPushAttrib(GL_ALL_ATTRIB_BITS);

  // This is to draw the axes when the mouse button is down
  glDisable(GL_LIGHTING);
  GLUquadricObj* Sphere;

  for(unsigned int i = 0; i < currentApp->m_parameters.getNumReceivers(); i++) {
    float x = currentApp->m_parameters.getReceiver(i).getP().x;
    float y = currentApp->m_parameters.getReceiver(i).getP().y;
    float z = currentApp->m_parameters.getReceiver(i).getP().z;
    //float r = 0.15f;
    float r = currentApp->m_parameters.getDx();
    glColor4f(0.f, 1.f, 0.f, 1.f);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();   
    glTranslatef(x,y,z);
    Sphere=gluNewQuadric();
    gluSphere(Sphere,r,20,20);
    gluDeleteQuadric(Sphere);
    glPopMatrix();
  }

  for(unsigned int i = 0; i < currentApp->m_parameters.getNumSources(); i++) {
    float x = currentApp->m_parameters.getSource(i).getP().x;
    float y = currentApp->m_parameters.getSource(i).getP().y;
    float z = currentApp->m_parameters.getSource(i).getP().z;
    float r = currentApp->m_parameters.getDx();
    glColor4f(1.f, 0.f, 0.f, 1.f);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();   
    glTranslatef(x,y,z);
    Sphere=gluNewQuadric();
    gluSphere(Sphere,r,20,20);
    gluDeleteQuadric(Sphere);
    glPopMatrix();
  }

  glPopAttrib();
}

void AppWindow::drawGeometry() {
  glColor4f(0.9f, 0.9f,0.9f, 0.3f);
  this->vbo_vector_[0].draw(GL_TRIANGLES, ((GeometryHandler*)this->data_)->getNumberOfTriangles()*3);
  glLineWidth(2);
  glColor4f(0.5f, 0.5f, 0.5f, 0.05f);
  glPolygonMode ( GL_FRONT_AND_BACK, GL_LINE ) ;
  this->vbo_vector_[0].draw(GL_TRIANGLES, ((GeometryHandler*)this->data_)->getNumberOfTriangles()*3);
  glPolygonMode ( GL_FRONT_AND_BACK, GL_FILL ) ;
}

void AppWindow::drawMeshVbo() {
  this->vbo_vector_[1].draw(GL_POINTS, currentApp->m_mesh.getNumberOfElements());
}

void AppWindow::drawPbos() {
  if(this->window_params_.draw_xy)
    this->pbo_vector_.at(0).draw((float)this->window_params_.current_z);
  if(this->window_params_.draw_xz)
    this->pbo_vector_.at(1).draw((float)this->window_params_.current_y);
  if(this->window_params_.draw_yz)
    this->pbo_vector_.at(2).draw((float)this->window_params_.current_x);
}

void AppWindow::drawStatusText() {
  float top = 780;
  float increment = 15;
  stringstream ss;
  ss<<"Number of Elements: "<<currentApp->m_mesh.getNumberOfElements()<<", Dx: "<<currentApp->m_parameters.getDx()*100<<" cm, Dt: "<<1.f/(float)currentApp->m_parameters.getSpatialFs()*1000<<" ms";
  drawString3D(ss.str().c_str(), nv::Vec3f(10,top,0), nv::Vec3f(0,0,0), NULL);
  ss.str(std::string());
  top -=increment;
  ss<<"Spatial fs: "<<currentApp->m_parameters.getSpatialFs();
  drawString3D(ss.str().c_str(), nv::Vec3f(10,top,0), nv::Vec3f(0,0,0), NULL);
  ss.str(std::string());
  top -=increment;
  ss<<"Update type: "<<schemes[currentApp->m_parameters.getUpdateType()];
  drawString3D(ss.str().c_str(), nv::Vec3f(10,top,0), nv::Vec3f(0,0,0), NULL);
  ss.str(std::string());
  top -=increment;
  ss<<"Step: "<<currentApp->current_step_;
  drawString3D(ss.str().c_str(), nv::Vec3f(10,top,0), nv::Vec3f(0,0,0), NULL);
  ss.str(std::string());
  top -=increment;
  ss<<"Time: "<<(float)(currentApp->current_step_)/(float)currentApp->m_parameters.getSpatialFs()*1000<<" ms";
  drawString3D(ss.str().c_str(), nv::Vec3f(10,top,0), nv::Vec3f(0,0,0), NULL);
  ss.str(std::string());
  top -=increment;
  ss<<"Calculation time per step: "<<(float)currentApp->getTimePerStep()*1000<<" ms";
  drawString3D(ss.str().c_str(), nv::Vec3f(10,top,0), nv::Vec3f(0,0,0), NULL);
  ss.str(std::string());
  top -=increment;
  ss<<"Calculation time for 1 s: "<<(float)currentApp->getTimePerStep()*(float)currentApp->m_parameters.getSpatialFs()<<" s";
  drawString3D(ss.str().c_str(), nv::Vec3f(10,top,0), nv::Vec3f(0,0,0), NULL);
  if(this->window_params_.draw_xy) {
    ss.str(std::string());
    top -=increment;
    ss<<"z slice: "<<this->window_params_.current_z<<", position: "<<this->window_params_.current_z*currentApp->m_parameters.getDx();
    drawString3D(ss.str().c_str(), nv::Vec3f(10,top,0), nv::Vec3f(0,0,0), NULL);
  }
  if(this->window_params_.draw_xz) {
    ss.str(std::string());
    top -=increment;
    ss<<"y slice: "<<this->window_params_.current_y<<", position: "<<this->window_params_.current_y*currentApp->m_parameters.getDx();
    drawString3D(ss.str().c_str(), nv::Vec3f(10,top,0), nv::Vec3f(0,0,0), NULL);
  }
  if(this->window_params_.draw_yz) {
    ss.str(std::string());
    top -=increment;
    ss<<"x slice: "<<this->window_params_.current_x<<", position: "<<this->window_params_.current_x*currentApp->m_parameters.getDx()<<" m";
    drawString3D(ss.str().c_str(), nv::Vec3f(10,top,0), nv::Vec3f(0,0,0), NULL);
  }
  ss.str(std::string());
  top -=increment;
  ss<<"press 'h' for help";
  drawString3D(ss.str().c_str(), nv::Vec3f(10,top,0), nv::Vec3f(0,0,0), NULL);

}

void AppWindow::drawHelp() {
  float top = 780;
  float right = 850;
  float increment = 15;
  stringstream ss;
  ss<<"p: run / stop";
  drawString3D(ss.str().c_str(), nv::Vec3f(right,top,0), nv::Vec3f(0,0,0), NULL);
  ss.str(std::string());
  top -=increment;
  ss<<"s: run step by step / continuously";
  drawString3D(ss.str().c_str(), nv::Vec3f(right,top,0), nv::Vec3f(0,0,0), NULL);
  ss.str(std::string());
  top -=increment;
  ss<<"r: reset simulation";
  drawString3D(ss.str().c_str(), nv::Vec3f(right,top,0), nv::Vec3f(0,0,0), NULL);
  ss.str(std::string());
  top -=increment;
  ss<<"x / y / z: show/hide x-, y-, z-slice";
  drawString3D(ss.str().c_str(), nv::Vec3f(right,top,0), nv::Vec3f(0,0,0), NULL);
  ss.str(std::string());
  top -=increment;
  ss<<"8 / 2: z-slice up / down";
  drawString3D(ss.str().c_str(), nv::Vec3f(right,top,0), nv::Vec3f(0,0,0), NULL);
  ss.str(std::string());
  top -=increment;
  ss<<"4 / 6: x-slice left / right";
  drawString3D(ss.str().c_str(), nv::Vec3f(right,top,0), nv::Vec3f(0,0,0), NULL);
  ss.str(std::string());
  top -=increment;
  ss<<"1 / 9: y-slice left / right";
  drawString3D(ss.str().c_str(), nv::Vec3f(right,top,0), nv::Vec3f(0,0,0), NULL);
  ss.str(std::string());
  top -=increment;
  ss<<"g: show/hide geometry";
  drawString3D(ss.str().c_str(), nv::Vec3f(right,top,0), nv::Vec3f(0,0,0), NULL);
  ss.str(std::string());
}
