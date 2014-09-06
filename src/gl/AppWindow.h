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

#ifndef APP_WINDOW_H
#define APP_WINDOW_H

// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <string>
#include <iostream>
#include "../base/cameraProto.hpp"
#include "../math/geomMath.h"
#include "AppVbo.h"
#include "AppPbo.h"

// Forward declaration
namespace FDTD {
  class App;
}
struct WindowParameters {
  int m_button;
  unsigned char key;
  bool m_down;
  bool k_down;
  bool shift;

  int mouse_x;
  int mouse_y;
  int p_mouse_x;
  int p_mouse_y;

  bool execution_pause;
  bool step;
  bool mesh_reset;
  bool invert_time;
  bool draw_geometry;
  bool draw_xy;
  bool draw_xz;
  bool draw_yz;
  bool help;
  unsigned int current_z;
  unsigned int current_y;
  unsigned int current_x;
  unsigned int selection;
};

class AppWindow {
public: 
  AppWindow() 
  : camera_(10.f, nv::Vec4f(0.f,0.f,0.f,0.f), nv::Vec4f(0,0,1,0)),
    vbo_vector_(),
    slice_orientation_(0),
    slice_dim_(),
    current_z_(0),
    pbo_w_(0),
    pbo_h_(0),
    window_id_(0)
    {};

  ~AppWindow() {};

  /// \brief Initializes the GL context
  /// \param argc, argv command line arguments from the main function or mockup ones
  void initializeGL(int argc, char **argv);
  
  /// \brief Initializes a GL window
  /// \param w, h Width and Height of the window
  void initializeWindow(int w, int h);
  
  /// \brief Starts the GL main loop. initializeGL and initializeWindow have to be called before
  /// \param currentApp Pointer to the current App instance. 
  void startMainLoop(FDTD::App* currentApp);

  // Callbacks
  void draw();
  void mouse(int button, int state, int x, int y);
  void keyboardDown(unsigned char key, int x, int y);
  void keyboardUp(unsigned char key, int x, int y);
  void motion(int x, int y);
  void close();

  // Vertex buffer handling
 
  void geometryToVbo(void* data);
  void createEmptyVbo(unsigned int num_coords, unsigned int num_verts);
  void createPixelBuffer(unsigned int dim_x, unsigned int dim_y, nv::Vec3f slice_dim);
  void addPixelBuffer(unsigned int dim_x, unsigned int dim_y,
                      nv::Vec3f u, nv::Vec3f v, nv::Vec3f offset);
  void deleteVbo(unsigned int idx);

  // Getters
  GLuint getVboVerticeId(unsigned int idx);
  GLuint getVboColorId(unsigned int idx);
  GLuint getPboIdAt(unsigned int idx);

  // Setters
  void setData(void* _data);

  // RAII Fail for python binding, do not copy the class in Python!
  AppWindow(const AppWindow& other) {};
  AppWindow& operator=(const AppWindow& other) {};
private:
  WindowParameters window_params_; 
  cameraProto camera_;
  int window_id_;
  void* data_;

  std::vector<AppVbo> vbo_vector_;  
  std::vector<AppPbo> pbo_vector_;

  nv::Vec3f slice_dim_;
  unsigned int slice_orientation_;
  unsigned int current_z_;
  unsigned int pbo_w_;
  unsigned int pbo_h_;

  // Setup function for callbacks
  
  /// \brief Setup the callback functions of the GL context
  void setupCallbacks();
  
  /// \brief reset the window parameters of the AppWindow instance
  void resetParameters() {
    window_params_.m_button = -1;
    window_params_.key = ' ';
    window_params_.m_down = false;
    window_params_.k_down = false;
    window_params_.mouse_x = 0;
    window_params_.mouse_y = 0;
    window_params_.p_mouse_x = 0;
    window_params_.p_mouse_y = 0;
    window_params_.execution_pause = true;
    window_params_.step = true;
    window_params_.mesh_reset = false;
    window_params_.invert_time = false;
    window_params_.draw_geometry = true;
    window_params_.draw_xy = true;
    window_params_.draw_xz = false;
    window_params_.draw_yz = false;
    window_params_.help = false;
    window_params_.current_z = 0;
    window_params_.current_y = 0;
    window_params_.current_x = 0;
    window_params_.selection = 0;
  }
  
  // Helper draw functions
  void drawAxes();
  void drawSourcesAndReceivers();
  void drawGeometry();
  void drawPbo();
  void drawPbos();
  void drawMeshVbo();
  void drawString3D(const char *str, nv::Vec3f pos, nv::Vec3f color, void *font);
  void drawStatusText();
  void drawHelp();
};


#endif
