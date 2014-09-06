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

#ifndef APP_VBO_H
#define APP_VBO_H

#include "glHelpers.h"

#include <vector>

//// Class to handle VBOs
// This Class is largely been influenced by the implementation of ofVBO of
// Openframeworks http://www.openframeworks.cc/ | https://github.com/openframeworks
// 
// 

class AppVbo {
public:
  AppVbo() 
  : buffer_size_(0),
    num_coords_(0),
    index_id_(0),
    vertex_id_(0),
    color_id_(0),
    normal_id_(0),
    texture_id_(0),
    allocated_(false),
    using_verts_(false),
    using_colors_(false),
    using_textures_(false),
    using_normals_(false),
    using_indices_(false),
    color_type_(0)
  {};

  ~AppVbo() {};
  
  void setVertexData(float* data, unsigned int num_coords, unsigned int num_verts, int usage);
  void setColorData(float* data, int usage);
  void setColorData(unsigned char* data, int usage);
  void setEmptyColorData(int usage);
  void setIndexData(unsigned int* data, int total, int usage);
  void setTextureData(float* data, int total, int usage, int strinde);
  void updateTextureData(float* data, int total);
  
  void enableColors() {this->using_colors_ = true;};
  void enableNormals() {this->using_normals_ = true;};
  void enableTextures() {this->using_textures_ = true;};

  void disableColors() {this->using_textures_ = false;};
  void disableNormals() {this->using_textures_ = false;};
  void disableTextures() {this->using_textures_ = false;};

  // Getters
  GLuint getVertId() {return this->vertex_id_;};
  GLuint getColorId() {return this->color_id_;};
  GLuint getNormalId() {return this->normal_id_;};
  GLuint getTexCoordId() {return this->texture_id_;};
  GLuint getIndexId() {return this->index_id_;};

  // Draw Functions
  void bindVbo();
  void unBindVbo();
  void clearVert();
  void clearColor();
  void clearTextures(); 
  void clear();
  void draw(int drawMode, unsigned int num);

private:
  unsigned int buffer_size_;
  unsigned int num_coords_;
  unsigned int num_indices_;

  GLuint index_id_;
  GLuint vertex_id_;
  GLuint color_id_;
  GLuint normal_id_;
  GLuint texture_id_;

  bool allocated_;

  bool using_verts_;
  bool using_colors_;
  bool using_textures_;
  bool using_normals_;
  bool using_indices_;

  unsigned color_type_;
};


#endif
