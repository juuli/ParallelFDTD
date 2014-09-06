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

#include "AppVbo.h"
#include "glHelpers.h"
#include <assert.h>
#include <iostream>

void AppVbo::setVertexData(float* data, unsigned int num_coords, unsigned int num_verts, int usage) {
  if(this->vertex_id_==0) {
    this->allocated_  = true;
    this->using_verts_ = true;
    glGenBuffers(1, &(this->vertex_id_));
    glsafe(glGetError(), "AppVbo: setVertexData: ");
  }

  this->buffer_size_ = num_verts*num_coords;
  this->num_coords_ = num_coords;

  
  glBindBuffer(GL_ARRAY_BUFFER,  this->vertex_id_);
  glsafe(glGetError(), "AppVBO::setVertexData::glBindBuffer");
  
  if(data)
    glBufferData(GL_ARRAY_BUFFER, this->buffer_size_*sizeof(float), data, usage);
  else
    glBufferData(GL_ARRAY_BUFFER, this->buffer_size_*sizeof(float), 0, usage);
  glsafe(glGetError(), "AppVBO::setColorData::glBufferData");
  
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Set the color buffer, here it is assumed that 4 coordinates are used
void AppVbo::setColorData(GLubyte* data,  int usage) {
  if(!this->allocated_) {
    std::cout<<"Cannot assign color to VBO: No vertices allocated"<<std::endl;
    return;
  }

  if(this->color_id_ == 0) {
    this->using_colors_ = true;
    glGenBuffers(1, &(this->color_id_));
    glsafe(glGetError(), "AppVbo: setColorData: ");
  }

  this->color_type_ = 0;

  glBindBuffer(GL_ARRAY_BUFFER, this->color_id_);
  
  if(data)
    glBufferData(GL_ARRAY_BUFFER, this->buffer_size_*sizeof(unsigned char), data, usage);
  else
    glBufferData(GL_ARRAY_BUFFER, this->buffer_size_*sizeof(unsigned char), 0, usage);

  glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void AppVbo::setColorData(float* data,  int usage) {
  if(!this->allocated_) {
    std::cout<<"Cannot assign color to VBO: No vertices allocated"<<std::endl;
    return;
  }

  if(this->color_id_ == 0) {
    this->using_colors_ = true;
    glGenBuffers(1, &(this->color_id_));
  }
  this->color_type_ = 1;

  glBindBuffer(GL_ARRAY_BUFFER, this->color_id_);
  
  // If no data is given, make an empty buffer
  if(data)
    glBufferData(GL_ARRAY_BUFFER, this->buffer_size_*sizeof(float), data, usage);
  else
    glBufferData(GL_ARRAY_BUFFER, this->buffer_size_*sizeof(float), 0, usage);
  
  glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void AppVbo::setEmptyColorData(int usage) {
  if(!this->allocated_) {
    std::cout<<"Cannot assign color to VBO: No vertices allocated"<<std::endl;
    return;
  }

  if(this->color_id_ == 0) {
    this->using_colors_ = true;
    glGenBuffers(1, &(this->color_id_));
  }

  glBindBuffer(GL_ARRAY_BUFFER, this->color_id_);
  glBufferData(GL_ARRAY_BUFFER, this->buffer_size_*sizeof(unsigned char), 0, usage);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void AppVbo::setIndexData(unsigned int* data, int total, int usage) {
  if(!this->allocated_) {
    std::cout<<"Cannot assign index to VBO: No vertices allocated"<<std::endl;
    return;
  }

  if(this->index_id_ == 0) {
    this->using_indices_ = true;
    glGenBuffers(1, &(this->index_id_));
  }

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->index_id_);
  
  // If no data is given, make an empty buffer

  glBufferData(GL_ELEMENT_ARRAY_BUFFER, total*sizeof(unsigned int), data, usage);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

}

void AppVbo::setTextureData(float* data, int total, int usage, int stride) {
  if(!this->allocated_) {
    std::cout<<"Cannot assign texture to VBO: No vertices allocated"<<std::endl;
    return;
  }
  if(this->texture_id_ == 0) {
    this->using_textures_ = true;
    glGenBuffers(1, &(this->texture_id_));
  }

  glBindBuffer(GL_ARRAY_BUFFER, this->getTexCoordId());
  glBufferData(GL_ARRAY_BUFFER, total*stride, data, usage);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void AppVbo::updateTextureData(float* data, int total) {
  if(this->texture_id_ != 0) {
    glBindBuffer(GL_ARRAY_BUFFER, this->texture_id_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, total, data);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }
}

void AppVbo::bindVbo() {
  if(this->allocated_) {
    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, this->vertex_id_); 
    glVertexPointer(4, GL_FLOAT, 0, 0);
  }

  if(this->using_colors_) {
    glEnableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, this->color_id_); 

    if(this->color_type_ == 0)
      glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0);
    if(this->color_type_ == 1)
      glColorPointer(4, GL_FLOAT, 0, 0);
    }

  if(this->using_textures_) {
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, this->texture_id_);
    glTexCoordPointer(2, GL_FLOAT, 0 ,0);
  }
}

void AppVbo::unBindVbo() {
  if(this->allocated_) 
    glDisableClientState(GL_VERTEX_ARRAY);

  if(this->using_colors_) 
    glDisableClientState(GL_COLOR_ARRAY);

  if(this->texture_id_)
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);

  glsafe(glGetError(), "AppVBO::Unbind");
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

}

void AppVbo::clearVert(){
  if(vertex_id_ != 0)
    glDeleteBuffers(1, &vertex_id_);
  vertex_id_ = 0;
  using_verts_ = false;
}

void AppVbo::clearColor() {
  if(color_id_ != 0)
    glDeleteBuffers(1, &color_id_);
  color_id_ = 0;
  using_colors_ = false;
}

void AppVbo::clearTextures() {
  if(texture_id_ != 0)
    glDeleteBuffers(1, &texture_id_);
  texture_id_ = 0;
  using_textures_ = false;
}

void AppVbo::clear() {
  clearVert();
  clearColor();
  clearTextures();
}

void AppVbo::draw(int drawMode, unsigned int num) {
  if(this->allocated_){
    this->bindVbo();
    if(this->using_indices_) {
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->index_id_);
      glDrawElements(drawMode, num, GL_UNSIGNED_INT, NULL);
    }
    else{
      glDrawArrays(drawMode, 0, this->buffer_size_);
    }

    glsafe(glGetError(), "AppVBO::draw");
    this->unBindVbo();
  }
}
