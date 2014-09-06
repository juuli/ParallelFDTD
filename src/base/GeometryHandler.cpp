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

#include "GeometryHandler.h"
#include "../global_includes.h"

void GeometryHandler::initialize(std::vector<unsigned int> indices, std::vector<float> vertices) {
  size_t indices_size = indices.size();
  size_t vertices_size = vertices.size();
  size_t indices_mod = indices_size%3;
  size_t vertices_mod = vertices_size%3;

  if(indices_mod != 0)
    log_msg<LOG_WARNING>(L"GeometryHandler::initialize - index vector not multiple of 3");

  if(vertices_mod != 0)
    log_msg<LOG_WARNING>(L"GeometryHandler::initialize - vertice vector not multiple of 3");

  this->indices_ = indices;
  this->vertices_ = vertices;

  log_msg<LOG_INFO>(
    L"GeometryHandler::initialize - Handler initialized with %d indices %d vertices %d triangles")
    % this->getNumberOfIndices() % this->getNumberOfVertices() %this->getNumberOfTriangles();

  // Search bounding box
  float max_x, max_y, max_z;
  float min_x, min_y, min_z;
  max_x = max_y = max_z = -9999999999.f; 
  min_x = min_y = min_z = 9999999999.f;

  for(unsigned int i = 0; i < this->getNumberOfVertices(); i++) {
    float current_x = *(this->getVertexAt(i));
    float current_y = *(this->getVertexAt(i)+1);
    float current_z = *(this->getVertexAt(i)+2);

    if(current_x < min_x) {min_x = current_x;}
    if(current_x > max_x) {max_x = current_x;}
    if(current_y < min_y) {min_y = current_y;}
    if(current_y > max_y) {max_y = current_y;}
    if(current_z < min_z) {min_z = current_z;}
    if(current_z > max_z) {max_z = current_z;}
  }
  this->bounding_box_max_.set(max_x, max_y, max_z);
  this->bounding_box_min_.set(min_x, min_y, min_z);
  this->offset_.set(min_x, min_y, min_z);
  // Get rid of the offset
  for(unsigned int i = 0; i < this->getNumberOfVertices(); i++) {
    *(this->getVertexAt(i)) -= bounding_box_min_.x;
    *(this->getVertexAt(i)+1) -= bounding_box_min_.y;
    *(this->getVertexAt(i)+2) -= bounding_box_min_.z;
  }
  
  this->bounding_box_max_ -= this->bounding_box_min_;
  this->bounding_box_min_ -= this->bounding_box_min_;

  nv::Vec3f dif = this->bounding_box_max_ - this->bounding_box_min_;
  this->longest_edge_ = MAX((MAX(dif.x, dif.y)), (dif.z));

  log_msg<LOG_INFO>(L"GeometryHandler::initialize - bounding box %f %f %f , %f %f %f ")
                    % min_x %min_y %min_z %max_x %max_y %max_z ;
  log_msg<LOG_INFO>(L"GeometryHandler::initialize - geometry offset %f %f %f ")
                    % this->offset_.x % this->offset_.y % this->offset_.z;
  log_msg<LOG_INFO>(L"GeometryHandler::initialize - longest edge %f") 
                    %this->longest_edge_ ;
}

void GeometryHandler::initialize(unsigned int* indices, float* vertices,
                                 unsigned int number_of_indices,
                                 unsigned int number_of_vertices) {
  std::vector< unsigned int > indice_vector;
  indice_vector.assign(number_of_indices, 0);

  std::vector< float > vertice_vector;
  vertice_vector.assign(number_of_vertices, 0.f);

  log_msg<LOG_INFO>
  (L"GeometryHandler::initialize from pointers - vertices: %d indices: %d")
  %number_of_vertices 
  %number_of_indices;


  for(unsigned int i = 0; i < number_of_indices; i++) {
    indice_vector.at(i) = indices[i];
  }

  for(unsigned int i = 0; i < number_of_vertices; i++){
    vertice_vector.at(i) = vertices[i];
  }

  log_msg<LOG_INFO>
  (L"GeometryHandler::initialize from pointers - vectors done, sizes %d, %d")
  %indice_vector.size()
  %vertice_vector.size();

  initialize(indice_vector, vertice_vector);
}

void GeometryHandler::setVertexAt(unsigned int i, float x, float  y, float z) {
  float* vert = this->getVertexAt(i);
  vert[0] = x;
  vert[1] = y;
  vert[2] = z;
}

void GeometryHandler::rotateGeometryAzimuth(float angle) {
  log_msg<LOG_INFO>(L"GeometryHandler::rotateGeometryAzimuth - rotate %f degrees") %angle;
  float rad = angle/180.f*(float)PI;
  for(unsigned int i = 0; i < this->getNumberOfVertices(); i++) {
    float* vert = this->getVertexAt(i);
    float x = vert[0];
    float y = vert[1];
    vert[0] = x*cosf(rad)-y*sinf(rad);
    vert[1] = x*sinf(rad)+y*cosf(rad);
  }

}

void GeometryHandler::rotateGeometryElevation(float angle) {
  log_msg<LOG_INFO>(L"GeometryHandler::rotateGeometryElevation - rotate %f degrees") %angle;
  float rad = angle/180.f*(float)PI;
  for(unsigned int i = 0; i < this->getNumberOfVertices(); i++) {
    float* vert = this->getVertexAt(i);
    float x = vert[0];
    float z = vert[2];
    vert[0] = z*cosf(rad)-x*sinf(rad);
    vert[1] = z*sinf(rad)+x*cosf(rad);
  }

}

float GeometryHandler::getSurfaceAreaAt(unsigned int idx) {
  float ret = 0.f;
  unsigned int* tri = getTriangleAt(idx);
  float* vert_0 = getVertexAt(tri[0]);
  float* vert_1 = getVertexAt(tri[1]);
  float* vert_2 = getVertexAt(tri[2]);

  nv::Vec3f u(vert_0[0]-vert_1[0],
            vert_0[1]-vert_1[1],
            vert_0[2]-vert_1[2]);
  nv::Vec3f v(vert_0[0]-vert_2[0],
            vert_0[1]-vert_2[1],
            vert_0[2]-vert_2[2]);

  ret = nv::length_(u.crossed(v))*0.5f;
  return ret;
}

float GeometryHandler::getTotalSurfaceArea() {
  float total_area = 0.f;
  for(unsigned int i = 0; i < this->getNumberOfTriangles(); i++) {
    total_area += getSurfaceAreaAt(i);
  }
  return total_area;
}

void GeometryHandler::setLayerIndices(std::vector<int> indices, std::string name) {
  int num_indices = (int)indices.size();
  int min = 99999999;
  int max = -99999999;
  for(int i = 0; i < num_indices; i++) {
    if(indices.at(i)<min)
      min = indices.at(i);
    if(indices.at(i)>max)
      max = indices.at(i);
  }
  log_msg<LOG_INFO>(L"GeometryHandler::setLayerIndices - name: %s, number of indices: %d , min val: %d max mal: %d") 
                    %name.c_str() %num_indices %min %max;

  if(max >= this->getNumberOfTriangles()) {
    log_msg<LOG_INFO>(L"GeometryHandler::setLayerIndices - index %d larger than number of triangles %d, returning") 
                      %max %this->getNumberOfTriangles();
    return;
  }    
  if(min < 0) {
    log_msg<LOG_INFO>(L"GeometryHandler::setLayerIndices - index less than zero, returning");
    return;
  }

  this->layers_[name] = indices;
}

