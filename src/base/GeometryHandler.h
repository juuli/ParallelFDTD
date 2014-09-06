#ifndef GEOMETRY_HANDLER_H
#define GEOMETRY_HANDLER_H

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
#include <vector>
#include <map>
#include <string>

///////////////////////////////////////////////////////////////////////////////
/// \brief Class that manages the geometry of the model
///////////////////////////////////////////////////////////////////////////////
class GeometryHandler {
public:
  GeometryHandler()
  : longest_edge_(0.f)
  {}
  ~GeometryHandler() {}

  /////////////////////////////////////////////////////////////////////////////
  /// \brief Initialize the geometry from vector datatype
  /// \param indices The indices specifying the triangles of the model.
  /// Only triangles are accepted!
  /// \param vertices The vertices of the model
  /////////////////////////////////////////////////////////////////////////////
  void initialize(std::vector<unsigned int> indices, std::vector<float> vertices);
  
  /////////////////////////////////////////////////////////////////////////////
  /// \brief Initialize the geometry from raw datatypes
  /// \param indices The indices specifying the triangles of the model
  /// \param vertices The vertices of the model
  /// \param number_of_indices The number of triangles indices in the model.
  /// The number of triangles is number_of_indices/3
  /// \param number_of_vertices The number of values in the vertex list.
  /// The number of vertices is number_of_vertices/3
  /////////////////////////////////////////////////////////////////////////////
  void initialize(unsigned int* indices, float* vertices,
                  unsigned int number_of_indices,
                  unsigned int number_of_vertces);

  /////////////////////////////////////////////////////////////////////////////
  /// \brief Rotate all the vertices of the model around z-axis
  /// \param angle The rotation angle in degrees
  /////////////////////////////////////////////////////////////////////////////
  void rotateGeometryAzimuth(float angle);
  
  /////////////////////////////////////////////////////////////////////////////
  /// \brief Rotate all the vertices of the model around y-axis NOT ELEVATION
  /// \param angle The rotation angle in degrees
  /////////////////////////////////////////////////////////////////////////////
  void rotateGeometryElevation(float angle);

  unsigned int* getIndexPtr() {return &indices_[0];}
  
  /// \brief Get a pointer to a triangle indice
  /// \param idx the index of the surface in the model
  /// \return A pointer to the first indice of the triangle
  unsigned int* getTriangleAt(unsigned int idx) {return &indices_[idx*3];}

  ///\return a pointer to the beginning of the vertex list
  float* getVerticePtr() {return &vertices_[0];}
  
  ///\param idx An index of the vertice in the vertex list
  ///\return a pointer to the first coordinate of the vertex in the list
  float* getVertexAt(unsigned int idx) {return &vertices_[idx*3];}

  /// \return number of indices definen triangles in the model
  unsigned int getNumberOfIndices() {return (unsigned int)indices_.size();}
  /// \return The number of triangles in the model (number of indices / 3)
  unsigned int getNumberOfTriangles() { return (unsigned int)indices_.size()/3;}
  /// \return The number of 3-D vertices in the model
  unsigned int getNumberOfVertices() {return (unsigned int)vertices_.size()/3;}

  /// \return The number of layers in the model
  unsigned int getNumberOfLayers() {return (unsigned int)layers_.size();}

  /// \return The name of the layer from the begining of the continaer
  ///         return an empty string if out of bounds
  std::string getLayerNameAt(int idx) {
    std::string ret = "";
    std::map< std::string, std::vector<int> >::iterator it = this->layers_.begin();
    if(idx < (int)this->getNumberOfLayers()) { std::advance(it, idx); ret = it->first;}
    return ret;
  }

  /// \brief Function to calculate horw many voxels will fit to the longest dimension
  /// of the geometry
  /// \param dx The length of the voxel edge
  /// \return Number of voxels that fit
  unsigned int getNumberOfLongEdgeNodes(float dx) {
    return (unsigned int)(this->longest_edge_/dx+0.5f);}
  
  ///\return the maximum coordinates of the bounding box. Minimum is set to 0,0,0
  nv::Vec3f getBoundingBox() {
    nv::Vec3f box = this->bounding_box_max_;
    return nv::Vec3f(box.x, box.y, box.z);}
    
  /// \param idx The index of the surface which area is of interest
  /// \return The area of surface at idx
  float getSurfaceAreaAt(unsigned int idx);
  
  /// \return the total surface area of the model
  float getTotalSurfaceArea();
  
  nv::Vec3f getGeometryOffset() {return this->offset_;}
  
  void setLayerIndices(std::vector<int> indices, std::string name); 

private:
  std::vector<unsigned int> indices_;              ///< Triangle indices of the model
  std::vector<float> vertices_;                    ///< Vertex coordinates of the model
  std::map< std::string, std::vector<int> > layers_; ///< Layers containig the name and list of triangles in the layer
  nv::Vec3f bounding_box_max_;                     ///< Minimum coordinates of the bounding box
  nv::Vec3f bounding_box_min_;                     ///< Maximum coordinates of the bounding box
  nv::Vec3f offset_;
  float longest_edge_;                ///< Longest edge of the model
  void setVertexAt(unsigned int i, float x, float  y, float z);
};

#endif
