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

#include "../global_includes.h"
#include "FileReader.h"

#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdio.h>

// Helper funciton
void FileReader::printCount(std::string type, bool reset) {
  log_msg<LOG_INFO>(L"FileReader %d %s values read") %this->counter %type.c_str();

  if(reset)
    this->counter = 0;
}

bool FileReader::readVTK(GeometryHandler* gh, std::string vtk_fp, float truncate_vertices_to) {
  bool ret = true;

  std::fstream inFile;
  inFile.open(vtk_fp.c_str() , std::fstream::in | std::fstream::out);

  if(!inFile.good()) {
    log_msg<LOG_ERROR>(L"FileReader::readVTK - Invalid Geomtery file %s") %vtk_fp.c_str();
    return false;
  }

  std::stringstream ss;

  
  unsigned int temp_uint;
  unsigned int temp_uint_val;
  float temp_float;
  std::string temp_string;
  std::string new_line;
  
  std::vector<float> vertices;
  std::vector<unsigned int> indices;

  int count = 0; 

  while(count < 20) {
    std::getline(inFile, new_line, '\n');
    new_line.append(" "); // append with whitespace to get the string stream running  
    ss.str(new_line);
    std::getline(ss, temp_string, ' ');

    if(temp_string == "POINTS") {
      ss >> temp_uint;
      vertices.assign(temp_uint*3, 0.f);
      //std::cout<<temp_uint<<std::endl;
      for(unsigned int i = 0; i < temp_uint*3; i++) {
        inFile>>temp_float;
        temp_float = nv::ROUND((temp_float*0.0254f)/truncate_vertices_to);
        temp_float = temp_float*truncate_vertices_to;
        vertices.at(i) = temp_float;
      }
    }

    if(temp_string == "POLYGONS") {
      ss >> temp_uint;
      //std::cout<<temp_uint<<std::endl;
      indices.assign(temp_uint*3, 0);
      for(unsigned int i = 0; i < temp_uint*3; i+=3) {
        unsigned int i1, i2, i3;
        inFile>>temp_uint_val>>i1>>i2>>i3;
        indices.at(i) = i1;
        indices.at(i+1) = i2;
        indices.at(i+2) = i3;
      }

    }
    temp_string = "";
    new_line = "";
    count++;
  }

  gh->initialize(indices, vertices);
  return ret;
}

std::vector<float> FileReader::readFloat(std::string fp) {
  float temp_float;
  std::vector<float> ret; 
  std::fstream inFile(fp.c_str() , std::fstream::in | std::fstream::out);
  
  if(!inFile.good()) {
    log_msg<LOG_DEBUG>(L"FileReade::readFloat: Invalid file %ss") % fp.c_str();
    return ret;
  }
  
  while(true) {
    inFile>>temp_float;

    if(!inFile.good())
      break;
    ret.push_back(temp_float);
  }

  log_msg<LOG_DEBUG>(L"FileReade::readFloat: %d floats read") % ret.size();
  inFile.close();
  return ret;
}
