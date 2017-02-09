#include "visualizationUtils.h"

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

void registerGLtoCuda(struct cudaGraphicsResource **resource, 
                      GLuint buffer, unsigned int vbo_res_flags) {
  cudasafe(cudaGraphicsGLRegisterBuffer(resource, buffer, vbo_res_flags), "registerGLtoCuda vertex register");
}

// Function which updates the pixel buffer registered to pbo_resource
void updatePixelBuffer(struct cudaGraphicsResource **pbo_resource, 
                       CudaMesh* d_mesh,
                       unsigned int current_slice,
                       unsigned int orientation,
                       unsigned int selector,
                       unsigned int scheme,
                       float dB) {
  uint3 block_ = make_uint3(0,0,0);
  uint3 grid_ = make_uint3(0,0,0);
  uint3 offset = make_uint3(0,0,0);
  unsigned int num_elements = d_mesh->getNumberOfElements();

  // Define the block and grid size according to the slice orientation
  switch(orientation) {
  case 0: // xy
    block_.x = d_mesh->getBlockX(); block_.y = d_mesh->getBlockY(); block_.z = 1;
    grid_.x = d_mesh->getGridDimX(); grid_.y =  d_mesh->getGridDimY(); grid_.z = 1;
    offset.x = 0; offset.y = 0; offset.z = current_slice;
    break;

  case 1: // xz
    block_.x = d_mesh->getBlockX(); block_.y = 1; block_.z = d_mesh->getBlockZ();
    grid_.x = d_mesh->getGridDimX(); grid_.y = 1; grid_.z = d_mesh->getDimZ()/d_mesh->getBlockZ()+1;
    offset.x = 0; offset.y = current_slice; offset.z = 0;
    break;

  case 2: // yz
    block_.x = 1; block_.y = d_mesh->getBlockY(); block_.z = d_mesh->getBlockZ();
    grid_.x = 1; grid_.y = d_mesh->getGridDimY(); grid_.z = d_mesh->getDimZ()/d_mesh->getBlockZ()+1;
    offset.x = current_slice; offset.y = 0; offset.z = 0;
    break;

  }
    
  for(unsigned int i = 0; i < d_mesh->getNumberOfPartitions(); i++) {
    cudasafe(cudaSetDevice(d_mesh->getDeviceAt(i)), "updatePixelBuffer");
    unsigned int dim_x = d_mesh->getDimX();
    unsigned int dim_y = d_mesh->getDimY();
    unsigned int dim_xy = d_mesh->getDimXY();
    unsigned int split = d_mesh->getFirstSliceIdx(i)*dim_xy;
    
    if(current_slice*dim_xy >= split){
      uchar4 *pixel_ptr;
      cudasafe(cudaGraphicsMapResources(1, pbo_resource, 0), "Map cuda pixel resource");
      size_t num_bytes_p;
      
      cudasafe(cudaGraphicsResourceGetMappedPointer((void **)&pixel_ptr, &num_bytes_p,
                                                *pbo_resource), "Get pixel resource pointer");
      
      dim3 block(block_.x, block_.y, block_.z);
      dim3 grid(grid_.x, grid_.y, grid_.z);

      unsigned int slice_offset = i == 0 ? 0 : dim_xy;
      // Launch render kernel for the current partition
      if(selector == 0) {
        float* d_P = d_mesh->getPressurePtrAt(i);
        renderPressuresPBO<<<grid, block>>>(pixel_ptr, d_P+slice_offset, dim_x, 
                         dim_xy, dim_y, num_elements, offset, orientation, dB);

      }
      if(selector == 1) {
        unsigned char* d_K = d_mesh->getPositionIdxPtrAt(i);
        renderPositionsPBO<<<grid, block>>>(pixel_ptr, d_K+slice_offset, dim_x, 
                dim_xy, dim_y, num_elements, offset, orientation, scheme);

      }
      if(selector == 2) {
        unsigned char* d_K = d_mesh->getPositionIdxPtrAt(i);
        renderSwitchPBO<<<grid, block>>>(pixel_ptr, d_K+slice_offset, dim_x, 
                dim_xy, dim_y, num_elements, offset, orientation, scheme);

      }
      if(selector == 3) {
        unsigned char* d_M = d_mesh->getMaterialIdxPtrAt(i);
        renderMaterialsPBO<<<grid, block>>>(pixel_ptr, d_M+slice_offset, dim_x, 
                dim_xy, dim_y, num_elements, offset, orientation, scheme);

      }
    }
  }

  cudasafe(cudaGraphicsUnmapResources(1, pbo_resource, 0), "Unmap pixel resource");
  cudasafe(cudaDeviceSynchronize(), "Device synch after pbo render");
}


void captureMesh(CudaMesh* d_mesh,
                 std::vector<unsigned int> &mesh_to_capture,
                 std::vector<float*> &mesh_captures,
                 unsigned int current_step) {
  for(unsigned int i = 0; i < mesh_to_capture.size(); i++) {
      if(current_step == mesh_to_capture.at(i)) {
        c_log_msg(LOG_INFO, "visualizationUtils.cu: captureMesh - capturing mesh step %d",
              current_step);
        
          float* data = fromDevice(d_mesh->getNumberOfElementsAt(0),
                                   d_mesh->getPressurePtrAt(0),
                                   0);
        
        // this allocation is freed in the app class destructor
        mesh_captures.push_back(data);
      }
    }// end capture loop
}

void captureCurrentSlice(CudaMesh* d_mesh,
                         float** pressure_data,
                         unsigned char** position_data,
                         unsigned int slice_to_capture,
                         unsigned int slice_orientation) {

  uint3 block_ = make_uint3(0,0,0);
  uint3 grid_ = make_uint3(0,0,0);
  uint3 offset = make_uint3(0,0,0);

  unsigned int slice = slice_to_capture;
  unsigned int orientation = slice_orientation;
  unsigned int num_partitions = d_mesh->getNumberOfPartitions();
  unsigned int dim_x = d_mesh->getDimX();
  unsigned int dim_y = d_mesh->getDimY();
  unsigned int dim_z = d_mesh->getDimZ();
  unsigned int dim_xy = d_mesh->getDimXY();

  unsigned int slice_size = 0;
  unsigned int slice_partition_size = 0;
  //float* pressure_data = (float*)NULL;
  //unsigned char* position_data = (unsigned char*)NULL;

  if(orientation == 0 && dim_z < slice) {
    c_log_msg(LOG_INFO, "visualizationUtils.cu: captureSliceFast - "
                        "slice %u out of bounds %u, no capture made", slice, dim_z);
    return;
  }
  if(orientation == 1 && dim_y < slice) {
    c_log_msg(LOG_INFO, "visualizationUtils.cu: captureSliceFast - "
                        "slice %u out of bounds %u, no capture made", slice, dim_y);
    return;
  }
  if(orientation == 2 && dim_x < slice) {
    c_log_msg(LOG_INFO, "visualizationUtils.cu: captureSliceFast - "
                        "slice %u out of bounds %u, no capture made", slice, dim_x);
    return;
  }

  // Define the block and grid size according to the slice orientation
  switch(orientation) {
    case 1: // xz
      block_.x = d_mesh->getBlockX(); block_.y = 1; block_.z = d_mesh->getBlockZ();
      grid_.x = d_mesh->getGridDimX(); grid_.y = 1; grid_.z = d_mesh->getDimZ()/d_mesh->getBlockZ()+1;
      offset.x = 0; offset.y = slice; offset.z = 0;
      slice_size = dim_x*dim_z;
      slice_partition_size = dim_x*(d_mesh->getPartitionSize()-1);
      *pressure_data = (float*)calloc(slice_size, sizeof(float));
      *position_data = (unsigned char*)calloc(slice_size, sizeof(unsigned char));
      break;

    case 2: // yz
      block_.x = 1; block_.y = d_mesh->getBlockY(); block_.z = d_mesh->getBlockZ();
      grid_.x = 1; grid_.y = d_mesh->getGridDimY(); grid_.z = d_mesh->getDimZ()/d_mesh->getBlockZ()+1;
      offset.x = slice; offset.y = 0; offset.z = 0;
      slice_size = dim_y*dim_z;
      slice_partition_size = dim_y*(d_mesh->getPartitionSize()-1);
      *pressure_data = (float*)calloc(slice_size, sizeof(float));
      *position_data = (unsigned char*)calloc(slice_size, sizeof(unsigned char));
      break;
  }// end switch

  // z slice is a simple copy
  if(orientation == 0){
    *pressure_data  = d_mesh->getSlice<float>(slice, orientation);
    *position_data = d_mesh->getPositionSlice(slice, orientation);
  }
  // Other planes have to be composed from different devices
  else {
    // Go through partitions
    for(unsigned int j = 0; j < num_partitions; j++) {
      int dev = d_mesh->getDeviceAt(j);
      cudaSetDevice(dev);
      int partition_size = (int)d_mesh->getPartitionSize(j);
      unsigned int num_elements = d_mesh->getNumberOfElementsAt(j);
 
      // Setup and launch kernel to capture slice
      dim3 block(block_.x, block_.y, block_.z);
      dim3 grid(grid_.x, grid_.y, grid_.z);

      // If partition is not the first one, offset by slice
      unsigned int slice_offset = (j == 0 ? 0 : dim_xy);
      
      // Get the pointers to the partition
      float* d_P = d_mesh->getPressurePtrAt(j);
      unsigned char* d_position_idx = d_mesh->getPositionIdxPtrAt(j);

      // Allocate slices from device
      float* d_capture_P = valueToDevice<float>(slice_size, 0.f, dev);
      unsigned char* d_capture_position = valueToDevice<unsigned char>(slice_size, 0x0, dev);

      captureSliceKernel<<<grid, block>>>(d_P+slice_offset, d_position_idx+slice_offset, 
                                          d_capture_P, d_capture_position, num_elements,
                                          dim_x, dim_y, dim_xy,
                                          offset, slice_partition_size, orientation);
      // Copy the slice data to host   
      copyDeviceToHost<float>(slice_partition_size, 
                              *pressure_data+(j*(slice_partition_size)), 
                              d_capture_P, dev);
      
      copyDeviceToHost<unsigned char>(slice_partition_size, 
                                      *position_data+(j*(slice_partition_size)), 
                                      d_capture_position, dev);

      // Destroy slice from the device
      destroyMem<float>(d_capture_P, dev);
      destroyMem<unsigned char>(d_capture_position, dev);
    }
  }
}

void captureSliceFast(CudaMesh* d_mesh,
                      std::vector<unsigned int> &step_to_capture,
                      std::vector<unsigned int> &slice_to_capture,
                      std::vector<unsigned int> &slice_orientation,
                      unsigned int current_step,
                      void (*captureCallback)(float*, unsigned char*, unsigned int, unsigned int, 
                                              unsigned int, unsigned int, unsigned int)) {
  // Loop through the slices assigned for capturing
  for(unsigned int i = 0; i < step_to_capture.size(); i++) {
    if(current_step == step_to_capture.at(i)) {

      unsigned int slice = slice_to_capture.at(i);
      unsigned int orientation = slice_orientation.at(i);

      c_log_msg(LOG_INFO, "visualizationUtils.cu: captureSliceFast - "
                          "capturing slice %d step %d",
                          slice, current_step);

      unsigned int d_x = 0;
      unsigned int d_y = 0;
      if(orientation == 0) {d_x = d_mesh->getDimX(); d_y = d_mesh->getDimY();};
      if(orientation == 1) {d_x = d_mesh->getDimX(); d_y = d_mesh->getDimZ();};
      if(orientation == 2) {d_x = d_mesh->getDimY(); d_y = d_mesh->getDimZ();};

      float* pressure_data = NULL;
      unsigned char* position_data = NULL;

      captureCurrentSlice(d_mesh, &pressure_data, &position_data, 
                          slice, orientation);

      captureCallback(pressure_data, position_data, d_x, d_y, slice, 
                      orientation, current_step);

      free(pressure_data);
      free(position_data);

    }// end step if
  }// end step loop
}// end function


///////////////////////////////////////////////////////////////////////////////
// Render kernels
///////////////////////////////////////////////////////////////////////////////

// Render the pressure values
__global__ void renderPressuresPBO(uchar4* pixels, float* P, unsigned int dim_x,
                                   unsigned int dim_xy, unsigned int dim_y, unsigned int num_elems,
                                   uint3 offset, unsigned int orientation, float dB) {

  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x + offset.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y + offset.y;
  unsigned int z = blockIdx.z*blockDim.z + threadIdx.z + offset.z;

  unsigned int idx = z*dim_xy+y*dim_x+x;
  if(idx < num_elems) {
    float c = (P[idx]);
    float sign = c>= 0.f ? 1.f : -1.f;
    c = ((log10(c*c)+dB)/dB);
    c = c>0?c:0.f;
    sign = c*sign;

    uchar4 color =  make_uchar4(0, 120.f-sign*125.f, 120.f+sign*125.f, c*255.f);

    if(orientation == 0) {
      idx = y*dim_x + x;
      pixels[idx] = color;
    }
    if(orientation == 1) {
      idx = z*dim_x + x;
      pixels[idx] = color;

    }
    if(orientation == 2) {
      idx = z*dim_y + y;
      pixels[idx] = color;
    }
  }
}

/// Render the position, corner/edge/wall byte
__global__ void renderPositionsPBO(uchar4* pixels, unsigned char* K, unsigned int dim_x,
                                   unsigned int dim_xy, unsigned int dim_y, unsigned int num_elems,
                                   uint3 offset, unsigned int orientation,
                                   unsigned int scheme) {

  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x + offset.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y + offset.y;
  unsigned int z = blockIdx.z*blockDim.z + threadIdx.z + offset.z;
  unsigned int idx = z*dim_xy+y*dim_x+x;
  if(idx < num_elems) {
    unsigned char c = (K[z*dim_xy+y*dim_x+x]);//*2000.f;
    unsigned char switchBit = c>>INSIDE_SWITCH;

    uchar4 color;
    if(scheme == 0 || scheme == 1) {
      c = c&FORWARD_POSITION_MASK;
      if(c == 0)
        color = make_uchar4(100, 100, 100, 255);
      if(c == 6)
        color = make_uchar4(0, 0, 255, 255);
      if(c == 5)
        color = make_uchar4(255, 0, 0, 255);
      if(c == 4)
        color = make_uchar4(0, 255, 0, 255);
      if(c == 3)
        color = make_uchar4(0, 255, 255, 255);
    }

    if(scheme == 2) {
      unsigned char dir_x = (c&DIR_X)>>0;
      unsigned char dir_y = (c&DIR_Y)>>1;
      unsigned char dir_z = (c&DIR_Z)>>2;
      unsigned char inner = (c&0x08)>>3;
      color = make_uchar4(255*dir_x-inner*100, 255*dir_y, 255*dir_z-inner*100, 245*switchBit+10);
    }
  
    if(orientation == 0) {
      idx = y*dim_x + x;
      pixels[idx] = color;
    }
    if(orientation == 1) {
      idx = z*dim_x + x;
      pixels[idx] = color;

    }
    if(orientation == 2) {
      idx = z*dim_y + y;
      pixels[idx] = color;
    }
  }
}

/// Render the inside/outside switch bit
__global__ void renderSwitchPBO(uchar4* pixels, unsigned char* K, unsigned int dim_x,
                                unsigned int dim_xy, unsigned int dim_y, unsigned int num_elems,
                                uint3 offset, unsigned int orientation,
                                unsigned int scheme) {

  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x + offset.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y + offset.y;
  unsigned int z = blockIdx.z*blockDim.z + threadIdx.z + offset.z;
  unsigned int idx = z*dim_xy+y*dim_x+x;

  if(idx < num_elems) {
    unsigned char c = (K[z*dim_xy+y*dim_x+x]);//*2000.f;
    unsigned char switchBit = (c>>INSIDE_SWITCH);
    uchar4 color;

    color = make_uchar4(255*switchBit, 255*switchBit, 255*switchBit, 255);
      if(orientation == 0) {
        idx = y*dim_x + x;
        pixels[idx] = color;
      }
      if(orientation == 1) {
        idx = z*dim_x + x;
        pixels[idx] = color;

      }
      if(orientation == 2) {
        idx = z*dim_y + y;
        pixels[idx] = color;
      }
  }
}

/// Render the inside/outside switch bit
__global__ void renderMaterialsPBO(uchar4* pixels, unsigned char* M, unsigned int dim_x,
                                  unsigned int dim_xy, unsigned int dim_y, unsigned int num_elems,
                                  uint3 offset, unsigned int orientation,
                                  unsigned int scheme) {

  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x + offset.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y + offset.y;
  unsigned int z = blockIdx.z*blockDim.z + threadIdx.z + offset.z;
  unsigned int idx = z*dim_xy+y*dim_x+x;

  if(idx < num_elems) {
    unsigned char mat = (M[z*dim_xy+y*dim_x+x]);//*2000.f;
    uchar4 color;

    if(mat == 0)
      color = make_uchar4(0, 0, 0, 255);
    if(mat == 1)
      color = make_uchar4(255, 0, 0, 255);
    if(mat == 2)
      color = make_uchar4(0, 255, 0, 255);
    if(mat == 3)
      color = make_uchar4(0, 0, 255, 255);

      if(orientation == 0) {
        idx = y*dim_x + x;
        pixels[idx] = color;
      }
      if(orientation == 1) {
        idx = z*dim_x + x;
        pixels[idx] = color;

      }
      if(orientation == 2) {
        idx = z*dim_y + y;
        pixels[idx] = color;
      }
  }
}
////// VBO render functions, 3D visualization
void renderBoundariesVBO(struct cudaGraphicsResource **vbo_resource, 
                         struct cudaGraphicsResource **color_resource,
                         unsigned char* d_K, unsigned char* d_B, 
                         unsigned int dimX,  unsigned int dimY, 
                         unsigned int dimZ, float dx) {

  // map OpenGL buffer object for writing from CUDA
  float4 *dptr;
  float4 *cptr;

  cudasafe(cudaGraphicsMapResources(1, vbo_resource, 0), "Map cuda vertex resource");
  size_t num_bytes_v;
  cudasafe(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes_v,
                                                    *vbo_resource), "Get Vertex resource pointer");
    
  cudasafe(cudaGraphicsMapResources(1, color_resource, 0), "Map cuda color resource");
    size_t num_bytes_c;
  cudasafe(cudaGraphicsResourceGetMappedPointer((void **)&cptr, &num_bytes_c,
                                                    *color_resource), "Get Color resource pointer");
  dim3 block(16, 16, 1);
  dim3 grid(dimX/block.x, dimY/block.y, 1);
  
  printf("Grid dimension x: %d, y: %d, z: %d\n", dimX/block.x, dimY/block.y, 1);
  printf("Number of threads: %d \n",(grid.x*grid.y*grid.z)*8*8*8 );
  printf("Number of elements: %d \n", dimX*dimY*dimZ );

  for(unsigned int i = 0; i < dimZ; i++) {
    boundaryRenderKernelVBO<<< grid, block>>>(dptr, cptr, dimX, dimY, d_K, d_B, i, dx);
  }
  
  cudasafe(cudaGraphicsUnmapResources(1, vbo_resource, 0), "Unmap vertex resource");
  cudasafe(cudaGraphicsUnmapResources(1, color_resource, 0), "Unmap color resource");
  
}


__global__ void renderPressuresVBO(float4* pos, float4* color, 
                                   float* P, unsigned int dimX, 
                                   unsigned int dimY, unsigned int dimZ, float dx) {

  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int z = blockIdx.z;

  // calculate element coordinates
  float u = x*dx;
  float w = y*dx;
  float v = z*dx;

  float c = P[z*dimX*dimY+y*dimX+x]*100;
  // write output vertex
  color[z*dimX*dimY+y*dimX+x] = make_float4(0.5f+c*10.f, 0.5f-c*10.f, 0.f, abs(c*100.f));
  pos[z*dimX*dimY+y*dimX+x] = make_float4(u, w, v, 1.f);
  
}

__global__ void boundaryRenderKernelVBO(float4 *pos, float4* color, 
                                        unsigned int dim_x, 
                                        unsigned int dim_y,
                                        unsigned char* d_K, unsigned char* d_B, 
                                        unsigned int slice, float dx) {
  
  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;
  unsigned int z = slice;

  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;
  
  unsigned int x = bx*blockDim.x + tx;
  unsigned int y = by*blockDim.y + ty;

    // calculate element coordinates
  float u = x*dx;
  float w = y*dx;
  float v = z*dx;

  int c = int(d_K[z*dim_x*dim_y+y*dim_x+x]&0X7F);
  float4 color_temp;

  switch(c) {
    case 0:
      color_temp = make_float4(0.f, 0.f, 0.f, 0.5f);
      break;
    case 6:
      color_temp = make_float4(0.f, 0.f, 1.f, 0.1f);
      break;
    case 5:
      color_temp = make_float4(1.f, 0.f, 0.f, 1.f);
      break;
    case 4:
      color_temp = make_float4(0.f, 1.f, 0.f, 1.f);
      break;
    case 3:
      color_temp = make_float4(0.f, 1.f, 1.f, 1.f);
      break;
    case 2:
      color_temp = make_float4(1.f, 1.f, 0.f, 1.f);
      break;
    case 1:
      color_temp = make_float4(1.f, 0.f, 1.f, 1.f);
      break;

    default:
      color_temp =  make_float4(1.f, 0.f, 1.f, 1.f);
  }
  
    // write output vertex

  color[z*dim_x*dim_y+y*dim_x+x] = color_temp;
  pos[z*dim_x*dim_y+y*dim_x+x] = make_float4(u, w, v, 1.f);
}

///////////////////////////
// Capture kernels

__global__ void captureSliceKernel(const float* __restrict p,
                                   const unsigned char* __restrict pos,
                                   float* capture_p,
                                   unsigned char* capture_pos, const unsigned int num_elems,
                                   const int dim_x, const int dim_y, const int dim_xy, 
                                   uint3 offset, const unsigned int limit, const int orientation) {

  int x = blockIdx.x*blockDim.x + threadIdx.x + offset.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y + offset.y;
  int z = blockIdx.z*blockDim.z + threadIdx.z + offset.z;

  int idx = z*dim_xy+y*dim_x+x;

  if(idx < num_elems) {
    int capture_idx = 0;
    if(orientation == 0)
      capture_idx = y*dim_x + x;
    if(orientation == 1) // xz
      capture_idx = z*dim_x + x;
    if(orientation == 2) // yz
      capture_idx = z*dim_y + y;

    if(capture_idx < limit) {
      capture_p[capture_idx] = p[idx];
      capture_pos[capture_idx] = pos[idx];
    }
  }
}
