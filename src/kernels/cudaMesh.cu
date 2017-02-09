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

#include "cudaUtils.h"
#include "cudaMesh.h"

void CudaMesh::toKowalczykScheme(unsigned char* d_pos, unsigned char* d_mat) {
  c_log_msg(LOG_INFO, "CudaMesh::toKowalczykScheme -  translate to kowalczyk");

  mesh_size_t num_elems = this->getNumberOfElements();
  mesh_size_t threadsPerBlock = 512;
  dim3 block_dim(threadsPerBlock);

  unsigned int numBlocks = (num_elems + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid_dim((unsigned int)ceil(sqrt(numBlocks)), (unsigned int)ceil(sqrt(numBlocks)));

  c_log_msg(LOG_VERBOSE, "CudaMesh::nodes2KowalczykScheme - Grid x: %u y: %u z: %u",
                          grid_dim.x, grid_dim.y, grid_dim.z);

  toKowalczykKernel<<<grid_dim, block_dim>>>(d_pos,
                                             d_mat,
                                             num_elems);

  cudasafe(cudaDeviceSynchronize(),
           "CudaMesh::toKowalczykScheme - cudaDeviceSynchronize at before return");
}

void CudaMesh::toBilbaoScheme(unsigned char* d_pos, unsigned char* d_mat) {
  c_log_msg(LOG_INFO, "CudaMesh::nodes2BilbaoScheme -  voxelizerDevice translate to bilbao");

  mesh_size_t num_elems = this->getNumberOfElements();

  mesh_size_t threadsPerBlock = 512;
  dim3 block_dim(threadsPerBlock);

  unsigned int numBlocks = (num_elems + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid_dim((unsigned int)ceil(sqrt(numBlocks)), (unsigned int)ceil(sqrt(numBlocks)));

  c_log_msg(LOG_INFO, "CudaMesh::nodes2Bilbao - Grid x: %u y: %u z: %u",
                         grid_dim.x, grid_dim.y, grid_dim.z);

  toBilbaoKernel<<<grid_dim, block_dim>>>(d_pos,
                                          d_mat,
                                          num_elems);

  cudasafe(cudaDeviceSynchronize(),
          "CudaMesh<LongNode>::nodes2BilbaoScheme - cudaDeviceSynchronize at before return");

  printCheckSum(d_pos, this->num_elements_, (char*)"node2Bilbao - return");
}

void CudaMesh::calculateBoundaries(const unsigned char* d_position_idx,
                                   unsigned char air_value,
                                   unsigned char out_value,
                                   unsigned char dev_idx) {

  mesh_size_t* d_air_nodes = valueToDevice(1, (mesh_size_t)0, dev_idx);
  mesh_size_t* d_boundary_nodes = valueToDevice(1, (mesh_size_t)0, dev_idx);

  mesh_size_t num_elems = this->getNumberOfElements();
  mesh_size_t threadsPerBlock = 512;
  dim3 block_dim(threadsPerBlock);

  unsigned int numBlocks = (num_elems + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid_dim((unsigned int)ceil(sqrt(numBlocks)), (unsigned int)ceil(sqrt(numBlocks)));

  calcBoundaries<<<grid_dim, block_dim>>>(d_position_idx,
                                          d_air_nodes, d_boundary_nodes,
                                          air_value, out_value, num_elems);
  cudasafe(cudaPeekAtLastError(),
    "CudaMesh::calculateBoundaries - peek before return");
  cudasafe(cudaDeviceSynchronize(),
    "CudaMesh::calculateBoundaries - "
    "cudaDeviceSynchronize at before return");

  mesh_size_t* h_air_elements = fromDevice(1, d_air_nodes,dev_idx);
  mesh_size_t* h_boundary_elements= fromDevice(1, d_boundary_nodes,dev_idx);

  destroyMem(d_air_nodes);
  destroyMem(d_boundary_nodes);

  // Assign counted values to the class
  this->num_air_elements_total_ = *h_air_elements;
  this->num_boundary_elements_total_ = *h_boundary_elements;
  free(h_air_elements);
  free(h_boundary_elements);
  c_log_msg(LOG_DEBUG, "CudaMesh::calculateBoundaries - air elements: %u boundary elements: %u",
        this->num_air_elements_total_, this->num_boundary_elements_total_);
}

void CudaMesh::calculateBoundariesHost(const unsigned char* h_position_idx,
                                       unsigned char air_value,
                                       unsigned char out_value) {
  mesh_size_t num_air_elements = 0;
  mesh_size_t num_boundary_elements = 0;
  for(mesh_size_t i = 0; i < this->getNumberOfElements(); i++) {
    if(h_position_idx[i] == air_value) {
      num_air_elements++;
    }
    if(h_position_idx[i] != out_value && h_position_idx[i] != air_value) {
      num_boundary_elements++;
    }
  }
  this->num_air_elements_total_ = num_air_elements;
  this->num_boundary_elements_total_ = num_boundary_elements;

  c_log_msg(LOG_DEBUG, "CudaMesh::calculateBoundariesHost - "
                       "air elements: %u boundary elements: %u",
                       this->num_air_elements_total_,
                       this->num_boundary_elements_total_);
}

void padWithZeros(unsigned char** d_mesh, uint3* dim,
                  unsigned int block_size_x,
                  unsigned int block_size_y,
                  unsigned int block_size_z) {

  unsigned int padX = 0;
  unsigned int padY = 0;
  unsigned int padZ = 0;

  unsigned int dim_x = (*dim).x;
  unsigned int dim_y = (*dim).y;
  unsigned int dim_z = (*dim).z;

  // Pad so that the dimensions are even with the grid size
  if((dim_x+padX)%block_size_x != 0)
    padX += (block_size_x-((dim_x+padX)%block_size_x));

  if((dim_y+padY)%block_size_y != 0)
    padY += (block_size_y-((dim_y+padY)%block_size_y));

  if((dim_z+padZ)%block_size_z != 0)
    padZ += (block_size_z-((dim_z+padZ)%block_size_z));

  // New size with the padding
  mesh_size_t newSize = (dim_x+padX)*(dim_y+padY)*(dim_z+padZ);

  unsigned char* d_mesh_new = valueToDevice(newSize, (unsigned char)0, 0);

  dim3 block(block_size_x, block_size_y, block_size_z);
  dim3 grid((int)ceil((float)(dim_x+padX)/(float)block.x),
            (int)ceil((float)(dim_y+padY)/(float)block.y),
          dim_z);

  padWithZerosKernel<<<grid, block>>>(d_mesh_new, *d_mesh,
                                      dim_x, dim_y, dim_z,
                                      padX, padY, padZ, 0);

  destroyMem(*d_mesh);
  (*d_mesh) = d_mesh_new;

  dim_x = dim_x+padX;
  dim_y = dim_y+padY;
  dim_z = dim_z+padZ;

  // update dimensions
  uint3 dim_;
  dim_.x = dim_x;
  dim_.y = dim_y;
  dim_.z = dim_z;
  cudasafe(cudaDeviceSynchronize(), "cudaUtils.cu: padWithZeros -cudaDeviceSynchronize at before return");
  *dim = dim_;
}

void padWithZeros(unsigned char** d_position_ptr,
                  unsigned char**d_material_ptr,
                  uint3* dim,
                  unsigned int block_size_x,
                  unsigned int block_size_y,
                  unsigned int block_size_z) {

  c_log_msg(LOG_INFO, "cudaUtils.cu: padWithZeros - Begin");
  uint3 dim_ = *dim;

  padWithZeros(d_position_ptr, &dim_, block_size_x, block_size_y, block_size_z);

  dim_ = *dim;

  padWithZeros(d_material_ptr, &dim_, block_size_x, block_size_y, block_size_z);

  c_log_msg(LOG_INFO, "cudaUtils.cu: padWithZeros - Return");
  // update dimensions
  *dim = dim_;

}

__host__ __device__ void toBilbao(unsigned char* d_position_ptr,
                                  unsigned char* d_material_ptr) {
  unsigned int k = (unsigned int)*d_position_ptr;

    if(k == 0){
      *d_position_ptr = (unsigned char)0;
      *d_material_ptr = (unsigned char)0;
      return;
    }

    if(k <= 8) {
      *d_position_ptr = (unsigned char)3;
      *d_position_ptr |= 0x80;
      return;
    }

    if(k > 8 && k <= 20) {
      *d_position_ptr = (unsigned char)4;
      *d_position_ptr |= 0x80;
      return;
    }

    if(k > 20 && k <= 26) {
      *d_position_ptr = (unsigned char)5;
      *d_position_ptr |= 0x80;
      return;
    }

    if(k ==27) {
      *d_position_ptr = (unsigned char)6;
      *d_position_ptr |= 0x80;
      *d_material_ptr = 0;
      return;
    }
}

__host__ __device__ void toKowalczyk(unsigned char* d_position_ptr,
                                     unsigned char* d_material_ptr) {
  unsigned int k = (unsigned int)*d_position_ptr;

  if(k == 0){
    *d_position_ptr = (unsigned char)0;
    *d_material_ptr = (unsigned char)0;
    return;
  }
  if(k == 1) { // down left in
    *d_position_ptr = 0|SIGN_Z|DIR_X|DIR_Y|DIR_Z|CENTERED_MASK;
    return;
  }
  if(k == 2) { // down right in
    *d_position_ptr = 0|SIGN_Z|SIGN_X|DIR_X|DIR_Y|DIR_Z|CENTERED_MASK;
    return;
  }
  if(k == 3) { // down left out
    *d_position_ptr = 0|SIGN_Z|SIGN_Y|DIR_X|DIR_Y|DIR_Z|CENTERED_MASK;
    return;
  }
  if(k == 4) { // down right out
    *d_position_ptr = 0|SIGN_Z|SIGN_Y|SIGN_X|DIR_X|DIR_Y|DIR_Z|CENTERED_MASK;
    return;
  }
  if(k == 5) { //  up left in
    *d_position_ptr = 0|DIR_X|DIR_Y|DIR_Z|CENTERED_MASK;
    return;
  }
  if(k == 6) { //  up right in
    *d_position_ptr = 0|SIGN_X|DIR_X|DIR_Y|DIR_Z|CENTERED_MASK;
    return;
  }
  if(k == 7) { // up left out
    *d_position_ptr = 0|SIGN_Y|DIR_X|DIR_Y|DIR_Z|CENTERED_MASK;
    return;
  }
  if(k == 8) { // up right out
    *d_position_ptr = 0|SIGN_Y|SIGN_X|DIR_X|DIR_Y|DIR_Z|CENTERED_MASK;
    return;
  }
  if(k == 9) { // 09 = Down, Left, Right, In
    *d_position_ptr = 0|DIR_Y|DIR_Z|SIGN_Z|CENTERED_MASK;
    return;
  }
  if(k == 10) { // 10 = Down, Left, Right, Out
    *d_position_ptr = 0|DIR_Y|DIR_Z|SIGN_Z|SIGN_Y|CENTERED_MASK;
    return;
  }
  if(k == 11) { // 11 = Down, Left, In, Out
    *d_position_ptr = 0|DIR_X|DIR_Z|SIGN_Z|CENTERED_MASK;
    return;
  }
  if(k == 12) { // 12 = Down, Right, In, Out
    *d_position_ptr = 0|DIR_X|DIR_Z|SIGN_Z|SIGN_X|CENTERED_MASK;
    return;
  }
  if(k == 13){ // 13 = Up, Left, Right, In
    *d_position_ptr = 0|DIR_Y|DIR_Z|CENTERED_MASK;
    return;
  }
  if(k == 14) { // 14 = Up, Left, Right, Out
    *d_position_ptr = 0|DIR_Y|DIR_Z|SIGN_Y|CENTERED_MASK;
    return;
  }
  if(k == 15) { // 15 = Up, Left, In, Out
    *d_position_ptr = 0|DIR_X|DIR_Z|CENTERED_MASK;
    return;
  }
  if(k == 16) { // 16 = Up, Right, In, Out
    *d_position_ptr = 0|DIR_X|DIR_Z|SIGN_X|CENTERED_MASK;
    return;
  }
  if(k == 17) { // 17 = Up, Down, Left, In
    *d_position_ptr = 0|DIR_Y|DIR_X|CENTERED_MASK;
    return;
  }
  if(k == 18) { // 18 = Up, Down, Right, In
    *d_position_ptr = 0|DIR_X|DIR_Y|SIGN_X|CENTERED_MASK;
    return;
  }
  if(k == 19) { // 19 = Up, Down, Left, Out
    *d_position_ptr = 0|DIR_X|DIR_Y|SIGN_Y|CENTERED_MASK;
    return;
  }
  if(k == 20) { // 20 = Up, Down, Right, Out
    *d_position_ptr = 0|DIR_Y|DIR_X|SIGN_Y|SIGN_X|CENTERED_MASK;
    return;
  }
  if(k == 21) { // 21 = Left, Right, In, Out, Down
    *d_position_ptr = 0|DIR_Z|SIGN_Z|CENTERED_MASK;
    return;
  }
  if(k == 22) { // 22 = Left, Right, Out, Down, Up
    *d_position_ptr =  0|DIR_Y|SIGN_Y|CENTERED_MASK;
    return;
  }
  if(k == 23) { // 23 = Left, Right, In, Down, Up
    *d_position_ptr = 0|DIR_Y|CENTERED_MASK;
    return;
  }
  if(k == 24) { // 24 = Right, In, Out, Down, Up
    *d_position_ptr = 0|DIR_X|SIGN_X|CENTERED_MASK;
    return;
  }
  if(k == 25) { // 25 = Left, In, Out, Down, Up
    *d_position_ptr = 0|DIR_X|CENTERED_MASK;
    return;
  }
  if(k == 26) { // 26 = Left, Right, In, Out, Up
    *d_position_ptr = 0|DIR_Z|CENTERED_MASK;
    return;
  }
  if(k == 27) { // 27 = Left, Right, In, Out, Down, Up - AIR NODE
    *d_position_ptr = 0|CENTERED_MASK;
    return;
  }
}

__global__ void toKowalczykKernel(unsigned char* d_position_ptr,
                                  unsigned char* d_material_ptr,
                                  mesh_size_t num_elems) {
  mesh_size_t idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < num_elems) {
    toKowalczyk(d_position_ptr+idx, d_material_ptr+idx);
  }
}


__global__ void toBilbaoKernel(unsigned char* d_position_ptr, unsigned char* d_material_ptr,
                               mesh_size_t num_elems) {
  mesh_size_t idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < num_elems) {
    toBilbao(d_position_ptr+idx, d_material_ptr+idx);
  }
}

__global__ void calcBoundaries(const unsigned char* d_position_ptr,
                               mesh_size_t* air,
                               mesh_size_t* boundary,
                               unsigned char air_value,
                               unsigned char outside_value,
                               mesh_size_t num_elems) {
  mesh_size_t idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < num_elems) {
    if(d_position_ptr[idx] == air_value) {
      atomicAdd(air,1);
    }
    if(d_position_ptr[idx] != outside_value && d_position_ptr[idx] != air_value) {
      atomicAdd(boundary,1);
    }
  }
}

__global__ void padWithZerosKernel(unsigned char* d_mesh_new,
                                   unsigned char* d_mesh_old,
                                   unsigned int dim_x, unsigned int dim_y,
                                   unsigned int dim_z,
                                   unsigned int block_x, unsigned int block_y,
                                   unsigned int block_z,
                                   unsigned int slice) {

  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int z = blockIdx.z;

  if(x < dim_x+1 && x > 0 && y < dim_y+1 &&  y > 0 && z < dim_z+1 && z > 0) {
    mesh_size_t old_idx = (z)*(dim_x)*(dim_y)+(y)*dim_x+x;
    mesh_size_t new_idx = z*(dim_x+block_x)*(dim_y+block_y)+y*(dim_x+block_x)+x;

    d_mesh_new[new_idx] = d_mesh_old[old_idx];
  }
}
