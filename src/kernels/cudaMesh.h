#ifndef CUDA_MESH_H
#define CUDA_MESH_H

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

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define INSIDE_SWITCH 7
#define FORWARD_POSITION_MASK 0X7F
#define CENTERED_MASK 0x80

#define DIR_X  0x01
#define DIR_Y 0x02
#define DIR_Z 0x04
#define SIGN_X 0x10
#define SIGN_Y 0x20
#define SIGN_Z 0x40

// Forward Declaration
class LongNode;
class ShortNode;
class Node;

///////////////////////////////////////////////////////////////////////////////
/// \brief Class that handles the simulation domain. Class contains device 
/// pointers to different meshes which are used in the simulation.
/// Different meshes are: <br>
/// <i>position index mesh:</i> Contains the orientation of each node <br>
/// <i>material index mesh:</i> Contains the material index of each node <br>
/// <i>2 X Pressure mesh: </i>Either a single precision of double precision mesh of <br>
///                pressure values for the current  and past pressure values 
///
///////////////////////////////////////////////////////////////////////////////

class CudaMesh {
public:
  CudaMesh() 
  : double_(false),
    dif_order_(0),
    number_of_partitions_(0),
    partition_size_(0),
    h_material_coef_ptr_(NULL),
    number_of_unique_materials_(0),
    h_parameter_ptr_(NULL),
    num_elements_(0),
    dim_x_(0),
    dim_y_(0),
    dim_z_(0),
    dim_xy_(0),
    dx_(0),
    block_size_x_(0),
    block_size_y_(0),
    block_size_z_(0)
  {};

  ~CudaMesh() {};

public:
  ////// Mesh
  bool double_; // mesh in double precision
  unsigned int dif_order_; // frequency depending boundaries

  std::vector<unsigned char*> position_idx_ptr_;
  std::vector<unsigned char*> material_idx_ptr_;
  std::vector<unsigned char*> packed_ptr_;

  // Different containers for the two precisions, more code
  // but straightforward interface with the CUDA code
  // optional TODO: Templating to reduce code

  // Single precision
  std::vector<float*> pressures_;
  std::vector<float*> pressures_past_;
  std::vector<float*> parameters_;
  std::vector<float*> materials_;
  
  // Double precision
  std::vector<double*> pressures_double_;
  std::vector<double*> pressures_past_double_;
  std::vector<double*> parameters_double_;
  std::vector<double*> materials_double_;

  // NEW template
  /*
  std::vector<T*> pressures_T_;
  std::vector<T*> pressures_past_T_;
  std::vector<T*> parameters_T_;
  std::vector<T*> materials_T_;
  */

  // Volumetric boundaries
  // for now, save the whole node
  std::vector< Node* > nodes_;

  // Digital impedance filters not implemented
  std::vector<float*> dif_;

  std::vector< unsigned int > device_list_;
  std::vector< std::vector< unsigned int> > partition_indexing_;
  std::vector<unsigned int> memory_splits_;
  std::vector<unsigned int> number_of_partition_elements_;
  unsigned int number_of_partitions_;
  unsigned int partition_size_;

  /////// Materials
  float* h_material_coef_ptr_; ///< Material coefficient on host, not to be deallocated
  double* h_material_coef_ptr_double_;  ///< Material coefficient on host, not to be deallocated
  unsigned int number_of_unique_materials_;  ///< number of unique materials on the mesh
  
  // Parameters
  float* h_parameter_ptr_; ///< Simulation parameters on host, not to be deallocated
  double* h_parameter_ptr_double_; ///< Simulation parameters on host, not to be deallocated

  ///// Mesh properties
  unsigned int num_elements_;
  unsigned int num_air_elements_total_;
  unsigned int num_boundary_elements_total_;
  unsigned int dim_x_;
  unsigned int dim_y_;
  unsigned int dim_z_;
  unsigned int dim_xy_;
  float dx_;

  unsigned int block_size_x_;
  unsigned int block_size_y_;
  unsigned int block_size_z_;

  // Private Member functions
  void toBilbaoScheme();
  void toKowalczykScheme();

public:
  void destroyPartitions() {
    c_log_msg(LOG_VERBOSE, "CudaMesh destructor");
    for(unsigned int i = 0; i < this->number_of_partitions_; i++) {
      cudaSetDevice(i);
      destroyMem(position_idx_ptr_.at(i));
      destroyMem(material_idx_ptr_.at(i));
      if(this->isDouble()) {
        destroyMem(pressures_double_.at(i));
        destroyMem(pressures_past_double_.at(i));
        destroyMem(materials_double_.at(i));
        destroyMem(parameters_double_.at(i));
      }
      else {
        destroyMem(pressures_.at(i));
        destroyMem(pressures_past_.at(i));
        destroyMem(materials_.at(i));
        destroyMem(parameters_.at(i));
      }
      
    }
    cudasafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize after destroy");
    c_log_msg(LOG_VERBOSE, "CudaMesh destructor returning");
  };

  float* getPressurePtrAt(unsigned int partition_idx) 
    {return this->pressures_.at(partition_idx);};

  float* getPastPressurePtrAt(unsigned int partition_idx) 
    {return this->pressures_past_.at(partition_idx);};

  float* getMaterialPtrAt(unsigned int partition_idx) 
    {return this->materials_.at(partition_idx);};

  float* getParameterPtrAt(unsigned int partition_idx) 
    {return this->parameters_.at(partition_idx);};

  double* getMaterialPtrDoubleAt(unsigned int partition_idx) 
    {return this->materials_double_.at(partition_idx);};

  double* getParameterPtrDoubleAt(unsigned int partition_idx) 
    {return this->parameters_double_.at(partition_idx);};

  double* getPressureDoublePtrAt(unsigned int partition_idx)
    {return this->pressures_double_.at(partition_idx);};

  double* getPastPressureDoublePtrAt(unsigned int partition_idx)
    {return this->pressures_past_double_.at(partition_idx);}

  unsigned char* getPositionIdxPtrAt(unsigned int partition_idx) 
    {return this->position_idx_ptr_.at(partition_idx);}

  unsigned char* getMaterialIdxPtrAt(unsigned int partition_idx) 
    {return this->material_idx_ptr_.at(partition_idx);}

  unsigned int getMemorySplitAt(unsigned int partition_idx)
    {return this->memory_splits_.at(partition_idx);}

  unsigned int getNumberOfElementsAt(unsigned int partition_idx)
    {return (unsigned int)this->partition_indexing_.at(partition_idx).size()*this->getDimXY();}

  unsigned int getNumberOfPartitions() {return (unsigned int)this->partition_indexing_.size();}
  unsigned int getPartitionSize() {return (unsigned int)this->partition_indexing_.at(0).size();}

  unsigned int getPartitionSize(int partition) {
    unsigned int ret = 0; unsigned int np = this->getNumberOfPartitions();
    if(partition < (int)np) ret = (unsigned int)this->partition_indexing_.at(partition).size();   
    return ret;}
  
  unsigned int getFirstSliceIdx(int partition) {return this->partition_indexing_.at(partition).at(0);}
  unsigned int getDeviceAt(int i) {return this->device_list_.at(i);}
  unsigned int getBlockX() {return this->block_size_x_;}
  unsigned int getBlockY() {return this->block_size_y_;}
  unsigned int getBlockZ() {return this->block_size_z_;}
  unsigned int getDimX() {return this->dim_x_;}
  unsigned int getDimY() {return this->dim_y_;}
  unsigned int getDimZ() {return this->dim_z_;}
  unsigned int getDimXY() {return this->dim_xy_;}
  unsigned int getGridDimX() {return (this->dim_x_)/(this->block_size_x_);}
  unsigned int getGridDimY() {return (this->dim_y_)/(this->block_size_y_);}
  unsigned int getGridDimZ() {return (this->dim_z_)/(this->block_size_z_);}
  unsigned int getNumberOfElements() {return this->num_elements_;}
  unsigned int getNumberOfAirElements() {return this->num_air_elements_total_;}
  unsigned int getNumberOfBoundaryElements() {return this->num_boundary_elements_total_;}
  bool isDouble() const {return this->double_;}
  void setDouble(bool is_double) {this->double_ = is_double;}


  inline int getElementIndex(int x, int y, int z) {
    return this->getDimXY()*z+this->getDimX()*y+x;
  }

  inline void getElementIdxAndDevice(unsigned int x, unsigned int y, unsigned int z, 
                                     int* dev_i, int* elem) {
    *dev_i = -1;
    *elem = -1;
    for(int i = 0; i < this->partition_indexing_.size(); i++) {
      if(z > *(this->partition_indexing_.at(i).end()-1))
        continue;
      if(z < this->partition_indexing_.at(i).at(0))
        break;
      unsigned int first = this->partition_indexing_.at(i).at(0);
      *elem = getElementIndex(x, y, z-first);
      *dev_i = i;
      break;
     }
    return;
  }
  
  inline int getDeviceOfElement(unsigned int x, unsigned int y, unsigned int z) {
    int ret = -1;
    for(int i = 0; i < this->partition_indexing_.size(); i++) {
      if(z > (int)*(this->partition_indexing_.at(i).end()-1))
        continue;
      if(z < this->partition_indexing_.at(i).at(0))
        break;
      ret = this->device_list_.at(i);
    }
    return ret;
  }

  std::vector< std::vector <unsigned int> > getPartitionIndexing(int num_parts,
                                                                 int dim) {
    int part_size = dim/num_parts;
    std::vector< std::vector<unsigned int> > ret;
    for(int i = 0; i < num_parts; i++) {  
      int s_inc = 1;
      int e_inc = 1;
    
      if(i == 0) {s_inc =0;}
      if(i == num_parts-1) {e_inc = 0;}
      int current_size = part_size+s_inc+e_inc;
  
      // If at the last part, get rest of the slices      
      if(i != 0 && i == num_parts-1) {
        int inc = dim-(i+1)*part_size;
        current_size += inc;
      }
      
      std::vector<unsigned int> slices(current_size, 0);  
      for(int j = 0; j < current_size; j++) {
        slices.at(j) = i*part_size-s_inc+j;
      }
      //std::cout<<"getPartitionIndexing - size slice: "<<slices.size()<<std::endl;
      ret.push_back(slices);
    }
//    std::cout<<"getPartitionIndexing - size cont: "<<ret.size()<<std::endl;
    return ret;
  }

  template<typename T>
  std::vector<T*> splitPartitions(std::vector< std::vector<unsigned int> >& slices,
                                 int slice_dim) {
    std::vector<T*> ret;
    for(int i = 0; i < slices.size(); i++) {
      T* n_val = (T*)calloc(slice_dim*slices.at(i).size(), sizeof(T));
      ret.push_back(n_val);
    }
    return ret;
  }

  // Set a sample on each slice corresponding to the index
  template<typename T>
  void setSample(unsigned int x, unsigned int y, unsigned int z, T sample,
                 std::vector<T*>& domain, 
                 std::vector< std::vector<unsigned int> >& slices) {
    unsigned int s = (unsigned int)slices.size();
    //#pragma omp parallel for
    for(int i = 0; i < s; i++) {
      if(z > *(slices.at(i).end()-1))
        continue;
      if(z < slices.at(i).at(0))
        break;

      cudaSetDevice(this->device_list_.at(i));
      int first = slices.at(i).at(0);
      T* dest = domain.at(i)+getElementIndex(x, y, z-first);
      cudasafe(cudaMemcpy(dest, &sample, sizeof(T), cudaMemcpyHostToDevice), "setSample T : Memcopy To device");
    }
  }

  template<typename T>
  void setSampleAt(unsigned int x, unsigned int y, unsigned int z, T sample, unsigned int partition, 
                std::vector<T*>& domain) {
    unsigned int offset = getElementIndex(x,y,z);
    T* dest = domain.at(partition)+offset;
    cudasafe(cudaMemcpy(dest, &sample, sizeof(T), cudaMemcpyHostToDevice), "setSample T : Memcopy To device");
  }

  template<typename T>
  void addSample(unsigned int x, unsigned int y, unsigned int z, T sample,
                 std::vector<T*>& domain, 
                 std::vector< std::vector< unsigned int> >& slices){
    unsigned int s = (unsigned int)slices.size();
    //#pragma omp parallel for
    for(int i = 0; i < s; i++) {
      if(z > *(slices.at(i).end()-1))
        continue;
      if(z < slices.at(i).at(0))
        break;
      cudaSetDevice(this->device_list_.at(i));
      int first = slices.at(i).at(0);

      T domain_smp = 0;
      T* dest = domain.at(i)+getElementIndex(x, y, z-first);
      cudasafe(cudaMemcpy(&domain_smp, dest, sizeof(T), cudaMemcpyDeviceToHost), "addSample T : Memcopy To Host");
      sample+=domain_smp;
      cudasafe(cudaMemcpy(dest, &sample, sizeof(T), cudaMemcpyHostToDevice), "addSample T : Memcopy To Device");
        
    }  
  }

  template<typename T>
  T getSampleAt(int x, int y, int z, int partition,
                std::vector<T*>& domain) {
    T ret = 0;
    if(z > this->partition_indexing_.at(partition).size()) {
      std::cout<<"z "<<z<<" out of range "<<this->partition_indexing_.at(partition).size()<<std::endl;
      return ret;
    }
    cudaSetDevice(this->device_list_.at(partition));
    int offset = this->getElementIndex(x,y,z);
    T* src = domain.at(partition)+offset;
    cudasafe(cudaMemcpy(&ret, src, sizeof(T), cudaMemcpyDeviceToHost), "getSampleAt T : Memcopy To Host");
    return ret;
  }

  template<typename T>
  T getSample(unsigned int x, unsigned int y, unsigned int z, 
              std::vector<T*>& domain, 
              std::vector< std::vector< unsigned int> >& slices) {
    T ret = 0.0;

    for(int i = 0; i < slices.size(); i++) {
      cudaSetDevice(this->device_list_.at(i));
      if(z > *(slices.at(i).end()-1))
        continue;
      if(z < slices.at(i).at(0))
        break;
      unsigned int first = slices.at(i).at(0);
      T* src = domain.at(i)+getElementIndex(x, y, z-first);
      cudasafe(cudaMemcpy(&ret, src, sizeof(T), cudaMemcpyDeviceToHost), "addSample T : Memcopy To Host");
      break;
     }
    return ret;   
  }

  template<typename T>
  T* getElement(unsigned int x, unsigned int y, unsigned int z, unsigned int& dev_i,
              std::vector<T*>& domain, 
              std::vector< std::vector< unsigned int> >& slices) {
    T* ret = (T*)NULL;

    for(int i = 0; i < slices.size(); i++) {
      if(z > *(slices.at(i).end()-1))
        continue;
      if(z < slices.at(i).at(0))
        break;
      int first = slices.at(i).at(0);
      ret = domain.at(i)+getElementIndex(x, y, z-first);
      dev_i = i;
      break;
     }
    return ret;   
  }

  template<typename T>
  T* getElementAt(unsigned int x, unsigned int y, unsigned int z, unsigned int partition,
                 std::vector<T*>& domain, 
                 std::vector< std::vector< unsigned int> >& slices) {
    return domain.at(partition)+getElementIndex(x, y, z);
  }

  template<typename T>
  void switchHalos(std::vector<T*>& domain,
                   std::vector< std::vector< unsigned int > >& slices) {
    unsigned int slice_s = this->getDimXY();
    #pragma omp parallel for
    for(unsigned int i = 0; i < (unsigned int)slices.size()-1; i++) {
      unsigned int src_1 =  *(slices.at(i).end()-2);
      unsigned int dest_1 =  slices.at(i+1).at(0);
      unsigned int src_2 =  slices.at(i+1).at(1);
      unsigned int dest_2 =  *(slices.at(i).end()-1);
      unsigned int f_i = slices.at(i).at(0);
      unsigned int f_i1 = slices.at(i+1).at(0);

      T* s1 = domain.at(i)+(src_1-f_i)*slice_s;
      T* d1 = domain.at(i+1)+(dest_1-f_i1)*slice_s;
      T* s2 = domain.at(i+1)+(src_2-f_i1)*slice_s;
      T* d2 = domain.at(i)+(dest_2-f_i)*slice_s;

      unsigned int d_1 = this->getDeviceAt(i);
      unsigned int d_2 = this->getDeviceAt(i+1);
      cudasafe(cudaMemcpyPeer(d1, d_2, 
                              s1, d_1, 
                              slice_s*sizeof(T)),
                              "CudaMesh::switchHalos - memcopyPeer i -> i+1");

     cudasafe(cudaMemcpyPeer(d2, d_1, 
                             s2, d_2, 
                             slice_s*sizeof(T)),
                             "CudaMesh::switchHalos - memcopyPeer i+1 -> i");
    }
    cudaDeviceSynchronize();
  }

  template <typename T>
  T* getElementAt(unsigned int x, unsigned int y, unsigned int z, unsigned int partition) {
    T* ret;
    if(this->isDouble()) 
      ret = this->getElementAt(x,y,z, partition,
                                 this->pressures_double_,
                                 this->partition_indexing_);
    else
      ret = this->getElementAt(x,y,z, partition,
                                 this->pressures_,
                                 this->partition_indexing_);
    
    return ret;
  }

  /// Function return a pointer to the element from the partitions which
  /// correspond to the given coordinates. If sample is in halo element, 
  /// halo indicates from which partition the sample is returned
  template <typename T>
  T* getElement(unsigned int x, unsigned int y, unsigned int z, unsigned int *halo) {
    T* ret;
    if(this->isDouble()) 
      ret = this->getElement(x,y,z, *halo, 
                               this->pressures_double_, 
                               this->partition_indexing_);
    else
      ret = this->getElement(x,y,z, *halo, 
                               this->pressures_, 
                               this->partition_indexing_);
    return ret;    
  }

  // Add sample to the existing pressure value of the mesh
  template <typename T>
  void addSample(T sample, unsigned int x, unsigned int y, unsigned int z) {
    if(this->isDouble()) {
      addSample(x,y,z, (double)sample,
                this->pressures_double_, this->partition_indexing_);
    }
    else {
      addSample(x,y,z, (float)sample,
                this->pressures_, this->partition_indexing_);
    }
  } // End function

  ///////////////////////////////////////////////////////////////////////////////
  /// \brief Set a sample in the mesh from a value from host memory
  /// \tparam The type of pressure data type, float / double
  /// \param sample Sample value to be assigned
  /// \param x, y, z The element coordinates in the mesh
  ///////////////////////////////////////////////////////////////////////////////
  template <typename T>
  void setSample(T sample, unsigned int x, unsigned int y, unsigned int z) {
    //c_log_msg(LOG_VERBOSE, "CudaMesh::setSample - %u %u %u",x,y,z);
    if(this->isDouble()) {
      setSample(x,y,z, (double)sample,
                this->pressures_double_, this->partition_indexing_);
    }
    else {
      setSample(x,y,z, (float)sample,
                this->pressures_, this->partition_indexing_);
    }
  } // End function

  ///////////////////////////////////////////////////////////////////////////////
  /// \brief Set a sample in the mesh from a value from host memory to a 
  ///  specific partition
  /// \tparam The type of pressure data type, float / double
  /// \param sample Sample value to be assigned
  /// \param x, y, z The element coordinates in the mesh
  /// \param partition Partition index, namely the device in which the memory lies
  ///////////////////////////////////////////////////////////////////////////////
  template <typename T>
  void setSampleAt(T sample, unsigned int x, unsigned int y, unsigned int z, unsigned int partition) {
    c_log_msg(LOG_VERBOSE, "CudaMesh::setSampleAt - %u %u %u partition %u",x,y,z, partition);
    if(this->isDouble()) {
      setSampleAt(x,y,z, (double)sample, partition, this->pressures_double_);
    }
    else {
      setSampleAt(x,y,z, (float)sample, partition, this->pressures_);
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// \brief Get a sample from the mesh 
  /// \tparam The type of pressure data type, float / double
  /// \param x, y, z The element coordinates of the sample in the mesh
  /// \return sample value in host memory
  ///////////////////////////////////////////////////////////////////////////////
  template <typename T>
  T getSample(unsigned int x, unsigned int y, unsigned int z) {
    c_log_msg(LOG_VERBOSE, "CudaMesh::getSample - %u %u %u",x,y,z);
    T sample = (T)0.0;
    if(this->isDouble()) {
      sample = getSample<double>(x,y,z, this->pressures_double_, this->partition_indexing_);
    }
    else {
      sample = getSample<float>(x,y,z, this->pressures_, this->partition_indexing_);
    }
    return sample;
  }
  
  ///////////////////////////////////////////////////////////////////////////////
  /// \brief Get a position idx value from the mesh in a position in
  /// a specific partition
  /// \param x, y, z The element coordinates of the sample in the mesh
  /// \return position index value
  ///////////////////////////////////////////////////////////////////////////////
  template <typename T>
  T getSampleAt(unsigned int x, unsigned int y, unsigned int z, unsigned int partition) {
    c_log_msg(LOG_VERBOSE, "CudaMesh::getSampleAt - %u %u %u partition %u",x,y,z, partition);
    T sample = (T)0.0;
    if(this->isDouble()) {
      sample = getSampleAt(x,y,z, partition, this->pressures_double_);
    }
    else {
      sample = getSampleAt(x,y,z, partition, this->pressures_);
    }
    return sample;
  }
  
  ///////////////////////////////////////////////////////////////////////////////
  /// \brief Get a position idx value from the mesh
  /// \param x, y, z The element coordinates of the sample in the mesh
  /// \return position index value
  ///////////////////////////////////////////////////////////////////////////////
  unsigned char getPositionSample(unsigned int x, unsigned int y, unsigned int z) {
    c_log_msg(LOG_VERBOSE, "CudaMesh::getPositionSample - %u %u %u",x,y,z);
    unsigned char sample = 0;
    sample = getSample(x,y,z, this->position_idx_ptr_, this->partition_indexing_);
    return sample;
  }


  // This function is ridiculously slow
  template<typename T>
  T* getSlice(unsigned int slice, unsigned int orientation) {
    T* data = NULL;

    /*
    unsigned int dim_x = 0;
    unsigned int dim_y = 0;
   
    if(orientation == 0) {dim_x = this->getDimX(); dim_y = this->getDimY();};
    if(orientation == 1) {dim_x = this->getDimX(); dim_y = this->getDimZ();};
    if(orientation == 2) {dim_x = this->getDimY(); dim_y = this->getDimZ();};
    */
    if(orientation == 0) {
      unsigned int dev_i;
      T* start_address = (T*)NULL;

       start_address = this->getElement(0u,0u,slice, dev_i,
                                         this->pressures_,
                                         this->partition_indexing_);
      
      data= fromDevice(this->getDimXY(), start_address, this->device_list_.at(dev_i));

    } // end orientation
     
   return data;
  }

  unsigned char* getPositionSlice(unsigned int slice, unsigned int orientation) {
    unsigned char* data = (unsigned char*)NULL;
    /*
    int dim_x, dim_y, dim_z:
    if(orientation == 0) {dim_x = this->getDimX(); dim_y = this->getDimY();};
    if(orientation == 1) {dim_x = this->getDimX(); dim_y = this->getDimZ();};
    if(orientation == 2) {dim_x = this->getDimY(); dim_y = this->getDimZ();};
    */

    if(orientation == 0) {
      unsigned char* start_address = NULL;
      unsigned int dev_i;
      start_address = this->getElement(0,0,slice, dev_i,
                                       this->position_idx_ptr_,
                                       this->partition_indexing_);

      data = fromDevice(this->getDimXY(), start_address, this->device_list_.at(dev_i));
    }
    return data;
  }

  void makePartition(unsigned int number_of_partitions, 
                     std::vector<unsigned int> device_list = std::vector<unsigned int>()) {
    c_log_msg(LOG_INFO, "CudaMesh::makePartition - begin, number of partitions %d, dim z %u", 
              number_of_partitions, this->getDimZ());

    printMemInfo("CudaMesh::makePartition - memory before partition", getCurrentDevice());


    if(device_list.size() == 0) {
      for(unsigned int i = 0; i < number_of_partitions; i++)
        device_list.push_back(i);  
    }

    this->device_list_ = device_list;
    this->partition_indexing_ = getPartitionIndexing(number_of_partitions, this->getDimZ());
    
    clock_t start_t;
    clock_t end_t;
    start_t = clock();
    
    // Grab pointers to the meshes and clear the container for partitions
    unsigned char* d_position_idx = this->position_idx_ptr_.at(0);
    unsigned char* d_material_idx = this->material_idx_ptr_.at(0);
    
    this->position_idx_ptr_.clear();
    this->material_idx_ptr_.clear();
    
    // Go through devices and copy the part to each
    for(unsigned int k = 0; k < number_of_partitions; k ++) {
      int i = this->device_list_.at(k);
      cudaSetDevice(i);
      int offset = this->partition_indexing_.at(k).at(0)*this->getDimXY();
      int size = (int)this->partition_indexing_.at(k).size()*this->getDimXY();

      if(number_of_partitions > 1) {
        unsigned char* position_idx = valueToDevice(size+1+dim_x_, (unsigned char)0, i);
        unsigned char* material_idx = valueToDevice(size+1+dim_x_, (unsigned char)0, i);

        cudasafe(cudaMemcpyPeer(position_idx, i, 
                                d_position_idx+offset, 0, size*sizeof(unsigned char)), 
                               "CudaMesh::makePartition - memcpyPeer position");
        cudasafe(cudaMemcpyPeer(material_idx, i, 
                                d_material_idx+offset, 0, size*sizeof(unsigned char)),
                               "CudaMesh::makePartition - memcpyPeer material");

        // Push the device pointer to containers
        this->position_idx_ptr_.push_back(position_idx);
        this->material_idx_ptr_.push_back(material_idx);
      }
      else {
        this->position_idx_ptr_.push_back(d_position_idx);
        this->material_idx_ptr_.push_back(d_material_idx);
      }
    } // End material / position partition

    // Free the original meshes from device 0 if number of partitions is > 1
    if(number_of_partitions > 1) {
         destroyMem(d_position_idx);
         destroyMem(d_material_idx);    
    }

    // Go through devices and allocate pressures
    for(unsigned int k = 0; k < number_of_partitions; k ++) {
      int i = this->device_list_.at(k);
      cudaSetDevice(i);
      int size = (int)this->partition_indexing_.at(k).size()*this->getDimXY();

      if(this->isDouble()) {
        c_log_msg(LOG_DEBUG, "CudaMesh::makePartition - allocating double pressures dev %u", i);
        double* P = toDevice<double>(size+1+dim_x_, i);
        double* P_past = toDevice<double>(size+1+dim_x_, i);
        
        // Push the device pointers to containers
        this->pressures_double_.push_back(P);
        this->pressures_past_double_.push_back(P_past);    

        // Allocate and copy material coefficients to all devices
        c_log_msg(LOG_DEBUG, "CudaMesh::makePartition - allocating double materials");
        this->materials_double_.push_back(toDevice<double>(this->number_of_unique_materials_*20,
                                                            this->h_material_coef_ptr_double_,
                                                            i));

        // Allocate and copy simulation parameters to all devices
        this->parameters_double_.push_back(toDevice<double>(10, this->h_parameter_ptr_double_, i));
      }
      else {
        c_log_msg(LOG_DEBUG, "CudaMesh::makePartition - allocating float pressures dev %u", i);
        float* P = toDevice<float>(size+1+dim_x_, i);
        float* P_past = toDevice<float>(size+1+dim_x_, i);
        // Push the device pointers to containers
        this->pressures_.push_back(P);
        this->pressures_past_.push_back(P_past);
        this->materials_.push_back(toDevice<float>(this->number_of_unique_materials_*20,
                                                   this->h_material_coef_ptr_,
                                                   i));
        // Allocate and copy simulation parameters to all devices
        this->parameters_.push_back(toDevice(4, this->h_parameter_ptr_, i));
      }
    }// End partition loop

    end_t = clock()-start_t;
    c_log_msg(LOG_INFO,"CudaMesh::MakePartition - time: %f seconds", ((float)end_t/CLOCKS_PER_SEC));

  } // End make partition


  /// \brief switch halo layers between partitions
  void switchHalos() {
    if(this->isDouble()) {
      this->switchHalos(this->pressures_double_, this->partition_indexing_);
    }
    else {
      this->switchHalos(this->pressures_, this->partition_indexing_);
    }
    cudasafe(cudaDeviceSynchronize(), "CudaMesh::switchHalos - Device synch after halo switch");
  }

  /// \brief switch pointers of current and past pressure meshes
  void flipPressurePointers() {
    for(unsigned int i = 0; i < this->getNumberOfPartitions(); i++){
      if(this->isDouble()) {
        double* temp = this->pressures_double_.at(i);
        this->pressures_double_.at(i) = this->pressures_past_double_.at(i);
        this->pressures_past_double_.at(i) = temp;
      }
      else {
        float* temp = this->pressures_.at(i);
        this->pressures_.at(i) = this->pressures_past_.at(i);
        this->pressures_past_.at(i) = temp;
      }
    }
  }

  /// \brief reset pressure values of the mesh to 0
  void resetPressures() {
    for(unsigned int i = 0; i < this->getNumberOfPartitions(); i++){
      float* temp = this->pressures_.at(i);
      float* temp_past = this->pressures_past_.at(i);
      unsigned int num_elements = this->getNumberOfElementsAt(i);
      resetData(num_elements, temp, i);
      resetData(num_elements, temp_past, i);
    }
    cudasafe(cudaDeviceSynchronize(), "Device synch after reset pressures");
  }
  
  /// \brief on hold
  void switchAirAndZero();
  
  ///////////////////////////////////////////////////////////////////////////////
  /// Setup the mesh data structure with voxelized geometry, simulation parameters
  /// and material parameters
  /// \param d_position_ptr Position indices defining the orientation of each node
  /// \param d_material_ptr Material indices defining the material index
  /// \param number_of_unique_materials Number of unique materials in the model
  /// \param material_coefficients material parameters used in the model
  /// \param voxelization_dim Dimensions of the voxelized geometry
  /// \param block_size Block size used which is to be used running FDTD kernels
  /// \param element_type Type of the position index
  ///                     0: forward-difference, 1: centered-difference
  /// \return sample value in host memory
  ///////////////////////////////////////////////////////////////////////////////
  void setupMesh(unsigned char* d_position_ptr,
                 unsigned char* d_material_ptr,
                 unsigned int number_of_unique_materials,
                 float* material_coefficients,
                 float* parameter_ptr,
                 uint3 voxelization_dim,
                 uint3 block_size, 
                 unsigned int element_type);
                 
  ////////////////////////////////////////////////////////////////////////////////
  /// Setup the mesh data structure with voxelized geometry, simulation parameters
  /// and material parameters
  /// \param d_position_ptr Position indices defining the orientation of each node
  /// \param d_material_ptr Material indices defining the material index 
  /// \param number_of_unique_materials Number of unique materials in the model
  /// \param material_coefficients material parameters used in the model
  /// \param voxelization_dim Dimensions of the voxelized geometry
  /// \param block_size Block size used which is to be used running FDTD kernels
  /// \param element_type Type of the position index
  ///                     0: forward-difference, 1: centered-difference
  /// \return sample value in host memory
  ///////////////////////////////////////////////////////////////////////////////
  void setupMeshDouble(unsigned char* d_position_ptr,
                       unsigned char* d_material_ptr,
                       unsigned int number_of_unique_materials,
                       double* material_coefficients,
                       double* parameter_ptr,
                       uint3 voxelization_dim,
                       uint3 block_size, 
                       unsigned int element_type);

};

/// \brief Pad the mesh with zeros
/// \param[in, out] d_mesh a 
/// \param[in, out] dim Dimensions of the mesh
/// \param block_size_x, block_size_y, block_size_z New block sizes in which the
/// new mesh dimensions are rounded to
void padWithZeros(unsigned char** d_mesh, uint3* dim, unsigned int block_size_x, 
                  unsigned int block_size_y, unsigned int block_size_z);

void padWithZeros(unsigned char** d_position_ptr, unsigned char** d_material_ptr, uint3* dim, 
                  unsigned int block_size_x, unsigned int block_size_y, 
                  unsigned int block_size_z);
/// \brief on hold
//void addMesh(CudaMesh& mesh_1, CudaMesh& mesh2,
//             int pos_x, int pos_y, int pos_z);

//////// Kernels
__global__ void padWithZerosKernel(unsigned char* d_mesh_new, 
                                   unsigned char* d_mesh_old,
                                   unsigned int dim_x, unsigned int dim_y, 
                                   unsigned int dim_z, 
                                   unsigned int block_x, unsigned int block_y, 
                                   unsigned int block_z, 
                                   unsigned int slice);

/// \brief Kernel setting the position index values to centered-difference scheme
__global__ void toKowalczykKernel(unsigned char* d_position_ptr, unsigned char* d_material_ptr, 
                                  unsigned int num_elems);

/// \brief Kernel setting the position index values to forward-difference scheme
__global__ void toBilbaoKernel(unsigned char* d_position_ptr, unsigned char* d_material_ptr, 
                               unsigned int num_elems);

/// \brief Kernel calculating the number of air and boundary nodes
__global__ void calcBoundaries(unsigned char* d_position_ptr, 
                               unsigned int* air, 
                               unsigned int* boundary, 
                               unsigned char air_value, 
                               unsigned char outside_value,
                               unsigned int num_elems);
/// \brief on hold                               
__global__ void getBoundaryIndicesKernel(unsigned char* d_position_ptr, 
                                         unsigned int* d_indices,
                                         unsigned char air, 
                                         unsigned char out, 
                                         unsigned int* pos);
/// \brief on hold
__global__ void checkInnerCorners(unsigned char* d_position_ptr, int dim_xy, int dim_x);

/// \brief on hold
__global__ void switchAirAndZeroKernel(unsigned char* d_position_ptr, unsigned int dim_xy, unsigned int dim_x);

/// \brief on hold
__global__ void validateSolids(unsigned char* d_position_ptr, unsigned char* d_position_ptr_new, 
                               unsigned char* d_material_ptr, int dim_x, int dim_xy, int dim_z);

/// \brief on hold                               
__global__ void validatePositionIndexes(unsigned char* d_position_ptr, unsigned char* d_position_ptr_new, 
                                        unsigned char* d_material_ptr, int dim_x, int dim_xy, int dim_z);
                                        
/// \brief A kernel to add a mesh to another, on hold                                        
__global__ void addSourceMesh(unsigned char* d_position_ptr_room, unsigned char* d_position_ptr_source,
                              int dim_xy_room, int dim_x_room,
                              int dim_xy_source, int dim_x_source,
                              int pos_x, int pos_y, int pos_z);
#endif
