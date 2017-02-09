# CUDA and Boost root directories may be found from the CMake output
# when compiling libParallelFDTD.a
CUDA_ROOT := /usr/local/cuda-8.0
CUDA_INCLUDE := $(CUDA_ROOT)/include
CUDA_LIB := $(CUDA_ROOT)/lib64/
BOOST_INCLUE := /usr/local/include/
BOOST_LIB := /usr/local/lib/
CXX := mpic++
INCL := -I$(CUDA_INCLUDE) -I$(BOOST_INCLUDE) -I/usr/local/include -I/home/jukka/Documents/koodit/dev_ParallelFDTD/src
LIBS := -L$(CUDA_LIB) -L$(BOOST_LIB) -L/usr/lib/x86_64-linux-gnu/ -L/home/jukka/Documents/koodit/dev_ParallelFDTD/src/MPI libParallelFDTD.a libVoxelizer.a -lhdf5 -lhdf5_hl -lcudart -lboost_system -lboost_thread -lboost_date_time

all: MPI_main MPI_main_domain

MPI_main: MPI_main.cpp MPI_Utils.cpp
	$(CXX) -o $@ -g MPI_main.cpp MPI_Utils.cpp sha256.cpp $(INCL) $(LIBS)

MPI_main_domain: MPI_main_fulldomain.cpp MPI_Utils.cpp
	$(CXX) -o $@ -g MPI_main_fulldomain.cpp MPI_Utils.cpp sha256.cpp $(INCL) $(LIBS)

clean:
	rm MPI_main
	rm MPI_main_domain

.PHONY: clean
