Parallel FDTD
=============

A FDTD solver for room acoustics using CUDA.

### Dependencies
- Boost Libraries, tested on 1.53.0, 1.55.0 and 1.56.00,  http://www.boost.org/users/history/version_1_55_0.html, Accessed May 2014
- CUDA 5.0-8.0, tested on compute capability 3.0 - 3.7, 5.2, 6.1
If visualization is compiled (set 'BUILD_VISUALIZATION' cmake flag - see below):
- Freeglut,  http://freeglut.sourceforge.net/ , Accessed May 2014  
- GLEW, tested on 1.9.0, http://glew.sourceforge.net/, Accessed May 2014  

### For MPI execution
- HDF5 libraries (serial libraries: hdf5 & hdf5_hl ; https://support.hdfgroup.org/HDF5/ )
- HDF5 for python (https://pypi.python.org/pypi/h5py accessed January 11. 2016)

## For the Python interface
- Python 2.7
- NumPy (Tested on 1.8.2), SciPy (Tested on 0.13.3)


The code is the most straightforward to compile with cmake. The cmake script contains  three targets: An executable which is to be used to check is the code running on the used machine and have and example of how the solver can be used from C++ code. Second target is a static library, which encapsulates the functionality. Third target is a dynamic library compiled with boost::python to allow the usage of the solver as a module in python interpreter.

The Matlab interface is not updated and is not guaranteed to work with the updated library.

Compiling
=========
To make sure the cmake is able to find the dependencies, it is highly possible that the cmake file at /ParallelFDTD/CMakeLists.txt has to edited to make the library and include directories match the system in use.

The compilation has been tested on:
- Ubuntu 14.04 LTS with GCC 4.8.4, CentOS 6, CentOS 7 with GCC 4.8.5
- Windows 7, Windows 10 with vc120, vc140 compilers

###1. Download and install the dependencies  

### 2. Clone and build the voxelizer library from https://github.com/hakarlss/Voxelizer  

Depending on the GPU card you have, add the CUDA compute capabilities to the compilation flags of the CMakeLists.txt file. An example for 6.1 compute capability:
```
set( CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE};
                             -gencode arch=compute_61,code=sm_61 
[...]
set( CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};
                             -gencode arch=compute_61,code=sm_61 
```


```
2.1 go to the folder of the repository  
2.2 git checkout next_2
2.3 mkdir build  
2.4 cd build  
```
### WINDOWS

You might have to set the boost variables in cmake:

```
set( BOOST_ROOT "C:/Program Files/boost/boost_1_55_0" )
set( Boost_INCLUDE_DIRS ${BOOST_ROOT})
set( BOOST_LIBRARYDIR ${BOOST_ROOT}/lib)
set( Boost_COMPILER "-vc140" )
set( BOOST_LIBRARYDIR /usr/lib64)
```
Depending on the GPU card you have, add the CUDA compute capabilities to the compilation flags of the CMakeLists.txt file. An example for 6.1 compute capability:
```
set( CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE};
                             -gencode arch=compute_61,code=sm_61 
[...]
set( CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};
                             -gencode arch=compute_61,code=sm_61 
```

Then open a VSxxxx (x64) Native Tools command prompt and follow the instructions:

```
2.5 cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=release ../  
2.6 nmake  
2.7 nmake install  
```

### Ubuntu / CentOS

```
2.5 cmake ../
2.6 make
2.7 make install
```

###4. Build ParallelFDTD with CMAKE  
To build the tests, python module (for linux only) and visualization, use the following flags. Real-time visualization is applicable only with a single GPU device. By default, the visualization is not compiled. The dependencies regarding the visualization naturally do not apply if compiled without the flag.
```
-DBUILD_TESTS=on
-DBUILD_PYTHON=on
-DBUILD_VISUALIZATION=on
```
with the cmake command.

```
4.1 go to the folder of the repository  
4.2 Copy the Voxelizer.lib / Voxelizer.a to folder /ParallelFDTD/lib/  
4.3 mkdir build  
4.4 cd build  
```

### WINDOWS

First copy the "Voxelizer.lib" library to the the ParallelFDTD folder\lib. 

You might have to set the boost variables in cmake:

```
set( BOOST_ROOT "C:/Program Files/boost/boost_1_55_0" )
set( Boost_INCLUDE_DIRS ${BOOST_ROOT})
set( BOOST_LIBRARYDIR ${BOOST_ROOT}/lib)
set( Boost_COMPILER "-vc140" )
set( BOOST_LIBRARYDIR /usr/lib64)
```
Depending on the GPU card you have, add the CUDA compute capabilities to the compilation flags of the CMakeLists.txt file. An example for 6.1 compute capability:
```
set( CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE};
                             -gencode arch=compute_61,code=sm_61 
[...]
set( CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};
                             -gencode arch=compute_61,code=sm_61 
```

Then open a VSxxxx (x64) Native Tools command prompt and follow the instructions:

```
4.5 cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=release ../  
4.6 nmake  
4.7 nmake install  
```

### Ubuntu / CentOS

```
4.5 cmake -DBUILD_PYTHON=on ../
4.6 make  
4.7 make install  
```

### 5. Compile MEX
```
  5.1 Start matlab  
  5.2 go to the folder /ParallelFDTD/matlab  
  5.3 Check that the library and include directories match the CMakeListst.txt
  5.4 type "compile"  
  5.5 if "compile done" message appears, test   
      the solver by running the testBench.m script
```

The solver should be good to go! Following the testBench.m it should be quite
straightforward to generate run scripts for your needs.

Practicalities
==============

In order to run a simulation, some practical things have to be done:

### Make a model of the space
One convenient software choice for building models for simulation is SketchUp Make: http://www.sketchup.com/. The software is free for non-commercial use, and has a handy plugin interface that allows the user to write ruby scripts for geometry modifications and file IO.

A specific requirement for the geometry is that it should be solid/watertight. In practice this means that the geometry can not contain single planes or holes. The geometry has to have a volume. For example, a balcony rail can not be represented with a single rectangle, the rail has to be drawn as a box, or as a volume that is part of the balcony itself. The reason for this is that the voxelizer makes intersection tests in order to figure out whether a node it is processing is inside or outside of the geometry. Therefore, a plane floating inside a solid box will skrew this calculation up hence after intersecting this floating plane, the voxelizer thinks it has jumped out of the geometry, when actually it is inside. A hole in the geometry will do the opposite; if the voxelizer hits a hole in the boundary of the model, it does not know it is outside regardless  of the intentions of the creator of the model. A volume inside a volume is fine.

A couple of tricks/tools to check the model:
- In SketchUp, make a "Group" (Edit -> Make Group) out of the geometry you want to export, check the "Entity info" window (if not visible: Window -> Entity info). If the entity info shows a volume for your model, it should be useable with ParallelFDTD. (Thanks for Alex Southern for this trick).
- IF the entity info does not give you a volume, you need to debug your model. A good tool for this is "Solid Inspector" ( goo.gl/I4UcS6 ), with which you can track down holes and spare planes and edges from the model.


When the geometry has been constructed, it has to be exported to more simple format in order to import it to Matlab. For this purpose a JSON exported is provided. The repository contains a rubyscript plugin for SketchUp in the file:

```
geom2json.rb
```

Copy this file to the sketchup Plugins Folder. The plugin folder of the application varies depending on the operating system. Best bet is to check from the SketchUp help website http://help.sketchup.com/en/article/38583 (Accessed 5.9.2014). After installing the plugin and restarting SketchUp, an option "Export to JSON" under the "Tools" menu should have appeared. Now by selecting all the surfaces of the geometry which are to be exported and executing this command, a "save as" dialogue should appear.

### Import the geometry to Matlab
The simulation tool does not rely on any specific CAD format. Any format which you have a Matlab importer available and that has the geometry described as list of vertices coordinates and triangle indices should do. As mentioned above, a requirement for the geometry is that it is solid/watertight. The format that is supported right away is JSON file that contains the vertice coordinates and triangle indices. Layer information is also really convenient for assigning materials for the model. JSON parser for Matlab (JSONlab by Qianqian Fang goo.gl/v2jnHx) is included in this repository.

### Run the simulation
The details of running simulations are reviewed in the scripts matlab/testBench.m for matlab, and python/testBench.py for Python.
