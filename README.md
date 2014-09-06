Parallel FDTD
=============

A FDTD solver for room acoustics using CUDA

Dependencies:
- Boost Libraries, tested on 1.53.0 and 1.55.0,  http://www.boost.org/users/history/version_1_55_0.html, Accessed May 2014
- CUDA 5.0, tested on compute capability 3.0  
- Freeglut,  http://freeglut.sourceforge.net/ , Accessed May 2014  
- GLEW, tested on 1.9.0, http://glew.sourceforge.net/, Accessed May 2014  

The code is the most straightforward to compile with cmake. The cmake script contains  three targets: An executable which is to be used to check is the code running on the used machine and have and example of how the solver can be used from C++ code. Second target is a static library, which encapsulates the functionality. The static library is used with the MEX interface. Third target is a dynamic library compiled with boost::python to allow the usage of the solver as a module in python interpreter. The implementation of the python interface is not complete. The python interface is working on Ubuntu 12.04 LTS OS, but for Windows 7, some unresolved issues still persist.

For quick access to the functionality of the solver, precompiled Mex files, example scripts, some needed Matlab functions and dynamic libraries are downloadable from https://github.com/juuli/ParallelFDTD/releases. If the precompiled Mex files run on your machine, you can skip the Compiling stage and jump over to Practicalities.

Compiling
=========
To make sure the cmake is able to find the dependencies, it is highly possible that the cmake file at /ParallelFDTD/CMakeLists.txt has to edited to make the library and include directories match the system in use.

The compilation has been tested on Windows 7 with Visual Studio 2012 compilers and on Ubuntu 12.04 LTS with GCC 4.6.3.

###1. Download and install the dependencies  

### 2. Clone and build the voxelizer library from https://github.com/hakarlss/Voxelizer  

```
2.1 go to the folder of the repository  
2.2 git checkout next
2.3 mkdir build  
2.4 cd build  
```
### WINDOWS

```
2.5 cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=release ../  
2.6 nmake  
2.7 nmake install  
```
### Ubuntu

```
2.5 cmake ../
2.6 make 
2.7 make install
```

###4. Build ParallelFDTD with CMAKE  

```
4.1 go to the folder of the repository  
4.2 Copy the Voxelizer.lib / Voxelizer.a to folder /ParallelFDTD/lib/  
4.3 mkdir build  
4.4 cd build  
```

### WINDOWS

```
4.5 cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=release ../  
4.6 nmake  
4.7 nmake install  
```

### Ubuntu

```
4.5 cmake ../
4.6 make  
4.7 make install  
```

To build the tests and the python module (for linux only), use flags
```
-DBUILD_TESTS=on
-DBUILD_PYTHON=on
```
with the cmake command.


### 5. Compile MEX
```
  5.1 Start matlab  
  5.2 go to the folder /ParallelFDTD/matlab  
  5.3 Check that the library and include directories match the CMakeListst.txt
  5.4 type "compileWIN" / "compileUNIX" depending on the platform  
  5.5 if "compile done" message appears, test   
      the solver by running the testBench.m script
```

The solver should be good to go! Following the instruction in testBench.m it should be quite straightforward to generate run scripts for your needs.

Practicalities
==============

In order to run a simulation, some practical things have to be done:

### Make a model of the space
One convenient software choice for building models for simulation is SketchUp Make: http://www.sketchup.com/. The software is free for non-commercial use, and has a handy plugin interface that allows the user to write ruby scripts for geometry modifications and file IO.

A specific requirement for the geometry is that it should be solid/watertight. In practice this means that the geometry can not contain single planes or holes. The geometry has to have a volume. For example, a balcony rail can not be represented with a single plane, the rail has to be drawn as a box, or as a volume that is part of the balcony itself. The reason for this is that the voxelizer makes intersection tests in order to figure out whether a node it is processing is inside or outside of the geometry. Therefore, a plane floating inside a solid box will skrew this calculation up hence after intersecting this floating plane, the voxelizer thinks it has jumped out of the geometry, when actually it is inside. A hole in the geometry will do the opposite; if the voxelizer hits a hole in the boundary of the model, it does not know it is outside regardless  of the intentions of the creator of the model. A volume inside a volume is fine.

A couple of tricks/tools to check the model:
- In SketchUp, make a "Group" (Edit -> Make Group) out of the geometry you want to export, check the "Entity info" window (if not visible: Window -> Entity info). If the entity info shows a volume for your model, it should be useable with ParallelFDTD. (Thanks for Alex Southern for this trick).
- IF the entity info does not give you a volume, you need to debug your model. A good tool for this is "Solid Inspector" ( http://goo.gl/I4UcS6 ), with which you can track down holes and spare planes and edges from the model.


When the geometry has been constructed, it has to be exported to more simple format in order to import it to Matlab. For this purpose a JSON exported is provided. The repository contains a rubyscript plugin for SketchUp in the file:

```
geom2json.rb
```

Copy this file to the sketchup Plugins Folder. The plugin folder of the application varies depending on the operating system. Best bet is to check from the SketchUp help website http://help.sketchup.com/en/article/38583 (Accessed 5.9.2014). After installing the plugin and restarting SketchUp, an option "Export to JSON" under the "Tools" menu should have appeared. Now by selecting all the surfaces of the geometry which are to be exported and executing this command, a "save as" dialog box should appear. 

### Import the geometry to Matlab
The simulation tool does not rely on any specific CAD format. Any format which you have a Matlab importer available and that has the geometry described as list of vertex coordinates and triangle indices should do. As mentioned above, a requirement for the geometry is that it is solid/watertight. The format that is supported right away is JSON file that contains the vertice coordinates and triangle indices. Layer information is also really convenient for assigning materials for the model. JSON parser for Matlab (JSONlab by Qianqian Fang http://goo.gl/v2jnHx) is included in this repository. Python has a native support for JSON.

### Run the simulation
The details of running simulations are reviewed in the scripts matlab/testBench.m for matlab, and python/testBench.py for Python.

