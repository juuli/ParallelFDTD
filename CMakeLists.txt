CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
#The name of the project

# In case one needs to force a compiler, use these ones:
# set(CMAKE_CXX_COMPILER "/_path_to_bin/g++")
# set(CMAKE_C_COMPILER "/_path_to_bin/...")

project(parallelFDTD)

set(SOURCES_CPP ${CMAKE_SOURCE_DIR}/src/base/GeometryHandler.cpp
                ${CMAKE_SOURCE_DIR}/src/base/MaterialHandler.cpp
                ${CMAKE_SOURCE_DIR}/src/base/SimulationParameters.cpp
                ${CMAKE_SOURCE_DIR}/src/base/SrcRec.cpp
                ${CMAKE_SOURCE_DIR}/src/io/FileReader.cpp
                ${CMAKE_SOURCE_DIR}/src/io/Image.cpp
                ${CMAKE_SOURCE_DIR}/src/App.cpp
                ${CMAKE_SOURCE_DIR}/src/logger.cpp )

if(BUILD_VISUALIZATION)
  set(SOURCES_CPP ${SOURCES_CPP} ${CMAKE_SOURCE_DIR}/src/gl/AppWindow.cpp
                                 ${CMAKE_SOURCE_DIR}/src/gl/AppPbo.cpp
                                 ${CMAKE_SOURCE_DIR}/src/gl/AppVbo.cpp
                                 ${CMAKE_SOURCE_DIR}/src/gl/glHelpers.cpp
                                 ${CMAKE_SOURCE_DIR}/src/base/cameraProto.cpp)
endif()

set(SOURCES_CU ${CMAKE_SOURCE_DIR}/src/kernels/cudaUtils.cu
               ${CMAKE_SOURCE_DIR}/src/kernels/visualizationUtils.cu
               ${CMAKE_SOURCE_DIR}/src/kernels/voxelizationUtils.cu
               ${CMAKE_SOURCE_DIR}/src/kernels/cudaMesh.cu
               ${CMAKE_SOURCE_DIR}/src/kernels/kernels3d.cu )

set( GENERATE_DOXYGEN_DOCS OFF )

set( BOOST_ROOT "C:/Users/jks/Documents/boost_1_63_0")
set( Boost_USE_STATIC_LIBS ON )
set( Boost_USE_MULTITHREADED ON )
set( Boost_USE_STATIC_RUNTIME OFF )
#set( Boost_NO_BOOST_CMAKE ON )

#If "find package" command gives errors, define the paths manually
#set( BOOST_ROOT "C:/Program Files/boost/boost_1_55_0" )0
#set( Boost_INCLUDE_DIRS ${BOOST_ROOT})
set( BOOST_LIBRARYDIR ${BOOST_ROOT}/stage/lib)
#set( BOOST_LIBRARYDIR /usr/lib64)

if(WIN32)
  find_package( Boost 1.41 REQUIRED COMPONENTS system thread date_time chrono python unit_test_framework)
endif()

if(UNIX)
  find_package( Boost 1.41 REQUIRED COMPONENTS system thread date_time python unit_test_framework)
endif()

if(BUILD_VISUALIZATION)
  add_definitions(-DCOMPILE_VISUALIZATION)
  find_package( OpenGL REQUIRED )
  find_package( GLUT REQUIRED )
  find_package( GLEW REQUIRED )
  if(UNIX)
    find_library(GLEW_LIBRARIES libGLEW.so /usr/lib/x86_64-linux-gnu/)
  endif()
else()
  set(GLUT_LIBRARIES "")
  set(OPENGL_LIBRARIES "")
  set(GLEW_LIBRARIES "")
endif()

find_package( CUDA REQUIRED )
find_package( PythonLibs REQUIRED)

# MAC and LINUX: explicitly use x11 GL
# CHANGE THE PATHS ACCORDINGLY!

# Example directories for macports installs
if(APPLE)
  set(GLUT_INCLUDE_DIR /opt/X11/include/GL)
  set(GLUT_glut_LIBRARY /opt/X11/lib/libglut.dylib)

  link_directories(/opt/local/lib)
  link_directories(/opt/x11/lib)
endif()

if(UNIX)
  find_library(GLEW_LIBRARIES libGLEW.so /usr/lib/x86_64-linux-gnu/)
  link_directories(/usr/local/lib)
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -m64 -fPIC -O3 -DBOOST_PYTHON_DYNAMIC_MODULE")
  set(unix_specific_libraries pthread rt)
endif()

set( VOXELIZER_ROOT ${CMAKE_SOURCE_DIR}/src/Voxelizer )
find_path( VOXELIZER_INCLUDE_DIR voxelizer.h ${VOXELIZER_ROOT}/include )

if(WIN32)
  find_library( VOXELIZER_LIB Voxelizer.lib ${CMAKE_SOURCE_DIR}/lib)
elseif(UNIX)
  find_library( VOXELIZER_LIB libVoxelizer.a ${CMAKE_SOURCE_DIR}/lib)
endif()

#set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W4" )

if(Boost_USE_STATIC_RUNTIME)
  foreach(flag_var
          CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
          CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
     if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
     endif(${flag_var} MATCHES "/MD")
  endforeach(flag_var)
endif()

set( CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE};
                             -gencode arch=compute_61,code=sm_61
                             -gencode arch=compute_35,code=sm_35
                             -gencode arch=compute_37,code=sm_37
                             -gencode arch=compute_20,code=sm_20
                             )

set( CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};
                             -gencode arch=compute_61,code=sm_61
                             -gencode arch=compute_35,code=sm_35
                             -gencode arch=compute_37,code=sm_37
                             -gencode arch=compute_20,code=sm_20
                             )

include_directories( ${OPENGL_INCLUDE_DIR} )
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${GLUT_INCLUDE_DIR} )
include_directories( ${GLEW_INCLUDE_DIRS} )
include_directories( ${VOXELIZER_INCLUDE_DIR} )
include_directories( ${PYTHON_INCLUDE_DIRS} )
include_directories(${CMAKE_SOURCE_DIR}/include)

message( STATUS "CUDA root: ${CUDA_TOOLKIT_ROOT_DIR}")
message( STATUS "Voxelizer ROOT: ${VOXELIZER_ROOT}" )
message( STATUS "OpenGL include dir: ${OPENGL_INCLUDE_DIR}" )
message( STATUS "Boost include dir: ${Boost_INCLUDE_DIRS}" )
message( STATUS "GLUT include dir: ${GLUT_INCLUDE_DIR}" )
message( STATUS "GLEW include dirs: ${GLEW_INCLUDE_DIRS}" )
message( STATUS "Voxelizer include dir: ${VOXELIZER_INCLUDE_DIR}" )
message( STATUS "Python: ${PYTHON_INCLUDE_DIRS}" )

message( STATUS "GLUT lib: ${GLUT_glut_LIBRARY}" )
message( STATUS "OpenGL lib: ${OPENGL_gl_LIBRARY}" )
message( STATUS "Boost lib: ${Boost_LIBRARIES}" )
message( STATUS "Boost python lib: ${Boost_PYTHON_LIBRARY}" )
message( STATUS "GLEW libs: ${GLEW_LIBRARIES}" )
message( STATUS "Voxelizer lib: ${VOXELIZER_LIB}" )
message( STATUS "Python lib: ${PYTHON_LIBRARIES}" )

message( STATUS "CMAKE_C_COMPILER_VERSION : ${CMAKE_C_COMPILER_VERSION}" )
message( STATUS "CMAKE_CXX_COMPILER_VERSION : ${CMAKE_CXX_COMPILER_VERSION}" )

message( STATUS "CMAKE_C_COMPILER_VERSION : ${CMAKE_C_COMPILER_VERSION}" )
message( STATUS "CMAKE_CXX_COMPILER_VERSION : ${CMAKE_CXX_COMPILER_VERSION}" )

message( STATUS "CMAKE_C_COMPILER_VERSION : ${CMAKE_C_COMPILER_VERSION}" )
message( STATUS "CMAKE_CXX_COMPILER_VERSION : ${CMAKE_CXX_COMPILER_VERSION}" )

message( STATUS "Doxygen ${GENERATE_DOXYGEN_DOCS}")

add_subdirectory(src)

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

if ( GENERATE_DOXYGEN_DOCS )
  add_subdirectory( doc )
endif ()
