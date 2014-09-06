
cuda_add_executable( parallelFDTD ${SOURCES_CPP} ${CMAKE_SOURCE_DIR}/src/main.cpp ${SOURCES_CU} )
cuda_add_library( libParallelFDTD ${SOURCES_CPP} ${SOURCES_CU} )

if(BUILD_PYTHON)
  cuda_add_library( libPyFDTD SHARED ${SOURCES_CPP} ${SOURCES_CU} ${CMAKE_SOURCE_DIR}/src/AppPy.cpp)
endif()

target_link_libraries( libParallelFDTD ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${VOXELIZER_LIB}
                                       ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY} 
                                       ${Boost_DATE_TIME_LIBRARY} ${unix_specific_libraries})
                                      
target_link_libraries( parallelFDTD ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES}  ${GLEW_LIBRARIES} 
                                    ${VOXELIZER_LIB} ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY} 
                                    ${Boost_DATE_TIME_LIBRARY} ${Boost_CHRONO_LIBRARY} ${unix_specific_libraries})

if(BUILD_PYTHON)
target_link_libraries( libPyFDTD ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} 
                                 ${VOXELIZER_LIB}  ${unix_specific_libraries} 
                                 ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
endif()


set_target_properties( parallelFDTD PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/ )
set_target_properties( libParallelFDTD PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/ )

if(WIN32)
install(FILES ${CMAKE_BINARY_DIR}/lib/libParallelFDTD.lib ${CMAKE_SOURCE_DIR}/lib/Voxelizer.lib
        DESTINATION ${CMAKE_SOURCE_DIR}/matlab/lib/)
endif()
if(UNIX)
install(FILES ${CMAKE_BINARY_DIR}/lib/liblibParallelFDTD.a ${CMAKE_SOURCE_DIR}/lib/libVoxelizer.a
        DESTINATION ${CMAKE_SOURCE_DIR}/matlab/lib/)
endif()

install(FILES ${CMAKE_SOURCE_DIR}/src/base/cameraProto.hpp 
              ${CMAKE_SOURCE_DIR}/src/base/GeometryHandler.h
              ${CMAKE_SOURCE_DIR}/src/base/MaterialHandler.h
              ${CMAKE_SOURCE_DIR}/src/base/SimulationParameters.h
              ${CMAKE_SOURCE_DIR}/src/base/SrcRec.h
              DESTINATION ${CMAKE_SOURCE_DIR}/matlab/includes/base/)

install(FILES ${CMAKE_SOURCE_DIR}/src/gl/AppPbo.h
              ${CMAKE_SOURCE_DIR}/src/gl/AppVbo.h
              ${CMAKE_SOURCE_DIR}/src/gl/AppWindow.h
              ${CMAKE_SOURCE_DIR}/src/gl/glHelpers.h
              DESTINATION ${CMAKE_SOURCE_DIR}/matlab/includes/gl/)

install(FILES ${CMAKE_SOURCE_DIR}/src/io/FileReader.h
              ${CMAKE_SOURCE_DIR}/src/io/Image.h
              DESTINATION ${CMAKE_SOURCE_DIR}/matlab/includes/io/)
              
install(FILES ${CMAKE_SOURCE_DIR}/src/kernels/cudaMesh.h 
              ${CMAKE_SOURCE_DIR}/src/kernels/cudaUtils.h
              ${CMAKE_SOURCE_DIR}/src/kernels/kernels3d.h
              ${CMAKE_SOURCE_DIR}/src/kernels/visualizationUtils.h
              ${CMAKE_SOURCE_DIR}/src/kernels/voxelizationUtils.h
              DESTINATION ${CMAKE_SOURCE_DIR}/matlab/includes/kernels/)
              
install(FILES ${CMAKE_SOURCE_DIR}/src/math/geomMath.h
              ${CMAKE_SOURCE_DIR}/src/math/Matrix.h
              ${CMAKE_SOURCE_DIR}/src/math/Quaternion.h
              ${CMAKE_SOURCE_DIR}/src/math/Vector.h
              DESTINATION ${CMAKE_SOURCE_DIR}/matlab/includes/math/)
              
install(FILES ${CMAKE_SOURCE_DIR}/src/App.h
              ${CMAKE_SOURCE_DIR}/src/global_includes.h
              ${CMAKE_SOURCE_DIR}/src/logger.h
              DESTINATION ${CMAKE_SOURCE_DIR}/matlab/includes/)
