
cuda_add_executable(CudaMeshTest ./CudaMeshTest.cpp ${SOURCES_CPP} ${SOURCES_CU} )
cuda_add_executable(CudaUtilsTest ./CudaUtilsTest.cpp ${SOURCES_CPP} ${SOURCES_CU} )
cuda_add_executable(FileReaderTest ./FileReaderTest.cpp ${SOURCES_CPP} ${SOURCES_CU} )
cuda_add_executable(GeometryHandlerTest ./GeometryHandlerTest.cpp ${SOURCES_CPP} ${SOURCES_CU} )
cuda_add_executable(MaterialHandlerTest ./MaterialHandlerTest.cpp ${SOURCES_CPP} ${SOURCES_CU} )

target_link_libraries( CudaMeshTest ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${Boost_LIBRARIES} ${GLEW_LIBRARIES} ${VOXELIZER_LIB}  )
target_link_libraries( CudaUtilsTest ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${Boost_LIBRARIES} ${GLEW_LIBRARIES} ${VOXELIZER_LIB}  )
target_link_libraries( FileReaderTest ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${Boost_LIBRARIES} ${GLEW_LIBRARIES} ${VOXELIZER_LIB}  )
target_link_libraries( GeometryHandlerTest ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${Boost_LIBRARIES} ${GLEW_LIBRARIES} ${VOXELIZER_LIB}  )
target_link_libraries( MaterialHandlerTest ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${Boost_LIBRARIES} ${GLEW_LIBRARIES} ${VOXELIZER_LIB}  )

#set_target_properties( CudaMeshTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/ )
#set_target_properties( CudaUtilsTest PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/ )
#set_target_properties( FileReaderTest PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/ )
#set_target_properties( GeometryHandlerTest PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/ )
