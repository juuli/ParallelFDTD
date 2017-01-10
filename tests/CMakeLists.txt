cuda_add_executable(AppTest ./AppTest.cpp ${SOURCES_CPP} ${SOURCES_CU} )
cuda_add_executable(CudaMeshTest ./CudaMeshTest.cpp ${SOURCES_CPP} ${SOURCES_CU} )
cuda_add_executable(CudaUtilsTest ./CudaUtilsTest.cpp ${SOURCES_CPP} ${SOURCES_CU} )
cuda_add_executable(FileReaderTest ./FileReaderTest.cpp ${SOURCES_CPP} ${SOURCES_CU} )
cuda_add_executable(GeometryHandlerTest ./GeometryHandlerTest.cpp ${SOURCES_CPP} ${SOURCES_CU} )
cuda_add_executable(MaterialHandlerTest ./MaterialHandlerTest.cpp ${SOURCES_CPP} ${SOURCES_CU} )
cuda_add_executable(VoxelizerTest ./VoxelizerTest.cpp ${SOURCES_CPP} ${SOURCES_CU} )

message(STATUS "Unit test framework: ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ")
message(STATUS "Boost system: ${Boost_SYSTEM_LIBRARY} ")

target_link_libraries( AppTest  ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES}
                                ${VOXELIZER_LIB}
                                ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY}
                                ${Boost_DATE_TIME_LIBRARY} ${unix_specific_libraries}
                                ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

target_link_libraries( CudaMeshTest ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES}
                                    ${VOXELIZER_LIB}
                                    ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY}
                                    ${Boost_DATE_TIME_LIBRARY} ${unix_specific_libraries}
                                    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

target_link_libraries( CudaUtilsTest ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES}
                                    ${VOXELIZER_LIB}
                                    ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY}
                                    ${Boost_DATE_TIME_LIBRARY} ${unix_specific_libraries}
                                    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

target_link_libraries( FileReaderTest ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES}
                                      ${VOXELIZER_LIB}
                                      ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY}
                                      ${Boost_DATE_TIME_LIBRARY} ${unix_specific_libraries}
                                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

target_link_libraries( GeometryHandlerTest ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES}
                                    ${VOXELIZER_LIB}
                                    ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY}
                                    ${Boost_DATE_TIME_LIBRARY} ${unix_specific_libraries}
                                    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

target_link_libraries( MaterialHandlerTest ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES}
                                    ${VOXELIZER_LIB}
                                    ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY}
                                    ${Boost_DATE_TIME_LIBRARY} ${unix_specific_libraries}
                                    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

target_link_libraries( VoxelizerTest ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES}
                                    ${VOXELIZER_LIB}
                                    ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY}
                                    ${Boost_DATE_TIME_LIBRARY} ${unix_specific_libraries}
                                    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
