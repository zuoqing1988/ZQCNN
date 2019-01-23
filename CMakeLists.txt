cmake_minimum_required(VERSION 3.0)

project(ZQCNN)

if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
		add_compile_options(-mavx2)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		add_compile_options(/QxAVX2)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		add_compile_options(/arch:AVX2)
endif()

set(ZQCNN_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/ZQ_GEMM
                        ${CMAKE_CURRENT_LIST_DIR}/ZQCNN
                        ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include                        
                        CACHE INTERNAL "ZQCNN include dirs")            
set(ZQCNN_LIBRARY_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/lib)                                    

add_subdirectory(ZQ_GEMM)
add_subdirectory(ZQCNN)
add_subdirectory(SamplesZQCNN)
add_subdirectory(SamplesZQlibFaceID)
