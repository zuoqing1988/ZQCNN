include_directories(${ZQCNN_INCLUDE_DIRS})
link_directories(${ZQCNN_LIBRARY_DIR})

file(GLOB sample_src ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.c)
add_executable(SampleGEMMCompare ${sample_src})
target_link_libraries(SampleGEMMCompare ZQ_GEMM)
if(WIN32)
    target_link_libraries(SampleGEMMCompare libiomp5md)
endif()
