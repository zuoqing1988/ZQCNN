
include_directories(${ZQCNN_INCLUDE_DIRS})
link_directories(${ZQCNN_LIBRARY_DIR})

file(GLOB SAMPLE_GEMM_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp
                      ${CMAKE_CURRENT_LIST_DIR}/*.c)

add_executable(SampleGEMMAsmCompare ${SAMPLE_GEMM_SRC})

if(SIMD_ARCH_TYPE MATCHES "arm")
    target_link_libraries(SampleGEMMAsmCompare ZQ_GEMM)
else()
    target_link_libraries(SampleGEMMAsmCompare ZQ_GEMM)
endif()
