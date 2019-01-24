
include_directories(${ZQCNN_INCLUDE_DIRS})

file(GLOB ZQ_GEMM_SRC ${CMAKE_CURRENT_LIST_DIR}/math/*.c)
add_library(ZQ_GEMM ${ZQ_GEMM_SRC})