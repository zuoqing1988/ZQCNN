
include_directories(${ZQCNN_INCLUDE_DIRS})
link_directories(${ZQCNN_LIBRARY_DIR})
file(GLOB ZQCNN_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp
                    ${CMAKE_CURRENT_LIST_DIR}/math/*.c
                    ${CMAKE_CURRENT_LIST_DIR}/layers_c/*.c)

add_library(ZQCNN ${ZQCNN_SRC})

if(WIN32)
    target_link_libraries(ZQCNN ZQ_GEMM mklml)
elseif(UNIX)
    target_link_libraries(ZQCNN ZQ_GEMM mklml_intel iomp5)
endif()
