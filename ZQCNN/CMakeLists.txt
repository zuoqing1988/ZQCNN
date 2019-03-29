
include_directories(${ZQCNN_INCLUDE_DIRS})
link_directories(${ZQCNN_LIBRARY_DIR})

file(GLOB ZQCNN_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp
                    ${CMAKE_CURRENT_LIST_DIR}/math/*.c
                    ${CMAKE_CURRENT_LIST_DIR}/layers_c/*.c
                    ${CMAKE_CURRENT_LIST_DIR}/layers_nchwc/*.c)

add_library(ZQCNN ${ZQCNN_SRC})

find_package(OpenMP)
# For CMake < 3.9, we need to make the target ourselves
if(NOT TARGET OpenMP::OpenMP_CXX)
    find_package(Threads REQUIRED)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX
            PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
    set_property(TARGET OpenMP::OpenMP_CXX
            PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
endif()

if(WIN32)
    target_link_libraries(ZQCNN ZQ_GEMM mklml)
elseif(UNIX)
    target_link_libraries(ZQCNN ZQ_GEMM gomp)
endif()
