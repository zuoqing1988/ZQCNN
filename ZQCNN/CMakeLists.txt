
include_directories(${ZQCNN_INCLUDE_DIRS})
link_directories(${ZQCNN_LIBRARY_DIR})

file(GLOB ZQCNN_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp
                    ${CMAKE_CURRENT_LIST_DIR}/math/*.c
                    ${CMAKE_CURRENT_LIST_DIR}/layers_c/*.c
                    ${CMAKE_CURRENT_LIST_DIR}/layers_nchwc/*.c)

# libncnn.a 里的 OpenMP SIMD 调用需要 __*_finite 的向量化体，必须带 -fopenmp 编
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/math/zq_libm_compat.c
        PROPERTIES COMPILE_FLAGS "-fopenmp" COMPILE_OPTIONS "-fopenmp")
endif()

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
    target_link_libraries(ZQCNN ZQ_GEMM)
    # MKL 只在使用 BLAS 时才需要（ZQ_CNN_USE_BLAS_GEMM），否则所有 exe 都会在运行时
    # 依赖 mklml.dll，没有装 MKL 运行库的机器直接起不来
    if(BLAS_TYPE MATCHES "openblas")
        target_link_libraries(ZQCNN mklml)
    endif()
elseif(UNIX)
    target_link_libraries(ZQCNN ZQ_GEMM gomp)
endif()
