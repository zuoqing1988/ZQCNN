
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

# BLAS 库在这里按 **根 CMakeLists 算好的 ZQCNN_BLAS_BACKEND** 链接，
# 不要再自己 MATCHES BLAS_TYPE（2026-10-03 改，附录 GL.3）。
#
# 原来这一段是：
#     if(WIN32)
#         target_link_libraries(ZQCNN ZQ_GEMM)
#         if(BLAS_TYPE MATCHES "openblas")
#             target_link_libraries(ZQCNN mklml)      # <- 用户要 openblas，链的是 MKL
#         endif()
#     elseif(UNIX)
#         target_link_libraries(ZQCNN ZQ_GEMM gomp)  # <- UNIX 分支根本不链 BLAS
#     endif()
# 三个问题叠在一起：
#   1. `MATCHES` 是**子串**匹配，openblas_zq_gemm 也会命中
#      —— 根 CMakeLists.txt:122 的注释早就写明这个坑并在**它自己那处**修了，
#      唯独这两处没跟上；
#   2. 用户要 OpenBLAS，链上去的却是 `mklml`；
#   3. UNIX 分支压根没有 BLAS 链接，所以 Linux 上 -DBLAS_TYPE=openblas
#      即使宏修对了也**链接不过**。
if(WIN32)
    target_link_libraries(ZQCNN ZQ_GEMM)
elseif(UNIX)
    target_link_libraries(ZQCNN ZQ_GEMM gomp)
endif()

if(ZQCNN_NEED_OPENBLAS)
    target_link_libraries(ZQCNN ${ZQCNN_OPENBLAS_LIB})
endif()

if(ZQCNN_NEED_MKL)
    # MKL 只在真的要用它时才链，否则所有 exe 都会在运行时依赖 mklml.dll，
    # 没装 MKL 运行库的机器直接起不来（这是 2026-10-02 把它默认关掉的原因）。
    target_link_libraries(ZQCNN ${ZQCNN_MKL_LIB})
endif()
