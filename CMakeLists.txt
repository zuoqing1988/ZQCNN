cmake_minimum_required(VERSION 3.0)

project(ZQCNN)

set(CMAKE_CXX_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

if(NOT SIMD_ARCH_TYPE)
    set(SIMD_ARCH_TYPE x86)
endif()

if(NOT BLAS_TYPE)
    set(BLAS_TYPE ZQ_GEMM)
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(archbit "x64")
else()
    set(archbit "x86")
endif()
if(WIN32)
    set(archos "win32")
elseif(UNIX)
    set(archos "unix")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_LIST_DIR}/cmake-out-${archos}-${archbit}/${CMAKE_BUILD_TYPE})
#message("CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# 多配置生成器(Visual Studio)会把可执行文件再放进 $(Configuration) 子目录,
# dll 也要跟着放, 否则 exe 起来就报缺少运行库
set(ZQCNN_DLL_OUTPUT_DIRS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
if(CMAKE_CONFIGURATION_TYPES)
    list(APPEND ZQCNN_DLL_OUTPUT_DIRS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release)
endif()

if(WIN32)
    # MKLINK is a cmd builtin (no MKLINK.exe exists) and directory junctions do not
    # require admin privileges, unlike symlinks. Link is the first argument.
    string(REPLACE "/" "\\" _win_data_src "${CMAKE_CURRENT_LIST_DIR}/data")
    string(REPLACE "/" "\\" _win_model_src "${CMAKE_CURRENT_LIST_DIR}/model")
    # 多配置生成器(Visual Studio)下可执行文件在 $(Configuration) 子目录里，
    # data/model 联接要跟着放，否则示例在产物目录里跑会 "empty image"
    foreach(_link_dir ${ZQCNN_DLL_OUTPUT_DIRS})
        string(REPLACE "/" "\\" _win_dst "${_link_dir}")
        if(NOT EXISTS ${_link_dir}/data)
            execute_process(COMMAND cmd /c mklink /J "${_win_dst}\\data" "${_win_data_src}" RESULT_VARIABLE _mklink_r1)
            if(_mklink_r1)
                execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/data ${_link_dir}/data)
            endif()
        endif()
        if(NOT EXISTS ${_link_dir}/model)
            execute_process(COMMAND cmd /c mklink /J "${_win_dst}\\model" "${_win_model_src}" RESULT_VARIABLE _mklink_r2)
            if(_mklink_r2)
                execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/model ${_link_dir}/model)
            endif()
        endif()
    endforeach()
    # 第三方 dll（caffe/libfacedetect/SeetaFace/OpenMP 等）拷到产物目录，示例才能直接跑
    file(GLOB ZQCNN_RUNTIME_DLLS ${CMAKE_CURRENT_LIST_DIR}/3rdparty/bin/*.dll)
    if(ZQCNN_RUNTIME_DLLS)
        foreach(_dll_dir ${ZQCNN_DLL_OUTPUT_DIRS})
            execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ZQCNN_RUNTIME_DLLS} ${_dll_dir})
        endforeach()
    endif()
elseif(UNIX)
    execute_process(COMMAND "ln" -nsf ${CMAKE_CURRENT_LIST_DIR}/data ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data)
    execute_process(COMMAND "ln" -nsf ${CMAKE_CURRENT_LIST_DIR}/model ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/model)
endif()

if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    #add_definitions(-Wall -Wextra -Wno-unused-function)
    add_definitions(-fPIC)
    if(CMAKE_BUILD_TYPE MATCHES "(Release|RELEASE|release)")
        add_definitions(-Ofast)
        add_definitions(-ffast-math)
    endif()
    if(SIMD_ARCH_TYPE MATCHES "(arm|arm64|arm64-fp16)")
        add_definitions(-DZQ_CNN_USE_ARM_NEON)
        if(SIMD_ARCH_TYPE MATCHES "arm64|arm64-fp16")
            add_definitions(-march=armv8-a)
            add_definitions(-DZQ_CNN_USE_ARM_NEON_ARMV8)
            add_definitions(-DZQ_CNN_USE_FMADD128)
            if(SIMD_ARCH_TYPE MATCHES "arm64-fp16")
                add_definitions(-DZQ_CNN_USE_ARM_NEON_FP16)
                add_definitions(-mfpu=neon-fp-armv8)
            endif()
        else()
            #add_definitions(-march=native)
            add_definitions(-mfpu=neon)
            add_definitions(-mfloat-abi=softfp)
        endif()
        if(BLAS_TYPE MATCHES "zq_gemm")
            add_definitions(-DZQ_CNN_USE_ZQ_GEMM=1)
        elseif(BLAS_TYPE MATCHES "openblas")
            add_definitions(-DZQ_CNN_USE_BLAS_GEMM=1)
        elseif(BLAS_TYPE MATCHES "openblas_zq_gemm")
            add_definitions(-DZQ_CNN_USE_BOTH_BLAS_ZQ_GEMM)
        endif()
    else()
        add_compile_options(-mavx2)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    add_compile_options(/QxAVX2)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # 源码里含中文注释，不加 /utf-8 时 MSVC 会按系统 ANSI 代码页解码，
    # 多字节序列可能被误判成引号/反斜杠导致莫名其妙的语法错误
    add_compile_options(/utf-8 /arch:AVX2)
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
add_subdirectory(SamplesZQGEMM)
add_subdirectory(SamplesZQBLAS)

