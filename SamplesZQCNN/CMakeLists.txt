
MACRO(SUBDIRLIST result curdir)
FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
SET(dirlist "")
FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
        LIST(APPEND dirlist ${child})
    ENDIF()
ENDFOREACH()
SET(${result} ${dirlist})
ENDMACRO()

find_package(OpenCV QUIET)
  # OpenCV 是动态库时, 把运行时 dll 拷到产物目录, 否则 exe 起来就报
  # "error while loading shared libraries: opencv_world342.dll"
  if(WIN32 AND OpenCV_DIR)
    file(GLOB OpenCV_RUNTIME_DLLS "${OpenCV_DIR}/x64/vc14/bin/*.dll" "${OpenCV_DIR}/x64/vc15/bin/*.dll" "${OpenCV_DIR}/../bin/*.dll")
    if(OpenCV_RUNTIME_DLLS)
      foreach(_dll_dir ${ZQCNN_DLL_OUTPUT_DIRS})
        execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OpenCV_RUNTIME_DLLS} ${_dll_dir})
      endforeach()
    endif()
  endif()
if(NOT OpenCV_FOUND)
    # Fallback to the bundled Windows binaries
    if(WIN32 AND EXISTS ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv/build/x64/vc14/lib/opencv_world342.lib)
        include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv/build/include)
        set(OpenCV_LIBS ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv/build/x64/vc14/lib/opencv_world342.lib)
        set(OpenCV_FOUND TRUE)
        message(STATUS "OpenCV not found by find_package, using bundled 3rdparty/opencv")
    endif()
endif()
if(NOT OpenCV_FOUND)
    message(WARNING "OpenCV not found: skipping SamplesZQCNN (core ZQCNN library is unaffected)")
    return()
endif()

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(${ZQCNN_INCLUDE_DIRS})
link_directories(${ZQCNN_LIBRARY_DIR})

SUBDIRLIST(SAMPLE_SUBDIRS ${CMAKE_CURRENT_LIST_DIR})
foreach(SAMPLE_SUBDIR ${SAMPLE_SUBDIRS})
    message(STATUS "add sample ${SAMPLE_SUBDIR}")
    file(GLOB sample_src ${CMAKE_CURRENT_LIST_DIR}/${SAMPLE_SUBDIR}/*.cpp
                        ${CMAKE_CURRENT_LIST_DIR}/${SAMPLE_SUBDIR}/*.c)
    add_executable(${SAMPLE_SUBDIR} ${sample_src})
    # 2026-10-03（附录 GL.4）：这里原来是一段 12 行的 MATCHES 阶梯
    # （arm / arm64 / 其它 三个分支，每个分支里又是三层 MATCHES），
    # 三处问题叠在一起：
    #   1. `MATCHES "(Openblas|OPENBLAS|openblas|OpenBLAS)"` 是**子串**匹配，
    #      而 `openblas_zq_gemm` **含** "openblas" —— 于是
    #      `elseif(BLAS_TYPE MATCHES "openblas_zq_gemm")` 那一支是**死代码**，
    #      自动选路根本链不到 ZQ_GEMM，`zq_gemm_32f_..._special` 会在链接期
    #      变成未定义符号。根 CMakeLists.txt:122 的注释早就写明这个坑
    #      并在它自己那处修了，唯独这里没跟上。
    #   2. x86 那个 `else()` 分支**永远只链 ZQ_GEMM**，所以 x86 上
    #      `-DBLAS_TYPE=openblas` 链不到 OpenBLAS。
    #   3. `target_link_libraries(... openblas ...)` 引用的 `openblas`
    #      **不是任何地方定义过的 target**（全仓只有 add_library(ZQCNN)、
    #      (ZQ_GEMM)、(OpenMP::OpenMP_CXX) 三个），CMake 会把它当成
    #      一个原始库名去链接。
    # 现在一律读根 CMakeLists 算好的 ZQCNN_BLAS_BACKEND。
    #
    # 顺序有讲究：GNU ld 对静态库要求"被依赖者在前"，
    # 所以 ZQCNN 在前，OpenBLAS / ZQ_GEMM 在后。
    target_link_libraries(${SAMPLE_SUBDIR} ZQCNN ${OpenCV_LIBS})
    if(ZQCNN_BLAS_BACKEND STREQUAL "zq_gemm")
        target_link_libraries(${SAMPLE_SUBDIR} ZQ_GEMM)
    elseif(ZQCNN_BLAS_BACKEND STREQUAL "openblas")
        target_link_libraries(${SAMPLE_SUBDIR} ${ZQCNN_OPENBLAS_LIB})
    elseif(ZQCNN_BLAS_BACKEND STREQUAL "openblas_zq_gemm")
        target_link_libraries(${SAMPLE_SUBDIR} ${ZQCNN_OPENBLAS_LIB} ZQ_GEMM)
    elseif(ZQCNN_BLAS_BACKEND STREQUAL "mkl")
        target_link_libraries(${SAMPLE_SUBDIR} ${ZQCNN_MKL_LIB})
    endif()
endforeach()

