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
# 多配置生成器的 $(Configuration) 子目录要在这里就建出来: 否则首次 configure 时
# mklink /J 会因为目标目录不存在而失败, 回退成 copy_directory, 之后仓库里新增的
# 测试图片就不会出现在产物目录里了(而且再也刷不回联接)
foreach(_dll_dir ${ZQCNN_DLL_OUTPUT_DIRS})
    file(MAKE_DIRECTORY ${_dll_dir})
endforeach()

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
    else()
        # -mfma 不能省: ZQA_HAVE_FMA 的判定是
        #   (ZQ_CNN_USE_FMADD256 && ...) || defined(__FMA__)
        # 而 __FMA__ **只有** -mfma / -march=haswell 及以上才会定义 —— gcc 的
        # -mavx2 **不隐含** FMA3。只给 -mavx2 时, zq_gemm_32f_align_c_asm.c 里
        # 每一处 vfmadd231ps 都会退化成 ZQA_FMA 的 vmulps + vaddps 两条指令
        # (再加一条 mov 到 ymm15 暂存), 内层循环的发射量翻倍。
        # 实测 gcc 9.4: -mavx2 -> __FMA__=0; -mavx2 -mfma -> __FMA__=1。
        # 安全性: 运行时的 ISA 守卫 (zq_gemm_32f_asm_isa_usable) 判的就是
        # "AVX2 **且** FMA", 两者一致; 而且这条分支本来就要求 AVX2。
        add_compile_options(-mavx2 -mfma)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    add_compile_options(/QxAVX2)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/utf-8 /arch:AVX2)
endif()

# 必须在下面 find_library 之前: HINTS 要用这个路径去 3rdparty/lib 里找。
set(ZQCNN_LIBRARY_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/lib)

# BLAS_TYPE 与 SIMD_ARCH_TYPE 无关, 之前这段嵌在 ARM 分支里, x86 上 -DBLAS_TYPE 完全无效;
# 而且 MATCHES 是子串匹配: 默认的 ZQ_GEMM 匹配不到 "zq_gemm", openblas_zq_gemm 又会被
# "openblas" 抢先命中。这里改成显式分支 + 精确比较。
#
# 2026-10-03（附录 GL）：本文件这一段本来就是对的, 但**下游三处都在自己
# 重新推导 BLAS_TYPE**, 而且用的是子串匹配:
#   * ZQCNN/CMakeLists.txt:34  `if(BLAS_TYPE MATCHES "openblas")` -> 链 **mklml**
#   * SamplesZQCNN/CMakeLists.txt:53/61 同一形状的 MATCHES 阶梯
# 结果是 -DBLAS_TYPE 在**两个平台都不可能生效**（实测见附录 GL.1）。
# 修法不是"再改一处 MATCHES"，而是**在这里算一次、导出成变量**，
# 让子目录只读变量、不再自己判断字符串。
string(TOLOWER "${BLAS_TYPE}" ZQCNN_BLAS_TYPE_LOWER)
if(ZQCNN_BLAS_TYPE_LOWER STREQUAL "openblas_zq_gemm")
    # 自动选路（运行时先试 ZQ_GEMM，失败再回落 cblas）**只在 ARM 分支里实现**：
    # 那 16 处 `#if __ARM_NEON && ZQ_CNN_USE_ZQ_GEMM && ZQ_CNN_USE_BLAS_GEMM`
    # 派发点（ZQCNN/layers_c/ 与 layers_nchwc/ 各 8 处）x86 上没有对应实现。
    # 原来在 x86 上这个值被**静默忽略**（宏从头到尾没人读），用户以为切了，
    # 实际跑的还是 ZQ_GEMM。**显式报错**比静默忽略好。
    if(NOT (SIMD_ARCH_TYPE MATCHES "^(arm|arm64|arm64-fp16)$"))
        message(FATAL_ERROR
            "BLAS_TYPE=openblas_zq_gemm 只在 SIMD_ARCH_TYPE=arm/arm64 上有意义：\n"
            "  自动选路的派发点只存在于 __ARM_NEON 分支"
            "（`#if __ARM_NEON && ZQ_CNN_USE_ZQ_GEMM && ZQ_CNN_USE_BLAS_GEMM`，共 16 处）。\n"
            "  x86 上这个值没有任何代码会读它 —— 原来是被静默忽略，"
            "所以这里改成 configure 直接失败。\n"
            "  x86 上想用 OpenBLAS 请用 -DBLAS_TYPE=openblas。")
    endif()
    add_definitions(-DZQ_CNN_USE_BOTH_BLAS_ZQ_GEMM=1)
    set(ZQCNN_BLAS_BACKEND "openblas_zq_gemm")
elseif(ZQCNN_BLAS_TYPE_LOWER STREQUAL "openblas")
    add_definitions(-DZQ_CNN_USE_BLAS_GEMM=1)
    set(ZQCNN_BLAS_BACKEND "openblas")
elseif(ZQCNN_BLAS_TYPE_LOWER STREQUAL "mkl")
    add_definitions(-DZQ_CNN_USE_MKL_GEMM=1)
    set(ZQCNN_BLAS_BACKEND "mkl")
elseif(ZQCNN_BLAS_TYPE_LOWER STREQUAL "zq_gemm")
    add_definitions(-DZQ_CNN_USE_ZQ_GEMM=1)
    set(ZQCNN_BLAS_BACKEND "zq_gemm")
else()
    message(FATAL_ERROR
        "BLAS_TYPE 只能是 zq_gemm / openblas / mkl / openblas_zq_gemm 之一，"
        "收到的是 '${BLAS_TYPE}'。\n"
        "  早先这里是**静默**落空（三个 STREQUAL 都不匹配 -> 什么都不定义 -> "
        "落到头文件默认值），拼错一个字母就悄无声息地退回 ZQ_GEMM。")
endif()

# 子目录一律读这个变量，**不要**再自己 MATCHES BLAS_TYPE。
# openblas_zq_gemm 不在这里加进 openblas 链接 —— 自动选路要两个库都在，
# 由 ZQCNN/CMakeLists.txt 按 ZQCNN_BLAS_BACKEND 决定。
if(ZQCNN_BLAS_BACKEND STREQUAL "openblas" OR ZQCNN_BLAS_BACKEND STREQUAL "openblas_zq_gemm")
    set(ZQCNN_NEED_OPENBLAS TRUE)
else()
    set(ZQCNN_NEED_OPENBLAS FALSE)
endif()
if(ZQCNN_BLAS_BACKEND STREQUAL "mkl")
    set(ZQCNN_NEED_MKL TRUE)
else()
    set(ZQCNN_NEED_MKL FALSE)
endif()
message(STATUS "ZQCNN BLAS backend: ${ZQCNN_BLAS_BACKEND}"
               " (openblas=${ZQCNN_NEED_OPENBLAS} mkl=${ZQCNN_NEED_MKL})")

# 判断一个库文件的**目标架构**，结果写进 out_ok / out_why。
#
# 为什么需要：仓库随附的 `3rdparty/lib/libopenblas.{so,a}` 是 **ARM** 的
# （实测：.so 是 "ELF 32-bit LSB shared object, ARM"；.a 的成员 e_machine=0x28）
# —— 它是给 `build.sh` 的 armeabi-v7a 用的，不是给 x86 的。
# 而 find_library 只按**文件名**找，于是 x86 上它会找到那个 ARM 库，
# 链接时报一句 `error adding symbols: file in wrong format`，
# 与"库不存在"完全不同的原因、同样难查的错误。
# 所以这里读 ELF/PE 头自己判一遍。
#
# 只读文件**前 20 字节**，不执行任何东西。
function(zq_check_lib_arch path out_ok out_why)
    set(${out_ok} TRUE PARENT_SCOPE)
    set(${out_why} "" PARENT_SCOPE)
    if(NOT EXISTS "${path}")
        set(${out_ok} FALSE PARENT_SCOPE)
        set(${out_why} "文件不存在" PARENT_SCOPE)
        return()
    endif()
    # 读前 256 KB：ELF 只需 20 字节，而 ar 归档要顺着成员表走到
    # 第一个**真的**是 ELF 的成员（见下面）。16 MB 的库读 256 KB 可以忽略。
    file(READ "${path}" _head LIMIT 262144 HEX)
    string(TOLOWER "${_head}" _head)
    string(LENGTH "${_head}" _have)          # hex 字符数

    # PE / MZ：Windows 的 .lib/.dll 导入库与 dll
    string(SUBSTRING "${_head}" 0 4 _mz)
    if(_mz STREQUAL "4d5a")
        if(CMAKE_SIZEOF_VOID_P EQUAL 4)
            set(${out_ok} FALSE PARENT_SCOPE)
            set(${out_why} "这是 32 位 PE 库，而当前工具链是 64 位" PARENT_SCOPE)
        endif()
        return()
    endif()

    # ELF：magic 是 7f 45 4c 46
    string(SUBSTRING "${_head}" 0 8 _magic)
    set(_off 0)
    if(NOT _magic STREQUAL "7f454c46")
        if(NOT _magic STREQUAL "213c6172")
            set(${out_ok} FALSE PARENT_SCOPE)
            set(${out_why} "既不是 ELF 也不是 PE，也不是 ar 归档（头 4 字节 ${_magic}）"
                PARENT_SCOPE)
            return()
        endif()
        # ar 归档。**不能简单跳过** —— 随附的 libopenblas.a 和
        # libopenblas.so 一样是 ARM 的，跳过检查等于把同一个洞留在静态库一侧。
        #
        # 但也**不能按成员表走**：ar 的 size 字段是 **ASCII 十进制文本**，
        # 而这里读的是 `file(READ ... HEX)` —— 拿到的是**数字的 hex**
        # （"100196" 读成 "313030313936"），CMake 没有 hex->字符串的原语，
        # 算不出下一个成员的起点。第一版按成员表走，结果走一步就断了。
        #
        # 改成**直接在整块 hex 里找第一个 ELF magic**。全局头只有 8 字节
        # （16 个 hex 字符），所以第一个命中必然在成员数据里。
        # 万一在符号表里撞上（100 KB 上约 2.4e-5 的概率），
        # 后面"识别不出架构就放行"那条会兜住 —— **本函数绝不能
        # 因为自己没读懂格式就把一次本来能成功的构建弄失败。**
        string(FIND "${_head}" "7f454c46" _pos)
        if(_pos LESS 16)
            return()          # 归档里没找到 ELF（例如 MSVC 的 .lib 导入库）
        endif()
        math(EXPR _off "${_pos} / 2")
    endif()

    # _off 是 ELF 头在 _head 里的**字节**偏移（0 = 文件本身，68 = ar 第一个成员）
    math(EXPR "_h4" "${_off} * 2 + 4")           # EI_CLASS
    math(EXPR "_h5" "${_off} * 2 + 10")          # EI_DATA
    math(EXPR "_hm" "${_off} * 2 + 36")          # e_machine，字节 18-19
    string(SUBSTRING "${_head}" ${_h4} 2 _class)   # EI_CLASS: 01=32bit 02=64bit
    string(SUBSTRING "${_head}" ${_h5} 2 _data)   # EI_DATA: 01=LE 02=BE
    string(SUBSTRING "${_head}" ${_hm} 4 _machine)  # e_machine
    # e_machine 是**目标字节序**存放的：小端 ELF 里 x86-64 存成 "3e 00"，
    # 不是 "00 3e"。第一版直接拿数值形式去比，结果 ARM 库被判成
    # "未知(machine=0x2800)" —— 报错信息里的架构名是错的，
    # 而**报错信息本身是这次改动要交付的东西**。
    if(_data STREQUAL "02")                     # 大端
        string(SUBSTRING "${_machine}" 0 2 _hi)
        string(SUBSTRING "${_machine}" 2 2 _lo)
        set(_machine "${_lo}${_hi}")
    endif()
    # 3e=x86-64 03=x86(32) 28=ARM(32) b7=AArch64（已按小端拼法）
    set(_arch "未知(e_machine=${_machine})")
    if(_machine STREQUAL "3e00")
        set(_arch "x86-64")
    elseif(_machine STREQUAL "0300")
        set(_arch "x86(32 位)")
    elseif(_machine STREQUAL "2800")
        set(_arch "ARM(32 位)")
    elseif(_machine STREQUAL "b700")
        set(_arch "AArch64")
    endif()

    # 期望值：SIMD_ARCH_TYPE 说 arm 就期望 ARM，否则按指针宽度定 x86
    if(SIMD_ARCH_TYPE MATCHES "arm64")
        set(_want "AArch64")
    elseif(SIMD_ARCH_TYPE MATCHES "arm")
        set(_want "ARM(32 位)")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_want "x86-64")
    else()
        set(_want "x86(32 位)")
    endif()

    # **只在我们确实认出了架构、而且与目标不符时才拒绝。**
    # 认不出（新的 e_machine、或 hex 里撞上了符号表）就放行：
    # 本函数的职责是把话说清楚，不是当判官。
    if(_arch MATCHES "^未知")
        return()
    endif()
    if(NOT _arch STREQUAL _want)
        set(${out_ok} FALSE PARENT_SCOPE)
        set(${out_why} "这个库是 ${_arch} 的，而当前构建要的是 ${_want}" PARENT_SCOPE)
    endif()
endfunction()

# 库文件在这里**发现一次**，导出成变量给所有子目录用。
# 放在子目录里做 find_library 的话，变量留在那个子目录的作用域里，
# 兄弟目录（SAMPLES*）看不到 —— 而静态库的依赖最终要由**可执行文件**提供。
if(ZQCNN_NEED_OPENBLAS)
    # 仓库随附的是 3rdparty/lib/libopenblas.a 与 libopenblas.so，
    # 但它们是 **ARM** 版（见上面 zq_check_lib_arch 的注释）；
    # Windows 上则**根本没有** openblas.lib。
    if(WIN32)
        find_library(ZQCNN_OPENBLAS_LIB
                     NAMES openblas libopenblas
                     HINTS ${ZQCNN_LIBRARY_DIR}
                     NO_DEFAULT_PATH)
        if(NOT ZQCNN_OPENBLAS_LIB)
            message(FATAL_ERROR
                "BLAS_TYPE=${ZQCNN_BLAS_BACKEND} 需要 OpenBLAS，"
                "但 3rdparty/lib 下没有 openblas.lib / libopenblas.lib。\n"
                "  仓库随附的那份是 **ARM(32 位)** 的，给 build.sh 的 armeabi-v7a 用。\n"
                "  Windows/x86 上请自己放一份对应架构的 OpenBLAS 进 3rdparty/lib，"
                "或者改用默认的 -DBLAS_TYPE=zq_gemm。")
        endif()
    else()
        find_library(ZQCNN_OPENBLAS_LIB
                     NAMES openblas
                     HINTS ${ZQCNN_LIBRARY_DIR})
        if(NOT ZQCNN_OPENBLAS_LIB)
            message(FATAL_ERROR
                "BLAS_TYPE=${ZQCNN_BLAS_BACKEND} 需要 OpenBLAS，但找不到 libopenblas。")
        endif()
        zq_check_lib_arch("${ZQCNN_OPENBLAS_LIB}" _ok _why)
        if(NOT _ok)
            message(FATAL_ERROR
                "BLAS_TYPE=${ZQCNN_BLAS_BACKEND} 找到的 OpenBLAS 用不了：\n"
                "  ${ZQCNN_OPENBLAS_LIB}\n"
                "  ${_why}\n"
                "  仓库随附的 3rdparty/lib/libopenblas.{so,a} 是 **ARM** 版"
                "（为 build.sh 的 armeabi-v7a 准备的）。\n"
                "  x86 上要用 OpenBLAS，请自己下载对应架构的版本放进 3rdparty/lib，"
                "或者改用默认的 -DBLAS_TYPE=zq_gemm。")
        endif()
    endif()
    message(STATUS "OpenBLAS: ${ZQCNN_OPENBLAS_LIB}")
endif()

if(ZQCNN_NEED_MKL)
    if(WIN32)
        find_library(ZQCNN_MKL_LIB NAMES mklml
                     HINTS ${ZQCNN_LIBRARY_DIR} NO_DEFAULT_PATH)
    else()
        find_library(ZQCNN_MKL_LIB NAMES mkl HINTS ${ZQCNN_LIBRARY_DIR})
    endif()
    if(NOT ZQCNN_MKL_LIB)
        message(FATAL_ERROR
            "BLAS_TYPE=mkl 需要 MKL，但没找到。\n"
            "  仓库随附的只有 Windows 的 3rdparty/lib/mklml.lib；"
            "Linux 需要自行下载 mklml_lnx 并放到 3rdparty/lib。\n"
            "  不想装 MKL 就用默认的 -DBLAS_TYPE=zq_gemm。")
    endif()
    message(STATUS "MKL: ${ZQCNN_MKL_LIB}")
endif()

set(ZQCNN_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/ZQ_GEMM
                        ${CMAKE_CURRENT_LIST_DIR}/ZQCNN
                        ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include
                        CACHE INTERNAL "ZQCNN include dirs")
# ZQCNN_LIBRARY_DIR 已经在本文件第 122 行设过了（find_library 要用它）。

add_subdirectory(ZQ_GEMM)
add_subdirectory(ZQCNN)
add_subdirectory(SamplesZQCNN)
add_subdirectory(SamplesZQlibFaceID)
add_subdirectory(SamplesZQGEMM)
add_subdirectory(SamplesZQBLAS)

