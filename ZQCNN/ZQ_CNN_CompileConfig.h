#ifndef _ZQ_CNN_COMPILE_CONFIG_H_
#define _ZQ_CNN_COMPILE_CONFIG_H_
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>



#define ZQ_CNN_SSETYPE_NONE 0
#define ZQ_CNN_SSETYPE_SSE 1
#define ZQ_CNN_SSETYPE_AVX 2
#define ZQ_CNN_SSETYPE_AVX2 3

#if defined(_WIN32)

#define ZQ_DECLSPEC_ALIGN32 __declspec(align(32))
#define ZQ_DECLSPEC_ALIGN16 __declspec(align(16))

// your settings
//
// **这五个宏一律用 #ifndef 包起来**（2026-10-03 改，附录 GL）。
// 原来它们是**无条件** #define，于是 CMake 的
//     add_definitions(-DZQ_CNN_USE_BLAS_GEMM=1)      // BLAS_TYPE=openblas
// 会被下面这行静默按回 0：gcc 报 "warning: ZQ_CNN_USE_BLAS_GEMM redefined"
// 并以头文件为准，MSVC 报 C4005 后同样以头文件为准。
// 后果是 `build-with-cmake.md:50` 教的 `-DBLAS_TYPE=openblas`
// **在任何平台都是空操作**（实测：宏值 0，链接行里没有 openblas）。
#ifndef ZQ_CNN_USE_SSETYPE
#define ZQ_CNN_USE_SSETYPE ZQ_CNN_SSETYPE_AVX2
#endif
#ifndef ZQ_CNN_USE_BLAS_GEMM
#define ZQ_CNN_USE_BLAS_GEMM 0 // if you want to use openblas, set to 1
#endif
#if ZQ_CNN_USE_BLAS_GEMM == 0
// 默认 0：ZQCNN 内部并不调用 cblas_*，开着它只会让示例程序链上 mklml.lib，
// 于是没装 MKL 运行库的机器上所有 exe 都起不来（error while loading shared libraries: mklml.dll）。
// 想用 MKL 就把它改成 1（此时需要自行提供 mklml.lib 与运行库），
// 或者 cmake -DZQ_CNN_USE_MKL_GEMM=1。
#ifndef ZQ_CNN_USE_MKL_GEMM
#define ZQ_CNN_USE_MKL_GEMM 0
#endif
#endif
#if (ZQ_CNN_USE_BLAS_GEMM == 0 && ZQ_CNN_USE_MKL_GEMM == 0)
#ifndef ZQ_CNN_USE_ZQ_GEMM
#define ZQ_CNN_USE_ZQ_GEMM 1
#endif
#endif


#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX2
#define ZQ_CNN_USE_FMADD128 1 
#define ZQ_CNN_USE_FMADD256 1 
#else
#define ZQ_CNN_USE_FMADD128 0
#define ZQ_CNN_USE_FMADD256 0 
#endif


/**   for linux system      **/
#else //#if !defined(_WIN32)

#define ZQ_DECLSPEC_ALIGN32 __attribute__((aligned(32)))
#define ZQ_DECLSPEC_ALIGN16 __attribute__((aligned(16)))

#if defined(ZQ_CNN_USE_ARM_NEON)
#define __ARM_NEON 1
#else
#define __ARM_NEON 0
#endif

#if defined(ZQ_CNN_USE_ARM_NEON_ARMV8)
#define __ARM_NEON_ARMV8 1
#else
#define __ARM_NEON_ARMV8 0
#endif

#if defined(ZQ_CNN_USE_ARM_NEON_FP16)
#define __ARM_NEON_FP16 1
#else
#define __ARM_NEON_FP16 0
#endif

#if __ARM_NEON
//#define ZQ_CNN_USE_FMADD128 1
#ifndef ZQ_CNN_USE_SSETYPE
#define ZQ_CNN_USE_SSETYPE ZQ_CNN_SSETYPE_NONE
#endif
#if defined(ZQ_CNN_USE_BOTH_BLAS_ZQ_GEMM) && ZQ_CNN_USE_BOTH_BLAS_ZQ_GEMM
#undef ZQ_CNN_USE_ZQ_GEMM
#undef ZQ_CNN_USE_BLAS_GEMM
#define ZQ_CNN_USE_ZQ_GEMM 1
#define ZQ_CNN_USE_BLAS_GEMM 1
#endif
#else
// your settings
#ifndef ZQ_CNN_USE_SSETYPE
#define ZQ_CNN_USE_SSETYPE ZQ_CNN_SSETYPE_AVX
#endif
#ifndef ZQ_CNN_USE_BLAS_GEMM
#define ZQ_CNN_USE_BLAS_GEMM 0 // if you want to use openblas, set to 1
#endif
#if ZQ_CNN_USE_BLAS_GEMM == 0
#ifndef ZQ_CNN_USE_MKL_GEMM
#define ZQ_CNN_USE_MKL_GEMM 0
#endif
#endif
#if (ZQ_CNN_USE_BLAS_GEMM == 0 && ZQ_CNN_USE_MKL_GEMM == 0)
#ifndef ZQ_CNN_USE_ZQ_GEMM
#define ZQ_CNN_USE_ZQ_GEMM 1
#endif
#endif


#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX2
#define ZQ_CNN_USE_FMADD128 1
#define ZQ_CNN_USE_FMADD256 1
#else
#define ZQ_CNN_USE_FMADD128 0
#define ZQ_CNN_USE_FMADD256 0
#endif

#endif //__ARM_NEON

#ifndef __int64 
#define __int64 long long
#endif

#ifndef __min
#define __min(a,b) ((a)<(b)?(a):(b))
#endif

#ifndef __max
#define __max(a,b) ((a)>(b)?(a):(b))
#endif

#ifndef _aligned_malloc
#define _aligned_malloc(x,y) memalign(y,x)
#endif

#ifndef _aligned_free
#define _aligned_free free
#endif

#ifndef fread_s
#define fread_s(a,b,c,d,e) fread(a,c,d,e)
#endif


#endif// defined(WIN32) || defined(_WINDOWS_)





// ---------------------------------------------------------------------------
// 兜底：**上面任何一个分支没定义到的开关，这里统一补成 0。**
//
// 为什么必须兜（2026-10-03 实测，附录 GL.2）：
//   * ARM 分支原来**只**在 `#if defined(ZQ_CNN_USE_BOTH_BLAS_ZQ_GEMM)` 里定义
//     三个后端开关，于是默认的 `-DZQ_CNN_USE_ARM_NEON` 构建（仓库根
//     `build.sh` 走的就是 armeabi-v7a）三个宏全部未定义；
//   * 非 ARM 分支里 `-DZQ_CNN_USE_BLAS_GEMM=1` 会让 MKL 那段整段跳过，
//     `-DZQ_CNN_USE_MKL_GEMM=1` 会让 ZQ_GEMM 那段整段跳过 —— 两者都留下
//     一个**未定义**的宏。
//
// `#if` 里"未定义"等价于 0，被**取值**时（printf、赋值、常量表达式）
// 直接编不过。统一成"定义成 0"之后，两种用法都对。
//
// 放在**整块平台分支之后**而不是散在各个分支里：散着写过一次，
// 结果 FMADD 那两条落进了 `#else` 分支里，ARM 路径反而没兜到。
// 一处兜底 = 一处能读懂。
#ifndef ZQ_CNN_USE_BLAS_GEMM
#define ZQ_CNN_USE_BLAS_GEMM 0
#endif
#ifndef ZQ_CNN_USE_MKL_GEMM
#define ZQ_CNN_USE_MKL_GEMM 0
#endif
#ifndef ZQ_CNN_USE_ZQ_GEMM
#define ZQ_CNN_USE_ZQ_GEMM 0
#endif
#ifndef ZQ_CNN_USE_SSETYPE
#define ZQ_CNN_USE_SSETYPE ZQ_CNN_SSETYPE_NONE
#endif
#ifndef ZQ_CNN_USE_FMADD128
#define ZQ_CNN_USE_FMADD128 0
#endif
#ifndef ZQ_CNN_USE_FMADD256
#define ZQ_CNN_USE_FMADD256 0
#endif


#endif// _ZQ_CNN_COMPILE_CONFIG_H_

