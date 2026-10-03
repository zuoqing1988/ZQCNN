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
//
// 2026-10-03（附录 GN）：Linux 侧从 AVX 改成 **AVX2**，与 Windows 侧对齐。
// 两条论据都是实测的，指向同一个动作：
//
//  1. **原来的档位差异没有换来任何兼容性收益。**
//     根 CMakeLists.txt:113 给**所有** gcc x86 构建加 `-mavx2 -mfma`，
//     :118 给 MSVC 加 `/arch:AVX2` —— **都不看 ZQ_CNN_USE_SSETYPE**。
//     所以两个平台**本来就都要求 AVX2+FMA 的 CPU**，
//     把 Linux 设在 AVX 挡不住任何老机器，只影响"哪些内核被编进来"。
//     （副作用：SSETYPE 也决定 FMADD 开关，见下面那段 `>= AVX2`，
//       于是同一个模型在两个平台上算出**不同的浮点数** ——
//       实测 6/6 形状的位模式都不同，而 max|C| 到 6 位有效数字相同。
//       见附录 GN.2。）
//
//  2. **性能上是赚的。** 25 个形状、三轮交错、空载机器，
//     SSETYPE=3 相对 SSETYPE=2 的 GF/s：
//        中位数 122% / 平均 134% / 最好 265%（512x512x512）
//        比值 < 100% 的只有 4 个形状（3x3x3 90%、192x192x192 91%、
//        384x128x384 80%、512x512x1 81%）
//     即：不是全面更快，但中位数与均值都明显为正，
//     而那几个回退的形状在真实推理里权重很低。
//
// 想回到 AVX：cmake -DZQ_CNN_USE_SSETYPE=1/2，或直接改这一行
// （现在有 #ifndef 包裹，命令行 -D 优先，见上面 GL.1）。
#ifndef ZQ_CNN_USE_SSETYPE
#define ZQ_CNN_USE_SSETYPE ZQ_CNN_SSETYPE_AVX2
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

