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
#define ZQ_CNN_USE_SSETYPE ZQ_CNN_SSETYPE_AVX2
#define ZQ_CNN_USE_BLAS_GEMM 0 // if you want to use openblas, set to 1
#if ZQ_CNN_USE_BLAS_GEMM == 0
#define ZQ_CNN_USE_MKL_GEMM 1
#endif
#if (ZQ_CNN_USE_BLAS_GEMM == 0 && ZQ_CNN_USE_MKL_GEMM == 0)
#define ZQ_CNN_USE_ZQ_GEMM 1
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
#define ZQ_CNN_USE_SSETYPE ZQ_CNN_SSETYPE_NONE
#if defined(ZQ_CNN_USE_BOTH_BLAS_ZQ_GEMM)
#define ZQ_CNN_USE_ZQ_GEMM 1
#define ZQ_CNN_USE_BLAS_GEMM 1
#endif
#else
// your settings
#define ZQ_CNN_USE_SSETYPE ZQ_CNN_SSETYPE_AVX
#define ZQ_CNN_USE_BLAS_GEMM 0 // if you want to use openblas, set to 1
#if ZQ_CNN_USE_BLAS_GEMM == 0
#define ZQ_CNN_USE_MKL_GEMM 0
#endif
#if (ZQ_CNN_USE_BLAS_GEMM == 0 && ZQ_CNN_USE_MKL_GEMM == 0)
#define ZQ_CNN_USE_ZQ_GEMM 1
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


#endif// _ZQ_CNN_COMPILE_CONFIG_H_
