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

// your settings
#define ZQ_CNN_USE_SSETYPE ZQ_CNN_SSETYPE_NONE
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
