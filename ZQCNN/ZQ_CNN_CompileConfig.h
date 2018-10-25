#ifndef _ZQ_CNN_COMPILE_CONFIG_H_
#define _ZQ_CNN_COMPILE_CONFIG_H_

#define ZQ_CNN_SSETYPE_NONE 0
#define ZQ_CNN_SSETYPE_SSE 1
#define ZQ_CNN_SSETYPE_AVX 2
#define ZQ_CNN_SSETYPE_AVX2 3


// your settings
#define ZQ_CNN_USE_SSETYPE ZQ_CNN_SSETYPE_AVX2
#define ZQ_CNN_USE_BLAS_GEMM 1 // if you want to use openblas, set to 1





#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX2
#define ZQ_CNN_USE_FMADD128 1 
#define ZQ_CNN_USE_FMADD256 1 
#else
#define ZQ_CNN_USE_FMADD128 0
#define ZQ_CNN_USE_FMADD256 0 
#endif

#endif
