#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "../ZQ_CNN_CompileConfig.h"
#if __ARM_NEON
#include <arm_neon.h>
#else
#if defined(__GNUC__)
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#include <smmintrin.h>
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#include <x86intrin.h>
#endif
#elif defined(_WIN32)
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#include <mmintrin.h> //MMX  
#include <xmmintrin.h> //SSE(include mmintrin.h)  
#include <emmintrin.h> //SSE2(include xmmintrin.h)  
#include <pmmintrin.h> //SSE3(include emmintrin.h)  
#include <tmmintrin.h>//SSSE3(include pmmintrin.h)  
#include <smmintrin.h>//SSE4.1(include tmmintrin.h)  
#include <nmmintrin.h>//SSE4.2(include smmintrin.h)  
#endif 
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#include <wmmintrin.h>//AES(include nmmintrin.h)  
#include <immintrin.h>//AVX(include wmmintrin.h)  
#include <intrin.h>//(include immintrin.h)  
#endif
#endif
#endif //__ARM_NEON

#include "math/zq_gemm_32f_align_c.h"
#if ZQ_CNN_USE_BLAS_GEMM
#include <openblas/cblas.h>
#elif ZQ_CNN_USE_MKL_GEMM
#include <mkl/mkl.h>
#endif

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif


#if __ARM_NEON

#define zq_mm_load_ps vld1q_f32
#define zq_mm_loadu_ps vld1q_f32
#define zq_mm_store_ps vst1q_f32
#define zq_mm_add_ps vaddq_f32
#define zq_mm_min_ps vminq_f32
#define zq_mm_max_ps vmaxq_f32
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps(A, B, C) vfmaq_f32(C, A, B)
#else
#define zq_mm_fmadd_ps(A, B, C) vaddq_f32(vmulq_f32(A, B), C)
#endif
#define zq_mm_mul_ps vmulq_f32
#define zq_mm_setzero_ps() vdupq_n_f32(0)
#define zq_mm_set1_ps vdupq_n_f32
#define zq_mm_type float32x4_t
#define zq_base_type float
#define zq_mm_align_size 4
#define zq_mm_align_size2 8
#define zq_mm_align_size3 12
#define zq_mm_align_size4 16
#define zq_mm_align_size5 20
#define zq_mm_align_size6 24
#define zq_mm_align_size7 28
#define zq_mm_align_size8 32
#define zq_mm_align_size16 64
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFF0
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3])

#if ZQ_CNN_USE_BLAS_GEMM
#define	zq_cblas_sgemm cblas_sgemm
#define zq_CblasRowMajor CblasRowMajor
#define zq_CblasNoTrans CblasNoTrans
#define zq_CblasTrans CblasTrans
#else
#define zq_CblasRowMajor 1
#define zq_CblasNoTrans 1
#define zq_CblasTrans 1
#define	zq_cblas_sgemm(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14)  \
      zq_gemm_32f_AnoTrans_Btrans_auto(x4,x5,x6,x8,x9,x10,x11,x13,x14)   
#endif

#define zq_cnn_innerproduct_gemm_nchwc_prepack4 zq_cnn_innerproduct_gemm_nchwc4_prepack4
#if __ARM_NEON && __ARM_NEON_ARMV8
#define zq_cnn_innerproduct_gemm_nchwc_prepack8_other zq_cnn_innerproduct_gemm_nchwc4_prepack8_other
#endif
#include "zq_cnn_innerproduct_gemm_nchwc_prepack4.h"
#undef zq_cnn_innerproduct_gemm_nchwc_prepack4
#if __ARM_NEON && _ARM_NEON_ARMV8
#undef zq_cnn_innerproduct_gemm_nchwc_prepack8_other
#endif


#if __ARM_NEON && __ARM_NEON_ARMV8
#define zq_cnn_innerproduct_gemm_nchwc_packed8_other zq_cnn_innerproduct_gemm_nchwc4_packed8_other
#endif
#define zq_cnn_innerproduct_gemm_nchwc_packed4 zq_cnn_innerproduct_gemm_nchwc4_packed4
#define zq_cnn_innerproduct_gemm_nchwc_general zq_cnn_innerproduct_gemm_nchwc4_general
#define zq_cnn_innerproduct_nchwc_noborder zq_cnn_innerproduct_nchwc4_noborder

#define WITH_BIAS 0
#define WITH_PRELU 0
#include "zq_cnn_innerproduct_gemm_nchwc_raw.h"
#include "zq_cnn_innerproduct_gemm_nchwc_packed4.h"
#undef WITH_BIAS
#undef WITH_PRELU

#if __ARM_NEON && __ARM_NEON_ARMV8
#undef	zq_cnn_innerproduct_gemm_nchwc_packed8_other
#endif
#undef zq_cnn_innerproduct_gemm_nchwc_packed4
#undef zq_cnn_innerproduct_gemm_nchwc_general
#undef zq_cnn_innerproduct_nchwc_noborder

#if __ARM_NEON && __ARM_NEON_ARMV8
#define zq_cnn_innerproduct_gemm_nchwc_packed8_other zq_cnn_innerproduct_gemm_nchwc4_packed8_other_with_bias
#endif
#define zq_cnn_innerproduct_gemm_nchwc_packed4 zq_cnn_innerproduct_gemm_nchwc4_packed4_with_bias
#define zq_cnn_innerproduct_gemm_nchwc_general zq_cnn_innerproduct_gemm_nchwc4_general_with_bias
#define zq_cnn_innerproduct_nchwc_noborder zq_cnn_innerproduct_nchwc4_noborder_with_bias
#define WITH_BIAS 1
#define WITH_PRELU 0
#include "zq_cnn_innerproduct_gemm_nchwc_raw.h"
#include "zq_cnn_innerproduct_gemm_nchwc_packed4.h"
#undef WITH_BIAS
#undef WITH_PRELU
#if __ARM_NEON && __ARM_NEON_ARMV8
#undef	zq_cnn_innerproduct_gemm_nchwc_packed8_other
#endif
#undef zq_cnn_innerproduct_gemm_nchwc_packed4
#undef zq_cnn_innerproduct_gemm_nchwc_general
#undef zq_cnn_innerproduct_nchwc_noborder

#if __ARM_NEON && __ARM_NEON_ARMV8
#define zq_cnn_innerproduct_gemm_nchwc_packed8_other zq_cnn_innerproduct_gemm_nchwc4_packed8_other_with_bias_prelu
#endif
#define zq_cnn_innerproduct_gemm_nchwc_packed4 zq_cnn_innerproduct_gemm_nchwc4_packed4_with_bias_prelu
#define zq_cnn_innerproduct_gemm_nchwc_general zq_cnn_innerproduct_gemm_nchwc4_general_with_bias_prelu
#define zq_cnn_innerproduct_nchwc_noborder zq_cnn_innerproduct_nchwc4_noborder_with_bias_prelu
#define WITH_BIAS 1
#define WITH_PRELU 1
#include "zq_cnn_innerproduct_gemm_nchwc_raw.h"
#include "zq_cnn_innerproduct_gemm_nchwc_packed4.h"
#undef WITH_BIAS
#undef WITH_PRELU
#if __ARM_NEON && __ARM_NEON_ARMV8
#undef	zq_cnn_innerproduct_gemm_nchwc_packed8_other
#endif
#undef zq_cnn_innerproduct_gemm_nchwc_packed4
#undef zq_cnn_innerproduct_gemm_nchwc_general
#undef zq_cnn_innerproduct_nchwc_noborder

#undef zq_cblas_sgemm
#undef zq_CblasRowMajor
#undef zq_CblasNoTrans
#undef zq_CblasTrans
#undef zq_mm_load_ps
#undef zq_mm_loadu_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_min_ps
#undef zq_mm_max_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
#undef zq_mm_set1_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#undef zq_mm_align_size2
#undef zq_mm_align_size3
#undef zq_mm_align_size4
#undef zq_mm_align_size5
#undef zq_mm_align_size6
#undef zq_mm_align_size7
#undef zq_mm_align_size8
#undef zq_mm_align_size16
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q

#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE

#define zq_mm_load_ps _mm_load_ps
#define zq_mm_loadu_ps _mm_loadu_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_add_ps _mm_add_ps
#define zq_mm_min_ps _mm_min_ps
#define zq_mm_max_ps _mm_max_ps
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps _mm_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm_add_ps(_mm_mul_ps(A, B), C)
#endif
#define zq_mm_mul_ps _mm_mul_ps
#define zq_mm_setzero_ps _mm_setzero_ps
#define zq_mm_set1_ps _mm_set1_ps
#define zq_mm_type __m128
#define zq_base_type float
#define zq_mm_align_size 4
#define zq_mm_align_size2 8
#define zq_mm_align_size3 12
#define zq_mm_align_size4 16
#define zq_mm_align_size5 20
#define zq_mm_align_size6 24
#define zq_mm_align_size7 28
#define zq_mm_align_size8 32
#define zq_mm_align_size16 64
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFF0
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3])

#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM)
#define	zq_cblas_sgemm cblas_sgemm
#define zq_CblasRowMajor CblasRowMajor
#define zq_CblasNoTrans CblasNoTrans
#define zq_CblasTrans CblasTrans
#else
#define zq_CblasRowMajor 1
#define zq_CblasNoTrans 1
#define zq_CblasTrans 1
#define	zq_cblas_sgemm(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14)  \
      zq_gemm_32f_AnoTrans_Btrans_auto(x4,x5,x6,x8,x9,x10,x11,x13,x14)   
#endif

#define zq_cnn_innerproduct_gemm_nchwc_prepack4 zq_cnn_innerproduct_gemm_nchwc4_prepack4
#include "zq_cnn_innerproduct_gemm_nchwc_prepack4.h"
#undef zq_cnn_innerproduct_gemm_nchwc_prepack4

#define zq_cnn_innerproduct_gemm_nchwc_packed4 zq_cnn_innerproduct_gemm_nchwc4_packed4
#define zq_cnn_innerproduct_gemm_nchwc_general zq_cnn_innerproduct_gemm_nchwc4_general
#define zq_cnn_innerproduct_nchwc_noborder zq_cnn_innerproduct_nchwc4_noborder

#define WITH_BIAS 0
#define WITH_PRELU 0
#include "zq_cnn_innerproduct_gemm_nchwc_raw.h"
#include "zq_cnn_innerproduct_gemm_nchwc_packed4.h"
#undef WITH_BIAS
#undef WITH_PRELU

#undef zq_cnn_innerproduct_gemm_nchwc_packed4
#undef zq_cnn_innerproduct_gemm_nchwc_general
#undef zq_cnn_innerproduct_nchwc_noborder

#define zq_cnn_innerproduct_gemm_nchwc_packed4 zq_cnn_innerproduct_gemm_nchwc4_packed4_with_bias
#define zq_cnn_innerproduct_gemm_nchwc_general zq_cnn_innerproduct_gemm_nchwc4_general_with_bias
#define zq_cnn_innerproduct_nchwc_noborder zq_cnn_innerproduct_nchwc4_noborder_with_bias
#define WITH_BIAS 1
#define WITH_PRELU 0
#include "zq_cnn_innerproduct_gemm_nchwc_raw.h"
#include "zq_cnn_innerproduct_gemm_nchwc_packed4.h"
#undef WITH_BIAS
#undef WITH_PRELU
#undef zq_cnn_innerproduct_gemm_nchwc_packed4
#undef zq_cnn_innerproduct_gemm_nchwc_general
#undef zq_cnn_innerproduct_nchwc_noborder

#define zq_cnn_innerproduct_gemm_nchwc_packed4 zq_cnn_innerproduct_gemm_nchwc4_packed4_with_bias_prelu
#define zq_cnn_innerproduct_gemm_nchwc_general zq_cnn_innerproduct_gemm_nchwc4_general_with_bias_prelu
#define zq_cnn_innerproduct_nchwc_noborder zq_cnn_innerproduct_nchwc4_noborder_with_bias_prelu
#define WITH_BIAS 1
#define WITH_PRELU 1
#include "zq_cnn_innerproduct_gemm_nchwc_raw.h"
#include "zq_cnn_innerproduct_gemm_nchwc_packed4.h"
#undef WITH_BIAS
#undef WITH_PRELU
#undef zq_cnn_innerproduct_gemm_nchwc_packed4
#undef zq_cnn_innerproduct_gemm_nchwc_general
#undef zq_cnn_innerproduct_nchwc_noborder

#undef zq_mm_load_ps
#undef zq_mm_loadu_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_min_ps
#undef zq_mm_max_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
#undef zq_mm_set1_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#undef zq_mm_align_size2
#undef zq_mm_align_size3
#undef zq_mm_align_size4
#undef zq_mm_align_size5
#undef zq_mm_align_size6
#undef zq_mm_align_size7
#undef zq_mm_align_size8
#undef zq_mm_align_size16
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q

#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX

#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_loadu_ps _mm256_loadu_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_add_ps _mm256_add_ps
#define zq_mm_min_ps _mm256_min_ps
#define zq_mm_max_ps _mm256_max_ps
#if ZQ_CNN_USE_FMADD256
#define zq_mm_fmadd_ps _mm256_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#endif
#define zq_mm_mul_ps _mm256_mul_ps
#define zq_mm_setzero_ps _mm256_setzero_ps
#define zq_mm_set1_ps _mm256_set1_ps
#define zq_mm_type __m256
#define zq_base_type float
#define zq_mm_align_size 8
#define zq_mm_align_size2 16
#define zq_mm_align_size3 24
#define zq_mm_align_size4 32
#define zq_mm_align_size5 40
#define zq_mm_align_size6 48
#define zq_mm_align_size7 56
#define zq_mm_align_size8 64
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFE0
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])
#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM)
#define	zq_cblas_sgemm cblas_sgemm
#define zq_CblasRowMajor CblasRowMajor
#define zq_CblasNoTrans CblasNoTrans
#define zq_CblasTrans CblasTrans
#else
#define zq_CblasRowMajor 1
#define zq_CblasNoTrans 1
#define zq_CblasTrans 1
#define	zq_cblas_sgemm(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14)  \
      zq_gemm_32f_AnoTrans_Btrans_auto(x4,x5,x6,x8,x9,x10,x11,x13,x14)  
#endif


#define zq_cnn_innerproduct_gemm_nchwc_general zq_cnn_innerproduct_gemm_nchwc8_general
#define zq_cnn_innerproduct_nchwc_noborder zq_cnn_innerproduct_nchwc8_noborder
#define WITH_BIAS 0
#define WITH_PRELU 0
#include "zq_cnn_innerproduct_gemm_nchwc_raw.h"
#undef WITH_BIAS
#undef WITH_PRELU

#undef zq_cnn_innerproduct_gemm_nchwc_general
#undef zq_cnn_innerproduct_nchwc_noborder

#define zq_cnn_innerproduct_gemm_nchwc_general zq_cnn_innerproduct_gemm_nchwc8_general_with_bias
#define zq_cnn_innerproduct_nchwc_noborder zq_cnn_innerproduct_nchwc8_noborder_with_bias
#define WITH_BIAS 1
#define WITH_PRELU 0
#include "zq_cnn_innerproduct_gemm_nchwc_raw.h"
#undef WITH_BIAS
#undef WITH_PRELU
#undef zq_cnn_innerproduct_gemm_nchwc_general
#undef zq_cnn_innerproduct_nchwc_noborder

#define zq_cnn_innerproduct_gemm_nchwc_general zq_cnn_innerproduct_gemm_nchwc8_general_with_bias_prelu
#define zq_cnn_innerproduct_nchwc_noborder zq_cnn_innerproduct_nchwc8_noborder_with_bias_prelu
#define WITH_BIAS 1
#define WITH_PRELU 1
#include "zq_cnn_innerproduct_gemm_nchwc_raw.h"
#undef WITH_BIAS
#undef WITH_PRELU
#undef zq_cnn_innerproduct_gemm_nchwc_general
#undef zq_cnn_innerproduct_nchwc_noborder

#undef zq_cblas_sgemm
#undef zq_CblasRowMajor
#undef zq_CblasNoTrans
#undef zq_CblasTrans
#undef zq_mm_load_ps
#undef zq_mm_loadu_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_min_ps
#undef zq_mm_max_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
#undef zq_mm_set1_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#undef zq_mm_align_size2
#undef zq_mm_align_size3
#undef zq_mm_align_size4
#undef zq_mm_align_size5
#undef zq_mm_align_size6
#undef zq_mm_align_size7
#undef zq_mm_align_size8
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q
#endif
#endif //__ARM_NEON


static inline float my_mm_load_ps(const float* ptr) { return *ptr; }
static inline void my_mm_store_ps(float* ptr, float val) { *ptr = val; }
static inline float my_mm_add_ps(float a, float b) { return a + b; }
static inline float my_mm_sub_ps(float a, float b) { return a - b; }
static inline float my_mm_mul_ps(float a, float b) { return a * b; }
static inline float my_mm_fmadd_ps(float a, float b, float c) { return a*b + c; }
static inline float my_mm_max_ps(float a, float b) { return a > b ? a : b; }
static inline float my_mm_min_ps(float a, float b) { return a < b ? a : b; }
static inline float my_mm_setzero_ps() { return 0; }
static inline float my_mm_set1_ps(float v) { return v; }

#define zq_mm_load_ps my_mm_load_ps
#define zq_mm_loadu_ps my_mm_load_ps
#define zq_mm_store_ps my_mm_store_ps
#define zq_mm_add_ps my_mm_add_ps
#define zq_mm_min_ps my_mm_min_ps
#define zq_mm_max_ps my_mm_max_ps
#define zq_mm_fmadd_ps my_mm_fmadd_ps
#define zq_mm_mul_ps my_mm_mul_ps
#define zq_mm_setzero_ps my_mm_setzero_ps
#define zq_mm_set1_ps my_mm_set1_ps
#define zq_mm_type float
#define zq_base_type float
#define zq_mm_align_size 1
#define zq_mm_align_size2 2
#define zq_mm_align_size3 3
#define zq_mm_align_size4 4
#define zq_mm_align_size5 5
#define zq_mm_align_size6 6
#define zq_mm_align_size7 7
#define zq_mm_align_size8 8
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFF0
#define zq_final_sum_q q[0]

#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM)
#define	zq_cblas_sgemm cblas_sgemm
#define zq_CblasRowMajor CblasRowMajor
#define zq_CblasNoTrans CblasNoTrans
#define zq_CblasTrans CblasTrans
#else
#define zq_CblasRowMajor 1
#define zq_CblasNoTrans 1
#define zq_CblasTrans 1
#define	zq_cblas_sgemm(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14)  \
      zq_gemm_32f_AnoTrans_Btrans_auto(x4,x5,x6,x8,x9,x10,x11,x13,x14)   
#endif

#define zq_cnn_innerproduct_gemm_nchwc_general zq_cnn_innerproduct_gemm_nchwc1_general
#define zq_cnn_innerproduct_nchwc_noborder zq_cnn_innerproduct_nchwc1_noborder

#define WITH_BIAS 0
#define WITH_PRELU 0
#include "zq_cnn_innerproduct_gemm_nchwc_raw.h"
#undef WITH_BIAS
#undef WITH_PRELU

#undef zq_cnn_innerproduct_gemm_nchwc_general
#undef zq_cnn_innerproduct_nchwc_noborder

#define zq_cnn_innerproduct_gemm_nchwc_general zq_cnn_innerproduct_gemm_nchwc1_general_with_bias
#define zq_cnn_innerproduct_nchwc_noborder zq_cnn_innerproduct_nchwc1_noborder_with_bias
#define WITH_BIAS 1
#define WITH_PRELU 0
#include "zq_cnn_innerproduct_gemm_nchwc_raw.h"
#undef WITH_BIAS
#undef WITH_PRELU
#undef zq_cnn_innerproduct_gemm_nchwc_general
#undef zq_cnn_innerproduct_nchwc_noborder

#define zq_cnn_innerproduct_gemm_nchwc_general zq_cnn_innerproduct_gemm_nchwc1_general_with_bias_prelu
#define zq_cnn_innerproduct_nchwc_noborder zq_cnn_innerproduct_nchwc1_noborder_with_bias_prelu
#define WITH_BIAS 1
#define WITH_PRELU 1
#include "zq_cnn_innerproduct_gemm_nchwc_raw.h"
#undef WITH_BIAS
#undef WITH_PRELU
#undef zq_cnn_innerproduct_gemm_nchwc_general
#undef zq_cnn_innerproduct_nchwc_noborder


#undef zq_cblas_sgemm
#undef zq_CblasRowMajor
#undef zq_CblasNoTrans
#undef zq_CblasTrans
#undef zq_mm_load_ps
#undef zq_mm_loadu_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_min_ps
#undef zq_mm_max_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
#undef zq_mm_set1_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#undef zq_mm_align_size2
#undef zq_mm_align_size3
#undef zq_mm_align_size4
#undef zq_mm_align_size5
#undef zq_mm_align_size6
#undef zq_mm_align_size7
#undef zq_mm_align_size8
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
