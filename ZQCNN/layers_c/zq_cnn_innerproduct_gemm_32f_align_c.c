#include <stdio.h>
#include <malloc.h>
#include <string.h>
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
#include"math/zq_gemm_32f_align_c.h"

#if ZQ_CNN_USE_BLAS_GEMM
#include <openblas/cblas.h>
#elif ZQ_CNN_USE_MKL_GEMM
#include <mkl/mkl.h>
#endif
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

#if __ARM_NEON
#define zq_cnn_innerproduct_gemm_32f_align_same_pixstep zq_cnn_innerproduct_gemm_32f_align128bit_same_pixstep
#define zq_cnn_innerproduct_gemm_32f_align_same_pixstep_batch zq_cnn_innerproduct_gemm_32f_align128bit_same_pixstep_batch
#define zq_mm_load_ps vld1q_f32
#define zq_mm_store_ps vst1q_f32
#define zq_mm_add_ps vaddq_f32
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps(A, B, C) vfmaq_f32(C, A, B)
#else
#define zq_mm_fmadd_ps(A, B, C) vaddq_f32(vmulq_f32(A, B), C)
#endif
#define zq_mm_mul_ps vmulq_f32
#define zq_mm_setzero_ps() vdupq_n_f32(0)
#define zq_mm_type float32x4_t
#define zq_base_type float
#define zq_mm_align_size 4
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
#include "zq_cnn_innerproduct_gemm_32f_align_c_raw.h"
#undef zq_cblas_sgemm
#undef zq_CblasRowMajor
#undef zq_CblasNoTrans
#undef zq_CblasTrans

#undef zq_cnn_innerproduct_gemm_32f_align_same_pixstep
#undef zq_cnn_innerproduct_gemm_32f_align_same_pixstep_batch
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q

#if __ARM_NEON_FP16
#define zq_cnn_innerproduct_gemm_32f_align_same_pixstep zq_cnn_innerproduct_gemm_16f_align128bit_same_pixstep
#define zq_cnn_innerproduct_gemm_32f_align_same_pixstep_batch zq_cnn_innerproduct_gemm_16f_align128bit_same_pixstep_batch
#define zq_mm_load_ps vld1q_f16
#define zq_mm_store_ps vst1q_f16
#define zq_mm_add_ps vaddq_f16
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps(A, B, C) vfmaq_f16(C, A, B)
#else
#define zq_mm_fmadd_ps(A, B, C) vaddq_f16(vmulq_f32(A, B), C)
#endif
#define zq_mm_mul_ps vmulq_f16
#define zq_mm_setzero_ps() vdupq_n_f16(0)
#define zq_mm_type float16x8_t
#define zq_base_type float16_t
#define zq_mm_align_size 8
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFF0
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])

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
#include "zq_cnn_innerproduct_gemm_32f_align_c_raw.h"
#undef zq_cblas_sgemm
#undef zq_CblasRowMajor
#undef zq_CblasNoTrans
#undef zq_CblasTrans

#undef zq_cnn_innerproduct_gemm_32f_align_same_pixstep
#undef zq_cnn_innerproduct_gemm_32f_align_same_pixstep_batch
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q
#endif//__ARM_NEON_FP16

#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#define zq_cnn_innerproduct_gemm_32f_align_same_pixstep zq_cnn_innerproduct_gemm_32f_align128bit_same_pixstep
#define zq_cnn_innerproduct_gemm_32f_align_same_pixstep_batch zq_cnn_innerproduct_gemm_32f_align128bit_same_pixstep_batch
#define zq_mm_load_ps _mm_load_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_add_ps _mm_add_ps
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps _mm_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm_add_ps(_mm_mul_ps(A, B), C)
#endif
#define zq_mm_mul_ps _mm_mul_ps
#define zq_mm_setzero_ps _mm_setzero_ps
#define zq_mm_type __m128
#define zq_base_type float
#define zq_mm_align_size 4
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
#include "zq_cnn_innerproduct_gemm_32f_align_c_raw.h"
#undef zq_cblas_sgemm
#undef zq_CblasRowMajor
#undef zq_CblasNoTrans
#undef zq_CblasTrans

#undef zq_cnn_innerproduct_gemm_32f_align_same_pixstep
#undef zq_cnn_innerproduct_gemm_32f_align_same_pixstep_batch
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#define zq_cnn_innerproduct_gemm_32f_align_same_pixstep zq_cnn_innerproduct_gemm_32f_align256bit_same_pixstep
#define zq_cnn_innerproduct_gemm_32f_align_same_pixstep_batch zq_cnn_innerproduct_gemm_32f_align256bit_same_pixstep_batch
#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_add_ps _mm256_add_ps
#if ZQ_CNN_USE_FMADD256
#define zq_mm_fmadd_ps _mm256_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#endif
#define zq_mm_mul_ps _mm256_mul_ps
#define zq_mm_setzero_ps _mm256_setzero_ps
#define zq_mm_type __m256
#define zq_base_type float
#define zq_mm_align_size 8
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
#include "zq_cnn_innerproduct_gemm_32f_align_c_raw.h"
#undef zq_cblas_sgemm
#undef zq_CblasRowMajor
#undef zq_CblasNoTrans
#undef zq_CblasTrans



#undef zq_cnn_innerproduct_gemm_32f_align_same_pixstep
#undef zq_cnn_innerproduct_gemm_32f_align_same_pixstep_batch
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q

#endif

#endif //__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
