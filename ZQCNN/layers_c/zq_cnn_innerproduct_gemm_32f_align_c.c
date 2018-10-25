#include <mmintrin.h> //MMX  
#include <xmmintrin.h> //SSE(include mmintrin.h)  
#include <emmintrin.h> //SSE2(include xmmintrin.h)  
#include <pmmintrin.h> //SSE3(include emmintrin.h)  
#include <tmmintrin.h>//SSSE3(include pmmintrin.h)  
#include <smmintrin.h>//SSE4.1(include tmmintrin.h)  
#include <nmmintrin.h>//SSE4.2(include smmintrin.h)  
#include <wmmintrin.h>//AES(include nmmintrin.h)  
#include <immintrin.h>//AVX(include wmmintrin.h)  
#include <intrin.h>//(include immintrin.h)  
#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <omp.h>
#include "..\ZQ_CNN_CompileConfig.h"
#if ZQ_CNN_USE_BLAS_GEMM
#include <cblas.h>
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif


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
#define zq_mm_align_size 4
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFF0
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3])

#include "zq_cnn_innerproduct_gemm_32f_align_c_raw.h"


#undef zq_cnn_innerproduct_gemm_32f_align_same_pixstep
#undef zq_cnn_innerproduct_gemm_32f_align_same_pixstep_batch
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
#undef zq_mm_type
#undef zq_mm_align_size
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q



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
#define zq_mm_align_size 8
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFE0
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])

#include "zq_cnn_innerproduct_gemm_32f_align_c_raw.h"


#undef zq_cnn_innerproduct_gemm_32f_align_same_pixstep
#undef zq_cnn_innerproduct_gemm_32f_align_same_pixstep_batch
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
#undef zq_mm_type
#undef zq_mm_align_size
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q


#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif

#endif