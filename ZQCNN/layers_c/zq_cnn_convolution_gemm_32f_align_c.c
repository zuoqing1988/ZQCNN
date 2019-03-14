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
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_kernel1x1 zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_kernel1x1
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_C4 zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_C4
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_batch zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_batch
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_C3 zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_C3
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_batch zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_batch
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
#include "zq_cnn_convolution_gemm_32f_align_c_raw.h"
#undef zq_cblas_sgemm
#undef zq_CblasRowMajor
#undef zq_CblasNoTrans
#undef zq_CblasTrans

#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_kernel1x1
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_C4
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_batch
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_C3
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_batch
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

#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep zq_cnn_conv_no_padding_gemm_16f_align128bit_same_pixstep
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_kernel1x1 zq_cnn_conv_no_padding_gemm_16f_align128bit_same_pixstep_kernel1x1
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_C4 zq_cnn_conv_no_padding_gemm_16f_align128bit_same_pixstep_C4
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_batch zq_cnn_conv_no_padding_gemm_16f_align128bit_same_pixstep_batch
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep zq_cnn_conv_no_padding_gemm_16f_align128bit_same_or_notsame_pixstep
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_C3 zq_cnn_conv_no_padding_gemm_16f_align128bit_same_or_notsame_pixstep_C3
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_batch zq_cnn_conv_no_padding_gemm_16f_align128bit_same_or_notsame_pixstep_batch
#define zq_mm_load_ps vld1q_f16
#define zq_mm_store_ps vst1q_f16
#define zq_mm_add_ps vaddq_f16
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps(A, B, C) vfmaq_f16(C, A, B)
#else
#define zq_mm_fmadd_ps(A, B, C) vaddq_f16(vmulq_f16(A, B), C)
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
#include "zq_cnn_convolution_gemm_32f_align_c_raw.h"
#undef zq_cblas_sgemm
#undef zq_CblasRowMajor
#undef zq_CblasNoTrans
#undef zq_CblasTrans

#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_kernel1x1
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_C4
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_batch
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_C3
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_batch
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

#endif //__ARM_NEON_FP16

#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_kernel1x1 zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_kernel1x1
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_C4 zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_C4
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_batch zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_batch
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_C3 zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_C3
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_batch zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_batch
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
#include "zq_cnn_convolution_gemm_32f_align_c_raw.h"
#undef zq_cblas_sgemm
#undef zq_CblasRowMajor
#undef zq_CblasNoTrans
#undef zq_CblasTrans

#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_kernel1x1
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_C4
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_batch
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_C3
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_batch
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

#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_kernel1x1 zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep_kernel1x1
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_C4 zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep_C4
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_batch zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep_batch
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_C3 zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep_C3
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_batch zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep_batch
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
#include "zq_cnn_convolution_gemm_32f_align_c_raw.h"
#undef zq_cblas_sgemm
#undef zq_CblasRowMajor
#undef zq_CblasNoTrans
#undef zq_CblasTrans

#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_kernel1x1
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_C4
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_batch
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_C3
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_batch
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
      zq_gemm_32f_align0_AnoTrans_Btrans(x4,x5,x6,x8,x9,x10,x11,x13,x14)  
#endif
	/*in_pixStep can be different with filter_pixStep,
	and the aligned channels should be set to zero*/
	void zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,
		int filter_H,
		int filter_W,
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		int dilation_H,
		int dilation_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		void** buffer,
		__int64 *buffer_len
	)
	{
		/************** image to col **************/
		int in_widthStep_mul_stride_H = in_widthStep*stride_H;
		int in_pixelStep_mul_stride_W = in_pixelStep*stride_W;
		int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
		int filter_pixStep_mul_filter_W = filter_pixelStep*filter_W;
		int matrix_A_cols = filter_H*filter_W*in_C;
		int matrix_A_rows = out_H*out_W;
		int matrix_B_cols = filter_N;
		int matrix_B_rows = filter_H*filter_W*filter_C;
		__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(float) + 31) / 32 * 32;
		__int64 need_B_buffer_len_align32 = (matrix_B_rows*matrix_B_cols * sizeof(float) + 31) / 32 * 32;
		__int64 need_C_buffer_len_align32 = 0;
		__int64 total_need_buffer_len;
		float* matrix_A = 0;
		float* matrix_Bt = 0;
		const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *cur_in_row_ptr,*cur_in_pix_ptr,*filter_slice_ptr,*filter_row_ptr,*filter_pix_ptr;
		int out_n, out_h, out_w, kn, kh, kw;
		float* matrix_A_row_ptr, *matrix_A_col_ptr, *cp_dst_ptr;
		float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		float* matrix_C = 0, *matrix_C_row_ptr;
		int out_row_idx;
		int out_HW = out_H*out_W;
		double t1, t2, t3, t4, t5;
		int need_allocate_tmp_out;
		t1 = omp_get_wtime();
		need_allocate_tmp_out = (out_pixelStep != filter_N) || (out_pixelStep*out_W != out_widthStep) || (out_widthStep*out_H != out_sliceStep);
		if (need_allocate_tmp_out)
		{
			need_C_buffer_len_align32 = (out_HW*filter_N * sizeof(float) + 31) / 32 * 32;
		}
		else
		{
			matrix_C = out_tensor4D_data;
		}
		
		total_need_buffer_len = need_A_buffer_len_align32 + need_B_buffer_len_align32 + need_C_buffer_len_align32;
		if (buffer == 0)
		{
			matrix_A = (float*)_aligned_malloc(need_A_buffer_len_align32, 32);
			matrix_Bt = (float*)_aligned_malloc(need_B_buffer_len_align32, 32);
			if (need_allocate_tmp_out)
				matrix_C = (float*)_aligned_malloc(need_C_buffer_len_align32, 32);
		}
		else
		{
			if (*buffer_len < total_need_buffer_len)
			{
				_aligned_free(*buffer);
				*buffer = _aligned_malloc(total_need_buffer_len, 32);
				*buffer_len = total_need_buffer_len;
			}
			matrix_A = *buffer;
			matrix_Bt = (float*)((char*)(*buffer) + need_A_buffer_len_align32);
			if (need_allocate_tmp_out)
				matrix_C = (float*)((char*)(*buffer) + need_A_buffer_len_align32 + need_B_buffer_len_align32);
		}
		

		cp_dst_ptr = matrix_Bt;
		for (kn = 0,filter_slice_ptr = filters_data; kn < filter_N; kn++,filter_slice_ptr+=filter_sliceStep)
		{
			for (kh = 0, filter_row_ptr = filter_slice_ptr; kh < filter_H; kh++, filter_row_ptr += filter_widthStep)
			{
				for (kw = 0, filter_pix_ptr = filter_row_ptr; kw < filter_W; kw++, filter_pix_ptr += filter_pixelStep)
				{
					memcpy(cp_dst_ptr, filter_pix_ptr, sizeof(float)*in_C);
					cp_dst_ptr += in_C;
				}
			}
		}

		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			t2 = omp_get_wtime();
			matrix_A_row_ptr = matrix_A;
			for (out_h = 0, in_row_ptr = in_slice_ptr; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
				{
					matrix_A_col_ptr = matrix_A_row_ptr;
					for (kh = 0, cur_in_row_ptr = in_pix_ptr;kh < filter_H;	kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep)
					{
						for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < filter_W; kw++, cur_in_pix_ptr += dilate_W_mul_in_pixStep)
						{
							memcpy(matrix_A_col_ptr, cur_in_pix_ptr, sizeof(float)*in_C);
							matrix_A_col_ptr += in_C;
						}
					}
					matrix_A_row_ptr += matrix_A_cols;
				}
			}
			t3 = omp_get_wtime();
			/*gemm*/
			zq_cblas_sgemm(zq_CblasRowMajor, zq_CblasNoTrans, zq_CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
				matrix_Bt, matrix_A_cols, 0.0f, matrix_C, matrix_B_cols);
			t4 = omp_get_wtime();
			if (need_allocate_tmp_out)
			{
				/*   col2im      */
				out_row_idx = 0;
				matrix_C_row_ptr = matrix_C;
				for (out_h = 0, out_row_ptr = out_slice_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
				{
					for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W; out_w++, out_pix_ptr += out_pixelStep)
					{
						memcpy(out_pix_ptr, matrix_C_row_ptr, sizeof(float)*matrix_B_cols);
						matrix_C_row_ptr += matrix_B_cols;
					}
				}
			}
			else
				matrix_C += out_sliceStep;
			t5 = omp_get_wtime();
		}
		if (buffer == 0)
		{
			_aligned_free(matrix_A);
			_aligned_free(matrix_Bt);
			if (need_allocate_tmp_out)
				_aligned_free(matrix_C);
		}
		/*if (filter_H == 3 && filter_W == 3 && filter_C == 3)
		{
			printf("total: %.3f ms, alloc %.3f ms, makeA: %.3f ms, gemm: %.3f ms, copy_C: %.3f ms\n", 1000 * (t5 - t1), 1000 * (t2 - t1),
				1000 * (t3 - t2), 1000 * (t4 - t3), 1000 * (t5 - t4));
		}*/
	}

	/*in_pixStep can be different with filter_pixStep,
	and the aligned channels should be set to zero*/
	void zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep_batch(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,
		int filter_H,
		int filter_W,
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		int dilation_H,
		int dilation_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		void** buffer,
		__int64 *buffer_len
	)
	{
		/************** image to col **************/
		int in_widthStep_mul_stride_H = in_widthStep*stride_H;
		int in_pixelStep_mul_stride_W = in_pixelStep*stride_W;
		int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
		int filter_pixStep_mul_filter_W = filter_pixelStep*filter_W;
		int matrix_A_cols = in_C;
		int matrix_A_rows = out_N*out_H*out_W;
		int matrix_B_cols = filter_N;
		int matrix_B_rows = filter_H*filter_W*filter_C;
		__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(float) + 31) / 32 * 32;
		__int64 need_B_buffer_len_align32 = (matrix_B_rows*matrix_B_cols * sizeof(float) + 31) / 32 * 32;
		__int64 need_C_buffer_len_align32 = 0;
		__int64 total_need_buffer_len;
		float* matrix_A = 0;
		float* matrix_Bt = 0;
		const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *cur_in_row_ptr, *cur_in_pix_ptr, *filter_slice_ptr, *filter_row_ptr, *filter_pix_ptr;
		int out_n, out_h, out_w, kn, kh, kw;
		float* matrix_A_row_ptr, *matrix_A_col_ptr, *cp_dst_ptr;
		float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		float* matrix_C = 0, *matrix_C_row_ptr;
		int out_row_idx;
		int out_NHW = out_N*out_H*out_W;
		double t1, t2, t3, t4, t5;
		t1 = omp_get_wtime();
		int need_allocate_tmp_out = (out_pixelStep != filter_N) || (out_pixelStep*out_W != out_widthStep) || (out_widthStep*out_H != out_sliceStep);
		if (need_allocate_tmp_out)
		{
			need_C_buffer_len_align32 = (out_NHW*filter_N * sizeof(float) + 31) / 32 * 32;
		}
		else
		{
			matrix_C = out_tensor4D_data;
		}

		total_need_buffer_len = need_A_buffer_len_align32 + need_B_buffer_len_align32 + need_C_buffer_len_align32;
		if (buffer == 0)
		{
			matrix_A = (float*)_aligned_malloc(need_A_buffer_len_align32, 32);
			matrix_Bt = (float*)_aligned_malloc(need_B_buffer_len_align32, 32);
			if (need_allocate_tmp_out)
				matrix_C = (float*)_aligned_malloc(need_C_buffer_len_align32, 32);
		}
		else
		{
			if (*buffer_len < total_need_buffer_len)
			{
				_aligned_free(*buffer);
				*buffer = _aligned_malloc(total_need_buffer_len, 32);
				*buffer_len = total_need_buffer_len;
			}
			matrix_A = *buffer;
			matrix_Bt = (float*)((char*)(*buffer) + need_A_buffer_len_align32);
			if (need_allocate_tmp_out)
				matrix_C = (float*)((char*)(*buffer) + need_A_buffer_len_align32 + need_B_buffer_len_align32);
		}
		t2 = omp_get_wtime();

		cp_dst_ptr = matrix_Bt;
		for (kn = 0, filter_slice_ptr = filters_data; kn < filter_N; kn++, filter_slice_ptr += filter_sliceStep)
		{
			for (kh = 0, filter_row_ptr = filter_slice_ptr; kh < filter_H; kh++, filter_row_ptr += filter_widthStep)
			{
				for (kw = 0, filter_pix_ptr = filter_row_ptr; kw < filter_W; kw++, filter_pix_ptr += filter_pixelStep)
				{
					memcpy(cp_dst_ptr, filter_pix_ptr, sizeof(float)*in_C);
					cp_dst_ptr += in_C;
				}
			}
		}

		matrix_A_row_ptr = matrix_A;
		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
				{
					for (kh = 0, cur_in_row_ptr = in_pix_ptr, matrix_A_col_ptr = matrix_A_row_ptr;
						kh < filter_H;
						kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep)
					{
						for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < filter_W; kw++, cur_in_pix_ptr += dilate_W_mul_in_pixStep)
						{
							memcpy(matrix_A_col_ptr, cur_in_pix_ptr, sizeof(float)*in_C);
							matrix_A_col_ptr += in_C;
						}
					}
					matrix_A_row_ptr += matrix_A_cols;
				}
			}
		}

		t3 = omp_get_wtime();
		/*gemm*/
		zq_cblas_sgemm(zq_CblasRowMajor, zq_CblasNoTrans, zq_CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
			matrix_Bt, matrix_A_cols, 0.0f, matrix_C, matrix_B_cols);
		t4 = omp_get_wtime();
		if (need_allocate_tmp_out)
		{
			/*   col2im      */
			out_row_idx = 0;
			matrix_C_row_ptr = matrix_C;
			for (out_n = 0, out_slice_ptr = out_tensor4D_data; out_n < out_N; out_n++)
			{
				for (out_h = 0, out_row_ptr = out_slice_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
				{
					for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W; out_w++, out_pix_ptr += out_pixelStep)
					{
						memcpy(out_pix_ptr, matrix_C_row_ptr, sizeof(float)*matrix_B_cols);
						matrix_C_row_ptr += matrix_B_cols;
					}
				}
			}
		}
		else
			matrix_C += out_sliceStep;
		t5 = omp_get_wtime();

		if (buffer == 0)
		{
			_aligned_free(matrix_A);
			_aligned_free(matrix_Bt);
			if (need_allocate_tmp_out)
				_aligned_free(matrix_C);
		}
		/*if (filter_H == 3 && filter_W == 3 && filter_C == 3)
		{
			printf("total: %.3f ms, alloc %.3f ms, makeA: %.3f ms, gemm: %.3f ms, copy_C: %.3f ms\n", 1000 * (t5 - t1), 1000 * (t2 - t1),
				1000 * (t3 - t2), 1000 * (t4 - t3), 1000 * (t5 - t4));
		}*/
	}

#if __ARM_NEON
#if __ARM_NEON_FP16
#define zq_base_type float16_t
	/*in_pixStep can be different with filter_pixStep,
	and the aligned channels should be set to zero*/
	void zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep(
		const zq_base_type* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const zq_base_type* filters_data,
		int filter_N,
		int filter_H,
		int filter_W,
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		int dilation_H,
		int dilation_W,
		zq_base_type* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		void** buffer,
		__int64 *buffer_len
	)
	{
		/************** image to col **************/
		int in_widthStep_mul_stride_H = in_widthStep*stride_H;
		int in_pixelStep_mul_stride_W = in_pixelStep*stride_W;
		int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
		int filter_pixStep_mul_filter_W = filter_pixelStep*filter_W;
		int matrix_A_cols = filter_H*filter_W*in_C;
		int matrix_A_rows = out_H*out_W;
		int matrix_B_cols = filter_N;
		int matrix_B_rows = filter_H*filter_W*filter_C;
		__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(zq_base_type) + 31) / 32 * 32;
		__int64 need_B_buffer_len_align32 = (matrix_B_rows*matrix_B_cols * sizeof(zq_base_type) + 31) / 32 * 32;
		__int64 need_C_buffer_len_align32 = 0;
		__int64 total_need_buffer_len;
		zq_base_type* matrix_A = 0;
		zq_base_type* matrix_Bt = 0;
		const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *cur_in_row_ptr, *cur_in_pix_ptr, *filter_slice_ptr, *filter_row_ptr, *filter_pix_ptr;
		int out_n, out_h, out_w, kn, kh, kw;
		zq_base_type* matrix_A_row_ptr, *matrix_A_col_ptr, *cp_dst_ptr;
		zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		zq_base_type* matrix_C = 0, *matrix_C_row_ptr;
		int out_row_idx;
		int out_HW = out_H*out_W;
		double t1, t2, t3, t4, t5;
		int need_allocate_tmp_out;
		t1 = omp_get_wtime();
		need_allocate_tmp_out = (out_pixelStep != filter_N) || (out_pixelStep*out_W != out_widthStep) || (out_widthStep*out_H != out_sliceStep);
		if (need_allocate_tmp_out)
		{
			need_C_buffer_len_align32 = (out_HW*filter_N * sizeof(zq_base_type) + 31) / 32 * 32;
		}
		else
		{
			matrix_C = out_tensor4D_data;
		}

		total_need_buffer_len = need_A_buffer_len_align32 + need_B_buffer_len_align32 + need_C_buffer_len_align32;
		if (buffer == 0)
		{
			matrix_A = (zq_base_type*)_aligned_malloc(need_A_buffer_len_align32, 32);
			matrix_Bt = (zq_base_type*)_aligned_malloc(need_B_buffer_len_align32, 32);
			if (need_allocate_tmp_out)
				matrix_C = (zq_base_type*)_aligned_malloc(need_C_buffer_len_align32, 32);
		}
		else
		{
			if (*buffer_len < total_need_buffer_len)
			{
				_aligned_free(*buffer);
				*buffer = _aligned_malloc(total_need_buffer_len, 32);
				*buffer_len = total_need_buffer_len;
			}
			matrix_A = *buffer;
			matrix_Bt = (zq_base_type*)((char*)(*buffer) + need_A_buffer_len_align32);
			if (need_allocate_tmp_out)
				matrix_C = (zq_base_type*)((char*)(*buffer) + need_A_buffer_len_align32 + need_B_buffer_len_align32);
		}


		cp_dst_ptr = matrix_Bt;
		for (kn = 0, filter_slice_ptr = filters_data; kn < filter_N; kn++, filter_slice_ptr += filter_sliceStep)
		{
			for (kh = 0, filter_row_ptr = filter_slice_ptr; kh < filter_H; kh++, filter_row_ptr += filter_widthStep)
			{
				for (kw = 0, filter_pix_ptr = filter_row_ptr; kw < filter_W; kw++, filter_pix_ptr += filter_pixelStep)
				{
					memcpy(cp_dst_ptr, filter_pix_ptr, sizeof(zq_base_type)*in_C);
					cp_dst_ptr += in_C;
				}
			}
		}

		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			t2 = omp_get_wtime();
			matrix_A_row_ptr = matrix_A;
			for (out_h = 0, in_row_ptr = in_slice_ptr; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
				{
					matrix_A_col_ptr = matrix_A_row_ptr;
					for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < filter_H; kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep)
					{
						for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < filter_W; kw++, cur_in_pix_ptr += dilate_W_mul_in_pixStep)
						{
							memcpy(matrix_A_col_ptr, cur_in_pix_ptr, sizeof(zq_base_type)*in_C);
							matrix_A_col_ptr += in_C;
						}
					}
					matrix_A_row_ptr += matrix_A_cols;
				}
			}
			t3 = omp_get_wtime();
			/*gemm*/
			zq_cblas_sgemm(zq_CblasRowMajor, zq_CblasNoTrans, zq_CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
				matrix_Bt, matrix_A_cols, 0.0f, matrix_C, matrix_B_cols);
			t4 = omp_get_wtime();
			if (need_allocate_tmp_out)
			{
				/*   col2im      */
				out_row_idx = 0;
				matrix_C_row_ptr = matrix_C;
				for (out_h = 0, out_row_ptr = out_slice_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
				{
					for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W; out_w++, out_pix_ptr += out_pixelStep)
					{
						memcpy(out_pix_ptr, matrix_C_row_ptr, sizeof(zq_base_type)*matrix_B_cols);
						matrix_C_row_ptr += matrix_B_cols;
					}
				}
			}
			else
				matrix_C += out_sliceStep;
			t5 = omp_get_wtime();
		}
		if (buffer == 0)
		{
			_aligned_free(matrix_A);
			_aligned_free(matrix_Bt);
			if (need_allocate_tmp_out)
				_aligned_free(matrix_C);
		}
		/*if (filter_H == 3 && filter_W == 3 && filter_C == 3)
		{
		printf("total: %.3f ms, alloc %.3f ms, makeA: %.3f ms, gemm: %.3f ms, copy_C: %.3f ms\n", 1000 * (t5 - t1), 1000 * (t2 - t1),
		1000 * (t3 - t2), 1000 * (t4 - t3), 1000 * (t5 - t4));
		}*/
	}

	/*in_pixStep can be different with filter_pixStep,
	and the aligned channels should be set to zero*/
	void zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep_batch(
		const zq_base_type* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const zq_base_type* filters_data,
		int filter_N,
		int filter_H,
		int filter_W,
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		int dilation_H,
		int dilation_W,
		zq_base_type* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		void** buffer,
		__int64 *buffer_len
	)
	{
		/************** image to col **************/
		int in_widthStep_mul_stride_H = in_widthStep*stride_H;
		int in_pixelStep_mul_stride_W = in_pixelStep*stride_W;
		int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
		int filter_pixStep_mul_filter_W = filter_pixelStep*filter_W;
		int matrix_A_cols = in_C;
		int matrix_A_rows = out_N*out_H*out_W;
		int matrix_B_cols = filter_N;
		int matrix_B_rows = filter_H*filter_W*filter_C;
		__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(zq_base_type) + 31) / 32 * 32;
		__int64 need_B_buffer_len_align32 = (matrix_B_rows*matrix_B_cols * sizeof(zq_base_type) + 31) / 32 * 32;
		__int64 need_C_buffer_len_align32 = 0;
		__int64 total_need_buffer_len;
		zq_base_type* matrix_A = 0;
		zq_base_type* matrix_Bt = 0;
		const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *cur_in_row_ptr, *cur_in_pix_ptr, *filter_slice_ptr, *filter_row_ptr, *filter_pix_ptr;
		int out_n, out_h, out_w, kn, kh, kw;
		zq_base_type* matrix_A_row_ptr, *matrix_A_col_ptr, *cp_dst_ptr;
		zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		zq_base_type* matrix_C = 0, *matrix_C_row_ptr;
		int out_row_idx;
		int out_NHW = out_N*out_H*out_W;
		double t1, t2, t3, t4, t5;
		t1 = omp_get_wtime();
		int need_allocate_tmp_out = (out_pixelStep != filter_N) || (out_pixelStep*out_W != out_widthStep) || (out_widthStep*out_H != out_sliceStep);
		if (need_allocate_tmp_out)
		{
			need_C_buffer_len_align32 = (out_NHW*filter_N * sizeof(zq_base_type) + 31) / 32 * 32;
		}
		else
		{
			matrix_C = out_tensor4D_data;
		}

		total_need_buffer_len = need_A_buffer_len_align32 + need_B_buffer_len_align32 + need_C_buffer_len_align32;
		if (buffer == 0)
		{
			matrix_A = (zq_base_type*)_aligned_malloc(need_A_buffer_len_align32, 32);
			matrix_Bt = (zq_base_type*)_aligned_malloc(need_B_buffer_len_align32, 32);
			if (need_allocate_tmp_out)
				matrix_C = (zq_base_type*)_aligned_malloc(need_C_buffer_len_align32, 32);
		}
		else
		{
			if (*buffer_len < total_need_buffer_len)
			{
				_aligned_free(*buffer);
				*buffer = _aligned_malloc(total_need_buffer_len, 32);
				*buffer_len = total_need_buffer_len;
			}
			matrix_A = *buffer;
			matrix_Bt = (zq_base_type*)((char*)(*buffer) + need_A_buffer_len_align32);
			if (need_allocate_tmp_out)
				matrix_C = (zq_base_type*)((char*)(*buffer) + need_A_buffer_len_align32 + need_B_buffer_len_align32);
		}
		t2 = omp_get_wtime();

		cp_dst_ptr = matrix_Bt;
		for (kn = 0, filter_slice_ptr = filters_data; kn < filter_N; kn++, filter_slice_ptr += filter_sliceStep)
		{
			for (kh = 0, filter_row_ptr = filter_slice_ptr; kh < filter_H; kh++, filter_row_ptr += filter_widthStep)
			{
				for (kw = 0, filter_pix_ptr = filter_row_ptr; kw < filter_W; kw++, filter_pix_ptr += filter_pixelStep)
				{
					memcpy(cp_dst_ptr, filter_pix_ptr, sizeof(zq_base_type)*in_C);
					cp_dst_ptr += in_C;
				}
			}
		}

		matrix_A_row_ptr = matrix_A;
		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
				{
					for (kh = 0, cur_in_row_ptr = in_pix_ptr, matrix_A_col_ptr = matrix_A_row_ptr;
						kh < filter_H;
						kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep)
					{
						for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < filter_W; kw++, cur_in_pix_ptr += dilate_W_mul_in_pixStep)
						{
							memcpy(matrix_A_col_ptr, cur_in_pix_ptr, sizeof(zq_base_type)*in_C);
							matrix_A_col_ptr += in_C;
						}
					}
					matrix_A_row_ptr += matrix_A_cols;
				}
			}
		}

		t3 = omp_get_wtime();
		/*gemm*/
		zq_cblas_sgemm(zq_CblasRowMajor, zq_CblasNoTrans, zq_CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
			matrix_Bt, matrix_A_cols, 0.0f, matrix_C, matrix_B_cols);
		t4 = omp_get_wtime();
		if (need_allocate_tmp_out)
		{
			/*   col2im      */
			out_row_idx = 0;
			matrix_C_row_ptr = matrix_C;
			for (out_n = 0, out_slice_ptr = out_tensor4D_data; out_n < out_N; out_n++)
			{
				for (out_h = 0, out_row_ptr = out_slice_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
				{
					for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W; out_w++, out_pix_ptr += out_pixelStep)
					{
						memcpy(out_pix_ptr, matrix_C_row_ptr, sizeof(zq_base_type)*matrix_B_cols);
						matrix_C_row_ptr += matrix_B_cols;
					}
				}
			}
		}
		else
			matrix_C += out_sliceStep;
		t5 = omp_get_wtime();

		if (buffer == 0)
		{
			_aligned_free(matrix_A);
			_aligned_free(matrix_Bt);
			if (need_allocate_tmp_out)
				_aligned_free(matrix_C);
		}
		/*if (filter_H == 3 && filter_W == 3 && filter_C == 3)
		{
		printf("total: %.3f ms, alloc %.3f ms, makeA: %.3f ms, gemm: %.3f ms, copy_C: %.3f ms\n", 1000 * (t5 - t1), 1000 * (t2 - t1),
		1000 * (t3 - t2), 1000 * (t4 - t3), 1000 * (t5 - t4));
		}*/
	}
#undef zq_base_type
#endif
#endif

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
