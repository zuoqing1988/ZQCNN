#include <stdio.h>
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
#endif

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

#if __ARM_NEON
#define zq_cnn_innerproduct_32f_align zq_cnn_innerproduct_32f_align128bit
#define	zq_cnn_innerproduct_32f_align_noborder zq_cnn_innerproduct_32f_align128bit_noborder
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

#include "zq_cnn_innerproduct_32f_align_c_raw.h"


#undef zq_cnn_innerproduct_32f_align 
#undef zq_cnn_innerproduct_32f_align_noborder
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
#define zq_cnn_innerproduct_32f_align zq_cnn_innerproduct_16f_align128bit
#define	zq_cnn_innerproduct_32f_align_noborder zq_cnn_innerproduct_16f_align128bit_noborder
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

#include "zq_cnn_innerproduct_32f_align_c_raw.h"


#undef zq_cnn_innerproduct_32f_align 
#undef zq_cnn_innerproduct_32f_align_noborder
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
#define zq_cnn_innerproduct_32f_align zq_cnn_innerproduct_32f_align128bit
#define	zq_cnn_innerproduct_32f_align_noborder zq_cnn_innerproduct_32f_align128bit_noborder
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

#include "zq_cnn_innerproduct_32f_align_c_raw.h"


#undef zq_cnn_innerproduct_32f_align 
#undef zq_cnn_innerproduct_32f_align_noborder
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
#define zq_cnn_innerproduct_32f_align zq_cnn_innerproduct_32f_align256bit
#define	zq_cnn_innerproduct_32f_align_noborder zq_cnn_innerproduct_32f_align256bit_noborder
#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_add_ps _mm256_add_ps
#if ZQ_CNN_USE_FMADD128
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

#include "zq_cnn_innerproduct_32f_align_c_raw.h"


#undef zq_cnn_innerproduct_32f_align 
#undef zq_cnn_innerproduct_32f_align_noborder
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

	void zq_cnn_innerproduct_32f_align0_general(
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
		//int filter_H, // must be in_H
		//int filter_W, // must be in_W
		//int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		float* out_tensor4D_data,
		//int out_N,	// must be in_N
		//int out_H,	// must be 1
		//int out_W,	// must be 1
		//int out_C,	// must be filter_N
		//int out_pixelStep,
		//int out_widthStep,
		int out_sliceStep
	)
	{
		int out_n, out_c, kh,kw,kc;
		const float* in_slice_ptr;
		const float* filter_slice_ptr, *filter_row_ptr, *filter_pix_ptr, *filter_c_ptr;
		float* out_slice_ptr, *out_c_ptr;
		const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
		float sum;
		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < in_N; out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (out_c = 0, out_c_ptr = out_slice_ptr, filter_slice_ptr = filters_data;
				out_c < filter_N;
				out_c++, out_c_ptr++, filter_slice_ptr += filter_sliceStep)
			{
				sum = 0;
				for (kh = 0, cur_in_row_ptr = in_slice_ptr, filter_row_ptr = filter_slice_ptr;
					kh < in_H;
					kh++, cur_in_row_ptr += in_widthStep, filter_row_ptr += filter_widthStep)
				{
					for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr, filter_pix_ptr = filter_row_ptr;
						kw < in_W;
						kw++, cur_in_pix_ptr += in_pixelStep, filter_pix_ptr += filter_pixelStep)
					{
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, filter_c_ptr = filter_pix_ptr;
							kc < in_C; kc++, cur_in_c_ptr++, filter_c_ptr++)
							sum += (*cur_in_c_ptr)*(*filter_c_ptr);
					}
				}
				*out_c_ptr = sum;
			}
		}

	}

	void zq_cnn_innerproduct_32f_align0_noborder(
		const float* in_tensor4D_data,
		int in_N,
		int in_HWC,//in_H*in_W*in_C
		//int in_W,
		//int in_C,
		//int in_pixelStep,
		//int in_widthStep,
		//int in_sliceStep,
		const float* filters_data,
		int filter_N,
		//int filter_H, // must be in_H
		//int filter_W, // must be in_W
		//int filter_C, // must be in_C
		//int filter_pixelStep,
		//int filter_widthStep,
		//int filter_sliceStep,
		float* out_tensor4D_data,
		//int out_N,	// must be in_N
		//int out_H,	// must be 1
		//int out_W,	// must be 1
		//int out_C,	// must be filter_N
		//int out_pixelStep,
		//int out_widthStep,
		int out_sliceStep
	)
	{
		int out_n, out_c, in_hwc;
		const float* in_slice_ptr;
		float* out_slice_ptr;
		const float* filter_slice_ptr, *filter_c_ptr;
		const float* cur_in_c_ptr;
		float* out_pos_ptr;
		float sum;
		
		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < in_N; out_n++, in_slice_ptr += in_HWC, out_slice_ptr += out_sliceStep)
		{
			out_pos_ptr = out_slice_ptr;
			for (out_c = 0, filter_slice_ptr = filters_data;
				out_c < filter_N;
				out_c++, filter_slice_ptr += in_HWC)
			{
				sum = 0;
				for (in_hwc = 0, cur_in_c_ptr = in_slice_ptr, filter_c_ptr = filter_slice_ptr;
					in_hwc < in_HWC;
					in_hwc++)
				{
					sum += *(cur_in_c_ptr++) * (*(filter_c_ptr++));
				}
				*(out_pos_ptr++) = sum;
			}
		}
	}

#if __ARM_NEON
#if __ARM_NEON_FP16
#define zq_base_type float16_t
	void zq_cnn_innerproduct_16f_align0_general(
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
		//int filter_H, // must be in_H
		//int filter_W, // must be in_W
		//int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		zq_base_type* out_tensor4D_data,
		//int out_N,	// must be in_N
		//int out_H,	// must be 1
		//int out_W,	// must be 1
		//int out_C,	// must be filter_N
		//int out_pixelStep,
		//int out_widthStep,
		int out_sliceStep
	)
	{
		int out_n, out_c, kh, kw, kc;
		const zq_base_type* in_slice_ptr;
		const zq_base_type* filter_slice_ptr, *filter_row_ptr, *filter_pix_ptr, *filter_c_ptr;
		zq_base_type* out_slice_ptr, *out_c_ptr;
		const zq_base_type* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
		zq_base_type sum;
		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < in_N; out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (out_c = 0, out_c_ptr = out_slice_ptr, filter_slice_ptr = filters_data;
				out_c < filter_N;
				out_c++, out_c_ptr++, filter_slice_ptr += filter_sliceStep)
			{
				sum = 0;
				for (kh = 0, cur_in_row_ptr = in_slice_ptr, filter_row_ptr = filter_slice_ptr;
					kh < in_H;
					kh++, cur_in_row_ptr += in_widthStep, filter_row_ptr += filter_widthStep)
				{
					for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr, filter_pix_ptr = filter_row_ptr;
						kw < in_W;
						kw++, cur_in_pix_ptr += in_pixelStep, filter_pix_ptr += filter_pixelStep)
					{
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, filter_c_ptr = filter_pix_ptr;
							kc < in_C; kc++, cur_in_c_ptr++, filter_c_ptr++)
							sum += (*cur_in_c_ptr)*(*filter_c_ptr);
					}
				}
				*out_c_ptr = sum;
			}
		}

	}

	void zq_cnn_innerproduct_16f_align0_noborder(
		const zq_base_type* in_tensor4D_data,
		int in_N,
		int in_HWC,//in_H*in_W*in_C
				   //int in_W,
				   //int in_C,
				   //int in_pixelStep,
				   //int in_widthStep,
				   //int in_sliceStep,
		const zq_base_type* filters_data,
		int filter_N,
		//int filter_H, // must be in_H
		//int filter_W, // must be in_W
		//int filter_C, // must be in_C
		//int filter_pixelStep,
		//int filter_widthStep,
		//int filter_sliceStep,
		zq_base_type* out_tensor4D_data,
		//int out_N,	// must be in_N
		//int out_H,	// must be 1
		//int out_W,	// must be 1
		//int out_C,	// must be filter_N
		//int out_pixelStep,
		//int out_widthStep,
		int out_sliceStep
	)
	{
		int out_n, out_c, in_hwc;
		const zq_base_type* in_slice_ptr;
		zq_base_type* out_slice_ptr;
		const zq_base_type* filter_slice_ptr, *filter_c_ptr;
		const zq_base_type* cur_in_c_ptr;
		zq_base_type* out_pos_ptr;
		zq_base_type sum;

		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < in_N; out_n++, in_slice_ptr += in_HWC, out_slice_ptr += out_sliceStep)
		{
			out_pos_ptr = out_slice_ptr;
			for (out_c = 0, filter_slice_ptr = filters_data;
				out_c < filter_N;
				out_c++, filter_slice_ptr += in_HWC)
			{
				sum = 0;
				for (in_hwc = 0, cur_in_c_ptr = in_slice_ptr, filter_c_ptr = filter_slice_ptr;
					in_hwc < in_HWC;
					in_hwc++)
				{
					sum += *(cur_in_c_ptr++) * (*(filter_c_ptr++));
				}
				*(out_pos_ptr++) = sum;
			}
		}
	}
#undef zq_base_type
#endif//__ARM_NEON_FP16
#endif//__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif