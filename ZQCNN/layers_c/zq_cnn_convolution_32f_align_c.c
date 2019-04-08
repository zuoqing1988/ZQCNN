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
#endif //__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

#if __ARM_NEON
#define zq_cnn_conv_no_padding_32f_kernel1x1 zq_cnn_conv_no_padding_32f_align128bit_kernel1x1
#define zq_cnn_conv_no_padding_32f_kernel1x1_C4 zq_cnn_conv_no_padding_32f_align128bit_kernel1x1_C4
#define zq_cnn_conv_no_padding_32f_kernel2x2 zq_cnn_conv_no_padding_32f_align128bit_kernel2x2
#define zq_cnn_conv_no_padding_32f_kernel3x3 zq_cnn_conv_no_padding_32f_align128bit_kernel3x3
#define zq_cnn_conv_no_padding_32f_kernel3x3_C3 zq_cnn_conv_no_padding_32f_align128bit_kernel3x3_C3
#define zq_cnn_conv_no_padding_32f_kernel3x3_C3_s1d1 zq_cnn_conv_no_padding_32f_align128bit_kernel3x3_C3_s1d1
#define zq_cnn_conv_no_padding_32f_kernel3x3_C3_s2d1 zq_cnn_conv_no_padding_32f_align128bit_kernel3x3_C3_s2d1
#define zq_cnn_conv_no_padding_32f_kernel5x5 zq_cnn_conv_no_padding_32f_align128bit_kernel5x5
#define zq_cnn_conv_no_padding_32f_general zq_cnn_conv_no_padding_32f_align128bit_general
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
#define zq_mm_align_size_mul_2 8
#define zq_mm_align_size_mul_3 12
#define zq_mm_align_size_mul_4 16
#define zq_mm_align_size_mul_5 20
#define zq_mm_align_size_mul_6 24
#define zq_mm_align_size_mul_7 28
#define zq_mm_align_size_mul_8 32
#define zq_mm_align_size_mul_16 64
#define zq_mm_align_size_mul_32 128
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFF0
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3])

#include "zq_cnn_convolution_32f_align_c_raw.h"


#undef zq_cnn_conv_no_padding_32f_kernel1x1
#undef zq_cnn_conv_no_padding_32f_kernel1x1_C4
#undef zq_cnn_conv_no_padding_32f_kernel2x2
#undef zq_cnn_conv_no_padding_32f_kernel3x3
#undef zq_cnn_conv_no_padding_32f_kernel3x3_C3
#undef zq_cnn_conv_no_padding_32f_kernel3x3_C3_s1d1
#undef zq_cnn_conv_no_padding_32f_kernel3x3_C3_s2d1
#undef zq_cnn_conv_no_padding_32f_kernel5x5
#undef zq_cnn_conv_no_padding_32f_general
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#undef zq_mm_align_size_mul_2
#undef zq_mm_align_size_mul_3
#undef zq_mm_align_size_mul_4
#undef zq_mm_align_size_mul_5
#undef zq_mm_align_size_mul_6
#undef zq_mm_align_size_mul_7
#undef zq_mm_align_size_mul_8
#undef zq_mm_align_size_mul_16
#undef zq_mm_align_size_mul_32
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q

#if __ARM_NEON_FP16
#define zq_cnn_conv_no_padding_32f_kernel1x1 zq_cnn_conv_no_padding_16f_align128bit_kernel1x1
#define zq_cnn_conv_no_padding_32f_kernel1x1_C4 zq_cnn_conv_no_padding_16f_align128bit_kernel1x1_C4
#define zq_cnn_conv_no_padding_32f_kernel2x2 zq_cnn_conv_no_padding_16f_align128bit_kernel2x2
#define zq_cnn_conv_no_padding_32f_kernel3x3 zq_cnn_conv_no_padding_16f_align128bit_kernel3x3
#define zq_cnn_conv_no_padding_32f_kernel3x3_C3 zq_cnn_conv_no_padding_16f_align128bit_kernel3x3_C3
#define zq_cnn_conv_no_padding_32f_kernel3x3_C3_s1d1 zq_cnn_conv_no_padding_16f_align128bit_kernel3x3_C3_s1d1
#define zq_cnn_conv_no_padding_32f_kernel3x3_C3_s2d1 zq_cnn_conv_no_padding_16f_align128bit_kernel3x3_C3_s2d1
#define zq_cnn_conv_no_padding_32f_kernel5x5 zq_cnn_conv_no_padding_16f_align128bit_kernel5x5
#define zq_cnn_conv_no_padding_32f_general zq_cnn_conv_no_padding_16f_align128bit_general
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
#define zq_mm_align_size_mul_2 16
#define zq_mm_align_size_mul_3 24
#define zq_mm_align_size_mul_4 32
#define zq_mm_align_size_mul_5 40
#define zq_mm_align_size_mul_6 48
#define zq_mm_align_size_mul_7 56
#define zq_mm_align_size_mul_8 64
#define zq_mm_align_size_mul_16 128
#define zq_mm_align_size_mul_32 256
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFF0
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7]+q[8])

#include "zq_cnn_convolution_32f_align_c_raw.h"


#undef zq_cnn_conv_no_padding_32f_kernel1x1
#undef zq_cnn_conv_no_padding_32f_kernel1x1_C4
#undef zq_cnn_conv_no_padding_32f_kernel2x2
#undef zq_cnn_conv_no_padding_32f_kernel3x3
#undef zq_cnn_conv_no_padding_32f_kernel3x3_C3
#undef zq_cnn_conv_no_padding_32f_kernel3x3_C3_s1d1
#undef zq_cnn_conv_no_padding_32f_kernel3x3_C3_s2d1
#undef zq_cnn_conv_no_padding_32f_kernel5x5
#undef zq_cnn_conv_no_padding_32f_general
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#undef zq_mm_align_size_mul_2
#undef zq_mm_align_size_mul_3
#undef zq_mm_align_size_mul_4
#undef zq_mm_align_size_mul_5
#undef zq_mm_align_size_mul_6
#undef zq_mm_align_size_mul_7
#undef zq_mm_align_size_mul_8
#undef zq_mm_align_size_mul_16
#undef zq_mm_align_size_mul_32
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q
#endif

#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#define zq_cnn_conv_no_padding_32f_kernel1x1 zq_cnn_conv_no_padding_32f_align128bit_kernel1x1
#define zq_cnn_conv_no_padding_32f_kernel1x1_C4 zq_cnn_conv_no_padding_32f_align128bit_kernel1x1_C4
#define zq_cnn_conv_no_padding_32f_kernel2x2 zq_cnn_conv_no_padding_32f_align128bit_kernel2x2
#define zq_cnn_conv_no_padding_32f_kernel3x3 zq_cnn_conv_no_padding_32f_align128bit_kernel3x3
#define zq_cnn_conv_no_padding_32f_kernel3x3_C3 zq_cnn_conv_no_padding_32f_align128bit_kernel3x3_C3
#define zq_cnn_conv_no_padding_32f_kernel3x3_C3_s1d1 zq_cnn_conv_no_padding_32f_align128bit_kernel3x3_C3_s1d1
#define zq_cnn_conv_no_padding_32f_kernel3x3_C3_s2d1 zq_cnn_conv_no_padding_32f_align128bit_kernel3x3_C3_s2d1
#define zq_cnn_conv_no_padding_32f_kernel5x5 zq_cnn_conv_no_padding_32f_align128bit_kernel5x5
#define zq_cnn_conv_no_padding_32f_general zq_cnn_conv_no_padding_32f_align128bit_general
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
#define zq_mm_align_size_mul_2 8
#define zq_mm_align_size_mul_3 12
#define zq_mm_align_size_mul_4 16
#define zq_mm_align_size_mul_5 20
#define zq_mm_align_size_mul_6 24
#define zq_mm_align_size_mul_7 28
#define zq_mm_align_size_mul_8 32
#define zq_mm_align_size_mul_16 64
#define zq_mm_align_size_mul_32 128
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFF0
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3])

#include "zq_cnn_convolution_32f_align_c_raw.h"


#undef zq_cnn_conv_no_padding_32f_kernel1x1
#undef zq_cnn_conv_no_padding_32f_kernel1x1_C4
#undef zq_cnn_conv_no_padding_32f_kernel2x2
#undef zq_cnn_conv_no_padding_32f_kernel3x3
#undef zq_cnn_conv_no_padding_32f_kernel3x3_C3
#undef zq_cnn_conv_no_padding_32f_kernel3x3_C3_s1d1
#undef zq_cnn_conv_no_padding_32f_kernel3x3_C3_s2d1
#undef zq_cnn_conv_no_padding_32f_kernel5x5
#undef zq_cnn_conv_no_padding_32f_general
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#undef zq_mm_align_size_mul_2
#undef zq_mm_align_size_mul_3
#undef zq_mm_align_size_mul_4
#undef zq_mm_align_size_mul_5
#undef zq_mm_align_size_mul_6
#undef zq_mm_align_size_mul_7
#undef zq_mm_align_size_mul_8
#undef zq_mm_align_size_mul_16
#undef zq_mm_align_size_mul_32
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q

#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#define zq_cnn_conv_no_padding_32f_kernel1x1 zq_cnn_conv_no_padding_32f_align256bit_kernel1x1
#define zq_cnn_conv_no_padding_32f_kernel1x1_C4 zq_cnn_conv_no_padding_32f_align256bit_kernel1x1_C4
#define zq_cnn_conv_no_padding_32f_kernel2x2 zq_cnn_conv_no_padding_32f_align256bit_kernel2x2
#define zq_cnn_conv_no_padding_32f_kernel3x3 zq_cnn_conv_no_padding_32f_align256bit_kernel3x3
#define zq_cnn_conv_no_padding_32f_kernel3x3_C3 zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C3
#define zq_cnn_conv_no_padding_32f_kernel3x3_C3_s1d1 zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C3_s1d1
#define zq_cnn_conv_no_padding_32f_kernel3x3_C3_s2d1 zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C3_s2d1
#define zq_cnn_conv_no_padding_32f_kernel5x5 zq_cnn_conv_no_padding_32f_align256bit_kernel5x5
#define zq_cnn_conv_no_padding_32f_general zq_cnn_conv_no_padding_32f_align256bit_general
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
#define zq_mm_align_size_mul_2 16
#define zq_mm_align_size_mul_3 24
#define zq_mm_align_size_mul_4 32
#define zq_mm_align_size_mul_5 40
#define zq_mm_align_size_mul_6 48
#define zq_mm_align_size_mul_7 56
#define zq_mm_align_size_mul_8 64
#define zq_mm_align_size_mul_16 128
#define zq_mm_align_size_mul_32 256
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFE0
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])


#include "zq_cnn_convolution_32f_align_c_raw.h"

#undef zq_cnn_conv_no_padding_32f_kernel1x1
#undef zq_cnn_conv_no_padding_32f_kernel1x1_C4
#undef zq_cnn_conv_no_padding_32f_kernel2x2
#undef zq_cnn_conv_no_padding_32f_kernel3x3
#undef zq_cnn_conv_no_padding_32f_kernel3x3_C3
#undef zq_cnn_conv_no_padding_32f_kernel3x3_C3_s1d1
#undef zq_cnn_conv_no_padding_32f_kernel3x3_C3_s2d1
#undef zq_cnn_conv_no_padding_32f_kernel5x5
#undef zq_cnn_conv_no_padding_32f_general
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#undef zq_mm_align_size_mul_2
#undef zq_mm_align_size_mul_3
#undef zq_mm_align_size_mul_4
#undef zq_mm_align_size_mul_5
#undef zq_mm_align_size_mul_6
#undef zq_mm_align_size_mul_7
#undef zq_mm_align_size_mul_8
#undef zq_mm_align_size_mul_16
#undef zq_mm_align_size_mul_32
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q

#endif

#endif //__ARM_NEON

	void zq_cnn_conv_no_padding_32f_align0_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_alignPixelStep,
		int in_widthStep,
		int in_SliceStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 1
		int filter_W, // must be 1
		int filter_C, // must be in_C
		int filter_alignPixelStep,
		int filter_widthStep,
		int filter_SliceStep,
		int stride_H,
		int stride_W,
		int dilation_H,
		int dilation_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_alignPixelStep,
		int out_alignWidthStep,
		int out_alignSliceStep
	)
	{
		float sum;
		const float* in_slice_ptr;
		const float* in_row_ptr;
		const float* in_pix_ptr;
		float* out_slice_ptr;
		float* out_row_ptr;
		float* out_pix_ptr;
		float* out_c_ptr;

		const float* cur_in_row_ptr;
		const float* cur_in_pix_ptr;
		const float* cur_in_c_ptr;
		const float* cur_filter_slice_ptr;
		const float* cur_filter_row_ptr;
		const float* cur_filter_pix_ptr;
		const float* cur_filter_c_ptr;

		int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
		int stride_W_mul_in_pixStep = stride_W*in_alignPixelStep;
		int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		int dilate_W_mul_in_pixStep = dilation_W*in_alignPixelStep;
		int out_n, out_h, out_w, out_c, kh, kw, kc;


		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_SliceStep, out_slice_ptr += out_alignSliceStep)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				out_h < out_H;
				out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_alignWidthStep)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					out_w < out_W;
					out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_alignPixelStep)
				{
					for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr += filter_SliceStep, out_c_ptr++)
					{
						sum = 0;
						for (kh = 0, cur_filter_row_ptr = cur_filter_slice_ptr, cur_in_row_ptr = in_pix_ptr;
							kh < filter_H;
							kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep, cur_filter_row_ptr += filter_widthStep)
						{
							for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr, cur_filter_pix_ptr = cur_filter_row_ptr;
								kw < filter_W;
								kw++, cur_in_pix_ptr += dilate_W_mul_in_pixStep, cur_filter_pix_ptr += filter_alignPixelStep)
							{
								for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
									kc < filter_C;
									kc++, cur_in_c_ptr++, cur_filter_c_ptr++)
								{
									sum += (*cur_in_c_ptr) * (*cur_filter_c_ptr);
								}
							}
						}
						*out_c_ptr = sum;
					}
				}
			}
		}
	}

#if __ARM_NEON
#if __ARM_NEON_FP16
#define zq_base_type float16_t
	void zq_cnn_conv_no_padding_16f_align0_general(
		const zq_base_type* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_alignPixelStep,
		int in_widthStep,
		int in_SliceStep,
		const zq_base_type* filters_data,
		int filter_N,
		int filter_H, // must be 1
		int filter_W, // must be 1
		int filter_C, // must be in_C
		int filter_alignPixelStep,
		int filter_widthStep,
		int filter_SliceStep,
		int stride_H,
		int stride_W,
		int dilation_H,
		int dilation_W,
		zq_base_type* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_alignPixelStep,
		int out_alignWidthStep,
		int out_alignSliceStep
	)
	{
		zq_base_type sum;
		const zq_base_type* in_slice_ptr;
		const zq_base_type* in_row_ptr;
		const zq_base_type* in_pix_ptr;
		zq_base_type* out_slice_ptr;
		zq_base_type* out_row_ptr;
		zq_base_type* out_pix_ptr;
		zq_base_type* out_c_ptr;

		const zq_base_type* cur_in_row_ptr;
		const zq_base_type* cur_in_pix_ptr;
		const zq_base_type* cur_in_c_ptr;
		const zq_base_type* cur_filter_slice_ptr;
		const zq_base_type* cur_filter_row_ptr;
		const zq_base_type* cur_filter_pix_ptr;
		const zq_base_type* cur_filter_c_ptr;

		int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
		int stride_W_mul_in_pixStep = stride_W*in_alignPixelStep;
		int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		int dilate_W_mul_in_pixStep = dilation_W*in_alignPixelStep;
		int out_n, out_h, out_w, out_c, kh, kw, kc;


		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_SliceStep, out_slice_ptr += out_alignSliceStep)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				out_h < out_H;
				out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_alignWidthStep)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					out_w < out_W;
					out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_alignPixelStep)
				{
					for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr += filter_SliceStep, out_c_ptr++)
					{
						sum = 0;
						for (kh = 0, cur_filter_row_ptr = cur_filter_slice_ptr, cur_in_row_ptr = in_pix_ptr;
							kh < filter_H;
							kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep, cur_filter_row_ptr += filter_widthStep)
						{
							for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr, cur_filter_pix_ptr = cur_filter_row_ptr;
								kw < filter_W;
								kw++, cur_in_pix_ptr += dilate_W_mul_in_pixStep, cur_filter_pix_ptr += filter_alignPixelStep)
							{
								for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
									kc < filter_C;
									kc++, cur_in_c_ptr++, cur_filter_c_ptr++)
								{
									sum += (*cur_in_c_ptr) * (*cur_filter_c_ptr);
								}
							}
						}
						*out_c_ptr = sum;
					}
				}
			}
		}
	}
#undef zq_base_type
#endif
#endif




#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
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
#define zq_mm_align_size_mul_2 16
#define zq_mm_align_size_mul_3 24
#define zq_mm_align_size_mul_4 32
#define zq_mm_align_size_mul_5 40
#define zq_mm_align_size_mul_6 48
#define zq_mm_align_size_mul_7 56
#define zq_mm_align_size_mul_8 64
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])

	/*if C is not 8 (i.e. 1,2,3,4,5,6,7), the padded channels of filters must be zero*/
	void zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C8(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 1,2,3,4,5,6,7,8
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 1,2,3,4,5,6,7,8
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
		int out_sliceStep
	)
	{
		zq_mm_type sum;
		ZQ_DECLSPEC_ALIGN32 float q[8];

		const float* in_slice_ptr;
		const float* in_row_ptr;
		const float* in_pix_ptr;
		float* out_slice_ptr;
		float* out_row_ptr;
		float* out_pix_ptr;
		float* out_c_ptr;

		const float* cur_in_row_ptr;
		const float* cur_in_pix_ptr;
		const float* cur_filter_slice_ptr;
		const float* cur_filter_row_ptr;
		const float* cur_filter_pix_ptr;

		int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
		int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
		int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
		int out_n, out_h, out_w, out_c;
		double t1, t2;
		t1 = omp_get_wtime();

		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				out_h < out_H;
				out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					out_w < out_W;
					out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
				{
					for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
					{
						sum = zq_mm_setzero_ps();
						cur_in_row_ptr = in_pix_ptr;	cur_filter_row_ptr = cur_filter_slice_ptr;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						zq_mm_store_ps(q, sum);
						*out_c_ptr = zq_final_sum_q;
					}
				}
			}

		}
		t2 = omp_get_wtime();
		//printf("base:conv3x3x8: %.3f ms\n", (t2 - t1) * 1000);

	}


	/*if C is not 16 (i.e. 9,10,11,12,13,14,15), the padded channels of filters must be zero*/
	void zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C16(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 9,10,11,12,13,14,15,16
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 9,10,11,12,13,14,15,16
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
		int out_sliceStep
	)
	{
		zq_mm_type sum;
		ZQ_DECLSPEC_ALIGN32 float q[8];

		const float* in_slice_ptr;
		const float* in_row_ptr;
		const float* in_pix_ptr;
		float* out_slice_ptr;
		float* out_row_ptr;
		float* out_pix_ptr;
		float* out_c_ptr;

		const float* cur_in_row_ptr;
		const float* cur_in_pix_ptr;
		const float* cur_filter_slice_ptr;
		const float* cur_filter_row_ptr;
		const float* cur_filter_pix_ptr;

		int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
		int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
		int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
		int out_n, out_h, out_w, out_c;
		double t1, t2;
		t1 = omp_get_wtime();

		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				out_h < out_H;
				out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					out_w < out_W;
					out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
				{
					for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
					{
						sum = zq_mm_setzero_ps();
						cur_in_row_ptr = in_pix_ptr;	cur_filter_row_ptr = cur_filter_slice_ptr;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						zq_mm_store_ps(q, sum);
						*out_c_ptr = zq_final_sum_q;
					}
				}
			}

		}
		t2 = omp_get_wtime();
		//printf("base:conv3x3x16: %.3f ms\n", (t2 - t1) * 1000);
	}


	/*if C is not 24 (i.e. 17,18,19,20,21,22,23), the padded channels of filters must be zero*/
	void zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C24(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 17,18,19,20,21,22,23,24
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 17,18,19,20,21,22,23,24
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
		int out_sliceStep
	)
	{
		zq_mm_type sum;
		ZQ_DECLSPEC_ALIGN32 float q[8];

		const float* in_slice_ptr;
		const float* in_row_ptr;
		const float* in_pix_ptr;
		float* out_slice_ptr;
		float* out_row_ptr;
		float* out_pix_ptr;
		float* out_c_ptr;

		const float* cur_in_row_ptr;
		const float* cur_in_pix_ptr;
		const float* cur_filter_slice_ptr;
		const float* cur_filter_row_ptr;
		const float* cur_filter_pix_ptr;

		int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
		int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
		int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
		int out_n, out_h, out_w, out_c;
		double t1, t2;
		t1 = omp_get_wtime();

		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				out_h < out_H;
				out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					out_w < out_W;
					out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
				{
					for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
					{
						sum = zq_mm_setzero_ps();
						cur_in_row_ptr = in_pix_ptr;	cur_filter_row_ptr = cur_filter_slice_ptr;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						zq_mm_store_ps(q, sum);
						*out_c_ptr = zq_final_sum_q;
					}
				}
			}

		}
		t2 = omp_get_wtime();
		//printf("base:conv3x3x24: %.3f ms\n", (t2 - t1) * 1000);
	}


	/*if C is not 32 (i.e. 25,26,27,28,29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C32(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 25,26,27,28,29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 25,26,27,28,29,30,31,32
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
		int out_sliceStep
	)
	{
		zq_mm_type sum;
		ZQ_DECLSPEC_ALIGN32 float q[8];

		const float* in_slice_ptr;
		const float* in_row_ptr;
		const float* in_pix_ptr;
		float* out_slice_ptr;
		float* out_row_ptr;
		float* out_pix_ptr;
		float* out_c_ptr;

		const float* cur_in_row_ptr;
		const float* cur_in_pix_ptr;
		const float* cur_filter_slice_ptr;
		const float* cur_filter_row_ptr;
		const float* cur_filter_pix_ptr;

		int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
		int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
		int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
		int out_n, out_h, out_w, out_c;
		double t1, t2;
		t1 = omp_get_wtime();

		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				out_h < out_H;
				out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					out_w < out_W;
					out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
				{
					for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
					{
						sum = zq_mm_setzero_ps();
						cur_in_row_ptr = in_pix_ptr;	cur_filter_row_ptr = cur_filter_slice_ptr;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_3), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_3), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_3), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_3), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_3), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_3), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_3), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_3), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_3), sum);
						zq_mm_store_ps(q, sum);
						*out_c_ptr = zq_final_sum_q;
					}
				}
			}

		}
		t2 = omp_get_wtime();
		//printf("base:conv3x3x32: %.3f ms\n", (t2 - t1) * 1000);
	}



#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
#undef zq_mm_type
#undef zq_mm_align_size
#undef zq_mm_align_size_mul_2
#undef zq_mm_align_size_mul_3
#undef zq_mm_align_size_mul_4
#undef zq_mm_align_size_mul_5
#undef zq_mm_align_size_mul_6
#undef zq_mm_align_size_mul_7
#undef zq_mm_align_size_mul_8
#undef zq_final_sum_q

#endif

#if __ARM_NEON
#if __ARM_NEON_FP16
#define zq_mm_load_ps vld1q_f16
#define zq_mm_store_ps vst1q_f16
#define zq_mm_add_ps vadd1q_f16
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps(A,B,C) vfmaq_f16(C,A,B)
#else
#define zq_mm_fmadd_ps(A, B, C) vaddq_f16(vmulq_f16(A, B), C)
#endif
#define zq_mm_mul_ps vmulq_f16
#define zq_mm_setzero_ps vdupq_n_f16(0)
#define zq_mm_type float16x8_t
#define zq_base_type float16_t
#define zq_mm_align_size 8
#define zq_mm_align_size_mul_2 16
#define zq_mm_align_size_mul_3 24
#define zq_mm_align_size_mul_4 32
#define zq_mm_align_size_mul_5 40
#define zq_mm_align_size_mul_6 48
#define zq_mm_align_size_mul_7 56
#define zq_mm_align_size_mul_8 64
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])

	/*if C is not 8 (i.e. 1,2,3,4,5,6,7), the padded channels of filters must be zero*/
	void zq_cnn_conv_no_padding_16f_align128bit_kernel3x3_C8(
		const zq_base_type* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 1,2,3,4,5,6,7,8
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const zq_base_type* filters_data,
		int filter_N,
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 1,2,3,4,5,6,7,8
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
		int out_sliceStep
	)
	{
		zq_mm_type sum;
		ZQ_DECLSPEC_ALIGN32 zq_base_type q[8];

		const zq_base_type* in_slice_ptr;
		const zq_base_type* in_row_ptr;
		const zq_base_type* in_pix_ptr;
		zq_base_type* out_slice_ptr;
		zq_base_type* out_row_ptr;
		zq_base_type* out_pix_ptr;
		zq_base_type* out_c_ptr;

		const zq_base_type* cur_in_row_ptr;
		const zq_base_type* cur_in_pix_ptr;
		const zq_base_type* cur_filter_slice_ptr;
		const zq_base_type* cur_filter_row_ptr;
		const zq_base_type* cur_filter_pix_ptr;

		int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
		int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
		int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
		int out_n, out_h, out_w, out_c;
		double t1, t2;
		t1 = omp_get_wtime();

		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				out_h < out_H;
				out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					out_w < out_W;
					out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
				{
					for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
					{
						sum = zq_mm_setzero_ps();
						cur_in_row_ptr = in_pix_ptr;	cur_filter_row_ptr = cur_filter_slice_ptr;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						zq_mm_store_ps(q, sum);
						*out_c_ptr = zq_final_sum_q;
					}
				}
			}

		}
		t2 = omp_get_wtime();
		//printf("base:conv3x3x8: %.3f ms\n", (t2 - t1) * 1000);

	}


	/*if C is not 16 (i.e. 9,10,11,12,13,14,15), the padded channels of filters must be zero*/
	void zq_cnn_conv_no_padding_16f_align128bit_kernel3x3_C16(
		const zq_base_type* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 9,10,11,12,13,14,15,16
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const zq_base_type* filters_data,
		int filter_N,
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 9,10,11,12,13,14,15,16
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
		int out_sliceStep
	)
	{
		zq_mm_type sum;
		ZQ_DECLSPEC_ALIGN32 zq_base_type q[8];

		const zq_base_type* in_slice_ptr;
		const zq_base_type* in_row_ptr;
		const zq_base_type* in_pix_ptr;
		zq_base_type* out_slice_ptr;
		zq_base_type* out_row_ptr;
		zq_base_type* out_pix_ptr;
		zq_base_type* out_c_ptr;

		const zq_base_type* cur_in_row_ptr;
		const zq_base_type* cur_in_pix_ptr;
		const zq_base_type* cur_filter_slice_ptr;
		const zq_base_type* cur_filter_row_ptr;
		const zq_base_type* cur_filter_pix_ptr;

		int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
		int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
		int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
		int out_n, out_h, out_w, out_c;
		double t1, t2;
		t1 = omp_get_wtime();

		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				out_h < out_H;
				out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					out_w < out_W;
					out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
				{
					for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
					{
						sum = zq_mm_setzero_ps();
						cur_in_row_ptr = in_pix_ptr;	cur_filter_row_ptr = cur_filter_slice_ptr;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						zq_mm_store_ps(q, sum);
						*out_c_ptr = zq_final_sum_q;
					}
				}
			}

		}
		t2 = omp_get_wtime();
		//printf("base:conv3x3x16: %.3f ms\n", (t2 - t1) * 1000);
	}


	/*if C is not 24 (i.e. 17,18,19,20,21,22,23), the padded channels of filters must be zero*/
	void zq_cnn_conv_no_padding_16f_align128bit_kernel3x3_C24(
		const zq_base_type* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 17,18,19,20,21,22,23,24
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const zq_base_type* filters_data,
		int filter_N,
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 17,18,19,20,21,22,23,24
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
		int out_sliceStep
	)
	{
		zq_mm_type sum;
		ZQ_DECLSPEC_ALIGN32 zq_base_type q[8];

		const zq_base_type* in_slice_ptr;
		const zq_base_type* in_row_ptr;
		const zq_base_type* in_pix_ptr;
		zq_base_type* out_slice_ptr;
		zq_base_type* out_row_ptr;
		zq_base_type* out_pix_ptr;
		zq_base_type* out_c_ptr;

		const zq_base_type* cur_in_row_ptr;
		const zq_base_type* cur_in_pix_ptr;
		const zq_base_type* cur_filter_slice_ptr;
		const zq_base_type* cur_filter_row_ptr;
		const zq_base_type* cur_filter_pix_ptr;

		int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
		int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
		int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
		int out_n, out_h, out_w, out_c;
		double t1, t2;
		t1 = omp_get_wtime();

		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				out_h < out_H;
				out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					out_w < out_W;
					out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
				{
					for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
					{
						sum = zq_mm_setzero_ps();
						cur_in_row_ptr = in_pix_ptr;	cur_filter_row_ptr = cur_filter_slice_ptr;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						zq_mm_store_ps(q, sum);
						*out_c_ptr = zq_final_sum_q;
					}
				}
			}

		}
		t2 = omp_get_wtime();
		//printf("base:conv3x3x24: %.3f ms\n", (t2 - t1) * 1000);
	}


	/*if C is not 32 (i.e. 25,26,27,28,29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_conv_no_padding_16f_align128bit_kernel3x3_C32(
		const zq_base_type* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 25,26,27,28,29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const zq_base_type* filters_data,
		int filter_N,
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 25,26,27,28,29,30,31,32
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
		int out_sliceStep
	)
	{
		zq_mm_type sum;
		ZQ_DECLSPEC_ALIGN32 zq_base_type q[8];

		const zq_base_type* in_slice_ptr;
		const zq_base_type* in_row_ptr;
		const zq_base_type* in_pix_ptr;
		zq_base_type* out_slice_ptr;
		zq_base_type* out_row_ptr;
		zq_base_type* out_pix_ptr;
		zq_base_type* out_c_ptr;

		const zq_base_type* cur_in_row_ptr;
		const zq_base_type* cur_in_pix_ptr;
		const zq_base_type* cur_filter_slice_ptr;
		const zq_base_type* cur_filter_row_ptr;
		const zq_base_type* cur_filter_pix_ptr;

		int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
		int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
		int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
		int out_n, out_h, out_w, out_c;
		double t1, t2;
		t1 = omp_get_wtime();

		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				out_h < out_H;
				out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					out_w < out_W;
					out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
				{
					for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
					{
						sum = zq_mm_setzero_ps();
						cur_in_row_ptr = in_pix_ptr;	cur_filter_row_ptr = cur_filter_slice_ptr;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_3), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_3), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_3), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_3), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_3), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_3), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_3), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_3), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_2), sum);
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size_mul_3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size_mul_3), sum);
						zq_mm_store_ps(q, sum);
						*out_c_ptr = zq_final_sum_q;
					}
				}
			}

		}
		t2 = omp_get_wtime();
		//printf("base:conv3x3x32: %.3f ms\n", (t2 - t1) * 1000);
	}



#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#undef zq_mm_align_size_mul_2
#undef zq_mm_align_size_mul_3
#undef zq_mm_align_size_mul_4
#undef zq_mm_align_size_mul_5
#undef zq_mm_align_size_mul_6
#undef zq_mm_align_size_mul_7
#undef zq_mm_align_size_mul_8
#undef zq_final_sum_q
#endif
#endif

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif