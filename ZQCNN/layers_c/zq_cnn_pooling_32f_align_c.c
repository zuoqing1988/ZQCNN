#include <stdlib.h>
#include <float.h>
#include <stdio.h>
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
#define zq_cnn_maxpooling_nopadding_suredivided_kernel2x2 zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel2x2
#define zq_cnn_maxpooling_nopadding_suredivided_kernel3x3 zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel3x3
#define zq_cnn_maxpooling_nopadding_suredivided_kernel5x5 zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel5x5
#define zq_cnn_maxpooling_nopadding_suredivided_general zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_general
#define zq_cnn_maxpooling_nopadding_nodivided_general zq_cnn_maxpooling_nopadding_nodivided_32f_align128bit_general
#define zq_cnn_avgpooling_nopadding_suredivided_kernel2x2 zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_kernel2x2
#define zq_cnn_avgpooling_nopadding_suredivided_kernel3x3 zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_kernel3x3
#define zq_cnn_avgpooling_nopadding_suredivided_kernel5x5 zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_kernel5x5
#define zq_cnn_avgpooling_nopadding_suredivided_general zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_general
#define zq_cnn_avgpooling_nopadding_nodivided_general zq_cnn_avgpooling_nopadding_nodivided_32f_align128bit_general
#define zq_mm_load_ps vld1q_f32
#define zq_mm_store_ps vst1q_f32
#define zq_mm_max_ps vmaxq_f32
#define zq_mm_add_ps vaddq_f32
#define zq_mm_mul_ps vmulq_f32
#define zq_mm_set1_ps vdupq_n_f32
#define zq_mm_type float32x4_t
#define zq_base_type float
#define zq_mm_align_size 4

#include "zq_cnn_pooling_32f_align_c_raw.h"


#undef zq_cnn_maxpooling_nopadding_suredivided_kernel2x2
#undef zq_cnn_maxpooling_nopadding_suredivided_kernel3x3
#undef zq_cnn_maxpooling_nopadding_suredivided_kernel5x5
#undef zq_cnn_maxpooling_nopadding_suredivided_general
#undef zq_cnn_maxpooling_nopadding_nodivided_general
#undef zq_cnn_avgpooling_nopadding_suredivided_kernel2x2
#undef zq_cnn_avgpooling_nopadding_suredivided_kernel3x3
#undef zq_cnn_avgpooling_nopadding_suredivided_kernel5x5
#undef zq_cnn_avgpooling_nopadding_suredivided_general
#undef zq_cnn_avgpooling_nopadding_nodivided_general
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_max_ps
#undef zq_mm_add_ps
#undef zq_mm_mul_ps
#undef zq_mm_set1_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size

#if __ARM_NEON_FP16
#define zq_cnn_maxpooling_nopadding_suredivided_kernel2x2 zq_cnn_maxpooling_nopadding_suredivided_16f_align128bit_kernel2x2
#define zq_cnn_maxpooling_nopadding_suredivided_kernel3x3 zq_cnn_maxpooling_nopadding_suredivided_16f_align128bit_kernel3x3
#define zq_cnn_maxpooling_nopadding_suredivided_kernel5x5 zq_cnn_maxpooling_nopadding_suredivided_16f_align128bit_kernel5x5
#define zq_cnn_maxpooling_nopadding_suredivided_general zq_cnn_maxpooling_nopadding_suredivided_16f_align128bit_general
#define zq_cnn_maxpooling_nopadding_nodivided_general zq_cnn_maxpooling_nopadding_nodivided_16f_align128bit_general
#define zq_cnn_avgpooling_nopadding_suredivided_kernel2x2 zq_cnn_avgpooling_nopadding_suredivided_16f_align128bit_kernel2x2
#define zq_cnn_avgpooling_nopadding_suredivided_kernel3x3 zq_cnn_avgpooling_nopadding_suredivided_16f_align128bit_kernel3x3
#define zq_cnn_avgpooling_nopadding_suredivided_kernel5x5 zq_cnn_avgpooling_nopadding_suredivided_16f_align128bit_kernel5x5
#define zq_cnn_avgpooling_nopadding_suredivided_general zq_cnn_avgpooling_nopadding_suredivided_16f_align128bit_general
#define zq_cnn_avgpooling_nopadding_nodivided_general zq_cnn_avgpooling_nopadding_nodivided_16f_align128bit_general
#define zq_mm_load_ps vld1q_f16
#define zq_mm_store_ps vst1q_f16
#define zq_mm_max_ps vmaxq_f16
#define zq_mm_add_ps vaddq_f16
#define zq_mm_mul_ps vmulq_f16
#define zq_mm_set1_ps vdupq_n_f16
#define zq_mm_type float16x8_t
#define zq_base_type float16_t
#define zq_mm_align_size 8

#include "zq_cnn_pooling_32f_align_c_raw.h"


#undef zq_cnn_maxpooling_nopadding_suredivided_kernel2x2
#undef zq_cnn_maxpooling_nopadding_suredivided_kernel3x3
#undef zq_cnn_maxpooling_nopadding_suredivided_kernel5x5
#undef zq_cnn_maxpooling_nopadding_suredivided_general
#undef zq_cnn_maxpooling_nopadding_nodivided_general
#undef zq_cnn_avgpooling_nopadding_suredivided_kernel2x2
#undef zq_cnn_avgpooling_nopadding_suredivided_kernel3x3
#undef zq_cnn_avgpooling_nopadding_suredivided_kernel5x5
#undef zq_cnn_avgpooling_nopadding_suredivided_general
#undef zq_cnn_avgpooling_nopadding_nodivided_general
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_max_ps
#undef zq_mm_add_ps
#undef zq_mm_mul_ps
#undef zq_mm_set1_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#endif//__ARM_NEON_FP16

#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#define zq_cnn_maxpooling_nopadding_suredivided_kernel2x2 zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel2x2
#define zq_cnn_maxpooling_nopadding_suredivided_kernel3x3 zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel3x3
#define zq_cnn_maxpooling_nopadding_suredivided_kernel5x5 zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel5x5
#define zq_cnn_maxpooling_nopadding_suredivided_general zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_general
#define zq_cnn_maxpooling_nopadding_nodivided_general zq_cnn_maxpooling_nopadding_nodivided_32f_align128bit_general
#define zq_cnn_avgpooling_nopadding_suredivided_kernel2x2 zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_kernel2x2
#define zq_cnn_avgpooling_nopadding_suredivided_kernel3x3 zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_kernel3x3
#define zq_cnn_avgpooling_nopadding_suredivided_kernel5x5 zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_kernel5x5
#define zq_cnn_avgpooling_nopadding_suredivided_general zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_general
#define zq_cnn_avgpooling_nopadding_nodivided_general zq_cnn_avgpooling_nopadding_nodivided_32f_align128bit_general
#define zq_mm_load_ps _mm_load_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_max_ps _mm_max_ps
#define zq_mm_add_ps _mm_add_ps
#define zq_mm_mul_ps _mm_mul_ps
#define zq_mm_set1_ps _mm_set1_ps
#define zq_mm_type __m128
#define zq_base_type float
#define zq_mm_align_size 4

#include "zq_cnn_pooling_32f_align_c_raw.h"


#undef zq_cnn_maxpooling_nopadding_suredivided_kernel2x2
#undef zq_cnn_maxpooling_nopadding_suredivided_kernel3x3
#undef zq_cnn_maxpooling_nopadding_suredivided_kernel5x5
#undef zq_cnn_maxpooling_nopadding_suredivided_general
#undef zq_cnn_maxpooling_nopadding_nodivided_general
#undef zq_cnn_avgpooling_nopadding_suredivided_kernel2x2
#undef zq_cnn_avgpooling_nopadding_suredivided_kernel3x3
#undef zq_cnn_avgpooling_nopadding_suredivided_kernel5x5
#undef zq_cnn_avgpooling_nopadding_suredivided_general
#undef zq_cnn_avgpooling_nopadding_nodivided_general
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_max_ps
#undef zq_mm_add_ps
#undef zq_mm_mul_ps
#undef zq_mm_set1_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#define zq_cnn_maxpooling_nopadding_suredivided_kernel2x2 zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_kernel2x2
#define zq_cnn_maxpooling_nopadding_suredivided_kernel3x3 zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_kernel3x3
#define zq_cnn_maxpooling_nopadding_suredivided_kernel5x5 zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_kernel5x5
#define zq_cnn_maxpooling_nopadding_suredivided_general zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_general
#define zq_cnn_maxpooling_nopadding_nodivided_general zq_cnn_maxpooling_nopadding_nodivided_32f_align256bit_general
#define zq_cnn_avgpooling_nopadding_suredivided_kernel2x2 zq_cnn_avgpooling_nopadding_suredivided_32f_align256bit_kernel2x2
#define zq_cnn_avgpooling_nopadding_suredivided_kernel3x3 zq_cnn_avgpooling_nopadding_suredivided_32f_align256bit_kernel3x3
#define zq_cnn_avgpooling_nopadding_suredivided_kernel5x5 zq_cnn_avgpooling_nopadding_suredivided_32f_align256bit_kernel5x5
#define zq_cnn_avgpooling_nopadding_suredivided_general zq_cnn_avgpooling_nopadding_suredivided_32f_align256bit_general
#define zq_cnn_avgpooling_nopadding_nodivided_general zq_cnn_avgpooling_nopadding_nodivided_32f_align256bit_general
#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_max_ps _mm256_max_ps
#define zq_mm_add_ps _mm256_add_ps
#define zq_mm_mul_ps _mm256_mul_ps
#define zq_mm_set1_ps _mm256_set1_ps
#define zq_mm_type __m256
#define zq_base_type float
#define zq_mm_align_size 8

#include "zq_cnn_pooling_32f_align_c_raw.h"

#undef zq_cnn_maxpooling_nopadding_suredivided_kernel2x2
#undef zq_cnn_maxpooling_nopadding_suredivided_kernel3x3
#undef zq_cnn_maxpooling_nopadding_suredivided_kernel5x5
#undef zq_cnn_maxpooling_nopadding_suredivided_general
#undef zq_cnn_maxpooling_nopadding_nodivided_general
#undef zq_cnn_avgpooling_nopadding_suredivided_kernel2x2
#undef zq_cnn_avgpooling_nopadding_suredivided_kernel3x3
#undef zq_cnn_avgpooling_nopadding_suredivided_kernel5x5
#undef zq_cnn_avgpooling_nopadding_suredivided_general
#undef zq_cnn_avgpooling_nopadding_nodivided_general
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_max_ps
#undef zq_mm_add_ps
#undef zq_mm_mul_ps
#undef zq_mm_set1_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#endif
#endif //__ARM_NEON

void zq_cnn_maxpooling_nopadding_32f_align0_general(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_alignPixelStep,
	int in_widthStep,
	int in_SliceStep,
	int kernel_H,
	int kernel_W,
	int stride_H,
	int stride_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
	int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
	int out_C,	// must be filter_N
	int out_alignPixelStep,
	int out_alignWidthStep,
	int out_alignSliceStep
)
{
	float max_val;
	const float* in_slice_ptr;
	const float* in_row_ptr;
	const float* in_pix_ptr;
	const float* in_c_ptr;
	float* out_slice_ptr;
	float* out_row_ptr;
	float* out_pix_ptr;
	float* out_c_ptr;

	const float* cur_in_row_ptr;
	const float* cur_in_pix_ptr;

	int in_widthStep_mul_strideH = stride_H*in_widthStep;
	int in_pixelStep_mul_strideW = stride_W*in_alignPixelStep;
	int out_n, out_h, out_w, out_c, kh, kw, final_kH, final_kW;

	if ((in_W - kernel_W) % stride_W == 0 && (in_H - kernel_H) % stride_H == 0)
	{
		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_SliceStep, out_slice_ptr += out_alignSliceStep)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				out_h < out_H;
				out_h++, in_row_ptr += in_widthStep_mul_strideH, out_row_ptr += out_alignWidthStep)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					out_w < out_W;
					out_w++, in_pix_ptr += in_pixelStep_mul_strideW, out_pix_ptr += out_alignPixelStep)
				{
					for (out_c = 0, out_c_ptr = out_pix_ptr, in_c_ptr = in_pix_ptr;
						out_c < out_C;
						out_c++, out_c_ptr++, in_c_ptr ++)
					{
						max_val = -FLT_MAX;
						for (kh = 0, cur_in_row_ptr = in_c_ptr;
							kh < kernel_H;
							kh++, cur_in_row_ptr += in_widthStep)
						{
							for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr;
								kw < kernel_W;
								kw++, cur_in_pix_ptr += in_alignPixelStep)
							{
								max_val = __max(max_val, *cur_in_pix_ptr);
							}
						}
						*out_c_ptr = max_val;
					}
				}
			}
		}
	}
	else
	{
		final_kH = __min(kernel_H, in_H - (out_H - 1)*stride_H);
		final_kW = __min(kernel_W, in_W - (out_W - 1)*stride_W);

		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_SliceStep, out_slice_ptr += out_alignSliceStep)
		{
			for (out_c = 0, in_c_ptr = in_slice_ptr, out_c_ptr = out_slice_ptr;
				out_c < out_C;
				out_c++, in_c_ptr ++, out_c_ptr ++)
			{
				for (out_h = 0, out_row_ptr = out_c_ptr, in_row_ptr = in_c_ptr;
					out_h < out_H - 1;
					out_h++, out_row_ptr += out_alignWidthStep, in_row_ptr += in_widthStep_mul_strideH)
				{
					for (out_w = 0, out_pix_ptr = out_row_ptr, in_pix_ptr = in_row_ptr;
						out_w < out_W - 1;
						out_w++, out_pix_ptr += out_alignPixelStep, in_pix_ptr += in_pixelStep_mul_strideW)
					{
						max_val = -FLT_MAX;
						for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < kernel_H; kh++, cur_in_row_ptr += in_widthStep)
						{
							for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < kernel_W; kw++, cur_in_pix_ptr += in_alignPixelStep)
							{
								max_val = __max(max_val, *cur_in_pix_ptr);
							}
						}
						*out_pix_ptr = max_val;
					}


					max_val = -FLT_MAX;
					for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < kernel_H; kh++, cur_in_row_ptr += in_widthStep)
					{
						for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < final_kW; kw++, cur_in_pix_ptr += in_alignPixelStep)
						{
							max_val = __max(max_val, *cur_in_pix_ptr);
						}
					}
					*out_pix_ptr = max_val;
				}

				for (out_w = 0, out_pix_ptr = out_row_ptr, in_pix_ptr = in_row_ptr;
					out_w < out_W - 1;
					out_w++, out_pix_ptr += out_alignPixelStep, in_pix_ptr += in_pixelStep_mul_strideW)
				{
					max_val = -FLT_MAX;
					for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < final_kH; kh++, cur_in_row_ptr += in_widthStep)
					{
						for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < kernel_W; kw++, cur_in_pix_ptr += in_alignPixelStep)
						{
							max_val = __max(max_val, *cur_in_pix_ptr);
						}
					}
					*out_pix_ptr = max_val;
				}


				max_val = -FLT_MAX;
				for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < final_kH; kh++, cur_in_row_ptr += in_widthStep)
				{
					for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < final_kW; kw++, cur_in_pix_ptr += in_alignPixelStep)
					{
						max_val = __max(max_val, *cur_in_pix_ptr);
					}
				}
				*out_pix_ptr = max_val;
			}
		}
	}
}

void zq_cnn_avgpooling_nopadding_32f_align0_general(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_alignPixelStep,
	int in_widthStep,
	int in_SliceStep,
	int kernel_H,
	int kernel_W,
	int stride_H,
	int stride_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
	int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
	int out_C,	// must be filter_N
	int out_alignPixelStep,
	int out_alignWidthStep,
	int out_alignSliceStep
)
{
	float sum_val;
	const float* in_slice_ptr;
	const float* in_row_ptr;
	const float* in_pix_ptr;
	const float* in_c_ptr;
	float* out_slice_ptr;
	float* out_row_ptr;
	float* out_pix_ptr;
	float* out_c_ptr;

	const float* cur_in_row_ptr;
	const float* cur_in_pix_ptr;

	int in_widthStep_mul_strideH = stride_H*in_widthStep;
	int in_pixelStep_mul_strideW = stride_W*in_alignPixelStep;
	int out_n, out_h, out_w, out_c, kh, kw, final_kH, final_kW;
	float scale = 1.0f / (kernel_H*kernel_W);
	if ((in_W - kernel_W) % stride_W == 0 && (in_H - kernel_H) % stride_H == 0)
	{
		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_SliceStep, out_slice_ptr += out_alignSliceStep)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				out_h < out_H;
				out_h++, in_row_ptr += in_widthStep_mul_strideH, out_row_ptr += out_alignWidthStep)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					out_w < out_W;
					out_w++, in_pix_ptr += in_pixelStep_mul_strideW, out_pix_ptr += out_alignPixelStep)
				{
					for (out_c = 0, out_c_ptr = out_pix_ptr, in_c_ptr = in_pix_ptr;
						out_c < out_C;
						out_c++, out_c_ptr++, in_c_ptr++)
					{
						sum_val = 0;
						for (kh = 0, cur_in_row_ptr = in_c_ptr;
							kh < kernel_H;
							kh++, cur_in_row_ptr += in_widthStep)
						{
							for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr;
								kw < kernel_W;
								kw++, cur_in_pix_ptr += in_alignPixelStep)
							{
								sum_val += *cur_in_pix_ptr;
							}
						}
						*out_c_ptr = sum_val*scale;
					}
				}
			}
		}
	}
	else
	{
		final_kH = __min(kernel_H, in_H - (out_H - 1)*stride_H);
		final_kW = __min(kernel_W, in_W - (out_W - 1)*stride_W);
		
		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_SliceStep, out_slice_ptr += out_alignSliceStep)
		{
			for (out_c = 0, in_c_ptr = in_slice_ptr, out_c_ptr = out_slice_ptr;
				out_c < out_C;
				out_c++, in_c_ptr++, out_c_ptr++)
			{
				for (out_h = 0, out_row_ptr = out_c_ptr, in_row_ptr = in_c_ptr;
					out_h < out_H - 1;
					out_h++, out_row_ptr += out_alignWidthStep, in_row_ptr += in_widthStep_mul_strideH)
				{
					for (out_w = 0, out_pix_ptr = out_row_ptr, in_pix_ptr = in_row_ptr;
						out_w < out_W - 1;
						out_w++, out_pix_ptr += out_alignPixelStep, in_pix_ptr += in_pixelStep_mul_strideW)
					{
						sum_val = 0;
						for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < kernel_H; kh++, cur_in_row_ptr += in_widthStep)
						{
							for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < kernel_W; kw++, cur_in_pix_ptr += in_alignPixelStep)
							{
								sum_val += *cur_in_pix_ptr;
							}
						}
						*out_pix_ptr = sum_val*scale;
					}


					sum_val = 0;
					for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < kernel_H; kh++, cur_in_row_ptr += in_widthStep)
					{
						for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < final_kW; kw++, cur_in_pix_ptr += in_alignPixelStep)
						{
							sum_val += *cur_in_pix_ptr;
						}
					}
					*out_pix_ptr = sum_val/(kernel_H*final_kW);
				}

				for (out_w = 0, out_pix_ptr = out_row_ptr, in_pix_ptr = in_row_ptr;
					out_w < out_W - 1;
					out_w++, out_pix_ptr += out_alignPixelStep, in_pix_ptr += in_pixelStep_mul_strideW)
				{
					sum_val = 0;
					for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < final_kH; kh++, cur_in_row_ptr += in_widthStep)
					{
						for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < kernel_W; kw++, cur_in_pix_ptr += in_alignPixelStep)
						{
							sum_val += *cur_in_pix_ptr;
						}
					}
					*out_pix_ptr = sum_val/(final_kH*kernel_W);
				}


				sum_val = 0;
				for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < final_kH; kh++, cur_in_row_ptr += in_widthStep)
				{
					for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < final_kW; kw++, cur_in_pix_ptr += in_alignPixelStep)
					{
						sum_val += *cur_in_pix_ptr;
					}
				}
				*out_pix_ptr = sum_val/(final_kH*final_kW);
			}
		}
	}
}

#if __ARM_NEON
#if __ARM_NEON_FP16
#define zq_base_type float16_t
void zq_cnn_maxpooling_nopadding_16f_align0_general(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_alignPixelStep,
	int in_widthStep,
	int in_SliceStep,
	int kernel_H,
	int kernel_W,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
	int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
	int out_C,	// must be filter_N
	int out_alignPixelStep,
	int out_alignWidthStep,
	int out_alignSliceStep
)
{
	zq_base_type max_val;
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	const zq_base_type* in_c_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr;

	const zq_base_type* cur_in_row_ptr;
	const zq_base_type* cur_in_pix_ptr;

	int in_widthStep_mul_strideH = stride_H*in_widthStep;
	int in_pixelStep_mul_strideW = stride_W*in_alignPixelStep;
	int out_n, out_h, out_w, out_c, kh, kw, final_kH, final_kW;

	if ((in_W - kernel_W) % stride_W == 0 && (in_H - kernel_H) % stride_H == 0)
	{
		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_SliceStep, out_slice_ptr += out_alignSliceStep)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				out_h < out_H;
				out_h++, in_row_ptr += in_widthStep_mul_strideH, out_row_ptr += out_alignWidthStep)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					out_w < out_W;
					out_w++, in_pix_ptr += in_pixelStep_mul_strideW, out_pix_ptr += out_alignPixelStep)
				{
					for (out_c = 0, out_c_ptr = out_pix_ptr, in_c_ptr = in_pix_ptr;
						out_c < out_C;
						out_c++, out_c_ptr++, in_c_ptr++)
					{
						max_val = -FLT_MAX;
						for (kh = 0, cur_in_row_ptr = in_c_ptr;
							kh < kernel_H;
							kh++, cur_in_row_ptr += in_widthStep)
						{
							for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr;
								kw < kernel_W;
								kw++, cur_in_pix_ptr += in_alignPixelStep)
							{
								max_val = __max(max_val, *cur_in_pix_ptr);
							}
						}
						*out_c_ptr = max_val;
					}
				}
			}
		}
	}
	else
	{
		final_kH = __min(kernel_H, in_H - (out_H - 1)*stride_H);
		final_kW = __min(kernel_W, in_W - (out_W - 1)*stride_W);

		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_SliceStep, out_slice_ptr += out_alignSliceStep)
		{
			for (out_c = 0, in_c_ptr = in_slice_ptr, out_c_ptr = out_slice_ptr;
				out_c < out_C;
				out_c++, in_c_ptr++, out_c_ptr++)
			{
				for (out_h = 0, out_row_ptr = out_c_ptr, in_row_ptr = in_c_ptr;
					out_h < out_H - 1;
					out_h++, out_row_ptr += out_alignWidthStep, in_row_ptr += in_widthStep_mul_strideH)
				{
					for (out_w = 0, out_pix_ptr = out_row_ptr, in_pix_ptr = in_row_ptr;
						out_w < out_W - 1;
						out_w++, out_pix_ptr += out_alignPixelStep, in_pix_ptr += in_pixelStep_mul_strideW)
					{
						max_val = -FLT_MAX;
						for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < kernel_H; kh++, cur_in_row_ptr += in_widthStep)
						{
							for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < kernel_W; kw++, cur_in_pix_ptr += in_alignPixelStep)
							{
								max_val = __max(max_val, *cur_in_pix_ptr);
							}
						}
						*out_pix_ptr = max_val;
					}


					max_val = -FLT_MAX;
					for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < kernel_H; kh++, cur_in_row_ptr += in_widthStep)
					{
						for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < final_kW; kw++, cur_in_pix_ptr += in_alignPixelStep)
						{
							max_val = __max(max_val, *cur_in_pix_ptr);
						}
					}
					*out_pix_ptr = max_val;
				}

				for (out_w = 0, out_pix_ptr = out_row_ptr, in_pix_ptr = in_row_ptr;
					out_w < out_W - 1;
					out_w++, out_pix_ptr += out_alignPixelStep, in_pix_ptr += in_pixelStep_mul_strideW)
				{
					max_val = -FLT_MAX;
					for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < final_kH; kh++, cur_in_row_ptr += in_widthStep)
					{
						for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < kernel_W; kw++, cur_in_pix_ptr += in_alignPixelStep)
						{
							max_val = __max(max_val, *cur_in_pix_ptr);
						}
					}
					*out_pix_ptr = max_val;
				}


				max_val = -FLT_MAX;
				for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < final_kH; kh++, cur_in_row_ptr += in_widthStep)
				{
					for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < final_kW; kw++, cur_in_pix_ptr += in_alignPixelStep)
					{
						max_val = __max(max_val, *cur_in_pix_ptr);
					}
				}
				*out_pix_ptr = max_val;
			}
		}
	}
}

void zq_cnn_avgpooling_nopadding_16f_align0_general(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_alignPixelStep,
	int in_widthStep,
	int in_SliceStep,
	int kernel_H,
	int kernel_W,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
	int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
	int out_C,	// must be filter_N
	int out_alignPixelStep,
	int out_alignWidthStep,
	int out_alignSliceStep
)
{
	zq_base_type sum_val;
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	const zq_base_type* in_c_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr;

	const zq_base_type* cur_in_row_ptr;
	const zq_base_type* cur_in_pix_ptr;

	int in_widthStep_mul_strideH = stride_H*in_widthStep;
	int in_pixelStep_mul_strideW = stride_W*in_alignPixelStep;
	int out_n, out_h, out_w, out_c, kh, kw, final_kH, final_kW;
	zq_base_type scale = 1.0f / (kernel_H*kernel_W);
	if ((in_W - kernel_W) % stride_W == 0 && (in_H - kernel_H) % stride_H == 0)
	{
		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_SliceStep, out_slice_ptr += out_alignSliceStep)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				out_h < out_H;
				out_h++, in_row_ptr += in_widthStep_mul_strideH, out_row_ptr += out_alignWidthStep)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					out_w < out_W;
					out_w++, in_pix_ptr += in_pixelStep_mul_strideW, out_pix_ptr += out_alignPixelStep)
				{
					for (out_c = 0, out_c_ptr = out_pix_ptr, in_c_ptr = in_pix_ptr;
						out_c < out_C;
						out_c++, out_c_ptr++, in_c_ptr++)
					{
						sum_val = 0;
						for (kh = 0, cur_in_row_ptr = in_c_ptr;
							kh < kernel_H;
							kh++, cur_in_row_ptr += in_widthStep)
						{
							for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr;
								kw < kernel_W;
								kw++, cur_in_pix_ptr += in_alignPixelStep)
							{
								sum_val += *cur_in_pix_ptr;
							}
						}
						*out_c_ptr = sum_val*scale;
					}
				}
			}
		}
	}
	else
	{
		final_kH = __min(kernel_H, in_H - (out_H - 1)*stride_H);
		final_kW = __min(kernel_W, in_W - (out_W - 1)*stride_W);

		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_SliceStep, out_slice_ptr += out_alignSliceStep)
		{
			for (out_c = 0, in_c_ptr = in_slice_ptr, out_c_ptr = out_slice_ptr;
				out_c < out_C;
				out_c++, in_c_ptr++, out_c_ptr++)
			{
				for (out_h = 0, out_row_ptr = out_c_ptr, in_row_ptr = in_c_ptr;
					out_h < out_H - 1;
					out_h++, out_row_ptr += out_alignWidthStep, in_row_ptr += in_widthStep_mul_strideH)
				{
					for (out_w = 0, out_pix_ptr = out_row_ptr, in_pix_ptr = in_row_ptr;
						out_w < out_W - 1;
						out_w++, out_pix_ptr += out_alignPixelStep, in_pix_ptr += in_pixelStep_mul_strideW)
					{
						sum_val = 0;
						for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < kernel_H; kh++, cur_in_row_ptr += in_widthStep)
						{
							for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < kernel_W; kw++, cur_in_pix_ptr += in_alignPixelStep)
							{
								sum_val += *cur_in_pix_ptr;
							}
						}
						*out_pix_ptr = sum_val*scale;
					}


					sum_val = 0;
					for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < kernel_H; kh++, cur_in_row_ptr += in_widthStep)
					{
						for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < final_kW; kw++, cur_in_pix_ptr += in_alignPixelStep)
						{
							sum_val += *cur_in_pix_ptr;
						}
					}
					*out_pix_ptr = sum_val / (kernel_H*final_kW);
				}

				for (out_w = 0, out_pix_ptr = out_row_ptr, in_pix_ptr = in_row_ptr;
					out_w < out_W - 1;
					out_w++, out_pix_ptr += out_alignPixelStep, in_pix_ptr += in_pixelStep_mul_strideW)
				{
					sum_val = 0;
					for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < final_kH; kh++, cur_in_row_ptr += in_widthStep)
					{
						for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < kernel_W; kw++, cur_in_pix_ptr += in_alignPixelStep)
						{
							sum_val += *cur_in_pix_ptr;
						}
					}
					*out_pix_ptr = sum_val / (final_kH*kernel_W);
				}


				sum_val = 0;
				for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < final_kH; kh++, cur_in_row_ptr += in_widthStep)
				{
					for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < final_kW; kw++, cur_in_pix_ptr += in_alignPixelStep)
					{
						sum_val += *cur_in_pix_ptr;
					}
				}
				*out_pix_ptr = sum_val / (final_kH*final_kW);
			}
		}
	}
}
#undef zq_base_type
#endif//__ARM_NEON_FP16
#endif//__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif