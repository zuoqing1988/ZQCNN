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
#endif//__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

#if __ARM_NEON

#define zq_mm_load_ps vld1q_f32
#define zq_mm_store_ps vst1q_f32
#define zq_mm_set1_ps vdupq_n_f32
#define zq_mm_setzero_ps() vdupq_n_f32(0)
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps(A, B, C) vfmaq_f32(C, A, B)
#else
#define zq_mm_fmadd_ps(A, B, C) vaddq_f32(vmulq_f32(A, B), C)
#endif
#define zq_mm_add_ps vaddq_f32
#define zq_mm_sub_ps vsubq_f32
#define zq_mm_mul_ps vmulq_f32
#define zq_mm_max_ps vmaxq_f32
#define zq_mm_min_ps vminq_f32
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
#define zq_mm_align_size32 128

#define zq_cnn_prelu_32f_align zq_cnn_prelu_32f_align128bit
#define zq_cnn_prelu_32f_align_sure_slope_lessthan1 zq_cnn_prelu_32f_align128bit_sure_slope_lessthan1

#define WITH_BIAS 0
#include "zq_cnn_prelu_32f_align_c_raw.h"
#undef WITH_BIAS 

#undef zq_cnn_prelu_32f_align
#undef zq_cnn_prelu_32f_align_sure_slope_lessthan1

#define zq_cnn_prelu_32f_align zq_cnn_addbias_prelu_32f_align128bit
#define zq_cnn_prelu_32f_align_sure_slope_lessthan1 zq_cnn_addbias_prelu_32f_align128bit_sure_slope_lessthan1

#define WITH_BIAS 1
#include "zq_cnn_prelu_32f_align_c_raw.h"
#undef WITH_BIAS 

#undef zq_cnn_prelu_32f_align
#undef zq_cnn_prelu_32f_align_sure_slope_lessthan1

#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_set1_ps
#undef zq_mm_setzero_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_add_ps
#undef zq_mm_sub_ps
#undef zq_mm_mul_ps
#undef zq_mm_max_ps
#undef zq_mm_min_ps
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
#undef zq_mm_align_size32

#if __ARM_NEON_FP16
#define zq_mm_load_ps vld1q_f16
#define zq_mm_store_ps vst1q_f16
#define zq_mm_set1_ps vdupq_n_f16
#define zq_mm_setzero_ps() vdupq_n_f16(0)
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps(A, B, C) vfmaq_f16(C, A, B)
#else
#define zq_mm_fmadd_ps(A, B, C) vaddq_f16(vmulq_f16(A, B), C)
#endif
#define zq_mm_add_ps vaddq_f16
#define zq_mm_sub_ps vsubq_f16
#define zq_mm_mul_ps vmulq_f16
#define zq_mm_max_ps vmaxq_f16
#define zq_mm_min_ps vminq_f16
#define zq_mm_type float16x8_t
#define zq_base_type float16_t
#define zq_mm_align_size 8
#define zq_mm_align_size2 16
#define zq_mm_align_size3 24
#define zq_mm_align_size4 32
#define zq_mm_align_size5 40
#define zq_mm_align_size6 48
#define zq_mm_align_size7 56
#define zq_mm_align_size8 64
#define zq_mm_align_size16 128
#define zq_mm_align_size32 256

#define zq_cnn_prelu_32f_align zq_cnn_prelu_16f_align128bit
#define zq_cnn_prelu_32f_align_sure_slope_lessthan1 zq_cnn_prelu_16f_align128bit_sure_slope_lessthan1

#define WITH_BIAS 0
#include "zq_cnn_prelu_32f_align_c_raw.h"
#undef WITH_BIAS

#undef zq_cnn_prelu_32f_align
#undef zq_cnn_prelu_32f_align_sure_slope_lessthan1

#define zq_cnn_prelu_32f_align zq_cnn_addbias_prelu_16f_align128bit
#define zq_cnn_prelu_32f_align_sure_slope_lessthan1 zq_cnn_addbias_prelu_16f_align128bit_sure_slope_lessthan1

#define WITH_BIAS 1
#include "zq_cnn_prelu_32f_align_c_raw.h"
#undef WITH_BIAS

#undef zq_cnn_prelu_32f_align
#undef zq_cnn_prelu_32f_align_sure_slope_lessthan1

#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_set1_ps
#undef zq_mm_setzero_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_add_ps
#undef zq_mm_sub_ps
#undef zq_mm_mul_ps
#undef zq_mm_max_ps
#undef zq_mm_min_ps
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
#undef zq_mm_align_size32
#endif//__ARM_NEON_FP16

#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#define zq_mm_load_ps _mm_load_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_set1_ps _mm_set1_ps
#define zq_mm_setzero_ps _mm_setzero_ps
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps _mm_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm_add_ps(_mm_mul_ps(A, B), C)
#endif
#define zq_mm_add_ps _mm_add_ps
#define zq_mm_sub_ps _mm_sub_ps
#define zq_mm_mul_ps _mm_mul_ps
#define zq_mm_max_ps _mm_max_ps
#define zq_mm_min_ps _mm_min_ps
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
#define zq_mm_align_size32 128

#define zq_cnn_prelu_32f_align zq_cnn_prelu_32f_align128bit
#define zq_cnn_prelu_32f_align_sure_slope_lessthan1 zq_cnn_prelu_32f_align128bit_sure_slope_lessthan1

#define WITH_BIAS 0
#include "zq_cnn_prelu_32f_align_c_raw.h"
#undef WITH_BIAS

#undef zq_cnn_prelu_32f_align
#undef zq_cnn_prelu_32f_align_sure_slope_lessthan1

#define zq_cnn_prelu_32f_align zq_cnn_addbias_prelu_32f_align128bit
#define zq_cnn_prelu_32f_align_sure_slope_lessthan1 zq_cnn_addbias_prelu_32f_align128bit_sure_slope_lessthan1

#define WITH_BIAS 1
#include "zq_cnn_prelu_32f_align_c_raw.h"
#undef WITH_BIAS

#undef zq_cnn_prelu_32f_align
#undef zq_cnn_prelu_32f_align_sure_slope_lessthan1

#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_set1_ps
#undef zq_mm_setzero_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_add_ps
#undef zq_mm_sub_ps
#undef zq_mm_mul_ps
#undef zq_mm_max_ps
#undef zq_mm_min_ps
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
#undef zq_mm_align_size32
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_set1_ps _mm256_set1_ps
#define zq_mm_setzero_ps _mm256_setzero_ps
#if ZQ_CNN_USE_FMADD256
#define zq_mm_fmadd_ps _mm256_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#endif
#define zq_mm_add_ps _mm256_add_ps
#define zq_mm_sub_ps _mm256_sub_ps
#define zq_mm_mul_ps _mm256_mul_ps
#define zq_mm_max_ps _mm256_max_ps
#define zq_mm_min_ps _mm256_min_ps
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
#define zq_mm_align_size16 128
#define zq_mm_align_size32 256

#define zq_cnn_prelu_32f_align zq_cnn_prelu_32f_align256bit
#define zq_cnn_prelu_32f_align_sure_slope_lessthan1 zq_cnn_prelu_32f_align256bit_sure_slope_lessthan1

#define WITH_BIAS 0
#include "zq_cnn_prelu_32f_align_c_raw.h"
#undef WITH_BIAS

#undef zq_cnn_prelu_32f_align
#undef zq_cnn_prelu_32f_align_sure_slope_lessthan1

#define zq_cnn_prelu_32f_align zq_cnn_addbias_prelu_32f_align256bit
#define zq_cnn_prelu_32f_align_sure_slope_lessthan1 zq_cnn_addbias_prelu_32f_align256bit_sure_slope_lessthan1

#define WITH_BIAS 1
#include "zq_cnn_prelu_32f_align_c_raw.h"
#undef WITH_BIAS

#undef zq_cnn_prelu_32f_align
#undef zq_cnn_prelu_32f_align_sure_slope_lessthan1

#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_set1_ps
#undef zq_mm_setzero_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_add_ps
#undef zq_mm_sub_ps
#undef zq_mm_mul_ps
#undef zq_mm_max_ps
#undef zq_mm_min_ps
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
#undef zq_mm_align_size32
#endif
#endif //__ARM_NEON

void zq_cnn_prelu_32f_align0(
	float* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* slope_data
)
{
	float data_v;
	int n, h, w, c;
	float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;

	for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
	{
		for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
		{
			for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
			{
				for (c = 0, c_ptr = pix_ptr; c < in_C; c++, c_ptr++)
				{
					data_v = *c_ptr;
					if (data_v < 0)
						data_v *= slope_data[c];
					*c_ptr = data_v;
				}
			}
		}
	}
}

void zq_cnn_addbias_prelu_32f_align0(
	float* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* bias,
	const float* slope_data
)
{
	float data_v;
	int n, h, w, c;
	float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;

	for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
	{
		for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
		{
			for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
			{
				for (c = 0, c_ptr = pix_ptr; c < in_C; c++, c_ptr++)
				{
					data_v = *c_ptr + bias[c];
					if (data_v < 0)
						data_v *= slope_data[c];
					*c_ptr = data_v;
				}
			}
		}
	}
}

#if __ARM_NEON
#if __ARM_NEON_FP16
#define zq_base_type float16_t
void zq_cnn_prelu_16f_align0(
	zq_base_type* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* slope_data
)
{
	zq_base_type data_v;
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;

	for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
	{
		for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
		{
			for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
			{
				for (c = 0, c_ptr = pix_ptr; c < in_C; c++, c_ptr++)
				{
					data_v = *c_ptr;
					if (data_v < 0)
						data_v *= slope_data[c];
					*c_ptr = data_v;
				}
			}
		}
	}
}

void zq_cnn_addbias_prelu_16f_align0(
	zq_base_type* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* bias,
	const zq_base_type* slope_data
)
{
	zq_base_type data_v;
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;

	for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
	{
		for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
		{
			for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
			{
				for (c = 0, c_ptr = pix_ptr; c < in_C; c++, c_ptr++)
				{
					data_v = *c_ptr + bias[c];
					if (data_v < 0)
						data_v *= slope_data[c];
					*c_ptr = data_v;
				}
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