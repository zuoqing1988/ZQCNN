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
#define zq_cnn_dropout_32f_align zq_cnn_dropout_32f_align128bit
#define zq_mm_load_ps vld1q_f32
#define zq_mm_store_ps vst1q_f32
#define zq_mm_mul_ps vmulq_f32
#define zq_mm_set1_ps vdupq_n_f32
#define zq_mm_setzero_ps() vdupq_n_f32(0)
#define zq_mm_type float32x4_t
#define zq_base_type float
#define zq_mm_align_size 4

#include "zq_cnn_dropout_32f_align_c_raw.h"


#undef zq_cnn_dropout_32f_align
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_mul_ps
#undef zq_mm_set1_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size

#if __ARM_NEON_FP16
#define zq_cnn_dropout_32f_align zq_cnn_dropout_16f_align128bit
#define zq_mm_load_ps vld1q_f16
#define zq_mm_store_ps vst1q_f16
#define zq_mm_mul_ps vmulq_f16
#define zq_mm_set1_ps vdupq_n_f16
#define zq_mm_setzero_ps() vdupq_n_f16(0)
#define zq_mm_type float16x8_t
#define zq_base_type float16_t
#define zq_mm_align_size 8

#include "zq_cnn_dropout_32f_align_c_raw.h"


#undef zq_cnn_dropout_32f_align
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_mul_ps
#undef zq_mm_set1_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#endif

#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#define zq_cnn_dropout_32f_align zq_cnn_dropout_32f_align128bit
#define zq_mm_load_ps _mm_load_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_mul_ps _mm_mul_ps
#define zq_mm_set1_ps _mm_set1_ps
#define zq_mm_type __m128
#define zq_base_type float
#define zq_mm_align_size 4

#include "zq_cnn_dropout_32f_align_c_raw.h"


#undef zq_cnn_dropout_32f_align
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_mul_ps
#undef zq_mm_set1_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#define zq_cnn_dropout_32f_align zq_cnn_dropout_32f_align256bit
#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_mul_ps _mm256_mul_ps
#define zq_mm_set1_ps _mm256_set1_ps
#define zq_mm_type __m256
#define zq_base_type float
#define zq_mm_align_size 8

#include "zq_cnn_dropout_32f_align_c_raw.h"


#undef zq_cnn_dropout_32f_align
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_mul_ps
#undef zq_mm_set1_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#endif
#endif //__ARM_NEON


	void zq_cnn_dropout_32f_align0(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float dropout_ratio
	)
	{
		float scale = 1.0f - dropout_ratio;
		int n, h, w, c;
		float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;

		if (scale == 1.0f)
		{
			return;
		}
		else
		{
			for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c++, c_ptr++)
						{
							*c_ptr *= scale;
						}
					}
				}
			}
		}
	}

#if __ARM_NEON
#if __ARM_NEON_FP16
#define zq_base_type float16_t
	void zq_cnn_dropout_16f_align0(
		zq_base_type* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float dropout_ratio
	)
	{
		float scale = 1.0f - dropout_ratio;
		int n, h, w, c;
		zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;

		if (scale == 1.0f)
		{
			return;
		}
		else
		{
			for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c++, c_ptr++)
						{
							*c_ptr *= scale;
						}
					}
				}
			}
		}
	}
#undef zq_base_type
#endif
#endif



#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif