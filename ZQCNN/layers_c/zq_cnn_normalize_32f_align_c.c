#include <stdio.h>
#include <math.h>
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
#define zq_cnn_normalize_not_across_spatial_32f_align zq_cnn_normalize_not_across_spatial_32f_align128bit
#define zq_cnn_normalize_across_spatial_32f_align zq_cnn_normalize_across_spatial_32f_align128bit
#define zq_mm_load_ps vld1q_f32
#define zq_mm_store_ps vst1q_f32
#define zq_mm_setzero_ps() vdupq_n_f32(0)
#define zq_mm_set1_ps vdupq_n_f32
#define zq_mm_add_ps vaddq_f32
#define zq_mm_mul_ps vmulq_f32
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps(A, B, C) vfmaq_f32(C, A, B)
#else
#define zq_mm_fmadd_ps(A, B, C) vaddq_f32(vmulq_f32(A, B), C)
#endif
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
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3])

#include "zq_cnn_normalize_32f_align_c_raw.h"


#undef zq_cnn_normalize_not_across_spatial_32f_align
#undef zq_cnn_normalize_across_spatial_32f_align
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_setzero_ps
#undef zq_mm_set1_ps
#undef zq_mm_add_ps
#undef zq_mm_mul_ps
#undef zq_mm_fmadd_ps
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
#undef zq_final_sum_q

#if __ARM_NEON_FP16
#define zq_cnn_normalize_not_across_spatial_32f_align zq_cnn_normalize_not_across_spatial_16f_align128bit
#define zq_cnn_normalize_across_spatial_32f_align zq_cnn_normalize_across_spatial_16f_align128bit
#define zq_mm_load_ps vld1q_f16
#define zq_mm_store_ps vst1q_f16
#define zq_mm_setzero_ps() vdupq_n_f16(0)
#define zq_mm_set1_ps vdupq_n_f16
#define zq_mm_add_ps vaddq_f16
#define zq_mm_mul_ps vmulq_f16
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps(A, B, C) vfmaq_f16(C, A, B)
#else
#define zq_mm_fmadd_ps(A, B, C) vaddq_f16(vmulq_f32(A, B), C)
#endif
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
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])

#include "zq_cnn_normalize_32f_align_c_raw.h"


#undef zq_cnn_normalize_not_across_spatial_32f_align
#undef zq_cnn_normalize_across_spatial_32f_align
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_setzero_ps
#undef zq_mm_set1_ps
#undef zq_mm_add_ps
#undef zq_mm_mul_ps
#undef zq_mm_fmadd_ps
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
#undef zq_final_sum_q
#endif//__ARM_NEON_FP16

#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#define zq_cnn_normalize_not_across_spatial_32f_align zq_cnn_normalize_not_across_spatial_32f_align128bit
#define zq_cnn_normalize_across_spatial_32f_align zq_cnn_normalize_across_spatial_32f_align128bit
#define zq_mm_load_ps _mm_load_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_setzero_ps _mm_setzero_ps
#define zq_mm_set1_ps _mm_set1_ps
#define zq_mm_add_ps _mm_add_ps
#define zq_mm_mul_ps _mm_mul_ps
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps _mm_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm_add_ps(_mm_mul_ps(A, B), C)
#endif
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
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3])

#include "zq_cnn_normalize_32f_align_c_raw.h"


#undef zq_cnn_normalize_not_across_spatial_32f_align
#undef zq_cnn_normalize_across_spatial_32f_align
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_setzero_ps
#undef zq_mm_set1_ps
#undef zq_mm_add_ps
#undef zq_mm_mul_ps
#undef zq_mm_fmadd_ps
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
#undef zq_final_sum_q
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#define zq_cnn_normalize_not_across_spatial_32f_align zq_cnn_normalize_not_across_spatial_32f_align256bit
#define zq_cnn_normalize_across_spatial_32f_align zq_cnn_normalize_across_spatial_32f_align256bit
#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_setzero_ps _mm256_setzero_ps
#define zq_mm_set1_ps _mm256_set1_ps
#define zq_mm_add_ps _mm256_add_ps
#define zq_mm_mul_ps _mm256_mul_ps
#if ZQ_CNN_USE_FMADD256
#define zq_mm_fmadd_ps _mm256_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#endif
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
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])


#include "zq_cnn_normalize_32f_align_c_raw.h"

#undef zq_cnn_normalize_not_across_spatial_32f_align
#undef zq_cnn_normalize_across_spatial_32f_align
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_setzero_ps
#undef zq_mm_set1_ps
#undef zq_mm_add_ps
#undef zq_mm_mul_ps
#undef zq_mm_fmadd_ps
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
#undef zq_final_sum_q
#endif
#endif //__ARM_NEON

	void zq_cnn_normalize_32f_align0(
		int across_spatial,
		int channel_shared,
		float* in_tensor4D_data,	// in & out
		const float* scale_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float eps
	)
	{
		float sum_v;
		int n, h, w, c;
		float* slice_ptr, *row_ptr, *pix_ptr;
		int HWC = in_N*in_H*in_C;
		if (across_spatial)
		{
			for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
			{
				sum_v = 0;
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0; c < in_C; c++)
							sum_v += pix_ptr[c] * pix_ptr[c];
					}
				}
				sum_v = 1.0f / sqrt(sum_v+eps);
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						if (channel_shared)
						{
							for (c = 0; c < in_C; c++)
								pix_ptr[c] *= sum_v*scale_data[0];
						}
						else
						{
							for (c = 0; c < in_C; c++)
								pix_ptr[c] *= sum_v*scale_data[c];
						}
					}
				}
			}
		}
		else
		{
			for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						sum_v = 0;
						for (c = 0; c < in_C; c++)
							sum_v += pix_ptr[c] * pix_ptr[c];
						sum_v = 1.0f / sqrt(sum_v);
						if (channel_shared)
						{
							for (c = 0; c < in_C; c++)
								pix_ptr[c] *= sum_v*scale_data[0];
						}
						else
						{
							for (c = 0; c < in_C; c++)
								pix_ptr[c] *= sum_v*scale_data[c];
						}
					}
				}
			}
		}
	}

#if __ARM_NEON
#if __ARM_NEON_FP16
#define zq_base_type float16_t
	void zq_cnn_normalize_32f_align0(
		int across_spatial,
		int channel_shared,
		zq_base_type* in_tensor4D_data,	// in & out
		const zq_base_type* scale_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const zq_base_type eps
	)
	{
		zq_base_type sum_v;
		int n, h, w, c;
		zq_base_type* slice_ptr, *row_ptr, *pix_ptr;
		int HWC = in_N*in_H*in_C;
		if (across_spatial)
		{
			for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
			{
				sum_v = 0;
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0; c < in_C; c++)
							sum_v += pix_ptr[c] * pix_ptr[c];
					}
				}
				sum_v = 1.0f / sqrt(sum_v + eps);
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						if (channel_shared)
						{
							for (c = 0; c < in_C; c++)
								pix_ptr[c] *= sum_v*scale_data[0];
						}
						else
						{
							for (c = 0; c < in_C; c++)
								pix_ptr[c] *= sum_v*scale_data[c];
						}
					}
				}
			}
		}
		else
		{
			for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						sum_v = 0;
						for (c = 0; c < in_C; c++)
							sum_v += pix_ptr[c] * pix_ptr[c];
						sum_v = 1.0f / sqrt(sum_v);
						if (channel_shared)
						{
							for (c = 0; c < in_C; c++)
								pix_ptr[c] *= sum_v*scale_data[0];
						}
						else
						{
							for (c = 0; c < in_C; c++)
								pix_ptr[c] *= sum_v*scale_data[c];
						}
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
