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
#define zq_cnn_addbias_32f_align zq_cnn_addbias_32f_align128bit
#define zq_mm_load_ps vld1q_f32
#define zq_mm_store_ps vst1q_f32
#define zq_mm_add_ps vaddq_f32
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

#include "zq_cnn_addbias_32f_align_c_raw.h"


#undef zq_cnn_addbias_32f_align
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
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
#define zq_cnn_addbias_32f_align zq_cnn_addbias_16f_align128bit
#define zq_mm_load_ps vld1q_f16
#define zq_mm_store_ps vst1q_f16
#define zq_mm_add_ps vaddq_f16
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

#include "zq_cnn_addbias_32f_align_c_raw.h"


#undef zq_cnn_addbias_32f_align
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
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

#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#define zq_cnn_addbias_32f_align zq_cnn_addbias_32f_align128bit
#define zq_mm_load_ps _mm_load_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_add_ps _mm_add_ps
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

#include "zq_cnn_addbias_32f_align_c_raw.h"


#undef zq_cnn_addbias_32f_align
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
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
#define zq_cnn_addbias_32f_align zq_cnn_addbias_32f_align256bit
#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_add_ps _mm256_add_ps
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


#include "zq_cnn_addbias_32f_align_c_raw.h"

#undef zq_cnn_addbias_32f_align
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
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

	void zq_cnn_addbias_32f_align0(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* bias_data,
		int bias_C // must be in_C
	)
	{
		float bias_v;
		int n, h, w, c;
		float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;

		for (c = 0, c_ptr = in_tensor4D_data; c < in_C; c++, c_ptr++)
		{
			bias_v = *(bias_data + c);
			for (n = 0, slice_ptr = c_ptr; n < in_N; n++, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						*pix_ptr += bias_v;
					}
				}
			}
		}
	}

#if __ARM_NEON
#if __ARM_NEON_FP16
#define zq_base_type float16_t

	void zq_cnn_addbias_16f_align0(
		zq_base_type* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const zq_base_type* bias_data,
		int bias_C // must be in_C
	)
	{
		zq_base_type bias_v;
		int n, h, w, c;
		zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;

		for (c = 0, c_ptr = in_tensor4D_data; c < in_C; c++, c_ptr++)
		{
			bias_v = *(bias_data + c);
			for (n = 0, slice_ptr = c_ptr; n < in_N; n++, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						*pix_ptr += bias_v;
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
