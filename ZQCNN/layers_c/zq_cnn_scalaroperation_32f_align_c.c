#include <stdio.h>
#include <stdlib.h>
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
#endif//__ARM_NEON


#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

#if __ARM_NEON
#define zq_mm_load_ps vld1q_f32
#define zq_mm_store_ps vst1q_f32
#define zq_mm_type float32x4_t
#define zq_base_type float
#define zq_mm_set1_ps vdupq_n_f32
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

#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_add_32f_align128bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_add_inplace_32f_align128bit
#define zq_mm_operation_ps vaddq_f32
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align

#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_mul_32f_align128bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_mul_inplace_32f_align128bit
#define zq_mm_operation_ps vmulq_f32
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align


#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_max_32f_align128bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_max_inplace_32f_align128bit
#define zq_mm_operation_ps vmaxq_f32
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align

#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_min_32f_align128bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_min_inplace_32f_align128bit
#define zq_mm_operation_ps vminq_f32
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align


#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_rminus_32f_align128bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_rminus_inplace_32f_align128bit
#define zq_mm_operation_ps(x,y) vsubq_f32(y,x)
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align

/*
#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_rdiv_32f_align128bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_rdiv_inplace_32f_align128bit
#define zq_mm_operation_ps(x,y) vdivq_f32(y,x)
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align
*/

#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_set1_ps
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

#if __ARM_NEON_FP16
#define zq_mm_load_ps vld1q_f16
#define zq_mm_store_ps vst1q_f16
#define zq_mm_type float16x8_t
#define zq_base_type float16_t
#define zq_mm_set1_ps vdupq_n_f16
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

#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_add_16f_align128bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_add_inplace_16f_align128bit
#define zq_mm_operation_ps vaddq_f16
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align

#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_mul_16f_align128bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_mul_inplace_16f_align128bit
#define zq_mm_operation_ps vmulq_f16
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align


#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_max_16f_align128bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_max_inplace_16f_align128bit
#define zq_mm_operation_ps vmaxq_f16
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align

#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_min_16f_align128bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_min_inplace_16f_align128bit
#define zq_mm_operation_ps vminq_f16
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align


#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_rminus_16f_align128bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_rminus_inplace_16f_align128bit
#define zq_mm_operation_ps(x,y) vsubq_f16(y,x)
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align

/*
#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_rdiv_16f_align128bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_rdiv_inplace_16f_align128bit
#define zq_mm_operation_ps(x,y) vdivq_f16(y,x)
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align
*/

#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_set1_ps
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
#endif//__ARM_NEON_FP16

#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#define zq_mm_load_ps _mm_load_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_type __m128
#define zq_base_type float
#define zq_mm_set1_ps _mm_set1_ps
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

#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_add_32f_align128bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_add_inplace_32f_align128bit
#define zq_mm_operation_ps _mm_add_ps
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align

#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_mul_32f_align128bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_mul_inplace_32f_align128bit
#define zq_mm_operation_ps _mm_mul_ps
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align


#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_max_32f_align128bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_max_inplace_32f_align128bit
#define zq_mm_operation_ps _mm_max_ps
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align

#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_min_32f_align128bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_min_inplace_32f_align128bit
#define zq_mm_operation_ps _mm_min_ps
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align


#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_rminus_32f_align128bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_rminus_inplace_32f_align128bit
#define zq_mm_operation_ps(x,y) _mm_sub_ps(y,x)
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align

#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_rdiv_32f_align128bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_rdiv_inplace_32f_align128bit
#define zq_mm_operation_ps(x,y) _mm_div_ps(y,x)
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align

#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_set1_ps
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

#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_type __m256
#define zq_base_type float
#define zq_mm_set1_ps _mm256_set1_ps
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



#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_add_32f_align256bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_add_inplace_32f_align256bit
#define zq_mm_operation_ps _mm256_add_ps
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align


#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_mul_32f_align256bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_mul_inplace_32f_align256bit
#define zq_mm_operation_ps _mm256_mul_ps
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align


#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_max_32f_align256bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_max_inplace_32f_align256bit
#define zq_mm_operation_ps _mm256_max_ps
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align

#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_min_32f_align256bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_min_inplace_32f_align256bit
#define zq_mm_operation_ps _mm256_min_ps
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align

#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_rminus_32f_align256bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_rminus_inplace_32f_align256bit
#define zq_mm_operation_ps(x,y) _mm256_sub_ps(y,x)
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align

#define zq_cnn_scalaroperation_32f_align zq_cnn_scalaroperation_rdiv_32f_align256bit
#define zq_cnn_scalaroperation_inplace_32f_align zq_cnn_scalaroperation_rdiv_inplace_32f_align256bit
#define zq_mm_operation_ps(x,y) _mm256_div_ps(y,x)
#include "zq_cnn_scalaroperation_32f_align_c_raw.h"
#undef zq_mm_operation_ps
#undef zq_cnn_scalaroperation_32f_align
#undef zq_cnn_scalaroperation_inplace_32f_align

#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_set1_ps
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
#endif
#endif//__ARM_NEON

	void zq_cnn_scalaroperation_add_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	)
	{
		int n, h, w, c;
		const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr;
		float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data; 
			n < in_N; 
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr; 
				h < in_H; 
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr; 
					w < in_W; 
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
				{
					for (c = 0; c < in_C; c++)
						out_pix_ptr[c] = in_pix_ptr[c] + scalar;
				}
			}
		}
	}

	void zq_cnn_scalaroperation_add_inplace_32f_align0(
		float scalar,
		float* data,
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	)
	{
		int n, h, w, c;
		float* slice_ptr, *row_ptr, *pix_ptr;
		for (n = 0, slice_ptr = data; n < N; n++, slice_ptr += sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < H;	h++, row_ptr += widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < W; w++, pix_ptr += pixelStep)
				{
					for (c = 0; c < C; c++)
						pix_ptr[c] += scalar;
				}
			}
		}
	}


	void zq_cnn_scalaroperation_mul_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	)
	{
		int n, h, w, c;
		const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr;
		float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < in_H;
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < in_W;
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
				{
					for (c = 0; c < in_C; c++)
						out_pix_ptr[c] = in_pix_ptr[c] * scalar;
				}
			}
		}
	}

	void zq_cnn_scalaroperation_mul_inplace_32f_align0(
		float scalar,
		float* data,
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	)
	{
		int n, h, w, c;
		float* slice_ptr, *row_ptr, *pix_ptr;
		for (n = 0, slice_ptr = data; n < N; n++, slice_ptr += sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < H; h++, row_ptr += widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < W; w++, pix_ptr += pixelStep)
				{
					for (c = 0; c < C; c++)
						pix_ptr[c] *= scalar;
				}
			}
		}
	}

	void zq_cnn_scalaroperation_max_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	)
	{
		int n, h, w, c;
		const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr;
		float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < in_H;
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < in_W;
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
				{
					for (c = 0; c < in_C; c++)
						out_pix_ptr[c] = __max(in_pix_ptr[c], scalar);
				}
			}
		}
	}

	void zq_cnn_scalaroperation_max_inplace_32f_align0(
		float scalar,
		float* data,
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	)
	{
		int n, h, w, c;
		float* slice_ptr, *row_ptr, *pix_ptr;
		for (n = 0, slice_ptr = data; n < N; n++, slice_ptr += sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < H; h++, row_ptr += widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < W; w++, pix_ptr += pixelStep)
				{
					for (c = 0; c < C; c++)
						pix_ptr[c] = __max(pix_ptr[c], scalar);
				}
			}
		}
	}

	void zq_cnn_scalaroperation_min_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	)
	{
		int n, h, w, c;
		const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr;
		float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < in_H;
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < in_W;
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
				{
					for (c = 0; c < in_C; c++)
						out_pix_ptr[c] = __min(in_pix_ptr[c], scalar);
				}
			}
		}
	}

	void zq_cnn_scalaroperation_min_inplace_32f_align0(
		float scalar,
		float* data,
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	)
	{
		int n, h, w, c;
		float* slice_ptr, *row_ptr, *pix_ptr;
		for (n = 0, slice_ptr = data; n < N; n++, slice_ptr += sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < H; h++, row_ptr += widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < W; w++, pix_ptr += pixelStep)
				{
					for (c = 0; c < C; c++)
						pix_ptr[c] = __min(pix_ptr[c], scalar);
				}
			}
		}
	}

	void zq_cnn_scalaroperation_pow_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	)
	{
		int n, h, w, c;
		const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr;
		float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < in_H;
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < in_W;
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
				{
					for (c = 0; c < in_C; c++)
						out_pix_ptr[c] = pow(in_pix_ptr[c], scalar);
				}
			}
		}
	}

	void zq_cnn_scalaroperation_pow_inplace_32f_align0(
		float scalar,
		float* data,
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	)
	{
		int n, h, w, c;
		float* slice_ptr, *row_ptr, *pix_ptr;
		for (n = 0, slice_ptr = data; n < N; n++, slice_ptr += sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < H; h++, row_ptr += widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < W; w++, pix_ptr += pixelStep)
				{
					for (c = 0; c < C; c++)
						pix_ptr[c] = pow(pix_ptr[c], scalar);
				}
			}
		}
	}

	void zq_cnn_scalaroperation_rdiv_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	)
	{
		int n, h, w, c;
		const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr;
		float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < in_H;
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < in_W;
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
				{
					for (c = 0; c < in_C; c++)
						out_pix_ptr[c] = scalar/in_pix_ptr[c];
				}
			}
		}
	}

	void zq_cnn_scalaroperation_rdiv_inplace_32f_align0(
		float scalar,
		float* data,
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	)
	{
		int n, h, w, c;
		float* slice_ptr, *row_ptr, *pix_ptr;
		for (n = 0, slice_ptr = data; n < N; n++, slice_ptr += sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < H; h++, row_ptr += widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < W; w++, pix_ptr += pixelStep)
				{
					for (c = 0; c < C; c++)
						pix_ptr[c] = scalar / pix_ptr[c];
				}
			}
		}
	}

	void zq_cnn_scalaroperation_rminus_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	)
	{
		int n, h, w, c;
		const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr;
		float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < in_H;
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < in_W;
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
				{
					for (c = 0; c < in_C; c++)
						out_pix_ptr[c] = scalar - in_pix_ptr[c];
				}
			}
		}
	}

	void zq_cnn_scalaroperation_rminus_inplace_32f_align0(
		float scalar,
		float* data,
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	)
	{
		int n, h, w, c;
		float* slice_ptr, *row_ptr, *pix_ptr;
		for (n = 0, slice_ptr = data; n < N; n++, slice_ptr += sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < H; h++, row_ptr += widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < W; w++, pix_ptr += pixelStep)
				{
					for (c = 0; c < C; c++)
						pix_ptr[c] = scalar - pix_ptr[c];
				}
			}
		}
	}

#if __ARM_NEON
#if __ARM_NEON_FP16
#define zq_base_type
	void zq_cnn_scalaroperation_add_16f_align0(
		zq_base_type scalar,
		const zq_base_type* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		zq_base_type* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	)
	{
		int n, h, w, c;
		const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr;
		zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < in_H;
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < in_W;
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
				{
					for (c = 0; c < in_C; c++)
						out_pix_ptr[c] = in_pix_ptr[c] + scalar;
				}
			}
		}
	}

	void zq_cnn_scalaroperation_add_inplace_16f_align0(
		zq_base_type scalar,
		zq_base_type* data,
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	)
	{
		int n, h, w, c;
		zq_base_type* slice_ptr, *row_ptr, *pix_ptr;
		for (n = 0, slice_ptr = data; n < N; n++, slice_ptr += sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < H; h++, row_ptr += widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < W; w++, pix_ptr += pixelStep)
				{
					for (c = 0; c < C; c++)
						pix_ptr[c] += scalar;
				}
			}
		}
	}


	void zq_cnn_scalaroperation_mul_16f_align0(
		zq_base_type scalar,
		const zq_base_type* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		zq_base_type* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	)
	{
		int n, h, w, c;
		const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr;
		zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < in_H;
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < in_W;
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
				{
					for (c = 0; c < in_C; c++)
						out_pix_ptr[c] = in_pix_ptr[c] * scalar;
				}
			}
		}
	}

	void zq_cnn_scalaroperation_mul_inplace_16f_align0(
		zq_base_type scalar,
		zq_base_type* data,
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	)
	{
		int n, h, w, c;
		zq_base_type* slice_ptr, *row_ptr, *pix_ptr;
		for (n = 0, slice_ptr = data; n < N; n++, slice_ptr += sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < H; h++, row_ptr += widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < W; w++, pix_ptr += pixelStep)
				{
					for (c = 0; c < C; c++)
						pix_ptr[c] *= scalar;
				}
			}
		}
	}

	void zq_cnn_scalaroperation_max_16f_align0(
		zq_base_type scalar,
		const zq_base_type* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		zq_base_type* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	)
	{
		int n, h, w, c;
		const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr;
		zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < in_H;
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < in_W;
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
				{
					for (c = 0; c < in_C; c++)
						out_pix_ptr[c] = __max(in_pix_ptr[c], scalar);
				}
			}
		}
	}

	void zq_cnn_scalaroperation_max_inplace_16f_align0(
		zq_base_type scalar,
		zq_base_type* data,
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	)
	{
		int n, h, w, c;
		zq_base_type* slice_ptr, *row_ptr, *pix_ptr;
		for (n = 0, slice_ptr = data; n < N; n++, slice_ptr += sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < H; h++, row_ptr += widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < W; w++, pix_ptr += pixelStep)
				{
					for (c = 0; c < C; c++)
						pix_ptr[c] = __max(pix_ptr[c], scalar);
				}
			}
		}
	}

	void zq_cnn_scalaroperation_min_16f_align0(
		zq_base_type scalar,
		const zq_base_type* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		zq_base_type* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	)
	{
		int n, h, w, c;
		const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr;
		zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < in_H;
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < in_W;
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
				{
					for (c = 0; c < in_C; c++)
						out_pix_ptr[c] = __min(in_pix_ptr[c], scalar);
				}
			}
		}
	}

	void zq_cnn_scalaroperation_min_inplace_16f_align0(
		zq_base_type scalar,
		zq_base_type* data,
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	)
	{
		int n, h, w, c;
		zq_base_type* slice_ptr, *row_ptr, *pix_ptr;
		for (n = 0, slice_ptr = data; n < N; n++, slice_ptr += sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < H; h++, row_ptr += widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < W; w++, pix_ptr += pixelStep)
				{
					for (c = 0; c < C; c++)
						pix_ptr[c] = __min(pix_ptr[c], scalar);
				}
			}
		}
	}

	void zq_cnn_scalaroperation_pow_16f_align0(
		zq_base_type scalar,
		const zq_base_type* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		zq_base_type* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	)
	{
		int n, h, w, c;
		const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr;
		zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < in_H;
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < in_W;
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
				{
					for (c = 0; c < in_C; c++)
						out_pix_ptr[c] = pow(in_pix_ptr[c], scalar);
				}
			}
		}
	}

	void zq_cnn_scalaroperation_pow_inplace_16f_align0(
		zq_base_type scalar,
		zq_base_type* data,
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	)
	{
		int n, h, w, c;
		zq_base_type* slice_ptr, *row_ptr, *pix_ptr;
		for (n = 0, slice_ptr = data; n < N; n++, slice_ptr += sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < H; h++, row_ptr += widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < W; w++, pix_ptr += pixelStep)
				{
					for (c = 0; c < C; c++)
						pix_ptr[c] = pow(pix_ptr[c], scalar);
				}
			}
		}
	}

	void zq_cnn_scalaroperation_rdiv_16f_align0(
		zq_base_type scalar,
		const zq_base_type* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		zq_base_type* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	)
	{
		int n, h, w, c;
		const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr;
		zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < in_H;
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < in_W;
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
				{
					for (c = 0; c < in_C; c++)
						out_pix_ptr[c] = scalar / in_pix_ptr[c];
				}
			}
		}
	}

	void zq_cnn_scalaroperation_rdiv_inplace_16f_align0(
		zq_base_type scalar,
		zq_base_type* data,
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	)
	{
		int n, h, w, c;
		zq_base_type* slice_ptr, *row_ptr, *pix_ptr;
		for (n = 0, slice_ptr = data; n < N; n++, slice_ptr += sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < H; h++, row_ptr += widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < W; w++, pix_ptr += pixelStep)
				{
					for (c = 0; c < C; c++)
						pix_ptr[c] = scalar / pix_ptr[c];
				}
			}
		}
	}

	void zq_cnn_scalaroperation_rminus_16f_align0(
		zq_base_type scalar,
		const zq_base_type* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		zq_base_type* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	)
	{
		int n, h, w, c;
		const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr;
		zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < in_H;
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < in_W;
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
				{
					for (c = 0; c < in_C; c++)
						out_pix_ptr[c] = scalar - in_pix_ptr[c];
				}
			}
		}
	}

	void zq_cnn_scalaroperation_rminus_inplace_16f_align0(
		zq_base_type scalar,
		zq_base_type* data,
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	)
	{
		int n, h, w, c;
		zq_base_type* slice_ptr, *row_ptr, *pix_ptr;
		for (n = 0, slice_ptr = data; n < N; n++, slice_ptr += sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < H; h++, row_ptr += widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < W; w++, pix_ptr += pixelStep)
				{
					for (c = 0; c < C; c++)
						pix_ptr[c] = scalar - pix_ptr[c];
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
