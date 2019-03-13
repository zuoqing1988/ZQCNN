#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "../ZQ_CNN_CompileConfig.h"
#if __ARM_NEON

#else
#if defined(__GNUC__)
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#include <smmintrin.h>
#include "../math/zq_sse_mathfun.h"
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#include <x86intrin.h>
#include "../math/zq_avx_mathfun.h"
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
#include "../math/zq_sse_mathfun.h"
#endif 
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#include <wmmintrin.h>//AES(include nmmintrin.h)  
#include <immintrin.h>//AVX(include wmmintrin.h)  
#include <intrin.h>//(include immintrin.h)  
#include "../math/zq_avx_mathfun.h"
#endif
#endif
#endif //__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

#if __ARM_NEON

#else

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#define zq_cnn_lrn_across_channels_32f_align zq_cnn_lrn_across_channels_32f_align128bit
#define zq_mm_load_ps _mm_load_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_add_ps _mm_add_ps
#define zq_mm_mul_ps _mm_mul_ps
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps _mm_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm_add_ps(_mm_mul_ps(A, B), C)
#endif
#define zq_mm_set1_ps _mm_set1_ps
#define zq_mm_log_ps zq_mm128_log_ps
#define zq_mm_exp_ps zq_mm128_exp_ps
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

#include "zq_cnn_lrn_32f_align_c_raw.h"


#undef zq_cnn_lrn_across_channels_32f_align
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_mul_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_set1_ps
#undef zq_mm_log_ps
#undef zq_mm_exp_ps
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
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#define zq_cnn_lrn_across_channels_32f_align zq_cnn_lrn_across_channels_32f_align256bit
#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_add_ps _mm256_add_ps
#define zq_mm_mul_ps _mm256_mul_ps
#if ZQ_CNN_USE_FMADD256
#define zq_mm_fmadd_ps _mm256_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#endif
#define zq_mm_set1_ps _mm256_set1_ps
#define zq_mm_log_ps zq_mm256_log_ps
#define zq_mm_exp_ps zq_mm256_exp_ps
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

#include "zq_cnn_lrn_32f_align_c_raw.h"


#undef zq_cnn_lrn_across_channels_32f_align
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_mul_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_set1_ps
#undef zq_mm_log_ps
#undef zq_mm_exp_ps
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
#endif

#endif //__ARM_NEON

	/* it is safe to use out_tensor4D_data = in_tensor4D_data */
	void zq_cnn_lrn_across_channels_32f_align0(
		int local_size,		// must be odd number
		float alpha,
		float beta,
		float k,								
		const float* in_tensor4D_data,
		int N,
		int H,
		int W,
		int C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_tensor4D_data,
		int out_pixStep,
		int out_widthStep,
		int out_sliceStep
	)
	{
		const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_c_ptr;
		float* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_c_ptr;
		int n, h, w, c;
		int pad_size = local_size / 2;
		int len = C + (pad_size << 1);
		float local_sum_square,pow_val;
		float* square_buf = (float*)malloc(sizeof(float)*len);
		float* accumulate_buf = (float*)malloc(sizeof(float)*(len+1));
		float alpha_div_local_size = alpha / (float)local_size;

		accumulate_buf[0] = 0;
		for (c = 0; c < pad_size; c++)
		{
			square_buf[c] = 0;
			square_buf[len - 1 - c] = 0;
			accumulate_buf[c + 1] = 0;
		}
		

		for (n = 0,in_slice_ptr = in_tensor4D_data,out_slice_ptr = out_tensor4D_data; 
			n < N; 
			n++,in_slice_ptr+=in_sliceStep,out_slice_ptr+=out_sliceStep)
		{
			for (h = 0,in_row_ptr = in_slice_ptr,out_row_ptr = out_slice_ptr; 
				h < H; 
				h++, in_row_ptr+=in_widthStep,out_row_ptr += out_widthStep)
			{
				for (w = 0,in_pix_ptr = in_row_ptr,out_pix_ptr = out_row_ptr; 
					w < W; 
					w++,in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixStep)
				{
					//compute x^2
					for (c = 0, in_c_ptr = in_pix_ptr; c < C; c++, in_c_ptr++)
					{
						square_buf[pad_size + c] = (*in_c_ptr)*(*in_c_ptr);
					}
					//compute accumulate
					for (c = pad_size; c < len; c++)
						accumulate_buf[c+1] = accumulate_buf[c] + square_buf[c];

					for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr; c < C; c++, in_c_ptr++, out_c_ptr++)
					{
						local_sum_square = accumulate_buf[c + local_size] - accumulate_buf[c];
						pow_val = pow(k + alpha_div_local_size*local_sum_square, -beta);
						*out_c_ptr = *in_c_ptr * pow_val;
					}
				}
			}
		}
	}

#if __ARM_NEON
#if __ARM_NEON_FP16
#define zq_base_type float16_t
	/* it is safe to use out_tensor4D_data = in_tensor4D_data */
	void zq_cnn_lrn_across_channels_32f_align0(
		int local_size,		// must be odd number
		zq_base_type alpha,
		zq_base_type beta,
		zq_base_type k,
		const zq_base_type* in_tensor4D_data,
		int N,
		int H,
		int W,
		int C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		zq_base_type* out_tensor4D_data,
		int out_pixStep,
		int out_widthStep,
		int out_sliceStep
	)
	{
		const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_c_ptr;
		zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_c_ptr;
		int n, h, w, c;
		int pad_size = local_size / 2;
		int len = C + (pad_size << 1);
		zq_base_type local_sum_square, pow_val;
		zq_base_type* square_buf = (zq_base_type*)malloc(sizeof(zq_base_type)*len);
		zq_base_type* accumulate_buf = (zq_base_type*)malloc(sizeof(zq_base_type)*(len + 1));
		zq_base_type alpha_div_local_size = alpha / (zq_base_type)local_size;

		accumulate_buf[0] = 0;
		for (c = 0; c < pad_size; c++)
		{
			square_buf[c] = 0;
			square_buf[len - 1 - c] = 0;
			accumulate_buf[c + 1] = 0;
		}


		for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			n < N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < H;
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < W;
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixStep)
				{
					//compute x^2
					for (c = 0, in_c_ptr = in_pix_ptr; c < C; c++, in_c_ptr++)
					{
						square_buf[pad_size + c] = (*in_c_ptr)*(*in_c_ptr);
					}
					//compute accumulate
					for (c = pad_size; c < len; c++)
						accumulate_buf[c + 1] = accumulate_buf[c] + square_buf[c];

					for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr; c < C; c++, in_c_ptr++, out_c_ptr++)
					{
						local_sum_square = accumulate_buf[c + local_size] - accumulate_buf[c];
						pow_val = pow(k + alpha_div_local_size*local_sum_square, -beta);
						*out_c_ptr = *in_c_ptr * pow_val;
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