#include <mmintrin.h> //MMX  
#include <xmmintrin.h> //SSE(include mmintrin.h)  
#include <emmintrin.h> //SSE2(include xmmintrin.h)  
#include <pmmintrin.h> //SSE3(include emmintrin.h)  
#include <tmmintrin.h>//SSSE3(include pmmintrin.h)  
#include <smmintrin.h>//SSE4.1(include tmmintrin.h)  
#include <nmmintrin.h>//SSE4.2(include smmintrin.h)  
#include <wmmintrin.h>//AES(include nmmintrin.h)  
#include <immintrin.h>//AVX(include wmmintrin.h)  
#include <intrin.h>//(include immintrin.h)  
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include "..\ZQ_CNN_CompileConfig.h"
#include "zq_cnn_batchnormscale_32f_align_c.h"

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

#ifndef FLOAT_EPS_FOR_DIV 
#define FLOAT_EPS_FOR_DIV 1e-32
#endif

#define zq_cnn_batchnormscale_32f_mean_var_scale_bias_align zq_cnn_batchnormscale_32f_mean_var_scale_bias_align128bit
#define zq_cnn_batchnormscale_32f_mean_var_scale_bias_align_omp zq_cnn_batchnormscale_32f_mean_var_scale_bias_align128bit_omp
#define zq_cnn_batchnorm_32f_mean_var_align zq_cnn_batchnorm_32f_mean_var_align128bit
#define zq_cnn_batchnorm_32f_mean_var_align_omp zq_cnn_batchnorm_32f_mean_var_align128bit_omp
#define zq_cnn_scale_32f_align zq_cnn_scale_32f_align128bit
#define zq_cnn_scale_32f_align_omp zq_cnn_scale_32f_align128bit_omp
#define zq_cnn_batchnorm_32f_b_a_align zq_cnn_batchnorm_32f_b_a_align128bit
#define zq_cnn_batchnorm_32f_b_a_align_omp zq_cnn_batchnorm_32f_b_a_align128bit_omp
#define zq_mm_load_ps _mm_load_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_add_ps _mm_add_ps
#define zq_mm_mul_ps _mm_mul_ps
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps _mm_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm_add_ps(_mm_mul_ps(A, B), C)
#endif
#define zq_mm_type __m128
#define zq_mm_align_size 4

#include "zq_cnn_batchnormscale_32f_align_c_raw.h"


#undef zq_cnn_batchnormscale_32f_mean_var_scale_bias_align
#undef zq_cnn_batchnormscale_32f_mean_var_scale_bias_align_omp
#undef zq_cnn_batchnorm_32f_mean_var_align
#undef zq_cnn_batchnorm_32f_mean_var_align_omp
#undef zq_cnn_scale_32f_align
#undef zq_cnn_scale_32f_align_omp
#undef zq_cnn_batchnorm_32f_b_a_align
#undef zq_cnn_batchnorm_32f_b_a_align_omp
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_mul_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_type
#undef zq_mm_align_size


#define zq_cnn_batchnormscale_32f_mean_var_scale_bias_align zq_cnn_batchnormscale_32f_mean_var_scale_bias_align256bit
#define zq_cnn_batchnormscale_32f_mean_var_scale_bias_align_omp zq_cnn_batchnormscale_32f_mean_var_scale_bias_align256bit_omp
#define zq_cnn_batchnorm_32f_mean_var_align zq_cnn_batchnorm_32f_mean_var_align256bit
#define zq_cnn_batchnorm_32f_mean_var_align_omp zq_cnn_batchnorm_32f_mean_var_align256bit_omp
#define zq_cnn_scale_32f_align zq_cnn_scale_32f_align256bit
#define zq_cnn_scale_32f_align_omp zq_cnn_scale_32f_align256bit_omp
#define zq_cnn_batchnorm_32f_b_a_align zq_cnn_batchnorm_32f_b_a_align256bit
#define zq_cnn_batchnorm_32f_b_a_align_omp zq_cnn_batchnorm_32f_b_a_align256bit_omp
#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_add_ps _mm256_add_ps
#define zq_mm_mul_ps _mm256_mul_ps
#if ZQ_CNN_USE_FMADD256
#define zq_mm_fmadd_ps _mm256_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#endif
#define zq_mm_type __m256
#define zq_mm_align_size 8

#include "zq_cnn_batchnormscale_32f_align_c_raw.h"


#undef zq_cnn_batchnormscale_32f_mean_var_scale_bias_align
#undef zq_cnn_batchnormscale_32f_mean_var_scale_bias_align_omp
#undef zq_cnn_batchnorm_32f_mean_var_align
#undef zq_cnn_batchnorm_32f_mean_var_align_omp
#undef zq_cnn_scale_32f_align
#undef zq_cnn_scale_32f_align_omp
#undef zq_cnn_batchnorm_32f_b_a_align
#undef zq_cnn_batchnorm_32f_b_a_align_omp
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_mul_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_type
#undef zq_mm_align_size

	/*
	a = bias - scale * mean / sqrt(var)
	b = scale / sqrt(var)
	value = b * value + a
	*/
	void zq_cnn_batchnormscale_32f_mean_var_scale_bias_align0(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float* mean_data,
		const float* var_data,
		const float* scale_data,
		const float* bias_data
	)
	{
		float* a, *b;
		int c;
		a = (float*)malloc(in_C);
		b = (float*)malloc(in_C);
		for (c = 0; c < in_C; c++)
		{
			b[c] = scale_data[c] / __max(sqrt(var_data[c]), FLOAT_EPS_FOR_DIV);
			a[c] = bias_data[c] - mean_data[c] * b[c];
		}

		zq_cnn_batchnorm_32f_b_a_align0(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep, b, a);
		free(a);
		free(b);
	}

	/*
	a = bias - scale * mean / sqrt(var)
	b = scale / sqrt(var)
	value = b * value + a
	*/
	void zq_cnn_batchnormscale_32f_mean_var_scale_bias_align0_omp(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float* mean_data,
		const float* var_data,
		const float* scale_data,
		const float* bias_data,
		int thread_count
	)
	{
		float* a, *b;
		int c;
		a = (float*)malloc(in_C);
		b = (float*)malloc(in_C);
		for (c = 0; c < in_C; c++)
		{
			b[c] = scale_data[c] / __max(sqrt(var_data[c]), FLOAT_EPS_FOR_DIV);
			a[c] = bias_data[c] - mean_data[c] * b[c];
		}

		zq_cnn_batchnorm_32f_b_a_align0_omp(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep, b, a, thread_count);
		free(a);
		free(b);
	}

	/*
	a = - mean / sqrt(var)
	b = 1 / sqrt(var)
	value = b * value + a
	*/
	void zq_cnn_batchnorm_32f_mean_var_align0(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float* mean_data,
		const float* var_data
	)
	{
		float* a, *b;
		int c;
		a = (float*)malloc(in_C);
		b = (float*)malloc(in_C);
		for (c = 0; c < in_C; c++)
		{
			b[c] = 1.0f / __max(sqrt(var_data[c]), FLOAT_EPS_FOR_DIV);
			a[c] = - mean_data[c] * b[c];
		}

		zq_cnn_batchnorm_32f_b_a_align0(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep, b, a);
		free(a);
		free(b);
	}

	/*
	a = - mean / sqrt(var)
	b = 1 / sqrt(var)
	value = b * value + a
	*/
	void zq_cnn_batchnorm_32f_mean_var_align0_omp(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float* mean_data,
		const float* var_data,
		int thread_count
	)
	{
		float* a, *b;
		int c;
		a = (float*)malloc(in_C);
		b = (float*)malloc(in_C);
		for (c = 0; c < in_C; c++)
		{
			b[c] = 1.0f / __max(sqrt(var_data[c]), FLOAT_EPS_FOR_DIV);
			a[c] = -mean_data[c] * b[c];
		}

		zq_cnn_batchnorm_32f_b_a_align0_omp(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep, b, a, thread_count);
		free(a);
		free(b);
	}

	void zq_cnn_scale_32f_align0(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float* scale_data,
		const float* bias_data	// bias can be NULL
	)
	{
		int n, h, w, c;
		float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
		if (bias_data != NULL)
		{
			for (n = 0, slice_ptr = in_data; n < in_C; n++, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c++, c_ptr++)
						{
							*c_ptr *= scale_data[c];
						}
					}
				}
			}
		}
		else
		{
			for (n = 0, slice_ptr = in_data; n < in_C; n++, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c++, c_ptr++)
						{
							*c_ptr *= scale_data[c];
							*c_ptr += bias_data[c];
						}
					}
				}
			}
		}
	}

	void zq_cnn_scale_32f_align0_omp(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float* scale_data,
		const float* bias_data,	// bias can be NULL
		int thread_count
	)
	{
		int chunk_size = (in_C + thread_count - 1) / thread_count;
		int c;
		if (bias_data != NULL)
		{
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
			for (c = 0; c < in_C; c++)
			{
				int n, h, w;
				float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
				c_ptr = in_data + c;
				for (n = 0, slice_ptr = c_ptr; n < in_C; n++, slice_ptr += in_sliceStep)
				{
					for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
					{
						for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixStep)
						{
							*pix_ptr *= scale_data[c];
						}
					}
				}
			}
		}
		else
		{
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
			for (c = 0; c < in_C; c++)
			{
				int n, h, w;
				float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
				c_ptr = in_data + c;
				for (n = 0, slice_ptr = in_data; n < in_C; n++, slice_ptr += in_sliceStep)
				{
					for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
					{
						for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixStep)
						{
							*pix_ptr *= scale_data[c];
							*pix_ptr += bias_data[c];
						}
					}
				}
			}
		}
	}

	/*
	a = bias - scale * mean / sqrt(var)
	b = scale / sqrt(var)
	value = b * value + a
	OR
	a = - mean / sqrt(var)
	b = 1 / sqrt(var)
	value = b * value + a
	*/
	void zq_cnn_batchnorm_32f_b_a_align0(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float* b_data,
		const float* a_data
	)
	{
		int n, h, w, c;
		float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
		const float *a_ptr, *b_ptr;
		for (n = 0, slice_ptr = in_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixStep)
				{
					for (c = 0, c_ptr = pix_ptr, b_ptr = b_data, a_ptr = a_data; c < in_C; c++, c_ptr++, b_ptr++, a_ptr++)
					{
						*c_ptr *= *b_ptr;
						*c_ptr += *a_ptr;
					}
				}
			}
		}
	}

	/*
	a = bias - scale * mean / sqrt(var)
	b = scale / sqrt(var)
	value = b * value + a
	OR
	a = - mean / sqrt(var)
	b = 1 / sqrt(var)
	value = b * value + a
	*/
	void zq_cnn_batchnorm_32f_b_a_align0_omp(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float* b_data,
		const float* a_data,
		int thread_count
	)
	{
		int chunk_size = (in_C + thread_count - 1) / thread_count;
		int c;

#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
		for (c = 0; c < in_C; c++)
		{
			int n, h, w;
			float* slice_ptr, *row_ptr, *pix_ptr;
			for (n = 0, slice_ptr = in_data + c; n < in_N; n++, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixStep)
					{
						*pix_ptr *= b_data[c];
						*pix_ptr += a_data[c];
					}
				}
			}
		}
	}

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif