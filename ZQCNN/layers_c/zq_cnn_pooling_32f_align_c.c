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
#include <float.h>
#include <stdio.h>

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

#define zq_cnn_maxpooling_nopadding_suredivided_kernel2x2 zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel2x2
#define zq_cnn_maxpooling_nopadding_suredivided_kernel2x2_omp zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel2x2_omp
#define zq_cnn_maxpooling_nopadding_suredivided_kernel3x3 zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel3x3
#define zq_cnn_maxpooling_nopadding_suredivided_kernel3x3_omp zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel3x3_omp
#define zq_cnn_maxpooling_nopadding_suredivided_kernel5x5 zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel5x5
#define zq_cnn_maxpooling_nopadding_suredivided_kernel5x5_omp zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel5x5_omp
#define zq_cnn_maxpooling_nopadding_suredivided_general zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_general
#define zq_cnn_maxpooling_nopadding_suredivided_general_omp zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_general_omp
#define zq_cnn_maxpooling_nopadding_nodivided_general zq_cnn_maxpooling_nopadding_nodivided_32f_align128bit_general
#define zq_cnn_maxpooling_nopadding_nodivided_general_omp zq_cnn_maxpooling_nopadding_nodivided_32f_align128bit_general_omp
#define zq_mm_load_ps _mm_load_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_max_ps _mm_max_ps
#define zq_mm_set1_ps _mm_set1_ps
#define zq_mm_type __m128
#define zq_mm_align_size 4

#include "zq_cnn_pooling_32f_align_c_raw.h"


#undef zq_cnn_maxpooling_nopadding_suredivided_kernel2x2
#undef zq_cnn_maxpooling_nopadding_suredivided_kernel2x2_omp
#undef zq_cnn_maxpooling_nopadding_suredivided_kernel3x3
#undef zq_cnn_maxpooling_nopadding_suredivided_kernel3x3_omp
#undef zq_cnn_maxpooling_nopadding_suredivided_kernel5x5
#undef zq_cnn_maxpooling_nopadding_suredivided_kernel5x5_omp
#undef zq_cnn_maxpooling_nopadding_suredivided_general
#undef zq_cnn_maxpooling_nopadding_suredivided_general_omp
#undef zq_cnn_maxpooling_nopadding_nodivided_general
#undef zq_cnn_maxpooling_nopadding_nodivided_general_omp
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_max_ps
#undef zq_mm_set1_ps
#undef zq_mm_type
#undef zq_mm_align_size

#define zq_cnn_maxpooling_nopadding_suredivided_kernel2x2 zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_kernel2x2
#define zq_cnn_maxpooling_nopadding_suredivided_kernel2x2_omp zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_kernel2x2_omp
#define zq_cnn_maxpooling_nopadding_suredivided_kernel3x3 zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_kernel3x3
#define zq_cnn_maxpooling_nopadding_suredivided_kernel3x3_omp zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_kernel3x3_omp
#define zq_cnn_maxpooling_nopadding_suredivided_kernel5x5 zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_kernel5x5
#define zq_cnn_maxpooling_nopadding_suredivided_kernel5x5_omp zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_kernel5x5_omp
#define zq_cnn_maxpooling_nopadding_suredivided_general zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_general
#define zq_cnn_maxpooling_nopadding_suredivided_general_omp zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_general_omp
#define zq_cnn_maxpooling_nopadding_nodivided_general zq_cnn_maxpooling_nopadding_nodivided_32f_align256bit_general
#define zq_cnn_maxpooling_nopadding_nodivided_general_omp zq_cnn_maxpooling_nopadding_nodivided_32f_align256bit_general_omp
#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_max_ps _mm256_max_ps
#define zq_mm_set1_ps _mm256_set1_ps
#define zq_mm_type __m256
#define zq_mm_align_size 8

#include "zq_cnn_pooling_32f_align_c_raw.h"

#undef zq_cnn_maxpooling_nopadding_suredivided_kernel2x2
#undef zq_cnn_maxpooling_nopadding_suredivided_kernel2x2_omp
#undef zq_cnn_maxpooling_nopadding_suredivided_kernel3x3
#undef zq_cnn_maxpooling_nopadding_suredivided_kernel3x3_omp
#undef zq_cnn_maxpooling_nopadding_suredivided_kernel5x5
#undef zq_cnn_maxpooling_nopadding_suredivided_kernel5x5_omp
#undef zq_cnn_maxpooling_nopadding_suredivided_general
#undef zq_cnn_maxpooling_nopadding_suredivided_general_omp
#undef zq_cnn_maxpooling_nopadding_nodivided_general
#undef zq_cnn_maxpooling_nopadding_nodivided_general_omp
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_max_ps
#undef zq_mm_set1_ps
#undef zq_mm_type
#undef zq_mm_align_size


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
#if 1
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

#else
	if ((in_W - kernel_W) % stride_W == 0 && (in_H - kernel_H) % stride_H == 0)
	{
		for (int n = 0; n < in_N; n++)
		{
			const float* cur_src = in_tensor4D_data + n*in_SliceStep;
			float* cur_dst = out_tensor4D_data + n*out_alignSliceStep;
			for (int c = 0; c < in_C; c++)
			{
				for (int out_h = 0; out_h < out_H; out_h++)
				{
					for (int out_w = 0; out_w < out_W; out_w++)
					{
						const float* pIn = cur_src + c + out_h*stride_H*in_widthStep + out_w*stride_W* in_alignPixelStep;
						float* pOut = cur_dst + c + out_h*out_alignWidthStep + out_w*out_alignPixelStep;
						pOut[0] = pIn[0];
						for (int kh = 0; kh < kernel_H; kh++)
						{
							for (int kw = 0; kw < kernel_W; kw++)
							{
								pOut[0] = __max(pOut[0], pIn[kh*in_widthStep + kw*in_alignPixelStep]);
								
							}
						}
						//printf("[%d,%d,%d]%f\n", out_h, out_w, c, pOut[0]);
					}
				}
			}
		}

	}
	else
	{
		for (int n = 0; n < in_N; n++)
		{
			const float* cur_src = in_tensor4D_data + n*in_SliceStep;
			float* cur_dst = out_tensor4D_data + n*out_alignSliceStep;
			for (int c = 0; c < in_C; c++)
			{
				for (int out_h = 0; out_h < out_H; out_h++)
				{
					for (int out_w = 0; out_w < out_W; out_w++)
					{
						const float* pIn = cur_src + c + out_h*stride_H*in_widthStep + out_w*stride_W*in_alignPixelStep;
						float* pOut = cur_dst + c + out_h*out_alignWidthStep + out_w * out_alignPixelStep;
						pOut[0] = pIn[0];
						int max_kh = __min(kernel_H, in_H - out_h*stride_H);
						int max_kw = __min(kernel_W, in_W - out_w*stride_W);
						for (int kh = 0; kh < max_kh; kh++)
						{
							for (int kw = 0; kw < max_kw; kw++)
							{
								pOut[0] = __max(pOut[0], pIn[kh*in_widthStep + kw*in_alignPixelStep]);
								
							}
						}
						//printf("[%d,%d,%d]%f\n", out_h, out_w, c, pOut[0]);
					}
				}
			}
		}
	}

#endif
}

void zq_cnn_maxpooling_nopadding_32f_align0_general_omp(
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
	int out_alignSliceStep,
	int thread_count
)
{
	int c;
	int chunk_size = (in_C + thread_count - 1) / thread_count;
	int in_widthStep_mul_strideH = stride_H*in_widthStep;
	int in_pixelStep_mul_strideW = stride_W*in_alignPixelStep;
	
	if ((in_W - kernel_W) % stride_W == 0 && (in_H - kernel_H) % stride_H == 0)
	{
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
		for (c = 0; c < in_C; c++)
		{
			int out_n, out_h, out_w, kh, kw;
			float max_val;
			const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr;
			float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
			const float* cur_in_row_ptr;
			const float* cur_in_pix_ptr;
			for (out_n = 0, in_slice_ptr = in_tensor4D_data + c, out_slice_ptr = out_tensor4D_data + c;
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
						max_val = -FLT_MAX;
						for (kh = 0, cur_in_row_ptr = in_pix_ptr;
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
						*out_pix_ptr = max_val;
					}
				}
			}
		}
	}
	else
	{
		int final_kH = __min(kernel_H, in_H - (out_H - 1)*stride_H);
		int final_kW = __min(kernel_W, in_W - (out_W - 1)*stride_W);
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
		for (c = 0; c < in_C; c++)
		{
			int out_n, out_h, out_w, kh, kw;
			float max_val;
			const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr;
			float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
			const float* cur_in_row_ptr;
			const float* cur_in_pix_ptr;
			for (out_n = 0, in_slice_ptr = in_tensor4D_data + c, out_slice_ptr = out_tensor4D_data + c;
				out_n < out_N;
				out_n++, in_slice_ptr += in_SliceStep, out_slice_ptr += out_alignSliceStep)
			{
				for (out_h = 0, out_row_ptr = out_slice_ptr, in_row_ptr = in_slice_ptr;
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

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif