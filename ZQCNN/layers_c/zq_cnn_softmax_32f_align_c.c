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
#include <float.h>
#include "../math/zq_avx_mathfun.h"
#include "../math/zq_sse_mathfun.h"
#include "..\ZQ_CNN_CompileConfig.h"

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif


#define zq_cnn_softmax_32f_align_C zq_cnn_softmax_32f_align128bit_C
#define zq_mm_load_ps _mm_load_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_max_ps _mm_max_ps
#define zq_mm_add_ps _mm_add_ps
#define zq_mm_sub_ps _mm_sub_ps
#define zq_mm_mul_ps _mm_mul_ps
#define zq_mm_set1_ps _mm_set1_ps
#define zq_mm_exp_ps zq_mm128_exp_ps//_mm_exp_ps
#define zq_mm_type __m128
#define zq_mm_align_size 4
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFF0
#define zq_final_max_q __max(q[0],__max(q[1],__max(q[2],q[3])))
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3])

#include "zq_cnn_softmax_32f_align_c_raw.h"


#undef zq_cnn_softmax_32f_align_C
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_max_ps
#undef zq_mm_add_ps
#undef zq_mm_sub_ps
#undef zq_mm_mul_ps
#undef zq_mm_set1_ps
#undef zq_mm_exp_ps
#undef zq_mm_type
#undef zq_mm_align_size
#undef zq_mm_bitor_longlong
#undef zq_final_max_q
#undef zq_final_sum_q

#define zq_cnn_softmax_32f_align_C zq_cnn_softmax_32f_align256bit_C
#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_max_ps _mm256_max_ps
#define zq_mm_add_ps _mm256_add_ps
#define zq_mm_sub_ps _mm256_sub_ps
#define zq_mm_mul_ps _mm256_mul_ps
#define zq_mm_set1_ps _mm256_set1_ps
#define zq_mm_exp_ps zq_mm256_exp_ps//_mm256_exp_ps
#define zq_mm_type __m256
#define zq_mm_align_size 8
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFE0
#define zq_final_max_q __max(q[0],__max(q[1],__max(q[2],__max(q[3],__max(q[4],__max(q[5],__max(q[6],q[7])))))))
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])

#include "zq_cnn_softmax_32f_align_c_raw.h"

#undef zq_cnn_softmax_32f_align_C
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_max_ps
#undef zq_mm_add_ps
#undef zq_mm_sub_ps
#undef zq_mm_mul_ps
#undef zq_mm_set1_ps
#undef zq_mm_exp_ps
#undef zq_mm_type
#undef zq_mm_align_size
#undef zq_mm_bitor_longlong
#undef zq_final_max_q
#undef zq_final_sum_q


void zq_cnn_softmax_32f_align0_C(
	float* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_alignPixelStep,
	int in_widthStep,
	int in_SliceStep
)
{
	/* value = exp( value - global max value )
	 sum all value
	 value = value / sum*/
	float max_val, tmp_val, sum_val;
	int n, h, w, c;
	float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
	for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_SliceStep)
	{
		for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
		{
			for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_alignPixelStep)
			{
				//compute max_val
				max_val = -FLT_MAX;
				for (c = 0, c_ptr = pix_ptr; c < in_C; c ++, c_ptr ++)
					max_val = __max(max_val, *(c_ptr));

				//compute sum

				sum_val = 0;
				for (c = 0, c_ptr = pix_ptr; c < in_C; c++, c_ptr++)
				{
					tmp_val = exp((*c_ptr) - max_val);
					sum_val += tmp_val;
					*c_ptr = tmp_val;
				}


				//divide
				sum_val = 1.0f / sum_val;
				for (c = 0, c_ptr = pix_ptr; c < in_C; c ++, c_ptr ++)
					*c_ptr *= sum_val;
			}
		}
	}
}

void zq_cnn_softmax_32f_align0_H(
	float* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_alignPixelStep,
	int in_widthStep,
	int in_SliceStep
)
{
	/* value = exp( value - global max value )
	sum all value
	value = value / sum*/
	float max_val, tmp_val, sum_val;
	int n, h, w, c;
	float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
	for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_SliceStep)
	{
		for (c = 0, c_ptr = slice_ptr; c < in_C; c++, c_ptr ++)
		{
			for (w = 0, pix_ptr = c_ptr; w < in_W; w++, pix_ptr += in_alignPixelStep)
			{
				//compute max_val
				max_val = -FLT_MAX;
				for (h = 0, row_ptr = pix_ptr; h < in_H; h++, row_ptr+=in_widthStep)
					max_val = __max(max_val, *(row_ptr));

				//compute sum

				sum_val = 0;
				for (h = 0, row_ptr = pix_ptr; h < in_H; h++, row_ptr+=in_widthStep)
				{
					tmp_val = exp((*row_ptr) - max_val);
					sum_val += tmp_val;
					*row_ptr = tmp_val;
				}


				//divide
				sum_val = 1.0f / sum_val;
				for (h = 0, row_ptr = pix_ptr; h < in_H; h++, row_ptr+=in_widthStep)
					*row_ptr *= sum_val;
			}
		}
	}
}


void zq_cnn_softmax_32f_align0_W(
	float* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_alignPixelStep,
	int in_widthStep,
	int in_SliceStep
)
{
	/* value = exp( value - global max value )
	sum all value
	value = value / sum*/
	float max_val, tmp_val, sum_val;
	int n, h, w, c;
	float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
	for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_SliceStep)
	{
		for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
		{
			for (c = 0, c_ptr = row_ptr; c < in_C; c++, c_ptr ++)
			{
				//compute max_val
				max_val = -FLT_MAX;
				for (w = 0, pix_ptr = c_ptr; w < in_W; w++, pix_ptr+=in_alignPixelStep)
					max_val = __max(max_val, *(pix_ptr));

				//compute sum

				sum_val = 0;
				for (w = 0, pix_ptr = c_ptr; w < in_W; w++, pix_ptr+=in_alignPixelStep)
				{
					tmp_val = exp((*pix_ptr) - max_val);
					sum_val += tmp_val;
					*pix_ptr = tmp_val;
				}


				//divide
				sum_val = 1.0f / sum_val;
				for (w = 0, pix_ptr = c_ptr; w < in_W; w++, pix_ptr+=in_alignPixelStep)
					*pix_ptr *= sum_val;
			}
		}
	}
}

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif