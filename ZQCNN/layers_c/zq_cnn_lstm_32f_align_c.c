#include <stdio.h>
#include <omp.h>
#include<math.h>
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
#endif

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

#if __ARM_NEON
#define zq_cnn_lstm_TF_32f_align zq_cnn_lstm_TF_32f_align128bit
#define zq_mm_load_ps vld1q_f32
#define zq_mm_store_ps vst1q_f32
#define zq_mm_add_ps vaddq_f32
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps(A, B, C) vfmaq_f32(C, A, B)
#else
#define zq_mm_fmadd_ps(A, B, C) vaddq_f32(vmulq_f32(A, B), C)
#endif
#define zq_mm_mul_ps vmulq_f32
#define zq_mm_setzero_ps() vdupq_n_f32(0)
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
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFF0
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3])

#include "zq_cnn_lstm_32f_align_c_raw.h"


#undef zq_cnn_lstm_TF_32f_align
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
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
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q

#if __ARM_NEON_FP16
#define zq_cnn_lstm_TF_32f_align zq_cnn_lstm_TF_32f_align128bit
#define zq_mm_load_ps vld1q_f16
#define zq_mm_store_ps vst1q_f16
#define zq_mm_add_ps vaddq_f16
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps(A, B, C) vfmaq_f16(C, A, B)
#else
#define zq_mm_fmadd_ps(A, B, C) vaddq_f16(vmulq_f16(A, B), C)
#endif
#define zq_mm_mul_ps vmulq_f16
#define zq_mm_setzero_ps() vdupq_n_f16(0)
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
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFF0
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7]+q[8])

#include "zq_cnn_lstm_32f_align_c_raw.h"


#undef zq_cnn_lstm_TF_32f_align
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
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
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q
#endif

#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#define zq_cnn_lstm_TF_32f_align zq_cnn_lstm_TF_32f_align128bit
#define zq_mm_load_ps _mm_load_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_add_ps _mm_add_ps
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps _mm_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm_add_ps(_mm_mul_ps(A, B), C)
#endif
#define zq_mm_mul_ps _mm_mul_ps
#define zq_mm_setzero_ps _mm_setzero_ps
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
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFF0
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3])

#include "zq_cnn_lstm_32f_align_c_raw.h"


#undef zq_cnn_lstm_TF_32f_align
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
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
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q

#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#define zq_cnn_lstm_TF_32f_align zq_cnn_lstm_TF_32f_align256bit
#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_add_ps _mm256_add_ps
#if ZQ_CNN_USE_FMADD256
#define zq_mm_fmadd_ps _mm256_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#endif
#define zq_mm_mul_ps _mm256_mul_ps
#define zq_mm_setzero_ps _mm256_setzero_ps
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
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFE0
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])


#include "zq_cnn_lstm_32f_align_c_raw.h"

#undef zq_cnn_lstm_TF_32f_align
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
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
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q

#endif

#endif //__ARM_NEON

	void zq_cnn_lstm_TF_32f_align0_general(
		const float* in_data,
		int in_N,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_sliceStep,
		const float* xc_I_data,
		int xc_I_pixelStep,
		int xc_I_sliceStep,
		const float* xc_F_data,
		int xc_F_pixelStep,
		int xc_F_sliceStep,
		const float* xc_O_data,
		int xc_O_pixelStep,
		int xc_O_sliceStep,
		const float* xc_G_data,
		int xc_G_pixelStep,
		int xc_G_sliceStep,
		const float* hc_I_data,
		int hc_I_pixelStep,
		int hc_I_sliceStep,
		const float* hc_F_data,
		int hc_F_pixelStep,
		int hc_F_sliceStep,
		const float* hc_O_data,
		int hc_O_pixelStep,
		int hc_O_sliceStep,
		const float* hc_G_data,
		int hc_G_pixelStep,
		int hc_G_sliceStep,
		const float* b_I_data,
		const float* b_F_data,
		const float* b_O_data,
		const float* b_G_data,
		float* out_data,
		int out_pixelStep,
		int out_sliceStep,
		int hidden_dim,
		int is_fw,
		float forget_bias,
		float cell_clip,
		void** buffer,
		__int64* buffer_len)
	{
		int out_n, t, q, i, ti;
		const float* in_slice_ptr, *x, *weight_xc_I, *weight_xc_F, *weight_xc_O, *weight_xc_G;
		const float* weight_hc_I, *weight_hc_F, *weight_hc_O, *weight_hc_G;
		float* out_slice_ptr, *out_pixel_ptr;
		
		float* h = (float*)malloc(hidden_dim * sizeof(float));
		float* cell = (float*)malloc(hidden_dim * sizeof(float));
		float* cs = (float*)malloc(hidden_dim * sizeof(float));
		float* I = (float*)malloc(hidden_dim * sizeof(float));
		float* F = (float*)malloc(hidden_dim * sizeof(float));
		float* cs_prev = (float*)malloc(hidden_dim * sizeof(float));
		float* ci = (float*)malloc(hidden_dim * sizeof(float));
		float* co = (float*)malloc(hidden_dim * sizeof(float));
		float* o = (float*)malloc(hidden_dim * sizeof(float));
		for (out_n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
			out_n < in_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			memset(h, 0, sizeof(float)*hidden_dim);
			memset(cell, 0, sizeof(float)*hidden_dim);
			for (t = 0; t < in_W; t++)
			{
				ti = is_fw ? t : in_W - 1 - t;
				x = in_slice_ptr + ti*in_pixelStep;
				for (q = 0; q < hidden_dim; q++)
				{
					weight_xc_I = xc_I_data + q*xc_I_pixelStep;
					weight_xc_F = xc_F_data + q*xc_F_pixelStep;
					weight_xc_O = xc_O_data + q*xc_O_pixelStep;
					weight_xc_G = xc_G_data + q*xc_G_pixelStep;
					weight_hc_I = hc_I_data + q*hc_I_pixelStep;
					weight_hc_F = hc_F_data + q*hc_F_pixelStep;
					weight_hc_O = hc_O_data + q*hc_O_pixelStep;
					weight_hc_G = hc_G_data + q*hc_G_pixelStep;
					I[q] = b_I_data[q];
					F[q] = b_F_data[q];
					ci[q] = b_G_data[q];
					o[q] = b_O_data[q];
					for (i = 0; i < in_C; i++)
					{
						I[q] += weight_xc_I[i] * x[i];
						F[q] += weight_xc_F[i] * x[i];
						ci[q] += weight_xc_G[i] * x[i];
						o[q] += weight_xc_O[i] * x[i];
					}
					for (i = 0; i < hidden_dim; i++)
					{
						I[q] += weight_hc_I[i] * h[i];
						F[q] += weight_hc_F[i] * h[i];
						ci[q] += weight_hc_G[i] * h[i];
						o[q] += weight_hc_O[i] * h[i];
					}
					F[q] += forget_bias; //forget_bias = 1.0f
				}

				////https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/contrib/rnn/ops/lstm_ops.cc
				////注意权重顺序：应该是i, ci, f, o，来源于以下链接
				////https://github.com/tensorflow/tensorflow/blob/722b96b22926dbc05881c35cb63fd342c6843112/tensorflow/core/kernels/rnn/lstm_ops_gpu.cu.cc
				/*python
					xh = [x, h_prev]
					[i, f, ci, o] = xh * w + b
					f = f + forget_bias

					if not use_peephole:
				        wci = wcf = wco = 0

					i = sigmoid(cs_prev * wci + i)
					f = sigmoid(cs_prev * wcf + f)
					ci = tanh(ci)
					cs = ci.*i + cs_prev.*f
					cs = clip(cs, cell_clip)
					o = sigmoid(cs * wco + o)
					co = tanh(cs)
					h = co.*o
				*/
				out_pixel_ptr = out_slice_ptr + ti*out_pixelStep;
				for (q = 0; q < hidden_dim; q++)
				{
					cs_prev[q] = cell[q];
					I[q] = 1.f / (1.f + (float)exp(-I[q]));				// i = sigmoid(cs_prev * wci + i)
					F[q] = 1.f / (1.f + (float)exp(-F[q]));				// f = sigmoid(cs_prev * wcf + f)
					ci[q] = (float)tanh(ci[q]);							// ci = tanh(ci)
					cs[q] = ci[q] * I[q] + cs_prev[q]*F[q];			// cs = ci.*i + cs_prev.*f	
					cs[q] = __min(cell_clip, __max(-cell_clip, cs[q]));				//cs = clip(cs, cell_clip)
					o[q] = 1.f / (1.f + (float)exp(-o[q]));				// o = sigmoid(cs * wco + o)
					co[q] = (float)tanh(cs[q]);							// co = tanh(cs)
					h[q] = co[q]*o[q];								// h = co.*o

					cell[q] = cs[q];
					out_pixel_ptr[q] = h[q];
				}

				// no cell output here
			}
		}
		free(h);
		free(cell);
		free(cs);
		free(I);
		free(F);
		free(ci);
		free(co);
		free(o);
	}


#if defined(__cplusplus) || defined(c_plusplus) 
		}
#endif