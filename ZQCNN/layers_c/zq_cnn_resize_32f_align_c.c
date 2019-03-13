#include <malloc.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
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
#define zq_cnn_resize_with_safeborder zq_cnn_resize_with_safeborder_32f_align128bit
#define zq_cnn_resize_without_safeborder zq_cnn_resize_without_safeborder_32f_align128bit
#define zq_mm_load_ps vld1q_f32
#define zq_mm_store_ps vst1q_f32
#define zq_mm_set1_ps vdupq_n_f32
#define zq_mm_add_ps vaddq_f32
#define zq_mm_mul_ps vmulq_f32
#define zq_mm_sub_ps vsubq_f32
#define zq_mm_type float32x4_t
#define zq_base_type float
#define zq_mm_align_size 4

#include "zq_cnn_resize_32f_align_c_raw.h"

#undef zq_cnn_resize_with_safeborder
#undef zq_cnn_resize_without_safeborder
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_set1_ps
#undef zq_mm_add_ps
#undef zq_mm_mul_ps
#undef zq_mm_sub_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size

#if __ARM_NEON_FP16
#define zq_cnn_resize_with_safeborder zq_cnn_resize_with_safeborder_16f_align128bit
#define zq_cnn_resize_without_safeborder zq_cnn_resize_without_safeborder_16f_align128bit
#define zq_mm_load_ps vld1q_f16
#define zq_mm_store_ps vst1q_f16
#define zq_mm_set1_ps vdupq_n_f16
#define zq_mm_add_ps vaddq_f16
#define zq_mm_mul_ps vmulq_f16
#define zq_mm_sub_ps vsubq_f16
#define zq_mm_type float16x8_t
#define zq_base_type float16_t
#define zq_mm_align_size 8

#include "zq_cnn_resize_32f_align_c_raw.h"

#undef zq_cnn_resize_with_safeborder
#undef zq_cnn_resize_without_safeborder
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_set1_ps
#undef zq_mm_add_ps
#undef zq_mm_mul_ps
#undef zq_mm_sub_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#endif//__ARM_NEON_FP16

#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#define zq_cnn_resize_with_safeborder zq_cnn_resize_with_safeborder_32f_align128bit
#define zq_cnn_resize_without_safeborder zq_cnn_resize_without_safeborder_32f_align128bit
#define zq_mm_load_ps _mm_load_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_set1_ps _mm_set1_ps
#define zq_mm_add_ps _mm_add_ps
#define zq_mm_mul_ps _mm_mul_ps
#define zq_mm_sub_ps _mm_sub_ps
#define zq_mm_type __m128
#define zq_base_type float
#define zq_mm_align_size 4

#include "zq_cnn_resize_32f_align_c_raw.h"

#undef zq_cnn_resize_with_safeborder
#undef zq_cnn_resize_without_safeborder
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_set1_ps
#undef zq_mm_add_ps
#undef zq_mm_mul_ps
#undef zq_mm_sub_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#define zq_cnn_resize_with_safeborder zq_cnn_resize_with_safeborder_32f_align256bit
#define zq_cnn_resize_without_safeborder zq_cnn_resize_without_safeborder_32f_align256bit
#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_set1_ps _mm256_set1_ps
#define zq_mm_add_ps _mm256_add_ps
#define zq_mm_mul_ps _mm256_mul_ps
#define zq_mm_sub_ps _mm256_sub_ps
#define zq_mm_type __m256
#define zq_base_type float
#define zq_mm_align_size 8

#include "zq_cnn_resize_32f_align_c_raw.h"

#undef zq_cnn_resize_with_safeborder
#undef zq_cnn_resize_without_safeborder
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_set1_ps
#undef zq_mm_add_ps
#undef zq_mm_mul_ps
#undef zq_mm_sub_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#endif
#endif//__ARM_NEON


	/*WARNING: when scaling to larger images, it may visit the coordinate input[-1][?] or input[?][-1].
	so, you should allocate the input image with border.
	Make sure in_off_x >= 0 && in_off_y >=0 && in_off_x+in_rect_width<=in_W && in_off_y+in_rect_height<= in_H,
	this function will not check it.

	if (out_W > in_rect_w && (in_off_x == 0 || in_off_x + in_rect_w == W)
	|| out_H > in_rect_h && (in_off_y == 0 || in_off_y + in_rect_h == H))
	can_call_safeborder = false;
	*/
	void zq_cnn_resize_with_safeborder_32f_align0(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int in_off_x,
		int in_off_y,
		int in_rect_width,
		int in_rect_height,
		float* out_tensor4D_data,
		int out_H,
		int out_W,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	)
	{
		
		int* x0 = (int*)malloc(sizeof(int)*(out_W));
		int* x1 = (int*)malloc(sizeof(int)*(out_W));
		float* sx = (float*)malloc(sizeof(float)*(out_W));
		float src_H = in_rect_height;
		float src_W = in_rect_width;
		float w_step = 1.0f / (float)out_W*src_W;
		float h_step = 1.0f / (float)out_H*src_H;
		float coord_y_ini = 0.5f * h_step - 0.5f + (float)in_off_y;
		float coord_x_ini = 0.5f*w_step - 0.5f + (float)in_off_x;
		float x0_f, y0_f;
		int y0, y1;
		float sy;
		float coord_x, coord_y;
		const float* in_slice_ptr, *in_row0_ptr, *in_row1_ptr;
		float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		int n, h, w, c, cur_x0, cur_x1;
		float cur_sx, v00, dx0, result0, v10, dx1, result1, dy, sum;
		
		/*********** compute the map and weight begin ************/
		// coord_x
		coord_x = coord_x_ini;
		for (w = 0; w < out_W; w++, coord_x += w_step)
		{
			x0_f = floor(coord_x);
			x0[w] = (int)x0_f;
			x1[w] = x0[w] + 1;
			sx[w] = coord_x - x0_f;
			x0[w] *= in_pixelStep;
			x1[w] *= in_pixelStep;
		}

		/*********** compute the map and weight end ************/

		for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{

			coord_y = coord_y_ini;
			for (h = 0, out_row_ptr = out_slice_ptr;
				h < out_H;
				h++, coord_y += h_step, out_row_ptr += out_widthStep)
			{
				y0_f = floor(coord_y);
				y0 = (int)y0_f;
				y1 = y0 + 1;
				sy = coord_y - y0_f;

				in_row0_ptr = in_slice_ptr + y0*in_widthStep;
				in_row1_ptr = in_slice_ptr + y1*in_widthStep;

				for (w = 0, out_pix_ptr = out_row_ptr; w < out_W; w++, out_pix_ptr += out_pixelStep)
				{
					cur_sx = sx[w];
					for (c = 0, cur_x0 = x0[w], cur_x1 = x1[w]; c < in_C; c++, cur_x0++, cur_x1++)
					{
						v00 = *(in_row0_ptr + cur_x0);
						dx0 = *(in_row0_ptr + cur_x1) - v00;
						result0 = v00 + dx0 * cur_sx;
						v10 = *(in_row1_ptr + cur_x0);
						dx1 = *(in_row1_ptr + cur_x1) - v10;
						result1 = v10 + dx1 * cur_sx;
						dy = result1 - result0;
						sum = result0 + dy* sy;
						*(out_pix_ptr + c) = sum;
					}
				}
			}
		}

		free(x0);
		free(x1);
		free(sx);
	}

	/*WARNING: when scaling to larger images, it may visit the coordinate input[-1][?] or input[?][-1], or input[H][?], input[?][W].
	this function will clamp to input[0][?] or input[?][0], or input[H-1][?], input[?][W-1]
	*/
	void zq_cnn_resize_without_safeborder_32f_align0(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int in_off_x,
		int in_off_y,
		int in_rect_width,
		int in_rect_height,
		float* out_tensor4D_data,
		int out_H,
		int out_W,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	)
	{
		int* x0 = (int*)malloc(sizeof(int)*(out_W));
		int* x1 = (int*)malloc(sizeof(int)*(out_W));
		float* sx = (float*)malloc(sizeof(float)*(out_W));
		float src_H = in_rect_height;
		float src_W = in_rect_width;
		float w_step = 1.0f / (float)out_W*src_W;
		float h_step = 1.0f / (float)out_H*src_H;
		float coord_y_ini = 0.5f * h_step - 0.5f + (float)in_off_y;
		float coord_x_ini = 0.5f*w_step - 0.5f + (float)in_off_x;
		float x0_f, y0_f;
		int y0, y1;
		float sy;
		float coord_x, coord_y;
		const float* in_slice_ptr, *in_row0_ptr, *in_row1_ptr;
		float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		int n, h, w, c, cur_x0, cur_x1;
		float cur_sx, v00, dx0, result0, v10, dx1, result1, dy, sum;

		/*********** compute the map and weight begin ************/
		// coord_x
		coord_x = coord_x_ini;
		for (w = 0; w < out_W; w++, coord_x += w_step)
		{
			x0_f = floor(coord_x);
			x0[w] = (int)x0_f;
			x1[w] = x0[w] + 1;
			sx[w] = coord_x - x0_f;
			x0[w] = __min(src_W - 1, __max(0, x0[w]));
			x1[w] = __min(src_W - 1, __max(0, x1[w]));
			x0[w] *= in_pixelStep;
			x1[w] *= in_pixelStep;
		}

		/*********** compute the map and weight end ************/

		for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{

			coord_y = coord_y_ini;
			for (h = 0, out_row_ptr = out_slice_ptr;
				h < out_H;
				h++, coord_y += h_step, out_row_ptr += out_widthStep)
			{
				y0_f = floor(coord_y);
				y0 = (int)y0_f;
				y1 = y0 + 1;
				sy = coord_y - y0_f;
				y0 = __min(in_H, __max(0, y0));
				y1 = __min(in_H, __max(0, y1));

				in_row0_ptr = in_slice_ptr + y0*in_widthStep;
				in_row1_ptr = in_slice_ptr + y1*in_widthStep;

				for (w = 0, out_pix_ptr = out_row_ptr; w < out_W; w++, out_pix_ptr += out_pixelStep)
				{
					cur_sx = sx[w];
					for (c = 0, cur_x0 = x0[w], cur_x1 = x1[w]; c < in_C; c++, cur_x0++, cur_x1++)
					{
						v00 = *(in_row0_ptr + cur_x0);
						dx0 = *(in_row0_ptr + cur_x1) - v00;
						result0 = v00 + dx0 * cur_sx;
						v10 = *(in_row1_ptr + cur_x0);
						dx1 = *(in_row1_ptr + cur_x1) - v10;
						result1 = v10 + dx1 * cur_sx;
						dy = result1 - result0;
						sum = result0 + dy* sy;
						*(out_pix_ptr + c) = sum;
					}
				}
			}
		}

		free(x0);
		free(x1);
		free(sx);
	}


#if __ARM_NEON
#if __ARM_NEON_FP16
#define zq_base_type float16_t
		/*WARNING: when scaling to larger images, it may visit the coordinate input[-1][?] or input[?][-1].
		so, you should allocate the input image with border.
		Make sure in_off_x >= 0 && in_off_y >=0 && in_off_x+in_rect_width<=in_W && in_off_y+in_rect_height<= in_H,
		this function will not check it.

		if (out_W > in_rect_w && (in_off_x == 0 || in_off_x + in_rect_w == W)
		|| out_H > in_rect_h && (in_off_y == 0 || in_off_y + in_rect_h == H))
		can_call_safeborder = false;
		*/
		void zq_cnn_resize_with_safeborder_16f_align0(
			const zq_base_type* in_tensor4D_data,
			int in_N,
			int in_H,
			int in_W,
			int in_C,
			int in_pixelStep,
			int in_widthStep,
			int in_sliceStep,
			int in_off_x,
			int in_off_y,
			int in_rect_width,
			int in_rect_height,
			zq_base_type* out_tensor4D_data,
			int out_H,
			int out_W,
			int out_pixelStep,
			int out_widthStep,
			int out_sliceStep
		)
		{

			int* x0 = (int*)malloc(sizeof(int)*(out_W));
			int* x1 = (int*)malloc(sizeof(int)*(out_W));
			zq_base_type* sx = (zq_base_type*)malloc(sizeof(zq_base_type)*(out_W));
			zq_base_type src_H = in_rect_height;
			zq_base_type src_W = in_rect_width;
			zq_base_type w_step = 1.0f / (zq_base_type)out_W*src_W;
			zq_base_type h_step = 1.0f / (zq_base_type)out_H*src_H;
			zq_base_type coord_y_ini = 0.5f * h_step - 0.5f + (zq_base_type)in_off_y;
			zq_base_type coord_x_ini = 0.5f*w_step - 0.5f + (zq_base_type)in_off_x;
			zq_base_type x0_f, y0_f;
			int y0, y1;
			zq_base_type sy;
			zq_base_type coord_x, coord_y;
			const zq_base_type* in_slice_ptr, *in_row0_ptr, *in_row1_ptr;
			zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
			int n, h, w, c, cur_x0, cur_x1;
			zq_base_type cur_sx, v00, dx0, result0, v10, dx1, result1, dy, sum;

			/*********** compute the map and weight begin ************/
			// coord_x
			coord_x = coord_x_ini;
			for (w = 0; w < out_W; w++, coord_x += w_step)
			{
				x0_f = floor(coord_x);
				x0[w] = (int)x0_f;
				x1[w] = x0[w] + 1;
				sx[w] = coord_x - x0_f;
				x0[w] *= in_pixelStep;
				x1[w] *= in_pixelStep;
			}

			/*********** compute the map and weight end ************/

			for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
				n < in_N;
				n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
			{

				coord_y = coord_y_ini;
				for (h = 0, out_row_ptr = out_slice_ptr;
					h < out_H;
					h++, coord_y += h_step, out_row_ptr += out_widthStep)
				{
					y0_f = floor(coord_y);
					y0 = (int)y0_f;
					y1 = y0 + 1;
					sy = coord_y - y0_f;

					in_row0_ptr = in_slice_ptr + y0*in_widthStep;
					in_row1_ptr = in_slice_ptr + y1*in_widthStep;

					for (w = 0, out_pix_ptr = out_row_ptr; w < out_W; w++, out_pix_ptr += out_pixelStep)
					{
						cur_sx = sx[w];
						for (c = 0, cur_x0 = x0[w], cur_x1 = x1[w]; c < in_C; c++, cur_x0++, cur_x1++)
						{
							v00 = *(in_row0_ptr + cur_x0);
							dx0 = *(in_row0_ptr + cur_x1) - v00;
							result0 = v00 + dx0 * cur_sx;
							v10 = *(in_row1_ptr + cur_x0);
							dx1 = *(in_row1_ptr + cur_x1) - v10;
							result1 = v10 + dx1 * cur_sx;
							dy = result1 - result0;
							sum = result0 + dy* sy;
							*(out_pix_ptr + c) = sum;
						}
					}
				}
			}

			free(x0);
			free(x1);
			free(sx);
		}

		/*WARNING: when scaling to larger images, it may visit the coordinate input[-1][?] or input[?][-1], or input[H][?], input[?][W].
		this function will clamp to input[0][?] or input[?][0], or input[H-1][?], input[?][W-1]
		*/
		void zq_cnn_resize_without_safeborder_16f_align0(
			const zq_base_type* in_tensor4D_data,
			int in_N,
			int in_H,
			int in_W,
			int in_C,
			int in_pixelStep,
			int in_widthStep,
			int in_sliceStep,
			int in_off_x,
			int in_off_y,
			int in_rect_width,
			int in_rect_height,
			zq_base_type* out_tensor4D_data,
			int out_H,
			int out_W,
			int out_pixelStep,
			int out_widthStep,
			int out_sliceStep
		)
		{
			int* x0 = (int*)malloc(sizeof(int)*(out_W));
			int* x1 = (int*)malloc(sizeof(int)*(out_W));
			zq_base_type* sx = (zq_base_type*)malloc(sizeof(zq_base_type)*(out_W));
			zq_base_type src_H = in_rect_height;
			zq_base_type src_W = in_rect_width;
			zq_base_type w_step = 1.0f / (zq_base_type)out_W*src_W;
			zq_base_type h_step = 1.0f / (zq_base_type)out_H*src_H;
			zq_base_type coord_y_ini = 0.5f * h_step - 0.5f + (zq_base_type)in_off_y;
			zq_base_type coord_x_ini = 0.5f*w_step - 0.5f + (zq_base_type)in_off_x;
			zq_base_type x0_f, y0_f;
			int y0, y1;
			zq_base_type sy;
			zq_base_type coord_x, coord_y;
			const zq_base_type* in_slice_ptr, *in_row0_ptr, *in_row1_ptr;
			zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
			int n, h, w, c, cur_x0, cur_x1;
			zq_base_type cur_sx, v00, dx0, result0, v10, dx1, result1, dy, sum;

			/*********** compute the map and weight begin ************/
			// coord_x
			coord_x = coord_x_ini;
			for (w = 0; w < out_W; w++, coord_x += w_step)
			{
				x0_f = floor(coord_x);
				x0[w] = (int)x0_f;
				x1[w] = x0[w] + 1;
				sx[w] = coord_x - x0_f;
				x0[w] = __min(src_W - 1, __max(0, x0[w]));
				x1[w] = __min(src_W - 1, __max(0, x1[w]));
				x0[w] *= in_pixelStep;
				x1[w] *= in_pixelStep;
			}

			/*********** compute the map and weight end ************/

			for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
				n < in_N;
				n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
			{

				coord_y = coord_y_ini;
				for (h = 0, out_row_ptr = out_slice_ptr;
					h < out_H;
					h++, coord_y += h_step, out_row_ptr += out_widthStep)
				{
					y0_f = floor(coord_y);
					y0 = (int)y0_f;
					y1 = y0 + 1;
					sy = coord_y - y0_f;
					y0 = __min(in_H, __max(0, y0));
					y1 = __min(in_H, __max(0, y1));

					in_row0_ptr = in_slice_ptr + y0*in_widthStep;
					in_row1_ptr = in_slice_ptr + y1*in_widthStep;

					for (w = 0, out_pix_ptr = out_row_ptr; w < out_W; w++, out_pix_ptr += out_pixelStep)
					{
						cur_sx = sx[w];
						for (c = 0, cur_x0 = x0[w], cur_x1 = x1[w]; c < in_C; c++, cur_x0++, cur_x1++)
						{
							v00 = *(in_row0_ptr + cur_x0);
							dx0 = *(in_row0_ptr + cur_x1) - v00;
							result0 = v00 + dx0 * cur_sx;
							v10 = *(in_row1_ptr + cur_x0);
							dx1 = *(in_row1_ptr + cur_x1) - v10;
							result1 = v10 + dx1 * cur_sx;
							dy = result1 - result0;
							sum = result0 + dy* sy;
							*(out_pix_ptr + c) = sum;
						}
					}
				}
			}

			free(x0);
			free(x1);
			free(sx);
		}

#undef zq_base_type
#endif//__ARM_NEON_FP16
#endif//__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif