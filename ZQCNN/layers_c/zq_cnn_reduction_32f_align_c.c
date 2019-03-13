#include <stdio.h>
#include "../ZQ_CNN_CompileConfig.h"

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif


	void zq_cnn_reduction_sum_32f_align0(
		const float* in_data,
		int N,
		int H,
		int W,
		int C,
		int axis,
		int keepdims,
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
		const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_c_ptr;
		float* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_c_ptr;
		if (keepdims == 0)
		{
			float sum = 0;
			for (n = 0, in_slice_ptr = in_data; n < N; n++, in_slice_ptr += in_sliceStep)
			{
				for (h = 0, in_row_ptr = in_slice_ptr;	h < H;	h++, in_row_ptr += in_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr; w < W;	w++, in_pix_ptr += in_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr; c < C; c++, in_c_ptr++)
						{
							sum += in_c_ptr[0];
						}
					}
				}
			}
			out_data[0] = sum;
		}
		else
		{
			switch (axis)
			{
			case 0:
			{
				for (h = 0, in_row_ptr = in_data, out_row_ptr = out_data;
					h < H;
					h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
						w < W;
						w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;
							c < C;
							c++, in_c_ptr++, out_c_ptr++)
						{
							out_slice_ptr = out_c_ptr;
							float sum = 0;
							for (n = 0, in_slice_ptr = in_c_ptr; n < N; n++, in_slice_ptr += in_sliceStep)
								sum += in_slice_ptr[0];

							out_slice_ptr[0] = sum;
						}
					}
				}
			}
			break;
			case 1:
			{
				for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
					n < N;
					n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
				{
					for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
						h < H;
						h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
					{
						for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
							w < W;
							w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
						{
							out_c_ptr = out_pix_ptr;
							float sum = 0;
							for (c = 0, in_c_ptr = in_pix_ptr; c < C; c++, in_c_ptr++)
								sum += in_c_ptr[0];
							out_c_ptr[0] = sum;
						}
					}
				}
			}
			break;
			case 2:
			{
				for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
					n < N;
					n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
				{
					for (w = 0, in_pix_ptr = in_slice_ptr, out_pix_ptr = out_slice_ptr;
						w < W;
						w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;
							c < C;
							c++, in_c_ptr++, out_c_ptr++)
						{
							out_row_ptr = out_c_ptr;
							float sum = 0;
							for (h = 0, in_row_ptr = in_c_ptr; h < H; h++, in_row_ptr += in_widthStep)
								sum += in_row_ptr[0];
							out_row_ptr[0] = sum;
						}
					}
				}
			}
			break;
			case 3:
			{
				for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
					n < N;
					n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
				{
					for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
						h < H;
						h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
					{
						for (c = 0, in_c_ptr = in_row_ptr, out_c_ptr = out_row_ptr;
							c < C;
							c++, in_c_ptr++, out_c_ptr++)
						{
							out_pix_ptr = out_c_ptr;
							float sum = 0;
							for (w = 0, in_pix_ptr = in_c_ptr; w < W; w++, in_pix_ptr += in_pixelStep)
								sum += in_pix_ptr[0];
							out_pix_ptr[0] = sum;
						}
					}
				}
			}
			break;
			}
		}
	}

	void zq_cnn_reduction_mean_32f_align0(
		const float* in_data,
		int N,
		int H,
		int W,
		int C,
		int axis,
		int keepdims,
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
		const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_c_ptr;
		float* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_c_ptr;

		if (keepdims == 0)
		{
			float sum = 0;
			for (n = 0, in_slice_ptr = in_data; n < N; n++, in_slice_ptr += in_sliceStep)
			{
				for (h = 0, in_row_ptr = in_slice_ptr; h < H; h++, in_row_ptr += in_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr; w < W; w++, in_pix_ptr += in_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr; c < C; c++, in_c_ptr++)
						{
							sum += in_c_ptr[0];
						}
					}
				}
			}
			out_data[0] = sum / (N*H*W*C);
		}
		else
		{
			float weight = 1.0f;
			if (axis == 0)
				weight /= N;
			else if (axis == 1)
				weight /= C;
			else if (axis == 2)
				weight /= H;
			else if (axis == 3)
				weight /= W;
			switch (axis)
			{
			case 0:
			{
				for (h = 0, in_row_ptr = in_data, out_row_ptr = out_data;
					h < H;
					h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
						w < W;
						w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;
							c < C;
							c++, in_c_ptr++, out_c_ptr++)
						{
							out_slice_ptr = out_c_ptr;
							float sum = 0;
							for (n = 0, in_slice_ptr = in_c_ptr; n < N; n++, in_slice_ptr += in_sliceStep)
								sum += in_slice_ptr[0];

							out_slice_ptr[0] = sum *weight;
						}
					}
				}
			}
			break;
			case 1:
			{
				for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
					n < N;
					n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
				{
					for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
						h < H;
						h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
					{
						for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
							w < W;
							w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
						{
							out_c_ptr = out_pix_ptr;
							float sum = 0;
							for (c = 0, in_c_ptr = in_pix_ptr; c < C; c++, in_c_ptr++)
								sum += in_c_ptr[0];
							out_c_ptr[0] = sum *weight;
						}
					}
				}
			}
			break;
			case 2:
			{
				for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
					n < N;
					n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
				{
					for (w = 0, in_pix_ptr = in_slice_ptr, out_pix_ptr = out_slice_ptr;
						w < W;
						w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;
							c < C;
							c++, in_c_ptr++, out_c_ptr++)
						{
							out_row_ptr = out_c_ptr;
							float sum = 0;
							for (h = 0, in_row_ptr = in_c_ptr; h < H; h++, in_row_ptr += in_widthStep)
								sum += in_row_ptr[0];
							out_row_ptr[0] = sum*weight;
						}
					}
				}
			}
			break;
			case 3:
			{
				for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
					n < N;
					n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
				{
					for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
						h < H;
						h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
					{
						for (c = 0, in_c_ptr = in_row_ptr, out_c_ptr = out_row_ptr;
							c < C;
							c++, in_c_ptr++, out_c_ptr++)
						{
							out_pix_ptr = out_c_ptr;
							float sum = 0;
							for (w = 0, in_pix_ptr = in_c_ptr; w < W; w++, in_pix_ptr += in_pixelStep)
								sum += in_pix_ptr[0];
							out_pix_ptr[0] = sum*weight;
						}
					}
				}
			}
			break;
			}
		}
	}

#if __ARM_NEON
#if __ARM_NEON_FP16
#define zq_base_type float16_t
	void zq_cnn_reduction_sum_32f_align0(
		const zq_base_type* in_data,
		int N,
		int H,
		int W,
		int C,
		int axis,
		int keepdims,
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
		const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_c_ptr;
		zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_c_ptr;
		if (keepdims == 0)
		{
			zq_base_type sum = 0;
			for (n = 0, in_slice_ptr = in_data; n < N; n++, in_slice_ptr += in_sliceStep)
			{
				for (h = 0, in_row_ptr = in_slice_ptr; h < H; h++, in_row_ptr += in_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr; w < W; w++, in_pix_ptr += in_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr; c < C; c++, in_c_ptr++)
						{
							sum += in_c_ptr[0];
						}
					}
				}
			}
			out_data[0] = sum;
		}
		else
		{
			switch (axis)
			{
			case 0:
			{
				for (h = 0, in_row_ptr = in_data, out_row_ptr = out_data;
					h < H;
					h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
						w < W;
						w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;
							c < C;
							c++, in_c_ptr++, out_c_ptr++)
						{
							out_slice_ptr = out_c_ptr;
							zq_base_type sum = 0;
							for (n = 0, in_slice_ptr = in_c_ptr; n < N; n++, in_slice_ptr += in_sliceStep)
								sum += in_slice_ptr[0];

							out_slice_ptr[0] = sum;
						}
					}
				}
			}
			break;
			case 1:
			{
				for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
					n < N;
					n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
				{
					for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
						h < H;
						h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
					{
						for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
							w < W;
							w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
						{
							out_c_ptr = out_pix_ptr;
							zq_base_type sum = 0;
							for (c = 0, in_c_ptr = in_pix_ptr; c < C; c++, in_c_ptr++)
								sum += in_c_ptr[0];
							out_c_ptr[0] = sum;
						}
					}
				}
			}
			break;
			case 2:
			{
				for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
					n < N;
					n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
				{
					for (w = 0, in_pix_ptr = in_slice_ptr, out_pix_ptr = out_slice_ptr;
						w < W;
						w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;
							c < C;
							c++, in_c_ptr++, out_c_ptr++)
						{
							out_row_ptr = out_c_ptr;
							zq_base_type sum = 0;
							for (h = 0, in_row_ptr = in_c_ptr; h < H; h++, in_row_ptr += in_widthStep)
								sum += in_row_ptr[0];
							out_row_ptr[0] = sum;
						}
					}
				}
			}
			break;
			case 3:
			{
				for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
					n < N;
					n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
				{
					for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
						h < H;
						h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
					{
						for (c = 0, in_c_ptr = in_row_ptr, out_c_ptr = out_row_ptr;
							c < C;
							c++, in_c_ptr++, out_c_ptr++)
						{
							out_pix_ptr = out_c_ptr;
							zq_base_type sum = 0;
							for (w = 0, in_pix_ptr = in_c_ptr; w < W; w++, in_pix_ptr += in_pixelStep)
								sum += in_pix_ptr[0];
							out_pix_ptr[0] = sum;
						}
					}
				}
			}
			break;
			}
		}
	}

	void zq_cnn_reduction_mean_32f_align0(
		const zq_base_type* in_data,
		int N,
		int H,
		int W,
		int C,
		int axis,
		int keepdims,
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
		const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_c_ptr;
		zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_c_ptr;

		if (keepdims == 0)
		{
			zq_base_type sum = 0;
			for (n = 0, in_slice_ptr = in_data; n < N; n++, in_slice_ptr += in_sliceStep)
			{
				for (h = 0, in_row_ptr = in_slice_ptr; h < H; h++, in_row_ptr += in_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr; w < W; w++, in_pix_ptr += in_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr; c < C; c++, in_c_ptr++)
						{
							sum += in_c_ptr[0];
						}
					}
				}
			}
			out_data[0] = sum / (N*H*W*C);
		}
		else
		{
			zq_base_type weight = 1.0f;
			if (axis == 0)
				weight /= N;
			else if (axis == 1)
				weight /= C;
			else if (axis == 2)
				weight /= H;
			else if (axis == 3)
				weight /= W;
			switch (axis)
			{
			case 0:
			{
				for (h = 0, in_row_ptr = in_data, out_row_ptr = out_data;
					h < H;
					h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
						w < W;
						w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;
							c < C;
							c++, in_c_ptr++, out_c_ptr++)
						{
							out_slice_ptr = out_c_ptr;
							zq_base_type sum = 0;
							for (n = 0, in_slice_ptr = in_c_ptr; n < N; n++, in_slice_ptr += in_sliceStep)
								sum += in_slice_ptr[0];

							out_slice_ptr[0] = sum *weight;
						}
					}
				}
			}
			break;
			case 1:
			{
				for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
					n < N;
					n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
				{
					for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
						h < H;
						h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
					{
						for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
							w < W;
							w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
						{
							out_c_ptr = out_pix_ptr;
							zq_base_type sum = 0;
							for (c = 0, in_c_ptr = in_pix_ptr; c < C; c++, in_c_ptr++)
								sum += in_c_ptr[0];
							out_c_ptr[0] = sum *weight;
						}
					}
				}
			}
			break;
			case 2:
			{
				for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
					n < N;
					n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
				{
					for (w = 0, in_pix_ptr = in_slice_ptr, out_pix_ptr = out_slice_ptr;
						w < W;
						w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;
							c < C;
							c++, in_c_ptr++, out_c_ptr++)
						{
							out_row_ptr = out_c_ptr;
							zq_base_type sum = 0;
							for (h = 0, in_row_ptr = in_c_ptr; h < H; h++, in_row_ptr += in_widthStep)
								sum += in_row_ptr[0];
							out_row_ptr[0] = sum*weight;
						}
					}
				}
			}
			break;
			case 3:
			{
				for (n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
					n < N;
					n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
				{
					for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
						h < H;
						h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
					{
						for (c = 0, in_c_ptr = in_row_ptr, out_c_ptr = out_row_ptr;
							c < C;
							c++, in_c_ptr++, out_c_ptr++)
						{
							out_pix_ptr = out_c_ptr;
							zq_base_type sum = 0;
							for (w = 0, in_pix_ptr = in_c_ptr; w < W; w++, in_pix_ptr += in_pixelStep)
								sum += in_pix_ptr[0];
							out_pix_ptr[0] = sum*weight;
						}
					}
				}
			}
			break;
			}
		}
	}
#undef zq_base_type
#endif//__ARM_NEON_FP16
#endif//__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
