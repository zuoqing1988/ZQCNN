#include <stdio.h>
#include <math.h>
#include "../ZQ_CNN_CompileConfig.h"

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif


	void zq_cnn_sqrt_32f_align0(
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
		float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
		for (n = 0, slice_ptr = data; n < N; n++, slice_ptr+=sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < H; h++, row_ptr += widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < W; w++, pix_ptr += pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr; c < C; c++, c_ptr++)
					{
						if (c_ptr[0] < 0)
							c_ptr[0] = 0;
						else
							c_ptr[0] = sqrt(c_ptr[0]);
					}
				}
			}
		}
	}

#if __ARM_NEON
#if __ARM_NEON_FP16
#define zq_base_type float16_t
	void zq_cnn_sqrt_16f_align0(
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
		zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
		for (n = 0, slice_ptr = data; n < N; n++, slice_ptr += sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < H; h++, row_ptr += widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < W; w++, pix_ptr += pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr; c < C; c++, c_ptr++)
					{
						if (c_ptr[0] < 0)
							c_ptr[0] = 0;
						else
							c_ptr[0] = sqrt(c_ptr[0]);
					}
				}
			}
		}
	}
#undef zq_base_type
#endif
#endif//__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
