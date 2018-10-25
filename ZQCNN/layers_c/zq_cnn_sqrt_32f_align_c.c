#include <stdio.h>
#include <math.h>
#include "..\ZQ_CNN_CompileConfig.h"
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

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
