#ifndef _ZQ_CNN_SQRT_32F_ALIGN_C_H_
#define _ZQ_CNN_SQRT_32F_ALIGN_C_H_
#include "../ZQ_CNN_CompileConfig.h"

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

#if __ARM_NEON
#if __ARM_NEON_FP16
	void zq_cnn_sqrt_16f_align0(
		float16_t* data,
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);
#endif//__ARM_NEON_FP16
#endif//__ARM_NEON

	void zq_cnn_sqrt_32f_align0(
		float* data,
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif
