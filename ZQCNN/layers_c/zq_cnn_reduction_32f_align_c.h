#ifndef _ZQ_CNN_REDUCTION_32F_ALIGN_C_H_
#define _ZQ_CNN_REDUCTION_32F_ALIGN_C_H_
#include "../ZQ_CNN_CompileConfig.h"

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

#if __ARM_NEON
#if __ARM_NEON_FP16
	void zq_cnn_reduction_sum_16f_align0(
		const float16_t* in_data,
		int N,
		int H,
		int W,
		int C,
		int axis,
		int keep_dims,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float16_t* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_reduction_mean_16f_align0(
		const float16_t* in_data,
		int N,
		int H,
		int W,
		int C,
		int axis,
		int keep_dims,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float16_t* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
#endif//__ARM_NEON_FP16
#endif//__ARM_NEON

	void zq_cnn_reduction_sum_32f_align0(
		const float* in_data,
		int N,
		int H,
		int W,
		int C,
		int axis,
		int keep_dims,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_reduction_mean_32f_align0(
		const float* in_data,
		int N,
		int H,
		int W,
		int C,
		int axis,
		int keep_dims,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif
