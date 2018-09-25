#ifndef _ZQ_CNN_REDUCTION_32F_ALIGN_C_H_
#define _ZQ_CNN_REDUCTION_32F_ALIGN_C_H_

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
