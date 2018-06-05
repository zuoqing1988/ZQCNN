#ifndef _ZQ_CNN_LRN_32F_ALIGN_C_H_
#define _ZQ_CNN_LRN_32F_ALIGN_C_H_

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	void zq_cnn_lrn_across_channels_32f_align0(
		int local_size,		// must be odd number
		float alpha,
		float beta,
		float k,								
		const float* in_tensor4D_data,	
		int N,
		int H,
		int W,
		int C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_tensor4D_data,
		int out_pixStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_lrn_across_channels_32f_align128bit(
		int local_size,		// must be odd number
		float alpha,
		float beta,
		float k,								
		const float* in_tensor4D_data,
		int N,
		int H,
		int W,
		int C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_tensor4D_data,
		int out_pixStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_lrn_across_channels_32f_align256bit(
		int local_size,		// must be odd number
		float alpha,
		float beta,
		float k,								
		const float* in_tensor4D_data,
		int N,
		int H,
		int W,
		int C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_tensor4D_data,
		int out_pixStep,
		int out_widthStep,
		int out_sliceStep
	);

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif
