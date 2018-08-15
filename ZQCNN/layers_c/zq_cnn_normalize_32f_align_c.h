#ifndef _ZQ_CNN_NORMALIZE_32F_ALIGN_C_H_
#define _ZQ_CNN_NORMALIZE_32F_ALIGN_C_H_

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	void zq_cnn_normalize_32f_align0(
		int across_spatial,
		int channel_shared,
		float* in_tensor4D_data,	// in & out
		const float* scale_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float eps
	);


	void zq_cnn_normalize_not_across_spatial_32f_align128bit(
		int channel_shared,
		float* in_tensor4D_data,	// in & out
		const float* scale_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float eps
	);

	void zq_cnn_normalize_across_spatial_32f_align128bit(
		int channel_shared,
		float* in_tensor4D_data,	// in & out
		const float* scale_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float eps
	);

	void zq_cnn_normalize_not_across_spatial_32f_align256bit(
		int channel_shared,
		float* in_tensor4D_data,	// in & out
		const float* scale_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float eps
	);

	void zq_cnn_normalize_across_spatial_32f_align256bit(
		int channel_shared,
		float* in_tensor4D_data,	// in & out
		const float* scale_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float eps
	);

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif
