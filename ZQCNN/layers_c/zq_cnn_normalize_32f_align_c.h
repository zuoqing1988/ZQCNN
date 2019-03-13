#ifndef _ZQ_CNN_NORMALIZE_32F_ALIGN_C_H_
#define _ZQ_CNN_NORMALIZE_32F_ALIGN_C_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif


#if __ARM_NEON

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

#if __ARM_NEON_FP16
	void zq_cnn_normalize_16f_align0(
		int across_spatial,
		int channel_shared,
		float16_t* in_tensor4D_data,	// in & out
		const float16_t* scale_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t eps
	);

	void zq_cnn_normalize_not_across_spatial_16f_align128bit(
		int channel_shared,
		float16_t* in_tensor4D_data,	// in & out
		const float16_t* scale_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t eps
	);

	void zq_cnn_normalize_across_spatial_16f_align128bit(
		int channel_shared,
		float16_t* in_tensor4D_data,	// in & out
		const float16_t* scale_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t eps
	);
#endif//__ARM_NEON_FP16

#else

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

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
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
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
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
#endif

#endif //__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif
