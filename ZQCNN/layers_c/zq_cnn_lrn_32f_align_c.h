#ifndef _ZQ_CNN_LRN_32F_ALIGN_C_H_
#define _ZQ_CNN_LRN_32F_ALIGN_C_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif



#if __ARM_NEON
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

#if __ARM_NEON_FP16
	void zq_cnn_lrn_across_channels_16f_align0(
		int local_size,		// must be odd number
		float16_t alpha,
		float16_t beta,
		float16_t k,
		const float16_t* in_tensor4D_data,
		int N,
		int H,
		int W,
		int C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float16_t* out_tensor4D_data,
		int out_pixStep,
		int out_widthStep,
		int out_sliceStep
	);

#endif//__ARM_NEON_FP16

#else

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

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
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
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
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
#endif

#endif //__ARM_NEON


#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif
