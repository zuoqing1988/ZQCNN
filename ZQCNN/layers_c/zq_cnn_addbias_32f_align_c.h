#ifndef _ZQ_CNN_ADD_BIAS_32F_ALIGN_C_H_
#define _ZQ_CNN_ADD_BIAS_32F_ALIGN_C_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif



#if __ARM_NEON

	void zq_cnn_addbias_32f_align0(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* bias_data
	);

	void zq_cnn_addbias_32f_align128bit(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_alignPixelStep,
		int in_widthStep,
		int in_SliceStep,
		const float* bias_data

#if __ARM_NEON_FP16
		void zq_cnn_addbias_16f_align0(
			float16_t* in_tensor4D_data,	// in & out
			int in_N,
			int in_H,
			int in_W,
			int in_C,
			int in_pixelStep,
			int in_widthStep,
			int in_sliceStep,
			const float16_t* bias_data
		);

	void zq_cnn_addbias_16f_align128bit(
		float16_t* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_alignPixelStep,
		int in_widthStep,
		int in_SliceStep,
		const float16_t* bias_data
#endif
	);
#else

	void zq_cnn_addbias_32f_align0(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* bias_data
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_addbias_32f_align128bit(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_alignPixelStep,
		int in_widthStep,
		int in_SliceStep,
		const float* bias_data
	);

#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_addbias_32f_align256bit(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* bias_data
	);

#endif

#endif //__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif
