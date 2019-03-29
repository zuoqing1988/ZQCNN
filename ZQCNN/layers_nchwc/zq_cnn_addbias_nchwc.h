#ifndef _ZQ_CNN_ADDBIAS_NCHWC_H_
#define _ZQ_CNN_ADDBIAS_NCHWC_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

#if __ARM_NEON

	void zq_cnn_addbias_nchwc1(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* bias_data
	);

	void zq_cnn_addbias_nchwc4(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* bias_data
	);	

#else

	void zq_cnn_addbias_nchwc1(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* bias_data
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_addbias_nchwc4(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* bias_data
	);

#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_addbias_nchwc8(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* bias_data
	);

#endif

#endif //__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif
