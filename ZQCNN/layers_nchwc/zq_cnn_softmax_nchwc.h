#ifndef _ZQ_CNN_SOFTMAX_NCHWC_H_
#define _ZQ_CNN_SOFTMAX_NCHWC_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	void zq_cnn_softmax_nchwc1_C(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep
	);

#if __ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)
	void zq_cnn_softmax_nchwc4_C(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep
	);
#endif //__ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_softmax_nchwc8_C(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep
	);
#endif

	void zq_cnn_softmax_nchwc1_H(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep
	);


	void zq_cnn_softmax_nchwc1_W(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep
	);


#if defined(__cplusplus) || defined(c_plusplus) //跨平台定义方法
}
#endif
#endif
