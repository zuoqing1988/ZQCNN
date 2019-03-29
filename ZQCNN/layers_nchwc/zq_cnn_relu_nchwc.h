#ifndef _ZQ_CNN_RELU_NCHWC_H_
#define _ZQ_CNN_RELU_NCHWC_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif



#if __ARM_NEON

	/*
	y = slope*min(0, x) + max(0, x)
	*/
	void zq_cnn_relu_nchwc1(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		float slope
	);

	/*
	y = slope*min(0,x)+max(0,x)
	*/
	void zq_cnn_relu_nchwc4(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		float slope
	);

#else

	/*
	y = slope*min(0, x) + max(0, x)
	*/
	void zq_cnn_relu_nchwc1(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		float slope
	);


#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	/*
	y = slope*min(0,x)+max(0,x)
	*/
	void zq_cnn_relu_nchwc4(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		float slope
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	/*
	y = slope*min(0,x)+max(0,x)
	*/
	void zq_cnn_relu_nchwc8(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		float slope
	);

#endif

#endif //__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) //跨平台定义方法
}
#endif
#endif