#ifndef _ZQ_CNN_PRELU_NCHWC_H_
#define _ZQ_CNN_PRELU_NCHWC_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif


	void zq_cnn_prelu_nchwc1(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* slope_data
	);

	void zq_cnn_addbias_prelu_nchwc1(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* bias,
		const float* slope_data
	);

	/*
	y = max(x,a*x)
	*/
	void zq_cnn_prelu_nchwc1_sure_slope_lessthan1(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* slope_data
	);

	/*
	y = max(x,a*x)
	*/
	void zq_cnn_addbias_prelu_nchwc1_sure_slope_lessthan1(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* bias,
		const float* slope_data
	);

#if __ARM_NEON || ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	/*
	y = max(0,x)+a*min(0,x)
	*/
	void zq_cnn_prelu_nchwc4(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* slope_data
	);

	/*
	y = max(0,x)+a*min(0,x)
	*/
	void zq_cnn_addbias_prelu_nchwc4(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* bias,
		const float* slope_data
	);


	/*
	y = max(x,a*x)
	*/
	void zq_cnn_prelu_nchwc4_sure_slope_lessthan1(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* slope_data
	);

	/*
	y = max(x,a*x)
	*/
	void zq_cnn_addbias_prelu_nchwc4_sure_slope_lessthan1(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* bias,
		const float* slope_data
	);
#endif //__ARM_NEON || ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	/*
	y = max(0,x)+a*min(0,x)
	*/
	void zq_cnn_prelu_nchwc8(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* slope_data
	);

	/*
	y = max(0,x)+a*min(0,x)
	*/
	void zq_cnn_addbias_prelu_nchwc8(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* bias,
		const float* slope_data
	);

	/*
	y = max(x,a*x)
	*/
	void zq_cnn_prelu_nchwc8_sure_slope_lessthan1(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* slope_data
	);

	/*
	y = max(x,a*x)
	*/
	void zq_cnn_addbias_prelu_nchwc8_sure_slope_lessthan1(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* bias,
		const float* slope_data
	);

#endif//ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX


#if defined(__cplusplus) || defined(c_plusplus) //跨平台定义方法
}
#endif
#endif