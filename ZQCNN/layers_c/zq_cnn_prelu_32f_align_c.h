#ifndef _ZQ_CNN_PRELU_32F_ALIGN_C_H_
#define _ZQ_CNN_PRELU_32F_ALIGN_C_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif


	void zq_cnn_prelu_32f_align0(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* slope_data
	);

	void zq_cnn_addbias_prelu_32f_align0(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* bias,
		const float* slope_data
	);

#if __ARM_NEON || ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	/*
	y = max(0,x)+a*min(0,x)
	*/
	void zq_cnn_prelu_32f_align128bit(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* slope_data
	);

	/*
	y = max(0,x)+a*min(0,x)
	*/
	void zq_cnn_addbias_prelu_32f_align128bit(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* bias,
		const float* slope_data
	);


	/*
	y = max(x,a*x)
	*/
	void zq_cnn_prelu_32f_align128bit_sure_slope_lessthan1(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* slope_data
	);

	/*
	y = max(x,a*x)
	*/
	void zq_cnn_addbias_prelu_32f_align128bit_sure_slope_lessthan1(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* bias,
		const float* slope_data
	);
#endif //__ARM_NEON || ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE

#if __ARM_NEON
#if __ARM_NEON_FP16
	void zq_cnn_prelu_16f_align0(
		float16_t* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* slope_data
	);

	void zq_cnn_addbias_prelu_16f_align0(
		float16_t* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* bias,
		const float16_t* slope_data
	);

	/*
	y = max(0,x)+a*min(0,x)
	*/
	void zq_cnn_prelu_16f_align128bit(
		float16_t* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* slope_data
	);

	/*
	y = max(0,x)+a*min(0,x)
	*/
	void zq_cnn_addbias_prelu_16f_align128bit(
		float16_t* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* bias,
		const float16_t* slope_data
	);


	/*
	y = max(x,a*x)
	*/
	void zq_cnn_prelu_16f_align128bit_sure_slope_lessthan1(
		float16_t* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* slope_data
	);

	/*
	y = max(x,a*x)
	*/
	void zq_cnn_addbias_prelu_16f_align128bit_sure_slope_lessthan1(
		float16_t* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* bias,
		const float16_t* slope_data
	);
#endif//__ARM_NEON_FP16
#endif//__ARM_NEON

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	/*
	y = max(0,x)+a*min(0,x)
	*/
	void zq_cnn_prelu_32f_align256bit(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* slope_data
	);

	/*
	y = max(0,x)+a*min(0,x)
	*/
	void zq_cnn_addbias_prelu_32f_align256bit(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* bias,
		const float* slope_data
	);

	/*
	y = max(x,a*x)
	*/
	void zq_cnn_prelu_32f_align256bit_sure_slope_lessthan1(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* slope_data
	);

	/*
	y = max(x,a*x)
	*/
	void zq_cnn_addbias_prelu_32f_align256bit_sure_slope_lessthan1(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* bias,
		const float* slope_data
	);

#endif//ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX


#if defined(__cplusplus) || defined(c_plusplus) //跨平台定义方法
}
#endif
#endif