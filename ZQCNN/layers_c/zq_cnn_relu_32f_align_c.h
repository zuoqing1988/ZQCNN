#ifndef _ZQ_CNN_RELU_32F_ALIGN_C_H_
#define _ZQ_CNN_RELU_32F_ALIGN_C_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	

#if __ARM_NEON

	/*
	y = slope*min(0, x) + max(0, x)
	*/
	void zq_cnn_relu_32f_align0(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float slope
	);

	/*
	y = min(6, max(0, x))
	*/
	void zq_cnn_relu6_32f_align0(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep
	);

	/*
	y = slope*min(0,x)+max(0,x)
	*/
	void zq_cnn_relu_32f_align128bit(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float slope
	);

	/*
	y = min(6, max(0, x))
	*/
	void zq_cnn_relu6_32f_align128bit(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep
	);

#if __ARM_NEON_FP16
	/*
	y = slope*min(0, x) + max(0, x)
	*/
	void zq_cnn_relu_16f_align0(
		float16_t* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float16_t slope
	);

	/*
	y = min(6, max(0, x))
	*/
	void zq_cnn_relu6_16f_align0(
		float16_t* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep
	);

	/*
	y = slope*min(0,x)+max(0,x)
	*/
	void zq_cnn_relu_16f_align128bit(
		float16_t* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float16_t slope
	);

	/*
	y = min(6, max(0, x))
	*/
	void zq_cnn_relu6_16f_align128bit(
		float16_t* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep
	);

#endif//__ARM_NEON_FP16

#else

	/*
	y = slope*min(0, x) + max(0, x)
	*/
	void zq_cnn_relu_32f_align0(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float slope
	);

	/*
	y = min(6, max(0, x))
	*/
	void zq_cnn_relu6_32f_align0(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE

	/*
	y = slope*min(0,x)+max(0,x)
	*/
	void zq_cnn_relu_32f_align128bit(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float slope
	);

	/*
	y = min(6, max(0, x))
	*/
	void zq_cnn_relu6_32f_align128bit(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	/*
	y = slope*min(0,x)+max(0,x)
	*/
	void zq_cnn_relu_32f_align256bit(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float slope
	);

	/*
	y = min(6, max(0, x))
	*/
	void zq_cnn_relu6_32f_align256bit(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep
	);

#endif

#endif //__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) //跨平台定义方法
}
#endif
#endif