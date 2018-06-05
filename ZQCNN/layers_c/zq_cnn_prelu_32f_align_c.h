#ifndef _ZQ_CNN_PRELU_32F_ALIGN_C_H_
#define _ZQ_CNN_PRELU_32F_ALIGN_C_H_

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


#if defined(__cplusplus) || defined(c_plusplus) //跨平台定义方法
}
#endif
#endif