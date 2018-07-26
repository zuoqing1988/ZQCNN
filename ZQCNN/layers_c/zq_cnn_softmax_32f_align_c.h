#ifndef _ZQ_CNN_SOFTMAX_32F_ALIGN_C_H_
#define _ZQ_CNN_SOFTMAX_32F_ALIGN_C_H_

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	void zq_cnn_softmax_32f_align0_C(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep
	);

	void zq_cnn_softmax_32f_align128bit_C(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep
	);

	void zq_cnn_softmax_32f_align256bit_C(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep
	);

	void zq_cnn_softmax_32f_align0_H(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep
	);


	void zq_cnn_softmax_32f_align0_W(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep
	);


#if defined(__cplusplus) || defined(c_plusplus) //跨平台定义方法
}
#endif
#endif
