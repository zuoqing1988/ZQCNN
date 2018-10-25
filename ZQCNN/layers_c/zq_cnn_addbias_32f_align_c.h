#ifndef _ZQ_CNN_ADD_BIAS_32F_ALIGN_C_H_
#define _ZQ_CNN_ADD_BIAS_32F_ALIGN_C_H_
#include "..\ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

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

	void zq_cnn_addbias_32f_align0_omp(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* bias_data,
		int thread_count
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

	void zq_cnn_addbias_32f_align128bit_omp(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_alignPixelStep,
		int in_widthStep,
		int in_SliceStep,
		const float* bias_data,
		int thread_count
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

	void zq_cnn_addbias_32f_align256bit_omp(
		float* in_tensor4D_data,	// in & out
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* bias_data,
		int thread_count
	);
#endif

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif
