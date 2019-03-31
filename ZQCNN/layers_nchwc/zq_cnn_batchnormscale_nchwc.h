#ifndef _ZQ_CNN_BATCH_NORM_SCALE_NCHWC_H_
#define _ZQ_CNN_BATCH_NORM_SCALE_NCHWC_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	/*
	a = bias - scale * mean / sqrt(var+eps)
	b = scale / sqrt(var+eps)
	value = b * value + a
	*/
	void zq_cnn_batchnormscale_mean_var_scale_bias_nchwc1(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* mean_data,
		const float* var_data,
		const float* scale_data,
		const float* bias_data,
		const float eps
	);



	/*
	a = - mean / sqrt(var+eps)
	b = 1 / sqrt(var+eps)
	value = b * value + a
	*/
	void zq_cnn_batchnorm_mean_var_nchwc1(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* mean_data,
		const float* var_data,
		const float eps
	);



	/*
	value = scale*value+bias
	*/
	void zq_cnn_scale_nchwc1(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* scale_data,
		const float* bias_data	// bias can be NULL
	);


	/*
	a = bias - slope * mean / sqrt(var+eps)
	b = slope / sqrt(var+eps)
	value = b * value + a
	OR
	a = - mean / sqrt(var+eps)
	b = 1 / sqrt(var+eps)
	value = b * value + a
	*/
	void zq_cnn_batchnorm_b_a_nchwc1(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* b_data,
		const float* a_data
	);

#if __ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)

	/*
	a = bias - scale * mean / sqrt(var+eps)
	b = scale / sqrt(var+eps)
	value = b * value + a
	*/
	void zq_cnn_batchnormscale_mean_var_scale_bias_nchwc4(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* mean_data,
		const float* var_data,
		const float* scale_data,
		const float* bias_data,
		const float eps
	);


	/*
	a = - mean / sqrt(var+eps)
	b = 1 / sqrt(var+eps)
	value = b * value + a
	*/
	void zq_cnn_batchnorm_mean_var_nchwc4(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* mean_data,
		const float* var_data,
		const float eps
	);


	/*
	value = scale*value+bias
	*/
	void zq_cnn_scale_nchwc4(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* scale_data,
		const float* bias_data	// bias can be NULL
	);



	/*
	a = bias - slope * mean / sqrt(var+eps)
	b = slope / sqrt(var+eps)
	value = b * value + a
	OR
	a = - mean / sqrt(var+eps)
	b = 1 / sqrt(var+eps)
	value = b * value + a
	*/
	void zq_cnn_batchnorm_b_a_nchwc4(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* b_data,
		const float* a_data
	);

#endif //__ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	/*
	a = bias - scale * mean / sqrt(var+eps)
	b = scale / sqrt(var+eps)
	value = b * value + a
	*/
	void zq_cnn_batchnormscale_mean_var_scale_bias_nchwc8(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* mean_data,
		const float* var_data,
		const float* scale_data,
		const float* bias_data,
		const float eps
	);


	/*
	a = - mean / sqrt(var+eps)
	b = 1 / sqrt(var+eps)
	value = b * value + a
	*/
	void zq_cnn_batchnorm_mean_var_nchwc8(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* mean_data,
		const float* var_data,
		const float eps
	);



	/*
	value = scale*value+bias
	*/
	void zq_cnn_scale_nchwc8(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* scale_data,
		const float* bias_data	// bias can be NULL
	);



	/*
	a = bias - slope * mean / sqrt(var+eps)
	b = slope / sqrt(var+eps)
	value = b * value + a
	OR
	a = - mean / sqrt(var+eps)
	b = 1 / sqrt(var+eps)
	value = b * value + a
	*/
	void zq_cnn_batchnorm_b_a_nchwc8(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* b_data,
		const float* a_data
	);


#endif	

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif


#endif
