#ifndef _ZQ_CNN_BATCH_NORM_SCALE_32F_ALIGN_C_H_
#define _ZQ_CNN_BATCH_NORM_SCALE_32F_ALIGN_C_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif



#if __ARM_NEON

	/*
	a = bias - scale * mean / sqrt(var+eps)
	b = scale / sqrt(var+eps)
	value = b * value + a
	*/
	void zq_cnn_batchnormscale_32f_mean_var_scale_bias_align0(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
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
	void zq_cnn_batchnorm_32f_mean_var_align0(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float* mean_data,
		const float* var_data,
		const float eps
	);



	/*
	value = scale*value+bias
	*/
	void zq_cnn_scale_32f_align0(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
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
	void zq_cnn_batchnorm_32f_b_a_align0(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float* b_data,
		const float* a_data
	);

	/*
	a = bias - scale * mean / sqrt(var+eps)
	b = scale / sqrt(var+eps)
	value = b * value + a
	*/
	void zq_cnn_batchnormscale_32f_mean_var_scale_bias_align128bit(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
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
	void zq_cnn_batchnorm_32f_mean_var_align128bit(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float* mean_data,
		const float* var_data,
		const float eps
	);


	/*
	value = scale*value+bias
	*/
	void zq_cnn_scale_32f_align128bit(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
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
	void zq_cnn_batchnorm_32f_b_a_align128bit(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float* b_data,
		const float* a_data
	);

#if __ARM_NEON_FP16
	/*
	a = bias - scale * mean / sqrt(var+eps)
	b = scale / sqrt(var+eps)
	value = b * value + a
	*/
	void zq_cnn_batchnormscale_16f_mean_var_scale_bias_align0(
		float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* mean_data,
		const float16_t* var_data,
		const float16_t* scale_data,
		const float16_t* bias_data,
		const float16_t eps
	);



	/*
	a = - mean / sqrt(var+eps)
	b = 1 / sqrt(var+eps)
	value = b * value + a
	*/
	void zq_cnn_batchnorm_16f_mean_var_align0(
		float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* mean_data,
		const float16_t* var_data,
		const float16_t eps
	);



	/*
	value = scale*value+bias
	*/
	void zq_cnn_scale_16f_align0(
		float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* scale_data,
		const float16_t* bias_data	// bias can be NULL
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
	void zq_cnn_batchnorm_16f_b_a_align0(
		float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* b_data,
		const float16_t* a_data
	);

	/*
	a = bias - scale * mean / sqrt(var+eps)
	b = scale / sqrt(var+eps)
	value = b * value + a
	*/
	void zq_cnn_batchnormscale_16f_mean_var_scale_bias_align128bit(
		float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* mean_data,
		const float16_t* var_data,
		const float16_t* scale_data,
		const float16_t* bias_data,
		const float16_t eps
	);



	/*
	a = - mean / sqrt(var+eps)
	b = 1 / sqrt(var+eps)
	value = b * value + a
	*/
	void zq_cnn_batchnorm_16f_mean_var_align128bit(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* mean_data,
		const float16_t* var_data,
		const float16_t eps
	);


	/*
	value = scale*value+bias
	*/
	void zq_cnn_scale_16f_align128bit(
		float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* scale_data,
		const float16_t* bias_data	// bias can be NULL
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
	void zq_cnn_batchnorm_16f_b_a_align128bit(
		float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* b_data,
		const float16_t* a_data
	);
#endif

#else

	/*
	a = bias - scale * mean / sqrt(var+eps)
	b = scale / sqrt(var+eps)
	value = b * value + a
	*/
	void zq_cnn_batchnormscale_32f_mean_var_scale_bias_align0(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
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
	void zq_cnn_batchnorm_32f_mean_var_align0(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float* mean_data,
		const float* var_data,
		const float eps
	);



	/*
	value = scale*value+bias
	*/
	void zq_cnn_scale_32f_align0(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
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
	void zq_cnn_batchnorm_32f_b_a_align0(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float* b_data,
		const float* a_data
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	/*
	a = bias - scale * mean / sqrt(var+eps)
	b = scale / sqrt(var+eps)
	value = b * value + a
	*/
	void zq_cnn_batchnormscale_32f_mean_var_scale_bias_align128bit(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
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
	void zq_cnn_batchnorm_32f_mean_var_align128bit(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float* mean_data,
		const float* var_data,
		const float eps
	);


	/*
	value = scale*value+bias
	*/
	void zq_cnn_scale_32f_align128bit(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
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
	void zq_cnn_batchnorm_32f_b_a_align128bit(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float* b_data,
		const float* a_data
	);


#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	/*
	a = bias - scale * mean / sqrt(var+eps)
	b = scale / sqrt(var+eps)
	value = b * value + a
	*/
	void zq_cnn_batchnormscale_32f_mean_var_scale_bias_align256bit(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
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
	void zq_cnn_batchnorm_32f_mean_var_align256bit(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float* mean_data,
		const float* var_data,
		const float eps
	);



	/*
	value = scale*value+bias
	*/
	void zq_cnn_scale_32f_align256bit(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
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
	void zq_cnn_batchnorm_32f_b_a_align256bit(
		float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixStep,
		int in_widthStep,
		int in_sliceStep,
		const float* b_data,
		const float* a_data
	);


#endif	

#endif //__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif


#endif
