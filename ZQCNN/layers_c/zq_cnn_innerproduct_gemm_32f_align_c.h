#ifndef _ZQ_CNN_INNER_PRODUCT_GEMM_32F_ALIGN_C_H_
#define _ZQ_CNN_INNER_PRODUCT_GEMM_32F_ALIGN_C_H_

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	/*in_pixStep must be equal to filter_pixStep,
	and the aligned channels should be set to zero*/
	void zq_cnn_innerproduct_gemm_32f_align128bit_same_pixstep(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*in_pixStep must be equal to filter_pixStep,
	and the aligned channels should be set to zero*/
	void zq_cnn_innerproduct_gemm_32f_align256bit_same_pixstep(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*in_pixStep can be different with filter_pixStep,
	and the aligned channels should be set to zero*/
	void zq_cnn_innerproduct_gemm_32f_same_or_notsame_pixstep(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*in_pixStep must be equal to filter_pixStep,
	and the aligned channels should be set to zero*/
	void zq_cnn_innerproduct_gemm_32f_align128bit_same_pixstep_batch(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*in_pixStep must be equal to filter_pixStep,
	and the aligned channels should be set to zero*/
	void zq_cnn_innerproduct_gemm_32f_align256bit_same_pixstep_batch(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif

#endif