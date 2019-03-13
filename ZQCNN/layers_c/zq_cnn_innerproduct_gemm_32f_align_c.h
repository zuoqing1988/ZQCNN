#ifndef _ZQ_CNN_INNER_PRODUCT_GEMM_32F_ALIGN_C_H_
#define _ZQ_CNN_INNER_PRODUCT_GEMM_32F_ALIGN_C_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

#if __ARM_NEON

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
		int out_sliceStep,
		void** buffer,
		__int64 *buffer_len
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
		int out_sliceStep,
		void** buffer,
		__int64 *buffer_len
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
		int out_sliceStep,
		void** buffer,
		__int64 *buffer_len
	);

#if __ARM_NEON_FP16
	/*in_pixStep must be equal to filter_pixStep,
	and the aligned channels should be set to zero*/
	void zq_cnn_innerproduct_gemm_16f_align128bit_same_pixstep(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		void** buffer,
		__int64 *buffer_len
	);

	/*in_pixStep can be different with filter_pixStep,
	and the aligned channels should be set to zero*/
	void zq_cnn_innerproduct_gemm_16f_same_or_notsame_pixstep(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		void** buffer,
		__int64 *buffer_len
	);

	/*in_pixStep must be equal to filter_pixStep,
	and the aligned channels should be set to zero*/
	void zq_cnn_innerproduct_gemm_16f_align128bit_same_pixstep_batch(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		void** buffer,
		__int64 *buffer_len
	);
#endif//__ARM_NEON_FP16

#else

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
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
		int out_sliceStep,
		void** buffer,
		__int64 *buffer_len
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
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
		int out_sliceStep,
		void** buffer,
		__int64 *buffer_len
	);
#endif

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
		int out_sliceStep,
		void** buffer,
		__int64 *buffer_len
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
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
		int out_sliceStep,
		void** buffer,
		__int64 *buffer_len
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
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
		int out_sliceStep,
		void** buffer,
		__int64 *buffer_len
	);
#endif

#endif //__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif

#endif