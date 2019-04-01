#ifndef _ZQ_CNN_INNER_PRODUCT_GEMM_NCHWC_H_
#define _ZQ_CNN_INNER_PRODUCT_GEMM_NCHWC_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	void zq_cnn_innerproduct_gemm_nchwc1_kernel1x1(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 1
		int filter_W, // must be 1
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc1_kernel1x1_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 1
		int filter_W, // must be 1
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc1_kernel1x1_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 1
		int filter_W, // must be 1
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc1_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc1_kernel2x2_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc1_kernel2x2_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc1_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc1_kernel3x3_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc1_kernel3x3_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* prelu,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc1_kernel7x7(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 7
		int filter_W, // must be 7
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc1_kernel7x7_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 7
		int filter_W, // must be 7
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc1_kernel7x7_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 7
		int filter_W, // must be 7
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* prelu,
		void** buffer,
		__int64* buffer_len
	);


	void zq_cnn_innerproduct_gemm_nchwc1_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, 
		int filter_W, 
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc1_general_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, 
		int filter_W, 
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc1_general_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, 
		int filter_W, 
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope,
		void** buffer,
		__int64* buffer_len
	);

#if __ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)

	void zq_cnn_innerproduct_gemm_nchwc4_kernel1x1(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 1
		int filter_W, // must be 1
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_kernel1x1_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 1
		int filter_W, // must be 1
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_kernel1x1_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 1
		int filter_W, // must be 1
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_kernel2x2_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_kernel2x2_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_kernel3x3_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_kernel3x3_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* prelu,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_kernel7x7(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, 
		int filter_W, 
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_kernel7x7_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 7
		int filter_W, // must be 7
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_kernel7x7_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 7
		int filter_W, // must be 7
		int filter_C,	
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, 
		int filter_W, 
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_general_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H,
		int filter_W,
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_general_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, 
		int filter_W, 
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope,
		void** buffer,
		__int64* buffer_len
	);

#endif// __ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_innerproduct_gemm_nchwc8_kernel1x1(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 1
		int filter_W, // must be 1
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc8_kernel1x1_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 1
		int filter_W, // must be 1
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc8_kernel1x1_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 1
		int filter_W, // must be 1
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc8_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc8_kernel2x2_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc8_kernel2x2_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope,
		void** buffer,
		__int64* buffer_len
	);
	
	void zq_cnn_innerproduct_gemm_nchwc8_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc8_kernel3x3_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc8_kernel3x3_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* prelu,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc8_kernel7x7(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 7
		int filter_W, // must be 7
		int filter_C,	
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc8_kernel7x7_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 7
		int filter_W, // must be 7
		int filter_C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc8_kernel7x7_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 3
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 7
		int filter_W, // must be 7
		int filter_C,	
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc8_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H,
		int filter_W,
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc8_general_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, 
		int filter_W, 
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc8_general_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N,
		int filter_H, 
		int filter_W, 
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope,
		void** buffer,
		__int64* buffer_len
	);

#endif//ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX


#if defined(__cplusplus) || defined(c_plusplus) //
}
#endif
#endif