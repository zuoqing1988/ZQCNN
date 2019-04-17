#ifndef _ZQ_CNN_INNER_PRODUCT_GEMM_NCHWC_H_
#define _ZQ_CNN_INNER_PRODUCT_GEMM_NCHWC_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	
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

	void zq_cnn_innerproduct_nchwc1_noborder(
		const float* in_tensor4D_data,
		int in_N,
		int in_HWC,
		const float* filters_data,
		int filter_N,
		float* out_tensor4D_data,
		int out_sliceStep
	);

	void zq_cnn_innerproduct_nchwc1_noborder_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_HWC,
		const float* filters_data,
		int filter_N,
		float* out_tensor4D_data,
		int out_sliceStep,
		const float* bias
	);

	void zq_cnn_innerproduct_nchwc1_noborder_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_HWC,
		const float* filters_data,
		int filter_N,
		float* out_tensor4D_data,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

#if __ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)

	void zq_cnn_innerproduct_gemm_nchwc4_prepack4(
		const float* filters_data,
		int N,
		int H,
		int W,
		int C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_packed4(
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* packed_filter,
		float* out_data,
		int out_N,
		int out_H,
		int out_W,
		int out_C,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_packed4_with_bias(
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* packed_filter,
		float* out_data,
		int out_N,
		int out_H,
		int out_W,
		int out_C,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_packed4_with_bias_prelu(
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* packed_filter,
		float* out_data,
		int out_N,
		int out_H,
		int out_W,
		int out_C,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope,
		void** buffer,
		__int64* buffer_len
	);

#if __ARM_NEON && __ARM_NEON_ARMV8

	void zq_cnn_innerproduct_gemm_nchwc4_prepack8_other(
		const float* filters_data,
		int N,
		int H,
		int W,
		int C,
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_packed8_other(
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* packed_filter,
		float* out_data,
		int out_N,
		int out_H,
		int out_W,
		int out_C,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_packed8_other_with_bias(
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* packed_filter,
		float* out_data,
		int out_N,
		int out_H,
		int out_W,
		int out_C,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		void** buffer,
		__int64* buffer_len
	);

	void zq_cnn_innerproduct_gemm_nchwc4_packed8_other_with_bias_prelu(
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* packed_filter,
		float* out_data,
		int out_N,
		int out_H,
		int out_W,
		int out_C,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope,
		void** buffer,
		__int64* buffer_len
	);

#endif

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

	void zq_cnn_innerproduct_nchwc4_noborder(
		const float* in_tensor4D_data,
		int in_N,
		int in_HWC,
		const float* filters_data,
		int filter_N,
		float* out_tensor4D_data,
		int out_sliceStep
	);

	void zq_cnn_innerproduct_nchwc4_noborder_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_HWC,
		const float* filters_data,
		int filter_N,
		float* out_tensor4D_data,
		int out_sliceStep,
		const float* bias
	);

	void zq_cnn_innerproduct_nchwc4_noborder_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_HWC,
		const float* filters_data,
		int filter_N,
		float* out_tensor4D_data,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

#endif// __ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	
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

	void zq_cnn_innerproduct_nchwc8_noborder(
		const float* in_tensor4D_data,
		int in_N,
		int in_HWC,
		const float* filters_data,
		int filter_N,
		float* out_tensor4D_data,
		int out_sliceStep
	);

	void zq_cnn_innerproduct_nchwc8_noborder_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_HWC,
		const float* filters_data,
		int filter_N,
		float* out_tensor4D_data,
		int out_sliceStep,
		const float* bias
	);

	void zq_cnn_innerproduct_nchwc8_noborder_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_HWC,
		const float* filters_data,
		int filter_N,
		float* out_tensor4D_data,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

#endif//ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX


#if defined(__cplusplus) || defined(c_plusplus) //
}
#endif
#endif