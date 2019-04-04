#ifndef _ZQ_CNN_DEPTHWISE_CONVOLUTION_NCHWC_H_
#define _ZQ_CNN_DEPTHWISE_CONVOLUTION_NCHWC_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	void zq_cnn_depthwise_conv_no_padding_nchwc1_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_general_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_general_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_kernel3x3_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_kernel3x3_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_kernel5x5_s1d1(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 5
		int filter_W, // must be 5
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_kernel5x5_s1d1_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 5
		int filter_W, // must be 5
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_kernel5x5_s1d1_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 5
		int filter_W, // must be 5
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_kernel3x3_s1d1(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_kernel3x3_s1d1_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_kernel3x3_s1d1_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_kernel3x3_s2d1(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_kernel3x3_s2d1_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_kernel3x3_s2d1_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_kernel2x2_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_kernel2x2_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_kernel2x2_s1d1(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_kernel2x2_s1d1_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc1_kernel2x2_s1d1_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

#if __ARM_NEON || ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_depthwise_conv_no_padding_nchwc4_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_general_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_general_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_kernel5x5_s1d1(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 5
		int filter_W, // must be 5
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_kernel5x5_s1d1_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 5
		int filter_W, // must be 5
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_kernel5x5_s1d1_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 5
		int filter_W, // must be 5
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3_s1d1(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3_s1d1_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3_s1d1_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3_s2d1(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3_s2d1_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3_s2d1_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);


	void zq_cnn_depthwise_conv_no_padding_nchwc4_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_kernel2x2_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_kernel2x2_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_kernel2x2_s1d1(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_kernel2x2_s1d1_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc4_kernel2x2_s1d1_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

#endif //__ARM_NEON || ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_depthwise_conv_no_padding_nchwc8_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_general_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_general_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_kernel3x3_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_kernel3x3_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_kernel5x5_s1d1(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 5
		int filter_W, // must be 5
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_kernel5x5_s1d1_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 5
		int filter_W, // must be 5
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_kernel5x5_s1d1_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 5
		int filter_W, // must be 5
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_kernel3x3_s1d1(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_kernel3x3_s1d1_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_kernel3x3_s1d1_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_kernel3x3_s2d1(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_kernel3x3_s2d1_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_kernel3x3_s2d1_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_kernel2x2_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_kernel2x2_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_kernel2x2_s1d1(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_kernel2x2_s1d1_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_nchwc8_kernel2x2_s1d1_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_widthStep,
		int filter_sliceStep,
		int filter_imStep,
		int stride_H,
		int stride_W,
		int dilate_H,
		int dilate_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep,
		const float* bias,
		const float* slope
	);

#endif// ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif