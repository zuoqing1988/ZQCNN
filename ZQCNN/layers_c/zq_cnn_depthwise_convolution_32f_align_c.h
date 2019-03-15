#ifndef _ZQ_CNN_DEPTHWISE_CONVOLUTION_32F_ALIGN_C_H_
#define _ZQ_CNN_DEPTHWISE_CONVOLUTION_32F_ALIGN_C_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	void zq_cnn_depthwise_conv_no_padding_32f_align0_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_depthwise_conv_no_padding_32f_align0_general_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_32f_align0_general_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

#if __ARM_NEON || ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_general_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_general_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 4 (i.e. 1,2,3), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C4(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 1,2,3,4
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 1,2,3,4
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 4 (i.e. 1,2,3), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C4_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 1,2,3,4
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 1,2,3,4
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 4 (i.e. 1,2,3), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C4_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 1,2,3,4
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 1,2,3,4
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 4 (i.e. 1,2,3), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C4(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 1,2,3,4
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 1,2,3,4
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 4 (i.e. 1,2,3), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C4_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 1,2,3,4
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 1,2,3,4
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 4 (i.e. 1,2,3), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C4_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 1,2,3,4
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 1,2,3,4
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 8 (i.e. 5,6,7), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C8(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 5,6,7,8
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 5,6,7,8
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 8 (i.e. 5,6,7), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C8_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 5,6,7,8
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 5,6,7,8
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 8 (i.e. 5,6,7), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C8_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 5,6,7,8
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 5,6,7,8
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 8 (i.e. 5,6,7), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C8(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 5,6,7,8
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 5,6,7,8
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 8 (i.e. 5,6,7), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C8_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 5,6,7,8
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 5,6,7,8
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 8 (i.e. 5,6,7), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C8_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 5,6,7,8
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 5,6,7,8
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 16 (i.e. 13,14,15), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C16(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 13,14,15,16
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 13,14,15,16
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 16 (i.e. 13,14,15), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C16_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 13,14,15,16
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 13,14,15,16
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 16 (i.e. 13,14,15), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C16_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 13,14,15,16
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 13,14,15,16
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 16 (i.e. 13,14,15), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C16(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 13,14,15,16
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 13,14,15,16
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 16 (i.e. 13,14,15), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C16_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 13,14,15,16
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 13,14,15,16
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 16 (i.e. 13,14,15), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C16_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 13,14,15,16
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 13,14,15,16
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);


	/*if C is not 24 (i.e. 21,22,23), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C24(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 21,22,23,24
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 21,22,23,24
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 24 (i.e. 21,22,23), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C24_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 21,22,23,24
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 21,22,23,24
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 24 (i.e. 21,22,23), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C24_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 21,22,23,24
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 21,22,23,24
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 24 (i.e. 21,22,23), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C24(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 21,22,23,24
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 21,22,23,24
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 24 (i.e. 21,22,23), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C24_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 21,22,23,24
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 21,22,23,24
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 24 (i.e. 21,22,23), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C24_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 21,22,23,24
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 21,22,23,24
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 32 (i.e. 29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C32(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 29,30,31,32
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 32 (i.e. 29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C32_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 29,30,31,32
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 32 (i.e. 29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C32_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 29,30,31,32
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 32 (i.e. 29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C32(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 29,30,31,32
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 32 (i.e. 29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C32_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 29,30,31,32
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 32 (i.e. 29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C32_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 29,30,31,32
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 32x , the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_Cdiv32(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 32x , the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_Cdiv32_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 32x , the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_Cdiv32_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* prelu
	);


	/*if C is not 32x , the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_Cdiv32(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 32x , the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_Cdiv32_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 32x , the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_Cdiv32_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);


	/*if C is not 64 (i.e. 61,62,63), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C64(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 61,62,63,64
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 61,62,63,64
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 64 (i.e. 61,62,63), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C64_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 61,62,63,64
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 61,62,63,64
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 64 (i.e. 61,62,63), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C64_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 61,62,63,64
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 61,62,63,64
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 64 (i.e. 61,62,63), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C64(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 61,62,63,64
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 61,62,63,64
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 64 (i.e. 61,62,63), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C64_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 61,62,63,64
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 61,62,63,64
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 64 (i.e. 61,62,63), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C64_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 61,62,63,64
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 61,62,63,64
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 128 (i.e. 125,126,127), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C128(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 125,126,127,128
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 125,126,127,128
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 128 (i.e. 125,126,127), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C128_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 125,126,127,128
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 125,126,127,128
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 128 (i.e. 125,126,127), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C128_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 125,126,127,128
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 125,126,127,128
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 128 (i.e. 125,126,127), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C128(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 125,126,127,128
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 125,126,127,128
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 128 (i.e. 125,126,127), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C128_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 125,126,127,128
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 125,126,127,128
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 128 (i.e. 125,126,127), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C128_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 125,126,127,128
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 125,126,127,128
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 256 (i.e. 253,254,255), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C256(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 253,254,255,256
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 253,254,255,256
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 256 (i.e. 253,254,255), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C256_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 253,254,255,256
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 253,254,255,256
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 256 (i.e. 253,254,255), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C256_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 253,254,255,256
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 253,254,255,256
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 256 (i.e. 253,254,255), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C256(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 253,254,255,256
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 253,254,255,256
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 256 (i.e. 253,254,255), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C256_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 253,254,255,256
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 253,254,255,256
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 256 (i.e. 253,254,255), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C256_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 253,254,255,256
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 253,254,255,256
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias, 
		const float* slope
	);
#endif //__ARM_NEON || ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE

#if __ARM_NEON
#if __ARM_NEON_FP16
	void zq_cnn_depthwise_conv_no_padding_16f_align0_general(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_depthwise_conv_no_padding_16f_align0_general_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	void zq_cnn_depthwise_conv_no_padding_16f_align0_general_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_general(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_general_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_general_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

	/*if C is not 8, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C8(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 8, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C8_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	/*if C is not 8, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C8_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

	/*if C is not 8, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C8(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 8, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C8_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	/*if C is not 8, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C8_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);


	/*if C is not 16, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C16(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 16, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C16_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	/*if C is not 16, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C16_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);


	/*if C is not 16, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C16(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 16, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C16_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	/*if C is not 16, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C16_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

	/*if C is not 24, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C24(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 24, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C24_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	/*if C is not 24, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C24_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

	/*if C is not 24, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C24(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 24, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C24_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	/*if C is not 24, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C24_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

	/*if C is not 32, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C32(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 32, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C32_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	/*if C is not 32, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C32_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

	/*if C is not 32, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C32(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 32, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C32_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	/*if C is not 32, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C32_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);


	/*if C is not 64, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C64(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 64, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C64_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	/*if C is not 64, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C64_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

	/*if C is not 64, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C64(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 64, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C64_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	/*if C is not 64, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C64_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

	/*if C is not 64x , the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_Cdiv64(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 64x , the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_Cdiv64_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	/*if C is not 64x , the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_Cdiv64_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

	/*if C is not 64x , the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_Cdiv64(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 64x , the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_Cdiv64_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	/*if C is not 64x , the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_Cdiv64_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

	/*if C is not 128, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C128(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 128, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C128_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	/*if C is not 128, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C128_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

	/*if C is not 128, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C128(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 128, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C128_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	/*if C is not 128, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C128_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

	/*if C is not 256, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C256(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 256, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C256_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	/*if C is not 256, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C256_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

	/*if C is not 256, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C256(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 256, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C256_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	/*if C is not 256, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C256_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

	/*if C is not 512, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C512(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 512, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C512_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	/*if C is not 512, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel3x3_C512_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

	/*if C is not 512, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C512(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 512, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C512_with_bias(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias
	);

	/*if C is not 512, the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_16f_align128bit_kernel2x2_C512_with_bias_prelu(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float16_t* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float16_t* bias,
		const float16_t* slope
	);

#endif //__ARM_NEON_FP16
#endif //__ARM_NEON


#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_general_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_general_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // 
		int filter_W, // 
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be in_C
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, // must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 8 (i.e. 1,2,3,4,5,6,7), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C8(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 1,2,3,4,5,6,7,8
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 1,2,3,4,5,6,7,8
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 8 (i.e. 1,2,3,4,5,6,7), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C8_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 1,2,3,4,5,6,7,8
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 1,2,3,4,5,6,7,8
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 8 (i.e. 1,2,3,4,5,6,7), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C8_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 1,2,3,4,5,6,7,8
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 1,2,3,4,5,6,7,8
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 8 (i.e. 1,2,3,4,5,6,7), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C8(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 1,2,3,4,5,6,7,8
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 1,2,3,4,5,6,7,8
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 8 (i.e. 1,2,3,4,5,6,7), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C8_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 1,2,3,4,5,6,7,8
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 1,2,3,4,5,6,7,8
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 8 (i.e. 1,2,3,4,5,6,7), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C8_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 1,2,3,4,5,6,7,8
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 1,2,3,4,5,6,7,8
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 16 (i.e. 9,10,11,12,13,14,15), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C16(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 9,10,11,12,13,14,15,16
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 9,10,11,12,13,14,15,16
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 16 (i.e. 9,10,11,12,13,14,15), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C16_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 9,10,11,12,13,14,15,16
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 9,10,11,12,13,14,15,16
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 16 (i.e. 9,10,11,12,13,14,15), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C16_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 9,10,11,12,13,14,15,16
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 9,10,11,12,13,14,15,16
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);


	/*if C is not 16 (i.e. 9,10,11,12,13,14,15), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C16(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 9,10,11,12,13,14,15,16
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 9,10,11,12,13,14,15,16
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 16 (i.e. 9,10,11,12,13,14,15), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C16_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 9,10,11,12,13,14,15,16
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 9,10,11,12,13,14,15,16
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 16 (i.e. 9,10,11,12,13,14,15), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C16_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 9,10,11,12,13,14,15,16
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 9,10,11,12,13,14,15,16
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 32 (i.e. 25,26,27,28,29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C24(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 25,26,27,28,29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 25,26,27,28,29,30,31,32
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 32 (i.e. 25,26,27,28,29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C24_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 25,26,27,28,29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 25,26,27,28,29,30,31,32
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 32 (i.e. 25,26,27,28,29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C24_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 25,26,27,28,29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 25,26,27,28,29,30,31,32
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 32 (i.e. 25,26,27,28,29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C24(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 25,26,27,28,29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 25,26,27,28,29,30,31,32
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 32 (i.e. 25,26,27,28,29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C24_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 25,26,27,28,29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 25,26,27,28,29,30,31,32
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 32 (i.e. 25,26,27,28,29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C24_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 25,26,27,28,29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 25,26,27,28,29,30,31,32
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 32 (i.e. 25,26,27,28,29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C32(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 25,26,27,28,29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 25,26,27,28,29,30,31,32
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 32 (i.e. 25,26,27,28,29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C32_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 25,26,27,28,29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 25,26,27,28,29,30,31,32
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 32 (i.e. 25,26,27,28,29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C32_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 25,26,27,28,29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 25,26,27,28,29,30,31,32
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);


	/*if C is not 32 (i.e. 25,26,27,28,29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C32(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 25,26,27,28,29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 25,26,27,28,29,30,31,32
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 32 (i.e. 25,26,27,28,29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C32_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 25,26,27,28,29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 25,26,27,28,29,30,31,32
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 32 (i.e. 25,26,27,28,29,30,31), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C32_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 25,26,27,28,29,30,31,32
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,	//must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 25,26,27,28,29,30,31,32
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 64 (i.e. 57,58,59,60,61,62,63), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C64(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 57,58,59,60,61,62,63,64
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 57,58,59,60,61,62,63,64
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 64 (i.e. 57,58,59,60,61,62,63), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C64_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 57,58,59,60,61,62,63,64
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 57,58,59,60,61,62,63,64
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 64 (i.e. 57,58,59,60,61,62,63), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C64_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 57,58,59,60,61,62,63,64
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 57,58,59,60,61,62,63,64
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 64x , the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_Cdiv64(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 64x , the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_Cdiv64_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 64x , the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_Cdiv64_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 64 (i.e. 57,58,59,60,61,62,63), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C64(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 57,58,59,60,61,62,63,64
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 57,58,59,60,61,62,63,64
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 64 (i.e. 57,58,59,60,61,62,63), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C64_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 57,58,59,60,61,62,63,64
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 57,58,59,60,61,62,63,64
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 64 (i.e. 57,58,59,60,61,62,63), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C64_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 57,58,59,60,61,62,63,64
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 57,58,59,60,61,62,63,64
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 64x , the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_Cdiv64(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 64x , the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_Cdiv64_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 64x , the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_Cdiv64_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 128 (i.e. 121,122,123,124,125,126,127), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C128(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 121,122,123,124,125,126,127,128
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 121,122,123,124,125,126,127,128
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 128 (i.e. 121,122,123,124,125,126,127), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C128_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 121,122,123,124,125,126,127,128
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 121,122,123,124,125,126,127,128
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 128 (i.e. 121,122,123,124,125,126,127), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C128_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 121,122,123,124,125,126,127,128
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 121,122,123,124,125,126,127,128
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 128 (i.e. 121,122,123,124,125,126,127), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C128(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 121,122,123,124,125,126,127,128
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 121,122,123,124,125,126,127,128
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 128 (i.e. 121,122,123,124,125,126,127), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C128_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 121,122,123,124,125,126,127,128
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 121,122,123,124,125,126,127,128
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 128 (i.e. 121,122,123,124,125,126,127), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C128_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 121,122,123,124,125,126,127,128
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 121,122,123,124,125,126,127,128
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 256 (i.e. 249,250,251,252,253,254,255), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C256(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 249,250,251,252,253,254,255,256
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 249,250,251,252,253,254,255,256
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 256 (i.e. 249,250,251,252,253,254,255), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C256_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 249,250,251,252,253,254,255,256
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 249,250,251,252,253,254,255,256
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 256 (i.e. 249,250,251,252,253,254,255), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C256_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 249,250,251,252,253,254,255,256
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 249,250,251,252,253,254,255,256
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 256 (i.e. 249,250,251,252,253,254,255), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C256(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 249,250,251,252,253,254,255,256
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 249,250,251,252,253,254,255,256
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 256 (i.e. 249,250,251,252,253,254,255), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C256_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 249,250,251,252,253,254,255,256
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 249,250,251,252,253,254,255,256
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 256 (i.e. 249,250,251,252,253,254,255), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C256_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 249,250,251,252,253,254,255,256
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 249,250,251,252,253,254,255,256
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 512 (i.e. 505,506,507,508,509,510,511), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C512(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 505,506,507,508,509,510,511,512
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 505,506,507,508,509,510,511,512
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 512 (i.e. 505,506,507,508,509,510,511), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C512_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 505,506,507,508,509,510,511,512
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 505,506,507,508,509,510,511,512
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 512 (i.e. 505,506,507,508,509,510,511), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C512_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 505,506,507,508,509,510,511,512
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 3
		int filter_W, // must be 3
		int filter_C,	//must be 505,506,507,508,509,510,511,512
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

	/*if C is not 512 (i.e. 505,506,507,508,509,510,511), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C512(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 505,506,507,508,509,510,511,512
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 505,506,507,508,509,510,511,512
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*if C is not 512 (i.e. 505,506,507,508,509,510,511), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C512_with_bias(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 505,506,507,508,509,510,511,512
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 505,506,507,508,509,510,511,512
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias
	);

	/*if C is not 512 (i.e. 505,506,507,508,509,510,511), the padded channels of filters must be zero*/
	void zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C512_with_bias_prelu(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,	//must be 505,506,507,508,509,510,511,512
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N, //must be 1
		int filter_H, // must be 2
		int filter_W, // must be 2
		int filter_C,	//must be 505,506,507,508,509,510,511,512
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		const float* bias,
		const float* slope
	);

#endif// ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif