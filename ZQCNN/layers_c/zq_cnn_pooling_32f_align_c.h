#ifndef _ZQ_CNN_POOLING_32F_ALIGN_C_H_
#define _ZQ_CNN_POOLING_32F_ALIGN_C_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif



#if __ARM_NEON

	void zq_cnn_maxpooling_nopadding_32f_align0_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_32f_align0_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);


	void zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel5x5(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_kernel5x5(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_maxpooling_nopadding_nodivided_32f_align128bit_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_nodivided_32f_align128bit_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

#if __ARM_NEON_FP16
	void zq_cnn_maxpooling_nopadding_16f_align0_general(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_16f_align0_general(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_maxpooling_nopadding_suredivided_16f_align128bit_kernel2x2(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_16f_align128bit_kernel2x2(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);


	void zq_cnn_maxpooling_nopadding_suredivided_16f_align128bit_kernel3x3(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_16f_align128bit_kernel3x3(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_maxpooling_nopadding_suredivided_16f_align128bit_kernel5x5(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_16f_align128bit_kernel5x5(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_maxpooling_nopadding_suredivided_16f_align128bit_general(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_16f_align128bit_general(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_maxpooling_nopadding_nodivided_16f_align128bit_general(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_nodivided_16f_align128bit_general(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
#endif//__ARM_NEON_FP16

#else
	void zq_cnn_maxpooling_nopadding_32f_align0_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_32f_align0_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);


	void zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel5x5(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_kernel5x5(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_maxpooling_nopadding_nodivided_32f_align128bit_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_nodivided_32f_align128bit_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX

	void zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_32f_align256bit_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_32f_align256bit_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_kernel5x5(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_32f_align256bit_kernel5x5(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_32f_align256bit_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);


	void zq_cnn_maxpooling_nopadding_nodivided_32f_align256bit_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_sixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_avgpooling_nopadding_nodivided_32f_align256bit_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_sixelStep,
		int in_widthStep,
		int in_sliceStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

#endif 

#endif//__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif
