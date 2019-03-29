#ifndef _ZQ_CNN_POOLING_NCHWC_H_
#define _ZQ_CNN_POOLING_NCHWC_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	void zq_cnn_maxpooling_nopadding_nchwc1_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_avgpooling_nopadding_nchwc1_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_maxpooling_nopadding_suredivided_nchwc1_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_nchwc1_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);


	void zq_cnn_maxpooling_nopadding_suredivided_nchwc1_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_nchwc1_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_maxpooling_nopadding_suredivided_nchwc1_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_nchwc1_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_maxpooling_nopadding_nodivided_nchwc1_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_avgpooling_nopadding_nodivided_nchwc1_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

#if __ARM_NEON

	void zq_cnn_maxpooling_nopadding_suredivided_nchwc4_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_nchwc4_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);


	void zq_cnn_maxpooling_nopadding_suredivided_nchwc4_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_nchwc4_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_maxpooling_nopadding_suredivided_nchwc4_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_nchwc4_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_maxpooling_nopadding_nodivided_nchwc4_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_avgpooling_nopadding_nodivided_nchwc4_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

#else
	
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_maxpooling_nopadding_suredivided_nchwc4_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_nchwc4_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);


	void zq_cnn_maxpooling_nopadding_suredivided_nchwc4_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_nchwc4_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_maxpooling_nopadding_suredivided_nchwc4_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_nchwc4_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_maxpooling_nopadding_nodivided_nchwc4_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_avgpooling_nopadding_nodivided_nchwc4_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX

	void zq_cnn_maxpooling_nopadding_suredivided_nchwc8_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_nchwc8_kernel2x2(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_maxpooling_nopadding_suredivided_nchwc8_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_nchwc8_kernel3x3(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_maxpooling_nopadding_suredivided_nchwc8_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_avgpooling_nopadding_suredivided_nchwc8_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);


	void zq_cnn_maxpooling_nopadding_nodivided_nchwc8_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_avgpooling_nopadding_nodivided_nchwc8_general(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_widthStep,
		int in_sliceStep,
		int in_imStep,
		int kernel_H,
		int kernel_W,
		int stride_H,
		int stride_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be ceil((in_H - filter_H)/stride_H) + 1
		int out_W,	// must be ceil((in_W - filter_W)/stride_W) + 1
		int out_C,	// must be filter_N
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

#endif 

#endif//__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif
