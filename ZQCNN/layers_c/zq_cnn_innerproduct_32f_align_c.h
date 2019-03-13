#ifndef _ZQ_CNN_INNER_PRODUCT_32F_ALIGN_C_H_
#define _ZQ_CNN_INNER_PRODUCT_32F_ALIGN_C_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif



#if __ARM_NEON

	void zq_cnn_innerproduct_32f_align0_general(
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
		//int filter_H, // must be in_H
		//int filter_W, // must be in_W
		//int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		float* out_tensor4D_data,
		//int out_N,	// must be in_N
		//int out_H,	// must be 1
		//int out_W,	// must be 1
		//int out_C,	// must be filter_N
		//int out_pixelStep,
		//int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_innerproduct_32f_align128bit(
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
		//int filter_H, // must be in_H
		//int filter_W, // must be in_W
		//int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		float* out_tensor4D_data,
		//int out_N,	// must be in_N
		//int out_H,	// must be 1
		//int out_W,	// must be 1
		//int out_C,	// must be filter_N
		//int out_pixelStep,
		//int out_widthStep,
		int out_sliceStep
	);


	void zq_cnn_innerproduct_32f_align0_noborder(
		const float* in_tensor4D_data,
		int in_N,
		int in_HWC,
		const float* filters_data,
		int filter_N,
		float* out_tensor4D_data,
		int out_sliceStep
	);

	void zq_cnn_innerproduct_32f_align128bit_noborder(
		const float* in_tensor4D_data,
		int in_N,
		int in_HWC,
		const float* filters_data,
		int filter_N,
		float* out_tensor4D_data,
		int out_sliceStep
	);

#if __ARM_NEON_FP16
	void zq_cnn_innerproduct_16f_align0_general(
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
		//int filter_H, // must be in_H
		//int filter_W, // must be in_W
		//int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		float16_t* out_tensor4D_data,
		//int out_N,	// must be in_N
		//int out_H,	// must be 1
		//int out_W,	// must be 1
		//int out_C,	// must be filter_N
		//int out_pixelStep,
		//int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_innerproduct_16f_align128bit(
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
		//int filter_H, // must be in_H
		//int filter_W, // must be in_W
		//int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		float16_t* out_tensor4D_data,
		//int out_N,	// must be in_N
		//int out_H,	// must be 1
		//int out_W,	// must be 1
		//int out_C,	// must be filter_N
		//int out_pixelStep,
		//int out_widthStep,
		int out_sliceStep
	);


	void zq_cnn_innerproduct_16f_align0_noborder(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_HWC,
		const float16_t* filters_data,
		int filter_N,
		float16_t* out_tensor4D_data,
		int out_sliceStep
	);

	void zq_cnn_innerproduct_16f_align128bit_noborder(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_HWC,
		const float16_t* filters_data,
		int filter_N,
		float16_t* out_tensor4D_data,
		int out_sliceStep
	);
#endif//__ARM_NEON_FP16

#else

	void zq_cnn_innerproduct_32f_align0_general(
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
		//int filter_H, // must be in_H
		//int filter_W, // must be in_W
		//int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		float* out_tensor4D_data,
		//int out_N,	// must be in_N
		//int out_H,	// must be 1
		//int out_W,	// must be 1
		//int out_C,	// must be filter_N
		//int out_pixelStep,
		//int out_widthStep,
		int out_sliceStep
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_innerproduct_32f_align128bit(
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
		//int filter_H, // must be in_H
		//int filter_W, // must be in_W
		//int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		float* out_tensor4D_data,
		//int out_N,	// must be in_N
		//int out_H,	// must be 1
		//int out_W,	// must be 1
		//int out_C,	// must be filter_N
		//int out_pixelStep,
		//int out_widthStep,
		int out_sliceStep
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_innerproduct_32f_align256bit(
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
		//int filter_H, // must be in_H
		//int filter_W, // must be in_W
		//int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		float* out_tensor4D_data,
		//int out_N,	// must be in_N
		//int out_H,	// must be 1
		//int out_W,	// must be 1
		//int out_C,	// must be filter_N
		//int out_pixelStep,
		//int out_widthStep,
		int out_sliceStep
	);
#endif


	void zq_cnn_innerproduct_32f_align0_noborder(
		const float* in_tensor4D_data,
		int in_N,
		int in_HWC,
		const float* filters_data,
		int filter_N,
		float* out_tensor4D_data,
		int out_sliceStep
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_innerproduct_32f_align128bit_noborder(
		const float* in_tensor4D_data,
		int in_N,
		int in_HWC,
		const float* filters_data,
		int filter_N,
		float* out_tensor4D_data,
		int out_sliceStep
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_innerproduct_32f_align256bit_noborder(
		const float* in_tensor4D_data,
		int in_N,
		int in_HWC,
		const float* filters_data,
		int filter_N,
		float* out_tensor4D_data,
		int out_sliceStep
	);
#endif

#endif //__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) //跨平台定义方法
}
#endif
#endif