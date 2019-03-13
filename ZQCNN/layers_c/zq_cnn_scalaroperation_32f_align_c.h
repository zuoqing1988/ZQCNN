#ifndef _ZQ_CNN_SCALAR_OPERATION_32F_ALIGN_C_H_
#define _ZQ_CNN_SCALAR_OPERATION_32F_ALIGN_C_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

#if __ARM_NEON
	void zq_cnn_scalaroperation_add_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_scalaroperation_add_32f_align128bit(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_scalaroperation_add_inplace_32f_align0(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

	void zq_cnn_scalaroperation_add_inplace_32f_align128bit(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

	void zq_cnn_scalaroperation_mul_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_scalaroperation_mul_32f_align128bit(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);


	void zq_cnn_scalaroperation_mul_inplace_32f_align0(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);


	void zq_cnn_scalaroperation_mul_inplace_32f_align128bit(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

	void zq_cnn_scalaroperation_max_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_scalaroperation_max_32f_align128bit(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_scalaroperation_max_inplace_32f_align0(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

	void zq_cnn_scalaroperation_max_inplace_32f_align128bit(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);


	void zq_cnn_scalaroperation_min_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_scalaroperation_min_32f_align128bit(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_scalaroperation_min_inplace_32f_align0(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

	void zq_cnn_scalaroperation_min_inplace_32f_align128bit(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

	
	void zq_cnn_scalaroperation_rdiv_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
	/*
	void zq_cnn_scalaroperation_rdiv_32f_align128bit(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
	*/
	void zq_cnn_scalaroperation_rdiv_inplace_32f_align0(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);
	/*
	void zq_cnn_scalaroperation_rdiv_inplace_32f_align128bit(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

	*/

	void zq_cnn_scalaroperation_rminus_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_scalaroperation_rminus_32f_align128bit(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_scalaroperation_rminus_inplace_32f_align0(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

	void zq_cnn_scalaroperation_rminus_inplace_32f_align128bit(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);
	void zq_cnn_scalaroperation_pow_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);


	void zq_cnn_scalaroperation_pow_inplace_32f_align0(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

#if __ARM_NEON_FP16
	void zq_cnn_scalaroperation_add_16f_align0(
		float16_t scalar,
		const float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float16_t* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_scalaroperation_add_16f_align128bit(
		float16_t scalar,
		const float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float16_t* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_scalaroperation_add_inplace_16f_align0(
		float16_t scalar,
		float16_t* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

	void zq_cnn_scalaroperation_add_inplace_16f_align128bit(
		float16_t scalar,
		float16_t* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

	void zq_cnn_scalaroperation_mul_16f_align0(
		float16_t scalar,
		const float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float16_t* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_scalaroperation_mul_16f_align128bit(
		float16_t scalar,
		const float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float16_t* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);


	void zq_cnn_scalaroperation_mul_inplace_16f_align0(
		float16_t scalar,
		float16_t* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);


	void zq_cnn_scalaroperation_mul_inplace_16f_align128bit(
		float16_t scalar,
		float16_t* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

	void zq_cnn_scalaroperation_max_16f_align0(
		float16_t scalar,
		const float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float16_t* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_scalaroperation_max_16f_align128bit(
		float16_t scalar,
		const float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float16_t* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_scalaroperation_max_inplace_16f_align0(
		float16_t scalar,
		float16_t* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

	void zq_cnn_scalaroperation_max_inplace_16f_align128bit(
		float16_t scalar,
		float16_t* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);


	void zq_cnn_scalaroperation_min_16f_align0(
		float16_t scalar,
		const float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float16_t* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_scalaroperation_min_16f_align128bit(
		float16_t scalar,
		const float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float16_t* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_scalaroperation_min_inplace_16f_align0(
		float16_t scalar,
		float16_t* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

	void zq_cnn_scalaroperation_min_inplace_16f_align128bit(
		float16_t scalar,
		float16_t* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);


	void zq_cnn_scalaroperation_rdiv_16f_align0(
		float16_t scalar,
		const float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float16_t* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
	/*
	void zq_cnn_scalaroperation_rdiv_16f_align128bit(
	float16_t scalar,
	const float16_t* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	float16_t* out_data,
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
	);
	*/
	void zq_cnn_scalaroperation_rdiv_inplace_16f_align0(
		float16_t scalar,
		float16_t* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);
	/*
	void zq_cnn_scalaroperation_rdiv_inplace_16f_align128bit(
	float16_t scalar,
	float16_t* data,	// in & out
	int N,
	int H,
	int W,
	int C,
	int pixelStep,
	int widthStep,
	int sliceStep
	);

	*/

	void zq_cnn_scalaroperation_rminus_16f_align0(
		float16_t scalar,
		const float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float16_t* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_scalaroperation_rminus_16f_align128bit(
		float16_t scalar,
		const float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float16_t* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	void zq_cnn_scalaroperation_rminus_inplace_16f_align0(
		float16_t scalar,
		float16_t* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

	void zq_cnn_scalaroperation_rminus_inplace_16f_align128bit(
		float16_t scalar,
		float16_t* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);
	void zq_cnn_scalaroperation_pow_16f_align0(
		float16_t scalar,
		const float16_t* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float16_t* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);


	void zq_cnn_scalaroperation_pow_inplace_16f_align0(
		float16_t scalar,
		float16_t* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);
#endif//__ARM_NEON_FP16

#else

	void zq_cnn_scalaroperation_add_32f_align0(
		float scalar,
		const float* in_data,	
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_scalaroperation_add_32f_align128bit(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_scalaroperation_add_32f_align256bit(
		float scalar,
		const float* in_data,	
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
#endif

	void zq_cnn_scalaroperation_add_inplace_32f_align0(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_scalaroperation_add_inplace_32f_align128bit(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_scalaroperation_add_inplace_32f_align256bit(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);
#endif

	void zq_cnn_scalaroperation_mul_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_scalaroperation_mul_32f_align128bit(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_scalaroperation_mul_32f_align256bit(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
#endif


	void zq_cnn_scalaroperation_mul_inplace_32f_align0(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);


#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_scalaroperation_mul_inplace_32f_align128bit(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_scalaroperation_mul_inplace_32f_align256bit(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);
#endif

	void zq_cnn_scalaroperation_max_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_scalaroperation_max_32f_align128bit(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_scalaroperation_max_32f_align256bit(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
#endif

	void zq_cnn_scalaroperation_max_inplace_32f_align0(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_scalaroperation_max_inplace_32f_align128bit(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_scalaroperation_max_inplace_32f_align256bit(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);
#endif


	void zq_cnn_scalaroperation_min_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_scalaroperation_min_32f_align128bit(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_scalaroperation_min_32f_align256bit(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
#endif

	void zq_cnn_scalaroperation_min_inplace_32f_align0(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_scalaroperation_min_inplace_32f_align128bit(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_scalaroperation_min_inplace_32f_align256bit(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);
#endif


	void zq_cnn_scalaroperation_rdiv_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_scalaroperation_rdiv_32f_align128bit(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_scalaroperation_rdiv_32f_align256bit(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
#endif

	void zq_cnn_scalaroperation_rdiv_inplace_32f_align0(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_scalaroperation_rdiv_inplace_32f_align128bit(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_scalaroperation_rdiv_inplace_32f_align256bit(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);
#endif

	void zq_cnn_scalaroperation_rminus_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_scalaroperation_rminus_32f_align128bit(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_scalaroperation_rminus_32f_align256bit(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
#endif

	void zq_cnn_scalaroperation_rminus_inplace_32f_align0(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_scalaroperation_rminus_inplace_32f_align128bit(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_scalaroperation_rminus_inplace_32f_align256bit(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);
#endif

	void zq_cnn_scalaroperation_pow_32f_align0(
		float scalar,
		const float* in_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		float* out_data,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);


	void zq_cnn_scalaroperation_pow_inplace_32f_align0(
		float scalar,
		float* data,	// in & out
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

#endif //__ARM_NEON


#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif

#endif