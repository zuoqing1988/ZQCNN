#ifndef _ZQ_CNN_ELTWISE_NCHWC_H_
#define _ZQ_CNN_ELTWISE_NCHWC_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	void zq_cnn_eltwise_sum_nchwc1(
		int in_tensor_num,	//must be >=2
		const float** in_tensor4D_data,
		int N,
		int H,
		int W,
		int C,
		const int* in_widthStep,
		const int* in_sliceStep,
		const int* in_imStep,
		float* out_tensor4D_data,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);


	void zq_cnn_eltwise_sum_with_weight_nchwc1(
		int in_tensor_num,	//must be >=2
		const float** in_tensor4D_data,
		const float* weight,
		int N,
		int H,
		int W,
		int C,
		const int* in_widthStep,
		const int* in_sliceStep,
		const int* in_imStep,
		float* out_tensor4D_data,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_eltwise_mul_nchwc1(
		int in_tensor_num,	//must be >=2
		const float** in_tensor4D_data,
		int N,
		int H,
		int W,
		int C,
		const int* in_widthStep,
		const int* in_sliceStep,
		const int* in_imStep,
		float* out_tensor4D_data,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);


	void zq_cnn_eltwise_max_nchwc1(
		int in_tensor_num,	//must be >=2
		const float** in_tensor4D_data,
		int N,
		int H,
		int W,
		int C,
		const int* in_widthStep,
		const int* in_sliceStep,
		const int* in_imStep,
		float* out_tensor4D_data,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

#if __ARM_NEON 

	void zq_cnn_eltwise_sum_nchwc4(
		int in_tensor_num,	//must be >=2
		const float** in_tensor4D_data,
		int N,
		int H,
		int W,
		int C,
		const int* in_widthStep,
		const int* in_sliceStep,
		const int* in_imStep,
		float* out_tensor4D_data,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_eltwise_sum_with_weight_nchwc4(
		int in_tensor_num,	//must be >=2
		const float** in_tensor4D_data,
		const float* weight,
		int N,
		int H,
		int W,
		int C,
		const int* in_widthStep,
		const int* in_sliceStep,
		const int* in_imStep,
		float* out_tensor4D_data,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_eltwise_mul_nchwc4(
		int in_tensor_num,	//must be >=2
		const float** in_tensor4D_data,
		int N,
		int H,
		int W,
		int C,
		const int* in_widthStep,
		const int* in_sliceStep,
		const int* in_imStep,
		float* out_tensor4D_data,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_eltwise_max_nchwc4(
		int in_tensor_num,	//must be >=2
		const float** in_tensor4D_data,
		int N,
		int H,
		int W,
		int C,
		const int* in_widthStep,
		const int* in_sliceStep,
		const int* in_imStep,
		float* out_tensor4D_data,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

#else

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	void zq_cnn_eltwise_sum_nchwc4(
		int in_tensor_num,	//must be >=2
		const float** in_tensor4D_data,
		int N,
		int H,
		int W,
		int C,
		const int* in_widthStep,
		const int* in_sliceStep,
		const int* in_imStep,
		float* out_tensor4D_data,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_eltwise_sum_with_weight_nchwc4(
		int in_tensor_num,	//must be >=2
		const float** in_tensor4D_data,
		const float* weight,
		int N,
		int H,
		int W,
		int C,
		const int* in_widthStep,
		const int* in_sliceStep,
		const int* in_imStep,
		float* out_tensor4D_data,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_eltwise_mul_nchwc4(
		int in_tensor_num,	//must be >=2
		const float** in_tensor4D_data,
		int N,
		int H,
		int W,
		int C,
		const int* in_widthStep,
		const int* in_sliceStep,
		const int* in_imStep,
		float* out_tensor4D_data,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_eltwise_max_nchwc4(
		int in_tensor_num,	//must be >=2
		const float** in_tensor4D_data,
		int N,
		int H,
		int W,
		int C,
		const int* in_widthStep,
		const int* in_sliceStep,
		const int* in_imStep,
		float* out_tensor4D_data,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	void zq_cnn_eltwise_sum_nchwc8(
		int in_tensor_num,	//must be >=2
		const float** in_tensor4D_data,
		int N,
		int H,
		int W,
		int C,
		const int* in_widthStep,
		const int* in_sliceStep,
		const int* in_imStep,
		float* out_tensor4D_data,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_eltwise_sum_with_weight_nchwc8(
		int in_tensor_num,	//must be >=2
		const float** in_tensor4D_data,
		const float* weight,
		int N,
		int H,
		int W,
		int C,
		const int* in_widthStep,
		const int* in_sliceStep,
		const int* in_imStep,
		float* out_tensor4D_data,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_eltwise_mul_nchwc8(
		int in_tensor_num,	//must be >=2
		const float** in_tensor4D_data,
		int N,
		int H,
		int W,
		int C,
		const int* in_widthStep,
		const int* in_sliceStep,
		const int* in_imStep,
		float* out_tensor4D_data,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);

	void zq_cnn_eltwise_max_nchwc8(
		int in_tensor_num,	//must be >=2
		const float** in_tensor4D_data,
		int N,
		int H,
		int W,
		int C,
		const int* in_widthStep,
		const int* in_sliceStep,
		const int* in_imStep,
		float* out_tensor4D_data,
		int out_widthStep,
		int out_sliceStep,
		int out_imStep
	);
#endif

#endif //__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif

#endif