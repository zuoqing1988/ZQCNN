#ifndef _ZQ_CNN_SQRT_32F_ALIGN_C_H_
#define _ZQ_CNN_SQRT_32F_ALIGN_C_H_

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	void zq_cnn_sqrt_32f_align0(
		float* data,
		int N,
		int H,
		int W,
		int C,
		int pixelStep,
		int widthStep,
		int sliceStep
	);

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif
