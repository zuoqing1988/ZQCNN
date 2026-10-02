/* ------------------------------------------------------------------------
 * `void** buffer` / `__int64* buffer_len` 这对参数的**所有权约定**
 * （2026-10-02 补，见 audit_k3_20261001.md 附录 BJ）
 *
 *   buffer == NULL  —— 内核自己 _aligned_malloc / _aligned_free，用完即走。
 *   buffer != NULL  —— 读写的是**调用方**持有的两块内存：
 *                        *buffer      指向一块 _aligned_malloc 出来的内存
 *                                      （或者 NULL，表示"还没分配过"）；
 *                        *buffer_len  是它的字节数。
 *                      容量不够时内核会 _aligned_free(*buffer) 再重新分配，
 *                      并把新指针/新长度写回去。
 *                      **返回之后这块内存归调用方，内核不再持有、也不再释放它。**
 *
 * 生产里这两个槽位是 `ZQ_CNN_Net::Buffer` 的 data/len（见 ZQ_CNN_Net.h），
 * 随 net 一起活着，由 net 负责最终释放。
 *
 * 为什么专门写这段
 * --------------
 * 写 `tools/zq_innerproduct_check.cpp` 时，测试里 `buffer` 是个**局部变量**、
 * 用完没 free，LeakSanitizer 报了 84 KB x 36 的"泄漏" —— 那是测试的错。
 * 但反过来说：这条约定一旦被误解成"内核会替你释放"，改代码的人就会在
 * 内核里加一句 free，**直接变成 double free**。
 * 16 个带这对参数的头/源文件里，一条说明都没有（全 0），所以补在这里。
 * ---------------------------------------------------------------------- */
#ifndef _ZQ_CNN_DECONVOLUTION_GEMM_32F_ALIGN_C_H_
#define _ZQ_CNN_DECONVOLUTION_GEMM_32F_ALIGN_C_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	

#if __ARM_NEON || ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE

	void zq_cnn_deconv_with_padding_gemm_32f_align128bit_k2s2(
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
		int filter_H, // must be 1
		int filter_W, // must be 1
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		int dilation_H,
		int dilation_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		int pad_top,
		int pad_bottom,
		int pad_left,
		int pad_right,
		void** buffer,
		__int64 *buffer_len
	);

#endif

#if __ARM_NEON
#if __ARM_NEON_FP16
	void zq_cnn_deconv_with_padding_gemm_16f_align128bit_k2s2(
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
		int filter_H, // must be 1
		int filter_W, // must be 1
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		int dilation_H,
		int dilation_W,
		float16_t* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		int pad_top,
		int pad_bottom,
		int pad_left,
		int pad_right,
		void** buffer,
		__int64 *buffer_len
	);

#endif//__ARM_NEON_FP16

#else

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX

	void zq_cnn_deconv_with_padding_gemm_32f_align256bit_k2s2(
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
		int filter_H, // must be 1
		int filter_W, // must be 1
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		int dilation_H,
		int dilation_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		int pad_top,
		int pad_bottom,
		int pad_left,
		int pad_right,
		void** buffer,
		__int64 *buffer_len
	);
#endif

#endif //__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif

#endif