#include "layers_c/zq_cnn_convolution_32f_align_c.h"
#include "layers_c/zq_cnn_depthwise_convolution_32f_align_c.h"
#include "layers_c/zq_cnn_convolution_gemm_32f_align_c.h"
#include "layers_c/zq_cnn_innerproduct_32f_align_c.h"
#include "layers_c/zq_cnn_innerproduct_gemm_32f_align_c.h"
#include "layers_c/zq_cnn_addbias_32f_align_c.h"
#include "layers_c/zq_cnn_softmax_32f_align_c.h"
#include "layers_c/zq_cnn_pooling_32f_align_c.h"
#include "layers_c/zq_cnn_prelu_32f_align_c.h"
#include "layers_c/zq_cnn_relu_32f_align_c.h"
#include "layers_c/zq_cnn_dropout_32f_align_c.h"
#include "layers_c/zq_cnn_batchnormscale_32f_align_c.h"
#include "layers_c/zq_cnn_eltwise_32f_align_c.h"
#include "layers_c/zq_cnn_scalaroperation_32f_align_c.h"
#include "layers_c/zq_cnn_lrn_32f_align_c.h"
#include "layers_c/zq_cnn_normalize_32f_align_c.h"
#include "layers_c/zq_cnn_reduction_32f_align_c.h"
#include "layers_c/zq_cnn_sqrt_32f_align_c.h"
#include "ZQ_CNN_Forward_SSEUtils.h"
#include "ZQ_CNN_BBoxUtils.h"
#include <algorithm>
#include <math.h>
#include "ZQ_CNN_CompileConfig.h"

using namespace ZQ;

void _convolution_handle_special_channel_case_N_equal_one(bool& has_handled, int align_mode, const float* in_data, int in_N, int in_H, int in_W, int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
	const float* filter_data, int filter_N, int filter_H, int filter_W, int filter_C, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
	int strideH, int strideW, int dilation_H, int dilation_W, float* out_data, int out_N, int out_H, int out_W, int out_C, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	has_handled = false;
	if (out_N != 1)
		return;

	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if __ARM_NEON
		if (filter_H == 3 && filter_W == 3 && in_C == 3)
		{
			if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel3x3_C3_s1d1(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;

			}
			else if (strideH == 2 && strideW == 2 && dilation_H == 1 && dilation_W == 1)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel3x3_C3_s2d1(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;

			}
		}
#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		if (filter_H == 3 && filter_W == 3 && in_C == 3)
		{
			if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel3x3_C3_s1d1(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;

			}
			else if (strideH == 2 && strideW == 2 && dilation_H == 1 && dilation_W == 1)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel3x3_C3_s2d1(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;

			}
		}
#endif
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		//if (filter_H == 3 && filter_W == 3)
		//{
		//	/*if (in_C == 3)
		//	{
		//		zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
		//			filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
		//			dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
		//		has_handled = true;
		//	}
		//	else */if (in_C <= 8)
		//	{
		//		zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C8(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
		//			filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
		//			dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
		//		has_handled = true;
		//	}
		//	else if (in_C <= 16)
		//	{
		//		zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C16(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
		//			filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
		//			dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
		//		has_handled = true;
		//	}
		//	else if (in_C <= 24)
		//	{
		//		zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C24(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
		//			filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
		//			dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
		//		has_handled = true;
		//	}
		//	else if (in_C <= 32)
		//	{
		//		zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C32(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
		//			filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
		//			dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
		//		has_handled = true;
		//	}
		//}
#endif
	}
}


void _convolution_handle_special_channel_case_N_largerthan_one(bool& has_handled, int align_mode, const float* in_data, int in_N, int in_H, int in_W, int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
	const float* filter_data, int filter_N, int filter_H, int filter_W, int filter_C, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
	int strideH, int strideW, int dilation_H, int dilation_W, float* out_data, int out_N, int out_H, int out_W, int out_C, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	has_handled = false;
	//if (out_N <= 1) //none is fast
		return;
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		if (filter_H == 3 && filter_W == 3)
		{
			if (in_C == 3)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel3x3_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
		}
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		if (filter_H == 3 && filter_W == 3)
		{
			if (in_C == 3)
			{
				zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			/*else if (in_C <= 8)
			{
				zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C8(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W,	out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}*/
			/*else if (in_C <= 16)
			{
				zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C16(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W,	out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (in_C <= 24)
			{
				zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C24(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W,	out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (in_C <= 32)
			{
				zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C32(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W,	out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}*/
		}
#endif
	}
}

#if __ARM_NEON

void _convolution_nopadding_case_N_equal_one(int align_mode, const float* in_data, int in_N, int in_H, int in_W, int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
	const float* filter_data, int filter_N, int filter_H, int filter_W, int filter_C, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
	int strideH, int strideW, int dilation_H, int dilation_W, float* out_data, int out_N, int out_H, int out_W, int out_C, int out_pixStep, int out_widthStep, int out_sliceStep,
	void** buffer, __int64* buffer_len)
{
	bool has_handled = false;
	int out_HW = out_H*out_W;
	int out_NHW = out_HW*out_N;
	int filter_HWC = filter_H*filter_W*filter_C;
	int batch_need_size = out_NHW*filter_HWC + filter_N*filter_HWC;
	
	_convolution_handle_special_channel_case_N_equal_one(has_handled, align_mode, in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
		filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
		dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);

#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
	//gemm method
	if (!has_handled)
	{
		if (in_pixStep == filter_pixStep)
		{
			if (1||(out_HW >= 16 && filter_HWC >= 32 && filter_N >= 4) 
				|| ((out_HW >= 16 && filter_H == 1 && filter_W == 1 && filter_C >= 8 && filter_N >= 4)))
			{
				if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
				{
					if (filter_H == 1 && filter_W == 1)
					{
						zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_kernel1x1(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					}
					else if (in_C == 4)
					{
						zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_C4(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep,buffer,buffer_len);
					}
					else
					{
						zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep,buffer,buffer_len);
					}
					has_handled = true;
				}
				else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
				{

				}
			}
		}
		else
		{
			if (1||(out_HW >= 16 && filter_HWC >= 32 && filter_N >= 4)
				|| ((out_HW >= 16 && filter_H == 1 && filter_W == 1 && filter_C >= 8 && filter_N >= 4)))
			{
				if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
				{
					if (in_C == 3)
					{
						zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					}
					else
					{
						zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					}
					has_handled = true;
				}
				else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
				{

				}
				else
				{
					zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep,buffer,buffer_len);
					has_handled = true;
				}
			}
		}
	}
	
#endif

	//backup method
	if (!has_handled)
	{
		if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
		{
			if (filter_H == 1 && filter_W == 1 && in_C == 4)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel1x1_C4(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
			//else if (filter_H == 2 && filter_W == 2 /*&& 0*//*seems slow*/)
			//{
			//	zq_cnn_conv_no_padding_32f_align128bit_kernel2x2(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
			//		filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
			//		dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			//}
			else if (filter_H == 3 && filter_W == 3)
			{
				if (filter_C == 3)
				{
					zq_cnn_conv_no_padding_32f_align128bit_kernel3x3_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}
				else
				{
					zq_cnn_conv_no_padding_32f_align128bit_kernel3x3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}

			}
			else if (filter_H == 5 && filter_W == 5)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel5x5(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else
			{
				zq_cnn_conv_no_padding_32f_align128bit_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
		}
		else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
		{
		}
		else
		{
			zq_cnn_conv_no_padding_32f_align0_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
				dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
		}
	}
}

#else

void _convolution_nopadding_case_N_equal_one(int align_mode, const float* in_data, int in_N, int in_H, int in_W, int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
	const float* filter_data, int filter_N, int filter_H, int filter_W, int filter_C, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
	int strideH, int strideW, int dilation_H, int dilation_W, float* out_data, int out_N, int out_H, int out_W, int out_C, int out_pixStep, int out_widthStep, int out_sliceStep,
	void** buffer, __int64* buffer_len)
{
	bool has_handled = false;
	int out_HW = out_H*out_W;
	int out_NHW = out_HW*out_N;
	int filter_HWC = filter_H*filter_W*filter_C;
	int batch_need_size = out_NHW*filter_HWC + filter_N*filter_HWC;

	_convolution_handle_special_channel_case_N_equal_one(has_handled, align_mode, in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
	filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
	dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);

#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
	//gemm method
	if (!has_handled)
	{
		if (in_pixStep == filter_pixStep)
		{
			if (1 || (out_HW >= 16 && filter_HWC >= 32 && filter_N >= 4)
				|| ((out_HW >= 16 && filter_H == 1 && filter_W == 1 && filter_C >= 8 && filter_N >= 4)))
			{
				if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
				{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
					if (filter_H == 1 && filter_W == 1)
					{
						zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_kernel1x1(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					}
					else if (in_C == 4)
					{
						zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_C4(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					}
					else
					{
						zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					}
					has_handled = true;
#endif
				}
				else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
				{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
					if (filter_H == 1 && filter_W == 1)
					{
						zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep_kernel1x1(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					}
					else if (in_C == 4)
					{
						zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep_C4(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					}
					else
					{
						zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					}
					has_handled = true;
#endif
				}
			}
		}
		else
		{
			if (1 || (out_HW >= 16 && filter_HWC >= 32 && filter_N >= 4)
				|| ((out_HW >= 16 && filter_H == 1 && filter_W == 1 && filter_C >= 8 && filter_N >= 4)))
			{
				if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
				{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
					if (in_C == 3)
					{
						zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					}
					else
					{
						zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					}
					has_handled = true;
#endif
				}
				else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
				{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
					if (in_C == 3)
					{
						zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					}
					else
					{
						zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					}
					has_handled = true;
#endif
				}
				else
				{
					zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					has_handled = true;
				}
			}
		}
	}

#endif

	//backup method
	if (!has_handled)
	{
		if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
			if (filter_H == 1 && filter_W == 1 && in_C == 4)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel1x1_C4(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
			//else if (filter_H == 2 && filter_W == 2 /*&& 0*//*seems slow*/)
			//{
			//	zq_cnn_conv_no_padding_32f_align128bit_kernel2x2(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
			//		filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
			//		dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			//}
			else if (filter_H == 3 && filter_W == 3)
			{
				if (filter_C == 3)
				{
					zq_cnn_conv_no_padding_32f_align128bit_kernel3x3_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}
				else
				{
					zq_cnn_conv_no_padding_32f_align128bit_kernel3x3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}

			}
			else if (filter_H == 5 && filter_W == 5)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel5x5(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else
			{
				zq_cnn_conv_no_padding_32f_align128bit_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
#endif
		}
		else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			if (filter_H == 1 && filter_W == 1 && in_C == 4)
			{
				zq_cnn_conv_no_padding_32f_align256bit_kernel1x1_C4(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
			//else if (filter_H == 2 && filter_W == 2 /*&& 0*//*seems slow*/)
			//{
			//	zq_cnn_conv_no_padding_32f_align256bit_kernel2x2(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
			//		filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
			//		dilation_H, dilation_W,	out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
			//}
			else
				if (filter_H == 3 && filter_W == 3)
				{
					if (filter_C == 3)
					{
						zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					}
					else if (filter_C <= 8)
					{
						zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C8(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					}
					else if (filter_C <= 16)
					{
						zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C16(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					}
					else if (filter_C <= 24)
					{
						zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C24(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					}
					else if (filter_C <= 32)
					{
						zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C32(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					}
					else
					{
						zq_cnn_conv_no_padding_32f_align256bit_kernel3x3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					}

				}
			/*else if (filter_H == 5 && filter_W == 5)
			{
			zq_cnn_conv_no_padding_32f_align256bit_kernel5x5(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
			filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
			dilation_H, dilation_W,	out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
			}*/
				else
				{
					zq_cnn_conv_no_padding_32f_align256bit_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}
#endif
		}
		else
		{
			zq_cnn_conv_no_padding_32f_align0_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
				dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
		}
	}
}

#endif //__ARM_NEON

#if __ARM_NEON
void _convolution_nopadding_case_N_largerthan_one(int align_mode, const float* in_data, int in_N, int in_H, int in_W, int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
	const float* filter_data, int filter_N, int filter_H, int filter_W, int filter_C, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
	int strideH, int strideW, int dilation_H, int dilation_W, float* out_data, int out_N, int out_H, int out_W, int out_C, int out_pixStep, int out_widthStep, int out_sliceStep,
	void** buffer, __int64* buffer_len)
{
	const static int batch_limited_size = 100 * 1024 * 1024;

	bool has_handled = false;
	int out_HW = out_H*out_W;
	int out_NHW = out_HW*out_N;
	int filter_HWC = filter_H*filter_W*filter_C;
	int batch_need_size = out_NHW*filter_HWC + filter_N*filter_HWC;

	/*_convolution_handle_special_channel_case_N_largerthan_one(has_handled, align_mode, in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
		filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
		dilation_H, dilation_W,	out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);*/

#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
	//gemm
	if (!has_handled)
	{
		if (in_pixStep == filter_pixStep)
		{
			if (out_HW >= 8)
			{
				if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
				{
					zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep,buffer,buffer_len);
					has_handled = true;
				}
				else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
				{
				}
			}
			else if (out_NHW >= 8 || (out_H == 1 && out_W == 1))
			{
				if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
				{
					zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_batch(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep,buffer,buffer_len);
					has_handled = true;
				}
				else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
				{
				}
			}
		}
		else
		{
			if (out_HW >= 8 && out_N >= 16)
			{
				if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
				{
					if (in_C <= 4)
					{
						zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep,buffer,buffer_len);
						has_handled = true;
					}
					else
					{
						zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep,buffer,buffer_len);
						has_handled = true;
					}
				}
				else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
				{
				}
				else
				{
					zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep,buffer,buffer_len);
					has_handled = true;
				}
			}
			else if (out_NHW >= 8 && out_N >= 16)
			{
				if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
				{
					zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_batch(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep,buffer,buffer_len);
					has_handled = true;
				}
				else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
				{
				}
				else
				{
					zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep_batch(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep,buffer,buffer_len);
					has_handled = true;
				}
			}
		}
	}
#endif

	//backup method
	if (!has_handled)
	{
		if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
		{
			if (filter_H == 1 && filter_W == 1)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel1x1(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else if (filter_H == 2 && filter_W == 2 /*&& 0*//*seems slow*/)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel2x2(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else
				if (filter_H == 3 && filter_W == 3)
				{
					zq_cnn_conv_no_padding_32f_align128bit_kernel3x3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}
				else if (filter_H == 5 && filter_W == 5)
				{
					zq_cnn_conv_no_padding_32f_align128bit_kernel5x5(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}
				else
				{
					zq_cnn_conv_no_padding_32f_align128bit_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}
		}
		else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
		{
		}
		else
		{
			zq_cnn_conv_no_padding_32f_align0_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
				dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
		}
	}
}

#else

void _convolution_nopadding_case_N_largerthan_one(int align_mode, const float* in_data, int in_N, int in_H, int in_W, int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
	const float* filter_data, int filter_N, int filter_H, int filter_W, int filter_C, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
	int strideH, int strideW, int dilation_H, int dilation_W, float* out_data, int out_N, int out_H, int out_W, int out_C, int out_pixStep, int out_widthStep, int out_sliceStep,
	void** buffer, __int64* buffer_len)
{
	const static int batch_limited_size = 100 * 1024 * 1024;

	bool has_handled = false;
	int out_HW = out_H*out_W;
	int out_NHW = out_HW*out_N;
	int filter_HWC = filter_H*filter_W*filter_C;
	int batch_need_size = out_NHW*filter_HWC + filter_N*filter_HWC;

	/*_convolution_handle_special_channel_case_N_largerthan_one(has_handled, align_mode, in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
	filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
	dilation_H, dilation_W,	out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);*/

#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
	//gemm
	if (!has_handled)
	{
		if (in_pixStep == filter_pixStep)
		{
			if (out_HW >= 8)
			{
				if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
				{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
					zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					has_handled = true;
#endif
				}
				else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
				{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
					zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					has_handled = true;
#endif
				}
			}
			else if (out_NHW >= 8 || (out_H == 1 && out_W == 1))
			{
				if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
				{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
					zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_batch(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					has_handled = true;
#endif
				}
				else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
				{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
					zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep_batch(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					has_handled = true;
#endif
				}
			}
		}
		else
		{
			if (out_HW >= 8 && out_N >= 16)
			{
				if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
				{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
					if (in_C <= 4)
					{
						zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
						has_handled = true;
					}
					else
					{
						zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
						has_handled = true;
					}
#endif
				}
				else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
				{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
					if (in_C <= 8)
					{
						zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
						has_handled = true;
					}
					else
					{
						zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
						has_handled = true;
					}
#endif
				}
				else
				{
					zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					has_handled = true;
				}
			}
			else if (out_NHW >= 8 && out_N >= 16)
			{
				if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
				{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
					zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_batch(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					has_handled = true;
#endif
				}
				else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
				{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
					zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep_batch(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					has_handled = true;
#endif
				}
				else
				{
					zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep_batch(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, buffer, buffer_len);
					has_handled = true;
				}
			}
		}
	}
#endif

	//backup method
	if (!has_handled)
	{
		if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
			if (filter_H == 1 && filter_W == 1)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel1x1(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else if (filter_H == 2 && filter_W == 2 /*&& 0*//*seems slow*/)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel2x2(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else
				if (filter_H == 3 && filter_W == 3)
				{
					zq_cnn_conv_no_padding_32f_align128bit_kernel3x3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}
				else if (filter_H == 5 && filter_W == 5)
				{
					zq_cnn_conv_no_padding_32f_align128bit_kernel5x5(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}
				else
				{
					zq_cnn_conv_no_padding_32f_align128bit_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}
#endif
		}
		else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			//if (filter_H == 1 && filter_W == 1)
			//{
			//	zq_cnn_conv_no_padding_32f_align256bit_kernel1x1(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
			//		filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
			//		dilation_H, dilation_W,	out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
			//}
			//else if (filter_H == 2 && filter_W == 2 /*&& 0*//*seems slow*/)
			//{
			//	zq_cnn_conv_no_padding_32f_align256bit_kernel2x2(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
			//		filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
			//		dilation_H, dilation_W,	out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
			//}
			//else 
			if (filter_H == 3 && filter_W == 3)
			{
				zq_cnn_conv_no_padding_32f_align256bit_kernel3x3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
			/*else if (filter_H == 5 && filter_W == 5)
			{
			zq_cnn_conv_no_padding_32f_align256bit_kernel5x5(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
			filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
			dilation_H, dilation_W,	out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
			}*/
			else
			{
				zq_cnn_conv_no_padding_32f_align256bit_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
#endif
		}
		else
		{
			zq_cnn_conv_no_padding_32f_align0_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
				dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
		}
	}
}
#endif //__ARM_NEON

void ZQ_CNN_Forward_SSEUtils::_convolution_nopadding(int align_mode, const float* in_data, int in_N, int in_H, int in_W, int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
	const float* filter_data, int filter_N, int filter_H, int filter_W, int filter_C, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
	int strideH, int strideW, int dilation_H, int dilation_W, float* out_data, int out_N, int out_H, int out_W, int out_C, int out_pixStep, int out_widthStep, int out_sliceStep,
	void** buffer, __int64* buffer_len)
{
	if (out_N > 1)
	{
		_convolution_nopadding_case_N_largerthan_one(align_mode, in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
			filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
			dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep,buffer,buffer_len);
	}
	else
	{
		_convolution_nopadding_case_N_equal_one(align_mode, in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
			filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
			dilation_H, dilation_W, out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep,buffer,buffer_len);
	}
}

#if __ARM_NEON

void ZQ_CNN_Forward_SSEUtils::_depthwise_convolution_nopadding(int align_mode, const float* in_data, int in_N, int in_H, int in_W, int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
	const float* filter_data, int filter_N, int filter_H, int filter_W, int filter_C, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
	int strideH, int strideW, float* out_data, int out_N, int out_H, int out_W, int out_C, int out_pixStep, int out_widthStep, int out_sliceStep, const float* bias, const float* slope)
{
	bool has_handled = false;

	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		if (filter_H == 3 && filter_W == 3)
		{
			int padded_C = (in_C + 3) >> 2 << 2;
			if (padded_C == 4)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C4(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if (slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C4_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C4_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);

				has_handled = true;
			}
			else if (padded_C == 8)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C8(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if (slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C8_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C8_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 16)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C16(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if (slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C16_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C16_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 32)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C32(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if (slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C32_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C32_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 64)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C64(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if (slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C64_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C64_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 128)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C128(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if (slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C128_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C128_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 256)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C256(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if (slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C256_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C256_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C % 32 == 0)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_Cdiv32(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if (slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_Cdiv32_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_Cdiv32_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if (slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
		}
		else if (filter_H == 2 && filter_W == 2)
		{
			int padded_C = (in_C + 3) >> 2 << 2;
			if (padded_C == 4)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C4(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if (slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C4_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C4_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 8)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C8(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if (slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C8_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C8_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 16)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C16(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if (slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C16_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C16_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 32)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C32(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if (slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C32_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C32_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 64)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C64(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if (slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C64_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C64_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 128)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C128(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if (slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C128_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C128_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 256)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C256(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if (slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C256_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C256_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C % 32 == 0)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_Cdiv32(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if (slope = NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_Cdiv32_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_Cdiv32_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if (slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
		}
		else
		{
			if (bias == NULL)
				zq_cnn_depthwise_conv_no_padding_32f_align128bit_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			else if (slope == NULL)
				zq_cnn_depthwise_conv_no_padding_32f_align128bit_general_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
			else
				zq_cnn_depthwise_conv_no_padding_32f_align128bit_general_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
			has_handled = true;
		}
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}

	if (!has_handled)
	{
		if (bias == NULL)
			zq_cnn_depthwise_conv_no_padding_32f_align0_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
				out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
		else if (slope == NULL)
			zq_cnn_depthwise_conv_no_padding_32f_align0_general_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
				out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
		else
			zq_cnn_depthwise_conv_no_padding_32f_align0_general_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
				out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
		has_handled = true;
	}
}


#else

void ZQ_CNN_Forward_SSEUtils::_depthwise_convolution_nopadding(int align_mode, const float* in_data, int in_N, int in_H, int in_W, int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
	const float* filter_data, int filter_N, int filter_H, int filter_W, int filter_C, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
	int strideH, int strideW, float* out_data, int out_N, int out_H, int out_W, int out_C, int out_pixStep, int out_widthStep, int out_sliceStep, const float* bias, const float* slope)
{
	bool has_handled = false;

	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		if (filter_H == 3 && filter_W == 3)
		{
			int padded_C = (in_C + 3) >> 2 << 2;
			if (padded_C == 4)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C4(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C4_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C4_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);

				has_handled = true;
			}
			else if (padded_C == 8)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C8(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C8_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C8_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 16)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C16(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C16_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C16_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 32)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C32(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C32_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C32_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 64)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C64(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C64_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C64_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 128)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C128(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C128_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C128_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 256)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C256(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C256_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C256_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C % 32 == 0)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_Cdiv32(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_Cdiv32_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_Cdiv32_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
		}
		else if (filter_H == 2 && filter_W == 2)
		{
			int padded_C = (in_C + 3) >> 2 << 2;
			if (padded_C == 4)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C4(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C4_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C4_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 8)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C8(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C8_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C8_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 16)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C16(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C16_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C16_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 32)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C32(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C32_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C32_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 64)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C64(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C64_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C64_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 128)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C128(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C128_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C128_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 256)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C256(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C256_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C256_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C % 32 == 0)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_Cdiv32(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope = NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_Cdiv32_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_Cdiv32_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
		}
		else
		{
			if (bias == NULL)
				zq_cnn_depthwise_conv_no_padding_32f_align128bit_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			else if(slope == NULL)
				zq_cnn_depthwise_conv_no_padding_32f_align128bit_general_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
			else
				zq_cnn_depthwise_conv_no_padding_32f_align128bit_general_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
			has_handled = true;
		}
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		if (filter_H == 3 && filter_W == 3)
		{
			int padded_C = (in_C + 7) >> 3 << 3;
			if (padded_C == 8)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C8(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C8_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C8_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 16)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C16(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C16_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C16_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 24)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C24(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C24_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C24_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 32)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C32(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C32_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C32_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 64)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C64(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C64_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C64_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 128)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C128(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C128_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C128_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 256)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C256(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C256_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C256_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 512)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C512(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C512_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C512_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C % 64 == 0)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_Cdiv64(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_Cdiv64_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_Cdiv64_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
		}
		else if (filter_H == 2 && filter_W == 2)
		{
			int padded_C = (in_C + 7) >> 3 << 3;
			if (padded_C == 8)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C8(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C8_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C8_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 16)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C16(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C16_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C16_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 24)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C24(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C24_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C24_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 32)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C32(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C32_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C32_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 64)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C64(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C64_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C64_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 128)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C128(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C128_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C128_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 256)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C256(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C256_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C256_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C == 512)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C512(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C512_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C512_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else if (padded_C % 64 == 0)
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_Cdiv64(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_Cdiv64_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_Cdiv64_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
			else
			{
				if (bias == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				else if(slope == NULL)
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
				else
					zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
				has_handled = true;
			}
		}
		else
		{
			if (bias == NULL)
				zq_cnn_depthwise_conv_no_padding_32f_align256bit_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			else if(slope == NULL)
				zq_cnn_depthwise_conv_no_padding_32f_align256bit_general_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
			else
				zq_cnn_depthwise_conv_no_padding_32f_align256bit_general_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
			has_handled = true;
		}
#endif
	}

	if (!has_handled)
	{
		if (bias == NULL)
			zq_cnn_depthwise_conv_no_padding_32f_align0_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
				out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
		else if(slope == NULL)
			zq_cnn_depthwise_conv_no_padding_32f_align0_general_with_bias(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
				out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
		else
			zq_cnn_depthwise_conv_no_padding_32f_align0_general_with_bias_prelu(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
				out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias, slope);
		has_handled = true;
	}
}

#endif

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_inner_product(int align_mode, const float* in_data, int in_N, int in_H, int in_W, 
	int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
	const float* filter_data, int filter_N, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
	float* out_data, int out_N, int out_sliceStep,void**buffer, __int64* buffer_len)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
		if (out_N >= 16 && filter_N >= 16 && in_pixStep == filter_pixStep)
		{
			zq_cnn_innerproduct_gemm_32f_align128bit_same_pixstep_batch(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_pixStep, filter_widthStep, filter_sliceStep, out_data, out_N, filter_N, 
				out_sliceStep, out_sliceStep, out_sliceStep,buffer, buffer_len);
		}
		else
#endif
		{
			if (in_pixStep * in_W == in_widthStep && in_widthStep*in_H == in_sliceStep
				&& filter_pixStep*in_W == filter_widthStep && filter_widthStep*in_H == filter_sliceStep)
			{
				zq_cnn_innerproduct_32f_align128bit_noborder(in_data, in_N, in_H*in_W*in_C, filter_data, filter_N, out_data, out_sliceStep);
			}
			else
			{
				zq_cnn_innerproduct_32f_align128bit(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_pixStep, filter_widthStep, filter_sliceStep, out_data, out_sliceStep);
			}
		}
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{

	}
	else
	{
		if (in_pixStep * in_W == in_widthStep && in_widthStep*in_H == in_sliceStep
			&& filter_pixStep*in_W == filter_widthStep && filter_widthStep*in_H == filter_sliceStep)
		{
			zq_cnn_innerproduct_32f_align0_noborder(in_data, in_N, in_H*in_W*in_C, filter_data, filter_N, out_data,out_sliceStep);
		}
		else
		{
			zq_cnn_innerproduct_32f_align0_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_pixStep, filter_widthStep, filter_sliceStep, out_data, out_sliceStep);
		}
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_inner_product(int align_mode, const float* in_data, int in_N, int in_H, int in_W,
	int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
	const float* filter_data, int filter_N, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
	float* out_data, int out_N, int out_sliceStep, void**buffer, __int64* buffer_len)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
		if (out_N >= 16 && filter_N >= 16 && in_pixStep == filter_pixStep)
		{
			zq_cnn_innerproduct_gemm_32f_align128bit_same_pixstep_batch(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_pixStep, filter_widthStep, filter_sliceStep, out_data, out_N, filter_N,
				out_sliceStep, out_sliceStep, out_sliceStep, buffer, buffer_len);
		}
		else
#endif
		{
			if (in_pixStep * in_W == in_widthStep && in_widthStep*in_H == in_sliceStep
				&& filter_pixStep*in_W == filter_widthStep && filter_widthStep*in_H == filter_sliceStep)
			{
				zq_cnn_innerproduct_32f_align128bit_noborder(in_data, in_N, in_H*in_W*in_C, filter_data, filter_N, out_data, out_sliceStep);
			}
			else
			{
				zq_cnn_innerproduct_32f_align128bit(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_pixStep, filter_widthStep, filter_sliceStep, out_data, out_sliceStep);
			}
		}
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
		if (out_N >= 16 && filter_N >= 16 && in_pixStep == filter_pixStep)
		{
			zq_cnn_innerproduct_gemm_32f_align256bit_same_pixstep_batch(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_pixStep, filter_widthStep, filter_sliceStep, out_data, out_N, filter_N,
				out_sliceStep, out_sliceStep, out_sliceStep, buffer, buffer_len);
		}
		else
#endif
		{

			if (in_pixStep * in_W == in_widthStep && in_widthStep*in_H == in_sliceStep
				&& filter_pixStep*in_W == filter_widthStep && filter_widthStep*in_H == filter_sliceStep)
			{
				zq_cnn_innerproduct_32f_align256bit_noborder(in_data, in_N, in_H*in_W*in_C,
					filter_data, filter_N, out_data, out_sliceStep);
			}
			else
			{
				zq_cnn_innerproduct_32f_align256bit(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_pixStep, filter_widthStep, filter_sliceStep, out_data, out_sliceStep);
			}
		}
#endif
	}
	else
	{
		if (in_pixStep * in_W == in_widthStep && in_widthStep*in_H == in_sliceStep
			&& filter_pixStep*in_W == filter_widthStep && filter_widthStep*in_H == filter_sliceStep)
		{
			zq_cnn_innerproduct_32f_align0_noborder(in_data, in_N, in_H*in_W*in_C, filter_data, filter_N, out_data, out_sliceStep);
		}
		else
		{
			zq_cnn_innerproduct_32f_align0_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_pixStep, filter_widthStep, filter_sliceStep, out_data, out_sliceStep);
		}
	}
}

#endif //__ARM_NEON

#if __ARM_NEON
void  ZQ_CNN_Forward_SSEUtils::_addbias(int align_mode, float* data, int N, int H, int W, int C, 
	int pixelStep, int widthStep, int sliceStep, const float* bias_Data)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_addbias_32f_align128bit(data, N, H, W, C, pixelStep, widthStep, sliceStep,bias_Data);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_addbias_32f_align0(data, N, H, W, C, pixelStep, widthStep, sliceStep, bias_Data);
	}
}

#else
void  ZQ_CNN_Forward_SSEUtils::_addbias(int align_mode, float* data, int N, int H, int W, int C,
	int pixelStep, int widthStep, int sliceStep, const float* bias_Data)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_addbias_32f_align128bit(data, N, H, W, C, pixelStep, widthStep, sliceStep, bias_Data);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_addbias_32f_align256bit(data, N, H, W, C, pixelStep, widthStep, sliceStep, bias_Data);
#endif
	}
	else
	{
		zq_cnn_addbias_32f_align0(data, N, H, W, C, pixelStep, widthStep, sliceStep, bias_Data);
	}
}
#endif //__ARM_NEON

void ZQ_CNN_Forward_SSEUtils::_softmax(int align_mode, int axis, float* data, int N, int H, int W, int C, 
	int pixStep, int widthStep, int sliceStep)
{
	if (axis == 1)
	{
		if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit && C >= 4)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
			zq_cnn_softmax_32f_align128bit_C(data, N, H, W, C, pixStep, widthStep, sliceStep);
#endif
		}
		else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit && C >= 8)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			zq_cnn_softmax_32f_align256bit_C(data, N, H, W, C, pixStep, widthStep, sliceStep);
#endif
		}
		else
		{
			zq_cnn_softmax_32f_align0_C(data, N, H, W, C, pixStep, widthStep, sliceStep);
		}
	}
	else if (axis == 2)
	{
		zq_cnn_softmax_32f_align0_H(data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
	else if (axis == 3)
	{
		zq_cnn_softmax_32f_align0_W(data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
}

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_dropout(int align_mode, float* data, int N, int H, int W, int C, 
	int pixStep, int widthStep, int sliceStep, float ratio)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_dropout_32f_align128bit(data, N, H, W, C, pixStep, widthStep, sliceStep, ratio);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_dropout_32f_align0(data, N, H, W, C, pixStep, widthStep, sliceStep, ratio);
	}
}

#else
void ZQ_CNN_Forward_SSEUtils::_dropout(int align_mode, float* data, int N, int H, int W, int C,
	int pixStep, int widthStep, int sliceStep, float ratio)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_dropout_32f_align128bit(data, N, H, W, C, pixStep, widthStep, sliceStep, ratio);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_dropout_32f_align256bit(data, N, H, W, C, pixStep, widthStep, sliceStep, ratio);
#endif
	}
	else
	{
		zq_cnn_dropout_32f_align0(data, N, H, W, C, pixStep, widthStep, sliceStep, ratio);
	}
}
#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_prelu(int align_mode, float* data, int N, int H, int W, int C, int pixelStep, int widthStep, int sliceStep, 
	const float* slope_Data)
{
	bool sure_slope_lessthan1 = false;
	if (N*H*W > 4)
	{
		sure_slope_lessthan1 = true;
		for (int i = 0; i < C; i++)
		{
			if (slope_Data[i] > 1)
			{
				sure_slope_lessthan1 = false;
				break;
			}
		}
	}

	if (sure_slope_lessthan1)
	{
		if (C == 1)
			align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_0);
		else if (C <= 4)
			align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_128bit);
		else if (C <= 8)
			align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_256bit);

		if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
		{
			zq_cnn_prelu_32f_align128bit_sure_slope_lessthan1(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope_Data);
		}
		else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
		{
		}
		else
		{
			zq_cnn_prelu_32f_align0(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope_Data);
		}
	}
	else
	{
		if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
		{
			zq_cnn_prelu_32f_align128bit(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope_Data);
		}
		else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
		{
		}
		else
		{
			zq_cnn_prelu_32f_align0(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope_Data);
		}
	}
}

void ZQ_CNN_Forward_SSEUtils::_addbias_prelu(int align_mode, float* data, int N, int H, int W, int C, int pixelStep, int widthStep, int sliceStep,
	const float* bias, const float* slope_Data)
{
	bool sure_slope_lessthan1 = false;
	if (N*H*W > 4)
	{
		sure_slope_lessthan1 = true;
		for (int i = 0; i < C; i++)
		{
			if (slope_Data[i] > 1)
			{
				sure_slope_lessthan1 = false;
				break;
			}
		}
	}

	if (sure_slope_lessthan1)
	{
		if (C == 1)
			align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_0);
		else if (C <= 4)
			align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_128bit);
		
		if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
		{
			zq_cnn_addbias_prelu_32f_align128bit_sure_slope_lessthan1(data, N, H, W, C, pixelStep, widthStep, sliceStep, bias, slope_Data);
		}
		else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
		{
		}
		else
		{
			zq_cnn_addbias_prelu_32f_align0(data, N, H, W, C, pixelStep, widthStep, sliceStep, bias, slope_Data);
		}
	}
	else
	{
		if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
		{
			zq_cnn_addbias_prelu_32f_align128bit(data, N, H, W, C, pixelStep, widthStep, sliceStep, bias, slope_Data);
		}
		else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
		{
		}
		else
		{
			zq_cnn_addbias_prelu_32f_align0(data, N, H, W, C, pixelStep, widthStep, sliceStep, bias, slope_Data);
		}
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_prelu(int align_mode, float* data, int N, int H, int W, int C, int pixelStep, int widthStep, int sliceStep,
	const float* slope_Data)
{
	bool sure_slope_lessthan1 = false;
	if (N*H*W > 4)
	{
		sure_slope_lessthan1 = true;
		for (int i = 0; i < C; i++)
		{
			if (slope_Data[i] > 1)
			{
				sure_slope_lessthan1 = false;
				break;
			}
		}
	}

	if (sure_slope_lessthan1)
	{
		if (C == 1)
			align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_0);
		else if (C <= 4)
			align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_128bit);
		else if (C <= 8)
			align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_256bit);

		if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
			zq_cnn_prelu_32f_align128bit_sure_slope_lessthan1(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope_Data);
#endif
		}
		else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			zq_cnn_prelu_32f_align256bit_sure_slope_lessthan1(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope_Data);
#endif
		}
		else
		{
			zq_cnn_prelu_32f_align0(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope_Data);
		}
	}
	else
	{
		if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
			zq_cnn_prelu_32f_align128bit(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope_Data);
#endif
		}
		else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			zq_cnn_prelu_32f_align256bit(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope_Data);
#endif
		}
		else
		{
			zq_cnn_prelu_32f_align0(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope_Data);
		}
	}
}

void ZQ_CNN_Forward_SSEUtils::_addbias_prelu(int align_mode, float* data, int N, int H, int W, int C, int pixelStep, int widthStep, int sliceStep,
	const float* bias, const float* slope_Data)
{
	bool sure_slope_lessthan1 = false;
	if (N*H*W > 4)
	{
		sure_slope_lessthan1 = true;
		for (int i = 0; i < C; i++)
		{
			if (slope_Data[i] > 1)
			{
				sure_slope_lessthan1 = false;
				break;
			}
		}
	}

	if (sure_slope_lessthan1)
	{
		if (C == 1)
			align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_0);
		else if (C <= 4)
			align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_128bit);
		else if (C <= 8)
			align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_256bit);

		if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
			zq_cnn_addbias_prelu_32f_align128bit_sure_slope_lessthan1(data, N, H, W, C, pixelStep, widthStep, sliceStep, bias, slope_Data);
#endif
		}
		else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			zq_cnn_addbias_prelu_32f_align256bit_sure_slope_lessthan1(data, N, H, W, C, pixelStep, widthStep, sliceStep, bias, slope_Data);
#endif
		}
		else
		{
			zq_cnn_addbias_prelu_32f_align0(data, N, H, W, C, pixelStep, widthStep, sliceStep, bias, slope_Data);
		}
	}
	else
	{
		if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
			zq_cnn_addbias_prelu_32f_align128bit(data, N, H, W, C, pixelStep, widthStep, sliceStep, bias, slope_Data);
#endif
		}
		else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			zq_cnn_addbias_prelu_32f_align256bit(data, N, H, W, C, pixelStep, widthStep, sliceStep, bias, slope_Data);
#endif
		}
		else
		{
			zq_cnn_addbias_prelu_32f_align0(data, N, H, W, C, pixelStep, widthStep, sliceStep, bias, slope_Data);
		}
	}
}

#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_relu(int align_mode, float* data, int N, int H, int W, int C, 
	int pixelStep, int widthStep, int sliceStep, float slope)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_relu_32f_align128bit(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_relu_32f_align0(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope);
	}
}

#else
void ZQ_CNN_Forward_SSEUtils::_relu(int align_mode, float* data, int N, int H, int W, int C,
	int pixelStep, int widthStep, int sliceStep, float slope)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_relu_32f_align128bit(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_relu_32f_align256bit(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope);
#endif
	}
	else
	{
		zq_cnn_relu_32f_align0(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope);
	}
}
#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_maxpooling(int align_mode, const float* in_data, int N, int in_H, int in_W, int C, 
	int in_pixStep, int in_widthStep, int in_sliceStep,
	int kernel_H, int kernel_W, int stride_H, int stride_W, float* out_data, int out_H, int out_W, int 
	out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (C == 1)
		align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_0);
	else if (C <= 4)
		align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_128bit);
	else if (C <= 8)
		align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_256bit);

	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		bool suredivided = (in_H - kernel_H) % stride_H == 0 && (in_W - kernel_W) % stride_W == 0;
		if (suredivided)
		{
			if (kernel_H == 2 && kernel_W == 2 && 0)
			{
				zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel2x2(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else if (kernel_H == 3 && kernel_W == 3)
			{
				zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel3x3(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else if (kernel_H == 5 && kernel_W == 5)
			{
				zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel5x5(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else
			{
				zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_general(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
		}
		else
		{
			zq_cnn_maxpooling_nopadding_nodivided_32f_align128bit_general(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
				kernel_H, kernel_W, stride_H, stride_W,
				out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
		}
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_maxpooling_nopadding_32f_align0_general(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
			kernel_H, kernel_W, stride_H, stride_W,
			out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_maxpooling(int align_mode, const float* in_data, int N, int in_H, int in_W, int C,
	int in_pixStep, int in_widthStep, int in_sliceStep,
	int kernel_H, int kernel_W, int stride_H, int stride_W, float* out_data, int out_H, int out_W, int
	out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (C == 1)
		align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_0);
	else if (C <= 4)
		align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_128bit);
	else if (C <= 8)
		align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_256bit);

	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		bool suredivided = (in_H - kernel_H) % stride_H == 0 && (in_W - kernel_W) % stride_W == 0;
		if (suredivided)
		{
			if (kernel_H == 2 && kernel_W == 2 && 0)
			{
				zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel2x2(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else if (kernel_H == 3 && kernel_W == 3)
			{
				zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel3x3(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else if (kernel_H == 5 && kernel_W == 5)
			{
				zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_kernel5x5(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else
			{
				zq_cnn_maxpooling_nopadding_suredivided_32f_align128bit_general(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
		}
		else
		{
			zq_cnn_maxpooling_nopadding_nodivided_32f_align128bit_general(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
				kernel_H, kernel_W, stride_H, stride_W,
				out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
		}
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		bool suredivided = (in_H - kernel_H) % stride_H == 0 && (in_W - kernel_W) % stride_W == 0;
		if (suredivided)
		{
			if (kernel_H == 2 && kernel_W == 2 && 0)
			{
				zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_kernel2x2(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else if (kernel_H == 3 && kernel_W == 3)
			{
				zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_kernel3x3(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else if (kernel_H == 5 && kernel_W == 5)
			{
				zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_kernel5x5(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else
			{
				zq_cnn_maxpooling_nopadding_suredivided_32f_align256bit_general(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
		}
		else
		{
			zq_cnn_maxpooling_nopadding_nodivided_32f_align256bit_general(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
				kernel_H, kernel_W, stride_H, stride_W,
				out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
		}
#endif
	}
	else
	{
		zq_cnn_maxpooling_nopadding_32f_align0_general(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
			kernel_H, kernel_W, stride_H, stride_W,
			out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
	}
}
#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_avgpooling(int align_mode, const float* in_data, int N, int in_H, int in_W, int C,
	int in_pixStep, int in_widthStep, int in_sliceStep,
	int kernel_H, int kernel_W, int stride_H, int stride_W, float* out_data, int out_H, int out_W, int
	out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (C == 1)
		align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_0);
	else if (C <= 4)
		align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_128bit);
	else if (C <= 8)
		align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_256bit);

	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		bool suredivided = (in_H - kernel_H) % stride_H == 0 && (in_W - kernel_W) % stride_W == 0;
		if (suredivided)
		{
			if (kernel_H == 2 && kernel_W == 2 && 0)
			{
				zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_kernel2x2(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else if (kernel_H == 3 && kernel_W == 3)
			{
				zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_kernel3x3(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else if (kernel_H == 5 && kernel_W == 5)
			{
				zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_kernel5x5(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else
			{
				zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_general(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
		}
		else
		{
			zq_cnn_avgpooling_nopadding_nodivided_32f_align128bit_general(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
				kernel_H, kernel_W, stride_H, stride_W,
				out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
		}
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_avgpooling_nopadding_32f_align0_general(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
			kernel_H, kernel_W, stride_H, stride_W,
			out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_avgpooling(int align_mode, const float* in_data, int N, int in_H, int in_W, int C,
	int in_pixStep, int in_widthStep, int in_sliceStep,
	int kernel_H, int kernel_W, int stride_H, int stride_W, float* out_data, int out_H, int out_W, int
	out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (C == 1)
		align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_0);
	else if (C <= 4)
		align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_128bit);
	else if (C <= 8)
		align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_256bit);

	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		bool suredivided = (in_H - kernel_H) % stride_H == 0 && (in_W - kernel_W) % stride_W == 0;
		if (suredivided)
		{
			if (kernel_H == 2 && kernel_W == 2 && 0)
			{
				zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_kernel2x2(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else if (kernel_H == 3 && kernel_W == 3)
			{
				zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_kernel3x3(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else if (kernel_H == 5 && kernel_W == 5)
			{
				zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_kernel5x5(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else
			{
				zq_cnn_avgpooling_nopadding_suredivided_32f_align128bit_general(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
		}
		else
		{
			zq_cnn_avgpooling_nopadding_nodivided_32f_align128bit_general(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
				kernel_H, kernel_W, stride_H, stride_W,
				out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
		}
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		bool suredivided = (in_H - kernel_H) % stride_H == 0 && (in_W - kernel_W) % stride_W == 0;
		if (suredivided)
		{
			if (kernel_H == 2 && kernel_W == 2 && 0)
			{
				zq_cnn_avgpooling_nopadding_suredivided_32f_align256bit_kernel2x2(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else if (kernel_H == 3 && kernel_W == 3)
			{
				zq_cnn_avgpooling_nopadding_suredivided_32f_align256bit_kernel3x3(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else if (kernel_H == 5 && kernel_W == 5)
			{
				zq_cnn_avgpooling_nopadding_suredivided_32f_align256bit_kernel5x5(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else
			{
				zq_cnn_avgpooling_nopadding_suredivided_32f_align256bit_general(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
					kernel_H, kernel_W, stride_H, stride_W,
					out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
			}
		}
		else
		{
			zq_cnn_avgpooling_nopadding_nodivided_32f_align256bit_general(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
				kernel_H, kernel_W, stride_H, stride_W,
				out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
		}
#endif
	}
	else
	{
		zq_cnn_avgpooling_nopadding_32f_align0_general(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
			kernel_H, kernel_W, stride_H, stride_W,
			out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
	}
}
#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_batchnorm(int align_mode, float* data, int N, int H, int W, int C, 
	int pixStep, int widthStep, int sliceStep, const float* mean, const float* var, const float eps)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_batchnorm_32f_mean_var_align128bit(data, N, H, W, C, pixStep, widthStep, sliceStep, mean, var, eps);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_batchnorm_32f_mean_var_align0(data, N, H, W, C, pixStep, widthStep, sliceStep, mean, var, eps);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_batchnorm(int align_mode, float* data, int N, int H, int W, int C,
	int pixStep, int widthStep, int sliceStep, const float* mean, const float* var, const float eps)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_batchnorm_32f_mean_var_align128bit(data, N, H, W, C, pixStep, widthStep, sliceStep, mean, var, eps);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_batchnorm_32f_mean_var_align256bit(data, N, H, W, C, pixStep, widthStep, sliceStep, mean, var, eps);
#endif
	}
	else
	{
		zq_cnn_batchnorm_32f_mean_var_align0(data, N, H, W, C, pixStep, widthStep, sliceStep, mean, var, eps);
	}
}
#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_batchnorm_scalebias(int align_mode, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	const float* mean, const float* var, const float* scale, const float* bias, const float eps)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_batchnormscale_32f_mean_var_scale_bias_align128bit(data, N, H, W, C, pixStep, widthStep, sliceStep, mean, var, scale,bias, eps);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_batchnormscale_32f_mean_var_scale_bias_align0(data, N, H, W, C, pixStep, widthStep, sliceStep, mean, var, scale, bias, eps);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_batchnorm_scalebias(int align_mode, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	const float* mean, const float* var, const float* scale, const float* bias, const float eps)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_batchnormscale_32f_mean_var_scale_bias_align128bit(data, N, H, W, C, pixStep, widthStep, sliceStep, mean, var, scale, bias, eps);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_batchnormscale_32f_mean_var_scale_bias_align256bit(data, N, H, W, C, pixStep, widthStep, sliceStep, mean, var, scale, bias, eps);
#endif
	}
	else
	{
		zq_cnn_batchnormscale_32f_mean_var_scale_bias_align0(data, N, H, W, C, pixStep, widthStep, sliceStep, mean, var, scale, bias, eps);
	}
}
#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_batchnorm_b_a(int align_mode, float* data, int N, int H, int W, int C, 
	int pixStep, int widthStep, int sliceStep, const float* b, const float* a)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_batchnorm_32f_b_a_align128bit(data, N, H, W, C, pixStep, widthStep, sliceStep, b,a);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_batchnorm_32f_b_a_align0(data, N, H, W, C, pixStep, widthStep, sliceStep, b,a);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_batchnorm_b_a(int align_mode, float* data, int N, int H, int W, int C,
	int pixStep, int widthStep, int sliceStep, const float* b, const float* a)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_batchnorm_32f_b_a_align128bit(data, N, H, W, C, pixStep, widthStep, sliceStep, b, a);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_batchnorm_32f_b_a_align256bit(data, N, H, W, C, pixStep, widthStep, sliceStep, b, a);
#endif
	}
	else
	{
		zq_cnn_batchnorm_32f_b_a_align0(data, N, H, W, C, pixStep, widthStep, sliceStep, b, a);
	}
}
#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_scalebias(int align_mode, float* data, int N, int H, int W, int C, 
	int pixStep, int widthStep, int sliceStep, const float* scale, const float* bias)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_scale_32f_align128bit(data, N, H, W, C, pixStep, widthStep, sliceStep, scale, bias);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_scale_32f_align0(data, N, H, W, C, pixStep, widthStep, sliceStep, scale, bias);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_scalebias(int align_mode, float* data, int N, int H, int W, int C,
	int pixStep, int widthStep, int sliceStep, const float* scale, const float* bias)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_scale_32f_align128bit(data, N, H, W, C, pixStep, widthStep, sliceStep, scale, bias);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_scale_32f_align256bit(data, N, H, W, C, pixStep, widthStep, sliceStep, scale, bias);
#endif
	}
	else
	{
		zq_cnn_scale_32f_align0(data, N, H, W, C, pixStep, widthStep, sliceStep, scale, bias);
	}
}
#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_eltwise_sum(int align_mode, int in_tensor_num, const float** in_data, int N, int H, int W, int C, const int* pixStep, const int* widthStep, const int* sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_eltwise_sum_32f_align128bit(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep,out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_eltwise_sum_32f_align0(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_eltwise_sum(int align_mode, int in_tensor_num, const float** in_data, int N, int H, int W, int C, const int* pixStep, const int* widthStep, const int* sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_eltwise_sum_32f_align128bit(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_eltwise_sum_32f_align256bit(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else
	{
		zq_cnn_eltwise_sum_32f_align0(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_eltwise_sum_with_weight(int align_mode, int in_tensor_num, const float** in_data, const float* weight, int N, int H, int W, int C,
	const int* pixStep, const int* widthStep, const int* sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_eltwise_sum_with_weight_32f_align128bit(in_tensor_num, in_data, weight, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_eltwise_sum_with_weight_32f_align0(in_tensor_num, in_data, weight, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_eltwise_sum_with_weight(int align_mode, int in_tensor_num, const float** in_data, const float* weight, int N, int H, int W, int C,
	const int* pixStep, const int* widthStep, const int* sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_eltwise_sum_with_weight_32f_align128bit(in_tensor_num, in_data, weight, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_eltwise_sum_with_weight_32f_align256bit(in_tensor_num, in_data, weight, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else
	{
		zq_cnn_eltwise_sum_with_weight_32f_align0(in_tensor_num, in_data, weight, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}
#endif//__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_eltwise_mul(int align_mode, int in_tensor_num, const float** in_data, int N, int H, int W, int C, const int* pixStep, const int* widthStep, const int* sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_eltwise_mul_32f_align128bit(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_eltwise_mul_32f_align0(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_eltwise_mul(int align_mode, int in_tensor_num, const float** in_data, int N, int H, int W, int C, const int* pixStep, const int* widthStep, const int* sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_eltwise_mul_32f_align128bit(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_eltwise_mul_32f_align256bit(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else
	{
		zq_cnn_eltwise_mul_32f_align0(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}
#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_eltwise_max(int align_mode, int in_tensor_num, const float** in_data, int N, int H, int W, int C, const int* pixStep, const int* widthStep, const int* sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_eltwise_max_32f_align128bit(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_eltwise_max_32f_align0(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_eltwise_max(int align_mode, int in_tensor_num, const float** in_data, int N, int H, int W, int C, const int* pixStep, const int* widthStep, const int* sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_eltwise_max_32f_align128bit(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_eltwise_max_32f_align256bit(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else
	{
		zq_cnn_eltwise_max_32f_align0(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}
#endif


void ZQ_CNN_Forward_SSEUtils::_reduction_sum(int align_mode, const float* in_data, int N, int H, int W, int C,  int axis, bool keepdims,
	int pixStep, int widthStep, int sliceStep, float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	zq_cnn_reduction_sum_32f_align0(in_data, N, H, W, C, axis, keepdims, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
}

void ZQ_CNN_Forward_SSEUtils::_reduction_mean(int align_mode, const float* in_data, int N, int H, int W, int C, int axis, bool keepdims,
	int pixStep, int widthStep, int sliceStep, float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	zq_cnn_reduction_mean_32f_align0(in_data, N, H, W, C, axis, keepdims, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
}

void ZQ_CNN_Forward_SSEUtils::_sqrt(int align_mode, float* in_data, int N, int H, int W, int C,
	int pixStep, int widthStep, int sliceStep)
{
	zq_cnn_sqrt_32f_align0(in_data, N, H, W, C, pixStep, widthStep, sliceStep);
}

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_scalaroperation_add(int align_mode, float scalar, const float* in_data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_scalaroperation_add_32f_align128bit(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_scalaroperation_add_32f_align0(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_scalaroperation_add(int align_mode, float scalar, const float* in_data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_scalaroperation_add_32f_align128bit(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_scalaroperation_add_32f_align256bit(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else
	{
		zq_cnn_scalaroperation_add_32f_align0(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}
#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_scalaroperation_add(int align_mode, float scalar, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_scalaroperation_add_inplace_32f_align128bit(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_scalaroperation_add_inplace_32f_align0(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_scalaroperation_add(int align_mode, float scalar, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_scalaroperation_add_inplace_32f_align128bit(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_scalaroperation_add_inplace_32f_align256bit(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
#endif
	}
	else
	{
		zq_cnn_scalaroperation_add_inplace_32f_align0(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
}
#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_scalaroperation_mul(int align_mode, float scalar, const float* in_data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_scalaroperation_mul_32f_align128bit(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_scalaroperation_mul_32f_align0(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_scalaroperation_mul(int align_mode, float scalar, const float* in_data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_scalaroperation_mul_32f_align128bit(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_scalaroperation_mul_32f_align256bit(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else
	{
		zq_cnn_scalaroperation_mul_32f_align0(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_scalaroperation_mul(int align_mode, float scalar, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_scalaroperation_mul_inplace_32f_align128bit(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_scalaroperation_mul_inplace_32f_align0(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_scalaroperation_mul(int align_mode, float scalar, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_scalaroperation_mul_inplace_32f_align128bit(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_scalaroperation_mul_inplace_32f_align256bit(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
#endif
	}
	else
	{
		zq_cnn_scalaroperation_mul_inplace_32f_align0(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
}

#endif//__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_scalaroperation_max(int align_mode, float scalar, const float* in_data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_scalaroperation_max_32f_align128bit(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_scalaroperation_max_32f_align0(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_scalaroperation_max(int align_mode, float scalar, const float* in_data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_scalaroperation_max_32f_align128bit(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_scalaroperation_max_32f_align256bit(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else
	{
		zq_cnn_scalaroperation_max_32f_align0(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_scalaroperation_max(int align_mode, float scalar, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_scalaroperation_max_inplace_32f_align128bit(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_scalaroperation_max_inplace_32f_align0(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_scalaroperation_max(int align_mode, float scalar, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_scalaroperation_max_inplace_32f_align128bit(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_scalaroperation_max_inplace_32f_align256bit(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
#endif
	}
	else
	{
		zq_cnn_scalaroperation_max_inplace_32f_align0(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
}
#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_scalaroperation_min(int align_mode, float scalar, const float* in_data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_scalaroperation_min_32f_align128bit(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_scalaroperation_min_32f_align0(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_scalaroperation_min(int align_mode, float scalar, const float* in_data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_scalaroperation_min_32f_align128bit(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_scalaroperation_min_32f_align256bit(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else
	{
		zq_cnn_scalaroperation_min_32f_align0(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_scalaroperation_min(int align_mode, float scalar, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_scalaroperation_min_inplace_32f_align128bit(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_scalaroperation_min_inplace_32f_align0(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_scalaroperation_min(int align_mode, float scalar, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_scalaroperation_min_inplace_32f_align128bit(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_scalaroperation_min_inplace_32f_align256bit(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
#endif
	}
	else
	{
		zq_cnn_scalaroperation_min_inplace_32f_align0(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
}

#endif //__ARM_NEON

#if __ARM_NEON

void ZQ_CNN_Forward_SSEUtils::_scalaroperation_rdiv(int align_mode, float scalar, const float* in_data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	/*if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_scalaroperation_rdiv_32f_align128bit(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else*/
	{
		zq_cnn_scalaroperation_rdiv_32f_align0(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_scalaroperation_rdiv(int align_mode, float scalar, const float* in_data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_scalaroperation_rdiv_32f_align128bit(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_scalaroperation_rdiv_32f_align256bit(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else
	{
		zq_cnn_scalaroperation_rdiv_32f_align0(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

#endif //__ARM_NEON

#if __ARM_NEON

void ZQ_CNN_Forward_SSEUtils::_scalaroperation_rdiv(int align_mode, float scalar, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep)
{
	/*if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_scalaroperation_rdiv_inplace_32f_align128bit(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else*/
	{
		zq_cnn_scalaroperation_rdiv_inplace_32f_align0(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_scalaroperation_rdiv(int align_mode, float scalar, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_scalaroperation_rdiv_inplace_32f_align128bit(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_scalaroperation_rdiv_inplace_32f_align256bit(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
#endif
	}
	else
	{
		zq_cnn_scalaroperation_rdiv_inplace_32f_align0(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
}

#endif

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_scalaroperation_rminus(int align_mode, float scalar, const float* in_data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_scalaroperation_rminus_32f_align128bit(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_scalaroperation_rminus_32f_align0(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_scalaroperation_rminus(int align_mode, float scalar, const float* in_data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_scalaroperation_rminus_32f_align128bit(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_scalaroperation_rminus_32f_align256bit(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else
	{
		zq_cnn_scalaroperation_rminus_32f_align0(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_scalaroperation_rminus(int align_mode, float scalar, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_scalaroperation_rminus_inplace_32f_align128bit(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_scalaroperation_rminus_inplace_32f_align0(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_scalaroperation_rminus(int align_mode, float scalar, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_scalaroperation_rminus_inplace_32f_align128bit(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_scalaroperation_rminus_inplace_32f_align256bit(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
#endif
	}
	else
	{
		zq_cnn_scalaroperation_rminus_inplace_32f_align0(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
}
#endif 

void ZQ_CNN_Forward_SSEUtils::_scalaroperation_pow(int align_mode, float scalar, const float* in_data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{	
	zq_cnn_scalaroperation_pow_32f_align0(scalar, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
}

void ZQ_CNN_Forward_SSEUtils::_scalaroperation_pow(int align_mode, float scalar, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep)
{
	zq_cnn_scalaroperation_pow_inplace_32f_align0(scalar, data, N, H, W, C, pixStep, widthStep, sliceStep);
}

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_lrn_across_channels(int align_mode, int local_size, float alpha, float beta, float k, 
	const float* in_data, int N, int H, int W, int C,int in_pixStep, int in_widthStep, int in_sliceStep, 
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	
	zq_cnn_lrn_across_channels_32f_align0(local_size, alpha, beta, k, in_data, N, H, W, C, in_pixStep, in_widthStep, in_sliceStep,
		out_data, out_pixStep, out_widthStep, out_sliceStep);
}

#else

void ZQ_CNN_Forward_SSEUtils::_lrn_across_channels(int align_mode, int local_size, float alpha, float beta, float k,
	const float* in_data, int N, int H, int W, int C, int in_pixStep, int in_widthStep, int in_sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		zq_cnn_lrn_across_channels_32f_align128bit(local_size, alpha, beta, k, in_data, N, H, W, C, in_pixStep, in_widthStep, in_sliceStep,
			out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		zq_cnn_lrn_across_channels_32f_align256bit(local_size, alpha, beta, k, in_data, N, H, W, C, in_pixStep, in_widthStep, in_sliceStep,
			out_data, out_pixStep, out_widthStep, out_sliceStep);
#endif
	}
	else
	{
		zq_cnn_lrn_across_channels_32f_align0(local_size, alpha, beta, k, in_data, N, H, W, C, in_pixStep, in_widthStep, in_sliceStep,
			out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}
#endif //__ARM_NEON

#if __ARM_NEON
void ZQ_CNN_Forward_SSEUtils::_normalize(int align_mode, bool across_spatial, bool channel_shared, float* in_data, const float* scale_data, int N, int H, int W, int C,
	int in_pixStep, int in_widthStep, int in_sliceStep, const float eps)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		if(across_spatial)
			zq_cnn_normalize_across_spatial_32f_align128bit(channel_shared, in_data, scale_data, N, H, W, C, in_pixStep, in_widthStep, in_sliceStep, eps);
		else
			zq_cnn_normalize_not_across_spatial_32f_align128bit(channel_shared, in_data, scale_data, N, H, W, C, in_pixStep, in_widthStep, in_sliceStep, eps);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
	}
	else
	{
		zq_cnn_normalize_32f_align0(across_spatial, channel_shared, in_data, scale_data, N, H, W, C, in_pixStep, in_widthStep, in_sliceStep, eps);
	}
}

#else

void ZQ_CNN_Forward_SSEUtils::_normalize(int align_mode, bool across_spatial, bool channel_shared, float* in_data, const float* scale_data, int N, int H, int W, int C,
	int in_pixStep, int in_widthStep, int in_sliceStep, const float eps)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		if (across_spatial)
			zq_cnn_normalize_across_spatial_32f_align128bit(channel_shared, in_data, scale_data, N, H, W, C, in_pixStep, in_widthStep, in_sliceStep, eps);
		else
			zq_cnn_normalize_not_across_spatial_32f_align128bit(channel_shared, in_data, scale_data, N, H, W, C, in_pixStep, in_widthStep, in_sliceStep, eps);
#endif
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		if (across_spatial)
			zq_cnn_normalize_across_spatial_32f_align256bit(channel_shared, in_data, scale_data, N, H, W, C, in_pixStep, in_widthStep, in_sliceStep, eps);
		else
			zq_cnn_normalize_not_across_spatial_32f_align256bit(channel_shared, in_data, scale_data, N, H, W, C, in_pixStep, in_widthStep, in_sliceStep, eps);
#endif
	}
	else
	{
		zq_cnn_normalize_32f_align0(across_spatial, channel_shared, in_data, scale_data, N, H, W, C, in_pixStep, in_widthStep, in_sliceStep, eps);
	}
}
#endif //__ARM_NEON

bool ZQ_CNN_Forward_SSEUtils::_prior_box(const ZQ_CNN_Tensor4D& input, const ZQ_CNN_Tensor4D& data,
	const std::vector<float>& min_sizes, const std::vector<float>& max_sizes,
	const std::vector<float>& aspect_ratios, const std::vector<float>& variance,
	bool flip, int num_priors, bool clip, int img_w, int img_h, float step_w, float step_h, float offset,
	ZQ_CNN_Tensor4D& output)
{
	const int layer_width = input.GetW();
	const int layer_height = input.GetH();
	if (layer_width <= 0 || layer_height <= 0)
	{
		output.ChangeSize(0, 0, 0, 0, 0, 0);
		return true;
	}
	int img_width, img_height;
	if (img_h == 0 || img_w == 0) 
	{
		img_width = data.GetW();
		img_height = data.GetH();
	}
	else 
	{
		img_width = img_w;
		img_height = img_h;
	}
	float step_width, step_height;
	if (step_w == 0 || step_h == 0) 
	{
		step_width = (float)img_width / layer_width;
		step_height = (float)img_height / layer_height;
	}
	else 
	{
		step_width = step_w;
		step_height = step_h;
	}

	int dim = layer_height * layer_width * num_priors * 4;
	int out_N = data.GetN();
	int out_C = 2;
	int out_H = dim;
	int out_W = 1;
	if (output.GetN() != out_N || output.GetC() != out_C || output.GetH() != out_H || output.GetW() != out_W)
		output.ChangeSize(out_N, out_H, out_W, out_C, 0, 0);

	int pixStep = output.GetPixelStep();
	for (int n = 0; n < out_N; n++)
	{
		float* out_ptr = output.GetFirstPixelPtr() + n*output.GetSliceStep();
		float* cur_ptr = out_ptr;
		int idx = 0;
		for (int h = 0; h < layer_height; h++)
		{
			for (int w = 0; w < layer_width; w++)
			{
				float center_x = (w + offset) * step_width;
				float center_y = (h + offset) * step_height;
				float box_width, box_height;
				for (int s = 0; s < min_sizes.size(); s++)
				{
					int cur_min_size = min_sizes[s];
					// first prior: aspect_ratio = 1, size = min_size
					box_width = box_height = cur_min_size;
					// xmin
					*cur_ptr = (center_x - box_width / 2.) / img_width;
					cur_ptr += pixStep;
					// ymin
					*cur_ptr = (center_y - box_height / 2.) / img_height;
					cur_ptr += pixStep;
					// xmax
					*cur_ptr = (center_x + box_width / 2.) / img_width;
					cur_ptr += pixStep;
					// ymax
					*cur_ptr = (center_y + box_height / 2.) / img_height;
					cur_ptr += pixStep;

					if (max_sizes.size() > 0)
					{
						if (min_sizes.size() != max_sizes.size())
							return false;
						int cur_max_size = max_sizes[s];
						// second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
						box_width = box_height = sqrt(cur_min_size * cur_max_size);
						// xmin
						*cur_ptr = (center_x - box_width / 2.) / img_width;
						cur_ptr += pixStep;
						// ymin
						*cur_ptr = (center_y - box_height / 2.) / img_height;
						cur_ptr += pixStep;
						// xmax
						*cur_ptr = (center_x + box_width / 2.) / img_width;
						cur_ptr += pixStep;
						// ymax
						*cur_ptr = (center_y + box_height / 2.) / img_height;
						cur_ptr += pixStep;
					}

					// rest of priors
					for (int r = 0; r < aspect_ratios.size(); r++)
					{
						float ar = aspect_ratios[r];
						if (fabs(ar - 1.0f) < 1e-6)
							continue;
						box_width = cur_min_size * sqrt(ar);
						box_height = cur_min_size / sqrt(ar);
						// xmin
						*cur_ptr = (center_x - box_width / 2.) / img_width;
						cur_ptr += pixStep;
						// ymin
						*cur_ptr = (center_y - box_height / 2.) / img_height;
						cur_ptr += pixStep;
						// xmax
						*cur_ptr = (center_x + box_width / 2.) / img_width;
						cur_ptr += pixStep;
						// ymax
						*cur_ptr = (center_y + box_height / 2.) / img_height;
						cur_ptr += pixStep;
					}
				}
			}
		}
		// clip the prior's coordidate such that it is within [0, 1]
		if (clip)
		{
			for (int d = 0; d < dim; ++d)
			{
				out_ptr[d*pixStep] = __min(1.0f, __max(0.0f, out_ptr[d*pixStep]));
			}
		}
		// set the variance.
		cur_ptr = out_ptr + 1;
		if (variance.size() == 1)
		{
			for (int i = 0; i < dim; i++)
				cur_ptr[i*pixStep] = variance[0];
		}
		else if(variance.size() == 4)
		{
			for (int h = 0; h < layer_height; h++)
			{
				for (int w = 0; w < layer_width; w++)
				{
					for (int i = 0; i < num_priors; i++)
					{
						for (int j = 0; j < 4; ++j)
						{
							*cur_ptr = variance[j];
							cur_ptr += pixStep;
						}
					}
				}
			}
		}
		else
		{
			return false;
		}
	}
	
	return true;
}

bool ZQ_CNN_Forward_SSEUtils::_prior_box_MXNET(const ZQ_CNN_Tensor4D& input,
	const std::vector<float>& sizes, const std::vector<float>& aspect_ratios, const std::vector<float>& variance,
	int num_priors, bool clip, float step_w, float step_h, float offset,
	ZQ_CNN_Tensor4D& output)
{
	const int layer_width = input.GetW();
	const int layer_height = input.GetH();
	if (layer_width <= 0 || layer_height <= 0)
	{
		output.ChangeSize(0, 0, 0, 0, 0, 0);
		return true;
	}

	float step_width, step_height;
	if (step_w <= 0 || step_h <= 0)
	{
		step_width = 1.0f / layer_width;
		step_height = 1.0f / layer_height;
	}
	else
	{
		step_width = step_w;
		step_height = step_h;
	}

	int dim = layer_height * layer_width * num_priors;
	int out_C = 1;
	int out_H = dim;
	int out_W = 4;
	if (output.GetC() != out_C || output.GetH() != out_H || output.GetW() != out_W)
		output.ChangeSize(1, out_H, out_W, out_C, 0, 0);

	int num_sizes = sizes.size();
	int num_ratios = aspect_ratios.size();
	int pixStep = output.GetPixelStep();
	float* out_ptr = output.GetFirstPixelPtr();
	float* cur_ptr = out_ptr;
	int idx = 0;
	for (int h = 0; h < layer_height; h++)
	{
		for (int w = 0; w < layer_width; w++)
		{
			float center_x = (w + offset) * step_width;
			float center_y = (h + offset) * step_height;
			float box_width, box_height;

			// ratio = 1, various sizes
			for (int i = 0; i < num_sizes; ++i)
			{
				float size = sizes[i];
				float w = size * layer_height / layer_width / 2;
				float h = size / 2;

				*cur_ptr = center_x - w;  // xmin
				cur_ptr += pixStep;
				*cur_ptr = center_y - h;  // ymin
				cur_ptr += pixStep;
				*cur_ptr = center_x + w;  // xmax
				cur_ptr += pixStep;
				*cur_ptr = center_y + h;  // ymax
				cur_ptr += pixStep;
			}
			// various ratios, size = min_size = size[0]
			float size = sizes[0];
			for (int j = 1; j < num_ratios; ++j)
			{
				float ratio = sqrtf(aspect_ratios[j]);
				float w = size *layer_height / layer_width * ratio / 2;
				float h = size / ratio / 2;
				*cur_ptr = center_x - w;  // xmin
				cur_ptr += pixStep;
				*cur_ptr = center_y - h;  // ymin
				cur_ptr += pixStep;
				*cur_ptr = center_x + w;  // xmax
				cur_ptr += pixStep;
				*cur_ptr = center_y + h;  // ymax
				cur_ptr += pixStep;
			}
		}
	}	
	return true;
}

bool ZQ_CNN_Forward_SSEUtils::_prior_box_text(const ZQ_CNN_Tensor4D& input, const ZQ_CNN_Tensor4D& data,
	const std::vector<float>& min_sizes, const std::vector<float>& max_sizes,
	const std::vector<float>& aspect_ratios, const std::vector<float>& variance,
	bool flip, int num_priors, bool clip, int img_w, int img_h, float step_w, float step_h, float offset,
	ZQ_CNN_Tensor4D& output)
{
	const int layer_width = input.GetW();
	const int layer_height = input.GetH();
	if (layer_width <= 0 || layer_height <= 0)
	{
		output.ChangeSize(0, 0, 0, 0, 0, 0);
		return true;
	}
	int img_width, img_height;
	if (img_h == 0 || img_w == 0)
	{
		img_width = data.GetW();
		img_height = data.GetH();
	}
	else
	{
		img_width = img_w;
		img_height = img_h;
	}
	float step_width, step_height;
	if (step_w == 0 || step_h == 0)
	{
		step_width = (float)img_width / layer_width;
		step_height = (float)img_height / layer_height;
	}
	else
	{
		step_width = step_w;
		step_height = step_h;
	}

	int dim = layer_height * layer_width * num_priors * 4;
	int out_N = data.GetN();
	int out_C = 2;
	int out_H = dim;
	int out_W = 1;
	if (output.GetN() != out_N || output.GetC() != out_C || output.GetH() != out_H || output.GetW() != out_W)
		output.ChangeSize(out_N, out_H, out_W, out_C, 0, 0);

	int pixStep = output.GetPixelStep();
	for (int n = 0; n < out_N; n++)
	{
		float* out_ptr = output.GetFirstPixelPtr() + n*output.GetSliceStep();
		float* cur_ptr = out_ptr;
		int idx = 0;
		for (int h = 0; h < layer_height; h++)
		{
			for (int w = 0; w < layer_width; w++)
			{
				float center_x = (w + 0.5) * step_width;
				float center_y = (h + 0.5) * step_height;
				float center_y_offset_1 = (h + 1.0) * step_height;
				float box_width, box_height;
				for (int s = 0; s < min_sizes.size(); s++)
				{
					int cur_min_size = min_sizes[s];
					// first prior: aspect_ratio = 1, size = min_size
					box_width = box_height = cur_min_size;
					// xmin
					*cur_ptr = (center_x - box_width / 2.0f) / img_width;
					cur_ptr += pixStep;
					// ymin
					*cur_ptr = (center_y - box_height / 2.0f) / img_height;
					cur_ptr += pixStep;
					
					// xmax
					*cur_ptr = (center_x + box_width / 2.0f) / img_width;
					cur_ptr += pixStep;

					// ymax
					*cur_ptr = (center_y + box_height / 2.0f) / img_height;
					cur_ptr += pixStep;

					
					// xmin
					*cur_ptr = (center_x - box_width / 2.0f) / img_width;
					cur_ptr += pixStep;
					// ymin
					*cur_ptr = (center_y_offset_1 - box_height / 2.0f) / img_height;
					cur_ptr += pixStep;
					// xmax
					*cur_ptr = (center_x + box_width / 2.0f) / img_width;
					cur_ptr += pixStep;
					// ymax
					*cur_ptr = (center_y_offset_1 + box_height / 2.0f) / img_height;
					cur_ptr += pixStep;

					
					if (max_sizes.size() > 0)
					{
						if (min_sizes.size() != max_sizes.size())
							return false;
						int cur_max_size = max_sizes[s];
						// second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
						box_width = box_height = sqrt(cur_min_size * cur_max_size);
						// xmin
						*cur_ptr = (center_x - box_width / 2.0f) / img_width;
						cur_ptr += pixStep;
						// ymin
						*cur_ptr = (center_y - box_height / 2.0f) / img_height;
						cur_ptr += pixStep;
						// xmax
						*cur_ptr = (center_x + box_width / 2.0f) / img_width;
						cur_ptr += pixStep;
						// ymax
						*cur_ptr = (center_y + box_height / 2.0f) / img_height;
						cur_ptr += pixStep;

						// xmin
						*cur_ptr = (center_x - box_width / 2.0f) / img_width;
						cur_ptr += pixStep;
						// ymin
						*cur_ptr = (center_y_offset_1 - box_height / 2.0f) / img_height;
						cur_ptr += pixStep;
						// xmax
						*cur_ptr = (center_x + box_width / 2.0f) / img_width;
						cur_ptr += pixStep;
						// ymax
						*cur_ptr = (center_y_offset_1 + box_height / 2.0f) / img_height;
						cur_ptr += pixStep;
						
					}

					// rest of priors
					for (int r = 0; r < aspect_ratios.size(); r++)
					{
						float ar = aspect_ratios[r];
						if (fabs(ar - 1.0f) < 1e-6)
							continue;
						box_width = cur_min_size * sqrt(ar);
						box_height = cur_min_size / sqrt(ar);
						// xmin
						*cur_ptr = (center_x - box_width / 2.0f) / img_width;
						cur_ptr += pixStep;
						// ymin
						*cur_ptr = (center_y - box_height / 2.0f) / img_height;
						cur_ptr += pixStep;
						// xmax
						*cur_ptr = (center_x + box_width / 2.0f) / img_width;
						cur_ptr += pixStep;
						// ymax
						*cur_ptr = (center_y + box_height / 2.0f) / img_height;
						cur_ptr += pixStep;

						
						// xmin
						*cur_ptr = (center_x - box_width / 2.0f) / img_width;
						cur_ptr += pixStep;
						// ymin
						*cur_ptr = (center_y_offset_1 - box_height / 2.0f) / img_height;
						cur_ptr += pixStep;
						// xmax
						*cur_ptr = (center_x + box_width / 2.0f) / img_width;
						cur_ptr += pixStep;
						// ymax
						*cur_ptr = (center_y_offset_1 + box_height / 2.0f) / img_height;
						cur_ptr += pixStep;
					}
				}
			}
		}
		// clip the prior's coordidate such that it is within [0, 1]
		if (clip)
		{
			for (int d = 0; d < dim; ++d)
			{
				out_ptr[d*pixStep] = __min(1.0f, __max(0.0f, out_ptr[d*pixStep]));
			}
		}
		// set the variance.
		cur_ptr = out_ptr + 1;
		if (variance.size() == 1)
		{
			for (int i = 0; i < dim; i++)
				cur_ptr[i*pixStep] = variance[0];
		}
		else if (variance.size() == 4)
		{
			for (int h = 0; h < layer_height; h++)
			{
				for (int w = 0; w < layer_width; w++)
				{
					for (int i = 0; i < num_priors; i++)
					{
						for (int j = 0; j < 4; ++j)
						{
							*cur_ptr = variance[j];
							cur_ptr += pixStep;
						}
					}
				}
			}
		}
		else
		{
			return false;
		}
	}

	return true;
}

bool ZQ_CNN_Forward_SSEUtils::_concat_NCHW_get_size(const std::vector<ZQ_CNN_Tensor4D*>& inputs, int axis, int& out_N, int& out_C, int& out_H, int& out_W)
{
	if (axis < 0 || axis >= 4)
		return false;
	int in_num = inputs.size();
	std::vector<ZQ_CNN_Tensor4D*> valid_inputs;
	for (int i = 0; i < inputs.size(); i++)
	{
		if (inputs[i] == 0)
			continue;
		inputs[i]->GetShape(out_N, out_C, out_H, out_W);
		if (out_N > 0 && out_C > 0 && out_H > 0 && out_W > 0)
			valid_inputs.push_back(inputs[i]);
	}

	if (valid_inputs.size() == 0)
	{
		out_N = out_H = out_W = out_C = 0;
		return true;
	}
	else if (valid_inputs.size() == 1)
	{
		valid_inputs[0]->GetShape(out_N, out_C, out_H, out_W);
		return true;
	}
	else
	{
		int standard_dim[4];
		valid_inputs[0]->GetShape(standard_dim[0], standard_dim[1], standard_dim[2], standard_dim[3]);
		int sum_out = standard_dim[axis];
		for (int i = 1; i < valid_inputs.size(); i++)
		{
			if (valid_inputs[i] == 0)
				return false;
			int cur_dim[4];
			valid_inputs[i]->GetShape(cur_dim[0], cur_dim[1], cur_dim[2], cur_dim[3]);
			for (int j = 0; j < 4; j++)
			{
				if (axis == j)
				{
					sum_out += cur_dim[j];
				}
				else if (cur_dim[j] != standard_dim[j])
				{
					return false;
				}
			}
		}
		standard_dim[axis] = sum_out;
		out_N = standard_dim[0];
		out_C = standard_dim[1];
		out_H = standard_dim[2];
		out_W = standard_dim[3];
		return true;
	}
}

bool ZQ_CNN_Forward_SSEUtils::_concat_NCHW(const std::vector<ZQ_CNN_Tensor4D*>& inputs, int axis, 
	ZQ_CNN_Tensor4D& output)
{
	int out_N, out_C, out_H, out_W;
	if (!_concat_NCHW_get_size(inputs, axis, out_N, out_C, out_H, out_W))
		return false;
	
	std::vector<ZQ_CNN_Tensor4D*> valid_inputs;
	for (int i = 0; i < inputs.size(); i++)
	{
		if (inputs[i] == 0)
			continue;
		
		if (inputs[i]->GetN() > 0 && inputs[i]->GetC() > 0 && inputs[i]->GetH() > 0 && inputs[i]->GetW() > 0)
			valid_inputs.push_back(inputs[i]);
	}

	if (axis < 0 || axis >= 4)
		return false;
	int in_num = valid_inputs.size();
	if (valid_inputs.size() == 0)
	{
		return output.ChangeSize(0, 0, 0, 0, 0, 0);
	}
	else if (valid_inputs.size() == 1)
	{
		return output.CopyData(*valid_inputs[0]);
	}
	else
	{
		if (output.GetN() != out_N || output.GetC() != out_C || output.GetH() != out_H || output.GetW() != out_W)
		{
			if (!output.ChangeSize(out_N, out_H, out_W, out_C, 0, 0))
				return false;
		}
		
		int out_pixStep = output.GetPixelStep();
		int out_widthStep = output.GetWidthStep();
		int out_sliceStep = output.GetSliceStep();
		
		float* out_ptr = output.GetFirstPixelPtr();
		for (int i = 0; i < valid_inputs.size(); i++)
		{
			int in_N = valid_inputs[i]->GetN();
			int in_C = valid_inputs[i]->GetC();
			int in_H = valid_inputs[i]->GetH();
			int in_W = valid_inputs[i]->GetW();
			int in_pixStep = valid_inputs[i]->GetPixelStep();
			int in_widthStep = valid_inputs[i]->GetWidthStep();
			int in_sliceStep = valid_inputs[i]->GetSliceStep();
			const float* in_ptr = valid_inputs[i]->GetFirstPixelPtr();
			const float* in_slice_ptr = in_ptr;
			float* out_slice_ptr = out_ptr;
			for (int n = 0; n < in_N; n++)
			{
				const float* in_row_ptr = in_slice_ptr;
				float* out_row_ptr = out_slice_ptr;
				for (int h = 0; h < in_H; h++)
				{
					const float* in_pix_ptr = in_row_ptr;
					float* out_pix_ptr = out_row_ptr;
					for (int w = 0; w < in_W; w++)
					{
						memcpy(out_pix_ptr, in_pix_ptr, sizeof(float)*in_C);
						in_pix_ptr += in_pixStep;
						out_pix_ptr += out_pixStep;
					}
					in_row_ptr += in_widthStep;
					out_row_ptr += out_widthStep;
				}
				in_slice_ptr += in_sliceStep;
				out_slice_ptr += out_sliceStep;
			}
			if (axis == 0)
				out_ptr += in_N*out_sliceStep;
			else if (axis == 1)
				out_ptr += in_C;
			else if (axis == 2)
				out_ptr += in_H*out_widthStep;
			else if (axis == 3)
				out_ptr += in_W*out_pixStep;
		}
		return true;
	}
}

bool ZQ_CNN_Forward_SSEUtils::_detection_output(const ZQ_CNN_Tensor4D& loc, const ZQ_CNN_Tensor4D& conf,
	const ZQ_CNN_Tensor4D& prior, int num_priors, int num_loc_classes, int num_classes, bool share_location,
	int background_label_id, ZQ_CNN_BBoxUtils::PriorBoxCodeType code_type, bool variance_encoded_in_target,
	float nms_thresh, float nms_eta, int nms_top_k, float confidence_thresh, int keep_top_k, 
	ZQ_CNN_Tensor4D& output)
{
	const int num = loc.GetN();

	int loc_len = loc.GetN() * loc.GetH() * loc.GetW() * loc.GetC();
	int conf_len = conf.GetN() * conf.GetH() * conf.GetW() * conf.GetC();
	int prior_len = prior.GetN() * prior.GetH() * prior.GetW() * prior.GetC();
	if (loc_len <= 0 || conf_len <= 0 || prior_len <= 0)
		return false;
	std::vector<float> loc_data(loc_len);
	std::vector<float> conf_data(conf_len);
	std::vector<float> prior_data(prior_len);
	loc.ConvertToCompactNCHW(&loc_data[0]);
	conf.ConvertToCompactNCHW(&conf_data[0]);
	prior.ConvertToCompactNCHW(&prior_data[0]);
	// Retrieve all location predictions.
	std::vector<ZQ_CNN_LabelBBox> all_loc_preds;
	ZQ_CNN_BBoxUtils::GetLocPredictions(&loc_data[0], num, num_priors, num_loc_classes, share_location, &all_loc_preds);

	// Retrieve all confidences.
	std::vector<std::map<int, std::vector<float> > > all_conf_scores;
	ZQ_CNN_BBoxUtils::GetConfidenceScores(&conf_data[0], num, num_priors, num_classes, &all_conf_scores);

	// Retrieve all prior bboxes. It is same within a batch since we assume all
	// images in a batch are of same dimension.
	std::vector<ZQ_CNN_NormalizedBBox> prior_bboxes;
	std::vector<std::vector<float> > prior_variances;
	ZQ_CNN_BBoxUtils::GetPriorBBoxes(&prior_data[0], num_priors, &prior_bboxes, &prior_variances);

	// Decode all loc predictions to bboxes.
	std::vector<ZQ_CNN_LabelBBox> all_decode_bboxes;
	const bool clip_bbox = false;
	if (!ZQ_CNN_BBoxUtils::DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num, share_location, num_loc_classes,
		background_label_id, code_type, variance_encoded_in_target, clip_bbox, &all_decode_bboxes))
		return false;

	int num_kept = 0;
	std::vector<std::map<int, std::vector<int> > > all_indices;
	for (int i = 0; i < num; ++i)
	{
		const ZQ_CNN_LabelBBox& decode_bboxes = all_decode_bboxes[i];
		const std::map<int, std::vector<float> >& conf_scores = all_conf_scores[i];
		std::map<int, std::vector<int> > indices;
		int num_det = 0;
		for (int c = 0; c < num_classes; ++c)
		{
			if (c == background_label_id)
			{
				// Ignore background class.
				continue;
			}
			if (conf_scores.find(c) == conf_scores.end())
			{
				// Something bad happened if there are no predictions for current label.
				printf("Could not find confidence predictions for label %d\n", c);
				//return false;
			}
			const std::vector<float>& scores = conf_scores.find(c)->second;
			int label = share_location ? -1 : c;
			if (decode_bboxes.find(label) == decode_bboxes.end())
			{
				// Something bad happened if there are no predictions for current label.
				printf("Could not find location predictions for label %d\n", label);
				continue;
			}
			const std::vector<ZQ_CNN_NormalizedBBox>& bboxes = decode_bboxes.find(label)->second;
			ZQ_CNN_BBoxUtils::ApplyNMSFast(bboxes, scores, confidence_thresh, nms_thresh, nms_eta, nms_top_k, &(indices[c]));
			num_det += indices[c].size();
		}
		if (keep_top_k > -1 && num_det > keep_top_k)
		{
			std::vector<std::pair<float, std::pair<int, int> > > score_index_pairs;
			for (std::map<int, std::vector<int> >::iterator it = indices.begin();
				it != indices.end(); ++it)
			{
				int label = it->first;
				const std::vector<int>& label_indices = it->second;
				if (conf_scores.find(label) == conf_scores.end())
				{
					// Something bad happened for current label.
					printf("Could not find location predictions for %d\n", label);
					continue;
				}
				const std::vector<float>& scores = conf_scores.find(label)->second;
				for (int j = 0; j < label_indices.size(); ++j)
				{
					int idx = label_indices[j];
					if (idx >= scores.size())
						return false;
					score_index_pairs.push_back(std::make_pair(
						scores[idx], std::make_pair(label, idx)));
				}
			}
			// Keep top k results per image.
			std::sort(score_index_pairs.begin(), score_index_pairs.end(),
				ZQ_CNN_BBoxUtils::SortScorePairDescend<std::pair<int, int> >);
			score_index_pairs.resize(keep_top_k);
			// Store the new indices.
			std::map<int, std::vector<int> > new_indices;
			for (int j = 0; j < score_index_pairs.size(); ++j)
			{
				int label = score_index_pairs[j].second.first;
				int idx = score_index_pairs[j].second.second;
				new_indices[label].push_back(idx);
			}
			all_indices.push_back(new_indices);
			num_kept += keep_top_k;
		}
		else {
			all_indices.push_back(indices);
			num_kept += num_det;
		}
	}

	std::vector<int> top_shape(2, 1);
	top_shape.push_back(num_kept);
	top_shape.push_back(7);
	float* out_ptr;
	if (num_kept == 0)
	{
		//printf("Couldn't find any detections\n");
		output.ChangeSize(num, 1, 1, 7, 0, 0);
		out_ptr = output.GetFirstPixelPtr();
		// Generate fake results per image.
		for (int i = 0; i < num; ++i)
		{
			out_ptr[0] = -1;
			out_ptr += output.GetSliceStep();
		}
	}
	else
	{
		output.ChangeSize(num_kept, 1, 1, 7, 0, 0);
	}

	out_ptr = output.GetFirstPixelPtr();
	int sliceStep = output.GetSliceStep();
	int count = 0;
	for (int i = 0; i < num; ++i)
	{
		const std::map<int, std::vector<float> >& conf_scores = all_conf_scores[i];
		const ZQ_CNN_LabelBBox& decode_bboxes = all_decode_bboxes[i];
		for (std::map<int, std::vector<int> >::iterator it = all_indices[i].begin();
			it != all_indices[i].end(); ++it)
		{
			int label = it->first;
			if (conf_scores.find(label) == conf_scores.end())
			{
				// Something bad happened if there are no predictions for current label.
				//LOG(FATAL) << "Could not find confidence predictions for " << label;
				continue;
			}
			const std::vector<float>& scores = conf_scores.find(label)->second;
			int loc_label = share_location ? -1 : label;
			if (decode_bboxes.find(loc_label) == decode_bboxes.end())
			{
				// Something bad happened if there are no predictions for current label.
				//LOG(FATAL) << "Could not find location predictions for " << loc_label;
				continue;
			}
			const std::vector<ZQ_CNN_NormalizedBBox>& bboxes = decode_bboxes.find(loc_label)->second;
			std::vector<int>& indices = it->second;

			for (int j = 0; j < indices.size(); ++j)
			{
				int idx = indices[j];
				out_ptr[count*sliceStep] = i;
				out_ptr[count * sliceStep + 1] = label;
				out_ptr[count * sliceStep + 2] = scores[idx];
				const ZQ_CNN_NormalizedBBox& bbox = bboxes[idx];
				out_ptr[count * sliceStep + 3] = bbox.col1;
				out_ptr[count * sliceStep + 4] = bbox.row1;
				out_ptr[count * sliceStep + 5] = bbox.col2;
				out_ptr[count * sliceStep + 6] = bbox.row2;
				++count;
			}
		}

	}
	return true;
}

bool ZQ_CNN_Forward_SSEUtils::_detection_output_MXNET(const ZQ_CNN_Tensor4D& loc, const ZQ_CNN_Tensor4D& conf,
	const ZQ_CNN_Tensor4D& prior, const std::vector<float>& variances, bool clip,
	float nms_thresh, int nms_top_k, float confidence_thresh, int keep_top_k,
	ZQ_CNN_Tensor4D& output)
{
	if (variances.size() != 4)
	{
		printf("Variance size must be 4\n");
		return false;
	}
		
	int num = loc.GetN();
	int loc_len = loc.GetN() * loc.GetH() * loc.GetW() * loc.GetC();
	int conf_len = conf.GetN() * conf.GetH() * conf.GetW() * conf.GetC();
	int prior_len = prior.GetN() * prior.GetH() * prior.GetW() * prior.GetC();
	if (loc_len <= 0 || conf_len <= 0 || prior_len <= 0)
		return false;
	std::vector<float> loc_data(loc_len);
	std::vector<float> conf_data(conf_len);
	std::vector<float> prior_data(prior_len);
	loc.ConvertToCompactNCHW(&loc_data[0]);
	conf.ConvertToCompactNCHW(&conf_data[0]);
	prior.ConvertToCompactNCHW(&prior_data[0]);

	int num_anchors = conf.GetH();
	int num_classes = conf.GetC();
	int conf_sliceStep = conf.GetSliceStep();
	int conf_pixStep = conf.GetPixelStep();
	int loc_sliceStep = loc.GetSliceStep();
	const float* p_anchor = prior.GetFirstPixelPtr();
	int prior_widthStep = prior.GetWidthStep();
	int prior_pixStep = prior.GetPixelStep();
	float tmp_buffer_bbox[4];
	std::vector<std::vector<float> > tmp_outs(num);
	int num_kept = 0;
	std::vector<ZQ_CNN_LabelBBox> all_bboxes(num);
	std::vector<std::map<int, std::vector<float> > > all_scores(num);
	std::vector<std::map<int, std::vector<int> > > all_indices(num);
	for (int n = 0; n < num; n++)
	{
		const float *p_cls_prob = conf.GetFirstPixelPtr() + n * conf_sliceStep;
		const float *p_loc_pred = loc.GetFirstPixelPtr() + n * loc_sliceStep;
		tmp_outs[n].resize(num_anchors * 6);
		float *p_out = &tmp_outs[n][0];
		ZQ_CNN_LabelBBox& bboxes = all_bboxes[n];
		std::map<int, std::vector<float> >& scores = all_scores[n];
		std::map<int, std::vector<int> > indices;
		int num_det = 0;

		for (int i = 0; i < num_anchors; i++) 
		{
			// find the predicted class id and probability
			float score = -1;
			int id = 0;
			for (int j = 1; j < num_classes; j++) 
			{
				float temp = p_cls_prob[i*conf_pixStep+j];
				if (temp > score) {
					score = temp;
					id = j;
				}
			}

			// [id, prob, xmin, ymin, xmax, ymax]
			
			int offset = i * 4;
			ZQ_CNN_BBoxUtils::TransformLocations_MXNET(tmp_buffer_bbox,
				&prior_data[0] + offset, &loc_data[0] + offset, clip,
				variances[0], variances[1], variances[2], variances[3]);

			if (id > 0 && score >= confidence_thresh) 
			{
				ZQ_CNN_NormalizedBBox cur_box;
				cur_box.label = id;
				cur_box.score = score;
				cur_box.col1 = tmp_buffer_bbox[0];
				cur_box.row1 = tmp_buffer_bbox[1];
				cur_box.col2 = tmp_buffer_bbox[2];
				cur_box.row2 = tmp_buffer_bbox[3];
				bboxes[id].push_back(cur_box);
				scores[id].push_back(score);
			}
		}

		for (int c = 1; c < num_classes; c++)
		{
			if (bboxes.find(c) != bboxes.end())
			{
				ZQ_CNN_BBoxUtils::ApplyNMSFast(bboxes[c], scores[c], confidence_thresh, nms_thresh, 1, nms_top_k, &(indices[c]));
				num_det += indices[c].size();
			}
		}

		if (keep_top_k > -1 && num_det > keep_top_k)
		{
			std::vector<std::pair<float, std::pair<int, int> > > score_index_pairs;
			for (std::map<int, std::vector<int> >::iterator it = indices.begin();
				it != indices.end(); ++it)
			{
				int label = it->first;
				const std::vector<int>& label_indices = it->second;
				const std::vector<float>& cur_scores = scores.find(label)->second;
				for (int j = 0; j < label_indices.size(); ++j)
				{
					int idx = label_indices[j];
					if (idx >= scores[label].size())
						return false;
					score_index_pairs.push_back(std::make_pair(
						cur_scores[idx], std::make_pair(label, idx)));
				}
			}
			// Keep top k results per image.
			std::sort(score_index_pairs.begin(), score_index_pairs.end(),
				ZQ_CNN_BBoxUtils::SortScorePairDescend<std::pair<int, int> >);
			score_index_pairs.resize(keep_top_k);
			// Store the new indices.
			std::map<int, std::vector<int> > new_indices;
			for (int j = 0; j < score_index_pairs.size(); ++j)
			{
				int label = score_index_pairs[j].second.first;
				int idx = score_index_pairs[j].second.second;
				new_indices[label].push_back(idx);
			}
			all_indices[n] = new_indices;
			num_kept += keep_top_k;
		}
		else 
		{
			all_indices[n] = indices;
			num_kept += num_det;
		}
	}

	std::vector<int> top_shape(2, 1);
	top_shape.push_back(num_kept);
	top_shape.push_back(7);
	float* out_ptr;
	if (num_kept == 0)
	{
		//printf("Couldn't find any detections\n");
		output.ChangeSize(num, 1, 1, 7, 0, 0);
		out_ptr = output.GetFirstPixelPtr();
		// Generate fake results per image.
		for (int i = 0; i < num; ++i)
		{
			out_ptr[0] = -1;
			out_ptr += output.GetSliceStep();
		}
	}
	else
	{
		output.ChangeSize(num_kept, 1, 1, 7, 0, 0);
	}

	out_ptr = output.GetFirstPixelPtr();
	int sliceStep = output.GetSliceStep();
	int count = 0;
	for (int i = 0; i < num; ++i)
	{
		const std::map<int, std::vector<float> >& conf_scores = all_scores[i];
		const ZQ_CNN_LabelBBox& decode_bboxes = all_bboxes[i];
		for (std::map<int, std::vector<int> >::iterator it = all_indices[i].begin();
			it != all_indices[i].end(); ++it)
		{
			int label = it->first;
			if (conf_scores.find(label) == conf_scores.end())
			{
				// Something bad happened if there are no predictions for current label.
				//LOG(FATAL) << "Could not find confidence predictions for " << label;
				continue;
			}
			const std::vector<float>& scores = conf_scores.find(label)->second;
			if (decode_bboxes.find(label) == decode_bboxes.end())
			{
				// Something bad happened if there are no predictions for current label.
				//LOG(FATAL) << "Could not find location predictions for " << loc_label;
				continue;
			}
			const std::vector<ZQ_CNN_NormalizedBBox>& bboxes = decode_bboxes.find(label)->second;
			std::vector<int>& indices = it->second;

			for (int j = 0; j < indices.size(); ++j)
			{
				int idx = indices[j];
				out_ptr[count*sliceStep] = i;
				out_ptr[count * sliceStep + 1] = label;
				out_ptr[count * sliceStep + 2] = scores[idx];
				const ZQ_CNN_NormalizedBBox& bbox = bboxes[idx];
				out_ptr[count * sliceStep + 3] = bbox.col1;
				out_ptr[count * sliceStep + 4] = bbox.row1;
				out_ptr[count * sliceStep + 5] = bbox.col2;
				out_ptr[count * sliceStep + 6] = bbox.row2;
				++count;
			}
		}

	}
	return true;
}