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
#include "layers_c/zq_cnn_lrn_32f_align_c.h"
#include "ZQ_CNN_Forward_SSEUtils.h"

using namespace ZQ;

void _convolution_handle_special_channel_case_N_equal_one(bool& has_handled, int align_mode, const float* in_data, int in_N, int in_H, int in_W, int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
	const float* filter_data, int filter_N, int filter_H, int filter_W, int filter_C, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
	int strideH, int strideW, float* out_data, int out_N, int out_H, int out_W, int out_C, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	has_handled = false;
	if (out_N != 1)
		return;
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		if (filter_H == 3 && filter_W == 3)
		{
			if (in_C == 3)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel3x3_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
		}
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
		if (filter_H == 3 && filter_W == 3)
		{
			if (in_C == 3)
			{
				zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (in_C <= 8)
			{
				zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C8(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (in_C <= 16)
			{
				zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C16(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (in_C <= 24)
			{
				zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C24(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (in_C <= 32)
			{
				zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C32(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
		}
	}
}

void _convolution_handle_special_channel_case_N_largerthan_one(bool& has_handled, int align_mode, const float* in_data, int in_N, int in_H, int in_W, int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
	const float* filter_data, int filter_N, int filter_H, int filter_W, int filter_C, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
	int strideH, int strideW, float* out_data, int out_N, int out_H, int out_W, int out_C, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	has_handled = false;
	//if (out_N <= 1) //none is fast
		return;
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		if (filter_H == 3 && filter_W == 3)
		{
			if (in_C == 3)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel3x3_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
		}
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
		if (filter_H == 3 && filter_W == 3)
		{
			if (in_C == 3)
			{
				zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			/*else if (in_C <= 8)
			{
				zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C8(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}*/
			/*else if (in_C <= 16)
			{
				zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C16(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (in_C <= 24)
			{
				zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C24(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (in_C <= 32)
			{
				zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C32(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}*/
		}
	}
}

void _convolution_nopadding_case_N_equal_one(int align_mode, const float* in_data, int in_N, int in_H, int in_W, int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
	const float* filter_data, int filter_N, int filter_H, int filter_W, int filter_C, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
	int strideH, int strideW, float* out_data, int out_N, int out_H, int out_W, int out_C, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	bool has_handled = false;
	int out_HW = out_H*out_W;
	int out_NHW = out_HW*out_N;
	int filter_HWC = filter_H*filter_W*filter_C;
	int batch_need_size = out_NHW*filter_HWC + filter_N*filter_HWC;
	
	_convolution_handle_special_channel_case_N_equal_one(has_handled, align_mode, in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
		filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
		out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);

	//gemm method
	if (!has_handled)
	{
		if (in_pixStep == filter_pixStep)
		{
			if (out_HW >= 16 && filter_HWC >= 32 && filter_N >= 4)
			{
				if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
				{
					zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					has_handled = true;
				}
				else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
				{
					zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					has_handled = true;
				}
			}
		}
		else
		{
			if (out_HW >= 16 && filter_HWC >= 32 && filter_N >= 4)
			{
				if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
				{
					zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					has_handled = true;
				}
				else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
				{
					zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					has_handled = true;
				}
				else
				{
					zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					has_handled = true;
				}
			}
		}
	}
	

	//backup method
	if (!has_handled)
	{
		if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
		{
			if (filter_H == 1 && filter_W == 1)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel1x1(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else if (filter_H == 2 && filter_W == 2 /*&& 0*//*seems slow*/)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel2x2(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else
				if (filter_H == 3 && filter_W == 3)
				{
					if (filter_C == 3)
					{
						zq_cnn_conv_no_padding_32f_align128bit_kernel3x3_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					}
					else
					{
						zq_cnn_conv_no_padding_32f_align128bit_kernel3x3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					}

				}
				else if (filter_H == 5 && filter_W == 5)
				{
					zq_cnn_conv_no_padding_32f_align128bit_kernel5x5(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}
				else
				{
					zq_cnn_conv_no_padding_32f_align128bit_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}

		}
		else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
		{
			//if (filter_H == 1 && filter_W == 1)
			//{
			//	zq_cnn_conv_no_padding_32f_align256bit_kernel1x1(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
			//		filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
			//		out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
			//}
			//else if (filter_H == 2 && filter_W == 2 /*&& 0*//*seems slow*/)
			//{
			//	zq_cnn_conv_no_padding_32f_align256bit_kernel2x2(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
			//		filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
			//		out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
			//}
			//else 
			if (filter_H == 3 && filter_W == 3)
			{
				if (filter_C == 3)
				{
					zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}
				else if (filter_C <= 8)
				{
					zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C8(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}
				else if (filter_C <= 16)
				{
					zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C16(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}
				else if (filter_C <= 24)
				{
					zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C24(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}
				else if (filter_C <= 32)
				{
					zq_cnn_conv_no_padding_32f_align256bit_kernel3x3_C32(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}
				else
				{
					zq_cnn_conv_no_padding_32f_align256bit_kernel3x3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}

			}
			/*else if (filter_H == 5 && filter_W == 5)
			{
			zq_cnn_conv_no_padding_32f_align256bit_kernel5x5(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
			filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
			out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
			}*/
			else
			{
				zq_cnn_conv_no_padding_32f_align256bit_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
		}
		else
		{
			zq_cnn_conv_no_padding_32f_align0_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
				out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
		}
	}
}


void _convolution_nopadding_case_N_largerthan_one(int align_mode, const float* in_data, int in_N, int in_H, int in_W, int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
	const float* filter_data, int filter_N, int filter_H, int filter_W, int filter_C, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
	int strideH, int strideW, float* out_data, int out_N, int out_H, int out_W, int out_C, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	const static int batch_limited_size = 100 * 1024 * 1024;

	bool has_handled = false;
	int out_HW = out_H*out_W;
	int out_NHW = out_HW*out_N;
	int filter_HWC = filter_H*filter_W*filter_C;
	int batch_need_size = out_NHW*filter_HWC + filter_N*filter_HWC;

	/*_convolution_handle_special_channel_case_N_largerthan_one(has_handled, align_mode, in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
		filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
		out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);*/

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
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					has_handled = true;
				}
				else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
				{
					zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					has_handled = true;
				}
			}
			else if (out_NHW >= 8)
			{
				if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
				{
					zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_batch(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					has_handled = true;
				}
				else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
				{
					zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep_batch(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					has_handled = true;
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
							out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
						has_handled = true;
					}
					else
					{
						zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
						has_handled = true;
					}
					
				}
				else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
				{
					if (in_C <= 8)
					{
						zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep_C3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
						has_handled = true;
					}
					else
					{
						zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
							filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
							out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
						has_handled = true;
					}		
				}
				else
				{
					zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					has_handled = true;
				}
			}
			else if (out_NHW >= 8 && out_N >= 16)
			{
				if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
				{
					zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_batch(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					has_handled = true;
				}
				else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
				{
					zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep_batch(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					has_handled = true;
				}
				else
				{
					zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep_batch(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
					has_handled = true;
				}
			}
		}
	}
	
	//backup method
	if (!has_handled)
	{
		if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
		{
			if (filter_H == 1 && filter_W == 1)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel1x1(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else if (filter_H == 2 && filter_W == 2 /*&& 0*//*seems slow*/)
			{
				zq_cnn_conv_no_padding_32f_align128bit_kernel2x2(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
			else
				if (filter_H == 3 && filter_W == 3)
				{
					zq_cnn_conv_no_padding_32f_align128bit_kernel3x3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}
				else if (filter_H == 5 && filter_W == 5)
				{
					zq_cnn_conv_no_padding_32f_align128bit_kernel5x5(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}
				else
				{
					zq_cnn_conv_no_padding_32f_align128bit_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
						filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
						out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				}

		}
		else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
		{
			//if (filter_H == 1 && filter_W == 1)
			//{
			//	zq_cnn_conv_no_padding_32f_align256bit_kernel1x1(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
			//		filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
			//		out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
			//}
			//else if (filter_H == 2 && filter_W == 2 /*&& 0*//*seems slow*/)
			//{
			//	zq_cnn_conv_no_padding_32f_align256bit_kernel2x2(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
			//		filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
			//		out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
			//}
			//else 
			if (filter_H == 3 && filter_W == 3)
			{
				zq_cnn_conv_no_padding_32f_align256bit_kernel3x3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
			/*else if (filter_H == 5 && filter_W == 5)
			{
			zq_cnn_conv_no_padding_32f_align256bit_kernel5x5(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
			filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
			out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep, bias);
			}*/
			else
			{
				zq_cnn_conv_no_padding_32f_align256bit_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			}
		}
		else
		{
			zq_cnn_conv_no_padding_32f_align0_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
				out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
		}
	}
}

void ZQ_CNN_Forward_SSEUtils::_convolution_nopadding(int align_mode, const float* in_data, int in_N, int in_H, int in_W, int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
	const float* filter_data, int filter_N, int filter_H, int filter_W, int filter_C, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
	int strideH, int strideW, float* out_data, int out_N, int out_H, int out_W, int out_C, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (out_N > 1)
	{
		_convolution_nopadding_case_N_largerthan_one(align_mode, in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
			filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
			out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
	}
	else
	{
		_convolution_nopadding_case_N_equal_one(align_mode, in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
			filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
			out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
	}
}

void ZQ_CNN_Forward_SSEUtils::_depthwise_convolution_nopadding(int align_mode, const float* in_data, int in_N, int in_H, int in_W, int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
	const float* filter_data, int filter_N, int filter_H, int filter_W, int filter_C, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
	int strideH, int strideW, float* out_data, int out_N, int out_H, int out_W, int out_C, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	bool has_handled = false;
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		if (filter_H == 3 && filter_W == 3)
		{
			int padded_C = (in_C + 3) >> 2 << 2;
			if (padded_C == 4)
			{
				zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C4(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (padded_C == 8)
			{
				zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C8(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (padded_C == 16)
			{
				zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C16(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (padded_C == 32)
			{
				zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C32(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (padded_C == 64)
			{
				zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C64(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (padded_C == 128)
			{
				zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C128(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (padded_C == 256)
			{
				zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C256(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else
			{
				zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_32f_align128bit_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
				out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			has_handled = true;
		}
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
		if (filter_H == 3 && filter_W == 3)
		{
			int padded_C = (in_C + 7) >> 3 << 3;
			if (padded_C == 8)
			{
				zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C8(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (padded_C == 16)
			{
				zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C16(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (padded_C == 32)
			{
				zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C32(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (padded_C == 64)
			{
				zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C64(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (padded_C == 128)
			{
				zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C128(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (padded_C == 256)
			{
				zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C256(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else if (padded_C == 512)
			{
				zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C512(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
			else
			{
				zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
					filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
					out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
				has_handled = true;
			}
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_32f_align256bit_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
				out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
			has_handled = true;
		}
	}

	if (!has_handled)
	{
		zq_cnn_depthwise_conv_no_padding_32f_align0_general(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
			filter_data, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
			out_data, out_N, out_H, out_W, out_C, out_pixStep, out_widthStep, out_sliceStep);
		has_handled = true;
	}
}

void ZQ_CNN_Forward_SSEUtils::_inner_product(int align_mode, const float* in_data, int in_N, int in_H, int in_W, 
	int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
	const float* filter_data, int filter_N, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
	float* out_data, int out_N, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		if (out_N >= 16 && filter_N >= 16 && in_pixStep == filter_pixStep)
		{
			zq_cnn_innerproduct_gemm_32f_align128bit_same_pixstep_batch(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_pixStep, filter_widthStep, filter_sliceStep, out_data, out_N, filter_N, out_sliceStep, out_sliceStep, out_sliceStep);
		}
		else
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

		if (out_N >= 16 && filter_N >= 16 && in_pixStep == filter_pixStep)
		{
			zq_cnn_innerproduct_gemm_32f_align256bit_same_pixstep_batch(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_data, filter_N, filter_pixStep, filter_widthStep, filter_sliceStep,out_data, out_N, filter_N, out_sliceStep, out_sliceStep, out_sliceStep);
		}
		else
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

void  ZQ_CNN_Forward_SSEUtils::_addbias(int align_mode, float* data, int N, int H, int W, int C, int pixelStep, int widthStep, int sliceStep, const float* bias_Data)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_addbias_32f_align128bit(data, N, H, W, C, pixelStep, widthStep, sliceStep,bias_Data);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
		zq_cnn_addbias_32f_align256bit(data, N, H, W, C, pixelStep, widthStep, sliceStep, bias_Data);
	}
	else
	{
		zq_cnn_addbias_32f_align0(data, N, H, W, C, pixelStep, widthStep, sliceStep, bias_Data);
	}
}

void ZQ_CNN_Forward_SSEUtils::_softmax(int align_mode, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit && C >= 4)
	{
		zq_cnn_softmax_32f_align128bit_C(data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit && C >= 8)
	{
		zq_cnn_softmax_32f_align256bit_C(data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
	else
	{
		zq_cnn_softmax_32f_align0_C(data, N, H, W, C, pixStep, widthStep, sliceStep);
	}
}


void ZQ_CNN_Forward_SSEUtils::_dropout(int align_mode, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep, float ratio)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_dropout_32f_align128bit(data, N, H, W, C, pixStep, widthStep, sliceStep, ratio);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
		zq_cnn_dropout_32f_align256bit(data, N, H, W, C, pixStep, widthStep, sliceStep, ratio);
	}
	else
	{
		zq_cnn_dropout_32f_align0(data, N, H, W, C, pixStep, widthStep, sliceStep, ratio);
	}
}

void ZQ_CNN_Forward_SSEUtils::_prelu(int align_mode, float* data, int N, int H, int W, int C, int pixelStep, int widthStep, int sliceStep, const float* slope_Data)
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
			zq_cnn_prelu_32f_align128bit_sure_slope_lessthan1(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope_Data);
		else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
		{
			zq_cnn_prelu_32f_align256bit_sure_slope_lessthan1(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope_Data);
		}
		else
		{
			zq_cnn_prelu_32f_align0(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope_Data);
		}
	}
	else
	{
		if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
			zq_cnn_prelu_32f_align128bit(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope_Data);
		else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
		{
			zq_cnn_prelu_32f_align256bit(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope_Data);
		}
		else
		{
			zq_cnn_prelu_32f_align0(data, N, H, W, C, pixelStep, widthStep, sliceStep, slope_Data);
		}
	}
}

void ZQ_CNN_Forward_SSEUtils::_relu(int align_mode, float* data, int N, int H, int W, int C, int pixelStep, int widthStep, int sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
		zq_cnn_relu_32f_align128bit(data, N, H, W, C, pixelStep, widthStep, sliceStep);
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
		zq_cnn_relu_32f_align256bit(data, N, H, W, C, pixelStep, widthStep, sliceStep);
	}
	else
	{
		zq_cnn_relu_32f_align0(data, N, H, W, C, pixelStep, widthStep, sliceStep);
	}
}


void ZQ_CNN_Forward_SSEUtils::_maxpooling(int align_mode, const float* in_data, int N, int in_H, int in_W, int C, int in_pixStep, int in_widthStep, int in_sliceStep,
	int kernel_H, int kernel_W, int stride_H, int stride_W, float* out_data, int out_H, int out_W, int out_pixStep, int out_widthStep, int out_sliceStep)
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
	}
	else
	{
		zq_cnn_maxpooling_nopadding_32f_align0_general(in_data, N, in_H, in_W, C, in_pixStep, in_widthStep, in_sliceStep,
			kernel_H, kernel_W, stride_H, stride_W,
			out_data, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);
	}
}



void ZQ_CNN_Forward_SSEUtils::_batchnorm(int align_mode, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	const float* mean, const float* var)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_batchnorm_32f_mean_var_align128bit(data, N, H, W, C, pixStep, widthStep, sliceStep, mean, var);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
		zq_cnn_batchnorm_32f_mean_var_align256bit(data, N, H, W, C, pixStep, widthStep, sliceStep, mean, var);
	}
	else
	{
		zq_cnn_batchnorm_32f_mean_var_align0(data, N, H, W, C, pixStep, widthStep, sliceStep, mean, var);
	}
}

void ZQ_CNN_Forward_SSEUtils::_batchnorm_scalebias(int align_mode, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	const float* mean, const float* var, const float* scale, const float* bias)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_batchnormscale_32f_mean_var_scale_bias_align128bit(data, N, H, W, C, pixStep, widthStep, sliceStep, mean, var, scale,bias);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
		zq_cnn_batchnormscale_32f_mean_var_scale_bias_align256bit(data, N, H, W, C, pixStep, widthStep, sliceStep, mean, var, scale, bias);
	}
	else
	{
		zq_cnn_batchnormscale_32f_mean_var_scale_bias_align0(data, N, H, W, C, pixStep, widthStep, sliceStep, mean, var, scale, bias);
	}
}


void ZQ_CNN_Forward_SSEUtils::_batchnorm_b_a(int align_mode, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	const float* b, const float* a)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_batchnorm_32f_b_a_align128bit(data, N, H, W, C, pixStep, widthStep, sliceStep, b,a);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
		zq_cnn_batchnorm_32f_b_a_align256bit(data, N, H, W, C, pixStep, widthStep, sliceStep, b,a);
	}
	else
	{
		zq_cnn_batchnorm_32f_b_a_align0(data, N, H, W, C, pixStep, widthStep, sliceStep, b,a);
	}
}

void ZQ_CNN_Forward_SSEUtils::_scalebias(int align_mode, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
	const float* scale, const float* bias)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_scale_32f_align128bit(data, N, H, W, C, pixStep, widthStep, sliceStep, scale, bias);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
		zq_cnn_scale_32f_align256bit(data, N, H, W, C, pixStep, widthStep, sliceStep, scale, bias);
	}
	else
	{
		zq_cnn_scale_32f_align0(data, N, H, W, C, pixStep, widthStep, sliceStep, scale, bias);
	}
}

void ZQ_CNN_Forward_SSEUtils::_eltwise_sum(int align_mode, int in_tensor_num, const float** in_data, int N, int H, int W, int C, const int* pixStep, const int* widthStep, const int* sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_eltwise_sum_32f_align128bit(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep,out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
		zq_cnn_eltwise_sum_32f_align256bit(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else
	{
		zq_cnn_eltwise_sum_32f_align0(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

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
		zq_cnn_eltwise_sum_with_weight_32f_align256bit(in_tensor_num, in_data, weight, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else
	{
		zq_cnn_eltwise_sum_with_weight_32f_align0(in_tensor_num, in_data, weight, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

void ZQ_CNN_Forward_SSEUtils::_eltwise_prod(int align_mode, int in_tensor_num, const float** in_data, int N, int H, int W, int C, const int* pixStep, const int* widthStep, const int* sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_eltwise_prod_32f_align128bit(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
		zq_cnn_eltwise_prod_32f_align256bit(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else
	{
		zq_cnn_eltwise_prod_32f_align0(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

void ZQ_CNN_Forward_SSEUtils::_eltwise_max(int align_mode, int in_tensor_num, const float** in_data, int N, int H, int W, int C, const int* pixStep, const int* widthStep, const int* sliceStep,
	float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_eltwise_max_32f_align128bit(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
		zq_cnn_eltwise_max_32f_align256bit(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else
	{
		zq_cnn_eltwise_max_32f_align0(in_tensor_num, in_data, N, H, W, C, pixStep, widthStep, sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}

void ZQ_CNN_Forward_SSEUtils::_lrn_across_channels(int align_mode, int local_size, float alpha, float beta, float k, const float* in_data, int N, int H, int W, int C,
	int in_pixStep, int in_widthStep, int in_sliceStep, float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{
	if (align_mode == ZQ_CNN_Tensor4D::ALIGN_128bit)
	{
		zq_cnn_lrn_across_channels_32f_align128bit(local_size, alpha, beta, k, in_data, N, H, W, C, in_pixStep, in_widthStep, in_sliceStep, 
			out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else if (align_mode == ZQ_CNN_Tensor4D::ALIGN_256bit)
	{
		zq_cnn_lrn_across_channels_32f_align256bit(local_size, alpha, beta, k, in_data, N, H, W, C, in_pixStep, in_widthStep, in_sliceStep,
			out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
	else
	{
		zq_cnn_lrn_across_channels_32f_align0(local_size, alpha, beta, k, in_data, N, H, W, C, in_pixStep, in_widthStep, in_sliceStep,
			out_data, out_pixStep, out_widthStep, out_sliceStep);
	}
}