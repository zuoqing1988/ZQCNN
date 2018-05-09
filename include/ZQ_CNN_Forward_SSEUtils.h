#ifndef _ZQ_CNN_FORWARD_SSE_UTILS_H_
#define _ZQ_CNN_FORWARD_SSE_UTILS_H_
#pragma once
#include "ZQ_CNN_Defines.h"
#include "ZQ_CNN_Tensor4D.h"
#include <vector>
#include <algorithm>
#include <fstream>
#include <omp.h>

namespace ZQ
{
	class ZQ_CNN_Forward_SSEUtils
	{
	public:

		static bool ConvolutionWithBias(ZQ_CNN_Tensor4D& input, const ZQ_CNN_Tensor4D& filters, const ZQ_CNN_Tensor4D& bias,
			int strideH, int strideW, int padH, int padW, ZQ_CNN_Tensor4D& output)
		{
			double t1 = omp_get_wtime();
			int in_N = input.GetN();
			int in_H = input.GetH();
			int in_W = input.GetW();
			int in_C = input.GetC();
			int filter_N = filters.GetN();
			int filter_H = filters.GetH();
			int filter_W = filters.GetW();
			int filter_C = filters.GetC();
			int out_N = output.GetN();
			int out_H = output.GetH();
			int out_W = output.GetW();
			int out_C = output.GetC();
			float bias_C = bias.GetC();
			if (filter_C != in_C || filter_N != bias_C)
				return false;

			int need_N = in_N;
			int need_H = (in_H - filter_H + (padH << 1)) / strideH + 1;
			int need_W = (in_W - filter_W + (padW << 1)) / strideW + 1;
			int need_C = filter_N;
			if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
			{
				output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0);
			}

			if (padH != 0 || padW != 0)
			{
				if (!input.Padding(padW, padH, 0))
					return false;
			}

			int in_sliceStep = input.GetSliceStep();
			int in_widthStep = input.GetWidthStep();
			int in_pixStep = input.GetPixelStep();
			int filter_sliceStep = filters.GetSliceStep();
			int filter_widthStep = filters.GetWidthStep();
			int filter_pixStep = filters.GetPixelStep();
			int out_sliceStep = output.GetSliceStep();
			int out_widthStep = output.GetWidthStep();
			int out_pixStep = output.GetPixelStep();
			float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW*in_pixStep;
			const float* filter_firstPixelData = filters.GetFirstPixelPtr();
			float* out_firstPixelData = output.GetFirstPixelPtr();
			const float* bias_firstPixelData = bias.GetFirstPixelPtr();

			int align_mode = __min((int)input.GetAlignType(), __min((int)filters.GetAlignType(), (int)output.GetAlignType()));
			if (in_C == 1)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_0);
			else if (in_C <= 4)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_128bit);
			else if (in_C <= 8)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_256bit);
			//output.Reset();
			_convolution_nopadding(align_mode,  in_firstPixelData, in_N, in_H + (padH << 1), in_W + (padW << 1), in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_pixStep, out_widthStep, out_sliceStep);
			_addbias(__min(bias.GetAlignType(), output.GetAlignType()), out_firstPixelData, need_N, need_H, need_W, need_C, out_pixStep, out_widthStep, out_sliceStep, bias_firstPixelData);
			double t2 = omp_get_wtime();
			//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
			return true;
		}

		static bool Convolution(ZQ_CNN_Tensor4D& input, const ZQ_CNN_Tensor4D& filters, int strideH, int strideW, int padH, int padW,	ZQ_CNN_Tensor4D& output)
		{
			int in_N = input.GetN();
			int in_H = input.GetH();
			int in_W = input.GetW();
			int in_C = input.GetC();
			int filter_N = filters.GetN();
			int filter_H = filters.GetH();
			int filter_W = filters.GetW();
			int filter_C = filters.GetC();
			int out_N = output.GetN();
			int out_H = output.GetH();
			int out_W = output.GetW();
			int out_C = output.GetC();
			if (filter_C != in_C)
				return false;

			int need_N = in_N;
			int need_H = (in_H - filter_H + (padH << 1)) / strideH + 1;
			int need_W = (in_W - filter_W + (padW << 1)) / strideW + 1;
			int need_C = filter_N;
			if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
			{
				output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0);
			}

			if (padH != 0 || padW != 0)
			{
				if (!input.Padding(padW, padH, 0))
					return false;
			}
			
			int in_sliceStep = input.GetSliceStep();
			int in_widthStep = input.GetWidthStep();
			int in_pixStep = input.GetPixelStep();
			int filter_sliceStep = filters.GetSliceStep();
			int filter_widthStep = filters.GetWidthStep();
			int filter_pixStep = filters.GetPixelStep();
			int out_sliceStep = output.GetSliceStep();
			int out_widthStep = output.GetWidthStep();
			int out_pixStep = output.GetPixelStep();
			float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW*in_pixStep;
			const float* filter_firstPixelData = filters.GetFirstPixelPtr();
			float* out_firstPixelData = output.GetFirstPixelPtr();

			int align_mode = __min((int)input.GetAlignType(), __min((int)filters.GetAlignType(), (int)output.GetAlignType()));
			if (in_C == 1)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_0);
			else if (in_C <= 4)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_128bit);
			else if (in_C <= 8)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_256bit);

			//output.Reset();
			_convolution_nopadding(align_mode, in_firstPixelData, in_N, in_H + (padH << 1), in_W + (padW << 1), in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_pixStep, out_widthStep, out_sliceStep);
			
			return true;
		}

		static bool DepthwiseConvolutionWithBias(ZQ_CNN_Tensor4D& input, const ZQ_CNN_Tensor4D& filters, const ZQ_CNN_Tensor4D& bias,
			int strideH, int strideW, int padH, int padW, ZQ_CNN_Tensor4D& output)
		{
			double t1 = omp_get_wtime();
			int in_N = input.GetN();
			int in_H = input.GetH();
			int in_W = input.GetW();
			int in_C = input.GetC();
			int filter_N = filters.GetN();
			int filter_H = filters.GetH();
			int filter_W = filters.GetW();
			int filter_C = filters.GetC();
			int out_N = output.GetN();
			int out_H = output.GetH();
			int out_W = output.GetW();
			int out_C = output.GetC();
			float bias_C = bias.GetC();
			if (filter_C != in_C || filter_N != 1)
				return false;

			int need_N = in_N;
			int need_H = (in_H - filter_H + (padH << 1)) / strideH + 1;
			int need_W = (in_W - filter_W + (padW << 1)) / strideW + 1;
			int need_C = in_C;
			if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
			{
				output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0);
			}

			if (padH != 0 || padW != 0)
			{
				if (!input.Padding(padW, padH, 0))
					return false;
			}

			int in_sliceStep = input.GetSliceStep();
			int in_widthStep = input.GetWidthStep();
			int in_pixStep = input.GetPixelStep();
			int filter_sliceStep = filters.GetSliceStep();
			int filter_widthStep = filters.GetWidthStep();
			int filter_pixStep = filters.GetPixelStep();
			int out_sliceStep = output.GetSliceStep();
			int out_widthStep = output.GetWidthStep();
			int out_pixStep = output.GetPixelStep();
			float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW*in_pixStep;
			const float* filter_firstPixelData = filters.GetFirstPixelPtr();
			float* out_firstPixelData = output.GetFirstPixelPtr();
			const float* bias_firstPixelData = bias.GetFirstPixelPtr();

			int align_mode = __min((int)input.GetAlignType(), __min((int)filters.GetAlignType(), (int)output.GetAlignType()));
			if (in_C == 1)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_0);
			else if (in_C <= 4)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_128bit);
			else if (in_C <= 8)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_256bit);
			//output.Reset();
			_depthwise_convolution_nopadding(align_mode, in_firstPixelData, in_N, in_H + (padH << 1), in_W + (padW << 1), in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_pixStep, out_widthStep, out_sliceStep);
			_addbias(__min(bias.GetAlignType(), output.GetAlignType()), out_firstPixelData, need_N, need_H, need_W, need_C, out_pixStep, out_widthStep, out_sliceStep, bias_firstPixelData);
			double t2 = omp_get_wtime();
			//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
			return true;
		}

		static bool DepthwiseConvolution(ZQ_CNN_Tensor4D& input, const ZQ_CNN_Tensor4D& filters, int strideH, int strideW, int padH, int padW, ZQ_CNN_Tensor4D& output)
		{
			int in_N = input.GetN();
			int in_H = input.GetH();
			int in_W = input.GetW();
			int in_C = input.GetC();
			int filter_N = filters.GetN();
			int filter_H = filters.GetH();
			int filter_W = filters.GetW();
			int filter_C = filters.GetC();
			int out_N = output.GetN();
			int out_H = output.GetH();
			int out_W = output.GetW();
			int out_C = output.GetC();
			if (filter_C != in_C || filter_N != 1)
				return false;

			int need_N = in_N;
			int need_H = (in_H - filter_H + (padH << 1)) / strideH + 1;
			int need_W = (in_W - filter_W + (padW << 1)) / strideW + 1;
			int need_C = in_C;
			if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
			{
				output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0);
			}

			if (padH != 0 || padW != 0)
			{
				if (!input.Padding(padW, padH, 0))
					return false;
			}

			int in_sliceStep = input.GetSliceStep();
			int in_widthStep = input.GetWidthStep();
			int in_pixStep = input.GetPixelStep();
			int filter_sliceStep = filters.GetSliceStep();
			int filter_widthStep = filters.GetWidthStep();
			int filter_pixStep = filters.GetPixelStep();
			int out_sliceStep = output.GetSliceStep();
			int out_widthStep = output.GetWidthStep();
			int out_pixStep = output.GetPixelStep();
			float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW*in_pixStep;
			const float* filter_firstPixelData = filters.GetFirstPixelPtr();
			float* out_firstPixelData = output.GetFirstPixelPtr();

			int align_mode = __min((int)input.GetAlignType(), __min((int)filters.GetAlignType(), (int)output.GetAlignType()));
			if (in_C == 1)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_0);
			else if (in_C <= 4)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_128bit);
			else if (in_C <= 8)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_256bit);

			//output.Reset();
			_depthwise_convolution_nopadding(align_mode, in_firstPixelData, in_N, in_H + (padH << 1), in_W + (padW << 1), in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_pixStep, filter_widthStep, filter_sliceStep, strideH, strideW,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_pixStep, out_widthStep, out_sliceStep);

			return true;
		}

		static bool InnerProductWithBias(ZQ_CNN_Tensor4D& input, const ZQ_CNN_Tensor4D& filters, const ZQ_CNN_Tensor4D& bias, ZQ_CNN_Tensor4D& output)
		{
			double t1 = omp_get_wtime();
			int in_N = input.GetN();
			int in_H = input.GetH();
			int in_W = input.GetW();
			int in_C = input.GetC();
			int filter_N = filters.GetN();
			int filter_H = filters.GetH();
			int filter_W = filters.GetW();
			int filter_C = filters.GetC();
			int out_N = output.GetN();
			int out_H = output.GetH();
			int out_W = output.GetW();
			int out_C = output.GetC();
			float bias_C = bias.GetC();
			if (filter_H != in_H || filter_W != in_W || filter_C != in_C || filter_N != bias_C)
				return false;

			int need_N = in_N;
			int need_H = 1;
			int need_W = 1;
			int need_C = filter_N;
			if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
			{
				output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0);
			}

			int in_sliceStep = input.GetSliceStep();
			int in_widthStep = input.GetWidthStep();
			int in_pixStep = input.GetPixelStep();
			int filter_sliceStep = filters.GetSliceStep();
			int filter_widthStep = filters.GetWidthStep();
			int filter_pixStep = filters.GetPixelStep();
			int out_sliceStep = output.GetSliceStep();
			int out_widthStep = output.GetWidthStep();
			int out_pixStep = output.GetPixelStep();
			float* in_firstPixelData = input.GetFirstPixelPtr();
			const float* filter_firstPixelData = filters.GetFirstPixelPtr();
			float* out_firstPixelData = output.GetFirstPixelPtr();
			const float* bias_firstPixelData = bias.GetFirstPixelPtr();

			int align_mode = __min((int)input.GetAlignType(), __min((int)filters.GetAlignType(), (int)output.GetAlignType()));
			if (in_C == 1)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_0);
			else if (in_C <= 4)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_128bit);
			else if (in_C <= 8)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_256bit);
			_inner_product(align_mode, in_firstPixelData, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_firstPixelData, filter_N, filter_pixStep, filter_widthStep, filter_sliceStep, 
				out_firstPixelData, need_N, out_sliceStep);
			_addbias(output.GetAlignType(), output.GetFirstPixelPtr(), need_N, need_H, need_W, need_C, out_pixStep, out_widthStep, out_sliceStep, bias_firstPixelData);
			double t2 = omp_get_wtime();
			//printf("utils:inner: %.3f ms\n", (t2 - t1)*1000);
			return true;
		}

		static bool InnerProduct(ZQ_CNN_Tensor4D& input, const ZQ_CNN_Tensor4D& filters, ZQ_CNN_Tensor4D& output)
		{
			int in_N = input.GetN();
			int in_H = input.GetH();
			int in_W = input.GetW();
			int in_C = input.GetC();
			int filter_N = filters.GetN();
			int filter_H = filters.GetH();
			int filter_W = filters.GetW();
			int filter_C = filters.GetC();
			int out_N = output.GetN();
			int out_H = output.GetH();
			int out_W = output.GetW();
			int out_C = output.GetC();
			if (filter_H != in_H || filter_W != in_W || filter_C != in_C)
				return false;

			int need_N = in_N;
			int need_H = 1;
			int need_W = 1;
			int need_C = filter_N;
			if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
			{
				output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0);
			}

			int in_sliceStep = input.GetSliceStep();
			int in_widthStep = input.GetWidthStep();
			int in_pixStep = input.GetPixelStep();
			int filter_sliceStep = filters.GetSliceStep();
			int filter_widthStep = filters.GetWidthStep();
			int filter_pixStep = filters.GetPixelStep();
			int out_sliceStep = output.GetSliceStep();
			int out_widthStep = output.GetWidthStep();
			int out_pixStep = output.GetPixelStep();
			float* in_firstPixelData = input.GetFirstPixelPtr();
			const float* filter_firstPixelData = filters.GetFirstPixelPtr();
			float* out_firstPixelData = output.GetFirstPixelPtr();
			
			int align_mode = __min((int)input.GetAlignType(), __min((int)filters.GetAlignType(), (int)output.GetAlignType()));
			if (in_C == 1)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_0);
			else if (in_C <= 4)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_128bit);
			else if (in_C <= 8)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_256bit);
			_inner_product(align_mode, in_firstPixelData, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
				filter_firstPixelData, filter_N, filter_pixStep, filter_widthStep, filter_sliceStep,
				out_firstPixelData, need_N, out_sliceStep);

			return true;
		}

		


		static void MaxPooling(const ZQ_CNN_Tensor4D &input, ZQ_CNN_Tensor4D &output, int kernel_H, int kernel_W, int stride_H, int stride_W)
		{
			int in_N = input.GetN();
			int in_H = input.GetH();
			int in_W = input.GetW();
			int in_C = input.GetC();
			int need_W = ceil((float)(in_W - kernel_W) / stride_W + 1);
			int need_H = ceil((float)(in_H - kernel_H) / stride_H + 1);
			int need_N = in_N;
			int need_C = in_C;
			bool suredivided = (in_H - kernel_H) % stride_H == 0 && (in_W - kernel_W) % stride_W == 0;
			if (output.GetN() != need_N || output.GetH() != need_H || output.GetW() != need_W || output.GetC() != need_C)
				output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0);

			int in_sliceStep = input.GetSliceStep();
			int in_widthStep = input.GetWidthStep();
			int in_pixStep = input.GetPixelStep();
			int out_sliceStep = output.GetSliceStep();
			int out_widthStep = output.GetWidthStep();
			int out_pixStep = output.GetPixelStep();
			const float* in_data = input.GetFirstPixelPtr();
			float* out_data = output.GetFirstPixelPtr();

			int align_mode = __min(input.GetAlignType(), output.GetAlignType());
			_maxpooling(align_mode, in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep, kernel_H, kernel_W, stride_H, stride_W, 
				out_data, need_H, need_W, out_pixStep, out_widthStep, out_sliceStep);
			
		}

		static bool PReLU(ZQ_CNN_Tensor4D &input, const ZQ_CNN_Tensor4D& slope)
		{
			int N = input.GetN();
			int H = input.GetH();
			int W = input.GetW();
			int C = input.GetC();
			if (slope.GetC() != C)
				return false;
			float* data = input.GetFirstPixelPtr();
			const float* slope_Data = slope.GetFirstPixelPtr();
			int pixelStep = input.GetPixelStep();
			int widthStep = input.GetWidthStep();
			int sliceStep = input.GetSliceStep();

			int align_mode = __min(input.GetAlignType(), slope.GetAlignType());
			_prelu(align_mode, data, N, H, W, C, pixelStep, widthStep, sliceStep, slope_Data);

			return true;
		}

		static void ReLU(ZQ_CNN_Tensor4D &input)
		{
			int N = input.GetN();
			int H = input.GetH();
			int W = input.GetW();
			int C = input.GetC();
			float* data = input.GetFirstPixelPtr();
			int pixelStep = input.GetPixelStep();
			int widthStep = input.GetWidthStep();
			int sliceStep = input.GetSliceStep();

			int align_mode = input.GetAlignType();
			_relu(align_mode, data, N, H, W, C, pixelStep, widthStep, sliceStep);
		}

		static void Dropout(ZQ_CNN_Tensor4D &input, float dropout_ratio)
		{
			if (dropout_ratio == 0.0f)
			{
				return;
			}
			else
			{
				int N = input.GetN();
				int H = input.GetH();
				int W = input.GetW();
				int C = input.GetC();
				int sliceStep = input.GetSliceStep();
				int widthStep = input.GetWidthStep();
				int pixStep = input.GetPixelStep();
				float* data = input.GetFirstPixelPtr();

				int align_mode = input.GetAlignType();
				_dropout(align_mode, data, N, H, W, C, pixStep, widthStep, sliceStep, dropout_ratio);
			}
		}

		static void SoftmaxChannel(ZQ_CNN_Tensor4D &input)
		{
			int N = input.GetN();
			int H = input.GetH();
			int W = input.GetW();
			int C = input.GetC();
			
			float* data = input.GetFirstPixelPtr();
			int sliceStep = input.GetSliceStep();
			int widthStep = input.GetWidthStep();
			int pixStep = input.GetPixelStep();

			int align_mode = input.GetAlignType();
			/*if (C <= 8)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_0);
			else if (C <= 16)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_128bit);
			else if (C <= 32)
				align_mode = __min(align_mode, ZQ_CNN_Tensor4D::ALIGN_256bit);
			*/
			_softmax(align_mode, data, N, H, W, C, pixStep, widthStep, sliceStep);
			
		}

		static bool BatchNormScaleBias(ZQ_CNN_Tensor4D &input, const ZQ_CNN_Tensor4D& mean, const ZQ_CNN_Tensor4D& var, const ZQ_CNN_Tensor4D& slope, const ZQ_CNN_Tensor4D& bias)
		{
			int N = input.GetN();
			int H = input.GetH();
			int W = input.GetW();
			int C = input.GetC();
			if (mean.GetC() != C || var.GetC() != C || slope.GetC() != C || bias.GetC() != C)
				return false;
			float* data = input.GetFirstPixelPtr();
			const float* mean_data = mean.GetFirstPixelPtr();
			const float* var_data = var.GetFirstPixelPtr();
			const float* slope_data = slope.GetFirstPixelPtr();
			const float* bias_data = bias.GetFirstPixelPtr();
			int pixelStep = input.GetPixelStep();
			int widthStep = input.GetWidthStep();
			int sliceStep = input.GetSliceStep();

			int align_mode = __min(input.GetAlignType(), __min(mean.GetAlignType(), __min(var.GetAlignType(), __min(slope.GetAlignType(), bias.GetAlignType()))));
			_batchnorm_scalebias(align_mode, data, N, H, W, C, pixelStep, widthStep, sliceStep, mean_data, var_data, slope_data, bias_data);
			return true;
		}

		static bool BatchNorm(ZQ_CNN_Tensor4D &input, const ZQ_CNN_Tensor4D& mean, const ZQ_CNN_Tensor4D& var)
		{
			int N = input.GetN();
			int H = input.GetH();
			int W = input.GetW();
			int C = input.GetC();
			if (mean.GetC() != C || var.GetC() != C)
				return false;
			float* data = input.GetFirstPixelPtr();
			const float* mean_data = mean.GetFirstPixelPtr();
			const float* var_data = var.GetFirstPixelPtr();
			int pixelStep = input.GetPixelStep();
			int widthStep = input.GetWidthStep();
			int sliceStep = input.GetSliceStep();

			int align_mode = __min(input.GetAlignType(), __min(mean.GetAlignType(), var.GetAlignType()));
			_batchnorm(align_mode, data, N, H, W, C, pixelStep, widthStep, sliceStep, mean_data, var_data);
			return true;
		}


		/*
		a = bias - slope * mean / sqrt(var)
		b = slope / sqrt(var)
		value = b * value + a
		*/
		static bool BatchNormScaleBias_Compute_b_a(ZQ_CNN_Tensor4D& b, ZQ_CNN_Tensor4D& a, const ZQ_CNN_Tensor4D& mean, const ZQ_CNN_Tensor4D& var, const ZQ_CNN_Tensor4D& scale, const ZQ_CNN_Tensor4D& bias)
		{
			int C = b.GetC();
			if (C == 0)
				return false;
			
			
			if (a .GetC() != C || mean.GetC() != C || var.GetC() != C || scale.GetC() != C || bias.GetC() != C)
				return false;
			float* b_data = b.GetFirstPixelPtr();
			float* a_data = a.GetFirstPixelPtr();
			const float* mean_data = mean.GetFirstPixelPtr();
			const float* var_data = var.GetFirstPixelPtr();
			const float* scale_data = scale.GetFirstPixelPtr();
			const float* bias_data = bias.GetFirstPixelPtr();
			for (int c = 0; c < C; c++)
			{
				b_data[c] = scale_data[c] / __max(sqrt(__max(var_data[c],0)),1e-32);
				a_data[c] = bias_data[c] - mean_data[c] * b_data[c];
			}
			return true;
		}

		/*
		a = - slope * mean / sqrt(var)
		b = slope / sqrt(var)
		value = b * value + a
		*/
		static bool BatchNormScale_Compute_b_a(ZQ_CNN_Tensor4D& b, ZQ_CNN_Tensor4D& a, const ZQ_CNN_Tensor4D& mean, const ZQ_CNN_Tensor4D& var, const ZQ_CNN_Tensor4D& scale)
		{
			int C = b.GetC();
			if (C == 0)
				return false;


			if (a.GetC() != C || mean.GetC() != C || var.GetC() != C || scale.GetC() != C)
				return false;
			float* b_data = b.GetFirstPixelPtr();
			float* a_data = a.GetFirstPixelPtr();
			const float* mean_data = mean.GetFirstPixelPtr();
			const float* var_data = var.GetFirstPixelPtr();
			const float* scale_data = scale.GetFirstPixelPtr();
			for (int c = 0; c < C; c++)
			{
				b_data[c] = scale_data[c] / sqrt(var_data[c]);
				a_data[c] =  - mean_data[c] * b_data[c];
			}
			return true;
		}

		/*
		a = - slope * mean / sqrt(var)
		b = slope / sqrt(var)
		value = b * value + a
		*/
		static bool BatchNorm_Compute_b_a(ZQ_CNN_Tensor4D& b, ZQ_CNN_Tensor4D& a, const ZQ_CNN_Tensor4D& mean, const ZQ_CNN_Tensor4D& var)
		{
			int C = b.GetC();
			if (C == 0)
				return false;


			if (a.GetC() != C || mean.GetC() != C || var.GetC() != C)
				return false;
			float* b_data = b.GetFirstPixelPtr();
			float* a_data = a.GetFirstPixelPtr();
			const float* mean_data = mean.GetFirstPixelPtr();
			const float* var_data = var.GetFirstPixelPtr();
			for (int c = 0; c < C; c++)
			{
				b_data[c] = 1.0f / sqrt(var_data[c]);
				a_data[c] = -mean_data[c] * b_data[c];
			}
			return true;
		}

		/*
		a = bias - slope * mean / sqrt(var)
		b = slope / sqrt(var)
		value = b * value + a
		***OR***
		a = -mean/sqrt(var)
		b = 1/sqrt(var)
		value = b * value + a
		*/
		static bool BatchNorm_b_a(ZQ_CNN_Tensor4D &input, const ZQ_CNN_Tensor4D& b, const ZQ_CNN_Tensor4D& a)
		{
			int N = input.GetN();
			int H = input.GetH();
			int W = input.GetW();
			int C = input.GetC();
			if (b.GetC() != C || a.GetC() != C)
				return false;
			float* data = input.GetFirstPixelPtr();
			const float* b_data = b.GetFirstPixelPtr();
			const float* a_data = a.GetFirstPixelPtr();
			int pixelStep = input.GetPixelStep();
			int widthStep = input.GetWidthStep();
			int sliceStep = input.GetSliceStep();

			int align_mode = __min(input.GetAlignType(), __min(b.GetAlignType(),a.GetAlignType()));
			_batchnorm_b_a(align_mode, data, N, H, W, C, pixelStep, widthStep, sliceStep, b_data, a_data);
			return true;
		}

		static bool ScaleWithBias(ZQ_CNN_Tensor4D& input, const ZQ_CNN_Tensor4D& scale, const ZQ_CNN_Tensor4D& bias)
		{
			int N = input.GetN();
			int H = input.GetH();
			int W = input.GetW();
			int C = input.GetC();
			if (scale.GetC() != C || scale.GetC() != C)
				return false;
			float* data = input.GetFirstPixelPtr();
			const float* scale_data = scale.GetFirstPixelPtr();
			const float* bias_data = bias.GetFirstPixelPtr();
			int pixelStep = input.GetPixelStep();
			int widthStep = input.GetWidthStep();
			int sliceStep = input.GetSliceStep();

			int align_mode = __min(input.GetAlignType(), __min(scale.GetAlignType(), bias.GetAlignType()));
			_scalebias(align_mode, data, N, H, W, C, pixelStep, widthStep, sliceStep, scale_data, bias_data);
			return true;
		}

		static bool Scale(ZQ_CNN_Tensor4D& input, const ZQ_CNN_Tensor4D& scale)
		{
			int N = input.GetN();
			int H = input.GetH();
			int W = input.GetW();
			int C = input.GetC();
			if (scale.GetC() != C)
				return false;
			float* data = input.GetFirstPixelPtr();
			const float* scale_data = scale.GetFirstPixelPtr();
			int pixelStep = input.GetPixelStep();
			int widthStep = input.GetWidthStep();
			int sliceStep = input.GetSliceStep();

			int align_mode = __min(input.GetAlignType(), scale.GetAlignType());
			_scalebias(align_mode, data, N, H, W, C, pixelStep, widthStep, sliceStep, scale_data, NULL);
			return true;
		}

		static bool Eltwise_Sum(const std::vector<const ZQ_CNN_Tensor4D*>& input, ZQ_CNN_Tensor4D& output)
		{
			int in_num = input.size();
			if (in_num < 2)
				return false;
			for (int i = 0; i < in_num; i++)
			{
				if (input[i] == 0) return false;
			}
			int N = input[0]->GetN();
			int H = input[0]->GetH();
			int W = input[0]->GetW();
			int C = input[0]->GetC();
			for (int i = 1; i < in_num; i++)
			{
				if (N != input[i]->GetN() || H != input[i]->GetH() || W != input[i]->GetW() || C != input[i]->GetC())
					return false;
			}
			if (output.GetN() != N || output.GetH() != H || output.GetW() != W || output.GetC() != C)
				output.ChangeSize(N, H, W, C, 0, 0);
			std::vector<const float*> in_tensor_data(in_num);
			std::vector<int> in_pixStep(in_num), in_widthStep(in_num), in_sliceStep(in_num);
			float* out_data = output.GetFirstPixelPtr();
			for (int i = 0; i < in_num; i++)
			{
				in_tensor_data[i] = input[i]->GetFirstPixelPtr();
				in_pixStep[i] = input[i]->GetPixelStep();
				in_widthStep[i] = input[i]->GetWidthStep();
				in_sliceStep[i] = input[i]->GetSliceStep();
			}
			int out_pixStep = output.GetPixelStep(), out_widthStep = output.GetWidthStep(), out_sliceStep = output.GetSliceStep();
			int align_mode = output.GetAlignType();
			for (int i = 0; i < in_num; i++)
				align_mode = __min(align_mode, input[i]->GetAlignType());
			_eltwise_sum(align_mode, in_num, &in_tensor_data[0], N, H, W, C, &in_pixStep[0], &in_widthStep[0], &in_sliceStep[0], out_data, out_pixStep, out_widthStep, out_sliceStep);
			return true;
		}

		static bool Eltwise_SumWithWeight(const std::vector<const ZQ_CNN_Tensor4D*>& input, const std::vector<float>& weight, ZQ_CNN_Tensor4D& output)
		{
			int in_num = input.size();
			if (in_num < 2 || in_num != weight.size())
				return false;
			for (int i = 0; i < in_num; i++)
			{
				if (input[i] == 0) return false;
			}
			int N = input[0]->GetN();
			int H = input[0]->GetH();
			int W = input[0]->GetW();
			int C = input[0]->GetC();
			for (int i = 1; i < in_num; i++)
			{
				if (N != input[i]->GetN() || H != input[i]->GetH() || W != input[i]->GetW() || C != input[i]->GetC())
					return false;
			}
			if (output.GetN() != N || output.GetH() != H || output.GetW() != W || output.GetC() != C)
				output.ChangeSize(N, H, W, C, 0, 0);
			std::vector<const float*> in_tensor_data(in_num);
			std::vector<int> in_pixStep(in_num), in_widthStep(in_num), in_sliceStep(in_num);
			float* out_data = output.GetFirstPixelPtr();
			for (int i = 0; i < in_num; i++)
			{
				in_tensor_data[i] = input[i]->GetFirstPixelPtr();
				in_pixStep[i] = input[i]->GetPixelStep();
				in_widthStep[i] = input[i]->GetWidthStep();
				in_sliceStep[i] = input[i]->GetSliceStep();
			}
			int out_pixStep = output.GetPixelStep(), out_widthStep = output.GetWidthStep(), out_sliceStep = output.GetSliceStep();
			int align_mode = output.GetAlignType();
			for (int i = 0; i < in_num; i++)
				align_mode = __min(align_mode, input[i]->GetAlignType());
			_eltwise_sum_with_weight(align_mode, in_num, &in_tensor_data[0], &weight[0], N, H, W, C, &in_pixStep[0], &in_widthStep[0], &in_sliceStep[0], out_data, out_pixStep, out_widthStep, out_sliceStep);
			return true;
		}

		static bool Eltwise_Prod(const std::vector<const ZQ_CNN_Tensor4D*>& input, ZQ_CNN_Tensor4D& output)
		{
			int in_num = input.size();
			if (in_num < 2)
				return false;
			for (int i = 0; i < in_num; i++)
			{
				if (input[i] == 0) return false;
			}
			int N = input[0]->GetN();
			int H = input[0]->GetH();
			int W = input[0]->GetW();
			int C = input[0]->GetC();
			for (int i = 1; i < in_num; i++)
			{
				if (N != input[i]->GetN() || H != input[i]->GetH() || W != input[i]->GetW() || C != input[i]->GetC())
					return false;
			}
			if (output.GetN() != N || output.GetH() != H || output.GetW() != W || output.GetC() != C)
				output.ChangeSize(N, H, W, C, 0, 0);
			std::vector<const float*> in_tensor_data(in_num);
			std::vector<int> in_pixStep(in_num), in_widthStep(in_num), in_sliceStep(in_num);
			float* out_data = output.GetFirstPixelPtr();
			for (int i = 0; i < in_num; i++)
			{
				in_tensor_data[i] = input[i]->GetFirstPixelPtr();
				in_pixStep[i] = input[i]->GetPixelStep();
				in_widthStep[i] = input[i]->GetWidthStep();
				in_sliceStep[i] = input[i]->GetSliceStep();
			}
			int out_pixStep = output.GetPixelStep(), out_widthStep = output.GetWidthStep(), out_sliceStep = output.GetSliceStep();
			int align_mode = output.GetAlignType();
			for (int i = 0; i < in_num; i++)
				align_mode = __min(align_mode, input[i]->GetAlignType());
			_eltwise_prod(align_mode, in_num, &in_tensor_data[0], N, H, W, C, &in_pixStep[0], &in_widthStep[0], &in_sliceStep[0], out_data, out_pixStep, out_widthStep, out_sliceStep);
			return true;

		}

		static bool Eltwise_Max(const std::vector<const ZQ_CNN_Tensor4D*>& input, ZQ_CNN_Tensor4D& output)
		{
			int in_num = input.size();
			if (in_num < 2)
				return false;
			for (int i = 0; i < in_num; i++)
			{
				if (input[i] == 0) return false;
			}
			int N = input[0]->GetN();
			int H = input[0]->GetH();
			int W = input[0]->GetW();
			int C = input[0]->GetC();
			for (int i = 1; i < in_num; i++)
			{
				if (N != input[i]->GetN() || H != input[i]->GetH() || W != input[i]->GetW() || C != input[i]->GetC())
					return false;
			}
			if (output.GetN() != N || output.GetH() != H || output.GetW() != W || output.GetC() != C)
				output.ChangeSize(N, H, W, C, 0, 0);
			std::vector<const float*> in_tensor_data(in_num);
			std::vector<int> in_pixStep(in_num), in_widthStep(in_num), in_sliceStep(in_num);
			float* out_data = output.GetFirstPixelPtr();
			for (int i = 0; i < in_num; i++)
			{
				in_tensor_data[i] = input[i]->GetFirstPixelPtr();
				in_pixStep[i] = input[i]->GetPixelStep();
				in_widthStep[i] = input[i]->GetWidthStep();
				in_sliceStep[i] = input[i]->GetSliceStep();
			}
			int out_pixStep = output.GetPixelStep(), out_widthStep = output.GetWidthStep(), out_sliceStep = output.GetSliceStep();
			int align_mode = output.GetAlignType();
			for (int i = 0; i < in_num; i++)
				align_mode = __min(align_mode, input[i]->GetAlignType());
			_eltwise_max(align_mode, in_num, &in_tensor_data[0], N, H, W, C, &in_pixStep[0], &in_widthStep[0], &in_sliceStep[0], out_data, out_pixStep, out_widthStep, out_sliceStep);
			return true;
		}

		static bool LRN_across_channels(const ZQ_CNN_Tensor4D& input, int local_size, float alpha, float beta, float k, ZQ_CNN_Tensor4D& output)
		{
			if (local_size % 2 != 1)
				return false;
			int N = input.GetN();
			int H = input.GetH();
			int W = input.GetW();
			int C = input.GetC();
			if (output.GetN() != N || output.GetH() != H || output.GetW() != W || output.GetC() != C)
			{
				output.ChangeSize(N, H, W, C, 0, 0);
			}

			int in_pixStep = input.GetPixelStep();
			int in_widthStep = input.GetWidthStep();
			int in_sliceStep = input.GetSliceStep();
			int out_pixStep = output.GetPixelStep();
			int out_widthStep = output.GetWidthStep();
			int out_sliceStep = output.GetSliceStep();
			const float* in_data = input.GetFirstPixelPtr();
			float* out_data = output.GetFirstPixelPtr();
			int align_mode = __min(input.GetAlignType(), output.GetAlignType());
			_lrn_across_channels(align_mode, local_size, alpha, beta, k, in_data, N, H, W, C, in_pixStep, in_widthStep, in_sliceStep, out_data, out_pixStep, out_widthStep, out_sliceStep);
			return true;
		}
	private:
		static ZQ_CNN_EXPORT void _convolution_nopadding(int align_mode, const float* in_data, int in_N, int in_H, int in_W,
			int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
			const float* filter_data, int filter_N, int filter_H, int filter_W, int filter_C, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
			int strideH, int strideW, float* out_data, int out_N, int out_H, int out_W, int out_C, int out_pixStep, int out_widthStep, int out_sliceStep);

		static ZQ_CNN_EXPORT void _depthwise_convolution_nopadding(int align_mode, const float* in_data, int in_N, int in_H, int in_W,
			int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
			const float* filter_data, int filter_N, int filter_H, int filter_W, int filter_C, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
			int strideH, int strideW, float* out_data, int out_N, int out_H, int out_W, int out_C, int out_pixStep, int out_widthStep, int out_sliceStep);

		static ZQ_CNN_EXPORT void _inner_product(int align_mode, const float* in_data, int in_N, int in_H, int in_W, int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
			const float* filter_data, int filter_N, int filter_pixStep, int filter_widthStep, int filter_sliceStep,
			float* out_data, int out_N, int out_sliceStep);

		static ZQ_CNN_EXPORT void _addbias(int align_mode, float* data, int N, int H, int W, int C, int pixelStep, int widthStep, int sliceStep, const float* bias_Data);

		static ZQ_CNN_EXPORT void _softmax(int align_mode, float* in_data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep);

		static ZQ_CNN_EXPORT void _dropout(int align_mode, float* in_data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep, float ratio);

		static ZQ_CNN_EXPORT void _prelu(int align_mode, float* data, int N, int H, int W, int C, int pixelStep, int widthStep, int sliceStep, const float* slope_Data);

		static ZQ_CNN_EXPORT void _relu(int align_mode, float* data, int N, int H, int W, int C, int pixelStep, int widthStep, int sliceStep);

		static ZQ_CNN_EXPORT void _maxpooling(int align_mode, const float* in_data, int N, int in_H, int in_W, int C, int in_pixStep, int in_widthStep, int in_sliceStep,
			int kernel_H, int kernel_W, int stride_H, int stride_W,	float* out_data, int out_H, int out_W, int out_pixStep, int out_widthStep, int out_sliceStep);

		static ZQ_CNN_EXPORT void _batchnorm(int align_mode, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep, const float* mean, const float* var);

		static ZQ_CNN_EXPORT void _batchnorm_scalebias(int align_mode, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
			const float* mean, const float* var, const float* slope, const float* bias);

		/*
		a = bias - slope * mean / sqrt(var)
		b = slope / sqrt(var)
		value = b * value + a
		***OR***
		a = -mean/sqrt(var)
		b = 1/sqrt(var)
		value = b * value + a
		*/
		static ZQ_CNN_EXPORT void _batchnorm_b_a(int align_mode, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep, const float* b, const float* a);

		static ZQ_CNN_EXPORT void _scalebias(int align_mode, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep, const float* scale, const float* bias);

		static ZQ_CNN_EXPORT void _eltwise_sum(int align_mode, int in_tensor_num, const float** in_data, int N, int H, int W, int C, const int* pixStep, const int* widthStep, const int* sliceStep,
			float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep);

		static ZQ_CNN_EXPORT void _eltwise_sum_with_weight(int align_mode, int in_tensor_num, const float** in_data, const float* weight, int N, int H, int W, int C,
			const int* pixStep, const int* widthStep, const int* sliceStep,
			float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep);

		static ZQ_CNN_EXPORT void _eltwise_prod(int align_mode, int in_tensor_num, const float** in_data, int N, int H, int W, int C, const int* pixStep, const int* widthStep, const int* sliceStep,
			float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep);

		static ZQ_CNN_EXPORT void _eltwise_max(int align_mode, int in_tensor_num, const float** in_data, int N, int H, int W, int C, const int* pixStep, const int* widthStep, const int* sliceStep,
			float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep);

		static ZQ_CNN_EXPORT void _lrn_across_channels(int align_mode, int local_size, float alpha, float beta, float k, const float* in_data, int N, int H, int W, int C,
			int in_pixStep, int in_widthStep, int in_sliceStep, float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep);
	};

}

#endif
