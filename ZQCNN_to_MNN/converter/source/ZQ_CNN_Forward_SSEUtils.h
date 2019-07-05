#ifndef _ZQ_CNN_FORWARD_SSE_UTILS_H_
#define _ZQ_CNN_FORWARD_SSE_UTILS_H_
#pragma once
#include "ZQ_CNN_Tensor4D.h"
#include "ZQ_CNN_CompileConfig.h"
#include <vector>
#include <algorithm>
#include <fstream>
#include <math.h>
#include <omp.h>

namespace ZQ
{
	class ZQ_CNN_Forward_SSEUtils
	{
	public:

		/*
		a = bias - slope * mean / sqrt(var+eps)
		b = slope / sqrt(var+eps)
		value = b * value + a
		*/
		static bool BatchNormScaleBias_Compute_b_a(ZQ_CNN_Tensor4D& b, ZQ_CNN_Tensor4D& a, const ZQ_CNN_Tensor4D& mean, const ZQ_CNN_Tensor4D& var, const ZQ_CNN_Tensor4D& scale, const ZQ_CNN_Tensor4D& bias, const float eps)
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
				b_data[c] = scale_data[c] / sqrt(__max(var_data[c]+eps,1e-32));
				a_data[c] = bias_data[c] - mean_data[c] * b_data[c];
			}
			return true;
		}

		/*
		a = - slope * mean / sqrt(var+eps)
		b = slope / sqrt(var+eps)
		value = b * value + a
		*/
		static bool BatchNormScale_Compute_b_a(ZQ_CNN_Tensor4D& b, ZQ_CNN_Tensor4D& a, const ZQ_CNN_Tensor4D& mean, const ZQ_CNN_Tensor4D& var, 
			const ZQ_CNN_Tensor4D& scale, const float eps)
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
				b_data[c] = scale_data[c] / sqrt(__max(var_data[c]+eps, 1e-32));
				a_data[c] =  - mean_data[c] * b_data[c];
			}
			return true;
		}

		/*
		a = - slope * mean / sqrt(var+eps)
		b = slope / sqrt(var+eps)
		value = b * value + a
		*/
		static bool BatchNorm_Compute_b_a(ZQ_CNN_Tensor4D& b, ZQ_CNN_Tensor4D& a, 
			const ZQ_CNN_Tensor4D& mean, const ZQ_CNN_Tensor4D& var, const float eps)
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
				b_data[c] = 1.0f / sqrt(__max(var_data[c]+eps,1e-32));
				a_data[c] = -mean_data[c] * b_data[c];
			}
			return true;
		}

	

		static bool Concat_NCHW_get_size(const std::vector<ZQ_CNN_Tensor4D*>& inputs, int axis, int& out_N, int& out_C, int& out_H, int& out_W)
		{
			return _concat_NCHW_get_size(inputs, axis, out_N, out_C, out_H, out_W);
		}


	private:

		static bool _concat_NCHW_get_size(const std::vector<ZQ_CNN_Tensor4D*>& inputs, int axis, int& out_N, int& out_C, int& out_H, int& out_W)
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
	};

}

#endif
