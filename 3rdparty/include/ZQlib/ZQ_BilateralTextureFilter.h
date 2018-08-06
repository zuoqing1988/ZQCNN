#ifndef _ZQ_BILATERAL_TEXTURE_FILTER_H_
#define _ZQ_BILATERAL_TEXTURE_FILTER_H_
#pragma once

#include "ZQ_BilateralTextureFilterOptions.h"
#include "ZQ_DoubleImage.h"

namespace ZQ
{
	class ZQ_BilateralTextureFilter
	{
	public:
		/* refer to the paper */
		template<class T>
		static bool BilateralTextureFilter(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_BilateralTextureFilterOptions& opt);

		/* refer to the paper */
		template<class T>
		static bool BilateralTextureFilter(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, int patch_size, int nIter)
		{
			if (patch_size % 2 != 1)
				return false;
			ZQ_BilateralTextureFilterOptions opt;
			opt.sigma_for_alpha = 5 * patch_size;
			opt.half_patch_size = patch_size / 2;
			opt.fsize = (patch_size * 2 - 1) / 2;
			opt.sigma_for_space = patch_size - 1;
			opt.sigma_for_value = 0.05;
			
			BilateralTextureFilter(input, output, opt);

			if (nIter > 1)
			{
				
				for (int i = 1; i < nIter; i++)
				{
					ZQ_DImage<T> tmp = output;
					BilateralTextureFilter(tmp, output, opt);
				}
			}
			return true;
		}
	};

	/****************************/

	template<class T>
	bool ZQ_BilateralTextureFilter::BilateralTextureFilter(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_BilateralTextureFilterOptions& opt)
	{
		int width = input.width();
		int height = input.height();
		int nChannels = input.nchannels();
		int nPixels = width*height;
		int nElements = nPixels*nChannels;
		const T*& input_data = input.data();
		output = input;
		T*& output_data = output.data();

		int half_patch_size = opt.half_patch_size;
		int patch_size = 2 * half_patch_size + 1;
		int paddedWidth = width + patch_size;
		int paddedHeight = height + patch_size;
		
		int fsize = opt.fsize;
		float sigma_s = opt.sigma_for_space;
		float sigma_v = opt.sigma_for_value;

		if (fsize < 1 || sigma_s <= 0 || sigma_v <= 0)
			return false;

		ZQ_DImage<double> gauss_weight(2 * fsize + 1, 2 * fsize + 1);
		double*& gauss_weight_data = gauss_weight.data();

		float sigma_s2 = sigma_s*sigma_s;
		float sigma_v2 = sigma_v*sigma_v;
		int fsize_len = fsize * 2 + 1;

		for (int hh = -fsize; hh <= fsize; hh++)
		{
			for (int ww = -fsize; ww <= fsize; ww++)
			{
				double dis2 = (double)hh*hh + ww*ww;
				gauss_weight_data[(hh + fsize)*fsize_len + ww + fsize] = exp(-0.5*dis2 / sigma_s2);
			}
		}

		for (int c = 0; c < nChannels; c++)
		{
			ZQ_DImage<T> I(width, height);
			T*& I_data = I.data();
			for (int i = 0; i < nPixels; i++)
				I_data[i] = input_data[i*nChannels + c];

			/* Compute B (uniform blurring) begin */
			ZQ_DImage<T> sumI_per_row(paddedWidth, height);
			ZQ_DImage<T> sumI_per_col(paddedWidth, paddedHeight);
			T*& sumI_per_row_data = sumI_per_row.data();
			T*& sumI_per_col_data = sumI_per_col.data();
			for (int h = 0; h < height; h++)
			{
				T* row_data = sumI_per_row_data + h*paddedWidth;
				row_data[0] = 0;
				for (int w = 1; w <= half_patch_size; w++)
					row_data[w] = row_data[w - 1] + I_data[h*width + 0];
				for (int w = half_patch_size + 1; w < half_patch_size + 1 + width; w++)
					row_data[w] = row_data[w - 1] + I_data[h*width + w - half_patch_size - 1];
				for (int w = half_patch_size + 1 + width; w < patch_size + width; w++)
					row_data[w] = row_data[w - 1] + I_data[h*width + width - 1];
			}
			for (int w = 0; w < paddedWidth; w++)
			{
				T* col_data = sumI_per_col_data + w;
				col_data[0] = 0;
				for (int h = 1; h <= half_patch_size; h++)
					col_data[h*paddedWidth] = col_data[(h - 1)*paddedWidth] + sumI_per_col_data[w];
				for (int h = half_patch_size + 1; h < half_patch_size + 1 + height; h++)
					col_data[h*paddedWidth] = col_data[(h - 1)*paddedWidth] + sumI_per_row_data[(h - half_patch_size - 1)*paddedWidth + w];
				for (int h = half_patch_size + 1 + height; h < patch_size + height; h++)
					col_data[h*paddedWidth] = col_data[(h - 1)*paddedWidth] + sumI_per_row_data[(height - 1)*paddedWidth + w];
			}


			ZQ_DImage<T> B(width, height);
			T*& B_data = B.data();
			double scale = 1.0 / (patch_size*patch_size);
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					B_data[h*width + w] = scale*(sumI_per_col_data[(h + patch_size)*paddedWidth + w + patch_size]
						+ sumI_per_col_data[h*paddedWidth + w]
						- sumI_per_col_data[(h + patch_size)*paddedWidth + w]
						- sumI_per_col_data[h*paddedWidth + w + patch_size]);
				}
			}
			/* Compute B (uniform blurring) end */


			/* Compute mRTV begin */
			ZQ_DImage<T> Ix, Iy, normNablaI(width,height);
			I.dx(Ix);
			I.dy(Iy);
			T*& Ix_data = Ix.data();
			T*& Iy_data = Iy.data();
			T*& normNablaI_data = normNablaI.data();
			for (int i = 0; i < nPixels; i++)
			{
				normNablaI_data[i] = sqrt(Ix_data[i] * Ix_data[i] + Iy_data[i] * Iy_data[i]);
			}

			ZQ_DImage<T> sumNabla_per_row(paddedWidth, height);
			ZQ_DImage<T> sumNabla_per_col(paddedWidth, paddedHeight);
			T*& sumNabla_per_row_data = sumNabla_per_row.data();
			T*& sumNabla_per_col_data = sumNabla_per_col.data();
			for (int h = 0; h < height; h++)
			{
				T* row_data = sumNabla_per_row_data + h*paddedWidth;
				row_data[0] = 0;
				for (int w = 1; w <= half_patch_size; w++)
					row_data[w] = row_data[w - 1] + 0;
				for (int w = half_patch_size + 1; w < half_patch_size + 1 + width; w++)
					row_data[w] = row_data[w - 1] + normNablaI_data[h*width + w - half_patch_size - 1];
				for (int w = half_patch_size + 1 + width; w < patch_size + width; w++)
					row_data[w] = row_data[w - 1] + 0;
			}
			for (int w = 0; w < paddedWidth; w++)
			{
				T* col_data = sumNabla_per_col_data + w;
				col_data[0] = 0;
				for (int h = 1; h <= half_patch_size; h++)
					col_data[h*paddedWidth] = col_data[(h - 1)*paddedWidth] + 0;
				for (int h = half_patch_size + 1; h < half_patch_size + 1 + height; h++)
					col_data[h*paddedWidth] = col_data[(h - 1)*paddedWidth] + sumNabla_per_row_data[(h - half_patch_size - 1)*paddedWidth + w];
				for (int h = half_patch_size + 1 + height; h < patch_size + height; h++)
					col_data[h*paddedWidth] = col_data[(h - 1)*paddedWidth] + 0;
			}

			ZQ_DImage<T> mRTV(width, height);
			T*& mRTV_data = mRTV.data();
			double eps = 1e-9;
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					int offset = h*width + w;
					double denom = eps + sumNabla_per_col_data[(h + patch_size)*paddedWidth + w + patch_size]
						+ sumNabla_per_col_data[h*paddedWidth + w]
						- sumNabla_per_col_data[(h + patch_size)*paddedWidth + w]
						- sumNabla_per_col_data[h*paddedWidth + w + patch_size];
					T max_I = I_data[offset];
					T min_I = I_data[offset];
					T max_Nabla = normNablaI_data[offset];
					int start_hh = __max(0, h - half_patch_size);
					int end_hh = __min(height - 1, h + half_patch_size);
					int start_ww = __max(0, w - half_patch_size);
					int end_ww = __min(width - 1, w + half_patch_size);
					for (int hh = start_hh; hh <= end_hh; hh++)
					{
						for (int ww = start_ww; ww <= end_ww; ww++)
						{
							int cur_off = hh*width + ww;
							max_I = __max(max_I, I_data[cur_off]);
							min_I = __min(min_I, I_data[cur_off]);
							max_Nabla = __max(max_Nabla, normNablaI_data[cur_off]);
						}
					}

					mRTV_data[offset] = (max_I - min_I)*max_Nabla / denom;
				}
			}
			/* Compute mRTV end */

			/* Compute G (guidance) begin */
			ZQ_DImage<T> G(width, height), G1(width,height);
			T*& G_data = G.data();
			T*& G1_data = G1.data();
			ZQ_DImage<T> alpha(width, height);
			T*& alpha_data = alpha.data();
			double sigma_for_alpha = opt.sigma_for_alpha;
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					int offset = h*width + w;
					T min_B = B_data[offset];
					T min_mRTV = mRTV_data[offset];
					int start_hh = __max(0, h - half_patch_size);
					int end_hh = __min(height - 1, h + half_patch_size);
					int start_ww = __max(0, w - half_patch_size);
					int end_ww = __min(width - 1, w + half_patch_size);
					for (int hh = start_hh; hh <= end_hh; hh++)
					{
						for (int ww = start_ww; ww <= end_ww; ww++)
						{
							int cur_off = hh*width + ww;
							if (mRTV_data[cur_off] < min_mRTV)
							{
								min_B = B_data[cur_off];
								min_mRTV = mRTV_data[cur_off];
							}
						}
					}

					G_data[offset] = min_B;
					alpha_data[offset] = 2 * (1.0 / (1.0 + exp(-sigma_for_alpha*(mRTV_data[offset] - min_mRTV))) - 0.5);
					G1_data[offset] = alpha_data[offset] * G_data[offset] + (1.0 - alpha_data[offset])*B_data[offset];
				}
			}
			/* Compute G (guidance) end */
			
			/* joint bilateral filter */
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					double sum_weight = 0;
					double sum_val = 0;
					double cur_val = G1_data[h*width + w];
					int start_hh = __max(0, h - fsize);
					int end_hh = __min(height - 1, h + fsize);
					int start_ww = __max(0, w - fsize);
					int end_ww = __min(width - 1, w + fsize);
					for (int hh = start_hh; hh <= end_hh; hh++)
					{
						for (int ww = start_ww; ww <= end_ww; ww++)
						{
							float tmp_val = G1_data[hh*width + ww];
							float val_dis2 = (cur_val - tmp_val)*(cur_val - tmp_val);
							float tmp_wei = exp(-0.5*val_dis2 / sigma_v2)*gauss_weight_data[(hh + fsize - h)*fsize_len + ww + fsize - w];
							sum_weight += tmp_wei;
							sum_val += tmp_wei*tmp_val;
						}
					}
					output_data[(h*width + w)*nChannels + c] = sum_val / sum_weight;
				}
			}
		}

		return true;
	}
}

#endif