#ifndef _ZQ_BILATERAL_FILTER_H_
#define _ZQ_BILATERAL_FILTER_H_
#pragma once 

#include "ZQ_DoubleImage.h"
#include "ZQ_BilateralFilterOptions.h"
#include <time.h>

namespace ZQ
{
	class ZQ_BilateralFilter
	{
	public:
		ZQ_BilateralFilter(){}
		~ZQ_BilateralFilter(){}

	public:
		template<class T>
		static bool BilateralFilter(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_BilateralFilterOptions& opt);
	};


	/*********************************************************************************/
	/********************************** definitions **********************************/
	/*********************************************************************************/

	/**************************  functions for users *****************************/


	template<class T>
	bool ZQ_BilateralFilter::BilateralFilter(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_BilateralFilterOptions& opt)
	{
		int width = input.width();
		int height = input.height();
		int nChannels = input.nchannels();
		output.allocate(width,height,nChannels);


		int fsize = opt.fsize;
		float sigma_s = opt.sigma_for_space;
		float sigma_v = opt.sigma_for_value;

		if(fsize < 1 || sigma_s <= 0 || sigma_v <= 0)
			return false;

		ZQ_DImage<float> gauss_weight(2*fsize+1,2*fsize+1);
		float*& gauss_weight_data = gauss_weight.data();

		float sigma_s2 = sigma_s*sigma_s;
		float sigma_v2 = sigma_v*sigma_v;
		int fsize_len = fsize*2+1;

		for(int hh = -fsize;hh <= fsize;hh++)
		{
			for(int ww = -fsize;ww <= fsize;ww++)
			{
				float dis2 = (float)hh*hh+ww*ww;
				gauss_weight_data[(hh+fsize)*fsize_len+ww+fsize] = exp(-0.5*dis2/sigma_s2);
			}
		}

		const T*& input_data = input.data();
		T*& output_data = output.data();

		for(int c = 0;c < nChannels;c++)
		{
			for(int h = 0;h < height;h++)
			{
				for(int w = 0;w < width;w++)
				{
					float sum_weight = 0;
					float sum_val = 0;
					float cur_val = input_data[(h*width+w)*nChannels+c];
					for(int hh = __max(0,h-fsize); hh <= __min(height-1,h+fsize);hh++)
					{
						for(int ww = __max(0,w-fsize); ww <= __min(width-1,w+fsize);ww++)
						{
							float tmp_val = input_data[(hh*width+ww)*nChannels+c];
							float val_dis2 = (cur_val-tmp_val)*(cur_val-tmp_val);
							float tmp_wei = exp(-0.5*val_dis2/sigma_v2)*gauss_weight_data[(hh+fsize-h)*fsize_len+ww+fsize-w];
							sum_weight += tmp_wei;
							sum_val += tmp_wei*tmp_val;
						}
					}
					output_data[(h*width+w)*nChannels+c] = sum_val/sum_weight;
				}
			}
		}
		return true;
	}


}

#endif