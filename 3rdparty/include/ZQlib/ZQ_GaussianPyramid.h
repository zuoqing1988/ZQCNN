#ifndef _ZQ_GAUSSIAN_PYRAMID_H_
#define _ZQ_GAUSSIAN_PYRAMID_H_

#pragma once

#include "ZQ_DoubleImage.h"
#include <vector>
#include <iostream>

namespace ZQ
{
	template<class T>
	class ZQ_GaussianPyramid
	{
	private:
		std::vector<ZQ_DImage<T>> ImPyramid;
		int nLevels;
		double ratio;
	public:
		ZQ_GaussianPyramid(void){nLevels = 0; ratio = 0;}
		~ZQ_GaussianPyramid(void){ImPyramid.clear();}
		double ConstructPyramid(const ZQ_DImage<T>& image, const double ratio = 0.5, const int minWidth = 16);
		int nlevels() const {return nLevels;};
		ZQ_DImage<T>& Image(int index) { return ImPyramid[index]; };
	};


	/*********************************************************************************/
	/********************************** definitions **********************************/
	/*********************************************************************************/

	template<class T>
	double ZQ_GaussianPyramid<T>::ConstructPyramid(const ZQ_DImage<T> &image, const double ratio, const int minWidth)
	{
		
		if(ratio > 0.9 || ratio < 0.4)
			this->ratio = 0.5;
		else
			this->ratio = ratio;

		int width = image.width();
		int height = image.height();
		// first decide how many levels
		int N1 = 1 + floor(log(__max((double)width, height) / (double)minWidth) / log(1.0 / ratio));
		// smaller size shouldn't be less than 6
		int	N2 = 1 + floor(log(__min((double)width, height) / 6.0) / log(1.0 / ratio));
		nLevels = __min(N1, N2);

		ImPyramid.clear();
		for(int i = 0;i < nLevels;i++)
		{
			ZQ_DImage<T> tmp;
			ImPyramid.push_back(tmp);
		}
		ImPyramid[0].copyData(image);
		
		double factor = sqrt(2.0);

		double smooth_sigma = sqrt(1.0/ratio) / factor;
		int fsize = (int)(1.5*smooth_sigma + 0.5);
		for (int i = 1; i < nLevels; i++)
		{
			ZQ_DImage<T> foo;

			ImPyramid[i - 1].GaussianSmoothing(foo, smooth_sigma, fsize);
			foo.imresizeBicubic(ImPyramid[i], ratio);
			//foo.imresize(ImPyramid[i],ratio);
		}
		return this->ratio;
	}
}




#endif