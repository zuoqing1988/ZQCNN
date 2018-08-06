#ifndef _ZQ_GAUSSIAN_PYRAMID_3D_H_
#define _ZQ_GAUSSIAN_PYRAMID_3D_H_

#pragma once

#include "ZQ_DoubleImage3D.h"
#include <vector>
#include <iostream>

namespace ZQ
{
	template<class T>
	class ZQ_GaussianPyramid3D
	{
	private:
		std::vector<ZQ_DImage3D<T>> ImPyramid;
		int nLevels;
		double ratio;
	public:
		ZQ_GaussianPyramid3D(void){nLevels = 0; ratio = 0;}
		~ZQ_GaussianPyramid3D(void){ImPyramid.clear();}
		double ConstructPyramid(const ZQ_DImage3D<T>& image, const double ratio = 0.5, const int minWidth = 16);
		int nlevels() const {return nLevels;};
		ZQ_DImage3D<T>& Image(int index) { return ImPyramid[index]; };
	};


	/*********************************************************************************/
	/********************************** definitions **********************************/
	/*********************************************************************************/

	template<class T>
	double ZQ_GaussianPyramid3D<T>::ConstructPyramid(const ZQ_DImage3D<T> &image, const double ratio, const int minWidth)
	{

		if(ratio > 0.9 || ratio < 0.4)
			this->ratio = 0.5;
		else
			this->ratio = ratio;

		// first decide how many levels
		nLevels = log((double)minWidth/image.width())/log(this->ratio)+1;
		ImPyramid.clear();
		for(int i = 0;i < nLevels;i++)
		{
			ZQ_DImage3D<T> tmp;
			ImPyramid.push_back(tmp);
		}
		ImPyramid[0].copyData(image);
		double baseSigma = (1.0/this->ratio-1);
		int n = log(0.25)/log(this->ratio);
		double nSigma=baseSigma*n;
		for(int i = 1;i < nLevels;i++)
		{
			ZQ_DImage3D<T> foo;
			if(i <= n)
			{
				double sigma = baseSigma*i;
				image.GaussianSmoothing(foo,sigma,sigma*3);
				foo.imresizeTricubic(ImPyramid[i],pow(this->ratio,i));
				//foo.imresize(ImPyramid[i],pow(ratio,i));
			}
			else
			{
				ImPyramid[i-n].GaussianSmoothing(foo,nSigma,nSigma*3);
				double rate = (double)pow(this->ratio,i)*image.width()/foo.width();
				foo.imresizeTricubic(ImPyramid[i],rate);
				//foo.imresize(ImPyramid[i],rate);
			}
		}
		return this->ratio;
	}
}




#endif