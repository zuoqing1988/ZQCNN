#ifndef _ZQ_TEXTURE_SAMPLER_H_
#define _ZQ_TEXTURE_SAMPLER_H_
#pragma once

#include "ZQ_DoubleImage.h"
#include <math.h>

namespace ZQ
{
	template<class T>
	class ZQ_TextureSampler
	{
	public:
		ZQ_TextureSampler();
		~ZQ_TextureSampler();

	private:
		bool has_data;
		ZQ_DImage<T> data;
		bool x_wrap_mode;
		bool y_wrap_mode;

	public:
		bool BindImage(const ZQ_DImage<T>& img, bool wrap_mode);
		bool BindImage(const ZQ_DImage<T>& img, bool x_wrap_mode, bool y_wrap_mode);
		bool Sample_NormalizedCoord(float x, float y, T* result, bool cubic) const;
		bool Sample(float x, float y, T* result, bool cubic) const;
		int nchannels() const { return data.nchannels(); }
	};

	/*********************************************************************************/
	/********************************** definitions **********************************/
	/*********************************************************************************/
	template<class T>
	ZQ_TextureSampler<T>::ZQ_TextureSampler()
	{
		x_wrap_mode = false;
		y_wrap_mode = false;
		has_data = false;
	}

	template<class T>
	ZQ_TextureSampler<T>::~ZQ_TextureSampler()
	{

	}

	template<class T>
	bool ZQ_TextureSampler<T>::BindImage(const ZQ_DImage<T>& img, bool warp_mode)
	{
		return BindImage(img, warp_mode, warp_mode);
	}

	template<class T>
	bool ZQ_TextureSampler<T>::BindImage(const ZQ_DImage<T>& img, bool x_wrap_mode, bool y_wrap_mode)
	{
		int width = img.width();
		int height = img.height();
		int nChannels = img.nchannels();
		if (width <= 0 || height <= 0 || nChannels <= 0)
			return false;

		this->x_wrap_mode = x_wrap_mode;
		this->y_wrap_mode = y_wrap_mode;
		int padding_width = width + 4;
		int padding_height = height + 4;
		data.allocate(padding_width, padding_height, nChannels);
		T*& ptr = data.data();
		const T*& img_data = img.data();

		int XSLICE = nChannels;
		int YSLICE = padding_width*nChannels;


		if (x_wrap_mode)
		{
			for (int j = 0; j < height; j++)
				memcpy(ptr + (j + 2)*YSLICE + XSLICE * 2, img_data + j*width*nChannels, sizeof(T)*width*nChannels);
		}
		else
		{
			for (int j = 0; j < height; j++)
				memcpy(ptr + (j + 2)*YSLICE + XSLICE * 2, img_data + j*width*nChannels, sizeof(T)*width*nChannels);
		}

		for (int j = 2; j <= height + 1; j++)
		{
			memcpy(ptr + j*YSLICE, ptr + j*YSLICE + (padding_width - 4)*XSLICE, sizeof(T)*XSLICE * 2);
			memcpy(ptr + j*YSLICE + (padding_width - 2)*XSLICE, ptr + j*YSLICE + XSLICE * 2, sizeof(T)*XSLICE * 2);
		}

		if (y_wrap_mode)
		{
			memcpy(ptr, ptr + (padding_height - 4)*YSLICE, sizeof(T)*YSLICE);
			memcpy(ptr + YSLICE, ptr + (padding_height - 3)*YSLICE, sizeof(T)*YSLICE);
			memcpy(ptr + (padding_height - 2)*YSLICE, ptr + YSLICE * 2, sizeof(T)*YSLICE);
			memcpy(ptr + (padding_height - 1)*YSLICE, ptr + YSLICE * 3, sizeof(T)*YSLICE);
		}
		else
		{
			memcpy(ptr, ptr + YSLICE * 2, sizeof(T)*YSLICE);
			memcpy(ptr + YSLICE, ptr + YSLICE * 2, sizeof(T)*YSLICE);
			memcpy(ptr + (padding_height - 2)*YSLICE, ptr + (padding_height - 3)*YSLICE, sizeof(T)*YSLICE);
			memcpy(ptr + (padding_height - 1)*YSLICE, ptr + (padding_height - 3)*YSLICE, sizeof(T)*YSLICE);
		}

		has_data = true;
		return true;
	}

	template<class T>
	bool ZQ_TextureSampler<T>::Sample_NormalizedCoord(float x, float y, T* result, bool cubic) const
	{
		if(!has_data)
			return false;

		int width = data.width();
		int height = data.height();
		int nChannels = data.nchannels();
		const T*& ptr = data.data();

		float real_x = 0,real_y = 0;
		if(x_wrap_mode)
		{
			real_x = (x - floor(x))*(width-4) + 1.5;
		}
		else
		{
			real_x = __min(1.0,__max(0,x)) * (width-4) + 1.5;
		}

		if (y_wrap_mode)
		{
			real_y = (y - floor(y))*(height - 4) + 1.5;
		}
		else
		{
			real_y = __min(1.0, __max(0, y)) * (height - 4) + 1.5;
		}

		if(cubic)
			ZQ_ImageProcessing::BicubicInterpolate(ptr,width,height,nChannels,real_x,real_y,result,false);
		else
			ZQ_ImageProcessing::BilinearInterpolate(ptr,width,height,nChannels,real_x,real_y,result,false);
		
		return true;
	}

	template<class T>
	bool ZQ_TextureSampler<T>::Sample(float x, float y, T* result, bool cubic) const
	{
		int width = data.width();
		int height = data.height();
		if(width <= 4 || height <= 4)
			return false;
		
		return Sample_NormalizedCoord(x/(width-4),y/(height-4),result,cubic);
	}
}

#endif