#ifndef _ZQ_TEXTURE_SAMPLER_3D_H_
#define _ZQ_TEXTURE_SAMPLER_3D_H_
#pragma once

#include "ZQ_DoubleImage3D.h"
#include <math.h>

namespace ZQ
{
	template<class T>
	class ZQ_TextureSampler3D
	{
	public:
		ZQ_TextureSampler3D();
		~ZQ_TextureSampler3D();

	private:
		bool has_data;
		ZQ_DImage3D<T> data;
		bool wrap_mode;

	public:
		bool BindImage(const ZQ_DImage3D<T>& img, bool wrap_mode);
		bool Sample_NormalizedCoord(float x, float y, float z, T* result, bool cubic) const;
		bool Sample(float x, float y, float z, T* result, bool cubic) const;
	};

	/*********************************************************************************/
	/********************************** definitions **********************************/
	/*********************************************************************************/
	template<class T>
	ZQ_TextureSampler3D<T>::ZQ_TextureSampler3D()
	{
		wrap_mode = false;
		has_data = false;
	}

	template<class T>
	ZQ_TextureSampler3D<T>::~ZQ_TextureSampler3D()
	{

	}

	template<class T>
	bool ZQ_TextureSampler3D<T>::BindImage(const ZQ_DImage3D<T>& img, bool wrap_mode)
	{
		int width = img.width();
		int height = img.height();
		int depth = img.depth();
		int nChannels = img.nchannels();
		if(width <= 0 || height <= 0 || depth <= 0 || nChannels <= 0)
			return false;

		this->wrap_mode = wrap_mode;
		int padding_width = width+2;
		int padding_height = height+2;
		int padding_depth = depth+2;
		data.allocate(padding_width,padding_height,padding_depth,nChannels);
		T*& ptr = data.data();
		const T*& img_data = img.data();

		int ZSLICE = padding_height*padding_width*nChannels;
		int YSLICE = padding_width*nChannels;
		int XSLICE = nChannels;

		if(wrap_mode)
		{
			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
					memcpy(ptr+(k+1)*ZSLICE+(j+1)*YSLICE+XSLICE,img_data+(k*height*width+j*width)*nChannels,sizeof(T)*width*nChannels);
			}
			
			for(int k = 1;k <= depth;k++)
			{
				for(int j = 1;j <= height;j++)
				{
					memcpy(ptr+k*ZSLICE+j*YSLICE,ptr+k*ZSLICE+j*YSLICE+(padding_width-2)*XSLICE,sizeof(T)*XSLICE);
					memcpy(ptr+k*ZSLICE+j*YSLICE+(padding_width-1)*XSLICE,ptr+k*ZSLICE+j*YSLICE+XSLICE,sizeof(T)*XSLICE);
				}
			}

			for(int k = 1;k <= depth;k++)
			{
				memcpy(ptr+k*ZSLICE,ptr+k*ZSLICE+(padding_height-2)*YSLICE,sizeof(T)*YSLICE);
				memcpy(ptr+k*ZSLICE+(padding_height-1)*YSLICE,ptr+k*ZSLICE+YSLICE,sizeof(T)*YSLICE);
			}
			
			memcpy(ptr,ptr+(padding_depth-2)*ZSLICE,sizeof(T)*ZSLICE);
			memcpy(ptr+(padding_depth-1)*ZSLICE,ptr+ZSLICE,sizeof(T)*ZSLICE);
			
		}
		else
		{
			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
					memcpy(ptr+(k+1)*ZSLICE+(j+1)*YSLICE+XSLICE,img_data+(k*height*width+j*width)*nChannels,sizeof(T)*width*nChannels);
			}

			for(int k = 1;k <= depth;k++)
			{
				for(int j = 1;j <= height;j++)
				{
					memcpy(ptr+k*ZSLICE+j*YSLICE,ptr+k*ZSLICE+j*YSLICE+XSLICE,sizeof(T)*XSLICE);
					memcpy(ptr+k*ZSLICE+j*YSLICE+(padding_width-1)*XSLICE,ptr+k*ZSLICE+j*YSLICE+(padding_width-2)*XSLICE,sizeof(T)*XSLICE);
				}
			}

			for(int k = 1;k <= depth;k++)
			{
				memcpy(ptr+k*ZSLICE,ptr+k*ZSLICE+YSLICE,sizeof(T)*YSLICE);
				memcpy(ptr+k*ZSLICE+(padding_height-1)*YSLICE,ptr+k*ZSLICE+(padding_height-2)*YSLICE,sizeof(T)*YSLICE);
			}

			memcpy(ptr,ptr+ZSLICE,sizeof(T)*ZSLICE);
			memcpy(ptr+(padding_depth-1)*ZSLICE,ptr+(padding_depth-2)*ZSLICE,sizeof(T)*ZSLICE);

		}
		has_data = true;
		return true;
	}

	template<class T>
	bool ZQ_TextureSampler3D<T>::Sample_NormalizedCoord(float x, float y, float z, T* result, bool cubic) const
	{
		if(!has_data)
			return false;

		int width = data.width();
		int height = data.height();
		int depth = data.depth();
		int nChannels = data.nchannels();
		const T*& ptr = data.data();

		float real_x = 0,real_y = 0,real_z = 0;
		if(wrap_mode)
		{
			real_x = (x - floor(x))*(width-2) + 0.5;
			real_y = (y - floor(y))*(height-2) + 0.5;
			real_z = (z - floor(z))*(depth-2) + 0.5;
		}
		else
		{
			real_x = __min(1.0,__max(0,x)) * (width-2) + 0.5;
			real_y = __min(1.0,__max(0,y)) * (height-2) + 0.5;
			real_z = __min(1.0,__max(0,z)) * (depth-2) + 0.5;
		}

		if(cubic)
			ZQ_ImageProcessing3D::TricubicInterpolate(ptr,width,height,depth,nChannels,real_x,real_y,real_z,result,false);
		else
			ZQ_ImageProcessing3D::TrilinearInterpolate(ptr,width,height,depth,nChannels,real_x,real_y,real_z,result,false);

		return true;
	}

	template<class T>
	bool ZQ_TextureSampler3D<T>::Sample(float x, float y, float z, T* result, bool cubic) const
	{
		int width = data.width();
		int height = data.height();
		int depth = data.depth();
		if(width <= 2 || height <= 2 || depth <= 2)
			return false;

		return Sample_NormalizedCoord(x/(width-2),y/(height-2),z/(depth-2),result,cubic);
	}
}

#endif