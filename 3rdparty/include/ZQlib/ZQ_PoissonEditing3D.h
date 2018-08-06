#ifndef _ZQ_POISSON_EDITING_3D_H_
#define _ZQ_POISSON_EDITING_3D_H_
#pragma once

#include "ZQ_DoubleImage3D.h"
#include "ZQ_PoissonEditing3DOptions.h"

namespace ZQ
{
	class ZQ_PoissonEditing3D
	{
	public:

		ZQ_PoissonEditing3D(){}
		~ZQ_PoissonEditing3D(){}

	public:
		template<class T>
		static bool PoissonEditing(const ZQ_DImage3D<T>& mask, const ZQ_DImage3D<T>& copy_in, const ZQ_DImage3D<T>& input, ZQ_DImage3D<T>& output, const ZQ_PoissonEditing3DOptions& opt);

		template<class T>
		static bool PoissonEditing(const ZQ_DImage3D<T>& mask, const ZQ_DImage3D<T>& gx, const ZQ_DImage3D<T>& gy, const ZQ_DImage3D<T>& gz, const ZQ_DImage3D<T>& input, ZQ_DImage3D<T>& output, const ZQ_PoissonEditing3DOptions& opt);
	};	

	/*********************************************************************************/
	/********************************** definitions **********************************/
	/*********************************************************************************/

	template<class T>
	bool ZQ_PoissonEditing3D::PoissonEditing(const ZQ_DImage3D<T>& mask, const ZQ_DImage3D<T>& copy_in, const ZQ_DImage3D<T>& input, ZQ_DImage3D<T>& output, const ZQ_PoissonEditing3DOptions& opt)
	{
		int width = mask.width();
		int height = mask.height();
		int depth = mask.depth();
		if(!mask.matchDimension(width,height,depth,1))
			return false;
		if(copy_in.width() != width || copy_in.height() != height || copy_in.depth() != depth || input.width() != width || input.height() != height || input.depth() != depth)
			return false;

		int nChannels = copy_in.nchannels();
		if(input.nchannels() != nChannels)
			return false;

		output.copyData(input);

		ZQ_DImage3D<T> lap(width,height,depth,nChannels);

		T*& lap_data = lap.data();
		const T*& mask_data = mask.data();
		const T*& copy_in_data = copy_in.data();
		T*& output_data = output.data();


		switch(opt.type)
		{
		case ZQ_PoissonEditing3DOptions::METHOD_NAIVE:
			{
				ZQ_ImageProcessing3D::Laplacian<T>(copy_in_data,lap_data,width,height,depth,nChannels);
			}
			break;
		}


		lap.Multiplywith(opt.grad_scale);

		int ZSLICE = height*width*nChannels;
		int YSLICE = width*nChannels;
		int XSLICE = nChannels;
		for(int sor_it = 0; sor_it < opt.nSORIteration;sor_it++)
		{
			for(int d = 0;d < depth;d++)
			{
				for(int h = 0;h < height;h++)
				{
					for(int w = 0;w < width;w++)
					{
						int offset_single = d*height*width+h*width+w;
						if(mask_data[offset_single] > 0.5 && h > 0 && h < height-1 && w > 0 && w < width-1)
						{
							for(int c = 0;c < nChannels;c++)
							{
								int offset = offset_single*nChannels+c;
								float coeff = 6;
								float sigma = output_data[offset+XSLICE]
								+output_data[offset-XSLICE]
								+output_data[offset+YSLICE]
								+output_data[offset-YSLICE]
								+output_data[offset+ZSLICE]
								+output_data[offset-ZSLICE]
								- lap_data[offset];

								output_data[offset] = sigma/coeff;
							}

						}
					}
				}
			}
			
		}

		return true;
	}

	template<class T>
	bool ZQ_PoissonEditing3D::PoissonEditing(const ZQ_DImage3D<T>& mask, const ZQ_DImage3D<T>& gx, const ZQ_DImage3D<T>& gy, const ZQ_DImage3D<T>& gz, 
						const ZQ_DImage3D<T>& input, ZQ_DImage3D<T>& output, const ZQ_PoissonEditing3DOptions& opt)
	{
		int width = mask.width();
		int height = mask.height();
		int depth = mask.depth();
		if(!mask.matchDimension(width,height,depth,1))
			return false;
		int nChannels = input.nchannels();

		if(!gx.matchDimension(width,height,depth,nChannels))
			return false;

		if(!gy.matchDimension(width,height,depth,nChannels))
			return false;

		if(!gz.matchDimension(width,height,depth,nChannels))
			return false;

		if(!input.matchDimension(width,height,depth,nChannels))
			return false;

		output.copyData(input);

		ZQ_DImage3D<T> lap(width,height,depth,nChannels);

		T*& lap_data = lap.data();
		const T*& mask_data = mask.data();
		const T*& gx_data = gx.data();
		const T*& gy_data = gy.data();
		const T*& gz_data = gz.data();
		T*& output_data = output.data();

		int XSLICE = nChannels;
		int YSLICE = width*nChannels;
		int ZSLICE = height*width*nChannels;
		for(int c = 0;c < nChannels;c++)
		{
			for(int d = 1; d < depth;d++)
			{
				for(int h = 0;h < height;h++)
				{
					for(int w = 0;w < width;w++)
					{
						int offset = (d*height*width+h*width+w)*nChannels+c;
						lap_data[offset] += gz_data[offset] - gz_data[offset-ZSLICE];
					}
				}
			}
			for(int d = 0;d < depth;d++)
			{
				for(int h = 1;h < height;h++)
				{
					for(int w = 0;w < width;w++)
					{
						int offset = (d*height*width+h*width+w)*nChannels+c;
						lap_data[offset] += gy_data[offset] - gy_data[offset-YSLICE];
					}
				}
			}
			
			for(int d = 0;d < depth;d++)
			{
				for(int h = 0;h < height;h++)
				{
					for(int w = 1;w < width;w++)
					{
						int offset = (d*height*width+h*width+w)*nChannels+c;
						lap_data[offset] += gx_data[offset] - gx_data[offset-XSLICE];
					}
				}
			}
		}

		lap.Multiplywith(opt.grad_scale);

		for(int sor_it = 0; sor_it < opt.nSORIteration;sor_it++)
		{
			for(int d = 0;d < depth;d++)
			{
				for(int h = 0;h < height;h++)
				{
					for(int w = 0;w < width;w++)
					{
						int offset_single = d*height*width+h*width+w;
						if(mask_data[offset_single] > 0.5 && h > 0 && h < height-1 && w > 0 && w < width-1)
						{
							for(int c = 0;c < nChannels;c++)
							{
								int offset = offset_single*nChannels+c;
								float coeff = 6;
								float sigma = output_data[offset+XSLICE]
								+output_data[offset-XSLICE]
								+output_data[offset+YSLICE]
								+output_data[offset-YSLICE]
								+output_data[offset+ZSLICE]
								+output_data[offset-ZSLICE]
								- lap_data[offset];

								output_data[offset] = sigma/coeff;
							}

						}
					}
				}
			}

		}

		return true;
	}

}

#endif