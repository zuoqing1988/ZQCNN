#ifndef _ZQ_POISSON_EDITING_H_
#define _ZQ_POISSON_EDITING_H_
#pragma once

#include "ZQ_DoubleImage.h"
#include "ZQ_PoissonEditingOptions.h"

namespace ZQ
{
	class ZQ_PoissonEditing
	{
	public:
		template<class T>
		static bool PoissonEditing(const ZQ_DImage<T>& mask, const ZQ_DImage<T>& copy_in, const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_PoissonEditingOptions& opt);

		template<class T>
		static bool PoissonEditing(const ZQ_DImage<T>& mask, const ZQ_DImage<T>& gx, const ZQ_DImage<T>& gy, const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_PoissonEditingOptions& opt);

		template<class T>
		static bool PoissonEditingAnisotropic(const ZQ_DImage<T>& mask, const ZQ_DImage<T>& dir, const ZQ_DImage<T>& copyin, const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_PoissonEditingOptions& opt);

		template<class T>
		static bool PoissonEditingAnisotropic(const ZQ_DImage<T>& mask, const ZQ_DImage<T>& dir, const ZQ_DImage<T>& gx, const ZQ_DImage<T>& gy, const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_PoissonEditingOptions& opt);
	};	

	/*********************************************************************************/
	/********************************** definitions **********************************/
	/*********************************************************************************/

	template<class T>
	bool ZQ_PoissonEditing::PoissonEditing(const ZQ_DImage<T>& mask, const ZQ_DImage<T>& copy_in, const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_PoissonEditingOptions& opt)
	{
		int width = mask.width();
		int height = mask.height();
		if(!mask.matchDimension(width,height,1))
			return false;
		if(copy_in.width() != width || copy_in.height() != height || input.width() != width || input.height() != height)
			return false;

		int nChannels = copy_in.nchannels();
		if(input.nchannels() != nChannels)
			return false;

		output.copyData(input);

		ZQ_DImage<T> lap(width,height,nChannels);

		T*& lap_data = lap.data();
		const T*& mask_data = mask.data();
		const T*& copy_in_data = copy_in.data();
		T*& output_data = output.data();


		switch(opt.type)
		{
		case ZQ_PoissonEditingOptions::METHOD_NAIVE:
			{
				ZQ_ImageProcessing::Laplacian<T>(copy_in_data,lap_data,width,height,nChannels, false);
			}
			break;

		case ZQ_PoissonEditingOptions::METHOD_MIXGRADIENT:
			{
				float w1 = opt.weight1;
				ZQ_DImage<T> copy_in_gx(width,height,nChannels),copy_in_gy(width,height,nChannels);
				ZQ_DImage<T> output_gx(width,height,nChannels),output_gy(width,height,nChannels);
				copy_in.dx(copy_in_gx);
				copy_in.dy(copy_in_gy);
				output.dx(output_gx);
				output.dy(output_gy);

				T*& copy_in_gx_data = copy_in_gx.data();
				T*& copy_in_gy_data = copy_in_gy.data();
				T*& output_gx_data = output_gx.data();
				T*& output_gy_data = output_gy.data();

				ZQ_DImage<T> gx(width,height,nChannels),gy(width,height,nChannels);
				T*& gx_data = gx.data();
				T*& gy_data = gy.data();

				for(int c = 0;c < nChannels;c++)
				{
					for(int i = 0;i < width*height*nChannels;i++)
					{
						float gx1 = copy_in_gx_data[i];
						float gy1 = copy_in_gy_data[i];
						float gx2 = output_gx_data[i];
						float gy2 = output_gy_data[i];

						gx_data[i] = gx1*w1 + gx2*(1-w1);
						gy_data[i] = gy1*w1 + gy2*(1-w1);

					}


					for(int h = 1;h < height;h++)
					{
						for(int w = 0;w < width;w++)
						{
							lap_data[(h*width+w)*nChannels+c] += gy_data[(h*width+w)*nChannels+c] - gy_data[((h-1)*width+w)*nChannels+c];
						}
					}
					for(int h = 0;h < height;h++)
					{
						for(int w = 1;w < width;w++)
						{
							lap_data[(h*width+w)*nChannels+c] += gx_data[(h*width+w)*nChannels+c] - gx_data[(h*width+w-1)*nChannels+c];
						}
					}
				}	

			}
			break;

		case ZQ_PoissonEditingOptions::METHOD_MAXGRADIENT:
			{
				ZQ_DImage<T> copy_in_gx(width,height,nChannels),copy_in_gy(width,height,nChannels);
				ZQ_DImage<T> output_gx(width,height,nChannels),output_gy(width,height,nChannels);
				copy_in.dx(copy_in_gx);
				copy_in.dy(copy_in_gy);
				output.dx(output_gx);
				output.dy(output_gy);

				T*& copy_in_gx_data = copy_in_gx.data();
				T*& copy_in_gy_data = copy_in_gy.data();
				T*& output_gx_data = output_gx.data();
				T*& output_gy_data = output_gy.data();

				ZQ_DImage<T> gx(width,height,nChannels),gy(width,height,nChannels);
				T*& gx_data = gx.data();
				T*& gy_data = gy.data();

				for(int c = 0;c < nChannels;c++)
				{
					for(int i = 0;i < width*height*nChannels;i++)
					{
						float gx1 = copy_in_gx_data[i];
						float gy1 = copy_in_gy_data[i];
						float gx2 = output_gx_data[i];
						float gy2 = output_gy_data[i];
						float len1 = gx1*gx1+gy1*gy1;
						float len2 = gx2*gx2+gy2*gy2;
						if(len1 > len2)
						{
							gx_data[i] = gx1;
							gy_data[i] = gy1;
						}
						else
						{
							gx_data[i] = gx2;
							gy_data[i] = gy2;
						}
					}


					for(int h = 1;h < height;h++)
					{
						for(int w = 0;w < width;w++)
						{
							lap_data[(h*width+w)*nChannels+c] += gy_data[(h*width+w)*nChannels+c] - gy_data[((h-1)*width+w)*nChannels+c];
						}
					}
					for(int h = 0;h < height;h++)
					{
						for(int w = 1;w < width;w++)
						{
							lap_data[(h*width+w)*nChannels+c] += gx_data[(h*width+w)*nChannels+c] - gx_data[(h*width+w-1)*nChannels+c];
						}
					}
				}	
			}
			break;
		}


		lap.Multiplywith(opt.grad_scale);

		for(int sor_it = 0; sor_it < opt.nSORIteration;sor_it++)
		{
			for(int h = 0;h < height;h++)
			{
				for(int w = 0;w < width;w++)
				{
					if(mask_data[h*width+w] > 0.5 && h > 0 && h < height-1 && w > 0 && w < width-1)
					{
						for(int c = 0;c < nChannels;c++)
						{
							float coeff = 4;
							float sigma = output_data[(h*width+w+1)*nChannels+c]
							+output_data[(h*width+w-1)*nChannels+c]
							+output_data[((h+1)*width+w)*nChannels+c]
							+output_data[((h-1)*width+w)*nChannels+c] 
							- lap_data[(h*width+w)*nChannels+c];

							output_data[(h*width+w)*nChannels+c] = sigma/coeff;
						}
					}
				}
			}
		}

		return true;
	}

	template<class T>
	bool ZQ_PoissonEditing::PoissonEditing(const ZQ_DImage<T>& mask, const ZQ_DImage<T>& gx, const ZQ_DImage<T>& gy, const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_PoissonEditingOptions& opt)
	{
		int width = mask.width();
		int height = mask.height();
		if(!mask.matchDimension(width,height,1))
			return false;
		int nChannels = input.nchannels();

		if(!gx.matchDimension(width,height,nChannels))
			return false;

		if(!gy.matchDimension(width,height,nChannels))
			return false;

		if(!input.matchDimension(width,height,nChannels))
			return false;

		output.copyData(input);

		ZQ_DImage<T> lap(width,height,nChannels);

		T*& lap_data = lap.data();
		const T*& mask_data = mask.data();
		const T*& gx_data = gx.data();
		const T*& gy_data = gy.data();
		T*& output_data = output.data();

		for(int c = 0;c < nChannels;c++)
		{
			for(int h = 1;h < height;h++)
			{
				for(int w = 0;w < width;w++)
				{
					lap_data[(h*width+w)*nChannels+c] += gy_data[(h*width+w)*nChannels+c] - gy_data[((h-1)*width+w)*nChannels+c];
				}
			}
			for(int h = 0;h < height;h++)
			{
				for(int w = 1;w < width;w++)
				{
					lap_data[(h*width+w)*nChannels+c] += gx_data[(h*width+w)*nChannels+c] - gx_data[(h*width+w-1)*nChannels+c];
				}
			}
		}

		lap.Multiplywith(opt.grad_scale);
		
		for(int sor_it = 0; sor_it < opt.nSORIteration;sor_it++)
		{
			for(int h = 0;h < height;h++)
			{
				for(int w = 0;w < width;w++)
				{
					if(mask_data[h*width+w] > 0.5 && h > 0 && h < height-1 && w > 0 && w < width-1)
					{
						for(int c = 0;c < nChannels;c++)
						{
							float coeff = 4;
							float sigma = output_data[(h*width+w+1)*nChannels+c]
							+output_data[(h*width+w-1)*nChannels+c]
							+output_data[((h+1)*width+w)*nChannels+c]
							+output_data[((h-1)*width+w)*nChannels+c] 
							- lap_data[(h*width+w)*nChannels+c];

							output_data[(h*width+w)*nChannels+c] = sigma/coeff;
						}
					}
				}
			}
		}

		return true;
	}

	template<class T>
	bool ZQ_PoissonEditing::PoissonEditingAnisotropic(const ZQ_DImage<T>& mask, const ZQ_DImage<T>& dir, const ZQ_DImage<T>& copyin, const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_PoissonEditingOptions& opt)
	{
		int width = mask.width();
		int height = mask.height();
		if(!mask.matchDimension(width,height,1))
			return false;

		if(!dir.matchDimension(width,height,2))
			return false;

		if(input.width() != width || input.height() != height)
			return false;

		int nChannels = input.nchannels();

		if(!copyin.matchDimension(width,height,nChannels))
			return false;
		
		output.copyData(input);

		ZQ_DImage<T> lap(width,height,nChannels);

		T*& lap_data = lap.data();
		const T*& mask_data = mask.data();
		T*& output_data = output.data();
		ZQ_DImage<bool> occupy(width,height);

		bool*& occupy_data = occupy.data();
		for(int i = 0;i < width*height;i++)
			occupy_data[i] = mask_data[i] <= 0.5;
	
		int SLICE = width*nChannels;

		float ratio = opt.anisotropic_ratio;
		ZQ_DImage<T> wx(width,height), wy(width,height);
		ZQ_DImage<T> wxx(width,height), wyy(width,height);

		T*& wxx_data = wxx.data();
		T*& wx_data = wx.data();
		T*& wyy_data = wyy.data();
		T*& wy_data = wy.data();
		const T*& dir_data = dir.data();

		ZQ_PoissonEditingOptions tmp_opt;
		tmp_opt.type = ZQ_PoissonEditingOptions::METHOD_NAIVE;
		tmp_opt.nSORIteration = opt.nSORIteration;
		ZQ_DImage<T> tmp_copyin(width,height,nChannels);
		tmp_copyin.reset();
		ZQ_PoissonEditing::PoissonEditing(mask,tmp_copyin,input,output,tmp_opt);

		//compute wx,wy 
		for(int pp = 0;pp < width*height;pp++)
		{
			float gx = dir_data[pp*2];
			float gy = dir_data[pp*2+1];
			float rad = atan2(gy,gx);
			float rx2 = 1.0f/((cos(rad)*cos(rad))/(ratio*ratio)+(sin(rad)*sin(rad)));
			float ry2 = 1.0f/((sin(rad)*sin(rad))/(ratio*ratio)+(cos(rad)*cos(rad)));
			wx_data[pp] = sqrt(rx2);
			wy_data[pp] = sqrt(ry2);
		}
		wx.dx(wxx);
		wy.dy(wyy);

		// estimate base field
		for(int sor_it = 0;sor_it < opt.nSORIteration;sor_it++)
		{
			for(int h = 0;h < height;h++)
			{
				for(int w = 0;w < width;w++)
				{
					int offset_single = h*width+w;
					int offset = offset_single*nChannels;
					if(!occupy_data[offset_single] && h > 0 && h < height-1 && w > 0 && w < width-1)
					{
						for(int c = 0;c < nChannels;c++)
						{
							float sigma = 0;
							float coeff = 0;

							coeff += wxx_data[offset_single];
							sigma += wxx_data[offset_single]*output_data[offset+nChannels+c];

							coeff += wyy_data[offset_single];
							sigma += wyy_data[offset_single]*output_data[offset+SLICE+c];

							coeff += 2*wx_data[offset_single];
							sigma += wx_data[offset_single]*(output_data[offset+nChannels+c]+output_data[offset-nChannels+c]);

							coeff += 2*wy_data[offset_single];
							sigma += wy_data[offset_single]*(output_data[offset+SLICE+c]+output_data[offset-SLICE+c]);

							output_data[offset+c] = sigma/coeff;
						}
					}
				}
			}
		}

		//estimate detailed field

		ZQ_DImage<T> detail(width,height,nChannels);
		ZQ_DImage<T> input_low(width,height,nChannels);
		input_low.reset();

		tmp_opt.type = ZQ_PoissonEditingOptions::METHOD_NAIVE;
		tmp_opt.nSORIteration = opt.nSORIteration;
		tmp_opt.grad_scale = opt.grad_scale;

		PoissonEditing(mask,copyin,input_low,detail,tmp_opt);

		output.Addwith(detail);
		
		return true;
	}

	template<class T>
	bool ZQ_PoissonEditing::PoissonEditingAnisotropic(const ZQ_DImage<T>& mask, const ZQ_DImage<T>& dir, const ZQ_DImage<T>& gx, const ZQ_DImage<T>& gy, const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_PoissonEditingOptions& opt)
	{
		int width = mask.width();
		int height = mask.height();
		if(!mask.matchDimension(width,height,1))
			return false;

		if(!dir.matchDimension(width,height,2))
			return false;

		if(input.width() != width || input.height() != height)
			return false;

		int nChannels = input.nchannels();

		if(!gx.matchDimension(width,height,nChannels))
			return false;

		if(!gy.matchDimension(width,height,nChannels))
			return false;

		output.copyData(input);

		ZQ_DImage<T> lap(width,height,nChannels);

		T*& lap_data = lap.data();
		const T*& mask_data = mask.data();
		T*& output_data = output.data();
		ZQ_DImage<bool> occupy(width,height);

		bool*& occupy_data = occupy.data();
		for(int i = 0;i < width*height;i++)
			occupy_data[i] = mask_data[i] <= 0.5;

		int SLICE = width*nChannels;

		float ratio = opt.anisotropic_ratio;
		ZQ_DImage<T> wx(width,height), wy(width,height);
		ZQ_DImage<T> wxx(width,height), wyy(width,height);

		T*& wxx_data = wxx.data();
		T*& wx_data = wx.data();
		T*& wyy_data = wyy.data();
		T*& wy_data = wy.data();
		const T*& dir_data = dir.data();

		ZQ_PoissonEditingOptions tmp_opt;
		tmp_opt.type = ZQ_PoissonEditingOptions::METHOD_NAIVE;
		tmp_opt.nSORIteration = opt.nSORIteration;
		ZQ_DImage<T> tmp_copyin(width,height,nChannels);
		tmp_copyin.reset();
		ZQ_PoissonEditing::PoissonEditing(mask,tmp_copyin,input,output,tmp_opt);

		//compute wx,wy 
		for(int pp = 0;pp < width*height;pp++)
		{
			float gx = dir_data[pp*2];
			float gy = dir_data[pp*2+1];
			float rad = atan2(gy,gx);
			float rx2 = 1.0f/((cos(rad)*cos(rad))/(ratio*ratio)+(sin(rad)*sin(rad)));
			float ry2 = 1.0f/((sin(rad)*sin(rad))/(ratio*ratio)+(cos(rad)*cos(rad)));
			wx_data[pp] = sqrt(rx2);
			wy_data[pp] = sqrt(ry2);
		}
		wx.dx(wxx);
		wy.dy(wyy);

		// estimate base field
		for(int sor_it = 0;sor_it < opt.nSORIteration;sor_it++)
		{
			for(int h = 0;h < height;h++)
			{
				for(int w = 0;w < width;w++)
				{
					int offset_single = h*width+w;
					int offset = offset_single*nChannels;
					if(!occupy_data[offset_single] && h > 0 && h < height-1 && w > 0 && w < width-1)
					{
						for(int c = 0;c < nChannels;c++)
						{
							float sigma = 0;
							float coeff = 0;

							coeff += wxx_data[offset_single];
							sigma += wxx_data[offset_single]*output_data[offset+nChannels+c];

							coeff += wyy_data[offset_single];
							sigma += wyy_data[offset_single]*output_data[offset+SLICE+c];

							coeff += 2*wx_data[offset_single];
							sigma += wx_data[offset_single]*(output_data[offset+nChannels+c]+output_data[offset-nChannels+c]);

							coeff += 2*wy_data[offset_single];
							sigma += wy_data[offset_single]*(output_data[offset+SLICE+c]+output_data[offset-SLICE+c]);

							output_data[offset+c] = sigma/coeff;
						}
					}
				}
			}
		}

		//estimate detailed field

		ZQ_DImage<T> detail(width,height,nChannels);
		ZQ_DImage<T> input_low(width,height,nChannels);
		input_low.reset();

		tmp_opt.type = ZQ_PoissonEditingOptions::METHOD_NAIVE;
		tmp_opt.nSORIteration = opt.nSORIteration;
		tmp_opt.grad_scale = opt.grad_scale;

		PoissonEditing(mask,gx,gy,input_low,detail,tmp_opt);
		
		output.Addwith(detail);

		return true;
	}

}

#endif