#ifndef _ZQ_INPAINTING_H_
#define _ZQ_INPAINTING_H_
#pragma once

#include "ZQ_DoubleImage.h"
#include "ZQ_GaussianPyramid.h"
#include "ZQ_InpaintingOptions.h"

namespace ZQ
{
	class ZQ_Inpainting
	{
	public:
		ZQ_Inpainting();
		~ZQ_Inpainting();

		template<class T>
		static bool Inpainting(const ZQ_DImage<T>& input, const ZQ_DImage<T>& mask, ZQ_DImage<T>& output, const ZQ_InpaintingOptions& opt);

	protected:
		template<class T>
		static bool InpaintingPDE3rdOrder(const ZQ_DImage<T>& input, const ZQ_DImage<T>& mask, ZQ_DImage<T>& output, const ZQ_InpaintingOptions& opt);

		template<class T>
		static bool InpaintingTextureSynthesis(const ZQ_DImage<T>& input, const ZQ_DImage<T>& mask, ZQ_DImage<T>& output, const ZQ_InpaintingOptions& opt);

	private:

		template<class T>
		static void PDE3rdOrder_solve_for_p(ZQ_DImage<T>& p, ZQ_DImage<T>& L, const ZQ_DImage<T>& mask, float lambda, int nSORIter);

		template<class T>
		static void PDE3rdOrder_solve_for_I(ZQ_DImage<T>& I, ZQ_DImage<T>& p, const ZQ_DImage<T>& mask, int nSORIter);

		template<class T>
		static void TextureSynthesis_compute_mask_hat(const ZQ_DImage<T>& mask, ZQ_DImage<T>& mask_hat, const ZQ_InpaintingOptions& opt);

		template<class T>
		static void TextureSynthesis_firstpass(const ZQ_DImage<T>& input, const ZQ_DImage<T>& mask, const ZQ_DImage<T>& mask_hat, ZQ_DImage<T>& output, const ZQ_InpaintingOptions& opt);

		template<class T>
		static void TextureSynthesis_succeedpass(const ZQ_DImage<T>& input, const ZQ_DImage<T>& mask, const ZQ_DImage<T>& mask_hat, ZQ_DImage<T>& output, const ZQ_InpaintingOptions& opt);
	};

	/*********************************************************************************/
	/********************************** definitions **********************************/
	/*********************************************************************************/

	template<class T>
	bool ZQ_Inpainting::Inpainting(const ZQ_DImage<T>& input, const ZQ_DImage<T>& mask, ZQ_DImage<T>& output, const ZQ_InpaintingOptions& opt)
	{
		switch(opt.type)
		{
		case ZQ_InpaintingOptions::METHOD_PDE_THIRD_ORDER:
			{
				return InpaintingPDE3rdOrder(input,mask,output,opt);
			}
			break;
		case ZQ_InpaintingOptions::METHOD_PYRAMID_TEXTURE_SYNTHESIS:
			{
				return InpaintingTextureSynthesis(input,mask,output,opt);
			}
			break;
		}
		return false;
	}

	template<class T>
	bool ZQ_Inpainting::InpaintingPDE3rdOrder(const ZQ_DImage<T>& input, const ZQ_DImage<T>& mask, ZQ_DImage<T>& output, const ZQ_InpaintingOptions& opt)
	{
		int width = input.width();
		int height = input.height();
		int nChannels = input.nchannels();

		if(!mask.matchDimension(width,height,1))
			return false;

		output.copyData(input);

		ZQ_DImage<T> mask_for_p(width,height,1);

		const T*& mask_ptr = mask.data();
		T *& mask_for_p_ptr = mask_for_p.data();

		const int dilate_size = 5;

		for(int h = 0;h < height;h++)
		{
			for(int w = 0;w < width;w++)
			{
				if(mask_ptr[h*width+w])
				{
					for(int hh = __max(0,h-dilate_size);hh <= __min(height-1,h+dilate_size);hh++)
					{
						for(int ww = __max(0,w-dilate_size);ww <= __min(width-1,w+dilate_size);ww++)
						{
							mask_for_p_ptr[hh*width+ww] = 1;
						}
					}
				}
			}
		}

		int nOuterIter = opt.nOuterIteration;
		int nSORIter = opt.nSORIteration;
		float lambda = opt.lambda;

		ZQ_DImage<T> p(width,height,nChannels);
		ZQ_DImage<T> L(width,height,nChannels);

		T*& I_ptr = output.data();
		T*& p_ptr = p.data();
		T*& L_ptr = L.data();

		for(int it = 0;it < nOuterIter;it++)
		{
			ZQ_ImageProcessing::Laplacian(I_ptr,L_ptr,width,height,nChannels, false);
			PDE3rdOrder_solve_for_p(p,L,mask_for_p,lambda,nSORIter);
			PDE3rdOrder_solve_for_I(output,p,mask,nSORIter);
		}

		return true;
	}

	template<class T>
	bool ZQ_Inpainting::InpaintingTextureSynthesis(const ZQ_DImage<T>& input, const ZQ_DImage<T>& mask, ZQ_DImage<T>& output, const ZQ_InpaintingOptions& opt)
	{
		int width = input.width();
		int height = input.height();
		int nChannels = input.nchannels();

		if(!mask.matchDimension(width,height,1))
			return false;

		float ratio = opt.ratio;
		int minWidth = opt.minWidth;
		int winWidth = opt.winWidth;
		int winHeight = opt.winHeight;
		int nOuterIter = opt.nOuterIteration;

		ZQ_GaussianPyramid<T> P_input,P_mask;

		P_input.ConstructPyramid(input,ratio,minWidth);
		P_mask.ConstructPyramid(mask,ratio,minWidth);


		int nLevels = P_input.nlevels();
		for(int k = nLevels-1;k >= 0 ;k--)
		{
			ZQ_DImage<T> tmp_input = P_input.Image(k);
			ZQ_DImage<T> tmp_mask = P_mask.Image(k);

			ZQ_DImage<T> mask_hat(mask);
			TextureSynthesis_compute_mask_hat(tmp_mask,mask_hat,opt);
			if(k == nLevels-1)
			{
				TextureSynthesis_firstpass(tmp_input,tmp_mask,mask_hat,output,opt);
			}
			else
			{
				int dst_width = tmp_mask.width();
				int dst_height = tmp_mask.height();
				output.imresize(dst_width,dst_height);
			}

			T*& output_data = output.data();
			T*& tmp_input_data = tmp_input.data();
			T*& tmp_mask_data = tmp_mask.data();

			for(int pp = 0;pp < tmp_input.npixels();pp++)
			{
				if(tmp_mask_data[pp] < 0.5)
				{
					for(int c = 0;c < nChannels;c++)
						output_data[pp*nChannels+c] = tmp_input_data[pp*nChannels+c];
				}
			}
			for(int i = 0;i < nOuterIter;i++)
			{
				TextureSynthesis_succeedpass(tmp_input,tmp_mask,mask_hat,output,opt);
			}
		}
		return true;
	}

	template<class T>
	void ZQ_Inpainting::PDE3rdOrder_solve_for_p(ZQ_DImage<T>& p, ZQ_DImage<T>& L, const ZQ_DImage<T>& mask, float lambda, int nSORIter)
	{
		int width = p.width();
		int height = p.height();
		int nChannels = p.nchannels();


		T*& p_ptr = p.data();
		T*& L_ptr = L.data();
		const T*& mask_ptr = mask.data();

		for(int it = 0;it < nSORIter;it++)
		{
			for(int h = 0;h < height;h++)
			{
				for(int w = 0;w < width;w++)
				{
					int offset = h*width+w;
					if(!mask_ptr[offset])
						continue;

					for(int c = 0;c < nChannels;c++)
					{
						float coeff = lambda, sigma = lambda*L_ptr[offset*nChannels+c];
						if(h > 0 && mask_ptr[(h-1)*width+w])
						{
							coeff += 1;
							sigma += p_ptr[((h-1)*width+w)*nChannels+c];
						}
						if(h < height-1 && mask_ptr[(h+1)*width+w])
						{
							coeff += 1;
							sigma += p_ptr[((h+1)*width+w)*nChannels+c];
						}
						if(w > 0 && mask_ptr[h*width+w-1])
						{
							coeff += 1;
							sigma += p_ptr[(h*width+w-1)*nChannels+c];
						}
						if(w < width-1 && mask_ptr[h*width+w+1])
						{
							coeff += 1;
							sigma += p_ptr[(h*width+w+1)*nChannels+c];
						}
						p_ptr[offset*nChannels+c] = sigma/coeff;
					}
				}
			}
		}

	}

	template<class T>
	void ZQ_Inpainting::PDE3rdOrder_solve_for_I(ZQ_DImage<T>& I, ZQ_DImage<T>& p, const ZQ_DImage<T>& mask, int nSORIter)
	{
		int width = I.width();
		int height = I.height();
		int nChannels = I.nchannels();

		T*& I_ptr = I.data();
		T*& p_ptr = p.data();
		const T*& mask_ptr = mask.data();

		for(int it = 0;it < nSORIter;it++)
		{
			for(int h = 0;h < height;h++)
			{
				for(int w = 0;w < width;w++)
				{
					int offset = h*width+w;
					if(!mask_ptr[offset])
						continue;

					for(int c = 0;c < nChannels;c++)
					{
						float coeff = 0, sigma = p_ptr[offset];
						if(h > 0)
						{
							coeff += 1;
							sigma += I_ptr[((h-1)*width+w)*nChannels+c];
						}
						if(h < height-1)
						{
							coeff += 1;
							sigma += I_ptr[((h+1)*width+w)*nChannels+c];
						}
						if(w > 0)
						{
							coeff += 1;
							sigma += I_ptr[(h*width+w-1)*nChannels+c];
						}
						if(w < width-1)
						{
							coeff += 1;
							sigma += I_ptr[(h*width+w+1)*nChannels+c];
						}
						I_ptr[offset*nChannels+c] = sigma/coeff;
					}
				}
			}
		}
	}


	template<class T>
	void ZQ_Inpainting::TextureSynthesis_compute_mask_hat(const ZQ_DImage<T>& mask, ZQ_DImage<T>& mask_hat,const ZQ_InpaintingOptions& opt)
	{
		int width = mask.width();
		int height = mask.height();

		const T*& mask_ptr = mask.data();
		T*& mask_hat_ptr = mask_hat.data();

		mask_hat.reset();

		int winWidth = opt.winWidth;
		int winHeight = opt.winHeight;

		for(int h = 0;h < height;h++)
		{
			for(int w = 0;w < width;w++)
			{
				if(mask_ptr[h*width+w] > 0.5)
				{

					for(int hh = __max(0,h-winHeight);hh <= __min(height-1,h+winHeight);hh++)
					{
						for(int ww = __max(0,w-winWidth);ww <= __min(width-1,w+winWidth);ww++)
						{
							mask_hat_ptr[hh*width+ww] = 1;
						}
					}
				}

			}

		}
	}

	template<class T>
	void ZQ_Inpainting::TextureSynthesis_firstpass(const ZQ_DImage<T>& input, const ZQ_DImage<T>& mask, const ZQ_DImage<T>& mask_hat, ZQ_DImage<T>& output, const ZQ_InpaintingOptions& opt)
	{
		int winWidth = opt.winWidth;
		int winHeight = opt.winHeight;
		bool display = opt.display;
		float probe = opt.probe;
		probe = __max(0,__min(1,probe));


		int width = input.width();
		int height = input.height();
		int nChannels = input.nchannels();

		ZQ_DImage<T> tmp_mask(mask);
		T*& tmp_mask_ptr = tmp_mask.data();
		const T*& mask_ptr = mask.data();
		const T*& mask_hat_ptr = mask_hat.data();
		const T*& input_ptr = input.data();
		T*& output_ptr = output.data();
		output.allocate(width,height,nChannels);

		for(int i = 0;i < height*width;i++)
		{
			for(int c = 0;c < nChannels;c++)
				output_ptr[i*nChannels+c] = mask_ptr[i] > 0.5 ? 0 : input_ptr[i*nChannels+c];
		}

		float gradWeight = opt.gradWeight;
		/*compute dx dy Begin*/
		ZQ_DImage<T> input_dx(width,height,nChannels),input_dy(width,height,nChannels);
		T*& input_dx_ptr = input_dx.data();
		T*& input_dy_ptr = input_dy.data();
		for(int h = 1;h < height;h++)
		{
			for(int w = 0;w < width;w++)
			{
				for(int c = 0;c < nChannels;c++)
					input_dy_ptr[(h*width+w)*nChannels+c] = input_ptr[(h*width+w)*nChannels+c] - input_ptr[((h-1)*width+w)*nChannels+c];
			}
		}
		for(int h = 0;h < height;h++)
		{
			for(int w = 1;w < width;w++)
			{
				for(int c = 0;c < nChannels;c++)
					input_dx_ptr[(h*width+w)*nChannels+c] = input_ptr[(h*width+w)*nChannels+c] - input_ptr[(h*width+w-1)*nChannels+c];
			}
		}
		/*compute dx dy End*/

		int size_w = 2*winWidth+1;
		int size_h = 2*winHeight+1;

		int count = 0;

		std::vector<int> search_x;
		std::vector<int> search_y;

		
		for(int hh = winHeight;hh < height-winHeight;hh++)
		{
			for(int ww = winWidth;ww < width-winWidth;ww++)
			{
				if(mask_hat_ptr[hh*width+ww] < 0.5)
				{
					search_x.push_back(ww);
					search_y.push_back(hh);
				}
			}
		}

		int total_search_count = search_x.size();
		int search_count = probe*total_search_count;


		for(int h = 0;h < height;h++)
		{
			for(int w = 0;w < width;w++)
			{		
				if(mask_ptr[h*width+w] > 0.5)
				{
					if(count % 100 == 0)
					{
						if(display)
							printf("count = %d\n",count);
					}
					count++;

					ZQ_DImage<T> local_mask(size_w,size_h);
					T*& local_mask_ptr = local_mask.data();

					/*cal local mask Begin*/

					for(int yy = -winHeight;yy <= winHeight;yy++)
					{
						for(int xx = -winWidth;xx <= winWidth;xx++)
						{
							local_mask_ptr[(yy+winHeight)*size_w+(xx+winWidth)] = 
								((h+yy >= 0) && (h+yy < height) && (w+xx >= 0) && (w+xx < width)
								&& (tmp_mask_ptr[(h+yy)*width+(w+xx)] < 0.5)) ? 1 : 0;
						}
					}


					/*cal local mask End*/

					bool has_result = false;
					int result_h = -1, result_w = -1; 
					float L2_error = 0;

					for(int ss = 0;ss < search_count;ss++)
					{
						int cur_search_x, cur_search_y;
						if(probe < 1)
						{
							int cur_idx = rand()%total_search_count;
							cur_search_x = search_x[cur_idx];
							cur_search_y = search_y[cur_idx];
						}
						else
						{
							cur_search_x = search_x[ss];
							cur_search_y = search_y[ss];
						}

						/*calculate tmp L2 error Begin*/
						float tmp_L2_error = 0;

						for(int yy = -winHeight;yy <= winHeight;yy++)
						{
							for(int xx = -winWidth;xx <= winWidth;xx++)
							{
								if(local_mask_ptr[(winHeight+yy)*size_w+(winWidth+xx)] > 0.5)
								{
									for(int c = 0;c < nChannels;c++)
									{
										float tmp_val_channel = input_ptr[((cur_search_y+yy)*width+(cur_search_x+xx))*nChannels+c] 
										- output_ptr[((h+yy)*width+(w+xx))*nChannels+c];
										tmp_L2_error += tmp_val_channel*tmp_val_channel;

										if(gradWeight > 0)
										{
											float tmp_dx = w+xx-1 < 0 ? 0 : (output_ptr[((h+yy)*width+(w+xx))*nChannels+c]
											- output_ptr[((h+yy)*size_w+(w+xx-1))*nChannels+c]);
											float tmp_dy = h+yy-1 < 0 ? 0 : (output_ptr[((h+yy)*width+(w+xx))*nChannels+c]
											- output_ptr[((h+yy-1)*width+(w+xx))*nChannels+c]);

											tmp_val_channel = input_dx_ptr[((cur_search_y+yy)*width+(cur_search_x+xx))*nChannels+c] - tmp_dx;
											tmp_L2_error += tmp_val_channel*tmp_val_channel;
											tmp_val_channel = input_dy_ptr[((cur_search_y+yy)*width+(cur_search_x+xx))*nChannels+c] - tmp_dy;
											tmp_L2_error += tmp_val_channel*tmp_val_channel;
										}
									}
								}
							}
						}

						/*calculate tmp L2 error End*/
						if(!has_result)
						{
							result_h = cur_search_y;
							result_w = cur_search_x;
							L2_error = tmp_L2_error;
							has_result = true;
						}
						else
						{
							if(L2_error > tmp_L2_error)
							{
								L2_error = tmp_L2_error;
								result_h = cur_search_y;
								result_w = cur_search_x;
							}

						}

					}

					
					/*search the best value for each hole pixel End*/
					for(int c = 0;c < nChannels;c++)
						output_ptr[(h*width+w)*nChannels+c] = input_ptr[(result_h*width+result_w)*nChannels+c];

					tmp_mask_ptr[h*width+w] = 0;
				}
				/* if mask end */
			}

		}
		/*for each hole pixel*/

	}

	template<class T>
	void ZQ_Inpainting::TextureSynthesis_succeedpass(const ZQ_DImage<T>& input, const ZQ_DImage<T>& mask, const ZQ_DImage<T>& mask_hat, ZQ_DImage<T>& output, const ZQ_InpaintingOptions& opt)
	{
		int winWidth = opt.winWidth;
		int winHeight = opt.winHeight;
		bool display = opt.display;

		float probe = opt.probe;
		probe = __max(0,__min(1,probe));


		int width = input.width();
		int height = input.height();
		int nChannels = input.nchannels();


		const T*& mask_ptr = mask.data();
		const T*& mask_hat_ptr = mask_hat.data();
		const T*& input_ptr = input.data();
		T*& output_ptr = output.data();

		float gradWeight = opt.gradWeight;
		/*compute dx dy Begin*/
		ZQ_DImage<T> input_dx(width,height,nChannels),input_dy(width,height,nChannels);
		T*& input_dx_ptr = input_dx.data();
		T*& input_dy_ptr = input_dy.data();
		for(int h = 1;h < height;h++)
		{
			for(int w = 0;w < width;w++)
			{
				for(int c = 0;c < nChannels;c++)
					input_dy_ptr[(h*width+w)*nChannels+c] = input_ptr[(h*width+w)*nChannels+c] - input_ptr[((h-1)*width+w)*nChannels+c];
			}
		}
		for(int h = 0;h < height;h++)
		{
			for(int w = 1;w < width;w++)
			{
				for(int c = 0;c < nChannels;c++)
					input_dx_ptr[(h*width+w)*nChannels+c] = input_ptr[(h*width+w)*nChannels+c] - input_ptr[(h*width+w-1)*nChannels+c];
			}
		}
		/*compute dx dy End*/

		int size_w = 2*winWidth+1;
		int size_h = 2*winHeight+1;

		int count = 0;

		std::vector<int> search_x;
		std::vector<int> search_y;


		for(int hh = winHeight;hh < height-winHeight;hh++)
		{
			for(int ww = winWidth;ww < width-winWidth;ww++)
			{
				if(mask_hat_ptr[hh*width+ww] < 0.5)
				{
					search_x.push_back(ww);
					search_y.push_back(hh);
				}
			}
		}

		int total_search_count = search_x.size();
		int search_count = probe*total_search_count;

		for(int h = 0;h < height;h++)
		{
			for(int w = 0;w < width;w++)
			{
				if(mask_ptr[h*width+w] > 0.5)
				{

					if(count % 100 == 0)
					{
						if(display)
							printf("count = %d\n",count);
					}
					count++;

					ZQ_DImage<T> local_mask(size_w,size_h);
					T*& local_mask_ptr = local_mask.data();

					/*cal local mask Begin*/
					for(int yy = -winHeight;yy <= winHeight;yy++)
					{
						for(int xx = -winWidth;xx <= winWidth;xx++)
						{
							local_mask_ptr[(yy+winHeight)*size_w+(xx+winWidth)] = 
								(h+yy >= 0) && (h+yy < height) && (w+xx >= 0) && (w+xx < width);				
						}
					}


					/*cal local mask End*/

					bool has_result = false;
					int result_h = 0, result_w = 0; 
					float L2_error = 0;
					

					for(int ss = 0;ss < search_count;ss++)
					{
						int cur_search_x, cur_search_y;
						if(probe < 1)
						{
							int cur_idx = rand()%total_search_count;
							cur_search_x = search_x[cur_idx];
							cur_search_y = search_y[cur_idx];
						}
						else
						{
							cur_search_x = search_x[ss];
							cur_search_y = search_y[ss];
						}

						/*calculate tmp L2 error Begin*/
						float tmp_L2_error = 0;


						for(int yy = -winHeight;yy <= winHeight;yy++)
						{
							for(int xx = -winWidth;xx <= winWidth;xx++)
							{
								if(local_mask_ptr[(winHeight+yy)*size_w+(winWidth+xx)] > 0.5)
								{
									for(int c = 0;c < nChannels;c++)
									{
										float tmp_val_channel = input_ptr[((cur_search_y+yy)*width+(cur_search_x+xx))*nChannels+c] 
										- output_ptr[((h+yy)*width+(w+xx))*nChannels+c];
										tmp_L2_error += tmp_val_channel*tmp_val_channel;

										if(gradWeight > 0)
										{
											float tmp_dx = w+xx-1 < 0 ? 0 : (output_ptr[((h+yy)*width+(w+xx))*nChannels+c]
											- output_ptr[((h+yy)*size_w+(w+xx-1))*nChannels+c]);
											float tmp_dy = h+yy-1 < 0 ? 0 : (output_ptr[((h+yy)*width+(w+xx))*nChannels+c]
											- output_ptr[((h+yy-1)*width+(w+xx))*nChannels+c]);

											tmp_val_channel = input_dx_ptr[((cur_search_y+yy)*width+(cur_search_x+xx))*nChannels+c] - tmp_dx;
											tmp_L2_error += tmp_val_channel*tmp_val_channel;
											tmp_val_channel = input_dy_ptr[((cur_search_y+yy)*width+(cur_search_x+xx))*nChannels+c] - tmp_dy;
											tmp_L2_error += tmp_val_channel*tmp_val_channel;
										}

									}
								}
							}
						}

						/*calculate tmp L2 error End*/
						if(!has_result)
						{

							result_h = cur_search_y;
							result_w = cur_search_x;
							L2_error = tmp_L2_error;
							has_result = true;
						}
						else
						{
							if(L2_error > tmp_L2_error)
							{
								L2_error = tmp_L2_error;

								result_h = cur_search_y;
								result_w = cur_search_x;

							}

						}
					}

					/*search the best value for each hole pixel End*/
					for(int c = 0;c < nChannels;c++)
						output_ptr[(h*width+w)*nChannels+c] = input_ptr[(result_h*width+result_w)*nChannels+c];
				}
				/* if mask end */
			}

		}
		/*for each hole pixel*/
	}
}

#endif