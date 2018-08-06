#ifndef _ZQ_BLEND_TWO_IMAGES_H_
#define _ZQ_BLEND_TWO_IMAGES_H_
#pragma once

#include "ZQ_ScatteredInterpolationRBF.h"
#include "ZQ_ImageProcessing.h"
#include "ZQ_WeightedMedian.h"

namespace ZQ
{
	class ZQ_BlendTwoImages
	{
	private:
		template<class T>
		static void _warpAndBlend(const int width, const int height, const int nChannels, const T* image1, const T* image2, const T* vels, const float weight1, T* out_image, int sample_mode, int blend_mode)
		{
			T* sample_result = new T[nChannels];

			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					int offset = i*width + j;

					if (sample_mode == 0)
					{
						float coord1_x = vels[offset * 4 + 0] + j;
						float coord1_y = vels[offset * 4 + 1] + i;
						float coord2_x = vels[offset * 4 + 2] + j;
						float coord2_y = vels[offset * 4 + 3] + i;
						if (blend_mode == 0)
						{
							ZQ_ImageProcessing::BilinearInterpolate(image1, width, height, nChannels, coord1_x, coord1_y, sample_result, false);
							for (int c = 0; c < nChannels; c++)
								out_image[offset*nChannels + c] = sample_result[c] * weight1;
							ZQ_ImageProcessing::BilinearInterpolate(image2, width, height, nChannels, coord2_x, coord2_y, sample_result, false);
							for (int c = 0; c < nChannels; c++)
								out_image[offset*nChannels + c] += sample_result[c] * (1 - weight1);
						}
						else if (blend_mode == 1)
						{
							ZQ_ImageProcessing::BilinearInterpolate(image1, width, height, nChannels, coord1_x, coord1_y, sample_result, false);
							for (int c = 0; c < nChannels; c++)
								out_image[offset*nChannels + c] = sample_result[c];
						}
						else
						{
							ZQ_ImageProcessing::BilinearInterpolate(image2, width, height, nChannels, coord2_x, coord2_y, sample_result, false);
							for (int c = 0; c < nChannels; c++)
								out_image[offset*nChannels + c] = sample_result[c];
						}
					}
					else if (sample_mode == 1)
					{
						float coord1_x = vels[offset * 4 + 0] + j;
						float coord1_y = vels[offset * 4 + 1] + i;
						float coord2_x = vels[offset * 4 + 2] + j;
						float coord2_y = vels[offset * 4 + 3] + i;
						if (blend_mode == 0)
						{
							ZQ_ImageProcessing::BicubicInterpolate(image1, width, height, nChannels, coord1_x, coord1_y, sample_result, false);
							for (int c = 0; c < nChannels; c++)
								out_image[offset*nChannels + c] = sample_result[c] * weight1;
							ZQ_ImageProcessing::BicubicInterpolate(image2, width, height, nChannels, coord2_x, coord2_y, sample_result, false);
							for (int c = 0; c < nChannels; c++)
								out_image[offset*nChannels + c] += sample_result[c] * (1 - weight1);
						}
						else if (blend_mode == 1)
						{
							ZQ_ImageProcessing::BicubicInterpolate(image1, width, height, nChannels, coord1_x, coord1_y, sample_result, false);
							for (int c = 0; c < nChannels; c++)
								out_image[offset*nChannels + c] = sample_result[c];
						}
						else
						{
							ZQ_ImageProcessing::BicubicInterpolate(image2, width, height, nChannels, coord2_x, coord2_y, sample_result, false);
							for (int c = 0; c < nChannels; c++)
								out_image[offset*nChannels + c] = sample_result[c];
						}
					}
					else
					{
						int coord1_x = vels[offset * 4 + 0] + j + 0.5;
						int coord1_y = vels[offset * 4 + 1] + i + 0.5;
						int coord2_x = vels[offset * 4 + 2] + j + 0.5;
						int coord2_y = vels[offset * 4 + 3] + i + 0.5;
						coord1_x == __min(width - 1, __max(0, coord1_x));
						coord1_y == __min(height - 1, __max(0, coord1_y));
						coord2_x == __min(width - 1, __max(0, coord2_x));
						coord2_y == __min(height - 1, __max(0, coord2_y));
						if (blend_mode == 0)
						{
							int sample_offset = coord1_y*width + coord1_x;
							for (int c = 0; c < nChannels; c++)
								out_image[offset*nChannels + c] = image1[sample_offset*nChannels + c] * weight1;
							sample_offset = coord2_y*width + coord2_x;
							for (int c = 0; c < nChannels; c++)
								out_image[offset*nChannels + c] += image2[sample_offset*nChannels + c] * (1 - weight1);
						}
						else if (blend_mode == 1)
						{
							int sample_offset = coord1_y*width + coord1_x;
							for (int c = 0; c < nChannels; c++)
								out_image[offset*nChannels + c] = image1[sample_offset*nChannels + c];
						}
						else
						{
							int sample_offset = coord2_y*width + coord2_x;
							for (int c = 0; c < nChannels; c++)
								out_image[offset*nChannels + c] = image2[sample_offset*nChannels + c];
						}
					}
				}
			}

			delete[]sample_result;
		}

	public:
		template<class T>
		static bool BlendTwoImages(const int width, const int height, const int nChannels, const T* image1, const T* image2, 
			const T* u, const T* v, const float weight1, T* out_image, int skip, int max_iter = 100, int num_of_neighbor = 5, double radius_scale = 1.5, int sample_mode = 0, const int blend_mode = 0)
		{
			if(image1 == 0 || image2 == 0 || u == 0 || v == 0 || out_image == 0)
				return false;

			memset(out_image,0,sizeof(T)*width*height*nChannels);


			ZQ_ScatteredInterpolationRBF<T> scatter;

			int nPixels = width*height;
			T* coord_for_img = new T[nPixels*2];
			T* vals_four_channels = new T[nPixels*4];

			memset(coord_for_img,0,sizeof(T)*nPixels*2);
			memset(vals_four_channels,0,sizeof(T)*nPixels*4);

			if(skip < 1) skip = 1;
			if(skip > 8) skip = 8;
			//int skip = 4;
			int nPts = 0;

			for(int i = 0;i < height;i+=skip)
			{
				for(int j = 0;j < width;j+=skip)
				{
					int offset = i*width+j;

					T cur_u = u[offset];
					T cur_v = v[offset];

					coord_for_img[nPts * 2 + 0] = j + cur_u*(1 - weight1);
					coord_for_img[nPts * 2 + 1] = i + cur_v*(1 - weight1);

					vals_four_channels[nPts * 4 + 0] = cur_u*(weight1 - 1);
					vals_four_channels[nPts * 4 + 1] = cur_v*(weight1 - 1);
					vals_four_channels[nPts * 4 + 2] = cur_u*weight1;
					vals_four_channels[nPts * 4 + 3] = cur_v*weight1;

					nPts++;
				}
			}

			T* interpolated_four_channels = new T[nPixels*4];
			memset(interpolated_four_channels,0,sizeof(T)*nPixels*4);

			scatter.SetLandmarks(nPts,2,coord_for_img,vals_four_channels,4);
			scatter.SolveCoefficient(num_of_neighbor,radius_scale,max_iter,ZQ_RBFKernel::COMPACT_CPC2);
			scatter.GridData2D(width,height,0,width-1,0,height-1,interpolated_four_channels);

			_warpAndBlend(width, height, nChannels, image1, image2, interpolated_four_channels, weight1, out_image, sample_mode, blend_mode);
			
			delete []vals_four_channels;
			delete []interpolated_four_channels;
			delete []coord_for_img;

			return true;
		}

		template<class T>
		static bool BlendTwoImagesByMedFilt(const int width, const int height, const int nChannels, const T* image1, const T* image2,
			const T* u, const T* v, const float weight1, T* out_image, int sample_mode = 0, int blend_mode = 0)
		{
			T* vels = new T[width*height * 4];
			bool* keep_mask = new bool[width*height];
			memset(vels, 0, sizeof(T)*width*height * 4);
			memset(keep_mask, 0, sizeof(bool)*width*height);

			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					int offset = h*width + w;
					T cur_u = u[offset] * (1.0 - weight1);
					T cur_v = v[offset] * (1.0 - weight1);
					T back_u = u[offset] * weight1;
					T back_v = v[offset] * weight1;
					int coord_x = w + cur_u + 0.5;
					int coord_y = h + cur_v + 0.5;
					if (coord_x >= 0 && coord_x <= width - 1 && coord_y >= 0 && coord_y <= height - 1)
					{
						int cur_offset = coord_y*width + coord_x;
						keep_mask[cur_offset] = true;
						vels[cur_offset * 4 + 0] = -cur_u;
						vels[cur_offset * 4 + 1] = -cur_v;
						vels[cur_offset * 4 + 2] = back_u;
						vels[cur_offset * 4 + 3] = back_v;
					}
				}
			}

			T* vels_filt = new T[width*height*4];
			ZQ_ImageProcessing::MedianFilterWithMask(vels, vels_filt, width, height, 4, 2, keep_mask, false);
		
			delete[]vels;
			delete[]keep_mask;

			_warpAndBlend<T>(width, height, nChannels, image1, image2, vels_filt, weight1, out_image, sample_mode, blend_mode);

		
			delete[]vels_filt;
			return true;
		}

		template<class T>
		static bool BlendTwoImagesByMedFiltWithMask(const int width, const int height, const int nChannels, const T* image1, const T* image2,
			const T* u, const T* v, const float weight1, T* out_image, bool* out_mask, int sample_mode = 0, int blend_mode = 0, float thresh_ratio = 0.5)
		{
			T* vels = new T[width*height * 4];
			bool* keep_mask = new bool[width*height];
			memset(vels, 0, sizeof(T)*width*height * 4);
			memset(keep_mask, 0, sizeof(bool)*width*height);
			

			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					int offset = h*width + w;
					T cur_u = u[offset] * (1.0 - weight1);
					T cur_v = v[offset] * (1.0 - weight1);
					T back_u = u[offset] * weight1;
					T back_v = v[offset] * weight1;
					int coord_x = w + cur_u + 0.5;
					int coord_y = h + cur_v + 0.5;
					if (coord_x >= 0 && coord_x <= width - 1 && coord_y >= 0 && coord_y <= height - 1)
					{
						int cur_offset = coord_y*width + coord_x;
						keep_mask[cur_offset] = true;
						vels[cur_offset * 4 + 0] = -cur_u;
						vels[cur_offset * 4 + 1] = -cur_v;
						vels[cur_offset * 4 + 2] = back_u;
						vels[cur_offset * 4 + 3] = back_v;
					}
				}
			}

			T* vels_filt = new T[width*height * 4];
			ZQ_ImageProcessing::MedianFilterWithMask(vels, vels_filt, out_mask, width, height, 4, 2, keep_mask,thresh_ratio);

			delete[]vels;
			delete[]keep_mask;

			_warpAndBlend<T>(width, height, nChannels, image1, image2, vels_filt, weight1, out_image, sample_mode, blend_mode);

			delete[]vels_filt;
			return true;
		}

		template<class T>
		static bool BlendTwoImagesByMedFiltWithMask(const int width, const int height, const int nChannels, const T* image1, const T* image2,
			const T* u, const T* v, const T* confidence, const float weight1, T* out_image, bool* out_mask, int sample_mode = 0, int blend_mode = 0, float thresh_ratio = 0.5)
		{
			T* vels = new T[width*height * 4];
			bool* visited_mask = new bool[width*height];
			memset(vels, 0, sizeof(T)*width*height * 4);
			memset(visited_mask, 0, sizeof(bool)*width*height);
			T* tmp_confidence = new T[width*height];
			memset(tmp_confidence, 0, sizeof(T)*width*height);


			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					int offset = h*width + w;
					T cur_u = u[offset] * (1.0 - weight1);
					T cur_v = v[offset] * (1.0 - weight1);
					T back_u = u[offset] * weight1;
					T back_v = v[offset] * weight1;
					int coord_x = w + cur_u + 0.5;
					int coord_y = h + cur_v + 0.5;
					if (coord_x >= 0 && coord_x <= width - 1 && coord_y >= 0 && coord_y <= height - 1)
					{
						int cur_offset = coord_y*width + coord_x;
						if (visited_mask[cur_offset])
						{
							T cur_confidence = confidence[offset];
							if (cur_confidence > tmp_confidence[cur_offset])
							{
								visited_mask[cur_offset] = true;
								tmp_confidence[cur_offset] = cur_confidence;
								vels[cur_offset * 4 + 0] = -cur_u;
								vels[cur_offset * 4 + 1] = -cur_v;
								vels[cur_offset * 4 + 2] = back_u;
								vels[cur_offset * 4 + 3] = back_v;
							}
						}
						else
						{
							visited_mask[cur_offset] = true;
							tmp_confidence[cur_offset] = confidence[offset];
							vels[cur_offset * 4 + 0] = -cur_u;
							vels[cur_offset * 4 + 1] = -cur_v;
							vels[cur_offset * 4 + 2] = back_u;
							vels[cur_offset * 4 + 3] = back_v;
						}
					}
				}
			}

			T* vels_filt = new T[width*height * 4];
			int fsize = 2;
			int XWIDTH = 2 * fsize + 1;
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					T* tmp_weights = new T[XWIDTH*XWIDTH];
					T* tmp_vals = new T[XWIDTH*XWIDTH];	
					int start_hh = __max(0, h - fsize);
					int end_hh = __min(height - 1, h + fsize);
					int start_ww = __max(0, w - fsize);
					int end_ww = __min(width - 1, w + fsize);
					for (int c = 0; c < 4; c++)
					{
						int tmp_num = 0;
						for (int hh = start_hh; hh <= end_hh; hh++)
						{
							for (int ww = start_ww; ww <= end_ww; ww++)
							{
								if (visited_mask[hh*width + ww])
								{
									tmp_weights[tmp_num] = tmp_confidence[hh*width + ww];
									tmp_vals[tmp_num] = vels[(hh*width + ww) * 4 + c];
									tmp_num++;
								}
							}
						}
						if (tmp_num >= 1)
						{
							T out_val;
							ZQ_WeightedMedian::FindMedian<T>(tmp_vals, tmp_weights, tmp_num, out_val);
							vels_filt[(h*width + w) * 4 + c] = out_val;
						}
						out_mask[h*width + w] = tmp_num >= thresh_ratio*XWIDTH*XWIDTH;
						
					}
					delete[]tmp_weights;
					delete[]tmp_vals;
				}
			}

			delete[]vels;
			delete[]visited_mask;
			delete[]tmp_confidence;

			_warpAndBlend<T>(width, height, nChannels, image1, image2, vels_filt, weight1, out_image, sample_mode, blend_mode);

			delete[]vels_filt;
			return true;
		}

	};
}


#endif