#ifndef _ZQ_IMAGE_PROCESSING_H_
#define _ZQ_IMAGE_PROCESSING_H_
#pragma once

#ifdef ZQLIB_USE_OPENMP
#include <omp.h>
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "ZQ_CubicInterpolation.h"
#include "ZQ_QuickSort.h"


namespace ZQ
{
	namespace ZQ_ImageProcessing
	{
		template<class T>
		T Median_value(T a, T b, T c);

		template<class T>
		void Sort_decend_3elements(T values[3]);

		/*basic functions*/
		template<class T>
		T EnforceRange(const T& x,const int& MaxValue);
	

		/*bilinear interpolation*/
		template<class T>
		void BilinearInterpolate(const T* pImage,const int width,const int height, const int nChannels, const float x, const float y, T* result, bool use_period_coord);

		/*bicubic interpolation*/
		template<class T>
		void BicubicInterpolate(const T* pImage, const int width, const int height, const int nChannels, const float x, const float y, T* result, bool use_period_coord);

		template<class T>
		T ImageMinValue(const T* pImage, const int width, const int height, const int nChannels, bool use_omp);

		template<class T>
		T ImageMaxValue(const T* pImage, const int width, const int height, const int nChannels, bool use_omp);

		template<class T>
		void ImageClamp(T* pImage, const T min_val, const T max_val, const int width, const int height, const int nChannels, bool use_omp);

		/*resize image*/
		template<class T>
		void ResizeImage(const T* pSrcImage, T* pDstImage, const int srcWidth, const int srcHeight, const int nChannels, const int dstWidth,const int dstHeight, bool use_omp);

		template<class T>
		void ResizeImage_NN(const T* pSrcImage, T* pDstImage, const int srcWidth, const int srcHeight, const int nChannels, const int dstWidth, const int dstHeight, bool use_omp);

		template<class T>
		void ResizeImageBicubic(const T* pSrcImage, T* pDstImage, const int srcWidth, const int srcHeight, const int nChannels, const int dstWidth, const int dstHeight, bool use_omp);

		/*resize a flow field on MAC grid
		* U : (width+1)*height,
		* V : width*(height+1) 
		*/
		template<class T>
		void ResizeFlow(const T* pSrcU, const T* pSrcV, T* pDstU, T* pDstV, const int srcWidth, const int srcHeight, const int dstWidth, const int dstHeight, bool use_omp);

		template<class T>
		void ResizeFlowBicubic(const T* pSrcU, const T* pSrcV, T* pDstU, T* pDstV, const int srcWidth, const int srcHeight, const int dstWidth, const int dstHeight, bool use_omp);

		template<class ImageType1, class ImageType2, class FilterType>
		void ImageFilter2D(const ImageType1* pSrcImage, ImageType2* pDstImage, const int width, const int height, const int nChannels, const FilterType* pfilter2D, const int xfsize, const int yfsize, bool use_omp);

		/*filter, x dimension*/
		template<class ImageType1, class ImageType2, class FilterType>
		void Hfiltering(const ImageType1* pSrcImage, ImageType2* pDstImage, const int width, const int height, const int nChannels, const FilterType* pfilter1D, const int fsize, bool use_omp);

		/*filter, y dimension*/
		template<class ImageType1, class ImageType2, class FilterType>
		void Vfiltering(const ImageType1* pSrcImage, ImageType2* pDstImage, const int width, const int height, const int nChannels, const FilterType* pfilter1D, const int fsize, bool use_omp);

		template<class T>
		void Laplacian(const T* pSrcImage, T* pDstImage, const int width, const int height, const int nChannels, bool use_omp);

		/*warp image*/
		template<class T>
		void WarpImage(T* pWarpIm2, const T* pIm2, const T* pU, const T* pV, const int width, const int height, const int nChannels, const T* pIm1 /*= 0*/, bool use_omp);

		template<class T>
		void WarpImageBicubic(T* pWarpIm2, const T* pIm2, const T* pU, const T* pV, const int width, const int height, const int nChannels, const T* pIm1 /*= 0*/, bool use_omp);

		/*median filter, each channel is handled separately*/
		template<class T>
		bool MedianFilter(const T* pSrcImage, T* pDstImage, const int width, const int height, int nChannels, const int fsize, bool use_omp);

		/*median filter, each channel is handled separately*/
		template<class T>
		bool MedianFilterWithMask(const T* pSrcImage, T* pDstImage, const int width, const int height, int nChannels, const int fsize, const bool* keep_mask, bool use_omp);

		/*median filter, each channel is handled separately*/
		template<class T>
		bool MedianFilterWithMask(const T* pSrcImage, T* pDstImage, bool* dst_mask, int width, int height, int nChannels, const int fsize, const bool* keep_mask, float thresh_ratio /*= 0.5*/, bool use_omp);

		template<class T>
		bool Edge(const T* pSrcImage, bool* edge, const int width, const int height, const float scale_of_mean /*= 4.0*/, bool use_omp);

		/*********************************************************************************/
		/********************************** definitions **********************************/
		/*********************************************************************************/

		template<class T>
		inline T Median_value(T a, T b, T c)
		{
			return a > b ? (b > c ? b : (a > c ? c : a)) : (a > c ? a : (b > c ? c : a));
		}

		template<class T>
		void Sort_decend_3elements(T values[3])
		{
			if (values[0] < values[1])
			{
				T tmp = values[0]; values[0] = values[1]; values[1] = tmp;
			}

			if (values[1] < values[2])
			{
				T tmp = values[0]; values[0] = values[1]; values[1] = tmp;
			}

			if (values[0] < values[1])
			{
				T tmp = values[0]; values[0] = values[1]; values[1] = tmp;
			}
		}

		template<class T>
		inline T EnforceRange(const T& x,const int& MaxValue)
		{
			return __min(__max(x,0),MaxValue-1);
		}


		template<class T>
		void BilinearInterpolate(const T* pImage,const int width,const int height, const int nChannels, const float x, const float y, T* result, bool use_period_coord)
		{
			double* val = new double[nChannels];
			memset(val, 0, sizeof(double)*nChannels);
			
			if(!use_period_coord)
			{
				float fx = EnforceRange(x,width);
				float fy = EnforceRange(y,height);
				int ix = floor(fx);
				int iy = floor(fy);
				float sx = fx-ix;
				float sy = fy-iy;


				for(int i = 0;i <= 1;i++)
				{
					for(int j = 0;j <= 1;j++)
					{
						int u = EnforceRange(ix+j,width);
						int v = EnforceRange(iy+i,height);
						
						for(int c = 0;c < nChannels;c++)
							val[c] += fabs(1-j-sx)*fabs(1-i-sy)*pImage[(v*width+u)*nChannels+c];
					}
				}
			}
			else
			{
				float shift_x = floor(x/width)*width;
				float shift_y = floor(y/height)*height;
				float xxx = x - shift_x;
				float yyy = y - shift_y;

				int ix = floor(xxx);
				int iy = floor(yyy);
				float sx = xxx - ix;
				float sy = yyy - iy;
				for(int i = 0;i <= 1;i++)
				{
					for(int j = 0;j <= 1;j++)
					{
						int u = (ix+j)%width;
						int v = (iy+i)%height;
						for(int c = 0;c < nChannels;c++)
							val[c] += fabs(1-j-sx)*fabs(1-i-sy)*pImage[(v*width+u)*nChannels+c];
					}
				}
			}	
			for (int c = 0; c < nChannels; c++)
				result[c] = val[c];
			delete[]val;
		}



		template<class T>
		void BicubicInterpolate(const T* pImage, const int width, const int height, const int nChannels, const float x, const float y, T* result, bool use_period_coord)
		{
			double* val = new double[nChannels];
			memset(val, 0, sizeof(double)*nChannels);

			if(!use_period_coord)
			{
				int ix = floor(x);
				int iy = floor(y);
				float sx = x-ix;
				float sy = y-iy;

				T data[16] = {0};
				for(int c = 0;c < nChannels;c++)
				{
					for(int i = 0;i < 4;i++)
					{
						for(int j = 0;j < 4;j++)
						{
							int cur_x = EnforceRange(ix-1+j,width);
							int cur_y = EnforceRange(iy-1+i,height);
							data[i*4+j] = pImage[(cur_y*width+cur_x)*nChannels+c];
						}
					}
					val[c] = ZQ_BicubicInterpolate(data,sx,sy);
				}
			}
			else
			{
				float shift_x = floor(x/width)*width;
				float shift_y = floor(y/height)*height;
				float xxx = x - shift_x;
				float yyy = y - shift_y;

				int ix = floor(xxx);
				int iy = floor(yyy);
				float sx = xxx - ix;
				float sy = yyy - iy;

				T data[16] = {0};
				for(int c = 0;c < nChannels;c++)
				{
					for(int i = 0;i < 4;i++)
					{
						for(int j = 0;j < 4;j++)
						{
							int cur_x = (ix-1+j+width)%width;
							int cur_y = (iy-1+i+height)%height;
							data[i*4+j] = pImage[(cur_y*width+cur_x)*nChannels+c];
						}
					}
					val[c] = ZQ_BicubicInterpolate(data,sx,sy);
				}
			}
			for (int c = 0; c < nChannels; c++)
				result[c] = val[c];
			delete[]val;
		}

		template<class T>
		T ImageMinValue(const T* pImage, const int width, const int height, const int nChannels, bool use_omp)
		{
#ifdef ZQLIB_USE_OPENMP
			if (use_omp)
			{
				int nElements = width*height*nChannels;
				if (nElements <= 0)
					return 0;
				T min_v = pImage[0];
				int nthreads = omp_get_max_threads();
				//printf("nthreads = %d\n", nthreads);
				T* min_vals = new T[nthreads];
				for (int i = 0; i < nthreads; i++)
					min_vals[i] = min_v;
				int num_per_thread = (nElements + nthreads - 1) / nthreads;
#pragma omp parallel for schedule(static)
				for (int k = 0; k < nthreads; k++)
				{
					int start = k*num_per_thread;
					int end = __min(nElements, (k + 1)*num_per_thread);
					for (int i = start; i < end; i++)
						min_vals[k] = __min(min_vals[k], pImage[i]);
				}
				for (int i = 0; i < nthreads; i++)
					min_v = __min(min_v, min_vals[i]);
				delete[]min_vals;
				return min_v;
			}
			else
			{
#endif
				
				int nElements = width*height*nChannels;
				if (nElements <= 0)
					return 0;
				T min_v = pImage[0];
				for (int i = 1; i < nElements; i++)
					min_v = __min(min_v, pImage[i]);
				return min_v;
#ifdef ZQLIB_USE_OPENMP
			}
#endif
		}

		template<class T>
		T ImageMaxValue(const T* pImage, const int width, const int height, const int nChannels, bool use_omp)
		{
#ifdef ZQLIB_USE_OPENMP
			if (use_omp)
			{
				int nElements = width*height*nChannels;
				if (nElements <= 0)
					return 0;
				T max_v = pImage[0];
				int nthreads = omp_get_max_threads();
				//printf("nthreads = %d\n", nthreads);
				T* max_vals = new T[nthreads];
				for (int i = 0; i < nthreads; i++)
					max_vals[i] = max_v;
				int num_per_thread = (nElements + nthreads - 1) / nthreads;
#pragma omp parallel for schedule(static)
				for (int k = 0; k < nthreads; k++)
				{
					int start = k*num_per_thread;
					int end = __max(nElements, (k + 1)*num_per_thread);
					for (int i = start; i < end; i++)
						max_vals[k] = __max(max_vals[k], pImage[i]);
				}
				for (int i = 0; i < nthreads; i++)
					max_v = __max(max_v, max_vals[i]);
				delete[]max_vals;
				return max_v;
			}
			else
			{
#endif

				int nElements = width*height*nChannels;
				if (nElements <= 0)
					return 0;
				T max_v = pImage[0];
				for (int i = 1; i < nElements; i++)
					max_v = __max(max_v, pImage[i]);
				return max_v;
#ifdef ZQLIB_USE_OPENMP
			}
#endif
		}

		template<class T>
		void ImageClamp(T* pImage, const T min_val, const T max_val, const int width, const int height, const int nChannels, bool use_omp)
		{
#ifdef ZQLIB_USE_OPENMP
			if (use_omp)
			{
				int nElements = width*height*nChannels;
				int nthreads = omp_get_num_threads();
				int num_per_thread = (nElements + nthreads - 1) / nthreads;
#pragma omp parallel for schedule(static)
				for (int k = 0; k < nthreads; k++)
				{
					int start = k*num_per_thread;
					int end = __min(nElements, (k + 1)*num_per_thread);
					for (int i = start; i < end; i++)
						pImage[i] = __min(max_val, __max(min_val, pImage[i]));
				}
			}
			else
			{
#endif
				int nElements = width*height*nChannels;
				for (int i = 0; i < nElements; i++)
					pImage[i] = __min(max_val, __max(min_val, pImage[i]));
#ifdef ZQLIB_USE_OPENMP
			}
#endif
		}

		template<class T>
		void ResizeImage(const T* pSrcImage, T* pDstImage, const int srcWidth, const int srcHeight, const int nChannels, const int dstWidth, const int dstHeight, bool use_omp)
		{
			memset(pDstImage, 0, sizeof(T)*dstWidth*dstHeight*nChannels);
#ifdef ZQLIB_USE_OPENMP
			if(use_omp)
			{
				int nthreads = omp_get_num_threads();
				int nPixels = dstWidth*dstHeight;
#pragma omp parallel for schedule(dynamic, (nPixels+nthreads-1)/nthreads)
				for (int pp = 0; pp < nPixels; pp++)
				{
					int i = pp / dstWidth;
					int j = pp % dstWidth;
					float coordx = (j + 0.5) / dstWidth*srcWidth - 0.5;
					float coordy = (i + 0.5) / dstHeight*srcHeight - 0.5;

					BilinearInterpolate(pSrcImage, srcWidth, srcHeight, nChannels, coordx, coordy, pDstImage + (i*dstWidth + j)*nChannels, false);
				}
			}
			else
			{
#endif
				for (int i = 0; i < dstHeight; i++)
				{
					for (int j = 0; j < dstWidth; j++)
					{
						float coordx = (j + 0.5) / dstWidth*srcWidth - 0.5;
						float coordy = (i + 0.5) / dstHeight*srcHeight - 0.5;

						BilinearInterpolate(pSrcImage, srcWidth, srcHeight, nChannels, coordx, coordy, pDstImage + (i*dstWidth + j)*nChannels, false);
					}
				}
#ifdef ZQLIB_USE_OPENMP
			}
#endif
		}

		template<class T>
		void ResizeImage_NN(const T* pSrcImage, T* pDstImage, const int srcWidth, const int srcHeight, const int nChannels, const int dstWidth, const int dstHeight, bool use_omp)
		{
#ifdef ZQLIB_USE_OPENMP
			if (use_omp)
			{
				int nthreads = omp_get_num_threads();
				int nPixels = dstHeight*dstWidth;
#pragma omp parallel for schedule(dynamic, (nPixels+nthreads-1)/nthreads)
				for (int pp = 0; pp < nPixels; pp++)
				{
					int i = pp / dstWidth;
					int j = pp % dstWidth;
					float coordx = (j + 0.5) / dstWidth*srcWidth - 0.5;
					float coordy = (i + 0.5) / dstHeight*srcHeight - 0.5;
					int ix = coordx + 0.5;
					int iy = coordy + 0.5;
					ix = __max(0, __min(ix, srcWidth - 1));
					iy = __max(0, __min(iy, srcHeight - 1));
					for (int c = 0; c < nChannels; c++)
						pDstImage[(i*dstWidth + j)*nChannels + c] = pSrcImage[(iy*srcWidth + ix)*nChannels + c];
				}
			}
			else
			{
#endif
				for (int i = 0; i < dstHeight; i++)
				{
					for (int j = 0; j < dstWidth; j++)
					{
						float coordx = (j + 0.5) / dstWidth*srcWidth - 0.5;
						float coordy = (i + 0.5) / dstHeight*srcHeight - 0.5;
						int ix = coordx + 0.5;
						int iy = coordy + 0.5;
						ix = __max(0, __min(ix, srcWidth - 1));
						iy = __max(0, __min(iy, srcHeight - 1));
						for (int c = 0; c < nChannels; c++)
							pDstImage[(i*dstWidth + j)*nChannels + c] = pSrcImage[(iy*srcWidth + ix)*nChannels + c];
					}
				}
#ifdef ZQLIB_USE_OPENMP
			}
#endif
		}


		template<class T>
		void ResizeImageBicubic(const T* pSrcImage, T* pDstImage, const int srcWidth, const int srcHeight, const int nChannels, const int dstWidth, const int dstHeight, bool use_omp)
		{
			memset(pDstImage,0,sizeof(T)*dstWidth*dstHeight*nChannels);

#ifdef ZQLIB_USE_OPENMP
			if(use_omp)
			{
				int nthreads = omp_get_num_threads();
				int nPixels = dstWidth*dstHeight;
#pragma omp parallel for schedule(dynamic, (nPixels+nthreads-1)/nthreads)
				for(int pp = 0;pp < nPixels;pp++)
				{
					int i = pp / dstWidth;
					int j = pp % dstWidth;
					T tmpData[16] = { 0 };

					float coordx = (j + 0.5) / dstWidth*srcWidth - 0.5;
					float coordy = (i + 0.5) / dstHeight*srcHeight - 0.5;

					int ix = floor(coordx);
					int iy = floor(coordy);
					float fx = coordx - ix;
					float fy = coordy - iy;


					for (int c = 0; c < nChannels; c++)
					{
						for (int s = 0; s < 4; s++)
						{
							for (int t = 0; t < 4; t++)
							{
								int tmpx = EnforceRange(ix - 1 + t, srcWidth);
								int tmpy = EnforceRange(iy - 1 + s, srcHeight);

								tmpData[s * 4 + t] = pSrcImage[(tmpy*srcWidth + tmpx)*nChannels + c];
							}
						}
						pDstImage[(i*dstWidth + j)*nChannels + c] = ZQ_BicubicInterpolate(tmpData, fx, fy);
					}
				}
			}
			else
			{
#endif
				for (int i = 0; i < dstHeight; i++)
				{
					T tmpData[16] = { 0 };
					for (int j = 0; j < dstWidth; j++)
					{
						float coordx = (j + 0.5) / dstWidth*srcWidth - 0.5;
						float coordy = (i + 0.5) / dstHeight*srcHeight - 0.5;

						int ix = floor(coordx);
						int iy = floor(coordy);
						float fx = coordx - ix;
						float fy = coordy - iy;


						for (int c = 0; c < nChannels; c++)
						{
							for (int s = 0; s < 4; s++)
							{
								for (int t = 0; t < 4; t++)
								{
									int tmpx = EnforceRange(ix - 1 + t, srcWidth);
									int tmpy = EnforceRange(iy - 1 + s, srcHeight);

									tmpData[s * 4 + t] = pSrcImage[(tmpy*srcWidth + tmpx)*nChannels + c];
								}
							}
							pDstImage[(i*dstWidth + j)*nChannels + c] = ZQ_BicubicInterpolate(tmpData, fx, fy);
						}
					}
				}
#ifdef ZQLIB_USE_OPENMP
			}
#endif
		}


		template<class T>
		void ResizeFlow(const T* pSrcU, const T* pSrcV, T* pDstU, T* pDstV, const int srcWidth, const int srcHeight, const int dstWidth, const int dstHeight, bool use_omp)
		{
			memset(pDstU, 0, sizeof(T)*(dstWidth + 1)*dstHeight);
			memset(pDstV, 0, sizeof(T)*dstWidth*(dstHeight + 1));

			//resize U
#ifdef ZQLIB_USE_OPENMP
			if (use_omp)
			{
				int nthreads = omp_get_num_threads();
				int nPixels = dstHeight*(dstWidth+1);
#pragma omp parallel for schedule(dynamic, (nPixels+nthreads-1)/nthreads)
				for(int pp = 0;pp < nPixels;pp++)
				{
					int i = pp / dstWidth;
					int j = pp % dstWidth;

					float coordx = (float)j / dstWidth*srcWidth;
					float coordy = (i + 0.5f) / dstHeight*srcHeight - 0.5f;

					BilinearInterpolate(pSrcU, srcWidth + 1, srcHeight, 1, coordx, coordy, pDstU + i*(dstWidth + 1) + j, false);
				}
			}
			else
			{
#endif
				for (int i = 0; i < dstHeight; i++)
				{
					for (int j = 0; j <= dstWidth; j++)
					{
						float coordx = (float)j / dstWidth*srcWidth;
						float coordy = (i + 0.5f) / dstHeight*srcHeight - 0.5f;

						BilinearInterpolate(pSrcU, srcWidth + 1, srcHeight, 1, coordx, coordy, pDstU + i*(dstWidth + 1) + j, false);
					}
				}
#ifdef ZQLIB_USE_OPENMP
			}

			//resize V
			if (use_omp)
			{
				int nthreads = omp_get_num_threads();
				int pp_num = (dstHeight+1)*dstWidth;
#pragma omp parallel for schedule(dynamic, (pp_num+nthreads-1)/nthreads)
				for (int pp = 0; pp < pp_num; pp++)
				{
					int i = pp / dstWidth;
					int j = pp % dstWidth;

					float coordx = (j + 0.5f) / dstWidth*srcWidth - 0.5f;
					float coordy = (float)i / dstHeight*srcHeight;

					BilinearInterpolate(pSrcV, srcWidth, srcHeight + 1, 1, coordx, coordy, pDstV + i*dstWidth + j, false);
				}
			}
			else
			{
#endif
				//resize V
				for (int i = 0; i <= dstHeight; i++)
				{
					for (int j = 0; j < dstWidth; j++)
					{
						float coordx = (j + 0.5f) / dstWidth*srcWidth - 0.5f;
						float coordy = (float)i / dstHeight*srcHeight;

						BilinearInterpolate(pSrcV, srcWidth, srcHeight + 1, 1, coordx, coordy, pDstV + i*dstWidth + j, false);
					}
				}
#ifdef ZQLIB_USE_OPENMP
			}
#endif
		}


		template<class T>
		void ResizeFlowBicubic(const T* pSrcU, const T* pSrcV, T* pDstU, T* pDstV, const int srcWidth, const int srcHeight, const int dstWidth, const int dstHeight, bool use_omp)
		{
			memset(pDstU,0,sizeof(T)*(dstWidth+1)*dstHeight);
			memset(pDstV,0,sizeof(T)*dstWidth*(dstHeight+1));

			

			//resize U
#ifdef ZQLIB_USE_OPENMP
			if(use_omp)
			{
				int nthreads = omp_get_num_threads();
				int pp_num = dstHeight*(dstWidth+1);
#pragma omp parallel for schedule(dynamic, (pp_num+nthreads-1)/nthreads)
				for(int pp = 0;pp < pp_num;p++)
				{
					int i = pp / (dstWidth+1);
					int j = pp % (dstWidth+1);
					T tmpData[16] = { 0 };

					float coordx = (float)j / dstWidth*srcWidth;
					float coordy = (i + 0.5f) / dstHeight*srcHeight - 0.5f;

					int ix = floor(coordx);
					int iy = floor(coordy);
					float fx = coordx - ix;
					float fy = coordy - iy;

					for (int s = 0; s < 4; s++)
					{
						for (int t = 0; t < 4; t++)
						{
							int tmpx = EnforceRange(ix - 1 + t, srcWidth + 1);
							int tmpy = EnforceRange(iy - 1 + s, srcHeight);

							tmpData[s * 4 + t] = pSrcU[tmpy*(srcWidth + 1) + tmpx];
						}
					}
					pDstU[i*(dstWidth + 1) + j] = ZQ_BicubicInterpolate(tmpData, fx, fy);
				}
			}
			else
			{
#endif
				for (int i = 0; i < dstHeight; i++)
				{
					T tmpData[16] = { 0 };
					for (int j = 0; j <= dstWidth; j++)
					{
						float coordx = (float)j / dstWidth*srcWidth;
						float coordy = (i + 0.5f) / dstHeight*srcHeight - 0.5f;

						int ix = floor(coordx);
						int iy = floor(coordy);
						float fx = coordx - ix;
						float fy = coordy - iy;

						for (int s = 0; s < 4; s++)
						{
							for (int t = 0; t < 4; t++)
							{
								int tmpx = EnforceRange(ix - 1 + t, srcWidth + 1);
								int tmpy = EnforceRange(iy - 1 + s, srcHeight);

								tmpData[s * 4 + t] = pSrcU[tmpy*(srcWidth + 1) + tmpx];
							}
						}
						pDstU[i*(dstWidth + 1) + j] = ZQ_BicubicInterpolate(tmpData, fx, fy);

					}
				}
#ifdef ZQLIB_USE_OPENMP
			}

			//resizeV
			if (use_omp)
			{
				int nthreads = omp_get_num_threads();
				int pp_num = (dstHeight+1)*dstWidth;
#pragma omp parallel for schedule(dynamic, (pp_num+nthreads-1)/nthreads)
				for(int pp = 0;pp < pp_num;pp++)
				{
					int i = pp / dstWidth;
					int j = pp % dstWidth;
					T tmpData[16] = { 0 };

					float coordx = (j + 0.5f) / dstWidth*srcWidth - 0.5f;
					float coordy = (float)i / dstHeight*srcHeight;
					int ix = floor(coordx);
					int iy = floor(coordy);
					float fx = coordx - ix;
					float fy = coordy - iy;

					for (int s = 0; s < 4; s++)
					{
						for (int t = 0; t < 4; t++)
						{
							int tmpx = EnforceRange(ix - 1 + t, srcWidth);
							int tmpy = EnforceRange(iy - 1 + s, srcHeight + 1);

							tmpData[s * 4 + t] = pSrcV[tmpy*srcWidth + tmpx];
						}
					}
					pDstV[i*dstWidth + j] = ZQ_BicubicInterpolate(tmpData, fx, fy);
				}
			}
			else
			{
#endif
				//resizeV
				for (int i = 0; i <= dstHeight; i++)
				{
					T tmpData[16] = { 0 };
					for (int j = 0; j < dstWidth; j++)
					{
						float coordx = (j + 0.5f) / dstWidth*srcWidth - 0.5f;
						float coordy = (float)i / dstHeight*srcHeight;
						int ix = floor(coordx);
						int iy = floor(coordy);
						float fx = coordx - ix;
						float fy = coordy - iy;

						for (int s = 0; s < 4; s++)
						{
							for (int t = 0; t < 4; t++)
							{
								int tmpx = EnforceRange(ix - 1 + t, srcWidth);
								int tmpy = EnforceRange(iy - 1 + s, srcHeight + 1);

								tmpData[s * 4 + t] = pSrcV[tmpy*srcWidth + tmpx];
							}
						}
						pDstV[i*dstWidth + j] = ZQ_BicubicInterpolate(tmpData, fx, fy);
					}
				}
#ifdef ZQLIB_USE_OPENMP
			}
#endif
		}

		template<class ImageType1, class ImageType2, class FilterType>
		void ImageFilter2D(const ImageType1* pSrcImage, ImageType2* pDstImage, const int width, const int height, const int nChannels, 
			const FilterType* pfilter2D, const int half_xfsize, const int half_yfsize, bool use_omp)
		{
			
			int XSIZE = 2 * half_xfsize + 1;
			for (int c = 0; c < nChannels; c++)
			{
#ifdef ZQLIB_USE_OPENMP
				if (use_omp)
				{
					int nthreads = omp_get_num_threads();
					int nPixels = height*width;
#pragma omp parallel for schedule(dynamic,(nPixels+nthreads-1)/nthreads)
					for(int pp = 0;pp < nPixels;pp++)
					{
						int i = pp / width;
						int j = pp / height;
						double tmp_val = 0;
						for (int xx = -half_xfsize; xx <= half_xfsize; xx++)
						{
							for (int yy = -half_yfsize; yy <= half_yfsize; yy++)
							{
								int ii = EnforceRange(i + yy, height);
								int jj = EnforceRange(j + xx, width);
								tmp_val += pSrcImage[(ii*width + jj)*nChannels + c] * pfilter2D[(yy + half_yfsize)*XSIZE + xx + half_xfsize];

							}
						}
						pDstImage[(i*width + j)*nChannels + c] = tmp_val;
					}
				}
				else
				{
#endif
					for (int i = 0; i < height; i++)
					{
						for (int j = 0; j < width; j++)
						{
							double tmp_val = 0;
							for (int xx = -half_xfsize; xx <= half_xfsize; xx++)
							{
								for (int yy = -half_yfsize; yy <= half_yfsize; yy++)
								{
									int ii = EnforceRange(i + yy, height);
									int jj = EnforceRange(j + xx, width);
									tmp_val += pSrcImage[(ii*width + jj)*nChannels + c] * pfilter2D[(yy + half_yfsize)*XSIZE + xx + half_xfsize];

								}
							}
							pDstImage[(i*width + j)*nChannels + c] = tmp_val;
						}
					}
#ifdef ZQLIB_USE_OPENMP
				}
#endif
			}
		}

		template<class ImageType1, class ImageType2, class FilterType>
		void ImageFilter2D_3x3_1channel(const ImageType1* pSrcImage, ImageType2* pDstImage, const int width, const int height, const FilterType* pfilter2D, bool use_omp)
		{
			//padding
			int padding_width = width + 2;
			int padding_height = height + 2;
			ImageType1* tmpImg = new ImageType1[padding_width*padding_height];
			for (int h = 1; h < height + 1; h++)
				memcpy(tmpImg + (h*padding_width + 1), pSrcImage + (h - 1)*width, sizeof(ImageType1)*width);
			memcpy(tmpImg, tmpImg + (padding_width + 1), sizeof(ImageType1)*width);
			memcpy(tmpImg + ((padding_height - 1)*padding_width + 1), tmpImg + ((padding_height - 2)*padding_width + 1), sizeof(ImageType1)*width);
			for (int h = 0; h < height + 2; h++)
			{
				memcpy(tmpImg + h*padding_width, tmpImg + (h*padding_width + 1), sizeof(ImageType1));
				memcpy(tmpImg + (h*padding_width + padding_width - 1), tmpImg + (h*padding_width + padding_width - 2), sizeof(ImageType1));
			}

			//
#ifdef ZQLIB_USE_OPENMP
			if (use_omp)
			{
				int nthreads = omp_get_num_threads();
				int nPixels = height*width;
#pragma omp parallel for schedule(dynamic, (nPixels+nthreads-1)/nthreads)
				for(int pp = 0;pp < nPixels;pp++)
				{
					int i = pp / width;
					int j = pp % width;

					double tmp_val = 0;
					for (int yy = 0; yy < 3; yy++)
					{
						for (int xx = 0; xx < 3; xx++)
						{
							int ii = i + yy;
							int jj = j + xx;
							tmp_val += tmpImg[ii*padding_width + jj] * pfilter2D[yy * 3 + xx];
						}
					}
					pDstImage[i*width + j] = tmp_val;
				}
			}
			else
			{
#endif
				for (int i = 0; i < height; i++)
				{
					for (int j = 0; j < width; j++)
					{
						double tmp_val = 0;
						for (int yy = 0; yy < 3; yy++)
						{
							for (int xx = 0; xx < 3; xx++)
							{
								int ii = i + yy;
								int jj = j + xx;
								tmp_val += tmpImg[ii*padding_width + jj] * pfilter2D[yy * 3 + xx];
							}
						}
						pDstImage[i*width + j] = tmp_val;
					}
				}
#ifdef ZQLIB_USE_OPENMP
			}
#endif
			delete[]tmpImg;
		}

		template<class ImageType1, class ImageType2, class FilterType>
		void Hfiltering(const ImageType1* pSrcImage, ImageType2* pDstImage, const int width, const int height, const int nChannels, const FilterType* pfilter1D, const int fsize, bool use_omp)
		{
			for (int c = 0; c < nChannels; c++)
			{
#ifdef ZQLIB_USE_OPENMP
				if (use_omp)
				{
					int nthreads = omp_get_num_threads();
					int nPixels = height*width;
#pragma omp parallel for schedule(dynamic, (nPixels+nthreads-1)/nthreads)
					for (int pp = 0; pp < nPixels; pp++)
					{
						int i = pp / width;
						int j = pp % width;

						double tmp_val = 0;
						for (int l = -fsize; l <= fsize; l++)
						{
							int jj = EnforceRange(j + l, width);
							tmp_val += pSrcImage[(i*width + jj)*nChannels + c] * pfilter1D[l + fsize];
						}
						pDstImage[(i*width + j)*nChannels + c] = tmp_val;
					}
				}
				else
				{
#endif
					for (int i = 0; i < height; i++)
					{
						for (int j = 0; j < width; j++)
						{
							double tmp_val = 0;
							for (int l = -fsize; l <= fsize; l++)
							{
								int jj = EnforceRange(j + l, width);
								tmp_val += pSrcImage[(i*width + jj)*nChannels + c] * pfilter1D[l + fsize];
							}
							pDstImage[(i*width + j)*nChannels + c] = tmp_val;
						}
					}
#ifdef ZQLIB_USE_OPENMP
				}
#endif
			}
			
		}


		template<class ImageType1, class ImageType2, class FilterType>
		void Vfiltering(const ImageType1* pSrcImage, ImageType2* pDstImage, const int width, const int height, const int nChannels, const FilterType* pfilter1D, const int fsize, bool use_omp)
		{
			for (int c = 0; c < nChannels; c++)
			{
#ifdef ZQLIB_USE_OPENMP
				if (use_omp)
				{
					int nthreads = omp_get_num_threads();
					int nPixels = height*width;
#pragma omp parallel for schedule(dynamic, (nPixels+nthreads-1)/nthreads)
					for (int pp = 0; pp < nPixels; pp++)
					{
						int i = pp / width;
						int j = pp % width;

						double tmp_val = 0;
						for (int l = -fsize; l <= fsize; l++)
						{
							int ii = EnforceRange(i + l, height);
							tmp_val += pSrcImage[(ii*width + j)*nChannels + c] * pfilter1D[l + fsize];
						}

						pDstImage[(i*width + j)*nChannels + c] = tmp_val;
					}
				}
				else
				{
#endif
					for (int i = 0; i < height; i++)
					{
						for (int j = 0; j < width; j++)
						{
							double tmp_val = 0;
							for (int l = -fsize; l <= fsize; l++)
							{
								int ii = EnforceRange(i + l, height);
								tmp_val += pSrcImage[(ii*width + j)*nChannels + c] * pfilter1D[l + fsize];
							}

							pDstImage[(i*width + j)*nChannels + c] = tmp_val;
						}
					}
#ifdef ZQLIB_USE_OPENMP
				}
#endif 
			}
		}


		template<class T>
		void Laplacian(const T* pSrcImage, T* pDstImage, const int width, const int height, const int nChannels, bool use_omp)
		{
			memset(pDstImage, 0, sizeof(T)*width*height*nChannels);
#ifdef ZQLIB_USE_OPENMP
			if(use_omp)
			{
				int nthreads = omp_get_num_threads();
#pragma omp parallel for schedule(dynamic, (height+nthreads-1)/nthreads)
				for (int i = 0; i < height; i++)
				{
					for (int j = 1; j < width - 1; j++)
					{
						for (int c = 0; c < nChannels; c++)
							pDstImage[(i*width + j)*nChannels + c] += pSrcImage[(i*width + j + 1)*nChannels + c] + pSrcImage[(i*width + j - 1)*nChannels + c];
					}
					for (int c = 0; c < nChannels; c++)
						pDstImage[(i*width + 0)*nChannels + c] += pSrcImage[(i*width + 0)*nChannels + c] + pSrcImage[(i*width + 1)*nChannels + c];
					for (int c = 0; c < nChannels; c++)
						pDstImage[(i*width + width - 1)*nChannels + c] += pSrcImage[(i*width + width - 2)*nChannels + c] + pSrcImage[(i*width + width - 1)*nChannels + c];
				}
			}
			else
			{
#endif
				for (int i = 0; i < height; i++)
				{
					for (int j = 1; j < width - 1; j++)
					{
						for (int c = 0; c < nChannels; c++)
							pDstImage[(i*width + j)*nChannels + c] += pSrcImage[(i*width + j + 1)*nChannels + c] + pSrcImage[(i*width + j - 1)*nChannels + c];
					}
					for (int c = 0; c < nChannels; c++)
						pDstImage[(i*width + 0)*nChannels + c] += pSrcImage[(i*width + 0)*nChannels + c] + pSrcImage[(i*width + 1)*nChannels + c];
					for (int c = 0; c < nChannels; c++)
						pDstImage[(i*width + width - 1)*nChannels + c] += pSrcImage[(i*width + width - 2)*nChannels + c] + pSrcImage[(i*width + width - 1)*nChannels + c];
				}
#ifdef ZQLIB_USE_OPENMP
			}

			if (use_omp)
			{
				int nthreads = omp_get_num_threads();
#pragma omp parallel for schedule(dynamic, (width+nthreads-1)/nthreads)
				for (int j = 0; j < width; j++)
				{
					for (int i = 1; i < height - 1; i++)
					{
						for (int c = 0; c < nChannels; c++)
							pDstImage[(i*width + j)*nChannels + c] += pSrcImage[((i + 1)*width + j)*nChannels + c] + pSrcImage[((i - 1)*width + j)*nChannels + c];
					}
					for (int c = 0; c < nChannels; c++)
						pDstImage[(0 * width + j)*nChannels + c] += pSrcImage[(0 * width + j)*nChannels + c] + pSrcImage[(1 * width + j)*nChannels + c];
					for (int c = 0; c < nChannels; c++)
						pDstImage[((height - 1)*width + j)*nChannels + c] += pSrcImage[((height - 2)*width + j)*nChannels + c] + pSrcImage[((height - 1)*width + j)*nChannels + c];
				}
			}
			else
			{
#endif
				for (int j = 0; j < width; j++)
				{
					for (int i = 1; i < height - 1; i++)
					{
						for (int c = 0; c < nChannels; c++)
							pDstImage[(i*width + j)*nChannels + c] += pSrcImage[((i + 1)*width + j)*nChannels + c] + pSrcImage[((i - 1)*width + j)*nChannels + c];
					}
					for(int c = 0;c < nChannels;c++)
						pDstImage[(0 * width + j)*nChannels + c] += pSrcImage[(0 * width + j)*nChannels + c] + pSrcImage[(1 * width + j)*nChannels + c];
					for (int c = 0; c < nChannels; c++)
						pDstImage[((height - 1)*width + j)*nChannels + c] += pSrcImage[((height - 2)*width + j)*nChannels + c] + pSrcImage[((height - 1)*width + j)*nChannels + c];
				}
#ifdef ZQLIB_USE_OPENMP
			}

			if (use_omp)
			{
				int nthreads = omp_get_num_threads();
				int nElements = width*height*nChannels;
#pragma omp parallel for schedule(dynamic, (nElements+nthreads-1)/nthreads)
				for (int i = 0; i < nElements; i++)
				{
					pDstImage[i] -= 4 * pSrcImage[i];
				}
			}
			else
			{
#endif
				for (int i = 0; i < height*width*nChannels; i++)
				{
					pDstImage[i] -= 4 * pSrcImage[i];
				}
#ifdef ZQLIB_USE_OPENMP
			}
#endif
		}


		template<class T>
		void WarpImage(T* pWarpIm2, const T* pIm2, const T* pU, const T* pV, const int width, const int height, const int nChannels, const T* pIm1 /* = 0 */, bool use_omp)
		{
			memset(pWarpIm2,0,sizeof(T)*width*height*nChannels);
#ifdef ZQLIB_USE_OPENMP
			if(use_omp)
			{
				int nthreads = omp_get_num_threads();
				int nPixels = width*height;
#pragma omp parallel for schedule(dynamic, (nPixels+nthreads-1)/nthreads)
				for (int pp = 0; pp < nPixels; pp++)
				{
					int i = pp / width;
					int j = pp % width;

					int offset = pp;
					float x = j + pU[offset];
					float y = i + pV[offset];
					if (x < 0 || x > width - 1 || y < 0 || y > height - 1)
					{
						if (pIm1)
						{
							for (int c = 0; c < nChannels; c++)
								pWarpIm2[offset*nChannels + c] = pIm1[offset*nChannels + c];
						}

						continue;
					}
					BilinearInterpolate(pIm2, width, height, nChannels, x, y, pWarpIm2 + offset*nChannels, false);
				}
			}
			else
			{
#endif
				for (int i = 0; i < height; i++)
				{
					for (int j = 0; j < width; j++)
					{
						int offset = i*width + j;
						float x = j + pU[offset];
						float y = i + pV[offset];
						if (x < 0 || x > width - 1 || y < 0 || y > height - 1)
						{
							if (pIm1)
							{
								for (int c = 0; c < nChannels; c++)
									pWarpIm2[offset*nChannels + c] = pIm1[offset*nChannels + c];
							}

							continue;
						}
						BilinearInterpolate(pIm2, width, height, nChannels, x, y, pWarpIm2 + offset*nChannels, false);
					}
				}
#ifdef ZQLIB_USE_OPENMP
			}
#endif
		}


		template<class T>
		void WarpImageBicubic(T* pWarpIm2, const T* pIm2, const T* pU, const T* pV, const int width, const int height, const int nChannels, const T* pIm1 /* = 0 */, bool use_omp)
		{
#ifdef ZQLIB_USE_OPENMP
			if (use_omp)
			{
				int nthreads = omp_get_num_threads();
				int nPixels = width*height;
#pragma omp parallel for schedule(dynamic, (nPixels+nthreads-1)/nthreads)
				for(int pp = 0;pp < nPixels;pp++)
				{
					int i = pp / width;
					int j = pp % width;
					T tmpData[16] = { 0 };
					int offset = pp;
					float x = j + pU[offset];
					float y = i + pV[offset];
					if (x < 0 || x > width - 1 || y < 0 || y > height - 1)
					{
						if (pIm1)
						{
							for (int c = 0; c < nChannels; c++)
								pWarpIm2[offset*nChannels + c] = pIm1[offset*nChannels + c];
						}

						continue;
					}

					int x0 = floor(x);
					int y0 = floor(y);
					float fx = x - x0;
					float fy = y - y0;

					for (int c = 0; c < nChannels; c++)
					{
						for (int s = 0; s < 4; s++)
						{
							for (int t = 0; t < 4; t++)
							{
								int tmpx = EnforceRange(x0 - 1 + t, width);
								int tmpy = EnforceRange(y0 - 1 + s, height);

								tmpData[s * 4 + t] = pIm2[(tmpy*width + tmpx)*nChannels + c];
							}
						}
						pWarpIm2[offset*nChannels + c] = ZQ_BicubicInterpolate(tmpData, fx, fy);
					}

				}
			}
			else
			{
#endif
				for (int i = 0; i < height; i++)
				{
					T tmpData[16] = { 0 };
					for (int j = 0; j < width; j++)
					{
						int offset = i*width + j;
						float x = j + pU[offset];
						float y = i + pV[offset];
						if (x < 0 || x > width - 1 || y < 0 || y > height - 1)
						{
							if (pIm1)
							{
								for (int c = 0; c < nChannels; c++)
									pWarpIm2[offset*nChannels + c] = pIm1[offset*nChannels + c];
							}

							continue;
						}

						int x0 = floor(x);
						int y0 = floor(y);
						float fx = x - x0;
						float fy = y - y0;

						for (int c = 0; c < nChannels; c++)
						{
							for (int s = 0; s < 4; s++)
							{
								for (int t = 0; t < 4; t++)
								{
									int tmpx = EnforceRange(x0 - 1 + t, width);
									int tmpy = EnforceRange(y0 - 1 + s, height);

									tmpData[s * 4 + t] = pIm2[(tmpy*width + tmpx)*nChannels + c];
								}
							}
							pWarpIm2[offset*nChannels + c] = ZQ_BicubicInterpolate(tmpData, fx, fy);
						}

					}
				}
#ifdef ZQLIB_USE_OPENMP
			}
#endif
		}

		/*median filter, each channel is handled separately*/
		template<class T>
		bool MedianFilter(const T* pSrcImage, T* pDstImage, const int width, const int height, int nChannels, const int fsize, bool use_omp)
		{
			if (fsize < 1)
				return false;

			int win_width = fsize * 2 + 1;
			
			for (int c = 0; c < nChannels; c++)
			{
#ifdef ZQLIB_USE_OPENMP
				if (use_omp)
				{
					int nthreads = omp_get_num_threads();
					int nPixels = width*height;
#pragma omp parallel for schedule(dynamic, (nPixels+nthreads-1)/nthreads)
					for(int pp = 0;pp < nPixels;pp++)
					{
						int h = pp / width;
						int w = pp % width;
						T* neighbor_vals = new T[win_width*win_width];

						int offset = pp;
						int cur_num = 0;
						int start_ii = __max(0, h - fsize);
						int end_ii = __min(height - 1, h + fsize);
						int start_jj = __max(0, w - fsize);
						int end_jj = __min(width - 1, w + fsize);
						for (int ii = start_ii; ii <= end_ii; ii++)
						{
							for (int jj = start_jj; jj <= end_jj; jj++)
							{
								neighbor_vals[cur_num] = pSrcImage[(ii*width + jj)*nChannels + c];
								cur_num++;
							}
						}
						ZQ_QuickSort::FindKthMax(neighbor_vals, cur_num, cur_num / 2, pDstImage[offset*nChannels + c]);
						delete[]neighbor_vals;
					}	
				}
				else
				{
#endif
					for (int h = 0; h < height; h++)
					{
						T* neighbor_vals = new T[win_width*win_width];
						for (int w = 0; w < width; w++)
						{
							int offset = h*width + w;
							int cur_num = 0;
							int start_ii = __max(0, h - fsize);
							int end_ii = __min(height - 1, h + fsize);
							int start_jj = __max(0, w - fsize);
							int end_jj = __min(width - 1, w + fsize);
							for (int ii = start_ii; ii <= end_ii; ii++)
							{
								for (int jj = start_jj; jj <= end_jj; jj++)
								{
									neighbor_vals[cur_num] = pSrcImage[(ii*width + jj)*nChannels + c];
									cur_num++;
								}
							}
							ZQ_QuickSort::FindKthMax(neighbor_vals, cur_num, cur_num / 2, pDstImage[offset*nChannels + c]);
						}
						delete[]neighbor_vals;
					}
#ifdef ZQLIB_USE_OPENMP
				}
#endif
			}
			
			return true;
		}

		template<class T>
		bool MedianFilter33_1channel(const T* pSrcImage, T* pDstImage, const int width, const int height, bool use_omp/* = false*/)
		{
			//padding
			int padding_width = width + 2;
			int padding_height = height + 2;
			T* tmpImg = new T[padding_width*padding_height];
			for (int h = 1; h < height + 1; h++)
				memcpy(tmpImg + (h*padding_width + 1), pSrcImage + (h - 1)*width, sizeof(T)*width);
			memcpy(tmpImg, tmpImg + (padding_width + 1), sizeof(T)*width);
			memcpy(tmpImg + ((padding_height - 1)*padding_width + 1), tmpImg + ((padding_height - 2)*padding_width + 1), sizeof(T)*width);
			for (int h = 0; h < height + 2; h++)
			{
				memcpy(tmpImg + h*padding_width, tmpImg + (h*padding_width + 1), sizeof(T));
				memcpy(tmpImg + (h*padding_width + padding_width - 1), tmpImg + (h*padding_width + padding_width - 2), sizeof(T));
			}

#ifdef ZQLIB_USE_OPENMP
			if(use_omp)
			{
				int nthreads = omp_get_num_threads();
#pragma omp parallel for schedule(dynamic, (height+nthreads-1)/nthreads)
				for (int h = 0; h < height; h++)
				{
					T col[3][3];
					col[0][0] = tmpImg[h*padding_width + 0];
					col[0][1] = tmpImg[(h + 1)*padding_width + 0];
					col[0][2] = tmpImg[(h + 2)*padding_width + 0];
					Sort_decend_3elements(col[0]);
					col[0][0] = tmpImg[h*padding_width + 1];
					col[0][1] = tmpImg[(h + 1)*padding_width + 1];
					col[0][2] = tmpImg[(h + 2)*padding_width + 1];
					Sort_decend_3elements(col[1]);
					int k = 2;

					for (int w = 0; w < width; w++)
					{
						col[k][0] = tmpImg[h*padding_width + w + 2];
						col[k][1] = tmpImg[(h + 1)*padding_width + w + 2];
						col[k][2] = tmpImg[(h + 2)*padding_width + w + 2];
						Sort_decend_3elements(col[k]);

						T a = __min(col[0][0], __min(col[1][0], col[2][0]));
						T b = Median_value(col[0][1], col[1][1], col[2][1]);
						T c = __max(col[0][2], __max(col[1][2], col[2][2]));
						pDstImage[h*width + w] = Median_value(a, b, c);
						k++;
						k %= 3;
					}
				}
			}
			else
			{
#endif
				for (int h = 0; h < height; h++)
				{
					T col[3][3];
					col[0][0] = tmpImg[h*padding_width + 0];
					col[0][1] = tmpImg[(h + 1)*padding_width + 0];
					col[0][2] = tmpImg[(h + 2)*padding_width + 0];
					Sort_decend_3elements(col[0]);
					col[0][0] = tmpImg[h*padding_width + 1];
					col[0][1] = tmpImg[(h + 1)*padding_width + 1];
					col[0][2] = tmpImg[(h + 2)*padding_width + 1];
					Sort_decend_3elements(col[1]);
					int k = 2;

					for (int w = 0; w < width; w++)
					{
						col[k][0] = tmpImg[h*padding_width + w + 2];
						col[k][1] = tmpImg[(h + 1)*padding_width + w + 2];
						col[k][2] = tmpImg[(h + 2)*padding_width + w + 2];
						Sort_decend_3elements(col[k]);

						T a = __min(col[0][0], __min(col[1][0], col[2][0]));
						T b = Median_value(col[0][1], col[1][1], col[2][1]);
						T c = __max(col[0][2], __max(col[1][2], col[2][2]));
						pDstImage[h*width + w] = Median_value(a, b, c);
						k++;
						k %= 3;
					}
				}
#ifdef ZQLIB_USE_OPENMP
			}
#endif
			delete[]tmpImg;
			
			return true;
		}

		/*median filter, each channel is handled separately*/
		template<class T>
		bool MedianFilterWithMask(const T* pSrcImage, T* pDstImage, const int width, const int height, int nChannels, const int fsize, const bool* keep_mask, bool use_omp)
		{
			if (fsize < 1)
				return false;

			memcpy(pDstImage, pSrcImage, sizeof(T)*width*height*nChannels);
			int win_width = fsize * 2 + 1;
			
			for (int c = 0; c < nChannels; c++)
			{
#ifdef ZQLIB_USE_OPENMP
				if(use_omp)
				{
					int nthreads = omp_get_num_threads();
					int nPixels = width*height;
#pragma omp parallel for schedule(dynamic,(nPixels+nthreads-1)/nthreads)
					for(int pp = 0;pp < nPixels;pp++)
					{
						int h = pp / width;
						int w = pp % width;
						int offset = h*width + w;
						if (keep_mask[offset])
							continue;
						T* neighbor_vals = new T[win_width*win_width];
						int cur_num = 0;
						for (int ii = __max(0, h - fsize); ii <= __min(height - 1, h + fsize); ii++)
						{
							for (int jj = __max(0, w - fsize); jj <= __min(width - 1, w + fsize); jj++)
							{
								if (keep_mask[ii*width + jj])
								{
									neighbor_vals[cur_num] = pSrcImage[(ii*width + jj)*nChannels + c];
									cur_num++;
								}

							}
						}
						if (cur_num > 0)
						{
							ZQ_QuickSort::FindKthMax(neighbor_vals, cur_num, cur_num / 2, pDstImage[(h*width + w)*nChannels + c]);
						}

						delete[]neighbor_vals;
					}
				}
				else
				{
#endif
					for (int h = 0; h < height; h++)
					{
						T* neighbor_vals = new T[win_width*win_width];
						for (int w = 0; w < width; w++)
						{
							int offset = h*width + w;
							if (keep_mask[offset])
								continue;
							int cur_num = 0;
							for (int ii = __max(0, h - fsize); ii <= __min(height - 1, h + fsize); ii++)
							{
								for (int jj = __max(0, w - fsize); jj <= __min(width - 1, w + fsize); jj++)
								{
									if (keep_mask[ii*width + jj])
									{
										neighbor_vals[cur_num] = pSrcImage[(ii*width + jj)*nChannels + c];
										cur_num++;
									}

								}
							}
							if (cur_num > 0)
							{
								ZQ_QuickSort::FindKthMax(neighbor_vals, cur_num, cur_num / 2, pDstImage[(h*width + w)*nChannels + c]);
							}
						}
						delete[]neighbor_vals;
					}
#ifdef ZQLIB_USE_OPENMP
				}
#endif
			}
			return true;
		}

		/*median filter, each channel is handled separately*/
		template<class T>
		bool MedianFilterWithMask(const T* pSrcImage, T* pDstImage, bool* dst_mask, int width, int height, int nChannels, const int fsize, const bool* keep_mask, float thresh_ratio /*= 0.5*/, bool use_omp)
		{
			if (fsize < 1)
				return false;

			memcpy(pDstImage, pSrcImage, sizeof(T)*width*height*nChannels);
			int win_width = fsize * 2 + 1;
			memcpy(dst_mask, keep_mask, sizeof(bool)*width*height);

			int thresh_num = thresh_ratio*win_width*win_width;
			
			for (int c = 0; c < nChannels; c++)
			{
#ifdef ZQLIB_USE_OPENMP
				if (use_omp)
				{
					int nthreads = omp_get_num_threads();
					int nPixels = width*height;
#pragma omp parallel for schedule(dynamic, (nPixels+nthreads-1)/nthreads)
					for(int pp = 0;pp < nPixels;pp++)
					{
						int h = pp / width;
						int w = pp % width;


						int offset = h*width + w;
						if (keep_mask[offset])
							continue;
						T* neighbor_vals = new T[win_width*win_width];
						int cur_num = 0;
						for (int ii = __max(0, h - fsize); ii <= __min(height - 1, h + fsize); ii++)
						{
							for (int jj = __max(0, w - fsize); jj <= __min(width - 1, w + fsize); jj++)
							{
								if (keep_mask[ii*width + jj])
								{
									neighbor_vals[cur_num] = pSrcImage[(ii*width + jj)*nChannels + c];
									cur_num++;
								}

							}
						}
						if (cur_num > 0)
						{
							ZQ_QuickSort::FindKthMax(neighbor_vals, cur_num, cur_num / 2, pDstImage[(h*width + w)*nChannels + c]);
							dst_mask[offset] = cur_num >= thresh_num;
						}

						delete[]neighbor_vals;
					}
				}
				else
				{
#endif
					for (int h = 0; h < height; h++)
					{
						T* neighbor_vals = new T[win_width*win_width];
						for (int w = 0; w < width; w++)
						{
							int offset = h*width + w;
							if (keep_mask[offset])
								continue;
							int cur_num = 0;
							for (int ii = __max(0, h - fsize); ii <= __min(height - 1, h + fsize); ii++)
							{
								for (int jj = __max(0, w - fsize); jj <= __min(width - 1, w + fsize); jj++)
								{
									if (keep_mask[ii*width + jj])
									{
										neighbor_vals[cur_num] = pSrcImage[(ii*width + jj)*nChannels + c];
										cur_num++;
									}

								}
							}
							if (cur_num > 0)
							{
								ZQ_QuickSort::FindKthMax(neighbor_vals, cur_num, cur_num / 2, pDstImage[(h*width + w)*nChannels + c]);
								dst_mask[offset] = cur_num >= thresh_num;
							}
						}
						delete[]neighbor_vals;
					}
#ifdef ZQLIB_USE_OPENMP
				}
#endif
			}
			
			return true;
		}

		template<class T>
		bool Edge(const T* pSrcImage, bool* edge, const int width, const int height, const float scale_of_mean, bool use_omp/* = false*/)
		{
			if (pSrcImage == 0 || edge == 0)
				return false;
			T sobel_filter_x[9] =
			{
				-1.0 / 8, 0, 1.0 / 8,
				-2.0 / 8, 0, 2.0 / 8,
				-1.0 / 8, 0, 1.0 / 8
			};
			T sobel_filter_y[9] =
			{
				-1.0 / 8, -2.0 / 8, -1.0 / 8,
				0, 0, 0,
				1.0 / 8, 2.0 / 8, 1.0 / 8
			};
			T* sobel_x = new T[width*height];
			T* sobel_y = new T[width*height];
			T* sobel_xy2 = new T[width*height];
			ImageFilter2D(pSrcImage, sobel_x, width, height, 1, sobel_filter_x, 1, 1, use_omp);
			ImageFilter2D(pSrcImage, sobel_y, width, height, 1, sobel_filter_y, 1, 1, use_omp);

			T sobel_sum = 0;
			int nPixels = width*height;
#ifdef ZQLIB_USE_OPENMP
			if(use_omp)
			{
				int nthreads = omp_get_num_threads();
#pragma omp parallel for schedule(dynamic, (nPixels+nthreads-1)/nthreads)
				for (int i = 0; i < nthreads; i++)
				{
					sobel_xy2[i] = sobel_x[i] * sobel_x[i] + sobel_y[i] * sobel_y[i];
					sobel_sum += sobel_xy2[i];
				}
			}
			else
			{
#endif
				for (int i = 0; i < nPixels; i++)
				{
					sobel_xy2[i] = sobel_x[i] * sobel_x[i] + sobel_y[i] * sobel_y[i];
					sobel_sum += sobel_xy2[i];
				}
#ifdef ZQLIB_USE_OPENMP
			}
#endif

			T sobel_mean = sobel_sum / (width*height);

			T thresh = sobel_mean*scale_of_mean;

#ifdef ZQLIB_USE_OPENMP
			if(use_omp)
			{
				int nthreads = omp_get_num_threads();
#pragma omp parallel for schedule(dynamic, (nPixels+nthreads-1)/nthreads)
				for (int i = 0; i < nPixels; i++)
				{
					edge[i] = sobel_xy2[i] > thresh;
				}
			}
			else
			{
#endif
				for (int i = 0; i < nPixels; i++)
				{
					edge[i] = sobel_xy2[i] > thresh;
				}
#ifdef ZQLIB_USE_OPENMP
			}
#endif
			delete[]sobel_xy2;
			delete[]sobel_x;
			delete[]sobel_y;

			return true;
		}
	}
}

#endif