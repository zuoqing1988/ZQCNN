#ifndef _ZQ_IMAGE_PROCESSING_3D_H_
#define _ZQ_IMAGE_PROCESSING_3D_H_
#pragma once

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "ZQ_CubicInterpolation.h"

namespace ZQ
{
	namespace ZQ_ImageProcessing3D
	{
		/*basic functions*/
		template<class T>
		T EnforceRange(const T& x,const int& MaxValue);

		/*trilinear interpolation*/
		template<class T>
		void TrilinearInterpolate(const T* pImage,const int width,const int height, const int depth, const int nChannels, const float x, const float y, const float z, 
									T* result, bool use_period_coord);

		/*tricubic interpolation*/
		template<class T>
		void TricubicInterpolate(const T* pImage, const int width, const int height, const int depth, const int nChannels, const float x, const float y, const float z, 
									T* result, bool use_period_coord);


		/*resize image*/
		template<class T>
		void ResizeImage(const T* pSrcImage, T* pDstImage, const int srcWidth, const int srcHeight, const int srcDepth, const int nChannels, 
									const int dstWidth,const int dstHeight, const int dstDepth);

		template<class T>
		void ResizeImageTricubic(const T* pSrcImage, T* pDstImage, const int srcWidth, const int srcHeight, const int srcDepth, const int nChannels, 
									const int dstWidth, const int dstHeight, const int dstDepth);

		/*resize a flow field on MAC grid
		* U : (width+1)*height*depth,
		* V : width*(height+1)*depth,
		* W : width*height*(depth+1)
		*/
		template<class T>
		void ResizeFlow(const T* pSrcU, const T* pSrcV, const T* pSrcW, T* pDstU, T* pDstV, T* pDstW, const int srcWidth, const int srcHeight, const int srcDepth, 
									const int dstWidth, const int dstHeight, const int dstDepth);

		template<class T>
		void ResizeFlowTricubic(const T* pSrcU, const T* pSrcV, const T* pSrcW, T* pDstU, T* pDstV, T* pDstW, const int srcWidth, const int srcHeight, const int srcDepth,
									const int dstWidth, const int dstHeight, const int dstDepth);

		/*filter, x dimension*/
		template<class T>
		void Hfiltering(const T* pSrcImage, T* pDstImage, const int width, const int height, const int depth, const int nChannels, const T* pfilter1D, const int fsize);

		/*filter, y dimension*/
		template<class T>
		void Vfiltering(const T* pSrcImage, T* pDstImage, const int width, const int height, const int depth, const int nChannels, const T* pfilter1D, const int fsize);

		/*filter, z dimension*/
		template<class T>
		void Dfiltering(const T* pSrcImage, T* pDstImage, const int width, const int height, const int depth, const int nChannels, const T* pfilter1D, const int fsize);

		template<class T>
		void Laplacian(const T* pSrcImage, T* pDstImage, const int width, const int height, const int depth, const int nChannels);

		/*warp image*/
		template<class T>
		void WarpImage(T* pWarpIm2, const T* pIm2, const T* pU, const T* pV, const T* pW, const int width, const int height, const int depth, const int nChannels,const T* pIm1 = 0);

		template<class T>
		void WarpImageTricubic(T* pWarpIm2, const T* pIm2,const T* pU, const T* pV, const T* pW, const int width, const int height, const int depth, const int nChannels,  const T* pIm1 = 0);

		/*********************************************************************************/
		/********************************** definitions **********************************/
		/*********************************************************************************/


		template<class T>
		T EnforceRange(const T& x,const int& MaxValue)
		{
			return __min(__max(x,0),MaxValue-1);
		}


		template<class T>
		void TrilinearInterpolate(const T* pImage,const int width,const int height, const int depth, const int nChannels, const float x, const float y, const float z, 
											T* result, bool use_period_coord)
		{
			memset(result,0,sizeof(T)*nChannels);

			if(!use_period_coord)
			{
				float fx = EnforceRange(x,width);
				float fy = EnforceRange(y,height);
				float fz = EnforceRange(z,depth);
				int ix = floor(fx);
				int iy = floor(fy);
				int iz = floor(fz);
				float sx = fx-ix;
				float sy = fy-iy;
				float sz = fz-iz;

				for(int k = 0;k <= 1;k++)
				{
					for(int j = 0;j <= 1;j++)
					{
						for(int i = 0;i <= 1;i++)
						{
							int u = EnforceRange(ix+i,width);
							int v = EnforceRange(iy+j,height);
							int w = EnforceRange(iz+k,depth);

							for(int c = 0;c < nChannels;c++)
								result[c] += fabs(1-i-sx)*fabs(1-j-sy)*fabs(1-k-sz)*pImage[(w*height*width+v*width+u)*nChannels+c];
						}
					}
				}			
			}
			else
			{
				float shift_x = floor(x/width)*width;
				float shift_y = floor(y/height)*height;
				float shift_z = floor(z/depth)*depth;
				float xxx = x - shift_x;
				float yyy = y - shift_y;
				float zzz = z - shift_z;

				int ix = floor(xxx);
				int iy = floor(yyy);
				int iz = floor(zzz);
				float sx = xxx - ix;
				float sy = yyy - iy;
				float sz = zzz - iz;
				for(int k = 0;k <= 1;k++)
				{
					for(int j = 0;j <= 1;j++)
					{
						for(int i = 0;i <= 1;i++)
						{
							int u = (ix+i)%width;
							int v = (iy+j)%height;
							int w = (iz+k)%depth;
							for(int c = 0;c < nChannels;c++)
								result[c] += fabs(1-i-sx)*fabs(1-j-sy)*fabs(1-k-sz)*pImage[(w*height*width+v*width+u)*nChannels+c];
						}
					}
				}
			}	
		}


		template<class T>
		void TricubicInterpolate(const T* pImage, const int width, const int height, const int depth, const int nChannels, const float x, const float y, const float z, 
										T* result, bool use_period_coord)
		{
			memset(result,0,sizeof(T)*nChannels);

			if(!use_period_coord)
			{
				int ix = floor(x);
				int iy = floor(y);
				int iz = floor(z);
				float sx = x-ix;
				float sy = y-iy;
				float sz = z-iz;

				T data[64] = {0};
				for(int c = 0;c < nChannels;c++)
				{
					for(int k = 0;k < 4;k++)
					{
						for(int j = 0;j < 4;j++)
						{
							for(int i = 0;i < 4;i++)
							{
								int cur_x = EnforceRange(ix-1+i,width);
								int cur_y = EnforceRange(iy-1+j,height);
								int cur_z = EnforceRange(iz-1+k,depth);
								data[k*16+j*4+i] = pImage[(cur_z*height*width+cur_y*width+cur_x)*nChannels+c];
							}
						}
					}
					
					result[c] = ZQ_TricubicInterpolate(data,sx,sy,sz);
				}
			}
			else
			{
				float shift_x = floor(x/width)*width;
				float shift_y = floor(y/height)*height;
				float shift_z = floor(z/depth)*depth;
				float xxx = x - shift_x;
				float yyy = y - shift_y;
				float zzz = z - shift_z;

				int ix = floor(xxx);
				int iy = floor(yyy);
				int iz = floor(zzz);
				float sx = xxx - ix;
				float sy = yyy - iy;
				float sz = zzz - iz;

				T data[64] = {0};
				for(int c = 0;c < nChannels;c++)
				{
					for(int k = 0;k < 4;k++)
					{
						for(int j = 0;j < 4;j++)
						{
							for(int i = 0;i < 4;i++)
							{
								int cur_x = (ix-1+i+width)%width;
								int cur_y = (iy-1+j+height)%height;
								int cur_z = (iz-1+k+depth)%depth;
								data[k*16+j*4+i] = pImage[(cur_z*height*width+cur_y*width+cur_x)*nChannels+c];
							}
						}
					}
					result[c] = ZQ_TricubicInterpolate(data,sx,sy,sz);
				}
			}
			
		}


		template<class T>
		void ResizeImage(const T* pSrcImage, T* pDstImage, const int srcWidth, const int srcHeight, const int srcDepth, const int nChannels, 
									const int dstWidth,const int dstHeight, const int dstDepth)
		{
			memset(pDstImage,0,sizeof(T)*dstWidth*dstHeight*dstDepth*nChannels);

			for(int k = 0;k < dstDepth;k++)
			{
				for(int j = 0;j < dstHeight;j++)
				{
					for(int i = 0;i < dstWidth;i++)
					{
						float coordx = (i+0.5)/dstWidth*srcWidth-0.5;
						float coordy = (j+0.5)/dstHeight*srcHeight-0.5;
						float coordz = (k+0.5)/dstDepth*srcDepth-0.5;

						TrilinearInterpolate(pSrcImage,srcWidth,srcHeight,srcDepth,nChannels,coordx,coordy,coordz,pDstImage+(k*dstHeight*dstWidth+j*dstWidth+i)*nChannels,false);
					}
				}
			}
		}


		template<class T>
		void ResizeImageTricubic(const T* pSrcImage, T* pDstImage, const int srcWidth, const int srcHeight, const int srcDepth, const int nChannels, 
									const int dstWidth, const int dstHeight, const int dstDepth)
		{
			memset(pDstImage,0,sizeof(T)*dstWidth*dstHeight*dstDepth*nChannels);

			T tmpData[64] = {0};

			for(int k = 0;k < dstDepth;k++)
			{
				for(int j = 0;j < dstHeight;j++)
				{
					for(int i = 0;i < dstWidth;i++)
					{
						float coordx = (i+0.5)/dstWidth*srcWidth-0.5;
						float coordy = (j+0.5)/dstHeight*srcHeight-0.5;
						float coordz = (k+0.5)/dstDepth*srcDepth-0.5;

						int ix = floor(coordx);
						int iy = floor(coordy);
						int iz = floor(coordz);
						float fx = coordx-ix;
						float fy = coordy-iy;
						float fz = coordz-iz;

						for(int c = 0;c < nChannels;c++)
						{
							for(int t = 0;t < 4;t++)
							{
								for(int s = 0;s < 4;s++)
								{
									for(int r = 0;r < 4;r++)
									{
										int tmpx = EnforceRange(ix-1+r,srcWidth);
										int tmpy = EnforceRange(iy-1+s,srcHeight);
										int tmpz = EnforceRange(iz-1+t,srcDepth);

										tmpData[t*16+s*4+r] = pSrcImage[(tmpz*srcHeight*srcWidth+tmpy*srcWidth+tmpx)*nChannels+c];
									}
								}
							}
							
							pDstImage[(k*dstHeight*dstWidth+j*dstWidth+i)*nChannels+c] = ZQ_TricubicInterpolate(tmpData,fx,fy,fz);
						}
					}
				}
			}
			
		}


		template<class T>
		void ResizeFlow(const T* pSrcU, const T* pSrcV, const T* pSrcW, T* pDstU, T* pDstV, T* pDstW, const int srcWidth, const int srcHeight, const int srcDepth, 
								const int dstWidth, const int dstHeight, const int dstDepth)
		{
			memset(pDstU,0,sizeof(T)*(dstWidth+1)*dstHeight*dstDepth);
			memset(pDstV,0,sizeof(T)*dstWidth*(dstHeight+1)*dstDepth);
			memset(pDstW,0,sizeof(T)*dstWidth*dstHeight*(dstDepth+1));

			//resize U
			for(int k = 0;k < dstDepth;k++)
			{
				for(int j = 0;j < dstHeight;j++)
				{
					for(int i = 0;i <= dstWidth;i++)
					{
						float coordx = (float)j/dstWidth*srcWidth;
						float coordy = (i+0.5f)/dstHeight*srcHeight - 0.5f;
						float coordz = (k+0.5f)/dstDepth*srcDepth - 0.5f;

						TrilinearInterpolate(pSrcU,srcWidth+1,srcHeight,srcDepth,1,coordx,coordy,coordz,pDstU+k*dstHeight*(dstWidth+1)+j*(dstWidth+1)+i,false);
					}
				}
			}
			

			//resize V
			for(int k = 0;k < dstDepth;k++)
			{
				for(int j = 0;j <= dstHeight;j++)
				{
					for(int i = 0;i < dstWidth;i++)
					{
						float coordx = (i+0.5f)/dstWidth*srcWidth - 0.5f;
						float coordy = (float)j/dstHeight*srcHeight;
						float coordz = (k+0.5f)/dstDepth*srcDepth - 0.5f;

						TrilinearInterpolate(pSrcV,srcWidth,srcHeight+1,srcDepth,1,coordx,coordy,coordz,pDstV+k*(dstHeight+1)*dstWidth+j*dstWidth+i,false);
					}
				}
			}

			//resize W
			for(int k = 0;k <= dstDepth;k++)
			{
				for(int j = 0;j < dstHeight;j++)
				{
					for(int i = 0;i < dstWidth;i++)
					{
						float coordx = (i+0.5f)/dstWidth*srcWidth - 0.5f;
						float coordy = (j+0.5f)/dstHeight*srcHeight - 0.5f;
						float coordz = (float)k/dstDepth*srcDepth;

						TrilinearInterpolate(pSrcW,srcWidth,srcHeight,srcDepth+1,1,coordx,coordy,coordz,pDstW+k*dstHeight*dstWidth+j*dstWidth+i,false);
					}
				}
			}
			
		}


		template<class T>
		void ResizeFlowTricubic(const T* pSrcU, const T* pSrcV, const T* pSrcW, T* pDstU, T* pDstV, T* pDstW, const int srcWidth, const int srcHeight, const int srcDepth,
							const int dstWidth, const int dstHeight, const int dstDepth)
		{
			memset(pDstU,0,sizeof(T)*(dstWidth+1)*dstHeight*dstDepth);
			memset(pDstV,0,sizeof(T)*dstWidth*(dstHeight+1)*dstDepth);
			memset(pDstW,0,sizeof(T)*dstWidth*dstHeight*(dstDepth+1));

			T tmpData[64] = {0};

			//resize U
			for(int k = 0;k < dstDepth;k++)
			{
				for(int j = 0;j < dstHeight;j++)
				{
					for(int i = 0;i <= dstWidth;i++)
					{
						float coordx = (float)i/dstWidth*srcWidth;
						float coordy = (j+0.5f)/dstHeight*srcHeight-0.5f;
						float coordz = (k+0.5f)/dstDepth*srcDepth-0.5f;

						int ix = floor(coordx);
						int iy = floor(coordy);
						int iz = floor(coordz);
						float fx = coordx-ix;
						float fy = coordy-iy;
						float fz = coordz-iz;

						for(int t = 0;t < 4;t++)
						{
							for(int s = 0;s < 4;s++)
							{
								for(int r = 0;r < 4;r++)
								{
									int tmpx = EnforceRange(ix-1+r,srcWidth+1);
									int tmpy = EnforceRange(iy-1+s,srcHeight);
									int tmpz = EnforceRange(iz-1+t,srcDepth);

									tmpData[t*16+s*4+r] = pSrcU[tmpz*srcHeight*(srcWidth+1)+tmpy*(srcWidth+1)+tmpx];
								}
							}

						}
						
						pDstU[k*dstHeight*(dstWidth+1)+j*(dstWidth+1)+i] = ZQ_TricubicInterpolate(tmpData,fx,fy,fz);

					}
				}
			}
			

			//resizeV
			for(int k = 0;k < dstDepth;k++)
			{
				for(int j = 0;j <= dstHeight;j++)
				{
					for(int i = 0;i < dstWidth;i++)
					{
						float coordx = (i+0.5f)/dstWidth*srcWidth-0.5f;
						float coordy = (float)j/dstHeight*srcHeight;
						float coordz = (k+0.5f)/dstDepth*srcDepth-0.5f;
						int ix = floor(coordx);
						int iy = floor(coordy);
						int iz = floor(coordz);
						float fx = coordx-ix;
						float fy = coordy-iy;
						float fz = coordz-iz;

						for(int t = 0;t < 4;t++)
						{
							for(int s = 0;s < 4;s++)
							{
								for(int r = 0;r < 4;r++)
								{
									int tmpx = EnforceRange(ix-1+r,srcWidth);
									int tmpy = EnforceRange(iy-1+s,srcHeight+1);
									int tmpz = EnforceRange(iz-1+t,srcDepth);

									tmpData[t*16+s*4+r] = pSrcV[tmpz*(srcHeight+1)*srcWidth+tmpy*srcWidth+tmpx];
								}
							}
						}
						
						pDstV[k*(dstHeight+1)*dstWidth+j*dstWidth+i] = ZQ_TricubicInterpolate(tmpData,fx,fy,fz);
					}
				}
			}

			//resizeW
			for(int k = 0;k <= dstDepth;k++)
			{
				for(int j = 0;j < dstHeight;j++)
				{
					for(int i = 0;i < dstWidth;i++)
					{
						float coordx = (i+0.5f)/dstWidth*srcWidth-0.5f;
						float coordy = (j+0.5f)/dstHeight*srcHeight-0.5f;
						float coordz = (float)k/dstDepth*srcDepth;
						int ix = floor(coordx);
						int iy = floor(coordy);
						int iz = floor(coordz);
						float fx = coordx-ix;
						float fy = coordy-iy;
						float fz = coordz-iz;

						for(int t = 0;t < 4;t++)
						{
							for(int s = 0;s < 4;s++)
							{
								for(int r = 0;r < 4;r++)
								{
									int tmpx = EnforceRange(ix-1+r,srcWidth);
									int tmpy = EnforceRange(iy-1+s,srcHeight);
									int tmpz = EnforceRange(iz-1+t,srcDepth+1);

									tmpData[t*16+s*4+r] = pSrcV[tmpz*srcHeight*srcWidth+tmpy*srcWidth+tmpx];
								}
							}
						}

						pDstV[k*dstHeight*dstWidth+j*dstWidth+i] = ZQ_TricubicInterpolate(tmpData,fx,fy,fz);
					}
				}
			}
			
		}


		template<class T>
		void Hfiltering(const T* pSrcImage, T* pDstImage, const int width, const int height, const int depth, const int nChannels, const T* pfilter1D, const int fsize)
		{
			memset(pDstImage,0,sizeof(T)*width*height*depth*nChannels);

			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 0;i < width;i++)
					{
						for(int l = -fsize; l <= fsize;l++)
						{
							int ii = EnforceRange(i+l,width);
							for(int c = 0;c < nChannels;c++)
								pDstImage[(k*height*width+j*width+i)*nChannels+c] += pSrcImage[(k*height*width+j*width+ii)*nChannels+c]*pfilter1D[l+fsize];
						}
					}
				}
			}
		}


		template<class T>
		void Vfiltering(const T* pSrcImage, T* pDstImage, const int width, const int height, const int depth, const int nChannels, const T* pfilter1D, const int fsize)
		{
			memset(pDstImage,0,sizeof(T)*width*height*depth*nChannels);

			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 0;i < width;i++)
					{
						for(int l = -fsize;l <= fsize;l++)
						{
							int jj = EnforceRange(j+l,height);
							for(int c = 0;c < nChannels;c++)
								pDstImage[(k*height*width+j*width+i)*nChannels+c] += pSrcImage[(k*height*width+jj*width+i)*nChannels+c]*pfilter1D[l+fsize];
						}
					}
				}
			}
		}

		template<class T>
		void Dfiltering(const T* pSrcImage, T* pDstImage, const int width, const int height, const int depth, const int nChannels, const T* pfilter1D, const int fsize)
		{
			memset(pDstImage,0,sizeof(T)*width*height*depth*nChannels);

			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 0;i < width;i++)
					{
						for(int l = -fsize;l <= fsize;l++)
						{
							int kk = EnforceRange(k+l,depth);
							for(int c = 0;c < nChannels;c++)
								pDstImage[(k*height*width+j*width+i)*nChannels+c] += pSrcImage[(kk*height*width+j*width+i)*nChannels+c]*pfilter1D[l+fsize];
						}
					}
				}
			}
		}


		template<class T>
		void Laplacian(const T* pSrcImage, T* pDstImage, const int width, const int height, const int depth, const int nChannels)
		{
			memset(pDstImage,0,sizeof(T)*width*height*depth*nChannels);

			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 1;i < width-1;i++)
					{
						for(int c = 0;c < nChannels;c++)
							pDstImage[(k*height*width+j*width+i)*nChannels+c] += pSrcImage[(k*height*width+j*width+i+1)*nChannels+c] + pSrcImage[(k*height*width+j*width+i-1)*nChannels+c];
					}
					for(int c = 0;c < nChannels;c++)
						pDstImage[(k*height*width+j*width+0)*nChannels+c] += pSrcImage[(k*height*width+j*width+0)*nChannels+c] + pSrcImage[(k*height*width+j*width+1)*nChannels+c];
					for(int c = 0;c < nChannels;c++)
						pDstImage[(k*height*width+j*width+width-1)*nChannels+c] += pSrcImage[(k*height*width+j*width+width-2)*nChannels+c] + pSrcImage[(k*height*width+j*width+width-1)*nChannels+c];
				}
			}
			

			for(int k = 0;k < depth;k++)
			{
				for(int i = 0;i < width;i++)
				{
					for(int j = 1;j < height-1;j++)
					{
						for(int c = 0;c < nChannels;c++)
							pDstImage[(k*height*width+j*width+i)*nChannels+c] += pSrcImage[(k*height*width+(j+1)*width+i)*nChannels+c] + pSrcImage[(k*height*width+(j-1)*width+i)*nChannels+c];
					}
					for(int c = 0;c < nChannels;c++)
						pDstImage[(k*height*width+0*width+i)*nChannels+c] += pSrcImage[(k*height*width+0*width+i)*nChannels+c] + pSrcImage[(k*height*width+1*width+i)*nChannels+c];
					for(int c = 0;c < nChannels;c++)
						pDstImage[(k*height*width+(height-1)*width+i)*nChannels+c] += pSrcImage[(k*height*width+(height-2)*width+i)*nChannels+c] + pSrcImage[(k*height*width+(height-1)*width+i)*nChannels+c];
				}
			}
			
			for(int j = 0;j < height;j++)
			{
				for(int i = 0;i < width;i++)
				{
					for(int k = 1;k < depth-1;k++)
					{
						for(int c = 0;c < nChannels;c++)
							pDstImage[(k*height*width+j*width+i)*nChannels+c] += pSrcImage[((k+1)*height*width+j*width+i)*nChannels+c] + pSrcImage[((k-1)*height*width+j*width+i)*nChannels+c];
					}
					for(int c = 0;c < nChannels;c++)
						pDstImage[(0*height*width+j*width+i)*nChannels+c] += pSrcImage[(0*height*width+j*width+i)*nChannels+c] + pSrcImage[(1*height*width+j*width+i)*nChannels+c];
					for(int c = 0;c < nChannels;c++)
						pDstImage[((depth-1)*height*width+j*width+i)*nChannels+c] += pSrcImage[((depth-2)*height*width+j*width+i)*nChannels+c] + pSrcImage[((depth-1)*height*width+j*width+i)*nChannels+c];
				}
			}

			for(int i = 0;i < depth*height*width*nChannels;i++)
			{
				pDstImage[i] -= 6*pSrcImage[i];
			}
		}


		template<class T>
		void WarpImage(T* pWarpIm2, const T* pIm2, const T* pU, const T* pV, const T* pW, const int width, const int height, const int depth, const int nChannels,const T* pIm1 /* = 0 */)
		{
			memset(pWarpIm2,0,sizeof(T)*width*height*depth*nChannels);

			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 0;i < width;i++)
					{
						int offset = k*height*width+j*width+i;
						float x = i+pU[offset];
						float y = j+pV[offset];
						float z = k+pW[offset];
						if(x < 0 || x > width-1 || y < 0 || y > height-1 || z < 0 || z > depth-1)
						{
							if(pIm1)
							{
								for(int c = 0;c < nChannels;c++)
									pWarpIm2[offset*nChannels+c] = pIm1[offset*nChannels+c];
							}

							continue;
						}
						TrilinearInterpolate(pIm2,width,height,depth,nChannels,x,y,z,pWarpIm2+offset*nChannels,false);
					}
				}
			}
		}


		template<class T>
		void WarpImageTricubic(T* pWarpIm2, const T* pIm2,const T* pU, const T* pV, const T* pW, const int width, const int height,  const int depth, const int nChannels, const T* pIm1 /* = 0 */)
		{
			T tmpData[64] = {0};

			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 0;i < width;i++)
					{
						int offset = k*height*width+j*width+i;
						float x = i + pU[offset];
						float y = j + pV[offset];
						float z = k + pW[offset];
						if(x < 0 || x > width-1 || y < 0 || y > height-1 || z < 0 || z > depth-1)
						{
							if(pIm1)
							{
								for(int c = 0;c < nChannels;c++)
									pWarpIm2[offset*nChannels+c] = pIm1[offset*nChannels+c];
							}

							continue;
						}

						int x0 = floor(x);
						int y0 = floor(y);
						int z0 = floor(z);
						float fx = x-x0;
						float fy = y-y0;
						float fz = z-z0;

						for(int c = 0;c < nChannels;c++)
						{
							for(int t = 0;t < 4;t++)
							{
								for(int s = 0;s < 4;s++)
								{
									for(int r = 0;r < 4;r++)
									{
										int tmpx = EnforceRange(x0-1+r,width);
										int tmpy = EnforceRange(y0-1+s,height);
										int tmpz = EnforceRange(z0-1+t,depth);

										tmpData[t*16+s*4+r] = pIm2[(tmpz*height*width+tmpy*width+tmpx)*nChannels+c];
									}
								}

							}
							
							pWarpIm2[offset*nChannels+c] = ZQ_TricubicInterpolate(tmpData,fx,fy,fz);
						}

					}
				}

			}
			
		}

	}
}

#endif