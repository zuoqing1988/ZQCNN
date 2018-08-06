#ifndef _ZQ_FIND_CORNERS_H_
#define _ZQ_FIND_CORNERS_H_
#pragma once

#include "ZQ_DoubleImage.h"
#include "ZQ_GaussianPyramid.h"
#include "ZQ_Vec2D.h"
#include "ZQ_BinaryImageProcessing.h"
//#include "ZQ_ImageIO.h"
#include "ZQ_MergeSort.h"
#include "ZQ_SVD.h"
#include <time.h>

namespace ZQ
{
	class ZQ_FindCorners
	{
	public:
		template<class T>
		static bool FindCorners(const ZQ_DImage<T>& img, ZQ_DImage<T>& crnrsgrid, bool& suspicious)
		{
			int width = img.width();
			int height = img.height();
			int nChannels = img.nchannels();
			if (nChannels != 1 || width < 50 || height < 50)
				return false;
			ZQ_DImage<T> image(img);
			image.AutoAdjust();


			int hwin = 3;//Harris window size
			float th = 0.5; // Parameter to adjust adaptive thresholding


			/* Make Pyramid
			Each level in pyramid consists of :
			im : image
			imadj : adjusted image
			ime : sobel edge image
			ix : x component of edge image
			iy : y component of edge image*/

			const int nolevels = 3; // no of levels
			ZQ_DImage<T> pyr_im[nolevels];
			ZQ_DImage<T> pyr_imadj[nolevels];
			ZQ_DImage<T> pyr_ime[nolevels];
			ZQ_DImage<T> pyr_ix[nolevels];
			ZQ_DImage<T> pyr_iy[nolevels];

			clock_t t1 = clock();
			for (int i = 0; i < nolevels; i++)
			{
				if (i == 0)
					pyr_im[i] = image;
				else
				{
					int dst_width = width / (pow(2.0, i)) + 0.5;
					int dst_height = height / (pow(2.0, i)) + 0.5;
					pyr_im[i].allocate(dst_width, dst_height, 1);
					ZQ_ImageProcessing::ResizeImage_NN(pyr_im[i - 1].data(), pyr_im[i].data(), pyr_im[i - 1].width(), pyr_im[i - 1].height(), 1, dst_width, dst_height, false);
					//pyr_im[i - 1].imresize(pyr_im[i], dst_width, dst_height);
				}
					
				_adaptimadj(pyr_im[i], pyr_imadj[i]);

				_getedges(pyr_im[i], pyr_ime[i], pyr_ix[i], pyr_iy[i]);
				/*ZQ_ImageIO::Show("im", pyr_im[i]);
				ZQ_ImageIO::Show("imadj", pyr_imadj[i]);
				ZQ_ImageIO::Show("ix", pyr_ix[i]);
				ZQ_ImageIO::Show("iy", pyr_iy[i]);
				cvWaitKey(0);*/
			}
			clock_t t2 = clock();
			//printf("build pyramid: %.3f secs\n", 0.001*(t2 - t1));

			// set nocrnrs and level
			int nocrnrs = -1;
			int level = 0;
			std::vector<ZQ_Vec2D> crnrpts, crnrs;
			std::vector<std::vector<int>> peaklocs;
			for (int i = 0; i < nolevels; i++)
			{
				ZQ_DImage<T> imgh, mimg, stdv;
				std::vector<ZQ_Vec2D> ctcrnrpts;
				_harristm(pyr_imadj[i], hwin, imgh);
				/*ZQ_ImageIO::Show("imgh", imgh);
				cvWaitKey(0);*/
				//ZQ_ImageIO::saveImage(imgh, "imgh.png");
				int win = __min(width, height) / 5 + 0.5;
				
				clock_t t21 = clock();
				_adaptstats(imgh, win, mimg, stdv);
				clock_t t22 = clock();
				//printf("stats = %.3f sec\n", 0.001*(t22 - t21));
				//ZQ_DImage<T> tmp_mimg(mimg);
				//tmp_mimg.AutoAdjust();
				//ZQ_ImageIO::saveImage(tmp_mimg, "mimg.png");
				////ZQ_ImageIO::Show("mimg", tmp_mimg);
				//ZQ_DImage<T> tmp_stdv(stdv);
				//tmp_stdv.AutoAdjust();
				//ZQ_ImageIO::saveImage(tmp_stdv, "stdv.png");
				///*ZQ_ImageIO::Show("stdv", tmp_stdv);
				//cvWaitKey(0);*/
				_getcrnrpts(imgh, mimg, stdv, th, ctcrnrpts);
				clock_t t23 = clock();
				//printf("getcrnrs = %.3f sec\n", 0.001*(t23 - t22));

				if (ctcrnrpts.size() == 0)
					continue;
				std::vector<ZQ_Vec2D> ctcrnrs;
				std::vector<std::vector<int>> ctpeaklocs;
				_chesscornerfilter(pyr_im[i], pyr_ime[i], ctcrnrpts, ctcrnrs, ctpeaklocs);
				clock_t t24 = clock();
				//printf("filter = %.3f sec\n", 0.001*(t24 - t23));
				if (ctcrnrs.size() > crnrs.size())
				{
					crnrs = ctcrnrs;
					crnrpts = ctcrnrpts;
					peaklocs = ctpeaklocs;
					level = i;
				}
			}

			if (crnrs.size() < 10)
				return false;

			clock_t t3 = clock();
			//printf("get corners: %.3f secs\n", 0.001*(t3 - t2));
			//Extract Grid
			_getgrid(crnrs, crnrpts, peaklocs, pyr_ix[level], pyr_iy[level], crnrsgrid);

			T*& crnrsgrid_data = crnrsgrid.data();
			int nelments = crnrsgrid.nelements();
			// adjust grid back to full scale
			for (int pp = 0; pp < nelments; pp++)
			{
				crnrsgrid_data[pp] *= pow(2.0, level);
			}
			for (int pp = 0; pp < crnrpts.size(); pp++)
			{
				crnrpts[pp] *= pow(2.0, level);
			}

			ZQ_DImage<T> crnrsgridfil;
			_filtergrid(crnrsgrid, crnrsgridfil);

			int crnrsgridfil_width = crnrsgridfil.width();
			int crnrsgridfil_height = crnrsgridfil.height();
			//check grid size
			if (crnrsgridfil_width < 3 || crnrsgridfil_height < 3)
				return false;


			//adjust grid direction
			ZQ_DImage<T> old_crnrsgridfil(crnrsgridfil);
			_adjgriddir(old_crnrsgridfil, crnrsgridfil);
			old_crnrsgridfil.clear();

			// get missing corners
			ZQ_DImage<T> gridfullrect;
			int nointerpolations = 0;
			_getmisscrnrs(crnrsgridfil, width, height, gridfullrect, nointerpolations);

			if (gridfullrect.nelements() == 0)
				return false;


			// adjust origin position
			ZQ_DImage<T> old_gridfullrect(gridfullrect);
			_adjgridorigin(old_gridfullrect, gridfullrect);
			old_gridfullrect.clear();


			ZQ_DImage<T> grid;
			int nobadpts = 0;
			int win;
			_getsubpixcrnrs(img, crnrpts, gridfullrect, grid, win, nobadpts);

			if (grid.width() < 3 || grid.height() < 3)
				return false;

			int susth = 4;
			suspicious = false;
			if (nobadpts > susth || nointerpolations > susth)
			{
				suspicious = true;
			}


			//Output Results
			crnrsgrid = grid;

			clock_t t4 = clock();
			//printf("extract grid: %.3f secs\n", 0.001*(t4 - t3));
			return true;
		}

	private:

		template<class T>
		static void _getsubpixcrnrs(const ZQ_DImage<T>& img, const std::vector<ZQ_Vec2D>& crnrpts, const ZQ_DImage<T>& grid, ZQ_DImage<T>& gridout, int& win, int& nobadpts)
		{
			/*
			function [gridout,win,nobadpts]=getsubpixcrnrs(img,crnrpts,grid)
			%GETSUBPIXCRNRS retruns the subpixel positions of the chessboard corners.
			%
			% GETSUBPIXCRNRS relies on the subpixel corner finder by
			% Jean-Yves Bouguet. The addition it introduces is the adaptive selection
			% of the window size.
			%
			% GETSUBPIXCRNRS also chooses a subset of the grid if certain corners do
			% converge using Bouget's code.
			*/


			int grid_width = grid.width();
			int grid_height = grid.height();
			const T*& grid_data = grid.data();

			// reshape into row vector
			std::vector<ZQ_Vec2D> gridpts;
			for (int i = 0; i < grid_width*grid_height; i++)
			{
				if (grid_data[i * 2] != 0)
				{
					gridpts.push_back(ZQ_Vec2D(grid_data[i * 2 + 0], grid_data[i * 2 + 1]));
				}
			}


			gridout.allocate(grid_width, grid_height, 2);
			ZQ_DImage<bool> bad(grid_width, grid_height);
			T*& gridout_data = gridout.data();
			bool*& bad_data = bad.data();

			win = 1e9;

			// get smallest window size first
			for (int cntr = 0; cntr < gridpts.size(); cntr++)
			{
				int ctwin = _getwin(img, gridpts[cntr], crnrpts) / 2.0 + 0.5;
				// get smallest win size
				if (ctwin < win && ctwin > 2)
					win = ctwin;
			}

			for (int h = 0; h < grid_height; h++)
			{
				for (int w = 0; w < grid_width; w++)
				{
					ZQ_Vec2D currentpt(grid_data[(h*grid_width + w) * 2 + 0], grid_data[(h*grid_width + w) * 2 + 1]);
					std::vector<ZQ_Vec2D> current_pts;
					current_pts.push_back(currentpt);
					std::vector<ZQ_Vec2D> subpxpt;
					std::vector<bool> goodpt, badpt;
					_subpixcrnr(current_pts, img, win, win, subpxpt, goodpt, badpt);
					gridout_data[(h*grid_width + w) * 2 + 0] = subpxpt[0].x;
					gridout_data[(h*grid_width + w) * 2 + 1] = subpxpt[0].y;
					bad_data[h*grid_width + w] = badpt[0];
				}
			}


			std::vector<int> badpts_row, badpts_col;
			for (int h = 0; h < grid_height; h++)
			{
				for (int w = 0; w < grid_width; w++)
				{
					if (bad_data[h*grid_width + w])
					{
						badpts_row.push_back(h);
						badpts_col.push_back(w);
					}
				}
			}
			nobadpts = badpts_row.size();

			ZQ_DImage<T> old_gridout(gridout);
			_getgoodsubrect(old_gridout, badpts_row, badpts_col, gridout);
		}

		template<class T>
		static void _subpixcrnr(const std::vector<ZQ_Vec2D>& xt, const ZQ_DImage<T>& I, int win_width, int win_height, std::vector<ZQ_Vec2D>& xc, std::vector<bool>& good, std::vector<bool>& bad)
		{
			/*
			function [xc,good,bad,type] = subpixcrnr(xt,I,wintx,winty,wx2,wy2)
			% SUBPIXCRNR gets the subpixel position of the chessboard corners.
			%
			% SUBPIXCRNR Finds the sub-pixel corners on the image I with initial guess xt
			% xt and xc are 2xN matrices. The first component is the x coordinate
			% (horizontal) and the second component is the y coordinate (vertical).
			%
			% USAGE:
			%     [xc] = subpixcrnr(xt,I);
			%
			% Based on Harris corner finder method.
			%
			% Finds corners to a precision below .1 pixel!
			% Oct. 14th, 1997 - UPDATED to work with vertical and horizontal edges as well!!!
			% Sept 1998 - UPDATED to handle diverged points: we keep the original points
			% good is a binary vector indicating wether a feature point has been properly
			% found.
			%
			% Add a zero zone of size wx2,wy2
			% July 15th, 1999 - Bug on the mask building... fixed + change to Gaussian mask with higher
			% resolution and larger number of iterations.
			%
			% California Institute of Technology
			% (c) Jean-Yves Bouguet -- Oct. 14th, 1997

			% double precision
			% I=double(I);
			*/

			bool line_feat = 1; // set to 1 to allow for extraction of line features.

			ZQ_DImage<T> mask(2 * win_width + 1, 2 * win_height + 1);
			T*& mask_data = mask.data();
			for (int h = 0; h < win_height * 2 + 1; h++)
			{
				for (int w = 0; w < win_width * 2 + 1; w++)
				{
					double hh = (h - win_height) / win_height;
					double ww = (w - win_width) / win_width;
					mask_data[h*(win_width * 2 + 1) + w] = exp(-hh*hh) * exp(-ww*ww);
				}
			}

			// another mask :
			ZQ_DImage<T> mask2(2 * win_width + 1, 2 * win_height + 1);
			T*& mask2_data = mask2.data();
			for (int h = 0; h < win_height * 2 + 1; h++)
			{
				for (int w = 0; w < win_width * 2 + 1; w++)
				{
					double hh = h - win_height;
					double ww = w - win_width;
					if (hh == 0 && ww == 0)
					{
						mask2_data[h*(win_width * 2 + 1) + w] = 1;
					}
					else
					{
						mask2_data[h*(win_width * 2 + 1) + w] = 1.0 / (hh*hh + ww*ww);
					}
				}
			}

			ZQ_DImage<int> off_h(2 * win_width + 1, 2 * win_height + 1);
			ZQ_DImage<int> off_w(2 * win_width + 1, 2 * win_height + 1);
			int*& off_h_data = off_h.data();
			int*& off_w_data = off_w.data();
			for (int h = 0; h < 2 * win_height + 1; h++)
			{
				for (int w = 0; w < 2 * win_width + 1; w++)
				{
					off_h_data[h*(2 * win_width + 1) + w] = h - win_height;
					off_w_data[h*(2 * win_width + 1) + w] = w - win_width;
				}
			}

			double resolution = 0.005;
			int MaxIter = 10;

			int im_height = I.height();
			int im_width = I.width();
			int im_nChannels = I.nchannels();
			const T*& I_data = I.data();

			int N = xt.size();
			xc = xt; // first guess... they don't move !!!
			std::vector<int> type(N);
			for (int i = 0; i < N; i++)
				type[i] = 0;

			for (int i = 0; i < N; i++)
			{
				ZQ_Vec2D v_extra(resolution + 1, 0); //just larger than resolution
				int compt = 0; //no iteration yet
				while (v_extra.Length() > resolution && compt < MaxIter)
				{
					double cIx = xc[i].x;
					double cIy = xc[i].y;
					int crIx = cIx + 0.5;
					int crIy = cIy + 0.5;
					double itIx = cIx - crIx;
					double itIy = cIy - crIy;
					T vIx[3];
					T vIy[3];
					if (itIx > 0)
					{
						vIx[2] = itIx;
						vIx[1] = 1.0 - itIx;
						vIx[0] = 0;
					}
					else
					{
						vIx[2] = itIx;
						vIx[1] = 1.0 + itIx;
						vIx[0] = -itIx;
					}
					if (itIy > 0)
					{
						vIy[2] = itIy;
						vIy[1] = 1.0 - itIy;
						vIy[0] = 0;
					}
					else
					{
						vIy[2] = 0;
						vIy[1] = 1.0 + itIy;
						vIy[0] = -itIy;
					}

					// What if the sub image is not in ?
					int xmin, xmax, ymin, ymax;
					if (crIx - win_width - 2 < 0)
					{
						xmin = 0;
						xmax = 2 * win_width + 4;
					}
					else if (crIx + win_width + 2 > im_width - 1)
					{
						xmax = im_width - 1;
						xmin = im_width - 1 - 2 * win_width - 4;
					}
					else
					{
						xmin = crIx - win_width - 2;
						xmax = crIx + win_width + 2;
					}

					if (crIy - win_height - 2 < 1)
					{
						ymin = 0;
						ymax = 2 * win_height + 4;
					}
					else if (crIy + win_height + 2 > im_height - 1)
					{
						ymax = im_height - 1;
						ymin = im_height - 1 - 2 * win_height - 4;
					}
					else
					{
						ymin = crIy - win_height - 2;
						ymax = crIy + win_height + 2;
					}

					int SI_width = xmax - xmin + 1;
					int SI_height = ymax - ymin + 1;
					ZQ_DImage<T> SI(SI_width, SI_height, im_nChannels);
					T*& SI_data = SI.data();
					for (int h = 0; h < SI_height; h++)
					{
						memcpy(SI_data + h*SI_width*im_nChannels, I_data + ((h + ymin)*im_width + xmin)*im_nChannels, sizeof(T)*SI_width*im_nChannels);
					}
					ZQ_DImage<T> tmp_SI(SI_width, SI_height, im_nChannels);
					T*& tmp_SI_data = tmp_SI.data();
					ZQ_ImageProcessing::Hfiltering(SI_data, tmp_SI_data, SI_width, SI_height, im_nChannels, vIx, 1, false);
					ZQ_ImageProcessing::Vfiltering(tmp_SI_data, SI_data, SI_width, SI_height, im_nChannels, vIy, 1, false);
					SI.swap(tmp_SI);

					SI.allocate(2 * win_width + 3, 2 * win_height + 3, im_nChannels);
					for (int h = 0; h < 2 * win_height + 3; h++)
					{
						memcpy(SI_data + h*(2 * win_width + 3)*im_nChannels, tmp_SI_data + ((h + 1)*SI_width + 1)*im_nChannels, sizeof(T)*(2 * win_width + 3)*im_nChannels);
					}

					ZQ_DImage<T> tmp_gx, tmp_gy;
					SI.dx_3pt(tmp_gx);
					SI.dy_3pt(tmp_gy);
					ZQ_DImage<T> gx(2 * win_width + 1, 2 * win_height + 1, im_nChannels);
					ZQ_DImage<T> gy(2 * win_width + 1, 2 * win_height + 1, im_nChannels);
					int tmp_g_width = tmp_gx.width();
					int tmp_g_height = tmp_gx.height();
					for (int h = 0; h < 2 * win_height + 1; h++)
					{
						memcpy(gx.data() + h*(2 * win_width + 1)*im_nChannels, tmp_gx.data() + ((h + 1)*tmp_g_width + 1)*im_nChannels, sizeof(T)*(2 * win_width + 1)*im_nChannels);
						memcpy(gy.data() + h*(2 * win_width + 1)*im_nChannels, tmp_gy.data() + ((h + 1)*tmp_g_width + 1)*im_nChannels, sizeof(T)*(2 * win_width + 1)*im_nChannels);
					}


					ZQ_DImage<T> px(2 * win_width + 1, 2 * win_height + 1);
					ZQ_DImage<T> py(2 * win_height + 1, 2 * win_height + 1);
					ZQ_DImage<T> gxx(2 * win_height + 1, 2 * win_height + 1);
					ZQ_DImage<T> gyy(2 * win_height + 1, 2 * win_height + 1);
					ZQ_DImage<T> gxy(2 * win_height + 1, 2 * win_height + 1);
					T*& px_data = px.data();
					T*& py_data = py.data();
					T*& gxx_data = gxx.data();
					T*& gyy_data = gyy.data();
					T*& gxy_data = gxy.data();
					gxx.Multiply(gx, gx, mask);
					gyy.Multiply(gy, gy, mask);
					gxy.Multiply(gx, gy, mask);

					int win_Xsize = 2 * win_width + 1;
					int win_Ysize = 2 * win_height + 1;
					double bb[2] = { 0 };
					double a = 0, b = 0, c = 0;
					for (int pp = 0; pp < win_Xsize*win_Ysize; pp++)
					{
						a += gxx_data[pp];
						b += gxy_data[pp];
						c += gyy_data[pp];
						double px = cIx + off_w_data[pp];
						double py = cIy + off_h_data[pp];
						bb[0] += gxx_data[pp] * px + gxy_data[pp] * py;
						bb[1] += gxy_data[pp] * px + gyy_data[pp] * py;
					}

					double dt = a*c - b*b;
					double xc2[2] = { (c*bb[0] - b*bb[1]) / dt, (a*bb[1] - b*bb[0]) / dt };

					if (line_feat)
					{
						double G[4] = { a, b, b, c };
						double U[4], S[4], V[4];
						ZQ_MathBase::SVD_Decompose(G, 2, 2, U, S, V);
						double xci_xc2[2] = { xc[i].x - xc2[0], xc[i].y - xc2[1] };
						xci_xc2[0] *= V[1];
						xci_xc2[1] *= V[3];
						double sum = xci_xc2[0] + xci_xc2[1];
						xc2[0] += sum*V[1];
						xc2[1] += sum*V[3];
						type[i] = 1;
					}

					if (isnan(xc2[0]) || isnan(xc2[1]))
						xc2[0] = xc2[1] = 0;

					v_extra.x = xc[i].x - xc2[0];
					v_extra.y = xc[i].y - xc2[1];
					xc[i].x = xc2[0];
					xc[i].y = xc2[1];

					compt++;

				}
			}

			// check for points that diverge :

			bad.resize(N);
			good.resize(N);
			for (int i = 0; i < N; i++)
			{
				bad[i] = fabs(xc[i].x - xt[i].x) > win_width || fabs(xc[i].y - xt[i].y) > win_height;
				good[i] = !bad[i];
				if (bad[i])
					xc[i] = xt[i];
			}
		}
		
		template<class T>
		static void _getgoodsubrect(const ZQ_DImage<T>& grid, const std::vector<int>& badpts_row, const std::vector<int>& badpts_col, ZQ_DImage<T>& gridout)
		{
			/*
			function gridout=getgoodsubrect(grid,badptsx,badptsy)
			%GETGOODSUBRECT extracts a grid subset of the input grid such that the output grid does not contain any bad points.
			%
			% GETGOODSUBRECT takes as input a MxNx2 matrix acting a chessboard grid.
			% It also takes two vectors containing the x and y coordinates of the bad
			% points. The two vectors must be the same size.
			%
			% The result returned is suboptimal. Optimal algorithms introduce
			% complexity beyond our basic requirements.
			%
			% USAGE:
			%     gridout=getgoodsubrect(grid,badptsx,badptsy);
			%
			% INPUTS:
			%     grid: MxNx2 chessboard grid
			%
			%     badptsx: x coordinates of the bad points
			%
			%     badptsy: y coordinates of the bad points
			%
			% OUTPUTS:
			%     gridout: a subset of the input grid without any bad points
			*/

			int grid_width = grid.width();
			int grid_height = grid.height();
			const T*& grid_data = grid.data();

			int hmin = 0;
			int hmax = grid_height - 1;
			int wmin = 0;
			int wmax = grid_width - 1;
			
			for (int i = 0; i < badpts_row.size(); i++)
			{
				int dist[4] = { badpts_row[i], badpts_col[i], grid_height - 1 - badpts_row[i], grid_width - 1 - badpts_col[i] };
				int minval = dist[0];
				int minidx = 0;
				for (int j = 1; j < 4; j++)
				{
					if (minval > dist[j])
					{
						minval = dist[j];
						minidx = j;
					}
				}

				switch (minidx)
				{
				case 0:
					if (badpts_row[i] >= hmin)
						hmin = badpts_row[i] + 1;
					break;
				case 1:
					if (badpts_col[i] >= wmin)
						wmin = badpts_col[i] + 1;
					break;
				case 2:
					if (badpts_row[i] <= hmax)
						hmax = badpts_row[i] - 1;
					break;
				case 3:
					if (badpts_col[i] <= wmax)
						wmax = badpts_col[i] - 1;
					break;
				default:
					break;
				}
			}
			
			int gridout_width = wmax - wmin + 1;
			int gridout_height = hmax - hmin + 1;
			gridout.allocate(grid_width, grid_height, 2);
			T*& gridout_data = gridout.data();
			for (int h = 0; h < gridout_height; h++)
			{
				memcpy(gridout_data + h*gridout_width * 2, grid_data + ((h + hmin)*grid_width + wmin) * 2, sizeof(T)*gridout_width * 2);
			}
		}
		
		template<class T>
		static void _adjgridorigin(const ZQ_DImage<T>& grid, ZQ_DImage<T>& gridout)
		{
			/*
			function gridout=adjgridorigin(grid)
			% ADJGRIDORIGIN adjusts the input grid's origin to ensure consistency among images.
			%
			% ADJGRIDORIGIN takes as input the grid from GETMISSCRNRS and returns a the
			% grid with the origin adjusted if necessary. If no adjustment is needed
			% the same grid is returned.
			%
			% INPUTS:
			%     grid: output of GETMISSCRNRS
			%
			% OUTPUTS:
			%     gridout: adjusted grid
			*/
			
			int width = grid.width();
			int height = grid.height();
			const T*& grid_data = grid.data();
			
			double x1 = grid_data[0];
			double y1 = grid_data[1];
			double x2 = grid_data[((height - 1)*width + width - 1) * 2 + 0];
			double y2 = grid_data[((height - 1)*width + width - 1) * 2 + 1];
			if (x1*x1 + y1*y1 > x2*x2 + y2*y2)
			{
				gridout.allocate(width, height, 2);
				T*& gridout_data = gridout.data();
				int gridout_width = gridout.width();

				for (int h = 0; h < height; h++)
				{
					for (int w = 0; w < width; w++)
					{
						gridout_data[((height - 1 - h)*gridout_width + width-1-w) * 2 + 0] = grid_data[(h*width + w) * 2 + 0];
						gridout_data[((height - 1 - h)*gridout_width + width-1-w) * 2 + 1] = grid_data[(h*width + w) * 2 + 1];
					}
				}
			}
			else
			{
				gridout = grid;
			}
		}
		

		template<class T>
		static void _getmisscrnrs(const ZQ_DImage<T>& grid, int im_width, int im_height, ZQ_DImage<T>& gridout, int& nointerpolations)
		{
			/*
			function [gridout,nointerpolations]=getmisscrnrs(grid,imsize,debug)
			% GETMISSCRNRS fills in the gaps in the grid by linear interpolation.
			%
			% GETMISSCRNRS successively inerpolates missing points by fitting a least
			% square line of the row and column the missing point belongs to and then
			% storing the point as the intersection of the two lines.
			%
			% USAGE:
			%     [gridout,nointerpolations]=getmisscrnrs(grid);
			%
			% INPUTS:
			%     grid: output of FILTERGRID
			%
			% OUTPUTS:
			%     gridout: grid with the missing corners interpolated
			%
			%     nointerpolations: the number of interpolations is used to detect the
			%     quality of the chessboard detection by FINDCORNERS
			*/

			//Function that interpolates the positions of missing corners
			gridout = grid;
			nointerpolations = 0;
			T*& gridout_data = gridout.data();

			// list of interpolations that lie outside the image
			int grid_width = grid.width();
			int grid_height = grid.height();
			ZQ_DImage<bool> outgrid(grid_width, grid_height);
			bool*& outgrid_data = outgrid.data();
			
			for (int h = 0; h < grid_height; h++)
			{
				for (int w = 0; w < grid_width; w++)
				{
					if (gridout_data[(h*grid_width + w) * 2 + 0] == 0)
					{
						std::vector<ZQ_Vec2D> nzerrow;
						std::vector<ZQ_Vec2D> nzercol;
						for (int i = 0; i < grid_width; i++)
						{
							double xval = gridout_data[(h*grid_width + i) * 2 + 0];
							double yval = gridout_data[(h*grid_width + i) * 2 + 1];
							if (xval > 0 && yval > 0)
							{
								nzerrow.push_back(ZQ_Vec2D(xval, yval));
							}
						}
						for (int i = 0; i < grid_height; i++)
						{
							double xval = gridout_data[(i*grid_width + w) * 2 + 0];
							double yval = gridout_data[(i*grid_width + w) * 2 + 1];
							if (xval > 0 && yval > 0)
							{
								nzercol.push_back(ZQ_Vec2D(xval, yval));
							}
						}
						if (nzerrow.size() < 2 || nzercol.size() < 2)
						{
							gridout.clear();
							return;
						}

						nointerpolations++;
						float p1, p1_1, p2, p2_1;
						_linefit(nzerrow, p1, p1_1);
						_linefit(nzercol, p2, p2_1);

						ZQ_Matrix<double> eqnsmat(2, 2), bmat(2,1), xmat(2,1);
						eqnsmat.SetData(0, 0, p1);
						eqnsmat.SetData(0, 1, -1);
						eqnsmat.SetData(1, 0, p2);
						eqnsmat.SetData(1, 1, -1);
						bmat.SetData(0, 0, -p1_1);
						bmat.SetData(1, 0, -p2_1);
						ZQ_SVD::Solve(eqnsmat, xmat, bmat);
						bool get_flag;
						double xmat_val1 = xmat.GetData(0, 0, get_flag);
						double xmat_val2 = xmat.GetData(1, 0, get_flag);
						gridout_data[(h*grid_width + w) * 2 + 0] = xmat_val1;
						gridout_data[(h*grid_width + w) * 2 + 1] = xmat_val2;
						if (xmat_val1 < 0 || xmat_val1 > im_width - 1 || xmat_val2 < 0 || xmat_val2 > im_height - 1)
						{
							//add to list for later processing
							outgrid_data[h*grid_width+w] = 1;
						}
					}
				}
			}
			

			// process outlist
			while (true)
			{
				bool has_found_flag = false;
				for (int i = 0; i < grid_width*grid_height; i++)
				{
					if (outgrid_data[i])
					{
						has_found_flag = true;
						break;
					}
				}
				if (!has_found_flag)
					break;

				int outgrid_width = outgrid.width();
				int outgrid_height = outgrid.height();

				int r, c;
				if (outgrid_data[0])
				{
					r = 0;
					c = 0;
				}
				else if (outgrid_data[outgrid_width - 1])
				{
					r = 0;
					c = outgrid_width - 1;
				}
				else if (outgrid_data[(outgrid_height-1)*outgrid_width])
				{
					r = outgrid_height - 1;
					c = 0;
				}
				else if (outgrid_data[(outgrid_height - 1)*outgrid_width + outgrid_width - 1])
				{
					r = outgrid_height - 1;
					c = outgrid_width - 1;
				}
				else
				{
					gridout.clear();
					return;
				}

				//remove row or column
				if (outgrid_width > outgrid_height)
				{
					//remove column
					ZQ_DImage<bool> old_outgrid(outgrid);
					ZQ_DImage<T> old_gridout(gridout);
					bool*& old_outgrid_data = old_outgrid.data();
					T*& old_gridout_data = old_gridout.data();
					outgrid.allocate(outgrid_width - 1, outgrid_height);
					gridout.allocate(outgrid_width - 1, outgrid_height, 2);
					int cidx = 0;
					for (int w = 0; w < outgrid_width; w++)
					{
						if (w == c)
							continue;

						for (int h = 0; h < outgrid_height; h++)
						{
							outgrid_data[h*(outgrid_width - 1) + cidx] = old_outgrid_data[h*outgrid_width + w];
							gridout_data[(h*(outgrid_width - 1) + cidx) * 2 + 0] = old_gridout_data[(h*outgrid_width + w) * 2 + 0];
							gridout_data[(h*(outgrid_width - 1) + cidx) * 2 + 1] = old_gridout_data[(h*outgrid_width + w) * 2 + 1];
						}
						cidx++;
					}
				}
				else
				{
					//remove row
					ZQ_DImage<bool> old_outgrid(outgrid);
					ZQ_DImage<T> old_gridout(gridout);
					bool*& old_outgrid_data = old_outgrid.data();
					T*& old_gridout_data = old_gridout.data();
					outgrid.allocate(outgrid_width, outgrid_height-1);
					gridout.allocate(outgrid_width, outgrid_height-1, 2);
					int ridx = 0;
					for (int h = 0; h < outgrid_height; h++)
					{
						if (h == r)
							continue;
						memcpy(outgrid_data + ridx*outgrid_width, old_outgrid_data + h*outgrid_width, sizeof(bool)*outgrid_width);
						memcpy(gridout_data + ridx*outgrid_width * 2, old_gridout_data + h*outgrid_width * 2, sizeof(T)*outgrid_width * 2);
						ridx++;
					}
				}
			}
		}
		

		template<class T>
		static void _adjgriddir(const ZQ_DImage<T>& grid, ZQ_DImage<T>& gridout)
		{
			/*function gridout=adjgriddir(grid)
			% ADJGRIDDIR adjusts the direction of the extracted grid to ensure consistency across all images.
			%
			% ADJGRIDDIR examines the direction of the grid rows and columns. It then
			% consequently adjusts the grid to a consistent direction.
			%
			% USAGE:
			%     gridout=adjgriddir(grid);
			%
			% INPUTS:
			%     grid: MxNx2 array, output of FILTERGRID
			%
			% OUTPUTS:
			%     gridout: adjusted MxNx2 or NxMx2 array
			*/

			int width = grid.width();
			int height = grid.height();
			const T*& grid_data = grid.data();

			std::vector<double> rowslope(height);
			std::vector<double> colslope(width);
			for (int i = 0; i < height; i++)
				rowslope[i] = 0;
			for (int i = 0; i < width; i++)
				colslope[i] = 0;

			// get slopes of rows
			for (int rowindx = 0; rowindx < height; rowindx++)
			{
				std::vector<ZQ_Vec2D> cur_pts;
				for (int j = 0; j < width; j++)
				{
					T xval = grid_data[(rowindx*width + j) * 2 + 0];
					T yval = grid_data[(rowindx*width + j) * 2 + 1];
					if (xval > 0 && yval > 0)
					{
						cur_pts.push_back(ZQ_Vec2D(xval, yval));
					}
				}
				float p1, p2;
				_linefit(cur_pts, p1, p2);
				rowslope[rowindx] = fabs(p1);
			}
			

			//get slopes of columns
			for (int colindx = 0; colindx < width; colindx++)
			{
				std::vector<ZQ_Vec2D> cur_pts;
				for (int i = 0; i < height; i++)
				{
					T xval = grid_data[(i*width + colindx) * 2 + 0];
					T yval = grid_data[(i*width + colindx) * 2 + 1];
					if (xval > 0 && yval > 0)
					{
						cur_pts.push_back(ZQ_Vec2D(xval, yval));
					}
				}
				float p1, p2;
				_linefit(cur_pts, p1, p2);
				colslope[colindx] = fabs(p1);
			}
		
			double mean_rowslope = 0;
			double mean_colslope = 0;
			for (int i = 0; i < height; i++)
				mean_rowslope += rowslope[i];
			mean_rowslope /= height;
			for (int i = 0; i < width; i++)
				mean_colslope += colslope[i];
			mean_colslope /= width;

			// adjust for the higher slope
			if (mean_colslope < mean_rowslope)
			{
				gridout.allocate(height, width, 2);
				T*& gridout_data = gridout.data();
				int gridout_width = gridout.width();
				
				for (int h = 0; h < height; h++)
				{
					for (int w = 0; w < width; w++)
					{
						gridout_data[((width - 1 - w)*gridout_width + h) * 2 + 0] = grid_data[(h*width + w) * 2 + 0];
						gridout_data[((width - 1 - w)*gridout_width + h) * 2 + 1] = grid_data[(h*width + w) * 2 + 1];
					}
				}
			}
			else
			{
				gridout = grid;
			}
		}


		static void _linefit(const std::vector<ZQ_Vec2D>& cur_pts, float& p1, float& p2)
		{
			p1 = p2 = 0;
			int N = cur_pts.size();
			if (N == 0 || N == 1)
				return;
			bool x_not_same = false;
			bool y_not_same = false;
			for (int i = 1; i < N; i++)
			{
				if (cur_pts[i].x != cur_pts[0].x)
				{
					x_not_same = true;
					break;
				}
			}
			for (int i = 1; i < N; i++)
			{
				if (cur_pts[i].y != cur_pts[0].y)
				{
					y_not_same = true;
					break;
				}
			}

			if (x_not_same)
			{
				ZQ_Matrix<double> A(N, 2), x(2, 1), b(N, 1);
				for (int i = 0; i < N; i++)
				{
					A.SetData(i, 0, cur_pts[i].x);
					A.SetData(i, 1, 1);
					b.SetData(i, 0, cur_pts[i].y);
				}
				ZQ_SVD::Solve(A, x, b);
				bool flag;
				p1 = x.GetData(0, 0, flag);
				p2 = x.GetData(1, 0, flag);
			}
			else if (y_not_same)
			{
				p1 = 1e32;
				p2 = 0;
			}
			else
			{
				p1 = 1;
				p2 = 0;
			}
		}

		template<class T>
		static void _filtergrid(const ZQ_DImage<T>& grid, ZQ_DImage<T>& gridout)
		{
			/*
			% FILTERGRID removes spur rows and columns existing after the grid arrangement process.
			%
			% FITLERGRID processes the input grid and iteratively removes rows and
			% columns until a rectangular grid is obtained.
			%
			% INPUTS:
			%     grid: MxNx2 matrix output by GETGRID
			%
			% OUTPUTS:
			%     gridout: VxWx2 matrix containing the filtered grid
			*/

			
			gridout.copyData(grid);
			
			
			while (true)
			{
				
				double row1count = 0;
				double row2count = 0;
				double col1count = 0;
				double col2count = 0;
				int width = gridout.width();
				int height = gridout.height();
				T*& gridout_data = gridout.data();
				double rowthresh = height / 2.0;
				double colthresh = width / 2.0;

				

				for (int x = 0; x < width; x++)
				{
					if (gridout_data[x * 2] != 0)
						row1count += 1;
					if (gridout_data[((height - 1)*width + x) * 2] != 0)
						row2count += 1;
				}
				for (int y = 0; y < height; y++)
				{
					if (gridout_data[(y*width) * 2] != 0)
						col1count += 1;
					if (gridout_data[(y*width + width - 1) * 2] != 0)
						col2count += 1;
				}

				/*bool debug = false;
				if (row1count != 0 || row2count != 0 || col1count != 0 || col2count != 0)
					debug = true;
				if (debug)
				{
					ZQ_DImage<T> old_grid_x, old_grid_y;
					gridout.separate(1, old_grid_x, old_grid_y);
					old_grid_x.imresize(20);
					old_grid_y.imresize(20);
					old_grid_x.AutoAdjust();
					old_grid_y.AutoAdjust();
					ZQ_ImageIO::Show("x", old_grid_x);
					ZQ_ImageIO::Show("y", old_grid_y);
					cvWaitKey(0);
				}*/
				
				
				row1count -= rowthresh;
				row2count -= rowthresh;
				col1count -= colthresh;
				col2count -= colthresh;

				//remove row or column with the least number of points

				
				double counts[4] = { row1count, row2count, col1count, col2count };
				double mincount = counts[0];
				int indx = 0;
				for (int ppp = 1; ppp < 4; ppp++)
				{
					if (mincount > counts[ppp])
					{
						mincount = counts[ppp];
						indx = ppp;
					}
				}

				if (mincount < 0)
				{
					ZQ_DImage<T> old_gridout(gridout);
					T*& old_gridout_data = old_gridout.data();
					int old_gridout_width = old_gridout.width();
					int old_gridout_height = old_gridout.height();
					T*& new_gridout_data = gridout.data();
					switch (indx)
					{
					case 0:
						gridout.allocate(old_gridout_width, old_gridout_height - 1, 2);
						for (int h = 0; h < old_gridout_height - 1; h++)
						{
							memcpy(new_gridout_data + h*old_gridout_width * 2, old_gridout_data + (h + 1)*old_gridout_width * 2, sizeof(T)*old_gridout_width * 2);
						}
						break;
					case 1:
						gridout.allocate(old_gridout_width, old_gridout_height - 1, 2);
						for (int h = 0; h < old_gridout_height - 1; h++)
						{
							memcpy(new_gridout_data + h*old_gridout_width * 2, old_gridout_data + h*old_gridout_width * 2, sizeof(T)*old_gridout_width * 2);
						}
						break;
					case 2:
						gridout.allocate(old_gridout_width - 1, old_gridout_height, 2);
						for (int h = 0; h < old_gridout_height; h++)
						{
							memcpy(new_gridout_data + h*(old_gridout_width - 1) * 2, old_gridout_data + (h*old_gridout_width + 1) * 2, sizeof(T)*(old_gridout_width - 1) * 2);
						}
						break;
					case 3:
						gridout.allocate(old_gridout_width - 1, old_gridout_height, 2);
						for (int h = 0; h < old_gridout_height; h++)
						{
							memcpy(new_gridout_data + h*(old_gridout_width - 1) * 2, old_gridout_data + h*old_gridout_width * 2, sizeof(T)*(old_gridout_width - 1) * 2);
						}
						break;
					}
				}
				else
					break;
			}
		}
		
		template<class T>
		static void _adaptimadj(const ZQ_DImage<T>& in_img, ZQ_DImage<T>& out_img)
		{
			/*ADAPTIMADJ adaptively adjusts the intensity of an image
			
			ADAPTIMADJ adaptively adjusts the intensity of the input image.The
			adjustment is performed according to the mean and standard deviation of
			the region around each pixel.
			
			 USAGE:
			    imgout = adaptimadj(img); default window size is min(size(img)) / 5,
			     default th is 1.
			
			 INPUTS:
			    img : grayscale double image
			    win : size of the rectangular region to inspect at each pixel
				th : parameters that tunes the degree of adjustment
			 OUTPUTS :
				imgout : adjusted image*/

			int width = in_img.width();
			int height = in_img.height();
			if (!out_img.matchDimension(width, height, 1))
				out_img.allocate(width, height);
			

			int win = __min(width / 5.0 + 0.5, height / 5.0 + 0.5);
			float th = 1;

			ZQ_DImage<T> mimg, stdv;
			_adaptstats(in_img, win, mimg, stdv);

			const T*& in_img_data = in_img.data();
			T*& mimg_data = mimg.data();
			T*& stdv_data = stdv.data();
			T*& out_img_data = out_img.data();
			for (int i = 0; i < width*height; i++)
			{
				float imax = mimg_data[i] + th * stdv_data[i];
				float imin = mimg_data[i] - th * stdv_data[i];
				imax = imax > 1 ? 1 : imax;
				imin = imin < 0 ? 0 : imin;
				if (imax == imin)
					out_img_data[i] = 0;
				else
				{
					out_img_data[i] = (in_img_data[i] - imin) / (imax - imin);
					out_img_data[i] = out_img_data[i] > 1 ? 1 : out_img_data[i];
					out_img_data[i] = out_img_data[i] < 0 ? 0 : out_img_data[i];
				}
			}
		}

		template<class T>
		static void _adaptstats(const ZQ_DImage<T>& img, int win, ZQ_DImage<T>& mimg, ZQ_DImage<T>& stdv)
		{
			/*ADAPTSTATS gets the local mean and standard deviations for each pixel in the image within a region.
			USAGE:
			[mimg, stdv] = adaptstats(img, win);
			[mimg, stdv] = adaptstats(img);
			
			INPUTS:
			    img : grayscale image of class double
				win : (optional) default value is min(size(img)) / 5, win is the size of
				    the region
			*/

			
			//% default win size
			//win = round(min(size(img)) / 5);
			
			int hwin = floor(win / 2.0);
			int width = img.width();
			int height = img.height();
			int new_width = width + hwin * 2 + 1;
			int new_height = height + hwin * 2 + 1;

			ZQ_DImage<T> tmp1(width, height), tmp2(width,height);
			ZQ_DImage<T> intimg(new_width, new_height);
			ZQ_DImage<T> intimg2(new_width, new_height);
			
			const T*& img_data = img.data();
			T*& tmp1_data = tmp1.data();
			T*& tmp2_data = tmp2.data();
			T*& intimg_data = intimg.data();
			T*& intimg2_data = intimg2.data();

			_cumsum(img, 1, tmp1);
			_cumsum(tmp1, 2, tmp2);
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
					intimg_data[(h + hwin + 1)*new_width + w + hwin + 1] = tmp2_data[h*width + w];
				for (int w = 0; w < hwin; w++)
					intimg_data[(h + hwin + 1)*new_width + width + hwin + 1 + w] = tmp2_data[h*width + width - 1];
			}
			for (int h = 0; h < hwin; h++)
			{
				for (int w = 0; w < new_width; w++)
					intimg_data[(height + hwin + 1 + h)*new_width + w] = intimg_data[(height + hwin)*new_width + w];
			}


			for (int i = 0; i < width*height; i++)
			{
				tmp2_data[i] = img_data[i] * img_data[i];
			}
			_cumsum(tmp2, 1, tmp1);
			_cumsum(tmp1, 2, tmp2);
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
					intimg2_data[(h + hwin + 1)*new_width + w + hwin + 1] = tmp2_data[h*width + w];
				for (int w = 0; w < hwin; w++)
					intimg2_data[(h + hwin + 1)*new_width + width + hwin + 1 + w] = tmp2_data[h*width + width - 1];
			}
			for (int h = 0; h < hwin; h++)
			{
				for (int w = 0; w < new_width; w++)
					intimg2_data[(height + hwin + 1 + h)*new_width + w] = intimg2_data[(height + hwin)*new_width + w];
			}

			/*ZQ_DImage<T> tmp_intimg(intimg), tmp_intimg2(intimg2);
			tmp_intimg.AutoAdjust();
			tmp_intimg2.AutoAdjust();
			ZQ_ImageIO::Show("intimg1", tmp_intimg);
			ZQ_ImageIO::Show("intimg2", tmp_intimg2);
			cvWaitKey(0);*/

			if (!mimg.matchDimension(width, height, 1))
				mimg.allocate(width, height, 1);
			//else
			//	mimg.reset();
			if (!stdv.matchDimension(width, height, 1))
				stdv.allocate(width, height, 1);
			//else
			//	stdv.reset();

			T*& mimg_data = mimg.data();
			T*& stdv_data = stdv.data();
			int n = win*win;
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					double sum1 = intimg_data[h*new_width + w] + intimg_data[(win + h)*new_width + win + w] - intimg_data[(win + h)*new_width + w] - intimg_data[h*new_width + win + w];
					mimg_data[h*width + w] = sum1 / n;
					double sum2 = intimg2_data[h*new_width + w] + intimg2_data[(win + h)*new_width + win + w] - intimg2_data[(win + h)*new_width + w] - intimg2_data[h*new_width + win + w];
					double vari = (sum2 - (mimg_data[h*width + w] * mimg_data[h*width + w])*n) / (n - 1);
					stdv_data[h*width + w] = sqrt(vari);
				}
			}
			/*ZQ_DImage<T> tmp_mimg(mimg), tmp_stdv(stdv);
			tmp_mimg.AutoAdjust();
			tmp_stdv.AutoAdjust();
			ZQ_ImageIO::Show("mimg", tmp_mimg);
			ZQ_ImageIO::Show("stdv", tmp_stdv);
			cvWaitKey(0);*/

		}

		template<class T>
		static void _cumsum(const ZQ_DImage<T>& in_img, int dim, ZQ_DImage<T>& out_img)
		{
			//Y = CUMSUM(X, DIM) cumulates along the dimension specified by DIM.
			int width = in_img.width();
			int height = in_img.height();
			if (!out_img.matchDimension(width, height, 1))
				out_img.allocate(width, height, 1);
			else
				out_img.reset();

			const T*& in_img_data = in_img.data();
			T*& out_img_data = out_img.data();
			if (dim == 1)
			{
				for (int w = 0; w < width; w++)
				{
					T sum = 0;
					for (int h = 0; h < height; h++)
					{
						sum += in_img_data[h*width + w];
						out_img_data[h*width + w] = sum;
					}
				}
			}
			else
			{
				for (int h = 0; h < height; h++)
				{
					T sum = 0;
					for (int w = 0; w < width; w++)
					{
						sum += in_img_data[h*width + w];
						out_img_data[h*width + w] = sum;
					}
				}
			}
		}

		template<class T>
		static void _getedges(const ZQ_DImage<T>& in_img, ZQ_DImage<T>& edge, ZQ_DImage<T>& ix, ZQ_DImage<T>& iy)
		{
			/*function[imge, ix, iy] = getedges(img)
			% GETEDGES gets the Sobel edge image.
			%
			% USAGE:
			%     imge = getedges(img)
			%
			%[imge, ix, iy] = getedges(img)
			%
			% INPUTS :
			%     img : grayscale double image
			%
			% OUTPUTS :
			%     imge : the edge image
			%
			%     ix : x component gradient image
			%
			%     iy : y component gradient image*/

			T xfilter2D[9] = 
			{
				1, 0, -1,
				2, 0, -2,
				1, 0, -1
			};

			T yfilter2D[9] =
			{
				1, 2, 1,
				0, 0, 0,
				-1, -2, -1
			};
				
			int width = in_img.width();
			int height = in_img.height();
			
			if (!edge.matchDimension(width, height, 1))
				edge.allocate(width, height, 1);
			if (!ix.matchDimension(width, height, 1))
				ix.allocate(width, height, 1);
			if (!iy.matchDimension(width, height, 1))
				iy.allocate(width, height, 1);

			const T*& in_img_data = in_img.data();
			T*& edge_data = edge.data();
			T*& ix_data = ix.data();
			T*& iy_data = iy.data();
			
			//ZQ_ImageProcessing::ImageFilter2D(in_img_data, ix_data, width, height, 1, xfilter2D,1,1);
			//ZQ_ImageProcessing::ImageFilter2D(in_img_data, iy_data, width, height, 1, yfilter2D,1,1);
			ZQ_ImageProcessing::ImageFilter2D_3x3_1channel(in_img_data, ix_data, width, height, xfilter2D, false);
			ZQ_ImageProcessing::ImageFilter2D_3x3_1channel(in_img_data, iy_data, width, height, yfilter2D, false);
			
			for (int i = 0; i < width*height; i++)
				edge_data[i] = ix_data[i] * ix_data[i] + iy_data[i] * iy_data[i];
			edge.AutoAdjust();
		}

		template<class T>
		static void _harristm(const ZQ_DImage<T>& in_img, int win, ZQ_DImage<T>& out_img)
		{
			/*function imgout = harristm(img, win)
			% HARRIS obtains the Harris transform of image.
			%
			% HARRIS takes gets the Harris transform image of an input grayscale image.
			%
			% USAGE:
			%     imgout = harris(img); if win is not specified the default value is
			%     used min(size(img)) / 140
			%
			%     imgout = harris(img, win); win is the window size of the Harris
			%     transform
			%
			% INPUTS:
			%     img : grayscale double class image
			%
			%     win : scalar specifying the window size
			%
			% OUTPUTS :
			%	  imgout : Harris transform image*/

			/*if ~exist('win', 'var') || isempty(win)
				win = round(min(size(img)) / 140);
			end*/

			T xfilter2D[9] = 
			{
				1, 0, -1,
				2, 0, -2,
				1, 0, -1
			};
			T yfilter2D[9] =
			{
				1, 2, 1,
				0, 0, 0,
				-1, -2, -1
			};

			int width = in_img.width();
			int height = in_img.height();

			ZQ_DImage<T> ix(width, height, 1);
			ZQ_DImage<T> iy(width, height, 1);
			const T*& in_img_data = in_img.data();
			T*& ix_data = ix.data();
			T*& iy_data = iy.data();
			ZQ_ImageProcessing::ImageFilter2D_3x3_1channel(in_img_data, ix_data, width, height, xfilter2D, false);
			ZQ_ImageProcessing::ImageFilter2D_3x3_1channel(in_img_data, iy_data, width, height, yfilter2D, false);

			if (!out_img.matchDimension(width, height, 1))
				out_img.allocate(width, height, 1);
			
			T*& out_img_data = out_img.data();

			int n = win*win;
			int hwin = win / 2;

			int new_width = width + 2*win;
			int new_height = height + 2*win;
			ZQ_DImage<T> pad_ix(new_width, new_height, 1);
			ZQ_DImage<T> pad_iy(new_width, new_height, 1);
			T*& pad_ix_data = pad_ix.data();
			T*& pad_iy_data = pad_iy.data();
			for (int h = 0; h < height; h++)
			{
				memcpy(pad_ix_data + (h + win)*new_width + win, ix_data + h*width, sizeof(T)*width);
				memcpy(pad_iy_data + (h + win)*new_width + win, iy_data + h*width, sizeof(T)*width);
			}
			
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					double a = 0, b = 0, c = 0;

					for (int i = -win + hwin + 1; i <= hwin; i++)
					{
						for (int j = -win + hwin + 1; j <= hwin; j++)
						{
							int x = w + j + win;
							int y = h + i + win;
							T cur_ix = pad_ix_data[y*new_width + x];
							T cur_iy = pad_iy_data[y*new_width + x];
							a += cur_ix*cur_ix;
							b += cur_iy*cur_iy;
							c += cur_ix*cur_iy;
						}
					}
					a /= n;
					b /= n;
					c /= n;
					out_img_data[h*width + w] = (a*b - c*c) / (a + b + 1e-16);
				}
			}

			out_img.AutoAdjust();
		}

		template<class T>
		static void _getcrnrpts(const ZQ_DImage<T>& imgh, const ZQ_DImage<T>& mimg, const ZQ_DImage<T>& stdv, float th, std::vector<ZQ_Vec2D>& crnrpts)
		{
			/*function crnrpts = getcrnrpts(imgh, mimg, stdv, th)

			GETCRNRPTS is a function that uses the mean and standard deviation images to adaptively obtain local maxima in an image.
			GETCRNRPTS can be applied on the Harris transform to obtain the Harris
				 corner points.
			
			USAGE:
			    crnrpts = getcrnrpts(imgh, mimg, stdv, th);
			
			INPUTS:
			    imgh : Harris transform of image
				mimg, stdv : output from ADAPTSTATS
				h : parameter to adjust thresholding
				
			OUTPUTS :
			    crnrpts : 2xN array with coordinates of corner points*/

			int width = imgh.width();
			int height = imgh.height();
			
			ZQ_DImage<T> tmp(width, height, 1);
			ZQ_DImage<T> imax(width, height, 1), imghl(width, height, 1), imghlf(width, height, 1);
			ZQ_DImage<bool> imghlf_b(width, height, 1);
			T*& imax_data = imax.data();
			T*& imghl_data = imghl.data();
			T*& imghlf_data = imghlf.data();
			T*& tmp_data = tmp.data();
			bool*& imghlf_b_data = imghlf_b.data();
			const T*& imgh_data = imgh.data();
			const T*& mimg_data = mimg.data();
			const T*& stdv_data = stdv.data();

			for (int i = 0; i < width*height; i++)
			{
				imax_data[i] = mimg_data[i] + th*stdv_data[i];
				imax_data[i] = __max(0, __min(1, imax_data[i]));
				imghl_data[i] = 0*imgh_data[i];
				if (imgh_data[i] > imax_data[i])
					imghl_data[i] = 1;
			}
			/*ZQ_ImageIO::Show("imghl", imghl);
			cvWaitKey(0);*/
			clock_t t1 = clock();
			ZQ_ImageProcessing::MedianFilter33_1channel(imghl_data, tmp_data, width, height, false);
			ZQ_ImageProcessing::MedianFilter33_1channel(tmp_data, imghlf_data, width, height, false);

			clock_t t2 = clock();
			//printf("medfilt :%.3f secs\n", 0.001*(t2 - t1));
			
			for (int i = 0; i < width*height; i++)
				imghlf_b_data[i] = imghlf_data[i] != 0;
			/*ZQ_ImageIO::Show("imghlf_b", imghlf_b);
			cvWaitKey(0);*/
			/*if (!ZQ_ImageIO::saveImage(imghlf_b, "imghlf_b.png"))
			{
				printf("failed to save %s\n", "imghlf_b.png");
			}*/
			ZQ_DImage<int> label(width, height, 1);
			int*& label_data = label.data();
			std::vector<int> area_size;
			
			
			ZQ_BinaryImageProcessing::BWlabel(imghlf_b_data, width, height, label_data, area_size);
			
			clock_t t3 = clock();
			//printf("bwlabel :%.3f secs\n", 0.001*(t3 - t2));
			
			/*ZQ_ImageIO::Show("label", label);
			cvWaitKey(0);*/
			int n = area_size.size();
			std::vector<double> meanx(n);
			std::vector<double> meany(n);
			std::vector<double> cnt(n);
			for (int i = 0; i < n; i++)
			{
				meanx[i] = 0;
				meany[i] = 0;
				cnt[i] = 0;
			}

			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					int offset = h*width + w;
					int cur_lab = label_data[offset];
					if (cur_lab > 0)
					{
						meanx[cur_lab - 1] += w;
						meany[cur_lab - 1] += h;
						cnt[cur_lab - 1] += 1;
					}
				}
			}

			crnrpts.resize(n);
			for (int i = 0; i < n; i++)
			{
				meanx[i] /= cnt[i];
				meany[i] /= cnt[i];
				crnrpts[i].x = (int)(meanx[i] + 0.5f);
				crnrpts[i].y = (int)(meany[i] + 0.5f);
			}
			
		}

		template<class T>
		static void _chesscornerfilter(const ZQ_DImage<T>& img, const ZQ_DImage<T>& imgedge, const std::vector<ZQ_Vec2D>& crnrpts, std::vector<ZQ_Vec2D>& crnrs, std::vector<std::vector<int>>& peaklocs)
		{
			/*function[crnrs, nocrnrs, peaklocs] = chesscornerfilter(img, imgedge, crnrpts, debug)
			CHESSCORNERFILTER filters Harris corners for chessboard corners.
			
			CHESSCORNERFILTER takes as input the image, the sobel edge image and
			the Harris corner points and outputs of the corner points that passed
			the filter the number of those points and the peak directions for each
			of the points.The peak direction is used by the grid extraction(refer
			to Bachelor Thesis by Abdallah Kassir 2009).
			The details of the filter are applied in the function VALIDCORNER
			
			USAGE:
			[crnrs, nocrnrs, peaklocs] = chesscornerfilter(img, imgedge, crnrpts, debug)
			
			INPUTS :
			    img : original grayscale image
			    imgedge : Sobel edge image
			    crnrpts : output of GETCRNRPTS
			OUTPUTS :
			    crnrs : 2xN array of corners that passed the filter
			    nocrnrs : number of corners found
			    peaklocs : required by GETGRID*/


			crnrs.clear();
			peaklocs.clear();

			const T*& img_data = img.data();
			const T*& imgedge_data = imgedge.data();
			int width = img.width();
			//get sweepmatrices, precalculation of these matrices allows for much faster program execution
			
			clock_t t1 = clock();
			ZQ_DImage<int> sweepmatx, sweepmaty;
			_sweepmatrix(img.width(),img.height(), sweepmatx, sweepmaty);
			int sweepwidth = sweepmatx.width();
			int*& sweepmatx_data = sweepmatx.data();
			int*& sweepmaty_data = sweepmaty.data();

			
			for (int index = 0; index < crnrpts.size(); index++)
			{
				int x = crnrpts[index].x;
				int y = crnrpts[index].y;
				
				int win = _getwin(img, crnrpts[index], crnrpts);
				if (win < 3)
					continue;

				int crop_size = 2 * win + 1;
				int sweepmatcrop_size = 1.3*win + 0.5;
				ZQ_DImage<T> imgcrop(crop_size, crop_size), imgedgecrop(crop_size, crop_size);
				T*& imgcrop_data = imgcrop.data();
				T*& imgedgecrop_data = imgedgecrop.data();
				for (int h = y - win; h <= y + win; h++)
				{
					for (int w = x - win; w <= x + win; w++)
					{
						imgcrop_data[(h - y + win)*crop_size + w - x + win] = img_data[h*width + w];
						imgedgecrop_data[(h - y + win)*crop_size + w - x + win] = imgedge_data[h*width + w];
					}
				}

				const int* sweepmatxcrop = sweepmatx.data();
				const int* sweepmatycrop = sweepmaty.data();
				
				bool valid = false;
				std::vector<int> plocs;
				_validcorner(imgcrop, imgedgecrop, sweepmatxcrop, sweepmatycrop, sweepwidth, sweepmatcrop_size, valid, plocs);
				if (valid)
				{
					crnrs.push_back(ZQ_Vec2D(x, y));
					peaklocs.push_back(plocs);
				}
			}


			clock_t t2 = clock();
			//printf("cross filter : %.3f secs\n", 0.001*(t2-t1));
		}

		static void _sweepmatrix(int img_width, int img_height, ZQ_DImage<int>& sweepmatx, ZQ_DImage<int>& sweepmaty)
		{
			/*function[sweepmatx, sweepmaty] = sweepmatrix(img)
			SWEEPMATRIX precalulates the x and y coordinates for the ray pixels used in CIRCSWEEP.
			
			SWEEPMATRIX sets up the matrices which when cropped properly can be
			directly used by CIRCSWEEP to find the ray sum results.The main use for
			this function is to improve computational efficiency.
			
			USAGE:
			[sweepmatx, sweepmaty] = sweepmatrix(img)
			INPUTS :
			    img : image(the main concern is the size of the image
			OUTPUTS:
			    sweepmatx : x coordinate sweep matrix, each column corresponds to the
					x coordinates of the pixels that lie under a ray of a certain angle.
					Angles increase progressively from 0 to 360 along dimension 2 of the
					array.
				
				sweepmaty : y coordinate sweep matrix, each column corresponds to the
					y coordinates of the pixels that lie under a ray of a certain angle.*/

			double m_pi = atan(1.0) * 4;
		
			int win = __min(img_width, img_height);
			int x_num = 180;
			int y_num = win;
			if (!sweepmatx.matchDimension(x_num, y_num, 1))
				sweepmatx.allocate(x_num, y_num, 1);
			if (!sweepmaty.matchDimension(x_num, y_num, 1))
				sweepmaty.allocate(x_num, y_num, 1);

			int*& sweepmatx_data = sweepmatx.data();
			int*& sweepmaty_data = sweepmaty.data();
			for (int w = 0; w < x_num; w++)
			{
				double theta = m_pi / 90.0*(w + 1);
				for (int h = 0; h < y_num; h++)
				{
					double r = h + 1;
					sweepmatx_data[h*x_num + w] = (int)(r*cos(theta)+0.5);
					sweepmaty_data[h*x_num + w] = (int)(r*sin(theta)+0.5);
				}
			}
		}

		template<class T>
		static int _getwin(const ZQ_DImage<T>& img, const ZQ_Vec2D& pt, const std::vector<ZQ_Vec2D>& crnrpts)
		{
			/*function win = getwin(img, pt, crnrpts)
			GETWIN chooses an appropriate window size for the chessboard corner filter
			GETWIN chooses the window size depending on the spread of the points.It
			simply chooses a window size which does not include any other points.

			USAGE:
			win = getwin(img, pt, crnrpts)

			INPUTS :
			img : grayscale image(only used for size)
			pt : coordinates of pixel of interest
			crnrpts : 2xN array of all other points

			OUTPUTS :
			win : window size*/

			int width = img.width();
			int height = img.height();

			float min_dis = 1e9;
			int k = -1;
			ZQ_Vec2D nearest_pt;
			for (int i = 0; i < crnrpts.size(); i++)
			{
				ZQ_Vec2D cur_dir = crnrpts[i] - pt;
				if (cur_dir.x == 0 && cur_dir.y == 0)
					continue;
				float cur_dis = cur_dir.Length();
				if (cur_dis < min_dis)
				{
					min_dis = cur_dis;
					k = i;
					nearest_pt = crnrpts[i];
				}
			}

			int win = __max(fabs(nearest_pt.x - pt.x), fabs(nearest_pt.y - pt.y)) - 2;
			
			if (pt.x - win < 0)
				win = pt.x ;
			if (pt.x + win > width - 1)
				win = width - 1 - pt.x;
			if (pt.y - win < 0)
				win = pt.y - 1;
			if (pt.y + win > height - 1)
				win = height - 1 - pt.y;

			return win;
		}
		
		template<class T>
		static void _validcorner(const ZQ_DImage<T>& img, const ZQ_DImage<T>& imgedge, const int* sweepmatx, const int* sweepmaty, int sweep_width, int sweep_height, bool& valid, std::vector<int>& peaklocs)
		{
			/*function[valid, peaklocs] = validcorner(img, imgedge, sweepmatx, sweepmaty, debug)
			VALIDCORNER checks if the input corner belongs to a chessboard.
			
			VALIDCORNER is used to indicate if a corner is a chessboard corner.It
				takes as input the cropped image and cropped edge image around the
				corner.To ensure fast program execution sweepmatx and sweepmaty have
				been used.They are necessary input to this function.
				
			USAGE:
			[valid, peaklocs] = validcorner(img, imgedge, sweepmatx, sweepmaty);
			
			INPUTS:
			    img : cropped grayscale image
				imgedge : cropped Sobel edge image
				sweepmatx : cropped sweepmatrix x, this is used for fast radial summation
				sweepmatx : cropped sweepmatrix y, this is used for fast radial summation
			OUTPUTS
				valid : scalar indicating wether the point is a chessboard corner of not, 1 : yes, 0 : no
				peaklocs : required by GETGRID*/
				
			// validcorner parameters
			float imadjsca = 0.8; // larger adjust scalars corresponds to less adjustment
			float imeadjsca = 1.8;
			float intth = 0.5;

			ZQ_DImage<T> imgedge_new;
			ZQ_DImage<T> img_new;
			//Adjust windowed image
			_adjimg(imgedge, imeadjsca, imgedge_new);//edge images need less adjustment than imgn

			ZQ_DImage<T> edgevalue, edgevaluesmd;
			_circsweep(imgedge_new, sweepmatx, sweepmaty, sweep_width, sweep_height, edgevalue, edgevaluesmd);

			std::vector<int> max_pos;
			std::vector<T> max_val;
			_peakdet(edgevalue, max_pos, max_val);
			if (max_pos.size() != 4)
			{
				valid = false;
				return;
			}

			std::vector<int> max_pos_smd;
			std::vector<T> max_val_smd;
			_peakdet(edgevaluesmd, max_pos_smd, max_val_smd);
			if (max_pos_smd.size() != 2)
			{
				valid = false;
				return;
			}

			_adjimg(img, imadjsca, img_new);
			ZQ_DImage<T> intvalue, intvaluesmd;

			_circsweep(img_new, sweepmatx, sweepmaty, sweep_width, sweep_height, intvalue, intvaluesmd);

			_sort(max_pos_smd);
			double crn1 = 0, crn2 = 0;
			int count1 = 0, count2 = 0;
			T*& intvaluesmd_data = intvaluesmd.data();
			int len = intvaluesmd.nelements();
			for (int i = 0; i <= max_pos_smd[0]; i++)
			{
				crn1 += intvaluesmd_data[i];
				count1++;
			}
			for (int i = max_pos_smd[1]; i < len; i++)
			{
				crn1 += intvaluesmd_data[i];
				count1++;
			}
			for (int i = max_pos_smd[0]; i <= max_pos_smd[1]; i++)
			{
				crn2 += intvaluesmd_data[i];
				count2++;
			}

			crn1 /= count1;
			crn2 /= count2;

			if (fabs(crn1 - crn2) > intth)
			{
				valid = true;
				peaklocs = max_pos;
			}
			else
			{
				valid = false;
			}

		}

		
		template<class T>
		static void _sort(std::vector<T>& v)
		{
			int size = v.size();
			for (int pass = 0; pass < size-1; pass++)
			{
				for (int i = 0; i < size - 1; i++)
				{
					if (v[i] > v[i + 1])
					{
						T tmp = v[i];
						v[i] = v[i + 1];
						v[i + 1] = tmp;
					}
				}
			}
		}

		template<class T>
		static void _adjimg(const ZQ_DImage<T>& in_img, float th, ZQ_DImage<T>& out_img)
		{
			/*function imgout = adjimg(img, th)
			ADJIMG adjusts the intensity of the input image.
			
			ADJIMG adjusts the intensity of the input image based on the mean and the
				standard deviation of the intensitites in the image.
				
			USAGE:
				imgout = adjimg(img, th); th tunes the adjustment, higher th values
				     results in less adjustment
				
			INPUTS:
			    img : input grayscale image of double class
				th : tuning parameter
				
			OUTPUTS :
				imgout : adjusted image*/

			double mimg = _mean2(in_img);
			double stdv = _std2(in_img);

			double imax = mimg + th*stdv;
			double imin = mimg - th*stdv;

			if (imax > 1)
				imax = 1;
			if (imin < 0)
				imin = 0;

			if (!out_img.matchDimension(in_img))
				out_img.allocate(in_img);
			int nelements = out_img.nelements();

			const T*& in_data = in_img.data();
			T*& out_data = out_img.data();

			if (imax == imin)
				out_img = in_img;
			else
			{
				for (int i = 0; i < nelements; i++)
				{
					double val = (in_data[i] - imin) / (imax - imin);
					val = __min(1, __max(0, val));
					out_data[i] = val;
				}
			}
		}

		template<class T>
		static double _mean2(const ZQ_DImage<T>& img)
		{
			int nelements = img.nelements();
			if (nelements == 0)
				return 0;

			double sum = 0;
			const T*& data = img.data();
			
			for (int i = 0; i < nelements; i++)
				sum += data[i];
		
			return sum / nelements;
		}

		template<class T>
		static double _std2(const ZQ_DImage<T>& img)
		{
			int nelements = img.nelements();
			if (nelements == 0 || nelements == 1)
				return 0;

			double sum = 0;
			const T*& data = img.data();
			for (int i = 0; i < nelements; i++)
				sum += data[i];
			double avg = sum / nelements;

			sum = 0;
			for (int i = 0; i < nelements; i++)
			{
				double tmp = data[i] - avg;
				sum += tmp*tmp;
			}

			return sqrt(sum / (nelements - 1));
		}

		template<class T>
		static void _circsweep(const ZQ_DImage<T>& img, const int* x, const int* y, int mat_xy_width, int mat_xy_height, ZQ_DImage<T>& values, ZQ_DImage<T>& valuessmd)
		{
			/*
			CIRCSWEEP sums the intensity of the input image along rays at all angles.
			
			CIRCSWEEP sums the intensity pixels lying along a ray at a certain angle.
			 This is done for all angles. The outputs theta and values correspond to
			 the summation with the rays going from 0 to 360 degrees from the centre
			 of the image to the border. The other two outputs correspond to the
			 summation along the entire ray passing through the centre with the angles
			 going from 0 to 180.
			
			USAGE:
			    [theta,values,thetasmd,valuessmd]=circsweep(img,x,y);
			
			INPUTS:
			    img: input grayscale image of class double
				x: the x coordinates of the pixels, this is fed from the output of
			     SWEEPMATRIX
				y: the y coordinates of the pixels, this is fed from the output of
			     SWEEPMATRIX
			
			OUTPUTS:
			    theta: angles from 0 to 360
				values: the sum of intensity values from centre to border
			    thetasmd: angles from 0 to 180
			    valuessmd: the sum of intensity values from border to border along
			     the centre of the image
			*/

			int width = img.width();
			int height = img.height();
			const T*& img_data = img.data();

			int win = width / 2;
			int cen = win;
			int mat_width = mat_xy_width;
			int mat_height = mat_xy_height;
			int half_mat_width = mat_width / 2;

			
			if (!values.matchDimension(1, mat_width, 1))
				values.allocate(1, mat_width, 1);
			else
				values.reset();
			if (!valuessmd.matchDimension(1, half_mat_width, 1))
				valuessmd.allocate(1, half_mat_width, 1);
			else
				valuessmd.reset();
			
			T*& values_data = values.data();
			T*& valuessmd_data = valuessmd.data();

			for (int ww = 0; ww < mat_width; ww++)
			{
				double sum = 0;
				for (int hh = 0; hh < mat_height; hh++)
				{
					int offset = hh*mat_width + ww;
					int cur_x = x[offset] + cen;
					int cur_y = y[offset] + cen;
					if (cur_x < 0 || cur_x > width - 1 || cur_y < 0 || cur_y > height - 1)
					{
						sum += 0;
					}
					else
						sum += img_data[cur_y*width + cur_x];
				}
				values_data[ww] = sum / mat_height;
			}
			for (int ww = 0; ww < half_mat_width; ww++)
			{
				valuessmd_data[ww] = 0.5*(values_data[ww] + values_data[ww + half_mat_width]);
			}
		}

		template<class T>
		static void _peakdet(const ZQ_DImage<T>& v, std::vector<int>& max_pos, std::vector<T>& max_val)
		{
			/*
			function maxtab=peakdet(v)
			PEAKDET Detect peaks in a vector
			
			[MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
			 maxima and minima ("peaks") in the vector V.
			 MAXTAB and MINTAB consists of two columns. Column 1
			 contains indices in V, and column 2 the found values.
			
			 With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
			 in MAXTAB and MINTAB are replaced with the corresponding
			 X-values.
			
			 A point is considered a maximum peak if it has the maximal
			 value, and was preceded (to the left) by a value lower by
			 DELTA.
			
			 Eli Billauer, 3.4.05 (Explicitly not copyrighted).
			 This function is released to the public domain; Any use is allowed.
			
			Modified by Abdallah Kassir 13/12/2009*/

			max_pos.clear();
			max_val.clear();

			const T*& v_data = v.data();
			int nelements = v.nelements();

			ZQ_DImage<int> x_in(nelements, 1, 1);
			ZQ_DImage<int> x_out(nelements, 1, 1);
			ZQ_DImage<T> v_out(nelements, 1, 1);
			
			T*& v_out_data = v_out.data();
			int*& x_in_data = x_in.data();
			int*& x_out_data = x_out.data();
			for (int i = 0; i < nelements; i++)
				x_in_data[i] = i;
			_cadj(v, x_in, v_out, x_out);

			double max_v = -1e9;
			double min_v = 1e9;
			int mxpos = -1;
			
			for (int i = 0; i < nelements; i++)
			{
				max_v = __max(max_v, v_out_data[i]);
				min_v = __min(min_v, v_out_data[i]);
			}
			double delta = (max_v - min_v) / 4;
			
			if (delta == 0)
				return ;

			max_v = -1e9;
			min_v = 1e9;
			int mx_pos = -1;
			bool lookformax = false;
			for (int i = 0; i < nelements; i++)
			{
				double cur_v = v_out_data[i];
				if (cur_v > max_v)
				{
					max_v = cur_v;
					mx_pos = x_out_data[i];
				}
				if (cur_v < min_v)
					min_v = cur_v;

				if (lookformax)
				{
					if (cur_v < max_v - delta)
					{
						max_pos.push_back(mx_pos);
						max_val.push_back(max_v);
						min_v = cur_v;
						lookformax = false;
					}
				}
				else
				{
					if (cur_v > min_v + delta)
					{
						max_v = cur_v;
						mx_pos = x_out_data[i];
						lookformax = true;
					}
				}
			}
		}

		template<class T>
		static void _cadj(const ZQ_DImage<T>& vin, const ZQ_DImage<int>& xin, ZQ_DImage<T>& vout, ZQ_DImage<int>& xout)
		{
			/*
			function [vout,xout]=cadj(vin,xin)
			% CADJ shifts a vector so that the starting value is always the smallest value.
			%
			% CADJ shifts the input vector preserving the order in the vector. The
			% function ensures that the smallest value is always the first element.
			%
			% USAGE:
			%     vout=cadj(vin);
			%
			%     [vout,xout]=cadj(vin,xin); xout is an adjusted version of xin
			%     according to the adjustment of vin and vout
			%
			% INPUTS:
			%     vin: input vector
			%
			%     xin: optional x coordinate vector
			%
			% OUTPUTS:
			%     vout: adjusted output vector
			%
			%     xout: adjusted x coordinate vector
			*/

			const T*& vin_data = vin.data();
			const int*& xin_data = xin.data();
			int nelements = vin.nelements();
			double min_v = vin_data[0];
			int pos = 0;
			for (int i = 1; i < nelements; i++)
			{
				if (min_v > vin_data[i])
				{
					min_v = vin_data[i];
					pos = i;
				}
			}

			if (!vout.matchDimension(nelements, 1, 1))
				vout.allocate(nelements, 1, 1);
			if (!xout.matchDimension(nelements, 1, 1))
				xout.allocate(nelements, 1, 1);

			T*& vout_data = vout.data();
			int*& xout_data = xout.data();
			for (int i = pos; i < nelements; i++)
			{
				vout_data[i - pos] = vin_data[i];
				xout_data[i - pos] = xin_data[i];
			}
			for (int i = 0; i < pos; i++)
			{
				vout_data[nelements - pos + i] = vin_data[i];
				xout_data[nelements - pos + i] = xin_data[i];
			}
		}

		template<class T>
		static void _getgrid(const std::vector<ZQ_Vec2D>& crnrs, const std::vector<ZQ_Vec2D>& pixs, const std::vector<std::vector<int>>& peaklocs, const ZQ_DImage<T>& ix, const ZQ_DImage<T>& iy, ZQ_DImage<T>& crnrsgridout)
		{
			/*
			function crnrsgridout=getgrid(crnrs,pixs,peaklocs,ix,iy,debug)
			% GETGRID arranges the points that pass the chessboard filter into a grid.
			%
			% GETGRID returns the arranged grid of the set of candidate chessboard
			% corners.
			%
			% USAGE:
			%     crnrsgridout=getgrid(crnrs,pixs,peaklocs,ix,iy);
			%
			% INPUTS:
			%     crnrs: 2xN matrix of the N candidate corners.
			%
			%     pixs: 2xM matrix of the M Harris corners
			%
			%     peaklocs: edge peak locations at each point (necesary for
			%     arrangement)
			%
			%     ix: the x component of the gradient image
			%
			%     iy: the y component of the gradient image
			%
			% OUTPUTS:
			%     crnrsgridout: MxNx2 dimensional matrix contianing arranged points
			*/

			int width = ix.width();
			int height = ix.height();
			const T*& ix_data = ix.data();
			const T*& iy_data = iy.data();

			// get mean point
			int nocrnrs = crnrs.size();
			ZQ_Vec2D centerpt(0, 0);

			for (int i = 0; i < nocrnrs; i++)
				centerpt += crnrs[i];
			centerpt *= 1.0 / nocrnrs;

			ZQ_Vec2D currentpt = crnrs[0];
			int ctindx = 0;
			ZQ_Vec2D dir = crnrs[0] - centerpt;
			float min_dis = dir.Length();
			for (int i = 1; i < nocrnrs; i++)
			{
				dir = crnrs[i] - centerpt;
				float cur_dis = dir.Length();
				if (min_dis > cur_dis)
				{
					min_dis = cur_dis;
					ctindx = i;
					currentpt = crnrs[i];
				}
			}
			
			//parameters
			double angth = atan(1.0) / 45.0*15.0;

			// always store first point
			bool valid = true;

			// inititalise matrices
			ZQ_DImage<int> crnrsgrid(nocrnrs, nocrnrs);
			int*& crnrsgrid_data = crnrsgrid.data();
			int crnrsgrid_width = crnrsgrid.width();
			int crnrsgrid_height = crnrsgrid.height();
			for (int i = 0; i < crnrsgrid_width*crnrsgrid_height; i++)
				crnrsgrid_data[i] = -1;
			ZQ_DImage<int> crnrsgriddone(nocrnrs, nocrnrs);
			int*& crnrsgriddone_data = crnrsgriddone.data();

			// place first point in the middle of the grid
			int	xg = nocrnrs / 2.0 - 0.5;
			int yg = nocrnrs / 2.0 - 0.5;

			// enumerate different directions
			int right = 0;
			int top = 3;
			int left = 2;
			int bottom = 1;

			// setup position matrix
			int posmat[9] = 
			{
				0,		top,	0,
				left,	0,	right,
				0,	bottom,		0
			};
			
			// set while loop flag
			bool notdone = true;
			// set loop counter
			// just to ensure safe performance, prevent loop from going on forever
			int loopcntr = 0;
			int looplimit = 1e6;

			while (notdone && loopcntr < looplimit)
			{
				loopcntr++;
				// get current point coords
				ZQ_Vec2D currentpt = crnrs[ctindx];
				
				float xc = currentpt.x;
				float yc = currentpt.y;
				// get surrounding chessboard corner properties(sorted by distance)
				
				std::vector<ZQ_Vec2D> surcrnrs;
				std::vector<int> surindx;
				_findnearest(currentpt, crnrs, 8, false, surcrnrs, surindx);
				int nn_num = surcrnrs.size();
				std::vector<ZQ_Vec2D> vecs(nn_num);
				for (int i = 0; i < nn_num; i++)
					vecs[i] = surcrnrs[i] - currentpt;

				std::vector<float> angles(nn_num);
				for (int i = 0; i < nn_num; i++)
				{
					float cur_angle = atan2(vecs[i].y, vecs[i].x);
					if (cur_angle <= 0)
						cur_angle += atan(1.0) * 8;
					angles[i] = cur_angle;
				}

				// setup the segment vectors to the corners in order to check for
				// validity of segment between points by summing ix and iy along segment

				std::vector<ZQ_Vec2D> vecsnrm(nn_num);
				std::vector<float> vecslen(nn_num);
				for (int i = 0; i < nn_num; i++)
				{
					float cur_len = vecs[i].Length();
					vecslen[i] = cur_len;
					vecsnrm[i] = vecs[i] * (1.0 / cur_len);
				}
				
				std::vector<double> ixvalue(nn_num);
				std::vector<double> iyvalue(nn_num);
				std::vector<double> segedgevalue(nn_num);
				for (int i = 0; i < nn_num; i++)
				{
					ixvalue[i] = 0;
					iyvalue[i] = 0;
					segedgevalue[i] = 0;
				}
				
				for (int crnrcnr = 0; crnrcnr < nn_num; crnrcnr++)
				{
					int cur_len = vecslen[crnrcnr] + 0.5f;
					int evcnr;
					for (evcnr = 1; evcnr <= cur_len; evcnr++)
					{
						int xev = xc + vecsnrm[crnrcnr].x * evcnr + 0.5f;
						int yev = yc + vecsnrm[crnrcnr].y * evcnr + 0.5f;
						ixvalue[crnrcnr] += ix_data[yev*width + xev];
						iyvalue[crnrcnr] += iy_data[yev*width + xev];
					}
					ZQ_Vec2D tmp_vv(ixvalue[crnrcnr] / evcnr, iyvalue[crnrcnr] / evcnr);
					segedgevalue[crnrcnr] = tmp_vv.Length();
				}
				float segedgemean = 0;
				for (int i = 0; i < nn_num; i++)
					segedgemean += segedgevalue[i];
				segedgemean /= nn_num;

				// get surrounding Harris point properties(sorted by distance)

				std::vector<ZQ_Vec2D> surpixs;
				std::vector<int> surpixs_idx;
				_findnearest(currentpt, pixs, 8, false, surpixs, surpixs_idx);
				int nn_num_pixs = surpixs.size();
				std::vector<ZQ_Vec2D> vecspixs(nn_num_pixs);
				for (int i = 0; i < nn_num_pixs; i++)
				{
					vecspixs[i] = surpixs[i] - currentpt;
				}
				std::vector<float> anglespixs(nn_num_pixs);
				for (int i = 0; i < nn_num_pixs; i++)
				{
					float cur_angle = atan2(vecspixs[i].y, vecspixs[i].x);
					if (cur_angle <= 0)
						cur_angle += 8.0*atan(1.0);
					anglespixs[i] = cur_angle;
				}
				
				std::vector<float> theta(180);
				for (int i = 0; i < 180; i++)
				{
					theta[i] = atan(1.0)*4.0 / 90 * (i + 1);
				}

				std::vector<int> locs = peaklocs[ctindx];
				int locs_size = locs.size();
				if (locs_size != 4)
				{
					printf("error: There should be 4 and only 4 peaks!\n");
				}
				std::vector<float> lineangles(locs_size);
				for (int i = 0; i < locs_size; i++)
				{
					lineangles[i] = theta[locs[i]];
				}
				

				int crosspixs[4] = { -1,-1,-1,-1 };
				for (int pk = 0; pk < 4; pk++)
				{
					for (int crnr = 0; crnr < nn_num_pixs; crnr++)
					{
						//check for angle proximity and segment edge projection
						if (_angprox(angles[crnr], lineangles[pk], angth) && segedgevalue[crnr]>segedgemean)
						{
							for (int pix = 0; pix < nn_num_pixs; pix++)
							{
								//check if a Harris corner lies in between
								if (_angprox(anglespixs[pix], lineangles[pk], angth))
								{
									if (fabs(surpixs[pix].x - surcrnrs[crnr].x) < 0.01 && fabs(surpixs[pix].y - surcrnrs[crnr].y) < 0.01)
									{
										crosspixs[pk] = surindx[crnr];
										break;
									}
									else
									{
										break;
									}
								}
							}
						}
					}
				}

				//Adjust cross
				for (int i = 0; i < 4; i++)
				{
					if (valid)
						break;
					for (int v = yg - 1; v <= yg + 1; v++)
					{
						if (valid)
							break;
						for (int u = xg - 1; u <= xg + 1; u++)
						{
							if (valid)
								break;
							if (crosspixs[i] == crnrsgrid_data[v*crnrsgrid_width + u] && crnrsgriddone_data[v*crnrsgrid_width + u] > 0) // check for valid corner, check for non zero value as well
							{
								valid = true; //a cross is valid if a match is found
								int k = posmat[(v - yg + 1) * 3 + (u - xg + 1)] - i;
								//printf("ctindx = %d, k = %d\n", ctindx, k);
								if (k < 0)
									k += 4;
								int crosspixs1[4];
								for (int ppp = k ; ppp < 4; ppp++)
									crosspixs1[ppp] = crosspixs[ppp - k];
								for (int ppp = 0; ppp < k; ppp++)
									crosspixs1[ppp] = crosspixs[4 - k + ppp];
								memcpy(crosspixs, crosspixs1, sizeof(int)* 4);
							}
						}
						
					}			
				}
				


				if (valid) //if connection found store
				{
					//draw cross matrix
					float cmat[9] =
					{
						-1, crosspixs[3], -1,
						crosspixs[2], ctindx, crosspixs[0],
						-1, crosspixs[1], -1
					};

					//store changes
					for (int yyy = yg - 1; yyy <= yg + 1; yyy++)
					{
						for (int xxx = xg - 1; xxx <= xg + 1; xxx++)
						{
							if (crnrsgrid_data[yyy*crnrsgrid_width + xxx] == -1)
								crnrsgrid_data[yyy*crnrsgrid_width + xxx] = cmat[(yyy - (yg - 1)) * 3 + (xxx - (xg - 1))];
						}
					}

					int cmatdone[9] = { 0, 1, 0, 1, 2, 1, 0, 1, 0 };
					for (int yyy = 0; yyy < 3; yyy++)
					{
						for (int xxx = 0; xxx < 3; xxx++)
						{
							int off_x = xxx + xg - 1;
							int off_y = yyy + yg - 1;
							if (cmat[yyy * 3 + xxx] >= 0 && off_x >= 0 && off_x < crnrsgrid_width && off_y >= 0 && off_y < crnrsgrid_height)
							{
								if (crnrsgriddone_data[(yyy + yg - 1)*crnrsgrid_width + (xxx + xg - 1)] < cmatdone[yyy*3+xxx])
								{
									crnrsgriddone_data[(yyy + yg - 1)*crnrsgrid_width + (xxx + xg - 1)] = cmatdone[yyy * 3 + xxx];
								}
							}
						}
					}
					
					valid = false;
				}
				else
				{
					crnrsgriddone_data[yg*crnrsgrid_width + xg] = 0;
				}
				
				/*ZQ_DImage<float> tmp_done(crnrsgrid_width, crnrsgrid_height);
				for (int i = 0; i < tmp_done.nelements(); i++)
					tmp_done.data()[i] = 0.25*crnrsgriddone_data[i];
				tmp_done.imresize(10);
				ZQ_ImageIO::Show("done", tmp_done);
				cvWaitKey(0);*/
				//get new point
				bool found_flag = false;
				for (int www = 0; www < crnrsgrid_width; www++)
				{
					for (int hhh = 0; hhh < crnrsgrid_height; hhh++)
					{
						if (crnrsgriddone_data[hhh*crnrsgrid_width + www] == 1)
						{
							yg = hhh;
							xg = www;
							found_flag = true;
							break;
						}
					}
					if (found_flag)
						break;
				}
				if (!found_flag)
					notdone = false;
				else
				{
					ctindx = crnrsgrid_data[yg*crnrsgrid_width + xg];
				}
			}


			// store x and y coords into matrix
			if (!crnrsgridout.matchDimension(crnrsgrid_width, crnrsgrid_height, 2))
				crnrsgridout.allocate(crnrsgrid_width, crnrsgrid_height, 2);
			else
				crnrsgridout.reset();

			T*& crnrgridout_data = crnrsgridout.data();
			
			for (int h = 0; h < crnrsgrid_height; h++)
			{
				for (int w = 0; w < crnrsgrid_width; w++)
				{
					int idx = crnrsgrid_data[h*crnrsgrid_width + w];
					if (idx >= 0)
					{
						crnrgridout_data[(h*crnrsgrid_width + w) * 2 + 0] = crnrs[idx].x;
						crnrgridout_data[(h*crnrsgrid_width + w) * 2 + 1] = crnrs[idx].y;
					}
				}
			}

			/*ZQ_DImage<T> sepx, sepy;
			crnrsgridout.separate(1, sepx, sepy);
			ZQ_ImageIO::Show("sepx", sepx);
			ZQ_ImageIO::Show("sepy", sepy);
			cvWaitKey(0);*/
		}
		
		static bool _angprox(double ang1, double ang2, double th)
		{
			return fabs(ang1 - ang2) < th || fabs(ang1 - ang2) > (atan(1.0) * 8 - th);
		}

		static void _findnearest(const ZQ_Vec2D& p, const std::vector<ZQ_Vec2D>& pts, int N, bool same, std::vector<ZQ_Vec2D>& nxpts, std::vector<int>& ixs)
		{
			/*
			% FINDNEAREST finds from a array of points the nearest to a certain point.
			%
			% FINDNEAREST finds the nearest num points to point p from points pts. If
			% same is input, the same point will be returned if it exists in pts.
			%
			% USAGE:
			%     [nxpts,ixs]=findnearest(p,pts,num); gets the nearest num pts
			%
			%     [nxpts,ixs]=findnearest(p,pts,num,1); the same point is returned if
			%     it exists
			%
			% INPUTS:
			%     pt: reference point
			%
			%     pts: 2xN array of points
			%
			%     num: number of nearest points required
			%
			%     same: flag, if input allows the function to return point pt
			%
			% OUTPUTS:
			%     nxpts: 2xnum array containing the found points
			%
			%     ixs: the column coordinates of the points
			*/

			nxpts.clear();
			ixs.clear();

			int num = pts.size();
			if (num == 0)
				return;

			
			double* dist2 = new double[num];
			int* idx = new int[num];
			for (int i = 0; i < num; i++)
				idx[i] = i;
			for (int i = 0; i < num; i++)
			{
				double cur_dis2 = (p.x - pts[i].x)*(p.x - pts[i].x) + (p.y - pts[i].y)*(p.y - pts[i].y);
				dist2[i] = cur_dis2;
			}
			ZQ_MergeSort::MergeSort(dist2, idx, num, true);
			
			if (same)
			{
				int real_N = 0;
				for (int j = 0; j < num && real_N < N; j++)
				{
					nxpts.push_back(pts[idx[j]]);
					ixs.push_back(idx[j]);
					real_N++;
				}
			}
			else
			{
				int real_N = 0;
				for (int j = 0; j < num && real_N < N; j++)
				{
					if (dist2[j] > 0)
					{
						nxpts.push_back(pts[idx[j]]);
						ixs.push_back(idx[j]);
						real_N++;
					}
				}
			}
		}

		
	};
}

#endif