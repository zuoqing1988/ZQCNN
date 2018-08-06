#ifndef _ZQ_CLOSED_FORM_IMAGE_MATTING_H_
#define _ZQ_CLOSED_FORM_IMAGE_MATTING_H_
#pragma once

#include "ZQ_DoubleImage.h"
#include "ZQ_BinaryImageProcessing.h"
#include "ZQ_Matrix.h"
#include "ZQ_SVD.h"
#include "ZQ_TaucsBase.h"
#include "ZQ_SparseMatrix.h"
#include "ZQ_ImageIO.h"
#include "ZQ_UmfpackSolver.h"
#include "ZQ_LSQRSolver.h"
#include "ZQ_PCGSolver.h"
#include "ZQ_MGMRESSolver.h"
#include <string.h>

namespace ZQ
{
	class ZQ_ClosedFormImageMatting
	{
		/*
		refer to the paper:
		A Closed-Form Solution to Natural Image Matting, 2008, TPAMI. Anat Levin, Dani Lischinski, and Yair Weiss. 
		*/
	public:

		/*
		use the umfpack solver (as I find ZQ_PCGSolver or ZQ_LSQRSolver cannnot work well)
		you should add the code "#define _USE_UMFPACK 1" at the beginning of the main.cpp or at the beginning of ZQ_UmfpackSolver.h
		or it will always return false
		*/
		template<class T>
		static bool SolveAlpha(const ZQ_DImage<T>& im, const ZQ_DImage<bool>& consts_map, const ZQ_DImage<T>& consts_vals, ZQ_DImage<T>& alpha, float epsilon = 1e-7, int win_size = 1, bool display = false);

		template<class T>
		static bool Coarse2FineSolveAlpha(const ZQ_DImage<T>& im, const ZQ_DImage<bool>& consts_map, const ZQ_DImage<T>& consts_vals, ZQ_DImage<T>& alpha, int max_level, float consts_thresh = 0.02, float epsilon = 1e-7, int win_size = 1, bool display = false);

		/*
		use ZQ_PCGsolver.
		*/
		template<class T>
		static bool SolveForeBack_4dir(const ZQ_DImage<T>& im, const ZQ_DImage<T> alpha, ZQ_DImage<T>& fore, ZQ_DImage<T>& back, int max_iter, bool display = false);
		
		template<class T>
		static bool SolveForeBack_2dir(const ZQ_DImage<T>& im, const ZQ_DImage<T> alpha, ZQ_DImage<T>& fore, ZQ_DImage<T>& back, int max_iter, bool display = false);

		template<class T>
		static bool SolveForeBack_ori_paper(const ZQ_DImage<T>& im, const ZQ_DImage<T> alpha, ZQ_DImage<T>& fore, ZQ_DImage<T>& back, int max_iter, bool display = false);

	private:

		template<class T> 
		static bool _buildMatrix_for_SolveAlpha(const ZQ_DImage<T>& im, const ZQ_DImage<bool>& consts_map, const ZQ_DImage<T>& consts_vals, int win_size, float epsilon, ZQ_SparseMatrix<T>& A_mat, ZQ_DImage<double>& b, bool tranposed = false);

		template<class T>
		static void _getGradientofAlpha_4dir(const ZQ_DImage<T>& alpha, ZQ_DImage<T>& alpha_gx, ZQ_DImage<T>& alpha_gy, ZQ_DImage<T>& alpha_g_ur, ZQ_DImage<T>& alpha_g_dr, float upthresh = 0.98, float downthresh = 0.02);

		template<class T>
		static void _getGradientofAlpha_2dir(const ZQ_DImage<T>& alpha, ZQ_DImage<T>& alpha_gx, ZQ_DImage<T>& alpha_gy, float upthresh = 0.98, float downthresh = 0.02);

		template<class T>
		static void _buildMatrix_for_SolveForeBack_4dir(const ZQ_DImage<T>& alpha, const ZQ_DImage<T>& wgf, const ZQ_DImage<T>& wgb, const ZQ_DImage<T>& wf, const ZQ_DImage<T>& wb,
			ZQ_SparseMatrix<T>& Amat, float upthresh = 0.98, float downthresh = 0.02);

		template<class T>
		static void _buildMatrix_for_SolveForeBack_2dir(const ZQ_DImage<T>& alpha, const ZQ_DImage<T>& wgf, const ZQ_DImage<T>& wgb, const ZQ_DImage<T>& wf, const ZQ_DImage<T>& wb,
			ZQ_SparseMatrix<T>& Amat, float upthresh = 0.98, float downthresh = 0.02);

		template<class T>
		static void _buildMatrix_for_SolveForeBack_ori_paper(const ZQ_DImage<T>& alpha, const ZQ_DImage<int>& unknown_index_map, ZQ_SparseMatrix<T>& Amat);

		template<class T>
		static void _downSampleImage(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, int size);

		template<class T>
		static void _downSample(const ZQ_DImage<T>& im, const ZQ_DImage<bool>& consts_map, const ZQ_DImage<T>& consts_vals, ZQ_DImage<T>& s_im, ZQ_DImage<bool>& s_const_map, ZQ_DImage<T>& s_consts_vals);

		template<class T>
		static void _upSampleAlphaUsingImage(const ZQ_DImage<T>& s_alpha, const ZQ_DImage<T>& s_im, const ZQ_DImage<T>& im, ZQ_DImage<T>& alpha, float epsilon, int win_size);

		template<class T>
		static void _upSampleImage(const ZQ_DImage<T>& input, int out_width, int out_height, ZQ_DImage<T>& output, int size);

		template<class T>
		static void _getLinearCoeff(const ZQ_DImage<T>& s_alpha, const ZQ_DImage<T>& s_im, ZQ_DImage<T>& s_coeff, float epsilon, int win_size);
		
		template<class T>
		static void _updateConstsMapConstsVals(const ZQ_DImage<T>& alpha, const ZQ_DImage<bool>& consts_map, const ZQ_DImage<T>& consts_vals, ZQ_DImage<bool>& out_consts_map, ZQ_DImage<T>& out_consts_vals, float consts_thresh = 0.02);

	};

	template<class T>
	bool ZQ_ClosedFormImageMatting::SolveAlpha(const ZQ_DImage<T>& im, const ZQ_DImage<bool>& consts_map, const ZQ_DImage<T>& consts_vals, ZQ_DImage<T>& alpha, float epsilon /*= 1e-7*/, int win_size /*= 1*/, bool display /*= false*/)
	{
		int width = im.width();
		int height = im.height();
		int nChannels = im.nchannels();
		if (!consts_map.matchDimension(width, height, 1) || !consts_vals.matchDimension(width, height, 1))
			return false;

		alpha.allocate(width, height, 1);
		const T*& im_data = im.data();
		const T*& consts_vals_data = consts_vals.data();
		T*& alpha_data = alpha.data();

		int Xsize = win_size * 2 + 1;
		int neighbor_size = Xsize*Xsize;

		int image_size = width*height;
		ZQ_DImage<bool> pfilter2D(Xsize, Xsize);
		bool*& pfilter2D_data = pfilter2D.data();
		for (int i = 0; i < Xsize*Xsize; i++)
			pfilter2D_data[i] = true;
		ZQ_DImage<bool> consts_map1(width, height);
		bool*& consts_map1_data = consts_map1.data();
		const bool*& consts_map_data = consts_map.data();
		ZQ_BinaryImageProcessing::Erode(consts_map_data, consts_map1_data, width, height, pfilter2D_data, win_size, win_size);

		ZQ_SparseMatrix<T> A_mat(width*height, width*height);
		ZQ_DImage<double> b;

		const int MODE_UMFPACK = 0, MODE_MGMRES = 1;
		int mode = MODE_UMFPACK;
		if (mode == MODE_UMFPACK)
		{
			if (!_buildMatrix_for_SolveAlpha(im, consts_map1, consts_vals, win_size, epsilon, A_mat, b, false))
			{
				if (display)
					printf("build matrix for SolveAlpha fail!\n");
				return false;
			}
		}
		else
		{
			if (!_buildMatrix_for_SolveAlpha(im, consts_map1, consts_vals, win_size, epsilon, A_mat, b, true))
			{
				if (display)
					printf("build matrix for SolveAlpha fail!\n");
				return false;
			}
		}
		/*A_mat.ExportToFile("Amat.txt");
		FILE* out = fopen("b.txt", "w");
		fprintf(out, "%d\n", b.npixels());
		for (int i = 0; i < b.npixels(); i++)
			fprintf(out, "%f\n", b.data()[i]);
		fclose(out);*/

		taucs_ccs_matrix* A = A_mat.ExportCCS(TAUCS_DOUBLE);
		if (A == 0)
		{
			if (display)
				printf("failed to allocate sparse matrix!\n");
			return false;
		}
		A_mat.Clear();
		
		if (display)
			printf("build matrix done!\n");
		ZQ_DImage<double> x(width*height, 1);
		double*& x_data = x.data();
		double*& b_data = b.data();
		bool ret_flag = false;
		if (mode == MODE_UMFPACK)
		{
			ret_flag = ZQ_UmfpackSolver::UmfpackSolve(A, b_data, x_data, display);
		}
		else
		{
			ret_flag = ZQ_MGMRESSolver::MGMResSolve(A, b_data, x_data, 1, 200, 1e-16, display);
		}
		ZQ_TaucsBase::ZQ_taucs_ccs_free(A);
		if (!ret_flag)
		{
			if (display)
			{
				printf("failed to solve alpha!\n");
				
				return false;
			}
		}
		
		for (int i = 0; i < image_size; i++)
		{
			alpha_data[i] = x_data[i];
		}
		//ZQ_ImageIO::Show("alpha", alpha);
		if (display)
			printf("solve alpha done!\n");
		return true;
	}

	template<class T>
	bool ZQ_ClosedFormImageMatting::Coarse2FineSolveAlpha(const ZQ_DImage<T>& im, const ZQ_DImage<bool>& consts_map, const ZQ_DImage<T>& consts_vals, ZQ_DImage<T>& alpha, 
		int max_level, float consts_thresh/* = 0.02*/, float epsilon/* = 1e-7*/, int win_size/* = 1*/, bool display/* = false*/)
	{
		if (max_level <= 1)
		{
			if (!SolveAlpha(im, consts_map, consts_vals, alpha, epsilon, win_size, display))
				return false;
			if (display)
				printf("max_level = %d done!\n", max_level);
			return true;
		}

		int width = im.width();
		int height = im.height();
		if (!consts_map.matchDimension(width, height, 1))
			return false;
		if (!consts_vals.matchDimension(width, height, 1))
			return false;

		
		ZQ_DImage<T> s_im, s_consts_vals, s_alpha;
		ZQ_DImage<bool> s_consts_map;
		_downSample(im, consts_map, consts_vals, s_im, s_consts_map, s_consts_vals);
		
		if (!Coarse2FineSolveAlpha(s_im, s_consts_map, s_consts_vals, s_alpha, max_level-1, consts_thresh, epsilon, win_size, display))
		{
			return false;
		}

		_upSampleAlphaUsingImage(s_alpha, s_im, im, alpha, epsilon, win_size);

		ZQ_DImage<bool> cur_consts_map(width, height);
		ZQ_DImage<T> cur_consts_vals(width, height);

		_updateConstsMapConstsVals(alpha, consts_map, consts_vals, cur_consts_map, cur_consts_vals, consts_thresh);
		
		if (!SolveAlpha(im, cur_consts_map, cur_consts_vals, alpha, epsilon, win_size, display))
		{
			return false;
		}
		if (display)
			printf("max_level = %d done!\n", max_level);
		return true;
	}

	template<class T>
	bool ZQ_ClosedFormImageMatting::SolveForeBack_4dir(const ZQ_DImage<T>& im, const ZQ_DImage<T> alpha, ZQ_DImage<T>& fore, ZQ_DImage<T>& back, int max_iter, bool display /*= false*/)
	{
		int width = im.width();
		int height = im.height();
		int image_size = width*height;
		int nChannels = im.nchannels();
		if (!alpha.matchDimension(width, height, 1))
			return false;
		if (!fore.matchDimension(width, height, nChannels))
			fore.allocate(width, height, nChannels);
		else
			fore.reset();
		if (!back.matchDimension(width, height, nChannels))
			back.allocate(width, height, nChannels);
		else
			back.reset();
		
		const T*& alpha_data = alpha.data();
		const T*& im_data = im.data();
		T*& fore_data = fore.data();
		T*& back_data = back.data();

		const float weight_for_wgf_wgb = 0.003;
		const float upweight_for_wf_wb = 100;
		const float midweight_for_wf_wb = 0.03;
		const float downweight_for_wf_wb = 0.01;
		const float upthresh_for_wf_wb = 0.98;
		const float midupthresh_for_wf_wb = 0.7;
		const float middownthresh_for_wf_wb = 0.3;
		const float downthresh_for_wf_wb = 0.02;

		ZQ_DImage<T> gx, gy, g_ur, g_dr;
		_getGradientofAlpha_4dir(alpha, gx, gy, g_ur, g_dr,upthresh_for_wf_wb,downthresh_for_wf_wb);
		T*& gx_data = gx.data();
		T*& gy_data = gy.data();
		T*& g_ur_data = g_ur.data();
		T*& g_dr_data = g_dr.data();

		ZQ_DImage<T> wgf(image_size * 4, 1);
		ZQ_DImage<T> wgb(image_size * 4, 1);
		T*& wgf_data = wgf.data();
		T*& wgb_data = wgb.data();

		
		for (int i = 0; i < image_size; i++)
		{
			wgf_data[i] = sqrt(fabs(gx_data[i])) + weight_for_wgf_wgb*(1 - alpha_data[i]);
			wgf_data[i + image_size] = sqrt(fabs(gy_data[i])) + weight_for_wgf_wgb*(1 - alpha_data[i]);
			wgf_data[i + image_size * 2] = sqrt(fabs(g_ur_data[i])) + weight_for_wgf_wgb*(1 - alpha_data[i]);
			wgf_data[i + image_size * 3] = sqrt(fabs(g_dr_data[i])) + weight_for_wgf_wgb*(1 - alpha_data[i]);

			wgb_data[i] = sqrt(fabs(gx_data[i])) + weight_for_wgf_wgb*alpha_data[i];
			wgb_data[i + image_size] = sqrt(fabs(gy_data[i])) + weight_for_wgf_wgb*alpha_data[i];
			wgb_data[i + image_size * 2] = sqrt(fabs(g_ur_data[i])) + weight_for_wgf_wgb*alpha_data[i];
			wgb_data[i + image_size * 3] = sqrt(fabs(g_dr_data[i])) + weight_for_wgf_wgb*alpha_data[i];
		}

		ZQ_DImage<T> wf(image_size, 1);
		ZQ_DImage<T> wb(image_size, 1);
		T*& wf_data = wf.data();
		T*& wb_data = wb.data();
		
		
		for (int i = 0; i < image_size; i++)
		{
			wf_data[i] = (alpha_data[i]>upthresh_for_wf_wb)*upweight_for_wf_wb
				+ (alpha_data[i] > midupthresh_for_wf_wb) *alpha_data[i] * midweight_for_wf_wb
				+ (alpha_data[i] < downthresh_for_wf_wb) * downweight_for_wf_wb;
			wb_data[i] = (alpha_data[i] < downthresh_for_wf_wb)*upweight_for_wf_wb
				+ (alpha_data[i]<middownthresh_for_wf_wb)*(1 - alpha_data[i])*midweight_for_wf_wb
				+ (alpha_data[i] > upthresh_for_wf_wb)*downweight_for_wf_wb;
		}
		ZQ_SparseMatrix<T> A_mat(image_size * 11, image_size * 2);
		_buildMatrix_for_SolveForeBack_4dir(alpha, wgf, wgb, wf, wb, A_mat, upthresh_for_wf_wb, downthresh_for_wf_wb);
		if (display)
			printf("build matrix for solve fore back done!\n");
		A_mat.ShrinkToFit();
		taucs_ccs_matrix* A = A_mat.ExportCCS(TAUCS_DOUBLE);
		if (A == 0)
		{
			if (display)
				printf("failed to allocate sparse matrix!\n");
			return false;
		}
		A_mat.Clear();
		
		ZQ_DImage<double> b(image_size * 11, 1);
		double*& b_data = b.data();
		double* bi = b_data;
		double* bs = bi + image_size * 2;
		double* bg = bs + image_size;
		
		ZQ_DImage<double> x(image_size * 2, 1);
		double*& x_data = x.data();

		for (int c = 0; c < nChannels; c++)
		{
			for (int i = 0; i < image_size; i++)
			{
				bi[i] = wf_data[i] * im_data[i*nChannels + c] * (alpha_data[i] > downthresh_for_wf_wb);
				bi[i + image_size] = wb_data[i] * im_data[i*nChannels + c] * (alpha_data[i] < upthresh_for_wf_wb);
			}
			for (int i = 0; i < image_size; i++)
				bs[i] = im_data[i*nChannels + c];

			
			ZQ_DImage<double> x0(image_size * 2, 1);
			double*& x0_data = x0.data();
			int it;
	
			if (!ZQ_PCGSolver::PCG_sparse_unsquare(A, b_data, x0_data, max_iter, 1e-16, x_data, it, display))
			{
				printf("failed to call ZQ_PCGSolver::PCG_sparse_unsquare!\n");
				ZQ_TaucsBase::ZQ_taucs_ccs_free(A);
				return false;
			}	
			for (int i = 0; i < image_size; i++)
			{
				fore_data[i*nChannels + c] = x_data[i];
				back_data[i*nChannels + c] = x_data[i + image_size];
			}
		}

		ZQ_TaucsBase::ZQ_taucs_ccs_free(A);

		return true;
	}

	template<class T>
	bool ZQ_ClosedFormImageMatting::SolveForeBack_2dir(const ZQ_DImage<T>& im, const ZQ_DImage<T> alpha, ZQ_DImage<T>& fore, ZQ_DImage<T>& back, int max_iter, bool display /*= false*/)
	{
		int width = im.width();
		int height = im.height();
		int image_size = width*height;
		int nChannels = im.nchannels();
		if (!alpha.matchDimension(width, height, 1))
			return false;
		if (!fore.matchDimension(width, height, nChannels))
			fore.allocate(width, height, nChannels);
		else
			fore.reset();
		if (!back.matchDimension(width, height, nChannels))
			back.allocate(width, height, nChannels);
		else
			back.reset();

		const T*& alpha_data = alpha.data();
		const T*& im_data = im.data();
		T*& fore_data = fore.data();
		T*& back_data = back.data();

		const float weight_for_wgf_wgb = 0.003;
		const float upweight_for_wf_wb = 100;
		const float midweight_for_wf_wb = 0.03;
		const float downweight_for_wf_wb = 0.01;
		const float upthresh_for_wf_wb = 0.98;
		const float midupthresh_for_wf_wb = 0.7;
		const float middownthresh_for_wf_wb = 0.3;
		const float downthresh_for_wf_wb = 0.02;

		ZQ_DImage<T> gx, gy, g_ur, g_dr;
		_getGradientofAlpha_2dir(alpha, gx, gy, upthresh_for_wf_wb, downthresh_for_wf_wb);
		T*& gx_data = gx.data();
		T*& gy_data = gy.data();
		
		ZQ_DImage<T> wgf(image_size * 2, 1);
		ZQ_DImage<T> wgb(image_size * 2, 1);
		T*& wgf_data = wgf.data();
		T*& wgb_data = wgb.data();


		for (int i = 0; i < image_size; i++)
		{
			wgf_data[i] = sqrt(fabs(gx_data[i])) + weight_for_wgf_wgb*(1 - alpha_data[i]);
			wgf_data[i + image_size] = sqrt(fabs(gy_data[i])) + weight_for_wgf_wgb*(1 - alpha_data[i]);

			wgb_data[i] = sqrt(fabs(gx_data[i])) + weight_for_wgf_wgb*alpha_data[i];
			wgb_data[i + image_size] = sqrt(fabs(gy_data[i])) + weight_for_wgf_wgb*alpha_data[i];
		}

		ZQ_DImage<T> wf(image_size, 1);
		ZQ_DImage<T> wb(image_size, 1);
		T*& wf_data = wf.data();
		T*& wb_data = wb.data();


		for (int i = 0; i < image_size; i++)
		{
			wf_data[i] = (alpha_data[i]>upthresh_for_wf_wb)*upweight_for_wf_wb
				+ (alpha_data[i] > midupthresh_for_wf_wb) *alpha_data[i] * midweight_for_wf_wb
				+ (alpha_data[i] < downthresh_for_wf_wb) * downweight_for_wf_wb;
			wb_data[i] = (alpha_data[i] < downthresh_for_wf_wb)*upweight_for_wf_wb
				+ (alpha_data[i]<middownthresh_for_wf_wb)*(1 - alpha_data[i])*midweight_for_wf_wb
				+ (alpha_data[i] > upthresh_for_wf_wb)*downweight_for_wf_wb;
		}
		ZQ_SparseMatrix<T> A_mat(image_size * 7, image_size * 2);
		_buildMatrix_for_SolveForeBack_2dir(alpha, wgf, wgb, wf, wb, A_mat, upthresh_for_wf_wb, downthresh_for_wf_wb);
		if (display)
			printf("build matrix for solve fore back done!\n");
		A_mat.ShrinkToFit();
		taucs_ccs_matrix* A = A_mat.ExportCCS(TAUCS_DOUBLE);
		if (A == 0)
		{
			if (display)
				printf("failed to allocate sparse matrix!\n");
			return false;
		}
		A_mat.Clear();

		ZQ_DImage<double> b(image_size * 7, 1);
		double*& b_data = b.data();
		double* bi = b_data;
		double* bs = bi + image_size * 2;
		double* bg = bs + image_size;

		ZQ_DImage<double> x(image_size * 2, 1);
		double*& x_data = x.data();

		for (int c = 0; c < nChannels; c++)
		{
			for (int i = 0; i < image_size; i++)
			{
				bi[i] = wf_data[i] * im_data[i*nChannels + c] * (alpha_data[i] > downthresh_for_wf_wb);
				bi[i + image_size] = wb_data[i] * im_data[i*nChannels + c] * (alpha_data[i] < upthresh_for_wf_wb);
			}
			for (int i = 0; i < image_size; i++)
				bs[i] = im_data[i*nChannels + c];


			ZQ_DImage<double> x0(image_size * 2, 1);
			double*& x0_data = x0.data();
			int it;

			if (!ZQ_PCGSolver::PCG_sparse_unsquare(A, b_data, x0_data, max_iter, 1e-16, x_data, it, display))
			{
				printf("failed to call ZQ_PCGSolver::PCG_sparse_unsquare!\n");
				ZQ_TaucsBase::ZQ_taucs_ccs_free(A);
				return false;
			}
			for (int i = 0; i < image_size; i++)
			{
				fore_data[i*nChannels + c] = x_data[i];
				back_data[i*nChannels + c] = x_data[i + image_size];
			}
		}

		ZQ_TaucsBase::ZQ_taucs_ccs_free(A);

		return true;
	}

	template<class T>
	bool ZQ_ClosedFormImageMatting::SolveForeBack_ori_paper(const ZQ_DImage<T>& im, const ZQ_DImage<T> alpha, ZQ_DImage<T>& fore, ZQ_DImage<T>& back, int max_iter, bool display /*= false*/)
	{
		int width = im.width();
		int height = im.height();
		int image_size = width*height;
		int nChannels = im.nchannels();
		if (!alpha.matchDimension(width, height, 1))
			return false;
		if (!fore.matchDimension(width, height, nChannels))
			fore.allocate(width, height, nChannels);
		else
			fore.reset();
		if (!back.matchDimension(width, height, nChannels))
			back.allocate(width, height, nChannels);
		else
			back.reset();

		const T*& alpha_data = alpha.data();
		const T*& im_data = im.data();
		T*& fore_data = fore.data();
		T*& back_data = back.data();

		ZQ_DImage<int> unknown_index_map(width, height);
		int*& unknown_index_data = unknown_index_map.data();
		for (int i = 0; i < width*height; i++)
			unknown_index_data[i] = -1;
		
		ZQ_DImage<bool> consts_map(width, height);
		ZQ_DImage<bool> dilate_consts_map(width, height);
		bool*& consts_map_data = consts_map.data();
		bool*& dilate_consts_map_data = dilate_consts_map.data();
		if (true)
		{
			for (int i = 0; i < width*height; i++)
				consts_map_data[i] = alpha_data[i] < 0.02 || alpha_data[i] >= 0.98;
			bool filter2D[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
			for (int i = 0; i < 10; i++)
			{
				ZQ_BinaryImageProcessing::Erode(consts_map_data, dilate_consts_map_data, width, height, filter2D, 1, 1);
				consts_map.swap(dilate_consts_map);
			}
		}
		
		
		int unknown_num = 0;
		for (int i = 0; i < width*height; i++)
		{
			if (consts_map_data[i])
				continue;
			unknown_index_data[i] = unknown_num;
			unknown_num++;
		}

		ZQ_SparseMatrix<T> A_mat(unknown_num * 2, unknown_num * 2);
		_buildMatrix_for_SolveForeBack_ori_paper(alpha, unknown_index_map, A_mat);
		if (display)
		{
			printf("build matrix for solve fore back done!\n");
			printf("m = %d, n = %d, nnz = %d\n", A_mat.GetRow(), A_mat.GetCol(), A_mat.GetNNZ());
		}
		A_mat.ShrinkToFit();
		taucs_ccs_matrix* A = A_mat.ExportCCS(TAUCS_DOUBLE);
		if (A == 0)
		{
			if (display)
				printf("failed to allocate sparse matrix!\n");
			return false;
		}
		A_mat.Clear();

		ZQ_DImage<double> b(unknown_num * 2, 1);
		double*& b_data = b.data();
		
		ZQ_DImage<double> x(unknown_num * 2, 1);
		double*& x_data = x.data();

		for (int c = 0; c < nChannels; c++)
		{
			for (int i = 0; i < image_size; i++)
			{
				int cur_id = unknown_index_data[i];
				if (cur_id >= 0)
				{
					b_data[cur_id] = alpha_data[i] * im_data[i*nChannels + c];
					b_data[cur_id + unknown_num] = (1 - alpha_data[i]) * im_data[i*nChannels + c];
				}
			}
			

			ZQ_DImage<double> x0(unknown_num * 2, 1);
			double*& x0_data = x0.data();
			int it;

			if (!ZQ_PCGSolver::PCG(A, b_data, x0_data, max_iter, 1e-16, x_data, it, display))
			{
				printf("failed to call ZQ_PCGSolver::PCG!\n");
				ZQ_TaucsBase::ZQ_taucs_ccs_free(A);
				return false;
			}
			if (display)
			{
				printf("pcg it = %d\n", it);
			}
			for (int i = 0; i < image_size; i++)
			{
				int cur_id = unknown_index_data[i];
				if (cur_id >= 0)
				{
					fore_data[i*nChannels + c] = x_data[cur_id];
					back_data[i*nChannels + c] = x_data[cur_id + unknown_num];
				}
				else
				{
					fore_data[i*nChannels + c] = alpha_data[i] > 0.5 ? im_data[i*nChannels + c] : 0;
					back_data[i*nChannels + c] = alpha_data[i] < 0.5 ? im_data[i*nChannels + c] : 0;
				}
				
			}
		}

		ZQ_TaucsBase::ZQ_taucs_ccs_free(A);

		return true;
	}

	template<class T>
	bool ZQ_ClosedFormImageMatting::_buildMatrix_for_SolveAlpha(const ZQ_DImage<T>& im, const ZQ_DImage<bool>& consts_map, const ZQ_DImage<T>& consts_vals, int win_size, float epsilon, ZQ_SparseMatrix<T>& A_mat, ZQ_DImage<double>& b, bool tranposed/* = false*/)
	{
		int width = im.width();
		int height = im.height();
		int nChannels = im.nchannels();
		const T*& im_data = im.data();
		const bool*& consts_map_data = consts_map.data();
		const T*& consts_vals_data = consts_vals.data();
		b.allocate(width*height, 1);
		double*& b_data = b.data();

		int neighbor_size = (win_size * 2 + 1)*(win_size * 2 + 1);
		ZQ_DImage<T> mu(1, 1, nChannels);
		T*& mu_data = mu.data();
		
		ZQ_DImage<T> win_im(neighbor_size, nChannels);
		T*& win_im_data = win_im.data();

		ZQ_Matrix<double> cov_mat(nChannels, nChannels);
		ZQ_Matrix<double> inv_cov_mat(nChannels, nChannels);

		ZQ_DImage<T> cov_val(nChannels, nChannels);
		T*& cov_val_data = cov_val.data();
		ZQ_DImage<T> inv_cov_val(nChannels, nChannels);
		T*& inv_cov_val_data = inv_cov_val.data();
		ZQ_DImage<T> win_im_inv_cov(neighbor_size, nChannels);
		T*& win_im_inv_cov_data = win_im_inv_cov.data();
		ZQ_DImage<T> win_t_vals(neighbor_size, neighbor_size);
		T*& win_t_vals_data = win_t_vals.data();

		ZQ_DImage<int> pixel_idx(neighbor_size, 1);
		int*& pixel_idx_data = pixel_idx.data();
		ZQ_DImage<T> row_sum(width*height, 1);
		T*& row_sum_data = row_sum.data();
		int len = 0;
		for (int w = win_size; w < width - win_size; w++)
		{
			for (int h = win_size; h < height - win_size; h++)
			{
				if (consts_map_data[h*width + w])
					continue;

				pixel_idx.reset();
				int win_idx = 0;
				for (int hh = h - win_size; hh <= h + win_size; hh++)
				{
					for (int ww = w - win_size; ww <= w + win_size; ww++)
					{
						for (int c = 0; c < nChannels; c++)
							win_im_data[win_idx*nChannels + c] = im_data[(hh*width + ww)*nChannels + c];
						pixel_idx_data[win_idx] = hh*width + ww;
						win_idx++;
					}
				}

				/** compute mu begin **/
				mu.reset();
				for (int ii = 0; ii < neighbor_size; ii++)
				{
					for (int c = 0; c < nChannels; c++)
						mu_data[c] += win_im_data[ii*nChannels + c];
				}
				for (int c = 0; c < nChannels; c++)
				{
					mu_data[c] /= neighbor_size;
				}
				/** compute mu end **/

				/** compute cov begin **/
				for (int ii = 0; ii < neighbor_size; ii++)
				{
					for (int c = 0; c < nChannels; c++)
						win_im_data[ii*nChannels + c] -= mu_data[c];
				}
				cov_val.reset();
				for (int ii = 0; ii < neighbor_size; ii++)
				{
					for (int ic = 0; ic < nChannels; ic++)
					{
						for (int jc = 0; jc < nChannels; jc++)
							cov_val_data[ic*nChannels + jc] += win_im_data[ii*nChannels + ic] * win_im_data[ii*nChannels + jc];
					}
				}
				for (int cc = 0; cc < nChannels*nChannels; cc++)
				{
					cov_val_data[cc] /= neighbor_size;
				}

				/** compute cov end **/

				/** compute inv(cov+eps/|wk|*eye(nChannels)) begin **/
				for (int ic = 0; ic < nChannels; ic++)
				{
					for (int jc = 0; jc < nChannels; jc++)
					{
						cov_mat.SetData(ic, jc, cov_val_data[ic*nChannels + jc] + ((ic == jc) ? (epsilon / (double)neighbor_size) : 0));
					}
				}
				if (!ZQ_SVD::Invert(cov_mat, inv_cov_mat))
				{
					return false;
				}
				for (int ic = 0; ic < nChannels; ic++)
				{
					for (int jc = 0; jc < nChannels; jc++)
					{
						bool flag;
						inv_cov_val_data[ic*nChannels + jc] = inv_cov_mat.GetData(ic, jc, flag);
					}
				}

				/** compute inv(cov+eps/|wk|*eye(nChannels)) end **/
				win_im_inv_cov.reset();
				for (int i = 0; i < neighbor_size; i++)
				{
					for (int j = 0; j < nChannels; j++)
					{
						for (int k = 0; k < nChannels; k++)
							win_im_inv_cov_data[i*nChannels + j] += win_im_data[i*nChannels + k] * inv_cov_val_data[k*nChannels + j];
					}
				}
				win_t_vals.reset();
				for (int i = 0; i < neighbor_size; i++)
				{
					for (int j = 0; j < neighbor_size; j++)
					{
						for (int k = 0; k < nChannels; k++)
							win_t_vals_data[i*neighbor_size + j] += win_im_inv_cov_data[i*nChannels + k] * win_im_data[j*nChannels + k];
					}
				}

				for (int i = 0; i < neighbor_size*neighbor_size; i++)
				{
					win_t_vals_data[i] += 1;
					win_t_vals_data[i] /= neighbor_size;
				}

				for (int ii = 0; ii < neighbor_size; ii++)
				{
					for (int jj = 0; jj < neighbor_size; jj++)
					{
						int row_idx = pixel_idx_data[ii];
						int col_idx = pixel_idx_data[jj];
						if (!tranposed)
						{
							A_mat.AddTo(row_idx, col_idx, -win_t_vals_data[ii*neighbor_size + jj]);
							
						}
						else
						{
							A_mat.AddTo(col_idx, row_idx, -win_t_vals_data[ii*neighbor_size + jj]);
						}

						row_sum_data[row_idx] += win_t_vals_data[ii*neighbor_size + jj];
					}
				}
			}
		}

		for (int i = 0; i < width*height; i++)
		{
			A_mat.AddTo(i, i, row_sum_data[i]);
		}
		double lambda = 100.0;
		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				int idx = h*width + w;
				A_mat.AddTo(idx, idx, lambda*(consts_map_data[idx] ? 1.0 : 0.0));
				b_data[idx] = lambda*(consts_map_data[idx] ? 1.0 : 0.0) * consts_vals_data[idx];
			}
		}
		A_mat.ShrinkToFit();
		return true;
	}

	template<class T>
	void ZQ_ClosedFormImageMatting::_getGradientofAlpha_4dir(const ZQ_DImage<T>& alpha, ZQ_DImage<T>& gx, ZQ_DImage<T>& gy, ZQ_DImage<T>& g_ur, ZQ_DImage<T>& g_dr, float upthresh/* = 0.98*/, float downthresh/* = 0.02*/)
	{
		int width = alpha.width();
		int height = alpha.height();
	
		if (!gx.matchDimension(width, height, 1))
			gx.allocate(width, height);
		else
			gx.reset();

		if (!gy.matchDimension(width, height, 1))
			gy.allocate(width, height);
		else
			gy.reset();

		if (!g_ur.matchDimension(width, height, 1))
			g_ur.allocate(width, height);
		else
			g_ur.reset();

		if (!g_dr.matchDimension(width, height, 1))
			g_dr.allocate(width, height);
		else
			g_dr.reset();

		const T*& alpha_data = alpha.data();
		T*& gx_data = gx.data();
		T*& gy_data = gy.data();
		T*& g_ur_data = g_ur.data();
		T*& g_dr_data = g_dr.data();

		ZQ_DImage<bool> mask(width, height);
		bool*& mask_data = mask.data();
		for (int i = 0; i < width*height; i++)
			mask_data[i] = alpha_data[i] >= downthresh && alpha_data[i] <= upthresh;

		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width - 1; w++)
			{
				int offset = h*width + w;
				int offset1 = offset + 1;
				if (!mask_data[offset] && !mask_data[offset1])
					continue;
				gx_data[offset] = alpha_data[offset] - alpha_data[offset1];
			}
		}
		
		for (int h = 0; h < height - 1; h++)
		{
			for (int w = 0; w < width; w++)
			{
				int offset = h*width + w;
				int offset1 = offset + width;
				if (!mask_data[offset] && !mask_data[offset1])
					continue;
				gy_data[offset] = alpha_data[offset] - alpha_data[offset1];
			}
		}

		for (int h = 0; h < height - 1; h++)
		{
			for (int w = 0; w < width - 1; w++)
			{
				int offset = h*width + w;
				int offset1 = offset + width + 1;
				if (!mask_data[offset] && !mask_data[offset1])
					continue;
				g_ur_data[offset] = alpha_data[offset] - alpha_data[offset1];
			}
		}

		for (int h = 1; h < height; h++)
		{
			for (int w = 0; w < width - 1; w++)
			{
				int offset = h*width + w;
				int offset1 = offset - width + 1;
				if (!mask_data[offset] && !mask_data[offset1])
					continue;
				g_dr_data[offset] = alpha_data[offset] - alpha_data[offset1];
			}
		}
	}
	template<class T>
	void ZQ_ClosedFormImageMatting::_getGradientofAlpha_2dir(const ZQ_DImage<T>& alpha, ZQ_DImage<T>& gx, ZQ_DImage<T>& gy, float upthresh/* = 0.98*/, float downthresh/* = 0.02*/)
	{
		int width = alpha.width();
		int height = alpha.height();

		if (!gx.matchDimension(width, height, 1))
			gx.allocate(width, height);
		else
			gx.reset();

		if (!gy.matchDimension(width, height, 1))
			gy.allocate(width, height);
		else
			gy.reset();

		const T*& alpha_data = alpha.data();
		T*& gx_data = gx.data();
		T*& gy_data = gy.data();
		
		ZQ_DImage<bool> mask(width, height);
		bool*& mask_data = mask.data();
		for (int i = 0; i < width*height; i++)
			mask_data[i] = alpha_data[i] >= downthresh && alpha_data[i] <= upthresh;

		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width - 1; w++)
			{
				int offset = h*width + w;
				int offset1 = offset + 1;
				if (!mask_data[offset] && !mask_data[offset1])
					continue;
				gx_data[offset] = alpha_data[offset] - alpha_data[offset1];
			}
		}

		for (int h = 0; h < height - 1; h++)
		{
			for (int w = 0; w < width; w++)
			{
				int offset = h*width + w;
				int offset1 = offset + width;
				if (!mask_data[offset] && !mask_data[offset1])
					continue;
				gy_data[offset] = alpha_data[offset] - alpha_data[offset1];
			}
		}
	}


	template<class T>
	void ZQ_ClosedFormImageMatting::_buildMatrix_for_SolveForeBack_4dir(const ZQ_DImage<T>& alpha, const ZQ_DImage<T>& wgf, const ZQ_DImage<T>& wgb, const ZQ_DImage<T>& wf, const ZQ_DImage<T>& wb,
		ZQ_SparseMatrix<T>& Amat, float upthresh/* = 0.98*/, float downthresh/* = 0.02*/)
	{
		int width = alpha.width();
		int height = alpha.height();
		int image_size = width*height;
		int col_off_f = 0;
		int col_off_b = image_size;
		int row_off_Ai = 0;
		int row_off_As = row_off_Ai + image_size * 2;
		int row_off_Ag = row_off_As + image_size;
		int row_off_Ag_wgf_gx = row_off_Ag;
		int row_off_Ag_wgf_gy = row_off_Ag_wgf_gx + image_size;
		int row_off_Ag_wgf_g_ur = row_off_Ag_wgf_gy + image_size;
		int row_off_Ag_wgf_g_dr = row_off_Ag_wgf_g_ur + image_size;
		int row_off_Ag_wgb_gx = row_off_Ag_wgf_g_dr + image_size;
		int row_off_Ag_wgb_gy = row_off_Ag_wgb_gx + image_size;
		int row_off_Ag_wgb_g_ur = row_off_Ag_wgb_gy + image_size;
		int row_off_Ag_wgb_g_dr = row_off_Ag_wgb_g_ur + image_size;

		const T*& alpha_data = alpha.data();
		const T*& wgf_data = wgf.data();
		const T* wgf_gx_data = wgf_data;
		const T* wgf_gy_data = wgf_gx_data + image_size;
		const T* wgf_g_ur_data = wgf_gy_data + image_size;
		const T* wgf_g_dr_data = wgf_g_ur_data + image_size;
		const T*& wgb_data = wgb.data();
		const T* wgb_gx_data = wgb_data;
		const T* wgb_gy_data = wgb_gx_data + image_size;
		const T* wgb_g_ur_data = wgb_gy_data + image_size;
		const T* wgb_g_dr_data = wgb_g_ur_data + image_size;
		const T*& wf_data = wf.data();
		const T*& wb_data = wb.data();


		//Ai
		for (int i = 0; i < image_size; i++)
		{
			int row_id_f = row_off_Ai + i;
			int col_id_f = i;
			int row_id_b = row_off_Ai + i + image_size;
			int col_id_b = i + image_size;
			Amat.AddTo(row_id_f, col_id_f, wf_data[i]);
			Amat.AddTo(row_id_b, col_id_b, wb_data[i]);
		}

		//As
		for (int i = 0; i < image_size; i++)
		{
			int row_id_f = row_off_As + i;
			int col_id_f = i;
			int row_id_b = row_off_As + i;
			int col_id_b = i + image_size;
			Amat.AddTo(row_id_f, col_id_f, alpha_data[i]);
			Amat.AddTo(row_id_b, col_id_b, 1 - alpha_data[i]);
		}

		//Ag

		ZQ_DImage<bool> mask(image_size,1);
		bool*& mask_data = mask.data();
		for (int i = 0; i < image_size; i++)
			mask_data[i] = alpha_data[i] >= downthresh && alpha_data[i] <= upthresh;

		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width - 1; w++)
			{
				int offset = h*width + w;
				int offset1 = offset + 1;
				if (!mask_data[offset] && !mask_data[offset1])
					continue;
				int row_id_f = row_off_Ag_wgf_gx + offset;
				int col_id_f1 = col_off_f + offset;
				int col_id_f2 = col_off_f + offset1;
				int row_id_b = row_off_Ag_wgb_gx + offset;
				int col_id_b1 = col_off_b + offset;
				int col_id_b2 = col_off_b + offset1;
				Amat.AddTo(row_id_f, col_id_f1, wgf_gx_data[offset]);
				Amat.AddTo(row_id_f, col_id_f2, -wgf_gx_data[offset]);
				Amat.AddTo(row_id_b, col_id_b1, wgb_gx_data[offset]);
				Amat.AddTo(row_id_b, col_id_b2, -wgb_gx_data[offset]);
			}
		}

		for (int h = 0; h < height - 1; h++)
		{
			for (int w = 0; w < width; w++)
			{
				int offset = h*width + w;
				int offset1 = offset + width;
				if (!mask_data[offset] && !mask_data[offset1])
					continue;
				int row_id_f = row_off_Ag_wgf_gy + offset;
				int col_id_f1 = col_off_f + offset;
				int col_id_f2 = col_off_f + offset1;
				int row_id_b = row_off_Ag_wgb_gy + offset;
				int col_id_b1 = col_off_b + offset;
				int col_id_b2 = col_off_b + offset1;
				Amat.AddTo(row_id_f, col_id_f1, wgf_gy_data[offset]);
				Amat.AddTo(row_id_f, col_id_f2, -wgf_gy_data[offset]);
				Amat.AddTo(row_id_b, col_id_b1, wgb_gy_data[offset]);
				Amat.AddTo(row_id_b, col_id_b2, -wgb_gy_data[offset]);
			}
		}

		for (int h = 0; h < height - 1; h++)
		{
			for (int w = 0; w < width - 1; w++)
			{
				int offset = h*width + w;
				int offset1 = offset + width + 1;
				if (!mask_data[offset] && !mask_data[offset1])
					continue;
				int row_id_f = row_off_Ag_wgf_g_ur + offset;
				int col_id_f1 = col_off_f + offset;
				int col_id_f2 = col_off_f + offset1;
				int row_id_b = row_off_Ag_wgb_g_ur + offset;
				int col_id_b1 = col_off_b + offset;
				int col_id_b2 = col_off_b + offset1;
				Amat.AddTo(row_id_f, col_id_f1, wgf_g_ur_data[offset]);
				Amat.AddTo(row_id_f, col_id_f2, -wgf_g_ur_data[offset]);
				Amat.AddTo(row_id_b, col_id_b1, wgb_g_ur_data[offset]);
				Amat.AddTo(row_id_b, col_id_b2, -wgb_g_ur_data[offset]);
			}
		}

		for (int h = 1; h < height; h++)
		{
			for (int w = 0; w < width - 1; w++)
			{
				int offset = h*width + w;
				int offset1 = offset - width + 1;
				if (!mask_data[offset] && !mask_data[offset1])
					continue;
				int row_id_f = row_off_Ag_wgf_g_dr + offset;
				int col_id_f1 = col_off_f + offset;
				int col_id_f2 = col_off_f + offset1;
				int row_id_b = row_off_Ag_wgb_g_dr + offset;
				int col_id_b1 = col_off_b + offset;
				int col_id_b2 = col_off_b + offset1;
				Amat.AddTo(row_id_f, col_id_f1, wgf_g_dr_data[offset]);
				Amat.AddTo(row_id_f, col_id_f2, -wgf_g_dr_data[offset]);
				Amat.AddTo(row_id_b, col_id_b1, wgb_g_dr_data[offset]);
				Amat.AddTo(row_id_b, col_id_b2, -wgb_g_dr_data[offset]);
			}
		}
	}

	template<class T>
	void ZQ_ClosedFormImageMatting::_buildMatrix_for_SolveForeBack_2dir(const ZQ_DImage<T>& alpha, const ZQ_DImage<T>& wgf, const ZQ_DImage<T>& wgb, const ZQ_DImage<T>& wf, const ZQ_DImage<T>& wb,
		ZQ_SparseMatrix<T>& Amat, float upthresh/* = 0.98*/, float downthresh/* = 0.02*/)
	{
		int width = alpha.width();
		int height = alpha.height();
		int image_size = width*height;
		int col_off_f = 0;
		int col_off_b = image_size;
		int row_off_Ai = 0;
		int row_off_As = row_off_Ai + image_size * 2;
		int row_off_Ag = row_off_As + image_size;
		int row_off_Ag_wgf_gx = row_off_Ag;
		int row_off_Ag_wgf_gy = row_off_Ag_wgf_gx + image_size;
		int row_off_Ag_wgb_gx = row_off_Ag_wgf_gy + image_size;
		int row_off_Ag_wgb_gy = row_off_Ag_wgb_gx + image_size;

		const T*& alpha_data = alpha.data();
		const T*& wgf_data = wgf.data();
		const T* wgf_gx_data = wgf_data;
		const T* wgf_gy_data = wgf_gx_data + image_size;
		const T*& wgb_data = wgb.data();
		const T* wgb_gx_data = wgb_data;
		const T* wgb_gy_data = wgb_gx_data + image_size;
		const T*& wf_data = wf.data();
		const T*& wb_data = wb.data();


		//Ai
		for (int i = 0; i < image_size; i++)
		{
			int row_id_f = row_off_Ai + i;
			int col_id_f = i;
			int row_id_b = row_off_Ai + i + image_size;
			int col_id_b = i + image_size;
			Amat.AddTo(row_id_f, col_id_f, wf_data[i]);
			Amat.AddTo(row_id_b, col_id_b, wb_data[i]);
		}

		//As
		for (int i = 0; i < image_size; i++)
		{
			int row_id_f = row_off_As + i;
			int col_id_f = i;
			int row_id_b = row_off_As + i;
			int col_id_b = i + image_size;
			Amat.AddTo(row_id_f, col_id_f, alpha_data[i]);
			Amat.AddTo(row_id_b, col_id_b, 1 - alpha_data[i]);
		}

		//Ag

		ZQ_DImage<bool> mask(image_size, 1);
		bool*& mask_data = mask.data();
		for (int i = 0; i < image_size; i++)
			mask_data[i] = alpha_data[i] >= downthresh && alpha_data[i] <= upthresh;

		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width - 1; w++)
			{
				int offset = h*width + w;
				int offset1 = offset + 1;
				if (!mask_data[offset] && !mask_data[offset1])
					continue;
				int row_id_f = row_off_Ag_wgf_gx + offset;
				int col_id_f1 = col_off_f + offset;
				int col_id_f2 = col_off_f + offset1;
				int row_id_b = row_off_Ag_wgb_gx + offset;
				int col_id_b1 = col_off_b + offset;
				int col_id_b2 = col_off_b + offset1;
				Amat.AddTo(row_id_f, col_id_f1, wgf_gx_data[offset]);
				Amat.AddTo(row_id_f, col_id_f2, -wgf_gx_data[offset]);
				Amat.AddTo(row_id_b, col_id_b1, wgb_gx_data[offset]);
				Amat.AddTo(row_id_b, col_id_b2, -wgb_gx_data[offset]);
			}
		}

		for (int h = 0; h < height - 1; h++)
		{
			for (int w = 0; w < width; w++)
			{
				int offset = h*width + w;
				int offset1 = offset + width;
				if (!mask_data[offset] && !mask_data[offset1])
					continue;
				int row_id_f = row_off_Ag_wgf_gy + offset;
				int col_id_f1 = col_off_f + offset;
				int col_id_f2 = col_off_f + offset1;
				int row_id_b = row_off_Ag_wgb_gy + offset;
				int col_id_b1 = col_off_b + offset;
				int col_id_b2 = col_off_b + offset1;
				Amat.AddTo(row_id_f, col_id_f1, wgf_gy_data[offset]);
				Amat.AddTo(row_id_f, col_id_f2, -wgf_gy_data[offset]);
				Amat.AddTo(row_id_b, col_id_b1, wgb_gy_data[offset]);
				Amat.AddTo(row_id_b, col_id_b2, -wgb_gy_data[offset]);
			}
		}
	}

	template<class T>
	void ZQ_ClosedFormImageMatting::_buildMatrix_for_SolveForeBack_ori_paper(const ZQ_DImage<T>& alpha, const ZQ_DImage<int>& unknown_index_map, ZQ_SparseMatrix<T>& Amat)
	{
		ZQ_DImage<T> alpha_x, alpha_y;
		alpha.dx(alpha_x, true);
		alpha.dy(alpha_y, true);

		const T*& alpha_data = alpha.data();
		T*& alpha_x_data = alpha_x.data();
		T*& alpha_y_data = alpha_y.data();
		const int*& unknown_index_data = unknown_index_map.data();
		int width = alpha.width();
		int height = alpha.height();
		
		int unknown_num = Amat.GetCol() / 2;
		int F_offset = 0;
		int B_offset = unknown_num;

		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				int offset = h*width + w;
				if (unknown_index_data[offset] < 0)
					continue;
				
				T cur_alpha = alpha_data[offset];
				T cur_1_alpha = 1 - cur_alpha;
				T cur_1_alpha_mul_alpha = cur_1_alpha*cur_alpha;
				int cur_id = unknown_index_data[offset];
				if (cur_alpha != 0)
				{
					Amat.AddTo(cur_id, cur_id, cur_alpha*cur_alpha);
				}
				if (cur_1_alpha != 0)
				{
					Amat.AddTo(cur_id + B_offset, cur_id + B_offset, cur_1_alpha*cur_1_alpha);
				}
				if (cur_1_alpha_mul_alpha != 0)
				{
					Amat.AddTo(cur_id, cur_id + B_offset, cur_1_alpha_mul_alpha);
					Amat.AddTo(cur_id + B_offset, cur_id, cur_1_alpha_mul_alpha);
				}

				//
				T cur_alpha_x = fabs(alpha_x_data[offset]);
				T cur_alpha_y = fabs(alpha_y_data[offset]);
				if (cur_alpha_x != 0 && w < width-1)
				{
					int neighbor_id = unknown_index_data[offset + 1];
					if (neighbor_id >= 0)
					{
						Amat.AddTo(cur_id, cur_id, cur_alpha_x);
						Amat.AddTo(neighbor_id, neighbor_id, cur_alpha_x);
						Amat.AddTo(cur_id, neighbor_id, -cur_alpha_x);
						Amat.AddTo(neighbor_id, cur_id, -cur_alpha_x);

						Amat.AddTo(cur_id + B_offset, cur_id + B_offset, cur_alpha_x);
						Amat.AddTo(neighbor_id + B_offset, neighbor_id + B_offset, cur_alpha_x);
						Amat.AddTo(cur_id + B_offset, neighbor_id + B_offset, -cur_alpha_x);
						Amat.AddTo(neighbor_id + B_offset, cur_id + B_offset, -cur_alpha_x);
					}
					
				}
				if (cur_alpha_y != 0 && h < height - 1)
				{
					int neighbor_id = unknown_index_data[offset + width];
					if (neighbor_id >= 0)
					{
						Amat.AddTo(cur_id, cur_id, cur_alpha_y);
						Amat.AddTo(neighbor_id, neighbor_id, cur_alpha_y);
						Amat.AddTo(cur_id, neighbor_id, -cur_alpha_y);
						Amat.AddTo(neighbor_id, cur_id, -cur_alpha_y);

						Amat.AddTo(cur_id + B_offset, cur_id + B_offset, cur_alpha_y);
						Amat.AddTo(neighbor_id + B_offset, neighbor_id + B_offset, cur_alpha_y);
						Amat.AddTo(cur_id + B_offset, neighbor_id + B_offset, -cur_alpha_y);
						Amat.AddTo(neighbor_id + B_offset, cur_id + B_offset, -cur_alpha_y);
					}
				}
			}
		}

	}

	template<class T>
	void ZQ_ClosedFormImageMatting::_downSampleImage(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, int size)
	{
		int width = input.width();
		int height = input.height();
		int nChannels = input.nchannels();
		const T*& input_data = input.data();
		int fsize = 0;
		T filter[5];
		if (size == 1)
		{
			fsize = 1;
			T cur_filter[3] = { 1.0 / 4, 2.0 / 4, 1.0 / 4 };
			memcpy(filter, cur_filter, sizeof(T)* 3);
		}
		else if (size ==2)
		{
			fsize = 2;
			T cur_filter[5] = { 1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16 };
			memcpy(filter, cur_filter, sizeof(T)* 5);
		}
		else
		{
			fsize = 0;
			T cur_filter[1] = { 1.0 };
			memcpy(filter, cur_filter, sizeof(T)* 1);
		}

		ZQ_DImage<T> tmp1(width, height, nChannels);
		ZQ_DImage<T> tmp2(width, height, nChannels);
		T*& tmp1_data = tmp1.data();
		T*& tmp2_data = tmp2.data();
		ZQ_ImageProcessing::Hfiltering(input_data, tmp1_data, width, height, nChannels, filter, fsize, false);
		ZQ_ImageProcessing::Vfiltering(tmp1_data, tmp2_data, width, height, nChannels, filter, fsize, false);
		int start_h = fsize;
		int end_h = height - 1 - fsize;
		int small_height = (end_h - start_h) / 2 + 1;
		int start_w = fsize;
		int end_w = width - 1 - fsize;
		int small_width = (end_w - start_w) / 2 + 1;
		output.allocate(small_width, small_height, nChannels);
		T*& output_data = output.data();
		for (int h = 0, bh = start_h; h < small_height; h++, bh += 2)
		{
			for (int w = 0, bw = start_w; w < small_width; w++, bw += 2)
			{
				for (int c = 0; c < nChannels; c++)
					output_data[(h*small_width + w)*nChannels + c] = tmp2_data[(bh*width + bw)*nChannels + c];
			}
		}
	}

	template<class T>
	void ZQ_ClosedFormImageMatting::_downSample(const ZQ_DImage<T>& im, const ZQ_DImage<bool>& consts_map, const ZQ_DImage<T>& consts_vals, ZQ_DImage<T>& s_im, ZQ_DImage<bool>& s_const_map, ZQ_DImage<T>& s_consts_vals)
	{
		_downSampleImage(im, s_im, 2);
		
		int width = consts_map.width();
		int height = consts_map.height();
		ZQ_DImage<double> consts_map_d(width, height), s_consts_map_d;
		double*& consts_map_d_data = consts_map_d.data();
		const bool*& consts_map_data = consts_map.data();
		for (int i = 0; i < width*height; i++)
			consts_map_d_data[i] = consts_map_data[i] ? 1.0 : 0.0;
		_downSampleImage(consts_map_d, s_consts_map_d, 2);
		int s_width = s_consts_map_d.width();
		int s_height = s_consts_map_d.height();
		s_const_map.allocate(s_width, s_height);
		bool*& s_consts_map_data = s_const_map.data();
		double*& s_consts_map_d_data = s_consts_map_d.data();
		for (int i = 0; i < s_width*s_height; i++)
			s_consts_map_data[i] = s_consts_map_d_data[i] > 0.5;

		
		_downSampleImage(consts_vals, s_consts_vals, 2);
		
		T*& s_consts_vals_data = s_consts_vals.data();
		for (int i = 0; i < s_consts_vals.npixels(); i++)
			s_consts_vals_data[i] = round(s_consts_vals_data[i]);
	}

	template<class T>
	void ZQ_ClosedFormImageMatting::_upSampleAlphaUsingImage(const ZQ_DImage<T>& s_alpha, const ZQ_DImage<T>& s_im, const ZQ_DImage<T>& im, ZQ_DImage<T>& alpha, float epsilon, int win_size)
	{
		ZQ_DImage<T> s_coeff, coeff;
		_getLinearCoeff(s_alpha, s_im, s_coeff, epsilon, win_size);
		
		int width = im.width();
		int height = im.height();
		int nChannels = im.nchannels();
		_upSampleImage(s_coeff, width, height, coeff, 1);
		
		alpha.allocate(width, height);
		T*& alpha_data = alpha.data();
		T*& coeff_data = coeff.data();
		const T*& im_data = im.data();
		for (int i = 0; i < width*height; i++)
		{
			alpha_data[i] = coeff_data[i*(nChannels + 1) + nChannels];
			for (int c = 0; c < nChannels; c++)
			{
				alpha_data[i] += coeff_data[i*(nChannels + 1) + c] * im_data[i*nChannels + c];
			}
		}
	}

	template<class T>
	void ZQ_ClosedFormImageMatting::_upSampleImage(const ZQ_DImage<T>& input, int out_width, int out_height, ZQ_DImage<T>& output, int size)
	{
		int width = input.width();
		int height = input.height();
		int nChannels = input.nchannels();
		const T*& input_data = input.data();
		int fsize = 0;
		T filter[5];
		if (size == 1)
		{
			fsize = 1;
			T cur_filter[3] = { 1.0 / 2, 2.0 / 2, 1.0 / 2 };
			memcpy(filter, cur_filter, sizeof(T)* 3);
		}
		else if (size == 2)
		{
			fsize = 2;
			T cur_filter[5] = { 1.0 / 8, 4.0 / 8, 6.0 / 8, 4.0 / 8, 1.0 / 8 };
			memcpy(filter, cur_filter, sizeof(T)* 5);
		}
		else
		{
			fsize = 0;
			T cur_filter[1] = { 1.0 };
			memcpy(filter, cur_filter, sizeof(T)* 1);
		}

		int hd = floor((out_height - height * 2 + 1) / 2.0);
		int hu = ceil((out_height - height * 2 + 1) / 2.0);
		int wd = floor((out_width - width * 2 + 1) / 2.0);
		int wu = ceil((out_width - width * 2 + 1) / 2.0);

		int n_width = out_width + fsize * 2;
		int n_height = out_height + fsize * 2;
		ZQ_DImage<T> n_im(n_width, n_height, nChannels);
		T*& n_im_data = n_im.data();
		for (int h = hd + fsize, ih = 0; h < n_height - hu - fsize; h += 2, ih++)
		{
			for (int w = wd + fsize, iw = 0; w < n_width - wu - fsize; w += 2, iw++)
			{
				memcpy(n_im_data + (h*n_width + w)*nChannels, input_data + (ih*width + iw)*nChannels, sizeof(T)*nChannels);
			}
		}
		
		for (int h = hd + fsize - 2; h >= 0; h-=2)
		{
			memcpy(n_im_data + h*n_width*nChannels, n_im_data + (hd + fsize)*n_width*nChannels, sizeof(T)*n_width*nChannels);
		}
		for (int h = n_height - 1 - hu - fsize + 2; h < n_height; h += 2)
		{
			memcpy(n_im_data + h*n_width*nChannels, n_im_data + (n_height - 1 - hu - fsize)*n_width*nChannels, sizeof(T)*n_width*nChannels);
		}
		
		for (int h = 0; h < n_height; h++)
		{
			for (int w = wd + fsize - 2; w >= 0; w -= 2)
			{
				memcpy(n_im_data + (h*n_width + w)*nChannels, n_im_data + (h*n_width + wd + fsize)*nChannels, sizeof(T)*nChannels);
			}
			for (int w = n_width - 1 - wu - fsize + 2; w < n_width; w += 2)
			{
				memcpy(n_im_data + (h*n_width + w)*nChannels, n_im_data + (h*n_width + n_width - 1 - wu - fsize)*nChannels, sizeof(T)*nChannels);
			}
		}
	
		ZQ_DImage<T> tmp1(n_width, n_height, nChannels);
		ZQ_ImageProcessing::Hfiltering(n_im_data, tmp1.data(), n_width, n_height, nChannels, filter, fsize, false);
		ZQ_ImageProcessing::Vfiltering(tmp1.data(), n_im_data, n_width, n_height, nChannels, filter, fsize, false);

		output.allocate(out_width, out_height, nChannels);
		T*& output_data = output.data();
		for (int h = 0; h < out_height; h++)
		{
			for (int w = 0; w < out_width; w++)
			{
				memcpy(output_data + (h*out_width + w)*nChannels, n_im_data + ((h + fsize)*n_width + w + fsize)*nChannels, sizeof(T)*nChannels);
			}
		}
	}

	template<class T>
	void ZQ_ClosedFormImageMatting::_getLinearCoeff(const ZQ_DImage<T>& alpha, const ZQ_DImage<T>& im, ZQ_DImage<T>& coeff, float epsilon, int win_size)
	{
		int win_Xsize = win_size * 2 + 1;
		int neighbor_size = win_Xsize*win_Xsize;
		int width = im.width();
		int height = im.height();
		int nChannels = im.nchannels();
		int image_size = width*height;

		coeff.allocate(width, height, nChannels + 1);
		const T*& im_data = im.data();
		const T*& alpha_data = alpha.data();
		T*& coeff_data = coeff.data();
		
		ZQ_DImage<int> neighbor_idx(neighbor_size, 1);
		int*& neighbor_idx_data = neighbor_idx.data();
		
		ZQ_Matrix<double> A_mat(neighbor_size + nChannels, nChannels + 1);
		ZQ_Matrix<double> b_mat(neighbor_size + nChannels, 1);
		ZQ_Matrix<double> x_mat(nChannels + 1,1);

		float rt_eps = sqrt(epsilon);
		for (int h = win_size; h < height - win_size; h++)
		{
			for (int w = win_size; w < width - win_size; w++)
			{
				int cur_neighbor_id = 0;
				for (int hh = h - win_size; hh <= h + win_size; hh++)
				{
					for (int ww = w - win_size; ww <= w + win_size; ww++)
					{
						neighbor_idx_data[cur_neighbor_id++] = hh*width + ww;
					}
				}

				for (int r = 0; r < neighbor_size; r++)
				{
					for (int c = 0; c < nChannels; c++)
						A_mat.SetData(r, c, im_data[neighbor_idx_data[r] * nChannels + c]);
					A_mat.SetData(r, nChannels, 1);
					b_mat.SetData(r, 0, alpha_data[neighbor_idx_data[r]]);
				}
				for (int c = 0; c < nChannels; c++)
				{
					A_mat.SetData(neighbor_size + c, c, rt_eps);
				}
				ZQ_SVD::Solve(A_mat, x_mat, b_mat);
				bool flag;
				for (int c = 0; c <= nChannels; c++)
					coeff_data[(h*width + w)*(nChannels + 1) + c] = x_mat.GetData(c, 0, flag);
			}
		}

		for (int h = 0; h < win_size; h++)
		{
			memcpy(coeff_data + h*width*(nChannels + 1), coeff_data + win_size*width*(nChannels + 1), sizeof(T)*width*(nChannels + 1));
		}
		for (int h = height - 1; h > height - 1 - win_size; h--)
		{
			memcpy(coeff_data + h*width*(nChannels + 1), coeff_data + (height - 1 - win_size)*width*(nChannels + 1), sizeof(T)*width*(nChannels + 1));
		}
		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < win_size; w++)
				memcpy(coeff_data + (h*width + w)*(nChannels + 1), coeff_data + (h*width + win_size)*(nChannels + 1), sizeof(T)*(nChannels + 1));
			for (int w = width - 1; w > width - 1 - win_size; w--)
				memcpy(coeff_data + (h*width + w)*(nChannels + 1), coeff_data + (h*width + width - 1 - win_size)*(nChannels + 1), sizeof(T)*(nChannels + 1));
		}
	}

	template<class T>
	void ZQ_ClosedFormImageMatting::_updateConstsMapConstsVals(const ZQ_DImage<T>& alpha, const ZQ_DImage<bool>& consts_map, const ZQ_DImage<T>& consts_vals, 
		ZQ_DImage<bool>& out_consts_map, ZQ_DImage<T>& out_consts_vals, float consts_thresh/* = 0.02*/)
	{
		int width = alpha.width();
		int height = alpha.height();
		ZQ_DImage<T> talpha(width, height);
		T*& talpha_data = talpha.data();
		const T*& alpha_data = alpha.data();
		const bool*& consts_map_data = consts_map.data();
		const T*& consts_vals_data = consts_vals.data();
		for (int i = 0; i < width*height; i++)
			talpha_data[i] = alpha_data[i] * (1 - consts_map_data[i]) + consts_vals_data[i];

		int erode_size = 1;
		bool erode_filter2D[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };

		ZQ_DImage<bool> up_alpha_map(width, height);
		ZQ_DImage<bool> down_alpha_map(width, height);
		ZQ_DImage<bool> tmp_alpha_map(width, height);
		bool*& up_alpha_map_data = up_alpha_map.data();
		bool*& down_alpha_map_data = down_alpha_map.data();
		bool*& tmp_alpha_map_data = tmp_alpha_map.data();


		for (int i = 0; i < width*height; i++)
			tmp_alpha_map_data[i] = alpha_data[i] >= 1 - consts_thresh;
		ZQ_BinaryImageProcessing::Erode(tmp_alpha_map_data, up_alpha_map_data, width, height, erode_filter2D, erode_size, erode_size);
		for (int i = 0; i < width*height; i++)
			tmp_alpha_map_data[i] = alpha_data[i] <= consts_thresh;
		ZQ_BinaryImageProcessing::Erode(tmp_alpha_map_data, down_alpha_map_data, width, height, erode_filter2D, erode_size, erode_size);

		if (!out_consts_map.matchDimension(width,height,1))
			out_consts_map.allocate(width, height,1);
		bool*& out_consts_map_data = out_consts_map.data();
		for (int i = 0; i < width*height; i++)
			out_consts_map_data[i] = consts_map_data[i] || up_alpha_map_data[i] || down_alpha_map_data[i];

		if (!out_consts_vals.matchDimension(width, height, 1))
			out_consts_vals.allocate(width, height, 1);
		T*& out_consts_vals_data = out_consts_vals.data();
		for (int i = 0; i < width*height; i++)
			out_consts_vals_data[i] = (int)(talpha_data[i] + 0.5)*out_consts_map_data[i];
	}
}

#endif