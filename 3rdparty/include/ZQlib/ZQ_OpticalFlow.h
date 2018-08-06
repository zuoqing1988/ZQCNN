#ifndef _ZQ_OPTICAL_FLOW_H_
#define _ZQ_OPTICAL_FLOW_H_
#pragma once

#include "ZQ_DoubleImage.h"
#include "ZQ_PoissonSolver.h"
#include "ZQ_ImageProcessing.h"
#include "ZQ_BinaryImageProcessing.h"
#include "ZQ_GaussianPyramid.h"
#include "ZQ_OpticalFlowOptions.h"
#include "ZQ_WeightedMedian.h"
#include <math.h>
#include <stdlib.h>
#include <vector>
#ifdef ZQLIB_USE_OPENMP
#include <omp.h>
#endif


namespace ZQ
{
	class ZQ_OpticalFlow
	{
	public:
		ZQ_OpticalFlow(void){}
		~ZQ_OpticalFlow(void){}


		/**********************************    functions for users     *******************************/
	public:
		template<class T>
		static void Coarse2Fine_HS_L2(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		/*for stereo disparity*/
		template<class T>
		static void Coarse2Fine_HS_L2(ZQ_DImage<T>& u, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Coarse2Fine_HS_L1(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		/*for stereo disparity*/
		template<class T>
		static void Coarse2Fine_HS_L1(ZQ_DImage<T>& u, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Coarse2Fine_HS_DL1(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Coarse2Fine_ADMM_L2(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Coarse2Fine_ADMM_DL1(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Coarse2Fine_OneDir_Inc_L2(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Coarse2Fine_OneDir_Inc_DL1(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Coarse2Fine_OneDir_Dec_L2(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Coarse2Fine_OneDir_Dec_DL1(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Coarse2Fine_TwoDir_Inc_L2(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Coarse2Fine_TwoDir_Inc_DL1(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Coarse2Fine_TwoDir_Dec_L2(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Coarse2Fine_TwoDir_Dec_DL1(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt);


		/************************* be careful to use this functions *****************************/
	public:
		template<class T>
		static void OneResolution_HS_L2(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		/*for stereo disparity*/
		template<class T>
		static void OneResolution_HS_L2(ZQ_DImage<T>& u, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_HS_L1(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		/*for stereo disparity*/
		template<class T>
		static void OneResolution_HS_L1(ZQ_DImage<T>& u, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_HS_DL1(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_ADMM_L2(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_ADMM_DL1(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_OneDir_Inc_L2(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_OneDir_Inc_DL1(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_OneDir_Dec_L2(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_OneDir_Dec_DL1(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_TwoDir_Inc_L2(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_TwoDir_Inc_DL1(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_TwoDir_Dec_L2(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_TwoDir_Dec_DL1(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt);


		/******************** if you are very familiar with this code, you can inherit this class and access these protected functions *********************************/
	protected:

		template<class T>
		static void getDxs(ZQ_DImage<T>& imdx, ZQ_DImage<T>& imdy, ZQ_DImage<T>& imdt, const ZQ_DImage<T>& im1, const ZQ_DImage<T>& im2, bool isSmooth = true);

		template<class T>
		static void warpFL(ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_DImage<T>& u, const ZQ_DImage<T>& v, bool isBicubic = false);

		template<class T>
		static void Laplacian(ZQ_DImage<T>& output, const ZQ_DImage<T>& input);

		template<class T>
		static void Laplacian(ZQ_DImage<T>& output, const ZQ_DImage<T>& input, const ZQ_DImage<T>& weight);

		// function to convert image to features
		template<class T>
		static void im2feature(ZQ_DImage<T>& imfeature, const ZQ_DImage<T>& im, bool isSmooth, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void detectOcclusion(ZQ_DImage<T>& occ, const ZQ_DImage<T>& u, const ZQ_DImage<T>& v, const ZQ_DImage<T>& im1, const ZQ_DImage<T>& im2, double sigma_div = 0.3, double sigma_im = 0.08);

		template<class T>
		static void detectOcclusion(ZQ_DImage<T>& occ, const ZQ_DImage<T>& u, const ZQ_DImage<T>& im1, const ZQ_DImage<T>& im2, double sigma_div = 0.3, double sigma_im = 0.08);

		template<class T>
		static void denoiseColorWeightedMedianFilter(ZQ_DImage<T>& out_u, ZQ_DImage<T>& out_v, const ZQ_DImage<T>& u, const ZQ_DImage<T>& v, const ZQ_DImage<T>& im, const ZQ_DImage<T>& occ, int weightedMedFiltSize = 2,
			int medFiltSize = 2, double sigma_i = 0.03);

		template<class T>
		static void denoiseColorWeightedMedianFilter(ZQ_DImage<T>& out_u, const ZQ_DImage<T>& u, const ZQ_DImage<T>& im, const ZQ_DImage<T>& occ, int weightedMedFiltSize = 2, int medFiltSize = 2, double sigma_i = 0.03);


		/**********************************  for ADMM frame  ********************8*******************/
		template<class T>
		static void ADMM_F_G(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt,
			void(*funcF)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, ZQ_DImage<T>& /*warpIm2*/, const ZQ_DImage<T>& /*Im1*/, const ZQ_DImage<T>& /*Im2*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_OpticalFlowOptions& /*opt*/),
			void(*funcG)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_OpticalFlowOptions& /*opt*/));

		template<class T>
		static void ADMM_F1_F2_G_first(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_DImage<T>& next_u, const ZQ_DImage<T>& next_v, const ZQ_OpticalFlowOptions& opt,
			void(*funcF1)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, ZQ_DImage<T>& /*warpIm2*/, const ZQ_DImage<T>& /*Im1*/, const ZQ_DImage<T>& /*Im2*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_OpticalFlowOptions& /*opt*/),
			void(*funcF2)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_DImage<T>& /*next_u*/, const ZQ_DImage<T>& /*next_v*/, const ZQ_OpticalFlowOptions& /*opt*/),
			void(*funcG)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_OpticalFlowOptions& /*opt*/));

		template<class T>
		static void ADMM_F1_F2_G_last(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_DImage<T>& pre_u, const ZQ_DImage<T>& pre_v, const ZQ_OpticalFlowOptions& opt,
			void(*funcF1)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, ZQ_DImage<T>& /*warpIm2*/, const ZQ_DImage<T>& /*Im1*/, const ZQ_DImage<T>& /*Im2*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_OpticalFlowOptions& /*opt*/),
			void(*funcF2)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_DImage<T>& /*pre_u*/, const ZQ_DImage<T>& /*pre_v*/, const ZQ_OpticalFlowOptions& /*opt*/),
			void(*funcG)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_OpticalFlowOptions& /*opt*/));


		template<class T>
		static void ADMM_F1_F2_G_middle(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_DImage<T>& pre_u, const ZQ_DImage<T>& pre_v, const ZQ_DImage<T>& next_u, const ZQ_DImage<T>& next_v, const ZQ_OpticalFlowOptions& opt,
			void(*funcF1)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, ZQ_DImage<T>& /*warpIm2*/, const ZQ_DImage<T>& /*Im1*/, const ZQ_DImage<T>& /*Im2*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_OpticalFlowOptions& /*opt*/),
			void(*funcF2)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_DImage<T>& /*pre_u*/, const ZQ_DImage<T>& /*pre_v*/, const ZQ_DImage<T>& /*next_u*/, const ZQ_DImage<T>& /*next_v*/, const ZQ_OpticalFlowOptions& /*opt*/),
			void(*funcG)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_OpticalFlowOptions& /*opt*/));


		/**********************     Proximal functions   ***********************/
		template<class T>
		static void Proximal_F_L2(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_DImage<T>& z_u, const ZQ_DImage<T>& z_v, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Proximal_F_DL1(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_DImage<T>& z_u, const ZQ_DImage<T>& z_v, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Proximal_G(ZQ_DImage<T>& u, ZQ_DImage<T>& v, const ZQ_DImage<T>& z_u, const ZQ_DImage<T>& z_v, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Proximal_F2_First(ZQ_DImage<T>& u, ZQ_DImage<T>& v, const ZQ_DImage<T>& z_u, const ZQ_DImage<T>& z_v, const ZQ_DImage<T>& next_u, const ZQ_DImage<T>& next_v, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Proximal_F2_Last(ZQ_DImage<T>& u, ZQ_DImage<T>& v, const ZQ_DImage<T>& z_u, const ZQ_DImage<T>& z_v, const ZQ_DImage<T>& pre_u, const ZQ_DImage<T>& pre_v, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Proximal_F2_Middle(ZQ_DImage<T>& u, ZQ_DImage<T>& v, const ZQ_DImage<T>& z_u, const ZQ_DImage<T>& z_v, const ZQ_DImage<T>& pre_u, const ZQ_DImage<T>& pre_v, const ZQ_DImage<T>& next_u, const ZQ_DImage<T>& next_v, const ZQ_OpticalFlowOptions& opt);

	};


	/*********************************************************************************/
	/********************************** definitions **********************************/
	/*********************************************************************************/

	/**************************  functions for users *****************************/
	template<class T>
	void ZQ_OpticalFlow::Coarse2Fine_HS_L2(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_GaussianPyramid<T> GPyramid1;
		ZQ_GaussianPyramid<T> GPyramid2;
		if (opt.displayRunningInfo)
		{
			printf("Constructing pyramid...");
		}

		double ratio =
			GPyramid1.ConstructPyramid(Im1, opt.ratioForPyramid, opt.minWidthForPyramid);
		GPyramid2.ConstructPyramid(Im2, opt.ratioForPyramid, opt.minWidthForPyramid);

		if (opt.displayRunningInfo)
			printf("done!\n");

		ZQ_DImage<T> Image1, Image2;

		for (int k = GPyramid1.nlevels() - 1; k >= 0; k--)
		{
			if (opt.displayRunningInfo)
				printf("Pyramid level %d \n", k);

			int width = GPyramid1.Image(k).width();
			int height = GPyramid1.Image(k).height();

			bool isSmooth = true;
			if (k == 0)
				isSmooth = false;

			im2feature(Image1, GPyramid1.Image(k), isSmooth, opt);
			im2feature(Image2, GPyramid2.Image(k), isSmooth, opt);

			if (k == GPyramid1.nlevels() - 1)
			{
				u.allocate(width, height);
				v.allocate(width, height);
				warpIm2.copyData(Image2);

			}
			else
			{
				u.imresize(width, height);
				u.Multiplywith(1.0 / ratio);
				v.imresize(width, height);
				v.Multiplywith(1.0 / ratio);
				warpFL(warpIm2, Image1, Image2, u, v, opt.useCubicWarping);

			}

			OneResolution_HS_L2(u, v, warpIm2, Image1, Image2, opt);
		}
	}


	template<class T>
	void ZQ_OpticalFlow::Coarse2Fine_HS_L2(ZQ_DImage<T>& u, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_GaussianPyramid<T> GPyramid1;
		ZQ_GaussianPyramid<T> GPyramid2;
		if (opt.displayRunningInfo)
		{
			printf("Constructing pyramid...");
		}

		double ratio =
			GPyramid1.ConstructPyramid(Im1, opt.ratioForPyramid, opt.minWidthForPyramid);
		GPyramid2.ConstructPyramid(Im2, opt.ratioForPyramid, opt.minWidthForPyramid);

		if (opt.displayRunningInfo)
			printf("done!\n");

		ZQ_DImage<T> Image1, Image2;
		ZQ_DImage<T> v;
		int nLevels = GPyramid1.nlevels();
		for (int k = nLevels - 1; k >= 0; k--)
		{
			if (opt.displayRunningInfo)
				printf("Pyramid level %d \n", k);

			int width = GPyramid1.Image(k).width();
			int height = GPyramid1.Image(k).height();

			bool isSmooth = true;
			if (k == 0)
				isSmooth = false;

			//im2feature(Image1, GPyramid1.Image(k), isSmooth, opt);
			//im2feature(Image2, GPyramid2.Image(k), isSmooth, opt);
			Image1 = GPyramid1.Image(k);
			Image2 = GPyramid2.Image(k);

			v.allocate(width, height);

			if (k == GPyramid1.nlevels() - 1)
			{
				u.allocate(width, height);
				warpIm2.copyData(Image2);
			}
			else
			{
				u.imresize(width, height);
				u.Multiplywith(1.0 / ratio);
				warpFL(warpIm2, Image1, Image2, u, v, opt.useCubicWarping);

			}
			if (opt.displayRunningInfo)
				printf("k = %d\n", k);
			OneResolution_HS_L2(u, warpIm2, Image1, Image2, opt);

		}
	}

	template<class T>
	void ZQ_OpticalFlow::Coarse2Fine_HS_L1(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_GaussianPyramid<T> GPyramid1;
		ZQ_GaussianPyramid<T> GPyramid2;
		if (opt.displayRunningInfo)
		{
			printf("Constructing pyramid...");
		}

		double ratio =
			GPyramid1.ConstructPyramid(Im1, opt.ratioForPyramid, opt.minWidthForPyramid);
		GPyramid2.ConstructPyramid(Im2, opt.ratioForPyramid, opt.minWidthForPyramid);

		if (opt.displayRunningInfo)
			printf("done!\n");

		ZQ_DImage<T> Image1, Image2;

		for (int k = GPyramid1.nlevels() - 1; k >= 0; k--)
		{
			if (opt.displayRunningInfo)
				printf("Pyramid level %d \n", k);

			int width = GPyramid1.Image(k).width();
			int height = GPyramid1.Image(k).height();

			bool isSmooth = true;
			if (k == 0)
				isSmooth = false;

			im2feature(Image1, GPyramid1.Image(k), isSmooth, opt);
			im2feature(Image2, GPyramid2.Image(k), isSmooth, opt);
			//Image1 = GPyramid1.Image(k);
			//Image2 = GPyramid2.Image(k);

			if (k == GPyramid1.nlevels() - 1)
			{
				u.allocate(width, height);
				v.allocate(width, height);
				warpIm2.copyData(Image2);

			}
			else
			{
				u.imresize(width, height);
				u.Multiplywith(1.0 / ratio);
				v.imresize(width, height);
				v.Multiplywith(1.0 / ratio);
				warpFL(warpIm2, Image1, Image2, u, v, opt.useCubicWarping);

			}

			OneResolution_HS_L1(u, v, warpIm2, Image1, Image2, opt);
		}
	}

	template<class T>
	void ZQ_OpticalFlow::Coarse2Fine_HS_L1(ZQ_DImage<T>& u, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_GaussianPyramid<T> GPyramid1;
		ZQ_GaussianPyramid<T> GPyramid2;
		if (opt.displayRunningInfo)
		{
			printf("Constructing pyramid...");
		}

		double ratio =
			GPyramid1.ConstructPyramid(Im1, opt.ratioForPyramid, opt.minWidthForPyramid);
		GPyramid2.ConstructPyramid(Im2, opt.ratioForPyramid, opt.minWidthForPyramid);

		if (opt.displayRunningInfo)
			printf("done!\n");

		ZQ_DImage<T> Image1, Image2, v;

		for (int k = GPyramid1.nlevels() - 1; k >= 0; k--)
		{
			if (opt.displayRunningInfo)
				printf("Pyramid level %d \n", k);

			int width = GPyramid1.Image(k).width();
			int height = GPyramid1.Image(k).height();

			bool isSmooth = true;
			if (k == 0)
				isSmooth = false;

			//im2feature(Image1, GPyramid1.Image(k), isSmooth, opt);
			//im2feature(Image2, GPyramid2.Image(k), isSmooth, opt);
			Image1 = GPyramid1.Image(k);
			Image2 = GPyramid2.Image(k);

			v.allocate(width, height);
			if (k == GPyramid1.nlevels() - 1)
			{
				u.allocate(width, height);
				warpIm2.copyData(Image2);
			}
			else
			{
				u.imresize(width, height);
				u.Multiplywith(1.0 / ratio);
				warpFL(warpIm2, Image1, Image2, u, v, opt.useCubicWarping);
			}

			OneResolution_HS_L1(u, warpIm2, Image1, Image2, opt);
		}
	}

	template<class T>
	void ZQ_OpticalFlow::Coarse2Fine_HS_DL1(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_GaussianPyramid<T> GPyramid1;
		ZQ_GaussianPyramid<T> GPyramid2;
		if (opt.displayRunningInfo)
		{
			printf("Constructing pyramid...");
		}

		double ratio =
			GPyramid1.ConstructPyramid(Im1, opt.ratioForPyramid, opt.minWidthForPyramid);
		GPyramid2.ConstructPyramid(Im2, opt.ratioForPyramid, opt.minWidthForPyramid);

		if (opt.displayRunningInfo)
			printf("done!\n");

		ZQ_DImage<T> Image1, Image2;

		for (int k = GPyramid1.nlevels() - 1; k >= 0; k--)
		{
			if (opt.displayRunningInfo)
				printf("Pyramid level %d \n", k);

			int width = GPyramid1.Image(k).width();
			int height = GPyramid1.Image(k).height();

			bool isSmooth = true;
			if (k == 0)
				isSmooth = false;

			im2feature(Image1, GPyramid1.Image(k), isSmooth, opt);
			im2feature(Image2, GPyramid2.Image(k), isSmooth, opt);

			if (k == GPyramid1.nlevels() - 1)
			{
				u.allocate(width, height);
				v.allocate(width, height);
				warpIm2.copyData(Image2);

			}
			else
			{
				u.imresize(width, height);
				u.Multiplywith(1.0 / ratio);
				v.imresize(width, height);
				v.Multiplywith(1.0 / ratio);
				warpFL(warpIm2, Image1, Image2, u, v, opt.useCubicWarping);

			}

			OneResolution_HS_DL1(u, v, warpIm2, Image1, Image2, opt);
		}
	}

	template<class T>
	void ZQ_OpticalFlow::Coarse2Fine_ADMM_L2(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_GaussianPyramid<T> GPyramid1;
		ZQ_GaussianPyramid<T> GPyramid2;

		if (opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio =
			GPyramid1.ConstructPyramid(Im1, opt.ratioForPyramid, opt.minWidthForPyramid);
		GPyramid2.ConstructPyramid(Im2, opt.ratioForPyramid, opt.minWidthForPyramid);

		if (opt.displayRunningInfo)
			printf("done!\n");

		ZQ_DImage<T> Image1, Image2;

		for (int k = GPyramid1.nlevels() - 1; k >= 0; k--)
		{
			if (opt.displayRunningInfo)
				printf("Pyramid level %d \n", k);

			int width = GPyramid1.Image(k).width();
			int height = GPyramid1.Image(k).height();

			bool isSmooth = true;
			if (k == 0)
				isSmooth = false;

			im2feature(Image1, GPyramid1.Image(k), isSmooth, opt);
			im2feature(Image2, GPyramid2.Image(k), isSmooth, opt);

			if (k == GPyramid1.nlevels() - 1)
			{
				u.allocate(width, height);
				v.allocate(width, height);
				warpIm2.copyData(Image2);

				OneResolution_HS_L2(u, v, warpIm2, Image1, Image2, opt);
			}
			else
			{
				u.imresize(width, height);
				u.Multiplywith(1.0 / ratio);
				v.imresize(width, height);
				v.Multiplywith(1.0 / ratio);
				warpFL(warpIm2, Image1, Image2, u, v, opt.useCubicWarping);

			}

			ADMM_F_G(u, v, warpIm2, Image1, Image2, opt, Proximal_F_L2<T>, Proximal_G<T>);
		}
	}

	template<class T>
	void ZQ_OpticalFlow::Coarse2Fine_ADMM_DL1(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_GaussianPyramid<T> GPyramid1;
		ZQ_GaussianPyramid<T> GPyramid2;

		if (opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio =
			GPyramid1.ConstructPyramid(Im1, opt.ratioForPyramid, opt.minWidthForPyramid);
		GPyramid2.ConstructPyramid(Im2, opt.ratioForPyramid, opt.minWidthForPyramid);

		if (opt.displayRunningInfo)
			printf("done!\n");

		ZQ_DImage<T> Image1, Image2;

		for (int k = GPyramid1.nlevels() - 1; k >= 0; k--)
		{
			if (opt.displayRunningInfo)
				printf("Pyramid level %d \n", k);

			int width = GPyramid1.Image(k).width();
			int height = GPyramid1.Image(k).height();

			bool isSmooth = true;
			if (k == 0)
				isSmooth = false;

			im2feature(Image1, GPyramid1.Image(k), isSmooth, opt);
			im2feature(Image2, GPyramid2.Image(k), isSmooth, opt);

			if (k == GPyramid1.nlevels() - 1)
			{
				u.allocate(width, height);
				v.allocate(width, height);
				warpIm2.copyData(Image2);

				OneResolution_HS_DL1(u, v, warpIm2, Image1, Image2, opt);
			}
			else
			{
				u.imresize(width, height);
				u.Multiplywith(1.0 / ratio);
				v.imresize(width, height);
				v.Multiplywith(1.0 / ratio);
				warpFL(warpIm2, Image1, Image2, u, v, opt.useCubicWarping);

			}

			ADMM_F_G(u, v, warpIm2, Image1, Image2, opt, Proximal_F_DL1<T>, Proximal_G<T>);
		}
	}


	template<class T>
	void ZQ_OpticalFlow::Coarse2Fine_OneDir_Inc_L2(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();

		if (u.size() != image_num - 1)
		{
			u.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
				u.push_back(tmp);
		}

		if (v.size() != image_num - 1)
		{
			v.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
				v.push_back(tmp);
		}

		if (warpIm.size() != image_num - 1)
		{
			warpIm.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
			{
				warpIm.push_back(tmp);
			}
		}

		/*  Constructing Pyramids */
		std::vector<ZQ_GaussianPyramid<T>> GPyramids(image_num);

		if (opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio = 0;
		for (int i = 0; i < image_num; i++)
			ratio = GPyramids[i].ConstructPyramid(Im[i], opt.ratioForPyramid, opt.minWidthForPyramid);

		if (opt.displayRunningInfo)
			printf("done!\n");

		/*   Handle images from low resolution to high resolution     */
		std::vector<ZQ_DImage<T>> Images(image_num);

		for (int k = GPyramids[0].nlevels() - 1; k >= 0; k--)
		{
			if (opt.displayRunningInfo)
				printf("Pyramid level %d \n", k);

			int width = GPyramids[0].Image(k).width();
			int height = GPyramids[0].Image(k).height();

			bool isSmooth = true;
			if (k == 0)
				isSmooth = false;

			for (int i = 0; i < image_num; i++)
				im2feature(Images[i], GPyramids[i].Image(k), isSmooth, opt);

			if (k == GPyramids[0].nlevels() - 1)
			{
				for (int i = 0; i < image_num - 1; i++)
				{
					u[i].allocate(width, height);
					v[i].allocate(width, height);
					warpIm[i].copyData(Images[i + 1]);
					OneResolution_HS_L2(u[i], v[i], warpIm[i], Images[i], Images[i + 1], opt);
				}
			}
			else
			{
				for (int i = 0; i < image_num - 1; i++)
				{
					u[i].imresize(width, height);
					u[i].Multiplywith(1.0 / ratio);
					v[i].imresize(width, height);
					v[i].Multiplywith(1.0 / ratio);
					warpFL(warpIm[i], Images[i], Images[i + 1], u[i], v[i], opt.useCubicWarping);
				}
			}

			OneResolution_OneDir_Inc_L2(u, v, warpIm, Images, opt);
		}
	}


	template<class T>
	void ZQ_OpticalFlow::Coarse2Fine_OneDir_Inc_DL1(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();

		if (u.size() != image_num - 1)
		{
			u.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
				u.push_back(tmp);
		}

		if (v.size() != image_num - 1)
		{
			v.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
				v.push_back(tmp);
		}

		if (warpIm.size() != image_num - 1)
		{
			warpIm.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
			{
				warpIm.push_back(tmp);
			}
		}

		/*  Constructing Pyramids */
		std::vector<ZQ_GaussianPyramid<T>> GPyramids(image_num);

		if (opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio = 0;
		for (int i = 0; i < image_num; i++)
			ratio = GPyramids[i].ConstructPyramid(Im[i], opt.ratioForPyramid, opt.minWidthForPyramid);

		if (opt.displayRunningInfo)
			printf("done!\n");

		/*   Handle images from low resolution to high resolution     */
		std::vector<ZQ_DImage<T>> Images(image_num);

		for (int k = GPyramids[0].nlevels() - 1; k >= 0; k--)
		{
			if (opt.displayRunningInfo)
				printf("Pyramid level %d \n", k);

			int width = GPyramids[0].Image(k).width();
			int height = GPyramids[0].Image(k).height();

			bool isSmooth = true;
			if (k == 0)
				isSmooth = false;

			for (int i = 0; i < image_num; i++)
				im2feature(Images[i], GPyramids[i].Image(k), isSmooth, opt);

			if (k == GPyramids[0].nlevels() - 1)
			{
				for (int i = 0; i < image_num - 1; i++)
				{
					u[i].allocate(width, height);
					v[i].allocate(width, height);
					warpIm[i].copyData(Images[i + 1]);
					OneResolution_HS_DL1(u[i], v[i], warpIm[i], Images[i], Images[i + 1], opt);
				}
			}
			else
			{
				for (int i = 0; i < image_num - 1; i++)
				{
					u[i].imresize(width, height);
					u[i].Multiplywith(1.0 / ratio);
					v[i].imresize(width, height);
					v[i].Multiplywith(1.0 / ratio);
					warpFL(warpIm[i], Images[i], Images[i + 1], u[i], v[i], opt.useCubicWarping);
				}
			}

			OneResolution_OneDir_Inc_DL1(u, v, warpIm, Images, opt);
		}
	}

	template<class T>
	void ZQ_OpticalFlow::Coarse2Fine_OneDir_Dec_L2(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();

		if (u.size() != image_num - 1)
		{
			u.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
				u.push_back(tmp);
		}

		if (v.size() != image_num - 1)
		{
			v.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
				v.push_back(tmp);
		}

		if (warpIm.size() != image_num - 1)
		{
			warpIm.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
			{
				warpIm.push_back(tmp);
			}
		}

		/*  Constructing Pyramids */
		std::vector<ZQ_GaussianPyramid<T>> GPyramids(image_num);

		if (opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio = 0;
		for (int i = 0; i < image_num; i++)
			ratio = GPyramids[i].ConstructPyramid(Im[i], opt.ratioForPyramid, opt.minWidthForPyramid);

		if (opt.displayRunningInfo)
			printf("done!\n");

		/*   Handle images from low resolution to high resolution     */
		std::vector<ZQ_DImage<T>> Images(image_num);

		for (int k = GPyramids[0].nlevels() - 1; k >= 0; k--)
		{
			if (opt.displayRunningInfo)
				printf("Pyramid level %d \n", k);

			int width = GPyramids[0].Image(k).width();
			int height = GPyramids[0].Image(k).height();

			bool isSmooth = true;
			if (k == 0)
				isSmooth = false;

			for (int i = 0; i < image_num; i++)
				im2feature(Images[i], GPyramids[i].Image(k), isSmooth, opt);

			if (k == GPyramids[0].nlevels() - 1)
			{
				for (int i = 0; i < image_num - 1; i++)
				{
					u[i].allocate(width, height);
					v[i].allocate(width, height);
					warpIm[i].copyData(Images[i + 1]);
					OneResolution_HS_L2(u[i], v[i], warpIm[i], Images[i], Images[i + 1], opt);
				}
			}
			else
			{
				for (int i = 0; i < image_num - 1; i++)
				{
					u[i].imresize(width, height);
					u[i].Multiplywith(1.0 / ratio);
					v[i].imresize(width, height);
					v[i].Multiplywith(1.0 / ratio);
					warpFL(warpIm[i], Images[i], Images[i + 1], u[i], v[i], opt.useCubicWarping);
				}
			}

			OneResolution_OneDir_Dec_L2(u, v, warpIm, Images, opt);
		}
	}


	template<class T>
	void ZQ_OpticalFlow::Coarse2Fine_OneDir_Dec_DL1(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();

		if (u.size() != image_num - 1)
		{
			u.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
				u.push_back(tmp);
		}

		if (v.size() != image_num - 1)
		{
			v.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
				v.push_back(tmp);
		}

		if (warpIm.size() != image_num - 1)
		{
			warpIm.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
			{
				warpIm.push_back(tmp);
			}
		}

		/*  Constructing Pyramids */
		std::vector<ZQ_GaussianPyramid<T>> GPyramids(image_num);

		if (opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio = 0;
		for (int i = 0; i < image_num; i++)
			ratio = GPyramids[i].ConstructPyramid(Im[i], opt.ratioForPyramid, opt.minWidthForPyramid);

		if (opt.displayRunningInfo)
			printf("done!\n");

		/*   Handle images from low resolution to high resolution     */
		std::vector<ZQ_DImage<T>> Images(image_num);

		for (int k = GPyramids[0].nlevels() - 1; k >= 0; k--)
		{
			if (opt.displayRunningInfo)
				printf("Pyramid level %d \n", k);

			int width = GPyramids[0].Image(k).width();
			int height = GPyramids[0].Image(k).height();

			bool isSmooth = true;
			if (k == 0)
				isSmooth = false;

			for (int i = 0; i < image_num; i++)
				im2feature(Images[i], GPyramids[i].Image(k), isSmooth, opt);

			if (k == GPyramids[0].nlevels() - 1)
			{
				for (int i = 0; i < image_num - 1; i++)
				{
					u[i].allocate(width, height);
					v[i].allocate(width, height);
					warpIm[i].copyData(Images[i + 1]);
					OneResolution_HS_DL1(u[i], v[i], warpIm[i], Images[i], Images[i + 1], opt);
				}
			}
			else
			{
				for (int i = 0; i < image_num - 1; i++)
				{
					u[i].imresize(width, height);
					u[i].Multiplywith(1.0 / ratio);
					v[i].imresize(width, height);
					v[i].Multiplywith(1.0 / ratio);
					warpFL(warpIm[i], Images[i], Images[i + 1], u[i], v[i], opt.useCubicWarping);
				}
			}

			OneResolution_OneDir_Dec_DL1(u, v, warpIm, Images, opt);
		}
	}

	template<class T>
	void ZQ_OpticalFlow::Coarse2Fine_TwoDir_Inc_L2(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();

		if (u.size() != image_num - 1)
		{
			u.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
				u.push_back(tmp);
		}

		if (v.size() != image_num - 1)
		{
			v.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
				v.push_back(tmp);
		}

		if (warpIm.size() != image_num - 1)
		{
			warpIm.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
			{
				warpIm.push_back(tmp);
			}
		}

		/*  Constructing Pyramids */
		std::vector<ZQ_GaussianPyramid<T>> GPyramids(image_num);

		if (opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio = 0;
		for (int i = 0; i < image_num; i++)
			ratio = GPyramids[i].ConstructPyramid(Im[i], opt.ratioForPyramid, opt.minWidthForPyramid);

		if (opt.displayRunningInfo)
			printf("done!\n");

		/*   Handle images from low resolution to high resolution     */
		std::vector<ZQ_DImage<T>> Images(image_num);

		for (int k = GPyramids[0].nlevels() - 1; k >= 0; k--)
		{
			if (opt.displayRunningInfo)
				printf("Pyramid level %d \n", k);

			int width = GPyramids[0].Image(k).width();
			int height = GPyramids[0].Image(k).height();

			bool isSmooth = true;
			if (k == 0)
				isSmooth = false;

			for (int i = 0; i < image_num; i++)
				im2feature(Images[i], GPyramids[i].Image(k), isSmooth, opt);

			if (k == GPyramids[0].nlevels() - 1)
			{
				for (int i = 0; i < image_num - 1; i++)
				{
					u[i].allocate(width, height);
					v[i].allocate(width, height);
					warpIm[i].copyData(Images[i + 1]);
					OneResolution_HS_L2(u[i], v[i], warpIm[i], Images[i], Images[i + 1], opt);
				}
			}
			else
			{
				for (int i = 0; i < image_num - 1; i++)
				{
					u[i].imresize(width, height);
					u[i].Multiplywith(1.0 / ratio);
					v[i].imresize(width, height);
					v[i].Multiplywith(1.0 / ratio);
					warpFL(warpIm[i], Images[i], Images[i + 1], u[i], v[i], opt.useCubicWarping);
				}
			}

			OneResolution_TwoDir_Inc_L2(u, v, warpIm, Images, opt);
		}
	}

	template<class T>
	void ZQ_OpticalFlow::Coarse2Fine_TwoDir_Inc_DL1(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();

		if (u.size() != image_num - 1)
		{
			u.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
				u.push_back(tmp);
		}

		if (v.size() != image_num - 1)
		{
			v.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
				v.push_back(tmp);
		}

		if (warpIm.size() != image_num - 1)
		{
			warpIm.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
			{
				warpIm.push_back(tmp);
			}
		}

		/*  Constructing Pyramids */
		std::vector<ZQ_GaussianPyramid<T>> GPyramids(image_num);

		if (opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio = 0;
		for (int i = 0; i < image_num; i++)
			ratio = GPyramids[i].ConstructPyramid(Im[i], opt.ratioForPyramid, opt.minWidthForPyramid);

		if (opt.displayRunningInfo)
			printf("done!\n");

		/*   Handle images from low resolution to high resolution     */
		std::vector<ZQ_DImage<T>> Images(image_num);

		for (int k = GPyramids[0].nlevels() - 1; k >= 0; k--)
		{
			if (opt.displayRunningInfo)
				printf("Pyramid level %d \n", k);

			int width = GPyramids[0].Image(k).width();
			int height = GPyramids[0].Image(k).height();

			bool isSmooth = true;
			if (k == 0)
				isSmooth = false;

			for (int i = 0; i < image_num; i++)
				im2feature(Images[i], GPyramids[i].Image(k), isSmooth, opt);

			if (k == GPyramids[0].nlevels() - 1)
			{
				for (int i = 0; i < image_num - 1; i++)
				{
					u[i].allocate(width, height);
					v[i].allocate(width, height);
					warpIm[i].copyData(Images[i + 1]);
					OneResolution_HS_DL1(u[i], v[i], warpIm[i], Images[i], Images[i + 1], opt);
				}
			}
			else
			{
				for (int i = 0; i < image_num - 1; i++)
				{
					u[i].imresize(width, height);
					u[i].Multiplywith(1.0 / ratio);
					v[i].imresize(width, height);
					v[i].Multiplywith(1.0 / ratio);
					warpFL(warpIm[i], Images[i], Images[i + 1], u[i], v[i], opt.useCubicWarping);
				}
			}

			OneResolution_TwoDir_Inc_DL1(u, v, warpIm, Images, opt);
		}
	}

	template<class T>
	void ZQ_OpticalFlow::Coarse2Fine_TwoDir_Dec_L2(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();

		if (u.size() != image_num - 1)
		{
			u.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
				u.push_back(tmp);
		}

		if (v.size() != image_num - 1)
		{
			v.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
				v.push_back(tmp);
		}

		if (warpIm.size() != image_num - 1)
		{
			warpIm.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
			{
				warpIm.push_back(tmp);
			}
		}

		/*  Constructing Pyramids */
		std::vector<ZQ_GaussianPyramid<T>> GPyramids(image_num);

		if (opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio = 0;
		for (int i = 0; i < image_num; i++)
			ratio = GPyramids[i].ConstructPyramid(Im[i], opt.ratioForPyramid, opt.minWidthForPyramid);

		if (opt.displayRunningInfo)
			printf("done!\n");

		/*   Handle images from low resolution to high resolution     */
		std::vector<ZQ_DImage<T>> Images(image_num);

		for (int k = GPyramids[0].nlevels() - 1; k >= 0; k--)
		{
			if (opt.displayRunningInfo)
				printf("Pyramid level %d \n", k);

			int width = GPyramids[0].Image(k).width();
			int height = GPyramids[0].Image(k).height();

			bool isSmooth = true;
			if (k == 0)
				isSmooth = false;

			for (int i = 0; i < image_num; i++)
				im2feature(Images[i], GPyramids[i].Image(k), isSmooth, opt);

			if (k == GPyramids[0].nlevels() - 1)
			{
				for (int i = 0; i < image_num - 1; i++)
				{
					u[i].allocate(width, height);
					v[i].allocate(width, height);
					warpIm[i].copyData(Images[i + 1]);
					OneResolution_HS_L2(u[i], v[i], warpIm[i], Images[i], Images[i + 1], opt);
				}
			}
			else
			{
				for (int i = 0; i < image_num - 1; i++)
				{
					u[i].imresize(width, height);
					u[i].Multiplywith(1.0 / ratio);
					v[i].imresize(width, height);
					v[i].Multiplywith(1.0 / ratio);
					warpFL(warpIm[i], Images[i], Images[i + 1], u[i], v[i], opt.useCubicWarping);
				}
			}

			OneResolution_TwoDir_Dec_L2(u, v, warpIm, Images, opt);
		}
	}


	template<class T>
	void ZQ_OpticalFlow::Coarse2Fine_TwoDir_Dec_DL1(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();

		if (u.size() != image_num - 1)
		{
			u.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
				u.push_back(tmp);
		}

		if (v.size() != image_num - 1)
		{
			v.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
				v.push_back(tmp);
		}

		if (warpIm.size() != image_num - 1)
		{
			warpIm.clear();
			ZQ_DImage<T> tmp;
			for (int i = 0; i < image_num - 1; i++)
			{
				warpIm.push_back(tmp);
			}
		}

		/*  Constructing Pyramids */
		std::vector<ZQ_GaussianPyramid<T>> GPyramids(image_num);

		if (opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio = 0;
		for (int i = 0; i < image_num; i++)
			ratio = GPyramids[i].ConstructPyramid(Im[i], opt.ratioForPyramid, opt.minWidthForPyramid);

		if (opt.displayRunningInfo)
			printf("done!\n");

		/*   Handle images from low resolution to high resolution     */
		std::vector<ZQ_DImage<T>> Images(image_num);

		for (int k = GPyramids[0].nlevels() - 1; k >= 0; k--)
		{
			if (opt.displayRunningInfo)
				printf("Pyramid level %d \n", k);

			int width = GPyramids[0].Image(k).width();
			int height = GPyramids[0].Image(k).height();

			bool isSmooth = true;
			if (k == 0)
				isSmooth = false;

			for (int i = 0; i < image_num; i++)
				im2feature(Images[i], GPyramids[i].Image(k), isSmooth, opt);

			if (k == GPyramids[0].nlevels() - 1)
			{
				for (int i = 0; i < image_num - 1; i++)
				{
					u[i].allocate(width, height);
					v[i].allocate(width, height);
					warpIm[i].copyData(Images[i + 1]);
					OneResolution_HS_DL1(u[i], v[i], warpIm[i], Images[i], Images[i + 1], opt);
				}
			}
			else
			{
				for (int i = 0; i < image_num - 1; i++)
				{
					u[i].imresize(width, height);
					u[i].Multiplywith(1.0 / ratio);
					v[i].imresize(width, height);
					v[i].Multiplywith(1.0 / ratio);
					warpFL(warpIm[i], Images[i], Images[i + 1], u[i], v[i], opt.useCubicWarping);
				}
			}

			OneResolution_TwoDir_Dec_DL1(u, v, warpIm, Images, opt);
		}
	}

	/****************************************************************************************************************************************/

	template<class T>
	void ZQ_OpticalFlow::OneResolution_HS_L2(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_DImage<T> imdx, imdy, imdt;

		int imWidth = Im1.width();
		int imHeight = Im1.height();
		int nChannels = Im1.nchannels();
		int nPixels = imWidth*imHeight;

		ZQ_DImage<T> du(imWidth, imHeight), dv(imWidth, imHeight); //for du, dv


		warpFL(warpIm2, Im1, Im2, u, v, opt.useCubicWarping);

		/************       Outer Loop Begin     *************/
		//refresh {u,v} in each loop
		for (int count = 0; count < opt.nOuterFixedPointIterations; count++)
		{
			// outer loop : {imdx, imdy, imdt} accoring to warpIm2, warpIm2 accoring to {u,v} 

			getDxs(imdx, imdy, imdt, Im1, warpIm2, true);

			du.reset();
			dv.reset();

			//compute imdtdx, imdxx, imdyy, imdtdx, imdtdy
			ZQ_DImage<T> imdxy, imdxx, imdyy, imdtdx, imdtdy;
			imdxx.Multiply(imdx, imdx);
			imdxy.Multiply(imdx, imdy);
			imdyy.Multiply(imdy, imdy);
			imdtdx.Multiply(imdx, imdt);
			imdtdy.Multiply(imdy, imdt);

			if (nChannels > 1)
			{
				imdxx.collapse();
				imdxy.collapse();
				imdyy.collapse();
				imdtdx.collapse();
				imdtdy.collapse();
			}

			ZQ_DImage<T> laplace_u(imWidth, imHeight);
			ZQ_DImage<T> laplace_v(imWidth, imHeight);

			Laplacian(laplace_u, u);
			Laplacian(laplace_v, v);

			T*& laplace_uData = laplace_u.data();
			T*& laplace_vData = laplace_v.data();


			// set omega
			double omega = opt.omegaForSOR;
			double alpha2 = opt.alpha*opt.alpha;
			double beta2 = opt.beta*opt.beta;

			T*& duData = du.data();
			T*& dvData = dv.data();
			T*& uData = u.data();
			T*& vData = v.data();
			T*& imdtdxData = imdtdx.data();
			T*& imdtdyData = imdtdy.data();
			T*& imdxyData = imdxy.data();
			T*& imdxxData = imdxx.data();
			T*& imdyyData = imdyy.data();



			/***   SOR Begin solve du,dv***/

			for (int k = 0; k < opt.nSORIterations; k++)
			{
				for (int i = 0; i < imHeight; i++)
				{
					for (int j = 0; j<imWidth; j++)
					{
						int offset = i * imWidth + j;
						double sigma1 = 0, sigma2 = 0, coeff = 0;
						double _weight;

						if (j>0)
						{
							_weight = 1;
							sigma1 += _weight*duData[offset - 1];
							sigma2 += _weight*dvData[offset - 1];
							coeff += _weight;

						}
						if (j<imWidth - 1)
						{
							_weight = 1;
							sigma1 += _weight*duData[offset + 1];
							sigma2 += _weight*dvData[offset + 1];
							coeff += _weight;
						}
						if (i>0)
						{
							_weight = 1;
							sigma1 += _weight*duData[offset - imWidth];
							sigma2 += _weight*dvData[offset - imWidth];
							coeff += _weight;
						}
						if (i < imHeight - 1)
						{
							_weight = 1;
							sigma1 += _weight*duData[offset + imWidth];
							sigma2 += _weight*dvData[offset + imWidth];
							coeff += _weight;
						}
						sigma1 *= alpha2;
						sigma2 *= alpha2;
						coeff *= alpha2;
						// compute u
						sigma1 += alpha2*laplace_uData[offset] - imdtdxData[offset] - imdxyData[offset] * dvData[offset] - beta2*uData[offset];
						double coeff1 = coeff + imdxxData[offset] + beta2;
						duData[offset] = (1 - omega)*duData[offset] + omega / coeff1*sigma1;
						// compute v
						sigma2 += alpha2*laplace_vData[offset] - imdtdyData[offset] - imdxyData[offset] * duData[offset] - beta2*vData[offset];
						double coeff2 = coeff + imdyyData[offset] + beta2;
						dvData[offset] = (1 - omega)*dvData[offset] + omega / coeff2*sigma2;
					}
				}
			}

			/***   SOR end solve du,dv***/
			if (opt.hasMaxUpdateLimit)
			{
				du.imclamp(-1.0, 1.0);
				dv.imclamp(-1.0, 1.0);
			}

			u.Addwith(du);
			v.Addwith(dv);


			warpFL(warpIm2, Im1, Im2, u, v, opt.useCubicWarping);

		}
		/************       Outer Loop End     *************/
	}

	template<class T>
	void ZQ_OpticalFlow::OneResolution_HS_L2(ZQ_DImage<T>& u, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_DImage<T> imdx, imdy, imdt;

		int imWidth = Im1.width();
		int imHeight = Im1.height();
		int nChannels = Im1.nchannels();
		int nPixels = imWidth*imHeight;

		ZQ_DImage<T> v(imWidth, imHeight);
		ZQ_DImage<T> du(imWidth, imHeight); //for du


		warpFL(warpIm2, Im1, Im2, u, v, opt.useCubicWarping);

		/************       Outer Loop Begin     *************/
		//refresh {u} in each loop
		for (int count = 0; count < opt.nOuterFixedPointIterations; count++)
		{
			// outer loop : {imdx, imdy, imdt} accoring to warpIm2, warpIm2 accoring to {u,v} 
			getDxs(imdx, imdy, imdt, Im1, warpIm2, true);

			du.reset();

			//compute imdtdx, imdxx
			ZQ_DImage<T> imdxx, imdtdx;
			imdxx.Multiply(imdx, imdx);
			imdtdx.Multiply(imdx, imdt);

			if (nChannels > 1)
			{
				imdxx.collapse();
				imdtdx.collapse();
			}

			ZQ_DImage<T> laplace_u(imWidth, imHeight);

			Laplacian(laplace_u, u);

			T*& laplace_uData = laplace_u.data();

			// set omega
			double omega = opt.omegaForSOR;
			double alpha2 = opt.alpha*opt.alpha;
			double beta2 = opt.beta*opt.beta;

			T*& duData = du.data();
			T*& uData = u.data();
			T*& imdtdxData = imdtdx.data();
			T*& imdxxData = imdxx.data();


			/***   SOR Begin solve du,dv***/

			for (int k = 0; k < opt.nSORIterations; k++)
			{
				for (int i = 0; i < imHeight; i++)
				{
					for (int j = 0; j<imWidth; j++)
					{
						int offset = i * imWidth + j;
						double sigma1 = 0, coeff = 0;
						double _weight;

						if (j>0)
						{
							_weight = 1;
							sigma1 += _weight*duData[offset - 1];
							coeff += _weight;
						}
						if (j<imWidth - 1)
						{
							_weight = 1;
							sigma1 += _weight*duData[offset + 1];
							coeff += _weight;
						}
						if (i>0)
						{
							_weight = 1;
							sigma1 += _weight*duData[offset - imWidth];
							coeff += _weight;
						}
						if (i < imHeight - 1)
						{
							_weight = 1;
							sigma1 += _weight*duData[offset + imWidth];
							coeff += _weight;
						}
						sigma1 *= alpha2;
						coeff *= alpha2;
						// compute u
						sigma1 += alpha2*laplace_uData[offset] - imdtdxData[offset] - beta2*uData[offset];
						double coeff1 = coeff + imdxxData[offset] + beta2;
						duData[offset] = (1 - omega)*duData[offset] + omega / coeff1*sigma1;
					}
				}
			}

			/***   SOR end solve du,dv***/

			if (opt.hasMaxUpdateLimit)
			{
				du.imclamp(-1.0, 1.0);
			}
			u.Addwith(du);

			warpFL(warpIm2, Im1, Im2, u, v, opt.useCubicWarping);
		}
		/************       Outer Loop End     *************/
	}

	template<class T>
	void ZQ_OpticalFlow::OneResolution_HS_DL1(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_DImage<T> imdx, imdy, imdt;

		int imWidth = Im1.width();
		int imHeight = Im1.height();
		int nChannels = Im1.nchannels();
		int nPixels = imWidth*imHeight;

		ZQ_DImage<T> du(imWidth, imHeight), dv(imWidth, imHeight);

		ZQ_DImage<T> Psi_1st(imWidth, imHeight, nChannels);

		ZQ_DImage<T> imdxdx, imdxdy, imdydy, imdtdx, imdtdy;

		ZQ_DImage<T> laplace_u, laplace_v;

		double varepsilon_phi = pow(0.001, 2);
		double varepsilon_psi = pow(0.001, 2);

		warpFL(warpIm2, Im1, Im2, u, v, opt.useCubicWarping);

		for (int out_it = 0; out_it < opt.nOuterFixedPointIterations; out_it++)
		{
			getDxs(imdx, imdy, imdt, Im1, warpIm2, true);

			du.reset();
			dv.reset();

			for (int in_it = 0; in_it < opt.nInnerFixedPointIterations; in_it++)
			{
				Psi_1st.reset();

				T*& psiData = Psi_1st.data();
				T*& imdxData = imdx.data();
				T*& imdyData = imdy.data();
				T*& imdtData = imdt.data();
				T*& duData = du.data();
				T*& dvData = dv.data();
				T*& uData = u.data();
				T*& vData = v.data();

				for (int i = 0; i < nPixels; i++)
				{
					for (int c = 0; c < nChannels; c++)
					{
						int offset = i*nChannels + c;
						double temp = imdtData[offset] + imdxData[offset] * duData[i] + imdyData[offset] * dvData[i];

						temp *= temp;
						psiData[offset] = 1 / (2 * sqrt(temp + varepsilon_psi));
					}
				}

				imdxdx.Multiply(Psi_1st, imdx, imdx);
				imdxdy.Multiply(Psi_1st, imdx, imdy);
				imdydy.Multiply(Psi_1st, imdy, imdy);
				imdtdx.Multiply(Psi_1st, imdx, imdt);
				imdtdy.Multiply(Psi_1st, imdy, imdt);

				if (nChannels > 1)
				{
					imdxdx.collapse();
					imdxdy.collapse();
					imdydy.collapse();
					imdtdx.collapse();
					imdtdy.collapse();
				}

				T*& imdxdxData = imdxdx.data();
				T*& imdxdyData = imdxdy.data();
				T*& imdydyData = imdydy.data();
				T*& imdtdxData = imdtdx.data();
				T*& imdtdyData = imdtdy.data();


				Laplacian(laplace_u, u);
				Laplacian(laplace_v, v);

				T*& laplace_uPtr = laplace_u.data();
				T*& laplace_vPtr = laplace_v.data();

				double omega = opt.omegaForSOR;
				double alpha2 = opt.alpha*opt.alpha;
				double beta2 = opt.beta*opt.beta;

				/*Begin SOR*/

				for (int sor_it = 0; sor_it < opt.nSORIterations; sor_it++)
				{
					for (int i = 0; i < imHeight; i++)
					{
						for (int j = 0; j < imWidth; j++)
						{
							int offset = i*imWidth + j;
							double sigma1 = 0, sigma2 = 0, coeff = 0;
							double _weight;


							if (j > 0)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset - 1];
								sigma2 += _weight*dvData[offset - 1];
								coeff += _weight;
							}
							if (j < imWidth - 1)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset + 1];
								sigma2 += _weight*dvData[offset + 1];
								coeff += _weight;
							}
							if (i > 0)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset - imWidth];
								sigma2 += _weight*dvData[offset - imWidth];
								coeff += _weight;
							}
							if (i < imHeight - 1)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset + imWidth];
								sigma2 += _weight*dvData[offset + imWidth];
								coeff += _weight;
							}
							sigma1 *= alpha2;
							sigma2 *= alpha2;
							coeff *= alpha2;

							sigma1 += alpha2*laplace_uPtr[offset] - imdtdxData[offset] - imdxdyData[offset] * dvData[offset] - beta2*uData[offset];
							double coeff1 = coeff + imdxdxData[offset] + beta2;
							duData[offset] = (1 - omega)*duData[offset] + omega / coeff1*sigma1;

							sigma2 += alpha2*laplace_vPtr[offset] - imdtdyData[offset] - imdxdyData[offset] * duData[offset] - beta2*vData[offset];
							double coeff2 = coeff + imdydyData[offset] + beta2;
							dvData[offset] = (1 - omega)*dvData[offset] + omega / coeff2*sigma2;
						}
					}
				}
				/*End SOR*/
			}

			if (opt.hasMaxUpdateLimit)
			{
				du.imclamp(-1.0, 1.0);
				dv.imclamp(-1.0, 1.0);
			}
			u.Addwith(du);
			v.Addwith(dv);

			warpFL(warpIm2, Im1, Im2, u, v, opt.useCubicWarping);
		}
	}


	template<class T>
	void ZQ_OpticalFlow::OneResolution_HS_L1(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_DImage<T> imdx, imdy, imdt;

		int imWidth = Im1.width();
		int imHeight = Im1.height();
		int nChannels = Im1.nchannels();
		int nPixels = imWidth*imHeight;

		ZQ_DImage<T> du(imWidth, imHeight), dv(imWidth, imHeight);
		ZQ_DImage<T> uu(imWidth, imHeight), vv(imWidth, imHeight);
		ZQ_DImage<T> ux(imWidth, imHeight), uy(imWidth, imHeight);
		ZQ_DImage<T> vx(imWidth, imHeight), vy(imWidth, imHeight);
		ZQ_DImage<T> Phi_gradu_1st(imWidth, imHeight);
		ZQ_DImage<T> Phi_u_1st(imWidth, imHeight);
		ZQ_DImage<T> Phi_v_1st(imWidth, imHeight);
		ZQ_DImage<T> Phi_data_1st(imWidth, imHeight, nChannels);

		ZQ_DImage<T> imdxdx, imdxdy, imdydy, imdtdx, imdtdy;

		ZQ_DImage<T> laplace_u, laplace_v;

		double varepsilon_phi = pow(0.001, 2);
		double varepsilon_psi = pow(0.001, 2);

		warpFL(warpIm2, Im1, Im2, u, v, opt.useCubicWarping);

		for (int out_it = 0; out_it < opt.nOuterFixedPointIterations; out_it++)
		{
			getDxs(imdx, imdy, imdt, Im1, warpIm2, true);

			du.reset();
			dv.reset();

			for (int in_it = 0; in_it < opt.nInnerFixedPointIterations; in_it++)
			{

				uu.Add(u, du);
				vv.Add(v, dv);

				uu.dx(ux, false);
				uu.dy(uy, false);
				vv.dx(vx, false);
				vv.dy(vy, false);

				Phi_gradu_1st.reset();

				T*& phi_graduData = Phi_gradu_1st.data();
				T*& uxData = ux.data();
				T*& uyData = uy.data();
				T*& vxData = vx.data();
				T*& vyData = vy.data();

				for (int i = 0; i < nPixels; i++)
				{
					double temp = uxData[i] * uxData[i] + uyData[i] * uyData[i] + vxData[i] * vxData[i] + vyData[i] * vyData[i];
					phi_graduData[i] = 0.5 / sqrt(temp + varepsilon_phi);
				}
				Phi_u_1st.reset();
				Phi_v_1st.reset();

				T*& phi_uData = Phi_u_1st.data();
				T*& phi_vData = Phi_v_1st.data();
				T*& uuData = uu.data();
				T*& vvData = vv.data();

				for (int i = 0; i < nPixels; i++)
				{
					phi_uData[i] = 0.5 / sqrt(uuData[i] * uuData[i] + varepsilon_phi);
					phi_vData[i] = 0.5 / sqrt(vvData[i] * vvData[i] + varepsilon_phi);
				}


				Phi_data_1st.reset();

				T*& phi_dataData = Phi_data_1st.data();
				T*& imdxData = imdx.data();
				T*& imdyData = imdy.data();
				T*& imdtData = imdt.data();
				T*& duData = du.data();
				T*& dvData = dv.data();
				T*& uData = u.data();
				T*& vData = v.data();

				for (int i = 0; i < nPixels; i++)
				{
					for (int c = 0; c < nChannels; c++)
					{
						int offset = i*nChannels + c;
						double temp = imdtData[offset] + imdxData[offset] * duData[i] + imdyData[offset] * dvData[i];

						temp *= temp;
						phi_dataData[offset] = 0.5 / sqrt(temp + varepsilon_psi);
					}
				}

				imdxdx.Multiply(Phi_data_1st, imdx, imdx);
				imdxdy.Multiply(Phi_data_1st, imdx, imdy);
				imdydy.Multiply(Phi_data_1st, imdy, imdy);
				imdtdx.Multiply(Phi_data_1st, imdx, imdt);
				imdtdy.Multiply(Phi_data_1st, imdy, imdt);

				if (nChannels > 1)
				{
					imdxdx.collapse();
					imdxdy.collapse();
					imdydy.collapse();
					imdtdx.collapse();
					imdtdy.collapse();
				}

				T*& imdxdxData = imdxdx.data();
				T*& imdxdyData = imdxdy.data();
				T*& imdydyData = imdydy.data();
				T*& imdtdxData = imdtdx.data();
				T*& imdtdyData = imdtdy.data();


				Laplacian(laplace_u, u, Phi_gradu_1st);
				Laplacian(laplace_v, v, Phi_gradu_1st);

				T*& laplace_uPtr = laplace_u.data();
				T*& laplace_vPtr = laplace_v.data();

				double omega = opt.omegaForSOR;
				double alpha = opt.alpha;
				double beta = opt.beta;

				/*Begin SOR*/

				for (int sor_it = 0; sor_it < opt.nSORIterations; sor_it++)
				{
					for (int i = 0; i < imHeight; i++)
					{
						for (int j = 0; j < imWidth; j++)
						{
							int offset = i*imWidth + j;
							double sigma1 = 0, sigma2 = 0, coeff = 0;
							double _weight;


							if (j > 0)
							{
								_weight = phi_graduData[offset - 1];
								sigma1 += _weight*duData[offset - 1];
								sigma2 += _weight*dvData[offset - 1];
								coeff += _weight;
							}
							if (j < imWidth - 1)
							{
								_weight = phi_graduData[offset];
								sigma1 += _weight*duData[offset + 1];
								sigma2 += _weight*dvData[offset + 1];
								coeff += _weight;
							}
							if (i > 0)
							{
								_weight = phi_graduData[offset - imWidth];
								sigma1 += _weight*duData[offset - imWidth];
								sigma2 += _weight*dvData[offset - imWidth];
								coeff += _weight;
							}
							if (i < imHeight - 1)
							{
								_weight = phi_graduData[offset];
								sigma1 += _weight*duData[offset + imWidth];
								sigma2 += _weight*dvData[offset + imWidth];
								coeff += _weight;
							}
							sigma1 *= alpha;
							sigma2 *= alpha;
							coeff *= alpha;

							sigma1 += alpha*laplace_uPtr[offset] - imdtdxData[offset] - imdxdyData[offset] * dvData[offset] - beta*phi_uData[offset] * uData[offset];
							double coeff1 = coeff + imdxdxData[offset] + beta*phi_uData[offset];
							duData[offset] = (1 - omega)*duData[offset] + omega / coeff1*sigma1;

							sigma2 += alpha*laplace_vPtr[offset] - imdtdyData[offset] - imdxdyData[offset] * duData[offset] - beta*phi_vData[offset] * vData[offset];
							double coeff2 = coeff + imdydyData[offset] + beta*phi_vData[offset];
							dvData[offset] = (1 - omega)*dvData[offset] + omega / coeff2*sigma2;
						}
					}
				}
				/*End SOR*/
			}

			if (opt.hasMaxUpdateLimit)
			{
				du.imclamp(-1.0, 1.0);
				dv.imclamp(-1.0, 1.0);
			}
			u.Addwith(du);
			v.Addwith(dv);

			/*IplImage* flow_img = ZQ_ImageIO::SaveFlowToColorImage(u, v, 0, 0, 16, 0);
			cvSaveImage("tmp_flow.png", flow_img);
			cvReleaseImage(&flow_img);*/
			//if (imWidth > 160)
			{
				//int weightedMedFiltIter = 1;
				int weightedMedFiltIter = opt.weightedMedFiltIter_for_occ_detect;
				for (int wmf_it = 0; wmf_it < weightedMedFiltIter; wmf_it++)
				{
					//double sigma_div = 0.3;
					//double sigma_im = 0.08;
					double sigma_div = opt.sigma_div_for_occ_detect;
					double sigma_im = opt.sigma_im_for_occ_detect;
					ZQ_DImage<T> occ;
					ZQ_DImage<T> tmp_u(u);
					ZQ_DImage<T> tmp_v(v);
					detectOcclusion(occ, tmp_u, tmp_v, Im1, Im2, sigma_div, sigma_im);
					//ZQ_ImageIO::saveImage(occ, "occ.png");
					//double sigma_im_for_denoise = 0.02;
					//int weightedMedFiltSize = 5;
					//int medFiltSize = 3;
					double sigma_im_for_denoise = opt.sigma_im_for_denoise;
					int weightedMedFiltSize = opt.weightedMedFiltSize_for_denoise;
					int medFiltSize = opt.medFiltSize_for_denoise;
					/*FILE* out = fopen("u.txt", "w");
					for (int h = 0; h < u.height(); h++)
					{
					for (int w = 0; w < u.width(); w++)
					{
					fprintf(out, "%f ", u.data()[h*u.width() + w]);
					}
					fprintf(out, "\n");
					}
					fclose(out);
					out = fopen("v.txt", "w");
					for (int h = 0; h < v.height(); h++)
					{
					for (int w = 0; w < v.width(); w++)
					{
					fprintf(out, "%f ", v.data()[h*v.width() + w]);
					}
					fprintf(out, "\n");
					}
					fclose(out);
					out = fopen("Im1.txt", "w");
					for (int h = 0; h < Im1.height(); h++)
					{
					for (int w = 0; w < Im1.width(); w++)
					{
					fprintf(out, "%f ", Im1.data()[h*Im1.width() + w]);
					}
					fprintf(out, "\n");
					}
					fclose(out);
					out = fopen("occ.txt", "w");
					for (int h = 0; h < occ.height(); h++)
					{
					for (int w = 0; w < occ.width(); w++)
					{
					fprintf(out, "%f ", occ.data()[h*occ.width() + w]);
					}
					fprintf(out, "\n");
					}
					fclose(out);*/
					denoiseColorWeightedMedianFilter(u, v, tmp_u, tmp_v, Im1, occ, weightedMedFiltSize, medFiltSize, sigma_im_for_denoise);
				}
				/*IplImage* flow_img = ZQ_ImageIO::SaveFlowToColorImage(u, v, 0, 0, 16, 0);
				cvSaveImage("tmp_flow_denoise.png", flow_img);
				cvReleaseImage(&flow_img);*/
			}

			warpFL(warpIm2, Im1, Im2, u, v, opt.useCubicWarping);
		}
	}

	template<class T>
	void ZQ_OpticalFlow::OneResolution_HS_L1(ZQ_DImage<T>& u, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_DImage<T> imdx, imdy, imdt;

		int imWidth = Im1.width();
		int imHeight = Im1.height();
		int nChannels = Im1.nchannels();
		int nPixels = imWidth*imHeight;

		ZQ_DImage<T> v(imWidth, imHeight);
		ZQ_DImage<T> du(imWidth, imHeight);
		ZQ_DImage<T> uu(imWidth, imHeight), vv(imWidth, imHeight);
		ZQ_DImage<T> ux(imWidth, imHeight), uy(imWidth, imHeight);
		ZQ_DImage<T> Phi_gradu_1st(imWidth, imHeight);
		ZQ_DImage<T> Phi_u_1st(imWidth, imHeight);
		ZQ_DImage<T> Phi_data_1st(imWidth, imHeight, nChannels);

		ZQ_DImage<T> imdxdx, imdtdx;

		ZQ_DImage<T> laplace_u;

		double varepsilon_phi = pow(0.001, 2);
		double varepsilon_psi = pow(0.001, 2);

		warpFL(warpIm2, Im1, Im2, u, v, opt.useCubicWarping);

		for (int out_it = 0; out_it < opt.nOuterFixedPointIterations; out_it++)
		{
			getDxs(imdx, imdy, imdt, Im1, warpIm2, true);

			du.reset();

			for (int in_it = 0; in_it < opt.nInnerFixedPointIterations; in_it++)
			{
				uu.Add(u, du);
				uu.dx(ux, false);
				uu.dy(uy, false);

				Phi_gradu_1st.reset();

				T*& phi_graduData = Phi_gradu_1st.data();
				T*& uxData = ux.data();
				T*& uyData = uy.data();

#ifdef ZQLIB_USE_OPENMP
				if ( && nPixels > ZQ_DImage<T>::THRESH_NUM_FOR_ENABLE_OPENMP)
				{
					int nthreads = omp_get_num_threads();
#pragma omp parallel for schedule(dynamic, (nPixels+ntheads-1)/nthreads)
					for (int i = 0; i < nPixels; i++)
					{
						double temp = uxData[i] * uxData[i] + uyData[i] * uyData[i];
						phi_graduData[i] = 0.5 / sqrt(temp + varepsilon_phi);
					}
				}
				else
				{
#endif
					for (int i = 0; i < nPixels; i++)
					{
						double temp = uxData[i] * uxData[i] + uyData[i] * uyData[i];
						phi_graduData[i] = 0.5 / sqrt(temp + varepsilon_phi);
					}
#ifdef ZQLIB_USE_OPENMP
				}
#endif

				Phi_u_1st.reset();

				T*& phi_uData = Phi_u_1st.data();
				T*& uuData = uu.data();

#ifdef ZQLIB_USE_OPENMP
				if (&& nPixels > ZQ_DImage<T>::THRESH_NUM_FOR_ENABLE_OPENMP)
				{
					int nthreads = omp_get_num_threads();
#pragma omp parallel for schedule(dynamic, (nPixels+nthreads-1)/nthreads)
					for (int i = 0; i < nPixels; i++)
					{
						phi_uData[i] = 0.5 / sqrt(uuData[i] * uuData[i] + varepsilon_phi);
					}
				}
				else
				{
#endif
					for (int i = 0; i < nPixels; i++)
					{
						phi_uData[i] = 0.5 / sqrt(uuData[i] * uuData[i] + varepsilon_phi);
					}
#ifdef ZQLIB_USE_OPENMP
				}
#endif


				Phi_data_1st.reset();

				T*& phi_dataData = Phi_data_1st.data();
				T*& imdxData = imdx.data();
				T*& imdtData = imdt.data();
				T*& duData = du.data();
				T*& uData = u.data();

#ifdef ZQLIB_USE_OPENMP
				if ( && nPixels > ZQ_DImage<T>::THRESH_NUM_FOR_ENABLE_OPENMP)
				{
					int nthreads = omp_get_num_threads();
#pragma omp parallel for schedule(dynamic, (nPixels+nthreads-1)/nthreads)
					for (int i = 0; i < nPixels; i++)
					{
						for (int c = 0; c < nChannels; c++)
						{
							int offset = i*nChannels + c;
							double temp = imdtData[offset] + imdxData[offset] * duData[i];

							temp *= temp;
							phi_dataData[offset] = 0.5 / sqrt(temp + varepsilon_psi);
						}
					}
				}
				else
				{
#endif
					for (int i = 0; i < nPixels; i++)
					{
						for (int c = 0; c < nChannels; c++)
						{
							int offset = i*nChannels + c;
							double temp = imdtData[offset] + imdxData[offset] * duData[i];

							temp *= temp;
							phi_dataData[offset] = 0.5 / sqrt(temp + varepsilon_psi);
						}
					}
#ifdef ZQLIB_USE_OPENMP
				}
#endif

				imdxdx.Multiply(Phi_data_1st, imdx, imdx);
				imdtdx.Multiply(Phi_data_1st, imdx, imdt);

				if (nChannels > 1)
				{
					imdxdx.collapse();
					imdtdx.collapse();
				}

				T*& imdxdxData = imdxdx.data();
				T*& imdtdxData = imdtdx.data();


				Laplacian(laplace_u, u, Phi_gradu_1st);

				T*& laplace_uPtr = laplace_u.data();

				double omega = opt.omegaForSOR;
				double alpha = opt.alpha;
				double beta = opt.beta;

				/*Begin SOR*/

				for (int sor_it = 0; sor_it < opt.nSORIterations; sor_it++)
				{
					for (int i = 0; i < imHeight; i++)
					{
						for (int j = 0; j < imWidth; j++)
						{
							int offset = i*imWidth + j;
							double sigma1 = 0, coeff = 0;
							double _weight;


							if (j > 0)
							{
								_weight = phi_graduData[offset - 1];
								sigma1 += _weight*duData[offset - 1];
								coeff += _weight;
							}
							if (j < imWidth - 1)
							{
								_weight = phi_graduData[offset];
								sigma1 += _weight*duData[offset + 1];
								coeff += _weight;
							}
							if (i > 0)
							{
								_weight = phi_graduData[offset - imWidth];
								sigma1 += _weight*duData[offset - imWidth];
								coeff += _weight;
							}
							if (i < imHeight - 1)
							{
								_weight = phi_graduData[offset];
								sigma1 += _weight*duData[offset + imWidth];
								coeff += _weight;
							}
							sigma1 *= alpha;
							coeff *= alpha;

							sigma1 += alpha*laplace_uPtr[offset] - imdtdxData[offset] - beta*phi_uData[offset] * uData[offset];
							double coeff1 = coeff + imdxdxData[offset] + beta*phi_uData[offset];
							duData[offset] = (1 - omega)*duData[offset] + omega / coeff1*sigma1;
						}
					}
				}
				/*End SOR*/
			}

			if (opt.hasMaxUpdateLimit)
			{
				du.imclamp(-1.0, 1.0);
			}
			u.Addwith(du);


			warpFL(warpIm2, Im1, Im2, u, v, opt.useCubicWarping);
		}
	}


	template<class T>
	void ZQ_OpticalFlow::OneResolution_ADMM_L2(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ADMM_F_G(u, v, warpIm2, Im1, Im2, opt, Proximal_F_L2<T>, Proximal_G<T>);
	}

	template<class T>
	void ZQ_OpticalFlow::OneResolution_ADMM_DL1(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ADMM_F_G(u, v, warpIm2, Im1, Im2, opt, Proximal_F_DL1<T>, Proximal_G<T>);
	}

	template<class T>
	void ZQ_OpticalFlow::OneResolution_OneDir_Inc_L2(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();
		int vel_num = image_num - 1;

		int width = Im[0].width();
		int height = Im[0].height();
		int nChannels = Im[0].nchannels();

		switch (opt.initType)
		{
		case ZQ_OpticalFlowOptions::NONE_AS_INIT:
			break;
		case ZQ_OpticalFlowOptions::L2_AS_INIT:
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);
				OneResolution_HS_L2(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt);
			}
			break;
		case ZQ_OpticalFlowOptions::ADMM_AS_INIT:
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);
				OneResolution_ADMM_L2(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt);
			}
			break;
		}

		for (int alt_it = 0; alt_it < opt.nAlterations; alt_it++)
		{
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);

				if (i == 0)
				{
					ADMM_F_G(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt, Proximal_F_L2<T>, Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_last(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i - 1], v[i - 1], opt, Proximal_F_L2<T>, Proximal_F2_Last<T>, Proximal_G<T>);
				}

			}
		}
	}


	template<class T>
	void ZQ_OpticalFlow::OneResolution_OneDir_Inc_DL1(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();
		int vel_num = image_num - 1;

		int width = Im[0].width();
		int height = Im[0].height();
		int nChannels = Im[0].nchannels();

		switch (opt.initType)
		{
		case ZQ_OpticalFlowOptions::NONE_AS_INIT:
			break;
		case ZQ_OpticalFlowOptions::L2_AS_INIT:
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);
				OneResolution_HS_DL1(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt);
			}
			break;
		case ZQ_OpticalFlowOptions::ADMM_AS_INIT:
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);
				OneResolution_ADMM_DL1(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt);
			}
			break;
		}

		for (int alt_it = 0; alt_it < opt.nAlterations; alt_it++)
		{
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);

				if (i == 0)
				{
					ADMM_F_G(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt, Proximal_F_DL1<T>, Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_last(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i - 1], v[i - 1], opt, Proximal_F_DL1<T>, Proximal_F2_Last<T>, Proximal_G<T>);
				}
			}
		}
	}


	template<class T>
	void ZQ_OpticalFlow::OneResolution_OneDir_Dec_L2(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();
		int vel_num = image_num - 1;

		int width = Im[0].width();
		int height = Im[0].height();
		int nChannels = Im[0].nchannels();

		switch (opt.initType)
		{
		case ZQ_OpticalFlowOptions::NONE_AS_INIT:
			break;
		case ZQ_OpticalFlowOptions::L2_AS_INIT:
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);
				OneResolution_HS_L2(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt);
			}
			break;
		case ZQ_OpticalFlowOptions::ADMM_AS_INIT:
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);
				OneResolution_ADMM_L2(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt);
			}
			break;
		}

		for (int alt_it = 0; alt_it < opt.nAlterations; alt_it++)
		{
			for (int i = vel_num - 1; i >= 0; i--)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);

				if (i == vel_num - 1)
				{
					ADMM_F_G(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt, Proximal_F_L2<T>, Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_first(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i + 1], v[i + 1], opt, Proximal_F_L2<T>, Proximal_F2_First<T>, Proximal_G<T>);
				}
			}
		}
	}


	template<class T>
	void ZQ_OpticalFlow::OneResolution_OneDir_Dec_DL1(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();
		int vel_num = image_num - 1;

		int width = Im[0].width();
		int height = Im[0].height();
		int nChannels = Im[0].nchannels();

		switch (opt.initType)
		{
		case ZQ_OpticalFlowOptions::NONE_AS_INIT:
			break;
		case ZQ_OpticalFlowOptions::L2_AS_INIT:
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);
				OneResolution_HS_DL1(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt);
			}
			break;
		case ZQ_OpticalFlowOptions::ADMM_AS_INIT:
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);
				OneResolution_ADMM_DL1(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt);
			}
			break;
		}

		for (int alt_it = 0; alt_it < opt.nAlterations; alt_it++)
		{
			for (int i = vel_num - 1; i >= 0; i--)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);

				if (i == vel_num - 1)
				{
					ADMM_F_G(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt, Proximal_F_DL1<T>, Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_first(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i + 1], v[i + 1], opt, Proximal_F_DL1<T>, Proximal_F2_First<T>, Proximal_G<T>);
				}
			}
		}
	}

	template<class T>
	void ZQ_OpticalFlow::OneResolution_TwoDir_Inc_L2(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();
		int vel_num = image_num - 1;

		int width = Im[0].width();
		int height = Im[0].height();
		int nChannels = Im[0].nchannels();

		switch (opt.initType)
		{
		case ZQ_OpticalFlowOptions::NONE_AS_INIT:
			break;
		case ZQ_OpticalFlowOptions::L2_AS_INIT:
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);
				OneResolution_HS_L2(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt);
			}
			break;
		case ZQ_OpticalFlowOptions::ADMM_AS_INIT:
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);
				OneResolution_ADMM_L2(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt);
			}
			break;
		}

		for (int alt_it = 0; alt_it < opt.nAlterations; alt_it++)
		{
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);

				if (i == 0)
				{
					ADMM_F1_F2_G_first(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i + 1], v[i + 1], opt, Proximal_F_L2<T>, Proximal_F2_First<T>, Proximal_G<T>);
				}
				else if (i == vel_num - 1)
				{
					ADMM_F1_F2_G_last(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i - 1], v[i - 1], opt, Proximal_F_L2<T>, Proximal_F2_Last<T>, Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_middle(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i - 1], v[i - 1], u[i + 1], v[i + 1], opt, Proximal_F_L2<T>, Proximal_F2_Middle<T>, Proximal_G<T>);
				}
			}

			if (!opt.isReflect)
				continue;

			for (int i = vel_num - 1; i >= 0; i--)
			{
				if (i == 0)
				{
					ADMM_F1_F2_G_first(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i + 1], v[i + 1], opt, Proximal_F_L2<T>, Proximal_F2_First<T>, Proximal_G<T>);
				}
				else if (i == vel_num - 1)
				{
					ADMM_F1_F2_G_last(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i - 1], v[i - 1], opt, Proximal_F_L2<T>, Proximal_F2_Last<T>, Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_middle(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i - 1], v[i - 1], u[i + 1], v[i + 1], opt, Proximal_F_L2<T>, Proximal_F2_Middle<T>, Proximal_G<T>);
				}
			}
		}
	}

	template<class T>
	void ZQ_OpticalFlow::OneResolution_TwoDir_Inc_DL1(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();
		int vel_num = image_num - 1;

		int width = Im[0].width();
		int height = Im[0].height();
		int nChannels = Im[0].nchannels();

		switch (opt.initType)
		{
		case ZQ_OpticalFlowOptions::NONE_AS_INIT:
			break;
		case ZQ_OpticalFlowOptions::L2_AS_INIT:
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);
				OneResolution_HS_DL1(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt);
			}
			break;
		case ZQ_OpticalFlowOptions::ADMM_AS_INIT:
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);
				OneResolution_ADMM_DL1(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt);
			}
			break;
		}

		for (int alt_it = 0; alt_it < opt.nAlterations; alt_it++)
		{
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);

				if (i == 0)
				{
					ADMM_F1_F2_G_first(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i + 1], v[i + 1], opt, Proximal_F_DL1<T>, Proximal_F2_First<T>, Proximal_G<T>);
				}
				else if (i == vel_num - 1)
				{
					ADMM_F1_F2_G_last(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i - 1], v[i - 1], opt, Proximal_F_DL1<T>, Proximal_F2_Last<T>, Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_middle(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i - 1], v[i - 1], u[i + 1], v[i + 1], opt, Proximal_F_DL1<T>, Proximal_F2_Middle<T>, Proximal_G<T>);
				}
			}

			if (!opt.isReflect)
				continue;

			for (int i = vel_num - 1; i >= 0; i--)
			{
				if (i == 0)
				{
					ADMM_F1_F2_G_first(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i + 1], v[i + 1], opt, Proximal_F_DL1<T>, Proximal_F2_First<T>, Proximal_G<T>);
				}
				else if (i == vel_num - 1)
				{
					ADMM_F1_F2_G_last(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i - 1], v[i - 1], opt, Proximal_F_DL1<T>, Proximal_F2_Last<T>, Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_middle(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i - 1], v[i - 1], u[i + 1], v[i + 1], opt, Proximal_F_DL1<T>, Proximal_F2_Middle<T>, Proximal_G<T>);
				}
			}
		}
	}


	template<class T>
	void ZQ_OpticalFlow::OneResolution_TwoDir_Dec_L2(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();
		int vel_num = image_num - 1;

		int width = Im[0].width();
		int height = Im[0].height();
		int nChannels = Im[0].nchannels();

		switch (opt.initType)
		{
		case ZQ_OpticalFlowOptions::NONE_AS_INIT:
			break;
		case ZQ_OpticalFlowOptions::L2_AS_INIT:
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);
				OneResolution_HS_L2(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt);
			}
			break;
		case ZQ_OpticalFlowOptions::ADMM_AS_INIT:
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);
				OneResolution_ADMM_L2(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt);
			}
			break;
		}

		for (int alt_it = 0; alt_it < opt.nAlterations; alt_it++)
		{
			for (int i = vel_num - 1; i >= 0; i--)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);

				if (i == 0)
				{
					ADMM_F1_F2_G_first(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i + 1], v[i + 1], opt, Proximal_F_L2<T>, Proximal_F2_First<T>, Proximal_G<T>);
				}
				else if (i == vel_num - 1)
				{
					ADMM_F1_F2_G_last(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i - 1], v[i - 1], opt, Proximal_F_L2<T>, Proximal_F2_Last<T>, Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_middle(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i - 1], v[i - 1], u[i + 1], v[i + 1], opt, Proximal_F_L2<T>, Proximal_F2_Middle<T>, Proximal_G<T>);
				}
			}

			if (!opt.isReflect)
				continue;

			for (int i = 0; i < vel_num; i++)
			{
				if (i == 0)
				{
					ADMM_F1_F2_G_first(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i + 1], v[i + 1], opt, Proximal_F_L2<T>, Proximal_F2_First<T>, Proximal_G<T>);
				}
				else if (i == vel_num - 1)
				{
					ADMM_F1_F2_G_last(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i - 1], v[i - 1], opt, Proximal_F_L2<T>, Proximal_F2_Last<T>, Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_middle(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i - 1], v[i - 1], u[i + 1], v[i + 1], opt, Proximal_F_L2<T>, Proximal_F2_Middle<T>, Proximal_G<T>);
				}
			}
		}
	}


	template<class T>
	void ZQ_OpticalFlow::OneResolution_TwoDir_Dec_DL1(std::vector<ZQ_DImage<T>>& u, std::vector<ZQ_DImage<T>>& v, std::vector<ZQ_DImage<T>>& warpIm, const std::vector<ZQ_DImage<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();
		int vel_num = image_num - 1;

		int width = Im[0].width();
		int height = Im[0].height();
		int nChannels = Im[0].nchannels();

		switch (opt.initType)
		{
		case ZQ_OpticalFlowOptions::NONE_AS_INIT:
			break;
		case ZQ_OpticalFlowOptions::L2_AS_INIT:
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);
				OneResolution_HS_DL1(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt);
			}
			break;
		case ZQ_OpticalFlowOptions::ADMM_AS_INIT:
			for (int i = 0; i < vel_num; i++)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);
				OneResolution_ADMM_DL1(u[i], v[i], warpIm[i], Im[i], Im[i + 1], opt);
			}
			break;
		}

		for (int alt_it = 0; alt_it < opt.nAlterations; alt_it++)
		{
			for (int i = vel_num - 1; i >= 0; i--)
			{
				if (!warpIm[i].matchDimension(width, height, nChannels))
					warpIm[i].allocate(width, height, nChannels);

				if (i == 0)
				{
					ADMM_F1_F2_G_first(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i + 1], v[i + 1], opt, Proximal_F_DL1<T>, Proximal_F2_First<T>, Proximal_G<T>);
				}
				else if (i == vel_num - 1)
				{
					ADMM_F1_F2_G_last(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i - 1], v[i - 1], opt, Proximal_F_DL1<T>, Proximal_F2_Last<T>, Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_middle(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i - 1], v[i - 1], u[i + 1], v[i + 1], opt, Proximal_F_DL1<T>, Proximal_F2_Middle<T>, Proximal_G<T>);
				}
			}

			if (!opt.isReflect)
				continue;

			for (int i = 0; i < vel_num; i++)
			{
				if (i == 0)
				{
					ADMM_F1_F2_G_first(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i + 1], v[i + 1], opt, Proximal_F_DL1<T>, Proximal_F2_First<T>, Proximal_G<T>);
				}
				else if (i == vel_num - 1)
				{
					ADMM_F1_F2_G_last(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i - 1], v[i - 1], opt, Proximal_F_DL1<T>, Proximal_F2_Last<T>, Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_middle(u[i], v[i], warpIm[i], Im[i], Im[i + 1], u[i - 1], v[i - 1], u[i + 1], v[i + 1], opt, Proximal_F_DL1<T>, Proximal_F2_Middle<T>, Proximal_G<T>);
				}
			}
		}
	}

	/**************************  protected functions  *****************************/

	template<class T>
	void ZQ_OpticalFlow::getDxs(ZQ_DImage<T>& imdx, ZQ_DImage<T>& imdy, ZQ_DImage<T>& imdt, const ZQ_DImage<T>& im1, const ZQ_DImage<T>& im2, bool isSmooth/* = true*/)
	{
		//double gfilter[5]={0.01,0.09,0.8,0.09,0.01};
		T gfilter[5] = { 0.02, 0.11, 0.74, 0.11, 0.02 };

		if (1)
		{

			if (isSmooth)
			{
				ZQ_DImage<T> Im1, Im2, Im;
				im1.imfilter_hv(Im1, gfilter, 2, gfilter, 2);
				im2.imfilter_hv(Im2, gfilter, 2, gfilter, 2);
				Im.copyData(Im1);
				Im.Multiplywith(0.4);
				Im.Addwith(Im2, 0.6);
				Im.dx(imdx, isSmooth);
				Im.dy(imdy, isSmooth);
				imdt.Subtract(Im2, Im1);
			}
			else
			{
				ZQ_DImage<T> Im;
				Im.copyData(im1);
				Im.Multiplywith(0.4);
				Im.Addwith(im2, 0.6);
				Im.dx(imdx, isSmooth);
				Im.dy(imdy, isSmooth);
				imdt.Subtract(im2, im1);
			}

		}
		else
		{
			ZQ_DImage<T> Im1, Im2;
			im1.imfilter_hv(Im1, gfilter, 2, gfilter, 2);
			im2.imfilter_hv(Im2, gfilter, 2, gfilter, 2);
			Im2.dx(imdx, true);
			Im2.dy(imdy, true);
			imdt.Subtract(Im2, Im1);
		}
	}

	template<class T>
	void ZQ_OpticalFlow::warpFL(ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_DImage<T>& u, const ZQ_DImage<T>& v, bool isBicubic/* = false*/)
	{
		int nPixels = Im2.npixels();
		if (warpIm2.matchDimension(Im2) == false)
			warpIm2.allocate(Im2.width(), Im2.height(), Im2.nchannels());
		if (!isBicubic)
			ZQ_ImageProcessing::WarpImage(warpIm2.data(), Im2.data(), u.data(), v.data(), Im2.width(), Im2.height(), Im2.nchannels(), Im1.data(), false);
		else
			ZQ_ImageProcessing::WarpImageBicubic(warpIm2.data(), Im2.data(), u.data(), v.data(), Im2.width(), Im2.height(), Im2.nchannels(), Im1.data(), false);
	}

	template<class T>
	void ZQ_OpticalFlow::Laplacian(ZQ_DImage<T>& output, const ZQ_DImage<T>& input)
	{
		if (output.matchDimension(input) == false)
			output.allocate(input);
		else
			output.reset();

		int nPixels = input.npixels();
		ZQ_ImageProcessing::Laplacian(input.data(), output.data(), input.width(), input.height(), input.nchannels(), false);
	}

	template<class T>
	void ZQ_OpticalFlow::Laplacian(ZQ_DImage<T>& output, const ZQ_DImage<T>& input, const ZQ_DImage<T>& weight)
	{
		if (output.matchDimension(input) == false)
			output.allocate(input);
		output.reset();

		if (input.matchDimension(weight) == false)
		{
			printf("Error in image dimension matching ZQ_OpticalFlow::Laplacian()!\n");
			return;
		}

		const T*& inputData = input.data();
		const T*& weightData = weight.data();
		int width = input.width();
		int height = input.height();
		int nPixels = width*height;
		int nChannels = input.nchannels();
		ZQ_DImage<T> foo(width, height);

		T*& fooData = foo.data();
		T*& outputData = output.data();


		for (int c = 0; c < nChannels; c++)
		{
			foo.reset();
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width - 1; j++)
				{
					int offset = i*width + j;
					fooData[offset] = (inputData[(offset + 1)*nChannels + c] - inputData[offset*nChannels + c])*weightData[offset*nChannels + c];
				}
			}
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					int offset = i*width + j;
					if (j < width - 1)
						outputData[offset*nChannels + c] += fooData[offset];
					if (j > 0)
						outputData[offset*nChannels + c] -= fooData[offset - 1];
				}
			}
			foo.reset();

			for (int i = 0; i < height - 1; i++)
			{
				for (int j = 0; j < width; j++)
				{
					int offset = i*width + j;
					fooData[offset] = (inputData[(offset + width)*nChannels + c] - inputData[offset*nChannels + c])*weightData[offset*nChannels + c];
				}
			}

			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					int offset = i*width + j;
					if (i < height - 1)
						outputData[offset*nChannels + c] += fooData[offset];
					if (i > 0)
						outputData[offset*nChannels + c] -= fooData[offset - width];
				}
			}
		}
	}


	// function to convert image to features
	template<class T>
	void ZQ_OpticalFlow::im2feature(ZQ_DImage<T>& imfeature, const ZQ_DImage<T>& im, bool isSmooth, const ZQ_OpticalFlowOptions& opt)
	{
		int width = im.width();
		int height = im.height();
		int nPixels = width*height;
		int nchannels = im.nchannels();
		if (nchannels == 1)
		{
			switch (opt.featureType)
			{
			case ZQ_OpticalFlowOptions::FEATURE_GRADIENT:
				if (true){
					imfeature.allocate(width, height, 3);
					ZQ_DImage<T> imdx, imdy;
					im.dx(imdx, isSmooth);
					im.dy(imdy, isSmooth);
					T*& data = imfeature.data();
					const T*& im_Data = im.data();
					T*& imdx_Data = imdx.data();
					T*& imdy_Data = imdy.data();

					for (int i = 0; i < height; i++)
					{
						for (int j = 0; j < width; j++)
						{
							int offset = i*width + j;
							data[offset * 3] = im_Data[offset];
							data[offset * 3 + 1] = imdx_Data[offset] * opt.gradWeight;
							data[offset * 3 + 2] = imdy_Data[offset] * opt.gradWeight;
						}
					}
				}
				break;

			case ZQ_OpticalFlowOptions::FEATURE_FORWARD_NEIGHBOR:
				if (true){
					imfeature.allocate(width, height, 3);
					T*& data = imfeature.data();
					const T*& im_Data = im.data();

					for (int h = 0; h < height; h++)
					{
						for (int w = 0; w < width; w++)
						{
							int offset = h*width + w;
							int hh[3] = { h, h, h + 1 };
							int ww[3] = { w, w + 1, w };
							float wei[3] = { 1, 1, 1 };
							for (int cid = 0; cid < 3; cid++)
							{
								int real_hh = __max(0, __min(height - 1, hh[cid]));
								int real_ww = __max(0, __min(width - 1, ww[cid]));
								data[offset * 3 + cid] = wei[cid] * im_Data[real_hh*width + real_ww];
							}
						}
					}
				}
				break;

			case ZQ_OpticalFlowOptions::FEATURE_BIDIRECTIONAL_NEIGHBOR:
				if (true){
					imfeature.allocate(width, height, 5);
					T*& data = imfeature.data();
					const T*& im_Data = im.data();

					for (int h = 0; h < height; h++)
					{
						for (int w = 0; w < width; w++)
						{
							int offset = h*width + w;
							int hh[5] = { h, h, h + 1, h, h - 1 };
							int ww[5] = { w, w + 1, w, w - 1, w };
							float wei[5] = { 1, 1, 1, 1, 1 };
							for (int cid = 0; cid < 5; cid++)
							{
								int real_hh = __max(0, __min(height - 1, hh[cid]));
								int real_ww = __max(0, __min(width - 1, ww[cid]));
								data[offset * 5 + cid] = wei[cid] * im_Data[real_hh*width + real_ww];
							}
						}
					}
				}
				break;
			}
		}
		else
		{
			imfeature.copyData(im);
		}
	}

	template<class T>
	void ZQ_OpticalFlow::detectOcclusion(ZQ_DImage<T>& occ, const ZQ_DImage<T>& u, const ZQ_DImage<T>& v, const ZQ_DImage<T>& im1, const ZQ_DImage<T>& im2, double sigma_div, double sigma_im)
	{
		if (!occ.matchDimension(u))
			occ.allocate(u);
		else
			occ.reset();

		ZQ_DImage<T> ux, vy, div, warpIm2, imdt;
		u.dx_3pt(ux);
		v.dy_3pt(vy);
		div.Add(ux, vy);
		int width = im1.width();
		int height = im1.height();
		int nPixels = width*height;
		int nChannles = im1.nchannels();
		warpIm2.allocate(width, height, nChannles);
		ZQ_ImageProcessing::WarpImage(warpIm2.data(), im2.data(), u.data(), v.data(), width, height, nChannles, im1.data(), false);
		imdt.Subtract(warpIm2, im1);
		T*& occ_data = occ.data();
		T*& div_data = div.data();
		T*& imdt_data = imdt.data();

		double sigma_div2 = sigma_div*sigma_div;
		double sigma_im2 = sigma_im*sigma_im;

		for (int i = 0; i < nPixels; i++)
		{
			occ_data[i] = exp(-0.5*div_data[i] * div_data[i] / sigma_div2) * exp(-0.5*imdt_data[i*nChannles] * imdt_data[i*nChannles] / sigma_im2);
		}
	}

	template<class T>
	void ZQ_OpticalFlow::detectOcclusion(ZQ_DImage<T>& occ, const ZQ_DImage<T>& u, const ZQ_DImage<T>& im1, const ZQ_DImage<T>& im2, double sigma_div, double sigma_im)
	{
		if (!occ.matchDimension(u))
			occ.allocate(u);
		else
			occ.reset();

		ZQ_DImage<T> div, warpIm2, imdt;
		u.dx_3pt(div);
		ZQ_DImage<T> v;
		v.allocate(u);
		int width = im1.width();
		int height = im1.height();
		int nPixels = width*height;
		int nChannles = im1.nchannels();
		warpIm2.allocate(width, height, nChannles);
		ZQ_ImageProcessing::WarpImage(warpIm2.data(), im2.data(), u.data(), v.data(), width, height, nChannles, im1.data(), false);
		imdt.Subtract(warpIm2, im1);
		if (imdt > 1)
			imdt.collapse();
		int nPixels = occ.npixels();
		T*& occ_data = occ.data();
		T*& div_data = div.data();
		T*& imdt_data = imdt.data();

		double sigma_div2 = sigma_div*sigma_div;
		double sigma_im2 = sigma_im*sigma_im;

		for (int i = 0; i < nPixels; i++)
		{
			occ_data[i] = exp(-0.5*div_data[i] * div_data[i] / sigma_div2) * exp(-0.5*imdt_data[i] * imdt_data[i] / sigma_im2);
		}
	}

	template<class T>
	void ZQ_OpticalFlow::denoiseColorWeightedMedianFilter(ZQ_DImage<T>& out_u, ZQ_DImage<T>& out_v, const ZQ_DImage<T>& u, const ZQ_DImage<T>& v, const ZQ_DImage<T>& im,
		const ZQ_DImage<T>& occ, int weightedMedFiltSize, int medFiltSize, double sigma_im)
	{
		if (medFiltSize > 1)
		{
			u.MedianFilter(out_u, medFiltSize);
			v.MedianFilter(out_v, medFiltSize);
		}
		else
		{
			out_u.copyData(u);
			out_v.copyData(v);
		}
		int width = u.width();
		int height = u.height();
		int nPixels = width*height;
		ZQ_DImage<bool> edge_x(width, height), edge_y(width, height), edge(width, height);

		bool*& edge_xData = edge_x.data();
		bool*& edge_yData = edge_y.data();
		bool*& edge_data = edge.data();
		const T*& uData = u.data();
		const T*& vData = v.data();
		ZQ_ImageProcessing::Edge(uData, edge_xData, width, height, 4.0f, false);
		ZQ_ImageProcessing::Edge(vData, edge_yData, width, height, 4.0f, false);

		for (int i = 0; i < width*height; i++)
			edge_data[i] = edge_xData[i] || edge_yData[i];


		//reuse edge_x for mask
		bool*& mask = edge_x.data();
		bool dilate_filter2D[25] =
		{
			1, 1, 1, 1, 1,
			1, 1, 1, 1, 1,
			1, 1, 1, 1, 1,
			1, 1, 1, 1, 1,
			1, 1, 1, 1, 1
		};

		ZQ_BinaryImageProcessing::Dilate(edge_data, mask, width, height, dilate_filter2D, 2, 2, false);
		//ZQ_ImageIO::saveImage(edge_x, "edge.png");

		double sigma_dis = 7;
		int weighted_XSIZE = 2 * weightedMedFiltSize + 1;
		ZQ_DImage<T> dis_weight(weighted_XSIZE, weighted_XSIZE);
		ZQ_DImage<T> weights(weighted_XSIZE, weighted_XSIZE);
		ZQ_DImage<T> neighbor_u(weighted_XSIZE, weighted_XSIZE);
		ZQ_DImage<T> neighbor_v(weighted_XSIZE, weighted_XSIZE);
		T*& dis_weight_data = dis_weight.data();
		T*& weights_data = weights.data();
		T*& neighbor_uData = neighbor_u.data();
		T*& neighbor_vData = neighbor_v.data();
		for (int i = -weightedMedFiltSize; i <= weightedMedFiltSize; i++)
		{
			for (int j = -weightedMedFiltSize; j <= weightedMedFiltSize; j++)
			{
				dis_weight_data[(i + weightedMedFiltSize)*(2 * weightedMedFiltSize + 1) + j + weightedMedFiltSize] = exp(-0.5*(i*i + j*j) / (sigma_dis*sigma_dis));
			}
		}

		T*& out_uData = out_u.data();
		T*& out_vData = out_v.data();
		const T*& im_data = im.data();
		const T*& occ_data = occ.data();
		int nChannels = im.nchannels();

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (!edge_data[i*width + j])
				{
					continue;
				}
				int num = 0;
				for (int yy = -weightedMedFiltSize; yy <= weightedMedFiltSize; yy++)
				{
					for (int xx = -weightedMedFiltSize; xx <= weightedMedFiltSize; xx++)
					{
						int ii = i + yy;
						int jj = j + xx;
						if (jj < 0)
							jj = -jj;
						else if (jj >= width)
							jj = width - 1 - (jj - (width - 1));
						if (ii < 0)
							ii = -ii;
						else if (ii >= height)
							ii = height - 1 - (ii - (height - 1));

						double color_dis2 = 0;
						for (int c = 0; c < nChannels; c++)
						{
							double tmp_color_dis = im_data[(ii*width + jj)*nChannels + c] - im_data[(i*width + j)*nChannels + c];
							color_dis2 += tmp_color_dis*tmp_color_dis;
						}
						color_dis2 /= nChannels;
						double color_weight = exp(-0.5*color_dis2 / (sigma_im*sigma_im));
						double occ_weight = occ_data[ii*width + jj];
						weights_data[num] = color_weight * occ_weight * dis_weight_data[(yy + weightedMedFiltSize)*(2 * weightedMedFiltSize + 1) + xx + weightedMedFiltSize];
						neighbor_uData[num] = uData[ii*width + jj];
						neighbor_vData[num] = vData[ii*width + jj];
						num++;
					}
				}
				double sum_weight = 0;
				for (int inum = 0; inum < num; inum++)
				{
					sum_weight += weights_data[inum];
				}
				for (int inum = 0; inum < num; inum++)
				{
					weights_data[inum] /= sum_weight;
				}
				ZQ_WeightedMedian::FindMedian(neighbor_uData, weights_data, num, out_uData[i*width + j]);
				ZQ_WeightedMedian::FindMedian(neighbor_vData, weights_data, num, out_vData[i*width + j]);
			}
		}
	}

	template<class T>
	void ZQ_OpticalFlow::denoiseColorWeightedMedianFilter(ZQ_DImage<T>& out_u, const ZQ_DImage<T>& u, const ZQ_DImage<T>& im, const ZQ_DImage<T>& occ, int weightedMedFiltSize, int medFiltSize, double sigma_im)
	{
		ZQ_DImage<T> tmp_u;
		if (medFiltSize > 1)
		{
			u.MedianFilter(tmp_u, medFiltSize);
		}
		else
		{
			tmp_u.copyData(u);
		}
		int width = u.width();
		int height = u.height();
		int nPixels = width*height;
		ZQ_DImage<bool> edge(width, height);

		bool*& edge_data = edge.data();
		T*& tmp_uData = tmp_u.data();
		ZQ_ImageProcessing::Edge(tmp_uData, edge_data, width, height, 4.0f, false);

		//ZQ_ImageIO::saveImage(edge, "mask1.png");

		ZQ_DImage<bool> mask(width, height);
		bool*& mask_data = mask.data();
		bool dilate_filter2D[25] =
		{
			1, 1, 1, 1, 1,
			1, 1, 1, 1, 1,
			1, 1, 1, 1, 1,
			1, 1, 1, 1, 1,
			1, 1, 1, 1, 1
		};
		ZQ_BinaryImageProcessing::Dilate(edge_data, mask_data, width, height, dilate_filter2D, 1, 1, false);

		//ZQ_ImageIO::saveImage(mask, "mask2.png");

		double sigma_dis = 7;
		int weighted_XSIZE = 2 * weightedMedFiltSize + 1;
		ZQ_DImage<T> dis_weight(weighted_XSIZE, weighted_XSIZE);
		ZQ_DImage<T> weights(weighted_XSIZE, weighted_XSIZE);
		ZQ_DImage<T> neighbor_u(weighted_XSIZE, weighted_XSIZE);
		T*& dis_weight_data = dis_weight.data();
		T*& weights_data = weights.data();
		T*& neighbor_uData = neighbor_u.data();
		for (int i = -weightedMedFiltSize; i <= weightedMedFiltSize; i++)
		{
			for (int j = -weightedMedFiltSize; j <= weightedMedFiltSize; j++)
			{
				dis_weight_data[(i + weightedMedFiltSize)*(2 * weightedMedFiltSize + 1) + j + weightedMedFiltSize] = exp(-0.5*(i*i + j*j) / (sigma_dis*sigma_dis));
			}
		}

		out_u.copyData(tmp_u);
		T*& out_uData = out_u.data();
		const T*& im_data = im.data();
		const T*& occ_data = occ.data();
		int nChannels = im.nchannels();

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (!mask_data[i*width + j])
					continue;
				int num = 0;
				for (int yy = -weightedMedFiltSize; yy <= weightedMedFiltSize; yy++)
				{
					for (int xx = -weightedMedFiltSize; xx <= weightedMedFiltSize; xx++)
					{
						int ii = i + yy;
						int jj = j + xx;
						if (ii < 0 || ii >= height || jj < 0 || jj >= width)
							continue;
						double color_dis2 = 0;
						for (int c = 0; c < nChannels; c++)
						{
							double tmp_color_dis = im_data[(ii*width + jj)*nChannels + c] - im_data[(i*width + j)*nChannels + c];
							color_dis2 = tmp_color_dis*tmp_color_dis;
						}
						color_dis2 /= nChannels;
						double color_weight = exp(-0.5*color_dis2 / (sigma_im*sigma_im));
						double occ_weight = occ_data[ii*width + jj];
						weights_data[num] = color_weight * occ_weight * dis_weight_data[(yy + weightedMedFiltSize)*(2 * weightedMedFiltSize + 1) + xx + weightedMedFiltSize];
						neighbor_uData[num] = tmp_uData[ii*width + jj];
						num++;
					}
				}
				ZQ_WeightedMedian::FindMedian(neighbor_uData, weights_data, num, out_uData[i*width + j]);
			}
		}
	}



	/**************************************************************************/

	template<class T>
	void ZQ_OpticalFlow::ADMM_F_G(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_OpticalFlowOptions& opt,
		void(*funcF)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, ZQ_DImage<T>& /*warpIm2*/, const ZQ_DImage<T>& /*Im1*/, const ZQ_DImage<T>& /*Im2*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_OpticalFlowOptions& /*opt*/),
		void(*funcG)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_OpticalFlowOptions& /*opt*/))
	{
		ZQ_DImage<T> u_for_F(u), v_for_F(v);
		ZQ_DImage<T> u_for_G(u), v_for_G(v);
		ZQ_DImage<T> u_for_q(u), v_for_q(v);
		ZQ_DImage<T> z_u, z_v;


		u_for_q.reset(), v_for_q.reset();


		for (int it = 0; it < opt.nADMMIterations; it++)
		{
			z_u.copyData(u_for_G);
			z_u.Addwith(u_for_q, -1.0 / opt.lambda);
			z_v.copyData(v_for_G);
			z_v.Addwith(v_for_q, -1.0 / opt.lambda);


			funcF(u_for_F, v_for_F, warpIm2, Im1, Im2, z_u, z_v, opt);

			z_u.copyData(u_for_F);
			z_u.Addwith(u_for_q, 1.0 / opt.lambda);
			z_v.copyData(v_for_F);
			z_v.Addwith(v_for_q, 1.0 / opt.lambda);

			funcG(u_for_G, v_for_G, z_u, z_v, opt);

			u_for_q.Addwith(u_for_F, opt.lambda);
			u_for_q.Addwith(u_for_G, -opt.lambda);
			v_for_q.Addwith(v_for_F, opt.lambda);
			v_for_q.Addwith(v_for_G, -opt.lambda);
		}
		u.copyData(u_for_F);
		v.copyData(v_for_F);
	}

	template<class T>
	void ZQ_OpticalFlow::ADMM_F1_F2_G_first(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_DImage<T>& next_u, const ZQ_DImage<T>& next_v, const ZQ_OpticalFlowOptions& opt,
		void(*funcF1)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, ZQ_DImage<T>& /*warpIm2*/, const ZQ_DImage<T>& /*Im1*/, const ZQ_DImage<T>& /*Im2*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_OpticalFlowOptions& /*opt*/),
		void(*funcF2)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_DImage<T>& /*next_u*/, const ZQ_DImage<T>& /*next_v*/, const ZQ_OpticalFlowOptions& /*opt*/),
		void(*funcG)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_OpticalFlowOptions& /*opt*/))
	{
		ZQ_DImage<T> u_for_F1(u), v_for_F1(v);
		ZQ_DImage<T> u_for_F2(u), v_for_F2(v);
		ZQ_DImage<T> u_for_G(u), v_for_G(v);
		ZQ_DImage<T> u_for_q1(u), v_for_q1(v);
		ZQ_DImage<T> u_for_q2(u), v_for_q2(v);
		ZQ_DImage<T> z_u, z_v;

		u_for_q1.reset(), v_for_q1.reset();
		u_for_q2.reset(), v_for_q2.reset();

		for (int it = 0; it < opt.nADMMIterations; it++)
		{
			z_u.copyData(u_for_G);
			z_u.Addwith(u_for_q1, -1.0 / opt.lambda);
			z_v.copyData(v_for_G);
			z_v.Addwith(v_for_q1, -1.0 / opt.lambda);

			funcF1(u_for_F1, v_for_F1, warpIm2, Im1, Im2, z_u, z_v, opt);

			z_u.copyData(u_for_G);
			z_u.Addwith(u_for_q2, -1.0 / opt.lambda);
			z_v.copyData(v_for_G);
			z_v.Addwith(v_for_q2, -1.0 / opt.lambda);

			funcF2(u_for_F2, v_for_F2, z_u, z_v, next_u, next_v, opt);

			z_u.copyData(u_for_F1);
			z_u.Addwith(u_for_q1, 1.0 / opt.lambda);
			z_u.Addwith(u_for_F2);
			z_u.Addwith(u_for_q2, 1.0 / opt.lambda);
			z_u.Multiplywith(0.5);

			z_v.copyData(v_for_F1);
			z_v.Addwith(v_for_q1, 1.0 / opt.lambda);
			z_v.Addwith(v_for_F2);
			z_v.Addwith(v_for_q2, 1.0 / opt.lambda);
			z_v.Multiplywith(0.5);

			funcG(u_for_G, v_for_G, z_u, z_v, opt);

			u_for_q1.Addwith(u_for_F1, opt.lambda);
			u_for_q1.Addwith(u_for_G, -opt.lambda);
			v_for_q1.Addwith(v_for_F1, opt.lambda);
			v_for_q1.Addwith(v_for_G, -opt.lambda);

			u_for_q2.Addwith(u_for_F2, opt.lambda);
			u_for_q2.Addwith(u_for_G, -opt.lambda);
			v_for_q2.Addwith(v_for_F2, opt.lambda);
			v_for_q2.Addwith(v_for_G, -opt.lambda);
		}
		u.copyData(u_for_F1);
		v.copyData(v_for_F1);
	}

	template<class T>
	void ZQ_OpticalFlow::ADMM_F1_F2_G_last(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_DImage<T>& pre_u, const ZQ_DImage<T>& pre_v, const ZQ_OpticalFlowOptions& opt,
		void(*funcF1)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, ZQ_DImage<T>& /*warpIm2*/, const ZQ_DImage<T>& /*Im1*/, const ZQ_DImage<T>& /*Im2*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_OpticalFlowOptions& /*opt*/),
		void(*funcF2)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_DImage<T>& /*pre_u*/, const ZQ_DImage<T>& /*pre_v*/, const ZQ_OpticalFlowOptions& /*opt*/),
		void(*funcG)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_OpticalFlowOptions& /*opt*/))
	{
		ZQ_DImage<T> u_for_F1(u), v_for_F1(v);
		ZQ_DImage<T> u_for_F2(u), v_for_F2(v);
		ZQ_DImage<T> u_for_G(u), v_for_G(v);
		ZQ_DImage<T> u_for_q1(u), v_for_q1(v);
		ZQ_DImage<T> u_for_q2(u), v_for_q2(v);
		ZQ_DImage<T> z_u, z_v;


		u_for_q1.reset(), v_for_q1.reset();
		u_for_q2.reset(), v_for_q2.reset();

		for (int it = 0; it < opt.nADMMIterations; it++)
		{
			z_u.copyData(u_for_G);
			z_u.Addwith(u_for_q1, -1.0 / opt.lambda);
			z_v.copyData(v_for_G);
			z_v.Addwith(v_for_q1, -1.0 / opt.lambda);

			funcF1(u_for_F1, v_for_F1, warpIm2, Im1, Im2, z_u, z_v, opt);

			z_u.copyData(u_for_G);
			z_u.Addwith(u_for_q2, -1.0 / opt.lambda);
			z_v.copyData(v_for_G);
			z_v.Addwith(v_for_q2, -1.0 / opt.lambda);

			funcF2(u_for_F2, v_for_F2, z_u, z_v, pre_u, pre_v, opt);

			z_u.copyData(u_for_F1);
			z_u.Addwith(u_for_q1, 1.0 / opt.lambda);
			z_u.Addwith(u_for_F2);
			z_u.Addwith(u_for_q2, 1.0 / opt.lambda);
			z_u.Multiplywith(0.5);

			z_v.copyData(v_for_F1);
			z_v.Addwith(v_for_q1, 1.0 / opt.lambda);
			z_v.Addwith(v_for_F2);
			z_v.Addwith(v_for_q2, 1.0 / opt.lambda);
			z_v.Multiplywith(0.5);

			funcG(u_for_G, v_for_G, z_u, z_v, opt);

			u_for_q1.Addwith(u_for_F1, opt.lambda);
			u_for_q1.Addwith(u_for_G, -opt.lambda);
			v_for_q1.Addwith(v_for_F1, opt.lambda);
			v_for_q1.Addwith(v_for_G, -opt.lambda);

			u_for_q2.Addwith(u_for_F2, opt.lambda);
			u_for_q2.Addwith(u_for_G, -opt.lambda);
			v_for_q2.Addwith(v_for_F2, opt.lambda);
			v_for_q2.Addwith(v_for_G, -opt.lambda);
		}
		u.copyData(u_for_F1);
		v.copyData(v_for_F1);
	}


	template<class T>
	void ZQ_OpticalFlow::ADMM_F1_F2_G_middle(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_DImage<T>& pre_u, const ZQ_DImage<T>& pre_v, const ZQ_DImage<T>& next_u, const ZQ_DImage<T>& next_v, const ZQ_OpticalFlowOptions& opt,
		void(*funcF1)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, ZQ_DImage<T>& /*warpIm2*/, const ZQ_DImage<T>& /*Im1*/, const ZQ_DImage<T>& /*Im2*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_OpticalFlowOptions& /*opt*/),
		void(*funcF2)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_DImage<T>& /*pre_u*/, const ZQ_DImage<T>& /*pre_v*/, const ZQ_DImage<T>& /*next_u*/, const ZQ_DImage<T>& /*next_v*/, const ZQ_OpticalFlowOptions& /*opt*/),
		void(*funcG)(ZQ_DImage<T>& /*u*/, ZQ_DImage<T>& /*v*/, const ZQ_DImage<T>& /*z_u*/, const ZQ_DImage<T>& /*z_v*/, const ZQ_OpticalFlowOptions& /*opt*/))
	{
		ZQ_DImage<T> u_for_F1(u), v_for_F1(v);
		ZQ_DImage<T> u_for_F2(u), v_for_F2(v);
		ZQ_DImage<T> u_for_G(u), v_for_G(v);
		ZQ_DImage<T> u_for_q1(u), v_for_q1(v);
		ZQ_DImage<T> u_for_q2(u), v_for_q2(v);
		ZQ_DImage<T> z_u, z_v;


		u_for_q1.reset(), v_for_q1.reset();
		u_for_q2.reset(), v_for_q2.reset();

		for (int it = 0; it < opt.nADMMIterations; it++)
		{
			z_u.copyData(u_for_G);
			z_u.Addwith(u_for_q1, -1.0 / opt.lambda);
			z_v.copyData(v_for_G);
			z_v.Addwith(v_for_q1, -1.0 / opt.lambda);

			funcF1(u_for_F1, v_for_F1, warpIm2, Im1, Im2, z_u, z_v, opt);

			z_u.copyData(u_for_G);
			z_u.Addwith(u_for_q2, -1.0 / opt.lambda);
			z_v.copyData(v_for_G);
			z_v.Addwith(v_for_q2, -1.0 / opt.lambda);

			funcF2(u_for_F2, v_for_F2, z_u, z_v, pre_u, pre_v, next_u, next_v, opt);

			z_u.copyData(u_for_F1);
			z_u.Addwith(u_for_q1, 1.0 / opt.lambda);
			z_u.Addwith(u_for_F2);
			z_u.Addwith(u_for_q2, 1.0 / opt.lambda);
			z_u.Multiplywith(0.5);

			z_v.copyData(v_for_F1);
			z_v.Addwith(v_for_q1, 1.0 / opt.lambda);
			z_v.Addwith(v_for_F2);
			z_v.Addwith(v_for_q2, 1.0 / opt.lambda);
			z_v.Multiplywith(0.5);

			funcG(u_for_G, v_for_G, z_u, z_v, opt);

			u_for_q1.Addwith(u_for_F1, opt.lambda);
			u_for_q1.Addwith(u_for_G, -opt.lambda);
			v_for_q1.Addwith(v_for_F1, opt.lambda);
			v_for_q1.Addwith(v_for_G, -opt.lambda);

			u_for_q2.Addwith(u_for_F2, opt.lambda);
			u_for_q2.Addwith(u_for_G, -opt.lambda);
			v_for_q2.Addwith(v_for_F2, opt.lambda);
			v_for_q2.Addwith(v_for_G, -opt.lambda);
		}
		u.copyData(u_for_F1);
		v.copyData(v_for_F1);
	}


	/****************************************************************************/
	template<class T>
	void ZQ_OpticalFlow::Proximal_F_L2(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_DImage<T>& z_u, const ZQ_DImage<T>& z_v, const ZQ_OpticalFlowOptions& opt)
	{
		int imWidth = Im1.width();
		int imHeight = Im1.height();
		int nChannels = Im1.nchannels();
		int nPixels = imWidth*imHeight;

		ZQ_DImage<T> du(imWidth, imHeight), dv(imWidth, imHeight); //for du, dv


		/* ProximalF(z_u,z_v,\lambda) = minimize_{u,v} \int {|I_2(x+u,y+v)-I_1(x,y)|^2} + \alpha^2 \int {|\nabla u|^2 + |\nabla v|^2} + \beta^2 \int {|u|^2 + |v|^2} + 0.5*\lambda \int {|u-z_u|^2 + |v-z_v|^2}
		*
		* The Euler-Lagrange equation is:
		*  I_t I_x + \beta^2 u + 0.5*\lambda(u-z_u) = \alpha^2 \Delta u
		*  I_t I_y + \beta^2 v + 0.5*\lambda(v-z_v) = \alpha^2 \Delta v
		*/


		warpFL(warpIm2, Im1, Im2, u, v, opt.useCubicWarping);

		/************       Outer Loop Begin     *************/
		//refresh {u,v} in each loop
		for (int count = 0; count < opt.nOuterFixedPointIterations; count++)
		{
			// outer loop : {imdx, imdy, imdt} accoring to warpIm2, warpIm2 accoring to {u,v} 
			ZQ_DImage<T> imdx, imdy, imdt;

			getDxs(imdx, imdy, imdt, Im1, warpIm2, true);

			du.reset();
			dv.reset();

			//compute imdtdx, imdxdx, imdydy, imdtdx, imdtdy
			ZQ_DImage<T> imdxdy, imdxdx, imdydy, imdtdx, imdtdy;
			imdxdx.Multiply(imdx, imdx);
			imdxdy.Multiply(imdx, imdy);
			imdydy.Multiply(imdy, imdy);
			imdtdx.Multiply(imdx, imdt);
			imdtdy.Multiply(imdy, imdt);

			if (nChannels > 1)
			{
				imdxdx.collapse();
				imdxdy.collapse();
				imdydy.collapse();
				imdtdx.collapse();
				imdtdy.collapse();
			}

			ZQ_DImage<T> laplace_u(imWidth, imHeight);
			ZQ_DImage<T> laplace_v(imWidth, imHeight);

			Laplacian(laplace_u, u);
			Laplacian(laplace_v, v);

			T*& laplace_uData = laplace_u.data();
			T*& laplace_vData = laplace_v.data();


			// set omega
			double omega = opt.omegaForSOR;
			double alpha2 = opt.alpha*opt.alpha;
			double beta2 = opt.beta*opt.beta;

			T*& duData = du.data();
			T*& dvData = dv.data();
			T*& uData = u.data();
			T*& vData = v.data();
			const T*& z_uData = z_u.data();
			const T*& z_vData = z_v.data();
			T*& imdtdxData = imdtdx.data();
			T*& imdtdyData = imdtdy.data();
			T*& imdxdyData = imdxdy.data();
			T*& imdxdxData = imdxdx.data();
			T*& imdydyData = imdydy.data();



			/***   SOR Begin solve du,dv***/

			for (int k = 0; k < opt.nSORIterations; k++)
			{
				for (int i = 0; i < imHeight; i++)
				{
					for (int j = 0; j<imWidth; j++)
					{
						int offset = i * imWidth + j;
						double sigma1 = 0, sigma2 = 0, coeff = 0;
						double _weight;

						if (j>0)
						{
							_weight = 1;
							sigma1 += _weight*duData[offset - 1];
							sigma2 += _weight*dvData[offset - 1];
							coeff += _weight;

						}
						if (j<imWidth - 1)
						{
							_weight = 1;
							sigma1 += _weight*duData[offset + 1];
							sigma2 += _weight*dvData[offset + 1];
							coeff += _weight;
						}
						if (i>0)
						{
							_weight = 1;
							sigma1 += _weight*duData[offset - imWidth];
							sigma2 += _weight*dvData[offset - imWidth];
							coeff += _weight;
						}
						if (i < imHeight - 1)
						{
							_weight = 1;
							sigma1 += _weight*duData[offset + imWidth];
							sigma2 += _weight*dvData[offset + imWidth];
							coeff += _weight;
						}
						sigma1 *= alpha2;
						sigma2 *= alpha2;
						coeff *= alpha2;
						// compute u
						sigma1 += alpha2*(laplace_uData[offset]) - imdtdxData[offset] - imdxdyData[offset] * dvData[offset] - beta2*uData[offset] - 0.5*opt.lambda*(uData[offset] - z_uData[offset]);
						double coeff1 = coeff + imdxdxData[offset] + beta2 + 0.5*opt.lambda;
						duData[offset] = (1 - omega)*duData[offset] + omega / coeff1*sigma1;
						// compute v
						sigma2 += alpha2*(laplace_vData[offset]) - imdtdyData[offset] - imdxdyData[offset] * duData[offset] - beta2*vData[offset] - 0.5*opt.lambda*(vData[offset] - z_vData[offset]);
						double coeff2 = coeff + imdydyData[offset] + beta2 + 0.5*opt.lambda;
						dvData[offset] = (1 - omega)*dvData[offset] + omega / coeff2*sigma2;

					}
				}
			}

			/***   SOR end solve du,dv***/


			u.Addwith(du);
			v.Addwith(dv);

			warpFL(warpIm2, Im1, Im2, u, v, opt.useCubicWarping);

		}
		/************       Outer Loop End     *************/
	}

	template<class T>
	void ZQ_OpticalFlow::Proximal_F_DL1(ZQ_DImage<T>& u, ZQ_DImage<T>& v, ZQ_DImage<T>& warpIm2, const ZQ_DImage<T>& Im1, const ZQ_DImage<T>& Im2, const ZQ_DImage<T>& z_u, const ZQ_DImage<T>& z_v, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_DImage<T> imdx, imdy, imdt;

		int imWidth = Im1.width();
		int imHeight = Im1.height();
		int nChannels = Im1.nchannels();
		int nPixels = imWidth*imHeight;

		ZQ_DImage<T> du(imWidth, imHeight), dv(imWidth, imHeight);

		ZQ_DImage<T> Psi_1st(imWidth, imHeight, nChannels);

		ZQ_DImage<T> imdxdx, imdxdy, imdydy, imdtdx, imdtdy;

		ZQ_DImage<T> laplace_u, laplace_v;

		double varepsilon_phi = pow(0.001, 2);
		double varepsilon_psi = pow(0.001, 2);

		warpFL(warpIm2, Im1, Im2, u, v, opt.useCubicWarping);


		for (int out_it = 0; out_it < opt.nOuterFixedPointIterations; out_it++)
		{
			getDxs(imdx, imdy, imdt, Im1, warpIm2, true);

			du.reset();
			dv.reset();

			for (int in_it = 0; in_it < opt.nInnerFixedPointIterations; in_it++)
			{
				Psi_1st.reset();

				T*& psiData = Psi_1st.data();
				T*& imdxData = imdx.data();
				T*& imdyData = imdy.data();
				T*& imdtData = imdt.data();
				T*& duData = du.data();
				T*& dvData = dv.data();
				T*& uData = u.data();
				T*& vData = v.data();

				for (int i = 0; i < nPixels; i++)
				{
					for (int c = 0; c < nChannels; c++)
					{
						int offset = i*nChannels + c;
						double temp = imdtData[offset] + imdxData[offset] * duData[i] + imdyData[offset] * dvData[i];

						temp *= temp;
						psiData[offset] = 1 / (2 * sqrt(temp + varepsilon_psi));
					}
				}

				imdxdx.Multiply(Psi_1st, imdx, imdx);
				imdxdy.Multiply(Psi_1st, imdx, imdy);
				imdydy.Multiply(Psi_1st, imdy, imdy);
				imdtdx.Multiply(Psi_1st, imdx, imdt);
				imdtdy.Multiply(Psi_1st, imdy, imdt);

				if (nChannels > 1)
				{
					imdxdx.collapse();
					imdxdy.collapse();
					imdydy.collapse();
					imdtdx.collapse();
					imdtdy.collapse();
				}

				T*& imdxdxData = imdxdx.data();
				T*& imdxdyData = imdxdy.data();
				T*& imdydyData = imdydy.data();
				T*& imdtdxData = imdtdx.data();
				T*& imdtdyData = imdtdy.data();
				const T*& z_uData = z_u.data();
				const T*& z_vData = z_v.data();


				Laplacian(laplace_u, u);
				Laplacian(laplace_v, v);

				T*& laplace_uData = laplace_u.data();
				T*& laplace_vData = laplace_v.data();

				double omega = opt.omegaForSOR;
				double alpha2 = opt.alpha*opt.alpha;
				double beta2 = opt.beta*opt.beta;

				/*Begin SOR*/

				for (int sor_it = 0; sor_it < opt.nSORIterations; sor_it++)
				{
					for (int i = 0; i < imHeight; i++)
					{
						for (int j = 0; j < imWidth; j++)
						{
							int offset = i*imWidth + j;
							double sigma1 = 0, sigma2 = 0, coeff = 0;
							double _weight;


							if (j > 0)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset - 1];
								sigma2 += _weight*dvData[offset - 1];
								coeff += _weight;
							}
							if (j < imWidth - 1)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset + 1];
								sigma2 += _weight*dvData[offset + 1];
								coeff += _weight;
							}
							if (i > 0)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset - imWidth];
								sigma2 += _weight*dvData[offset - imWidth];
								coeff += _weight;
							}
							if (i < imHeight - 1)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset + imWidth];
								sigma2 += _weight*dvData[offset + imWidth];
								coeff += _weight;
							}
							sigma1 *= alpha2;
							sigma2 *= alpha2;
							coeff *= alpha2;

							// compute u
							sigma1 += alpha2*(laplace_uData[offset]) - imdtdxData[offset] - imdxdyData[offset] * dvData[offset] - beta2*uData[offset] - 0.5*opt.lambda*(uData[offset] - z_uData[offset]);
							double coeff1 = coeff + imdxdxData[offset] + beta2 + 0.5*opt.lambda;
							duData[offset] = (1 - omega)*duData[offset] + omega / coeff1*sigma1;
							// compute v
							sigma2 += alpha2*(laplace_vData[offset]) - imdtdyData[offset] - imdxdyData[offset] * duData[offset] - beta2*vData[offset] - 0.5*opt.lambda*(vData[offset] - z_vData[offset]);
							double coeff2 = coeff + imdydyData[offset] + beta2 + 0.5*opt.lambda;
							dvData[offset] = (1 - omega)*dvData[offset] + omega / coeff2*sigma2;
						}
					}
				}
				/*End SOR*/
			}
			u.Addwith(du);
			v.Addwith(dv);

			warpFL(warpIm2, Im1, Im2, u, v, opt.useCubicWarping);
		}
	}

	template<class T>
	void ZQ_OpticalFlow::Proximal_G(ZQ_DImage<T>& u, ZQ_DImage<T>& v, const ZQ_DImage<T>& z_u, const ZQ_DImage<T>& z_v, const ZQ_OpticalFlowOptions& opt)
	{
		u.copyData(z_u);
		v.copyData(z_v);

		int width = u.width();
		int height = u.height();

		ZQ_PoissonSolver::SolveOpenPoissonSOR(u.data(), v.data(), width, height, opt.nPoissonIterations, opt.displayRunningInfo);
	}

	template<class T>
	void ZQ_OpticalFlow::Proximal_F2_First(ZQ_DImage<T>& u, ZQ_DImage<T>& v, const ZQ_DImage<T>& z_u, const ZQ_DImage<T>& z_v, const ZQ_DImage<T>& next_u, const ZQ_DImage<T>& next_v, const ZQ_OpticalFlowOptions& opt)
	{
		int imWidth = u.width();
		int imHeight = u.height();
		int nPixels = imWidth*imHeight;
		ZQ_DImage<T> warpU, warpV;

		ZQ_DImage<T> du(imWidth, imHeight), dv(imWidth, imHeight);

		double gamma = opt.gamma * opt.alpha * opt.alpha;

		for (int out_it = 0; out_it < opt.nAdvectFixedPointIterations; out_it++)
		{
			warpFL(warpU, u, next_u, u, v, opt.useCubicWarping);
			warpFL(warpV, v, next_v, u, v, opt.useCubicWarping);
			ZQ_PoissonSolver::SolveOpenPoissonSOR(warpU.data(), warpV.data(), imWidth, imHeight, opt.nPoissonIterations);

			T*& warpUData = warpU.data();
			T*& warpVData = warpV.data();

			T*& uData = u.data();
			T*& vData = v.data();
			const T*& z_uData = z_u.data();
			const T*& z_vData = z_v.data();

			for (int i = 0; i < nPixels; i++)
			{
				uData[i] = (gamma*warpUData[i] + 0.5*opt.lambda*z_uData[i]) / (gamma + 0.5*opt.lambda);
				vData[i] = (gamma*warpVData[i] + 0.5*opt.lambda*z_vData[i]) / (gamma + 0.5*opt.lambda);
			}
		}
	}

	template<class T>
	void ZQ_OpticalFlow::Proximal_F2_Last(ZQ_DImage<T>& u, ZQ_DImage<T>& v, const ZQ_DImage<T>& z_u, const ZQ_DImage<T>& z_v, const ZQ_DImage<T>& pre_u, const ZQ_DImage<T>& pre_v, const ZQ_OpticalFlowOptions& opt)
	{
		int imWidth = u.width();
		int imHeight = u.height();
		int nPixels = imWidth*imHeight;
		ZQ_DImage<T> warpU, warpV;

		ZQ_DImage<T> du(imWidth, imHeight), dv(imWidth, imHeight);

		double gamma = opt.gamma * opt.alpha * opt.alpha;

		for (int out_it = 0; out_it < opt.nAdvectFixedPointIterations; out_it++)
		{
			ZQ_DImage<T> tmp_u(u), tmp_v(v);
			tmp_u.Multiplywith(-1);
			tmp_v.Multiplywith(-1);

			warpFL(warpU, u, pre_u, tmp_u, tmp_v, opt.useCubicWarping);
			warpFL(warpV, v, pre_v, tmp_u, tmp_v, opt.useCubicWarping);

			ZQ_PoissonSolver::SolveOpenPoissonSOR(warpU.data(), warpV.data(), imWidth, imHeight, opt.nPoissonIterations);

			T*& warpUData = warpU.data();
			T*& warpVData = warpV.data();

			T*& uData = u.data();
			T*& vData = v.data();
			const T*& z_uData = z_u.data();
			const T*& z_vData = z_v.data();

			for (int i = 0; i < nPixels; i++)
			{
				uData[i] = (gamma*warpUData[i] + 0.5*opt.lambda*z_uData[i]) / (gamma + 0.5*opt.lambda);
				vData[i] = (gamma*warpVData[i] + 0.5*opt.lambda*z_vData[i]) / (gamma + 0.5*opt.lambda);
			}
		}
	}

	template<class T>
	void ZQ_OpticalFlow::Proximal_F2_Middle(ZQ_DImage<T>& u, ZQ_DImage<T>& v, const ZQ_DImage<T>& z_u, const ZQ_DImage<T>& z_v, const ZQ_DImage<T>& pre_u, const ZQ_DImage<T>& pre_v, const ZQ_DImage<T>& next_u, const ZQ_DImage<T>& next_v, const ZQ_OpticalFlowOptions& opt)
	{
		int imWidth = u.width();
		int imHeight = u.height();
		int nPixels = imWidth*imHeight;
		ZQ_DImage<T> warpU_pre, warpV_pre;
		ZQ_DImage<T> warpU_nex, warpV_nex;

		double gamma = opt.gamma * opt.alpha * opt.alpha;

		for (int out_it = 0; out_it < opt.nAdvectFixedPointIterations; out_it++)
		{
			ZQ_DImage<T> tmp_u(u), tmp_v(v);
			tmp_u.Multiplywith(-1);
			tmp_v.Multiplywith(-1);

			warpFL(warpU_pre, u, pre_u, tmp_u, tmp_v, opt.useCubicWarping);
			warpFL(warpV_pre, v, pre_v, tmp_u, tmp_v, opt.useCubicWarping);

			warpFL(warpU_nex, u, next_u, u, v, opt.useCubicWarping);
			warpFL(warpV_nex, v, next_v, u, v, opt.useCubicWarping);

			ZQ_DImage<T> warpU_sum, warpV_sum;
			warpU_sum.Add(warpU_pre, warpU_nex);
			warpV_sum.Add(warpV_pre, warpV_nex);

			ZQ_PoissonSolver::SolveOpenPoissonSOR(warpU_sum.data(), warpV_sum.data(), imWidth, imHeight, opt.nPoissonIterations);

			T*& warpU_sumData = warpU_sum.data();
			T*& warpV_sumData = warpV_sum.data();

			T*& uData = u.data();
			T*& vData = v.data();
			const T*& z_uData = z_u.data();
			const T*& z_vData = z_v.data();

			for (int i = 0; i < nPixels; i++)
			{
				uData[i] = (gamma*warpU_sumData[i] + 0.5*opt.lambda*z_uData[i]) / (2 * gamma + 0.5*opt.lambda);
				vData[i] = (gamma*warpV_sumData[i] + 0.5*opt.lambda*z_vData[i]) / (2 * gamma + 0.5*opt.lambda);
			}
		}
	}
}


#endif