#ifndef _ZQ_OPTICAL_FLOW_3D_H_
#define _ZQ_OPTICAL_FLOW_3D_H_
#pragma once

#include "ZQ_DoubleImage3D.h"
#include "ZQ_PoissonSolver3D.h"
#include "ZQ_ImageProcessing3D.h"
#include "ZQ_GaussianPyramid3D.h"
#include "ZQ_OpticalFlowOptions.h"
#include <math.h>
#include <stdlib.h>
#include <vector>


namespace ZQ
{  
	class ZQ_OpticalFlow3D
	{
	public:
		ZQ_OpticalFlow3D(void){}
		~ZQ_OpticalFlow3D(void){}


		/**********************************    functions for users     *******************************/
	public:
		template<class T>
		static void Coarse2Fine_HS_L2(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Coarse2Fine_HS_L1(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt);
		
		template<class T>
		static void Coarse2Fine_HS_DL1(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Coarse2Fine_ADMM_L2(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt);
		
		template<class T>
		static void Coarse2Fine_ADMM_DL1(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Coarse2Fine_OneDir_Inc_L2(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt);
		
		template<class T>
		static void Coarse2Fine_OneDir_Inc_DL1(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt);
		
		template<class T>
		static void Coarse2Fine_OneDir_Dec_L2(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt);
		
		template<class T>
		static void Coarse2Fine_OneDir_Dec_DL1(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt);
		
		template<class T>
		static void Coarse2Fine_TwoDir_Inc_L2(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt);
		
		template<class T>
		static void Coarse2Fine_TwoDir_Inc_DL1(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt);
		
		template<class T>
		static void Coarse2Fine_TwoDir_Dec_L2(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt);
		
		template<class T>
		static void Coarse2Fine_TwoDir_Dec_DL1(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt);


		/************************* be careful to use this functions *****************************/
	public:
		template<class T>
		static void OneResolution_HS_L2(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_HS_L1(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_HS_DL1(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_ADMM_L2(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_ADMM_DL1(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_OneDir_Inc_L2(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_OneDir_Inc_DL1(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_OneDir_Dec_L2(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_OneDir_Dec_DL1(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_TwoDir_Inc_L2(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_TwoDir_Inc_DL1(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_TwoDir_Dec_L2(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void OneResolution_TwoDir_Dec_DL1(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt);
	

		/******************** if you are very familiar with this code, you can inherit this class and access these protected functions *********************************/
	protected:

		template<class T>
		static void getDxs(ZQ_DImage3D<T>& imdx, ZQ_DImage3D<T>& imdy, ZQ_DImage3D<T>& imdz, ZQ_DImage3D<T>& imdt, const ZQ_DImage3D<T>& im1, const ZQ_DImage3D<T>& im2, bool isSmooth = true);

		template<class T>
		static void warpFL(ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_DImage3D<T>& u, const ZQ_DImage3D<T>& v, const ZQ_DImage3D<T>& w, bool isBicubic = false);

		template<class T>
		static void Laplacian(ZQ_DImage3D<T>& output, const ZQ_DImage3D<T>& input);

		template<class T>
		static void Laplacian(ZQ_DImage3D<T>& output, const ZQ_DImage3D<T>& input, const ZQ_DImage3D<T>& weight);

		// function to convert image to features
		template<class T>
		static void im2feature(ZQ_DImage3D<T>& imfeature, const ZQ_DImage3D<T>& im, bool isSmooth, const ZQ_OpticalFlowOptions& opt);

	
		/**********************************  for ADMM frame  ********************8*******************/
		template<class T>
		static void ADMM_F_G(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt,
			void (*funcF)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, ZQ_DImage3D<T>& /*warpIm2*/, const ZQ_DImage3D<T>& /*Im1*/, const ZQ_DImage3D<T>& /*Im2*/, const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_OpticalFlowOptions& /*opt*/),
			void (*funcG)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_OpticalFlowOptions& /*opt*/));

		template<class T>
		static void ADMM_F1_F2_G_first(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_DImage3D<T>& next_u, const ZQ_DImage3D<T>& next_v, const ZQ_DImage3D<T>& next_w, const ZQ_OpticalFlowOptions& opt,
			void (*funcF1)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, ZQ_DImage3D<T>& /*warpIm2*/, const ZQ_DImage3D<T>& /*Im1*/, const ZQ_DImage3D<T>& /*Im2*/,  const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_OpticalFlowOptions& /*opt*/),
			void (*funcF2)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_DImage3D<T>& /*next_u*/, const ZQ_DImage3D<T>& /*next_v*/, const ZQ_DImage3D<T>& /*next_w*/, const ZQ_OpticalFlowOptions& /*opt*/),
			void (*funcG)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_OpticalFlowOptions& /*opt*/));

		template<class T>
		static void ADMM_F1_F2_G_last(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_DImage3D<T>& pre_u, const ZQ_DImage3D<T>& pre_v, const ZQ_DImage3D<T>& pre_w, const ZQ_OpticalFlowOptions& opt,
			void (*funcF1)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, ZQ_DImage3D<T>& /*warpIm2*/, const ZQ_DImage3D<T>& /*Im1*/, const ZQ_DImage3D<T>& /*Im2*/,  const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_OpticalFlowOptions& /*opt*/),
			void (*funcF2)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_DImage3D<T>& /*pre_u*/, const ZQ_DImage3D<T>& /*pre_v*/, const ZQ_DImage3D<T>& /*pre_w*/, const ZQ_OpticalFlowOptions& /*opt*/),
			void (*funcG)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_OpticalFlowOptions& /*opt*/));


		template<class T>
		static void ADMM_F1_F2_G_middle(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_DImage3D<T>& pre_u, const ZQ_DImage3D<T>& pre_v, const ZQ_DImage3D<T>& pre_w, const ZQ_DImage3D<T>& next_u, const ZQ_DImage3D<T>& next_v, const ZQ_DImage3D<T>& next_w, const ZQ_OpticalFlowOptions& opt,
			void (*funcF1)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, ZQ_DImage3D<T>& /*warpIm2*/, const ZQ_DImage3D<T>& /*Im1*/, const ZQ_DImage3D<T>& /*Im2*/,  const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_OpticalFlowOptions& /*opt*/),
			void (*funcF2)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_DImage3D<T>& /*pre_u*/, const ZQ_DImage3D<T>& /*pre_v*/, const ZQ_DImage3D<T>& /*pre_w*/, const ZQ_DImage3D<T>& /*next_u*/, const ZQ_DImage3D<T>& /*next_v*/, const ZQ_DImage3D<T>& /*next_w*/, const ZQ_OpticalFlowOptions& /*opt*/),
			void (*funcG)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_OpticalFlowOptions& /*opt*/));


		/**********************     Proximal functions   ***********************/
		template<class T>
		static void Proximal_F_L2(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_DImage3D<T>& z_u, const ZQ_DImage3D<T>& z_v, const ZQ_DImage3D<T>& z_w, const ZQ_OpticalFlowOptions& opt); 

		template<class T>
		static void Proximal_F_DL1(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_DImage3D<T>& z_u, const ZQ_DImage3D<T>& z_v, const ZQ_DImage3D<T>& z_w, const ZQ_OpticalFlowOptions& opt); 

		template<class T>
		static void Proximal_G(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, const ZQ_DImage3D<T>& z_u, const ZQ_DImage3D<T>& z_v, const ZQ_DImage3D<T>& z_w, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Proximal_F2_First(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, const ZQ_DImage3D<T>& z_u, const ZQ_DImage3D<T>& z_v, const ZQ_DImage3D<T>& z_w, const ZQ_DImage3D<T>& next_u, const ZQ_DImage3D<T>& next_v, const ZQ_DImage3D<T>& next_w, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Proximal_F2_Last(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, const ZQ_DImage3D<T>& z_u, const ZQ_DImage3D<T>& z_v, const ZQ_DImage3D<T>& z_w, const ZQ_DImage3D<T>& pre_u, const ZQ_DImage3D<T>& pre_v, const ZQ_DImage3D<T>& pre_w, const ZQ_OpticalFlowOptions& opt);

		template<class T>
		static void Proximal_F2_Middle(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, const ZQ_DImage3D<T>& z_u, const ZQ_DImage3D<T>& z_v, const ZQ_DImage3D<T>& z_w, const ZQ_DImage3D<T>& pre_u, const ZQ_DImage3D<T>& pre_v, const ZQ_DImage3D<T>& pre_w, const ZQ_DImage3D<T>& next_u, const ZQ_DImage3D<T>& next_v, const ZQ_DImage3D<T>& next_w, const ZQ_OpticalFlowOptions& opt);

	};


	/*********************************************************************************/
	/********************************** definitions **********************************/
	/*********************************************************************************/

	/**************************  functions for users *****************************/
	template<class T>
	void ZQ_OpticalFlow3D::Coarse2Fine_HS_L2(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_GaussianPyramid3D<T> GPyramid1;
		ZQ_GaussianPyramid3D<T> GPyramid2;
		if(opt.displayRunningInfo)
		{
			printf("Constructing pyramid...");
		}
		
		double ratio = 
		GPyramid1.ConstructPyramid(Im1,opt.ratioForPyramid,opt.minWidthForPyramid);
		GPyramid2.ConstructPyramid(Im2,opt.ratioForPyramid,opt.minWidthForPyramid);
			
		if(opt.displayRunningInfo)
			printf("done!\n");

		ZQ_DImage3D<T> Image1,Image2;

		for(int k = GPyramid1.nlevels()-1;k >= 0;k--)
		{
			if(opt.displayRunningInfo)
				printf("Pyramid level %d \n",k);

			int width = GPyramid1.Image(k).width();
			int height = GPyramid1.Image(k).height();
			int depth = GPyramid1.Image(k).depth();

			bool isSmooth = true;
			if(k == 0)
				isSmooth = false;

			im2feature(Image1,GPyramid1.Image(k),isSmooth,opt);
			im2feature(Image2,GPyramid2.Image(k),isSmooth,opt);

			if(k == GPyramid1.nlevels()-1)
			{
				u.allocate(width,height,depth);
				v.allocate(width,height,depth);
				w.allocate(width,height,depth);
				warpIm2.copyData(Image2);

			}
			else
			{
				u.imresize(width,height,depth);
				u.Multiplywith(1.0/ratio);
				v.imresize(width,height,depth);
				v.Multiplywith(1.0/ratio);
				w.imresize(width,height,depth);
				w.Multiplywith(1.0/ratio);
				warpFL(warpIm2,Image1,Image2,u,v,w,opt.useCubicWarping);

			}

			OneResolution_HS_L2(u,v,w,warpIm2,Image1,Image2,opt);
		}
	}

	template<class T>
	void ZQ_OpticalFlow3D::Coarse2Fine_HS_L1(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_GaussianPyramid3D<T> GPyramid1;
		ZQ_GaussianPyramid3D<T> GPyramid2;
		if(opt.displayRunningInfo)
		{
			printf("Constructing pyramid...");
		}

		double ratio = 
		GPyramid1.ConstructPyramid(Im1,opt.ratioForPyramid,opt.minWidthForPyramid);
		GPyramid2.ConstructPyramid(Im2,opt.ratioForPyramid,opt.minWidthForPyramid);

		if(opt.displayRunningInfo)
			printf("done!\n");

		ZQ_DImage3D<T> Image1,Image2;

		for(int k = GPyramid1.nlevels()-1;k >= 0;k--)
		{
			if(opt.displayRunningInfo)
				printf("Pyramid level %d \n",k);

			int width = GPyramid1.Image(k).width();
			int height = GPyramid1.Image(k).height();
			int depth = GPyramid1.Image(k).depth();

			bool isSmooth = true;
			if(k == 0)
				isSmooth = false;

			im2feature(Image1,GPyramid1.Image(k),isSmooth,opt);
			im2feature(Image2,GPyramid2.Image(k),isSmooth,opt);

			if(k == GPyramid1.nlevels()-1)
			{
				u.allocate(width,height,depth);
				v.allocate(width,height,depth);
				w.allocate(width,height,depth);
				warpIm2.copyData(Image2);

			}
			else
			{
				u.imresize(width,height,depth);
				u.Multiplywith(1.0/ratio);
				v.imresize(width,height,depth);
				v.Multiplywith(1.0/ratio);
				w.imresize(width,height,depth);
				w.Multiplywith(1.0/ratio);
				warpFL(warpIm2,Image1,Image2,u,v,w,opt.useCubicWarping);

			}

			OneResolution_HS_L1(u,v,w,warpIm2,Image1,Image2,opt);
		}
	}

	template<class T>
	void ZQ_OpticalFlow3D::Coarse2Fine_HS_DL1(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_GaussianPyramid3D<T> GPyramid1;
		ZQ_GaussianPyramid3D<T> GPyramid2;
		if(opt.displayRunningInfo)
		{
			printf("Constructing pyramid...");
		}

		double ratio = 
		GPyramid1.ConstructPyramid(Im1,opt.ratioForPyramid,opt.minWidthForPyramid);
		GPyramid2.ConstructPyramid(Im2,opt.ratioForPyramid,opt.minWidthForPyramid);

		if(opt.displayRunningInfo)
			printf("done!\n");

		ZQ_DImage3D<T> Image1,Image2;

		for(int k = GPyramid1.nlevels()-1;k >= 0;k--)
		{
			if(opt.displayRunningInfo)
				printf("Pyramid level %d \n",k);

			int width = GPyramid1.Image(k).width();
			int height = GPyramid1.Image(k).height();
			int depth = GPyramid1.Image(k).depth();

			bool isSmooth = true;
			if(k == 0)
				isSmooth = false;

			im2feature(Image1,GPyramid1.Image(k),isSmooth,opt);
			im2feature(Image2,GPyramid2.Image(k),isSmooth,opt);

			if(k == GPyramid1.nlevels()-1)
			{
				u.allocate(width,height,depth);
				v.allocate(width,height,depth);
				w.allocate(width,height,depth);
				warpIm2.copyData(Image2);

			}
			else
			{
				u.imresize(width,height,depth);
				u.Multiplywith(1.0/ratio);
				v.imresize(width,height,depth);
				v.Multiplywith(1.0/ratio);
				w.imresize(width,height,depth);
				w.Multiplywith(1.0/ratio);
				warpFL(warpIm2,Image1,Image2,u,v,w,opt.useCubicWarping);

			}

			OneResolution_HS_DL1(u,v,w,warpIm2,Image1,Image2,opt);
		}
	}

	template<class T>
	void ZQ_OpticalFlow3D::Coarse2Fine_ADMM_L2(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_GaussianPyramid3D<T> GPyramid1;
		ZQ_GaussianPyramid3D<T> GPyramid2;
		
		if(opt.displayRunningInfo)
			printf("Constructing pyramid...");
		
		double ratio = 
		GPyramid1.ConstructPyramid(Im1,opt.ratioForPyramid,opt.minWidthForPyramid);
		GPyramid2.ConstructPyramid(Im2,opt.ratioForPyramid,opt.minWidthForPyramid);
		
		if(opt.displayRunningInfo)
			printf("done!\n");

		ZQ_DImage3D<T> Image1,Image2;

		for(int k = GPyramid1.nlevels()-1;k >= 0;k--)
		{
			if(opt.displayRunningInfo)
				printf("Pyramid level %d \n",k);

			int width = GPyramid1.Image(k).width();
			int height = GPyramid1.Image(k).height();
			int depth = GPyramid1.Image(k).depth();

			bool isSmooth = true;
			if(k == 0)
				isSmooth = false;

			im2feature(Image1,GPyramid1.Image(k),isSmooth,opt);
			im2feature(Image2,GPyramid2.Image(k),isSmooth,opt);

			if(k == GPyramid1.nlevels()-1)
			{
				u.allocate(width,height,depth);
				v.allocate(width,height,depth);
				w.allocate(width,height,depth);
				warpIm2.copyData(Image2);

				OneResolution_HS_L2(u,v,w,warpIm2,Image1,Image2,opt);
			}
			else
			{
				u.imresize(width,height,depth);
				u.Multiplywith(1.0/ratio);
				v.imresize(width,height,depth);
				v.Multiplywith(1.0/ratio);
				w.imresize(width,height,depth);
				w.Multiplywith(1.0/ratio);
				warpFL(warpIm2,Image1,Image2,u,v,w,opt.useCubicWarping);

			}

			ADMM_F_G(u,v,w,warpIm2,Image1,Image2,opt,Proximal_F_L2<T>,Proximal_G<T>);
		}
	}

	template<class T>
	void ZQ_OpticalFlow3D::Coarse2Fine_ADMM_DL1(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_GaussianPyramid3D<T> GPyramid1;
		ZQ_GaussianPyramid3D<T> GPyramid2;

		if(opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio = 
		GPyramid1.ConstructPyramid(Im1,opt.ratioForPyramid,opt.minWidthForPyramid);
		GPyramid2.ConstructPyramid(Im2,opt.ratioForPyramid,opt.minWidthForPyramid);

		if(opt.displayRunningInfo)
			printf("done!\n");

		ZQ_DImage3D<T> Image1,Image2;

		for(int k = GPyramid1.nlevels()-1;k >= 0;k--)
		{
			if(opt.displayRunningInfo)
				printf("Pyramid level %d \n",k);

			int width = GPyramid1.Image(k).width();
			int height = GPyramid1.Image(k).height();
			int depth = GPyramid1.Image(k).depth();

			bool isSmooth = true;
			if(k == 0)
				isSmooth = false;

			im2feature(Image1,GPyramid1.Image(k),isSmooth,opt);
			im2feature(Image2,GPyramid2.Image(k),isSmooth,opt);

			if(k == GPyramid1.nlevels()-1)
			{
				u.allocate(width,height,depth);
				v.allocate(width,height,depth);
				w.allocate(width,height,depth);
				warpIm2.copyData(Image2);

				OneResolution_HS_DL1(u,v,w,warpIm2,Image1,Image2,opt);
			}
			else
			{
				u.imresize(width,height,depth);
				u.Multiplywith(1.0/ratio);
				v.imresize(width,height,depth);
				v.Multiplywith(1.0/ratio);
				w.imresize(width,height,depth);
				w.Multiplywith(1.0/ratio);
				warpFL(warpIm2,Image1,Image2,u,v,w,opt.useCubicWarping);
			}

			ADMM_F_G(u,v,w,warpIm2,Image1,Image2,opt,Proximal_F_DL1<T>,Proximal_G<T>);
		}
	}


	template<class T>
	void ZQ_OpticalFlow3D::Coarse2Fine_OneDir_Inc_L2(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();

		if(u.size() != image_num-1)
		{
			u.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				u.push_back(tmp);
		}

		if(v.size() != image_num-1)
		{
			v.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				v.push_back(tmp);
		}

		if(w.size() != image_num-1)
		{
			w.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				w.push_back(tmp);
		}

		if(warpIm.size() != image_num-1)
		{
			warpIm.clear();
			ZQ_DImage3D<T> tmp;
			for (int i = 0;i < image_num-1;i++)
			{
				warpIm.push_back(tmp);
			}
		}

		/*  Constructing Pyramids */
		std::vector<ZQ_GaussianPyramid3D<T>> GPyramids(image_num);

		if(opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio = 0;
		for(int i = 0;i < image_num;i++)
			ratio = GPyramids[i].ConstructPyramid(Im[i],opt.ratioForPyramid,opt.minWidthForPyramid);

		if(opt.displayRunningInfo)
			printf("done!\n");

		/*   Handle images from low resolution to high resolution     */
		std::vector<ZQ_DImage3D<T>> Images(image_num);

		for(int k = GPyramids[0].nlevels()-1;k >= 0;k--)
		{
			if(opt.displayRunningInfo)
				printf("Pyramid level %d \n",k);

			int width = GPyramids[0].Image(k).width();
			int height = GPyramids[0].Image(k).height();
			int depth = GPyramids[0].Image(k).depth();

			bool isSmooth = true;
			if(k == 0)
				isSmooth = false;

			for(int i = 0;i < image_num;i++)
				im2feature(Images[i],GPyramids[i].Image(k),isSmooth,opt);

			if(k == GPyramids[0].nlevels()-1)
			{
				for(int i = 0;i < image_num-1;i++)
				{
					u[i].allocate(width,height,depth);
					v[i].allocate(width,height,depth);
					w[i].allocate(width,height,depth);
					warpIm[i].copyData(Images[i+1]);
					OneResolution_HS_L2(u[i],v[i],w[i],warpIm[i],Images[i],Images[i+1],opt);
				}
			}
			else
			{
				for(int i = 0;i < image_num-1;i++)
				{
					u[i].imresize(width,height,depth);
					u[i].Multiplywith(1.0/ratio);
					v[i].imresize(width,height,depth);
					v[i].Multiplywith(1.0/ratio);
					w[i].imresize(width,height,depth);
					w[i].Multiplywith(1.0/ratio);
					warpFL(warpIm[i],Images[i],Images[i+1],u[i],v[i],w[i],opt.useCubicWarping);
				}
			}

			OneResolution_OneDir_Inc_L2(u,v,w,warpIm,Images,opt);
		}
	}


	template<class T>
	void ZQ_OpticalFlow3D::Coarse2Fine_OneDir_Inc_DL1(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();

		if(u.size() != image_num-1)
		{
			u.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				u.push_back(tmp);
		}

		if(v.size() != image_num-1)
		{
			v.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				v.push_back(tmp);
		}

		if(w.size() != image_num-1)
		{
			w.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				w.push_back(tmp);
		}

		if(warpIm.size() != image_num-1)
		{
			warpIm.clear();
			ZQ_DImage3D<T> tmp;
			for (int i = 0;i < image_num-1;i++)
			{
				warpIm.push_back(tmp);
			}
		}

		/*  Constructing Pyramids */
		std::vector<ZQ_GaussianPyramid3D<T>> GPyramids(image_num);

		if(opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio = 0;
		for(int i = 0;i < image_num;i++)
			ratio = GPyramids[i].ConstructPyramid(Im[i],opt.ratioForPyramid,opt.minWidthForPyramid);

		if(opt.displayRunningInfo)
			printf("done!\n");

		/*   Handle images from low resolution to high resolution     */
		std::vector<ZQ_DImage3D<T>> Images(image_num);

		for(int k = GPyramids[0].nlevels()-1;k >= 0;k--)
		{
			if(opt.displayRunningInfo)
				printf("Pyramid level %d \n",k);

			int width = GPyramids[0].Image(k).width();
			int height = GPyramids[0].Image(k).height();
			int depth = GPyramids[0].Image(k).depth();

			bool isSmooth = true;
			if(k == 0)
				isSmooth = false;

			for(int i = 0;i < image_num;i++)
				im2feature(Images[i],GPyramids[i].Image(k),isSmooth,opt);

			if(k == GPyramids[0].nlevels()-1)
			{
				for(int i = 0;i < image_num-1;i++)
				{
					u[i].allocate(width,height,depth);
					v[i].allocate(width,height,depth);
					w[i].allocate(width,height,depth);
					warpIm[i].copyData(Images[i+1]);
					OneResolution_HS_DL1(u[i],v[i],w[i],warpIm[i],Images[i],Images[i+1],opt);
				}
			}
			else
			{
				for(int i = 0;i < image_num-1;i++)
				{
					u[i].imresize(width,height,depth);
					u[i].Multiplywith(1.0/ratio);
					v[i].imresize(width,height,depth);
					v[i].Multiplywith(1.0/ratio);
					w[i].imresize(width,height,depth);
					w[i].Multiplywith(1.0/ratio);
					warpFL(warpIm[i],Images[i],Images[i+1],u[i],v[i],w[i],opt.useCubicWarping);
				}
			}

			OneResolution_OneDir_Inc_DL1(u,v,w,warpIm,Images,opt);
		}
	}

	template<class T>
	void ZQ_OpticalFlow3D::Coarse2Fine_OneDir_Dec_L2(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();

		if(u.size() != image_num-1)
		{
			u.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				u.push_back(tmp);
		}

		if(v.size() != image_num-1)
		{
			v.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				v.push_back(tmp);
		}

		if(w.size() != image_num-1)
		{
			w.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				w.push_back(tmp);
		}

		if(warpIm.size() != image_num-1)
		{
			warpIm.clear();
			ZQ_DImage3D<T> tmp;
			for (int i = 0;i < image_num-1;i++)
			{
				warpIm.push_back(tmp);
			}
		}

		/*  Constructing Pyramids */
		std::vector<ZQ_GaussianPyramid3D<T>> GPyramids(image_num);

		if(opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio = 0;
		for(int i = 0;i < image_num;i++)
			ratio = GPyramids[i].ConstructPyramid(Im[i],opt.ratioForPyramid,opt.minWidthForPyramid);

		if(opt.displayRunningInfo)
			printf("done!\n");

		/*   Handle images from low resolution to high resolution     */
		std::vector<ZQ_DImage3D<T>> Images(image_num);

		for(int k = GPyramids[0].nlevels()-1;k >= 0;k--)
		{
			if(opt.displayRunningInfo)
				printf("Pyramid level %d \n",k);

			int width = GPyramids[0].Image(k).width();
			int height = GPyramids[0].Image(k).height();
			int depth = GPyramids[0].Image(k).depth();

			bool isSmooth = true;
			if(k == 0)
				isSmooth = false;

			for(int i = 0;i < image_num;i++)
				im2feature(Images[i],GPyramids[i].Image(k),isSmooth,opt);

			if(k == GPyramids[0].nlevels()-1)
			{
				for(int i = 0;i < image_num-1;i++)
				{
					u[i].allocate(width,height,depth);
					v[i].allocate(width,height,depth);
					w[i].allocate(width,height,depth);
					warpIm[i].copyData(Images[i+1]);
					OneResolution_HS_L2(u[i],v[i],w[i],warpIm[i],Images[i],Images[i+1],opt);
				}
			}
			else
			{
				for(int i = 0;i < image_num-1;i++)
				{
					u[i].imresize(width,height,depth);
					u[i].Multiplywith(1.0/ratio);
					v[i].imresize(width,height,depth);
					v[i].Multiplywith(1.0/ratio);
					w[i].imresize(width,height,depth);
					w[i].Multiplywith(1.0/ratio);
					warpFL(warpIm[i],Images[i],Images[i+1],u[i],v[i],w[i],opt.useCubicWarping);
				}
			}

			OneResolution_OneDir_Dec_L2(u,v,w,warpIm,Images,opt);
		}
	}


	template<class T>
	void ZQ_OpticalFlow3D::Coarse2Fine_OneDir_Dec_DL1(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();

		if(u.size() != image_num-1)
		{
			u.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				u.push_back(tmp);
		}

		if(v.size() != image_num-1)
		{
			v.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				v.push_back(tmp);
		}

		if(w.size() != image_num-1)
		{
			w.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				w.push_back(tmp);
		}

		if(warpIm.size() != image_num-1)
		{
			warpIm.clear();
			ZQ_DImage3D<T> tmp;
			for (int i = 0;i < image_num-1;i++)
			{
				warpIm.push_back(tmp);
			}
		}

		/*  Constructing Pyramids */
		std::vector<ZQ_GaussianPyramid3D<T>> GPyramids(image_num);

		if(opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio = 0;
		for(int i = 0;i < image_num;i++)
			ratio = GPyramids[i].ConstructPyramid(Im[i],opt.ratioForPyramid,opt.minWidthForPyramid);

		if(opt.displayRunningInfo)
			printf("done!\n");

		/*   Handle images from low resolution to high resolution     */
		std::vector<ZQ_DImage3D<T>> Images(image_num);

		for(int k = GPyramids[0].nlevels()-1;k >= 0;k--)
		{
			if(opt.displayRunningInfo)
				printf("Pyramid level %d \n",k);

			int width = GPyramids[0].Image(k).width();
			int height = GPyramids[0].Image(k).height();
			int depth = GPyramids[0].Image(k).depth();

			bool isSmooth = true;
			if(k == 0)
				isSmooth = false;

			for(int i = 0;i < image_num;i++)
				im2feature(Images[i],GPyramids[i].Image(k),isSmooth,opt);

			if(k == GPyramids[0].nlevels()-1)
			{
				for(int i = 0;i < image_num-1;i++)
				{
					u[i].allocate(width,height,depth);
					v[i].allocate(width,height,depth);
					w[i].allocate(width,height,depth);
					warpIm[i].copyData(Images[i+1]);
					OneResolution_HS_DL1(u[i],v[i],w[i],warpIm[i],Images[i],Images[i+1],opt);
				}
			}
			else
			{
				for(int i = 0;i < image_num-1;i++)
				{
					u[i].imresize(width,height,depth);
					u[i].Multiplywith(1.0/ratio);
					v[i].imresize(width,height,depth);
					v[i].Multiplywith(1.0/ratio);
					w[i].imresize(width,height,depth);
					w[i].Multiplywith(1.0/ratio);
					warpFL(warpIm[i],Images[i],Images[i+1],u[i],v[i],w[i],opt.useCubicWarping);
				}
			}

			OneResolution_OneDir_Dec_DL1(u,v,w,warpIm,Images,opt);
		}
	}

	template<class T>
	void ZQ_OpticalFlow3D::Coarse2Fine_TwoDir_Inc_L2(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();

		if(u.size() != image_num-1)
		{
			u.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				u.push_back(tmp);
		}

		if(v.size() != image_num-1)
		{
			v.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				v.push_back(tmp);
		}

		if(w.size() != image_num-1)
		{
			w.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				w.push_back(tmp);
		}

		if(warpIm.size() != image_num-1)
		{
			warpIm.clear();
			ZQ_DImage3D<T> tmp;
			for (int i = 0;i < image_num-1;i++)
			{
				warpIm.push_back(tmp);
			}
		}

		/*  Constructing Pyramids */
		std::vector<ZQ_GaussianPyramid3D<T>> GPyramids(image_num);

		if(opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio = 0;
		for(int i = 0;i < image_num;i++)
			ratio = GPyramids[i].ConstructPyramid(Im[i],opt.ratioForPyramid,opt.minWidthForPyramid);

		if(opt.displayRunningInfo)
			printf("done!\n");

		/*   Handle images from low resolution to high resolution     */
		std::vector<ZQ_DImage3D<T>> Images(image_num);

		for(int k = GPyramids[0].nlevels()-1;k >= 0;k--)
		{
			if(opt.displayRunningInfo)
				printf("Pyramid level %d \n",k);

			int width = GPyramids[0].Image(k).width();
			int height = GPyramids[0].Image(k).height();
			int depth = GPyramids[0].Image(k).depth();

			bool isSmooth = true;
			if(k == 0)
				isSmooth = false;

			for(int i = 0;i < image_num;i++)
				im2feature(Images[i],GPyramids[i].Image(k),isSmooth,opt);

			if(k == GPyramids[0].nlevels()-1)
			{
				for(int i = 0;i < image_num-1;i++)
				{
					u[i].allocate(width,height,depth);
					v[i].allocate(width,height,depth);
					w[i].allocate(width,height,depth);
					warpIm[i].copyData(Images[i+1]);
					OneResolution_HS_L2(u[i],v[i],w[i],warpIm[i],Images[i],Images[i+1],opt);
				}
			}
			else
			{
				for(int i = 0;i < image_num-1;i++)
				{
					u[i].imresize(width,height,depth);
					u[i].Multiplywith(1.0/ratio);
					v[i].imresize(width,height,depth);
					v[i].Multiplywith(1.0/ratio);
					w[i].imresize(width,height,depth);
					w[i].Multiplywith(1.0/ratio);
					warpFL(warpIm[i],Images[i],Images[i+1],u[i],v[i],w[i],opt.useCubicWarping);
				}
			}

			OneResolution_TwoDir_Inc_L2(u,v,w,warpIm,Images,opt);
		}
	}

	template<class T>
	void ZQ_OpticalFlow3D::Coarse2Fine_TwoDir_Inc_DL1(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();

		if(u.size() != image_num-1)
		{
			u.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				u.push_back(tmp);
		}

		if(v.size() != image_num-1)
		{
			v.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				v.push_back(tmp);
		}

		if(w.size() != image_num-1)
		{
			w.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				w.push_back(tmp);
		}

		if(warpIm.size() != image_num-1)
		{
			warpIm.clear();
			ZQ_DImage3D<T> tmp;
			for (int i = 0;i < image_num-1;i++)
			{
				warpIm.push_back(tmp);
			}
		}

		/*  Constructing Pyramids */
		std::vector<ZQ_GaussianPyramid3D<T>> GPyramids(image_num);

		if(opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio = 0;
		for(int i = 0;i < image_num;i++)
			ratio = GPyramids[i].ConstructPyramid(Im[i],opt.ratioForPyramid,opt.minWidthForPyramid);

		if(opt.displayRunningInfo)
			printf("done!\n");

		/*   Handle images from low resolution to high resolution     */
		std::vector<ZQ_DImage3D<T>> Images(image_num);

		for(int k = GPyramids[0].nlevels()-1;k >= 0;k--)
		{
			if(opt.displayRunningInfo)
				printf("Pyramid level %d \n",k);

			int width = GPyramids[0].Image(k).width();
			int height = GPyramids[0].Image(k).height();
			int depth = GPyramids[0].Image(k).depth();

			bool isSmooth = true;
			if(k == 0)
				isSmooth = false;

			for(int i = 0;i < image_num;i++)
				im2feature(Images[i],GPyramids[i].Image(k),isSmooth,opt);

			if(k == GPyramids[0].nlevels()-1)
			{
				for(int i = 0;i < image_num-1;i++)
				{
					u[i].allocate(width,height,depth);
					v[i].allocate(width,height,depth);
					w[i].allocate(width,height,depth);
					warpIm[i].copyData(Images[i+1]);
					OneResolution_HS_DL1(u[i],v[i],w[i],warpIm[i],Images[i],Images[i+1],opt);
				}
			}
			else
			{
				for(int i = 0;i < image_num-1;i++)
				{
					u[i].imresize(width,height,depth);
					u[i].Multiplywith(1.0/ratio);
					v[i].imresize(width,height,depth);
					v[i].Multiplywith(1.0/ratio);
					w[i].imresize(width,height,depth);
					w[i].Multiplywith(1.0/ratio);
					warpFL(warpIm[i],Images[i],Images[i+1],u[i],v[i],w[i],opt.useCubicWarping);
				}
			}

			OneResolution_TwoDir_Inc_DL1(u,v,w,warpIm,Images,opt);
		}
	}

	template<class T>
	void ZQ_OpticalFlow3D::Coarse2Fine_TwoDir_Dec_L2(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();

		if(u.size() != image_num-1)
		{
			u.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				u.push_back(tmp);
		}

		if(v.size() != image_num-1)
		{
			v.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				v.push_back(tmp);
		}

		if(w.size() != image_num-1)
		{
			w.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				w.push_back(tmp);
		}

		if(warpIm.size() != image_num-1)
		{
			warpIm.clear();
			ZQ_DImage3D<T> tmp;
			for (int i = 0;i < image_num-1;i++)
			{
				warpIm.push_back(tmp);
			}
		}

		/*  Constructing Pyramids */
		std::vector<ZQ_GaussianPyramid3D<T>> GPyramids(image_num);

		if(opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio = 0;
		for(int i = 0;i < image_num;i++)
			ratio = GPyramids[i].ConstructPyramid(Im[i],opt.ratioForPyramid,opt.minWidthForPyramid);

		if(opt.displayRunningInfo)
			printf("done!\n");

		/*   Handle images from low resolution to high resolution     */
		std::vector<ZQ_DImage3D<T>> Images(image_num);

		for(int k = GPyramids[0].nlevels()-1;k >= 0;k--)
		{
			if(opt.displayRunningInfo)
				printf("Pyramid level %d \n",k);

			int width = GPyramids[0].Image(k).width();
			int height = GPyramids[0].Image(k).height();
			int depth = GPyramids[0].Image(k).depth();

			bool isSmooth = true;
			if(k == 0)
				isSmooth = false;

			for(int i = 0;i < image_num;i++)
				im2feature(Images[i],GPyramids[i].Image(k),isSmooth,opt);

			if(k == GPyramids[0].nlevels()-1)
			{
				for(int i = 0;i < image_num-1;i++)
				{
					u[i].allocate(width,height,depth);
					v[i].allocate(width,height,depth);
					w[i].allocate(width,height,depth);
					warpIm[i].copyData(Images[i+1]);
					OneResolution_HS_L2(u[i],v[i],w[i],warpIm[i],Images[i],Images[i+1],opt);
				}
			}
			else
			{
				for(int i = 0;i < image_num-1;i++)
				{
					u[i].imresize(width,height,depth);
					u[i].Multiplywith(1.0/ratio);
					v[i].imresize(width,height,depth);
					v[i].Multiplywith(1.0/ratio);
					w[i].imresize(width,height,depth);
					w[i].Multiplywith(1.0/ratio);
					warpFL(warpIm[i],Images[i],Images[i+1],u[i],v[i],w[i],opt.useCubicWarping);
				}
			}

			OneResolution_TwoDir_Dec_L2(u,v,w,warpIm,Images,opt);
		}
	}


	template<class T>
	void ZQ_OpticalFlow3D::Coarse2Fine_TwoDir_Dec_DL1(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();

		if(u.size() != image_num-1)
		{
			u.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				u.push_back(tmp);
		}

		if(v.size() != image_num-1)
		{
			v.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				v.push_back(tmp);
		}

		if(w.size() != image_num-1)
		{
			w.clear();
			ZQ_DImage3D<T> tmp;
			for(int i = 0;i < image_num-1;i++)
				w.push_back(tmp);
		}

		if(warpIm.size() != image_num-1)
		{
			warpIm.clear();
			ZQ_DImage3D<T> tmp;
			for (int i = 0;i < image_num-1;i++)
			{
				warpIm.push_back(tmp);
			}
		}

		/*  Constructing Pyramids */
		std::vector<ZQ_GaussianPyramid3D<T>> GPyramids(image_num);

		if(opt.displayRunningInfo)
			printf("Constructing pyramid...");

		double ratio = 0;
		for(int i = 0;i < image_num;i++)
			ratio = GPyramids[i].ConstructPyramid(Im[i],opt.ratioForPyramid,opt.minWidthForPyramid);

		if(opt.displayRunningInfo)
			printf("done!\n");

		/*   Handle images from low resolution to high resolution     */
		std::vector<ZQ_DImage3D<T>> Images(image_num);

		for(int k = GPyramids[0].nlevels()-1;k >= 0;k--)
		{
			if(opt.displayRunningInfo)
				printf("Pyramid level %d \n",k);

			int width = GPyramids[0].Image(k).width();
			int height = GPyramids[0].Image(k).height();
			int depth = GPyramids[0].Image(k).depth();

			bool isSmooth = true;
			if(k == 0)
				isSmooth = false;

			for(int i = 0;i < image_num;i++)
				im2feature(Images[i],GPyramids[i].Image(k),isSmooth,opt);

			if(k == GPyramids[0].nlevels()-1)
			{
				for(int i = 0;i < image_num-1;i++)
				{
					u[i].allocate(width,height,depth);
					v[i].allocate(width,height,depth);
					w[i].allocate(width,height,depth);
					warpIm[i].copyData(Images[i+1]);
					OneResolution_HS_DL1(u[i],v[i],w[i],warpIm[i],Images[i],Images[i+1],opt);
				}
			}
			else
			{
				for(int i = 0;i < image_num-1;i++)
				{
					u[i].imresize(width,height,depth);
					u[i].Multiplywith(1.0/ratio);
					v[i].imresize(width,height,depth);
					v[i].Multiplywith(1.0/ratio);
					w[i].imresize(width,height,depth);
					w[i].Multiplywith(1.0/ratio);
					warpFL(warpIm[i],Images[i],Images[i+1],u[i],v[i],w[i],opt.useCubicWarping);
				}
			}

			OneResolution_TwoDir_Dec_DL1(u,v,w,warpIm,Images,opt);
		}
	}

	/****************************************************************************************************************************************/

	template<class T>
	void ZQ_OpticalFlow3D::OneResolution_HS_L2(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>&w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_DImage3D<T> imdx,imdy,imdz,imdt;

		int imWidth = Im1.width();
		int imHeight = Im1.height();
		int imDepth = Im1.depth();
		int nChannels = Im1.nchannels();
		int nPixels = imWidth*imHeight*imDepth;

		ZQ_DImage3D<T> du(imWidth,imHeight,imDepth),dv(imWidth,imHeight,imDepth),dw(imWidth,imHeight,imDepth); //for du, dv


		warpFL(warpIm2,Im1,Im2,u,v,w,opt.useCubicWarping);

		/************       Outer Loop Begin     *************/
		//refresh {u,v} in each loop
		for(int count = 0; count < opt.nOuterFixedPointIterations;count++)
		{
			// outer loop : {imdx, imdy, imdt} accoring to warpIm2, warpIm2 accoring to {u,v} 
			ZQ_DImage3D<T> imdx,imdy,imdz,imdt;

			getDxs(imdx,imdy,imdz,imdt,Im1,warpIm2,true);

			du.reset();
			dv.reset();
			dw.reset();

			//compute imdtdx, imdxx, imdyy, imdtdx, imdtdy
			ZQ_DImage3D<T> imdxx,imdxy,imdxz,imdyy,imdyz,imdzz,imdtdx,imdtdy,imdtdz;
			imdxx.Multiply(imdx,imdx);
			imdxy.Multiply(imdx,imdy);
			imdxz.Multiply(imdx,imdz);
			imdyy.Multiply(imdy,imdy);
			imdyz.Multiply(imdy,imdz);
			imdzz.Multiply(imdz,imdz);
			imdtdx.Multiply(imdx,imdt);
			imdtdy.Multiply(imdy,imdt);
			imdtdz.Multiply(imdz,imdt);

			if(nChannels>1)
			{
				imdxx.collapse();
				imdxy.collapse();
				imdxz.collapse();
				imdyy.collapse();
				imdyz.collapse();
				imdzz.collapse();
				imdtdx.collapse();
				imdtdy.collapse();
				imdtdz.collapse();
			}

			ZQ_DImage3D<T> laplace_u(imWidth,imHeight,imDepth);
			ZQ_DImage3D<T> laplace_v(imWidth,imHeight,imDepth);
			ZQ_DImage3D<T> laplace_w(imWidth,imHeight,imDepth);

			Laplacian(laplace_u,u);
			Laplacian(laplace_v,v);
			Laplacian(laplace_w,w);

			T*& laplace_uData = laplace_u.data();
			T*& laplace_vData = laplace_v.data();
			T*& laplace_wData = laplace_w.data();


			// set omega
			double omega = opt.omegaForSOR;
			double alpha2 = opt.alpha*opt.alpha;
			double beta2 = opt.beta*opt.beta;

			T*& duData = du.data();
			T*& dvData = dv.data();
			T*& dwData = dw.data();
			T*& uData = u.data();
			T*& vData = v.data();
			T*& wData = w.data();
			T*& imdtdxData = imdtdx.data();
			T*& imdtdyData = imdtdy.data();
			T*& imdtdzData = imdtdz.data();
			T*& imdxxData = imdxx.data();
			T*& imdxyData = imdxy.data();
			T*& imdxzData = imdxz.data();
			T*& imdyyData = imdyy.data();
			T*& imdyzData = imdyz.data();
			T*& imdzzData = imdzz.data();



			/***   SOR Begin solve du,dv***/

			for(int it = 0; it < opt.nSORIterations; it++)
			{
				for(int k = 0;k < imDepth;k++)
				{
					for(int j = 0; j < imHeight; j++)
					{
						for(int i = 0; i < imWidth; i++)
						{
							int offset = k*imHeight*imWidth + j*imWidth+i;
							double sigma1 = 0, sigma2 = 0, sigma3 = 0, coeff = 0;
							double _weight;

							if(i > 0)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset-1];
								sigma2 += _weight*dvData[offset-1];
								sigma3 += _weight*dwData[offset-1];
								coeff  += _weight;

							}
							if(i < imWidth-1)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset+1];
								sigma2 += _weight*dvData[offset+1];
								sigma3 += _weight*dwData[offset+1];
								coeff  += _weight;
							}
							if(j > 0)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset-imWidth];
								sigma2 += _weight*dvData[offset-imWidth];
								sigma3 += _weight*dwData[offset-imWidth];
								coeff  += _weight;
							}
							if(j < imHeight-1)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset+imWidth];
								sigma2 += _weight*dvData[offset+imWidth];
								sigma3 += _weight*dwData[offset+imWidth];
								coeff  += _weight;
							}
							if(k > 0)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset-imWidth*imHeight];
								sigma2 += _weight*dvData[offset-imWidth*imHeight];
								sigma3 += _weight*dwData[offset-imWidth*imHeight];
								coeff  += _weight;
							}
							if(k < imDepth-1)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset+imWidth*imHeight];
								sigma2 += _weight*dvData[offset+imWidth*imHeight];
								sigma3 += _weight*dwData[offset+imWidth*imHeight];
								coeff  += _weight;
							}
							sigma1 *= alpha2;
							sigma2 *= alpha2;
							sigma3 *= alpha2;
							coeff *= alpha2;
							// compute u
							sigma1 += alpha2*laplace_uData[offset] - imdtdxData[offset] - imdxyData[offset]*dvData[offset] - imdxzData[offset]*dwData[offset] - beta2*uData[offset];
							double coeff1 = coeff + imdxxData[offset] + beta2;
							duData[offset] = (1-omega)*duData[offset] + omega/coeff1*sigma1;
							// compute v
							sigma2 += alpha2*laplace_vData[offset] - imdtdyData[offset] - imdxyData[offset]*duData[offset] - imdyzData[offset]*dwData[offset] - beta2*vData[offset];
							double coeff2 = coeff + imdyyData[offset] + beta2;
							dvData[offset] = (1-omega)*dvData[offset] + omega/coeff2*sigma2;
							// compute w
							sigma3 += alpha2*laplace_wData[offset] - imdtdzData[offset] - imdxzData[offset]*duData[offset] - imdyzData[offset]*dvData[offset] - beta2*wData[offset];
							double coeff3 = coeff + imdzzData[offset] + beta2;
							dwData[offset] = (1-omega)*dwData[offset] + omega/coeff3*sigma3;
						}
					}
				}
			}

			/***   SOR end solve du,dv***/

			u.Addwith(du);
			v.Addwith(dv);
			w.Addwith(dw);

			warpFL(warpIm2,Im1,Im2,u,v,w,opt.useCubicWarping);

		}
		/************       Outer Loop End     *************/
	}

	template<class T>
	void ZQ_OpticalFlow3D::OneResolution_HS_DL1(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_DImage3D<T> imdx,imdy,imdz,imdt;

		int imWidth = Im1.width();
		int imHeight = Im1.height();
		int imDepth = Im1.depth();
		int nChannels = Im1.nchannels();
		int nPixels = imWidth*imHeight*imDepth;

		ZQ_DImage3D<T> du(imWidth,imHeight,imDepth),dv(imWidth,imHeight,imDepth),dw(imWidth,imHeight,imDepth);

		ZQ_DImage3D<T> Psi_1st(imWidth,imHeight,imDepth,nChannels);

		ZQ_DImage3D<T> imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz;

		ZQ_DImage3D<T> laplace_u,laplace_v,laplace_w;

		double varepsilon_phi = pow(0.001,2);
		double varepsilon_psi = pow(0.001,2);

		warpFL(warpIm2,Im1,Im2,u,v,w,opt.useCubicWarping);

		for(int out_it = 0;out_it < opt.nOuterFixedPointIterations;out_it++)
		{
			getDxs(imdx,imdy,imdz,imdt,Im1,warpIm2,true);

			du.reset();
			dv.reset();
			dw.reset();

			for(int in_it = 0; in_it < opt.nInnerFixedPointIterations;in_it++)
			{
				Psi_1st.reset();

				T*& psiData = Psi_1st.data();
				T*& imdxData = imdx.data();
				T*& imdyData = imdy.data();
				T*& imdzData = imdz.data();
				T*& imdtData = imdt.data();
				T*& duData = du.data();
				T*& dvData = dv.data();
				T*& dwData = dw.data();
				T*& uData = u.data();
				T*& vData = v.data();
				T*& wData = w.data();

				for(int i = 0;i < nPixels;i++)
				{
					for(int c = 0;c < nChannels;c++)
					{
						int offset = i*nChannels+c;
						double temp = imdtData[offset]+imdxData[offset]*duData[i]+imdyData[offset]*dvData[i]+imdzData[offset]*dwData[i];

						temp *= temp;
						psiData[offset]=1/(2*sqrt(temp+varepsilon_psi));
					}
				}

				imdxdx.Multiply(Psi_1st,imdx,imdx);
				imdxdy.Multiply(Psi_1st,imdx,imdy);
				imdxdz.Multiply(Psi_1st,imdx,imdz);
				imdydy.Multiply(Psi_1st,imdy,imdy);
				imdydz.Multiply(Psi_1st,imdy,imdz);
				imdzdz.Multiply(Psi_1st,imdz,imdz);
				imdtdx.Multiply(Psi_1st,imdx,imdt);
				imdtdy.Multiply(Psi_1st,imdy,imdt);
				imdtdz.Multiply(Psi_1st,imdz,imdt);

				if(nChannels > 1)
				{
					imdxdx.collapse();
					imdxdy.collapse();
					imdxdz.collapse();
					imdydy.collapse();
					imdydz.collapse();
					imdzdz.collapse();
					imdtdx.collapse();
					imdtdy.collapse();
					imdtdz.collapse();
				}

				T*& imdxdxData = imdxdx.data();
				T*& imdxdyData = imdxdy.data();
				T*& imdxdzData = imdxdz.data();
				T*& imdydyData = imdydy.data();
				T*& imdydzData = imdydz.data();
				T*& imdzdzData = imdzdz.data();
				T*& imdtdxData = imdtdx.data();
				T*& imdtdyData = imdtdy.data();
				T*& imdtdzData = imdtdz.data();


				Laplacian(laplace_u,u);
				Laplacian(laplace_v,v);
				Laplacian(laplace_w,w);

				T*& laplace_uPtr = laplace_u.data();
				T*& laplace_vPtr = laplace_v.data();
				T*& laplace_wPtr = laplace_w.data();

				double omega = opt.omegaForSOR;
				double alpha2 = opt.alpha*opt.alpha;
				double beta2 = opt.beta*opt.beta;

				/*Begin SOR*/

				for(int sor_it = 0; sor_it < opt.nSORIterations; sor_it++)
				{
					for(int k = 0;k < imDepth;k++)
					{
						for(int j = 0; j < imHeight; j++)
						{
							for(int i = 0; i < imWidth; i++)
							{
								int offset = k*imWidth*imHeight+j*imWidth+i;
								double sigma1 = 0, sigma2 = 0, sigma3 = 0, coeff = 0;
								double _weight;

								if(i > 0)
								{
									_weight = 1;
									sigma1 += _weight*duData[offset-1];
									sigma2 += _weight*dvData[offset-1];
									sigma3 += _weight*dwData[offset-1];
									coeff  += _weight;
								}
								if(i < imWidth-1)
								{
									_weight = 1;
									sigma1 += _weight*duData[offset+1];
									sigma2 += _weight*dvData[offset+1];
									sigma3 += _weight*dwData[offset+1];
									coeff  += _weight;
								}
								if(j > 0)
								{
									_weight = 1;
									sigma1 += _weight*duData[offset-imWidth];
									sigma2 += _weight*dvData[offset-imWidth];
									sigma3 += _weight*dwData[offset-imWidth];
									coeff  += _weight;
								}
								if(j < imHeight-1)
								{
									_weight = 1;
									sigma1 += _weight*duData[offset+imWidth];
									sigma2 += _weight*dvData[offset+imWidth];
									sigma3 += _weight*dwData[offset+imWidth];
									coeff  += _weight;
								}
								if(k > 0)
								{
									_weight = 1;
									sigma1 += _weight*duData[offset-imWidth*imHeight];
									sigma2 += _weight*dvData[offset-imWidth*imHeight];
									sigma3 += _weight*dwData[offset-imWidth*imHeight];
									coeff  += _weight;
								}
								if(k < imDepth-1)
								{
									_weight = 1;
									sigma1 += _weight*duData[offset+imWidth*imHeight];
									sigma2 += _weight*dvData[offset+imWidth*imHeight];
									sigma3 += _weight*dwData[offset+imWidth*imHeight];
									coeff  += _weight;
								}
								sigma1 *= alpha2;
								sigma2 *= alpha2;
								sigma3 *= alpha2;
								coeff  *= alpha2;

								sigma1 += alpha2*laplace_uPtr[offset] - imdtdxData[offset] - imdxdyData[offset]*dvData[offset] - imdxdzData[offset]*dwData[offset] - beta2*uData[offset];
								double coeff1 = coeff + imdxdxData[offset] + beta2;
								duData[offset] = (1-omega)*duData[offset] + omega/coeff1*sigma1;

								sigma2 += alpha2*laplace_vPtr[offset] - imdtdyData[offset] - imdxdyData[offset]*duData[offset] - imdydzData[offset]*dwData[offset] - beta2*vData[offset];
								double coeff2 = coeff + imdydyData[offset] + beta2;
								dvData[offset] = (1-omega)*dvData[offset] + omega/coeff2*sigma2;

								sigma3 += alpha2*laplace_wPtr[offset] - imdtdzData[offset] - imdxdzData[offset]*duData[offset] - imdydzData[offset]*dvData[offset] - beta2*wData[offset];
								double coeff3 = coeff + imdzdzData[offset] + beta2;
								dwData[offset] = (1-omega)*dwData[offset] + omega/coeff3*sigma3;
							}
						}
					}	
				}
				/*End SOR*/
			}
			u.Addwith(du);
			v.Addwith(dv);
			w.Addwith(dw);

			warpFL(warpIm2,Im1,Im2,u,v,w,opt.useCubicWarping);
		}
	}


	template<class T>
	void ZQ_OpticalFlow3D::OneResolution_HS_L1(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_DImage3D<T> imdx,imdy,imdz,imdt;

		int imWidth = Im1.width();
		int imHeight = Im1.height();
		int imDepth = Im1.depth();
		int nChannels = Im1.nchannels();
		int nPixels = imWidth*imHeight*imDepth;

		ZQ_DImage3D<T> du(imWidth,imHeight,imDepth),dv(imWidth,imHeight,imDepth),dw(imWidth,imHeight,imDepth);
		ZQ_DImage3D<T> uu(imWidth,imHeight,imDepth),vv(imWidth,imHeight,imDepth),ww(imWidth,imHeight,imDepth);
		ZQ_DImage3D<T> ux(imWidth,imHeight,imDepth),uy(imWidth,imHeight,imDepth),uz(imWidth,imHeight,imDepth);
		ZQ_DImage3D<T> vx(imWidth,imHeight,imDepth),vy(imWidth,imHeight,imDepth),vz(imWidth,imHeight,imDepth);
		ZQ_DImage3D<T> wx(imWidth,imHeight,imDepth),wy(imWidth,imHeight,imDepth),wz(imWidth,imHeight,imDepth);
		ZQ_DImage3D<T> Phi_gradu_1st(imWidth,imHeight,imDepth);
		ZQ_DImage3D<T> Phi_u_1st(imWidth,imHeight,imDepth);
		ZQ_DImage3D<T> Phi_v_1st(imWidth,imHeight,imDepth);
		ZQ_DImage3D<T> Phi_w_1st(imWidth,imHeight,imDepth);
		ZQ_DImage3D<T> Phi_data_1st(imWidth,imHeight,imDepth,nChannels);

		ZQ_DImage3D<T> imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz;

		ZQ_DImage3D<T> laplace_u,laplace_v,laplace_w;

		double varepsilon_phi = pow(0.001,2);
		double varepsilon_psi = pow(0.001,2);

		warpFL(warpIm2,Im1,Im2,u,v,w,opt.useCubicWarping);

		for(int out_it = 0;out_it < opt.nOuterFixedPointIterations;out_it++)
		{
			getDxs(imdx,imdy,imdz,imdt,Im1,warpIm2,true);

			du.reset();
			dv.reset();
			dw.reset();

			for(int in_it = 0; in_it < opt.nInnerFixedPointIterations;in_it++)
			{

				uu.Add(u,du);
				vv.Add(v,dv);
				ww.Add(w,dw);

				uu.dx(ux);
				uu.dy(uy);
				uu.dz(uz);
				vv.dx(vx);
				vv.dy(vy);
				vv.dz(vz);
				ww.dx(wx);
				ww.dy(wy);
				ww.dz(wz);

				Phi_gradu_1st.reset();

				T*& phi_graduData = Phi_gradu_1st.data();
				T*& uxData = ux.data();
				T*& uyData = uy.data();
				T*& uzData = uz.data();
				T*& vxData = vx.data();
				T*& vyData = vy.data();
				T*& vzData = vz.data();
				T*& wxData = wx.data();
				T*& wyData = wy.data();
				T*& wzData = wz.data();

				for(int i = 0;i < nPixels;i++)
				{
					double temp = uxData[i]*uxData[i]+uyData[i]*uyData[i]+uzData[i]*uzData[i]
									+vxData[i]*vxData[i]+vyData[i]*vyData[i]+vzData[i]*vzData[i]
									+wxData[i]*wxData[i]+wyData[i]*wyData[i]+wzData[i]*wzData[i];

					phi_graduData[i] = 0.5/sqrt(temp+varepsilon_phi);

				}

				Phi_u_1st.reset();
				Phi_v_1st.reset();
				Phi_w_1st.reset();

				T*& phi_uData = Phi_u_1st.data();
				T*& phi_vData = Phi_v_1st.data();
				T*& phi_wData = Phi_w_1st.data();
				T*& uuData = uu.data();
				T*& vvData = vv.data();
				T*& wwData = ww.data();

				for(int i = 0;i < nPixels;i++)
				{
					phi_uData[i] = 0.5/sqrt(uuData[i]*uuData[i]+varepsilon_phi);
					phi_vData[i] = 0.5/sqrt(vvData[i]*vvData[i]+varepsilon_phi);
					phi_wData[i] = 0.5/sqrt(wwData[i]*wwData[i]+varepsilon_phi);
				}


				Phi_data_1st.reset();

				T*& phi_dataData = Phi_data_1st.data();
				T*& imdxData = imdx.data();
				T*& imdyData = imdy.data();
				T*& imdzData = imdz.data();
				T*& imdtData = imdt.data();
				T*& duData = du.data();
				T*& dvData = dv.data();
				T*& dwData = dw.data();
				T*& uData = u.data();
				T*& vData = v.data();
				T*& wData = w.data();


				for(int i = 0;i < nPixels;i++)
				{
					for(int c = 0;c < nChannels;c++)
					{
						int offset = i*nChannels+c;
						double temp = imdtData[offset]+imdxData[offset]*duData[i]+imdyData[offset]*dvData[i]+imdzData[offset]*dwData[i];

						temp *= temp;
						phi_dataData[offset] = 0.5/sqrt(temp+varepsilon_psi);
					}
				}

				imdxdx.Multiply(Phi_data_1st,imdx,imdx);
				imdxdy.Multiply(Phi_data_1st,imdx,imdy);
				imdxdz.Multiply(Phi_data_1st,imdx,imdz);
				imdydy.Multiply(Phi_data_1st,imdy,imdy);
				imdydz.Multiply(Phi_data_1st,imdy,imdz);
				imdzdz.Multiply(Phi_data_1st,imdz,imdz);
				imdtdx.Multiply(Phi_data_1st,imdx,imdt);
				imdtdy.Multiply(Phi_data_1st,imdy,imdt);
				imdtdz.Multiply(Phi_data_1st,imdz,imdt);

				if(nChannels > 1)
				{
					imdxdx.collapse();
					imdxdy.collapse();
					imdxdz.collapse();
					imdydy.collapse();
					imdydz.collapse();
					imdzdz.collapse();
					imdtdx.collapse();
					imdtdy.collapse();
					imdtdz.collapse();
				}

				T*& imdxdxData = imdxdx.data();
				T*& imdxdyData = imdxdy.data();
				T*& imdxdzData = imdxdz.data();
				T*& imdydyData = imdydy.data();
				T*& imdydzData = imdydz.data();
				T*& imdzdzData = imdzdz.data();
				T*& imdtdxData = imdtdx.data();
				T*& imdtdyData = imdtdy.data();
				T*& imdtdzData = imdtdz.data();


				Laplacian(laplace_u,u,Phi_gradu_1st);
				Laplacian(laplace_v,v,Phi_gradu_1st);
				Laplacian(laplace_w,w,Phi_gradu_1st);

				T*& laplace_uPtr = laplace_u.data();
				T*& laplace_vPtr = laplace_v.data();
				T*& laplace_wPtr = laplace_w.data();

				double omega = opt.omegaForSOR;
				double alpha = opt.alpha;
				double beta = opt.beta;

				/*Begin SOR*/

				for(int sor_it = 0; sor_it < opt.nSORIterations; sor_it++)
				{
					for(int k = 0;k < imDepth;k++)
					{
						for(int j = 0; j < imHeight; j++)
						{
							for(int i = 0; i < imWidth; i++)
							{
								int offset = k*imWidth*imHeight+j*imWidth+i;
								double sigma1 = 0, sigma2 = 0, sigma3 = 0, coeff = 0;
								double _weight;


								if(i > 0)
								{
									_weight = phi_graduData[offset-1];
									sigma1 += _weight*duData[offset-1];
									sigma2 += _weight*dvData[offset-1];
									sigma3 += _weight*dwData[offset-1];
									coeff  += _weight;
								}
								if(i < imWidth-1)
								{
									_weight = phi_graduData[offset];
									sigma1 += _weight*duData[offset+1];
									sigma2 += _weight*dvData[offset+1];
									sigma3 += _weight*dwData[offset+1];
									coeff  += _weight;
								}
								if(j > 0)
								{
									_weight = phi_graduData[offset-imWidth];
									sigma1 += _weight*duData[offset-imWidth];
									sigma2 += _weight*dvData[offset-imWidth];
									sigma3 += _weight*dwData[offset-imWidth];
									coeff  += _weight;
								}
								if(j < imHeight-1)
								{
									_weight = phi_graduData[offset];
									sigma1 += _weight*duData[offset+imWidth];
									sigma2 += _weight*dvData[offset+imWidth];
									sigma3 += _weight*dwData[offset+imWidth];
									coeff  += _weight;
								}
								if(k > 0)
								{
									_weight = phi_graduData[offset];
									sigma1 += _weight*duData[offset-imWidth*imHeight];
									sigma2 += _weight*dvData[offset-imWidth*imHeight];
									sigma3 += _weight*dwData[offset-imWidth*imHeight];
									coeff  += _weight;
								}
								if(k < imDepth-1)
								{
									_weight = phi_graduData[offset];
									sigma1 += _weight*duData[offset+imWidth*imHeight];
									sigma2 += _weight*dvData[offset+imWidth*imHeight];
									sigma3 += _weight*dwData[offset+imWidth*imHeight];
									coeff  += _weight;
								}
								sigma1 *= alpha;
								sigma2 *= alpha;
								sigma3 *= alpha;
								coeff  *= alpha;

								sigma1 += alpha*laplace_uPtr[offset] - imdtdxData[offset] - imdxdyData[offset]*dvData[offset] - imdxdzData[offset]*dwData[offset] - beta*phi_uData[offset]*uData[offset];
								double coeff1 = coeff + imdxdxData[offset] + beta*phi_uData[offset];
								duData[offset] = (1-omega)*duData[offset] + omega/coeff1*sigma1;

								sigma2 += alpha*laplace_vPtr[offset] - imdtdyData[offset] - imdxdyData[offset]*duData[offset] - imdydzData[offset]*dwData[offset] - beta*phi_vData[offset]*vData[offset];
								double coeff2 = coeff + imdydyData[offset] + beta*phi_vData[offset];
								dvData[offset] = (1-omega)*dvData[offset] + omega/coeff2*sigma2;

								sigma3 += alpha*laplace_wPtr[offset] - imdtdzData[offset] - imdxdzData[offset]*duData[offset] - imdydzData[offset]*dvData[offset] - beta*phi_wData[offset]*wData[offset];
								double coeff3 = coeff + imdzdzData[offset] + beta*phi_wData[offset];
								dwData[offset] = (1-omega)*dwData[offset] + omega/coeff3*sigma3;
							}
						}

					}
					
				}
				/*End SOR*/
			}
			u.Addwith(du);
			v.Addwith(dv);
			w.Addwith(dw);

			warpFL(warpIm2,Im1,Im2,u,v,w,opt.useCubicWarping);
		}
	}


	template<class T>
	void ZQ_OpticalFlow3D::OneResolution_ADMM_L2(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ADMM_F_G(u,v,w,warpIm2,Im1,Im2,opt,Proximal_F_L2<T>,Proximal_G<T>);
	}

	template<class T>
	void ZQ_OpticalFlow3D::OneResolution_ADMM_DL1(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt)
	{
		ADMM_F_G(u,v,w,warpIm2,Im1,Im2,opt,Proximal_F_DL1<T>,Proximal_G<T>);
	}

	template<class T>
	void ZQ_OpticalFlow3D::OneResolution_OneDir_Inc_L2(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();
		int vel_num = image_num-1;

		int width = Im[0].width();
		int height = Im[0].height();
		int depth = Im[0].depth();
		int nChannels = Im[0].nchannels();

		switch (opt.initType)
		{
		case ZQ_OpticalFlowOptions::NONE_AS_INIT:
			break;
		case ZQ_OpticalFlowOptions::L2_AS_INIT:
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);
				OneResolution_HS_L2(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt);
			}
			break;
		case ZQ_OpticalFlowOptions::ADMM_AS_INIT:
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);
				OneResolution_ADMM_L2(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt);
			}
			break;
		}

		for(int alt_it = 0;alt_it < opt.nAlterations;alt_it++)
		{
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);

				if(i == 0)
				{
					ADMM_F_G(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt,Proximal_F_L2<T>,Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_last(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i-1],v[i-1],w[i-1],opt,Proximal_F_L2<T>,Proximal_F2_Last<T>,Proximal_G<T>);
				}
				
			}
		}
	}


	template<class T>
	void ZQ_OpticalFlow3D::OneResolution_OneDir_Inc_DL1(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();
		int vel_num = image_num-1;

		int width = Im[0].width();
		int height = Im[0].height();
		int depth = Im[0].depth();
		int nChannels = Im[0].nchannels();

		switch (opt.initType)
		{
		case ZQ_OpticalFlowOptions::NONE_AS_INIT:
			break;
		case ZQ_OpticalFlowOptions::L2_AS_INIT:
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);
				OneResolution_HS_DL1(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt);
			}
			break;
		case ZQ_OpticalFlowOptions::ADMM_AS_INIT:
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);
				OneResolution_ADMM_DL1(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt);
			}
			break;
		}

		for(int alt_it = 0;alt_it < opt.nAlterations;alt_it++)
		{
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);

				if(i == 0)
				{
					ADMM_F_G(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt,Proximal_F_DL1<T>,Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_last(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i-1],v[i-1],w[i-1],opt,Proximal_F_DL1<T>,Proximal_F2_Last<T>,Proximal_G<T>);
				}
			}
		}
	}


	template<class T>
	void ZQ_OpticalFlow3D::OneResolution_OneDir_Dec_L2(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();
		int vel_num = image_num-1;

		int width = Im[0].width();
		int height = Im[0].height();
		int depth = Im[0].depth();
		int nChannels = Im[0].nchannels();

		switch (opt.initType)
		{
		case ZQ_OpticalFlowOptions::NONE_AS_INIT:
			break;
		case ZQ_OpticalFlowOptions::L2_AS_INIT:
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);
				OneResolution_HS_L2(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt);
			}
			break;
		case ZQ_OpticalFlowOptions::ADMM_AS_INIT:
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);
				OneResolution_ADMM_L2(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt);
			}
			break;
		}

		for(int alt_it = 0;alt_it < opt.nAlterations;alt_it++)
		{
			for(int i = vel_num-1;i >= 0;i--)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);

				if(i == vel_num-1)
				{
					ADMM_F_G(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt,Proximal_F_L2<T>,Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_first(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i+1],v[i+1],w[i+1],opt,Proximal_F_L2<T>,Proximal_F2_First<T>,Proximal_G<T>);
				}
			}
		}
	}


	template<class T>
	void ZQ_OpticalFlow3D::OneResolution_OneDir_Dec_DL1(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();
		int vel_num = image_num-1;

		int width = Im[0].width();
		int height = Im[0].height();
		int depth = Im[0].depth();
		int nChannels = Im[0].nchannels();

		switch (opt.initType)
		{
		case ZQ_OpticalFlowOptions::NONE_AS_INIT:
			break;
		case ZQ_OpticalFlowOptions::L2_AS_INIT:
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);
				OneResolution_HS_DL1(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt);
			}
			break;
		case ZQ_OpticalFlowOptions::ADMM_AS_INIT:
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);
				OneResolution_ADMM_DL1(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt);
			}
			break;
		}

		for(int alt_it = 0;alt_it < opt.nAlterations;alt_it++)
		{
			for(int i = vel_num-1;i >= 0;i--)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);

				if(i == vel_num-1)
				{
					ADMM_F_G(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt,Proximal_F_DL1<T>,Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_first(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i+1],v[i+1],w[i+1],opt,Proximal_F_DL1<T>,Proximal_F2_First<T>,Proximal_G<T>);
				}
			}
		}
	}

	template<class T>
	void ZQ_OpticalFlow3D::OneResolution_TwoDir_Inc_L2(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();
		int vel_num = image_num-1;

		int width = Im[0].width();
		int height = Im[0].height();
		int depth = Im[0].depth();
		int nChannels = Im[0].nchannels();

		switch (opt.initType)
		{
		case ZQ_OpticalFlowOptions::NONE_AS_INIT:
			break;
		case ZQ_OpticalFlowOptions::L2_AS_INIT:
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);
				OneResolution_HS_L2(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt);
			}
			break;
		case ZQ_OpticalFlowOptions::ADMM_AS_INIT:
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);
				OneResolution_ADMM_L2(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt);
			}
			break;
		}

		for(int alt_it = 0;alt_it < opt.nAlterations;alt_it++)
		{
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);

				if(i == 0)
				{
					ADMM_F1_F2_G_first(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i+1],v[i+1],w[i+1],opt,Proximal_F_L2<T>,Proximal_F2_First<T>,Proximal_G<T>);
				}
				else if(i == vel_num-1)
				{
					ADMM_F1_F2_G_last(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i-1],v[i-1],w[i-1],opt,Proximal_F_L2<T>,Proximal_F2_Last<T>,Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_middle(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i-1],v[i-1],w[i-1],u[i+1],v[i+1],w[i+1],opt,Proximal_F_L2<T>,Proximal_F2_Middle<T>,Proximal_G<T>);
				}
			}

			if(!opt.isReflect)
				continue;

			for(int i = vel_num-1;i >= 0 ;i--)
			{
				if(i == 0)
				{
					ADMM_F1_F2_G_first(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i+1],v[i+1],w[i+1],opt,Proximal_F_L2<T>,Proximal_F2_First<T>,Proximal_G<T>);
				}
				else if(i == vel_num-1)
				{
					ADMM_F1_F2_G_last(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i-1],v[i-1],w[i-1],opt,Proximal_F_L2<T>,Proximal_F2_Last<T>,Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_middle(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i-1],v[i-1],w[i-1],u[i+1],v[i+1],w[i+1],opt,Proximal_F_L2<T>,Proximal_F2_Middle<T>,Proximal_G<T>);
				}
			}
		}
	}

	template<class T>
	void ZQ_OpticalFlow3D::OneResolution_TwoDir_Inc_DL1(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();
		int vel_num = image_num-1;

		int width = Im[0].width();
		int height = Im[0].height();
		int depth = Im[0].depth();
		int nChannels = Im[0].nchannels();

		switch (opt.initType)
		{
		case ZQ_OpticalFlowOptions::NONE_AS_INIT:
			break;
		case ZQ_OpticalFlowOptions::L2_AS_INIT:
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);
				OneResolution_HS_DL1(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt);
			}
			break;
		case ZQ_OpticalFlowOptions::ADMM_AS_INIT:
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);
				OneResolution_ADMM_DL1(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt);
			}
			break;
		}

		for(int alt_it = 0;alt_it < opt.nAlterations;alt_it++)
		{
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);

				if(i == 0)
				{
					ADMM_F1_F2_G_first(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i+1],v[i+1],w[i+1],opt,Proximal_F_DL1<T>,Proximal_F2_First<T>,Proximal_G<T>);
				}
				else if(i == vel_num-1)
				{
					ADMM_F1_F2_G_last(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i-1],v[i-1],w[i-1],opt,Proximal_F_DL1<T>,Proximal_F2_Last<T>,Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_middle(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i-1],v[i-1],w[i-1],u[i+1],v[i+1],w[i+1],opt,Proximal_F_DL1<T>,Proximal_F2_Middle<T>,Proximal_G<T>);
				}
			}

			if(!opt.isReflect)
				continue;

			for(int i = vel_num-1;i >= 0 ;i--)
			{
				if(i == 0)
				{
					ADMM_F1_F2_G_first(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i+1],v[i+1],w[i+1],opt,Proximal_F_DL1<T>,Proximal_F2_First<T>,Proximal_G<T>);
				}
				else if(i == vel_num-1)
				{
					ADMM_F1_F2_G_last(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i-1],v[i-1],w[i-1],opt,Proximal_F_DL1<T>,Proximal_F2_Last<T>,Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_middle(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i-1],v[i-1],w[i-1],u[i+1],v[i+1],w[i+1],opt,Proximal_F_DL1<T>,Proximal_F2_Middle<T>,Proximal_G<T>);
				}
			}
		}
	}


	template<class T>
	void ZQ_OpticalFlow3D::OneResolution_TwoDir_Dec_L2(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();
		int vel_num = image_num-1;

		int width = Im[0].width();
		int height = Im[0].height();
		int depth = Im[0].depth();
		int nChannels = Im[0].nchannels();

		switch (opt.initType)
		{
		case ZQ_OpticalFlowOptions::NONE_AS_INIT:
			break;
		case ZQ_OpticalFlowOptions::L2_AS_INIT:
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);
				OneResolution_HS_L2(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt);
			}
			break;
		case ZQ_OpticalFlowOptions::ADMM_AS_INIT:
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);
				OneResolution_ADMM_L2(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt);
			}
			break;
		}

		for(int alt_it = 0;alt_it < opt.nAlterations;alt_it++)
		{
			for(int i = vel_num-1;i >= 0;i--)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);

				if(i == 0)
				{
					ADMM_F1_F2_G_first(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i+1],v[i+1],w[i+1],opt,Proximal_F_L2<T>,Proximal_F2_First<T>,Proximal_G<T>);
				}
				else if(i == vel_num-1)
				{
					ADMM_F1_F2_G_last(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i-1],v[i-1],w[i-1],opt,Proximal_F_L2<T>,Proximal_F2_Last<T>,Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_middle(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i-1],v[i-1],w[i-1],u[i+1],v[i+1],w[i+1],opt,Proximal_F_L2<T>,Proximal_F2_Middle<T>,Proximal_G<T>);
				}
			}

			if(!opt.isReflect)
				continue;

			for(int i = 0;i < vel_num ;i++)
			{
				if(i == 0)
				{
					ADMM_F1_F2_G_first(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i+1],v[i+1],w[i+1],opt,Proximal_F_L2<T>,Proximal_F2_First<T>,Proximal_G<T>);
				}
				else if(i == vel_num-1)
				{
					ADMM_F1_F2_G_last(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i-1],v[i-1],w[i-1],opt,Proximal_F_L2<T>,Proximal_F2_Last<T>,Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_middle(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i-1],v[i-1],w[i-1],u[i+1],v[i+1],w[i+1],opt,Proximal_F_L2<T>,Proximal_F2_Middle<T>,Proximal_G<T>);
				}
			}
		}
	}


	template<class T>
	void ZQ_OpticalFlow3D::OneResolution_TwoDir_Dec_DL1(std::vector<ZQ_DImage3D<T>>& u, std::vector<ZQ_DImage3D<T>>& v, std::vector<ZQ_DImage3D<T>>& w, std::vector<ZQ_DImage3D<T>>& warpIm, const std::vector<ZQ_DImage3D<T>>& Im, const ZQ_OpticalFlowOptions& opt)
	{
		int image_num = Im.size();
		int vel_num = image_num-1;

		int width = Im[0].width();
		int height = Im[0].height();
		int depth = Im[0].depth();
		int nChannels = Im[0].nchannels();

		switch (opt.initType)
		{
		case ZQ_OpticalFlowOptions::NONE_AS_INIT:
			break;
		case ZQ_OpticalFlowOptions::L2_AS_INIT:
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);
				OneResolution_HS_DL1(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt);
			}
			break;
		case ZQ_OpticalFlowOptions::ADMM_AS_INIT:
			for(int i = 0;i < vel_num;i++)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);
				OneResolution_ADMM_DL1(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],opt);
			}
			break;
		}

		for(int alt_it = 0;alt_it < opt.nAlterations;alt_it++)
		{
			for(int i = vel_num-1;i >= 0;i--)
			{
				if(!warpIm[i].matchDimension(width,height,depth,nChannels))
					warpIm[i].allocate(width,height,depth,nChannels);

				if(i == 0)
				{
					ADMM_F1_F2_G_first(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i+1],v[i+1],w[i+1],opt,Proximal_F_DL1<T>,Proximal_F2_First<T>,Proximal_G<T>);
				}
				else if(i == vel_num-1)
				{
					ADMM_F1_F2_G_last(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i-1],v[i-1],w[i-1],opt,Proximal_F_DL1<T>,Proximal_F2_Last<T>,Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_middle(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i-1],v[i-1],w[i-1],u[i+1],v[i+1],w[i+1],opt,Proximal_F_DL1<T>,Proximal_F2_Middle<T>,Proximal_G<T>);
				}
			}

			if(!opt.isReflect)
				continue;

			for(int i = 0;i < vel_num ;i++)
			{
				if(i == 0)
				{
					ADMM_F1_F2_G_first(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i+1],v[i+1],w[i+1],opt,Proximal_F_DL1<T>,Proximal_F2_First<T>,Proximal_G<T>);
				}
				else if(i == vel_num-1)
				{
					ADMM_F1_F2_G_last(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i-1],v[i-1],w[i-1],opt,Proximal_F_DL1<T>,Proximal_F2_Last<T>,Proximal_G<T>);
				}
				else
				{
					ADMM_F1_F2_G_middle(u[i],v[i],w[i],warpIm[i],Im[i],Im[i+1],u[i-1],v[i-1],w[i-1],u[i+1],v[i+1],w[i+1],opt,Proximal_F_DL1<T>,Proximal_F2_Middle<T>,Proximal_G<T>);
				}
			}
		}
	}

	/**************************  protected functions  *****************************/

	template<class T>
	void ZQ_OpticalFlow3D::getDxs(ZQ_DImage3D<T>& imdx, ZQ_DImage3D<T>& imdy, ZQ_DImage3D<T>& imdz, ZQ_DImage3D<T>& imdt, const ZQ_DImage3D<T>& im1, const ZQ_DImage3D<T>& im2, bool isSmooth/* = true*/)
	{
		//double gfilter[5]={0.01,0.09,0.8,0.09,0.01};
		T gfilter[5]={0.02,0.11,0.74,0.11,0.02};

		if(1)
		{

			if(isSmooth)
			{
				ZQ_DImage3D<T> Im1,Im2,Im;
				im1.imfilter_hvd(Im1,gfilter,2,gfilter,2,gfilter,2);
				im2.imfilter_hvd(Im2,gfilter,2,gfilter,2,gfilter,2);
				Im.copyData(Im1);
				Im.Multiplywith(0.4);
				Im.Addwith(Im2,0.6);
				Im.dx(imdx,isSmooth);
				Im.dy(imdy,isSmooth);
				Im.dz(imdz,isSmooth);
				imdt.Subtract(Im2,Im1);
			}
			else
			{
				ZQ_DImage3D<T> Im;
				Im.copyData(im1);
				Im.Multiplywith(0.4);
				Im.Addwith(im2,0.6);
				Im.dx(imdx,isSmooth);
				Im.dy(imdy,isSmooth);
				Im.dz(imdz,isSmooth);
				imdt.Subtract(im2,im1);
			}

		}
		else
		{
			ZQ_DImage3D<T> Im1,Im2;	
			im1.imfilter_hvd(Im1,gfilter,2,gfilter,2,gfilter,2);
			im2.imfilter_hvd(Im2,gfilter,2,gfilter,2,gfilter,2);
			Im2.dx(imdx,true);
			Im2.dy(imdy,true);
			Im2.dz(imdz,true);
			imdt.Subtract(Im2,Im1);
		}
	}

	template<class T>
	void ZQ_OpticalFlow3D::warpFL(ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_DImage3D<T>& u, const ZQ_DImage3D<T>& v, const ZQ_DImage3D<T>& w, bool isBicubic/* = false*/)
	{
		if(warpIm2.matchDimension(Im2) == false)
			warpIm2.allocate(Im2.width(),Im2.height(),Im2.depth(),Im2.nchannels());
		if(!isBicubic)
			ZQ_ImageProcessing3D::WarpImage(warpIm2.data(),Im2.data(),u.data(),v.data(),w.data(),Im2.width(),Im2.height(),Im2.depth(),Im2.nchannels(),Im1.data());
		else
			ZQ_ImageProcessing3D::WarpImageTricubic(warpIm2.data(),Im2.data(),u.data(),v.data(),w.data(),Im2.width(),Im2.height(),Im2.depth(),Im2.nchannels(),Im1.data());
	}

	template<class T>
	void ZQ_OpticalFlow3D::Laplacian(ZQ_DImage3D<T>& output, const ZQ_DImage3D<T>& input)
	{
		if(output.matchDimension(input) == false)
			output.allocate(input);
		else
			output.reset();
		
		ZQ_ImageProcessing3D::Laplacian(input.data(),output.data(),input.width(),input.height(),input.depth(),input.nchannels());
	}

	template<class T>
	void ZQ_OpticalFlow3D::Laplacian(ZQ_DImage3D<T>& output, const ZQ_DImage3D<T>& input, const ZQ_DImage3D<T>& weight)
	{
		if(output.matchDimension(input) == false)
			output.allocate(input);
		output.reset();

		if(input.matchDimension(weight) == false)
		{
			printf("Error in image dimension matching ZQ_OpticalFlow::Laplacian()!\n");
			return;
		}

		const T*& inputData = input.data();
		const T*& weightData = weight.data();
		int width = input.width();
		int height = input.height();
		int depth = input.depth();
		int nChannels = input.nchannels();
		ZQ_DImage3D<T> foo(width,height,depth);

		T*& fooData = foo.data();
		T*& outputData = output.data();


		for(int c = 0;c < nChannels;c++)
		{
			foo.reset();
			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 0;i < width-1;i++)
					{
						int offset = k*width*height+j*width+i;
						fooData[offset]=(inputData[(offset+1)*nChannels+c]-inputData[offset*nChannels+c])*weightData[offset*nChannels+c];
					}
				}
			}
			
			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 0;i < width;i++)
					{
						int offset = k*width*height+j*width+i;
						if(i < width-1)
							outputData[offset*nChannels+c] += fooData[offset];
						if(i > 0)
							outputData[offset*nChannels+c] -= fooData[offset-1];
					}
				}

			}
			
			foo.reset();
			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height-1;j++)
				{
					for(int i = 0;i < width;i++)
					{
						int offset = k*width*height+j*width+i;
						fooData[offset] = (inputData[(offset+width)*nChannels+c]-inputData[offset*nChannels+c])*weightData[offset*nChannels+c];
					}
				}
			}

			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 0;i < width;i++)
					{
						int offset = k*width*height+j*width+i;
						if(j < height-1)
							outputData[offset*nChannels+c] += fooData[offset];
						if(j > 0)
							outputData[offset*nChannels+c] -= fooData[offset-width];
					}
				}
			}

			foo.reset();
			for(int k = 0;k < depth-1;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 0;i < width;i++)
					{
						int offset = k*width*height+j*width+i;
						fooData[offset] = (inputData[(offset+width*height)*nChannels+c]-inputData[offset*nChannels+c])*weightData[offset*nChannels+c];
					}
				}
			}

			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 0;i < width;i++)
					{
						int offset = k*width*height+j*width+i;
						if(k < depth-1)
							outputData[offset*nChannels+c] += fooData[offset];
						if(k > 0)
							outputData[offset*nChannels+c] -= fooData[offset-width*height];
					}
				}
			}
		}
	}
		

	// function to convert image to features
	template<class T>
	void ZQ_OpticalFlow3D::im2feature(ZQ_DImage3D<T>& imfeature, const ZQ_DImage3D<T>& im, bool isSmooth, const ZQ_OpticalFlowOptions& opt)
	{
		int width = im.width();
		int height = im.height();
		int depth = im.depth();
		int nchannels = im.nchannels();
		if(nchannels == 1)
		{
			switch(opt.featureType)
			{
			case ZQ_OpticalFlowOptions::FEATURE_GRADIENT:
				{
					imfeature.allocate(width,height,depth,4);
					ZQ_DImage3D<T> imdx,imdy,imdz;
					im.dx(imdx,isSmooth);
					im.dy(imdy,isSmooth);
					im.dz(imdz,isSmooth);
					T*& data = imfeature.data();
					const T*& im_Data = im.data();
					T*& imdx_Data = imdx.data();
					T*& imdy_Data = imdy.data();
					T*& imdz_Data = imdz.data();
					for(int k = 0;k < depth;k++)
					{
						for(int j = 0;j < height;j++)
						{
							for(int i = 0;i < width;i++)
							{
								int offset = k*width*height+j*width+i;
								data[offset*4+0] = im_Data[offset];
								data[offset*4+1] = imdx_Data[offset];
								data[offset*4+2] = imdy_Data[offset];
								data[offset*4+3] = imdz_Data[offset];
							}
						}
					}
				}
				break;

			case ZQ_OpticalFlowOptions::FEATURE_FORWARD_NEIGHBOR:
				{
					imfeature.allocate(width,height,depth,4);
					T*& data = imfeature.data();
					const T*& im_Data = im.data();

					for(int d = 0;d < depth;d++)
					{
						for(int h = 0;h < height;h++)
						{
							for(int w = 0;w < width;w++)
							{
								int offset = d*height*width+h*width+w;
								int dd[4] = {d,d,d,d+1};
								int hh[4] = {h,h,h+1,h};
								int ww[4] = {w,w+1,w,w};
								float wei[4] = {1,1,1,1};
								for(int cid = 0;cid < 4;cid++)
								{
									int real_dd = __max(0,__min(depth-1,dd[cid]));
									int real_hh = __max(0,__min(height-1,hh[cid]));
									int real_ww = __max(0,__min(width-1,ww[cid]));
									data[offset*4+cid] = wei[cid]*im_Data[real_dd*height*width+real_hh*width+real_ww];
								}
							}
						}
					}
					
				}
				break;

			case ZQ_OpticalFlowOptions::FEATURE_BIDIRECTIONAL_NEIGHBOR:
				{
					imfeature.allocate(width,height,depth,7);
					T*& data = imfeature.data();
					const T*& im_Data = im.data();

					for(int d = 0;d < depth;d++)
					{
						for(int h = 0;h < height;h++)
						{
							for(int w = 0;w < width;w++)
							{
								int offset = d*height*width+h*width+w;
								int dd[7] = {d,d,d,d+1,d,d,d-1};
								int hh[7] = {h,h,h+1,h,h,h-1,h};
								int ww[7] = {w,w+1,w,w,w-1,w,w};
								float wei[7] = {1,1,1,1,1,1,1};
								for(int cid = 0;cid < 7;cid++)
								{
									int real_dd = __max(0,__min(depth-1,dd[cid]));
									int real_hh = __max(0,__min(height-1,hh[cid]));
									int real_ww = __max(0,__min(width-1,ww[cid]));
									data[offset*7+cid] = wei[cid]*im_Data[real_dd*height*width+real_hh*width+real_ww];
								}
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

	/**************************************************************************/

	template<class T>
	void ZQ_OpticalFlow3D::ADMM_F_G(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_OpticalFlowOptions& opt,
		void (*funcF)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, ZQ_DImage3D<T>& /*warpIm2*/, const ZQ_DImage3D<T>& /*Im1*/, const ZQ_DImage3D<T>& /*Im2*/, const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_OpticalFlowOptions& /*opt*/),
		void (*funcG)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_OpticalFlowOptions& /*opt*/))
	{
		ZQ_DImage3D<T> u_for_F(u),v_for_F(v),w_for_F(w);
		ZQ_DImage3D<T> u_for_G(u),v_for_G(v),w_for_G(w);
		ZQ_DImage3D<T> u_for_q(u),v_for_q(v),w_for_q(w);
		ZQ_DImage3D<T> z_u,z_v,z_w;

		u_for_q.reset(),v_for_q.reset(),w_for_q.reset();


		for(int it = 0;it < opt.nADMMIterations;it++)
		{
			z_u.copyData(u_for_G);
			z_u.Addwith(u_for_q,-1.0);
			z_v.copyData(v_for_G);
			z_v.Addwith(v_for_q,-1.0);
			z_w.copyData(w_for_G);
			z_w.Addwith(w_for_q,-1.0);


			funcF(u_for_F,v_for_F,w_for_F,warpIm2,Im1,Im2,z_u,z_v,z_w,opt);

			z_u.copyData(u_for_F);
			z_u.Addwith(u_for_q,1.0);
			z_v.copyData(v_for_F);
			z_v.Addwith(v_for_q,1.0);
			z_w.copyData(w_for_F);
			z_w.Addwith(w_for_q,1.0);

			funcG(u_for_G,v_for_G,w_for_G,z_u,z_v,z_w,opt);

			u_for_q.Addwith(u_for_F,1);
			u_for_q.Addwith(u_for_G,-1);
			v_for_q.Addwith(v_for_F,1);
			v_for_q.Addwith(v_for_G,-1);
			w_for_q.Addwith(w_for_F,1);
			w_for_q.Addwith(w_for_F,-1);
		}
		u.copyData(u_for_F);
		v.copyData(v_for_F);
		w.copyData(w_for_F);
	}

	template<class T>
	void ZQ_OpticalFlow3D::ADMM_F1_F2_G_first(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_DImage3D<T>& next_u, const ZQ_DImage3D<T>& next_v, const ZQ_DImage3D<T>& next_w, const ZQ_OpticalFlowOptions& opt,
		void (*funcF1)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, ZQ_DImage3D<T>& /*warpIm2*/, const ZQ_DImage3D<T>& /*Im1*/, const ZQ_DImage3D<T>& /*Im2*/, const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_OpticalFlowOptions& /*opt*/),
		void (*funcF2)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_DImage3D<T>& /*next_u*/, const ZQ_DImage3D<T>& /*next_v*/, const ZQ_DImage3D<T>& /*next_w*/, const ZQ_OpticalFlowOptions& /*opt*/),
		void (*funcG)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_OpticalFlowOptions& /*opt*/))
	{
		ZQ_DImage3D<T> u_for_F1(u),v_for_F1(v),w_for_F1(w);
		ZQ_DImage3D<T> u_for_F2(u),v_for_F2(v),w_for_F2(w);
		ZQ_DImage3D<T> u_for_G(u),v_for_G(v),w_for_G(w);
		ZQ_DImage3D<T> u_for_q1(u),v_for_q1(v),w_for_q1(w);
		ZQ_DImage3D<T> u_for_q2(u),v_for_q2(v),w_for_q2(w);
		ZQ_DImage3D<T> z_u,z_v,z_w;

		u_for_q1.reset(),v_for_q1.reset(),w_for_q1.reset();
		u_for_q2.reset(),v_for_q2.reset(),w_for_q2.reset();

		for(int it = 0;it < opt.nADMMIterations;it++)
		{
			z_u.copyData(u_for_G);
			z_u.Addwith(u_for_q1,-1.0);
			z_v.copyData(v_for_G);
			z_v.Addwith(v_for_q1,-1.0);
			z_w.copyData(w_for_G);
			z_w.Addwith(w_for_q1,-1.0);

			funcF1(u_for_F1,v_for_F1,w_for_F1,warpIm2,Im1,Im2,z_u,z_v,z_w,opt);

			z_u.copyData(u_for_G);
			z_u.Addwith(u_for_q2,-1.0);
			z_v.copyData(v_for_G);
			z_v.Addwith(v_for_q2,-1.0);
			z_w.copyData(w_for_G);
			z_w.Addwith(w_for_q2,-1.0);

			funcF2(u_for_F2,v_for_F2,w_for_F2,z_u,z_v,z_w,next_u,next_v,next_w,opt);

			z_u.copyData(u_for_F1);
			z_u.Addwith(u_for_q1,1.0);
			z_u.Addwith(u_for_F2);
			z_u.Addwith(u_for_q2,1.0);
			z_u.Multiplywith(0.5);

			z_v.copyData(v_for_F1);
			z_v.Addwith(v_for_q1,1.0);
			z_v.Addwith(v_for_F2);
			z_v.Addwith(v_for_q2,1.0);
			z_v.Multiplywith(0.5);

			z_w.copyData(w_for_F1);
			z_w.Addwith(w_for_q1,1.0);
			z_w.Addwith(w_for_F2);
			z_w.Addwith(w_for_q2,1.0);
			z_w.Multiplywith(0.5);

			funcG(u_for_G,v_for_G,w_for_G,z_u,z_v,z_w,opt);

			u_for_q1.Addwith(u_for_F1,1);
			u_for_q1.Addwith(u_for_G,-1);
			v_for_q1.Addwith(v_for_F1,1);
			v_for_q1.Addwith(v_for_G,-1);
			w_for_q1.Addwith(w_for_F1,1);
			w_for_q1.Addwith(w_for_G,-1);

			u_for_q2.Addwith(u_for_F2,1);
			u_for_q2.Addwith(u_for_G,-1);
			v_for_q2.Addwith(v_for_F2,1);
			v_for_q2.Addwith(v_for_G,-1);
			w_for_q2.Addwith(w_for_F2,1);
			w_for_q2.Addwith(w_for_G,-1);
		}
		u.copyData(u_for_F1);
		v.copyData(v_for_F1);
		w.copyData(w_for_F1);
	}

	template<class T>
	void ZQ_OpticalFlow3D::ADMM_F1_F2_G_last(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_DImage3D<T>& pre_u, const ZQ_DImage3D<T>& pre_v, const ZQ_DImage3D<T>& pre_w, const ZQ_OpticalFlowOptions& opt,
		void (*funcF1)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, ZQ_DImage3D<T>& /*warpIm2*/, const ZQ_DImage3D<T>& /*Im1*/, const ZQ_DImage3D<T>& /*Im2*/,  const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_OpticalFlowOptions& /*opt*/),
		void (*funcF2)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_DImage3D<T>& /*pre_u*/, const ZQ_DImage3D<T>& /*pre_v*/, const ZQ_DImage3D<T>& /*pre_w*/, const ZQ_OpticalFlowOptions& /*opt*/),
		void (*funcG)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_OpticalFlowOptions& /*opt*/))
	{
		ZQ_DImage3D<T> u_for_F1(u),v_for_F1(v),w_for_F1(w);
		ZQ_DImage3D<T> u_for_F2(u),v_for_F2(v),w_for_F2(w);
		ZQ_DImage3D<T> u_for_G(u),v_for_G(v),w_for_G(w);
		ZQ_DImage3D<T> u_for_q1(u),v_for_q1(v),w_for_q1(w);
		ZQ_DImage3D<T> u_for_q2(u),v_for_q2(v),w_for_q2(w);
		ZQ_DImage3D<T> z_u,z_v,z_w;

		u_for_q1.reset(),v_for_q1.reset(),w_for_q1.reset();
		u_for_q2.reset(),v_for_q2.reset(),w_for_q2.reset();

		for(int it = 0;it < opt.nADMMIterations;it++)
		{
			z_u.copyData(u_for_G);
			z_u.Addwith(u_for_q1,-1.0);
			z_v.copyData(v_for_G);
			z_v.Addwith(v_for_q1,-1.0);
			z_w.copyData(w_for_G);
			z_w.Addwith(w_for_q1,-1.0);

			funcF1(u_for_F1,v_for_F1,w_for_F1,warpIm2,Im1,Im2,z_u,z_v,z_w,opt);

			z_u.copyData(u_for_G);
			z_u.Addwith(u_for_q2,-1.0);
			z_v.copyData(v_for_G);
			z_v.Addwith(v_for_q2,-1.0);
			z_w.copyData(w_for_G);
			z_w.Addwith(w_for_q2,-1.0);

			funcF2(u_for_F2,v_for_F2,w_for_F2,z_u,z_v,z_w,pre_u,pre_v,pre_w,opt);

			z_u.copyData(u_for_F1);
			z_u.Addwith(u_for_q1,1.0);
			z_u.Addwith(u_for_F2);
			z_u.Addwith(u_for_q2,1.0);
			z_u.Multiplywith(0.5);

			z_v.copyData(v_for_F1);
			z_v.Addwith(v_for_q1,1.0);
			z_v.Addwith(v_for_F2);
			z_v.Addwith(v_for_q2,1.0);
			z_v.Multiplywith(0.5);

			z_w.copyData(w_for_F1);
			z_w.Addwith(w_for_q1,1.0);
			z_w.Addwith(w_for_F2);
			z_w.Addwith(w_for_q2,1.0);
			z_w.Multiplywith(0.5);

			funcG(u_for_G,v_for_G,w_for_G,z_u,z_v,z_w,opt);

			u_for_q1.Addwith(u_for_F1,1);
			u_for_q1.Addwith(u_for_G,-1);
			v_for_q1.Addwith(v_for_F1,1);
			v_for_q1.Addwith(v_for_G,-1);
			w_for_q1.Addwith(w_for_F1,1);
			w_for_q1.Addwith(w_for_G,-1);

			u_for_q2.Addwith(u_for_F2,1);
			u_for_q2.Addwith(u_for_G,-1);
			v_for_q2.Addwith(v_for_F2,1);
			v_for_q2.Addwith(v_for_G,-1);
			w_for_q2.Addwith(w_for_F2,1);
			w_for_q2.Addwith(w_for_G,-1);
		}
		u.copyData(u_for_F1);
		v.copyData(v_for_F1);
		w.copyData(w_for_F1);
	}


	template<class T>
	void ZQ_OpticalFlow3D::ADMM_F1_F2_G_middle(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_DImage3D<T>& pre_u, const ZQ_DImage3D<T>& pre_v, const ZQ_DImage3D<T>& pre_w, const ZQ_DImage3D<T>& next_u, const ZQ_DImage3D<T>& next_v, const ZQ_DImage3D<T>& next_w, const ZQ_OpticalFlowOptions& opt,
		void (*funcF1)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, ZQ_DImage3D<T>& /*warpIm2*/, const ZQ_DImage3D<T>& /*Im1*/, const ZQ_DImage3D<T>& /*Im2*/,  const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_OpticalFlowOptions& /*opt*/),
		void (*funcF2)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_DImage3D<T>& /*pre_u*/, const ZQ_DImage3D<T>& /*pre_v*/,const ZQ_DImage3D<T>& /*pre_w*/, const ZQ_DImage3D<T>& /*next_u*/, const ZQ_DImage3D<T>& /*next_v*/, const ZQ_DImage3D<T>& /*next_w*/, const ZQ_OpticalFlowOptions& /*opt*/),
		void (*funcG)(ZQ_DImage3D<T>& /*u*/, ZQ_DImage3D<T>& /*v*/, ZQ_DImage3D<T>& /*w*/, const ZQ_DImage3D<T>& /*z_u*/, const ZQ_DImage3D<T>& /*z_v*/, const ZQ_DImage3D<T>& /*z_w*/, const ZQ_OpticalFlowOptions& /*opt*/))
	{
		ZQ_DImage3D<T> u_for_F1(u),v_for_F1(v),w_for_F1(w);
		ZQ_DImage3D<T> u_for_F2(u),v_for_F2(v),w_for_F2(w);
		ZQ_DImage3D<T> u_for_G(u),v_for_G(v),w_for_G(w);
		ZQ_DImage3D<T> u_for_q1(u),v_for_q1(v),w_for_q1(w);
		ZQ_DImage3D<T> u_for_q2(u),v_for_q2(v),w_for_q2(w);
		ZQ_DImage3D<T> z_u,z_v,z_w;

		u_for_q1.reset(),v_for_q1.reset(),w_for_q1.reset();
		u_for_q2.reset(),v_for_q2.reset(),w_for_q2.reset();

		for(int it = 0;it < opt.nADMMIterations;it++)
		{
			z_u.copyData(u_for_G);
			z_u.Addwith(u_for_q1,-1.0);
			z_v.copyData(v_for_G);
			z_v.Addwith(v_for_q1,-1.0);
			z_w.copyData(w_for_G);
			z_w.Addwith(w_for_q1,-1.0);

			funcF1(u_for_F1,v_for_F1,w_for_F1,warpIm2,Im1,Im2,z_u,z_v,z_w,opt);

			z_u.copyData(u_for_G);
			z_u.Addwith(u_for_q2,-1.0);
			z_v.copyData(v_for_G);
			z_v.Addwith(v_for_q2,-1.0);
			z_w.copyData(w_for_G);
			z_w.Addwith(w_for_q2,-1.0);

			funcF2(u_for_F2,v_for_F2,w_for_F2,z_u,z_v,z_w,pre_u,pre_v,pre_w,next_u,next_v,next_w,opt);

			z_u.copyData(u_for_F1);
			z_u.Addwith(u_for_q1,1.0);
			z_u.Addwith(u_for_F2);
			z_u.Addwith(u_for_q2,1.0);
			z_u.Multiplywith(0.5);

			z_v.copyData(v_for_F1);
			z_v.Addwith(v_for_q1,1.0);
			z_v.Addwith(v_for_F2);
			z_v.Addwith(v_for_q2,1.0);
			z_v.Multiplywith(0.5);

			z_w.copyData(w_for_F1);
			z_w.Addwith(w_for_q1,1.0);
			z_w.Addwith(w_for_F2);
			z_w.Addwith(w_for_q2,1.0);
			z_w.Multiplywith(0.5);

			funcG(u_for_G,v_for_G,w_for_G,z_u,z_v,z_w,opt);

			u_for_q1.Addwith(u_for_F1,1);
			u_for_q1.Addwith(u_for_G,-1);
			v_for_q1.Addwith(v_for_F1,1);
			v_for_q1.Addwith(v_for_G,-1);
			w_for_q1.Addwith(w_for_F1,1);
			w_for_q1.Addwith(w_for_G,-1);

			u_for_q2.Addwith(u_for_F2,1);
			u_for_q2.Addwith(u_for_G,-1);
			v_for_q2.Addwith(v_for_F2,1);
			v_for_q2.Addwith(v_for_G,-1);
			w_for_q2.Addwith(w_for_F2,1);
			w_for_q2.Addwith(w_for_G,-1);
		}
		u.copyData(u_for_F1);
		v.copyData(v_for_F1);
		w.copyData(w_for_F1);
	}


	/****************************************************************************/
	template<class T>
	void ZQ_OpticalFlow3D::Proximal_F_L2(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_DImage3D<T>& z_u, const ZQ_DImage3D<T>& z_v, const ZQ_DImage3D<T>& z_w, const ZQ_OpticalFlowOptions& opt)
	{
		int imWidth = Im1.width();
		int imHeight = Im1.height();
		int imDepth = Im1.depth();
		int nChannels = Im1.nchannels();
		int nPixels = imWidth*imHeight*imDepth;

		ZQ_DImage3D<T> du(imWidth,imHeight,imDepth),dv(imWidth,imHeight,imDepth),dw(imWidth,imHeight,imDepth); //for du, dv


		/* ProximalF(z_u,z_v,\lambda) = minimize_{u,v} \int {|I_2(x+u,y+v)-I_1(x,y)|^2} + \alpha^2 \int {|\nabla u|^2 + |\nabla v|^2} + \beta^2 \int {|u|^2 + |v|^2} + \lambda \int {|u-z_u|^2 + |v-z_v|^2} 
		*
		* The Euler-Lagrange equation is:
		*  I_t I_x + \beta^2 u + \lambda(u-z_u) = \alpha^2 \Delta u 
		*  I_t I_y + \beta^2 v + \lambda(v-z_v) = \alpha^2 \Delta v
		*/


		warpFL(warpIm2,Im1,Im2,u,v,w,opt.useCubicWarping);

		/************       Outer Loop Begin     *************/
		//refresh {u,v} in each loop
		for(int count = 0; count < opt.nOuterFixedPointIterations;count++)
		{
			// outer loop : {imdx, imdy, imdt} accoring to warpIm2, warpIm2 accoring to {u,v} 
			ZQ_DImage3D<T> imdx,imdy,imdz,imdt;

			getDxs(imdx,imdy,imdz,imdt,Im1,warpIm2,true);

			du.reset();
			dv.reset();
			dw.reset();

			//compute imdtdx, imdxdx, imdydy, imdtdx, imdtdy
			ZQ_DImage3D<T> imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz;
			imdxdx.Multiply(imdx,imdx);
			imdxdy.Multiply(imdx,imdy);
			imdxdz.Multiply(imdx,imdz);
			imdydy.Multiply(imdy,imdy);
			imdydz.Multiply(imdy,imdz);
			imdzdz.Multiply(imdz,imdz);
			imdtdx.Multiply(imdx,imdt);
			imdtdy.Multiply(imdy,imdt);
			imdtdz.Multiply(imdz,imdt);

			if(nChannels>1)
			{
				imdxdx.collapse();
				imdxdy.collapse();
				imdxdz.collapse();
				imdydy.collapse();
				imdydz.collapse();
				imdzdz.collapse();
				imdtdx.collapse();
				imdtdy.collapse();
				imdtdz.collapse();
			}

			ZQ_DImage3D<T> laplace_u(imWidth,imHeight,imDepth);
			ZQ_DImage3D<T> laplace_v(imWidth,imHeight,imDepth);
			ZQ_DImage3D<T> laplace_w(imWidth,imHeight,imDepth);

			Laplacian(laplace_u,u);
			Laplacian(laplace_v,v);
			Laplacian(laplace_w,w);

			T*& laplace_uData = laplace_u.data();
			T*& laplace_vData = laplace_v.data();
			T*& laplace_wData = laplace_w.data();


			// set omega
			double omega = opt.omegaForSOR;
			double alpha2 = opt.alpha*opt.alpha;
			double beta2 = opt.beta*opt.beta;

			T*& duData = du.data();
			T*& dvData = dv.data();
			T*& dwData = dw.data();
			T*& uData = u.data();
			T*& vData = v.data();
			T*& wData = w.data();
			const T*& z_uData = z_u.data();
			const T*& z_vData = z_v.data();
			const T*& z_wData = z_w.data();
			T*& imdtdxData = imdtdx.data();
			T*& imdtdyData = imdtdy.data();
			T*& imdtdzData = imdtdz.data();
			T*& imdxdxData = imdxdx.data();
			T*& imdxdyData = imdxdy.data();
			T*& imdxdzData = imdxdz.data();
			T*& imdydyData = imdydy.data();
			T*& imdydzData = imdydz.data();
			T*& imdzdzData = imdzdz.data();



			/***   SOR Begin solve du,dv***/

			for(int sor_it = 0; sor_it < opt.nSORIterations; sor_it++)
			{
				for(int k = 0;k < imDepth;k++)
				{
					for(int j = 0; j<imHeight; j++)
					{
						for(int i = 0; i<imWidth; i++)
						{
							int offset = k*imWidth*imHeight+j*imWidth+i;
							double sigma1 = 0, sigma2 = 0, sigma3 = 0, coeff = 0;
							double _weight;

							if(i > 0)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset-1];
								sigma2 += _weight*dvData[offset-1];
								sigma3 += _weight*dwData[offset-1];
								coeff  += _weight;

							}
							if(i < imWidth-1)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset+1];
								sigma2 += _weight*dvData[offset+1];
								sigma3 += _weight*dwData[offset+1];
								coeff  += _weight;
							}
							if(j > 0)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset-imWidth];
								sigma2 += _weight*dvData[offset-imWidth];
								sigma3 += _weight*dwData[offset-imWidth];
								coeff  += _weight;
							}
							if(j < imHeight-1)
							{
								_weight = 1;
								sigma1  += _weight*duData[offset+imWidth];
								sigma2  += _weight*dvData[offset+imWidth];
								sigma3  += _weight*dwData[offset+imWidth];
								coeff   += _weight;
							}
							if(k > 0)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset-imWidth*imHeight];
								sigma2 += _weight*dvData[offset-imWidth*imHeight];
								sigma3 += _weight*dwData[offset-imWidth*imHeight];
								coeff  += _weight;
							}
							if(k < imDepth-1)
							{
								_weight = 1;
								sigma1 += _weight*duData[offset+imWidth*imHeight];
								sigma2 += _weight*dvData[offset+imWidth*imHeight];
								sigma3 += _weight*dwData[offset+imWidth*imHeight];
								coeff  += _weight;
							}
							sigma1 *= alpha2;
							sigma2 *= alpha2;
							sigma3 *= alpha2;
							coeff  *= alpha2;
							// compute u
							sigma1 += alpha2*(laplace_uData[offset]) - imdtdxData[offset] - imdxdyData[offset]*dvData[offset] - imdxdzData[offset]*dwData[offset] - beta2*uData[offset] - 0.5*opt.lambda*(uData[offset] - z_uData[offset]);
							double coeff1 = coeff + imdxdxData[offset] + beta2 + 0.5*opt.lambda;
							duData[offset] = (1-omega)*duData[offset] + omega/coeff1*sigma1;
							// compute v
							sigma2 += alpha2*(laplace_vData[offset]) - imdtdyData[offset] - imdxdyData[offset]*duData[offset] - imdydzData[offset]*dwData[offset] - beta2*vData[offset] - 0.5*opt.lambda*(vData[offset] - z_vData[offset]);
							double coeff2 = coeff + imdydyData[offset] + beta2 + 0.5*opt.lambda;
							dvData[offset] = (1-omega)*dvData[offset] + omega/coeff2*sigma2;
							// compute w
							sigma3 += alpha2*(laplace_wData[offset]) - imdtdzData[offset] - imdxdzData[offset]*duData[offset] - imdydzData[offset]*dvData[offset] - beta2*wData[offset] - 0.5*opt.lambda*(wData[offset] - z_wData[offset]);
							double coeff3 = coeff + imdzdzData[offset] + beta2 + 0.5*opt.lambda;
							dwData[offset] = (1-omega)*dwData[offset] + omega/coeff3*sigma3;

						}
					}

				}
				
			}

			/***   SOR end solve du,dv***/


			u.Addwith(du);
			v.Addwith(dv);
			w.Addwith(dw);

			warpFL(warpIm2,Im1,Im2,u,v,w,opt.useCubicWarping);

		}
		/************       Outer Loop End     *************/
	}

	template<class T>
	void ZQ_OpticalFlow3D::Proximal_F_DL1(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, ZQ_DImage3D<T>& warpIm2, const ZQ_DImage3D<T>& Im1, const ZQ_DImage3D<T>& Im2, const ZQ_DImage3D<T>& z_u, const ZQ_DImage3D<T>& z_v, const ZQ_DImage3D<T>& z_w, const ZQ_OpticalFlowOptions& opt)
	{
		ZQ_DImage3D<T> imdx,imdy,imdz,imdt;

		int imWidth = Im1.width();
		int imHeight = Im1.height();
		int imDepth = Im1.depth();
		int nChannels = Im1.nchannels();
		int nPixels = imWidth*imHeight*imDepth;

		ZQ_DImage3D<T> du(imWidth,imHeight,imDepth),dv(imWidth,imHeight,imDepth),dw(imWidth,imHeight,imDepth);

		ZQ_DImage3D<T> Psi_1st(imWidth,imHeight,imDepth,nChannels);

		ZQ_DImage3D<T> imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz;

		ZQ_DImage3D<T> laplace_u,laplace_v,laplace_w;

		double varepsilon_phi = pow(0.001,2);
		double varepsilon_psi = pow(0.001,2);

		warpFL(warpIm2,Im1,Im2,u,v,w,opt.useCubicWarping);


		for(int out_it = 0;out_it < opt.nOuterFixedPointIterations;out_it++)
		{
			getDxs(imdx,imdy,imdz,imdt,Im1,warpIm2,true);

			du.reset();
			dv.reset();
			dw.reset();

			for(int in_it = 0; in_it < opt.nInnerFixedPointIterations;in_it++)
			{
				Psi_1st.reset();

				T*& psiData = Psi_1st.data();
				T*& imdxData = imdx.data();
				T*& imdyData = imdy.data();
				T*& imdzData = imdz.data();
				T*& imdtData = imdt.data();
				T*& duData = du.data();
				T*& dvData = dv.data();
				T*& dwData = dw.data();
				T*& uData = u.data();
				T*& vData = v.data();
				T*& wData = w.data();


				for(int i = 0;i < nPixels;i++)
				{
					for(int c = 0;c < nChannels;c++)
					{
						int offset = i*nChannels+c;
						double temp = imdtData[offset]+imdxData[offset]*duData[i]+imdyData[offset]*dvData[i]+imdzData[offset]*dwData[i];

						temp *= temp;
						psiData[offset]=1/(2*sqrt(temp+varepsilon_psi));
					}
				}

				imdxdx.Multiply(Psi_1st,imdx,imdx);
				imdxdy.Multiply(Psi_1st,imdx,imdy);
				imdxdz.Multiply(Psi_1st,imdx,imdz);
				imdydy.Multiply(Psi_1st,imdy,imdy);
				imdydz.Multiply(Psi_1st,imdy,imdz);
				imdzdz.Multiply(Psi_1st,imdz,imdz);
				imdtdx.Multiply(Psi_1st,imdx,imdt);
				imdtdy.Multiply(Psi_1st,imdy,imdt);
				imdtdz.Multiply(Psi_1st,imdz,imdt);

				if(nChannels > 1)
				{
					imdxdx.collapse();
					imdxdy.collapse();
					imdxdz.collapse();
					imdydy.collapse();
					imdydz.collapse();
					imdzdz.collapse();
					imdtdx.collapse();
					imdtdy.collapse();
					imdtdz.collapse();
				}

				T*& imdxdxData = imdxdx.data();
				T*& imdxdyData = imdxdy.data();
				T*& imdxdzData = imdxdz.data();
				T*& imdydyData = imdydy.data();
				T*& imdydzData = imdydz.data();
				T*& imdzdzData = imdzdz.data();
				T*& imdtdxData = imdtdx.data();
				T*& imdtdyData = imdtdy.data();
				T*& imdtdzData = imdtdz.data();
				const T*& z_uData = z_u.data();
				const T*& z_vData = z_v.data();
				const T*& z_wData = z_w.data();


				Laplacian(laplace_u,u);
				Laplacian(laplace_v,v);
				Laplacian(laplace_w,w);

				T*& laplace_uData = laplace_u.data();
				T*& laplace_vData = laplace_v.data();
				T*& laplace_wData = laplace_w.data();

				double omega = opt.omegaForSOR;
				double alpha2 = opt.alpha*opt.alpha;
				double beta2 = opt.beta*opt.beta;

				/*Begin SOR*/

				for(int sor_it = 0; sor_it < opt.nSORIterations; sor_it++)
				{
					for(int k = 0;k < imDepth;k++)
					{
						for(int j = 0; j < imHeight; j++)
						{
							for(int i = 0; i < imWidth; i++)
							{
								int offset = k*imWidth*imHeight+j*imWidth+i;
								double sigma1 = 0, sigma2 = 0, sigma3 = 0, coeff = 0;
								double _weight;


								if(i > 0)
								{
									_weight = 1;
									sigma1 += _weight*duData[offset-1];
									sigma2 += _weight*dvData[offset-1];
									sigma3 += _weight*dwData[offset-1];
									coeff  += _weight;
								}
								if(i < imWidth-1)
								{
									_weight = 1;
									sigma1 += _weight*duData[offset+1];
									sigma2 += _weight*dvData[offset+1];
									sigma3 += _weight*dwData[offset+1];
									coeff  += _weight;
								}
								if(j > 0)
								{
									_weight = 1;
									sigma1 += _weight*duData[offset-imWidth];
									sigma2 += _weight*dvData[offset-imWidth];
									sigma3 += _weight*dwData[offset-imWidth];
									coeff  += _weight;
								}
								if(j < imHeight-1)
								{
									_weight = 1;
									sigma1 += _weight*duData[offset+imWidth];
									sigma2 += _weight*dvData[offset+imWidth];
									sigma3 += _weight*dwData[offset+imWidth];
									coeff  += _weight;
								}
								if(k > 0)
								{
									_weight = 1;
									sigma1 += _weight*duData[offset-imWidth*imHeight];
									sigma2 += _weight*dvData[offset-imWidth*imHeight];
									sigma3 += _weight*dwData[offset-imWidth*imHeight];
									coeff  += _weight;
								}
								if(k < imDepth-1)
								{
									_weight = 1;
									sigma1 += _weight*duData[offset+imWidth*imHeight];
									sigma2 += _weight*dvData[offset+imWidth*imHeight];
									sigma3 += _weight*dwData[offset+imWidth*imHeight];
									coeff  += _weight;
								}
								sigma1 *= alpha2;
								sigma2 *= alpha2;
								sigma3 *= alpha2;
								coeff  *= alpha2;

								// compute u
								sigma1 += alpha2*(laplace_uData[offset]) - imdtdxData[offset] - imdxdyData[offset]*dvData[offset] - imdxdzData[offset]*dwData[offset] - beta2*uData[offset] - 0.5*opt.lambda*(uData[offset] - z_uData[offset]);
								double coeff1 = coeff + imdxdxData[offset] + beta2 + 0.5*opt.lambda;
								duData[offset] = (1-omega)*duData[offset] + omega/coeff1*sigma1;
								// compute v
								sigma2 += alpha2*(laplace_vData[offset]) - imdtdyData[offset] - imdxdyData[offset]*duData[offset] - imdydzData[offset]*dwData[offset] - beta2*vData[offset] - 0.5*opt.lambda*(vData[offset] - z_vData[offset]);
								double coeff2 = coeff + imdydyData[offset] + beta2 + 0.5*opt.lambda;
								dvData[offset] = (1-omega)*dvData[offset] + omega/coeff2*sigma2;

								// compute w
								sigma3 += alpha2*(laplace_wData[offset]) - imdtdzData[offset] - imdxdzData[offset]*duData[offset] - imdydzData[offset]*dvData[offset] - beta2*wData[offset] - 0.5*opt.lambda*(wData[offset] - z_wData[offset]);
								double coeff3 = coeff + imdzdzData[offset] + beta2 + 0.5*opt.lambda;
								dwData[offset] = (1-omega)*dwData[offset] + omega/coeff3*sigma3;
							}
						}

					}
					
				}
				/*End SOR*/
			}
			u.Addwith(du);
			v.Addwith(dv);
			w.Addwith(dw);

			warpFL(warpIm2,Im1,Im2,u,v,w,opt.useCubicWarping);
		}
	}

	template<class T>
	void ZQ_OpticalFlow3D::Proximal_G(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, const ZQ_DImage3D<T>& z_u, const ZQ_DImage3D<T>& z_v, const ZQ_DImage3D<T>& z_w, const ZQ_OpticalFlowOptions& opt)
	{
		u.copyData(z_u);
		v.copyData(z_v);
		w.copyData(z_w);

		int width = u.width();
		int height = u.height();
		int depth = u.depth();

		ZQ_PoissonSolver3D::SolveOpenPoissonSOR(u.data(),v.data(),w.data(),width,height,depth,opt.nPoissonIterations,opt.displayRunningInfo);
	}

	template<class T>
	void ZQ_OpticalFlow3D::Proximal_F2_First(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, const ZQ_DImage3D<T>& z_u, const ZQ_DImage3D<T>& z_v, const ZQ_DImage3D<T>& z_w, const ZQ_DImage3D<T>& next_u, const ZQ_DImage3D<T>& next_v, const ZQ_DImage3D<T>& next_w, const ZQ_OpticalFlowOptions& opt)
	{
		int imWidth = u.width();
		int imHeight = u.height();
		int imDepth = u.depth();

		ZQ_DImage3D<T> warpU,warpV,warpW;

		double gamma = opt.gamma * opt.alpha * opt.alpha;

		for(int out_it = 0;out_it < opt.nAdvectFixedPointIterations;out_it++)
		{
			warpFL(warpU,u,next_u,u,v,w,opt.useCubicWarping);
			warpFL(warpV,v,next_v,u,v,w,opt.useCubicWarping);
			warpFL(warpW,w,next_w,u,v,w,opt.useCubicWarping);
			ZQ_PoissonSolver3D::SolveOpenPoissonSOR(warpU.data(),warpV.data(),warpW.data(),imWidth,imHeight,imDepth,opt.nPoissonIterations);

			T*& warpUData = warpU.data();
			T*& warpVData = warpV.data();
			T*& warpWData = warpW.data();

			T*& uData = u.data();
			T*& vData = v.data();
			T*& wData = w.data();
			const T*& z_uData = z_u.data();
			const T*& z_vData = z_v.data();
			const T*& z_wData = z_w.data();

			for(int k = 0;k < imDepth;k++)
			{
				for(int j = 0;j < imHeight;j++)
				{
					for(int i = 0;i < imWidth;i++)
					{
						int offset = k*imWidth*imHeight+j*imWidth+i;
						uData[offset] = (gamma*warpUData[offset]+opt.lambda*z_uData[offset])/(gamma+opt.lambda);
						vData[offset] = (gamma*warpVData[offset]+opt.lambda*z_vData[offset])/(gamma+opt.lambda);
						wData[offset] = (gamma*warpWData[offset]+opt.lambda*z_wData[offset])/(gamma+opt.lambda);
					}
				}
			}
		}
	}

	template<class T>
	void ZQ_OpticalFlow3D::Proximal_F2_Last(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, const ZQ_DImage3D<T>& z_u, const ZQ_DImage3D<T>& z_v, const ZQ_DImage3D<T>& z_w, const ZQ_DImage3D<T>& pre_u, const ZQ_DImage3D<T>& pre_v, const ZQ_DImage3D<T>& pre_w, const ZQ_OpticalFlowOptions& opt)
	{
		int imWidth = u.width();
		int imHeight = u.height();
		int imDepth = u.depth();

		ZQ_DImage3D<T> warpU,warpV,warpW;

		double gamma = opt.gamma * opt.alpha * opt.alpha;

		for(int out_it = 0;out_it < opt.nAdvectFixedPointIterations;out_it++)
		{
			ZQ_DImage3D<T> tmp_u(u),tmp_v(v),tmp_w(w);
			tmp_u.Multiplywith(-1);
			tmp_v.Multiplywith(-1);
			tmp_w.Multiplywith(-1);

			warpFL(warpU,u,pre_u,tmp_u,tmp_v,tmp_w,opt.useCubicWarping);
			warpFL(warpV,v,pre_v,tmp_u,tmp_v,tmp_w,opt.useCubicWarping);
			warpFL(warpW,w,pre_w,tmp_u,tmp_v,tmp_w,opt.useCubicWarping);

			ZQ_PoissonSolver3D::SolveOpenPoissonSOR(warpU.data(),warpV.data(),warpW.data(),imWidth,imHeight,imDepth,opt.nPoissonIterations);

			T*& warpUData = warpU.data();
			T*& warpVData = warpV.data();
			T*& warpWData = warpW.data();

			T*& uData = u.data();
			T*& vData = v.data();
			T*& wData = w.data();
			const T*& z_uData = z_u.data();
			const T*& z_vData = z_v.data();
			const T*& z_wData = z_w.data();

			for(int k = 0;k < imDepth;k++)
			{
				for(int j = 0;j < imHeight;j++)
				{
					for(int i = 0;i < imWidth;i++)
					{
						int offset = k*imWidth*imHeight+j*imWidth+i;
						uData[offset] = (gamma*warpUData[offset]+opt.lambda*z_uData[offset])/(gamma+opt.lambda);
						vData[offset] = (gamma*warpVData[offset]+opt.lambda*z_vData[offset])/(gamma+opt.lambda);
						wData[offset] = (gamma*warpWData[offset]+opt.lambda*z_wData[offset])/(gamma+opt.lambda);
					}
				}
			}
			
		}
	}

	template<class T>
	void ZQ_OpticalFlow3D::Proximal_F2_Middle(ZQ_DImage3D<T>& u, ZQ_DImage3D<T>& v, ZQ_DImage3D<T>& w, const ZQ_DImage3D<T>& z_u, const ZQ_DImage3D<T>& z_v, const ZQ_DImage3D<T>& z_w, 
									const ZQ_DImage3D<T>& pre_u, const ZQ_DImage3D<T>& pre_v, const ZQ_DImage3D<T>& pre_w, const ZQ_DImage3D<T>& next_u, const ZQ_DImage3D<T>& next_v, const ZQ_DImage3D<T>& next_w, const ZQ_OpticalFlowOptions& opt)
	{
		int imWidth = u.width();
		int imHeight = u.height();
		int imDepth = u.depth();

		ZQ_DImage3D<T> warpU_pre,warpV_pre,warpW_pre;
		ZQ_DImage3D<T> warpU_nex,warpV_nex,warpW_nex;

		double gamma = opt.gamma * opt.alpha * opt.alpha;

		for(int out_it = 0;out_it < opt.nAdvectFixedPointIterations;out_it++)
		{
			ZQ_DImage3D<T> tmp_u(u),tmp_v(v),tmp_w(w);
			tmp_u.Multiplywith(-1);
			tmp_v.Multiplywith(-1);
			tmp_w.Multiplywith(-1);

			warpFL(warpU_pre,u,pre_u,tmp_u,tmp_v,tmp_w,opt.useCubicWarping);
			warpFL(warpV_pre,v,pre_v,tmp_u,tmp_v,tmp_w,opt.useCubicWarping);
			warpFL(warpW_pre,w,pre_w,tmp_u,tmp_v,tmp_w,opt.useCubicWarping);

			warpFL(warpU_nex,u,next_u,u,v,w,opt.useCubicWarping);
			warpFL(warpV_nex,v,next_v,u,v,w,opt.useCubicWarping);
			warpFL(warpW_nex,w,next_w,u,v,w,opt.useCubicWarping);

			ZQ_DImage3D<T> warpU_sum,warpV_sum,warpW_sum;
			warpU_sum.Add(warpU_pre,warpU_nex);
			warpV_sum.Add(warpV_pre,warpV_nex);
			warpW_sum.Add(warpW_pre,warpW_nex);

			ZQ_PoissonSolver3D::SolveOpenPoissonSOR(warpU_sum.data(),warpV_sum.data(),warpW_sum.data(),imWidth,imHeight,imDepth,opt.nPoissonIterations);

			T*& warpU_sumData = warpU_sum.data();
			T*& warpV_sumData = warpV_sum.data();
			T*& warpW_sumData = warpW_sum.data();

			T*& uData = u.data();
			T*& vData = v.data();
			T*& wData = w.data();
			const T*& z_uData = z_u.data();
			const T*& z_vData = z_v.data();
			const T*& z_wData = z_w.data();

			for(int k = 0;k < imDepth;k++)
			{
				for(int j = 0;j < imHeight;j++)
				{
					for(int i = 0;i < imWidth;i++)
					{
						int offset = k*imWidth*imHeight+j*imWidth+i;
						uData[offset] = (gamma*warpU_sumData[offset]+opt.lambda*z_uData[offset])/(2*gamma+opt.lambda);
						vData[offset] = (gamma*warpV_sumData[offset]+opt.lambda*z_vData[offset])/(2*gamma+opt.lambda);
						wData[offset] = (gamma*warpW_sumData[offset]+opt.lambda*z_wData[offset])/(2*gamma+opt.lambda);
					}
				}
			}
		}
	}

}


#endif