#ifndef _ZQ_OPTICAL_FLOW_OPTIONS_H_
#define _ZQ_OPTICAL_FLOW_OPTIONS_H_
#pragma once

namespace ZQ
{
	class ZQ_OpticalFlowOptions
	{
		enum CONST_VAL{FILE_BUFF_LEN = 200};
	public:
		enum MethodType {
			METHOD_HS_L2,			//Horn-Schunck full L2 penalty
			METHOD_HS_L2_OCCUPY,
			METHOD_HS_L1,			//Horn-Schunck full eps-L1 penalty
			METHOD_HS_DL1,			//Horn-Schunck Data-term eps-L1 penalty
			METHOD_ADMM_L2,			//Div-Free using ADMM method, full L2 penalty
			METHOD_ADMM_L2_OCCUPY,
			METHOD_ADMM_DL1,		//Div-Free using ADMM method, Data-term eps-L1 penalty

			//all the following methods (ONEDIR and TWODIR ) use ADMM optimization method

			// one direction spatial-temporal methods regarding Navier-Stokes equations
			METHOD_ONEDIR_INC_L2,	//from the first frame to the last
			METHOD_ONEDIR_INC_DL1,	//from the first frame to the last
			METHOD_ONEDIR_DEC_L2,	//from the last frame to the first
			METHOD_ONEDIR_DEC_DL1,	//from the last frame to the first

			// two directions spatial-temporal methods regarding Navier-Stokes equations
			METHOD_TWODIR_INC_L2,	//from the first frame to the last
			METHOD_TWODIR_INC_L2_OCCUPY,
			METHOD_TWODIR_INC_DL1,	//from the first frame to the last
			METHOD_TWODIR_DEC_L2,	//from the last frame to the first
			METHOD_TWODIR_DEC_L2_OCCUPY,
			METHOD_TWODIR_DEC_DL1	//from the last frame to the first
		};

		enum FeatureType{
			FEATURE_GRADIENT,
			FEATURE_FORWARD_NEIGHBOR,
			FEATURE_BIDIRECTIONAL_NEIGHBOR
		};

		enum InitType{
			NONE_AS_INIT,			
			L2_AS_INIT,
			ADMM_AS_INIT
		};

		MethodType methodType;
		FeatureType featureType;
		double gradWeight;
		int nAlterations;					// for TWODIR methods
		int nADMMIterations;				// for methods using ADMM optimization (TWODIR, ONEDIR, ADMM)
		double lambda;						// for ADMM optimization
		int nOuterFixedPointIterations;		// for all methods
		int nInnerFixedPointIterations;		// for L1 penalty (full L1 or Data-term L1)
		int nSORIterations;					// the kernel solver is using SOR
		double omegaForSOR;					// the omega value for SOR, 1<= omega < 2
		int nPoissonIterations;				// for methods regarding Navier-Stokes Div-Free constraint
		int nAdvectFixedPointIterations;	// for methods regarding Navier-Stokes advections
		bool useCubicWarping;				
		double ratioForPyramid;
		double minWidthForPyramid;
		double alpha;						// weighting for smoothing-term (||\nabla \mathbf{u}||), when using L2 penalty on this term, the real weighting is alpha*alpha
		double beta;						// weighting for kenimatic-term (||\mathbf{u}||),        when using L2 penalty on this term, the real weighting is beta*beta
		double gamma;						// weighting for temporal coherence term, real weighing is : alpha*alpha*gamma for L2 
		double maxRad;
		bool displayRunningInfo;		
		char maskFile[FILE_BUFF_LEN];
		bool hasMaxUpdateLimit;

		//other paras
		bool isReflect;					//for TWODIR methods
		InitType initType;				//for ONEDIR and TWODIR methods
		int cudaDeviceID;				//for CUDA 
		bool OutOfCore;					//for out of core
		bool use_omp;					// for OpenMP

		// for occlusion detect
		int weightedMedFiltIter_for_occ_detect;	//occlusion detect, Iterations
		double sigma_div_for_occ_detect;		//occlusion detect, sigma divergence
		double sigma_im_for_occ_detect;			//occlusion detect, sigma image intensity
		double sigma_im_for_denoise;			//occlusion detect and denoise, sigma image intensity
		int weightedMedFiltSize_for_denoise;	//occlusion detect and denoise, weighted median filter half size
		int medFiltSize_for_denoise;			//occulison detect and denoise, median filter half size

		ZQ_OpticalFlowOptions()
		{
			Reset();
		}

		static void GetDefaultOptions_HS_L1(ZQ_OpticalFlowOptions& opt)
		{
			opt.methodType = METHOD_HS_L1;
			opt.featureType = FEATURE_GRADIENT;
			opt.gradWeight = 1;
			opt.nOuterFixedPointIterations = 5;
			opt.nInnerFixedPointIterations = 3;
			opt.nSORIterations = 100;
			opt.omegaForSOR = 1.0;
			opt.nAdvectFixedPointIterations = 3;
			opt.useCubicWarping = false;
			opt.ratioForPyramid = 0.5;
			opt.minWidthForPyramid = 16;
			opt.alpha = 0.012;
			opt.beta = 0;
			opt.displayRunningInfo = false;
			
			opt.hasMaxUpdateLimit = true;
			opt.weightedMedFiltIter_for_occ_detect = 1;
			opt.sigma_div_for_occ_detect = 0.3;
			opt.sigma_im_for_occ_detect = 20.0 / 255;
			opt.sigma_im_for_denoise = 7.0 / 255.0;
			opt.weightedMedFiltSize_for_denoise = 7;
			opt.medFiltSize_for_denoise = 2;
		}

		void Reset()
		{
			methodType = METHOD_HS_L2;
			featureType = FEATURE_GRADIENT;
			gradWeight = 1;
			nAlterations = 5;
			nADMMIterations = 20;
			lambda = 0.005;
			nOuterFixedPointIterations = 10;
			nInnerFixedPointIterations = 3;
			nSORIterations = 30;
			omegaForSOR = 1.0;
			nPoissonIterations = 50;
			nAdvectFixedPointIterations = 3;
			useCubicWarping = false;
			ratioForPyramid = 0.5;
			minWidthForPyramid = 16;
			alpha = 0.06;
			beta = 0;
			gamma = 0;
			maxRad = 12;
			displayRunningInfo = false;
			maskFile[0] = '\0';
			hasMaxUpdateLimit = false;
			isReflect = false;
			initType = ADMM_AS_INIT;
			cudaDeviceID = 0;
			OutOfCore = false;
			use_omp = false;

			weightedMedFiltIter_for_occ_detect = 0;	
			sigma_div_for_occ_detect = 0.3;		
			sigma_im_for_occ_detect = 0.08;			
			sigma_im_for_denoise = 0.03;			
			weightedMedFiltSize_for_denoise = 5;	
			medFiltSize_for_denoise = 0;
		}

		bool HandleParas(int argc, const char** argv)
		{
			for(int i = 0;i < argc;i++)
			{
				if(_strcmpi(argv[i],"methodtype") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					if(_strcmpi(argv[i],"HS_L2") == 0)
						methodType = METHOD_HS_L2;
					else if(_strcmpi(argv[i],"HS_L2_OCCUPY") == 0)
						methodType = METHOD_HS_L2_OCCUPY;
					else if(_strcmpi(argv[i],"HS_DL1") == 0)
						methodType = METHOD_HS_DL1;
					else if(_strcmpi(argv[i],"HS_L1") == 0)
						methodType = METHOD_HS_L1;
					else if(_strcmpi(argv[i],"ADMM_L2") == 0)
						methodType = METHOD_ADMM_L2;
					else if(_strcmpi(argv[i],"ADMM_L2_OCCUPY") == 0)
						methodType = METHOD_ADMM_L2_OCCUPY;
					else if(_strcmpi(argv[i],"ADMM_DL1") == 0)
						methodType = METHOD_ADMM_DL1;
					else if(_strcmpi(argv[i],"ONEDIR_INC_L2") == 0)
						methodType = METHOD_ONEDIR_INC_L2;
					else if(_strcmpi(argv[i],"ONEDIR_INC_DL1") == 0)
						methodType = METHOD_ONEDIR_INC_DL1;
					else if(_strcmpi(argv[i],"ONEDIR_DEC_L2") == 0)
						methodType = METHOD_ONEDIR_DEC_L2;
					else if(_strcmpi(argv[i],"ONEDIR_DEC_DL1") == 0)
						methodType = METHOD_ONEDIR_DEC_DL1;
					else if(_strcmpi(argv[i],"TWODIR_INC_L2") == 0)
						methodType = METHOD_TWODIR_INC_L2;
					else if(_strcmpi(argv[i],"TWODIR_INC_L2_OCCUPY") == 0)
						methodType = METHOD_TWODIR_INC_L2_OCCUPY;
					else if(_strcmpi(argv[i],"TWODIR_INC_DL1") == 0)
						methodType = METHOD_TWODIR_INC_DL1;
					else if(_strcmpi(argv[i],"TWODIR_DEC_L2") == 0)
						methodType = METHOD_TWODIR_DEC_L2;
					else if(_strcmpi(argv[i],"TWODIR_DEC_L2_OCCUPY") == 0)
						methodType = METHOD_TWODIR_DEC_L2_OCCUPY;
					else if(_strcmpi(argv[i],"TWODIR_DEC_DL1") == 0)
						methodType = METHOD_TWODIR_DEC_DL1;
					else
					{
						printf("unknown methodType: %s\n",argv[i]);
						return false;
					}
				}
				else if(_strcmpi(argv[i],"featuretype") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					if(_strcmpi(argv[i],"GRADIENT") == 0)
						featureType = FEATURE_GRADIENT;
					else if(_strcmpi(argv[i],"FORWARD_NEIGHBOR") == 0)
						featureType = FEATURE_FORWARD_NEIGHBOR;
					else if(_strcmpi(argv[i],"BIDIRECTIONAL_NEIGHBOR") == 0)
						featureType = FEATURE_BIDIRECTIONAL_NEIGHBOR;
					else
					{
						printf("unknown methodType: %s\n",argv[i]);
						return false;
					}

				}
				else if (_strcmpi(argv[i], "gradWeight") == 0)
				{
					i++;
					if (i >= argc)
					{
						printf("the value of %s ?\n", argv[i - 1]);
						return false;
					}
					gradWeight = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"nAltIter") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}

					nAlterations = atoi(argv[i]);
				}
				else if(_strcmpi(argv[i],"nAdmmIter") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					nADMMIterations = atoi(argv[i]);
				}
				else if(_strcmpi(argv[i],"nOuterIter") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					nOuterFixedPointIterations = atoi(argv[i]);
				}
				else if(_strcmpi(argv[i],"nInnerIter") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					nInnerFixedPointIterations = atoi(argv[i]);
				}
				else if(_strcmpi(argv[i],"nPoissonIter") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					nPoissonIterations = atoi(argv[i]);
				}
				else if(_strcmpi(argv[i],"nSORIter") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					nSORIterations = atoi(argv[i]);
				}
				else if(_strcmpi(argv[i],"nAdvectIter") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					nAdvectFixedPointIterations = atoi(argv[i]);
				}
				else if(_strcmpi(argv[i],"alpha") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					alpha = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"beta") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					beta = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"gamma") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					gamma = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"lambda") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					lambda = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"omega") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					omegaForSOR = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"ratio") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					ratioForPyramid = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"minWidth") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					minWidthForPyramid = atoi(argv[i]);
				}
				else if(_strcmpi(argv[i],"maxRad") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					maxRad = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"cubic") == 0)
				{
					useCubicWarping = true;
				}
				else if(_strcmpi(argv[i],"display") == 0)
				{
					displayRunningInfo = true;
				}
				else if(_strcmpi(argv[i],"maskFile") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					strcpy_s(maskFile, FILE_BUFF_LEN, argv[i]);
				}
				else if (_strcmpi(argv[i], "hasMaxUpdateLimit") == 0)
				{
					hasMaxUpdateLimit = true;
				}
				else if(_strcmpi(argv[i],"reflect") == 0)
				{
					isReflect = true;
				}
				else if(_strcmpi(argv[i],"initType") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					if(_strcmpi(argv[i],"NONE_AS_INIT") == 0)
						initType = NONE_AS_INIT;
					else if(_strcmpi(argv[i],"L2_AS_INIT") == 0)
						initType = L2_AS_INIT;
					else if(_strcmpi(argv[i],"ADMM_AS_INIT") == 0)
						initType = ADMM_AS_INIT;
					else
					{
						printf("unknown initType: %s\n",argv[i]);
						return false;
					}
				}
				else if(_strcmpi(argv[i],"CUDADeviceID") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					cudaDeviceID = atoi(argv[i]);
				}
				else if(_strcmpi(argv[i],"OutOfCore") == 0)
				{
					OutOfCore = true;
				}
				else if (_strcmpi(argv[i], "useOMP") == 0)
				{
					use_omp = true;
				}
				else
				{
					printf("unknown para: %s\n",argv[i]);
					return false;
				}
			}
			return true;
		}
	};
}
#endif