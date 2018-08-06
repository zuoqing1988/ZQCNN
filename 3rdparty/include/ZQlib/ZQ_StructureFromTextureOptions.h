#ifndef _ZQ_STRUCTURE_FROM_TEXTURE_OPTIONS_H_
#define _ZQ_STRUCTURE_FROM_TEXTURE_OPTIONS_H_

namespace ZQ
{
	class ZQ_StructureFromTextureOptions
	{
	public:
		enum MethodType
		{
			TYPE_TVL1,
			TYPE_TVL2,
			TYPE_RTVL1_OLD,
			TYPE_RTVL1,
			TYPE_RTVL2,
			TYPE_PENALTY_GRADIENT_WEIGHT
		};
		enum PenaltyGradientWeight
		{
			WEIGHT_RTV,
			WEIGHT_RTV_MIX,
			WEIGHT_WLS,
			WEIGHT_WLS_MIX
		};

		MethodType type;
		PenaltyGradientWeight penaltyWeightType;
		float epsilon;
		float epsilon_for_s;
		float sigma_for_filter;
		float fsize_for_filter;
		float norm_for_rtv;
		float norm_for_wls;
		float norm_for_dataterm;
		float epsilon_for_d;
		float weight;
		int nOuterIteration;
		int nSolverIteration;

		ZQ_StructureFromTextureOptions()
		{
			Reset();
		}
		void Reset()
		{
			type = TYPE_RTVL1;
			penaltyWeightType = WEIGHT_RTV;
			epsilon = 1e-3;
			epsilon_for_s = 2*1e-2;
			sigma_for_filter = 1;
			fsize_for_filter = 2;
			weight = 0.003;
			norm_for_rtv = 1;
			norm_for_wls = 1.2;
			norm_for_dataterm = 2;
			epsilon_for_d = 1e-6;
			nOuterIteration = 5;
			nSolverIteration = 10;
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
					if(_strcmpi(argv[i],"RTVL1_OLD") == 0)
						type = TYPE_RTVL1_OLD;
					else if(_strcmpi(argv[i],"RTVL1") == 0)
						type = TYPE_RTVL1;
					else if(_strcmpi(argv[i],"RTVL2") == 0)
						type = TYPE_RTVL2;
					else if(_strcmpi(argv[i],"TVL1") == 0)
						type = TYPE_TVL1;
					else if(_strcmpi(argv[i],"TVL2") == 0)
						type = TYPE_TVL2;
					else if(_strcmpi(argv[i],"PENALTY_GRADIENT_WEIGHT") == 0)
						type = TYPE_PENALTY_GRADIENT_WEIGHT;
					else
					{
						printf("unknown methodType: %s\n",argv[i]);
						return false;
					}

				}
				else if(_strcmpi(argv[i],"penaltyWeightType") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					if(_strcmpi(argv[i],"WEIGHT_RTV") == 0)
						penaltyWeightType = WEIGHT_RTV;
					else if(_strcmpi(argv[i],"WEIGHT_RTV_MIX") == 0)
						penaltyWeightType = WEIGHT_RTV_MIX;
					else if(_strcmpi(argv[i],"WEIGHT_WLS") == 0)
						penaltyWeightType = WEIGHT_WLS;
					else if(_strcmpi(argv[i],"WEIGHT_WLS_MIX") == 0)
						penaltyWeightType = WEIGHT_WLS_MIX;
					else
					{
						printf("unknown penaltyWeightType: %s\n",argv[i]);
						return false;
					}
				}
				else if(_strcmpi(argv[i],"epsilon") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					epsilon = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"epsilon_s") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					epsilon_for_s = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"epsilon_d") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					epsilon_for_d = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"sigma") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					sigma_for_filter = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"fsize") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					fsize_for_filter = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"weight") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					weight = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"norm_for_dataterm") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					norm_for_dataterm = atof(argv[i]);

				}
				else if(_strcmpi(argv[i],"norm_for_wls") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					norm_for_wls = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"norm_for_rtv") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					norm_for_rtv = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"nOuterIteration") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}

					nOuterIteration = atoi(argv[i]);
				}
				else if(_strcmpi(argv[i],"nSolverIteration") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					nSolverIteration = atoi(argv[i]);
				}
				else
				{
					printf("unknown para: %s\n",argv[i]);
					return false;
				}
			}
			return true;
		}

		void showArgs()
		{
			printf(
				"[\"methodtype\" \t \"TVL1\", \"TVL2\", \"RTVL1_OLD\", \"RTVL1\" or \"RTVL2\"]\n"
				"[\"penaltyWeightType\" \t \"WEIGHT_RTV\", \"WEIGHT_RTV_MIX\", \"WEIGHT_WLS\", \"WEIGHT_WLS_MIX\"]\n"
				"[\"epsilon\" \t epsilon]\n"
				"[\"epsilon_s\" \t epsilon_s]\n"
				"[\"epsilon_d\" \t epsilon_d]\n"
				"[\"norm_for_rtv\" \t norm_for_rtv]\n"
				"[\"norm_for_wls\" \t norm_for_wls]\n"
				"[\"norm_for_dataterm\" \t norm_for_dataterm]\n"
				"[\"sigma\" \t sigma]\n"
				"[\"fsize\" \t fsize]\n"
				"[\"weight\" \t weight]\n"
				"[\"nOuterIteration\" \t nOuterIteration]\n"
				"[\"nSolverIteration\" \t nSolverIteration]\n"
				);
		}
	};
}

#endif