#ifndef _ZQ_POISSON_EDITING_OPTIONS_H_
#define _ZQ_POISSON_EDITING_OPTIONS_H_
#pragma once

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

namespace ZQ
{
	class ZQ_PoissonEditingOptions
	{
	public:
		enum MethodType{
			METHOD_NAIVE,
			METHOD_MIXGRADIENT,
			METHOD_MAXGRADIENT,
			METHOD_ANISOTROPIC
		};
	public:
		ZQ_PoissonEditingOptions() {Reset();}
		~ZQ_PoissonEditingOptions() {}

		MethodType type;

		int nSORIteration;  
		float weight1;
		float grad_scale;
		float anisotropic_ratio;
		bool display;

	public:
		void Reset()
		{
			type = METHOD_NAIVE;
			nSORIteration = 500;
			weight1 = 0.5;
			grad_scale = 1.0f;
			anisotropic_ratio = 1.0f;
			display = false;
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
					if(_strcmpi(argv[i],"naive") == 0)
						type = METHOD_NAIVE;
					else if(_strcmpi(argv[i],"mixgradient") == 0)
						type = METHOD_MIXGRADIENT;
					else if(_strcmpi(argv[i],"maxgradient") == 0)
						type = METHOD_MAXGRADIENT;
					else if(_strcmpi(argv[i],"anisotropic") == 0)
						type = METHOD_ANISOTROPIC;
					else
					{
						printf("unknown methodType: %s\n",argv[i]);
						return false;
					}

				}
				else if(_strcmpi(argv[i],"nSORIteration") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					nSORIteration = atoi(argv[i]);
				}
				else if(_strcmpi(argv[i],"weight1") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					weight1 = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"grad_scale") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					grad_scale = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"anisotropic_ratio") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					anisotropic_ratio = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"display") == 0)
				{
					display = true;
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
