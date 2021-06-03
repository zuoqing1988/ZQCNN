#ifndef _ZQ_BILATERAL_FILTER_OPTIONS_H_
#define _ZQ_BILATERAL_FILTER_OPTIONS_H_
#pragma once 

#include <stdio.h>
#include <string.h>

namespace ZQ
{
	class ZQ_BilateralFilterOptions
	{
	public:
		
		int fsize;
		float sigma_for_space;
		float sigma_for_value;

		ZQ_BilateralFilterOptions()
		{
			Reset();
		}
		void Reset()
		{
			fsize = 3;
			sigma_for_space = 2;
			sigma_for_value = 0.1;
		}

		bool HandleParas(int argc, const char** argv)
		{
			for(int i = 0;i < argc;i++)
			{
#if defined(_WIN32)
				if(_strcmpi(argv[i],"fsize") == 0)
#else
				if (strcmp(argv[i], "fsize") == 0)
#endif
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					fsize = atof(argv[i]);
				}
#if defined(_WIN32)
				else if(_strcmpi(argv[i],"sigma_s") == 0)
#else
				else if (strcmp(argv[i], "sigma_s") == 0)
#endif
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					sigma_for_space = atof(argv[i]);
				}
#if defined(_WIN32)
				else if(_strcmpi(argv[i],"sigma_v") == 0)
#else
				else if (strcmp(argv[i], "sigma_v") == 0)
#endif
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					sigma_for_value = atof(argv[i]);
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
				"[\"sigma_s\" \t sigma_for_space]\n"
				"[\"sigma_v\" \t sigma_for_value]\n"
				"[\"fsize\" \t fsize]\n"
				);
		}
	};
}

#endif