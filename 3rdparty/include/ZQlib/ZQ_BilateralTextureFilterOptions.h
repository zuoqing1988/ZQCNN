#ifndef _ZQ_BILATERAL_TEXTURE_FILTER_OPTIONS_H_
#define _ZQ_BILATERAL_TEXTURE_FILTER_OPTIONS_H_
#pragma once

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

namespace ZQ
{
	class ZQ_BilateralTextureFilterOptions
	{
	public:
		int half_patch_size;
		int fsize;
		float sigma_for_alpha;
		float sigma_for_space;
		float sigma_for_value;

		ZQ_BilateralTextureFilterOptions()
		{
			Reset();
		}
		void Reset()
		{
			half_patch_size = 3;
			sigma_for_alpha = 5 * (2 * half_patch_size + 1);
			fsize = 3;
			sigma_for_space = 2;
			sigma_for_value = 0.1;
		}

		bool HandleParas(int argc, const char** argv)
		{
			for (int i = 0; i < argc; i++)
			{
				if (_strcmpi(argv[i], "half_patch_size") == 0)
				{
					i++;
					if (i >= argc)
					{
						printf("the value of %s ?\n", argv[i - 1]);
						return false;
					}
					half_patch_size = atoi(argv[i]);
				}
				else if (_strcmpi(argv[i], "sigma_alpha") == 0)
				{
					i++;
					if (i >= argc)
					{
						printf("the value of %s ?\n", argv[i - 1]);
						return false;
					}
					sigma_for_alpha = atof(argv[i]);
				}
				else if (_strcmpi(argv[i], "fsize") == 0)
				{
					i++;
					if (i >= argc)
					{
						printf("the value of %s ?\n", argv[i - 1]);
						return false;
					}
					fsize = atoi(argv[i]);
				}
				else if (_strcmpi(argv[i], "sigma_s") == 0)
				{
					i++;
					if (i >= argc)
					{
						printf("the value of %s ?\n", argv[i - 1]);
						return false;
					}
					sigma_for_space = atof(argv[i]);
				}
				else if (_strcmpi(argv[i], "sigma_v") == 0)
				{
					i++;
					if (i >= argc)
					{
						printf("the value of %s ?\n", argv[i - 1]);
						return false;
					}
					sigma_for_value = atof(argv[i]);
				}
				else
				{
					printf("unknown para: %s\n", argv[i]);
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
				"[\"half_patch_size\" \t half_patch_size]\n"
				"[\"sigma_alpha\" \t sigma_for_alpha]\n"
				);
		}
	};
}

#endif