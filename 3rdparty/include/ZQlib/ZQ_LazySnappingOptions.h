#ifndef _ZQ_LAZY_SNAPPING_OPTIONS_H_
#define _ZQ_LAZY_SNAPPING_OPTIONS_H_
#pragma once

#include <string.h>
#include <stdio.h>

namespace ZQ
{
	class ZQ_LazySnappingOptions
	{
	public:
		ZQ_LazySnappingOptions(){ Reset(); }
		~ZQ_LazySnappingOptions(){}

	public:
		float lambda_for_E2;
		float color_scale_for_E2;
		float lambda_for_E3;
		float sigma_for_E3;
		int dilate_erode_size;
		int area_thresh;

		void Reset()
		{
			lambda_for_E2 = 200;
			color_scale_for_E2 = 255;
			lambda_for_E3 = 10;
			sigma_for_E3 = 0.7;
			dilate_erode_size = 2;
			area_thresh = 500;
		}

		bool HandleArgs(const int argc, const char** argv)
		{
			for (int k = 0; k < argc; k++)
			{
				if (_strcmpi(argv[k], "lamda_for_E2") == 0)
				{
					k++;
					if (k >= argc)
					{
						printf("the value of %s ?\n", argv[k-1]);
						return false;
					}
					lambda_for_E2 = atof(argv[k]);
				}
				else if (_strcmpi(argv[k], "color_scale_for_E2") == 0)
				{
					k++;
					if (k >= argc)
					{
						printf("the value of %s ?\n", argv[k - 1]);
						return false;
					}
					color_scale_for_E2 = atof(argv[k]);
				}
				else if (_strcmpi(argv[k], "lambda_for_E3") == 0)
				{
					k++;
					if (k >= argc)
					{
						printf("the value of %s ?\n", argv[k - 1]);
						return false;
					}
					lambda_for_E3 = atof(argv[k]);
				}
				else if (_strcmpi(argv[k], "sigma_for_E3") == 0)
				{
					k++;
					if (k >= argc)
					{
						printf("the value of %s ?\n", argv[k - 1]);
						return false;
					}
					sigma_for_E3 = atof(argv[k]);
				}
				else if (_strcmpi(argv[k], "dilate_erode_size") == 0)
				{
					k++;
					if (k >= argc)
					{
						printf("the value of %s\n", argv[k - 1]);
						return false;
					}
					dilate_erode_size = atoi(argv[k]);
				}
				else if (_strcmpi(argv[k], "area_thresh") == 0)
				{
					k++;
					if (k >= argc)
					{
						printf("the value of %s\n", argv[k - 1]);
						return false;
					}
					area_thresh = atoi(argv[k]);
				}
				else
				{
					printf("unknown parameters %s\n", argv[k]);
					return false;
				}
			}
			return true;
		}
	};
}
#endif