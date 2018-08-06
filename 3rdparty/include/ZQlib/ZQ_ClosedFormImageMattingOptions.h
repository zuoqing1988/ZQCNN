#ifndef _ZQ_CLOSED_FORM_IMAGE_MATTING_OPTIONS_H_
#define _ZQ_CLOSED_FORM_IMAGE_MATTING_OPTIONS_H_
#pragma once
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

namespace ZQ
{
	class ZQ_ClosedFormImageMattingOptions
	{
	public:
		enum  SolveForeBackMode
		{
			SOLVE_FORE_BACK_4DIR,
			SOLVE_FORE_BACK_2DIR,
			SOLVE_FORE_BACK_ORI_PAPER
		};

		ZQ_ClosedFormImageMattingOptions()
		{
			Reset();
		}

		int win_size;
		float epsilon;
		int max_level_for_c2f;
		float consts_thresh_for_c2f;
		bool display;
		int max_iter_for_solve_fb;
		SolveForeBackMode solve_fb_mode;

		void Reset()
		{
			win_size = 1;
			epsilon = 1e-7;
			max_level_for_c2f = 1;
			consts_thresh_for_c2f = 0.02;
			display = false;
			max_iter_for_solve_fb = 100;
			solve_fb_mode = SOLVE_FORE_BACK_ORI_PAPER;
		}

		bool HandleArgs(const int argc, const char** argv)
		{
			for (int k = 0; k < argc; k++)
			{
				if (_strcmpi(argv[k], "win_size") == 0)
				{
					k++;
					if (k >= argc)
					{
						printf("the value of %s ?\n", argv[k-1]);
						return false;
					}
					win_size = atoi(argv[k]);
					win_size = win_size > 2 ? 2 : win_size;
					win_size = win_size < 1 ? 1 : win_size;
				}
				else if (_strcmpi(argv[k], "epsilon") == 0)
				{
					k++;
					if (k >= argc)
					{
						printf("the value of %s ?\n", argv[k-1]);
						return false;
					}
					epsilon = atof(argv[k]);
				}
				else if (_strcmpi(argv[k], "max_level") == 0)
				{
					k++;
					if (k >= argc)
					{
						printf("the value of %s\n", argv[k-1]);
						return false;
					}
					max_level_for_c2f = atoi(argv[k]);
				}
				else if (_strcmpi(argv[k], "consts_thresh") == 0)
				{
					k++;
					if (k >= argc)
					{
						printf("the value of %s\n", argv[k-1]);
						return false;
					}
					consts_thresh_for_c2f = atoi(argv[k]);
				}
				else if (_strcmpi(argv[k], "display") == 0)
				{
					display = true;
				}
				else if (_strcmpi(argv[k], "max_iter_for_solve_fb") == 0)
				{
					k++;
					if (k >= argc)
					{
						printf("the value of %s\n", argv[k - 1]);
						return false;
					}
					max_iter_for_solve_fb = atoi(argv[k]);
				}
				else if (_strcmpi(argv[k], "solve_fb_mode") == 0)
				{
					k++;
					if (k >= argc)
					{
						printf("the value of %s\n", argv[k - 1]);
						return false;
					}
					if (_strcmpi(argv[k], "4dir") == 0)
						solve_fb_mode = SOLVE_FORE_BACK_4DIR;
					else if (_strcmpi(argv[k], "2dir") == 0)
						solve_fb_mode = SOLVE_FORE_BACK_2DIR;
					else if (_strcmpi(argv[k], "ori_paper") == 0)
						solve_fb_mode = SOLVE_FORE_BACK_ORI_PAPER;
					else
					{
						printf("unknown parameters %s\n", argv[k]);
						return false;
					}
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