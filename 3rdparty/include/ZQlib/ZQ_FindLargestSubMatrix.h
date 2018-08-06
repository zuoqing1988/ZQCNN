#ifndef _ZQ_FIND_LARGEST_SUB_MATRIX_H_
#define _ZQ_FIND_LARGEST_SUB_MATRIX_H_
#pragma once

#include "ZQ_FindLargestRectInHistgram.h"

namespace ZQ
{
	class ZQ_FindLargestSubMatrix
	{
	public:
		static void FindLargestSubMatrix(const bool* flag, unsigned int in_width, unsigned int in_height, int& off_x, int& off_y, int& width, int& height)
		{
			unsigned int* hist = new unsigned int[in_height*in_width];

			for (int w = 0; w < in_width; w++)
			{
				hist[(in_height - 1)*in_width + w] = flag[(in_height - 1)*in_width + w];
			}

			for (int h = in_height - 2; h >= 0; h--)
			{
				for (int w = 0; w < in_width; w++)
				{
					if (flag[h*in_width + w])
						hist[h*in_width + w] = hist[(h + 1)*in_width + w] + 1;
					else
						hist[h*in_width + w] = 0;
				}
			}

			off_x = 0; off_y = 0; width = 0; height = 0;
			long long max_area = 0;
			for (int h = 0; h < in_height; h++)
			{
				int cur_off, cur_w, cur_h;
				long long cur_area;
				ZQ_FindLargestRectInHistgram::FindLargestRectInHistgram(in_width, hist + h*in_width, cur_off, cur_w, cur_h, cur_area);
				if (cur_area > max_area)
				{
					off_x = cur_off;
					off_y = h;
					width = cur_w;
					height = cur_h;
					max_area = cur_area;
				}
				if (max_area > (long long)(in_height - 1 - h)*in_width)
					break;
			}
			delete[]hist;
		}
	};
}

#endif