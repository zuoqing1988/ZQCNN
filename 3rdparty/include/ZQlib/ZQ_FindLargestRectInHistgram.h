#ifndef _ZQ_FIND_LARGEST_RECT_IN_HISTGRAM_H_
#define _ZQ_FIND_LARGEST_RECT_IN_HISTGRAM_H_
#pragma once

#include <stack>

namespace ZQ
{
	class ZQ_FindLargestRectInHistgram
	{
		class Node
		{
		public:
			int height;
			int start_idx;
			Node(int _height, int _idx) :height(_height), start_idx(_idx){}
		};

	public:
		static void FindLargestRectInHistgram(int len, const unsigned int* hist, int& offset, int& width, int& height, long long& area)
		{
			std::stack<Node> s;
		
			s.push(Node(-1, 0));

			offset = 0;
			width = 0;
			height = 0;
			area = 0;

			for (int i = 0; i <= len; i++)
			{
				int tmp_height;
				if (i == len)
				{
					tmp_height = 0;
				}
				else
				{
					tmp_height = hist[i];
				}

				Node t(tmp_height, i);
				while (s.top().height > tmp_height)
				{
					t = s.top();
					s.pop();

					long long cur_area = (long long)(i - t.start_idx) * t.height;
					if (cur_area > area)
					{
						area = cur_area;
						offset = t.start_idx;
						width = i - t.start_idx;
						height = t.height;
					}
				}
				s.push(Node(tmp_height, t.start_idx));
			}
		}
	};
}

#endif