#ifndef _ZQ_CNN_BBOX_H_
#define _ZQ_CNN_BBOX_H_
#pragma once
#include <string.h>
#include <stdio.h>
#include <vector>
#include <map>

namespace ZQ
{
	class ZQ_CNN_NormalizedBBox
	{
	public:
		float col1;
		float col2;
		float row1;
		float row2;
		int label;
		bool difficult;
		float score;
		float size;

		ZQ_CNN_NormalizedBBox()
		{
			col1 = col2 = row1 = row2 = 0;
			label = -1;
			difficult = false;
			score = 0;
			size = 1;
		}
	};

	using ZQ_CNN_LabelBBox = std::map<int, std::vector<ZQ_CNN_NormalizedBBox> >;

	class ZQ_CNN_BBox
	{
	public:
		float score;
		int row1;
		int col1;
		int row2;
		int col2;
		float area;
		bool exist;
		bool need_check_overlap_count;
		float ppoint[10];
		float regreCoord[4];
		float scale_x;
		float scale_y;

		ZQ_CNN_BBox()
		{
			memset(this, 0, sizeof(ZQ_CNN_BBox));
			scale_x = 1;
			scale_y = 1;
		}

		~ZQ_CNN_BBox() {}

		bool ReadFromBinary(FILE* in)
		{
			if (fread(this, sizeof(ZQ_CNN_BBox), 1, in) != 1)
				return false;
			return true;
		}

		bool WriteBinary(FILE* out) const
		{
			if (fwrite(this, sizeof(ZQ_CNN_BBox), 1, out) != 1)
				return false;
			return true;
		}
	};

	class ZQ_CNN_BBox106
	{
	public:
		float score;
		int row1;
		int col1;
		int row2;
		int col2;
		float area;
		bool exist;
		bool need_check_overlap_count;
		float ppoint[212];
		float regreCoord[4];

		ZQ_CNN_BBox106()
		{
			memset(this, 0, sizeof(ZQ_CNN_BBox106));
		}

		~ZQ_CNN_BBox106() {}

		bool ReadFromBinary(FILE* in)
		{
			if (fread(this, sizeof(ZQ_CNN_BBox106), 1, in) != 1)
				return false;
			return true;
		}

		bool WriteBinary(FILE* out) const
		{
			if (fwrite(this, sizeof(ZQ_CNN_BBox106), 1, out) != 1)
				return false;
			return true;
		}
	};

	class ZQ_CNN_OrderScore
	{
	public:
		float score;
		int oriOrder;

		ZQ_CNN_OrderScore()
		{
			memset(this, 0, sizeof(ZQ_CNN_OrderScore));
		}
	};
}
#endif