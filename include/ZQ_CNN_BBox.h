#ifndef _ZQ_CNN_BBOX_H_
#define _ZQ_CNN_BBOX_H_
#pragma once
#include <string.h>
#include <stdio.h>

namespace ZQ
{
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
		float ppoint[10];
		float regreCoord[4];

		ZQ_CNN_BBox()
		{
			memset(this, 0, sizeof(ZQ_CNN_BBox));
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