#ifndef _ZQ_CNN_BBOX_H_
#define _ZQ_CNN_BBOX_H_
#pragma once

// 审计修复 2026-10-03（附录 EU）：本头以前**不**引入编译配置，于是
// `__int64` / `__min` / `__max` 这三个 MSVC 内建在 gcc 上无处可寻。
// 受害的是把本头拉进来的 ZQlibFaceID：
//     ZQ_FaceGroup.h -> ZQ_CNN_BBox.h        （ZQ_FaceContainerForVideo.h 用了 __int64）
//     ZQ_FaceDetector.h -> ZQ_CNN_BBox.h     （ZQ_FaceExtractor.h 用了 __min）
// 两个头此前被 C1 门禁归成 NEEDS_LIB（"缺外部库"）—— **分错类了**，
// 它们不缺库，只是缺这三个宏。
//
// 为什么放在这里：ZQ_CNN_BBox.h 是 ZQlibFaceID 那一侧的**公共祖先**
// （ZQ_FaceGroup.h / ZQ_FaceDetector.h 都 include 它），
// 挂在这里一次就把整条链覆盖了。
//
// ZQ_CNN_CompileConfig.h 自带 include guard，且这三个宏都是 `#ifndef` 保护的，
// MSVC 下它们本来就是编译器内建，所以对 Windows 侧零影响。
#include "ZQ_CNN_CompileConfig.h"

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
		float scale_x;
		float scale_y;
		bool has_headposegaze;
		float headposegaze[9];
		float center_and_rot[3];

		ZQ_CNN_BBox106()
		{
			memset(this, 0, sizeof(ZQ_CNN_BBox106));
			scale_x = 1;
			scale_y = 1;
			has_headposegaze = false;
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

	class ZQ_CNN_BBox240
	{
	public:
		ZQ_CNN_BBox106 box;
		float left_brow_eye_ppoint[70];
		float right_brow_eye_ppoint[70];
		float mouth_ppoint[128];

		ZQ_CNN_BBox240()
		{
			memset(this, 0, sizeof(ZQ_CNN_BBox240));
		}

		~ZQ_CNN_BBox240() {}

		bool ReadFromBinary(FILE* in)
		{
			if (fread(this, sizeof(ZQ_CNN_BBox240), 1, in) != 1)
				return false;
			return true;
		}

		bool WriteBinary(FILE* out) const
		{
			if (fwrite(this, sizeof(ZQ_CNN_BBox240), 1, out) != 1)
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