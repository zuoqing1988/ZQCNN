#ifndef _ZQ_FACE_DETECTOR_H_
#define _ZQ_FACE_DETECTOR_H_
#pragma once

#include "ZQ_CNN_BBox.h"
#include "ZQ_PixelFormat.h"
#include <vector>
namespace ZQ
{
	class ZQ_FaceDetector
	{
	public:
		virtual bool Init(const std::string model_root = "model", int thread_num = 1) = 0;

		virtual bool FindFace(const unsigned char* img, int width, int height, int widthStep, ZQ_PixelFormat pixFmt,
			int min_size, float scale, std::vector<ZQ_CNN_BBox>& bbox) = 0;

		virtual bool FindFaceROI(const unsigned char* img, int width, int height, int widthStep, ZQ_PixelFormat pixFmt,
			float roi_min_x, float roi_min_y, float roi_max_x, float roi_max_y,
			int min_size, float scale, std::vector<ZQ_CNN_BBox>& bbox) = 0;
	};
}

#endif
