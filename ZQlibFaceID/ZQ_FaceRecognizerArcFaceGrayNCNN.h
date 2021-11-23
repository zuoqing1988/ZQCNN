#ifndef _ZQ_FACE_RECOGNIZER_ARC_FACE_GRAY_NCNN_H_
#define _ZQ_FACE_RECOGNIZER_ARC_FACE_GRAY_NCNN_H_
#pragma once

#include "ZQ_FaceRecognizerSphereFaceGrayNCNN.h"

namespace ZQ
{
	class ZQ_FaceRecognizerArcFaceGrayNCNN : public ZQ_FaceRecognizerSphereFaceGrayNCNN
	{
	public:
		virtual int GetCropWidth() const { return 112; }

		virtual int GetCropHeight() const { return 112; }
	};
}

#endif
