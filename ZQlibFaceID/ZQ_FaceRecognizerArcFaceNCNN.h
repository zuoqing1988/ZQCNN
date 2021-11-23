#ifndef _ZQ_FACE_RECOGNIZER_ARC_FACE_NCNN_H_
#define _ZQ_FACE_RECOGNIZER_ARC_FACE_NCNN_H_
#pragma once

#include "ZQ_FaceRecognizerSphereFaceNCNN.h"

namespace ZQ
{
	class ZQ_FaceRecognizerArcFaceNCNN : public ZQ_FaceRecognizerSphereFaceNCNN
	{
	public:
		virtual int GetCropWidth() const { return 112; }

		virtual int GetCropHeight() const { return 112; }
	};
}

#endif

