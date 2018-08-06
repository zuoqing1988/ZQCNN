#ifndef _ZQ_FACE_RECOGNIZER_ARC_FACE_MINICAFFE_H_
#define _ZQ_FACE_RECOGNIZER_ARC_FACE_MINICAFFE_H_
#pragma once

#include "ZQ_FaceRecognizerSphereFaceMiniCaffe.h"

namespace ZQ
{
	class ZQ_FaceRecognizerArcFaceMiniCaffe : public ZQ_FaceRecognizerSphereFaceMiniCaffe
	{
	public:
		virtual int GetCropWidth() const { return 112; }

		virtual int GetCropHeight() const { return 112; }
	};
}

#endif
