#ifndef _ZQ_FACE_RECOGNIZER_AGE_GENDER_MINICAFFE_H_
#define _ZQ_FACE_RECOGNIZER_AGE_GENDER_MINICAFFE_H_
#pragma once

#include "ZQ_FaceRecognizerSphereFaceMiniCaffe.h"

namespace ZQ
{
	class ZQ_FaceRecognizerAgeGenderMiniCaffe : public ZQ_FaceRecognizerSphereFaceMiniCaffe
	{
	public:
		virtual int GetCropWidth() const { return 160; }

		virtual int GetCropHeight() const { return 160; }
	};
}

#endif
