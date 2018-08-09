#ifndef _ZQ_FACE_RECOGNIZER_AGE_GENDER_ZQCNN_H_
#define _ZQ_FACE_RECOGNIZER_AGE_GENDER_ZQCNN_H_
#pragma once

#include "ZQ_FaceRecognizerSphereFaceZQCNN.h"

namespace ZQ
{
	class ZQ_FaceRecognizerAgeGenderZQCNN : public ZQ_FaceRecognizerSphereFaceZQCNN
	{
	public:
		virtual int GetCropWidth() const { return 160; }

		virtual int GetCropHeight() const { return 160; }
	};
}

#endif
