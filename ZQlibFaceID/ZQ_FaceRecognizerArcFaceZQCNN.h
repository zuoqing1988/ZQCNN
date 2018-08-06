#ifndef _ZQ_FACE_RECOGNIZER_ARC_FACE_ZQCNN_H_
#define _ZQ_FACE_RECOGNIZER_ARC_FACE_ZQCNN_H_
#pragma once

#include "ZQ_FaceRecognizerSphereFaceZQCNN.h"

namespace ZQ
{
	class ZQ_FaceRecognizerArcFaceZQCNN : public ZQ_FaceRecognizerSphereFaceZQCNN
	{
	public:
		virtual int GetCropWidth() const { return 112; }

		virtual int GetCropHeight() const { return 112; }
	};
}

#endif
