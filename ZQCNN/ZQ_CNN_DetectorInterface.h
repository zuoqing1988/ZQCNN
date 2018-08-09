#ifndef _ZQ_CNN_DETECTOR_INTERFACE_H_
#define _ZQ_CNN_DETECTOR_INTERFACE_H_
#pragma once

#include <stdlib.h>

namespace ZQ
{
	class ZQ_CNN_DetectorInterface
	{
	public:
		virtual bool Initialize(const void* input_args) = 0;
		virtual bool Detect(const unsigned char* rgb_image, int width, int height, int widthStep, const void* detect_arg, void* output_detected_data) = 0;
		virtual bool DrawResult(unsigned char* rgb_image, int width, int height, int widthStep, const void* detected_data) = 0;
	};

	class ZQ_DetectorStupid : ZQ_CNN_DetectorInterface
	{
	public:
		virtual bool Initialize(const void* input_args)
		{
			return false;
		}

		virtual bool Detect(const unsigned char* rgb_image, int width, int height, int widthStep, void* output_detected_data)
		{
			//detect nothing
			return false;
		}

		virtual bool DrawResult(unsigned char* rgb_image, int width, int height, int widthStep, const void* detected_data)
		{
			//draw nothing
			return false;
		}
	};
}

#endif
