#ifndef _ZQ_JPEG_CODEC_DEFINES_H_
#define _ZQ_JPEG_CODEC_DEFINES_H_
#pragma once

#include "jpeglib.h"

namespace ZQ
{
	class ZQ_JpegCodecColorType
	{
	public:
		enum ColorTypeInput
		{
			IN_GRAY,
			IN_RGB,
			IN_YCbCr_422,
			IN_EXT_BGR
		};

		enum ColorTypeOutput
		{
			OUT_GRAY,
			OUT_RGB,
			OUT_YCbCr,
			OUT_EXT_BGR,
			OUT_EXT_BGRA,
			OUT_EXT_RGBA
		};

		static J_COLOR_SPACE GetJpegColorType(ColorTypeInput type)
		{
			switch (type)
			{
			case IN_GRAY:
				return J_COLOR_SPACE::JCS_GRAYSCALE;
				break;
			case IN_RGB:
				return J_COLOR_SPACE::JCS_RGB;
				break;
			case IN_EXT_BGR:
				return J_COLOR_SPACE::JCS_EXT_BGR;
				break;
			case IN_YCbCr_422:
				return J_COLOR_SPACE::JCS_YCbCr;
				break;
			default:
				return J_COLOR_SPACE::JCS_UNKNOWN;
				break;
			}
		}

		static J_COLOR_SPACE GetJpegColorType(ColorTypeOutput type)
		{
			switch (type)
			{
			case OUT_GRAY:
				return J_COLOR_SPACE::JCS_GRAYSCALE;
				break;
			case OUT_RGB:
				return J_COLOR_SPACE::JCS_RGB;
				break;
			case OUT_EXT_BGR:
				return J_COLOR_SPACE::JCS_EXT_BGR;
				break;
			case OUT_EXT_BGRA:
				return J_COLOR_SPACE::JCS_EXT_BGRA;
				break;
			case OUT_EXT_RGBA:
				return J_COLOR_SPACE::JCS_EXT_RGBA;
				break;
			case OUT_YCbCr:
				return J_COLOR_SPACE::JCS_YCbCr;
				break;
			default:
				return J_COLOR_SPACE::JCS_UNKNOWN;
				break;
			}
		}

		static int GetJpegColorTypeNumChannels(ColorTypeInput type)
		{
			switch (type)
			{
			case IN_GRAY:
				return 1;
				break;
			case IN_RGB:
				return 3;
				break;
			case IN_EXT_BGR:
				return 3;
				break;
			case IN_YCbCr_422:
				return 3;
				break;
			default:
				return 0;
				break;
			}
		}

		static int GetJpegColorTypeNumChannels(ColorTypeOutput type)
		{
			switch (type)
			{
			case OUT_GRAY:
				return 1;
				break;
			case OUT_RGB:
				return 3;
				break;
			case OUT_EXT_RGBA:
				return 4;
				break;
			case OUT_EXT_BGR:
				return 3;
				break;
			case OUT_EXT_BGRA:
				return 4;
				break;
			case OUT_YCbCr:
				return 3;
				break;
			default:
				return 0;
				break;
			}
		}
	};
}

#endif