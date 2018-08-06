#ifndef _ZQ_JPEG_DECODER_H_
#define _ZQ_JPEG_DECODER_H_
#pragma once

#include <stdio.h>
#include <string.h>
#include "jpeglib.h"
#include "ZQ_JpegCodecDefines.h"

namespace ZQ
{
	class ZQ_JpegDecoder
	{
	public:
		/* make sure pDst == 0, otherwise there will be memory leak */
		static bool Decode(const unsigned char* pSrc, const unsigned long srcLen, ZQ_JpegCodecColorType::ColorTypeOutput type, 
			unsigned char*& pDst, int& width, int& height, int& nChannels, int& widthStep, int alignN = 4)
		{
			J_COLOR_SPACE out_jcs_type = ZQ_JpegCodecColorType::GetJpegColorType(type);

			if (pSrc == 0 || out_jcs_type == JCS_UNKNOWN)
				return false;

			jpeg_decompress_struct cinfo;
			jpeg_error_mgr jerr;

			cinfo.err = jpeg_std_error(&jerr);
			jpeg_create_decompress(&cinfo);
			jpeg_mem_src(&cinfo, pSrc, srcLen);
			if (JPEG_HEADER_OK != jpeg_read_header(&cinfo, TRUE))
			{
				jpeg_finish_decompress(&cinfo);
				jpeg_destroy_decompress(&cinfo);
				return false;
			}

			cinfo.out_color_space = out_jcs_type;
			if (!jpeg_start_decompress(&cinfo))
				return false;

			width = cinfo.output_width;
			height = cinfo.output_height;
			nChannels = cinfo.output_components;
			if (alignN > 1)
				widthStep = (width*nChannels + alignN - 1) / alignN * alignN;
			else
				widthStep = width*nChannels;

			pDst = (unsigned char*)malloc(widthStep * height);
			memset(pDst, 0, sizeof(unsigned char)* widthStep * height);

			JSAMPARRAY buffer;
			buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, width*nChannels, 1);

			unsigned char *point = pDst;
			while (cinfo.output_scanline < height)
			{
				jpeg_read_scanlines(&cinfo, buffer, 1);   // read one line
				memcpy(point, *buffer, width*nChannels);    
				point += widthStep;
			}

			jpeg_finish_decompress(&cinfo);
			jpeg_destroy_decompress(&cinfo);

			return true;
		}

		static bool Decode_with_allocated(const unsigned char* pSrc, const unsigned long srcLen, ZQ_JpegCodecColorType::ColorTypeOutput type, 
			unsigned char*& pDst, int& width, int& height, int& nChannels, const int widthStep)
		{
			J_COLOR_SPACE out_jcs_type = ZQ_JpegCodecColorType::GetJpegColorType(type);

			if (pSrc == 0 || out_jcs_type == JCS_UNKNOWN)
				return false;

			jpeg_decompress_struct cinfo;
			jpeg_error_mgr jerr;

			cinfo.err = jpeg_std_error(&jerr);
			jpeg_create_decompress(&cinfo);
			jpeg_mem_src(&cinfo, pSrc, srcLen);
			if (JPEG_HEADER_OK != jpeg_read_header(&cinfo, TRUE))
			{
				jpeg_finish_decompress(&cinfo);
				jpeg_destroy_decompress(&cinfo);
				return false;
			}
			cinfo.out_color_space = out_jcs_type;
			
			if (!jpeg_start_decompress(&cinfo))
				return false;

			width = cinfo.output_width;
			height = cinfo.output_height;
			nChannels = cinfo.output_components;
			
			JSAMPARRAY buffer;
			buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, width*nChannels, 1);

			unsigned char *point = pDst;
			while (cinfo.output_scanline < height)
			{
				jpeg_read_scanlines(&cinfo, buffer, 1);   // read one line
				memcpy(point, *buffer, width*nChannels);
				point += widthStep;
			}

			jpeg_finish_decompress(&cinfo);
			jpeg_destroy_decompress(&cinfo);

			return true;
		}
		
	};
}

#endif