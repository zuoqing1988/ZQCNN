#ifndef _ZQ_JPEG_ENCODER_H_
#define _ZQ_JPEG_ENCODER_H_
#pragma once

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include "jpeglib.h"
#include "ZQ_JpegCodecDefines.h"

namespace ZQ
{
	class ZQ_JpegEncoder
	{
	public:
		static bool Encode(const unsigned char* pSrc, const int width, const int height, const int nChannels, const int widthStep,
			ZQ_JpegCodecColorType::ColorTypeInput type, unsigned char*& pDst, unsigned long& dstLen, int quality = 90)
		{
			if (pSrc == 0)
				return false;

			if (type == ZQ_JpegCodecColorType::IN_RGB || type == ZQ_JpegCodecColorType::IN_EXT_BGR)
			{
				return Encode_BGRorRGB(pSrc, width, height, nChannels, widthStep, type, pDst, dstLen, quality);
			}
			else if (type == ZQ_JpegCodecColorType::IN_YCbCr_422)
			{
				return Encode_YUV422_plane(width, height, pSrc, width, pSrc + width*height, width / 2, pSrc + width*height * 3 / 2, 
					width / 2, pDst, dstLen, quality);
			}

			return false;
		}

		static bool Encode_BGRorRGB(const unsigned char* pSrc, const int width, const int height, const int nChannels, const int widthStep, 
			ZQ_JpegCodecColorType::ColorTypeInput type, unsigned char*& pDst, unsigned long& dstLen, int quality = 90)
		{
			J_COLOR_SPACE in_jcs_type = ZQ_JpegCodecColorType::GetJpegColorType(type);

			if (in_jcs_type != JCS_RGB && in_jcs_type != JCS_EXT_BGR)
				return false;

			if (pSrc == 0)
				return false;
			
			jpeg_compress_struct cinfo;
			jpeg_error_mgr jerr;

			cinfo.err = jpeg_std_error(&jerr);
			jpeg_create_compress(&cinfo);
			jpeg_mem_dest(&cinfo, &pDst, &dstLen);
			cinfo.image_width = width;
			cinfo.image_height = height;
			cinfo.input_components = nChannels;
			cinfo.in_color_space = in_jcs_type;
			
			
			jpeg_set_defaults(&cinfo);
			jpeg_set_quality(&cinfo, quality, true);

			
			jpeg_start_compress(&cinfo, true);

			JSAMPROW row_pointer[1] = { 0 };
			while (cinfo.next_scanline < height) {
				row_pointer[0] = (JSAMPLE*)pSrc + cinfo.next_scanline * widthStep;
				jpeg_write_scanlines(&cinfo, row_pointer, 1);
			}

			jpeg_finish_compress(&cinfo);
			jpeg_destroy_compress(&cinfo);

			return true;
		}

		static bool Encode_YUV422_plane(int width, int height, 
			const unsigned char* pSrc_Y, int Y_widthStep,
			const unsigned char* pSrc_U, int U_widthStep,
			const unsigned char* pSrc_V, int V_widthStep,
			unsigned char*& pDst, unsigned long& dstLen, int quality = 90)
		{
			if (pSrc_Y == 0 || pSrc_U == 0 ||pSrc_V == 0)
				return false;

			struct jpeg_compress_struct cinfo;
			struct jpeg_error_mgr jerr;
			//JSAMPROW row_pointer[1];  /* pointer to JSAMPLE row[s] */
			//int row_stride;    /* physical row width in image buffer */
			JSAMPIMAGE  buffer;
			int buf_width[3];
			int buf_height[3];
			int widthSteps[3] = { Y_widthStep, U_widthStep, V_widthStep };
			const unsigned char *yuv[3];

			yuv[0] = pSrc_Y;
			yuv[1] = pSrc_U;
			yuv[2] = pSrc_V;

			cinfo.err = jpeg_std_error(&jerr);
			jpeg_create_compress(&cinfo);
			jpeg_mem_dest(&cinfo, &pDst, &dstLen);
			cinfo.image_width = width;
			cinfo.image_height = height;
			cinfo.input_components = 3;
			cinfo.in_color_space = JCS_RGB;

			jpeg_set_defaults(&cinfo);
			jpeg_set_quality(&cinfo, quality, TRUE);

			cinfo.raw_data_in = TRUE;
			cinfo.jpeg_color_space = JCS_YCbCr;
			cinfo.comp_info[0].h_samp_factor = 2;
			cinfo.comp_info[0].v_samp_factor = 1;

			jpeg_start_compress(&cinfo, TRUE);

			buffer = (JSAMPIMAGE)(*cinfo.mem->alloc_small) ((j_common_ptr)&cinfo, JPOOL_IMAGE, 3 * sizeof(JSAMPARRAY));
			
			for (int band = 0; band <3; band++)
			{
				buf_width[band] = cinfo.comp_info[band].width_in_blocks * DCTSIZE;
				buf_height[band] = cinfo.comp_info[band].v_samp_factor * DCTSIZE;
				buffer[band] = (*cinfo.mem->alloc_sarray) ((j_common_ptr)&cinfo, JPOOL_IMAGE, buf_width[band], buf_height[band]);
			}


			int max_line = cinfo.max_v_samp_factor*DCTSIZE;

			unsigned char* tmp_pSrc, *tmp_pDst;
			for (int counter = 0; cinfo.next_scanline < cinfo.image_height; counter++)
			{
				//buffer image copy.
				for (int band = 0; band <3; band++)
				{
					int mem_size = buf_width[band];
					tmp_pDst = (unsigned char*)buffer[band][0];
					tmp_pSrc = (unsigned char*)yuv[band] + counter*buf_height[band] * widthSteps[band];

					for (int i = 0; i <buf_height[band]; i++)
					{
						memcpy(tmp_pDst, tmp_pSrc, mem_size);
						tmp_pSrc += widthSteps[band];
						tmp_pDst += buf_width[band];
					}
				}
				jpeg_write_raw_data(&cinfo, buffer, max_line);
			}

			jpeg_finish_compress(&cinfo);
			jpeg_destroy_compress(&cinfo);

			return true;
		}



		static bool SaveImage(const unsigned char* pSrc, const int width, const int height, const int nChannels, const int widthStep,
			ZQ_JpegCodecColorType::ColorTypeInput type, const char* filename, int quality = 90)
		{
			FILE* out = 0;
			
			if (0 != fopen_s(&out,filename, "wb"))
				return false;

			unsigned char* pDst = 0;
			unsigned long dstLen = 0;
			if (!Encode(pSrc, width, height, nChannels, widthStep, type, pDst, dstLen, quality))
			{
				fclose(out);
				return false;
			}
			bool ret_flag = (dstLen == fwrite(pDst, sizeof(unsigned char), dstLen, out));

			free(pDst);
			fclose(out);
			return ret_flag;
		}

	};
}

#endif