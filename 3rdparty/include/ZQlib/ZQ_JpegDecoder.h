#ifndef _ZQ_JPEG_DECODER_H_
#define _ZQ_JPEG_DECODER_H_
#pragma once

#include <stdio.h>
#include <string.h>
#include <setjmp.h>
#include "jpeglib.h"
#include "ZQ_JpegCodecDefines.h"

namespace ZQ
{
	class ZQ_JpegDecoder
	{
	public:
		/*
		 * libjpeg 的默认错误处理器(jpeg_std_error)在遇到损坏数据时直接调用 exit(),
		 * 会把整个宿主进程带走。必须换成带 longjmp 的自定义处理器, 让 Decode 返回 false。
		 */
		struct ZQ_JpegErrorMgr
		{
			struct jpeg_error_mgr pub;
			jmp_buf setjmp_buffer;
		};

		static void ZQ_Jpeg_error_exit(j_common_ptr cinfo)
		{
			ZQ_JpegErrorMgr* err = (ZQ_JpegErrorMgr*)cinfo->err;
			char buffer[JMSG_LENGTH_MAX];
			(*cinfo->err->format_message)(cinfo, buffer);
			longjmp(err->setjmp_buffer, 1);
		}

		static void ZQ_Jpeg_output_message(j_common_ptr cinfo)
		{
			// 静默: 默认实现会往 stderr 刷
		}

		/* make sure pDst == 0, otherwise there will be memory leak */
		static bool Decode(const unsigned char* pSrc, const unsigned long srcLen, ZQ_JpegCodecColorType::ColorTypeOutput type,
			unsigned char*& pDst, int& width, int& height, int& nChannels, int& widthStep, int alignN = 4)
		{
			J_COLOR_SPACE out_jcs_type = ZQ_JpegCodecColorType::GetJpegColorType(type);

			if (pSrc == 0 || out_jcs_type == JCS_UNKNOWN)
				return false;

			jpeg_decompress_struct cinfo;
			ZQ_JpegErrorMgr jerr;
			volatile bool cinfo_created = false;
			unsigned char* volatile dst_buf = 0;
			/* 损坏的 JPEG 会让 libjpeg 从 error_exit longjmp 回来, 由我们收尾并返回 false;
			   不装 setjmp 的话 libjpeg 会直接 exit(), 整个宿主进程被一张图片带走 */
			if (setjmp(jerr.setjmp_buffer))
			{
				if (dst_buf != 0)
				{
					free(dst_buf);
					dst_buf = 0;
					pDst = 0;
				}
				if (cinfo_created)
				{
					jpeg_destroy_decompress(&cinfo);
					cinfo_created = false;
				}
				return false;
			}
			cinfo.err = jpeg_std_error(&jerr.pub);
			jerr.pub.error_exit = ZQ_Jpeg_error_exit;
			jerr.pub.output_message = ZQ_Jpeg_output_message;
			jpeg_create_decompress(&cinfo);
			cinfo_created = true;
			jpeg_mem_src(&cinfo, pSrc, srcLen);
			if (JPEG_HEADER_OK != jpeg_read_header(&cinfo, TRUE))
			{
				jpeg_abort_decompress(&cinfo);
				jpeg_destroy_decompress(&cinfo);
				cinfo_created = false;
				return false;
			}

			cinfo.out_color_space = out_jcs_type;
			if (!jpeg_start_decompress(&cinfo))
			{
				// 原来直接 return false，cinfo 与 JPOOL 全部泄漏
				jpeg_abort_decompress(&cinfo);
				jpeg_destroy_decompress(&cinfo);
				cinfo_created = false;
				return false;
			}

			width = cinfo.output_width;
			height = cinfo.output_height;
			nChannels = cinfo.output_components;
			if (alignN > 1)
				widthStep = (width*nChannels + alignN - 1) / alignN * alignN;
			else
				widthStep = width*nChannels;

			// widthStep * height 是 int*int, 一张超大 JPEG(3*30000*24000) 就能溢出成负,
			// 随后 memset 用 size_t 提升后的"真实大尺寸"去清一个"被截断的小 buffer" -> 确定性堆溢出
			unsigned long long total_size = (unsigned long long)widthStep * (unsigned)height;
			if (total_size == 0 || total_size > 0x7FFFFFFFULL)
			{
				jpeg_destroy_decompress(&cinfo);
				return false;
			}
			dst_buf = (unsigned char*)malloc((size_t)total_size);
			if (dst_buf == 0)
			{
				jpeg_destroy_decompress(&cinfo);
				return false;
			}
			memset(dst_buf, 0, (size_t)total_size);

			JSAMPARRAY buffer;
			buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, width*nChannels, 1);

			unsigned char *point = dst_buf;
			while (cinfo.output_scanline < height)
			{
				jpeg_read_scanlines(&cinfo, buffer, 1);   // read one line
				memcpy(point, *buffer, width*nChannels);    
				point += widthStep;
			}

			jpeg_finish_decompress(&cinfo);
			jpeg_destroy_decompress(&cinfo);
			cinfo_created = false;
			pDst = dst_buf;

			return true;
		}

		static bool Decode_with_allocated(const unsigned char* pSrc, const unsigned long srcLen, ZQ_JpegCodecColorType::ColorTypeOutput type, 
			unsigned char*& pDst, int& width, int& height, int& nChannels, const int widthStep)
		{
			J_COLOR_SPACE out_jcs_type = ZQ_JpegCodecColorType::GetJpegColorType(type);

			if (pSrc == 0 || out_jcs_type == JCS_UNKNOWN)
				return false;

			jpeg_decompress_struct cinfo;
			ZQ_JpegErrorMgr jerr;
			volatile bool cinfo_created = false;
			unsigned char* volatile dst_buf = 0;
			/* 损坏的 JPEG 会让 libjpeg 从 error_exit longjmp 回来, 由我们收尾并返回 false;
			   不装 setjmp 的话 libjpeg 会直接 exit(), 整个宿主进程被一张图片带走 */
			if (setjmp(jerr.setjmp_buffer))
			{
				if (dst_buf != 0)
				{
					free(dst_buf);
					dst_buf = 0;
					pDst = 0;
				}
				if (cinfo_created)
				{
					jpeg_destroy_decompress(&cinfo);
					cinfo_created = false;
				}
				return false;
			}
			cinfo.err = jpeg_std_error(&jerr.pub);
			jerr.pub.error_exit = ZQ_Jpeg_error_exit;
			jerr.pub.output_message = ZQ_Jpeg_output_message;
			jpeg_create_decompress(&cinfo);
			cinfo_created = true;
			jpeg_mem_src(&cinfo, pSrc, srcLen);
			if (JPEG_HEADER_OK != jpeg_read_header(&cinfo, TRUE))
			{
				jpeg_abort_decompress(&cinfo);
				jpeg_destroy_decompress(&cinfo);
				cinfo_created = false;
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

			unsigned char *point = dst_buf;
			while (cinfo.output_scanline < height)
			{
				jpeg_read_scanlines(&cinfo, buffer, 1);   // read one line
				memcpy(point, *buffer, width*nChannels);
				point += widthStep;
			}

			jpeg_finish_decompress(&cinfo);
			jpeg_destroy_decompress(&cinfo);
			cinfo_created = false;
			pDst = dst_buf;

			return true;
		}
		
	};
}

#endif