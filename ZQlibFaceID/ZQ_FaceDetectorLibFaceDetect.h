#ifndef _ZQ_FACE_DETECTOR_LIB_FACE_DETECT_H_
#define _ZQ_FACE_DETECTOR_LIB_FACE_DETECT_H_
#pragma once
#include "ZQ_FaceDetector.h"
#include "facedetect-dll.h"
#include <malloc.h>
#ifdef _WIN64
#pragma comment(lib,"libfacedetect-x64.lib")
#else
#pragma comment(lib,"libfacedetect.lib")
#pragma 
#endif
namespace ZQ
{
	class ZQ_FaceDetectorLibFaceDetect : public ZQ_FaceDetector
	{
	private:
		enum CONST_VAL { DETECT_BUFFER_SIZE = 0x20000 };

	public:
		ZQ_FaceDetectorLibFaceDetect() : pBuffer(0)
		{

		}

		~ZQ_FaceDetectorLibFaceDetect()
		{
			if (pBuffer)
				free(pBuffer);
			pBuffer = 0;
		}

		virtual bool Init(const std::string model_root = "model", int thread_num = 1)
		{
			if (pBuffer == 0)
			{
				pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
			}
			return pBuffer != 0;
		}

		virtual bool FindFace(const unsigned char* img, int width, int height, int widthStep, ZQ_PixelFormat pixFmt,
			int min_size, float scale, std::vector<ZQ_CNN_BBox>& bbox)
		{
			if (scale < 1)
				scale = 1.0f / scale;
			return _find_face_roi(img, width, height, widthStep, pixFmt, 0, 0, 1, 1, min_size, scale, bbox);
		}

		virtual bool FindFaceROI(const unsigned char* img, int width, int height, int widthStep, ZQ_PixelFormat pixFmt,
			float roi_min_x, float roi_min_y, float roi_max_x, float roi_max_y,
			int min_size, float scale, std::vector<ZQ_CNN_BBox>& bbox)
		{
			if (scale < 1)
				scale = 1.0f / scale;
			return _find_face_roi(img, width, height, widthStep, pixFmt, roi_min_x, roi_min_y, roi_max_x, roi_max_y,
				min_size, scale, bbox);
		}

	private:
		unsigned char * pBuffer;

	private:
		bool _find_face_roi(const unsigned char* img, int width, int height, int widthStep, ZQ_PixelFormat pixFmt,
			float roi_min_x, float roi_min_y, float roi_max_x, float roi_max_y,
			int min_size, float scale, std::vector<ZQ_CNN_BBox>& bbox)
		{
			if (pBuffer == 0)
				return false;

			int rect_off_x = roi_min_x * width + 0.5f;
			int rect_off_y = roi_min_y * height + 0.5f;
			int rect_width = (roi_max_x - roi_min_x)*width + 0.5f;
			int rect_height = (roi_max_y - roi_min_y)*height + 0.5f;
			if (rect_off_x < 0 || rect_off_y < 0 || rect_width < min_size || rect_height < min_size
				|| rect_off_x + rect_width > width || rect_off_y + rect_height>height)
				return false;

			unsigned char* gray_img = (unsigned char*)malloc(rect_width*rect_height);
			if (gray_img == 0)
				return false;

			switch (pixFmt)
			{
			case ZQ_PIXEL_FMT_GRAY:
				for (int h = 0; h < rect_height; h++)
				{
					for (int w = 0; w < rect_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + w;
						unsigned char* cur_pix_ptr = gray_img + h*rect_width + w;
						cur_pix_ptr[0] = ori_pix_ptr[0];
					}
				}
				break;
			case ZQ_PIXEL_FMT_BGR:
				for (int h = 0; h < rect_height; h++)
				{
					for (int w = 0; w < rect_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w+rect_off_x) * 3;
						unsigned char* cur_pix_ptr = gray_img + h*rect_width + w;
						cur_pix_ptr[0] = ((int)ori_pix_ptr[2] * 299 + (int)ori_pix_ptr[1] * 587 + (int)ori_pix_ptr[0] * 114 + 500) / 1000;
					}
				}
				break;
			case ZQ_PIXEL_FMT_RGB:
				for (int h = 0; h < rect_height; h++)
				{
					for (int w = 0; w < rect_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x) * 3;
						unsigned char* cur_pix_ptr = gray_img + h*rect_width + w;
						cur_pix_ptr[0] = ((int)ori_pix_ptr[0] * 299 + (int)ori_pix_ptr[1] * 587 + (int)ori_pix_ptr[2] * 114 + 500) / 1000;
					}
				}
				break;
			case ZQ_PIXEL_FMT_BGRX:
				for (int h = 0; h < rect_height; h++)
				{
					for (int w = 0; w < rect_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x) * 4;
						unsigned char* cur_pix_ptr = gray_img + h*rect_width + w;
						cur_pix_ptr[0] = ((int)ori_pix_ptr[2] * 299 + (int)ori_pix_ptr[1] * 587 + (int)ori_pix_ptr[0] * 114 + 500) / 1000;
					}
				}
				break;
			case ZQ_PIXEL_FMT_RGBX:
				for (int h = 0; h < rect_height; h++)
				{
					for (int w = 0; w < rect_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x) * 4;
						unsigned char* cur_pix_ptr = gray_img + h*rect_width + w;
						cur_pix_ptr[0] = ((int)ori_pix_ptr[0] * 299 + (int)ori_pix_ptr[1] * 587 + (int)ori_pix_ptr[2] * 114 + 500) / 1000;
					}
				}
				break;
			case ZQ_PIXEL_FMT_XBGR:
				for (int h = 0; h < rect_height; h++)
				{
					for (int w = 0; w < rect_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x) * 4;
						unsigned char* cur_pix_ptr = gray_img + h*rect_width + w;
						cur_pix_ptr[0] = ((int)ori_pix_ptr[3] * 299 + (int)ori_pix_ptr[2] * 587 + (int)ori_pix_ptr[1] * 114 + 500) / 1000;
					}
				}
				break;
			case ZQ_PIXEL_FMT_XRGB:
				for (int h = 0; h < rect_height; h++)
				{
					for (int w = 0; w < rect_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x) * 4;
						unsigned char* cur_pix_ptr = gray_img + h*rect_width + w;
						cur_pix_ptr[0] = ((int)ori_pix_ptr[1] * 299 + (int)ori_pix_ptr[2] * 587 + (int)ori_pix_ptr[3] * 114 + 500) / 1000;
					}
				}
				break;
			default:
				free(gray_img); gray_img = 0;
				return false;
			}


			int* pResults = facedetect_multiview_reinforce(pBuffer, gray_img, rect_width, rect_height, rect_width, scale, 3, min_size, 0, true);
			//int* pResults = facedetect_frontal(pBuffer, gray_img, rect_width, rect_height, rect_width, 1.2f, 5, min_size, 0, true);
			free(gray_img); gray_img = 0;

			if (pResults == 0)
				return false;

			int num = *pResults;
			bbox.resize(num);

			for (int i = 0; i < num; i++)
			{
				short * p = ((short*)(pResults + 1)) + 142 * i;
				int x = p[0];
				int y = p[1];
				int w = p[2];
				int h = p[3];
				int neighbors = p[4];
				int angle = p[5];

				short* landmark = p + 6;
				bbox[i].row1 = y;
				bbox[i].col1 = x;
				bbox[i].row2 = y+h;
				bbox[i].col2 = x+w;
				bbox[i].area = w*h;
				bbox[i].exist = true;
				bbox[i].score = 1;
				bbox[i].ppoint[0] = 1.0 / 6 * (landmark[2 * 36 + 0] + landmark[2 * 37 + 0] + landmark[2 * 38 + 0] + landmark[2 * 39 + 0] + landmark[2 * 40 + 0] + landmark[2 * 41 + 0]);
				bbox[i].ppoint[5] = 1.0 / 6 * (landmark[2 * 36 + 1] + landmark[2 * 37 + 1] + landmark[2 * 38 + 1] + landmark[2 * 39 + 1] + landmark[2 * 40 + 1] + landmark[2 * 41 + 1]);
				bbox[i].ppoint[1] = 1.0 / 6 * (landmark[2 * 42 + 0] + landmark[2 * 43 + 0] + landmark[2 * 44 + 0] + landmark[2 * 45 + 0] + landmark[2 * 46 + 0] + landmark[2 * 47 + 0]);
				bbox[i].ppoint[6] = 1.0 / 6 * (landmark[2 * 42 + 1] + landmark[2 * 43 + 1] + landmark[2 * 44 + 1] + landmark[2 * 45 + 1] + landmark[2 * 46 + 1] + landmark[2 * 47 + 1]);
				bbox[i].ppoint[2] = landmark[2 * 30 + 0];
				bbox[i].ppoint[7] = landmark[2 * 30 + 1];
				bbox[i].ppoint[3] = 0.5*(landmark[2 * 48 + 0] + landmark[2 * 60 + 0]);
				bbox[i].ppoint[8] = 0.5*(landmark[2 * 48 + 1] + landmark[2 * 60 + 1]);
				bbox[i].ppoint[4] = 0.5*(landmark[2 * 54 + 0] + landmark[2 * 64 + 0]);
				bbox[i].ppoint[9] = 0.5*(landmark[2 * 54 + 1] + landmark[2 * 64 + 1]);
			}

			for (int i = 0; i < num; i++)
			{
				bbox[i].col1 += rect_off_x;
				bbox[i].col2 += rect_off_x;
				bbox[i].row1 += rect_off_y;
				bbox[i].row2 += rect_off_y;
				for (int pp = 0; pp < 5; pp++)
				{
					bbox[i].ppoint[pp] += rect_off_x;
					bbox[i].ppoint[pp + 5] += rect_off_y;
				}
			}

			return true;
		}
	};
}
#endif
