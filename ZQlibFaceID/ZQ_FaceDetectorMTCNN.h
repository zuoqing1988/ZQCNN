#ifndef _ZQ_FACE_DETECTOR_MTCNN_H_
#define _ZQ_FACE_DETECTOR_MTCNN_H_
#pragma once
#include "ZQ_FaceDetector.h"
#include "ZQ_CNN_MTCNN.h"
namespace ZQ
{
	class ZQ_FaceDetectorMTCNN : public ZQ_FaceDetector
	{
	public:
		ZQ_FaceDetectorMTCNN()
		{
			thresh[0] = 0.6;
			thresh[1] = 0.7;
			thresh[2] = 0.7;
			nms_thresh[0] = 0.5;
			nms_thresh[1] = 0.5;
			nms_thresh[2] = 0.5;
		}
		~ZQ_FaceDetectorMTCNN()
		{

		}

		virtual bool Init(const std::string model_root = "model", int thread_num = 1)
		{
			return mtcnn.Init(model_root+"/det1.zqparams", model_root + "/det1_bgr.nchwbin",
				model_root + "/det2.zqparams", model_root + "/det2_bgr.nchwbin",
				model_root + "/det3.zqparams", model_root + "/det3_bgr.nchwbin", thread_num);
		}
		void SetThresh(float thresh_p, float thresh_r, float thresh_o, float nms_thresh_p, float nms_thresh_r, float nms_thresh_o)
		{
			thresh[0] = thresh_p;
			thresh[1] = thresh_r;
			thresh[2] = thresh_o;
			nms_thresh[0] = nms_thresh_p;
			nms_thresh[1] = nms_thresh_r;
			nms_thresh[2] = nms_thresh_o;
		}

		virtual bool FindFace(const unsigned char* img, int width, int height, int widthStep, ZQ_PixelFormat pixFmt,
			int min_size, float scale, std::vector<ZQ_CNN_BBox>& bbox)
		{
			if (scale > 1)
				scale = 1.0f / scale;
			if (min_size < 12)
				return false;
			
			if (pixFmt == ZQ_PIXEL_FMT_BGR)
			{
				mtcnn.SetPara(width, height, min_size, thresh[0], thresh[1], thresh[2], nms_thresh[0], nms_thresh[1], nms_thresh[2], scale);
				return mtcnn.Find(img, width, height, widthStep, bbox);
			}
			else
			{
				return _find_face_roi(img, width, height, widthStep, pixFmt, 0, 0, 1, 1, min_size, scale, bbox);
			}
		}

		virtual bool FindFaceROI(const unsigned char* img, int width, int height, int widthStep, ZQ_PixelFormat pixFmt,
			float roi_min_x, float roi_min_y, float roi_max_x, float roi_max_y,
			int min_size, float scale, std::vector<ZQ_CNN_BBox>& bbox)
		{
			if (scale > 1)
				scale = 1.0f / scale;
			if (min_size < 12)
				return false;
			if (roi_min_x == 0 && roi_min_y == 0 && roi_max_x == 1 && roi_max_y == 1 && pixFmt == ZQ_PIXEL_FMT_BGR)
			{
				mtcnn.SetPara(width, height, min_size, thresh[0], thresh[1], thresh[2], nms_thresh[0], nms_thresh[1], nms_thresh[2], scale);
				return mtcnn.Find(img, width, height, widthStep, bbox);
			}
			else
			{
				return _find_face_roi(img, width, height, widthStep, pixFmt, 0, 0, 1, 1, min_size, scale, bbox);
			}
		}

	private:
		ZQ_CNN_MTCNN mtcnn;
		float thresh[3];
		float nms_thresh[3];

	private:
		bool _find_face_roi(const unsigned char* img, int width, int height, int widthStep, ZQ_PixelFormat pixFmt,
			float roi_min_x, float roi_min_y, float roi_max_x, float roi_max_y,
			int min_size, float scale, std::vector<ZQ_CNN_BBox>& bbox)
		{
			int rect_off_x = roi_min_x * width + 0.5f;
			int rect_off_y = roi_min_y * height + 0.5f;
			int rect_width = (roi_max_x - roi_min_x)*width + 0.5f;
			int rect_height = (roi_max_y - roi_min_y)*height + 0.5f;
			if (rect_off_x < 0 || rect_off_y < 0 || rect_width < min_size || rect_height < min_size
				|| rect_off_x + rect_width > width || rect_off_y + rect_height>height)
				return false;

			unsigned char* bgr_img = (unsigned char*)malloc(rect_width*rect_height * 3);
			if (bgr_img == 0)
				return false;
			switch (pixFmt)
			{
			case ZQ_PIXEL_FMT_GRAY:
				for (int h = 0; h < rect_height; h++)
				{
					for (int w = 0; w < rect_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x);
						unsigned char* cur_pix_ptr = bgr_img + (h*rect_width + w) * 3;
						cur_pix_ptr[0] = ori_pix_ptr[0];
						cur_pix_ptr[1] = ori_pix_ptr[0];
						cur_pix_ptr[2] = ori_pix_ptr[0];
					}
				}
				break;
			case ZQ_PIXEL_FMT_BGR:
				for (int h = 0; h < rect_height; h++)
				{
					for (int w = 0; w < rect_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x) *3;
						unsigned char* cur_pix_ptr = bgr_img + (h*rect_width + w) * 3;
						cur_pix_ptr[0] = ori_pix_ptr[0];
						cur_pix_ptr[1] = ori_pix_ptr[1];
						cur_pix_ptr[2] = ori_pix_ptr[2];
					}
				}
				break;
			case ZQ_PIXEL_FMT_RGB:
				for (int h = 0; h < rect_height; h++)
				{
					for (int w = 0; w < rect_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x) * 3;
						unsigned char* cur_pix_ptr = bgr_img + (h*rect_width + w) * 3;
						cur_pix_ptr[0] = ori_pix_ptr[2];
						cur_pix_ptr[1] = ori_pix_ptr[1];
						cur_pix_ptr[2] = ori_pix_ptr[0];
					}
				}
				break;
			case ZQ_PIXEL_FMT_BGRX:
				for (int h = 0; h < rect_height; h++)
				{
					for (int w = 0; w < rect_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x) * 4;
						unsigned char* cur_pix_ptr = bgr_img + (h*rect_width + w) * 3;
						cur_pix_ptr[0] = ori_pix_ptr[0];
						cur_pix_ptr[1] = ori_pix_ptr[1];
						cur_pix_ptr[2] = ori_pix_ptr[2];
					}
				}
				break;
			case ZQ_PIXEL_FMT_RGBX:
				for (int h = 0; h < rect_height; h++)
				{
					for (int w = 0; w < rect_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x) * 4;
						unsigned char* cur_pix_ptr = bgr_img + (h*rect_width + w) * 3;
						cur_pix_ptr[0] = ori_pix_ptr[2];
						cur_pix_ptr[1] = ori_pix_ptr[1];
						cur_pix_ptr[2] = ori_pix_ptr[0];
					}
				}
				break;
			case ZQ_PIXEL_FMT_XBGR:
				for (int h = 0; h < rect_height; h++)
				{
					for (int w = 0; w < rect_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x) * 4;
						unsigned char* cur_pix_ptr = bgr_img + (h*rect_width + w) * 3;
						cur_pix_ptr[0] = ori_pix_ptr[1];
						cur_pix_ptr[1] = ori_pix_ptr[2];
						cur_pix_ptr[2] = ori_pix_ptr[3];
					}
				}
				break;
			case ZQ_PIXEL_FMT_XRGB:
				for (int h = 0; h < rect_height; h++)
				{
					for (int w = 0; w < rect_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x) * 4;
						unsigned char* cur_pix_ptr = bgr_img + (h*rect_width + w) * 3;
						cur_pix_ptr[0] = ori_pix_ptr[3];
						cur_pix_ptr[1] = ori_pix_ptr[2];
						cur_pix_ptr[2] = ori_pix_ptr[1];
					}
				}
				break;
			default:
				free(bgr_img); bgr_img = 0;
				return false;
				break;
			}

			mtcnn.SetPara(rect_width, rect_height, min_size, thresh[0], thresh[1], thresh[2], nms_thresh[0], nms_thresh[1], nms_thresh[2], scale);
			bool ret = mtcnn.Find(bgr_img, rect_width, rect_height, rect_width * 3, bbox);
			free(bgr_img); bgr_img = 0;

			if (ret)
			{
				int num = bbox.size();
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
			}

			return ret;
		}
	};
}
#endif
