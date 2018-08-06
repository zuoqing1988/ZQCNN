#ifndef _ZQ_LAZY_SNAPPING_GUI_H_
#define _ZQ_LAZY_SNAPPING_GUI_H_
#pragma once

#include <opencv\cv.h>
#include <vector>
#include "ZQ_DoubleImage.h"
#include "ZQ_LazySnapping.h"
#include "ZQ_StructureFromTexture.h"
#include "ZQ_ScanLinePolygonFill.h"

namespace ZQ
{
	template<class T>
	class ZQ_LazySnappingGUI
	{
	private:
		ZQ_LazySnappingGUI()
		{
			winName = "Lazy Snapping @ Zuo Qing";
			ZQ_LazySnappingGUI::currentMode = 0;
			image = 0;
			imageDraw = 0;
			has_scaled = false;
			lazySnap = 0;
			cur_mouse_pos_x = 0;
			cur_mouse_pos_y = 0;
			has_last_pos = false;
			last_mouse_pos_x = 0;
			last_mouse_pos_y = 0;
			color_draw[0][0] = 0;	color_draw[0][1] = 0; color_draw[0][2] = 255;
			color_draw[1][0] = 255;	color_draw[1][1] = 0; color_draw[1][2] = 0;
		}
		~ZQ_LazySnappingGUI() 
		{
			_clear();
		}
	public:
		static ZQ_LazySnappingGUI<T>* GetInstance()
		{
			static ZQ_LazySnappingGUI instance;
			return &instance;
		}
		static const int MAX_CLUSTER_NUM = 32;
		static const int standard_width = 640;
		static const int standard_height = 480;
	private:
		std::string winName;
		std::vector<CvPoint> forePts;
		std::vector<CvPoint> backPts;
		std::vector<CvPoint> add_forePts;
		std::vector<CvPoint> add_backPts;
		int currentMode;
		IplImage* image;
		IplImage* imageDraw;
		ZQ_DImage<T> scaled_image, im2;
		ZQ_LazySnappingOptions ls_opt;
		ZQ_StructureFromTextureOptions opt;
		ZQ_DImage<bool> mask;
		bool has_scaled;
		ZQ_LazySnapping<T, MAX_CLUSTER_NUM>* lazySnap;
		int cur_mouse_pos_x;
		int cur_mouse_pos_y;
		bool has_last_pos;
		int last_mouse_pos_x;
		int last_mouse_pos_y;
		double color_draw[2][3];

	public:
		bool Run(const ZQ_DImage<T>& ori_image, ZQ_DImage<T>& tri_map)
		{
			_clear();
			opt.fsize_for_filter = 1;
			ls_opt.dilate_erode_size = 2;

			int ori_width = ori_image.width();
			int ori_height = ori_image.height();
			if (ori_width > standard_width || ori_height > standard_height)
			{
				double scale_x = (double)standard_width / ori_width;
				double scale_y = (double)standard_height / ori_height;
				double scale = __min(scale_x, scale_y);
				int dst_width = ori_width*scale + 0.5;
				int dst_height = ori_height*scale + 0.5;
				ori_image.imresize(scaled_image, dst_width, dst_height);
				im2 = scaled_image;
				if (!ZQ_StructureFromTexture::StructureFromTexture(scaled_image, im2, opt))
				{
					printf("failed to run StructureFromTexture\n");
					return false;
				}
				has_scaled = true;
			}
			else
			{
				im2 = ori_image;
				if (!ZQ_StructureFromTexture::StructureFromTexture(ori_image, im2, opt))
				{
					printf("failed to run StructureFromTexture\n");
					return false;
				}
				has_scaled = false;
			}


			int width = im2.width();
			int height = im2.height();
			int nChannels = im2.nchannels();
			mask.allocate(width, height, 1);

			lazySnap = new ZQ_LazySnapping<T, MAX_CLUSTER_NUM>(width, height);
			//lazySnap->SetEnableE3(true);
			lazySnap->SetImage(im2, ls_opt.lambda_for_E2, ls_opt.color_scale_for_E2, ls_opt.lambda_for_E3, ls_opt.sigma_for_E3);


			image = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
			T*& im2_data = im2.data();
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					int offset = h*width + w;
					if (nChannels == 1)
						cvSet2D(image, h, w, cvScalar(im2_data[offset] * 255, im2_data[offset] * 255, im2_data[offset] * 255));
					else
						cvSet2D(image, h, w, cvScalar(im2_data[offset * 3 + 0] * 255, im2_data[offset * 3 + 1] * 255, im2_data[offset * 3 + 2] * 255));
				}
			}
			cvNamedWindow(winName.c_str(), 1);
			cvSetMouseCallback(winName.c_str(), _mouseHandler, this);
			imageDraw = cvCloneImage(image);
			cvShowImage(winName.c_str(), image);

			while (true)
			{
				int c = cvWaitKey(0);
				c = (char)c;
				if (c == 27 || c == 's' || c == 'S')
				{
					break;
				}
				else if (c == 'e' || c == 'E')
				{
					cvReleaseImage(&imageDraw);
					imageDraw = cvCloneImage(image);
					bool b = lazySnap->GetEnabledE3();
					lazySnap->SetEnableE3(!b);
					if (!ZQ_LazySnapping<T, MAX_CLUSTER_NUM>::FilterMask(lazySnap->GetForegroundMaskPtr(), mask.data(), mask.width(), mask.height(),
						ls_opt.area_thresh, ls_opt.dilate_erode_size))
					{
						memcpy(mask.data(), lazySnap->GetForegroundMaskPtr(), sizeof(bool)*mask.width()*mask.height());
					}
					_drawMask(imageDraw, mask);
					_drawSelectPoints(imageDraw, forePts, backPts, color_draw);
					cvShowImage(winName.c_str(), imageDraw);
				}
				else if (c == 'r' || c == 'R')
				{
					lazySnap->SetImage(im2, ls_opt.lambda_for_E2, ls_opt.color_scale_for_E2, ls_opt.lambda_for_E3, ls_opt.sigma_for_E3);
					imageDraw = cvCloneImage(image);
					forePts.clear();
					backPts.clear();
					add_forePts.clear();
					add_backPts.clear();
					currentMode = 0;
					cvShowImage(winName.c_str(), image);
				}
				else if (c == 'b' || c == 'B')
				{
					currentMode = 1;
				}
				else if (c == 'f' || c == 'F')
				{
					currentMode = 0;
				}
				cvNamedWindow(winName.c_str(), 1);
			}
			cvReleaseImage(&image);
			cvReleaseImage(&imageDraw);
			if (has_scaled)
			{
				mask.imresize(ori_width, ori_height);
			}

			tri_map.allocate(ori_width, ori_height);
			for (int i = 0; i < ori_width*ori_height; i++)
			{
				tri_map.data()[i] = mask.data()[i] ? 1.0 : 0;
			}
			cvDestroyWindow(winName.c_str());
			return true;
		}

	private:
		void _clear()
		{
			forePts.clear();
			backPts.clear();
			add_forePts.clear();
			add_backPts.clear();
			currentMode = 0;
			if (image)
			{
				cvReleaseImage(&image);
			}
			if (imageDraw)
			{
				cvReleaseImage(&imageDraw);
			}

			scaled_image.clear();
			im2.clear();
			mask.clear();
			has_scaled = false;
			if (lazySnap)
			{
				delete lazySnap;
				lazySnap = 0;
			}
			cur_mouse_pos_x = 0;
			cur_mouse_pos_y = 0;
			has_last_pos = false;
			last_mouse_pos_x = 0;
			last_mouse_pos_y = 0;
		}

		static void _mouseHandler(int event, int x, int y, int flags, void* param)
		{
			ZQ_LazySnappingGUI* ptr = (ZQ_LazySnappingGUI*)param;
			ptr->cur_mouse_pos_x = x;
			ptr->cur_mouse_pos_y = y;
			if (event == CV_EVENT_LBUTTONUP)
			{
				int add_fore_num = ptr->add_forePts.size();
				int add_back_num = ptr->add_backPts.size();
				if (add_fore_num == 0 && add_back_num == 0)
				{
					return;
				}
				cvReleaseImage(&ptr->imageDraw);
				ptr->imageDraw = cvCloneImage(ptr->image);
				int* add_fore_pts = NULL;
				int* add_back_pts = NULL;
				if (add_fore_num > 0)
				{
					add_fore_pts = new int[add_fore_num * 2];

					for (int i = 0; i < add_fore_num; i++)
					{
						add_fore_pts[i * 2 + 0] = ptr->add_forePts[i].x;
						add_fore_pts[i * 2 + 1] = ptr->add_forePts[i].y;
					}
					if (!ptr->lazySnap->EditSnappingAddForeground(add_fore_num, add_fore_pts))
					{
						delete[]add_fore_pts;
						add_fore_pts = NULL;
					}
					else
					{
						ptr->forePts.insert(ptr->forePts.end(), ptr->add_forePts.begin(), ptr->add_forePts.end());
						ptr->add_forePts.clear();
					}
				}

				if (add_back_num > 0)
				{
					add_back_pts = new int[add_back_num * 2];

					for (int i = 0; i < add_back_num; i++)
					{
						add_back_pts[i * 2 + 0] = ptr->add_backPts[i].x;
						add_back_pts[i * 2 + 1] = ptr->add_backPts[i].y;

					}

					if (!ptr->lazySnap->EditSnappingAddBackground(add_back_num, add_back_pts))
					{
						delete[]add_back_pts;
						add_back_pts = NULL;
					}
					else
					{
						ptr->backPts.insert(ptr->backPts.end(), ptr->add_backPts.begin(), ptr->add_backPts.end());
						ptr->add_backPts.clear();
					}
				}
				if (!ZQ_LazySnapping<T, MAX_CLUSTER_NUM>::FilterMask(ptr->lazySnap->GetForegroundMaskPtr(), 
					ptr->mask.data(), ptr->mask.width(), ptr->mask.height(),
					ptr->ls_opt.area_thresh, ptr->ls_opt.dilate_erode_size))
				{
					memcpy(ptr->mask.data(), ptr->lazySnap->GetForegroundMaskPtr(), sizeof(bool)*ptr->mask.width()*ptr->mask.height());
				}
				_drawMask(ptr->imageDraw, ptr->mask);
				if (add_fore_pts)
				{
					delete[]add_fore_pts;
				}
				add_fore_pts = 0;
				if (add_back_pts)
				{
					delete[]add_back_pts;
				}
				add_back_pts = 0;

				_drawSelectPoints(ptr->imageDraw, ptr->forePts, ptr->backPts, ptr->color_draw);
				cvShowImage(ptr->winName.c_str(), ptr->imageDraw);

				ptr->has_last_pos = false;
			}
			else if (event == CV_EVENT_LBUTTONDOWN)
			{
				ptr->last_mouse_pos_x = ptr->cur_mouse_pos_x;
				ptr->last_mouse_pos_y = ptr->cur_mouse_pos_y;
				ptr->has_last_pos = true;
			}
			else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))
			{
				if (!ptr->has_last_pos)
				{
					CvPoint pt = cvPoint(x, y);
					if (ptr->currentMode == 0)
					{
						ptr->add_forePts.push_back(pt);
					}
					else
					{
						ptr->add_backPts.push_back(pt);
					}
					CvScalar color = cvScalar(ptr->color_draw[ptr->currentMode][0], ptr->color_draw[ptr->currentMode][1], 
						ptr->color_draw[ptr->currentMode][2]);
					cvCircle(ptr->imageDraw, pt, 2, color);
					cvShowImage(ptr->winName.c_str(), ptr->imageDraw);
				}
				else
				{
					int im_width = ptr->imageDraw->width;
					int im_height = ptr->imageDraw->height;
					if (ptr->last_mouse_pos_x >= 0 && ptr->last_mouse_pos_x < im_width 
						&& ptr->last_mouse_pos_y >= 0 && ptr->last_mouse_pos_y < im_height 
						&& ptr->cur_mouse_pos_x >= 0 && ptr->cur_mouse_pos_x < im_width 
						&& ptr->cur_mouse_pos_y >= 0 && ptr->cur_mouse_pos_y < im_height)
					{

						std::vector<ZQ_Vec2D> pixels;
						ZQ_ScanLinePolygonFill::FillOneStrokeWithClip(
							ZQ_Vec2D(ptr->last_mouse_pos_x, ptr->last_mouse_pos_y), 
							ZQ_Vec2D(ptr->cur_mouse_pos_x, ptr->cur_mouse_pos_y), 1, im_width, im_height, pixels);

						for (int p = 0; p < pixels.size(); p++)
						{
							CvPoint pt = cvPoint(pixels[p].x, pixels[p].y);
							if (ptr->currentMode == 0)
							{
								ptr->add_forePts.push_back(pt);
							}
							else
							{
								ptr->add_backPts.push_back(pt);
							}
							CvScalar color = cvScalar(ptr->color_draw[ptr->currentMode][0], ptr->color_draw[ptr->currentMode][1], 
								ptr->color_draw[ptr->currentMode][2]);
							cvCircle(ptr->imageDraw, pt, 2, color);
						}
						cvShowImage(ptr->winName.c_str(), ptr->imageDraw);
					}
				}
				ptr->last_mouse_pos_x = ptr->cur_mouse_pos_x;
				ptr->last_mouse_pos_y = ptr->cur_mouse_pos_y;
				ptr->has_last_pos = true;
			}
		}

		static void _drawMask(IplImage* imageDraw, const ZQ_DImage<bool>& mask)
		{
			_drawMask(imageDraw, mask.data());
		}

		static void _drawMask(IplImage* imageDraw, const bool* mask)
		{
			int width = imageDraw->width;
			int height = imageDraw->height;
			CvScalar border_color = cvScalar(0, 255, 0);
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					int offset = h*width + w;
					if (!mask[offset] && ((w > 0 && mask[offset - 1]) || (w < width - 1 && mask[offset + 1]) || (h > 0 && mask[offset - width]) || (h < height - 1 && mask[offset + width])))
					{
						//cvSet2D(imageDraw, h, w, border_color);
						cvCircle(imageDraw, cvPoint(w, h), 2, border_color, 2);
					}

				}
			}
		}

		static void _drawSelectPoints(IplImage* imageDraw, const std::vector<CvPoint>& forePts, const std::vector<CvPoint>& backPts, const double color_draw[2][3])
		{
			{
				for (int i = 0; i < forePts.size(); i++)
				{
					CvScalar color = cvScalar(color_draw[0][0], color_draw[0][1], color_draw[0][2]);
					cvCircle(imageDraw, forePts[i], 2, color);
				}
				for (int i = 0; i < backPts.size(); i++)
				{
					CvScalar color = cvScalar(color_draw[1][0], color_draw[1][1], color_draw[1][2]);
					cvCircle(imageDraw, backPts[i], 2, color);
				}
			}
		}
	};
}

#endif