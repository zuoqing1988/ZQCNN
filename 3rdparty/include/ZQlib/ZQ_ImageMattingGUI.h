#ifndef _ZQ_IMAGE_MATTING_GUI_H_
#define _ZQ_IMAGE_MATTING_GUI_H_
#pragma once 

#include "ZQ_DoubleImage.h"
#include "ZQ_BinaryImageProcessing.h"
#include "ZQ_ClosedFormImageMatting.h"
#include "ZQ_LazySnappingGUI.h"
#include "ZQ_ImageIO.h"

namespace ZQ
{
	template<class T>
	class ZQ_ImageMattingGUI
	{
	private:
		ZQ_ImageMattingGUI()
		{
			winName = "Image Matting @ Zuo Qing";
			render_winName = "render";
			has_ori_trimap = false;
			background_img = 0;
			zoom_background_img = 0;
			draw_img = 0;
			eraser_half_size = 5;
			eraser_max_half_size = 50;
			eraser_min_half_size = 1;
			eraser_pos_x = 0;
			eraser_pos_y = 0;
			eraser_has_last_pos = false;
			eraser_last_pos_x = 0;
			eraser_last_pos_y = 0;
			erase_mode = false;
			eraser_type = ZQ_ImageMattingGUI::ERASER_TYPE_UNKNOWN;
			updated_flag = false;
			zoom_scale = 5;
			zoom_mode = false;
			zoom_center_x = 0;
			zoom_center_y = 0;
			cur_mouse_pos_x = 0;
			cur_mouse_pos_y = 0;
		}
		~ZQ_ImageMattingGUI()
		{
			_clear();
		}

	public:
		static ZQ_ImageMattingGUI* GetInstance()
		{
			static ZQ_ImageMattingGUI instance;
			return &instance;
		}

		enum EraserType{ERASER_TYPE_FORE, ERASER_TYPE_BACK, ERASER_TYPE_UNKNOWN};
	private:
		std::string winName;
		std::string render_winName;
		ZQ_DImage<T> ori_image;
		bool has_ori_trimap;
		ZQ_DImage<T> ori_trimap;
		ZQ_DImage<T> trimap;
		ZQ_DImage<float> tex;
		ZQ_DImage<float> tex_alpha;
		ZQ_DImage<T> fore;
		ZQ_DImage<T> back;
		ZQ_DImage<T> alpha;
		IplImage* background_img;
		IplImage* zoom_background_img;
		IplImage* draw_img;
		int eraser_half_size;
		int eraser_max_half_size;
		int eraser_min_half_size;
		int eraser_pos_x;
		int eraser_pos_y;
		bool eraser_has_last_pos;
		int eraser_last_pos_x;
		int eraser_last_pos_y;
		bool erase_mode;
		EraserType eraser_type;
		bool updated_flag;
		int zoom_scale;
		bool zoom_mode;
		int zoom_center_x;
		int zoom_center_y;
		int cur_mouse_pos_x;
		int cur_mouse_pos_y;

	public:
		bool Run(const ZQ_DImage<T>& ori_img,  ZQ_DImage<T>& fore, ZQ_DImage<T>& back,
			ZQ_DImage<T>&alpha)
		{
			ZQ_DImage<T> tmp_trimap;
			if (!ZQ_LazySnappingGUI<T>::GetInstance()->Run(ori_img, tmp_trimap))
			{
				printf("failed to specify trimap by lazy snapping!\n");
				return false;
			}
			
			int width = tmp_trimap.width();
			int height = tmp_trimap.height();
			T*& tmp_trimap_data = tmp_trimap.data();
			ZQ_DImage<bool> fore_map(width, height);
			ZQ_DImage<bool> back_map(width, height);
			ZQ_DImage<bool> tmp(width, height);
			bool*& fore_map_data = fore_map.data();
			bool*& back_map_data = back_map.data();
			bool*& tmp_data = tmp.data();
			for (int i = 0; i < width*height; i++)
			{
				if (tmp_trimap_data[i] == 1)
					fore_map_data[i] = true;
				else if (tmp_trimap_data[i] == 0)
					back_map_data[i] = true;
			}
			bool pfilter2D[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
			for (int i = 0; i < 10; i++)
			{
				ZQ_BinaryImageProcessing::Erode(fore_map_data, tmp_data, width, height, pfilter2D, 1, 1);
				fore_map.swap(tmp);
			}
			for (int i = 0; i < 10; i++)
			{
				ZQ_BinaryImageProcessing::Erode(back_map_data, tmp_data, width, height, pfilter2D, 1, 1);
				back_map.swap(tmp);
			}

			for (int i = 0; i < width*height; i++)
			{
				if (fore_map_data[i])
					tmp_trimap_data[i] = 1;
				else if (back_map_data[i])
					tmp_trimap_data[i] = 0;
				else
					tmp_trimap_data[i] = 0.5;
			}

			return Run(ori_img, tmp_trimap, fore, back, alpha);
		}

		bool Run(const ZQ_DImage<T>& ori_img, const ZQ_DImage<T>& ori_tri_img, ZQ_DImage<T>& fore, ZQ_DImage<T>& back, 
			ZQ_DImage<T>&alpha)
		{
			_clear();
			ori_image = ori_img;
			ori_trimap = ori_tri_img;

			int width = ori_image.width();
			int height = ori_image.height();
			int nChannels = ori_image.nchannels();
			if (!ori_trimap.matchDimension(width, height, 1))
			{
				printf("dimension dont match!\n");
				return false;
			}

			T*& ori_image_data = ori_image.data();
			cvReleaseImage(&background_img);
			background_img = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);

			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					if (nChannels == 1)
					{
						CvScalar sca;
						for (int c = 0; c < 3; c++)
							sca.val[c] = ori_image_data[h*width + w] * 255;
						cvSet2D(background_img, h, w, sca);
					}
					else if (nChannels == 3)
					{
						CvScalar sca;
						for (int c = 0; c < 3; c++)
							sca.val[c] = ori_image_data[(h*width + w) * 3 + c] * 255;
						cvSet2D(background_img, h, w, sca);
					}
				}
			}
			cvReleaseImage(&zoom_background_img);
			zoom_background_img = cvCreateImage(cvSize(width*zoom_scale, height*zoom_scale), IPL_DEPTH_8U, 3);
			cvResize(background_img, zoom_background_img, CV_INTER_NN);

			trimap = ori_trimap;
			/////////////////////

			draw_img = cvCloneImage(background_img);
			cvNamedWindow(winName.c_str());
			cvShowImage(winName.c_str(), draw_img);
			cvSetMouseCallback(winName.c_str(), _mouseHandler, this);

			while (true)
			{
				updated_flag = false;
				int c = cvWaitKey(30);
				c = (char)c;
				if (c == 27)
				{
					break;
				}
				else if (c == 'r' || c == 'R')
				{
					trimap = ori_trimap;
					//_draw();
					updated_flag = true;
				}
				else if (c == 's' || c == 'S')
				{
					cvDestroyWindow(render_winName.c_str());
					_saveAsBack();
				}
				else if (c == 'e')
				{
					erase_mode = !erase_mode;
					updated_flag = true;
				}
				else if (c == 'f')
				{
					eraser_type = ERASER_TYPE_FORE;
					updated_flag = true;
				}
				else if (c == 'b')
				{
					eraser_type = ERASER_TYPE_BACK;
					updated_flag = true;
				}
				else if (c == 'o')
				{
					eraser_half_size--;
					eraser_half_size = __max(eraser_half_size, eraser_min_half_size);
					updated_flag = true;
				}
				else if (c == 'p')
				{
					eraser_half_size++;
					eraser_half_size = __min(eraser_half_size, eraser_max_half_size);
					updated_flag = true;
				}
				else if (c == 'u')
				{
					eraser_type = ERASER_TYPE_UNKNOWN;
					updated_flag = true;
				}
				else if (c == 'g')
				{
					cvDestroyWindow(render_winName.c_str());
					_go();
				}
				else if (c == 'z')
				{
					if (!zoom_mode)
					{
						zoom_mode = true;
						zoom_center_x = cur_mouse_pos_x;
						zoom_center_y = cur_mouse_pos_y;
						updated_flag = true;
					}
					else
					{
						zoom_mode = false;
						updated_flag = true;
					}
				}
				if (updated_flag)
				{
					if (!zoom_mode)
					{
						cvReleaseImage(&draw_img);
						draw_img = cvCloneImage(background_img);
					}
					else
					{
						int width = ori_image.width();
						int height = ori_image.height();
						CvRect rect = cvRect(zoom_center_x*zoom_scale - zoom_center_x, zoom_center_y*zoom_scale - zoom_center_y, width, height);
						cvSetImageROI(zoom_background_img, rect);
						cvCopy(zoom_background_img, draw_img);
						cvResetImageROI(zoom_background_img);
					}

					_draw();
				}
				cvNamedWindow(winName.c_str(), 1);
				cvShowImage(winName.c_str(), draw_img);
			}
			fore = this->fore;
			back = this->back;
			alpha = this->alpha;
			return true;
		}

	private:
		void _clear()
		{
			if (background_img)
			{
				cvReleaseImage(&background_img);
				background_img = 0;
			}
			if (zoom_background_img)
			{
				cvReleaseImage(&zoom_background_img);
				zoom_background_img;
			}
			if (draw_img)
			{
				cvReleaseImage(&draw_img);
				draw_img = 0;
			}
		}

		static void _mouseHandler(int event, int x, int y, int flags, void* param)
		{
			ZQ_ImageMattingGUI* ptr = (ZQ_ImageMattingGUI*)param;
			ptr->cur_mouse_pos_x = x;
			ptr->cur_mouse_pos_y = y;
			if (event == CV_EVENT_LBUTTONDOWN)
			{
				if (ptr->erase_mode)
				{
					if (!ptr->zoom_mode)
					{
						ptr->eraser_pos_x = x;
						ptr->eraser_pos_y = y;
					}
					else
					{
						ptr->eraser_pos_x = (x - ptr->zoom_center_x) / ptr->zoom_scale + ptr->zoom_center_x;
						ptr->eraser_pos_y = (y - ptr->zoom_center_y) / ptr->zoom_scale + ptr->zoom_center_y;
					}
					ptr->_erase();
					//ptr->_draw();
					ptr->updated_flag = true;
					ptr->eraser_last_pos_x = ptr->eraser_pos_x;
					ptr->eraser_last_pos_y = ptr->eraser_pos_y;
					ptr->eraser_has_last_pos = true;
				}
			}
			else if (event == CV_EVENT_LBUTTONUP)
			{
				ptr->eraser_has_last_pos = false;
			}
			else if (event == CV_EVENT_MOUSEMOVE)
			{
				if (flags & CV_EVENT_FLAG_LBUTTON)
				{
					if (ptr->erase_mode)
					{
						if (!ptr->zoom_mode)
						{
							ptr->eraser_pos_x = x;
							ptr->eraser_pos_y = y;
						}
						else
						{
							ptr->eraser_pos_x = (x - ptr->zoom_center_x) / ptr->zoom_scale + ptr->zoom_center_x;
							ptr->eraser_pos_y = (y - ptr->zoom_center_y) / ptr->zoom_scale + ptr->zoom_center_y;
						}
						ptr->_erase();
						//ptr->_draw();
						ptr->updated_flag = true;
						ptr->eraser_last_pos_x = ptr->eraser_pos_x;
						ptr->eraser_last_pos_y = ptr->eraser_pos_y;
						ptr->eraser_has_last_pos = true;
					}
				}
				else
				{
					if (ptr->erase_mode)
					{
						if (!ptr->zoom_mode)
						{
							ptr->eraser_pos_x = x;
							ptr->eraser_pos_y = y;
						}
						else
						{
							ptr->eraser_pos_x = (x - ptr->zoom_center_x) / ptr->zoom_scale + ptr->zoom_center_x;
							ptr->eraser_pos_y = (y - ptr->zoom_center_y) / ptr->zoom_scale + ptr->zoom_center_y;
						}
						//ptr->_draw();
						ptr->updated_flag = true;

					}
				}

			}
			else if (event == CV_EVENT_RBUTTONDOWN)
			{
			}

		}

		void _draw()
		{
			if (erase_mode)
			{
				_drawTrimap(draw_img);
				_drawErazer(draw_img);
			}
		}

		void _drawTrimap(IplImage* img)
		{
			double fore_color[4] = { 0.6, 0.6, 0.6, 0.3 };
			double back_color[4] = { 0.6, 0.0, 0.0, 0.3 };
			int width = trimap.width();
			int height = trimap.height();
			T*& trimap_data = trimap.data();
			if (!zoom_mode)
			{
				for (int h = 0; h < height; h++)
				{
					for (int w = 0; w < width; w++)
					{
						if (trimap_data[h*width + w] == 1)
						{
							CvScalar sca = cvGet2D(img, h, w);
							for (int c = 0; c < 3; c++)
							{
								sca.val[c] = fore_color[c] * 255 + (1 - fore_color[3]) * sca.val[c];
							}
							cvSet2D(img, h, w, sca);
						}
						else if (trimap_data[h*width + w] == 0)
						{
							CvScalar sca = cvGet2D(img, h, w);
							for (int c = 0; c < 3; c++)
							{
								sca.val[c] = back_color[c] * 255 + (1 - back_color[3]) * sca.val[c];
							}
							cvSet2D(img, h, w, sca);
						}
					}
				}
			}
			else
			{
				for (int h = 0; h < height; h++)
				{
					for (int w = 0; w < width; w++)
					{
						int real_h = (double)(h - zoom_center_y) / zoom_scale + zoom_center_y + 0.5;
						int real_w = (double)(w - zoom_center_x) / zoom_scale + zoom_center_x + 0.5;
						real_h = __min(height - 1, __max(0, real_h));
						real_w = __min(width - 1, __max(0, real_w));
						if (trimap_data[real_h*width + real_w] == 1)
						{
							CvScalar sca = cvGet2D(img, h, w);
							for (int c = 0; c < 3; c++)
							{
								sca.val[c] = fore_color[c] * 255 + (1 - fore_color[3]) * sca.val[c];
							}
							cvSet2D(img, h, w, sca);
						}
						else if (trimap_data[real_h*width + real_w] == 0)
						{
							CvScalar sca = cvGet2D(img, h, w);
							for (int c = 0; c < 3; c++)
							{
								sca.val[c] = back_color[c] * 255 + (1 - back_color[3]) * sca.val[c];
							}
							cvSet2D(img, h, w, sca);
						}
					}
				}
			}
		}

		void _drawErazer(IplImage* img)
		{
			if (!zoom_mode)
			{
				CvPoint pt1 = cvPoint(eraser_pos_x - eraser_half_size, eraser_pos_y - eraser_half_size);
				CvPoint pt2 = cvPoint(eraser_pos_x - eraser_half_size, eraser_pos_y + eraser_half_size);
				CvPoint pt3 = cvPoint(eraser_pos_x + eraser_half_size, eraser_pos_y + eraser_half_size);
				CvPoint pt4 = cvPoint(eraser_pos_x + eraser_half_size, eraser_pos_y - eraser_half_size);
				CvScalar eraser_color;
				if (eraser_type == ERASER_TYPE_FORE)
				{
					eraser_color = cvScalar(255, 255, 255);
				}
				else if (eraser_type == ERASER_TYPE_BACK)
				{
					eraser_color = cvScalar(0, 0, 0);
				}
				else
					eraser_color = cvScalar(120, 120, 120);
				cvLine(img, pt1, pt2, eraser_color);
				cvLine(img, pt2, pt3, eraser_color);
				cvLine(img, pt3, pt4, eraser_color);
				cvLine(img, pt4, pt1, eraser_color);
			}
			else
			{
				int half_size = eraser_half_size * zoom_scale;
				int pos_x = (eraser_pos_x - zoom_center_x)*zoom_scale + zoom_center_x;
				int pos_y = (eraser_pos_y - zoom_center_y)*zoom_scale + zoom_center_y;
				CvPoint pt1 = cvPoint(pos_x - half_size, pos_y - half_size);
				CvPoint pt2 = cvPoint(pos_x - half_size, pos_y + half_size);
				CvPoint pt3 = cvPoint(pos_x + half_size, pos_y + half_size);
				CvPoint pt4 = cvPoint(pos_x + half_size, pos_y - half_size);
				CvScalar eraser_color;
				if (eraser_type == ERASER_TYPE_FORE)
				{
					eraser_color = cvScalar(255, 255, 255);
				}
				else if (eraser_type == ERASER_TYPE_BACK)
				{
					eraser_color = cvScalar(0, 0, 0);
				}
				else
					eraser_color = cvScalar(120, 120, 120);
				cvLine(img, pt1, pt2, eraser_color);
				cvLine(img, pt2, pt3, eraser_color);
				cvLine(img, pt3, pt4, eraser_color);
				cvLine(img, pt4, pt1, eraser_color);
			}
		}

		void _drawMarker(IplImage* img)
		{
			if (!zoom_mode)
			{
				if (has_marker)
				{
					CvPoint pt1 = cvPoint(marker[0], marker[1]);
					CvPoint pt2 = cvPoint(marker[2], marker[3]);
					CvPoint pt3 = cvPoint(marker[4], marker[5]);
					CvPoint pt4 = cvPoint(marker[6], marker[7]);
					CvScalar marker_color = cvScalar(0, 255, 0);
					cvLine(img, pt1, pt2, marker_color);
					cvLine(img, pt2, pt3, marker_color);
					cvLine(img, pt3, pt4, marker_color);
					cvLine(img, pt4, pt1, marker_color);
				}
			}
			else
			{
				if (has_marker)
				{
					CvPoint pt1 = cvPoint((marker[0] - zoom_center_x)*zoom_scale + zoom_center_x, (marker[1] - zoom_center_y)*zoom_scale + zoom_center_y);
					CvPoint pt2 = cvPoint((marker[2] - zoom_center_x)*zoom_scale + zoom_center_x, (marker[3] - zoom_center_y)*zoom_scale + zoom_center_y);
					CvPoint pt3 = cvPoint((marker[4] - zoom_center_x)*zoom_scale + zoom_center_x, (marker[5] - zoom_center_y)*zoom_scale + zoom_center_y);
					CvPoint pt4 = cvPoint((marker[6] - zoom_center_x)*zoom_scale + zoom_center_x, (marker[7] - zoom_center_y)*zoom_scale + zoom_center_y);
					CvScalar marker_color = cvScalar(0, 255, 0);
					cvLine(img, pt1, pt2, marker_color);
					cvLine(img, pt2, pt3, marker_color);
					cvLine(img, pt3, pt4, marker_color);
					cvLine(img, pt4, pt1, marker_color);
				}
			}

		}

		void _erase()
		{
			int width = trimap.width();
			int height = trimap.height();
			T*& trimap_data = trimap.data();
			double val = 0;
			if (eraser_type == ERASER_TYPE_FORE)
				val = 1;
			else if (eraser_type == ERASER_TYPE_BACK)
				val = 0;
			else
			{
				val = 0.5;
			}


			for (int h = eraser_pos_y - eraser_half_size; h <= eraser_pos_y + eraser_half_size; h++)
			{
				for (int w = eraser_pos_x - eraser_half_size; w <= eraser_pos_x + eraser_half_size; w++)
				{
					if (h >= 0 && h < height && w >= 0 && w < width)
						trimap_data[h*width + w] = val;
				}
			}

			if (eraser_has_last_pos && eraser_last_pos_x >= 0 && eraser_last_pos_x < width && eraser_last_pos_y >= 0 
				&& eraser_last_pos_y < height && eraser_pos_x >= 0 && eraser_pos_x < width && eraser_pos_y >= 0 
				&& eraser_pos_y < height)
			{
				std::vector<ZQ_Vec2D> pixels;
				
				ZQ_ScanLinePolygonFill::FillOneStrokeWithClip(ZQ_Vec2D(eraser_last_pos_x, eraser_last_pos_y), 
					ZQ_Vec2D(eraser_pos_x, eraser_pos_y), eraser_half_size, width, height, pixels);

				for (int p = 0; p < pixels.size(); p++)
				{
					int x = pixels[p].x;
					int y = pixels[p].y;
					if (x >= 0 && x < width && y >= 0 && y < height)
						trimap_data[y*width + x] = val;
				}
			}
		}

		void _go()
		{
			clock_t t1 = clock();
			double eps = 1e-9;
			int win_size = 1;
			int width = ori_image.width();
			int height = ori_image.height();
			ZQ_DImage<bool> consts_map(width, height);
			T*& trimap_data = trimap.data();
			bool*& consts_map_data = consts_map.data();
			for (int i = 0; i < width*height; i++)
			{
				consts_map_data[i] = trimap_data[i] == 0 || trimap_data[i] == 1;
			}
			if (!ZQ_ClosedFormImageMatting::SolveAlpha(ori_image, consts_map, trimap, alpha, eps, win_size, true))
			{
				printf("failed to solve alpha!\n");
				return;
			}

			clock_t t2 = clock();
			printf("solve alpha cost: %.3f seconds\n", 0.001*(t2 - t1));

			int max_iter = 100;
			bool flag = ZQ_ClosedFormImageMatting::SolveForeBack_ori_paper(ori_image, alpha, fore, back, max_iter, true);

			clock_t t3 = clock();
			printf("solve fore&back cost: %.3f seconds\n", 0.001*(t3 - t2));
			
			printf("done\n");
			_renderForeBackAlpha();
		}

		void _saveAsBack()
		{
			int width = ori_image.width();
			int height = ori_image.height();
			int nChannels = ori_image.nchannels();
			back = ori_image;
			fore.allocate(width, height, nChannels);
			alpha.allocate(width, height);

			printf("done\n");
		}

		void _renderForeBackAlpha()
		{
			float background_white = 0.7;
			float background_black = 0.3;
			int width = fore.width();
			int height = fore.height();
			int nChannels = fore.nchannels();
			ZQ_DImage<T> fore_mul_alpha(width,height,3);
			T*& fore_mul_alpha_data = fore_mul_alpha.data();
			T*& fore_data = fore.data();
			T*& alpha_data = alpha.data();
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					int offset = h*width + w;
					float cur_alpha = alpha_data[offset];
					float back_value = background_white * (1 - cur_alpha);
					fore_mul_alpha_data[offset * 3 + 0] = fore_data[offset * 3 + 0] * cur_alpha + back_value;
					fore_mul_alpha_data[offset * 3 + 1] = fore_data[offset * 3 + 1] * cur_alpha + back_value;
					fore_mul_alpha_data[offset * 3 + 2] = fore_data[offset * 3 + 2] * cur_alpha + back_value;
				}
			}
			ZQ_ImageIO::Show(render_winName.c_str(), fore_mul_alpha);
		}
	};
}
#endif