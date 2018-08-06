#ifndef __ZQ_IMAGE_IO_H_
#define __ZQ_IMAGE_IO_H_
#pragma once

#include <stdio.h>
#include "ZQ_DoubleImage.h"
#include <typeinfo>
#include "opencv2\opencv.hpp"

namespace ZQ
{
	class ZQ_ImageIO
	{
	public:
		static void DrawToWindow(int win_width, int win_height, const std::string& name, const cv::Mat& image)
		{
			int dst_width = win_width;
			int dst_height = win_height;
			int cur_width = image.cols;
			int cur_height = image.rows;
			float dst_ratio = dst_width / (float)dst_height;
			float cur_ratio = cur_width / (float)cur_height;
			if (dst_ratio > cur_ratio)
			{
				int need_width = cur_height*dst_ratio;
				int left = (need_width - cur_width) / 2;
				int right = (need_width - cur_width) - left;
				cv::Mat dst_image;
				cv::copyMakeBorder(image, dst_image, 0, 0, left, right, cv::BORDER_CONSTANT);
				cv::imshow(name, dst_image);
			}
			else if (dst_ratio < cur_ratio)
			{
				int need_height = cur_width / dst_ratio;
				int top = (need_height - cur_height) / 2;
				int bottom = (need_height - cur_height) - top;
				cv::Mat dst_image;
				cv::copyMakeBorder(image, dst_image, top, bottom, 0, 0, cv::BORDER_CONSTANT);
				cv::imshow(name, dst_image);
			}
			else
			{
				cv::imshow(name, image);
			}
		}

		template<class T>
		static void Show(const char* winName, const ZQ_DImage<T>& im)
		{
			int width = im.width();
			int height = im.height();
			int nChannels = im.nchannels();
			const T*& im_data = im.data();
			if (nChannels == 1)
			{
				IplImage* show_img = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
				for (int h = 0; h < height; h++)
				{
					for (int w = 0; w < width; w++)
					{
						cvSetReal2D(show_img, h, w, im_data[h*width + w] * 255);
					}
				}
				cvNamedWindow(winName);
				cvShowImage(winName, show_img);
				cvReleaseImage(&show_img);				
			}
			else if (nChannels <= 4)
			{
				IplImage* show_img = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 4);
				for (int h = 0; h < height; h++)
				{
					for (int w = 0; w < width; w++)
					{
						CvScalar sca;
						for (int c = 0; c < nChannels; c++)
							sca.val[c] = im_data[(h*width + w)*nChannels + c] * 255;
						cvSet2D(show_img, h, w, sca);
					}
				}
				cvNamedWindow(winName);
				cvShowImage(winName, show_img);
				cvReleaseImage(&show_img);
			}
		}

		/*save tga image only support 3 channel image, data are in range [0,1] and arranged [R G B]*/
		template<class T>
		static bool writeTGAimage(const T* data, const int width, const int height, const int nChannels, const char* filename)
		{
			if (nChannels != 3)
			{
				printf("writeTGAImage only support 3 channel image\n");
				return false;
			}

			if (data == 0)
				return false;

			FILE* out = fopen(filename, "wb");
			if (out == 0)
				return false;

			// The image header
			unsigned char header[18] = { 0 }; // char = byte
			header[2] = 2; // truecolor
			header[12] = width % 256;
			header[13] = width / 256;
			header[14] = height % 256;
			header[15] = height / 256;
			header[16] = 24; // bits per pixel
			header[17] = 0x20;

			fwrite(header, 1, 18, out);

			unsigned char* bytes = new unsigned char[height*width * 3];
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					bytes[(y*width + x) * 3 + 0] = __max(0, __min(255, (int)(data[(y*width + x) * 3 + 2] * 255)));
					bytes[(y*width + x) * 3 + 1] = __max(0, __min(255, (int)(data[(y*width + x) * 3 + 1] * 255)));
					bytes[(y*width + x) * 3 + 2] = __max(0, __min(255, (int)(data[(y*width + x) * 3 + 0] * 255)));
				}
			}
			fwrite(bytes, sizeof(unsigned char), width*height * 3, out);
			delete[]bytes;
			fclose(out);
			return true;
		}

		template<class T>
		static bool writePFMimage(const T* data, const int width, const int height, const int nChannels, const char* filename)
		{
			if (nChannels != 3 && nChannels != 1)
			{
				printf("writePFMimage only support 1 or 3 channel image\n");
				return false;
			}

			if (data == 0)
				return false;

			FILE* out = fopen(filename, "wb");
			if (out == 0)
				return false;

			if (nChannels == 1)
			{
				fprintf(out, "Pf\n %d %d 1\n", width, height);
				for (int i = 0; i < width*height; i++)
				{
					float val = data[i];
					fwrite(&val, sizeof(float), 1, out);
				}
				fclose(out);
			}
			else if (nChannels == 3)
			{
				fprintf(out, "PF\n %d %d 1\n", width, height);
				for (int i = 0; i < width*height; i++)
				{
					float rgb[3];
					rgb[0] = data[i * 3 + 0];
					rgb[1] = data[i * 3 + 1];
					rgb[2] = data[i * 3 + 2];
					fwrite(&rgb, sizeof(float), 3, out);
				}
				fclose(out);
			}
			return true;
		}

		/*load save gray image or rgb image*/
		template<class T>
		static bool loadImage(ZQ_DImage<T>& im, const char* filename, int iscolor = 0)
		{
			FILE* in = 0;
			if(0 != fopen_s(&in, filename, "r"))
				return false;
			fclose(in);
			IplImage* img = cvLoadImage(filename, iscolor);
			if (img == NULL)
				return false;

			int width = img->width;
			int height = img->height;
			int nChannels = img->nChannels;

			if (img->nChannels != 3 && img->nChannels != 1)
			{
				printf("channels should be 1 or 3 -- funtion ZQ_ImageIO::LoadImage()\n");
				cvReleaseImage(&img);
				return false;
			}

			im.allocate(width, height, nChannels);

			T*& im_Data = im.data();

			if (_strcmpi(typeid(T).name(), "float") == 0 || _strcmpi(typeid(T).name(), "double") == 0)
			{
				for (int i = 0; i < height; i++)
				{
					for (int j = 0; j < width; j++)
					{
						CvScalar scalar = cvGet2D(img, i, j);
						for (int c = 0; c < nChannels; c++)
							im_Data[(i*width + j)*nChannels + c] = scalar.val[c] / 255.0;
					}
				}
			}
			else
			{
				for (int i = 0; i < height; i++)
				{
					for (int j = 0; j < width; j++)
					{
						CvScalar scalar = cvGet2D(img, i, j);
						for (int c = 0; c < nChannels; c++)
							im_Data[(i*width + j)*nChannels + c] = scalar.val[c];
					}
				}
			}
			

			cvReleaseImage(&img);
			return true;
		}

		template<class T>
		static bool saveImage(const ZQ_DImage<T>& im, const char* filename)
		{
			FILE* out = 0;
			if(0 != fopen_s(&out, filename, "w"))
				return false;
			fclose(out);

			int width = im.width();
			int height = im.height();
			int nChannels = im.nchannels();

			IplImage* img = 0;
			switch (nChannels)
			{
			case 1:
				img = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
				break;
			case 3:
				img = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
				break;
			default:
				return false;
			}

			const T*& im_Data = im.data();

			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					CvScalar scalar;
					for (int c = 0; c < nChannels; c++)
						scalar.val[c] = im_Data[(i*width + j)*nChannels + c] * 255.0;

					cvSet2D(img, i, j, scalar);
				}
			}

			cvSaveImage(filename, img);
			cvReleaseImage(&img);
			return true;
		}

		/*save flow field, color_type is 0 or 1*/
		template<class T>
		static cv::Mat SaveFlowToColorImage(const ZQ_DImage<T>& u, const ZQ_DImage<T>& v, bool user_input, float max_rad, int wheelSize, int color_type, bool display = false)
		{
			int width = u.width();
			int height = v.height();

			T* img;
			if (color_type == 0)
				img = FlowToColor0(width, height, u.data(), v.data(), user_input, max_rad, display);
			else
				img = FlowToColor1(width, height, u.data(), v.data(), user_input, max_rad, display);
			int show_width = width + wheelSize;
			int show_height = MAX(height, wheelSize);
			cv::Mat show_img = cv::Mat(show_height, show_width, CV_MAKETYPE(8, 3),cv::Scalar(0,0,0));
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					show_img.ptr<uchar>(i)[j * 3 + 0] = img[(i*width + j) * 3 + 2];
					show_img.ptr<uchar>(i)[j * 3 + 1] = img[(i*width + j) * 3 + 1];
					show_img.ptr<uchar>(i)[j * 3 + 2] = img[(i*width + j) * 3 + 0];
				}
			}

			delete[]img;
			img = 0;

			cv::Mat colorwheel;
			if (color_type == 0)
				colorwheel = MakeColorWheelImage0(wheelSize, user_input, max_rad);
			else
				colorwheel = MakeColorWheelImage1(wheelSize, user_input, max_rad);

			for (int i = 0; i < wheelSize; i++)
			{
				for (int j = 0; j < wheelSize; j++)
				{
					show_img.ptr<uchar>(i)[(j + width) * 3 + 0] = colorwheel.ptr<uchar>(i)[j * 3 + 0];
					show_img.ptr<uchar>(i)[(j + width) * 3 + 1] = colorwheel.ptr<uchar>(i)[j * 3 + 1];
					show_img.ptr<uchar>(i)[(j + width) * 3 + 2] = colorwheel.ptr<uchar>(i)[j * 3 + 2];
				}
			}
			return show_img;
		}


		template<class T>
		static cv::Mat SaveFlowToColorImage(const ZQ_DImage<T>& u1, const ZQ_DImage<T>& v1, const ZQ_DImage<T>& u2, const ZQ_DImage<T>& v2,
			bool user_input, float max_rad, int wheelSize, int color_type, bool display = false)
		{
			int width1 = u1.width();
			int height1 = v1.height();
			int width2 = u2.width();
			int height2 = v2.height();

			if (height1 != height2)
				return 0;

			int width = width1 + width2;
			int height = height1;

			ZQ_DImage<T> vx(width, height), vy(width, height);

			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width1; j++)
				{
					vx.data()[i*width + j] = u1.data()[i*width1 + j];
					vy.data()[i*width + j] = v1.data()[i*width1 + j];
				}
				for (int j = 0; j < width2; j++)
				{
					vx.data()[i*width + j + width1] = u2.data()[i*width2 + j];
					vy.data()[i*width + j + width1] = v2.data()[i*width2 + j];
				}
			}



			T* img;
			if (color_type == 0)
				img = FlowToColor0(width, height, vx.data(), vy.data(), user_input, max_rad, display);
			else
				img = FlowToColor1(width, height, vx.data(), vy.data(), user_input, max_rad, display);

			int show_width = width + wheelSize;
			int show_height = MAX(height, wheelSize);
			cv::Mat show_img = cv::Mat(show_height, show_width, CV_MAKETYPE(8, 3), cv::Scalar(0, 0, 0));
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					show_img.ptr<uchar>(i)[j * 3 + 0] = img[(i*width + j) * 3 + 2];
					show_img.ptr<uchar>(i)[j * 3 + 1] = img[(i*width + j) * 3 + 1];
					show_img.ptr<uchar>(i)[j * 3 + 2] = img[(i*width + j) * 3 + 0];
				}
			}

			delete[]img;
			img = 0;

			cv::Mat colorwheel;
			if (color_type == 0)
				colorwheel = MakeColorWheelImage0(wheelSize, user_input, max_rad);
			else
				colorwheel = MakeColorWheelImage1(wheelSize, user_input, max_rad);

			for (int i = 0; i < wheelSize; i++)
			{
				for (int j = 0; j < wheelSize; j++)
				{
					show_img.ptr<uchar>(i)[(j + width) * 3 + 0] = colorwheel.ptr<uchar>(i)[j * 3 + 0];
					show_img.ptr<uchar>(i)[(j + width) * 3 + 1] = colorwheel.ptr<uchar>(i)[j * 3 + 1];
					show_img.ptr<uchar>(i)[(j + width) * 3 + 2] = colorwheel.ptr<uchar>(i)[j * 3 + 2];
				}
			}
			return show_img;
		}

		/*handle flow field*/
		template<class T>
		static T* FlowToColor0(const int w, const int h, const T* uu, const T* vv, bool user_input_max_rad = false, float user_max_rad = 10.0, bool display = false)
		{
			T* img = new T[w*h * 3];
			T* u = new T[w*h];
			T* v = new T[w*h];
			memcpy(u, uu, sizeof(T)*w*h);
			memcpy(v, vv, sizeof(T)*w*h);

			double unknown_flow_thresh = 1e9;
			const double PI = 3.1415926535;
			double maxu = -999;
			double maxv = -999;

			double minu = 999;
			double minv = 999;
			double maxrad = -1;

			for (int i = 0; i < w*h; i++)
			{
				if (fabs(u[i]) > unknown_flow_thresh || fabs(v[i]) > unknown_flow_thresh)
				{
					u[i] = 0;
					v[i] = 0;
				}
			}
			for (int i = 0; i < w*h; i++)
			{
				if (maxu < u[i])
					maxu = u[i];
				if (maxv < v[i])
					maxv = v[i];
				if (minu > u[i])
					minu = u[i];
				if (minv > v[i])
					minv = v[i];
				double rad = sqrt(u[i] * u[i] + v[i] * v[i]);
				if (maxrad < rad)
					maxrad = rad;
			}

			if (display)
				printf("max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n", maxrad, minu, maxu, minv, maxv);

			if (maxrad < 1e-9)
				maxrad = 1e-9;

			if (user_input_max_rad)
				maxrad = user_max_rad;

			for (int i = 0; i < w*h; i++)
			{
				u[i] /= maxrad;
				v[i] /= maxrad;
			}

			T* colorwheel = 0;
			int ncols = 0;

			MakeColorWheel(ncols, colorwheel);

			T* rad = new T[w*h];
			T* a = new T[w*h];
			T* fk = new T[w*h];
			int* k0 = new int[w*h];
			int* k1 = new int[w*h];
			T* f = new T[w*h];
			for (int i = 0; i < w*h; i++)
			{
				rad[i] = sqrt(u[i] * u[i] + v[i] * v[i]);
				if (rad[i] > 1)
					rad[i] = 1;
				a[i] = atan2(-v[i], -u[i]) / PI;
				fk[i] = (a[i] + 1) / 2.0*(ncols - 1.0) + 1;
				k0[i] = (int)fk[i];
				k1[i] = k0[i] + 1;
				if (k1[i] == ncols + 1)
					k1[i] = 1;
				f[i] = fk[i] - k0[i];
			}

			for (int c = 0; c < 3; c++)
			{
				for (int i = 0; i < w*h; i++)
				{
					double col0 = colorwheel[(k0[i] - 1) * 3 + c] / 255.0;
					double col1 = colorwheel[(k1[i] - 1) * 3 + c] / 255.0;
					double col = (1.0 - f[i])*col0 + f[i] * col1;
					if (rad[i] <= 1)
						col = 1.0 - rad[i] * (1 - col);
					else
						col *= 0.75;
					img[i * 3 + c] = 255.0*col;
				}
			}
			delete[]colorwheel;
			delete[]rad;
			delete[]a;
			delete[]fk;
			delete[]k0;
			delete[]k1;
			delete[]f;
			delete[]u;
			delete[]v;

			return img;
		}

		template<class T>
		static void MakeColorWheel(int& ncols, T*& colorwheel)
		{
			int RY = 15;
			int YG = 6;
			int GC = 4;
			int CB = 11;
			int BM = 13;
			int MR = 6;

			ncols = RY + YG + GC + CB + BM + MR;

			//colorwheel = zeros(ncols, 3); % r g b
			colorwheel = new T[ncols * 3];
			memset(colorwheel, 0, sizeof(T)*ncols * 3);

			int col = 0;
			//%RY
			for (int i = 0; i < RY; i++)
			{
				colorwheel[i * 3 + 0] = 255.0;
				colorwheel[i * 3 + 1] = 255.0*i / RY;
			}
			col += RY;

			//%YG
			for (int i = 0; i < YG; i++)
			{
				colorwheel[(col + i) * 3 + 0] = 255.0 - 255.0*i / YG;
				colorwheel[(col + i) * 3 + 1] = 255.0;
			}
			col += YG;

			//%GC
			for (int i = 0; i < GC; i++)
			{
				colorwheel[(col + i) * 3 + 1] = 255.0;
				colorwheel[(col + i) * 3 + 2] = 255.0*i / GC;
			}
			col += GC;


			//%CB
			for (int i = 0; i < CB; i++)
			{
				colorwheel[(col + i) * 3 + 1] = 255.0 - 255.0*i / CB;
				colorwheel[(col + i) * 3 + 2] = 255.0;
			}
			col += CB;


			//%BM
			for (int i = 0; i < BM; i++)
			{
				colorwheel[(col + i) * 3 + 2] = 255.0;
				colorwheel[(col + i) * 3 + 0] = 255.0*i / BM;
			}
			col += BM;

			//%MR
			for (int i = 0; i < MR; i++)
			{
				colorwheel[(col + i) * 3 + 2] = 255.0 - 255.0*i / MR;
				colorwheel[(col + i) * 3 + 0] = 255.0;
			}
		}

		template<class T>
		static T* FlowToColor1(const int w, const int h, const T* uu, const T* vv, bool user_input_max_rad = false, float user_max_rad = 10.0, bool display = false)
		{
			T* hsv_img = new T[w*h * 3];
			T* rgb_img = new T[w*h * 3];
			T* u = new T[w*h];
			T* v = new T[w*h];
			memcpy(u, uu, sizeof(T)*w*h);
			memcpy(v, vv, sizeof(T)*w*h);

			double unknown_flow_thresh = 1e9;
			double maxu = -9999;
			double maxv = -9999;

			double minu = 9999;
			double minv = 9999;
			double maxrad = -1;

			for (int i = 0; i < w*h; i++)
			{
				if (fabs(u[i]) > unknown_flow_thresh || fabs(v[i]) > unknown_flow_thresh)
				{
					u[i] = 0;
					v[i] = 0;
				}
			}


			T* theta = new T[w*h];
			T* rho = new T[w*h];


			for (int i = 0; i < w*h; i++)
			{
				theta[i] = atan2(v[i], u[i]);
				rho[i] = sqrt(u[i] * u[i] + v[i] * v[i]);
				theta[i] = (theta[i] + atan(1.0) * 4) / (2.0*atan(1.0) * 4);//pi = 4.0*atan(1.0)
			}

			double max_rho = -1;

			for (int i = 0; i < w*h; i++)
			{
				if (maxu < u[i])
					maxu = u[i];
				if (maxv < v[i])
					maxv = v[i];
				if (minu > u[i])
					minu = u[i];
				if (minv > v[i])
					minv = v[i];
				max_rho = max_rho > rho[i] ? max_rho : rho[i];
			}

			if (display)
				printf("max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n", max_rho, minu, maxu, minv, maxv);

			if (max_rho == 0)
				max_rho = 0.001;

			if (user_input_max_rad)
			{
				for (int i = 0; i < w*h; i++)
					rho[i] /= user_max_rad;
			}
			else
			{
				for (int i = 0; i < w*h; i++)
					rho[i] /= max_rho;
			}

			for (int i = 0; i < w*h; i++)
			{
				hsv_img[i * 3 + 0] = theta[i];
				hsv_img[i * 3 + 1] = 1;
				hsv_img[i * 3 + 2] = rho[i];
			}

			HSV2RGB(w*h, hsv_img, rgb_img);

			delete[]hsv_img;
			delete[]theta;
			delete[]rho;
			delete[]u;
			delete[]v;

			return rgb_img;
		}

		template<class T>
		static void HSV2RGB(const int npixels, const T* hsv_image, T* rgb_image)
		{
			T* h = new T[npixels];
			T* s = new T[npixels];
			T* v = new T[npixels];
			T* k = new T[npixels];
			T* f = new T[npixels];
			T* t = new T[npixels];
			T* n = new T[npixels];
			T* p = new T[npixels];
			T* e = new T[npixels];
			T* r = new T[npixels];
			T* g = new T[npixels];
			T* b = new T[npixels];

			for (int i = 0; i < npixels; i++)
			{
				h[i] = hsv_image[3 * i + 0];
				s[i] = hsv_image[3 * i + 1];
				v[i] = hsv_image[3 * i + 2];
			}

			for (int i = 0; i < npixels; i++)
			{
				h[i] *= 6;
				k[i] = (int)(h[i] - 1e-9);
				f[i] = h[i] - k[i];
				t[i] = 1.0 - s[i];
				n[i] = 1.0 - s[i] * f[i];
				p[i] = 1.0 - (s[i] * (1.0 - f[i]));
				e[i] = 1.0;
				r[i] = (k[i] == 0)*e[i] + (k[i] == 1)*n[i] + (k[i] == 2)*t[i] + (k[i] == 3)*t[i] + (k[i] == 4)*p[i] + (k[i] == 5)*e[i];
				g[i] = (k[i] == 0)*p[i] + (k[i] == 1)*e[i] + (k[i] == 2)*e[i] + (k[i] == 3)*n[i] + (k[i] == 4)*t[i] + (k[i] == 5)*t[i];
				b[i] = (k[i] == 0)*t[i] + (k[i] == 1)*t[i] + (k[i] == 2)*p[i] + (k[i] == 3)*1.0 + (k[i] == 4)*1.0 + (k[i] == 5)*n[i];
			}

			double max_rgb = -1;
			for (int i = 0; i < npixels; i++)
			{
				max_rgb = (r[i] > max_rgb) ? r[i] : max_rgb;
				max_rgb = (g[i] > max_rgb) ? g[i] : max_rgb;
				max_rgb = (b[i] > max_rgb) ? b[i] : max_rgb;
			}
			if (max_rgb == 0)
				max_rgb = 1.0;

			for (int i = 0; i < npixels; i++)
			{
				f[i] = v[i] / max_rgb;

				rgb_image[i * 3 + 0] = f[i] * r[i] * 255;
				rgb_image[i * 3 + 1] = f[i] * g[i] * 255;
				rgb_image[i * 3 + 2] = f[i] * b[i] * 255;
			}

			delete[]h;
			delete[]s;
			delete[]v;
			delete[]k;
			delete[]f;
			delete[]t;
			delete[]n;
			delete[]p;
			delete[]e;
			delete[]r;
			delete[]g;
			delete[]b;
		}

		static cv::Mat MakeColorWheelImage0(int N, bool user_input, float max_rad)
		{
			float cx = (N - 1)*0.5;
			float cy = (N - 1)*0.5;
			float* u = new float[N*N];
			float* v = new float[N*N];
			memset(u, 0, sizeof(float)*N*N);
			memset(v, 0, sizeof(float)*N*N);
			for (int i = 0; i < N; i++)
			{
				for (int j = 0; j < N; j++)
				{
					if ((i - cy)*(i - cy) + (j - cx)*(j - cx) < 0.25*N*N)
					{
						if (user_input)
						{
							v[i*N + j] = (i - cy) / (float)N * 2 * max_rad;
							u[i*N + j] = (j - cx) / (float)N * 2 * max_rad;
						}
						else
						{
							v[i*N + j] = i - cy;
							u[i*N + j] = j - cx;
						}
					}
				}
			}
			float* img = FlowToColor0(N, N, u, v, user_input, max_rad, false);
			cv::Mat show_img = cv::Mat(N, N, CV_MAKETYPE(8, 3));
			for (int i = 0; i < N; i++)
			{
				for (int j = 0; j < N; j++)
				{
					show_img.ptr<uchar>(i)[j * 3 + 0] = img[(i*N + j) * 3 + 2];
					show_img.ptr<uchar>(i)[j * 3 + 1] = img[(i*N + j) * 3 + 1];
					show_img.ptr<uchar>(i)[j * 3 + 2] = img[(i*N + j) * 3 + 0];
				}
			}
			delete[]u;
			delete[]v;
			delete[]img;
			return show_img;
		}

		static cv::Mat MakeColorWheelImage1(int N, bool user_input, float max_rad)
		{
			float cx = (N - 1)*0.5;
			float cy = (N - 1)*0.5;
			float* u = new float[N*N];
			float* v = new float[N*N];
			memset(u, 0, sizeof(float)*N*N);
			memset(v, 0, sizeof(float)*N*N);
			for (int i = 0; i < N; i++)
			{
				for (int j = 0; j < N; j++)
				{
					if ((i - cy)*(i - cy) + (j - cx)*(j - cx) < 0.25*N*N)
					{
						if (user_input)
						{
							v[i*N + j] = (i - cy) / (float)N * 2 * max_rad;
							u[i*N + j] = (j - cx) / (float)N * 2 * max_rad;
						}
						else
						{
							v[i*N + j] = i - cy;
							u[i*N + j] = j - cx;
						}
					}
				}
			}
			float* img = FlowToColor1(N, N, u, v, user_input, max_rad, false);
			cv::Mat show_img = cv::Mat(N, N, CV_MAKETYPE(8, 3));
			for (int i = 0; i < N; i++)
			{
				for (int j = 0; j < N; j++)
				{
					show_img.ptr<uchar>(i)[j * 3 + 0] = img[(i*N + j) * 3 + 2];
					show_img.ptr<uchar>(i)[j * 3 + 1] = img[(i*N + j) * 3 + 1];
					show_img.ptr<uchar>(i)[j * 3 + 2] = img[(i*N + j) * 3 + 0];
				}
			}
			delete[]u;
			delete[]v;
			delete[]img;
			return show_img;
		}
	};
}

#endif