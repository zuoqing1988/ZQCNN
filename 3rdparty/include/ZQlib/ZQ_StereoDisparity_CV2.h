#ifndef _ZQ_STEREO_DISPARITY_CV2_H_
#define _ZQ_STEREO_DISPARITY_CV2_H_
#pragma once 

#include "opencv2/calib3d/calib3d.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/contrib/contrib.hpp"  
#include "cv.h"
#include "cvaux.h"
#include "cxcore.h"
//#include "highgui.h"
#include "ZQ_DoubleImage.h"

namespace ZQ
{
	class ZQ_StereoDisparity_CV2
	{
	public:
		template<class T>
		static bool Disparity_SGBM(const ZQ_DImage<T>& left_img, const ZQ_DImage<T>& right_img, ZQ_DImage<T>& disparity, const float max_disparity, bool fullDP = false)
		{
			int width = left_img.width();
			int height = left_img.height();
			int nChannels = left_img.nchannels();
			if (!left_img.matchDimension(right_img))
				return false;
			int type;

			switch(nChannels)
			{
				case 1:
					type = CV_8UC1;
					break;
				case 2:
					type = CV_8UC2;
					break;
				case 3:
					type = CV_8UC3;
					break;
				case 4:
					type = CV_8UC4;
					break;
				default:
					return false;
			}
			
			cv::Mat img1(cvSize(width, height), type);
			cv::Mat img2(cvSize(width, height), type);

			int elementSize = img1.elemSize();
			const T*& left_data = left_img.data();
			const T*& right_data = right_img.data();
			for (int h = 0; h < height; h++)
			{
				uchar* ptr1 = img1.ptr<uchar>(h);
				uchar* ptr2 = img2.ptr<uchar>(h);
				for (int w = 0; w < width; w++)
				{
					for (int c = 0; c < nChannels; c++)
					{
						ptr1[w*elementSize + c] = left_data[(h*width + w)*nChannels + c] * 255;
						ptr2[w*elementSize + c] = right_data[(h*width + w)*nChannels + c] * 255;
					}
				}
			}

			cv::Mat disp;
			Disparity_SGBM(img1, img2, disp, max_disparity, fullDP);

			if (!disparity.matchDimension(width, height, 1))
				disparity.allocate(width, height, 1);

			T*& disparity_data = disparity.data();
			for (int h = 0; h < height; h++)
			{
				short* ptr = disp.ptr<short>(h);
				for (int w = 0; w < width; w++)
					disparity_data[h*width + w] = -ptr[w]/16.0f;
			}

			return true;
		}

		template<class T>
		static bool Disparity_GC(const ZQ_DImage<T>& left_img, const ZQ_DImage<T>& right_img, ZQ_DImage<T>& left_disparity, ZQ_DImage<T>& right_disparity, const float max_disparity)
		{
			int width = left_img.width();
			int height = left_img.height();
			int nChannels = left_img.nchannels();
			if (!left_img.matchDimension(right_img))
				return false;
			
			const T*& left_data = left_img.data();
			const T*& right_data = right_img.data();
			IplImage* img1 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
			IplImage* img2 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
			if (nChannels == 1)
			{
				for (int h = 0; h < height; h++)
				{
					uchar* ptr1 = (uchar*)(img1->imageData) + img1->widthStep;
					uchar* ptr2 = (uchar*)(img2->imageData) + img2->widthStep;
					for (int w = 0; w < width; w++)
					{
						ptr1[w] = left_data[h*width + w] * 255;
						ptr2[w] = right_data[h*width + w] * 255;
					}
				}
			}
			else 
			{
				IplImage* color1 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
				IplImage* color2 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
				int real_nchannels = nChannels > 3 ? 3 : nChannels;
				for (int h = 0; h < height; h++)
				{
					for (int w = 0; w < width; w++)
					{
						CvScalar sca1,sca2;
						for (int c = 0; c < real_nchannels; c++)
						{
							sca1.val[c] = left_data[(h*width + w)*nChannels + c] * 255;
							sca2.val[c] = right_data[(h*width + w)*nChannels + c] * 255;
						}
						cvSet2D(color1, h, w, sca1);
						cvSet2D(color2, h, w, sca2);
					}
				}
				cvConvertImage(color1, img1, CV_BGR2GRAY);
				cvConvertImage(color2, img2, CV_BGR2GRAY);
				cvReleaseImage(&color1);
				cvReleaseImage(&color2);
			}
			

			CvMat* left_disp = cvCreateMat(height, width, CV_16S);
			CvMat* right_disp = cvCreateMat(height, width, CV_16S);

			Disparity_GC(img1, img2, left_disp, right_disp, max_disparity);

			if (!left_disparity.matchDimension(width, height, 1))
				left_disparity.allocate(width, height, 1);
			if (!right_disparity.matchDimension(width, height, 1))
				right_disparity.allocate(width, height, 1);

			T*& left_disparity_data = left_disparity.data();
			T*& right_disparity_data = right_disparity.data();
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					int offset = h*width + w;
					left_disparity_data[offset] = cvGetReal2D(left_disp, h, w);
					right_disparity_data[offset] = cvGetReal2D(right_disp, h, w);
				}
			}
			cvReleaseMat(&left_disp);
			cvReleaseMat(&right_disp);
			cvReleaseImage(&img1);
			cvReleaseImage(&img2);
			return true;
		}

		static void Disparity_SGBM(const cv::Mat& left_im, const cv::Mat& right_im, cv::Mat& disparity, const float max_disparity, bool fullDP = false)
		{
			int cn = left_im.channels();
			
			cv::StereoSGBM sgbm;
			sgbm.preFilterCap = 63;
			sgbm.SADWindowSize = 7;			
			sgbm.P1 = 8 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
			sgbm.P2 = 32 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
			sgbm.minDisparity = 0;
			sgbm.numberOfDisparities = max_disparity;
			sgbm.uniquenessRatio = 15;
			sgbm.speckleWindowSize = 100;
			sgbm.speckleRange = 32;
			sgbm.disp12MaxDiff = 1;
			sgbm.fullDP = fullDP;

			sgbm(left_im, right_im, disparity);
		}

		static void Disparity_GC(const CvArr* left_gray, const CvArr* right_gray, CvArr* left_disp, CvArr* right_disp, const float max_disparity)
		{
			CvStereoGCState* state = cvCreateStereoGCState(max_disparity, 2);
			cvFindStereoCorrespondenceGC(left_gray, right_gray, left_disp, right_disp, state, 0);
			cvReleaseStereoGCState(&state);
		}
	};
}

#endif