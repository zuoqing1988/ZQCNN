#define _CRT_SECURE_NO_WARNINGS
#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "ZQ_CNN_CompileConfig.h"
#include "ZQ_CNN_PersonPose.h"
#if defined(_WIN32)
#include "ZQlib/ZQ_PutTextCN.h"
#endif
#if ZQ_CNN_USE_BLAS_GEMM
#if __ARM_NEON
#include <openblas/cblas.h>
#else
#include <openblas/cblas.h>
#pragma comment(lib,"libopenblas.lib")
#endif
#elif ZQ_CNN_USE_MKL_GEMM
#include <mkl/mkl.h>
#pragma comment(lib,"mklml.lib")
#else
#pragma comment(lib,"ZQ_GEMM.lib")
#endif

using namespace ZQ;
using namespace std;
using namespace cv;


void Draw(cv::Mat& img, const std::vector<ZQ_CNN_PersonPose::BBox>& output)
{
	for (int nn = 0; nn < output.size(); nn++)
	{
		const ZQ_CNN_PersonPose::BBox& bbox = output[nn];
		cv::rectangle(img, cv::Point(bbox.col1, bbox.row1), cv::Point(bbox.col2, bbox.row2), cv::Scalar(0, 255, 0));
		
		static const int skeleton[26] = {
			0,1,
			2,5,
			2,8,
			5,11,
			8,11,
			2,3,
			3,4,
			5,6,
			6,7,
			8,9,
			9,10,
			11,12,
			12,13
		};
		for (int i = 0; i < 13; i++)
		{
			int id1 = skeleton[i * 2 + 0];
			int id2 = skeleton[i * 2 + 1];
			if (bbox.points[id1 * 3 + 2] > 0 && bbox.points[id2 * 3 + 2] > 0)
			{
				cv::Point pt1(bbox.points[id1 * 3 + 0], bbox.points[id1 * 3 + 1]);
				cv::Point pt2(bbox.points[id2 * 3 + 0], bbox.points[id2 * 3 + 1]);
				cv::line(img, pt1, pt2, cv::Scalar(255, 0, 0), 2);
			}
		}

		char buf[10];

		for (int i = 0; i < 14; i++)
		{
			cv::Point pt = cv::Point(bbox.points[i * 3], bbox.points[i * 3 + 1]);
			if (bbox.points[i * 3 + 2] > 0)
				cv::circle(img, pt, 2, cv::Scalar(0, 0, 250), 2);

#if defined(_WIN32)
			sprintf_s(buf, 10, "%d", i);
#else
			sprintf(buf, "%d", i);
#endif
#if defined(_WIN32)
			ZQ_PutTextCN::PutTextCN(img, buf, pt, cv::Scalar(100, 0, 0), 12);
#endif
		}
	}
}


int main()
{
	int num_threads = 1;

#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(num_threads);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(num_threads);
#endif
	ZQ_CNN_PersonPose detector;


#if defined(_WIN32)
	if (!detector.Init("model/MobileNetSSD_deploy.zqparams", "model/MobileNetSSD_deploy.nchwbin", "detection_out", 15,
		"model/Pose-cpm.zqparams", "model/Pose-cpm.nchwbin", "Convolutional_Pose_Machine/stage_5_out"))
#else
	if (!detector.Init("../../model/MobileNetSSD_deploy.zqparams", "../../model/MobileNetSSD_deploy.nchwbin", "detection_out", 15,
		"../../model/det17-dw112.zqparams", "../../model/det17-dw112-5340.nchwbin", "conv6-3"))
#endif
	{
		cout << "failed to init!\n";
		return EXIT_FAILURE;
	}

#if defined(_WIN32)
	Mat img = imread("data/pose-0911-15.jpg", 1);
#else
	Mat img = imread("../../data/1.jpg", 1);
#endif
	if (img.empty())
	{
		cout << "failed to load image\n";
		return EXIT_FAILURE;
	}
	if (img.channels() == 1)
		cv::cvtColor(img, img, CV_GRAY2BGR);

	std::vector<ZQ_CNN_PersonPose::BBox> output;
	detector.Detect(output, img.data, img.cols, img.rows, img.step[0], 0.5, false);

	Draw(img, output);
	namedWindow("PersonPose");

	imshow("PersonPose", img);
	cv::imwrite("PersonPose.jpg", img);
	waitKey(0);
	return EXIT_SUCCESS;
}