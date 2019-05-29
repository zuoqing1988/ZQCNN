#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_VideoFaceDetection.h"
#if defined(_WIN32)
#include "ZQlib/ZQ_PutTextCN.h"
#endif
#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "ZQ_CNN_CompileConfig.h"
#if ZQ_CNN_USE_BLAS_GEMM
#if __ARM_NEON
#include <openblas/cblas.h>
#else
#include <openblas/cblas.h>
#pragma comment(lib,"libopenblas.lib")
#endif
#elif ZQ_CNN_USE_MKL_GEMM
#include "mkl/mkl.h"
#pragma comment(lib,"mklml.lib")
#else
#pragma comment(lib,"ZQ_GEMM.lib")
#endif
using namespace ZQ;
using namespace std;
using namespace cv;


static void Draw(cv::Mat &image, const std::vector<ZQ_CNN_BBox106>& thirdBbox)
{
	std::vector<ZQ_CNN_BBox106>::const_iterator it = thirdBbox.begin();
	for (; it != thirdBbox.end(); it++)
	{
		if ((*it).exist)
		{
			if (it->score > 0.7)
			{
				cv::rectangle(image, cv::Point((*it).col1, (*it).row1), cv::Point((*it).col2, (*it).row2), cv::Scalar(0, 0, 255), 2, 8, 0);
			}
			else
			{
				cv::rectangle(image, cv::Point((*it).col1, (*it).row1), cv::Point((*it).col2, (*it).row2), cv::Scalar(0, 255, 0), 2, 8, 0);
			}

			for (int num = 0; num < 106; num++)
				circle(image, cv::Point(*(it->ppoint + num * 2) + 0.5f, *(it->ppoint + num * 2 + 1) + 0.5f), 2, cv::Scalar(0, 255, 0), -1);
		}
		else
		{
			printf("not exist!\n");
		}
	}
}

int run_cam()
{
	int num_threads = 1;
#if ZQ_CNN_USE_BLAS_GEMM
	printf("set openblas thread_num = %d\n", num_threads);
	openblas_set_num_threads(num_threads);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(num_threads);
#endif

	std::vector<ZQ_CNN_BBox106> thirdBbox106;
	ZQ_CNN_VideoFaceDetection detector;
	std::string result_name;
	detector.TurnOffShowDebugInfo();
	//mtcnn.SetLimit(300, 50, 20);
	int thread_num = 0;
	
#if defined(_WIN32)
	if (!detector.Init(
		"model/det1-dw20-plus.zqparams", "model/det1-dw20-plus.nchwbin",
		"model/det2-dw24-p0.zqparams", "model/det2-dw24-p0.nchwbin",
		"model/det3-dw48-p0.zqparams", "model/det3-dw48-p0.nchwbin",
		thread_num, true,
		"model/det5-dw112.zqparams", "model/det5-dw112.nchwbin"
#else
	if (!detector.Init(
		"../../model/det1-dw20-plus.zqparams", "../../model/det1-dw20-plus.nchwbin",
		"../../model/det2-dw24-plus.zqparams", "../../model/det2-dw24-plus.nchwbin",
		"../../model/det3-dw48-plus.zqparams", "../../model/det3-dw48-plus.nchwbin",
		thread_num, true,
		"../../model/det5-dw112.zqparams", "../../model/det5-dw112.nchwbin"
#endif
	))
	{
		cout << "failed to init!\n";
		return EXIT_FAILURE;
	}

	detector.Message(ZQ_CNN_VideoFaceDetection::VFD_MSG_WEIGHT_DECAY, 0.5);
	//cv::VideoCapture cap("video_20190518_172153_540P.mp4");
	//cv::VideoCapture cap("video_20190528_093741.mp4"); 
	cv::VideoCapture cap(0);
	cv::VideoWriter writer;
	cv::Mat image0;
	cv::namedWindow("show");
	while (true)
	{
		cap >> image0;
		if (image0.empty())
			break;

		//cv::resize(image0, image0, cv::Size(), 0.5, 0.5);
		//cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		//cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		if (!writer.isOpened())
			writer.open("cam-trace4.mp4", CV_FOURCC('X', 'V', 'I', 'D'), 25, cv::Size(image0.cols, image0.rows));
		detector.SetPara(image0.cols, image0.rows, 120, 0.5, 0.6, 0.8, 0.4, 0.5, 0.5, 0.709, 3, 20, 4, 25);
		
		//mtcnn.TurnOnShowDebugInfo();
		static int fr_id = 0;
		if (!detector.Find(image0.data, image0.cols, image0.rows, image0.step[0], thirdBbox106))
		{
			printf("%d\n", fr_id);
		}
		
		fr_id++;
		
		Draw(image0, thirdBbox106);
		
		imshow("show", image0);
		writer << image0;
		int key = cv::waitKey(20);
		if (key == 27)
			break;
	}

	return EXIT_SUCCESS;
}

int main()
{
	//return run_fig();
	return run_cam();
}
