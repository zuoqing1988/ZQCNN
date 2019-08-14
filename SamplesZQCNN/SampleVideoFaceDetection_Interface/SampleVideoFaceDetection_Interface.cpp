#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_Tensor4D.h"
#include "ZQ_CNN_VideoFaceDetection_Interface.h"
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
	using ZQ_CNN_VideoFaceDetection = ZQ_CNN_VideoFaceDetection_Interface<ZQ_CNN_Net, ZQ_CNN_Tensor4D_NHW_C_Align128bit, ZQ_CNN_Tensor4D>;
	ZQ_CNN_VideoFaceDetection detector;
	std::string result_name;
	detector.TurnOffShowDebugInfo();
	detector.TurnOffFilterIOU();
	//mtcnn.SetLimit(300, 50, 20);
	int thread_num = 0;

#if defined(_WIN32)
	if (!detector.Init(
		"model/det1-dw20-plus.zqparams", "model/det1-dw20-plus.nchwbin",
		"model/det2-dw24-p0.zqparams", "model/det2-dw24-p0.nchwbin",
		"model/det3-dw48-p0.zqparams", "model/det3-dw48-p0.nchwbin",
		thread_num, true,
		"model/det5-dw112.zqparams", "model/det5-dw112-16000.nchwbin"
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

	detector.Message(ZQ_CNN_VideoFaceDetection::VFD_MSG_MAX_TRACE_NUM, 6);
	detector.Message(ZQ_CNN_VideoFaceDetection::VFD_MSG_WEIGHT_DECAY, 0.2);
	//cv::VideoCapture cap("video_20190518_172153_540P.mp4");
	//cv::VideoCapture cap("video_20190612_094223.mp4"); 
	//cv::VideoCapture cap("video_20190702_083029.mp4");
	//cv::VideoCapture cap("V90715-124118.mp4");
	//cv::VideoCapture cap("video_20190528_093054_540P.mp4");
	//cv::VideoCapture cap("video_20190806_190129.mp4");
	cv::VideoCapture cap("video_20190809_094755.mp4");
	//cv::VideoCapture cap(0);
	cv::VideoWriter writer;
	cv::Mat image0, ori_im;
	cv::namedWindow("show");
	while (true)
	{
		cap >> image0;

		if (image0.empty())
			break;
		//printf("w x h = %d x %d\n", image0.cols, image0.rows);
		//cv::flip(image0, image0, 1);
		image0 = image0(cv::Rect(656, 0, 607, 1080));
		//cv::resize(image0, image0, cv::Size(), 0.5, 0.5);
		image0.copyTo(ori_im);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		/*cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);*/

		if (!writer.isOpened())
			writer.open("cam-trace6-16000-5-v-7.mp4", CV_FOURCC('X', 'V', 'I', 'D'), 25, cv::Size(image0.cols, image0.rows));
		detector.SetPara(image0.cols, image0.rows, 120, 0.5, 0.6, 0.8, 0.4, 0.5, 0.5, 0.709, 3, 20, 4, 25);

		//mtcnn.TurnOnShowDebugInfo();
		static int fr_id = 0;
		if (!detector.Find(image0.data, image0.cols, image0.rows, image0.step[0], thirdBbox106))
		{
			printf("%d\n", fr_id);
		}

		Draw(ori_im, thirdBbox106);

		imshow("show", ori_im);
		char buf[200];
		sprintf_s(buf, 20, "out5-old\\%d.png", fr_id);
		//cv::imwrite(buf, ori_im);
		writer << ori_im;
		int key = cv::waitKey(20);
		if (key == 27)
			break;

		fr_id++;
	}

	return EXIT_SUCCESS;
}

int main()
{
	//return run_fig();
	return run_cam();
}
