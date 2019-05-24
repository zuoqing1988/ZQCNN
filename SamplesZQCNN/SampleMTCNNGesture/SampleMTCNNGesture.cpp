#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_MTCNN_AspectRatio.h"
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

static void Draw(cv::Mat &image, const std::vector<ZQ_CNN_BBox>& thirdBbox)
{
	std::vector<ZQ_CNN_BBox>::const_iterator it = thirdBbox.begin();
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

			for (int num = 0; num < 5; num++)
				circle(image, cv::Point(*(it->ppoint + num) + 0.5f, *(it->ppoint + num + 5) + 0.5f), 1, cv::Scalar(0, 255, 255), -1);
		}
		else
		{
			printf("not exist!\n");
		}
	}
}

int run_fig()
{
	int num_threads = 1;
#if ZQ_CNN_USE_BLAS_GEMM
	printf("set openblas thread_num = %d\n", num_threads);
	openblas_set_num_threads(num_threads);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(num_threads);
#endif
#if defined(_WIN32)
	Mat image0 = cv::imread("data/hand6.jpg", 1);
#else
	Mat image0 = cv::imread("../../data/11.jpg", 1);
#endif
	if (image0.empty())
	{
		cout << "empty image\n";
		return EXIT_FAILURE;
	}
	//cv::resize(image0, image0, cv::Size(), 2, 2);
	if (image0.channels() == 1)
		cv::cvtColor(image0, image0, CV_GRAY2BGR);
	//cv::convertScaleAbs(image0, image0, 2.0);
	/* TIPS: when finding tiny faces for very big image, gaussian blur is very useful for Pnet*/
	bool run_blur = true;
	int kernel_size = 3, sigma = 2;
	if (image0.cols * image0.rows >= 2500 * 1600)
	{
		run_blur = false;
		kernel_size = 5;
		sigma = 3;
	}
	else if (image0.cols * image0.rows >= 1920 * 1080)
	{
		run_blur = false;
		kernel_size = 3;
		sigma = 2;
	}
	else
	{
		run_blur = false;
	}

	if (run_blur)
	{
		cv::Mat blur_image0;
		int nBlurIters = 1000;
		double t00 = omp_get_wtime();
		for (int i = 0; i < nBlurIters; i++)
			cv::GaussianBlur(image0, blur_image0, cv::Size(kernel_size, kernel_size), sigma, sigma);
		double t01 = omp_get_wtime();
		printf("[%d] blur cost %.3f secs, 1 blur costs %.3f ms\n", nBlurIters, t01 - t00, 1000 * (t01 - t00) / nBlurIters);
		cv::GaussianBlur(image0, image0, cv::Size(kernel_size, kernel_size), sigma, sigma);
	}

	std::vector<ZQ_CNN_BBox> thirdBbox;
	ZQ_CNN_MTCNN_AspectRatio mtcnn;
	std::string result_name;
	mtcnn.TurnOnShowDebugInfo();
	//mtcnn.SetLimit(300, 50, 20);
	bool gesture = true;
	int thread_num = 1;
	bool special_handle_very_big_face = false;
	result_name = "resultdet.jpg";

#if defined(_WIN32)
	if (!mtcnn.Init("model/handdet1-dw20-fast.zqparams", "model/handdet1-dw20-fast.nchwbin",
		"model/handdet2-dw24-fast.zqparams", "model/handdet2-dw24-fast.nchwbin",
		"model/handdet3-dw48-fast.zqparams", "model/handdet3-dw48-fast.nchwbin",
		thread_num
#else
	if (!mtcnn.Init("../../model/handdet1-dw20-fast.zqparams", "../../model/handdet1-dw20-fast.nchwbin",
		"../../model/handdet2-dw24-fast.zqparams", "../../model/handdet2-dw24-fast.nchwbin",
		"../../model/handdet3-dw48-fast.zqparams", "../../model/handdet3-dw48-fast.nchwbin",
		thread_num
#endif
	))
	{
		cout << "failed to init!\n";
		return EXIT_FAILURE;
	}

	mtcnn.SetPara(image0.cols, image0.rows, 80, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.709, 3, 20, 4, special_handle_very_big_face);

	mtcnn.TurnOffShowDebugInfo();
	//mtcnn.TurnOnShowDebugInfo();
	int iters = 1;
	double t1 = omp_get_wtime();
	for (int i = 0; i < iters; i++)
	{
		if (i == iters / 2)
			mtcnn.TurnOnShowDebugInfo();
		else
			mtcnn.TurnOffShowDebugInfo();
		if (!mtcnn.Find(image0.data, image0.cols, image0.rows, image0.step[0], thirdBbox))
		{
			cout << "failed to find face!\n";
			//return EXIT_FAILURE;
			continue;
		}
	}
	double t2 = omp_get_wtime();
	printf("total %.3f s / %d = %.3f ms\n", t2 - t1, iters, 1000 * (t2 - t1) / iters);

	namedWindow("result");
	Draw(image0, thirdBbox);
	imwrite(result_name, image0);
	imshow("result", image0);

	waitKey(0);
	return EXIT_SUCCESS;
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

	std::vector<ZQ_CNN_BBox> thirdBbox;
	std::vector<ZQ_CNN_BBox> thirdBbox_last;
	ZQ_CNN_MTCNN_AspectRatio mtcnn;
	std::string result_name;
	//mtcnn.TurnOnShowDebugInfo();
	//mtcnn.SetLimit(300, 50, 20);
	bool gesture = true;
	int thread_num = 3;
	bool special_handle_very_big_face = false;
	result_name = "resultdet.jpg";

#if defined(_WIN32)
	if (!mtcnn.Init("model/handdet1-dw20-fast.zqparams", "model/handdet1-dw20-fast.nchwbin",
		"model/handdet2-dw24-fast.zqparams", "model/handdet2-dw24-fast.nchwbin",
		"model/handdet3-dw48-fast.zqparams", "model/handdet3-dw48-fast.nchwbin",
		thread_num
#else
	if (!mtcnn.Init("../../model/handdet1-dw20-fast.zqparams", "../../model/handdet1-dw20-fast.nchwbin",
		"../../model/handdet2-dw24-fast.zqparams", "../../model/handdet2-dw24-fast.nchwbin",
		"../../model/handdet3-dw48-fast.zqparams", "../../model/handdet3-dw48-fast.nchwbin",
		thread_num
#endif
	))
	{
		cout << "failed to init!\n";
		return EXIT_FAILURE;
	}

	

	//cv::VideoCapture cap("video_20190507_170946.mp4");
	cv::VideoCapture cap(0);
	cv::VideoWriter writer;
	cv::Mat image0;
	cv::namedWindow("show");
	while (true)
	{
		cap >> image0;
		if (image0.empty())
			break;

		cv::resize(image0, image0, cv::Size(), 0.5, 0.5);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		if (!writer.isOpened())
			writer.open("cam1.mp4", CV_FOURCC('D', 'I', 'V', 'X'), 25, cv::Size(image0.cols, image0.rows));
		mtcnn.SetPara(image0.cols, image0.rows, 80, 0.6, 0.7, 0.8, 0.5, 0.5, 0.5, 0.709, 3, 20, 4, special_handle_very_big_face);
		mtcnn.TurnOnShowDebugInfo();
		static int fr_id = 0;
		if (!mtcnn.Find(image0.data, image0.cols, image0.rows, image0.step[0], thirdBbox))
		{
			thirdBbox = thirdBbox_last;
		}
		if (thirdBbox.size() == 0)
		{
			printf("%d\n", fr_id);
		}
		fr_id++;
		Draw(image0, thirdBbox);

		thirdBbox_last = thirdBbox;
		imshow("show", image0);
		writer << image0;
		int key = cv::waitKey(10);
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
