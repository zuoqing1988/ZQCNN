#include "ZQ_CNN_Net_NCHWC.h"
#include "ZQ_CNN_MTCNN_NCHWC.h"
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
				circle(image, cv::Point(*(it->ppoint + num * 2) + 0.5f, *(it->ppoint + num * 2 + 1) + 0.5f), 1, cv::Scalar(0, 255, 255), -1);
		}
		else
		{
			printf("not exist!\n");
		}
	}
}

int main()
{
	int num_threads = 1;
#if ZQ_CNN_USE_BLAS_GEMM
	printf("set openblas thread_num = %d\n", num_threads);
	openblas_set_num_threads(num_threads);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(num_threads);
#endif
#if defined(_WIN32)
	Mat image0 = cv::imread("data/4_320x240.jpg", 1);
#else
	Mat image0 = cv::imread("../../data/4_320x240.jpg", 1);
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
	std::vector<ZQ_CNN_BBox106> thirdBbox106;
	ZQ_CNN_MTCNN_NCHWC mtcnn;
	std::string result_name;
	mtcnn.TurnOnShowDebugInfo();
	//mtcnn.SetLimit(300, 50, 20);
	const int use_pnet20 = true;
	bool landmark106 = false;
	int thread_num = 1;
	bool special_handle_very_big_face = false;
	result_name = "resultdet.jpg";
	if (use_pnet20)
	{
		if (landmark106)
		{
#if defined(_WIN32)
			if (!mtcnn.Init("model/det1-dw20-fast.zqparams", "model/det1-dw20-fast.nchwbin",
				"model/det2-dw24-fast.zqparams", "model/det2-dw24-fast.nchwbin",
				//"model/det2.zqparams", "model/det2_bgr.nchwbin",
				"model/det3-dw48-fast.zqparams", "model/det3-dw48-fast.nchwbin",
				thread_num, true,
				"model/det5-dw64-v3s.zqparams", "model/det5-dw64-v3s.nchwbin"
				//"model/det3.zqparams", "model/det3_bgr.nchwbin"
#else
			if (!mtcnn.Init("../../model/det1-dw20-fast.zqparams", "../../model/det1-dw20-fast.nchwbin",
				"../../model/det2-dw24-fast.zqparams", "../../model/det2-dw24-fast.nchwbin",
				//"../../model/det2.zqparams", "../../model/det2_bgr.nchwbin",
				"../../model/det3-dw48-fast.zqparams", "../../model/det3-dw48-fast.nchwbin",
				thread_num, true,
				"../../model/det5-dw64-v3s.zqparams", "../../model/det5-dw64-v3s.nchwbin"
				//"../../model/det3.zqparams", "../../model/det3_bgr.nchwbin"
#endif
			))
			{
				cout << "failed to init!\n";
				return EXIT_FAILURE;
			}
		}
		else
		{
#if defined(_WIN32)
			if (!mtcnn.Init("model/det1-dw20-fast.zqparams", "model/det1-dw20-fast.nchwbin",
				"model/det2-dw24-fast.zqparams", "model/det2-dw24-fast.nchwbin",
				//"model\\det2.zqparams", "model\\det2_bgr.nchwbin",
				"model/det3-dw48-fast.zqparams", "model/det3-dw48-fast.nchwbin",
				//"model/det3.zqparams", "model/det3_bgr.nchwbin",
				thread_num, false,
				"model/det4-dw48-v2n.zqparams", "model/det4-dw48-v2n.nchwbin"
				//"model/det3.zqparams", "model/det3_bgr.nchwbin"
#else
			if (!mtcnn.Init("../../model/det1-dw20-fast.zqparams", "../../model/det1-dw20-fast.nchwbin",
				"../../model/det2-dw24-fast.zqparams", "../../model/det2-dw24-fast.nchwbin",
				//"model/det2.zqparams", "model/det2_bgr.nchwbin",
				"../../model/det3-dw48-fast.zqparams", "../../model/det3-dw48-fast.nchwbin",
				thread_num, false,
				"model/det4-dw48-v2s.zqparams", "model/det4-dw48-v2s.nchwbin"
				//"../../model/det3.zqparams", "../../model/det3_bgr.nchwbin"
#endif
			))
			{
				cout << "failed to init!\n";
				return EXIT_FAILURE;
			}
		}
		mtcnn.SetPara(image0.cols, image0.rows, 20, 0.5, 0.6, 0.8, 0.4, 0.5, 0.5, 0.709, 3, 20, 4, special_handle_very_big_face);
	}
	else
	{
#if defined(_WIN32)
		if (!mtcnn.Init("model/det1.zqparams", "model/det1_bgr.nchwbin",
			"model/det2.zqparams", "model/det2_bgr.nchwbin",
			"model/det3.zqparams", "model/det3_bgr.nchwbin", thread_num))
#else
		if (!mtcnn.Init("../../model/det1.zqparams", "../../model/det1_bgr.nchwbin",
			"../../model/det2.zqparams", "../../model/det2_bgr.nchwbin",
			"../../model/det3.zqparams", "../../model/det3_bgr.nchwbin", thread_num))
#endif
		{
			cout << "failed to init!\n";
			return EXIT_FAILURE;
		}

		mtcnn.SetPara(image0.cols, image0.rows, 20, 0.6, 0.7, 0.7, 0.4, 0.5, 0.5, 0.709, 4, 12, 2, special_handle_very_big_face);
	}
	mtcnn.TurnOffShowDebugInfo();
	//mtcnn.TurnOnShowDebugInfo();
	int iters = 100;
	double t1 = omp_get_wtime();
	for (int i = 0; i < iters; i++)
	{
		if (i == iters / 2)
			mtcnn.TurnOnShowDebugInfo();
		else
			mtcnn.TurnOffShowDebugInfo();
		if (landmark106 && use_pnet20)
		{
			if (!mtcnn.Find106(image0.data, image0.cols, image0.rows, image0.step[0], thirdBbox106))
			{
				cout << "failed to find face!\n";
				//return EXIT_FAILURE;
				continue;
			}
		}
		else
		{
			if (!mtcnn.Find(image0.data, image0.cols, image0.rows, image0.step[0], thirdBbox))
			{
				cout << "failed to find face!\n";
				//return EXIT_FAILURE;
				continue;
			}
		}
	}
	double t2 = omp_get_wtime();
	printf("total %.3f s / %d = %.3f ms\n", t2 - t1, iters, 1000 * (t2 - t1) / iters);

	namedWindow("result");
	if (landmark106 && use_pnet20)
		Draw(image0, thirdBbox106);
	else
		Draw(image0, thirdBbox);
	imwrite(result_name, image0);
	imshow("result", image0);

	waitKey(0);
	return EXIT_SUCCESS;
}
