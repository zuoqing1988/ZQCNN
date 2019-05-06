#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_MTCNN_AspectRatio.h"
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
	
	std::vector<ZQ_CNN_BBox> thirdBbox;
	ZQ_CNN_MTCNN_AspectRatio mtcnn;
	std::string result_name;
	mtcnn.TurnOnShowDebugInfo();
	//mtcnn.SetLimit(30, 5);
	int thread_num = 0;
	bool special_handle_very_big_face = false;
	result_name = "MTCNN-AspectRatio.jpg";

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

	mtcnn.SetPara(image0.cols, image0.rows, 20, 0.5, 0.6, 0.7, 0.5, 0.5, 0.5, 0.709, 3, 20, 4, special_handle_very_big_face);

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

	namedWindow("MTCNN-AspectRatio");
	Draw(image0, thirdBbox);
	imwrite(result_name, image0);
	imshow("MTCNN-AspectRatio", image0);

	waitKey(0);
	return EXIT_SUCCESS;
}
