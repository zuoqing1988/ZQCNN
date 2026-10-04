#if defined(_WIN32)
#include "ZQ_FaceDetectorMTCNN.h"
#include "opencv2\opencv.hpp"
#include <iostream>
#include "ZQ_CNN_CompileConfig.h"
#if ZQ_CNN_USE_BLAS_GEMM
#include <openblas\cblas.h>
#pragma comment(lib,"libopenblas.lib")
#elif ZQ_CNN_USE_MKL_GEMM
#include <mkl\mkl.h>
#pragma comment(lib,"mklml.lib")
#else
#pragma comment(lib,"ZQ_GEMM.lib")
#endif

using namespace std;
using namespace cv;
using namespace ZQ;

void Draw(cv::Mat &image, const std::vector<ZQ_CNN_BBox>& thirdBbox)
{
	std::vector<ZQ_CNN_BBox>::const_iterator it = thirdBbox.begin();
	for (; it != thirdBbox.end(); it++)
	{
		if ((*it).exist)
		{
			if (it->score > 0.9)
			{
				cv::rectangle(image, cv::Point((*it).col1, (*it).row1), cv::Point((*it).col2, (*it).row2), cv::Scalar(0, 0, 255), 2, 8, 0);
			}
			else
			{
				cv::rectangle(image, cv::Point((*it).col1, (*it).row1), cv::Point((*it).col2, (*it).row2), cv::Scalar(0, 255, 0), 2, 8, 0);
			}

			for (int num = 0; num < 5; num++)
				circle(image, cv::Point(*(it->ppoint + num) + 0.5f, *(it->ppoint + num + 5) + 0.5f), 3, cv::Scalar(0, 255, 255), -1);
		}
		else
		{
			printf("not exist!\n");
		}
	}
}

int main()
{
#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(1);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(1);
#endif
	ZQ_FaceDetector* mtcnn = new ZQ_FaceDetectorMTCNN();
	if (!mtcnn->Init("model",8))
	{
		cout << "failed to init mtcnn\n";
		return EXIT_FAILURE;
	}
	
	Mat img = imread("data/4.jpg");
	if (img.empty())
	{
		cout << "failed to load image data/4.jpg\n";
		return EXIT_FAILURE;
	}
	vector<ZQ_CNN_BBox> result_mtcnn;
	int iters = 1000;
	double t1 = omp_get_wtime();
	for (int i = 0; i < iters; i++)
	{
		if (!mtcnn->FindFaceROI(img.data, img.cols, img.rows, img.step[0], ZQ_PIXEL_FMT_BGR,
			0.1, 0.1, 0.9, 0.9, 40, 0.709, result_mtcnn))
		{
			cout << "failed to find face using MTCNN\n";
			return EXIT_FAILURE;
		}
	}
	double t2 = omp_get_wtime();
	printf("%d iters cost %.3f secs, 1 iter costs %.3f ms\n", iters, t2 - t1, 1000 * (t2 - t1) / iters);

	// 检出数打出来（2026-10-04 加，附录 GS.3 的补齐）。
	// 口径与上面 Draw() 一致：**只数 exist 为真的框** —— Draw 对 exist 为假的
	// 走的是另一个分支，直接用 size() 会把没检出的那些也算进去。
	// 按用户要求 namedWindow/imshow 全被注释掉了，画好的图**没有任何地方
	// 能看见**，stdout 是结果唯一的出口；没有这一行，这个 sample
	// 在回归里就只是"rc=0 + 一行耗时"，坏掉了也看不出来。
	{
		int n_exist = 0;
		for (size_t i = 0; i < result_mtcnn.size(); i++)
			if (result_mtcnn[i].exist)
				n_exist++;
		printf("final found num: %d\n", n_exist);
	}

	Mat draw_mtcnn;
	img.copyTo(draw_mtcnn);
	Draw(draw_mtcnn, result_mtcnn);
	// namedWindow("MTCNN");
	// imshow("MTCNN", draw_mtcnn);
	// waitKey(0);
	delete mtcnn;
	return EXIT_SUCCESS;
}

#else
#include <stdio.h>
int main(int argc, const char** argv)
{
	printf("%s only support windows\n", argv[0]);
	return 0;
}
#endif
