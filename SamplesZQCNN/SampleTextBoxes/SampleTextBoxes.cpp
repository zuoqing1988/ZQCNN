#include "ZQ_CNN_TextBoxes.h"
#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "ZQ_CNN_CompileConfig.h"
#if ZQ_CNN_USE_BLAS_GEMM
#include <openblas/cblas.h>
#pragma comment(lib,"libopenblas.lib")
#elif ZQ_CNN_USE_MKL_GEMM
#include <mkl/mkl.h>
#pragma comment(lib,"mklml.lib")
#else
#pragma comment(lib,"ZQ_GEMM.lib")
#endif
using namespace ZQ;
using namespace std;
using namespace cv;



int main()
{
	int thread_num = 4;
#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(thread_num);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(thread_num);
#endif
	Mat img0 = cv::imread("data/0113.jpg", 1);
	if (img0.empty())
	{
		cout << "empty image\n";
		return EXIT_FAILURE;
	}
	//resize(img0, img0, Size(), 0.5, 0.5);

	ZQ_CNN_TextBoxes detector;
	if (!detector.Init("model/TextBoxes_icdar13.zqparams", "model/TextBoxes_icdar13.nchwbin", "detection_out"))
	{
		printf("failed to init detector!\n");
		return false;
	}

	int out_iter = 1;
	int iters = 1;
	std::vector<ZQ_CNN_TextBoxes::BBox> output;
	const float kScoreThreshold = 0.5f;
	int H = img0.rows, W = img0.cols;
	std::vector<int> target_W, target_H;
	target_H.push_back(H); target_W.push_back(W);
	target_H.push_back(H*0.7); target_W.push_back(W*0.7);
	target_H.push_back(H*0.5); target_W.push_back(W*0.5);
	target_H.push_back(H*0.35); target_W.push_back(W*0.35);
	target_H.push_back(H*0.25); target_W.push_back(W*0.25);
	target_H.push_back(300); target_W.push_back(300);
	for (int out_it = 0; out_it < out_iter; out_it++)
	{
		double t1 = omp_get_wtime();
		for (int it = 0; it < iters; it++)
		{
			if (!detector.Detect(output, img0.data, img0.cols, img0.rows, img0.step[0], kScoreThreshold, target_W, target_H, true))
			{
				cout << "failed to run\n";
				return EXIT_FAILURE;
			}
		}
		double t2 = omp_get_wtime();
		printf("[%d] times cost %.3f s, 1 iter cost %.3f ms\n", iters, t2 - t1, 1000 * (t2 - t1) / iters);
	}
	const char* kClassNames[] = { "__background__", "text" };
	
	// draw
	for (auto& bbox : output)
	{
		cv::Rect rect(bbox.col1, bbox.row1, bbox.col2 - bbox.col1 + 1, bbox.row2 - bbox.row1 + 1);
		cv::rectangle(img0, rect, cv::Scalar(0, 0, 255), 2);
		char buff[300];
#if defined(_WIN32)
		sprintf_s(buff, 300, "%s: %.2f", kClassNames[bbox.label], bbox.score);
#else
		sprintf(buff, "%s: %.2f", kClassNames[bbox.label], bbox.score);
#endif
		cv::putText(img0, buff, cv::Point(bbox.col1, bbox.row1), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));
	}

	cv::imwrite("./textboxes-result.jpg", img0);
	cv::imshow("ZQCNN-TextBoxes", img0);
	cv::waitKey(0);

	return EXIT_SUCCESS;
}
