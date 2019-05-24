#include "ZQ_CNN_CascadeOnet.h"
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
	int num_threads = 1;

#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(num_threads);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(num_threads);
#endif

	ZQ_CNN_CascadeOnet net1;
	net1.TurnOnShowDebugInfo();
#if defined(_WIN32)
	if (!net1.Init(
		"model/det3-dw48-plus.zqparams", "model/det3-dw48-plus.nchwbin",
		"model/det3-dw48-plus.zqparams", "model/det3-dw48-plus.nchwbin",
		"model/det3-dw48-plus.zqparams", "model/det3-dw48-plus.nchwbin"))
#else
	if (!net1.Init(
		"../../model/det3-dw48-plus.zqparams", "../../model/det3-dw48-plus.nchwbin",
		"../../model/det3-dw48-plus.zqparams", "../../model/det3-dw48-plus.nchwbin",
		"../../model/det3-dw48-plus.zqparams", "../../model/det3-dw48-plus.nchwbin"))
#endif
	{
		cout << "failed to load model\n";
		return EXIT_FAILURE;
	}
	
#if defined(_WIN32)
	Mat ori_img = imread("data/CascadeOnet.jpg", 1);
#else
	Mat ori_img = imread("../../data/CascadeOnet.jpg", 1);
#endif
	Mat img;
	if (ori_img.empty())
	{
		cout << "failed to load image\n";
		return EXIT_FAILURE;
	}
	if (ori_img.channels() == 1)
		cv::cvtColor(ori_img, ori_img, CV_GRAY2BGR);
	int ori_width = ori_img.cols;
	int ori_height = ori_img.rows;
	Mat draw_img;
	ori_img.copyTo(draw_img);

	std::vector<ZQ_CNN_BBox> results;

	if (!net1.Find(ori_img.data, ori_img.cols, ori_img.rows, ori_img.step[0], 0, 0, ori_img.cols, ori_img.rows, results, 3))
	{
		std::cout << "failed\n";
		return EXIT_FAILURE;
	}
	
	for (int i = 0; i < results.size(); i++)
	{
		Rect rect = Rect(cv::Point(results[i].col1, results[i].row1), cv::Point(results[i].col2, results[i].row2));
		rectangle(draw_img, rect, cv::Scalar(0, results[i].score*255, 0), i+1);
	}
	
	namedWindow("box");

	imshow("box", draw_img);
	imwrite("box.jpg", draw_img);
	waitKey(0);
	return EXIT_SUCCESS;
}