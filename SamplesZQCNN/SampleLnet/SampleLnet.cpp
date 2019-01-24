#include "ZQ_CNN_Net.h"
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

	ZQ_CNN_Net net1,net2;
	if (!net1.LoadFrom("model/det4-dw48-normal.zqparams", "model/det4-dw48-normal-770.nchwbin")
	//if (!net1.LoadFrom("model/det3-dw48.zqparams", "model/det3-dw48-5.nchwbin")
	||!net2.LoadFrom("model/det3.zqparams", "model/det3_bgr.nchwbin"))
	//|| !net2.LoadFrom("model/det4-dw48-1.zqparams", "model/det4-dw48-1-7.nchwbin"))
	{
		cout << "failed to load model\n";
		return EXIT_FAILURE;
	}
	printf("num_MulAdd = %.1f M\n", net1.GetNumOfMulAdd() / (1024.0*1024.0));
	printf("num_MulAdd = %.1f M\n", net2.GetNumOfMulAdd() / (1024.0*1024.0));

	Mat img = imread("data/onet6.jpg",1);
	if (img.empty())
	{
		cout << "failed to load image\n";
		return EXIT_FAILURE;
	}
	if (img.channels() == 1)
		cv::cvtColor(img, img, CV_GRAY2BGR);
	cv::resize(img, img, cv::Size(48, 48));
	Mat draw_img;
	img.copyTo(draw_img);
	cv::resize(draw_img, draw_img, cv::Size(), 10, 10);
	
	ZQ_CNN_Tensor4D_NHW_C_Align128bit input;
	input.ConvertFromBGR(img.data, img.cols, img.rows, img.step[0]);
	if (!net1.Forward(input) || !net2.Forward(input))
	{
		cout << "failed to forward\n";	
		return EXIT_FAILURE;
	}

	const ZQ_CNN_Tensor4D* landmark1 = net1.GetBlobByName("conv6-3");
	const ZQ_CNN_Tensor4D* landmark2 = net2.GetBlobByName("conv6-3");
	if (landmark1 == 0 || landmark2 == 0)
	{
		cout << "failed to get blob conv6-3\n";
		return EXIT_FAILURE;
	}
	const float* landmark1_data = landmark1->GetFirstPixelPtr();
	const float* landmark2_data = landmark2->GetFirstPixelPtr();
	for (int i = 0; i < 5; i++)
	{
		cv::circle(draw_img, cv::Point(480 * landmark1_data[i], 480 * landmark1_data[i + 5]), 2, cv::Scalar(0, 0, 120+30*i), 2);
		cv::circle(draw_img, cv::Point(480 * landmark2_data[i], 480 * landmark2_data[i + 5]), 2, cv::Scalar(0, 120+30*i, 0), 2);
	}

	namedWindow("landmark");
	
	imshow("landmark", draw_img);
	cv::imwrite("landmark.jpg", draw_img);
	waitKey(0);
	return EXIT_SUCCESS;
}