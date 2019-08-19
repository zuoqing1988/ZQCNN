#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_Landmark240.h"
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
	int show_H = 640, show_W = 640;
	ZQ_CNN_Net net1;
	ZQ_CNN_Landmark240 net2;
	net2.TurnOnShowDebugInfo();

#if defined(_WIN32)
	if (!net1.LoadFrom("model/det5-dw112.zqparams", "model/det5-dw112.nchwbin", true, 1e-9, true))
#else
	if (!net1.LoadFrom("../../model/det5-dw112.zqparams", "../../model/det5-dw112.nchwbin", true, 1e-9, true))
#endif
	{
		cout << "failed to load model\n";
		return EXIT_FAILURE;
	}
	printf("num_MulAdd = %.1f M\n", net1.GetNumOfMulAdd() / (1024.0*1024.0));

#if defined(_WIN32)
	if (!net2.Init(
		"model/det6-dw64-left.zqparams", "model/det6-dw64-left.nchwbin", 
		"model/det6-dw64-right.zqparams", "model/det6-dw64-right.nchwbin",
		"model/det6-dw48-mouth.zqparams", "model/det6-dw48-mouth.nchwbin"
		))
#else
	if (!net2.Init(
		"../../model/det6-dw64-left.zqparams", "../../model/det6-dw64-left.nchwbin",
		"../../model/det6-dw64-right.zqparams", "../../model/det6-dw64-right.nchwbin",
		"../../model/det6-dw48-mouth.zqparams", "../../model/det6-dw48-mouth.nchwbin"
	))
#endif
	{
		cout << "failed to load model\n";
		return EXIT_FAILURE;
	}
	int net1_H, net1_W, net1_C;
	net1.GetInputDim(net1_C, net1_H, net1_W);
	
	
#if defined(_WIN32)
	Mat img = imread("data/test21.jpg", 1);
#else
	Mat img = imread("../../data/onet.jpg", 1);
#endif
	if (img.empty())
	{
		cout << "failed to load image\n";
		return EXIT_FAILURE;
	}
	if (img.channels() == 1)
		cv::cvtColor(img, img, CV_GRAY2BGR);
	Mat img1;
	cv::resize(img, img1, cv::Size(net1_W, net1_H));
	Mat draw_img;
	cv::resize(img, draw_img, cv::Size(show_W, show_H));

	ZQ_CNN_Tensor4D_NHW_C_Align128bit input;
	ZQ_CNN_Tensor4D_NHW_C_Align128bit Lnet106_image;
	ZQ_CNN_BBox106 box106;
	ZQ_CNN_BBox240 box240;

	input.ConvertFromBGR(img.data, img.cols, img.rows, img.step[0]);
	for (int out_it = 0; out_it < 3; out_it++)
	{
		input.ResizeBilinear(Lnet106_image, net1_W, net1_H, -1, -1);
		int nIters = 100;
		double t1 = omp_get_wtime();
		for (int i = 0; i < nIters; i++)
			net1.Forward(Lnet106_image);
		double t2 = omp_get_wtime();
		printf("net1 %.3f s / %d = %.3f ms\n", t2 - t1, nIters, 1000 * (t2 - t1) / nIters);
		printf("last time of net1:\n conv = %.3f ms, dwonv = %.3f ms, bns = %.3f ms, prelu = %.3f ms\n",
			1000 * net1.GetLastTimeOfLayerType("Convolution"),
			1000 * net1.GetLastTimeOfLayerType("DepthwiseConvolution"),
			1000 * net1.GetLastTimeOfLayerType("BatchNormScale"),
			1000 * net1.GetLastTimeOfLayerType("PReLU")
		);

		const ZQ_CNN_Tensor4D* landmark1 = net1.GetBlobByName("conv6-3");
		const float* landmark1_data = landmark1->GetFirstPixelPtr();
		for (int i = 0; i < 106; i++)
		{
			box106.ppoint[i * 2 + 0] = img.cols * landmark1_data[i * 2 + 0];
			box106.ppoint[i * 2 + 1] = img.rows * landmark1_data[i * 2 + 1];
		}

		double t3 = omp_get_wtime();
		for (int i = 0; i < nIters; i++)
			net2.Find(input, box106, box240);
		double t4 = omp_get_wtime();
		printf("net2 %.3f s / %d = %.3f ms\n", t4 - t3, nIters, 1000 * (t4 - t3) / nIters);
	}

	int idx = 0;
	for (int i = 0; i < 106; i++,idx++)
	{
		char buf[10];
#if defined(_WIN32)
		sprintf_s(buf, 10, "%d", idx);
#else
		sprintf(buf, "%d", i);
#endif
		cv::Point pt;
		pt = cv::Point(show_W * box240.box.ppoint[i * 2 + 0] / img.cols, show_H * box240.box.ppoint[i * 2 + 1] / img.rows);
		
#if defined(_WIN32)
		//ZQ_PutTextCN::PutTextCN(draw_img, buf, pt, cv::Scalar(100, 0, 0), 12);
#endif
		cv::circle(draw_img, pt, 2, cv::Scalar(0, 0, 250), 2);
	}

	for (int i = 0; i < 35; i++, idx++)
	{
		char buf[10];
#if defined(_WIN32)
		sprintf_s(buf, 10, "%d", idx);
#else
		sprintf(buf, "%d", i);
#endif
		cv::Point pt;
		pt = cv::Point(show_W * box240.left_brow_eye_ppoint[i * 2 + 0] / img.cols, show_H * box240.left_brow_eye_ppoint[i * 2 + 1] / img.rows);

#if defined(_WIN32)
		//ZQ_PutTextCN::PutTextCN(draw_img, buf, pt, cv::Scalar(100, 0, 0), 12);
#endif
		cv::circle(draw_img, pt, 2, cv::Scalar(0, 250, 0), 2);
	}

	for (int i = 0; i < 35; i++, idx++)
	{
		char buf[10];
#if defined(_WIN32)
		sprintf_s(buf, 10, "%d", idx);
#else
		sprintf(buf, "%d", i);
#endif
		cv::Point pt;
		pt = cv::Point(show_W * box240.right_brow_eye_ppoint[i * 2 + 0] / img.cols, show_H * box240.right_brow_eye_ppoint[i * 2 + 1] / img.rows);

#if defined(_WIN32)
		//ZQ_PutTextCN::PutTextCN(draw_img, buf, pt, cv::Scalar(100, 0, 0), 12);
#endif
		cv::circle(draw_img, pt, 2, cv::Scalar(0, 250, 0), 2);
	}

	for (int i = 0; i < 64; i++, idx++)
	{
		char buf[10];
#if defined(_WIN32)
		sprintf_s(buf, 10, "%d", idx);
#else
		sprintf(buf, "%d", i);
#endif
		cv::Point pt;
		pt = cv::Point(show_W * box240.mouth_ppoint[i * 2 + 0] / img.cols, show_H * box240.mouth_ppoint[i * 2 + 1] / img.rows);

#if defined(_WIN32)
		//ZQ_PutTextCN::PutTextCN(draw_img, buf, pt, cv::Scalar(100, 0, 0), 12);
#endif
		cv::circle(draw_img, pt, 2, cv::Scalar(0, 250, 0), 2);
	}
	
	namedWindow("landmark240");

	imshow("landmark240", draw_img);
	cv::imwrite("landmark240.jpg", draw_img);
	waitKey(0);
	return EXIT_SUCCESS;
}