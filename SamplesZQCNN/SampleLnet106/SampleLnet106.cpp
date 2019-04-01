#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_Net_NCHWC.h"
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
#if __ARM_NEON
	ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC4> net2;
#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC8> net2;
#elif ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC4> net2;
#else
	ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC1> net2;
#endif
#endif
	
#if defined(_WIN32)
	if (!net1.LoadFrom("model/det5-dw96-v2s.zqparams", "model/det5-dw96-v2s.nchwbin",true,1e-9,true)
		//|| !net2.LoadFrom("model/det3.zqparams", "model/det3_bgr.nchwbin", true, 1e-9))
		|| !net2.LoadFrom("model/det5-dw96-v2s.zqparams", "model/det5-dw96-v2s.nchwbin", true, 1e-9, true))
#else
	if (!net1.LoadFrom("../../model/det5-dw96-v2s.zqparams", "../../model/det5-dw96-v2s.nchwbin", true, 1e-9, true)
		//|| !net2.LoadFrom("../../model/det3.zqparams", "../../model/det3_bgr.nchwbin", true, 1e-9))
		|| !net2.LoadFrom("../../model/det5-dw96-v2s.zqparams", "../../model/det5-dw96-v2s.nchwbin", true, 1e-9, true))
#endif
	{
		cout << "failed to load model\n";
		return EXIT_FAILURE;
	}
	printf("num_MulAdd = %.1f M\n", net1.GetNumOfMulAdd() / (1024.0*1024.0));
	printf("num_MulAdd = %.1f M\n", net2.GetNumOfMulAdd() / (1024.0*1024.0));
	int net1_H, net1_W, net1_C;
	int net2_H, net2_W, net2_C;
	net1.GetInputDim(net1_C, net1_H, net1_W);
	net2.GetInputDim(net2_C, net2_H, net2_W);
#if defined(_WIN32)
	Mat img = imread("data/onet.jpg", 1);
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
	Mat img1, img2;
	cv::resize(img, img1, cv::Size(net1_W, net1_H));
	cv::resize(img, img2, cv::Size(net2_W, net2_H));
	Mat draw_img;
	cv::resize(img, draw_img, cv::Size(show_W, show_H));

	ZQ_CNN_Tensor4D_NHW_C_Align128bit input1;
#if __ARM_NEON 
	ZQ_CNN_Tensor4D_NCHWC4 input2;
#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	ZQ_CNN_Tensor4D_NCHWC8 input2;
#elif ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	ZQ_CNN_Tensor4D_NCHWC4 input2;
#else
	ZQ_CNN_Tensor4D_NCHWC1 input2;
#endif
#endif

	input1.ConvertFromBGR(img1.data, img1.cols, img1.rows, img1.step[0]);
	input2.ConvertFromBGR(img2.data, img2.cols, img2.rows, img2.step[0]);
	for (int out_it = 0; out_it < 3; out_it++)
	{
		int nIters = 1000;
		double t1 = omp_get_wtime();
		for (int i = 0; i < nIters; i++)
			net1.Forward(input1);
		double t2 = omp_get_wtime();
		for (int i = 0; i < nIters; i++)
			net2.Forward(input2);
		double t3 = omp_get_wtime();
		printf("net1 %.3f s / %d = %.3f ms\n", t2 - t1, nIters, 1000 * (t2 - t1) / nIters);
		printf("net2 %.3f s / %d = %.3f ms\n", t3 - t2, nIters, 1000 * (t3 - t2) / nIters);
		printf("last time of net1:\n conv = %.3f ms, dwonv = %.3f ms, bns = %.3f ms, prelu = %.3f ms\n",
			1000 * net1.GetLastTimeOfLayerType("Convolution"),
			1000 * net1.GetLastTimeOfLayerType("DepthwiseConvolution"),
			1000 * net1.GetLastTimeOfLayerType("BatchNormScale"),
			1000 * net1.GetLastTimeOfLayerType("PReLU")
		);
		printf("last time of net2:\n conv = %.3f ms, dwonv = %.3f ms, bns = %.3f ms, prelu = %.3f ms\n",
			1000 * net2.GetLastTimeOfLayerType("Convolution"),
			1000 * net2.GetLastTimeOfLayerType("DepthwiseConvolution"),
			1000 * net2.GetLastTimeOfLayerType("BatchNormScale"),
			1000 * net2.GetLastTimeOfLayerType("PReLU")
		);
	}
	const ZQ_CNN_Tensor4D* landmark1 = net1.GetBlobByName("conv6-3");
#if __ARM_NEON 
	const ZQ_CNN_Tensor4D_NCHWC4* landmark2 = net2.GetBlobByName("conv6-3");
#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	const ZQ_CNN_Tensor4D_NCHWC8* landmark2 = net2.GetBlobByName("conv6-3");
#elif ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	const ZQ_CNN_Tensor4D_NCHWC4* landmark2 = net2.GetBlobByName("conv6-3");
#else
	const ZQ_CNN_Tensor4D_NCHWC1* landmark2 = net2.GetBlobByName("conv6-3");
#endif
#endif
	
	if (landmark1 == 0 || landmark2 == 0)
	{
		cout << "failed to get blob conv6-3\n";
		return EXIT_FAILURE;
	}
	const float* landmark2_data = landmark1->GetFirstPixelPtr();
	const float* landmark1_data = landmark2->GetFirstPixelPtr();
	for (int i = 0; i < 106; i++)
	{
		char buf[10];
#if defined(_WIN32)
		sprintf_s(buf,10, "%d", i);
#else
		sprintf(buf, "%d", i);
#endif
		cv::Point pt = cv::Point(show_W * landmark1_data[i * 2], show_H * landmark1_data[i * 2 + 1]);
#if defined(_WIN32)
		ZQ_PutTextCN::PutTextCN(draw_img, buf, pt, cv::Scalar(100, 0, 0), 12);
#endif
		cv::circle(draw_img, pt, 2, cv::Scalar(0, 0, 250), 2);
	}
	/*for (int i = 0; i < 5; i++)
	{
		cv::circle(draw_img, cv::Point(show_W * landmark2_data[i], show_H * landmark2_data[i + 5]), 2, cv::Scalar(0, 120 + 30 * i, 0), 2);
	}*/

	namedWindow("landmark");

	imshow("landmark", draw_img);
	cv::imwrite("landmark.jpg", draw_img);
	waitKey(0);
	return EXIT_SUCCESS;
}