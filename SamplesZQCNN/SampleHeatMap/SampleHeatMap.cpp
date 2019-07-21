#define _CRT_SECURE_NO_WARNINGS
#include "ZQ_CNN_Net.h"
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
	int show_H = 512, show_W = 512;
	ZQ_CNN_Net net1;

#if defined(_WIN32)
	if (!net1.LoadFrom("model/ZQCNN-model.zqparams", "model/ZQCNN-model.nchwbin", false, 0, false))
#else
	if (!net1.LoadFrom("../../model/det17-dw112-hm.zqparams", "./../model/det17-dw112-hm-5340.nchwbin", true, 1e-9, true))
#endif
	{
		cout << "failed to load model\n";
		return EXIT_FAILURE;
	}
	printf("num_MulAdd = %.1f M\n", net1.GetNumOfMulAdd() / (1024.0*1024.0));
	int net1_H, net1_W, net1_C;
	net1.GetInputDim(net1_C, net1_H, net1_W);
#if defined(_WIN32)
	Mat img = imread("data/pose.jpg", 1);
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
	//cv::cvtColor(img, img, CV_BGR2RGB);
	Mat img1, img2;
	cv::resize(img, img1, cv::Size(net1_W, net1_H));
	Mat draw_img;
	cv::resize(img, draw_img, cv::Size(show_W, show_H));

	ZQ_CNN_Tensor4D_NHW_C_Align128bit input1;

	std::vector<float> raw(net1_W*net1_H * 3,1);
	for (int w = 0; w < net1_W; w++)
	{
		for (int c = 0; c < 3; c++)
		{
			raw[c*net1_H*net1_W + 0 * net1_W + w] = 2;
		}
	}
	//input1.ConvertFromCompactNCHW(&raw[0],1,3,net1_H,net1_W);
	input1.ConvertFromBGR(img1.data, img1.cols, img1.rows, img1.step[0], 0, 1);
	for (int out_it = 0; out_it < 1; out_it++)
	{
		int nIters = 1;
		double t1 = omp_get_wtime();
		for (int i = 0; i < nIters; i++)
			net1.Forward(input1);
		double t2 = omp_get_wtime();
		printf("net1 %.3f s / %d = %.3f ms\n", t2 - t1, nIters, 1000 * (t2 - t1) / nIters);
		printf("last time of net1:\n conv = %.3f ms, dwonv = %.3f ms, bns = %.3f ms, prelu = %.3f ms\n",
			1000 * net1.GetLastTimeOfLayerType("Convolution"),
			1000 * net1.GetLastTimeOfLayerType("DepthwiseConvolution"),
			1000 * net1.GetLastTimeOfLayerType("BatchNormScale"),
			1000 * net1.GetLastTimeOfLayerType("PReLU")
		);
	}
	const ZQ_CNN_Tensor4D* heatmap = net1.GetBlobByName("CPM/stage_2_out");

	if (heatmap == 0)
	{
		cout << "failed to get blob heatmap\n";
		return EXIT_FAILURE;
	}
	const float* heatmap_data = heatmap->GetFirstPixelPtr();
	int hm_H = heatmap->GetH();
	int hm_W = heatmap->GetW();
	int hm_C = heatmap->GetC();
	int hm_widthStep = heatmap->GetWidthStep();
	int hm_pixStep = heatmap->GetPixelStep();
	cv::convertScaleAbs(draw_img, draw_img, 0.3);

	for (int c = 0; c < hm_C; c++)
	{
		cv::Mat temp_img;
		draw_img.copyTo(temp_img);
		cv::Mat hm_img = cv::Mat(hm_H, hm_W, CV_8UC1);
		for (int h = 0; h < hm_H; h++)
		{
			for (int w = 0; w < hm_W; w++)
			{
				hm_img.data[h*hm_img.step[0] + w] = __min(255, __max(0, heatmap_data[c+h*hm_widthStep + w*hm_pixStep] * 255));
			}
		}
		cv::resize(hm_img, hm_img, cv::Size(draw_img.cols, draw_img.rows), 0, 0, CV_INTER_NN);
		for (int h = 0; h < draw_img.rows; h++)
		{
			for (int w = 0; w < draw_img.cols; w++)
			{
				float ori = draw_img.data[h*draw_img.step[0] + w * 3 + 2] + 0.7*hm_img.data[h*hm_img.step[0] + w];
				temp_img.data[h*draw_img.step[0] + w * 3 + 2] = __min(255, __max(0, ori));
			}
		}
		char buf_name[100];
		sprintf(buf_name, "heatmap_%d", c);
		namedWindow(buf_name);

		imshow(buf_name, temp_img);
		sprintf(buf_name, "heatmap_%d.jpg", c);
		cv::imwrite(buf_name, temp_img);
	}
	
	waitKey(0);
	return EXIT_SUCCESS;
}