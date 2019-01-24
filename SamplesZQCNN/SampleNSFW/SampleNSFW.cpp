#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_NSFW.h"
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

	ZQ_CNN_NSFW nsfw;
	if (!nsfw.Init("model/nsfw.zqparams", "model/nsfw.nchwbin", "prob"))
	{
		printf("failed to init net\n");
		return EXIT_FAILURE;
	}

	Mat image = cv::imread("data/sex.jpg", 1);
	if (image.empty())
	{
		cout << "empty image\n";
		return EXIT_FAILURE;
	}
	cv::resize(image, image, cv::Size(224, 224));
	std::vector<float> output;
	nsfw.Detect(output, image.data, image.cols, image.rows, image.step[0], false);
	for (int i = 0; i < output.size(); i++)
	{
		printf("%.3f ", output[i]);
	}
	printf("\n");

	image = cv::imread("data/4.jpg", 1);
	if (image.empty())
	{
		cout << "empty image\n";
		return EXIT_FAILURE;
	}
	cv::resize(image, image, cv::Size(224, 224));
	
	nsfw.Detect(output, image.data, image.cols, image.rows, image.step[0], false);
	for (int i = 0; i < output.size(); i++)
	{
		printf("%.3f ", output[i]);
	}
	printf("\n");
	return EXIT_SUCCESS;
}

int main1()
{
	int num_threads = 1;

#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(num_threads);
#endif

	Mat image = cv::imread("data/sex.jpg", 1);
	if (image.empty())
	{
		cout << "empty image\n";
		return EXIT_FAILURE;
	}
	cv::resize(image, image, cv::Size(224, 224));

	ZQ_CNN_Tensor4D_NHW_C_Align128bit input;
	input.ChangeSize(1, image.rows, image.cols, 3, 0, 0);
	float* input_data = input.GetFirstPixelPtr();
	int widthStep = input.GetWidthStep();
	int pixStep = input.GetPixelStep();
	for (int h = 0; h < image.rows; h++)
	{
		uchar* ptr = image.data + h*image.step[0];
		float* dst = input_data + h*widthStep;
		for (int w = 0; w < image.cols; w++)
		{
			dst[w*pixStep + 0] = (float)ptr[w * 3 + 0] - 102.9801f;
			dst[w*pixStep + 1] = (float)ptr[w * 3 + 1] - 115.9465f;
			dst[w*pixStep + 2] = (float)ptr[w * 3 + 2] - 122.7717f;
		}
	}
	
	std::string out_blob_name = "prob";
	ZQ_CNN_Net net;

	if (!net.LoadFrom("model/nsfw.zqparams", "model/nsfw.nchwbin"))
	{
		cout << "failed to load net\n";
		return EXIT_FAILURE;
	}
	printf("num_MulAdd = %.3f M\n", net.GetNumOfMulAdd() / (1024.0*1024.0));
	//net.TurnOnShowDebugInfo();
	int iters = 100;
	double t1 = omp_get_wtime();
	for (int it = 0; it < iters; it++)
	{
		double t3 = omp_get_wtime();
		if (!net.Forward(input))
		{
			cout << "failed to run\n";
			return EXIT_FAILURE;
		}
		double t4 = omp_get_wtime();
		//printf("forward costs: %.3f ms\n", 1000 * (t4 - t3));
	}
	double t2 = omp_get_wtime();
	printf("[%d] times cost %.3f s, 1 iter cost %.3f ms\n", iters, t2 - t1, 1000 * (t2 - t1) / iters);

	const ZQ_CNN_Tensor4D* ptr = net.GetBlobByName(out_blob_name);
	int dim = ptr->GetC();
	std::vector<float> data(dim);
	memcpy(&data[0], ptr->GetFirstPixelPtr(), sizeof(float)*dim);
	printf("score = %.3f\n", data[0]);
	return EXIT_SUCCESS;
}
