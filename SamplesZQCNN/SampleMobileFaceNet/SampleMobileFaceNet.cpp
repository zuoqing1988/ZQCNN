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
	printf("This is an example for loading MobileFaceNet model converted by mxnet2zqcnn.exe\n\n");
	printf("If you don't have the model, you can run mxnet2zqcnn.exe like this:\n"
		"mxnet2zqcnn.exe model-symbol.json model-0000.params test.zqparams test.nchwbin\n");
	printf("\nThen open test.zqparams in notepad, and add \"C=3 H=112 W=112\" for Input layer\n\n");
	printf("The automatically converted proto file has two many blobs, wasting too much time.\n"
	"You can remove some blobs and layers by hand!\n\n\n");

#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(1);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(1);
#endif

	Mat image0 = cv::imread("data/00_.jpg", 1);
	if (image0.empty())
	{
		cout << "empty image\n";
		return EXIT_FAILURE;
	}
	Mat image1 = cv::imread("data/01_.jpg", 1);
	if (image1.empty())
	{
		cout << "empty image\n";
		return EXIT_FAILURE;
	}

	ZQ_CNN_Tensor4D_NHW_C_Align128bit input0, input1;
	input0.ConvertFromBGR(image0.data, image0.cols, image0.rows, image0.step[0], 0, 1);
	input1.ConvertFromBGR(image1.data, image1.cols, image1.rows, image1.step[0], 0, 1);

	std::string out_blob_name = "fc1";
	ZQ_CNN_Net net;

	if (!net.LoadFrom("model/test.zqparams", "model/test.nchwbin"))
	{
		cout << "failed to load net\n";
		return EXIT_FAILURE;
	}

	int iters = 100;
	double t1 = omp_get_wtime();
	for (int it = 0; it < iters; it++)
	{
		double t3 = omp_get_wtime();
		if (!net.Forward(input0))
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
	std::vector<float> feat0(dim);
	memcpy(&feat0[0], ptr->GetFirstPixelPtr(), sizeof(float)*dim);


	double t3 = omp_get_wtime();
	for (int it = 0; it < iters; it++)
	{
		if (!net.Forward(input1))
		{
			cout << "failed to run\n";
			return EXIT_FAILURE;
		}
	}
	double t4 = omp_get_wtime();
	printf("[%d] times cost %.3f s, 1 iter cost %.3f ms\n", iters, t4 - t3, 1000 * (t4 - t3) / iters);

	ptr = net.GetBlobByName(out_blob_name);
	std::vector<float> feat1(dim);
	memcpy(&feat1[0], ptr->GetFirstPixelPtr(), sizeof(float)*dim);
	float score = 0;
	float len0 = 0, len1 = 0;
	for (int i = 0; i < dim; i++)
	{
		score += feat0[i] * feat1[i];
		len0 += feat0[i] * feat0[i];
		len1 += feat1[i] * feat1[i];
	}
	len0 = sqrt(len0);
	len1 = sqrt(len1);
	score /= (len0*len1 + 1e-64);
	for (int i = 0; i < dim; i++)
	{
		feat0[i] /= len0;
		feat1[i] /= len1;
	}
	std::cout << "feat0[0] = " << feat0[0] << "\n";
	std::cout << "feat1[0] = " << feat1[0] << "\n";
	std::cout << "Similarity score: " << score << "\n";

	return EXIT_SUCCESS;
}
