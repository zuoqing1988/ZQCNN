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
	Mat image = cv::imread("data/00_.jpg", 1);
	if (image.empty())
	{
		cout << "empty image\n";
		return EXIT_FAILURE;
	}
	
	ZQ_CNN_Tensor4D_NHW_C_Align128bit input;
	input.ConvertFromBGR(image.data, image.cols, image.rows, image.step[0]);
	
	std::string out_blob_name = "fc1";
	ZQ_CNN_Net net;

	if (!net.LoadFrom("model/GA112.zqparams", "model/GA112-100.nchwbin"))
	{
		cout << "failed to load net\n";
		return EXIT_FAILURE;
	}
	printf("num_MulAdd = %.3f M\n", net.GetNumOfMulAdd()/(1024.0*1024.0));
	//net.TurnOnShowDebugInfo();
	//net.TurnOffUseBuffer();
	//net.TurnOnUseBuffer();
	int iters = 10000;
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
	/*for (int w = 0; w < dim; w += 2)
	{
		printf("%12.3f %12.3f\n", data[w], data[w + 1]);
	}*/
	int gender = data[0] < data[1] ? 1 : 0;
	int age = 0;
	float confidence = 0.8;
	float range = log(confidence/(1-confidence));
	int age_min = 0;
	int age_max = 0;
	for (int w = 2; w < dim; w += 2)
	{
		age += (data[w] - data[w + 1]) > 0 ? 1 : 0;
		age_min += (data[w] - data[w + 1] - range) > 0 ? 1 : 0;
		age_max += (data[w] - data[w + 1] + range) > 0 ? 1 : 0;
	}
	printf("gender= %s, age = %d, (%d,%d)\n", gender ? "Male" : "Female", 
		age+10, age_min - age, age_max - age);

	return EXIT_SUCCESS;
}
