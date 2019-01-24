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
#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(1);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(1);
#endif
	for (int out_it = 0; out_it < 1; out_it++)
	{
		Mat image0 = cv::imread("data/00011.jpg", 0);
		if (image0.empty())
		{
			cout << "empty image\n";
			return EXIT_FAILURE;
		}

		cv::resize(image0, image0, cv::Size(42, 42));
		ZQ_CNN_Tensor4D_NHW_C_Align128bit input0, input1;
		input0.ConvertFromGray(image0.data, image0.cols, image0.rows, image0.step[0], 0, 1);


		ZQ_CNN_Net net;
		if (!net.LoadFrom("model/FacialNet.zqparam", "model/FacialNet.nchwbin"))
		{
			cout << "failed to load net\n";
			return EXIT_FAILURE;
		}

		int iters = 1;
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

		const ZQ_CNN_Tensor4D* ptr = net.GetBlobByName("prob");
		int dim = ptr->GetC();
		std::vector<float> feat0(dim);
		memcpy(&feat0[0], ptr->GetFirstPixelPtr(), sizeof(float)*dim);
		for (int i = 0; i < dim; i++)
			printf("%.3f ", feat0[i]);
		printf("\n");

	}
	return EXIT_SUCCESS;
}