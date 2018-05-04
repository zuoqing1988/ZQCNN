#include "ZQ_CNN_Net.h"
#include <cblas.h>
#include <vector>
#include <iostream>
#include "opencv2\opencv.hpp"
using namespace ZQ;
using namespace std;
using namespace cv;
int main()
{
	for (int out_it = 0; out_it < 1; out_it++)
	{

		openblas_set_num_threads(1);
		Mat image0 = cv::imread("data\\00.jpg", 1);
		if (image0.empty())
		{
			cout << "empty image\n";
			return EXIT_FAILURE;
		}
		Mat image1 = cv::imread("data\\01.jpg", 1);
		if (image1.empty())
		{
			cout << "empty image\n";
			return EXIT_FAILURE;
		}

		ZQ_CNN_Tensor4D_NHW_C_Align128bit input0, input1;
		input0.ConvertFromBGR(image0.data, image0.cols, image0.rows, image0.step[0]);
		input1.ConvertFromBGR(image1.data, image1.cols, image1.rows, image1.step[0]);


		ZQ_CNN_Net net;
		if (!net.LoadFrom("model\\sphereface04bn256.zqparams", "model\\sphereface04bn256.nchwbin"))
		//if (!net.LoadFrom("model\\sphereface20.zqparams", "model\\sphereface20.nchwbin"))
		//if (!net.LoadFrom("model\\sphereface04.zqparams", "model\\sphereface04.nchwbin"))
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

		const ZQ_CNN_Tensor4D* ptr = net.GetBlobByName("fc5");
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

		ptr = net.GetBlobByName("fc5");
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
		std::cout << "feat0[0] = " << feat0[0] << "\n";
		std::cout << "feat1[0] = " << feat1[0] << "\n";
		std::cout << "Similarity score: " << score << "\n";
	}
	return EXIT_SUCCESS;
}
