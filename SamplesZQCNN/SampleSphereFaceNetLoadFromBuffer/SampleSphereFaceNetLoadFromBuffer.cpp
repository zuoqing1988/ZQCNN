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

bool load_to_buffer(const std::string& param_file, const std::string& model_file,
	std::vector<char>& param_buffer, std::vector<char>& model_buffer)
{
	param_buffer.clear();
	model_buffer.clear();

	FILE* in = 0;
#if defined(_WIN32)
	if (0 != fopen_s(&in, param_file.c_str(), "rb"))
#else
	if(0 == (in = fopen(param_file.c_str(), "rb")))
#endif
	{
		cout << "failed to open " << param_file << "\n";
		return false;
	}
#if defined(_WIN32)
	_fseeki64(in, 0, SEEK_END);
	__int64 param_buffer_len = _ftelli64(in);
	param_buffer.resize(param_buffer_len);
	_fseeki64(in, 0, SEEK_SET);
#else
	fseek(in, 0, SEEK_END);
	__int64 param_buffer_len = ftell(in);
	param_buffer.resize(param_buffer_len);
	fseek(in, 0, SEEK_SET);
#endif

	if (param_buffer_len > 0)
	{
		fread_s(&param_buffer[0], param_buffer_len, 1, param_buffer_len, in);
	}
	else
	{
		cout << "empty file " << param_file << "\n";
		return false;
	}
	fclose(in);
#if defined(_WIN32)
	if (0 != fopen_s(&in, model_file.c_str(), "rb"))
	{
		cout << "failed to open " << model_file << "\n";
		return false;
	}
	_fseeki64(in, 0, SEEK_END);
	__int64 model_buffer_len = _ftelli64(in);
	model_buffer.resize(model_buffer_len);
	_fseeki64(in, 0, SEEK_SET);
	if (model_buffer_len > 0)
	{
		fread_s(&model_buffer[0], model_buffer_len, 1, model_buffer_len, in);
	}
	else
	{
		cout << "empty file " << model_file << "\n";
		return false;
	}
#else
	if (0 == (in = fopen(model_file.c_str(), "rb")))
	{
		cout << "failed to open " << model_file << "\n";
		return false;
	}
	fseek(in, 0, SEEK_END);
	__int64 model_buffer_len = ftell(in);
	model_buffer.resize(model_buffer_len);
	fseek(in, 0, SEEK_SET);
	if (model_buffer_len > 0)
	{
		fread(&model_buffer[0], 1, model_buffer_len, in);
	}
	else
	{
		cout << "empty file " << model_file << "\n";
		return false;
	}
#endif
	fclose(in);

	return true;
}

int main()
{
	int num_threads = 1;

#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(num_threads);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(num_threads);
#endif

	std::string out_blob_name = "fc5";
	std::string param_file = "model/mobilefacenet-res8-16-32-8-dim512.zqparams";
	std::string model_file = "model/mobilefacenet-res8-16-32-8-dim512.nchwbin";
	
	std::vector<char> param_buffer, model_buffer;
	if (!load_to_buffer(param_file, model_file, param_buffer, model_buffer))
	{
		cout << "failed to load to buffer\n";
		return EXIT_FAILURE;
	}

	const char* param_ptr = &param_buffer[0];
	const char* model_ptr = &model_buffer[0];
	__int64 param_buffer_len = param_buffer.size();
	__int64 model_buffer_len = model_buffer.size();
	
	ZQ_CNN_Net net;
	if (!net.LoadFromBuffer(param_ptr,param_buffer_len, model_ptr, model_buffer_len, false))
	{
		cout << "failed to load net\n";
		return EXIT_FAILURE;
	}

	const ZQ_CNN_Tensor4D* ptr = net.GetBlobByName(out_blob_name);
	if (ptr == 0)
	{
		cout << "The blob " << out_blob_name << " does not exist!\n";
		return EXIT_FAILURE;
	}

	int input_H, input_W, input_C;
	net.GetInputDim(input_C, input_H, input_W);

	Mat image0, image1;
	if (input_C == 3 && input_H == 112 && input_W == 112)
	{
		std::string name = "data/00_.jpg";
		image0 = cv::imread(name, 1);
		if (image0.empty())
		{
			cout << name << " does not exist!\n";
			return EXIT_FAILURE;
		}
		name = "data/01_.jpg";
		image1 = cv::imread(name, 1);
		if (image1.empty())
		{
			cout << name << " does not exist!\n";
			return EXIT_FAILURE;
		}
	}
	else if (input_C == 3 && input_H == 112 && input_W == 96)
	{
		std::string name = "data/00.jpg";
		image0 = cv::imread(name, 1);
		if (image0.empty())
		{
			cout << name << " does not exist!\n";
			return EXIT_FAILURE;
		}
		name = "data/01.jpg";
		image1 = cv::imread(name, 1);
		if (image1.empty())
		{
			cout << name << " does not exist!\n";
			return EXIT_FAILURE;
		}
	}
	else
	{
		cout << "unsupported resolution: WxHxC = " << input_W << "x" << input_H << "x" << input_C << "\n";
		return EXIT_FAILURE;
	}

	//net.TurnOnShowDebugInfo();
	for (int out_it = 0; out_it < 10; out_it++)
	{
		ZQ_CNN_Tensor4D_NHW_C_Align128bit input0, input1;
		input0.ConvertFromBGR(image0.data, image0.cols, image0.rows, image0.step[0]);
		input1.ConvertFromBGR(image1.data, image1.cols, image1.rows, image1.step[0]);

		printf("num_MulAdd: %.3f M\n", net.GetNumOfMulAdd() / (1024.0*1024.0));
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

		ptr = net.GetBlobByName(out_blob_name);
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
		//std::cout << "feat0[0] = " << feat0[0] << "\n";
		//std::cout << "feat1[0] = " << feat1[0] << "\n";
		std::cout << "Similarity score: " << score << "\n";
	}
	return EXIT_SUCCESS;
}
