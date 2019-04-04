#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_Net_NCHWC.h"
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
#if !defined(_WIN32)
#include <sched.h>
#endif
using namespace ZQ;
using namespace std;
using namespace cv;
int main(int argc, const char** argv)
{
#if !defined(_WIN32)
	if (argc != 1)
	{
		cpu_set_t mask;
		CPU_ZERO(&mask);
		CPU_SET(atoi(argv[1]), &mask);
		if (sched_setaffinity(0, sizeof(mask), &mask) < 0) {
			perror("sched_setaffinity");
		}
	}
#endif
	int num_threads = 1;
#if __ARM_NEON
	int iters = 100;
#else
	int iters = 1000;
#endif

#if ZQ_CNN_USE_BLAS_GEMM
	printf("set openblas thread_num = %d\n",num_threads);
	openblas_set_num_threads(num_threads);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(num_threads);
#endif

	std::string out_blob_name = "fc5";
	ZQ_CNN_Net net;
#if __ARM_NEON 
	ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC4> net_nchwc;
#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC8> net_nchwc;
#elif ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC4> net_nchwc;
#else
	ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC1> net_nchwc;
#endif
#endif

#if defined(_WIN32)
	//if (!net.LoadFrom("model/mobilefacenet-res4-8-16-4-dim512.zqparams", "model/mobilefacenet-res4-8-16-4-dim512-emore.nchwbin", false))
	//if (!net.LoadFrom("model/mobilefacenet-res1-3-5-2-dim128-112X96.zqparams", "model/mobilefacenet-res1-3-5-2-dim128-112X96.nchwbin", false))
	//if (!net.LoadFrom("model/mobilefacenet-GNAP.zqparams", "model/mobilefacenet-GNAP.nchwbin", true,1e-9,true))
	//if (!net.LoadFrom("model/mobilefacenet-v1.zqparams", "model/mobilefacenet-v1.nchwbin", true, 1e-9, true))
	if (!net.LoadFrom("model/test.zqparams", "model/test.nchwbin", true, 1e-9, true))
	//if (!net.LoadFrom("model/mobilefacenet-res8-16-32-8-dim512.zqparams", "model/mobilefacenet-res8-16-32-8-dim512.nchwbin", true,1e-16))
	//if (!net.LoadFrom("model/sphereface20.zqparams", "model/sphereface20.nchwbin", true, 1e-12,true))
	//if (!net.LoadFrom("model/mobilefacenet-v112X96.zqparams", "model/mobilefacenet-v112X96.nchwbin",false))
	//if (!net.LoadFrom("model/mobilefacenet-res2-6-10-2-dim128.zqparams", "model/mobilefacenet-res2-6-10-2-dim128-emore.nchwbin", true,1e-9, true))
	//if (!net.LoadFrom("model/mobilefacenet-res2-6-10-2-dim512-112X96.zqparams", "model/mobilefacenet-res2-6-10-2-dim512-112X96.nchwbin", false,1e-12))
	//if (!net.LoadFrom("model/model-r34-am.zqparams", "model/model-r34-am.nchwbin"))
	//if (!net.LoadFrom("model/model-r50-am.zqparams", "model/model-r50-am.nchwbin"))
	//if (!net.LoadFrom("model/model-r100-am.zqparams", "model/model-r100-am.nchwbin", false))
#else
	//if (!net.LoadFrom("../../model/mobilefacenet-res4-8-16-4-dim512.zqparams", "../../model/mobilefacenet-res4-8-16-4-dim512-emore.nchwbin", false))
	//if (!net.LoadFrom("../../model/mobilefacenet-res1-3-5-2-dim128-112X96.zqparams", "../../model/mobilefacenet-res1-3-5-2-dim128-112X96.nchwbin", false))
	//if (!net.LoadFrom("../../model/mobilefacenet-GNAP.zqparams", "../../model/mobilefacenet-GNAP.nchwbin", true, 1e-9, true))
	if (!net.LoadFrom("../../model/mobilefacenet-v1.zqparams", "../../model/mobilefacenet-v1.nchwbin", true, 1e-9, true))
	//if (!net.LoadFrom("../../model/mobilefacenet-res8-16-32-8-dim512.zqparams", "../../model/mobilefacenet-res8-16-32-8-dim512.nchwbin", true,1e-16))
	//if (!net.LoadFrom("../../model/sphereface04bn256.zqparams", "../../model/sphereface04bn256.nchwbin", true, 1e-12))
	//if (!net.LoadFrom("../../model/mobilefacenet-v112X96.zqparams", "../../model/mobilefacenet-v112X96.nchwbin",false))
	//if (!net.LoadFrom("../../model/mobilefacenet-res2-6-10-2-dim128.zqparams", "../../model/mobilefacenet-res2-6-10-2-dim128-emore.nchwbin", true, 1e-9, true))
	//if (!net.LoadFrom("../../model/mobilefacenet-res2-6-10-2-dim512-112X96.zqparams", "../../model/mobilefacenet-res2-6-10-2-dim512-112X96.nchwbin", false,1e-12))
	//if (!net.LoadFrom("../../model/model-r34-am.zqparams", "../../model/model-r34-am.nchwbin"))
	//if (!net.LoadFrom("../../model/model-r50-am.zqparams", "../../model/model-r50-am.nchwbin"))
	//if (!net.LoadFrom("../../model/model-r100-am.zqparams", "../../model/model-r100-am.nchwbin", false))
#endif
	
	{
		cout << "failed to load net\n";
		return EXIT_FAILURE;
	}


#if defined(_WIN32)
	//if (!net_nchwc.LoadFrom("model/mobilefacenet-res4-8-16-4-dim512.zqparams", "model/mobilefacenet-res4-8-16-4-dim512-emore.nchwbin", false))
	//if (!net_nchwc.LoadFrom("model/mobilefacenet-res1-3-5-2-dim128-112X96.zqparams", "model/mobilefacenet-res1-3-5-2-dim128-112X96.nchwbin", false))
	//if (!net_nchwc.LoadFrom("model/mobilefacenet-GNAP.zqparams", "model/mobilefacenet-GNAP.nchwbin", true,1e-9,true))
	if (!net_nchwc.LoadFrom("model/test.zqparams", "model/test.nchwbin", true, 1e-9, true))
	//if (!net_nchwc.LoadFrom("model/mobilefacenet-v1.zqparams", "model/mobilefacenet-v1.nchwbin", true, 1e-9, true))
	//if (!net_nchwc.LoadFrom("model/mobilefacenet-res8-16-32-8-dim512.zqparams", "model/mobilefacenet-res8-16-32-8-dim512.nchwbin", true,1e-16))
	//if (!net_nchwc.LoadFrom("model/sphereface20.zqparams", "model/sphereface20.nchwbin", true, 1e-12, true))
	//if (!net_nchwc.LoadFrom("model/mobilefacenet-v112X96.zqparams", "model/mobilefacenet-v112X96.nchwbin",false))
	//if (!net_nchwc.LoadFrom("model/mobilefacenet-res2-6-10-2-dim128.zqparams", "model/mobilefacenet-res2-6-10-2-dim128-emore.nchwbin", true,1e-9, true))
	//if (!net_nchwc.LoadFrom("model/mobilefacenet-res2-6-10-2-dim512-112X96.zqparams", "model/mobilefacenet-res2-6-10-2-dim512-112X96.nchwbin", false,1e-12))
	//if (!net_nchwc.LoadFrom("model/model-r34-am.zqparams", "model/model-r34-am.nchwbin"))
	//if (!net_nchwc.LoadFrom("model/model-r50-am.zqparams", "model/model-r50-am.nchwbin"))
	//if (!net_nchwc.LoadFrom("model/model-r100-am.zqparams", "model/model-r100-am.nchwbin", false))
#else
	//if (!net_nchwc.LoadFrom("../../model/mobilefacenet-res4-8-16-4-dim512.zqparams", "../../model/mobilefacenet-res4-8-16-4-dim512-emore.nchwbin", false))
	//if (!net_nchwc.LoadFrom("../../model/mobilefacenet-res1-3-5-2-dim128-112X96.zqparams", "../../model/mobilefacenet-res1-3-5-2-dim128-112X96.nchwbin", false))
	//if (!net_nchwc.LoadFrom("../../model/mobilefacenet-GNAP.zqparams", "../../model/mobilefacenet-GNAP.nchwbin", true, 1e-9, true))
	if (!net_nchwc.LoadFrom("../../model/mobilefacenet-v1.zqparams", "../../model/mobilefacenet-v1.nchwbin", true, 1e-9, true))
	//if (!net_nchwc.LoadFrom("../../model/mobilefacenet-res8-16-32-8-dim512.zqparams", "../../model/mobilefacenet-res8-16-32-8-dim512.nchwbin", true,1e-16))
	//if (!net_nchwc.LoadFrom("../../model/sphereface04bn256.zqparams", "../../model/sphereface04bn256.nchwbin", true, 1e-12))
	//if (!net_nchwc.LoadFrom("../../model/mobilefacenet-v112X96.zqparams", "../../model/mobilefacenet-v112X96.nchwbin",false))
	//if (!net_nchwc.LoadFrom("../../model/mobilefacenet-res2-6-10-2-dim128.zqparams", "../../model/mobilefacenet-res2-6-10-2-dim128-emore.nchwbin", true, 1e-9, true))
	//if (!net_nchwc.LoadFrom("../../model/mobilefacenet-res2-6-10-2-dim512-112X96.zqparams", "../../model/mobilefacenet-res2-6-10-2-dim512-112X96.nchwbin", false,1e-12))
	//if (!net_nchwc.LoadFrom("../../model/model-r34-am.zqparams", "../../model/model-r34-am.nchwbin"))
	//if (!net_nchwc.LoadFrom("../../model/model-r50-am.zqparams", "../../model/model-r50-am.nchwbin"))
	//if (!net_nchwc.LoadFrom("../../model/model-r100-am.zqparams", "../../model/model-r100-am.nchwbin", false))
#endif

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

	const ZQ_CNN_Tensor4D_NCHWC* ptr2 = net_nchwc.GetBlobByName(out_blob_name);
	if (ptr2 == 0)
	{
		cout << "The blob " << out_blob_name << " does not exist!\n";
		return EXIT_FAILURE;
	}

	int input_H, input_W, input_C;
	net.GetInputDim(input_C, input_H, input_W);

	Mat image0, image1;
	if (input_C == 3 && input_H == 112 && input_W == 112)
	{
#if defined(_WIN32)
		std::string name = "data/00_.jpg";
#else
		std::string name = "../../data/00_.jpg";
#endif
		image0 = cv::imread(name, 1);
		if (image0.empty())
		{
			cout << name << " does not exist!\n";
			return EXIT_FAILURE;
		}
#if defined(_WIN32)
		name = "data/01_.jpg";
#else
		name = "../../data/01_.jpg";
#endif
		image1 = cv::imread(name, 1);
		if (image1.empty())
		{
			cout << name << " does not exist!\n";
			return EXIT_FAILURE;
		}
	}
	else if(input_C == 3 && input_H == 112 && input_W == 96)
	{
#if defined(_WIN32)
		std::string name = "data/00.jpg";
#else
		std::string name = "../../data/00.jpg";
#endif
		image0 = cv::imread(name, 1);
		if (image0.empty())
		{
			cout << name << " does not exist!\n";
			return EXIT_FAILURE;
		}
#if defined(_WIN32)
		name = "data/01.jpg";
#else
		name = "../../data/01.jpg";
#endif
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
	//net.TurnOffUseBuffer();
	//net_nchwc.TurnOffUseBuffer();
	for (int out_it = 0; out_it < 10; out_it++)
	{
		ZQ_CNN_Tensor4D_NHW_C_Align128bit input0, input1;
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
		input0.ConvertFromBGR(image0.data, image0.cols, image0.rows, image0.step[0]);
		input1.ConvertFromBGR(image1.data, image1.cols, image1.rows, image1.step[0]);
		input2.ConvertFromBGR(image0.data, image0.cols, image0.rows, image0.step[0]);

		printf("num_MulAdd: %.3f M, (conv: %.3f M, dwconv: %.3f M)\n", 
			net.GetNumOfMulAdd() / (1024.0*1024.0),
			net.GetNumOfMulAddConv() / (1024.0*1024.0),
			net.GetNumOfMulAddDwConv() / (1024.0*1024.0));

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

		double t5 = omp_get_wtime();
		for (int it = 0; it < iters; it++)
		{
			if (!net_nchwc.Forward(input2))
			{
				cout << "failed to run\n";
				return EXIT_FAILURE;
			}
		}
		double t6 = omp_get_wtime();
		printf("[%d] times cost %.3f s, 1 iter cost %.3f ms\n", iters, t6 - t5, 1000 * (t6 - t5) / iters);

		printf("last time: inner = %.3f ms, conv = %.3f ms, dwonv = %.3f ms, bns = %.3f ms, prelu = %.3f ms, eltwise = %.3f\n",
			1000 * net.GetLastTimeOfLayerType("InnerProduct"),
			1000 * net.GetLastTimeOfLayerType("Convolution"),
			1000 * net.GetLastTimeOfLayerType("DepthwiseConvolution"),
			1000 * net.GetLastTimeOfLayerType("BatchNormScale"),
			1000 * net.GetLastTimeOfLayerType("PReLU"),
			1000 * net.GetLastTimeOfLayerType("Eltwise")
		);
		printf("last time: inner = %.3f ms, conv = %.3f ms, dwonv = %.3f ms, bns = %.3f ms, prelu = %.3f ms, eltwise = %.3f\n",
			1000 * net_nchwc.GetLastTimeOfLayerType("InnerProduct"),
			1000 * net_nchwc.GetLastTimeOfLayerType("Convolution"),
			1000 * net_nchwc.GetLastTimeOfLayerType("DepthwiseConvolution"),
			1000 * net_nchwc.GetLastTimeOfLayerType("BatchNormScale"),
			1000 * net_nchwc.GetLastTimeOfLayerType("PReLU"),
			1000 * net_nchwc.GetLastTimeOfLayerType("Eltwise")
		);
		ptr2 = net_nchwc.GetBlobByName(out_blob_name);
		std::vector<float> feat2(dim);
		memcpy(&feat2[0], ptr2->GetFirstPixelPtr(), sizeof(float)*dim);
		float score = 0;
		float score2 = 0;
		float len0 = 0, len1 = 0, len2 = 0;
		for (int i = 0; i < dim; i++)
		{
			score += feat0[i] * feat1[i];
			score2 += feat0[i] * feat2[i];
			len0 += feat0[i] * feat0[i];
			len1 += feat1[i] * feat1[i];
			len2 += feat2[i] * feat2[i];
		}
		len0 = sqrt(len0);
		len1 = sqrt(len1);
		len2 = sqrt(len2);
		score /= (len0*len1 + 1e-64);
		score2 /= (len0*len2 + 1e-64);
		for (int i = 0; i < dim; i++)
		{
			feat0[i] /= len0;
			feat1[i] /= len1;
			feat2[i] /= len2;
		}
		std::cout << "len0 = " << len0 << "\n";
		std::cout << "len1 = " << len1 << "\n";
		std::cout << "len2 = " << len2 << "\n";
		std::cout << "Similarity score(0,1): " << score << "\n";
		std::cout << "Similarity score(0,2): " << score2 << "\n";
	}
	return EXIT_SUCCESS;
}
