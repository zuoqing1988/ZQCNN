#if defined(_WIN32)
#include "ZQ_FaceRecognizerArcFaceZQCNN.h"
#include "ZQ_FaceRecognizerSphereFaceZQCNN.h"
#include "ZQ_FaceDatabaseMaker.h"
#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_CompileConfig.h"
#if ZQ_CNN_USE_BLAS_GEMM
#include <openblas\cblas.h>
#pragma comment(lib,"libopenblas.lib")
#elif ZQ_CNN_USE_MKL_GEMM
#include <mkl\mkl.h>
#pragma comment(lib,"mklml.lib")
#else
#pragma comment(lib,"ZQ_GEMM.lib")
#endif

using namespace cv;
using namespace ZQ;

int main(int argc, const char** argv)
{
#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(1);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(1);
#endif
	if (argc < 6)
	{
		printf("Use: %s src_root out_file proto_file model_file out_blob_name [max_thread_num]\n", argv[0]);
		return EXIT_FAILURE;
	}
	int max_thread_num = 4;
	std::string src_root = argv[1];
	std::string out_file = argv[2];
	std::string proto_file = argv[3];
	std::string model_file = argv[4];
	std::string out_blob_name = argv[5];
	if (argc > 6)
		max_thread_num = atoi(argv[6]);

	max_thread_num = __max(1, __min(max_thread_num, omp_get_num_procs() - 1));
	std::vector<ZQ_FaceRecognizer*> recognizers(max_thread_num);
	
	if(1)
	{
		ZQ_CNN_Net net;
		if (!net.LoadFrom(proto_file, model_file, false))
		{
			printf("failed to load net (%s, %s)\n", proto_file.c_str(), model_file.c_str());
			return EXIT_FAILURE;
		}

		int c, h, w;
		net.GetInputDim(c, h, w);
		if (c == 3 && h == 112 && w == 112)
		{
			for (int i = 0; i < max_thread_num; i++)
			{
				recognizers[i] = new ZQ_FaceRecognizerArcFaceZQCNN();
				if (!recognizers[i]->Init("", proto_file, model_file, out_blob_name))
				{
					for (int j = 0; j < i; j++)
						delete recognizers[j];
				}
			}
		}
		else if(c == 3 && h == 112 && w == 96)
		{
			for (int i = 0; i < max_thread_num; i++)
			{
				recognizers[i] = new ZQ_FaceRecognizerSphereFaceZQCNN();
				if (!recognizers[i]->Init("", proto_file, model_file, out_blob_name))
				{
					for (int j = 0; j < i; j++)
						delete recognizers[j];
				}
			}
		}
		else
		{
			printf("face recognizer must be H*W*3, H*W = 112*112 or H*W = 112*96\n");
			return EXIT_FAILURE;
		}
	}
	
	double t1 = omp_get_wtime();
	bool ret = ZQ_FaceDatabaseMaker::DetectOutliersInDatabase(recognizers, src_root, max_thread_num, out_file);
	double t2 = omp_get_wtime();
	if (!ret)
	{
		printf("failed to run detect outliers\n");
	}
	else
	{
		printf("cost: %.3f secs\n", t2 - t1);
	}
	
	for (int j = 0; j < max_thread_num; j++)
		delete recognizers[j];
	return ret ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else
#include <stdio.h>
int main(int argc, const char** argv)
{
	printf("%s only support windows\n", argv[0]);
	return 0;
}
#endif
