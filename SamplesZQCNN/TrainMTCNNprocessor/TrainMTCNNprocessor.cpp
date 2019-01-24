#include "TrainMTCNNprocessor.h"
#include "ZQ_CNN_CompileConfig.h"
#if ZQ_CNN_USE_BLAS_GEMM
#include <openblas/cblas.h>
#pragma comment(lib,"libopenblas.lib")
#elif ZQ_CNN_USE_MKL_GEMM
#include "mkl/mkl.h"
#pragma comment(lib,"mklml.lib")
#else
#pragma comment(lib,"ZQ_GEMM.lib")
#endif
using namespace ZQ;
using namespace std;
using namespace cv;


int generate_wider_prob(int argc, const char** argv);

int generate_data(int argc, const char** argv);

int generate_landmark(int argc, const char** argv);

int main(int argc, const char** argv)
{
	int num_threads = 1;
#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(num_threads);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(num_threads);
#endif

	if (argc < 2)
	{
		printf("%s check_wider_prob args\n", argv[0]);
		printf("%s generate_data args\n", argv[0]);
		printf("%s generate_landmark args\n", argv[0]);
		return EXIT_FAILURE;
	}

	if (strcmp(argv[1], "generate_wider_prob") == 0)
	{
		return generate_wider_prob(argc, argv);
	}
	else if (strcmp(argv[1], "generate_data") == 0)
	{
		return generate_data(argc, argv);
	}
	else if (strcmp(argv[1], "generate_landmark") == 0)
		return generate_landmark(argc, argv);
	
	printf("unknown method: %s\n", argv[1]);

	return EXIT_FAILURE;
}
int generate_wider_prob(int argc, const char** argv)
{
	if (argc < 4)
	{
		printf("%s %s anno_file prob_file [onet.zqparams] [onet.nchwbin]\n", argv[0], argv[1]);
		return EXIT_FAILURE;
	}
	const char* anno_file = argv[2];
	const char* prob_file = argv[3];
	const char* param_file = "model/det3.zqparams";
	const char* model_file = "model/det3_bgr.nchwbin";
	const char* out_blob_name = "prob1";
	if (argc >= 5)
		param_file = argv[4];
	if (argc >= 6)
		model_file = argv[5];
	if (argc >= 7)
		out_blob_name = argv[6];
	
	bool ret = TrainMTCNNprocessor::generateWiderProb(anno_file, prob_file, param_file, model_file, out_blob_name);
	return ret ? EXIT_SUCCESS : EXIT_FAILURE;
}


int generate_data(int argc, const char** argv)
{
	if (argc < 6)
	{
		printf("%s %s root anno_file prob_file size [base_num] [thread_num] [prob_thresh]\n", argv[0], argv[1]);
		return EXIT_FAILURE;
	}
	const char* root = argv[2];
	const char* anno_file = argv[3];
	const char* prob_file = argv[4];
	int size = atoi(argv[5]);
	int base_num = 1;
	int thread_num = 1;
	float prob_thresh = 0.3;
	if (argc >= 7)
		base_num = __max(1, atoi(argv[6]));
	if (argc >= 8)
		thread_num = __max(1, atoi(argv[7]));
	if (argc >= 9)
		prob_thresh = atof(argv[8]);
	bool ret = TrainMTCNNprocessor::generate_data(size, root, anno_file, prob_file, base_num, thread_num, prob_thresh);
	return ret ? EXIT_SUCCESS : EXIT_FAILURE;
}

int generate_landmark(int argc, const char** argv)
{
	if (argc < 7)
	{
		printf("%s %s root celeba_img_fold bbox_file landmark_file size [base_num] [thread_num]\n", argv[0], argv[1]);
		return EXIT_FAILURE;
	}
	const char* root = argv[2];
	const char* celeba_img_fold = argv[3];
	const char* bbox_file = argv[4];
	const char* landmark_file = argv[5];
	int size = atoi(argv[6]);
	int base_num = 1;
	int thread_num = 1;
	if (argc >= 8)
		base_num = __max(1, atoi(argv[7]));
	if (argc >= 9)
		thread_num = __max(1, atoi(argv[8]));
	bool ret = TrainMTCNNprocessor::generate_landmark(size, root, celeba_img_fold, bbox_file, landmark_file, base_num, thread_num);
	return ret ? EXIT_SUCCESS : EXIT_FAILURE;
}