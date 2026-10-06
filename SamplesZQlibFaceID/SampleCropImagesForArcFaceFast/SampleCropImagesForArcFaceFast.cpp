#if defined(_WIN32)
#include "ZQ_FaceDatabaseMaker.h"
#include "ZQ_FaceRecognizerArcFaceZQCNN.h"
#include "ZQ_FaceDetectorLibFaceDetect.h"
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
using namespace std;
using namespace ZQ;

bool CropImagesForDatabase(const std::string& src_fold, const std::string& dst_fold, int max_thread_num, bool strict_check, const std::string& model_name)
{
	int real_num_threads = __max(1, __min(max_thread_num, omp_get_num_procs() - 1));

	std::vector<ZQ_FaceDetectorLibFaceDetect> detectors(real_num_threads);
	std::vector<ZQ_FaceRecognizerArcFaceZQCNN> recognizers(real_num_threads);

	for (int i = 0; i < real_num_threads; i++)
	{
		detectors[i].Init();
	}

	std::vector<ZQ_FaceDetector*> ptr_detectors(real_num_threads);
	// 审计修复 2026-10-07（附录 JC）：原来**只**初始化了检测器，
	// recognizer 被裸着交给 ZQ_FaceDatabaseMaker —— 而
	// `ZQ_FaceRecognizer::Init` 是**纯虚**，它就是「把网络加载进来」这一步，
	// MakeDatabase 又只查指针非空、不查是否已初始化。
	// 实测症状：sample 返回 **RC=0**、0 张图、耗时 0.000000s，一个错都不报。
	// 同族的 SampleCropImagesForSeetaFace 一直是对的（它接收模型并逐个 Init），
	// 形态上像是「Init 多了模型参数」那次 API 变更时只有它跟着改了。
	for (int i = 0; i < real_num_threads; i++)
	{
		// SphereFace/ArcFace 的 Init 按**名字**映射到 model/ 下的预置路径：
		// 04bn256 / 06bn512 / mobile-10bn512。
		if (!recognizers[i].Init(model_name))
		{
			printf("failed to load recognizer model: %s\n", model_name.c_str());
			return false;
		}
	}
	printf("load recognizer done!\n");
	std::vector<ZQ_FaceRecognizer*> ptr_recognizers(real_num_threads);
	for (int i = 0; i < real_num_threads; i++)
	{
		ptr_detectors[i] = &detectors[i];
		ptr_recognizers[i] = &recognizers[i];
	}
	return ZQ_FaceDatabaseMaker::CropImagesForDatabase(ptr_detectors, ptr_recognizers, src_fold, dst_fold, real_num_threads, strict_check, "err_log.txt", true);
}

int main(int argc, const char** argv)
{

	if (argc < 3)
	{
		std::cout << "Use: " << std::string(argv[0]) << " src_root dst_root [max_thread_num] [strict_check(0/1)] [model_name]\n";
		return EXIT_FAILURE;
	}

	int max_thread_num = 32;
	bool strict_check = false;
	if (argc > 3)
		max_thread_num = atoi(argv[3]);
	if (argc > 4)
		strict_check = atoi(argv[4]);
	std::string model_name = "04bn256";
	if (argc > 5)
		model_name = argv[5];
	if (!CropImagesForDatabase(argv[1], argv[2], max_thread_num, strict_check, model_name))
	{
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
#else
#include <stdio.h>
int main(int argc, const char** argv)
{
	printf("%s only support windows\n", argv[0]);
	return 0;
}
#endif