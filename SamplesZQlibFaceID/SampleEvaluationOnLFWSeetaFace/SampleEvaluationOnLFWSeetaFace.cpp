#if defined(_WIN32)
#include "ZQ_FaceIDPrecisionEvaluation.h"
#include "ZQ_FaceRecognizerSeetaFace.h"

using namespace std;
using namespace ZQ;

bool EvaluationSeetaFaceMiniCaffeOnLFW(const std::string& model_file,
	const std::string& list_file, const std::string& folder, int max_thread_num = 4, bool use_flip = false)
{
	int real_num_threads = __max(1, __min(max_thread_num, omp_get_num_procs() - 1));

	std::vector<ZQ_FaceRecognizer*> recognizers(real_num_threads);

	for (int i = 0; i < real_num_threads; i++)
	{
		recognizers[i] = new ZQ_FaceRecognizerSeetaFace();
		if (!recognizers[i]->Init(model_file))
		{
			printf("failed to load model_file: %s\n", model_file.c_str());
			return false;
		}
	}
	printf("load sphereface done!\n");

	return ZQ_FaceIDPrecisionEvaluation::EvaluationOnLFW(recognizers, list_file, folder, use_flip);
}


int main(int argc, const char** argv)
{
	int max_thread_num = 4;
	bool use_flip = true;
	if (argc < 4)
	{
		printf("%s model_file list_file folder [max_thread_num] [use_flip]\n", argv[0]);
		return EXIT_FAILURE;
	}
	if (argc > 4)
		max_thread_num = atoi(argv[4]);
	if (argc > 5)
		use_flip = atoi(argv[5]);

	double t1 = omp_get_wtime();
	if (!EvaluationSeetaFaceMiniCaffeOnLFW(string(argv[1]), string(argv[2]), string(argv[3]), max_thread_num, use_flip))
		return EXIT_FAILURE;
	double t2 = omp_get_wtime();
	printf("total cost: %.3f secs\n", t2 - t1);
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