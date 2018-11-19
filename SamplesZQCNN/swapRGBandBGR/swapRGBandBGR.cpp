#include "ZQ_CNN_Net.h"
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

int main(int argc, const char** argv)
{
	if (argc < 4)
	{
		printf("%s param_file model_file out_model_file [layer1] [layer2] ...\n", argv[0]);
		return EXIT_FAILURE;
	}
	std::string param_file = argv[1];
	std::string model_file = argv[2];
	std::string out_model_file = argv[3];
	std::vector<std::string> layer_names;
	for (int i = 4; i < argc; i++)
		layer_names.push_back(std::string(argv[i]));

	ZQ_CNN_Net net;
	if (!net.LoadFrom(param_file, model_file, false))
	{
		printf("failed to load net: %s, %s\n", param_file.c_str(), model_file.c_str());
		return EXIT_FAILURE;
	}

	if (!net.SwapInputRGBandBGR(layer_names))
	{
		printf("failed to swap RGB and BGR for net: %s, %s\n", param_file.c_str(), model_file.c_str());
		return EXIT_FAILURE;
	}

	if (!net.SaveModel(out_model_file))
	{
		printf("failed to save model to %s\n", out_model_file.c_str());
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}