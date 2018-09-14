#include "ZQ_FaceDatabaseMaker.h"
#include "ZQ_FaceRecognizerArcFaceZQCNN.h"
#include "ZQ_FaceRecognizerSphereFaceZQCNN.h"
#include <stdio.h>
#include "ZQ_CNN_ComplieConfig.h"
#if ZQ_CNN_USE_BLAS_GEMM
#include <cblas.h>
#pragma comment(lib,"libopenblas.lib")
#endif
using namespace ZQ;

int make_database_compact(int argc, char** argv);
int compute_similarity_all_pairs(int argc, char**argv);
int load_database_compact(ZQ_FaceDatabaseCompact& database, const std::string& feats_file, const std::string& names_file);

int main(int argc, char** argv)
{
	cv::setNumThreads(1);
#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(1);
#endif

	if (argc < 2)
	{
		printf("%s make_compact [args]\n", argv[0]);
		printf("%s compute_similarity [args]\n", argv[0]);
		return EXIT_FAILURE;
	}
	if (_strcmpi(argv[1], "make_compact") == 0)
	{
		return make_database_compact(argc, argv);
	}
	else if (_strcmpi(argv[1], "compute_similarity") == 0)
	{
		return compute_similarity_all_pairs(argc, argv);
	}
	else
	{
		printf("%s make_compact [args]\n", argv[0]);
		printf("%s compute_similarity [args]\n", argv[0]);
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

int make_database_compact(int argc, char** argv)
{
	ZQ_FaceDatabaseMaker::MakeDatabaseType type = ZQ_FaceDatabaseMaker::UPDATE_WHO_NOT_HAVE_FEATS;
	int max_thread_num = 1;
	bool show = false;
	if (argc < 8)
	{
		printf("%s %s database_root feats_file names_file prototxt_file caffemodel_file out_blob_name [type] [max_thread_num] [show]\n", argv[0], argv[1]);
		return EXIT_FAILURE;
	}

	std::string database_root = argv[2];
	std::string feats_file = argv[3];
	std::string names_file = argv[4];
	std::string prototxt_file = argv[5];
	std::string caffemodel_file = argv[6];
	std::string out_blob_name = argv[7];
	if (argc >= 9)
		type = (ZQ_FaceDatabaseMaker::MakeDatabaseType)atoi(argv[8]);
	if (argc >= 10)
		max_thread_num = atoi(argv[9]);
	if (argc >= 11)
		show = atoi(argv[10]);

	max_thread_num = __max(1, __min(omp_get_num_procs() - 1, max_thread_num));
	std::vector<ZQ_FaceRecognizerArcFaceZQCNN> recognizer_112X112(max_thread_num);
	std::vector<ZQ_FaceRecognizerSphereFaceZQCNN> recognizer_112X96(max_thread_num);
	ZQ_CNN_Net* net = new ZQ_CNN_Net();
	if (!net->LoadFrom(prototxt_file, caffemodel_file, false))
	{
		printf("failed to init recognizer with model %s %s\n", prototxt_file.c_str(), caffemodel_file.c_str());
		delete net;
		return EXIT_FAILURE;
	}
	int c, h, w;
	net->GetInputDim(c, h, w);
	delete net;

	std::vector<ZQ_FaceRecognizer*> recognizers;
	if (c == 3 && h == 112 && w == 112)
	{
		for (int i = 0; i < max_thread_num; i++)
		{
			if (!recognizer_112X112[i].Init("", prototxt_file, caffemodel_file, out_blob_name))
			{
				printf("failed to init recognizer with model %s %s\n", prototxt_file.c_str(), caffemodel_file.c_str());
				return EXIT_FAILURE;
			}
		}
		for (int i = 0; i < max_thread_num; i++)
		{
			recognizers.push_back(&recognizer_112X112[i]);
		}

		if (!ZQ_FaceDatabaseMaker::MakeDatabaseCompactAlreadyCropped(recognizers, database_root, feats_file, names_file, type, show, max_thread_num))
		{
			return EXIT_FAILURE;
		}
	}
	else if (c == 3 && h == 112 && w == 96)
	{
		for (int i = 0; i < max_thread_num; i++)
		{
			if (!recognizer_112X96[i].Init("", prototxt_file, caffemodel_file, out_blob_name))
			{
				printf("failed to init recognizer with model %s %s\n", prototxt_file.c_str(), caffemodel_file.c_str());
				return EXIT_FAILURE;
			}
		}
		for (int i = 0; i < max_thread_num; i++)
		{
			recognizers.push_back(&recognizer_112X112[i]);
		}

		if (!ZQ_FaceDatabaseMaker::MakeDatabaseCompactAlreadyCropped(recognizers, database_root, feats_file, names_file, type, show, max_thread_num))
		{
			return EXIT_FAILURE;
		}
	}
	else
	{
		printf("face recognizer must be H*W*3, H*W = 112*112 or H*W = 112*96\n");
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

int compute_similarity_all_pairs(int argc, char**argv)
{
	if (argc < 6)
	{
		printf("%s %s out_score_file out_flag_file feats_file names_file [max_thread_num]\n", argv[0], argv[1]);
		return EXIT_FAILURE;
	}

	int max_thread_num = 4;
	const std::string out_score_file = argv[2];
	const std::string out_flag_file = argv[3];
	const std::string feats_file = argv[4];
	const std::string names_file = argv[5];
	
	if (argc > 6)
		max_thread_num = atoi(argv[6]);
	ZQ_FaceDatabaseCompact database;
	if (EXIT_FAILURE == load_database_compact(database, feats_file, names_file))
	{
		printf("failed to load database %s %s\n", feats_file.c_str(), names_file.c_str());
		return EXIT_FAILURE;
	}

	if (!database.ExportSimilarityForAllPairs(out_score_file, out_flag_file, max_thread_num))
	{
		printf("failed\n");
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

int load_database_compact(ZQ_FaceDatabaseCompact& database, const std::string& feats_file, const std::string& names_file)
{
	if (database.LoadFromFile(feats_file.c_str(), names_file.c_str()))
		return EXIT_SUCCESS;
	else
		return EXIT_FAILURE;
}
