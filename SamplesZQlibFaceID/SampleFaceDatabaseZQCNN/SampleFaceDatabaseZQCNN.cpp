#if defined(_WIN32)
#include "ZQ_FaceDatabaseMaker.h"
#include "ZQ_FaceRecognizerArcFaceZQCNN.h"
#include "ZQ_FaceRecognizerSphereFaceZQCNN.h"
#include "ZQ_MergeSort.h"
#include <stdio.h>
#include <io.h>
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
using namespace ZQ;

int make_database(int argc, char** argv, bool compact);
int select_subset(int argc, char** argv);
int select_subset_desired_num(int argc, char** argv);
int copy_subset_to_fold(int argc, char** argv);
int compute_similarity_all_pairs(int argc, char**argv, bool compact);
int detect_repeat_person(int argc, char** argv, bool compact);
int detect_lowest_pair(int argc, char** argv);
int evaluate_tar_far(int argc, char** argv);
int load_database(ZQ_FaceDatabase& database, const std::string& feats_file, const std::string& names_file);
int load_database_compact(ZQ_FaceDatabaseCompact& database, const std::string& feats_file, const std::string& names_file);

int main(int argc, char** argv)
{
	cv::setNumThreads(1);
#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(1);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(1);
#endif

	if (argc < 2)
	{
		printf("%s make [args]\n", argv[0]);
		printf("%s make_compact [args]\n", argv[0]);
		printf("%s select_subset [args]\n", argv[0]);
		printf("%s select_subset_desired_num [args]\n", argv[0]);
		printf("%s copy_subset_to_fold [args]\n", argv[0]);
		printf("%s compute_similarity [args]\n", argv[0]);
		printf("%s compute_similarity_compact [args]\n", argv[0]);
		printf("%s detect_repeat [args]\n", argv[0]);
		printf("%s detect_repeat_compact [args]\n", argv[0]);
		printf("%s detect_lowest_pair [args]\n", argv[0]);
		printf("%s evaluate_tar_far [args]\n",argv[0]);
		return EXIT_FAILURE;
	}
	if (_strcmpi(argv[1], "make") == 0)
	{
		return make_database(argc, argv, false);
	}
	else if (_strcmpi(argv[1], "make_compact") == 0)
	{
		return make_database(argc, argv, true);
	}
	else if (_strcmpi(argv[1], "select_subset") == 0)
	{
		return select_subset(argc, argv);
	}
	else if (_strcmpi(argv[1], "select_subset_desired_num") == 0)
	{
		return select_subset_desired_num(argc, argv);
	}
	else if (_strcmpi(argv[1], "copy_subset_to_fold") == 0)
	{
		return copy_subset_to_fold(argc, argv);
	}
	else if (_strcmpi(argv[1], "compute_similarity") == 0)
	{
		return compute_similarity_all_pairs(argc, argv, false);
	}
	else if (_strcmpi(argv[1], "compute_similarity_compact") == 0)
	{
		return compute_similarity_all_pairs(argc, argv, true);
	}
	else if (_strcmpi(argv[1], "detect_repeat") == 0)
	{
		return detect_repeat_person(argc, argv, false);
	}
	else if (_strcmpi(argv[1], "detect_repeat_compact") == 0)
	{
		return detect_repeat_person(argc, argv, true);
	}
	else if (_strcmpi(argv[1], "detect_lowest_pair") == 0)
	{
		return detect_lowest_pair(argc, argv);
	}
	else if (_strcmpi(argv[1], "evaluate_tar_far") == 0)
	{
		return evaluate_tar_far(argc, argv);
	}
	else
	{
		printf("%s make [args]\n", argv[0]);
		printf("%s make_compact [args]\n", argv[0]);
		printf("%s select_subset [args]\n", argv[0]);
		printf("%s select_subset_desired_num [args]\n", argv[0]);
		printf("%s copy_subset_to_fold [args]\n", argv[0]);
		printf("%s compute_similarity [args]\n", argv[0]);
		printf("%s compute_similarity_compact [args]\n", argv[0]);
		printf("%s detect_repeat [args]\n", argv[0]);
		printf("%s detect_repeat_compact [args]\n", argv[0]);
		printf("%s detect_lowest_pair [args]\n", argv[0]);
		printf("%s evaluate_tar_far [args]\n", argv[0]);
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
int make_database(int argc, char** argv, bool compact)
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

		if (compact)
		{
			if (!ZQ_FaceDatabaseMaker::MakeDatabaseCompactAlreadyCropped(recognizers, database_root, feats_file, names_file, type, show, max_thread_num))
			{
				return EXIT_FAILURE;
			}
		}
		else
		{
			if (!ZQ_FaceDatabaseMaker::MakeDatabaseAlreadyCropped(recognizers, database_root, feats_file, names_file, type, show, max_thread_num))
			{
				return EXIT_FAILURE;
			}
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
			recognizers.push_back(&recognizer_112X96[i]);
		}
		if (compact)
		{
			if (!ZQ_FaceDatabaseMaker::MakeDatabaseCompactAlreadyCropped(recognizers, database_root, feats_file, names_file, type, show, max_thread_num))
			{
				return EXIT_FAILURE;
			}

		}
		else
		{
			if (!ZQ_FaceDatabaseMaker::MakeDatabaseAlreadyCropped(recognizers, database_root, feats_file, names_file, type, show, max_thread_num))
			{
				return EXIT_FAILURE;
			}
		}
	}
	else
	{
		printf("face recognizer must be H*W*3, H*W = 112*112 or H*W = 112*96\n");
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

int select_subset(int argc, char** argv)
{
	if (argc < 7)
	{
		printf("%s %s out_file feats_file names_file num_image_thresh similarity_thresh [max_thread_num] \n", argv[0], argv[1]);
		return EXIT_FAILURE;
	}

	int max_thread_num = 4;	
	const std::string out_file = argv[2];
	const std::string feats_file = argv[3];
	const std::string names_file = argv[4];
	int num_image_thresh = atoi(argv[5]);
	float similarity_thresh = atof(argv[6]);
	
	if (argc > 7)
		max_thread_num = atoi(argv[5]);
	
	max_thread_num = __max(1, __min(max_thread_num, omp_get_num_procs() - 1));
	ZQ_FaceDatabase database;
	if (EXIT_FAILURE == load_database(database, feats_file, names_file))
	{
		printf("failed to load database %s %s\n", feats_file.c_str(), names_file.c_str());
		return EXIT_FAILURE;
	}

	if (!database.SelectSubset(out_file, max_thread_num, num_image_thresh, similarity_thresh))
	{
		printf("failed\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

int select_subset_desired_num(int argc, char** argv)
{
	if (argc < 9)
	{
		printf("%s %s out_file feats_file names_file desired_person_num min_desired_image_num max_desired_image_num similarity_thresh [max_thread_num] \n", argv[0], argv[1]);
		return EXIT_FAILURE;
	}

	int max_thread_num = 4;
	const std::string out_file = argv[2];
	const std::string feats_file = argv[3];
	const std::string names_file = argv[4];
	int desired_person_num = atoi(argv[5]);
	int min_desired_img_num = atoi(argv[6]);
	int max_desired_img_num = atoi(argv[7]);
	float similarity_thresh = atof(argv[8]);
	if (argc > 9)
		max_thread_num = atoi(argv[9]);

	max_thread_num = __max(1, __min(max_thread_num, omp_get_num_procs() - 1));
	ZQ_FaceDatabase database;
	if (EXIT_FAILURE == load_database(database, feats_file, names_file))
	{
		printf("failed to load database %s %s\n", feats_file.c_str(), names_file.c_str());
		return EXIT_FAILURE;
	}

	if (!database.SelectSubsetDesiredNum(out_file, desired_person_num, min_desired_img_num, max_desired_img_num,
		max_thread_num, similarity_thresh))
	{
		printf("failed\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

bool zq_copy_file(const char *src, const char *dst, char* buf, int buf_size) 
{
	FILE* in = 0, *out = 0;
	if (0 != fopen_s(&in, src, "rb"))
	{
		printf("failed to open %s\n", src);
		return false;
	}
	if (0 != fopen_s(&out, dst, "wb"))
	{
		printf("failed to create %s\n", dst);
		fclose(in);
		return false;
	}
	
	int read_count = 0;
	while ((read_count = fread(buf, 1, buf_size, in)) > 0)
		fwrite(buf, 1, read_count, out);
	fclose(in);
	fclose(out);
	return true;
}

int copy_subset_to_fold(int argc, char** argv)
{
	if (argc != 4)
	{
		printf("%s %s list_file dst_fold\n", argv[0], argv[1]);
		return EXIT_FAILURE;
	}
	const std::string list_file = argv[2];
	const std::string dst_fold = argv[3];
	FILE* in = 0;
	if (0 != fopen_s(&in, list_file.c_str(), "r"))
	{
		printf("failed to open %s\n", list_file.c_str());
		return EXIT_FAILURE;
	}

	std::ostringstream oss;
	oss << "@echo off";
	system(oss.str().c_str());
	oss.str("");
	oss << "mkdir " << dst_fold;
	system(oss.str().c_str());

	const int BUF_LEN = 1024;
	char buf[BUF_LEN] = { 0 };
	char person_name[BUF_LEN] = { 0 };
	const int buffer_for_copy_size = 1024 * 1024;
	std::vector<char> buffer_for_copy(buffer_for_copy_size);
	__int64 copied_num = 0;
	while (true)
	{
		buf[0] = '\0';
		fgets(buf, BUF_LEN - 1, in);
		if (buf[0] == '\0')
			break;

		int len = strlen(buf);
		if (buf[len - 1] == '\n')
			buf[--len] = '\0';
		if (buf[0] == '\0')
			continue;
		bool copy_done = false;

		bool found1 = false;
		bool found2 = false;
		int j1, j2;
		for (j1 = len - 1; j1 >= 0; j1--)
		{
			if (buf[j1] == '\\')
			{
				found1 = true;
				break;
			}
		}
		if (found1)
		{
			for (j2 = j1 - 1; j2 >= 0; j2--)
			{
				if (buf[j2] == '\\')
				{
					found2 = true;
					break;
				}
			}
		}
		else
		{
			printf("not found1\n");
		}
		if (found2)
		{
			strncpy_s(person_name, BUF_LEN - 1, buf + j2 + 1, j1 - j2 - 1);
			person_name[j1 - j2 - 1] = '\0';
			oss.str("");
			oss << dst_fold << "\\" << std::string(person_name);
			std::string dst_person_fold = oss.str();
			if (0 != _access(dst_person_fold.c_str(), 0))
			{
				oss.str("");
				oss << "mkdir " << dst_person_fold;
				system(oss.str().c_str());
			}
			dst_person_fold.append("\\");
			dst_person_fold.append(buf + j1 + 1);
			copy_done = zq_copy_file(buf, dst_person_fold.c_str(), &buffer_for_copy[0], buffer_for_copy_size);
		}
		else
		{
			printf("not found2\n");
		}

		if (!copy_done)
		{
			printf("failed to copy file %s\n", buf);
		}
		else
			copied_num++;
		if (copied_num % 1000 == 0)
			printf("%lld files copied\n", copied_num);
	}
	fclose(in);
	return EXIT_SUCCESS;
}

int compute_similarity_all_pairs(int argc, char**argv, bool compact)
{
	if (argc < 7)
	{
		printf("%s %s out_score_file out_flag_file out_info_file feats_file names_file [max_thread_num] [quantization]\n", argv[0], argv[1]);
		return EXIT_FAILURE;
	}

	int max_thread_num = 4;
	bool quantization = false;
	const std::string out_score_file = argv[2];
	const std::string out_flag_file = argv[3];
	const std::string out_info_file = argv[4];
	const std::string feats_file = argv[5];
	const std::string names_file = argv[6];
	
	if (argc > 7)
		max_thread_num = atoi(argv[7]);
	if (argc > 8)
		quantization = atoi(argv[8]);
	ZQ_FaceDatabaseCompact database_compact;
	ZQ_FaceDatabase database;

	__int64 all_pair_num = 0, same_pair_num = 0, notsame_pair_num = 0;
	if (compact)
	{
		if (EXIT_FAILURE == load_database_compact(database_compact, feats_file, names_file))
		{
			printf("failed to load database %s %s\n", feats_file.c_str(), names_file.c_str());
			return EXIT_FAILURE;
		}

		if (!database_compact.ExportSimilarityForAllPairs(out_score_file, out_flag_file,
			all_pair_num, same_pair_num, notsame_pair_num, max_thread_num, quantization))
		{
			printf("failed\n");
			return EXIT_FAILURE;
		}
	}
	else
	{
		if (EXIT_FAILURE == load_database(database, feats_file, names_file))
		{
			printf("failed to load database %s %s\n", feats_file.c_str(), names_file.c_str());
			return EXIT_FAILURE;
		}

		if (!database.ExportSimilarityForAllPairs(out_score_file, out_flag_file,
			all_pair_num, same_pair_num, notsame_pair_num, max_thread_num, quantization))
		{
			printf("failed\n");
			return EXIT_FAILURE;
		}
	}
	
	FILE* out = 0;
	if (0 != fopen_s(&out, out_info_file.c_str(), "w"))
	{
		printf("failed to create file %s\n", out_info_file.c_str());
		return EXIT_FAILURE;
	}
	fprintf(out, "%lld %lld %lld\n", all_pair_num, same_pair_num, notsame_pair_num);
	fclose(out);
	printf("all_pair_num:%lld, same_pair_num:%lld, notsame_pair_num:%lld\n", all_pair_num, same_pair_num, notsame_pair_num);
	return EXIT_SUCCESS;
}

int detect_repeat_person(int argc, char**argv, bool compact)
{
	if (argc < 5)
	{
		printf("%s %s out_file feats_file names_file [similarity_thresh] [max_thread_num] [only_pivot]\n", argv[0], argv[1]);
		return EXIT_FAILURE;
	}

	float similarity_thresh = 0.5f;
	int max_thread_num = 4;
	bool only_pivot = true;
	const std::string out_file = argv[2];
	const std::string feats_file = argv[3];
	const std::string names_file = argv[4];
	if (argc > 5)
		similarity_thresh = atof(argv[5]);
	if (argc > 6)
		max_thread_num = atoi(argv[6]);
	if (argc > 7)
		only_pivot = atoi(argv[7]);
	max_thread_num = __max(1, __min(max_thread_num, omp_get_num_procs() - 1));
	ZQ_FaceDatabaseCompact database_compact;
	ZQ_FaceDatabase database;

	if (compact)
	{
		if (EXIT_FAILURE == load_database_compact(database_compact, feats_file, names_file))
		{
			printf("failed to load database %s %s\n", feats_file.c_str(), names_file.c_str());
			return EXIT_FAILURE;
		}

		if (!database_compact.DetectRepeatPerson(out_file, max_thread_num, similarity_thresh, only_pivot))
		{
			printf("failed\n");
			return EXIT_FAILURE;
		}
	}
	else
	{
		if (EXIT_FAILURE == load_database(database, feats_file, names_file))
		{
			printf("failed to load database %s %s\n", feats_file.c_str(), names_file.c_str());
			return EXIT_FAILURE;
		}

		if (!database.DetectRepeatPerson(out_file, max_thread_num, similarity_thresh))
		{
			printf("failed\n");
			return EXIT_FAILURE;
		}
	}

	return EXIT_SUCCESS;
}

int detect_lowest_pair(int argc, char** argv)
{
	if (argc < 5)
	{
		printf("%s %s out_file feats_file names_file [similarity_thresh] [max_thread_num]\n", argv[0], argv[1]);
		return EXIT_FAILURE;
	}
	float similarity_thresh = 0.5f;
	int max_thread_num = 4;
	const std::string out_file = argv[2];
	const std::string feats_file = argv[3];
	const std::string names_file = argv[4];
	if (argc > 5)
		similarity_thresh = atof(argv[5]);
	if (argc > 6)
		max_thread_num = atoi(argv[6]);
	max_thread_num = __max(1, __min(max_thread_num, omp_get_num_procs() - 1));
	ZQ_FaceDatabase database;

	
	if (EXIT_FAILURE == load_database(database, feats_file, names_file))
	{
		printf("failed to load database %s %s\n", feats_file.c_str(), names_file.c_str());
		return EXIT_FAILURE;
	}

	if (!database.DetectLowestPair(out_file, max_thread_num, similarity_thresh))
	{
		printf("failed\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

int evaluate_tar_far(int argc, char** argv)
{
	if (argc < 5)
	{
		printf("%s %s score_file flag_file info_file [quantization]\n", argv[0], argv[1]);
		return EXIT_FAILURE;
	}

	bool quantization = false;
	const std::string score_file = argv[2];
	const std::string flag_file = argv[3];
	const std::string info_file = argv[4];
	if (argc > 5)
		quantization = atoi(argv[5]);

	std::string dst_score_file = score_file + ".sort";
	std::string dst_flag_file = flag_file + ".sort";
	
	__int64 all_pair_num = 0, same_pair_num = 0, notsame_pair_num = 0;
	FILE* in = 0;
	if (0 != fopen_s(&in, info_file.c_str(), "r"))
	{
		printf("failed to open %s\n", info_file.c_str());
		return EXIT_FAILURE;
	}

	if (1 != fscanf_s(in, "%lld",&all_pair_num)
		|| 1 != fscanf_s(in, "%lld", &same_pair_num)
		|| 1 != fscanf_s(in, "%lld", &notsame_pair_num))
	{
		printf("failed to read info from %s\n", info_file.c_str());
		return EXIT_FAILURE;
	}
	fclose(in);
	printf("all_pair_num = %lld, same_pair_num = %lld, notsame_pair_num = %lld\n",
		all_pair_num, same_pair_num, notsame_pair_num);

	if (quantization)
	{
		if (!ZQ_MergeSort::MergeSortWithData_OOC<short>(score_file.c_str(), dst_score_file.c_str(), flag_file.c_str(), dst_flag_file.c_str(),
			1, false, 1024 * 1024 * 2))
		{
			printf("failed to run MergeSortWithData_OOC\n");
			return EXIT_FAILURE;
		}
	}
	else
	{
		if (!ZQ_MergeSort::MergeSortWithData_OOC<float>(score_file.c_str(), dst_score_file.c_str(), flag_file.c_str(), dst_flag_file.c_str(),
			1, false, 1024 * 1024 * 2))
		{
			printf("failed to run MergeSortWithData_OOC\n");
			return EXIT_FAILURE;
		}
	}

	FILE* in1 = 0, *in2 = 0;
	if (0 != fopen_s(&in1, dst_score_file.c_str(), "rb"))
	{
		printf("failed to open %s\n", dst_score_file.c_str());
		return EXIT_FAILURE;
	}
	if (0 != fopen_s(&in2, dst_flag_file.c_str(), "rb"))
	{
		printf("failed to open %s\n", dst_flag_file.c_str());
		fclose(in1);
		return EXIT_FAILURE;
	}
	/**************/

	int cur_stage = 0;
	double stage_far_thresh[] = 
	{
		1e-8, 2*1e-8, 5*1e-8, 
		1e-7, 2*1e-7, 5*1e-7,
		1e-6, 2*1e-6, 5*1e-6,
		1e-5, 2*1e-5, 5*1e-5,
		1e-4, 2*1e-4, 5*1e-4,
		1e-3
	};
	int stage_num = sizeof(stage_far_thresh) / sizeof(double);
	std::vector<double> stage_num_thresh(stage_num);
	for (int i = 0; i < stage_num; i++)
		stage_num_thresh[i] = stage_far_thresh[i] * notsame_pair_num;

	/***************/
	__int64 cur_far_num = 0, cur_tar_num = 0;
	float simi_thresh = 10;
	int buffer_size = 1024 * 1024 * 100;
	std::vector<short> short_score_buffer;
	std::vector<float> score_buffer;
	if (quantization)
		short_score_buffer.resize(buffer_size);
	else
		score_buffer.resize(buffer_size);
	std::vector<char> flag_buffer(buffer_size);
	__int64 read_count1 = 0, read_count2;
	__int64 rest_count = all_pair_num;
	while (rest_count > 0)
	{
		if(quantization)
			read_count1 = fread(&short_score_buffer[0], sizeof(short), buffer_size, in1);
		else
			read_count1 = fread(&score_buffer[0], sizeof(float), buffer_size, in1);
		if (read_count1 == 0)
		{
			printf("failed to read desired data\n");
			fclose(in1);
			fclose(in2);
			return EXIT_FAILURE;
		}
		read_count2 = fread(&flag_buffer[0], 1, buffer_size, in2);
		if (read_count2 == 0)
		{
			printf("failed to read desired data\n");
			fclose(in1);
			fclose(in2);
			return EXIT_FAILURE;
		}
		if(read_count1 != read_count2
			|| (read_count1 != buffer_size && read_count1 != rest_count))
		{
			printf("score_file flag_file info_file not match\n");
			fclose(in1);
			fclose(in2);
			return EXIT_FAILURE;
		}

		for (int i = 0; i < read_count1; i++)
		{
			if (cur_stage >= stage_num)
				break;
			if (cur_far_num > stage_num_thresh[cur_stage])
			{
				printf("thresh = %8.5f, far = %12e, tar = %.5f\n", 
					quantization ? ((float)short_score_buffer[i]/SHRT_MAX): score_buffer[i], 
					(double)cur_far_num/notsame_pair_num,
					(double)cur_tar_num / same_pair_num);
				cur_stage++;
			}

			if (flag_buffer[i])
			{
				cur_tar_num++;
			}
			else
			{
				cur_far_num++;
			}
		}
		rest_count -= read_count1;
	}
	fclose(in1);
	fclose(in2);
	return EXIT_SUCCESS;
}

int load_database(ZQ_FaceDatabase& database, const std::string& feats_file, const std::string& names_file)
{
	if (database.LoadFromFileBinay(feats_file.c_str(), names_file.c_str()))
		return EXIT_SUCCESS;
	else
		return EXIT_FAILURE;
}

int load_database_compact(ZQ_FaceDatabaseCompact& database, const std::string& feats_file, const std::string& names_file)
{
	if (database.LoadFromFile(feats_file.c_str(), names_file.c_str()))
		return EXIT_SUCCESS;
	else
		return EXIT_FAILURE;
}


#else
#include <stdio.h>
int main(int argc, const char** argv)
{
	printf("%s only support windows\n", argv[0]);
	return 0;
}
#endif