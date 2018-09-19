#include "ZQ_FaceRecognizerArcFaceZQCNN.h"
#include "ZQ_CNN_ComplieConfig.h"
#if ZQ_CNN_USE_BLAS_GEMM
#include <cblas.h>
#pragma comment(lib,"libopenblas.lib")
#endif
using namespace ZQ;
using namespace cv;
using namespace std;

int main()
{
#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(1);
#endif
	ZQ_FaceRecognizer* recognizer[1] = { 0 };
	std::string prototxt_file = "model/model-r50-am.zqparams";
	std::string caffemodel_file = "model/model-r50-am.nchwbin";
	std::string out_blob_name = "fc5";
	for (int i = 0; i < 1000; i++)
	{
		std::string prototxt_file = "./model/model-r50-am.zqparams";
		std::string caffemodel_file = "./model/model-r50-am.nchwbin";
		std::string out_blob_name = "fc5";
		ZQ_FaceRecognizerArcFaceZQCNN* pFaceZQCNN = new ZQ_FaceRecognizerArcFaceZQCNN();
		if (!pFaceZQCNN->Init("", prototxt_file, caffemodel_file, out_blob_name))
		{
			cout << "failed to init arcface\n";
			return 0;
		}
		delete pFaceZQCNN;
	}

	bool fail_flag = false;
	recognizer[0] = new ZQ_FaceRecognizerArcFaceZQCNN();
	if (!recognizer[0]->Init("", prototxt_file, caffemodel_file, out_blob_name))
	{
		cout << "failed to init arcface\n";
		fail_flag = true;
	}

	if (fail_flag)
	{
		if (recognizer[0]) delete recognizer[0];
		return EXIT_FAILURE;
	}

	Mat img0 = imread("data\\00_.jpg");
	Mat img1 = imread("data\\01_.jpg");
	double t1 = omp_get_wtime();
	int iters = 200;
	for (int it = 0; it < iters; it++)
	{
		int feat_dim = recognizer[0]->GetFeatDim();
		std::vector<float> feat0(feat_dim), feat1(feat_dim);
		recognizer[0]->ExtractFeature(img0.data, img0.step[0], ZQ_PIXEL_FMT_BGR, &feat0[0], true);
		recognizer[0]->ExtractFeature(img1.data, img1.step[0], ZQ_PIXEL_FMT_BGR, &feat1[0], true);
		float score = recognizer[0]->CalSimilarity(&feat0[0], &feat1[0]);
		cout << "gets score " << score << "\n";

	}
	double t2 = omp_get_wtime();
	printf("time cost: %.3f/%d = %.3f secs\n", t2 - t1, iters * 2, (t2 - t1) / iters / 2.0);
	if (recognizer[0]) delete recognizer[0];

	return EXIT_SUCCESS;
}