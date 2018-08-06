#include "ZQ_FaceRecognizerArcFaceMiniCaffe.h"
#include <cblas.h>
using namespace ZQ;
using namespace cv;
using namespace std;

int main()
{
	openblas_set_num_threads(1);
	ZQ_FaceRecognizer* recognizer[1] = { 0 };
	std::string prototxt_file = "model/model-r50-am.prototxt";
	std::string caffemodel_file = "model/model-r50-am.caffemodel";
	std::string out_blob_name = "fc1";
	bool fail_flag = false;
	recognizer[0] = new ZQ_FaceRecognizerArcFaceMiniCaffe();
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
	int iters = 1;
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
	printf("time cost: %.3f/%d = %.3f secs\n", t2 - t1, iters*2, (t2-t1)/iters/2.0);
	if (recognizer[0]) delete recognizer[0];

	return EXIT_SUCCESS;
}