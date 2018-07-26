#include "ZQ_CNN_Net.h"
#include <cblas.h>
#include <vector>
#include <iostream>
#include "opencv2\opencv.hpp"
using namespace ZQ;
using namespace std;
using namespace cv;

struct BBox {
	float x1, y1, x2, y2, score;
	int label;
};

int main()
{
	openblas_set_num_threads(1);
	Mat img0 = cv::imread("data\\004545.jpg", 1);
	if (img0.empty())
	{
		cout << "empty image\n";
		return EXIT_FAILURE;
	}
	int width = img0.cols;
	int height = img0.rows;
	cv::Mat img1;
	cv::resize(img0, img1, cv::Size(300, 300));
	ZQ_CNN_Tensor4D_NHW_C_Align128bit input0, input1;
	input0.ConvertFromBGR(img1.data, img1.cols, img1.rows, img1.step[0]);


	ZQ_CNN_Net net;
	if (!net.LoadFrom("model\\MobileNetSSD_deploy.zqparams", "model\\MobileNetSSD_deploy.nchwbin"))
	{
		cout << "failed to load net\n";
		return EXIT_FAILURE;
	}

	int iters = 100;
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

	const ZQ_CNN_Tensor4D* ptr = net.GetBlobByName("detection_out");
	// get output, shape is N x 7
	if (ptr == 0)
	{
		printf("maybe the output blob is incorrect\n");
		return EXIT_FAILURE;
	}

	const float kScoreThreshold = 0.5f;
	const char* kClassNames[] = { "__background__", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair",
		"cow", "diningtable", "dog", "horse",
		"motorbike", "person", "pottedplant",
		"sheep", "sofa", "train", "tvmonitor" };
	const float* result_data = ptr->GetFirstPixelPtr();
	int sliceStep = ptr->GetSliceStep();
	int N = ptr->GetN();
	printf("detected = %d\n", N);
	vector<BBox> detections;
	for (int k = 0; k < N; k++) 
	{
		if (result_data[0] != -1 && result_data[2] > kScoreThreshold) 
		{
			// [image_id, label, score, xmin, ymin, xmax, ymax]
			BBox bbox;
			bbox.x1 = result_data[3] * width;
			bbox.y1 = result_data[4] * height;
			bbox.x2 = result_data[5] * width;
			bbox.y2 = result_data[6] * height;
			bbox.score = result_data[2];
			bbox.label = static_cast<int>(result_data[1]);
			detections.push_back(bbox);
		}
		result_data += sliceStep;
	}

	// draw
	for (auto& bbox : detections) 
	{
		cv::Rect rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1 + 1, bbox.y2 - bbox.y1 + 1);
		cv::rectangle(img0, rect, cv::Scalar(0, 0, 255), 2);
		char buff[300];
		sprintf_s(buff, 300, "%s: %.2f", kClassNames[bbox.label], bbox.score);
		cv::putText(img0, buff, cv::Point(bbox.x1, bbox.y1), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));
	}

	cv::imwrite("./ssd-result.jpg", img0);
	cv::imshow("ZQCNN-SSD", img0);
	cv::waitKey(0);

	return EXIT_SUCCESS;
}
