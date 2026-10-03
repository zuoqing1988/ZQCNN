#include "ZQ_CNN_SSD.h"
#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"
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
using namespace std;
using namespace cv;



int main()
{
	int thread_num = 1;

#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(thread_num);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(thread_num);
#endif

	Mat img0;
#if defined(_WIN32)
	//Mat img1 = cv::imread("data/dog.jpg", 1);
	//Mat img1 = cv::imread("data/004545.jpg", 1);
	Mat img1 = cv::imread("data/4_320x240.jpg", 1);
#else
	//Mat img1 = cv::imread("../../data/dog.jpg", 1);
	//Mat img1 = cv::imread("../../data/004545.jpg", 1);
	Mat img1 = cv::imread("../../data/4_320x240.jpg", 1);
#endif
	
	if (img1.empty())
	{
		cout << "empty image\n";
		return EXIT_FAILURE;
	}
	cv::cvtColor(img1, img0, CV_BGR2RGB);
	//img0 = img1;
	ZQ_CNN_SSD detector;
#if defined(_WIN32)
	//if (!detector.Init("model/MobileNetSSD_deploy.zqparams", "model/MobileNetSSD_deploy.nchwbin", "detection_out"))
	if (!detector.Init("model/libfacedetection.zqparams", "model/libfacedetection.nchwbin", "detection_out"))
	//if (!detector.Init("model/ssd-300.zqparams", "model/ssd-300.nchwbin", "detection", true))
#else
	//if (!detector.Init("../../model/MobileNetSSD_deploy.zqparams", "../../model/MobileNetSSD_deploy.nchwbin", "detection_out"))
	if (!detector.Init("../../model/libfacedetection.zqparams", "../../model/libfacedetection.nchwbin", "detection_out"))
		//if (!detector.Init("../../model/ssd-300.zqparams", "../../model/ssd-300.nchwbin", "detection", true))
#endif
	{
		printf("failed to init detector!\n");
		return false;
	}
	
	int out_iter = 10;
	int iters = 100;
	std::vector<ZQ_CNN_SSD::BBox> output;
	const float kScoreThreshold = 0.3f;
	for (int out_it = 0; out_it < out_iter; out_it++)
	{
		double t1 = omp_get_wtime();
		for (int it = 0; it < iters; it++)
		{
			if(!detector.Detect(output, img0.data, img0.cols, img0.rows, img0.step[0], kScoreThreshold, false))
			{
				cout << "failed to run\n";
				return EXIT_FAILURE;
			}
		}
		double t2 = omp_get_wtime();
		printf("[%d] times cost %.3f s, 1 iter cost %.3f ms\n", iters, t2 - t1, 1000 * (t2 - t1) / iters);
	}
	/*const char* kClassNames[] = { "__background__", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair",
		"cow", "diningtable", "dog", "horse",
		"motorbike", "person", "pottedplant",
		"sheep", "sofa", "train", "tvmonitor" };*/
	const char* kClassNames[] = { "__background__", "face"};
	//const char* kClassNames[] = { "__background__", "eye", "nose", "mouth", "face" };
	
	
	// draw
	// 检出数打出来（2026-10-04 加，附录 GS.2）：`output` 本来就是
	// std::vector<BBox>，框数一直在手上，只是原来只把它画进图片里 ——
	// 而 `imshow` 按用户要求注释掉了，于是无头环境下**结果没有任何出口**。
	// 没有这一行，"SSD 路径坏掉了"与"它跑得好好的"在回归里都是
	// rc=0 + 一堆计时，**分不出来**。
	printf("final found num: %d\n", (int)output.size());
	for (auto& bbox : output)
	{
		cv::Rect rect(bbox.col1, bbox.row1, bbox.col2 - bbox.col1 + 1, bbox.row2 - bbox.row1 + 1);
		cv::rectangle(img1, rect, cv::Scalar(0, 0, 255), 2);
		char buff[300];
		const char* class_name = "unknown";
		if (bbox.label >= 0 && bbox.label < (int)(sizeof(kClassNames) / sizeof(kClassNames[0])))
			class_name = kClassNames[bbox.label];
#if defined(_WIN32)
		sprintf_s(buff, 300, "%s: %.2f", class_name, bbox.score);
#else
		sprintf(buff, "%s: %.2f", class_name, bbox.score);
#endif
		cv::putText(img1, buff, cv::Point(bbox.col1, bbox.row1), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));
	}

	cv::imwrite("./ssd-result.jpg", img1);
	// cv::imshow("ZQCNN-SSD", img1);
	// cv::waitKey(0);

	return EXIT_SUCCESS;
}
