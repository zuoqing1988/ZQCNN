#include "ZQ_FaceDetectorMTCNN.h"
#include "opencv2\opencv.hpp"
#include <iostream>
#include "ZQ_CNN_ComplieConfig.h"
#if ZQ_CNN_USE_BLAS_GEMM
#include <cblas.h>
#pragma comment(lib,"libopenblas.lib")
#endif

using namespace std;
using namespace cv;
using namespace ZQ;

void Draw(cv::Mat &image, const std::vector<ZQ_CNN_BBox>& thirdBbox)
{
	std::vector<ZQ_CNN_BBox>::const_iterator it = thirdBbox.begin();
	for (; it != thirdBbox.end(); it++)
	{
		if ((*it).exist)
		{
			if (it->score > 0.9)
			{
				cv::rectangle(image, cv::Point((*it).col1, (*it).row1), cv::Point((*it).col2, (*it).row2), cv::Scalar(0, 0, 255), 2, 8, 0);
			}
			else
			{
				cv::rectangle(image, cv::Point((*it).col1, (*it).row1), cv::Point((*it).col2, (*it).row2), cv::Scalar(0, 255, 0), 2, 8, 0);
			}

			for (int num = 0; num < 5; num++)
				circle(image, cv::Point(*(it->ppoint + num) + 0.5f, *(it->ppoint + num + 5) + 0.5f), 3, cv::Scalar(0, 255, 255), -1);
		}
		else
		{
			printf("not exist!\n");
		}
	}
}

int main()
{
#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(1);
#endif
	ZQ_FaceDetector* mtcnn = new ZQ_FaceDetectorMTCNN();
	if (!mtcnn->Init())
	{
		cout << "failed to init mtcnn\n";
		return EXIT_FAILURE;
	}
	
	Mat img = imread("data\\4.jpg");
	vector<ZQ_CNN_BBox> result_mtcnn;
	if (!mtcnn->FindFaceROI(img.data, img.cols, img.rows, img.step[0], ZQ_PIXEL_FMT_BGR, 
		0.1, 0.1, 0.9, 0.9, 40, 0.709, result_mtcnn))
	{
		cout << "failed to find face using MTCNN\n";
		return EXIT_FAILURE;
	}
	
	Mat draw_mtcnn;
	img.copyTo(draw_mtcnn);
	Draw(draw_mtcnn, result_mtcnn);
	namedWindow("MTCNN");
	imshow("MTCNN", draw_mtcnn);
	waitKey(0);
	delete mtcnn;
	return EXIT_SUCCESS;
}