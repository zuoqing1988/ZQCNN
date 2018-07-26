#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_MTCNN.h"
#include <cblas.h>
#include <vector>
#include <iostream>
#include "opencv2\opencv.hpp"
using namespace ZQ;
using namespace std;
using namespace cv;

static void Draw(cv::Mat &image, const std::vector<ZQ_CNN_BBox>& thirdBbox)
{
	std::vector<ZQ_CNN_BBox>::const_iterator it = thirdBbox.begin();
	for (; it != thirdBbox.end(); it++)
	{
		if ((*it).exist)
		{
			if (it->score > 0.7)
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

	openblas_set_num_threads(1);
	Mat image0 = cv::imread("data\\4.jpg", 1);
	if (image0.empty())
	{
		cout << "empty image\n";
		return EXIT_FAILURE;
	}

	std::vector<ZQ_CNN_BBox> thirdBbox;
	ZQ_CNN_MTCNN mtcnn;
	std::string result_name;


	result_name = "resultdet.jpg";
	if (!mtcnn.Init("model\\det1.param", "model\\det1.nchwbin", "model\\det2.param",
		"model\\det2.nchwbin", "model\\det3.param", "model\\det3.nchwbin"))
	{
		cout << "failed to init!\n";
		return EXIT_FAILURE;
	}

	mtcnn.SetPara(image0.cols, image0.rows, 60, 0.6, 0.7, 0.7, 0.5, 0.5, 0.5);

	int iters = 100;
	double t1 = omp_get_wtime();
	for (int i = 0; i < iters; i++)
	{
		if (!mtcnn.Find(image0.data, image0.cols, image0.rows, image0.step[0], thirdBbox))
		{
			cout << "failed to find face!\n";
			return EXIT_FAILURE;
		}
	}
	double t2 = omp_get_wtime();
	printf("total %.3f s / %d = %.3f ms\n", t2 - t1, iters, 1000 * (t2 - t1) / iters);

	Draw(image0, thirdBbox);

	
	//cv::resize(image0, image0, cv::Size(), 0.5, 0.5);

	namedWindow("resultSSE");
	imwrite(result_name, image0);
	imshow("resultSSE", image0);


	waitKey(0);
	return EXIT_SUCCESS;
}
