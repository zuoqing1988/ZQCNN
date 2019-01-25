#if defined(_WIN32)
#include "ZQ_FaceDetectorLibFaceDetect.h"
#include "opencv2\opencv.hpp"
#include <iostream>

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
	ZQ_FaceDetector* libfacedetect = new ZQ_FaceDetectorLibFaceDetect();
	if (!libfacedetect->Init())
	{
		cout << "failed to init libfacedetect\n";
		return EXIT_FAILURE;
	}

	//Mat img = imread("data/4.jpg");
	Mat img = imread("data/test2.jpg");
	vector<ZQ_CNN_BBox> result_libfacedetect;
	
	if (!libfacedetect->FindFaceROI(img.data, img.cols, img.rows, img.step[0], ZQ_PIXEL_FMT_BGR, 
		0.0, 0.0, 1.0, 1.0, 20, 1.2, result_libfacedetect))
	{
		cout << "failed to find face using LibFaceDetect\n";
		return EXIT_FAILURE;
	}
	cout << "result_libfacedetect: " << result_libfacedetect.size() << "\n";
	Mat draw_libfacedetect;
	img.copyTo(draw_libfacedetect);
	Draw(draw_libfacedetect, result_libfacedetect);
	imwrite("libfacedetect.jpg", draw_libfacedetect);
	namedWindow("LibFaceDetect");
	imshow("LibFaceDetect", draw_libfacedetect);
	waitKey(0);	
	delete libfacedetect;
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