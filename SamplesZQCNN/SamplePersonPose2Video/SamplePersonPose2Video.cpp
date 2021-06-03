#define _CRT_SECURE_NO_WARNINGS
#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "ZQ_CNN_CompileConfig.h"
#include "ZQ_CNN_PersonPose2.h"
#if defined(_WIN32)
#include "ZQlib/ZQ_PutTextCN.h"
#endif
#if ZQ_CNN_USE_BLAS_GEMM
#if __ARM_NEON
#include <openblas/cblas.h>
#else
#include <openblas/cblas.h>
#pragma comment(lib,"libopenblas.lib")
#endif
#elif ZQ_CNN_USE_MKL_GEMM
#include <mkl/mkl.h>
#pragma comment(lib,"mklml.lib")
#else
#pragma comment(lib,"ZQ_GEMM.lib")
#endif

using namespace ZQ;
using namespace std;
using namespace cv;


void Draw14(cv::Mat& img, const std::vector<ZQ_CNN_PersonPose2::BBox>& output)
{
	for (int nn = 0; nn < output.size(); nn++)
	{
		const ZQ_CNN_PersonPose2::BBox& bbox = output[nn];
		cv::rectangle(img, cv::Point(bbox.col1, bbox.row1), cv::Point(bbox.col2, bbox.row2), cv::Scalar(255, 0, 0), 2);

		static const int skeleton[26] = {
			0,1,
			2,5,
			2,8,
			5,11,
			8,11,
			2,3,
			3,4,
			5,6,
			6,7,
			8,9,
			9,10,
			11,12,
			12,13
		};
		for (int i = 0; i < 13; i++)
		{
			int id1 = skeleton[i * 2 + 0];
			int id2 = skeleton[i * 2 + 1];
			if (bbox.points[id1 * 3 + 2] > 0 && bbox.points[id2 * 3 + 2] > 0)
			{
				cv::Point pt1(bbox.points[id1 * 3 + 0], bbox.points[id1 * 3 + 1]);
				cv::Point pt2(bbox.points[id2 * 3 + 0], bbox.points[id2 * 3 + 1]);
				cv::line(img, pt1, pt2, cv::Scalar(255, 0, 0), 2);
			}
		}

		char buf[10];

		for (int i = 0; i < 14; i++)
		{
			cv::Point pt = cv::Point(bbox.points[i * 3], bbox.points[i * 3 + 1]);
			if (bbox.points[i * 3 + 2] > 0.8)
				cv::circle(img, pt, 5, cv::Scalar(0, 0, 250), -1);
			else if (bbox.points[i * 3 + 2] > 0.5)
				cv::circle(img, pt, 5, cv::Scalar(0, 150, 250), -1);
			else if (bbox.points[i * 3 + 2] > 0)
				cv::circle(img, pt, 5, cv::Scalar(0, 250, 0), -1);

#if defined(_WIN32)
			sprintf_s(buf, 10, "%d", i);
#else
			sprintf(buf, "%d", i);
#endif
#if defined(_WIN32)
			ZQ_PutTextCN::PutTextCN(img, buf, pt, cv::Scalar(100, 0, 0), 12);
#endif
		}
	}
}

void Draw10(cv::Mat& img, const std::vector<ZQ_CNN_PersonPose2::BBox>& output)
{
	for (int nn = 0; nn < output.size(); nn++)
	{
		const ZQ_CNN_PersonPose2::BBox& bbox = output[nn];
		cv::rectangle(img, cv::Point(bbox.col1, bbox.row1), cv::Point(bbox.col2, bbox.row2), cv::Scalar(0, 255, 0), 2);

		static const int skeleton[18] = {
			0,1,
			2,5,
			2,8,
			5,9,
			8,9,
			2,3,
			3,4,
			5,6,
			6,7
		};
		for (int i = 0; i < 9; i++)
		{
			int id1 = skeleton[i * 2 + 0];
			int id2 = skeleton[i * 2 + 1];
			if (bbox.points[id1 * 3 + 2] > 0 && bbox.points[id2 * 3 + 2] > 0)
			{
				cv::Point pt1(bbox.points[id1 * 3 + 0], bbox.points[id1 * 3 + 1]);
				cv::Point pt2(bbox.points[id2 * 3 + 0], bbox.points[id2 * 3 + 1]);
				cv::line(img, pt1, pt2, cv::Scalar(0, 255, 0), 2);
			}
		}

		char buf[10];

		for (int i = 0; i < 10; i++)
		{
			cv::Point pt = cv::Point(bbox.points[i * 3], bbox.points[i * 3 + 1]);
			if (bbox.points[i * 3 + 2] > 0.8)
				cv::circle(img, pt, 5, cv::Scalar(0, 0, 250), -1);
			else if (bbox.points[i * 3 + 2] > 0.5)
				cv::circle(img, pt, 5, cv::Scalar(0, 150, 250), -1);
			else if (bbox.points[i * 3 + 2] > 0)
				cv::circle(img, pt, 5, cv::Scalar(0, 250, 0), -1);

#if defined(_WIN32)
			sprintf_s(buf, 10, "%d", i);
#else
			sprintf(buf, "%d", i);
#endif
#if defined(_WIN32)
			ZQ_PutTextCN::PutTextCN(img, buf, pt, cv::Scalar(100, 0, 0), 12);
#endif
		}
	}
}


int main()
{
	int num_threads = 1;

#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(num_threads);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(num_threads);
#endif
	ZQ_CNN_PersonPose2 detector;


#if defined(_WIN32)
	if (!detector.Init("model/MobileNetSSD_deploy.zqparams", "model/MobileNetSSD_deploy.nchwbin", "detection_out", 15,
		"model/Pose-zq161.zqparams", "model/Pose-zq161.nchwbin", "CPM/stage_0_out",
		"model/Pose-zq138-half.zqparams", "model/Pose-zq138-half.nchwbin", "CPM/stage_0_out"))
#else
	if (!detector.Init("../../model/MobileNetSSD_deploy.zqparams", "../../model/MobileNetSSD_deploy.nchwbin", "detection_out", 15,
		"../../model/Pose-zq139.zqparams", "../../model/Pose-zq139.nchwbin", "CPM/stage_0_out",
		"../../model/Pose-zq138-half.zqparams", "../../model/Pose-zq138-half.nchwbin", "CPM/stage_0_out"))
#endif
	{
		cout << "failed to init!\n";
		return EXIT_FAILURE;
	}
	
	//cv::VideoCapture cap("video_20200912_143517.mp4");//12
	//cv::VideoCapture cap("video_20200912_150755.mp4");//13
	cv::VideoCapture cap("video_20200912_152947.mp4");//14
	
	cv::VideoWriter writer;
	cv::Mat image0, ori_im, last_im;
	std::vector<ZQ_CNN_PersonPose2::BBox> output;
	//FILE* out = fopen("11-2dinfo.txt", "w");
	cv::namedWindow("show");

	while (true)
	{
		cap >> image0;

		if (image0.empty())
			break;

		static int fr_id = 0;
		cv::resize(image0, image0, cv::Size(), 0.5, 0.5);
		cv::transpose(image0, image0);
		cv::flip(image0, image0, 0);
		
		
		image0.copyTo(ori_im);
		//image0 = image0(cv::Rect(0, 0, image0.cols / 2, image0.rows));
		/*cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);*/
		//image0.copyTo(ori_im);

		int filter_type = 2;
		if (!writer.isOpened())
		{
			writer.open("personpose2-161-138-14.mp4", CV_FOURCC('X', 'V', 'I', 'D'), 25, cv::Size(ori_im.cols, ori_im.rows));
		}

		if (!detector.DetectVideoSinglePerson(output, image0.data, image0.cols, image0.rows, image0.step[0], 0.5, false, filter_type))
		{
			printf("%d\n", fr_id);
		}
		ori_im.copyTo(last_im);
		//printf("fr_id = %d!\n", fr_id);
		if (output.size() > 0 && (!output[0].half_mode))
			Draw14(ori_im, output);
		else
			Draw10(ori_im, output);

		imshow("show", ori_im);
		char buf[200];
#if defined(_WIN32)
		sprintf_s(buf, 200, "pose-11\\%d.jpg", fr_id);
#else
		sprintf(buf, "pose-11\\%d.jpg", fr_id);
#endif
		
		/*fprintf(out, "%d.jpg %d ", fr_id, output.size());
		for (int nn = 0; nn < output.size(); nn++)
		{
		fprintf(out, "%d ", output[nn].num_points);
		for (int kk = 0; kk < output[nn].num_points; kk++)
		{
		fprintf(out, "%.1f %.1f %.3f ", output[nn].points[kk * 3 + 0], output[nn].points[kk * 3 + 1], output[nn].points[kk * 3 + 2]);
		}
		}
		fprintf(out, "\n");*/
		//cv::imwrite(buf, ori_im);
		writer << ori_im;
		int key = cv::waitKey(10);
		if (key == 27)
			break;

		fr_id++;

	}
	//fclose(out);

	return EXIT_SUCCESS;

}