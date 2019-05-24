#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_MTCNN.h"
#if defined(_WIN32)
#include "ZQlib/ZQ_PutTextCN.h"
#endif
#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "ZQ_CNN_CompileConfig.h"
#if ZQ_CNN_USE_BLAS_GEMM
#if __ARM_NEON
#include <openblas/cblas.h>
#else
#include <openblas/cblas.h>
#pragma comment(lib,"libopenblas.lib")
#endif
#elif ZQ_CNN_USE_MKL_GEMM
#include "mkl/mkl.h"
#pragma comment(lib,"mklml.lib")
#else
#pragma comment(lib,"ZQ_GEMM.lib")
#endif
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
				circle(image, cv::Point(*(it->ppoint + num) + 0.5f, *(it->ppoint + num + 5) + 0.5f), 1, cv::Scalar(0, 255, 255), -1);
		}
		else
		{
			printf("not exist!\n");
		}
	}
}

static void Draw(cv::Mat &image, const std::vector<ZQ_CNN_BBox106>& thirdBbox)
{
	std::vector<ZQ_CNN_BBox106>::const_iterator it = thirdBbox.begin();
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

			for (int num = 0; num < 106; num++)
				circle(image, cv::Point(*(it->ppoint + num * 2) + 0.5f, *(it->ppoint + num * 2 + 1) + 0.5f), 2, cv::Scalar(0, 255, 0), -1);
		}
		else
		{
			printf("not exist!\n");
		}
	}
}

class filter
{
private:
	ZQ_CNN_BBox106 last;
	float cur_weight;
	float restart;
public:
	filter() 
	{ 
		cur_weight = 0.5; 
		restart = true;
	}
	~filter() {}
	void filtering(std::vector<ZQ_CNN_BBox106>& pts)
	{
		if (pts.size() != 1)
			restart = true;
		else
		{
			if (!restart)
			{
				_filtering(pts[0]);
			}
			last = pts[0];
			restart = false;
		}	
	}
private:
	void _filtering(ZQ_CNN_BBox106& pts)
	{
		for (int i = 0; i < 212; i++)
		{
			pts.ppoint[i] *= cur_weight;
			pts.ppoint[i] += (1.0 - cur_weight)*last.ppoint[i];
		}
	}
};

int run_cam()
{
	int num_threads = 1;
#if ZQ_CNN_USE_BLAS_GEMM
	printf("set openblas thread_num = %d\n", num_threads);
	openblas_set_num_threads(num_threads);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(num_threads);
#endif

	filter my_filter;
	std::vector<ZQ_CNN_BBox> thirdBbox;
	std::vector<ZQ_CNN_BBox> thirdBbox_last;
	std::vector<ZQ_CNN_BBox106> thirdBbox106;
	std::vector<ZQ_CNN_BBox106> thirdBbox106_last;
	ZQ_CNN_MTCNN mtcnn;
	std::string result_name;
	mtcnn.TurnOffShowDebugInfo();
	//mtcnn.SetLimit(300, 50, 20);
	const int use_pnet20 = true;
	bool landmark106 = false;
	int thread_num = 0;
	bool special_handle_very_big_face = true;
	result_name = "resultdet.jpg";
	if (use_pnet20)
	{
		if (landmark106)
		{
#if defined(_WIN32)
			if (!mtcnn.Init(
				//"model/det1-dw20-fast.zqparams", "model/det1-dw20-fast.nchwbin",
				"model/det1-dw20-plus.zqparams", "model/det1-dw20-plus.nchwbin",
				//"model/det2-dw24-fast.zqparams", "model/det2-dw24-fast.nchwbin",
				"model/det2-dw24-plus.zqparams", "model/det2-dw24-plus.nchwbin",
				//"model/det3-dw48-fast.zqparams", "model/det3-dw48-fast.nchwbin", 
				"model/det3-dw48-plus.zqparams", "model/det3-dw48-plus.nchwbin",
				thread_num, true,
				"model/det5-dw112.zqparams", "model/det5-dw112.nchwbin"
				//"model/det5-dw96-v3s.zqparams", "model/det5-dw96-v3s.nchwbin"
#else
			if (!mtcnn.Init("../../model/det1-dw20-fast.zqparams", "../../model/det1-dw20-fast.nchwbin",
				"../../model/det2-dw24-fast.zqparams", "../../model/det2-dw24-fast.nchwbin",
				//"../../model/det2.zqparams", "../../model/det2_bgr.nchwbin",
				"../../model/det3-dw48-fast.zqparams", "../../model/det3-dw48-fast.nchwbin",
				thread_num, true,
				"../../model/det5-dw64-v3s.zqparams", "../../model/det5-dw64-v3s.nchwbin"
				//"../../model/det3.zqparams", "../../model/det3_bgr.nchwbin"
#endif
			))
			{
				cout << "failed to init!\n";
				return EXIT_FAILURE;
			}
		}
		else
		{
#if defined(_WIN32)
			if (!mtcnn.Init(
				//"model/det1-dw20-fast.zqparams", "model/det1-dw20-fast.nchwbin",
				"model/det1-dw20-plus.zqparams", "model/det1-dw20-plus.nchwbin",
				//"model/det2-dw24-fast.zqparams", "model/det2-dw24-fast.nchwbin",
				"model/det2-dw24-plus.zqparams", "model/det2-dw24-plus.nchwbin",
				//"model/det3-dw48-fast.zqparams", "model/det3-dw48-fast.nchwbin", 
				"model/det3-dw48-plus.zqparams", "model/det3-dw48-plus.nchwbin",
				thread_num, true,
				"model/det4-dw64-v3s.zqparams", "model/det4-dw64-v3s.nchwbin"
				//"model/det3.zqparams", "model/det3_bgr.nchwbin"
#else
			if (!mtcnn.Init("../../model/det1-dw20-fast.zqparams", "../../model/det1-dw20-fast.nchwbin",
				"../../model/det2-dw24-fast.zqparams", "../../model/det2-dw24-fast.nchwbin",
				//"model/det2.zqparams", "model/det2_bgr.nchwbin",
				"../../model/det3-dw48-fast.zqparams", "../../model/det3-dw48-fast.nchwbin",
				thread_num, false,
				"model/det4-dw48-v2s.zqparams", "model/det4-dw48-v2s.nchwbin"
				//"../../model/det3.zqparams", "../../model/det3_bgr.nchwbin"
#endif
			))
			{
				cout << "failed to init!\n";
				return EXIT_FAILURE;
			}
		}
	}
	else
	{
#if defined(_WIN32)
		if (!mtcnn.Init("model/det1.zqparams", "model/det1_bgr.nchwbin",
			"model/det2.zqparams", "model/det2_bgr.nchwbin",
			"model/det3.zqparams", "model/det3_bgr.nchwbin", thread_num))
#else
		if (!mtcnn.Init("../../model/det1.zqparams", "../../model/det1_bgr.nchwbin",
			"../../model/det2.zqparams", "../../model/det2_bgr.nchwbin",
			"../../model/det3.zqparams", "../../model/det3_bgr.nchwbin", thread_num))
#endif
		{
			cout << "failed to init!\n";
			return EXIT_FAILURE;
		}
	}


	cv::VideoCapture cap("video_20190518_172153_540P.mp4");
	//cv::VideoCapture cap(0);
	cv::VideoWriter writer;
	cv::Mat image0;
	cv::namedWindow("show");
	while (true)
	{
		cap >> image0;
		if (image0.empty())
			break;

		//cv::resize(image0, image0, cv::Size(), 0.5, 0.5);
		//cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		//cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		if (!writer.isOpened())
			writer.open("cam1.mp4", CV_FOURCC('X', 'V', 'I', 'D'), 25, cv::Size(image0.cols, image0.rows));
		if(use_pnet20)
			mtcnn.SetPara(image0.cols, image0.rows, 120, 0.5, 0.6, 0.8, 0.4, 0.5, 0.5, 0.709, 3, 20, 4, special_handle_very_big_face);
		else
			mtcnn.SetPara(image0.cols, image0.rows, 120, 0.6, 0.7, 0.8, 0.5, 0.5, 0.5, 0.709, 4, 20, 2, special_handle_very_big_face);

		//mtcnn.TurnOnShowDebugInfo();
		static int fr_id = 0;
		if (landmark106)
		{
			if (!mtcnn.Find106(image0.data, image0.cols, image0.rows, image0.step[0], thirdBbox106))
			{
				thirdBbox106 = thirdBbox106_last;
			}
			if (thirdBbox106.size() == 0)
			{
				printf("%d\n", fr_id);
				thirdBbox106 = thirdBbox106_last;
			}
		}
		else
		{
			if (!mtcnn.Find(image0.data, image0.cols, image0.rows, image0.step[0], thirdBbox))
			{
				thirdBbox = thirdBbox_last;
			}
			if (thirdBbox.size() == 0)
			{
				printf("%d\n", fr_id);
				thirdBbox = thirdBbox_last;
			}
		}
		
		fr_id++;
		if (landmark106)
		{
			my_filter.filtering(thirdBbox106);
			Draw(image0, thirdBbox106);
			thirdBbox106_last = thirdBbox106;
		}
		else
		{
			Draw(image0, thirdBbox);
			thirdBbox_last = thirdBbox;
		}
		imshow("show", image0);
		writer << image0;
		int key = cv::waitKey(20);
		if (key == 27)
			break;
	}

	return EXIT_SUCCESS;
}

int main()
{
	//return run_fig();
	return run_cam();
}
