#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_MTCNN_old.h"
#include "ZQ_CNN_MTCNN.h"
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
				circle(image, cv::Point(*(it->ppoint + num * 2) + 0.5f, *(it->ppoint + num * 2 + 1) + 0.5f), 1, cv::Scalar(0, 255, 255), -1);
		}
		else
		{
			printf("not exist!\n");
		}
	}
}

int main(int argc, const char** argv)
{
	if (argc != 4)
	{
		printf("%s in_list out_list in_folder\n",argv[0]);
		return EXIT_FAILURE;
	}
	int num_threads = 1;
#if ZQ_CNN_USE_BLAS_GEMM
	printf("set openblas thread_num = %d\n", num_threads);
	openblas_set_num_threads(num_threads);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(num_threads);
#endif

	FILE* in = fopen(argv[1], "r");
	if (in == 0)
	{
		printf("failed to open %s\n", argv[1]);
		return EXIT_FAILURE;
	}
	FILE* out = fopen(argv[2], "w");
	if (out == 0)
	{
		printf("failed to create %s\n", argv[2]);
		fclose(in);
		return EXIT_FAILURE;
	}

	ZQ_CNN_MTCNN mtcnn;
	std::string result_name;
	mtcnn.TurnOffShowDebugInfo();
	//mtcnn.SetLimit(300, 50, 20);
	
	int thread_num = 0;
	bool special_handle_very_big_face = false;


	if (!mtcnn.Init("model/det1-dw20-plus.zqparams", "model/det1-dw20-plus.nchwbin",
		"model/det2-dw24-plus.zqparams", "model/det2-dw24-plus.nchwbin",
		"model/det3-dw48-plus.zqparams", "model/det3-dw48-plus.nchwbin",
		thread_num, false,
		"model/det4-dw48-v2n.zqparams", "model/det4-dw48-v2n.nchwbin"
	))
	{
		cout << "failed to init!\n";
		return EXIT_FAILURE;
	}
	
	char buf[512], filename[512];
	int im_id = 0;
	while (true)
	{
		buf[0] = '\0';
		fgets(buf, 512, in);
		if (buf[0] == '\0')
			break;
		sscanf(buf, "%s", filename);
		sprintf(buf, "%s\\%s", argv[3], filename);
		im_id++;
		if (im_id % 100 == 0)
		{
			printf("%d done\n", im_id);
		}
		Mat image0 = cv::imread(buf, 1);
		if (image0.empty())
		{
			printf("failed to load image %s\n", buf);
			fprintf(out, "%s 0\n", filename);
			continue;
		}


		if (image0.channels() == 1)
			cv::cvtColor(image0, image0, CV_GRAY2BGR);
		//cv::convertScaleAbs(image0, image0, 2.0);
		/* TIPS: when finding tiny faces for very big image, gaussian blur is very useful for Pnet*/
		bool run_blur = true;
		int kernel_size = 3, sigma = 2;
		if (image0.cols * image0.rows >= 2500 * 1600)
		{
			run_blur = false;
			kernel_size = 5;
			sigma = 3;
		}
		else if (image0.cols * image0.rows >= 1920 * 1080)
		{
			run_blur = false;
			kernel_size = 3;
			sigma = 2;
		}
		else
		{
			run_blur = false;
		}

		if (run_blur)
		{
			cv::GaussianBlur(image0, image0, cv::Size(kernel_size, kernel_size), sigma, sigma);
		}

		mtcnn.SetPara(image0.cols, image0.rows, 80, 0.5, 0.6, 0.8, 0.4, 0.5, 0.5, 0.709, 3, 20, 4, special_handle_very_big_face);

		std::vector<ZQ_CNN_BBox> thirdBbox;
		
		//mtcnn.TurnOnShowDebugInfo();

		if (!mtcnn.Find(image0.data, image0.cols, image0.rows, image0.step[0], thirdBbox))
		{
			cout << "failed to find face!\n";
			//return EXIT_FAILURE;
		}

		fprintf(out, "%s", filename);
		fprintf(out, " %d", thirdBbox.size());
		for (int j = 0; j < thirdBbox.size(); j++)
		{
			ZQ_CNN_BBox& cur_box = thirdBbox[j];
			fprintf(out, " %d %d %d %d", cur_box.col1, cur_box.row1, cur_box.col2, cur_box.row2);
		}
		fprintf(out, "\n");

	}

	fclose(in);
	fclose(out);
	
	return EXIT_SUCCESS;
}
