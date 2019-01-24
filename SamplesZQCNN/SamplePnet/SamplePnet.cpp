#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_NSFW.h"
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
	int num_threads = 1;

#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(num_threads);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(num_threads);
#endif

	ZQ_CNN_Net net;
	int stride = 4;
	int cell_size = 20;
	if (!net.LoadFrom("model/det1-dw20-v1.zqparams", "model/det1-dw20-v1-16.nchwbin"))
	//if (!net.LoadFrom("model/det1.zqparams", "model/det1_bgr.nchwbin"))
	{
		cout << "failed to load model\n";
		return EXIT_FAILURE;
	}
	printf("num_MulAdd = %.1f M\n", net.GetNumOfMulAdd() / (1024.0*1024.0));

	Mat img = imread("data/4.jpg");
	if (img.empty())
	{
		cout << "failed to load image\n";
		return EXIT_FAILURE;
	}
	Mat draw_img;
	img.copyTo(draw_img);
	for (int h = 0; h < draw_img.rows; h++)
	{
		for (int w = 0; w < draw_img.cols; w++)
		{
			draw_img.data[h*draw_img.step[0] + w * 3 + 0] *= 0.2;
			draw_img.data[h*draw_img.step[0] + w * 3 + 1] *= 0.2;
			draw_img.data[h*draw_img.step[0] + w * 3 + 2] *= 0.2;
		}
	}
	float scale = 1.0f;
	float factor = 0.709f;
	for (int it = 0; it < 9; it++)
	{
		float cur_scale = scale*pow(factor, it);
		Mat scale_img;
		resize(img, scale_img, Size(), cur_scale, cur_scale);
		if (scale_img.cols < cell_size || scale_img.rows < cell_size)
			break;
		ZQ_CNN_Tensor4D_NHW_C_Align128bit input;
		input.ConvertFromBGR(scale_img.data, scale_img.cols, scale_img.rows, scale_img.step[0]);
		if (!net.Forward(input))
		{
			cout << "failed to forward " << it << "\n";
			continue;
		}

		const ZQ_CNN_Tensor4D* prob = net.GetBlobByName("prob1");
		if (prob == 0)
		{
			cout << "failed to get blob prob\n";
			return EXIT_FAILURE;
		}
		const float* prob_data = prob->GetFirstPixelPtr();
		int prob_width = prob->GetW();
		int prob_height = prob->GetH();
		int prob_widthStep = prob->GetWidthStep();
		int prob_pixStep = prob->GetPixelStep();
		Mat prob_img(Size(scale_img.cols, scale_img.rows), CV_MAKETYPE(8, 3), Scalar(0));
		for (int h = 0; h < prob_height; h++)
		{
			for (int w = 0; w < prob_width; w++)
			{
				uchar cur_prob = __max(0, __min(255, 255.0f*prob_data[h*prob_widthStep + w*prob_pixStep + 1]));
				if (cur_prob < (255*0.7))
					cur_prob = 0;
				int start_hh = h * stride;
				int end_hh = __min(start_hh + cell_size, scale_img.rows);
				int start_ww = w * stride;
				int end_ww = __min(start_ww + cell_size, scale_img.cols);
				for (int hh = start_hh; hh < end_hh; hh++)
				{
					for (int ww = start_ww; ww < end_ww; ww++)
					{
						uchar* dst_ptr = prob_img.data + hh*prob_img.step[0] + ww * 3 + 2;
						*dst_ptr = __max(*dst_ptr, cur_prob);
					}
				}
			}
		}
		resize(prob_img, prob_img, cv::Size(img.cols, img.rows), 0, 0, CV_INTER_CUBIC);

		for (int h = 0; h < img.rows; h++)
		{
			for (int w = 0; w < img.cols; w++)
			{
				uchar* dst_ptr = draw_img.data + h*img.step[0] + w * 3 + 2;
				uchar* cur_ptr = prob_img.data + h*prob_img.step[0] + w * 3 + 2;
				*dst_ptr = __max(*dst_ptr, *cur_ptr);
			}
		}
	}

	namedWindow("prob");
	while (draw_img.cols > 1920 || draw_img.rows > 1080)
		cv::resize(draw_img, draw_img, cv::Size(), 0.5, 0.5);
	imshow("prob", draw_img);
	waitKey(0);
	return EXIT_SUCCESS;
}