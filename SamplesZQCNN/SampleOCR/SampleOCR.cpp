#include "ZQ_CNN_Net.h"
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
	if(!net.LoadFrom("model/ocr.zqparams","model/ocr.nchwbin"))
	{
		printf("failed to init net\n");
		return EXIT_FAILURE;
	}

	//Mat image = cv::imread("data/2013-12-09.jpg", 0);
	Mat image = cv::imread("data/LSVNP41Z7B2731969.jpg", 0);
	//Mat image = cv::imread("data/LFV2A2156B3533072.jpg", 0);
	if (image.empty())
	{
		cout << "empty image\n";
		return EXIT_FAILURE;
	}
	int ori_width = image.cols;
	int ori_height = image.rows;
	int dst_height = 32;
	int dst_width = (float)ori_width / ori_height*dst_height;
	cv::resize(image, image, cv::Size(dst_width, dst_height));
	ZQ_CNN_Tensor4D_NHW_C_Align128bit input;
	input.ConvertFromGray(image.data, dst_width, dst_height, image.step[0],0,1/255.0);
	if (!net.Forward(input))
	{
		printf("failed to run net\n");
		return EXIT_FAILURE;
	}
	const ZQ_CNN_Tensor4D* ptr = net.GetBlobByName("11_107");
	int N = ptr->GetN();
	int H = ptr->GetH();
	int W = ptr->GetW();
	int C = ptr->GetC();
	printf("N=%d,H=%d,W=%d,C=%d\n", N, H, W, C);
	const float* pixel_ptr = ptr->GetFirstPixelPtr();
	int PixelStep = ptr->GetPixelStep();
	std::vector<int> ids;
	for (int w = 0; w < W; w++)
	{
		float max_val = -FLT_MAX;
		int id = -1;
		for (int c = 0; c < C; c++)
		{
			if (max_val < pixel_ptr[w*PixelStep + c])
			{
				id = c;
				max_val = pixel_ptr[w*PixelStep + c];
			}
		}
		ids.push_back(id);
	}
	
	int blank = 11315;
	std::map<int, char> map_id_to_char;
	map_id_to_char.insert(std::make_pair(14, '-'));
	for (int i = 0; i < 10; i++)
		map_id_to_char.insert(std::make_pair(17 + i, '0' + i));
	for (int i = 0; i < 26; i++)
		map_id_to_char.insert(std::make_pair(34 + i, 'A' + i));
	for (int i = 0; i < 26; i++)
		map_id_to_char.insert(std::make_pair(68 + i, 'a' + i));
	
	int last_id = blank;
	std::vector<int> final_ids;
	for (int i = 0; i < ids.size(); i++)
	{
		//printf("%d ", ids[i]);
		if (ids[i] != blank)
		{
			if (ids[i] != last_id)
			{
				
				final_ids.push_back(ids[i]);
			}
		}
		last_id = ids[i];
	}

	const std::map<int, char>::iterator end_iter = map_id_to_char.end();
	for (int i = 0; i < final_ids.size(); i++)
	{
		if (map_id_to_char.find(final_ids[i]) != end_iter)
			putchar(map_id_to_char[final_ids[i]]);
		else
			putchar('?');
	}
	printf("\n");

	
	return EXIT_SUCCESS;
}
