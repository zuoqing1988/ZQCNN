// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <omp.h>
#include "net.h"


int main(int argc, char** argv)
{
	ncnn::Net net;
	net.load_param("mobilefacenet.param");
	net.load_model("mobilefacenet.bin");
    const char* imagepath1 = "00_.jpg";
	const char* imagepath2 = "01_.jpg";
	cv::Mat img1 = cv::imread(imagepath1);
	cv::Mat img2 = cv::imread(imagepath2);
	ncnn::Mat in1 = ncnn::Mat::from_pixels_resize(img1.data, ncnn::Mat::PIXEL_BGR, img1.cols, img1.rows, 112, 112);
	ncnn::Mat in2 = ncnn::Mat::from_pixels_resize(img2.data, ncnn::Mat::PIXEL_BGR, img2.cols, img2.rows, 112, 112);

	const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
	const float norm_vals[3] = { 1.0 / 127.5,1.0 / 127.5,1.0 / 127.5 };
	//in1.substract_mean_normalize(mean_vals, norm_vals);
	//in2.substract_mean_normalize(mean_vals, norm_vals);

	ncnn::Mat out1, out2;
	printf("begin\n");
	
	
	int out_iters = 10;
	for (int out_it = 0; out_it < out_iters; out_it++)
	{
		int iters = 100;
		double t1 = omp_get_wtime();
		for (int i = 0; i < iters; i++)
		{
			ncnn::Extractor ex1 = net.create_extractor();
			ncnn::Extractor ex2 = net.create_extractor();
			ex1.set_light_mode(false);
			ex2.set_light_mode(false);
			ex1.set_num_threads(8);
			ex2.set_num_threads(8);
			ex1.input("data", in1);
			ex1.extract("fc1", out1);
			ex2.input("data", in2);
			ex2.extract("fc1", out2);
		}
		double t2 = omp_get_wtime();
		printf("[%d] cost %.3f s, 1 iter costs %.3f ms\n", iters * 2, t2 - t1, 1000*(t2 - t1) / 2.0 / iters);
	}
	int dim = 128;
	double sum = 0, len1 = 0, len2 = 0;
	float* data1 = (float*)out1.data;
	float* data2 = (float*)out2.data;
	for (int i = 0; i < dim; i++)
	{
		sum += data1[i]*data2[i];
		len1 += data1[i] * data1[i];
		len2 += data2[i] * data2[i];
	}
	printf("score = %f\n", sum / sqrt(len1*len2));
    return 0;
}

