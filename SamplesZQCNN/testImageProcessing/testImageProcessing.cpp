#include "ZQ_CNN_CompileConfig.h"
#include <stdio.h>
#include <malloc.h>
#include <opencv2/opencv.hpp>
#include <time.h>
#if __ARM_NEON
#include <arm_neon.h>
#endif
void resize_nn_c1(const unsigned char* src, int srcw, int srch, int src_widthStep,
	unsigned char* dst, int w, int h, int widthStep);

void resize_nn_c2(const unsigned char* src, int srcw, int srch, int src_widthStep,
	unsigned char* dst, int w, int h, int widthStep);

void resize_nn_c3(const unsigned char* src, int srcw, int srch, int src_widthStep,
	unsigned char* dst, int w, int h, int widthStep);

#if __ARM_NEON
void resize_nn_c3_arm32(const unsigned char* src, int srcw, int srch, int src_widthStep,
	unsigned char* dst, int w, int h, int widthStep);
#endif

int main(int argc, const char** argv)
{
	if (argc < 2)
	{
		printf("%s img_name\n", argv[0]);
		return 0;
	}

	int dst_W = 640, dst_H = 360;
	cv::Mat src_img, dst_img_cv, dst_img_my;
	src_img = cv::imread(argv[1]);
	if (src_img.empty())
	{
		printf("failed to load image %s\n", argv[1]);
		return 0;
	}
	cv::resize(src_img, dst_img_my, cv::Size(dst_W, dst_H));

	int nIters = 1000;
	for (int i = 0; i < 10; i++)
	{
		clock_t t1 = clock();
		for (int i = 0; i < nIters; i++)
			cv::resize(src_img, dst_img_cv, cv::Size(dst_W, dst_H));
		clock_t t2 = clock();
		for (int i = 0; i < nIters; i++)
			resize_nn_c3(src_img.data, src_img.cols, src_img.rows, src_img.step[0], dst_img_my.data, dst_W, dst_H, dst_img_my.step[0]);
		clock_t t3 = clock();
#if __ARM_NEON
		for (int i = 0; i < nIters; i++)
			resize_nn_c3_arm32(src_img.data, src_img.cols, src_img.rows, src_img.step[0], dst_img_my.data, dst_W, dst_H, dst_img_my.step[0]);
		clock_t t4 = clock();
		printf("cv      :%12.5f ms\n", (t2 - t1)*1e-6);
		printf("my      :%12.5f ms\n", (t3 - t2)*1e-6);
		printf("my_arm32:%12.5f ms\n", (t4 - t3)*1e-6);
#else
		printf("cv:%d\nmy:%d\n", t2 - t1, t3 - t2);
#endif
	}

	cv::namedWindow("cv");
	cv::namedWindow("my");
	cv::imshow("cv", dst_img_cv);
	cv::imshow("my", dst_img_my);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}

void resize_nn_c1(const unsigned char* src, int srcw, int srch, int src_widthStep,
	unsigned char* dst, int w, int h, int widthStep)
{
	double scale_x = (double)srcw / w;
	double scale_y = (double)srch / h;
	int* coord_x = (int*)malloc(w * sizeof(int));
	const unsigned char* cur_src_ptr;
	unsigned char* cur_dst_ptr;
	float fx, fy;
	int ix, iy;
	for (int dx = 0; dx < w; dx++)
	{
		fx = (float)((dx + 0.5f) * scale_x - 0.5f);
		ix = fx + 0.5f;
		ix = ix < 0 ? 0 : ix;
		ix = ix >= srcw ? srcw - 1 : ix;
		coord_x[dx] = ix;
	}

	cur_dst_ptr = dst;
	for (int dy = 0; dy < h; dy++)
	{
		fy = (float)((dy + 0.5f) * scale_y - 0.5f);
		iy = fy + 0.5f;
		iy = iy < 0 ? 0 : iy;
		iy = iy >= srch ? srch - 1 : iy;
		cur_src_ptr = src + iy*src_widthStep;
		for (int dx = 0; dx < w; dx++)
		{
			ix = coord_x[dx];
			cur_dst_ptr[dx] = cur_src_ptr[ix];
		}
		cur_dst_ptr += widthStep;
	}
	free(coord_x);
}

void resize_nn_c2(const unsigned char* src, int srcw, int srch, int src_widthStep,
	unsigned char* dst, int w, int h, int widthStep)
{
	double scale_x = (double)srcw / w;
	double scale_y = (double)srch / h;
	int* coord_x = (int*)malloc(w * sizeof(int));
	const unsigned char* cur_src_ptr;
	unsigned char* cur_dst_ptr;
	float fx, fy;
	int ix, iy;
	for (int dx = 0; dx < w; dx++)
	{
		fx = (float)((dx + 0.5f) * scale_x - 0.5f);
		ix = fx + 0.5f;
		ix = ix < 0 ? 0 : ix;
		ix = ix >= srcw ? srcw - 1 : ix;
		coord_x[dx] = ix * 2;
	}

	cur_dst_ptr = dst;
	for (int dy = 0; dy < h; dy++)
	{
		fy = (float)((dy + 0.5f) * scale_y - 0.5f);
		iy = fy + 0.5f;
		iy = iy < 0 ? 0 : iy;
		iy = iy >= srch ? srch - 1 : iy;
		cur_src_ptr = src + iy*src_widthStep;
		for (int dx = 0; dx < w; dx++)
		{
			ix = coord_x[dx];
			cur_dst_ptr[dx * 2] = cur_src_ptr[ix];
			cur_dst_ptr[dx * 2 + 1] = cur_src_ptr[ix + 1];
		}
		cur_dst_ptr += widthStep;
	}
	free(coord_x);
}

void resize_nn_c3(const unsigned char* src, int srcw, int srch, int src_widthStep,
	unsigned char* dst, int w, int h, int widthStep)
{
	double scale_x = (double)srcw / w;
	double scale_y = (double)srch / h;
	int* coord_x = (int*)malloc(w * sizeof(int));
	const unsigned char* cur_src_ptr;
	unsigned char* cur_dst_ptr;
	float fx, fy;
	int ix, iy;
	for (int dx = 0; dx < w; dx++)
	{
		fx = (float)((dx + 0.5f) * scale_x - 0.5f);
		ix = fx + 0.5f;
		ix = ix < 0 ? 0 : ix;
		ix = ix >= srcw ? srcw - 1 : ix;
		coord_x[dx] = ix * 3;
	}

	cur_dst_ptr = dst;
	for (int dy = 0; dy < h; dy++)
	{
		fy = (float)((dy + 0.5f) * scale_y - 0.5f);
		iy = fy + 0.5f;
		iy = iy < 0 ? 0 : iy;
		iy = iy >= srch ? srch - 1 : iy;
		cur_src_ptr = src + iy*src_widthStep;
		for (int dx = 0; dx < w; dx++)
		{
			ix = coord_x[dx];
			cur_dst_ptr[dx * 3] = cur_src_ptr[ix];
			cur_dst_ptr[dx * 3 + 1] = cur_src_ptr[ix + 1];
			cur_dst_ptr[dx * 3 + 2] = cur_src_ptr[ix + 2];
		}
		cur_dst_ptr += widthStep;
	}
	free(coord_x);
}

#if __ARM_NEON

void resize_nn_c3_arm32(const unsigned char* src, int srcw, int srch, int src_widthStep,
	unsigned char* dst, int w, int h, int widthStep)
{
	double scale_x = (double)srcw / w;
	double scale_y = (double)srch / h;
	int* coord_x = (int*)malloc(w * sizeof(int));
	const unsigned char* cur_src_ptr0, *cur_src_ptr1, *cur_src_ptr2, *cur_src_ptr3;
	unsigned char* cur_dst_ptr0, *cur_dst_ptr1, *cur_dst_ptr2, *cur_dst_ptr3;
	unsigned char* cur_pix_ptr0, *cur_pix_ptr1, *cur_pix_ptr2, *cur_pix_ptr3;
	int widthStep2 = widthStep << 1;
	int widthStep3 = widthStep2 + widthStep;
	int widthStep4 = widthStep << 2;
	float fx, fy;
	int ix, iy, dx, dy;
	for (int dx = 0; dx < w; dx++)
	{
		fx = (float)((dx + 0.5f) * scale_x - 0.5f);
		ix = fx + 0.5f;
		ix = ix < 0 ? 0 : ix;
		ix = ix >= srcw ? srcw - 1 : ix;
		coord_x[dx] = ix * 3;
	}

	cur_dst_ptr0 = dst;
	cur_dst_ptr1 = cur_dst_ptr0 + widthStep;
	cur_dst_ptr2 = cur_dst_ptr1 + widthStep;
	cur_dst_ptr3 = cur_dst_ptr2 + widthStep;
	dy = 0;
	for (; dy < h-3; dy+=4)
	{
		fy = (float)((dy + 0.5f) * scale_y - 0.5f);
		iy = fy + 0.5f;
		iy = iy < 0 ? 0 : iy;
		iy = iy >= srch ? srch - 1 : iy;
		cur_src_ptr0 = src + iy*src_widthStep;
		fy = (float)((dy + 1.5f) * scale_y - 0.5f);
		iy = fy + 0.5f;
		iy = iy < 0 ? 0 : iy;
		iy = iy >= srch ? srch - 1 : iy;
		cur_src_ptr1 = src + iy*src_widthStep;
		fy = (float)((dy + 2.5f) * scale_y - 0.5f);
		iy = fy + 0.5f;
		iy = iy < 0 ? 0 : iy;
		iy = iy >= srch ? srch - 1 : iy;
		cur_src_ptr2 = src + iy*src_widthStep;
		fy = (float)((dy + 3.5f) * scale_y - 0.5f);
		iy = fy + 0.5f;
		iy = iy < 0 ? 0 : iy;
		iy = iy >= srch ? srch - 1 : iy;
		cur_src_ptr3 = src + iy*src_widthStep;
		dx = 0;
		cur_pix_ptr0 = cur_dst_ptr0;
		cur_pix_ptr1 = cur_dst_ptr1;
		cur_pix_ptr2 = cur_dst_ptr2;
		cur_pix_ptr3 = cur_dst_ptr3;
		for (; dx < w - 16; dx += 8)
		{
			ix = coord_x[dx];
			vst1_u8(cur_pix_ptr0, vld1_u8(cur_src_ptr0 + ix));
			vst1_u8(cur_pix_ptr1, vld1_u8(cur_src_ptr1 + ix));
			vst1_u8(cur_pix_ptr2, vld1_u8(cur_src_ptr2 + ix));
			vst1_u8(cur_pix_ptr3, vld1_u8(cur_src_ptr3 + ix));
			cur_pix_ptr0 += 3;
			cur_pix_ptr1 += 3;
			cur_pix_ptr2 += 3;
			cur_pix_ptr3 += 3;
			ix = coord_x[dx + 1];
			vst1_u8(cur_pix_ptr0, vld1_u8(cur_src_ptr0 + ix));
			vst1_u8(cur_pix_ptr1, vld1_u8(cur_src_ptr1 + ix));
			vst1_u8(cur_pix_ptr2, vld1_u8(cur_src_ptr2 + ix));
			vst1_u8(cur_pix_ptr3, vld1_u8(cur_src_ptr3 + ix));
			cur_pix_ptr0 += 3;
			cur_pix_ptr1 += 3;
			cur_pix_ptr2 += 3;
			cur_pix_ptr3 += 3;
			ix = coord_x[dx + 2];
			vst1_u8(cur_pix_ptr0, vld1_u8(cur_src_ptr0 + ix));
			vst1_u8(cur_pix_ptr1, vld1_u8(cur_src_ptr1 + ix));
			vst1_u8(cur_pix_ptr2, vld1_u8(cur_src_ptr2 + ix));
			vst1_u8(cur_pix_ptr3, vld1_u8(cur_src_ptr3 + ix));
			cur_pix_ptr0 += 3;
			cur_pix_ptr1 += 3;
			cur_pix_ptr2 += 3;
			cur_pix_ptr3 += 3;
			ix = coord_x[dx + 3];
			vst1_u8(cur_pix_ptr0, vld1_u8(cur_src_ptr0 + ix));
			vst1_u8(cur_pix_ptr1, vld1_u8(cur_src_ptr1 + ix));
			vst1_u8(cur_pix_ptr2, vld1_u8(cur_src_ptr2 + ix));
			vst1_u8(cur_pix_ptr3, vld1_u8(cur_src_ptr3 + ix));
			cur_pix_ptr0 += 3;
			cur_pix_ptr1 += 3;
			cur_pix_ptr2 += 3;
			cur_pix_ptr3 += 3;
			ix = coord_x[dx + 4];
			vst1_u8(cur_pix_ptr0, vld1_u8(cur_src_ptr0 + ix));
			vst1_u8(cur_pix_ptr1, vld1_u8(cur_src_ptr1 + ix));
			vst1_u8(cur_pix_ptr2, vld1_u8(cur_src_ptr2 + ix));
			vst1_u8(cur_pix_ptr3, vld1_u8(cur_src_ptr3 + ix));
			cur_pix_ptr0 += 3;
			cur_pix_ptr1 += 3;
			cur_pix_ptr2 += 3;
			cur_pix_ptr3 += 3;
			ix = coord_x[dx + 5];
			vst1_u8(cur_pix_ptr0, vld1_u8(cur_src_ptr0 + ix));
			vst1_u8(cur_pix_ptr1, vld1_u8(cur_src_ptr1 + ix));
			vst1_u8(cur_pix_ptr2, vld1_u8(cur_src_ptr2 + ix));
			vst1_u8(cur_pix_ptr3, vld1_u8(cur_src_ptr3 + ix));
			cur_pix_ptr0 += 3;
			cur_pix_ptr1 += 3;
			cur_pix_ptr2 += 3;
			cur_pix_ptr3 += 3;
			ix = coord_x[dx + 6];
			vst1_u8(cur_pix_ptr0, vld1_u8(cur_src_ptr0 + ix));
			vst1_u8(cur_pix_ptr1, vld1_u8(cur_src_ptr1 + ix));
			vst1_u8(cur_pix_ptr2, vld1_u8(cur_src_ptr2 + ix));
			vst1_u8(cur_pix_ptr3, vld1_u8(cur_src_ptr3 + ix));
			cur_pix_ptr0 += 3;
			cur_pix_ptr1 += 3;
			cur_pix_ptr2 += 3;
			cur_pix_ptr3 += 3;
			ix = coord_x[dx + 7];
			vst1_u8(cur_pix_ptr0, vld1_u8(cur_src_ptr0 + ix));
			vst1_u8(cur_pix_ptr1, vld1_u8(cur_src_ptr1 + ix));
			vst1_u8(cur_pix_ptr2, vld1_u8(cur_src_ptr2 + ix));
			vst1_u8(cur_pix_ptr3, vld1_u8(cur_src_ptr3 + ix));
			cur_pix_ptr0 += 3;
			cur_pix_ptr1 += 3;
			cur_pix_ptr2 += 3;
			cur_pix_ptr3 += 3;
		}
		for (; dx < w; dx++)
		{
			ix = coord_x[dx];
			cur_pix_ptr0[0] = cur_src_ptr0[ix];
			cur_pix_ptr1[0] = cur_src_ptr1[ix];
			cur_pix_ptr2[0] = cur_src_ptr2[ix];
			cur_pix_ptr3[0] = cur_src_ptr3[ix];
			cur_pix_ptr0[1] = cur_src_ptr0[ix + 1];
			cur_pix_ptr1[1] = cur_src_ptr1[ix + 1];
			cur_pix_ptr2[1] = cur_src_ptr2[ix + 1];
			cur_pix_ptr3[1] = cur_src_ptr3[ix + 1];
			cur_pix_ptr0[2] = cur_src_ptr0[ix + 2];
			cur_pix_ptr1[2] = cur_src_ptr1[ix + 2];
			cur_pix_ptr2[2] = cur_src_ptr2[ix + 2];
			cur_pix_ptr3[2] = cur_src_ptr3[ix + 2];
			cur_pix_ptr0 += 3;
			cur_pix_ptr1 += 3;
			cur_pix_ptr2 += 3;
			cur_pix_ptr3 += 3;
		}
		cur_dst_ptr0 += widthStep2;
		cur_dst_ptr1 += widthStep2;
		cur_dst_ptr2 += widthStep2;
		cur_dst_ptr3 += widthStep2;
	}

	for (; dy < h; dy ++)
	{
		fy = (float)((dy + 0.5f) * scale_y - 0.5f);
		iy1 = fy + 0.5f;
		iy1 = iy1 < 0 ? 0 : iy1;
		iy1 = iy1 >= srch ? srch - 1 : iy1;
		cur_src_ptr1 = src + iy1*src_widthStep;
		dx = 0;
		cur_pix_ptr1 = cur_dst_ptr1;
		for (; dx < w - 16; dx += 8)
		{
			ix = coord_x[dx];
			vst1_u8(cur_pix_ptr1, vld1_u8(cur_src_ptr1 + ix));
			cur_pix_ptr1 += 3;
			ix = coord_x[dx + 1];
			vst1_u8(cur_pix_ptr1, vld1_u8(cur_src_ptr1 + ix));
			cur_pix_ptr1 += 3;
			ix = coord_x[dx + 2];
			vst1_u8(cur_pix_ptr1, vld1_u8(cur_src_ptr1 + ix));
			cur_pix_ptr1 += 3;
			ix = coord_x[dx + 3];
			vst1_u8(cur_pix_ptr1, vld1_u8(cur_src_ptr1 + ix));
			cur_pix_ptr1 += 3;
			ix = coord_x[dx + 4];
			vst1_u8(cur_pix_ptr1, vld1_u8(cur_src_ptr1 + ix));
			cur_pix_ptr1 += 3;
			ix = coord_x[dx + 5];
			vst1_u8(cur_pix_ptr1, vld1_u8(cur_src_ptr1 + ix));
			cur_pix_ptr1 += 3;
			ix = coord_x[dx + 6];
			vst1_u8(cur_pix_ptr1, vld1_u8(cur_src_ptr1 + ix));
			cur_pix_ptr1 += 3;
			ix = coord_x[dx + 7];
			vst1_u8(cur_pix_ptr1, vld1_u8(cur_src_ptr1 + ix));
			cur_pix_ptr1 += 3;
		}
		for (; dx < w; dx++)
		{
			ix = coord_x[dx];
			cur_pix_ptr1[0] = cur_src_ptr1[ix];
			cur_pix_ptr1[1] = cur_src_ptr1[ix + 1];
			cur_pix_ptr1[2] = cur_src_ptr1[ix + 2];
			cur_pix_ptr1 += 3;
		}
		cur_dst_ptr1 += widthStep;
	}
	free(coord_x);
}

#endif
