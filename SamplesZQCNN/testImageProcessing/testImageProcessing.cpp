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

void transpose_c3(const unsigned char* src, int w, int h, int src_widthStep,
	unsigned char* dst, int widthStep);

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
	cv::Mat src_img, dst_img_cv, dst_img_my, trans_img_cv, trans_img_my;
	src_img = cv::imread(argv[1]);
	if (src_img.empty())
	{
		printf("failed to load image %s\n", argv[1]);
		return 0;
	}
	cv::resize(src_img, dst_img_my, cv::Size(dst_W, dst_H));
	cv::resize(src_img, trans_img_my, cv::Size(dst_H, dst_W));
	int nOutIters = 5;
	int nIters = 1000;
	for (int i = 0; i < nOutIters; i++)
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
		printf("cv      :%12.5f ms\n", (t2 - t1)*1e-3/nIters);
		printf("my      :%12.5f ms\n", (t3 - t2)*1e-3/nIters);
		printf("my_arm32:%12.5f ms\n", (t4 - t3)*1e-3/nIters);
#else
		printf("cv:%d\nmy:%d\n", t2 - t1, t3 - t2);
#endif
	}
	printf("test transpose...\n");
	for (int i = 0; i < nOutIters; i++)
	{
		clock_t t1 = clock();
		for (int i = 0; i < nIters; i++)
			cv::transpose(dst_img_cv, trans_img_cv);
		clock_t t2 = clock();
		for (int i = 0; i < nIters; i++)
			transpose_c3(dst_img_my.data, dst_img_my.cols, dst_img_my.rows, dst_img_my.step[0], trans_img_my.data, trans_img_my.step[0]);
		clock_t t3 = clock();
#if __ARM_NEON
		printf("cv      :%12.5f ms\n", (t2 - t1)*1e-3 / nIters);
		printf("my      :%12.5f ms\n", (t3 - t2)*1e-3 / nIters);
#else
		printf("cv:%d\nmy:%d\n", t2 - t1, t3 - t2);
#endif
	}

	cv::namedWindow("cv");
	cv::namedWindow("my");
	cv::imshow("cv", dst_img_cv);
	cv::imshow("my", dst_img_my);
	cv::namedWindow("cv_trans");
	cv::namedWindow("my_trans");
	cv::imshow("cv_trans", trans_img_cv);
	cv::imshow("my_trans", trans_img_my);
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
	unsigned char* cur_dst_ptr, *cur_pix_ptr;
	register float fx, fy;
	register int ix, iy, dx, dy;
	for (dx = 0; dx < w; dx++)
	{
		fx = (float)((dx + 0.5f) * scale_x - 0.5f);
		ix = fx + 0.5f;
		ix = ix < 0 ? 0 : ix;
		ix = ix >= srcw ? srcw - 1 : ix;
		coord_x[dx] = ix * 3;
	}

	cur_dst_ptr = dst;
	if (w % 8 == 0)
	{
		for (dy = 0; dy < h; dy++)
		{
			fy = (float)((dy + 0.5f) * scale_y - 0.5f);
			iy = fy + 0.5f;
			iy = iy < 0 ? 0 : iy;
			iy = iy >= srch ? srch - 1 : iy;
			cur_src_ptr = src + iy*src_widthStep;
			cur_pix_ptr = cur_dst_ptr;
			dx = 0;
			for (; dx < w; dx += 8)
			{
				ix = coord_x[dx];
				cur_pix_ptr[0] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 2];
				cur_pix_ptr[0] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 3];
				cur_pix_ptr[0] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 4];
				cur_pix_ptr[0] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 5];
				cur_pix_ptr[0] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 6];
				cur_pix_ptr[0] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 7];
				cur_pix_ptr[0] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
			}
			cur_dst_ptr += widthStep;
		}
	}
	else
	{
		for (dy = 0; dy < h; dy++)
		{
			fy = (float)((dy + 0.5f) * scale_y - 0.5f);
			iy = fy + 0.5f;
			iy = iy < 0 ? 0 : iy;
			iy = iy >= srch ? srch - 1 : iy;
			cur_src_ptr = src + iy*src_widthStep;
			cur_pix_ptr = cur_dst_ptr;
			dx = 0;
			for (; dx < w - 7; dx += 8)
			{
				ix = coord_x[dx];
				cur_pix_ptr[0] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 2];
				cur_pix_ptr[0] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 3];
				cur_pix_ptr[0] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 4];
				cur_pix_ptr[0] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 5];
				cur_pix_ptr[0] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 6];
				cur_pix_ptr[0] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 7];
				cur_pix_ptr[0] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
			}
			for (; dx < w; dx++)
			{
				ix = coord_x[dx];
				cur_pix_ptr[0] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
			}
			cur_dst_ptr += widthStep;
		}
	}
	free(coord_x);
}

void transpose_c3(const unsigned char* src, int w, int h, int src_widthStep,
	unsigned char* dst, int widthStep)
{
	const unsigned char* cur_src_c_ptr, *cur_src_pix_ptr;
	unsigned char* cur_dst_ptr, *cur_pix_ptr;
	register int dx, dy;
	if (h % 8 == 0)
	{
		cur_src_c_ptr = src;
		cur_dst_ptr = dst;
		for (dy = 0; dy < w; dy++, cur_src_c_ptr += 3,cur_dst_ptr += widthStep)
		{
			cur_src_pix_ptr = cur_src_c_ptr;
			cur_pix_ptr = cur_dst_ptr;
			for (dx = 0; dx < h; dx += 8)
			{
				cur_pix_ptr[0] = cur_src_pix_ptr[0];
				cur_pix_ptr[1] = cur_src_pix_ptr[1];
				cur_pix_ptr[2] = cur_src_pix_ptr[2];
				cur_pix_ptr += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr[0] = cur_src_pix_ptr[0];
				cur_pix_ptr[1] = cur_src_pix_ptr[1];
				cur_pix_ptr[2] = cur_src_pix_ptr[2];
				cur_pix_ptr += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr[0] = cur_src_pix_ptr[0];
				cur_pix_ptr[1] = cur_src_pix_ptr[1];
				cur_pix_ptr[2] = cur_src_pix_ptr[2];
				cur_pix_ptr += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr[0] = cur_src_pix_ptr[0];
				cur_pix_ptr[1] = cur_src_pix_ptr[1];
				cur_pix_ptr[2] = cur_src_pix_ptr[2];
				cur_pix_ptr += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr[0] = cur_src_pix_ptr[0];
				cur_pix_ptr[1] = cur_src_pix_ptr[1];
				cur_pix_ptr[2] = cur_src_pix_ptr[2];
				cur_pix_ptr += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr[0] = cur_src_pix_ptr[0];
				cur_pix_ptr[1] = cur_src_pix_ptr[1];
				cur_pix_ptr[2] = cur_src_pix_ptr[2];
				cur_pix_ptr += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr[0] = cur_src_pix_ptr[0];
				cur_pix_ptr[1] = cur_src_pix_ptr[1];
				cur_pix_ptr[2] = cur_src_pix_ptr[2];
				cur_pix_ptr += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr[0] = cur_src_pix_ptr[0];
				cur_pix_ptr[1] = cur_src_pix_ptr[1];
				cur_pix_ptr[2] = cur_src_pix_ptr[2];
				cur_pix_ptr += 3;
				cur_src_pix_ptr += src_widthStep;
			}
		}
	}
	else
	{
		cur_src_c_ptr = src;
		cur_dst_ptr = dst;
		for (dy = 0; dy < w; dy++, cur_src_c_ptr += 3, cur_dst_ptr += widthStep)
		{
			cur_src_pix_ptr = cur_src_c_ptr;
			cur_pix_ptr = cur_dst_ptr;
			for (dx = 0; dx < h; dx += 8)
			{
				cur_pix_ptr[0] = cur_src_pix_ptr[0];
				cur_pix_ptr[1] = cur_src_pix_ptr[1];
				cur_pix_ptr[2] = cur_src_pix_ptr[2];
				cur_pix_ptr += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr[0] = cur_src_pix_ptr[0];
				cur_pix_ptr[1] = cur_src_pix_ptr[1];
				cur_pix_ptr[2] = cur_src_pix_ptr[2];
				cur_pix_ptr += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr[0] = cur_src_pix_ptr[0];
				cur_pix_ptr[1] = cur_src_pix_ptr[1];
				cur_pix_ptr[2] = cur_src_pix_ptr[2];
				cur_pix_ptr += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr[0] = cur_src_pix_ptr[0];
				cur_pix_ptr[1] = cur_src_pix_ptr[1];
				cur_pix_ptr[2] = cur_src_pix_ptr[2];
				cur_pix_ptr += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr[0] = cur_src_pix_ptr[0];
				cur_pix_ptr[1] = cur_src_pix_ptr[1];
				cur_pix_ptr[2] = cur_src_pix_ptr[2];
				cur_pix_ptr += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr[0] = cur_src_pix_ptr[0];
				cur_pix_ptr[1] = cur_src_pix_ptr[1];
				cur_pix_ptr[2] = cur_src_pix_ptr[2];
				cur_pix_ptr += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr[0] = cur_src_pix_ptr[0];
				cur_pix_ptr[1] = cur_src_pix_ptr[1];
				cur_pix_ptr[2] = cur_src_pix_ptr[2];
				cur_pix_ptr += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr[0] = cur_src_pix_ptr[0];
				cur_pix_ptr[1] = cur_src_pix_ptr[1];
				cur_pix_ptr[2] = cur_src_pix_ptr[2];
				cur_pix_ptr += 3;
				cur_src_pix_ptr += src_widthStep;
			}
			for (; dx < h; dx ++)
			{
				cur_pix_ptr[0] = cur_src_pix_ptr[0];
				cur_pix_ptr[1] = cur_src_pix_ptr[1];
				cur_pix_ptr[2] = cur_src_pix_ptr[2];
				cur_pix_ptr += 3;
				cur_src_pix_ptr += src_widthStep;
			}
		}
	}
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
		cur_dst_ptr0 += widthStep4;
		cur_dst_ptr1 += widthStep4;
		cur_dst_ptr2 += widthStep4;
		cur_dst_ptr3 += widthStep4;
	}

	for (; dy < h; dy ++)
	{
		fy = (float)((dy + 0.5f) * scale_y - 0.5f);
		iy = fy + 0.5f;
		iy = iy < 0 ? 0 : iy;
		iy = iy >= srch ? srch - 1 : iy;
		cur_src_ptr0 = src + iy*src_widthStep;
		dx = 0;
		cur_pix_ptr0 = cur_dst_ptr0;
		for (; dx < w - 16; dx += 8)
		{
			ix = coord_x[dx];
			vst1_u8(cur_pix_ptr0, vld1_u8(cur_src_ptr0 + ix));
			cur_pix_ptr0 += 3;
			ix = coord_x[dx + 1];
			vst1_u8(cur_pix_ptr0, vld1_u8(cur_src_ptr0 + ix));
			cur_pix_ptr0 += 3;
			ix = coord_x[dx + 2];
			vst1_u8(cur_pix_ptr0, vld1_u8(cur_src_ptr0 + ix));
			cur_pix_ptr0 += 3;
			ix = coord_x[dx + 3];
			vst1_u8(cur_pix_ptr0, vld1_u8(cur_src_ptr0 + ix));
			cur_pix_ptr0 += 3;
			ix = coord_x[dx + 4];
			vst1_u8(cur_pix_ptr0, vld1_u8(cur_src_ptr0 + ix));
			cur_pix_ptr0 += 3;
			ix = coord_x[dx + 5];
			vst1_u8(cur_pix_ptr0, vld1_u8(cur_src_ptr0 + ix));
			cur_pix_ptr0 += 3;
			ix = coord_x[dx + 6];
			vst1_u8(cur_pix_ptr0, vld1_u8(cur_src_ptr0 + ix));
			cur_pix_ptr0 += 3;
			ix = coord_x[dx + 7];
			vst1_u8(cur_pix_ptr0, vld1_u8(cur_src_ptr0 + ix));
			cur_pix_ptr0 += 3;
		}
		for (; dx < w; dx++)
		{
			ix = coord_x[dx];
			cur_pix_ptr0[0] = cur_src_ptr0[ix];
			cur_pix_ptr0[1] = cur_src_ptr0[ix + 1];
			cur_pix_ptr0[2] = cur_src_ptr0[ix + 2];
			cur_pix_ptr0 += 3;
		}
		cur_dst_ptr0 += widthStep;
	}
	free(coord_x);
}

#endif
