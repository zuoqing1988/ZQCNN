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

void bgr2bgra(const unsigned char* src, int srcw, int srch, int src_widthStep,
	unsigned char* dst, int widthStep);

void resize_nn_bgra2rgb(const unsigned char* src, int srcw, int srch, int src_widthStep,
	unsigned char* dst, int w, int h, int widthStep);

void transpose_c3(const unsigned char* src, int w, int h, int src_widthStep,
	unsigned char* dst, int widthStep);

void transpose_c3_h4w4(const unsigned char* src, int w, int h, int src_widthStep,
	unsigned char* dst, int widthStep);

void transpose_c3_h8w8(const unsigned char* src, int w, int h, int src_widthStep,
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
	cv::Mat src_img, dst_img_cv, dst_img_my, dst_img_my_rgb, trans_img_cv, trans_img_my, bgra_img;
	src_img = cv::imread(argv[1]);
	if (src_img.empty())
	{
		printf("failed to load image %s\n", argv[1]);
		return 0;
	}
	cv::resize(src_img, dst_img_my, cv::Size(dst_W, dst_H));
	cv::resize(src_img, dst_img_my_rgb, cv::Size(dst_W, dst_H));
	cv::resize(src_img, trans_img_my, cv::Size(dst_H, dst_W));
	bgra_img = cv::Mat(src_img.rows, src_img.cols, CV_8UC4);
	bgr2bgra(src_img.data, src_img.cols, src_img.rows, src_img.step[0], bgra_img.data, bgra_img.step[0]);
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
		for (int i = 0; i < nIters; i++)
			resize_nn_bgra2rgb(bgra_img.data, bgra_img.cols, bgra_img.rows, bgra_img.step[0], dst_img_my_rgb.data, dst_W, dst_H, dst_img_my_rgb.step[0]);
		clock_t t4 = clock();
#if __ARM_NEON
		for (int i = 0; i < nIters; i++)
			resize_nn_c3_arm32(src_img.data, src_img.cols, src_img.rows, src_img.step[0], dst_img_my.data, dst_W, dst_H, dst_img_my.step[0]);
		clock_t t5 = clock();
		printf("cv         :%12.5f ms\n", (t2 - t1)*1e-3 / nIters);
		printf("my         :%12.5f ms\n", (t3 - t2)*1e-3 / nIters);
		printf("my_arm32   :%12.5f ms\n", (t5 - t4)*1e-3 / nIters);
		printf("my_bgra2rgb:%12.5f ms\n", (t4 - t3)*1e-3 / nIters);
#else
		printf("cv         :%12.5f ms\n", (t2 - t1)*1e-3 / nIters);
		printf("my         :%12.5f ms\n", (t3 - t2)*1e-3 / nIters);
		printf("my_bgra2rgb:%12.5f ms\n", (t4 - t3)*1e-3 / nIters);
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
		for (int i = 0; i < nIters; i++)
			transpose_c3_h4w4(dst_img_my.data, dst_img_my.cols, dst_img_my.rows, dst_img_my.step[0], trans_img_my.data, trans_img_my.step[0]);
		clock_t t4 = clock();
		for (int i = 0; i < nIters; i++)
			transpose_c3_h8w8(dst_img_my.data, dst_img_my.cols, dst_img_my.rows, dst_img_my.step[0], trans_img_my.data, trans_img_my.step[0]);
		clock_t t5 = clock();

		printf("cv      :%12.5f ms\n", (t2 - t1)*1e-3 / nIters);
		printf("my      :%12.5f ms\n", (t3 - t2)*1e-3 / nIters);
		printf("my_h4w4 :%12.5f ms\n", (t4 - t3)*1e-3 / nIters);
		printf("my_h8w8 :%12.5f ms\n", (t5 - t4)*1e-3 / nIters);
	}

	cv::namedWindow("cv");
	cv::namedWindow("my");
	cv::namedWindow("my_rgb");
	cv::imshow("cv", dst_img_cv);
	cv::imshow("my", dst_img_my);
	cv::imshow("my_rgb", dst_img_my_rgb);
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

void bgr2bgra(const unsigned char* src, int srcw, int srch, int src_widthStep,
	unsigned char* dst, int widthStep)
{
	const unsigned char* src_row_ptr, *src_pix_ptr;
	unsigned char* dst_row_ptr, *dst_pix_ptr;
	int i, j;
	if (srcw % 8 == 0)
	{
		src_row_ptr = src;
		dst_row_ptr = dst;
		for (i = 0; i < srch; i++, src_row_ptr += src_widthStep, dst_row_ptr += widthStep)
		{
			j = 0;
			src_pix_ptr = src_row_ptr;
			dst_pix_ptr = dst_row_ptr;
			for (; j < srcw; j += 8)
			{
				dst_pix_ptr[0] = src_pix_ptr[0];
				dst_pix_ptr[1] = src_pix_ptr[1];
				dst_pix_ptr[2] = src_pix_ptr[2];
				dst_pix_ptr += 4;
				src_pix_ptr += 3;
				dst_pix_ptr[0] = src_pix_ptr[0];
				dst_pix_ptr[1] = src_pix_ptr[1];
				dst_pix_ptr[2] = src_pix_ptr[2];
				dst_pix_ptr += 4;
				src_pix_ptr += 3;
				dst_pix_ptr[0] = src_pix_ptr[0];
				dst_pix_ptr[1] = src_pix_ptr[1];
				dst_pix_ptr[2] = src_pix_ptr[2];
				dst_pix_ptr += 4;
				src_pix_ptr += 3;
				dst_pix_ptr[0] = src_pix_ptr[0];
				dst_pix_ptr[1] = src_pix_ptr[1];
				dst_pix_ptr[2] = src_pix_ptr[2];
				dst_pix_ptr += 4;
				src_pix_ptr += 3;
				dst_pix_ptr[0] = src_pix_ptr[0];
				dst_pix_ptr[1] = src_pix_ptr[1];
				dst_pix_ptr[2] = src_pix_ptr[2];
				dst_pix_ptr += 4;
				src_pix_ptr += 3;
				dst_pix_ptr[0] = src_pix_ptr[0];
				dst_pix_ptr[1] = src_pix_ptr[1];
				dst_pix_ptr[2] = src_pix_ptr[2];
				dst_pix_ptr += 4;
				src_pix_ptr += 3;
				dst_pix_ptr[0] = src_pix_ptr[0];
				dst_pix_ptr[1] = src_pix_ptr[1];
				dst_pix_ptr[2] = src_pix_ptr[2];
				dst_pix_ptr += 4;
				src_pix_ptr += 3;
				dst_pix_ptr[0] = src_pix_ptr[0];
				dst_pix_ptr[1] = src_pix_ptr[1];
				dst_pix_ptr[2] = src_pix_ptr[2];
				dst_pix_ptr += 4;
				src_pix_ptr += 3;
			}
		}
	}
	else
	{
		src_row_ptr = src;
		dst_row_ptr = dst;
		for (i = 0; i < srch; i++, src_row_ptr += src_widthStep, dst_row_ptr += widthStep)
		{
			j = 0;
			src_pix_ptr = src_row_ptr;
			dst_pix_ptr = dst_row_ptr;
			for (; j < srcw - 7; j += 8)
			{
				dst_pix_ptr[0] = src_pix_ptr[0];
				dst_pix_ptr[1] = src_pix_ptr[1];
				dst_pix_ptr[2] = src_pix_ptr[2];
				dst_pix_ptr += 4;
				src_pix_ptr += 3;
				dst_pix_ptr[0] = src_pix_ptr[0];
				dst_pix_ptr[1] = src_pix_ptr[1];
				dst_pix_ptr[2] = src_pix_ptr[2];
				dst_pix_ptr += 4;
				src_pix_ptr += 3;
				dst_pix_ptr[0] = src_pix_ptr[0];
				dst_pix_ptr[1] = src_pix_ptr[1];
				dst_pix_ptr[2] = src_pix_ptr[2];
				dst_pix_ptr += 4;
				src_pix_ptr += 3;
				dst_pix_ptr[0] = src_pix_ptr[0];
				dst_pix_ptr[1] = src_pix_ptr[1];
				dst_pix_ptr[2] = src_pix_ptr[2];
				dst_pix_ptr += 4;
				src_pix_ptr += 3;
				dst_pix_ptr[0] = src_pix_ptr[0];
				dst_pix_ptr[1] = src_pix_ptr[1];
				dst_pix_ptr[2] = src_pix_ptr[2];
				dst_pix_ptr += 4;
				src_pix_ptr += 3;
				dst_pix_ptr[0] = src_pix_ptr[0];
				dst_pix_ptr[1] = src_pix_ptr[1];
				dst_pix_ptr[2] = src_pix_ptr[2];
				dst_pix_ptr += 4;
				src_pix_ptr += 3;
				dst_pix_ptr[0] = src_pix_ptr[0];
				dst_pix_ptr[1] = src_pix_ptr[1];
				dst_pix_ptr[2] = src_pix_ptr[2];
				dst_pix_ptr += 4;
				src_pix_ptr += 3;
				dst_pix_ptr[0] = src_pix_ptr[0];
				dst_pix_ptr[1] = src_pix_ptr[1];
				dst_pix_ptr[2] = src_pix_ptr[2];
				dst_pix_ptr += 4;
				src_pix_ptr += 3;
			}
			for (; j < srcw; j ++)
			{
				dst_pix_ptr[0] = src_pix_ptr[0];
				dst_pix_ptr[1] = src_pix_ptr[1];
				dst_pix_ptr[2] = src_pix_ptr[2];
				dst_pix_ptr += 4;
				src_pix_ptr += 3;
			}
		}
	}
}

void resize_nn_bgra2rgb(const unsigned char* src, int srcw, int srch, int src_widthStep,
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
		coord_x[dx] = ix * 4;
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
				cur_pix_ptr[2] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 2];
				cur_pix_ptr[2] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 3];
				cur_pix_ptr[2] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 4];
				cur_pix_ptr[2] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 5];
				cur_pix_ptr[2] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 6];
				cur_pix_ptr[2] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 7];
				cur_pix_ptr[2] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix + 2];
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
				cur_pix_ptr[2] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 1];
				cur_pix_ptr[2] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 2];
				cur_pix_ptr[2] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 3];
				cur_pix_ptr[2] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 4];
				cur_pix_ptr[2] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 5];
				cur_pix_ptr[2] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 6];
				cur_pix_ptr[2] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
				ix = coord_x[dx + 7];
				cur_pix_ptr[2] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix + 2];
				cur_pix_ptr += 3;
			}
			for (; dx < w; dx++)
			{
				ix = coord_x[dx];
				cur_pix_ptr[2] = cur_src_ptr[ix];
				cur_pix_ptr[1] = cur_src_ptr[ix + 1];
				cur_pix_ptr[0] = cur_src_ptr[ix + 2];
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
	const unsigned char* cur_src_c_ptr;
	const unsigned char* cur_src_pix_ptr;
	unsigned char* cur_dst_ptr0, *cur_dst_ptr1, *cur_dst_ptr2, *cur_dst_ptr3;
	unsigned char* cur_pix_ptr0, *cur_pix_ptr1, *cur_pix_ptr2, *cur_pix_ptr3;
	register int dx, dy;
	int src_widthStep2 = src_widthStep << 1;
	int src_widthStep3 = src_widthStep2 + src_widthStep;
	int src_widthStep4 = src_widthStep << 2;
	int widthStep2 = widthStep << 1;
	int widthStep3 = widthStep2 + widthStep;
	int widthStep4 = widthStep << 2;

	if (h % 8 == 0)
	{
		cur_src_c_ptr = src;
		cur_dst_ptr0 = dst;
		cur_dst_ptr1 = dst+widthStep;
		cur_dst_ptr2 = dst+widthStep2;
		cur_dst_ptr3 = dst+widthStep3;
		for (dy = 0; dy < w-3; dy+=4, cur_src_c_ptr += 12,
			cur_dst_ptr0 += widthStep4, 
			cur_dst_ptr1 += widthStep4, 
			cur_dst_ptr2 += widthStep4, 
			cur_dst_ptr3 += widthStep4)
		{
			cur_src_pix_ptr = cur_src_c_ptr;
			cur_pix_ptr0 = cur_dst_ptr0;
			cur_pix_ptr1 = cur_dst_ptr1;
			cur_pix_ptr2 = cur_dst_ptr2;
			cur_pix_ptr3 = cur_dst_ptr3;
			for (dx = 0; dx < h; dx += 8)
			{
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr1[0] = cur_src_pix_ptr[3];
				cur_pix_ptr1[1] = cur_src_pix_ptr[4];
				cur_pix_ptr1[2] = cur_src_pix_ptr[5];
				cur_pix_ptr2[0] = cur_src_pix_ptr[6];
				cur_pix_ptr2[1] = cur_src_pix_ptr[7];
				cur_pix_ptr2[2] = cur_src_pix_ptr[8];
				cur_pix_ptr3[0] = cur_src_pix_ptr[9];
				cur_pix_ptr3[1] = cur_src_pix_ptr[10];
				cur_pix_ptr3[2] = cur_src_pix_ptr[11];
				cur_pix_ptr0 += 3;
				cur_pix_ptr1 += 3;
				cur_pix_ptr2 += 3;
				cur_pix_ptr3 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr1[0] = cur_src_pix_ptr[3];
				cur_pix_ptr1[1] = cur_src_pix_ptr[4];
				cur_pix_ptr1[2] = cur_src_pix_ptr[5];
				cur_pix_ptr2[0] = cur_src_pix_ptr[6];
				cur_pix_ptr2[1] = cur_src_pix_ptr[7];
				cur_pix_ptr2[2] = cur_src_pix_ptr[8];
				cur_pix_ptr3[0] = cur_src_pix_ptr[9];
				cur_pix_ptr3[1] = cur_src_pix_ptr[10];
				cur_pix_ptr3[2] = cur_src_pix_ptr[11];
				cur_pix_ptr0 += 3;
				cur_pix_ptr1 += 3;
				cur_pix_ptr2 += 3;
				cur_pix_ptr3 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr1[0] = cur_src_pix_ptr[3];
				cur_pix_ptr1[1] = cur_src_pix_ptr[4];
				cur_pix_ptr1[2] = cur_src_pix_ptr[5];
				cur_pix_ptr2[0] = cur_src_pix_ptr[6];
				cur_pix_ptr2[1] = cur_src_pix_ptr[7];
				cur_pix_ptr2[2] = cur_src_pix_ptr[8];
				cur_pix_ptr3[0] = cur_src_pix_ptr[9];
				cur_pix_ptr3[1] = cur_src_pix_ptr[10];
				cur_pix_ptr3[2] = cur_src_pix_ptr[11];
				cur_pix_ptr0 += 3;
				cur_pix_ptr1 += 3;
				cur_pix_ptr2 += 3;
				cur_pix_ptr3 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr1[0] = cur_src_pix_ptr[3];
				cur_pix_ptr1[1] = cur_src_pix_ptr[4];
				cur_pix_ptr1[2] = cur_src_pix_ptr[5];
				cur_pix_ptr2[0] = cur_src_pix_ptr[6];
				cur_pix_ptr2[1] = cur_src_pix_ptr[7];
				cur_pix_ptr2[2] = cur_src_pix_ptr[8];
				cur_pix_ptr3[0] = cur_src_pix_ptr[9];
				cur_pix_ptr3[1] = cur_src_pix_ptr[10];
				cur_pix_ptr3[2] = cur_src_pix_ptr[11];
				cur_pix_ptr0 += 3;
				cur_pix_ptr1 += 3;
				cur_pix_ptr2 += 3;
				cur_pix_ptr3 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr1[0] = cur_src_pix_ptr[3];
				cur_pix_ptr1[1] = cur_src_pix_ptr[4];
				cur_pix_ptr1[2] = cur_src_pix_ptr[5];
				cur_pix_ptr2[0] = cur_src_pix_ptr[6];
				cur_pix_ptr2[1] = cur_src_pix_ptr[7];
				cur_pix_ptr2[2] = cur_src_pix_ptr[8];
				cur_pix_ptr3[0] = cur_src_pix_ptr[9];
				cur_pix_ptr3[1] = cur_src_pix_ptr[10];
				cur_pix_ptr3[2] = cur_src_pix_ptr[11];
				cur_pix_ptr0 += 3;
				cur_pix_ptr1 += 3;
				cur_pix_ptr2 += 3;
				cur_pix_ptr3 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr1[0] = cur_src_pix_ptr[3];
				cur_pix_ptr1[1] = cur_src_pix_ptr[4];
				cur_pix_ptr1[2] = cur_src_pix_ptr[5];
				cur_pix_ptr2[0] = cur_src_pix_ptr[6];
				cur_pix_ptr2[1] = cur_src_pix_ptr[7];
				cur_pix_ptr2[2] = cur_src_pix_ptr[8];
				cur_pix_ptr3[0] = cur_src_pix_ptr[9];
				cur_pix_ptr3[1] = cur_src_pix_ptr[10];
				cur_pix_ptr3[2] = cur_src_pix_ptr[11];
				cur_pix_ptr0 += 3;
				cur_pix_ptr1 += 3;
				cur_pix_ptr2 += 3;
				cur_pix_ptr3 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr1[0] = cur_src_pix_ptr[3];
				cur_pix_ptr1[1] = cur_src_pix_ptr[4];
				cur_pix_ptr1[2] = cur_src_pix_ptr[5];
				cur_pix_ptr2[0] = cur_src_pix_ptr[6];
				cur_pix_ptr2[1] = cur_src_pix_ptr[7];
				cur_pix_ptr2[2] = cur_src_pix_ptr[8];
				cur_pix_ptr3[0] = cur_src_pix_ptr[9];
				cur_pix_ptr3[1] = cur_src_pix_ptr[10];
				cur_pix_ptr3[2] = cur_src_pix_ptr[11];
				cur_pix_ptr0 += 3;
				cur_pix_ptr1 += 3;
				cur_pix_ptr2 += 3;
				cur_pix_ptr3 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr1[0] = cur_src_pix_ptr[3];
				cur_pix_ptr1[1] = cur_src_pix_ptr[4];
				cur_pix_ptr1[2] = cur_src_pix_ptr[5];
				cur_pix_ptr2[0] = cur_src_pix_ptr[6];
				cur_pix_ptr2[1] = cur_src_pix_ptr[7];
				cur_pix_ptr2[2] = cur_src_pix_ptr[8];
				cur_pix_ptr3[0] = cur_src_pix_ptr[9];
				cur_pix_ptr3[1] = cur_src_pix_ptr[10];
				cur_pix_ptr3[2] = cur_src_pix_ptr[11];
				cur_pix_ptr0 += 3;
				cur_pix_ptr1 += 3;
				cur_pix_ptr2 += 3;
				cur_pix_ptr3 += 3;
				cur_src_pix_ptr += src_widthStep;
			}
		}
		for (; dy < w; dy++, cur_src_c_ptr += 3, cur_dst_ptr0 += widthStep)
		{
			cur_src_pix_ptr = cur_src_c_ptr;
			cur_pix_ptr0 = cur_dst_ptr0;
			for (dx = 0; dx < h; dx += 8)
			{
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr0 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr0 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr0 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr0 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr0 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr0 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr0 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr0 += 3;
				cur_src_pix_ptr += src_widthStep;
			}
		}
	}
	else
	{
		cur_src_c_ptr = src;
		cur_dst_ptr0 = dst;
		for (dy = 0; dy < w; dy++, cur_src_c_ptr += 3, cur_dst_ptr0 += widthStep)
		{
			cur_src_pix_ptr = cur_src_c_ptr;
			cur_pix_ptr0 = cur_dst_ptr0;
			for (dx = 0; dx < h; dx += 8)
			{
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr0 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr0 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr0 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr0 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr0 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr0 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr0 += 3;
				cur_src_pix_ptr += src_widthStep;
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr0 += 3;
				cur_src_pix_ptr += src_widthStep;
			}
			for (; dx < h; dx ++)
			{
				cur_pix_ptr0[0] = cur_src_pix_ptr[0];
				cur_pix_ptr0[1] = cur_src_pix_ptr[1];
				cur_pix_ptr0[2] = cur_src_pix_ptr[2];
				cur_pix_ptr0 += 3;
				cur_src_pix_ptr += src_widthStep;
			}
		}
	}
}

void transpose_c3_h4w4(const unsigned char* src, int w, int h, int src_widthStep,
	unsigned char* dst, int widthStep)
{
	const unsigned char* cur_src_c_ptr;
	const unsigned char* cur_src_pix_ptr0, *cur_src_pix_ptr1, *cur_src_pix_ptr2, *cur_src_pix_ptr3;
	unsigned char* cur_dst_ptr0, *cur_dst_ptr1, *cur_dst_ptr2, *cur_dst_ptr3;
	unsigned char* cur_pix_ptr0, *cur_pix_ptr1, *cur_pix_ptr2, *cur_pix_ptr3;
	register int dx, dy;
	int src_widthStep2 = src_widthStep << 1;
	int src_widthStep3 = src_widthStep2 + src_widthStep;
	int src_widthStep4 = src_widthStep << 2;
	int widthStep2 = widthStep << 1;
	int widthStep3 = widthStep2 + widthStep;
	int widthStep4 = widthStep << 2;

	cur_src_c_ptr = src;
	cur_dst_ptr0 = dst;
	cur_dst_ptr1 = dst + widthStep;
	cur_dst_ptr2 = dst + widthStep2;
	cur_dst_ptr3 = dst + widthStep3;
	for (dy = 0; dy < w; dy += 4, cur_src_c_ptr += 12,
		cur_dst_ptr0 += widthStep4,
		cur_dst_ptr1 += widthStep4,
		cur_dst_ptr2 += widthStep4,
		cur_dst_ptr3 += widthStep4)
	{
		cur_src_pix_ptr0 = cur_src_c_ptr;
		cur_src_pix_ptr1 = cur_src_c_ptr + src_widthStep;
		cur_src_pix_ptr2 = cur_src_c_ptr + src_widthStep2;
		cur_src_pix_ptr3 = cur_src_c_ptr + src_widthStep3;
		cur_pix_ptr0 = cur_dst_ptr0;
		cur_pix_ptr1 = cur_dst_ptr1;
		cur_pix_ptr2 = cur_dst_ptr2;
		cur_pix_ptr3 = cur_dst_ptr3;
		for (dx = 0; dx < h; dx += 4)
		{
			cur_pix_ptr0[0] = cur_src_pix_ptr0[0];
			cur_pix_ptr0[1] = cur_src_pix_ptr0[1];
			cur_pix_ptr0[2] = cur_src_pix_ptr0[2];
			cur_pix_ptr0[3] = cur_src_pix_ptr1[0];
			cur_pix_ptr0[4] = cur_src_pix_ptr1[1];
			cur_pix_ptr0[5] = cur_src_pix_ptr1[2];
			cur_pix_ptr0[6] = cur_src_pix_ptr2[0];
			cur_pix_ptr0[7] = cur_src_pix_ptr2[1];
			cur_pix_ptr0[8] = cur_src_pix_ptr2[2];
			cur_pix_ptr0[9] = cur_src_pix_ptr3[0];
			cur_pix_ptr0[10] = cur_src_pix_ptr3[1];
			cur_pix_ptr0[11] = cur_src_pix_ptr3[2];
			cur_pix_ptr1[0] = cur_src_pix_ptr0[3];
			cur_pix_ptr1[1] = cur_src_pix_ptr0[4];
			cur_pix_ptr1[2] = cur_src_pix_ptr0[5];
			cur_pix_ptr1[3] = cur_src_pix_ptr1[3];
			cur_pix_ptr1[4] = cur_src_pix_ptr1[4];
			cur_pix_ptr1[5] = cur_src_pix_ptr1[5];
			cur_pix_ptr1[6] = cur_src_pix_ptr2[3];
			cur_pix_ptr1[7] = cur_src_pix_ptr2[4];
			cur_pix_ptr1[8] = cur_src_pix_ptr2[5];
			cur_pix_ptr1[9] = cur_src_pix_ptr3[3];
			cur_pix_ptr1[10] = cur_src_pix_ptr3[4];
			cur_pix_ptr1[11] = cur_src_pix_ptr3[5];
			cur_pix_ptr2[0] = cur_src_pix_ptr0[6];
			cur_pix_ptr2[1] = cur_src_pix_ptr0[7];
			cur_pix_ptr2[2] = cur_src_pix_ptr0[8];
			cur_pix_ptr2[3] = cur_src_pix_ptr1[6];
			cur_pix_ptr2[4] = cur_src_pix_ptr1[7];
			cur_pix_ptr2[5] = cur_src_pix_ptr1[8];
			cur_pix_ptr2[6] = cur_src_pix_ptr2[6];
			cur_pix_ptr2[7] = cur_src_pix_ptr2[7];
			cur_pix_ptr2[8] = cur_src_pix_ptr2[8];
			cur_pix_ptr2[9] = cur_src_pix_ptr3[6];
			cur_pix_ptr2[10] = cur_src_pix_ptr3[7];
			cur_pix_ptr2[11] = cur_src_pix_ptr3[8];
			cur_pix_ptr3[0] = cur_src_pix_ptr0[9];
			cur_pix_ptr3[1] = cur_src_pix_ptr0[10];
			cur_pix_ptr3[2] = cur_src_pix_ptr0[11];
			cur_pix_ptr3[3] = cur_src_pix_ptr1[9];
			cur_pix_ptr3[4] = cur_src_pix_ptr1[10];
			cur_pix_ptr3[5] = cur_src_pix_ptr1[11];
			cur_pix_ptr3[6] = cur_src_pix_ptr2[9];
			cur_pix_ptr3[7] = cur_src_pix_ptr2[10];
			cur_pix_ptr3[8] = cur_src_pix_ptr2[11];
			cur_pix_ptr3[9] = cur_src_pix_ptr3[9];
			cur_pix_ptr3[10] = cur_src_pix_ptr3[10];
			cur_pix_ptr3[11] = cur_src_pix_ptr3[11];
			cur_pix_ptr0 += 12;
			cur_pix_ptr1 += 12;
			cur_pix_ptr2 += 12;
			cur_pix_ptr3 += 12;
			cur_src_pix_ptr0 += src_widthStep4;
			cur_src_pix_ptr1 += src_widthStep4;
			cur_src_pix_ptr2 += src_widthStep4;
			cur_src_pix_ptr3 += src_widthStep4;
		}
	}
}

void transpose_c3_h8w8(const unsigned char* src, int w, int h, int src_widthStep,
	unsigned char* dst, int widthStep)
{
	const unsigned char* cur_src_c_ptr;
	const unsigned char* cur_src_pix_ptr0, *cur_src_pix_ptr1, *cur_src_pix_ptr2, *cur_src_pix_ptr3;
	const unsigned char* cur_src_pix_ptr4, *cur_src_pix_ptr5, *cur_src_pix_ptr6, *cur_src_pix_ptr7;
	unsigned char* cur_dst_ptr0, *cur_dst_ptr1, *cur_dst_ptr2, *cur_dst_ptr3;
	unsigned char* cur_dst_ptr4, *cur_dst_ptr5, *cur_dst_ptr6, *cur_dst_ptr7;
	unsigned char* cur_pix_ptr0, *cur_pix_ptr1, *cur_pix_ptr2, *cur_pix_ptr3;
	unsigned char* cur_pix_ptr4, *cur_pix_ptr5, *cur_pix_ptr6, *cur_pix_ptr7;
	register int dx, dy;
	int src_widthStep2 = src_widthStep << 1;
	int src_widthStep3 = src_widthStep2 + src_widthStep;
	int src_widthStep4 = src_widthStep << 2;
	int src_widthStep5 = src_widthStep4 + src_widthStep;
	int src_widthStep6 = src_widthStep4 + src_widthStep2;
	int src_widthStep7 = src_widthStep4 + src_widthStep3;
	int src_widthStep8 = src_widthStep << 3;
	int widthStep2 = widthStep << 1;
	int widthStep3 = widthStep2 + widthStep;
	int widthStep4 = widthStep << 2;
	int widthStep5 = widthStep4 + widthStep;
	int widthStep6 = widthStep4 + widthStep2;
	int widthStep7 = widthStep4 + widthStep3;
	int widthStep8 = widthStep << 3;

	cur_src_c_ptr = src;
	cur_dst_ptr0 = dst;
	cur_dst_ptr1 = dst + widthStep;
	cur_dst_ptr2 = dst + widthStep2;
	cur_dst_ptr3 = dst + widthStep3;
	cur_dst_ptr4 = dst + widthStep4;
	cur_dst_ptr5 = dst + widthStep5;
	cur_dst_ptr6 = dst + widthStep6;
	cur_dst_ptr7 = dst + widthStep7;
	for (dy = 0; dy < w; dy += 8, cur_src_c_ptr += 24,
		cur_dst_ptr0 += widthStep8,
		cur_dst_ptr1 += widthStep8,
		cur_dst_ptr2 += widthStep8,
		cur_dst_ptr3 += widthStep8,
		cur_dst_ptr4 += widthStep8,
		cur_dst_ptr5 += widthStep8,
		cur_dst_ptr6 += widthStep8,
		cur_dst_ptr7 += widthStep8)
	{
		cur_src_pix_ptr0 = cur_src_c_ptr;
		cur_src_pix_ptr1 = cur_src_c_ptr + src_widthStep;
		cur_src_pix_ptr2 = cur_src_c_ptr + src_widthStep2;
		cur_src_pix_ptr3 = cur_src_c_ptr + src_widthStep3;
		cur_src_pix_ptr4 = cur_src_c_ptr + src_widthStep4;
		cur_src_pix_ptr5 = cur_src_c_ptr + src_widthStep5;
		cur_src_pix_ptr6 = cur_src_c_ptr + src_widthStep6;
		cur_src_pix_ptr7 = cur_src_c_ptr + src_widthStep7;
		cur_pix_ptr0 = cur_dst_ptr0;
		cur_pix_ptr1 = cur_dst_ptr1;
		cur_pix_ptr2 = cur_dst_ptr2;
		cur_pix_ptr3 = cur_dst_ptr3;
		cur_pix_ptr4 = cur_dst_ptr4;
		cur_pix_ptr5 = cur_dst_ptr5;
		cur_pix_ptr6 = cur_dst_ptr6;
		cur_pix_ptr7 = cur_dst_ptr7;
		for (dx = 0; dx < h; dx += 8)
		{
			cur_pix_ptr0[0] = cur_src_pix_ptr0[0];
			cur_pix_ptr0[1] = cur_src_pix_ptr0[1];
			cur_pix_ptr0[2] = cur_src_pix_ptr0[2];
			cur_pix_ptr0[3] = cur_src_pix_ptr1[0];
			cur_pix_ptr0[4] = cur_src_pix_ptr1[1];
			cur_pix_ptr0[5] = cur_src_pix_ptr1[2];
			cur_pix_ptr0[6] = cur_src_pix_ptr2[0];
			cur_pix_ptr0[7] = cur_src_pix_ptr2[1];
			cur_pix_ptr0[8] = cur_src_pix_ptr2[2];
			cur_pix_ptr0[9] = cur_src_pix_ptr3[0];
			cur_pix_ptr0[10] = cur_src_pix_ptr3[1];
			cur_pix_ptr0[11] = cur_src_pix_ptr3[2];
			cur_pix_ptr0[12] = cur_src_pix_ptr4[0];
			cur_pix_ptr0[13] = cur_src_pix_ptr4[1];
			cur_pix_ptr0[14] = cur_src_pix_ptr4[2];
			cur_pix_ptr0[15] = cur_src_pix_ptr5[0];
			cur_pix_ptr0[16] = cur_src_pix_ptr5[1];
			cur_pix_ptr0[17] = cur_src_pix_ptr5[2];
			cur_pix_ptr0[18] = cur_src_pix_ptr6[0];
			cur_pix_ptr0[19] = cur_src_pix_ptr6[1];
			cur_pix_ptr0[20] = cur_src_pix_ptr6[2];
			cur_pix_ptr0[21] = cur_src_pix_ptr7[0];
			cur_pix_ptr0[22] = cur_src_pix_ptr7[1];
			cur_pix_ptr0[23] = cur_src_pix_ptr7[2];
			cur_pix_ptr1[0] = cur_src_pix_ptr0[3];
			cur_pix_ptr1[1] = cur_src_pix_ptr0[4];
			cur_pix_ptr1[2] = cur_src_pix_ptr0[5];
			cur_pix_ptr1[3] = cur_src_pix_ptr1[3];
			cur_pix_ptr1[4] = cur_src_pix_ptr1[4];
			cur_pix_ptr1[5] = cur_src_pix_ptr1[5];
			cur_pix_ptr1[6] = cur_src_pix_ptr2[3];
			cur_pix_ptr1[7] = cur_src_pix_ptr2[4];
			cur_pix_ptr1[8] = cur_src_pix_ptr2[5];
			cur_pix_ptr1[9] = cur_src_pix_ptr3[3];
			cur_pix_ptr1[10] = cur_src_pix_ptr3[4];
			cur_pix_ptr1[11] = cur_src_pix_ptr3[5];
			cur_pix_ptr1[12] = cur_src_pix_ptr4[3];
			cur_pix_ptr1[13] = cur_src_pix_ptr4[4];
			cur_pix_ptr1[14] = cur_src_pix_ptr4[5];
			cur_pix_ptr1[15] = cur_src_pix_ptr5[3];
			cur_pix_ptr1[16] = cur_src_pix_ptr5[4];
			cur_pix_ptr1[17] = cur_src_pix_ptr5[5];
			cur_pix_ptr1[18] = cur_src_pix_ptr6[3];
			cur_pix_ptr1[19] = cur_src_pix_ptr6[4];
			cur_pix_ptr1[20] = cur_src_pix_ptr6[5];
			cur_pix_ptr1[21] = cur_src_pix_ptr7[3];
			cur_pix_ptr1[22] = cur_src_pix_ptr7[4];
			cur_pix_ptr1[23] = cur_src_pix_ptr7[5];
			cur_pix_ptr2[0] = cur_src_pix_ptr0[6];
			cur_pix_ptr2[1] = cur_src_pix_ptr0[7];
			cur_pix_ptr2[2] = cur_src_pix_ptr0[8];
			cur_pix_ptr2[3] = cur_src_pix_ptr1[6];
			cur_pix_ptr2[4] = cur_src_pix_ptr1[7];
			cur_pix_ptr2[5] = cur_src_pix_ptr1[8];
			cur_pix_ptr2[6] = cur_src_pix_ptr2[6];
			cur_pix_ptr2[7] = cur_src_pix_ptr2[7];
			cur_pix_ptr2[8] = cur_src_pix_ptr2[8];
			cur_pix_ptr2[9] = cur_src_pix_ptr3[6];
			cur_pix_ptr2[10] = cur_src_pix_ptr3[7];
			cur_pix_ptr2[11] = cur_src_pix_ptr3[8];
			cur_pix_ptr2[12] = cur_src_pix_ptr4[6];
			cur_pix_ptr2[13] = cur_src_pix_ptr4[7];
			cur_pix_ptr2[14] = cur_src_pix_ptr4[8];
			cur_pix_ptr2[15] = cur_src_pix_ptr5[6];
			cur_pix_ptr2[16] = cur_src_pix_ptr5[7];
			cur_pix_ptr2[17] = cur_src_pix_ptr5[8];
			cur_pix_ptr2[18] = cur_src_pix_ptr6[6];
			cur_pix_ptr2[19] = cur_src_pix_ptr6[7];
			cur_pix_ptr2[20] = cur_src_pix_ptr6[8];
			cur_pix_ptr2[21] = cur_src_pix_ptr7[6];
			cur_pix_ptr2[22] = cur_src_pix_ptr7[7];
			cur_pix_ptr2[23] = cur_src_pix_ptr7[8];
			cur_pix_ptr3[0] = cur_src_pix_ptr0[9];
			cur_pix_ptr3[1] = cur_src_pix_ptr0[10];
			cur_pix_ptr3[2] = cur_src_pix_ptr0[11];
			cur_pix_ptr3[3] = cur_src_pix_ptr1[9];
			cur_pix_ptr3[4] = cur_src_pix_ptr1[10];
			cur_pix_ptr3[5] = cur_src_pix_ptr1[11];
			cur_pix_ptr3[6] = cur_src_pix_ptr2[9];
			cur_pix_ptr3[7] = cur_src_pix_ptr2[10];
			cur_pix_ptr3[8] = cur_src_pix_ptr2[11];
			cur_pix_ptr3[9] = cur_src_pix_ptr3[9];
			cur_pix_ptr3[10] = cur_src_pix_ptr3[10];
			cur_pix_ptr3[11] = cur_src_pix_ptr3[11];
			cur_pix_ptr3[12] = cur_src_pix_ptr4[9];
			cur_pix_ptr3[13] = cur_src_pix_ptr4[10];
			cur_pix_ptr3[14] = cur_src_pix_ptr4[11];
			cur_pix_ptr3[15] = cur_src_pix_ptr5[9];
			cur_pix_ptr3[16] = cur_src_pix_ptr5[10];
			cur_pix_ptr3[17] = cur_src_pix_ptr5[11];
			cur_pix_ptr3[18] = cur_src_pix_ptr6[9];
			cur_pix_ptr3[19] = cur_src_pix_ptr6[10];
			cur_pix_ptr3[20] = cur_src_pix_ptr6[11];
			cur_pix_ptr3[21] = cur_src_pix_ptr7[9];
			cur_pix_ptr3[22] = cur_src_pix_ptr7[10];
			cur_pix_ptr3[23] = cur_src_pix_ptr7[11];
			cur_pix_ptr4[0] = cur_src_pix_ptr0[12];
			cur_pix_ptr4[1] = cur_src_pix_ptr0[13];
			cur_pix_ptr4[2] = cur_src_pix_ptr0[14];
			cur_pix_ptr4[3] = cur_src_pix_ptr1[12];
			cur_pix_ptr4[4] = cur_src_pix_ptr1[13];
			cur_pix_ptr4[5] = cur_src_pix_ptr1[14];
			cur_pix_ptr4[6] = cur_src_pix_ptr2[12];
			cur_pix_ptr4[7] = cur_src_pix_ptr2[13];
			cur_pix_ptr4[8] = cur_src_pix_ptr2[14];
			cur_pix_ptr4[9] = cur_src_pix_ptr3[12];
			cur_pix_ptr4[10] = cur_src_pix_ptr3[13];
			cur_pix_ptr4[11] = cur_src_pix_ptr3[14];
			cur_pix_ptr4[12] = cur_src_pix_ptr4[12];
			cur_pix_ptr4[13] = cur_src_pix_ptr4[13];
			cur_pix_ptr4[14] = cur_src_pix_ptr4[14];
			cur_pix_ptr4[15] = cur_src_pix_ptr5[12];
			cur_pix_ptr4[16] = cur_src_pix_ptr5[13];
			cur_pix_ptr4[17] = cur_src_pix_ptr5[14];
			cur_pix_ptr4[18] = cur_src_pix_ptr6[12];
			cur_pix_ptr4[19] = cur_src_pix_ptr6[13];
			cur_pix_ptr4[20] = cur_src_pix_ptr6[14];
			cur_pix_ptr4[21] = cur_src_pix_ptr7[12];
			cur_pix_ptr4[22] = cur_src_pix_ptr7[13];
			cur_pix_ptr4[23] = cur_src_pix_ptr7[14];
			cur_pix_ptr5[0] = cur_src_pix_ptr0[15];
			cur_pix_ptr5[1] = cur_src_pix_ptr0[16];
			cur_pix_ptr5[2] = cur_src_pix_ptr0[17];
			cur_pix_ptr5[3] = cur_src_pix_ptr1[15];
			cur_pix_ptr5[4] = cur_src_pix_ptr1[16];
			cur_pix_ptr5[5] = cur_src_pix_ptr1[17];
			cur_pix_ptr5[6] = cur_src_pix_ptr2[15];
			cur_pix_ptr5[7] = cur_src_pix_ptr2[16];
			cur_pix_ptr5[8] = cur_src_pix_ptr2[17];
			cur_pix_ptr5[9] = cur_src_pix_ptr3[15];
			cur_pix_ptr5[10] = cur_src_pix_ptr3[16];
			cur_pix_ptr5[11] = cur_src_pix_ptr3[17];
			cur_pix_ptr5[12] = cur_src_pix_ptr4[15];
			cur_pix_ptr5[13] = cur_src_pix_ptr4[16];
			cur_pix_ptr5[14] = cur_src_pix_ptr4[17];
			cur_pix_ptr5[15] = cur_src_pix_ptr5[15];
			cur_pix_ptr5[16] = cur_src_pix_ptr5[16];
			cur_pix_ptr5[17] = cur_src_pix_ptr5[17];
			cur_pix_ptr5[18] = cur_src_pix_ptr6[15];
			cur_pix_ptr5[19] = cur_src_pix_ptr6[16];
			cur_pix_ptr5[20] = cur_src_pix_ptr6[17];
			cur_pix_ptr5[21] = cur_src_pix_ptr7[15];
			cur_pix_ptr5[22] = cur_src_pix_ptr7[16];
			cur_pix_ptr5[23] = cur_src_pix_ptr7[17];
			cur_pix_ptr6[0] = cur_src_pix_ptr0[18];
			cur_pix_ptr6[1] = cur_src_pix_ptr0[19];
			cur_pix_ptr6[2] = cur_src_pix_ptr0[20];
			cur_pix_ptr6[3] = cur_src_pix_ptr1[18];
			cur_pix_ptr6[4] = cur_src_pix_ptr1[19];
			cur_pix_ptr6[5] = cur_src_pix_ptr1[20];
			cur_pix_ptr6[6] = cur_src_pix_ptr2[18];
			cur_pix_ptr6[7] = cur_src_pix_ptr2[19];
			cur_pix_ptr6[8] = cur_src_pix_ptr2[20];
			cur_pix_ptr6[9] = cur_src_pix_ptr3[18];
			cur_pix_ptr6[10] = cur_src_pix_ptr3[19];
			cur_pix_ptr6[11] = cur_src_pix_ptr3[20];
			cur_pix_ptr6[12] = cur_src_pix_ptr4[18];
			cur_pix_ptr6[13] = cur_src_pix_ptr4[19];
			cur_pix_ptr6[14] = cur_src_pix_ptr4[20];
			cur_pix_ptr6[15] = cur_src_pix_ptr5[18];
			cur_pix_ptr6[16] = cur_src_pix_ptr5[19];
			cur_pix_ptr6[17] = cur_src_pix_ptr5[20];
			cur_pix_ptr6[18] = cur_src_pix_ptr6[18];
			cur_pix_ptr6[19] = cur_src_pix_ptr6[19];
			cur_pix_ptr6[20] = cur_src_pix_ptr6[20];
			cur_pix_ptr6[21] = cur_src_pix_ptr7[18];
			cur_pix_ptr6[22] = cur_src_pix_ptr7[19];
			cur_pix_ptr6[23] = cur_src_pix_ptr7[20];
			cur_pix_ptr7[0] = cur_src_pix_ptr0[21];
			cur_pix_ptr7[1] = cur_src_pix_ptr0[22];
			cur_pix_ptr7[2] = cur_src_pix_ptr0[23];
			cur_pix_ptr7[3] = cur_src_pix_ptr1[21];
			cur_pix_ptr7[4] = cur_src_pix_ptr1[22];
			cur_pix_ptr7[5] = cur_src_pix_ptr1[23];
			cur_pix_ptr7[6] = cur_src_pix_ptr2[21];
			cur_pix_ptr7[7] = cur_src_pix_ptr2[22];
			cur_pix_ptr7[8] = cur_src_pix_ptr2[23];
			cur_pix_ptr7[9] = cur_src_pix_ptr3[21];
			cur_pix_ptr7[10] = cur_src_pix_ptr3[22];
			cur_pix_ptr7[11] = cur_src_pix_ptr3[23];
			cur_pix_ptr7[12] = cur_src_pix_ptr4[21];
			cur_pix_ptr7[13] = cur_src_pix_ptr4[22];
			cur_pix_ptr7[14] = cur_src_pix_ptr4[23];
			cur_pix_ptr7[15] = cur_src_pix_ptr5[21];
			cur_pix_ptr7[16] = cur_src_pix_ptr5[22];
			cur_pix_ptr7[17] = cur_src_pix_ptr5[23];
			cur_pix_ptr7[18] = cur_src_pix_ptr6[21];
			cur_pix_ptr7[19] = cur_src_pix_ptr6[22];
			cur_pix_ptr7[20] = cur_src_pix_ptr6[23];
			cur_pix_ptr7[21] = cur_src_pix_ptr7[21];
			cur_pix_ptr7[22] = cur_src_pix_ptr7[22];
			cur_pix_ptr7[23] = cur_src_pix_ptr7[23];
			cur_pix_ptr0 += 24;
			cur_pix_ptr1 += 24;
			cur_pix_ptr2 += 24;
			cur_pix_ptr3 += 24;
			cur_pix_ptr4 += 24;
			cur_pix_ptr5 += 24;
			cur_pix_ptr6 += 24;
			cur_pix_ptr7 += 24;
			cur_src_pix_ptr0 += src_widthStep8;
			cur_src_pix_ptr1 += src_widthStep8;
			cur_src_pix_ptr2 += src_widthStep8;
			cur_src_pix_ptr3 += src_widthStep8;
			cur_src_pix_ptr4 += src_widthStep8;
			cur_src_pix_ptr5 += src_widthStep8;
			cur_src_pix_ptr6 += src_widthStep8;
			cur_src_pix_ptr7 += src_widthStep8;
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


