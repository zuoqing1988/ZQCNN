#include "ZQ_CNN_MTCNN.h"
#include "ZQlib/ZQ_FindSimilarityLandmark.h"
#include <opencv2/opencv.hpp>
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
using namespace cv;

void get_face_mask(const Mat& img, const ZQ_CNN_BBox106& box, Mat& mask);
void draw_convex_hull(Mat& mask, std::vector<Point2f> vec, Scalar color);
void correct_colours(const Mat& img1, const Mat& img2, const ZQ_CNN_BBox106& box1, Mat& correct_im2);

int main(int argc, const char** argv)
{
	if (argc != 4)
	{
		printf("%s img1 img2 out\n");
		return 1;
	}
	ZQ_CNN_MTCNN mtcnn;
#if defined(_WIN32)
	if (!mtcnn.Init(
		"model/det1-dw20-fast.zqparams", "model/det1-dw20-fast.nchwbin",
		"model/det2-dw24-fast.zqparams", "model/det2-dw24-fast.nchwbin",
		"model/det3-dw48-fast.zqparams", "model/det3-dw48-fast.nchwbin", 1,
		true, "model/det5-dw112.zqparams", "model/det5-dw112.nchwbin")
		)
#else
	if (!mtcnn.Init(
		"../../model/det1-dw20-fast.zqparams", "../../model/det1-dw20-fast.nchwbin",
		"../../model/det2-dw24-fast.zqparams", "../../model/det2-dw24-fast.nchwbin",
		"../../model/det3-dw48-fast.zqparams", "../../model/det3-dw48-fast.nchwbin", 1,
		true, "../../model/det5-dw112.zqparams", "../../model/det5-dw112.nchwbin")
		)
#endif
	{
		printf("failed to init MTCNN!\n");
		return 1;
	}
	
	Mat img1, img2;
	const char* img1_name = argv[1];
	const char* img2_name = argv[2];
	const char* out_name = argv[3];
	img1 = imread(img1_name);
	img2 = imread(img2_name);
	if (img1.empty())
	{
		printf("failed to read image %s\n", img1_name);
		return 1;
	}
	if (img2.empty())
	{
		printf("failed to read image %s\n", img2_name);
		return 1;
	}
	std::vector<ZQ_CNN_BBox106> box1,box2;
	mtcnn.SetPara(img1.cols, img1.rows, 40, 0.5, 0.6, 0.8, 0.4, 0.5, 0.5, 0.709, 3, 20, 4, true, true, 1);
	if (!mtcnn.Find106(img1.data, img1.cols, img1.rows, img1.step[0], box1) || box1.size() == 0)
	{
		printf("failed to detect face for image %s\n", img1_name);
		return 1;
	}
	if (box1.size() != 1)
	{
		printf("Two many faces detected in image %s\n", img1_name);
		return 1;
	}
	mtcnn.SetPara(img2.cols, img2.rows, 20, 0.5, 0.6, 0.8, 0.4, 0.5, 0.5, 0.709, 3, 20, 4, true, true, 1);
	if (!mtcnn.Find106(img2.data, img2.cols, img2.rows, img2.step[0], box2) || box2.size() == 0)
	{
		printf("failed to detect face for image %s\n", img2_name);
		return 1;
	}
	if (box2.size() != 1)
	{
		printf("Two many faces detected in image %s\n", img2_name);
		return 1;
	}

	/****/
	float transform[6];
	ZQ_FindSimilarityLandmark::FindSimilarityLandmark(106, box2[0].ppoint, box1[0].ppoint, transform);
	Mat trans(2, 3, CV_32FC1);
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			trans.ptr<float>(i)[j] = transform[i*3+j];
		}
	}

	Mat mask2, warped_mask2, mask1, combine_mask;
	Mat warped_img2, warped_correct_img2;
	get_face_mask(img2, box2[0], mask2);
	warpAffine(mask2, warped_mask2, trans,Size(img1.cols,img1.rows));
	warpAffine(img2, warped_img2, trans, Size(img1.cols, img1.rows));
	get_face_mask(img1, box1[0], mask1);
	correct_colours(img1, warped_img2, box1[0], warped_correct_img2);
	
	mask1.copyTo(combine_mask);
	
	for (int h = 0; h < mask1.rows; h++)
	{
		for (int w = 0; w < mask1.cols; w++)
		{
			combine_mask.ptr<float>(h)[w] = __max(mask1.ptr<float>(h)[w], warped_mask2.ptr<float>(h)[w]);
		}
	}
	/*namedWindow("combine_mask");
	imshow("combine_mask", combine_mask);
	waitKey(0);*/
	Mat out_im;
	img1.copyTo(out_im);
	int C = img1.channels();
	for (int h = 0; h < img1.rows; h++)
	{
		for (int w = 0; w < img1.cols; w++)
		{
			float alpha = combine_mask.ptr<float>(h)[w];
			for (int c = 0; c < C; c++)
			{
				float out_val = img1.data[h*img1.step[0] + w*C + c] * (1 - alpha) 
					+ warped_correct_img2.data[h*warped_correct_img2.step[0] + w*C + c] * alpha;
				out_im.data[h*out_im.step[0] + w*C + c] = __min(255, __max(0, out_val));
			}
		}
	}
	imwrite(out_name, out_im);
	printf("done!\n");

	namedWindow("ori1");
	namedWindow("ori2");
	namedWindow("out");
	imshow("ori1", img1);
	imshow("ori2", img2);
	imshow("out", out_im);
	waitKey(0);
	return 0;
}

void get_face_mask(const Mat& img, const ZQ_CNN_BBox106& box, Mat& mask)
{
	mask = Mat(img.rows, img.cols, CV_32FC1);

	std::vector<Point2f> left_brow_vec;
	for (int i = 33; i <= 37; i++)
	{
		left_brow_vec.push_back(Point2f(box.ppoint[i * 2], box.ppoint[i * 2 + 1]));
	}
	for (int i = 64; i <= 67; i++)
	{
		left_brow_vec.push_back(Point2f(box.ppoint[i * 2], box.ppoint[i * 2 + 1]));
	}

	std::vector<Point2f> right_brow_vec;
	for (int i = 38; i <= 42; i++)
	{
		right_brow_vec.push_back(Point2f(box.ppoint[i * 2], box.ppoint[i * 2 + 1]));
	}
	for (int i = 68; i <= 71; i++)
	{
		right_brow_vec.push_back(Point2f(box.ppoint[i * 2], box.ppoint[i * 2 + 1]));
	}

	std::vector<Point2f> left_eye_vec;
	for (int i = 52; i <= 57; i++)
	{
		left_eye_vec.push_back(Point2f(box.ppoint[i * 2], box.ppoint[i * 2 + 1]));
	}
	for (int i = 72; i <= 74; i++)
	{
		left_eye_vec.push_back(Point2f(box.ppoint[i * 2], box.ppoint[i * 2 + 1]));
	}

	std::vector<Point2f> right_eye_vec;
	for (int i = 58; i <= 63; i++)
	{
		right_eye_vec.push_back(Point2f(box.ppoint[i * 2], box.ppoint[i * 2 + 1]));
	}
	for (int i = 75; i <= 77; i++)
	{
		right_eye_vec.push_back(Point2f(box.ppoint[i * 2], box.ppoint[i * 2 + 1]));
	}

	std::vector<Point2f> nose_vec;
	for (int i = 43; i <= 51; i++)
	{
		nose_vec.push_back(Point2f(box.ppoint[i * 2], box.ppoint[i * 2 + 1]));
	}
	for (int i = 78; i <= 83; i++)
	{
		nose_vec.push_back(Point2f(box.ppoint[i * 2], box.ppoint[i * 2 + 1]));
	}

	std::vector<Point2f> mouth_vec;
	for (int i = 84; i <= 103; i++)
	{
		mouth_vec.push_back(Point2f(box.ppoint[i * 2], box.ppoint[i * 2 + 1]));
	}
	
	//Group1: left_brow+left_eye+right_brow+right_eye
	//Group2: nose+mouth
	std::vector<Point2f> group1, group2;
	group1.insert(group1.end(), left_brow_vec.begin(), left_brow_vec.end());
	group1.insert(group1.end(), left_eye_vec.begin(), left_eye_vec.end());
	group1.insert(group1.end(), right_brow_vec.begin(), right_brow_vec.end());
	group1.insert(group1.end(), right_eye_vec.begin(), right_eye_vec.end());

	group2.insert(group2.end(), nose_vec.begin(), nose_vec.end());
	group2.insert(group2.end(), mouth_vec.begin(), mouth_vec.end());

	draw_convex_hull(mask, group1, Scalar(1));
	draw_convex_hull(mask, group2, Scalar(1));
	
	int kernel_size = (box.col2 - box.col1 + box.row2 - box.row1)*0.05;
	kernel_size /= 2;
	kernel_size = __max(1, kernel_size);
	kernel_size *= 2;
	kernel_size += 1;
	GaussianBlur(mask, mask, cv::Size(kernel_size, kernel_size), kernel_size*0.5, kernel_size*0.5);
	for (int h = 0; h < mask.rows; h++)
	{
		for (int w = 0; w < mask.cols; w++)
		{
			float val = mask.ptr<float>(h)[w];
			mask.ptr<float>(h)[w] = val > 0 ? 1 : 0;
		}
	}
	GaussianBlur(mask, mask, cv::Size(kernel_size, kernel_size), kernel_size*0.5, kernel_size*0.5);
}
	
void draw_convex_hull(Mat& mask, std::vector<Point2f> vec, Scalar color)
{
	
	std::vector<Point2f> hull;
	
	cv::convexHull(vec, hull);
	std::vector<Point> hull_int;
	for (int i = 0; i < hull.size(); i++)
	{
		hull_int.push_back(Point(hull[i].x, hull[i].y));
	}
	cv::fillConvexPoly(mask, hull_int, color);
}


void correct_colours(const Mat& img1, const Mat& img2, const ZQ_CNN_BBox106& box1, Mat& correct_im2)
{
	img2.copyTo(correct_im2);
	float sum_left_x = 0, sum_left_y = 0;
	float weight_left = 0;
	for (int i = 52; i <= 57; i++)
	{
		sum_left_x += box1.ppoint[i * 2];
		sum_left_y += box1.ppoint[i * 2 + 1];
		weight_left += 1;
	}
	for (int i = 72; i <= 74; i++)
	{
		sum_left_x += box1.ppoint[i * 2];
		sum_left_y += box1.ppoint[i * 2 + 1];
		weight_left += 1;
	}

	float sum_right_x = 0, sum_right_y = 0;
	float weight_right = 0;
	for (int i = 58; i <= 63; i++)
	{
		sum_right_x += box1.ppoint[i * 2];
		sum_right_y += box1.ppoint[i * 2 + 1];
		weight_right += 1;
	}
	for (int i = 75; i <= 77; i++)
	{
		sum_right_x += box1.ppoint[i * 2];
		sum_right_y += box1.ppoint[i * 2 + 1];
		weight_right += 1;
	}
	sum_left_x /= weight_left;
	sum_left_y /= weight_left;
	sum_right_x /= weight_right;
	sum_right_y /= weight_right;
	float dis = sqrt((sum_left_x - sum_right_x)*(sum_left_x - sum_right_x) + (sum_left_y - sum_right_y)*(sum_left_y - sum_right_y));

	int blur_amount = 0.6*dis;
	blur_amount /= 2;
	blur_amount = __max(blur_amount, 1);
	blur_amount *= 2;
	blur_amount += 1;

	Mat im1_blur, im2_blur;
	GaussianBlur(img1, im1_blur, Size(blur_amount, blur_amount), blur_amount / 2, blur_amount / 2);
	GaussianBlur(img2, im2_blur, Size(blur_amount, blur_amount), blur_amount / 2, blur_amount / 2);

	int C = img2.channels();
	int Step2 = img2.step[0];
	int Step_blur1 = im1_blur.step[0];
	int Step_blur2 = im2_blur.step[0];
	for (int h = 0; h < img2.rows; h++)
	{
		for (int w = 0; w < img2.cols; w++)
		{
			for (int c = 0; c < C; c++)
			{
				float val2 = im2_blur.data[h*Step_blur2 + w*C + c];
				float val1 = im1_blur.data[h*Step_blur1 + w*C + c];
				if (val2 <= 1)
					val2 += 128;
				float cur_val = img2.data[h*Step2 + w*C + c];
				correct_im2.data[h*Step2 + w*C + c] = __min(255, __max(0,cur_val*val1 / val2));
			}
		}
	}

}