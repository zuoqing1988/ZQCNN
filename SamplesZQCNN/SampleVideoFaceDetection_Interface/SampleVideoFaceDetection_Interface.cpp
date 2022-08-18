#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_Tensor4D.h"
#include "ZQ_CNN_VideoFaceDetection_Interface.h"
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


void Draw(cv::Mat &image, const std::vector<ZQ_CNN_BBox106>& thirdBbox);



int run_cam()
{
	int num_threads = 1;
#if ZQ_CNN_USE_BLAS_GEMM
	printf("set openblas thread_num = %d\n", num_threads);
	openblas_set_num_threads(num_threads);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(num_threads);
#endif

	std::vector<ZQ_CNN_BBox106> thirdBbox106;
	using ZQ_CNN_VideoFaceDetection = ZQ_CNN_VideoFaceDetection_Interface<ZQ_CNN_Net, ZQ_CNN_Tensor4D_NHW_C_Align128bit, ZQ_CNN_Tensor4D>;
	ZQ_CNN_VideoFaceDetection detector;
	std::string result_name;
	detector.TurnOffShowDebugInfo();
	detector.TurnOffFilterIOU();
	//mtcnn.SetLimit(300, 50, 20);
	int thread_num = 0;

#if defined(_WIN32)
	if (!detector.Init(
		"model/det1-dw20-plus.zqparams", "model/det1-dw20-plus.nchwbin",
		"model/det2-dw24-p0.zqparams", "model/det2-dw24-p0.nchwbin",
		"model/det3-dw48-p0.zqparams", "model/det3-dw48-p0.nchwbin",
		thread_num, true,
		"model/det5-112-gray.zqparams", "model/det5-112-gray.nchwbin",
		true,
		"model/headposegaze-112-gray.zqparams", "model/headposegaze-112-gray.nchwbin"
#else
	if (!detector.Init(
		"../../model/det1-dw20-plus.zqparams", "../../model/det1-dw20-plus.nchwbin",
		"../../model/det2-dw24-plus.zqparams", "../../model/det2-dw24-plus.nchwbin",
		"../../model/det3-dw48-plus.zqparams", "../../model/det3-dw48-plus.nchwbin",
		thread_num, true,
		"../../model/det5-112-gray.zqparams", "../../model/det5-112-gray.nchwbin",
		true,
		"../../model/headposegaze-112-gray.zqparams", "../../model/headposegaze-112-gray.nchwbin"
#endif
	))
	{
		cout << "failed to init!\n";
		return EXIT_FAILURE;
	}

	detector.Message(ZQ_CNN_VideoFaceDetection::VFD_MSG_MAX_TRACE_NUM, 6);
	detector.Message(ZQ_CNN_VideoFaceDetection::VFD_MSG_WEIGHT_DECAY, 0.2);
	//cv::VideoCapture cap("video_20190518_172153_540P.mp4");
	//cv::VideoCapture cap("video_20190612_094223.mp4"); 
	//cv::VideoCapture cap("video_20190702_083029.mp4");
	//cv::VideoCapture cap("V90715-124118.mp4");
	//cv::VideoCapture cap("video_20190528_093054_540P.mp4");
	//cv::VideoCapture cap("video_20190806_190129.mp4");
	//cv::VideoCapture cap("video_20190809_094755.mp4");
	cv::VideoCapture cap(0);
	cv::VideoWriter writer;
	cv::Mat image0, ori_im;
	cv::namedWindow("show");
	while (true)
	{
		cap >> image0;

		if (image0.empty())
			break;
		//printf("w x h = %d x %d\n", image0.cols, image0.rows);
		//cv::flip(image0, image0, 1);
		//image0 = image0(cv::Rect(656, 0, 607, 1080));
		//cv::resize(image0, image0, cv::Size(), 0.5, 0.5);
		//cv::imshow("show_ori", image0);
		//cv::waitKey(0);
		image0.copyTo(ori_im);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		/*cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);
		cv::GaussianBlur(image0, image0, cv::Size(3, 3), 2, 2);*/

		if (!writer.isOpened())
			writer.open("cam-trace6.mp4", CV_FOURCC('X', 'V', 'I', 'D'), 25, cv::Size(image0.cols, image0.rows));
		detector.SetPara(image0.cols, image0.rows, 120, 0.5, 0.6, 0.8, 0.4, 0.5, 0.5, 0.709, 3, 20, 4, 25);

		//mtcnn.TurnOnShowDebugInfo();
		static int fr_id = 0;
		if (!detector.Find(image0.data, image0.cols, image0.rows, image0.step[0], thirdBbox106))
		{
			printf("%d\n", fr_id);
		}

		Draw(ori_im, thirdBbox106);

		imshow("show", ori_im);
		char buf[200];
#if defined(_WIN32)
		sprintf_s(buf, 200, "out5-old\\%d.png", fr_id);
#else
		sprintf(buf, "out5-old\\%d.png", fr_id);
#endif
		//cv::imwrite(buf, ori_im);
		writer << ori_im;
		int key = cv::waitKey(20);
		if (key == 27)
			break;

		fr_id++;
	}

	return EXIT_SUCCESS;
}

int main()
{
	//return run_fig();
	return run_cam();
}


void draw_axis(cv::Mat& image, const float R[9], float fx, float fy, float cx, float cy,
	float pt_x, float pt_y)
{
	float ModelView[16] =
	{
		R[0],R[1],R[2],(pt_x - cx) / fx,
		R[3],R[4],R[5],(cy - pt_y) / fy,
		R[6],R[7],R[8],-1,
		0,0,0,1
	};
	float axis_pts[4][4] =
	{
		{ 0,   0,   0,   1 },
		{ 0.05,   0,   0,   1 },
		{ 0,0.05,   0,   1 },
		{ 0,   0,0.05,   1 }
	};
	float pts2D[4][4];
	for (int i = 0; i < 4; i++)
	{
		ZQ_MathBase::MatrixMul(ModelView, axis_pts[i], 4, 4, 1, pts2D[i]);
		float _x = pts2D[i][0];
		float _y = pts2D[i][1];
		float _z = pts2D[i][2];
		float x = -_x / _z;
		float y = -_y / _z;
		x = x*fx + cx;
		y = -y*fy + cy;//from opengl Y to opencv Y
		pts2D[i][0] = x;
		pts2D[i][1] = y;
	}
	cv::Point pt0(pts2D[0][0], pts2D[0][1]);
	cv::Point pt1(pts2D[1][0], pts2D[1][1]);
	cv::Point pt2(pts2D[2][0], pts2D[2][1]);
	cv::Point pt3(pts2D[3][0], pts2D[3][1]);
	cv::line(image, pt0, pt1, cv::Scalar(0, 0, 255), 2);
	cv::line(image, pt0, pt2, cv::Scalar(0, 255, 0), 2);
	cv::line(image, pt0, pt3, cv::Scalar(255, 0, 0), 2);
}

void YawPitchRoll2Rotation(const float pitch, const float yaw, const float roll, float R[9])
{
	float sinx = sin(pitch);
	float cosx = cos(pitch);
	float siny = sin(yaw);
	float cosy = cos(yaw);
	float sinz = sin(roll);
	float cosz = cos(roll);
	R[0] = cosz*cosy - sinz*sinx*siny;
	R[1] = -sinz*cosx;
	R[2] = cosz*siny + sinz*sinx*cosy;
	R[3] = sinz*cosy + cosz*sinx*siny;
	R[4] = cosz*cosx;
	R[5] = sinz*siny - cosz*sinx*cosy;
	R[6] = -cosx*siny;
	R[7] = sinx;
	R[8] = cosx*cosy;
}

void Draw(cv::Mat &image, const std::vector<ZQ_CNN_BBox106>& thirdBbox)
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


			if (it->has_headposegaze)
			{
				float cx = image.cols / 2;
				float cy = image.rows / 2;
				const float m_pi = 3.1415926535f;
				float fovy = 20.0 / 180.0*m_pi;
				float focal = image.rows / 2.0f / tan(0.5f*fovy);
				float vir_xaxis[3], vir_yaxis[3];
				float vir_zaxis[3] = {
					cx - it->center_and_rot[0],
					-(cy - it->center_and_rot[1]),
					focal
				};
				ZQ_MathBase::Normalize(3, vir_zaxis);
				float vir_yaxis_tmp[3] = {
					sin(-it->center_and_rot[2]),
					cos(-it->center_and_rot[2]),
					0
				};
				ZQ_MathBase::CrossProduct(vir_yaxis_tmp, vir_zaxis, vir_xaxis);
				ZQ_MathBase::Normalize(3, vir_xaxis);
				ZQ_MathBase::CrossProduct(vir_zaxis, vir_xaxis, vir_yaxis);
				float vir_R[9] = {
					vir_xaxis[0],vir_yaxis[0],vir_zaxis[0],
					vir_xaxis[1],vir_yaxis[1],vir_zaxis[1],
					vir_xaxis[2],vir_yaxis[2],vir_zaxis[2]
				};
				/*printf("vir_R=\n");
				for (int h = 0; h < 3; h++)
				{
				for (int w = 0; w < 3; w++)
				{
				printf("%12.5f", vir_R[h * 3 + w]);
				}
				printf("\n");
				}*/


				float R_ori[9], R[9];
				YawPitchRoll2Rotation(it->headposegaze[0], it->headposegaze[1], it->headposegaze[2], R_ori);
				ZQ_MathBase::MatrixMul(vir_R, R_ori, 3, 3, 3, R);
				draw_axis(image, R, focal, focal, cx, cy, it->ppoint[92], it->ppoint[93]);
				{
					float tmp_y[3] = { R_ori[1],R_ori[4],R_ori[7] };
					float leye_z[3] = { it->headposegaze[6], it->headposegaze[7], it->headposegaze[8] };
					float leye_x[3], leye_y[3];
					ZQ_MathBase::CrossProduct(tmp_y, leye_z, leye_x);
					ZQ_MathBase::Normalize(3, leye_x);
					ZQ_MathBase::CrossProduct(leye_z, leye_x, leye_y);
					float left_R_ori[9] = {
						leye_x[0],leye_y[0],leye_z[0],
						leye_x[1],leye_y[1],leye_z[1],
						leye_x[2],leye_y[2],leye_z[2]
					};
					float left_R[9];
					ZQ_MathBase::MatrixMul(vir_R, left_R_ori, 3, 3, 3, left_R);
					float pt0_x = 0.5f*(it->ppoint[144] + it->ppoint[146]);
					float pt0_y = 0.5f*(it->ppoint[145] + it->ppoint[147]);
					draw_axis(image, left_R, focal, focal, cx, cy, pt0_x, pt0_y);
				}
				{
					float tmp_y[3] = { R_ori[1],R_ori[4],R_ori[7] };
					float reye_z[3] = { it->headposegaze[3], it->headposegaze[4], it->headposegaze[5] };
					float reye_x[3], reye_y[3];
					ZQ_MathBase::CrossProduct(tmp_y, reye_z, reye_x);
					ZQ_MathBase::Normalize(3, reye_x);
					ZQ_MathBase::CrossProduct(reye_z, reye_x, reye_y);
					float right_R_ori[9] = {
						reye_x[0],reye_y[0],reye_z[0],
						reye_x[1],reye_y[1],reye_z[1],
						reye_x[2],reye_y[2],reye_z[2]
					};
					float right_R[9];
					ZQ_MathBase::MatrixMul(vir_R, right_R_ori, 3, 3, 3, right_R);
					float pt0_x = 0.5f*(it->ppoint[150] + it->ppoint[152]);
					float pt0_y = 0.5f*(it->ppoint[151] + it->ppoint[153]);
					draw_axis(image, right_R, focal, focal, cx, cy, pt0_x, pt0_y);
				}
			}
		}
		else
		{
			printf("not exist!\n");
		}
	}
}