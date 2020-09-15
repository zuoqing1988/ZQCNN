#ifndef _ZQ_RECTIFY_CARD_H_
#define _ZQ_RECTIFY_CARD_H_
#pragma once

#include <opencv2/opencv.hpp>

#include <time.h>
#include <iostream>
#include <vector>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ZQ_Vec2D.h"
#include "ZQ_Ray2D.h"
#include "ZQ_SVD.h"
#include "ZQ_LSD.h"

namespace ZQ
{
	class ZQ_RectifyCard
	{
		class ZQ_LineSeg
		{
		public:
			ZQ_Vec2D pt0, pt1;

			float Length() const
			{
				float x = pt1.x - pt0.x;
				float y = pt1.y - pt0.y;
				return sqrt(x*x + y*y);
			}

			float Angle() const
			{
				const double m_pi = 3.1415926535;
				float x = pt1.x - pt0.x;
				float y = pt1.y - pt0.y;
				if (x >= 0)
				{
					return atan2(y, x) / m_pi * 180;
				}
				else
				{
					return atan2(-y, -x) / m_pi * 180;
				}
			}

			float DistanceTo(const ZQ_Vec2D pt) const
			{
				ZQ_Vec2D dir01(pt1.x - pt0.x, pt1.y - pt0.y);
				if (!dir01.Normalized())
					return 0;
				else
				{
					ZQ_Vec2D dir02(pt.x - pt0.x, pt.y - pt0.y);
					float proj_len = dir02.DotProduct(dir01);
					ZQ_Vec2D dir02_p = dir02 - dir01*proj_len;
					return dir02_p.Length();
				}
			}
		};

		class ZQ_LineSegWithInfo
		{
		public:
			ZQ_LineSeg line;
			float angle;
			float length;
			float distance;

			void UpdateInfo(const ZQ_Vec2D center)
			{
				angle = line.Angle();
				length = line.Length();
				distance = line.DistanceTo(center);
			}
		};

	private:
		static bool _is_in_rect(cv::Rect& rect, ZQ_Vec2D& pt)
		{
			return pt.x >= rect.x && pt.x < rect.x + rect.width && pt.y >= rect.y && pt.y < rect.y + rect.height;
		}

		static void _choose_line_by_angle(const std::vector<ZQ_LineSegWithInfo>& input, std::vector<ZQ_LineSegWithInfo>& output, int angle_range = 2)
		{
			float bucket[360] = { 0 };
			const int range = abs(angle_range);

			for (int i = 0; i < input.size(); i++)
			{
				float cur_angle = input[i].angle;
				float cur_len = input[i].length;
				for (int j = -range; j <= range; j++)
				{
					int tmp_angle = cur_angle + j + 0.5;
					tmp_angle = (tmp_angle + 360) % 360;
					bucket[tmp_angle] += cur_len;
				}
			}

			int max_angle = -1;
			float max_bucket = -1;
			for (int i = 0; i < 360; i++)
			{
				if (max_bucket < bucket[i])
				{
					max_bucket = bucket[i];
					max_angle = i;
				}
			}

			output.clear();
			for (int i = 0; i < input.size(); i++)
			{
				float cur_angle = input[i].angle;
				int cur_i = cur_angle + 0.5;
				if (abs((cur_i - max_angle + 360) % 360) <= range)
					output.push_back(input[i]);
			}
		}

		static void _choose_line_by_distance(const std::vector<ZQ_LineSegWithInfo>& input, std::vector<ZQ_LineSegWithInfo>& output, int dis_range = 1)
		{
			//printf("!\n");
			float max_distance = -1;
			float min_distance = 1e9;
			for (int i = 0; i < input.size(); i++)
			{
				float cur_dis = input[i].distance;
				max_distance = __max(max_distance, cur_dis);
				min_distance = __min(min_distance, cur_dis);
			}

			int min_dis_i = floor(min_distance);
			int max_dis_i = ceil(max_distance);
			int num_bucket = max_dis_i - min_dis_i + 1;
			//printf("!!\n");
			//printf("num_bucket = %d\n", num_bucket);
			std::vector<double> bucket(num_bucket, 0);
			const int range = abs(dis_range);
			//printf("!!!\n");
			for (int i = 0; i < input.size(); i++)
			{
				float cur_dis = input[i].distance;
				float cur_len = input[i].length;
				for (int j = -range; j <= range; j++)
				{
					int tmp_dis = cur_dis + j + 0.5;
					if (tmp_dis >= min_dis_i && tmp_dis <= max_dis_i)
					{
						bucket[tmp_dis - min_dis_i] += cur_len;
					}
				}
			}
			//printf("!!!!\n");
			int max_dis = -1;
			float max_bucket = -1;
			for (int i = 0; i < num_bucket; i++)
			{
				if (max_bucket < bucket[i])
				{
					max_bucket = bucket[i];
					max_dis = i + min_dis_i;
				}
			}
			//printf("!!!!!\n");
			output.clear();
			for (int i = 0; i < input.size(); i++)
			{
				float cur_dis = input[i].distance;
				int cur_dis_i = cur_dis + 0.5;
				if (abs(cur_dis_i - max_dis) <= range)
				{
					output.push_back(input[i]);
				}
			}
		}

		static bool _regress_line(const std::vector<ZQ_LineSegWithInfo>& input, ZQ_Ray2D& ray)
		{
			if (input.size() == 0)
				return false;
			if (input.size() == 1)
			{
				ray.origin = (input[0].line.pt0 + input[0].line.pt1) *0.5;
				ray.dir = input[0].line.pt1 - input[0].line.pt0;
				return ray.dir.Normalized();
			}
			else
			{
				ray.origin = ZQ_Vec2D(0, 0);
				ray.dir = ZQ_Vec2D(0, 0);
				float sum_weight = 0;
				for (int i = 0; i < input.size(); i++)
				{
					int cur_weight = input[i].length;
					ray.origin += input[i].line.pt0 * cur_weight;
					ray.origin += input[i].line.pt1 * cur_weight;
					sum_weight += cur_weight;
				}
				ray.origin *= 1.0 / (2.0*sum_weight);
				ZQ_Matrix<double> A(2 * input.size(), 2);
				for (int i = 0; i < input.size(); i++)
				{
					ZQ_Vec2D dir0 = input[i].line.pt0 - ray.origin;
					ZQ_Vec2D dir1 = input[i].line.pt1 - ray.origin;
					A.SetData(i * 2, 0, dir0.x);
					A.SetData(i * 2, 1, dir0.y);
					A.SetData(i * 2 + 1, 0, dir1.x);
					A.SetData(i * 2 + 1, 1, dir1.y);
				}
				ZQ_Matrix<double> u(2 * input.size(), 2), s(2, 2), v(2, 2);
				if (!ZQ_SVD::Decompose(A, u, s, v))
					return false;
				const double* ptr = v.GetDataPtr();
				ray.dir.x = ptr[0];
				ray.dir.y = ptr[2];
				return true;
			}
		}

		static cv::Mat _find_translation_scale_x(const std::vector<cv::Point2f>& src_pts, const std::vector<cv::Point2f>& dst_pts, bool& ret)
		{
			int n_pts = src_pts.size();
			ZQ_Matrix<double> Ax(n_pts, 2), Xx(2, 1), Bx(n_pts, 1);
			for (int i = 0; i < n_pts; i++)
			{
				Ax.SetData(i, 0, src_pts[i].x);
				Ax.SetData(i, 1, 1);
				Bx.SetData(i, 0, dst_pts[i].x);
			}

			float sx = 1, tx = 0;
			float sy = 1, ty = 0;
			ret = false;
			if (ZQ_SVD::Solve(Ax, Xx, Bx))
			{
				bool flag;
				sx = Xx.GetData(0, 0, flag);
				tx = Xx.GetData(1, 0, flag);
				ret = true;
			}

			cv::Mat tranform_mat = cv::Mat(3, 3, CV_64FC1);
			tranform_mat.ptr<double>(0)[0] = sx;
			tranform_mat.ptr<double>(0)[1] = 0;
			tranform_mat.ptr<double>(0)[2] = tx;
			tranform_mat.ptr<double>(1)[0] = 0;
			tranform_mat.ptr<double>(1)[1] = sy;
			tranform_mat.ptr<double>(1)[2] = ty;
			tranform_mat.ptr<double>(2)[0] = 0;
			tranform_mat.ptr<double>(2)[1] = 0;
			tranform_mat.ptr<double>(2)[2] = 1;
			return tranform_mat;
		}

		static cv::Mat _find_translation_scale_y(const std::vector<cv::Point2f>& src_pts, const std::vector<cv::Point2f>& dst_pts, bool& ret)
		{
			int n_pts = src_pts.size();
			ZQ_Matrix<double> Ay(n_pts, 2), Xy(2, 1), By(n_pts, 1);
			for (int i = 0; i < n_pts; i++)
			{
				Ay.SetData(i, 0, src_pts[i].y);
				Ay.SetData(i, 1, 1);
				By.SetData(i, 0, dst_pts[i].y);
			}

			float sx = 1, tx = 0;
			float sy = 1, ty = 0;
			ret = false;
			if (ZQ_SVD::Solve(Ay, Xy, By))
			{
				bool flag;
				sy = Xy.GetData(0, 0, flag);
				ty = Xy.GetData(1, 0, flag);
				ret = true;
			}

			cv::Mat tranform_mat = cv::Mat(3, 3, CV_64FC1);
			tranform_mat.ptr<double>(0)[0] = sx;
			tranform_mat.ptr<double>(0)[1] = 0;
			tranform_mat.ptr<double>(0)[2] = tx;
			tranform_mat.ptr<double>(1)[0] = 0;
			tranform_mat.ptr<double>(1)[1] = sy;
			tranform_mat.ptr<double>(1)[2] = ty;
			tranform_mat.ptr<double>(2)[0] = 0;
			tranform_mat.ptr<double>(2)[1] = 0;
			tranform_mat.ptr<double>(2)[2] = 1;
			return tranform_mat;
		}

		static cv::Mat _mul_two_matrix(const cv::Mat& mat1, const cv::Mat& mat2)
		{
			cv::Mat out(3, 3, CV_64FC1);
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					double sum = 0;
					for (int k = 0; k < 3; k++)
					{
						sum += mat1.ptr<double>(i)[k] * mat2.ptr<double>(k)[j];
					}
					out.ptr<double>(i)[j] = sum;
				}
			}
			return out;
		}

		static void _transform_lines(const cv::Mat& trans_mat, const std::vector<ZQ_Vec2D>& lines, cv::Rect roi, std::vector<ZQ_Vec2D>& out_lines)
		{
			out_lines.clear();
			int line_num = lines.size() / 2;
			if (line_num == 0)
				return ;
			int nPts = line_num * 2;
			std::vector<double> in_buffer(nPts * 3);
			std::vector<double> out_buffer(nPts * 3);
			double mat[9] = 
			{
				trans_mat.ptr<double>(0)[0],trans_mat.ptr<double>(0)[1],trans_mat.ptr<double>(0)[2],
				trans_mat.ptr<double>(1)[0],trans_mat.ptr<double>(1)[1],trans_mat.ptr<double>(1)[2],
				trans_mat.ptr<double>(2)[0],trans_mat.ptr<double>(2)[1],trans_mat.ptr<double>(2)[2]
			};
			//std::cout << trans_mat << std::endl;

			for (int i = 0; i < nPts; i++)
			{
				in_buffer[0 * nPts + i] = lines[i].x;
				in_buffer[1 * nPts + i] = lines[i].y;
				in_buffer[2 * nPts + i] = 1;
			}
			ZQ_MathBase::MatrixMul(mat, &in_buffer[0], 3, 3, line_num * 2, &out_buffer[0]);
			for (int i = 0; i < line_num; i++)
			{
				float x0 = out_buffer[0 * nPts + i * 2] / out_buffer[2 * nPts + i * 2] - roi.x;
				float y0 = out_buffer[1 * nPts + i * 2] / out_buffer[2 * nPts + i * 2] - roi.y;
				float x1 = out_buffer[0 * nPts + i * 2 + 1] / out_buffer[2 * nPts + i * 2 + 1] - roi.x;
				float y1 = out_buffer[1 * nPts + i * 2 + 1] / out_buffer[2 * nPts + i * 2 + 1] - roi.y;
				if (x0 >= 0 && x0 < roi.width
					&& y0 >= 0 && y0 < roi.height
					&& x1 >= 0 && x1 < roi.width
					&& y1 >= 0 && y1 < roi.height)
				{
					out_lines.push_back(ZQ_Vec2D(x0, y0));
					out_lines.push_back(ZQ_Vec2D(x1, y1));
				}
			}
		}

		static void _detect_lines(const cv::Mat& im_gray, std::vector<ZQ_Vec2D>& lines)
		{
			ZQ_LSD::ntuple_list detected_lines;
			ZQ_LSD::image_double  image = ZQ_LSD::new_image_double(im_gray.cols, im_gray.rows);
			uchar* im_src = (uchar*)im_gray.data;
			int xsize = image->xsize;
			int ysize = image->ysize;

			for (int y = 0; y < ysize; y++)
			{
				for (int x = 0; x < xsize; x++)
				{
					image->data[y*xsize + x] = im_src[y*im_gray.step[0] + x];
				}
			}
			detected_lines = ZQ_LSD::lsd(image);
			ZQ_LSD::free_image_double(image);
			
			int dim = detected_lines->dim;
			lines.clear();
			for (unsigned int j = 0; j < detected_lines->size; j++)
			{
				lines.push_back(ZQ_Vec2D(detected_lines->values[j*dim + 0], detected_lines->values[j*dim + 1]));
				lines.push_back(ZQ_Vec2D(detected_lines->values[j*dim + 2], detected_lines->values[j*dim + 3]));
			}
			ZQ_LSD::free_ntuple_list(detected_lines);
		}

		static bool _detectRectCorners(const cv::Mat& show_debug, std::vector<ZQ_Vec2D>& lines, int srcWidth, int srcHeight, 
			ZQ_Vec2D& pt_lt, ZQ_Vec2D& pt_lb, ZQ_Vec2D& pt_rt, ZQ_Vec2D& pt_rb,
			int angle_range = 2, int dis_range = 0, bool display = false, int border_size = -1, int min_len = 10)
		{
			int border_x = srcWidth / 2;
			int border_y = srcHeight / 2;
			if (border_size > 0)
			{
				border_x = border_size;
				border_y = border_size;
			}
			ZQ_Vec2D center(srcWidth*0.5, srcHeight*0.5);
			cv::Rect rectL(0, 0, border_x, srcHeight);
			cv::Rect rectR(srcWidth - 1 - border_x, 0, border_x, srcHeight);
			cv::Rect rectT(0, 0, srcWidth, border_y);
			cv::Rect rectB(0, srcHeight - 1 - border_y, srcWidth, border_y);
			std::vector<ZQ_LineSegWithInfo> lineTop, lineBottom, lineLeft, lineRight;

			int line_num = lines.size() / 2;
			for (unsigned int i = 0; i < line_num; i++)
			{
				ZQ_LineSegWithInfo lineseg;
				ZQ_Vec2D& pt0 = lineseg.line.pt0;
				ZQ_Vec2D& pt1 = lineseg.line.pt1;
				pt0 = lines[i*2];
				pt1 = lines[i * 2 + 1];
				lineseg.UpdateInfo(center);
				if (lineseg.length > min_len)
				{
					if (_is_in_rect(rectL, pt0) && _is_in_rect(rectL, pt1)
						&& (lineseg.angle < -70 || lineseg.angle > 70))
					{
						lineLeft.push_back(lineseg);
					}

					if (_is_in_rect(rectR, pt0) && _is_in_rect(rectR, pt1)
						&& (lineseg.angle < -70 || lineseg.angle > 70))
					{
						lineRight.push_back(lineseg);

					}

					if (_is_in_rect(rectT, pt0) && _is_in_rect(rectT, pt1)
						&& (lineseg.angle > -20 && lineseg.angle < 20))
					{
						lineTop.push_back(lineseg);
					}

					if (_is_in_rect(rectB, pt0) && _is_in_rect(rectB, pt1)
						&& (lineseg.angle > -20 && lineseg.angle < 20))
					{
						lineBottom.push_back(lineseg);
					}
				}
			}

			if (display)
			{
				cv::Mat show;
				if (show_debug.cols != srcWidth || show_debug.rows != srcHeight)
					show = cv::Mat(srcHeight, srcWidth, CV_MAKE_TYPE(8, 3));
				else
					show_debug.copyTo(show);
				for (int i = 0; i < lineTop.size(); i++)
				{
					ZQ_Vec2D& pt0 = lineTop[i].line.pt0;
					ZQ_Vec2D& pt1 = lineTop[i].line.pt1;
					cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
				}
				for (int i = 0; i < lineBottom.size(); i++)
				{
					ZQ_Vec2D& pt0 = lineBottom[i].line.pt0;
					ZQ_Vec2D& pt1 = lineBottom[i].line.pt1;
					cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
				}
				for (int i = 0; i < lineLeft.size(); i++)
				{
					ZQ_Vec2D& pt0 = lineLeft[i].line.pt0;
					ZQ_Vec2D& pt1 = lineLeft[i].line.pt1;
					cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
				}
				for (int i = 0; i < lineRight.size(); i++)
				{
					ZQ_Vec2D& pt0 = lineRight[i].line.pt0;
					ZQ_Vec2D& pt1 = lineRight[i].line.pt1;
					cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
				}
				cv::namedWindow("show1");
				cv::imshow("show1", show);
				cv::waitKey(0);
			}

			if (lineTop.size() == 0 || lineBottom.size() == 0 || lineLeft.size() == 0 || lineRight.size() == 0)
				return false;

			std::vector<ZQ_LineSegWithInfo> lineT, lineB, lineL, lineR;
			_choose_line_by_angle(lineTop, lineT, angle_range);
			_choose_line_by_angle(lineBottom, lineB, angle_range);
			_choose_line_by_angle(lineLeft, lineL, angle_range);
			_choose_line_by_angle(lineRight, lineR, angle_range);

			//printf("hello\n");
			lineTop.swap(lineT);
			lineBottom.swap(lineB);
			lineLeft.swap(lineL);
			lineRight.swap(lineR);
			//printf("hello2\n");
			
			if (display)
			{
				cv::Mat show;
				if (show_debug.cols != srcWidth || show_debug.rows != srcHeight)
					show = cv::Mat(srcHeight, srcWidth, CV_MAKE_TYPE(8, 3));
				else
					show_debug.copyTo(show);
				for (int i = 0; i < lineTop.size(); i++)
				{
					ZQ_Vec2D& pt0 = lineTop[i].line.pt0;
					ZQ_Vec2D& pt1 = lineTop[i].line.pt1;
					cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
				}
				for (int i = 0; i < lineBottom.size(); i++)
				{
					ZQ_Vec2D& pt0 = lineBottom[i].line.pt0;
					ZQ_Vec2D& pt1 = lineBottom[i].line.pt1;
					cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
				}
				for (int i = 0; i < lineLeft.size(); i++)
				{
					ZQ_Vec2D& pt0 = lineLeft[i].line.pt0;
					ZQ_Vec2D& pt1 = lineLeft[i].line.pt1;
					cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
				}
				for (int i = 0; i < lineRight.size(); i++)
				{
					ZQ_Vec2D& pt0 = lineRight[i].line.pt0;
					ZQ_Vec2D& pt1 = lineRight[i].line.pt1;
					cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
				}
				cv::namedWindow("show2");
				cv::imshow("show2", show);
				cv::waitKey(0);
			}

			if (lineTop.size() == 0 || lineBottom.size() == 0 || lineLeft.size() == 0 || lineRight.size() == 0)
				return false;

			_choose_line_by_distance(lineTop, lineT, dis_range);
			_choose_line_by_distance(lineBottom, lineB, dis_range);
			_choose_line_by_distance(lineLeft, lineL, dis_range);
			_choose_line_by_distance(lineRight, lineR, dis_range);
			
			if (display)
			{
				cv::Mat show;
				if (show_debug.cols != srcWidth || show_debug.rows != srcHeight)
					show = cv::Mat(srcHeight, srcWidth, CV_MAKE_TYPE(8, 3));
				else
					show_debug.copyTo(show);
				for (int i = 0; i < lineT.size(); i++)
				{
					ZQ_Vec2D& pt0 = lineT[i].line.pt0;
					ZQ_Vec2D& pt1 = lineT[i].line.pt1;
					cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
				}
				for (int i = 0; i < lineB.size(); i++)
				{
					ZQ_Vec2D& pt0 = lineB[i].line.pt0;
					ZQ_Vec2D& pt1 = lineB[i].line.pt1;
					cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
				}
				for (int i = 0; i < lineL.size(); i++)
				{
					ZQ_Vec2D& pt0 = lineL[i].line.pt0;
					ZQ_Vec2D& pt1 = lineL[i].line.pt1;
					cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
				}
				for (int i = 0; i < lineR.size(); i++)
				{
					ZQ_Vec2D& pt0 = lineR[i].line.pt0;
					ZQ_Vec2D& pt1 = lineR[i].line.pt1;
					cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
				}
				cv::namedWindow("show3");
				cv::imshow("show3", show);
				cv::waitKey(0);
			}

			if (lineT.size() == 0 || lineB.size() == 0 || lineL.size() == 0 || lineR.size() == 0)
				return false;

			

			/* line regression */
			ZQ_Ray2D rayT, rayB, rayL, rayR;
			_regress_line(lineT, rayT);
			_regress_line(lineB, rayB);
			_regress_line(lineL, rayL);
			_regress_line(lineR, rayR);

			if (display)
			{
				cv::Mat show;
				if (show_debug.cols != srcWidth || show_debug.rows != srcHeight)
					show = cv::Mat(srcHeight, srcWidth, CV_MAKE_TYPE(8, 3));
				else
					show_debug.copyTo(show);
				float max_ray_len = srcWidth + srcHeight;
				cv::line(show, cv::Point(rayT.origin.x - max_ray_len*rayT.dir.x, rayT.origin.y - max_ray_len*rayT.dir.y),
					cv::Point(rayT.origin.x + max_ray_len*rayT.dir.x, rayT.origin.y + max_ray_len*rayT.dir.y),
					cv::Scalar(0, 255, 255), 1, CV_AA);
				cv::line(show, cv::Point(rayB.origin.x - max_ray_len*rayB.dir.x, rayB.origin.y - max_ray_len*rayB.dir.y),
					cv::Point(rayB.origin.x + max_ray_len*rayB.dir.x, rayB.origin.y + max_ray_len*rayB.dir.y),
					cv::Scalar(0, 255, 255), 1, CV_AA);
				cv::line(show, cv::Point(rayL.origin.x - max_ray_len*rayL.dir.x, rayL.origin.y - max_ray_len*rayL.dir.y),
					cv::Point(rayL.origin.x + max_ray_len*rayL.dir.x, rayL.origin.y + max_ray_len*rayL.dir.y),
					cv::Scalar(0, 255, 255), 1, CV_AA);
				cv::line(show, cv::Point(rayR.origin.x - max_ray_len*rayR.dir.x, rayR.origin.y - max_ray_len*rayR.dir.y),
					cv::Point(rayR.origin.x + max_ray_len*rayR.dir.x, rayR.origin.y + max_ray_len*rayR.dir.y),
					cv::Scalar(0, 255, 255), 1, CV_AA);
				cv::namedWindow("show4");
				cv::imshow("show4", show);
				cv::waitKey(0);
			}

			/* get corners */

			ZQ_Vec2D corner_LT, corner_LB, corner_RT, corner_RB;
			float depth1, depth2;
			ZQ_Ray2D::RayCross(rayL, rayT, depth1, depth2, pt_lt);
			ZQ_Ray2D::RayCross(rayL, rayB, depth1, depth2, pt_lb);
			ZQ_Ray2D::RayCross(rayR, rayT, depth1, depth2, pt_rt);
			ZQ_Ray2D::RayCross(rayR, rayB, depth1, depth2, pt_rb);

			if (display)
			{
				cv::destroyWindow("show1");
				cv::destroyWindow("show2");
				cv::destroyWindow("show3");
				cv::destroyWindow("show4");
			}

			return true;
		}

	public:
		
		static bool DetectRectCorners(const cv::Mat& src, ZQ_Vec2D& pt_lt, ZQ_Vec2D& pt_lb, ZQ_Vec2D& pt_rt, ZQ_Vec2D& pt_rb,
			int angle_range = 2, int dis_range = 0, bool display = false, int border_size = -1, int min_len = 10)
		{
			std::vector<ZQ_Vec2D> lines;
			clock_t start = clock();
			cv::Mat im_gray;
			cv::cvtColor(src, im_gray, CV_BGR2GRAY);
			_detect_lines(im_gray, lines);
			clock_t finish = clock();
			double duration = (double)(finish - start) / CLOCKS_PER_SEC;
			if (display)
			{
				std::cout << "total time of extract lines is:" << duration << std::endl;
			}

			return _detectRectCorners(src, lines, src.cols, src.rows, pt_lt, pt_lb, pt_rt, pt_rb, angle_range, dis_range, display, border_size, min_len);
			//cv::Mat show;
			//if (display)
			//	src.copyTo(show);

			//int srcWidth = src.cols;
			//int srcHeight = src.rows;
			//int border_x = srcWidth / 2;
			//int border_y = srcHeight / 2;
			//if (border_size > 0)
			//{
			//	border_x = border_size;
			//	border_y = border_size;
			//}
			//ZQ_Vec2D center(srcWidth*0.5, srcHeight*0.5);
			//cv::Rect rectL(0, 0, border_x, srcHeight);
			//cv::Rect rectR(srcWidth - 1 - border_x, 0, border_x, srcHeight);
			//cv::Rect rectT(0, 0, srcWidth, border_y);
			//cv::Rect rectB(0, srcHeight - 1 - border_y, srcWidth, border_y);
			//std::vector<ZQ_LineSegWithInfo> lineTop, lineBottom, lineLeft, lineRight;

			//for (unsigned int j = 0; j < detected_lines->size; j++)
			//{
			//	ZQ_LineSegWithInfo lineseg;
			//	ZQ_Vec2D& pt0 = lineseg.line.pt0;
			//	ZQ_Vec2D& pt1 = lineseg.line.pt1;
			//	pt0 = ZQ_Vec2D(detected_lines->values[j*dim + 0], detected_lines->values[j*dim + 1]);
			//	pt1 = ZQ_Vec2D(detected_lines->values[j*dim + 2], detected_lines->values[j*dim + 3]);
			//	lineseg.UpdateInfo(center);
			//	if(display)
			//		cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 200, 0), 1, CV_AA);
			//	if (lineseg.length > min_len)
			//	{
			//		if (_is_in_rect(rectL, pt0) && _is_in_rect(rectL, pt1)
			//			&& (lineseg.angle < -70 || lineseg.angle > 70))
			//		{
			//			if (display)
			//			{
			//				cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
			//			}
			//			lineLeft.push_back(lineseg);
			//		}

			//		if (_is_in_rect(rectR, pt0) && _is_in_rect(rectR, pt1)
			//			&& (lineseg.angle < -70 || lineseg.angle > 70))
			//		{
			//			if (display)
			//			{
			//				cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
			//			}
			//			lineRight.push_back(lineseg);

			//		}

			//		if (_is_in_rect(rectT, pt0) && _is_in_rect(rectT, pt1)
			//			&& (lineseg.angle > -20 && lineseg.angle < 20))
			//		{
			//			if (display)
			//			{
			//				cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
			//			}
			//			lineTop.push_back(lineseg);
			//		}

			//		if (_is_in_rect(rectB, pt0) && _is_in_rect(rectB, pt1)
			//			&& (lineseg.angle > -20 && lineseg.angle < 20))
			//		{
			//			if (display)
			//			{
			//				cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
			//			}
			//			lineBottom.push_back(lineseg);
			//		}
			//	}
			//}
			//ZQ_LSD::free_ntuple_list(detected_lines);



			//if (display)
			//{
			//	cv::namedWindow("show1");
			//	cv::imshow("show1", show);
			//	cv::waitKey(0);
			//}

			//if (lineTop.size() == 0 || lineBottom.size() == 0 || lineLeft.size() == 0 || lineRight.size() == 0)
			//	return false;

			//std::vector<ZQ_LineSegWithInfo> lineT, lineB, lineL, lineR;
			//_choose_line_by_angle(lineTop, lineT, angle_range);
			//_choose_line_by_angle(lineBottom, lineB, angle_range);
			//_choose_line_by_angle(lineLeft, lineL, angle_range);
			//_choose_line_by_angle(lineRight, lineR, angle_range);

			//if (display)
			//{
			//	src.copyTo(show);
			//	for (int i = 0; i < lineT.size(); i++)
			//	{
			//		ZQ_Vec2D& pt0 = lineT[i].line.pt0;
			//		ZQ_Vec2D& pt1 = lineT[i].line.pt1;
			//		cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
			//	}
			//	for (int i = 0; i < lineB.size(); i++)
			//	{
			//		ZQ_Vec2D& pt0 = lineB[i].line.pt0;
			//		ZQ_Vec2D& pt1 = lineB[i].line.pt1;
			//		cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
			//	}
			//	for (int i = 0; i < lineL.size(); i++)
			//	{
			//		ZQ_Vec2D& pt0 = lineL[i].line.pt0;
			//		ZQ_Vec2D& pt1 = lineL[i].line.pt1;
			//		cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
			//	}
			//	for (int i = 0; i < lineR.size(); i++)
			//	{
			//		ZQ_Vec2D& pt0 = lineR[i].line.pt0;
			//		ZQ_Vec2D& pt1 = lineR[i].line.pt1;
			//		cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
			//	}
			//	cv::namedWindow("show2");
			//	cv::imshow("show2", show);
			//	cv::waitKey(0);
			//}
			////printf("hello\n");
			//lineTop.swap(lineT);
			//lineBottom.swap(lineB);
			//lineLeft.swap(lineL);
			//lineRight.swap(lineR);
			////printf("hello2\n");

			//if (lineTop.size() == 0 || lineBottom.size() == 0 || lineLeft.size() == 0 || lineRight.size() == 0)
			//	return false;

			//_choose_line_by_distance(lineTop, lineT, dis_range);
			//_choose_line_by_distance(lineBottom, lineB, dis_range);
			//_choose_line_by_distance(lineLeft, lineL, dis_range);
			//_choose_line_by_distance(lineRight, lineR, dis_range);
			////printf("hello3\n");
			//if (display)
			//{
			//	src.copyTo(show);
			//	for (int i = 0; i < lineT.size(); i++)
			//	{
			//		ZQ_Vec2D& pt0 = lineT[i].line.pt0;
			//		ZQ_Vec2D& pt1 = lineT[i].line.pt1;
			//		cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
			//	}
			//	for (int i = 0; i < lineB.size(); i++)
			//	{
			//		ZQ_Vec2D& pt0 = lineB[i].line.pt0;
			//		ZQ_Vec2D& pt1 = lineB[i].line.pt1;
			//		cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
			//	}
			//	for (int i = 0; i < lineL.size(); i++)
			//	{
			//		ZQ_Vec2D& pt0 = lineL[i].line.pt0;
			//		ZQ_Vec2D& pt1 = lineL[i].line.pt1;
			//		cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
			//	}
			//	for (int i = 0; i < lineR.size(); i++)
			//	{
			//		ZQ_Vec2D& pt0 = lineR[i].line.pt0;
			//		ZQ_Vec2D& pt1 = lineR[i].line.pt1;
			//		cv::line(show, cv::Point(pt0.x, pt0.y), cv::Point(pt1.x, pt1.y), cv::Scalar(0, 255, 0), 2, CV_AA);
			//	}
			//	cv::namedWindow("show3");
			//	cv::imshow("show3", show);
			//	cv::waitKey(0);
			//}

			//if (lineT.size() == 0 || lineB.size() == 0 || lineL.size() == 0 || lineR.size() == 0)
			//	return false;

			///* line regression */
			//ZQ_Ray2D rayT, rayB, rayL, rayR;
			//_regress_line(lineT, rayT);
			//_regress_line(lineB, rayB);
			//_regress_line(lineL, rayL);
			//_regress_line(lineR, rayR);

			//if (display)
			//{
			//	src.copyTo(show);
			//	float max_ray_len = srcWidth + srcHeight;
			//	cv::line(show, cv::Point(rayT.origin.x - max_ray_len*rayT.dir.x, rayT.origin.y - max_ray_len*rayT.dir.y),
			//		cv::Point(rayT.origin.x + max_ray_len*rayT.dir.x, rayT.origin.y + max_ray_len*rayT.dir.y),
			//		cv::Scalar(0, 255, 255), 1, CV_AA);
			//	cv::line(show, cv::Point(rayB.origin.x - max_ray_len*rayB.dir.x, rayB.origin.y - max_ray_len*rayB.dir.y),
			//		cv::Point(rayB.origin.x + max_ray_len*rayB.dir.x, rayB.origin.y + max_ray_len*rayB.dir.y),
			//		cv::Scalar(0, 255, 255), 1, CV_AA);
			//	cv::line(show, cv::Point(rayL.origin.x - max_ray_len*rayL.dir.x, rayL.origin.y - max_ray_len*rayL.dir.y),
			//		cv::Point(rayL.origin.x + max_ray_len*rayL.dir.x, rayL.origin.y + max_ray_len*rayL.dir.y),
			//		cv::Scalar(0, 255, 255), 1, CV_AA);
			//	cv::line(show, cv::Point(rayR.origin.x - max_ray_len*rayR.dir.x, rayR.origin.y - max_ray_len*rayR.dir.y),
			//		cv::Point(rayR.origin.x + max_ray_len*rayR.dir.x, rayR.origin.y + max_ray_len*rayR.dir.y),
			//		cv::Scalar(0, 255, 255), 1, CV_AA);
			//	cv::namedWindow("show4");
			//	cv::imshow("show4", show);
			//	cv::waitKey(0);
			//}

			///* get corners */

			//ZQ_Vec2D corner_LT, corner_LB, corner_RT, corner_RB;
			//float depth1, depth2;
			//ZQ_Ray2D::RayCross(rayL, rayT, depth1, depth2, pt_lt);
			//ZQ_Ray2D::RayCross(rayL, rayB, depth1, depth2, pt_lb);
			//ZQ_Ray2D::RayCross(rayR, rayT, depth1, depth2, pt_rt);
			//ZQ_Ray2D::RayCross(rayR, rayB, depth1, depth2, pt_rb);
			//if (display)
			//{
			//	cv::destroyWindow("show1");
			//	cv::destroyWindow("show2");
			//	cv::destroyWindow("show3");
			//	cv::destroyWindow("show4");
			//}
			return true;
		}


		static bool RectifyDriverCard(const cv::Mat& src, int dstWidth, int dstHeight, cv::Mat& dst, cv::Mat& transmtx,
			int border_width = 50, bool display = false)
		{
			int srcWidth = src.cols;
			int srcHeight = src.rows;
			std::vector<ZQ_Vec2D> lines;
			clock_t start = clock();
			cv::Mat im_gray;
			cv::cvtColor(src, im_gray, CV_BGR2GRAY);
			_detect_lines(im_gray, lines);
			clock_t finish = clock();
			double duration = (double)(finish - start) / CLOCKS_PER_SEC;
			if (display)
			{
				std::cout << "total time of extract lines is:" << duration << std::endl;
				cv::Mat show;
				src.copyTo(show);
				for (int i = 0; i < lines.size() / 2; i++)
				{
					float x0 = lines[i * 2 + 0].x;
					float y0 = lines[i * 2 + 0].y;
					float x1 = lines[i * 2 + 1].x;
					float y1 = lines[i * 2 + 1].y;
					cv::line(show, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 1, CV_AA);
				}
				cv::Rect rectL(0, 0, border_width, srcHeight);
				cv::Rect rectR(srcWidth - 1 - border_width, 0, border_width, srcHeight);
				cv::Rect rectT(0, 0, srcWidth, border_width);
				cv::Rect rectB(0, srcHeight - 1 - border_width, srcWidth, border_width);
				cv::rectangle(show, rectL, cv::Scalar(0, 0, 255));
				cv::rectangle(show, rectR, cv::Scalar(0, 0, 255));
				cv::rectangle(show, rectT, cv::Scalar(0, 0, 255));
				cv::rectangle(show, rectB, cv::Scalar(0, 0, 255));
				cv::namedWindow("show");
				cv::imshow("show", show);
				cv::waitKey(0);
			}

			/* get corners */
			ZQ_Vec2D corner_LT, corner_LB, corner_RT, corner_RB;

			if (!_detectRectCorners(src, lines, srcWidth, srcHeight, corner_LT, corner_LB, corner_RT, corner_RB, 2, 1, display, border_width))
				return false;

			float standard_width = 440, standard_height = 300;
			float rects[12] =
			{
				12,145,140,270,
				175,210,325,260,
				120,43,185,86
			};
			
			std::vector<cv::Point2f> src_pts, dst_pts;
			src_pts.push_back(cv::Point2f(corner_LT.x, corner_LT.y));
			src_pts.push_back(cv::Point2f(corner_LB.x, corner_LB.y));
			src_pts.push_back(cv::Point2f(corner_RT.x, corner_RT.y));
			src_pts.push_back(cv::Point2f(corner_RB.x, corner_RB.y));
			dst_pts.push_back(cv::Point2f(0, 0));
			dst_pts.push_back(cv::Point2f(0, standard_height));
			dst_pts.push_back(cv::Point2f(standard_width, 0));
			dst_pts.push_back(cv::Point2f(standard_width, standard_height));
			cv::Mat trans_mat1 = cv::getPerspectiveTransform(src_pts, dst_pts);
			

			cv::Rect red_roi(cv::Point(rects[0], rects[1]), cv::Point(rects[2], rects[3]));
			cv::Rect c1_roi(cv::Point(rects[4], rects[5]), cv::Point(rects[6], rects[7]));
			cv::Rect id_roi(cv::Point(rects[8], rects[9]), cv::Point(rects[10], rects[11]));

			cv::Mat warp_img;
			cv::warpPerspective(src, warp_img, trans_mat1, cv::Size(standard_width, standard_height));
			if (display)
			{
				cv::Mat show;
				std::vector<ZQ_Vec2D> lines_warp;
				_transform_lines(trans_mat1, lines, cv::Rect(0,0,standard_width,standard_height), lines_warp);
				warp_img.copyTo(show);
				for (int i = 0; i < lines_warp.size() / 2; i++)
				{
					float x0 = lines_warp[i * 2 + 0].x;
					float y0 = lines_warp[i * 2 + 0].y;
					float x1 = lines_warp[i * 2 + 1].x;
					float y1 = lines_warp[i * 2 + 1].y;
					cv::line(show, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 1, CV_AA);
				}
				cv::rectangle(show, red_roi, cv::Scalar(255, 0, 0));
				cv::rectangle(show, c1_roi, cv::Scalar(255, 0, 0));
				cv::rectangle(show, id_roi, cv::Scalar(255, 0, 0));
				cv::namedWindow("show");
				cv::imshow("show", show);
				cv::waitKey(0);
			}

			std::vector<ZQ_Vec2D> lines_red, lines_c1, lines_id;
			_transform_lines(trans_mat1, lines, red_roi, lines_red);
			_transform_lines(trans_mat1, lines, c1_roi, lines_c1);
			_transform_lines(trans_mat1, lines, id_roi, lines_id);

            ZQ_Vec2D red_pt_lt, red_pt_lb, red_pt_rt, red_pt_rb;
			ZQ_Vec2D c1_pt_lt, c1_pt_lb, c1_pt_rt, c1_pt_rb;
			ZQ_Vec2D id_pt_lt, id_pt_lb, id_pt_rt, id_pt_rb;
			if (!_detectRectCorners(warp_img(red_roi), lines_red, red_roi.width, red_roi.height, red_pt_lt, red_pt_lb, red_pt_rt, red_pt_rb, 2, 0, display,-1,10)
				|| !_detectRectCorners(warp_img(c1_roi), lines_c1, c1_roi.width, c1_roi.height, c1_pt_lt, c1_pt_lb, c1_pt_rt, c1_pt_rb, 2, 0, display,-1,10)
				|| !_detectRectCorners(warp_img(id_roi), lines_id, id_roi.width, id_roi.height, id_pt_lt, id_pt_lb, id_pt_rt, id_pt_rb, 2, 0, display,-1,5))
				return false;

			int template_width = 1663;
			int template_height = 1139;
			float template_coords[24] =
			{
				70,611,67,978,435,611,434,978,
				676,855,677,964,1180,856,1180,964,
				486,204,485,278,636,203,636,277
			};

			for (int i = 0; i < 12; i++)
			{
				template_coords[i * 2] /= template_width;
				template_coords[i * 2 + 1] /= template_height;
			}


			src_pts.clear();
			dst_pts.clear();
			//src_pts.push_back(cv::Point2f(red_pt_lt.x + red_roi.x, red_pt_lt.y + red_roi.y));
			//src_pts.push_back(cv::Point2f(red_pt_lb.x + red_roi.x, red_pt_lb.y + red_roi.y));
			src_pts.push_back(cv::Point2f(red_pt_rt.x + red_roi.x, red_pt_rt.y + red_roi.y));
			src_pts.push_back(cv::Point2f(red_pt_rb.x + red_roi.x, red_pt_rb.y + red_roi.y));
			src_pts.push_back(cv::Point2f(c1_pt_lt.x + c1_roi.x, c1_pt_lt.y + c1_roi.y));
			src_pts.push_back(cv::Point2f(c1_pt_lb.x + c1_roi.x, c1_pt_lb.y + c1_roi.y));
			src_pts.push_back(cv::Point2f(c1_pt_rt.x + c1_roi.x, c1_pt_rt.y + c1_roi.y));
			src_pts.push_back(cv::Point2f(c1_pt_rb.x + c1_roi.x, c1_pt_rb.y + c1_roi.y));
			src_pts.push_back(cv::Point2f(id_pt_lt.x + id_roi.x, id_pt_lt.y + id_roi.y));
			src_pts.push_back(cv::Point2f(id_pt_lb.x + id_roi.x, id_pt_lb.y + id_roi.y));
			src_pts.push_back(cv::Point2f(id_pt_rt.x + id_roi.x, id_pt_rt.y + id_roi.y));
			src_pts.push_back(cv::Point2f(id_pt_rb.x + id_roi.x, id_pt_rb.y + id_roi.y));
			//dst_pts.push_back(cv::Point2f(template_coords[0] * dstWidth, template_coords[1] * dstHeight));
			//dst_pts.push_back(cv::Point2f(template_coords[2] * dstWidth, template_coords[3] * dstHeight));
			dst_pts.push_back(cv::Point2f(template_coords[4] * dstWidth, template_coords[5] * dstHeight));
			dst_pts.push_back(cv::Point2f(template_coords[6] * dstWidth, template_coords[7] * dstHeight));
			dst_pts.push_back(cv::Point2f(template_coords[8] * dstWidth, template_coords[9] * dstHeight));
			dst_pts.push_back(cv::Point2f(template_coords[10] * dstWidth, template_coords[11] * dstHeight));
			dst_pts.push_back(cv::Point2f(template_coords[12] * dstWidth, template_coords[13] * dstHeight));
			dst_pts.push_back(cv::Point2f(template_coords[14] * dstWidth, template_coords[15] * dstHeight));
			dst_pts.push_back(cv::Point2f(template_coords[16] * dstWidth, template_coords[17] * dstHeight));
			dst_pts.push_back(cv::Point2f(template_coords[18] * dstWidth, template_coords[19] * dstHeight));
			dst_pts.push_back(cv::Point2f(template_coords[20] * dstWidth, template_coords[21] * dstHeight));
			dst_pts.push_back(cv::Point2f(template_coords[22] * dstWidth, template_coords[23] * dstHeight));

			//cv::Mat trans_mat2 = cv::findHomography(src_pts, dst_pts);			
			bool ret;
			cv::Mat trans_mat_x = _find_translation_scale_x(src_pts, dst_pts, ret);
			cv::Mat trans_mat_y = _find_translation_scale_y(src_pts, dst_pts, ret);
			cv::Mat trans_mat2 = trans_mat_x*trans_mat_y;
			//std::cout << trans_mat1 << std::endl;
			//std::cout << trans_mat1.depth() << std::endl;
			//std::cout << trans_mat2 << std::endl;
			//std::cout << trans_mat2.depth() << std::endl;
			transmtx = _mul_two_matrix(trans_mat2, trans_mat1);
			//std::cout << transmtx << std::endl;
			cv::warpPerspective(src, dst, transmtx, cv::Size(dstWidth, dstHeight));

			return true;
		}

		static bool RectifyCard(const cv::Mat& src, int dstWidth, int dstHeight, cv::Mat& dst, cv::Mat& transmtx,
			int border_width = 50, bool display = false)
		{

			std::vector<ZQ_Vec2D> lines;
			clock_t start = clock();
			cv::Mat im_gray;
			cv::cvtColor(src, im_gray, CV_BGR2GRAY);
			_detect_lines(im_gray, lines);
			clock_t finish = clock();
			double duration = (double)(finish - start) / CLOCKS_PER_SEC;
			if (display)
			{
				std::cout << "total time of extract lines is:" << duration << std::endl;
			}

			/* get corners */

			ZQ_Vec2D corner_LT, corner_LB, corner_RT, corner_RB;
			int srcWidth = src.cols;
			int srcHeight = src.rows;
			if (!_detectRectCorners(src, lines, srcWidth, srcHeight, corner_LT, corner_LB, corner_RT, corner_RB, 2, 1, display, border_width))
				return false;


			float scale_x = 1.0;
			float scale_y = 1.0;

			std::vector<cv::Point2f> src_pts, dst_pts;
			src_pts.push_back(cv::Point2f((corner_LT.x + 0.5)*scale_x - 0.5, (corner_LT.y + 0.5)*scale_y - 0.5));
			src_pts.push_back(cv::Point2f((corner_LB.x + 0.5)*scale_x - 0.5, (corner_LB.y + 0.5)*scale_y - 0.5));
			src_pts.push_back(cv::Point2f((corner_RT.x + 0.5)*scale_x - 0.5, (corner_RT.y + 0.5)*scale_y - 0.5));
			src_pts.push_back(cv::Point2f((corner_RB.x + 0.5)*scale_x - 0.5, (corner_RB.y + 0.5)*scale_y - 0.5));
			dst_pts.push_back(cv::Point2f(0, 0));
			dst_pts.push_back(cv::Point2f(0, dstHeight));
			dst_pts.push_back(cv::Point2f(dstWidth, 0));
			dst_pts.push_back(cv::Point2f(dstWidth, dstHeight));

			transmtx = cv::getPerspectiveTransform(src_pts, dst_pts);
			cv::warpPerspective(src, dst, transmtx, cv::Size(dstWidth, dstHeight));

			return true;
		}

	};
}

#endif
