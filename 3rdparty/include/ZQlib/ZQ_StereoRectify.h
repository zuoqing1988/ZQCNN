#ifndef _ZQ_STEREO_RECTIFY_H_
#define _ZQ_STEREO_RECTIFY_H_
#pragma once

#include "ZQ_Rodrigues.h"
#include "ZQ_SVD.h"
#include "ZQ_Vec3D.h"
#include "ZQ_Ray3D.h"
#include "ZQ_DoubleImage.h"
#include "ZQ_GaussianPyramid.h"
#include "ZQ_CameraCalibration.h"
#include "ZQ_OpticalFlow.h"


namespace ZQ
{
	class ZQ_StereoRectify
	{
	public:
		/***********************/
		/* stereo rectify
		/**********************/
		template<class T>
		static bool stereo_rectify(int width, int height, const T left_ori_rT[6], const T right_ori_rT[6], const T left_ori_fc_cc_alpha_kc[10], const T right_ori_fc_cc_alpha_kc[10],
			T left_rectify_rT[6], T right_rectify_rT[6], T left_rectify_fc_cc_alpha_kc[10], T right_rectify_fc_cc_alpha_kc[10],
			T* left_map_rectify_from_ori, bool* left_mask_rectify_from_ori, T* right_map_rectify_from_ori, bool* right_mask_rectify_from_ori, bool zAxis_in = false);

	public:
		template<class T>
		static bool stereo_disparity_cross_check(const ZQ_DImage<T>& left_disparity, const ZQ_DImage<bool>& left_rectify_mask, const ZQ_DImage<T>& right_disparity, 
			const ZQ_DImage<bool>& right_rectify_mask, ZQ_DImage<bool>& left_disparity_mask, ZQ_DImage<bool>& right_disparity_mask, double tol_E = 1.0);
		
		template<class T>
		static bool stereo_disparity_opticalflow_L2(const ZQ_DImage<T>& left_rectify_img, const ZQ_DImage<T>& right_rectify_img,
			ZQ_DImage<T>& disparity, int max_disparity, double alpha, int nFPIter, int nSORIter, double ratio = 0.8, bool occ_detect = false);

		template<class T>
		static bool stereo_disparity_opticalflow_L1(const ZQ_DImage<T>& left_rectify_img, const ZQ_DImage<T>& right_rectify_img,
			ZQ_DImage<T>& disparity, int max_disparity, double alpha, int nFPIter, int nInnerIter, int nSORIter, double ratio = 0.8, bool occ_detect = false);
	};

	/*************************************************************************************************************************************************/
	/*************************************************************************************************************************************************/

	/***********************/
	/* stereo rectify
	/**********************/
	template<class T>
	bool ZQ_StereoRectify::stereo_rectify(int width, int height, const T left_ori_rT[6], const T right_ori_rT[6], const T left_ori_fc_cc_alpha_kc[10], const T right_ori_fc_cc_alpha_kc[10],
		T left_rectify_rT[6], T right_rectify_rT[6], T left_rectify_fc_cc_alpha_kc[10], T right_rectify_fc_cc_alpha_kc[10],
		T* left_map_rectify_from_ori, bool* left_mask_rectify_from_ori, T* right_map_rectify_from_ori, bool* right_mask_rectify_from_ori, bool zAxis_in /* = false */)
	{
		const T* left_ori_r = left_ori_rT;
		const T* left_ori_T = left_ori_rT + 3;
		const T* right_ori_r = right_ori_rT;
		const T* right_ori_T = right_ori_rT + 3;

		T left_ori_R[9], right_ori_R[9];
		ZQ_Rodrigues::ZQ_Rodrigues_r2R(left_ori_r, left_ori_R);	
		ZQ_Rodrigues::ZQ_Rodrigues_r2R(right_ori_r, right_ori_R);

		ZQ_Vec3D ray1_o(left_ori_T[0], left_ori_T[1], left_ori_T[2]);
		ZQ_Vec3D ray1_d(left_ori_R[2], left_ori_R[5], left_ori_R[8]);
		ZQ_Vec3D ray2_o(right_ori_T[0], right_ori_T[1], right_ori_T[2]);
		ZQ_Vec3D ray2_d(right_ori_R[2], right_ori_R[5], right_ori_R[8]);


		ZQ_Vec3D Xdir = ray2_o - ray1_o;
		if (Xdir.Length() == 0)
			return false;

		Xdir.Normalized();
		ZQ_Vec3D tmp_z = (ray1_d + ray2_d)*0.5;
		
		ZQ_Vec3D Ydir = tmp_z.CrossProduct(Xdir);
		if (Ydir.Length() == 0)
			return false;

		Ydir.Normalized();
		ZQ_Vec3D Zdir = Xdir.CrossProduct(Ydir);
		if (Zdir.Length() == 0)
			return false;

		Zdir.Normalized();

		T rectify_R[9] =
		{
			Xdir.x, Ydir.x, Zdir.x,
			Xdir.y, Ydir.y, Zdir.y,
			Xdir.z, Ydir.z, Zdir.z
		};

		if (!ZQ_Rodrigues::ZQ_Rodrigues_R2r(rectify_R, left_rectify_rT))
			return false;
		if (!ZQ_Rodrigues::ZQ_Rodrigues_R2r(rectify_R, right_rectify_rT))
			return false;
		memcpy(left_rectify_rT + 3, left_ori_rT + 3, sizeof(T)* 3);
		memcpy(right_rectify_rT + 3, right_ori_rT + 3, sizeof(T)* 3);

		/**************************/
		double fx1 = left_ori_fc_cc_alpha_kc[0], fy1 = left_ori_fc_cc_alpha_kc[1];
		double fx2 = right_ori_fc_cc_alpha_kc[0], fy2 = right_ori_fc_cc_alpha_kc[1];

		if (fx1 <= 0 || fy1 <= 0 || fx2 <= 0 || fy2 <= 0)
			return false;

		double focal = sqrt(sqrt(fabs(fx1*fx2*fy1*fy2)));
		
		/* compute the range of the rectified images*/
		T R_from_world_to_rectify[9] = 
		{
			rectify_R[0], rectify_R[3], rectify_R[6],
			rectify_R[1], rectify_R[4], rectify_R[7],
			rectify_R[2], rectify_R[5], rectify_R[8]
		};
		T R_from_left_to_rectify[9], R_from_right_to_rectify[9];
		ZQ_MathBase::MatrixMul(R_from_world_to_rectify, left_ori_R, 3, 3, 3, R_from_left_to_rectify);
		ZQ_MathBase::MatrixMul(R_from_world_to_rectify, right_ori_R, 3, 3, 3, R_from_right_to_rectify);
		bool left_to_rectify_valid_flag[4] = { 0 }, right_to_rectify_valid_flag[4] = { 0 };
		T left_to_rectify_corners[4][2], right_to_rectify_corners[4][2];
		T corners[4][2] = 
		{
			{ 0, 0 }, { 0, height - 1 }, { width - 1, 0 }, { width - 1, height - 1 }
		};
		for (int pp = 0; pp < 4; pp++)
		{
			T tmp_in[2] = 
			{
				(corners[pp][0] - left_ori_fc_cc_alpha_kc[2]) / left_ori_fc_cc_alpha_kc[0],
				(corners[pp][1] - left_ori_fc_cc_alpha_kc[3]) / left_ori_fc_cc_alpha_kc[1]
			};
			T tmp_out[2];
			ZQ_CameraCalibration::undistort_points_oulu(1, tmp_in, left_ori_fc_cc_alpha_kc + 5, tmp_out);
			tmp_out[0] -= left_ori_fc_cc_alpha_kc[4] * tmp_out[1];

			T rectify_pt[3];
			
			if (!zAxis_in)
			{
				T ori_pt[3] = { tmp_out[0], tmp_out[1], -1 };
				ZQ_MathBase::MatrixMul(R_from_left_to_rectify, ori_pt, 3, 3, 1, rectify_pt);
				rectify_pt[2] = -rectify_pt[2];
			}
			else
			{
				T ori_pt[3] = { tmp_out[0], tmp_out[1], 1 };
				ZQ_MathBase::MatrixMul(R_from_left_to_rectify, ori_pt, 3, 3, 1, rectify_pt);
			}

			if (rectify_pt[2] <= 0)
			{
				left_to_rectify_valid_flag[pp] = false;
			}
			else
			{
				left_to_rectify_valid_flag[pp] = true;
				left_to_rectify_corners[pp][0] = rectify_pt[0] / rectify_pt[2] * focal;
				left_to_rectify_corners[pp][1] = rectify_pt[1] / rectify_pt[2] * focal;
			}
		}

		for (int pp = 0; pp < 4; pp++)
		{
			T tmp_in[2] =
			{
				(corners[pp][0] - right_ori_fc_cc_alpha_kc[2]) / right_ori_fc_cc_alpha_kc[0],
				(corners[pp][1] - right_ori_fc_cc_alpha_kc[3]) / right_ori_fc_cc_alpha_kc[1]
			};
			T tmp_out[2];
			ZQ_CameraCalibration::undistort_points_oulu(1, tmp_in, right_ori_fc_cc_alpha_kc + 5, tmp_out);
			tmp_out[0] -= right_ori_fc_cc_alpha_kc[4] * tmp_out[1];

			T rectify_pt[3];

			if (!zAxis_in)
			{
				T ori_pt[3] = { tmp_out[0], tmp_out[1], -1 };
				ZQ_MathBase::MatrixMul(R_from_right_to_rectify, ori_pt, 3, 3, 1, rectify_pt);
				rectify_pt[2] = -rectify_pt[2];
			}
			else
			{
				T ori_pt[3] = { tmp_out[0], tmp_out[1], 1 };
				ZQ_MathBase::MatrixMul(R_from_right_to_rectify, ori_pt, 3, 3, 1, rectify_pt);
			}

			if (rectify_pt[2] <= 0)
			{
				right_to_rectify_valid_flag[pp] = false;
			}
			else
			{
				right_to_rectify_valid_flag[pp] = true;
				right_to_rectify_corners[pp][0] = rectify_pt[0] / rectify_pt[2] * focal;
				right_to_rectify_corners[pp][1] = rectify_pt[1] / rectify_pt[2] * focal;
			}
		}

		memset(left_rectify_fc_cc_alpha_kc + 4, 0, sizeof(T)* 6);
		memset(right_rectify_fc_cc_alpha_kc + 4, 0, sizeof(T)* 6);
		left_rectify_fc_cc_alpha_kc[0] = focal;
		left_rectify_fc_cc_alpha_kc[1] = focal;
		right_rectify_fc_cc_alpha_kc[0] = focal;
		right_rectify_fc_cc_alpha_kc[1] = focal;
		/**************** determine cx, cy ************************/
		if (!left_to_rectify_valid_flag[0] || !left_to_rectify_valid_flag[1] || !left_to_rectify_valid_flag[2] || !left_to_rectify_valid_flag[3]
			|| !right_to_rectify_valid_flag[0] || !right_to_rectify_valid_flag[1] || !right_to_rectify_valid_flag[2] || !right_to_rectify_valid_flag[3])
		{
			left_rectify_fc_cc_alpha_kc[2] = (left_ori_fc_cc_alpha_kc[2] + right_ori_fc_cc_alpha_kc[2])*0.5;
			left_rectify_fc_cc_alpha_kc[3] = (left_ori_fc_cc_alpha_kc[3] + right_ori_fc_cc_alpha_kc[3])*0.5;
			
			right_rectify_fc_cc_alpha_kc[2] = (left_ori_fc_cc_alpha_kc[2] + right_ori_fc_cc_alpha_kc[2])*0.5;
			right_rectify_fc_cc_alpha_kc[3] = (left_ori_fc_cc_alpha_kc[3] + right_ori_fc_cc_alpha_kc[3])*0.5;
		}
		else
		{
			T left_min_x = left_to_rectify_corners[0][0], left_max_x = left_to_rectify_corners[0][0];
			T left_min_y = left_to_rectify_corners[0][1], left_max_y = left_to_rectify_corners[0][1];
			T right_min_x = right_to_rectify_corners[0][0], right_max_x = right_to_rectify_corners[0][0];
			T right_min_y = right_to_rectify_corners[0][1], right_max_y = right_to_rectify_corners[0][1];
			for (int pp = 1; pp < 4; pp++)
			{
				left_min_x = __min(left_min_x, left_to_rectify_corners[pp][0]);
				left_max_x = __max(left_max_x, left_to_rectify_corners[pp][0]);
				left_min_y = __min(left_min_y, left_to_rectify_corners[pp][1]);
				left_max_y = __max(left_max_y, left_to_rectify_corners[pp][1]);

				right_min_x = __min(right_min_x, right_to_rectify_corners[pp][0]);
				right_max_x = __max(right_max_x, right_to_rectify_corners[pp][0]);
				right_min_y = __min(right_min_y, right_to_rectify_corners[pp][1]);
				right_max_y = __max(right_max_y, right_to_rectify_corners[pp][1]);
			}

			/*cx*/
			left_rectify_fc_cc_alpha_kc[2] = width*0.5;
			right_rectify_fc_cc_alpha_kc[2] = width*0.5;
			

			T all_min_y = __min(left_min_y, right_min_y);
			T all_max_y = __min(left_max_y, right_max_y);
			/* cy */
			if (all_min_y <= 0 && all_max_y >= 0)
			{
				left_rectify_fc_cc_alpha_kc[3] = 0.5*height - 0.5*(all_min_y + all_max_y);
				right_rectify_fc_cc_alpha_kc[3] = 0.5*height - 0.5*(all_min_y + all_max_y);
			}
			else if (all_max_y < 0)
			{
				left_rectify_fc_cc_alpha_kc[3] = __min(2 * height, height - all_max_y);
				right_rectify_fc_cc_alpha_kc[3] = __min(2 * height, height - all_max_y);
			}
			else // all_min_y > 0
			{
				left_rectify_fc_cc_alpha_kc[3] = __max(-height, -all_min_y);
				right_rectify_fc_cc_alpha_kc[3] = __max(-height, -all_min_y);
			}
		}

		/**************************/
		T R_from_world_to_left[9] =
		{
			left_ori_R[0], left_ori_R[3], left_ori_R[6],
			left_ori_R[1], left_ori_R[4], left_ori_R[7],
			left_ori_R[2], left_ori_R[5], left_ori_R[8]
		};
		T R_from_rectify_to_left[9];
		ZQ_MathBase::MatrixMul(R_from_world_to_left, rectify_R, 3, 3, 3, R_from_rectify_to_left);

		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				T rectify_x = w - left_rectify_fc_cc_alpha_kc[2];
				T rectify_y = h - left_rectify_fc_cc_alpha_kc[3];
				T ori_pt[3];

				if (!zAxis_in)
				{
					T rectify_pt[3] = { rectify_x, rectify_y, -focal };
					ZQ_MathBase::MatrixMul(R_from_rectify_to_left, rectify_pt, 3, 3, 1, ori_pt);
					ori_pt[2] = -ori_pt[2];
				}
				else
				{
					T rectify_pt[3] = { rectify_x, rectify_y, focal };
					ZQ_MathBase::MatrixMul(R_from_rectify_to_left, rectify_pt, 3, 3, 1, ori_pt);
				}
				
				if (ori_pt[2] <= 0)
				{
					left_mask_rectify_from_ori[h*width + w] = false;
					left_map_rectify_from_ori[(h*width + w) * 2 + 0] = 0;
					left_map_rectify_from_ori[(h*width + w) * 2 + 1] = 0;
				}
				else
				{
					T ori_undist_pt[2] = 
					{
						ori_pt[0] / ori_pt[2],
						ori_pt[1] / ori_pt[2]
					};
					T ori_dist_pt[2];
					ZQ_CameraCalibration::distort_points(1, ori_undist_pt, left_ori_fc_cc_alpha_kc + 5, ori_dist_pt);
					T ori_x = (ori_dist_pt[0] + ori_dist_pt[1] * left_ori_fc_cc_alpha_kc[4])*left_ori_fc_cc_alpha_kc[0] + left_ori_fc_cc_alpha_kc[2];
					T ori_y = ori_dist_pt[1]*left_ori_fc_cc_alpha_kc[1] + left_ori_fc_cc_alpha_kc[3];
					if (ori_x < 0 || ori_x > width - 1 || ori_y < 0 || ori_y > height - 1)
					{
						left_mask_rectify_from_ori[h*width + w] = false;
						left_map_rectify_from_ori[(h*width + w) * 2 + 0] = __min(width-1,__max(0,ori_x));
						left_map_rectify_from_ori[(h*width + w) * 2 + 1] = __min(height-1,__max(0,ori_y));
					}
					else
					{
						left_mask_rectify_from_ori[h*width + w] = true;
						left_map_rectify_from_ori[(h*width + w) * 2 + 0] = ori_x;
						left_map_rectify_from_ori[(h*width + w) * 2 + 1] = ori_y;
					}
				}
			}
		}


		T R_from_world_to_right[9] =
		{
			right_ori_R[0], right_ori_R[3], right_ori_R[6],
			right_ori_R[1], right_ori_R[4], right_ori_R[7],
			right_ori_R[2], right_ori_R[5], right_ori_R[8]
		};

		T R_from_rectify_to_right[9];
		ZQ_MathBase::MatrixMul(R_from_world_to_right, rectify_R, 3, 3, 3, R_from_rectify_to_right);

		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				T rectify_x = w - right_rectify_fc_cc_alpha_kc[2];
				T rectify_y = h - right_rectify_fc_cc_alpha_kc[3];
				T ori_pt[3];

				if (!zAxis_in)
				{
					T rectify_pt[3] = { rectify_x, rectify_y, -focal };
					ZQ_MathBase::MatrixMul(R_from_rectify_to_right, rectify_pt, 3, 3, 1, ori_pt);
					ori_pt[2] = -ori_pt[2];
				}
				else
				{
					T rectify_pt[3] = { rectify_x, rectify_y, focal };
					ZQ_MathBase::MatrixMul(R_from_rectify_to_right, rectify_pt, 3, 3, 1, ori_pt);
				}
				
				if (ori_pt[2] <= 0)
				{
					right_mask_rectify_from_ori[h*width + w] = false;
					right_map_rectify_from_ori[(h*width + w) * 2 + 0] = 0;
					right_map_rectify_from_ori[(h*width + w) * 2 + 1] = 0;
				}
				else
				{
					T ori_undist_pt[2] =
					{
						ori_pt[0] / ori_pt[2],
						ori_pt[1] / ori_pt[2]
					};
					T ori_dist_pt[2];
					ZQ_CameraCalibration::distort_points(1, ori_undist_pt, right_ori_fc_cc_alpha_kc + 5, ori_dist_pt);
					T ori_x = (ori_dist_pt[0] + ori_dist_pt[1] * right_ori_fc_cc_alpha_kc[4])*right_ori_fc_cc_alpha_kc[0] + right_ori_fc_cc_alpha_kc[2];
					T ori_y = ori_dist_pt[1] * right_ori_fc_cc_alpha_kc[1] + right_ori_fc_cc_alpha_kc[3];
					if (ori_x < 0 || ori_x > width - 1 || ori_y < 0 || ori_y > height - 1)
					{
						right_mask_rectify_from_ori[h*width + w] = false;
						right_map_rectify_from_ori[(h*width + w) * 2 + 0] = __min(width - 1, __max(0, ori_x));
						right_map_rectify_from_ori[(h*width + w) * 2 + 1] = __min(height - 1, __max(0, ori_y));
					}
					else
					{
						right_mask_rectify_from_ori[h*width + w] = true;
						right_map_rectify_from_ori[(h*width + w) * 2 + 0] = ori_x;
						right_map_rectify_from_ori[(h*width + w) * 2 + 1] = ori_y;
					}
				}
			}
		}

		return true;
	}


	template<class T>
	bool ZQ_StereoRectify::stereo_disparity_cross_check(const ZQ_DImage<T>& left_disparity, const ZQ_DImage<bool>& left_rectify_mask, const ZQ_DImage<T>& right_disparity,
		const ZQ_DImage<bool>& right_rectify_mask, ZQ_DImage<bool>& left_disparity_mask, ZQ_DImage<bool>& right_disparity_mask, double tol_E /*= 1.0*/)
	{
		int width = left_disparity.width();
		int height = left_disparity.height();
		if (left_disparity.nchannels() != 1 || !right_disparity.matchDimension(width, height, 1)
			|| !left_rectify_mask.matchDimension(width, height, 1) || !right_rectify_mask.matchDimension(width,height,1))
			return false;

		left_disparity_mask.allocate(width, height);
		right_disparity_mask.allocate(width, height);

		const T*& left_data = left_disparity.data();
		const T*& right_data = right_disparity.data();
		const bool*& left_rectify_mask_data = left_rectify_mask.data();
		const bool*& right_rectify_mask_data = right_rectify_mask.data();
		bool*& left_disparity_mask_data = left_disparity_mask.data();
		bool*& right_disparity_mask_data = right_disparity_mask.data();

		ZQ_DImage<T> warp_disparity(width, height);
		T*& warp_data = warp_disparity.data();

		ZQ_DImage<T> v(width, height);

		ZQ_ImageProcessing::WarpImage(warp_data, right_data, left_data, v.data(), width, height, 1, (const T*)NULL, false);
		for (int i = 0; i < width*height; i++)
		{
			if (left_rectify_mask_data[i] && fabs(left_data[i] + warp_data[i]) < tol_E)
				left_disparity_mask_data[i] = true;
		}

		ZQ_ImageProcessing::WarpImage(warp_data, left_data, right_data, v.data(), width, height, 1, (const T*)NULL, false);
		for (int i = 0; i < width*height; i++)
		{
			if (right_rectify_mask_data[i] && fabs(right_data[i] + warp_data[i]) < tol_E)
				right_disparity_mask_data[i] = true;
		}
		return true;
	}

	template<class T>
	bool ZQ_StereoRectify::stereo_disparity_opticalflow_L2(const ZQ_DImage<T>& left_rectify_img, const ZQ_DImage<T>& right_rectify_img,
		ZQ_DImage<T>& disparity, int max_disparity, double alpha, int nFPIter, int nSORIter, double ratio /*= 0.8*/, bool occ_detect/* = false*/)
	{
		int width = left_rectify_img.width();
		int height = left_rectify_img.height();
		int nChannels = left_rectify_img.nchannels();
		if (!right_rectify_img.matchDimension(width, height, nChannels))
			return false;

		max_disparity = abs(max_disparity);
		int min_width = width / (max_disparity + 1.0);

		ZQ_OpticalFlowOptions opt;
		opt.alpha = alpha;
		opt.beta = 0;
		opt.ratioForPyramid = ratio;
		opt.nOuterFixedPointIterations = nFPIter;
		opt.nSORIterations = nSORIter;
		opt.minWidthForPyramid = min_width;
		opt.weightedMedFiltIter_for_occ_detect = occ_detect ? 1 : 0;

		ZQ_DImage<T> warpI2;
		ZQ_OpticalFlow::Coarse2Fine_HS_L2(disparity, warpI2, left_rectify_img, right_rectify_img, opt);

		return true;
	}

	template<class T>
	bool ZQ_StereoRectify::stereo_disparity_opticalflow_L1(const ZQ_DImage<T>& left_rectify_img, const ZQ_DImage<T>& right_rectify_img,
		ZQ_DImage<T>& disparity, int max_disparity, double alpha, int nFPIter, int nInnerIter, int nSORIter, double ratio /* = 0.8 */, bool occ_detect/* = false*/)
	{
		int width = left_rectify_img.width();
		int height = left_rectify_img.height();
		int nChannels = left_rectify_img.nchannels();
		if (!right_rectify_img.matchDimension(width, height, nChannels))
			return false;

		max_disparity = abs(max_disparity);
		int min_width = width / (max_disparity + 1.0);

		ZQ_OpticalFlowOptions opt;
		opt.alpha = alpha;
		opt.beta = 0;
		opt.ratioForPyramid = ratio; 
		opt.nOuterFixedPointIterations = nFPIter;
		opt.nInnerFixedPointIterations = nInnerIter;
		opt.nSORIterations = nSORIter;
		opt.minWidthForPyramid = min_width;
		opt.weightedMedFiltIter_for_occ_detect = occ_detect ? 1 : 0;

		ZQ_DImage<T> warpI2;
		ZQ_OpticalFlow::Coarse2Fine_HS_L1(disparity, warpI2, left_rectify_img, right_rectify_img, opt);

		return true;
	}
}

#endif