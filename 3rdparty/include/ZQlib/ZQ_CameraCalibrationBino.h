#ifndef _ZQ_CAMERA_CALIBRATION_BINO_H_
#define _ZQ_CAMERA_CALIBRATION_BINO_H_
#pragma once

#include "ZQ_CameraCalibrationMono.h"
#include "ZQ_QuickSort.h"
#include <math.h>

namespace ZQ
{
	class ZQ_CameraCalibrationBino
	{
	public:
		template<class T>
		static bool CalibrateBinocularCamera(int nViews, int nPts, const T* X3, const T* left_X2, const T* right_X2, T right_to_left_rT[6], T* out_right_rT,
			const T left_fc[2], const T left_cc[2], const T left_kc[5], const T left_alpha_c, const T right_fc[2], const T right_cc[2], const T right_kc[5], const T right_alpha_c, bool zAxis_in,
			const bool* left_active_images = 0, const bool* right_active_images = 0, const T* left_rT = 0, const T* right_rT = 0, int max_iter = 300, bool sparse_solver = false, bool display = false);

		template<class T>
		static bool CalibrateBinocularCamera2(int nViews, int nPts, const T* X3, const T* left_X2, const T* right_X2, T right_to_left_rT[6], T* out_right_rT,
			T left_fc[2], T left_cc[2], T left_kc[5], T& left_alpha_c, T right_fc[2], T right_cc[2], T right_kc[5], T& right_alpha_c, ZQ_CameraCalibrationMono::Calib_Method method, bool zAxis_in,
			const bool* left_active_images = 0, const bool* right_active_images = 0, const T* left_rT = 0, const T* right_rT = 0, int max_iter = 300, bool sparse_solver = false, bool display = false);

	public:
		template<class T>
		static bool _get_left_rT_from_right_rT_fun(const T* right_rT, const T* right_to_left_rT, T* left_rT);

		template<class T>
		static bool _get_left_rT_from_right_rT_jac(const T* right_rT, const T* right_to_left_rT, T* d_l_rT_d_r_rT, T* d_l_rT_d_r2l_rT);

	private:

		template<class T>
		class Calib_Bino_Data_Header
		{
		public:
			const T* X3;
			const T* left_X2;
			const T* right_X2;
			int n_views;
			int n_pts;
			const T* left_fc_cc_alpha_kc;
			const T* right_fc_cc_alpha_kc;
			ZQ_CameraCalibrationMono::Calib_Method method;
			bool zAxis_in;

		};

		template<class T>
		static bool _calib_bino_with_known_intrinsic_fun(const T* p, T* hx, int m, int n, const void* data);

		template<class T>
		static bool _calib_bino_with_known_intrinsic_jac(const T* p, T* jx, int m, int n, const void* data);

		template<class T>
		static bool _calib_bino_with_known_intrinsic_jac_sparse(const T* p, taucs_ccs_matrix*& jx, int m, int n, const void* data);

		template<class T>
		static bool _calib_bino_fun(const T* p, T* hx, int m, int n, const void* data);

		template<class T>
		static bool _calib_bino_jac(const T* p, T* jx, int m, int n, const void* data);

		template<class T>
		static bool _calib_bino_jac_sparse(const T* p, taucs_ccs_matrix*& jx, int m, int n, const void* data);

		template<class T>
		static bool _calib_bino_with_known_intrinsic_with_init(int nViews, int nPts, const T* X3, const T* left_X2, const T* right_X2,
			const T left_fc[2], const T left_cc[2], const T left_kc[5], const T left_alpha_c,
			const T right_fc[2], const T right_cc[2], const T right_kc[5], const T right_alpha_c, bool zAxis_in,
			int max_iter_levmar, T* right_rT, T* right_to_left_rT, double& avg_err_square, bool sparse_solver);

		template<class T>
		static bool _calib_bino_with_init(int nViews, int nPts, const T* X3, const T* left_X2, const T* right_X2,
			T left_fc[2], T left_cc[2], T left_kc[5], T& left_alpha_c,
			T right_fc[2], T right_cc[2], T right_kc[5], T& right_alpha_c, ZQ_CameraCalibrationMono::Calib_Method method, bool zAxis_in,
			int max_iter_levmar, T* right_rT, T* right_to_left_rT, double& avg_err_square, bool sparse_solver);


		template<class T>
		static bool _compute_err_calib(int nViews, int nPts, const T* X3, const T* left_X2, const T* right_X2, const T* right_rT, const T* right_to_left_rT,
			const T left_fc[2], const T left_cc[2], const T left_kc[5], const T left_alpha_c, const T right_fc[2], const T right_cc[2], const T right_kc[5], const T right_alpha_c,
			double& err_std, double& max_err, bool zAxis_in);

		template<class T>
		static bool _estimate_uncertainties(int nViews, int nPts, const T* X3, const T* right_rT, const T* right_to_left_rT, double sigma_x,
			const T left_fc[2], const T left_cc[2], const T left_kc[5], const T left_alpha_c, const T right_fc[2], const T right_cc[2], const T right_kc[5], const T right_alpha_c,
			T right_to_left_rT_err[6], bool zAxis_in);
	};


	template<class T>
	bool ZQ_CameraCalibrationBino::_get_left_rT_from_right_rT_fun(const T* right_rT, const T* right_to_left_rT, T* left_rT)
	{
		T right_R[9];
		T right_to_left_R[9];
		T left_R[9];
		ZQ_Rodrigues::ZQ_Rodrigues_r2R(right_rT, right_R);
		ZQ_Rodrigues::ZQ_Rodrigues_r2R(right_to_left_rT, right_to_left_R);
		ZQ_MathBase::MatrixMul(right_to_left_R, right_R, 3, 3, 3, left_R);
		if (!ZQ_Rodrigues::ZQ_Rodrigues_R2r(left_R, left_rT))
			return false;
		left_rT[3] = right_to_left_R[0] * right_rT[3] + right_to_left_R[1] * right_rT[4] + right_to_left_R[2] * right_rT[5] + right_to_left_rT[3];
		left_rT[4] = right_to_left_R[3] * right_rT[3] + right_to_left_R[4] * right_rT[4] + right_to_left_R[5] * right_rT[5] + right_to_left_rT[4];
		left_rT[5] = right_to_left_R[6] * right_rT[3] + right_to_left_R[7] * right_rT[4] + right_to_left_R[8] * right_rT[5] + right_to_left_rT[5];
		return true;
	}

	template<class T>
	bool ZQ_CameraCalibrationBino::_get_left_rT_from_right_rT_jac(const T* right_rT, const T* right_to_left_rT, T* d_l_rT_d_r_rT, T* d_l_rT_d_r2l_rT)
	{
		T right_R[9], d_r_R_d_r_r[27];
		T right_to_left_R[9], d_r2l_R_d_r2l_r[27];
		T left_R[9];
		ZQ_Rodrigues::ZQ_Rodrigues_r2R(right_rT, right_R, d_r_R_d_r_r);
		ZQ_Rodrigues::ZQ_Rodrigues_r2R(right_to_left_rT, right_to_left_R, d_r2l_R_d_r2l_r);
		ZQ_MathBase::MatrixMul(right_to_left_R, right_R, 3, 3, 3, left_R);

		T d_l_R_d_r2l_R[81] = { 0 };
		for (int i = 0; i < 3; i++)
		{
			d_l_R_d_r2l_R[(i * 3 + 0) * 9 + i * 3 + 0] = right_R[0]; d_l_R_d_r2l_R[(i * 3 + 0) * 9 + i * 3 + 1] = right_R[3]; d_l_R_d_r2l_R[(i * 3 + 0) * 9 + i * 3 + 2] = right_R[6];
			d_l_R_d_r2l_R[(i * 3 + 1) * 9 + i * 3 + 0] = right_R[1]; d_l_R_d_r2l_R[(i * 3 + 1) * 9 + i * 3 + 1] = right_R[4]; d_l_R_d_r2l_R[(i * 3 + 1) * 9 + i * 3 + 2] = right_R[7];
			d_l_R_d_r2l_R[(i * 3 + 2) * 9 + i * 3 + 0] = right_R[2]; d_l_R_d_r2l_R[(i * 3 + 2) * 9 + i * 3 + 1] = right_R[5]; d_l_R_d_r2l_R[(i * 3 + 2) * 9 + i * 3 + 2] = right_R[8];
		}
		T d_l_R_d_r_R[81] = { 0 };
		for (int i = 0; i < 3; i++)
		{
			d_l_R_d_r_R[(i * 3 + 0) * 9 + 0] = right_to_left_R[i * 3 + 0]; d_l_R_d_r_R[(i * 3 + 0) * 9 + 3] = right_to_left_R[i * 3 + 1]; d_l_R_d_r_R[(i * 3 + 0) * 9 + 6] = right_to_left_R[i * 3 + 2];
			d_l_R_d_r_R[(i * 3 + 1) * 9 + 1] = right_to_left_R[i * 3 + 0]; d_l_R_d_r_R[(i * 3 + 1) * 9 + 4] = right_to_left_R[i * 3 + 1]; d_l_R_d_r_R[(i * 3 + 1) * 9 + 7] = right_to_left_R[i * 3 + 2];
			d_l_R_d_r_R[(i * 3 + 2) * 9 + 2] = right_to_left_R[i * 3 + 0]; d_l_R_d_r_R[(i * 3 + 2) * 9 + 5] = right_to_left_R[i * 3 + 1]; d_l_R_d_r_R[(i * 3 + 2) * 9 + 8] = right_to_left_R[i * 3 + 2];
		}

		T d_l_R_d_r2l_r[27], d_l_R_d_r_r[27];
		ZQ_MathBase::MatrixMul(d_l_R_d_r2l_R, d_r2l_R_d_r2l_r, 9, 9, 3, d_l_R_d_r2l_r);
		ZQ_MathBase::MatrixMul(d_l_R_d_r_R, d_r_R_d_r_r, 9, 9, 3, d_l_R_d_r_r);
		T left_r[3], d_l_r_d_l_R[27];
		if (!ZQ_Rodrigues::ZQ_Rodrigues_R2r(left_R, left_r, d_l_r_d_l_R))
			return false;
		T d_l_r_d_r2l_r[9], d_l_r_d_r_r[9];
		ZQ_MathBase::MatrixMul(d_l_r_d_l_R, d_l_R_d_r2l_r, 3, 9, 3, d_l_r_d_r2l_r);
		ZQ_MathBase::MatrixMul(d_l_r_d_l_R, d_l_R_d_r_r, 3, 9, 3, d_l_r_d_r_r);

		T d_l_T_d_r2l_T[9] =
		{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1
		};
		T d_l_T_d_r_T[9] =
		{
			right_to_left_R[0], right_to_left_R[1], right_to_left_R[2],
			right_to_left_R[3], right_to_left_R[4], right_to_left_R[5],
			right_to_left_R[6], right_to_left_R[7], right_to_left_R[8]
		};

		T d_l_T_d_r2l_R[27] =
		{
			right_rT[3], right_rT[4], right_rT[5], 0, 0, 0, 0, 0, 0,
			0, 0, 0, right_rT[3], right_rT[4], right_rT[5], 0, 0, 0,
			0, 0, 0, 0, 0, 0, right_rT[3], right_rT[4], right_rT[5]
		};
		T d_l_T_d_r2l_r[9];
		ZQ_MathBase::MatrixMul(d_l_T_d_r2l_R, d_r2l_R_d_r2l_r, 3, 9, 3, d_l_T_d_r2l_r);
		memset(d_l_rT_d_r_rT, 0, sizeof(T) * 36);
		memset(d_l_rT_d_r2l_rT, 0, sizeof(T) * 36);
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				d_l_rT_d_r_rT[i * 6 + j] = d_l_r_d_r_r[i * 3 + j];
				d_l_rT_d_r2l_rT[i * 6 + j] = d_l_r_d_r2l_r[i * 3 + j];
				d_l_rT_d_r_rT[(i + 3) * 6 + j + 3] = d_l_T_d_r_T[i * 3 + j];
				d_l_rT_d_r2l_rT[(i + 3) * 6 + j + 3] = d_l_T_d_r2l_T[i * 3 + j];
				d_l_rT_d_r2l_rT[(i + 3) * 6 + j] = d_l_T_d_r2l_r[i * 3 + j];
			}
		}
		return true;
	}

	template<class T>
	bool ZQ_CameraCalibrationBino::_calib_bino_with_known_intrinsic_fun(const T* p, T* hx, int m, int n, const void* data)
	{
		const Calib_Bino_Data_Header<T>* ptr = (const Calib_Bino_Data_Header<T>*)data;
		int nPts = ptr->n_pts;
		int nViews = ptr->n_views;
		const T* X3 = ptr->X3;
		const T* left_X2 = ptr->left_X2;
		const T* right_X2 = ptr->right_X2;

		const T* left_fc_cc_alpha_kc = ptr->left_fc_cc_alpha_kc;
		const T* right_fc_cc_alpha_kc = ptr->right_fc_cc_alpha_kc;

		bool zAxis_in = ptr->zAxis_in;

		const T* right_to_left_rT = p;
		const T* right_rT = p + 6;

		int left_X2_offset = 0;
		int right_X2_offset = nViews*nPts * 2;

		ZQ_DImage<T> xp_im(nPts * 2, 1, 1);
		T*& xp = xp_im.data();
		for (int vv = 0; vv < nViews; vv++)
		{
			if (!ZQ_CameraProjection::project_points_fun(nPts, X3 + nPts * 3 * vv, right_rT + vv * 6, right_fc_cc_alpha_kc, right_fc_cc_alpha_kc + 2, right_fc_cc_alpha_kc + 5, right_fc_cc_alpha_kc[4], xp, zAxis_in))
				return false;
			ZQ_MathBase::VecMinus(nPts * 2, xp, right_X2 + nPts * 2 * vv, hx + right_X2_offset + nPts * 2 * vv);
		}

		for (int vv = 0; vv < nViews; vv++)
		{
			T left_rT[6];
			if (!_get_left_rT_from_right_rT_fun(right_rT + vv * 6, right_to_left_rT, left_rT))
				return false;
			if (!ZQ_CameraProjection::project_points_fun(nPts, X3 + nPts * 3 * vv, left_rT, left_fc_cc_alpha_kc, left_fc_cc_alpha_kc + 2, left_fc_cc_alpha_kc + 5, left_fc_cc_alpha_kc[4], xp, zAxis_in))
				return false;
			ZQ_MathBase::VecMinus(nPts * 2, xp, left_X2 + nPts * 2 * vv, hx + left_X2_offset + nPts * 2 * vv);
		}
		return true;
	}

	template<class T>
	bool ZQ_CameraCalibrationBino::_calib_bino_with_known_intrinsic_jac(const T* p, T* jx, int m, int n, const void* data)
	{
		const Calib_Bino_Data_Header<T>* ptr = (const Calib_Bino_Data_Header<T>*)data;
		int nPts = ptr->n_pts;
		int nViews = ptr->n_views;
		const T* X3 = ptr->X3;
		const T* left_X2 = ptr->left_X2;
		const T* right_X2 = ptr->right_X2;

		const T* left_fc_cc_alpha_kc = ptr->left_fc_cc_alpha_kc;
		const T* right_fc_cc_alpha_kc = ptr->right_fc_cc_alpha_kc;

		bool zAxis_in = ptr->zAxis_in;

		const T* right_to_left_rT = p;
		const T* right_rT = p + 6;

		int left_X2_offset = 0;
		int right_X2_offset = nViews*nPts * 2 * (6 + nViews * 6);

		memset(jx, 0, sizeof(T)*m*n);

		ZQ_DImage<T> dxdrT_im(nPts * 2 * 6, 1, 1);
		ZQ_DImage<T> dxd_r2l_rT_im(nPts * 2 * 6, 1, 1);
		ZQ_DImage<T> dxd_r_rT_im(nPts * 2 * 6, 1, 1);
		T*& dxdrT = dxdrT_im.data();
		T*& dxd_r2l_rT = dxd_r2l_rT_im.data();
		T*& dxd_r_rT = dxd_r_rT_im.data();

		for (int vv = 0; vv < nViews; vv++)
		{
			if (!ZQ_CameraProjection::project_points_jac(nPts, X3, right_rT, right_fc_cc_alpha_kc, right_fc_cc_alpha_kc + 2, right_fc_cc_alpha_kc + 5, right_fc_cc_alpha_kc[4], dxdrT, (T*)0, (T*)0, (T*)0, (T*)0, zAxis_in))
				return false;
			for (int i = 0; i < nPts * 2; i++)
			{
				memcpy(jx + right_X2_offset + (vv*nPts * 2 + i)*(6 + nViews * 6) + vv * 6 + 6, dxdrT + i * 6, sizeof(T) * 6);
			}
		}
		for (int vv = 0; vv < nViews; vv++)
		{
			T left_rT[6], d_l_rT_d_r_rT[36], d_l_rT_d_r2l_rT[36];
			if (!_get_left_rT_from_right_rT_fun(right_rT + vv * 6, right_to_left_rT, left_rT))
				return false;
			if (!_get_left_rT_from_right_rT_jac(right_rT + vv * 6, right_to_left_rT, d_l_rT_d_r_rT, d_l_rT_d_r2l_rT))
				return false;

			if (!ZQ_CameraProjection::project_points_jac(nPts, X3, left_rT, left_fc_cc_alpha_kc, left_fc_cc_alpha_kc + 2, left_fc_cc_alpha_kc + 5, left_fc_cc_alpha_kc[4], dxdrT, (T*)0, (T*)0, (T*)0, (T*)0, zAxis_in))
				return false;

			ZQ_MathBase::MatrixMul(dxdrT, d_l_rT_d_r_rT, 2 * nPts, 6, 6, dxd_r_rT);
			ZQ_MathBase::MatrixMul(dxdrT, d_l_rT_d_r2l_rT, 2 * nPts, 6, 6, dxd_r2l_rT);

			for (int i = 0; i < nPts * 2; i++)
			{
				memcpy(jx + left_X2_offset + (vv*nPts * 2 + i)*(6 + nViews * 6) + vv * 6 + 6, dxd_r_rT + i * 6, sizeof(T) * 6);
				memcpy(jx + left_X2_offset + (vv*nPts * 2 + i)*(6 + nViews * 6), dxd_r2l_rT + i * 6, sizeof(T) * 6);
			}
		}

		return true;
	}

	template<class T>
	bool ZQ_CameraCalibrationBino::_calib_bino_with_known_intrinsic_jac_sparse(const T* p, taucs_ccs_matrix*& jx, int m, int n, const void* data)
	{
		const Calib_Bino_Data_Header<T>* ptr = (const Calib_Bino_Data_Header<T>*)data;
		int nPts = ptr->n_pts;
		int nViews = ptr->n_views;
		const T* X3 = ptr->X3;
		const T* left_X2 = ptr->left_X2;
		const T* right_X2 = ptr->right_X2;

		const T* left_fc_cc_alpha_kc = ptr->left_fc_cc_alpha_kc;
		const T* right_fc_cc_alpha_kc = ptr->right_fc_cc_alpha_kc;

		bool zAxis_in = ptr->zAxis_in;

		const T* right_to_left_rT = p;
		const T* right_rT = p + 6;

		int left_X2_offset = 0;
		int right_X2_offset = nViews*nPts * 2 * (6 + nViews * 6);

		ZQ_SparseMatrix<T> sp_jx_mat(n, m);

		ZQ_DImage<T> dxdrT_im(nPts * 2 * 6, 1, 1);
		ZQ_DImage<T> dxd_r2l_rT_im(nPts * 2 * 6, 1, 1);
		ZQ_DImage<T> dxd_r_rT_im(nPts * 2 * 6, 1, 1);
		T*& dxdrT = dxdrT_im.data();
		T*& dxd_r2l_rT = dxd_r2l_rT_im.data();
		T*& dxd_r_rT = dxd_r_rT_im.data();

		for (int vv = 0; vv < nViews; vv++)
		{
			if (!ZQ_CameraProjection::project_points_jac(nPts, X3, right_rT, right_fc_cc_alpha_kc, right_fc_cc_alpha_kc + 2, right_fc_cc_alpha_kc + 5, right_fc_cc_alpha_kc[4], dxdrT, (T*)0, (T*)0, (T*)0, (T*)0, zAxis_in))
				return false;

			int row_off = nViews * nPts * 2 + vv*nPts * 2;
			for (int i = 0; i < nPts * 2; i++)
			{
				int cur_row = row_off + i;
				for (int j = 0; j < 6; j++)
					sp_jx_mat.AddTo(cur_row, vv * 6 + 6 + j, dxdrT[i * 6 + j]);
			}
		}
		for (int vv = 0; vv < nViews; vv++)
		{
			T left_rT[6], d_l_rT_d_r_rT[36], d_l_rT_d_r2l_rT[36];
			if (!_get_left_rT_from_right_rT_fun(right_rT + vv * 6, right_to_left_rT, left_rT))
				return false;
			if (!_get_left_rT_from_right_rT_jac(right_rT + vv * 6, right_to_left_rT, d_l_rT_d_r_rT, d_l_rT_d_r2l_rT))
				return false;

			if (!ZQ_CameraProjection::project_points_jac(nPts, X3, left_rT, left_fc_cc_alpha_kc, left_fc_cc_alpha_kc + 2, left_fc_cc_alpha_kc + 5, left_fc_cc_alpha_kc[4], dxdrT, (T*)0, (T*)0, (T*)0, (T*)0, zAxis_in))
				return false;

			ZQ_MathBase::MatrixMul(dxdrT, d_l_rT_d_r_rT, 2 * nPts, 6, 6, dxd_r_rT);
			ZQ_MathBase::MatrixMul(dxdrT, d_l_rT_d_r2l_rT, 2 * nPts, 6, 6, dxd_r2l_rT);

			int row_off = vv*nPts * 2;
			for (int i = 0; i < nPts * 2; i++)
			{
				int cur_row = row_off + i;
				for (int j = 0; j < 6; j++)
					sp_jx_mat.AddTo(cur_row, vv * 6 + 6 + j, dxd_r_rT[i * 6 + j]);
				for (int j = 0; j < 6; j++)
					sp_jx_mat.AddTo(cur_row, j, dxd_r2l_rT[i * 6 + j]);
			}
		}

		if (jx)
		{
			ZQ_TaucsBase::ZQ_taucs_ccs_free(jx); jx = 0;
		}
		if (_strcmpi(typeid(T).name(), "float") == 0)
			jx = sp_jx_mat.ExportCCS(TAUCS_SINGLE);
		else if (_strcmpi(typeid(T).name(), "double") == 0)
			jx = sp_jx_mat.ExportCCS(TAUCS_DOUBLE);
		else
			return false;
		return true;
	}

	template<class T>
	bool ZQ_CameraCalibrationBino::_calib_bino_fun(const T* p, T* hx, int m, int n, const void* data)
	{
		const Calib_Bino_Data_Header<T>* ptr = (const Calib_Bino_Data_Header<T>*)data;
		int nPts = ptr->n_pts;
		int nViews = ptr->n_views;
		const T* X3 = ptr->X3;
		const T* left_X2 = ptr->left_X2;
		const T* right_X2 = ptr->right_X2;

		ZQ_CameraCalibrationMono::Calib_Method method = ptr->method;
		bool zAxis_in = ptr->zAxis_in;
		const T* right_to_left_rT;
		const T* right_rT;

		T left_fc[2], left_cc[2], left_alpha_c, left_kc[5];
		T right_fc[2], right_cc[2], right_alpha_c, right_kc[5];

		int intrin_num;
		switch (method)
		{
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K5:
			intrin_num = 10;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = p[4];
			memcpy(left_kc, p + 5, sizeof(T) * 5);
			right_fc[0] = p[10]; right_fc[1] = p[11];	right_cc[0] = p[12]; right_cc[1] = p[13];	right_alpha_c = p[14];
			memcpy(right_kc, p + 15, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K4:
			intrin_num = 9;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = p[4];
			memcpy(left_kc, p + 5, sizeof(T) * 4); left_kc[4] = ptr->left_fc_cc_alpha_kc[9];
			right_fc[0] = p[9]; right_fc[1] = p[10];	right_cc[0] = p[11]; right_cc[1] = p[12];	right_alpha_c = p[13];
			memcpy(right_kc, p + 14, sizeof(T) * 4); right_kc[4] = ptr->right_fc_cc_alpha_kc[9];
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K2:
			intrin_num = 7;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = p[4];
			memcpy(left_kc, p + 5, sizeof(T) * 2); memcpy(left_kc + 2, ptr->left_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			right_fc[0] = p[7]; right_fc[1] = p[8];	right_cc[0] = p[9]; right_cc[1] = p[10];	right_alpha_c = p[11];
			memcpy(right_kc, p + 12, sizeof(T) * 2); memcpy(right_kc + 2, ptr->right_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA:
			intrin_num = 5;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = p[4];
			memcpy(left_kc, ptr->left_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			right_fc[0] = p[5]; right_fc[1] = p[6];	right_cc[0] = p[7]; right_cc[1] = p[8];	right_alpha_c = p[9];
			memcpy(right_kc, ptr->right_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_K5:
			intrin_num = 9;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, p + 4, sizeof(T) * 5);
			right_fc[0] = p[9]; right_fc[1] = p[10];	right_cc[0] = p[11]; right_cc[1] = p[12];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, p + 13, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_K4:
			intrin_num = 8;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, p + 4, sizeof(T) * 4); left_kc[4] = ptr->left_fc_cc_alpha_kc[9];
			right_fc[0] = p[8]; right_fc[1] = p[9];	right_cc[0] = p[10]; right_cc[1] = p[11];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, p + 12, sizeof(T) * 4); right_kc[4] = ptr->right_fc_cc_alpha_kc[9];
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_K2:
			intrin_num = 6;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, p + 4, sizeof(T) * 2); memcpy(left_kc + 2, ptr->left_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			right_fc[0] = p[6]; right_fc[1] = p[7];	right_cc[0] = p[8]; right_cc[1] = p[9];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, p + 10, sizeof(T) * 2); memcpy(right_kc + 2, ptr->right_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C:
			intrin_num = 4;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, ptr->left_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			right_fc[0] = p[4]; right_fc[1] = p[5];	right_cc[0] = p[6]; right_cc[1] = p[7];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, ptr->right_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K5:
			intrin_num = 9;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = p[3];
			memcpy(left_kc, p + 4, sizeof(T) * 5);
			right_fc[0] = p[9]; right_fc[1] = p[9];	right_cc[0] = p[10]; right_cc[1] = p[11];	right_alpha_c = p[12];
			memcpy(right_kc, p + 13, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K4:
			intrin_num = 8;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = p[3];
			memcpy(left_kc, p + 4, sizeof(T) * 4); left_kc[4] = ptr->left_fc_cc_alpha_kc[9];
			right_fc[0] = p[8]; right_fc[1] = p[8];	right_cc[0] = p[9]; right_cc[1] = p[10];	right_alpha_c = p[11];
			memcpy(right_kc, p + 12, sizeof(T) * 4); right_kc[4] = ptr->right_fc_cc_alpha_kc[9];
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K2:
			intrin_num = 6;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = p[3];
			memcpy(left_kc, p + 4, sizeof(T) * 2); memcpy(left_kc + 2, ptr->left_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			right_fc[0] = p[6]; right_fc[1] = p[6];	right_cc[0] = p[7]; right_cc[1] = p[8];	right_alpha_c = p[9];
			memcpy(right_kc, p + 10, sizeof(T) * 4); memcpy(right_kc + 2, ptr->right_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA:
			intrin_num = 4;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = p[3];
			memcpy(left_kc, ptr->left_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			right_fc[0] = p[4]; right_fc[1] = p[4];	right_cc[0] = p[5]; right_cc[1] = p[6];	right_alpha_c = p[7];
			memcpy(right_kc, ptr->right_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_K5:
			intrin_num = 8;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, p + 3, sizeof(T) * 5);
			right_fc[0] = p[8]; right_fc[1] = p[8];	right_cc[0] = p[9]; right_cc[1] = p[10];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, p + 11, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_K4:
			intrin_num = 7;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, p + 3, sizeof(T) * 4); left_kc[4] = ptr->left_fc_cc_alpha_kc[9];
			right_fc[0] = p[7]; right_fc[1] = p[7];	right_cc[0] = p[8]; right_cc[1] = p[9];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, p + 10, sizeof(T) * 4); right_kc[4] = ptr->right_fc_cc_alpha_kc[9];
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_K2:
			intrin_num = 5;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, p + 3, sizeof(T) * 2); memcpy(left_kc + 2, ptr->left_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			right_fc[0] = p[5]; right_fc[1] = p[5];	right_cc[0] = p[6]; right_cc[1] = p[7];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, p + 8, sizeof(T) * 2); memcpy(right_kc + 2, ptr->right_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C:
			intrin_num = 3;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, ptr->left_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			right_fc[0] = p[3]; right_fc[1] = p[3];	right_cc[0] = p[4]; right_cc[1] = p[5];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, ptr->right_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			break;
		default:
			return false;
			break;
		}

		right_to_left_rT = p + intrin_num * 2;
		right_rT = right_to_left_rT + 6;

		int left_X2_offset = 0;
		int right_X2_offset = nViews*nPts * 2;

		ZQ_DImage<T> xp_im(nPts * 2, 1, 1);
		T*& xp = xp_im.data();
		for (int vv = 0; vv < nViews; vv++)
		{
			if (!ZQ_CameraProjection::project_points_fun(nPts, X3 + nPts * 3 * vv, right_rT + vv * 6, right_fc, right_cc, right_kc, right_alpha_c, xp, zAxis_in))
				return false;
			ZQ_MathBase::VecMinus(nPts * 2, xp, right_X2 + nPts * 2 * vv, hx + right_X2_offset + nPts * 2 * vv);
		}

		for (int vv = 0; vv < nViews; vv++)
		{
			T left_rT[6];
			if (!_get_left_rT_from_right_rT_fun(right_rT + vv * 6, right_to_left_rT, left_rT))
				return false;
			if (!ZQ_CameraProjection::project_points_fun(nPts, X3 + nPts * 3 * vv, left_rT, left_fc, left_cc, left_kc, left_alpha_c, xp, zAxis_in))
				return false;
			ZQ_MathBase::VecMinus(nPts * 2, xp, left_X2 + nPts * 2 * vv, hx + left_X2_offset + nPts * 2 * vv);
		}
		return true;
	}

	template<class T>
	bool ZQ_CameraCalibrationBino::_calib_bino_jac(const T* p, T* jx, int m, int n, const void* data)
	{
		const Calib_Bino_Data_Header<T>* ptr = (const Calib_Bino_Data_Header<T>*)data;
		int nPts = ptr->n_pts;
		int nViews = ptr->n_views;
		const T* X3 = ptr->X3;
		const T* left_X2 = ptr->left_X2;
		const T* right_X2 = ptr->right_X2;

		ZQ_CameraCalibrationMono::Calib_Method method = ptr->method;
		bool zAxis_in = ptr->zAxis_in;
		const T* right_to_left_rT;
		const T* right_rT;

		T left_fc[2], left_cc[2], left_alpha_c, left_kc[5];
		T right_fc[2], right_cc[2], right_alpha_c, right_kc[5];
		int intrin_num;
		switch (method)
		{
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K5:
			intrin_num = 10;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = p[4];
			memcpy(left_kc, p + 5, sizeof(T) * 5);
			right_fc[0] = p[10]; right_fc[1] = p[11];	right_cc[0] = p[12]; right_cc[1] = p[13];	right_alpha_c = p[14];
			memcpy(right_kc, p + 15, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K4:
			intrin_num = 9;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = p[4];
			memcpy(left_kc, p + 5, sizeof(T) * 4); left_kc[4] = ptr->left_fc_cc_alpha_kc[9];
			right_fc[0] = p[9]; right_fc[1] = p[10];	right_cc[0] = p[11]; right_cc[1] = p[12];	right_alpha_c = p[13];
			memcpy(right_kc, p + 14, sizeof(T) * 4); right_kc[4] = ptr->right_fc_cc_alpha_kc[9];
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K2:
			intrin_num = 7;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = p[4];
			memcpy(left_kc, p + 5, sizeof(T) * 2); memcpy(left_kc + 2, ptr->left_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			right_fc[0] = p[7]; right_fc[1] = p[8];	right_cc[0] = p[9]; right_cc[1] = p[10];	right_alpha_c = p[11];
			memcpy(right_kc, p + 12, sizeof(T) * 2); memcpy(right_kc + 2, ptr->right_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA:
			intrin_num = 5;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = p[4];
			memcpy(left_kc, ptr->left_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			right_fc[0] = p[5]; right_fc[1] = p[6];	right_cc[0] = p[7]; right_cc[1] = p[8];	right_alpha_c = p[9];
			memcpy(right_kc, ptr->right_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_K5:
			intrin_num = 9;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, p + 4, sizeof(T) * 5);
			right_fc[0] = p[9]; right_fc[1] = p[10];	right_cc[0] = p[11]; right_cc[1] = p[12];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, p + 13, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_K4:
			intrin_num = 8;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, p + 4, sizeof(T) * 4); left_kc[4] = ptr->left_fc_cc_alpha_kc[9];
			right_fc[0] = p[8]; right_fc[1] = p[9];	right_cc[0] = p[10]; right_cc[1] = p[11];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, p + 12, sizeof(T) * 4); right_kc[4] = ptr->right_fc_cc_alpha_kc[9];
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_K2:
			intrin_num = 6;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, p + 4, sizeof(T) * 2); memcpy(left_kc + 2, ptr->left_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			right_fc[0] = p[6]; right_fc[1] = p[7];	right_cc[0] = p[8]; right_cc[1] = p[9];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, p + 10, sizeof(T) * 2); memcpy(right_kc + 2, ptr->right_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C:
			intrin_num = 4;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, ptr->left_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			right_fc[0] = p[4]; right_fc[1] = p[5];	right_cc[0] = p[6]; right_cc[1] = p[7];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, ptr->right_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K5:
			intrin_num = 9;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = p[3];
			memcpy(left_kc, p + 4, sizeof(T) * 5);
			right_fc[0] = p[9]; right_fc[1] = p[9];	right_cc[0] = p[10]; right_cc[1] = p[11];	right_alpha_c = p[12];
			memcpy(right_kc, p + 13, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K4:
			intrin_num = 8;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = p[3];
			memcpy(left_kc, p + 4, sizeof(T) * 4); left_kc[4] = ptr->left_fc_cc_alpha_kc[9];
			right_fc[0] = p[8]; right_fc[1] = p[8];	right_cc[0] = p[9]; right_cc[1] = p[10];	right_alpha_c = p[11];
			memcpy(right_kc, p + 12, sizeof(T) * 4); right_kc[4] = ptr->right_fc_cc_alpha_kc[9];
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K2:
			intrin_num = 6;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = p[3];
			memcpy(left_kc, p + 4, sizeof(T) * 2); memcpy(left_kc + 2, ptr->left_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			right_fc[0] = p[6]; right_fc[1] = p[6];	right_cc[0] = p[7]; right_cc[1] = p[8];	right_alpha_c = p[9];
			memcpy(right_kc, p + 10, sizeof(T) * 4); memcpy(right_kc + 2, ptr->right_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA:
			intrin_num = 4;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = p[3];
			memcpy(left_kc, ptr->left_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			right_fc[0] = p[4]; right_fc[1] = p[4];	right_cc[0] = p[5]; right_cc[1] = p[6];	right_alpha_c = p[7];
			memcpy(right_kc, ptr->right_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_K5:
			intrin_num = 8;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, p + 3, sizeof(T) * 5);
			right_fc[0] = p[8]; right_fc[1] = p[8];	right_cc[0] = p[9]; right_cc[1] = p[10];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, p + 11, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_K4:
			intrin_num = 7;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, p + 3, sizeof(T) * 4); left_kc[4] = ptr->left_fc_cc_alpha_kc[9];
			right_fc[0] = p[7]; right_fc[1] = p[7];	right_cc[0] = p[8]; right_cc[1] = p[9];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, p + 10, sizeof(T) * 4); right_kc[4] = ptr->right_fc_cc_alpha_kc[9];
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_K2:
			intrin_num = 5;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, p + 3, sizeof(T) * 2); memcpy(left_kc + 2, ptr->left_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			right_fc[0] = p[5]; right_fc[1] = p[5];	right_cc[0] = p[6]; right_cc[1] = p[7];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, p + 8, sizeof(T) * 2); memcpy(right_kc + 2, ptr->right_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C:
			intrin_num = 3;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, ptr->left_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			right_fc[0] = p[3]; right_fc[1] = p[3];	right_cc[0] = p[4]; right_cc[1] = p[5];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, ptr->right_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			break;
		default:
			return false;
			break;
		}

		int unknown_num = intrin_num * 2 + nViews * 6 + 6;
		right_to_left_rT = p + intrin_num * 2;
		right_rT = right_to_left_rT + 6;

		int left_X2_offset = 0;
		int right_X2_offset = nViews*nPts * 2;

		memset(jx, 0, sizeof(T)*m*n);
		ZQ_DImage<T> dxdf_im(nPts * 2 * 2, 1, 1);
		ZQ_DImage<T> dxdc_im(nPts * 2 * 2, 1, 1);
		ZQ_DImage<T> dxdk_im(nPts * 2 * 5, 1, 1);
		ZQ_DImage<T> dxdalpha_im(nPts * 2, 1, 1);
		ZQ_DImage<T> dxdrT_im(nPts * 2 * 6, 1, 1);
		ZQ_DImage<T> dxd_r2l_rT_im(nPts * 2 * 6, 1, 1);
		ZQ_DImage<T> dxd_r_rT_im(nPts * 2 * 6, 1, 1);
		T*& dxdf = dxdf_im.data();
		T*& dxdc = dxdc_im.data();
		T*& dxdk = dxdk_im.data();
		T*& dxdalpha = dxdalpha_im.data();
		T*& dxdrT = dxdrT_im.data();
		T*& dxd_r2l_rT = dxd_r2l_rT_im.data();
		T*& dxd_r_rT = dxd_r_rT_im.data();


		for (int vv = 0; vv < nViews; vv++)
		{
			if (!ZQ_CameraProjection::project_points_jac(nPts, X3, right_rT, right_fc, right_cc, right_kc, right_alpha_c, dxdrT, dxdf, dxdc, dxdk, dxdalpha, zAxis_in))
				return false;

			int row_off = right_X2_offset + vv * 2 * nPts;
			switch (method)
			{
			case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + intrin_num + 0, dxdf + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 2, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 4, dxdalpha + i, sizeof(T) * 1);
					memcpy(jx + cur_row*m + intrin_num + 5, dxdk + i * 5, sizeof(T) * 5);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + intrin_num + 0, dxdf + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 2, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 4, dxdalpha + i, sizeof(T) * 1);
					memcpy(jx + cur_row*m + intrin_num + 5, dxdk + i * 5, sizeof(T) * 4);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + intrin_num + 0, dxdf + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 2, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 4, dxdalpha + i, sizeof(T) * 1);
					memcpy(jx + cur_row*m + intrin_num + 5, dxdk + i * 5, sizeof(T) * 2);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + intrin_num + 0, dxdf + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 2, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 4, dxdalpha + i, sizeof(T) * 1);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + intrin_num + 0, dxdf + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 2, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 4, dxdk + i * 5, sizeof(T) * 5);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + intrin_num + 0, dxdf + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 2, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 4, dxdk + i * 5, sizeof(T) * 4);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + intrin_num + 0, dxdf + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 2, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 4, dxdk + i * 5, sizeof(T) * 2);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + intrin_num + 0, dxdf + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 2, dxdc + i * 2, sizeof(T) * 2);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + intrin_num + 0] = dxdf[i * 2 + 0] + dxdf[i * 2 + 1];
					memcpy(jx + cur_row*m + intrin_num + 1, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 3, dxdalpha + i, sizeof(T) * 1);
					memcpy(jx + cur_row*m + intrin_num + 4, dxdk + i * 5, sizeof(T) * 5);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + intrin_num + 0] = dxdf[i * 2 + 0] + dxdf[i * 2 + 1];
					memcpy(jx + cur_row*m + intrin_num + 1, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 3, dxdalpha + i, sizeof(T) * 1);
					memcpy(jx + cur_row*m + intrin_num + 4, dxdk + i * 5, sizeof(T) * 4);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + intrin_num + 0] = dxdf[i * 2 + 0] + dxdf[i * 2 + 1];
					memcpy(jx + cur_row*m + intrin_num + 1, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 3, dxdalpha + i, sizeof(T) * 1);
					memcpy(jx + cur_row*m + intrin_num + 4, dxdk + i * 5, sizeof(T) * 2);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + intrin_num + 0] = dxdf[i * 2 + 0] + dxdf[i * 2 + 1];
					memcpy(jx + cur_row*m + intrin_num + 1, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 3, dxdalpha + i, sizeof(T) * 1);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + intrin_num + 0] = dxdf[i * 2 + 0] + dxdf[i * 2 + 1];
					memcpy(jx + cur_row*m + intrin_num + 1, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 3, dxdk + i * 5, sizeof(T) * 5);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + intrin_num + 0] = dxdf[i * 2 + 0] + dxdf[i * 2 + 1];
					memcpy(jx + cur_row*m + intrin_num + 1, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 3, dxdk + i * 5, sizeof(T) * 4);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + intrin_num + 0] = dxdf[i * 2 + 0] + dxdf[i * 2 + 1];
					memcpy(jx + cur_row*m + intrin_num + 1, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + intrin_num + 3, dxdk + i * 5, sizeof(T) * 2);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + intrin_num + 0] = dxdf[i * 2 + 0] + dxdf[i * 2 + 1];
					memcpy(jx + cur_row*m + intrin_num + 1, dxdc + i * 2, sizeof(T) * 2);
				}
				break;
			default:
				return false;
				break;
			}
			for (int i = 0; i < nPts * 2; i++)
			{
				int cur_row = row_off + i;
				memcpy(jx + cur_row*m + intrin_num * 2 + vv * 6 + 6, dxdrT + i * 6, sizeof(T) * 6);
			}
		}

		for (int vv = 0; vv < nViews; vv++)
		{
			T left_rT[6], d_l_rT_d_r_rT[36], d_l_rT_d_r2l_rT[36];
			if (!_get_left_rT_from_right_rT_fun(right_rT + vv * 6, right_to_left_rT, left_rT))
				return false;
			if (!_get_left_rT_from_right_rT_jac(right_rT + vv * 6, right_to_left_rT, d_l_rT_d_r_rT, d_l_rT_d_r2l_rT))
				return false;

			if (!ZQ_CameraProjection::project_points_jac(nPts, X3, left_rT, left_fc, left_cc, left_kc, left_alpha_c, dxdrT, dxdf, dxdc, dxdk, dxdalpha, zAxis_in))
				return false;

			int row_off = left_X2_offset + vv * 2 * nPts;
			switch (method)
			{
			case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + 0, dxdf + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 2, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 4, dxdalpha + i, sizeof(T) * 1);
					memcpy(jx + cur_row*m + 5, dxdk + i * 5, sizeof(T) * 5);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + 0, dxdf + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 2, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 4, dxdalpha + i, sizeof(T) * 1);
					memcpy(jx + cur_row*m + 5, dxdk + i * 5, sizeof(T) * 4);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + 0, dxdf + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 2, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 4, dxdalpha + i, sizeof(T) * 1);
					memcpy(jx + cur_row*m + 5, dxdk + i * 5, sizeof(T) * 2);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + 0, dxdf + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 2, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 4, dxdalpha + i, sizeof(T) * 1);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + 0, dxdf + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 2, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 4, dxdk + i * 5, sizeof(T) * 5);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + 0, dxdf + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 2, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 4, dxdk + i * 5, sizeof(T) * 4);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + 0, dxdf + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 2, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 4, dxdk + i * 5, sizeof(T) * 2);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + 0, dxdf + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 2, dxdc + i * 2, sizeof(T) * 2);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + 0] = dxdf[i * 2 + 0] + dxdf[i * 2 + 1];
					memcpy(jx + cur_row*m + 1, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 3, dxdalpha + i, sizeof(T) * 1);
					memcpy(jx + cur_row*m + 4, dxdk + i * 5, sizeof(T) * 5);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + 0] = dxdf[i * 2 + 0] + dxdf[i * 2 + 1];
					memcpy(jx + cur_row*m + 1, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 3, dxdalpha + i, sizeof(T) * 1);
					memcpy(jx + cur_row*m + 4, dxdk + i * 5, sizeof(T) * 4);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + 0] = dxdf[i * 2 + 0] + dxdf[i * 2 + 1];
					memcpy(jx + cur_row*m + 1, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 3, dxdalpha + i, sizeof(T) * 1);
					memcpy(jx + cur_row*m + 4, dxdk + i * 5, sizeof(T) * 2);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + 0] = dxdf[i * 2 + 0] + dxdf[i * 2 + 1];
					memcpy(jx + cur_row*m + 1, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 3, dxdalpha + i, sizeof(T) * 1);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + 0] = dxdf[i * 2 + 0] + dxdf[i * 2 + 1];
					memcpy(jx + cur_row*m + 1, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 3, dxdk + i * 5, sizeof(T) * 5);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + 0] = dxdf[i * 2 + 0] + dxdf[i * 2 + 1];
					memcpy(jx + cur_row*m + 1, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 3, dxdk + i * 5, sizeof(T) * 4);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + 0] = dxdf[i * 2 + 0] + dxdf[i * 2 + 1];
					memcpy(jx + cur_row*m + 1, dxdc + i * 2, sizeof(T) * 2);
					memcpy(jx + cur_row*m + 3, dxdk + i * 5, sizeof(T) * 2);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + 0] = dxdf[i * 2 + 0] + dxdf[i * 2 + 1];
					memcpy(jx + cur_row*m + 1, dxdc + i * 2, sizeof(T) * 2);
				}
				break;
			default:
				return false;
				break;
			}
			ZQ_MathBase::MatrixMul(dxdrT, d_l_rT_d_r_rT, 2 * nPts, 6, 6, dxd_r_rT);
			ZQ_MathBase::MatrixMul(dxdrT, d_l_rT_d_r2l_rT, 2 * nPts, 6, 6, dxd_r2l_rT);

			for (int i = 0; i < nPts * 2; i++)
			{
				int cur_row = row_off + i;
				memcpy(jx + left_X2_offset + cur_row*m + intrin_num * 2 + vv * 6 + 6, dxd_r_rT + i * 6, sizeof(T) * 6);
				memcpy(jx + left_X2_offset + cur_row*m + intrin_num * 2, dxd_r2l_rT + i * 6, sizeof(T) * 6);
			}
		}

		return true;
	}

	template<class T>
	bool ZQ_CameraCalibrationBino::_calib_bino_jac_sparse(const T* p, taucs_ccs_matrix*& jx, int m, int n, const void* data)
	{
		const Calib_Bino_Data_Header<T>* ptr = (const Calib_Bino_Data_Header<T>*)data;
		int nPts = ptr->n_pts;
		int nViews = ptr->n_views;
		const T* X3 = ptr->X3;
		const T* left_X2 = ptr->left_X2;
		const T* right_X2 = ptr->right_X2;

		ZQ_CameraCalibrationMono::Calib_Method method = ptr->method;
		bool zAxis_in = ptr->zAxis_in;
		const T* right_to_left_rT;
		const T* right_rT;

		T left_fc[2], left_cc[2], left_alpha_c, left_kc[5];
		T right_fc[2], right_cc[2], right_alpha_c, right_kc[5];
		int intrin_num;
		switch (method)
		{
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K5:
			intrin_num = 10;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = p[4];
			memcpy(left_kc, p + 5, sizeof(T) * 5);
			right_fc[0] = p[10]; right_fc[1] = p[11];	right_cc[0] = p[12]; right_cc[1] = p[13];	right_alpha_c = p[14];
			memcpy(right_kc, p + 15, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K4:
			intrin_num = 9;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = p[4];
			memcpy(left_kc, p + 5, sizeof(T) * 4); left_kc[4] = ptr->left_fc_cc_alpha_kc[9];
			right_fc[0] = p[9]; right_fc[1] = p[10];	right_cc[0] = p[11]; right_cc[1] = p[12];	right_alpha_c = p[13];
			memcpy(right_kc, p + 14, sizeof(T) * 4); right_kc[4] = ptr->right_fc_cc_alpha_kc[9];
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K2:
			intrin_num = 7;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = p[4];
			memcpy(left_kc, p + 5, sizeof(T) * 2); memcpy(left_kc + 2, ptr->left_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			right_fc[0] = p[7]; right_fc[1] = p[8];	right_cc[0] = p[9]; right_cc[1] = p[10];	right_alpha_c = p[11];
			memcpy(right_kc, p + 12, sizeof(T) * 2); memcpy(right_kc + 2, ptr->right_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA:
			intrin_num = 5;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = p[4];
			memcpy(left_kc, ptr->left_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			right_fc[0] = p[5]; right_fc[1] = p[6];	right_cc[0] = p[7]; right_cc[1] = p[8];	right_alpha_c = p[9];
			memcpy(right_kc, ptr->right_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_K5:
			intrin_num = 9;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, p + 4, sizeof(T) * 5);
			right_fc[0] = p[9]; right_fc[1] = p[10];	right_cc[0] = p[11]; right_cc[1] = p[12];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, p + 13, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_K4:
			intrin_num = 8;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, p + 4, sizeof(T) * 4); left_kc[4] = ptr->left_fc_cc_alpha_kc[9];
			right_fc[0] = p[8]; right_fc[1] = p[9];	right_cc[0] = p[10]; right_cc[1] = p[11];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, p + 12, sizeof(T) * 4); right_kc[4] = ptr->right_fc_cc_alpha_kc[9];
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_K2:
			intrin_num = 6;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, p + 4, sizeof(T) * 2); memcpy(left_kc + 2, ptr->left_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			right_fc[0] = p[6]; right_fc[1] = p[7];	right_cc[0] = p[8]; right_cc[1] = p[9];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, p + 10, sizeof(T) * 2); memcpy(right_kc + 2, ptr->right_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C:
			intrin_num = 4;
			left_fc[0] = p[0]; left_fc[1] = p[1];	left_cc[0] = p[2]; left_cc[1] = p[3];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, ptr->left_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			right_fc[0] = p[4]; right_fc[1] = p[5];	right_cc[0] = p[6]; right_cc[1] = p[7];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, ptr->right_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K5:
			intrin_num = 9;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = p[3];
			memcpy(left_kc, p + 4, sizeof(T) * 5);
			right_fc[0] = p[9]; right_fc[1] = p[9];	right_cc[0] = p[10]; right_cc[1] = p[11];	right_alpha_c = p[12];
			memcpy(right_kc, p + 13, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K4:
			intrin_num = 8;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = p[3];
			memcpy(left_kc, p + 4, sizeof(T) * 4); left_kc[4] = ptr->left_fc_cc_alpha_kc[9];
			right_fc[0] = p[8]; right_fc[1] = p[8];	right_cc[0] = p[9]; right_cc[1] = p[10];	right_alpha_c = p[11];
			memcpy(right_kc, p + 12, sizeof(T) * 4); right_kc[4] = ptr->right_fc_cc_alpha_kc[9];
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K2:
			intrin_num = 6;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = p[3];
			memcpy(left_kc, p + 4, sizeof(T) * 2); memcpy(left_kc + 2, ptr->left_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			right_fc[0] = p[6]; right_fc[1] = p[6];	right_cc[0] = p[7]; right_cc[1] = p[8];	right_alpha_c = p[9];
			memcpy(right_kc, p + 10, sizeof(T) * 4); memcpy(right_kc + 2, ptr->right_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA:
			intrin_num = 4;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = p[3];
			memcpy(left_kc, ptr->left_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			right_fc[0] = p[4]; right_fc[1] = p[4];	right_cc[0] = p[5]; right_cc[1] = p[6];	right_alpha_c = p[7];
			memcpy(right_kc, ptr->right_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_K5:
			intrin_num = 8;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, p + 3, sizeof(T) * 5);
			right_fc[0] = p[8]; right_fc[1] = p[8];	right_cc[0] = p[9]; right_cc[1] = p[10];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, p + 11, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_K4:
			intrin_num = 7;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, p + 3, sizeof(T) * 4); left_kc[4] = ptr->left_fc_cc_alpha_kc[9];
			right_fc[0] = p[7]; right_fc[1] = p[7];	right_cc[0] = p[8]; right_cc[1] = p[9];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, p + 10, sizeof(T) * 4); right_kc[4] = ptr->right_fc_cc_alpha_kc[9];
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_K2:
			intrin_num = 5;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, p + 3, sizeof(T) * 2); memcpy(left_kc + 2, ptr->left_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			right_fc[0] = p[5]; right_fc[1] = p[5];	right_cc[0] = p[6]; right_cc[1] = p[7];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, p + 8, sizeof(T) * 2); memcpy(right_kc + 2, ptr->right_fc_cc_alpha_kc + 7, sizeof(T) * 3);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C:
			intrin_num = 3;
			left_fc[0] = p[0]; left_fc[1] = p[0];	left_cc[0] = p[1]; left_cc[1] = p[2];	left_alpha_c = ptr->left_fc_cc_alpha_kc[4];
			memcpy(left_kc, ptr->left_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			right_fc[0] = p[3]; right_fc[1] = p[3];	right_cc[0] = p[4]; right_cc[1] = p[5];	right_alpha_c = ptr->right_fc_cc_alpha_kc[4];
			memcpy(right_kc, ptr->right_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			break;
		default:
			return false;
			break;
		}

		int unknown_num = intrin_num * 2 + nViews * 6 + 6;
		right_to_left_rT = p + intrin_num * 2;
		right_rT = right_to_left_rT + 6;

		int left_X2_offset = 0;
		int right_X2_offset = nViews*nPts * 2;

		ZQ_SparseMatrix<T> sp_jx_mat(n, m);

		ZQ_DImage<T> dxdf_im(nPts * 2 * 2, 1, 1);
		ZQ_DImage<T> dxdc_im(nPts * 2 * 2, 1, 1);
		ZQ_DImage<T> dxdk_im(nPts * 2 * 5, 1, 1);
		ZQ_DImage<T> dxdalpha_im(nPts * 2, 1, 1);
		ZQ_DImage<T> dxdrT_im(nPts * 2 * 6, 1, 1);
		ZQ_DImage<T> dxd_r2l_rT_im(nPts * 2 * 6, 1, 1);
		ZQ_DImage<T> dxd_r_rT_im(nPts * 2 * 6, 1, 1);
		T*& dxdf = dxdf_im.data();
		T*& dxdc = dxdc_im.data();
		T*& dxdk = dxdk_im.data();
		T*& dxdalpha = dxdalpha_im.data();
		T*& dxdrT = dxdrT_im.data();
		T*& dxd_r2l_rT = dxd_r2l_rT_im.data();
		T*& dxd_r_rT = dxd_r_rT_im.data();

		for (int vv = 0; vv < nViews; vv++)
		{
			if (!ZQ_CameraProjection::project_points_jac(nPts, X3, right_rT, right_fc, right_cc, right_kc, right_alpha_c, dxdrT, dxdf, dxdc, dxdk, dxdalpha, zAxis_in))
				return false;

			int row_off = right_X2_offset + vv * 2 * nPts;
			switch (method)
			{
			case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 2, dxdc[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, intrin_num + 4, dxdalpha[i]);
					for (int j = 0; j < 5; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 5, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 2, dxdc[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, intrin_num + 4, dxdalpha[i]);
					for (int j = 0; j < 4; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 5, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 2, dxdc[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, intrin_num + 4, dxdalpha[i]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 5, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 2, dxdc[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, intrin_num + 4, dxdalpha[i]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 2, dxdc[i * 2 + j]);
					for (int j = 0; j < 5; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 4, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 2, dxdc[i * 2 + j]);
					for (int j = 0; j < 4; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 4, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 2, dxdc[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 4, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 2, dxdc[i * 2 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + 0, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 1, dxdc[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, intrin_num + 3, dxdalpha[i]);
					for (int j = 0; j < 5; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 4, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + 0, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 1, dxdc[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, intrin_num + 3, dxdalpha[i]);
					for (int j = 0; j < 4; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 4, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + 0, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 1, dxdc[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, intrin_num + 3, dxdalpha[i]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 4, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + 0, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 1, dxdc[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, intrin_num + 3, dxdalpha[i]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + 0, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 1, dxdc[i * 2 + j]);
					for (int j = 0; j < 5; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 3, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + 0, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 1, dxdc[i * 2 + j]);
					for (int j = 0; j < 4; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 3, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + 0, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 1, dxdc[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 3, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + 0, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, intrin_num + j + 1, dxdc[i * 2 + j]);
				}
				break;
			default:
				return false;
				break;
			}

			for (int i = 0; i < nPts * 2; i++)
			{
				int cur_row = row_off + i;
				for (int j = 0; j < 6; j++)
					sp_jx_mat.AddTo(cur_row, intrin_num * 2 + vv * 6 + 6 + j, dxdrT[i * 6 + j]);
			}
		}

		for (int vv = 0; vv < nViews; vv++)
		{
			T left_rT[6], d_l_rT_d_r_rT[36], d_l_rT_d_r2l_rT[36];
			if (!_get_left_rT_from_right_rT_fun(right_rT + vv * 6, right_to_left_rT, left_rT))
				return false;
			if (!_get_left_rT_from_right_rT_jac(right_rT + vv * 6, right_to_left_rT, d_l_rT_d_r_rT, d_l_rT_d_r2l_rT))
				return false;

			if (!ZQ_CameraProjection::project_points_jac(nPts, X3, left_rT, left_fc, left_cc, left_kc, left_alpha_c, dxdrT, dxdf, dxdc, dxdk, dxdalpha, zAxis_in))
				return false;

			int row_off = left_X2_offset + vv * 2 * nPts;
			switch (method)
			{
			case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 2, dxdc[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, 4, dxdalpha[i]);
					for (int j = 0; j < 5; j++)
						sp_jx_mat.AddTo(cur_row, j + 5, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 2, dxdc[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, 4, dxdalpha[i]);
					for (int j = 0; j < 4; j++)
						sp_jx_mat.AddTo(cur_row, j + 5, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 2, dxdc[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, 4, dxdalpha[i]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 5, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 2, dxdc[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, 4, dxdalpha[i]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 2, dxdc[i * 2 + j]);
					for (int j = 0; j < 5; j++)
						sp_jx_mat.AddTo(cur_row, j + 4, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 2, dxdc[i * 2 + j]);
					for (int j = 0; j < 4; j++)
						sp_jx_mat.AddTo(cur_row, j + 4, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 2, dxdc[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 4, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F2_C:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 2, dxdc[i * 2 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, 0, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 1, dxdc[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, 3, dxdalpha[i]);
					for (int j = 0; j < 5; j++)
						sp_jx_mat.AddTo(cur_row, j + 4, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, 0, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 1, dxdc[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, 3, dxdalpha[i]);
					for (int j = 0; j < 4; j++)
						sp_jx_mat.AddTo(cur_row, j + 4, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, 0, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 1, dxdc[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, 3, dxdalpha[i]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 4, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, 0, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 1, dxdc[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, 3, dxdalpha[i]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, 0, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 1, dxdc[i * 2 + j]);
					for (int j = 0; j < 5; j++)
						sp_jx_mat.AddTo(cur_row, j + 3, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, 0, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 1, dxdc[i * 2 + j]);
					for (int j = 0; j < 4; j++)
						sp_jx_mat.AddTo(cur_row, j + 3, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, 0, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 1, dxdc[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 3, dxdk[i * 5 + j]);
				}
				break;
			case ZQ_CameraCalibrationMono::CALIB_F1_C:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, 0, dxdf[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 1, dxdc[i * 2 + j]);
				}
				break;
			default:
				return false;
				break;
			}

			ZQ_MathBase::MatrixMul(dxdrT, d_l_rT_d_r_rT, 2 * nPts, 6, 6, dxd_r_rT);
			ZQ_MathBase::MatrixMul(dxdrT, d_l_rT_d_r2l_rT, 2 * nPts, 6, 6, dxd_r2l_rT);

			for (int i = 0; i < nPts * 2; i++)
			{
				int cur_row = row_off + i;
				for (int j = 0; j < 6; j++)
					sp_jx_mat.AddTo(cur_row, intrin_num * 2 + vv * 6 + 6 + j, dxd_r_rT[i * 6 + j]);
				for (int j = 0; j < 6; j++)
					sp_jx_mat.AddTo(cur_row, intrin_num * 2 + j, dxd_r2l_rT[i * 6 + j]);
			}
		}

		if (jx)
		{
			ZQ_TaucsBase::ZQ_taucs_ccs_free(jx); jx = 0;
		}
		if (_strcmpi(typeid(T).name(), "float") == 0)
			jx = sp_jx_mat.ExportCCS(TAUCS_SINGLE);
		else if (_strcmpi(typeid(T).name(), "double") == 0)
			jx = sp_jx_mat.ExportCCS(TAUCS_DOUBLE);
		else
			return false;
		return true;
	}

	template<class T>
	bool ZQ_CameraCalibrationBino::_calib_bino_with_known_intrinsic_with_init(int nViews, int nPts, const T* X3, const T* left_X2, const T* right_X2,
		const T left_fc[2], const T left_cc[2], const T left_kc[5], const T left_alpha_c,
		const T right_fc[2], const T right_cc[2], const T right_kc[5], const T right_alpha_c, bool zAxis_in,
		int max_iter_levmar, T* right_rT, T* right_to_left_rT, double& avg_err_square, bool sparse_solver)
	{
		Calib_Bino_Data_Header<T> data;
		data.n_pts = nPts;
		data.n_views = nViews;
		T left_fc_cc_alpha_kc[10] =
		{
			left_fc[0], left_fc[1], left_cc[0], left_cc[1], left_alpha_c, left_kc[0], left_kc[1], left_kc[2], left_kc[3], left_kc[4]
		};
		T right_fc_cc_alpha_kc[10] =
		{
			right_fc[0], right_fc[1], right_cc[0], right_cc[1], right_alpha_c, right_kc[0], right_kc[1], right_kc[2], right_kc[3], right_kc[4]
		};
		data.left_fc_cc_alpha_kc = left_fc_cc_alpha_kc;
		data.right_fc_cc_alpha_kc = right_fc_cc_alpha_kc;
		data.X3 = X3;
		data.left_X2 = left_X2;
		data.right_X2 = right_X2;
		data.zAxis_in = zAxis_in;

		ZQ_DImage<T> hx_im(nViews*nPts * 4, 1);
		ZQ_DImage<T> p_im(nViews * 6 + 6, 1);
		T*& hx = hx_im.data();
		T*& p = p_im.data();
		memcpy(p, right_to_left_rT, sizeof(T) * 6);
		memcpy(p + 6, right_rT, sizeof(T)*nViews * 6);

		if (!sparse_solver)
		{
			ZQ_LevMarOptions opts;
			ZQ_LevMarReturnInfos infos;
			opts.tol_e_square = 1e-45;
			opts.tol_max_jte = 1e-45;
			opts.tol_dx_square = 1e-45;
			if (!ZQ_LevMar::ZQ_LevMar_Der<T>(_calib_bino_with_known_intrinsic_fun<T>, _calib_bino_with_known_intrinsic_jac<T>, p, hx,
				nViews * 6 + 6, nViews * 2 * nPts * 2, max_iter_levmar, opts, infos, &data))
			{
				return false;
			}
			avg_err_square = infos.final_e_square / (nPts * nViews * 4);
		}
		else
		{
			ZQ_SparseLevMarOptions opts;
			ZQ_SparseLevMarReturnInfos infos;
			opts.tol_e_square = 1e-45;
			opts.tol_max_jte = 1e-45;
			opts.tol_dx_square = 1e-45;
			if (!ZQ_SparseLevMar::ZQ_SparseLevMar_Der<T>(_calib_bino_with_known_intrinsic_fun<T>, _calib_bino_with_known_intrinsic_jac_sparse<T>, p, hx,
				nViews * 6 + 6, nViews * 2 * nPts * 2, max_iter_levmar, opts, infos, &data))
			{
				return false;
			}
			avg_err_square = infos.final_e_square / (nPts * nViews * 4);
		}


		memcpy(right_to_left_rT, p, sizeof(T) * 6);
		memcpy(right_rT, p + 6, sizeof(T) * 6 * nViews);

		return true;
	}

	template<class T>
	bool ZQ_CameraCalibrationBino::_calib_bino_with_init(int nViews, int nPts, const T* X3, const T* left_X2, const T* right_X2,
		T left_fc[2], T left_cc[2], T left_kc[5], T& left_alpha_c,
		T right_fc[2], T right_cc[2], T right_kc[5], T& right_alpha_c, ZQ_CameraCalibrationMono::Calib_Method method, bool zAxis_in,
		int max_iter_levmar, T* right_rT, T* right_to_left_rT, double& avg_err_square, bool sparse_solver)
	{
		Calib_Bino_Data_Header<T> data;
		data.n_pts = nPts;
		data.n_views = nViews;
		T left_fc_cc_alpha_kc[10] =
		{
			left_fc[0], left_fc[1], left_cc[0], left_cc[1], left_alpha_c, left_kc[0], left_kc[1], left_kc[2], left_kc[3], left_kc[4]
		};
		T right_fc_cc_alpha_kc[10] =
		{
			right_fc[0], right_fc[1], right_cc[0], right_cc[1], right_alpha_c, right_kc[0], right_kc[1], right_kc[2], right_kc[3], right_kc[4]
		};
		data.left_fc_cc_alpha_kc = left_fc_cc_alpha_kc;
		data.right_fc_cc_alpha_kc = right_fc_cc_alpha_kc;
		data.X3 = X3;
		data.left_X2 = left_X2;
		data.right_X2 = right_X2;
		data.method = method;
		data.zAxis_in = zAxis_in;

		ZQ_DImage<T> hx_im(nPts * 2 * nViews * 2, 1);
		ZQ_DImage<T> p_im(20 + nViews * 6 + 6, 1);
		T*& hx = hx_im.data();
		T*& p = p_im.data();

		int intrin_num;
		int unknown_num;
		switch (method)
		{
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K5:
			intrin_num = 10;
			memcpy(p, left_fc_cc_alpha_kc, sizeof(T) * 10);
			memcpy(p + 10, right_fc_cc_alpha_kc, sizeof(T) * 10);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K4:
			intrin_num = 9;
			memcpy(p, left_fc_cc_alpha_kc, sizeof(T) * 9);
			memcpy(p + 9, right_fc_cc_alpha_kc, sizeof(T) * 9);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K2:
			intrin_num = 7;
			memcpy(p, left_fc_cc_alpha_kc, sizeof(T) * 7);
			memcpy(p + 7, right_fc_cc_alpha_kc, sizeof(T) * 7);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA:
			intrin_num = 5;
			memcpy(p, left_fc_cc_alpha_kc, sizeof(T) * 5);
			memcpy(p + 5, right_fc_cc_alpha_kc, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_K5:
			intrin_num = 9;
			memcpy(p, left_fc_cc_alpha_kc, sizeof(T) * 4);
			memcpy(p + 4, left_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			memcpy(p + 9, right_fc_cc_alpha_kc, sizeof(T) * 4);
			memcpy(p + 13, right_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_K4:
			intrin_num = 8;
			memcpy(p, left_fc_cc_alpha_kc, sizeof(T) * 4);
			memcpy(p + 4, left_fc_cc_alpha_kc + 5, sizeof(T) * 4);
			memcpy(p + 8, right_fc_cc_alpha_kc, sizeof(T) * 4);
			memcpy(p + 12, right_fc_cc_alpha_kc + 5, sizeof(T) * 4);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_K2:
			intrin_num = 6;
			memcpy(p, left_fc_cc_alpha_kc, sizeof(T) * 4);
			memcpy(p + 4, left_fc_cc_alpha_kc + 5, sizeof(T) * 2);
			memcpy(p + 6, right_fc_cc_alpha_kc, sizeof(T) * 4);
			memcpy(p + 10, right_fc_cc_alpha_kc + 5, sizeof(T) * 2);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C:
			intrin_num = 4;
			memcpy(p, left_fc_cc_alpha_kc, sizeof(T) * 4);
			memcpy(p + 4, right_fc_cc_alpha_kc, sizeof(T) * 4);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K5:
			intrin_num = 9;
			p[0] = left_fc_cc_alpha_kc[0];
			memcpy(p + 1, left_fc_cc_alpha_kc + 2, sizeof(T) * 8);
			p[9] = right_fc_cc_alpha_kc[0];
			memcpy(p + 10, right_fc_cc_alpha_kc + 2, sizeof(T) * 8);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K4:
			intrin_num = 8;
			p[0] = left_fc_cc_alpha_kc[0];
			memcpy(p + 1, left_fc_cc_alpha_kc + 2, sizeof(T) * 7);
			p[8] = right_fc_cc_alpha_kc[0];
			memcpy(p + 9, right_fc_cc_alpha_kc + 2, sizeof(T) * 7);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K2:
			intrin_num = 6;
			p[0] = left_fc_cc_alpha_kc[0];
			memcpy(p + 1, left_fc_cc_alpha_kc + 2, sizeof(T) * 5);
			p[6] = right_fc_cc_alpha_kc[0];
			memcpy(p + 7, right_fc_cc_alpha_kc + 2, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA:
			intrin_num = 4;
			p[0] = left_fc_cc_alpha_kc[0];
			memcpy(p + 1, left_fc_cc_alpha_kc + 2, sizeof(T) * 3);
			p[4] = right_fc_cc_alpha_kc[0];
			memcpy(p + 5, right_fc_cc_alpha_kc + 2, sizeof(T) * 3);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_K5:
			intrin_num = 8;
			p[0] = left_fc_cc_alpha_kc[0];
			memcpy(p + 1, left_fc_cc_alpha_kc + 2, sizeof(T) * 2);
			memcpy(p + 3, left_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			p[8] = right_fc_cc_alpha_kc[0];
			memcpy(p + 9, right_fc_cc_alpha_kc + 2, sizeof(T) * 2);
			memcpy(p + 11, right_fc_cc_alpha_kc + 5, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_K4:
			intrin_num = 7;
			p[0] = left_fc_cc_alpha_kc[0];
			memcpy(p + 1, left_fc_cc_alpha_kc + 2, sizeof(T) * 2);
			memcpy(p + 3, left_fc_cc_alpha_kc + 5, sizeof(T) * 4);
			p[7] = right_fc_cc_alpha_kc[0];
			memcpy(p + 8, right_fc_cc_alpha_kc + 2, sizeof(T) * 2);
			memcpy(p + 10, right_fc_cc_alpha_kc + 5, sizeof(T) * 4);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_K2:
			intrin_num = 5;
			p[0] = left_fc_cc_alpha_kc[0];
			memcpy(p + 1, left_fc_cc_alpha_kc + 2, sizeof(T) * 2);
			memcpy(p + 3, left_fc_cc_alpha_kc + 5, sizeof(T) * 2);
			p[5] = right_fc_cc_alpha_kc[0];
			memcpy(p + 6, right_fc_cc_alpha_kc + 2, sizeof(T) * 2);
			memcpy(p + 8, right_fc_cc_alpha_kc + 5, sizeof(T) * 2);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C:
			intrin_num = 3;
			p[0] = left_fc_cc_alpha_kc[0];
			memcpy(p + 1, left_fc_cc_alpha_kc + 2, sizeof(T) * 2);
			p[3] = right_fc_cc_alpha_kc[0];
			memcpy(p + 4, right_fc_cc_alpha_kc + 2, sizeof(T) * 2);
			break;
		default:
			return false;
			break;
		}

		unknown_num = intrin_num * 2 + nViews * 6 + 6;
		memcpy(p + intrin_num * 2, right_to_left_rT, sizeof(T) * 6);
		memcpy(p + intrin_num * 2 + 6, right_rT, sizeof(T)*nViews * 6);
		/************************/

		if (!sparse_solver)
		{
			ZQ_LevMarOptions opts;
			ZQ_LevMarReturnInfos infos;
			opts.tol_e_square = 1e-45;
			opts.tol_max_jte = 1e-45;
			opts.tol_dx_square = 1e-45;
			if (!ZQ_LevMar::ZQ_LevMar_Der<T>(_calib_bino_fun<T>, _calib_bino_jac<T>, p, hx,
				unknown_num, nViews * 2 * nPts * 2, max_iter_levmar, opts, infos, &data))
			{
				return false;
			}
			avg_err_square = infos.final_e_square / (nPts * nViews * 2 * 2);
		}
		else
		{
			ZQ_SparseLevMarOptions opts;
			ZQ_SparseLevMarReturnInfos infos;
			opts.tol_e_square = 1e-45;
			opts.tol_max_jte = 1e-45;
			opts.tol_dx_square = 1e-45;
			if (!ZQ_SparseLevMar::ZQ_SparseLevMar_Der<T>(_calib_bino_fun<T>, _calib_bino_jac_sparse<T>, p, hx,
				unknown_num, nViews * 2 * nPts * 2, max_iter_levmar, opts, infos, &data))
			{
				return false;
			}
			avg_err_square = infos.final_e_square / (nPts * nViews * 2 * 2);
		}


		switch (method)
		{
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K5:
			memcpy(left_fc, p, sizeof(T) * 2);
			memcpy(left_cc, p + 2, sizeof(T) * 2);
			left_alpha_c = p[4];
			memcpy(left_kc, p + 5, sizeof(T) * 5);
			memcpy(right_fc, p + intrin_num, sizeof(T) * 2);
			memcpy(right_cc, p + 2 + intrin_num, sizeof(T) * 2);
			right_alpha_c = p[4 + intrin_num];
			memcpy(right_kc, p + 5 + intrin_num, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K4:
			memcpy(left_fc, p, sizeof(T) * 2);
			memcpy(left_cc, p + 2, sizeof(T) * 2);
			left_alpha_c = p[4];
			memcpy(left_kc, p + 5, sizeof(T) * 4);
			memcpy(right_fc, p + intrin_num, sizeof(T) * 2);
			memcpy(right_cc, p + 2 + intrin_num, sizeof(T) * 2);
			right_alpha_c = p[4 + intrin_num];
			memcpy(right_kc, p + 5 + intrin_num, sizeof(T) * 4);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA_K2:
			memcpy(left_fc, p, sizeof(T) * 2);
			memcpy(left_cc, p + 2, sizeof(T) * 2);
			left_alpha_c = p[4];
			memcpy(left_kc, p + 5, sizeof(T) * 2);
			memcpy(right_fc, p + intrin_num, sizeof(T) * 2);
			memcpy(right_cc, p + 2 + intrin_num, sizeof(T) * 2);
			right_alpha_c = p[4 + intrin_num];
			memcpy(right_kc, p + 5 + intrin_num, sizeof(T) * 2);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_ALPHA:
			memcpy(left_fc, p, sizeof(T) * 2);
			memcpy(left_cc, p + 2, sizeof(T) * 2);
			left_alpha_c = p[4];
			memcpy(right_fc, p + intrin_num, sizeof(T) * 2);
			memcpy(right_cc, p + 2 + intrin_num, sizeof(T) * 2);
			right_alpha_c = p[4 + intrin_num];
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_K5:
			memcpy(left_fc, p, sizeof(T) * 2);
			memcpy(left_cc, p + 2, sizeof(T) * 2);
			memcpy(left_kc, p + 4, sizeof(T) * 5);
			memcpy(right_fc, p + intrin_num, sizeof(T) * 2);
			memcpy(right_cc, p + 2 + intrin_num, sizeof(T) * 2);
			memcpy(right_kc, p + 4 + intrin_num, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_K4:
			memcpy(left_fc, p, sizeof(T) * 2);
			memcpy(left_cc, p + 2, sizeof(T) * 2);
			memcpy(left_kc, p + 4, sizeof(T) * 4);
			memcpy(right_fc, p + intrin_num, sizeof(T) * 2);
			memcpy(right_cc, p + 2 + intrin_num, sizeof(T) * 2);
			memcpy(right_kc, p + 4 + intrin_num, sizeof(T) * 4);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C_K2:
			memcpy(left_fc, p, sizeof(T) * 2);
			memcpy(left_cc, p + 2, sizeof(T) * 2);
			memcpy(left_kc, p + 4, sizeof(T) * 2);
			memcpy(right_fc, p + intrin_num, sizeof(T) * 2);
			memcpy(right_cc, p + 2 + intrin_num, sizeof(T) * 2);
			memcpy(right_kc, p + 4 + intrin_num, sizeof(T) * 2);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F2_C:
			memcpy(left_fc, p, sizeof(T) * 2);
			memcpy(left_cc, p + 2, sizeof(T) * 2);
			memcpy(right_fc, p + intrin_num, sizeof(T) * 2);
			memcpy(right_cc, p + 2 + intrin_num, sizeof(T) * 2);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K5:
			left_fc[0] = left_fc[1] = p[0];
			memcpy(left_cc, p + 1, sizeof(T) * 2);
			left_alpha_c = p[3];
			memcpy(left_kc, p + 4, sizeof(T) * 5);
			right_fc[0] = right_fc[1] = p[0 + intrin_num];
			memcpy(right_cc, p + 1 + intrin_num, sizeof(T) * 2);
			right_alpha_c = p[3 + intrin_num];
			memcpy(right_kc, p + 4 + intrin_num, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K4:
			left_fc[0] = left_fc[1] = p[0];
			memcpy(left_cc, p + 1, sizeof(T) * 2);
			left_alpha_c = p[3];
			memcpy(left_kc, p + 4, sizeof(T) * 4);
			right_fc[0] = right_fc[1] = p[0 + intrin_num];
			memcpy(right_cc, p + 1 + intrin_num, sizeof(T) * 2);
			right_alpha_c = p[3 + intrin_num];
			memcpy(right_kc, p + 4 + intrin_num, sizeof(T) * 4);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA_K2:
			left_fc[0] = left_fc[1] = p[0];
			memcpy(left_cc, p + 1, sizeof(T) * 2);
			left_alpha_c = p[3];
			memcpy(left_kc, p + 4, sizeof(T) * 2);
			right_fc[0] = right_fc[1] = p[0 + intrin_num];
			memcpy(right_cc, p + 1 + intrin_num, sizeof(T) * 2);
			right_alpha_c = p[3 + intrin_num];
			memcpy(right_kc, p + 4 + intrin_num, sizeof(T) * 2);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_ALPHA:
			left_fc[0] = left_fc[1] = p[0];
			memcpy(left_cc, p + 1, sizeof(T) * 2);
			left_alpha_c = p[3];
			right_fc[0] = right_fc[1] = p[0 + intrin_num];
			memcpy(right_cc, p + 1 + intrin_num, sizeof(T) * 2);
			right_alpha_c = p[3 + intrin_num];
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_K5:
			left_fc[0] = left_fc[1] = p[0];
			memcpy(left_cc, p + 1, sizeof(T) * 2);
			memcpy(left_kc, p + 3, sizeof(T) * 5);
			right_fc[0] = right_fc[1] = p[0 + intrin_num];
			memcpy(right_cc, p + 1 + intrin_num, sizeof(T) * 2);
			memcpy(right_kc, p + 3 + intrin_num, sizeof(T) * 5);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_K4:
			left_fc[0] = left_fc[1] = p[0];
			memcpy(left_cc, p + 1, sizeof(T) * 2);
			memcpy(left_kc, p + 3, sizeof(T) * 4);
			right_fc[0] = right_fc[1] = p[0 + intrin_num];
			memcpy(right_cc, p + 1 + intrin_num, sizeof(T) * 2);
			memcpy(right_kc, p + 3 + intrin_num, sizeof(T) * 4);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C_K2:
			left_fc[0] = left_fc[1] = p[0];
			memcpy(left_cc, p + 1, sizeof(T) * 2);
			memcpy(left_kc, p + 3, sizeof(T) * 2);
			right_fc[0] = right_fc[1] = p[0 + intrin_num];
			memcpy(right_cc, p + 1 + intrin_num, sizeof(T) * 2);
			memcpy(right_kc, p + 3 + intrin_num, sizeof(T) * 2);
			break;
		case ZQ_CameraCalibrationMono::CALIB_F1_C:
			left_fc[0] = left_fc[1] = p[0];
			memcpy(left_cc, p + 1, sizeof(T) * 2);
			right_fc[0] = right_fc[1] = p[0 + intrin_num];
			memcpy(right_cc, p + 1 + intrin_num, sizeof(T) * 2);
			break;
		default:
			return false;
			break;
		}

		memcpy(right_to_left_rT, p + intrin_num * 2, sizeof(T) * 6);
		memcpy(right_rT, p + intrin_num * 2 + 6, sizeof(T) * 6 * nViews);

		return true;
	}

	template<class T>
	bool ZQ_CameraCalibrationBino::_compute_err_calib(int nViews, int nPts, const T* X3, const T* left_X2, const T* right_X2, const T* right_rT, const T* right_to_left_rT,
		const T left_fc[2], const T left_cc[2], const T left_kc[5], const T left_alpha_c, const T right_fc[2], const T right_cc[2], const T right_kc[5], const T right_alpha_c,
		double& err_std, double& max_err, bool zAxis_in)
	{
		ZQ_DImage<T> err_x_im(nViews*nPts * 2 * 2, 1, 1);
		T*& err_x = err_x_im.data();
		ZQ_DImage<T> p_im(nViews * 6 + 6, 1, 1);
		T*& p = p_im.data();
		memcpy(p, right_to_left_rT, sizeof(T) * 6);
		memcpy(p + 6, right_rT, sizeof(T) * 6 * nViews);
		Calib_Bino_Data_Header<T> data;
		data.n_pts = nPts;
		data.n_views = nViews;
		T left_fc_cc_alpha_kc[10] =
		{
			left_fc[0], left_fc[1], left_cc[0], left_cc[1], left_alpha_c, left_kc[0], left_kc[1], left_kc[2], left_kc[3], left_kc[4]
		};
		T right_fc_cc_alpha_kc[10] =
		{
			right_fc[0], right_fc[1], right_cc[0], right_cc[1], right_alpha_c, right_kc[0], right_kc[1], right_kc[2], right_kc[3], right_kc[4]
		};
		data.left_fc_cc_alpha_kc = left_fc_cc_alpha_kc;
		data.right_fc_cc_alpha_kc = right_fc_cc_alpha_kc;
		data.X3 = X3;
		data.left_X2 = left_X2;
		data.right_X2 = right_X2;
		data.zAxis_in = zAxis_in;
		if (!_calib_bino_with_known_intrinsic_fun(p, err_x, nViews * 6 + 6, nViews*nPts * 4, &data))
			return false;

		max_err = 0;
		for (int i = 0; i < nViews*nPts * 4; i++)
		{
			max_err = __max(max_err, fabs(err_x[i]));
		}
		double mean_err = 0;
		for (int i = 0; i < nViews*nPts * 4; i++)
			mean_err += err_x[i];
		mean_err /= (nViews*nPts * 4);

		err_std = 0;
		for (int i = 0; i < nViews*nPts * 4; i++)
		{
			double diff = err_x[i] - mean_err;
			err_std += diff*diff;
		}
		err_std = sqrt(err_std / (nViews*nPts * 4 - 1));
		return true;
	}

	template<class T>
	bool ZQ_CameraCalibrationBino::_estimate_uncertainties(int nViews, int nPts, const T* X3, const T* right_rT, const T* right_to_left_rT, double sigma_x,
		const T left_fc[2], const T left_cc[2], const T left_kc[5], const T left_alpha_c, const T right_fc[2], const T right_cc[2], const T right_kc[5], const T right_alpha_c,
		T right_to_left_rT_err[6], bool zAxis_in)
	{
		int unknown_num = nViews * 6 + 6;
		ZQ_DImage<T> p_im(unknown_num, 1, 1);
		T*& p = p_im.data();
		memcpy(p, right_to_left_rT, sizeof(T) * 6);
		memcpy(p + 6, right_rT, sizeof(T) * 6 * nViews);
		Calib_Bino_Data_Header<T> data;
		data.n_pts = nPts;
		data.n_views = nViews;
		T left_fc_cc_alpha_kc[10] =
		{
			left_fc[0], left_fc[1], left_cc[0], left_cc[1], left_alpha_c, left_kc[0], left_kc[1], left_kc[2], left_kc[3], left_kc[4]
		};
		T right_fc_cc_alpha_kc[10] =
		{
			right_fc[0], right_fc[1], right_cc[0], right_cc[1], right_alpha_c, right_kc[0], right_kc[1], right_kc[2], right_kc[3], right_kc[4]
		};
		data.left_fc_cc_alpha_kc = left_fc_cc_alpha_kc;
		data.right_fc_cc_alpha_kc = right_fc_cc_alpha_kc;
		data.X3 = X3;
		data.left_X2 = 0;
		data.right_X2 = 0;
		data.zAxis_in = zAxis_in;


		ZQ_DImage<T> jx_im(nViews*nPts * 4 * unknown_num, 1);
		T*& jx = jx_im.data();
		if (!_calib_bino_with_known_intrinsic_jac(p, jx, unknown_num, nViews*nPts * 4, &data))
			return false;

		ZQ_Matrix<double> JJ2(unknown_num, unknown_num), inv_JJ2(unknown_num, unknown_num);

		for (int i = 0; i < unknown_num; i++)
		{
			for (int j = 0; j < unknown_num; j++)
			{
				double sum = 0;
				for (int k = 0; k < nViews*nPts * 4; k++)
				{
					sum += jx[k*unknown_num + i] * jx[k*unknown_num + j];
				}
				JJ2.SetData(i, j, sum);
			}
		}
		if (!ZQ_SVD::Invert(JJ2, inv_JJ2))
		{
			return false;
		}

		bool flag;
		for (int i = 0; i < 6; i++)
		{
			double val = inv_JJ2.GetData(i, i, flag);
			if (!flag || val < 0)
			{
				//return false;
			}
			right_to_left_rT_err[i] = sqrt(fabs(val))*sigma_x;
		}
		return true;
	}


	template<class T>
	bool ZQ_CameraCalibrationBino::CalibrateBinocularCamera(int nViews, int nPts, const T* X3, const T* left_X2, const T* right_X2, T right_to_left_rT[6], T * out_right_rT,
		const T left_fc[2], const T left_cc[2], const T left_kc[5], const T left_alpha_c, const T right_fc[2], const T right_cc[2], const T right_kc[5], const T right_alpha_c, bool zAxis_in,
		const bool* left_active_images/* = 0*/, const bool* right_active_images/* = 0*/, const T* left_rT/* = 0*/, const T* right_rT/* = 0*/, int max_iter/* = 300*/, bool sparse_solver /* = false*/, bool display /*= false*/)
	{
		ZQ_DImage<bool> active_images_im(nViews, 1, 1);
		bool*& active_images = active_images_im.data();
		if (left_active_images == 0 && right_active_images != 0)
		{
			memcpy(active_images, right_active_images, sizeof(bool)*nViews);
		}
		else if (left_active_images != 0 && right_active_images == 0)
		{
			memcpy(active_images, right_active_images, sizeof(bool)*nViews);
		}
		else if (left_active_images == 0 && right_active_images == 0)
		{
			memset(active_images, 1, sizeof(bool)*nViews);
		}
		else
		{
			for (int i = 0; i < nViews; i++)
				active_images[i] = left_active_images[i] && right_active_images[i];
		}
		ZQ_DImage<int> index_map_im(nViews, 1, 1);
		int*& index_map = index_map_im.data();
		for (int i = 0; i < nViews; i++)
			index_map[i] = -1;
		int active_num = 0;
		for (int i = 0; i < nViews; i++)
		{
			if (active_images[i])
			{
				index_map[i] = active_num++;
			}
		}
		if (active_num == 0)
			return false;
		/////////

		ZQ_DImage<T> tmp_X3_im(active_num * 3 * nPts, 1, 1);
		ZQ_DImage<T> tmp_left_X2_im(active_num * 2 * nPts, 1, 1);
		ZQ_DImage<T> tmp_right_X2_im(active_num * 2 * nPts, 1, 1);
		ZQ_DImage<T> tmp_left_rT_im(active_num * 6, 1, 1);
		ZQ_DImage<T> tmp_right_rT_im(active_num * 6, 1, 1);
		T*& tmp_X3 = tmp_X3_im.data();
		T*& tmp_left_X2 = tmp_left_X2_im.data();
		T*& tmp_right_X2 = tmp_right_X2_im.data();
		T*& tmp_left_rT = tmp_left_rT_im.data();
		T*& tmp_right_rT = tmp_right_rT_im.data();

		bool posit_as_init = true;
		for (int vv = 0; vv < nViews; vv++)
		{
			if (active_images[vv])
			{
				int idx = index_map[vv];
				memcpy(tmp_X3 + idx * 3 * nPts, X3 + vv * 3 * nPts, sizeof(T) * 3 * nPts);
				memcpy(tmp_left_X2 + idx * 2 * nPts, left_X2 + vv * 2 * nPts, sizeof(T) * 2 * nPts);
				memcpy(tmp_right_X2 + idx * 2 * nPts, right_X2 + vv * 2 * nPts, sizeof(T) * 2 * nPts);

				if (left_rT == 0/* || nViews < 50*/)
				{
					if (posit_as_init)
					{
						double avg_E = 0;
						if (!ZQ_CameraPoseEstimation::PositCoplanarRobust(nPts, X3 + vv * 3 * nPts, left_X2 + vv * 2 * nPts, left_fc, left_cc, left_kc, left_alpha_c, 20, 100, 1, tmp_left_rT + idx * 6, avg_E, zAxis_in))
							return false;
					}
					else
					{
						if (!ZQ_CameraCalibrationMono::_compute_extrinsic_param(1, nPts, left_X2 + vv * 2 * nPts, X3 + vv * 3 * nPts, left_fc, left_cc, left_kc, left_alpha_c, tmp_left_rT + idx * 6, active_images + vv, 100, 1e6, false, zAxis_in))
							return false;
					}
				}
				else
				{
					memcpy(tmp_left_rT + idx * 6, left_rT + vv * 6, sizeof(T) * 6);
				}

				if (right_rT == 0/* || nViews < 50*/)
				{
					if (posit_as_init)
					{
						double avg_E = 0;
						if (!ZQ_CameraPoseEstimation::PositCoplanarRobust(nPts, X3 + vv * 3 * nPts, right_X2 + vv * 2 * nPts, right_fc, right_cc, right_kc, right_alpha_c, 20, 100, 1, tmp_right_rT + idx * 6, avg_E, zAxis_in))
							return false;
					}
					else
					{
						if (!ZQ_CameraCalibrationMono::_compute_extrinsic_param(1, nPts, right_X2 + vv * 2 * nPts, X3 + vv * 3 * nPts, right_fc, right_cc, right_kc, right_alpha_c, tmp_right_rT + idx * 6, active_images + vv, 100, 1e6, false, zAxis_in))
							return false;
					}
				}
				else
				{
					memcpy(tmp_right_rT + idx * 6, right_rT + vv * 6, sizeof(T) * 6);
				}
			}
		}



		/********  init right_to_left_rT  Begin  *****/
		ZQ_DImage<T> tmp_r2l_rT_im(active_num, 6);
		T*& tmp_r2l_rT = tmp_r2l_rT_im.data();
		for (int vv = 0; vv < active_num; vv++)
		{
			T R_m2rc[9];
			ZQ_Rodrigues::ZQ_Rodrigues_r2R(tmp_right_rT + vv * 6, R_m2rc);
			T Mat_m2rc[16] =
			{
				R_m2rc[0], R_m2rc[1], R_m2rc[2], tmp_right_rT[3],
				R_m2rc[3], R_m2rc[4], R_m2rc[5], tmp_right_rT[4],
				R_m2rc[6], R_m2rc[7], R_m2rc[8], tmp_right_rT[5],
				0, 0, 0, 1
			};
			T R_m2lc[9];
			ZQ_Rodrigues::ZQ_Rodrigues_r2R(tmp_left_rT + vv * 6, R_m2lc);
			T Mat_m2lc[16] =
			{
				R_m2lc[0], R_m2lc[1], R_m2lc[2], tmp_left_rT[3],
				R_m2lc[3], R_m2lc[4], R_m2lc[5], tmp_left_rT[4],
				R_m2lc[6], R_m2lc[7], R_m2lc[8], tmp_left_rT[5],
				0, 0, 0, 1
			};
			T Mat_rc2m[16];
			ZQ_MathBase::MatrixInverse(Mat_m2rc, 4, Mat_rc2m);
			T Mat_rc2lc[16];
			ZQ_MathBase::MatrixMul(Mat_m2lc, Mat_rc2m, 4, 4, 4, Mat_rc2lc);
			T R_rc2lc[9] =
			{
				Mat_rc2lc[0], Mat_rc2lc[1], Mat_rc2lc[2],
				Mat_rc2lc[4], Mat_rc2lc[5], Mat_rc2lc[6],
				Mat_rc2lc[8], Mat_rc2lc[9], Mat_rc2lc[10]
			};
			T tmp_right_to_left_rT[6];
			if (!ZQ_Rodrigues::ZQ_Rodrigues_R2r(R_rc2lc, tmp_right_to_left_rT))
				return false;
			tmp_right_to_left_rT[3] = Mat_rc2lc[3]; tmp_right_to_left_rT[4] = Mat_rc2lc[7]; tmp_right_to_left_rT[5] = Mat_rc2lc[11];
			for (int i = 0; i < 6; i++)
				tmp_r2l_rT[i*active_num + vv] = tmp_right_to_left_rT[i];
		}
		for (int i = 0; i < 6; i++)
			ZQ_QuickSort::FindKthMax(tmp_r2l_rT + i*active_num, active_num, active_num / 2, right_to_left_rT[i]);

		/********  init right_to_left_rT and left_to_right_rT   End  *****/

		double avg_err_square;
		if (!_calib_bino_with_known_intrinsic_with_init(active_num, nPts, tmp_X3, tmp_left_X2, tmp_right_X2, left_fc, left_cc, left_kc, left_alpha_c,
			right_fc, right_cc, right_kc, right_alpha_c, zAxis_in, max_iter, tmp_right_rT, right_to_left_rT, avg_err_square, sparse_solver))
			return false;

		for (int vv = 0; vv < nViews; vv++)
		{
			int id = index_map[vv];
			if (id >= 0)
				memcpy(out_right_rT + vv * 6, tmp_right_rT + id * 6, sizeof(T) * 6);
			else
			{
				memset(out_right_rT + vv * 6, 0, sizeof(T) * 6);
			}
		}
		/***************************************/
		if (display)
			printf("avg_err_square = %f\n", avg_err_square);
		double err_std = 0, max_err = 0;
		if (!_compute_err_calib(active_num, nPts, tmp_X3, tmp_left_X2, tmp_right_X2, tmp_right_rT, right_to_left_rT, left_fc, left_cc, left_kc, left_alpha_c, right_fc, right_cc, right_kc, right_alpha_c, err_std, max_err, zAxis_in))
			return false;

		T right_to_left_rT_err[6];
		if (!_estimate_uncertainties(active_num, nPts, tmp_X3, tmp_right_rT, right_to_left_rT, err_std * 3, left_fc, left_cc, left_kc, left_alpha_c, right_fc, right_cc, right_kc, right_alpha_c, right_to_left_rT_err, zAxis_in))
			return false;
		if (display)
		{
			printf("Calibration results after optimization (with uncertainties):\n");
			printf("Right to Left Rotation:    r = [ %3.5f %3.5f %3.5f] <> [%3.5f %3.5f %3.5f]\n", right_to_left_rT[0], right_to_left_rT[1], right_to_left_rT[2], right_to_left_rT_err[0], right_to_left_rT_err[1], right_to_left_rT_err[2]);
			printf("Right to Left Translation: T = [ %3.5f %3.5f %3.5f] <> [%3.5f %3.5f %3.5f]\n", right_to_left_rT[3], right_to_left_rT[4], right_to_left_rT[5], right_to_left_rT_err[3], right_to_left_rT_err[4], right_to_left_rT_err[5]);
			printf("Pixel error: err_std = [ %3.5f]\n", err_std);
			printf("Pixel error: err_max = [ %3.5f]\n", max_err);
			printf("Note: The numerical errors are approximately three times the standard deviations (for reference).\n");
		}
		return true;
	}

	template<class T>
	bool ZQ_CameraCalibrationBino::CalibrateBinocularCamera2(int nViews, int nPts, const T* X3, const T* left_X2, const T* right_X2, T right_to_left_rT[6], T * out_right_rT,
		T left_fc[2], T left_cc[2], T left_kc[5], T& left_alpha_c, T right_fc[2], T right_cc[2], T right_kc[5], T& right_alpha_c, ZQ_CameraCalibrationMono::Calib_Method method, bool zAxis_in,
		const bool* left_active_images/* = 0*/, const bool* right_active_images/* = 0*/, const T* left_rT/* = 0*/, const T* right_rT/* = 0*/, int max_iter/* = 300*/, bool sparse_solver /* = false*/, bool display /*= false*/)
	{
		ZQ_DImage<bool> active_images_im(nViews, 1, 1);
		bool*& active_images = active_images_im.data();
		if (left_active_images == 0 && right_active_images != 0)
		{
			memcpy(active_images, right_active_images, sizeof(bool)*nViews);
		}
		else if (left_active_images != 0 && right_active_images == 0)
		{
			memcpy(active_images, right_active_images, sizeof(bool)*nViews);
		}
		else if (left_active_images == 0 && right_active_images == 0)
		{
			memset(active_images, 1, sizeof(bool)*nViews);
		}
		else
		{
			for (int i = 0; i < nViews; i++)
				active_images[i] = left_active_images[i] && right_active_images[i];
		}
		ZQ_DImage<int> index_map_im(nViews, 1, 1);
		int*& index_map = index_map_im.data();
		for (int i = 0; i < nViews; i++)
			index_map[i] = -1;
		int active_num = 0;
		for (int i = 0; i < nViews; i++)
		{
			if (active_images[i])
			{
				index_map[i] = active_num++;
			}
		}
		if (active_num == 0)
			return false;
		/////////

		ZQ_DImage<T> tmp_X3_im(active_num * 3 * nPts, 1, 1);
		ZQ_DImage<T> tmp_left_X2_im(active_num * 2 * nPts, 1, 1);
		ZQ_DImage<T> tmp_right_X2_im(active_num * 2 * nPts, 1, 1);
		ZQ_DImage<T> tmp_left_rT_im(active_num * 6, 1, 1);
		ZQ_DImage<T> tmp_right_rT_im(active_num * 6, 1, 1);
		T*& tmp_X3 = tmp_X3_im.data();
		T*& tmp_left_X2 = tmp_left_X2_im.data();
		T*& tmp_right_X2 = tmp_right_X2_im.data();
		T*& tmp_left_rT = tmp_left_rT_im.data();
		T*& tmp_right_rT = tmp_right_rT_im.data();

		bool posit_as_init = true;
		for (int vv = 0; vv < nViews; vv++)
		{
			if (active_images[vv])
			{
				int idx = index_map[vv];
				memcpy(tmp_X3 + idx * 3 * nPts, X3 + vv * 3 * nPts, sizeof(T) * 3 * nPts);
				memcpy(tmp_left_X2 + idx * 2 * nPts, left_X2 + vv * 2 * nPts, sizeof(T) * 2 * nPts);
				memcpy(tmp_right_X2 + idx * 2 * nPts, right_X2 + vv * 2 * nPts, sizeof(T) * 2 * nPts);

				if (left_rT == 0/* || nViews < 50*/)
				{
					if (posit_as_init)
					{
						double avg_E = 0;
						if (!ZQ_CameraPoseEstimation::PositCoplanarRobust(nPts, X3 + vv * 3 * nPts, left_X2 + vv * 2 * nPts, left_fc, left_cc, left_kc, left_alpha_c, 20, 100, 1, tmp_left_rT + idx * 6, avg_E, zAxis_in))
							return false;
					}
					else
					{
						if (!ZQ_CameraCalibrationMono::_compute_extrinsic_param(1, nPts, left_X2 + vv * 2 * nPts, X3 + vv * 3 * nPts, left_fc, left_cc, left_kc, left_alpha_c, tmp_left_rT + idx * 6, active_images + vv, 100, 1e6, false, zAxis_in))
							return false;
					}
				}
				else
				{
					memcpy(tmp_left_rT + idx * 6, left_rT + vv * 6, sizeof(T) * 6);
				}

				if (right_rT == 0/* || nViews < 50*/)
				{
					if (posit_as_init)
					{
						double avg_E = 0;
						if (!ZQ_CameraPoseEstimation::PositCoplanarRobust(nPts, X3 + vv * 3 * nPts, right_X2 + vv * 2 * nPts, right_fc, right_cc, right_kc, right_alpha_c, 20, 100, 1, tmp_right_rT + idx * 6, avg_E, zAxis_in))
							return false;
					}
					else
					{
						if (!ZQ_CameraCalibrationMono::_compute_extrinsic_param(1, nPts, right_X2 + vv * 2 * nPts, X3 + vv * 3 * nPts, right_fc, right_cc, right_kc, right_alpha_c, tmp_right_rT + idx * 6, active_images + vv, 100, 1e6, false, zAxis_in))
							return false;
					}
				}
				else
				{
					memcpy(tmp_right_rT + idx * 6, right_rT + vv * 6, sizeof(T) * 6);
				}
			}
		}



		/********  init right_to_left_rT  Begin  *****/
		ZQ_DImage<T> tmp_r2l_rT_im(active_num, 6);
		T*& tmp_r2l_rT = tmp_r2l_rT_im.data();
		for (int vv = 0; vv < active_num; vv++)
		{
			T R_m2rc[9];
			ZQ_Rodrigues::ZQ_Rodrigues_r2R(tmp_right_rT + vv * 6, R_m2rc);
			T Mat_m2rc[16] =
			{
				R_m2rc[0], R_m2rc[1], R_m2rc[2], tmp_right_rT[3],
				R_m2rc[3], R_m2rc[4], R_m2rc[5], tmp_right_rT[4],
				R_m2rc[6], R_m2rc[7], R_m2rc[8], tmp_right_rT[5],
				0, 0, 0, 1
			};
			T R_m2lc[9];
			ZQ_Rodrigues::ZQ_Rodrigues_r2R(tmp_left_rT + vv * 6, R_m2lc);
			T Mat_m2lc[16] =
			{
				R_m2lc[0], R_m2lc[1], R_m2lc[2], tmp_left_rT[3],
				R_m2lc[3], R_m2lc[4], R_m2lc[5], tmp_left_rT[4],
				R_m2lc[6], R_m2lc[7], R_m2lc[8], tmp_left_rT[5],
				0, 0, 0, 1
			};
			T Mat_rc2m[16];
			ZQ_MathBase::MatrixInverse(Mat_m2rc, 4, Mat_rc2m);
			T Mat_rc2lc[16];
			ZQ_MathBase::MatrixMul(Mat_m2lc, Mat_rc2m, 4, 4, 4, Mat_rc2lc);
			T R_rc2lc[9] =
			{
				Mat_rc2lc[0], Mat_rc2lc[1], Mat_rc2lc[2],
				Mat_rc2lc[4], Mat_rc2lc[5], Mat_rc2lc[6],
				Mat_rc2lc[8], Mat_rc2lc[9], Mat_rc2lc[10]
			};
			T tmp_right_to_left_rT[6];
			if (!ZQ_Rodrigues::ZQ_Rodrigues_R2r(R_rc2lc, tmp_right_to_left_rT))
				return false;
			tmp_right_to_left_rT[3] = Mat_rc2lc[3]; tmp_right_to_left_rT[4] = Mat_rc2lc[7]; tmp_right_to_left_rT[5] = Mat_rc2lc[11];
			for (int i = 0; i < 6; i++)
				tmp_r2l_rT[i*active_num + vv] = tmp_right_to_left_rT[i];
		}
		for (int i = 0; i < 6; i++)
			ZQ_QuickSort::FindKthMax(tmp_r2l_rT + i*active_num, active_num, active_num / 2, right_to_left_rT[i]);


		/********  init right_to_left_rT and left_to_right_rT   End  *****/

		double avg_err_square;
		if (!_calib_bino_with_init(active_num, nPts, tmp_X3, tmp_left_X2, tmp_right_X2, left_fc, left_cc, left_kc, left_alpha_c,
			right_fc, right_cc, right_kc, right_alpha_c, method, zAxis_in, max_iter, tmp_right_rT, right_to_left_rT, avg_err_square, sparse_solver))
			return false;

		for (int vv = 0; vv < nViews; vv++)
		{
			int id = index_map[vv];
			if (id >= 0)
				memcpy(out_right_rT + vv * 6, tmp_right_rT + id * 6, sizeof(T) * 6);
			else
			{
				memset(out_right_rT + vv * 6, 0, sizeof(T) * 6);
			}
		}
		/***************************************/
		if (display)
			printf("avg_err_square = %f\n", avg_err_square);
		double err_std = 0, max_err = 0;
		if (!_compute_err_calib(active_num, nPts, tmp_X3, tmp_left_X2, tmp_right_X2, tmp_right_rT, right_to_left_rT, left_fc, left_cc, left_kc, left_alpha_c, right_fc, right_cc, right_kc, right_alpha_c, err_std, max_err, zAxis_in))
			return false;

		T right_to_left_rT_err[6];
		if (!_estimate_uncertainties(active_num, nPts, tmp_X3, tmp_right_rT, right_to_left_rT, err_std * 3, left_fc, left_cc, left_kc, left_alpha_c, right_fc, right_cc, right_kc, right_alpha_c, right_to_left_rT_err, zAxis_in))
			return false;
		if (display)
		{
			printf("Calibration results after optimization (with uncertainties):\n");
			printf("Focal Length:     L_fc = [ %3.5f   %3.5f ]\n", left_fc[0], left_fc[1]);
			printf("Principal point:  L_cc = [ %3.5f   %3.5f ]\n", left_cc[0], left_cc[1]);
			printf("Skew:        L_alpha_c = [ %3.5f ]\n", left_alpha_c);
			printf("Distortion:       L_kc = [ %3.5f   %3.5f   %3.5f   %3.5f  %5.5f ]\n", left_kc[0], left_kc[1], left_kc[2], left_kc[3], left_kc[4]);
			printf("Focal Length:     R_fc = [ %3.5f   %3.5f ]\n", right_fc[0], right_fc[1]);
			printf("Principal point:  R_cc = [ %3.5f   %3.5f ]\n", right_cc[0], right_cc[1]);
			printf("Skew:        R_alpha_c = [ %3.5f ]\n", right_alpha_c);
			printf("Distortion:       R_kc = [ %3.5f   %3.5f   %3.5f   %3.5f  %5.5f ]\n", right_kc[0], right_kc[1], right_kc[2], right_kc[3], right_kc[4]);
			printf("Right to Left Rotation:    r = [ %3.5f %3.5f %3.5f] <> [%3.5f %3.5f %3.5f]\n", right_to_left_rT[0], right_to_left_rT[1], right_to_left_rT[2], right_to_left_rT_err[0], right_to_left_rT_err[1], right_to_left_rT_err[2]);
			printf("Right to Left Translation: T = [ %3.5f %3.5f %3.5f] <> [%3.5f %3.5f %3.5f]\n", right_to_left_rT[3], right_to_left_rT[4], right_to_left_rT[5], right_to_left_rT_err[3], right_to_left_rT_err[4], right_to_left_rT_err[5]);
			printf("Pixel error: err_std = [ %3.5f]\n", err_std);
			printf("Pixel error: err_max = [ %3.5f]\n", max_err);
			printf("Note: The numerical errors are approximately three times the standard deviations (for reference).\n");
		}
		return true;
	}
}

#endif