#ifndef _ZQ_CAMERA_CALIBRATION_MULTI_H_
#define _ZQ_CAMERA_CALIBRATION_MULTI_H_
#pragma once

#include "ZQ_CameraCalibrationMono.h"
#include "ZQ_CameraCalibrationBino.h"

namespace ZQ
{
	class ZQ_CameraCalibrationMulti
	{
	public:

		/*
		const T* X3;				//the checkboard coordinates
		const int* visible_num;		//how many checkboards can be seen in each cam
		const int* visible_offset;	//the offset for fetch visible checkboard idx
		const int* visible_idx;		//the visible checkboard idx.
		//From visible_idx[visible_offset[i]] to visible_idx[visible_offset[i]+visible_num[i]-1] is the visible idx for cam i.
		const T* X2;				//the 2D coordinates.
		//From X2[visible_offset[i]* n_pts *2] to X2[(visible_offset[i]+visible_num[i])*n_pts*2 -1] is for cam i.
		//From X2[visible_offset[i]* n_pts *2] to X2[visible_offset[i]*n_pts*2 + n_pts*2 -1] is for the first visible checkboard of cam i.
		const T* fc_cc_alpha_kc;	//[fc_cc_alpha_kc0, fc_cc_alpha_kc1, ...]
		*/
		template<class T>
		static bool CalibrateMultiCamera(int nCheckboards, int nPts, int nCams, const T* X3, const T* X2, const int* visible_num, const int* visible_offset,
			const int* visible_idx, const T* fc_cc_alpha_c_kc, bool zAxis_in, T* board_in_cam0_rT, T* cam0_to_othercam_rT, int max_iter = 300,
			bool sparse_solver = false, bool display = false)
		{

			double avg_err_square = 0;
			if (!_calib_multi_with_known_intrinsic(nCheckboards, nPts, nCams, X3, X2, visible_num, visible_offset, visible_idx, fc_cc_alpha_c_kc, zAxis_in,
				max_iter, board_in_cam0_rT, cam0_to_othercam_rT, avg_err_square, sparse_solver))
			{
				return false;
			}

			if (display)
				printf("avg_err_square = %f\n", avg_err_square);
			double err_std = 0, max_err = 0;
			if (!_compute_err_calib(nCheckboards, nPts, nCams, X3, X2, visible_num, visible_offset, visible_idx, board_in_cam0_rT, cam0_to_othercam_rT, fc_cc_alpha_c_kc, err_std, max_err, zAxis_in))
				return false;

			if (display)
			{
				printf("Calibration results after optimization:\n");
				printf("Pixel error: err_std = [ %3.5f]\n", err_std);
				printf("Pixel error: err_max = [ %3.5f]\n", max_err);
			}

			return true;
		}

	private:
		template<class T>
		class Calib_Multi_Data_Header
		{
		public:
			int n_checkboards;
			int n_pts;
			int n_cams;
			const T* X3;				//the checkboard coordinates
			const int* visible_num;		//how many checkboards can be seen in each cam
			const int* visible_offset;	//the offset for fetch visible checkboard idx 
			const int* visible_idx;		//the visible checkboard idx.  
										//From visible_idx[visible_offset[i]] to visible_idx[visible_offset[i]+visible_num[i]-1] is the visible idx for cam i.
			const T* X2;				//the 2D coordinates.  
										//From X2[visible_offset[i]* n_pts *2] to X2[(visible_offset[i]+visible_num[i])*n_pts*2 -1] is for cam i.
										//From X2[visible_offset[i]* n_pts *2] to X2[visible_offset[i]*n_pts*2 + n_pts*2 -1] is for the first visible checkboard of cam i.
			const T* fc_cc_alpha_kc;	//[fc_cc_alpha_kc0, fc_cc_alpha_kc1, ...]
			ZQ_CameraCalibrationMono::Calib_Method method;
			bool zAxis_in;
		};

		template<class T>
		static bool _calib_multi_with_known_intrinsic_fun(const T* p, T* hx, int m, int n, const void* data);

		template<class T>
		static bool _calib_multi_with_known_intrinsic_jac(const T* p, T* jx, int m, int n, const void* data);

		template<class T>
		static bool _calib_multi_with_known_intrinsic_jac_sparse(const T* p, taucs_ccs_matrix*& jx, int m, int n, const void* data);

		template<class T>
		static bool _calib_multi_with_known_intrinsic_with_init(int nCheckboards, int nPts, int nCams, const T* X3, const T* X2, const int* visible_num, const int* visible_offset, const int* visible_idx,
			const T* fc_cc_alpha_kc, bool zAxis_in, int max_iter_levmar, T* board_in_cam0_rT, T* cam0_to_othercam_rT, double& avg_err_square, bool sparse_solver);

		template<class T>
		static bool _calib_multi_with_known_intrinsic(int nCheckboards, int nPts, int nCams, const T* X3, const T* X2, const int* visible_num, const int* visible_offset, const int* visible_idx,
			const T* fc_cc_alpha_kc, bool zAxis_in, int max_iter_levmar, T* checkboard_rT_to_cam0, T* cam0_to_othercam_rT, double& avg_err_square, bool sparse_solver);

		/*Calibrate multi-camera.
		The intrinsic parameters are known.
		X3 is a series of checkboard coordinates.
		The exterior parameters are estimated by using "posit_coplanar_robust".
		visible_num, visible_offset,visible_idx and X2 must be valid.
		*/
		template<class T>
		static bool _calib_multi_with_known_intrinsic_init(int nCheckboards, int nPts, int nCams, const T* X3, const T* X2, const int* visible_num, const int* visible_offset, const int* visible_idx,
			const T* fc_cc_alpha_kc, bool zAxis_in, int max_iter_levmar, T* board_in_cam0_rT, T* cam0_to_othercam_rT, double& avg_err_square);

		template<class T>
		static bool _compute_err_calib(int nCheckboards, int nPts, int nCams, const T* X3, const T* X2, const int* visible_num, const int* visible_offset, const int* visible_idx,
			const T* board_in_cam0_rT, const T* cam0_to_othercam_rT, const T* fc_cc_alpha_kc, double& err_std, double& max_err, bool zAxis_in);
	};

	/*************************************************************************************************/

	template<class T>
	bool ZQ_CameraCalibrationMulti::_calib_multi_with_known_intrinsic_fun(const T* p, T* hx, int m, int n, const void* data)
	{
		const Calib_Multi_Data_Header<T>* ptr = (const Calib_Multi_Data_Header<T>*)data;
		int N = ptr->n_pts;
		int n_checkboards = ptr->n_checkboards;
		int n_cams = ptr->n_cams;
		const T* X3 = ptr->X3;
		const T* X2 = ptr->X2;
		const T* fc_cc_alpha_kc = ptr->fc_cc_alpha_kc;
		const int* visible_num = ptr->visible_num;
		const int* visible_offset = ptr->visible_offset;
		const int* visible_idx = ptr->visible_idx;
		const bool zAxis_in = ptr->zAxis_in;

		const T* checkboard_rT_to_cam0 = p;
		const T* cam0_to_othercam_rT = p + n_checkboards * 6;

		ZQ_DImage<T> xp_im(N * 2, 1, 1);
		T*& xp = xp_im.data();

		T checkboard_rT_to_cam_i[6];
		for (int i = 0; i < n_cams; i++)
		{
			const T* fc = fc_cc_alpha_kc + i * 10;
			const T* cc = fc_cc_alpha_kc + i * 10 + 2;
			const T alpha_c = *(fc_cc_alpha_kc + i * 10 + 4);
			const T* kc = fc_cc_alpha_kc + i * 10 + 5;
			int cur_vis_off = visible_offset[i];
			int cur_vis_num = visible_num[i];
			for (int j = 0; j < cur_vis_num; j++)
			{
				int cur_checkboard_idx = visible_idx[cur_vis_off + j];
				if (i == 0)
				{
					memcpy(checkboard_rT_to_cam_i, checkboard_rT_to_cam0 + cur_checkboard_idx * 6, sizeof(T) * 6);
				}
				else
				{
					if (!ZQ_CameraCalibrationBino::_get_left_rT_from_right_rT_fun(checkboard_rT_to_cam0 + cur_checkboard_idx * 6,
						cam0_to_othercam_rT + (i - 1) * 6, checkboard_rT_to_cam_i))
					{
						return false;
					}
				}
				if (!ZQ_CameraProjection::project_points_fun(N, X3, checkboard_rT_to_cam_i, fc, cc, kc, alpha_c, xp, zAxis_in))
				{
					return false;
				}
				ZQ_MathBase::VecMinus(N * 2, xp, X2 + (cur_vis_off + j)* N * 2, hx + (cur_vis_off + j)* N * 2);
			}
		}
		return true;

	}

	template<class T>
	bool ZQ_CameraCalibrationMulti::_calib_multi_with_known_intrinsic_jac(const T* p, T* jx, int m, int n, const void* data)
	{
		const Calib_Multi_Data_Header<T>* ptr = (const Calib_Multi_Data_Header<T>*)data;
		int N = ptr->n_pts;
		int n_checkboards = ptr->n_checkboards;
		int n_cams = ptr->n_cams;
		const T* X3 = ptr->X3;
		const T* X2 = ptr->X2;
		const T* fc_cc_alpha_kc = ptr->fc_cc_alpha_kc;
		const int* visible_num = ptr->visible_num;
		const int* visible_offset = ptr->visible_offset;
		const int* visible_idx = ptr->visible_idx;
		const bool zAxis_in = ptr->zAxis_in;

		memset(jx, 0, sizeof(T)*m*n);

		int cols = n_checkboards * 6 + (n_cams - 1) * 6;
		const T* checkboard_rT_to_cam0 = p;
		const T* cam0_to_othercam_rT = p + n_checkboards * 6;
		int total_visible_num = visible_offset[n_cams - 1] + visible_num[n_cams - 1];
		T checkboard_rT_to_cam_i[6];

		ZQ_DImage<T> dxdrT_im(N * 2 * 6, 1, 1);
		ZQ_DImage<T> dxd_r2l_rT_im(N * 2 * 6, 1, 1);
		ZQ_DImage<T> dxd_r_rT_im(N * 2 * 6, 1, 1);
		T*& dxdrT = dxdrT_im.data();
		T*& dxd_r2l_rT = dxd_r2l_rT_im.data();
		T*& dxd_r_rT = dxd_r_rT_im.data();

		//for(int i = 0;i < 1;i++)
		{
			const T* fc = fc_cc_alpha_kc;
			const T* cc = fc_cc_alpha_kc + 2;
			const T alpha_c = *(fc_cc_alpha_kc + 4);
			const T* kc = fc_cc_alpha_kc + 5;
			int cur_vis_off = visible_offset[0];
			int cur_vis_num = visible_num[0];
			for (int j = 0; j < cur_vis_num; j++)
			{
				int cur_checkboard_idx = visible_idx[cur_vis_off + j];
				const T* right_rT = checkboard_rT_to_cam0 + cur_checkboard_idx * 6;
				if (!ZQ_CameraProjection::project_points_jac(N, X3, right_rT, fc, cc, kc, alpha_c, dxdrT, (T*)0, (T*)0, (T*)0, (T*)0, zAxis_in))
					return false;

				int right_X2_offset = (cur_vis_off + j)*N * 2 * cols;
				for (int rr = 0; rr < N * 2; rr++)
				{
					T* cur_row = jx + right_X2_offset + rr*cols;
					memcpy(cur_row + cur_checkboard_idx * 6, dxdrT + rr * 6, sizeof(T) * 6);
				}
			}
		}

		for (int i = 1; i < n_cams; i++)
		{
			const T* fc = fc_cc_alpha_kc + i * 10;
			const T* cc = fc_cc_alpha_kc + i * 10 + 2;
			const T alpha_c = *(fc_cc_alpha_kc + i * 10 + 4);
			const T* kc = fc_cc_alpha_kc + i * 10 + 5;
			int cur_vis_off = visible_offset[i];
			int cur_vis_num = visible_num[i];
			for (int j = 0; j < cur_vis_num; j++)
			{
				int cur_checkboard_idx = visible_idx[cur_vis_off + j];
				T left_rT[6], d_l_rT_d_r_rT[36], d_l_rT_d_r2l_rT[36];
				const T* right_rT = checkboard_rT_to_cam0 + cur_checkboard_idx * 6;
				const T* right_to_left_rT = cam0_to_othercam_rT + (i - 1) * 6;
				if (!ZQ_CameraCalibrationBino::_get_left_rT_from_right_rT_fun(right_rT, right_to_left_rT, left_rT))
					return false;
				if (!ZQ_CameraCalibrationBino::_get_left_rT_from_right_rT_jac(right_rT, right_to_left_rT, d_l_rT_d_r_rT, d_l_rT_d_r2l_rT))
					return false;

				if (!ZQ_CameraProjection::project_points_jac(N, X3, left_rT, fc, cc, kc, alpha_c, dxdrT, (T*)0, (T*)0, (T*)0, (T*)0, zAxis_in))
					return false;

				ZQ_MathBase::MatrixMul(dxdrT, d_l_rT_d_r_rT, 2 * N, 6, 6, dxd_r_rT);
				ZQ_MathBase::MatrixMul(dxdrT, d_l_rT_d_r2l_rT, 2 * N, 6, 6, dxd_r2l_rT);

				int left_X2_offset = (cur_vis_off + j)*N * 2 * cols;
				for (int rr = 0; rr < N * 2; rr++)
				{
					T* cur_row = jx + left_X2_offset + rr*cols;
					memcpy(cur_row + cur_checkboard_idx * 6, dxd_r_rT + rr * 6, sizeof(T) * 6);
					memcpy(cur_row + n_checkboards * 6 + (i - 1) * 6, dxd_r2l_rT + rr * 6, sizeof(T) * 6);
				}
			}
		}
		return true;
	}

	template<class T>
	bool ZQ_CameraCalibrationMulti::_calib_multi_with_known_intrinsic_jac_sparse(const T* p, taucs_ccs_matrix*& jx, int m, int n, const void* data)
	{
		const Calib_Multi_Data_Header<T>* ptr = (const Calib_Multi_Data_Header<T>*)data;
		int N = ptr->n_pts;
		int n_checkboards = ptr->n_checkboards;
		int n_cams = ptr->n_cams;
		const T* X3 = ptr->X3;
		const T* X2 = ptr->X2;
		const T* fc_cc_alpha_kc = ptr->fc_cc_alpha_kc;
		const int* visible_num = ptr->visible_num;
		const int* visible_offset = ptr->visible_offset;
		const int* visible_idx = ptr->visible_idx;
		const bool zAxis_in = ptr->zAxis_in;

		ZQ_SparseMatrix<T> sp_jx_mat(n, m);
		int cols = n_checkboards * 6 + (n_cams - 1) * 6;
		const T* checkboard_rT_to_cam0 = p;
		const T* cam0_to_othercam_rT = p + n_checkboards * 6;
		int total_visible_num = visible_offset[n_cams - 1] + visible_num[n_cams - 1];
		T checkboard_rT_to_cam_i[6];

		ZQ_DImage<T> dxdrT_im(N * 2 * 6, 1, 1);
		ZQ_DImage<T> dxd_r2l_rT_im(N * 2 * 6, 1, 1);
		ZQ_DImage<T> dxd_r_rT_im(N * 2 * 6, 1, 1);
		T*& dxdrT = dxdrT_im.data();
		T*& dxd_r2l_rT = dxd_r2l_rT_im.data();
		T*& dxd_r_rT = dxd_r_rT_im.data();

		//for(int i = 0;i < 1;i++)
		{
			const T* fc = fc_cc_alpha_kc;
			const T* cc = fc_cc_alpha_kc + 2;
			const T alpha_c = *(fc_cc_alpha_kc + 4);
			const T* kc = fc_cc_alpha_kc + 5;
			int cur_vis_off = visible_offset[0];
			int cur_vis_num = visible_num[0];
			for (int j = 0; j < cur_vis_num; j++)
			{
				int cur_checkboard_idx = visible_idx[cur_vis_off + j];
				const T* right_rT = checkboard_rT_to_cam0 + cur_checkboard_idx * 6;
				if (!ZQ_CameraProjection::project_points_jac(N, X3, right_rT, fc, cc, kc, alpha_c, dxdrT, (T*)0, (T*)0, (T*)0, (T*)0, zAxis_in))
					return false;

				int right_X2_offset = (cur_vis_off + j)*N * 2;
				for (int rr = 0; rr < N * 2; rr++)
				{
					int cur_row = right_X2_offset + rr;
					for (int tt = 0; tt < 6; tt++)
						sp_jx_mat.AddTo(cur_row, cur_checkboard_idx * 6 + tt, dxdrT[rr * 6 + tt]);
				}
			}
		}

		for (int i = 1; i < n_cams; i++)
		{
			const T* fc = fc_cc_alpha_kc + i * 10;
			const T* cc = fc_cc_alpha_kc + i * 10 + 2;
			const T alpha_c = *(fc_cc_alpha_kc + i * 10 + 4);
			const T* kc = fc_cc_alpha_kc + i * 10 + 5;
			int cur_vis_off = visible_offset[i];
			int cur_vis_num = visible_num[i];
			for (int j = 0; j < cur_vis_num; j++)
			{
				int cur_checkboard_idx = visible_idx[cur_vis_off + j];
				T left_rT[6], d_l_rT_d_r_rT[36], d_l_rT_d_r2l_rT[36];
				const T* right_rT = checkboard_rT_to_cam0 + cur_checkboard_idx * 6;
				const T* right_to_left_rT = cam0_to_othercam_rT + (i - 1) * 6;
				if (!ZQ_CameraCalibrationBino::_get_left_rT_from_right_rT_fun(right_rT, right_to_left_rT, left_rT))
					return false;
				if (!ZQ_CameraCalibrationBino::_get_left_rT_from_right_rT_jac(right_rT, right_to_left_rT, d_l_rT_d_r_rT, d_l_rT_d_r2l_rT))
					return false;

				if (!ZQ_CameraProjection::project_points_jac(N, X3, left_rT, fc, cc, kc, alpha_c, dxdrT, (T*)0, (T*)0, (T*)0, (T*)0, zAxis_in))
					return false;

				ZQ_MathBase::MatrixMul(dxdrT, d_l_rT_d_r_rT, 2 * N, 6, 6, dxd_r_rT);
				ZQ_MathBase::MatrixMul(dxdrT, d_l_rT_d_r2l_rT, 2 * N, 6, 6, dxd_r2l_rT);

				int left_X2_offset = (cur_vis_off + j)*N * 2;
				for (int rr = 0; rr < N * 2; rr++)
				{
					int cur_row = left_X2_offset + rr;
					for (int tt = 0; tt < 6; tt++)
						sp_jx_mat.AddTo(cur_row, cur_checkboard_idx * 6 + tt, dxd_r_rT[rr * 6 + tt]);
					for (int tt = 0; tt < 6; tt++)
						sp_jx_mat.AddTo(cur_row, n_checkboards * 6 + (i - 1) * 6 + tt, dxd_r2l_rT[rr * 6 + tt]);
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
		}
		return true;
	}

	/*Calibrate multi-camera.
	The intrinsic parameters are known.
	X3 is a series of checkboard coordinates.
	The exterior parameters are estimated by using "posit_coplanar_robust".
	visible_num, visible_offset,visible_idx and X2 must be valid.
	*/
	template<class T>
	bool ZQ_CameraCalibrationMulti::_calib_multi_with_known_intrinsic_with_init(int nCheckboards, int nPts, int nCams, const T* X3, const T* X2,
		const int* visible_num, const int* visible_offset, const int* visible_idx,
		const T* fc_cc_alpha_kc, bool zAxis_in, int max_iter_levmar, T* board_in_cam0_rT, T* cam0_to_othercam_rT, double& avg_err_square, bool sparse_solver)
	{
		Calib_Multi_Data_Header<T> data;
		data.n_checkboards = nCheckboards;
		data.n_pts = nPts;
		data.n_cams = nCams;
		data.X3 = X3;
		data.X2 = X2;
		data.visible_num = visible_num;
		data.visible_offset = visible_offset;
		data.visible_idx = visible_idx;
		data.fc_cc_alpha_kc = fc_cc_alpha_kc;
		data.zAxis_in = zAxis_in;

		int unknown_num = nCheckboards * 6 + (nCams - 1) * 6;
		int measure_num = nCheckboards*nPts * 2 * nCams;
		ZQ_DImage<T> hx_im(measure_num, 1);
		ZQ_DImage<T> p_im(unknown_num, 1);
		T*& hx = hx_im.data();
		T*& p = p_im.data();
		memcpy(p, board_in_cam0_rT, sizeof(T) * 6 * nCheckboards);
		memcpy(p + nCheckboards * 6, cam0_to_othercam_rT, sizeof(T) * 6 * (nCams - 1));

		if (!sparse_solver)
		{
			ZQ_LevMarOptions opts;
			ZQ_LevMarReturnInfos infos;
			opts.tol_e_square = 1e-45;
			opts.tol_max_jte = 1e-45;
			opts.tol_dx_square = 1e-45;
			if (!ZQ_LevMar::ZQ_LevMar_Der<T>(_calib_multi_with_known_intrinsic_fun<T>, _calib_multi_with_known_intrinsic_jac<T>, p, hx,
				unknown_num, measure_num, max_iter_levmar, opts, infos, &data))
			{
				return false;
			}
			avg_err_square = infos.final_e_square / (measure_num);
		}
		else
		{
			ZQ_SparseLevMarOptions opts;
			ZQ_SparseLevMarReturnInfos infos;
			opts.tol_e_square = 1e-45;
			opts.tol_max_jte = 1e-45;
			opts.tol_dx_square = 1e-45;
			if (!ZQ_SparseLevMar::ZQ_SparseLevMar_Der<T>(_calib_multi_with_known_intrinsic_fun<T>, _calib_multi_with_known_intrinsic_jac_sparse<T>, p, hx,
				unknown_num, measure_num, max_iter_levmar, opts, infos, &data))
			{
				return false;
			}
			avg_err_square = infos.final_e_square / (measure_num);
		}

		memcpy(board_in_cam0_rT, p, sizeof(T) * 6 * nCheckboards);
		memcpy(cam0_to_othercam_rT, p + nCheckboards * 6, sizeof(T) * 6 * (nCams - 1));

		return true;
	}

	template<class T>
	static bool ZQ_CameraCalibrationMulti::_calib_multi_with_known_intrinsic(int nCheckboards, int nPts, int nCams, const T* X3, const T* X2, const int* visible_num, const int* visible_offset, const int* visible_idx,
		const T* fc_cc_alpha_kc, bool zAxis_in, int max_iter_levmar, T* checkboard_rT_to_cam0, T* cam0_to_othercam_rT, double& avg_err_square, bool sparse_solver)
	{

		T* tmp_checkboard_rT_to_cam0;
		if (checkboard_rT_to_cam0 == 0)
			tmp_checkboard_rT_to_cam0 = new T[nCheckboards * 6];
		else
			tmp_checkboard_rT_to_cam0 = checkboard_rT_to_cam0;
		double tol_E = 0;
		if (!_calib_multi_with_known_intrinsic_init(nCheckboards, nPts, nCams, X3, X2, visible_num, visible_offset, visible_idx, fc_cc_alpha_kc, zAxis_in,
			max_iter_levmar, tmp_checkboard_rT_to_cam0, cam0_to_othercam_rT, tol_E))
		{
			if (checkboard_rT_to_cam0 == 0)
				delete[] tmp_checkboard_rT_to_cam0;
			return false;
		}

		if (!_calib_multi_with_known_intrinsic_with_init(nCheckboards, nPts, nCams, X3, X2, visible_num, visible_offset, visible_idx, fc_cc_alpha_kc, zAxis_in, max_iter_levmar,
			tmp_checkboard_rT_to_cam0, cam0_to_othercam_rT, avg_err_square, sparse_solver))
		{
			if (checkboard_rT_to_cam0 == 0)
				delete[] tmp_checkboard_rT_to_cam0;
			return false;
		}

		if (checkboard_rT_to_cam0 == 0)
			delete[] tmp_checkboard_rT_to_cam0;
		return true;
	}

	/*Calibrate multi-camera.
	The intrinsic parameters are known.
	X3 is a series of checkboard coordinates.
	The exterior parameters are estimated by using "posit_coplanar_robust".
	visible_num, visible_offset,visible_idx and X2 must be valid.
	*/
	template<class T>
	bool ZQ_CameraCalibrationMulti::_calib_multi_with_known_intrinsic_init(int nCheckboards, int nPts, int nCams, const T* X3, const T* X2, const int* visible_num, const int* visible_offset, const int* visible_idx,
		const T* fc_cc_alpha_kc, bool zAxis_in, int max_iter_levmar, T* board_in_cam0_rT, T* cam0_to_othercam_rT, double& avg_err_square)
	{
		std::vector<bool> visible_map(nCams*nCheckboards);
		std::vector<int> offset_map(nCams*nCheckboards);
		for (int i = 0; i < visible_map.size(); i++)
			visible_map[i] = false;
		memset(&offset_map[0], 0, sizeof(int)*nCams*nCheckboards);
		for (int i = 0; i < nCams; i++)
		{
			int cur_visible_offset = visible_offset[i];
			int cur_visible_num = visible_num[i];
			for (int j = 0; j < cur_visible_num; j++)
			{
				int cur_visible_idx = visible_idx[cur_visible_offset + j];
				visible_map[i*nCheckboards + cur_visible_idx] = true;
				offset_map[i*nCheckboards + cur_visible_idx] = cur_visible_offset + j;
			}
		}

		std::vector<T> checkboard_to_cam_rT_map(nCheckboards*nCams * 6);
		std::vector<double> checkboard_to_cam_err_map(nCheckboards*nCams);
		memset(&checkboard_to_cam_rT_map[0], 0, sizeof(T)*nCheckboards*nCams * 6);
		memset(&checkboard_to_cam_err_map[0], 0, sizeof(double)*nCheckboards*nCams);
		for (int i = 0; i < nCams; i++)
		{
			const T* fc = fc_cc_alpha_kc + i * 10;
			const T* cc = fc_cc_alpha_kc + i * 10 + 2;
			const T alpha_c = *(fc_cc_alpha_kc + i * 10 + 4);
			const T* kc = fc_cc_alpha_kc + i * 10 + 5;
			for (int j = 0; j < nCheckboards; j++)
			{
				if (visible_map[i*nCheckboards + j])
				{
					int cur_offset = offset_map[i*nCheckboards + j];
					if (!ZQ_CameraPoseEstimation::PositCoplanarRobust(nPts, X3, X2 + cur_offset*nPts * 2, fc, cc, kc, alpha_c, 20, max_iter_levmar, 1.0,
						&checkboard_to_cam_rT_map[0] + (j*nCams + i) * 6, checkboard_to_cam_err_map[j*nCams + i], zAxis_in))
					{
						return false;
					}
				}
			}
		}

		std::vector<T> cam_R(9 * nCams);
		std::vector<T> cam_T(3 * nCams);
		std::vector<T> checkboard_R(9 * nCheckboards);
		std::vector<T> checkboard_T(3 * nCheckboards);
		std::vector<int> cam_visited(nCams);
		for (int i = 0; i < cam_visited.size(); i++)
			cam_visited[i] = 0;
		std::vector<int> checkboard_visited(nCheckboards);
		for (int i = 0; i < checkboard_visited.size(); i++)
			checkboard_visited[i] = 0;
		std::vector<double> cam_visited_err(nCams);
		std::vector<double> checkboard_visited_err(nCheckboards);

		cam_R[0] = cam_R[4] = cam_R[8] = 1;
		cam_R[1] = cam_R[2] = cam_R[3] = cam_R[5] = cam_R[6] = cam_R[7] = 0;
		cam_T[0] = cam_T[1] = cam_T[2] = 0;
		cam_visited_err[0] = 0;
		cam_visited[0] = 2;

		/*Search in Shortest Path*/


		while (true)
		{
			bool any_change = false;
			for (int i = 0; i < nCams; i++)
			{
				if (cam_visited[i])
				{
					for (int j = 0; j < nCheckboards; j++)
					{
						if (visible_map[i*nCheckboards + j])
						{
							switch (checkboard_visited[j])
							{
							case 0:
							{
								checkboard_visited[j] = 1;
								checkboard_visited_err[j] = cam_visited_err[i] + checkboard_to_cam_err_map[j*nCams + i];
								T tmp_R[9];
								T* tmp_T = &checkboard_to_cam_rT_map[0] + (j*nCams + i) * 6 + 3;
								ZQ_Rodrigues::ZQ_Rodrigues_r2R(&checkboard_to_cam_rT_map[0] + (j*nCams + i) * 6, tmp_R);

								ZQ_MathBase::MatrixMul(&cam_R[0] + i * 9, tmp_R, 3, 3, 3, &checkboard_R[0] + j * 9);
								ZQ_MathBase::MatrixMul(&cam_R[0] + i * 9, tmp_T, 3, 3, 1, &checkboard_T[0] + j * 3);
								checkboard_T[j * 3 + 0] += cam_T[i * 3 + 0];
								checkboard_T[j * 3 + 1] += cam_T[i * 3 + 1];
								checkboard_T[j * 3 + 2] += cam_T[i * 3 + 2];
							}
							break;
							case 1:
							{
								double tmp_err = cam_visited_err[i] + checkboard_to_cam_err_map[j*nCams + i];
								if (tmp_err < checkboard_visited_err[j])
								{
									checkboard_visited_err[j] = tmp_err;
									T tmp_R[9];
									T* tmp_T = &checkboard_to_cam_rT_map[0] + (j*nCams + i) * 6 + 3;
									ZQ_Rodrigues::ZQ_Rodrigues_r2R(&checkboard_to_cam_rT_map[0] + (j*nCams + i) * 6, tmp_R);

									ZQ_MathBase::MatrixMul(&cam_R[0] + i * 9, tmp_R, 3, 3, 3, &checkboard_R[0] + j * 9);
									ZQ_MathBase::MatrixMul(&cam_R[0] + i * 9, tmp_T, 3, 3, 1, &checkboard_T[0] + j * 3);
									checkboard_T[j * 3 + 0] += cam_T[i * 3 + 0];
									checkboard_T[j * 3 + 1] += cam_T[i * 3 + 1];
									checkboard_T[j * 3 + 2] += cam_T[i * 3 + 2];
								}
							}
							break;
							case 2:
								break;
							}
						}

					}
				}
			}

			//
			for (int j = 0; j < nCheckboards; j++)
			{
				if (checkboard_visited[j] == 1)
				{
					any_change = true;
					checkboard_visited[j] = 2;
				}
			}
			if (!any_change)
			{
				break;
			}

			//

			for (int j = 0; j < nCheckboards; j++)
			{
				if (checkboard_visited[j])
				{
					for (int i = 0; i < nCams; i++)
					{
						if (visible_map[i*nCheckboards + j])
						{
							switch (cam_visited[i])
							{
							case 0:
							{
								cam_visited[i] = 1;
								cam_visited_err[i] = checkboard_visited_err[j] + checkboard_to_cam_err_map[j*nCams + i];
								T tmp_R[9];
								T* tmp_T = &checkboard_to_cam_rT_map[0] + (j*nCams + i) * 6 + 3;
								ZQ_Rodrigues::ZQ_Rodrigues_r2R(&checkboard_to_cam_rT_map[0] + (j*nCams + i) * 6, tmp_R);

								//inv_R = R^T
								T inv_R[9] = {
									tmp_R[0], tmp_R[3], tmp_R[6],
									tmp_R[1], tmp_R[4], tmp_R[7],
									tmp_R[2], tmp_R[5], tmp_R[8]
								};
								//inv_T = -inv_R*T
								T inv_T[3];
								ZQ_MathBase::MatrixMul(inv_R, tmp_T, 3, 3, 1, inv_T);
								inv_T[0] = -inv_T[0];
								inv_T[1] = -inv_T[1];
								inv_T[2] = -inv_T[2];
								//
								ZQ_MathBase::MatrixMul(&checkboard_R[0] + j * 9, inv_R, 3, 3, 3, &cam_R[0] + i * 9);
								ZQ_MathBase::MatrixMul(&checkboard_R[0] + j * 9, inv_T, 3, 3, 1, &cam_T[0] + i * 3);
								cam_T[i * 3 + 0] += checkboard_T[j * 3 + 0];
								cam_T[i * 3 + 1] += checkboard_T[j * 3 + 1];
								cam_T[i * 3 + 2] += checkboard_T[j * 3 + 2];
							}
							break;
							case 1:
							{
								double tmp_err = checkboard_visited_err[j] + checkboard_to_cam_err_map[j*nCams + i];
								if (tmp_err < cam_visited_err[i])
								{
									cam_visited_err[i] = tmp_err;
									T tmp_R[9];
									T* tmp_T = &checkboard_to_cam_rT_map[0] + (j*nCams + i) * 6 + 3;
									ZQ_Rodrigues::ZQ_Rodrigues_r2R(&checkboard_to_cam_rT_map[0] + (j*nCams + i) * 6, tmp_R);

									//inv_R = R^T
									T inv_R[9] = {
										tmp_R[0], tmp_R[3], tmp_R[6],
										tmp_R[1], tmp_R[4], tmp_R[7],
										tmp_R[2], tmp_R[5], tmp_R[8]
									};
									//inv_T = -inv_R*T
									T inv_T[3];
									ZQ_MathBase::MatrixMul(inv_R, tmp_T, 3, 3, 1, inv_T);
									inv_T[0] = -inv_T[0];
									inv_T[1] = -inv_T[1];
									inv_T[2] = -inv_T[2];
									//
									ZQ_MathBase::MatrixMul(&checkboard_R[0] + j * 9, inv_R, 3, 3, 3, &cam_R[0] + i * 9);
									ZQ_MathBase::MatrixMul(&checkboard_R[0] + j * 9, inv_T, 3, 3, 1, &cam_T[0] + i * 3);
									cam_T[i * 3 + 0] += checkboard_T[j * 3 + 0];
									cam_T[i * 3 + 1] += checkboard_T[j * 3 + 1];
									cam_T[i * 3 + 2] += checkboard_T[j * 3 + 2];
								}
							}
							break;
							case 2:
								break;
							}
						}

					}
				}
			}
			//
			for (int i = 0; i < nCams; i++)
			{
				if (cam_visited[i] == 1)
				{
					any_change = true;
					cam_visited[i] = 2;
				}
			}
			if (!any_change)
			{
				break;
			}
		}
		//////

		//
		bool done_flag = true;
		for (int i = 0; i < nCams; i++)
		{
			if (cam_visited[i] == 0)
			{
				done_flag = false;
				break;
			}
		}
		for (int j = 0; j < nCheckboards; j++)
		{
			if (checkboard_visited[j] == 0)
			{
				done_flag = false;
				break;
			}
		}
		if (!done_flag)
		{
			return false;
		}
		for (int j = 0; j < nCheckboards; j++)
		{
			if (!ZQ_Rodrigues::ZQ_Rodrigues_R2r(&checkboard_R[0] + j * 9, board_in_cam0_rT + j * 6))
			{
				return false;
			}
			memcpy(board_in_cam0_rT + j * 6 + 3, &checkboard_T[0] + j * 3, sizeof(T) * 3);
		}

		for (int i = 1; i < nCams; i++)
		{
			T* tmp_R = &cam_R[0] + i * 9;
			T* tmp_T = &cam_T[0] + i * 3;
			//inv_R = R^T
			T inv_R[9] = {
				tmp_R[0], tmp_R[3], tmp_R[6],
				tmp_R[1], tmp_R[4], tmp_R[7],
				tmp_R[2], tmp_R[5], tmp_R[8]
			};
			//inv_T = -inv_R*T
			T inv_T[3];
			ZQ_MathBase::MatrixMul(inv_R, tmp_T, 3, 3, 1, inv_T);
			inv_T[0] = -inv_T[0];
			inv_T[1] = -inv_T[1];
			inv_T[2] = -inv_T[2];
			if (!ZQ_Rodrigues::ZQ_Rodrigues_R2r(inv_R, cam0_to_othercam_rT + (i - 1) * 6))
			{
				return false;
			}
			memcpy(cam0_to_othercam_rT + (i - 1) * 6 + 3, inv_T, sizeof(T) * 3);
		}
		return true;
	}

	template<class T>
	bool ZQ_CameraCalibrationMulti::_compute_err_calib(int nCheckboards, int nPts, int nCams, const T* X3, const T* X2, const int* visible_num, const int* visible_offset, const int* visible_idx,
		const T* board_in_cam0_rT, const T* cam0_to_othercam_rT, const T* fc_cc_alpha_kc, double& err_std, double& max_err, bool zAxis_in)
	{
		ZQ_DImage<T> err_x_im(nCheckboards*nPts * 2 * nCams, 1, 1);
		T*& err_x = err_x_im.data();
		ZQ_DImage<T> p_im(nCheckboards * 6 + (nCams - 1) * 6, 1, 1);
		T*& p = p_im.data();
		memcpy(p, board_in_cam0_rT, sizeof(T) * 6 * nCheckboards);
		memcpy(p + 6 * nCheckboards, cam0_to_othercam_rT, sizeof(T) * 6 * (nCams - 1));
		Calib_Multi_Data_Header<T> data;
		data.n_pts = nPts;
		data.n_cams = nCams;
		data.n_checkboards = nCheckboards;
		data.fc_cc_alpha_kc = fc_cc_alpha_kc;
		data.X3 = X3;
		data.X2 = X2;
		data.visible_num = visible_num;
		data.visible_offset = visible_offset;
		data.visible_idx = visible_idx;
		data.zAxis_in = zAxis_in;
		if (!_calib_multi_with_known_intrinsic_fun(p, err_x, nCheckboards * 6 + (nCams - 1) * 6, nCheckboards*nPts * 2 * nCams, &data))
			return false;

		max_err = 0;
		for (int i = 0; i < nCheckboards*nPts * 2 * nCams; i++)
		{
			max_err = __max(max_err, fabs(err_x[i]));
		}
		double mean_err = 0;
		for (int i = 0; i <nCheckboards*nPts * 2 * nCams; i++)
			mean_err += err_x[i];
		mean_err /= (nCheckboards*nPts * 2 * nCams);

		err_std = 0;
		for (int i = 0; i < nCheckboards*nPts * 2 * nCams; i++)
		{
			double diff = err_x[i] - mean_err;
			err_std += diff*diff;
		}
		err_std = sqrt(err_std / (nCheckboards*nPts * 2 * nCams - 1));
		return true;
	}

}
#endif
