#ifndef _ZQ_CALIBRATION_H_
#define _ZQ_CALIBRATION_H_
#include "ZQ_Rodrigues.h"
#include "ZQ_SVD.h"
#include "ZQ_LevMar.h"
#include "ZQ_Vec3D.h"
#include "ZQ_Ray3D.h"
#include <string.h>
#include <vector>

namespace ZQ
{
	class ZQ_Calibration
	{
	public:
		/* left hand coordinates*/
		template<class T>
		static void proj_no_distortion(int n, const T A[9], const T R[9], const T t[3], const T* X3, T* X2, double eps = 1e-9);

		template<class T>
		static void distortion_k2(int n, const T center[2], const T k[2], const T* inX2, T* outX2);

		/* left hand coordinates*/
		template<class T>
		static void proj_distortion_k2(int n, const T A[9], const T R[9], const T t[3], const T center[2], const T k[2], const T* X3, T* X2, double eps = 1e-9);

	private:

		template<class T>
		class StickCalib_Data_Header
		{
		public:
			const T* len_pts_to_A;
			const T* X2;
			int n_views;
			int n_pts;
			double eps;
			const T* k;
			const T* intrinsic_para;
			const T* A_pos;
			const T* theta_phi;
			double L;
			const T* h;
		};
		
		template<class T>
		class Calib_Data_Header 
		{
		public:
			const T* X3;
			const T* X2;
			int n_cams;
			int n_pts;
			double eps;
			const T* k;
			const T* intrinsic_para;
			const T* rT;
		};

		template<class T>
		class Binocalib_Data_Header
		{
		public:
			const T* X3;
			const T* left_X2;
			const T* right_X2;
			int n_views;
			int n_pts;
			double eps;
			const T* left_intrinsic_para;
			const T* right_intrinsic_para;
		};

		template<class T>
		class Multicalib_Data_Header
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
			double eps;
			const T* intrinsic_para;	//[fx1, fy1, u1, v1, fx2, fy2, u2, v2, ...]
			
		};

		template<class T>
		class Posit_Coplanar_Node
		{
		public:
			T kk[3];
			T R[9],tt[3],rr[3];
			double Z0;
			double Error;
		};

	private:
		/*
		refer to the paper:
		Camera Calibration with One-Dimensional Objects[J]. Zhang Z. PAMI 2004.
		compute the camera intrinsic paramter with at least 6 views (one end of the stick must be fixed)
		*/
		template<class T>
		static bool _stickCalib_compute_B_A_ratio(const T* a, const T* b, const T*c, double len_ac, double len_bc, double& ratio);

		template<class T>
		static bool _stickCalib_compute_h(const T* a, const T* b, const T* c, double len_ac, double len_bc, T* h);

	public:
		/*
		refer to the paper:
		Camera Calibration with One-Dimensional Objects[J]. Zhang Z. PAMI 2004.
		compute the camera intrinsic paramter with at least 6 views (one end of the stick must be fixed)
		*/
		template<class T>
		static bool stickCalib_closed_form_solution(int n_views, double len_ac, double len_bc, const T* X2, T* intrinsic_para);

	private:
		template<class T>
		static void _theta_phi_to_dir_func(const T* theta_phi, T* dir);

		template<class T>
		static void _theta_phi_to_dir_jac(const T* theta_phi, T* ddir_dthetaphi);

		/*calibrate camera: intrinsic parameter,
		based on Lev-Mar optimization, need good initialization.
		left hand coordinates.
		*/
		template<class T>
		static bool _stickCalib_estimate_no_distortion_func(const T* p, T* hx, int m, int n, const void* data);

		template<class T>
		static bool _stickCalib_estimate_no_distortion_jac(const T* p, T* jx, int m, int n, const void* data);
	
		template<class T>
		static bool _stickCalib_estimate_zA_A_func(const T* p, T* hx, int m, int n, const void* data);

		template<class T>
		static bool _stickCalib_estimate_zA_A_jac(const T* p, T* jx, int m, int n, const void* data);

	public:
		/*calibrate camera: intrinsic parameter,
		based on Lev-Mar optimization, need good initialization.
		left hand coordinates.
		*/

		template<class T>
		static bool stickCalib_estimate_no_distortion_init(int width, int height, int n_views, int n_pts, const T* len_pts_to_A, const T* X2, int max_iter, T* intrinsc_para, T* A_pos, T* theta_phi);

		template<class T>
		static bool stickCalib_estimate_no_distortion_with_init(int n_views, int n_pts, const T* len_pts_to_A, const T* X2, int max_iter, T* intrinsic_para, T* A_pos, T* theta_phi, double& avg_err_square, double eps = 1e-9);

		template<class T>
		static bool stickCalib_estimate_no_distortion_without_init(int width, int height, int n_views, int n_pts, const T* len_pts_to_A, const T* X2, int max_iter, T* intrinsic_para, T* A_pos, T* theta_phi, double& avg_err_square, double eps = 1e-9);

	private:
		/*
		refer to the paper:
		A flexible new technique for camera calibration[J]. Zhang Z. PAMI 2000.
		compute the camera intrinsic parameter with at least  3 checkboard images. 
		left hand coordinates.
		*/

		template<class T>
		static bool _estimate_H_func(const T* x, T* fx, int m, int n, const void* data);

		template<class T>
		static bool _estimate_H_jac(const T* p, T* jx, int m, int n, const void* data);

	public:
		template<class T>
		static bool _estimate_H(T* H, int n_pts, const T* X3, const T* X2, int max_iter, double eps = 1e-9, bool has_init = false);
	
	private:
		template<class T>
		static void _get_v_i_j(const T* H, int i, int j, T* vij);

		template<class T>
		static void _get_v_no_distortion(const T* H, T* row1, T* row2);

		template<class T>
		static bool _compute_b(const T* VV, int m, int n, T* b);

		template<class T>
		static bool _compute_A_from_b(const T* b, T* A);

		template<class T>
		static bool _compute_RT_from_AH(const T* H, const T* A, T* R, T* tt);

	public:
		/*
		refer to the paper:
		A flexible new technique for camera calibration[J]. Zhang Z. PAMI 2000.
		compute the camera intrinsic parameter with at least  3 checkboard images. 
		left hand coordinates.
		*/
		template<class T>
		static bool calib_estimate_k_int_rT_init(int n_cams, int n_pts, const T* X3, const T* X2, int max_iter, T* intrinsic_para, T* rT, double eps  = 1e-9);

	private:
		/*calibrate camera: intrinsic parameter,
		based on Lev-Mar optimization, need good initialization.
		left hand coordinates.
		*/
		template<class T>
		static bool _calib_estimate_no_distortion_func(const T* p, T* hx, int m, int n, const void* data);

		template<class T>
		static bool _calib_estimate_no_distortion_jac(const T* p, T* jx, int m, int n, const void* data);

	public:
		/*calibrate camera: intrinsic parameter,
		based on Lev-Mar optimization, need good initialization.
		left hand coordinates.
		*/
		
		template<class T>
		static bool calib_estimate_no_distortion_with_init(int n_cams, int n_pts, const T* X3, const T* X2, int max_iter,T* intrinsic_para, T* rT, double& avg_err_square, double eps = 1e-9);

	private:
		/*calibrate camera: intrinsic parameter and distortion (k1,k2),
		based on Lev-Mar optimization, need good initialization.
		left hand coordinates.
		*/
		template<class T>
		static bool _calib_estimate_k_int_rT_func(const T* p, T* hx, int m, int n, const void* data);
		
		template<class T>
		static bool _calib_estimate_k_int_rT_jac(const T* p, T* jx, int m, int n, const void* data);

	private:
		/*calibrate camera: intrinsic parameter
		fix distortion (k1,k2),
		based on Lev-Mar optimization, need good initialization.
		left hand coordinates.
		*/
		template<class T>
		static bool _calib_estimate_int_rT_fix_k_func(const T* p, T* hx, int m, int n, const void* data);

		template<class T>
		static bool _calib_estimate_int_rT_fix_k_jac(const T* p, T* jx, int m, int n, const void* data);
	
		template<class T>
		static bool _estimate_k_with_int_rT(int n_cams, int n_pts, const T* X3, const T* X2, T* k, const T* intrinsic_para, const T* rT, double eps);

	public:
		/*calibrate camera: intrinsic parameter and distortion (k1,k2),
		based on Lev-Mar optimization, need good initialization.
		left hand coordinates.
		*/

		template<class T>
		static bool calib_estimate_k_int_rT_with_init(int n_cams, int n_pts,const T* X3, const T* X2, int max_iter, T* k,  T* intrinsic_para, T* rT, double& avg_err_square, double eps = 1e-9);

		template<class T>
		static bool calib_estimate_int_rT_fix_k_with_init(int n_cams, int n_pts, const T* X3, const T* X2, int max_iter, const T* k, T* intrinsic_para, T* rT, double& avg_err_square, double eps = 1e-9);

		/* I have found this function cannot work well unless distortion is almost zero.*/
		template<class T>
		static bool calib_estimate_k_int_rT_without_init(int n_cams, int n_pts, const T* X3, const T* X2, int max_iter, T* k, T* intrinsic_para, T* rT, double& avg_err_square, double eps = 1e-9);

		/* this function can work well with max_iter > 10 from my experience,
		but alt_iter should be 100~10000, when the distortion is small, 100 or less iteration can get convegence*/
		template<class T>
		static bool calib_estimate_k_int_rT_alt_with_init(int n_cams, int n_pts, const T* X3, const T* X2, int alt_iter, int max_iter, T* k, T* intrinsic_para, T* rT, double& avg_err_square, double eps = 1e-9, bool display = true);

		/* this function can work well with max_iter > 10 from my experience,
		but alt_iter should be 100~10000, when the distortion is small, 100 or less iteration can get convegence*/
		template<class T>
		static bool calib_estimate_k_int_rT_alt_without_init(int n_cams, int n_pts, const T* X3, const T* X2, int alt_iter, int max_iter, T* k, T* intrinsic_para, T* rT, double& avg_err_square, double eps = 1e-9, bool display = true);

	private:
		/*pose estimation based on Lev-Mar optimization, need good initialization.
		left hand coordinates.
		*/
		template<class T>
		static bool _pose_estimate_no_distortion_func(const T* p, T* hx, int m, int n, const void* data);
		
		template<class T>
		static bool _pose_estimate_no_distortion_jac(const T* p, T* jx, int m, int n, const void* data);
		
	public:
		/*pose estimation based on Lev-Mar optimization, need good initialization.
		left hand coordinates.
		*/

		template<class T>
		static bool pose_estimate_no_distortion_with_init(int n_pts, const T* X3, const T* X2, int max_iter, const T* intrinsic_para, T* rT, double& avg_err_square, double eps = 1e-9);

	public:
		/*
		refer to the paper: 
		Model-Based Object Pose in 25 Lines of Codes. Daniel F. DeMenthon and Larry S. Davis. IJCV,1995.
		left hand coordinates.
		intrinsic_para[0-3]: fx, fy, u0, v0.  with no distortion.
		rT[0-5]: rx, ry, rz, Tx, Ty, Tz.  (rx,ry,rz,rw) is a quaternion.   
		*/
		template<class T>
		static bool posit_no_coplanar(int n_pts, const T* X3, const T* X2, int max_iter, const T* intrinsic_para, T* rT);

		/*
		refer to the paper:
		iterative pose estimation using coplanar points. Denis Oberkampf, Daniel F. DeMenthon, Larry  S. Davis. CVPR, 1993. 
		left hand coordinates.
		intrinsic_para[0-3]: fx, fy, u0, v0. with no distortion.
		rT[0-5]: rx, ry, rz, Tx, Ty, Tz.  (rx,ry,rz,rw) is a quaternion.
		*/
		template<class T>
		static bool posit_coplanar(int n_pts, const T* X3, const T* X2, int max_iter, double tol_avg_E, const T* intrinsic_para, T* rT, T* reproj_err_square, double eps = 1e-9);

		/*
		left hand coordinates.
		The base idea is to use the method proposed in the paper:
		iterative pose estimation using coplanar points. Denis Oberkampf, Daniel F. DeMenthon, Larry  S. Davis. CVPR, 1993. 
		However, I find it cannot make sure the method always converge to the optimal solution.
		But the translation seems to be near the optimal one according to my observations.
		So I choose 9 rotations to run Lev-Mar solvers to find a best solution.
		If all choices do not give a solution with avg_E < tol_avg_E, the best solution of the 9 will be returned,
		otherwise, the first one satisfying avg_E < tol_avg_E will be returned.
		*/
		template<class T>
		static bool posit_coplanar_robust(int n_pts, const T* X3, const T* X2, int max_iter_posit, int max_iter_levmar, double tol_E, const T* intrinsic_para, T* rT, double& avg_E, double eps = 1e-9);

	private:
		/*Calibrate binocular camera. 
		The intrinsic parameters are known.
		X3 is a series of checkboard coordinates.
		The exterior parameters are estimated by using "posit_coplanar_robust".
		*/
		template<class T>
		static bool _binocalib_with_known_intrinsic_init(int n_views, int n_pts, const T* X3, const T* left_X2, const T* right_X2, const T* left_intrinsic_para, const T* right_intrinsic_para,
			int max_iter_posit, int max_iter_levmar, double tol_E, T* right_rT, T* right_to_left_rT, double& avg_err_square, double eps = 1e-9);

		template<class T>
		static bool _binocalib_with_known_intrinsic_func(const T* p, T* hx, int m, int n, const void* data);

		template<class T>
		static bool _binocalib_with_known_intrinsic_jac(const T* p, T* jx, int m, int n, const void* data);

	public:

		/*Calibrate binocular camera.
		The intrinsic parameters are known.
		X3 is a series of checkboard coordinates.
		The exterior parameters are estimated by using "posit_coplanar_robust".
		*/
		template<class T>
		static bool binocalib_with_known_intrinsic_with_init(int n_views, int n_pts, const T* X3, const T* left_X2, const T* right_X2, const T* left_intrinsic_para, const T* right_intrinsic_para,
			int max_iter_levmar, T* right_rT, T* right_to_left_rT, double& avg_err_square, double eps  = 1e-9 );

		/*Calibrate binocular camera.
		The intrinsic parameters are known.
		X3 is a series of checkboard coordinates.
		The exterior parameters are estimated by using "posit_coplanar_robust".
		*/
		template<class T>
		static bool binocalib_with_known_intrinsic(int n_views, int n_pts, const T* X3, const T* left_X2, const T* right_X2, const T* left_intrinsic_para, const T* right_intrinsic_para,
			int max_iter_posit, int max_iter_levmar, double tol_E, T* right_rT, T* right_to_left_rT, double& avg_err_square, double eps = 1e-9);
	
private:
		/*Calibrate multi-camera.
		The intrinsic parameters are known.
		X3 is a series of checkboard coordinates.
		The exterior parameters are estimated by using "posit_coplanar_robust".
		visible_num, visible_offset,visible_idx and X2 must be valid.
		*/
		template<class T>
		static bool _multicalib_with_known_intrinsic_init(int n_checkboards, int n_cams, int n_pts, const T* X3, const T* X2, const int* visible_num, const int* visible_offset, const int* visible_idx,
			const T* intrinsic_para, int max_iter_posit, int max_iter_levmar, double tol_E, T* tmp_checkboard_rT_to_cam0, T* cam0_to_othercam_rT, double eps = 1e-9);

		template<class T>
		static bool _multicalib_with_known_intrinsic_func(const T* p, T* hx, int m, int n, const void* data);

		template<class T>
		static bool _multicalib_with_known_intrinsic_jac(const T* p, T* jx, int m, int n, const void* data);
				
	public:
		/*Calibrate multi-camera.
		The intrinsic parameters are known.
		X3 is a series of checkboard coordinates.
		The exterior parameters are estimated by using "posit_coplanar_robust".
		visible_num, visible_offset,visible_idx and X2 must be valid.
		*/
		template<class T>
		static bool multicalib_with_known_intrinsic_with_init(int n_checkboards, int n_cams, int n_pts, const T* X3, const T* X2, const int* visible_num, const int* visible_offset, const int* visible_idx,
			const T* intrinsic_para, int max_iter_levmar, T* checkboard_rT_to_cam0, T* cam0_to_othercam_rT, double& avg_err_square, double eps = 1e-9);

		/*Calibrate multi-camera.
		The intrinsic parameters are known.
		X3 is a series of checkboard coordinates.
		The exterior parameters are estimated by using "posit_coplanar_robust".
		visible_num, visible_offset,visible_idx and X2 must be valid.
		*/
		template<class T>
		static bool multicalib_with_known_intrinsic(int n_checkboards, int n_cams, int n_pts, const T* X3, const T* X2, const int* visible_num, const int* visible_offset, const int* visible_idx,
			const T* intrinsic_para, int max_iter_posit, int max_iter_levmar, double tol_E, T* checkboard_rT_to_cam0, T* cam0_to_othercam_rT, double& avg_err_square, double eps = 1e-9);
	};

	/*************************************************************************************************************************************************/
	/*************************************************************************************************************************************************/

	/*left hand coordinates.*/
	template<class T>
	void ZQ_Calibration::proj_no_distortion(int n, const T A[], const T R[], const T t[], const T* X3, T* X2, double eps /* = 1e-6 */)
	{
		for(int i = 0;i < n;i++)
		{
			double tmp_x3[3] = 
			{
				R[0]*X3[i*3+0] + R[1]*X3[i*3+1] + R[2]*X3[i*3+2] + t[0],
				R[3]*X3[i*3+0] + R[4]*X3[i*3+1] + R[5]*X3[i*3+2] + t[1],
				R[6]*X3[i*3+0] + R[7]*X3[i*3+1] + R[8]*X3[i*3+2] + t[2]
			};

			double tmp_x2[3] = 
			{
				A[0]*tmp_x3[0] + A[1]*tmp_x3[1] + A[2]*tmp_x3[2],
				A[3]*tmp_x3[0] + A[4]*tmp_x3[1] + A[5]*tmp_x3[2],
				A[6]*tmp_x3[0] + A[7]*tmp_x3[1] + A[8]*tmp_x3[2]
			};

			
			X2[i*2+0] = tmp_x2[0]*tmp_x2[2]/(tmp_x2[2]*tmp_x2[2]+eps*eps);
			X2[i*2+1] = tmp_x2[1]*tmp_x2[2]/(tmp_x2[2]*tmp_x2[2]+eps*eps);
		}
	}

	template<class T>
	void ZQ_Calibration::distortion_k2(int n, const T center[2], const T k[2], const T* inX2, T* outX2)
	{
		double u0 = center[0];
		double v0 = center[1];

		for(int i = 0;i < n;i++)
		{
			double X = inX2[i*2+0]-u0;
			double Y = inX2[i*2+1]-v0;
			double R2 = X*X+Y*Y;
			outX2[i*2+0] = inX2[i*2+0] + X*(k[0]*R2+k[1]*R2*R2);
			outX2[i*2+1] = inX2[i*2+1] + Y*(k[0]*R2+k[1]*R2*R2);
		}
	}


	/*left hand coordinates.*/
	template<class T>
	void ZQ_Calibration::proj_distortion_k2(int n, const T A[9], const T R[9], const T t[3], const T center[2], const T k[2], const T* X3, T* X2, double eps /* = 1e-6 */)
	{
		proj_no_distortion(n,A,R,t,X3,X2,eps);
		distortion_k2(n,center,k,X2,X2);
	}

	/*
	refer to the paper:
	Camera Calibration with One-Dimensional Objects[J]. Zhang Z. PAMI 2004.
	compute the camera intrinsic paramter with at least 6 views (one end of the stick must be fixed)
	*/

	template<class T>
	bool ZQ_Calibration::_stickCalib_compute_B_A_ratio(const T* a, const T* b, const T*c, double len_ac, double len_bc, double& ratio)
	{
		ZQ_Vec3D a_(a[0], a[1], 1);
		ZQ_Vec3D b_(b[0], b[1], 1);
		ZQ_Vec3D c_(c[0], c[1], 1);
		ZQ_Vec3D axc = a_.CrossProduct(c_);
		ZQ_Vec3D bxc = b_.CrossProduct(c_);

		double lambda_A = (len_bc) / (len_ac + len_bc);
		double lambda_B = 1 - lambda_A;

		double bottom = lambda_B * bxc.DotProduct(bxc);
		double top = lambda_A * axc.DotProduct(bxc);
		if (bottom == 0)
			return false;
		ratio = -top / bottom;
		return true;
	}

	template<class T>
	bool ZQ_Calibration::_stickCalib_compute_h(const T* a, const T* b, const T* c, double len_ac, double len_bc, T* h)
	{
		double ratio;
		if (!_stickCalib_compute_B_A_ratio(a, b, c, len_ac, len_bc, ratio))
			return false;

		h[0] = a[0] - ratio*b[0];
		h[1] = a[1] - ratio*b[1];
		h[2] = 1 - ratio;
		return true;
	}

	/*
	refer to the paper:
	Camera Calibration with One-Dimensional Objects[J]. Zhang Z. PAMI 2004.
	compute the camera intrinsic paramter with at least 6 views (one end of the stick must be fixed)
	*/
	template<class T>
	bool ZQ_Calibration::stickCalib_closed_form_solution(int n_views, double len_ac, double len_bc, const T* X2, T* intrinsic_para)
	{
		int valid_num = 0;
		T* VV = new T[n_views * 6];
		T h[3];
		for (int i = 0; i < n_views; i++)
		{
			if (!_stickCalib_compute_h(X2 + 6 * i, X2 + 4 + 6 * i, X2 + 2 + 6 * i, len_ac, len_bc, h))
			{
				printf("warning: view %d is not valid, we have removed it and continue\n", i);
			}
			else
			{
				VV[valid_num * 6 + 0] = h[0] * h[0];
				VV[valid_num * 6 + 1] = 2 * h[0] * h[1];
				VV[valid_num * 6 + 2] = h[1] * h[1];
				VV[valid_num * 6 + 3] = 2 * h[0] * h[2];
				VV[valid_num * 6 + 4] = 2 * h[1] * h[2];
				VV[valid_num * 6 + 5] = 2 * h[2] * h[2];
				valid_num++;
			}
		}
		if (valid_num < 6)
		{
			printf("only %d views is valid, 6 is needed at least\n", valid_num);
			delete[]VV;
			return false;
		}

		double L2 = (len_ac + len_bc)*(len_ac + len_bc);
		ZQ_Matrix<double> Amat(valid_num, 6), bmat(valid_num,1), xmat(6,1);
		for (int i = 0; i < valid_num; i++)
		{
			for (int j = 0; j < 6; j++)
				Amat.SetData(i, j, VV[i * 6 + j]);
			bmat.SetData(i, 0, L2);
		}

		FILE* out = fopen("Vmat.txt","w");
		for (int i = 0; i < valid_num; i++)
		{
			for (int j = 0; j < 6; j++)
				fprintf(out, "%e ", VV[i * 6 + j]);
			fprintf(out,"\n");
		}
		fclose(out);

		ZQ_SVD::Solve(Amat, xmat, bmat);
		delete[]VV;

		bool flag;
		double x1 = xmat.GetData(0, 0, flag);
		double x2 = xmat.GetData(1, 0, flag);
		double x3 = xmat.GetData(2, 0, flag);
		double x4 = xmat.GetData(3, 0, flag);
		double x5 = xmat.GetData(4, 0, flag);
		double x6 = xmat.GetData(5, 0, flag);

		if (0)
		{
			double x1x3_x2x2 = x1*x3 - x2*x2;
			if (x1x3_x2x2 <= 0 || x1 <= 0)
			{
				return false;
			}
			double v0 = (x2*x4 - x1*x5) / x1x3_x2x2;
			double tmp = x6 - (x4*x4 + v0*(x2*x4 - x1*x5)) / x1;
			if (tmp <= 0)
				return false;
			double zA = sqrt(tmp);
			double alpha = sqrt(zA / x1);
			double beta = sqrt(zA*x1 / x1x3_x2x2);
			double gamma = -x2*alpha*alpha*beta / zA;
			double u0 = gamma*v0 / alpha - x4*alpha*alpha / zA;
			intrinsic_para[0] = alpha;
			intrinsic_para[1] = beta;
			intrinsic_para[2] = u0;
			intrinsic_para[3] = v0; 
		}
		else //fix gamma = 0
		{
			if (x1 <= 0 || x3 <= 0)
				return false;
			double u0 = -x4 / x1;
			double v0 = -x5 / x3;
			double tmp = x6 + x4*u0 + x5*v0;
			if (tmp <= 0)
				return false;

			double zA = sqrt(tmp);
			double alpha = sqrt(tmp / x1);
			double beta = sqrt(tmp / x3);
			intrinsic_para[0] = alpha;
			intrinsic_para[1] = beta;
			intrinsic_para[2] = u0;
			intrinsic_para[3] = v0;
		}

		
		return true;
	}

	template<class T>
	void ZQ_Calibration::_theta_phi_to_dir_func(const T* theta_phi, T* dir)
	{
		dir[0] = sin(theta_phi[0])*cos(theta_phi[1]);
		dir[1] = sin(theta_phi[0])*sin(theta_phi[1]);
		dir[2] = cos(theta_phi[0]);
	}

	template<class T>
	void ZQ_Calibration::_theta_phi_to_dir_jac(const T* theta_phi, T* ddir_dthetaphi)
	{
		ddir_dthetaphi[0] = cos(theta_phi[0])*cos(theta_phi[1]);
		ddir_dthetaphi[1] = -sin(theta_phi[0])*sin(theta_phi[1]);
		ddir_dthetaphi[2] = cos(theta_phi[0])*sin(theta_phi[1]);
		ddir_dthetaphi[3] = sin(theta_phi[0])*cos(theta_phi[1]);
		ddir_dthetaphi[4] = -sin(theta_phi[0]);
		ddir_dthetaphi[5] = 0;
	}


	/*stick calibrate camera: intrinsic parameter,
	based on Lev-Mar optimization, need good initialization.
	left hand coordinates.
	the fixed point of each view must be arranged at the first position
	*/
	template<class T>
	bool ZQ_Calibration::_stickCalib_estimate_no_distortion_func(const T* p, T* hx, int m, int n, const void* data)
	{
		const StickCalib_Data_Header<T>* ptr = (const StickCalib_Data_Header<T>*)data;
		const T* len_pts_to_A = ptr->len_pts_to_A;
		const T* X2 = ptr->X2;
		const int n_pts = ptr->n_pts;
		const int n_views = ptr->n_views;
		const double eps = ptr->eps;
		T intrinsic_A[9] = 
		{
			p[0], 0, p[2],
			0, p[1], p[3],
			0,0,1
		};
		const T* A_pos = p + 4;
		const T* theta_phi = p + 4 + 3;

		for (int cc = 0; cc < n_views; cc++)
		{
			T dir[3];
			_theta_phi_to_dir_func(theta_phi + 2 * cc, dir);
			for (int i = 0; i < n_pts; i++)
			{
				T X3[3] =
				{
					A_pos[0] + dir[0] * len_pts_to_A[i],
					A_pos[1] + dir[1] * len_pts_to_A[i],
					A_pos[2] + dir[2] * len_pts_to_A[i]
				};
				T tmp_x3[3];
				ZQ_MathBase::MatrixMul(intrinsic_A, X3, 3, 3, 1, tmp_x3);
				
				T tmp_x2[2] =
				{
					tmp_x3[0] * tmp_x3[2] / (tmp_x3[2] * tmp_x3[2] + eps*eps),
					tmp_x3[1] * tmp_x3[2] / (tmp_x3[2] * tmp_x3[2] + eps*eps)
				};


				hx[(cc*n_pts + i) * 2 + 0] = tmp_x2[0] - X2[(cc*n_pts + i) * 2 + 0];
				hx[(cc*n_pts + i) * 2 + 1] = tmp_x2[1] - X2[(cc*n_pts + i) * 2 + 1];
			}
		}
		
		return true;
	}

	/*stick calibrate camera: intrinsic parameter,
	based on Lev-Mar optimization, need good initialization.
	left hand coordinates.
	the fixed point of each view must be arranged at the first position
	*/
	template<class T>
	bool ZQ_Calibration::_stickCalib_estimate_no_distortion_jac(const T* p, T* jx, int m, int n, const void* data)
	{
		const StickCalib_Data_Header<T>* ptr = (const StickCalib_Data_Header<T>*)data;
		const T* len_pts_to_A = ptr->len_pts_to_A;
		const T* X2 = ptr->X2;
		const int N = ptr->n_pts;
		const int n_views = ptr->n_views;
		const double eps = ptr->eps;

		memset(jx, 0, sizeof(T)*m*n);

		//
		double alpha = p[0];
		double beta = p[1];
		double u0 = p[2];
		double v0 = p[3];

		T A[9] = {
			alpha, 0, u0,
			0, beta, v0,
			0, 0, 1
		};

		T dAdint[36] = {
			1, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 1,
			0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 0
		};

		const T* A_pos = p + 4;
		const T* theta_phi = p + 4 + 3;

		for (int cc = 0; cc < n_views; cc++)
		{
			T dir[3], ddir_dthetaphi[6];
			_theta_phi_to_dir_func(theta_phi + cc * 2, dir);
			_theta_phi_to_dir_jac(theta_phi + cc * 2, ddir_dthetaphi);
			
			int cur_row_shift = N * 2 * cc;
			for (int i = 0; i < N; i++)
			{
				int cur_row0 = cur_row_shift + i * 2 + 0;
				int cur_row1 = cur_row_shift + i * 2 + 1;
				
				//var1 = A_pos + len_pts_to_A * dir;
				T var1[3] = { A_pos[0] + len_pts_to_A[i] * dir[0], A_pos[1] + len_pts_to_A[i] * dir[1], A_pos[2] + len_pts_to_A[i] * dir[2] };

				T dvar1dApos[9] = {
					1, 0, 0,
					0, 1, 0,
					0, 0, 1
				};
				T dvar1ddir[9] =
				{
					len_pts_to_A[i], 0, 0,
					0, len_pts_to_A[i], 0,
					0, 0, len_pts_to_A[i]
				};

				//
				T dvar1dthetaphi[6] = { 0 };
				ZQ_MathBase::MatrixMul(dvar1ddir, ddir_dthetaphi, 3, 3, 2, dvar1dthetaphi);


				//var2 = A*var1: dvar2dA, dvar2dvar1
				T var2[3] = {
					A[0] * var1[0] + A[1] * var1[1] + A[2] * var1[2],
					A[3] * var1[0] + A[4] * var1[1] + A[5] * var1[2],
					A[6] * var1[0] + A[7] * var1[1] + A[8] * var1[2]
				};

				T dvar2dA[27] = {
					var1[0], var1[1], var1[2], 0, 0, 0, 0, 0, 0,
					0, 0, 0, var1[0], var1[1], var1[2], 0, 0, 0,
					0, 0, 0, 0, 0, 0, var1[0], var1[1], var1[2]
				};
				T dvar2dvar1[9] = {
					A[0], A[1], A[2],
					A[3], A[4], A[5],
					A[6], A[7], A[8]
				};
				T dvar2dApos[9] = { 0 };
				ZQ_MathBase::MatrixMul(dvar2dvar1, dvar1dApos, 3, 3, 3, dvar2dApos);
				T dvar2dthetaphi[6] = { 0 };
				ZQ_MathBase::MatrixMul(dvar2dvar1, dvar1dthetaphi, 3, 3, 2, dvar2dthetaphi);
				T dvar2dint[12] = { 0 };
				ZQ_MathBase::MatrixMul(dvar2dA, dAdint, 3, 9, 4, dvar2dint);



				//X2 = [var2(1)/var2(3);var2(2)/var2(3)]
				T cur_X2[2] = {
					var2[0] * var2[2] / (var2[2] * var2[2] + eps*eps),
					var2[1] * var2[2] / (var2[2] * var2[2] + eps*eps)
				};

				T dX2dvar2[6] =
				{
					var2[2] / (var2[2] * var2[2] + eps*eps), 0, -var2[0] / (var2[2] * var2[2] + eps*eps),
					0, var2[2] / (var2[2] * var2[2] + eps*eps), -var2[1] / (var2[2] * var2[2] + eps*eps)
				};

				T dX2dint[8] = { 0 };
				T dX2dApos[6] = { 0 };
				T dX2dthetaphi[4] = { 0 };
				ZQ_MathBase::MatrixMul(dX2dvar2, dvar2dint, 2, 3, 4, dX2dint);
				ZQ_MathBase::MatrixMul(dX2dvar2, dvar2dApos, 2, 3, 3, dX2dApos);
				ZQ_MathBase::MatrixMul(dX2dvar2, dvar2dthetaphi, 2, 3, 2, dX2dthetaphi);

				for (int iii = 0; iii < 4; iii++)
				{
					jx[cur_row0*m + iii] = dX2dint[iii];
					jx[cur_row1*m + iii] = dX2dint[4 + iii];
				}
				for (int iii = 0; iii < 3; iii++)
				{
					jx[cur_row0*m + 4 + iii] = dX2dApos[iii];
					jx[cur_row1*m + 4 + iii] = dX2dApos[3 + iii];
				}
				for (int iii = 0; iii < 2; iii++)
				{
					jx[cur_row0*m + 4 + 3 + cc * 2 + iii] = dX2dthetaphi[iii];
					jx[cur_row1*m + 4 + 3 + cc * 2 + iii] = dX2dthetaphi[2 + iii];
				}
			}
		}
		return true;
	}

	template<class T>
	bool ZQ_Calibration::_stickCalib_estimate_zA_A_func(const T* p, T* hx, int m, int n, const void* data)
	{
		const StickCalib_Data_Header<T>* ptr = (const StickCalib_Data_Header<T>*)data;
		const T* all_h = ptr->h;
		const int n_views = ptr->n_views;
		const double L = ptr->L;
		
		//B = A^-T^A-1
		const T zA = p[0];
		const T alpha = p[1];
		const T beta = p[2];
		const T u0 = p[3];
		const T v0 = p[4];

		T alpha2 = alpha*alpha;
		T beta2 = beta*beta;
		T u02 = u0*u0;
		T v02 = v0*v0;
		T B[9] =
		{
			1.0 / alpha2, 0, -u0 / alpha2,
			0, 1.0 / beta2, -v0 / beta2,
			-u0 / alpha2, -v0 / beta2, u02 / alpha2 + v02 / beta2 + 1.0
		};

		for (int cc = 0; cc < n; cc++)
		{
			const T* h = all_h + 3 * cc;
			T Bh[3], hTBh[1];
			ZQ_MathBase::MatrixMul(B, h, 3, 3, 1, Bh);
			ZQ_MathBase::MatrixMul(h, Bh, 1, 3, 1, hTBh);
			hx[cc] = zA*zA*hTBh[0] - L*L;
		}
		return true;
	}

	template<class T>
	bool ZQ_Calibration::_stickCalib_estimate_zA_A_jac(const T* p, T* jx, int m, int n, const void* data)
	{
		const StickCalib_Data_Header<T>* ptr = (const StickCalib_Data_Header<T>*)data;
		const T* all_h = ptr->h;
		const int n_views = ptr->n_views;

		memset(jx, 0, sizeof(T)*m*n);
		//B = A^-T^A-1
		const T zA = p[0];
		const T alpha = p[1];
		const T beta = p[2];
		const T u0 = p[3];
		const T v0 = p[4];

		T alpha2 = alpha*alpha;
		T beta2 = beta*beta;
		T u02 = u0*u0;
		T v02 = v0*v0;
		T B[9] = 
		{
			1.0/alpha2, 0, -u0/alpha2,
			0, 1.0/beta2, -v0/beta2,
			-u0/alpha2, -v0/beta2, u02/alpha2+v02/beta2+1.0
		};

		T dBdint[36] =
		{
			-2.0 / (alpha2*alpha), 0, 0, 0,
			0, 0, 0, 0,
			2.0*u0 / (alpha2*alpha), 0, -1.0 / alpha2, 0,
			0, 0, 0, 0,
			0, -2.0 / (beta2*beta), 0, 0,
			0, 2.0*v0 / (beta2*beta), 0, -1.0 / beta2,
			2.0*u0 / (alpha2*alpha), 0, -1.0 / alpha2, 0,
			0, 2.0*v0 / (beta2*beta), 0, -1.0 / beta2,
			-2.0*u02 / (alpha2*alpha), -2.0*v02 / (beta2*beta), 2 * u0 / alpha2, 2 * v0 / beta2
		};

		for (int cc = 0; cc < n; cc++)
		{
			const T* h = all_h + 3 * cc;
			T Bh[3], hTBh[1];
			ZQ_MathBase::MatrixMul(B, h, 3, 3, 1, Bh);
			ZQ_MathBase::MatrixMul(h, Bh, 1, 3, 1, hTBh);
			T dhTBhdB[9] = 
			{
				h[0] * h[0], h[0] * h[1], h[0] * h[2], h[0] * h[1], h[1] * h[1], h[1] * h[2], h[0] * h[2], h[1] * h[2], h[2] * h[2]
			};
			T dhTBhdint[4];
			ZQ_MathBase::MatrixMul(dhTBhdB, dBdint, 1, 9, 4, dhTBhdint);
			jx[cc*m + 0] = 2 * zA*hTBh[0];
			jx[cc*m + 1] = zA*zA*dhTBhdint[0];
			jx[cc*m + 2] = zA*zA*dhTBhdint[1];
			jx[cc*m + 3] = zA*zA*dhTBhdint[2];
			jx[cc*m + 4] = zA*zA*dhTBhdint[3];
		}
		return true;
	}

	/*stick calibrate camera: intrinsic parameter,
	based on Lev-Mar optimization, need good initialization.
	left hand coordinates.
	the fixed point of each view must be arranged at the first position
	*/
	template<class T>
	bool ZQ_Calibration::stickCalib_estimate_no_distortion_init(int width, int height, int n_views, int n_pts, const T* len_pts_to_A, const T* X2, int max_iter, T* intrinsic_para, T* A_pos, T* theta_phi)
	{
		T p[5];
		T cx = width*0.5;
		T cy = height*0.5;
		T focal_x = cx;
		T focal_y = cy;

		int C_id = 1;
		int B_id = 2;
		T len_ac = len_pts_to_A[1];
		T len_bc = len_pts_to_A[2] - len_pts_to_A[1];
		if (len_pts_to_A[1] > len_pts_to_A[2])
		{
			C_id = 2;
			B_id = 1;
			len_ac = len_pts_to_A[2];
			len_bc = len_pts_to_A[1] - len_pts_to_A[2];
		}

		int valid_num = 0;
		T* all_h = new T[n_views * 3];
		T* VV = new T[n_views * 6];
		for (int i = 0; i < n_views; i++)
		{
			T* h = all_h + 3 * i;
			if (!_stickCalib_compute_h(X2 + n_pts * 2 * i, X2 + n_pts * 2 * i + B_id * 2, X2 + n_pts * 2 * i + C_id * 2, len_ac, len_bc, h))
			{
				printf("warning: view %d is not valid, we have removed it and continue\n", i);
			}
			else
			{
				VV[valid_num * 6 + 0] = h[0] * h[0];
				VV[valid_num * 6 + 1] = 2 * h[0] * h[1];
				VV[valid_num * 6 + 2] = h[1] * h[1];
				VV[valid_num * 6 + 3] = 2 * h[0] * h[2];
				VV[valid_num * 6 + 4] = 2 * h[1] * h[2];
				VV[valid_num * 6 + 5] = 2 * h[2] * h[2];
				valid_num++;
			}
		}

		if (valid_num < 6)
		{
			printf("only %d views is valid, 6 is needed at least\n", valid_num);
			delete[]VV;
			delete[]all_h;
			return false;
		}

		double L2 = (len_ac + len_bc)*(len_ac + len_bc);
		ZQ_Matrix<double> Amat(valid_num, 6), bmat(valid_num, 1), xmat(6, 1);
		for (int i = 0; i < valid_num; i++)
		{
			for (int j = 0; j < 6; j++)
				Amat.SetData(i, j, VV[i * 6 + j]);
			bmat.SetData(i, 0, L2);
		}

		/*FILE* out = fopen("Vmat.txt", "w");
		for (int i = 0; i < valid_num; i++)
		{
			for (int j = 0; j < 6; j++)
				fprintf(out, "%e ", VV[i * 6 + j]);
			fprintf(out, "\n");
		}
		fclose(out);*/

		ZQ_SVD::Solve(Amat, xmat, bmat);
		delete[]VV;

		bool flag;
		double x1 = xmat.GetData(0, 0, flag);
		double x2 = xmat.GetData(1, 0, flag);
		double x3 = xmat.GetData(2, 0, flag);
		double x4 = xmat.GetData(3, 0, flag);
		double x5 = xmat.GetData(4, 0, flag);
		double x6 = xmat.GetData(5, 0, flag);

		
		if (x1 <= 0 || x3 <= 0)
		{
			return false;
			delete[]all_h;
		}
		
		double u0 = -x4 / x1;
		double v0 = -x5 / x3;
		
		//assume fovy = 60 degree
		const double m_pi = 4 * atan(1.0);
		double focal = height*0.5 / tan(0.5*m_pi / 3.0);
		double alpha = focal;
		double beta = focal;
		double zA = sqrt(x1*alpha*alpha);

		p[0] = zA;
		p[1] = alpha;
		p[2] = beta;
		p[3] = u0;
		p[4] = v0;

		///
		StickCalib_Data_Header<T> data;
		data.L = len_ac + len_bc;
		data.n_views = n_views;
		data.h = all_h;

		ZQ_LevMarOptions opts;
		ZQ_LevMarReturnInfos infos;
		opts.tol_max_jte = 1e-45;
		opts.tol_dx_square = 1e-45;
		opts.tol_e_square = 1e-45;

		T* hx = new T[n_views];
		memset(hx, 0, sizeof(T)*n_views);
		
		if (!ZQ_LevMar::ZQ_LevMar_Der<T>(_stickCalib_estimate_zA_A_func<T>, _stickCalib_estimate_zA_A_jac<T>, p, hx, 5, n_views, max_iter, opts, infos, &data))
		{
			delete[]hx;
			delete[]all_h;
			return false;
		}
		delete[]hx;

		/**********************/
		zA = p[0];
		alpha = p[1];
		beta = p[2];
		u0 = p[3];
		v0 = p[4];
		T A_1[9] =
		{
			1.0 / alpha, 0, -u0 / alpha,
			0, 1.0 / beta, -v0 / beta,
			0, 0, 1
		};
		
		A_pos[0] = 0;
		A_pos[1] = 0;
		A_pos[2] = 0;
		for (int i = 0; i < n_views; i++)
		{
			const T* a = X2 + n_pts * 2;
			T a_[3] = { a[0], a[1], 1 };
			T A_1_mul_a_[3];
			ZQ_MathBase::MatrixMul(A_1, a_, 3, 3, 1, A_1_mul_a_);
			A_pos[0] += zA*A_1_mul_a_[0];
			A_pos[1] += zA*A_1_mul_a_[1];
			A_pos[2] += zA*A_1_mul_a_[2];
		}
		A_pos[0] /= n_views;
		A_pos[1] /= n_views;
		A_pos[2] /= n_views;

		for (int i = 0; i < n_views; i++)
		{
			T dir[3];
			ZQ_MathBase::MatrixMul(A_1, all_h + i * 3, 3, 3, 1, dir);
			double len2 = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
			if (len2 == 0)
			{
				delete[]all_h;
				return false;
			}
			double len = sqrt(len2);
			dir[0] /= -len;
			dir[1] /= -len;
			dir[2] /= -len;
			theta_phi[i * 2 + 0] = acos(dir[2]);
			theta_phi[i * 2 + 1] = atan2(dir[1], dir[0]);
		}
		intrinsic_para[0] = alpha;
		intrinsic_para[1] = beta;
		intrinsic_para[2] = u0;
		intrinsic_para[3] = v0;
		delete[]all_h;

		return true;
	}

	/*stick calibrate camera: intrinsic parameter,
	based on Lev-Mar optimization, need good initialization.
	left hand coordinates.
	the fixed point of each view must be arranged at the first position
	*/
	template<class T>
	bool ZQ_Calibration::stickCalib_estimate_no_distortion_with_init(int n_views, int n_pts, const T* len_pts_to_A, const T* X2, int max_iter, T* intrinsic_para, T* A_pos, T* theta_phi, double& avg_err_square, double eps)
	{
		///
		StickCalib_Data_Header<T> data;
		data.n_pts = n_pts;
		data.n_views = n_views;
		data.X2 = X2;
		data.len_pts_to_A = len_pts_to_A;
		data.eps = eps;

		ZQ_LevMarOptions opts;
		ZQ_LevMarReturnInfos infos;
		opts.tol_max_jte = 1e-45;
		opts.tol_dx_square = 1e-45;
		opts.tol_e_square = 1e-45;

		T* hx = new T[n_pts * 2 * n_views];
		memset(hx, 0, sizeof(T)*n_pts * 2 * n_views);
		T* p = new T[4 + 3 + n_views * 2];
		memcpy(p, intrinsic_para, sizeof(T)* 4);
		memcpy(p + 4, A_pos, sizeof(T)* 3);
		memcpy(p + 4 + 3, theta_phi, sizeof(T)* 2 * n_views);

		if (!ZQ_LevMar::ZQ_LevMar_Der<T>(_stickCalib_estimate_no_distortion_func<T>, _stickCalib_estimate_no_distortion_jac<T>, p, hx, 4+3 + n_views * 2, n_pts * 2 * n_views, max_iter, opts, infos, &data))
		{
			delete[]hx;
			delete[]p;
			return false;
		}

		avg_err_square = infos.final_e_square / (n_pts*n_views);
		memcpy(intrinsic_para, p, sizeof(T)* 4);
		memcpy(A_pos, p + 4, sizeof(T)* 3);
		memcpy(theta_phi, p + 4 + 3, sizeof(T)* 2 * n_views);
		delete[]hx;
		delete[]p;
		return true;
	}

	/*stick calibrate camera: intrinsic parameter,
	based on Lev-Mar optimization.
	left hand coordinates.
	the fixed point of each view must be arranged at the first position
	*/
	template<class T>
	bool ZQ_Calibration::stickCalib_estimate_no_distortion_without_init(int width, int height, int n_views, int n_pts, const T* len_pts_to_A, const T* X2, int max_iter, T* intrinsic_para, T* A_pos, T* theta_phi, double& avg_err_square, double eps)
	{
		if (!stickCalib_estimate_no_distortion_init(width, height, n_views, n_pts, len_pts_to_A, X2, max_iter, intrinsic_para, theta_phi))
			return false;

		return stickCalib_estimate_no_distortion_with_init(n_views, n_pts, len_pts_to_A, X2, max_iter, intrinsic_para, A_pos, theta_phi, avg_err_square, eps);
	}

	/*
	refer to the paper:
	A flexible new technique for camera calibration[J]. Zhang Z. PAMI 2000.
	compute the camera intrinsic parameter with at least  3 checkboard images.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::_estimate_H_func(const T* p, T* hx, int m, int n, const void* data)
	{
		const Calib_Data_Header<T>* ptr = (const Calib_Data_Header<T>*)data;
		const T* X3 = ptr->X3;
		const T* X2 = ptr->X2;
		const int N = ptr->n_pts;
		const double eps = ptr->eps;

		for(int i = 0;i < N;i++)
		{
			double tmp_x3[3] = 
			{
				p[0]*X3[i*3+0] + p[1]*X3[i*3+1] + p[2]*X3[i*3+2],
				p[3]*X3[i*3+0] + p[4]*X3[i*3+1] + p[5]*X3[i*3+2],
				p[6]*X3[i*3+0] + p[7]*X3[i*3+1] + 1*X3[i*3+2]
			};

			double tmp_x2[2] =
			{
				tmp_x3[0]*tmp_x3[2]/(tmp_x3[2]*tmp_x3[2]+eps*eps),
				tmp_x3[1]*tmp_x3[2]/(tmp_x3[2]*tmp_x3[2]+eps*eps)
			};


			hx[i*2+0] = tmp_x2[0] - X2[i*2+0];
			hx[i*2+1] = tmp_x2[1] - X2[i*2+1];
		}
		return true;
	}


	/*
	refer to the paper:
	A flexible new technique for camera calibration[J]. Zhang Z. PAMI 2000.
	compute the camera intrinsic parameter with at least  3 checkboard images.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::_estimate_H_jac(const T* p, T* jx, int m, int n, const void* data)
	{
		const Calib_Data_Header<T>* ptr = (const Calib_Data_Header<T>*)data;
		const T* X3 = ptr->X3;
		const int N = ptr->n_pts;

		const double eps = ptr->eps;

		//n === N*2, m===8
		memset(jx,0,sizeof(T)*m*n);
		for(int i = 0;i < N;i++)
		{

			double tmp_x3[3] = 
			{
				p[0]*X3[i*3+0] + p[1]*X3[i*3+1] + p[2]*X3[i*3+2],
				p[3]*X3[i*3+0] + p[4]*X3[i*3+1] + p[5]*X3[i*3+2],
				p[6]*X3[i*3+0] + p[7]*X3[i*3+1] + 1*X3[i*3+2]
			};

			jx[(2*i+0)*m+0] = tmp_x3[2]/(tmp_x3[2]*tmp_x3[2]+eps*eps)*X3[i*3+0];
			jx[(2*i+0)*m+1] = tmp_x3[2]/(tmp_x3[2]*tmp_x3[2]+eps*eps)*X3[i*3+1];
			jx[(2*i+0)*m+2] = tmp_x3[2]/(tmp_x3[2]*tmp_x3[2]+eps*eps)*X3[i*3+2];
			jx[(2*i+0)*m+6] = -tmp_x3[0]/(tmp_x3[2]*tmp_x3[2]+eps*eps)*X3[i*3+0];
			jx[(2*i+0)*m+7] = -tmp_x3[0]/(tmp_x3[2]*tmp_x3[2]+eps*eps)*X3[i*3+1];
			jx[(2*i+0)*m+8] = -tmp_x3[0]/(tmp_x3[2]*tmp_x3[2]+eps*eps)*X3[i*3+2];
			jx[(2*i+1)*m+3] = tmp_x3[2]/(tmp_x3[2]*tmp_x3[2]+eps*eps)*X3[i*3+0];
			jx[(2*i+1)*m+4] = tmp_x3[2]/(tmp_x3[2]*tmp_x3[2]+eps*eps)*X3[i*3+1];
			jx[(2*i+1)*m+5] = tmp_x3[2]/(tmp_x3[2]*tmp_x3[2]+eps*eps)*X3[i*3+2];
			jx[(2*i+1)*m+6] = -tmp_x3[1]/(tmp_x3[2]*tmp_x3[2]+eps*eps)*X3[i*3+0];
			jx[(2*i+1)*m+7] = -tmp_x3[1]/(tmp_x3[2]*tmp_x3[2]+eps*eps)*X3[i*3+1];
			//jx[(2*i+1)*m+8] = -tmp_x3[1]/(tmp_x3[2]*tmp_x3[2]+eps*eps)*X3[i*3+2];
		}
		return true;

	}

	/*
	refer to the paper:
	A flexible new technique for camera calibration[J]. Zhang Z. PAMI 2000.
	compute the camera intrinsic parameter with at least  3 checkboard images.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::_estimate_H(T* H, int n_pts, const T* X3, const T* X2, int max_iter, double eps /* = 1e-9*/, bool has_init)
	{

		if (!has_init)
		{
			H[1] = H[2] = H[3] = H[5] = H[6] = H[7] = 0;
			H[0] = H[4] = 1;
			H[8] = 1;
		}
		

		///
		Calib_Data_Header<T> data;
		data.n_pts = n_pts;
		data.X3 = X3;
		data.X2 = X2;
		data.eps = eps;

		ZQ_LevMarOptions opts;
		opts.tol_max_jte = 1e-45;
		opts.tol_dx_square = 1e-45;
		opts.tol_e_square = 1e-45;
		ZQ_LevMarReturnInfos infos;
		T* hx = new T[n_pts*2];
		memset(hx,0,sizeof(T)*n_pts*2);

		if(!ZQ_LevMar::ZQ_LevMar_Der<T>(_estimate_H_func<T>,_estimate_H_jac<T>,H,hx,8,n_pts*2,max_iter,opts,infos,&data))
		{
			delete []hx;
			return false;
		}
		delete []hx;
		return true;
	}

	
	/*
	refer to the paper:
	A flexible new technique for camera calibration[J]. Zhang Z. PAMI 2000.
	compute the camera intrinsic parameter with at least  3 checkboard images. 
	left hand coordinates.
	*/
	template<class T>
	void ZQ_Calibration::_get_v_i_j(const T* H, int i, int j, T* vij)
	{
		vij[0] = H[i]*H[j];
		vij[1] = H[1*3+i]*H[j] + H[i]*H[1*3+j];
		vij[2] = H[2*3+i]*H[j] + H[i]*H[2*3+j];
		vij[3] = H[1*3+i]*H[1*3+j];
		vij[4] = H[2*3+i]*H[1*3+j]+H[1*3+i]*H[2*3+j];
		vij[5] = H[2*3+i]*H[2*3+j];
	}

	/*
	refer to the paper:
	A flexible new technique for camera calibration[J]. Zhang Z. PAMI 2000.
	compute the camera intrinsic parameter with at least  3 checkboard images. 
	left hand coordinates.
	*/
	template<class T>
	void ZQ_Calibration::_get_v_no_distortion(const T* H, T* row1, T* row2)
	{
		T v11[6],v22[6],v12[6];
		_get_v_i_j(H,0,0,v11);
		_get_v_i_j(H,0,1,v12);
		_get_v_i_j(H,1,1,v22);
		memcpy(row1,v12,sizeof(T)*6);
		for(int i = 0;i < 6;i++)
			row2[i] = v11[i]-v22[i];
	}


	/*
	refer to the paper:
	A flexible new technique for camera calibration[J]. Zhang Z. PAMI 2000.
	compute the camera intrinsic parameter with at least  3 checkboard images. 
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::_compute_b(const T* VV, int m, int n, T* b)
	{
		ZQ_Matrix<T> VVmat(m,n),U(m,n),S(n,n),V(n,n);
		for(int i = 0;i < m;i++)
		{
			for(int j = 0;j < n;j++)
				VVmat.SetData(i,j,VV[i*n+j]);
		}

		if(!ZQ_SVD::Decompose(VVmat,U,S,V))
			return false;
		for(int i = 0;i < n;i++)
		{
			bool flag;
			b[i] = V.GetData(i,n-1,flag);
		}
		return true;
	}


	/*
	refer to the paper:
	A flexible new technique for camera calibration[J]. Zhang Z. PAMI 2000.
	compute the camera intrinsic parameter with at least  3 checkboard images. 
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::_compute_A_from_b(const T* b, T* A)
	{
		double B[9] = 
		{
			b[0], b[1], b[2],
			b[1], b[3], b[4],
			b[2], b[4], b[5]
		};

		if(B[0*3+0]*B[1*3+1]-B[0*3+1]*B[0*3+1] == 0)
			return false;
		double v0 = (B[0*3+1]*B[0*3+2]-B[0*3+0]*B[1*3+2])/(B[0*3+0]*B[1*3+1]-B[0*3+1]*B[0*3+1]);
		if(B[0] == 0)
			return false;
		double lambda = B[2*3+2] - (B[0*3+2]*B[0*3+2]+v0*(B[0*3+1]*B[0*3+2]-B[0*3+0]*B[1*3+2]))/B[0];
		if(B[0*3+0] == 0)
			return false;
		if(lambda/B[0*3+0] < 0)
			return false;
		double alpha = sqrt((lambda/B[0*3+0]));
		if(B[0*3+0]*B[1*3+1]-B[0*3+1]*B[0*3+1] == 0)
			return false;
		if(lambda*B[0*3+0]/(B[0*3+0]*B[1*3+1]-B[0*3+1]*B[0*3+1]) < 0)
			return false;
		double beta = sqrt(lambda*B[0*3+0]/(B[0*3+0]*B[1*3+1]-B[0*3+1]*B[0*3+1]));
		if(lambda == 0)
			return false;
		double gamma = -B[0*3+1]*alpha*alpha*beta/lambda;
		if(beta == 0)
			return false;
		double u0 = gamma*v0/beta-B[0*3+2]*alpha*alpha/lambda;

		A[0] = alpha;
		A[1] = gamma;
		A[2] = u0;
		A[3] = 0;
		A[4] = beta;
		A[5] = v0;
		A[6] = 0;
		A[7] = 0;
		A[8] = 1;

		return true;
	}


	/*
	refer to the paper:
	A flexible new technique for camera calibration[J]. Zhang Z. PAMI 2000.
	compute the camera intrinsic parameter with at least  3 checkboard images. 
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::_compute_RT_from_AH(const T* H, const T* A, T* R, T* tt)
	{
		ZQ_Matrix<double> Amat(3,3),Hmat(3,3),RTmat(3,3);
		for(int i = 0;i < 3;i++)
		{
			for(int j = 0;j < 3;j++)
			{
				Amat.SetData(i,j,A[i*3+j]);
				Hmat.SetData(i,j,H[i*3+j]);
			}
		}

		if(!ZQ_SVD::Solve(Amat,RTmat,Hmat))
			return false;

		double r1[3],r2[3],t[3],r3[3];
		bool flag;
		r1[0] = RTmat.GetData(0,0,flag);
		r1[1] = RTmat.GetData(1,0,flag);
		r1[2] = RTmat.GetData(2,0,flag);
		r2[0] = RTmat.GetData(0,1,flag);
		r2[1] = RTmat.GetData(1,1,flag);
		r2[2] = RTmat.GetData(2,1,flag);
		t[0] = RTmat.GetData(0,2,flag);
		t[1] = RTmat.GetData(1,2,flag);
		t[2] = RTmat.GetData(2,2,flag);
		double scale1 = sqrt(r1[0]*r1[0]+r1[1]*r1[1]+r1[2]*r1[2]);
		double scale2 = sqrt(r2[0]*r2[0]+r2[1]*r2[1]+r2[2]*r2[2]);

		if(t[2] < 0)
		{
			scale1 = -scale1;
			scale2 = -scale2;
		}

		if(scale1 != 0)
		{
			r1[0] /= scale1;
			r1[1] /= scale1;
			r1[2] /= scale1;

			t[0] /= scale1;
			t[1] /= scale1;
			t[2] /= scale1;
		}
		else
		{
			return false;
		}

		if(scale2 != 0)
		{
			r2[0] /= scale2;
			r2[1] /= scale2;
			r2[2] /= scale2;
		}
		else
		{
			return false;
		}

		r3[0] = r1[1]*r2[2]-r1[2]*r2[1];
		r3[1] = -r1[0]*r2[2]+r1[2]*r2[0];
		r3[2] = r1[0]*r2[1]-r1[1]*r2[0];


		R[0] = r1[0]; R[1] = r2[0]; R[2] = r3[0];
		R[3] = r1[1]; R[4] = r2[1]; R[5] = r3[1];
		R[6] = r1[2]; R[7] = r2[2]; R[8] = r3[2];
		tt[0] = t[0]; tt[1] = t[1]; tt[2] = t[2];

		return true;
	}


	/*
	refer to the paper:
	A flexible new technique for camera calibration[J]. Zhang Z. PAMI 2000.
	compute the camera intrinsic parameter with at least  3 checkboard images.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::calib_estimate_k_int_rT_init(int n_cams, int n_pts, const T* X3, const T* X2, int max_iter, T* intrinsic_para, T* rT, double eps /* = 1e-9 */ )
	{
		if(n_cams < 3)
			return false;

		T* H = new T[9*n_cams];
		T* M = new T[3*n_pts];
		for(int i = 0;i < n_pts;i++)
		{
			M[i*3+0] = X3[i*3+0];
			M[i*3+1] = X3[i*3+1];
			M[i*3+2] = 1;
		}

		for(int cc = 0;cc < n_cams;cc++)
		{
			if(!_estimate_H(H+cc*9,n_pts,M,X2+n_pts*2*cc,max_iter,eps))
			{
				delete []H;
				delete []M;
				return false;
			}
		}

		T* V = new T[2*n_cams*6];
		for(int cc = 0;cc < n_cams;cc++)
		{
			_get_v_no_distortion(H+cc*9,V+(cc*2)*6,V+(cc*2+1)*6);
		}


		T b[6];
		if(!_compute_b(V,2*n_cams,6,b))
		{
			delete []H;
			delete []M;
			delete []V;
			return false;
		}

		T rec_A[9];
		if(!_compute_A_from_b(b,rec_A))
		{
			delete []H;
			delete []M;
			delete []V;
			return false;
		}

		for(int cc = 0; cc < n_cams;cc++)
		{
			T rec_R[9],rec_T[3];
			if(!_compute_RT_from_AH(H+cc*9,rec_A,rec_R,rec_T))
			{
				delete []H;
				delete []M;
				delete []V;
				return false;
			}
			if(!ZQ_Rodrigues::ZQ_Rodrigues_R2r_fun(rec_R,rT+cc*6))
			{
				delete []H;
				delete []M;
				delete []V;
				return false;
			}
			memcpy(rT+cc*6+3,rec_T,sizeof(double)*3);
		}


		intrinsic_para[0] = rec_A[0];
		intrinsic_para[1] = rec_A[4];
		intrinsic_para[2] = rec_A[2];
		intrinsic_para[3] = rec_A[5];

		delete []H;
		delete []M;
		delete []V;
		return true;
	}

	/*calibrate camera: intrinsic parameter,
	based on Lev-Mar optimization, need good initialization.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::_calib_estimate_no_distortion_func(const T* p, T* hx, int m, int n, const void* data)
	{
		const Calib_Data_Header<T>* ptr = (const Calib_Data_Header<T>*)data;
		const T* X3 = ptr->X3;
		const T* X2 = ptr->X2;
		const int N = ptr->n_pts;
		const int n_cams = ptr->n_cams;

		const double eps = ptr->eps;

		//intrinsic_num = 5, ext_num = 6 * n_cams
		int intrinsic_num = 5;
		double alpha = p[0];
		double beta = p[1];
		double u0 = p[2];
		double v0 = p[3];

		T A[9] = {
			alpha, 0, u0,
			0,beta,v0,
			0,0,1
		};

		T* tmp_X2 = new T[N*2];

		for(int cc = 0;cc < n_cams;cc++)
		{
			const T* r = p+4 + cc*6;
			const T* t = p+4 + cc*6 + 3;
			T R[9];
			ZQ_Rodrigues::ZQ_Rodrigues_r2R(r, R);
			proj_no_distortion(N,A,R,t,X3,tmp_X2,eps);

			for(int i = 0;i < N*2;i++)
				hx[N*2*cc+i] = tmp_X2[i] - X2[N*2*cc+i];
		}
		delete []tmp_X2;
		return true;
	}

	
	/*calibrate camera: intrinsic parameter,
	based on Lev-Mar optimization, need good initialization.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::_calib_estimate_no_distortion_jac(const T* p, T* jx, int m, int n, const void* data)
	{
		const Calib_Data_Header<T>* ptr = (const Calib_Data_Header<T>*)data;
		const T* X3 = ptr->X3;
		const T* X2 = ptr->X2;
		const int N = ptr->n_pts;
		const int n_cams = ptr->n_cams;

		const double eps = ptr->eps;

		memset(jx,0,sizeof(T)*m*n);

		//
		double alpha = p[0];
		double beta = p[1];
		double u0 = p[2];
		double v0 = p[3];

		T A[9] = {
			alpha, 0, u0,
			0,beta,v0,
			0,0,1
		};

		T dAdint[36] = {
			1,0,0,0,
			0,0,0,0,
			0,0,1,0,
			0,0,0,0,
			0,1,0,0,
			0,0,0,1,
			0,0,0,0,
			0,0,0,0,
			0,0,0,0
		};

		//intrinsic_num = 4, ext_num = 6 * n_cams
		for(int cc = 0;cc < n_cams;cc++)
		{
			T dRdr[27],R[9];
			const T *tt = p+4+cc*6+3;
			ZQ_Rodrigues::ZQ_Rodrigues_r2R(p + 4 + 6 * cc, R, dRdr);

			int cur_row_shift = N*2*cc;
			for(int i = 0;i < N;i++)
			{
				int cur_row0 = cur_row_shift + i*2+0;
				int cur_row1 = cur_row_shift + i*2+1;
				T cur_X3[3] = {X3[i*3+0],X3[i*3+1],X3[i*3+2]};

				//var1 = R*X3+T: dvar1dR,dvar1dT
				T var1[3] = {
					R[0]*cur_X3[0]+R[1]*cur_X3[1]+R[2]*cur_X3[2] + tt[0],
					R[3]*cur_X3[0]+R[4]*cur_X3[1]+R[5]*cur_X3[2] + tt[1],
					R[6]*cur_X3[0]+R[7]*cur_X3[1]+R[8]*cur_X3[2] + tt[2]
				};

				T dvar1dR[27] = {
					cur_X3[0],cur_X3[1],cur_X3[2],0,0,0,0,0,0,
					0,0,0,cur_X3[0],cur_X3[1],cur_X3[2],0,0,0,
					0,0,0,0,0,0,cur_X3[0],cur_X3[1],cur_X3[2]
				};
				T dvar1dT[9] = 
				{
					1,0,0,
					0,1,0,
					0,0,1
				};

				//
				T dvar1drT[18] = {0};
				for(int iii = 0;iii < 3;iii++)
				{
					for(int jjj = 0;jjj < 3;jjj++)
					{
						for(int kkk = 0;kkk < 9;kkk++)
							dvar1drT[iii*6+jjj] += dvar1dR[iii*9+kkk]*dRdr[kkk*3+jjj];
						dvar1drT[iii*6+jjj+3] = dvar1dT[iii*3+jjj];
					}
				}


				//var2 = A*var1: dvar2dA, dvar2dvar1
				T var2[3] = {
					A[0]*var1[0]+A[1]*var1[1]+A[2]*var1[2],
					A[3]*var1[0]+A[4]*var1[1]+A[5]*var1[2],
					A[6]*var1[0]+A[7]*var1[1]+A[8]*var1[2]
				};

				T dvar2dA[27] = {
					var1[0],var1[1],var1[2],0,0,0,0,0,0,
					0,0,0,var1[0],var1[1],var1[2],0,0,0,
					0,0,0,0,0,0,var1[0],var1[1],var1[2]
				};
				T dvar2dvar1[9] = {
					A[0],A[1],A[2],
					A[3],A[4],A[5],
					A[6],A[7],A[8]
				};
				T dvar2drT[18] = {0};
				ZQ_MathBase::MatrixMul(dvar2dvar1, dvar1drT, 3, 3, 6, dvar2drT);
				
				T dvar2dint[12] = {0};
				ZQ_MathBase::MatrixMul(dvar2dA, dAdint, 3, 9, 4, dvar2dint);
				


				//X2 = [var2(1)/var2(3);var2(2)/var2(3)]
				T cur_X2[2] = {
					var2[0]*var2[2]/(var2[2]*var2[2]+eps*eps),
					var2[1]*var2[2]/(var2[2]*var2[2]+eps*eps)
				};

				T dX2dvar2[6] = 
				{
					var2[2]/(var2[2]*var2[2]+eps*eps),0,-var2[0]/(var2[2]*var2[2]+eps*eps),
					0,var2[2]/(var2[2]*var2[2]+eps*eps),-var2[1]/(var2[2]*var2[2]+eps*eps)
				};

				T dX2dint[8] = {0};
				T dX2drT[12] = {0};
				ZQ_MathBase::MatrixMul(dX2dvar2, dvar2dint, 2, 3, 4, dX2dint);

				ZQ_MathBase::MatrixMul(dX2dvar2, dvar2drT, 2, 3, 6, dX2drT);

				for(int iii = 0;iii < 4;iii++)
				{
					jx[cur_row0*m+iii] = dX2dint[iii];
					jx[cur_row1*m+iii] = dX2dint[4+iii];
				}

				for(int iii = 0;iii < 6;iii++)
				{
					jx[cur_row0*m+4+cc*6+iii] = dX2drT[iii];
					jx[cur_row1*m+4+cc*6+iii] = dX2drT[6+iii];
				}
			}
		}
		return true;
	}


	/*calibrate camera: intrinsic parameter,
	based on Lev-Mar optimization, need good initialization.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::calib_estimate_no_distortion_with_init(int n_cams, int n_pts, const T* X3, const T* X2, int max_iter,T* intrinsic_para, T* rT, double& avg_err_square, double eps /* = 1e-9 */)
	{

		///
		Calib_Data_Header<T> data;
		data.n_pts = n_pts;
		data.n_cams = n_cams;
		data.X3 = X3;
		data.X2 = X2;
		data.eps = eps;

		ZQ_LevMarOptions opts;
		ZQ_LevMarReturnInfos infos;
		opts.tol_max_jte = 1e-45;
		opts.tol_dx_square = 1e-45;
		opts.tol_e_square = 1e-45;

		T* hx = new T[n_pts*2*n_cams];
		memset(hx,0,sizeof(T)*n_pts*2*n_cams);
		T* p = new T[4+n_cams*6];
		memcpy(p,intrinsic_para,sizeof(T)*4);
		memcpy(p+4,rT,sizeof(T)*6*n_cams);
		
		if(!ZQ_LevMar::ZQ_LevMar_Der<T>(_calib_estimate_no_distortion_func<T>,_calib_estimate_no_distortion_jac<T>,p,hx,4+n_cams*6,n_pts*2*n_cams,max_iter,opts,infos,&data))
		{
			delete []hx;
			delete []p;
			return false;
		}

		avg_err_square = infos.final_e_square / (n_pts*n_cams);
		memcpy(intrinsic_para,p,sizeof(T)*4);
		memcpy(rT,p+4,sizeof(T)*6*n_cams);
		return true;
		
	}


	/*calibrate camera: intrinsic parameter and distortion (k1,k2),
	based on Lev-Mar optimization, need good initialization.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::_calib_estimate_k_int_rT_func(const T* p, T* hx, int m, int n, const void* data)
	{
		const Calib_Data_Header<T>* ptr = (const Calib_Data_Header<T>*)data;
		const T* X3 = ptr->X3;
		const T* X2 = ptr->X2;
		const int N = ptr->n_pts;
		const int n_cams = ptr->n_cams;

		const double eps = ptr->eps;

		//intrinsic_num = 4, k_num = 2, ext_num = 6 * n_cams
		
		double alpha = p[0];
		double beta = p[1];
		double u0 = p[2];
		double v0 = p[3];

		const T* center = p + 2;
		const T* k = p + 4;

		T A[9] = {
			alpha, 0, u0,
			0,beta,v0,
			0,0,1
		};

		T* tmp_X2 = new T[N*2];

		for(int cc = 0;cc < n_cams;cc++)
		{
			const T* r = p+6 + cc*6;
			const T* t = p+6 + cc*6 + 3;
			T R[9];
			if(!ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun(r,R))
			{
				//printf("error: Rodrigues r to R failed:(%d)%s\n",__LINE__,__FILE__);
				delete []tmp_X2;
				return false;
			}

			proj_distortion_k2(N,A,R,t,center,k,X3,tmp_X2,eps);

			for(int i = 0;i < N*2;i++)
				hx[N*2*cc+i] = tmp_X2[i] - X2[N*2*cc+i];
		}
		delete []tmp_X2;
		return true;

	}


	/*calibrate camera: intrinsic parameter and distortion (k1,k2),
	based on Lev-Mar optimization, need good initialization.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::_calib_estimate_k_int_rT_jac(const T* p, T* jx, int m, int n, const void* data)
	{
		const Calib_Data_Header<T>* ptr = (const Calib_Data_Header<T>*)data;
		const T* X3 = ptr->X3;
		const T* X2 = ptr->X2;
		const int N = ptr->n_pts;
		const int n_cams = ptr->n_cams;

		const double eps = ptr->eps;

		memset(jx,0,sizeof(T)*m*n);

		//
		double alpha = p[0];
		double beta = p[1];
		double u0 = p[2];
		double v0 = p[3];

		const T* center = p + 2;
		const T* k = p + 4;
		const T* rT = p + 6;

		T A[9] = {
			alpha, 0, u0,
			0,beta,v0,
			0,0,1
		};

		T dAdint[36] = {
			1,0,0,0,
			0,0,0,0,
			0,0,1,0,
			0,0,0,0,
			0,1,0,0,
			0,0,0,1,
			0,0,0,0,
			0,0,0,0,
			0,0,0,0
		};

		//intrinsic_num = 4, ext_num = 6 * n_cams
		for(int cc = 0;cc < n_cams;cc++)
		{
			T dRdr[27],R[9];
			const T *tt = rT+cc*6+3;
			if(!ZQ_Rodrigues::ZQ_Rodrigues_r2R_jac(rT+6*cc,dRdr))
			{
				//printf("error: Rodrigues r to R failed:(%d)%s\n",__LINE__,__FILE__);
				return false;
			}
			if(!ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun(rT+6*cc,R))
			{
				//printf("error: Rodrigues r to R failed:(%d)%s\n",__LINE__,__FILE__);
				return false;
			}

			int cur_row_shift = N*2*cc;
			for(int i = 0;i < N;i++)
			{
				int cur_row0 = cur_row_shift + i*2+0;
				int cur_row1 = cur_row_shift + i*2+1;
				T cur_X3[3] = {X3[i*3+0],X3[i*3+1],X3[i*3+2]};

				//var1 = R*X3+T: dvar1dR,dvar1dT
				T var1[3] = {
					R[0]*cur_X3[0]+R[1]*cur_X3[1]+R[2]*cur_X3[2] + tt[0],
					R[3]*cur_X3[0]+R[4]*cur_X3[1]+R[5]*cur_X3[2] + tt[1],
					R[6]*cur_X3[0]+R[7]*cur_X3[1]+R[8]*cur_X3[2] + tt[2]
				};

				T dvar1dR[27] = {
					cur_X3[0],cur_X3[1],cur_X3[2],0,0,0,0,0,0,
					0,0,0,cur_X3[0],cur_X3[1],cur_X3[2],0,0,0,
					0,0,0,0,0,0,cur_X3[0],cur_X3[1],cur_X3[2]
				};
				T dvar1dT[9] = 
				{
					1,0,0,
					0,1,0,
					0,0,1
				};

				//
				T dvar1drT[18] = {0};
				for(int iii = 0;iii < 3;iii++)
				{
					for(int jjj = 0;jjj < 3;jjj++)
					{
						for(int kkk = 0;kkk < 9;kkk++)
							dvar1drT[iii*6+jjj] += dvar1dR[iii*9+kkk]*dRdr[kkk*3+jjj];
						dvar1drT[iii*6+jjj+3] = dvar1dT[iii*3+jjj];
					}
				}


				//var2 = A*var1: dvar2dA, dvar2dvar1
				T var2[3] = {
					A[0]*var1[0]+A[1]*var1[1]+A[2]*var1[2],
					A[3]*var1[0]+A[4]*var1[1]+A[5]*var1[2],
					A[6]*var1[0]+A[7]*var1[1]+A[8]*var1[2]
				};

				T dvar2dA[27] = {
					var1[0],var1[1],var1[2],0,0,0,0,0,0,
					0,0,0,var1[0],var1[1],var1[2],0,0,0,
					0,0,0,0,0,0,var1[0],var1[1],var1[2]
				};
				T dvar2dvar1[9] = {
					A[0],A[1],A[2],
					A[3],A[4],A[5],
					A[6],A[7],A[8]
				};
				T dvar2drT[18] = {0};
				ZQ_MathBase::MatrixMul(dvar2dvar1, dvar1drT, 3, 3, 6, dvar2drT);
				
				T dvar2dint[12] = {0};
				ZQ_MathBase::MatrixMul(dvar2dA, dAdint, 3, 9, 4, dvar2dint);


				//X2 = [var2(1)/var2(3);var2(2)/var2(3)]
				T cur_X2[2] = {
					var2[0]*var2[2]/(var2[2]*var2[2]+eps*eps),
					var2[1]*var2[2]/(var2[2]*var2[2]+eps*eps)
				};

				T dX2dvar2[6] = 
				{
					var2[2]/(var2[2]*var2[2]+eps*eps),0,-var2[0]/(var2[2]*var2[2]+eps*eps),
					0,var2[2]/(var2[2]*var2[2]+eps*eps),-var2[1]/(var2[2]*var2[2]+eps*eps)
				};

				T dX2dint[8] = {0};
				T dX2drT[12] = {0};
				ZQ_MathBase::MatrixMul(dX2dvar2, dvar2dint, 2, 3, 4, dX2dint);
				

				ZQ_MathBase::MatrixMul(dX2dvar2, dvar2drT, 2, 3, 6, dX2drT);
				

				// var3 = x-u: 
				// var4 = var3^2
				// var5 = y-v
				// var6 = var5^2;
				// var7 = var4+var6
				// var8 = var7^2
				double var3 = cur_X2[0] - u0;
				double var4 = var3*var3;
				double var5 = cur_X2[1] - v0;
				double var6 = var5*var5;
				double var7 = var4+var6;
				double var8 = var7*var7;

				//
				T dudint[4] = {0,0,1,0};
				T dvdint[4] = {0,0,0,1};
				T dvar3dint[4];
				T dvar4dint[4];
				T dvar5dint[4];
				T dvar6dint[4];
				T dvar7dint[4];
				T dvar8dint[4];

				const T* dvar3drT = dX2drT;
				const T* dvar5drT = dX2drT+6;

				T dvar4drT[6];
				T dvar6drT[6];
				T dvar7drT[6];
				T dvar8drT[6];

				for(int iii = 0;iii < 4;iii++)
				{
					dvar3dint[iii] = dX2dint[iii] - dudint[iii];
					dvar4dint[iii] = 2*var3*dvar3dint[iii];
					dvar5dint[iii] = dX2dint[4+iii] - dvdint[iii];
					dvar6dint[iii] = 2*var5*dvar5dint[iii];
					dvar7dint[iii] = dvar4dint[iii]+dvar6dint[iii];
					dvar8dint[iii] = 2*var7*dvar7dint[iii];
				}

				for(int iii = 0;iii < 6;iii++)
				{
					dvar4drT[iii] = 2*var3*dvar3drT[iii];
					dvar6drT[iii] = 2*var5*dvar5drT[iii];
					dvar7drT[iii] = dvar4drT[iii] + dvar6drT[iii];
					dvar8drT[iii] = 2*var7*dvar7drT[iii];
				}

				T dxbardint[4];
				T dybardint[4];
				T dxbardrT[6];
				T dybardrT[6];
				for(int iii = 0;iii < 4;iii++)
				{
					dxbardint[iii] = dX2dint[iii]   + dvar3dint[iii]*(k[0]*var7+k[1]*var8) + var3*(k[0]*dvar7dint[iii]+k[1]*dvar8dint[iii]);
					dybardint[iii] = dX2dint[4+iii] + dvar5dint[iii]*(k[0]*var7+k[1]*var8) + var5*(k[0]*dvar7dint[iii]+k[1]*dvar8dint[iii]);
				}
				for(int iii = 0;iii < 6;iii++)
				{
					dxbardrT[iii] = dX2drT[iii]   + dvar3drT[iii]*(k[0]*var7+k[1]*var8) + var3*(k[0]*dvar7drT[iii]+k[1]*dvar8drT[iii]);
					dybardrT[iii] = dX2drT[iii+6] + dvar5drT[iii]*(k[0]*var7+k[1]*var8) + var5*(k[0]*dvar7drT[iii]+k[1]*dvar8drT[iii]);
				}

				double x_u0 = cur_X2[0] - u0;
				double y_v0 = cur_X2[1] - v0;
				double dXbardk1 = x_u0*(x_u0*x_u0+y_v0*y_v0);
				double dXbardk2 = x_u0*((x_u0*x_u0+y_v0*y_v0)*(x_u0*x_u0+y_v0*y_v0));
				double dYbardk1 = y_v0*(x_u0*x_u0+y_v0*y_v0);
				double dYbardk2 = y_v0*((x_u0*x_u0+y_v0*y_v0)*(x_u0*x_u0+y_v0*y_v0));

				jx[cur_row0*m+4] = dXbardk1;
				jx[cur_row0*m+5] = dXbardk2;
				jx[cur_row1*m+4] = dYbardk1;
				jx[cur_row1*m+5] = dYbardk2;


				for(int iii = 0;iii < 4;iii++)
				{
					jx[cur_row0*m+iii] = dxbardint[iii];
					jx[cur_row1*m+iii] = dybardint[iii];
				}

				for(int iii = 0;iii < 6;iii++)
				{
					jx[cur_row0*m+6+cc*6+iii] = dxbardrT[iii];
					jx[cur_row1*m+6+cc*6+iii] = dybardrT[iii];
				}
			}
		}
		return true;
	}

	/*calibrate camera: intrinsic parameter
	fix distortion (k1,k2),
	based on Lev-Mar optimization, need good initialization.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::_calib_estimate_int_rT_fix_k_func(const T* p, T* hx, int m, int n, const void* data)
	{
		const Calib_Data_Header<T>* ptr = (const Calib_Data_Header<T>*)data;
		const T* X3 = ptr->X3;
		const T* X2 = ptr->X2;
		const int N = ptr->n_pts;
		const int n_cams = ptr->n_cams;
		const T* k = ptr->k;

		const double eps = ptr->eps;

		//intrinsic_num = 4, k_num = 2, ext_num = 6 * n_cams

		double alpha = p[0];
		double beta = p[1];
		double u0 = p[2];
		double v0 = p[3];

		const T* center = p + 2;

		T A[9] = {
			alpha, 0, u0,
			0, beta, v0,
			0, 0, 1
		};

		T* tmp_X2 = new T[N * 2];

		for (int cc = 0; cc < n_cams; cc++)
		{
			const T* r = p + 4 + cc * 6;
			const T* t = p + 4 + cc * 6 + 3;
			T R[9];
			if (!ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun(r, R))
			{
				//printf("error: Rodrigues r to R failed:(%d)%s\n",__LINE__,__FILE__);
				delete[]tmp_X2;
				return false;
			}

			proj_distortion_k2(N, A, R, t, center, k, X3, tmp_X2, eps);

			for (int i = 0; i < N * 2; i++)
				hx[N * 2 * cc + i] = tmp_X2[i] - X2[N * 2 * cc + i];
		}
		delete[]tmp_X2;
		return true;
	}

	template<class T>
	bool ZQ_Calibration::_calib_estimate_int_rT_fix_k_jac(const T* p, T* jx, int m, int n, const void* data)
	{
		const Calib_Data_Header<T>* ptr = (const Calib_Data_Header<T>*)data;
		const T* X3 = ptr->X3;
		const T* X2 = ptr->X2;
		const int N = ptr->n_pts;
		const int n_cams = ptr->n_cams;
		const T* k = ptr->k;

		const double eps = ptr->eps;

		memset(jx, 0, sizeof(T)*m*n);

		//
		double alpha = p[0];
		double beta = p[1];
		double u0 = p[2];
		double v0 = p[3];

		const T* center = p + 2;
		const T* rT = p + 4;

		T A[9] = {
			alpha, 0, u0,
			0, beta, v0,
			0, 0, 1
		};

		T dAdint[36] = {
			1, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 1,
			0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 0
		};

		//intrinsic_num = 4, ext_num = 6 * n_cams
		for (int cc = 0; cc < n_cams; cc++)
		{
			T dRdr[27], R[9];
			const T *tt = rT + cc * 6 + 3;
			if (!ZQ_Rodrigues::ZQ_Rodrigues_r2R_jac(rT + 6 * cc, dRdr))
			{
				//printf("error: Rodrigues r to R failed:(%d)%s\n",__LINE__,__FILE__);
				return false;
			}
			if (!ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun(rT + 6 * cc, R))
			{
				//printf("error: Rodrigues r to R failed:(%d)%s\n",__LINE__,__FILE__);
				return false;
			}

			int cur_row_shift = N * 2 * cc;
			for (int i = 0; i < N; i++)
			{
				int cur_row0 = cur_row_shift + i * 2 + 0;
				int cur_row1 = cur_row_shift + i * 2 + 1;
				T cur_X3[3] = { X3[i * 3 + 0], X3[i * 3 + 1], X3[i * 3 + 2] };

				//var1 = R*X3+T: dvar1dR,dvar1dT
				T var1[3] = {
					R[0] * cur_X3[0] + R[1] * cur_X3[1] + R[2] * cur_X3[2] + tt[0],
					R[3] * cur_X3[0] + R[4] * cur_X3[1] + R[5] * cur_X3[2] + tt[1],
					R[6] * cur_X3[0] + R[7] * cur_X3[1] + R[8] * cur_X3[2] + tt[2]
				};

				T dvar1dR[27] = {
					cur_X3[0], cur_X3[1], cur_X3[2], 0, 0, 0, 0, 0, 0,
					0, 0, 0, cur_X3[0], cur_X3[1], cur_X3[2], 0, 0, 0,
					0, 0, 0, 0, 0, 0, cur_X3[0], cur_X3[1], cur_X3[2]
				};
				T dvar1dT[9] =
				{
					1, 0, 0,
					0, 1, 0,
					0, 0, 1
				};

				//
				T dvar1drT[18] = { 0 };
				for (int iii = 0; iii < 3; iii++)
				{
					for (int jjj = 0; jjj < 3; jjj++)
					{
						for (int kkk = 0; kkk < 9; kkk++)
							dvar1drT[iii * 6 + jjj] += dvar1dR[iii * 9 + kkk] * dRdr[kkk * 3 + jjj];
						dvar1drT[iii * 6 + jjj + 3] = dvar1dT[iii * 3 + jjj];
					}
				}


				//var2 = A*var1: dvar2dA, dvar2dvar1
				T var2[3] = {
					A[0] * var1[0] + A[1] * var1[1] + A[2] * var1[2],
					A[3] * var1[0] + A[4] * var1[1] + A[5] * var1[2],
					A[6] * var1[0] + A[7] * var1[1] + A[8] * var1[2]
				};

				T dvar2dA[27] = {
					var1[0], var1[1], var1[2], 0, 0, 0, 0, 0, 0,
					0, 0, 0, var1[0], var1[1], var1[2], 0, 0, 0,
					0, 0, 0, 0, 0, 0, var1[0], var1[1], var1[2]
				};
				T dvar2dvar1[9] = {
					A[0], A[1], A[2],
					A[3], A[4], A[5],
					A[6], A[7], A[8]
				};
				T dvar2drT[18] = { 0 };
				ZQ_MathBase::MatrixMul(dvar2dvar1, dvar1drT, 3, 3, 6, dvar2drT);

				T dvar2dint[12] = { 0 };
				ZQ_MathBase::MatrixMul(dvar2dA, dAdint, 3, 9, 4, dvar2dint);


				//X2 = [var2(1)/var2(3);var2(2)/var2(3)]
				T cur_X2[2] = {
					var2[0] * var2[2] / (var2[2] * var2[2] + eps*eps),
					var2[1] * var2[2] / (var2[2] * var2[2] + eps*eps)
				};

				T dX2dvar2[6] =
				{
					var2[2] / (var2[2] * var2[2] + eps*eps), 0, -var2[0] / (var2[2] * var2[2] + eps*eps),
					0, var2[2] / (var2[2] * var2[2] + eps*eps), -var2[1] / (var2[2] * var2[2] + eps*eps)
				};

				T dX2dint[8] = { 0 };
				T dX2drT[12] = { 0 };
				ZQ_MathBase::MatrixMul(dX2dvar2, dvar2dint, 2, 3, 4, dX2dint);


				ZQ_MathBase::MatrixMul(dX2dvar2, dvar2drT, 2, 3, 6, dX2drT);


				// var3 = x-u: 
				// var4 = var3^2
				// var5 = y-v
				// var6 = var5^2;
				// var7 = var4+var6
				// var8 = var7^2
				double var3 = cur_X2[0] - u0;
				double var4 = var3*var3;
				double var5 = cur_X2[1] - v0;
				double var6 = var5*var5;
				double var7 = var4 + var6;
				double var8 = var7*var7;

				//
				T dudint[4] = { 0, 0, 1, 0 };
				T dvdint[4] = { 0, 0, 0, 1 };
				T dvar3dint[4];
				T dvar4dint[4];
				T dvar5dint[4];
				T dvar6dint[4];
				T dvar7dint[4];
				T dvar8dint[4];

				const T* dvar3drT = dX2drT;
				const T* dvar5drT = dX2drT + 6;

				T dvar4drT[6];
				T dvar6drT[6];
				T dvar7drT[6];
				T dvar8drT[6];

				for (int iii = 0; iii < 4; iii++)
				{
					dvar3dint[iii] = dX2dint[iii] - dudint[iii];
					dvar4dint[iii] = 2 * var3*dvar3dint[iii];
					dvar5dint[iii] = dX2dint[4 + iii] - dvdint[iii];
					dvar6dint[iii] = 2 * var5*dvar5dint[iii];
					dvar7dint[iii] = dvar4dint[iii] + dvar6dint[iii];
					dvar8dint[iii] = 2 * var7*dvar7dint[iii];
				}

				for (int iii = 0; iii < 6; iii++)
				{
					dvar4drT[iii] = 2 * var3*dvar3drT[iii];
					dvar6drT[iii] = 2 * var5*dvar5drT[iii];
					dvar7drT[iii] = dvar4drT[iii] + dvar6drT[iii];
					dvar8drT[iii] = 2 * var7*dvar7drT[iii];
				}

				T dxbardint[4];
				T dybardint[4];
				T dxbardrT[6];
				T dybardrT[6];
				for (int iii = 0; iii < 4; iii++)
				{
					dxbardint[iii] = dX2dint[iii] + dvar3dint[iii] * (k[0] * var7 + k[1] * var8) + var3*(k[0] * dvar7dint[iii] + k[1] * dvar8dint[iii]);
					dybardint[iii] = dX2dint[4 + iii] + dvar5dint[iii] * (k[0] * var7 + k[1] * var8) + var5*(k[0] * dvar7dint[iii] + k[1] * dvar8dint[iii]);
				}
				for (int iii = 0; iii < 6; iii++)
				{
					dxbardrT[iii] = dX2drT[iii] + dvar3drT[iii] * (k[0] * var7 + k[1] * var8) + var3*(k[0] * dvar7drT[iii] + k[1] * dvar8drT[iii]);
					dybardrT[iii] = dX2drT[iii + 6] + dvar5drT[iii] * (k[0] * var7 + k[1] * var8) + var5*(k[0] * dvar7drT[iii] + k[1] * dvar8drT[iii]);
				}


				for (int iii = 0; iii < 4; iii++)
				{
					jx[cur_row0*m + iii] = dxbardint[iii];
					jx[cur_row1*m + iii] = dybardint[iii];
				}

				for (int iii = 0; iii < 6; iii++)
				{
					jx[cur_row0*m + 4 + cc * 6 + iii] = dxbardrT[iii];
					jx[cur_row1*m + 4 + cc * 6 + iii] = dybardrT[iii];
				}
			}
		}
		return true;
	}

	template<class T>
	bool ZQ_Calibration::_estimate_k_with_int_rT(int n_cams, int n_pts, const T* X3, const T* X2, T* k, const T* intrinsic_para, const T* rT, double eps)
	{
		//intrinsic_num = 4, k_num = 2, ext_num = 6 * n_cams

		double alpha = intrinsic_para[0];
		double beta = intrinsic_para[1];
		double u0 = intrinsic_para[2];
		double v0 = intrinsic_para[3];

		T A[9] = {
			alpha, 0, u0,
			0, beta, v0,
			0, 0, 1
		};

		ZQ_Matrix<T> mat_A(n_cams*n_pts * 2, 2), mat_b(n_cams*n_pts*2,1), mat_x(2,1);

		T* tmp_X2 = new T[n_cams * n_pts * 2];

		for (int cc = 0; cc < n_cams; cc++)
		{
			const T* r = rT + cc * 6;
			const T* t = rT + cc * 6 + 3;
			T R[9];
			if (!ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun(r, R))
			{
				//printf("error: Rodrigues r to R failed:(%d)%s\n",__LINE__,__FILE__);
				delete[]tmp_X2;
				return false;
			}

			proj_no_distortion(n_pts, A, R, t, X3, tmp_X2 + cc*n_pts * 2, eps);
			for (int pp = 0; pp < n_pts; pp++)
			{
				T x_i = tmp_X2[cc*n_pts * 2 + pp * 2 + 0] - u0;
				T y_i = tmp_X2[cc*n_pts * 2 + pp * 2 + 1] - v0;
				T x_o = X2[cc*n_pts * 2 + pp * 2 + 0] - u0;
				T y_o = X2[cc*n_pts * 2 + pp * 2 + 1] - v0;
				T r2_i = x_i*x_i + y_i*y_i;
				T r4_i = r2_i*r2_i;
				mat_A.SetData(cc*n_pts * 2 + pp * 2 + 0, 0, x_i*r2_i);
				mat_A.SetData(cc*n_pts * 2 + pp * 2 + 0, 1, x_i*r4_i);
				mat_A.SetData(cc*n_pts * 2 + pp * 2 + 1, 0, y_i*r2_i);
				mat_A.SetData(cc*n_pts * 2 + pp * 2 + 1, 1, y_i*r4_i);
				mat_b.SetData(cc*n_pts * 2 + pp * 2 + 0, 0, x_o - x_i);
				mat_b.SetData(cc*n_pts * 2 + pp * 2 + 1, 0, y_o - y_i);
			}
		}

		if (0)
		{
			ZQ_Matrix<T> mat_At = mat_A.GetTransposeMatrix();
			ZQ_Matrix<T> mat_AtA = mat_At*mat_A;
			ZQ_Matrix<T> mat_Atb = mat_At*mat_b;
			if (!ZQ_SVD::Solve(mat_AtA, mat_x, mat_Atb))
			{
				delete[]tmp_X2;
				return false;
			}
		}
		else
		{
			if (!ZQ_SVD::Solve(mat_A, mat_x, mat_b))
			{
			delete[]tmp_X2;
			return false;
			}
		}
		
		bool flag;
		k[0] = mat_x.GetData(0, 0, flag);
		k[1] = mat_x.GetData(1, 0, flag);
		delete[]tmp_X2;
		return true;
	}

	/*calibrate camera: intrinsic parameter and distortion (k1,k2),
	based on Lev-Mar optimization, need good initialization.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::calib_estimate_k_int_rT_with_init(int n_cams, int n_pts,const T* X3, const T* X2, int max_iter, T* k,  T* intrinsic_para, T* rT, double& avg_err_square, double eps /* = 1e-9 */)
	{
		///
		Calib_Data_Header<T> data;
		data.n_pts = n_pts;
		data.n_cams = n_cams;
		data.X3 = X3;
		data.X2 = X2;
		data.eps = eps;
		

		ZQ_LevMarOptions opts;
		ZQ_LevMarReturnInfos infos;
		opts.tol_max_jte = 1e-45;
		opts.tol_dx_square = 1e-45;
		opts.tol_e_square = 1e-45;

		T* hx = new T[n_pts*2*n_cams];
		memset(hx,0,sizeof(T)*n_pts*2*n_cams);
		T* p = new T[6+n_cams*6];
		memcpy(p,intrinsic_para,sizeof(T)*4);
		memcpy(p+4,k,sizeof(T)*2);
		memcpy(p+6,rT,sizeof(T)*6*n_cams);

		if(!ZQ_LevMar::ZQ_LevMar_Der<T>(_calib_estimate_k_int_rT_func<T>,_calib_estimate_k_int_rT_jac<T>,p,hx,6+n_cams*6,n_pts*n_cams*2,max_iter,opts,infos,&data))
		{
			delete []hx;
			delete []p;
			return false;
		}
		avg_err_square = infos.final_e_square / (n_pts*n_cams);

		memcpy(intrinsic_para,p,sizeof(T)*4);
		memcpy(k,p+4,sizeof(T)*2);
		memcpy(rT,p+6,sizeof(T)*6*n_cams);
		delete []p;
		delete []hx;
		return true;
	}

	template<class T>
	bool ZQ_Calibration::calib_estimate_int_rT_fix_k_with_init(int n_cams, int n_pts, const T* X3, const T* X2, int max_iter, const T* k, T* intrinsic_para, T* rT, double& avg_err_square, double eps)
	{
		///
		Calib_Data_Header<T> data;
		data.n_pts = n_pts;
		data.n_cams = n_cams;
		data.X3 = X3;
		data.X2 = X2;
		data.eps = eps;
		data.k = k;


		ZQ_LevMarOptions opts;
		ZQ_LevMarReturnInfos infos;
		opts.tol_max_jte = 1e-45;
		opts.tol_dx_square = 1e-45;
		opts.tol_e_square = 1e-45;

		T* hx = new T[n_pts * 2 * n_cams];
		memset(hx, 0, sizeof(T)*n_pts * 2 * n_cams);
		T* p = new T[4 + n_cams * 6];
		memcpy(p, intrinsic_para, sizeof(T)* 4);
		memcpy(p + 4, rT, sizeof(T)* 6 * n_cams);

		if (!ZQ_LevMar::ZQ_LevMar_Der<T>(_calib_estimate_int_rT_fix_k_func<T>, _calib_estimate_int_rT_fix_k_jac<T>, p, hx, 4 + n_cams * 6, n_pts*n_cams * 2, max_iter, opts, infos, &data))
		{
			delete[]hx;
			delete[]p;
			return false;
		}
		avg_err_square = infos.final_e_square / (n_pts*n_cams);

		memcpy(intrinsic_para, p, sizeof(T)* 4);
		memcpy(rT, p + 4, sizeof(T)* 6 * n_cams);
		delete[]p;
		delete[]hx;
		return true;
	}

	/*calibrate camera: intrinsic parameter and distortion (k1,k2),
	based on Lev-Mar optimization, do not need good initialization.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::calib_estimate_k_int_rT_without_init(int n_cams, int n_pts, const T* X3, const T* X2, int max_iter, T* k, T* intrinsic_para, T* rT, double& avg_err_square, double eps /* = 1e-9 */)
	{

		if(!calib_estimate_k_int_rT_init(n_cams,n_pts,X3,X2,max_iter,intrinsic_para,rT,eps))
			return false;

		k[0] = k[1] = 0;

		if(!calib_estimate_k_int_rT_with_init(n_cams,n_pts,X3,X2,max_iter,k,intrinsic_para,rT,avg_err_square,eps))
			return false;
		return true; 
	}

	/*calibrate camera: intrinsic parameter and distortion (k1,k2),
	based on Lev-Mar optimization,  need good initialization.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::calib_estimate_k_int_rT_alt_with_init(int n_cams, int n_pts, const T* X3, const T* X2, int alt_iter, int max_iter, T* k, T* intrinsic_para, T* rT, double& avg_err_square, double eps /* = 1e-9 */, bool display /* = true */)
	{
		for (int it = 0; it < alt_iter; it++)
		{
			if (!_estimate_k_with_int_rT(n_cams, n_pts, X3, X2, k, intrinsic_para, rT, eps))
			{
				return false;
			}
			if (!calib_estimate_int_rT_fix_k_with_init(n_cams, n_pts, X3, X2, max_iter, k, intrinsic_para, rT, avg_err_square, eps))
			{
				return false;
			}
			if (display)
			{
				printf("focal=[%8.1f %8.1f], center=[%8.1f %8.1f], k=[%8.2e %8.2e], avg_err_square = %8.2e\n",
					intrinsic_para[0], intrinsic_para[1], intrinsic_para[2], intrinsic_para[3], k[0], k[1], avg_err_square);
			}
		}

		return true;
	}


	/*calibrate camera: intrinsic parameter and distortion (k1,k2),
	based on Lev-Mar optimization, do not need good initialization.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::calib_estimate_k_int_rT_alt_without_init(int n_cams, int n_pts, const T* X3, const T* X2, int alt_iter, int max_iter, T* k, T* intrinsic_para, T* rT, double& avg_err_square, double eps /* = 1e-9 */, bool display /* = true */)
	{

		if (!calib_estimate_k_int_rT_init(n_cams, n_pts, X3, X2, max_iter, intrinsic_para, rT, eps))
			return false;

		k[0] = k[1] = 0;

		for (int it = 0; it < alt_iter; it++)
		{
			if (!_estimate_k_with_int_rT(n_cams, n_pts, X3, X2, k, intrinsic_para, rT, eps))
			{
				return false;
			}
			
			if (!calib_estimate_int_rT_fix_k_with_init(n_cams, n_pts, X3, X2, max_iter, k, intrinsic_para, rT, avg_err_square, eps))
			{
				return false;
			}
			if (display)
			{
				printf("focal=[%8.1f %8.1f], center=[%8.1f %8.1f], k=[%8.2e %8.2e], avg_err_square = %8.2e\n",
					intrinsic_para[0], intrinsic_para[1], intrinsic_para[2], intrinsic_para[3], k[0], k[1], avg_err_square);
			}
		}

		return true;
	}


	/*pose estimation based on Lev-Mar optimization, need good initialization.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::_pose_estimate_no_distortion_func(const T* p, T* hx, int m, int n, const void* data)
	{
		const Calib_Data_Header<T>* ptr = (const Calib_Data_Header<T>*)data;
		double eps = ptr->eps;
		int N = ptr->n_pts;
		const T* X3 = ptr->X3;
		const T* X2 = ptr->X2;

		const T* intrinsic_para = ptr->intrinsic_para;

		T A[9] = {
			intrinsic_para[0],0,intrinsic_para[2],
			0,intrinsic_para[1],intrinsic_para[3],
			0,0,1
		};

		const T* r = p;
		const T* tt = p+3;

		T R[9];
		
		ZQ_Rodrigues::ZQ_Rodrigues_r2R(r, R);
		
		T* tmp_X2 = new T[N * 2];

		proj_no_distortion(N,A,R,tt,X3,tmp_X2,eps);

		for(int i = 0;i < N;i++)
		{
			hx[i*2+0] = tmp_X2[i*2+0] - X2[i*2+0];
			hx[i*2+1] = tmp_X2[i*2+1] - X2[i*2+1];
		}

		delete []tmp_X2;
		return true;
	}

	/*pose estimation based on Lev-Mar optimization, need good initialization.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::_pose_estimate_no_distortion_jac(const T* p, T* jx, int m, int n, const void* data)
	{
		const Calib_Data_Header<T>* ptr = (const Calib_Data_Header<T>*)data;
		const T* X3 = ptr->X3;
		const T* X2 = ptr->X2;
		const int N = ptr->n_pts;

		const double eps = ptr->eps;
		const T* intrinsic_para = ptr->intrinsic_para;

		memset(jx,0,sizeof(T)*m*n);

		//
		T A[9] = {
			intrinsic_para[0],0,intrinsic_para[2],
			0,intrinsic_para[1],intrinsic_para[3],
			0,0,1
		};

		
		T dRdr[27],R[9];
		const T *tt = p+3;
		ZQ_Rodrigues::ZQ_Rodrigues_r2R(p, R, dRdr);
	
		for(int i = 0;i < N;i++)
		{
			int cur_row0 = i*2+0;
			int cur_row1 = i*2+1;
			T cur_X3[3] = {X3[i*3+0],X3[i*3+1],X3[i*3+2]};

			//var1 = R*X3+T: dvar1dR,dvar1dT
			T var1[3] = {
				R[0]*cur_X3[0]+R[1]*cur_X3[1]+R[2]*cur_X3[2] + tt[0],
				R[3]*cur_X3[0]+R[4]*cur_X3[1]+R[5]*cur_X3[2] + tt[1],
				R[6]*cur_X3[0]+R[7]*cur_X3[1]+R[8]*cur_X3[2] + tt[2]
			};

			T dvar1dR[27] = {
				cur_X3[0],cur_X3[1],cur_X3[2],0,0,0,0,0,0,
				0,0,0,cur_X3[0],cur_X3[1],cur_X3[2],0,0,0,
				0,0,0,0,0,0,cur_X3[0],cur_X3[1],cur_X3[2]
			};
			T dvar1dT[9] = 
			{
				1,0,0,
				0,1,0,
				0,0,1
			};

			//
			T dvar1drT[18] = {0};
			for(int iii = 0;iii < 3;iii++)
			{
				for(int jjj = 0;jjj < 3;jjj++)
				{
					for(int kkk = 0;kkk < 9;kkk++)
						dvar1drT[iii*6+jjj] += dvar1dR[iii*9+kkk]*dRdr[kkk*3+jjj];
					dvar1drT[iii*6+jjj+3] = dvar1dT[iii*3+jjj];
				}
			}


			//var2 = A*var1: dvar2dA, dvar2dvar1
			T var2[3] = {
				A[0]*var1[0]+A[1]*var1[1]+A[2]*var1[2],
				A[3]*var1[0]+A[4]*var1[1]+A[5]*var1[2],
				A[6]*var1[0]+A[7]*var1[1]+A[8]*var1[2]
			};
			
			T dvar2dvar1[9] = {
				A[0],A[1],A[2],
				A[3],A[4],A[5],
				A[6],A[7],A[8]
			};
			T dvar2drT[18] = {0};
			ZQ_MathBase::MatrixMul(dvar2dvar1, dvar1drT, 3, 3, 6, dvar2drT);

			//X2 = [var2(1)/var2(3);var2(2)/var2(3)]
			T cur_X2[2] = {
				var2[0]*var2[2]/(var2[2]*var2[2]+eps*eps),
				var2[1]*var2[2]/(var2[2]*var2[2]+eps*eps)
			};

			T dX2dvar2[6] = 
			{
				var2[2]/(var2[2]*var2[2]+eps*eps),0,-var2[0]/(var2[2]*var2[2]+eps*eps),
				0,var2[2]/(var2[2]*var2[2]+eps*eps),-var2[1]/(var2[2]*var2[2]+eps*eps)
			};

			T dX2drT[12] = {0};
			
			ZQ_MathBase::MatrixMul(dX2dvar2, dvar2drT, 2, 3, 6, dX2drT);


			for(int iii = 0;iii < 6;iii++)
			{
				jx[cur_row0*m+iii] = dX2drT[iii];
				jx[cur_row1*m+iii] = dX2drT[6+iii];
			}
		}
		return true;
	}

	/*pose estimation based on Lev-Mar optimization, need good initialization.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_Calibration::pose_estimate_no_distortion_with_init(int n_pts, const T* X3, const T* X2, int max_iter, const T* intrinsic_para, T* rT, double& avg_err_square, double eps /* = 1e-9 */)
	{
		Calib_Data_Header<T> data;
		data.n_pts = n_pts;
		data.eps = eps;
		data.intrinsic_para = intrinsic_para;
		data.X3 = X3;
		data.X2 = X2;

		ZQ_LevMarOptions opts;
		ZQ_LevMarReturnInfos infos;
		opts.tol_e_square = 1e-16;
		opts.tol_max_jte = 1e-16;
		opts.tol_dx_square = 1e-16;

		T* hx = new T[n_pts*2];
		memset(hx,0,sizeof(T)*n_pts*2);

		if(!ZQ_LevMar::ZQ_LevMar_Der<T>(_pose_estimate_no_distortion_func<T>,_pose_estimate_no_distortion_jac<T>,rT,hx,6,n_pts*2,max_iter,opts,infos,&data))
		{
			delete []hx;
			return false;
		}
		avg_err_square = infos.final_e_square/n_pts;
		delete []hx;
		return true;
	}


	/*
	refer to the paper: 
	Model-Based Object Pose in 25 Lines of Codes. Daniel F. DeMenthon and Larry S. Davis. IJCV,1995.
	left hand coordinates.
	intrinsic_para[0-3]: fx, fy, u0, v0. with no distortion.
	rT[0-5]: rx, ry, rz, Tx, Ty, Tz.  (rx,ry,rz,rw) is a quaternion.
	*/
	template<class T>
	bool ZQ_Calibration::posit_no_coplanar(int n_pts, const T* X3, const T* X2, int max_iter, const T* intrinsic_para, T* rT)
	{
		
		double focal_len = intrinsic_para[0];
		if(focal_len == 0)
			return false;
		double scale_y = intrinsic_para[1]/focal_len;


		T* A = new T[(n_pts-1)*3];
		T* B = new T[3*(n_pts-1)];
		T* X = new T[n_pts*2];
		for(int i = 0;i < n_pts-1;i++)
		{
			A[i*3+0] = X3[(i+1)*3+0]-X3[0];
			A[i*3+1] = X3[(i+1)*3+1]-X3[1];
			A[i*3+2] = X3[(i+1)*3+2]-X3[2];
		}

	
		for(int i = 0;i < n_pts;i++)
		{
			X[i*2+0] = X2[i*2+0]-intrinsic_para[2];
			X[i*2+1] = (X2[i*2+1]-intrinsic_para[3])*scale_y;
		}

		ZQ_Matrix<double> Amat(n_pts-1,3),Bmat(3,n_pts-1);
		for(int i = 0;i < n_pts-1;i++)
		{
			for(int j = 0;j < 3;j++)
			{
				Amat.SetData(i,j,A[i*3+j]);
			}
		}

		ZQ_SVD::Invert(Amat,Bmat);
		for(int i = 0;i < n_pts-1;i++)
		{
			for(int j = 0;j < 3;j++)
			{
				bool flag;
				B[j*(n_pts-1)+i] = Bmat.GetData(j,i,flag);
			}
		}

		T* epsilon = new T[n_pts-1];
		T* xx = new T[n_pts-1];
		T* yy = new T[n_pts-1];

		memset(epsilon,0,sizeof(T)*(n_pts-1));

		double Z0 = 0;

		double last_ii[3],last_jj[3];
		double tolX = 1e-16;

		int it;
		for(it = 0;it < max_iter;it++)
		{
			for(int i = 0;i < n_pts-1;i++)
			{
				xx[i] = X[(i+1)*2+0]*(1+epsilon[i]) - X[0];
				yy[i] = X[(i+1)*2+1]*(1+epsilon[i]) - X[1];
			}

			double II[3] = {0},JJ[3] = {0};
			for(int i = 0;i < 3;i++)
			{
				for(int j = 0;j < n_pts-1;j++)
				{
					II[i] += B[i*(n_pts-1)+j]*xx[j];
					JJ[i] += B[i*(n_pts-1)+j]*yy[j];
				}
			}

			double lenII = sqrt(II[0]*II[0]+II[1]*II[1]+II[2]*II[2]);
			double lenJJ = sqrt(JJ[0]*JJ[0]+JJ[1]*JJ[1]+JJ[2]*JJ[2]);

			if(lenII == 0 || lenJJ == 0)
			{
				delete []A;
				delete []B;
				delete []X;
				delete []epsilon;
				delete []xx;
				delete []yy;
				return false;
			}
			double ii[3] = {II[0]/lenII,II[1]/lenII,II[2]/lenII};
			double jj[3] = {JJ[0]/lenJJ,JJ[1]/lenJJ,JJ[2]/lenJJ};
			
			
			if(it != 0)
			{
				double delta_ii = 0,delta_jj=0;
				for(int i = 0;i < 3;i++)
				{
					delta_ii += (last_ii[i]-ii[i])*(last_ii[i]-ii[i]);
					delta_jj += (last_jj[i]-jj[i])*(last_jj[i]-jj[i]);
				}
				if(delta_ii < tolX*tolX && delta_jj < tolX*tolX)
					break;
			}

			memcpy(last_ii,ii,sizeof(double)*3);
			memcpy(last_jj,jj,sizeof(double)*3);

			double s = 0.5*(lenII+lenJJ);
			Z0 = focal_len/s;
			double kk[3] = {
				ii[1]*jj[2]-ii[2]*jj[1],
				ii[2]*jj[0]-ii[0]*jj[2],
				ii[0]*jj[1]-ii[1]*jj[0]
			};
			for(int i = 0;i < n_pts-1;i++)
				epsilon[i] = (A[i*3+0]*kk[0]+A[i*3+1]*kk[1]+A[i*3+2]*kk[2])/Z0;


		}
		//printf("it = %d\n",it);

		
		double kk[3] = {
			last_ii[1]*last_jj[2]-last_ii[2]*last_jj[1],
			last_ii[2]*last_jj[0]-last_ii[0]*last_jj[2],
			last_ii[0]*last_jj[1]-last_ii[1]*last_jj[0]
		};
		
		double len_kk = sqrt(kk[0]*kk[0]+kk[1]*kk[1]+kk[2]*kk[2]);
		if(len_kk == 0)
		{
			delete []A;
			delete []B;
			delete []X;
			delete []epsilon;
			delete []xx;
			delete []yy;
			return false;
		}

		kk[0] /= len_kk;
		kk[1] /= len_kk;
		kk[2] /= len_kk;

		double jj[3] = 
		{
			kk[1]*last_ii[2]-kk[2]*last_ii[1],
			kk[2]*last_ii[0]-kk[0]*last_ii[2],
			kk[0]*last_ii[1]-kk[1]*last_ii[0]
		};

		double O[3] = {
			X3[0]-Z0/focal_len*(X[0]*last_ii[0] + X[1]*jj[0] + focal_len*kk[0]),
			X3[1]-Z0/focal_len*(X[0]*last_ii[1] + X[1]*jj[1] + focal_len*kk[1]),
			X3[2]-Z0/focal_len*(X[0]*last_ii[2] + X[1]*jj[2] + focal_len*kk[2])
		};
		ZQ_Matrix<double> Mmat(4,4),invM(4,4);
		for(int i = 0;i < 3;i++)
		{
			Mmat.SetData(i,0,last_ii[i]);
			Mmat.SetData(i,1,jj[i]);
			Mmat.SetData(i,2,kk[i]);
			Mmat.SetData(i,3,O[i]);
		}
		Mmat.SetData(3,3,1);

		ZQ_SVD::Invert(Mmat,invM);

		T R[9];
		T* tt = rT+3;
		for(int i = 0;i < 3;i++)
		{
			bool flag;
			for(int j = 0;j < 3;j++)
			{
				R[i*3+j] = invM.GetData(i,j,flag);
			}
			tt[i] = invM.GetData(i,3,flag);
		}
		ZQ_Rodrigues::ZQ_Rodrigues_R2r(R,rT);
		
		
		delete []A;
		delete []B;
		delete []X;
		delete []epsilon;
		delete []xx;
		delete []yy;
		return true;

	}

	

	/*
	refer to the paper:
	iterative pose estimation using coplanar feature points. Denis Oberkampf, Daniel F. DeMenthon, Larry  S. Davis. CVIU, 1995. 
	left hand coordinates.
	intrinsic_para[0-3]: fx, fy, u0, v0. with no distortion.
	rT[0-5]: rx, ry, rz, Tx, Ty, Tz.  (rx,ry,rz,rw) is a quaternion.
	*/
	template<class T>
	bool ZQ_Calibration::posit_coplanar(int n_pts, const T* X3, const T* X2, int max_iter, double tol_E, const T* intrinsic_para, T* rT, T* reproj_err_square, double eps /* = 1e-9 */)
	{
		double focal_len = intrinsic_para[0];
		if(focal_len == 0)
			return false;
		double scale_y = intrinsic_para[1]/focal_len;

		T int_A[9] = {
			intrinsic_para[0],0, intrinsic_para[2],
			0,intrinsic_para[1],intrinsic_para[3],
			0,0,1
		};

		double tol_Error = n_pts*tol_E*tol_E;


		T* A = new T[n_pts*3];
		T* B = new T[3*n_pts];
		T* X = new T[n_pts*2];
		for(int i = 0;i < n_pts;i++)
		{
			A[i*3+0] = X3[i*3+0]-X3[0];
			A[i*3+1] = X3[i*3+1]-X3[1];
			A[i*3+2] = X3[i*3+2]-X3[2];
		}


		for(int i = 0;i < n_pts;i++)
		{
			X[i*2+0] = X2[i*2+0]-intrinsic_para[2];
			X[i*2+1] = (X2[i*2+1]-intrinsic_para[3])*scale_y;
		}

		ZQ_Matrix<double> Amat(n_pts,3),Bmat(3,n_pts);
		for(int i = 0;i < n_pts;i++)
		{
			for(int j = 0;j < 3;j++)
			{
				Amat.SetData(i,j,A[i*3+j]);
			}
		}

		if(!ZQ_SVD::Invert(Amat,Bmat))
		{
			delete []A;
			delete []B;
			delete []X;
			return false;
		}
		for(int i = 0;i < n_pts;i++)
		{
			for(int j = 0;j < 3;j++)
			{
				bool flag;
				B[j*n_pts+i] = Bmat.GetData(j,i,flag);
			}
		}

		ZQ_Matrix<double> Umat(n_pts,3),Smat(3,3),Vmat(3,3);
		double center[3] = {0};
		for(int i = 0;i < n_pts;i++)
		{
			center[0] += X3[i*3+0];
			center[1] += X3[i*3+1];
			center[2] += X3[i*3+2];
		}
		center[0] /= n_pts;
		center[1] /= n_pts;
		center[2] /= n_pts;
		for(int i = 0;i < n_pts;i++)
		{
			for(int j = 0;j < 3;j++)
			{
				Amat.SetData(i,j,A[i*3+j] - center[j]);
			}
		}
		if(!ZQ_SVD::Decompose(Amat,Umat,Smat,Vmat))
		{
			delete []A;
			delete []B;
			delete []X;
			return false;
		}

		double u[3];
		for(int i = 0;i < 3;i++)
		{
			bool flag;
			u[i] = Vmat.GetData(i,2,flag);
		}
		if(u[2] < 0)
		{
			u[0] = -u[0];
			u[1] = -u[1];
			u[2] = -u[2];
		}
		

		std::vector<Posit_Coplanar_Node<T>> last_candidates;
		std::vector<Posit_Coplanar_Node<T>> cur_candidates;
		Posit_Coplanar_Node<T> node;
		last_candidates.push_back(node);
		int it = 0;
		double diverge_ratio = 1.1;
		int selection_thresh = 8;
		
		do{
			//printf("%d ",it);
			bool has_find_solution = false;
			cur_candidates.clear();
			for(int cc = 0;cc < last_candidates.size();cc++)
			{
				Posit_Coplanar_Node<T> cur_node = last_candidates[cc];
				Posit_Coplanar_Node<T> new_node[2];
				
				double I0[3] = {0}, J0[3] = {0};
				for(int i = 0;i < n_pts;i++)
				{
					double epsilon;
					if(it == 0)
						epsilon = 0;
					else
						epsilon = 1.0/cur_node.Z0*(A[i*3+0]*cur_node.kk[0]+A[i*3+1]*cur_node.kk[1]+A[i*3+2]*cur_node.kk[2]);
					double xx = X[i*2+0]*(1.0+epsilon) - X[0];
					double yy = X[i*2+1]*(1.0+epsilon) - X[1];
					I0[0] += B[0*n_pts+i]*xx;
					I0[1] += B[1*n_pts+i]*xx;
					I0[2] += B[2*n_pts+i]*xx;
					J0[0] += B[0*n_pts+i]*yy;
					J0[1] += B[1*n_pts+i]*yy;
					J0[2] += B[2*n_pts+i]*yy;
				}

				double I0J0 = I0[0]*J0[0] + I0[1]*J0[1] + I0[2]*J0[2];
				double J02_I02 = (J0[0]*J0[0]+J0[1]*J0[1]+J0[2]*J0[2]) - (I0[0]*I0[0]+I0[1]*I0[1]+I0[2]*I0[2]);

				/*lambda*mu = -I0J0
				lambda^2 - mu^2 = J02-I02
				*/

				double delta = J02_I02*J02_I02+4*I0J0*I0J0;
				double lambda[2],mu[2];

				if(1)
				{
					double q = 0;
					if(J02_I02 <= 0)
						q = 0.5*(J02_I02-sqrt(delta));
					else
						q = 0.5*(J02_I02+sqrt(delta));

					
					if(q >= 0)
					{
						lambda[0] = sqrt(q);
						lambda[1] = -sqrt(q);
						if (lambda[0] == 0.0) 
						{
							mu[0] = 0.0;
							mu[1] = 0.0;
						}
						else
						{
							mu[0] = -I0J0/lambda[0];
							mu[1] = -I0J0/lambda[1];
						}
					}
					else
					{
						lambda[0] = sqrt(-(I0J0*I0J0)/q);
						lambda[1] = -lambda[0];
						if(lambda[0] == 0)
						{
							mu[0] = sqrt(-J02_I02);
							mu[1] = -mu[0];
						}
						else
						{
							mu[0] = -I0J0/lambda[0];
							mu[1] = -mu[0];
						}
					}

				}
				else
				{
					lambda[0] = sqrt(0.5*(J02_I02+sqrt(delta)));
					lambda[1] = -lambda[0];
					if(lambda[0] != 0)
					{
						mu[0] = -I0J0/lambda[0];
						mu[1] = -mu[0];
					}
					else
					{
						mu[0] = mu[1] = 0;
					}
				}
				

				//////////
				T II[2][3] = {
					{I0[0]+lambda[0]*u[0], I0[1]+lambda[0]*u[1], I0[2]+lambda[0]*u[2]},
					{I0[0]+lambda[1]*u[0], I0[1]+lambda[1]*u[1], I0[2]+lambda[1]*u[2]}
				};
				T JJ[2][3] = {
					{J0[0]+mu[0]*u[0], J0[1]+mu[0]*u[1], J0[2]+mu[0]*u[2]},
					{J0[0]+mu[1]*u[0], J0[1]+mu[1]*u[1], J0[2]+mu[1]*u[2]}
				};

				T cur_R[2][9];
				T cur_tt[2][3];
				T cur_Z0[2];
				T cur_kk[2][3];
				T cur_E[2];
				bool discard_flag[2] = {false,false};
				for(int dd = 0;dd < 2;dd++)
				{
					double lenII = sqrt(II[dd][0]*II[dd][0]+II[dd][1]*II[dd][1]+II[dd][2]*II[dd][2]);
					double lenJJ = sqrt(JJ[dd][0]*JJ[dd][0]+JJ[dd][1]*JJ[dd][1]+JJ[dd][2]*JJ[dd][2]);
					if(lenII == 0 || lenJJ == 0)
					{
						delete []A;
						delete []B;
						delete []X;
						return false;
					}
					double ii[3] = {II[dd][0]/lenII,II[dd][1]/lenII,II[dd][2]/lenII};
					double jj[3] = {JJ[dd][0]/lenJJ,JJ[dd][1]/lenJJ,JJ[dd][2]/lenJJ};
					double s = 0.5*(lenII+lenJJ);
					cur_Z0[dd] = focal_len/s;
					cur_kk[dd][0] = ii[1]*jj[2]-ii[2]*jj[1];
					cur_kk[dd][1] = ii[2]*jj[0]-ii[0]*jj[2];
					cur_kk[dd][2] = ii[0]*jj[1]-ii[1]*jj[0];

					double len_kk = sqrt(cur_kk[dd][0]*cur_kk[dd][0]+cur_kk[dd][1]*cur_kk[dd][1]+cur_kk[dd][2]*cur_kk[dd][2]);
					if(len_kk == 0)
					{
						delete []A;
						delete []B;
						delete []X;
						return false;
					}

					cur_kk[dd][0] /= len_kk;
					cur_kk[dd][1] /= len_kk;
					cur_kk[dd][2] /= len_kk;
					jj[0] = cur_kk[dd][1]*ii[2]-cur_kk[dd][2]*ii[1];
					jj[1] = cur_kk[dd][2]*ii[0]-cur_kk[dd][0]*ii[2];
					jj[2] = cur_kk[dd][0]*ii[1]-cur_kk[dd][1]*ii[0];

					double O[3] = {
						X3[0]-cur_Z0[dd]/focal_len*(X[0]*ii[0] + X[1]*jj[0] + focal_len*cur_kk[dd][0]),
						X3[1]-cur_Z0[dd]/focal_len*(X[0]*ii[1] + X[1]*jj[1] + focal_len*cur_kk[dd][1]),
						X3[2]-cur_Z0[dd]/focal_len*(X[0]*ii[2] + X[1]*jj[2] + focal_len*cur_kk[dd][2])
					};
					ZQ_Matrix<double> Mmat(4,4),invM(4,4);
					for(int i = 0;i < 3;i++)
					{
						Mmat.SetData(i,0,ii[i]);
						Mmat.SetData(i,1,jj[i]);
						Mmat.SetData(i,2,cur_kk[dd][i]);
						Mmat.SetData(i,3,O[i]);
					}
					Mmat.SetData(3,3,1);

					ZQ_SVD::Invert(Mmat,invM);

					for(int i = 0;i < 3;i++)
					{
						bool flag;
						for(int j = 0;j < 3;j++)
						{
							cur_R[dd][i*3+j] = invM.GetData(i,j,flag);
						}
						cur_tt[dd][i] = invM.GetData(i,3,flag);
					}
					T* cur_X2 = new T[n_pts*2];
					proj_no_distortion(n_pts,int_A,cur_R[dd],cur_tt[dd],X3,cur_X2,eps);
					cur_E[dd] = 0;
					for(int i = 0;i < n_pts;i++)
					{
						cur_E[dd] += (cur_X2[i*2+0]-X2[i*2+0])*(cur_X2[i*2+0]-X2[i*2+0])+(cur_X2[i*2+1]-X2[i*2+1])*(cur_X2[i*2+1]-X2[i*2+1]);
					}
					delete []cur_X2;
					for(int i = 0;i < n_pts;i++)
					{

						double tmp_pts_Z = cur_R[dd][6]*X3[i*3+0]+cur_R[dd][7]*X3[i*3+1]+cur_R[dd][8]*X3[i*3+2] + cur_tt[dd][2];
						if(tmp_pts_Z < 0)
						{
							discard_flag[dd] = true;
							break;
						}
					}

					
					new_node[dd].Error = cur_E[dd];
					memcpy(new_node[dd].R,cur_R[dd],sizeof(T)*9);
					memcpy(new_node[dd].tt,cur_tt[dd],sizeof(T)*3);
					ZQ_Rodrigues::ZQ_Rodrigues_R2r_fun(cur_R[dd],new_node[dd].rr);
					memcpy(new_node[dd].kk,cur_kk[dd],sizeof(T)*3);
					new_node[dd].Z0 = cur_Z0[dd];

				}/* for dd*/
				

				if(it == 0)
				{
					cur_candidates.push_back(new_node[0]);
					cur_candidates.push_back(new_node[1]);
				}
				else
				{
					if(last_candidates.size() < selection_thresh)
					{
						if(!discard_flag[0]/* && (new_node[0].Error < cur_node.Error*diverge_ratio || new_node[0].Error < tol_Error)*/)
							cur_candidates.push_back(new_node[0]);
						if(!discard_flag[1]/* && (new_node[1].Error < cur_node.Error*diverge_ratio || new_node[1].Error < tol_Error)*/)
							cur_candidates.push_back(new_node[1]);

					}
					else
					{
						if(!discard_flag[0] && !discard_flag[1])
						{
							if(new_node[0].Error < new_node[1].Error)
								cur_candidates.push_back(new_node[0]);
							else
								cur_candidates.push_back(new_node[1]);
						}
						else
						{

							if(!discard_flag[0])
								cur_candidates.push_back(new_node[0]);
							if(!discard_flag[1])
								cur_candidates.push_back(new_node[1]);
						}
					}

					if(new_node[0].Error < tol_Error || new_node[1].Error < tol_Error)
						has_find_solution = true;
				}

			}/*for cc*/

			
			if(cur_candidates.size() == 0)
				break;
			last_candidates = cur_candidates;
			//printf("%d ",last_candidates.size());
			if(has_find_solution)
				break;

			it++;

		}while(it <= max_iter );
		
		//printf("\n");

		//printf("it = %d\n",it);
		if(last_candidates.size() == 0)
		{
			delete []A;
			delete []B;
			delete []X;
			return false;
		}

		double min_error = last_candidates[0].Error;
		int best_idx = 0;
		for(int cc = 1;cc < last_candidates.size();cc++)
		{
			if(last_candidates[cc].Error < min_error)
			{
				best_idx = cc;
				min_error = last_candidates[cc].Error;
			}
		}

		ZQ_Rodrigues::ZQ_Rodrigues_R2r_fun(last_candidates[best_idx].R,rT);
		memcpy(rT+3,last_candidates[best_idx].tt,sizeof(T)*3);

		if(reproj_err_square != 0)
		{
			T* reproj_X2 = new T[n_pts*2];
			proj_no_distortion(n_pts,int_A,last_candidates[best_idx].R,last_candidates[best_idx].tt,X3,reproj_X2,eps);
			for(int i = 0;i < n_pts;i++)
			{
				reproj_err_square[i] = (double)(reproj_X2[i*2+0]-X2[i*2+0])*(reproj_X2[i*2+0]-X2[i*2+0])+(reproj_X2[i*2+1]-X2[i*2+1])*(reproj_X2[i*2+1]-X2[i*2+1]);
			}
			delete []reproj_X2;

		}
		

		delete []A;
		delete []B;
		delete []X;

		return true;
	}


	/*
	left hand coordinates.
	The base idea is to use the method proposed in the paper:
	iterative pose estimation using coplanar points. Denis Oberkampf, Daniel F. DeMenthon, Larry  S. Davis. CVPR, 1993. 
	However, I find it cannot make sure the method always converge to the optimal solution.
	But the translation seems to be near the optimal one according to my observations.
	So I choose 9 rotations to run Lev-Mar solvers to find a best solution.
	If all choices do not give a solution with avg_E < tol_avg_E, the best solution of the 9 will be returned,
	otherwise, the first one satisfying avg_E < tol_avg_E will be returned.
	*/
	template<class T>
	bool ZQ_Calibration::posit_coplanar_robust(int n_pts, const T* X3, const T* X2, int max_iter_posit, int max_iter_levmar, double tol_E, const T* intrinsic_para, T* rT, double& avg_E, double eps /* = 1e-9 */)
	{
		T* reproj_err_square = new T[n_pts];
		memset(reproj_err_square,0,sizeof(T)*n_pts);
		
		T init_rT[6];
		memcpy(init_rT,rT,sizeof(T)*6);
		if(!ZQ::ZQ_Calibration::posit_coplanar(n_pts,X3,X2,max_iter_posit,tol_E,intrinsic_para,init_rT,reproj_err_square,eps))
		{
			delete []reproj_err_square;
			return false;
		}
		T tol_E_square = tol_E*tol_E;
		T avg_E_square = 0;
		for(int i = 0;i < n_pts;i++)
			avg_E_square += reproj_err_square[i]/n_pts;
		delete []reproj_err_square;

		if(avg_E_square <= tol_E_square)
		{	
			memcpy(rT,init_rT,sizeof(T)*6);
			avg_E = sqrt(avg_E_square);
			return true;
		}
		


		T rand_rr[10][3] = 
		{
			{init_rT[0],init_rT[1],init_rT[2]},
			{0,0,0},
			{0.2,0.1,0.1},
			{0.2,0.1,-0.1},
			{0.2,-0.1,0.1},
			{0.2,-0.1,0.1},
			{-0.2,0.1,0.1},
			{-0.2,0.1,-0.1},
			{-0.2,-0.1,0.1},
			{-0.2,-0.1,-0.1}
		};

		T tmp_rT[6];
		
		for(int i = 0;i < 10;i++)
		{
			memcpy(tmp_rT,rand_rr[i],sizeof(T)*3);
			memcpy(tmp_rT+3,init_rT+3,sizeof(T)*3);
			double cur_avg_E_square = 0;
			if(!pose_estimate_no_distortion_with_init(n_pts,X3,X2,max_iter_levmar,intrinsic_para,tmp_rT,cur_avg_E_square,eps))
			{
				continue;
			}
			else
			{
				if(cur_avg_E_square < avg_E_square)
				{
					avg_E_square = cur_avg_E_square;
					memcpy(rT,tmp_rT,sizeof(T)*6);
				}
				if(avg_E_square <= tol_E_square)
					break;
			}
		}
	
		avg_E = sqrt(avg_E_square);
		return true;
	}

	/*Calibrate binocular camera.
	The intrinsic parameters are known.
	X3 is a series of checkboard coordinates.
	The exterior parameters are estimated by using "posit_coplanar_robust".
	*/
	template<class T>
	bool ZQ_Calibration::_binocalib_with_known_intrinsic_init(int n_views, int n_pts, const T* X3, const T* left_X2, const T* right_X2, const T* left_intrinsic_para, const T* right_intrinsic_para,
		int max_iter_posit, int max_iter_levmar, double tol_E, T* right_rT, T* right_to_left_rT, double& avg_err_square, double eps /* = 1e-9 */)
	{
		if (n_views <= 0)
			return false;

		T* left_rT = new T[n_views * 6];
		double* left_avg_E = new double[n_views];
		for (int i = 0; i < n_views; i++)
		{
			if (!posit_coplanar_robust(n_pts, X3, left_X2 + i*n_pts * 2, max_iter_posit, max_iter_levmar, tol_E, left_intrinsic_para, left_rT + i * 6, left_avg_E[i], eps))
			{
				delete[]left_rT;
				delete[]left_avg_E;
				return false;
			}
			printf("left_avg_E[%d] = %f\n", i, left_avg_E[i]);
			printf("left_rT[%d] = (%f, %f, %f, %f, %f, %f\n", i, left_rT[i * 6 + 0], left_rT[i * 6 + 1], left_rT[i * 6 + 2], left_rT[i * 6 + 3], left_rT[i * 6 + 4], left_rT[i * 6 + 5]);
		}

		//T* right_rT = new T[n_views * 6];
		double* right_avg_E = new double[n_views];
		for (int i = 0; i < n_views; i++)
		{
			if (!posit_coplanar_robust(n_pts, X3, right_X2 + i*n_pts * 2, max_iter_posit, max_iter_levmar, tol_E, right_intrinsic_para, right_rT + i * 6, right_avg_E[i], eps))
			{
				delete[]left_rT;
				delete[]left_avg_E;
				//delete[]right_rT;
				delete[]right_avg_E;
				return false;
			}
		}

		int best_id = 0;
		double best_avg_E = left_avg_E[0] * left_avg_E[0] + right_avg_E[0] * right_avg_E[0];
		for (int i = 1; i < n_views; i++)
		{
			double cur_avg_E = left_avg_E[i] * left_avg_E[i] + right_avg_E[i] * right_avg_E[i];
			if (cur_avg_E < best_avg_E)
			{
				best_avg_E = cur_avg_E;
				best_id = i;
			}
		}


		T* left_r = left_rT + best_id * 6;
		T* left_T = left_rT + best_id * 6 + 3;
		T* right_r = right_rT + best_id * 6;
		T* right_T = right_rT + best_id * 6 + 3;
		T left_R[9], right_R[9];
		if (!ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun(left_r, left_R)
			|| !ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun(right_r, right_R))
		{
			delete[]left_rT;
			delete[]left_avg_E;
			//delete[]right_rT;
			delete[]right_avg_E;
			return false;
		}

		ZQ_Matrix<T> mat_X3_to_left(4, 4), mat_X3_to_right(4, 4);
		for (int hh = 0; hh < 3; hh++)
		{
			for (int ww = 0; ww < 3; ww++)
			{
				mat_X3_to_left.SetData(hh, ww, left_R[hh * 3 + ww]);
				mat_X3_to_right.SetData(hh, ww, right_R[hh * 3 + ww]);
				mat_X3_to_left.SetData(3, ww, 0);
				mat_X3_to_right.SetData(3, ww, 0);
			}
			mat_X3_to_left.SetData(hh, 3, left_T[hh]);
			mat_X3_to_right.SetData(hh, 3, right_T[hh]);
		}
		mat_X3_to_left.SetData(3, 3, 1);
		mat_X3_to_right.SetData(3, 3, 1);

		ZQ_Matrix<T> mat_right_to_X3(4, 4);
		ZQ_SVD::Invert(mat_X3_to_right, mat_right_to_X3);

		ZQ_Matrix<T> mat_right_to_left = mat_X3_to_left * mat_right_to_X3;

		T right_to_left_R[9], right_to_left_T[3];
		for (int hh = 0; hh < 3; hh++)
		{
			bool flag;
			for (int ww = 0; ww < 3; ww++)
			{
				right_to_left_R[hh * 3 + ww] = mat_right_to_left.GetData(hh, ww, flag);

			}
			right_to_left_T[hh] = mat_right_to_left.GetData(hh, 3, flag);
		}

		memcpy(right_to_left_rT + 3, right_to_left_T, sizeof(T)* 3);
		if (!ZQ_Rodrigues::ZQ_Rodrigues_R2r_fun(right_to_left_R, right_to_left_rT))
		{
			delete[]left_rT;
			delete[]left_avg_E;
			//delete[]right_rT;
			delete[]right_avg_E;
			return false;
		}
		//memcpy(rT, right_rT, sizeof(T)*n_views * 6);

		delete[]left_rT;
		delete[]left_avg_E;
		//delete[]right_rT;
		delete[]right_avg_E;
		return true;
	}

	template<class T>
	bool ZQ_Calibration::_binocalib_with_known_intrinsic_func(const T* p, T* hx, int m, int n, const void* data)
	{
		const Binocalib_Data_Header<T>* ptr = (const Binocalib_Data_Header<T>*)data;
		double eps = ptr->eps;
		int N = ptr->n_pts;
		int n_views = ptr->n_views;
		const T* X3 = ptr->X3;
		const T* left_X2 = ptr->left_X2;
		const T* right_X2 = ptr->right_X2;

		const T* left_intrinsic_para = ptr->left_intrinsic_para;
		const T* right_intrinsic_para = ptr->right_intrinsic_para;

		T left_A[9] = {
			left_intrinsic_para[0], 0, left_intrinsic_para[2], 
			0, left_intrinsic_para[1], left_intrinsic_para[3],
			0, 0, 1
		};
		T right_A[9] = {
			right_intrinsic_para[0], 0, right_intrinsic_para[2],
			0, right_intrinsic_para[1], right_intrinsic_para[3],
			0, 0, 1
		};

		const T* right_to_left_r = p + n_views * 6;
		const T* right_to_left_T = p + n_views * 6 + 3;
		const T* right_rT = p;
		
		T right_to_left_R[9];
		
		if (!ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun(right_to_left_r, right_to_left_R))
		{
			//printf("error: Rodrigues r to R failed:(%d)%s\n",__LINE__,__FILE__);
			return false;
		}

		
		T* tmp_left_X2 = new T[N * 2 * n_views];
		
		for (int cc = 0; cc < n_views; cc++)
		{
			T right_R[9];
			const T* right_r = right_rT + cc * 6;
			const T* right_T = right_rT + cc * 6 + 3;
			if (!ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun(right_r, right_R))
			{
				delete[]tmp_left_X2;
				return false;
			}
			T left_R[9], left_T[3];
			ZQ_MathBase::MatrixMul(right_to_left_R, right_R, 3, 3, 3, left_R);
			ZQ_MathBase::MatrixMul(right_to_left_R, right_T, 3, 3, 1, left_T);
			left_T[0] += right_to_left_T[0];
			left_T[1] += right_to_left_T[1];
			left_T[2] += right_to_left_T[2];

			proj_no_distortion(N, left_A, left_R, left_T, X3, tmp_left_X2 + cc*N * 2, eps);

			int offset = cc*N * 2;
			for (int i = 0; i < N; i++)
			{
				hx[offset + i * 2 + 0] = tmp_left_X2[offset + i * 2 + 0] - left_X2[offset + i * 2 + 0];
				hx[offset + i * 2 + 1] = tmp_left_X2[offset + i * 2 + 1] - left_X2[offset + i * 2 + 1];
			}
		}
		
		delete[]tmp_left_X2;

		T* tmp_right_X2 = new T[N * 2 * n_views];

		for (int cc = 0; cc < n_views; cc++)
		{
			T right_R[9];
			const T* right_r = right_rT + cc * 6;
			const T* right_T = right_rT + cc * 6 + 3;
			if (!ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun(right_r, right_R))
			{
				delete[]tmp_right_X2;
				return false;
			}

			proj_no_distortion(N, right_A, right_R, right_T, X3, tmp_right_X2 + cc*N * 2, eps);

			int offset = cc*N * 2;
			int all_offset = (n_views + cc)*N * 2;
			for (int i = 0; i < N; i++)
			{
				hx[all_offset + i * 2 + 0] = tmp_right_X2[offset + i * 2 + 0] - right_X2[offset + i * 2 + 0];
				hx[all_offset + i * 2 + 1] = tmp_right_X2[offset + i * 2 + 1] - right_X2[offset + i * 2 + 1];
			}
		}

		delete[]tmp_right_X2;
		return true;
	}

	template<class T>
	bool ZQ_Calibration::_binocalib_with_known_intrinsic_jac(const T* p, T* jx, int m, int n, const void* data)
	{
		const Binocalib_Data_Header<T>* ptr = (const Binocalib_Data_Header<T>*)data;
		double eps = ptr->eps;
		int N = ptr->n_pts;
		int n_views = ptr->n_views;
		const T* X3 = ptr->X3;
		const T* left_X2 = ptr->left_X2;
		const T* right_X2 = ptr->right_X2;

		const T* left_intrinsic_para = ptr->left_intrinsic_para;
		const T* right_intrinsic_para = ptr->right_intrinsic_para;

		memset(jx, 0, sizeof(T)*m*n);

		//
		T left_A[9] = {
			left_intrinsic_para[0], 0, left_intrinsic_para[2],
			0, left_intrinsic_para[1], left_intrinsic_para[3],
			0, 0, 1
		};
		T right_A[9] = {
			right_intrinsic_para[0], 0, right_intrinsic_para[2],
			0, right_intrinsic_para[1], right_intrinsic_para[3],
			0, 0, 1
		};

		const T* right_to_left_r = p + n_views * 6;
		const T* right_to_left_T = p + n_views * 6 + 3;
		const T* right_rT = p;

		const T* T_r2l = right_to_left_T;
		const T* r_r2l = right_to_left_r;
		
		T R_r2l[9];
		T dR_r2l_dr_r2l[27];
		if (!ZQ_Rodrigues::ZQ_Rodrigues_r2R_jac(r_r2l, dR_r2l_dr_r2l))
		{
			//printf("error: Rodrigues r to R failed:(%d)%s\n",__LINE__,__FILE__);
			return false;
		}
		if (!ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun(r_r2l, R_r2l))
		{
			//printf("error: Rodrigues r to R failed:(%d)%s\n",__LINE__,__FILE__);
			return false;
		}

		// handle left camera
		for (int cc = 0; cc < n_views; cc++)
		{
			T dR_r_dr_r[27], R_r[9];
			const T* T_r = p + cc * 6 + 3;
			const T* r_r = p + cc * 6;
			if (!ZQ_Rodrigues::ZQ_Rodrigues_r2R_jac(r_r, dR_r_dr_r))
			{
				//printf("error: Rodrigues r to R failed:(%d)%s\n",__LINE__,__FILE__);
				return false;
			}
			if (!ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun(r_r, R_r))
			{
				//printf("error: Rodrigues r to R failed:(%d)%s\n",__LINE__,__FILE__);
				return false;
			}
			for (int i = 0; i < N; i++)
			{
				int cur_row0 = cc*N * 2 + i * 2 + 0;
				int cur_row1 = cc*N * 2 + i * 2 + 1;
				T cur_X3[3] = { X3[i * 3 + 0], X3[i * 3 + 1], X3[i * 3 + 2] };

				//X3_l = R_r*X3+T_r: dX3_l_dR_r,dX3_l_dT_r
				T X3_l[3] = {
					R_r[0] * cur_X3[0] + R_r[1] * cur_X3[1] + R_r[2] * cur_X3[2] + T_r[0],
					R_r[3] * cur_X3[0] + R_r[4] * cur_X3[1] + R_r[5] * cur_X3[2] + T_r[1],
					R_r[6] * cur_X3[0] + R_r[7] * cur_X3[1] + R_r[8] * cur_X3[2] + T_r[2]
				};

				T dX3_l_dR_r[27] = {
					cur_X3[0], cur_X3[1], cur_X3[2], 0, 0, 0, 0, 0, 0,
					0, 0, 0, cur_X3[0], cur_X3[1], cur_X3[2], 0, 0, 0,
					0, 0, 0, 0, 0, 0, cur_X3[0], cur_X3[1], cur_X3[2]
				};
				T dX3_l_dT_r[9] =
				{
					1, 0, 0,
					0, 1, 0,
					0, 0, 1
				};

				//
				T dX3_l_drT_r[18] = { 0 };
				for (int iii = 0; iii < 3; iii++)
				{
					for (int jjj = 0; jjj < 3; jjj++)
					{
						for (int kkk = 0; kkk < 9; kkk++)
							dX3_l_drT_r[iii * 6 + jjj] += dX3_l_dR_r[iii * 9 + kkk] * dR_r_dr_r[kkk * 3 + jjj];
						dX3_l_drT_r[iii * 6 + jjj + 3] = dX3_l_dT_r[iii * 3 + jjj];
					}
				}

				/***************************/
				//var1 = R_r2l*X3_l+T_r2l: dvar1_dR_r2l,dvar1_dT_r2l
				T var1[3] = {
					R_r2l[0] * X3_l[0] + R_r2l[1] * X3_l[1] + R_r2l[2] * X3_l[2] + T_r2l[0],
					R_r2l[3] * X3_l[0] + R_r2l[4] * X3_l[1] + R_r2l[5] * X3_l[2] + T_r2l[1],
					R_r2l[6] * X3_l[0] + R_r2l[7] * X3_l[1] + R_r2l[8] * X3_l[2] + T_r2l[2]
				};

				T dvar1_dR_r2l[27] = {
					X3_l[0], X3_l[1], X3_l[2], 0, 0, 0, 0, 0, 0,
					0, 0, 0, X3_l[0], X3_l[1], X3_l[2], 0, 0, 0,
					0, 0, 0, 0, 0, 0, X3_l[0], X3_l[1], X3_l[2]
				};
				T dvar1_dT_r2l[9] =
				{
					1, 0, 0,
					0, 1, 0,
					0, 0, 1
				};

				//
				T dvar1_drT_r2l[18] = { 0 };
				for (int iii = 0; iii < 3; iii++)
				{
					for (int jjj = 0; jjj < 3; jjj++)
					{
						for (int kkk = 0; kkk < 9; kkk++)
							dvar1_drT_r2l[iii * 6 + jjj] += dvar1_dR_r2l[iii * 9 + kkk] * dR_r2l_dr_r2l[kkk * 3 + jjj];
						dvar1_drT_r2l[iii * 6 + jjj + 3] = dvar1_dT_r2l[iii * 3 + jjj];
					}
				}
				
				//
				const T* dvar1_dX3_l = R_r2l;
				T dvar1_drT_r[18] = { 0 };
				ZQ_MathBase::MatrixMul(dvar1_dX3_l, dX3_l_drT_r, 3, 3, 6, dvar1_drT_r);

				/**********************************/
				//var2 = A*var1:  dvar2dvar1
				const T* A = left_A;
				T var2[3] = {
					A[0] * var1[0] + A[1] * var1[1] + A[2] * var1[2],
					A[3] * var1[0] + A[4] * var1[1] + A[5] * var1[2],
					A[6] * var1[0] + A[7] * var1[1] + A[8] * var1[2]
				};

				T dvar2_dvar1[9] = {
					A[0], A[1], A[2],
					A[3], A[4], A[5],
					A[6], A[7], A[8]
				};
				T dvar2_drT_r[18] = { 0 };
				T dvar2_drT_r2l[18] = { 0 };
				ZQ_MathBase::MatrixMul(dvar2_dvar1, dvar1_drT_r, 3, 3, 6, dvar1_drT_r);
				ZQ_MathBase::MatrixMul(dvar2_dvar1, dvar1_drT_r2l, 3, 3, 6, dvar2_drT_r2l);
				
				/******************/
				//X2 = [var2(1)/var2(3);var2(2)/var2(3)]
				T cur_X2[2] = {
					var2[0] * var2[2] / (var2[2] * var2[2] + eps*eps),
					var2[1] * var2[2] / (var2[2] * var2[2] + eps*eps)
				};

				T dX2dvar2[6] =
				{
					var2[2] / (var2[2] * var2[2] + eps*eps), 0, -var2[0] / (var2[2] * var2[2] + eps*eps),
					0, var2[2] / (var2[2] * var2[2] + eps*eps), -var2[1] / (var2[2] * var2[2] + eps*eps)
				};

				T dX2drT_r[12] = { 0 };
				T dX2drT_r2l[12] = { 0 };
				ZQ_MathBase::MatrixMul(dX2dvar2, dvar2_drT_r, 2, 3, 3, dX2drT_r);
				ZQ_MathBase::MatrixMul(dX2dvar2, dvar2_drT_r2l, 2, 3, 3, dX2drT_r2l);

				/***************/
				int offset_rT_r = cc*6;
				int offset_rT_r2l = n_views * 6;
				for (int iii = 0; iii < 6; iii++)
				{
					jx[cur_row0*m + offset_rT_r + iii] = dX2drT_r[iii];
					jx[cur_row1*m + offset_rT_r + iii] = dX2drT_r[6 + iii];

					jx[cur_row0*m + offset_rT_r2l + iii] = dX2drT_r2l[iii];
					jx[cur_row1*m + offset_rT_r2l + iii] = dX2drT_r2l[6 + iii];
				}
			}
		}
		

		//handle right camera
		/****************************************/
		for (int cc = 0; cc < n_views; cc++)
		{
			T dR_r_dr_r[27], R_r[9];
			const T* T_r = p + cc * 6 + 3;
			const T* r_r = p + cc * 6;
			if (!ZQ_Rodrigues::ZQ_Rodrigues_r2R_jac(r_r, dR_r_dr_r))
			{
				//printf("error: Rodrigues r to R failed:(%d)%s\n",__LINE__,__FILE__);
				return false;
			}
			if (!ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun(r_r, R_r))
			{
				//printf("error: Rodrigues r to R failed:(%d)%s\n",__LINE__,__FILE__);
				return false;
			}
			for (int i = 0; i < N; i++)
			{
				int cur_row0 = (n_views + cc)*N * 2 + i * 2 + 0;
				int cur_row1 = (n_views + cc)*N * 2 + i * 2 + 1;
				T cur_X3[3] = { X3[i * 3 + 0], X3[i * 3 + 1], X3[i * 3 + 2] };

				/***************************/
				//var1 = R_r*X3+T_r: dvar1_dR_r,dvar1_dT_r
				T var1[3] = {
					R_r[0] * cur_X3[0] + R_r[1] * cur_X3[1] + R_r[2] * cur_X3[2] + T_r[0],
					R_r[3] * cur_X3[0] + R_r[4] * cur_X3[1] + R_r[5] * cur_X3[2] + T_r[1],
					R_r[6] * cur_X3[0] + R_r[7] * cur_X3[1] + R_r[8] * cur_X3[2] + T_r[2]
				};

				T dvar1_dR_r[27] = {
					cur_X3[0], cur_X3[1], cur_X3[2], 0, 0, 0, 0, 0, 0,
					0, 0, 0, cur_X3[0], cur_X3[1], cur_X3[2], 0, 0, 0,
					0, 0, 0, 0, 0, 0, cur_X3[0], cur_X3[1], cur_X3[2]
				};
				T dvar1_dT_r[9] =
				{
					1, 0, 0,
					0, 1, 0,
					0, 0, 1
				};

				//
				T dvar1_drT_r[18] = { 0 };
				for (int iii = 0; iii < 3; iii++)
				{
					for (int jjj = 0; jjj < 3; jjj++)
					{
						for (int kkk = 0; kkk < 9; kkk++)
							dvar1_drT_r[iii * 6 + jjj] += dvar1_dR_r[iii * 9 + kkk] * dR_r_dr_r[kkk * 3 + jjj];
						dvar1_drT_r[iii * 6 + jjj + 3] = dvar1_dT_r[iii * 3 + jjj];
					}
				}

				/**********************************/
				//var2 = A*var1:  dvar2dvar1
				const T* A = right_A;
				T var2[3] = {
					A[0] * var1[0] + A[1] * var1[1] + A[2] * var1[2],
					A[3] * var1[0] + A[4] * var1[1] + A[5] * var1[2],
					A[6] * var1[0] + A[7] * var1[1] + A[8] * var1[2]
				};

				T dvar2_dvar1[9] = {
					A[0], A[1], A[2],
					A[3], A[4], A[5],
					A[6], A[7], A[8]
				};
				T dvar2_drT_r[18] = { 0 };
				ZQ_MathBase::MatrixMul(dvar2_dvar1, dvar1_drT_r, 3, 3, 6, dvar1_drT_r);

				/******************/
				//X2 = [var2(1)/var2(3);var2(2)/var2(3)]
				T cur_X2[2] = {
					var2[0] * var2[2] / (var2[2] * var2[2] + eps*eps),
					var2[1] * var2[2] / (var2[2] * var2[2] + eps*eps)
				};

				T dX2dvar2[6] =
				{
					var2[2] / (var2[2] * var2[2] + eps*eps), 0, -var2[0] / (var2[2] * var2[2] + eps*eps),
					0, var2[2] / (var2[2] * var2[2] + eps*eps), -var2[1] / (var2[2] * var2[2] + eps*eps)
				};

				T dX2drT_r[12] = { 0 };
				ZQ_MathBase::MatrixMul(dX2dvar2, dvar2_drT_r, 2, 3, 6, dX2drT_r);

				/***************/
				int offset_rT_r = cc * 6;
				for (int iii = 0; iii < 6; iii++)
				{
					jx[cur_row0*m + offset_rT_r + iii] = dX2drT_r[iii];
					jx[cur_row1*m + offset_rT_r + iii] = dX2drT_r[6 + iii];
				}
			}
		}
		return true;
	}

	/*Calibrate binocular camera.
	The intrinsic parameters are known.
	X3 is a series of checkboard coordinates.
	The exterior parameters are estimated by using "posit_coplanar_robust".
	*/
	template<class T>
	bool ZQ_Calibration::binocalib_with_known_intrinsic_with_init(int n_views, int n_pts, const T* X3, const T* left_X2, const T* right_X2, const T* left_intrinsic_para, const T* right_intrinsic_para,
		int max_iter_levmar, T* right_rT, T* right_to_left_rT, double& avg_err_square, double eps /* = 1e-9 */)
	{
		Binocalib_Data_Header<T> data;
		data.n_pts = n_pts;
		data.n_views = n_views;
		data.eps = eps;
		data.left_intrinsic_para = left_intrinsic_para;
		data.right_intrinsic_para = right_intrinsic_para;
		data.X3 = X3;
		data.left_X2 = left_X2;
		data.right_X2 = right_X2;

		ZQ_LevMarOptions opts;
		ZQ_LevMarReturnInfos infos;
		opts.tol_e_square = 1e-45;
		opts.tol_max_jte = 1e-45;
		opts.tol_dx_square = 1e-45;

		T* hx = new T[n_views * 2 * n_pts * 2];
		memset(hx, 0, sizeof(T)*n_views * 2 * n_pts * 2);

		T* unknowns = new T[n_views * 6 + 6];
		memcpy(unknowns, right_rT, sizeof(T)*n_views * 6);
		memcpy(unknowns + n_views * 6, right_to_left_rT, sizeof(T)* 6);
		if (!ZQ_LevMar::ZQ_LevMar_Der<T>(_binocalib_with_known_intrinsic_func<T>, _binocalib_with_known_intrinsic_jac<T>, unknowns, hx,
			n_views*6+6, n_views*2*n_pts*2, max_iter_levmar, opts, infos, &data))
		{
			delete[]hx;
			delete[]unknowns;
			return false;
		}
		memcpy(right_rT, unknowns, sizeof(T)*n_views * 6);
		memcpy(right_to_left_rT, unknowns + n_views * 6, sizeof(T)* 6);
		avg_err_square = infos.final_e_square / (n_pts * n_views * 2);

		delete[]hx;
		delete[]unknowns;
		return true;
	}


	/*Calibrate binocular camera.
	The intrinsic parameters are known.
	X3 is a series of checkboard coordinates.
	The exterior parameters are estimated by using "posit_coplanar_robust".
	*/
	template<class T>
	bool ZQ_Calibration::binocalib_with_known_intrinsic(int n_views, int n_pts, const T* X3, const T* left_X2, const T* right_X2, const T* left_intrinsic_para, const T* right_intrinsic_para,
		int max_iter_posit, int max_iter_levmar, double tol_E, T* right_rT, T* right_to_left_rT, double& avg_err_square, double eps /* = 1e-9 */)
	{
		if (n_views <= 0)
			return false;

		T* tmp_right_rT;
		if (right_rT == 0)
			tmp_right_rT = new T[n_views * 6];
		else
			tmp_right_rT = right_rT;

		if (!_binocalib_with_known_intrinsic_init(n_views, n_pts, X3, left_X2, right_X2, left_intrinsic_para, right_intrinsic_para, max_iter_posit, max_iter_levmar, tol_E, right_rT, right_to_left_rT, eps))
		{
			if (right_rT == 0)
				delete[] tmp_right_rT;
			return false;
		}

		if (!binocalib_with_known_intrinsic_with_init(n_views, n_pts, X3, left_X2, right_X2, left_intrinsic_para, right_intrinsic_para, max_iter_levmar, right_rT, right_to_left_rT, avg_err_square, eps))
		{
			if (right_rT == 0)
				delete[] tmp_right_rT;
			return false;
		}

		if (right_rT == 0)
			delete[]tmp_right_rT;
		return true;
	}

	/*Calibrate multi-camera.
	The intrinsic parameters are known.
	X3 is a series of checkboard coordinates.
	The exterior parameters are estimated by using "posit_coplanar_robust".
	visible_num, visible_offset,visible_idx and X2 must be valid.
	*/
	template<class T>
	bool ZQ_Calibration::_multicalib_with_known_intrinsic_init(int n_checkboards, int n_cams, int n_pts, const T* X3, const T* X2, const int* visible_num, const int* visible_offset, const int* visible_idx,
		const T* intrinsic_para, int max_iter_posit, int max_iter_levmar, double tol_E, T* checkboard_rT_to_cam0, T* cam0_to_othercam_rT, double eps /*= 1e-9*/)
	{
		bool* visible_map = new bool[n_cams*n_checkboards];
		int* offset_map = new int[n_cams*n_checkboards];
		memset(visible_map, 0, sizeof(bool)*n_cams*n_checkboards);
		memset(offset_map, 0, sizeof(int)*n_cams*n_checkboards);
		for (int i = 0; i < n_cams; i++)
		{
			int cur_visible_offset = visible_offset[i];
			int cur_visible_num = visible_num[i];
			for (int j = 0; j < cur_visible_num; j++)
			{
				int cur_visible_idx = visible_idx[cur_visible_offset+j];
				visible_map[i*n_checkboards + cur_visible_idx] = true;
				offset_map[i*n_checkboards + cur_visible_idx] = cur_visible_offset + j;
			}
		}
		
		T* checkboard_to_cam_rT_map = new T[n_checkboards*n_cams * 6];
		double* checkboard_to_cam_err_map = new double[n_checkboards*n_cams];
		memset(checkboard_to_cam_rT_map, 0, sizeof(T)*n_checkboards*n_cams * 6);
		memset(checkboard_to_cam_err_map, 0, sizeof(double)*n_checkboards*n_cams);
		for (int i = 0; i < n_cams; i++)
		{
			for (int j = 0; j < n_checkboards; j++)
			{
				if (visible_map[i*n_checkboards + j])
				{
					int cur_offset = offset_map[i*n_checkboards + j];
					if (!posit_coplanar_robust(n_pts, X3, X2 + cur_offset*n_pts * 2, max_iter_posit, max_iter_levmar, tol_E, intrinsic_para + i * 4, 
						checkboard_to_cam_rT_map + (j*n_cams + i) * 6, checkboard_to_cam_err_map[j*n_cams + i], eps))
					{
						delete[]checkboard_to_cam_err_map;
						delete[]checkboard_to_cam_rT_map;
						delete[]visible_map;
						delete[]offset_map;
						return false;
					}
				}
			}
		}

		delete[]offset_map;

		T* cam_R = new T[9 * n_cams];
		T* cam_T = new T[3 * n_cams];
		T* checkboard_R = new T[9 * n_checkboards];
		T* checkboard_T = new T[3 * n_checkboards];
		int* cam_visited = new int[n_cams]; 
		memset(cam_visited, 0, sizeof(int)*n_cams);
		int* checkboard_visited = new int[n_checkboards];
		memset(checkboard_visited, 0, sizeof(int)*n_checkboards);
		double* cam_visited_err = new double[n_cams];
		double* checkboard_visited_err = new double[n_checkboards];
		
		cam_R[0] = cam_R[4] = cam_R[8] = 1;
		cam_R[1] = cam_R[2] = cam_R[3] = cam_R[5] = cam_R[6] = cam_R[7] = 0;
		cam_T[0] = cam_T[1] = cam_T[2] = 0;
		cam_visited_err[0] = 0;
		cam_visited[0] = 2;

		/*Search in Shortest Path*/
		
		
		while (true)
		{
			bool any_change = false;
			for (int i = 0; i < n_cams; i++)
			{
				if (cam_visited[i])
				{
					for (int j = 0; j < n_checkboards; j++)
					{
						if (visible_map[i*n_checkboards + j])
						{
							switch (checkboard_visited[j])
							{
							case 0:
							{
									  checkboard_visited[j] = 1;
									  checkboard_visited_err[j] = cam_visited_err[i] + checkboard_to_cam_err_map[j*n_cams + i];
									  T tmp_R[9];
									  T* tmp_T = checkboard_to_cam_rT_map + (j*n_cams + i) * 6 + 3;
									  ZQ_Rodrigues::ZQ_Rodrigues_r2R(checkboard_to_cam_rT_map + (j*n_cams + i) * 6, tmp_R);
									 
									  ZQ_MathBase::MatrixMul(cam_R + i * 9, tmp_R, 3, 3, 3, checkboard_R + j * 9);
									  ZQ_MathBase::MatrixMul(cam_R + i * 9, tmp_T, 3, 3, 1, checkboard_T + j * 3);
									  checkboard_T[j * 3 + 0] += cam_T[i * 3 + 0];
									  checkboard_T[j * 3 + 1] += cam_T[i * 3 + 1];
									  checkboard_T[j * 3 + 2] += cam_T[i * 3 + 2];
							}
								break;
							case 1:
							{
									  double tmp_err = cam_visited_err[i] + checkboard_to_cam_err_map[j*n_cams + i];
									  if (tmp_err < checkboard_visited_err[j])
									  {
										  checkboard_visited_err[j] = tmp_err;
										  T tmp_R[9];
										  T* tmp_T = checkboard_to_cam_rT_map + (j*n_cams + i) * 6 + 3;
										  ZQ_Rodrigues::ZQ_Rodrigues_r2R(checkboard_to_cam_rT_map + (j*n_cams + i) * 6, tmp_R);

										  ZQ_MathBase::MatrixMul(cam_R + i * 9, tmp_R, 3, 3, 3, checkboard_R + j * 9);
										  ZQ_MathBase::MatrixMul(cam_R + i * 9, tmp_T, 3, 3, 1, checkboard_T + j * 3);
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
			for (int j = 0; j < n_checkboards; j++)
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

			for (int j = 0; j < n_checkboards; j++)
			{
				if (checkboard_visited[j])
				{
					for (int i = 0; i < n_cams; i++)
					{
						if (visible_map[i*n_checkboards + j])
						{
							switch (cam_visited[i])
							{
							case 0:
							{
									  cam_visited[i] = 1;
									  cam_visited_err[i] = checkboard_visited_err[j] + checkboard_to_cam_err_map[j*n_cams + i];
									  T tmp_R[9];
									  T* tmp_T = checkboard_to_cam_rT_map + (j*n_cams + i) * 6 + 3;
									  ZQ_Rodrigues::ZQ_Rodrigues_r2R(checkboard_to_cam_rT_map + (j*n_cams + i) * 6, tmp_R);
									 
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
									  ZQ_MathBase::MatrixMul(checkboard_R + j * 9, inv_R, 3, 3, 3, cam_R + i * 9);
									  ZQ_MathBase::MatrixMul(checkboard_R + j * 9, inv_T, 3, 3, 1, cam_T + i * 3);
									  cam_T[i * 3 + 0] += checkboard_T[j * 3 + 0];
									  cam_T[i * 3 + 1] += checkboard_T[j * 3 + 1];
									  cam_T[i * 3 + 2] += checkboard_T[j * 3 + 2];
							}
								break;
							case 1:
							{
									  double tmp_err = checkboard_visited_err[j] + checkboard_to_cam_err_map[j*n_cams + i];
									  if (tmp_err < cam_visited_err[i])
									  {
										  cam_visited_err[i] = tmp_err;
										  T tmp_R[9];
										  T* tmp_T = checkboard_to_cam_rT_map + (j*n_cams + i) * 6 + 3;
										  ZQ_Rodrigues::ZQ_Rodrigues_r2R(checkboard_to_cam_rT_map + (j*n_cams + i) * 6, tmp_R);

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
										  ZQ_MathBase::MatrixMul(checkboard_R + j * 9, inv_R, 3, 3, 3, cam_R + i * 9);
										  ZQ_MathBase::MatrixMul(checkboard_R + j * 9, inv_T, 3, 3, 1, cam_T + i * 3);
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
			for (int i = 0; i < n_cams; i++)
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
		delete[]checkboard_to_cam_err_map;
		delete[]checkboard_to_cam_rT_map;
		delete[]visible_map;
		delete[]cam_visited_err;
		delete[]checkboard_visited_err;

		//
		bool done_flag = true;
		for (int i = 0; i < n_cams; i++)
		{
			if (cam_visited[i] == 0)
			{
				done_flag = false;
				break;
			}
		}
		for (int j = 0; j < n_checkboards; j++)
		{
			if (checkboard_visited[j] == 0)
			{
				done_flag = false;
				break;
			}
		}

		delete[]checkboard_visited;
		delete[]cam_visited;

		if (!done_flag)
		{
			delete[]cam_R;
			delete[]cam_T;
			delete[]checkboard_R;
			delete[]checkboard_T;
			return false;
		}
		for (int j = 0; j < n_checkboards; j++)
		{
			if (!ZQ_Rodrigues::ZQ_Rodrigues_R2r(checkboard_R + j * 9, checkboard_rT_to_cam0 + j * 6))
			{
				delete[]cam_R;
				delete[]cam_T;
				delete[]checkboard_R;
				delete[]checkboard_T;
				return false;
			}
			memcpy(checkboard_rT_to_cam0 + j * 6 + 3, checkboard_T + j * 3, sizeof(T)* 3);
		}
		delete[]checkboard_R;
		delete[]checkboard_T;

		for (int i = 1; i < n_cams; i++)
		{
			T* tmp_R = cam_R + i * 9;
			T* tmp_T = cam_T + i * 3;
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
			if (!ZQ_Rodrigues::ZQ_Rodrigues_R2r(inv_R, cam0_to_othercam_rT + (i-1) * 6))
			{
				delete[]cam_R;
				delete[]cam_T;
				return false;
			}
			memcpy(cam0_to_othercam_rT + (i-1) * 6 + 3, inv_T, sizeof(T)* 3);
		}
		delete[]cam_R;
		delete[]cam_T;
		
		return true;
	}

	template<class T>
	bool ZQ_Calibration::_multicalib_with_known_intrinsic_func(const T* p, T* hx, int m, int n, const void* data)
	{
		const Multicalib_Data_Header<T>* ptr = (const Multicalib_Data_Header<T>*)data;
		double eps = ptr->eps;
		int N = ptr->n_pts;
		int n_checkboards = ptr->n_checkboards;
		int n_cams = ptr->n_cams;
		const T* X3 = ptr->X3;
		const T* X2 = ptr->X2;
		const T* intrinsic_para = ptr->intrinsic_para;
		const int* visible_num = ptr->visible_num;
		const int* visible_offset = ptr->visible_offset;
		const int* visible_idx = ptr->visible_idx;

		const T* checkboard_rT_to_cam0 = p;
		const T* cam0_to_othercam_rT = p + n_checkboards * 6;
		
		
		int total_visible_num = visible_offset[n_cams - 1] + visible_num[n_cams - 1];
		T* tmp_X2 = new T[total_visible_num*N * 2];

		for (int i = 0; i < n_cams; i++)
		{
			T A[9] = 
			{
				intrinsic_para[i * 4 + 0], 0, intrinsic_para[i*4+2],
				0, intrinsic_para[i * 4 + 1], intrinsic_para[i*4+3],
				0,0,1
			};

			int cur_vis_off = visible_offset[i];
			int cur_vis_num = visible_num[i];
			for (int j = 0; j < cur_vis_num; j++)
			{
				int cur_checkboard_idx = visible_idx[cur_vis_off + j];
				T cam_R[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
				T cam_T[3] = { 0, 0, 0 };
				T tmp_r[3];
				if (i != 0)
				{
					memcpy(tmp_r, cam0_to_othercam_rT + (i - 1) * 6, sizeof(T)* 3);
					ZQ_Rodrigues::ZQ_Rodrigues_autoscale(tmp_r);
					if (!ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun(tmp_r, cam_R))
					{
						delete[]tmp_X2;
						return false;
					}
					memcpy(cam_T, cam0_to_othercam_rT + (i - 1) * 6 + 3, sizeof(T)* 3);
				}
				T checkboard_R[9];
				const T* checkboard_T = checkboard_rT_to_cam0 + cur_checkboard_idx * 6 + 3;
				memcpy(tmp_r, checkboard_rT_to_cam0 + cur_checkboard_idx * 6, sizeof(T)* 3);
				ZQ_Rodrigues::ZQ_Rodrigues_autoscale(tmp_r);
				if (!ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun(tmp_r, checkboard_R))
				{
					delete[]tmp_X2;
					return false;
				}

				T total_R[9], total_T[3];
				ZQ_MathBase::MatrixMul(cam_R, checkboard_R, 3, 3, 3, total_R);
				ZQ_MathBase::MatrixMul(cam_R, checkboard_T, 3, 3, 1, total_T);
				total_T[0] += cam_T[0];
				total_T[1] += cam_T[1];
				total_T[2] += cam_T[2];

				proj_no_distortion(N, A, total_R, total_T, X3, tmp_X2 + (cur_vis_off + j)*N * 2, eps);
			}
		}
		
		for (int i = 0; i < total_visible_num*N * 2;i++)
		{
			hx[i] = tmp_X2[i] - X2[i];
		}

		
		delete[]tmp_X2;
		return true;

	}

	template<class T>
	bool ZQ_Calibration::_multicalib_with_known_intrinsic_jac(const T* p, T* jx, int m, int n, const void* data)
	{
		const Multicalib_Data_Header<T>* ptr = (const Multicalib_Data_Header<T>*)data;
		double eps = ptr->eps;
		int N = ptr->n_pts;
		int n_checkboards = ptr->n_checkboards;
		int n_cams = ptr->n_cams;
		const T* X3 = ptr->X3;
		const T* X2 = ptr->X2;
		const T* intrinsic_para = ptr->intrinsic_para;
		const int* visible_num = ptr->visible_num;
		const int* visible_offset = ptr->visible_offset;
		const int* visible_idx = ptr->visible_idx;

		memset(jx, 0, sizeof(T)*m*n);

		const T* checkboard_rT_to_cam0 = p;
		const T* cam0_to_othercam_rT = p + n_checkboards * 6;
		int total_visible_num = visible_offset[n_cams - 1] + visible_num[n_cams - 1];

		T tmp_r[3];
		/***  cam i==0 ***/
		{
			int i = 0;
			T A[9] =
			{
				intrinsic_para[i * 4 + 0], 0, intrinsic_para[i * 4 + 2],
				0, intrinsic_para[i * 4 + 1], intrinsic_para[i * 4 + 3],
				0, 0, 1
			};
			T dAdint[36] =
			{
				1, 0, 0, 0,
				0, 0, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 0, 1,
				0, 0, 0, 0,
				0, 0, 0, 0,
				0, 0, 0, 0
			};

			int cur_vis_off = visible_offset[i];
			int cur_vis_num = visible_num[i];
			for (int j = 0; j < cur_vis_num; j++)
			{
				int cur_checkboard_idx = visible_idx[cur_vis_off + j];
				T dR_0_dr_0[27] = { 0 }, R_0[9] = { 0 };
				const T* T_0 = checkboard_rT_to_cam0 + cur_checkboard_idx * 6 + 3;
				const T* r_0 = checkboard_rT_to_cam0 + cur_checkboard_idx * 6;
				memcpy(tmp_r, r_0, sizeof(T)* 3);
				ZQ_Rodrigues::ZQ_Rodrigues_autoscale(tmp_r);
				ZQ_Rodrigues::ZQ_Rodrigues_r2R_jac(tmp_r, R_0, dR_0_dr_0);
				
				for (int pp = 0; pp < N; pp++)
				{
					int cur_row0 = (cur_vis_off + j)*N * 2 + pp * 2 + 0;
					int cur_row1 = (cur_vis_off + j)*N * 2 + pp * 2 + 1;
					T cur_X3[3] = { X3[pp * 3 + 0], X3[pp * 3 + 1], X3[pp * 3 + 2] };


					/***************************/
					//var1 = R_0*X3+T_0: dvar1_dR_0,dvar1_dT_0
					T var1[3] = {
						R_0[0] * cur_X3[0] + R_0[1] * cur_X3[1] + R_0[2] * cur_X3[2] + T_0[0],
						R_0[3] * cur_X3[0] + R_0[4] * cur_X3[1] + R_0[5] * cur_X3[2] + T_0[1],
						R_0[6] * cur_X3[0] + R_0[7] * cur_X3[1] + R_0[8] * cur_X3[2] + T_0[2]
					};

					T dvar1_dR_0[27] = {
						cur_X3[0], cur_X3[1], cur_X3[2], 0, 0, 0, 0, 0, 0,
						0, 0, 0, cur_X3[0], cur_X3[1], cur_X3[2], 0, 0, 0,
						0, 0, 0, 0, 0, 0, cur_X3[0], cur_X3[1], cur_X3[2]
					};
					T dvar1_dT_0[9] =
					{
						1, 0, 0,
						0, 1, 0,
						0, 0, 1
					};

					//
					T dvar1_drT_0[18] = { 0 };
					for (int iii = 0; iii < 3; iii++)
					{
						for (int jjj = 0; jjj < 3; jjj++)
						{
							for (int kkk = 0; kkk < 9; kkk++)
								dvar1_drT_0[iii * 6 + jjj] += dvar1_dR_0[iii * 9 + kkk] * dR_0_dr_0[kkk * 3 + jjj];
							dvar1_drT_0[iii * 6 + jjj + 3] = dvar1_dT_0[iii * 3 + jjj];
						}
					}


					/**********************************/
					//var2 = A*var1:  dvar2dvar1

					T var2[3] = {
						A[0] * var1[0] + A[1] * var1[1] + A[2] * var1[2],
						A[3] * var1[0] + A[4] * var1[1] + A[5] * var1[2],
						A[6] * var1[0] + A[7] * var1[1] + A[8] * var1[2]
					};

					T dvar2_dvar1[9] = {
						A[0], A[1], A[2],
						A[3], A[4], A[5],
						A[6], A[7], A[8]
					};
					T dvar2_drT_0[18] = { 0 };
					ZQ_MathBase::MatrixMul(dvar2_dvar1, dvar1_drT_0, 3, 3, 6, dvar2_drT_0);

					/******************/
					//X2 = [var2(1)/var2(3);var2(2)/var2(3)]
					T cur_X2[2] = {
						var2[0] * var2[2] / (var2[2] * var2[2] + eps*eps),
						var2[1] * var2[2] / (var2[2] * var2[2] + eps*eps)
					};

					T dX2dvar2[6] =
					{
						var2[2] / (var2[2] * var2[2] + eps*eps), 0, -var2[0] / (var2[2] * var2[2] + eps*eps),
						0, var2[2] / (var2[2] * var2[2] + eps*eps), -var2[1] / (var2[2] * var2[2] + eps*eps)
					};

					T dX2drT_0[12] = { 0 };
					ZQ_MathBase::MatrixMul(dX2dvar2, dvar2_drT_0, 2, 3, 6, dX2drT_0);

					/***************/
					int offset_rT_0 = cur_checkboard_idx * 6;
					for (int iii = 0; iii < 6; iii++)
					{
						jx[cur_row0*m + offset_rT_0 + iii] = dX2drT_0[iii];
						jx[cur_row1*m + offset_rT_0 + iii] = dX2drT_0[6 + iii];

					}
				}
			}
		}

		/******* cam i >= 1  ******/
		for (int i = 1; i < n_cams; i++)
		{
			T A[9] =
			{
				intrinsic_para[i * 4 + 0], 0, intrinsic_para[i * 4 + 2],
				0, intrinsic_para[i * 4 + 1], intrinsic_para[i * 4 + 3],
				0, 0, 1
			};
			T dAdint[36] = 
			{
				1,0,0,0,
				0,0,0,0,
				0,0,1,0,
				0,0,0,0,
				0,1,0,0,
				0,0,0,1,
				0,0,0,0,
				0,0,0,0,
				0,0,0,0
			};

			
			T R_02i[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
			T T_02i[3] = { 0, 0, 0 };
			T dR_02i_dr_02i[27] = { 0 };

			const T* r_02i = cam0_to_othercam_rT + (i - 1) * 6;
			const T* t_02i = cam0_to_othercam_rT + (i - 1) * 6 + 3;
			memcpy(tmp_r, r_02i, sizeof(T)* 3);
			ZQ_Rodrigues::ZQ_Rodrigues_autoscale(tmp_r);
			if (!ZQ_Rodrigues::ZQ_Rodrigues_r2R_jac(tmp_r, dR_02i_dr_02i))
			{
				//printf("error: Rodrigues r to R failed:(%d)%s\n",__LINE__,__FILE__);
				return false;
			}
			if (!ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun(tmp_r, R_02i))
			{
				//printf("error: Rodrigues r to R failed:(%d)%s\n",__LINE__,__FILE__);
				return false;
			}
			memcpy(T_02i, t_02i, sizeof(T)* 3);


			int cur_vis_off = visible_offset[i];
			int cur_vis_num = visible_num[i];
			for (int j = 0; j < cur_vis_num; j++)
			{
				int cur_checkboard_idx = visible_idx[cur_vis_off + j];
				T dR_0_dr_0[27] = { 0 }, R_0[9] = { 0 };
				const T* T_0 = checkboard_rT_to_cam0 + cur_checkboard_idx * 6 + 3;
				const T* r_0 = checkboard_rT_to_cam0 + cur_checkboard_idx * 6;
				memcpy(tmp_r, r_0, sizeof(T)* 3);
				ZQ_Rodrigues::ZQ_Rodrigues_autoscale(tmp_r);
				ZQ_Rodrigues::ZQ_Rodrigues_r2R_jac(tmp_r, R_0, dR_0_dr_0);
				for (int pp = 0; pp < N; pp++)
				{
					int cur_row0 = (cur_vis_off + j)*N * 2 + pp * 2 + 0;
					int cur_row1 = (cur_vis_off + j)*N * 2 + pp * 2 + 1;
					T cur_X3[3] = { X3[pp * 3 + 0], X3[pp * 3 + 1], X3[pp * 3 + 2] };

					//X3_i = R*X3+T: dX3_i_dR_0,dX3_i_dT_0
					T X3_i[3] = {
						R_0[0] * cur_X3[0] + R_0[1] * cur_X3[1] + R_0[2] * cur_X3[2] + T_0[0],
						R_0[3] * cur_X3[0] + R_0[4] * cur_X3[1] + R_0[5] * cur_X3[2] + T_0[1],
						R_0[6] * cur_X3[0] + R_0[7] * cur_X3[1] + R_0[8] * cur_X3[2] + T_0[2]
					};

					T dX3_i_dR_0[27] = {
						cur_X3[0], cur_X3[1], cur_X3[2], 0, 0, 0, 0, 0, 0,
						0, 0, 0, cur_X3[0], cur_X3[1], cur_X3[2], 0, 0, 0,
						0, 0, 0, 0, 0, 0, cur_X3[0], cur_X3[1], cur_X3[2]
					};
					T dX3_i_dT_0[9] =
					{
						1, 0, 0,
						0, 1, 0,
						0, 0, 1
					};

					//
					T dX3_i_drT_0[18] = { 0 };
					for (int iii = 0; iii < 3; iii++)
					{
						for (int jjj = 0; jjj < 3; jjj++)
						{
							for (int kkk = 0; kkk < 9; kkk++)
								dX3_i_drT_0[iii * 6 + jjj] += dX3_i_dR_0[iii * 9 + kkk] * dR_0_dr_0[kkk * 3 + jjj];
							dX3_i_drT_0[iii * 6 + jjj + 3] = dX3_i_dT_0[iii * 3 + jjj];
						}
					}

					/***************************/
					//var1 = R_02i*X3_i+T_02i: dvar1_dR_02i,dvar1_dT_02i
					T var1[3] = {
						R_02i[0] * X3_i[0] + R_02i[1] * X3_i[1] + R_02i[2] * X3_i[2] + T_02i[0],
						R_02i[3] * X3_i[0] + R_02i[4] * X3_i[1] + R_02i[5] * X3_i[2] + T_02i[1],
						R_02i[6] * X3_i[0] + R_02i[7] * X3_i[1] + R_02i[8] * X3_i[2] + T_02i[2]
					};

					T dvar1_dR_02i[27] = {
						X3_i[0], X3_i[1], X3_i[2], 0, 0, 0, 0, 0, 0,
						0, 0, 0, X3_i[0], X3_i[1], X3_i[2], 0, 0, 0,
						0, 0, 0, 0, 0, 0, X3_i[0], X3_i[1], X3_i[2]
					};
					T dvar1_dT_02i[9] =
					{
						1, 0, 0,
						0, 1, 0,
						0, 0, 1
					};

					//
					T dvar1_drT_02i[18] = { 0 };
					for (int iii = 0; iii < 3; iii++)
					{
						for (int jjj = 0; jjj < 3; jjj++)
						{
							for (int kkk = 0; kkk < 9; kkk++)
								dvar1_drT_02i[iii * 6 + jjj] += dvar1_dR_02i[iii * 9 + kkk] * dR_02i_dr_02i[kkk * 3 + jjj];
							dvar1_drT_02i[iii * 6 + jjj + 3] = dvar1_dT_02i[iii * 3 + jjj];
						}
					}

					//
					const T* dvar1_dX3_i = R_02i;
					T dvar1_drT_0[18] = { 0 };
					ZQ_MathBase::MatrixMul(dvar1_dX3_i, dX3_i_drT_0, 3, 3, 6, dvar1_drT_0);

					/**********************************/
					//var2 = A*var1:  dvar2dvar1
					
					T var2[3] = {
						A[0] * var1[0] + A[1] * var1[1] + A[2] * var1[2],
						A[3] * var1[0] + A[4] * var1[1] + A[5] * var1[2],
						A[6] * var1[0] + A[7] * var1[1] + A[8] * var1[2]
					};

					T dvar2_dvar1[9] = {
						A[0], A[1], A[2],
						A[3], A[4], A[5],
						A[6], A[7], A[8]
					};
					T dvar2_drT_0[18] = { 0 };
					T dvar2_drT_02i[18] = { 0 };
					ZQ_MathBase::MatrixMul(dvar2_dvar1, dvar1_drT_0, 3, 3, 6, dvar2_drT_0);
					ZQ_MathBase::MatrixMul(dvar2_dvar1, dvar1_drT_02i, 3, 3, 6, dvar2_drT_02i);

					/******************/
					//X2 = [var2(1)/var2(3);var2(2)/var2(3)]
					T cur_X2[2] = {
						var2[0] * var2[2] / (var2[2] * var2[2] + eps*eps),
						var2[1] * var2[2] / (var2[2] * var2[2] + eps*eps)
					};

					T dX2dvar2[6] =
					{
						var2[2] / (var2[2] * var2[2] + eps*eps), 0, -var2[0] / (var2[2] * var2[2] + eps*eps),
						0, var2[2] / (var2[2] * var2[2] + eps*eps), -var2[1] / (var2[2] * var2[2] + eps*eps)
					};

					T dX2drT_0[12] = { 0 };
					T dX2drT_02i[12] = { 0 };
					ZQ_MathBase::MatrixMul(dX2dvar2, dvar2_drT_0, 2, 3, 6, dX2drT_0);
					ZQ_MathBase::MatrixMul(dX2dvar2, dvar2_drT_02i, 2, 3, 6, dX2drT_02i);

					/***************/
					int offset_rT_0 = cur_checkboard_idx * 6;
					int offset_rT_02i = n_checkboards * 6 + (i-1)*6;
					for (int iii = 0; iii < 6; iii++)
					{
						jx[cur_row0*m + offset_rT_0 + iii] = dX2drT_0[iii];
						jx[cur_row1*m + offset_rT_0 + iii] = dX2drT_0[6 + iii];

						jx[cur_row0*m + offset_rT_02i + iii] = dX2drT_02i[iii];
						jx[cur_row1*m + offset_rT_02i + iii] = dX2drT_02i[6 + iii];
					}
				}
			}
		}
		/*for (int i = 0; i < n; i++)
		{
			if (i % 20 == 0)
				printf("\n");
			printf("[%d]:%e\n", i,jx[i*m + 6]);
		}*/
		return true;
	}

	/*Calibrate multi-camera.
	The intrinsic parameters are known.
	X3 is a series of checkboard coordinates.
	The exterior parameters are estimated by using "posit_coplanar_robust".
	visible_num, visible_offset,visible_idx and X2 must be valid.
	*/
	template<class T>
	bool ZQ_Calibration::multicalib_with_known_intrinsic_with_init(int n_checkboards, int n_cams, int n_pts, const T* X3, const T* X2, const int* visible_num, const int* visible_offset, const int* visible_idx,
		const T* intrinsic_para, int max_iter_levmar, T* checkboard_rT_to_cam0, T* cam0_to_othercam_rT, double& avg_err_square, double eps /*= 1e-9*/)
	{
		Multicalib_Data_Header<T> data;
		data.n_checkboards = n_checkboards;
		data.n_pts = n_pts;
		data.n_cams = n_cams;
		data.X3 = X3;
		data.X2 = X2;
		data.visible_num = visible_num;
		data.visible_offset = visible_offset;
		data.visible_idx = visible_idx;
		data.eps = eps;
		data.intrinsic_para = intrinsic_para;

		ZQ_LevMarOptions opts;
		ZQ_LevMarReturnInfos infos;
		opts.tol_e_square = 1e-45;
		opts.tol_max_jte = 1e-45;
		opts.tol_dx_square = 1e-45;

		int total_visible_num = visible_offset[n_cams - 1] + visible_num[n_cams - 1];
		T* hx = new T[total_visible_num * n_pts * 2];
		memset(hx, 0, sizeof(T)*total_visible_num * n_pts * 2);

		T* unknowns = new T[n_checkboards * 6 + (n_cams - 1) * 6];
		memcpy(unknowns, checkboard_rT_to_cam0, sizeof(T)*n_checkboards * 6);
		memcpy(unknowns + n_checkboards * 6, cam0_to_othercam_rT, sizeof(T)* (n_cams - 1) * 6);
		if (!ZQ_LevMar::ZQ_LevMar_Der<T>(_multicalib_with_known_intrinsic_func<T>, _multicalib_with_known_intrinsic_jac<T>, unknowns, hx,
			n_checkboards * 6 + (n_cams - 1) * 6, total_visible_num * n_pts * 2, max_iter_levmar, opts, infos, &data))
		{
			delete[]hx;
			delete[]unknowns;
			return false;
		}
		memcpy(checkboard_rT_to_cam0, unknowns, sizeof(T)*n_checkboards * 6);
		memcpy(cam0_to_othercam_rT, unknowns + n_checkboards * 6, sizeof(T)* (n_cams - 1) * 6);
		avg_err_square = infos.final_e_square / (total_visible_num * n_pts);

		delete[]hx;
		delete[]unknowns;
		return true;
	}

	template<class T>
	bool ZQ_Calibration::multicalib_with_known_intrinsic(int n_checkboards, int n_cams, int n_pts, const T* X3, const T* X2, const int* visible_num, const int* visible_offset, const int* visible_idx,
		const T* intrinsic_para, int max_iter_posit, int max_iter_levmar, double tol_E, T* checkboard_rT_to_cam0, T* cam0_to_othercam_rT, double& avg_err_square, double eps /* = 1e-9*/)
	{

		T* tmp_checkboard_rT_to_cam0;
		if (checkboard_rT_to_cam0 == 0)
			tmp_checkboard_rT_to_cam0 = new T[n_checkboards * 6];
		else
			tmp_checkboard_rT_to_cam0 = checkboard_rT_to_cam0;
		if (!_multicalib_with_known_intrinsic_init(n_checkboards, n_cams, n_pts, X3, X2, visible_num, visible_offset, visible_idx, intrinsic_para, max_iter_posit,
			max_iter_levmar, tol_E, tmp_checkboard_rT_to_cam0, cam0_to_othercam_rT, eps))
		{
			if (checkboard_rT_to_cam0 == 0)
				delete[] tmp_checkboard_rT_to_cam0;
			return false;
		}

		if (!multicalib_with_known_intrinsic_with_init(n_checkboards, n_cams, n_pts, X3, X2, visible_num, visible_offset, visible_idx, intrinsic_para, max_iter_levmar,
			tmp_checkboard_rT_to_cam0, cam0_to_othercam_rT, avg_err_square, eps))
		{
			if (checkboard_rT_to_cam0 == 0)
				delete[] tmp_checkboard_rT_to_cam0;
			return false;
		}

		if (checkboard_rT_to_cam0 == 0)
			delete[] tmp_checkboard_rT_to_cam0;
		return true;
	}

}

#endif