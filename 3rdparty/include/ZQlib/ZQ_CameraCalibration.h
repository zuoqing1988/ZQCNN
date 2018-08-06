#ifndef _ZQ_CAMERA_CALIBRATION_H_
#define _ZQ_CAMERA_CALIBRATION_H_
#pragma once 

#include "ZQ_MathBase.h"
#include "ZQ_SVD.h"
#include "ZQ_Rodrigues.h"
#include "ZQ_DoubleImage.h"
#include "ZQ_LevMar.h"
#include "ZQ_SparseLevMar.h"
#include "ZQ_SparseMatrix.h"

namespace ZQ
{
	class ZQ_CameraCalibration
	{
	public:
		enum Calib_Method
		{
			CALIB_F2_C_ALPHA_K5,
			CALIB_F2_C_ALPHA_K4,
			CALIB_F2_C_ALPHA_K2,
			CALIB_F2_C_ALPHA,
			CALIB_F2_C_K5,
			CALIB_F2_C_K4,
			CALIB_F2_C_K2,
			CALIB_F2_C,
			CALIB_F1_C_ALPHA_K5,
			CALIB_F1_C_ALPHA_K4,
			CALIB_F1_C_ALPHA_K2,
			CALIB_F1_C_ALPHA,
			CALIB_F1_C_K5,
			CALIB_F1_C_K4,
			CALIB_F1_C_K2,
			CALIB_F1_C
		};
	public:

		template<class T>
		static void distort_points(int nPts, const T* in_x, const T k[5], T* out_x);

		template<class T>
		static bool project_points_fun(int nPts, const T* X, const T* rT, const T* f, const T* c, const T* k, const T alpha, T* xp, bool zAxis_in);

		template<class T>
		static bool project_points_fun(int nPts, const T* X, const T* rT, const T* f, const T* c, const T alpha, T* xp, bool zAxis_in);

		template<class T>
		static bool project_points_jac(int nPts, const T* X, const T* rT, const T* f, const T* c, const T* k, const T alpha, T* dxpdrT, T* dxpdf, T* dxpdc, T* dxpdk, T* dxpdalpha, bool zAxis_in);

		template<class T>
		static bool project_points_jac(int nPts, const T* X, const T* rT, const T* f, const T*c, const T alpha, T* dxpdrT, T* dxpdf, T* dxpdc, T* dxpdalpha, bool zAxis_in);

		template<class T>
		static void undistort_points(int nPts, const T* x_in, const T k, T* x_out);

		template<class T>
		static void undistort_points_oulu(int nPts, const T* x_in, const T k[5], T* x_out);

		template<class T>
		static void normalize_pixels(int nPts, const T* x_in, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, T* x_out);

		template<class T>
		static bool CalibrateCamera(int nViews, int nPts, int width, int height, const T* X2, const T* X3, T fc[2], T cc[2], T kc[5], T& alpha_c, T* rT, bool* active_images, Calib_Method method = CALIB_F2_C_ALPHA_K5, bool zAxis_in = true, int max_iter = 300, double tol_E = 5.0, bool sparse_solver = true, bool display = false);
		
	public:
		template<class T>
		static bool _compute_extrinsic_param(int nViews, int nPts, const T* X2, const T* X3, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, T* rT, bool* active_images, int max_iter, double thresh_cond, bool check_cond, bool zAxis_in);

	private:
		template<class T>
		static bool _compute_JJ3_ex3(int nViews, int nPts, const T* X2, const T* X3, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, const T* rT, bool* active_images, ZQ_Matrix<double>& JJ3, ZQ_Matrix<double>& ex3,
			double thresh_cond, bool check_cond, bool zAxis_in);

		template<class T>
		static bool _estimate_uncertainties(int nViews, const bool* active_images, const ZQ_Matrix<double>& JJ3, double sigma_x, T fc_err[2], T cc_err[2], T kc_err[5], T& alpha_err);

	public:
		template<class T>
		static bool _compute_err_calib(int nViews, int nPts, const T* X2, const T* X3, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, const T* rT, const bool* active_images, double& err_std, double& max_err, bool zAxis_in);
	private:
		template<class T>
		static bool _compute_extrinsic_init(int nPts, const T* X2, const T* X3, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, T rT[6], bool zAxis_in);

		template<class T>
		static bool _compute_extrinsic_refine(int nPts, const T* rT_init, const T* X2, const T* X3, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, T rT[6], double* JJ, bool zAxis_in, int max_iter = 20, double thresh_cond = 1e16);

		template<class T>
		static bool _compute_homography(int nPts, const T* m, const T* M, T H[9], T Hnorm[9], T inv_Hnorm[9]);

	private:// for Lev-Mar optimization
		template<class T>
		class Calib_Data_Header
		{
		public:
			const T* X3;
			const T* X2;
			int n_views;
			int n_pts;
			double eps;
			const T* fc_cc_alpha_kc;
			const T* rT;
			Calib_Method method;
			bool zAxis_in;
		};

		/*calibrate camera:
		based on Lev-Mar optimization, need good initialization.
		left hand coordinates.
		*/
		template<class T> static bool _calib_estimate_fun(const T* p, T* hx, int m, int n, const void* data);
		template<class T> static bool _calib_estimate_jac(const T* p, T* jx, int m, int n, const void* data);
		template<class T> static bool _calib_estimate_jac_sparse(const T* p, taucs_ccs_matrix*& jx, int m, int n, const void* data);

	public:
		/*calibrate camera:
		based on Lev-Mar optimization, need good initialization.
		left hand coordinates.
		*/
		template<class T>
		static bool _calib_estimate_LevMar(int nViews, int nPts, const T* X3, const T* X2, int max_iter, T fc[2], T cc[2], T kc[5], T& alpha_c, T* rT, double& avg_err_square, Calib_Method method, bool zAxis_in, bool sparse_solver, bool display);

		/*****************************************************************************************************************************/
		/*****************************************************************************************************************************/
		/*****************************************************************************************************************************/
		/***********************************  POSE Estimation     ****************************************/
		/***********************************  POSE Estimation     ****************************************/
		/***********************************  POSE Estimation     ****************************************/
		/*****************************************************************************************************************************/
		/*****************************************************************************************************************************/
		/*****************************************************************************************************************************/
	public:
		/*
		refer to the paper:
		iterative pose estimation using coplanar feature points. Denis Oberkampf, Daniel F. DeMenthon, Larry  S. Davis. CVIU, 1995.
		rT[0-5]: rx, ry, rz, Tx, Ty, Tz.  (rx,ry,rz,rw) is a quaternion.
		*/
		template<class T>
		static bool PositCoplanar(int nPts, const T* X3, const T* X2, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, int max_iter, double tol_E, T* rT, T* reproj_err_square, bool zAxis_in);

		/*
		The base idea is to use the method proposed in the paper:
		iterative pose estimation using coplanar points. Denis Oberkampf, Daniel F. DeMenthon, Larry  S. Davis. CVPR, 1993.
		However, I find it cannot make sure the method always converge to the optimal solution.
		But the translation seems to be near the optimal one according to my observations.
		So I choose 9 rotations to run Lev-Mar solvers to find a best solution.
		If all choices do not give a solution with avg_E < tol_avg_E, the best solution of the 9 will be returned,
		otherwise, the first one satisfying avg_E < tol_avg_E will be returned.
		*/
		template<class T>
		static bool PositCoplanarRobust(int nPts, const T* X3, const T* X2, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, int max_iter_posit, int max_iter_levmar, double tol_E, T* rT, double& avg_E, bool zAxis_in);

	private:
		template<class T>
		class Posit_Coplanar_Node
		{
		public:
			T kk[3];
			T R[9], tt[3], rr[3];
			double Z0;
			double Error;
		};

		template<class T>
		class Pose_Estimation_Data_Header
		{
		public:
			int nPts;
			const T* fc_cc_alpha_kc;
			const T* X3;
			const T* X2;
			bool zAxis_in;
		};

		template<class T>
		static bool _pose_estimation_fun(const T* p, T* hx, int m, int n, const void* data);

		template<class T>
		static bool _pose_estimation_jac(const T* p, T* jx, int m, int n, const void* data);

	public:
		template<class T>
		static bool pose_estimation_with_init(int nPts, const T* X3, const T* X2, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, int max_iter, T* rT, double& avg_err_square, bool zAxis_in);


	};
	/**************************************************************************************************************************************************************/

	template<class T>
	void ZQ_CameraCalibration::distort_points(int nPts, const T* in_x, const T k[5], T* out_x)
	{
		for (int pp = 0; pp < nPts; pp++)
		{
			double x = in_x[pp * 2 + 0];
			double y = in_x[pp * 2 + 1];
			double r2 = x*x + y*y;
			double r4 = r2*r2;
			double r6 = r2*r4;
			double xx = x*(1 + k[0] * r2 + k[1] * r4 + k[4] * r6) + 2 * k[2] * x*y + k[3] * (r2 + 2 * x*x);
			double yy = y*(1 + k[0] * r2 + k[1] * r4 + k[4] * r6) + k[2] * (r2 + 2 * y*y) + 2 * k[3] * x*y;
			out_x[pp * 2 + 0] = xx;
			out_x[pp * 2 + 1] = yy;
		}
	}

	template<class T>
	bool ZQ_CameraCalibration::project_points_fun(int nPts, const T* X, const T* rT, const T* f, const T* c, const T* k, const T alpha, T* xp, bool zAxis_in)
	{
		/*
		%
		%	[xp,dxpdom,dxpdT,dxpdf,dxpdc,dxpdk, dxpdalpha] = project_points2(X,om,T,f,c,k,alpha)
		%
		%	Projects a 3D structure onto the image plane.
		%
		%	INPUT:
		%		X: 3D structure in the world coordinate frame (3xN matrix for N points)
		%		(r,T): Rigid motion parameters between world coordinate frame and camera reference frame
		%             r: rotation vector (3x1 vector); T: translation vector (3x1 vector)
		%       f: camera focal length in units of horizontal and vertical pixel units (2x1 vector)
		%       c: principal point location in pixel units (2x1 vector)
		%       k: Distortion coefficients (radial and tangential) (5x1 vector)
		%       alpha: Skew coefficient between x and y pixel (alpha = 0 <=> square pixels)
		%
		%	OUTPUT:
		%		xp: Projected pixel coordinates (2xN matrix for N points)
		%       dxpdrT: Derivative of xp with respect to rT ((2N)x6 matrix)
		%       dxpdf: Derivative of xp with respect to f ((2N)x2 matrix if f is 2x1)
		%       dxpdc: Derivative of xp with respect to c ((2N)x2 matrix)
		%       dxpdk: Derivative of xp with respect to k ((2N)x5 matrix)
		%		dxpdalpha: Derivative of xp with respect to alpha ((2N)x1 matrix)
		%
		%	Definitions:
		%		Let P be a point in 3D of coordinates X in the world reference frame (stored in the matrix X)
		%		The coordinate vector of P in the camera reference frame is: Xc = R*X + T
		%		where R is the rotation matrix corresponding to the rotation vector r: R = rodrigues(r);
		%		call x, y and z the 3 coordinates of Xc: x = Xc(1); y = Xc(2); z = Xc(3);
		%		The pinehole projection coordinates of P is [a;b] where a=x/z and b=y/z.
		%		call r^2 = a^2 + b^2.
		%		The distorted point coordinates are: xd = [xx;yy] where:
		%
		%		xx = a * (1 + kc(1)*r^2 + kc(2)*r^4 + kc(5)*r^6)      +      2*kc(3)*a*b + kc(4)*(r^2 + 2*a^2);
		%		yy = b * (1 + kc(1)*r^2 + kc(2)*r^4 + kc(5)*r^6)      +      kc(3)*(r^2 + 2*b^2) + 2*kc(4)*a*b;
		%
		%	The left terms correspond to radial distortion (6th degree), the right terms correspond to tangential distortion
		%
		%	Finally, convertion into pixel coordinates: The final pixel coordinates vector xp=[xxp;yyp] where:
		%
		%	xxp = f(1)*(xx + alpha*yy) + c(1)
		%	yyp = f(2)*yy + c(2)
		%
		%
		%	NOTE: About 90 percent of the code takes care fo computing the Jacobian matrices
		*/

		T R[9];
		ZQ_Rodrigues::ZQ_Rodrigues_r2R(rT, R);
		
		for (int pp = 0; pp < nPts; pp++)
		{
			const T* XX = X + pp * 3;
			const T* ttt = rT + 3;
			T Y[3] =
			{
				R[0] * XX[0] + R[1] * XX[1] + R[2] * XX[2] + ttt[0],
				R[3] * XX[0] + R[4] * XX[1] + R[5] * XX[2] + ttt[1],
				R[6] * XX[0] + R[7] * XX[1] + R[8] * XX[2] + ttt[2]
			};

			if (Y[2] == 0)
			{
				return false;
			}

			T inv_Z = 1.0 / Y[2];
			if (!zAxis_in)
				inv_Z = -inv_Z;
			T x[2] = { Y[0] * inv_Z, Y[1] * inv_Z };

			// Add distortion :
			T r2 = x[0] * x[0] + x[1] * x[1];
			T r4 = r2*r2;
			T r6 = r2*r2*r2;
			
			// Radial distortion :
			T cdist = 1.0 + k[0] * r2 + k[1] * r4 + k[4] * r6;

			T xd1[2] =
			{
				x[0] * cdist, x[1] * cdist
			};

			// tangential distortion :
			T a1 = 2 * x[0] * x[1];
			T a2 = r2 + 2 * x[0] * x[0];
			T a3 = r2 + 2 * x[1] * x[1];
			T delta_x[2] =
			{
				k[2] * a1 + k[3] * a2, k[2] * a3 + k[3] * a1
			};
			
			T xd2[2] = { xd1[0] + delta_x[0], xd1[1] + delta_x[1] };
			
			//Add Skew
			T xd3[2] =
			{
				xd2[0] + alpha*xd2[1], xd2[1]
			};

			
			// Pixel coordinates :
			T xxp[2] =
			{
				xd3[0] * f[0] + c[0],
				xd3[1] * f[1] + c[1]
			};
			

			///////////
			memcpy(xp + pp * 2, xxp, sizeof(T)* 2);
		}
		return true;
	}
	
	template<class T>
	bool ZQ_CameraCalibration::project_points_fun(int nPts, const T* X, const T* rT, const T* f, const T* c, const T alpha, T* xp, bool zAxis_in)
	{
		T R[9];
		ZQ_Rodrigues::ZQ_Rodrigues_r2R(rT, R);

		for (int pp = 0; pp < nPts; pp++)
		{
			const T* XX = X + pp * 3;
			const T* ttt = rT + 3;
			T Y[3] =
			{
				R[0] * XX[0] + R[1] * XX[1] + R[2] * XX[2] + ttt[0],
				R[3] * XX[0] + R[4] * XX[1] + R[5] * XX[2] + ttt[1],
				R[6] * XX[0] + R[7] * XX[1] + R[8] * XX[2] + ttt[2]
			};

			if (Y[2] == 0)
				return false;

			T inv_Z = 1.0 / Y[2];
			if (!zAxis_in)
				inv_Z = -inv_Z;

			T x[2] = { Y[0] * inv_Z, Y[1] * inv_Z };

			
			//Add Skew
			T xd3[2] =
			{
				x[0] + alpha*x[1], x[1]
			};


			// Pixel coordinates :
			T xxp[2] =
			{
				xd3[0] * f[0] + c[0],
				xd3[1] * f[1] + c[1]
			};


			///////////
			memcpy(xp + pp * 2, xxp, sizeof(T)* 2);
		}
		return true;
	}

	template<class T>
	bool ZQ_CameraCalibration::project_points_jac(int nPts, const T* X, const T* rT, const T* f, const T* c, const T* k, const T alpha, T* dxpdrT, T* dxpdf, T* dxpdc, T* dxpdk, T* dxpdalpha, bool zAxis_in)
	{
		/*
		%
		%	[xp,dxpdom,dxpdT,dxpdf,dxpdc,dxpdk, dxpdalpha] = project_points2(X,om,T,f,c,k,alpha)
		%
		%	Projects a 3D structure onto the image plane.
		%
		%	INPUT: 
		%		X: 3D structure in the world coordinate frame (3xN matrix for N points)
		%		(r,T): Rigid motion parameters between world coordinate frame and camera reference frame
		%             r: rotation vector (3x1 vector); T: translation vector (3x1 vector)
		%       f: camera focal length in units of horizontal and vertical pixel units (2x1 vector)
		%       c: principal point location in pixel units (2x1 vector)
		%       k: Distortion coefficients (radial and tangential) (5x1 vector)
		%       alpha: Skew coefficient between x and y pixel (alpha = 0 <=> square pixels)
		%
		%	OUTPUT: 
		%		xp: Projected pixel coordinates (2xN matrix for N points)
		%       dxpdrT: Derivative of xp with respect to rT ((2N)x6 matrix)
		%       dxpdf: Derivative of xp with respect to f ((2N)x2 matrix if f is 2x1)
		%       dxpdc: Derivative of xp with respect to c ((2N)x2 matrix)
		%       dxpdk: Derivative of xp with respect to k ((2N)x5 matrix)
		%		dxpdalpha: Derivative of xp with respect to alpha ((2N)x1 matrix)
		%
		%	Definitions:
		%		Let P be a point in 3D of coordinates X in the world reference frame (stored in the matrix X)
		%		The coordinate vector of P in the camera reference frame is: Xc = R*X + T
		%		where R is the rotation matrix corresponding to the rotation vector r: R = rodrigues(r);
		%		call x, y and z the 3 coordinates of Xc: x = Xc(1); y = Xc(2); z = Xc(3);
		%		The pinehole projection coordinates of P is [a;b] where a=x/z and b=y/z.
		%		call r^2 = a^2 + b^2.
		%		The distorted point coordinates are: xd = [xx;yy] where:
		%
		%		xx = a * (1 + kc(1)*r^2 + kc(2)*r^4 + kc(5)*r^6)      +      2*kc(3)*a*b + kc(4)*(r^2 + 2*a^2);
		%		yy = b * (1 + kc(1)*r^2 + kc(2)*r^4 + kc(5)*r^6)      +      kc(3)*(r^2 + 2*b^2) + 2*kc(4)*a*b;
		%
		%	The left terms correspond to radial distortion (6th degree), the right terms correspond to tangential distortion
		%
		%	Finally, convertion into pixel coordinates: The final pixel coordinates vector xp=[xxp;yyp] where:
		%
		%	xxp = f(1)*(xx + alpha*yy) + c(1)
		%	yyp = f(2)*yy + c(2)
		%
		%
		%	NOTE: About 90 percent of the code takes care fo computing the Jacobian matrices
		*/

		T R[9], dRdr[27];
		ZQ_Rodrigues::ZQ_Rodrigues_r2R(rT, R, dRdr);		

		for (int pp = 0; pp < nPts; pp++)
		{
			const T* XX = X + pp * 3;
			const T* ttt = rT + 3;
			T Y[3] = 
			{
				R[0] * XX[0] + R[1] * XX[1] + R[2] * XX[2] + ttt[0],
				R[3] * XX[0] + R[4] * XX[1] + R[5] * XX[2] + ttt[1],
				R[6] * XX[0] + R[7] * XX[1] + R[8] * XX[2] + ttt[2]
			};

			if (Y[2] == 0)
				return false;

			T dYdR[27] =
			{
				XX[0], XX[1], XX[2], 0, 0, 0, 0, 0, 0,
				0, 0, 0, XX[0], XX[1], XX[2], 0, 0, 0,
				0, 0, 0, 0, 0, 0, XX[0], XX[1], XX[2]
			};
			T dYdr[9] = { 0 };
			ZQ_MathBase::MatrixMul(dYdR, dRdr, 3, 9, 3, dYdr);
			T dYdT[9] =
			{
				1, 0, 0,
				0, 1, 0,
				0, 0, 1
			};
			T inv_Z = 1.0 / Y[2];
			if (!zAxis_in)
				inv_Z = -inv_Z;
			T x[2] = { Y[0] * inv_Z, Y[1] * inv_Z };
			
			T dxdY[6] = 
			{
				inv_Z, 0, -x[0]*inv_Z,
				0, inv_Z, -x[1]*inv_Z
			};
			if (!zAxis_in)
			{
				dxdY[2] = -dxdY[2];
				dxdY[5] = -dxdY[5];
			}


			T dxdrT[12] = { 0 };
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					for (int k = 0; k < 3; k++)
					{
						dxdrT[i * 6 + j] += dxdY[i * 3 + k] * dYdr[k * 3 + j];
						dxdrT[i * 6 + j + 3] += dxdY[i * 3 + k] * dYdT[k * 3 + j];
					}
				}
			}

			// Add distortion :
			T r2 = x[0] * x[0] + x[1] * x[1];
			T dr2dx[2] = 
			{
				2 * x[0], 2 * x[1]
			};

			T r4 = r2*r2;
			T dr4dx[2] = 
			{
				4 * r2*x[0], 4 * r2*x[1]
			};

			T r6 = r2*r2*r2;
			T dr6dx[2] = 
			{
				6 * r4*x[0], 6 * r4*x[1]
			};

			// Radial distortion :
			T cdist = 1.0 + k[0] * r2 + k[1] * r4 + k[4] * r6;
			T dcdistdx[2] = 
			{
				k[0] * dr2dx[0] + k[1] * dr4dx[0] + k[4] * dr6dx[0],
				k[0] * dr2dx[1] + k[1] * dr4dx[1] + k[4] * dr6dx[1]
			};
			T dcdistdk[5] =
			{
				r2, r4, 0, 0, r6
			};

			T xd1[2] =
			{
				x[0] * cdist, x[1] * cdist
			};
			T dxd1dx[4] = 
			{
				cdist + x[0] * dcdistdx[0], x[0] * dcdistdx[1],
				x[1] * dcdistdx[0], cdist + x[1] * dcdistdx[1]
			};
			T dxd1dk[10] = 
			{
				x[0] * dcdistdk[0], x[0] * dcdistdk[1], x[0] * dcdistdk[2], x[0] * dcdistdk[3], x[0] * dcdistdk[4],
				x[1] * dcdistdk[0], x[1] * dcdistdk[1], x[1] * dcdistdk[2], x[1] * dcdistdk[3], x[1] * dcdistdk[4]
			};

			// tangential distortion :
			T a1 = 2 * x[0] * x[1];
			T a2 = r2 + 2 * x[0] * x[0];
			T a3 = r2 + 2 * x[1] * x[1];
			T delta_x[2] =
			{
				k[2] * a1 + k[3] * a2, k[2] * a3 + k[3] * a1
			};
			T ddelta_xdx[4] = 
			{
				k[2] * 2 * x[1] + k[3] * (dr2dx[0] + 4 * x[0]), k[2] * 2 * x[0] + k[3] * dr2dx[1],
				k[2] * dr2dx[0] + k[3] * 2 * x[1], k[2] * (dr2dx[1] + 4 * x[1]) + k[3] * 2 * x[0]
			};
			T ddelta_xdk[10] =
			{
				0, 0, a1, a2, 0,
				0, 0, a3, a1, 0
			};

			T xd2[2] = { xd1[0] + delta_x[0], xd1[1] + delta_x[1] };
			T dxd2dx[4] = 
			{
				dxd1dx[0] + ddelta_xdx[0], dxd1dx[1] + ddelta_xdx[1],
				dxd1dx[2] + ddelta_xdx[2], dxd1dx[3] + ddelta_xdx[3]
			};
			T dxd2dk[10];
			ZQ_MathBase::VecPlus(10, dxd1dk, ddelta_xdk, dxd2dk);

			//Add Skew
			T xd3[2] = 
			{
				xd2[0] + alpha*xd2[1], xd2[1]
			};

			T dxd3dx[4] = 
			{
				dxd2dx[0] + alpha* dxd2dx[2], dxd2dx[1] + alpha* dxd2dx[3],
				dxd2dx[2], dxd2dx[3]
			};
			T dxd3dk[10];
			for (int i = 0; i < 5; i++)
			{
				dxd3dk[0 * 5 + i] = dxd2dk[i] + alpha*dxd2dk[5 + i];
				dxd3dk[1 * 5 + i] = dxd2dk[5 + i];
			}
			T dxd3dalpha[2] =
			{
				xd2[1], 0
			};

			// Pixel coordinates :
			T xxp[2] = 
			{
				xd3[0] * f[0] + c[0],
				xd3[1] * f[1] + c[1]
			};
			T dxxpdx[4] = 
			{
				f[0] * dxd3dx[0], f[0] * dxd3dx[1],
				f[1] * dxd3dx[2], f[1] * dxd3dx[3]
			};
			T dxxpdrT[12] = { 0 };
			ZQ_MathBase::MatrixMul(dxxpdx, dxdrT, 2, 2, 6, dxxpdrT);
			T dxxpdf[4] =
			{
				xd3[0], 0,
				0, xd3[1]
			};
			T dxxpdc[4] =
			{
				1, 0,
				0, 1
			};
			T dxxpdk[10];
			for (int i = 0; i < 5; i++)
			{
				dxxpdk[0 * 5 + i] = dxd3dk[i] * f[0];
				dxxpdk[1 * 5 + i] = dxd3dk[5 + i] * f[1];
			}
			
			T dxxpdxd3[4] =
			{
				f[0], 0,
				0, f[1]
			};
			T dxxpdalpha[2];
			ZQ_MathBase::MatrixMul(dxxpdxd3, dxd3dalpha, 2, 2, 1, dxxpdalpha);
			
			///////////
			if (dxpdrT != 0)
				memcpy(dxpdrT + pp * 12, dxxpdrT, sizeof(T)* 12);
			if (dxpdk != 0)
				memcpy(dxpdk + pp * 10, dxxpdk, sizeof(T)* 10);
			if (dxpdf != 0)
				memcpy(dxpdf + pp * 4, dxxpdf, sizeof(T)* 4);
			if (dxpdc != 0)
				memcpy(dxpdc + pp * 4, dxxpdc, sizeof(T)* 4);
			if (dxpdalpha != 0)
				memcpy(dxpdalpha + pp * 2, dxxpdalpha, sizeof(T)* 2);
		}

		return true;
	}

	
	template<class T>
	bool ZQ_CameraCalibration::project_points_jac(int nPts, const T* X, const T* rT, const T* f, const T* c, const T alpha, T* dxpdrT, T* dxpdf, T* dxpdc, T* dxpdalpha, bool zAxis_in)
	{
		T R[9], dRdr[27];
		ZQ_Rodrigues::ZQ_Rodrigues_r2R(rT, R, dRdr);

		for (int pp = 0; pp < nPts; pp++)
		{
			const T* XX = X + pp * 3;
			const T* ttt = rT + 3;
			T Y[3] =
			{
				R[0] * XX[0] + R[1] * XX[1] + R[2] * XX[2] + ttt[0],
				R[3] * XX[0] + R[4] * XX[1] + R[5] * XX[2] + ttt[1],
				R[6] * XX[0] + R[7] * XX[1] + R[8] * XX[2] + ttt[2]
			};

			if (Y[2] == 0)
				return false;

			T dYdR[27] =
			{
				XX[0], XX[1], XX[2], 0, 0, 0, 0, 0, 0,
				0, 0, 0, XX[0], XX[1], XX[2], 0, 0, 0,
				0, 0, 0, 0, 0, 0, XX[0], XX[1], XX[2]
			};
			T dYdr[9] = { 0 };
			ZQ_MathBase::MatrixMul(dYdR, dRdr, 3, 9, 3, dYdr);
			T dYdT[9] =
			{
				1, 0, 0,
				0, 1, 0,
				0, 0, 1
			};
			T inv_Z = 1.0 / Y[2];
			if (!zAxis_in)
				inv_Z = -inv_Z;
			T x[2] = { Y[0] * inv_Z, Y[1] * inv_Z };

			T dxdY[6] =
			{
				inv_Z, 0, -x[0] * inv_Z,
				0, inv_Z, -x[1] * inv_Z
			};
			if (!zAxis_in)
			{
				dxdY[2] = -dxdY[2];
				dxdY[5] = -dxdY[5];
			}

			T dxdrT[12] = { 0 };
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					for (int k = 0; k < 3; k++)
					{
						dxdrT[i * 6 + j] += dxdY[i * 3 + k] * dYdr[k * 3 + j];
						dxdrT[i * 6 + j + 3] += dxdY[i * 3 + k] * dYdT[k * 3 + j];
					}
				}
			}

			
			//Add Skew
			T xd3[2] =
			{
				x[0] + alpha*x[1], x[1]
			};

			T dxd3dx[4] =
			{
				1, alpha,
				0, 1
			};
			
			T dxd3dalpha[2] =
			{
				x[1], 0
			};

			// Pixel coordinates :
			T xxp[2] =
			{
				xd3[0] * f[0] + c[0],
				xd3[1] * f[1] + c[1]
			};
			
			T dxxpdx[4] =
			{
				f[0] * dxd3dx[0], f[0] * dxd3dx[1],
				f[1] * dxd3dx[2], f[1] * dxd3dx[3]
			};
			T dxxpdrT[12] = { 0 };
			ZQ_MathBase::MatrixMul(dxxpdx, dxdrT, 2, 2, 6, dxxpdrT);
			T dxxpdf[4] =
			{
				xd3[0], 0,
				0, xd3[1]
			};
			T dxxpdc[4] =
			{
				1, 0,
				0, 1
			};
			T dxxpdxd3[4] =
			{
				f[0], 0,
				0, f[1]
			};
			T dxxpdalpha[2];
			ZQ_MathBase::MatrixMul(dxxpdxd3, dxd3dalpha, 2, 2, 1, dxxpdalpha);

			///////////
			if (dxpdrT != 0)
				memcpy(dxpdrT + pp * 12, dxxpdrT, sizeof(T)* 12);
			if (dxpdf != 0)
				memcpy(dxpdf + pp * 4, dxxpdf, sizeof(T)* 4);
			if (dxpdc != 0)
				memcpy(dxpdc + pp * 4, dxxpdc, sizeof(T)* 4);
			if (dxpdalpha != 0)
				memcpy(dxpdalpha + pp * 2, dxxpdalpha, sizeof(T)* 2);
		}

		return true;
	}

	template<class T>
	void ZQ_CameraCalibration::undistort_points(int nPts, const T* x_in, const T k, T* x_out)
	{
		/*
		%   compensates the radial distortion of the camera
		%	   on the image plane.
		%
		%   x_in : the image points got without considering the
		%            radial distortion.
		%   x_out : The image plane points after correction for the distortion
		%
		%	x_out and x_in are Nx2 arrays
		%
		%   NOTE : This compensation has to be done after the substraction
		%          of the center of projection, and division by the focal length.
		%
		%       (do it up to a second order approximation)
		*/

		for (int i = 0; i < nPts; i++)
		{
			double radius_2 = (double)x_in[i * 2 + 0] * x_in[i * 2 + 0] + x_in[i * 2 + 1] * x_in[i * 2 + 1];
			double radial_distortion = 1.0 + k*radius_2;
			double radius_2_comp = radius_2 / radial_distortion;
			radial_distortion = 1.0 + k*radius_2_comp;
			x_out[i * 2 + 0] = x_in[i * 2 + 0] / radius_distortion;
			x_out[i * 2 + 1] = x_in[i * 2 + 1] / radius_distortion;
		}
	}

	template<class T>
	void ZQ_CameraCalibration::undistort_points_oulu(int nPts, const T* x_in, const T k[5], T* x_out)
	{
		/*
		%	Compensates for radial and tangential distortion. Model From Oulu university.
		%
		%	INPUT: x_in: distorted (normalized) point coordinates in the image plane (Nx2 matrix)
		%       k: Distortion coefficients (radial and tangential) (5x1 vector)
		%
		%	OUTPUT: x_out: undistorted (normalized) point coordinates in the image plane (Nx2 matrix)
		%
		%	Method: Iterative method for compensation.
		%
		%	NOTE: This compensation has to be done after the subtraction
		%      of the principal point, and division by the focal length.
		*/

		double k1 = k[0];
		double k2 = k[1];
		double k3 = k[4];
		double p1 = k[2];
		double p2 = k[3];
		for (int i = 0; i < nPts; i++)
		{
			// initial guess
			double x[2] = { x_in[i * 2 + 0], x_in[i * 2 + 1] };

			for (int iter = 0; iter < 20; iter++)
			{
				double r_2 = x[0] * x[0] + x[1] * x[1];
				double k_radial = 1.0 + k1*r_2 + k2*r_2*r_2 + k3*r_2*r_2*r_2;
				double delta_x[2] =
				{
					2.0*p1*x[0] * x[1] + p2*(r_2 + 2.0*x[0] * x[0]),
					p1*(r_2 + 2.0*x[1] * x[1]) + 2.0*p2*x[0] * x[1]
				};
				x[0] = (x_in[i * 2 + 0] - delta_x[0]) / k_radial;
				x[1] = (x_in[i * 2 + 1] - delta_x[1]) / k_radial;
			}
			x_out[i * 2 + 0] = x[0];
			x_out[i * 2 + 1] = x[1];
		}
	}

	template<class T>
	void ZQ_CameraCalibration::normalize_pixels(int nPts, const T* x_in, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, T* x_out)
	{
		/*
		%	Computes the normalized coordinates xn given the pixel coordinates x_kk
		%	and the intrinsic camera parameters fc, cc and kc.
		%
		%	INPUT:
		%		x_in: Feature locations on the images
		%       fc: Camera focal length
		%       cc: Principal point coordinates
		%       kc: Distortion coefficients
		%       alpha_c: Skew coefficient
		%
		%	OUTPUT:
		%		x_out: Normalized feature locations on the image plane (a NX2 matrix)
		%
		*/

		T* x_distort = new T[nPts * 2];
		for (int i = 0; i < nPts; i++)
		{
			// First: Subtract principal point, and divide by the focal length :
			x_distort[i * 2 + 0] = (x_in[i * 2 + 0] - cc[0]) / fc[0];
			x_distort[i * 2 + 1] = (x_in[i * 2 + 1] - cc[1]) / fc[1];

			// Second: undo skew
			x_distort[i * 2 + 0] -= alpha_c * x_distort[i * 2 + 1];
		}

		// Third: Compensate for lens distortion :
		undistort_points_oulu(nPts, x_distort, kc, x_out);
		delete[]x_distort;
	}

	template<class T>
	bool ZQ_CameraCalibration::_compute_extrinsic_param(int nViews, int nPts, const T* X2, const T* X3, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, T* rT, bool* active_images, int max_iter, double thresh_cond, bool check_cond, bool zAxis_in)
	{
		ZQ_DImage<double> JJ(2 * nPts, 6);
		double*& JJ_data = JJ.data();
		for (int vv = 0; vv < nViews; vv++)
		{
			if (!active_images[vv])
				continue;
			const T* cur_X2 = X2 + vv*nPts * 2;
			const T* cur_X3 = X3 + vv*nPts * 3;
			T rT_init[6] = { 0, 0, 0, 0, 0, 80 };
			if (!_compute_extrinsic_init(nPts, cur_X2, cur_X3, fc, cc, kc, alpha_c, rT_init, zAxis_in))
				return false;
			if (!_compute_extrinsic_refine(nPts, rT_init, cur_X2, cur_X3, fc, cc, kc, alpha_c, rT + vv * 6, JJ_data, zAxis_in, max_iter, thresh_cond))
				return false;

			if (check_cond)
			{
				bool cond_succ, is_singular;
				double cond_num = ZQ_MathBase::Cond_by_double_svd(JJ_data, 2 * nPts, 6, cond_succ, is_singular);
				if (!cond_succ || is_singular || cond_num > thresh_cond)
				{
					active_images[vv] = false;
					printf("Warning: View #%d ill-conditioned. This image is now set inactive.\n", vv);
				}
			}
		}
		return true;
	}

	template<class T>
	bool ZQ_CameraCalibration::_compute_JJ3_ex3(int nViews, int nPts, const T* X2, const T* X3, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, const T* rT, bool* active_images, ZQ_Matrix<double>& JJ3, ZQ_Matrix<double>& ex3, 
		double thresh_cond, bool check_cond, bool zAxis_in)
	{
		ZQ_DImage<T> ex_kk(nPts * 2, 1);
		ZQ_DImage<T> x(nPts * 2, 1);
		ZQ_DImage<T> dxdrT(nPts * 2, 6);
		ZQ_DImage<T> dxdf(nPts * 2, 2);
		ZQ_DImage<T> dxdc(nPts * 2, 2);
		ZQ_DImage<T> dxdk(nPts * 2, 5);
		ZQ_DImage<T> dxdalpha(nPts * 2, 1);
		T*& ex_kk_data = ex_kk.data();
		T*& x_data = x.data();
		T*& dxdrT_data = dxdrT.data();
		T*& dxdf_data = dxdf.data();
		T*& dxdc_data = dxdc.data();
		T*& dxdk_data = dxdk.data();
		T*& dxdalpha_data = dxdalpha.data();

		ZQ_DImage<T> A(2 * nPts, 10);
		ZQ_DImage<T> AtA(10, 10);
		ZQ_DImage<T> BtB(6, 6);
		ZQ_DImage<T> AtB(10, 6);
		T*& A_data = A.data();
		T*& AtA_data = AtA.data();
		T*& BtB_data = BtB.data();
		T*& AtB_data = AtB.data();

		JJ3.Reset();
		ex3.Reset();
		for (int vv = 0; vv < nViews; vv++)
		{
			if (!active_images[vv])
			{
				continue;
			}

			const T* rT_vv = rT + vv * 6;
			const T* X2_vv = X2 + vv*nPts * 2;
			const T* X3_vv = X3 + vv*nPts * 3;

			bool proj_flag = project_points_fun(nPts, X3_vv, rT_vv, fc, cc, kc, alpha_c, x_data, zAxis_in) &&
				project_points_jac(nPts, X3_vv, rT_vv, fc, cc, kc, alpha_c, dxdrT_data, dxdf_data, dxdc_data, dxdk_data, dxdalpha_data, zAxis_in);
			if (!proj_flag)
			{
				return false;
			}
			ZQ_MathBase::VecMinus(2 * nPts, X2_vv, x_data, ex_kk_data);

			for (int i = 0; i < 2 * nPts; i++)
			{
				memcpy(A_data + i * 10, dxdf_data + i * 2, sizeof(T)* 2);
				memcpy(A_data + i * 10 + 2, dxdc_data + i * 2, sizeof(T)* 2);
				memcpy(A_data + i * 10 + 4, dxdalpha_data + i, sizeof(T)* 1);
				memcpy(A_data + i * 10 + 5, dxdk_data + i * 5, sizeof(T)* 5);
			}

			//A'*A
			for (int i = 0; i < 10; i++)
			{
				for (int j = 0; j < 10; j++)
				{
					AtA_data[i * 10 + j] = 0;
					for (int k = 0; k < 2 * nPts; k++)
						AtA_data[i * 10 + j] += A_data[k * 10 + i] * A_data[k * 10 + j];
				}
			}

			//B'*B
			for (int i = 0; i < 6; i++)
			{
				for (int j = 0; j < 6; j++)
				{
					BtB_data[i * 6 + j] = 0;
					for (int k = 0; k < 2 * nPts; k++)
						BtB_data[i * 6 + j] += dxdrT_data[k * 6 + i] * dxdrT_data[k * 6 + j];
				}
			}

			//A'*B
			for (int i = 0; i < 10; i++)
			{
				for (int j = 0; j < 6; j++)
				{
					AtB_data[i * 6 + j] = 0;
					for (int k = 0; k < 2 * nPts; k++)
						AtB_data[i * 6 + j] += A_data[k * 10 + i] * dxdrT_data[k * 6 + j];
				}
			}

			//JJ3(1:10, 1 : 10) = JJ3(1:10, 1 : 10) + A'*A;
			for (int i = 0; i < 10; i++)
			{
				for (int j = 0; j < 10; j++)
					JJ3.AddWith(i, j, AtA_data[i * 10 + j]);
			}

			//JJ3(10 + 6 * (kk - 1) + 1:10 + 6 * (kk - 1) + 6, 10 + 6 * (kk - 1) + 1 : 10 + 6 * (kk - 1) + 6) = B'*B;
			for (int i = 0; i < 6; i++)
			{
				for (int j = 0; j < 6; j++)
					JJ3.SetData(i + 10 + vv * 6, j + 10 + vv * 6, BtB_data[i * 6 + j]);
			}

			//JJ3(1:10, 10 + 6 * (kk - 1) + 1 : 10 + 6 * (kk - 1) + 6) = A'*B;
			//JJ3(10 + 6 * (kk - 1) + 1:10 + 6 * (kk - 1) + 6, 1 : 10) = (A'*B)';
			for (int i = 0; i < 10; i++)
			{
				for (int j = 0; j < 6; j++)
				{
					JJ3.SetData(i, 10 + j * 6 * vv, AtB_data[i * 6 + j]);
					JJ3.SetData(10 + j * 6 * vv, i, AtB_data[i * 6 + j]);
				}
			}

			//ex3(1:10) = ex3(1:10) + A'*exkk(:);
			for (int i = 0; i < 10; i++)
			{
				double At_exkk = 0;
				for (int k = 0; k < 2 * nPts; k++)
					At_exkk += A_data[k * 10 + i] * ex_kk_data[k];
				ex3.AddWith(i, 0, At_exkk);
			}

			//ex3(10 + 6 * (kk - 1) + 1:10 + 6 * (kk - 1) + 6) = B'*exkk(:);
			for (int i = 0; i < 6; i++)
			{
				double Bt_exkk = 0;
				for (int k = 0; k < 2 * nPts; k++)
					Bt_exkk += dxdrT_data[k * 6 + i] * ex_kk_data[k];
				ex3.AddWith(10 + 6 * vv + i, 0, Bt_exkk);
			}

			if (check_cond)
			{
				bool cond_succ, is_singular;
				double cond_num = ZQ_MathBase::Cond_by_double_svd(dxdrT_data, 2 * nPts, 6, cond_succ, is_singular);
				if (!cond_succ || is_singular || cond_num > thresh_cond)
				{
					active_images[vv] = false;
				}
			}
		}
		return true;
	}

	template<class T>
	bool ZQ_CameraCalibration::_estimate_uncertainties(int nViews, const bool* active_images, const ZQ_Matrix<double>& JJ3, double sigma_x, T fc_err[2], T cc_err[2], T kc_err[5], T& alpha_err)
	{
		ZQ_DImage<int> index_map(nViews, 1);
		int*& index_map_data = index_map.data();
		int active_num = 0;
		for (int vv = 0; vv < nViews; vv++)
			index_map_data[vv] = -1;
		for (int vv = 0; vv < nViews; vv++)
		{
			if (active_images[vv])
			{
				index_map_data[vv] = active_num++;
			}
		}

		int unknown_all_num = 10 + nViews * 6;
		const double* JJ3_data = JJ3.GetDataPtr();
		
		ZQ_Matrix<double> JJ2(10 + 6 * active_num, 10 + 6 * active_num);
		ZQ_Matrix<double> inv_JJ2(10 + 6 * active_num, 10 + 6 * active_num);
		
		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				JJ2.SetData(i, j, JJ3_data[i*unknown_all_num + j]);
			}
			for (int vv = 0; vv < nViews; vv++)
			{
				if (!active_images[vv])
					continue;

				int idx = index_map_data[vv];
				for (int j = 0; j < 6; j++)
				{
					JJ2.SetData(i, 10 + idx * 6 + j, JJ3_data[i*unknown_all_num + 10 + vv * 6 + j]);
					JJ2.SetData(10 + idx * 6 + j, i, JJ3_data[(10 + vv * 6 + j)*unknown_all_num + i]);
				}
			}
		}
		
		for (int vv_i = 0; vv_i < nViews; vv_i++)
		{
			if (!active_images[vv_i])
				continue;
			int idx_i = index_map_data[vv_i];
			for (int vv_j = 0; vv_j < nViews; vv_j++)
			{
				if (!active_images[vv_j])
					continue;
				int idx_j = index_map_data[vv_j];
				for (int i = 0; i < 6; i++)
				{
					for (int j = 0; j < 6; j++)
						JJ2.SetData(10 + idx_i * 6 + i, 10 + idx_j * 6 + j, JJ3_data[(10 + idx_i * 6 + i)*unknown_all_num + (10 + idx_j * 6 + j)]);
				}
			}
		}

		if (!ZQ_SVD::Invert(JJ2, inv_JJ2))
		{
			return false;
		}

		bool flag;
		for (int i = 0; i < 2; i++)
		{
			fc_err[i] = inv_JJ2.GetData(i, i, flag);
			if (!flag || fc_err[i] < 0)
			{
				//return false;
			}
			fc_err[i] = sqrt(fabs(fc_err[i]))*sigma_x;
		}
		for (int i = 0; i < 2; i++)
		{
			cc_err[i] = inv_JJ2.GetData(i+2, i+2, flag);
			if (!flag || cc_err[i] < 0)
			{
				//return false;
			}
			cc_err[i] = sqrt(fabs(cc_err[i]))*sigma_x;
		}
		for (int i = 0; i < 1; i++)
		{
			alpha_err = inv_JJ2.GetData(i + 4, i + 4, flag);
			if (!flag || alpha_err < 0)
			{
				//return false;
			}
			alpha_err = sqrt(fabs(alpha_err))*sigma_x;
		}
		for (int i = 0; i < 5; i++)
		{
			kc_err[i] = inv_JJ2.GetData(i + 5, i + 5, flag);
			if (!flag || kc_err[i] < 0)
			{
				//return false;
			}
			kc_err[i] = sqrt(fabs(kc_err[i]))*sigma_x;
		}
		return true;
	}

	template<class T>
	bool ZQ_CameraCalibration::_compute_err_calib(int nViews, int nPts, const T* X2, const T* X3, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, const T* rT, const bool* active_images, double& err_std, double& max_err, bool zAxis_in)
	{
		ZQ_DImage<T> x(nViews * 2 * nPts, 1);
		T*& x_data = x.data();
		int active_num = 0;
		//Reproject the patterns on the images, and compute the pixel errors :
		for (int vv = 0; vv < nViews; vv++)
		{
			if (!active_images[vv])
				continue;
			active_num++;
			const T* cur_X3 = X3 + vv*nPts * 3;
			const T* cur_X2 = X2 + vv*nPts * 2;
			const T* cur_rT = rT + 6 * vv;
			T* cur_x_data = x_data + vv*nPts * 2;
			if (!project_points_fun(nPts, cur_X3, cur_rT, fc, cc, kc, alpha_c, cur_x_data, zAxis_in))
				return false;

			for (int i = 0; i < nPts * 2; i++)
				cur_x_data[i] -= cur_X2[i];
		}
		if (active_num == 0)
		{
			err_std = 0;
			return true;
		}

		max_err = 0;
		double mean_val = 0;
		for (int vv = 0; vv < nViews; vv++)
		{
			if (!active_images[vv])
				continue;
			for (int i = 0; i < nPts * 2; i++)
			{
				mean_val += x_data[2 * nPts*vv + i];
				max_err = __max(max_err, fabs(x_data[2 * nPts*vv + i]));
			}
		}
		
		mean_val /= (active_num * 2 * nPts);
		err_std = 0;
		
		for (int vv = 0; vv < nViews; vv++)
		{
			if (!active_images[vv])
				continue;
			for (int i = 0; i < nPts * 2; i++)
			{
				err_std += (mean_val - x_data[2 * nPts*vv + i])*(mean_val - x_data[2 * nPts*vv + i]);
			}
		}
		err_std = sqrt(err_std / (active_num * 2 * nPts - 1));

		return true;
	}

	template<class T>
	bool ZQ_CameraCalibration::CalibrateCamera(int nViews, int nPts, int width, int height, const T* X2, const T* X3, T fc[2], T cc[2], T kc[5], T& alpha_c, T* rT, bool* active_images, ZQ_CameraCalibration::Calib_Method method, bool zAxis_in, int max_iter, double tol_E, bool sparse_solver, bool display)
	{
		/*
		%	Main calibration function. Computes the intrinsic andextrinsic parameters.
		%
		%	INPUT:
		%		X2: Feature locations on the images
		%       X3: Corresponding grid coordinates
		%
		%	OUTPUT:
		%		fc: Camera focal length
		%       cc: Principal point coordinates
		%       alpha_c: Skew coefficient
		%       kc: Distortion coefficients
		%       rT: 3D rotation-translation vectors attached to the grid positions in space
		*/

		bool check_cond = true;
		double thresh_cond = 1e6;

		// init fc, cc, alpha_c, kc
		double init_fovY = 35.0 / 45.0*atan(1.0);
		alpha_c = 0;
		fc[0] = fc[1] = height*0.5 / tan(0.5*init_fovY);
		cc[0] = width*0.5; cc[1] = height*0.5;
		memset(kc, 0, sizeof(T)* 5);
		//fc[0] = 3766;  fc[1] = 3768;
		//cc[0] = 832; cc[1] = 395;
		//kc[0] = -0.1326597;   kc[1] = 2.2019646; kc[2] = -0.0124381; kc[3] = -0.0056662; kc[4] = 0;

		if (display)
			printf("init extrinsic...");
		
		// Computes the extrinsic parameters for all the active calibration images
		_compute_extrinsic_param(nViews, nPts, X2, X3, fc, cc, kc, alpha_c, rT, active_images, 20, thresh_cond, check_cond, zAxis_in);
		

		if (display)
			printf("done!\n");


		
		/**************************** Lev-Mar Begin *******************************/
		double lm_fc[2] = { fc[0], fc[1] };
		double lm_cc[2] = { cc[0], cc[1] };
		double lm_alpha_c = alpha_c;
		double lm_kc[5] = { kc[0], kc[1], kc[2], kc[3], kc[4] };
		ZQ_DImage<int> index_map(nViews, 1);
		ZQ_DImage<double> lm_rT(nViews * 6, 1);
		int*& index_map_data = index_map.data();
		double*& lm_rT_data = lm_rT.data();
		int outer_it = 0;
		while (true)
		{
			int active_num = 0;
			for (int vv = 0; vv < nViews; vv++)
				index_map_data[vv] = -1;
			for (int vv = 0; vv < nViews; vv++)
			{
				if (active_images[vv])
				{
					index_map_data[vv] = active_num++;
				}
			}
			if (active_num < 3)
				return false;
			//
			
			ZQ_DImage<double> lm_X3(active_num*nPts * 3, 1);
			ZQ_DImage<double> lm_X2(active_num*nPts * 2, 1);
			ZQ_DImage<double> lm_rT(active_num * 6, 1);
			double*& lm_X3_data = lm_X3.data();
			double*& lm_X2_data = lm_X2.data();
			
			/*copy in*/
			for (int vv = 0; vv < nViews; vv++)
			{
				int idx = index_map_data[vv];
				if (idx >= 0)
				{
					for (int i = 0; i < nPts * 3; i++)
						lm_X3_data[idx*nPts * 3 + i] = X3[vv*nPts * 3 + i];
					for (int i = 0; i < nPts * 2; i++)
						lm_X2_data[idx*nPts * 2 + i] = X2[vv*nPts * 2 + i];
					for (int i = 0; i < 6; i++)
						lm_rT_data[idx * 6 + i] = rT[vv * 6 + i];
				}
			}

			double avg_err_square = 0;
			// I have found it's important to estimate a good initial focal length
			if (outer_it == 0)
			{
				if (display)
					printf("initialize using method CALIB_F2_C_ALPHA_K5!");
				
				if (!_calib_estimate_LevMar(active_num, nPts, lm_X3_data, lm_X2_data, 30, lm_fc, lm_cc, lm_kc, lm_alpha_c, lm_rT_data, avg_err_square, CALIB_F2_C_ALPHA_K5, zAxis_in, sparse_solver, display))
					return false;
				
				if (display)
					printf("init done!\n");
				for (int vv = 0; vv < active_num; vv++)
				{
					double avg_E = 0;
					//if (display)
					//	printf("posit %d,", vv);
					
					PositCoplanarRobust(nPts, lm_X3_data + nPts * 3 * vv, lm_X2_data + nPts * 2 * vv, lm_fc, lm_cc, lm_kc, lm_alpha_c, 10, 20, 1, lm_rT_data + vv * 6, avg_E, zAxis_in);
					//if (display)
					//	printf("done!\n");
				}

				memset(lm_kc, 0, sizeof(T)* 5);
				lm_alpha_c = 0;
			}
			
			if (!_calib_estimate_LevMar(active_num, nPts, lm_X3_data, lm_X2_data, 30, lm_fc, lm_cc, lm_kc, lm_alpha_c, lm_rT_data, avg_err_square, method, zAxis_in, sparse_solver, display))
				return false;

			/*copy out*/
			fc[0] = lm_fc[0]; fc[1] = lm_fc[1];
			cc[0] = lm_cc[0]; cc[1] = lm_cc[1];
			alpha_c = lm_alpha_c;
			kc[0] = lm_kc[0]; kc[1] = lm_kc[1]; kc[2] = lm_kc[2]; kc[3] = lm_kc[3]; kc[4] = lm_kc[4];
			for (int vv = 0; vv < nViews; vv++)
			{
				int idx = index_map_data[vv];
				if (idx >= 0)
				{
					for (int i = 0; i < 6; i++)
						rT[vv * 6 + i] = lm_rT_data[idx * 6 + i];
				}
			}
			/* check convergence */
			bool has_converged = true;
			for (int vv = 0; vv < nViews; vv++)
			{
				int idx = index_map_data[vv];
				if (idx < 0)
					continue;
				
				double sigma_x = 0, max_err = 0;
				if (!_compute_err_calib(1, nPts, lm_X2_data+nPts*2*idx, lm_X3_data+nPts*3*idx, lm_fc, lm_cc, lm_kc, lm_alpha_c, lm_rT_data+6*idx, active_images+vv, sigma_x, max_err,zAxis_in))
					return false;
				if (max_err > tol_E)
				{
					active_images[vv] = false;
					has_converged = false;
				}
			}
			if (has_converged)
				break;
			outer_it++;
		}

		/**************************** Lev-Mar End *******************************/

		//-------------------------- - Computation of the error of estimation :
		if (display)
			printf("Estimation of uncertainties...\n");
		int unknown_all_num = 10 + 6 * nViews;
		ZQ_Matrix<double> JJ3(unknown_all_num, unknown_all_num);
		ZQ_Matrix<double> ex3(unknown_all_num, 1);

		double sigma_x = 0, max_err = 0;
		if (!_compute_err_calib(nViews, nPts, X2, X3, fc, cc, kc, alpha_c, rT, active_images, sigma_x, max_err,zAxis_in))
			return false;
		if (!_compute_JJ3_ex3(nViews, nPts, X2, X3, fc, cc, kc, alpha_c, rT, active_images, JJ3, ex3, thresh_cond, false, zAxis_in))
			return false;

		T fc_err[2];
		T cc_err[2];
		T kc_err[5];
		T alpha_err;
		if (!_estimate_uncertainties(nViews, active_images, JJ3, sigma_x * 3, fc_err, cc_err, kc_err, alpha_err))
			return false;

		double m_pi = atan(1.0) * 4;
		double degree_per_pi = 180.0 / m_pi;
		if (display)
		{
			printf("Calibration results after optimization (with uncertainties):\n");
			printf("Focal Length:     fc = [ %3.5f   %3.5f ] <> [ %3.5f   %3.5f ]\n", fc[0], fc[1], fc_err[0], fc_err[1]);
			printf("Principal point:  cc = [ %3.5f   %3.5f ] <> [ %3.5f   %3.5f ]\n", cc[0], cc[1], cc_err[0], cc_err[1]);
			printf("Skew:        alpha_c = [ %3.5f ] <> [ %3.5f  ]\n", alpha_c, alpha_err);
			printf("Distortion:       kc = [ %3.5f   %3.5f   %3.5f   %3.5f  %5.5f ] <> [ %3.5f   %3.5f   %3.5f   %3.5f  %5.5f ]\n", kc[0], kc[1], kc[2], kc[3], kc[4], kc_err[0], kc_err[1], kc_err[2], kc_err[3], kc_err[4]);
			printf("Pixel error: err_std = [ %3.5f]\n", sigma_x);
			printf("Pixel error: err_max = [ %3.5f]\n", max_err);
			printf("Note: The numerical errors are approximately three times the standard deviations (for reference).\n");
		}

		return true;
	}

	template<class T>
	bool ZQ_CameraCalibration::_compute_extrinsic_init(int nPts, const T* X2, const T* X3, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, T rT[6], bool zAxis_in)
	{
		/*
		%	Computes the extrinsic parameters attached to a 3D structure X3 given its projection
		%	on the image plane	X2 and the intrinsic camera parameters fc, cc and kc.
		%	Works with planar and non-planar structures.
		%
		%	INPUT: 
		%		X2: Feature locations on the images
		%       X3: Corresponding grid coordinates
		%       fc: Camera focal length
		%       cc: Principal point coordinates
		%       kc: Distortion coefficients
		%       alpha_c: Skew coefficient
		%
		%	OUTPUT: 
		%		rT: 3D rotation vector attached to the grid positions in space
		%			and 3D translation vector attached to the grid positions in space
		%
		%	Method: Computes the normalized point coordinates, then computes the 3D pose
		%
		%	Important functions called within that program:
		%
		%	normalize_pixel: Computes the normalize image point coordinates.
		%
		%	pose3D: Computes the 3D pose of the structure given the normalized image projection.
		%
		%	project_points: Computes the 2D image projections of a set of 3D points
		*/

		const double eps = 1e-16;
		
		// Compute the normalized coordinates :
		ZQ_DImage<T> Xn_mat(nPts * 2, 1);
		ZQ_DImage<T> Y_mat(nPts * 3, 1);
		T*& Xn = Xn_mat.data();
		T*& Y = Y_mat.data();
		normalize_pixels(nPts, X2, fc, cc, kc, alpha_c, Xn);
		
		// Check for planarity of the structure :
		double X_mean[3] = { 0, 0, 0 };
		for (int i = 0; i < nPts; i++)
		{
			X_mean[0] += X3[i * 3 + 0];
			X_mean[1] += X3[i * 3 + 1];
			X_mean[2] += X3[i * 3 + 2];
		}
		X_mean[0] /= nPts;
		X_mean[1] /= nPts;
		X_mean[2] /= nPts;

		
		for (int i = 0; i < nPts; i++)
		{
			Y[i * 3 + 0] = X3[i * 3 + 0] - X_mean[0];
			Y[i * 3 + 1] = X3[i * 3 + 1] - X_mean[1];
			Y[i * 3 + 2] = X3[i * 3 + 2] - X_mean[2];
		}

		double YY[9] = { 0 }, U[9], S[9], V[9];
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				for (int k = 0; k < nPts; k++)
				{
					YY[i * 3 + j] += Y[k * 3 + i] * Y[k * 3 + j];
				}
			}
		}

		if (!ZQ_MathBase::SVD_Decompose(YY, 3, 3, U, S, V))
		{
			return false;
		}
		
		double r = S[8] / S[4];

		if (r < 1e-3 || nPts < 5) // 1e-3, 1e-4, norm(X3(3, :)) < eps,  Test of planarity
		{
			double R_transform[9] =
			{
				V[0], V[3], V[6],
				V[1], V[4], V[7],
				V[2], V[5], V[8]
			};

			if (R_transform[2] * R_transform[2] + R_transform[5] * R_transform[5] < 1e-12) //norm(R_transform(1:2, 3)) < 1e-6,
			{
				R_transform[0] = R_transform[4] = R_transform[8] = 1;
				R_transform[1] = R_transform[2] = R_transform[3] = R_transform[5] = R_transform[6] = R_transform[7] = 0;
			}

			if (ZQ_MathBase::Det(3, R_transform) < 0)
			{
				for (int i = 0; i < 9; i++)
					R_transform[i] = -R_transform[i];
			}

			double T_transform[3];
			ZQ_MathBase::MatrixMul(R_transform, X_mean, 3, 3, 1, T_transform);
			for (int i = 0; i < 3; i++)
				T_transform[i] = -T_transform[i];

			ZQ_DImage<T> X3_new_mat(nPts, 3);
			T*& X3_new = X3_new_mat.data();

			for (int i = 0; i < nPts; i++)
			{
				X3_new[i * 3 + 0] = R_transform[0] * X3[i * 3 + 0] + R_transform[1] * X3[i * 3 + 1] + R_transform[2] * X3[i * 3 + 2] + T_transform[0];
				X3_new[i * 3 + 1] = R_transform[3] * X3[i * 3 + 0] + R_transform[4] * X3[i * 3 + 1] + R_transform[5] * X3[i * 3 + 2] + T_transform[1];
				X3_new[i * 3 + 2] = R_transform[6] * X3[i * 3 + 0] + R_transform[7] * X3[i * 3 + 1] + R_transform[8] * X3[i * 3 + 2] + T_transform[2];
			}


			// Compute the planar homography :
			ZQ_DImage<T> X3_new_row12_mat(nPts * 2, 1);
			T*& X3_new_row12 = X3_new_row12_mat.data();
			for (int i = 0; i < nPts; i++)
			{
				X3_new_row12[i * 2 + 0] = X3_new[i * 3 + 0];
				X3_new_row12[i * 2 + 1] = X3_new[i * 3 + 1];
			}

			T H[9], Hnorm[9], inv_Hnorm[9];
			bool suc_flag = _compute_homography(nPts, Xn, X3_new_row12, H, Hnorm, inv_Hnorm);
			
			if (!suc_flag)
			{
				return false;
			}

			if (!zAxis_in)
			{
				for (int i = 6; i < 9; i++)
					H[i] = -H[i];
			}

			double sc = 0.5*(sqrt((double)H[0] * H[0] + H[3] * H[3] + H[6] * H[6]) + sqrt((double)H[1] * H[1] + H[4] * H[4] + H[7] * H[7]));
			double inv_sc = (sc == 0) ? (1.0 / (eps*eps)) : (1.0 / sc);
			for (int i = 0; i < 9; i++)
				H[i] *= inv_sc;

			double u1[3] = { H[0], H[3], H[6] };
			double u1_nrm = sqrt(u1[0] * u1[0] + u1[1] * u1[1] + u1[2] * u1[2]);
			if (u1_nrm == 0)
			{
				u1[0] = 1;
				u1[1] = u1[2] = 0;
			}
			else
			{
				u1[0] /= u1_nrm;
				u1[1] /= u1_nrm;
				u1[2] /= u1_nrm;
			}
			
			double dot_u1_H2 = u1[0] * H[1] + u1[1] * H[4] + u1[2] * H[7];
			double u2[3] = { H[1] - dot_u1_H2*u1[0], H[4] - dot_u1_H2*u1[1], H[7] - dot_u1_H2*u1[2] };
			double u2_nrm = sqrt(u2[0] * u2[0] + u2[1] * u2[1] + u2[2] * u2[2]);
			if (u2_nrm == 0)
			{
				u2[0] = 0;
				u2[1] = 1;
				u2[2] = 0;
			}
			else
			{
				u2[0] /= u2_nrm;
				u2[1] /= u2_nrm;
				u2[2] /= u2_nrm;
			}
			

			double u3[3] =
			{
				u1[1] * u2[2] - u1[2] * u2[1],
				u1[2] * u2[0] - u1[0] * u2[2],
				u1[0] * u2[1] - u1[1] * u2[0]
			};

			double RRR[9] = {
				u1[0], u2[0], u3[0],
				u1[1], u2[1], u3[1],
				u1[2], u2[2], u3[2]
			};
			double rrr[3];
			if (!ZQ_Rodrigues::ZQ_Rodrigues_R2r(RRR, rrr))
			{
				return false;
			}
			ZQ_Rodrigues::ZQ_Rodrigues_r2R(rrr, RRR);
			double ttt[3] = { H[2], H[5], H[8] };
			
			for (int i = 0; i < 3; i++)
			{
				rT[i + 3] = ttt[i] + RRR[i * 3 + 0] * T_transform[0] + RRR[i * 3 + 1] * T_transform[1] + RRR[i * 3 + 2] * T_transform[2];
			}
			double RRR2[9] = { 0 };
			ZQ_MathBase::MatrixMul(RRR, R_transform, 3, 3, 3, RRR2);
			if (!ZQ_Rodrigues::ZQ_Rodrigues_R2r(RRR2, rrr))
				return false;
			ZQ_Rodrigues::ZQ_Rodrigues_r2R(rrr, RRR2);
			rT[0] = rrr[0];
			rT[1] = rrr[1];
			rT[2] = rrr[2];
		}
		else //Non planar structure
		{
			// Computes an initial guess for extrinsic parameters(works for general 3d structure, not planar!!!) :
			// The DLT method is applied here!!

			ZQ_DImage<double> J_mat(2 * nPts, 12);
			double*& J = J_mat.data();
			
			for (int i = 0; i < nPts; i++)
			{
				double xX[3] = { Xn[i * 2 + 0] * X3[i * 3 + 0], Xn[i * 2 + 0] * X3[i * 3 + 1], Xn[i * 2 + 0] * X3[i * 3 + 2] };
				double yX[3] = { Xn[i * 2 + 1] * X3[i * 3 + 0], Xn[i * 2 + 1] * X3[i * 3 + 1], Xn[i * 2 + 1] * X3[i * 3 + 2] };
				J[(i * 2 + 0) * 12 + 0] = -X3[i * 3 + 0];
				J[(i * 2 + 0) * 12 + 3] = -X3[i * 3 + 1];
				J[(i * 2 + 0) * 12 + 6] = -X3[i * 3 + 2];
				J[(i * 2 + 1) * 12 + 1] = X3[i * 3 + 0];
				J[(i * 2 + 1) * 12 + 4] = X3[i * 3 + 1];
				J[(i * 2 + 1) * 12 + 7] = X3[i * 3 + 2];
				J[(i * 2 + 0) * 12 + 2] = xX[0];
				J[(i * 2 + 0) * 12 + 5] = xX[1];
				J[(i * 2 + 0) * 12 + 8] = xX[2];
				J[(i * 2 + 1) * 12 + 2] = -yX[0];
				J[(i * 2 + 1) * 12 + 5] = -yX[1];
				J[(i * 2 + 1) * 12 + 8] = -yX[2];
				J[(i * 2 + 0) * 12 + 11] = Xn[i * 2 + 0];
				J[(i * 2 + 1) * 12 + 11] = -Xn[i * 2 + 1];
				J[(i * 2 + 0) * 12 + 9] = -1;
				J[(i * 2 + 1) * 12 + 10] = 1;
			}

			ZQ_DImage<double> JJ_mat(12, 12);
			double*& JJ = JJ_mat.data();
			for (int i = 0; i < 12; i++)
			{
				for (int j = 0; j < 12; j++)
				{
					JJ[i * 12 + j] = 0;
					for (int k = 0; k < 2 * nPts; k++)
						JJ[i * 12 + j] += J[k * 12 + i] * J[k * 12 + j];
				}
			}
			ZQ_DImage<double> U_mat(12, 12), S_mat(12, 12), V_mat(12, 12);
			double*& U = U_mat.data();
			double*& S = S_mat.data();
			double*& V = V_mat.data();
			bool suc_flag = ZQ_MathBase::SVD_Decompose(JJ, 12, 12, U, S, V);
			if (!suc_flag)
			{
				return false;
			}

			double RR[9] = 
			{
				V[0 * 12 + 11], V[3 * 12 + 11], V[6 * 12 + 11],
				V[1 * 12 + 11], V[4 * 12 + 11], V[7 * 12 + 11],
				V[2 * 12 + 11], V[5 * 12 + 11], V[8 * 12 + 11]
			};

			if (ZQ_MathBase::Det(3, RR) < 0)
			{
				for (int i = 0; i < 12; i++)
					V[i * 12 + 11] = -V[i * 12 + 11];
				for (int i = 0; i < 9; i++)
					RR[i] = -RR[i];
			}

			double Ur[9], Sr[9], Vr[9];
			if(!ZQ_MathBase::SVD_Decompose(RR, 3, 3, Ur, Sr, Vr))
			{
				return false;
			}

			double Rckk[9] = { 0 };
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					for (int k = 0; k < 3; k++)
					{
						Rckk[i * 3 + j] += Ur[i * 3 + k] * Vr[j * 3 + k];
					}
				}
			}

			double len_v = 0;
			for (int i = 0; i < 9; i++)
				len_v += V[i * 12 + 11] * V[i * 12 + 11];
			len_v = sqrt(len_v);
			double len_rckk = 0;
			for (int i = 0; i < 9; i++)
				len_rckk += Rckk[i] * Rckk[i];
			len_rckk = sqrt(len_rckk);

			double sc = len_v / len_rckk;
			double inv_sc = (sc == 0) ? (1.0 / (eps*eps)) : (1.0 / sc);
			for (int i = 0; i < 3; i++)
			{
				rT[i + 3] = V[(i + 9) * 12 + 11] * inv_sc;
			}

			if (!zAxis_in)
			{
				Rckk[2] = -Rckk[2];
				Rckk[5] = -Rckk[5];
				Rckk[6] = -Rckk[6];
				Rckk[7] = -Rckk[7];
				rT[5] = -rT[5];
			}
			double rrr[3];
			if(!ZQ_Rodrigues::ZQ_Rodrigues_R2r(Rckk, rrr))
			{
				return false;
			}

			ZQ_Rodrigues::ZQ_Rodrigues_r2R(rrr, Rckk);
			
			for (int i = 0; i < 3; i++)
				rT[i] = rrr[i];
		}
		return true;
	}

	template<class T>
	bool ZQ_CameraCalibration::_compute_extrinsic_refine(int nPts, const T* rT_init, const T* X2, const T* X3, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, T rT[6], double* JJ, bool zAxis_in, int max_iter /* = 20*/, double thresh_cond /* = 1e16 */)
	{
		/*
		%	[rT,R, JJ] = compute_extrinsic_refine(rT_init,X2,X3,fc,cc,kc,alpha_c,max_iter)
		%
		%	Computes the extrinsic parameters attached to a 3D structure X_kk given its projection
		%	on the image plane x_kk and the intrinsic camera parameters fc, cc and kc.
		%	Works with planar and non-planar structures.
		%
		%	INPUT: 
		%		X2: Feature locations on the images
		%       X3: Corresponding grid coordinates
		%       fc: Camera focal length
		%       cc: Principal point coordinates
		%       kc: Distortion coefficients
		%       alpha_c: Skew coefficient
		%       max_iter: Maximum number of iterations
		%
		%	OUTPUT: 
		%		rT: 3D rotation-translation vector attached to the grid positions in space
		%
		%	Method: Computes the normalized point coordinates, then computes the 3D pose
		%
		%	Important functions called within that program:
		%
		%	normalize_pixel: Computes the normalize image point coordinates.
		%
		%	pose3D: Computes the 3D pose of the structure given the normalized image projection.
		%
		%	project_points.m: Computes the 2D image projections of a set of 3D points
		*/

		
		//	thresh_cond = inf;
		// max_iter = 20;


		// Initialization:
		memcpy(rT, rT_init, sizeof(T)* 6);


		//// Final optimization(minimize the reprojection error in pixel) : through Gradient Descent :
		ZQ_DImage<T> x_im_mat(nPts * 2, 1);
		ZQ_DImage<T> ex_im_mat(nPts * 2, 1);
		ZQ_DImage<T> dxdrT_im_mat(2 * nPts, 6);
		T*& x = x_im_mat.data();
		T*& ex = ex_im_mat.data();
		T*& dxdrT = dxdrT_im_mat.data();
		
		ZQ_Matrix<double> JJmat(2 * nPts, 6);
		ZQ_Matrix<double> exmat(2 * nPts, 1);
		double change = 1;
		int iter = 0;
		while (change > 1e-10 && iter < max_iter)
		{
			if (!project_points_fun(nPts, X3, rT, fc, cc, kc, alpha_c, x, zAxis_in)
				|| !project_points_jac(nPts, X3, rT, fc, cc, kc, alpha_c, dxdrT, (T*)0, (T*)0, (T*)0, (T*)0, zAxis_in))
			{
				return false;
			}
			ZQ_Matrix<double> para(6, 1);
			ZQ_MathBase::VecMinus(nPts * 2, X2, x, ex);
			for (int i = 0; i < nPts * 2; i++)
			{
				JJmat.SetData(i, 0, dxdrT[i * 6 + 0]);
				JJmat.SetData(i, 1, dxdrT[i * 6 + 1]);
				JJmat.SetData(i, 2, dxdrT[i * 6 + 2]);
				JJmat.SetData(i, 3, dxdrT[i * 6 + 3]);
				JJmat.SetData(i, 4, dxdrT[i * 6 + 4]);
				JJmat.SetData(i, 5, dxdrT[i * 6 + 5]);
				exmat.SetData(i, 0, ex[i]);
			}
			memcpy(JJ, JJmat.GetDataPtr(), sizeof(double)* 2 * nPts * 6);
			bool cond_succ = true, is_singular = false;
			double cond_num = ZQ_MathBase::Cond_by_double_svd(JJmat.GetDataPtr(), 2 * nPts, 6, cond_succ, is_singular);
			if (!cond_succ)
			{
				return false;
			}
			if (is_singular || cond_num > thresh_cond)
			{
				change = 0;
				break;
			}

			if (!ZQ_SVD::Solve(JJmat, para, exmat))
			{
				return false;
			}

			T para_innov[6] = { 0 };
			double len_innov = 0;
			for (int i = 0; i < 6; i++)
			{
				bool flag;
				para_innov[i] = para.GetData(i, 0, flag);
				len_innov += para_innov[i] * para_innov[i];
			}
			for (int i = 0; i < 6; i++)
			{
				rT[i] += para_innov[i];
			}
			len_innov = sqrt(len_innov);
			double len_rT = 0;
			for (int i = 0; i < 6; i++)
			{
				len_rT += rT[i] * rT[i];
			}
			len_rT = sqrt(len_rT);
			change = len_innov / (len_rT + 1e-32);
			iter++;
		}
		return true;
	}
	
	template<class T>
	bool ZQ_CameraCalibration::_compute_homography(int nPts, const T* m, const T* M, T H[9], T Hnorm[9], T inv_Hnorm[9])
	{
		/*	
		%	Computes the planar homography between the point coordinates on the plane (m) and the image
		%	point coordinates (M).
		%
		%	INPUT: m: homogeneous coordinates in the image plane (N*2 matrix)
		%		   M: homogeneous coordinates in the plane in 3D (N*2 matrix)
		%
		%	OUTPUT: H: Homography matrix (3x3 homogeneous matrix)
		%			Hnorm: Normalization matrix used on the points before homography computation
		%               (useful for numerical stability is points in pixel coordinates)
		%			inv_Hnorm: The inverse of Hnorm
		%
		%	Definition: m ~ H*M where "~" means equal up to a non zero scalar factor.
		%
		%	Method: First computes an initial guess for the homography through quasi-linear method.
		%			Then, if the total number of points is larger than 4, optimize the solution by minimizing
		%			the reprojection error (in the least squares sense).
		%
		%
		%	Important functions called within that program:
		%
		%	undistort_point_oulu: Undistorts pixel coordinates.
		%
		%	compute_homography.m: Computes the planar homography between points on the grid in 3D, and the image plane.
		%
		%	project_points.m: Computes the 2D image projections of a set of 3D points, and also returns te Jacobian
		%                  matrix (derivative with respect to the intrinsic and extrinsic parameters).
		%                  This function is called within the minimization loop.

		*/

		const double eps = 1e-16;

		// Prenormalization of point coordinates(very important) : (Affine normalization)
		T mxx = 0;
		T myy = 0;
		for (int i = 0; i < nPts; i++)
		{
			mxx += m[i * 2 + 0];
			myy += m[i * 2 + 1];
		}
		mxx /= nPts;
		myy /= nPts;

		T scxx = 0;
		T scyy = 0;
		for (int i = 0; i < nPts; i++)
		{
			scxx += fabs(m[i * 2 + 0] - mxx);
			scyy += fabs(m[i * 2 + 1] - myy);
		}

		double inv_scxx = (scxx == 0) ? (1.0 / (eps*eps)) : (1.0 / scxx);
		double inv_scyy = (scyy == 0) ? (1.0 / (eps*eps)) : (1.0 / scyy);
		Hnorm[0] = inv_scxx;	Hnorm[1] = 0;			Hnorm[2] = -mxx *inv_scxx;
		Hnorm[3] = 0;			Hnorm[4] = inv_scyy;	Hnorm[5] = -myy *inv_scyy;
		Hnorm[6] = 0;			Hnorm[7] = 0;			Hnorm[8] = 1;
		
		inv_Hnorm[0] = scxx;	inv_Hnorm[1] = 0;		inv_Hnorm[2] = mxx;
		inv_Hnorm[3] = 0;		inv_Hnorm[4] = scyy;	inv_Hnorm[5] = myy; 
		inv_Hnorm[6] = 0;		inv_Hnorm[7] = 0;		inv_Hnorm[8] = 1;

		ZQ_DImage<T> mn(nPts * 2, 1);
		T*& mn_data = mn.data();
		for (int i = 0; i < nPts; i++)
		{
			mn_data[i * 2 + 0] = Hnorm[0] * m[i * 2 + 0] + Hnorm[1] * m[i * 2 + 1] + Hnorm[2];
			mn_data[i * 2 + 1] = Hnorm[3] * m[i * 2 + 0] + Hnorm[4] * m[i * 2 + 1] + Hnorm[5];
		}
		
		// Compute the homography between m and mn :

		// Build the matrix :
		ZQ_DImage<double> tmp_L(2 * nPts * 9, 1);
		double*& tmp_L_data = tmp_L.data();
	
		ZQ_DImage<double> V(9, 9);
		double*& V_data = V.data();
		for (int i = 0; i < nPts; i++)
		{
			tmp_L_data[(i * 2 + 0) * 9 + 0] = M[i * 2 + 0];
			tmp_L_data[(i * 2 + 0) * 9 + 1] = M[i * 2 + 1];
			tmp_L_data[(i * 2 + 0) * 9 + 2] = 1;
			tmp_L_data[(i * 2 + 1) * 9 + 3] = M[i * 2 + 0];
			tmp_L_data[(i * 2 + 1) * 9 + 4] = M[i * 2 + 1];
			tmp_L_data[(i * 2 + 1) * 9 + 5] = 1;
			tmp_L_data[(i * 2 + 0) * 9 + 6] = mn_data[i * 2 + 0] * M[i * 2 + 0];
			tmp_L_data[(i * 2 + 0) * 9 + 7] = mn_data[i * 2 + 0] * M[i * 2 + 1];
			tmp_L_data[(i * 2 + 0) * 9 + 8] = mn_data[i * 2 + 0] * 1;
			tmp_L_data[(i * 2 + 1) * 9 + 6] = mn_data[i * 2 + 1] * M[i * 2 + 0];
			tmp_L_data[(i * 2 + 1) * 9 + 7] = mn_data[i * 2 + 1] * M[i * 2 + 1];
			tmp_L_data[(i * 2 + 1) * 9 + 8] = mn_data[i * 2 + 1] * 1;
		}
	
		bool svd_flag = false;
		if (nPts > 4)
		{
			ZQ_DImage<double> L(9, 9);
			double*& L_data = L.data();
			for (int i = 0; i < 9; i++)
			{
				for (int j = 0; j < 9; j++)
				{
					L_data[i * 9 + j] = 0;
					for (int k = 0; k < 2 * nPts; k++)
					{
						L_data[i * 9 + j] += tmp_L_data[k * 9 + i] * tmp_L_data[k * 9 + j];
					}
				}
			}
			ZQ_DImage<double> U(9, 9);
			ZQ_DImage<double> S(9, 9);
			double*& U_data = U.data();
			double*& S_data = S.data();
			svd_flag = ZQ_MathBase::SVD_Decompose(L_data, 9, 9, U_data, S_data, V_data);
		}
		else
		{
			ZQ_DImage<double> U(2 * nPts, 2 * nPts);
			ZQ_DImage<double> S(2 * nPts, 9);
			double*& U_data = U.data();
			double*& S_data = S.data();
			svd_flag = ZQ_MathBase::SVD_Decompose(tmp_L_data, 2 * nPts, 9, U_data, S_data, V_data);
		}

		if (!svd_flag)
		{
			return false;
		}

		T Hrem[9];
		double inv_V99 = (V_data[8 * 9 + 8] == 0) ? (1.0 / (eps*eps)) : (1.0 / V_data[8 * 9 + 8]);
		for (int i = 0; i < 9; i++)
			Hrem[i] = V_data[i * 9 + 8] *inv_V99;
		

		// Final homography :
		ZQ_MathBase::MatrixMul(inv_Hnorm, Hrem, 3, 3, 3, H);

		// Homography refinement if there are more than 4 points:
		if (nPts > 4)
		{
			double hhv[8] = { H[0], H[1], H[2], H[3], H[4], H[5], H[6], H[7] };
			ZQ_DImage<double> J(2 * nPts, 8);
			ZQ_DImage<double> merr(2 * nPts, 1);
			double*& J_data = J.data();
			double*& merr_data = merr.data();
			
			ZQ_Matrix<double> Jmat(nPts * 2, 8), xmat(8, 1), bmat(nPts * 2, 1);
			for (int iter = 0; iter < 10; iter++)
			{
				double mrep[3], MMM[3];
				for (int i = 0; i < nPts; i++)
				{
					mrep[0] = H[0] * M[i * 2 + 0] + H[1] * M[i * 2 + 1] + H[2];
					mrep[1] = H[3] * M[i * 2 + 0] + H[4] * M[i * 2 + 1] + H[5];
					mrep[2] = H[6] * M[i * 2 + 0] + H[7] * M[i * 2 + 1] + H[8];

					double inv_mrep2 = (mrep[2] == 0) ? (1.0 / (eps*eps)) : (1.0 / mrep[2]);
					MMM[0] = M[i * 2 + 0] * inv_mrep2;
					MMM[1] = M[i * 2 + 1] * inv_mrep2;
					MMM[2] = inv_mrep2;

					J_data[(i * 2 + 0) * 8 + 0] = -MMM[0];
					J_data[(i * 2 + 0) * 8 + 1] = -MMM[1];
					J_data[(i * 2 + 0) * 8 + 2] = -MMM[2];
					J_data[(i * 2 + 1) * 8 + 3] = -MMM[0];
					J_data[(i * 2 + 1) * 8 + 4] = -MMM[1];
					J_data[(i * 2 + 1) * 8 + 5] = -MMM[2];

					mrep[0] *= inv_mrep2;
					mrep[1] *= inv_mrep2;
					mrep[2] = 1;

					J_data[(i * 2 + 0) * 8 + 6] = mrep[0] * MMM[0];
					J_data[(i * 2 + 0) * 8 + 7] = mrep[0] * MMM[1];
					J_data[(i * 2 + 1) * 8 + 6] = mrep[1] * MMM[0];
					J_data[(i * 2 + 1) * 8 + 7] = mrep[1] * MMM[1];

					merr_data[i * 2 + 0] = m[i * 2 + 0] - mrep[0];
					merr_data[i * 2 + 1] = m[i * 2 + 1] - mrep[1];
				}

				double hh_innov[8];
				for (int i = 0; i < nPts * 2; i++)
				{
					for (int j = 0; j < 8; j++)
					{
						Jmat.SetData(i, j, J_data[i * 8 + j]);
					}
					bmat.SetData(i, 0, merr_data[i]);
				}
				if (!ZQ_SVD::Solve(Jmat, xmat, bmat))
				{
					return false;
				}

				bool flag;
				for (int i = 0; i < 8; i++)
					hh_innov[i] = xmat.GetData(i, 0, flag);

				for (int i = 0; i < 8; i++)
				{
					hhv[i] -= hh_innov[i];
					H[i] = hhv[i];
				}
				H[8] = 1;
			}
		}
		return true;
	}

	/****************************************************************************************************/

	/*calibrate camera:
	based on Lev-Mar optimization, need good initialization.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_CameraCalibration::_calib_estimate_fun(const T* p, T* hx, int m, int n, const void* data)
	{
		const Calib_Data_Header<T>* ptr = (const Calib_Data_Header<T>*)data;
		const T* X3 = ptr->X3;
		const T* X2 = ptr->X2;
		const int nPts = ptr->n_pts;
		const int nViews = ptr->n_views;
		Calib_Method method = ptr->method;
		bool zAxis_in = ptr->zAxis_in;
		T fc[2], cc[2], alpha_c, kc[5];
		const T* rT = 0;
		

		switch (method)
		{
		case CALIB_F2_C_ALPHA_K5:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = p[4];
			kc[0] = p[5]; kc[1] = p[6]; kc[2] = p[7]; kc[3] = p[8]; kc[4] = p[9];
			rT = p + 10;
			break;
		case CALIB_F2_C_ALPHA_K4:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = p[4];
			kc[0] = p[5]; kc[1] = p[6]; kc[2] = p[7]; kc[3] = p[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 9;
			break;
		case CALIB_F2_C_ALPHA_K2:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = p[4];
			kc[0] = p[5]; kc[1] = p[6]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 7;
			break;
		case CALIB_F2_C_ALPHA:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = p[4];
			kc[0] = ptr->fc_cc_alpha_kc[5]; kc[1] = ptr->fc_cc_alpha_kc[6]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 5;
			break;
		case CALIB_F2_C_K5:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = p[4]; kc[1] = p[5]; kc[2] = p[6]; kc[3] = p[7]; kc[4] = p[8];
			rT = p + 9;
			break;
		case CALIB_F2_C_K4:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = p[4]; kc[1] = p[5]; kc[2] = p[6]; kc[3] = p[7]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 8;
			break;
		case CALIB_F2_C_K2:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = p[4]; kc[1] = p[5]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 6;
			break;
		case CALIB_F2_C:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = ptr->fc_cc_alpha_kc[5]; kc[1] = ptr->fc_cc_alpha_kc[6]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 4;
			break;
		case CALIB_F1_C_ALPHA_K5:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = p[3];
			kc[0] = p[4]; kc[1] = p[5]; kc[2] = p[6]; kc[3] = p[7]; kc[4] = p[8];
			rT = p + 9;
			break;
		case CALIB_F1_C_ALPHA_K4:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = p[3];
			kc[0] = p[4]; kc[1] = p[5]; kc[2] = p[6]; kc[3] = p[7]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 8;
			break;
		case CALIB_F1_C_ALPHA_K2:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = p[3];
			kc[0] = p[4]; kc[1] = p[5]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 6;
			break;
		case CALIB_F1_C_ALPHA:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = p[3];
			kc[0] = ptr->fc_cc_alpha_kc[5]; kc[1] = ptr->fc_cc_alpha_kc[6]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 4;
			break;
		case CALIB_F1_C_K5:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = p[3]; kc[1] = p[4]; kc[2] = p[5]; kc[3] = p[6]; kc[4] = p[7];
			rT = p + 8;
			break;
		case CALIB_F1_C_K4:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = p[3]; kc[1] = p[4]; kc[2] = p[5]; kc[3] = p[6]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 7;
			break;
		case CALIB_F1_C_K2:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = p[3]; kc[1] = p[4]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 5;
			break;
		case CALIB_F1_C:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = ptr->fc_cc_alpha_kc[5]; kc[1] = ptr->fc_cc_alpha_kc[6]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 3;
			break;
		default:
			return false;
			break;
		}

		ZQ_DImage<T> tmp_X2(nPts * 2, 1);
		T*& tmp_X2_data = tmp_X2.data();
		for (int vv = 0; vv < nViews; vv++)
		{
			if (!project_points_fun(nPts, X3 + vv*nPts * 3, rT + vv * 6, fc, cc, kc, alpha_c, tmp_X2_data,zAxis_in))
				return false;
			ZQ_MathBase::VecMinus(nPts * 2, tmp_X2_data, X2 + nPts * 2 * vv, hx + nPts * 2 * vv);
		}
		return true;
	}

	template<class T>
	bool ZQ_CameraCalibration::_calib_estimate_jac(const T* p, T* jx, int m, int n, const void* data)
	{
		const Calib_Data_Header<T>* ptr = (const Calib_Data_Header<T>*)data;
		const T* X3 = ptr->X3;
		const T* X2 = ptr->X2;
		const int nPts = ptr->n_pts;
		const int nViews = ptr->n_views;
		Calib_Method method = ptr->method;
		bool zAxis_in = ptr->zAxis_in;
		T fc[2], cc[2], alpha_c, kc[5];
		const T* rT = 0;
		
		switch (method)
		{
		case CALIB_F2_C_ALPHA_K5:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = p[4];
			kc[0] = p[5]; kc[1] = p[6]; kc[2] = p[7]; kc[3] = p[8]; kc[4] = p[9];
			rT = p + 10;
			break;
		case CALIB_F2_C_ALPHA_K4:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = p[4];
			kc[0] = p[5]; kc[1] = p[6]; kc[2] = p[7]; kc[3] = p[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 9;
			break;
		case CALIB_F2_C_ALPHA_K2:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = p[4];
			kc[0] = p[5]; kc[1] = p[6]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 7;
			break;
		case CALIB_F2_C_ALPHA:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = p[4];
			kc[0] = ptr->fc_cc_alpha_kc[5]; kc[1] = ptr->fc_cc_alpha_kc[6]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 5;
			break;
		case CALIB_F2_C_K5:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = p[4]; kc[1] = p[5]; kc[2] = p[6]; kc[3] = p[7]; kc[4] = p[8];
			rT = p + 9;
			break;
		case CALIB_F2_C_K4:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = p[4]; kc[1] = p[5]; kc[2] = p[6]; kc[3] = p[7]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 8;
			break;
		case CALIB_F2_C_K2:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = p[4]; kc[1] = p[5]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 6;
			break;
		case CALIB_F2_C:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = ptr->fc_cc_alpha_kc[5]; kc[1] = ptr->fc_cc_alpha_kc[6]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 4;
			break;
		case CALIB_F1_C_ALPHA_K5:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = p[3];
			kc[0] = p[4]; kc[1] = p[5]; kc[2] = p[6]; kc[3] = p[7]; kc[4] = p[8];
			rT = p + 9;
			break;
		case CALIB_F1_C_ALPHA_K4:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = p[3];
			kc[0] = p[4]; kc[1] = p[5]; kc[2] = p[6]; kc[3] = p[7]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 8;
			break;
		case CALIB_F1_C_ALPHA_K2:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = p[3];
			kc[0] = p[4]; kc[1] = p[5]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 6;
			break;
		case CALIB_F1_C_ALPHA:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = p[3];
			kc[0] = ptr->fc_cc_alpha_kc[5]; kc[1] = ptr->fc_cc_alpha_kc[6]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = p[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 4;
			break;
		case CALIB_F1_C_K5:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = p[3]; kc[1] = p[4]; kc[2] = p[5]; kc[3] = p[6]; kc[4] = p[7];
			rT = p + 8;
			break;
		case CALIB_F1_C_K4:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = p[3]; kc[1] = p[4]; kc[2] = p[5]; kc[3] = p[6]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 7;
			break;
		case CALIB_F1_C_K2:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = p[3]; kc[1] = p[4]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 5;
			break;
		case CALIB_F1_C:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = ptr->fc_cc_alpha_kc[5]; kc[1] = ptr->fc_cc_alpha_kc[6]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 3;
			break;
		default:
			return false;
			break;
		}

		ZQ_DImage<T> dxdrT(nPts * 2, 6);
		ZQ_DImage<T> dxdf(nPts * 2, 2);
		ZQ_DImage<T> dxdc(nPts * 2, 2);
		ZQ_DImage<T> dxdalpha(nPts * 2, 1);
		ZQ_DImage<T> dxdk(nPts * 2, 5);

		T*& dxdrT_data = dxdrT.data();
		T*& dxdf_data = dxdf.data();
		T*& dxdc_data = dxdc.data();
		T*& dxdalpha_data = dxdalpha.data();
		T*& dxdk_data = dxdk.data();

		
		memset(jx, 0, sizeof(T)*m*n);
		for (int vv = 0; vv < nViews; vv++)
		{
			if (!project_points_jac(nPts, X3 + nPts * 3 * vv, rT + 6 * vv, fc, cc, kc, alpha_c, dxdrT_data, dxdf_data, dxdc_data, dxdk_data, dxdalpha_data,zAxis_in))
				return false;
			
			int row_off = vv * 2 * nPts;
			switch (method)
			{
			case CALIB_F2_C_ALPHA_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + 0, dxdf_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 2, dxdc_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 4, dxdalpha_data + i, sizeof(T)* 1);
					memcpy(jx + cur_row*m + 5, dxdk_data + i * 5, sizeof(T)* 5);
					memcpy(jx + cur_row*m + 10 + vv * 6, dxdrT_data + i * 6, sizeof(T)* 6);
				}
				break;
			case CALIB_F2_C_ALPHA_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + 0, dxdf_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 2, dxdc_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 4, dxdalpha_data + i, sizeof(T)* 1);
					memcpy(jx + cur_row*m + 5, dxdk_data + i * 5, sizeof(T)* 4);
					memcpy(jx + cur_row*m + 9 + vv * 6, dxdrT_data + i * 6, sizeof(T)* 6);
				}
				break;
			case CALIB_F2_C_ALPHA_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + 0, dxdf_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 2, dxdc_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 4, dxdalpha_data + i, sizeof(T)* 1);
					memcpy(jx + cur_row*m + 5, dxdk_data + i * 5, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 7 + vv * 6, dxdrT_data + i * 6, sizeof(T)* 6);
				}
				break;
			case CALIB_F2_C_ALPHA:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + 0, dxdf_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 2, dxdc_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 4, dxdalpha_data + i, sizeof(T)* 1);
					memcpy(jx + cur_row*m + 5 + vv * 6, dxdrT_data + i * 6, sizeof(T)* 6);
				}
				break;
			case CALIB_F2_C_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + 0, dxdf_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 2, dxdc_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 4, dxdk_data + i * 5, sizeof(T)* 5);
					memcpy(jx + cur_row*m + 9 + vv * 6, dxdrT_data + i * 6, sizeof(T)* 6);
				}
				break;
			case CALIB_F2_C_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + 0, dxdf_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 2, dxdc_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 4, dxdk_data + i * 5, sizeof(T)* 4);
					memcpy(jx + cur_row*m + 8 + vv * 6, dxdrT_data + i * 6, sizeof(T)* 6);
				}
				break;
			case CALIB_F2_C_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + 0, dxdf_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 2, dxdc_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 4, dxdk_data + i * 5, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 6 + vv * 6, dxdrT_data + i * 6, sizeof(T)* 6);
				}
				break;
			case CALIB_F2_C:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					memcpy(jx + cur_row*m + 0, dxdf_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 2, dxdc_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 4 + vv * 6, dxdrT_data + i * 6, sizeof(T)* 6);
				}
				break;
			case CALIB_F1_C_ALPHA_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + 0] = dxdf_data[i * 2 + 0] + dxdf_data[i * 2 + 1];
					memcpy(jx + cur_row*m + 1, dxdc_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 3, dxdalpha_data + i, sizeof(T)* 1);
					memcpy(jx + cur_row*m + 4, dxdk_data + i * 5, sizeof(T)* 5);
					memcpy(jx + cur_row*m + 9 + vv * 6, dxdrT_data + i * 6, sizeof(T)* 6);
				}
				break;
			case CALIB_F1_C_ALPHA_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + 0] = dxdf_data[i * 2 + 0] + dxdf_data[i * 2 + 1];
					memcpy(jx + cur_row*m + 1, dxdc_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 3, dxdalpha_data + i, sizeof(T)* 1);
					memcpy(jx + cur_row*m + 4, dxdk_data + i * 5, sizeof(T)* 4);
					memcpy(jx + cur_row*m + 8 + vv * 6, dxdrT_data + i * 6, sizeof(T)* 6);
				}
				break;
			case CALIB_F1_C_ALPHA_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + 0] = dxdf_data[i * 2 + 0] + dxdf_data[i * 2 + 1];
					memcpy(jx + cur_row*m + 1, dxdc_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 3, dxdalpha_data + i, sizeof(T)* 1);
					memcpy(jx + cur_row*m + 4, dxdk_data + i * 5, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 6 + vv * 6, dxdrT_data + i * 6, sizeof(T)* 6);
				}
				break;
			case CALIB_F1_C_ALPHA:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + 0] = dxdf_data[i * 2 + 0] + dxdf_data[i * 2 + 1];
					memcpy(jx + cur_row*m + 1, dxdc_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 3, dxdalpha_data + i, sizeof(T)* 1);
					memcpy(jx + cur_row*m + 4 + vv * 6, dxdrT_data + i * 6, sizeof(T)* 6);
				}
				break;
			case CALIB_F1_C_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + 0] = dxdf_data[i * 2 + 0] + dxdf_data[i * 2 + 1];
					memcpy(jx + cur_row*m + 1, dxdc_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 3, dxdk_data + i * 5, sizeof(T)* 5);
					memcpy(jx + cur_row*m + 8 + vv * 6, dxdrT_data + i * 6, sizeof(T)* 6);
				}
				break;
			case CALIB_F1_C_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + 0] = dxdf_data[i * 2 + 0] + dxdf_data[i * 2 + 1];
					memcpy(jx + cur_row*m + 1, dxdc_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 3, dxdk_data + i * 5, sizeof(T)* 4);
					memcpy(jx + cur_row*m + 7 + vv * 6, dxdrT_data + i * 6, sizeof(T)* 6);
				}
				break;
			case CALIB_F1_C_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + 0] = dxdf_data[i * 2 + 0] + dxdf_data[i * 2 + 1];
					memcpy(jx + cur_row*m + 1, dxdc_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 3, dxdk_data + i * 5, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 5 + vv * 6, dxdrT_data + i * 6, sizeof(T)* 6);
				}
				break;
			case CALIB_F1_C:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					jx[cur_row*m + 0] = dxdf_data[i * 2 + 0] + dxdf_data[i * 2 + 1];
					memcpy(jx + cur_row*m + 1, dxdc_data + i * 2, sizeof(T)* 2);
					memcpy(jx + cur_row*m + 3 + vv * 6, dxdrT_data + i * 6, sizeof(T)* 6);
				}
				break;
			default:
				return false;
				break;
			}
		}
		return true;
	}

	template<class T>
	bool ZQ_CameraCalibration::_calib_estimate_jac_sparse(const T* p, taucs_ccs_matrix*& jx, int m, int n, const void* data)
	{
		const Calib_Data_Header<T>* ptr = (const Calib_Data_Header<T>*)data;
		const T* X3 = ptr->X3;
		const T* X2 = ptr->X2;
		const int nPts = ptr->n_pts;
		const int nViews = ptr->n_views;
		Calib_Method method = ptr->method;
		bool zAxis_in = ptr->zAxis_in;
		T fc[2], cc[2], alpha_c, kc[5];
		const T* rT = 0;

		switch (method)
		{
		case CALIB_F2_C_ALPHA_K5:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = p[4];
			kc[0] = p[5]; kc[1] = p[6]; kc[2] = p[7]; kc[3] = p[8]; kc[4] = p[9];
			rT = p + 10;
			break;
		case CALIB_F2_C_ALPHA_K4:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = p[4];
			kc[0] = p[5]; kc[1] = p[6]; kc[2] = p[7]; kc[3] = p[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 9;
			break;
		case CALIB_F2_C_ALPHA_K2:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = p[4];
			kc[0] = p[5]; kc[1] = p[6]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 7;
			break;
		case CALIB_F2_C_ALPHA:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = p[4];
			kc[0] = ptr->fc_cc_alpha_kc[5]; kc[1] = ptr->fc_cc_alpha_kc[6]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 5;
			break;
		case CALIB_F2_C_K5:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = p[4]; kc[1] = p[5]; kc[2] = p[6]; kc[3] = p[7]; kc[4] = p[8];
			rT = p + 9;
			break;
		case CALIB_F2_C_K4:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = p[4]; kc[1] = p[5]; kc[2] = p[6]; kc[3] = p[7]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 8;
			break;
		case CALIB_F2_C_K2:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = p[4]; kc[1] = p[5]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 6;
			break;
		case CALIB_F2_C:
			fc[0] = p[0]; fc[1] = p[1];	cc[0] = p[2]; cc[1] = p[3];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = ptr->fc_cc_alpha_kc[5]; kc[1] = ptr->fc_cc_alpha_kc[6]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 4;
			break;
		case CALIB_F1_C_ALPHA_K5:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = p[3];
			kc[0] = p[4]; kc[1] = p[5]; kc[2] = p[6]; kc[3] = p[7]; kc[4] = p[8];
			rT = p + 9;
			break;
		case CALIB_F1_C_ALPHA_K4:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = p[3];
			kc[0] = p[4]; kc[1] = p[5]; kc[2] = p[6]; kc[3] = p[7]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 8;
			break;
		case CALIB_F1_C_ALPHA_K2:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = p[3];
			kc[0] = p[4]; kc[1] = p[5]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 6;
			break;
		case CALIB_F1_C_ALPHA:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = p[3];
			kc[0] = ptr->fc_cc_alpha_kc[5]; kc[1] = ptr->fc_cc_alpha_kc[6]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = p[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 4;
			break;
		case CALIB_F1_C_K5:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = p[3]; kc[1] = p[4]; kc[2] = p[5]; kc[3] = p[6]; kc[4] = p[7];
			rT = p + 8;
			break;
		case CALIB_F1_C_K4:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = p[3]; kc[1] = p[4]; kc[2] = p[5]; kc[3] = p[6]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 7;
			break;
		case CALIB_F1_C_K2:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = p[3]; kc[1] = p[4]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 5;
			break;
		case CALIB_F1_C:
			fc[0] = p[0]; fc[1] = p[0];	cc[0] = p[1]; cc[1] = p[2];	alpha_c = ptr->fc_cc_alpha_kc[4];
			kc[0] = ptr->fc_cc_alpha_kc[5]; kc[1] = ptr->fc_cc_alpha_kc[6]; kc[2] = ptr->fc_cc_alpha_kc[7]; kc[3] = ptr->fc_cc_alpha_kc[8]; kc[4] = ptr->fc_cc_alpha_kc[9];
			rT = p + 3;
			break;
		default:
			return false;
			break;
		}

		ZQ_DImage<T> dxdrT(nPts * 2, 6);
		ZQ_DImage<T> dxdf(nPts * 2, 2);
		ZQ_DImage<T> dxdc(nPts * 2, 2);
		ZQ_DImage<T> dxdalpha(nPts * 2, 1);
		ZQ_DImage<T> dxdk(nPts * 2, 5);

		T*& dxdrT_data = dxdrT.data();
		T*& dxdf_data = dxdf.data();
		T*& dxdc_data = dxdc.data();
		T*& dxdalpha_data = dxdalpha.data();
		T*& dxdk_data = dxdk.data();

		ZQ_SparseMatrix<T> sp_jx_mat(n, m);
		
		for (int vv = 0; vv < nViews; vv++)
		{
			if (!project_points_jac(nPts, X3 + nPts * 3 * vv, rT + 6 * vv, fc, cc, kc, alpha_c, dxdrT_data, dxdf_data, dxdc_data, dxdk_data, dxdalpha_data,zAxis_in))
				return false;

			int row_off = vv * 2 * nPts;
			switch (method)
			{
			case CALIB_F2_C_ALPHA_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j, dxdf_data[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 2, dxdc_data[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, 4, dxdalpha_data[i]);
					for (int j = 0; j < 5; j++)
						sp_jx_mat.AddTo(cur_row, j + 5, dxdk_data[i * 5 + j]);
					for (int j = 0; j < 6; j++)
						sp_jx_mat.AddTo(cur_row, j + vv * 6 + 10, dxdrT_data[i * 6 + j]);
				}
				break;
			case CALIB_F2_C_ALPHA_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j, dxdf_data[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 2, dxdc_data[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, 4, dxdalpha_data[i]);
					for (int j = 0; j < 4; j++)
						sp_jx_mat.AddTo(cur_row, j + 5, dxdk_data[i * 5 + j]);
					for (int j = 0; j < 6; j++)
						sp_jx_mat.AddTo(cur_row, j + vv * 6 + 9, dxdrT_data[i * 6 + j]);
				}
				break;
			case CALIB_F2_C_ALPHA_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j, dxdf_data[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 2, dxdc_data[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, 4, dxdalpha_data[i]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 5, dxdk_data[i * 5 + j]);
					for (int j = 0; j < 6; j++)
						sp_jx_mat.AddTo(cur_row, j + vv * 6 + 7, dxdrT_data[i * 6 + j]);
				}
				break;
			case CALIB_F2_C_ALPHA:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j, dxdf_data[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 2, dxdc_data[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, 4, dxdalpha_data[i]);
					for (int j = 0; j < 6; j++)
						sp_jx_mat.AddTo(cur_row, j + vv * 6 + 5, dxdrT_data[i * 6 + j]);
				}
				break;
			case CALIB_F2_C_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j, dxdf_data[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 2, dxdc_data[i * 2 + j]);
					for (int j = 0; j < 5; j++)
						sp_jx_mat.AddTo(cur_row, j + 4, dxdk_data[i * 5 + j]);
					for (int j = 0; j < 6; j++)
						sp_jx_mat.AddTo(cur_row, j + vv * 6 + 9, dxdrT_data[i * 6 + j]);
				}
				break;
			case CALIB_F2_C_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j, dxdf_data[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 2, dxdc_data[i * 2 + j]);
					for (int j = 0; j < 4; j++)
						sp_jx_mat.AddTo(cur_row, j + 4, dxdk_data[i * 5 + j]);
					for (int j = 0; j < 6; j++)
						sp_jx_mat.AddTo(cur_row, j + vv * 6 + 8, dxdrT_data[i * 6 + j]);
				}
				break;
			case CALIB_F2_C_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j, dxdf_data[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 2, dxdc_data[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 4, dxdk_data[i * 5 + j]);
					for (int j = 0; j < 6; j++)
						sp_jx_mat.AddTo(cur_row, j + vv * 6 + 6, dxdrT_data[i * 6 + j]);
				}
				break;
			case CALIB_F2_C:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j, dxdf_data[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 2, dxdc_data[i * 2 + j]);
					for (int j = 0; j < 6; j++)
						sp_jx_mat.AddTo(cur_row, j + vv * 6 + 4, dxdrT_data[i * 6 + j]);
				}
				break;
			case CALIB_F1_C_ALPHA_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, 0, dxdf_data[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 1, dxdc_data[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, 3, dxdalpha_data[i]);
					for (int j = 0; j < 5; j++)
						sp_jx_mat.AddTo(cur_row, j + 4, dxdk_data[i * 5 + j]);
					for (int j = 0; j < 6; j++)
						sp_jx_mat.AddTo(cur_row, j + vv * 6 + 9, dxdrT_data[i * 6 + j]);
				}
				break;
			case CALIB_F1_C_ALPHA_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, 0, dxdf_data[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 1, dxdc_data[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, 3, dxdalpha_data[i]);
					for (int j = 0; j < 4; j++)
						sp_jx_mat.AddTo(cur_row, j + 4, dxdk_data[i * 5 + j]);
					for (int j = 0; j < 6; j++)
						sp_jx_mat.AddTo(cur_row, j + vv * 6 + 8, dxdrT_data[i * 6 + j]);
				}
				break;
			case CALIB_F1_C_ALPHA_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, 0, dxdf_data[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 1, dxdc_data[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, 3, dxdalpha_data[i]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 4, dxdk_data[i * 5 + j]);
					for (int j = 0; j < 6; j++)
						sp_jx_mat.AddTo(cur_row, j + vv * 6 + 6, dxdrT_data[i * 6 + j]);
				}
				break;
			case CALIB_F1_C_ALPHA:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, 0, dxdf_data[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 1, dxdc_data[i * 2 + j]);
					sp_jx_mat.AddTo(cur_row, 3, dxdalpha_data[i]);
					for (int j = 0; j < 6; j++)
						sp_jx_mat.AddTo(cur_row, j + vv * 6 + 4, dxdrT_data[i * 6 + j]);
				}
				break;
			case CALIB_F1_C_K5:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, 0, dxdf_data[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 1, dxdc_data[i * 2 + j]);
					for (int j = 0; j < 5; j++)
						sp_jx_mat.AddTo(cur_row, j + 3, dxdk_data[i * 5 + j]);
					for (int j = 0; j < 6; j++)
						sp_jx_mat.AddTo(cur_row, j + vv * 6 + 8, dxdrT_data[i * 6 + j]);
				}
				break;
			case CALIB_F1_C_K4:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, 0, dxdf_data[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 1, dxdc_data[i * 2 + j]);
					for (int j = 0; j < 4; j++)
						sp_jx_mat.AddTo(cur_row, j + 3, dxdk_data[i * 5 + j]);
					for (int j = 0; j < 6; j++)
						sp_jx_mat.AddTo(cur_row, j + vv * 6 + 7, dxdrT_data[i * 6 + j]);
				}
				break;
			case CALIB_F1_C_K2:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, 0, dxdf_data[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 1, dxdc_data[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 3, dxdk_data[i * 5 + j]);
					for (int j = 0; j < 6; j++)
						sp_jx_mat.AddTo(cur_row, j + vv * 6 + 5, dxdrT_data[i * 6 + j]);
				}
				break;
			case CALIB_F1_C:
				for (int i = 0; i < nPts * 2; i++)
				{
					int cur_row = row_off + i;
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, 0, dxdf_data[i * 2 + j]);
					for (int j = 0; j < 2; j++)
						sp_jx_mat.AddTo(cur_row, j + 1, dxdc_data[i * 2 + j]);
					for (int j = 0; j < 6; j++)
						sp_jx_mat.AddTo(cur_row, j + vv * 6 + 3, dxdrT_data[i * 6 + j]);
				}
				break;
			default:
				return false;
				break;
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
	bool ZQ_CameraCalibration::_calib_estimate_LevMar(int nViews, int nPts, const T* X3, const T* X2, int max_iter, T fc[2], T cc[2], T kc[5], T& alpha_c, T* rT, double& avg_err_square, Calib_Method method, bool zAxis_in, bool sparse_solver, bool display)
	{
		///
		T fc_cc_alpha_kc[10] =
		{
			fc[0], fc[1], cc[0], cc[1], alpha_c, kc[0], kc[1], kc[2], kc[3], kc[4]
		};

		Calib_Data_Header<T> data;
		data.n_pts = nPts;
		data.n_views = nViews;
		data.X3 = X3;
		data.X2 = X2;
		data.fc_cc_alpha_kc = fc_cc_alpha_kc;
		data.rT = rT;
		data.method = method;
		data.zAxis_in = zAxis_in;


		ZQ_DImage<T> hx_im(nPts * 2 * nViews, 1);
		ZQ_DImage<T> p_im;
		T*& hx = hx_im.data();
		T*& p = p_im.data();

		int unknown_num;
		switch (method)
		{
		case CALIB_F2_C_ALPHA_K5:
			unknown_num = 10 + nViews * 6;
			p_im.allocate(unknown_num, 1);
			memcpy(p, fc_cc_alpha_kc, sizeof(T)* 10);
			memcpy(p + 10, rT, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F2_C_ALPHA_K4:
			unknown_num = 9 + nViews * 6;
			p_im.allocate(unknown_num, 1);
			memcpy(p, fc_cc_alpha_kc, sizeof(T)* 9);
			memcpy(p + 9, rT, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F2_C_ALPHA_K2:
			unknown_num = 7 + nViews * 6;
			p_im.allocate(unknown_num, 1);
			memcpy(p, fc_cc_alpha_kc, sizeof(T)* 7);
			memcpy(p + 7, rT, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F2_C_ALPHA:
			unknown_num = 5 + nViews * 6;
			p_im.allocate(unknown_num, 1);
			memcpy(p, fc_cc_alpha_kc, sizeof(T)* 5);
			memcpy(p + 5, rT, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F2_C_K5:
			unknown_num = 9 + nViews * 6;
			p_im.allocate(unknown_num, 1);
			memcpy(p, fc_cc_alpha_kc, sizeof(T)* 4);
			memcpy(p + 4, fc_cc_alpha_kc + 5, sizeof(T)* 5);
			memcpy(p + 9, rT, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F2_C_K4:
			unknown_num = 8 + nViews * 6;
			p_im.allocate(unknown_num, 1);
			memcpy(p, fc_cc_alpha_kc, sizeof(T)* 4);
			memcpy(p + 4, fc_cc_alpha_kc + 5, sizeof(T)* 4);
			memcpy(p + 8, rT, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F2_C_K2:
			unknown_num = 6 + nViews * 6;
			p_im.allocate(unknown_num, 1);
			memcpy(p, fc_cc_alpha_kc, sizeof(T)* 4);
			memcpy(p + 4, fc_cc_alpha_kc + 5, sizeof(T)* 2);
			memcpy(p + 6, rT, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F2_C:
			unknown_num = 4 + nViews * 6;
			p_im.allocate(unknown_num, 1);
			memcpy(p, fc_cc_alpha_kc, sizeof(T)* 4);
			memcpy(p + 4, rT, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F1_C_ALPHA_K5:
			unknown_num = 9 + nViews * 6;
			p_im.allocate(unknown_num, 1);
			p[0] = fc_cc_alpha_kc[0];
			memcpy(p + 1, fc_cc_alpha_kc + 2, sizeof(T)* 8);
			memcpy(p + 9, rT, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F1_C_ALPHA_K4:
			unknown_num = 8 + nViews * 6;
			p_im.allocate(unknown_num, 1);
			p[0] = fc_cc_alpha_kc[0];
			memcpy(p + 1, fc_cc_alpha_kc + 2, sizeof(T)* 7);
			memcpy(p + 8, rT, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F1_C_ALPHA_K2:
			unknown_num = 6 + nViews * 6;
			p_im.allocate(unknown_num, 1);
			p[0] = fc_cc_alpha_kc[0];
			memcpy(p + 1, fc_cc_alpha_kc + 2, sizeof(T)* 5);
			memcpy(p + 6, rT, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F1_C_ALPHA:
			unknown_num = 4 + nViews * 6;
			p_im.allocate(unknown_num, 1);
			p[0] = fc_cc_alpha_kc[0];
			memcpy(p + 1, fc_cc_alpha_kc + 2, sizeof(T)* 3);
			memcpy(p + 4, rT, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F1_C_K5:
			unknown_num = 8 + nViews * 6;
			p_im.allocate(unknown_num, 1);
			p[0] = fc_cc_alpha_kc[0];
			memcpy(p + 1, fc_cc_alpha_kc + 2, sizeof(T)* 2);
			memcpy(p + 3, fc_cc_alpha_kc + 5, sizeof(T)* 5);
			memcpy(p + 8, rT, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F1_C_K4:
			unknown_num = 7 + nViews * 6;
			p_im.allocate(unknown_num, 1);
			p[0] = fc_cc_alpha_kc[0];
			memcpy(p + 1, fc_cc_alpha_kc + 2, sizeof(T)* 2);
			memcpy(p + 3, fc_cc_alpha_kc + 5, sizeof(T)* 4);
			memcpy(p + 7, rT, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F1_C_K2:
			unknown_num = 5 + nViews * 6;
			p_im.allocate(unknown_num, 1);
			p[0] = fc_cc_alpha_kc[0];
			memcpy(p + 1, fc_cc_alpha_kc + 2, sizeof(T)* 2);
			memcpy(p + 3, fc_cc_alpha_kc + 5, sizeof(T)* 2);
			memcpy(p + 5, rT, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F1_C:
			unknown_num = 3 + nViews * 6;
			p_im.allocate(unknown_num, 1);
			p[0] = fc_cc_alpha_kc[0];
			memcpy(p + 1, fc_cc_alpha_kc + 2, sizeof(T)* 2);
			memcpy(p + 3, rT, sizeof(T)* 6 * nViews);
			break;
		default:
			return false;
			break;
		}
		/***********************************/

		

		
		if (!sparse_solver)
		{
			ZQ_LevMarOptions opts;
			ZQ_LevMarReturnInfos infos;
			opts.tol_max_jte = 1e-45;
			opts.tol_dx_square = 1e-45;
			opts.tol_e_square = 1e-45;
			if (!ZQ_LevMar::ZQ_LevMar_Der<T>(_calib_estimate_fun<T>, _calib_estimate_jac<T>, p, hx, unknown_num, nPts*nViews * 2, max_iter, opts, infos, &data, display))
			{
				return false;
			}
			avg_err_square = infos.final_e_square / (nPts*nViews);
		}
		else
		{
			ZQ_SparseLevMarOptions opts;
			ZQ_SparseLevMarReturnInfos infos;
			opts.tol_max_jte = 1e-45;
			opts.tol_dx_square = 1e-45;
			opts.tol_e_square = 1e-45;
			if (!ZQ_SparseLevMar::ZQ_SparseLevMar_Der<T>(_calib_estimate_fun<T>, _calib_estimate_jac_sparse<T>, p, hx, unknown_num, nPts*nViews * 2, max_iter, opts, infos, &data, display))
			{
				return false;
			}
			avg_err_square = infos.final_e_square / (nPts*nViews);
		}
		
		

		switch (method)
		{
		case CALIB_F2_C_ALPHA_K5:
			memcpy(fc, p, sizeof(T)* 2);
			memcpy(cc, p+2, sizeof(T)* 2);
			alpha_c = p[4];
			memcpy(kc, p+5, sizeof(T)* 5);
			memcpy(rT, p + 10, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F2_C_ALPHA_K4:
			memcpy(fc, p, sizeof(T)* 2);
			memcpy(cc, p + 2, sizeof(T)* 2);
			alpha_c = p[4];
			memcpy(kc, p + 5, sizeof(T)* 4);
			memcpy(rT, p + 9, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F2_C_ALPHA_K2:
			memcpy(fc, p, sizeof(T)* 2);
			memcpy(cc, p + 2, sizeof(T)* 2);
			alpha_c = p[4];
			memcpy(kc, p + 5, sizeof(T)* 2);
			memcpy(rT, p + 7, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F2_C_ALPHA:
			memcpy(fc, p, sizeof(T)* 2);
			memcpy(cc, p + 2, sizeof(T)* 2);
			alpha_c = p[4];
			memcpy(rT, p + 5, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F2_C_K5:
			memcpy(fc, p, sizeof(T)* 2);
			memcpy(cc, p + 2, sizeof(T)* 2);
			memcpy(kc, p + 4, sizeof(T)* 5);
			memcpy(rT, p + 9, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F2_C_K4:
			memcpy(fc, p, sizeof(T)* 2);
			memcpy(cc, p + 2, sizeof(T)* 2);
			memcpy(kc, p + 4, sizeof(T)* 4);
			memcpy(rT, p + 8, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F2_C_K2:
			memcpy(fc, p, sizeof(T)* 2);
			memcpy(cc, p + 2, sizeof(T)* 2);
			memcpy(kc, p + 4, sizeof(T)* 2);
			memcpy(rT, p + 6, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F2_C:
			memcpy(fc, p, sizeof(T)* 2);
			memcpy(cc, p + 2, sizeof(T)* 2);
			memcpy(rT, p + 4, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F1_C_ALPHA_K5:
			fc[0] = fc[1] = p[0];
			memcpy(cc, p + 1, sizeof(T)* 2);
			alpha_c = p[3];
			memcpy(kc, p + 4, sizeof(T)* 5);
			memcpy(rT, p + 9, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F1_C_ALPHA_K4:
			fc[0] = fc[1] = p[0];
			memcpy(cc, p + 1, sizeof(T)* 2);
			alpha_c = p[3];
			memcpy(kc, p + 4, sizeof(T)* 4);
			memcpy(rT, p + 8, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F1_C_ALPHA_K2:
			fc[0] = fc[1] = p[0];
			memcpy(cc, p + 1, sizeof(T)* 2);
			alpha_c = p[3];
			memcpy(kc, p + 4, sizeof(T)* 2);
			memcpy(rT, p + 6, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F1_C_ALPHA:
			fc[0] = fc[1] = p[0];
			memcpy(cc, p + 1, sizeof(T)* 2);
			alpha_c = p[3];
			memcpy(rT, p + 4, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F1_C_K5:
			fc[0] = fc[1] = p[0];
			memcpy(cc, p + 1, sizeof(T)* 2);
			memcpy(kc, p + 3, sizeof(T)* 5);
			memcpy(rT, p + 8, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F1_C_K4:
			fc[0] = fc[1] = p[0];
			memcpy(cc, p + 1, sizeof(T)* 2);
			memcpy(kc, p + 3, sizeof(T)* 4);
			memcpy(rT, p + 7, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F1_C_K2:
			fc[0] = fc[1] = p[0];
			memcpy(cc, p + 1, sizeof(T)* 2);
			memcpy(kc, p + 3, sizeof(T)* 2);
			memcpy(rT, p + 5, sizeof(T)* 6 * nViews);
			break;
		case CALIB_F1_C:
			fc[0] = fc[1] = p[0];
			memcpy(cc, p + 1, sizeof(T)* 2);
			memcpy(rT, p + 3, sizeof(T)* 6 * nViews);
			break;
		default:
			return false;
			break;
		}
		
		return true;
	}

	/*******************************************************************************************************************************/
	/*******************************************************************************************************************************/
	/*******************************************************************************************************************************/

	/*
	refer to the paper:
	iterative pose estimation using coplanar feature points. Denis Oberkampf, Daniel F. DeMenthon, Larry  S. Davis. CVIU, 1995.
	rT[0-5]: rx, ry, rz, Tx, Ty, Tz.  (rx,ry,rz,rw) is a quaternion.
	*/
	template<class T>
	bool ZQ_CameraCalibration::PositCoplanar(int nPts, const T* X3, const T* X2, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, int max_iter, double tol_E, T* rT, T* reproj_err_square, bool zAxis_in)
	{
		if (fc[0] <= 0 || fc[1] <= 0)
			return false;
		T focal_len = fc[0];
		T int_A[9] = {
			fc[0], fc[0] * alpha_c, cc[0],
			0, fc[1], cc[1],
			0, 0, 1
		};

		ZQ_DImage<T> tmp_X2_im(nPts * 2, 1);
		T*& tmp_X2 = tmp_X2_im.data();

		for (int pp = 0; pp < nPts; pp++)
		{
			T tmp_y = (X2[pp * 2 + 1] - cc[1]) / fc[1];
			T tmp_x = (X2[pp * 2 + 0] - cc[0]) / fc[0] - alpha_c*tmp_y;
			T tmp_in[2] = { tmp_x, tmp_y };
			T tmp_out[2];
			ZQ_CameraCalibration::undistort_points_oulu(1, tmp_in, kc, tmp_out);
			tmp_X2[pp * 2 + 0] = fc[0] * (tmp_out[0] + alpha_c*tmp_out[1]) + cc[0];
			tmp_X2[pp * 2 + 1] = fc[1] * tmp_out[1] + cc[1];
		}

		double scale_y = fc[1] / fc[0];

		double tol_Error = nPts*tol_E*tol_E;

		ZQ_DImage<T> A_im(nPts * 3, 1);
		ZQ_DImage<T> B_im(nPts * 3, 1);
		ZQ_DImage<T> X_im(nPts * 2, 1);
		T*& A = A_im.data();
		T*& B = B_im.data();
		T*& X = X_im.data();
		for (int i = 0; i < nPts; i++)
		{
			A[i * 3 + 0] = X3[i * 3 + 0] - X3[0];
			A[i * 3 + 1] = X3[i * 3 + 1] - X3[1];
			A[i * 3 + 2] = X3[i * 3 + 2] - X3[2];
		}


		for (int i = 0; i < nPts; i++)
		{
			X[i * 2 + 0] = X2[i * 2 + 0] - cc[0];
			X[i * 2 + 1] = (X2[i * 2 + 1] - cc[1])*scale_y;
		}

		ZQ_Matrix<double> Amat(nPts, 3), Bmat(3, nPts);
		for (int i = 0; i < nPts; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				Amat.SetData(i, j, A[i * 3 + j]);
			}
		}

		if (!ZQ_SVD::Invert(Amat, Bmat))
		{
			return false;
		}
		for (int i = 0; i < nPts; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				bool flag;
				B[j*nPts + i] = Bmat.GetData(j, i, flag);
			}
		}

		ZQ_Matrix<double> Umat(nPts, 3), Smat(3, 3), Vmat(3, 3);
		double center[3] = { 0 };
		for (int i = 0; i < nPts; i++)
		{
			center[0] += X3[i * 3 + 0];
			center[1] += X3[i * 3 + 1];
			center[2] += X3[i * 3 + 2];
		}
		center[0] /= nPts;
		center[1] /= nPts;
		center[2] /= nPts;
		for (int i = 0; i < nPts; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				Amat.SetData(i, j, A[i * 3 + j] - center[j]);
			}
		}
		if (!ZQ_SVD::Decompose(Amat, Umat, Smat, Vmat))
		{
			return false;
		}

		double u[3];
		for (int i = 0; i < 3; i++)
		{
			bool flag;
			u[i] = Vmat.GetData(i, 2, flag);
		}
		if (u[2] < 0)
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
		//int selection_thresh = 8;
		int selection_thresh = 2;

		do{
			bool has_find_solution = false;
			cur_candidates.clear();
			for (int cccc = 0; cccc < last_candidates.size(); cccc++)
			{
				Posit_Coplanar_Node<T> cur_node = last_candidates[cccc];
				Posit_Coplanar_Node<T> new_node[2];

				double I0[3] = { 0 }, J0[3] = { 0 };
				for (int i = 0; i < nPts; i++)
				{
					double epsilon;
					if (it == 0)
						epsilon = 0;
					else
						epsilon = 1.0 / cur_node.Z0*(A[i * 3 + 0] * cur_node.kk[0] + A[i * 3 + 1] * cur_node.kk[1] + A[i * 3 + 2] * cur_node.kk[2]);
					double xx = X[i * 2 + 0] * (1.0 + epsilon) - X[0];
					double yy = X[i * 2 + 1] * (1.0 + epsilon) - X[1];
					I0[0] += B[0 * nPts + i] * xx;
					I0[1] += B[1 * nPts + i] * xx;
					I0[2] += B[2 * nPts + i] * xx;
					J0[0] += B[0 * nPts + i] * yy;
					J0[1] += B[1 * nPts + i] * yy;
					J0[2] += B[2 * nPts + i] * yy;
				}

				double I0J0 = I0[0] * J0[0] + I0[1] * J0[1] + I0[2] * J0[2];
				double J02_I02 = (J0[0] * J0[0] + J0[1] * J0[1] + J0[2] * J0[2]) - (I0[0] * I0[0] + I0[1] * I0[1] + I0[2] * I0[2]);

				/*lambda*mu = -I0J0
				lambda^2 - mu^2 = J02-I02
				*/

				double delta = J02_I02*J02_I02 + 4 * I0J0*I0J0;
				double lambda[2], mu[2];
				if (1)
				{
					double q = 0;
					if (J02_I02 <= 0)
						q = 0.5*(J02_I02 - sqrt(delta));
					else
						q = 0.5*(J02_I02 + sqrt(delta));


					if (q >= 0)
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
							mu[0] = -I0J0 / lambda[0];
							mu[1] = -I0J0 / lambda[1];
						}
					}
					else
					{
						lambda[0] = sqrt(-(I0J0*I0J0) / q);
						lambda[1] = -lambda[0];
						if (lambda[0] == 0)
						{
							mu[0] = sqrt(-J02_I02);
							mu[1] = -mu[0];
						}
						else
						{
							mu[0] = -I0J0 / lambda[0];
							mu[1] = -mu[0];
						}
					}

				}
				else
				{
					lambda[0] = sqrt(0.5*(J02_I02 + sqrt(delta)));
					lambda[1] = -lambda[0];
					if (lambda[0] != 0)
					{
						mu[0] = -I0J0 / lambda[0];
						mu[1] = -mu[0];
					}
					else
					{
						mu[0] = mu[1] = 0;
					}
				}

				//////////
				T II[2][3] = {
					{ I0[0] + lambda[0] * u[0], I0[1] + lambda[0] * u[1], I0[2] + lambda[0] * u[2] },
					{ I0[0] + lambda[1] * u[0], I0[1] + lambda[1] * u[1], I0[2] + lambda[1] * u[2] }
				};
				T JJ[2][3] = {
					{ J0[0] + mu[0] * u[0], J0[1] + mu[0] * u[1], J0[2] + mu[0] * u[2] },
					{ J0[0] + mu[1] * u[0], J0[1] + mu[1] * u[1], J0[2] + mu[1] * u[2] }
				};

				T cur_R[2][9];
				T cur_tt[2][3];
				T cur_Z0[2];
				T cur_kk[2][3];
				T cur_E[2];
				bool discard_flag[2] = { false, false };
				for (int dd = 0; dd < 2; dd++)
				{
					double lenII = sqrt(II[dd][0] * II[dd][0] + II[dd][1] * II[dd][1] + II[dd][2] * II[dd][2]);
					double lenJJ = sqrt(JJ[dd][0] * JJ[dd][0] + JJ[dd][1] * JJ[dd][1] + JJ[dd][2] * JJ[dd][2]);
					if (lenII == 0 || lenJJ == 0)
					{
						return false;
					}
					double ii[3] = { II[dd][0] / lenII, II[dd][1] / lenII, II[dd][2] / lenII };
					double jj[3] = { JJ[dd][0] / lenJJ, JJ[dd][1] / lenJJ, JJ[dd][2] / lenJJ };
					double s = 0.5*(lenII + lenJJ);
					cur_Z0[dd] = focal_len / s;
					cur_kk[dd][0] = ii[1] * jj[2] - ii[2] * jj[1];
					cur_kk[dd][1] = ii[2] * jj[0] - ii[0] * jj[2];
					cur_kk[dd][2] = ii[0] * jj[1] - ii[1] * jj[0];

					double len_kk = sqrt(cur_kk[dd][0] * cur_kk[dd][0] + cur_kk[dd][1] * cur_kk[dd][1] + cur_kk[dd][2] * cur_kk[dd][2]);
					if (len_kk == 0)
					{
						return false;
					}

					cur_kk[dd][0] /= len_kk;
					cur_kk[dd][1] /= len_kk;
					cur_kk[dd][2] /= len_kk;
					jj[0] = cur_kk[dd][1] * ii[2] - cur_kk[dd][2] * ii[1];
					jj[1] = cur_kk[dd][2] * ii[0] - cur_kk[dd][0] * ii[2];
					jj[2] = cur_kk[dd][0] * ii[1] - cur_kk[dd][1] * ii[0];

					double O[3] = {
						X3[0] - cur_Z0[dd] / focal_len*(X[0] * ii[0] + X[1] * jj[0] + focal_len*cur_kk[dd][0]),
						X3[1] - cur_Z0[dd] / focal_len*(X[0] * ii[1] + X[1] * jj[1] + focal_len*cur_kk[dd][1]),
						X3[2] - cur_Z0[dd] / focal_len*(X[0] * ii[2] + X[1] * jj[2] + focal_len*cur_kk[dd][2])
					};
					ZQ_Matrix<double> Mmat(4, 4), invM(4, 4);
					for (int i = 0; i < 3; i++)
					{
						Mmat.SetData(i, 0, ii[i]);
						Mmat.SetData(i, 1, jj[i]);
						Mmat.SetData(i, 2, cur_kk[dd][i]);
						Mmat.SetData(i, 3, O[i]);
					}
					Mmat.SetData(3, 3, 1);

					ZQ_SVD::Invert(Mmat, invM);

					for (int i = 0; i < 3; i++)
					{
						bool flag;
						for (int j = 0; j < 3; j++)
						{
							cur_R[dd][i * 3 + j] = invM.GetData(i, j, flag);
						}
						cur_tt[dd][i] = invM.GetData(i, 3, flag);
					}
					ZQ_DImage<T> cur_X2_im(nPts * 2, 1);
					T*& cur_X2 = cur_X2_im.data();
					T cur_rT[6];
					ZQ_Rodrigues::ZQ_Rodrigues_R2r(cur_R[dd], cur_rT);
					memcpy(cur_rT + 3, cur_tt[dd], sizeof(T)* 3);
					ZQ_CameraCalibration::project_points_fun(nPts, X3, cur_rT, fc, cc, kc, alpha_c, cur_X2, zAxis_in);
					cur_E[dd] = 0;
					for (int i = 0; i < nPts; i++)
					{
						cur_E[dd] += (cur_X2[i * 2 + 0] - X2[i * 2 + 0])*(cur_X2[i * 2 + 0] - X2[i * 2 + 0]) + (cur_X2[i * 2 + 1] - X2[i * 2 + 1])*(cur_X2[i * 2 + 1] - X2[i * 2 + 1]);
					}
					for (int i = 0; i < nPts; i++)
					{

						double tmp_pts_Z = cur_R[dd][6] * X3[i * 3 + 0] + cur_R[dd][7] * X3[i * 3 + 1] + cur_R[dd][8] * X3[i * 3 + 2] + cur_tt[dd][2];
						if (tmp_pts_Z < 0)
						{
							discard_flag[dd] = true;
							break;
						}
					}

					new_node[dd].Error = cur_E[dd];
					memcpy(new_node[dd].R, cur_R[dd], sizeof(T)* 9);
					memcpy(new_node[dd].tt, cur_tt[dd], sizeof(T)* 3);
					ZQ_Rodrigues::ZQ_Rodrigues_R2r(cur_R[dd], new_node[dd].rr);
					memcpy(new_node[dd].kk, cur_kk[dd], sizeof(T)* 3);
					new_node[dd].Z0 = cur_Z0[dd];

				}/* for dd*/

				if (it == 0)
				{
					cur_candidates.push_back(new_node[0]);
					cur_candidates.push_back(new_node[1]);
				}
				else
				{
					if (last_candidates.size() < selection_thresh)
					{
						if (!discard_flag[0]/* && (new_node[0].Error < cur_node.Error*diverge_ratio || new_node[0].Error < tol_Error)*/)
							cur_candidates.push_back(new_node[0]);
						if (!discard_flag[1]/* && (new_node[1].Error < cur_node.Error*diverge_ratio || new_node[1].Error < tol_Error)*/)
							cur_candidates.push_back(new_node[1]);

					}
					else
					{
						if (!discard_flag[0] && !discard_flag[1])
						{
							if (new_node[0].Error < new_node[1].Error)
								cur_candidates.push_back(new_node[0]);
							else
								cur_candidates.push_back(new_node[1]);
						}
						else
						{

							if (!discard_flag[0])
								cur_candidates.push_back(new_node[0]);
							if (!discard_flag[1])
								cur_candidates.push_back(new_node[1]);
						}
					}

					if (new_node[0].Error < tol_Error || new_node[1].Error < tol_Error)
						has_find_solution = true;
				}

			}/*for cccc*/

			if (cur_candidates.size() == 0)
				break;
			last_candidates = cur_candidates;
			//printf("%d ",last_candidates.size());
			if (has_find_solution)
				break;

			it++;

		} while (it <= max_iter);

		//printf("\n");

		//printf("it = %d\n",it);
		if (last_candidates.size() == 0)
		{
			return false;
		}

		double min_error = last_candidates[0].Error;
		int best_idx = 0;
		for (int cccc = 1; cccc < last_candidates.size(); cccc++)
		{
			if (last_candidates[cccc].Error < min_error)
			{
				best_idx = cccc;
				min_error = last_candidates[cccc].Error;
			}
		}

		ZQ_Rodrigues::ZQ_Rodrigues_R2r(last_candidates[best_idx].R, rT);
		memcpy(rT + 3, last_candidates[best_idx].tt, sizeof(T)* 3);
		if (!zAxis_in)
			rT[5] = -rT[5];

		if (reproj_err_square != 0)
		{
			ZQ_DImage<T> reproj_X2_im(nPts * 2, 1);
			T*& reproj_X2 = reproj_X2_im.data();
			T cur_rT[6];
			memcpy(cur_rT, last_candidates[best_idx].rr, sizeof(T)* 3);
			memcpy(cur_rT + 3, last_candidates[best_idx].tt, sizeof(T)* 3);
			if (!zAxis_in)
				cur_rT[5] = -cur_rT[5];
			ZQ_CameraCalibration::project_points_fun(nPts, X3, cur_rT, fc, cc, kc, alpha_c, reproj_X2, zAxis_in);
			for (int i = 0; i < nPts; i++)
			{
				reproj_err_square[i] = (double)(reproj_X2[i * 2 + 0] - X2[i * 2 + 0])*(reproj_X2[i * 2 + 0] - X2[i * 2 + 0]) + (reproj_X2[i * 2 + 1] - X2[i * 2 + 1])*(reproj_X2[i * 2 + 1] - X2[i * 2 + 1]);
			}
		}

		return true;
	}

	/*
	The base idea is to use the method proposed in the paper:
	iterative pose estimation using coplanar points. Denis Oberkampf, Daniel F. DeMenthon, Larry  S. Davis. CVPR, 1993.
	However, I find it cannot make sure the method always converge to the optimal solution.
	But the translation seems to be near the optimal one according to my observations.
	So I choose 9 rotations to run Lev-Mar solvers to find a best solution.
	If all choices do not give a solution with avg_E < tol_avg_E, the best solution of the 9 will be returned,
	otherwise, the first one satisfying avg_E < tol_avg_E will be returned.
	*/
	template<class T>
	bool ZQ_CameraCalibration::PositCoplanarRobust(int nPts, const T* X3, const T* X2, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, int max_iter_posit, int max_iter_levmar, double tol_E, T* rT, double& avg_E, bool zAxis_in)
	{
		ZQ_DImage<T> reproj_err_square_im(nPts, 1);
		T*& reproj_err_square = reproj_err_square_im.data();

		T init_rT[6] = { 0 };
		//memcpy(init_rT, rT, sizeof(T)* 6);
		if (!PositCoplanar(nPts, X3, X2, fc, cc, kc, alpha_c, max_iter_posit, tol_E, init_rT, reproj_err_square, zAxis_in))
		{
			return false;
		}
		T tol_E_square = tol_E*tol_E;
		T avg_E_square = 0;
		for (int i = 0; i < nPts; i++)
			avg_E_square += reproj_err_square[i] / nPts;

		if (avg_E_square <= tol_E_square)
		{
			memcpy(rT, init_rT, sizeof(T)* 6);
			avg_E = sqrt(avg_E_square);
			return true;
		}



		T rand_rr[10][3] =
		{
			{ init_rT[0], init_rT[1], init_rT[2] },
			{ 0, 0, 0 },
			{ 0.2, 0.1, 0.1 },
			{ 0.2, 0.1, -0.1 },
			{ 0.2, -0.1, 0.1 },
			{ 0.2, -0.1, 0.1 },
			{ -0.2, 0.1, 0.1 },
			{ -0.2, 0.1, -0.1 },
			{ -0.2, -0.1, 0.1 },
			{ -0.2, -0.1, -0.1 }
		};

		T tmp_rT[6];

		for (int i = 0; i < 10; i++)
		{
			memcpy(tmp_rT, rand_rr[i], sizeof(T)* 3);
			memcpy(tmp_rT + 3, init_rT + 3, sizeof(T)* 3);
			double cur_avg_E_square = 0;
			if (!pose_estimation_with_init(nPts, X3, X2, fc, cc, kc, alpha_c, max_iter_levmar, tmp_rT, cur_avg_E_square, zAxis_in))
			{
				continue;
			}
			else
			{
				if (cur_avg_E_square < avg_E_square)
				{
					avg_E_square = cur_avg_E_square;
					memcpy(rT, tmp_rT, sizeof(T)* 6);
				}
				if (avg_E_square <= tol_E_square)
					break;
			}
		}

		avg_E = sqrt(avg_E_square);
		return true;
	}


	/*pose estimation based on Lev-Mar optimization, need good initialization.*/
	template<class T>
	bool ZQ_CameraCalibration::_pose_estimation_fun(const T* p, T* hx, int m, int n, const void* data)
	{
		const Pose_Estimation_Data_Header<T>* ptr = (const Pose_Estimation_Data_Header<T>*)data;

		int nPts = ptr->nPts;
		const T* X3 = ptr->X3;
		const T* X2 = ptr->X2;
		const T* fc = ptr->fc_cc_alpha_kc;
		const T* cc = ptr->fc_cc_alpha_kc + 2;
		const T alpha_c = ptr->fc_cc_alpha_kc[4];
		const T* kc = ptr->fc_cc_alpha_kc + 5;
		bool zAxis_in = ptr->zAxis_in;

		ZQ_DImage<T> tmp_X2_im(nPts * 2, 1);
		T*& tmp_X2 = tmp_X2_im.data();

		if (!ZQ_CameraCalibration::project_points_fun(nPts, X3, p, fc, cc, kc, alpha_c, tmp_X2, zAxis_in))
			return false;

		ZQ_MathBase::VecMinus(nPts * 2, tmp_X2, X2, hx);
		return true;
	}

	/*pose estimation based on Lev-Mar optimization, need good initialization.*/
	template<class T>
	bool ZQ_CameraCalibration::_pose_estimation_jac(const T* p, T* jx, int m, int n, const void* data)
	{
		const Pose_Estimation_Data_Header<T>* ptr = (const Pose_Estimation_Data_Header<T>*)data;

		int nPts = ptr->nPts;
		const T* X3 = ptr->X3;
		const T* fc = ptr->fc_cc_alpha_kc;
		const T* cc = ptr->fc_cc_alpha_kc + 2;
		const T alpha_c = ptr->fc_cc_alpha_kc[4];
		const T* kc = ptr->fc_cc_alpha_kc + 5;
		bool zAxis_in = ptr->zAxis_in;

		if (!ZQ_CameraCalibration::project_points_jac(nPts, X3, p, fc, cc, kc, alpha_c, jx, (T*)0, (T*)0, (T*)0, (T*)0, zAxis_in))
			return false;
		return true;
	}

	/*pose estimation based on Lev-Mar optimization, need good initialization.*/
	template<class T>
	bool ZQ_CameraCalibration::pose_estimation_with_init(int nPts, const T* X3, const T* X2, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, int max_iter, T* rT, double& avg_err_square, bool zAxis_in)
	{
		Pose_Estimation_Data_Header<T> data;
		data.nPts = nPts;
		data.X3 = X3;
		data.X2 = X2;
		data.zAxis_in = zAxis_in;
		T fc_cc_alpha_kc[10] =
		{
			fc[0], fc[1], cc[0], cc[1], alpha_c, kc[0], kc[1], kc[2], kc[3], kc[4]
		};
		data.fc_cc_alpha_kc = fc_cc_alpha_kc;
		ZQ_LevMarOptions opts;
		ZQ_LevMarReturnInfos infos;
		opts.tol_e_square = 1e-16;
		opts.tol_max_jte = 1e-16;
		opts.tol_dx_square = 1e-16;

		ZQ_DImage<T> hx_im(nPts * 2, 1);
		T*& hx = hx_im.data();

		if (!ZQ_LevMar::ZQ_LevMar_Der<T>(_pose_estimation_fun<T>, _pose_estimation_jac<T>, rT, hx, 6, nPts * 2, max_iter, opts, infos, &data))
		{
			return false;
		}
		avg_err_square = infos.final_e_square / nPts;
		return true;
	}
}

#endif