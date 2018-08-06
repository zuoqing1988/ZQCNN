#ifndef _ZQ_CAMERA_PROJECTION_H_
#define _ZQ_CAMERA_PROJECTION_H_
#pragma once 

#include "ZQ_MathBase.h"
#include "ZQ_Rodrigues.h"

namespace ZQ
{
	class ZQ_CameraProjection
	{
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
	};
	/**************************************************************************************************************************************************************/

	template<class T>
	void ZQ_CameraProjection::distort_points(int nPts, const T* in_x, const T k[5], T* out_x)
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
	bool ZQ_CameraProjection::project_points_fun(int nPts, const T* X, const T* rT, const T* f, const T* c, const T* k, const T alpha, T* xp, bool zAxis_in)
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
			memcpy(xp + pp * 2, xxp, sizeof(T) * 2);
		}
		return true;
	}

	template<class T>
	bool ZQ_CameraProjection::project_points_fun(int nPts, const T* X, const T* rT, const T* f, const T* c, const T alpha, T* xp, bool zAxis_in)
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
			memcpy(xp + pp * 2, xxp, sizeof(T) * 2);
		}
		return true;
	}

	template<class T>
	bool ZQ_CameraProjection::project_points_jac(int nPts, const T* X, const T* rT, const T* f, const T* c, const T* k, const T alpha, T* dxpdrT, T* dxpdf, T* dxpdc, T* dxpdk, T* dxpdalpha, bool zAxis_in)
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
				memcpy(dxpdrT + pp * 12, dxxpdrT, sizeof(T) * 12);
			if (dxpdk != 0)
				memcpy(dxpdk + pp * 10, dxxpdk, sizeof(T) * 10);
			if (dxpdf != 0)
				memcpy(dxpdf + pp * 4, dxxpdf, sizeof(T) * 4);
			if (dxpdc != 0)
				memcpy(dxpdc + pp * 4, dxxpdc, sizeof(T) * 4);
			if (dxpdalpha != 0)
				memcpy(dxpdalpha + pp * 2, dxxpdalpha, sizeof(T) * 2);
		}

		return true;
	}


	template<class T>
	bool ZQ_CameraProjection::project_points_jac(int nPts, const T* X, const T* rT, const T* f, const T* c, const T alpha, T* dxpdrT, T* dxpdf, T* dxpdc, T* dxpdalpha, bool zAxis_in)
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
				memcpy(dxpdrT + pp * 12, dxxpdrT, sizeof(T) * 12);
			if (dxpdf != 0)
				memcpy(dxpdf + pp * 4, dxxpdf, sizeof(T) * 4);
			if (dxpdc != 0)
				memcpy(dxpdc + pp * 4, dxxpdc, sizeof(T) * 4);
			if (dxpdalpha != 0)
				memcpy(dxpdalpha + pp * 2, dxxpdalpha, sizeof(T) * 2);
		}

		return true;
	}

	template<class T>
	void ZQ_CameraProjection::undistort_points(int nPts, const T* x_in, const T k, T* x_out)
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
	void ZQ_CameraProjection::undistort_points_oulu(int nPts, const T* x_in, const T k[5], T* x_out)
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
	void ZQ_CameraProjection::normalize_pixels(int nPts, const T* x_in, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, T* x_out)
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
}

#endif