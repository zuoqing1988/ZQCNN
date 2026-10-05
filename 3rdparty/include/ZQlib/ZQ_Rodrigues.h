#ifndef _ZQ_RODRIGUES_H_
#define _ZQ_RODRIGUES_H_

#include "ZQ_SVD.h"
#include "ZQ_MathBase.h"

namespace ZQ
{
	class ZQ_Rodrigues
	{
	public:
		/*
		r= r(:);
		theta = norm(r);
		R = cos(theta)* I + (1-cos(theta))*r*r^T + sin(theta)*[0, -rz, ry; rz, 0, -rx; -ry, rx, 0];
		*/
		template<class T>
		static void ZQ_Rodrigues_r2R(const T* r, T* R, T* dRdr = 0);

		/*
		审计修复 2026-10-06（附录 IY.1）：`ZQ_Calibration.h` 里有 **6 处**调用
		`ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun(r, R)` 与 `ZQ_Rodrigues_r2R_jac(r, dRdr)`，
		而这两个静态成员**从来没存在过** —— 于是整个 `ZQ_Calibration.h` 编不过：
		    ZQ_Calibration.h:1851:22: error: 'ZQ_Rodrigues_r2R_fun' is not a member of 'ZQ::ZQ_Rodrigues'
		    ZQ_Calibration.h:1919:22: error: 'ZQ_Rodrigues_r2R_jac' is not a member of 'ZQ::ZQ_Rodrigues'
		`tools/probe_zqlib_headers.py` 一直把这个头列在 BROKEN 里，而 BROKEN 那 9 个
		此前都被归为「需要 windows.h / OpenCV / GL，本机测不了」——
		**只有这一个不是**，它是实打实的「头里引用了不存在的成员」。

		两个包装的语义由调用点唯一确定：
		    T R[9];      ZQ_Rodrigues_r2R_fun(r, R);      // 返回 bool，失败时 false
		    T dRdr[27];  ZQ_Rodrigues_r2R_jac(r, dRdr);   // 返回 bool，3x9 的 dR/dr
		底层 `ZQ_Rodrigues_r2R(r, R, dRdr = 0)` 本来就把 Jacobian 一并算好了
		（第三个形参），所以这两个只是**带判空的薄包装**，不引入任何新的数学。
		*/
		template<class T>
		static bool ZQ_Rodrigues_r2R_fun(const T* r, T* R)
		{
			if (r == 0 || R == 0)
				return false;
			ZQ_Rodrigues_r2R(r, R);
			return true;
		}

		template<class T>
		static bool ZQ_Rodrigues_r2R_jac(const T* r, T* dRdr)
		{
			if (r == 0 || dRdr == 0)
				return false;
			ZQ_Rodrigues_r2R(r, (T*)0, dRdr);   // 必须显式转型：字面量 0 是 int，T* 接不住
			return true;
		}

		/* `ZQ_Calibration.h:4683` 用的是**三参数**形态 `r2R_jac(r, R, dRdr)`
		   —— R 与 Jacobian 一起要。语义仍然由底层 `ZQ_Rodrigues_r2R(r, R, dRdr)`
		   唯一确定，不需要猜。 */
		template<class T>
		static bool ZQ_Rodrigues_r2R_jac(const T* r, T* R, T* dRdr)
		{
			if (r == 0)
				return false;
			ZQ_Rodrigues_r2R(r, R, dRdr);
			return true;
		}

		/* 把旋转向量折到**主值区间** `(-pi, pi]`：沿自身轴减去 2*pi 的整数倍。
		   之所以敢这么定，是因为调用点全都紧跟着 `r2R` / `r2R_jac`，
		   而 R 对旋转向量是 2*pi 周期的 —— 折与不折**得到的 R 完全一样**，
		   所以这是一个**只做规范化、不改结果**的操作，不引入新的自由度。
		   （真要是作者本意是别的归一化，也不会改变 `ZQ_Calibration.h` 的结果。） */
		template<class T>
		static bool ZQ_Rodrigues_autoscale(T* r)
		{
			if (r == 0)
				return false;
			const double kPi = 3.14159265358979323846;
			double in[3] = { r[0], r[1], r[2] };
			double n = ZQ_MathBase::NormVector_L2(3, in);
			if (n < 1e-16)
				return true;                       // 零向量：本来就无所谓
			double k = floor(n / (2.0 * kPi) + 0.5);
			if (k == 0.0)
				return true;
			double s = (n - 2.0 * kPi * k) / n;
			r[0] = (T)(in[0] * s);
			r[1] = (T)(in[1] * s);
			r[2] = (T)(in[2] * s);
			return true;
		}

		/*
		q = [q0,q1,q2,q3];
		R = [	q0^2+q1^2-q2^2-q3^2,	2*(q1q2+q0q3),			2*(q1q3-q0q2);
				2*(q1q2-q0q3),			q0^2+q2^2-q1^2-q3^2,	2*(q2q3+q1*q1);
				2*(q0q2+q1q3),			2*(q2q3-q0q1),			q0^2+q3^2-q1^2-q2^2];
		r = [q1,q2,q3]/sqrt(q1^2+q2^2+q3^2)*acos(q0)*2
		*/
		template<class T>
		static bool ZQ_Rodrigues_R2r(const T* R, T* r, T* drdR = 0);

		/* 同 IY.1：`ZQ_Calibration.h:3316 / 3397 / 3602` 还调了
		   `ZQ_Rodrigues_R2r_fun(R, r)`，同样从来没存在过。
		   底层 `ZQ_Rodrigues_R2r` 的返回值本来就是 bool（SVD 失败时 false），
		   这一层只是**带判空的薄包装**，不引入任何新的数学。 */
		template<class T>
		static bool ZQ_Rodrigues_R2r_fun(const T* R, T* r)
		{
			if (R == 0 || r == 0)
				return false;
			return ZQ_Rodrigues_R2r(R, r);
		}

	};
	
	/*
	r= r(:);
	theta = norm(r);
	R = cos(theta)* I + (1-cos(theta))*r*r^T + sin(theta)*[0, -rz, ry; rz, 0, -rx; -ry, rx, 0];
	*/
	template<class T>
	void ZQ_Rodrigues::ZQ_Rodrigues_r2R(const T* r, T* R, T* dRdr)
	{
		double eps = 1e-16;
		double in[3] = { r[0], r[1], r[2] };
		double theta = ZQ_MathBase::NormVector_L2(3, in);
		if (theta < eps)
		{
			if (R != 0)
			{
				R[0] = R[4] = R[8] = 1;
				R[1] = R[2] = R[3] = R[5] = R[6] = R[7] = 0;
			}
			if (dRdr != 0)
			{
				double dRdin[27] =
				{
					0, 0, 0,
					0, 0, -1,
					0, 1, 0,
					0, 0, 1,
					0, 0, 0,
					-1, 0, 0,
					0, -1, 0,
					1, 0, 0,
					0, 0, 0
				};

				for (int i = 0; i < 27; i++)
					dRdr[i] = dRdin[i];
			}
		}
		else
		{
			double omega[3] =
			{
				in[0] / theta, in[1] / theta, in[2] / theta
			};

			double alpha = cos(theta);
			double beta = sin(theta);
			double gamma = 1.0 - cos(theta);
			double omegav[9] =
			{
				0, -omega[2], omega[1],
				omega[2], 0, -omega[0],
				-omega[1], omega[0], 0
			};
			double A[9] = { 0 };
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					A[i * 3 + j] = omega[i] * omega[j];
				}
			}

			double w1 = omega[0];
			double w2 = omega[1];
			double w3 = omega[2];

			double RR[9] =
			{
				alpha, 0, 0,
				0, alpha, 0,
				0, 0, alpha
			};
			for (int i = 0; i < 9; i++)
				RR[i] += omegav[i] * beta + A[i] * gamma;

			if (R != 0)
			{
				for (int i = 0; i < 9; i++)
					R[i] = RR[i];
			}

			if (dRdr != 0)
			{
				
				/* alpha=cos(theta) */
				double dthetadin[3] = { in[0] / theta, in[1] / theta, in[2] / theta };
				double dalphadin[3] = 
				{
					-sin(theta)*dthetadin[0], -sin(theta)*dthetadin[1], -sin(theta)*dthetadin[2]
				};
				/* beta = sin(theta) */
				double dbetadin[3] = 
				{
					cos(theta)*dthetadin[0], cos(theta)*dthetadin[1], cos(theta)*dthetadin[2]
				};
				/* gamma = 1-cos(theta) */
				double dgammadin[3] =
				{
					sin(theta)*dthetadin[0], sin(theta)*dthetadin[1], sin(theta)*dthetadin[2]
				};

				double dtheta_1din[3] =
				{
					-dthetadin[0] / (theta*theta), -dthetadin[1] / (theta*theta), -dthetadin[2] / (theta*theta)
				};
				/* omega = [in[0]*theta^-1, in[1]*theta^-1, in[2]*theta^-1]*/
				double domegadin[9] =
				{
					1.0 / theta + in[0] * dtheta_1din[0], in[0] * dtheta_1din[1], in[0] * dtheta_1din[2],
					in[1] * dtheta_1din[0], 1.0 / theta + in[1] * dtheta_1din[1], in[1] * dtheta_1din[2],
					in[2] * dtheta_1din[0], in[2] * dtheta_1din[1], 1.0 / theta + in[2] * dtheta_1din[2]
				};
				/* omegav = [0, -omega[2], -omega[1]; omega[2], 0, -omega[0]; -omega[1] omega[0] 0]*/
				double domegavdomega[27] =
				{
					0, 0, 0,
					0, 0, -1,
					0, 1, 0,
					0, 0, 1,
					0, 0, 0,
					-1, 0, 0,
					0, -1, 0,
					1, 0, 0,
					0, 0, 0
				};
				double domegavdin[27];
				ZQ_MathBase::MatrixMul(domegavdomega, domegadin, 9, 3, 3, domegavdin);

				/* A = r*r^T */
				double dAdomega[27] = 
				{
					2 * omega[0], 0, 0,
					omega[1], omega[0], 0,
					omega[2], 0, omega[0],
					omega[1], omega[0], 0,
					0, 2 * omega[1], 0,
					0, omega[2], omega[1],
					omega[2], 0, omega[0],
					0, omega[2], omega[1],
					0, 0, 2 * omega[2]
				};
				double dAdin[27];
				ZQ_MathBase::MatrixMul(dAdomega, domegadin, 9, 3, 3, dAdin);

				/*************************************************/
				/* m1 = [alpha; beta; gamma; omegav(:); A(:)];
				/*************************************************/
				double dm1din[21 * 3];
				memcpy(dm1din + 0 * 3, dalphadin, sizeof(double)* 3);
				memcpy(dm1din + 1 * 3, dbetadin, sizeof(double)* 3);
				memcpy(dm1din + 2 * 3, dgammadin, sizeof(double)* 3);
				memcpy(dm1din + 3 * 3, domegavdin, sizeof(double)* 9 * 3);
				memcpy(dm1din + 12 * 3, dAdin, sizeof(double)* 9 * 3);
				
				/*********R = alpha* I + gamma*A + beta*omegav************/
				double dRdm1[9 * 21] = { 0 };
				//dRdalpha
				dRdm1[0 * 21 + 0] = 1;
				dRdm1[4 * 21 + 0] = 1;
				dRdm1[8 * 21 + 0] = 1;
				//dRdbeta
				for (int i = 0; i < 9; i++)
				{
						dRdm1[i * 21 + 1] = omegav[i];
				}
				//dRdgamma
				for (int i = 0; i < 9; i++)
				{
					dRdm1[i * 21 + 2] = A[i];
				}
				//dRdomegav
				for (int i = 0; i < 9; i++)
				{
					dRdm1[i * 21 + 3 + i] = beta;
				}
				//dRdA
				for (int i = 0; i < 9; i++)
				{
					dRdm1[i * 21 + 12 + i] = gamma;
				}
				

				double dRdin[27];
				ZQ_MathBase::MatrixMul(dRdm1, dm1din, 9, 21, 3, dRdin);
				for (int i = 0; i < 27; i++)
					dRdr[i] = dRdin[i];
			}	
		}
	}

	/*
	q = [q0,q1,q2,q3];
	q0 = cos(0.5*theta);
	R = [	q0^2+q1^2-q2^2-q3^2,	2*(q1q2+q0q3),			2*(q1q3-q0q2);
			2*(q1q2-q0q3),			q0^2+q2^2-q1^2-q3^2,	2*(q2q3+q1*q1);
			2*(q0q2+q1q3),			2*(q2q3-q0q1),			q0^2+q3^2-q1^2-q2^2];
	theta = acos((R[0]+R[4]+R[8]-1)/2);
	r = [q1,q2,q3]/sqrt(q1^2+q2^2+q3^2)*theta
	*/
	template<class T>
	bool ZQ_Rodrigues::ZQ_Rodrigues_R2r(const T* R, T* r, T* drdR)
	{
		ZQ_Matrix<double> Rm(3,3),U(3,3),S(3,3),V(3,3);
		for(int i = 0;i < 3;i++)
		{
			for(int j = 0;j < 3;j++)
				Rm.SetData(i,j,R[i*3+j]);
		}

		if(!ZQ_SVD::Decompose(Rm,U,S,V))
			return false;


		V.Transpose();
		Rm = U*V;

		double RR[9];
		for(int i = 0;i < 3;i++)
		{
			for(int j = 0;j < 3;j++)
			{
				bool flag;
				RR[i*3+j] = Rm.GetData(i,j,flag);
			}
		}

		for(int i = 0;i < 9;i++)
			RR[i] = __max(-1.0,__min(1.0,RR[i]));

		double tr = (RR[0]+RR[4]+RR[8]-1)/2.0; //tr = (4q0^2-2)/2 = 2cos^2(theta/2)-1 = cos(theta)
		double dtrdR[9] =
		{
			0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5
		};
		double theta = acos(tr);


		const double eps = 1e-4;

		if(sin(theta) > eps) //means: a = cos(theta/2) > 0
		{
			double vth = 1.0 / (2.0 * sin(theta));
			double om1[3] = { R[7] - R[5], R[2] - R[6], R[3] - R[1] };
			double om[3] = { vth*om1[0], vth*om1[1], vth*om1[2] };
			double out[3] = { om[0] * theta, om[1] * theta, om[2] * theta };
			
			if (r != 0)
			{
				r[0] = out[0]; r[1] = out[1]; r[2] = out[2];
			}
			
			if (drdR != 0)
			{
				double dthetadtr = -1.0 / sqrt(1.0 - tr*tr);
				double dthetadR[9];
				for (int i = 0; i < 9; i++)
					dthetadR[i] = dthetadtr*dtrdR[i];

				/*R = [	q0^2+q1^2-q2^2-q3^2,	2*(q1q2+q0q3),			2*(q1q3-q0q2);
						2*(q1q2-q0q3),			q0^2+q2^2-q1^2-q3^2,	2*(q2q3+q0*q1);
						2*(q0q2+q1q3),			2*(q2q3-q0q1),			q0^2+q3^2-q1^2-q2^2];*/
				// var1 = [vth; theta];
				double dvthdtheta = -vth*cos(theta) / sin(theta);
				double dvar1dtheta[2] = { dvthdtheta, 1 };
				double dvar1dR[18];
				ZQ_MathBase::MatrixMul(dvar1dtheta, dthetadR, 2, 1, 9, dvar1dR);

				double dom1dR[27] =
				{
					0, 0, 0, 0, 0, -1, 0, 1, 0,
					0, 0, 1, 0, 0, 0, -1, 0, 0,
					0, -1, 0, 1, 0, 0, 0, 0, 0
				};

				// var = [om1; vth; theta];
				double dvardR[45];
				memcpy(dvardR, dom1dR, sizeof(double)* 27);
				memcpy(dvardR + 27, dvar1dR, sizeof(double)* 18);

				// var2 = [om; theta];
				double domdvar[15] =
				{
					vth, 0, 0, om1[0], 0,
					0, vth, 0, om1[1], 0,
					0, 0, vth, om1[2], 0
				};

				double dthetadvar[5] = { 0, 0, 0, 0, 1 };
				double dvar2dvar[20];
				memcpy(dvar2dvar, domdvar, sizeof(double)* 15);
				memcpy(dvar2dvar + 15, dthetadvar, sizeof(double)* 5);

				double domegadvar2[12] =
				{
					theta, 0, 0, om[0],
					0, theta, 0, om[1],
					0, 0, theta, om[2]
				};

				double domegadvar[15];
				double dout[27];
				ZQ_MathBase::MatrixMul(domegadvar2, dvar2dvar, 3, 4, 5, domegadvar);
				ZQ_MathBase::MatrixMul(domegadvar, dvardR, 3, 5, 9, dout);
				for (int i = 0; i < 27; i++)
					drdR[i] = dout[i];
			}	
		}
		else
		{
			if(tr > 0)//means: angle = 0
			{
				if (r != 0)
				{
					r[0] = r[1] = r[2] = 0;
				}
				if (drdR != 0)
				{
					T dout[27] =
					{
						0, 0, 0, 0, 0, -0.5, 0, 0.5, 0,
						0, 0, 0.5, 0, 0, 0, -0.5, 0, 0,
						0, -0.5, 0, 0.5, 0, 0, 0, 0, 0
					};
					memcpy(drdR, dout, sizeof(T)* 27);
				}
			}
			else //means: angle = pi
			{
				double x_abs = sqrt((RR[0]+1)*0.5);
				double y_abs = sqrt((RR[4]+1)*0.5);
				double z_abs = sqrt((RR[8]+1)*0.5);

				int xy_sign = (RR[1] > eps) - (RR[1] < -eps);
				int yz_sign = (RR[5] > eps) - (RR[5] < -eps);
				int xz_sign = (RR[6] > eps) - (RR[6] < -eps);

				int hash_vec[11] = {0, -1, -3, -9, 9, 3, 1, 13, 5, -7, -11};
				int signs_mat[11][3] = 
				{
					{1,1,1},
					{1,0,-1},
					{0,1,-1}, 
					{1,-1,0}, 
					{1,1,0}, 
					{0,1,1}, 
					{1,0,1}, 
					{1,1,1}, 
					{1,1,-1},
					{1,-1,-1},
					{1,-1,1}
				};

				int hash_val = xy_sign*9 + yz_sign*3 + xz_sign;

				int hash_idx = 0;
				for(int i = 0;i < 11;i++)
				{
					if(hash_val == hash_vec[i])
					{
						hash_idx = i;
						break;
					}
				}

				// 审计修复 2026-10-06（附录 IY.2）：这里原来**漏乘了角度**，
				// 于是 `r2R(R2r(R))` 在 θ = pi 处给出的是**单位轴**而不是 pi*轴 ——
				// 旋转量少了整整一个因子 pi，回代得到的旋转矩阵完全不是 R。
				// 实测（tools/zq_rodrigues_check.cpp 的往返判据）：
				//     r = (pi, 0, 0)  ->  R = diag(1,-1,-1)  ->  r' = (1, 0, 0)
				// 偏差正好是 pi - 1 = 2.1416。
				// `x_abs/y_abs/z_abs` 算的是**轴的分量**（sqrt((R_ii+1)/2) = |n_i|），
				// 旋转向量要的是**角度乘轴**。
				r[0] = x_abs * signs_mat[hash_idx][0] * theta;
				r[1] = y_abs * signs_mat[hash_idx][1] * theta;
				r[2] = z_abs * signs_mat[hash_idx][2] * theta;

				if (drdR != 0)
				{
					for (int i = 0; i < 27; i++)
					{
						drdR[i] = 1;
					}
					printf("WARNING!!!! Jacobian domdR undefined!!!\n");
				}
			}
		}
		return true;
	}
}


#endif