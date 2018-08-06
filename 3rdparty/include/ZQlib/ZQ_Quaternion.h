#ifndef _ZQ_QUATERNION_H_
#define _ZQ_QUATERNION_H_
#pragma once

#include "ZQ_SVD.h"

namespace ZQ
{
	template<class T>
	class ZQ_Quaternion
	{
	public:
		T x, y, z, w;

		static void Quat2Rot(const ZQ_Quaternion& quat, T R[9])
		{
			double x = quat.x;
			double y = quat.y;
			double z = quat.z;
			double w = quat.w;

			double x2 = x*x;
			double y2 = y*y;
			double z2 = z*z;
			double w2 = w*w;

			R[0] = w2 + x2 - y2 - z2;
			R[1] = 2 * (x*y - z*w);
			R[2] = 2 * (x*z + y*w);
			R[3] = 2 * (x*y + z*w);
			R[4] = w2 + y2 - x2 - z2;
			R[5] = 2 * (y*z - x*w);
			R[6] = 2 * (x*z - y*w);
			R[7] = 2 * (y*z + x*w);
			R[8] = w2 + z2 - x2 - y2;
		}

		static bool Rot2Quat(const T R[9], ZQ_Quaternion& quat)
		{
			ZQ_Matrix<double> Rm(3, 3), U(3, 3), S(3, 3), V(3, 3);
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
					Rm.SetData(i, j, R[i * 3 + j]);
			}

			if (!ZQ_SVD::Decompose(Rm, U, S, V))
				return false;


			V.Transpose();
			Rm = U*V;

			double RR[9];
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					bool flag;
					RR[i * 3 + j] = Rm.GetData(i, j, flag);
				}
			}

			for (int i = 0; i < 9; i++)
				RR[i] = __max(-1.0, __min(1.0, RR[i]));

			double costheta = (RR[0] + RR[4] + RR[8] - 1) / 2.0;
			double theta = acos(costheta);


			const double eps = 1e-6;

			if (sin(theta) > eps) //means: a = cos(theta/2) > 0
			{
				double w2 = (RR[0] + RR[4] + RR[8] + 1.0) / 4;
				double w = sqrt(w2);
				double xw = (RR[7] - RR[5]) / 4;
				double yw = (RR[2] - RR[6]) / 4;
				double zw = (RR[3] - RR[1]) / 4;

				quat.x = xw / w;
				quat.y = yw / w;
				quat.z = zw / w;
				quat.w = w;
			}
			else
			{
				if (costheta > 0)//means: angle = 0
				{
					double w2 = (RR[0] + RR[4] + RR[8] + 1.0) / 4;
					double w = sqrt(w2);
					double xw = (RR[7] - RR[5]) / 4;
					double yw = (RR[2] - RR[6]) / 4;
					double zw = (RR[3] - RR[1]) / 4;

					quat.x = xw / w;
					quat.y = yw / w;
					quat.z = zw / w;
					quat.w = w;
				}
				else //means: angle = pi
				{
					double x_abs = sqrt((RR[0] + 1)*0.5);
					double y_abs = sqrt((RR[4] + 1)*0.5);
					double z_abs = sqrt((RR[8] + 1)*0.5);

					int xy_sign = (RR[1] > eps) - (RR[1] < -eps);
					int yz_sign = (RR[5] > eps) - (RR[5] < -eps);
					int xz_sign = (RR[6] > eps) - (RR[6] < -eps);

					int hash_vec[11] = { 0, -1, -3, -9, 9, 3, 1, 13, 5, -7, -11 };
					int signs_mat[11][3] =
					{
						{ 1, 1, 1 },
						{ 1, 0, -1 },
						{ 0, 1, -1 },
						{ 1, -1, 0 },
						{ 1, 1, 0 },
						{ 0, 1, 1 },
						{ 1, 0, 1 },
						{ 1, 1, 1 },
						{ 1, 1, -1 },
						{ 1, -1, -1 },
						{ 1, -1, 1 }
					};

					int hash_val = xy_sign * 9 + yz_sign * 3 + xz_sign;

					int hash_idx = 0;
					for (int i = 0; i < 11; i++)
					{
						if (hash_val == hash_vec[i])
						{
							hash_idx = i;
							break;
						}
					}

					quat.x = x_abs * signs_mat[hash_idx][0];
					quat.y = y_abs * signs_mat[hash_idx][1];
					quat.z = z_abs * signs_mat[hash_idx][2];
					quat.w = sqrt(1.0 - quat.x*quat.x - quat.y*quat.y - quat.z*quat.z);
				}
			}
			return true;
		}

		ZQ_Quaternion()
		{
			x = y = z = 0;
			w = 1;
		}

		ZQ_Quaternion(T x, T y, T z, T w)
		{
			this->x = x;
			this->y = y;
			this->z = z;
			this->w = w;
		}

		~ZQ_Quaternion()
		{
		}

		friend ZQ_Quaternion operator +(const ZQ_Quaternion& v1, const ZQ_Quaternion &v2)
		{
			return ZQ_Quaternion(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
		}

		friend ZQ_Quaternion operator -(const ZQ_Quaternion& v1, const ZQ_Quaternion &v2)
		{
			return ZQ_Quaternion(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w);
		}

		ZQ_Quaternion operator -() const 
		{
			return ZQ_Quaternion(-x, -y, -z, -w);
		}

		void operator +=(const ZQ_Quaternion& v)
		{
			x += v.x;
			y += v.y;
			z += v.z;
			w += w.z;
		}


		ZQ_Quaternion operator *(T scale) const
		{
			return ZQ_Quaternion(x*scale, y*scale, z*scale, w*scale);
		}

		T Dot(const ZQ_Quaternion& other) const
		{
			return x*other.x + y*other.y + z*other.z + w*other.w;
		}

		void operator *=(T scale)
		{
			x *= scale;
			y *= scale;
			z *= scale;
			w *= scale;
		}

		

		T Length() const
		{
			return sqrt(x*x + y*y + z*z + w*w);
		}

		bool Normalized()
		{
			T length = sqrt(x*x + y*y + z*z + w*w);
			if (length == 0)
			{
				return false;
			}
			else
			{
				x /= length;
				y /= length;
				z /= length;
				w /= length;
				return true;
			}
		}

		static double Dot(const ZQ_Quaternion q1, const ZQ_Quaternion q2)
		{
			return q1.x*q2.x + q1.y*q2.y + q1.z*q2.z + q1.w*q2.w;
		}

		static ZQ_Quaternion Slerp(const ZQ_Quaternion q1, const ZQ_Quaternion q2, double t)
		{
			// Only unit quaternions are valid rotations.
			// Normalize to avoid undefined behavior.
			ZQ_Quaternion v1 = q1;
			ZQ_Quaternion v2 = q2;
			v1.Normalized();
			v2.Normalized();

			// Compute the cosine of the angle between the two vectors.
			double dot = Dot(v1, v2);

			const double DOT_THRESHOLD = 0.9995;
			if (fabs(dot) > DOT_THRESHOLD) 
			{
				// If the inputs are too close for comfort, linearly interpolate
				// and normalize the result.
				ZQ_Quaternion result = v1 + (v2 - v1)*t;
				result.Normalized();
				return result;
			}

			// If the dot product is negative, the quaternions
			// have opposite handed-ness and slerp won't take
			// the shorter path. Fix by reversing one quaternion.
			if (dot < 0.0f) 
			{
				v2 = -v2;
				dot = -dot;
			}

			dot = __min(1, __max(-1, dot));// Robustness: Stay within domain of acos()  
			double theta_0 = acos(dot);  // theta_0 = angle between input vectors
			double theta = theta_0*t;    // theta = angle between v0 and result 

			ZQ_Quaternion v3 = v2 - v1*dot;
			v3.Normalized();              // { v0, v2 } is now an orthonormal basis

			return v1*cos(theta) + v3*sin(theta);

		}
	};

	
}

#endif
