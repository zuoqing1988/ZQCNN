#ifndef _ZQ_POSE_ESTIMATION_H_
#define _ZQ_POSE_ESTIMATION_H_
#pragma once

#include "ZQ_CameraCalibration.h"

namespace ZQ
{
	class ZQ_PoseEstimation
	{
	public:
		/*
		refer to the paper:
		iterative pose estimation using coplanar feature points. Denis Oberkampf, Daniel F. DeMenthon, Larry  S. Davis. CVIU, 1995.
		left hand coordinates.
		intrinsic_para[0-3]: fx, fy, u0, v0. with no distortion.
		rT[0-5]: rx, ry, rz, Tx, Ty, Tz.  (rx,ry,rz,rw) is a quaternion.
		*/
		template<class T>
		static bool PositCoplanar(int nPts, const T* X3, const T* X2, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, int max_iter, double tol_E, T* rT, T* reproj_err_square, bool zAxis_in);

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


	/*******************************************************************************************************************************/

	/*
	refer to the paper:
	iterative pose estimation using coplanar feature points. Denis Oberkampf, Daniel F. DeMenthon, Larry  S. Davis. CVIU, 1995.
	left hand coordinates.
	intrinsic_para[0-3]: fx, fy, u0, v0. with no distortion.
	rT[0-5]: rx, ry, rz, Tx, Ty, Tz.  (rx,ry,rz,rw) is a quaternion.
	*/
	template<class T>
	bool ZQ_PoseEstimation::PositCoplanar(int nPts, const T* X3, const T* X2, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, int max_iter, double tol_E, T* rT, T* reproj_err_square, bool zAxis_in)
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
					ZQ_CameraCalibration::project_points_fun(nPts, X3, cur_rT, fc, cc, kc, alpha_c, cur_X2,zAxis_in);
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
			memcpy(cur_rT+3, last_candidates[best_idx].tt, sizeof(T)* 3);
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
	bool ZQ_PoseEstimation::PositCoplanarRobust(int nPts, const T* X3, const T* X2, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, int max_iter_posit, int max_iter_levmar, double tol_E, T* rT, double& avg_E, bool zAxis_in)
	{
		ZQ_DImage<T> reproj_err_square_im(nPts, 1);
		T*& reproj_err_square = reproj_err_square_im.data();

		T init_rT[6] = { 0 };
		//memcpy(init_rT, rT, sizeof(T)* 6);
		if (!ZQ_PoseEstimation::PositCoplanar(nPts, X3, X2, fc,cc,kc,alpha_c,max_iter_posit, tol_E, init_rT, reproj_err_square,zAxis_in))
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
			if (!pose_estimation_with_init(nPts, X3, X2, fc, cc, kc, alpha_c, max_iter_levmar, tmp_rT, cur_avg_E_square,zAxis_in))
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


	/*pose estimation based on Lev-Mar optimization, need good initialization.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_PoseEstimation::_pose_estimation_fun(const T* p, T* hx, int m, int n, const void* data)
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

		if (!ZQ_CameraCalibration::project_points_fun(nPts, X3, p, fc, cc, kc, alpha_c, tmp_X2,zAxis_in))
			return false;

		ZQ_MathBase::VecMinus(nPts * 2, tmp_X2, X2, hx);
		return true;
	}

	/*pose estimation based on Lev-Mar optimization, need good initialization.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_PoseEstimation::_pose_estimation_jac(const T* p, T* jx, int m, int n, const void* data)
	{
		const Pose_Estimation_Data_Header<T>* ptr = (const Pose_Estimation_Data_Header<T>*)data;

		int nPts = ptr->nPts;
		const T* X3 = ptr->X3;
		const T* fc = ptr->fc_cc_alpha_kc;
		const T* cc = ptr->fc_cc_alpha_kc + 2;
		const T alpha_c = ptr->fc_cc_alpha_kc[4];
		const T* kc = ptr->fc_cc_alpha_kc + 5;
		bool zAxis_in = ptr->zAxis_in;

		if (!ZQ_CameraCalibration::project_points_jac(nPts, X3, p, fc, cc, kc, alpha_c, jx, (T*)0, (T*)0, (T*)0, (T*)0,zAxis_in))
			return false;
		return true;
	}

	/*pose estimation based on Lev-Mar optimization, need good initialization.
	left hand coordinates.
	*/
	template<class T>
	bool ZQ_PoseEstimation::pose_estimation_with_init(int nPts, const T* X3, const T* X2, const T fc[2], const T cc[2], const T kc[5], const T alpha_c, int max_iter, T* rT, double& avg_err_square,bool zAxis_in)
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
