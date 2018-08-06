#ifndef _ZQ_SHAPE_DEFORMATION_H_
#define _ZQ_SHAPE_DEFORMATION_H_
#pragma once

#include "ZQ_PCGSolver.h"
#include "ZQ_SparseMatrix.h"
#include "ZQ_Matrix.h"
#include "ZQ_MathBase.h"
#include "ZQ_SVD.h"
#include <vector>
#include "ZQ_Quaternion.h"
#include <map>
#include <typeinfo>

namespace ZQ
{
	/*
	refer to the paper:
	As-Rigid-As-Possible Shape Interpolation, 2000.
	*/
	template<class T>
	class ZQ_ShapeInterpolation
	{
		struct _mat22
		{
			T val[4];
		};
		struct _mat46
		{
			T val[24];
		};
	public:
		ZQ_ShapeInterpolation();
		~ZQ_ShapeInterpolation();

	private:
		int triangle_num;
		int point_num;
		int pivot_id;
		int unknown_num;
		std::vector<int> x_index;
		std::vector<int> y_index;
		std::vector<int> indices;
		std::vector<T> src_pos;
		std::vector<T> dst_pos;
		bool symmetric;
		std::vector<_mat22> R;
		std::vector<_mat22> S;
		std::vector<_mat22> R_back;
		std::vector<_mat22> S_back;
		std::vector<double> map_from_pivot_to_B;
		taucs_ccs_matrix* mat_from_u_to_B;
	public:
		/*indices, verts1, verts2 will be copied in this function*/
		bool BuildMatrix(int nTri, const int* indices, int nPts, const T* verts1, const T* verts2, bool symmetric = true, int pivot_id = 0);

		/*make sure out_verts has the same size as verts1, verts2*/
		bool Interpolation(float t, T* out_verts, int max_iter = 100);

	private:
		void _clear();

		/*
		A p_i + l = q_i, i \in {1,2,3}
		=> A p12 = q12
		   A p13 = q13
		   Donate P = [p12 p13], Q = [q12 q13],
		=> A P = Q
		=> A = Q P^-1
		*/
		bool _compute_A(const T* p1, const T* p2, const T* p3, const T* q1, const T* q2, const T* q3, _mat22& A);
		
		/* $B p_i + l = q_i, i \in {1,2,3}$,
		donate $\mathbf{q} = (q1x, q1y, q2x, q2y, q3x, q3y)^T$.
		Let $p_i$ be constant, then $B$ is linear with $\mathbf{q}$
			B = C \mathbf{q}, C \in R^{4\times6}.
		*/
		bool _compute_Coeff_of_B(const T* p1, const T* p2, const T* p3, _mat46& coeff);
		
		bool _compute_R_S_from_A(const _mat22& A, _mat22& R, _mat22& S);

		void _compute_A_from_R_S_t(const _mat22& R, const _mat22& S, float t ,_mat22& A);
	};


	/*******************************   definitions   **********************************/

	template<class T>
	ZQ_ShapeInterpolation<T>::ZQ_ShapeInterpolation()
	{
		triangle_num = 0;
		point_num = 0;
		mat_from_u_to_B = 0;
	}

	template<class T>
	ZQ_ShapeInterpolation<T>::~ZQ_ShapeInterpolation()
	{
		_clear();
	}

	/*indices, verts1, verts2 will be copied in this function*/
	template<class T>
	bool ZQ_ShapeInterpolation<T>::BuildMatrix(int nTri, const int* indices, int nPts, const T* verts1, const T* verts2, bool symmetric/* = true*/, int pivot_id/* = 0*/)
	{
		if (nTri <= 0 || nPts <= 0 || indices == 0 || verts1 == 0 || verts2 == 0 || pivot_id < 0 || pivot_id >= nPts)
			return false;

		_clear();

		this->symmetric = symmetric;
		triangle_num = nTri;
		point_num = nPts;
		this->pivot_id = pivot_id;
		this->indices.resize(nTri * 3);
		src_pos.resize(nPts * 2);
		dst_pos.resize(nPts * 2);
		memcpy(&(this->indices[0]), indices, sizeof(int) * 3 * nTri);
		memcpy(&src_pos[0], verts1, sizeof(T)*nPts * 2);
		memcpy(&dst_pos[0], verts2, sizeof(T)*nPts * 2);
		x_index.resize(point_num);
		y_index.resize(point_num);
		int idx = 0;
		for (int i = 0; i < point_num; i++)
		{
			if (pivot_id == i)
			{
				x_index[i] = -1;
				y_index[i] = -1;
			}
			else
			{
				x_index[i] = idx++;
				y_index[i] = idx++;
			}
		}
		unknown_num = idx;

		R.resize(triangle_num);
		S.resize(triangle_num);
		if (symmetric)
		{
			R_back.resize(triangle_num);
			S_back.resize(triangle_num);
		}
		
		_mat22 A;
		_mat46 coeff;
		
		int dim = triangle_num * 4;
		int row_num = symmetric ? (dim + dim) : dim;
		ZQ_SparseMatrix<float> map_u_to_B(row_num, unknown_num, 4);
		map_from_pivot_to_B.resize(row_num*2);
		memset(&map_from_pivot_to_B[0], 0, sizeof(T)*row_num*2);
		for (int i = 0; i < triangle_num; i++)
		{
			int v_id0 = indices[i * 3 + 0];
			int v_id1 = indices[i * 3 + 1];
			int v_id2 = indices[i * 3 + 2];
			if (!_compute_A(&src_pos[0] + v_id0 * 2, &src_pos[0] + v_id1 * 2, &src_pos[0] + v_id2 * 2,
				&dst_pos[0] + v_id0 * 2, &dst_pos[0] + v_id1 * 2, &dst_pos[0] + v_id2 * 2, A))
			{
				_clear();
				return false;
			}
			if (!_compute_R_S_from_A(A, R[i], S[i]))
			{
				_clear();
				return false;
			}
			if (!_compute_Coeff_of_B(&src_pos[0] + v_id0 * 2, &src_pos[0] + v_id1 * 2, &src_pos[0] + v_id2 * 2, coeff))
			{
				_clear();
				return false;
			}
			for (int h = 0; h < 4; h++)
			{
				if (coeff.val[h * 6 + 0] != 0)
				{
					int cur_idx = x_index[v_id0];
					if (cur_idx >= 0)
						map_u_to_B.AddTo(i * 4 + h, cur_idx, coeff.val[h * 6 + 0]);
					else
						map_from_pivot_to_B[(i * 4 + h) * 2 + 0] = coeff.val[h * 6 + 0];
				}
				if (coeff.val[h * 6 + 1] != 0)
				{
					int cur_idx = y_index[v_id0];
					if (cur_idx >= 0)
						map_u_to_B.AddTo(i * 4 + h, cur_idx, coeff.val[h * 6 + 1]);
					else
						map_from_pivot_to_B[(i * 4 + h) * 2 + 1] = coeff.val[h * 6 + 1];
				}
				if (coeff.val[h * 6 + 2] != 0)
				{
					int cur_idx = x_index[v_id1];
					if (cur_idx >= 0)
						map_u_to_B.AddTo(i * 4 + h, cur_idx, coeff.val[h * 6 + 2]);
					else
						map_from_pivot_to_B[(i * 4 + h) * 2 + 0] = coeff.val[h * 6 + 2];
				}
				if (coeff.val[h * 6 + 3] != 0)
				{
					int cur_idx = y_index[v_id1];
					if (cur_idx >= 0)
						map_u_to_B.AddTo(i * 4 + h, cur_idx, coeff.val[h * 6 + 3]);
					else
						map_from_pivot_to_B[(i * 4 + h) * 2 + 1] = coeff.val[h * 6 + 3];
				}
				if (coeff.val[h * 6 + 4] != 0)
				{
					int cur_idx = x_index[v_id2];
					if (cur_idx >= 0)
						map_u_to_B.AddTo(i * 4 + h, cur_idx, coeff.val[h * 6 + 4]);
					else
						map_from_pivot_to_B[(i * 4 + h) * 2 + 0] = coeff.val[h * 6 + 4];
				}
				if (coeff.val[h * 6 + 5] != 0)
				{
					int cur_idx = y_index[v_id2];
					if (cur_idx >= 0)
						map_u_to_B.AddTo(i * 4 + h, cur_idx, coeff.val[h * 6 + 5]);
					else
						map_from_pivot_to_B[(i * 4 + h) * 2 + 1] = coeff.val[h * 6 + 5];
				}
			}

			if (!symmetric)
				continue;

			if (!_compute_A(&dst_pos[0] + v_id0 * 2, &dst_pos[0] + v_id1 * 2, &dst_pos[0] + v_id2 * 2,
				&src_pos[0] + v_id0 * 2, &src_pos[0] + v_id1 * 2, &src_pos[0] + v_id2 * 2, A))
			{
				_clear();
				return false;
			}
			if (!_compute_R_S_from_A(A, R_back[i], S_back[i]))
			{
				_clear();
				return false;
			}
			if (!_compute_Coeff_of_B(&dst_pos[0] + v_id0 * 2, &dst_pos[0] + v_id1 * 2, &dst_pos[0] + v_id2 * 2, coeff))
			{
				_clear();
				return false;
			}
			for (int h = 0; h < 4; h++)
			{
				if (coeff.val[h * 6 + 0] != 0)
				{
					int cur_idx = x_index[v_id0];
					if (cur_idx >= 0)
						map_u_to_B.AddTo(dim+i * 4 + h, cur_idx, coeff.val[h * 6 + 0]);
					else
						map_from_pivot_to_B[(dim + i * 4 + h) * 2 + 0] = coeff.val[h * 6 + 0];
				}
				if (coeff.val[h * 6 + 1] != 0)
				{
					int cur_idx = y_index[v_id0];
					if (cur_idx >= 0)
						map_u_to_B.AddTo(dim + i * 4 + h, cur_idx, coeff.val[h * 6 + 1]);
					else
						map_from_pivot_to_B[(dim + i * 4 + h) * 2 + 1] = coeff.val[h * 6 + 1];
				}
				if (coeff.val[h * 6 + 2] != 0)
				{
					int cur_idx = x_index[v_id1];
					if (cur_idx >= 0)
						map_u_to_B.AddTo(dim + i * 4 + h, cur_idx, coeff.val[h * 6 + 2]);
					else
						map_from_pivot_to_B[(dim + i * 4 + h) * 2 + 0] = coeff.val[h * 6 + 2];
				}
				if (coeff.val[h * 6 + 3] != 0)
				{
					int cur_idx = y_index[v_id1];
					if (cur_idx >= 0)
						map_u_to_B.AddTo(dim + i * 4 + h, cur_idx, coeff.val[h * 6 + 3]);
					else
						map_from_pivot_to_B[(dim + i * 4 + h) * 2 + 1] = coeff.val[h * 6 + 3];
				}
				if (coeff.val[h * 6 + 4] != 0)
				{
					int cur_idx = x_index[v_id2];
					if (cur_idx >= 0)
						map_u_to_B.AddTo(dim + i * 4 + h, cur_idx, coeff.val[h * 6 + 4]);
					else
						map_from_pivot_to_B[(dim + i * 4 + h) * 2 + 0] = coeff.val[h * 6 + 4];
				}
				if (coeff.val[h * 6 + 5] != 0)
				{
					int cur_idx = y_index[v_id2];
					if (cur_idx >= 0)
						map_u_to_B.AddTo(dim + i * 4 + h, cur_idx, coeff.val[h * 6 + 5]);
					else
						map_from_pivot_to_B[(dim + i * 4 + h) * 2 + 1] = coeff.val[h * 6 + 5];
				}
			}
		}
		mat_from_u_to_B = map_u_to_B.ExportCCS(TAUCS_DOUBLE);
		return true;
	}

	/*make sure out_verts has the same size as verts1, verts2*/
	template<class T>
	bool ZQ_ShapeInterpolation<T>::Interpolation(float t, T* out_verts, int max_iter/* = 100*/)
	{
		if (out_verts == 0 || triangle_num == 0 || R.size() != triangle_num || S.size() != triangle_num)
			return false;

		double pivot_vert[2] = {
			src_pos[pivot_id * 2 + 0] * (1 - t) + dst_pos[pivot_id * 2 + 0] * t,
			src_pos[pivot_id * 2 + 1] * (1 - t) + dst_pos[pivot_id * 2 + 1] * t
		};

		int dim = triangle_num * 4;
		int row_num = symmetric ? (dim + dim) : dim;
		std::vector<double> b(row_num);
		ZQ_MathBase::MatrixMul(&map_from_pivot_to_B[0], pivot_vert, row_num, 2, 1, &b[0]);

		_mat22 A;
		for (int i = 0; i < triangle_num; i++)
		{
			_compute_A_from_R_S_t(R[i], S[i], t, A);
			for(int j = 0;j < 4;j++)
				b[i * 4 + j] = -b[i * 4 + j] + A.val[j];
			if (!symmetric)
				continue;
			_compute_A_from_R_S_t(R_back[i], S_back[i], 1.0f-t, A);
			for (int j = 0; j < 4; j++)
				b[dim+i * 4 + j] = -b[dim+i * 4 + j] + A.val[j];
		}
		
		std::vector<double> x0(unknown_num);
		std::vector<double> x(unknown_num);
		memset(&x0[0], 0, sizeof(double)*unknown_num);
		
		int it;
		ZQ_PCGSolver::PCG_sparse_unsquare(mat_from_u_to_B, &b[0], &x0[0], max_iter, 1e-9, &x[0], it, false);

		for (int i = 0; i < point_num; i++)
		{
			out_verts[i * 2 + 0] = x_index[i] >= 0 ? x[x_index[i]] : pivot_vert[0];
			out_verts[i * 2 + 1] = y_index[i] >= 0 ? x[y_index[i]] : pivot_vert[1];
		}
		return true;
	}


	template<class T>
	void ZQ_ShapeInterpolation<T>::_clear()
	{
		triangle_num = 0;
		point_num = 0;
		unknown_num = 0;
		R.clear();
		S.clear();
		map_from_pivot_to_B.clear();
		x_index.clear();
		y_index.clear();
		src_pos.clear();
		dst_pos.clear();
		indices.clear();
		if (mat_from_u_to_B)
		{
			ZQ_TaucsBase::ZQ_taucs_ccs_free(mat_from_u_to_B);
			mat_from_u_to_B = 0;
		}
	}

	template<class T>
	bool ZQ_ShapeInterpolation<T>::_compute_A(const T* p1, const T* p2, const T* p3, const T* q1, const T* q2, const T* q3, _mat22& A)
	{
		/*
		A p_i + l = q_i, i \in {1,2,3}
		=> A p12 = q12
		   A p13 = q13
		=> A P = Q
		=> A = Q P^-1
		*/

		T p12[2] = { p2[0] - p1[0], p2[1] - p1[1] };
		T p13[2] = { p3[0] - p1[0], p3[1] - p1[1] };
		T q12[2] = { q2[0] - q1[0], q2[1] - q1[1] };
		T q13[2] = { q3[0] - q1[0], q3[1] - q1[1] };
		ZQ_Matrix<double> matP(2, 2), matP_1(2,2), matQ(2,2);
		matP.SetData(0, 0, p12[0]); 
		matP.SetData(1, 0, p12[1]);
		matP.SetData(0, 1, p13[0]);
		matP.SetData(1, 1, p13[1]);
		matQ.SetData(0, 0, q12[0]);
		matQ.SetData(1, 0, q12[1]);
		matQ.SetData(0, 1, q13[0]);
		matQ.SetData(1, 1, q13[1]);
		if (!ZQ_SVD::Invert(matP, matP_1))
			return false;

		ZQ_Matrix<double> matA = matQ*matP_1;
		bool flag;
		A.val[0] = matA.GetData(0, 0, flag);
		A.val[1] = matA.GetData(0, 1, flag);
		A.val[2] = matA.GetData(1, 0, flag);
		A.val[3] = matA.GetData(1, 1, flag);

		return true;
	}

	/* $B p_i + l = q_i, i \in {1,2,3}$,
	donate $\mathbf{q} = (q1x, q1y, q2x, q2y, q3x, q3y)^T$.
	Let $p_i$ be constant, then $B$ is linear with $\mathbf{q}$
	B = C \mathbf{q}, C \in R^{4\times6}.
	*/
	template<class T>
	bool ZQ_ShapeInterpolation<T>::_compute_Coeff_of_B(const T* p1, const T* p2, const T* p3, _mat46& coeff)
	{
		/*
		A p_i + l = q_i, i \in {1,2,3}
		=> A p12 = q12
		A p13 = q13
		=> A P = Q
		=> A = Q P^-1
		*/

		T p12[2] = { p2[0] - p1[0], p2[1] - p1[1] };
		T p13[2] = { p3[0] - p1[0], p3[1] - p1[1] };
		ZQ_Matrix<double> matP(2, 2), matP_1(2, 2);
		matP.SetData(0, 0, p12[0]);
		matP.SetData(1, 0, p12[1]);
		matP.SetData(0, 1, p13[0]);
		matP.SetData(1, 1, p13[1]);
		if (!ZQ_SVD::Invert(matP, matP_1))
			return false;

		/*
		Donate P^-1 = (a0,a1,a2,a3)^T, Q = (Q0,Q1,Q2,Q3)^T,
		then B = Q P^-1 can be written as:
		b0 = Q0*a0 + Q1*a2 = a0*(q2x-q1x) + a2*(q3x-q1x) = (-a0-a2,     0,     a0,     0,     a2,     0) * \mathbf{q}
		b1 = Q0*a1 + Q1*a3 = a1*(q2x-q1x) + a3*(q3x-q1x) = (-a1-a3,     0,     a1,     0,     a3,     0) * \mathbf{q}
		b2 = Q2*a0 + Q3*a2 = a0*(q2y-q1y) + a2*(q3y-q1y) = (     0,-a0-a2,      0,    a0,      0,    a2) * \mathbf{q}
		b3 = Q2*a1 + Q3*a3 = a1*(q2y-q1y) + a3*(q3y-q1y) = (     0,-a1-a3,      0,    a1,      0,    a3) * \mathbf{q}
		*/

		const double* a = matP_1.GetDataPtr();
		memset(coeff.val, 0, sizeof(T) * 24);
		coeff.val[0 * 6 + 0] = -a[0] - a[2];	coeff.val[0 * 6 + 2] = a[0];	coeff.val[0 * 6 + 4] = a[2];
		coeff.val[1 * 6 + 0] = -a[1] - a[3];	coeff.val[1 * 6 + 2] = a[1];	coeff.val[1 * 6 + 4] = a[3];
		coeff.val[2 * 6 + 1] = -a[0] - a[2];	coeff.val[2 * 6 + 3] = a[0];	coeff.val[2 * 6 + 5] = a[2];
		coeff.val[3 * 6 + 1] = -a[1] - a[3];	coeff.val[3 * 6 + 3] = a[1];	coeff.val[3 * 6 + 5] = a[3];
		return true;
	}

	template<class T>
	bool ZQ_ShapeInterpolation<T>::_compute_R_S_from_A(const _mat22& A, _mat22& R, _mat22& S)
	{
		ZQ_Matrix<double> matA(2, 2), matU(2,2), matD(2,2), matV(2,2);
		matA.SetData(0, 0, A.val[0]);
		matA.SetData(0, 1, A.val[1]);
		matA.SetData(1, 0, A.val[2]);
		matA.SetData(1, 1, A.val[3]);
		if (!ZQ_SVD::Decompose(matA, matU, matD, matV))
			return false;

		ZQ_Matrix<double>& mat_R_a = matU;
		ZQ_Matrix<double>& mat_R_b_T = matV;
		ZQ_Matrix<double> mat_R_b = matV.GetTransposeMatrix();

		ZQ_Matrix<double> mat_R_r = mat_R_a*mat_R_b;
		ZQ_Matrix<double> mat_S = mat_R_b_T* matD * mat_R_b;
		
		bool flag;
		R.val[0] = mat_R_r.GetData(0, 0, flag);
		R.val[1] = mat_R_r.GetData(0, 1, flag);
		R.val[2] = mat_R_r.GetData(1, 0, flag);
		R.val[3] = mat_R_r.GetData(1, 1, flag);
		S.val[0] = mat_S.GetData(0, 0, flag);
		S.val[1] = mat_S.GetData(0, 1, flag);
		S.val[2] = mat_S.GetData(1, 0, flag);
		S.val[3] = mat_S.GetData(1, 1, flag);
		return true;
	}

	template<class T>
	void ZQ_ShapeInterpolation<T>::_compute_A_from_R_S_t(const _mat22& R, const _mat22& S, float t, _mat22& A)
	{
		/*
		A = R((1-t)I + tS)
		*/
		T tmp1[4] = { 1 - t,0,0,1 - t };
		T tmp2[4] = { S.val[0] * t,S.val[1] * t, S.val[2] * t, S.val[3] * t };
		T tmp3[4];

		T RR[9] = {
			R.val[0], R.val[1], 0,
			R.val[2], R.val[3], 0,
			0, 0, 1
		};
		ZQ_Quaternion<T> quat, quat_identity;
		ZQ_Quaternion<T>::Rot2Quat(RR, quat);
		ZQ_Quaternion<T> quat_t = ZQ_Quaternion<T>::Slerp(quat_identity, quat, t);
		ZQ_Quaternion<T>::Quat2Rot(quat_t,RR);
		T R_t[4] = { RR[0],RR[1],RR[3],RR[4] };
		ZQ_MathBase::VecPlus(4, tmp1, tmp2, tmp3);
		ZQ_MathBase::MatrixMul(R_t, tmp3, 2, 2, 2, A.val);
	}
}

#endif