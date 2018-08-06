#ifndef _ZQ_SVD_H_
#define _ZQ_SVD_H_
#pragma once

#include "ZQ_Matrix.h"
#include <typeinfo>

namespace ZQ
{
	class ZQ_SVD
	{
	public:
		/*  A = U * S * V^T  */
		template<class T>
		static bool Decompose(const ZQ_Matrix<T>& A, ZQ_Matrix<T>& U, ZQ_Matrix<T>& SS, ZQ_Matrix<T>& V);

		template<class T>
		static bool Invert(const ZQ_Matrix<T>& mat, ZQ_Matrix<T>& invMat);

		template<class T>
		static bool Solve(const ZQ_Matrix<T>& A, ZQ_Matrix<T>& x, const ZQ_Matrix<T>& b);
	};

	/**************** definitions *****************************/

	/*  A = U * S * V^T  */
	template<class T>
	bool ZQ_SVD::Decompose(const ZQ_Matrix<T>& A, ZQ_Matrix<T>& U, ZQ_Matrix<T>& SS, ZQ_Matrix<T>& V)
	{
		int m = A.GetRowDim();
		int n = A.GetColDim();
		int sdim = __min(m,n);

		if(U.GetRowDim() != m || U.GetColDim() != sdim)
			return false;

		if(V.GetRowDim() != n || V.GetColDim() != sdim)
			return false;

		if(SS.GetRowDim() != sdim || SS.GetColDim() != sdim)
			return false;

		const T* A_ptr = A.GetDataPtr();
		T* U_ptr = U.GetDataPtr();
		T* V_ptr = V.GetDataPtr();
		T* S_ptr = SS.GetDataPtr();

		if (_strcmpi(typeid(T).name(), "double") == 0)
		{
			return ZQ_MathBase::SVD_Decompose(A_ptr, m, n, U_ptr, S_ptr, V_ptr);
		}
		else
		{
			double* A1 = new double[m*n];
			double* SS1 = new double[sdim*sdim];
			double* U1 = new double[m*sdim];
			double* V1 = new double[n*sdim];
			for (int i = 0; i < m*n; i++)
			{
				A1[i] = A_ptr[i];
			}
			if (!ZQ_MathBase::SVD_Decompose(A1, m, n, U1, SS1, V1))
			{
				delete[]A1;
				delete[]SS1;
				delete[]U1;
				delete[]V1;
				return false;
			}
			for (int i = 0; i < sdim*sdim; i++)
				S_ptr[i] = SS1[i];
			for (int i = 0; i < m*sdim; i++)
				U_ptr[i] = U1[i];
			for (int i = 0; i < n*sdim; i++)
				V_ptr[i] = V1[i];
			delete[]A1;
			delete[]SS1;
			delete[]U1;
			delete[]V1;
			return true;
		}
	}

	template<class T>
	bool ZQ_SVD::Invert(const ZQ_Matrix<T>& mat, ZQ_Matrix<T>& invMat)
	{
		int m = mat.GetRowDim();
		int n = mat.GetColDim();
		if(m == 0 || n == 0)
			return false;

		if(m != invMat.GetColDim() || n != invMat.GetRowDim())
			return false;

		int sdim = __min(m,n);
		ZQ_Matrix<T> U(m,sdim),V(n,sdim),SS(sdim,sdim);

		if(!Decompose(mat,U,SS,V))
			return false;

		T* S_ptr = SS.GetDataPtr();

		for(int i = 0;i < sdim;i++)
		{
			if(S_ptr[i*sdim+i] != 0)
				S_ptr[i*sdim+i] = 1.0/S_ptr[i*sdim+i];
		}

		T* U_ptr = U.GetDataPtr();
		T* V_ptr = V.GetDataPtr();
		T* inv_ptr = invMat.GetDataPtr();
		T* VS = new T[n*sdim];

		for(int i = 0;i < n;i++)
		{
			for(int j = 0;j < sdim;j++)
			{
				VS[i*sdim+j] = V_ptr[i*sdim+j]*S_ptr[j*sdim+j];
			}
		}

		for(int i = 0;i < n;i++)
		{
			for(int j = 0;j < m;j++)
			{
				T sum = 0;
				for(int k = 0;k < sdim;k++)
				{
					sum += VS[i*sdim+k]*U_ptr[j*sdim+k];
				}
				inv_ptr[i*m+j] = sum;
			}
		}
		delete []VS;

		return true;
	}

	template<class T>
	bool ZQ_SVD::Solve(const ZQ_Matrix<T>& A, ZQ_Matrix<T>& x, const ZQ_Matrix<T>& b)
	{
		int m = A.GetRowDim();
		int dim = A.GetColDim();
		int n = b.GetColDim();
		if( x.GetRowDim() != dim || x.GetColDim() != n || b.GetRowDim() != m)
			return false;

		ZQ_Matrix<T> invA(dim,m);
		Invert(A,invA);

		T* inv_ptr = invA.GetDataPtr();
		T* b_ptr = b.GetDataPtr();
		T* x_ptr = x.GetDataPtr();


		for(int i = 0;i < dim;i++)
		{
			for(int j = 0;j < n;j++)
			{
				T sum = 0;
				for(int k = 0;k < m;k++)
				{
					sum += inv_ptr[i*m+k]*b_ptr[k*n+j];
				}
				x_ptr[i*n+j] = sum;
			}
		}
		return true;
	}
}

#endif