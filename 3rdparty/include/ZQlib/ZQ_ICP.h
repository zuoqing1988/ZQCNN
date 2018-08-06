#ifndef _ZQ_ICP_H_
#define _ZQ_ICP_H_
#pragma once

#include "ZQ_SVD.h"
#include "ZQ_KDTree.h"
#include "ZQ_MathBase.h"


namespace ZQ
{
	class ZQ_ICP
	{
	public:
		template<class T>
		static bool IterativeClosestPoint(const T* model, int nModel, const T* data, int nData, int maxIter, double thresh, T R[9], T t[3]);

		template<class T>
		static bool IterativeClosestPoint(const T* model, int nModel, const T* data, int nData, int maxIter, double thresh, const T R0[9], const T t0[3], T R[9], T t[3]);

	private:
		template<class T>
		static bool _estimateNaiveInitRt(const T* model, int nModel, const T* data, int nData, T R[9], T t[3]);
	};

	/******************  definitions  *****************/

	template<class T>
	bool ZQ_ICP::IterativeClosestPoint(const T* model, int nModel, const T* data, int nData, int maxIter, double thresh, T R[9], T t[3])
	{
		if (model == 0 || data == 0 || nModel == 0 || nData == 0)
			return false;

		T R0[9], t0[3];
		if (!_estimateNaiveInitRt(model, nModel, data, nData, R0, t0))
			return false;

		if (!IterativeClosestPoint(model, nModel, data, nData, maxIter, thresh, R0, t0, R, t))
			return false;
		return true;
	}

	template<class T>
	bool ZQ_ICP::IterativeClosestPoint(const T* model, int nModel, const T* data, int nData, int maxIter, double thresh, const T R0[9], const T t0[3],
		T R[9], T t[3])
	{
		if (model == 0 || data == 0 || nModel == 0 || nData == 0)
			return false;

		int dim = 3;

		ZQ_KDTree<T> tree;
		if(!tree.BuildKDTree(model, nModel, 3, 10))
		{
			return false;
		}
		
		T* mean_data = new T[dim];
		T* mean_model2 = new T[dim];
		memset(mean_data, 0, sizeof(T)*dim);
		for (int i = 0; i < nData; i++)
		{
			for (int d = 0; d < dim; d++)
				mean_data[d] += data[i*dim + d];
		}
		for (int d = 0; d < dim; d++)
			mean_data[d] /= nData;


		memcpy(R, R0, sizeof(T) * 9);
		memcpy(t, t0, sizeof(T) * 3);

		T* data2 = new T[nData*dim];
		T* model2 = new T[nData*dim];
		for (int it = 0; it < maxIter; it++)
		{
			/*compute data2 Begin*/
			for (int i = 0; i < nData; i++)
			{
				for (int di = 0; di < dim; di++)
				{
					data2[i*dim + di] = t[di];
					for (int dj = 0; dj < dim; dj++)
					{
						data2[i*dim + di] += R[di*dim + dj] * data[i*dim + dj];
					}
				}
			}
			/*compute data2 End*/

			double cur_error = 0;
			for (int i = 0; i < nData; i++)
			{
				for (int d = 0; d < dim; d++)
					cur_error += (model2[i*dim + d] - data2[i*dim + d])*(model2[i*dim + d] - data2[i*dim + d]);
			}

			if (cur_error < thresh*thresh)
				break;


			/**  find corresponding points  Begin  **/
			for (int i = 0; i < nData; i++)
			{
				int idx;
				T dist;
				if (!tree.AnnSearch(data2 + dim*i, 1, &idx, &dist))
				{
					delete[]data2;
					delete[]model2;
					delete[]mean_data;
					delete[]mean_model2;
					return false;
				}
				memcpy(model2 + dim*i, model + dim*idx, sizeof(T)*dim);
			}
			/**  find corresponding points  End  **/

			/* solve R, t Begin */

			memset(mean_model2, 0, sizeof(T)*dim);
			for (int i = 0; i < nData; i++)
			{
				for (int d = 0; d < dim; d++)
					mean_model2[d] += model2[i*dim + d];
			}
			for (int d = 0; d < dim; d++)
				mean_model2[d] /= nData;


			T* C = new T[dim*dim];
			memset(C, 0, sizeof(T)*dim*dim);
			for (int i = 0; i < nData; i++)
			{
				for (int di = 0; di < dim; di++)
				{
					for (int dj = 0; dj < dim; dj++)
						C[di*dim + dj] += data[i*dim + di] * model2[i*dim + dj];
				}
			}
			for (int di = 0; di < dim; di++)
			{
				for (int dj = 0; dj < dim; dj++)
					C[di*dim + dj] -= nData*mean_data[di] * mean_model2[dj];
			}

			ZQ_Matrix<T> Cmat(dim, dim), Umat(dim, dim), Smat(dim, dim), Vmat(dim, dim);
			for (int di = 0; di < dim; di++)
			{
				for (int dj = 0; dj < dim; dj++)
				{
					Cmat.SetData(di, dj, C[di*dim + dj]);
				}
			}
			ZQ_SVD::Decompose(Cmat, Umat, Smat, Vmat);

			for (int di = 0; di < dim; di++)
			{
				for (int dj = 0; dj < dim; dj++)
				{
					bool tmpflag;
					R[di*dim + dj] = 0;
					for (int k = 0; k < dim; k++)
						R[di*dim + dj] += Vmat.GetData(di, k, tmpflag)*Umat.GetData(dj, k, tmpflag);
				}
			}

			for (int di = 0; di < dim; di++)
			{
				t[di] = mean_model2[di];
				for (int dj = 0; dj < dim; dj++)
					t[di] -= R[di*dim + dj] * mean_data[dj];
			}

			/* solve R, t End */
		}


		delete[]data2;
		delete[]model2;
		delete[]mean_data;
		delete[]mean_model2;
		return true;
	}

	template<class T>
	bool ZQ_ICP::_estimateNaiveInitRt(const T* model, int nModel, const T* data, int nData, T R[9], T t[3])
	{
		if (model == 0 || data == 0 || nModel <= 0 || nData <= 0)
			return false;

		const int dim = 3;
		T model_c[dim] = { 0 };
		T data_c[dim] = { 0 };

		for (int i = 0; i < nModel; i++)
		{
			for (int d = 0; d < dim; d++)
				model_c[d] += model[i*dim + d];
		}
		for (int d = 0; d < dim; d++)
			model_c[d] /= nModel;

		for (int i = 0; i < nData; i++)
		{
			for (int d = 0; d < dim; d++)
				data_c[d] += data[i*dim + d];
		}
		for (int d = 0; d < dim; d++)
			data_c[d] /= nData;

		memset(R, 0, sizeof(T) * 9);
		for (int d = 0; d < dim; d++)
			R[d*dim + d] = 1;

		T Rqc[dim];
		ZQ_MathBase::MatrixMul(R, data_c, dim, dim, 1, Rqc);
		for (int d = 0; d < dim; d++)
			t[d] = model_c[d] - Rqc[d];
		return true;
	}
}

#endif
