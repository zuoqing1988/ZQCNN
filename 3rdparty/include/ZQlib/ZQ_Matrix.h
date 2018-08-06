#ifndef _ZQ_MATRIX_H_
#define _ZQ_MATRIX_H_
#pragma once
#include <string.h>
#include <assert.h>
#include <malloc.h>
#include "ZQ_MathBase.h"

namespace ZQ
{

	template<class T>
	class ZQ_Matrix
	{
	private:
		int nRow,nCol;
		T* data;

		ZQ_Matrix();

	public:
		ZQ_Matrix(unsigned int nRow, unsigned int nCol);
		~ZQ_Matrix();

		ZQ_Matrix(const ZQ_Matrix& other);

		void operator = (const ZQ_Matrix& other);

	public:

		int GetRowDim() const {return nRow;}

		int GetColDim() const {return nCol;}

		T GetData(int i, int j, bool& flag) const;

		bool SetData(int i, int j, T value);

		bool AddWith(int i, int j, T value);

		T* GetDataPtr() const{ return this->data;}

		ZQ_Matrix* Clone() const;

		void Transpose();
		
		ZQ_Matrix GetTransposeMatrix() const;

		ZQ_Matrix operator *(const ZQ_Matrix& mat) const;

		bool Reshape(int dst_row, int dst_col);

		void Reset();

		static bool MatrixMul(const ZQ_Matrix& src1, const ZQ_Matrix& src2, ZQ_Matrix& dst);
	};

	/********************* definitions ***************************/

	template<class T>
	ZQ_Matrix<T>::ZQ_Matrix()
	{
		nRow = 0;
		nCol = 0;
		data = 0;
	}

	template<class T>
	ZQ_Matrix<T>::ZQ_Matrix(unsigned int nRow, unsigned int nCol)
	{
		this->nRow = nRow;
		this->nCol = nCol;
		this->data = (T*)malloc(sizeof(T)*nRow*nCol);
		memset(this->data,0,sizeof(T)*nRow*nCol);
	}

	template<class T>
	ZQ_Matrix<T>::~ZQ_Matrix()
	{
		
		if(this->data)
		{
			free(this->data);
			this->data = 0;
		}
		nRow = 0;
		nCol = 0;
	}

	template<class T>
	ZQ_Matrix<T>::ZQ_Matrix(const ZQ_Matrix& other)
	{
		nRow = other.nRow;
		nCol = other.nCol;
		data = (T*)malloc(sizeof(T)*nRow*nCol);
		memcpy(data,other.data,sizeof(T)*nRow*nCol);
	}

	template<class T>
	void ZQ_Matrix<T>::operator = (const ZQ_Matrix& other)
	{
		if(data)
			free(data);

		nRow = other.nRow;
		nCol = other.nCol;
		data = (T*)malloc(sizeof(T)*nRow*nCol);
		memcpy(data,other.data,sizeof(T)*nRow*nCol);
	}

	template<class T>
	T ZQ_Matrix<T>::GetData(int i, int j, bool& flag) const
	{
		if(i < 0 || i >= nRow || j < 0 || j >= nCol)
		{
			flag = false;
			return 0;
		}
		else
		{
			flag = true;
			return data[i*nCol+j];
		}
	}

	template<class T>
	bool ZQ_Matrix<T>::SetData(int i, int j, T value)
	{
		if(i < 0 || i >= nRow || j < 0 || j >= nCol)
		{
			return false;
		}
		else
		{
			data[i*nCol+j] = value;
			return true;
		}
	}

	template<class T>
	bool ZQ_Matrix<T>::AddWith(int i, int j, T value)
	{
		if (i < 0 || i >= nRow || j < 0 || j >= nCol)
		{
			return false;
		}
		else
		{
			data[i*nCol + j] += value;
			return true;
		}
	}

	template<class T>
	ZQ_Matrix<T>* ZQ_Matrix<T>::Clone() const
	{
		ZQ_Matrix* tmp = new ZQ_Matrix(nRow,nCol);
		memcpy(tmp->data,this->data,sizeof(T)*nRow*nCol);
		return tmp;
	}

	template<class T>
	void ZQ_Matrix<T>::Transpose()
	{
		if(nRow > 0 && nCol > 0)
		{
			T* tmp_data = (T*)malloc(sizeof(T)*nRow*nCol);
			int r = nCol;
			int c = nRow;
			for(int i = 0;i < r;i++)
			{
				for(int j = 0;j < c;j++)
				{
					tmp_data[i*c+j] = this->data[j*nCol+i];
				}
			}
			memcpy(this->data,tmp_data,sizeof(T)*nRow*nCol);
			this->nRow = r;
			this->nCol = c;
			free(tmp_data);
		}
	}

	template<class T>
	ZQ_Matrix<T> ZQ_Matrix<T>::GetTransposeMatrix() const
	{
		ZQ_Matrix tmp(*this);
		tmp.Transpose();
		return tmp;
	}


	template<class T>
	ZQ_Matrix<T> ZQ_Matrix<T>::operator *(const ZQ_Matrix& mat) const
	{
		assert(nCol == mat.nRow);
		ZQ_Matrix tmp(nRow,mat.nCol);

		for(int i = 0;i < nRow;i++)
		{
			for(int j = 0;j < mat.nCol;j++)
			{
				T sum = 0;
				for(int k = 0;k < nCol;k++)
					sum += data[i*nCol+k] * mat.data[k*mat.nCol+j];
				tmp.data[i*mat.nCol+j] = sum;
			}
		}
		return tmp;
	}

	template<class T>
	bool ZQ_Matrix<T>::Reshape(int dst_row, int dst_col)
	{
		if(dst_row >= 0 && dst_col >= 0 && nRow*nCol == dst_row*dst_col)
		{
			nRow = dst_row;
			nCol = dst_col;
			return true;
		}
		else
		{
			return false;
		}
	}

	template<class T>
	void ZQ_Matrix<T>::Reset()
	{
		memset(data, 0, sizeof(T)*nRow*nCol);
	}

	template<class T>
	bool ZQ_Matrix<T>::MatrixMul(const ZQ_Matrix& src1, const ZQ_Matrix& src2, ZQ_Matrix& dst)
	{
		if(src1.nCol != src2.nRow)
			return false;
		if(dst.nRow != src1.nRow || dst.nCol != src2.nCol)
			return false;
		const T* src_ptr1 = src1.GetDataPtr();
		const T* src_ptr2 = src2.GetDataPtr();
		T* dst_ptr = dst.GetDataPtr();
		ZQ_MathBase::MatrixMul(src_ptr1,src_ptr2,src1.nRow,src1.nCol,src2.nCol,dst_ptr);
		return true;
	}

}

#endif