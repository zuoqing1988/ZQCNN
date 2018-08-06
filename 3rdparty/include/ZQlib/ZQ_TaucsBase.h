#ifndef _ZQ_TAUCS_BASE_H_
#define _ZQ_TAUCS_BASE_H_
#pragma once

#include "ZQ_taucs.h"
#include "ZQ_MathBase.h"
#include <vector>
#include <map>
#include <math.h>


#define ZQ_DOUBLE 0
#define ZQ_FLOAT 1
#define ZQ_NORM_ONE 1
#define ZQ_NORM_TWO 2
#define ZQ_NORM_INF 3
#define ZQ_SEARCH_ASCENT 0
#define ZQ_SEARCH_DESCEND 1 
#define ZQ_INF 1e12


namespace ZQ
{
	class ZQ_TaucsBase
	{
	public:
		template<class T>
		static taucs_ccs_matrix* ZQ_taucs_ccs_CreateMatrixFromColumns(std::vector< std::map<int,T> > & cols, int nRows, int flags)
		{
			if(flags != TAUCS_DOUBLE && flags != TAUCS_SINGLE)
				return 0;
			// count nnz:
			int nCols = (int)cols.size();

			int nnz = 0;
			for (int counter = 0; counter < nCols; counter++)
			{
				nnz += (int)cols[counter].size();
			}

			taucs_ccs_matrix *matC = ZQ_taucs_ccs_create(nRows,nCols,nnz,flags);
			if (matC == 0)
				return 0;


			// copy cols into matC
			std::map<int,T>::const_iterator rit;
			int rowptrC = 0;

			if (flags & TAUCS_DOUBLE)
			{
				for (int c = 0; c < nCols; c++)
				{
					matC->colptr[c] = rowptrC;
					for (rit = cols[c].begin(); rit != cols[c].end(); ++rit)
					{
						matC->rowind[rowptrC] = rit->first;
						matC->values.d[rowptrC] = rit->second;
						++rowptrC;
					}
				}
			}
			else
			{
				for (int c = 0; c < nCols; c++)
				{
					matC->colptr[c] = rowptrC;
					for (rit = cols[c].begin(); rit != cols[c].end(); ++rit)
					{
						matC->rowind[rowptrC] = rit->first;
						matC->values.s[rowptrC] = rit->second;
						++rowptrC;
					}
				}
			}
			

			matC->colptr[nCols] = nnz;
			return matC;
		}

		static void ZQ_taucs_ccs_free(taucs_ccs_matrix* A)
		{
			if(A)
			{
				if(A->flags & TAUCS_DOUBLE)
				{
					if(A->values.d)
						delete []A->values.d;
				}
				else if(A->flags & TAUCS_SINGLE)
				{
					if(A->values.s)
						delete []A->values.s;
				}
				if(A->colptr)
					delete []A->colptr;
				if(A->rowind)
					delete []A->rowind;
				delete A;
			}
		}

		static taucs_ccs_matrix* ZQ_taucs_ccs_create(const int row, const int col, const int nnz, const int flag)
		{
			if( row <= 0 || col <= 0 || nnz < 0 || ((flag != TAUCS_DOUBLE) && (flag != TAUCS_SINGLE)))
				return 0;
			bool sucflag = true;
			taucs_ccs_matrix* A = new taucs_ccs_matrix;
			A->flags = flag;

			A->m = row;
			A->n = col;
			A->rowind = 0;
			A->colptr = new int[col+1];
			if(A->colptr == 0)
			{
				sucflag = false;
				delete A;
				return 0;
			}
			if(nnz > 0)
			{
				A->rowind = new int[nnz];
				if(A->rowind == 0)
				{
					sucflag = false;
					delete []A->colptr;
					delete A;
					return 0;
				}
				if(flag == TAUCS_DOUBLE)
				{
					A->values.d = new double[nnz];
					if(A->values.d == 0)
					{
						sucflag = false;
						delete []A->rowind;
						delete []A->colptr;
						delete A;
					}
				}
				else
				{
					A->values.s = new float[nnz];
					if(A->values.s == 0)
					{
						sucflag = false;
						delete []A->rowind;
						delete []A->colptr;
						delete A;
					}
				}
			}
			else
			{
				A->rowind = 0;
				A->values.d = 0;
			}
			return A;
		}

		static taucs_ccs_matrix* ZQ_taucs_ccs_matrixTranspose(const taucs_ccs_matrix* A)
		{
			if(A == 0 || (A->flags != TAUCS_DOUBLE && A->flags != TAUCS_SINGLE))
				return 0;
			taucs_ccs_matrix* ret;
			ret = ZQ_taucs_ccs_create(A->n, A->m, A->colptr[A->n], A->flags);
			if (! ret)
				return NULL;

			if(A->flags & TAUCS_DOUBLE)
			{
				// non-symmetric matrix -> need to build data structure.
				// we'll go over the columns and build the rows
				std::vector< std::vector<int> >       rows(A->m);
				std::vector< std::vector<double> > values_mm(A->m);
				for (int c = 0; c < A->n; ++c) 
				{
					for (int rowi = A->colptr[c]; rowi < A->colptr[c+1]; ++rowi) 
					{
						rows[A->rowind[rowi]].push_back(c);
						values_mm[A->rowind[rowi]].push_back(A->values.d[rowi]);
					}
				}

				// copying the rows as columns in ret
				int cind = 0;
				for (int r = 0; r < A->m; ++r) 
				{
					ret->colptr[r] = cind;
					for (int j = 0; j < (int)rows[r].size(); ++j) 
					{
						ret->rowind[cind] = rows[r][j];
						ret->values.d[cind] = values_mm[r][j];
						cind++;
					}
				}
				ret->colptr[A->m] = cind;
			}
			else 
			{
				// non-symmetric matrix -> need to build data structure.
				// we'll go over the columns and build the rows
				std::vector< std::vector<int> >       rows(A->m);
				std::vector< std::vector<float> > values_mm(A->m);
				for (int c = 0; c < A->n; ++c) 
				{
					for (int rowi = A->colptr[c]; rowi < A->colptr[c+1]; ++rowi) 
					{
						rows[A->rowind[rowi]].push_back(c);
						values_mm[A->rowind[rowi]].push_back(A->values.s[rowi]);
					}
				}

				// copying the rows as columns in ret
				int cind = 0;
				for (int r = 0; r < A->m; ++r) 
				{
					ret->colptr[r] = cind;
					for (int j = 0; j < (int)rows[r].size(); ++j) 
					{
						ret->rowind[cind] = rows[r][j];
						ret->values.s[cind] = values_mm[r][j];
						cind++;
					}
				}
				ret->colptr[A->m] = cind;
			}

			return ret;
		}

		//when you find this function cost too much memory, you must find a better way to implement C = A * B;
		static taucs_ccs_matrix* ZQ_taucs_ccs_mul2NonSymmetricMatrices(const taucs_ccs_matrix *matA, const taucs_ccs_matrix *matB)
		{
			if(matA == 0 || matB == 0)
				return 0;
			if(matA->flags != matB->flags)
				return 0;
			// Compatibility of dimensions        
			if (matA->m != matB->n)
				return 0;
			if(matA->flags != TAUCS_DOUBLE && matA->flags != TAUCS_SINGLE)
				return 0;
			if(matA->flags == TAUCS_DOUBLE)
			{

				// (m x n)*(n x k) = (m x k)
				int m=matA->m;
				int n=matA->n;
				int k=matB->n;

				double biv, valA;
				int rowInd, rowA;
				std::vector<std::map<int, double> > rowsC(k);
				for (int i=0; i<k; ++i) {
					// creating column i of C
					std::map<int, double> & mapRow2Val = rowsC[i];
					// travel on bi
					int aaa,bbb; aaa=matB->colptr[i];bbb=matB->colptr[i+1];
					/*if( */
					for (int rowptrBi = matB->colptr[i];rowptrBi < matB->colptr[i+1];++rowptrBi) {
						rowInd = matB->rowind[rowptrBi];
						biv = matB->values.d[rowptrBi];
						// make biv*a_{rowInd} and insert into mapRow2Val
						for (int rowptrA=matA->colptr[rowInd];rowptrA<matA->colptr[rowInd+1];++rowptrA) {
							rowA=matA->rowind[rowptrA];
							valA=matA->values.d[rowptrA];
							// insert valA*biv into map
							std::map<int, double>::iterator it = mapRow2Val.find(rowA);
							if (it == mapRow2Val.end()) {
								// first time
								mapRow2Val[rowA] = valA*biv;
							}
							else {
								it->second = it->second + valA*biv;
							}
						}
					}
					// now column i is created
				}

				return ZQ_taucs_ccs_CreateMatrixFromColumns(rowsC,m,TAUCS_DOUBLE);
			}
			else
			{
				// (m x n)*(n x k) = (m x k)
				int m=matA->m;
				int n=matA->n;
				int k=matB->n;

				float biv, valA;
				int rowInd, rowA;
				std::vector<std::map<int, float> > rowsC(k);
				for (int i=0; i<k; ++i) {
					// creating column i of C
					std::map<int, float> & mapRow2Val = rowsC[i];
					// travel on bi
					int aaa,bbb; aaa=matB->colptr[i];bbb=matB->colptr[i+1];
					/*if( */
					for (int rowptrBi = matB->colptr[i];rowptrBi < matB->colptr[i+1];++rowptrBi) {
						rowInd = matB->rowind[rowptrBi];
						biv = matB->values.s[rowptrBi];
						// make biv*a_{rowInd} and insert into mapRow2Val
						for (int rowptrA=matA->colptr[rowInd];rowptrA<matA->colptr[rowInd+1];++rowptrA) {
							rowA=matA->rowind[rowptrA];
							valA=matA->values.s[rowptrA];
							// insert valA*biv into map
							std::map<int, float>::iterator it = mapRow2Val.find(rowA);
							if (it == mapRow2Val.end()) {
								// first time
								mapRow2Val[rowA] = valA*biv;
							}
							else {
								it->second = it->second + valA*biv;
							}
						}
					}
					// now column i is created
				}

				return ZQ_taucs_ccs_CreateMatrixFromColumns(rowsC,m,TAUCS_SINGLE);
			}
		}

		static taucs_ccs_matrix* ZQ_taucs_ccs_add2NonSymmetricMatrices(const taucs_ccs_matrix *matA, const taucs_ccs_matrix *matB)
		{
			if(matA == 0 || matB == 0)
				return 0;
			if(matA->flags != matB->flags)
				return 0;
			// Compatibility of dimensions        
			if (matA->m != matB->m || matA->n != matB->n)
				return 0;
			if(matA->flags != TAUCS_DOUBLE && matA->flags != TAUCS_SINGLE)
				return 0;
			if(matA->flags == TAUCS_DOUBLE)
			{
				int m=matA->m;
				int n=matA->n;

				std::vector<std::map<int, double> > rowsC(n);
				for (int i = 0; i<n; ++i) 
				{
					std::map<int, double> & mapRow2Val = rowsC[i];

					for (int rowptrAi = matA->colptr[i];rowptrAi < matA->colptr[i+1];++rowptrAi) 
					{
						int rowInd = matA->rowind[rowptrAi];
						double val = matA->values.d[rowptrAi];

						//std::map<int, double>::iterator it = mapRow2Val.find(rowInd);
						//if (it == mapRow2Val.end()) 
						//{
						mapRow2Val[rowInd] = val;
						//}
						//else 
						//{
						//	it->second = it->second + val;
						//}
					}

					for (int rowptrBi = matB->colptr[i];rowptrBi < matB->colptr[i+1];++rowptrBi) 
					{
						int rowInd = matB->rowind[rowptrBi];
						double val = matB->values.d[rowptrBi];

						std::map<int, double>::iterator it = mapRow2Val.find(rowInd);
						if (it == mapRow2Val.end()) 
						{
							mapRow2Val[rowInd] = val;
						}
						else 
						{
							it->second = it->second + val;
						}
					}

					// now column i is created
				}

				return ZQ_taucs_ccs_CreateMatrixFromColumns(rowsC,m,TAUCS_DOUBLE);
			}
			else
			{
				int m=matA->m;
				int n=matA->n;

				std::vector<std::map<int, double> > rowsC(n);
				for (int i = 0; i<n; ++i) 
				{
					std::map<int, double> & mapRow2Val = rowsC[i];

					for (int rowptrAi = matA->colptr[i];rowptrAi < matA->colptr[i+1];++rowptrAi) 
					{
						int rowInd = matA->rowind[rowptrAi];
						float val = matA->values.s[rowptrAi];

						//std::map<int, double>::iterator it = mapRow2Val.find(rowInd);
						//if (it == mapRow2Val.end()) 
						//{
						mapRow2Val[rowInd] = val;
						//}
						//else 
						//{
						//	it->second = it->second + val;
						//}
					}

					for (int rowptrBi = matB->colptr[i];rowptrBi < matB->colptr[i+1];++rowptrBi) 
					{
						int rowInd = matB->rowind[rowptrBi];
						float val = matB->values.s[rowptrBi];

						std::map<int, double>::iterator it = mapRow2Val.find(rowInd);
						if (it == mapRow2Val.end()) 
						{
							mapRow2Val[rowInd] = val;
						}
						else 
						{
							it->second = it->second + val;
						}
					}

					// now column i is created
				}

				return ZQ_taucs_ccs_CreateMatrixFromColumns(rowsC,m,TAUCS_SINGLE);
			}
		}

		static taucs_ccs_matrix* ZQ_taucs_ccs_scaleMatrix(const taucs_ccs_matrix *A, float scale)
		{
			if(A == 0 || ((A->flags) & TAUCS_SYMMETRIC) || (((A->flags) & TAUCS_DOUBLE) == 0 && ((A->flags)&TAUCS_SINGLE) ==0))
				return 0;


			int m = A->m;
			int n = A->n;

			if(A->flags == TAUCS_DOUBLE)
			{
				std::vector<std::map<int, double> > rowsC(n);
				if(scale == 0)
				{
					return ZQ_taucs_ccs_CreateMatrixFromColumns(rowsC,m,TAUCS_DOUBLE);
				}
				else
				{
					int nnz = A->colptr[n];

					taucs_ccs_matrix *matC = ZQ_taucs_ccs_create(m,n,nnz,A->flags);
					if (matC == 0)
						return 0;

					memcpy(matC->colptr,A->colptr,sizeof(int)*(n+1));
					memcpy(matC->rowind,A->rowind,sizeof(int)*nnz);
					for(int i = 0;i < nnz;i++)
						matC->values.d[i] = A->values.d[i]*scale;


					return matC;
				}
			}
			else
			{
				std::vector<std::map<int, double> > rowsC(n);
				if(scale == 0)
				{
					return ZQ_taucs_ccs_CreateMatrixFromColumns(rowsC,m,TAUCS_SINGLE);
				}
				else
				{
					int nnz = A->colptr[n];

					taucs_ccs_matrix *matC = ZQ_taucs_ccs_create(m,n,nnz,A->flags);
					if (matC == 0)
						return 0;

					memcpy(matC->colptr,A->colptr,sizeof(int)*(n+1));
					memcpy(matC->rowind,A->rowind,sizeof(int)*nnz);
					for(int i = 0;i < nnz;i++)
						matC->values.s[i] = A->values.s[i]*scale;


					return matC;
				}
			}
		}

		template<class T>
		static void ZQ_taucs_ccs_matrix_time_vec(const taucs_ccs_matrix* A, const T* X, T* B)
		{
			if(A == 0 || ((A->flags) & TAUCS_SYMMETRIC) || (((A->flags) & TAUCS_DOUBLE) == 0 && ((A->flags)&TAUCS_SINGLE) ==0))
				return ;
			if(X == 0 || B == 0)
				return;


			int m = A->m;
			int n = A->n;
			T* src = (T*)X;
			T* dst = (T*)B;
			memset(dst,0,sizeof(T)*m);
			for(int i = 0;i < n;i++)
			{
				int start = A->colptr[i];
				int num = A->colptr[i+1] - start;
				for(int j = 0;j < num;j++)
				{
					int rowind = A->rowind[start+j];
					double val = ((T*)(A->values.d))[start+j];
					dst[rowind] += val * src[i];
				}
			}
			
		}

		template<class T>
		static void ZQ_taucs_ccs_vec_time_matrix(const T* X, const taucs_ccs_matrix* A, T* B)
		{
			if(A == 0 || ((A->flags) & TAUCS_SYMMETRIC) || (((A->flags) & TAUCS_DOUBLE) == 0 && ((A->flags)&TAUCS_SINGLE) ==0))
				return ;
			if(X == 0 || B == 0)
				return ;


			int m = A->m;
			int n = A->n;
			T* src = (T*)X;
			T* dst = (T*)B;
			memset(dst,0,sizeof(T)*n);
			for(int i = 0;i < n;i++)
			{
				int start = A->colptr[i];
				int num = A->colptr[i+1] - start;
				for(int j = 0;j < num;j++)
				{
					int rowind = A->rowind[start+j];
					double val = ((T*)(A->values.d))[start+j];
					dst[i] += val * src[rowind];
				}
			}
		}

		/*this function is very slow when matrix is very sparse*/
		static taucs_ccs_matrix* ZQ_taucs_ccs_GetAtA(const taucs_ccs_matrix* A)
		{
			if(A == 0 || (A->flags) & TAUCS_SYMMETRIC || (((A->flags) & TAUCS_DOUBLE) == 0 && ((A->flags) & TAUCS_SINGLE) == 0))
				return 0;

			/*datatype TAUCS_DOUBLE*/
			if((A->flags) & TAUCS_DOUBLE)
			{
				int row = A->m;
				int col = A->n;
				double** values = (double**)malloc(sizeof(double*)*col);
				int** indexs = (int**)malloc(sizeof(double*)*col);
				int* colptr = new int[col+1];
				double* tmpvalue = new double[col];
				int* tmprowind = new int[col];
				int tmpcount = 0;

				colptr[0] = 0;
				for(int i = 0;i < col;i++)
				{
					tmpcount = 0;
					for(int j = 0;j < col;j++)
					{
						int start1 = A->colptr[j];
						int start2 = A->colptr[i];
						int num1 = A->colptr[j+1] - A->colptr[j];
						int num2 = A->colptr[i+1] - A->colptr[i];
						double val = ZQ_MathBase::DotProductSparse(row,num1,(A->rowind)+start1,(A->values.d)+start1,num2,
							(A->rowind)+start2,(A->values.d)+start2);
						if(val != 0)
						{
							tmpvalue[tmpcount] = val;
							tmprowind[tmpcount] = j;
							tmpcount ++;
						}
					}
					if(tmpcount == 0)
					{
						values[i] = 0;
						indexs[i] = 0;
						colptr[i+1] = colptr[i];
					}
					else
					{
						values[i] = new double[tmpcount];
						indexs[i] = new int[tmpcount];
						colptr[i+1] = colptr[i] + tmpcount;
						memcpy(values[i],tmpvalue,tmpcount*sizeof(double));
						memcpy(indexs[i],tmprowind,tmpcount*sizeof(int));
					}
				}
				taucs_ccs_matrix* AtA = ZQ_taucs_ccs_create(col,col,colptr[col],TAUCS_DOUBLE);
				AtA->colptr = colptr;
				for(int i = 0;i < col;i++)
				{
					int count  = colptr[i+1] - colptr[i];
					if(count > 0)
					{
						memcpy((AtA->values.d)+colptr[i],values[i],count*sizeof(double));
						memcpy((AtA->rowind)+colptr[i],indexs[i],count*sizeof(int));
					}
				}
				for(int i = 0;i < col;i++)
				{
					if(values[i])
						delete []values[i];
					if(indexs[i])
						delete []indexs[i];
				}
				delete []values;
				delete []indexs;
				delete []tmpvalue;
				delete []tmprowind;
				return AtA;
			}
			/*datatype TAUCS_SINGLE*/
			else
			{
				int row = A->m;
				int col = A->n;
				float** values = (float**)malloc(sizeof(float*)*col);
				int** indexs = (int**)malloc(sizeof(float*)*col);
				int* colptr = new int[col+1];
				float* tmpvalue = new float[col];
				int* tmprowind = new int[col];
				int tmpcount = 0;

				colptr[0] = 0;
				for(int i = 0;i < col;i++)
				{
					tmpcount = 0;
					for(int j = 0;j < col;j++)
					{
						int start1 = A->colptr[j];
						int start2 = A->colptr[i];
						int num1 = A->colptr[j+1] - A->colptr[j];
						int num2 = A->colptr[i+1] - A->colptr[i];
						float val = ZQ_MathBase::DotProductSparse(row,num1,(A->rowind)+start1,(A->values.s)+start1,num2,
							(A->rowind)+start2,(A->values.s)+start2);
						if(val != 0)
						{
							tmpvalue[tmpcount] = val;
							tmprowind[tmpcount] = j;
							tmpcount ++;
						}
					}
					if(tmpcount == 0)
					{
						values[i] = 0;
						indexs[i] = 0;
						colptr[i+1] = colptr[i];
					}
					else
					{
						values[i] = new float[tmpcount];
						indexs[i] = new int[tmpcount];
						colptr[i+1] = colptr[i] + tmpcount;
						memcpy(values[i],tmpvalue,tmpcount*sizeof(float));
						memcpy(indexs[i],tmprowind,tmpcount*sizeof(int));
					}
				}
				taucs_ccs_matrix* AtA = ZQ_taucs_ccs_create(col,col,colptr[col],TAUCS_SINGLE);
				AtA->colptr = colptr;
				for(int i = 0;i < col;i++)
				{
					int count  = colptr[i+1] - colptr[i];
					if(count > 0)
					{
						memcpy((AtA->values.s)+colptr[i],values[i],count*sizeof(float));
						memcpy((AtA->rowind)+colptr[i],indexs[i],count*sizeof(int));
					}
				}
				for(int i = 0;i < col;i++)
				{
					if(values[i])
						delete []values[i];
					if(indexs[i])
						delete []indexs[i];
				}
				delete []values;
				delete []indexs;
				delete []tmpvalue;
				delete []tmprowind;
				return AtA;
			}
		}

		template<class T>
		static bool ZQ_taucs_ccs_GetAtADiag(const taucs_ccs_matrix* A, T* diag)
		{
			if(A == 0 || (A->flags) & TAUCS_SYMMETRIC || (((A->flags) & TAUCS_DOUBLE) == 0 && ((A->flags) & TAUCS_SINGLE) == 0))
				return false;

			if(strcmp(typeid(T).name(),"float") != 0 && strcmp(typeid(T).name(),"double") != 0)
				return false;
			
			int col = A->n;
			for(int i = 0;i < col;i++)
			{

				int start = A->colptr[i];
				int end = A->colptr[i+1];
				double dot_sum = 0;
				for(int j = start;j < end;j++)
				{
					double val = ((T*)(A->values.d))[j];
					dot_sum += val*val;
				}
				diag[i] = dot_sum;
			}
			return true;
		}

		/*R is output ,and upperbandw is 0*/
		template<class T>
		static void ZQ_hprecon(const taucs_ccs_matrix* H,const T* DM, const T* DG, T* R)
		{
			if(H == 0 || H->m != H->n)
				return ;
			int n = H->n;
			T* DMptr = (T*)DM;
			T* DGptr = (T*)DG;
			T* tmp = new T[n];

			/*H = DM*H*DM + DG;
			epsi = .0001*ones(n,1);
			dnrms = sqrt(sum(H.*H))';
			d = max(sqrt(dnrms),epsi);
			R = sparse(1:n,1:n,full(d));*/

			for(int j = 0;j < n;j++)
			{
				int start = H->colptr[j];
				int num = H->colptr[j+1] - start;
				tmp[j] = 0;
				for(int kk = 0;kk < num;kk++)
				{
					int i = H->rowind[start+kk];
					double val = ((T*)(H->values.d))[start+kk];
					val *= DMptr[i] * DMptr[j];
					if(i == j)
						val += DGptr[i];
					tmp[j] += val*val;
				}
			}
			double epsi = 0.0001;
			for(int i = 0;i < n;i++)
			{
				tmp[i] = sqrt(tmp[i]);
				tmp[i] = sqrt(tmp[i]);
				tmp[i] = __max(tmp[i],epsi);
			}
			memcpy(R,tmp,sizeof(T)*n);
			delete []tmp;
			tmp = 0;

			
		}

	};
}

#endif