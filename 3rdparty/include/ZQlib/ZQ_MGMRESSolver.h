#ifndef _ZQ_MGMRES_SOLVER_H_
#define _ZQ_MGMRES_SOLVER_H_
#pragma once

#include "ZQ_taucs.h"
#include <string.h>
#include <malloc.h>
#include <stdio.h>

namespace ZQ
{
	class ZQ_MGMRESSolver
	{
	public:
		static bool MGMResSolve(const taucs_ccs_matrix* A, const void* b, void* x, int nOuterIter, int nInnerIter, double tol, bool display)
		{
			int m = A->m;
			int n = A->n;
			int nnz = A->colptr[n];
			if (m != n)
				return false;

			if (A->flags == TAUCS_DOUBLE)
				return pmgmres_ilu_cr<double>(n, nnz, A->colptr, A->rowind, A->values.d, (double*)x, (const double*)b, nOuterIter, nInnerIter, tol, tol, display);
			else if (A->flags == TAUCS_SINGLE)
				return pmgmres_ilu_cr<float>(n, nnz, A->colptr, A->rowind, A->values.s, (float*)x, (const float*)b, nOuterIter, nInnerIter, tol, tol, display);

			return true;
		}

	public:

		/*****  Changed from mgmres *****************************************************/
		/*  mgmres.hpp
		/*  Thanks to Philip Sakievich for correcting mismatches between this HPP
		/*  file and the corresponding CPP file, 23 March 2016!
		/********************************************************************************/
		
		template<class T>
		static void atx_cr(int n, int nz_num, const int* ia, const int* ja, const T* a, const T* x[], T* w)
		{
			//****************************************************************************80
			//
			//  Purpose:
			//
			//    ATX_CR computes A'*x for a matrix stored in sparse compressed row form.
			//
			//  Discussion:
			//
			//    The Sparse Compressed Row storage format is used.
			//
			//    The matrix A is assumed to be sparse.  To save on storage, only
			//    the nonzero entries of A are stored.  The vector JA stores the
			//    column index of the nonzero value.  The nonzero values are sorted
			//    by row, and the compressed row vector IA then has the property that
			//    the entries in A and JA that correspond to row I occur in indices
			//    IA[I] through IA[I+1]-1.
			//
			//    For this version of MGMRES, the row and column indices are assumed
			//    to use the C/C++ convention, in which indexing begins at 0.
			//
			//    If your index vectors IA and JA are set up so that indexing is based 
			//    at 1, then each use of those vectors should be shifted down by 1.
			//
			//  Licensing:
			//
			//    This code is distributed under the GNU LGPL license. 
			//
			//  Modified:
			//
			//    18 July 2007
			//
			//  Author:
			//
			//    Original C version by Lili Ju.
			//    C++ version by John Burkardt.
			//
			//  Reference:
			//
			//    Richard Barrett, Michael Berry, Tony Chan, James Demmel,
			//    June Donato, Jack Dongarra, Victor Eijkhout, Roidan Pozo,
			//    Charles Romine, Henk van der Vorst,
			//    Templates for the Solution of Linear Systems:
			//    Building Blocks for Iterative Methods,
			//    SIAM, 1994,
			//    ISBN: 0898714710,
			//    LC: QA297.8.T45.
			//
			//    Tim Kelley,
			//    Iterative Methods for Linear and Nonlinear Equations,
			//    SIAM, 2004,
			//    ISBN: 0898713528,
			//    LC: QA297.8.K45.
			//
			//    Yousef Saad,
			//    Iterative Methods for Sparse Linear Systems,
			//    Second Edition,
			//    SIAM, 2003,
			//    ISBN: 0898715342,
			//    LC: QA188.S17.
			//
			//  Parameters:
			//
			//    Input, int N, the order of the system.
			//
			//    Input, int NZ_NUM, the number of nonzeros.
			//
			//    Input, int IA[N+1], JA[NZ_NUM], the row and column indices
			//    of the matrix values.  The row vector has been compressed.
			//
			//    Input, double A[NZ_NUM], the matrix values.
			//
			//    Input, double X[N], the vector to be multiplied by A'.
			//
			//    Output, double W[N], the value of A'*X.
			//

			memset(w, 0, sizeof(T)*n);

			for (int i = 0; i < n; i++)
			{
				int k1 = ia[i];
				int k2 = ia[i + 1];
				for (int k = k1; k < k2; k++)
				{
					//A(i,ja(k)) ==> A'(ja(k),i)
					w[ja[k]] += a[k] * x[i];
				}
			}
			return;
		}

		template<class T>
		static void atx_st(int n, int nz_num, const int* ia, const int* ja, const T* a, const T* x, T* w)
		{
			//****************************************************************************80
			//
			//  Purpose:
			//
			//    ATX_ST computes A'*x for a matrix stored in sparse triplet form.
			//
			//  Discussion:
			//
			//    The matrix A is assumed to be sparse.  To save on storage, only
			//    the nonzero entries of A are stored.  For instance, the K-th nonzero
			//    entry in the matrix is stored by:
			//
			//      A(K) = value of entry,
			//      IA(K) = row of entry,
			//      JA(K) = column of entry.
			//
			//  Licensing:
			//
			//    This code is distributed under the GNU LGPL license. 
			//
			//  Modified:
			//
			//    18 July 2007
			//
			//  Author:
			//
			//    Original C version by Lili Ju.
			//    C++ version by John Burkardt.
			//
			//  Reference:
			//
			//    Richard Barrett, Michael Berry, Tony Chan, James Demmel,
			//    June Donato, Jack Dongarra, Victor Eijkhout, Roidan Pozo,
			//    Charles Romine, Henk van der Vorst,
			//    Templates for the Solution of Linear Systems:
			//    Building Blocks for Iterative Methods,
			//    SIAM, 1994,
			//    ISBN: 0898714710,
			//    LC: QA297.8.T45.
			//
			//    Tim Kelley,
			//    Iterative Methods for Linear and Nonlinear Equations,
			//    SIAM, 2004,
			//    ISBN: 0898713528,
			//    LC: QA297.8.K45.
			//
			//    Yousef Saad,
			//    Iterative Methods for Sparse Linear Systems,
			//    Second Edition,
			//    SIAM, 2003,
			//    ISBN: 0898715342,
			//    LC: QA188.S17.
			//
			//  Parameters:
			//
			//    Input, int N, the order of the system.
			//
			//    Input, int NZ_NUM, the number of nonzeros.
			//
			//    Input, int IA[NZ_NUM], JA[NZ_NUM], the row and column indices
			//    of the matrix values.
			//
			//    Input, double A[NZ_NUM], the matrix values.
			//
			//    Input, double X[N], the vector to be multiplied by A'.
			//
			//    Output, double W[N], the value of A'*X.
			//

			memset(w, 0, sizeof(T)*n);

			for (int k = 0; k < nz_num; k++)
			{
				int i = ia[k];
				int j = ja[k];
				w[j] += a[k] * x[i];
			}
			return;
		}

		template<class T>
		static void ax_cr(int n, int nz_num, const int* ia, const int* ja, const T* a, const T* x, T* w)
		{
			//****************************************************************************80
			//
			//  Purpose:
			//
			//    AX_CR computes A*x for a matrix stored in sparse compressed row form.
			//
			//  Discussion:
			//
			//    The Sparse Compressed Row storage format is used.
			//
			//    The matrix A is assumed to be sparse.  To save on storage, only
			//    the nonzero entries of A are stored.  The vector JA stores the
			//    column index of the nonzero value.  The nonzero values are sorted
			//    by row, and the compressed row vector IA then has the property that
			//    the entries in A and JA that correspond to row I occur in indices
			//    IA[I] through IA[I+1]-1.
			//
			//    For this version of MGMRES, the row and column indices are assumed
			//    to use the C/C++ convention, in which indexing begins at 0.
			//
			//    If your index vectors IA and JA are set up so that indexing is based 
			//    at 1, then each use of those vectors should be shifted down by 1.
			//
			//  Licensing:
			//
			//    This code is distributed under the GNU LGPL license. 
			//
			//  Modified:
			//
			//    18 July 2007
			//
			//  Author:
			//
			//    Original C version by Lili Ju.
			//    C++ version by John Burkardt.
			//
			//  Reference:
			//
			//    Richard Barrett, Michael Berry, Tony Chan, James Demmel,
			//    June Donato, Jack Dongarra, Victor Eijkhout, Roidan Pozo,
			//    Charles Romine, Henk van der Vorst,
			//    Templates for the Solution of Linear Systems:
			//    Building Blocks for Iterative Methods,
			//    SIAM, 1994,
			//    ISBN: 0898714710,
			//    LC: QA297.8.T45.
			//
			//    Tim Kelley,
			//    Iterative Methods for Linear and Nonlinear Equations,
			//    SIAM, 2004,
			//    ISBN: 0898713528,
			//    LC: QA297.8.K45.
			//
			//    Yousef Saad,
			//    Iterative Methods for Sparse Linear Systems,
			//    Second Edition,
			//    SIAM, 2003,
			//    ISBN: 0898715342,
			//    LC: QA188.S17.
			//
			//  Parameters:
			//
			//    Input, int N, the order of the system.
			//
			//    Input, int NZ_NUM, the number of nonzeros.
			//
			//    Input, int IA[N+1], JA[NZ_NUM], the row and column indices
			//    of the matrix values.  The row vector has been compressed.
			//
			//    Input, double A[NZ_NUM], the matrix values.
			//
			//    Input, double X[N], the vector to be multiplied by A.
			//
			//    Output, double W[N], the value of A*X.
			//

			memset(w, 0, sizeof(T)*n);
			for (int i = 0; i < n; i++)
			{
				int k1 = ia[i];
				int k2 = ia[i + 1];
				for (int k = k1; k < k2; k++)
				{
					w[i] += a[k] * x[ja[k]];
				}
			}
			return;
		}

		template<class T>
		static void ax_st(int n, int nz_num, const int* ia, const int* ja, const T* a, const T* x, T* w)
		{
			//****************************************************************************80
			//
			//  Purpose:
			//
			//    AX_ST computes A*x for a matrix stored in sparse triplet form.
			//
			//  Discussion:
			//
			//    The matrix A is assumed to be sparse.  To save on storage, only
			//    the nonzero entries of A are stored.  For instance, the K-th nonzero
			//    entry in the matrix is stored by:
			//
			//      A(K) = value of entry,
			//      IA(K) = row of entry,
			//      JA(K) = column of entry.
			//
			//  Licensing:
			//
			//    This code is distributed under the GNU LGPL license. 
			//
			//  Modified:
			//
			//    18 July 2007
			//
			//  Author:
			//
			//    Original C version by Lili Ju.
			//    C++ version by John Burkardt.
			//
			//  Reference:
			//
			//    Richard Barrett, Michael Berry, Tony Chan, James Demmel,
			//    June Donato, Jack Dongarra, Victor Eijkhout, Roidan Pozo,
			//    Charles Romine, Henk van der Vorst,
			//    Templates for the Solution of Linear Systems:
			//    Building Blocks for Iterative Methods,
			//    SIAM, 1994,
			//    ISBN: 0898714710,
			//    LC: QA297.8.T45.
			//
			//    Tim Kelley,
			//    Iterative Methods for Linear and Nonlinear Equations,
			//    SIAM, 2004,
			//    ISBN: 0898713528,
			//    LC: QA297.8.K45.
			//
			//    Yousef Saad,
			//    Iterative Methods for Sparse Linear Systems,
			//    Second Edition,
			//    SIAM, 2003,
			//    ISBN: 0898715342,
			//    LC: QA188.S17.
			//
			//  Parameters:
			//
			//    Input, int N, the order of the system.
			//
			//    Input, int NZ_NUM, the number of nonzeros.
			//
			//    Input, int IA[NZ_NUM], JA[NZ_NUM], the row and column indices
			//    of the matrix values.
			//
			//    Input, double A[NZ_NUM], the matrix values.
			//
			//    Input, double X[N], the vector to be multiplied by A.
			//
			//    Output, double W[N], the value of A*X.
			//

			memset(w, 0, sizeof(T)*n);

			for (int k = 0; k < nz_num; k++)
			{
				int i = ia[k];
				int j = ja[k];
				w[i] += a[k] * x[j];
			}
			return;
		}

		static void diagonal_pointer_cr(int n, int nz_num, const int* ia, const int* ja, int* ua)
		{
			//****************************************************************************80
			//
			//  Purpose:
			//
			//    DIAGONAL_POINTER_CR finds diagonal entries in a sparse compressed row matrix.
			//
			//  Discussion:
			//
			//    The matrix A is assumed to be stored in compressed row format.  Only
			//    the nonzero entries of A are stored.  The vector JA stores the
			//    column index of the nonzero value.  The nonzero values are sorted
			//    by row, and the compressed row vector IA then has the property that
			//    the entries in A and JA that correspond to row I occur in indices
			//    IA[I] through IA[I+1]-1.
			//
			//    The array UA can be used to locate the diagonal elements of the matrix.
			//
			//    It is assumed that every row of the matrix includes a diagonal element,
			//    and that the elements of each row have been ascending sorted.
			//
			//  Licensing:
			//
			//    This code is distributed under the GNU LGPL license. 
			//
			//  Modified:
			//
			//    18 July 2007
			//
			//  Author:
			//
			//    Original C version by Lili Ju.
			//    C++ version by John Burkardt.
			//
			//  Parameters:
			//
			//    Input, int N, the order of the system.
			//
			//    Input, int NZ_NUM, the number of nonzeros.
			//
			//    Input, int IA[N+1], JA[NZ_NUM], the row and column indices
			//    of the matrix values.  The row vector has been compressed.  On output,
			//    the order of the entries of JA may have changed because of the sorting.
			//
			//    Output, int UA[N], the index of the diagonal element of each row.
			//

			for (int i = 0; i < n; i++)
			{
				ua[i] = -1;
				int j1 = ia[i];
				int j2 = ia[i + 1];

				for (int j = j1; j < j2; j++)
				{
					if (ja[j] == i)
					{
						ua[i] = j;
					}
				}

			}
			return;
		}

		template<class T>
		static bool ilu_cr(int n, int nz_num, const int* ia, const int* ja, const T* a, int* ua, T* l)
		{
			//****************************************************************************80
			//
			//  Purpose:
			//
			//    ILU_CR computes the incomplete LU factorization of a matrix.
			//
			//  Discussion:
			//
			//    The matrix A is assumed to be stored in compressed row format.  Only
			//    the nonzero entries of A are stored.  The vector JA stores the
			//    column index of the nonzero value.  The nonzero values are sorted
			//    by row, and the compressed row vector IA then has the property that
			//    the entries in A and JA that correspond to row I occur in indices
			//    IA[I] through IA[I+1]-1.
			//
			//  Licensing:
			//
			//    This code is distributed under the GNU LGPL license. 
			//
			//  Modified:
			//
			//    25 July 2007
			//
			//  Author:
			//
			//    Original C version by Lili Ju.
			//    C++ version by John Burkardt.
			//
			//  Parameters:
			//
			//    Input, int N, the order of the system.
			//
			//    Input, int NZ_NUM, the number of nonzeros.
			//
			//    Input, int IA[N+1], JA[NZ_NUM], the row and column indices
			//    of the matrix values.  The row vector has been compressed.
			//
			//    Input, double A[NZ_NUM], the matrix values.
			//
			//    Input, int UA[N], the index of the diagonal element of each row.
			//
			//    Output, double L[NZ_NUM], the ILU factorization of A.
			//

			
			T tl;
			int jrow;
			int* iw = (int*)malloc(sizeof(int)*n);
			if (iw == 0)
				return false;
			
			//  Copy A.
			memcpy(l, a, sizeof(T)*nz_num);

			
			for (int i = 0; i < n; i++)
			{
				//  IW points to the nonzero entries in row I.
				for (int j = 0; j < n; j++)
				{
					iw[j] = -1;
				}

				for (int k = ia[i]; k < ia[i + 1]; k++)
				{
					iw[ja[k]] = k;
				}

				int j = ia[i];
				do
				{
					jrow = ja[j];
					if (i <= jrow)
					{
						break;
					}
					tl = l[j] * l[ua[jrow]];
					l[j] = tl;
					for (int jj = ua[jrow] + 1; jj < ia[jrow + 1]; jj++)
					{
						int jw = iw[ja[jj]];
						if (jw != -1)
						{
							l[jw] = l[jw] - tl * l[jj];
						}
					}
					j++;
				} while (j < ia[i + 1]);

				ua[i] = j;

				if (jrow != i)
				{
					
					printf("\n");
					printf("ILU_CR - Fatal error!\n");
					printf("  JROW != I\n");
					printf("  JROW = %d\n",jrow);
					printf("  I    = %d\n",i);
					free(iw);
					return false;
				}

				if (l[j] == 0.0)
				{
					printf("\n");
					printf("ILU_CR - Fatal error!\n");
					printf("  Zero pivot on step I = %d\n",i);
					printf("  L[%d] = 0.0\n",j);
					free(iw);
					return false;
				}

				l[j] = 1.0 / l[j];
			}

			for (int k = 0; k < n; k++)
			{
				l[ua[k]] = 1.0 / l[ua[k]];
			}
			
			free(iw);
			return true;
		}
		
		template<class T>
		static bool lus_cr(int n, int nz_num, const int* ia, const int* ja, const T* l, const int* ua, const T* r, T* z)
		{
			//****************************************************************************80
			//
			//  Purpose:
			//
			//    LUS_CR applies the incomplete LU preconditioner.
			//
			//  Discussion:
			//
			//    The linear system M * Z = R is solved for Z.  M is the incomplete
			//    LU preconditioner matrix, and R is a vector supplied by the user.
			//    So essentially, we're solving L * U * Z = R.
			//
			//    The matrix A is assumed to be stored in compressed row format.  Only
			//    the nonzero entries of A are stored.  The vector JA stores the
			//    column index of the nonzero value.  The nonzero values are sorted
			//    by row, and the compressed row vector IA then has the property that
			//    the entries in A and JA that correspond to row I occur in indices
			//    IA[I] through IA[I+1]-1.
			//
			//  Licensing:
			//
			//    This code is distributed under the GNU LGPL license. 
			//
			//  Modified:
			//
			//    18 July 2007
			//
			//  Author:
			//
			//    Original C version by Lili Ju.
			//    C++ version by John Burkardt.
			//
			//  Parameters:
			//
			//    Input, int N, the order of the system.
			//
			//    Input, int NZ_NUM, the number of nonzeros.
			//
			//    Input, int IA[N+1], JA[NZ_NUM], the row and column indices
			//    of the matrix values.  The row vector has been compressed.
			//
			//    Input, double L[NZ_NUM], the matrix values.
			//
			//    Input, int UA[N], the index of the diagonal element of each row.
			//
			//    Input, double R[N], the right hand side.
			//
			//    Output, double Z[N], the solution of the system M * Z = R.
			//

			
			T* w = (T*)malloc(sizeof(T)*n);
			if (w == 0)
				return false;
			
			//  Copy R in.
			memcpy(w, r, sizeof(T)*n);
			
			//
			//  Solve L * w = w where L is unit lower triangular.
			//
			for (int i = 1; i < n; i++)
			{
				for (int j = ia[i]; j < ua[i]; j++)
				{
					w[i] -= l[j] * w[ja[j]];
				}
			}
			//
			//  Solve U * w = w, where U is upper triangular.
			//
			for (int i = n - 1; 0 <= i; i--)
			{
				for (int j = ua[i] + 1; j < ia[i + 1]; j++)
				{
					w[i] -= l[j] * w[ja[j]];
				}
				w[i] = w[i] / l[ua[i]];
			}
			//
			//  Copy Z out.
			//
			memcpy(z, w, sizeof(T)*n);
			//
			//  Free memory.
			//
			free(w);
			return true;
		}

		template<class T>
		static bool mgmres_st(int n, int nz_num, const int* ia, const int* ja, const T* a, T* x, const T* rhs, int itr_max, int mr, double tol_abs, double tol_rel, bool display)
		{
			//
			//  Purpose:
			//
			//    MGMRES_ST applies restarted GMRES to a matrix in sparse triplet form.
			//
			//  Discussion:
			//
			//    The linear system A*X=B is solved iteratively.
			//
			//    The matrix A is assumed to be stored in sparse triplet form.  Only
			//    the nonzero entries of A are stored.  For instance, the K-th nonzero
			//    entry in the matrix is stored by:
			//
			//      A(K) = value of entry,
			//      IA(K) = row of entry,
			//      JA(K) = column of entry.
			//
			//    The "matrices" H and V are treated as one-dimensional vectors
			//    which store the matrix data in row major form.
			//
			//    This requires that references to H[I][J] be replaced by references
			//    to H[I+J*(MR+1)] and references to V[I][J] by V[I+J*N].
			//
			//  Licensing:
			//
			//    This code is distributed under the GNU LGPL license. 
			//
			//  Modified:
			//
			//    18 July 2007
			//
			//  Author:
			//
			//    Original C version by Lili Ju.
			//    C++ version by John Burkardt.
			//
			//  Reference:
			//
			//    Richard Barrett, Michael Berry, Tony Chan, James Demmel,
			//    June Donato, Jack Dongarra, Victor Eijkhout, Roidan Pozo,
			//    Charles Romine, Henk van der Vorst,
			//    Templates for the Solution of Linear Systems:
			//    Building Blocks for Iterative Methods,
			//    SIAM, 1994,
			//    ISBN: 0898714710,
			//    LC: QA297.8.T45.
			//
			//    Tim Kelley,
			//    Iterative Methods for Linear and Nonlinear Equations,
			//    SIAM, 2004,
			//    ISBN: 0898713528,
			//    LC: QA297.8.K45.
			//
			//    Yousef Saad,
			//    Iterative Methods for Sparse Linear Systems,
			//    Second Edition,
			//    SIAM, 2003,
			//    ISBN: 0898715342,
			//    LC: QA188.S17.
			//
			//  Parameters:
			//
			//    Input, int N, the order of the linear system.
			//
			//    Input, int NZ_NUM, the number of nonzero matrix values.
			//
			//    Input, int IA[NZ_NUM], JA[NZ_NUM], the row and column indices
			//    of the matrix values.
			//
			//    Input, double A[NZ_NUM], the matrix values.
			//
			//    Input/output, double X[N]; on input, an approximation to
			//    the solution.  On output, an improved approximation.
			//
			//    Input, double RHS[N], the right hand side of the linear system.
			//
			//    Input, int ITR_MAX, the maximum number of (outer) iterations to take.
			//
			//    Input, int MR, the maximum number of (inner) iterations to take.
			//    MR must be less than N.
			//
			//    Input, double TOL_ABS, an absolute tolerance applied to the
			//    current residual.
			//
			//    Input, double TOL_REL, a relative tolerance comparing the
			//    current residual to the initial residual.
			//

			double av;
			double delta = 1.0e-03;	
			double htmp;
			
			int k_copy;
			double mu;
			double rho;
			double rho_tol;

			if (n < mr)
			{
				if (display)
				{
					cerr << "\n";
					cerr << "MGMRES_ST - Fatal error!\n";
					cerr << "  N < MR.\n";
					cerr << "  N = " << n << "\n";
					cerr << "  MR = " << mr << "\n";

				}
				return false;
			}

			
			T* c = (T*)malloc(sizeof(T)*mr);
			T* g = (T*)malloc(sizeof(T)*(mr + 1));
			T* h = (T*)malloc(sizeof(T)*mr*(mr + 1));
			T* r = (T*)malloc(sizeof(T)*n);
			T* s = (T*)malloc(sizeof(T)*mr);
			T* v = (T*)malloc(sizeof(T)*n*(mr + 1));
			T* y = (T*)malloc(sizeof(T)*(mr + 1));
			if (c == 0 || g == 0 || h == 0 || r == 0 || s == 0 || v == 0 || y == 0)
			{
				if (c) free(c);
				if (g) free(g);
				if (h) free(h);
				if (r) free(r);
				if (s) free(s);
				if (v) free(v);
				if (y) free(y);
				return false;
			}

			int itr_used = 0;

			for (int itr = 1; itr <= itr_max; itr++)
			{
				ax_st(n, nz_num, ia, ja, a, x, r);

				for (int i = 0; i < n; i++)
				{
					r[i] = rhs[i] - r[i];
				}

				rho = sqrt(r8vec_dot(n, r, r));

				if (display)
				{
					cout << "  ITR = " << itr << "  Residual = " << rho << "\n";
				}

				if (itr == 1)
				{
					rho_tol = rho * tol_rel;
				}

				for (int i = 0; i < n; i++)
				{
					v[i + 0 * n] = r[i] / rho;
				}

				g[0] = rho;
				for (int i = 1; i <= mr; i++)
				{
					g[i] = 0.0;
				}

				for (int i = 0; i < mr + 1; i++)
				{
					for (int j = 0; j < mr; j++)
					{
						h[i + j*(mr + 1)] = 0.0;
					}
				}

				for (int k = 1; k <= mr; k++)
				{
					k_copy = k;

					ax_st(n, nz_num, ia, ja, a, v + (k - 1)*n, v + k*n);

					av = sqrt(r8vec_dot(n, v + k*n, v + k*n));

					for (int j = 1; j <= k; j++)
					{
						h[(j - 1) + (k - 1)*(mr + 1)] = r8vec_dot(n, v + k*n, v + (j - 1)*n);
						for (int i = 0; i < n; i++)
						{
							v[i + k*n] = v[i + k*n] - h[(j - 1) + (k - 1)*(mr + 1)] * v[i + (j - 1)*n];
						}
					}

					h[k + (k - 1)*(mr + 1)] = sqrt(r8vec_dot(n, v + k*n, v + k*n));

					if ((av + delta * h[k + (k - 1)*(mr + 1)]) == av)
					{
						for (int j = 1; j <= k; j++)
						{
							htmp = r8vec_dot(n, v + k*n, v + (j - 1)*n);
							h[(j - 1) + (k - 1)*(mr + 1)] = h[(j - 1) + (k - 1)*(mr + 1)] + htmp;
							for (int i = 0; i < n; i++)
							{
								v[i + k*n] = v[i + k*n] - htmp * v[i + (j - 1)*n];
							}
						}
						h[k + (k - 1)*(mr + 1)] = sqrt(r8vec_dot(n, v + k*n, v + k*n));
					}

					if (h[k + (k - 1)*(mr + 1)] != 0.0)
					{
						for (int i = 0; i < n; i++)
						{
							v[i + k*n] = v[i + k*n] / h[k + (k - 1)*(mr + 1)];
						}
					}

					if (1 < k)
					{
						for (int i = 1; i <= k + 1; i++)
						{
							y[i - 1] = h[(i - 1) + (k - 1)*(mr + 1)];
						}
						for (int j = 1; j <= k - 1; j++)
						{
							mult_givens(c[j - 1], s[j - 1], j - 1, y);
						}
						for (int i = 1; i <= k + 1; i++)
						{
							h[i - 1 + (k - 1)*(mr + 1)] = y[i - 1];
						}
					}
					mu = sqrt(pow(h[(k - 1) + (k - 1)*(mr + 1)], 2) + pow(h[k + (k - 1)*(mr + 1)], 2));
					c[k - 1] = h[(k - 1) + (k - 1)*(mr + 1)] / mu;
					s[k - 1] = -h[k + (k - 1)*(mr + 1)] / mu;
					h[(k - 1) + (k - 1)*(mr + 1)] = c[k - 1] * h[(k - 1) + (k - 1)*(mr + 1)]
						- s[k - 1] * h[k + (k - 1)*(mr + 1)];
					h[k + (k - 1)*(mr + 1)] = 0;
					mult_givens(c[k - 1], s[k - 1], k - 1, g);

					rho = fabs(g[k]);

					itr_used++;

					if (display)
					{
						cout << "  K =   " << k << "  Residual = " << rho << "\n";
					}

					if (rho <= rho_tol && rho <= tol_abs)
					{
						break;
					}
				}

				int k = k_copy - 1;
				y[k] = g[k] / h[k + k*(mr + 1)];

				for (int i = k; 1 <= i; i--)
				{
					y[i - 1] = g[i - 1];
					for (int j = i + 1; j <= k + 1; j++)
					{
						y[i - 1] = y[i - 1] - h[(i - 1) + (j - 1)*(mr + 1)] * y[j - 1];
					}
					y[i - 1] = y[i - 1] / h[(i - 1) + (i - 1)*(mr + 1)];
				}

				for (int i = 1; i <= n; i++)
				{
					for (int j = 1; j <= k + 1; j++)
					{
						x[i - 1] = x[i - 1] + v[(i - 1) + (j - 1)*n] * y[j - 1];
					}
				}

				if (rho <= rho_tol && rho <= tol_abs)
				{
					break;
				}
			}

			if (display)
			{
				cout << "\n";
				cout << "MGMRES_ST\n";
				cout << "  Number of iterations = " << itr_used << "\n";
				cout << "  Final residual = " << rho << "\n";
			}
			//
			//  Free memory.
			//
			free(c);
			free(g);
			free(h);
			free(r);
			free(s);
			free(v);
			free(y);

			return true;
		}

		template<class T>
		static void mult_givens(double c, double s, int k, T* g)
		{
			//
			//  Purpose:
			//
			//    MULT_GIVENS applies a Givens rotation to two successive entries of a vector.
			//
			//  Licensing:
			//
			//    This code is distributed under the GNU LGPL license. 
			//
			//  Modified:
			//
			//    08 August 2006
			//
			//  Author:
			//
			//    Original C version by Lili Ju.
			//    C++ version by John Burkardt.
			//
			//  Reference:
			//
			//    Richard Barrett, Michael Berry, Tony Chan, James Demmel,
			//    June Donato, Jack Dongarra, Victor Eijkhout, Roidan Pozo,
			//    Charles Romine, Henk van der Vorst,
			//    Templates for the Solution of Linear Systems:
			//    Building Blocks for Iterative Methods,
			//    SIAM, 1994,
			//    ISBN: 0898714710,
			//    LC: QA297.8.T45.
			//
			//    Tim Kelley,
			//    Iterative Methods for Linear and Nonlinear Equations,
			//    SIAM, 2004,
			//    ISBN: 0898713528,
			//    LC: QA297.8.K45.
			//
			//    Yousef Saad,
			//    Iterative Methods for Sparse Linear Systems,
			//    Second Edition,
			//    SIAM, 2003,
			//    ISBN: 0898715342,
			//    LC: QA188.S17.
			//
			//  Parameters:
			//
			//    Input, double C, S, the cosine and sine of a Givens
			//    rotation.
			//
			//    Input, int K, indicates the location of the first vector entry.
			//
			//    Input/output, double G[K+2], the vector to be modified.  On output,
			//    the Givens rotation has been applied to entries G(K) and G(K+1).
			//

			
			T g1 = c * g[k] - s * g[k + 1];
			T g2 = s * g[k] + c * g[k + 1];
			g[k] = g1;
			g[k + 1] = g2;
		}

		template<class T>
		static bool pmgmres_ilu_cr(int n, int nz_num, const int* ia, const int* ja, const T* a, T* x, const T* rhs, int itr_max, int mr, double tol_abs, double tol_rel, bool display)
		{
			//
			//  Purpose:
			//
			//    PMGMRES_ILU_CR applies the preconditioned restarted GMRES algorithm.
			//
			//  Discussion:
			//
			//    The matrix A is assumed to be stored in compressed row format.  Only
			//    the nonzero entries of A are stored.  The vector JA stores the
			//    column index of the nonzero value.  The nonzero values are sorted
			//    by row, and the compressed row vector IA then has the property that
			//    the entries in A and JA that correspond to row I occur in indices
			//    IA[I] through IA[I+1]-1.
			//
			//    This routine uses the incomplete LU decomposition for the
			//    preconditioning.  This preconditioner requires that the sparse
			//    matrix data structure supplies a storage position for each diagonal
			//    element of the matrix A, and that each diagonal element of the
			//    matrix A is not zero.
			//
			//    Thanks to Jesus Pueblas Sanchez-Guerra for supplying two
			//    corrections to the code on 31 May 2007.
			//
			//    This implementation of the code stores the doubly-dimensioned arrays
			//    H and V as vectors.  However, it follows the C convention of storing
			//    them by rows, rather than my own preference for storing them by
			//    columns.   I may come back and change this some time.
			//
			//  Licensing:
			//
			//    This code is distributed under the GNU LGPL license. 
			//
			//  Modified:
			//
			//    26 July 2007
			//
			//  Author:
			//
			//    Original C version by Lili Ju.
			//    C++ version by John Burkardt.
			//
			//  Reference:
			//
			//    Richard Barrett, Michael Berry, Tony Chan, James Demmel,
			//    June Donato, Jack Dongarra, Victor Eijkhout, Roidan Pozo,
			//    Charles Romine, Henk van der Vorst,
			//    Templates for the Solution of Linear Systems:
			//    Building Blocks for Iterative Methods,
			//    SIAM, 1994.
			//    ISBN: 0898714710,
			//    LC: QA297.8.T45.
			//
			//    Tim Kelley,
			//    Iterative Methods for Linear and Nonlinear Equations,
			//    SIAM, 2004,
			//    ISBN: 0898713528,
			//    LC: QA297.8.K45.
			//
			//    Yousef Saad,
			//    Iterative Methods for Sparse Linear Systems,
			//    Second Edition,
			//    SIAM, 2003,
			//    ISBN: 0898715342,
			//    LC: QA188.S17.
			//
			//  Parameters:
			//
			//    Input, int N, the order of the linear system.
			//
			//    Input, int NZ_NUM, the number of nonzero matrix values.
			//
			//    Input, int IA[N+1], JA[NZ_NUM], the row and column indices
			//    of the matrix values.  The row vector has been compressed.
			//
			//    Input, double A[NZ_NUM], the matrix values.
			//
			//    Input/output, double X[N]; on input, an approximation to
			//    the solution.  On output, an improved approximation.
			//
			//    Input, double RHS[N], the right hand side of the linear system.
			//
			//    Input, int ITR_MAX, the maximum number of (outer) iterations to take.
			//
			//    Input, int MR, the maximum number of (inner) iterations to take.
			//    MR must be less than N.
			//
			//    Input, double TOL_ABS, an absolute tolerance applied to the
			//    current residual.
			//
			//    Input, double TOL_REL, a relative tolerance comparing the
			//    current residual to the initial residual.
			//

			double av;
			
			double delta = 1.0e-03;
			double htmp;
			int i;
			int itr;
			int j;
			int k;
			int k_copy;
			double mu;
			double rho;
			double rho_tol;
			
			int itr_used = 0;

			T* c = (T*)malloc(sizeof(T)*(mr + 1));
			T* g = (T*)malloc(sizeof(T)*(mr + 1));
			T* h = (T*)malloc(sizeof(T)*mr*(mr + 1));
			T* l = (T*)malloc(sizeof(T)*(ia[n] + 1));
			T* r = (T*)malloc(sizeof(T)*n);
			T* s = (T*)malloc(sizeof(T)*(mr + 1));
			int* ua = (int*)malloc(sizeof(int)*n);
			T* v = (T*)malloc(sizeof(T)*n*(mr + 1));
			T* y = (T*)malloc(sizeof(T)*(mr + 1));			

			if (c == 0 || g == 0 || h == 0 || l == 0 || r == 0 || s == 0 || ua == 0 || v == 0 || y == 0)
			{
				if (c) free(c);
				if (g) free(g);
				if (h) free(h);
				if (l) free(l);
				if (r) free(r);
				if (s) free(s);
				if (ua) free(ua);
				if (v) free(v);
				if (y) free(y);
				return false;
			}

			//rearrange_cr(n, nz_num, ia, ja, a);

			diagonal_pointer_cr(n, nz_num, ia, ja, ua);

			ilu_cr(n, nz_num, ia, ja, a, ua, l);

			if (display)
			{
				printf("\n");
				printf("PMGMRES_ILU_CR\n");
				printf("  Number of unknowns = %d\n",n);
			}

			for (int itr = 0; itr < itr_max; itr++)
			{
				ax_cr(n, nz_num, ia, ja, a, x, r);

				for (int i = 0; i < n; i++)
				{
					r[i] = rhs[i] - r[i];
				}

				lus_cr(n, nz_num, ia, ja, l, ua, r, r);

				rho = sqrt(r8vec_dot(n, r, r));

				if (display)
				{
					printf("  ITR = %d  Residual =  %f\n", itr, rho);
				}

				if (itr == 0)
				{
					rho_tol = rho * tol_rel;
				}

				for (int i = 0; i < n; i++)
				{
					v[0 * n + i] = r[i] / rho;
				}

				g[0] = rho;
				for (int i = 1; i < mr + 1; i++)
				{
					g[i] = 0.0;
				}

				for (int i = 0; i < mr + 1; i++)
				{
					for (int j = 0; j < mr; j++)
					{
						h[i*(mr)+j] = 0.0;
					}
				}

				for (k = 0; k < mr; k++)
				{
					k_copy = k;

					ax_cr(n, nz_num, ia, ja, a, v + k*n, v + (k + 1)*n);

					lus_cr(n, nz_num, ia, ja, l, ua, v + (k + 1)*n, v + (k + 1)*n);

					av = sqrt(r8vec_dot(n, v + (k + 1)*n, v + (k + 1)*n));

					for (int j = 0; j <= k; j++)
					{
						h[j*mr + k] = r8vec_dot(n, v + (k + 1)*n, v + j*n);
						for (int i = 0; i < n; i++)
						{
							v[(k + 1)*n + i] = v[(k + 1)*n + i] - h[j*mr + k] * v[j*n + i];
						}
					}
					h[(k + 1)*mr + k] = sqrt(r8vec_dot(n, v + (k + 1)*n, v + (k + 1)*n));

					if ((av + delta * h[(k + 1)*mr + k]) == av)
					{
						for (int j = 0; j < k + 1; j++)
						{
							htmp = r8vec_dot(n, v + (k + 1)*n, v + j*n);
							h[j*mr + k] = h[j*mr + k] + htmp;
							for (int i = 0; i < n; i++)
							{
								v[(k + 1)*n + i] = v[(k + 1)*n + i] - htmp * v[j*n + i];
							}
						}
						h[(k + 1)*mr + k] = sqrt(r8vec_dot(n, v + (k + 1)*n, v + (k + 1)*n));
					}

					if (h[(k + 1)*mr + k] != 0.0)
					{
						for (int i = 0; i < n; i++)
						{
							v[(k + 1)*n + i] = v[(k + 1)*n + i] / h[(k + 1)*mr + k];
						}
					}

					if (0 < k)
					{
						for (int i = 0; i < k + 2; i++)
						{
							y[i] = h[i*mr + k];
						}
						for (int j = 0; j < k; j++)
						{
							mult_givens(c[j], s[j], j, y);
						}
						for (int i = 0; i < k + 2; i++)
						{
							h[i*mr + k] = y[i];
						}
					}
					mu = sqrt(h[k*mr + k] * h[k*mr + k] + h[(k + 1)*mr + k] * h[(k + 1)*mr + k]);
					c[k] = h[k*mr + k] / mu;
					s[k] = -h[(k + 1)*mr + k] / mu;
					h[k*mr + k] = c[k] * h[k*mr + k] - s[k] * h[(k + 1)*mr + k];
					h[(k + 1)*mr + k] = 0.0;
					mult_givens(c[k], s[k], k, g);

					rho = fabs(g[k + 1]);

					itr_used = itr_used + 1;

					if (display)
					{
						printf("  K   = %d  Residual = %f\n", k, rho);
					}

					if (rho <= rho_tol && rho <= tol_abs)
					{
						break;
					}
				}

				k = k_copy;

				y[k] = g[k] / h[k*mr + k];
				for (int i = k - 1; 0 <= i; i--)
				{
					y[i] = g[i];
					for (int j = i + 1; j < k + 1; j++)
					{
						y[i] = y[i] - h[i*mr + j] * y[j];
					}
					y[i] = y[i] / h[i*mr + i];
				}
				for (int i = 0; i < n; i++)
				{
					for (int j = 0; j < k + 1; j++)
					{
						x[i] = x[i] + v[j*n + i] * y[j];
					}
				}

				if (rho <= rho_tol && rho <= tol_abs)
				{
					break;
				}
			}

			if (display)
			{
				printf("\n");
				printf("PMGMRES_ILU_CR:\n");
				printf("  Iterations = %d\n",itr_used);
				printf("  Final residual = %f\n", rho);
			}
			//
			//  Free memory.
			//
			free(c);
			free(g);
			free(h);
			free(l);
			free(r);
			free(s);
			free(ua);
			free(v);
			free(y);
			return true;
		}

		template<class T>
		static T r8vec_dot(int n, const T* a1, const T* a2)
		{
			//
			//  Purpose:
			//
			//    R8VEC_DOT computes the dot product of a pair of R8VEC's.
			//
			//  Licensing:
			//
			//    This code is distributed under the GNU LGPL license. 
			//
			//  Modified:
			//
			//    03 July 2005
			//
			//  Author:
			//
			//    John Burkardt
			//
			//  Parameters:
			//
			//    Input, int N, the number of entries in the vectors.
			//
			//    Input, double A1[N], A2[N], the two vectors to be considered.
			//
			//    Output, double R8VEC_DOT, the dot product of the vectors.
			//

			
			T value = 0.0;
			for (int i = 0; i < n; i++)
			{
				value += a1[i] * a2[i];
			}
			return value;
		}

		template<class T>
		static bool r8vec_uniform_01(int n, int &seed, T*& out, bool display)
		{
			//
			//  Purpose:
			//
			//    R8VEC_UNIFORM_01 returns a unit pseudorandom R8VEC.
			//
			//  Discussion:
			//
			//    This routine implements the recursion
			//
			//      seed = 16807 * seed mod ( 2^31 - 1 )
			//      unif = seed / ( 2^31 - 1 )
			//
			//    The integer arithmetic never requires more than 32 bits,
			//    including a sign bit.
			//
			//  Licensing:
			//
			//    This code is distributed under the GNU LGPL license. 
			//
			//  Modified:
			//
			//    19 August 2004
			//
			//  Author:
			//
			//    John Burkardt
			//
			//  Reference:
			//
			//    Paul Bratley, Bennett Fox, Linus Schrage,
			//    A Guide to Simulation,
			//    Second Edition,
			//    Springer, 1987,
			//    ISBN: 0387964673,
			//    LC: QA76.9.C65.B73.
			//
			//    Bennett Fox,
			//    Algorithm 647:
			//    Implementation and Relative Efficiency of Quasirandom
			//    Sequence Generators,
			//    ACM Transactions on Mathematical Software,
			//    Volume 12, Number 4, December 1986, pages 362-376.
			//
			//    Pierre L'Ecuyer,
			//    Random Number Generation,
			//    in Handbook of Simulation,
			//    edited by Jerry Banks,
			//    Wiley, 1998,
			//    ISBN: 0471134031,
			//    LC: T57.62.H37.
			//
			//    Peter Lewis, Allen Goodman, James Miller,
			//    A Pseudo-Random Number Generator for the System/360,
			//    IBM Systems Journal,
			//    Volume 8, Number 2, 1969, pages 136-143.
			//
			//  Parameters:
			//
			//    Input, int N, the number of entries in the vector.
			//
			//    Input/output, int &SEED, a seed for the random number generator.
			//
			//    Output, double R8VEC_UNIFORM_01[N], the vector of pseudorandom values.
			//

			if (seed == 0)
			{
				if (display)
				{
					printf("\n");
					printf("R8VEC_UNIFORM_01 - Fatal error!\n");
					printf("  Input value of SEED = 0.\n");
				}
				return false;
			}

			out = (T*)malloc(sizeof(T)*n);
			if (out == 0)
				return false;

			for (int i = 0; i < n; i++)
			{
				k = seed / 127773;

				seed = 16807 * (seed - k * 127773) - k * 2836;

				if (seed < 0)
				{
					seed = seed + 2147483647;
				}

				r[i] = (double)(seed)* 4.656612875E-10;
			}

			return true;
		}

		template<class T>
		static void rearrange_cr(int n, int nz_num, int* ia, int* ja, T* a)
		{
			//  Purpose:
			//
			//    REARRANGE_CR sorts a sparse compressed row matrix.
			//
			//  Discussion:
			//
			//    This routine guarantees that the entries in the CR matrix
			//    are properly sorted.
			//
			//    After the sorting, the entries of the matrix are rearranged in such
			//    a way that the entries of each column are listed in ascending order
			//    of their column values.
			//
			//    The matrix A is assumed to be stored in compressed row format.  Only
			//    the nonzero entries of A are stored.  The vector JA stores the
			//    column index of the nonzero value.  The nonzero values are sorted
			//    by row, and the compressed row vector IA then has the property that
			//    the entries in A and JA that correspond to row I occur in indices
			//    IA[I] through IA[I+1]-1.
			//
			//  Licensing:
			//
			//    This code is distributed under the GNU LGPL license. 
			//
			//  Modified:
			//
			//    18 July 2007
			//
			//  Author:
			//
			//    Original C version by Lili Ju.
			//    C++ version by John Burkardt.
			//
			//  Parameters:
			//
			//    Input, int N, the order of the system.
			//
			//    Input, int NZ_NUM, the number of nonzeros.
			//
			//    Input, int IA[N+1], the compressed row index.
			//
			//    Input/output, int JA[NZ_NUM], the column indices.  On output,
			//    the order of the entries of JA may have changed because of the sorting.
			//
			//    Input/output, double A[NZ_NUM], the matrix values.  On output, the
			//    order of the entries may have changed because of the sorting.
			//


			for (int i = 0; i < n; i++)
			{
				int j1 = ia[i];
				int j2 = ia[i + 1];
				int is = j2 - j1;

				for (int k = 1; k < is; k++)
				{
					for (int j = j1; j < j2 - k; j++)
					{
						if (ja[j + 1] < ja[j])
						{
							int itemp = ja[j + 1];
							ja[j + 1] = ja[j];
							ja[j] = itemp;

							T dtemp = a[j + 1];
							a[j + 1] = a[j];
							a[j] = dtemp;
						}
					}
				}
			}
			return;
		}
	};
}

#endif