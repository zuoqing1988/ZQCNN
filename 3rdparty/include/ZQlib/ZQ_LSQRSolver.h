#ifndef _ZQ_LSQR_SOLVER_H_
#define _ZQ_LSQR_SOLVER_H_
#pragma once

#include "ZQ_taucs.h"
#include "ZQ_TaucsBase.h"
#include "ZQ_LSQRUtils.h"

namespace ZQ
{
	class ZQ_LSQRSolver
	{
	public:
		template<class T>
		static bool LSQRSolve(taucs_ccs_matrix* A, T* b, T* x0, int max_it, double tol, T* x, int& it, bool display = false);

	private:
		template<class T>
		static void _aprod(int mode, int m, int n, T* x, T* y, void* UsrWrk);
	};


	/********************************  definitions **********************************************/

	template<class T>
	bool ZQ_LSQRSolver::LSQRSolve(taucs_ccs_matrix* A, T* b, T* x0, int max_it, double tol, T* x, int& it, bool display /* = false */ )
	{
		if( A == 0 || x0 == 0 || b == 0 || x == 0 || (A->flags & TAUCS_DOUBLE == 0 && A->flags & TAUCS_SINGLE == 0))
			return false;

		int m = A->m;
		int n = A->n;

		double damp = 0;
		T* v = new T[n];
		T* w = new T[n];
		T* se = 0;
		double atol = tol;
		double btol = tol;
		double conlim = 1e9;
		FILE* nout = 0;
		int istop_out = 0;

		T anorm_out = 0;
		T acond_out = 0;
		T rnorm_out = 0;
		T arnorm_out = 0;
		T xnorm_out = 0;

		memcpy(x,x0,sizeof(T)*n);
		ZQ_LSQRUtils::lsqr<T>(m,n,_aprod,damp,A,b,v,w,x,
			se,atol,btol,conlim,max_it,
			nout,&istop_out,&it,&anorm_out,&acond_out,&rnorm_out,&arnorm_out,&xnorm_out);

		if(display)
		{
			printf("it = %d\n",it);
			printf("condition number is %f\n",acond_out);
			printf("stopflag = %d\n",istop_out);
			printf("rnorm=%f\n",rnorm_out);
		}
		delete []w;
		delete []v;

		return true;

	}

	
	template<class T>
	void ZQ_LSQRSolver::_aprod(int mode, int m, int n, T* x, T* y, void* UsrWrk)
	{
		if(mode == 1)
		{
			T* tmp = new T[m];
			taucs_ccs_matrix* A = (taucs_ccs_matrix*)UsrWrk;
			ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec(A,x,tmp);
			for(int i = 0;i < m;i++)
				y[i] += tmp[i];
			delete []tmp;

		}
		else if (mode == 2)
		{
			T* tmp = new T[n];
			taucs_ccs_matrix* A = (taucs_ccs_matrix*)UsrWrk;
			ZQ_TaucsBase::ZQ_taucs_ccs_vec_time_matrix(y,A,tmp);
			for(int i = 0;i < n;i++)
				x[i] += tmp[i];
			delete []tmp;
		}
	}

}


#endif