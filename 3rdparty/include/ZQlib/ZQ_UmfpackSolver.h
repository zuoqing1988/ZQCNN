#ifndef _ZQ_UMFPACK_SOLVER_H_
#define _ZQ_UMFPACK_SOLVER_H_

#if _USE_UMFPACK
#include "umfpack.h"
#ifdef _DEBUG
#pragma comment(lib,"libumfpackd.lib")
#else
#pragma comment(lib,"libumfpack.lib")
#endif
#endif

#include "ZQ_taucs.h"
namespace ZQ
{
	class ZQ_UmfpackSolver
	{
	public:
		template<class T>
		static bool UmfpackSolve(const taucs_ccs_matrix* A, const T* b, T* x, bool display = false);
	};

	/******************/
	template<class T>
	bool ZQ_UmfpackSolver::UmfpackSolve(const taucs_ccs_matrix* A, const T* b, T* x, bool display /*= false*/)
	{
#if _USE_UMFPACK
		if (strcmp(typeid(T).name(), "double") == 0)
		{
			if (A->flags & TAUCS_DOUBLE == 0)
			{
				if (display)
					printf("only double is supported by ZQ_UmfpackSolver!\n");
				return false;
			}
		}
		else
		{
			if (display)
				printf("only double is supported by ZQ_UmfpackSolver!\n");
			return false;
		}
			

		int row = A->m;
		int col = A->n;
		
		void *Symbolic, *Numeric;
		int* Ap = A->colptr;
		int* Ai = A->rowind;
		double* Ax = A->values.d;
		double *null = (double *)NULL;
		int ret_code = umfpack_di_symbolic(row, col, Ap, Ai, Ax, &Symbolic, null, null);
		if (ret_code < 0)
		{
			if (display)
				printf("umfpack_di_symbolic ret_code: %d\n", ret_code);
			return false;
		}
		

		ret_code = umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, null, null);
		umfpack_di_free_symbolic(&Symbolic);
		if (ret_code < 0)
		{
			if (display)
				printf("umfpack_di_numeric ret_code: %d\n", ret_code);
			return false;
		}
			

		ret_code = umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, (double*)x, (double*)b, Numeric, null, null);
		umfpack_di_free_numeric(&Numeric);
		if (ret_code < 0)
		{
			if (display)
				printf("umfpack_di_solve ret_code: %d\n", ret_code);
			return false;
		}

		return true;
#else
		if(display)
			printf("umfpack is not available!\n");
		return false;
#endif
	}
}

#endif