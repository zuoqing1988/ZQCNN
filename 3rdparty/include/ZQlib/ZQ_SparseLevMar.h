#ifndef _ZQ_SPARSE_LEVMAR_H_
#define _ZQ_SPARSE_LEVMAR_H_
#pragma once


#include "ZQ_PCGSolver.h"
#include "ZQ_taucs.h"
#include "ZQ_TaucsBase.h"
#include <math.h>
#include <ZQ_Matrix.h>

//#define _USE_UMFPACK 1

#if _USE_UMFPACK
#include "ZQ_UmfpackSolver.h"
#endif

namespace ZQ
{
	class ZQ_SparseLevMarOptions
	{
	public:
		ZQ_SparseLevMarOptions():init_mu(0.001),tol_max_jte(1e-16),tol_dx_square(1e-16),tol_e_square(1e-16),pcg_solver_max_iter(100),pcg_solver_tol(1e-16){}
		~ZQ_SparseLevMarOptions(){}

		double init_mu;
		double tol_max_jte;
		double tol_dx_square;
		double tol_e_square;
		int pcg_solver_max_iter;
		double pcg_solver_tol;
	};

	class ZQ_SparseLevMarReturnInfos
	{
	public:
		enum EXIT_CODE{
			EXIT_CODE_FAILED,
			EXIT_CODE_SMALL_JTE,				//stopped by small gradient J^T e
			EXIT_CODE_SMALL_DX,					//stopped by small Dp
			EXIT_CODE_MAX_ITER,					//stopped by max iter
			EXIT_CODE_SINGULAR_MATRIX,			//singular matrix. Restart from current p with increased mu 
			EXIT_CODE_NO_FURTHER_REDUCTION,		//no further error reduction is possible. Restart with increased mu
			EXIT_CODE_SMALL_E,					//stopped by small ||e||_2
			EXIT_CODE_FUNC_FAILED,				//func failed
			EXIT_CODE_JACF_FAILED
		};


		ZQ_SparseLevMarReturnInfos():init_e_square(0),final_e_square(0),final_max_jte(0),final_dx_square(0),exit_code(EXIT_CODE_FAILED),
			iter_count(0),func_count(0),jacf_count(0),linsolver_count(0),linsolver_it_count(0)
		{

		}
		~ZQ_SparseLevMarReturnInfos(){}

		double init_e_square;
		double final_e_square;
		double final_max_jte;
		double final_dx_square;
		EXIT_CODE exit_code;
		int iter_count;
		int func_count;
		int jacf_count;
		int linsolver_count;
		int linsolver_it_count;
	};

	class ZQ_SparseLevMar
	{
	public:
		/*
		 func:	functional relation describing measurements. x \in R^m, fx \in R^n
		 jacf:	function to evaluate the Jacobian. \part fx / \part x
		 x:   	unknowns
		 fx:	measurements
		 m:		dim of unknowns
		 n:		dim of measurements
		 max_iter: max iteration
		*/
		template<class T>
		static bool ZQ_SparseLevMar_Der(
			bool (*func)(const T* x, T* fx, int m, int n, const void* data),
			bool (*jacf)(const T* x, taucs_ccs_matrix*& jx, int m, int n, const void* data),
			T* x, 
			const T* fx, 
			int m, 
			int n, 
			int max_iter,
			const ZQ_SparseLevMarOptions& opts,
			ZQ_SparseLevMarReturnInfos& info,
			const void* data,
			bool display = false
			);

	private:
		template<class T>
		static bool _augment_jx(const taucs_ccs_matrix* jx, int m, int n, double sqrt_mu, taucs_ccs_matrix*& aug_jx);
	};

	/*******************   definitions  ********************/

	template<class T>
	bool ZQ_SparseLevMar::ZQ_SparseLevMar_Der(
		bool (*func)(const T* x, T* fx, int m, int n, const void* data),//functional relation describing measurements. x \in R^m, fx \in R^n
		bool (*jacf)(const T* x, taucs_ccs_matrix*& jx, int m, int n, const void* data),//function to evaluate the Jacobian. \part fx / \part x
		T *x,															//I/0, initial estimates
		const T *fx,													//I, measurement
		int m,															//I, unknowns dimension
		int n,															//I, measurement dimension
		int max_iter,													
		const ZQ::ZQ_SparseLevMarOptions &opts, ZQ::ZQ_SparseLevMarReturnInfos &info, const void *data, bool display)
	{
		if(n <= 0 || m <= 0 || n < m || func == 0 || jacf == 0)
		{
			info.exit_code = ZQ_SparseLevMarReturnInfos::EXIT_CODE_FAILED;
			return false;
		}


		ZQ_Matrix<T> e_mat(n+m,1);
		ZQ_Matrix<T> fxbar_mat(n,1);
		ZQ_Matrix<T> jacTe_mat(m,1);
		ZQ_Matrix<T> jacTjac_mat(m,m);
		ZQ_Matrix<T> Dx_mat(m,1);
		ZQ_Matrix<T> Dx0_mat(m,1);
		ZQ_Matrix<T> diag_jacTjac_mat(m,1);
		ZQ_Matrix<T> xDx_mat(m,1);
		taucs_ccs_matrix* jac_mat = 0;

		T* e = e_mat.GetDataPtr();
		T* fxbar = fxbar_mat.GetDataPtr();
		T* jacTe = jacTe_mat.GetDataPtr();
		T* Dx = Dx_mat.GetDataPtr();
		T* Dx0 = Dx0_mat.GetDataPtr();
		memset(Dx0,0,sizeof(T)*m);
		T* diag_jacTjac = diag_jacTjac_mat.GetDataPtr();
		T* xDx = xDx_mat.GetDataPtr();		

		double tau = opts.init_mu;
		double tol_max_jte = opts.tol_max_jte;
		double tol_dx_sq = opts.tol_dx_square;
		double tol_e_sq = opts.tol_e_square;

		const double epsilon = 1e-12;
		int nu = 2;
		double x_eL2;
		double mu = 0;


		info.func_count = 1;
		if(!(*func)(x,fxbar,m,n,data))
		{
			info.exit_code = ZQ_SparseLevMarReturnInfos::EXIT_CODE_FUNC_FAILED;
			return false;
		}		

		/* compute e=x - f(p) and its L2 norm */
		x_eL2 = 0;
		for(int i = 0;i < n;i++)
		{
			e[i] = fx[i] - fxbar[i];
			x_eL2 += e[i]*e[i];
		}
		info.init_e_square = x_eL2;
		info.final_e_square = x_eL2;

		info.func_count = 0;
		info.jacf_count = 0;
		info.iter_count = 0;
		info.linsolver_count = 0;
		info.linsolver_it_count = 0;
		for(int it = 0;it < max_iter; it++, info.iter_count++)
		{	
			if(x_eL2 <= tol_e_sq)
			{
				info.exit_code = ZQ_SparseLevMarReturnInfos::EXIT_CODE_SMALL_E;
				return true;
			}

			/*** compute jacobian ****/
			info.jacf_count ++;
			if(!(*jacf)(x,jac_mat,m,n,data))
			{
				info.exit_code = ZQ_SparseLevMarReturnInfos::EXIT_CODE_JACF_FAILED;
				return false;
			}

			/* compute jacTe, diag_jacTjac*/
			ZQ_TaucsBase::ZQ_taucs_ccs_vec_time_matrix(e,jac_mat,jacTe);
			if(!ZQ_TaucsBase::ZQ_taucs_ccs_GetAtADiag(jac_mat,diag_jacTjac))
			{
				info.exit_code = ZQ_SparseLevMarReturnInfos::EXIT_CODE_FAILED;
				ZQ_TaucsBase::ZQ_taucs_ccs_free(jac_mat);
				jac_mat = 0;
				return false;
			}
			

			/* Compute ||J^T e||_inf, ||p||^2 */
			double jacTe_max = 0.0;
			double x_L2 = 0;
			for(int i = 0;i < m;i++)
			{
				double tmp = fabs(jacTe[i]);
				if(jacTe_max < tmp)
					jacTe_max = tmp;
				x_L2 += x[i]*x[i];
			}
			info.final_max_jte = jacTe_max;

			/* check for convergence */
			if(jacTe_max <= opts.tol_max_jte)
			{
				info.final_dx_square = 0;
				info.exit_code = ZQ_SparseLevMarReturnInfos::EXIT_CODE_SMALL_JTE;
				return true;
			}

			/* compute initial damping factor */
			if(it == 0)
			{
				double max_diag_jacTjac = diag_jacTjac[0];
				/* find max diagonal element */
				for(int i = 1; i < m; i++)
				{
					if(diag_jacTjac[i] > max_diag_jacTjac)
						max_diag_jacTjac = diag_jacTjac[i];
				}
				mu = tau*max_diag_jacTjac;
			}

			/* determine increment using adaptive damping */
			while(1)
			{
				/* augment normal equations */
				double sqrt_mu = sqrt(mu);
				taucs_ccs_matrix* aug_jac_mat = 0;
				if(!_augment_jx<T>(jac_mat,m,n,sqrt_mu,aug_jac_mat))
				{
					info.exit_code = ZQ_SparseLevMarReturnInfos::EXIT_CODE_FAILED;
					ZQ_TaucsBase::ZQ_taucs_ccs_free(jac_mat);
					jac_mat = 0;
					return false;
				}

				/* solve augmented equations */
				
#if _USE_UMFPACK

				taucs_ccs_matrix* aug_jac_T_mat = ZQ_TaucsBase::ZQ_taucs_ccs_matrixTranspose(aug_jac_mat);
				taucs_ccs_matrix* jacTjac = ZQ_TaucsBase::ZQ_taucs_ccs_mul2NonSymmetricMatrices(aug_jac_T_mat, aug_jac_mat);
				if (!ZQ_UmfpackSolver::UmfpackSolve(jacTjac, jacTe, Dx, display))
				{
					ZQ_TaucsBase::ZQ_taucs_ccs_free(aug_jac_T_mat);
					ZQ_TaucsBase::ZQ_taucs_ccs_free(jacTjac);
					aug_jac_T_mat = 0;
					jacTjac = 0;
					return false;
				}
				ZQ_TaucsBase::ZQ_taucs_ccs_free(aug_jac_T_mat);
				ZQ_TaucsBase::ZQ_taucs_ccs_free(jacTjac);
				aug_jac_T_mat = 0;
				jacTjac = 0;
				info.linsolver_it_count++;
#else
				int pcg_it;
				if(!ZQ_PCGSolver::PCG_sparse_unsquare(aug_jac_mat,e,Dx0,opts.pcg_solver_max_iter,opts.pcg_solver_tol,Dx,pcg_it,display))
				{
					info.exit_code = ZQ_SparseLevMarReturnInfos::EXIT_CODE_NO_FURTHER_REDUCTION;
					ZQ_TaucsBase::ZQ_taucs_ccs_free(jac_mat);
					ZQ_TaucsBase::ZQ_taucs_ccs_free(aug_jac_mat);
					jac_mat = 0;
					aug_jac_mat = 0;
					return false;
				}
				info.linsolver_it_count += pcg_it;
#endif
				ZQ_TaucsBase::ZQ_taucs_ccs_free(aug_jac_mat);
				aug_jac_mat = 0;


				/* compute p's new estimate and ||Dp||^2 */
				double Dx_L2 = 0;
				for(int i = 0; i < m; i++)
				{
					xDx[i] = x[i] + Dx[i];
					Dx_L2 += Dx[i]*Dx[i];
				}
				info.final_dx_square = Dx_L2;
				if(Dx_L2 < tol_dx_sq)
				{
					info.exit_code = ZQ_SparseLevMarReturnInfos::EXIT_CODE_SMALL_DX;
					return true;
				}
				if(Dx_L2 >= (x_L2+tol_dx_sq)/(epsilon*epsilon))
				{
					info.exit_code = ZQ_SparseLevMarReturnInfos::EXIT_CODE_SINGULAR_MATRIX;
					return false;
				}

				info.func_count++;
				if(!(*func)(xDx,fxbar,m,n,data))
				{
					info.exit_code = ZQ_SparseLevMarReturnInfos::EXIT_CODE_FUNC_FAILED;
					return false;
				}

				/* compute ||e(xDx)||_2 */
				double xDx_eL2 = 0;
				for(int i = 0;i < n;i++)
				{
					double tmp = fx[i] - fxbar[i];
					fxbar[i] = tmp; //temporal store e in fxbar
					xDx_eL2 += tmp*tmp;
				}


				double dL = 0;
				for(int i = 0;i < m;i++)
				{
					dL += Dx[i]*(mu*Dx[i]+jacTe[i]);
				}

				double dF = x_eL2 - xDx_eL2;
				if(dL > 0 && dF > 0)/* reduction in error, increment is accepted */
				{
					double tmp = 2.0*dF/dL - 1.0;
					tmp = 1.0 - tmp*tmp*tmp;
					mu = mu*__max(1.0/3.0,tmp);
					nu = 2;

					for(int i = 0;i < m;i++)
						x[i] = xDx[i];

					for(int i = 0;i < n;i++)
						e[i] = fxbar[i];

					x_eL2 = xDx_eL2;
					info.final_e_square = x_eL2;
					break;
				}
				else
				{
					/* if this point is reached,
					* the error did not reduce; the increment must be rejected
					*/

					mu *= nu;
					int nu2 = nu<<1; // 2*nu;
					if(nu2 <= nu) /* nu has wrapped around (overflown) */
					{ 
						info.exit_code = ZQ_SparseLevMarReturnInfos::EXIT_CODE_NO_FURTHER_REDUCTION;
						ZQ_TaucsBase::ZQ_taucs_ccs_free(jac_mat);
						jac_mat = 0;
						return false;
					}
					nu = nu2;
				}
			}
		}

		info.exit_code = ZQ_SparseLevMarReturnInfos::EXIT_CODE_MAX_ITER;
		ZQ_TaucsBase::ZQ_taucs_ccs_free(jac_mat);
		jac_mat = 0;

		return true;
	}

	template<class T>
	bool ZQ_SparseLevMar::_augment_jx(const taucs_ccs_matrix* jx, int m, int n, double sqrt_mu, taucs_ccs_matrix*& aug_jx)
	{
		if(jx == 0 || jx->m != n || jx->n != m)
			return false;
		int flag;
		if(strcmp(typeid(T).name(),"float") == 0)
			flag = TAUCS_SINGLE;
		else if(strcmp(typeid(T).name(),"double") == 0)
			flag = TAUCS_DOUBLE;
		else
			return false;

		if(aug_jx)
			ZQ_TaucsBase::ZQ_taucs_ccs_free(aug_jx);

		aug_jx = ZQ_TaucsBase::ZQ_taucs_ccs_create(n+m,m,jx->colptr[m]+m,flag);
		if(aug_jx == 0)
			return false;

		for(int i = 0;i < m;i++)
		{
			int old_start = jx->colptr[i];
			int old_end = jx->colptr[i+1];
			int new_start = old_start + i;

			int j;
			for(j = 0;j < old_end - old_start;j++)
			{
				aug_jx->rowind[new_start+j] = jx->rowind[old_start+j];
				((T*)(aug_jx->values.d))[new_start+j] = ((T*)(jx->values.d))[old_start+j];
			}
			aug_jx->rowind[new_start+j] = n+i;
			((T*)(aug_jx->values.d))[new_start+j] = sqrt_mu;
			aug_jx->colptr[i] = jx->colptr[i] + i;
		}
		aug_jx->colptr[m] = jx->colptr[m]+m;

		return true;
	}
}

#endif