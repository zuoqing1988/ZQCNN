#ifndef _ZQ_LEVMAR_H_
#define _ZQ_LEVMAR_H_
#pragma once

#include "ZQ_SVD.h"
#include <math.h>

namespace ZQ
{
	class ZQ_LevMarOptions
	{
	public:
		ZQ_LevMarOptions():init_mu(0.001),tol_max_jte(1e-16),tol_dx_square(1e-16),tol_e_square(1e-16){}
		~ZQ_LevMarOptions(){}

		double init_mu;
		double tol_max_jte;
		double tol_dx_square;
		double tol_e_square;
	};

	class ZQ_LevMarReturnInfos
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


		ZQ_LevMarReturnInfos():init_e_square(0),final_e_square(0),final_max_jte(0),final_dx_square(0),exit_code(EXIT_CODE_FAILED),
			iter_count(0),func_count(0),jacf_count(0),linsolver_count(0)
		{

		}
		~ZQ_LevMarReturnInfos(){}

		double init_e_square;
		double final_e_square;
		double final_max_jte;
		double final_dx_square;
		EXIT_CODE exit_code;
		int iter_count;
		int func_count;
		int jacf_count;
		int linsolver_count;
	};

	class ZQ_LevMar
	{
	public:
		template<class T>
		static bool ZQ_LevMar_Der(
			bool (*func)(const T* x, T* fx, int m, int n, const void* data),
			bool (*jacf)(const T* x, T* jx, int m, int n, const void* data),
			T* x, 
			const T* fx, 
			int m, 
			int n, 
			int max_iter,
			const ZQ_LevMarOptions& opts,
			ZQ_LevMarReturnInfos& info,
			const void* data,
			bool display = false);

	};

	/*******************   definitions  ********************/

	template<class T>
	bool ZQ_LevMar::ZQ_LevMar_Der(
		bool (*func)(const T* x, T* fx, int m, int n, const void* data),//functional relation describing measurements. x \in R^m, fx \in R^n
		bool (*jacf)(const T* x, T* jx, int m, int n, const void* data),//function to evaluate the Jacobian. \part fx / \part x
		T *x,															//I/0, initial estimates
		const T *fx,													//I, measurement
		int m,															//I, unknowns dimension
		int n,															//I, measurement dimension
		int max_iter,													
		const ZQ::ZQ_LevMarOptions &opts, ZQ::ZQ_LevMarReturnInfos &info, const void *data, bool display)
	{
		if(n <= 0 || m <= 0 || n < m || func == 0 || jacf == 0)
		{
			info.exit_code = ZQ_LevMarReturnInfos::EXIT_CODE_FAILED;
			return false;
		}


		ZQ_Matrix<T> e_mat(n,1);
		ZQ_Matrix<T> fxbar_mat(n,1);
		ZQ_Matrix<T> jacTe_mat(m,1);
		ZQ_Matrix<T> jac_mat(n,m);
		ZQ_Matrix<T> jacTjac_mat(m,m);
		ZQ_Matrix<T> Dx_mat(m,1);
		ZQ_Matrix<T> diag_jacTjac_mat(m,1);
		ZQ_Matrix<T> xDx_mat(m,1);

		T* e = e_mat.GetDataPtr();
		T* fxbar = fxbar_mat.GetDataPtr();
		T* jacTe = jacTe_mat.GetDataPtr();
		T* jac = jac_mat.GetDataPtr();
		T* jacTjac = jacTjac_mat.GetDataPtr();
		T* Dx = Dx_mat.GetDataPtr();
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
			info.exit_code = ZQ_LevMarReturnInfos::EXIT_CODE_FUNC_FAILED;
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

		info.jacf_count = 0;

		info.iter_count = 0;
		for(int it = 0;it < max_iter; it++, info.iter_count++)
		{	
			if(x_eL2 <= tol_e_sq)
			{
				info.exit_code = ZQ_LevMarReturnInfos::EXIT_CODE_SMALL_E;
				return true;
			}

			/*** compute jacobian ****/
			info.jacf_count ++;
			if(!(*jacf)(x,jac,m,n,data))
			{
				info.exit_code = ZQ_LevMarReturnInfos::EXIT_CODE_JACF_FAILED;
				return false;
			}

			/* compute jacTjac, jacTe*/
			memset(jacTjac,0,sizeof(T)*m*m);
			memset(jacTe,0,sizeof(T)*m);
			for(int i = 0;i < m;i++)
			{
				for(int j = i;j < m;j++)
				{
					for(int k = 0;k < n;k++)
						jacTjac[i*m+j] += jac[k*m+i]*jac[k*m+j];
					jacTjac[j*m+i] = jacTjac[i*m+j];
				}
				for(int j = 0;j < n;j++)
					jacTe[i] += jac[j*m+i]*e[j];
			}

			/* Compute ||J^T e||_inf, ||p||^2, diag_jacTjac*/
			double jacTe_max = 0.0;
			double x_L2 = 0;
			for(int i = 0;i < m;i++)
			{
				double tmp = fabs(jacTe[i]);
				if(jacTe_max < tmp)
					jacTe_max = tmp;
				x_L2 += x[i]*x[i];
				diag_jacTjac[i] = jacTjac[i*m+i];
			}
			info.final_max_jte = jacTe_max;

			/* check for convergence */
			if(jacTe_max <= opts.tol_max_jte)
			{
				info.final_dx_square = 0;
				info.exit_code = ZQ_LevMarReturnInfos::EXIT_CODE_SMALL_JTE;
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
				for(int i = 0; i < m; i++)
				{
					jacTjac[i*m+i] += mu;
				}

				/* solve augmented equations */
				info.linsolver_count++;
				
 				if(!ZQ_SVD::Solve(jacTjac_mat,Dx_mat,jacTe_mat))
				{
					info.exit_code = ZQ_LevMarReturnInfos::EXIT_CODE_NO_FURTHER_REDUCTION;
					return false;
				}

				
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
					info.exit_code = ZQ_LevMarReturnInfos::EXIT_CODE_SMALL_DX;
					return true;
				}
				if(Dx_L2 >= (x_L2+tol_dx_sq)/(epsilon*epsilon))
				{
					info.exit_code = ZQ_LevMarReturnInfos::EXIT_CODE_SINGULAR_MATRIX;
					return false;
				}

				info.func_count++;
				if(!(*func)(xDx,fxbar,m,n,data))
				{
					info.exit_code = ZQ_LevMarReturnInfos::EXIT_CODE_FUNC_FAILED;
					return false;
				}

				/* compute ||e(xDx)||_2 */
				double xDx_eL2 = 0;
				for(int i = 0;i < n;i++)
				{
					double tmp = fx[i] - fxbar[i];
					/*printf("[%d]%e ", i, tmp);
					if ((i + 1) % 10 == 0)
						printf("\n");*/
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
						info.exit_code = ZQ_LevMarReturnInfos::EXIT_CODE_NO_FURTHER_REDUCTION;
						return false;
					}
					nu = nu2;

					for(int i = 0; i < m; i++) /* restore diagonal J^T J entries */
						jacTjac[i*m+i]=diag_jacTjac[i];

				}
			}
		}

		info.exit_code = ZQ_LevMarReturnInfos::EXIT_CODE_MAX_ITER;

		return true;
	}
}

#endif