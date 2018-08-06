#ifndef _ZQ_PCG_SOLVER_H_
#define _ZQ_PCG_SOLVER_H_
#pragma once

#include "ZQ_taucs.h"
#include "ZQ_TaucsBase.h"
#include <typeinfo>

namespace ZQ
{
	class ZQ_PCGSolver
	{
	public:
		/* minimize 0.5*x^THx - f^Tx*/
		template<class T>
		static bool PCG(const taucs_ccs_matrix* H, const T* f, const T* x0,const int max_iter, const double tol, T* x, int& it, bool display = false);

		/* minimize ||Ax-b||_2^2*/
		template<class T>
		static bool PCG_sparse_unsquare(const taucs_ccs_matrix* A, const T * f, const T* x0, const int max_iter, const double tol, T* x, int& it,bool display = false);


		/* minimize 0.5*x^THx - f^Tx*/
		template<class T>
		static bool PCG_BQP(const taucs_ccs_matrix* H, const T* f0, const T* x0, const T* l, const T* u,const int max_iter, const double tolx, const double tolf,
			T* x, double& val,int& it, int& exitcode, bool display = false);

	private:
		template<class T>
		static T* _getMatrixDiag(const taucs_ccs_matrix* A, int& row);

		template<class T>
		static T* _getAtAdiag(const taucs_ccs_matrix* A, int& row);

		template<class T>
		static void _definev(const int n, const T* g, const T* x, const T* l, const T* u, T* v, T* dv);

		template<class T>
		static void _fquad(int n, T* x, const T* c, taucs_ccs_matrix* H, void (*fn)(const taucs_ccs_matrix*, const T* ,T*), taucs_ccs_matrix * D,double& val, T* g);
		
		template<class T>
		static void _reflect(int n,const T* y, const T* l, const T* u, T* x,T* sigma);

		template<class T>
		static void _shiftsc(int n, const T* xstart_in, const T* l_in, const T* u_in, T* typx, void (*fn)(const taucs_ccs_matrix* , const T* ,T*), const T* c_in, taucs_ccs_matrix* H,
			T* xstart_out, T* l_out, T* u_out, T* ds, taucs_ccs_matrix** DS, T* c_out);

		template<class T>
		static void _unshsca(int n, T* x_in, T* l, T* u, taucs_ccs_matrix* DS, T* x_out);

		template<class T>
		static void _perturb(const int n, const T* x_in , const T* l, const T* u, const double del, const T* y_in, const T* sigma,
			int& flag, T* x_out, T* y_out);

		template<class T>
		static void _biqpbox(int n, const T* s, const T* c, const double strg, const T* x, const T* y, const T* sigma,
			const T* l, const T* u, const double oval, const double po, const double normg, const taucs_ccs_matrix* DS,const taucs_ccs_matrix* H,
			void (*fn)(const taucs_ccs_matrix*, const T*, T*),T* nx, T* nsig, double& alpha);

		template<class T>
		static void _drqpbox(const int n,const T* D, const taucs_ccs_matrix* DS, const T* grad, const double delta, const T* g, const T* dv, void (*fn)(const taucs_ccs_matrix*, const T*,T*),
			void (*fnpc)(const taucs_ccs_matrix*, const T*, const T*, T* ), const double tol,const taucs_ccs_matrix* H, const int kmax, T* s, int& posdef, int& pcgit);

		/*only solve n = 1,2*/
		template<class T>
		static void _trust(int n, const T* g, const T* H, const double delta, T* s);

		/*only solve n = 1,2*/
		template<class T>
		static void _eig(int n, const T* H, T* V, T* D);

		template<class T>
		static double _seceqn(int n,const double lamda,const T* eigval,const T* alpha,const double delta);

		template<class T>
		static void _rfzero(int n,const double lamda,const int itbnd, const T* eigval,const T* alpha,const double delta,const double tol,
			double& b, double& c, int& count);

		template<class T>
		static void _preproj(const T* r,const T* DR, T* w, int n);

		template<class T>
		static void _pcgr(const T* DM,const T* DG,const T* g,const int kmax,const double tol,void (*fn)(const taucs_ccs_matrix*,const T*, T*),
			taucs_ccs_matrix* H,const T* R,T* p,int& posdef, int& k);

	};
		
	/* minimize 0.5*x^THx - f^Tx*/
	template<class T>
	bool ZQ_PCGSolver::PCG(const taucs_ccs_matrix* H, const T* f, const T* x0,const int max_iter, const double tol, T* x, int& it, bool display /* = false */ )
	{
		double tol1 = (tol == 0) ? 1e-6 : tol;
		if(strcmp(typeid(T).name(),"float") == 0)
		{
			if(((H->flags) & TAUCS_SINGLE) == 0)
				return false;
		}
		else if(strcmp(typeid(T).name(),"double") == 0)
		{
			if(((H->flags) & TAUCS_DOUBLE) == 0)
				return false;
		}
		else
			return false;

		const double eps = 1e-9;

		int row = 0;
		T* M = (T*)(_getMatrixDiag<T>(H,row));
		if(M == 0)
		{
			return false;
		}

		for(int ii = 0;ii < row;ii++)
		{
			if(fabs(M[ii]) < eps)
			{
				if(M[ii] > 0)
					M[ii] = eps;
				else if(M[ii] < 0)
					M[ii] = -eps;
				else
					M[ii] = 0;
			}
			else 
				M[ii] = 1/M[ii];
		}
		T * tmp = new T[row];
		T * r = new T[row];
		T * d = new T[row];
		T * q = new T[row];
		T * s = new T[row];

		memcpy(x,x0,sizeof(T)*row);
		int i = 0;
		T* b = (T*)f;
		T* xx = (T*)x;

		ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec((taucs_ccs_matrix*)H,xx,tmp);

		ZQ_MathBase::VecMinus(row,b,tmp,r);

		ZQ_MathBase::VecMul(row,M,r,d);
		double delta_new = ZQ_MathBase::DotProduct(row,r,d);
		double delta0 = delta_new;
		double delta_old;
		int change = sqrt((double)row);
		if(change < 50)
			change = 50;

		double alpha = 0;
		while(i < max_iter && delta_new > tol1 * tol1 * delta0)
		{
			ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec((taucs_ccs_matrix*)H,d,q);
			alpha = delta_new / ZQ_MathBase::DotProduct(row,d,q);
			for(int ii = 0;ii < row;ii++)
				xx[ii] += alpha*d[ii];
			if(i%change == 0)
			{
				ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec((taucs_ccs_matrix*)H,xx,tmp);
				ZQ_MathBase::VecMinus(row,b,tmp,r);
			}
			else
			{
				for(int ii = 0;ii < row;ii++)
					r[ii] -= alpha*q[ii];
			}
			ZQ_MathBase::VecMul(row,M,r,s);
			delta_old = delta_new;
			delta_new = ZQ_MathBase::DotProduct(row,r,s);
			double beta = delta_new / delta_old;
			for(int ii = 0;ii < row;ii++)
				d[ii] = s[ii] + beta*d[ii];
			i++;
			if(i %50 == 0)
			{
				if(display)
					printf("iteration = %d, delta_new = %e\n",i,delta_new);
			}
		}
		it = i;
		delete [](M);
		delete [](tmp);
		delete [](r);
		delete [](d);
		delete [](q);
		delete [](s);
		M = 0;
		tmp = 0;
		r = 0;
		d = 0;
		q = 0;
		s = 0;

		return true;
	}

	/* minimize ||Ax-b||_2^2*/
	template<class T>
	bool ZQ_PCGSolver::PCG_sparse_unsquare(const taucs_ccs_matrix* A, const T * f, const T* x0, const int max_iter, const double tol, T* x, int& it,bool display /* = false */ )
	{
		double tol1 = (tol == 0) ? 1e-6 : tol;
		if(strcmp(typeid(T).name(),"float") == 0)
		{
			if(((A->flags) & TAUCS_SINGLE) == 0)
				return false;
		}
		else if(strcmp(typeid(T).name(),"double") == 0)
		{
			if(((A->flags) & TAUCS_DOUBLE) == 0)
				return false;
		}
		else
			return false;

		const double eps = 1e-9;

		int m = A->m;
		int n = A->n;
		int row = 0;
		T* M = (T*)(_getAtAdiag<T>(A,row));
		if(M == 0)
		{
			printf("M is null\n");
			return false;
		}
		for(int ii = 0;ii < row;ii++)
		{
			if(fabs(M[ii]) < eps)
			{
				if(M[ii] > 0)
					M[ii] = eps;
				else if(M[ii] < 0)
					M[ii] = -eps;
				else
					M[ii] = 0;
			}
			else 
				M[ii] = 1/M[ii];
		}
		T * tmp1 = new T[m];
		T * tmp2 = new T[row];
		T * r = new T[row];
		T * d = new T[row];
		T * q = new T[row];
		T * s = new T[row];
		T * b = new T[row];

		memcpy(x,x0,sizeof(T)*row);
		int i = 0;
		ZQ_TaucsBase::ZQ_taucs_ccs_vec_time_matrix(f,A,b);
		T* xx = (T*)x;
		ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec(A,xx,tmp1);
		ZQ_TaucsBase::ZQ_taucs_ccs_vec_time_matrix(tmp1,A,tmp2);

		ZQ_MathBase::VecMinus(row,b,tmp2,r);

		ZQ_MathBase::VecMul(row,M,r,d);
		double delta_new = ZQ_MathBase::DotProduct(row,r,d);
		double delta0 = delta_new;
		double delta_old;
		int change = sqrt((double)row);
		if(change < 50)
			change = 50;

		double alpha = 0;
		while(i < max_iter && delta_new > tol1 * tol1 * delta0)
		{
			ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec(A,d,tmp1);
			ZQ_TaucsBase::ZQ_taucs_ccs_vec_time_matrix(tmp1,A,q);
			alpha = delta_new / ZQ_MathBase::DotProduct(row,d,q);
			for(int ii = 0;ii < row;ii++)
				xx[ii] += alpha*d[ii];
			if(i%change == 0)
			{
				ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec(A,xx,tmp1);
				ZQ_TaucsBase::ZQ_taucs_ccs_vec_time_matrix(tmp1,A,tmp2);
				ZQ_MathBase::VecMinus(row,b,tmp2,r);
				//printf("[it=%d],residual:%f\n",i,ZQ_Dot(r,r,row,ZQ_DOUBLE));
			}
			else
			{
				for(int ii = 0;ii < row;ii++)
					r[ii] -= alpha*q[ii];
			}
			ZQ_MathBase::VecMul(row,M,r,s);
			delta_old = delta_new;
			delta_new = ZQ_MathBase::DotProduct(row,r,s);
			double beta = delta_new / delta_old;
			for(int ii = 0;ii < row;ii++)
				d[ii] = s[ii] + beta*d[ii];
			i++;
			if(i %50 == 0)
			{
				if(display)
					printf("iteration = %d, delta_new = %e\n",i,delta_new);
			}
		}
		it = i;
		delete [](M);
		delete [](tmp1);
		delete [](tmp2);
		delete [](r);
		delete [](d);
		delete [](q);
		delete [](s);
		delete [](b);
		M = 0;
		tmp1 = 0;
		tmp2 = 0;
		r = 0;
		d = 0;
		q = 0;
		s = 0;
		b = 0;
		return true;
	}

	/* minimize 0.5*x^THx - f^Tx*/
	template<class T>
	bool ZQ_PCGSolver::PCG_BQP(const taucs_ccs_matrix* H, const T* f0, const T* x0, const T* l, const T* u,const int max_iter, const double tolx, const double tolf,
		T* x, double& val,int& it, int& exitcode, bool display /* = false */ )
	{
		if(strcmp(typeid(T).name(),"float") == 0)
		{
			if(H->flags & TAUCS_SINGLE == 0)
				return false;
		}
		else if(strcmp(typeid(T).name(),"double") == 0)
		{
			if(H->flags & TAUCS_DOUBLE == 0)
				return false;
		}
		else
			return false;


		int n = H->n;
		T* lptr = (T*)l;
		T* uptr = (T*)u;
		T* xptr = (T*)x;
		T* low_in = new T[n];
		T* up_in = new T[n];
		T* low_out = new T[n];
		T* up_out = new T[n];
		T* typx = new T[n];
		T* c = new T[n];
		T* xstart = new T[n];
		T* ox = new T[n];
		T* y = new T[n];
		T* v = new T[n];
		T* dv = new T[n];
		T* sigma = new T[n];
		T* osig = new T[n];
		T* g = new T[n]; 
		T* dd = new T[n];
		T* D = new T[n];
		T* grad = new T[n];
		T* s = new T[n];
		T* ds = new T[n];
		T* tmp = new T[n];

		T* f = new T[n];
		for(int i = 0;i < n;i++)
			f[i] = -f0[i];

		int pcflag = 0;
		int kmax = __max(1,int(n/2));
		it = 0;
		double delta;
		taucs_ccs_matrix* DS = 0;

		double tolx2 = sqrt(tolx);
		double tolf2 = sqrt(tolf);

		for(int i = 0;i < n;i++)
		{
			low_in[i] = (lptr[i] < -ZQ_INF) ? -ZQ_INF : lptr[i];
			up_in[i] = (uptr[i] > ZQ_INF) ? ZQ_INF : uptr[i];
			typx[i] = 1.0;
			v[i] = 0.0;
			dv[i] = 1.0;
			sigma[i] = 1.0;
		}
		memset(g,0,sizeof(T)*n);

		_shiftsc<T>(n,x0,low_in,up_in,typx,ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec,f,(taucs_ccs_matrix*)H,xstart,low_out,up_out,ds,&DS,c);

		double dellow = 1.0;
		double delup = 1000.0;
		int npcg = 1;
		double digits = ZQ_INF;

		bool done = false;
		double del = 1e-9;
		double pcgtol = 0.1;

		int posdef = 1;
		int pcgit = 0;

		memcpy(xptr,xstart,sizeof(T)*n);
		memcpy(y,xptr,sizeof(T)*n);

		double oval = ZQ_INF;
		double prediff = 0.0;
		double diff;
		double strg;
		double ostrg;

		_fquad<T>(n,xptr,c,(taucs_ccs_matrix*)H,ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec,DS,val,g);
		_definev(n,g,x,low_out,up_out,v,dv);

		ZQ_MathBase::VecMul(n,v,g,tmp);
		double csnrm = ZQ_MathBase::NormVector_Linf(n,tmp);


		if(csnrm == 0)
		{

			T* dir = new T[n];
			for(int i = 0;i < n;i++)
			{
				if(up_out[i] - xptr[i] > xptr[i] - low_out[i])
					dir[i] = 1;
				else
					dir[i] = -1;
			}
			for(int i = 0;i < n;i++)
			{
				xptr[i] = xptr[i] + dir[i] * (rand()%100001/100000.0)*__max(up_out[i] - xptr[i], xptr[i] - low_out[i])*0.1;
				y[i] = xptr[i];
			}

			_fquad<T>(n,xptr,c,(taucs_ccs_matrix*)H,ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec,DS,val,g);
			delete []dir;
		}

		while(!done)
		{
			it++;
			_definev(n,g,xptr,low_out,up_out,v,dv);

			ZQ_MathBase::VecMul(n,v,g,tmp);
			csnrm = ZQ_MathBase::NormVector_Linf(n,tmp);

			double normv = ZQ_MathBase::NormVector_L2(n,v);


			delta = __min( __max( dellow,normv), delup);

			diff = fabs(oval - val);
			if(it > 1)
				digits = prediff / (__max(diff, 1e-9));

			prediff = diff;
			oval = val;

			if(diff < tolf * (1+fabs(oval)))
			{
				exitcode = 3;
				done = true;
			}
			else if(diff < tolf2 * (1+fabs(oval) && digits < 3.5) && (posdef != 0))
			{
				exitcode = 3;
				done = true;
			}
			else if(csnrm < tolf && (posdef != 0) && it > 1)
			{
				exitcode = 1;
				done = true;
			}

			if(!done)
			{
				for(int i = 0;i < n;i++)
					dd[i] = fabs(v[i]);
				for(int i = 0;i < n;i++)
				{
					D[i] = sqrt(dd[i])*sigma[i];
					grad[i] = D[i]*g[i];
				}
				double normg = ZQ_MathBase::NormVector_L2(n,grad);

				_drqpbox<T>(n,D,DS,grad,delta,g,dv,ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec,ZQ_TaucsBase::ZQ_hprecon,pcgtol,H,kmax,s,posdef,pcgit);
				/*[s,posdef,pcgit] = ZQ_drqpbox(D,DS,grad,delta,g,dv,mtxmpy,...
				pcmtx,pcflags,pcgtol,H,0,kmax);*/

				npcg += pcgit;

				ZQ_MathBase::VecMul(n,sigma,g,tmp);
				strg = ZQ_MathBase::DotProduct(n,s,tmp);

				memcpy(ox,x,sizeof(T)*n);
				memcpy(osig,sigma,sizeof(T)*n);
				ostrg = strg;
				if(strg > 0)
				{
					exitcode = -2;
					done = true;
				}
				else
				{
					double alpha = 0;
					_biqpbox<T>(n,s,c,ostrg,ox,y,osig,low_out,up_out,oval,posdef,normg,DS,H,ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec,x,sigma,alpha);
					/*[x,sigma,alpha] = biqpbox(s,c,ostrg,ox,y,osig,l,u,oval,posdef,...
					normg,DS,mtxmpy,H,0,varargin{:});*/

					if(alpha == 0)
					{
						exitcode = -4;
						done = true;
					}
					for(int i = 0;i < n;i++)
						y[i] += alpha * s[i];
					T* x_in = new T[n];
					T* y_in = new T[n];
					memcpy(x_in,xptr,sizeof(T)*n);
					memcpy(y_in,y,sizeof(T)*n);

					int pert = 0;
					_perturb(n,x_in,low_out,up_out,del,y_in,sigma,pert,x,y);
					delete []x_in;
					delete []y_in;
					x_in = 0;
					y_in = 0;
					_fquad<T>(n,xptr,c,(taucs_ccs_matrix*)H,ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec,DS,val,g);

				}
				if(it > max_iter)
				{
					exitcode = 0;
					done = true;
				}
			}
		}
		memcpy(ox,xptr,sizeof(T)*n);
		_unshsca(n,ox,low_in,up_in,DS,xptr);
		_fquad<T>(n,xptr,f,(taucs_ccs_matrix*)H,ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec,0,val,g);

		delete [] low_in;	low_in = 0;
		delete [] up_in;	up_in = 0;
		delete [] low_out;	low_out = 0;
		delete [] up_out;	up_out = 0;
		delete [] typx;		typx = 0;
		delete [] c;		c = 0;
		delete [] xstart;	xstart = 0;
		delete [] ox;		ox = 0;
		delete [] y;		y = 0;
		delete [] v;		v = 0;
		delete [] dv;		dv = 0;
		delete [] sigma;	sigma = 0;
		delete [] osig;		osig = 0;
		delete [] g;		g = 0;
		delete [] dd;		dd = 0;
		delete [] D;		D = 0;
		delete [] grad;		grad = 0;
		delete [] s;		s = 0;
		delete [] ds;		ds = 0;
		delete [] tmp;		tmp = 0;
		delete [] f;		f = 0;
		ZQ_TaucsBase::ZQ_taucs_ccs_free(DS);	DS = 0;


		return true;

	}

	template<class T>
	T* ZQ_PCGSolver::_getMatrixDiag(const taucs_ccs_matrix* A, int& row)
	{
		if(A == 0 || ((A->flags) & TAUCS_SYMMETRIC) || (((A->flags) & TAUCS_DOUBLE) == 0 && ((A->flags) & TAUCS_SINGLE) == 0))
			return 0;
		row = A->m;

		T* diag = new T[row];

		for(int i = 0;i < row;i++)
		{
			int num = A->colptr[i+1] - A->colptr[i];
			int start = A->colptr[i];
			int index;
			if((index = ZQ_MathBase::BinarySearch(num,(A->rowind)+start,i,true)) == -1)
				diag[i] = 0;
			else
				diag[i] = ((T*)(A->values.d))[start+index];
		}
		return diag;
	}

	template<class T>
	T* ZQ_PCGSolver::_getAtAdiag(const taucs_ccs_matrix* A, int& row)
	{
		if(A == 0 || ((A->flags) & TAUCS_SYMMETRIC) || (((A->flags) & TAUCS_DOUBLE) == 0 && ((A->flags) & TAUCS_SINGLE) == 0))
			return 0;
		int m = A->m;
		int n = A->n;
		row = n;

		T* diag = new T[row];

		for(int i = 0;i < row;i++)
		{
			int num = A->colptr[i+1] - A->colptr[i];
			int start = A->colptr[i];

			double sum = 0;
			for(int index = 0;index < num;index++)
			{
				double val = ((T*)(A->values.d))[index+start];
				sum += val*val;
			}
			diag[i] = sum;
		}
		return diag;

	}

	template<class T>
	void ZQ_PCGSolver::_definev(const int n, const T* g, const T* x, const T* l, const T* u, T* v, T* dv)
	{
		T* gptr = (T*)g;
		T* xptr = (T*)x;
		T* lptr = (T*)l;
		T* uptr = (T*)u;
		T* vptr = (T*)v;
		T* dvptr = (T*)dv;

		memset(vptr,0,sizeof(T)*n);
		memset(dvptr,0,sizeof(T)*n);

		for(int i = 0;i < n;i++)
		{
			if(gptr[i] < 0 && uptr[i] < ZQ_INF)
			{
				vptr[i] = xptr[i] - uptr[i];
				dvptr[i] = 1; 
			}
			else if(gptr[i] >= 0 && lptr[i] > -ZQ_INF)
			{
				vptr[i] = xptr[i] - lptr[i];
				dvptr[i] = 1;
			}
			else if(gptr[i] < 0 && uptr[i] == ZQ_INF)
			{
				vptr[i] = -1;
				dvptr[i] = 0;
			}
			else if(gptr[i] >= 0&& lptr[i] == -ZQ_INF)
			{
				vptr[i] = 1;
				dvptr[i] = 0;
			}
		}
	}

	template<class T>
	void ZQ_PCGSolver::_fquad(int n, T* x, const T* c, taucs_ccs_matrix* H, void (*fn)(const taucs_ccs_matrix*, const T* ,T*), taucs_ccs_matrix * D,double& val, T* g)
	{
		T* xptr = (T*)x;
		T* cptr = (T*)c;
		T* gptr = (T*)g;
		if(D == 0)
		{
			T* tmp = new T[n];
			T* w = new T[n];
			fn(H,x,w);
			ZQ_MathBase::VecPlus(n,w,cptr,gptr);
			for(int i = 0;i < n;i++)
				tmp[i] = 0.5*w[i]+cptr[i];
			val = ZQ_MathBase::DotProduct(n,xptr,tmp);
			delete []tmp;
			delete []w;
			tmp = 0;
			w = 0;
		}
		else
		{
			T* w = new T[n];
			T* ww = new T[n];
			T* tmp = new T[n];
			ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec(D,x,w);
			fn(H,w,ww);
			ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec(D,ww,w);
			ZQ_MathBase::VecPlus(n,w,cptr,gptr);
			for(int i = 0;i < n;i++)
				tmp[i] = 0.5*w[i]+cptr[i];
			val = ZQ_MathBase::DotProduct(n,xptr,tmp);
			delete []w;
			delete []ww;
			delete []tmp;
			w = 0;
			ww = 0;
			tmp = 0;

		}
	}

	template<class T>
	void ZQ_PCGSolver::_reflect(int n,const T* y, const T* l, const T* u, T* x,T* sigma)
	{
		T w;
		T* yptr = (T*)y;
		T* lptr = (T*)l;
		T* uptr = (T*)u;
		T* xptr = (T*)x;
		T* sigmaptr = (T*)sigma;
		memset(xptr,0,sizeof(T)*n);
		memset(sigmaptr,0,sizeof(T)*n);

		for(int i = 0;i < n;i++)
		{
			if((lptr[i] == 0) && (uptr[i]==1))
			{
				w = ZQ_MathBase::Rem<T>(fabs(yptr[i]),2.0);
				xptr[i] = __min(w,2-w);
				if(w <= 2-w)
					sigmaptr[i] = ZQ_MathBase::Sign(yptr[i]);
				else
					sigmaptr[i] = -ZQ_MathBase::Sign(yptr[i]);

			}
			else if((lptr[i] == 0) && (uptr[i] == ZQ_INF))
			{
				xptr[i] = fabs(yptr[i]);
				sigmaptr[i] = ZQ_MathBase::Sign(yptr[i]);

			}
			else if((lptr[i] == -ZQ_INF) && (uptr[i] == 1))
			{
				if(yptr[i] <= 1)
				{
					xptr[i] = yptr[i];
					sigmaptr[i] = 1.0;
				}
				else
				{
					xptr[i] = 2.0 - yptr[i];
					sigmaptr[i] = -1.0;
				}
			}
			else if((lptr[i] == -ZQ_INF) && (uptr[i] == ZQ_INF))
			{
				sigmaptr[i] = 1.0;
				xptr[i] = yptr[i];
			}
		}
		for(int i = 0;i < n;i++)
			sigmaptr[i] = sigmaptr[i] + 1.0 - fabs(sigmaptr[i]);
	}

	template<class T>
	void ZQ_PCGSolver::_shiftsc(int n, const T* xstart_in, const T* l_in, const T* u_in, T* typx, void (*fn)(const taucs_ccs_matrix* , const T* ,T*), const T* c_in, taucs_ccs_matrix* H,
		T* xstart_out, T* l_out, T* u_out, T* ds, taucs_ccs_matrix** DS, T* c_out)
	{
		if(strcmp(typeid(T).name(),"float") == 0)
			*DS = ZQ_TaucsBase::ZQ_taucs_ccs_create(n,n,n,TAUCS_SINGLE);
		else if(strcmp(typeid(T).name(),"double") == 0)
			*DS = ZQ_TaucsBase::ZQ_taucs_ccs_create(n,n,n,TAUCS_DOUBLE);
		else
		{
			*DS = 0;
			return;
		}

		T* xstart_inptr = (T*)xstart_in;
		T* l_inptr = (T*)l_in;
		T* u_inptr = (T*)u_in;
		T* typxptr = (T*)typx;
		T* c_inptr = (T*)c_in;
		T* xstart_outptr = (T*)xstart_out;
		T* l_outptr = (T*)l_out;
		T* u_outptr = (T*)u_out;
		T* dsptr = (T*)ds;
		T* c_outptr = (T*)c_out;


		for(int i = 0;i < n;i++)
		{
			dsptr[i] = 1;
			((T*)((*DS)->values.d))[i]  =dsptr[i];
			(*DS)->colptr[i] = i;
			(*DS)->rowind[i] = i;
		}
		(*DS)->colptr[n] = n;


		bool* flag1 = new bool[n];
		memset(flag1,0,sizeof(bool)*n);
		T* vshift = new T[n];
		memset(vshift,0,sizeof(T)*n);
		T* tmp = new T[n];

		int count = 0;
		for(int i = 0;i < n;i++)
		{
			if(l_inptr[i] == -ZQ_INF && u_inptr[i] == ZQ_INF)
			{
				flag1[i] = true;
				count ++;
			}
		}

		memcpy(xstart_out,xstart_in,sizeof(T)*n);
		memcpy(l_out,l_in,sizeof(T)*n);
		memcpy(u_out,u_in,sizeof(T)*n);
		if(c_in != 0)
			memcpy(c_out,c_in,sizeof(T)*n);
		if(count == n)
		{
			delete []flag1;
			flag1 = 0;
			return ;
		}

		for(int i = 0;i < n;i++)
		{
			if(flag1[i])
			{
				dsptr[i] = __max(fabs(typxptr[i]),dsptr[i]);
			}
			else if(l_inptr[i] == -ZQ_INF && u_inptr[i] < ZQ_INF)
			{
				xstart_outptr[i] = xstart_outptr[i] + 1.0 - u_outptr[i];
				vshift[i] = u_outptr[i] - 1.0;
				u_outptr[i] = 1.0;
			}
			else if(l_inptr[i] > -ZQ_INF && u_inptr[i] == ZQ_INF)
			{
				xstart_outptr[i] = xstart_outptr[i] - l_outptr[i];
				vshift[i] = l_outptr[i];
				l_outptr[i] = 0.0;
			}
			else if(l_inptr[i] > -ZQ_INF && u_inptr[i] < ZQ_INF)
			{
				xstart_outptr[i] = xstart_outptr[i] - l_outptr[i];
				vshift[i] = l_outptr[i];
				u_outptr[i] = u_outptr[i] - l_outptr[i];
				l_outptr[i] = 0.0;
				dsptr[i] = fabs(u_outptr[i]);
			}
		}
		T* w = new T[n];
		fn(H,vshift,w);

		for(int i = 0;i < n;i++)
			c_outptr[i] += w[i];
		for(int i = 0;i < n;i++)
			((T*)((*DS)->values.d))[i] = dsptr[i];

		if(c_in != 0)
		{
			memcpy(tmp,c_outptr,sizeof(T)*n);
			ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec((*DS),tmp,c_out);
		}

		for(int i = 0;i < n;i++)
		{
			if(dsptr[i] != 0)
			{
				u_outptr[i] /= dsptr[i];
				xstart_outptr[i] /= dsptr[i];
			}
			else
			{
				u_outptr[i] = ZQ_INF;
				xstart_outptr[i] = ZQ_INF;
			}
		}
		delete []flag1;
		delete []w;
		delete []vshift; vshift = 0;
		delete []tmp;
		flag1 = 0;
		w = 0;
		tmp = 0;
	}


	template<class T>
	void ZQ_PCGSolver::_unshsca(int n, T* x_in, T* l, T* u, taucs_ccs_matrix* DS, T* x_out)
	{
		ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec(DS,x_in,x_out);
		T* lptr = (T*)l;
		T* uptr = (T*)u;
		T* x_outptr = (T*)x_out;

		for(int i = 0;i < n;i++)
		{
			if(lptr[i] == -ZQ_INF && uptr[i] < ZQ_INF)
			{
				x_outptr[i] += uptr[i] - 1.0;
			}
			else if(lptr[i] > -ZQ_INF && uptr[i] == ZQ_INF)
			{
				x_outptr[i] += lptr[i];
			}
			else if(lptr[i] > -ZQ_INF && uptr[i] < ZQ_INF)
			{
				x_outptr[i] += lptr[i];
			}
		}
	}

	template<class T>
	void ZQ_PCGSolver::_perturb(const int n, const T* x_in , const T* l, const T* u, const double del, const T* y_in, const T* sigma,
		int& flag, T* x_out, T* y_out)
	{
		double delta = (del < 1e-12) ? 1e-12 : del;

		T* x_inptr = (T*)x_in;
		T* y_inptr = (T*)y_in;
		T* lptr = (T*)l;
		T* uptr = (T*)u;
		T* sigmaptr = (T*)sigma;
		T* x_outptr = (T*)x_out;
		T* y_outptr = (T*)y_out;
		memcpy(x_out,x_in,sizeof(T)*n);
		memcpy(y_out,y_in,sizeof(T)*n);

		flag = 0;
		if(y_in == 0)
		{
			for(int i = 0;i < n;i++)
			{
				if(x_outptr[i] < lptr[i] + delta)
				{
					x_outptr[i] += delta;
					flag = 1;
				}
				else if(x_outptr[i] > uptr[i] - delta)
				{
					x_outptr[i] -= delta;
					flag = 1;
				}
			}
		}
		else
		{
			for(int i = 0;i < n;i++)
			{
				if(x_outptr[i] < lptr[i] + delta)
				{
					x_outptr[i] += delta;
					y_outptr[i] += delta*sigmaptr[i];
					flag = 1;
				}
				else if(x_outptr[i] > uptr[i] - delta)
				{
					x_outptr[i] -= delta;
					y_outptr[i] -= delta*sigmaptr[i];
					flag = 1;
				}
			}
		}

	}

	template<class T>
	void ZQ_PCGSolver::_biqpbox(int n, const T* s, const T* c, const double strg, const T* x, const T* y, const T* sigma,
		const T* l, const T* u, const double oval, const double po, const double normg, const taucs_ccs_matrix* DS,const taucs_ccs_matrix* H,
		void (*fn)(const taucs_ccs_matrix*, const T*, T*),T* nx, T* nsig, double& alpha)
	{
		double fac = 1e8;
		int kbnd = 14;
		int kbnd2 = 4;
		double lsig = 1e-8;
		double usig = 0.9;
		int k = 0;
		double left = 0;
		double right = 0;
		double mid = 0;
		double sbnd = 0;
		double strgr = 0;
		double strgl = 0;
		double nstrg = 0;
		double strgmid = 0;
		double val = 0;
		double leftval = 0;
		double rightval = 0;

		T* sptr = (T*)s;
		T* cptr = (T*)c;
		T* xptr = (T*)x;
		T* yptr = (T*)y;
		T* sigmaptr = (T*)sigma;
		T* lptr = (T*)l;
		T* uptr = (T*)u;
		T* nxptr = (T*)nx;
		T* nsigptr = (T*)nsig;


		T* ss = new T[n];
		T* w = new T[n];
		T* ww = new T[n];
		T* righty = new T[n];
		T* rightx = new T[n];
		T* sigr = new T[n];
		T* rightw = new T[n];
		T* rightg = new T[n];
		T* leftw = new T[n];
		T* leftg = new T[n];
		T* leftx = new T[n];
		T* lefty = new T[n];
		T* sigl = new T[n];
		T* midy = new T[n];
		T* midx = new T[n];
		T* sigm = new T[n];
		T* midw = new T[n];
		T* midg = new T[n];
		T* tmp = new T[n];


		ZQ_MathBase::VecMul(n,sigmaptr,sptr,ss);
		ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec((taucs_ccs_matrix*)DS,ss,w);
		fn((taucs_ccs_matrix*)H,w,ww);
		ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec((taucs_ccs_matrix*)DS,ww,w);

		double term2 = 0.5*ZQ_MathBase::DotProduct(n,ss,w);

		if(term2 > 0)
			term2 = 0;

		if(po > 0)
		{
			alpha = 1;
			right = 1;
			sbnd = fac * normg;

		}
		else
		{
			alpha = 2;
			right = 2;
			sbnd = ZQ_INF;
		}
		for(int i = 0;i < n;i++)
			righty[i] = yptr[i] + alpha * sptr[i];

		_reflect(n,righty,l,u,rightx,sigr);
		ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec((taucs_ccs_matrix*)DS,rightx,w);
		fn((taucs_ccs_matrix*)H,w,ww);
		ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec((taucs_ccs_matrix*)DS,ww,rightw);
		ZQ_MathBase::VecPlus(n,rightw,c,rightg);

		for(int i = 0;i < n;i++)
			tmp[i] = 0.5*rightw[i]+cptr[i];
		val = ZQ_MathBase::DotProduct(n,rightx,tmp);

		ZQ_MathBase::VecMul(n,sigr,rightg,tmp);
		strgr = ZQ_MathBase::DotProduct(n,sptr,tmp);
		if(val < oval + lsig*(alpha*strg + (alpha*alpha)*term2))
		{
			memcpy(nx,rightx,sizeof(T)*n);
			memcpy(nsig,sigr,sizeof(T)*n);
			nstrg = strgr;
		}
		else
		{
			left = 0;
			memcpy(lefty,y,sizeof(T)*n);
			memcpy(leftx,x,sizeof(T)*n);
			memcpy(sigl,sigma,sizeof(T)*n);
			strgl = strg;
			//			memcpy(strgl,strg,sizeof(T)*n);
			for(k = 0;k < kbnd;k++)
			{
				mid = 0.5*(left+right);
				for(int i = 0;i < n;i++)
					midy[i] = yptr[i]+mid*sptr[i];
				_reflect(n,midy,l,u,midx,sigm);
				ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec((taucs_ccs_matrix*)DS,midx,w);
				fn((taucs_ccs_matrix*)H,w,ww);
				ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec((taucs_ccs_matrix*)DS,ww,midw);
				ZQ_MathBase::VecPlus(n,midw,c,midg);
				for(int i = 0;i < n;i++)
					tmp[i] = 0.5*midw[i]+cptr[i];
				val = ZQ_MathBase::DotProduct(n,midx,tmp);
				ZQ_MathBase::VecMul(n,sigm,midg,tmp);
				strgmid = ZQ_MathBase::DotProduct(n,s,tmp);
				if(val < oval + usig*(mid*strg + (mid*mid)*term2))
				{
					left = mid;
					strgl = strgmid;
				}
				else if(val > oval + lsig*(mid*strg + (mid*mid)*term2))
				{
					right = mid; 
					strgr = strgmid;
				}
				else
				{
					memcpy(nx,midx,sizeof(T)*n);
					memcpy(nsig,sigm,sizeof(T)*n);
					alpha = mid;
					nstrg = strgmid;
					break;
				}

			}

		}
		if(kbnd2 <= 0)
			return ;
		if(k == kbnd)
		{
			alpha = 0; 
			memcpy(nx,x,sizeof(T)*n); 
			memcpy(nsig,sigma,sizeof(T)*n);
			return ;
		}

		if(nstrg < 0)
		{

			left = alpha;
			for(int i = 0;i < n;i++)
				lefty[i] = yptr[i] + alpha*sptr[i];
			leftval = val;
			memcpy(leftx,nx,sizeof(T)*n);
			right = alpha + __min(0.5,sbnd) * alpha;
			for(int i = 0;i < n;i++)
				righty[i] = yptr[i] + right*sptr[i];
			_reflect(n,righty,l,u,rightx,sigr);
			ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec((taucs_ccs_matrix*)DS,rightx,w);
			fn((taucs_ccs_matrix*)H,w,ww);
			ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec((taucs_ccs_matrix*)DS,ww,rightw);
			ZQ_MathBase::VecPlus(n,rightw,c,rightg);
			for(int i = 0;i < n;i++)
				tmp[i] = 0.5*rightw[i]+cptr[i];
			rightval = ZQ_MathBase::DotProduct(n,rightx,tmp);
			ZQ_MathBase::VecMul(n,sigr,rightg,tmp);
			strgr = ZQ_MathBase::DotProduct(n,s,tmp);
			for(int k = 0;k < kbnd2;k++)
			{
				mid = 0.5*(left+right);
				for(int i = 0;i < n;i++)
					midy[i] = yptr[i]+mid*sptr[i];
				_reflect(n,midy,l,u,midx,sigm);
				ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec((taucs_ccs_matrix*)DS,midx,w);
				fn((taucs_ccs_matrix*)H,w,ww);
				ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec((taucs_ccs_matrix*)DS,ww,midw);
				ZQ_MathBase::VecPlus(n,midw,c,midg);
				for(int i = 0;i < n;i++)
					tmp[i] = 0.5*midw[i]+cptr[i];
				val = ZQ_MathBase::DotProduct(n,midx,tmp);
				ZQ_MathBase::VecMul(n,sigm,midg,tmp);
				strgmid = ZQ_MathBase::DotProduct(n,s,tmp);
				if((strgmid < 0) && (val < leftval))
				{
					left = mid; 
					leftval = val;
				}
				else
				{
					right = mid; 
					rightval = val;
				}
			}
		}
		else
		{
			right = alpha;
			for(int i = 0;i < n;i++)
				righty[i] = yptr[i] + alpha*sptr[i];
			rightval = val;
			memcpy(rightx,nx,sizeof(T)*n);
			left = alpha - __min( 0.5, sbnd) * alpha;
			for(int i = 0;i < n;i++)
				lefty[i] = yptr[i] + left*sptr[i];
			_reflect(n,lefty,l,u,leftx,sigl);
			ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec((taucs_ccs_matrix*)DS,leftx,w);
			fn((taucs_ccs_matrix*)H,w,ww);
			ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec((taucs_ccs_matrix*)DS,ww,leftw);
			ZQ_MathBase::VecPlus(n,leftw,c,leftg);
			for(int i = 0;i < n;i++)
				tmp[i] = 0.5*leftw[i]+cptr[i];
			leftval = ZQ_MathBase::DotProduct(n,leftx,tmp);
			ZQ_MathBase::VecMul(n,sigl,leftg,tmp);
			strgl = ZQ_MathBase::DotProduct(n,s,tmp);

			for(int ki = 0;ki < kbnd2;ki++)
			{
				mid = 0.5*(left+right);
				for(int i = 0;i < n;i++)
					midy[i] = yptr[i] + mid*sptr[i];
				_reflect(n,midy,l,u,midx,sigm);
				ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec((taucs_ccs_matrix*)DS,midx,w);
				fn((taucs_ccs_matrix*)H,w,ww);
				ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec((taucs_ccs_matrix*)DS,ww,midw);
				ZQ_MathBase::VecPlus(n,midw,c,midg);
				for(int i = 0;i < n;i++)
					tmp[i] = 0.5*midw[i]+cptr[i];
				val = ZQ_MathBase::DotProduct(n,midx,tmp);
				ZQ_MathBase::VecMul(n,sigm,midg,tmp);
				strgmid = ZQ_MathBase::DotProduct(n,s,tmp);
				if((strgmid > 0) && (val < rightval))
				{
					right = mid; 
					rightval = val;
				}
				else
				{
					left = mid; 
					leftval = val;
				}
			}

		}
		if( leftval <= rightval)
		{
			alpha = left; 
			nstrg = strgl;
		}
		else
		{
			alpha = right; nstrg = strgr;
		}
		for(int i = 0;i < n;i++)
			midy[i] = yptr[i] + alpha*sptr[i];
		_reflect(n,midy,l,u,nx,nsig);

		delete []ss;
		delete []w;
		delete []ww;
		delete []righty;
		delete []rightx;
		delete []sigr;
		delete []rightw;
		delete []rightg;
		delete []leftw;
		delete []leftg;
		delete []leftx;
		delete []lefty;
		delete []sigl;
		delete []midy;
		delete []midx;
		delete []sigm;
		delete []midw;
		delete []midg;
		delete []tmp;
	}

	template<class T>
	void ZQ_PCGSolver::_drqpbox(const int n,const T* D, const taucs_ccs_matrix* DS, const T* grad, const double delta, const T* g, const T* dv, void (*fn)(const taucs_ccs_matrix*, const T*,T*),
		void (*fnpc)(const taucs_ccs_matrix*, const T*, const T*, T* ), const double tol,const taucs_ccs_matrix* H, const int kmax, T* s, int& posdef, int& pcgit)
	{
		int tsize = 0;
		pcgit = 0;
		double tol2 = 1e-9;
		double tau = 1e-4;
		int Zdim = 1;

		T* gptr = (T*)g;
		T* dvptr = (T*)dv;
		T* gradptr = (T*)grad;
		T* sptr = (T*)s;
		T* Dptr = (T*)D;

		T* DM = new T[n];
		T* ddiag = new T[n];
		T* R = new T[n];
		T* v1 = new T[n];
		T* v2 = new T[n];
		T* Z1 = new T[n];
		T* Z2 = new T[n];
		T* w = new T[n];
		T* ww = new T[n];
		T* ss = new T[n];
		T* tmp = new T[n];



		ZQ_TaucsBase::ZQ_taucs_ccs_vec_time_matrix(D,(taucs_ccs_matrix*)DS,DM);
		for(int i = 0;i < n;i++)
			ddiag[i] = fabs(gptr[i])*dvptr[i];
		for(int i = 0;i < n;i++)
		{
			if(fabs(gptr[i]) + DM[i] < tau)
				ddiag[i] = tau;
		}
		fnpc((taucs_ccs_matrix*)H,DM,ddiag,R);
		_pcgr(DM,ddiag,grad,kmax,tol,ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec<T>,(taucs_ccs_matrix*)H,R, v1, posdef, pcgit);
		//	[v1,posdef,pcgit] = pcgr(DM,DG,grad,kmax,tol,mtxmpy,H,R,permR,'hessprecon',pcoptions,varargin{:});

		double normv1 = ZQ_MathBase::NormVector_L2(n,v1);
		if(normv1 == 0)
			memset(s,0,sizeof(T)*n);
		else
		{
			for(int i = 0;i < n;i++)
				v1[i] /= normv1;
			memcpy(Z1,v1,sizeof(T)*n);
			if(posdef < 1)
			{
				for(int i = 0;i < n;i++)
					tmp[i] = ZQ_MathBase::Sign(gradptr[i]);
				ZQ_MathBase::VecMul(n,D,tmp,v2);
				double normv2 = ZQ_MathBase::NormVector_L2(n,v2);
				for(int i = 0;i < n;i++)
					v2[i] /= normv2;
				double tmpdot = ZQ_MathBase::DotProduct(n,v1,v2);
				for(int i = 0;i < n;i++)
					v2[i] -= v1[i]*tmpdot;
				normv2 = ZQ_MathBase::NormVector_L2(n,v2);
				if(normv2 > tol2)
				{
					for(int i = 0;i < n;i++)
						v2[i] /= normv2;
					memcpy(Z2,v2,sizeof(T)*n);
					Zdim = 2;
				}
			}
			else
			{
				double normgrad = ZQ_MathBase::NormVector_L2(n,grad);
				for(int i = 0;i < n;i++)
					v2[i] = gradptr[i] / normgrad;
				double tmpdot = ZQ_MathBase::DotProduct(n,v1,v2);
				for(int i = 0;i < n;i++)
					v2[i] -= v1[i]*tmpdot;
				double normv2 = ZQ_MathBase::NormVector_L2(n,v2);
				if(normv2 > tol2)
				{
					for(int i = 0;i < n;i++)
						v2[i] /= normv2;
					memcpy(Z2,v2,sizeof(T)*n);
					Zdim = 2;
				}
			}
			if(Zdim == 1)
			{
				T MM[1];
				T SS[1];
				T RHS[1];

				ZQ_MathBase::VecMul(n,DM,Z1,w);
				fn((taucs_ccs_matrix*)H,w,ww);
				MM[0] = ZQ_MathBase::DotProduct(n,w,ww);
				ZQ_MathBase::VecMul(n,ddiag,Z1,tmp);
				MM[0] += ZQ_MathBase::DotProduct(n,Z1,tmp);
				RHS[0] = ZQ_MathBase::DotProduct(n,Z1,grad);
				_trust(1,RHS,MM,delta,SS);
				//[ss,qpval,po,fcnt,lambda] = trust(rhs,MM,delta);
				for(int i = 0;i < n;i++)
					ss[i] = Z1[i]*SS[0];
				for(int i = 0;i < n;i++)
					sptr[i] = fabs(Dptr[i])*ss[i];
			}
			/*Zdim = 2*/
			else
			{
				T MM[4];
				T SS[2];
				T RHS[2];
				T* tmp1 = new T[n];
				T* tmp2 = new T[n];
				ZQ_MathBase::VecMul(n,DM,Z1,tmp1);
				ZQ_MathBase::VecMul(n,DM,Z2,tmp2);
				MM[0] = MM[1] = MM[2] = MM[3] = 0;
				fn((taucs_ccs_matrix*)H,tmp1,tmp);
				MM[0] += ZQ_MathBase::DotProduct(n,tmp1,tmp);
				MM[2] += ZQ_MathBase::DotProduct(n,tmp2,tmp);
				fn((taucs_ccs_matrix*)H,tmp2,tmp);
				MM[1] += ZQ_MathBase::DotProduct(n,tmp1,tmp);
				MM[3] += ZQ_MathBase::DotProduct(n,tmp2,tmp);
				ZQ_MathBase::VecMul(n,ddiag,Z1,tmp1);
				MM[0] += ZQ_MathBase::DotProduct(n,Z1,tmp1);
				MM[2] += ZQ_MathBase::DotProduct(n,Z2,tmp1);
				ZQ_MathBase::VecMul(n,ddiag,Z2,tmp2);
				MM[1] += ZQ_MathBase::DotProduct(n,Z1,tmp2);
				MM[3] += ZQ_MathBase::DotProduct(n,Z2,tmp2);

				RHS[0] = ZQ_MathBase::DotProduct(n,Z1,grad);
				RHS[1] = ZQ_MathBase::DotProduct(n,Z2,grad);
				_trust(2,RHS,MM,delta,SS);
				//[ss,qpval,po,fcnt,lambda] = trust(rhs,MM,delta);

				for(int i = 0;i < n;i++)
					ss[i] = Z1[i]*SS[0] + Z2[i]*SS[1];
				for(int i = 0;i < n;i++)
					sptr[i] = fabs(Dptr[i])*ss[i];
				delete []tmp1;
				delete []tmp2;

			}
		}

		delete []DM;
		delete []ddiag;
		delete []R;
		delete []v1;
		delete []v2;
		delete []Z1;
		delete []Z2;
		delete []w;
		delete []ww;
		delete []ss;
		delete []tmp;
	}

	/*only solve n = 1,2*/
	template<class T>
	void ZQ_PCGSolver::_trust(int n, const T* g, const T* H, const double delta, T* s)
	{
		T *Hptr = new T[n*n];
		T *gptr = new T[n];
		T *sptr = new T[n];
		T* coeff = new T[n];
		T* Vptr = new T[n*n];
		T* D = new T[n];
		T* alpha = new T[n];
		T* lam = new T[n];
		T* w = new T[n];
		T* tmp = new T[n];

		for(int i = 0;i < n*n;i++)
			Hptr[i] = H[i];
		for(int i = 0;i < n;i++)
			gptr[i] = g[i];

		double tol = 1e-12;
		double tol2 = 1e-8;
		int key = 0;
		int itbnd = 50;
		int lamda = 0;
		int posdef = 0;
		double laminit = 0;
		double val;

		memset(coeff,0,sizeof(T)*n);

		_eig(n,Hptr,Vptr,D);
		int count = 0;
		T mineig = 0;
		int jmin = 0;
		ZQ_MathBase::FindMin(n,D,mineig,jmin);
		ZQ_MathBase::MatrixMul(g,Vptr,1,n,n,alpha);
		for(int i = 0;i < n;i++)
			alpha[i] = -alpha[i];
		double sig = ZQ_MathBase::Sign(alpha[jmin]) + ((alpha[jmin] == 0)?1:0);

		if(mineig > 0)
		{
			ZQ_MathBase::VecDiv(n,alpha,D,coeff);
			lamda = 0;
			ZQ_MathBase::MatrixMul(Vptr,coeff,n,n,1,sptr);
			posdef = 1;
			double norms = ZQ_MathBase::NormVector_L2(n,sptr);
			if(norms < 1.2*delta)
				key = 1;
			else
				laminit = 0;
		}
		else
		{
			laminit = -mineig;
			posdef = 0;
		}

		if(key == 0)
		{
			if(_seceqn(n,laminit,D,alpha,delta) > 0)
			{
				double b = 0;
				double c = 0;
				_rfzero(n,laminit,itbnd,D,alpha,delta,tol,b,c,count);
				double vval = fabs(_seceqn(n,b,D,alpha,delta));
				if( vval <= tol2)
				{
					lamda = b; 
					key = 2;
					for(int i = 0;i < n;i++)
						lam[i] = lamda;
					ZQ_MathBase::VecPlus(n,D,lam,w);
					for(int i = 0;i < n;i++)
					{
						if(w[i] != 0)
							coeff[i] = alpha[i] / w[i];
						else
						{
							if(alpha[i] == 0)
								coeff[i] = 0;
							else
								coeff[i] = ZQ_INF;
						}
					}
					/*ignore this matlab code*/
					//coeff(isnan(coeff))=0;
					ZQ_MathBase::MatrixMul(Vptr,coeff,n,n,1,sptr);
					double norms = ZQ_MathBase::NormVector_L2(n,sptr);

					if( (norms > 1.2*delta) || (norms < 0.8*delta))
					{
						key = 5; 
						lamda = -mineig;
					}
				}
				else
				{
					lamda = -mineig; 
					key = 3;
				}
			}
			else
			{
				lamda = -mineig; 
				key = 4;
			}
			for(int i = 0;i < n;i++)
				lam[i] = lamda;
			if (key > 2) 
			{
				for(int i = 0;i < n;i++)
				{
					if(fabs(D[i]+lam[i]) < 1e-12)
						alpha[i] = 0;
				}
			}
			ZQ_MathBase::VecPlus(n,D,lam,w);

			for(int i = 0;i < n;i++)
			{
				if(w[i] != 0)
					coeff[i] = alpha[i] / w[i];
				else
				{
					if(alpha[i] == 0)
						coeff[i] = 0;
					else
						coeff[i] = ZQ_INF;
				}
			}
			/*ignore this matlab code*/
			//coeff(isnan(coeff))=0;
			ZQ_MathBase::MatrixMul(Vptr,coeff,n,n,1,sptr);
			double norms = ZQ_MathBase::NormVector_L2(n,sptr);

			if( (key > 2) && (norms < 0.8*delta))
			{
				double beta = sqrt(delta*delta - norms*norms);
				for(int i = 0;i < n;i++)
					sptr[i] += beta*sig*Vptr[i*n+jmin];	
			}
			if( (key > 2) && (norms > 1.2*delta))
			{
				double b = 0;
				double c = 0;
				_rfzero(n,laminit,itbnd,D,alpha,delta,tol,b,c,count);
				lamda = b; 
				for(int i = 0;i < n;i++)
					lam[i] = lamda;
				ZQ_MathBase::VecPlus(n,D,lam,w);
				for(int i = 0;i < n;i++)
				{
					if(w[i] != 0)
						coeff[i] = alpha[i] / w[i];
					else
					{
						if(alpha[i] == 0)
							coeff[i] = 0;
						else
							coeff[i] = ZQ_INF;
					}
				}
				/*ignore this matlab code*/
				//coeff(isnan(coeff))=0;
				ZQ_MathBase::MatrixMul(Vptr,coeff,n,n,1,sptr);
				double norms = ZQ_MathBase::NormVector_L2(n,sptr);

			}
		}

		ZQ_MathBase::MatrixMul(Hptr,sptr,n,n,1,tmp);
		val = ZQ_MathBase::DotProduct(n,sptr,tmp);
		val *= 0.5;
		val += ZQ_MathBase::DotProduct(n,gptr,sptr);


		for(int i = 0;i < n;i++)
			s[i] = sptr[i];



		delete []Hptr;
		delete []gptr;
		delete []sptr;
		delete []coeff;
		delete []Vptr;
		delete []D;
		delete []alpha;
		delete []lam;
		delete []w;
		delete []tmp;

	}

	/*only solve n = 1,2*/
	template<class T>
	void ZQ_PCGSolver::_eig(int n, const T* H, T* V, T* D)
	{
		T* Dptr = (T*)D;
		T* Hptr = (T*)H;
		T* Vptr = (T*)V;
		if(n == 1)
		{
			Dptr[0] = Hptr[0];
			Vptr[0] = 1;
		}
		else if(n==2)
		{
			double a = Hptr[0];
			double b = Hptr[1];
			double c = Hptr[2];
			double d = Hptr[3];
			double lamda1 = 0.5*(a+d-sqrt((a-d)*(a-d)+4*b*c));
			double lamda2 = 0.5*(a+d+sqrt((a-d)*(a-d)+4*b*c));
			double v1[2],v2[2];
			if(b != 0)
			{
				v1[0] = 1;
				v1[1] = (lamda1-a)/b;
				double nrm = sqrt(v1[0]*v1[0] + v1[1]*v1[1]);
				v1[0] /= nrm;
				v1[1] /= nrm;
			}
			else
			{
				v1[0] = 0;
				v1[1] = 1;
			}
			if(c != 0)
			{
				v2[0] = (lamda2-d)/c;
				v2[1] = 1;
				double nrm = sqrt(v2[0]*v2[0]+v2[1]*v2[1]);
				v2[0] /= nrm;
				v2[1] /= nrm;
			}
			else
			{
				v2[0] = 1;
				v2[1] = 0;
			}
			Dptr[0] = lamda1;
			Dptr[1] = lamda2;
			Vptr[0] = v1[0];
			Vptr[2] = v1[1];
			Vptr[1] = v2[0];
			Vptr[3] = v2[1];
		}
	}

	template<class T>
	double ZQ_PCGSolver::_seceqn(int n,const double lamda,const T* eigval,const T* alpha,const double delta)
	{
		T* M = new T[n];
		T* MC = new T[n];
		T* MM = new T[n];
		for(int i = 0;i < n;i++)
			M[i] = eigval[i] + lamda;
		memcpy(MC,M,sizeof(T)*n);
		memcpy(MM,alpha,sizeof(T)*n);
		for(int i = 0;i < n;i++)
		{
			if(M[i] != 0)
				M[i] = MM[i] / M[i];
			else
				M[i] = ZQ_INF;
		}
		for(int i = 0;i < n;i++)
			M[i] = M[i]*M[i];

		double val = 0;
		for(int i = 0;i < n;i++)
			val += M[i];
		val = sqrt(1.0/val);
		val = 1.0/delta - val;
		delete []M;
		delete []MC;
		delete []MM;
		return val;
	}

	template<class T>
	void ZQ_PCGSolver::_rfzero(int n,const double lamda,const int itbnd, const T* eigval,const T* alpha,const double delta,const double tol,
		double& b, double& c, int& count)
	{
		int itfun = 0;
		double x = lamda;
		double dx = 0;
		if(x != 0)
			dx = fabs(x)/2.0;
		else
			dx = 0.5;
		double a = x;
		c = a;
		double fa = _seceqn(n,a,eigval,alpha,delta);
		itfun++;
		b = x+1;
		double fb = _seceqn(n,b,eigval,alpha,delta);
		itfun++;

		while((fa>0) == (fb>0))
		{
			dx *= 2;
			if ((fa > 0) != (fb > 0))
				break;
			b = x + dx;  
			fb = _seceqn(n,b,eigval,alpha,delta);
			itfun++;

			if(itfun > itbnd)
				break;
		}

		double fc = fb;
		double d,e;
		double m,s,p,q,r;
		double toler;
		while (fb != 0)
		{
			if((fb > 0) == (fc > 0))
			{
				c = a;  
				fc = fa;
				d = b - a;  
				e = d;
			}
			if(fabs(fc) < fabs(fb))
			{
				a = b;    
				b = c;    
				c = a;
				fa = fb;  
				fb = fc;  
				fc = fa;
			}
			if(itfun > itbnd) 
				break;
			m = 0.5*(c - b);
			toler = 2.0*tol*__max(fabs(b),1.0);
			if((fabs(m) <= toler) || (fb == 0.0))
				break;
			if((fabs(e) < toler) || (fabs(fa) <= fabs(fb)))
			{
				//% Bisection
				d = m;  
				e = m;
			}
			else
			{
				//% Interpolation
				s = fb/fa;
				if((a == c))
				{
					//% Linear interpolation
					p = 2.0*m*s;
					q = 1.0 - s;
				}
				else
				{
					//% Inverse quadratic interpolation
					q = fa/fc;
					r = fb/fc;
					p = s*(2.0*m*q*(q - r) - (b - a)*(r - 1.0));
					q = (q - 1.0)*(r - 1.0)*(s - 1.0);
				}
				if(p > 0)
					q = -q; 
				else 
					p = -p;
				//% Is interpolated point acceptable
				if( (2.0*p < 3.0*m*q - fabs(toler*q)) && (p < fabs(0.5*e*q)))
				{
					e = d;  
					d = p/q;
				}
				else
				{
					d = m;  
					e = m;
				}
			}
			//% Next point
			a = b;
			fa = fb;
			if( fabs(d) > toler)
				b = b + d;
			else
			{
				if (b > c) 
					b = b - toler;
				else
					b = b + toler;
			}
			fb = _seceqn(n,b,eigval,alpha,delta);
			itfun++;
		}
		count = itfun;
	}

	template<class T>
	void ZQ_PCGSolver::_preproj(const T* r,const T* DR, T* w, int n)
	{
		T* wptr = (T*)w;
		T* rptr = (T*)r;
		T* DRptr = (T*)DR;
		for(int i = 0;i < n;i++)
			wptr[i] = rptr[i] /(DRptr[i]*DRptr[i]);
	}

	template<class T>
	void ZQ_PCGSolver::_pcgr(const T* DM,const T* DG,const T* g,const int kmax,const double tol,void (*fn)(const taucs_ccs_matrix*,const T*, T*),
		taucs_ccs_matrix* H,const T* R,T* p,int& posdef, int& k)
	{
		int n = H->m;

		T* DMptr = (T*)DM;
		T* DGptr = (T*)DG;
		T* gptr = (T*)g;
		T* Rptr = (T*)R;
		T* pptr = (T*)p;

		T* r = new T[n];
		T* z = new T[n];
		T* d = new T[n];
		T* w = new T[n];
		T* tmp = new T[n];

		memset(p,0,sizeof(T)*n);
		for(int i = 0;i < n;i++)
			r[i] = -gptr[i];
		_preproj(r,R,z,n);
		double znorm = ZQ_MathBase::NormVector_L2(n,z);
		double stoptol = tol*znorm;
		double inner2 = 0;
		double inner1 = ZQ_MathBase::DotProduct(n,r,z);
		posdef = 1;
		double alpha,beta,denom;
		for(k = 1;k <= kmax;k++)
		{
			if(k==1)
				memcpy(d,z,sizeof(T)*n);
			else
			{
				beta = inner1/inner2;
				for(int i = 0;i < n;i++)
					d[i] = z[i] + beta*d[i];
			}
			ZQ_MathBase::VecMul(n,DM,d,tmp);
			fn((taucs_ccs_matrix*)H,tmp,w);
			for(int i = 0;i < n;i++)
				tmp[i] = DMptr[i]*w[i]+DGptr[i]*d[i];
			denom = ZQ_MathBase::DotProduct(n,d,tmp);
			if(denom <= 0)
			{
				double normd = ZQ_MathBase::NormVector_L2(n,d);
				if(normd == 0)
					memcpy(p,d,sizeof(T)*n);
				else   
				{
					for(int i = 0;i < n;i++)
						pptr[i] = d[i] / normd;
				}
				posdef = 0;
				break;
			}
			else
			{
				alpha = inner1/denom;
				for(int i = 0;i < n;i++)
				{
					pptr[i] += alpha*d[i];
					r[i] -= alpha*tmp[i];
				}
			}
			_preproj(r,R,z,n);
			if(ZQ_MathBase::NormVector_L2(n,z) <= stoptol) 
				break; 

			inner2 = inner1;
			inner1 = ZQ_MathBase::DotProduct(n,r,z);
		}
		delete []r;
		delete []z;
		delete []d;
		delete []w;
		delete []tmp;
	}

}

#endif