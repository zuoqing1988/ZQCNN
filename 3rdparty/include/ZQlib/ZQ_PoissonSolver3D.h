#ifndef _ZQ_POISSON_SOLVER_3D_H_
#define _ZQ_POISSON_SOLVER_3D_H_
#pragma once

#include "ZQ_PCGSolver.h"
#include "ZQ_SparseMatrix.h"
#include <typeinfo>

namespace ZQ
{
	namespace ZQ_PoissonSolver3D
	{

		template<class T>
		void ComputeDivergence(int width, int height, int depth, const T* mac_u, const T* mac_v, const T* mac_w, T* b);

		template<class T>
		void AdjustOpen(int width, int height, int depth, T* mac_u, T* mac_v, T* mac_w, const T* p);

		template<class T>
		void AdjustOpen(int width, int height, int depth, const T* occupy, T* mac_u, T* mac_v, T* mac_w, const T* p);

		template<class T>
		void AdjustClosed(int width, int height, int depth, T* mac_u, T* mac_v, T* mac_w, const T* p);

		template<class T>
		void AdjustClosed(int width, int height, int depth, const T* occupy, T* mac_u, T* mac_v, T* mac_w, const T* p);
		
		template<class T>
		void RegularGridtoMAC(int width, int height, int depth, const T* u, const T* v, const T* w, T* mac_u, T* mac_v, T* mac_w, bool use_peroid_coord = false);

		template<class T>
		void MACtoRegularGrid(int width, int height, int depth, const T* mac_u, const T* mac_v, const T* mac_w, T* u, T* v, T* w);

		template<class T>
		int BuildOpenPoisson(int width, int height, int depth, taucs_ccs_matrix** A, bool display = false);

		template<class T>
		int BuildClosedPoisson(int width, int height, int depth, taucs_ccs_matrix** A, bool display = false);

		template<class T>
		void SolveOpenPoisson(T* u, T* v, T* w, int width, int height, int depth, const taucs_ccs_matrix* A, int maxiter, bool display = false);

		template<class T>
		void SolveOpenPoisson_MACGrid(T* mac_u, T* mac_v, T* mac_w, int width, int height, int depth, const taucs_ccs_matrix* A, int maxiter, bool display = false);

		template<class T>
		void SolveClosedPoisson(T* u, T* v, T* w, int width, int height, int depth, const taucs_ccs_matrix* A, int maxiter, bool display = false);

		template<class T>
		void SolveClosedPoisson_MACGrid(T* mac_u, T* mac_v, T* mac_w, int width, int height, int depth, const taucs_ccs_matrix* A, int maxiter, bool display = false);

		template<class T>
		void SolveOpenPoissonSOR(T* u, T* v, T* w, int width, int height, int depth, int nSORIterations, bool display = false);

		template<class T>
		void SolveOpenPoissonSOR_MACGrid(T* mac_u, T* mac_v, T* mac_w, int width, int height, int depth, int nSORIterations, bool display = false);

		template<class T>
		void SolveOpenPoissonRedBlack(T* u, T* v, T* w, int width, int height, int depth, int nIterations, bool display = false);

		template<class T>
		void SolveOpenPoissonRedBlack_MACGrid(T* mac_u, T* mac_v, T* mac_w, int width, int height, int depth, int nIterations, bool display = false);

		//for scene with obstacles
		template<class T>
		int BuildOpenPoisson(int width, int height, int depth, const bool* occupy, taucs_ccs_matrix** A, bool display = false);

		template<class T>
		int BuildClosedPoisson(int width, int height, int depth, const bool* occupy, taucs_ccs_matrix** A, bool display = false);

		template<class T>
		void SolveOpenPoisson_MACGrid(T* mac_u, T* mac_v, T* mac_w, int width, int height, int depth, const bool* occupy, const taucs_ccs_matrix* A, int maxiter, bool display = false);

		template<class T>
		void SolveClosedPoisson_MACGrid(T* mac_u, T* mac_v, T* mac_w, int width, int height, int depth, const bool* occupy, const taucs_ccs_matrix* A, int maxiter, bool display = false);

		template<class T>
		void SolveOpenPoissonSOR_MACGrid(T* mac_u, T* mac_v, T* mac_w, int width, int height, int depth, const bool* occupy, int nSORIterations, bool display = false);

		template<class T>
		void SolveOpenPoissonRedBlack_MACGrid(T* mac_u, T* mac_v, T* mac_w, int width, int height, int depth, const bool* occupy, int nIterations, bool display = false);

		/***************************************************************************************************/
		/***************************************************************************************************/
		/**************************             definitions             ************************************/
		/***************************************************************************************************/
		/***************************************************************************************************/

		template<class T>
		void ComputeDivergence(int width, int height, int depth, const T* mac_u, const T* mac_v, const T* mac_w, T* b)
		{
			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 0;i < width;i++)
					{
						b[k*height*width+j*width+i] = mac_u[k*height*(width+1)+j*(width+1)+(i+1)] - mac_u[k*height*(width+1)+j*(width+1)+i] 
						+ mac_v[k*(height+1)*width+(j+1)*width+i] - mac_v[k*(height+1)*width+j*width+i]
						+ mac_w[(k+1)*height*width+j*width+i] - mac_w[k*height*width+j*width+i];
					}
				}
			}
		}

		template<class T>
		void AdjustOpen(int width, int height, int depth, T* mac_u, T* mac_v, T* mac_w, const T* p)
		{
			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 1;i < width;i++)
					{
						mac_u[k*height*(width+1)+j*(width+1)+i] -= p[k*height*width+j*width+i] - p[k*height*width+j*width+(i-1)];
					}
					mac_u[k*height*(width+1)+j*(width+1)+0] -= p[k*height*width+j*width+0] - 0;
					mac_u[k*height*(width+1)+j*(width+1)+width] -= 0 - p[k*height*width+j*width+width-1];
				}
			}


			for(int k = 0;k < depth;k++)
			{
				for(int i = 0;i < width;i++)
				{
					for(int j = 1;j < height;j++)
					{
						mac_v[k*(height+1)*width+j*width+i] -= p[k*height*width+j*width+i] - p[k*height*width+(j-1)*width+i];
					}
					mac_v[k*(height+1)*width+0*width+i] -= p[k*height*width+0*width+i] - 0;
					mac_v[k*(height+1)*width+height*width+i] -= 0 - p[k*height*width+(height-1)*width+i];
				}
			}

			for(int j = 0;j < height;j++)
			{
				for(int i = 0;i < width;i++)
				{
					for(int k = 1;k < depth;k++)
					{
						mac_w[k*height*width+j*width+i] -= p[k*height*width+j*width+i] - p[(k-1)*height*width+j*width+i];
					}
					mac_w[0*height*width+j*width+i] -= p[0*height*width+j*width+i] - 0;
					mac_w[depth*height*width+j*width+i] -= 0 - p[(depth-1)*height*width+j*width+i];
				}
			}
		}

		template<class T>
		void AdjustOpen(int width, int height, int depth, const bool* occupy, T* mac_u, T* mac_v, T* mac_w, const T* p)
		{
			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 1;i < width;i++)
					{
						if(!occupy[k*height*width+j*width+i] && !occupy[k*height*width+j*width+i-1])
							mac_u[k*height*(width+1)+j*(width+1)+i] -= p[k*height*width+j*width+i] - p[k*height*width+j*width+(i-1)];
					}
					if(!occupy[k*height*width+j*width+0])
						mac_u[k*height*(width+1)+j*(width+1)+0] -= p[k*height*width+j*width+0] - 0;
					if(!occupy[k*height*width+j*width+width-1])
						mac_u[k*height*(width+1)+j*(width+1)+width] -= 0 - p[k*height*width+j*width+width-1];
				}
			}


			for(int k = 0;k < depth;k++)
			{
				for(int i = 0;i < width;i++)
				{
					for(int j = 1;j < height;j++)
					{
						if(!occupy[k*height*width+j*width+i] && !occupy[k*height*width+(j-1)*width+i])
							mac_v[k*(height+1)*width+j*width+i] -= p[k*height*width+j*width+i] - p[k*height*width+(j-1)*width+i];
					}
					if(!occupy[k*height*width+0*width+i])
						mac_v[k*(height+1)*width+0*width+i] -= p[k*height*width+0*width+i] - 0;
					if(!occupy[k*height*width+(height-1)*width+i])
						mac_v[k*(height+1)*width+height*width+i] -= 0 - p[k*height*width+(height-1)*width+i];
				}
			}

			for(int j = 0;j < height;j++)
			{
				for(int i = 0;i < width;i++)
				{
					for(int k = 1;k < depth;k++)
					{
						if(!occupy[k*height*width+j*width+i] && !occupy[(k-1)*height*width+j*width+i])
							mac_w[k*height*width+j*width+i] -= p[k*height*width+j*width+i] - p[(k-1)*height*width+j*width+i];
					}
					if(!occupy[0*height*width+j*width+i])
						mac_w[0*height*width+j*width+i] -= p[0*height*width+j*width+i] - 0;
					if(!occupy[(depth-1)*height*width+j*width+i])
						mac_w[depth*height*width+j*width+i] -= 0 - p[(depth-1)*height*width+j*width+i];
				}
			}
		}

		template<class T>
		void AdjustClosed(int width, int height, int depth, T* mac_u, T* mac_v, T* mac_w, const T* p)
		{
			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 1;i < width;i++)
					{
						mac_u[k*height*(width+1)+j*(width+1)+i] -= p[k*height*width+j*width+i] - p[k*height*width+j*width+(i-1)];
					}
				}
			}


			for(int k = 0;k < depth;k++)
			{
				for(int i = 0;i < width;i++)
				{
					for(int j = 1;j < height;j++)
					{
						mac_v[k*(height+1)*width+j*width+i] -= p[k*height*width+j*width+i] - p[k*height*width+(j-1)*width+i];
					}
				}
			}

			for(int j = 0;j < height;j++)
			{
				for(int i = 0;i < width;i++)
				{
					for(int k = 1;k < depth;k++)
					{
						mac_w[k*height*width+j*width+i] -= p[k*height*width+j*width+i] - p[(k-1)*height*width+j*width+i];
					}
				}
			}
		}

		template<class T>
		void AdjustClosed(int width, int height, int depth, const bool* occupy, T* mac_u, T* mac_v, T* mac_w, const T* p)
		{
			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 1;i < width;i++)
					{
						if(!occupy[k*height*width+j*width+i] && !occupy[k*height*width+j*width+i-1])
							mac_u[k*height*(width+1)+j*(width+1)+i] -= p[k*height*width+j*width+i] - p[k*height*width+j*width+(i-1)];
					}
				}
			}


			for(int k = 0;k < depth;k++)
			{
				for(int i = 0;i < width;i++)
				{
					for(int j = 1;j < height;j++)
					{
						if(!occupy[k*height*width+j*width+i] && !occupy[k*height*width+(j-1)*width+i])
							mac_v[k*(height+1)*width+j*width+i] -= p[k*height*width+j*width+i] - p[k*height*width+(j-1)*width+i];
					}
				}
			}

			for(int j = 0;j < height;j++)
			{
				for(int i = 0;i < width;i++)
				{
					for(int k = 1;k < depth;k++)
					{
						if(!occupy[k*height*width+j*width+i] && !occupy[(k-1)*height*width+j*width+i])
							mac_w[k*height*width+j*width+i] -= p[k*height*width+j*width+i] - p[(k-1)*height*width+j*width+i];
					}
				}
			}
		}

		template<class T>
		void RegularGridtoMAC(int width, int height, int depth, const T* u, const T* v, const T* w, T* mac_u, T* mac_v, T* mac_w, bool use_peroid_coord /* = false*/)
		{
			if(!use_peroid_coord)
			{
				for(int k = 0 ;k < depth;k++)
				{
					for(int j = 0;j < height;j++)
					{
						for(int i = 1;i < width;i++)
						{
							mac_u[k*height*(width+1)+j*(width+1)+i] = 0.5*(u[k*height*width+j*width+(i-1)]+u[k*height*width+j*width+i]);
						}
						mac_u[k*height*(width+1)+j*(width+1)+0] = u[k*height*width+j*width+0];
						mac_u[k*height*(width+1)+j*(width+1)+width] = u[k*height*width+j*width+width-1];
					}
				}

				for(int k = 0;k < depth;k++)
				{
					for(int i = 0;i < width;i++)
					{
						for(int j = 1;j < height;j++)
						{
							mac_v[k*(height+1)*width+j*width+i] = 0.5*(v[k*height*width+(j-1)*width+i]+v[k*height*width+j*width+i]);
						}
						mac_v[k*(height+1)*width+0*width+i] = v[k*height*width+0*width+i];
						mac_v[k*(height+1)*width+height*width+i] = v[k*height*width+(height-1)*width+i];
					}
				}

				for(int j = 0;j < height;j++)
				{
					for(int i = 0;i < width;i++)
					{
						for(int k = 1;k < depth;k++)
						{
							mac_w[k*height*width+j*width+i] = 0.5*(w[(k-1)*height*width+j*width+i]+w[k*height*width+j*width+i]);
						}
						mac_w[0*height*width+j*width+i] = w[0*height*width+j*width+i];
						mac_w[depth*height*width+j*width+i] = w[(depth-1)*height*width+j*width+i];
					}
				}
			}
			else
			{
				for(int k = 0;k < depth;k++)
				{
					for(int j = 0;j < height;j++)
					{
						for(int i = 1;i < width;i++)
						{
							mac_u[k*height*(width+1)+j*(width+1)+i] = 0.5*(u[k*height*width+j*width+(i-1)]+u[k*height*width+j*width+i]);
						}
						mac_u[k*height*(width+1)+j*(width+1)+0] = 0.5*(u[k*height*width+j*width+0] + u[k*height*width+j*width+width-1]);
						mac_u[k*height*(width+1)+j*(width+1)+width] = 0.5*(u[k*height*width+j*width+0] + u[k*height*width+j*width+width-1]);
					}
				}

				for(int k = 0;k < depth;k++)
				{
					for(int i = 0;i < width;i++)
					{
						for(int j = 1;j < height;j++)
						{
							mac_v[k*(height+1)*width+j*width+i] = 0.5*(v[k*height*width+(j-1)*width+i]+v[k*height*width+j*width+i]);
						}
						mac_v[k*(height+1)*width+0*width+i] = 0.5*(v[k*height*width+0*width+i] + v[k*height*width+(height-1)*width+i]);
						mac_v[k*(height+1)*width+height*width+i] = 0.5*(v[k*height*width+0*width+i] + v[k*height*width+(height-1)*width+i]);
					}
				}

				for(int j = 0;j < height;j++)
				{
					for(int i = 0;i < width;i++)
					{
						for(int k = 1;k < depth;k++)
						{
							mac_w[k*height*width+j*width+i] = 0.5*(w[(k-1)*height*width+j*width+i]+w[k*height*width+j*width+i]);
						}
						mac_w[0*height*width+j*width+i] = 0.5*(w[0*height*width+j*width+i] + w[(depth-1)*height*width+j*width+i]);
						mac_w[depth*height*width+j*width+i] = 0.5*(w[0*height*width+j*width+i] + w[(depth-1)*height*width+j*width+i]);
					}
				}
			}
		}

		template<class T>
		void MACtoRegularGrid(int width, int height, int depth, const T* mac_u, const T* mac_v, const T* mac_w, T* u, T* v, T* w)
		{
			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 0;i < width;i++)
					{
						u[k*height*width+j*width+i] = 0.5*(mac_u[k*height*(width+1)+j*(width+1)+i]+mac_u[k*height*(width+1)+j*(width+1)+i+1]);
						v[k*height*width+j*width+i] = 0.5*(mac_v[k*(height+1)*width+j*width+i]+mac_v[k*(height+1)*width+(j+1)*width+i]);
						w[k*height*width+j*width+i] = 0.5*(mac_w[k*height*width+j*width+i]+mac_w[(k+1)*height*width+j*width+i]);
					}
				}
			}
		}

		template<class T>
		int BuildOpenPoisson(int width, int height, int depth, taucs_ccs_matrix** A, bool display)
		{
			int flag;
			if(strcmp(typeid(T).name(),"float") == 0)
				flag = TAUCS_SINGLE;
			else if(strcmp(typeid(T).name(),"double") == 0)
				flag = TAUCS_DOUBLE;
			else 
				return 0;

			int dim = width*height*depth;

			ZQ_SparseMatrix<T> mat(dim,dim);

			int KSLICE = height*width;
			int JSLICE = width;
			int ISLICE = 1;

			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 0;i < width;i++)
					{
						int offset = k*height*width+j*width+i;
						int row_id = offset;

						std::vector<int> indices;
						if(i > 0)
							indices.push_back(offset-ISLICE);
						if(i < width-1)
							indices.push_back(offset+ISLICE);
						if(j > 0)
							indices.push_back(offset-JSLICE);
						if(j < height-1)
							indices.push_back(offset+JSLICE);
						if(k > 0)
							indices.push_back(offset-KSLICE);
						if(k < depth-1)
							indices.push_back(offset+KSLICE);


						float count = 6.0;
						for(int cc = 0;cc < indices.size();cc++)
						{
							mat.SetValue(row_id,indices[cc], 1);
						}
						mat.SetValue(row_id,row_id, -count);
					}
				}
			}

			*A = mat.ExportCCS(flag);

			if((*A) == 0)
			{
				if(display)
				{
					printf("create taucs_ccs_matrix fail\n");
				}
				return 0;
			}

			if(display)
			{
				int m = (*A)->m;
				int n = (*A)->n;
				int nnz = (*A)->colptr[n];
				printf("dim: %d x %d, nnz: %d\n",m,n,nnz);
			}

			return (dim);
		}

		template<class T>
		int BuildClosedPoisson(int width, int height, int depth, taucs_ccs_matrix** A, bool display)
		{
			int flag;
			if(strcmp(typeid(T).name(),"float") == 0)
				flag = TAUCS_SINGLE;
			else if(strcmp(typeid(T).name(),"double") == 0)
				flag = TAUCS_DOUBLE;
			else 
				return 0;

			int dim = width*height*depth;
			int row_num = dim+1;
			int col_num = dim;
			ZQ_SparseMatrix<T> mat(row_num,col_num);

			int KSLICE = height*width;
			int JSLICE = width;
			int ISLICE = 1;

			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 0;i < width;i++)
					{
						int offset = k*height*width+j*width+i;
						int row_id = offset;

						std::vector<int> indices;
						if(i > 0)
							indices.push_back(offset-ISLICE);
						if(i < width-1)
							indices.push_back(offset+ISLICE);
						if(j > 0)
							indices.push_back(offset-JSLICE);
						if(j < height-1)
							indices.push_back(offset+JSLICE);
						if(k > 0)
							indices.push_back(offset-KSLICE);
						if(k < depth-1)
							indices.push_back(offset+KSLICE);

						float count = indices.size();

						for(int cc = 0;cc < indices.size();cc++)
						{
							int cur_index = indices[cc];
							mat.SetValue(row_id,cur_index, 1);
						}

						mat.SetValue(row_id,row_id, -count);
					}
				}
			}

			mat.SetValue(row_num-1,0,1);


			*A = mat.ExportCCS(flag);

			if((*A) == 0)
			{
				if(display)
				{
					printf("create taucs_ccs_matrix fail\n");
				}
				return 0;
			}

			if(display)
			{
				int m = (*A)->m;
				int n = (*A)->n;
				int nnz = (*A)->colptr[n];
				printf("dim: %d x %d, nnz: %d\n",m,n,nnz);
			}

			return (col_num);
		}

		template<class T>
		void SolveOpenPoisson(T* u, T* v, T* w, int width, int height, int depth, const taucs_ccs_matrix* A, int maxiter, bool display)
		{
			T* mac_u = new T[(width+1)*height*depth];
			T* mac_v = new T[width*(height+1)*depth];
			T* mac_w = new T[width*height*(depth+1)];

			RegularGridtoMAC(width,height,depth,u,v,w,mac_u,mac_v,mac_w);

			SolveOpenPoisson_MACGrid(mac_u,mac_v,mac_w,width,height,depth,A,maxiter,display);

			MACtoRegularGrid(width,height,depth,mac_u,mac_v,mac_w,u,v,w);

			delete []mac_u;
			delete []mac_v;
			delete []mac_w;
		}

		template<class T>
		void SolveOpenPoisson_MACGrid(T* mac_u, T* mac_v, T* mac_w, int width, int height, int depth, const taucs_ccs_matrix* A, int maxiter, bool display)
		{
			int dim = width*height*depth;
			T* b = new T[width*height*depth];

			int it = 0;
			int max_iter = maxiter;
			double tol = 1e-12;

			T* x0 = new T[dim];
			T* x = new T[dim];
			memset(x0,0,sizeof(x0)*dim);
			memset(x,0,sizeof(x)*dim);

			ComputeDivergence(width,height,depth,mac_u,mac_v,mac_w,b);

			ZQ_PCGSolver::PCG_sparse_unsquare(A,b,x0,max_iter,tol,x,it,display);

			AdjustOpen(width,height,depth,mac_u,mac_v,mac_w,x);
			delete []x0;
			delete []x;
			delete []b;

		}

		template<class T>
		void SolveClosedPoisson(T* u, T* v, T* w, int width, int height, int depth, const taucs_ccs_matrix* A, int maxiter, bool display)
		{
			T* mac_u = new T[(width+1)*height*depth];
			T* mac_v = new T[width*(height+1)*depth];
			T* mac_w = new T[width*height*(depth+1)];

			RegularGridtoMAC(width,height,depth,u,v,w,mac_u,mac_v,mac_w);

			SolveClosedPoisson_MACGrid(mac_u,mac_v,mac_w,width,height,depth,A,maxiter,display);

			MACtoRegularGrid(width,height,depth,mac_u,mac_v,mac_w,u,v,w);

			delete []mac_u;
			delete []mac_v;
			delete []mac_w;
		}

		template<class T>
		void SolveClosedPoisson_MACGrid(T* mac_u, T* mac_v, T* mac_w, int width, int height, int depth, const taucs_ccs_matrix* A, int maxiter, bool display)
		{
			int dim = width*height*depth;
			T* b = new T[dim+1];

			int it = 0;
			int max_iter = maxiter;
			double tol = 1e-12;
			T* x0 = new T[dim];
			T* x = new T[dim];
			
			memset(x0,0,sizeof(T)*dim);
			memset(x,0,sizeof(T)*dim);


			ComputeDivergence(width,height,depth,mac_u,mac_v,mac_w,b);
			b[dim] = 0;

			ZQ_PCGSolver::PCG_sparse_unsquare(A,b,x0,max_iter,tol,x,it,display);

			AdjustClosed(width,height,depth,mac_u,mac_v,mac_w,x);
			delete []x0;
			delete []x;
			delete []b;
		}

		template<class T>
		void SolveOpenPoissonSOR(T* u, T* v, T* w, int width, int height, int depth, int nSORIterations, bool display)
		{
			T* mac_u = new T[(width+1)*height*depth];
			T* mac_v = new T[width*(height+1)*depth];
			T* mac_w = new T[width*height*(depth+1)];

			RegularGridtoMAC(width,height,depth,u,v,w,mac_u,mac_v,mac_w);

			SolveOpenPoissonSOR_MACGrid(mac_u,mac_v,mac_w,width,height,depth,nSORIterations,display);

			MACtoRegularGrid(width,height,depth,mac_u,mac_v,mac_w,u,v,w);

			delete []mac_u;
			delete []mac_v;	
			delete []mac_w;
		}

		template<class T>
		void SolveOpenPoissonSOR_MACGrid(T* mac_u, T* mac_v, T* mac_w, int width, int height, int depth, int nSORIterations, bool display)
		{
			int dim = width*height*depth;

			T* b = new T[dim];
			T* x = new T[dim];
			memset(x,0,sizeof(T)*dim);

			ComputeDivergence(width,height,depth,mac_u,mac_v,mac_w,b);

			int KSLICE = height*width;
			int JSLICE = width;
			int ISLICE = 1;

			// Begin SOR
			for(int sor_it = 0;sor_it < nSORIterations;sor_it++)
			{
				for(int k = 0;k < depth;k++)
				{
					for(int j = 0;j < height;j++)
					{
						for(int i = 0;i < width;i++)
						{
							int offset = k*height*width+j*width+i;
							double coeff = 0,sigma = 0;

							coeff = 6;

							if(k == 0)
							{
								sigma += x[offset+KSLICE];
							}
							else if(k == depth-1)
							{
								sigma += x[offset-KSLICE];
							}
							else
							{
								sigma += x[offset-KSLICE]+x[offset+KSLICE];
							}

							if(j == 0)
							{
								sigma += x[offset+JSLICE];
							}
							else if(j == height-1)
							{
								sigma += x[offset-JSLICE];
							}
							else 
							{
								sigma += x[offset-JSLICE]+x[offset+JSLICE];
							}

							if(i == 0)
							{
								sigma += x[offset+ISLICE];
							}
							else if(i == width-1)
							{
								sigma += x[offset-ISLICE];
							}
							else
							{
								sigma += x[offset+ISLICE]+x[offset-ISLICE];
							}
							sigma -= b[offset];
							x[offset] = sigma/coeff;
						}
					}
				}
			}

			//End SOR

			AdjustOpen(width,height,depth,mac_u,mac_v,mac_w,x);
			delete []b;
			delete []x;

		}


		template<class T>
		void SolveOpenPoissonRedBlack(T* u, T* v, T* w, int width, int height, int depth, int nIterations, bool display)
		{
			T* mac_u = new T[(width+1)*height*depth];
			T* mac_v = new T[width*(height+1)*depth];
			T* mac_w = new T[width*height*(depth+1)];

			RegularGridtoMAC(width,height,depth,u,v,w,mac_u,mac_v,mac_w);

			SolveOpenPoissonRedBlack_MACGrid(mac_u,mac_v,mac_w,width,height,depth,nIterations,display);

			MACtoRegularGrid(width,height,depth,mac_u,mac_v,mac_w,u,v,w);

			delete []mac_u;
			delete []mac_v;	
			delete []mac_w;
		}


		template<class T>
		void SolveOpenPoissonRedBlack_MACGrid(T* mac_u, T* mac_v, T* mac_w, int width, int height, int depth, int nIterations, bool display)
		{
			int dim = width*height*depth;

			T* b = new T[dim];
			T* x = new T[dim];
			memset(x,0,sizeof(T)*dim);

			ComputeDivergence(width,height,depth,mac_u,mac_v,mac_w,b);


			int KSLICE = height*width;
			int JSLICE = width;
			int ISLICE = 1;

			// Begin Red-Black
			for(int rb_it = 0;rb_it < nIterations;rb_it++)
			{

				// handle red
				for(int k = 0;k < depth;k++)
				{
					for(int j = 0;j < height;j++)
					{
						for(int i = 0;i < width;i++)
						{
							if((i+j+k)%2 == 1)
								continue;

							int offset = k*height*width+j*width+i;
							double coeff = 0,sigma = 0;

							coeff = 6;

							if(k == 0)
							{
								sigma += x[offset+KSLICE];
							}
							else if(k == depth-1)
							{
								sigma += x[offset-KSLICE];
							}
							else
							{
								sigma += x[offset-KSLICE]+x[offset+KSLICE];
							}

							if(j == 0)
							{
								sigma += x[offset+JSLICE];
							}
							else if(j == height-1)
							{
								sigma += x[offset-JSLICE];
							}
							else 
							{
								sigma += x[offset-JSLICE]+x[offset+JSLICE];
							}

							if(i == 0)
							{
								sigma += x[offset+ISLICE];
							}
							else if(i == width-1)
							{
								sigma += x[offset-ISLICE];
							}
							else
							{
								sigma += x[offset+ISLICE]+x[offset-ISLICE];
							}
							sigma -= b[offset];
							x[offset] = sigma/coeff;
						}
					}
				}


				//handle black

				for(int k = 0;k < depth;k++)
				{
					for(int j = 0;j < height;j++)
					{
						for(int i = 0;i < width;i++)
						{
							if((i+j+k)%2 == 0)
								continue;

							int offset = k*height*width+j*width+i;
							double coeff = 0,sigma = 0;

							coeff = 6;

							if(k == 0)
							{
								sigma += x[offset+KSLICE];
							}
							else if(k == depth-1)
							{
								sigma += x[offset-KSLICE];
							}
							else
							{
								sigma += x[offset-KSLICE]+x[offset+KSLICE];
							}

							if(j == 0)
							{
								sigma += x[offset+JSLICE];
							}
							else if(j == height-1)
							{
								sigma += x[offset-JSLICE];
							}
							else 
							{
								sigma += x[offset-JSLICE]+x[offset+JSLICE];
							}

							if(i == 0)
							{
								sigma += x[offset+ISLICE];
							}
							else if(i == width-1)
							{
								sigma += x[offset-ISLICE];
							}
							else
							{
								sigma += x[offset+ISLICE]+x[offset-ISLICE];
							}
							sigma -= b[offset];
							x[offset] = sigma/coeff;
						}
					}
				}

			}

			//End Red-Black

			AdjustOpen(width,height,depth,mac_u,mac_v,mac_w,x);

			delete []b;
			delete []x;

		}


		/**********************************     for scene with obstacle      *************************************/

		template<class T>
		int BuildOpenPoisson(int width, int height, int depth, const bool* occupy, taucs_ccs_matrix** A, bool display)
		{	
			int flag;
			if(strcmp(typeid(T).name(),"float") == 0)
				flag = TAUCS_SINGLE;
			else if(strcmp(typeid(T).name(),"double") == 0)
				flag = TAUCS_DOUBLE;
			else
				return 0;

			int dim = width*height*depth;

			ZQ_SparseMatrix<T> mat(dim,dim);

			int KSLICE = height*width;
			int JSLICE = width;
			int ISLICE = 1;


			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 0;i < width;i++)
					{
						int offset = k*height*width+j*width+i;

						int row_id = offset;

						if(occupy[offset])
						{
							mat.SetValue(offset,offset,1);
							continue;
						}

						std::vector<int> indices;
						if(k > 0)
							indices.push_back(offset-KSLICE);
						if(k < depth-1)
							indices.push_back(offset+KSLICE);
						if(j > 0)
							indices.push_back(offset-JSLICE);
						if(j < height-1)
							indices.push_back(offset+JSLICE);
						if(i > 0)
							indices.push_back(offset-ISLICE);
						if(i < width-1)
							indices.push_back(offset+ISLICE);

						float count = 6 - indices.size();
						for(int cc = 0;cc < indices.size();cc++)
						{
							int cur_index = indices[cc];
							if(!occupy[cur_index])
							{
								count += 1;
								mat.SetValue(row_id,cur_index, 1);
							}
						}

						mat.SetValue(row_id,row_id, -count);

					}
				}
			}

			*A = mat.ExportCCS(flag);
			if((*A) == 0)
			{
				if(display)
				{
					printf("create taucs_ccs_matrix fail\n");
				}
				return 0;
			}

			if(display)
			{
				int m = (*A)->m;
				int n = (*A)->n;
				int nnz = (*A)->colptr[n];
				printf("dim: %d x %d, nnz: %d\n",m,n,nnz);
			}

			return dim;
		}


		template<class T>
		int BuildClosedPoisson(int width, int height, int depth, const bool* occupy, taucs_ccs_matrix** A, bool display)
		{
			int flag;
			if(strcmp(typeid(T).name(),"float") == 0)
				flag = TAUCS_SINGLE;
			else if(strcmp(typeid(T).name(),"double") == 0)
				flag = TAUCS_DOUBLE;
			else
				return 0;

			bool haveFirst = false;
			int first = -1;
			for(int k = 0;k < depth*height*width;k++)
			{
				if(!occupy[k])
				{
					haveFirst = true;
					first = k;
				}
			}

			if(!haveFirst)
			{
				return 0;
			}

			int dim = width*height*depth;
			int col_num = dim;
			int row_num = dim+1;

			ZQ_SparseMatrix<T> mat(row_num,col_num);

			int KSLICE = height*width;
			int JSLICE = width;
			int ISLICE = 1;

			for(int k = 0;k < depth;k++)
			{
				for(int j = 0;j < height;j++)
				{
					for(int i = 0;i < width;i++)
					{
						int offset = k*height*width+j*width+i;
						int row_id = offset;

						if(occupy[offset])
						{
							mat.SetValue(offset,offset,1);
							continue;
						}

						std::vector<int> indices;
						if(k > 0)
							indices.push_back(offset-KSLICE);
						if(k < depth-1)
							indices.push_back(offset+KSLICE);
						if(j > 0)
							indices.push_back(offset-JSLICE);
						if(j < height-1)
							indices.push_back(offset+JSLICE);
						if(i > 0)
							indices.push_back(offset-ISLICE);
						if(i < width-1)
							indices.push_back(offset+ISLICE);

						float count = 0.0;
						for(int cc = 0;cc < indices.size();cc++)
						{
							int cur_index = indices[cc];
							if(!occupy[cur_index])
							{
								count += 1;
								
								mat.SetValue(row_id,cur_index, 1);
							}
						}

						mat.SetValue(row_id,row_id, -count);
					}
				}
			}

			mat.SetValue(row_num-1,first,1);

			*A = mat.ExportCCS(flag);

			if((*A) == 0)
			{
				if(display)
				{
					printf("create taucs_ccs_matrix fail\n");
				}
				return 0;
			}

			if(display)
			{
				int m = (*A)->m;
				int n = (*A)->n;
				int nnz = (*A)->colptr[n];
				printf("dim: %d x %d, nnz: %d\n",m,n,nnz);
			}

			return dim;
		}

		template<class T>
		void SolveOpenPoisson_MACGrid(T* mac_u, T* mac_v, T* mac_w, int width, int height, int depth, const bool* occupy, const taucs_ccs_matrix* A, int maxiter, bool display)
		{
			int dim = width*height*depth;

			int max_iter = maxiter;
			double tol = 1e-12;
			int it = 0;

			T* b = new T[dim];
			memset(b,0,sizeof(T)*dim);
			T* x = new T[dim];
			memset(x,0,sizeof(T)*dim);
			T* x0 = new T[dim];
			memset(x0,0,sizeof(T)*dim);

			ComputeDivergence(width,height,depth,mac_u,mac_v,mac_w,b);

			ZQ_PCGSolver::PCG_sparse_unsquare(A,b,x0,max_iter,tol,x,it,display);

			AdjustOpen(width,height,depth,occupy,mac_u,mac_v,mac_w,x);

			delete []x0;
			delete []x;
			delete []b;
		}

		template<class T>
		void SolveClosedPoisson_MACGrid(T* mac_u, T* mac_v, T* mac_w, int width, int height, int depth, const bool* occupy, const taucs_ccs_matrix* A, int maxiter, bool display)
		{
			int dim = width*height*depth;

			int max_iter = maxiter;
			double tol = 1e-12;
			int it = 0;

			T* b = new T[dim+1];
			T* x0 = new T[dim];
			T* x = new T[dim];
			memset(x0,0,sizeof(T)*dim);
			memset(x,0,sizeof(T)*dim);

			ComputeDivergence(width,height,depth,mac_u,mac_v,mac_w,b);
			b[dim] = 0;

			ZQ_PCGSolver::PCG_sparse_unsquare(A,b,x0,max_iter,tol,x,it,display);
			AdjustClosed(width,height,depth,occupy,mac_u,mac_v,mac_w,x);
			delete []b;
			delete []x0;
			delete []x;
		}

		template<class T>
		void SolveOpenPoissonSOR_MACGrid(T* mac_u, T* mac_v, T* mac_w, int width, int height, int depth, const bool* occupy, int nSORIterations, bool display)
		{
			int dim = width*height*depth;

			T* b = new T[dim];
			T* p = new T[dim];
			memset(p,0,sizeof(T)*dim);

			ComputeDivergence(width,height,depth,mac_u,mac_v,mac_w,b);

			int KSLICE = height*width;
			int JSLICE = width;
			int ISLICE = 1;

			// Begin SOR
			for(int sor_it = 0;sor_it < nSORIterations;sor_it++)
			{

				for(int k = 0;k < depth;k++)
				{
					for(int j = 0;j < height;j++)
					{
						for(int i = 0;i < width;i++)
						{
							int offset = k*height*width+j*width+i;
							if(occupy[offset])
								continue;

							double coeff = 0,sigma = 0;

							if(k == 0)
							{
								if(!occupy[offset+KSLICE])
								{
									sigma += p[offset+KSLICE];
									coeff += 1;
								}
								coeff += 1;
							}
							else if(k == depth-1)
							{
								if(!occupy[offset-KSLICE])
								{
									sigma += p[offset-KSLICE];
									coeff += 1;
								}
								coeff += 1;
							}
							else
							{
								if(!occupy[offset+KSLICE])
								{
									sigma += p[offset+KSLICE];
									coeff += 1;
								}
								if(!occupy[offset-KSLICE])
								{
									sigma += p[offset-KSLICE];
									coeff += 1;
								}
							}

							if(j == 0)
							{
								if(!occupy[offset+JSLICE])
								{
									sigma += p[offset+JSLICE];
									coeff += 1;
								}
								coeff += 1;
							}
							else if(j == height-1)
							{
								if(!occupy[offset-JSLICE])
								{
									sigma += p[offset-JSLICE];
									coeff += 1;
								}
								coeff += 1;
							}
							else
							{
								if(!occupy[offset+JSLICE])
								{
									sigma += p[offset+JSLICE];
									coeff += 1;
								}
								if(!occupy[offset-JSLICE])
								{
									sigma += p[offset-JSLICE];
									coeff += 1;
								}
							}


							if(i == 0)
							{
								if(!occupy[offset+ISLICE])
								{
									sigma += p[offset+ISLICE];
									coeff += 1;
								}
								coeff += 1;
							}
							else if(i == width-1)
							{
								if(!occupy[offset-ISLICE])
								{
									sigma += p[offset-ISLICE];
									coeff += 1;
								}
								coeff += 1;
							}
							else
							{
								if(!occupy[offset+ISLICE])
								{
									sigma += p[offset+ISLICE];
									coeff += 1;
								}
								if(!occupy[offset-ISLICE])
								{
									sigma += p[offset-ISLICE];
									coeff += 1;
								}
							}

							sigma -= b[offset];
							if(coeff == 0)
								p[offset] = 0;
							else
								p[offset] = sigma/coeff;
						}
					}
				}
			}

			//End SOR

			AdjustOpen(width,height,depth,occupy,mac_u,mac_v,mac_w,p);

			delete []b;
			delete []p;
		}

		template<class T>
		void SolveOpenPoissonRedBlack_MACGrid(T* mac_u, T* mac_v, T* mac_w, int width, int height, int depth, const bool* occupy, int nIterations, bool display)
		{
			int dim = width*height*depth;

			T* b = new T[dim];
			T* p = new T[dim];
			memset(p,0,sizeof(T)*dim);

			ComputeDivergence(width,height,depth,mac_u,mac_v,mac_w,b);

			int KSLICE = height*width;
			int JSLICE = width;
			int ISLICE = 1;

			// Begin Red-Black
			for(int rb_it = 0;rb_it < nIterations;rb_it++)
			{
				// handle red
				for(int k = 0;k < depth;k++)
				{
					for(int j = 0;j < height;j++)
					{
						for(int i = 0;i < width;i++)
						{
							if((i+j+k)%2 == 1)
								continue;

							int offset = k*height*width+j*width+i;
							if(occupy[offset])
								continue;

							double coeff = 0,sigma = 0;

							if(k == 0)
							{
								if(!occupy[offset+KSLICE])
								{
									sigma += p[offset+KSLICE];
									coeff += 1;
								}
								coeff += 1;
							}
							else if(k == depth-1)
							{
								if(!occupy[offset-KSLICE])
								{
									sigma += p[offset-KSLICE];
									coeff += 1;
								}
								coeff += 1;
							}
							else
							{
								if(!occupy[offset+KSLICE])
								{
									sigma += p[offset+KSLICE];
									coeff += 1;
								}
								if(!occupy[offset-KSLICE])
								{
									sigma += p[offset-KSLICE];
									coeff += 1;
								}
							}

							if(j == 0)
							{
								if(!occupy[offset+JSLICE])
								{
									sigma += p[offset+JSLICE];
									coeff += 1;
								}
								coeff += 1;
							}
							else if(j == height-1)
							{
								if(!occupy[offset-JSLICE])
								{
									sigma += p[offset-JSLICE];
									coeff += 1;
								}
								coeff += 1;
							}
							else
							{
								if(!occupy[offset+JSLICE])
								{
									sigma += p[offset+JSLICE];
									coeff += 1;
								}
								if(!occupy[offset-JSLICE])
								{
									sigma += p[offset-JSLICE];
									coeff += 1;
								}
							}


							if(i == 0)
							{
								if(!occupy[offset+ISLICE])
								{
									sigma += p[offset+ISLICE];
									coeff += 1;
								}
								coeff += 1;
							}
							else if(i == width-1)
							{
								if(!occupy[offset-ISLICE])
								{
									sigma += p[offset-ISLICE];
									coeff += 1;
								}
								coeff += 1;
							}
							else
							{
								if(!occupy[offset+ISLICE])
								{
									sigma += p[offset+ISLICE];
									coeff += 1;
								}
								if(!occupy[offset-ISLICE])
								{
									sigma += p[offset-ISLICE];
									coeff += 1;
								}
							}

							sigma -= b[offset];
							if(coeff == 0)
								p[offset] = 0;
							else
								p[offset] = sigma/coeff;
						}
					}
				}

				//handle black

				for(int k = 0;k < depth;k++)
				{
					for(int j = 0;j < height;j++)
					{
						for(int i = 0;i < width;i++)
						{
							if((i+j+k)%2 == 0)
								continue;

							int offset = k*height*width+j*width+i;
							if(occupy[offset])
								continue;

							double coeff = 0,sigma = 0;

							if(k == 0)
							{
								if(!occupy[offset+KSLICE])
								{
									sigma += p[offset+KSLICE];
									coeff += 1;
								}
								coeff += 1;
							}
							else if(k == depth-1)
							{
								if(!occupy[offset-KSLICE])
								{
									sigma += p[offset-KSLICE];
									coeff += 1;
								}
								coeff += 1;
							}
							else
							{
								if(!occupy[offset+KSLICE])
								{
									sigma += p[offset+KSLICE];
									coeff += 1;
								}
								if(!occupy[offset-KSLICE])
								{
									sigma += p[offset-KSLICE];
									coeff += 1;
								}
							}

							if(j == 0)
							{
								if(!occupy[offset+JSLICE])
								{
									sigma += p[offset+JSLICE];
									coeff += 1;
								}
								coeff += 1;
							}
							else if(j == height-1)
							{
								if(!occupy[offset-JSLICE])
								{
									sigma += p[offset-JSLICE];
									coeff += 1;
								}
								coeff += 1;
							}
							else
							{
								if(!occupy[offset+JSLICE])
								{
									sigma += p[offset+JSLICE];
									coeff += 1;
								}
								if(!occupy[offset-JSLICE])
								{
									sigma += p[offset-JSLICE];
									coeff += 1;
								}
							}


							if(i == 0)
							{
								if(!occupy[offset+ISLICE])
								{
									sigma += p[offset+ISLICE];
									coeff += 1;
								}
								coeff += 1;
							}
							else if(i == width-1)
							{
								if(!occupy[offset-ISLICE])
								{
									sigma += p[offset-ISLICE];
									coeff += 1;
								}
								coeff += 1;
							}
							else
							{
								if(!occupy[offset+ISLICE])
								{
									sigma += p[offset+ISLICE];
									coeff += 1;
								}
								if(!occupy[offset-ISLICE])
								{
									sigma += p[offset-ISLICE];
									coeff += 1;
								}
							}

							sigma -= b[offset];
							if(coeff == 0)
								p[offset] = 0;
							else
								p[offset] = sigma/coeff;
						}
					}
				}
			}
			//End Red-Black

			AdjustOpen(width,height,depth,occupy,mac_u,mac_v,mac_w,p);

			delete []b;
			delete []p;

		}
	}	
}


#endif