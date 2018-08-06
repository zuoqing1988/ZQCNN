#ifndef _ZQ_GRID_DEFORMATION_3D_H_
#define _ZQ_GRID_DEFORMATION_3D_H_
#pragma once

#include "ZQ_GridDeformation3DOptions.h"
#include "ZQ_SparseMatrix.h"
#include "ZQ_PCGSolver.h"
#include "ZQ_Matrix.h"
#include "ZQ_SVD.h"
#include <typeinfo>

namespace ZQ
{
	/******************************************************
	/* Referred to the paper:
	/* As-Rigid-As-Possible Surface Modeling, 2007
	/******************************************************/
	template<class T>
	class ZQ_GridDeformation3D
	{
		struct _mat33
		{
			T val[9];
		};
	public:
		ZQ_GridDeformation3D();
		~ZQ_GridDeformation3D();

	private:
		int width,height,depth;
		bool* nouseful_flag;   //true: means the point is blank
		bool* fixed_flag;      //true: means the point is control point
		int* x_index;
		int* y_index;
		int* z_index;
		int nouseful_num;
		int unknown_num;
		int fixed_num;
		taucs_ccs_matrix* G;
		taucs_ccs_matrix* B;
		taucs_ccs_matrix* G_dist;
		taucs_ccs_matrix* B_dist;
		taucs_ccs_matrix* H;
		taucs_ccs_matrix* D;

		T invFC_m[24];

		ZQ_GridDeformation3DOptions option;
		

	public:
		void ClearMatrix();
		bool BuildMatrix(const int width, const int height, const int depth, const bool* nouseful_flag, const bool* fixed_flag, const ZQ_GridDeformation3DOptions& opt);

		bool Deformation(const T* init_coord, T* out_coord, bool has_good_init = false);
	public:
		bool _deformation_laplacian(const T* init_coord, T* out_coord, const int iteration);
		bool _deformation_laplacian_XLOOP(const T* init_coord, T* out_coord, const int iteration);
		bool _deformation_ARAP_VERT(const T* init_coord, T* out_coord, const int nFPIter, const int iteration);
		bool _deformation_ARAP_VERT_XLOOP(const T* init_coord, T* out_coord, const int nFPIter, const int iteration);

	private:
		bool _addMatrixTo_ARAP_VERT(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight,
			const ZQ_GridDeformation3DOptions::NeighborType type = ZQ_GridDeformation3DOptions::NEIGHBOR_6);
		bool _addMatrixTo_ARAP_VERT_XLOOP(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight,
			const ZQ_GridDeformation3DOptions::NeighborType type = ZQ_GridDeformation3DOptions::NEIGHBOR_6);

		bool _estimateR_for_ARAP_VERT(const T* init_coord, std::vector<_mat33>& Rmats, const T scale, const ZQ_GridDeformation3DOptions::NeighborType type);
		bool _estimateR_for_ARAP_VERT_XLOOP(const T* init_coord, std::vector<_mat33>& Rmats, const T scale, const ZQ_GridDeformation3DOptions::NeighborType type);
		bool _solve_for_ARAP_VERT(const T* init_coord, const std::vector<_mat33>& Rmats, T* out_coord, const int iteration, const T scale, const ZQ_GridDeformation3DOptions::NeighborType type);
		bool _solve_for_ARAP_VERT_XLOOP(const T* init_coord, const std::vector<_mat33>& Rmats, T* out_coord, const int iteration, const T scale, const ZQ_GridDeformation3DOptions::NeighborType type);
	};

	/**************************  definitions  ********************************/

	template<class T>
	ZQ_GridDeformation3D<T>::ZQ_GridDeformation3D()
	{
		width = 0;
		height = 0;
		depth = 0;
		nouseful_flag = 0;
		fixed_flag = 0;
		x_index = 0;
		y_index = 0;
		z_index = 0;
		G = 0;
		B = 0;
		G_dist = 0;
		B_dist = 0;
		H = 0;
		D = 0;
	}

	template<class T>
	ZQ_GridDeformation3D<T>::~ZQ_GridDeformation3D()
	{
		ClearMatrix();
	}

	template<class T>
	void ZQ_GridDeformation3D<T>::ClearMatrix()
	{
		if(G)
		{
			ZQ_TaucsBase::ZQ_taucs_ccs_free(G);
			G = 0;
		}
		if(B)
		{
			ZQ_TaucsBase::ZQ_taucs_ccs_free(B);
			B = 0;
		}
		if(G_dist)
		{
			ZQ_TaucsBase::ZQ_taucs_ccs_free(G_dist);
			G_dist = 0;
		}
		if(B_dist)
		{
			ZQ_TaucsBase::ZQ_taucs_ccs_free(B_dist);
			B_dist = 0;
		}
		if(H)
		{
			ZQ_TaucsBase::ZQ_taucs_ccs_free(H);
			H = 0;
		}
		if(D)
		{
			ZQ_TaucsBase::ZQ_taucs_ccs_free(D);
			D = 0;
		}
		if(nouseful_flag)
		{
			delete []nouseful_flag;
			nouseful_flag = 0;
		}
		if(fixed_flag)
		{
			delete []fixed_flag;
			fixed_flag = 0;
		}
		if(x_index)
		{
			delete []x_index;
			x_index = 0;
		}
		if(y_index)
		{
			delete []y_index;
			y_index = 0;
		}
		if(z_index)
		{
			delete []z_index;
			z_index = 0;
		}
		width = 0;
		height = 0;
		depth = 0;
	}

	template<class T>
	bool ZQ_GridDeformation3D<T>::BuildMatrix(const int width, const int height, const int depth, const bool* nouseful_flag, const bool* fixed_flag, const ZQ_GridDeformation3DOptions& opt)
	{
		int taucs_flag;
		if(strcmp(typeid(T).name(),"float") == 0)
			taucs_flag = TAUCS_SINGLE;
		else if(strcmp(typeid(T).name(),"double") == 0)
			taucs_flag = TAUCS_DOUBLE;
		else
			return false;


		ClearMatrix();

		this->width = width;
		this->height = height;
		this->depth = depth;
		this->nouseful_flag = new bool[width*height*depth];
		this->fixed_flag = new bool[width*height*depth];
		memcpy(this->nouseful_flag,nouseful_flag,sizeof(bool)*width*height*depth);
		memcpy(this->fixed_flag,fixed_flag,sizeof(bool)*width*height*depth);
		x_index = new int[width*height*depth];
		y_index = new int[width*height*depth];
		z_index = new int[width*height*depth];
		memset(x_index,0,sizeof(int)*width*height*depth);
		memset(y_index,0,sizeof(int)*width*height*depth);
		memset(z_index,0,sizeof(int)*width*height*depth);
		option = opt;

		nouseful_num =  0;
		unknown_num = 0;
		for(int i = 0;i < width*height*depth;i++)
		{
			if(nouseful_flag[i])
				nouseful_num += 3;
			if(!nouseful_flag[i] && !fixed_flag[i])
				unknown_num += 3;
		}
		if(unknown_num == 0)
		{
			return false;
		}

		if(unknown_num > (width*height*depth*3 - nouseful_num - 6))
			return false;

		fixed_num = width*height*depth*3 - nouseful_num - unknown_num;

		int cur_unkonwn_index = 0, cur_fixed_index = 0;
		for(int i = 0;i < width*height*depth;i++)
		{
			if(!nouseful_flag[i])
			{
				if(fixed_flag[i])
				{
					x_index[i] = cur_fixed_index++;
					y_index[i] = cur_fixed_index++;
					z_index[i] = cur_fixed_index++;
				}
				else
				{
					x_index[i] = cur_unkonwn_index++;
					y_index[i] = cur_unkonwn_index++;
					z_index[i] = cur_unkonwn_index++;
				}
			}
		}

		ZQ_SparseMatrix<T> Gmat(unknown_num,unknown_num);
		ZQ_SparseMatrix<T> Bmat(unknown_num,fixed_num);

		switch(opt.methodType)
		{
		case ZQ_GridDeformation3DOptions::METHOD_LAPLACIAN:
			{

			}
			break;
		case ZQ_GridDeformation3DOptions::METHOD_LAPLACIAN_XLOOP:
			{

			}
			break;
		case ZQ_GridDeformation3DOptions::METHOD_ARAP_VERT_AS_CENTER:
			{
				if(!_addMatrixTo_ARAP_VERT(Gmat,Bmat,1,opt.neighborType))
					return false;
				G = Gmat.ExportCCS(taucs_flag);
				B = Bmat.ExportCCS(taucs_flag);
			}
			break;
		case ZQ_GridDeformation3DOptions::METHOD_ARAP_VERT_AS_CENTER_XLOOP:
			{
				if(!_addMatrixTo_ARAP_VERT_XLOOP(Gmat,Bmat,1,opt.neighborType))
					return false;
				G = Gmat.ExportCCS(taucs_flag);
				B = Bmat.ExportCCS(taucs_flag);
			}
			break;
		default:
			{
				return false;
			}
		}

		return true;
	}


	template<class T>
	bool ZQ_GridDeformation3D<T>::Deformation(const T* init_coord, T* out_coord, bool has_good_init)
	{
		switch(option.methodType)
		{
		case ZQ_GridDeformation3DOptions::METHOD_LAPLACIAN:
			{
				if(!_deformation_laplacian(init_coord,out_coord,option.iteration))
					return false;
			}
			break;
		case ZQ_GridDeformation3DOptions::METHOD_LAPLACIAN_XLOOP:
			{
				if(!_deformation_laplacian_XLOOP(init_coord,out_coord,option.iteration))
					return false;
			}
			break;
		case ZQ_GridDeformation3DOptions::METHOD_ARAP_VERT_AS_CENTER:
			{
				if(!has_good_init)
				{
					if(!_deformation_laplacian(init_coord,out_coord,option.iteration))
					{
						return false;
					}
					ZQ_DImage3D<T> tmp_coord(width,height,depth,3);
					memcpy(tmp_coord.data(),out_coord,sizeof(T)*width*height*depth*3);
					if(!_deformation_ARAP_VERT(tmp_coord.data(),out_coord,option.FPIteration,option.iteration))
						return false;
				}
				else
				{
					if(!_deformation_ARAP_VERT(init_coord,out_coord,option.FPIteration,option.iteration))
						return false;
				}
			}
			break;
		case ZQ_GridDeformation3DOptions::METHOD_ARAP_VERT_AS_CENTER_XLOOP:
			{
				if(!has_good_init)
				{
					if(!_deformation_laplacian_XLOOP(init_coord,out_coord,option.iteration))
					{
						return false;
					}
					ZQ_DImage3D<T> tmp_coord(width,height,depth,3);
					memcpy(tmp_coord.data(),out_coord,sizeof(T)*width*height*depth*3);
					if(!_deformation_ARAP_VERT_XLOOP(tmp_coord.data(),out_coord,option.FPIteration,option.iteration))
						return false;
				}
				else
				{
					if(!_deformation_ARAP_VERT_XLOOP(init_coord,out_coord,option.FPIteration,option.iteration))
						return false;
				}
			}
			break;
		}

		return true;
	}

	template<class T>
	bool ZQ_GridDeformation3D<T>::_deformation_laplacian(const T* init_coord, T* out_coord, const int iteration)
	{
		int taucs_flag;
		if(strcmp(typeid(T).name(),"float") == 0)
			taucs_flag = TAUCS_SINGLE;
		else if(strcmp(typeid(T).name(),"double") == 0)
			taucs_flag = TAUCS_DOUBLE;
		else
			return false;

		float scale = option.distance;

		int* unknown_idx = new int[width*height*depth];
		int* useful_idx = new int[width*height*depth];
		for(int i = 0;i < width*height*depth;i++)
		{
			unknown_idx[i] = -1;
			useful_idx[i] = -1;
		}

		T* lap_x = new T[width*height*depth];
		T* lap_y = new T[width*height*depth];
		T* lap_z = new T[width*height*depth];
		int* neighbor_num = new int[width*height*depth];
		memset(lap_x,0,sizeof(T)*width*height*depth);
		memset(lap_y,0,sizeof(T)*width*height*depth);
		memset(lap_z,0,sizeof(T)*width*height*depth);
		memset(neighbor_num,0,sizeof(int)*width*height*depth);

		int N_unknown = 0;
		int N_useful = 0;
		for(int k = 0;k < depth;k++)
		{
			for(int j = 0;j < height;j++)
			{
				for(int i = 0;i < width;i++)
				{
					int offset = k*height*width+j*width+i;
					if(nouseful_flag[offset])
						continue;

					useful_idx[offset] = N_useful++;

					if(!fixed_flag[offset])
						unknown_idx[offset] = N_unknown++;

					int sum_neighbor_num = 0;
					T sum_neighbor_x = 0;
					T sum_neighbor_y = 0;
					T sum_neighbor_z = 0;
					if(k-1 >= 0 && !nouseful_flag[(k-1)*height*width+j*width+i])
					{
						sum_neighbor_num++;
						sum_neighbor_z -= scale;
					}
					if(k+1 <= depth-1 && !nouseful_flag[(k+1)*height*width+j*width+i])
					{
						sum_neighbor_num++;
						sum_neighbor_z += scale;
					}
					if(j-1 >= 0 && !nouseful_flag[k*height*width+(j-1)*width+i])
					{
						sum_neighbor_num++;
						sum_neighbor_y -= scale;
					}
					if(j+1 <= height-1 && !nouseful_flag[k*height*width+(j+1)*width+i])
					{
						sum_neighbor_num++;
						sum_neighbor_y += scale;
					}
					if(i-1 >= 0 && !nouseful_flag[k*height*width+j*width+i-1])
					{
						sum_neighbor_num++;
						sum_neighbor_x -= scale;
					}
					if(i+1 <= width-1 && !nouseful_flag[k*height*width+j*width+i+1])
					{
						sum_neighbor_num++;
						sum_neighbor_x += scale;
					}
					neighbor_num[offset] = sum_neighbor_num;
					if(sum_neighbor_num > 0)
					{
						lap_x[offset] = sum_neighbor_x/sum_neighbor_num;
						lap_y[offset] = sum_neighbor_y/sum_neighbor_num;
						lap_z[offset] = sum_neighbor_z/sum_neighbor_num;
					}
				}
			}
		}

		ZQ_SparseMatrix<T> Amat(N_useful,N_unknown);
		T* b_x = new T[N_useful];
		T* b_y = new T[N_useful];
		T* b_z = new T[N_useful];

		for(int k = 0;k < depth;k++)
		{
			for(int j = 0;j < height;j++)
			{
				for(int i = 0;i < width;i++)
				{
					int offset = k*height*width+j*width+i;
					if(nouseful_flag[offset])
						continue;

					int row_idx = useful_idx[offset];

					b_x[row_idx] = lap_x[offset];
					b_y[row_idx] = lap_y[offset];
					b_z[row_idx] = lap_z[offset];

					if(fixed_flag[offset])
					{
						b_x[row_idx] += init_coord[offset*3+0];
						b_y[row_idx] += init_coord[offset*3+1];
						b_z[row_idx] += init_coord[offset*3+2];
					}
					else
					{
						Amat.AddTo(row_idx,unknown_idx[offset],-1);
					}

					if(neighbor_num[offset] > 0)
					{
						float weight = 1.0/neighbor_num[offset];
						int offset_2 = (k-1)*height*width+j*width+i;
						if(k-1 >= 0 && !nouseful_flag[offset_2])
						{
							if(fixed_flag[offset_2])
							{
								b_x[row_idx] -= weight*init_coord[offset_2*3+0];
								b_y[row_idx] -= weight*init_coord[offset_2*3+1];
								b_z[row_idx] -= weight*init_coord[offset_2*3+2];
							}
							else
							{
								Amat.AddTo(row_idx,unknown_idx[offset_2],weight);
							}
						}
						offset_2 = (k+1)*height*width+j*width+i;
						if(k+1 <= depth-1 && !nouseful_flag[offset_2])
						{
							if(fixed_flag[offset_2])
							{
								b_x[row_idx] -= weight*init_coord[offset_2*3+0];
								b_y[row_idx] -= weight*init_coord[offset_2*3+1];
								b_z[row_idx] -= weight*init_coord[offset_2*3+2];
							}
							else
							{
								Amat.AddTo(row_idx,unknown_idx[offset_2],weight);
							}
						}
						offset_2 = k*height*width+(j-1)*width+i;
						if(j-1 >= 0 && !nouseful_flag[offset_2])
						{
							if(fixed_flag[offset_2])
							{
								b_x[row_idx] -= weight*init_coord[offset_2*3+0];
								b_y[row_idx] -= weight*init_coord[offset_2*3+1];
								b_z[row_idx] -= weight*init_coord[offset_2*3+2];
							}
							else
							{
								Amat.AddTo(row_idx,unknown_idx[offset_2],weight);
							}
						}
						offset_2 = k*height*width+(j+1)*width+i;
						if(j+1 <= height-1 && !nouseful_flag[offset_2])
						{
							if(fixed_flag[offset_2])
							{
								b_x[row_idx] -= weight*init_coord[offset_2*3+0];
								b_y[row_idx] -= weight*init_coord[offset_2*3+1];
								b_z[row_idx] -= weight*init_coord[offset_2*3+2];
							}
							else
							{
								Amat.AddTo(row_idx,unknown_idx[offset_2],weight);
							}
						}
						offset_2 = k*height*width+j*width+i-1;
						if(i-1 >= 0 && !nouseful_flag[offset_2])
						{
							if(fixed_flag[offset_2])
							{
								b_x[row_idx] -= weight*init_coord[offset_2*3+0];
								b_y[row_idx] -= weight*init_coord[offset_2*3+1];
								b_z[row_idx] -= weight*init_coord[offset_2*3+2];
							}
							else
							{
								Amat.AddTo(row_idx,unknown_idx[offset_2],weight);
							}
						}
						offset_2 = k*height*width+j*width+i+1;
						if(i+1 <= width-1 && !nouseful_flag[offset_2])
						{
							if(fixed_flag[offset_2])
							{
								b_x[row_idx] -= weight*init_coord[offset_2*3+0];
								b_y[row_idx] -= weight*init_coord[offset_2*3+1];
								b_z[row_idx] -= weight*init_coord[offset_2*3+2];
							}
							else
							{
								Amat.AddTo(row_idx,unknown_idx[offset_2],weight);
							}
						}
					}
				}
			}
		}

		/*FILE* out = fopen("mat.txt","w");
		for(int i = 0;i < 64;i++)
		{
		for(int j = 0;j < 56;j++)
		{
		fprintf(out,"%f ",Amat.GetValue(i,j));
		}
		fprintf(out,"\n");
		}
		fclose(out);
		FILE* out1 = fopen("b.txt","w");
		for(int i = 0;i < 64;i++)
		{
		fprintf(out1,"%f\n",b_x[i]);
		}
		fclose(out1);*/

		taucs_ccs_matrix* A = Amat.ExportCCS(taucs_flag);
		double tol = 1e-16;
		int it = 0;

		T* x0 = new T[N_unknown*3];
		T* x = new T[N_unknown*3];
		memset(x0,0,sizeof(T)*N_unknown*3);
		ZQ_PCGSolver::PCG_sparse_unsquare(A,b_x,x0,iteration,tol,x,it);
		ZQ_PCGSolver::PCG_sparse_unsquare(A,b_y,x0+N_unknown,iteration,tol,x+N_unknown,it);
		ZQ_PCGSolver::PCG_sparse_unsquare(A,b_z,x0+N_unknown*2,iteration,tol,x+N_unknown*2,it);

		for(int pp = 0;pp < width*height*depth;pp++)
		{
			if(nouseful_flag[pp])
				continue;
			if(fixed_flag[pp])
			{
				out_coord[pp*3+0] = init_coord[pp*3+0];
				out_coord[pp*3+1] = init_coord[pp*3+1];
				out_coord[pp*3+2] = init_coord[pp*3+2];
			}
			else
			{
				out_coord[pp*3+0] = x[unknown_idx[pp]];
				out_coord[pp*3+1] = x[unknown_idx[pp]+N_unknown];
				out_coord[pp*3+2] = x[unknown_idx[pp]+N_unknown*2];
			}
		}

		delete []x;
		delete []x0;
		delete []b_x;
		delete []b_y;
		delete []b_z;
		delete []lap_x;
		delete []lap_y;
		delete []lap_z;
		delete []useful_idx;
		delete []unknown_idx;
		delete []neighbor_num;
		ZQ_TaucsBase::ZQ_taucs_ccs_free(A);
		return true;
	}


	template<class T>
	bool ZQ_GridDeformation3D<T>::_deformation_laplacian_XLOOP(const T* init_coord, T* out_coord, const int iteration)
	{
		int taucs_flag;
		if(strcmp(typeid(T).name(),"float") == 0)
			taucs_flag = TAUCS_SINGLE;
		else if(strcmp(typeid(T).name(),"double") == 0)
			taucs_flag = TAUCS_DOUBLE;
		else
			return false;


		float scale = option.distance;

		int* unknown_idx = new int[width*height*depth];
		int* useful_idx = new int[width*height*depth];
		for(int i = 0;i < width*height*depth;i++)
		{
			unknown_idx[i] = -1;
			useful_idx[i] = -1;
		}

		T* lap_x = new T[width*height*depth];
		T* lap_y = new T[width*height*depth];
		T* lap_z = new T[width*height*depth];
		int* neighbor_num = new int[width*height*depth];
		memset(lap_x,0,sizeof(T)*width*height*depth);
		memset(lap_y,0,sizeof(T)*width*height*depth);
		memset(lap_z,0,sizeof(T)*width*height*depth);
		memset(neighbor_num,0,sizeof(int)*width*height*depth);

		int N_unknown = 0;
		int N_useful = 0;
		for(int k = 0;k < depth;k++)
		{
			for(int j = 0;j < height;j++)
			{
				for(int i = 0;i < width;i++)
				{
					int offset = k*height*width+j*width+i;
					if(nouseful_flag[offset])
						continue;

					useful_idx[offset] = N_useful++;

					if(!fixed_flag[offset])
						unknown_idx[offset] = N_unknown++;

					int sum_neighbor_num = 0;
					T sum_neighbor_x = 0;
					T sum_neighbor_y = 0;
					T sum_neighbor_z = 0;
					if(k-1 >= 0 && !nouseful_flag[(k-1)*height*width+j*width+i])
					{
						sum_neighbor_num++;
						sum_neighbor_z -= scale;
					}
					if(k+1 <= depth-1 && !nouseful_flag[(k+1)*height*width+j*width+i])
					{
						sum_neighbor_num++;
						sum_neighbor_z += scale;
					}
					if(j-1 >= 0 && !nouseful_flag[k*height*width+(j-1)*width+i])
					{
						sum_neighbor_num++;
						sum_neighbor_y -= scale;
					}
					if(j+1 <= height-1 && !nouseful_flag[k*height*width+(j+1)*width+i])
					{
						sum_neighbor_num++;
						sum_neighbor_y += scale;
					}
					if(!nouseful_flag[k*height*width+j*width+(i-1+width)%width])
					{
						sum_neighbor_num++;
						sum_neighbor_x -= scale;
					}
					if(!nouseful_flag[k*height*width+j*width+(i+1)%width])
					{
						sum_neighbor_num++;
						sum_neighbor_x += scale;
					}
					neighbor_num[offset] = sum_neighbor_num;
					if(sum_neighbor_num > 0)
					{
						lap_x[offset] = sum_neighbor_x/sum_neighbor_num;
						lap_y[offset] = sum_neighbor_y/sum_neighbor_num;
						lap_z[offset] = sum_neighbor_z/sum_neighbor_num;
					}
				}
			}
		}

		ZQ_SparseMatrix<T> Amat(N_useful,N_unknown);
		T* b_x = new T[N_useful];
		T* b_y = new T[N_useful];
		T* b_z = new T[N_useful];

		for(int k = 0;k < depth;k++)
		{
			for(int j = 0;j < height;j++)
			{
				for(int i = 0;i < width;i++)
				{
					int offset = k*height*width+j*width+i;
					if(nouseful_flag[offset])
						continue;

					int row_idx = useful_idx[offset];

					b_x[row_idx] = lap_x[offset];
					b_y[row_idx] = lap_y[offset];
					b_z[row_idx] = lap_z[offset];

					if(fixed_flag[offset])
					{
						b_x[row_idx] += init_coord[offset*3+0];
						b_y[row_idx] += init_coord[offset*3+1];
						b_z[row_idx] += init_coord[offset*3+2];
					}
					else
					{
						Amat.AddTo(row_idx,unknown_idx[offset],-1);
					}

					if(neighbor_num[offset] > 0)
					{
						float weight = 1.0/neighbor_num[offset];
						int offset_2 = (k-1)*height*width+j*width+i;
						if(k-1 >= 0 && !nouseful_flag[offset_2])
						{
							if(fixed_flag[offset_2])
							{
								b_x[row_idx] -= weight*init_coord[offset_2*3+0];
								b_y[row_idx] -= weight*init_coord[offset_2*3+1];
								b_z[row_idx] -= weight*init_coord[offset_2*3+2];
							}
							else
							{
								Amat.AddTo(row_idx,unknown_idx[offset_2],weight);
							}
						}
						offset_2 = (k+1)*height*width+j*width+i;
						if(k+1 <= depth-1 && !nouseful_flag[offset_2])
						{
							if(fixed_flag[offset_2])
							{
								b_x[row_idx] -= weight*init_coord[offset_2*3+0];
								b_y[row_idx] -= weight*init_coord[offset_2*3+1];
								b_z[row_idx] -= weight*init_coord[offset_2*3+2];
							}
							else
							{
								Amat.AddTo(row_idx,unknown_idx[offset_2],weight);
							}
						}
						offset_2 = k*height*width+(j-1)*width+i;
						if(j-1 >= 0 && !nouseful_flag[offset_2])
						{
							if(fixed_flag[offset_2])
							{
								b_x[row_idx] -= weight*init_coord[offset_2*3+0];
								b_y[row_idx] -= weight*init_coord[offset_2*3+1];
								b_z[row_idx] -= weight*init_coord[offset_2*3+2];
							}
							else
							{
								Amat.AddTo(row_idx,unknown_idx[offset_2],weight);
							}
						}
						offset_2 = k*height*width+(j+1)*width+i;
						if(j+1 <= height-1 && !nouseful_flag[offset_2])
						{
							if(fixed_flag[offset_2])
							{
								b_x[row_idx] -= weight*init_coord[offset_2*3+0];
								b_y[row_idx] -= weight*init_coord[offset_2*3+1];
								b_z[row_idx] -= weight*init_coord[offset_2*3+2];
							}
							else
							{
								Amat.AddTo(row_idx,unknown_idx[offset_2],weight);
							}
						}
						offset_2 = k*height*width+j*width+(i-1+width)%width;
						if(!nouseful_flag[offset_2])
						{
							if(fixed_flag[offset_2])
							{
								b_x[row_idx] -= weight*init_coord[offset_2*3+0];
								b_y[row_idx] -= weight*init_coord[offset_2*3+1];
								b_z[row_idx] -= weight*init_coord[offset_2*3+2];
							}
							else
							{
								Amat.AddTo(row_idx,unknown_idx[offset_2],weight);
							}
						}
						offset_2 = k*height*width+j*width+(i+1)%width;
						if(!nouseful_flag[offset_2])
						{
							if(fixed_flag[offset_2])
							{
								b_x[row_idx] -= weight*init_coord[offset_2*3+0];
								b_y[row_idx] -= weight*init_coord[offset_2*3+1];
								b_z[row_idx] -= weight*init_coord[offset_2*3+2];
							}
							else
							{
								Amat.AddTo(row_idx,unknown_idx[offset_2],weight);
							}
						}
					}
				}
			}
		}

		taucs_ccs_matrix* A = Amat.ExportCCS(taucs_flag);
		double tol = 1e-16;
		int it = 0;

		T* x0 = new T[N_unknown*3];
		T* x = new T[N_unknown*3];
		memset(x0,0,sizeof(T)*N_unknown*3);
		ZQ_PCGSolver::PCG_sparse_unsquare(A,b_x,x0,iteration,tol,x,it);
		ZQ_PCGSolver::PCG_sparse_unsquare(A,b_y,x0+N_unknown,iteration,tol,x+N_unknown,it);
		ZQ_PCGSolver::PCG_sparse_unsquare(A,b_z,x0+N_unknown*2,iteration,tol,x+N_unknown*2,it);

		for(int pp = 0;pp < width*height*depth;pp++)
		{
			if(nouseful_flag[pp])
				continue;
			if(fixed_flag[pp])
			{
				out_coord[pp*3+0] = init_coord[pp*3+0];
				out_coord[pp*3+1] = init_coord[pp*3+1];
				out_coord[pp*3+2] = init_coord[pp*3+2];
			}
			else
			{
				out_coord[pp*3+0] = x[unknown_idx[pp]];
				out_coord[pp*3+1] = x[unknown_idx[pp]+N_unknown];
				out_coord[pp*3+2] = x[unknown_idx[pp]+N_unknown*2];
			}
		}

		delete []x;
		delete []x0;
		delete []b_x;
		delete []b_y;
		delete []b_z;
		delete []lap_x;
		delete []lap_y;
		delete []lap_z;
		delete []useful_idx;
		delete []unknown_idx;
		delete []neighbor_num;
		ZQ_TaucsBase::ZQ_taucs_ccs_free(A);
		return true;
	}


	template<class T>
	bool ZQ_GridDeformation3D<T>::_deformation_ARAP_VERT(const T* init_coord, T* out_coord, const int nFPIter, const int iteration)
	{
		ZQ_DImage3D<T> tmp_img(width,height,depth,3);
		T*& tmp_data = tmp_img.data();
		memcpy(tmp_data,init_coord,sizeof(T)*width*height*depth*3);

		for(int fp_it = 0;fp_it < nFPIter;fp_it++)
		{
			std::vector<_mat33> Rmats;
			_estimateR_for_ARAP_VERT(tmp_data,Rmats,option.distance,option.neighborType);
			_solve_for_ARAP_VERT(tmp_data,Rmats,out_coord,iteration,option.distance,option.neighborType);
			memcpy(tmp_data,out_coord,sizeof(T)*width*height*depth*3);
		}

		return true;
	}


	template<class T>
	bool ZQ_GridDeformation3D<T>::_deformation_ARAP_VERT_XLOOP(const T* init_coord, T* out_coord, const int nFPIter, const int iteration)
	{
		ZQ_DImage3D<T> tmp_img(width,height,depth,3);
		T*& tmp_data = tmp_img.data();
		memcpy(tmp_data,init_coord,sizeof(T)*width*height*depth*3);

		for(int fp_it = 0;fp_it < nFPIter;fp_it++)
		{
			std::vector<_mat33> Rmats;
			_estimateR_for_ARAP_VERT_XLOOP(tmp_data,Rmats,option.distance,option.neighborType);
			_solve_for_ARAP_VERT_XLOOP(tmp_data,Rmats,out_coord,iteration,option.distance,option.neighborType);
			memcpy(tmp_data,out_coord,sizeof(T)*width*height*depth*3);
		}

		return true;
	}


	template<class T>
	bool ZQ_GridDeformation3D<T>::_addMatrixTo_ARAP_VERT(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight, const ZQ_GridDeformation3DOptions::NeighborType type)
	{

		T arap_weight = weight*weight;

		int dir_x[26] = {1,0,0,-1,0,0,  1,-1,1,-1,  1,-1,1,-1,  0,0,0,0,   1,-1,1,-1,1,-1,1,-1};
		int dir_y[26] = {0,1,0,0,-1,0,  1,1,-1,-1,  0,0,0,0,    1,-1,1,-1, 1,1,-1,-1,1,1,-1,-1};
		int dir_z[26] = {0,0,1,0,0,-1,  0,0,0,0,    1,1,-1,-1,  1,1,-1,-1, 1,1,1,1,-1,-1,-1,-1};
		int neighbor_loop_num = 6;
		switch(type)
		{
		case ZQ_GridDeformation3DOptions::NEIGHBOR_6:
			neighbor_loop_num = 6;
			break;
		case ZQ_GridDeformation3DOptions::NEIGHBOR_26:
			neighbor_loop_num = 26;
			break;
		default:
			return false;
			break;
		}

		for(int k = 0;k < depth;k++)
		{
			for(int j = 0;j < height;j++)
			{
				for(int i = 0;i < width;i++)
				{
					int offset = k*height*width + j*width + i;
					if(nouseful_flag[offset])
						continue;

					for(int dd = 0;dd < neighbor_loop_num;dd++)
					{
						int i_2 = i+dir_x[dd];
						int j_2 = j+dir_y[dd];
						int k_2 = k+dir_z[dd];
						int offset_2 = k_2*height*width + j_2*width + i_2;
						if(0 <= i_2 && i_2 < width && 0 <= j_2 && j_2 < height && 0 <= k_2 && k_2 < depth && !nouseful_flag[offset_2])
						{
							if(!fixed_flag[offset])
							{
								if(!fixed_flag[offset_2])
								{
									Gmat.AddTo(x_index[offset],x_index[offset],arap_weight);
									Gmat.AddTo(y_index[offset],y_index[offset],arap_weight);
									Gmat.AddTo(z_index[offset],z_index[offset],arap_weight);

									Gmat.AddTo(x_index[offset_2],x_index[offset_2],arap_weight);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],arap_weight);
									Gmat.AddTo(z_index[offset_2],z_index[offset_2],arap_weight);

									Gmat.AddTo(x_index[offset],x_index[offset_2],-arap_weight);
									Gmat.AddTo(x_index[offset_2],x_index[offset],-arap_weight);
									Gmat.AddTo(y_index[offset],y_index[offset_2],-arap_weight);
									Gmat.AddTo(y_index[offset_2],y_index[offset],-arap_weight);
									Gmat.AddTo(z_index[offset],z_index[offset_2],-arap_weight);
									Gmat.AddTo(z_index[offset_2],z_index[offset],-arap_weight);
								}
								else
								{
									Gmat.AddTo(x_index[offset],x_index[offset],arap_weight);
									Gmat.AddTo(y_index[offset],y_index[offset],arap_weight);
									Gmat.AddTo(z_index[offset],z_index[offset],arap_weight);

									Bmat.AddTo(x_index[offset],x_index[offset_2],-2*arap_weight);
									Bmat.AddTo(y_index[offset],y_index[offset_2],-2*arap_weight);
									Bmat.AddTo(z_index[offset],z_index[offset_2],-2*arap_weight);

								}
							}
							else
							{
								if(!fixed_flag[offset_2])
								{
									Gmat.AddTo(x_index[offset_2],x_index[offset_2],arap_weight);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],arap_weight);
									Gmat.AddTo(z_index[offset_2],z_index[offset_2],arap_weight);

									Bmat.AddTo(x_index[offset_2],x_index[offset],-2*arap_weight);
									Bmat.AddTo(y_index[offset_2],y_index[offset],-2*arap_weight);
									Bmat.AddTo(z_index[offset_2],z_index[offset],-2*arap_weight);
								}
								else
								{

								}
							}
						}
					}
				}
			}
		}
		return true;
	}


	template<class T>
	bool ZQ_GridDeformation3D<T>::_addMatrixTo_ARAP_VERT_XLOOP(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight, const ZQ_GridDeformation3DOptions::NeighborType type)
	{

		T arap_weight = weight*weight;

		int dir_x[26] = {1,0,0,-1,0,0,  1,-1,1,-1,  1,-1,1,-1,  0,0,0,0,   1,-1,1,-1,1,-1,1,-1};
		int dir_y[26] = {0,1,0,0,-1,0,  1,1,-1,-1,  0,0,0,0,    1,-1,1,-1, 1,1,-1,-1,1,1,-1,-1};
		int dir_z[26] = {0,0,1,0,0,-1,  0,0,0,0,    1,1,-1,-1,  1,1,-1,-1, 1,1,1,1,-1,-1,-1,-1};
		int neighbor_loop_num = 6;
		switch(type)
		{
		case ZQ_GridDeformation3DOptions::NEIGHBOR_6:
			neighbor_loop_num = 6;
			break;
		case ZQ_GridDeformation3DOptions::NEIGHBOR_26:
			neighbor_loop_num = 26;
			break;
		default:
			return false;
			break;
		}

		for(int k = 0;k < depth;k++)
		{
			for(int j = 0;j < height;j++)
			{
				for(int i = 0;i < width;i++)
				{
					int offset = k*height*width + j*width + i;
					if(nouseful_flag[offset])
						continue;

					for(int dd = 0;dd < neighbor_loop_num;dd++)
					{
						int i_2 = (i+dir_x[dd]+width)%width;
						int j_2 = j+dir_y[dd];
						int k_2 = k+dir_z[dd];
						int offset_2 = k_2*height*width + j_2*width + i_2;
						if(0 <= j_2 && j_2 < height && 0 <= k_2 && k_2 < depth && !nouseful_flag[offset_2])
						{
							if(!fixed_flag[offset])
							{
								if(!fixed_flag[offset_2])
								{
									Gmat.AddTo(x_index[offset],x_index[offset],arap_weight);
									Gmat.AddTo(y_index[offset],y_index[offset],arap_weight);
									Gmat.AddTo(z_index[offset],z_index[offset],arap_weight);

									Gmat.AddTo(x_index[offset_2],x_index[offset_2],arap_weight);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],arap_weight);
									Gmat.AddTo(z_index[offset_2],z_index[offset_2],arap_weight);

									Gmat.AddTo(x_index[offset],x_index[offset_2],-arap_weight);
									Gmat.AddTo(x_index[offset_2],x_index[offset],-arap_weight);
									Gmat.AddTo(y_index[offset],y_index[offset_2],-arap_weight);
									Gmat.AddTo(y_index[offset_2],y_index[offset],-arap_weight);
									Gmat.AddTo(z_index[offset],z_index[offset_2],-arap_weight);
									Gmat.AddTo(z_index[offset_2],z_index[offset],-arap_weight);
								}
								else
								{
									Gmat.AddTo(x_index[offset],x_index[offset],arap_weight);
									Gmat.AddTo(y_index[offset],y_index[offset],arap_weight);
									Gmat.AddTo(z_index[offset],z_index[offset],arap_weight);

									Bmat.AddTo(x_index[offset],x_index[offset_2],-2*arap_weight);
									Bmat.AddTo(y_index[offset],y_index[offset_2],-2*arap_weight);
									Bmat.AddTo(z_index[offset],z_index[offset_2],-2*arap_weight);

								}
							}
							else
							{
								if(!fixed_flag[offset_2])
								{
									Gmat.AddTo(x_index[offset_2],x_index[offset_2],arap_weight);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],arap_weight);
									Gmat.AddTo(z_index[offset_2],z_index[offset_2],arap_weight);

									Bmat.AddTo(x_index[offset_2],x_index[offset],-2*arap_weight);
									Bmat.AddTo(y_index[offset_2],y_index[offset],-2*arap_weight);
									Bmat.AddTo(z_index[offset_2],z_index[offset],-2*arap_weight);
								}
								else
								{

								}
							}
						}
					}
				}
			}
		}

		return true;
	}


	template<class T>
	bool ZQ_GridDeformation3D<T>::_estimateR_for_ARAP_VERT(const T* init_coord, std::vector<_mat33>& Rmats, const T scale, const ZQ_GridDeformation3DOptions::NeighborType type)
	{
		int dir_x[26] = {1,0,0,-1,0,0,  1,-1,1,-1,  1,-1,1,-1,  0,0,0,0,   1,-1,1,-1,1,-1,1,-1};
		int dir_y[26] = {0,1,0,0,-1,0,  1,1,-1,-1,  0,0,0,0,    1,-1,1,-1, 1,1,-1,-1,1,1,-1,-1};
		int dir_z[26] = {0,0,1,0,0,-1,  0,0,0,0,    1,1,-1,-1,  1,1,-1,-1, 1,1,1,1,-1,-1,-1,-1};
		int neighbor_loop_num = 6;
		switch(type)
		{
		case ZQ_GridDeformation3DOptions::NEIGHBOR_6:
			neighbor_loop_num = 6;
			break;
		case ZQ_GridDeformation3DOptions::NEIGHBOR_26:
			neighbor_loop_num = 26;
			break;
		default:
			return false;
			break;
		}

		Rmats.clear();
		Rmats.resize(width*height*depth);


		for(int k = 0;k < depth;k++)
		{
			for(int j = 0;j < height;j++)
			{
				for(int i = 0;i < width;i++)
				{
					int offset = k*height*width + j*width + i;
					if(nouseful_flag[offset])
						continue;

					float S[9] = {0};
					for(int dd = 0;dd < neighbor_loop_num;dd++)
					{
						int i_2 = i+dir_x[dd];
						int j_2 = j+dir_y[dd];
						int k_2 = k+dir_z[dd];
						int offset_2 = k_2*height*width + j_2*width + i_2;
						if(0 <= i_2 && i_2 < width && 0 <= j_2 && j_2 < height && 0 <= k_2 && k_2 < depth && !nouseful_flag[offset_2])
						{
							float cur_v[3] = {
								init_coord[offset_2*3+0]-init_coord[offset*3],
								init_coord[offset_2*3+1]-init_coord[offset*3+1],
								init_coord[offset_2*3+2]-init_coord[offset*3+2]
							};
							S[0] += dir_x[dd]*cur_v[0]*scale;
							S[1] += dir_x[dd]*cur_v[1]*scale;
							S[2] += dir_x[dd]*cur_v[2]*scale;
							S[3] += dir_y[dd]*cur_v[0]*scale;
							S[4] += dir_y[dd]*cur_v[1]*scale;
							S[5] += dir_y[dd]*cur_v[2]*scale;
							S[6] += dir_z[dd]*cur_v[0]*scale;
							S[7] += dir_z[dd]*cur_v[1]*scale;
							S[8] += dir_z[dd]*cur_v[2]*scale;
						}
					}

					ZQ_Matrix<T> Smat(3,3),U(3,3),D(3,3),V(3,3);
					Smat.SetData(0,0,S[0]);
					Smat.SetData(0,1,S[1]);
					Smat.SetData(0,2,S[2]);
					Smat.SetData(1,0,S[3]);
					Smat.SetData(1,1,S[4]);
					Smat.SetData(1,2,S[5]);
					Smat.SetData(2,0,S[6]);
					Smat.SetData(2,1,S[7]);
					Smat.SetData(2,2,S[8]);

					ZQ_SVD::Decompose(Smat,U,D,V);
					ZQ_Matrix<T> Rmat = V*U.GetTransposeMatrix();
					bool flag;
					Rmats[offset].val[0] = Rmat.GetData(0,0,flag);
					Rmats[offset].val[1] = Rmat.GetData(0,1,flag);
					Rmats[offset].val[2] = Rmat.GetData(0,2,flag);
					Rmats[offset].val[3] = Rmat.GetData(1,0,flag);
					Rmats[offset].val[4] = Rmat.GetData(1,1,flag);
					Rmats[offset].val[5] = Rmat.GetData(1,2,flag);
					Rmats[offset].val[6] = Rmat.GetData(2,0,flag);
					Rmats[offset].val[7] = Rmat.GetData(2,1,flag);
					Rmats[offset].val[8] = Rmat.GetData(2,2,flag);
				}
			}
		}


		return true;
	}


	template<class T>
	bool ZQ_GridDeformation3D<T>::_estimateR_for_ARAP_VERT_XLOOP(const T* init_coord, std::vector<_mat33>& Rmats, const T scale, const ZQ_GridDeformation3DOptions::NeighborType type)
	{
		int dir_x[26] = {1,0,0,-1,0,0,  1,-1,1,-1,  1,-1,1,-1,  0,0,0,0,   1,-1,1,-1,1,-1,1,-1};
		int dir_y[26] = {0,1,0,0,-1,0,  1,1,-1,-1,  0,0,0,0,    1,-1,1,-1, 1,1,-1,-1,1,1,-1,-1};
		int dir_z[26] = {0,0,1,0,0,-1,  0,0,0,0,    1,1,-1,-1,  1,1,-1,-1, 1,1,1,1,-1,-1,-1,-1};
		int neighbor_loop_num = 6;
		switch(type)
		{
		case ZQ_GridDeformation3DOptions::NEIGHBOR_6:
			neighbor_loop_num = 6;
			break;
		case ZQ_GridDeformation3DOptions::NEIGHBOR_26:
			neighbor_loop_num = 26;
			break;
		default:
			return false;
			break;
		}

		Rmats.clear();
		Rmats.resize(width*height*depth);

		for(int k = 0;k < depth;k++)
		{
			for(int j = 0;j < height;j++)
			{
				for(int i = 0;i < width;i++)
				{
					int offset = k*height*width + j*width + i;
					if(nouseful_flag[offset])
						continue;

					T S[9] = {0};
					for(int dd = 0;dd < neighbor_loop_num;dd++)
					{
						int i_2 = (i+dir_x[dd]+width)%width;
						int j_2 = j+dir_y[dd];
						int k_2 = k+dir_z[dd];
						int offset_2 = k_2*height*width + j_2*width + i_2;
						if(0 <= j_2 && j_2 < height && 0 <= k_2 && k_2 < depth && !nouseful_flag[offset_2])
						{
							T cur_v[3] = {
								init_coord[offset_2*3+0]-init_coord[offset*3],
								init_coord[offset_2*3+1]-init_coord[offset*3+1],
								init_coord[offset_2*3+2]-init_coord[offset*3+2]
							};
							S[0] += dir_x[dd]*cur_v[0]*scale;
							S[1] += dir_x[dd]*cur_v[1]*scale;
							S[2] += dir_x[dd]*cur_v[2]*scale;
							S[3] += dir_y[dd]*cur_v[0]*scale;
							S[4] += dir_y[dd]*cur_v[1]*scale;
							S[5] += dir_y[dd]*cur_v[2]*scale;
							S[6] += dir_z[dd]*cur_v[0]*scale;
							S[7] += dir_z[dd]*cur_v[1]*scale;
							S[8] += dir_z[dd]*cur_v[2]*scale;
						}
					}

					ZQ_Matrix<T> Smat(3,3),U(3,3),D(3,3),V(3,3);
					Smat.SetData(0,0,S[0]);
					Smat.SetData(0,1,S[1]);
					Smat.SetData(0,2,S[2]);
					Smat.SetData(1,0,S[3]);
					Smat.SetData(1,1,S[4]);
					Smat.SetData(1,2,S[5]);
					Smat.SetData(2,0,S[6]);
					Smat.SetData(2,1,S[7]);
					Smat.SetData(2,2,S[8]);

					ZQ_SVD::Decompose(Smat,U,D,V);
					ZQ_Matrix<T> Rmat = V*U.GetTransposeMatrix();
					bool flag;
					Rmats[offset].val[0] = Rmat.GetData(0,0,flag);
					Rmats[offset].val[1] = Rmat.GetData(0,1,flag);
					Rmats[offset].val[2] = Rmat.GetData(0,2,flag);
					Rmats[offset].val[3] = Rmat.GetData(1,0,flag);
					Rmats[offset].val[4] = Rmat.GetData(1,1,flag);
					Rmats[offset].val[5] = Rmat.GetData(1,2,flag);
					Rmats[offset].val[6] = Rmat.GetData(2,0,flag);
					Rmats[offset].val[7] = Rmat.GetData(2,1,flag);
					Rmats[offset].val[8] = Rmat.GetData(2,2,flag);
				}
			}
		}

		return true;
	}

	template<class T>
	bool ZQ_GridDeformation3D<T>::_solve_for_ARAP_VERT(const T* init_coord, const std::vector<_mat33>& Rmats, T* out_coord, const int iteration, const T scale, const ZQ_GridDeformation3DOptions::NeighborType type)
	{
		int dir_x[26] = {1,0,0,-1,0,0,  1,-1,1,-1,  1,-1,1,-1,  0,0,0,0,   1,-1,1,-1,1,-1,1,-1};
		int dir_y[26] = {0,1,0,0,-1,0,  1,1,-1,-1,  0,0,0,0,    1,-1,1,-1, 1,1,-1,-1,1,1,-1,-1};
		int dir_z[26] = {0,0,1,0,0,-1,  0,0,0,0,    1,1,-1,-1,  1,1,-1,-1, 1,1,1,1,-1,-1,-1,-1};
		int neighbor_loop_num = 6;
		switch(type)
		{
		case ZQ_GridDeformation3DOptions::NEIGHBOR_6:
			neighbor_loop_num = 6;
			break;
		case ZQ_GridDeformation3DOptions::NEIGHBOR_26:
			neighbor_loop_num = 26;
			break;
		default:
			return false;
			break;
		}

		T* init_unknown = new T[unknown_num];
		T* fixed_val = new T[fixed_num];
		for(int pp = 0;pp < width*height*depth;pp++)
		{
			if(!nouseful_flag[pp])
			{
				if(fixed_flag[pp])
				{
					fixed_val[x_index[pp]] = init_coord[pp*3];
					fixed_val[y_index[pp]] = init_coord[pp*3+1];
					fixed_val[z_index[pp]] = init_coord[pp*3+2];
				}
				else
				{
					init_unknown[x_index[pp]] = init_coord[pp*3];
					init_unknown[y_index[pp]] = init_coord[pp*3+1];
					init_unknown[z_index[pp]] = init_coord[pp*3+2];
				}
			}

		}

		T* f = new T[unknown_num];
		ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec(B,fixed_val,f);

		for(int k = 0;k < depth;k++)
		{
			for(int j = 0;j < height;j++)
			{
				for(int i = 0;i < width;i++)
				{
					int offset = k*height*width + j*width + i;
					if(nouseful_flag[offset])
						continue;

					for(int dd = 0;dd < neighbor_loop_num;dd++)
					{
						int i_2 = i+dir_x[dd];
						int j_2 = j+dir_y[dd];
						int k_2 = k+dir_z[dd];
						int offset_2 = k_2*height*width + j_2*width + i_2;
						if(0 <= i_2 && i_2 < width && 0 <= j_2 && j_2 < height && 0 <= k_2 && k_2 < depth && !nouseful_flag[offset_2])
						{
							T old_v[3] = {dir_x[dd]*scale,dir_y[dd]*scale,dir_z[dd]*scale};
							T cur_v[3] = {
								Rmats[offset].val[0] * old_v[0] + Rmats[offset].val[1] * old_v[1] + Rmats[offset].val[2] * old_v[2],
								Rmats[offset].val[3] * old_v[0] + Rmats[offset].val[4] * old_v[1] + Rmats[offset].val[5] * old_v[2],
								Rmats[offset].val[6] * old_v[0] + Rmats[offset].val[7] * old_v[1] + Rmats[offset].val[8] * old_v[2]
							};

							if(!fixed_flag[offset])
							{
								f[x_index[offset]] += 2*cur_v[0];
								f[y_index[offset]] += 2*cur_v[1];
								f[z_index[offset]] += 2*cur_v[2];
							}
							if(!fixed_flag[offset_2])
							{
								f[x_index[offset_2]] -= 2*cur_v[0];
								f[y_index[offset_2]] -= 2*cur_v[1];
								f[z_index[offset_2]] -= 2*cur_v[2];
							}
						}
					}
				}
			}
		}


		for(int i = 0;i < unknown_num;i++)
			f[i] *= -0.5;

		double tol = 1e-16;
		T* unknown_vals = new T[unknown_num];
		int it = 0;
		ZQ_PCGSolver::PCG(G,f,init_unknown,iteration,tol,unknown_vals,it,true);

		for(int i = 0;i < width*height*depth;i++)
		{
			if(nouseful_flag[i])
				continue;
			if(fixed_flag[i])
			{
				out_coord[i*3] = init_coord[i*3];
				out_coord[i*3+1] = init_coord[i*3+1];
				out_coord[i*3+2] = init_coord[i*3+2];
			}
			else
			{
				out_coord[i*3] = unknown_vals[x_index[i]];
				out_coord[i*3+1] = unknown_vals[y_index[i]];
				out_coord[i*3+2] = unknown_vals[z_index[i]];
			}
		}

		delete []f;
		delete []unknown_vals;
		delete []init_unknown;
		delete []fixed_val;

		return true;
	}


	template<class T>
	bool ZQ_GridDeformation3D<T>::_solve_for_ARAP_VERT_XLOOP(const T* init_coord, const std::vector<_mat33>& Rmats, T* out_coord, const int iteration, const T scale, const ZQ_GridDeformation3DOptions::NeighborType type)
	{
		int dir_x[26] = {1,0,0,-1,0,0,  1,-1,1,-1,  1,-1,1,-1,  0,0,0,0,   1,-1,1,-1,1,-1,1,-1};
		int dir_y[26] = {0,1,0,0,-1,0,  1,1,-1,-1,  0,0,0,0,    1,-1,1,-1, 1,1,-1,-1,1,1,-1,-1};
		int dir_z[26] = {0,0,1,0,0,-1,  0,0,0,0,    1,1,-1,-1,  1,1,-1,-1, 1,1,1,1,-1,-1,-1,-1};
		int neighbor_loop_num = 6;
		switch(type)
		{
		case ZQ_GridDeformation3DOptions::NEIGHBOR_6:
			neighbor_loop_num = 6;
			break;
		case ZQ_GridDeformation3DOptions::NEIGHBOR_26:
			neighbor_loop_num = 26;
			break;
		default:
			return false;
			break;
		}

		T* init_unknown = new T[unknown_num];
		T* fixed_val = new T[fixed_num];
		for(int pp = 0;pp < width*height*depth;pp++)
		{
			if(!nouseful_flag[pp])
			{
				if(fixed_flag[pp])
				{
					fixed_val[x_index[pp]] = init_coord[pp*3];
					fixed_val[y_index[pp]] = init_coord[pp*3+1];
					fixed_val[z_index[pp]] = init_coord[pp*3+2];
				}
				else
				{
					init_unknown[x_index[pp]] = init_coord[pp*3];
					init_unknown[y_index[pp]] = init_coord[pp*3+1];
					init_unknown[z_index[pp]] = init_coord[pp*3+2];
				}
			}

		}

		T* f = new T[unknown_num];
		ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec(B,fixed_val,f);

		for(int k = 0;k < depth;k++)
		{
			for(int j = 0;j < height;j++)
			{
				for(int i = 0;i < width;i++)
				{
					int offset = k*height*width + j*width + i;
					if(nouseful_flag[offset])
						continue;

					for(int dd = 0;dd < neighbor_loop_num;dd++)
					{
						int i_2 = (i+dir_x[dd]+width)%width;
						int j_2 = j+dir_y[dd];
						int k_2 = k+dir_z[dd];
						int offset_2 = k_2*height*width + j_2*width + i_2;
						if(0 <= j_2 && j_2 < height && 0 <= k_2 && k_2 < depth && !nouseful_flag[offset_2])
						{
							T old_v[3] = {dir_x[dd]*scale,dir_y[dd]*scale,dir_z[dd]*scale};
							T cur_v[3] = {
								Rmats[offset].val[0] * old_v[0] + Rmats[offset].val[1] * old_v[1] + Rmats[offset].val[2] * old_v[2],
								Rmats[offset].val[3] * old_v[0] + Rmats[offset].val[4] * old_v[1] + Rmats[offset].val[5] * old_v[2],
								Rmats[offset].val[6] * old_v[0] + Rmats[offset].val[7] * old_v[1] + Rmats[offset].val[8] * old_v[2]
							};

							if(!fixed_flag[offset])
							{
								f[x_index[offset]] += 2*cur_v[0];
								f[y_index[offset]] += 2*cur_v[1];
								f[z_index[offset]] += 2*cur_v[2];
							}
							if(!fixed_flag[offset_2])
							{
								f[x_index[offset_2]] -= 2*cur_v[0];
								f[y_index[offset_2]] -= 2*cur_v[1];
								f[z_index[offset_2]] -= 2*cur_v[2];
							}
						}
					}
				}
			}
		}


		for(int i = 0;i < unknown_num;i++)
			f[i] *= -0.5;

		double tol = 1e-16;
		T* unknown_vals = new T[unknown_num];
		int it = 0;
		ZQ_PCGSolver::PCG(G,f,init_unknown,iteration,tol,unknown_vals,it,true);

		for(int i = 0;i < width*height*depth;i++)
		{
			if(nouseful_flag[i])
				continue;
			if(fixed_flag[i])
			{
				out_coord[i*3] = init_coord[i*3];
				out_coord[i*3+1] = init_coord[i*3+1];
				out_coord[i*3+2] = init_coord[i*3+2];
			}
			else
			{
				out_coord[i*3] = unknown_vals[x_index[i]];
				out_coord[i*3+1] = unknown_vals[y_index[i]];
				out_coord[i*3+2] = unknown_vals[z_index[i]];
			}
		}

		delete []f;
		delete []unknown_vals;
		delete []init_unknown;
		delete []fixed_val;

		return true;
	}
}


#endif