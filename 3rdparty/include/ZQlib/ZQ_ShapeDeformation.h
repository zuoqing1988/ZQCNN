#ifndef _ZQ_SHAPE_DEFORMATION_H_
#define _ZQ_SHAPE_DEFORMATION_H_
#pragma once

#include "ZQ_ShapeDeformationOptions.h"
#include "ZQ_PCGSolver.h"
#include "ZQ_SparseMatrix.h"
#include "ZQ_Matrix.h"
#include "ZQ_MathBase.h"
#include "ZQ_SVD.h"
#include <vector>
#include <map>
#include <typeinfo>

namespace ZQ
{
	/*
	I write the method ARAP_Triangle referring to the paper:
	As-Rigid-As-Possible Shape Manipulation, 2005.
	The ARAP_Vert is similary to ARAP_Triangle.
	*/
	template<class T>
	class ZQ_ShapeDeformation
	{
		struct _mat22
		{
			T val[4];
		};
	public:
		ZQ_ShapeDeformation();
		~ZQ_ShapeDeformation();

	private:
		int triangle_num;
		int point_num;
		bool* fixed_flag;
		int fixed_num;
		int unknown_num;
		int* x_index;
		int* y_index;
		int* indices;
		T* start_pos;
		std::vector<std::map<int,T>> oneloop_neighbor;

		taucs_ccs_matrix* H;
		taucs_ccs_matrix* G;

		ZQ_ShapeDeformationOptions option;

	public:
		bool BuildMatrix(int nTri, const int* indices, int nPts, const T* verts, const bool* fixflag, const ZQ_ShapeDeformationOptions& opt);
		bool Deformation(const T* init_coords, T* out_coords);

		bool _deformationLaplacian(const T* init_coords, T* out_coords, int max_iter);
		bool _deformationARAP_Vert(const T* init_coords, T* out_coords, int FP_iter, int max_iter, bool with_good_init = false);
		bool _deformationARAP_Triangle(const T* init_coords, T* out_coords, int FP_iter, int max_iter, bool with_good_init = false);

	private:
		void _clear();
		bool _build_oneloop_neighbor();

		bool _build_matrix_for_ARAP_VERT();
		bool _build_matrix_for_ARAP_TRIANGLE();
		bool _estimate_R_for_ARAP_VERT(const T* init_coords, std::vector<_mat22>& Rmats);
		bool _estimate_R_for_ARAP_TRIANGLE(const T* init_coords, std::vector<_mat22>& Rmats);
		bool _solve_for_ARAP_VERT(const T* init_coords, const std::vector<_mat22>& Rmats, T* out_coords, int max_iter);
		bool _solve_for_ARAP_TRIANGLE(const T* init_coords, const std::vector<_mat22>& Rmats, T* out_coords, int max_iter);
	};


	/*******************************   definitions   **********************************/

	template<class T>
	ZQ_ShapeDeformation<T>::ZQ_ShapeDeformation()
	{
		triangle_num = 0;
		point_num = 0;
		fixed_num = 0;
		fixed_flag = 0;
		x_index = 0;
		y_index = 0;
		indices = 0;
		start_pos = 0;
		H = 0;
		G = 0;
	}

	template<class T>
	ZQ_ShapeDeformation<T>::~ZQ_ShapeDeformation()
	{
		_clear();
	}

	template<class T>
	bool ZQ_ShapeDeformation<T>::BuildMatrix(int nTri, const int* indices, int nPts, const T* verts, const bool* fixflag, const ZQ_ShapeDeformationOptions& opt)
	{
		if(nTri <= 0 || nPts <= 0 || indices == 0 || verts == 0 || fixflag == 0)
			return false;

		_clear();

		this->triangle_num = nTri;
		this->point_num = nPts;
		this->indices = new int[nTri*3];
		this->start_pos = new T[nPts*2];
		this->fixed_flag = new bool[nPts];
		memcpy(this->indices,indices,sizeof(int)*3*nTri);
		memcpy(this->fixed_flag,fixflag,sizeof(bool)*nPts);
		memcpy(this->start_pos,verts,sizeof(T)*nPts*2);
		option = opt;

		fixed_num = 0;
		for(int pp = 0;pp < point_num;pp++)
		{
			if(fixed_flag[pp])
				fixed_num++;
		}
		if(!_build_oneloop_neighbor())
		{
			_clear();
			return false;
		}

		unknown_num = (point_num-fixed_num);
		x_index = new int[point_num];
		y_index = new int[point_num];
		int cur_unknown_idx = 0;
		int cur_fixed_idx = 0;
		for(int pp = 0;pp < point_num;pp++)
		{
			if(fixed_flag[pp])
			{
				x_index[pp] = cur_fixed_idx++;
				y_index[pp] = cur_fixed_idx++;
			}
			else
			{
				x_index[pp] = cur_unknown_idx++;
				y_index[pp] = cur_unknown_idx++;
			}
		}

		switch (opt.methodType)
		{
		case ZQ_ShapeDeformationOptions::METHOD_LAPLACIAN:
			{
			}
			break;
		case ZQ_ShapeDeformationOptions::METHOD_ARAP_VERT_AS_CENTER:
			{
				_build_matrix_for_ARAP_VERT();
			}
			break;
		case ZQ_ShapeDeformationOptions::METHOD_ARAP_TRIANGLE_AS_CENTER:
			{
				_build_matrix_for_ARAP_TRIANGLE();
			}
			break;
		}
		return true;
	}


	template<class T>
	bool ZQ_ShapeDeformation<T>::Deformation(const T* init_coords, T* out_coords)
	{
		switch(option.methodType)
		{
		case ZQ_ShapeDeformationOptions::METHOD_LAPLACIAN:
			{
				return _deformationLaplacian(init_coords,out_coords,option.Iteration);
			}
			break;
		case ZQ_ShapeDeformationOptions::METHOD_ARAP_VERT_AS_CENTER:
			{
				return _deformationARAP_Vert(init_coords,out_coords,option.FPIteration,option.Iteration,false);
			}
			break;
		case ZQ_ShapeDeformationOptions::METHOD_ARAP_TRIANGLE_AS_CENTER:
			{
				return _deformationARAP_Triangle(init_coords,out_coords,option.FPIteration,option.Iteration,false);
			}
			break;
		}
		return false;
	}

	template<class T>
	bool ZQ_ShapeDeformation<T>::_deformationLaplacian(const T* init_coords, T* out_coords, int max_iter)
	{
		int taucs_flag;
		if(strcmp(typeid(T).name(),"float") == 0)
			taucs_flag = TAUCS_SINGLE;
		else if(strcmp(typeid(T).name(),"double") == 0)
			taucs_flag = TAUCS_DOUBLE;
		else 
			return false;

		if(triangle_num == 0)
			return false;

		T* neighbor_weight = new T[point_num];
		T* laplace_x = new T[point_num];
		T* laplace_y = new T[point_num];
		for(int pp = 0;pp < point_num;pp++)
		{
			if(oneloop_neighbor[pp].size() == 0)
			{
				delete []neighbor_weight;
				delete []laplace_x;
				delete []laplace_y;
				return false;
			}
			neighbor_weight[pp] = 1.0/oneloop_neighbor[pp].size();
		}

		std::map<int,T>::iterator map_it;
		for(int pp = 0;pp < point_num;pp++)
		{
			float cur_x = start_pos[pp*2+0];
			float cur_y = start_pos[pp*2+1];
			float sum_x = 0;
			float sum_y = 0;
			for(map_it = oneloop_neighbor[pp].begin();map_it != oneloop_neighbor[pp].end();++map_it)
			{
				int id = map_it->first;
				sum_x += start_pos[id*2+0];
				sum_y += start_pos[id*2+1];
			}
			sum_x *= neighbor_weight[pp];
			sum_y *= neighbor_weight[pp];
			laplace_x[pp] = sum_x-cur_x;
			laplace_y[pp] = sum_y-cur_y;
		}

		ZQ_SparseMatrix<T> Amat(point_num*2,unknown_num*2);
		T* b = new T[point_num*2];
		for(int pp = 0;pp < point_num;pp++)
		{
			b[pp*2+0] = laplace_x[pp];
			b[pp*2+1] = laplace_y[pp];
			if(fixed_flag[pp])
			{
				b[pp*2+0] += init_coords[pp*2+0];
				b[pp*2+1] += init_coords[pp*2+1];
			}
			else
			{
				Amat.AddTo(pp*2+0,x_index[pp],-1);
				Amat.AddTo(pp*2+1,y_index[pp],-1);
			}

			for(map_it = oneloop_neighbor[pp].begin();map_it != oneloop_neighbor[pp].end();++map_it)
			{
				int id = map_it->first;
				if(fixed_flag[id])
				{
					b[pp*2+0] -= neighbor_weight[pp]*init_coords[id*2+0];
					b[pp*2+1] -= neighbor_weight[pp]*init_coords[id*2+1];
				}
				else
				{
					Amat.AddTo(pp*2+0,x_index[id],neighbor_weight[pp]);
					Amat.AddTo(pp*2+1,y_index[id],neighbor_weight[pp]);
				}
			}
		}

		taucs_ccs_matrix* A = Amat.ExportCCS(taucs_flag);

		double tol = 1e-16;
		int it = 0;

		T* x0 = new T[unknown_num*2];
		T* x = new T[unknown_num*2];
		memset(x0,0,sizeof(T)*unknown_num*2);
		ZQ_PCGSolver::PCG_sparse_unsquare(A,b,x0,max_iter,tol,x,it);
		for(int pp = 0;pp < point_num;pp++)
		{
			if(fixed_flag[pp])
			{
				out_coords[pp*2+0] = init_coords[pp*2+0];
				out_coords[pp*2+1] = init_coords[pp*2+1];
			}
			else
			{
				out_coords[pp*2+0] = x[x_index[pp]];
				out_coords[pp*2+1] = x[y_index[pp]];
			}
		}

		delete []x;
		delete []x0;
		delete []b;
		delete []laplace_x;
		delete []laplace_y;
		delete []neighbor_weight;
		ZQ_TaucsBase::ZQ_taucs_ccs_free(A);
		return true;
	}

	template<class T>
	bool ZQ_ShapeDeformation<T>::_deformationARAP_Vert(const T* init_coords, T* out_coords, int FP_iter, int max_iter, bool with_good_init /* = false */)
	{
		if(triangle_num == 0)
			return false;

		T* tmp_coords = new T[point_num*2];
		if(!with_good_init)
		{
			if(!_deformationLaplacian(init_coords,tmp_coords,max_iter))
			{
				delete []tmp_coords;
				return false;
			}
		}
		else
		{
			memcpy(tmp_coords,init_coords,sizeof(T)*point_num*2);
		}

		std::vector<_mat22> Rmats;
		for(int fp_it = 0;fp_it < FP_iter;fp_it++)
		{
			_estimate_R_for_ARAP_VERT(tmp_coords,Rmats);
			_solve_for_ARAP_VERT(tmp_coords,Rmats,out_coords,max_iter);
			memcpy(tmp_coords,out_coords,sizeof(T)*point_num*2);
		}

		delete []tmp_coords;
		return true;
	}


	template<class T>
	bool ZQ_ShapeDeformation<T>::_deformationARAP_Triangle(const T* init_coords, T* out_coords, int FP_iter, int max_iter, bool with_good_init /* = false */)
	{
		if(triangle_num == 0)
			return false;

		T* tmp_coords = new T[point_num*2];
		if(!with_good_init)
		{
			if(!_deformationLaplacian(init_coords,tmp_coords,max_iter))
			{
				delete []tmp_coords;
				return false;
			}
		}
		else
		{
			memcpy(tmp_coords,init_coords,sizeof(T)*point_num*2);
		}

		std::vector<_mat22> Rmats;
		for(int fp_it = 0;fp_it < FP_iter;fp_it++)
		{
			_estimate_R_for_ARAP_TRIANGLE(tmp_coords,Rmats);
			_solve_for_ARAP_TRIANGLE(tmp_coords,Rmats,out_coords,max_iter);
			memcpy(tmp_coords,out_coords,sizeof(T)*point_num*2);
		}
		delete []tmp_coords;

		return true;
	}

	template<class T>
	void ZQ_ShapeDeformation<T>::_clear()
	{
		triangle_num = 0;
		point_num = 0;
		fixed_num = 0;
		unknown_num = 0;
		if(fixed_flag)
		{
			delete []fixed_flag;
			fixed_flag = 0;
		}
		if(indices)
		{
			delete []indices;
			indices = 0;
		}
		if(start_pos)
		{
			delete []start_pos;
			start_pos = 0;
		}
		if(H)
		{
			ZQ_TaucsBase::ZQ_taucs_ccs_free(H);
			H = 0;
		}
		if(G)
		{
			ZQ_TaucsBase::ZQ_taucs_ccs_free(G);
			G = 0;
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
		oneloop_neighbor.clear();
	}

	template<class T>
	bool ZQ_ShapeDeformation<T>::_build_oneloop_neighbor()
	{
		oneloop_neighbor.clear();
		oneloop_neighbor.resize(point_num);

		for(int tr = 0;tr < triangle_num;tr++)
		{
			int id0 = indices[tr*3+0];
			int id1 = indices[tr*3+1];
			int id2 = indices[tr*3+2];
			oneloop_neighbor[id0][id1] = 0;
			oneloop_neighbor[id0][id2] = 0;
			oneloop_neighbor[id1][id0] = 0;
			oneloop_neighbor[id1][id2] = 0;
			oneloop_neighbor[id2][id0] = 0;
			oneloop_neighbor[id2][id0] = 0;
		}

		for(int i = 0;i < point_num;i++)
		{
			if(oneloop_neighbor[i].size() == 0)
				return false;
		}

		for(int tr = 0;tr < triangle_num;tr++)
		{
			int id0 = indices[tr*3+0];
			int id1 = indices[tr*3+1];
			int id2 = indices[tr*3+2];

			T v0_1[2] = {start_pos[id1*2+0]-start_pos[id0*2+0],start_pos[id1*2+1]-start_pos[id0*2+1]};
			T v0_2[2] = {start_pos[id2*2+0]-start_pos[id0*2+0],start_pos[id2*2+1]-start_pos[id0*2+1]};
			T v1_2[2] = {start_pos[id2*2+0]-start_pos[id1*2+0],start_pos[id2*2+1]-start_pos[id1*2+1]};
			T v1_0[2] = {-v0_1[0],-v0_1[1]};
			T v2_0[2] = {-v0_2[0],-v0_2[1]};
			T v2_1[2] = {-v1_2[0],-v1_2[1]};

			T len0_1 = sqrt(v0_1[0]*v0_1[0]+v0_1[1]*v0_1[1]);
			T len1_2 = sqrt(v1_2[0]*v1_2[0]+v1_2[1]*v1_2[1]);
			T len2_0 = sqrt(v2_0[0]*v2_0[0]+v2_0[1]*v2_0[1]);

			if(len0_1 == 0 || len1_2 == 0 || len2_0 == 0)
				return false;

			T cos0 = (v0_1[0]*v0_2[0]+v0_1[1]*v0_2[1])/(len0_1*len2_0);
			T cos1 = (v1_0[0]*v1_2[0]+v1_0[1]*v1_2[1])/(len0_1*len1_2);
			T cos2 = (v2_0[0]*v2_1[0]+v2_0[1]*v2_1[1])/(len1_2*len2_0);

			/*T angle0 = 0.5*acos(cos0);
			T angle1 = 0.5*acos(cos1);
			T angle2 = 0.5*acos(cos2);

			T cot0 = cos(angle0)/sin(angle0);
			T cot1 = cos(angle1)/sin(angle1);
			T cot2 = cos(angle2)/sin(angle2);*/

			T angle0 = acos(cos0);
			T angle1 = acos(cos1);
			T angle2 = acos(cos2);

			T cot0 = cos0/sin(angle0);
			T cot1 = cos1/sin(angle1);
			T cot2 = cos2/sin(angle2);

			oneloop_neighbor[id0][id1] += 0.5*cot2;
			oneloop_neighbor[id1][id0] += 0.5*cot2;
			oneloop_neighbor[id0][id2] += 0.5*cot1;
			oneloop_neighbor[id2][id0] += 0.5*cot1;
			oneloop_neighbor[id1][id2] += 0.5*cot0;
			oneloop_neighbor[id2][id1] += 0.5*cot0;
		}

		std::map<int,T>::iterator map_it;
		for(int pp = 0;pp < point_num;pp++)
		{
			T sum_weight = 0;
			for(map_it = oneloop_neighbor[pp].begin(); map_it != oneloop_neighbor[pp].end(); ++map_it)
			{
				sum_weight += map_it->second;
			}
			for(map_it = oneloop_neighbor[pp].begin(); map_it != oneloop_neighbor[pp].end(); ++map_it)
			{
				map_it->second /= sum_weight;
			}
		}

		return true;
	}

	template<class T>
	bool ZQ_ShapeDeformation<T>::_build_matrix_for_ARAP_VERT()
	{
		int taucs_flag;
		if(strcmp(typeid(T).name(),"float") == 0)
			taucs_flag = TAUCS_SINGLE;
		else if(strcmp(typeid(T).name(),"double") == 0)
			taucs_flag = TAUCS_DOUBLE;
		else
			return false;
		
		ZQ_SparseMatrix<T> Hmat(unknown_num*2,unknown_num*2);
		ZQ_SparseMatrix<T> Gmat(unknown_num*2,fixed_num*2);

		std::map<int,T>::iterator map_it;
		for(int pp = 0;pp < point_num;pp++)
		{
			for(map_it = oneloop_neighbor[pp].begin(); map_it != oneloop_neighbor[pp].end(); ++map_it)
			{
				int id = map_it->first;
				T weight = oneloop_neighbor[pp][id];

				if(!fixed_flag[pp])
				{
					if(!fixed_flag[id])
					{
						Hmat.AddTo(x_index[pp],x_index[pp],weight);
						Hmat.AddTo(y_index[pp],y_index[pp],weight);

						Hmat.AddTo(x_index[id],x_index[id],weight);
						Hmat.AddTo(y_index[id],y_index[id],weight);

						Hmat.AddTo(x_index[pp],x_index[id],-weight);
						Hmat.AddTo(x_index[id],x_index[pp],-weight);
						Hmat.AddTo(y_index[pp],y_index[id],-weight);
						Hmat.AddTo(y_index[id],y_index[pp],-weight);
					}
					else
					{
						Hmat.AddTo(x_index[pp],x_index[pp],weight);
						Hmat.AddTo(y_index[pp],y_index[pp],weight);

						Gmat.AddTo(x_index[pp],x_index[id],-2*weight);
						Gmat.AddTo(y_index[pp],y_index[id],-2*weight);
					}
				}
				else
				{
					if(!fixed_flag[id])
					{					
						Hmat.AddTo(x_index[id],x_index[id],weight);
						Hmat.AddTo(y_index[id],y_index[id],weight);

						Gmat.AddTo(x_index[id],x_index[pp],-2*weight);
						Gmat.AddTo(y_index[id],y_index[pp],-2*weight);
					}
					else
					{

					}
				}
			}
		}

		H = Hmat.ExportCCS(taucs_flag);
		G = Gmat.ExportCCS(taucs_flag);

		return true;
	}

	template<class T>
	bool ZQ_ShapeDeformation<T>::_build_matrix_for_ARAP_TRIANGLE()
	{
		int taucs_flag;
		if(strcmp(typeid(T).name(),"float") == 0)
			taucs_flag = TAUCS_SINGLE;
		else if(strcmp(typeid(T).name(),"double") == 0)
			taucs_flag = TAUCS_DOUBLE;
		else
			return false;

		ZQ_SparseMatrix<T> Hmat(unknown_num*2,unknown_num*2);
		ZQ_SparseMatrix<T> Gmat(unknown_num*2,fixed_num*2);

		for(int tr = 0;tr < triangle_num;tr++)
		{
			int id0 = indices[tr*3+0];
			int id1 = indices[tr*3+1];
			int id2 = indices[tr*3+2];

			/*
			* (p1-p2-v12)^2 + (p2-p3-v23)^2 + (p3-p1-v31)^2
			*
			*/

			if(!fixed_flag[id0])
			{
				if(!fixed_flag[id1])
				{
					if(!fixed_flag[id2])
					{
						Hmat.AddTo(x_index[id0],x_index[id0],2);
						Hmat.AddTo(y_index[id0],y_index[id0],2);

						Hmat.AddTo(x_index[id1],x_index[id1],2);
						Hmat.AddTo(y_index[id1],y_index[id1],2);

						Hmat.AddTo(x_index[id2],x_index[id2],2);
						Hmat.AddTo(y_index[id2],y_index[id2],2);

						Hmat.AddTo(x_index[id0],x_index[id1],-1);
						Hmat.AddTo(x_index[id1],x_index[id0],-1);
						Hmat.AddTo(y_index[id0],y_index[id1],-1);
						Hmat.AddTo(y_index[id1],y_index[id0],-1);

						Hmat.AddTo(x_index[id0],x_index[id2],-1);
						Hmat.AddTo(x_index[id2],x_index[id0],-1);
						Hmat.AddTo(y_index[id0],y_index[id2],-1);
						Hmat.AddTo(y_index[id2],y_index[id0],-1);

						Hmat.AddTo(x_index[id1],x_index[id2],-1);
						Hmat.AddTo(x_index[id2],x_index[id1],-1);
						Hmat.AddTo(y_index[id1],y_index[id2],-1);
						Hmat.AddTo(y_index[id2],y_index[id1],-1);
					}
					else
					{
						Hmat.AddTo(x_index[id0],x_index[id0],2);
						Hmat.AddTo(y_index[id0],y_index[id0],2);

						Hmat.AddTo(x_index[id1],x_index[id1],2);
						Hmat.AddTo(y_index[id1],y_index[id1],2);

						Hmat.AddTo(x_index[id0],x_index[id1],-1);
						Hmat.AddTo(x_index[id1],x_index[id0],-1);
						Hmat.AddTo(y_index[id0],y_index[id1],-1);
						Hmat.AddTo(y_index[id1],y_index[id0],-1);

						Gmat.AddTo(x_index[id0],x_index[id2],-2);
						Gmat.AddTo(y_index[id0],y_index[id2],-2);

						Gmat.AddTo(x_index[id1],x_index[id2],-2);
						Gmat.AddTo(y_index[id1],y_index[id2],-2);
					}
				}
				else 
				{
					if(!fixed_flag[id2])
					{
						Hmat.AddTo(x_index[id0],x_index[id0],2);
						Hmat.AddTo(y_index[id0],y_index[id0],2);

						Hmat.AddTo(x_index[id2],x_index[id2],2);
						Hmat.AddTo(y_index[id2],y_index[id2],2);

						Gmat.AddTo(x_index[id0],x_index[id1],-2);
						Gmat.AddTo(y_index[id0],y_index[id1],-2);

						Hmat.AddTo(x_index[id0],x_index[id2],-1);
						Hmat.AddTo(x_index[id2],x_index[id0],-1);
						Hmat.AddTo(y_index[id0],y_index[id2],-1);
						Hmat.AddTo(y_index[id2],y_index[id0],-1);

						Gmat.AddTo(x_index[id2],x_index[id1],-2);
						Gmat.AddTo(y_index[id2],y_index[id1],-2);
					}
					else
					{
						Hmat.AddTo(x_index[id0],x_index[id0],2);
						Hmat.AddTo(y_index[id0],y_index[id0],2);

						Gmat.AddTo(x_index[id0],x_index[id1],-2);
						Gmat.AddTo(y_index[id0],y_index[id1],-2);

						Gmat.AddTo(x_index[id0],x_index[id2],-2);
						Gmat.AddTo(y_index[id0],y_index[id2],-2);
					}
				}
			}
			else
			{
				if(!fixed_flag[id1])
				{
					if(!fixed_flag[id2])
					{
						Hmat.AddTo(x_index[id1],x_index[id1],2);
						Hmat.AddTo(y_index[id1],y_index[id1],2);

						Hmat.AddTo(x_index[id2],x_index[id2],2);
						Hmat.AddTo(y_index[id2],y_index[id2],2);

						Gmat.AddTo(x_index[id1],x_index[id0],-2);
						Gmat.AddTo(y_index[id1],y_index[id0],-2);

						Gmat.AddTo(x_index[id2],x_index[id0],-2);
						Gmat.AddTo(y_index[id2],y_index[id0],-2);

						Hmat.AddTo(x_index[id1],x_index[id2],-1);
						Hmat.AddTo(x_index[id2],x_index[id1],-1);
						Hmat.AddTo(y_index[id1],y_index[id2],-1);
						Hmat.AddTo(y_index[id2],y_index[id1],-1);
					}
					else
					{
						Hmat.AddTo(x_index[id1],x_index[id1],2);
						Hmat.AddTo(y_index[id1],y_index[id1],2);

						Gmat.AddTo(x_index[id1],x_index[id0],-2);
						Gmat.AddTo(x_index[id1],x_index[id0],-2);

						Gmat.AddTo(x_index[id1],x_index[id2],-2);
						Gmat.AddTo(y_index[id1],y_index[id2],-2);
					}
				}
				else 
				{
					if(!fixed_flag[id2])
					{
						Hmat.AddTo(x_index[id2],x_index[id2],2);
						Hmat.AddTo(y_index[id2],y_index[id2],2);

						Gmat.AddTo(x_index[id2],x_index[id0],-2);
						Gmat.AddTo(y_index[id2],y_index[id0],-2);

						Gmat.AddTo(x_index[id2],x_index[id1],-2);
						Gmat.AddTo(y_index[id2],y_index[id1],-2);
					}
					else
					{

					}
				}
			}
		}

		H = Hmat.ExportCCS(taucs_flag);
		G = Gmat.ExportCCS(taucs_flag);

		return true;
	}

	template<class T>
	bool  ZQ_ShapeDeformation<T>::_estimate_R_for_ARAP_VERT(const T* init_coords, std::vector<_mat22>& Rmats)
	{
		Rmats.clear();
		Rmats.resize(point_num);
		std::map<int,T>::iterator map_it;
		for(int pp = 0;pp < point_num;pp++)
		{
			T S[4] = {0,0,0,0};
			T old_x = start_pos[pp*2+0];
			T old_y = start_pos[pp*2+1];
			T cur_x = init_coords[pp*2+0];
			T cur_y = init_coords[pp*2+1];
			for(map_it = oneloop_neighbor[pp].begin();map_it != oneloop_neighbor[pp].end();++map_it)
			{
				int id = map_it->first;
				T weight = oneloop_neighbor[pp][id];
				T old_vx = start_pos[id*2+0] - old_x;
				T old_vy = start_pos[id*2+1] - old_y;
				T cur_vx = init_coords[id*2+0] - cur_x;
				T cur_vy = init_coords[id*2+1] - cur_y;
				S[0] += weight*old_vx*cur_vx;
				S[1] += weight*old_vx*cur_vy;
				S[2] += weight*old_vy*cur_vx;
				S[3] += weight*old_vy*cur_vy;
			}
			ZQ_Matrix<T> Smat(2,2), U(2,2),D(2,2),V(2,2);
			Smat.SetData(0,0,S[0]);
			Smat.SetData(0,1,S[1]);
			Smat.SetData(1,0,S[2]);
			Smat.SetData(1,1,S[3]);

			ZQ_SVD::Decompose(Smat,U,D,V);

			ZQ_Matrix<T> Rmat = V*U.GetTransposeMatrix();

			bool flag;
			Rmats[pp].val[0] = Rmat.GetData(0,0,flag);
			Rmats[pp].val[1] = Rmat.GetData(0,1,flag);
			Rmats[pp].val[2] = Rmat.GetData(1,0,flag);
			Rmats[pp].val[3] = Rmat.GetData(1,1,flag);
		}

		return true;
	}


	template<class T>
	bool ZQ_ShapeDeformation<T>::_estimate_R_for_ARAP_TRIANGLE(const T* init_coords, std::vector<_mat22>& Rmats)
	{
		Rmats.clear();
		Rmats.resize(triangle_num);

		for(int tr = 0;tr < triangle_num;tr++)
		{
			int id0 = indices[tr*3+0];
			int id1 = indices[tr*3+1];
			int id2 = indices[tr*3+2];

			T pp[6] = 
			{
				start_pos[id0*2+0]-start_pos[id1*2+0],
				start_pos[id0*2+1]-start_pos[id1*2+1],
				start_pos[id1*2+0]-start_pos[id2*2+0],
				start_pos[id1*2+1]-start_pos[id2*2+1],
				start_pos[id2*2+0]-start_pos[id0*2+0],
				start_pos[id2*2+1]-start_pos[id0*2+1]
			};

			T qq[6] = 
			{
				init_coords[id0*2+0]-init_coords[id1*2+0],
				init_coords[id0*2+1]-init_coords[id1*2+1],
				init_coords[id1*2+0]-init_coords[id2*2+0],
				init_coords[id1*2+1]-init_coords[id2*2+1],
				init_coords[id2*2+0]-init_coords[id0*2+0],
				init_coords[id2*2+1]-init_coords[id0*2+1]
			};

			ZQ_Matrix<T> coeff(2,2);

			coeff.SetData(0,0,pp[0]*qq[0]+pp[2]*qq[2]+pp[4]*qq[4]);
			coeff.SetData(0,1,pp[0]*qq[1]+pp[2]*qq[3]+pp[4]*qq[5]);
			coeff.SetData(1,0,pp[1]*qq[0]+pp[3]*qq[2]+pp[5]*qq[4]);
			coeff.SetData(1,1,pp[1]*qq[1]+pp[3]*qq[3]+pp[5]*qq[5]);

			ZQ_Matrix<T> U(2,2),D(2,2),V(2,2);

			ZQ_SVD::Decompose(coeff,U,D,V);

			ZQ_Matrix<T> Rmat = V*U.GetTransposeMatrix();

			bool flag;
			Rmats[tr].val[0] = Rmat.GetData(0,0,flag);
			Rmats[tr].val[1] = Rmat.GetData(0,1,flag);
			Rmats[tr].val[2] = Rmat.GetData(1,0,flag);
			Rmats[tr].val[3] = Rmat.GetData(1,1,flag);
		}

		return true;
	}

	template<class T>
	bool ZQ_ShapeDeformation<T>::_solve_for_ARAP_VERT(const T* init_coords, const std::vector<_mat22>& Rmats, T* out_coords, int max_iter)
	{
		T* fixed_val = new T[fixed_num*2];
		for(int pp = 0;pp < point_num;pp++)
		{
			if(fixed_flag[pp])
			{
				fixed_val[x_index[pp]] = init_coords[pp*2];
				fixed_val[y_index[pp]] = init_coords[pp*2+1];
			}
		}

		T* f = new T[unknown_num*2];

		ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec(G,fixed_val,f);

		std::map<int,T>::iterator map_it;
		for(int pp = 0;pp < point_num;pp++)
		{
			T old_x = start_pos[pp*2+0];
			T old_y = start_pos[pp*2+1];
			for(map_it = oneloop_neighbor[pp].begin(); map_it != oneloop_neighbor[pp].end(); ++map_it)
			{
				int id = map_it->first;
				T weight = oneloop_neighbor[pp][id];
				T old_vx = start_pos[id*2+0] - old_x;
				T old_vy = start_pos[id*2+1] - old_y;
				T vx = Rmats[pp].val[0] * old_vx + Rmats[pp].val[1] * old_vy;
				T vy = Rmats[pp].val[2] * old_vx + Rmats[pp].val[3] * old_vy;

				if(!fixed_flag[pp])
				{
					if(!fixed_flag[id])
					{
						f[x_index[pp]] += 2*weight*vx;
						f[y_index[pp]] += 2*weight*vy;

						f[x_index[id]] += -2*weight*vx;
						f[y_index[id]] += -2*weight*vy;
					}
					else
					{
						f[x_index[pp]] += 2*weight*vx;
						f[y_index[pp]] += 2*weight*vy;
					}
				}
				else
				{
					if(!fixed_flag[id])
					{
						f[x_index[id]] += -2*weight*vx;
						f[y_index[id]] += -2*weight*vy;
					}
					else
					{

					}
				}
			}
		}

		for(int i = 0;i < unknown_num*2;i++)
			f[i] *= -0.5;

		double tol = 1e-16;
		int it = 0;
		T* x0 = new T[unknown_num*2];
		T* x = new T[unknown_num*2];
		memset(x0,0,sizeof(T)*unknown_num*2);
		ZQ_PCGSolver::PCG(H,f,x0,max_iter,tol,x,it);

		for(int pp = 0;pp < point_num;pp++)
		{
			if(fixed_flag[pp])
			{
				out_coords[pp*2+0] = init_coords[pp*2+0];
				out_coords[pp*2+1] = init_coords[pp*2+1];
			}
			else
			{
				out_coords[pp*2+0] = x[x_index[pp]];
				out_coords[pp*2+1] = x[y_index[pp]];
			}
		}

		delete []x;
		delete []x0;
		delete []f;
		delete []fixed_val;
		return true;
	}

	template<class T>
	bool ZQ_ShapeDeformation<T>::_solve_for_ARAP_TRIANGLE(const T* init_coords, const std::vector<_mat22>& Rmats, T* out_coords, int max_iter)
	{
		T* fixed_val = new T[fixed_num*2];
		for(int pp = 0;pp < point_num;pp++)
		{
			if(fixed_flag[pp])
			{
				fixed_val[x_index[pp]] = init_coords[pp*2];
				fixed_val[y_index[pp]] = init_coords[pp*2+1];
			}
		}

		T* f = new T[unknown_num*2];

		ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec(G,fixed_val,f);

		for(int tr = 0;tr < triangle_num;tr++)
		{
			int id0 = indices[tr*3+0];
			int id1 = indices[tr*3+1];
			int id2 = indices[tr*3+2];

			T pp[6] = 
			{
				start_pos[id0*2+0]-start_pos[id1*2+0],
				start_pos[id0*2+1]-start_pos[id1*2+1],
				start_pos[id1*2+0]-start_pos[id2*2+0],
				start_pos[id1*2+1]-start_pos[id2*2+1],
				start_pos[id2*2+0]-start_pos[id0*2+0],
				start_pos[id2*2+1]-start_pos[id0*2+1]
			};

			T qq[6];

			ZQ_MathBase::MatrixMul(Rmats[tr].val,pp,2,2,3,qq);

			if(!fixed_flag[id0])
			{
				f[x_index[id0]] -= 2*qq[0];
				f[y_index[id0]] -= 2*qq[1];
				f[x_index[id0]] += 2*qq[4];
				f[y_index[id0]] += 2*qq[5];
			}
			if(!fixed_flag[id1])
			{
				f[x_index[id1]] += 2*qq[0];
				f[y_index[id1]] += 2*qq[1];
				f[x_index[id1]] -= 2*qq[2];
				f[y_index[id1]] -= 2*qq[3];
			}
			if(!fixed_flag[id2])
			{
				f[x_index[id2]] += 2*qq[2];
				f[y_index[id2]] += 2*qq[3];
				f[x_index[id2]] -= 2*qq[4];
				f[y_index[id2]] -= 2*qq[5];
			}
		}

		for(int i = 0;i < unknown_num*2;i++)
			f[i] *= -0.5;

		double tol = 1e-16;
		int it = 0;
		T* x0 = new T[unknown_num*2];
		T* x = new T[unknown_num*2];
		memset(x0,0,sizeof(T)*unknown_num*2);
		
		ZQ_PCGSolver::PCG(H,f,x0,max_iter,tol,x,it);

		for(int pp = 0;pp < point_num;pp++)
		{
			if(fixed_flag[pp])
			{
				out_coords[pp*2+0] = init_coords[pp*2+0];
				out_coords[pp*2+1] = init_coords[pp*2+1];
			}
			else
			{
				out_coords[pp*2+0] = x[x_index[pp]];
				out_coords[pp*2+1] = x[y_index[pp]];
			}
		}

		delete []x;
		delete []x0;
		delete []f;
		delete []fixed_val;
		return true;
	}

}

#endif