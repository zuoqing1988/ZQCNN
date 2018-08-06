#ifndef _ZQ_GRID_DEFORMATION_H_
#define _ZQ_GRID_DEFORMATION_H_
#pragma once

#include "ZQ_GridDeformationOptions.h"
#include "ZQ_SparseMatrix.h"
#include "ZQ_PCGSolver.h"
#include "ZQ_DoubleImage.h"
#include "ZQ_SVD.h"
#include "ZQ_MathBase.h"
#include "ZQ_Matrix.h"
#include <typeinfo>

namespace ZQ
{
	/******************************************************
	/* Referred to the paper:
	/* As-Rigid-As-Possible Shape Manipulation, 2005
	/* Section 4.1: Step one: scale-free construction
	/* Section 4.2: Step twp: scale adjustment
	/*
	/* *********************************
	/*
	/* As-Rigid-As-Possible Surface Modeling, 2007
	/* 
	/******************************************************/

	template<class T>
	class ZQ_GridDeformation
	{
		struct _mat22
		{
			T val[4];
		};

	public:
		ZQ_GridDeformation();
		~ZQ_GridDeformation();

	private:
		int width,height;
		bool* nouseful_flag;   //true: means the point is blank
		bool* fixed_flag;      //true: means the point is control point
		int* x_index;
		int* y_index;
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

		ZQ_GridDeformationOptions option;
		

	public:
		void ClearMatrix();
		bool BuildMatrix(const int width, const int height, const bool* nouseful_flag, const bool* fixed_flag, const ZQ_GridDeformationOptions& opt);

		bool Deformation(const T* init_coord, T* out_coord, bool has_good_init = false);
	public:

		bool _deformation_without_scaling(const T* init_coord, T* out_coord, const int iteration, int& it);

		/*should be called after deformation_without_scaling*/
		bool _scaling(const T* init_coord, T* out_coord, const int iteration);

		bool _deformation_with_distance(const T* init_coord, T* out_coord, const int nFPIter, const int iteration);
		bool _deformation_with_distance_XLOOP(const T* init_coord, T* out_coord, const int nFPIter, const int iteration);

		bool _deformation_ARAP_VERT(const T* init_coord, T* out_coord, const int nFPIter, const int iteration);
		bool _deformation_ARAP_VERT_XLOOP(const T* init_coord, T* out_coord, const int nFPIter, const int iteration);

	private:
		bool _addMatrixTo_LINE(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight,
			const ZQ_GridDeformationOptions::NeighborType type = ZQ_GridDeformationOptions::NEIGHBOR_4);
		bool _addMatrixTo_LINE_XLOOP(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight,
			const ZQ_GridDeformationOptions::NeighborType type = ZQ_GridDeformationOptions::NEIGHBOR_4);
		bool _addMatrixTo_ANGLE(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight, 
			const ZQ_GridDeformationOptions::NeighborType type = ZQ_GridDeformationOptions::NEIGHBOR_4);
		bool _addMatrixTo_ANGLE_XLOOP(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight, 
			const ZQ_GridDeformationOptions::NeighborType type = ZQ_GridDeformationOptions::NEIGHBOR_4);
		bool _addMatrixTo_DISTANCE(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight);
		bool _addMatrixTo_DISTANCE_XLOOP(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight);
		bool _addMatrixTo_ARAP_VERT(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight,
			const ZQ_GridDeformationOptions::NeighborType type = ZQ_GridDeformationOptions::NEIGHBOR_4);
		bool _addMatrixTo_ARAP_VERT_XLOOP(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight,
			const ZQ_GridDeformationOptions::NeighborType type = ZQ_GridDeformationOptions::NEIGHBOR_4);

		void _buildMatrixInvFC();
		bool _bestFitTriangle(const T input[6], bool fixed1, bool fixed2, bool fixed3, const T distance, T output[6]);
		bool _buildMatrix_for_scaling();

		bool _estimateR_for_ARAP_VERT(const T* init_coord, std::vector<_mat22>& Rmats, const T scale, const ZQ_GridDeformationOptions::NeighborType type);
		bool _estimateR_for_ARAP_VERT_XLOOP(const T* init_coord, std::vector<_mat22>& Rmats, const T scale, const ZQ_GridDeformationOptions::NeighborType type);
		bool _solve_for_ARAP_VERT(const T* init_coord, const std::vector<_mat22>& Rmats, T* out_coord, const int iteration, const T scale, const ZQ_GridDeformationOptions::NeighborType type);
		bool _solve_for_ARAP_VERT_XLOOP(const T* init_coord, const std::vector<_mat22>& Rmats, T* out_coord, const int iteration, const T scale, const ZQ_GridDeformationOptions::NeighborType type);
	};


	/************************  definitions  ****************************/

	template<class T>
	ZQ_GridDeformation<T>::ZQ_GridDeformation()
	{
		width = 0;
		height = 0;
		nouseful_flag = 0;
		fixed_flag = 0;
		x_index = 0;
		y_index = 0;
		G = 0;
		B = 0;
		G_dist = 0;
		B_dist = 0;
		H = 0;
		D = 0;
		_buildMatrixInvFC();
	}

	template<class T>
	ZQ_GridDeformation<T>::~ZQ_GridDeformation()
	{
		ClearMatrix();
	}

	template<class T>
	void ZQ_GridDeformation<T>::ClearMatrix()
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
		width = 0;
		height = 0;
	}

	template<class T>
	bool ZQ_GridDeformation<T>::BuildMatrix(const int width, const int height, const bool* nouseful_flag, const bool* fixed_flag, const ZQ_GridDeformationOptions& opt)
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
		this->nouseful_flag = new bool[width*height];
		this->fixed_flag = new bool[width*height];
		memcpy(this->nouseful_flag,nouseful_flag,sizeof(bool)*width*height);
		memcpy(this->fixed_flag,fixed_flag,sizeof(bool)*width*height);
		x_index = new int[width*height];
		y_index = new int[width*height];
		memset(x_index,0,sizeof(int)*width*height);
		memset(y_index,0,sizeof(int)*width*height);
		option = opt;

		nouseful_num =  0;
		unknown_num = 0;
		for(int i = 0;i < width*height;i++)
		{
			if(nouseful_flag[i])
				nouseful_num += 2;
			if(!nouseful_flag[i] && !fixed_flag[i])
				unknown_num += 2;
		}
		if(unknown_num == 0)
		{
			return false;
		}

		if(unknown_num > (width*height*2 - nouseful_num - 4))
			return false;

		fixed_num = width*height*2 - nouseful_num - unknown_num;

		int cur_unkonwn_index = 0, cur_fixed_index = 0;
		for(int i = 0;i < width*height;i++)
		{
			if(!nouseful_flag[i])
			{
				if(fixed_flag[i])
				{
					x_index[i] = cur_fixed_index++;
					y_index[i] = cur_fixed_index++;
				}
				else
				{
					x_index[i] = cur_unkonwn_index++;
					y_index[i] = cur_unkonwn_index++;
				}
			}
		}

		ZQ_SparseMatrix<T> Gmat(unknown_num,unknown_num);
		ZQ_SparseMatrix<T> Bmat(unknown_num,fixed_num);

		switch(opt.methodType)
		{
		case ZQ_GridDeformationOptions::METHOD_LINE_ANGLE_ENERGY:
			{
				if(!_addMatrixTo_LINE(Gmat,Bmat,opt.line_weight,opt.neighborType))
					return false;
				if(!_addMatrixTo_ANGLE(Gmat,Bmat,opt.angle_weight,opt.neighborType))
					return false;
				G = Gmat.ExportCCS(taucs_flag);
				B = Bmat.ExportCCS(taucs_flag);

			}
			break;
		case ZQ_GridDeformationOptions::METHOD_LINE_ANGLE_ENERGY_SCALING:
			{
				if(!_addMatrixTo_LINE(Gmat,Bmat,opt.line_weight,opt.neighborType))
					return false;
				if(!_addMatrixTo_ANGLE(Gmat,Bmat,opt.angle_weight,opt.neighborType))
					return false;
				if(!_buildMatrix_for_scaling())
					return false;

				G = Gmat.ExportCCS(taucs_flag);
				B = Bmat.ExportCCS(taucs_flag);

			}
			break;
		case ZQ_GridDeformationOptions::METHOD_LINE_ANGLE_DISTANCE_ENERGY:
			{
				if(!_addMatrixTo_LINE(Gmat,Bmat,opt.line_weight,opt.neighborType))
					return false;
				if(!_addMatrixTo_ANGLE(Gmat,Bmat,opt.angle_weight,opt.neighborType))
					return false;
				G = Gmat.ExportCCS(taucs_flag);
				B = Bmat.ExportCCS(taucs_flag);
				if(!_addMatrixTo_DISTANCE(Gmat,Bmat,opt.distance_weight))
					return false;
				G_dist = Gmat.ExportCCS(taucs_flag);
				B_dist = Bmat.ExportCCS(taucs_flag);
			}
			break;
		case ZQ_GridDeformationOptions::METHOD_LINE_ANGLE_ENERGY_XLOOP:
			{
				if(!_addMatrixTo_LINE_XLOOP(Gmat,Bmat,opt.line_weight,opt.neighborType))
					return false;
				if(!_addMatrixTo_ANGLE_XLOOP(Gmat,Bmat,opt.angle_weight,opt.neighborType))
					return false;
				G = Gmat.ExportCCS(taucs_flag);
				B = Bmat.ExportCCS(taucs_flag);
			}
			break;
		case ZQ_GridDeformationOptions::METHOD_LINE_ANGLE_DISTANCE_ENERGY_XLOOP:
			{
				if(!_addMatrixTo_LINE_XLOOP(Gmat,Bmat,opt.line_weight,opt.neighborType))
					return false;
				if(!_addMatrixTo_ANGLE_XLOOP(Gmat,Bmat,opt.angle_weight,opt.neighborType))
					return false;
				G = Gmat.ExportCCS(taucs_flag);
				B = Bmat.ExportCCS(taucs_flag);
				if(!_addMatrixTo_DISTANCE_XLOOP(Gmat,Bmat,opt.distance_weight))
					return false;
				G_dist = Gmat.ExportCCS(taucs_flag);
				B_dist = Bmat.ExportCCS(taucs_flag);
			}
			break;
		case ZQ_GridDeformationOptions::METHOD_ARAP_VERT_AS_CENTER:
			{
				if(!_addMatrixTo_ARAP_VERT(Gmat,Bmat,1,opt.neighborType))
					return false;
				G = Gmat.ExportCCS(taucs_flag);
				B = Bmat.ExportCCS(taucs_flag);
			}
			break;
		case ZQ_GridDeformationOptions::METHOD_ARAP_VERT_AS_CENTER_XLOOP:
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
	bool ZQ_GridDeformation<T>::Deformation(const T* init_coord, T* out_coord, bool has_good_init)
	{
		if(!has_good_init 
			|| option.methodType == ZQ_GridDeformationOptions::METHOD_LINE_ANGLE_ENERGY 
			|| option.methodType == ZQ_GridDeformationOptions::METHOD_LINE_ANGLE_ENERGY_XLOOP)
		{
			int it = 0;
			if(!_deformation_without_scaling(init_coord,out_coord,option.iteration,it))
				return false;
		}

		switch(option.methodType)
		{
		case ZQ_GridDeformationOptions::METHOD_LINE_ANGLE_ENERGY_SCALING:
			{
				ZQ_DImage<T> tmp_coord(width,height,2);
				if(has_good_init)
					memcpy(tmp_coord.data(),init_coord,sizeof(T)*width*height*2);
				else
					memcpy(tmp_coord.data(),out_coord,sizeof(T)*width*height*2);
				if(!_scaling(tmp_coord.data(),out_coord,option.iteration))
					return false;
			}
			break;
		case ZQ_GridDeformationOptions::METHOD_LINE_ANGLE_DISTANCE_ENERGY:
			{
				ZQ_DImage<T> tmp_coord(width,height,2);
				if(has_good_init)
					memcpy(tmp_coord.data(),init_coord,sizeof(T)*width*height*2);
				else
					memcpy(tmp_coord.data(),out_coord,sizeof(T)*width*height*2);
				if(!_deformation_with_distance(tmp_coord.data(),out_coord,option.FPIteration,option.iteration))
					return false;
			}
			break;
		case ZQ_GridDeformationOptions::METHOD_LINE_ANGLE_DISTANCE_ENERGY_XLOOP:
			{
				ZQ_DImage<T> tmp_coord(width,height,2);
				if(has_good_init)
					memcpy(tmp_coord.data(),init_coord,sizeof(T)*width*height*2);
				else
					memcpy(tmp_coord.data(),out_coord,sizeof(T)*width*height*2);
				if(!_deformation_with_distance_XLOOP(tmp_coord.data(),out_coord,option.FPIteration,option.iteration))
					return false;
			}
			break;
		case ZQ_GridDeformationOptions::METHOD_ARAP_VERT_AS_CENTER:
			{
				ZQ_DImage<T> tmp_coord(width,height,2);
				if(has_good_init)
					memcpy(tmp_coord.data(),init_coord,sizeof(T)*width*height*2);
				else
					memcpy(tmp_coord.data(),out_coord,sizeof(T)*width*height*2);
				if(!_deformation_ARAP_VERT(tmp_coord.data(),out_coord,option.FPIteration,option.iteration))
					return false;
			}
			break;
		case ZQ_GridDeformationOptions::METHOD_ARAP_VERT_AS_CENTER_XLOOP:
			{
				ZQ_DImage<T> tmp_coord(width,height,2);
				if(has_good_init)
					memcpy(tmp_coord.data(),init_coord,sizeof(T)*width*height*2);
				else
					memcpy(tmp_coord.data(),out_coord,sizeof(T)*width*height*2);
				if(!_deformation_ARAP_VERT_XLOOP(tmp_coord.data(),out_coord,option.FPIteration,option.iteration))
					return false;
			}
			break;
		}

		return true;
	}

	template<class T>
	bool ZQ_GridDeformation<T>::_deformation_without_scaling(const T* init_coord, T* out_coord, const int iteration, int& it)
	{
		if(G == 0 || B == 0)
			return false;

		T* fixed_values = new T[fixed_num];
		T* init_unknown = new T[unknown_num];
		for(int i = 0;i < width*height;i++)
		{
			if(nouseful_flag[i])
				continue;
			if(fixed_flag[i])
			{
				fixed_values[x_index[i]] = init_coord[i*2];
				fixed_values[y_index[i]] = init_coord[i*2+1];
			}
			else
			{
				init_unknown[x_index[i]] = init_coord[i*2];
				init_unknown[y_index[i]] = init_coord[i*2+1];
			}
		}

		T* unknown_values = new T[unknown_num];

		T* f = new T[unknown_num];
		ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec(B,fixed_values,f);
		for(int i = 0;i < unknown_num;i++)
			f[i] *= -0.5;

		int max_iter = iteration;
		double tol = 1e-16;
		it = 0;
		ZQ_PCGSolver::PCG(G,f,init_unknown,max_iter,tol,unknown_values,it);

		for(int i = 0;i < width*height;i++)
		{
			if(nouseful_flag[i])
				continue;
			if(fixed_flag[i])
			{
				out_coord[i*2] = init_coord[i*2];
				out_coord[i*2+1] = init_coord[i*2+1];
			}
			else
			{
				out_coord[i*2] = unknown_values[x_index[i]];
				out_coord[i*2+1] = unknown_values[y_index[i]];
			}
		}

		delete []fixed_values;
		delete []init_unknown;
		delete []unknown_values;
		delete []f;
		return true;
	}

	template<class T>
	bool ZQ_GridDeformation<T>::_scaling(const T* init_coord, T* out_coord, const int iteration)
	{
		if(H == 0 || D == 0)
			return false;

		ZQ_DImage<T> f0_img(unknown_num,1,1);
		T*& f0 = f0_img.data();
		memset(f0,0,sizeof(T)*unknown_num);
		int dir_x[4] = {1,0,-1,0};
		int dir_y[4] = {0,1,0,-1};
		for(int i = 0;i < height;i++)
		{
			for(int j = 0;j < width;j++)
			{
				int offset = i*width+j;
				if(nouseful_flag[offset])
					continue;

				for(int dd = 0;dd < 4;dd++)
				{
					int i_2 = i+dir_y[dd];
					int j_2 = j+dir_x[dd];
					int i_3 = i+dir_y[(dd+1)%4];
					int j_3 = j+dir_x[(dd+1)%4];
					if(0 <= i_2 && i_2 < height && 0 <= j_2 && j_2 < width && 0 <= i_3 && i_3 < height && 0 <= j_3 && j_3 < width
						&& !nouseful_flag[i_2*width+j_2] && !nouseful_flag[i_3*width+j_3])
					{
						int offset_2 = i_2*width+j_2;
						int offset_3 = i_3*width+j_3;
						T input[6] = {
							init_coord[offset*2+0],init_coord[offset*2+1],
							init_coord[offset_2*2+0],init_coord[offset_2*2+1],
							init_coord[offset_3*2+0],init_coord[offset_3*2+1]
						};
						T output[6];
						if(!_bestFitTriangle(input,fixed_flag[offset],fixed_flag[offset_2],fixed_flag[offset_3],option.distance,output))
						{
							return false;
						}


						/*
						((x1-x2)-(X1-X2))^2 + ((y1-y2)-(Y1-Y2))^2
						+ ((x2-x3)-(X2-X3))^2 + ((y2-y3)-(Y2-Y3))^2
						+ ((x3-x1)-(X3-X1))^2 + ((y3-y1)-(Y3-Y1))^2
						= 2x1^2 + 2y1^2 + 2x2^2 + 2y2^2 + 2x3^2 +2y3^2
						+ (-2)x1x2 + (-2)y1y2
						+ (-2)x1x3 + (-2)y1y3
						+ (-2)x2x3 + (-2)y2y3
						+ x1*(-4X1+2X2+2X3) + y1*(-4Y1+2Y2+2Y3)
						+ x2*(-4X2+2X1+2X3) + y2*(-4Y2+2Y1+2Y3)
						+ x3*(-4X3+2X1+2X2) + y3*(-4Y3+2Y1+2Y2)
						+ const_value
						*/

						if(!fixed_flag[offset])
						{
							f0[x_index[offset]] += -4*output[0] + 2*output[2] + 2*output[4];
							f0[y_index[offset]] += -4*output[1] + 2*output[3] + 2*output[5];
						}
						if(!fixed_flag[offset_2])
						{
							f0[x_index[offset_2]] += -4*output[2] + 2*output[0] + 2*output[4];
							f0[y_index[offset_2]] += -4*output[3] + 2*output[1] + 2*output[5];
						}
						if(!fixed_flag[offset_3])
						{
							f0[x_index[offset_3]] += -4*output[4] + 2*output[0] + 2*output[2];
							f0[y_index[offset_3]] += -4*output[5] + 2*output[1] + 2*output[3];
						}
					}
				}
			}
		}

		T* fixed_values = new T[fixed_num];
		T* init_unknown = new T[unknown_num];
		for(int i = 0;i < width*height;i++)
		{
			if(fixed_flag[i])
			{
				fixed_values[x_index[i]] = init_coord[i*2];
				fixed_values[y_index[i]] = init_coord[i*2+1];
			}
			else
			{
				init_unknown[x_index[i]] = init_coord[i*2];
				init_unknown[y_index[i]] = init_coord[i*2+1];
			}
		}

		T* unknown_values = new T[unknown_num];

		T* f = new T[unknown_num];
		ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec(D,fixed_values,f);
		for(int i = 0;i < unknown_num;i++)
		{
			f[i] += f0[i];
			f[i] *= -0.5;
		}

		int max_iter = iteration;
		double tol = 1e-16;
		int it = 0;
		ZQ_PCGSolver::PCG(H,f,init_unknown,max_iter,tol,unknown_values,it);

		for(int i = 0;i < width*height;i++)
		{
			if(fixed_flag[i])
			{
				out_coord[i*2] = init_coord[i*2];
				out_coord[i*2+1] = init_coord[i*2+1];
			}
			else
			{
				out_coord[i*2] = unknown_values[x_index[i]];
				out_coord[i*2+1] = unknown_values[y_index[i]];
			}
		}

		delete []fixed_values;
		delete []init_unknown;
		delete []unknown_values;
		delete []f;

		return true;
	}

	template<class T>
	bool ZQ_GridDeformation<T>::_deformation_with_distance(const T* init_coord, T* out_coord, const int nFPIter, const int iteration)
	{
		if(G_dist == 0 || B_dist == 0)
			return false;

		int dir_x[4] = {1,0,-1,0};
		int dir_y[4] = {0,1,0,-1};

		float dist_weight = option.distance_weight*option.distance_weight;

		ZQ_DImage<T> tmp_img(width,height,2);
		T*& tmp_data = tmp_img.data();
		memcpy(tmp_data,init_coord,sizeof(T)*width*height*2);
		for(int nfpit = 0;nfpit < nFPIter;nfpit++)
		{
			ZQ_DImage<T> f0_img(unknown_num,1,1);
			T*& f0 = f0_img.data();
			memset(f0,0,sizeof(T)*unknown_num);

			for(int i = 0;i < height;i++)
			{
				for(int j = 0;j < width;j++)
				{
					int offset = i*width+j;
					if(nouseful_flag[offset])
						continue;

					for(int dd = 0;dd < 4;dd++)
					{
						int i_2 = i+dir_y[dd];
						int j_2 = j+dir_x[dd];
						if(0 <= i_2 && i_2 < height && 0 <= j_2 && j_2 < width && !nouseful_flag[i_2*width+j_2])
						{
							int offset_2 = i_2*width+j_2;
							float v_dir[4] = {
								tmp_data[offset_2*2+0]-tmp_data[offset*2+0],
								tmp_data[offset_2*2+1]-tmp_data[offset*2+1]
							};

							float v_len = sqrt(v_dir[0]*v_dir[0]+v_dir[1]*v_dir[1]);
							if(v_len != 0)
							{
								v_dir[0] *= option.distance/v_len;
								v_dir[1] *= option.distance/v_len;
							}


							/*
							((x1-x2)-(X1-X2))^2 + ((y1-y2)-(Y1-Y2))^2
							= x1^2 + y1^2 + x2^2 + y2^2
							+ (-2)x1x2 + (-2)y1y2
							+ x1*(-2X1+2X2) + y1*(-2Y1+2Y2)
							+ x2*(-2X2+2X1) + y2*(-2Y2+2Y1)
							+ const_value
							*/

							if(!fixed_flag[offset])
							{
								f0[x_index[offset]] += 2*v_dir[0]*dist_weight;
								f0[y_index[offset]] += 2*v_dir[1]*dist_weight;
							}
							if(!fixed_flag[offset_2])
							{
								f0[x_index[offset_2]] += -2*v_dir[0]*dist_weight;
								f0[y_index[offset_2]] += -2*v_dir[1]*dist_weight;
							}
						}
					}
				}
			}

			T* fixed_values = new T[fixed_num];
			T* init_unknown = new T[unknown_num];
			for(int i = 0;i < width*height;i++)
			{
				if(nouseful_flag[i])
					continue;
				if(fixed_flag[i])
				{
					fixed_values[x_index[i]] = tmp_data[i*2];
					fixed_values[y_index[i]] = tmp_data[i*2+1];
				}
				else
				{
					init_unknown[x_index[i]] = tmp_data[i*2];
					init_unknown[y_index[i]] = tmp_data[i*2+1];
				}
			}

			T* unknown_values = new T[unknown_num];

			T* f = new T[unknown_num];
			ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec(B_dist,fixed_values,f);
			for(int i = 0;i < unknown_num;i++)
			{
				f[i] += f0[i];
				f[i] *= -0.5;
			}

			int max_iter = iteration;
			double tol = 1e-16;
			int it = 0;
			ZQ_PCGSolver::PCG(G_dist,f,init_unknown,max_iter,tol,unknown_values,it,true);

			for(int i = 0;i < width*height;i++)
			{
				if(nouseful_flag[i])
					continue;
				if(fixed_flag[i])
				{
					out_coord[i*2] = tmp_data[i*2];
					out_coord[i*2+1] = tmp_data[i*2+1];
				}
				else
				{
					out_coord[i*2] = unknown_values[x_index[i]];
					out_coord[i*2+1] = unknown_values[y_index[i]];
				}
			}

			delete []fixed_values;
			delete []init_unknown;
			delete []unknown_values;
			delete []f;

			memcpy(tmp_data,out_coord,sizeof(T)*width*height*2);
		}

		return true;
	}

	template<class T>
	bool ZQ_GridDeformation<T>::_deformation_with_distance_XLOOP(const T* init_coord, T* out_coord, const int nFPIter, const int iteration)
	{
		if(G_dist == 0 || B_dist == 0)
			return false;

		int dir_x[4] = {1,0,-1,0};
		int dir_y[4] = {0,1,0,-1};

		float dist_weight = option.distance_weight*option.distance_weight;

		ZQ_DImage<T> tmp_img(width,height,2);
		T*& tmp_data = tmp_img.data();
		memcpy(tmp_data,init_coord,sizeof(T)*width*height*2);
		for(int nfpit = 0;nfpit < nFPIter;nfpit++)
		{
			ZQ_DImage<T> f0_img(unknown_num,1,1);
			T*& f0 = f0_img.data();
			memset(f0,0,sizeof(T)*unknown_num);

			for(int i = 0;i < height;i++)
			{
				for(int j = 0;j < width;j++)
				{
					int offset = i*width+j;
					if(nouseful_flag[offset])
						continue;

					for(int dd = 0;dd < 4;dd++)
					{
						int i_2 = i+dir_y[dd];
						int j_2 = j+dir_x[dd];
						j_2 = (j_2+width)%width;
						int offset_2 = i_2*width+j_2;
						if(0 <= i_2 && i_2 < height && !nouseful_flag[offset_2])
						{
							T v_dir[4] = {
								tmp_data[offset_2*2+0]-tmp_data[offset*2+0],
								tmp_data[offset_2*2+1]-tmp_data[offset*2+1]
							};

							T v_len = sqrt(v_dir[0]*v_dir[0]+v_dir[1]*v_dir[1]);
							if(v_len != 0)
							{
								v_dir[0] *= option.distance/v_len;
								v_dir[1] *= option.distance/v_len;
							}


							/*
							((x1-x2)-(X1-X2))^2 + ((y1-y2)-(Y1-Y2))^2
							= x1^2 + y1^2 + x2^2 + y2^2
							+ (-2)x1x2 + (-2)y1y2
							+ x1*(-2X1+2X2) + y1*(-2Y1+2Y2)
							+ x2*(-2X2+2X1) + y2*(-2Y2+2Y1)
							+ const_value
							*/

							if(!fixed_flag[offset])
							{
								f0[x_index[offset]] += 2*v_dir[0]*dist_weight;
								f0[y_index[offset]] += 2*v_dir[1]*dist_weight;
							}
							if(!fixed_flag[offset_2])
							{
								f0[x_index[offset_2]] += -2*v_dir[0]*dist_weight;
								f0[y_index[offset_2]] += -2*v_dir[1]*dist_weight;
							}
						}
					}
				}
			}

			T* fixed_values = new T[fixed_num];
			T* init_unknown = new T[unknown_num];
			for(int i = 0;i < width*height;i++)
			{
				if(fixed_flag[i])
				{
					fixed_values[x_index[i]] = tmp_data[i*2];
					fixed_values[y_index[i]] = tmp_data[i*2+1];
				}
				else
				{
					init_unknown[x_index[i]] = tmp_data[i*2];
					init_unknown[y_index[i]] = tmp_data[i*2+1];
				}
			}

			T* unknown_values = new T[unknown_num];

			T* f = new T[unknown_num];
			ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec(B_dist,fixed_values,f);
			for(int i = 0;i < unknown_num;i++)
			{
				f[i] += f0[i];
				f[i] *= -0.5;
			}

			int max_iter = iteration;
			double tol = 1e-16;
			int it = 0;
			ZQ_PCGSolver::PCG(G_dist,f,init_unknown,max_iter,tol,unknown_values,it);

			for(int i = 0;i < width*height;i++)
			{
				if(fixed_flag[i])
				{
					out_coord[i*2] = tmp_data[i*2];
					out_coord[i*2+1] = tmp_data[i*2+1];
				}
				else
				{
					out_coord[i*2] = unknown_values[x_index[i]];
					out_coord[i*2+1] = unknown_values[y_index[i]];
				}
			}

			delete []fixed_values;
			delete []init_unknown;
			delete []unknown_values;
			delete []f;

			memcpy(tmp_data,out_coord,sizeof(T)*width*height*2);
		}

		return true;
	}

	template<class T>
	bool ZQ_GridDeformation<T>::_deformation_ARAP_VERT(const T* init_coord, T* out_coord, const int nFPIter, const int iteration)
	{
		ZQ_DImage<T> tmp_img(width,height,2);
		T*& tmp_data = tmp_img.data();
		memcpy(tmp_data,init_coord,sizeof(T)*width*height*2);

		for(int fp_it = 0;fp_it < nFPIter;fp_it++)
		{
			std::vector<_mat22> Rmats;
			_estimateR_for_ARAP_VERT(tmp_data,Rmats,option.distance,option.neighborType);
			_solve_for_ARAP_VERT(tmp_data,Rmats,out_coord,iteration,option.distance,option.neighborType);
			memcpy(tmp_data,out_coord,sizeof(T)*width*height*2);
		}

		return true;
	}


	template<class T>
	bool ZQ_GridDeformation<T>::_deformation_ARAP_VERT_XLOOP(const T* init_coord, T* out_coord, const int nFPIter, const int iteration)
	{
		ZQ_DImage<T> tmp_img(width,height,2);
		T*& tmp_data = tmp_img.data();
		memcpy(tmp_data,init_coord,sizeof(T)*width*height*2);

		for(int fp_it = 0;fp_it < nFPIter;fp_it++)
		{
			std::vector<_mat22> Rmats;
			_estimateR_for_ARAP_VERT_XLOOP(tmp_data,Rmats,option.distance,option.neighborType);
			_solve_for_ARAP_VERT_XLOOP(tmp_data,Rmats,out_coord,iteration,option.distance,option.neighborType);
			memcpy(tmp_data,out_coord,sizeof(T)*width*height*2);
		}

		return true;
	}

	template<class T>
	bool ZQ_GridDeformation<T>::_addMatrixTo_LINE(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight, const ZQ_GridDeformationOptions::NeighborType type)
	{
		if(weight == 0)
			return true;

		/**********   LINE  ENERGY Begin  **************/
		T line_energy_scale = weight*weight;
		int dir_x[12] = {1,0,-1,0, 1,-1,-1,1, 2,0,-2,0};
		int dir_y[12] = {0,1,0,-1, 1,1,-1,-1, 0,2,0,-2};
		int neighbor_loop_num = 4;
		switch(type)
		{
		case ZQ_GridDeformationOptions::NEIGHBOR_4:
			neighbor_loop_num = 4;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_8:
			neighbor_loop_num = 8;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_12:
			neighbor_loop_num = 12;
			break;
		default:
			return false;
			break;
		}
		for(int i = 0;i < height;i++)
		{
			for(int j = 0;j < width;j++)
			{
				int offset = i*width+j;
				if(nouseful_flag[offset])
					continue;

				if(!fixed_flag[offset])
				{
					for(int ll = 0;ll < neighbor_loop_num;ll+=4)
					{
						for(int dd = 0;dd < 2;dd++)
						{
							int i_2 = i+dir_y[dd+ll];
							int j_2 = j+dir_x[dd+ll];
							int i_3 = i+dir_y[dd+2+ll];
							int j_3 = j+dir_x[dd+2+ll];
							int offset_2 = i_2*width+j_2;
							int offset_3 = i_3*width+j_3;
							if(0 <= i_2 && i_2 < height && 0 <= j_2 && j_2 < width && 0 <= i_3 && i_3 < height && 0 <= j_3 && j_3 < width
								&& !nouseful_flag[offset_2] && !nouseful_flag[offset_3])
							{
								/*
								(2x1-x2-x3)^2+(2y1-y2-y3)^2
								= 4x1^2 + 4y1^2 + x2^2 + y2^2 + x3^2 + y3^2
								+ (-4)x1x2 + (-4)y1y2 
								+ (-4)x1x3 + (-4)y1y3
								+ 2x2x3 + 2y2y3 
								*/
								Gmat.AddTo(x_index[offset],x_index[offset],4*line_energy_scale);
								Gmat.AddTo(y_index[offset],y_index[offset],4*line_energy_scale);
								if(!fixed_flag[offset_2] && !fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*line_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset_3],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_3],1*line_energy_scale);

									Gmat.AddTo(x_index[offset],x_index[offset_2],-2*line_energy_scale);
									Gmat.AddTo(x_index[offset_2],x_index[offset],-2*line_energy_scale);
									Gmat.AddTo(y_index[offset],y_index[offset_2],-2*line_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset],-2*line_energy_scale);

									Gmat.AddTo(x_index[offset],x_index[offset_3],-2*line_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset],-2*line_energy_scale);
									Gmat.AddTo(y_index[offset],y_index[offset_3],-2*line_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset],-2*line_energy_scale);

									Gmat.AddTo(x_index[offset_2],x_index[offset_3],1*line_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset_2],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_3],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_2],1*line_energy_scale);
								}
								else if(!fixed_flag[offset_2] && fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*line_energy_scale);

									Gmat.AddTo(x_index[offset],x_index[offset_2],-2*line_energy_scale);
									Gmat.AddTo(x_index[offset_2],x_index[offset],-2*line_energy_scale);
									Gmat.AddTo(y_index[offset],y_index[offset_2],-2*line_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset],-2*line_energy_scale);

									Bmat.AddTo(x_index[offset],x_index[offset_3],-4*line_energy_scale);
									Bmat.AddTo(y_index[offset],y_index[offset_3],-4*line_energy_scale);

									Bmat.AddTo(x_index[offset_2],x_index[offset_3],2*line_energy_scale);
									Bmat.AddTo(y_index[offset_2],y_index[offset_3],2*line_energy_scale);
								}
								else if(fixed_flag[offset_2] && !fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_3],x_index[offset_3],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_3],1*line_energy_scale);

									Bmat.AddTo(x_index[offset],x_index[offset_2],-4*line_energy_scale);
									Bmat.AddTo(y_index[offset],y_index[offset_2],-4*line_energy_scale);

									Gmat.AddTo(x_index[offset],x_index[offset_3],-2*line_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset],-2*line_energy_scale);
									Gmat.AddTo(y_index[offset],y_index[offset_3],-2*line_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset],-2*line_energy_scale);

									Bmat.AddTo(x_index[offset_3],x_index[offset_2],2*line_energy_scale);
									Bmat.AddTo(y_index[offset_3],y_index[offset_2],2*line_energy_scale);
								}
								else
								{
									Bmat.AddTo(x_index[offset],x_index[offset_2],-4*line_energy_scale);
									Bmat.AddTo(y_index[offset],y_index[offset_2],-4*line_energy_scale);

									Bmat.AddTo(x_index[offset],x_index[offset_3],-4*line_energy_scale);
									Bmat.AddTo(y_index[offset],y_index[offset_3],-4*line_energy_scale);
								}
							}
						}
					}
				}
				// fixed_flag[offset] == true
				else
				{
					for(int ll = 0;ll < neighbor_loop_num;ll+=4)
					{
						for(int dd = 0;dd < 2;dd++)
						{
							int i_2 = i+dir_y[dd+ll];
							int j_2 = j+dir_x[dd+ll];
							int i_3 = i+dir_y[dd+2+ll];
							int j_3 = j+dir_x[dd+2+ll];
							int offset_2 = i_2*width+j_2;
							int offset_3 = i_3*width+j_3;
							if(0 <= i_2 && i_2 < height && 0 <= j_2 && j_2 < width && 0 <= i_3 && i_3 < height && 0 <= j_3 && j_3 < width
								&& !nouseful_flag[offset_2] && !nouseful_flag[offset_3])
							{
								/*
								(2x1-x2-x3)^2+(2y1-y2-y3)^2
								= 4x1^2 + 4y1^2 + x2^2 + y2^2 + x3^2 + y3^2
								+ (-4)x1x2 + (-4)y1y2 
								+ (-4)x1x3 + (-4)y1y3
								+ 2x2x3 + 2y2y3 
								*/
								if(!fixed_flag[offset_2] && !fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*line_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset_3],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_3],1*line_energy_scale);

									Bmat.AddTo(x_index[offset_2],x_index[offset],-4*line_energy_scale);
									Bmat.AddTo(y_index[offset_2],y_index[offset],-4*line_energy_scale);

									Bmat.AddTo(x_index[offset_3],x_index[offset],-4*line_energy_scale);
									Bmat.AddTo(y_index[offset_3],y_index[offset],-4*line_energy_scale);

									Gmat.AddTo(x_index[offset_2],x_index[offset_3],1*line_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset_2],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_3],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_2],1*line_energy_scale);
								}
								else if(!fixed_flag[offset_2] && fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*line_energy_scale);

									Bmat.AddTo(x_index[offset_2],x_index[offset],-4*line_energy_scale);
									Bmat.AddTo(y_index[offset_2],y_index[offset],-4*line_energy_scale);

									Bmat.AddTo(x_index[offset_2],x_index[offset_3],2*line_energy_scale);
									Bmat.AddTo(y_index[offset_2],y_index[offset_3],2*line_energy_scale);
								}
								else if(fixed_flag[offset_2] && !fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_3],x_index[offset_3],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_3],1*line_energy_scale);

									Bmat.AddTo(x_index[offset_3],x_index[offset],-4*line_energy_scale);
									Bmat.AddTo(y_index[offset_3],y_index[offset],-4*line_energy_scale);

									Bmat.AddTo(x_index[offset_3],x_index[offset_2],2*line_energy_scale);
									Bmat.AddTo(y_index[offset_3],y_index[offset_2],2*line_energy_scale);
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

		/**********   LINE  ENERGY End    **************/
		return true;
	}


	template<class T>
	bool ZQ_GridDeformation<T>::_addMatrixTo_LINE_XLOOP(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight, const ZQ_GridDeformationOptions::NeighborType type)
	{
		if(weight == 0)
			return true;

		/**********   LINE  ENERGY Begin  **************/
		T line_energy_scale = weight*weight;
		int dir_x[12] = {1,0,-1,0, 1,-1,-1,1, 2,0,-2,0};
		int dir_y[12] = {0,1,0,-1, 1,1,-1,-1, 0,2,0,-2};
		int neighbor_loop_num = 4;
		switch(type)
		{
		case ZQ_GridDeformationOptions::NEIGHBOR_4:
			neighbor_loop_num = 4;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_8:
			neighbor_loop_num = 8;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_12:
			neighbor_loop_num = 12;
			break;
		default:
			return false;
			break;
		}
		for(int i = 0;i < height;i++)
		{
			for(int j = 0;j < width;j++)
			{
				int offset = i*width+j;
				if(nouseful_flag[offset])
					continue;

				if(!fixed_flag[offset])
				{
					for(int ll = 0;ll < neighbor_loop_num;ll+=4)
					{
						for(int dd = 0;dd < 2;dd++)
						{
							int i_2 = i+dir_y[dd+ll];
							int j_2 = j+dir_x[dd+ll];
							int i_3 = i+dir_y[dd+2+ll];
							int j_3 = j+dir_x[dd+2+ll];
							j_2 = (j_2+width)%width;
							j_3 = (j_3+width)%width;
							int offset_2 = i_2*width+j_2;
							int offset_3 = i_3*width+j_3;
							if(0 <= i_2 && i_2 < height && 0 <= i_3 && i_3 < height
								&& !nouseful_flag[offset_2] && !nouseful_flag[offset_3])
							{
								/*
								(2x1-x2-x3)^2+(2y1-y2-y3)^2
								= 4x1^2 + 4y1^2 + x2^2 + y2^2 + x3^2 + y3^2
								+ (-4)x1x2 + (-4)y1y2 
								+ (-4)x1x3 + (-4)y1y3
								+ 2x2x3 + 2y2y3 
								*/
								Gmat.AddTo(x_index[offset],x_index[offset],4*line_energy_scale);
								Gmat.AddTo(y_index[offset],y_index[offset],4*line_energy_scale);
								if(!fixed_flag[offset_2] && !fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*line_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset_3],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_3],1*line_energy_scale);

									Gmat.AddTo(x_index[offset],x_index[offset_2],-2*line_energy_scale);
									Gmat.AddTo(x_index[offset_2],x_index[offset],-2*line_energy_scale);
									Gmat.AddTo(y_index[offset],y_index[offset_2],-2*line_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset],-2*line_energy_scale);

									Gmat.AddTo(x_index[offset],x_index[offset_3],-2*line_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset],-2*line_energy_scale);
									Gmat.AddTo(y_index[offset],y_index[offset_3],-2*line_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset],-2*line_energy_scale);

									Gmat.AddTo(x_index[offset_2],x_index[offset_3],1*line_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset_2],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_3],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_2],1*line_energy_scale);
								}
								else if(!fixed_flag[offset_2] && fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*line_energy_scale);

									Gmat.AddTo(x_index[offset],x_index[offset_2],-2*line_energy_scale);
									Gmat.AddTo(x_index[offset_2],x_index[offset],-2*line_energy_scale);
									Gmat.AddTo(y_index[offset],y_index[offset_2],-2*line_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset],-2*line_energy_scale);

									Bmat.AddTo(x_index[offset],x_index[offset_3],-4*line_energy_scale);
									Bmat.AddTo(y_index[offset],y_index[offset_3],-4*line_energy_scale);

									Bmat.AddTo(x_index[offset_2],x_index[offset_3],2*line_energy_scale);
									Bmat.AddTo(y_index[offset_2],y_index[offset_3],2*line_energy_scale);
								}
								else if(fixed_flag[offset_2] && !fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_3],x_index[offset_3],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_3],1*line_energy_scale);

									Bmat.AddTo(x_index[offset],x_index[offset_2],-4*line_energy_scale);
									Bmat.AddTo(y_index[offset],y_index[offset_2],-4*line_energy_scale);

									Gmat.AddTo(x_index[offset],x_index[offset_3],-2*line_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset],-2*line_energy_scale);
									Gmat.AddTo(y_index[offset],y_index[offset_3],-2*line_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset],-2*line_energy_scale);

									Bmat.AddTo(x_index[offset_3],x_index[offset_2],2*line_energy_scale);
									Bmat.AddTo(y_index[offset_3],y_index[offset_2],2*line_energy_scale);
								}
								else
								{
									Bmat.AddTo(x_index[offset],x_index[offset_2],-4*line_energy_scale);
									Bmat.AddTo(y_index[offset],y_index[offset_2],-4*line_energy_scale);

									Bmat.AddTo(x_index[offset],x_index[offset_3],-4*line_energy_scale);
									Bmat.AddTo(y_index[offset],y_index[offset_3],-4*line_energy_scale);
								}
							}
						}
					}
				}
				// fixed_flag[offset] == true
				else
				{
					for(int ll = 0;ll < neighbor_loop_num;ll+=4)
					{
						for(int dd = 0;dd < 2;dd++)
						{
							int i_2 = i+dir_y[dd+ll];
							int j_2 = j+dir_x[dd+ll];
							int i_3 = i+dir_y[dd+2+ll];
							int j_3 = j+dir_x[dd+2+ll];
							j_2 = (j_2+width)%width;
							j_3 = (j_3+width)%width;
							int offset_2 = i_2*width+j_2;
							int offset_3 = i_3*width+j_3;
							if(0 <= i_2 && i_2 < height && 0 <= i_3 && i_3 < height
								&& !nouseful_flag[offset_2] && !nouseful_flag[offset_3])
							{
								/*
								(2x1-x2-x3)^2+(2y1-y2-y3)^2
								= 4x1^2 + 4y1^2 + x2^2 + y2^2 + x3^2 + y3^2
								+ (-4)x1x2 + (-4)y1y2 
								+ (-4)x1x3 + (-4)y1y3
								+ 2x2x3 + 2y2y3 
								*/
								if(!fixed_flag[offset_2] && !fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*line_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset_3],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_3],1*line_energy_scale);

									Bmat.AddTo(x_index[offset_2],x_index[offset],-4*line_energy_scale);
									Bmat.AddTo(y_index[offset_2],y_index[offset],-4*line_energy_scale);

									Bmat.AddTo(x_index[offset_3],x_index[offset],-4*line_energy_scale);
									Bmat.AddTo(y_index[offset_3],y_index[offset],-4*line_energy_scale);

									Gmat.AddTo(x_index[offset_2],x_index[offset_3],1*line_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset_2],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_3],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_2],1*line_energy_scale);
								}
								else if(!fixed_flag[offset_2] && fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*line_energy_scale);

									Bmat.AddTo(x_index[offset_2],x_index[offset],-4*line_energy_scale);
									Bmat.AddTo(y_index[offset_2],y_index[offset],-4*line_energy_scale);

									Bmat.AddTo(x_index[offset_2],x_index[offset_3],2*line_energy_scale);
									Bmat.AddTo(y_index[offset_2],y_index[offset_3],2*line_energy_scale);
								}
								else if(fixed_flag[offset_2] && !fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_3],x_index[offset_3],1*line_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_3],1*line_energy_scale);

									Bmat.AddTo(x_index[offset_3],x_index[offset],-4*line_energy_scale);
									Bmat.AddTo(y_index[offset_3],y_index[offset],-4*line_energy_scale);

									Bmat.AddTo(x_index[offset_3],x_index[offset_2],2*line_energy_scale);
									Bmat.AddTo(y_index[offset_3],y_index[offset_2],2*line_energy_scale);
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

		/**********   LINE  ENERGY End    **************/
		return true;
	}

	template<class T>
	bool ZQ_GridDeformation<T>::_addMatrixTo_ANGLE(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight, const ZQ_GridDeformationOptions::NeighborType type)
	{
		if(weight == 0)
			return true;

		/**********   ANGLE ENERGY Begin  **************/
		float angle_energy_scale = weight*weight;

		int dir_x[12] = {1,0,-1,0, 1,-1,-1,1, 2,0,-2,0};
		int dir_y[12] = {0,1,0,-1, 1,1,-1,-1, 0,2,0,-2};
		int neighbor_loop_num = 4;
		switch(type)
		{
		case ZQ_GridDeformationOptions::NEIGHBOR_4:
			neighbor_loop_num = 4;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_8:
			neighbor_loop_num = 8;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_12:
			neighbor_loop_num = 12;
			break;
		default:
			return false;
			break;
		}
		for(int i = 0;i < height;i++)
		{
			for(int j = 0;j < width;j++)
			{
				int offset = i*width+j;
				if(nouseful_flag[offset])
					continue;

				if(!fixed_flag[offset])
				{
					for(int ll = 0;ll < neighbor_loop_num;ll+=4)
					{
						for(int dd = 0;dd < 4;dd++)
						{
							int i_2 = i+dir_y[dd+ll];
							int j_2 = j+dir_x[dd+ll];
							int i_3 = i+dir_y[(dd+1)%4+ll];
							int j_3 = j+dir_x[(dd+1)%4+ll];
							int offset_2 = i_2*width+j_2;
							int offset_3 = i_3*width+j_3;
							if(0 <= i_2 && i_2 < height && 0 <= j_2 && j_2 < width && 0 <= i_3 && i_3 < height && 0 <= j_3 && j_3 < width
								&& !nouseful_flag[offset_2] && !nouseful_flag[offset_3])
							{
								/*
								(x3-x1+y2-y1)^2+(y3-y1+x1-x2)^2
								= 2x1^2 + 2y1^2 + x2^2 + y2^2 + x3^2 + y3^2
								+ (-2)x1x2 + (-2)x1y2 + 2y1x2 + (-2)y1y2 
								+ (-2)x1x3 + (-2)y1x3 + 2x1y3 + (-2)y1y3
								+ (-2)x2y3 + 2y2x3 
								*/
								Gmat.AddTo(x_index[offset],x_index[offset],2*angle_energy_scale);
								Gmat.AddTo(y_index[offset],y_index[offset],2*angle_energy_scale);
								if(!fixed_flag[offset_2] && !fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset_3],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_3],1*angle_energy_scale);

									Gmat.AddTo(x_index[offset],x_index[offset_2],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_2],x_index[offset],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset],y_index[offset_2],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],x_index[offset],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset],x_index[offset_2],1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_2],y_index[offset],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset],y_index[offset_2],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset],-1*angle_energy_scale);

									Gmat.AddTo(x_index[offset],x_index[offset_3],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset],x_index[offset_3],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_3],y_index[offset],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset],y_index[offset_3],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],x_index[offset],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset],y_index[offset_3],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset],-1*angle_energy_scale);

									Gmat.AddTo(x_index[offset_2],y_index[offset_3],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],x_index[offset_2],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],x_index[offset_3],1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_3],y_index[offset_2],1*angle_energy_scale);
								}
								else if(!fixed_flag[offset_2] && fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*angle_energy_scale);

									Gmat.AddTo(x_index[offset],x_index[offset_2],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_2],x_index[offset],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset],y_index[offset_2],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],x_index[offset],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset],x_index[offset_2],1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_2],y_index[offset],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset],y_index[offset_2],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset],-1*angle_energy_scale);

									Bmat.AddTo(x_index[offset],x_index[offset_3],-2*angle_energy_scale);
									Bmat.AddTo(y_index[offset],x_index[offset_3],-2*angle_energy_scale);
									Bmat.AddTo(x_index[offset],y_index[offset_3],2*angle_energy_scale);
									Bmat.AddTo(y_index[offset],y_index[offset_3],-2*angle_energy_scale);

									Bmat.AddTo(x_index[offset_2],y_index[offset_3],-2*angle_energy_scale);
									Bmat.AddTo(y_index[offset_2],x_index[offset_3],2*angle_energy_scale);
								}
								else if(fixed_flag[offset_2] && !fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_3],x_index[offset_3],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_3],1*angle_energy_scale);

									Bmat.AddTo(x_index[offset],x_index[offset_2],-2*angle_energy_scale);
									Bmat.AddTo(x_index[offset],y_index[offset_2],-2*angle_energy_scale);
									Bmat.AddTo(y_index[offset],x_index[offset_2],2*angle_energy_scale);
									Bmat.AddTo(y_index[offset],y_index[offset_2],-2*angle_energy_scale);

									Gmat.AddTo(x_index[offset],x_index[offset_3],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset],x_index[offset_3],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_3],y_index[offset],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset],y_index[offset_3],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],x_index[offset],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset],y_index[offset_3],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset],-1*angle_energy_scale);

									Bmat.AddTo(y_index[offset_3],x_index[offset_2],-2*angle_energy_scale);
									Bmat.AddTo(x_index[offset_3],y_index[offset_2],2*angle_energy_scale);
								}
								else
								{
									Bmat.AddTo(x_index[offset],x_index[offset_2],-2*angle_energy_scale);	
									Bmat.AddTo(x_index[offset],y_index[offset_2],-2*angle_energy_scale);
									Bmat.AddTo(y_index[offset],x_index[offset_2],2*angle_energy_scale);
									Bmat.AddTo(y_index[offset],y_index[offset_2],-2*angle_energy_scale);

									Bmat.AddTo(x_index[offset],x_index[offset_3],-2*angle_energy_scale);
									Bmat.AddTo(y_index[offset],x_index[offset_3],-2*angle_energy_scale);
									Bmat.AddTo(x_index[offset],y_index[offset_3],2*angle_energy_scale);
									Bmat.AddTo(y_index[offset],y_index[offset_3],-2*angle_energy_scale);
								}
							}
						}
					}	
				}
				//fixed[offset] == true
				else
				{
					for(int ll = 0;ll < neighbor_loop_num;ll+=4)
					{
						for(int dd = 0;dd < 4;dd++)
						{
							int i_2 = i+dir_y[dd+ll];
							int j_2 = j+dir_x[dd+ll];
							int i_3 = i+dir_y[(dd+1)%4+ll];
							int j_3 = j+dir_x[(dd+1)%4+ll];
							int offset_2 = i_2*width+j_2;
							int offset_3 = i_3*width+j_3;
							if(0 <= i_2 && i_2 < height && 0 <= j_2 && j_2 < width && 0 <= i_3 && i_3 < height && 0 <= j_3 && j_3 < width
								&& !nouseful_flag[offset_2] && !nouseful_flag[offset_3])
							{
								/*
								(x3-x1+y2-y1)^2+(y3-y1+x1-x2)^2
								= 2x1^2 + 2y1^2 + x2^2 + y2^2 + x3^2 + y3^2
								+ (-2)x1x2 + (-2)x1y2 + 2y1x2 + (-2)y1y2 
								+ (-2)x1x3 + (-2)y1x3 + 2x1y3 + (-2)y1y3
								+ (-2)x2y3 + 2y2x3 
								*/
								if(!fixed_flag[offset_2] && !fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset_3],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_3],1*angle_energy_scale);

									Bmat.AddTo(x_index[offset_2],x_index[offset],-2*angle_energy_scale);
									Bmat.AddTo(y_index[offset_2],x_index[offset],-2*angle_energy_scale);
									Bmat.AddTo(x_index[offset_2],y_index[offset],2*angle_energy_scale);
									Bmat.AddTo(y_index[offset_2],y_index[offset],-2*angle_energy_scale);

									Bmat.AddTo(x_index[offset_3],x_index[offset],-2*angle_energy_scale);								
									Bmat.AddTo(x_index[offset_3],y_index[offset],-2*angle_energy_scale);
									Bmat.AddTo(y_index[offset_3],x_index[offset],2*angle_energy_scale);
									Bmat.AddTo(y_index[offset_3],y_index[offset],-2*angle_energy_scale);

									Gmat.AddTo(x_index[offset_2],y_index[offset_3],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],x_index[offset_2],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],x_index[offset_3],1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_3],y_index[offset_2],1*angle_energy_scale);
								}
								else if(!fixed_flag[offset_2] && fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*angle_energy_scale);

									Bmat.AddTo(x_index[offset_2],x_index[offset],-2*angle_energy_scale);								
									Bmat.AddTo(y_index[offset_2],x_index[offset],-2*angle_energy_scale);
									Bmat.AddTo(x_index[offset_2],y_index[offset],2*angle_energy_scale);
									Bmat.AddTo(y_index[offset_2],y_index[offset],-2*angle_energy_scale);

									Bmat.AddTo(x_index[offset_2],y_index[offset_3],-2*angle_energy_scale);
									Bmat.AddTo(y_index[offset_2],x_index[offset_3],2*angle_energy_scale);
								}
								else if(fixed_flag[offset_2] && !fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_3],x_index[offset_3],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_3],1*angle_energy_scale);

									Bmat.AddTo(x_index[offset_3],x_index[offset],-2*angle_energy_scale);
									Bmat.AddTo(x_index[offset_3],y_index[offset],-2*angle_energy_scale);
									Bmat.AddTo(y_index[offset_3],x_index[offset],2*angle_energy_scale);
									Bmat.AddTo(y_index[offset_3],y_index[offset],-2*angle_energy_scale);

									Bmat.AddTo(y_index[offset_3],x_index[offset_2],-2*angle_energy_scale);
									Bmat.AddTo(x_index[offset_3],y_index[offset_2],2*angle_energy_scale);
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
		/**********   ANGLE ENERGY End  **************/
		return true;
	}


	template<class T>
	bool ZQ_GridDeformation<T>::_addMatrixTo_ANGLE_XLOOP(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight, const ZQ_GridDeformationOptions::NeighborType type)
	{
		if(weight == 0)
			return true;

		/**********   ANGLE ENERGY Begin  **************/
		T angle_energy_scale = weight*weight;

		int dir_x[12] = {1,0,-1,0, 1,-1,-1,1, 2,0,-2,0};
		int dir_y[12] = {0,1,0,-1, 1,1,-1,-1, 0,2,0,-2};
		int neighbor_loop_num = 4;
		switch(type)
		{
		case ZQ_GridDeformationOptions::NEIGHBOR_4:
			neighbor_loop_num = 4;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_8:
			neighbor_loop_num = 8;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_12:
			neighbor_loop_num = 12;
			break;
		default:
			return false;
			break;
		}
		for(int i = 0;i < height;i++)
		{
			for(int j = 0;j < width;j++)
			{
				int offset = i*width+j;
				if(nouseful_flag[offset])
					continue;

				if(!fixed_flag[offset])
				{
					for(int ll = 0;ll < neighbor_loop_num;ll+=4)
					{
						for(int dd = 0;dd < 4;dd++)
						{
							int i_2 = i+dir_y[dd+ll];
							int j_2 = j+dir_x[dd+ll];
							int i_3 = i+dir_y[(dd+1)%4+ll];
							int j_3 = j+dir_x[(dd+1)%4+ll];
							j_2 = (j_2+width)%width;
							j_3 = (j_3+width)%width;
							int offset_2 = i_2*width+j_2;
							int offset_3 = i_3*width+j_3;
							if(0 <= i_2 && i_2 < height && 0 <= i_3 && i_3 < height 
								&& !nouseful_flag[offset_2] && !nouseful_flag[offset_3])
							{
								/*
								(x3-x1+y2-y1)^2+(y3-y1+x1-x2)^2
								= 2x1^2 + 2y1^2 + x2^2 + y2^2 + x3^2 + y3^2
								+ (-2)x1x2 + (-2)x1y2 + 2y1x2 + (-2)y1y2 
								+ (-2)x1x3 + (-2)y1x3 + 2x1y3 + (-2)y1y3
								+ (-2)x2y3 + 2y2x3 
								*/
								Gmat.AddTo(x_index[offset],x_index[offset],2*angle_energy_scale);
								Gmat.AddTo(y_index[offset],y_index[offset],2*angle_energy_scale);
								if(!fixed_flag[offset_2] && !fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset_3],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_3],1*angle_energy_scale);

									Gmat.AddTo(x_index[offset],x_index[offset_2],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_2],x_index[offset],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset],y_index[offset_2],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],x_index[offset],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset],x_index[offset_2],1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_2],y_index[offset],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset],y_index[offset_2],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset],-1*angle_energy_scale);

									Gmat.AddTo(x_index[offset],x_index[offset_3],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset],x_index[offset_3],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_3],y_index[offset],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset],y_index[offset_3],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],x_index[offset],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset],y_index[offset_3],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset],-1*angle_energy_scale);

									Gmat.AddTo(x_index[offset_2],y_index[offset_3],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],x_index[offset_2],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],x_index[offset_3],1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_3],y_index[offset_2],1*angle_energy_scale);
								}
								else if(!fixed_flag[offset_2] && fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*angle_energy_scale);

									Gmat.AddTo(x_index[offset],x_index[offset_2],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_2],x_index[offset],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset],y_index[offset_2],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],x_index[offset],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset],x_index[offset_2],1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_2],y_index[offset],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset],y_index[offset_2],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset],-1*angle_energy_scale);

									Bmat.AddTo(x_index[offset],x_index[offset_3],-2*angle_energy_scale);
									Bmat.AddTo(y_index[offset],x_index[offset_3],-2*angle_energy_scale);
									Bmat.AddTo(x_index[offset],y_index[offset_3],2*angle_energy_scale);
									Bmat.AddTo(y_index[offset],y_index[offset_3],-2*angle_energy_scale);

									Bmat.AddTo(x_index[offset_2],y_index[offset_3],-2*angle_energy_scale);
									Bmat.AddTo(y_index[offset_2],x_index[offset_3],2*angle_energy_scale);
								}
								else if(fixed_flag[offset_2] && !fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_3],x_index[offset_3],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_3],1*angle_energy_scale);

									Bmat.AddTo(x_index[offset],x_index[offset_2],-2*angle_energy_scale);
									Bmat.AddTo(x_index[offset],y_index[offset_2],-2*angle_energy_scale);
									Bmat.AddTo(y_index[offset],x_index[offset_2],2*angle_energy_scale);
									Bmat.AddTo(y_index[offset],y_index[offset_2],-2*angle_energy_scale);

									Gmat.AddTo(x_index[offset],x_index[offset_3],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset],x_index[offset_3],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_3],y_index[offset],-1*angle_energy_scale);
									Gmat.AddTo(x_index[offset],y_index[offset_3],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],x_index[offset],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset],y_index[offset_3],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset],-1*angle_energy_scale);

									Bmat.AddTo(y_index[offset_3],x_index[offset_2],-2*angle_energy_scale);
									Bmat.AddTo(x_index[offset_3],y_index[offset_2],2*angle_energy_scale);
								}
								else
								{
									Bmat.AddTo(x_index[offset],x_index[offset_2],-2*angle_energy_scale);	
									Bmat.AddTo(x_index[offset],y_index[offset_2],-2*angle_energy_scale);
									Bmat.AddTo(y_index[offset],x_index[offset_2],2*angle_energy_scale);
									Bmat.AddTo(y_index[offset],y_index[offset_2],-2*angle_energy_scale);

									Bmat.AddTo(x_index[offset],x_index[offset_3],-2*angle_energy_scale);
									Bmat.AddTo(y_index[offset],x_index[offset_3],-2*angle_energy_scale);
									Bmat.AddTo(x_index[offset],y_index[offset_3],2*angle_energy_scale);
									Bmat.AddTo(y_index[offset],y_index[offset_3],-2*angle_energy_scale);
								}
							}
						}
					}
				}
				//fixed[offset] == true
				else
				{
					for(int ll = 0;ll < neighbor_loop_num;ll+=4)
					{
						for(int dd = 0;dd < 4;dd++)
						{
							int i_2 = i+dir_y[dd+ll];
							int j_2 = j+dir_x[dd+ll];
							int i_3 = i+dir_y[(dd+1)%4+ll];
							int j_3 = j+dir_x[(dd+1)%4+ll];
							j_2 = (j_2+width)%width;
							j_3 = (j_3+width)%width;
							int offset_2 = i_2*width+j_2;
							int offset_3 = i_3*width+j_3;
							if(0 <= i_2 && i_2 < height && 0 <= i_3 && i_3 < height 
								&& !nouseful_flag[offset_2] && !nouseful_flag[offset_3])
							{
								/*
								(x3-x1+y2-y1)^2+(y3-y1+x1-x2)^2
								= 2x1^2 + 2y1^2 + x2^2 + y2^2 + x3^2 + y3^2
								+ (-2)x1x2 + (-2)x1y2 + 2y1x2 + (-2)y1y2 
								+ (-2)x1x3 + (-2)y1x3 + 2x1y3 + (-2)y1y3
								+ (-2)x2y3 + 2y2x3 
								*/
								if(!fixed_flag[offset_2] && !fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_3],x_index[offset_3],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_3],1*angle_energy_scale);

									Bmat.AddTo(x_index[offset_2],x_index[offset],-2*angle_energy_scale);
									Bmat.AddTo(y_index[offset_2],x_index[offset],-2*angle_energy_scale);
									Bmat.AddTo(x_index[offset_2],y_index[offset],2*angle_energy_scale);
									Bmat.AddTo(y_index[offset_2],y_index[offset],-2*angle_energy_scale);

									Bmat.AddTo(x_index[offset_3],x_index[offset],-2*angle_energy_scale);								
									Bmat.AddTo(x_index[offset_3],y_index[offset],-2*angle_energy_scale);
									Bmat.AddTo(y_index[offset_3],x_index[offset],2*angle_energy_scale);
									Bmat.AddTo(y_index[offset_3],y_index[offset],-2*angle_energy_scale);

									Gmat.AddTo(x_index[offset_2],y_index[offset_3],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],x_index[offset_2],-1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],x_index[offset_3],1*angle_energy_scale);
									Gmat.AddTo(x_index[offset_3],y_index[offset_2],1*angle_energy_scale);
								}
								else if(!fixed_flag[offset_2] && fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*angle_energy_scale);

									Bmat.AddTo(x_index[offset_2],x_index[offset],-2*angle_energy_scale);								
									Bmat.AddTo(y_index[offset_2],x_index[offset],-2*angle_energy_scale);
									Bmat.AddTo(x_index[offset_2],y_index[offset],2*angle_energy_scale);
									Bmat.AddTo(y_index[offset_2],y_index[offset],-2*angle_energy_scale);

									Bmat.AddTo(x_index[offset_2],y_index[offset_3],-2*angle_energy_scale);
									Bmat.AddTo(y_index[offset_2],x_index[offset_3],2*angle_energy_scale);
								}
								else if(fixed_flag[offset_2] && !fixed_flag[offset_3])
								{
									Gmat.AddTo(x_index[offset_3],x_index[offset_3],1*angle_energy_scale);
									Gmat.AddTo(y_index[offset_3],y_index[offset_3],1*angle_energy_scale);

									Bmat.AddTo(x_index[offset_3],x_index[offset],-2*angle_energy_scale);
									Bmat.AddTo(x_index[offset_3],y_index[offset],-2*angle_energy_scale);
									Bmat.AddTo(y_index[offset_3],x_index[offset],2*angle_energy_scale);
									Bmat.AddTo(y_index[offset_3],y_index[offset],-2*angle_energy_scale);

									Bmat.AddTo(y_index[offset_3],x_index[offset_2],-2*angle_energy_scale);
									Bmat.AddTo(x_index[offset_3],y_index[offset_2],2*angle_energy_scale);
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
		/**********   ANGLE ENERGY End  **************/
		return true;
	}

	template<class T>
	bool ZQ_GridDeformation<T>::_addMatrixTo_DISTANCE(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight)
	{
		T dist_weight = option.distance_weight*option.distance_weight;

		int dir_x[4] = {1,0,-1,0};
		int dir_y[4] = {0,1,0,-1};
		for(int i = 0;i < height;i++)
		{
			for(int j = 0;j < width;j++)
			{
				int offset = i*width+j;
				if(nouseful_flag[offset])
					continue;

				for(int dd = 0;dd < 4;dd++)
				{
					int i_2 = i+dir_y[dd];
					int j_2 = j+dir_x[dd];
					int offset_2 = i_2*width+j_2;
					if(0 <= i_2 && i_2 < height && 0 <= j_2 && j_2 < width && !nouseful_flag[offset_2])
					{
						/*
						((x1-x2)-(X1-X2))^2 + ((y1-y2)-(Y1-Y2))^2
						= x1^2 + y1^2 + x2^2 + y2^2
						+ (-2)x1x2 + (-2)y1y2
						+ x1*(-2X1+2X2) + y1*(-2Y1+2Y2)
						+ x2*(-2X2+2X1) + y2*(-2Y2+2Y1)
						+ const_value
						*/


						if(!fixed_flag[offset] && !fixed_flag[offset_2])
						{
							Gmat.AddTo(x_index[offset],x_index[offset],1*dist_weight);
							Gmat.AddTo(y_index[offset],y_index[offset],1*dist_weight);
							Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*dist_weight);
							Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*dist_weight);

							Gmat.AddTo(x_index[offset],x_index[offset_2],-1*dist_weight);
							Gmat.AddTo(x_index[offset_2],x_index[offset],-1*dist_weight);
							Gmat.AddTo(y_index[offset],y_index[offset_2],-1*dist_weight);
							Gmat.AddTo(y_index[offset_2],y_index[offset],-1*dist_weight);
						}
						else if(!fixed_flag[offset] && fixed_flag[offset_2])
						{
							Gmat.AddTo(x_index[offset],x_index[offset],1*dist_weight);
							Gmat.AddTo(y_index[offset],y_index[offset],1*dist_weight);

							Bmat.AddTo(x_index[offset],x_index[offset_2],-2*dist_weight);
							Bmat.AddTo(y_index[offset],y_index[offset_2],-2*dist_weight);
						}
						else if(fixed_flag[offset] && !fixed_flag[offset_2])
						{
							Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*dist_weight);
							Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*dist_weight);

							Bmat.AddTo(x_index[offset_2],x_index[offset],-2*dist_weight);
							Bmat.AddTo(y_index[offset_2],y_index[offset],-2*dist_weight);
						}
						else 
						{
						}
					}
				}
			}
		}
		return true;
	}


	template<class T>
	bool ZQ_GridDeformation<T>::_addMatrixTo_DISTANCE_XLOOP(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight)
	{
		float dist_weight = option.distance_weight*option.distance_weight;

		int dir_x[4] = {1,0,-1,0};
		int dir_y[4] = {0,1,0,-1};
		for(int i = 0;i < height;i++)
		{
			for(int j = 0;j < width;j++)
			{
				int offset = i*width+j;
				if(nouseful_flag[offset])
					continue;

				for(int dd = 0;dd < 4;dd++)
				{
					int i_2 = i+dir_y[dd];
					int j_2 = j+dir_x[dd];
					j_2 = (j_2+width)%width;
					int offset_2 = i_2*width+j_2;
					if(0 <= i_2 && i_2 < height && !nouseful_flag[offset_2])
					{
						/*
						((x1-x2)-(X1-X2))^2 + ((y1-y2)-(Y1-Y2))^2
						= x1^2 + y1^2 + x2^2 + y2^2
						+ (-2)x1x2 + (-2)y1y2
						+ x1*(-2X1+2X2) + y1*(-2Y1+2Y2)
						+ x2*(-2X2+2X1) + y2*(-2Y2+2Y1)
						+ const_value
						*/


						if(!fixed_flag[offset] && !fixed_flag[offset_2])
						{
							Gmat.AddTo(x_index[offset],x_index[offset],1*dist_weight);
							Gmat.AddTo(y_index[offset],y_index[offset],1*dist_weight);
							Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*dist_weight);
							Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*dist_weight);

							Gmat.AddTo(x_index[offset],x_index[offset_2],-1*dist_weight);
							Gmat.AddTo(x_index[offset_2],x_index[offset],-1*dist_weight);
							Gmat.AddTo(y_index[offset],y_index[offset_2],-1*dist_weight);
							Gmat.AddTo(y_index[offset_2],y_index[offset],-1*dist_weight);
						}
						else if(!fixed_flag[offset] && fixed_flag[offset_2])
						{
							Gmat.AddTo(x_index[offset],x_index[offset],1*dist_weight);
							Gmat.AddTo(y_index[offset],y_index[offset],1*dist_weight);

							Bmat.AddTo(x_index[offset],x_index[offset_2],-2*dist_weight);
							Bmat.AddTo(y_index[offset],y_index[offset_2],-2*dist_weight);
						}
						else if(fixed_flag[offset] && !fixed_flag[offset_2])
						{
							Gmat.AddTo(x_index[offset_2],x_index[offset_2],1*dist_weight);
							Gmat.AddTo(y_index[offset_2],y_index[offset_2],1*dist_weight);

							Bmat.AddTo(x_index[offset_2],x_index[offset],-2*dist_weight);
							Bmat.AddTo(y_index[offset_2],y_index[offset],-2*dist_weight);
						}
						else 
						{
						}
					}
				}
			}
		}
		return true;
	}

	template<class T>
	bool ZQ_GridDeformation<T>::_addMatrixTo_ARAP_VERT(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight, const ZQ_GridDeformationOptions::NeighborType type)
	{

		T arap_weight = weight*weight;

		int dir_x[12] = {1,0,-1,0, 1,-1,-1,1, 2,0,-2,0};
		int dir_y[12] = {0,1,0,-1, 1,1,-1,-1, 0,2,0,-2};
		int neighbor_loop_num = 4;
		switch(type)
		{
		case ZQ_GridDeformationOptions::NEIGHBOR_4:
			neighbor_loop_num = 4;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_8:
			neighbor_loop_num = 8;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_12:
			neighbor_loop_num = 12;
			break;
		default:
			return false;
			break;
		}

		for(int i = 0;i < height;i++)
		{
			for(int j = 0;j < width;j++)
			{
				int offset = i*width+j;
				if(nouseful_flag[offset])
					continue;

				for(int dd = 0;dd < neighbor_loop_num;dd++)
				{
					int i_2 = i+dir_y[dd];
					int j_2 = j+dir_x[dd];
					int offset_2 = i_2*width+j_2;

					if(0 <= j_2 && j_2 < width && 0 <= i_2 && i_2 < height && !nouseful_flag[offset_2])
					{
						if(!fixed_flag[offset])
						{
							if(!fixed_flag[offset_2])
							{
								Gmat.AddTo(x_index[offset],x_index[offset],arap_weight);
								Gmat.AddTo(y_index[offset],y_index[offset],arap_weight);

								Gmat.AddTo(x_index[offset_2],x_index[offset_2],arap_weight);
								Gmat.AddTo(y_index[offset_2],y_index[offset_2],arap_weight);

								Gmat.AddTo(x_index[offset],x_index[offset_2],-arap_weight);
								Gmat.AddTo(x_index[offset_2],x_index[offset],-arap_weight);
								Gmat.AddTo(y_index[offset],y_index[offset_2],-arap_weight);
								Gmat.AddTo(y_index[offset_2],y_index[offset],-arap_weight);
							}
							else
							{
								Gmat.AddTo(x_index[offset],x_index[offset],arap_weight);
								Gmat.AddTo(y_index[offset],y_index[offset],arap_weight);

								Bmat.AddTo(x_index[offset],x_index[offset_2],-2*arap_weight);
								Bmat.AddTo(y_index[offset],y_index[offset_2],-2*arap_weight);
							}
						}
						else
						{
							if(!fixed_flag[offset_2])
							{
								Gmat.AddTo(x_index[offset_2],x_index[offset_2],arap_weight);
								Gmat.AddTo(y_index[offset_2],y_index[offset_2],arap_weight);

								Bmat.AddTo(x_index[offset_2],x_index[offset],-2*arap_weight);
								Bmat.AddTo(y_index[offset_2],y_index[offset],-2*arap_weight);
							}
							else
							{

							}
						}
					}
				}
			}
		}

		return true;
	}


	template<class T>
	bool ZQ_GridDeformation<T>::_addMatrixTo_ARAP_VERT_XLOOP(ZQ_SparseMatrix<T>& Gmat, ZQ_SparseMatrix<T>& Bmat, const T weight, const ZQ_GridDeformationOptions::NeighborType type)
	{

		T arap_weight = weight*weight;

		int dir_x[12] = {1,0,-1,0, 1,-1,-1,1, 2,0,-2,0};
		int dir_y[12] = {0,1,0,-1, 1,1,-1,-1, 0,2,0,-2};
		int neighbor_loop_num = 4;
		switch(type)
		{
		case ZQ_GridDeformationOptions::NEIGHBOR_4:
			neighbor_loop_num = 4;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_8:
			neighbor_loop_num = 8;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_12:
			neighbor_loop_num = 12;
			break;
		default:
			return false;
			break;
		}
		for(int i = 0;i < height;i++)
		{
			for(int j = 0;j < width;j++)
			{
				int offset = i*width+j;
				if(nouseful_flag[offset])
					continue;

				for(int dd = 0;dd < neighbor_loop_num;dd++)
				{
					int i_2 = i+dir_y[dd];
					int j_2 = j+dir_x[dd];
					j_2 = (j_2+width)%width;
					int offset_2 = i_2*width+j_2;

					if(0 <= i_2 && i_2 < height && !nouseful_flag[offset_2])
					{
						if(!fixed_flag[offset])
						{
							if(!fixed_flag[offset_2])
							{
								Gmat.AddTo(x_index[offset],x_index[offset],arap_weight);
								Gmat.AddTo(y_index[offset],y_index[offset],arap_weight);

								Gmat.AddTo(x_index[offset_2],x_index[offset_2],arap_weight);
								Gmat.AddTo(y_index[offset_2],y_index[offset_2],arap_weight);

								Gmat.AddTo(x_index[offset],x_index[offset_2],-arap_weight);
								Gmat.AddTo(x_index[offset_2],x_index[offset],-arap_weight);
								Gmat.AddTo(y_index[offset],y_index[offset_2],-arap_weight);
								Gmat.AddTo(y_index[offset_2],y_index[offset],-arap_weight);
							}
							else
							{
								Gmat.AddTo(x_index[offset],x_index[offset],arap_weight);
								Gmat.AddTo(y_index[offset],y_index[offset],arap_weight);

								Bmat.AddTo(x_index[offset],x_index[offset_2],-2*arap_weight);
								Bmat.AddTo(y_index[offset],y_index[offset_2],-2*arap_weight);
							}
						}
						else
						{
							if(!fixed_flag[offset_2])
							{
								Gmat.AddTo(x_index[offset_2],x_index[offset_2],arap_weight);
								Gmat.AddTo(y_index[offset_2],y_index[offset_2],arap_weight);

								Bmat.AddTo(x_index[offset_2],x_index[offset],-2*arap_weight);
								Bmat.AddTo(y_index[offset_2],y_index[offset],-2*arap_weight);
							}
							else
							{

							}
						}
					}
				}
			}
		}

		return true;
	}

	template<class T>
	void ZQ_GridDeformation<T>::_buildMatrixInvFC()
	{
		/*
		E = (x1-X1)^2 + (y1-Y1)^2 + (x2-X2)^2 + (y2-Y2)^2 + (x1+y1-y2-X3)^2 + (y1+x2-x1-Y3)^2
		\partial E = 0 ==>
		\partial_x1 : (x1-X1)+(x1+y1-y2-X3)+(x1-y1-x2+Y3) = 0    ==>  3x1-x2-y2 = X1+X3-Y3
		\partial_y1 : (y1-Y1)+(x1+y1-y2-X3)+(y1+x2-x1-Y3) = 0    ==>  3y1+x2-y2 = Y1+X3+Y3
		\partial_x2 : (x2-X2)+(y1+x2-x1-Y3) = 0                  ==> -x1+y1+2x2 = X2+Y3
		\partial_y2 : (y2-Y2)+(y2-x1-y1+X3) = 0                  ==> -x1-y1+2y2 = Y2-X3
		*/

		ZQ_Matrix<T> F(4,4),invF(4,4);
		F.SetData(0,0,3);	F.SetData(0,2,-1);	F.SetData(0,3,-1);
		F.SetData(1,1,3);	F.SetData(1,2,1);	F.SetData(1,3,-1);
		F.SetData(2,0,-1);	F.SetData(2,1,1);	F.SetData(2,2,2);
		F.SetData(3,0,-1);	F.SetData(3,1,-1);	F.SetData(3,3,2);
		ZQ_SVD::Invert(F,invF);

		T C_m[24] = 
		{
			1,0,0,0,1,-1,
			0,1,0,0,1,1,
			0,0,1,0,0,1,
			0,0,0,1,-1,0
		};
		T invF_m[16];
		bool flag;
		for(int i = 0;i < 4;i++)
		{
			for(int j = 0;j < 4;j++)
				invF_m[i*4+j] = invF.GetData(i,j,flag);
		}

		ZQ_MathBase::MatrixMul(invF_m,C_m,4,4,6,invFC_m);

	}

	template<class T>
	bool ZQ_GridDeformation<T>::_bestFitTriangle(const T input[6], bool fixed1, bool fixed2, bool fixed3, const T distance, T output[6])
	{
		ZQ_MathBase::MatrixMul(invFC_m,input,4,6,1,output);
		output[4] = output[0] + output[1] - output[3];
		output[5] = output[1] + output[2] - output[0];
		T len = sqrt((output[2]-output[0])*(output[2]-output[0]) + (output[3]-output[1])*(output[3]-output[1]));
		if(len == 0)
			return false;

		T scale = distance/len;

		if(!fixed1 && !fixed2 && !fixed3)
		{
			T cx = (output[0]+output[2]+output[4])/3;
			T cy = (output[1]+output[3]+output[5])/3;
			for(int i = 0;i < 3;i++)
			{
				output[i*2+0] = (output[i*2+0]-cx)*scale+cx;
				output[i*2+1] = (output[i*2+1]-cy)*scale+cy;
			}
		}
		else if(fixed1 && !fixed2 && !fixed3)
		{
			T shift_x = input[0] - output[0];
			T shift_y = input[1] - output[1];
			for(int i = 0;i < 3;i++)
			{
				output[i*2+0] += shift_x;
				output[i*2+1] += shift_y;

				output[i*2+0] = (output[i*2+0]-input[0])*scale+input[0];
				output[i*2+1] = (output[i*2+1]-input[1])*scale+input[1];
			}
		}
		else if(!fixed1 && fixed2 && !fixed3)
		{
			T shift_x = input[2] - output[2];
			T shift_y = input[3] - output[3];
			for(int i = 0;i < 3;i++)
			{
				output[i*2+0] += shift_x;
				output[i*2+1] += shift_y;

				output[i*2+0] = (output[i*2+0]-input[2])*scale+input[2];
				output[i*2+1] = (output[i*2+1]-input[3])*scale+input[3];
			}
		}
		else if(!fixed1 && !fixed2 && fixed3)
		{
			T shift_x = input[4] - output[4];
			T shift_y = input[5] - output[5];
			for(int i = 0;i < 3;i++)
			{
				output[i*2+0] += shift_x;
				output[i*2+1] += shift_y;

				output[i*2+0] = (output[i*2+0]-input[4])*scale+input[4];
				output[i*2+1] = (output[i*2+1]-input[5])*scale+input[5];
			}
		}
		else if(fixed1 && fixed2 && !fixed3)
		{
			memcpy(output,input,sizeof(T)*6);
			T dir1_2[2] = {input[2]-input[0],input[3]-input[1]};
			T dir[2] = {-dir1_2[1],dir1_2[0]};
			T len_d = sqrt(dir[0]*dir[0]+dir[1]*dir[1]);
			if(len_d == 0)
			{
				return false;
			}

			output[4] = input[0] + dir[0]/len_d*distance;
			output[5] = input[1] + dir[1]/len_d*distance;
		}
		else if(fixed1 && !fixed2 && fixed3)
		{
			memcpy(output,input,sizeof(T)*6);
			T dir1_3[2] = {input[4]-input[0],input[5]-input[1]};
			T dir[2] = {dir1_3[1],-dir1_3[0]};
			T len_d = sqrt(dir[0]*dir[0]+dir[1]*dir[1]);
			if(len_d == 0)
			{
				return false;
			}

			output[2] = input[0] + dir[0]/len_d*distance;
			output[3] = input[1] + dir[1]/len_d*distance;

		}
		else if(!fixed1 && fixed2 && fixed3)
		{
			memcpy(output,input,sizeof(T)*6);
		}
		else
		{
			memcpy(output,input,sizeof(T)*6);
		}

		return true;
	}

	template<class T>
	bool ZQ_GridDeformation<T>::_buildMatrix_for_scaling()
	{
		int taucs_flag;
		if(strcmp(typeid(T).name(),"float") == 0)
			taucs_flag = TAUCS_SINGLE;
		else if(strcmp(typeid(T).name(),"double") == 0)
			taucs_flag = TAUCS_DOUBLE;
		else
			return false;

		ZQ_SparseMatrix<T> Hmat(unknown_num,unknown_num);
		ZQ_SparseMatrix<T> Dmat(unknown_num,fixed_num);

		int dir_x[4] = {1,0,-1,0};
		int dir_y[4] = {0,1,0,-1};
		for(int i = 0;i < height;i++)
		{
			for(int j = 0;j < width;j++)
			{
				int offset = i*width+j;
				if(nouseful_flag[offset])
					continue;

				for(int dd = 0;dd < 4;dd++)
				{
					int i_2 = i+dir_y[dd];
					int j_2 = j+dir_x[dd];
					int i_3 = i+dir_y[(dd+1)%4];
					int j_3 = j+dir_x[(dd+1)%4];
					if(0 <= i_2 && i_2 < height && 0 <= j_2 && j_2 < width && 0 <= i_3 && i_3 < height && 0 <= j_3 && j_3 < width
						&& !nouseful_flag[i_2*width+j_2] && !nouseful_flag[i_3*width+j_3])
					{
						int offset_2 = i_2*width+j_2;
						int offset_3 = i_3*width+j_3;

						/*
						((x1-x2)-(X1-X2))^2 + ((y1-y2)-(Y1-Y2))^2
						+ ((x2-x3)-(X2-X3))^2 + ((y2-y3)-(Y2-Y3))^2
						+ ((x3-x1)-(X3-X1))^2 + ((y3-y1)-(Y3-Y1))^2
						= 2x1^2 + 2y1^2 + 2x2^2 + 2y2^2 + 2x3^2 +2y3^2
						+ (-2)x1x2 + (-2)y1y2
						+ (-2)x1x3 + (-2)y1y3
						+ (-2)x2x3 + (-2)y2y3
						+ x1*(-4X1+2X2+2X3) + y1*(-4Y1+2Y2+2Y3)
						+ x2*(-4X2+2X1+2X3) + y2*(-4Y2+2Y1+2Y3)
						+ x3*(-4X3+2X1+2X2) + y3*(-4Y3+2Y1+2Y2)
						+ const_value
						*/

						if(!fixed_flag[offset])
						{
							if(!fixed_flag[offset_2] && !fixed_flag[offset_3])
							{
								Hmat.AddTo(x_index[offset],x_index[offset],2);
								Hmat.AddTo(y_index[offset],y_index[offset],2);
								Hmat.AddTo(x_index[offset_2],x_index[offset_2],2);
								Hmat.AddTo(y_index[offset_2],y_index[offset_2],2);
								Hmat.AddTo(x_index[offset_3],x_index[offset_3],2);
								Hmat.AddTo(y_index[offset_3],y_index[offset_3],2);

								Hmat.AddTo(x_index[offset],x_index[offset_2],-1);
								Hmat.AddTo(x_index[offset_2],x_index[offset],-1);
								Hmat.AddTo(y_index[offset],y_index[offset_2],-1);
								Hmat.AddTo(y_index[offset_2],y_index[offset],-1);

								Hmat.AddTo(x_index[offset],x_index[offset_3],-1);
								Hmat.AddTo(x_index[offset_3],x_index[offset],-1);
								Hmat.AddTo(y_index[offset],y_index[offset_3],-1);
								Hmat.AddTo(y_index[offset_3],y_index[offset],-1);

								Hmat.AddTo(x_index[offset_2],x_index[offset_3],-1);
								Hmat.AddTo(x_index[offset_3],x_index[offset_2],-1);
								Hmat.AddTo(y_index[offset_2],y_index[offset_3],-1);
								Hmat.AddTo(y_index[offset_3],y_index[offset_2],-1);

							}
							else if(!fixed_flag[offset_2] && fixed_flag[offset_3])
							{
								Hmat.AddTo(x_index[offset],x_index[offset],2);
								Hmat.AddTo(y_index[offset],y_index[offset],2);
								Hmat.AddTo(x_index[offset_2],x_index[offset_2],2);
								Hmat.AddTo(y_index[offset_2],y_index[offset_2],2);

								Hmat.AddTo(x_index[offset],x_index[offset_2],-1);
								Hmat.AddTo(x_index[offset_2],x_index[offset],-1);
								Hmat.AddTo(y_index[offset],y_index[offset_2],-1);
								Hmat.AddTo(y_index[offset_2],y_index[offset],-1);

								Dmat.AddTo(x_index[offset],x_index[offset_3],-2);
								Dmat.AddTo(y_index[offset],y_index[offset_3],-2);

								Dmat.AddTo(x_index[offset_2],x_index[offset_3],-2);
								Dmat.AddTo(y_index[offset_2],y_index[offset_3],-2);
							}
							else if(fixed_flag[offset_2] && !fixed_flag[offset_3])
							{
								Hmat.AddTo(x_index[offset],x_index[offset],2);
								Hmat.AddTo(y_index[offset],y_index[offset],2);
								Hmat.AddTo(x_index[offset_3],x_index[offset_3],2);
								Hmat.AddTo(y_index[offset_3],y_index[offset_3],2);

								Dmat.AddTo(x_index[offset],x_index[offset_2],-2);
								Dmat.AddTo(y_index[offset],y_index[offset_2],-2);

								Hmat.AddTo(x_index[offset],x_index[offset_3],-1);
								Hmat.AddTo(x_index[offset_3],x_index[offset],-1);
								Hmat.AddTo(y_index[offset],y_index[offset_3],-1);
								Hmat.AddTo(y_index[offset_3],y_index[offset],-1);

								Dmat.AddTo(x_index[offset_3],x_index[offset_2],-2);
								Dmat.AddTo(y_index[offset_3],y_index[offset_2],-2);
							}
							else
							{
								Hmat.AddTo(x_index[offset],x_index[offset],2);
								Hmat.AddTo(y_index[offset],y_index[offset],2);

								Dmat.AddTo(x_index[offset],x_index[offset_2],-2);
								Dmat.AddTo(y_index[offset],y_index[offset_2],-2);

								Dmat.AddTo(x_index[offset],x_index[offset_3],-2);
								Dmat.AddTo(y_index[offset],y_index[offset_3],-2);
							}
						}
						else
						{
							if(!fixed_flag[offset_2] && !fixed_flag[offset_3])
							{
								Hmat.AddTo(x_index[offset_2],x_index[offset_2],2);
								Hmat.AddTo(y_index[offset_2],y_index[offset_2],2);
								Hmat.AddTo(x_index[offset_3],x_index[offset_3],2);
								Hmat.AddTo(y_index[offset_3],y_index[offset_3],2);

								Dmat.AddTo(x_index[offset_2],x_index[offset],-2);
								Dmat.AddTo(y_index[offset_2],y_index[offset],-2);

								Dmat.AddTo(x_index[offset_3],x_index[offset],-2);
								Dmat.AddTo(y_index[offset_3],y_index[offset],-2);

								Hmat.AddTo(x_index[offset_2],x_index[offset_3],-1);
								Hmat.AddTo(x_index[offset_3],x_index[offset_2],-1);
								Hmat.AddTo(y_index[offset_2],y_index[offset_3],-1);
								Hmat.AddTo(y_index[offset_3],y_index[offset_2],-1);

							}
							else if(!fixed_flag[offset_2] && fixed_flag[offset_3])
							{
								Hmat.AddTo(x_index[offset_2],x_index[offset_2],2);
								Hmat.AddTo(y_index[offset_2],y_index[offset_2],2);

								Dmat.AddTo(x_index[offset_2],x_index[offset],-2);
								Dmat.AddTo(y_index[offset_2],y_index[offset],-2);

								Dmat.AddTo(x_index[offset_2],x_index[offset_3],-2);
								Dmat.AddTo(y_index[offset_2],y_index[offset_3],-2);
							}
							else if(fixed_flag[offset_2] && !fixed_flag[offset_3])
							{
								Hmat.AddTo(x_index[offset_3],x_index[offset_3],2);
								Hmat.AddTo(y_index[offset_3],y_index[offset_3],2);

								Dmat.AddTo(x_index[offset_3],x_index[offset],-2);
								Dmat.AddTo(y_index[offset_3],y_index[offset],-2);

								Dmat.AddTo(x_index[offset_3],x_index[offset_2],-2);
								Dmat.AddTo(y_index[offset_3],y_index[offset_2],-2);
							}
							else
							{
							}
						}
					}
				}
			}
		}

		if(H)
			ZQ_TaucsBase::ZQ_taucs_ccs_free(H);
		H = Hmat.ExportCCS(taucs_flag);
		if(D)
			ZQ_TaucsBase::ZQ_taucs_ccs_free(D);
		D = Dmat.ExportCCS(taucs_flag);
		return true;
	}

	template<class T>
	bool ZQ_GridDeformation<T>::_estimateR_for_ARAP_VERT(const T* init_coord, std::vector<_mat22>& Rmats, const T scale, const ZQ_GridDeformationOptions::NeighborType type)
	{
		int dir_x[12] = {1,0,-1,0, 1,-1,-1,1, 2,0,-2,0};
		int dir_y[12] = {0,1,0,-1, 1,1,-1,-1, 0,2,0,-2};
		int neighbor_loop_num = 4;
		switch(type)
		{
		case ZQ_GridDeformationOptions::NEIGHBOR_4:
			neighbor_loop_num = 4;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_8:
			neighbor_loop_num = 8;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_12:
			neighbor_loop_num = 12;
			break;
		default:
			return false;
			break;
		}

		Rmats.clear();
		Rmats.resize(width*height);

		for(int i = 0;i < height;i++)
		{
			for(int j = 0;j < width;j++)
			{
				int offset = i*width+j;
				if(nouseful_flag[offset])
					continue;

				T S[4] = {0,0,0,0};
				for(int dd = 0; dd < neighbor_loop_num; dd++)
				{
					int i_2 = i+dir_y[dd];
					int j_2 = j+dir_x[dd];
					int offset_2 = i_2*width+j_2;
					if(0 <= i_2 && i_2 < height && 0 <= j_2 && j_2 < width && !nouseful_flag[offset_2])
					{
						float cur_v[2] = {init_coord[offset_2*2]-init_coord[offset*2],init_coord[offset_2*2+1]-init_coord[offset*2+1]};
						S[0] += dir_x[dd]*cur_v[0]*scale;
						S[1] += dir_x[dd]*cur_v[1]*scale;
						S[2] += dir_y[dd]*cur_v[0]*scale;
						S[3] += dir_y[dd]*cur_v[1]*scale;
					}
				}

				ZQ_Matrix<T> Smat(2,2),U(2,2),D(2,2),V(2,2);
				Smat.SetData(0,0,S[0]);
				Smat.SetData(0,1,S[1]);
				Smat.SetData(1,0,S[2]);
				Smat.SetData(1,1,S[3]);

				ZQ_SVD::Decompose(Smat,U,D,V);
				ZQ_Matrix<T> Rmat = V*U.GetTransposeMatrix();
				bool flag;
				Rmats[offset].val[0] = Rmat.GetData(0,0,flag);
				Rmats[offset].val[1] = Rmat.GetData(0,1,flag);
				Rmats[offset].val[2] = Rmat.GetData(1,0,flag);
				Rmats[offset].val[3] = Rmat.GetData(1,1,flag);
			}
		}

		return true;
	}


	template<class T>
	bool ZQ_GridDeformation<T>::_estimateR_for_ARAP_VERT_XLOOP(const T* init_coord, std::vector<_mat22>& Rmats, const T scale, const ZQ_GridDeformationOptions::NeighborType type)
	{
		int dir_x[12] = {1,0,-1,0, 1,-1,-1,1, 2,0,-2,0};
		int dir_y[12] = {0,1,0,-1, 1,1,-1,-1, 0,2,0,-2};
		int neighbor_loop_num = 4;
		switch(type)
		{
		case ZQ_GridDeformationOptions::NEIGHBOR_4:
			neighbor_loop_num = 4;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_8:
			neighbor_loop_num = 8;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_12:
			neighbor_loop_num = 12;
			break;
		default:
			return false;
			break;
		}

		Rmats.clear();
		Rmats.resize(width*height);

		for(int i = 0;i < height;i++)
		{
			for(int j = 0;j < width;j++)
			{
				int offset = i*width+j;
				if(nouseful_flag[offset])
					continue;

				T S[4] = {0,0,0,0};
				for(int dd = 0; dd < neighbor_loop_num; dd++)
				{
					int i_2 = i+dir_y[dd];
					int j_2 = j+dir_x[dd];
					j_2 = (j_2+width)%width;
					int offset_2 = i_2*width+j_2;
					if(0 <= i_2 && i_2 < height && !nouseful_flag[offset_2])
					{
						float cur_v[2] = {init_coord[offset_2*2]-init_coord[offset*2],init_coord[offset_2*2+1]-init_coord[offset*2+1]};
						S[0] += dir_x[dd]*cur_v[0]*scale;
						S[1] += dir_x[dd]*cur_v[1]*scale;
						S[2] += dir_y[dd]*cur_v[0]*scale;
						S[3] += dir_y[dd]*cur_v[1]*scale;
					}
				}

				ZQ_Matrix<T> Smat(2,2),U(2,2),D(2,2),V(2,2);
				Smat.SetData(0,0,S[0]);
				Smat.SetData(0,1,S[1]);
				Smat.SetData(1,0,S[2]);
				Smat.SetData(1,1,S[3]);

				ZQ_SVD::Decompose(Smat,U,D,V);
				ZQ_Matrix<T> Rmat = V*U.GetTransposeMatrix();
				bool flag;
				Rmats[offset].val[0] = Rmat.GetData(0,0,flag);
				Rmats[offset].val[1] = Rmat.GetData(0,1,flag);
				Rmats[offset].val[2] = Rmat.GetData(1,0,flag);
				Rmats[offset].val[3] = Rmat.GetData(1,1,flag);
			}
		}

		return true;
	}

	template<class T>
	bool ZQ_GridDeformation<T>::_solve_for_ARAP_VERT(const T* init_coord, const std::vector<_mat22>& Rmats, T* out_coord, const int iteration, const T scale, const ZQ_GridDeformationOptions::NeighborType type)
	{

		int dir_x[12] = {1,0,-1,0, 1,-1,-1,1, 2,0,-2,0};
		int dir_y[12] = {0,1,0,-1, 1,1,-1,-1, 0,2,0,-2};
		int neighbor_loop_num = 4;
		switch(type)
		{
		case ZQ_GridDeformationOptions::NEIGHBOR_4:
			neighbor_loop_num = 4;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_8:
			neighbor_loop_num = 8;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_12:
			neighbor_loop_num = 12;
			break;
		default:
			return false;
			break;
		}

		T* init_unknown = new T[unknown_num];
		T* fixed_val = new T[fixed_num];
		for(int pp = 0;pp < width*height;pp++)
		{
			if(!nouseful_flag[pp])
			{
				if(fixed_flag[pp])
				{
					fixed_val[x_index[pp]] = init_coord[pp*2];
					fixed_val[y_index[pp]] = init_coord[pp*2+1];
				}
				else
				{
					init_unknown[x_index[pp]] = init_coord[pp*2];
					init_unknown[y_index[pp]] = init_coord[pp*2+1];
				}
			}

		}

		T* f = new T[unknown_num];
		ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec(B,fixed_val,f);

		for(int i = 0;i < height;i++)
		{
			for(int j = 0;j < width;j++)
			{
				int offset = i*width+j;
				if(nouseful_flag[offset])
					continue;

				for(int dd = 0; dd < neighbor_loop_num; dd++)
				{
					int i_2 = i+dir_y[dd];
					int j_2 = j+dir_x[dd];
					int offset_2 = i_2*width+j_2;
					if(0 <= i_2 && i_2 < height && 0 <= j_2 && j_2 < width && !nouseful_flag[offset_2])
					{
						float old_v[2] = {dir_x[dd]*scale,dir_y[dd]*scale};
						float cur_v[2] = {
							Rmats[offset].val[0] * old_v[0] + Rmats[offset].val[1] * old_v[1],
							Rmats[offset].val[2] * old_v[0] + Rmats[offset].val[3] * old_v[1]
						};

						if(!fixed_flag[offset])
						{
							f[x_index[offset]] += 2*cur_v[0];
							f[y_index[offset]] += 2*cur_v[1];
						}
						if(!fixed_flag[offset_2])
						{
							f[x_index[offset_2]] -= 2*cur_v[0];
							f[y_index[offset_2]] -= 2*cur_v[1];
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

		for(int i = 0;i < width*height;i++)
		{
			if(nouseful_flag[i])
				continue;
			if(fixed_flag[i])
			{
				out_coord[i*2] = init_coord[i*2];
				out_coord[i*2+1] = init_coord[i*2+1];
			}
			else
			{
				out_coord[i*2] = unknown_vals[x_index[i]];
				out_coord[i*2+1] = unknown_vals[y_index[i]];
			}
		}

		delete []f;
		delete []unknown_vals;
		delete []init_unknown;
		delete []fixed_val;

		return true;
	}


	template<class T>
	bool ZQ_GridDeformation<T>::_solve_for_ARAP_VERT_XLOOP(const T* init_coord, const std::vector<_mat22>& Rmats, T* out_coord, const int iteration, const T scale, const ZQ_GridDeformationOptions::NeighborType type)
	{

		int dir_x[12] = {1,0,-1,0, 1,-1,-1,1, 2,0,-2,0};
		int dir_y[12] = {0,1,0,-1, 1,1,-1,-1, 0,2,0,-2};
		int neighbor_loop_num = 4;
		switch(type)
		{
		case ZQ_GridDeformationOptions::NEIGHBOR_4:
			neighbor_loop_num = 4;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_8:
			neighbor_loop_num = 8;
			break;
		case ZQ_GridDeformationOptions::NEIGHBOR_12:
			neighbor_loop_num = 12;
			break;
		default:
			return false;
			break;
		}

		T* init_unknown = new T[unknown_num];
		T* fixed_val = new T[fixed_num];
		for(int pp = 0;pp < width*height;pp++)
		{
			if(!nouseful_flag[pp])
			{
				if(fixed_flag[pp])
				{
					fixed_val[x_index[pp]] = init_coord[pp*2];
					fixed_val[y_index[pp]] = init_coord[pp*2+1];
				}
				else
				{
					init_unknown[x_index[pp]] = init_coord[pp*2];
					init_unknown[y_index[pp]] = init_coord[pp*2+1];
				}
			}

		}

		T* f = new T[unknown_num];
		ZQ_TaucsBase::ZQ_taucs_ccs_matrix_time_vec(B,fixed_val,f);

		for(int i = 0;i < height;i++)
		{
			for(int j = 0;j < width;j++)
			{
				int offset = i*width+j;
				if(nouseful_flag[offset])
					continue;

				for(int dd = 0; dd < neighbor_loop_num; dd++)
				{
					int i_2 = i+dir_y[dd];
					int j_2 = j+dir_x[dd];
					j_2 = (j_2+width)%width;
					int offset_2 = i_2*width+j_2;
					if(0 <= i_2 && i_2 < height && !nouseful_flag[offset_2])
					{
						float old_v[2] = {dir_x[dd]*scale,dir_y[dd]*scale};
						float cur_v[2] = {
							Rmats[offset].val[0] * old_v[0] + Rmats[offset].val[1] * old_v[1],
							Rmats[offset].val[2] * old_v[0] + Rmats[offset].val[3] * old_v[1]
						};

						if(!fixed_flag[offset])
						{
							f[x_index[offset]] += 2*cur_v[0];
							f[y_index[offset]] += 2*cur_v[1];
						}
						if(!fixed_flag[offset_2])
						{
							f[x_index[offset_2]] -= 2*cur_v[0];
							f[y_index[offset_2]] -= 2*cur_v[1];
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
		ZQ_PCGSolver::PCG(G,f,init_unknown,iteration,tol,unknown_vals,it);

		for(int i = 0;i < width*height;i++)
		{
			if(nouseful_flag[i])
				continue;
			if(fixed_flag[i])
			{
				out_coord[i*2] = init_coord[i*2];
				out_coord[i*2+1] = init_coord[i*2+1];
			}
			else
			{
				out_coord[i*2] = unknown_vals[x_index[i]];
				out_coord[i*2+1] = unknown_vals[y_index[i]];
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