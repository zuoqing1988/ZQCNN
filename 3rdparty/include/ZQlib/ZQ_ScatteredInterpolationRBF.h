#ifndef _ZQ_SCATTERED_INTERPOLATION_RBF_H_
#define _ZQ_SCATTERED_INTERPOLATION_RBF_H_

#pragma once

#include "ZQ_PCGSolver.h"
#include "ZQ_SparseMatrix.h"
#include "ZQ_RBFKernel.h"
#include "ZQ_KDTree.h"
#include <typeinfo>


namespace ZQ
{
	template<class T>
	class ZQ_ScatteredInterpolationRBF
	{
	public:
		ZQ_ScatteredInterpolationRBF();
		~ZQ_ScatteredInterpolationRBF();

	private:
		int npoints;
		T* points;
		int dim;
		T* values;
		int nChannels;
		T* radius;
		T* coefficients;
		T* linear_coeff;

		ZQ_RBFKernel::RBF_TYPE type;

	public:
		/*This function will copy the input data*/
		bool SetLandmarks(int n, int dim, T* pts, T* vals, int nChannels = 1);

		bool SolveCoefficient(int number_of_neigbor, double scale, int max_iter, ZQ_RBFKernel::RBF_TYPE type = ZQ_RBFKernel::GLOBAL_GAUSS, bool display = false);

		/*this must be called after Solve Coefficient 
		and the memory of the arguments are allocated in this function*/
		void CopyDataOut(int& nPoints, int& dim, T*& ptsCoords, T*& radius, int& nChannels, T*& coefficients, T*& linear_coeff);

		bool Interpolate(int num, const T* pts, T* vals);

		bool GridData2D(int xDim, int yDim, double xmin, double xmax, double ymin, double ymax, T* data);

		bool GridData3D(int xDim, int yDim, int zDim, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, T* data);

	private:
		void _clear();

		void _selectRadius(const ZQ_KDTree<T>* tree, int number_of_neighbor, double scale);

		double _kernel(double d, double radius);

		double _length(int dim, const T* v1, const T* v2);
		double _length2(int dim, const T* v1, const T* v2);
	};

	/**************************  definitions  **********************************/


	template<class T>
	ZQ_ScatteredInterpolationRBF<T>::ZQ_ScatteredInterpolationRBF()
	{
		npoints = 0;
		points = 0;
		values = 0;
		dim = 0;
		radius = 0;
		nChannels = 1;
		coefficients = 0;
		linear_coeff = 0;
		type = ZQ_RBFKernel::GLOBAL_GAUSS;
	}


	template<class T>
	ZQ_ScatteredInterpolationRBF<T>::~ZQ_ScatteredInterpolationRBF()
	{
		_clear();
	}


	template<class T>
	bool ZQ_ScatteredInterpolationRBF<T>::SetLandmarks(int n, int dim, T* pts, T* vals, int nChannels)
	{
		if(n <= 0 || dim <= 0 || pts == 0 || vals == 0 || nChannels < 1)
		{
			return false;
		}
		_clear();
		npoints = n;
		this->dim = dim;
		this->nChannels = nChannels;
		points = new T[n*dim];
		memcpy(points,pts,sizeof(T)*n*dim);
		values = new T[n*nChannels];
		for(int c = 0;c < nChannels;c++)
		{
			for(int i = 0;i < n;i++)
			{
				values[c*n+i] = vals[i*nChannels+c];
			}
		}

		return true;
	}


	template<class T>
	bool ZQ_ScatteredInterpolationRBF<T>::SolveCoefficient(int number_of_neighbor, double scale, int max_iter, ZQ_RBFKernel::RBF_TYPE type, bool display)
	{
		if(npoints < 1)
			return false;

		int taucs_flag;
		if(strcmp(typeid(T).name(),"float") == 0)
			taucs_flag = TAUCS_SINGLE;
		else if(strcmp(typeid(T).name(),"double") == 0)
			taucs_flag = TAUCS_DOUBLE;
		else
			return false;

		this->type = type;
		if(radius)
			delete []radius;
		if(coefficients)
			delete []coefficients;
		if(linear_coeff)
			delete []linear_coeff;
		radius = new T[npoints];
		coefficients = new T[npoints*nChannels];
		linear_coeff = new T[(dim+1)*nChannels];


		ZQ_KDTree<T>* tree = new ZQ_KDTree<T>();
		tree->BuildKDTree(points,npoints,dim,10);

		if (display)
			printf("build KDTree done!\n");

		_selectRadius(tree,number_of_neighbor,scale);

		if (display)
			printf("select radius done!\n");


		ZQ_SparseMatrix<T>* mat = new ZQ_SparseMatrix<T>(npoints+dim+1,npoints+dim+1);

		T* sum_weight = new T[npoints];
		memset(sum_weight,0,sizeof(T)*npoints);
		for(int i = 0;i < npoints;i++)
		{
			int result_num = 0;
			bool flag = tree->AnnFixRadiusSearchCountReturnNum(points+i*dim,radius[i],result_num);
			int* idx = new int[result_num];
			T* dist = new T[result_num];
			flag = tree->AnnFixRadiusSearch(points+i*dim,radius[i],result_num,idx,dist);

			for(int j = 0;j < result_num;j++)
			{
				double weight = _kernel(_length(dim,points+i*dim,points+idx[j]*dim),radius[i]);
				sum_weight[idx[j]] += weight;

				if(weight > 0)
				{
					mat->SetValue(idx[j],i,weight);
				}
			}
			delete []idx;
			delete []dist;
		}


		delete []sum_weight;

		for(int i = 0;i < npoints;i++)
		{
			for(int d = 0;d < dim;d++)
			{
				mat->SetValue(i,npoints+d,points[i*dim+d]);
				mat->SetValue(npoints+d,i,points[i*dim+d]);
			}
			mat->SetValue(i,npoints+dim,1);
			mat->SetValue(npoints+dim,i,1);
		}

		if (display)
			printf("build matrix done!\n");

		
		taucs_ccs_matrix* A = mat->ExportCCS(taucs_flag);
		delete mat; mat = 0;


		T* x0 = new T[npoints+dim+1];
		memset(x0,0,sizeof(T)*(npoints+dim+1));
		int it = 0;
		double tol = 1e-12;

		for(int c = 0;c < nChannels;c++)
		{
			T* x = new T[npoints+dim+1];
			T* b = new T[npoints+dim+1];
			memset(b,0,sizeof(T)*(npoints+dim+1));
			memcpy(b,values+c*npoints,sizeof(T)*npoints);
			ZQ_PCGSolver::PCG_sparse_unsquare(A,b,x0,max_iter,tol,x,it,display);
			memcpy(coefficients+c*npoints,x,sizeof(T)*npoints);
			memcpy(linear_coeff+c*(dim+1),x+npoints,sizeof(T)*(dim+1));
			delete []x;
			delete []b;
		}
		delete []x0;
		delete tree;
		ZQ_TaucsBase::ZQ_taucs_ccs_free(A);
		return true;
	}


	template<class T>
	void ZQ_ScatteredInterpolationRBF<T>::CopyDataOut(int& nPoints, int& dim, T*& ptsCoords, T*& radius, int& nChannels, T*& coefficients, T*& linear_coeff)
	{
		nPoints = this->npoints;
		dim = this->dim;
		nChannels = this->nChannels;
		ptsCoords = new T[nPoints*dim];
		radius = new T[nPoints];
		memcpy(ptsCoords,this->points,sizeof(T)*nPoints*dim);
		memcpy(radius,this->radius,sizeof(T)*nPoints);
		
		coefficients = new T[nPoints*nChannels];
		for(int pp = 0;pp < nPoints;pp++)
		{
			for(int c = 0;c < nChannels;c++)
			{
				coefficients[pp*nChannels+c] = this->coefficients[nPoints*c+pp];
			}
		}
		linear_coeff = new T[(dim+1)*nChannels];
		for(int dd = 0;dd < dim+1;dd++)
		{
			for(int c = 0;c < nChannels;c++)
			{
				linear_coeff[dd*nChannels+c] = this->linear_coeff[(dim+1)*c+dd];
			}
		}
	}


	template<class T>
	bool ZQ_ScatteredInterpolationRBF<T>::Interpolate(int num, const T* pts, T* vals)
	{
		if(pts == 0 || vals == 0)
			return false;

		if(npoints == 0)
			return false;

		if(coefficients == 0 || radius == 0)
			return false;

		for(int i = 0;i < num;i++)
		{
			for(int c = 0;c < nChannels;c++)
			{
				double result = 0;
				double sum_weight = 0;

				for(int j = 0;j < npoints;j++)
				{
					double cur_weight = _kernel(_length(dim,pts+i*dim,points+j*dim),radius[j]);
					result += cur_weight * coefficients[j+c*npoints];
				}

				for(int d = 0;d < dim;d++)
					result += pts[i*dim+d]*linear_coeff[d+c*(dim+1)];
				result += linear_coeff[dim+c*(dim+1)];
				vals[i*nChannels+c] = result;
			}
		}
		return true;
	}


	template<class T>
	bool ZQ_ScatteredInterpolationRBF<T>::GridData2D(int xDim, int yDim, double xmin, double xmax, double ymin, double ymax, T* data)
	{
		if(dim != 2)
			return false;

		if(data == 0)
			return false;

		if(xDim < 2 || yDim < 2)
			return false;

		if(xmin >= xmax || ymin >= ymax)
			return false;

		if(npoints == 0 || coefficients == 0 || radius == 0)
			return false;

		memset(data,0,sizeof(T)*xDim*yDim*nChannels);

		double x_per_len = (xmax-xmin)/(xDim-1);
		double y_per_len = (ymax-ymin)/(yDim-1);

		T cur_pt[2];

		for(int i = 0;i < npoints;i++)
		{
			double cx = points[i*dim+0];
			double cy = points[i*dim+1];
			double x_start = (cx-radius[i]-xmin)/x_per_len;
			double x_end = (cx+radius[i]-xmin)/x_per_len;
			double y_start = (cy-radius[i]-ymin)/y_per_len;
			double y_end = (cy+radius[i]-ymin)/y_per_len;


			for(int h = __max(0,y_start-1);h <= __min(y_end,yDim-1);h++)
			{
				for(int w = __max(0,x_start-1);w <= __min(x_end,xDim-1);w++)
				{
					cur_pt[0] = xmin+w*x_per_len;
					cur_pt[1] = ymin+h*y_per_len;
					double weight = _kernel(_length(dim,cur_pt,points+i*dim),radius[i]);

					for(int c = 0;c < nChannels;c++)
						data[(h*xDim+w)*nChannels+c] += weight*coefficients[i+c*npoints];
				}
			}
		}

		for(int h = 0;h < yDim;h++)
		{
			for(int w = 0;w < xDim;w++)
			{
				cur_pt[0] = xmin+w*x_per_len;
				cur_pt[1] = ymin+h*y_per_len;
				for(int c = 0;c < nChannels;c++)
				{
					data[(h*xDim+w)*nChannels+c] += linear_coeff[0+c*3]*cur_pt[0]+linear_coeff[1+c*3]*cur_pt[1]+linear_coeff[2+c*3];
				}
			}
		}	
		return true;
	}


	template<class T>
	bool ZQ_ScatteredInterpolationRBF<T>::GridData3D(int xDim, int yDim, int zDim, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, T* data)
	{
		if(dim != 3)
			return false;

		if(data == 0)
			return false;

		if(xDim < 2 || yDim < 2 || zDim < 2)
			return false;

		if(xmin >= xmax || ymin >= ymax || zmin >= zmax)
			return false;

		if(npoints == 0 || coefficients == 0 || radius == 0)
			return false;

		memset(data,0,sizeof(T)*xDim*yDim*zDim*nChannels);

		double x_per_len = (xmax-xmin)/(xDim-1);
		double y_per_len = (ymax-ymin)/(yDim-1);
		double z_per_len = (zmax-zmin)/(zDim-1);

		T cur_pt[3];

		for(int i = 0;i < npoints;i++)
		{
			double cx = points[i*dim+0];
			double cy = points[i*dim+1];
			double cz = points[i*dim+2];
			double x_start = (cx-radius[i]-xmin)/x_per_len;
			double x_end = (cx+radius[i]-xmin)/x_per_len;
			double y_start = (cy-radius[i]-ymin)/y_per_len;
			double y_end = (cy+radius[i]-ymin)/y_per_len;
			double z_start = (cz-radius[i]-zmin)/z_per_len;
			double z_end = (cz+radius[i]-zmin)/z_per_len;


			for(int d = __max(0,z_start-1); d <= __min(z_end,zDim-1); d++)
			{
				for(int h = __max(0,y_start-1);h <= __min(y_end,yDim-1);h++)
				{
					for(int w = __max(0,x_start-1);w <= __min(x_end,xDim-1);w++)
					{
						cur_pt[0] = xmin+w*x_per_len;
						cur_pt[1] = ymin+h*y_per_len;
						cur_pt[2] = zmin+d*z_per_len;
						double weight = _kernel(_length(dim,cur_pt,points+i*dim),radius[i]);
						int offset = d*yDim*xDim+h*xDim+w;
						for(int c = 0;c < nChannels;c++)
							data[offset*nChannels+c] += weight*coefficients[i+c*npoints];
					}
				}
			}
		}


		for(int d = 0;d < zDim;d++)
		{
			for(int h = 0;h < yDim;h++)
			{
				for(int w = 0;w < xDim;w++)
				{
					cur_pt[0] = xmin+w*x_per_len;
					cur_pt[1] = ymin+h*y_per_len;
					cur_pt[2] = zmin+d*z_per_len;
					for(int c = 0;c < nChannels;c++)
					{
						data[(d*yDim*xDim+h*xDim+w)*nChannels+c] += linear_coeff[0+c*4]*cur_pt[0]+linear_coeff[1+c*4]*cur_pt[1]+linear_coeff[2+c*4]*cur_pt[2]+linear_coeff[3+c*4];
					}
				}
			}
		}

		return true;
	}


	template<class T>
	void ZQ_ScatteredInterpolationRBF<T>::_clear()
	{
		if(points)
		{
			delete []points;
			points = 0;
		}
		if(values)
		{
			delete []values;
			values = 0;
		}
		if(radius)
		{
			delete []radius;
			radius = 0;
		}
		if(coefficients)
		{
			delete []coefficients;
			coefficients = 0;
		}
		if(linear_coeff)
		{
			delete []linear_coeff;
			linear_coeff = 0;
		}
		dim = 0;
		npoints = 0;
	}

	template<class T>
	void ZQ_ScatteredInterpolationRBF<T>::_selectRadius(const ZQ_KDTree<T>* tree, int number_of_neighbor, double scale)
	{
		int k = number_of_neighbor >= npoints ? npoints-1 : number_of_neighbor;
		int* idx = new int[k];
		T* dist = new T[k];
		memset(idx,0,sizeof(int)*k);
		memset(dist,0,sizeof(T)*k);

		for(int i = 0;i < npoints;i++)
		{
			tree->AnnSearch(points+i*dim,k,idx,dist,0);
			double max_dist = dist[0];
			for(int j = 1;j < k;j++)
			{
				if(max_dist < dist[j])
					max_dist = dist[j];
			}
			radius[i] = scale*sqrt(max_dist);
		}
		delete []idx;
		delete []dist;
	}


	template<class T>
	double ZQ::ZQ_ScatteredInterpolationRBF<T>::_kernel(double d, double radius)
	{
		bool flag = false;
		switch(type)
		{
		case ZQ_RBFKernel::COMPACT_CPC0:
		case ZQ_RBFKernel::COMPACT_CPC2:
		case ZQ_RBFKernel::COMPACT_CPC4:
		case ZQ_RBFKernel::COMPACT_CPC6:
		case ZQ_RBFKernel::COMPACT_CTPS_C0:
		case ZQ_RBFKernel::COMPACT_CTPS_C1:
		case ZQ_RBFKernel::COMPACT_CTPS_C2A:
		case ZQ_RBFKernel::COMPACT_CTPS_C2B:
			return ZQ_RBFKernel::_compact_kernel(flag,type,d,radius);
			break;
		case ZQ_RBFKernel::GLOBAL_GAUSS:
		default:
			{
				if(fabs(d) >= fabs(radius))
					return 0;
				return ZQ_RBFKernel::_global_kernel(flag,ZQ_RBFKernel::GLOBAL_GAUSS,d*3,radius);
			}	
		}
	}


	template<class T>
	double ZQ::ZQ_ScatteredInterpolationRBF<T>::_length(int dim, const T *v1, const T *v2)
	{
		double len2 = 0;
		for(int i = 0;i < dim;i++)
			len2 += (v1[i]-v2[i])*(v1[i]-v2[i]);

		return sqrt(len2);
	}

	template<class T>
	double ZQ::ZQ_ScatteredInterpolationRBF<T>::_length2(int dim, const T* v1, const T* v2)
	{
		double len2 = 0;
		for(int i = 0;i < dim;i++)
			len2 += (v1[i]-v2[i])*(v1[i]-v2[i]);

		return len2;
	}

}


#endif