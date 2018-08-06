#ifndef _ZQ_WARPING_H_
#define _ZQ_WARPING_H_
#pragma once

#include "ZQ_ScatteredInterpolationRBF.h"
#include "ZQ_PCGSolver.h"
#include "ZQ_SparseMatrix.h"

namespace ZQ
{
	template<class T, const int NDim>
	class ZQ_Warping
	{
	public:
		ZQ_Warping(){}
		~ZQ_Warping(){}

	private:
		ZQ_ScatteredInterpolationRBF<T> rbf_solver;

	public:
		bool Solve(int npts, T* before_warping_pts, T* after_warping_pts, int max_iter, int neighbor_num = 4, double scale = 3)
		{
			if(!rbf_solver.SetLandmarks(npts,NDim,before_warping_pts,after_warping_pts,NDim))
				return false;
			if(!rbf_solver.SolveCoefficient(neighbor_num,scale,max_iter,ZQ_RBFKernel::COMPACT_CPC2,false))
				return false;
			return true;
		}

		bool WarpCoord(int npts, T* input_pts, T* output_pts)
		{
			return rbf_solver.Interpolate(npts,input_pts,output_pts);
		}
	};
}
#endif