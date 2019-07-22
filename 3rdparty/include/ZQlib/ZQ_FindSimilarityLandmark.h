#ifndef _ZQ_FIND_SIMILARITY_LANDMARK_H_
#define _ZQ_FIND_SIMILARITY_LANDMARK_H_
#pragma once

#include "ZQ_Matrix.h"
#include "ZQ_SVD.h"
#include <stdio.h>
#include <time.h>

namespace ZQ
{
	class ZQ_FindSimilarityLandmark
	{
	public:
		template<class BaseType>
		static bool FindSimilarityLandmark(int nPts, const BaseType* found_coord, const BaseType* standard_coord, BaseType* transform)
		{
			ZQ_Matrix<double> mat(2, 3);
			if (!_findSimilarity(nPts, found_coord, standard_coord, mat))
				return false;
			const double* mat_ptr = mat.GetDataPtr();
			for (int i = 0; i < 6; i++)
			{
				transform[i] = mat_ptr[i];
			}
			return true;
		}

	private:
		template<class BaseType>
		static bool _findNonreflectiveSimilarity(int nPts, const BaseType* uv, const BaseType* xy, ZQ_Matrix<double>& transform)
		{
			/*
			%
			% For a nonreflective similarity :
			%
			% let sc = s*cos(theta)
			% let ss = s*sin(theta)
			%
			%				  [sc -ss
			%[u v] = [x y 1] * ss  sc
			%                  tx  ty]
			%
			% There are 4 unknowns: sc, ss, tx, ty.
			%
			% Another way to write this is :
			%
			% u = [x y 1 0] * [sc
			%                  ss
			%                  tx
			%                  ty]
			%
			% v = [y -x 0 1] * [sc
			%                   ss
			%                   tx
			%                   ty]
			%
			% With 2 or more correspondence points we can combine the u equations and
			% the v equations for one linear system to solve for sc, ss, tx, ty.
			%
			%[u1] = [x1  y1  1  0] * [sc]
			%[u2]   [x2  y2  1  0]   [ss]
			%[...]  [...]            [tx]
			%[un]   [xn  yn  1  0]   [ty]
			%[v1]   [y1 -x1  0  1]
			%[v2]   [y2 -x2  0  1]
			%[...]  [...]
			%[vn]   [yn - xn  0  1]
			%
			% Or rewriting the above matrix equation :
			% U = X * r, where r = [sc ss tx ty]'
			% so r = X\U.
			%
			x = xy(:, 1);
			y = xy(:, 2);
			X = [x   y  ones(M, 1)   zeros(M, 1);
			y  -x  zeros(M, 1)  ones(M, 1)];
			u = uv(:, 1);
			v = uv(:, 2);
			U = [u; v];
			% We know that X * r = U
			if rank(X) >= 2 * K
			r = X \ U;
			else
			error(message('images:cp2tform:twoUniquePointsReq'))
			end
			sc = r(1);
			ss = r(2);
			tx = r(3);
			ty = r(4);
			Tinv = [sc -ss 0;
			ss  sc 0;
			tx  ty 1];
			T = inv(Tinv);
			T(:, 3) = [0 0 1]';
			trans = maketform('affine', T);
			*/

			
			ZQ_Matrix<double> X(nPts * 2, 4);
			ZQ_Matrix<double> U(nPts * 2, 1);
			for (int i = 0; i < nPts; i++)
			{
				X.SetData(i, 0, xy[i * 2 + 0]);
				X.SetData(i, 1, xy[i * 2 + 1]);
				X.SetData(i, 2, 1);
				X.SetData(i, 3, 0);
				X.SetData(i + nPts, 0, xy[i * 2 + 1]);
				X.SetData(i + nPts, 1, -xy[i * 2 + 0]);
				X.SetData(i + nPts, 2, 0);
				X.SetData(i + nPts, 3, 1);
				U.SetData(i, 0, uv[i * 2 + 0]);
				U.SetData(i + nPts, 0, uv[i * 2 + 1]);
			}
			ZQ_Matrix<double> r(4, 1);
			//double t1 = omp_get_wtime();
			if(!ZQ_SVD::Solve(X,r,U))
			{
				printf("failed to solve\n");
				return false;
			}
			//double t2 = omp_get_wtime();
			//printf("solve:%.3f\n", t2 - t1);
			bool flag;
			double sc = r.GetData(0, 0, flag);
			double ss = r.GetData(1, 0, flag);
			double tx = r.GetData(2, 0, flag);
			double ty = r.GetData(3, 0, flag);

			double Tinv[9] =
			{
				sc, -ss, 0,
				ss, sc, 0,
				tx, ty, 1
			};

			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
					transform.SetData(i, j, Tinv[i * 3 + j]);
			}
			return true;
		}

		template<class BaseType>
		static bool _findSimilarity(int nPts, const BaseType* uv, const BaseType* xy, ZQ_Matrix<double>& transform)
		{
			/*
			function [trans, output] = findSimilarity(uv,xy,options)
			%
			% The similarities are a superset of the nonreflective similarities as they may
			% also include reflection.
			%
			% let sc = s*cos(theta)
			% let ss = s*sin(theta)
			%
			%                   [ sc -ss
			% [u v] = [x y 1] *   ss  sc
			%                     tx  ty]
			%
			%          OR
			%
			%                   [ sc  ss
			% [u v] = [x y 1] *   ss -sc
			%                     tx  ty]
			%
			% Algorithm:
			% 1) Solve for trans1, a nonreflective similarity.
			% 2) Reflect the xy data across the Y-axis,
			%    and solve for trans2r, also a nonreflective similarity.
			% 3) Transform trans2r to trans2, undoing the reflection done in step 2.
			% 4) Use TFORMFWD to transform uv using both trans1 and trans2,
			%    and compare the results, returning the transformation corresponding
			%    to the smaller L2 norm.
			% Need to reset options.K to prepare for calls to findNonreflectiveSimilarity.
			% This is safe because we already checked that there are enough point pairs.
			options.K = 2;
			% Solve for trans1
			[trans1, output] = findNonreflectiveSimilarity(uv,xy,options);
			% Solve for trans2
			% manually reflect the xy data across the Y-axis
			xyR = xy;
			xyR(:,1) = -1*xyR(:,1);
			trans2r  = findNonreflectiveSimilarity(uv,xyR,options);
			% manually reflect the tform to undo the reflection done on xyR
			TreflectY = [-1  0  0;
			0  1  0;
			0  0  1];
			trans2 = maketform('affine', trans2r.tdata.T * TreflectY);
			% Figure out if trans1 or trans2 is better
			xy1 = tformfwd(trans1,uv);
			norm1 = norm(xy1-xy);
			xy2 = tformfwd(trans2,uv);
			norm2 = norm(xy2-xy);
			if norm1 <= norm2
			trans = trans1;
			else
			trans = trans2;
			end
			*/

			
			ZQ_Matrix<double> transform1(3,3), transform2R(3,3), transform2(3,3);
			//clock_t t1 = clock();
			_findNonreflectiveSimilarity(nPts, uv, xy, transform1);
			//clock_t t2 = clock();
			
			BaseType* xyR = new BaseType[nPts * 2];
			for (int i = 0; i < nPts; i++)
			{
				xyR[i * 2 + 0] = -xy[i * 2 + 0];
				xyR[i * 2 + 1] = xy[i * 2 + 1];
			}
			clock_t t3 = clock();
			_findNonreflectiveSimilarity(nPts, uv, xyR, transform2R);
			clock_t t4 = clock();
			
			delete[]xyR;

			const double TreflectY[9] =
			{
				-1, 0,  0,
				0,  1,  0,
				0,  0,  1
			};
			ZQ_Matrix<double> TreflectY_mat(3, 3);
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
					TreflectY_mat.SetData(i, j, TreflectY[i * 3 + j]);
			}
			transform2 = transform2R*TreflectY_mat;

			const double* transform1_ptr = transform1.GetDataPtr();
			const double* transform2_ptr = transform2.GetDataPtr();
			//forward transform
			double norm1 = 0, norm2 = 0;
			for (int p = 0; p < nPts; p++)
			{
				double uv1_x = transform1_ptr[0] * xy[p * 2 + 0] + transform1_ptr[3] * xy[p * 2 + 1] + transform1_ptr[6];
				double uv1_y = transform1_ptr[1] * xy[p * 2 + 0] + transform1_ptr[4] * xy[p * 2 + 1] + transform1_ptr[7];
				double uv2_x = transform2_ptr[0] * xy[p * 2 + 0] + transform2_ptr[3] * xy[p * 2 + 1] + transform2_ptr[6];
				double uv2_y = transform2_ptr[1] * xy[p * 2 + 0] + transform2_ptr[4] * xy[p * 2 + 1] + transform2_ptr[7];

				norm1 += (uv[p * 2 + 0] - uv1_x)*(uv[p * 2 + 0] - uv1_x) + (uv[p * 2 + 1] - uv1_y)*(uv[p * 2 + 1] - uv1_y);
				norm2 += (uv[p * 2 + 0] - uv2_x)*(uv[p * 2 + 0] - uv2_x) + (uv[p * 2 + 1] - uv2_y)*(uv[p * 2 + 1] - uv2_y);
			}

			clock_t t5 = clock();
			ZQ_Matrix<double> tmp(3,3);
			if (norm1 < norm2)
			{
				if (!ZQ_SVD::Invert(transform1, tmp))
					return false;
			}	
			else
			{
				if (!ZQ_SVD::Invert(transform2, tmp))
					return false;
			}
			clock_t t6 = clock();

			const double* tmp_ptr = tmp.GetDataPtr();
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					transform.SetData(i, j, tmp_ptr[j * 3 + i]);
				}
			}
			//printf("%f,%f,%f\n", 0.001*(t2 - t1), 0.001*(t4 - t3), 0.001*(t6 - t5));

			return true;
		}
	};
}

#endif
