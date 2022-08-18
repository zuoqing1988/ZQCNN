#ifndef _ZQ_CNN_FACE_CROP_UTILS_H_
#define _ZQ_CNN_FACE_CROP_UTILS_H_
#pragma once

#include "ZQ_CNN_Tensor4D.h"
#include "ZQlib/ZQ_SVD.h"
#include <omp.h>

namespace ZQ
{
	class ZQ_CNN_FaceCropUtils
	{
	public:

		static bool CropImage_112x112(const ZQ_CNN_Tensor4D& img, const float* facial5point, ZQ_CNN_Tensor4D& crop, float fill_val = 0.0f)
		{
			float coord5point[10] =
			{
				30.2946 + 8, 51.6963,
				65.5318 + 8, 51.5014,
				48.0252 + 8, 71.7366,
				33.5493 + 8, 92.3655,
				62.7299 + 8, 92.2041
			};

			float transform[6];
			//clock_t t1 = clock();
			if (!_findSimilarity(5, facial5point, coord5point, transform))
				return false;
			ZQ_Matrix<double> trans(3, 3), tmp(3, 3);
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 3; j++)
					tmp.SetData(i, j, transform[i * 3 + j]);
			}
			tmp.SetData(2, 0, 0);
			tmp.SetData(2, 1, 0);
			tmp.SetData(2, 2, 1);
			if (!ZQ_SVD::Invert(tmp, trans))
				return false;
			double* ptr = trans.GetDataPtr();
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 3; j++)
					transform[i * 3 + j] = ptr[i * 3 + j];
			}
			//clock_t t2 = clock();
			int dst_W = 112, dst_H = 112;
			std::vector<float> map_x(dst_W*dst_H), map_y(dst_W*dst_H);
			for (int h = 0; h < dst_H; h++)
			{
				for (int w = 0; w < dst_W; w++)
				{
					float x = w*transform[0] + h*transform[1] + transform[2];
					float y = w*transform[3] + h*transform[4] + transform[5];
					map_x[h*dst_W + w] = x;
					map_y[h*dst_W + w] = y;
					//printf("%d,%d = %.1f %.1f\n", h,w,x, y);
				}
			}
			if (!img.Remap(crop, dst_W, dst_H, 1, 1, map_x, map_y, true, fill_val))
				return false;
			//clock_t t3 = clock();
			//printf("findtrans:%.3f, warp:%.3f\n", 0.001*(t2 - t1), 0.001*(t3 - t2));
			return true;
		}

		/*
		tranform = [ sc, ss, tx;
		-ss, sc, ty];
		tx,ty is translation£¬
		angle = atan(ss,sc),
		*/
		static bool CropImage_112x112_translate_scale_roll(const ZQ_CNN_Tensor4D& img,
			const float* facial5point, ZQ_CNN_Tensor4D& crop, float transform[6], float fill_val = 0.0f)
		{
			float coord5point[10] =
			{
				30.2946 + 8, 51.6963,
				65.5318 + 8, 51.5014,
				48.0252 + 8, 71.7366,
				33.5493 + 8, 92.3655,
				62.7299 + 8, 92.2041
			};

			//clock_t t1 = clock();
			float tmp_trans[9];
			if (!_findNonreflectiveSimilarity(5, facial5point, coord5point, tmp_trans))
				return false;
			//clock_t t2 = clock();
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					transform[i * 3 + j] = tmp_trans[j * 3 + i];
				}
			}
			int dst_W = 112, dst_H = 112;
			std::vector<float> map_x(dst_W*dst_H), map_y(dst_W*dst_H);
			for (int h = 0; h < dst_H; h++)
			{
				for (int w = 0; w < dst_W; w++)
				{
					float x = w*transform[0] + h*transform[1] + transform[2];
					float y = w*transform[3] + h*transform[4] + transform[5];
					map_x[h*dst_W + w] = x;
					map_y[h*dst_W + w] = y;
				}
			}
			if (!img.Remap(crop, dst_W, dst_H, 1, 1, map_x, map_y, true, 0))
				return false;
			//clock_t t3 = clock();
			//printf("findtrans:%.3f, warp:%.3f\n", 0.001*(t2 - t1), 0.001*(t3 - t2));
			return true;
		}

	private:
		static bool _findNonreflectiveSimilarity(int nPts, const float* uv, const float* xy, float transform[9])
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
			double t1 = omp_get_wtime();
			if (!ZQ_SVD::Solve(X, r, U))
			{
				printf("failed to solve\n");
				return false;
			}
			double t2 = omp_get_wtime();
			//printf("solve:%.3f\n", t2 - t1);
			double* ptr = r.GetDataPtr();
			float sc = ptr[0];
			float ss = ptr[1];
			float tx = ptr[2];
			float ty = ptr[3];

			float Tinv[9] =
			{
				sc, -ss, 0,
				ss, sc, 0,
				tx, ty, 1
			};

			/*for (int i = 0; i < 3; i++)
			{
			for (int j = 0; j < 3; j++)
			{
			printf("%12.5f", Tinv[i * 3 + j]);
			}
			printf("\n");
			}*/
			memcpy(transform, Tinv, sizeof(float) * 9);
			return true;
		}

		static bool _findSimilarity(int nPts, const float* uv, const float* xy, float transform[6])
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

			float transform1[9], transform2R[9], transform2[9];
			if (!_findNonreflectiveSimilarity(nPts, uv, xy, transform1))
				return false;
			float* xyR = new float[nPts * 2];
			for (int i = 0; i < nPts; i++)
			{
				xyR[i * 2 + 0] = -xy[i * 2 + 0];
				xyR[i * 2 + 1] = xy[i * 2 + 1];
			}
			if (!_findNonreflectiveSimilarity(nPts, uv, xyR, transform2R))
				return false;

			const float TreflectY[9] =
			{
				-1, 0,  0,
				0,  1,  0,
				0,  0,  1
			};
			ZQ_MathBase::MatrixMul(transform2R, TreflectY, 3, 3, 3, transform2);

			//forward transform
			float norm1 = 0, norm2 = 0;
			for (int p = 0; p < nPts; p++)
			{
				float uv1_x = transform1[0] * xy[p * 2 + 0] + transform1[3] * xy[p * 2 + 1] + transform1[6];
				float uv1_y = transform1[1] * xy[p * 2 + 0] + transform1[4] * xy[p * 2 + 1] + transform1[7];
				float uv2_x = transform2[0] * xy[p * 2 + 0] + transform2[3] * xy[p * 2 + 1] + transform2[6];
				float uv2_y = transform2[1] * xy[p * 2 + 0] + transform2[4] * xy[p * 2 + 1] + transform2[7];

				norm1 += (uv[p * 2 + 0] - uv1_x)*(uv[p * 2 + 0] - uv1_x) + (uv[p * 2 + 1] - uv1_y)*(uv[p * 2 + 1] - uv1_y);
				norm2 += (uv[p * 2 + 0] - uv2_x)*(uv[p * 2 + 0] - uv2_x) + (uv[p * 2 + 1] - uv2_y)*(uv[p * 2 + 1] - uv2_y);
			}

			ZQ_Matrix<double> tmp(3, 3);
			if (norm1 < norm2)
			{
				ZQ_Matrix<double> trans1_mat(3, 3);
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						trans1_mat.SetData(i, j, transform1[i * 3 + j]);
					}
				}
				if (!ZQ_SVD::Invert(trans1_mat, tmp))
					return false;
			}
			else
			{
				ZQ_Matrix<double> trans2_mat(3, 3);
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						trans2_mat.SetData(i, j, transform2[i * 3 + j]);
					}
				}
				if (!ZQ_SVD::Invert(trans2_mat, tmp))
					return false;
			}

			double* ptr = tmp.GetDataPtr();
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					transform[i * 3 + j] = ptr[j * 3 + i];
				}
			}

			return true;
		}
	};
}
#endif
