#ifndef _ZQ_FACE_RECOGNIZER_UTILS_H_
#define _ZQ_FACE_RECOGNIZER_UTILS_H_
#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>
#include <omp.h>

namespace ZQ
{
	class ZQ_FaceRecognizerUtils
	{
	public:

		template<class BaseType>
		static bool CropImage_112x96(const cv::Mat& img, const BaseType* facial5point, cv::Mat& crop)
		{
			cv::Size designed_size(96, 112);
			BaseType coord5point[10] =
			{
				30.2946, 51.6963,
				65.5318, 51.5014,
				48.0252, 71.7366,
				33.5493, 92.3655,
				62.7299, 92.2041
			};

			cv::Mat transform;
			clock_t t1 = clock();
			_findSimilarity(5, facial5point, coord5point, transform);
			clock_t t2 = clock();
			cv::warpAffine(img, crop, transform, designed_size);
			clock_t t3 = clock();
			//printf("findtrans:%.3f, warp:%.3f\n", 0.001*(t2 - t1), 0.001*(t3 - t2));
			return true;
		}

		template<class BaseType>
		static bool CropImage_112x112(const cv::Mat& img, const BaseType* facial5point, cv::Mat& crop)
		{
			cv::Size designed_size(112, 112);
			BaseType coord5point[10] =
			{
				30.2946+8, 51.6963,
				65.5318+8, 51.5014,
				48.0252+8, 71.7366,
				33.5493+8, 92.3655,
				62.7299+8, 92.2041
			};

			cv::Mat transform;
			clock_t t1 = clock();
			_findSimilarity(5, facial5point, coord5point, transform);
			clock_t t2 = clock();
			cv::warpAffine(img, crop, transform, designed_size);
			clock_t t3 = clock();
			//printf("findtrans:%.3f, warp:%.3f\n", 0.001*(t2 - t1), 0.001*(t3 - t2));
			return true;
		}

		template<class BaseType>
		static bool CropImage_160x160(const cv::Mat& img, const BaseType* facial5point, cv::Mat& crop)
		{
			cv::Size designed_size(160, 160);
			BaseType coord5point[10] =
			{
				30.2946 + 32, 51.6963 + 24,
				65.5318 + 32, 51.5014 + 24,
				48.0252 + 32, 71.7366 + 24,
				33.5493 + 32, 92.3655 + 24,
				62.7299 + 32, 92.2041 + 24
			};

			cv::Mat transform;
			clock_t t1 = clock();
			_findSimilarity(5, facial5point, coord5point, transform);
			clock_t t2 = clock();
			cv::warpAffine(img, crop, transform, designed_size);
			clock_t t3 = clock();
			//printf("findtrans:%.3f, warp:%.3f\n", 0.001*(t2 - t1), 0.001*(t3 - t2));
			return true;
		}

	private:
		template<class BaseType>
		static void _findNonreflectiveSimilarity(int nPts, const BaseType* uv, const BaseType* xy, cv::Mat& transform)
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

			int type = CV_32FC1;
			if (_strcmpi(typeid(BaseType).name(), "double") == 0)
				type = CV_64FC1;
			//int type = CV_64FC1;
			//using TmpType = double;
			using TmpType = BaseType;
			cv::Mat X(nPts * 2, 4, type);
			cv::Mat U(nPts * 2, 1, type);
			for (int i = 0; i < nPts; i++)
			{
				X.ptr<TmpType>(i)[0] = xy[i * 2 + 0];
				X.ptr<TmpType>(i)[1] = xy[i * 2 + 1];
				X.ptr<TmpType>(i)[2] = 1;
				X.ptr<TmpType>(i)[3] = 0;
				X.ptr<TmpType>(i + nPts)[0] = xy[i * 2 + 1];
				X.ptr<TmpType>(i + nPts)[1] = -xy[i * 2 + 0];
				X.ptr<TmpType>(i + nPts)[2] = 0;
				X.ptr<TmpType>(i + nPts)[3] = 1;
				U.ptr<TmpType>(i)[0] = uv[i * 2 + 0];
				U.ptr<TmpType>(i + nPts)[0] = uv[i * 2 + 1];
			}
			cv::Mat r(4, 1, type);
			double t1 = omp_get_wtime();
			if (!cv::solve(X, U, r, cv::DECOMP_SVD))
			{
				std::cout << "failed to solve\n";
				return;
			}
			double t2 = omp_get_wtime();
			//printf("solve:%.3f\n", t2 - t1);
			TmpType sc = r.ptr<TmpType>(0)[0];
			TmpType ss = r.ptr<TmpType>(1)[0];
			TmpType tx = r.ptr<TmpType>(2)[0];
			TmpType ty = r.ptr<TmpType>(3)[0];

			TmpType Tinv[9] =
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
			cv::Mat Tinv_mat(3, 3, type), T_mat(3, 3, type);
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
					Tinv_mat.ptr<TmpType>(i)[j] = Tinv[i * 3 + j];
			}
			transform = Tinv_mat;

		}

		template<class BaseType>
		static void _findSimilarity(int nPts, const BaseType* uv, const BaseType* xy, cv::Mat& transform)
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

			int type = CV_32FC1;
			if (_strcmpi(typeid(BaseType).name(), "double") == 0)
				type = CV_64FC1;

			//int type = CV_64FC1;
			//using TmpType = double;
			using TmpType = BaseType;
			cv::Mat transform1, transform2R, transform2;
			clock_t t1 = clock();
			_findNonreflectiveSimilarity(nPts, uv, xy, transform1);
			clock_t t2 = clock();
			/*for (int i = 0; i < 3; i++)
			{
			for (int j = 0; j < 3; j++)
			{
			printf("%12.5f", transform1.ptr<BaseType>(i)[j]);
			}
			printf("\n");
			}*/
			BaseType* xyR = new BaseType[nPts * 2];
			for (int i = 0; i < nPts; i++)
			{
				xyR[i * 2 + 0] = -xy[i * 2 + 0];
				xyR[i * 2 + 1] = xy[i * 2 + 1];
			}
			clock_t t3 = clock();
			_findNonreflectiveSimilarity(nPts, uv, xyR, transform2R);
			clock_t t4 = clock();
			/*for (int i = 0; i < 3; i++)
			{
			for (int j = 0; j < 3; j++)
			{
			printf("%12.5f", transform2R.ptr<BaseType>(i)[j]);
			}
			printf("\n");
			}*/

			const TmpType TreflectY[9] =
			{
				-1, 0,  0,
				0,  1,  0,
				0,  0,  1
			};
			cv::Mat TreflectY_mat(3, 3, type);
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
					TreflectY_mat.ptr<TmpType>(i)[j] = TreflectY[i * 3 + j];
			}
			transform2 = transform2R*TreflectY_mat;

			/*for (int i = 0; i < 3; i++)
			{
			for (int j = 0; j < 3; j++)
			{
			printf("%12.5f", transform2.ptr<BaseType>(i)[j]);
			}
			printf("\n");
			}*/

			//forward transform
			TmpType norm1 = 0, norm2 = 0;
			for (int p = 0; p < nPts; p++)
			{
				TmpType uv1_x = transform1.ptr<TmpType>(0)[0] * xy[p * 2 + 0] + transform1.ptr<TmpType>(1)[0] * xy[p * 2 + 1] + transform1.ptr<TmpType>(2)[0];
				TmpType uv1_y = transform1.ptr<TmpType>(0)[1] * xy[p * 2 + 0] + transform1.ptr<TmpType>(1)[1] * xy[p * 2 + 1] + transform1.ptr<TmpType>(2)[1];
				TmpType uv2_x = transform2.ptr<TmpType>(0)[0] * xy[p * 2 + 0] + transform2.ptr<TmpType>(1)[0] * xy[p * 2 + 1] + transform2.ptr<TmpType>(2)[0];
				TmpType uv2_y = transform2.ptr<TmpType>(0)[1] * xy[p * 2 + 0] + transform2.ptr<TmpType>(1)[1] * xy[p * 2 + 1] + transform2.ptr<TmpType>(2)[1];

				norm1 += (uv[p * 2 + 0] - uv1_x)*(uv[p * 2 + 0] - uv1_x) + (uv[p * 2 + 1] - uv1_y)*(uv[p * 2 + 1] - uv1_y);
				norm2 += (uv[p * 2 + 0] - uv2_x)*(uv[p * 2 + 0] - uv2_x) + (uv[p * 2 + 1] - uv2_y)*(uv[p * 2 + 1] - uv2_y);
			}

			clock_t t5 = clock();
			cv::Mat tmp;
			if (norm1 < norm2)
				cv::invert(transform1, tmp, cv::DECOMP_SVD);
			else
				cv::invert(transform2, tmp, cv::DECOMP_SVD);
			clock_t t6 = clock();

			cv::Mat trans(2, 3, type);
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					trans.ptr<TmpType>(i)[j] = tmp.ptr<TmpType>(j)[i];
					//printf("%f ", trans.ptr<TmpType>(i)[j]);
				}
				//printf("\n");
			}

			transform = trans;
			//printf("%f,%f,%f\n", 0.001*(t2 - t1), 0.001*(t4 - t3), 0.001*(t6 - t5));
		}
	};
}
#endif
