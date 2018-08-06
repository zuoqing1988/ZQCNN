#ifndef _ZQ_KMEANS_H_
#define _ZQ_KMEANS_H_
#pragma once

#include <stdlib.h>
#include <string.h>

namespace ZQ
{
	template<class T>
	class ZQ_Kmeans
	{
	public:
		static bool Kmeans_with_init(int nPts, int dim, int k, const T* pts, const T* init_centers, int* idx, T* out_centers, double thresh = 1e-9)
		{
			if (nPts <= 0 || dim <= 0 || k > nPts || pts == 0 || idx == 0 || out_centers == 0)
				return false;

			double thresh2 = thresh*thresh;
			if (thresh2 < 1e-32)
				thresh2 = 1e-32;

			memcpy(out_centers, init_centers, sizeof(T)*k*dim);
			int* sum_kid = new int[k];
			T* sum_centers = new T[k*dim];
			while (true)
			{
				for (int i = 0; i < nPts; i++)
				{
					T min_dis2 = _distance2(dim, pts + i*dim, out_centers);
					int min_kid = 0;
					for (int j = 1; j < k; j++)
					{
						T cur_dis2 = _distance2(dim, pts + i*dim, out_centers + j*dim);
						if (min_dis2 > cur_dis2)
						{
							min_kid = j;
							min_dis2 = cur_dis2;
						}
					}
					idx[i] = min_kid;
				}
				
				
				memset(sum_kid, 0, sizeof(int)*k);
				memset(sum_centers, 0, sizeof(T)*k*dim);
				for (int i = 0; i < nPts; i++)
				{
					int kid = idx[i];
					sum_kid[kid]++;
					for (int j = 0; j < dim; j++)
						sum_centers[kid*dim + j] += pts[i*dim+j];
					
				}
				for (int i = 0; i < k; i++)
				{
					if (sum_kid[i] > 0)
					{
						for (int j = 0; j < dim; j++)
							sum_centers[i*dim + j] /= sum_kid[i];
					}
				}
				T delta2 = _distance2(k*dim, sum_centers, out_centers);
				if (delta2 < thresh2)
					break;
				
				memcpy(out_centers, sum_centers, sizeof(T)*k*dim);
			}
			memcpy(out_centers, sum_centers, sizeof(T)*k*dim);
			delete[]sum_centers;
			delete[]sum_kid;
			return true;
		}

		static bool KmeansNormVec_with_init(int nPts, int dim, int k, const T* pts, const T* init_centers, int* idx, T* out_centers, double thresh = 1e-9)
		{
			if (nPts <= 0 || dim <= 0 || k > nPts || pts == 0 || idx == 0 || out_centers == 0)
				return false;

			thresh = fabs(thresh);
			if (thresh < 1e-32)
				thresh = 1e-32;

			memcpy(out_centers, init_centers, sizeof(T)*k*dim);
			int* sum_kid = new int[k];
			T* sum_centers = new T[k*dim];
			while (true)
			{
				for (int i = 0; i < nPts; i++)
				{
					T min_dis = _distance_normvec(dim, pts + i*dim, out_centers);
					int min_kid = 0;
					for (int j = 1; j < k; j++)
					{
						T cur_dis = _distance_normvec(dim, pts + i*dim, out_centers + j*dim);
						if (min_dis > cur_dis)
						{
							min_kid = j;
							min_dis = cur_dis;
						}
					}
					idx[i] = min_kid;
				}


				memset(sum_kid, 0, sizeof(int)*k);
				memset(sum_centers, 0, sizeof(T)*k*dim);
				for (int i = 0; i < nPts; i++)
				{
					int kid = idx[i];
					sum_kid[kid]++;
					for (int j = 0; j < dim; j++)
						sum_centers[kid*dim + j] += pts[i*dim + j];

				}
				for (int i = 0; i < k; i++)
				{
					_normlize(dim, sum_centers + i*dim);
				}
				T delta = _distance_normvec(k*dim, sum_centers, out_centers);
				if (delta < thresh)
					break;

				memcpy(out_centers, sum_centers, sizeof(T)*k*dim);
			}
			memcpy(out_centers, sum_centers, sizeof(T)*k*dim);
			delete[]sum_centers;
			delete[]sum_kid;
			return true;
		}
		
		static bool Kmeans(int nPts, int dim, int k, const T* pts, int* idx, T* out_centers, T* init_centers = 0, double thresh = 1e-9)
		{
			if (nPts <= 0 || dim <= 0 || k > nPts || pts == 0 || idx == 0 || out_centers == 0)
				return false;

			T* tmp_init_centers = 0;
			if (init_centers == 0)
				tmp_init_centers = new T[k*dim];
			else
				tmp_init_centers = init_centers;

			if (!_select_init_center(nPts, dim, k, pts, tmp_init_centers))
			{
				if (init_centers == 0)
					delete[]tmp_init_centers;
				return false;
			}

			if (!Kmeans_with_init(nPts, dim, k, pts, tmp_init_centers, idx, out_centers, thresh))
			{
				if (init_centers == 0)
					delete[]tmp_init_centers;
				return false;
			}
			if (init_centers == 0)
				delete[]tmp_init_centers;
			return true;
		}

		static bool KmeansNormVec(int nPts, int dim, int k, const T* pts, int* idx, T* out_centers, T* init_centers = 0, double thresh = 1e-9)
		{
			if (nPts <= 0 || dim <= 0 || k > nPts || pts == 0 || idx == 0 || out_centers == 0)
				return false;

			T* tmp_init_centers = 0;
			if (init_centers == 0)
				tmp_init_centers = new T[k*dim];
			else
				tmp_init_centers = init_centers;

			if (!_select_init_center(nPts, dim, k, pts, tmp_init_centers))
			{
				if (init_centers == 0)
					delete[]tmp_init_centers;
				return false;
			}

			if (!KmeansNormVec_with_init(nPts, dim, k, pts, tmp_init_centers, idx, out_centers, thresh))
			{
				if (init_centers == 0)
					delete[]tmp_init_centers;
				return false;
			}
			if (init_centers == 0)
				delete[]tmp_init_centers;
			return true;
		}

	//private:
	public:
		static bool _select_init_center(int nPts, int dim, int k, const T* pts, T* init_centers)
		{
			int* idx = new int[nPts];
			int* init_idx = new int[k];
			for (int i = 0; i < nPts; i++)
				idx[i] = i;
			for (int i = 0; i < k; i++)
			{
				int rand_id = rand() % (nPts-i);
				init_idx[i] = idx[rand_id];
				int tmp = idx[rand_id];
				idx[rand_id] = idx[nPts - i - 1];
				idx[nPts - i - 1] = tmp;
			}

			for (int i = 0; i < k; i++)
			{
				memcpy(init_centers + i*dim, pts + init_idx[i] * dim, sizeof(T)*dim);
			}
			delete[]idx;
			delete[]init_idx;
			return true;
		}

		static T _distance2(int dim, const T* pt1, const T* pt2)
		{
			T result = 0;
			for (int i = 0; i < dim; i++)
			{
				result += (pt1[i] - pt2[i])*(pt1[i] - pt2[i]);
			}
			return result;
		}

		static T _distance_normvec(int dim, const T* pt1, const T* pt2)
		{
			T result = 0;
			for (int i = 0; i < dim; i++)
			{
				result += pt1[i] * pt2[i];
			}
			return 1 - result;
		}

		

		static void _normlize(int dim, T* vec)
		{
			T len = 0;
			for (int i = 0; i < dim; i++)
			{
				len += vec[i] * vec[i];
			}
			len = sqrt(len);
			if (len != 0)
			{
				for (int i = 0; i < dim; i++)
					vec[i] /= len;
			}
		}
	};
}

#endif