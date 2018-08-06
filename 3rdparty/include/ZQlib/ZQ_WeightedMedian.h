#ifndef _ZQ_WEIGHTED_MEDIAN_H_
#define _ZQ_WEIGHTED_MEDIAN_H_
#pragma once
#include "ZQ_QuickSort.h"
#include <string.h>

namespace ZQ
{
	class ZQ_WeightedMedian
	{
	public:
		template<class T>
		static bool FindMedian(const T* vals, const T* weights, int num, T& output)
		{
			if (vals == 0 || weights == 0)
				return false;

			for (int i = 0; i < num;i++)
			if (weights[i] <= 0)
				return false;

			T* sort_vals = new T[num];
			memcpy(sort_vals, vals, sizeof(T)*num);
			int* idx = new int[num];
			for (int i = 0; i < num; i++)
				idx[i] = i;
			ZQ_QuickSort::QuickSort(sort_vals, idx, num, false);
			T* sort_weights = new T[num];
			for (int i = 0; i < num; i++)
				sort_weights[i] = weights[idx[i]];
			T total_weight = 0;
			for (int i = 0; i < num; i++)
			{
				total_weight += sort_weights[i];
			}
			int inf_num = 0;
			T cur_sum_weight = total_weight;
			for (int i = num - 1; i >= 0; i--)
			{
				inf_num++;
				cur_sum_weight -= sort_weights[i] * 2;
				if (cur_sum_weight <= 0)
					break;
			}

			output = sort_vals[num - inf_num];
			delete[]idx;
			delete[]sort_vals;
			delete[]sort_weights;
			return true;
		}
	};
}

#endif