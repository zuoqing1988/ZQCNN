#ifndef _ZQ_MERGE_SORT_H_
#define _ZQ_MERGE_SORT_H_

#include <stdlib.h>

namespace ZQ
{
	class ZQ_MergeSort
	{
	public:
		template<class T>
		static void MergeSort(T* vals, int num, bool ascending_dir);

		template<class T>
		static void MergeSort(T* vals, int* idx, int num, bool ascending_dir);
		
	private:
		template<class T>
		static void _mergeSort(T* vals,int start, int end, bool ascending_dir);

		template<class T>
		static void _mergeSort(T* vals, int* idx, int start, int end, bool ascending_dir);
	};
		
	/********************* definitions ***********************************/

	template<class T>
	void ZQ_MergeSort::MergeSort(T* vals, int num, bool ascending_dir)
	{
		_mergeSort(vals,0,num-1,ascending_dir);
	}

	template<class T>
	void ZQ_MergeSort::MergeSort(T* vals, int* idx, int num, bool ascending_dir)
	{
		_mergeSort(vals,idx,0,num-1,ascending_dir);
	}

	template<class T>
	void ZQ_MergeSort::_mergeSort(T* vals,int start, int end, bool ascending_dir)
	{
		if(start >= end)
			return ;

		int mid = (start+end)/2;
		int left_len = mid-start+1;
		int right_len = end-mid;
		T* tmp_left = new T[left_len];
		T* tmp_right = new T[right_len];
		for(int i = 0;i < left_len;i++)
			tmp_left[i] = vals[start+i];
		for(int i = 0;i < right_len;i++)
			tmp_right[i] = vals[mid+1+i];

		_mergeSort(tmp_left,0,left_len-1,ascending_dir);
		_mergeSort(tmp_right,0,right_len-1,ascending_dir);

		int i_idx = 0;
		int j_idx = 0;
		int k_idx = 0;
		for(;i_idx < left_len && j_idx < right_len;)
		{
			if((tmp_left[i_idx] > tmp_right[j_idx]) != ascending_dir)
			{
				vals[start+k_idx] = tmp_left[i_idx];
				i_idx++;
				k_idx++;
			}
			else
			{
				vals[start+k_idx] = tmp_right[j_idx];
				j_idx++;
				k_idx++;
			}
		}
		if(i_idx == left_len)
		{
			for(;j_idx < right_len;j_idx++,k_idx++)
				vals[start+k_idx] = tmp_right[j_idx];
		}
		else
		{
			for(; i_idx < left_len;i_idx++,k_idx++)
				vals[start+k_idx] = tmp_left[i_idx];
		}
		delete []tmp_left;
		delete []tmp_right;
		return ;
	}

	template<class T>
	void ZQ_MergeSort::_mergeSort(T* vals, int* idx, int start, int end, bool ascending_dir)
	{
		if(start >= end)
			return ;

		int mid = (start+end)/2;
		int left_len = mid-start+1;
		int right_len = end-mid;
		T* tmp_left = new T[left_len];
		T* tmp_right = new T[right_len];
		int* tmp_left_idx = new int[left_len];
		int* tmp_right_idx = new int[right_len];
		for(int i = 0;i < left_len;i++)
		{
			tmp_left[i] = vals[start+i];
			tmp_left_idx[i] = idx[start+i];
		}
		for(int i = 0;i < right_len;i++)
		{
			tmp_right[i] = vals[mid+1+i];
			tmp_right_idx[i] = idx[mid+1+i];
		}

		_mergeSort(tmp_left,tmp_left_idx,0,left_len-1,ascending_dir);
		_mergeSort(tmp_right,tmp_right_idx,0,right_len-1,ascending_dir);

		int i_idx = 0;
		int j_idx = 0;
		int k_idx = 0;
		for(;i_idx < left_len && j_idx < right_len;)
		{
			if((tmp_left[i_idx] > tmp_right[j_idx]) != ascending_dir)
			{
				vals[start+k_idx] = tmp_left[i_idx];
				idx[start+k_idx] = tmp_left_idx[i_idx];
				i_idx++;
				k_idx++;
			}
			else
			{
				vals[start+k_idx] = tmp_right[j_idx];
				idx[start+k_idx] = tmp_right_idx[j_idx];
				j_idx++;
				k_idx++;
			}
		}
		if(i_idx == left_len)
		{
			for(;j_idx < right_len;j_idx++,k_idx++)
			{
				vals[start+k_idx] = tmp_right[j_idx];
				idx[start+k_idx] = tmp_right_idx[j_idx];
			}
		}
		else
		{
			for(; i_idx < left_len;i_idx++,k_idx++)
			{
				vals[start+k_idx] = tmp_left[i_idx];
				idx[start+k_idx] = tmp_left_idx[i_idx];
			}
		}
		delete []tmp_left;
		delete []tmp_right;
		delete []tmp_left_idx;
		delete []tmp_right_idx;
		return ;
	}

}

#endif