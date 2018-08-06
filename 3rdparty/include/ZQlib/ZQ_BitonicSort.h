#ifndef _ZQ_BITONIC_SORT_H_
#define _ZQ_BITONIC_SORT_H_

#include <math.h>

namespace ZQ
{
	class ZQ_BitonicSort
	{
	public:
		template<class T>
		static bool Sort_Recursive(T* data, int len, int start_idx, bool ascending_dir);

		template<class T>
		static bool Sort_Recursive(T* data, int* idx, int len, int start_idx, bool ascending_dir);
		
		template<class T>
		static bool Sort(T* data, int len, bool ascending_dir);

		template<class T>
		static bool Sort(T* data, int* idx, int len, bool ascending_dir);

	private:

		template<class T>
		static bool _merge_Recursive(T* data, int len, int start_idx, bool ascending_dir);

		template<class T>
		static bool _merge_Recursive(T* data, int* idx, int len, int start_idx, bool ascending_dir);

		template<class T>
		static void _merge(T* data, int len, int sort_lvl, int merge_lvl, bool ascending_dir);

		template<class T>
		static void _merge(T* data, int* idx, int len, int sort_lvl, int merge_lvl, bool ascending_dir);

		template<class T>
		static void _compare(T* data, int i, int j, bool ascending_dir);	

		template<class T>
		static void _compare(T* data, int* idx, int i, int j, bool ascending_dir);	
	};

	/*************************************************************/

	template<class T>
	bool ZQ_BitonicSort::Sort_Recursive(T* data, int len, int start_idx, bool ascending_dir)
	{
		if (len>1)
		{
			if(len%2 != 0)
				return false;
			int k = len/2;
			if(!Sort_Recursive(data, k, start_idx, true))
				return false;
			if(!Sort_Recursive(data, k, start_idx+k, false))
				return false;

			if(!_merge_Recursive(data, len, start_idx, ascending_dir))
				return false;
		}
		return true;
	}

	template<class T>
	bool ZQ_BitonicSort::Sort_Recursive(T* data, int* idx, int len, int start_idx, bool ascending_dir)
	{
		if (len>1)
		{
			if(len%2 != 0)
				return false;
			int k = len/2;
			if(!Sort_Recursive(data, idx, k, start_idx, true))
				return false;
			if(!Sort_Recursive(data, idx, k, start_idx+k, false))
				return false;

			if(!_merge_Recursive(data, idx, len, start_idx, ascending_dir))
				return false;
		}
		return true;
	}

	template<class T>
	bool ZQ_BitonicSort::Sort(T* data, int len, bool ascending_dir)
	{
		int max_levels = 0;
		int cur_len = len;
		while(cur_len > 1)
		{
			if(cur_len%2 != 0)
				return false;
			max_levels ++;
			cur_len /= 2;
		}

		for(int sort_lvl = 1; sort_lvl <= max_levels;sort_lvl++)
		{
			for(int merge_lvl = sort_lvl;merge_lvl > 0;merge_lvl--)
				_merge(data,len,sort_lvl,merge_lvl,ascending_dir);
		}
		return true;
	}

	template<class T>
	bool ZQ_BitonicSort::Sort(T* data, int* idx, int len, bool ascending_dir)
	{
		int max_levels = 0;
		int cur_len = len;
		while(cur_len > 1)
		{
			if(cur_len%2 != 0)
				return false;
			max_levels ++;
			cur_len /= 2;
		}

		for(int sort_lvl = 1; sort_lvl <= max_levels;sort_lvl++)
		{
			for(int merge_lvl = sort_lvl;merge_lvl > 0;merge_lvl--)
				_merge(data,idx,len,sort_lvl,merge_lvl,ascending_dir);
		}
		return true;
	}

	template<class T>
	bool ZQ_BitonicSort::_merge_Recursive(T* data, int len, int start_idx, bool ascending_dir)
	{
		if (len>1)
		{
			if(len%2 != 0)
				return false;
			int k=len/2;
			for (int i= start_idx ; i < start_idx+k; i++)
				_compare(data, i, i+k, ascending_dir);
			if(!_merge_Recursive(data, k, start_idx, ascending_dir))
				return false;
			if(!_merge_Recursive(data, k, start_idx+k, ascending_dir))
				return false;
		}
		return true;
	}

	template<class T>
	bool ZQ_BitonicSort::_merge_Recursive(T* data, int* idx, int len, int start_idx, bool ascending_dir)
	{
		if (len>1)
		{
			if(len%2 != 0)
				return false;
			int k=len/2;
			for (int i= start_idx ; i < start_idx+k; i++)
				_compare(data, idx, i, i+k, ascending_dir);
			if(!_merge_Recursive(data, idx, k, start_idx, ascending_dir))
				return false;
			if(!_merge_Recursive(data, idx, k, start_idx+k, ascending_dir))
				return false;
		}
		return true;
	}

	template<class T>
	void ZQ_BitonicSort::_merge(T* data, int len, int sort_lvl, int merge_lvl, bool ascending_dir)
	{
		int sort_block_size = 1;
		for(int ll = 0;ll < sort_lvl;ll++)
			sort_block_size *= 2;

		int merge_block_size = 1;
		for(int ll = 0;ll < merge_lvl;ll++)
			merge_block_size *= 2;

		int merge_step_size = merge_block_size/2;

		for(int i = 0;i < len/2;i++)
		{
			int cur_sort_block_idx = i/(sort_block_size/2);
			int cur_dir = (cur_sort_block_idx%2 == 0) ? ascending_dir : (!ascending_dir);

			int cur_merge_block_idx = i/(merge_block_size/2);
			int cur_merge_block_off = i%(merge_block_size/2);

			int real_i = cur_merge_block_idx*merge_block_size+cur_merge_block_off;
			int real_j = real_i+merge_step_size;
			_compare(data,real_i,real_j,cur_dir);
		}
	}

	template<class T>
	void ZQ_BitonicSort::_merge(T* data, int* idx, int len, int sort_lvl, int merge_lvl, bool ascending_dir)
	{
		int sort_block_size = 1;
		for(int ll = 0;ll < sort_lvl;ll++)
			sort_block_size *= 2;

		int merge_block_size = 1;
		for(int ll = 0;ll < merge_lvl;ll++)
			merge_block_size *= 2;

		int merge_step_size = merge_block_size/2;

		for(int i = 0;i < len/2;i++)
		{
			int cur_sort_block_idx = i/(sort_block_size/2);
			int cur_dir = (cur_sort_block_idx%2 == 0) ? ascending_dir : (!ascending_dir);

			int cur_merge_block_idx = i/(merge_block_size/2);
			int cur_merge_block_off = i%(merge_block_size/2);

			int real_i = cur_merge_block_idx*merge_block_size+cur_merge_block_off;
			int real_j = real_i+merge_step_size;
			_compare(data,idx,real_i,real_j,cur_dir);
		}
	}

	template<class T>
	void ZQ_BitonicSort::_compare(T* data, int i, int j, bool ascending_dir)
	{
		if (ascending_dir == (data[i]>data[j]))
		{
			T tmp = data[i];
			data[i] = data[j];
			data[j] = tmp;
		}
	}

	template<class T>
	void ZQ_BitonicSort::_compare(T* data, int* idx, int i, int j, bool ascending_dir)
	{
		if (ascending_dir == (data[i]>data[j]))
		{
			T tmp = data[i];
			data[i] = data[j];
			data[j] = tmp;
			int tmp_idx = idx[i];
			idx[i] = idx[j];
			idx[j] = tmp_idx;
		}
	}
}

#endif