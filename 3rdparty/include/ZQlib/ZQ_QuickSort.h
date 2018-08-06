#ifndef _ZQ_QUICK_SORT_H_
#define _ZQ_QUICK_SORT_H_

#include <stdlib.h>

namespace ZQ
{
	class ZQ_QuickSort
	{
	public:
		template<class T>
		static void QuickSort(T* vals, int num, bool ascending_dir);

		template<class T>
		static void QuickSort(T* vals, int* idx, int num, bool ascending_dir);

		/*0<= k < num*/
		template<class T>
		static bool FindKthMax(T* vals, int num, int k, T& output);
		
		/*0<= k < num*/
		template<class T>
		static bool FindKthMax(T* vals, int* idx, int num, int k, T& output, int& out_idx);

	private:
		template<class T>
		static void _quickSort(T* vals, int start, int end, bool ascending_dir);

		template<class T>
		static void _quickSort(T* vals, int* idx, int start, int end, bool ascending_dir);

		template<class T>
		static bool _findKthMax(T* vals, int start, int end, int k, T& output);

		template<class T>
		static bool _findKthMax(T* vals, int* idx, int start, int end, int k, T& output, int& out_idx);

		template<class T>
		static void _swap(T* vals, int i, int j);

		template<class T>
		static void _swap(T* vals, int* idx, int i, int j);
	};

	/********************* definitions ***********************************/

	template<class T>
	void ZQ_QuickSort::QuickSort(T* vals, int num, bool ascending_dir)
	{
		_quickSort(vals, 0, num - 1, ascending_dir);
	}

	template<class T>
	void ZQ_QuickSort::QuickSort(T* vals, int* idx, int num, bool ascending_dir)
	{
		_quickSort(vals, idx, 0, num - 1, ascending_dir);
	}

	template<class T>
	bool ZQ_QuickSort::FindKthMax(T* vals, int num, int k, T& output)
	{
		return _findKthMax(vals, 0, num - 1, k, output);
	}

	template<class T>
	bool ZQ_QuickSort::FindKthMax(T* vals, int* idx, int num, int k, T& output, int& out_idx)
	{
		return _findKthMax(vals, idx, 0, num - 1, k, output, out_idx);
	}

	template<class T>
	void ZQ_QuickSort::_quickSort(T* vals, int start, int end, bool ascending_dir)
	{
		if (start >= end)
			return;

		int len = end - start + 1;
		int rand_idx = rand() % len + start;
		_swap(vals, start, rand_idx);

		T tmp_val = vals[start];
		int i = start;
		int j = end;
		if (ascending_dir)
		{
			while (i < j)
			{
				for (; vals[j] >= tmp_val && i < j; j--);
				vals[i] = vals[j];
				if (i == j)
					break;
				i++;
				for (; vals[i] < tmp_val && i < j; i++);
				vals[j] = vals[i];
				j--;
			}
			vals[i] = tmp_val;
		}
		else
		{
			while (i < j)
			{
				for (; vals[j] < tmp_val && i < j; j--);
				vals[i] = vals[j];
				if (i == j)
					break;
				i++;
				for (; vals[i] >= tmp_val && i < j; i++);
				vals[j] = vals[i];
				j--;
			}
			vals[i] = tmp_val;
		}

		_quickSort(vals, start, i - 1, ascending_dir);
		_quickSort(vals, i + 1, end, ascending_dir);
		
		return;
	}

	template<class T>
	void ZQ_QuickSort::_quickSort(T* vals, int* idx, int start, int end, bool ascending_dir)
	{
		if (start >= end)
			return;

		int len = end - start + 1;
		int rand_idx = rand() % len + start;
		_swap(vals, idx, start, rand_idx);

		T tmp_val = vals[start];
		int tmp_idx = idx[start];
		int i = start;
		int j = end;
		if (ascending_dir)
		{
			while (i < j)
			{
				for (; vals[j] >= tmp_val && i < j; j--);
				vals[i] = vals[j];
				idx[i] = idx[j];
				if (i == j)
					break;
				i++;
				for (; vals[i] < tmp_val && i < j; i++);
				vals[j] = vals[i];
				idx[j] = idx[i];
				j--;
			}
			vals[i] = tmp_val;
			idx[i] = tmp_idx;
		}
		else
		{
			while (i < j)
			{
				for (; vals[j] < tmp_val && i < j; j--);
				vals[i] = vals[j];
				idx[i] = idx[j];
				if (i == j)
					break;
				i++;
				for (; vals[i] >= tmp_val && i < j; i++);
				vals[j] = vals[i];
				idx[j] = idx[i];
				j--;
			}
			vals[i] = tmp_val;
			idx[i] = tmp_idx;
		}
		_quickSort(vals, idx, start, i - 1, ascending_dir);
		_quickSort(vals, idx, i + 1, end, ascending_dir);
		return;
	}

	template<class T>
	bool ZQ_QuickSort::_findKthMax(T* vals, int start, int end, int k, T& output)
	{
		if (start > end || k < start || k > end)
			return false;

		int len = end - start + 1;
		int rand_idx = rand() % len + start;
		_swap(vals, start, rand_idx);

		T tmp_val = vals[start];
		int i = start;
		int j = end;
		
		while (i < j)
		{
			for (; vals[j] < tmp_val && i < j; j--);
			vals[i] = vals[j];
			if (i == j)
				break;
			i++;
			for (; vals[i] >= tmp_val && i < j; i++);
			vals[j] = vals[i];
			j--;
		}
		vals[i] = tmp_val;

		if (i == k)
		{
			output = tmp_val;
			return true;
		}
		else if (i > k)
		{
			return _findKthMax(vals, start, i - 1, k, output);
		}
		else
		{
			return _findKthMax(vals, i + 1, end, k, output);
		}
	}

	template<class T>
	bool ZQ_QuickSort::_findKthMax(T* vals, int* idx, int start, int end, int k, T& output, int& out_idx)
	{
		if (start > end || k < start || k > end)
			return false;

		int len = end - start + 1;
		int rand_idx = rand() % len + start;
		_swap(vals, idx, start, rand_idx);

		T tmp_val = vals[start];
		int tmp_idx = idx[start];
		int i = start;
		int j = end;

		while (i < j)
		{
			for (; vals[j] < tmp_val && i < j; j--);
			vals[i] = vals[j];
			idx[i] = idx[j];
			if (i == j)
				break;
			i++;
			for (; vals[i] >= tmp_val && i < j; i++);
			vals[j] = vals[i];
			idx[j] = idx[i];
			j--;
		}
		vals[i] = tmp_val;
		vals[i] = tmp_idx;

		if (i == k)
		{
			output = tmp_val;
			out_idx = tmp_idx;
			return true;
		}
		else if (i > k)
		{
			return _findKthMax(vals, idx, start, i - 1, k, output, out_idx);
		}
		else
		{
			return _findKthMax(vals, idx, i + 1, end, k, output, out_idx);
		}
	}

	template<class T>
	void ZQ_QuickSort::_swap(T* vals, int i, int j)
	{
		T tmp = vals[i];
		vals[i] = vals[j];
		vals[j] = tmp;
	}

	template<class T>
	void ZQ_QuickSort::_swap(T* vals, int* idx, int i, int j)
	{
		T tmp_v = vals[i];
		vals[i] = vals[j];
		vals[j] = tmp_v;
		int tmp_i = idx[i];
		idx[i] = idx[j];
		idx[j] = tmp_i;
	}
}

#endif