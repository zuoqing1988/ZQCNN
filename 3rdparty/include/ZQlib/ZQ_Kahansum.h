#ifndef _ZQ_KAHANSUM_H_
#define _ZQ_KAHANSUM_H_
#pragma once

namespace ZQ
{
	template<class T>
	T ZQ_KahanSum(const T* input, const int n)
	{
		T y = 0;
		T t = 0;
		T sum = 0;
		T c = 0;
		for(int i = 0;i < n;i++)
		{
			y = input[i] - c;
			t = sum + y;
			c = (t - sum) - y;
			sum = t;
		}
		return sum;
	}
}



#endif