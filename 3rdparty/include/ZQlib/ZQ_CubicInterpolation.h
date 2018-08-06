#ifndef _ZQ_CUBIC_INTERPOLATION_H_
#define _ZQ_CUBIC_INTERPOLATION_H_
#pragma once

namespace ZQ
{

	template<class T>
	T ZQ_CubicInterpolate(const T* p, float x);

	template<class T>
	T ZQ_BicubicInterpolate(const T* p, float x, float y);

	template<class T>
	T ZQ_TricubicInterpolate(const T* p, float x, float y, float z);

	template<class T>
	T ZQ_nCubicInterpolate(int n, const T* p, const float* coordinates);


	/*********************************************************************************/
	/********************************** definitions **********************************/
	/*********************************************************************************/

	template<class T>
	T ZQ_CubicInterpolate(const T* p, float x)
	{
		double dk = 0.5*(p[2]-p[0]);
		double dk1 = 0.5*(p[3]-p[1]);
		double deltak = (double)p[2]-p[1];

		if(deltak == 0)
			dk = dk1 = 0;
		else
		{
			if(dk*deltak < 0)
				dk = 0;
			if(dk1*deltak < 0)
				dk1 = 0;
		}

		double a3 = dk+dk1-2*deltak;
		double a2 = 3*deltak - 2*dk - dk1;
		double a1 = dk;
		double a0 = p[1];

		return a0 + x*(a1 + x*(a2+x*a3));
	}

	template<class T>
	T ZQ_BicubicInterpolate(const T* p, float x, float y)
	{
		T arr[4];
		arr[0] = ZQ_CubicInterpolate(p, x);
		arr[1] = ZQ_CubicInterpolate(p+4, x);
		arr[2] = ZQ_CubicInterpolate(p+8, x);
		arr[3] = ZQ_CubicInterpolate(p+12, x);
		return ZQ_CubicInterpolate(arr, y);
	}

	template<class T>
	T ZQ_TricubicInterpolate(const T* p, float x, float y, float z)
	{
		T arr[4];
		arr[0] = ZQ_BicubicInterpolate(p, x, y);
		arr[1] = ZQ_BicubicInterpolate(p+16, x, y);
		arr[2] = ZQ_BicubicInterpolate(p+32, x, y);
		arr[3] = ZQ_BicubicInterpolate(p+48, x, y);
		return ZQ_CubicInterpolate(arr, z);
	}

	template<class T>
	T ZQ_nCubicInterpolate(int n, const T* p, const float* coordinates)
	{
		if (n == 1) 
		{
			return ZQ_CubicInterpolate(p, coordinates[0]);
		}
		else 
		{
			T arr[4];
			int skip = 1 << ((n - 1) * 2);
			arr[0] = ZQ_nCubicInterpolate(n-1, p, coordinates);
			arr[1] = ZQ_nCubicInterpolate(n-1, p+skip, coordinates);
			arr[2] = ZQ_nCubicInterpolate(n-1, p+2*skip, coordinates);
			arr[3] = ZQ_nCubicInterpolate(n-1, p+3*skip, coordinates);
			return ZQ_CubicInterpolate(arr, coordinates[n-1]);
		}
	}
}

#endif