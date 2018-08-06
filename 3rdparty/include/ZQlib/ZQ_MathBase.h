#ifndef _ZQ_MATH_BASE_H_
#define _ZQ_MATH_BASE_H_
#pragma once

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <typeinfo>

namespace ZQ
{
	class ZQ_MathBase
	{
	private:
		template<class T>
		static void _swap(T& val1, T& val2){T tmp = val1; val1 = val2; val2 = tmp;}
	public:
		template<class T>
		static T Length2(int n, const T* data);

		template<class T>
		static T NormVector_L2(int n, const T* data);
		
		template<class T>
		static T NormVector_Linf(int n, const T* data);
		
		template<class T>
		static T NormVector_L1(int n, const T* data);

		template<class T>
		static void Normalize(int n, T* data);
		
		template<class T>
		static T DotProduct(int n, const T* v1, const T* v2);
		
		template<class T>
		static T DotProductSparse(int n, int len1, const int* idx1, const T* val1, int len2, const int* idx2, const T* val2);

		template<class T>
		static void CrossProduct(const T v1[3], const T v2[3], T v3[3]);

		template<class T>
		static int BinarySearch(int n, const T* data, T val, bool ascend_order);

		template<class T>
		static void VecPlus(int n, const T* src1, const T* src2, T* dst);

		template<class T>
		static void VecPlus(int n, const T* src1, T weight1, const T* src2, T weight2, T* dst);

		template<class T>
		static void VecMinus(int n, const T* src1, const T* src2, T* dst);

		template<class T>
		static void VecMinus(int n, const T* src1, T weight1, const T* src2, T weight2, T* dst);

		template<class T>
		static void VecMul(int n, const T* src1, const T* src2, T* dst);

		/*take care div zero*/
		template<class T>
		static void VecDiv(int n, const T* src1, const T* src2, T* dst);

		template<class T>
		static int Sign(T val);

		template<class T>
		static T Rem(T x, T y);

		template<class T>
		static void FindMin(int n, const T* data, T& minval, int& idx);

		template<class T>
		static void FindMax(int n, const T* data, T& maxval, int& idx);
		
		template<class T>
		static void MatrixMul(const T* mat1, const T* mat2, int row1, int col1, int col2, T* mat3);

		template<class T>
		static void MatrixIdentity(T* mat, int n);

		template<class T>
		static void MatrixTranspose(const T* src, int row, int col, T* dst);

		template<class T>
		static bool MatrixInverse(const T* src, int n, T* dst);

		template<class T>
		static T Det(int n, const T* A);

		/*
		I have found in some cases this function goes into an endless loop.
		for <class float>.
		I suggest one should use a robust code library to replace this function.
		*/
		template<class T>
		static bool SVD_Decompose(const T* mat, int row, int col, T* Umat, T* Smat, T* Vmat);

		template<class T>
		static double Cond_by_double_svd(const T* mat, int row, int col, bool& succ, bool& is_singular);

	};


	/****************  definitions *********************/

	template<class T>
	T ZQ_MathBase::Length2(int n, const T* data)
	{
		T result = 0;
		for(int i = 0;i < n;i++)
			result += data[i]*data[i];
		return result;
	}


	template<class T>
	T ZQ_MathBase::NormVector_L2(int n, const T *data)
	{
		return sqrt((double)Length2(n,data));
	}

	template<class T>
	T ZQ_MathBase::NormVector_Linf(int n, const T *data)
	{
		T result = 0;
		for(int i = 0;i < n;i++)
		{
			T tmp = data[i] > 0 ? data[i] : -data[i];
			if(result < tmp)
				result = tmp;
		}
		return result;
	}

	template<class T>
	T ZQ_MathBase::NormVector_L1(int n, const T* data)
	{
		T result = 0;
		for(int i = 0;i < n;i++)
		{
			T tmp = data[i] > 0 ? data[i] : -data[i];
			result += tmp;
		}
		return result;
	}

	template<class T>
	void ZQ_MathBase::Normalize(int n, T* data)
	{
		T len2 = Length2(n,data);
		if(len2 == 0)
			return;

		T len = sqrt((double)len2);
		for(int i = 0;i < n;i++)
			data[i] /= len;
	}

	template<class T>
	T ZQ_MathBase::DotProduct(int n, const T* v1, const T* v2)
	{
		T result = 0;
		for(int i = 0;i < n;i++)
			result += v1[i]*v2[i];
		return result;
	}

	template<class T>
	T ZQ_MathBase::DotProductSparse(int n, int len1, const int* idx1, const T* val1, int len2, const int* idx2, const T* val2)
	{
		T result = 0;
		for(int i = 0,j = 0; i < len1 && j < len2 && idx1[i] < n && idx2[j] < n;)
		{
			if(idx1[i] == idx2[j])
			{
				result += val1[i] * val2[j];
				i++;
				j++;
			}
			else if(idx1[i] > idx2[j])
			{
				j++;
			}
			else
			{
				i++;
			}
		}
		return result;
	}

	template<class T>
	void ZQ_MathBase::CrossProduct(const T v1[3], const T v2[3], T v3[3])
	{
		v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
		v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
		v3[2] = v1[0] * v2[1] - v2[0] * v1[1];
	}

	template<class T>
	int ZQ_MathBase::BinarySearch(int n, const T* data, T val, bool ascend_order)
	{
		int index = -1;
		if( n == 0)
			return index;

		if(ascend_order)
		{
			int low = 0;
			int high = n-1;
			while(low <= high)
			{
				int mid = (low+high)/2;
				if(val == data[mid])
				{
					index = mid;
					break;
				}
				else if(val > data[mid])
				{
					low = mid+1;
				}
				else
				{
					high = mid-1;
				}
			}

		}
		else
		{
			int low = 0;
			int high = n-1;
			while(low <= high)
			{
				int mid = (low+high)/2;
				if(val == data[mid])
				{
					index = mid;
					break;
				}
				else if(val < data[mid])
				{
					low = mid+1;
				}
				else
				{
					high = mid-1;
				}
			}

		}
		return index;
	}

	template<class T>
	void ZQ_MathBase::VecPlus(int n, const T* src1, const T* src2, T* dst)
	{
		for(int i = 0;i < n;i++)
			dst[i] = src1[i]+src2[i];
	}

	template<class T>
	void ZQ_MathBase::VecPlus(int n, const T* src1, T weight1, const T* src2, T weight2, T* dst)
	{
		for(int i = 0;i < n;i++)
			dst[i] = src1[i]*weight1 + src2[i]*weight2;
	}

	template<class T>
	void ZQ_MathBase::VecMinus(int n, const T* src1, const T* src2, T* dst)
	{
		for(int i = 0;i < n;i++)
			dst[i] = src1[i] - src2[i];
	}

	template<class T>
	void ZQ_MathBase::VecMinus(int n, const T* src1, T weight1, const T* src2, T weight2, T* dst)
	{
		for(int i = 0;i < n;i++)
			dst[i] = src1[i]*weight1 - src2[i]*weight2;
	}

	template<class T>
	void ZQ_MathBase::VecMul(int n, const T* src1, const T* src2, T* dst)
	{
		for(int i = 0;i < n;i++)
			dst[i] = src1[i]*src2[i];
	}

	/*take care div zero*/
	template<class T>
	void ZQ_MathBase::VecDiv(int n, const T* src1, const T* src2, T* dst)
	{
		for(int i = 0;i < n;i++)
			dst[i] = src1[i]/src2[i];
	}

	template<class T>
	int ZQ_MathBase::Sign(T val)
	{
		if(val == 0)
			return 0;
		else
			return val > 0 ? 1 : -1;
	}

	template<class T>
	T ZQ_MathBase::Rem(T x, T y)
	{
		if(y == 0)
			return 0;
		else
			return x - floor(x/y)*y;
	}

	template<class T>
	void ZQ_MathBase::FindMin(int n, const T* data, T& minval, int& idx)
	{
		if(n <= 0)
			return;
		
		minval = data[0];
		idx = 0;
		for(int i = 1;i < n;i++)
		{
			if(minval > data[i])
			{
				minval = data[i];
				idx = i;
			}
		}
	}

	template<class T>
	void ZQ_MathBase::FindMax(int n, const T* data, T& maxval, int& idx)
	{
		if(n <= 0)
			return;

		maxval = data[0];
		idx = 0;
		for(int i = 1;i < n;i++)
		{
			if(maxval < data[i])
			{
				maxval = data[i];
				idx = i;
			}
		}
	}

	template<class T>
	void ZQ_MathBase::MatrixMul(const T* mat1, const T* mat2, int row1, int col1, int col2, T* mat3)
	{
		memset(mat3,0,sizeof(T)*row1*col2);
		for(int i = 0;i < row1;i++)
		{
			for(int j = 0;j < col2;j++)
			{
				for(int k = 0;k < col1;k++)
					mat3[i*col2+j] += mat1[i*col1+k] * mat2[k*col2+j];
			}
		}
	}

	template<class T>
	void ZQ_MathBase::MatrixIdentity(T* mat, int n)
	{
		memset(mat,0,sizeof(T)*n*n);
		for(int i = 0;i < n;i++)
			mat[i*n+i] = 1;
	}

	template<class T>
	void ZQ_MathBase::MatrixTranspose(const T* src, int row, int col, T* dst)
	{
		for(int i = 0;i < row;i++)
		{
			for(int j = 0;j < col;j++)
				dst[j*row+i] = src[i*col+j];
		}
	}

	template<class T>
	bool ZQ_MathBase::MatrixInverse(const T* src, int n, T* dst)
	{
		T* a = new T[n*n];
		for(int i = 0;i < n;i++)
		{
			for(int j = 0;j < n;j++)
				a[i*n+j] = src[i*n+j];
		}

		int* is = new int[n];
		int* js = new int[n];

		for(int k = 0; k < n;k++)
		{ 
			T d = 0.0;
			for(int i = k;i < n;i++)
			{
				for(int j = k;j < n;j++)
				{
					int l = i * n + j;
					T p = a[l] > 0 ? a[l] : -a[l];
					if(p > d)
					{
						d = p;
						is[k] = i;
						js[k] = j;
					}
				}
			}

			if(d == 0)
			{
				delete []is;
				delete []js;
				delete []a;
				return false;
			}

			if(is[k] != k)
			{
				for(int j = 0;j < n;j++)
				{
					int u = k*n + j;
					int v = is[k]*n + j;
					_swap(a[u],a[v]);
				}
			}
			if (js[k] != k)
			{
				for(int i = 0;i < n;i++)
				{
					int u = i*n + k;
					int v = i*n + js[k];
					_swap(a[u],a[v]);
				}
			}
			int l = k*n + k;
			a[l] = 1.0 / a[l];
			for(int j = 0;j < n;j++)
			{
				if(j != k)
				{
					int u = k*n + j;
					a[u] *= a[l];
				}
			}
			for(int i = 0;i < n;i++)
			{
				if(i != k)
				{
					for(int j = 0;j < n;j++)
					{
						if (j != k)
						{ 
							int u = i*n + j;
							a[u] -= a[i*n+k] * a[k*n+j];
						}
					}
				}
			}
			for(int i = 0;i < n;i++)
			{
				if(i != k)
				{
					int u = i*n + k;
					a[u] *= -a[l];
				}
			}
		}
		for(int k = n-1;k >= 0;k--)
		{
			if(js[k] != k)
			{
				for(int j = 0;j < n;j++)
				{
					int u = k*n + j;
					int v = js[k]*n + j;
					_swap(a[u],a[v]);
				}
			}
			if(is[k] != k)
			{
				for(int i = 0;i < n;i++)
				{ 
					int u = i*n + k;
					int v = i*n + is[k];
					_swap(a[u],a[v]);
				}
			}
		}
		delete []is;
		delete []js;
		memcpy(dst,a,sizeof(T)*n*n);
		delete []a;
		return true;
	}

	template<class T>
	T ZQ_MathBase::Det(int n, const T* A)
	{
		if (n <= 0)
			return 0;
		else if (n == 1)
		{
			return A[0];
		}
		else if (n == 2)
		{
			return A[0] * A[3] - A[1] * A[2];
		}
		else if (n == 3)
		{
			return A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[5]);
		}
		else
		{
			T result = 0;
			T* tmp_A = new T[(n - 1)*(n - 1)];
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < n - 1; j++)
				{
					for (int k = 0, real_k = 0; k < n; k++)
					{
						if (k == i)
							continue;
						else
						{
							tmp_A[j*(n - 1) + real_k] = A[(j + 1)*n + k];
							real_k++;
						}
					}
				}
				result += (i % 2 == 0 ? 1 : -1)*Det(n - 1, tmp_A);
			}
			delete[]tmp_A;
			return result;
		}
	}

	template<class T>
	bool ZQ_MathBase::SVD_Decompose(const T *mat, int row, int col, T *Umat, T *Smat, T *Vmat)
	{
		int sdim = __min(row,col);

		bool transpose_flag = false;
		if(row < col)
			transpose_flag = true;

		int m,n;
		if(transpose_flag)
		{
			m = col;
			n = row;
		}
		else
		{
			m = row;
			n = col;
		}
		T** A = (T**)malloc(sizeof(T*)* m);
		for(int i = 0;i < m;i++)
		{
			A[i] = (T*)malloc(sizeof(T)*n);
			for(int j = 0;j < n;j++)
			{
				if(!transpose_flag)
					A[i][j] = mat[i*n+j];
				else
					A[i][j] = mat[j*m+i];
			}
		}

		T** U = (T**)malloc(sizeof(T*)* m);
		for(int i = 0;i < m;i++)
		{
			U[i] = (T*)malloc(sizeof(T)*m); 
			memset(U[i],0,sizeof(T)*m);
		}

		T** V = (T**)malloc(sizeof(T*)* n);
		for(int i = 0;i < n;i++)
		{
			V[i] = (T*)malloc(sizeof(T)*n);
			memset(V[i],0,sizeof(T)*n);
		}

		T* S = (T*)malloc(sizeof(T)*sdim);
		memset(S,0,sizeof(T)*sdim);

		T* e = (T*)malloc(sizeof(T)*n);
		T* work = (T*)malloc(sizeof(T)*m);

		int wantu = 1;
		int wantv = 1;

		// Reduce A to bidiagonal form, storing the diagonal elements
		// in s and the super-diagonal elements in e.
		int nct = __min( m-1, n );
		int nrt = __max( 0, n-2 );

		for(int k = 0; k < __max(nct,nrt); k++ )
		{
			if( k < nct )
			{
				// Compute the transformation for the k-th column and
				// place the k-th diagonal in s[k].
				// Compute 2-norm of k-th column without under/overflow.
				S[k] = 0;
				for(int i = k; i < m; i++ )
					S[k] = hypot( S[k], A[i][k] );

				if( S[k] != 0 )
				{
					if( A[k][k] < 0 )
						S[k] = -S[k];

					for(int i = k; i < m; i++ )
						A[i][k] /= S[k];
					A[k][k] += 1;
				}
				S[k] = -S[k];
			}

			for(int j = k+1; j < n; j++ )
			{
				if( (k < nct) && ( S[k] != 0 ) )
				{
					// apply the transformation
					//double t = 0;
					T t = 0;
					for(int i = k; i < m; i++ )
						t += A[i][k] * A[i][j];

					t = -t / A[k][k];
					for(int i = k; i < m; i++ )
						A[i][j] += t*A[i][k];
				}
				e[j] = A[k][j];
			}

			// Place the transformation in U for subsequent back
			// multiplication.
			if( wantu & (k < nct) )
			{
				for(int i = k; i < m; i++ )
					U[i][k] = A[i][k];
			}

			if( k < nrt )
			{
				// Compute the k-th row transformation and place the
				// k-th super-diagonal in e[k].
				// Compute 2-norm without under/overflow.
				e[k] = 0;
				for(int i = k+1; i < n; i++ )
					e[k] = hypot( e[k], e[i] );

				if( e[k] != 0 )
				{
					if( e[k+1] < 0 )
						e[k] = -e[k];

					for(int i = k+1; i < n; i++ )
						e[i] /= e[k];
					e[k+1] += 1;
				}
				e[k] = -e[k];

				if( (k+1 < m) && ( e[k] != 0 ) )
				{
					// apply the transformation
					for(int i = k+1; i < m; i++ )
						work[i] = 0;

					for(int j = k+1; j < n; j++ )
					{
						for(int i = k+1; i < m; i++ )
							work[i] += e[j] * A[i][j];
					}

					for(int j = k+1; j < n; j++ )
					{
						//double t = -e[j]/e[k+1];
						T t = -e[j] / e[k + 1];
						for(int i = k+1; i < m; i++ )
							A[i][j] += t * work[i];
					}
				}

				// Place the transformation in V for subsequent
				// back multiplication.
				if( wantv )
					for(int i = k+1; i < n; i++ )
						V[i][k] = e[i];
			}
		}

		// Set up the final bidiagonal matrix or order p.
		int p = n;

		if( nct < n )
			S[nct] = A[nct][nct];
		if( m < p )
			S[p-1] = 0;

		if( nrt+1 < p )
			e[nrt] = A[nrt][p-1];
		e[p-1] = 0;

		// if required, generate U
		if( wantu )
		{
			for(int j = nct; j < n; j++ )
			{
				for(int i = 0; i < m; i++ )
					U[i][j] = 0;
				U[j][j] = 1;
			}

			for(int k = nct-1; k >= 0; k-- )
			{
				if( S[k] != 0 )
				{
					for(int j = k+1; j < n; j++ )
					{
						//double t = 0;
						T t = 0;
						for(int i = k; i < m; i++ )
							t += U[i][k] * U[i][j];
						t = -t / U[k][k];

						for(int i = k; i < m; i++ )
							U[i][j] += t * U[i][k];
					}

					for(int i = k; i < m; i++ )
						U[i][k] = -U[i][k];
					U[k][k] = 1 + U[k][k];

					for(int i = 0; i < k-1; i++ )
						U[i][k] = 0;
				}
				else
				{
					for(int i = 0; i < m; i++ )
						U[i][k] = 0;
					U[k][k] = 1;
				}
			}
		}

		// if required, generate V
		if( wantv )
		{
			for(int k = n-1; k >= 0; k-- )
			{
				if( (k < nrt) && ( e[k] != 0 ) )
					for(int j = k+1; j < n; j++ )
					{
						//double t = 0;
						T t = 0;
						for(int i = k+1; i < n; i++ )
							t += V[i][k] * V[i][j];
						t = -t / V[k+1][k];

						for(int i = k+1; i < n; i++ )
							V[i][j] += t * V[i][k];
					}

					for(int i = 0; i < n; ++i )
						V[i][k] = 0;
					V[k][k] = 1;
			}

			// main iteration loop for the singular values
			int pp = p-1;
			int iter = 0;
			//double eps = pow( 2.0, -52.0 );
			T eps = pow(2.0, -52.0);

			while( p > 0 )
			{
				int k = 0;
				int kase = 0;

				// Here is where a test for too many iterations would go.
				// This section of the program inspects for negligible
				// elements in the s and e arrays. On completion the
				// variables kase and k are set as follows.
				// kase = 1     if s(p) and e[k-1] are negligible and k<p
				// kase = 2     if s(k) is negligible and k<p
				// kase = 3     if e[k-1] is negligible, k<p, and
				//				s(k), ..., s(p) are not negligible
				// kase = 4     if e(p-1) is negligible (convergence).
				for(k = p-2; k >= -1; k-- )
				{
					if( k == -1 )
						break;

					if( fabs(e[k]) <= eps*( fabs(S[k])+fabs(S[k+1]) ) )
					{
						e[k] = 0;
						break;
					}
				}

				if( k == p-2 )
					kase = 4;
				else
				{
					int ks;
					for( ks=p-1; ks>=k; --ks )
					{
						if( ks == k )
							break;

						//double t = ( (ks != p) ? fabs(e[ks]) : 0 ) + ( (ks != k+1) ? fabs(e[ks-1]) : 0 );
						T t = ((ks != p) ? fabs(e[ks]) : 0) + ((ks != k + 1) ? fabs(e[ks - 1]) : 0);

						if( fabs(S[ks]) <= eps*t )
						{
							S[ks] = 0;
							break;
						}
					}

					if( ks == k )
						kase = 3;
					else if( ks == p-1 )
						kase = 1;
					else
					{
						kase = 2;
						k = ks;
					}
				}
				k++;

				// Perform the task indicated by kase.
				switch( kase )
				{
					// deflate negligible s(p)
				case 1:
					{
						//double f = e[p-2];
						T f = e[p - 2];
						e[p-2] = 0;

						for(int j = p-2; j >= k; j-- )
						{
							//double t = hypot( (double)S[j], f );
							//double cs = S[j] / t;
							//double sn = f / t;
							T t = hypot(S[j], f);
							T cs = S[j] / t;
							T sn = f / t;

							S[j] = t;

							if( j != k )
							{
								f = -sn * e[j-1];
								e[j-1] = cs * e[j-1];
							}

							if( wantv )
							{
								for(int i=0; i<n; ++i )
								{
									t = cs*V[i][j] + sn*V[i][p-1];
									V[i][p-1] = -sn*V[i][j] + cs*V[i][p-1];
									V[i][j] = t;
								}
							}
						}
					}
					break;

					// split at negligible s(k)
				case 2:
				{
						  //double f = e[k-1];
						  T f = e[k - 1];
						  e[k - 1] = 0;

						  for (int j = k; j < p; j++)
						  {
							  //double t = hypot((double)S[j], f);
							  //double cs = S[j] / t;
							  //double sn = f / t;
							  T t = hypot(S[j], f);
							  T cs = S[j] / t;
							  T sn = f / t;

							  S[j] = t;
							  f = -sn * e[j];
							  e[j] = cs * e[j];

							  if (wantu)
							  {
								  for (int i = 0; i < m; i++)
								  {
									  t = cs*U[i][j] + sn*U[i][k - 1];
									  U[i][k - 1] = -sn*U[i][j] + cs*U[i][k - 1];
									  U[i][j] = t;
								  }
							  }
						  }
					}
					break;

					// perform one qr step
				case 3:
					{
						// calculate the shift
						//double scale = __max( __max( __max( __max(fabs(S[p-1]), fabs(S[p-2]) ), fabs(e[p-2]) ),fabs(S[k]) ), fabs(e[k]) );
						//double sp = S[p-1] / scale;
						//double spm1 = S[p-2] / scale;
						//double epm1 = e[p-2] / scale;
						//double sk = S[k] / scale;
						//double ek = e[k] / scale;
						//double b = ( (spm1+sp)*(spm1-sp) + epm1*epm1 ) / 2.0;
						//double c = (sp*epm1) * (sp*epm1);
						//double shift = 0;
						T scale = __max(__max(__max(__max(fabs(S[p - 1]), fabs(S[p - 2])), fabs(e[p - 2])), fabs(S[k])), fabs(e[k]));
						T sp = S[p - 1] / scale;
						T spm1 = S[p - 2] / scale;
						T epm1 = e[p - 2] / scale;
						T sk = S[k] / scale;
						T ek = e[k] / scale;
						T b = ((spm1 + sp)*(spm1 - sp) + epm1*epm1) / 2.0;
						T c = (sp*epm1) * (sp*epm1);
						T shift = 0;

						if( ( b != 0 ) || ( c != 0 ) )
						{
							shift = sqrt( b*b+c );
							if( b < 0 )
								shift = -shift;
							shift = c / ( b+shift );
						}
						//double f = (sk+sp)*(sk-sp) + shift;
						//double g = sk * ek;
						T f = (sk+sp)*(sk-sp) + shift;
						T g = sk * ek;

						// chase zeros
						for(int j = k; j < p-1; j++ )
						{
							//double t = hypot( f, g );
							//double cs = f / t;
							//double sn = g / t;
							T t = hypot(f, g);
							T cs = f / t;
							T sn = g / t;

							if( j != k )
								e[j-1] = t;

							f = cs*S[j] + sn*e[j];
							e[j] = cs*e[j] - sn*S[j];
							g = sn * S[j+1];
							S[j+1] = cs * S[j+1];

							if( wantv )
							{
								for(int i = 0; i < n; i++ )
								{
									t = cs*V[i][j] + sn*V[i][j+1];
									V[i][j+1] = -sn*V[i][j] + cs*V[i][j+1];
									V[i][j] = t;
								}
							}

							t = hypot( f, g );
							cs = f / t;
							sn = g / t;
							S[j] = t;
							f = cs*e[j] + sn*S[j+1];
							S[j+1] = -sn*e[j] + cs*S[j+1];
							g = sn * e[j+1];
							e[j+1] = cs * e[j+1];

							if( wantu && ( j < m-1 ) )
							{
								for(int i = 0; i < m; i++ )
								{
									t = cs*U[i][j] + sn*U[i][j+1];
									U[i][j+1] = -sn*U[i][j] + cs*U[i][j+1];
									U[i][j] = t;
								}
							}
						}
						e[p-2] = f;
						iter = iter + 1;
					}
					break;

					// convergence
				case 4:
					{
						// Make the singular values positive.
						if( S[k] <= 0 )
						{
							S[k] = ( S[k] < 0 ) ? -S[k] : 0;
							if( wantv )
							{
								for(int i = 0; i <= pp; i++ )
									V[i][k] = -V[i][k];
							}
						}

						// Order the singular values.
						while( k < pp )
						{
							if( S[k] >= S[k+1] )
								break;

							//double t = S[k];
							T t = S[k];
							S[k] = S[k+1];
							S[k+1] = t;

							if( wantv && ( k < n-1 ) )
							{
								for(int i = 0; i < n; i++ )
									_swap( V[i][k], V[i][k+1] );
							}

							if( wantu && ( k < m-1 ) )
							{
								for(int i = 0; i < m; i++ )
									_swap( U[i][k], U[i][k+1] );
							}
							k++;
						}
						iter = 0;
						p--;
					}
					break;
				}
			}
		}

		//

		memset(Smat,0,sizeof(T)*sdim*sdim);
		for(int i = 0;i < sdim;i++)
			Smat[i*sdim+i] = S[i];


		if(!transpose_flag)
		{
			for(int i = 0;i < m;i++)
			{
				for(int j = 0;j < n;j++)
				{
					Umat[i*n+j] = U[i][j];
				}
			}

			for(int i = 0;i < n;i++)
			{
				for(int j = 0;j < n;j++)
				{
					Vmat[i*n+j] = V[i][j];
				}
			}
			
		}
		else
		{
			for(int i = 0;i < m;i++)
			{
				for(int j = 0;j < n;j++)
				{
					Vmat[i*n+j] = U[i][j];
				}
			}

			for(int i = 0;i < n;i++)
			{
				for(int j = 0;j < n;j++)
				{
					Umat[i*n+j] = V[i][j];
				}
			}
		}

		for(int i = 0;i < m;i++)
		{
			free(A[i]);
			free(U[i]);
		}
		for(int i = 0;i < n;i++)
		{
			free(V[i]);
		}

		free(e);
		free(work);
		free(U);
		free(V);
		free(S);
		free(A);
		return true;
	}

	template<class T>
	double ZQ_MathBase::Cond_by_double_svd(const T* mat, int row, int col, bool& succ, bool& is_singular)
	{
		is_singular = false;
		succ = false;
		
		if (_strcmpi(typeid(T).name(), "double") == 0)
		{
			double* U = (double*)malloc(sizeof(double)*row*row);
			double* S = (double*)malloc(sizeof(double)*row*col);
			double* V = (double*)malloc(sizeof(double)*col*col);
			bool svd_flag = SVD_Decompose((const double*)mat, row, col, U, S, V);
			
			free(U); U = 0;
			free(V); V = 0;
			
			if (!svd_flag)
			{
				free(S); S = 0;
				return 1e32;
			}
	
			succ = true;

			int N = __min(row, col);
			if (S[(N-1)*col + (N-1)] == 0)
			{
				is_singular = true;
				free(S); S = 0;
				return 1e32;
			}
			is_singular = false;
			double result = S[0] / S[(N-1)*col + (N-1)];
			free(S); S = 0;
			return result;
		}
		else
		{
			
			double* val = (double*)malloc(sizeof(double)*row*col);
			for (int i = 0; i < row*col; i++)
				val[i] = mat[i];

			double* U = (double*)malloc(sizeof(double)*row*row);
			double* S = (double*)malloc(sizeof(double)*row*col);
			double* V = (double*)malloc(sizeof(double)*col*col);
			bool svd_flag = SVD_Decompose(val, row, col, U, S, V);

			free(U); U = 0;
			free(V); V = 0;
			delete[]val; val = 0;
			if (!svd_flag)
			{
				free(S); S = 0;
				return 1e32;
			}
			succ = true;

			int N = __min(row, col);
			if (S[(N-1)*col + (N-1)] == 0)
			{
				is_singular = true;
				free(S); S = 0;
				return 1e32;
			}
			is_singular = false;
			double result = S[0] / S[(N-1)*col + (N-1)];
			free(S); S = 0;
			return result;
		}
	}

}



#endif