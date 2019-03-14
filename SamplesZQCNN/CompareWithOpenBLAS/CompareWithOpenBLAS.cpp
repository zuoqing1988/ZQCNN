#include <malloc.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include "math/zq_gemm_32f_align_c.h"
#include "ZQ_CNN_Tensor4D.h"
#include "ZQ_CNN_CompileConfig.h"
using namespace ZQ;


bool check_value(int M, int N, const float* C1, int ldc1, const float* C2, int ldc2, float thresh = 1e-5, bool show = false);
double _test_gemm_value();
double _test_gemv(int M, int N, int K, int iters = 1000);
float _test_im2col(int in_H, int in_W, int filter_N, int filter_C, int stride_H, int stride_W, int iters = 1000);
double _test_gemm(int M, int N, int K, int iters = 1000, float thresh = 1e-4, bool show = false);



#if defined(_WIN32)
#include <mmintrin.h> //MMX  
#include <xmmintrin.h> //SSE(include mmintrin.h)  
#include <emmintrin.h> //SSE2(include xmmintrin.h)  
#include <pmmintrin.h> //SSE3(include emmintrin.h)  
#include <tmmintrin.h>//SSSE3(include pmmintrin.h)  
#include <smmintrin.h>//SSE4.1(include tmmintrin.h)  
#include <nmmintrin.h>//SSE4.2(include smmintrin.h)  
#include <wmmintrin.h>//AES(include nmmintrin.h)  
#include <immintrin.h>//AVX(include wmmintrin.h)  
#include <intrin.h>//(include immintrin.h)  
#if ZQ_CNN_USE_BLAS_GEMM
#include <openblas/cblas.h>
#pragma comment(lib,"libopenblas.lib")
#elif ZQ_CNN_USE_MKL_GEMM
#include <mkl/mkl.h>
#pragma comment(lib,"mklml.lib")
#endif
#pragma comment(lib,"ZQ_GEMM.lib")

#if ZQ_CNN_USE_FMADD256
#define zq_mm_fmadd_ps _mm256_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#endif



double _test_gemm2(int M, int N, int K, int iters = 1000)
{
	int padK = (K + 7) >> 3 << 3;
	float* A = (float*)_aligned_malloc(M*padK * sizeof(float), 32);
	float* B = (float*)_aligned_malloc(padK*N * sizeof(float), 32);
	float* C = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* q1 = (float*)_aligned_malloc(32, 32);
	float* q2 = (float*)_aligned_malloc(32, 32);
	float* q3 = (float*)_aligned_malloc(32, 32);
	float* q4 = (float*)_aligned_malloc(32, 32);


	for (int i = 0; i < M*padK; i++)
		A[i] = rand() % 10001 / 5000.0f - 1.0f;
	for (int i = 0; i < padK*N; i++)
		B[i] = rand() % 10001 / 5000.0f - 1.0f;
	double t1 = omp_get_wtime(), t2, mul_count, gflops;
	double time1 = FLT_MAX;
	if (K % 64 == 0)
	{
		for (int it = 0; it < iters; it++)
		{
			float* Aptr = A;
			float* A_c_ptr;
			float* Cptr = C;
			for (int m = 0; m < M; m++, Aptr += K)
			{
				float* Bptr = B;
				float* B_c_ptr1;
				float* B_c_ptr2;
				float* B_c_ptr3;
				float* B_c_ptr4;
				int n, k;
				for (n = 0; n < N - 3; n += 4, Bptr += padK)
				{
					register __m256 sum_vec1 = _mm256_setzero_ps();
					register __m256 sum_vec2 = _mm256_setzero_ps();
					register __m256 sum_vec3 = _mm256_setzero_ps();
					register __m256 sum_vec4 = _mm256_setzero_ps();
					register __m256 a_vec;
					float* Bptr1 = Bptr;
					float* Bptr2 = Bptr1 + padK;
					float* Bptr3 = Bptr2 + padK;
					float* Bptr4 = Bptr3 + padK;
					for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
						k < padK;
						k += 64, A_c_ptr += 64, B_c_ptr1 += 64, B_c_ptr2 += 64, B_c_ptr3 += 64, B_c_ptr4 += 64)
					{
						a_vec = _mm256_load_ps(A_c_ptr);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1), sum_vec1);
						sum_vec2 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr2), sum_vec2);
						sum_vec3 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr3), sum_vec3);
						sum_vec4 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr4), sum_vec4);
						a_vec = _mm256_load_ps(A_c_ptr + 8);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 8), sum_vec1);
						sum_vec2 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr2 + 8), sum_vec2);
						sum_vec3 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr3 + 8), sum_vec3);
						sum_vec4 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr4 + 8), sum_vec4);
						a_vec = _mm256_load_ps(A_c_ptr + 16);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 16), sum_vec1);
						sum_vec2 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr2 + 16), sum_vec2);
						sum_vec3 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr3 + 16), sum_vec3);
						sum_vec4 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr4 + 16), sum_vec4);
						a_vec = _mm256_load_ps(A_c_ptr + 24);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 24), sum_vec1);
						sum_vec2 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr2 + 24), sum_vec2);
						sum_vec3 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr3 + 24), sum_vec3);
						sum_vec4 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr4 + 24), sum_vec4);
						a_vec = _mm256_load_ps(A_c_ptr + 32);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 32), sum_vec1);
						sum_vec2 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr2 + 32), sum_vec2);
						sum_vec3 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr3 + 32), sum_vec3);
						sum_vec4 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr4 + 32), sum_vec4);
						a_vec = _mm256_load_ps(A_c_ptr + 40);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 40), sum_vec1);
						sum_vec2 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr2 + 40), sum_vec2);
						sum_vec3 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr3 + 40), sum_vec3);
						sum_vec4 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr4 + 40), sum_vec4);
						a_vec = _mm256_load_ps(A_c_ptr + 48);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 48), sum_vec1);
						sum_vec2 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr2 + 48), sum_vec2);
						sum_vec3 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr3 + 48), sum_vec3);
						sum_vec4 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr4 + 48), sum_vec4);
						a_vec = _mm256_load_ps(A_c_ptr + 56);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 56), sum_vec1);
						sum_vec2 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr2 + 56), sum_vec2);
						sum_vec3 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr3 + 56), sum_vec3);
						sum_vec4 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr4 + 56), sum_vec4);
					}
					_mm256_store_ps(q1, sum_vec1);
					_mm256_store_ps(q2, sum_vec2);
					_mm256_store_ps(q3, sum_vec3);
					_mm256_store_ps(q4, sum_vec4);
					_mm_store_ps(q1, _mm_add_ps(_mm_load_ps(q1), _mm_load_ps(q1 + 4)));
					_mm_store_ps(q2, _mm_add_ps(_mm_load_ps(q2), _mm_load_ps(q2 + 4)));
					_mm_store_ps(q3, _mm_add_ps(_mm_load_ps(q3), _mm_load_ps(q3 + 4)));
					_mm_store_ps(q4, _mm_add_ps(_mm_load_ps(q4), _mm_load_ps(q4 + 4)));
					*(Cptr++) = q1[0] + q1[1] + q1[2] + q1[3];
					*(Cptr++) = q2[0] + q2[1] + q2[2] + q2[3];
					*(Cptr++) = q3[0] + q3[1] + q3[2] + q3[3];
					*(Cptr++) = q4[0] + q4[1] + q4[2] + q4[3];
				}
				for (; n < N; n++, Bptr += padK)
				{
					register __m256 sum_vec1 = _mm256_setzero_ps();
					register __m256 a_vec;
					float* Bptr1 = Bptr;
					for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1;
						k < padK;
						k += 64, A_c_ptr += 64, B_c_ptr1 += 64)
					{
						a_vec = _mm256_load_ps(A_c_ptr);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1), sum_vec1);
						a_vec = _mm256_load_ps(A_c_ptr + 8);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 8), sum_vec1);
						a_vec = _mm256_load_ps(A_c_ptr + 16);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 16), sum_vec1);
						a_vec = _mm256_load_ps(A_c_ptr + 24);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 24), sum_vec1);
						a_vec = _mm256_load_ps(A_c_ptr + 32);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 32), sum_vec1);
						a_vec = _mm256_load_ps(A_c_ptr + 40);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 40), sum_vec1);
						a_vec = _mm256_load_ps(A_c_ptr + 48);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 48), sum_vec1);
						a_vec = _mm256_load_ps(A_c_ptr + 56);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 56), sum_vec1);
					}
					_mm256_store_ps(q1, sum_vec1);
					_mm_store_ps(q1, _mm_add_ps(_mm_load_ps(q1), _mm_load_ps(q1 + 4)));
					*(Cptr++) = q1[0] + q1[1] + q1[2] + q1[3];
				}
			}

		}

		t2 = omp_get_wtime();
		time1 = t2 - t1;
		mul_count = (double)M*N*K*iters;
		gflops = mul_count / (1 << 30) / (t2 - t1);
		printf("C[0] = %f\n", C[0]);
		printf("%d x %d x %d * %d = %.3e, time = %.3f s, my gflops = %.3f\n", M, N, K, iters, mul_count, time1, gflops);

	}
	else if (K % 32 == 0)
	{
		for (int it = 0; it < iters; it++)
		{
			float* Aptr = A;
			float* A_c_ptr;
			float* Cptr = C;
			for (int m = 0; m < M; m++, Aptr += padK)
			{
				float* Bptr = B;
				float* B_c_ptr1;
				float* B_c_ptr2;
				float* B_c_ptr3;
				float* B_c_ptr4;
				int n, k;
				for (n = 0; n < N - 3; n += 4, Bptr += padK)
				{
					register __m256 sum_vec1 = _mm256_setzero_ps();
					register __m256 sum_vec2 = _mm256_setzero_ps();
					register __m256 sum_vec3 = _mm256_setzero_ps();
					register __m256 sum_vec4 = _mm256_setzero_ps();
					register __m256 a_vec;
					float* Bptr1 = Bptr;
					float* Bptr2 = Bptr1 + padK;
					float* Bptr3 = Bptr2 + padK;
					float* Bptr4 = Bptr3 + padK;
					for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
						k < padK;
						k += 32, A_c_ptr += 32, B_c_ptr1 += 32, B_c_ptr2 += 32, B_c_ptr3 += 32, B_c_ptr4 += 32)
					{
						a_vec = _mm256_load_ps(A_c_ptr);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1), sum_vec1);
						sum_vec2 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr2), sum_vec2);
						sum_vec3 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr3), sum_vec3);
						sum_vec4 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr4), sum_vec4);
						a_vec = _mm256_load_ps(A_c_ptr + 8);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 8), sum_vec1);
						sum_vec2 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr2 + 8), sum_vec2);
						sum_vec3 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr3 + 8), sum_vec3);
						sum_vec4 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr4 + 8), sum_vec4);
						a_vec = _mm256_load_ps(A_c_ptr + 16);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 16), sum_vec1);
						sum_vec2 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr2 + 16), sum_vec2);
						sum_vec3 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr3 + 16), sum_vec3);
						sum_vec4 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr4 + 16), sum_vec4);
						a_vec = _mm256_load_ps(A_c_ptr + 24);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 24), sum_vec1);
						sum_vec2 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr2 + 24), sum_vec2);
						sum_vec3 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr3 + 24), sum_vec3);
						sum_vec4 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr4 + 24), sum_vec4);
					}
					_mm256_store_ps(q1, sum_vec1);
					_mm256_store_ps(q2, sum_vec2);
					_mm256_store_ps(q3, sum_vec3);
					_mm256_store_ps(q4, sum_vec4);
					_mm_store_ps(q1, _mm_add_ps(_mm_load_ps(q1), _mm_load_ps(q1 + 4)));
					_mm_store_ps(q2, _mm_add_ps(_mm_load_ps(q2), _mm_load_ps(q2 + 4)));
					_mm_store_ps(q3, _mm_add_ps(_mm_load_ps(q3), _mm_load_ps(q3 + 4)));
					_mm_store_ps(q4, _mm_add_ps(_mm_load_ps(q4), _mm_load_ps(q4 + 4)));
					*(Cptr++) = q1[0] + q1[1] + q1[2] + q1[3];
					*(Cptr++) = q2[0] + q2[1] + q2[2] + q2[3];
					*(Cptr++) = q3[0] + q3[1] + q3[2] + q3[3];
					*(Cptr++) = q4[0] + q4[1] + q4[2] + q4[3];
				}
				for (; n < N; n++, Bptr += padK)
				{
					register __m256 sum_vec1 = _mm256_setzero_ps();
					register __m256 a_vec;
					float* Bptr1 = Bptr;
					for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1;
						k < K;
						k += 32, A_c_ptr += 32, B_c_ptr1 += 32)
					{
						a_vec = _mm256_load_ps(A_c_ptr);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1), sum_vec1);
						a_vec = _mm256_load_ps(A_c_ptr + 8);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 8), sum_vec1);
						a_vec = _mm256_load_ps(A_c_ptr + 16);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 16), sum_vec1);
						a_vec = _mm256_load_ps(A_c_ptr + 24);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 24), sum_vec1);
					}
					_mm256_store_ps(q1, sum_vec1);
					_mm_store_ps(q1, _mm_add_ps(_mm_load_ps(q1), _mm_load_ps(q1 + 4)));
					*(Cptr++) = q1[0] + q1[1] + q1[2] + q1[3];
				}
			}

		}

		t2 = omp_get_wtime();
		time1 = t2 - t1;
		mul_count = (double)M*N*K*iters;
		gflops = mul_count / (1 << 30) / (t2 - t1);
		printf("C[0] = %f\n", C[0]);
		printf("%d x %d x %d * %d = %.3e, time = %.3f s, my gflops = %.3f\n", M, N, K, iters, mul_count, time1, gflops);
	}
	else
	{
		for (int it = 0; it < iters; it++)
		{
			float* Aptr = A;
			float* A_c_ptr;
			float* Cptr = C;
			for (int m = 0; m < M; m++, Aptr += padK)
			{
				float* Bptr = B;
				float* B_c_ptr1;
				float* B_c_ptr2;
				float* B_c_ptr3;
				float* B_c_ptr4;
				int n, k;
				for (n = 0; n < N - 3; n += 4, Bptr += padK)
				{
					register __m256 sum_vec1 = _mm256_setzero_ps();
					register __m256 sum_vec2 = _mm256_setzero_ps();
					register __m256 sum_vec3 = _mm256_setzero_ps();
					register __m256 sum_vec4 = _mm256_setzero_ps();
					register __m256 a_vec;
					float* Bptr1 = Bptr;
					float* Bptr2 = Bptr1 + padK;
					float* Bptr3 = Bptr2 + padK;
					float* Bptr4 = Bptr3 + padK;
					for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
						k < padK - 31;
						k += 32, A_c_ptr += 32, B_c_ptr1 += 32, B_c_ptr2 += 32, B_c_ptr3 += 32, B_c_ptr4 += 32)
					{
						a_vec = _mm256_load_ps(A_c_ptr);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1), sum_vec1);
						sum_vec2 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr2), sum_vec2);
						sum_vec3 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr3), sum_vec3);
						sum_vec4 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr4), sum_vec4);
						a_vec = _mm256_load_ps(A_c_ptr + 8);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 8), sum_vec1);
						sum_vec2 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr2 + 8), sum_vec2);
						sum_vec3 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr3 + 8), sum_vec3);
						sum_vec4 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr4 + 8), sum_vec4);
						a_vec = _mm256_load_ps(A_c_ptr + 16);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 16), sum_vec1);
						sum_vec2 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr2 + 16), sum_vec2);
						sum_vec3 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr3 + 16), sum_vec3);
						sum_vec4 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr4 + 16), sum_vec4);
						a_vec = _mm256_load_ps(A_c_ptr + 24);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 24), sum_vec1);
						sum_vec2 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr2 + 24), sum_vec2);
						sum_vec3 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr3 + 24), sum_vec3);
						sum_vec4 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr4 + 24), sum_vec4);
					}
					for (; k < padK;
						k += 8, A_c_ptr += 8, B_c_ptr1 += 8, B_c_ptr2 += 8, B_c_ptr3 += 8, B_c_ptr4 += 8)
					{
						a_vec = _mm256_load_ps(A_c_ptr);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1), sum_vec1);
						sum_vec2 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr2), sum_vec2);
						sum_vec3 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr3), sum_vec3);
						sum_vec4 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr4), sum_vec4);
					}
					_mm256_store_ps(q1, sum_vec1);
					_mm256_store_ps(q2, sum_vec2);
					_mm256_store_ps(q3, sum_vec3);
					_mm256_store_ps(q4, sum_vec4);
					_mm_store_ps(q1, _mm_add_ps(_mm_load_ps(q1), _mm_load_ps(q1 + 4)));
					_mm_store_ps(q2, _mm_add_ps(_mm_load_ps(q2), _mm_load_ps(q2 + 4)));
					_mm_store_ps(q3, _mm_add_ps(_mm_load_ps(q3), _mm_load_ps(q3 + 4)));
					_mm_store_ps(q4, _mm_add_ps(_mm_load_ps(q4), _mm_load_ps(q4 + 4)));
					*(Cptr++) = q1[0] + q1[1] + q1[2] + q1[3];
					*(Cptr++) = q2[0] + q2[1] + q2[2] + q2[3];
					*(Cptr++) = q3[0] + q3[1] + q3[2] + q3[3];
					*(Cptr++) = q4[0] + q4[1] + q4[2] + q4[3];
				}
				for (; n < N; n++, Bptr += padK)
				{
					register __m256 sum_vec1 = _mm256_setzero_ps();
					register __m256 a_vec;
					float* Bptr1 = Bptr;
					for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1;
						k < padK - 31;
						k += 32, A_c_ptr += 32, B_c_ptr1 += 32)
					{
						a_vec = _mm256_load_ps(A_c_ptr);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1), sum_vec1);
						a_vec = _mm256_load_ps(A_c_ptr + 8);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 8), sum_vec1);
						a_vec = _mm256_load_ps(A_c_ptr + 16);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 16), sum_vec1);
						a_vec = _mm256_load_ps(A_c_ptr + 24);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1 + 24), sum_vec1);
					}
					for (; k < padK;
						k += 8, A_c_ptr += 8, B_c_ptr1 += 8)
					{
						a_vec = _mm256_load_ps(A_c_ptr);
						sum_vec1 = zq_mm_fmadd_ps(a_vec, _mm256_load_ps(B_c_ptr1), sum_vec1);
					}
					_mm256_store_ps(q1, sum_vec1);
					_mm_store_ps(q1, _mm_add_ps(_mm_load_ps(q1), _mm_load_ps(q1 + 4)));
					*(Cptr++) = q1[0] + q1[1] + q1[2] + q1[3];
				}
			}

		}

		t2 = omp_get_wtime();
		time1 = t2 - t1;
		mul_count = (double)M*N*K*iters;
		gflops = mul_count / (1 << 30) / (t2 - t1);
		printf("C[0] = %f\n", C[0]);
		printf("%d x %d x %d * %d = %.3e, time = %.3f s, my gflops = %.3f\n", M, N, K, iters, mul_count, time1, gflops);
	}



	t1 = omp_get_wtime();
	for (int i = 0; i < iters; i++)
	{
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, padK, 1.0, A, padK, B, padK, 0, C, N);

	}
	printf("C[0] = %f\n", C[0]);
	t2 = omp_get_wtime();
	mul_count = (double)M*N*K*iters;
	gflops = mul_count / (1 << 30) / (t2 - t1);
	double  time2 = t2 - t1;
	printf("%d x %d x %d * %d = %.3e, time = %.3f s, gemm gflops = %.3f\n", M, N, K, iters, mul_count, time2, gflops);

	_aligned_free(A);
	_aligned_free(B);
	_aligned_free(C);
	_aligned_free(q1);
	_aligned_free(q2);
	_aligned_free(q3);
	_aligned_free(q4);


	return __min(time1, time2) / iters;
}


int main()
{
	int num_threads = 1;
#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(num_threads);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(num_threads);
#endif
	_test_gemm(56 * 56, 64, 32, 10000);
	_test_gemm(56 * 56, 128, 64, 10000);
	_test_gemm(56 * 56, 64, 128, 10000);
	_test_gemm(28 * 28, 256, 128, 10000);
	_test_gemm(28 * 28, 128, 256, 10000);
	_test_gemm(14 * 14, 256, 512, 10000);
	_test_gemm(14 * 14, 512, 256, 10000);
	_test_gemm(7 * 7, 256, 512, 10000);
	_test_gemm(7 * 7, 512, 256, 10000);
	
	_test_gemm(128, 128, 128, 80000);
	_test_gemm(128, 128, 128, 80000);
	_test_gemm(128, 128, 128, 80000);
	_test_gemm(256, 256, 256, 10000);
	_test_gemm(256, 256, 256, 10000);
	_test_gemm(256, 256, 256, 10000);
	_test_gemm(512, 512, 512, 20000);
	_test_gemm(512, 512, 512, 20000);
	_test_gemm(512, 512, 512, 20000);
	_test_gemm(1024, 1024, 1024, 40000);
	_test_gemm(1024, 1024, 1024, 40000);
	_test_gemm(1024, 1024, 1024, 40000);
	/*double total_sum = 0;
	_test_gemm_value();
	total_sum += 1*_test_im2col(112, 96, 64, 3, 2, 2,1000);
	total_sum += 2 * _test_im2col(56, 48, 64, 64, 1, 1,1000);
	total_sum+=1*_test_im2col(56, 48, 128, 64, 2, 2,1000);
	total_sum+=4*_test_im2col(28, 24, 128, 128, 1, 1,1000);
	total_sum += 1*_test_im2col(28, 24, 256, 128, 2, 2,1000);
	total_sum += 8*_test_im2col(14, 12, 256, 256, 1, 1,1000);
	total_sum += 1 * _test_im2col(14, 12, 512, 256, 2, 2,1000);
	total_sum += 2 * _test_im2col(7, 6, 512, 512, 1, 1,1000);
	total_sum += _test_gemm(1 * 1, 512, 7 * 6 * 512,1000);
	printf("total: %.3f\n", total_sum);*/

	for (int i = 0; i < 100000; i++)
	{
		int M = rand()%1000 + 1;
		int N = rand() % 1000 + 1;
		int K = rand() % 1000 + 1;
		_test_gemm(M, N, K, 1, 1e-4, true);
		//_test_gemm(43, 604, 64, 1, 1e-4, true);
	}
	return 0;
	_test_gemm(56 * 48, 64, 3 * 3 * 3,10000);
	_test_gemm(56 * 48, 64, 3 * 3 * 4,10000);
	_test_gemm(28 * 24, 128, 3 * 3 * 64,10000);
	_test_gemm(14 * 12, 256, 3 * 3 * 128,10000);
	_test_gemm(7 * 6, 512, 3 * 3 * 256,10000);
	_test_gemm(1 * 1, 512, 7 * 6 * 512,10000);
	return 0;
	//compare gemv
	//_test_gemv(56 * 48, 64, 3 * 3 * 3,1000);
	//_test_gemv(56 * 48, 64, 3 * 3 * 4, 1000);
	//_test_gemv(28 * 24, 128, 3 * 3 * 64,1000);
	//_test_gemv(14 * 12, 256, 3 * 3 * 128,1000);
	//_test_gemv(7 * 6, 512, 3 * 3 * 256,1000);
	//_test_gemv(1 * 1, 512, 7 * 6 * 512,1000);

	//compare  MTCNN Pnet
	_test_gemm(214 * 382, 10, 3 * 3 * 3, 1000);
	_test_gemm(214 * 382, 10, 3 * 3 * 4, 1000);
	_test_gemm(105 * 189, 16, 3 * 3 * 10, 1000);
	_test_gemm(105 * 189, 16, 3 * 3 * 16, 1000);
	_test_gemm(103 * 187, 32, 3 * 3 * 16, 1000);
	_test_gemm(103 * 187, 2, 1 * 1 * 32, 1000);
	_test_gemv(103 * 187, 2, 1 * 1 * 32, 1000);
	_test_gemm(103 * 187, 4, 1 * 1 * 32, 1000);
	_test_gemv(103 * 187, 4, 1 * 1 * 32, 1000);
	

	//other tests
	_test_gemm(92, 128, 3 * 3 * 64);
	_test_gemm(256,  2, 6 * 7 * 512);
	_test_gemm(256, 4, 6 * 7 * 512);
	_test_gemm(256, 10, 6 * 7 * 512);
	_test_gemm(1, 1024 * 512, 3 * 3 * 64);
	_test_gemm(56*48, 64, 3*3*64);
	_test_gemm(56 * 48+1, 64+1, 3 * 3 * 64+1);
	_test_gemm(28*24, 128, 3 * 3 * 128);
	_test_gemm(28 * 24+1, 128+1, 3 * 3 * 128+1);
	_test_gemm(14 * 12, 256, 3 * 3 * 256);
	_test_gemm(14 * 12+1, 256+1, 3 * 3 * 256+1);
	_test_gemm(7 * 6, 512, 3 * 3 * 512);
	_test_gemm(7 * 6+1, 512+1, 3 * 3 * 512+1);
	_test_gemm(512, 512, 3 * 3 * 512);
	_test_gemm(513, 512 + 1, 3 * 3 * 512 + 1);
	_test_gemm(1024, 1024, 1024);
	_test_gemm(1025, 1025, 1025);

	_test_gemm(1, 1024, 1024, 10000);
	_test_gemm2(1, 1024, 1024, 10000);
	_test_gemm(320 * 240, 16, 64, 1000);
	_test_gemm2(320 * 240, 16, 64, 1000);
	_test_gemm(105 * 189, 8, 3 * 3 * 16, 1000);
	_test_gemm2(105 * 189, 8, 3 * 3 * 16, 1000);
	_test_gemm(105 * 189, 16, 3 * 3 * 24, 1000);
	_test_gemm2(105 * 189, 16, 3 * 3 * 24, 1000);
	_test_gemm(105 * 189, 16, 3 * 3 * 32, 1000);
	_test_gemm2(105 * 189, 16, 3 * 3 * 32, 1000);
	_test_gemm(105 * 189, 24, 3 * 3 * 32, 1000);
	_test_gemm2(105 * 189, 24, 3 * 3 * 32, 1000);
	_test_gemm(105 * 189, 24, 3 * 3 * 48, 1000);
	_test_gemm2(105 * 189, 24, 3 * 3 * 48, 1000);
	_test_gemm(105 * 189, 32, 3 * 3 * 48, 1000);
	_test_gemm2(105 * 189, 32, 3 * 3 * 48, 1000);
	_test_gemm(105 * 189, 32, 3 * 3 * 64, 1000);
	_test_gemm2(105 * 189, 32, 3 * 3 * 64, 1000);
	return EXIT_SUCCESS;
}

#else
#include "ZQ_CNN_CompileConfig.h"
#if ZQ_CNN_USE_BLAS_GEMM
#include <openblas/cblas.h>
#include <sched.h>

int main(int argc, const char** argv)
{
	if (argc != 1)
	{
		cpu_set_t mask;
		CPU_ZERO(&mask);
		CPU_SET(atoi(argv[1]), &mask);
		if (sched_setaffinity(0, sizeof(mask), &mask) < 0) {
			perror("sched_setaffinity");
		}
	}

	int num_threads = 1;
#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(num_threads);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(num_threads);
#endif
	/*double total_sum = 0;
	_test_gemm_value();
	total_sum += 1*_test_im2col(112, 96, 64, 3, 2, 2,1000);
	total_sum += 2 * _test_im2col(56, 48, 64, 64, 1, 1,1000);
	total_sum+=1*_test_im2col(56, 48, 128, 64, 2, 2,1000);
	total_sum+=4*_test_im2col(28, 24, 128, 128, 1, 1,1000);
	total_sum += 1*_test_im2col(28, 24, 256, 128, 2, 2,1000);
	total_sum += 8*_test_im2col(14, 12, 256, 256, 1, 1,1000);
	total_sum += 1 * _test_im2col(14, 12, 512, 256, 2, 2,1000);
	total_sum += 2 * _test_im2col(7, 6, 512, 512, 1, 1,1000);
	total_sum += _test_gemm(1 * 1, 512, 7 * 6 * 512,1000);
	printf("total: %.3f\n", total_sum);*/

	/*for (int i = 0; i < 100000; i++)
	{
		int M = rand() % 1000 + 1;
		int N = rand() % 1000 + 1;
		int K = rand() % 1000 + 1;
		_test_gemm(M, N, K, 1, 1e-4, true);
	}
	return 0;*/

	printf("test det1-dw20-fast\n");
	_test_gemm(47 * 63, 8, 28, 20000);
	_test_gemm(47 * 63, 8, 28, 20000);
	_test_gemm(45 * 61, 16, 8, 50000);
	_test_gemm(22 * 30, 24, 16, 50000);
	_test_gemm(20 * 28, 1, 24, 50000);
	_test_gemm(20 * 28, 4, 24, 50000);
	printf("test det2-dw24-fast\n");
	for (int i = 1; i <= 64; i++)
	{
		printf("batchsize= %d\n", i);
		_test_gemm(22 * 22*i, 16, 28, 50000/i);
		_test_gemm(22 * 22*i, 16, 28, 50000/i);
		_test_gemm(11 * 11*i, 32, 16, 50000/i);
		_test_gemm(5 * 5*i, 64, 32, 50000/i);
		_test_gemm(3 * 3*i, 128, 64, 20000/i);
		_test_gemm(1*i, 2, 128, 10000/i);
		_test_gemm(1*i, 4, 128, 10000/i);
	}
	
	printf("test det3-dw48-fast\n");
	for (int i = 1; i <= 64; i++)
	{
		printf("batchsize= %d\n", i);
		_test_gemm(48 * 48*i, 16, 28, 10000/i);
		_test_gemm(48 * 48*i, 16, 28, 10000/i);
		_test_gemm(24 * 24*i, 32, 16, 10000/i);
		_test_gemm(12 * 12*i, 64, 32, 10000/i);
		_test_gemm(5 * 5*i, 64, 64, 10000/i);
		_test_gemm(3 * 3*i, 128, 64, 10000/i);
		_test_gemm(1*i, 2, 128, 10000/i);
		_test_gemm(1*i, 4, 128, 10000/i);
	}
	
	printf("test det5-dw96-v2s\n");
	for (int i = 1; i <= 64; i++)
	{
		printf("batchsize= %d\n", i);
		_test_gemm(94 * 94*i, 32, 28, 1000/i);
		_test_gemm(93 * 93*i, 32, 32, 1000/i);
		_test_gemm(46 * 46*i, 64, 32, 1000/i);
		_test_gemm(45 * 45*i, 64, 64, 1000/i);
		_test_gemm(22 * 22*i, 64, 64, 2000/i);
		_test_gemm(21 * 21*i, 64, 64, 2000/i);
		_test_gemm(10 * 10*i, 128, 64, 5000/i);
		_test_gemm(9 * 9*i, 128, 128, 5000/i);
		_test_gemm(4 * 4*i, 256, 128, 10000/i);
		_test_gemm(3 * 3*i, 256, 256, 10000/i);
		_test_gemm(1 * 1*i, 212, 256, 10000/i);
	}
	
	printf("test mobilefacenet\n");
	_test_gemm(56 * 56, 64, 28, 1000);
	_test_gemm(56 * 56, 64, 28, 1000);
	_test_gemm(56 * 56, 128, 64, 1000);
	_test_gemm(28 * 28, 64, 128, 1000);
	_test_gemm(28 * 28, 128, 64, 1000);
	_test_gemm(28 * 28, 256, 64, 1000);
	_test_gemm(14 * 14, 128, 256, 1000);
	_test_gemm(14 * 14, 256, 128, 1000);
	_test_gemm(14 * 14, 512, 128, 1000);
	_test_gemm(7 * 7, 256, 128, 1000);
	_test_gemm(7 * 7, 128, 256, 1000);
	_test_gemm(1 * 1, 128, 512, 1000);
	_test_gemm(1 * 1, 256, 512, 1000);
	_test_gemm(1 * 1, 512, 512, 1000);
	printf("test mobilefacenet-res2-6-10-2\n");
	_test_gemm(56 * 56, 64, 28, 1000);
	_test_gemm(56 * 56, 64, 28, 1000);
	_test_gemm(56 * 56, 64, 64, 1000);
	_test_gemm(56 * 56, 128, 64, 1000);
	_test_gemm(28 * 28, 128, 128, 1000);
	_test_gemm(28 * 28, 256, 128, 1000);
	_test_gemm(14 * 14, 256, 256, 1000);
	_test_gemm(14 * 14, 512, 256, 1000);
	_test_gemm(7 * 7, 512, 512, 1000);
	_test_gemm(1 * 1, 128, 512, 1000);
	_test_gemm(1 * 1, 256, 512, 1000);
	_test_gemm(1 * 1, 512, 512, 1000);

	//return 0;
	//compare gemv
	//_test_gemv(56 * 48, 64, 3 * 3 * 3,1000);
	//_test_gemv(56 * 48, 64, 3 * 3 * 4, 1000);
	//_test_gemv(28 * 24, 128, 3 * 3 * 64,1000);
	//_test_gemv(14 * 12, 256, 3 * 3 * 128,1000);
	//_test_gemv(7 * 6, 512, 3 * 3 * 256,1000);
	//_test_gemv(1 * 1, 512, 7 * 6 * 512,1000);

	//compare  MTCNN Pnet
	_test_gemm(214 * 382, 10, 3 * 3 * 3, 1000);
	_test_gemm(214 * 382, 10, 3 * 3 * 4, 1000);
	_test_gemm(105 * 189, 16, 3 * 3 * 10, 1000);
	_test_gemm(105 * 189, 16, 3 * 3 * 16, 1000);
	_test_gemm(103 * 187, 32, 3 * 3 * 16, 1000);
	_test_gemm(103 * 187, 2, 1 * 1 * 32, 1000);
	_test_gemv(103 * 187, 2, 1 * 1 * 32, 1000);
	_test_gemm(103 * 187, 4, 1 * 1 * 32, 1000);
	_test_gemv(103 * 187, 4, 1 * 1 * 32, 1000);


	//other tests
	_test_gemm(92, 128, 3 * 3 * 64,100);
	_test_gemm(256, 2, 6 * 7 * 512,100);
	_test_gemm(256, 4, 6 * 7 * 512,100);
	_test_gemm(256, 10, 6 * 7 * 512,100);
	_test_gemm(1, 1024 * 512, 3 * 3 * 64,100);
	_test_gemm(56 * 48, 64, 3 * 3 * 64,100);
	_test_gemm(56 * 48 + 1, 64 + 1, 3 * 3 * 64 + 1,100);
	_test_gemm(28 * 24, 128, 3 * 3 * 128,100);
	_test_gemm(28 * 24 + 1, 128 + 1, 3 * 3 * 128 + 1,100);
	_test_gemm(14 * 12, 256, 3 * 3 * 256,100);
	_test_gemm(14 * 12 + 1, 256 + 1, 3 * 3 * 256 + 1,100);
	_test_gemm(7 * 6, 512, 3 * 3 * 512,100);
	_test_gemm(7 * 6 + 1, 512 + 1, 3 * 3 * 512 + 1,100);
	_test_gemm(512, 512, 3 * 3 * 512,100);
	_test_gemm(513, 512 + 1, 3 * 3 * 512 + 1,100);
	_test_gemm(1024, 1024, 1024,10);
	_test_gemm(1025, 1025, 1025,10);

	_test_gemm(1, 1024, 1024, 1000);
	_test_gemm(320 * 240, 16, 64, 100);
	_test_gemm(105 * 189, 8, 3 * 3 * 16, 100);
	_test_gemm(105 * 189, 16, 3 * 3 * 24, 100);
	_test_gemm(105 * 189, 16, 3 * 3 * 32, 100);
	_test_gemm(105 * 189, 24, 3 * 3 * 32, 100);
	_test_gemm(105 * 189, 24, 3 * 3 * 48, 100);
	_test_gemm(105 * 189, 32, 3 * 3 * 48, 100);
	_test_gemm(105 * 189, 32, 3 * 3 * 64, 100);
	return 0;
}

#else
int main(int argc, const char** argv)
{
	printf("%s only support openblas\n", argv[0]);
	return 0;
}
#endif
#endif

bool check_value(int M, int N, const float* C1, int ldc1, const float* C2, int ldc2, float thresh, bool show)
{
	int m, n;
	const float* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	float v1, v2;
	bool ret = true;
	for (m = 0, Cptr1 = C1, Cptr2 = C2; m < M; m++, Cptr1 += ldc1, Cptr2 += ldc2)
	{
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n++)
		{
			v1 = *C_c_ptr1;
			v2 = *C_c_ptr2;
			float scale = __max(fabs(v1), fabs(v2));
			float real_thresh = __max(thresh, thresh*scale);
			if (fabs(v1 - v2) > real_thresh)
			{
				if (show)
					printf("%d,%d = %f %f\n", m, n, v1, v2);
				ret = false;
			}
			C_c_ptr1++;
			C_c_ptr2++;
		}
	}
	return ret;
}

double _test_gemm_value()
{
	float A[] = {
		1,2,3,
		4,5,6
	};
	float BT[] = {
		2, 1,3,
		4,5,6
	};
	float C[4];
	for (int it = 0; it < 10; it++)
	{
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 2, 2, 3, 1, A, 3, BT, 3, 0, C, 2);
		printf("%f %f %f %f\n", C[0], C[1], C[2], C[3]);
	}
	return 0;

}
double _test_gemv(int M, int N, int K, int iters)
{

	int padK = (K + 7) >> 3 << 3;
	/*N = (N + 7) >> 3 << 3;*/
	float* A = (float*)_aligned_malloc(M* padK * sizeof(float), 32);
	float* B = (float*)_aligned_malloc(padK*N * sizeof(float), 32);
	float* C = (float*)_aligned_malloc(M* N * sizeof(float), 32);
	float* q = (float*)_aligned_malloc(32, 32);


	for (int i = 0; i < M*padK; i++)
	{
		//A[i] = rand() % 10001 / 5000.0f - 1.0f;
		A[i] = i;
	}
	for (int i = 0; i < padK*N; i++)
	{
		B[i] = rand() % 10001 / 5000.0f - 1.0f;
		B[i] = i;
	}
	double t1 = omp_get_wtime(), t2, mul_count, gflops;

	t1 = omp_get_wtime();
	for (int i = 0; i < iters; i++)
	{
		for (int m = 0; m < M; m++)
			cblas_sgemv(CblasRowMajor, CblasNoTrans, N, K, 1.0, B, padK, A + m*padK, 1, 0, C + m*N, 1);
	}
	//printf("C[0] = %f\n", C[0]);
	t2 = omp_get_wtime();
	mul_count = (double)M*N*K*iters;
	gflops = mul_count / (1 << 30) / (t2 - t1);

	printf("%d x %d x %d * %d = %.3e, time = %.3f s, gemv gflops = %.3f\n", M, N, K, iters, mul_count, t2 - t1, gflops);

	_aligned_free(A);
	_aligned_free(B);
	_aligned_free(C);
	_aligned_free(q);
	return (t2 - t1) / iters;
}



float _test_im2col(int in_H, int in_W, int filter_N, int filter_C, int stride_H, int stride_W, int iters)
{
	ZQ_CNN_Tensor4D_NHW_C_Align256bit input, filters, output;
	int in_N = 1, in_C = filter_C;
	int filter_H = 3, filter_W = 3;
	int pad_H = 1, pad_W = 1;
	int out_N = in_N;
	int out_H = (in_H - filter_H + pad_H * 2) / stride_H + 1;
	int out_W = (in_W - filter_W + pad_W * 2) / stride_W + 1;
	int out_C = filter_N;

	input.ChangeSize(in_N, in_H, in_W, in_C, pad_H, pad_W);
	filters.ChangeSize(filter_N, filter_H, filter_W, filter_C, 0, 0);
	output.ChangeSize(out_N, out_H, out_W, out_C, 0, 0);


	double t1 = omp_get_wtime();
	float* matrix_C = (float*)malloc(out_H*out_W*filter_N * sizeof(float));
	for (int it = 0; it < iters; it++)
	{
		int in_widthStep = input.GetWidthStep();
		int in_widthStep_mul_stride_H = in_widthStep*stride_H;
		int in_pixelStep = input.GetPixelStep();
		int in_pixelStep_mul_stride_W = in_pixelStep*stride_W;
		int filter_pixStep = filters.GetPixelStep();
		int filter_pixStep_mul_filter_W = filter_pixStep*filter_W;
		int filter_SliceStep = filters.GetSliceStep();
		int out_pixStep = output.GetPixelStep();
		int out_widthStep = output.GetWidthStep();
		int matrix_A_cols = filter_SliceStep;
		int matrix_A_rows = out_H*out_W;
		int matrix_B_cols = filter_N;
		int matrix_B_rows = filter_SliceStep;

		const float* in_firstPixel = input.GetFirstPixelPtr() - pad_H*in_widthStep - pad_W*in_pixelStep;
		float* out_firstPixel = output.GetFirstPixelPtr();
		float* matrix_A = (float*)_aligned_malloc(matrix_A_rows*matrix_A_cols * sizeof(float), 32);
		const float* in_row_ptr, *in_pix_ptr, *cur_in_row_ptr;
		int out_h, out_w, kh;
		float* matrix_A_row_ptr = matrix_A, *matrix_A_col_ptr;
		for (out_h = 0, in_row_ptr = in_firstPixel; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
			{

				for (kh = 0, cur_in_row_ptr = in_pix_ptr, matrix_A_col_ptr = matrix_A_row_ptr;
					kh < filter_H;
					kh++, cur_in_row_ptr += in_widthStep, matrix_A_col_ptr += filter_pixStep_mul_filter_W)
					memcpy(matrix_A_col_ptr, cur_in_row_ptr, sizeof(float)*filter_pixStep_mul_filter_W);
			}
		}
		const float* matrix_B = filters.GetFirstPixelPtr();

		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
			matrix_B, matrix_A_cols, 0, matrix_C, matrix_B_cols);
		int out_row_idx = 0;
		for (out_h = 0; out_h < out_H; out_h++)
		{
			for (out_w = 0; out_w < out_w; out_w++)
			{
				memcpy(out_firstPixel + out_h*out_widthStep + out_w*out_pixStep, matrix_C + out_row_idx*matrix_B_cols, sizeof(float)*matrix_B_cols);
			}
		}
		_aligned_free(matrix_A);

	}


	free(matrix_C);
	double t2 = omp_get_wtime();
	double mul_count = (double)out_H*out_W*filter_N*filter_H*filter_W*filter_C*iters;
	double gflops = mul_count / (1 << 30) / (t2 - t1);

	printf("%dx%d * %dx%dx%dx%d * %d = %.3e, time = %.3f s, gemm gflops = %.3f\n", out_H, out_W, filter_N, filter_H, filter_W, filter_C, iters, mul_count, t2 - t1, gflops);

	return (t2 - t1) / iters;

}


double _test_gemm(int M, int N, int K, int iters, float thresh, bool show)
{
	int padK = (K + 7) >> 3 << 3;
	float* A = (float*)_aligned_malloc(M*padK * sizeof(float), 32);
	float* B = (float*)_aligned_malloc(padK*N * sizeof(float), 32);
	float* C1 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C2 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* q = (float*)_aligned_malloc(32, 32);


	for (int i = 0; i < M*padK; i++)
		A[i] = rand() % 10001 / 5000.0f - 1.0f;
	for (int i = 0; i < padK*N; i++)
		B[i] = rand() % 10001 / 5000.0f - 1.0f;
	double t1 = omp_get_wtime(), t2, mul_count, gflops;
	double time1 = FLT_MAX;
	{
		for (int i = 0; i < iters; i++)
		{
			zq_gemm_32f_AnoTrans_Btrans_auto(M, N, K, A, padK, B, padK, C1, N);
		}
		t2 = omp_get_wtime();
		time1 = t2 - t1;
		mul_count = (double)M*N*K*iters;
		gflops = mul_count / (1 << 30) / (t2 - t1);
		//printf("C1[0] = %f\n", C1[0]);
		printf("%d x %d x %d * %d = %.3e, time = %.3f s, my gflops = %.3f\n", M, N, K, iters, mul_count, time1, gflops);
	}



	t1 = omp_get_wtime();
	for (int i = 0; i < iters; i++)
	{
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, A, padK, B, padK, 0, C2, N);
	}
	//printf("C2[0] = %f\n", C2[0]);
	t2 = omp_get_wtime();
	mul_count = (double)M*N*K*iters;
	gflops = mul_count / (1 << 30) / (t2 - t1);
	double  time2 = t2 - t1;
	printf("%d x %d x %d * %d = %.3e, time = %.3f s, gemm gflops = %.3f\n", M, N, K, iters, mul_count, time2, gflops);

	printf("check = %s\n", check_value(M, N, C1, N, C2, N, thresh, show) ? "True" : "False");
	_aligned_free(A);
	_aligned_free(B);
	_aligned_free(C1);
	_aligned_free(C2);
	_aligned_free(q);


	return __min(time1, time2) / iters;
}
