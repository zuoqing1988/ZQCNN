#if defined(_WIN32)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
//#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "MatMul_ABt.h"
#include "MatMul_AB.h"
#include "math/zq_gemm_32f_align_c.h"
#pragma comment(lib,"ZQ_GEMM.lib")

#if defined(_WIN32)
double time_scale = 0.001;
#else
double time_scale = 0.000001;
#endif

void test_ABt(int M, int N, int K, int nIters, float thresh = 1e-4, bool show = false);

void test_AB(int M, int N, int K, int nIters, float thresh = 1e-4, bool show = false);

bool check_value(int M, int N, const float* C1, int ldc1, const float* C2, int ldc2, float thresh = 1e-5, bool show = false);

int main()
{
	/*while (true)
	{
		int M = rand() % 1000+1;
		int N = rand() % 1000+1;
		int K = rand() % 1000+1;
		test_ABt(M, N, K, 1, 1e-4, true);
	}*/
	test_ABt(128, 128, 128, 5000);
	////return 0;
	test_ABt(256, 256, 256, 1000);
	test_ABt(512, 512, 512, 200);
	test_ABt(1024, 1024, 1024, 50);

	//return 0;
	/*test_ABt(56 * 56, 16, 8, 8000);
	test_ABt(16, 56 * 56, 8, 8000);

	test_ABt(56 * 56, 24, 16, 8000);
	test_ABt(24, 56 * 56, 16, 8000);

	test_ABt(56 * 56, 32, 24, 8000);
	test_ABt(32, 56 * 56, 24, 8000);

	test_ABt(56 * 56, 64, 27, 8000);
	test_ABt(64, 56 * 56, 27, 8000);*/

	test_ABt(56 * 56, 64, 28, 8000);
	test_ABt(64, 56 * 56, 28, 8000);


	test_ABt(56 * 56, 64, 32, 8000);
	test_ABt(64, 56 * 56, 32, 8000);
	//return 0;
	test_ABt(56 * 56, 64, 64, 800);
	test_ABt(64, 56 * 56, 64, 800);
	
	test_ABt(56 * 56, 128, 64, 800);
	test_ABt(128, 56 * 56, 64, 800);
	
	test_ABt(28 * 28, 128, 128, 800);
	test_ABt(128, 28 * 28, 128, 800);

	test_ABt(28 * 28, 256, 128, 800);
	test_ABt(256, 28 * 28, 128, 800);
	
	test_ABt(14 * 14, 256, 256, 800);
	test_ABt(256, 14 * 14, 256, 800);
	
	test_ABt(14 * 14, 512, 256, 800);
	test_ABt(512, 14 * 14, 256, 800);
	
	test_ABt(7 * 7, 512, 512, 800);
	test_ABt(512, 7 * 7, 512, 800);
	

	// test 1x1 kernel
	test_ABt(56 * 56, 16, 16, 8000);
	test_ABt(16, 56 * 56, 16, 8000);
	test_ABt(56 * 56, 32, 16, 4000);
	test_ABt(32, 56 * 56, 16, 4000);
	test_ABt(56 * 56, 64, 16, 2000);
	test_ABt(64, 56 * 56, 16, 2000);
	test_ABt(56 * 56, 128, 16, 1000);
	test_ABt(128, 56 * 56, 16, 1000);
	test_ABt(56 * 56, 32, 32, 20000);
	test_ABt(56 * 56, 64, 32, 10000);
	test_ABt(56 * 56, 128, 32, 10000);
	test_ABt(56 * 56, 256, 32, 10000);
	test_ABt(56 * 56, 64, 64, 2000);
	test_ABt(56 * 56, 128, 64, 1000);
	test_ABt(56 * 56, 256, 64, 1000);
	test_ABt(56 * 56, 512, 64, 1000);
	test_ABt(56 * 56, 128, 128, 1000);
	test_ABt(56 * 56, 256, 128, 500);
	test_ABt(56 * 56, 512, 128, 500);
	test_ABt(56 * 56, 1024, 128, 200);
	test_ABt(56 * 56, 256, 256, 200);
	test_ABt(56 * 56, 512, 256, 100);
	test_ABt(56 * 56, 1024, 256, 50);
	test_ABt(56 * 56, 512, 512, 100);
	test_ABt(56 * 56, 1024, 512, 50);
	test_ABt(56 * 56, 1024, 1024, 50);


	test_ABt(28 * 28, 16, 16, 50000);
	test_ABt(28 * 28, 32, 16, 30000);
	test_ABt(28 * 28, 64, 16, 20000);
	test_ABt(28 * 28, 128, 16, 10000);
	test_ABt(28 * 28, 32, 32, 50000);
	test_ABt(28 * 28, 64, 32, 20000);
	test_ABt(28 * 28, 128, 32, 10000);
	test_ABt(28 * 28, 256, 32, 10000);
	test_ABt(28 * 28, 64, 64, 50000);
	test_ABt(28 * 28, 128, 64, 20000);
	test_ABt(28 * 28, 256, 64, 10000);
	test_ABt(28 * 28, 512, 64, 10000);
	test_ABt(28 * 28, 128, 128, 2000);
	test_ABt(28 * 28, 256, 128, 1000);
	test_ABt(28 * 28, 512, 128, 1000);
	test_ABt(28 * 28, 1024, 128, 500);
	test_ABt(28 * 28, 256, 256, 1000);
	test_ABt(28 * 28, 512, 256, 500);
	test_ABt(28 * 28, 1024, 256, 500);
	test_ABt(28 * 28, 512, 512, 200);
	test_ABt(28 * 28, 1024, 512, 100);
	test_ABt(28 * 28, 1024, 1024, 100);


	test_ABt(14 * 14, 16, 16, 50000*4);
	test_ABt(14 * 14, 32, 16, 30000*4);
	test_ABt(14 * 14, 64, 16, 20000*4);
	test_ABt(14 * 14, 128, 16, 10000*4);
	test_ABt(14 * 14, 32, 32, 50000*4);
	test_ABt(14 * 14, 64, 32, 20000*4);
	test_ABt(14 * 14, 128, 32, 10000*4);
	test_ABt(14 * 14, 256, 32, 10000*4);
	test_ABt(14 * 14, 64, 64, 50000*4);
	test_ABt(14 * 14, 128, 64, 20000*4);
	test_ABt(14 * 14, 256, 64, 10000*4);
	test_ABt(14 * 14, 512, 64, 10000*4);
	test_ABt(14 * 14, 128, 128, 2000*4);
	test_ABt(14 * 14, 256, 128, 1000*4);
	test_ABt(14 * 14, 512, 128, 1000*4);
	test_ABt(14 * 14, 1024, 128, 500*4);
	test_ABt(14 * 14, 256, 256, 1000*4);
	test_ABt(14 * 14, 512, 256, 500*4);
	test_ABt(14 * 14, 1024, 256, 500*4);
	test_ABt(14 * 14, 512, 512, 200*4);
	test_ABt(14 * 14, 1024, 512, 100*4);
	test_ABt(14 * 14, 1024, 1024, 100*4);


	test_ABt(7 * 7, 16, 16, 50000 * 16);
	test_ABt(7 * 7, 32, 16, 30000 * 16);
	test_ABt(7 * 7, 64, 16, 20000 * 16);
	test_ABt(7 * 7, 128, 16, 10000 * 16);
	test_ABt(7 * 7, 32, 32, 50000 * 16);
	test_ABt(7 * 7, 64, 32, 20000 * 16);
	test_ABt(7 * 7, 128, 32, 10000 * 16);
	test_ABt(7 * 7, 256, 32, 10000 * 16);
	test_ABt(7 * 7, 64, 64, 50000 * 16);
	test_ABt(7 * 7, 128, 64, 20000 * 16);
	test_ABt(7 * 7, 256, 64, 10000 * 16);
	test_ABt(7 * 7, 512, 64, 10000 * 16);
	test_ABt(7 * 7, 128, 128, 2000 * 16);
	test_ABt(7 * 7, 256, 128, 1000 * 16);
	test_ABt(7 * 7, 512, 128, 1000 * 16);
	test_ABt(7 * 7, 1024, 128, 500 * 16);
	test_ABt(7 * 7, 256, 256, 1000 * 16);
	test_ABt(7 * 7, 512, 256, 500 * 16);
	test_ABt(7 * 7, 1024, 256, 500 * 16);
	test_ABt(7 * 7, 512, 512, 200 * 16);
	test_ABt(7 * 7, 1024, 512, 100 * 16);
	test_ABt(7 * 7, 1024, 1024, 100 * 16);

	test_ABt(3 * 3, 16, 16, 50000 * 100);
	test_ABt(3 * 3, 32, 16, 30000 * 100);
	test_ABt(3 * 3, 64, 16, 20000 * 100);
	test_ABt(3 * 3, 128, 16, 10000 * 100);
	test_ABt(3 * 3, 32, 32, 50000 * 100);
	test_ABt(3 * 3, 64, 32, 20000 * 100);
	test_ABt(3 * 3, 128, 32, 10000 * 100);
	test_ABt(3 * 3, 256, 32, 10000 * 100);
	test_ABt(3 * 3, 64, 64, 50000 * 100);
	test_ABt(3 * 3, 128, 64, 20000 * 100);
	test_ABt(3 * 3, 256, 64, 10000 * 100);
	test_ABt(3 * 3, 512, 64, 10000 * 100);
	test_ABt(3 * 3, 128, 128, 2000 * 100);
	test_ABt(3 * 3, 256, 128, 1000 * 100);
	test_ABt(3 * 3, 512, 128, 1000 * 100);
	test_ABt(3 * 3, 1024, 128, 500 * 100);
	test_ABt(3 * 3, 256, 256, 1000 * 100);
	test_ABt(3 * 3, 512, 256, 500 * 100);
	test_ABt(3 * 3, 1024, 256, 500 * 100);
	test_ABt(3 * 3, 512, 512, 200 * 100);
	test_ABt(3 * 3, 1024, 512, 100 * 100);
	test_ABt(3 * 3, 1024, 1024, 100 * 100);




	test_ABt(56 * 56, 16, 3 * 3 * 3, 10000);
	test_ABt(56 * 56, 32, 3 * 3 * 3, 10000);
	test_ABt(56 * 56, 64, 3 * 3 * 3, 10000);

	test_ABt(56 * 56, 16, 3 * 3 * 8, 10000);
	test_ABt(56 * 56, 32, 3 * 3 * 8, 10000);
	test_ABt(56 * 56, 64, 3 * 3 * 8, 10000);
	test_ABt(56 * 56, 128, 3 * 3 * 8, 10000);

	test_ABt(56 * 56, 16, 3 * 3 * 16, 10000);
	test_ABt(56 * 56, 32, 3 * 3 * 16, 10000);
	test_ABt(56 * 56, 64, 3 * 3 * 16, 10000);
	test_ABt(56 * 56, 128, 3 * 3 * 16, 10000);

	test_ABt(28 * 28, 64, 3 * 3 * 32, 10000);
	test_ABt(28 * 28, 128, 3 * 3 * 32, 2000);
	test_ABt(28 * 28, 256, 3 * 3 * 32, 1000);
	test_ABt(28 * 28, 512, 3 * 3 * 32, 1000);

	test_ABt(1024, 1, 1024, 40000);
	test_ABt(1024, 2, 1024, 20000);
	test_ABt(1024, 4, 1024, 10000);
	test_ABt(4, 8, 1024, 1000000);
	test_ABt(8, 8, 1024, 1000000);

	test_ABt(4, 4, 512, 10000000);
	test_ABt(4, 8, 512, 10000000);
	test_ABt(8, 8, 512, 1000000);

	test_ABt(4, 4, 256, 10000000);
	test_ABt(4, 8, 256, 10000000);
	test_ABt(8, 8, 256, 10000000);

	test_ABt(28 * 28, 4, 1024, 100000);

	test_ABt(28 * 28, 48, 512, 10000);
	test_ABt(48, 28 * 28, 512, 10000);
	test_ABt(28 * 28, 64, 3 * 3 * 64, 5000);
	test_ABt(28 * 28, 128, 3 * 3 * 64, 2000);
	test_ABt(28 * 28, 256, 3 * 3 * 64, 1000);
	test_ABt(28 * 28, 512, 3 * 3 * 64, 1000);

	
	test_AB(56 * 56, 8, 3 * 3 * 3, 1000);
	test_AB(56 * 56, 16, 3 * 3 * 3, 1000);
	test_AB(56 * 56, 32, 3 * 3 * 3, 1000);
	test_AB(56 * 56, 64, 3 * 3 * 3, 1000);

	test_AB(56 * 56, 16, 3 * 3 * 8, 1000);
	test_AB(56 * 56, 32, 3 * 3 * 8, 1000);
	test_AB(56 * 56, 64, 3 * 3 * 8, 1000);
	test_AB(56 * 56, 128, 3 * 3 * 8, 1000);

	test_AB(56 * 56, 16, 3 * 3 * 16, 1000);
	test_AB(56 * 56, 32, 3 * 3 * 16, 1000);
	test_AB(56 * 56, 64, 3 * 3 * 16, 1000);
	test_AB(56 * 56, 128, 3 * 3 * 16, 1000);

	return 0;
}

void test_ABt(int M, int N, int K, int nIters, float thresh, bool show)
{
	int padK = (K + 7) >> 3 << 3;
	/*if (padK % 128 == 0)
		padK += 8;*/
	double mul_count = (double)M*N*K*nIters / (1024.0*1024.0*1024.0);
	float* A = (float*)_aligned_malloc(M*padK * sizeof(float), 32);
	float* Bt = (float*)_aligned_malloc(N*padK * sizeof(float), 32);
	float* C0 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	float* C1_1 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C1_2 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C1_3 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C1_4 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C1_5 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C1_6 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C1_7 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C1_8 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C1_9 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C1_10 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C1_11 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C1_12 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C1_13 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C1_14 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	float* C2_1 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C2_2 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C2_3 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C2_4 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C2_5 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C2_6 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C2_7 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C2_8 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C2_9 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C2_10 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C2_11 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C2_12 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C2_13 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C2_14 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
#endif
	for (int i = 0; i < M*padK; i++)
		A[i] = rand() % 10001 / 5000.0f - 1.0f;
	for (int i = 0; i < padK*N; i++)
		Bt[i] = rand() % 10001 / 5000.0f - 1.0f;
	double time = 1;
	clock_t t0 = clock();
	int naive_nIters = 1;
	double navie_mul_count = (double)M*N*K*naive_nIters/(1024.0*1024.0*1024.0);
	for (int i = 0; i < naive_nIters; i++)
	{
		MatMul0_ABt(M, N, K, A, padK, Bt, padK, C0, N);
	}
	clock_t t1 = clock(); time = __max(1e-9,time_scale*(t1 - t0));
	//printf("%d x %d x %d, cost = %.3f s, naive gflops = %.3f\n", M, N, K, time, navie_mul_count / time);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	clock_t t1_1 = clock();
	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N1(M, N, K, A, padK, Bt, padK, C1_1, N);
	}
	clock_t t1_2 = clock(); time = time_scale*(t1_2 - t1_1);
	printf("%d x %d x %d, cost = %.3f s, SSE-M1-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N2(M, N, K, A, padK, Bt, padK, C1_2, N);
	}
	clock_t t1_3 = clock(); time = time_scale*(t1_3 - t1_2);
	printf("%d x %d x %d, cost = %.3f s, SSE-M1-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);
	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N4(M, N, K, A, padK, Bt, padK, C1_3, N);
	}
	clock_t t1_4 = clock(); time = time_scale*(t1_4 - t1_3);
	printf("%d x %d x %d, cost = %.3f s, SSE-M1-N4 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N8(M, N, K, A, padK, Bt, padK, C1_4, N);
	}
	clock_t t1_5 = clock(); time = time_scale*(t1_5 - t1_4);
	printf("%d x %d x %d, cost = %.3f s, SSE-M1-N8 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N1(M, N, K, A, padK, Bt, padK, C1_5, N);
	}
	clock_t t1_6 = clock(); time = time_scale*(t1_6 - t1_5);
	printf("%d x %d x %d, cost = %.3f s, SSE-M2-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N2(M, N, K, A, padK, Bt, padK, C1_6, N);
	}
	clock_t t1_7 = clock(); time = time_scale*(t1_7 - t1_6);
	printf("%d x %d x %d, cost = %.3f s, SSE-M2-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N4(M, N, K, A, padK, Bt, padK, C1_7, N);
	}
	clock_t t1_8 = clock(); time = time_scale*(t1_8 - t1_7);
	printf("%d x %d x %d, cost = %.3f s, SSE-M2-N4 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N8(M, N, K, A, padK, Bt, padK, C1_8, N);
	}
	clock_t t1_9 = clock(); time = time_scale*(t1_9 - t1_8);
	printf("%d x %d x %d, cost = %.3f s, SSE-M2-N8 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, padK, Bt, padK, C1_9, N);
	}
	clock_t t1_10 = clock(); time = time_scale*(t1_10 - t1_9);
	printf("%d x %d x %d, cost = %.3f s, SSE-M4-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N2(M, N, K, A, padK, Bt, padK, C1_10, N);
	}
	clock_t t1_11 = clock(); time = time_scale*(t1_11 - t1_10);
	printf("%d x %d x %d, cost = %.3f s, SSE-M4-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, padK, Bt, padK, C1_11, N);
	}
	clock_t t1_12 = clock(); time = time_scale*(t1_12 - t1_11);
	printf("%d x %d x %d, cost = %.3f s, SSE-M4-N4 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_N1(M, N, K, A, padK, Bt, padK, C1_12, N);
	}
	clock_t t1_13 = clock(); time = time_scale*(t1_13 - t1_12);
	printf("%d x %d x %d, cost = %.3f s, SSE-M8-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_N2(M, N, K, A, padK, Bt, padK, C1_13, N);
	}
	clock_t t1_14 = clock(); time = time_scale*(t1_14 - t1_13);
	printf("%d x %d x %d, cost = %.3f s, SSE-M8-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M16_N1(M, N, K, A, padK, Bt, padK, C1_14, N);
	}
	clock_t t1_15 = clock(); time = time_scale*(t1_15 - t1_14);
	printf("%d x %d x %d, cost = %.3f s, SSE-M16-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	clock_t t2_1 = clock();
	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N1(M, N, K, A, padK, Bt, padK, C2_1, N);
	}
	clock_t t2_2 = clock(); time = time_scale*(t2_2 - t2_1);
	printf("%d x %d x %d, cost = %.3f s, AVX-M1-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N2(M, N, K, A, padK, Bt, padK, C2_2, N);
	}
	clock_t t2_3 = clock(); time = time_scale*(t2_3 - t2_2);
	printf("%d x %d x %d, cost = %.3f s, AVX-M1-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);
	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N4(M, N, K, A, padK, Bt, padK, C2_3, N);
	}
	clock_t t2_4 = clock(); time = time_scale*(t2_4 - t2_3);
	printf("%d x %d x %d, cost = %.3f s, AVX-M1-N4 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N8(M, N, K, A, padK, Bt, padK, C2_4, N);
	}
	clock_t t2_5 = clock(); time = time_scale*(t2_5 - t2_4);
	printf("%d x %d x %d, cost = %.3f s, AVX-M1-N8 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N1(M, N, K, A, padK, Bt, padK, C2_5, N);
	}
	clock_t t2_6 = clock(); time = time_scale*(t2_6 - t2_5);
	printf("%d x %d x %d, cost = %.3f s, AVX-M2-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N2(M, N, K, A, padK, Bt, padK, C2_6, N);
	}
	clock_t t2_7 = clock(); time = time_scale*(t2_7 - t2_6);
	printf("%d x %d x %d, cost = %.3f s, AVX-M2-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N4(M, N, K, A, padK, Bt, padK, C2_7, N);
	}
	clock_t t2_8 = clock(); time = time_scale*(t2_8 - t2_7);
	printf("%d x %d x %d, cost = %.3f s, AVX-M2-N4 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N8(M, N, K, A, padK, Bt, padK, C2_8, N);
	}
	clock_t t2_9 = clock(); time = time_scale*(t2_9 - t2_8);
	printf("%d x %d x %d, cost = %.3f s, AVX-M2-N8 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N1(M, N, K, A, padK, Bt, padK, C2_9, N);
	}
	clock_t t2_10 = clock(); time = time_scale*(t2_10 - t2_9);
	printf("%d x %d x %d, cost = %.3f s, AVX-M4-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N2(M, N, K, A, padK, Bt, padK, C2_10, N);
	}
	clock_t t2_11 = clock(); time = time_scale*(t2_11 - t2_10);
	printf("%d x %d x %d, cost = %.3f s, AVX-M4-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N4(M, N, K, A, padK, Bt, padK, C2_11, N);
	}
	clock_t t2_12 = clock(); time = time_scale*(t2_12 - t2_11);
	printf("%d x %d x %d, cost = %.3f s, AVX-M4-N4 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_N1(M, N, K, A, padK, Bt, padK, C2_12, N);
	}
	clock_t t2_13 = clock(); time = time_scale*(t2_13 - t2_12);
	printf("%d x %d x %d, cost = %.3f s, AVX-M8-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_N2(M, N, K, A, padK, Bt, padK, C2_13, N);
	}
	clock_t t2_14 = clock(); time = time_scale*(t2_14 - t2_13);
	printf("%d x %d x %d, cost = %.3f s, AVX-M8-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M16_N1(M, N, K, A, padK, Bt, padK, C2_14, N);
	}
	clock_t t2_15 = clock(); time = time_scale*(t2_15 - t2_14);
	printf("%d x %d x %d, cost = %.3f s, AVX-M16-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	printf("check C0 C1_1 = %s\n", check_value(M, N, C0, N, C1_1, N, thresh, show) ? "True" : "False");
	printf("check C0 C1_2 = %s\n", check_value(M, N, C0, N, C1_2, N, thresh, show) ? "True" : "False");
	printf("check C0 C1_3 = %s\n", check_value(M, N, C0, N, C1_3, N, thresh, show) ? "True" : "False");
	printf("check C0 C1_4 = %s\n", check_value(M, N, C0, N, C1_4, N, thresh, show) ? "True" : "False");
	printf("check C0 C1_5 = %s\n", check_value(M, N, C0, N, C1_5, N, thresh, show) ? "True" : "False");
	printf("check C0 C1_6 = %s\n", check_value(M, N, C0, N, C1_6, N, thresh, show) ? "True" : "False");
	printf("check C0 C1_7 = %s\n", check_value(M, N, C0, N, C1_7, N, thresh, show) ? "True" : "False");
	printf("check C0 C1_8 = %s\n", check_value(M, N, C0, N, C1_8, N, thresh, show) ? "True" : "False");
	printf("check C0 C1_9 = %s\n", check_value(M, N, C0, N, C1_9, N, thresh, show) ? "True" : "False");
	printf("check C0 C1_10 = %s\n", check_value(M, N, C0, N, C1_10, N, thresh, show) ? "True" : "False");
	printf("check C0 C1_11 = %s\n", check_value(M, N, C0, N, C1_11, N, thresh, show) ? "True" : "False");
	printf("check C0 C1_12 = %s\n", check_value(M, N, C0, N, C1_12, N, thresh, show) ? "True" : "False");
	printf("check C0 C1_13 = %s\n", check_value(M, N, C0, N, C1_13, N, thresh, show) ? "True" : "False");
	printf("check C0 C1_14 = %s\n", check_value(M, N, C0, N, C1_14, N, thresh, show) ? "True" : "False");
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	printf("check C0 C2_1 = %s\n", check_value(M, N, C0, N, C2_1, N, thresh, show) ? "True" : "False");
	printf("check C0 C2_2 = %s\n", check_value(M, N, C0, N, C2_2, N, thresh, show) ? "True" : "False");
	printf("check C0 C2_3 = %s\n", check_value(M, N, C0, N, C2_3, N, thresh, show) ? "True" : "False");
	printf("check C0 C2_4 = %s\n", check_value(M, N, C0, N, C2_4, N, thresh, show) ? "True" : "False");
	printf("check C0 C2_5 = %s\n", check_value(M, N, C0, N, C2_5, N, thresh, show) ? "True" : "False");
	printf("check C0 C2_6 = %s\n", check_value(M, N, C0, N, C2_6, N, thresh, show) ? "True" : "False");
	printf("check C0 C2_7 = %s\n", check_value(M, N, C0, N, C2_7, N, thresh, show) ? "True" : "False");
	printf("check C0 C2_8 = %s\n", check_value(M, N, C0, N, C2_8, N, thresh, show) ? "True" : "False");
	printf("check C0 C2_9 = %s\n", check_value(M, N, C0, N, C2_9, N, thresh, show) ? "True" : "False");
	printf("check C0 C2_10 = %s\n", check_value(M, N, C0, N, C2_10, N, thresh, show) ? "True" : "False");
	printf("check C0 C2_11 = %s\n", check_value(M, N, C0, N, C2_11, N, thresh, show) ? "True" : "False");
	printf("check C0 C2_12 = %s\n", check_value(M, N, C0, N, C2_12, N, thresh, show) ? "True" : "False");
	printf("check C0 C2_13 = %s\n", check_value(M, N, C0, N, C2_13, N, thresh, show) ? "True" : "False");
	printf("check C0 C2_14 = %s\n", check_value(M, N, C0, N, C2_14, N, thresh, show) ? "True" : "False");
#endif
	_aligned_free(A);
	_aligned_free(Bt);
	_aligned_free(C0);
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	_aligned_free(C1_1);
	_aligned_free(C1_2);
	_aligned_free(C1_3);
	_aligned_free(C1_4);
	_aligned_free(C1_5);
	_aligned_free(C1_6);
	_aligned_free(C1_7);
	_aligned_free(C1_8);
	_aligned_free(C1_9);
	_aligned_free(C1_10);
	_aligned_free(C1_11);
	_aligned_free(C1_12);
	_aligned_free(C1_13);
	_aligned_free(C1_14);
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	_aligned_free(C2_1);
	_aligned_free(C2_2);
	_aligned_free(C2_3);
	_aligned_free(C2_4);
	_aligned_free(C2_5);
	_aligned_free(C2_6);
	_aligned_free(C2_7);
	_aligned_free(C2_8);
	_aligned_free(C2_9);
	_aligned_free(C2_10);
	_aligned_free(C2_11);
	_aligned_free(C2_12);
	_aligned_free(C2_13);
	_aligned_free(C2_14);
#endif
}

void test_AB(int M, int N, int K, int nIters, float thresh, bool show)
{
	int padK = (K + 7) >> 3 << 3;
	int padN = (N + 7) >> 3 << 3;
	double mul_count = (double)M*N*K*nIters / (1024.0*1024.0*1024.0);
	float* A = (float*)_aligned_malloc(M*padK * sizeof(float), 32);
	float* B = (float*)_aligned_malloc(padN*K * sizeof(float), 32);
	float* C0 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C1 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C2 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C3 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C4 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C5 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C6 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	for (int i = 0; i < M*padK; i++)
		A[i] = rand() % 10001 / 5000.0f - 1.0f;
	for (int i = 0; i < padN*K; i++)
		B[i] = rand() % 10001 / 5000.0f - 1.0f;
	double time = 1;
	clock_t t0 = clock();
	for (int i = 0; i < nIters; i++)
	{
		MatMul0_AB(M, N, K, A, padK, B, padN, C0, N);
	}
	clock_t t1 = clock(); time = time_scale*(t1 - t0);
	printf("%d x %d x %d, cost = %.3f s, MM0 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		MatMul1_AB(M, N, K, A, padK, B, padN, C1, N);
	}
	clock_t t2 = clock(); time = time_scale*(t2 - t1);
	printf("%d x %d x %d, cost = %.3f s, MM1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		MatMul2_AB(M, N, K, A, padK, B, padN, C2, N);
	}
	clock_t t3 = clock(); time = time_scale*(t3 - t2);
	printf("%d x %d x %d, cost = %.3f s, MM2 gflops = %.3f\n", M, N, K, time, mul_count / time);
	for (int i = 0; i < nIters; i++)
	{
		MatMul3_AB(M, N, K, A, padK, B, padN, C3, N);
	}
	clock_t t4 = clock(); time = time_scale*(t4 - t3);
	printf("%d x %d x %d, cost = %.3f s, MM3 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		MatMul4_AB(M, N, K, A, padK, B, padN, C4, N);
	}
	clock_t t5 = clock(); time = time_scale*(t5 - t4);
	printf("%d x %d x %d, cost = %.3f s, MM4 gflops = %.3f\n", M, N, K, time, mul_count / time);


	printf("check C0 C1 = %s\n", check_value(M, N, C0, N, C1, N, thresh, show) ? "True" : "False");
	printf("check C0 C2 = %s\n", check_value(M, N, C0, N, C2, N, thresh, show) ? "True" : "False");
	printf("check C0 C3 = %s\n", check_value(M, N, C0, N, C3, N, thresh, show) ? "True" : "False");
	printf("check C0 C4 = %s\n", check_value(M, N, C0, N, C4, N, thresh, show) ? "True" : "False");

	_aligned_free(A);
	_aligned_free(B);
	_aligned_free(C0);
	_aligned_free(C1);
	_aligned_free(C2);
	_aligned_free(C3);
	_aligned_free(C4);
}


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
