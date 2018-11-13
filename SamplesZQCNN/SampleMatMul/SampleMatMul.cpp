#include <intrin.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "MatMul_ABt.h"
#include "MatMul_AB.h"
#include "math\zq_gemm_32f_align_c.h"

void test_ABt(int M, int N, int K, int nIters, float thresh = 1e-4, bool show = false);

void test_AB(int M, int N, int K, int nIters, float thresh = 1e-4, bool show = false);

bool check_value(int M, int N, const float* C1, int ldc1, const float* C2, int ldc2, float thresh = 1e-5, bool show = false);

int main()
{
	//while (true)
	//{
	//	int M = rand() % 1000+1;
	//	int N = rand() % 1000+1;
	//	int K = rand() % 1000+1;
	//	test_ABt(M, N, K, 1, 1e-4, true);
	//	//test_ABt(805, 628, 576, 1, 1e-4, true);
	//}

	test_ABt(56 * 56, 64, 27, 800);
	test_ABt(64, 56 * 56, 27, 800);

	test_ABt(56 * 56, 64, 32, 800);
	test_ABt(64, 56 * 56, 32, 800);

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
	float* C1 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C2 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C3 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C4 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C5 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C6 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C7 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C8 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C9 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C10 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C11 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C12 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C13 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C14 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C15 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C16 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C17 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C18 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C19 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C20 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C21 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C22 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C23 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C24 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C25 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C26 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C27 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C28 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	for (int i = 0; i < M*padK; i++)
		A[i] = rand() % 10001 / 5000.0f - 1.0f;
	for (int i = 0; i < padK*N; i++)
		Bt[i] = rand() % 10001 / 5000.0f - 1.0f;
	double time = 1;
	double t0 = omp_get_wtime();
	//for (int i = 0; i < nIters; i++)
	{
		MatMul0_ABt(M, N, K, A, padK, Bt, padK, C0, N);
	}
	double t1 = omp_get_wtime();
	//printf("%d x %d x %d, cost = %.3f s, MM0 gflops = %.3f\n", M, N, K, t1 - t0, mul_count / (t1 - t0));

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N1(M, N, K, A, padK, Bt, padK, C1, N);
	}
	double t2 = omp_get_wtime(); time = t2 - t1;
	printf("%d x %d x %d, cost = %.3f s, SSE-M1-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N2(M, N, K, A, padK, Bt, padK, C2, N);
	}
	double t3 = omp_get_wtime(); time = t3 - t2;
	printf("%d x %d x %d, cost = %.3f s, SSE-M1-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);
	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N4(M, N, K, A, padK, Bt, padK, C3, N);
	}
	double t4 = omp_get_wtime(); time = t4 - t3;
	printf("%d x %d x %d, cost = %.3f s, SSE-M1-N4 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N8(M, N, K, A, padK, Bt, padK, C4, N);
	}
	double t5 = omp_get_wtime(); time = t5 - t4;
	printf("%d x %d x %d, cost = %.3f s, SSE-M1-N8 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N1(M, N, K, A, padK, Bt, padK, C5, N);
	}
	double t6 = omp_get_wtime(); time = t6 - t5;
	printf("%d x %d x %d, cost = %.3f s, SSE-M2-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N2(M, N, K, A, padK, Bt, padK, C6, N);
	}
	double t7 = omp_get_wtime(); time = t7 - t6;
	printf("%d x %d x %d, cost = %.3f s, SSE-M2-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N4(M, N, K, A, padK, Bt, padK, C7, N);
	}
	double t8 = omp_get_wtime(); time = t8 - t7;
	printf("%d x %d x %d, cost = %.3f s, SSE-M2-N4 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N8(M, N, K, A, padK, Bt, padK, C8, N);
	}
	double t9 = omp_get_wtime(); time = t9 - t8;
	printf("%d x %d x %d, cost = %.3f s, SSE-M2-N8 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, padK, Bt, padK, C9, N);
	}
	double t10 = omp_get_wtime(); time = t10 - t9;
	printf("%d x %d x %d, cost = %.3f s, SSE-M4-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N2(M, N, K, A, padK, Bt, padK, C10, N);
	}
	double t11 = omp_get_wtime(); time = t11 - t10;
	printf("%d x %d x %d, cost = %.3f s, SSE-M4-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, padK, Bt, padK, C11, N);
	}
	double t12 = omp_get_wtime(); time = t12 - t11;
	printf("%d x %d x %d, cost = %.3f s, SSE-M4-N4 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_N1(M, N, K, A, padK, Bt, padK, C12, N);
	}
	double t13 = omp_get_wtime(); time = t13 - t12;
	printf("%d x %d x %d, cost = %.3f s, SSE-M8-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_N2(M, N, K, A, padK, Bt, padK, C13, N);
	}
	double t14 = omp_get_wtime(); time = t14 - t13;
	printf("%d x %d x %d, cost = %.3f s, SSE-M8-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N1(M, N, K, A, padK, Bt, padK, C14, N);
	}
	double t15 = omp_get_wtime(); time = t15 - t14;
	printf("%d x %d x %d, cost = %.3f s, AVX-M1-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N2(M, N, K, A, padK, Bt, padK, C15, N);
	}
	double t16 = omp_get_wtime(); time = t16 - t15;
	printf("%d x %d x %d, cost = %.3f s, AVX-M1-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);
	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N4(M, N, K, A, padK, Bt, padK, C16, N);
	}
	double t17 = omp_get_wtime(); time = t17 - t16;
	printf("%d x %d x %d, cost = %.3f s, AVX-M1-N4 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N8(M, N, K, A, padK, Bt, padK, C17, N);
	}
	double t18 = omp_get_wtime(); time = t18 - t17;
	printf("%d x %d x %d, cost = %.3f s, AVX-M1-N8 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N1(M, N, K, A, padK, Bt, padK, C18, N);
	}
	double t19 = omp_get_wtime(); time = t19 - t18;
	printf("%d x %d x %d, cost = %.3f s, AVX-M2-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N2(M, N, K, A, padK, Bt, padK, C19, N);
	}
	double t20 = omp_get_wtime(); time = t20 - t19;
	printf("%d x %d x %d, cost = %.3f s, AVX-M2-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N4(M, N, K, A, padK, Bt, padK, C20, N);
	}
	double t21 = omp_get_wtime(); time = t21 - t20;
	printf("%d x %d x %d, cost = %.3f s, AVX-M2-N4 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N8(M, N, K, A, padK, Bt, padK, C21, N);
	}
	double t22 = omp_get_wtime(); time = t22 - t21;
	printf("%d x %d x %d, cost = %.3f s, AVX-M2-N8 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N1(M, N, K, A, padK, Bt, padK, C22, N);
	}
	double t23 = omp_get_wtime(); time = t23 - t22;
	printf("%d x %d x %d, cost = %.3f s, AVX-M4-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N2(M, N, K, A, padK, Bt, padK, C23, N);
	}
	double t24 = omp_get_wtime(); time = t24 - t23;
	printf("%d x %d x %d, cost = %.3f s, AVX-M4-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N4(M, N, K, A, padK, Bt, padK, C24, N);
	}
	double t25 = omp_get_wtime(); time = t25 - t24;
	printf("%d x %d x %d, cost = %.3f s, AVX-M4-N4 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_N1(M, N, K, A, padK, Bt, padK, C25, N);
	}
	double t26 = omp_get_wtime(); time = t26 - t25;
	printf("%d x %d x %d, cost = %.3f s, AVX-M8-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_N2(M, N, K, A, padK, Bt, padK, C26, N);
	}
	double t27 = omp_get_wtime(); time = t27 - t26;
	printf("%d x %d x %d, cost = %.3f s, AVX-M8-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align128bit_AnoTrans_Btrans_M16_N1(M, N, K, A, padK, Bt, padK, C27, N);
	}
	double t28 = omp_get_wtime(); time = t28 - t27;
	printf("%d x %d x %d, cost = %.3f s, SSE-M16-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M16_N1(M, N, K, A, padK, Bt, padK, C28, N);
	}
	double t29 = omp_get_wtime(); time = t27 - t26;
	printf("%d x %d x %d, cost = %.3f s, AVX-M16-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	printf("check C0 C1 = %s\n", check_value(M, N, C0, N, C1, N, thresh, show) ? "True" : "False");
	printf("check C0 C2 = %s\n", check_value(M, N, C0, N, C2, N, thresh, show) ? "True" : "False");
	printf("check C0 C3 = %s\n", check_value(M, N, C0, N, C3, N, thresh, show) ? "True" : "False");
	printf("check C0 C4 = %s\n", check_value(M, N, C0, N, C4, N, thresh, show) ? "True" : "False");
	printf("check C0 C5 = %s\n", check_value(M, N, C0, N, C5, N, thresh, show) ? "True" : "False");
	printf("check C0 C6 = %s\n", check_value(M, N, C0, N, C6, N, thresh, show) ? "True" : "False");
	printf("check C0 C7 = %s\n", check_value(M, N, C0, N, C7, N, thresh, show) ? "True" : "False");
	printf("check C0 C8 = %s\n", check_value(M, N, C0, N, C8, N, thresh, show) ? "True" : "False");
	printf("check C0 C9 = %s\n", check_value(M, N, C0, N, C9, N, thresh, show) ? "True" : "False");
	printf("check C0 C10 = %s\n", check_value(M, N, C0, N, C10, N, thresh, show) ? "True" : "False");
	printf("check C0 C11 = %s\n", check_value(M, N, C0, N, C11, N, thresh, show) ? "True" : "False");
	printf("check C0 C12 = %s\n", check_value(M, N, C0, N, C12, N, thresh, show) ? "True" : "False");
	printf("check C0 C13 = %s\n", check_value(M, N, C0, N, C13, N, thresh, show) ? "True" : "False");
	printf("check C0 C14 = %s\n", check_value(M, N, C0, N, C14, N, thresh, show) ? "True" : "False");
	printf("check C0 C15 = %s\n", check_value(M, N, C0, N, C15, N, thresh, show) ? "True" : "False");
	printf("check C0 C16 = %s\n", check_value(M, N, C0, N, C16, N, thresh, show) ? "True" : "False");
	printf("check C0 C17 = %s\n", check_value(M, N, C0, N, C17, N, thresh, show) ? "True" : "False");
	printf("check C0 C18 = %s\n", check_value(M, N, C0, N, C18, N, thresh, show) ? "True" : "False");
	printf("check C0 C19 = %s\n", check_value(M, N, C0, N, C19, N, thresh, show) ? "True" : "False");
	printf("check C0 C20 = %s\n", check_value(M, N, C0, N, C20, N, thresh, show) ? "True" : "False");
	printf("check C0 C21 = %s\n", check_value(M, N, C0, N, C21, N, thresh, show) ? "True" : "False");
	printf("check C0 C22 = %s\n", check_value(M, N, C0, N, C22, N, thresh, show) ? "True" : "False");
	printf("check C0 C23 = %s\n", check_value(M, N, C0, N, C23, N, thresh, show) ? "True" : "False");
	printf("check C0 C24 = %s\n", check_value(M, N, C0, N, C24, N, thresh, show) ? "True" : "False");
	printf("check C0 C25 = %s\n", check_value(M, N, C0, N, C25, N, thresh, show) ? "True" : "False");
	printf("check C0 C26 = %s\n", check_value(M, N, C0, N, C26, N, thresh, show) ? "True" : "False");
	printf("check C0 C27 = %s\n", check_value(M, N, C0, N, C27, N, thresh, show) ? "True" : "False");
	printf("check C0 C28 = %s\n", check_value(M, N, C0, N, C28, N, thresh, show) ? "True" : "False");
	_aligned_free(A);
	_aligned_free(Bt);
	_aligned_free(C0);
	_aligned_free(C1);
	_aligned_free(C2);
	_aligned_free(C3);
	_aligned_free(C4);
	_aligned_free(C5);
	_aligned_free(C6);
	_aligned_free(C7);
	_aligned_free(C8);
	_aligned_free(C9);
	_aligned_free(C10);
	_aligned_free(C11);
	_aligned_free(C12);
	_aligned_free(C13);
	_aligned_free(C14);
	_aligned_free(C15);
	_aligned_free(C16);
	_aligned_free(C17);
	_aligned_free(C18);
	_aligned_free(C19);
	_aligned_free(C20);
	_aligned_free(C21);
	_aligned_free(C22);
	_aligned_free(C23);
	_aligned_free(C24);
	_aligned_free(C25);
	_aligned_free(C26);
	_aligned_free(C27);
	_aligned_free(C28);
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
	double t0 = omp_get_wtime();
	for (int i = 0; i < nIters; i++)
	{
		MatMul0_AB(M, N, K, A, padK, B, padN, C0, N);
	}
	double t1 = omp_get_wtime();
	printf("%d x %d x %d, cost = %.3f s, MM0 gflops = %.3f\n", M, N, K, t1 - t0, mul_count / (t1 - t0));

	for (int i = 0; i < nIters; i++)
	{
		MatMul1_AB(M, N, K, A, padK, B, padN, C1, N);
	}
	double t2 = omp_get_wtime();
	printf("%d x %d x %d, cost = %.3f s, MM1 gflops = %.3f\n", M, N, K, t2 - t1, mul_count / (t2 - t1));

	for (int i = 0; i < nIters; i++)
	{
		MatMul2_AB(M, N, K, A, padK, B, padN, C2, N);
	}
	double t3 = omp_get_wtime();
	printf("%d x %d x %d, cost = %.3f s, MM2 gflops = %.3f\n", M, N, K, t3 - t2, mul_count / (t3 - t2));
	for (int i = 0; i < nIters; i++)
	{
		MatMul3_AB(M, N, K, A, padK, B, padN, C3, N);
	}
	double t4 = omp_get_wtime();
	printf("%d x %d x %d, cost = %.3f s, MM3 gflops = %.3f\n", M, N, K, t4 - t3, mul_count / (t4 - t3));

	for (int i = 0; i < nIters; i++)
	{
		MatMul4_AB(M, N, K, A, padK, B, padN, C4, N);
	}
	double t5 = omp_get_wtime();
	printf("%d x %d x %d, cost = %.3f s, MM4 gflops = %.3f\n", M, N, K, t5 - t4, mul_count / (t5 - t4));


	/*for (int i = 0; i < nIters; i++)
	{
		MatMul5_AB(M, N, K, A, padK, B, padN, C5, N);
	}
	double t6 = omp_get_wtime();
	printf("%d x %d x %d, cost = %.3f s, MM5 gflops = %.3f\n", M, N, K, t6 - t5, mul_count / (t6 - t5));

	for (int i = 0; i < nIters; i++)
	{
		MatMul6_AB(M, N, K, A, padK, B, padN, C6, N);
	}
	double t7 = omp_get_wtime();
	printf("%d x %d x %d, cost = %.3f s, MM6 gflops = %.3f\n", M, N, K, t7 - t6, mul_count / (t7 - t6));*/
	printf("check C0 C1 = %s\n", check_value(M, N, C0, N, C1, N, thresh, show) ? "True" : "False");
	printf("check C0 C2 = %s\n", check_value(M, N, C0, N, C2, N, thresh, show) ? "True" : "False");
	printf("check C0 C3 = %s\n", check_value(M, N, C0, N, C3, N, thresh, show) ? "True" : "False");
	printf("check C0 C4 = %s\n", check_value(M, N, C0, N, C4, N, thresh, show) ? "True" : "False");
	//printf("check C0 C5 = %s\n", check_value(M, N, C0, N, C5, N, thresh, show) ? "True" : "False");
	//printf("check C0 C6 = %s\n", check_value(M, N, C0, N, C6, N, thresh, show) ? "True" : "False");

	_aligned_free(A);
	_aligned_free(B);
	_aligned_free(C0);
	_aligned_free(C1);
	_aligned_free(C2);
	_aligned_free(C3);
	_aligned_free(C4);
	_aligned_free(C5);
	_aligned_free(C6);
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
