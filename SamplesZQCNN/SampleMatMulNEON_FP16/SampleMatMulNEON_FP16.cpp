#include "ZQ_CNN_CompileConfig.h"
#if __ARM_NEON_FP16
#include <arm_neon.h>

#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "math/zq_gemm_16f_align_c.h"
#pragma comment(lib,"ZQ_GEMM.lib")

double time_scale = 0.000001;

void test_ABt(int M, int N, int K, int nIters, float16_t thresh = 1e-4, bool show = false);

bool check_value(int M, int N, const float16_t* C1, int ldc1, const float16_t* C2, int ldc2, float16_t thresh = 1e-5, bool show = false);

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


	test_ABt(14 * 14, 16, 16, 50000 * 4);
	test_ABt(14 * 14, 32, 16, 30000 * 4);
	test_ABt(14 * 14, 64, 16, 20000 * 4);
	test_ABt(14 * 14, 128, 16, 10000 * 4);
	test_ABt(14 * 14, 32, 32, 50000 * 4);
	test_ABt(14 * 14, 64, 32, 20000 * 4);
	test_ABt(14 * 14, 128, 32, 10000 * 4);
	test_ABt(14 * 14, 256, 32, 10000 * 4);
	test_ABt(14 * 14, 64, 64, 50000 * 4);
	test_ABt(14 * 14, 128, 64, 20000 * 4);
	test_ABt(14 * 14, 256, 64, 10000 * 4);
	test_ABt(14 * 14, 512, 64, 10000 * 4);
	test_ABt(14 * 14, 128, 128, 2000 * 4);
	test_ABt(14 * 14, 256, 128, 1000 * 4);
	test_ABt(14 * 14, 512, 128, 1000 * 4);
	test_ABt(14 * 14, 1024, 128, 500 * 4);
	test_ABt(14 * 14, 256, 256, 1000 * 4);
	test_ABt(14 * 14, 512, 256, 500 * 4);
	test_ABt(14 * 14, 1024, 256, 500 * 4);
	test_ABt(14 * 14, 512, 512, 200 * 4);
	test_ABt(14 * 14, 1024, 512, 100 * 4);
	test_ABt(14 * 14, 1024, 1024, 100 * 4);


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

	return 0;
}


void MatMul0_ABt(int M, int N, int K, const float16_t* A, int lda, const float16_t* Bt, int ldb, float16_t* C, int ldc)
{
	int m, n, k;
	float16_t sum;
	const float16_t* A_row_ptr, *B_row_ptr;
	float16_t *C_row_ptr;
	const float16_t* A_c_ptr, *B_c_ptr;
	for (m = 0, A_row_ptr = A, C_row_ptr = C; m < M; m++, A_row_ptr += lda, C_row_ptr += ldc)
	{
		for (n = 0, B_row_ptr = Bt; n < N; n++, B_row_ptr += ldb)
		{
			sum = 0;
			for (k = 0, A_c_ptr = A_row_ptr, B_c_ptr = B_row_ptr; k < K; k++)
				sum += (*(A_c_ptr++)) * (*(B_c_ptr++));
			C_row_ptr[n] = sum;
		}
	}
}

void test_ABt(int M, int N, int K, int nIters, float16_t thresh, bool show)
{
	int padK = (K + 7) >> 3 << 3;
	/*if (padK % 128 == 0)
	padK += 8;*/
	double mul_count = (double)M*N*K*nIters / (1024.0*1024.0*1024.0);
	float16_t* A = (float16_t*)_aligned_malloc(M*padK * sizeof(float16_t), 32);
	float16_t* Bt = (float16_t*)_aligned_malloc(N*padK * sizeof(float16_t), 32);
	float16_t* C0 = (float16_t*)_aligned_malloc(M*N * sizeof(float16_t), 32);

	float16_t* C1_1 = (float16_t*)_aligned_malloc(M*N * sizeof(float16_t), 32);
	float16_t* C1_2 = (float16_t*)_aligned_malloc(M*N * sizeof(float16_t), 32);
	float16_t* C1_3 = (float16_t*)_aligned_malloc(M*N * sizeof(float16_t), 32);
	float16_t* C1_4 = (float16_t*)_aligned_malloc(M*N * sizeof(float16_t), 32);
	float16_t* C1_5 = (float16_t*)_aligned_malloc(M*N * sizeof(float16_t), 32);
	float16_t* C1_6 = (float16_t*)_aligned_malloc(M*N * sizeof(float16_t), 32);
	float16_t* C1_7 = (float16_t*)_aligned_malloc(M*N * sizeof(float16_t), 32);
	float16_t* C1_8 = (float16_t*)_aligned_malloc(M*N * sizeof(float16_t), 32);
	float16_t* C1_9 = (float16_t*)_aligned_malloc(M*N * sizeof(float16_t), 32);
	float16_t* C1_10 = (float16_t*)_aligned_malloc(M*N * sizeof(float16_t), 32);
	float16_t* C1_11 = (float16_t*)_aligned_malloc(M*N * sizeof(float16_t), 32);
	float16_t* C1_12 = (float16_t*)_aligned_malloc(M*N * sizeof(float16_t), 32);
	float16_t* C1_13 = (float16_t*)_aligned_malloc(M*N * sizeof(float16_t), 32);
	float16_t* C1_14 = (float16_t*)_aligned_malloc(M*N * sizeof(float16_t), 32);

	for (int i = 0; i < M*padK; i++)
		A[i] = rand() % 10001 / 5000.0f - 1.0f;
	for (int i = 0; i < padK*N; i++)
		Bt[i] = rand() % 10001 / 5000.0f - 1.0f;
	double time = 1;
	clock_t t0 = clock();
	int naive_nIters = __max(1, nIters / 10);
	double navie_mul_count = (double)M*N*K*naive_nIters / (1024.0*1024.0*1024.0);
	for (int i = 0; i < naive_nIters; i++)
	{
		MatMul0_ABt(M, N, K, A, padK, Bt, padK, C0, N);
	}
	clock_t t1 = clock(); time = __max(1e-9, time_scale*(t1 - t0));
	printf("%d x %d x %d, cost = %.3f s, naive gflops = %.3f\n", M, N, K, time, navie_mul_count / time);

	clock_t t1_1 = clock();
	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_16f_align128bit_AnoTrans_Btrans_M1_N1(M, N, K, A, padK, Bt, padK, C1_1, N);
	}
	clock_t t1_2 = clock(); time = time_scale*(t1_2 - t1_1);
	printf("%d x %d x %d, cost = %.3f s, SSE-M1-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_16f_align128bit_AnoTrans_Btrans_M1_N2(M, N, K, A, padK, Bt, padK, C1_2, N);
	}
	clock_t t1_3 = clock(); time = time_scale*(t1_3 - t1_2);
	printf("%d x %d x %d, cost = %.3f s, SSE-M1-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);
	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_16f_align128bit_AnoTrans_Btrans_M1_N4(M, N, K, A, padK, Bt, padK, C1_3, N);
	}
	clock_t t1_4 = clock(); time = time_scale*(t1_4 - t1_3);
	printf("%d x %d x %d, cost = %.3f s, SSE-M1-N4 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_16f_align128bit_AnoTrans_Btrans_M1_N8(M, N, K, A, padK, Bt, padK, C1_4, N);
	}
	clock_t t1_5 = clock(); time = time_scale*(t1_5 - t1_4);
	printf("%d x %d x %d, cost = %.3f s, SSE-M1-N8 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_16f_align128bit_AnoTrans_Btrans_M2_N1(M, N, K, A, padK, Bt, padK, C1_5, N);
	}
	clock_t t1_6 = clock(); time = time_scale*(t1_6 - t1_5);
	printf("%d x %d x %d, cost = %.3f s, SSE-M2-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_16f_align128bit_AnoTrans_Btrans_M2_N2(M, N, K, A, padK, Bt, padK, C1_6, N);
	}
	clock_t t1_7 = clock(); time = time_scale*(t1_7 - t1_6);
	printf("%d x %d x %d, cost = %.3f s, SSE-M2-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_16f_align128bit_AnoTrans_Btrans_M2_N4(M, N, K, A, padK, Bt, padK, C1_7, N);
	}
	clock_t t1_8 = clock(); time = time_scale*(t1_8 - t1_7);
	printf("%d x %d x %d, cost = %.3f s, SSE-M2-N4 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_16f_align128bit_AnoTrans_Btrans_M2_N8(M, N, K, A, padK, Bt, padK, C1_8, N);
	}
	clock_t t1_9 = clock(); time = time_scale*(t1_9 - t1_8);
	printf("%d x %d x %d, cost = %.3f s, SSE-M2-N8 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_16f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, padK, Bt, padK, C1_9, N);
	}
	clock_t t1_10 = clock(); time = time_scale*(t1_10 - t1_9);
	printf("%d x %d x %d, cost = %.3f s, SSE-M4-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_16f_align128bit_AnoTrans_Btrans_M4_N2(M, N, K, A, padK, Bt, padK, C1_10, N);
	}
	clock_t t1_11 = clock(); time = time_scale*(t1_11 - t1_10);
	printf("%d x %d x %d, cost = %.3f s, SSE-M4-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_16f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, padK, Bt, padK, C1_11, N);
	}
	clock_t t1_12 = clock(); time = time_scale*(t1_12 - t1_11);
	printf("%d x %d x %d, cost = %.3f s, SSE-M4-N4 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_16f_align128bit_AnoTrans_Btrans_M8_N1(M, N, K, A, padK, Bt, padK, C1_12, N);
	}
	clock_t t1_13 = clock(); time = time_scale*(t1_13 - t1_12);
	printf("%d x %d x %d, cost = %.3f s, SSE-M8-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_16f_align128bit_AnoTrans_Btrans_M8_N2(M, N, K, A, padK, Bt, padK, C1_13, N);
	}
	clock_t t1_14 = clock(); time = time_scale*(t1_14 - t1_13);
	printf("%d x %d x %d, cost = %.3f s, SSE-M8-N2 gflops = %.3f\n", M, N, K, time, mul_count / time);

	for (int i = 0; i < nIters; i++)
	{
		zq_gemm_16f_align128bit_AnoTrans_Btrans_M16_N1(M, N, K, A, padK, Bt, padK, C1_14, N);
	}
	clock_t t1_15 = clock(); time = time_scale*(t1_15 - t1_14);
	printf("%d x %d x %d, cost = %.3f s, SSE-M16-N1 gflops = %.3f\n", M, N, K, time, mul_count / time);

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

	_aligned_free(A);
	_aligned_free(Bt);
	_aligned_free(C0);
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

}

bool check_value(int M, int N, const float16_t* C1, int ldc1, const float16_t* C2, int ldc2, float16_t thresh, bool show)
{
	int m, n;
	const float16_t* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	float16_t v1, v2;
	bool ret = true;
	for (m = 0, Cptr1 = C1, Cptr2 = C2; m < M; m++, Cptr1 += ldc1, Cptr2 += ldc2)
	{
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n++)
		{
			v1 = *C_c_ptr1;
			v2 = *C_c_ptr2;
			float16_t scale = __max(fabs(v1), fabs(v2));
			float16_t real_thresh = __max(thresh, thresh*scale);
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


#else
#include <stdio.h>
int main(int agrc, const char** argv)
{
	printf("%s only supports arm neon fp16\n", argv[0]);
	return 0;
}
#endif

