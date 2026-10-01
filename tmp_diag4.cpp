/*
 * SampleGEMMAsmCompare
 *
 * 手写内联汇编版 GEMM 与 intrinsic 版 GEMM 的 A/B 对比:
 *   1) 正确性: 同一组随机数据分别调用
 *        zq_gemm_32f_AnoTrans_Btrans_auto      (intrinsic 版)
 *        zq_gemm_32f_AnoTrans_Btrans_auto_asm  (汇编内核版)
 *      并额外用一个 double 精度的朴素三重循环做参照, 比较三者的最大绝对误差。
 *   2) 性能: 分别计时, 打印 GFLOP/s。
 *
 * 覆盖的尺寸包含各种尾部情况:
 *   M/N 不是 4 (或 8) 的倍数, K 不是 8 的倍数, M=1, N=1, K=1 等退化情况。
 *
 * 缓冲分配遵循 intrinsic 版的约定 (它用 _mm256_load_ps 读 A/Bt):
 *   行长度取 padK = ceil(K/8)*8, 补零, 起始地址 32 字节对齐。
 *   汇编内核只读 [0, K), 不依赖补零, 但这里按同一约定分配以便对比。
 *
 * 不依赖 OpenCV。
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

#include "math/zq_gemm_32f_align_c.h"
#include "math/zq_gemm_32f_align_c_asm.h"

static unsigned int g_rand_state = 123456789u;

static float randf()
{
	g_rand_state = g_rand_state * 1103515245u + 12345u;
	unsigned int v = (g_rand_state >> 9) & 0x7fffffu;
	return (float)((double)v / (double)0x7fffffu * 2.0 - 1.0);
}

#define CAN 0x5150DEAD
static volatile int g_can_pad[64];
static void time_gemm_asm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc, int iters)
{
	int can[64];
	for (int q = 0; q < 64; q++) can[q] = CAN;
	volatile double junk[16];
	for (int q = 0; q < 16; q++) junk[q] = (double)q * 1.5;
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < iters; i++)
		zq_gemm_32f_AnoTrans_Btrans_auto_asm(M, N, K, A, lda, Bt, ldb, C, ldc);
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	double sec = std::chrono::duration<double>(t1 - t0).count();
	int nbad = 0;
	for (int q = 0; q < 64; q++) if (can[q] != CAN) { if (!nbad) printf("[CANARY sm %d @%d=%08X]", M, q, can[q]); nbad++; }
	for (int q = 0; q < 16; q++) if (junk[q] != (double)q * 1.5) { printf("[JUNK sm %d @%d=%g]", M, q, junk[q]); nbad++; }
	printf("[sm %dx%dx%d A=%p B=%p C=%p t0=%lld t1=%lld]", M,N,K,(void*)A,(void*)Bt,(void*)C,(long long)t0.time_since_epoch().count(),(long long)t1.time_since_epoch().count());
	double flop = 2.0 * (double)M * (double)N * (double)K * (double)iters;
	printf("        asm  %8.2f GF/s", flop / sec / 1e9);
}

static void time_gemm_ref(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc, int iters)
{
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < iters; i++)
		zq_gemm_32f_AnoTrans_Btrans_auto(M, N, K, A, lda, Bt, ldb, C, ldc);
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	double sec = std::chrono::duration<double>(t1 - t0).count();
	double flop = 2.0 * (double)M * (double)N * (double)K * (double)iters;
	printf("intr %8.2f GF/s", flop / sec / 1e9);
}

static void naive_gemm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, double* C, int ldc)
{
	for (int m = 0; m < M; m++)
	{
		for (int n = 0; n < N; n++)
		{
			double sum = 0;
			for (int k = 0; k < K; k++)
				sum += (double)A[m * lda + k] * (double)Bt[n * ldb + k];
			C[m * ldc + n] = sum;
		}
	}
}

struct Case { int M, N, K, iters; };

int main()
{
	static const Case cases[] = {
		{   1,  1,   1, 200000 },
		{   1,  8,  64,  50000 },
		{   8,  1,  64,  50000 },
		{   2,  4,   8, 100000 },
		{   3,  7,   5, 100000 },
		{   7,  5,  13, 100000 },
		{   5,  9,  17,  50000 },
		{  13, 11,   7,  50000 },
		{  17, 19,  27,  20000 },
		{   6,  6,   6, 100000 },
		{   4,  4,   4, 100000 },
		{  16,  8,  32,  20000 },
		{   9, 17,  33,  20000 },
		{  32, 32,  32,   5000 },
		{  64, 64, 128,   2000 },
		{ 100, 70,  50,   2000 },
		{ 128,128, 256,    500 },
		{ 313,  32,  28,    500 },
	};

	const int case_num = (int)(sizeof(cases) / sizeof(cases[0]));
	int failed = 0;
	double worst = 0;

	printf("=== GEMM: hand-written assembly vs intrinsic ===\n");
	printf("%5s %5s %5s | %12s %12s | ", "M", "N", "K", "maxerr(a-asm)", "maxerr(a-dbl)");

	for (int ci = 0; ci < case_num; ci++)
	{
		const int M = cases[ci].M, N = cases[ci].N, K = cases[ci].K;
		const int iters = cases[ci].iters;
		const int padK = (K + 7) / 8 * 8;
		const int lda = padK, ldb = padK, ldc = N;
		const size_t asz = (size_t)M * lda, bsz = (size_t)N * ldb, csz = (size_t)M * ldc;

		float* A = (float*)_aligned_malloc(asz * sizeof(float), 32);
		float* Bt = (float*)_aligned_malloc(bsz * sizeof(float), 32);
		float* C1 = (float*)_aligned_malloc(csz * sizeof(float), 32);
		float* C2 = (float*)_aligned_malloc(csz * sizeof(float), 32);
		std::vector<double> Cref(csz ? csz : 1, 0.0);

		if (!A || !Bt || !C1 || !C2)
		{
			printf("alloc failed\n");
			return 1;
		}

		for (size_t i = 0; i < asz; i++) A[i] = 0;
		for (size_t i = 0; i < bsz; i++) Bt[i] = 0;
		for (int m = 0; m < M; m++)
			for (int k = 0; k < K; k++)
				A[(size_t)m * lda + k] = randf();
		for (int n = 0; n < N; n++)
			for (int k = 0; k < K; k++)
				Bt[(size_t)n * ldb + k] = randf();

		/* C 预置成非 0 的垃圾值: 顺便验证两版都是"覆盖写入"而不是累加 */
		for (size_t i = 0; i < csz; i++) { C1[i] = -12345.0f; C2[i] = -12345.0f; }

		zq_gemm_32f_AnoTrans_Btrans_auto(M, N, K, A, lda, Bt, ldb, C1, ldc);
		zq_gemm_32f_AnoTrans_Btrans_auto_asm(M, N, K, A, lda, Bt, ldb, C2, ldc);
		naive_gemm(M, N, K, A, lda, Bt, ldb, &Cref[0], ldc);

		double e_asm = 0, e_ref = 0, e_scale = 1e-6;
		for (int m = 0; m < M; m++)
		{
			for (int n = 0; n < N; n++)
			{
				size_t idx = (size_t)m * ldc + n;
				double d = fabs((double)C1[idx] - (double)C2[idx]);
				double r = fabs((double)C1[idx] - Cref[idx]);
				if (d > e_asm) e_asm = d;
				if (r > e_ref) e_ref = r;
				if (fabs(Cref[idx]) > e_scale) e_scale = fabs(Cref[idx]);
			}
		}
		if (e_asm > worst) worst = e_asm;

		printf("\n%5d %5d %5d | %12.3e %12.3e | ", M, N, K, e_asm, e_ref);
		fflush(stdout);
		time_gemm_ref(M, N, K, A, lda, Bt, ldb, C1, ldc, iters);
		time_gemm_asm(M, N, K, A, lda, Bt, ldb, C2, ldc, iters);
		int bad = (e_asm > 1e-3) || (e_ref > 1e-3) || (e_asm > e_scale * 1e-4);
		if (bad) failed++;
		printf(" %s\n", bad ? "  <== FAIL" : "");
		fflush(stdout);

		_aligned_free(A);
		_aligned_free(Bt);
		_aligned_free(C1);
		_aligned_free(C2);
	}

	printf("--------------------------------------------------------------\n");
	printf("worst |intrinsic - asm| over all cases = %.6e\n", worst);
	printf("result: %s (%d case(s) failed)\n", failed == 0 ? "PASS" : "FAIL", failed);
	return failed == 0 ? 0 : 1;
}
