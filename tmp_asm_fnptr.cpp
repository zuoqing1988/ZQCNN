/* 临时排查：函数指针方式调用 + 栈 canary */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <malloc.h>
#include <chrono>

#include "zq_gemm_32f_align_c.h"
#include "zq_gemm_32f_align_c_asm.h"

#define PAD 64

typedef void (*gemm_fn)(int, int, int, const float*, int, const float*, int, float*, int);

static double time_gemm(gemm_fn fn, int M, int N, int K, const float* A, const float* Bt, float* C,
	int lda, int ldb, int ldc, int iters)
{
	fn(M, N, K, A, lda, Bt, ldb, C, ldc);
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < iters; i++)
		fn(M, N, K, A, lda, Bt, ldb, C, ldc);
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	double sec = std::chrono::duration<double>(t1 - t0).count() / iters;
	return 2.0 * M * N * K / sec / 1e9;
}

int main(int argc, char** argv)
{
	int M = argc > 1 ? atoi(argv[1]) : 16;
	int N = argc > 2 ? atoi(argv[2]) : 16;
	int K = argc > 3 ? atoi(argv[3]) : 16;
	int lda = (K + 7) / 8 * 8, ldb = lda, ldc = (N + 7) / 8 * 8;
	size_t na = (size_t)lda * M, nb = (size_t)ldb * N, nc = (size_t)ldc * M;
	volatile double canary = 12345.6789;

	float* A = (float*)_aligned_malloc((na + PAD) * sizeof(float), 32);
	float* B = (float*)_aligned_malloc((nb + PAD) * sizeof(float), 32);
	float* C = (float*)_aligned_malloc((nc + PAD) * sizeof(float), 32);
	for (size_t i = 0; i < na + PAD; i++) A[i] = 0.0f;
	for (size_t i = 0; i < nb + PAD; i++) B[i] = 0.0f;
	for (size_t i = 0; i < nc + PAD; i++) C[i] = 0.0f;
	for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) A[(size_t)i * lda + k] = (float)((i + k) % 5) - 2.0f;
	for (int j = 0; j < N; j++) for (int k = 0; k < K; k++) B[(size_t)j * ldb + k] = (float)((j * 2 + k) % 7) - 3.0f;

	double gi = time_gemm(zq_gemm_32f_AnoTrans_Btrans_auto, M, N, K, A, B, C, lda, ldb, ldc, 50);
	printf("intrinsic via fnptr: %.2f GF/s  canary=%.1f  C0=%.3f", gi, (double)canary, C[0]);
	printf("\n");
	double ga = time_gemm(zq_gemm_32f_AnoTrans_Btrans_auto_asm, M, N, K, A, B, C, lda, ldb, ldc, 50);
	printf("asm       via fnptr: %.2f GF/s  canary=%.1f  C0=%.3f", ga, (double)canary, C[0]);
	printf("\n");

	_aligned_free(A); _aligned_free(B); _aligned_free(C);
	return 0;
}
