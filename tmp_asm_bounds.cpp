/* 临时排查：auto_asm 全矩阵正确性 + 越界/栈 canary（Windows 验证用） */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <malloc.h>

#include "zq_gemm_32f_align_c.h"
#include "zq_gemm_32f_align_c_asm.h"

#define PAD 64

int main(int argc, char** argv)
{
	int M = argc > 1 ? atoi(argv[1]) : 32;
	int N = argc > 2 ? atoi(argv[2]) : 32;
	int K = argc > 3 ? atoi(argv[3]) : 32;
	int lda = (K + 7) / 8 * 8, ldb = lda, ldc = (N + 7) / 8 * 8;
	size_t na = (size_t)lda * M, nb = (size_t)ldb * N, nc = (size_t)ldc * M;
	volatile double canary = 12345.6789;

	float* A = (float*)_aligned_malloc((na + PAD) * sizeof(float), 32);
	float* B = (float*)_aligned_malloc((nb + PAD) * sizeof(float), 32);
	float* C = (float*)_aligned_malloc((nc + PAD) * sizeof(float), 32);
	float* R = (float*)_aligned_malloc((nc + PAD) * sizeof(float), 32);
	for (size_t i = 0; i < na + PAD; i++) A[i] = 0.0f;
	for (size_t i = 0; i < nb + PAD; i++) B[i] = 0.0f;
	for (size_t i = 0; i < nc + PAD; i++) { C[i] = 0.0f; R[i] = 0.0f; }
	for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) A[(size_t)i * lda + k] = (float)((i + k) % 5) - 2.0f;
	for (int j = 0; j < N; j++) for (int k = 0; k < K; k++) B[(size_t)j * ldb + k] = (float)((j * 2 + k) % 7) - 3.0f;

	zq_gemm_32f_AnoTrans_Btrans_auto(M, N, K, A, lda, B, ldb, R, ldc);
	zq_gemm_32f_AnoTrans_Btrans_auto_asm(M, N, K, A, lda, B, ldb, C, ldc);

	double worst = 0; int wi = -1, wj = -1;
	for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
		double d = fabs((double)R[(size_t)i * ldc + j] - C[(size_t)i * ldc + j]);
		if (d > worst) { worst = d; wi = i; wj = j; }
	}
	int oob_a = 0, oob_b = 0, oob_c = 0, oob_r = 0;
	for (size_t i = na; i < na + PAD; i++) if (A[i] != 0.0f) oob_a++;
	for (size_t i = nb; i < nb + PAD; i++) if (B[i] != 0.0f) oob_b++;
	for (size_t i = nc; i < nc + PAD; i++) if (C[i] != 0.0f) oob_c++;
	for (size_t i = nc; i < nc + PAD; i++) if (R[i] != 0.0f) oob_r++;

	memset(C, 0, (nc + PAD) * sizeof(float));
	for (int it = 0; it < 50; it++)
		zq_gemm_32f_AnoTrans_Btrans_auto_asm(M, N, K, A, lda, B, ldb, C, ldc);
	int oob_x50 = 0;
	for (size_t i = nc; i < nc + PAD; i++) if (C[i] != 0.0f) oob_x50++;

	fprintf(stderr, "M=%d N=%d K=%d oobA=%d oobB=%d oobC=%d oobR=%d oobX50=%d canary=%.1f", M, N, K, oob_a, oob_b, oob_c, oob_r, oob_x50, (double)canary);
	fprintf(stderr, " maxdiff=%.3e at (%d,%d) %s", worst, wi, wj, (worst == 0 && !oob_a && !oob_b && !oob_c && !oob_r && !oob_x50) ? "OK" : "BAD");
	fprintf(stderr, "
");

	_aligned_free(A); _aligned_free(B); _aligned_free(C); _aligned_free(R);
	return 0;
}
