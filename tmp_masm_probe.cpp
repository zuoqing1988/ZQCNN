/* 临时排查：直接调用 MASM 微内核，隔离 driver 与 kernel */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <malloc.h>

extern "C" void zq_gemm_32f_asm_core_m2n4(const float* a0, const float* a1, const float* b0,
	int k8, int s1, int s3, float* c0, float* c1);
extern "C" void zq_gemm_32f_asm_core_m1n8(const float* a0, const float* b0, const float* b4,
	int k8, int s1, int s3, float* c0);
extern "C" void zq_gemm_32f_asm_core_m1n4(const float* a0, const float* b0,
	int k8, int s1, int s3, float* c0);

enum { K = 8, LD = 64 };

static void init(float* A, float* B, int nrows_a, int nrows_b)
{
	memset(A, 0, (size_t)LD * nrows_a * sizeof(float));
	memset(B, 0, (size_t)LD * nrows_b * sizeof(float));
	for (int i = 0; i < nrows_a; i++)
		for (int k = 0; k < K; k++)
			A[i * LD + k] = (float)(i * 8 + k);
	for (int j = 0; j < nrows_b; j++)
		for (int k = 0; k < K; k++)
			B[j * LD + k] = (float)(j * 2 + k);
}

int main()
{
	float* A = (float*)_aligned_malloc(LD * 16 * sizeof(float), 32);
	float* B = (float*)_aligned_malloc(LD * 16 * sizeof(float), 32);
	float* C = (float*)_aligned_malloc(LD * 16 * sizeof(float), 32);
	float expect[8];

	init(A, B, 4, 8);
	memset(C, 0, LD * 16 * sizeof(float));
	zq_gemm_32f_asm_core_m2n4(A + 0, A + LD, B + 0, 1, LD * 4, 3 * LD * 4, C + 0, C + LD);
	for (int j = 0; j < 4; j++) { double s = 0; for (int k = 0; k < K; k++) s += (double)A[k] * B[j * LD + k]; expect[j] = (float)s; }
	printf("m2n4 row0: C=%.1f,%.1f,%.1f,%.1f  expect=%.1f,%.1f,%.1f,%.1f  A0=%.1f  %s", 
		C[0], C[1], C[2], C[3], expect[0], expect[1], expect[2], expect[3], A[0],
		(fabs(C[0] - expect[0]) < 1e-3 && fabs(A[0] - 0.0f) < 1e-6) ? "OK" : "BAD");
	printf("\n");

	init(A, B, 4, 8);
	memset(C, 0, LD * 16 * sizeof(float));
	zq_gemm_32f_asm_core_m1n8(A + 0, B + 0, B + 4 * LD, 1, LD * 4, 3 * LD * 4, C + 0);
	for (int j = 0; j < 8; j++) { double s = 0; for (int k = 0; k < K; k++) s += (double)A[k] * B[j * LD + k]; expect[j] = (float)s; }
	printf("m1n8      : C=%.1f,%.1f,%.1f,%.1f | %.1f,%.1f,%.1f,%.1f  expect=%.1f..%.1f  A0=%.1f  %s",
		C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7], expect[0], expect[7], A[0],
		(fabs(C[0] - expect[0]) < 1e-3 && fabs(C[4] - expect[4]) < 1e-3 && fabs(A[0] - 0.0f) < 1e-6) ? "OK" : "BAD");
	printf("\n");

	init(A, B, 4, 8);
	memset(C, 0, LD * 16 * sizeof(float));
	zq_gemm_32f_asm_core_m1n4(A + 0, B + 0, 1, LD * 4, 3 * LD * 4, C + 0);
	for (int j = 0; j < 4; j++) { double s = 0; for (int k = 0; k < K; k++) s += (double)A[k] * B[j * LD + k]; expect[j] = (float)s; }
	printf("m1n4      : C=%.1f,%.1f,%.1f,%.1f  expect=%.1f,%.1f,%.1f,%.1f  A0=%.1f  %s",
		C[0], C[1], C[2], C[3], expect[0], expect[1], expect[2], expect[3], A[0],
		(fabs(C[0] - expect[0]) < 1e-3 && fabs(A[0] - 0.0f) < 1e-6) ? "OK" : "BAD");
	printf("\n");

	_aligned_free(A); _aligned_free(B); _aligned_free(C);
	return 0;
}
