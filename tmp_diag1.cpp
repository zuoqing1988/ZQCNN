// temporary diagnostic - not part of the project
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <malloc.h>

#include "zq_gemm_32f_align_c.h"
#include "zq_gemm_32f_align_c_asm.h"

typedef void (*gemm_fn)(int, int, int, const float*, int, const float*, int, float*, int);

#define CANARY 0xC0FFEE12

struct Guard {
	int m, n, k;
	int lda, ldb, ldc;
	volatile int pad0[16];
	float* A;
	float* B;
	float* C;
	volatile int pad1[16];
};

// deliberately big frame so the callee has room to scribble into our frame
static int run_case(const char* tag, int M, int N, int K, int iters)
{
	Guard g;
	g.m = M; g.n = N; g.k = K;
	for (int i = 0; i < 16; i++) { g.pad0[i] = CANARY; g.pad1[i] = CANARY; }

	int padK = (K + 7) / 8 * 8;
	int lda = padK, ldb = padK, ldc = (N + 7) / 8 * 8;
	g.lda = lda; g.ldb = ldb; g.ldc = ldc;

	size_t na = (size_t)M * lda, nb = (size_t)N * ldb, nc = (size_t)M * ldc;
	const int SENT = 64;
	float* A = (float*)_aligned_malloc((na + SENT) * sizeof(float), 32);
	float* B = (float*)_aligned_malloc((nb + SENT) * sizeof(float), 32);
	float* C = (float*)_aligned_malloc((nc + SENT) * sizeof(float), 32);
	g.A = A; g.B = B; g.C = C;
	for (size_t i = 0; i < na; i++) A[i] = 1.0f;
	for (size_t i = 0; i < nb; i++) B[i] = 2.0f;
	for (size_t i = 0; i < nc; i++) C[i] = 0.0f;
	for (int i = 0; i < SENT; i++) { A[na + i] = -777.0f; B[nb + i] = -777.0f; C[nc + i] = -777.0f; }

	gemm_fn fn = zq_gemm_32f_AnoTrans_Btrans_auto_asm;
	for (int it = 0; it < iters; it++)
		fn(M, N, K, A, lda, B, ldb, C, ldc);

	int bad = 0;
	for (int i = 0; i < 16; i++) {
		if (g.pad0[i] != CANARY) { printf("  %s: pad0[%d] CLOBBERED = %08X\n", tag, i, g.pad0[i]); bad = 1; }
		if (g.pad1[i] != CANARY) { printf("  %s: pad1[%d] CLOBBERED = %08X\n", tag, i, g.pad1[i]); bad = 1; }
	}
	if (g.m != M || g.n != N || g.k != K || g.lda != lda || g.ldb != ldb || g.ldc != ldc || g.A != A || g.B != B || g.C != C) {
		printf("  %s: scalars/ptrs CLOBBERED m=%d n=%d k=%d lda=%d ldb=%d ldc=%d A=%p B=%p C=%p\n",
			tag, g.m, g.n, g.k, g.lda, g.ldb, g.ldc, (void*)g.A, (void*)g.B, (void*)g.C);
		bad = 2;
	}
	int sent_bad = 0;
	for (int i = 0; i < SENT; i++) {
		if (A[na + i] != -777.0f) { if (!sent_bad) printf("  %s: A sentinel at +%d = %f\n", tag, i, A[na + i]); sent_bad++; }
		if (B[nb + i] != -777.0f) { if (!sent_bad) printf("  %s: B sentinel at +%d = %f\n", tag, i, B[nb + i]); sent_bad++; }
		if (C[nc + i] != -777.0f) { if (!sent_bad) printf("  %s: C sentinel at +%d = %f\n", tag, i, C[nc + i]); sent_bad++; }
	}
	if (sent_bad) bad = 3;

	_aligned_free(A); _aligned_free(B); _aligned_free(C);
	printf("%-14s %4dx%-4dx%-4d iters=%-6d  %s\n", tag, M, N, K, iters,
		bad == 0 ? "OK" : (bad == 1 ? "STACK SMASH" : (bad == 2 ? "SCALAR SMASH" : "HEAP OOB")));
	return bad;
}

int main()
{
	printf("A=%p B=%p (approx stack)\n", (void*)&printf, (void*)main);
	int bad = 0;
	bad |= run_case("1x1x1",    1,   1,   1, 100);
	bad |= run_case("1x8x64",   1,   8,  64, 100);
	bad |= run_case("8x1x64",   8,   1,  64, 100);
	bad |= run_case("2x4x8",    2,   4,   8, 100);
	bad |= run_case("3x7x5",    3,   7,   5, 100);
	bad |= run_case("7x5x13",   7,   5,  13, 100);
	bad |= run_case("5x9x17",   5,   9,  17, 100);
	bad |= run_case("13x11x7", 13,  11,   7, 100);
	bad |= run_case("17x19x27",17,  19,  27, 100);
	bad |= run_case("6x6x6",    6,   6,   6, 100);
	bad |= run_case("4x4x4",    4,   4,   4, 100);
	bad |= run_case("16x8x32", 16,   8,  32, 100);
	bad |= run_case("9x17x33",  9,  17,  33, 100);
	bad |= run_case("32x32x32",32,  32,  32, 10);
	bad |= run_case("64x64x128",64, 64, 128,  3);
	bad |= run_case("100x70x50",100,70, 50,  3);
	bad |= run_case("128x128x256",128,128,256,1);
	bad |= run_case("313x32x28",313,32, 28,  1);
	printf("overall: %s\n", bad ? "BAD" : "ALL OK");
	return bad;
}
