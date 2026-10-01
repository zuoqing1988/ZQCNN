// temporary diagnostic: find how far past the logical end each buffer gets written
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <malloc.h>
#include "zq_gemm_32f_align_c.h"
#include "zq_gemm_32f_align_c_asm.h"

typedef void (*gemm_fn)(int, int, int, const float*, int, const float*, int, float*, int);

struct Buf { float* p; size_t cap, used; };

static void probe(const char* nm, Buf& b, int M, int N, int K, int lda, int ldb, int ldc, const char* which)
{
	if (strcmp(nm, which) != 0) return;
	int first = -1, last = -1, cnt = 0;
	for (size_t i = b.used; i < b.cap; i++) {
		if (b.p[i] != (float)0xA5A5A5A5) {
			if (first < 0) first = (int)i;
			last = (int)i; cnt++;
		}
	}
	if (cnt)
		printf("    OOB %dx%dx%d buf=%s : %d floats past end, first=+%d last=+%d (bytes past end %d..%d)\n",
			M, N, K, nm, cnt, first, last, first * 4, (last + 1) * 4);
	else
		printf("    ok  %dx%dx%d buf=%s\n", M, N, K, nm);
}

static int run(int M, int N, int K)
{
	int padK = (K + 7) / 8 * 8;
	int lda = padK, ldb = padK, ldc = (N + 7) / 8 * 8;
	size_t na = (size_t)M * lda, nb = (size_t)N * ldb, nc = (size_t)M * ldc;
	const size_t SLACK = 4096;
	Buf A{ (float*)malloc((na + SLACK) * 4), na + SLACK, na };
	Buf B{ (float*)malloc((nb + SLACK) * 4), nb + SLACK, nb };
	Buf C{ (float*)malloc((nc + SLACK) * 4), nc + SLACK, nc };
	Buf R{ (float*)malloc((nc + SLACK) * 4), nc + SLACK, nc };
	for (size_t i = 0; i < A.cap; i++) A.p[i] = (float)0xA5A5A5A5;
	for (size_t i = 0; i < B.cap; i++) B.p[i] = (float)0xA5A5A5A5;
	for (size_t i = 0; i < C.cap; i++) C.p[i] = (float)0xA5A5A5A5;
	for (size_t i = 0; i < R.cap; i++) R.p[i] = (float)0xA5A5A5A5;
	for (size_t i = 0; i < na; i++) A.p[i] = 0.5f;
	for (size_t i = 0; i < nb; i++) B.p[i] = 0.25f;
	for (size_t i = 0; i < nc; i++) { C.p[i] = 0; R.p[i] = 0; }

	gemm_fn fn = zq_gemm_32f_AnoTrans_Btrans_auto_asm;
	for (int it = 0; it < 3; it++) fn(M, N, K, A.p, lda, B.p, ldb, R.p, ldc);

	printf("  %dx%dx%d  (lda=%d ldb=%d ldc=%d, used A=%zu B=%zu C=%zu)\n", M, N, K, lda, ldb, ldc, na, nb, nc);
	probe("A", A, M, N, K, lda, ldb, ldc, "A");
	probe("B", B, M, N, K, lda, ldb, ldc, "B");
	probe("R", R, M, N, K, lda, ldb, ldc, "R");

	// numerical check too
	float* Rc = (float*)malloc(nc * 4);
	zq_gemm_32f_AnoTrans_Btrans_auto(M, N, K, A.p, lda, B.p, ldb, C.p, ldc);
	double mx = 0;
	for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
		double d = fabs((double)C.p[(size_t)m * ldc + n] - (double)R.p[(size_t)m * ldc + n]);
		if (d > mx) mx = d;
	}
	printf("    maxdiff = %.4e\n", mx);
	free(Rc); free(C.p); free(B.p); free(A.p); free(R.p);
	return 0;
}

int main()
{
	run(16, 16, 16);
	run(32, 32, 32);
	run(64, 64, 64);
	run(128, 128, 128);
	run(1, 8, 64);
	run(8, 1, 64);
	run(2, 4, 8);
	run(7, 5, 13);
	run(5, 9, 17);
	run(3, 7, 5);
	run(13, 11, 7);
	run(17, 19, 27);
	run(6, 6, 6);
	run(4, 4, 4);
	run(16, 8, 32);
	run(9, 17, 33);
	run(256, 256, 256);
	run(1024, 1024, 1024);
	run(1, 1024, 1024);
	run(1024, 1, 1024);
	run(313, 32, 28);
	run(33, 17, 65);
	return 0;
}
