// temporary diagnostic - faithful mirror of SampleGEMMCompare with per-stage sentinel checks
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include "zq_gemm_32f_align_c.h"
#include "zq_gemm_32f_align_c_asm.h"

static const size_t MAX_ELEM = 4u * 1024 * 1024 + 1024;
static float g_A[MAX_ELEM];
static float g_B[MAX_ELEM];
static float g_C[MAX_ELEM];
static float g_R[MAX_ELEM];
__declspec(align(32))
static float g_pad_a[64], g_pad_b[64], g_pad_c[64], g_pad_r[64];

static int align8(int x) { return (x + 7) / 8 * 8; }
static void fill_random(float* v, size_t n, unsigned seed)
{
	unsigned s = seed;
	for (size_t i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; v[i] = ((float)((s >> 8) & 0xFFFF) / 32768.0f) - 1.0f; }
}
static double diff_at(const float* a, const float* b, int M, int N, int ldc)
{
	double m = 0;
	for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
		double d = (double)a[(size_t)i * ldc + j] - (double)b[(size_t)i * ldc + j];
		if (d < 0) d = -d; if (d > m) m = d;
	}
	return m;
}
typedef void (*gemm_fn)(int, int, int, const float*, int, const float*, int, float*, int);
static double now_sec()
{
	std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
	return std::chrono::duration<double>(t.time_since_epoch()).count();
}
static double time_gemm(gemm_fn fn, int M, int N, int K, const float* A, const float* Bt, float* C,
	int lda, int ldb, int ldc, int iters)
{
	double t0, t1;
	fn(M, N, K, A, lda, Bt, ldb, C, ldc);
	t0 = now_sec();
	for (int i = 0; i < iters; i++) fn(M, N, K, A, lda, Bt, ldb, C, ldc);
	t1 = now_sec();
	double sec = (t1 - t0) / iters;
	return 2.0 * (double)M * (double)N * (double)K / sec / 1e9;
}

static int pad_check(const char* stage)
{
	const float* pads[4] = { g_pad_a, g_pad_b, g_pad_c, g_pad_r };
	const char* nm[4] = { "pad_a", "pad_b", "pad_c", "pad_r" };
	int bad = 0;
	for (int q = 0; q < 4; q++)
		for (int i = 0; i < 64; i++)
			if (pads[q][i] != 0.0f) {
				if (!bad) printf("      %-22s DIRTY: %s[%d]=%g raw=%08X\n", stage, nm[q], i,
					(double)pads[q][i], *(const unsigned*)&pads[q][i]);
				bad = 1;
			}
	if (!bad) printf("      %-22s clean\n", stage);
	fflush(stdout);
	return bad;
}

struct Shape { int M, N, K; };
static Shape shapes[] = {
	{ 16, 16, 16 }, { 32, 32, 32 }, { 64, 64, 64 }, { 128, 128, 128 },
	{ 256, 256, 256 }, { 512, 512, 512 }, { 1024, 1024, 1024 },
	{ 1, 1024, 1024 }, { 1024, 1, 1024 },
	{ 8, 2048, 2048 }, { 2048, 8, 2048 },
	{ 128, 128, 4096 }, { 4096, 128, 128 },
	{ 1152, 256, 1152 }, { 512, 512, 2048 },
	{ 7, 5, 13 }, { 33, 17, 65 },
};

int main()
{
	printf("g_A=%p g_B=%p g_C=%p g_R=%p pad_a=%p pad_b=%p pad_c=%p pad_r=%p (end=%p)\n",
		(void*)g_A, (void*)g_B, (void*)g_C, (void*)g_R,
		(void*)g_pad_a, (void*)g_pad_b, (void*)g_pad_c, (void*)g_pad_r,
		(void*)(g_pad_r + 64));
	const int num_shapes = (int)(sizeof(shapes) / sizeof(shapes[0]));
	for (int s = 0; s < num_shapes; s++)
	{
		const int M = shapes[s].M, N = shapes[s].N, K = shapes[s].K;
		const int lda = align8(K), ldb = align8(K), ldc = align8(N);
		const size_t na = (size_t)lda * M, nb = (size_t)ldb * N, nc = (size_t)ldc * M;
		printf("  %dx%dx%d  lda=%d ldb=%d ldc=%d\n", M, N, K, lda, ldb, ldc);
		fill_random(g_A, na, 12345u + (unsigned)s);
		fill_random(g_B, nb, 67890u + (unsigned)s);
		for (int i = 0; i < M; i++) for (int k = K; k < lda; k++) g_A[(size_t)i * lda + k] = 0;
		for (int j = 0; j < N; j++) for (int k = K; k < ldb; k++) g_B[(size_t)j * ldb + k] = 0;
		memset(g_C, 0, nc * sizeof(float));
		memset(g_R, 0, nc * sizeof(float));
		memset(g_pad_a, 0, sizeof(g_pad_a));
		memset(g_pad_b, 0, sizeof(g_pad_b));
		memset(g_pad_c, 0, sizeof(g_pad_c));
		memset(g_pad_r, 0, sizeof(g_pad_r));
		pad_check("after memset");

		zq_gemm_32f_AnoTrans_Btrans_auto(M, N, K, g_A, lda, g_B, ldb, g_C, ldc);
		pad_check("after intrinsic");
		zq_gemm_32f_AnoTrans_Btrans_auto_asm(M, N, K, g_A, lda, g_B, ldb, g_R, ldc);
		pad_check("after asm");
		double err_asm = diff_at(g_C, g_R, M, N, ldc);
		pad_check("after diff_at");

		double work = 2.0 * M * N * K;
		int iters = work > 2e8 ? 3 : (work > 2e7 ? 10 : 50);
		double g_intr = time_gemm(zq_gemm_32f_AnoTrans_Btrans_auto, M, N, K, g_A, g_B, g_C, lda, ldb, ldc, iters);
		pad_check("after time_gemm intr");
		double g_asm = time_gemm(zq_gemm_32f_AnoTrans_Btrans_auto_asm, M, N, K, g_A, g_B, g_R, lda, ldb, ldc, iters);
		pad_check("after time_gemm asm");
		printf("    => intr=%g asm=%g err=%g   localdoubles_ok=%d%d\n", g_intr, g_asm, err_asm,
			g_intr == g_intr, g_asm == g_asm);
		fflush(stdout);
	}
	return 0;
}
