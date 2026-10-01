/*
 * SampleGEMMCompare.cpp
 *
 * ZQ_GEMM 内核选型基准：把四条路径放在同一组矩阵尺寸上对比
 *   1) ZQ_GEMM intrinsic   zq_gemm_32f_AnoTrans_Btrans_auto
 *   2) ZQ_GEMM inline asm  zq_gemm_32f_AnoTrans_Btrans_auto_asm
 *   3) Intel MKL           cblas_sgemm(RowMajor, NoTrans, Trans)
 *   4) OpenBLAS            cblas_sgemm(RowMajor, NoTrans, Trans)
 *
 * MKL / OpenBLAS 全部用运行时动态加载（LoadLibrary/dlopen + GetProcAddress/dlsym），
 * 本机没装就跳过，不在链接期绑死某个 BLAS。MKL 加载前固定
 * MKL_THREADING_LAYER=SEQUENTIAL：默认线程层要么找不到 libgomp / libiomp5md 直接崩，
 * 要么是多线程，和单线程的 ZQ_GEMM 没法直接比。
 *
 * 语义与 ZQ_GEMM 一致：
 *   C[i][j] = sum_k A[i*lda+k] * Bt[j*ldb+k]
 *   A: M x K, Bt: N x K, C: M x N, 全部行主序, C 被覆盖写入
 * 对应 CBLAS 行主序就是 C = A * Bt^T，即 transB = CblasTrans。
 *
 * 对齐契约：ZQ_GEMM 的 intrinsic 内核用对齐访存（vmovaps），与 ZQCNN 内部一致——
 * 缓冲区 32 字节对齐，lda/ldb/ldc 为 8 的倍数，K 方向 padding 必须补 0。
 * 汇编版用非对齐访存，要求更松，但基准按 intrinsic 的契约分配，保证公平。
 *
 * 用法：
 *   SampleGEMMCompare                       默认尺寸组，自动探测 MKL / OpenBLAS
 *   SampleGEMMCompare --no-mkl              跳过 MKL
 *   SampleGEMMCompare <mkl_lib> [ob_lib]    指定动态库路径
 * 环境变量 ZQCNN_MKL_LIB / ZQCNN_OPENBLAS_LIB 优先级最高。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <chrono>

#if defined(_WIN32)
#  include <windows.h>
   typedef void* lib_handle_t;
   static lib_handle_t lib_open(const char* p) { return (lib_handle_t)LoadLibraryA(p); }
   static void* lib_sym(lib_handle_t h, const char* n) { return (void*)GetProcAddress((HMODULE)h, n); }
#else
#  include <dlfcn.h>
#  include <unistd.h>
   typedef void* lib_handle_t;
   static lib_handle_t lib_open(const char* p) { return dlopen(p, RTLD_NOW | RTLD_LOCAL); }
   static void* lib_sym(lib_handle_t h, const char* n) { return dlsym(h, n); }
#endif

#include "zq_gemm_32f_align_c.h"
#include "zq_gemm_32f_align_c_asm.h"

typedef void (*gemm_fn)(int, int, int, const float*, int, const float*, int, float*, int);
typedef void (*cblas_sgemm_fn)(int, int, int, int, int, int, float, const float*, int,
	const float*, int, float, float*, int);

enum { CblasRowMajor = 101, CblasNoTrans = 111, CblasTrans = 112 };

struct BlasLib
{
	cblas_sgemm_fn sgemm;
	std::string name;
	bool ok;
	BlasLib() : sgemm(NULL), ok(false) {}
};

static void force_mkl_sequential()
{
#if defined(_WIN32)
	_putenv_s("MKL_THREADING_LAYER", "SEQUENTIAL");
#else
	setenv("MKL_THREADING_LAYER", "SEQUENTIAL", 1);
#endif
}

static bool load_blas(BlasLib& out, const char* const* candidates, int num_candidates,
	lib_handle_t (*open_fn)(const char*), void* (*sym_fn)(lib_handle_t, const char*))
{
	for (int i = 0; i < num_candidates; i++)
	{
		if (candidates[i] == NULL || candidates[i][0] == '\0')
			continue;
		lib_handle_t h = open_fn(candidates[i]);
		if (h == NULL)
			continue;
		void* s = sym_fn(h, "cblas_sgemm");
		if (s == NULL)
		{
			printf("  [%s] 打开成功但找不到 cblas_sgemm，跳过\n", candidates[i]);
			continue;
		}
		// 立刻做一次 4x4 自检，确认这个库在本机真的能算
		static float sa[16] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
		static float sb[16] = { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
		static float sc[16] = { 0 };
		((cblas_sgemm_fn)s)(CblasRowMajor, CblasNoTrans, CblasTrans, 4, 4, 4, 1.0f, sa, 4, sb, 4, 0.0f, sc, 4);
		if (fabs(sc[0] - 8.0f) > 1e-3f)
		{
			printf("  [%s] 自检结果异常 (%.3f, 期望 8.0)，跳过\n", candidates[i], sc[0]);
			continue;
		}
		out.sgemm = (cblas_sgemm_fn)s;
		out.name = candidates[i];
		out.ok = true;
		return true;
	}
	return false;
}

struct Shape { int M, N, K; };

static int align8(int x) { return (x + 7) / 8 * 8; }

static double now_sec()
{
	std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
	return std::chrono::duration<double>(t.time_since_epoch()).count();
}

static void fill_random(float* v, size_t n, unsigned seed)
{
	unsigned s = seed;
	for (size_t i = 0; i < n; i++)
	{
		s = s * 1664525u + 1013904223u;
		v[i] = ((float)((s >> 8) & 0xFFFF) / 32768.0f) - 1.0f;
	}
}

static double diff_at(const float* a, const float* b, int M, int N, int ldc)
{
	double m = 0;
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
		{
			double d = (double)a[(size_t)i * ldc + j] - (double)b[(size_t)i * ldc + j];
			if (d < 0) d = -d;
			if (d > m) m = d;
		}
	return m;
}

static double time_gemm(gemm_fn fn, int M, int N, int K, const float* A, const float* Bt, float* C,
	int lda, int ldb, int ldc, int iters)
{
	fn(M, N, K, A, lda, Bt, ldb, C, ldc); // warmup
	double t0 = now_sec();
	for (int i = 0; i < iters; i++)
		fn(M, N, K, A, lda, Bt, ldb, C, ldc);
	double t1 = now_sec();
	double sec = (t1 - t0) / iters;
	return 2.0 * (double)M * (double)N * (double)K / sec / 1e9;
}

static double time_blas(cblas_sgemm_fn fn, int M, int N, int K, const float* A, const float* Bt,
	float* C, int lda, int ldb, int ldc, int iters)
{
	fn(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, lda, Bt, ldb, 0.0f, C, ldc);
	double t0 = now_sec();
	for (int i = 0; i < iters; i++)
		fn(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, lda, Bt, ldb, 0.0f, C, ldc);
	double t1 = now_sec();
	double sec = (t1 - t0) / iters;
	return 2.0 * (double)M * (double)N * (double)K / sec / 1e9;
}

// 固定容量的静态工作区：最大 1152x256x1152 / 1024^3 / 4096x4x4096 都能装下，
// 不做动态分配，避免把分配器行为混进性能对比里
static const size_t MAX_ELEM = 4u * 1024 * 1024 + 1024;
static float g_A[MAX_ELEM];
static float g_B[MAX_ELEM];
static float g_C[MAX_ELEM];
static float g_R[MAX_ELEM];
#if defined(_WIN32)
__declspec(align(32))
#endif
static float g_pad_a[64], g_pad_b[64], g_pad_c[64], g_pad_r[64];

// 注意：尺寸表必须放在文件作用域。函数体内的 `static Shape shapes[] = {...}`
// 会被 MSVC 解析成"带默认参数的函数声明"（Type name[] 形似参数列表）而报语法错。
static Shape shapes[] = {
	{ 16, 16, 16 }, { 32, 32, 32 }, { 64, 64, 64 }, { 128, 128, 128 },
	{ 256, 256, 256 }, { 512, 512, 512 }, { 1024, 1024, 1024 },
	{ 1, 1024, 1024 }, { 1024, 1, 1024 },
	{ 4096, 4, 4096 }, { 4, 4096, 4096 }, { 8, 2048, 2048 }, { 2048, 8, 2048 },
	{ 128, 128, 4096 }, { 4096, 128, 128 },
	{ 1152, 256, 1152 }, { 512, 512, 2048 },
	{ 7, 5, 13 }, { 33, 17, 65 },
};

int main(int argc, char* argv[])
{
#if defined(_WIN32)
	const char* kMklCandidates[] = { "3rdparty/mkl_runtime/win/mkl_rt.3.dll", "3rdparty/mkl_runtime/win/mkl_rt.2.dll", "3rdparty/mkl_runtime/win/mklml.dll", "mkl_rt.2.dll", "mklml.dll" };
	const char* kObCandidates[] = { "3rdparty/lib/libopenblas.dll", "libopenblas.dll" };
#else
	const char* kMklCandidates[] = { "3rdparty/mkl_runtime/linux/libmkl_rt.so.2", "3rdparty/mkl_runtime/linux/libmkl_rt.so", "libmkl_rt.so.2", "libmkl_rt.so" };
	const char* kObCandidates[] = { "3rdparty/lib/libopenblas.so", "libopenblas.so" };
#endif
	bool no_mkl = false;
	const char* mkl_path = getenv("ZQCNN_MKL_LIB");
	const char* ob_path = getenv("ZQCNN_OPENBLAS_LIB");
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "--no-mkl") == 0)
			no_mkl = true;
		else if (mkl_path == NULL)
			mkl_path = argv[i];
		else if (ob_path == NULL)
			ob_path = argv[i];
	}
	if (mkl_path != NULL) kMklCandidates[0] = mkl_path;
	if (ob_path != NULL) kObCandidates[0] = ob_path;

	force_mkl_sequential();
	BlasLib mkl, ob;
	bool has_mkl = !no_mkl && load_blas(mkl, kMklCandidates, 5, lib_open, lib_sym);
	bool has_ob = load_blas(ob, kObCandidates, 2, lib_open, lib_sym);
	printf("MKL     : %s\n", has_mkl ? mkl.name.c_str() : "(not found, skipped)");
	printf("OpenBLAS: %s\n", has_ob ? ob.name.c_str() : "(not found, skipped)");
	fflush(stdout);

	const int num_shapes = (int)(sizeof(shapes) / sizeof(shapes[0]));

	printf("\n%-16s %9s %9s %9s %9s %8s %8s  %s\n",
		"MxNxK", "intrinsic", "asm", "MKL(1T)", "OpenBLAS", "asm/MKL", "asm/intr", "err(asm)");
	for (int s = 0; s < num_shapes; s++)
	{
		const int M = shapes[s].M, N = shapes[s].N, K = shapes[s].K;
		const int lda = align8(K), ldb = align8(K), ldc = align8(N);
		const size_t na = (size_t)lda * M, nb = (size_t)ldb * N, nc = (size_t)ldc * M;
		if (na > MAX_ELEM || nb > MAX_ELEM || nc > MAX_ELEM)
		{
			printf("skip %dx%dx%d (workspace too small)\n", M, N, K);
			continue;
		}

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

		double work = 2.0 * M * N * K;
		int iters = work > 2e8 ? 3 : (work > 2e7 ? 10 : 50);

		zq_gemm_32f_AnoTrans_Btrans_auto(M, N, K, g_A, lda, g_B, ldb, g_C, ldc);
		zq_gemm_32f_AnoTrans_Btrans_auto_asm(M, N, K, g_A, lda, g_B, ldb, g_R, ldc);
		double err_asm = diff_at(g_C, g_R, M, N, ldc);

		double g_intr = time_gemm(zq_gemm_32f_AnoTrans_Btrans_auto, M, N, K, g_A, g_B, g_C, lda, ldb, ldc, iters);
		double g_asm = time_gemm(zq_gemm_32f_AnoTrans_Btrans_auto_asm, M, N, K, g_A, g_B, g_R, lda, ldb, ldc, iters);
		double g_mkl = 0, g_ob = 0, err_mkl = 0, err_ob = 0;
		if (has_mkl)
		{
			g_mkl = time_blas(mkl.sgemm, M, N, K, g_A, g_B, g_R, lda, ldb, ldc, iters);
			err_mkl = diff_at(g_C, g_R, M, N, ldc);
		}
		if (has_ob)
		{
			g_ob = time_blas(ob.sgemm, M, N, K, g_A, g_B, g_R, lda, ldb, ldc, iters);
			err_ob = diff_at(g_C, g_R, M, N, ldc);
		}

		int oob = 0;
		for (int i = 0; i < 64 && oob == 0; i++)
		{
			if (g_pad_a[i] != 0.0f) oob++;
			if (g_pad_b[i] != 0.0f) oob++;
			if (g_pad_c[i] != 0.0f) oob++;
			if (g_pad_r[i] != 0.0f) oob++;
			if (g_pad_a[i] != 0.0f) oob++;
			if (g_pad_b[i] != 0.0f) oob++;
			if (g_pad_c[i] != 0.0f) oob++;
			if (g_pad_r[i] != 0.0f) oob++;
			if (g_pad_a[i] != 0.0f) oob++;
			if (g_pad_b[i] != 0.0f) oob++;
			if (g_pad_c[i] != 0.0f) oob++;
			if (g_pad_r[i] != 0.0f) oob++;
			if (g_pad_a[i] != 0.0f) oob++;
			if (g_pad_b[i] != 0.0f) oob++;
			if (g_pad_c[i] != 0.0f) oob++;
			if (g_pad_r[i] != 0.0f) oob++;
		}

		char name[32], ratio[16];
		sprintf(name, "%dx%dx%d", M, N, K);
		if (has_mkl && g_mkl > 0) sprintf(ratio, "%.0f%%", 100.0 * g_asm / g_mkl);
		else strcpy(ratio, "-");
		printf("%-16s %9.2f %9.2f %9.2f %9.2f %8s %8.2f  %.1e%s%s\n",
			name, g_intr, g_asm, g_mkl, g_ob, ratio,
			(g_intr > 0 ? g_asm / g_intr : 0), err_asm,
			(oob ? "  OOB!" : ""), (err_asm > 1e-3 ? "  <== FAIL" : ""));
		fflush(stdout);
	}
	return 0;
}
