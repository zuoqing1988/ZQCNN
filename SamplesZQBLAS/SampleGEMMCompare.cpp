/*
 * SampleGEMMCompare.cpp
 *
 * ZQ_GEMM 内核选型基准：把四条路径放在同一组矩阵尺寸上对比
 *   1) ZQ_GEMM intrinsic   zq_gemm_32f_AnoTrans_Btrans_auto
 *   2) ZQ_GEMM inline asm  zq_gemm_32f_AnoTrans_Btrans_auto_asm
 *   3) Intel MKL           cblas_sgemm(RowMajor, NoTrans, Trans)
 *   4) OpenBLAS            cblas_sgemm(RowMajor, NoTrans, Trans)
 *
 * MKL / OpenBLAS 全部用运行时动态加载（LoadLibrary/dlopen + GetProcAddress），
 * 本机没装就直接跳过对应条目，不影响其它路径，也不需要在链接期绑死某个 BLAS。
 *
 * 语义与 ZQ_GEMM 一致：
 *   C[i][j] = sum_k A[i*lda+k] * Bt[j*ldb+k]
 *   A: M x K, Bt: N x K, C: M x N, 全部行主序, C 被覆盖写入
 * 对应 CBLAS 行主序就是 C = A * Bt^T，即 transB = CblasTrans。
 *
 * 用法：
 *   SampleGEMMCompare                     默认尺寸组
 *   SampleGEMMCompare [mkl_lib路径] [openblas_lib路径]
 * 环境变量 ZQCNN_MKL_LIB / ZQCNN_OPENBLAS_LIB 优先级最高。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <vector>
#include <chrono>

#include "zq_gemm_32f_align_c.h"
#include "zq_gemm_32f_auto.h"
#include "zq_gemm_32f_align_c_asm.h"

#if defined(_WIN32)
#  include <windows.h>
   typedef void* lib_handle_t;
   static lib_handle_t lib_open(const char* p) { return (lib_handle_t)LoadLibraryA(p); }
   static void* lib_sym(lib_handle_t h, const char* n) { return (void*)GetProcAddress((HMODULE)h, n); }
   static const char* kLibExt = ".dll";
#else
#  include <dlfcn.h>
   typedef void* lib_handle_t;
   static lib_handle_t lib_open(const char* p) { return dlopen(p, RTLD_NOW | RTLD_LOCAL); }
   static void* lib_sym(lib_handle_t h, const char* n) { return dlsym(h, n); }
   static const char* kLibExt = ".so";
#endif

typedef void (*cblas_sgemm_fn)(int Order, int TransA, int TransB,
	int M, int N, int K, float alpha, const float* A, int lda,
	const float* B, int ldb, float beta, float* C, int ldc);

enum { CblasRowMajor = 101, CblasNoTrans = 111, CblasTrans = 112 };

struct BlasLib
{
	cblas_sgemm_fn sgemm;
	void (*set_threads)(int);
	std::string name;
	bool multi_thread_capable;
};

static bool load_blas(BlasLib& out, const char* candidates[], int num_candidates,
	void* (*open_fn)(const char*), void* (*sym_fn)(void*, const char*), const char* sym_name)
{
	for (int i = 0; i < num_candidates; i++)
	{
		if (candidates[i] == NULL || candidates[i][0] == '\0')
			continue;
		void* h = open_fn(candidates[i]);
		if (h == NULL)
			continue;
		void* s = sym_fn(h, sym_name);
		if (s == NULL)
		{
			printf("  [%s] 打开成功但找不到 %s，跳过\n", candidates[i], sym_name);
			continue;
		}
		out.sgemm = (cblas_sgemm_fn)s;
		out.name = candidates[i];
		void* t = sym_fn(h, out.multi_thread_capable ? "mkl_set_num_threads" : "openblas_set_num_threads");
		out.set_threads = (void(*)(int))t;
		return true;
	}
	return false;
}

struct Shape { int M, N, K; };

static double now_sec()
{
	return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void fill_random(std::vector<float>& v, unsigned seed)
{
	unsigned s = seed;
	for (size_t i = 0; i < v.size(); i++)
	{
		s = s * 1664525u + 1013904223u;
		v[i] = ((float)((s >> 8) & 0xFFFF) / 32768.0f) - 1.0f;
	}
}

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
	double m = 0;
	size_t n = a.size() < b.size() ? a.size() : b.size();
	for (size_t i = 0; i < n; i++)
	{
		double d = (double)a[i] - (double)b[i];
		if (d < 0) d = -d;
		if (d > m) m = d;
	}
	return m;
}

static double time_gemm(void (*fn)(int, int, int, const float*, int, const float*, int, float*, int),
	int M, int N, int K, const std::vector<float>& A, const std::vector<float>& Bt, std::vector<float>& C,
	int lda, int ldb, int ldc, int iters)
{
	fn(M, N, K, &A[0], lda, &Bt[0], ldb, &C[0], ldc); // warmup
	double t0 = now_sec();
	for (int i = 0; i < iters; i++)
		fn(M, N, K, &A[0], lda, &Bt[0], ldb, &C[0], ldc);
	double t1 = now_sec();
	double sec = (t1 - t0) / iters;
	return 2.0 * (double)M * (double)N * (double)K / sec / 1e9; // GFLOP/s
}

static double time_blas(cblas_sgemm_fn fn, int M, int N, int K, const std::vector<float>& A,
	const std::vector<float>& Bt, std::vector<float>& C, int lda, int ldb, int ldc, int iters)
{
	fn(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, &A[0], lda, &Bt[0], ldb, 0.0f, &C[0], ldc);
	double t0 = now_sec();
	for (int i = 0; i < iters; i++)
		fn(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, &A[0], lda, &Bt[0], ldb, 0.0f, &C[0], ldc);
	double t1 = now_sec();
	double sec = (t1 - t0) / iters;
	return 2.0 * (double)M * (double)N * (double)K / sec / 1e9;
}

int main(int argc, char* argv[])
{
#if defined(_WIN32)
	const char* kMklCandidates[] = { "3rdparty/mkl_runtime/win/mkl_rt.3.dll", "3rdparty/mkl_runtime/win/mkl_rt.2.dll", "3rdparty/mkl_runtime/win/mklml.dll", "mkl_rt.2.dll", "mklml.dll" };
	const char* kObCandidates[] = { "3rdparty/lib/libopenblas.dll", "libopenblas.dll" };
#else
	const char* kMklCandidates[] = { "3rdparty/mkl_runtime/linux/libmkl_rt.so.2", "3rdparty/mkl_runtime/linux/libmkl_rt.so", "libmkl_rt.so.2", "libmkl_rt.so" };
	const char* kObCandidates[] = { "3rdparty/lib/libopenblas.so", "libopenblas.so" };
#endif
	const char* mkl_path = getenv("ZQCNN_MKL_LIB");
	const char* ob_path = getenv("ZQCNN_OPENBLAS_LIB");
	if (argc > 1) mkl_path = argv[1];
	if (argc > 2) ob_path = argv[2];

	BlasLib mkl; mkl.sgemm = NULL; mkl.set_threads = NULL; mkl.multi_thread_capable = true;
	BlasLib ob; ob.sgemm = NULL; ob.set_threads = NULL; ob.multi_thread_capable = false;
	if (mkl_path != NULL) kMklCandidates[0] = mkl_path;
	if (ob_path != NULL) kObCandidates[0] = ob_path;

	bool has_mkl = load_blas(mkl, kMklCandidates, 5, lib_open, lib_sym, "cblas_sgemm");
	bool has_ob = load_blas(ob, kObCandidates, 2, lib_open, lib_sym, "cblas_sgemm");
	printf("MKL    : %s\n", has_mkl ? mkl.name.c_str() : "(未找到，已跳过)");
	printf("OpenBLAS: %s\n", has_ob ? ob.name.c_str() : "(未找到，已跳过)");

	// 单线程对齐比较：ZQ_GEMM 是单线程实现，BLAS 也压到 1 线程
	if (has_mkl && mkl.set_threads) mkl.set_threads(1);
	if (has_ob && ob.set_threads) ob.set_threads(1);

	Shape shapes[] = {
		{ 16, 16, 16 }, { 32, 32, 32 }, { 64, 64, 64 }, { 128, 128, 128 },
		{ 256, 256, 256 }, { 512, 512, 512 }, { 1024, 1024, 1024 },
		{ 1, 1024, 1024 }, { 1024, 1, 1024 },
		{ 4096, 4, 4096 }, { 4, 4096, 4096 }, { 8, 2048, 2048 }, { 2048, 8, 2048 },
		{ 128, 128, 4096 }, { 4096, 128, 128 },
		{ 1152, 256, 1152 }, { 512, 512, 2048 },
		{ 7, 5, 13 }, { 33, 17, 65 },
	};
	const int num_shapes = (int)(sizeof(shapes) / sizeof(shapes[0]));

	printf("\n%-22s %10s %10s %10s %10s   %8s %8s\n",
		"MxNxK", "intrinsic", "asm", "MKL(1T)", "OpenBLAS", "asm/MKL", "asm/intr");
	for (int s = 0; s < num_shapes; s++)
	{
		const int M = shapes[s].M, N = shapes[s].N, K = shapes[s].K;
		const int lda = K, ldb = K, ldc = N;
		std::vector<float> A((size_t)M * K), Bt((size_t)N * K), C((size_t)M * N);
		fill_random(A, 12345u + (unsigned)s);
		fill_random(Bt, 67890u + (unsigned)s);

		double work = 2.0 * M * N * K;
		int iters = work > 2e8 ? 3 : (work > 2e7 ? 10 : 50);

		std::vector<float> C_int(C), C_asm(C);
		zq_gemm_32f_AnoTrans_Btrans_auto(M, N, K, &A[0], lda, &Bt[0], ldb, &C_int[0], ldc);
		zq_gemm_32f_AnoTrans_Btrans_auto_asm(M, N, K, &A[0], lda, &Bt[0], ldb, &C_asm[0], ldc);
		double err_asm = max_abs_diff(C_int, C_asm);

		double g_intr = time_gemm(zq_gemm_32f_AnoTrans_Btrans_auto, M, N, K, A, Bt, C, lda, ldb, ldc, iters);
		double g_asm = time_gemm(zq_gemm_32f_AnoTrans_Btrans_auto_asm, M, N, K, A, Bt, C, lda, ldb, ldc, iters);

		double g_mkl = 0, g_ob = 0, err_mkl = 0, err_ob = 0;
		std::vector<float> C_ref(C);
		if (has_mkl)
		{
			g_mkl = time_blas(mkl.sgemm, M, N, K, A, Bt, C_ref, lda, ldb, ldc, iters);
			err_mkl = max_abs_diff(C_int, C_ref);
		}
		if (has_ob)
		{
			g_ob = time_blas(ob.sgemm, M, N, K, A, Bt, C_ref, lda, ldb, ldc, iters);
			err_ob = max_abs_diff(C_int, C_ref);
		}

		char name[32];
		sprintf(name, "%dx%dx%d", M, N, K);
		printf("%-22s %10.2f %10.2f %10.2f %10.2f   %8s %8.2f   err(asm)=%.2e err(mkl)=%.2e err(ob)=%.2e\n",
			name, g_intr, g_asm, g_mkl, g_ob,
			(has_mkl && g_mkl > 0) ? (sprintf(name, "%.0f%%", 100.0 * g_asm / g_mkl), name) : "-",
			(g_intr > 0 ? g_asm / g_intr : 0), err_asm, err_mkl, err_ob);
		fflush(stdout);
	}

	if (has_mkl)
	{
		printf("\n-- MKL 多线程（全核）参考 --\n");
		if (mkl.set_threads) mkl.set_threads(0);
		Shape s = { 1024, 1024, 1024 };
		std::vector<float> A((size_t)s.M * s.K), Bt((size_t)s.N * s.K), C((size_t)s.M * s.N);
		fill_random(A, 1); fill_random(Bt, 2);
		double g_mkl_mt = time_blas(mkl.sgemm, s.M, s.N, s.K, A, Bt, C, s.K, s.K, s.N, 3);
		double g_asm = time_gemm(zq_gemm_32f_AnoTrans_Btrans_auto_asm, s.M, s.N, s.K, A, Bt, C, s.K, s.K, s.N, 3);
		printf("1024^3: asm(单线程) %.2f GFLOP/s, MKL(全核) %.2f GFLOP/s\n", g_asm, g_mkl_mt);
	}
	return 0;
}
