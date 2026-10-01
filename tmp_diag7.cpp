// temporary diagnostic: locate the out-of-bounds write with a guard page
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <windows.h>
#include "zq_gemm_32f_align_c.h"
#include "zq_gemm_32f_align_c_asm.h"

static const DWORD PAT = 0xA5A5A5A5u;

struct Blk { char* base; char* data; size_t bytes; };

// one region: [guard][data ... end exactly at page boundary][guard]
static Blk make_guarded(size_t bytes)
{
	size_t pg = 4096;
	size_t npages = (bytes + pg - 1) / pg;
	char* base = (char*)VirtualAlloc(NULL, (npages + 2) * pg, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
	VirtualProtect(base, pg, PAGE_READWRITE, NULL);
	VirtualProtect(base + pg + npages * pg, pg, PAGE_GUARD, NULL);
	Blk b; b.base = base; b.data = base + pg; b.bytes = npages * pg;
	memset(b.data, 0xA5, bytes);
	return b;
}

typedef void (*gemm_fn)(int, int, int, const float*, int, const float*, int, float*, int);

static int g_M, g_N, g_K;
static void report(EXCEPTION_POINTERS* ep)
{
	printf("  >> ACCESS VIOLATION in %dx%dx%d : %s addr=%p  fault-instr around rip=%p\n",
		g_M, g_N, g_K,
		ep->ExceptionRecord->ExceptionInformation[0] ? "WRITE" : "READ",
		ep->ExceptionRecord->ExceptionInformation[1],
		ep->ExceptionRecord->ExceptionAddress);
	// dump the faulting instruction bytes
	unsigned char* p = (unsigned char*)ep->ExceptionRecord->ExceptionAddress;
	printf("  >> bytes:");
	for (int i = 0; i < 12; i++) printf(" %02X", p[i]);
	printf("\n");
	fflush(stdout);
	ExitProcess(7);
}

static int try_one(int M, int N, int K)
{
	int padK = (K + 7) / 8 * 8;
	int lda = padK, ldb = padK, ldc = (N + 7) / 8 * 8;
	size_t na = (size_t)M * lda, nb = (size_t)N * ldb, nc = (size_t)M * ldc;
	Blk A = make_guarded(na * 4);
	Blk B = make_guarded(nb * 4);
	Blk C = make_guarded(nc * 4);
	float* a = (float*)A.data; float* b = (float*)B.data; float* c = (float*)C.data;
	for (size_t i = 0; i < na; i++) a[i] = 0.5f;
	for (size_t i = 0; i < nb; i++) b[i] = 0.25f;
	for (size_t i = 0; i < nc; i++) c[i] = 0;
	// also report the guard-page geometry so we can tell how far past the end it went
	printf("%4dx%-4dx%-4d  A.data=%p end=%p  B.data=%p end=%p  C.data=%p end=%p  (C end == guard start)\n",
		M, N, K, A.data, A.data + A.bytes, B.data, B.data + B.bytes, C.data, C.data + C.bytes);
	fflush(stdout);
	g_M = M; g_N = N; g_K = K;
	gemm_fn fn = zq_gemm_32f_AnoTrans_Btrans_auto_asm;
	__try {
		fn(M, N, K, a, lda, b, ldb, c, ldc);
	}
	__except (report(GetExceptionInformation()), EXCEPTION_EXECUTE_HANDLER) {
	}
	printf("       -> no fault\n");
	fflush(stdout);
	VirtualFree(A.base, 0, MEM_RELEASE);
	VirtualFree(B.base, 0, MEM_RELEASE);
	VirtualFree(C.base, 0, MEM_RELEASE);
	return 0;
}

int main()
{
	printf("ZQ_GEMM asm guard-page probe\n");
	try_one(16, 16, 16);
	try_one(32, 32, 32);
	try_one(64, 64, 64);
	try_one(1, 8, 64);
	try_one(8, 1, 64);
	try_one(2, 4, 8);
	try_one(7, 5, 13);
	try_one(5, 9, 17);
	try_one(3, 7, 5);
	try_one(128, 128, 128);
	return 0;
}
