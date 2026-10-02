// zq_gemm_ab_perf.cpp —— dispatcher 守卫的 A/B 计时驱动（附录 BQ）
//
// 用途：给 tools/ab_time_two_binaries.sh 提供一个「一次跑一个形状、打印一行
// `total = X ms`」的驱动。那边负责 ABABAB 交替 + 取中位数（噪声下限约 7%）。
//
// 为什么要专门做这个（附录 BP.7 / BQ）
// ------------------------------------
// BP.1 给 dispatcher 入口加了一条守卫 `K % align != 0 -> fallback`。我给它的
// 论证是**结构性**的：对 K%align==0 的形状，守卫之后的函数体逐字节不变。
// 3254 个格子全绿是**功能性**证据，但都不是"量过没影响"。
// 本驱动把"before / after"两个二进制编出来交替跑，补上计时那一半。
//
// 缓冲区必须 32 字节对齐（附录 BP.8）：256bit 那一族用的是**对齐**载入
// （_mm256_load_ps），glibc malloc 只给 16 字节。
// **这条在 ASan 下看不出来** —— ASan 的分配器给 32 —— 所以任何在 ASan 下写的
// 测试永远看不到它，一去掉 ASan 就第一次调用就崩。
//
// 用法:
//   ./zq_gemm_ab_perf <tag> <M> <N> <K>
//   ./zq_gemm_ab_perf <tag>              <- 用编译期定的形状 (AB_M/AB_N/AB_K)
// 打印:
//   <tag> M=.. N=.. K=.. total = 12.345678 ms  gf/s = 23.45
//
// 为什么要有编译期默认：tools/ab_time_two_binaries.sh 是**不带参数**调用二进制的
// （它只接两个二进制路径 + 轮数 + grep 的 key），所以每个形状要编两个二进制。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <malloc.h>
#include <time.h>

// 32 字节对齐的分配（附录 BP.8）。
// 不要直接用 _aligned_malloc：gcc 的 <malloc.h> 在这台机器上**不声明**它
// （早先几个测试能用是因为它们 include 了 ZQCNN 的头，被顺带带进来的）。
// 也不能用 std::vector / malloc —— 只给 16 字节。
#if defined(_WIN32)
#include <malloc.h>
static void* zq_ab_aligned(size_t bytes, size_t align)
{
    void* p = _aligned_malloc(bytes, align);
    if (p) memset(p, 0, bytes);
    return p;
}
static void zq_ab_free(void* p) { _aligned_free(p); }
#else
static void* zq_ab_aligned(size_t bytes, size_t align)
{
    if (align < sizeof(void*)) align = sizeof(void*);
    size_t rounded = ((bytes + align - 1) / align) * align;
    void* p = 0;
    if (posix_memalign(&p, align, rounded) != 0) return 0;
    memset(p, 0, rounded);
    return p;
}
static void zq_ab_free(void* p) { free(p); }
#endif

#ifndef AB_M
#define AB_M 512
#endif
#ifndef AB_N
#define AB_N 512
#endif
#ifndef AB_K
#define AB_K 512
#endif

extern "C" void zq_gemm_32f_AnoTrans_Btrans_auto(
    int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

// 确定性伪随机：不用 rand()，要跨运行逐位可比
static float rv(int seed, int i)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)i * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

static double now_s()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char** argv)
{
    // tools/ab_time_two_binaries.sh 是**不带参数**调用二进制的，
    // 所以 tag / 形状都必须可选（形状取编译期 AB_M/AB_N/AB_K）。
    // 注意无参数时 argc == 1，所以这里**不能**写 argc < 2 就报错。
    const char* tag = "-";
    int M = AB_M, N = AB_N, K = AB_K;
    if (argc >= 2) tag = argv[1];
    if (argc > 4) { M = atoi(argv[2]); N = atoi(argv[3]); K = atoi(argv[4]); }

    // 附录 BP.8：32 字节对齐，不是 malloc 也不是 std::vector
    float* A  = (float*)zq_ab_aligned(sizeof(float) * (size_t)M * K, 32);
    float* Bt = (float*)zq_ab_aligned(sizeof(float) * (size_t)N * K, 32);
    float* C  = (float*)zq_ab_aligned(sizeof(float) * (size_t)M * N, 32);
    if (A == 0 || Bt == 0 || C == 0) { printf("alloc failed\n"); return 2; }
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = rv(1, (int)i);
    for (size_t i = 0; i < (size_t)N * K; i++) Bt[i] = rv(2, (int)i);

    for (int w = 0; w < 2; w++)
        zq_gemm_32f_AnoTrans_Btrans_auto(M, N, K, A, K, Bt, K, C, N);

    // 取 3 次里最好的一次：减少被别的进程干扰的概率
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now_s();
        zq_gemm_32f_AnoTrans_Btrans_auto(M, N, K, A, K, Bt, K, C, N);
        double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    // 顺便自检一下结果不是全 0（全 0 的话"很快"就没有意义）
    double sum = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) sum += C[i];
    printf("%s M=%d N=%d K=%d total = %.6f ms  gf/s = %.3f  chk=%.6f\n",
           tag, M, N, K, best * 1000.0, 2.0 * M * N * K / (best * 1e9), sum);
    zq_ab_free(A); zq_ab_free(Bt); zq_ab_free(C);
    return 0;
}
