// zq_gemm_shape_check.cpp —— zq_gemm_32f_AnoTrans_Btrans_auto 的 (M,N,K) 形状覆盖
//
// 起因（audit_k3_20261001.md 附录 BN.2）
// --------------------------------------------
// `tools/zq_innerproduct_check.cpp`（NCHW 的 innerproduct）只测了
// N ∈ {16,17,20} x filter_N ∈ {16,17,24,33} —— 那是**生产条件**
// `out_N >= 16 && filter_N >= 16` 限定的范围（附录 BC 查出来的）。
// 于是 ZQ_GEMM 的 auto dispatcher 在 M<16 / N<16 上的行为**一次都没被测过**。
//
// 而 NCHWC 的 innerproduct（附录 BN.1 那个文件）**没有**这个下限：
// `zq_cnn_innerproduct_gemm_nchwc*_general` 对任意
// (out_N, filter_N) 都直接调 sgemm，于是 M=2 / N=7 这种形状**真的会被走到**。
// 本文件就是把那张没测过的表补上，并找出边界在哪。
//
// 为什么每个用例 fork 一个子进程
// ------------------------------
// 缺陷是**崩溃**（SEGV），不是算错。ASan 碰到 SEGV 直接 abort 掉整个进程，
// 于是第一版只跑到 "M=2 N=2 K=27" 就没了，后面整片区域都不知道是什么情况。
// 一旦崩溃就停的测试只能告诉你"有一个坏了"，没法告诉你"哪些是好的" ——
// 而后者恰恰是判断边界、以及判断生产有没有踩到的关键。
// 所以这里每个用例 fork 一个子进程：父进程 waitpid 拿退出码/信号，
// 崩溃算"该用例失败"并继续往下扫。
//
// 只在 Linux 下编（run_zqlib_checks.py 的入口就是 g++ / WSL）。
//
// 参考实现：最朴素的三重循环。C[m][n] = sum_k A[m][k] * Bt[n][k]
// （调用点是 CblasTrans，所以 B 是按行存的，lda_B 才是"每行 K 个"）
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

extern "C" void zq_gemm_32f_AnoTrans_Btrans_auto(
    int M, int N, int K,
    const float* A, int lda,
    const float* Bt, int ldb,
    float* C, int ldc);

// **必须**：ASan 默认在出错时是 `exit(1)` 而不是 abort —— 于是"崩溃"和
// "算错"在 waitpid 看来都是 exit code 1，分不开（第一版 134 个 FAIL 里
// 两种混在一起）。设成 abort_on_error=1 之后崩溃才会变成 SIGABRT。
// 这个函数在 ASan 初始化**之前**被调用，所以设在这里是有效的。
extern "C" const char* __asan_default_options()
{
    return "abort_on_error=1:handle_segv=1:detect_leaks=0";
}

static float rv(int seed, int i)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)i * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

// 在**子进程**里跑。返回 0=通过 1=算错；崩溃由父进程从信号判定。
// 子进程的 stderr 要接到 /dev/null：ASan 的报告走 stderr，会把父进程 stdout 上
// 的网格那一行拦腰截断（第一版的网格就是这样变成"每行只有一个 X"的）。
static int run_case(int M, int N, int K)
{
    FILE* devnull = freopen("/dev/null", "w", stderr);
    (void)devnull;
    std::vector<float> A((size_t)M * K), Bt((size_t)N * K), ref((size_t)M * N), got((size_t)M * N);
    for (size_t i = 0; i < A.size(); i++) A[i] = rv(1, (int)i);
    for (size_t i = 0; i < Bt.size(); i++) Bt[i] = rv(2, (int)i);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double s = 0;
            for (int k = 0; k < K; k++) s += (double)A[(size_t)m * K + k] * (double)Bt[(size_t)n * K + k];
            ref[(size_t)m * N + n] = (float)s;
        }
    memset(&got[0], 0, sizeof(float) * got.size());
    zq_gemm_32f_AnoTrans_Btrans_auto(M, N, K, &A[0], K, &Bt[0], K, &got[0], N);
    double maxrel = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double d = fabs((double)got[(size_t)m * N + n] - ref[(size_t)m * N + n]);
            double den = fabs(ref[(size_t)m * N + n]) > 1e-6 ? fabs(ref[(size_t)m * N + n]) : 1.0;
            if (d / den > maxrel) maxrel = d / den;
        }
    return maxrel > 2e-4 ? 1 : 0;
}

static int g_fail = 0, g_case = 0;
static int g_crash = 0, g_wrong = 0;

// '.'=ok  'X'=崩溃  'x'=结果错。返回格子字符。
static char one(int M, int N, int K)
{
    g_case++;
    fflush(stdout);
    pid_t pid = fork();
    if (pid < 0) { printf("  fork 失败\n"); g_fail++; return '?'; }
    if (pid == 0) { _exit(run_case(M, N, K)); }
    int st = 0;
    waitpid(pid, &st, 0);
    if (WIFSIGNALED(st)) { g_fail++; g_crash++; return 'X'; }
    if (WEXITSTATUS(st) != 0) { g_fail++; g_wrong++; return 'x'; }
    return '.';
}

// 打成网格：行 M、列 N，一眼看得出边界
static void grid(const char* title, int K)
{
    static const int NN = 9;
    printf("--- K=%d  (行 M=1..8, 列 N=1..9; '.' ok  'X' 崩溃  'x' 结果错) ---\n", K);
    printf("      ");
    for (int n = 1; n <= NN; n++) printf(" N=%-3d", n);
    printf("\n");
    for (int M = 1; M <= 8; M++) {
        printf("  M=%-3d", M);
        for (int N = 1; N <= NN; N++) printf("  %-4c", one(M, N, K));
        printf("\n");
        fflush(stdout);
    }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("zq_gemm_32f_AnoTrans_Btrans_auto 的 (M,N,K) 形状覆盖（附录 BN.2）\n");
    printf("每个用例 fork 一个子进程：崩溃只算该用例失败，不会中断整张表\n");
    printf("ASAN_OPTIONS 里加了 abort_on_error=1，否则崩溃是 exit(1) 而不是 SIGABRT，\n");
    printf("会跟「算错」分不开（第一版就是把两种混在一起报的 134 个 FAIL）\n\n");

    // M = out_N（batch），N = filter_N，K = 滤波器的 K 维。
    // K=27 是 3x3x3，也就是**每个 CNN 的 RGB 首层** —— 最该被测到的一档。
    static const int KS[] = { 27, 108, 512, 3136 };
    for (int ki = 0; ki < 4; ki++) grid(NULL, KS[ki]);

    printf("--- 生产条件那一侧 (M,N >= 16) 抽查，确认不是「全都坏了」---\n");
    {
        static const int ms[] = {16, 17, 20, 32};
        static const int ns[] = {16, 17, 24, 33};
        int bad = 0;
        for (int a = 0; a < 4; a++)
            for (int b = 0; b < 4; b++)
                if (one(ms[a], ns[b], 27) != '.') bad++;
        printf("  M,N >= 16: %d/16 通过\n", 16 - bad);
    }
    printf("\n%s (共 %d 个用例; 崩溃 %d, 结果错 %d)\n",
           g_fail ? "FAILED" : "PASSED", g_case, g_crash, g_wrong);
    return g_fail ? 1 : 0;
}
