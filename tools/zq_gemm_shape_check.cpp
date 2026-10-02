// zq_gemm_shape_check.cpp —— zq_gemm_32f_AnoTrans_Btrans_auto 的 (M,N,K) 形状**安全图**
//
// 起因（audit_k3_20261001.md 附录 BN.3 / BO）
// -----------------------------------------------
// 附录 BN 定位到：dispatcher 在 K=27 / K=108 且 M>=2 时**崩溃**，而它是
// NCHW conv/deconv/innerproduct + NCHWC conv/innerproduct 共 16 个调用点的
// 公共入口（K=27 就是 3x3x3，即每个 CNN 的 RGB 首层）。
//
// 但 BN 只测了 M<=8 / N<=9 / 4 个 K，**不足以写出一道保守的守卫**：
// 守卫只能拦"现在就是坏的"那些形状，而"哪些形状是好的"必须**测出来**，
// 不能靠读 1.6 万行内核去推 —— 上一轮就是这么差点把一条假缺陷报上去的。
// 所以本文件把那张表补全：M 12 档 x N 15 档 x K 18 档 = 3240 个用例。
//
// 为什么每个用例 fork 一个子进程（三条，附录 BN.5）
// --------------------------------------------------
// 1) 崩溃会吃掉整张表：ASan 碰 SEGV 直接 abort，一崩就停的测试只能告诉你
//    "有一个坏了"，没法告诉你"哪些是好的" —— 而后者恰恰是写守卫的前提。
// 2) ASan 默认是 exit(1) 不是 abort，于是"崩溃"和"算错"在 waitpid 看来都是
//    exit code 1。必须用 __asan_default_options 设 abort_on_error=1 才分得开。
// 3) 子进程的 stderr 重定向走 zq_check_child.h（附录 CZ）：
//    不接 /dev/null 是因为那样连 sanitizer 的报告本身都看不见了；
//    仍然要重定向是因为不重定向的话 ASan 的报告会拦腰截断网格那一行。
//
// 输出：每个 K 一张 M x N 的网格，'.'  ok / 'X' 崩溃 / 'x' 结果错
//
// 参考实现：最朴素的三重循环。C[m][n] = sum_k A[m][k] * Bt[n][k]
// （调用点是 CblasTrans，Bt 按行存，每行 K 个，所以 ldb 才是"每行 K 个"）
#include "zq_check_child.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include <unistd.h>
#include <sys/wait.h>

extern "C" void zq_gemm_32f_AnoTrans_Btrans_auto(
    int M, int N, int K,
    const float* A, int lda,
    const float* Bt, int ldb,
    float* C, int ldc);

// 必须在 ASan 初始化**之前**被调用才有效：默认 abort_on_error=0 时
// 崩溃是 exit(1)，跟"算错"分不开。
extern "C" const char* __asan_default_options()
{
    return "abort_on_error=1:detect_leaks=0";
}

static float rv(int seed, int i)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)i * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

// 在**子进程**里跑。返回 0=通过 1=结果错。崩溃由父进程从信号判定。
static int run_case(int M, int N, int K)
{
    zq_child_silence_stderr();
    (void)dn;
    std::vector<float> A((size_t)M * K), Bt((size_t)N * K), ref((size_t)M * N), got((size_t)M * N);
    for (size_t i = 0; i < A.size(); i++) A[i] = rv(1, (int)i);
    for (size_t i = 0; i < Bt.size(); i++) Bt[i] = rv(2, (int)i);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double s = 0;
            for (int k = 0; k < K; k++) s += (double)A[(size_t)m * K + k] * (double)Bt[(size_t)n * K + k];
            ref[(size_t)m * N + n] = (float)s;
        }
    // 行的 2-范数（下面那个判据的分母要用）
    std::vector<double> na((size_t)M), nb((size_t)N);
    for (int m = 0; m < M; m++) {
        double t = 0;
        for (int k = 0; k < K; k++) { double v = A[(size_t)m * K + k]; t += v * v; }
        na[m] = sqrt(t);
    }
    for (int n = 0; n < N; n++) {
        double t = 0;
        for (int k = 0; k < K; k++) { double v = Bt[(size_t)n * K + k]; t += v * v; }
        nb[n] = sqrt(t);
    }
    memset(&got[0], 0, sizeof(float) * got.size());
    zq_gemm_32f_AnoTrans_Btrans_auto(M, N, K, &A[0], K, &Bt[0], K, &got[0], N);
    // 判据用**后向误差**（backward error），不是相对误差 —— 这一条改错过三次，
    // 理由见文件末尾的「判据为什么不能是相对误差」。
    //     err(m,n) = |got - exp| / (||A_m|| * ||B_n||)
    // 这个量就是"把结果往回推一步"的容差，与求和过程中的抵消无关。
    double maxrel = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double d = fabs((double)got[(size_t)m * N + n] - ref[(size_t)m * N + n]);
            double den = na[m] * nb[n];
            if (den < 1e-30) den = 1.0;
            if (d / den > maxrel) maxrel = d / den;
        }
    return maxrel > 1e-5 ? 1 : 0;
}

static char one(int M, int N, int K)
{
    fflush(stdout);
    pid_t pid = fork();
    if (pid < 0) return '?';
    if (pid == 0) { _exit(run_case(M, N, K)); }
    int st = 0;
    waitpid(pid, &st, 0);
    if (WIFSIGNALED(st)) return 'X';
    return WEXITSTATUS(st) == 0 ? '.' : 'x';
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);

    // M 12 档：1..8（noborders 之外的 batch 常见值）+ 16/17/20/32
    //         （16 是 NCHW innerproduct 生产守卫 out_N>=16 的下界，17/20 是
    //           附录 BI 实测过的那几档，32 是常见的 batch）
    static const int MS[] = { 1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 20, 32 };
    const int NM = (int)(sizeof(MS) / sizeof(MS[0]));
    // N 15 档：1..8 + 16/17/24/32/33/64/128（同样含生产守卫 filter_N>=16 那一带）
    static const int NS[] = { 1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 24, 32, 33, 64, 128 };
    const int NN = (int)(sizeof(NS) / sizeof(NS[0]));
    // K 18 档：dispatcher 里 special-case 的全部（16/24/27/28/32/64/72/128/144/
    //          256/512/1024）+ 6 个**没有** special-case 的（走 back up 分支）：
    //          17/33/108/112/150/3136
    static const int KS[] = { 16, 17, 24, 27, 28, 32, 33, 64, 72, 108, 112,
                              128, 144, 150, 256, 512, 1024, 3136 };
    const int NK = (int)(sizeof(KS) / sizeof(KS[0]));

    printf("zq_gemm_32f_AnoTrans_Btrans_auto 形状安全图（附录 BO）\n");
    printf("每个用例 fork 一个子进程；ASAN abort_on_error=1（否则崩溃与算错分不开）\n");
    printf("M %d 档 x N %d 档 x K %d 档 = %d 个用例\n", NM, NN, NK, NM * NN * NK);
    printf("'.' ok   'X' 崩溃   'x' 结果错\n\n");

    int g_ok = 0, g_crash = 0, g_wrong = 0;
    for (int ki = 0; ki < NK; ki++) {
        int K = KS[ki];
        printf("=== K=%-5d (M x N) ===\n", K);
        printf("%-7s", "");
        for (int j = 0; j < NN; j++) printf("N=%-5d", NS[j]);
        printf("\n");
        for (int i = 0; i < NM; i++) {
            printf("M=%-5d", MS[i]);
            for (int j = 0; j < NN; j++) {
                char c = one(MS[i], NS[j], K);
                if (c == '.') g_ok++;
                else if (c == 'X') g_crash++;
                else g_wrong++;
                printf("%-7c", c);
            }
            printf("\n");
        }
        printf("\n");
    }
    // ---- production 形状：M = out_N*out_H*out_W。上面的表只到 M=32，
    // 而真实卷积的 M 是上千。**BO.7 那条"守卫碰不到快路径"的证明就靠这一段**
    // —— 如果 K%8==0 在大 M 上也是坏的，那条证明就不成立，守卫就可能有性能代价。
    printf("=== production 形状（M = out_N*out_H*out_W，上千）===\n");
    {
        struct P { int M, N, K; };
        static const P prods[] = {
            { 1024,  64,  144 },   // 3x3, C=16
            { 1024,  64,  288 },   // 3x3, C=32
            {  784, 128,  144 },   // 56x56 的 3x3
            {  784, 256,  288 },
            {  196, 512,  576 },   // 14x14 的 3x3, C=64
            { 1024,  16,  512 },   // 1x1
            { 1024,  64, 1024 },   // 1x1
            { 3136,  64,  144 },   // 56x56 的 1x1
            { 3136, 256,  288 },
            { 2500,  32, 3136 },   // ArcFace 那种 7x7x512
            {  512,  10,   32 },   // MTCNN R-net 首层那个量级
            // K 不是 8 的倍数的那一片（现在会崩，修完应当通过）
            {  512,  10,   27 },   // 3x3x3
            { 3136,  32,    9 },   // 3x3, C=1
            { 1024,  64,  108 },
        };
        for (size_t i = 0; i < sizeof(prods) / sizeof(prods[0]); i++) {
            char c = one(prods[i].M, prods[i].N, prods[i].K);
            if (c == '.') g_ok++;
            else if (c == 'X') g_crash++;
            else g_wrong++;
            printf("  M=%-5d N=%-4d K=%-5d %c\n", prods[i].M, prods[i].N, prods[i].K, c);
        }
        printf("\n");
    }
    printf("总计: ok %d, 崩溃 %d, 结果错 %d  (%d 个网格用例 + %d 个 production 形状)\n",
           g_ok, g_crash, g_wrong, NM * NN * NK, 14);
    // 注意：崩溃/结果错的形状**现在就是坏的**，所以这个测试当前应当是红的。
    // 门禁里它是 SKIP（理由见 run_zqlib_checks.py 的 SKIP['zq_gemm_shape']），
    // 修好之后把 SKIP 那一条删掉即可。
    return (g_crash || g_wrong) ? 1 : 0;
}

// 判据为什么**不能**是相对误差（这一点我错了三次，值得写下来）
// ------------------------------------------------------------
// 第一版用的是 `|got-exp| / max(|exp|, 1)`，阈值 2e-4。它在 K=1024 上报了一堆
// "结果错"：
//
//     M=1024 N=64 K=1024  max_rel=5.5e-02 @ (m=441,n=11) got=0.000046 exp=0.000043
//
// 看着像"大形状下静默算错"。其实是**我的判据错了**：
// 1024 个量级 ~0.29 的乘积求和，中间值在 ±9 量级，某个 (m,n) 上正好抵消到
// 4.6e-5 —— float32 只有 7.2 位有效数字，抵消 5 个数量级之后**绝对**误差
// 自然还有 1e-6 量级，除以 4.6e-5 就是 12%。**这是 float32 的固有性质，
// 不是内核的错。**
//
// 于是同一批数字又让我差点报一条假缺陷。GEMM 正确的判据是**后向误差**：
//
//     |got - exp| / (||A_m||_2 * ||B_n||_2)
//
// 分母是"这一格的计算尺度"，与结果被抵消到多小无关。判据换成它之后
// K=1024 那些格子全部通过，而真正坏掉的形状（K 不是 8 的倍数那一片）
// 仍然是**崩溃**—— 崩溃与判据无关，所以那条发现不受影响。
//
// 教训与附录 BC/BI 一脉相承：**先确认判据本身是对的，再去解释结果**。
