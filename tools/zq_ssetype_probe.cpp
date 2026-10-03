// SSETYPE 四档的行为探针（附录 GM）。
//
// 由 `tools/check_ssetype_matrix.py` 在四种 `-DZQ_CNN_USE_SSETYPE` 下
// 各编一遍 ZQ_GEMM 并运行本文件，逐档断言**后向误差**在容差内。
//
// 为什么要有它
// ------------
// 附录 H4 记着「`ZQ_CNN_SSETYPE_NONE`（=0）在 x86 上**编不过**」，
// 还把"有 `#else` 走标量"的那处归给了 `zq_gemm_32f_auto.c`
// （实际在 `zq_gemm_32f_align_c.c`，而 `auto.c` 的派发链是
// `#if AVX / #elif SSE / #else align0`）。2026-10-03 实测：**四档全部
// 编译通过、链接通过、后向误差都在 1e-8 量级**。
//
// 一个**记错的遗留项**比一个没记的遗留项更贵 —— 它会让人以为有活要干，
// 于是有人照着去"修"一个不存在的问题。所以配置空间要**测**，
// 不是"看一眼默认值就跳过"。
//
// 三条自己踩过的坑写在下面，**别再踩**：
//   1. 缓冲区必须 **64 字节对齐**。align256bit 内核用对齐加载，
//      用 malloc（glibc 只保证 16 字节）时 AVX/AVX2 两档**直接段错误**。
//      而那两档正是默认配置、全量回归跑得好好的 ——
//      所以那个段错误是**探针违约**，不是库的缺陷。
//   2. 编译旗标要用**真实的** `-mavx2 -mfma`（根 CMakeLists.txt:113 对
//      所有 gcc x86 构建都加，与 SSETYPE 无关）。第一版按档位配
//      `-mavx` / `-msse4.2`，于是 AVX 档凭空挂了一个 TU ——
//      那是**测试写错**。
//   3. 判据用**后向误差**（AGENTS.md：GEMM 只能用后向误差），
//      而且 C 预先填成 NaN：万一某个档位真的什么都不写，NaN 会留在结果里，
//      比"误差很大"更容易一眼认出来。
#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" {
void zq_gemm_32f_AnoTrans_Btrans_auto(int M, int N, int K, const float *A,
                                      int lda, const float *Bt, int ldb,
                                      float *C, int ldc);
// 汇编入口（附录 GR）。它此前**只有两个对比 sample 会调**，
// 而 sample 只在默认档跑 —— 于是这一整条入口、以及它内部的
// `ZQ_GEMM_ISA=off` 强制回落分支都**从未被门禁验过**。
// SSETYPE=0/1 时 ZQA_IMPL=0，这个入口就是 ZQA_FALLBACK（转发到上面那个），
// 所以低两档顺带把回落路径也覆盖了。
void zq_gemm_32f_AnoTrans_Btrans_auto_asm(int M, int N, int K, const float *A,
                                          int lda, const float *Bt, int ldb,
                                          float *C, int ldc);
}

static void ref_gemm(int M, int N, int K, const float *A, int lda,
                     const float *Bt, int ldb, float *C, int ldc)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double s = 0.0;
            for (int k = 0; k < K; k++)
                s += (double)A[m * lda + k] * (double)Bt[n * ldb + k];
            C[m * ldc + n] = (float)s;
        }
}

static double back_err(int M, int N, int K, const float *A, int lda,
                       const float *Bt, int ldb, const float *C, int ldc)
{
    double num = 0, an = 0, bn = 0;
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++)
            an += (double)A[m * lda + k] * A[m * lda + k];
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
            bn += (double)Bt[n * ldb + k] * Bt[n * ldb + k];
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double s = 0.0;
            for (int k = 0; k < K; k++)
                s += (double)A[m * lda + k] * (double)Bt[n * ldb + k];
            double d = s - (double)C[m * ldc + n];
            num += d * d;
        }
    return sqrt(num) / (sqrt(an) * sqrt(bn) + 1e-30);
}

static float *aligned_buf(size_t nfloat)
{
    size_t bytes = ((nfloat * sizeof(float)) + 63) / 64 * 64;
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0)
        return NULL;
    return (float *)p;
}

int main()
{
    // 刻意混入 K 不是任何向量宽度整数倍的形状（3 / 7 / 9 / 12 / 32），
    // 好让"每档各走各的路径"这件事真的被覆盖到。
    static const int shapes[][3] = {
        {8, 8, 12}, {16, 16, 32}, {4, 6, 3},
        {32, 8, 9},  {1, 8, 7},   {64, 64, 64},
    };
    const int ns = (int)(sizeof(shapes) / sizeof(shapes[0]));
    const double TOL = 1e-4;          // float32 的后向误差量级是 1e-8~1e-7
    int bad = 0;
    // 两条入口都要验：intrinsic 派发器 与 汇编入口（附录 GR）。
    for (int which = 0; which < 2; which++) {
        const char *tag = which ? "汇编入口" : "intrinsic ";
        for (int t = 0; t < ns; t++) {
            int M = shapes[t][0], N = shapes[t][1], K = shapes[t][2];
            float *A = aligned_buf((size_t)M * K);
            float *B = aligned_buf((size_t)N * K);
            float *C = aligned_buf((size_t)M * N);
            float *R = (float *)malloc(sizeof(float) * M * N);
            if (!A || !B || !C || !R) { printf("ALLOC-FAIL\n"); return 2; }
            for (int i = 0; i < M * K; i++) A[i] = (float)((i * 37 % 17) - 8) / 8.0f;
            for (int i = 0; i < N * K; i++) B[i] = (float)((i * 53 % 23) - 11) / 11.0f;
            for (int i = 0; i < M * N; i++) C[i] = (float)NAN;   // 见坑 3
            ref_gemm(M, N, K, A, K, B, K, R, N);
            if (which)
                zq_gemm_32f_AnoTrans_Btrans_auto_asm(M, N, K, A, K, B, K, C, N);
            else
                zq_gemm_32f_AnoTrans_Btrans_auto(M, N, K, A, K, B, K, C, N);
            double e = back_err(M, N, K, A, K, B, K, C, N);
            int nan = 0;
            for (int i = 0; i < M * N; i++) if (std::isnan(C[i])) nan = 1;
            printf("  %s M=%-3d N=%-3d K=%-3d 后向误差=%-12.4g NaN残留=%s\n",
                   tag, M, N, K, e, nan ? "是" : "否");
            if (e > TOL || nan) bad++;
            free(A); free(B); free(C); free(R);
        }
    }
    printf("%s\n", bad ? "本档有超差或 NaN 残留" : "本档全部在容差内");
    return bad ? 1 : 0;
}
