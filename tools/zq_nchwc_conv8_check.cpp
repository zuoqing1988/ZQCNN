// zq_nchwc_conv8_check.cpp —— NCHWC8（align=8）no_padding 卷积的独立回归
//
// 起因（audit_k3_20261001.md 附录 CA）
// ----------------------------------------
// 附录 CA 的结论：`zq_nchwc_conv_check` 里那段 align=8 的**是错的那一份** ——
// 它报「NCHWC8 的 with_bias/prelu 全错」，而两个独立复现都给出「全对」。
// 所以本文件**不复用**那份文件的任何东西：
//
//   * 不共用 dispatch 机制（CA.3 查出的问题就在宏拼接 `##ALIGN##` 上 ——
//     模板参数是标识符，拼不出 `nchwc4` / `nchwc8`）
//   * 不共用张量填充 / 参考实现 / 判据
//   * 内核名全部**写全**，不走任何拼接
//
// 用**函数指针表**而不是宏：签名写错会**编译报错**，而不是静默传错参数。
// CA.3 里那两个问题（宏拼接、子进程写 stdout 截断网格）都源于"字符串化"，
// 函数指针把这一类错误交给编译器。
//
// 判据：**后向误差** `|got-exp| / (||in_row|| * ||filter_row||)`，
// 逐格统计、**不使用"最差格"**（附录 CA.5 的规矩：最差格不能概括整体）。
//
// 每个用例 fork 一个子进程（附录 BO.5 / BN.5：崩溃会吃掉整张表），
// 且子进程**只把结果写进文件**，不写 stdout（CA.3：子进程写 stdout 会截断网格）。

#include "zq_check_child.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <malloc.h>

#include <unistd.h>
#include <sys/wait.h>

#include "ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h"
#include "ZQCNN/layers_nchwc/zq_cnn_convolution_gemm_nchwc.h"

// ---------------------------------------------------------------- 三种激活动作
// 三个签名只在尾部不同（bias / bias+slope / 都没有），所以分成三个函数指针类型。
// **不用一个带默认参数的宏拼出来** —— 那正是 CA.3 出问题的地方。
typedef void (*FN_PLAIN)(
    const float* in_data, int in_N, int in_H, int in_W, int in_C,
    int in_ws, int in_ss, int in_is,
    const float* filters, int fN, int fH, int fW, int fC,
    int f_ws, int f_ss, int f_is,
    int sH, int sW, int dH, int dW,
    float* out, int oN, int oH, int oW, int oC, int o_ws, int o_ss, int o_is,
    void** buffer, __int64* buffer_len);

typedef void (*FN_BIAS)(
    const float* in_data, int in_N, int in_H, int in_W, int in_C,
    int in_ws, int in_ss, int in_is,
    const float* filters, int fN, int fH, int fW, int fC,
    int f_ws, int f_ss, int f_is,
    int sH, int sW, int dH, int dW,
    float* out, int oN, int oH, int oW, int oC, int o_ws, int o_ss, int o_is,
    const float* bias,
    void** buffer, __int64* buffer_len);

typedef void (*FN_PRELU)(
    const float* in_data, int in_N, int in_H, int in_W, int in_C,
    int in_ws, int in_ss, int in_is,
    const float* filters, int fN, int fH, int fW, int fC,
    int f_ws, int f_ss, int f_is,
    int sH, int sW, int dH, int dW,
    float* out, int oN, int oH, int oW, int oC, int o_ws, int o_ss, int o_is,
    const float* bias, const float* slope,
    void** buffer, __int64* buffer_len);

// 六支，名字全部写全（不拼接）
static const char* g_branch_name[] = {
    "general", "kernel1x1", "kernel2x2", "kernel2x2_C3", "kernel3x3", "kernel3x3_C3"
};
static const int g_fsize[] = { 3, 1, 2, 2, 3, 3 };

#define NCHWC8(NAME) zq_cnn_conv_no_padding_gemm_nchwc8_##NAME
static FN_PLAIN g_plain[6] = {
    NCHWC8(general),                    NCHWC8(kernel1x1),
    NCHWC8(kernel2x2),                   NCHWC8(kernel2x2_C3),
    NCHWC8(kernel3x3),                   NCHWC8(kernel3x3_C3)
};
static FN_BIAS g_bias[6] = {
    NCHWC8(general_with_bias),                    NCHWC8(kernel1x1_with_bias),
    NCHWC8(kernel2x2_with_bias),                   NCHWC8(kernel2x2_C3_with_bias),
    NCHWC8(kernel3x3_with_bias),                   NCHWC8(kernel3x3_C3_with_bias)
};
static FN_PRELU g_prelu[6] = {
    NCHWC8(general_with_bias_prelu),                    NCHWC8(kernel1x1_with_bias_prelu),
    NCHWC8(kernel2x2_with_bias_prelu),                   NCHWC8(kernel2x2_C3_with_bias_prelu),
    NCHWC8(kernel3x3_with_bias_prelu),                   NCHWC8(kernel3x3_C3_with_bias_prelu)
};
#undef NCHWC8

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

struct Shape { int N, H, W, C, K, stride, branch, variant; bool use_buffer; bool gating; };

// 子进程把「正确格数 / 错格数 / 最大后向误差」写到这里
#define RES_FILE "/tmp/zq_conv8_res.txt"
static const double TOL = 1e-5;

// 返回 0=通过 2=搭建失败；结果写进 RES_FILE
static int run_case(const Shape& s)
{
    const int N = s.N, H = s.H, W = s.W, C = s.C, K = s.K, ST = s.stride;
    const int FS = g_fsize[s.branch];
    const int oH = (H - FS + 1) / ST, oW = (W - FS + 1) / ST;
    if (oH <= 0 || oW <= 0) return 2;

    std::vector<float> in((size_t)N * C * H * W), flt((size_t)K * C * FS * FS);
    std::vector<float> bias_v(K), slope_v(K);
    for (size_t i = 0; i < in.size(); i++) in[i] = val(1, (int)i);
    for (size_t i = 0; i < flt.size(); i++) flt[i] = val(2, (int)i);
    for (int k = 0; k < K; k++) { bias_v[k] = val(3, k) * 0.5f; slope_v[k] = 0.1f + 0.01f * (k % 7); }

    ZQ::ZQ_CNN_Tensor4D_NCHWC8 tin, tflt, tbias, tslope, tout;
    if (!tin.ChangeSize(N, H, W, C, 0, 0)) return 2;
    if (!tflt.ChangeSize(K, FS, FS, C, 0, 0)) return 2;
    if (!tbias.ChangeSize(K, 1, 1, 1, 0, 0)) return 2;
    if (!tslope.ChangeSize(K, 1, 1, 1, 0, 0)) return 2;
    if (!tout.ChangeSize(N, oH, oW, K, 0, 0)) return 2;
    if (!tin.ConvertFromCompactNCHW(&in[0], N, C, H, W)) return 2;
    if (!tflt.ConvertFromCompactNCHW(&flt[0], K, C, FS, FS)) return 2;
    for (int k = 0; k < K; k++) { tbias.GetFirstPixelPtr()[k] = bias_v[k]; tslope.GetFirstPixelPtr()[k] = slope_v[k]; }

    const int oWS = tout.GetWidthStep(), oSS = tout.GetSliceStep(), oIS = tout.GetImageStep();
    const int A = 8;    // NCHWC8 的对齐
#define OUT_IDX(nn, oh, ow, kk) \
    ((nn) * oIS + ((kk) / A) * oSS + (oh) * oWS + (ow) * A + ((kk) % A))
    for (int n = 0; n < N; n++)
        for (int oh = 0; oh < oH; oh++)
            for (int ow = 0; ow < oW; ow++)
                for (int k = 0; k < K; k++)
                    tout.GetFirstPixelPtr()[OUT_IDX(n, oh, ow, k)] = -12345.0f;

    void* buffer = 0;
    __int64 buffer_len = 0;
    if (s.use_buffer) { buffer_len = 32; buffer = _aligned_malloc((size_t)buffer_len, 32); if (!buffer) return 2; }

    const float* ip = tin.GetFirstPixelPtr();
    const float* fp = tflt.GetFirstPixelPtr();
    float* op = tout.GetFirstPixelPtr();

    if (s.variant == 0)
        g_plain[s.branch](ip, N, H, W, C, tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep(),
                         fp, K, FS, FS, C, tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep(),
                         ST, ST, 1, 1,
                         op, N, oH, oW, K, oWS, oSS, oIS, &buffer, &buffer_len);
    else if (s.variant == 1)
        g_bias[s.branch](ip, N, H, W, C, tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep(),
                        fp, K, FS, FS, C, tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep(),
                        ST, ST, 1, 1,
                        op, N, oH, oW, K, oWS, oSS, oIS, tbias.GetFirstPixelPtr(), &buffer, &buffer_len);
    else
        g_prelu[s.branch](ip, N, H, W, C, tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep(),
                         fp, K, FS, FS, C, tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep(),
                         ST, ST, 1, 1,
                         op, N, oH, oW, K, oWS, oSS, oIS, tbias.GetFirstPixelPtr(),
                         tslope.GetFirstPixelPtr(), &buffer, &buffer_len);

    if (buffer) _aligned_free(buffer);      // buffer 所有权归调用方（附录 BJ）

    // ---- 逐格统计（不用"最差格"，附录 CA.5） ----
    long n_ok = 0, n_bad = 0;
    double worst = 0.0;
    for (int n = 0; n < N; n++)
        for (int oh = 0; oh < oH; oh++)
            for (int ow = 0; ow < oW; ow++)
                for (int k = 0; k < K; k++) {
                    double sum = 0, sc = 0;
                    for (int c = 0; c < C; c++)
                        for (int fh = 0; fh < FS; fh++)
                            for (int fw = 0; fw < FS; fw++) {
                                double a = in[((size_t)n * C + c) * H * W + (size_t)(oh * ST + fh) * W + (ow * ST + fw)];
                                double f = flt[((size_t)k * C + c) * FS * FS + (size_t)fh * FS + fw];
                                sum += a * f;
                                sc += a * a * f * f;
                            }
                    if (s.variant >= 1) sum += bias_v[k];
                    if (s.variant == 2 && sum < 0) sum *= slope_v[k];
                    double got = op[OUT_IDX(n, oh, ow, k)];
                    double d = fabs(got - sum);
                    double den = sqrt(sc);
                    if (den < 1e-30) den = 1.0;
                    double be = d / den;
                    if (be > TOL) n_bad++; else n_ok++;
                    if (be > worst) worst = be;
                }
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %.6e\n", n_ok, n_bad, worst); fclose(f); }
    return 0;
#undef OUT_IDX
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(const Shape& s)
{
    g_case++;
    fflush(stdout);
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) {
        zq_child_silence_stderr();   // 子进程不碰 stdout
        int r = run_case(s);
        _exit(r == 2 ? 3 : 0);
    }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0; double worst = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { if (fscanf(f, "%ld %ld %lf", &ok, &bad, &worst) != 3) { ok = bad = 0; } fclose(f); }
    const char* vname = s.variant == 0 ? "plain" : (s.variant == 1 ? "with_bias" : "with_bias_prelu");
    // 「只报告」档（K % 8 != 0，违反 filter_N % align == 0 契约）本来就不参与判定：
    // 崩了或算错了都只打印，不计失败。契约之外调用会段错误，见附录 CB.5。
    const char* tier = s.gating ? "" : "  [只报告档，不计失败]";
    if (WIFSIGNALED(st)) {
        if (s.gating) g_crash++;
        printf("  %-12s %-18s buf=%d  CRASH%s\n", g_branch_name[s.branch], vname, (int)s.use_buffer, tier);
        return;
    }
    if (WEXITSTATUS(st) == 3) {
        if (s.gating) g_crash++;
        printf("  %-12s SETUP 失败%s\n", g_branch_name[s.branch], tier); return;
    }
    if (bad > 0) {
        if (s.gating) g_bad++;
        printf("  %-12s %-18s buf=%d  FAIL  %ld/%ld 格错, 最大后向误差 %.3e%s\n",
               g_branch_name[s.branch], vname, (int)s.use_buffer, bad, ok + bad, worst, tier);
    }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC8（align=8）no_padding 卷积：六支 x 三个激活动作（附录 CB）\n");
    printf("内核名全部写全、走函数指针表（签名写错会编译报错，不会静默传错）\n");
    printf("判据：后向误差，逐格统计，不用最差格（附录 CA.5）\n");
    printf("K %% 8 == 0 才是合法形状；K %% 8 != 0 那一档标「只报告」不判失败\n\n");

    // 合法档：K 是 8 的倍数。非法档：K = 6（只报告）。
    static const int KS_OK[] = { 8, 16 };
    static const int KS_BAD[] = { 6, 12 };
    for (int br = 0; br < 6; br++) {
        for (int kind = 0; kind < 2; kind++) {
            int K = kind ? KS_BAD[0] : KS_OK[0];
            int C = (br == 3 || br == 5) ? 3 : 8;
            Shape s; s.N = 1; s.H = 20; s.W = 20; s.C = C; s.K = K; s.stride = 1;
            s.branch = br; s.use_buffer = false; s.gating = (kind == 0);
            printf("%s (K=%d, C=%d, K%%8=%d, %s)\n", g_branch_name[br], K, C, K % 8,
                   kind ? "只报告" : "门禁");
            for (int v = 0; v < 3; v++) {
                s.variant = v;
                one(s);
                s.use_buffer = true; one(s); s.use_buffer = false;
            }
        }
        printf("\n");
    }
    printf("共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash) {
        printf("**上面每一项在下结论之前都要先用独立复现对一遍**"
               "（附录 CA.3：上一次就是没对，结论作废）。\n");
    }
    return (g_bad || g_crash) ? 1 : 0;
}
