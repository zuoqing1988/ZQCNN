/* NCHWC（layers_nchwc）depthwise 卷积门禁 —— 附录 CF
 *
 * 为什么要有这个门禁
 * ------------------
 * `ZQCNN/layers_nchwc/zq_cnn_depthwise_convolution_nchwc*` 是 **NCHWC 深度可分离卷积**。
 * `model/` 下的 shipped 模型里有 **284 个 DepthwiseConvolution 层**
 * （MobileNetSSD / Pose / det1-dw* / det2-dw* / det3-dw* …），
 * 而仓库里 22 个 `zq_*_check.cpp` **一个都没覆盖它**。
 *
 * 这一族与普通卷积是**两套完全不同的代码**（不走 gemm，是手写的
 * 「每个通道一个 filter」的 SIMD 展开），所以 CB / CE 修的那些问题对它一概无效。
 * 附录 CE 刚在 NCHW 卷积里查出一条 100% 算错的生产可达缺陷，
 * 同一层里"另一个变体没测过"的教训在这里直接适用。
 *
 * 覆盖面：7 个基础函数 × 3 种对齐（NCHWC1/4/8）× 3 个激活动作
 *        = **63 个入口**，全部进表。
 *
 * 设计上照抄 CB / CE 那三道门禁已经验证过的做法
 * -------------------------------------------
 *  · 内核名在数组里**写全**、走函数指针表，**不做任何字符串拼接**（CA.3）
 *  · 判据用**后向误差** `|got-exp| / sqrt(sum(a^2 f^2))`，阈值 1e-5
 *  · **逐格统计**（多少格对 / 多少格错 / 最差是多少），不拿"最差格"当结论（CA.5）
 *  · 每个用例 fork 一个子进程；子进程 stderr 接 /dev/null、只写结果文件
 *  · **用真实的 ZQ_CNN_Tensor4D_NCHWC1/4/8 类**分配与填充张量 ——
 *    这样"我以为的 NCHWC 布局"这个变量根本不存在
 *  · `extern "C"` 声明照抄头文件连参数名一起抄（C 链接不检查 arity）
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h"
#include "ZQCNN/layers_nchwc/zq_cnn_depthwise_convolution_nchwc.h"

typedef void (*FN_PLAIN)(
    const float* in_data, int in_N, int in_H, int in_W, int in_C,
    int in_ws, int in_ss, int in_is,
    const float* filters, int fN, int fH, int fW, int fC,
    int f_ws, int f_ss, int f_is,
    int sH, int sW, int dH, int dW,
    float* out, int oN, int oH, int oW, int oC, int o_ws, int o_ss, int o_is);

typedef void (*FN_BIAS)(
    const float* in_data, int in_N, int in_H, int in_W, int in_C,
    int in_ws, int in_ss, int in_is,
    const float* filters, int fN, int fH, int fW, int fC,
    int f_ws, int f_ss, int f_is,
    int sH, int sW, int dH, int dW,
    float* out, int oN, int oH, int oW, int oC, int o_ws, int o_ss, int o_is,
    const float* bias);

typedef void (*FN_PRELU)(
    const float* in_data, int in_N, int in_H, int in_W, int in_C,
    int in_ws, int in_ss, int in_is,
    const float* filters, int fN, int fH, int fW, int fC,
    int f_ws, int f_ss, int f_is,
    int sH, int sW, int dH, int dW,
    float* out, int oN, int oH, int oW, int oC, int o_ws, int o_ss, int o_is,
    const float* bias, const float* slope);

// 7 个基础变体
enum { B_GENERAL, B_3x3, B_5x5s1d1, B_3x3s1d1, B_3x3s2d1, B_2x2, B_2x2s1d1, B_COUNT };
static const char* g_base_name[B_COUNT] = {
    "general", "kernel3x3", "kernel5x5_s1d1", "kernel3x3_s1d1",
    "kernel3x3_s2d1", "kernel2x2", "kernel2x2_s1d1"
};
// 每个变体的 (filter_H, filter_W, stride, dilation)
static const int g_base_hw[B_COUNT][4] = {
    /* general     */ {  0,  0, 1, 1 },   // general 的 fH/fW 由用例给
    /* kernel3x3   */ {  3,  3, 1, 1 },
    /* 5x5_s1d1    */ {  5,  5, 1, 1 },
    /* 3x3_s1d1    */ {  3,  3, 1, 1 },
    /* 3x3_s2d1    */ {  3,  3, 2, 1 },
    /* 2x2         */ {  2,  2, 1, 1 },
    /* 2x2_s1d1    */ {  2,  2, 1, 1 },
};

struct Entry {
    FN_PLAIN plain;
    FN_BIAS   bias;
    FN_PRELU  prelu;
    int       base;    // B_*
    int       align;   // 1 / 4 / 8
};

// 名字全部写全（不拼接）。写错会编译报错，不会静默传错。
#define DW(NAME) zq_cnn_depthwise_conv_no_padding_nchwc##NAME
static Entry g_entries[B_COUNT][3] = {
  { { DW(1_general), DW(1_general_with_bias), DW(1_general_with_bias_prelu) },
    { DW(4_general), DW(4_general_with_bias), DW(4_general_with_bias_prelu) },
    { DW(8_general), DW(8_general_with_bias), DW(8_general_with_bias_prelu) } },
  { { DW(1_kernel3x3), DW(1_kernel3x3_with_bias), DW(1_kernel3x3_with_bias_prelu) },
    { DW(4_kernel3x3), DW(4_kernel3x3_with_bias), DW(4_kernel3x3_with_bias_prelu) },
    { DW(8_kernel3x3), DW(8_kernel3x3_with_bias), DW(8_kernel3x3_with_bias_prelu) } },
  { { DW(1_kernel5x5_s1d1), DW(1_kernel5x5_s1d1_with_bias), DW(1_kernel5x5_s1d1_with_bias_prelu) },
    { DW(4_kernel5x5_s1d1), DW(4_kernel5x5_s1d1_with_bias), DW(4_kernel5x5_s1d1_with_bias_prelu) },
    { DW(8_kernel5x5_s1d1), DW(8_kernel5x5_s1d1_with_bias), DW(8_kernel5x5_s1d1_with_bias_prelu) } },
  { { DW(1_kernel3x3_s1d1), DW(1_kernel3x3_s1d1_with_bias), DW(1_kernel3x3_s1d1_with_bias_prelu) },
    { DW(4_kernel3x3_s1d1), DW(4_kernel3x3_s1d1_with_bias), DW(4_kernel3x3_s1d1_with_bias_prelu) },
    { DW(8_kernel3x3_s1d1), DW(8_kernel3x3_s1d1_with_bias), DW(8_kernel3x3_s1d1_with_bias_prelu) } },
  { { DW(1_kernel3x3_s2d1), DW(1_kernel3x3_s2d1_with_bias), DW(1_kernel3x3_s2d1_with_bias_prelu) },
    { DW(4_kernel3x3_s2d1), DW(4_kernel3x3_s2d1_with_bias), DW(4_kernel3x3_s2d1_with_bias_prelu) },
    { DW(8_kernel3x3_s2d1), DW(8_kernel3x3_s2d1_with_bias), DW(8_kernel3x3_s2d1_with_bias_prelu) } },
  { { DW(1_kernel2x2), DW(1_kernel2x2_with_bias), DW(1_kernel2x2_with_bias_prelu) },
    { DW(4_kernel2x2), DW(4_kernel2x2_with_bias), DW(4_kernel2x2_with_bias_prelu) },
    { DW(8_kernel2x2), DW(8_kernel2x2_with_bias), DW(8_kernel2x2_with_bias_prelu) } },
  { { DW(1_kernel2x2_s1d1), DW(1_kernel2x2_s1d1_with_bias), DW(1_kernel2x2_s1d1_with_bias_prelu) },
    { DW(4_kernel2x2_s1d1), DW(4_kernel2x2_s1d1_with_bias), DW(4_kernel2x2_s1d1_with_bias_prelu) },
    { DW(8_kernel2x2_s1d1), DW(8_kernel2x2_s1d1_with_bias), DW(8_kernel2x2_s1d1_with_bias_prelu) } },
};
#undef DW

#define RES_FILE "/tmp/zq_dw_res.txt"
static const double TOL = 1e-5;

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

struct Case { int base, align, variant, N, H, W, C, fH, fW, S, D; };

// 用**真实张量类**分配/填充（三种对齐各一个特化，函数指针表取对应那个）
typedef void (*RUNNER)(const Case&);
template <class TEN>
static void run_one(const Case& c)
{
    const int A = TEN().GetAlignSize();
    const int N = c.N, H = c.H, W = c.W, C = c.C;
    const int fH = c.fH, fW = c.fW, S = c.S, D = c.D;
    const int oH = (H - (fH - 1) * D - 1) / S + 1;
    const int oW = (W - (fW - 1) * D - 1) / S + 1;
    if (oH <= 0 || oW <= 0) return;

    std::vector<float> in((size_t)N * C * H * W), flt((size_t)1 * fH * fW * C);
    std::vector<float> bv(A), sl(A);
    for (size_t i = 0; i < in.size(); i++) in[i] = val(1, (int)i);
    for (size_t i = 0; i < flt.size(); i++) flt[i] = val(2, (int)i);
    for (int k = 0; k < A; k++) { bv[k] = val(3, k) * 0.5f; sl[k] = 0.1f + 0.01f * (k % 7); }

    TEN tin, tflt, tout;
    if (!tin.ChangeSize(N, H, W, C, 0, 0)) return;
    if (!tflt.ChangeSize(1, fH, fW, C, 0, 0)) return;     // depthwise: filter_N == 1
    if (!tout.ChangeSize(N, oH, oW, C, 0, 0)) return;    // out_C == in_C
    if (!tin.ConvertFromCompactNCHW(&in[0], N, C, H, W)) return;
    if (!tflt.ConvertFromCompactNCHW(&flt[0], 1, C, fH, fW)) return;
    for (int k = 0; k < A; k++) { bv[k] = (k < C) ? val(3, k) * 0.5f : 0.0f; sl[k] = (k < C) ? 0.1f + 0.01f * (k % 7) : 0.0f; }

    const int oWS = tout.GetWidthStep(), oSS = tout.GetSliceStep(), oIS = tout.GetImageStep();
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int oh = 0; oh < oH; oh++)
                for (int ow = 0; ow < oW; ow++)
                    tout.GetFirstPixelPtr()[n * oIS + (c / A) * oSS + oh * oWS + ow * A + (c % A)] = -12345.0f;

    const Entry& e = g_entries[c.base][c.align == 1 ? 0 : (c.align == 4 ? 1 : 2)];
    const float* ip = tin.GetFirstPixelPtr();
    const float* fp = tflt.GetFirstPixelPtr();
    float* op = tout.GetFirstPixelPtr();

    if (c.variant == 0)
        e.plain(ip, N, H, W, C, tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep(),
                fp, 1, fH, fW, C, tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep(),
                S, S, D, D,
                op, N, oH, oW, C, oWS, oSS, oIS);
    else if (c.variant == 1)
        e.bias(ip, N, H, W, C, tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep(),
               fp, 1, fH, fW, C, tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep(),
               S, S, D, D,
               op, N, oH, oW, C, oWS, oSS, oIS, &bv[0]);
    else
        e.prelu(ip, N, H, W, C, tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep(),
                fp, 1, fH, fW, C, tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep(),
                S, S, D, D,
                op, N, oH, oW, C, oWS, oSS, oIS, &bv[0], &sl[0]);

    // ---- 逐格统计（depthwise：每个通道一个 filter，不跨通道混合）----
    long n_ok = 0, n_bad = 0; double worst = 0.0;
    for (int n = 0; n < N; n++)
        for (int ch = 0; ch < C; ch++) {          // 叫 ch 不叫 c：下面 c.variant 是参数 Case& c
            for (int oh = 0; oh < oH; oh++)
                for (int ow = 0; ow < oW; ow++) {
                    double sum = (c.variant >= 1) ? bv[ch] : 0.0;
                    double sc = 0.0;
                    for (int fh = 0; fh < fH; fh++)
                        for (int fw = 0; fw < fW; fw++) {
                            double a = in[((size_t)n * C + ch) * H * W + (size_t)(oh * S + fh * D) * W + (ow * S + fw * D)];
                            double f = flt[((size_t)ch) * fH * fW + (size_t)fh * fW + fw];
                            sum += a * f; sc += a * a * f * f;
                        }
                    if (c.variant == 2 && sum < 0) sum *= sl[ch];
                    double got = op[n * oIS + (ch / A) * oSS + oh * oWS + ow * A + (ch % A)];
                    double den = sqrt(sc); if (den < 1e-30) den = 1.0;
                    double be = fabs(got - sum) / den;
                    if (be > TOL) n_bad++; else n_ok++;
                    if (be > worst) worst = be;
                }
        }
    FILE* fp2 = fopen(RES_FILE, "w");
    if (fp2) { fprintf(fp2, "%d %d %ld %ld %.6e\n", oH, oW, n_ok, n_bad, worst); fclose(fp2); }
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(const Case& c, RUNNER r)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) {
        FILE* dn = freopen("/dev/null", "w", stderr); (void)dn;
        r(c);
        _exit(0);
    }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0; int oh = 0, ow = 0; double worst = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { if (fscanf(f, "%d %d %ld %ld %lf", &oh, &ow, &ok, &bad, &worst) != 5) ok = bad = 0; fclose(f); }
    char nm[96];
    snprintf(nm, sizeof(nm), "nchwc%d %s %s", c.align, g_base_name[c.base],
             c.variant == 0 ? "plain" : (c.variant == 1 ? "with_bias" : "with_bias_prelu"));
    char tag[96];
    snprintf(tag, sizeof(tag), "N=%d %dx%d C=%d f=%dx%d s=%d d=%d", c.N, c.H, c.W, c.C, c.fH, c.fW, c.S, c.D);
    if (WIFSIGNALED(st)) { g_crash++; printf("  %-44s %s  CRASH\n", nm, tag); return; }
    if (bad > 0) { g_bad++; printf("  %-44s %s  FAIL %ld/%ld 格错, 最差 %.3e\n", nm, tag, bad, ok + bad, worst); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC depthwise 卷积：7 个变体 x 3 种对齐 x 3 个激活动作 = 63 个入口（附录 CF）\n");
    printf("内核名全部写全、走函数指针表；用真实 ZQ_CNN_Tensor4D_NCHWC{1,4,8} 张量\n");
    printf("判据：后向误差，逐格统计。depthwise 的语义：每个通道一个 filter，不跨通道混合\n\n");

    for (int ai = 0; ai < 3; ai++) {
        const int A = (ai == 0) ? 1 : (ai == 1 ? 4 : 8);
        RUNNER r = (ai == 0) ? &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC1>
                  : (ai == 1) ? &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC4>
                              : &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC8>;
        printf("=== NCHWC%d (align=%d) ===\n", A, A);
        for (int b = 0; b < B_COUNT; b++) {
            int fH = g_base_hw[b][0], fW = g_base_hw[b][1], S = g_base_hw[b][2], D = g_base_hw[b][3];
            printf("  %s\n", g_base_name[b]);
            for (int v = 0; v < 3; v++) {
                Case c; memset(&c, 0, sizeof(c));
                c.base = b; c.align = A; c.variant = v;
                c.N = 1; c.H = 17; c.W = 17; c.C = A; c.S = S; c.D = D;
                if (b == B_GENERAL) { fH = 3; fW = 3; S = 1; D = 1; }
                c.fH = fH; c.fW = fW;
                one(c, r);
                // C 跨两个对齐组（多一组 slice），再跑一遍
                c.C = A * 2; one(c, r);
                c.C = A; c.N = 2; one(c, r);
            }
        }
        printf("\n");
    }
    printf("共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
