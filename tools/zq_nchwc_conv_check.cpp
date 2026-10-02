// zq_nchwc_conv_check.cpp —— NCHWC「no_padding」卷积族的 kernel 契约图（附录 BS / BT / BW / BZ）
//
// 起因（audit_k3_20261001.md 附录 BS 起）
// ------------------------------------------
// 附录 BS 覆盖了 NCHWC4 的 3x3 两支，BT 把它扩成**六支**（general / kernel1x1 /
// kernel2x2 / kernel2x2_C3 / kernel3x3 / kernel3x3_C3）x 3 个激活动作的契约图，
// 定位到 `filter_N % 4 == 0` 这条统一契约，并把 kernel2x2_C3 的根因查清（附录 BX）。
//
// BZ 把它扩到 **align=8**（NCHWC8）那一整套：对齐宽度从 4 变成 8，
// 布局公式、补齐槽位数、内核名**全都不一样**，而
// `SampleLnet106` / `SampleSphereFaceNet` 两个 sample 就是走 NCHWC8 的，
// 所以它不是"死代码"而是**生产在跑、零测试覆盖**。
//
// 为什么"契约图"是这个测试的形态
// ------------------------------
// 缺陷是**静默算错**（不是崩），而判据只能用后向误差（附录 BO.3：相对误差对抵消敏感）。
// 每个用例 fork 一个子进程（附录 BO.5 / BN.5：崩溃会吃掉整张表）。
//
// 故意违约的形状（filter_N%4 != 0 那一档）标成「只报告、不判失败」：
// 门禁要 pin 住的是**「合法形状必须算对」**，不是「非法形状必须算错」——
// 后者会在有人把 col2im 的尾巴补好之后把门禁变红。
//
// pad：只测 pad=0。no_padding 族本身不含 padding 逻辑，调用方自己把指针挪过边框。
// 带 pad 的那部分记在附录 BT.5，**不假装验过**。

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

static int nbad = 0, ninfo = 0, ncrash = 0, ncase = 0;

// 附录 BW：故意往**输入张量的补齐通道**里填非 0 值，看结果会不会变。
static float g_poison = 0.0f;
// 子进程把「输出签名 + 后向误差」写到这里，父进程 waitpid 之后读（附录 BW）
#define CHK_FILE "/tmp/zq_conv_chk.txt"

// kernel 种类
enum { K_GEN = 0, K_1x1, K_2x2, K_2x2_C3, K_3x3, K_3x3_C3 };
static const char* k_name[] = { "general", "kernel1x1", "kernel2x2", "kernel2x2_C3", "kernel3x3", "kernel3x3_C3" };
static int k_fsize[] = { 3, 1, 2, 2, 3, 3 };   // general 用 3x3
static const char* g_vname_of(int b) { return (b >= 0 && b <= 5) ? k_name[b] : "?"; }

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

struct Shape { int N, H, W, C, K, stride, kernel; };

// ---- 内核名拼接：ALIGN 必须是 4 或 8 这样的**字面量**（所以在调用处展开） ----
// 注意：IN_A / FILT_A / MID_A / OUT_A 这些宏**在分支宏体内部**用，不作为参数传 ——
// 它们本身含逗号，当成可变参数传会把宏参数个数撑爆（g++ 直接报
// "macro passed 23 arguments, but takes just 5"）。
#define ZQ_BRANCH_PLAIN(ALIGN, B)                                                  \
    switch (B) {                                                                   \
    case K_1x1:     zq_cnn_conv_no_padding_gemm_nchwc##ALIGN##_kernel1x1(           \
                      IN_A, FILT_A, MID_A, OUT_A, &buffer, &buffer_len); break;      \
    case K_2x2:     zq_cnn_conv_no_padding_gemm_nchwc##ALIGN##_kernel2x2(           \
                      IN_A, FILT_A, MID_A, OUT_A, &buffer, &buffer_len); break;      \
    case K_2x2_C3:  zq_cnn_conv_no_padding_gemm_nchwc##ALIGN##_kernel2x2_C3(        \
                      IN_A, FILT_A, MID_A, OUT_A, &buffer, &buffer_len); break;      \
    case K_3x3:     zq_cnn_conv_no_padding_gemm_nchwc##ALIGN##_kernel3x3(           \
                      IN_A, FILT_A, MID_A, OUT_A, &buffer, &buffer_len); break;      \
    case K_3x3_C3:  zq_cnn_conv_no_padding_gemm_nchwc##ALIGN##_kernel3x3_C3(        \
                      IN_A, FILT_A, MID_A, OUT_A, &buffer, &buffer_len); break;      \
    default:         zq_cnn_conv_no_padding_gemm_nchwc##ALIGN##_general(             \
                      IN_A, FILT_A, MID_A, OUT_A, &buffer, &buffer_len); break;      \
    }

#define ZQ_BRANCH_BIAS(ALIGN, B)                                                  \
    switch (B) {                                                                   \
    case K_1x1:     zq_cnn_conv_no_padding_gemm_nchwc##ALIGN##_kernel1x1_with_bias(     \
                      IN_A, FILT_A, MID_A, OUT_A, bp, &buffer, &buffer_len); break;     \
    case K_2x2:     zq_cnn_conv_no_padding_gemm_nchwc##ALIGN##_kernel2x2_with_bias(     \
                      IN_A, FILT_A, MID_A, OUT_A, bp, &buffer, &buffer_len); break;     \
    case K_2x2_C3:  zq_cnn_conv_no_padding_gemm_nchwc##ALIGN##_kernel2x2_C3_with_bias(  \
                      IN_A, FILT_A, MID_A, OUT_A, bp, &buffer, &buffer_len); break;     \
    case K_3x3:     zq_cnn_conv_no_padding_gemm_nchwc##ALIGN##_kernel3x3_with_bias(     \
                      IN_A, FILT_A, MID_A, OUT_A, bp, &buffer, &buffer_len); break;     \
    case K_3x3_C3:  zq_cnn_conv_no_padding_gemm_nchwc##ALIGN##_kernel3x3_C3_with_bias(  \
                      IN_A, FILT_A, MID_A, OUT_A, bp, &buffer, &buffer_len); break;     \
    default:         zq_cnn_conv_no_padding_gemm_nchwc##ALIGN##_general_with_bias(       \
                      IN_A, FILT_A, MID_A, OUT_A, bp, &buffer, &buffer_len); break;     \
    }

#define ZQ_BRANCH_PRELU(ALIGN, B)                                                 \
    switch (B) {                                                                   \
    case K_1x1:     zq_cnn_conv_no_padding_gemm_nchwc##ALIGN##_kernel1x1_with_bias_prelu(     \
                      IN_A, FILT_A, MID_A, OUT_A, bp, sp, &buffer, &buffer_len); break;     \
    case K_2x2:     zq_cnn_conv_no_padding_gemm_nchwc##ALIGN##_kernel2x2_with_bias_prelu(     \
                      IN_A, FILT_A, MID_A, OUT_A, bp, sp, &buffer, &buffer_len); break;     \
    case K_2x2_C3:  zq_cnn_conv_no_padding_gemm_nchwc##ALIGN##_kernel2x2_C3_with_bias_prelu(  \
                      IN_A, FILT_A, MID_A, OUT_A, bp, sp, &buffer, &buffer_len); break;     \
    case K_3x3:     zq_cnn_conv_no_padding_gemm_nchwc##ALIGN##_kernel3x3_with_bias_prelu(     \
                      IN_A, FILT_A, MID_A, OUT_A, bp, sp, &buffer, &buffer_len); break;     \
    case K_3x3_C3:  zq_cnn_conv_no_padding_gemm_nchwc##ALIGN##_kernel3x3_C3_with_bias_prelu(  \
                      IN_A, FILT_A, MID_A, OUT_A, bp, sp, &buffer, &buffer_len); break;     \
    default:         zq_cnn_conv_no_padding_gemm_nchwc##ALIGN##_general_with_bias_prelu(       \
                      IN_A, FILT_A, MID_A, OUT_A, bp, sp, &buffer, &buffer_len); break;     \
    }

#define ZQ_DISPATCH_ALL(ALIGN)                                                  \
    do {                                                                           \
        if (variant == 0)      { ZQ_BRANCH_PLAIN(ALIGN, s.kernel) }                 \
        else if (variant == 1) { ZQ_BRANCH_BIAS (ALIGN, s.kernel) }                 \
        else                   { ZQ_BRANCH_PRELU(ALIGN, s.kernel) }                 \
    } while (0)

// 返回 0 = 通过；非 0 = 结果不对（2 = 搭建失败）。崩溃由父进程判信号。
// ALIGN 是这一支的对齐（4 或 8）—— 布局公式、补齐槽位数、内核名都由它决定。
template<class TT, int ALIGN>
static int run_case(const Shape& s, int variant, bool use_buffer)
{
    const int N = s.N, H = s.H, W = s.W, C = s.C, K = s.K, ST = s.stride;
    const int FS = k_fsize[s.kernel];
    const int oH = (H - FS + 1) / ST, oW = (W - FS + 1) / ST;   // dilation=1
    if (oH <= 0 || oW <= 0) return 0;

    std::vector<float> in_nchw((size_t)N * C * H * W);
    std::vector<float> flt_nchw((size_t)K * C * FS * FS);
    std::vector<float> bias_v(K), slope_v(K);
    for (size_t i = 0; i < in_nchw.size(); i++) in_nchw[i] = val(1, (int)i);
    for (size_t i = 0; i < flt_nchw.size(); i++) flt_nchw[i] = val(2, (int)i);
    for (int k = 0; k < K; k++) {
        bias_v[k] = val(3, k) * 0.5f;
        slope_v[k] = 0.1f + 0.01f * (k % 7);
    }

    TT tin, tflt, tbias, tslope, tout;
    if (!tin.ChangeSize(N, H, W, C, 0, 0)) return 2;
    if (!tflt.ChangeSize(K, FS, FS, C, 0, 0)) return 2;
    if (!tbias.ChangeSize(K, 1, 1, 1, 0, 0)) return 2;
    if (!tslope.ChangeSize(K, 1, 1, 1, 0, 0)) return 2;
    if (!tout.ChangeSize(N, oH, oW, K, 0, 0)) return 2;
    if (!tin.ConvertFromCompactNCHW(&in_nchw[0], N, C, H, W)) return 2;
    if (!tflt.ConvertFromCompactNCHW(&flt_nchw[0], K, C, FS, FS)) return 2;

    // 往输入的**补齐通道**（k >= C）填非 0 值（附录 BW）。ALIGN=8 时补齐 5 个通道。
    if (g_poison != 0.0f) {
        const int iWS = tin.GetWidthStep(), iSS = tin.GetSliceStep(), iIS = tin.GetImageStep();
        for (int n = 0; n < N; n++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    for (int k = C; k < ALIGN; k++)
                        tin.GetFirstPixelPtr()[n * iIS + (k / ALIGN) * iSS + h * iWS
                                                + w * ALIGN + (k % ALIGN)] = g_poison;
    }

    const int oWS = tout.GetWidthStep(), oSS = tout.GetSliceStep(), oIS = tout.GetImageStep();
#define OUT_IDX(nn, ohh, oww, kk) \
    ((nn) * oIS + ((kk) / ALIGN) * oSS + (ohh) * oWS + (oww) * ALIGN + ((kk) % ALIGN))
    for (int n = 0; n < N; n++)
        for (int oh = 0; oh < oH; oh++)
            for (int ow = 0; ow < oW; ow++)
                for (int k = 0; k < K; k++)
                    tout.GetFirstPixelPtr()[OUT_IDX(n, oh, ow, k)] = -12345.0f;

    void* buffer = 0;
    __int64 buffer_len = 0;
    if (use_buffer) {
        buffer_len = 32;                       // 故意给很少，看内核会不会自己扩容
        buffer = _aligned_malloc((size_t)buffer_len, 32);
        if (buffer == 0) return 2;
    }

    const float* ip = tin.GetFirstPixelPtr();
    const float* fp = tflt.GetFirstPixelPtr();
    float* op = tout.GetFirstPixelPtr();
    const float* bp = tbias.GetFirstPixelPtr();
    const float* sp = tslope.GetFirstPixelPtr();

#define IN_A    ip, N, H, W, C, tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep()
#define FILT_A  fp, K, FS, FS, C, tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep()
#define OUT_A   op, N, oH, oW, K, oWS, oSS, oIS
#define MID_A   ST, ST, 1, 1

    // `##ALIGN##` 只能拼**预处理记号**，而 ALIGN 在这里是**模板参数**（是标识符，
    // 不是 4/8 这样的字面量），所以必须在这里用字面量再展开一次。
    // ALIGN 是编译期常量，多余的那一支会被优化掉。
    if (ALIGN == 4) { ZQ_DISPATCH_ALL(4); }
    else            { ZQ_DISPATCH_ALL(8); }
#undef IN_A
#undef FILT_A
#undef OUT_A
#undef MID_A

    if (buffer) _aligned_free(buffer);      // buffer 所有权归调用方（附录 BJ）

    double max_rel = 0.0, chk = 0.0, g_worst = 0, e_worst = 0;
    int wn = -1, wh = -1, ww = -1, wk = -1;
    for (int n = 0; n < N; n++)
        for (int oh = 0; oh < oH; oh++)
            for (int ow = 0; ow < oW; ow++)
                for (int k = 0; k < K; k++) {
                    chk += fabs(op[OUT_IDX(n, oh, ow, k)]);
                    double sum = 0, sc = 0;
                    for (int c = 0; c < C; c++)
                        for (int fh = 0; fh < FS; fh++)
                            for (int fw = 0; fw < FS; fw++) {
                                double a = in_nchw[((size_t)n * C + c) * H * W + (size_t)(oh * ST + fh) * W + (ow * ST + fw)];
                                double f = flt_nchw[((size_t)k * C + c) * FS * FS + (size_t)fh * FS + fw];
                                sum += a * f;
                                sc += a * a * f * f;
                            }
                    if (variant >= 1) sum += bias_v[k];
                    if (variant == 2 && sum < 0) sum *= slope_v[k];
                    double got = op[OUT_IDX(n, oh, ow, k)];
                    double d = fabs(got - sum);
                    double den = sqrt(sc);
                    if (den < 1e-30) den = 1.0;
                    if (d / den > max_rel) { max_rel = d / den; wn = n; wh = oh; ww = ow; wk = k; g_worst = got; e_worst = sum; }
                }
    {
        FILE* f = fopen(CHK_FILE, "w");
        if (f) { fprintf(f, "%.10e %.10e\n", chk, max_rel); fclose(f); }
    }
    if (max_rel > 1e-5) {
        // 把最差那一格的 got / exp / 差 打出来 —— 判断"是不是 bias 被数了 N 次"
        // 就靠这个比值（附录 BZ）
        printf("\n    [详细] align=%d %s 最差格 n=%d oh=%d ow=%d k=%d: got=%.8f exp=%.8f"
               " 差=%+.8f  exp-got=%+.8f\n",
               ALIGN, g_vname_of(s.kernel), wn, wh, ww, wk, g_worst, e_worst,
               g_worst - e_worst, e_worst - g_worst);
        printf("    [详细] 该 filter 的 bias = %+.8f   (got-exp)/bias = %.4f\n",
               bias_v[wk < (int)bias_v.size() ? wk : 0],
               (bias_v[wk < (int)bias_v.size() ? wk : 0] != 0.0)
                   ? (g_worst - e_worst) / bias_v[wk < (int)bias_v.size() ? wk : 0] : 0.0);
    }
    return max_rel > 1e-5 ? 1 : 0;        // 后向误差
#undef OUT_IDX
}

// 跑一个用例（fork），返回 '.' / 'X' / 'x' / '?'，并把签名写进 sig_chk/sig_rel
template<class TT, int ALIGN>
static char one_case(const Shape& s, int variant, bool use_buffer, double& sig_chk, double& sig_rel)
{
    fflush(stdout);
    remove(CHK_FILE);
    pid_t pid = fork();
    if (pid == 0) {
        FILE* dn = freopen("/dev/null", "w", stderr); (void)dn;   // 别让 ASan 报告截断网格
        int r = run_case<TT, ALIGN>(s, variant, use_buffer);
        _exit(r == 2 ? 3 : r);
    }
    int st = 0; waitpid(pid, &st, 0);
    sig_chk = 0; sig_rel = 0;
    FILE* f = fopen(CHK_FILE, "r");
    if (f) { if (fscanf(f, "%lf %lf", &sig_chk, &sig_rel) != 2) { sig_chk = sig_rel = 0; } fclose(f); }
    if (WIFSIGNALED(st)) { ncrash++; return 'X'; }
    if (WEXITSTATUS(st) == 1) return 'x';
    if (WEXITSTATUS(st) == 3) { ncrash++; return 'S'; }
    return '.';
}

template<class TT, int ALIGN>
static void grid(const char* tag)
{
    printf("\n========== %s（align=%d）==========\n", tag, ALIGN);
    printf("（每格两列 = 内部 malloc / 复用 buffer；三段 = plain / with_bias / with_bias_prelu）\n");
    printf("%-12s %-6s %-9s%-24s|%-24s|\n", "kernel", "K%A", "", "plain", "with_bias | with_bias_prelu");
    static const int KS_OK[] = { 8, 12, 16 };
    static const int KS_BAD[] = { 6, 10, 14 };
    for (int kk = 0; kk <= K_3x3_C3; kk++) {
        int C = (kk == K_2x2_C3 || kk == K_3x3_C3) ? 3 : 8;
        for (int which = 0; which < 2; which++) {
            int K = which ? KS_BAD[0] : KS_OK[0];
            const bool informational = (which != 0);
            Shape s; s.N = 1; s.H = 20; s.W = 20; s.C = C; s.K = K; s.stride = 1; s.kernel = kk;
            printf("%-12s %-6d %-9s", k_name[kk], K % ALIGN, informational ? "(只报告)" : "(门禁)");
            for (int v = 0; v < 3; v++) {
                for (int ub = 0; ub < 2; ub++) {
                    double c = 0, r = 0;
                    char ch = one_case<TT, ALIGN>(s, v, ub != 0, c, r);
                    ncase++;
                    const char* tag2 = (ch == '.') ? "ok" : (ch == 'x' ? "WRONG" : (ch == 'X' ? "CRASH" : "SETUP"));
                    if (ch != '.') {
                        if (informational) { ninfo++; tag2 = "info:WRONG"; }
                        else nbad++;
                    }
                    printf("%-12s", tag2);
                }
                printf("| ");
            }
            printf("\n");
        }
    }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC no_padding 卷积：各分支的 filter_N %% align 契约图（附录 BS / BT / BZ）\n");
    printf("每个用例 fork 一个子进程，崩溃只算该用例失败（附录 BO.5 / BN.5）\n");

    grid<ZQ::ZQ_CNN_Tensor4D_NCHWC4, 4>("NCHWC4");
    grid<ZQ::ZQ_CNN_Tensor4D_NCHWC8, 8>("NCHWC8");

    // ---- 附录 BW 的 padding 对照实验（只在 align=4 上做，align=8 同理） ----
    {
        Shape s; s.N = 1; s.H = 20; s.W = 20; s.C = 3; s.K = 8; s.stride = 1; s.kernel = K_2x2_C3;
        printf("\n--- 附录 BW：kernel2x2_C3, C=3, K=8，比较输入的第 4 个（补齐）通道 ---\n");
        double cz = 0, rz = 0, cp = 0, rp = 0;
        g_poison = 0.0f;
        char a = one_case<ZQ::ZQ_CNN_Tensor4D_NCHWC4, 4>(s, 1, false, cz, rz);
        g_poison = 7.0f;
        char b = one_case<ZQ::ZQ_CNN_Tensor4D_NCHWC4, 4>(s, 1, false, cp, rp);
        g_poison = 0.0f;
        printf("  补齐通道=0    %s  签名=%.8g  后向误差=%.3e\n", a == '.' ? "对" : "错", cz, rz);
        printf("  补齐通道=7.0  %s  签名=%.8g  后向误差=%.3e\n", b == '.' ? "对" : "错", cp, rp);
        double d = fabs(cp - cz), den = (fabs(cz) > 1e-12) ? fabs(cz) : 1.0;
        printf("  两个签名的相对差 = %.3e  ->  %s\n", d / den,
               (d / den > 1e-9) ? "**输入的补齐通道参与了计算**"
                                : "补齐通道**没**参与计算 -> BV.4 那条耦合被排除");
    }

    printf("\n共 %d 个用例：门禁里仍错 %d，只报告（故意违约的形状）%d，崩溃/搭建失败 %d\n",
           ncase, nbad, ninfo, ncrash);
    return nbad ? 1 : 0;
}
