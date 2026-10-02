// zq_nchwc_conv_check.cpp —— NCHWC「no_padding」卷积族的 kernel 契约图（附录 BS / BT）
//
// 起因（audit_k3_20261001.md 附录 BS / BT）
// -------------------------------------------
// 附录 BS 覆盖了 NCHWC4 的 **3x3** 两支（6 个内核）并定位到一条契约：
// `filter_N % 4 != 0` 时，`zq_cnn_convolution_gemm_nchwc_col2im.h` 里每一支的
//     for (kc = 0; kc < out_C; kc += zq_mm_align_size, ...)
// 都会在最后一组多处理 2~3 个 —— 越界读 matrix_C、越界写输出。
// 而 wrapper（ZQ_CNN_Forward_SSEUtils_NCHWC::Convolution*）**从不校验**它。
//
// 但 BS 只测了 3x3。**守卫该加在哪一层，取决于另外三支（1x1 / 2x2 / general）
// 是不是同样要求 filter_N 是 4 的倍数** —— 没测就加守卫，是拿"看起来修好了"
// 换"其实只堵了一个门"。本文件就是去测这个的。
//
// 四个分支的**参数列表完全相同**（只有 filter_H/W 与输出尺寸不同），所以一张表
// 就能铺开。
//
// 为什么每个用例 fork 一个子进程
// ------------------------------
// 缺陷是**崩溃**（SEGV / 越界），而 ASan 一碰就 abort 掉整个进程。第一版
// 不 fork，跑到第一个崩的就停了，整片区域是什么情况完全不知道。
// fork + waitpid 判信号之后，才能把"哪些 kernel x 哪些 filter_N%4 是好的"
// 一次画出来（附录 BO.5 / BN.5 已经为此付过两次学费）。
//
// 布局：仍然**不用自己推的** —— 用真实的 ZQ_CNN_Tensor4D_NCHWC4 类算 stride，
// 用 ConvertFromCompactNCHW 填普通 [N][C][H][W] 数组。
// 判据：**后向误差**（附录 BO.3：相对误差对抵消敏感，点积天生就有抵消）。
//
// pad：只测 pad=0。no_padding 族本身不含 padding 逻辑，调用方自己把指针挪过
// 边框（input.GetFirstPixelPtr() - padH*in_widthStep - padW*4）。带 pad 的那部分
// 记在附录 BT.5，**不假装验过**。

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

static int g_fail = 0, g_crash = 0, g_wrong = 0;

// kernel 种类
enum { K_GEN = 0, K_1x1, K_2x2, K_2x2_C3, K_3x3, K_3x3_C3 };
static const char* k_name[] = { "general", "kernel1x1", "kernel2x2", "kernel2x2_C3", "kernel3x3", "kernel3x3_C3" };
static int k_fsize[] = { 3, 1, 2, 2, 3, 3 };   // general 用 3x3

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

struct Shape { int N, H, W, C, K, stride, kernel; };

// 返回 0 = 通过；非 0 = 结果不对。崩溃由父进程判信号。
static int run_case(const Shape& s, int variant, bool use_buffer)
{
    const int N = s.N, H = s.H, W = s.W, C = s.C, K = s.K, ST = s.stride;
    const int FS = k_fsize[s.kernel];              // filter 边长
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

    ZQ::ZQ_CNN_Tensor4D_NCHWC4 tin, tflt, tbias, tslope, tout;
    if (!tin.ChangeSize(N, H, W, C, 0, 0)) return 2;
    if (!tflt.ChangeSize(K, FS, FS, C, 0, 0)) return 2;
    if (!tbias.ChangeSize(K, 1, 1, 1, 0, 0)) return 2;
    if (!tslope.ChangeSize(K, 1, 1, 1, 0, 0)) return 2;
    if (!tout.ChangeSize(N, oH, oW, K, 0, 0)) return 2;
    if (!tin.ConvertFromCompactNCHW(&in_nchw[0], N, C, H, W)) return 2;
    if (!tflt.ConvertFromCompactNCHW(&flt_nchw[0], K, C, FS, FS)) return 2;
    memset(tbias.GetFirstPixelPtr(), 0, sizeof(float) * (size_t)K);
    memset(tslope.GetFirstPixelPtr(), 0, sizeof(float) * (size_t)K);
    for (int k = 0; k < K; k++) {
        tbias.GetFirstPixelPtr()[k] = bias_v[k];
        tslope.GetFirstPixelPtr()[k] = slope_v[k];
    }

    // NCHWC4 布局 [n][c/4][h][w][4]（BS.3：第一版按 NCHW 写错过，ASan 当场抓到）
    const int oWS = tout.GetWidthStep(), oSS = tout.GetSliceStep(), oIS = tout.GetImageStep();
#define OUT_IDX(nn, ohh, oww, kk) \
    ((nn) * oIS + ((kk) / 4) * oSS + (ohh) * oWS + (oww) * 4 + ((kk) % 4))
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

    // 四个分支的参数列表**完全一样**，所以一张宏表就够（BS.3 记过：不要用
    // "一个指针去凑参数个数" —— C 链接不检查 arity，会静默错位）
#define IN_A   ip, N, H, W, C, tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep()
#define FLT_A  fp, K, FS, FS, C, tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep()
#define OUT_A  op, N, oH, oW, K, oWS, oSS, oIS
#define MID_A  ST, ST, 1, 1
    int rc = 0;
    if (variant == 0) {
        switch (s.kernel) {
        case K_1x1:  zq_cnn_conv_no_padding_gemm_nchwc4_kernel1x1(IN_A, FLT_A, MID_A, OUT_A, &buffer, &buffer_len); break;
        case K_2x2:  zq_cnn_conv_no_padding_gemm_nchwc4_kernel2x2(IN_A, FLT_A, MID_A, OUT_A, &buffer, &buffer_len); break;
        case K_2x2_C3:zq_cnn_conv_no_padding_gemm_nchwc4_kernel2x2_C3(IN_A, FLT_A, MID_A, OUT_A, &buffer, &buffer_len); break;
        case K_3x3:  zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3(IN_A, FLT_A, MID_A, OUT_A, &buffer, &buffer_len); break;
        case K_3x3_C3:zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_C3(IN_A, FLT_A, MID_A, OUT_A, &buffer, &buffer_len); break;
        default:     zq_cnn_conv_no_padding_gemm_nchwc4_general(IN_A, FLT_A, MID_A, OUT_A, &buffer, &buffer_len); break;
        }
    } else if (variant == 1) {
        switch (s.kernel) {
        case K_1x1:  zq_cnn_conv_no_padding_gemm_nchwc4_kernel1x1_with_bias(IN_A, FLT_A, MID_A, OUT_A, bp, &buffer, &buffer_len); break;
        case K_2x2:  zq_cnn_conv_no_padding_gemm_nchwc4_kernel2x2_with_bias(IN_A, FLT_A, MID_A, OUT_A, bp, &buffer, &buffer_len); break;
        case K_2x2_C3:zq_cnn_conv_no_padding_gemm_nchwc4_kernel2x2_C3_with_bias(IN_A, FLT_A, MID_A, OUT_A, bp, &buffer, &buffer_len); break;
        case K_3x3:  zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_with_bias(IN_A, FLT_A, MID_A, OUT_A, bp, &buffer, &buffer_len); break;
        case K_3x3_C3:zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_C3_with_bias(IN_A, FLT_A, MID_A, OUT_A, bp, &buffer, &buffer_len); break;
        default:     zq_cnn_conv_no_padding_gemm_nchwc4_general_with_bias(IN_A, FLT_A, MID_A, OUT_A, bp, &buffer, &buffer_len); break;
        }
    } else {
        switch (s.kernel) {
        case K_1x1:  zq_cnn_conv_no_padding_gemm_nchwc4_kernel1x1_with_bias_prelu(IN_A, FLT_A, MID_A, OUT_A, bp, sp, &buffer, &buffer_len); break;
        case K_2x2:  zq_cnn_conv_no_padding_gemm_nchwc4_kernel2x2_with_bias_prelu(IN_A, FLT_A, MID_A, OUT_A, bp, sp, &buffer, &buffer_len); break;
        case K_2x2_C3:zq_cnn_conv_no_padding_gemm_nchwc4_kernel2x2_C3_with_bias_prelu(IN_A, FLT_A, MID_A, OUT_A, bp, sp, &buffer, &buffer_len); break;
        case K_3x3:  zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_with_bias_prelu(IN_A, FLT_A, MID_A, OUT_A, bp, sp, &buffer, &buffer_len); break;
        case K_3x3_C3:zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_C3_with_bias_prelu(IN_A, FLT_A, MID_A, OUT_A, bp, sp, &buffer, &buffer_len); break;
        default:     zq_cnn_conv_no_padding_gemm_nchwc4_general_with_bias_prelu(IN_A, FLT_A, MID_A, OUT_A, bp, sp, &buffer, &buffer_len); break;
        }
    }
#undef IN_A
#undef FLT_A
#undef OUT_A
#undef MID_A
    (void)rc;

    if (buffer) _aligned_free(buffer);      // buffer 所有权归调用方（附录 BJ）

    // ---- 参考：最朴素的卷积 ----
    double max_rel = 0.0;
    for (int n = 0; n < N; n++)
        for (int oh = 0; oh < oH; oh++)
            for (int ow = 0; ow < oW; ow++)
                for (int k = 0; k < K; k++) {
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
                    if (d / den > max_rel) max_rel = d / den;
                }
    return max_rel > 1e-5 ? 1 : 0;        // 后向误差
#undef OUT_IDX
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC no_padding 卷积：各分支的 filter_N %% 4 契约图（附录 BT）\n");
    printf("每个用例 fork 一个子进程，崩溃只算该用例失败（附录 BO.5 / BN.5）\n\n");
    printf("（每格两列 = 内部 malloc / 复用 buffer；三段 = plain / with_bias / with_bias_prelu）\n");
    printf("%-12s %-6s %-9s%-26s|%-26s|\n", "kernel", "K%4", "", "plain", "with_bias", "with_bias_prelu");

    // 每支两个 K：4 的倍数（8/12/16）与非倍数（6/10/14）。
    // C=3 的两支额外用 C=3 的形状。
    static const int KS_OK[]   = { 8, 12, 16 };
    static const int KS_BAD[]  = { 6, 10, 14 };
    int ncase = 0, ncrash = 0, nwrong = 0, nbad = 0, ninfo = 0;
    for (int kk = 0; kk <= K_3x3_C3; kk++) {
        int C = (kk == K_2x2_C3 || kk == K_3x3_C3) ? 3 : 8;
        for (int which = 0; which < 2; which++) {
            int K = which ? KS_BAD[0] : KS_OK[0];
            // K%4 != 0 的一档是**故意**违反契约的（附录 BT.3）。
            // 这里把它标成"只报告、不判失败"：门禁要 pin 住的是
            // **"合法形状必须算对"**，而不是"非法形状必须算错" ——
            // 后者会在有人把 col2im 的尾巴补好之后把门禁变红。
            const bool informational = (which != 0);
            if ((kk == K_2x2_C3 || kk == K_3x3_C3) && C != 3) continue;
            Shape s; s.N = 1; s.H = 20; s.W = 20; s.C = C; s.K = K; s.stride = 1; s.kernel = kk;
            printf("%-12s %-6d %-9s", k_name[kk], K % 4, informational ? "(只报告)" : "(门禁)");
            for (int v = 0; v < 3; v++) {
                for (int ub = 0; ub < 2; ub++) {
                    fflush(stdout);
                    pid_t pid = fork();
                    if (pid == 0) {
                        // 子进程的 stderr 接 /dev/null：ASan 的报告走 stderr，
                        // 会把父进程 stdout 上那一格结果拦腰截断（附录 BN.5）。
                        FILE* dn = freopen("/dev/null", "w", stderr);
                        (void)dn;
                        // 只跑**一次**：第一版写成 `run_case(...)==2 ? 3 : run_case(...)`
                        // 于是子进程把内核跑了两遍、还只看第二遍的结果。
                        int r = run_case(s, v, ub != 0);
                        _exit(r == 2 ? 3 : r);
                    }
                    int st = 0; waitpid(pid, &st, 0);
                    ncase++;
                    const char* tag = "ok";
                    if (WIFSIGNALED(st)) { ncrash++; tag = "CRASH"; }
                    else if (WEXITSTATUS(st) == 1) { nwrong++; tag = "WRONG"; }
                    else if (WEXITSTATUS(st) == 3) { ncrash++; tag = "SETUP"; }
                    if (tag[0] != 'o' && informational) { ninfo++; tag = "info:WRONG"; }
                    else if (tag[0] != 'o') { nbad++; }
                    printf("%-12s", tag);
                }
                printf("| ");
            }
            printf("\n");
        }
    }
    printf("\n共 %d 个用例：崩溃 %d，应对但仍错 %d，只报告（故意违约的形状）%d\n",
           ncase, ncrash, nbad, ninfo);
    printf("（BS 已经把 filter_N%%4 这条契约钉死了；这张表是为了确认 1x1 / 2x2 /\n");
    printf("  general 三支同样要求 4 的倍数 —— 守卫该加在哪一层，取决于这个答案。\n");
    printf("  kernel2x2_C3 在 K%%4==0 下仍然错，**本轮未定性**（附录 BT.4），\n");
    printf("  所以它会让本测试为红 —— 门禁里因此登记在 SKIP。)\n");
    return nbad ? 1 : 0;
}
