/* NCHWC「batch 维不变性」门禁 —— 附录 IV（2026-10-06）
 *
 * 判据不是"和手写参考比"，而是一条**不变式**：
 *
 *     把 N=2 的张量整批送进算子，得到的第 n 张图
 *     必须与「只把第 n 张图（N=1）单独送进同一个算子」逐位相同。
 *
 * 为什么值得单独做一道门禁
 * -------------------------
 * 附录 IU.1 / IU.2 两条缺陷都是「batch 维推进指针时用了 sliceStep 而不是
 * imStep」。它们的共同特点是**自己跟自己对不上**却看不出来：
 *   - N == 1 时 image 循环只跑一圈，推进量乘 0；
 *   - C 是 align 的整数倍时 dst_imStep == dst_sliceStep（ChangeSize 里
 *     dst_imStep = ceil(dst_C/align) * dst_sliceStep，ceil 恒为 1）。
 * 两层遮蔽同时成立时，不越界、不崩、ASan/UBSan 全静默。
 *
 * 而这条不变式**不需要任何参考实现**：只要 N=2 且 C 不是 align 的整数倍，
 * 任何"batch 维步进写错"的错误都会让两侧不等。**它测的是"图与图之间
 * 有没有互相串"，而这正是这类缺陷的全部内容。**
 *
 * 每组用例跑两个 C：
 *   C = A     （align 的整数倍 —— 遮蔽成立，用来确认"放过时确实放过"）
 *   C = A + 2 （不是整数倍 —— 遮蔽不成立，这一档才有鉴别力）
 * 两档都跑，是因为"只在有鉴别力的那档红"是这类缺陷的正常形态，
 * 只跑有鉴别力的那档会让人以为"只有一种配置有问题"。
 *
 * 沿用 CI/CJ/CL 那一套：每个 op 一个子进程，显式判"没读到结果文件"= 失败。
 */
#include "zq_check_child.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <unistd.h>
#include <sys/wait.h>
#include "ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h"
#include "ZQCNN/ZQ_CNN_Forward_SSEUtils_NCHWC.h"

#define RES_FILE "/tmp/zq_batch_res.txt"

// 状态码。**「op 返回 false / 形状摆不下」必须与「崩了」分开报** ——
// 混在一起会把探针自己的形状错误说成库的缺陷（本文件「没有证据就不要断言原因」）。
#define ST_RAN     0
#define ST_REJECT  2   // 契约不符 / 形状摆不下，这一组不判失败
static void write_res(int st, long n_ok, long n_bad, double worst)
{
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%d %ld %ld %.6e\n", st, n_ok, n_bad, worst); fclose(f); }
}

typedef float (*VALFN)(int seed, int idx);
static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

// op 名单。用 enum 而不是函数指针，因为每个 op 的参数表都不一样，
// 而"名字写全 + 参数由用例给"更不容易抄错。
enum Op {
    OP_RELU, OP_PRELU, OP_ADDBIAS_PRELU, OP_BN_B_A, OP_SOFTMAX,
    OP_MAXPOOL, OP_AVGPOOL, OP_ELTWISE_SUM, OP_DEPTHWISE, OP_CONV,
    OP_N
};
static const char* g_op_name[OP_N] = {
    "ReLU", "PReLU", "AddBiasPReLU", "BatchNorm_b_a", "Softmax",
    "MaxPooling", "AVGPooling", "Eltwise_Sum", "DepthwiseConvolution", "Convolution"
};

struct Case { int op, N, H, W, C, kH, kW, S, axis, pad, filter_N; };

template <class TEN>
static void run_one(const Case& c)
{
    const int A = TEN().GetAlignSize();
    const int N = c.N, H = c.H, W = c.W, C = c.C;

    std::vector<float> in((size_t)N * C * H * W);
    for (size_t i = 0; i < in.size(); i++) in[i] = val(1, (int)i);

    // ---------------- 整批（N=2） ----------------
    TEN ti, to;
    if (!ti.ChangeSize(N, H, W, C, 0, 0)) return;
    if (!ti.ConvertFromCompactNCHW(&in[0], N, C, H, W)) return;

    int oH = H, oW = W, oC = C;
    if (c.op == OP_MAXPOOL || c.op == OP_AVGPOOL) {
        oH = (H - c.kH + c.S - 1) / c.S + 1;
        oW = (W - c.kW + c.S - 1) / c.S + 1;
    } else if (c.op == OP_DEPTHWISE) {
        oH = (H + 2 * c.pad - c.kH) / c.S + 1;
        oW = (W + 2 * c.pad - c.kW) / c.S + 1;
        oC = C;                       // depthwise：输出通道 = 输入通道
    } else if (c.op == OP_CONV) {
        oH = (H + 2 * c.pad - c.kH) / c.S + 1;
        oW = (W + 2 * c.pad - c.kW) / c.S + 1;
        // **输出通道是 filter_N，不是 in_C**。我第一版把输出张量按 in_C 开，
        // 而 `ConvertToCompactNCHW` 写出去的是 filter_N 个通道 ——
        // 于是比对时越过了实际写入的末尾、拿未初始化内存当结果，
        // 报出来的是「一张图全错」，而那是我探针的缓冲区开小了。
        // （本文件「没有证据就不要断言原因」。）
        oC = A;
    }
    if (oH <= 0 || oW <= 0) return;
    if (!to.ChangeSize(N, oH, oW, oC, 0, 0)) return;
    // 输出先填成哨兵：万一某个 op 一格都没写，比对会以"和单图一致"的形式通过，
    // 而不是因为算对了才通过。这里用 -12345.0，与输入值域 (±1) 差得很远。
    {
        std::vector<float> fill((size_t)N * to.GetImageStep(), -12345.0f);
        memcpy(to.GetFirstPixelPtr(), &fill[0], fill.size() * sizeof(float));
    }

    // 逐通道张量（bias / slope / filters）
    TEN tb, ts, tf;
    const float* pb = 0; const float* ps = 0; const float* pf = 0;
    if (c.op == OP_PRELU || c.op == OP_ADDBIAS_PRELU) {
        std::vector<float> sv((size_t)C);
        for (int i = 0; i < C; i++) sv[i] = 0.25f * (float)(i % 5) - 0.5f;
        if (!ts.ChangeSize(1, 1, 1, C, 0, 0)) return;
        if (!ts.ConvertFromCompactNCHW(&sv[0], 1, C, 1, 1)) return;
        ps = ts.GetFirstPixelPtr();
    }
    if (c.op == OP_ADDBIAS_PRELU || c.op == OP_BN_B_A) {
        std::vector<float> bv((size_t)C);
        for (int i = 0; i < C; i++) bv[i] = 0.5f * (float)(i % 7) - 1.0f;
        if (!tb.ChangeSize(1, 1, 1, C, 0, 0)) return;
        if (!tb.ConvertFromCompactNCHW(&bv[0], 1, C, 1, 1)) return;
        pb = tb.GetFirstPixelPtr();
    }
    if (c.op == OP_DEPTHWISE || c.op == OP_CONV) {
        // 契约（ZQ_CNN_Forward_SSEUtils_NCHWC.cpp:1130）：
        //     DepthwiseConvolution 要求 filter_C == in_C 且 filter_N == 1；
        //     Convolution 要求 filters 是 [N=filter_N][H=kH][W=kW][C=in_C]。
        // 我第一版把 filters 摆成 [N=C][H=kH][W=kW][C=1]，于是 op 直接 return false，
        // 而探针把"没跑完"报成 **崩溃** —— 那是**探针的形状错了**，
        // 不是被测代码的缺陷（本文件「没有证据就不要断言原因」）。
        const int fN = (c.op == OP_DEPTHWISE) ? 1 : A;
        std::vector<float> fv((size_t)fN * c.kH * c.kW * C);
        for (size_t i = 0; i < fv.size(); i++) fv[i] = val(2, (int)i);
        if (!tf.ChangeSize(fN, c.kH, c.kW, C, 0, 0)) return;
        if (!tf.ConvertFromCompactNCHW(&fv[0], fN, C, c.kH, c.kW)) return;
        pf = tf.GetFirstPixelPtr();
    }

    std::vector<const TEN*> ein;
    TEN ein1;
    if (c.op == OP_ELTWISE_SUM) {
        std::vector<float> in2((size_t)N * C * H * W);
        for (size_t i = 0; i < in2.size(); i++) in2[i] = val(3, (int)i);
        if (!ein1.ChangeSize(N, H, W, C, 0, 0)) return;
        if (!ein1.ConvertFromCompactNCHW(&in2[0], N, C, H, W)) return;
        ein.push_back(&ti); ein.push_back(&ein1);
    }

    bool ok = true;
    switch (c.op) {
    case OP_RELU:   ok = ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::ReLU(ti, 0.125f); break;
    case OP_PRELU:  ok = ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::PReLU(ti, ts); break;
    case OP_ADDBIAS_PRELU: ok = ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::AddBiasPReLU(ti, tb, ts); break;
    case OP_BN_B_A:  ok = ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::BatchNorm_b_a(ti, tb, tb); break;
    case OP_SOFTMAX: ok = ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::Softmax(ti, c.axis); break;
    case OP_MAXPOOL: ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::MaxPooling(ti, to, c.kH, c.kW, c.S, c.S, false); ok = true; break;
    case OP_AVGPOOL: ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::AVGPooling(ti, to, c.kH, c.kW, c.S, c.S, false); ok = true; break;
    case OP_ELTWISE_SUM: ok = ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::Eltwise_Sum(ein, to); break;
    case OP_DEPTHWISE: ok = ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::DepthwiseConvolution(ti, tf, c.S, c.S, 1, 1, c.pad, c.pad, to); break;
    case OP_CONV:    ok = ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::Convolution(ti, tf, c.S, c.S, 1, 1, c.pad, c.pad, to); break;
    default: break;
    }
    if (!ok) { write_res(ST_REJECT, 0, 0, 0.0); return; }

    // ---------------- 单图（第 n 张，N=1） ----------------
    std::vector<float> batch_out((size_t)N * oC * oH * oW);
    to.ConvertToCompactNCHW(&batch_out[0]);

    long n_ok = 0, n_bad = 0; double worst = 0.0;
    for (int n = 0; n < N; n++) {
        TEN si, so;
        if (!si.ChangeSize(1, H, W, C, 0, 0)) continue;
        if (!si.ConvertFromCompactNCHW(&in[(size_t)n * C * H * W], 1, C, H, W)) continue;
        if (!so.ChangeSize(1, oH, oW, oC, 0, 0)) continue;
        std::vector<float> fill((size_t)so.GetImageStep(), -12345.0f);
        memcpy(so.GetFirstPixelPtr(), &fill[0], fill.size() * sizeof(float));

        bool ok1 = true;
        std::vector<const TEN*> sin_;
        TEN sin1;
        if (c.op == OP_ELTWISE_SUM) {
            std::vector<float> in2((size_t)C * H * W);
            for (size_t i = 0; i < in2.size(); i++) in2[i] = val(3, (int)(i + (size_t)n * C * H * W));
            if (!sin1.ChangeSize(1, H, W, C, 0, 0)) continue;
            if (!sin1.ConvertFromCompactNCHW(&in2[0], 1, C, H, W)) continue;
            sin_.push_back(&si); sin_.push_back(&sin1);
        }
        switch (c.op) {
        case OP_RELU:   ok1 = ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::ReLU(si, 0.125f); break;
        case OP_PRELU:  ok1 = ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::PReLU(si, ts); break;
        case OP_ADDBIAS_PRELU: ok1 = ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::AddBiasPReLU(si, tb, ts); break;
        case OP_BN_B_A:  ok1 = ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::BatchNorm_b_a(si, tb, tb); break;
        case OP_SOFTMAX: ok1 = ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::Softmax(si, c.axis); break;
        case OP_MAXPOOL: ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::MaxPooling(si, so, c.kH, c.kW, c.S, c.S, false); ok1 = true; break;
        case OP_AVGPOOL: ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::AVGPooling(si, so, c.kH, c.kW, c.S, c.S, false); ok1 = true; break;
        case OP_ELTWISE_SUM: ok1 = ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::Eltwise_Sum(sin_, so); break;
        case OP_DEPTHWISE: ok1 = ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::DepthwiseConvolution(si, tf, c.S, c.S, 1, 1, c.pad, c.pad, so); break;
        case OP_CONV:    ok1 = ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::Convolution(si, tf, c.S, c.S, 1, 1, c.pad, c.pad, so); break;
        default: break;
        }
        if (!ok1) { write_res(ST_REJECT, 0, 0, 0.0); return; }

        std::vector<float> solo_out((size_t)oC * oH * oW);
        so.ConvertToCompactNCHW(&solo_out[0]);
        const float* got = &batch_out[(size_t)n * oC * oH * oW];
        for (size_t i = 0; i < solo_out.size(); i++) {
            double d = fabs(got[i] - solo_out[i]);
            if (d > worst) worst = d;
            if (d > 0) n_bad++; else n_ok++;
        }
    }
    write_res(ST_RAN, n_ok, n_bad, worst);
}

typedef void (*RUNNER)(const Case&);

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0, g_skip = 0;

static void one(const Case& c, RUNNER r, int align)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) { zq_child_silence_stderr(); r(c); _exit(0); }
    int st = 0; waitpid(pid, &st, 0);
    int status = -1; long ok = 0, bad = 0; double worst = 0; int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%d %ld %ld %lf", &status, &ok, &bad, &worst) == 4); fclose(f); }
    char tag[160];
    snprintf(tag, sizeof(tag), "nchwc%d C=%-3d %dx%d op=%s axis=%d",
             align, c.C, c.H, c.W, g_op_name[c.op], c.axis);
    if (!have) { g_crash++; printf("  %-56s  没跑完（退出码 %d）\n", tag, WEXITSTATUS(st)); return; }
    if (WIFSIGNALED(st)) { g_crash++; printf("  %-56s  CRASH(信号 %d)\n", tag, WTERMSIG(st)); return; }
    if (status == ST_REJECT) { g_skip++; return; }
    if (bad > 0) { g_bad++; printf("  %-56s  FAIL %ld/%ld 格不同, 最大差 %.6e\n", tag, bad, ok + bad, worst); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC batch 维不变性：out[n](N=2) 必须逐位等于 out[n](N=1)\n");
    printf("每组两个 C：A（整数倍，遮蔽成立）与 A+2（整数倍不成立，有鉴别力）\n\n");

    RUNNER r[3] = { &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC1>,
                    &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC4>,
                    &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC8> };

    // 每个 op 的形状。H/W 取成 6x7：两维都**不是** kernel 的整数倍，
    // 且都不与 stride 整除，于是 pooling / conv 的边界分支真的会被走到。
    for (int o = 0; o < OP_N; o++) {
        const int o0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash, p0 = g_skip;
        for (int ai = 0; ai < 3; ai++) {
            const int A = (ai == 0) ? 1 : (ai == 1 ? 4 : 8);
            for (int m = 0; m < 2; m++) {
                for (int rep = 0; rep < 2; rep++) {
                    Case c; memset(&c, 0, sizeof(c));
                    c.op = o; c.N = 2; c.H = 6; c.W = 7; c.C = (m ? A + 2 : A);
                    c.kH = 3; c.kW = 3; c.S = 2; c.pad = 1; c.axis = rep; c.filter_N = 4;
                    if (c.op == OP_MAXPOOL || c.op == OP_AVGPOOL) { c.pad = 0; c.axis = 0; }
                    one(c, r[ai], A);
                }
            }
        }
        printf("  %-22s  %2d 个用例：对 %d，错 %d，崩 %d，契约不符跳过 %d\n",
               g_op_name[o], g_case - o0, g_ok - k0, g_bad - b0, g_crash - x0, g_skip - p0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d，契约不符跳过 %d\n",
           g_case, g_ok, g_bad, g_crash, g_skip);
    return (g_bad || g_crash) ? 1 : 0;
}