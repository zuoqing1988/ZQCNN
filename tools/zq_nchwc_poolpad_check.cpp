/* NCHWC 池化的**带 padding** 行为门禁 —— 附录 F2
 *
 * 判据：NCHWC 的池化在 `pad != 0` 时，必须等于「把输入零填充到
 * (H + pad_top + pad_bottom) x (W + pad_left + pad_right)，再按 kernel/stride
 * 做池化」这个**明文定义**。
 *
 * 背景
 * ----
 * 修之前 `ZQ_CNN_Layer_NCHWC_Pooling::ReadParam` **解析了** `pad`
 * （写进 pad_H/pad_W），而 `Forward` 调的 `MaxPooling(...)` **签名里根本没有 pad**
 * —— 那两个成员解析完就没人用了。模型文件写 `pad 1` 会**加载成功、
 * 然后静默按无 pad 计算**，输出**形状就错一格**。
 *
 * 为什么判据不写成「NCHWC == NCHW」
 * ------------------------------
 * 因为 **NCHW 那一支自己有缺陷**。实测（同一份数据，4x4，值=行*4+列+1，k=2 s=2 pad=1）：
 *
 *     输入 c0
 *        1   2   3   4
 *        5   6   7   8
 *        9  10  11  12
 *       13  14  15  16
 *     三种实现给出的第 2 行：
 *       NCHW           7   8   9      <- 窗口起点落在**数据首行**
 *       NCHWC（修后）  9  11  12      <- 窗口起点落在 -pad 行（= 明文定义）
 *       手算参考       9  11  12
 *
 * NCHW 的代码里确实写了 `GetFirstPixelPtr() - pad_H_top*in_widthStep - pad_W_left*in_pixStep`，
 * 但实测窗口起点并没有真的退到 -pad 行 —— **代码表达的意图与实际行为不一致**。
 * 拿 NCHW 当参考等于把一个缺陷固化成"标准"。
 * 这一族缺陷靠"和孪生实现比"抓不到：AGENTS.md 那条「同仓的两份实现互为对照」
 * 的**适用条件是两边都对**。
 *
 * 两段覆盖（缺一不可）
 * ------------------
 * ① **前向**：直接调 `MaxPooling/AVGPooling`，验内核与 padding 的正确性。
 * ② **层**：走 `ReadParam -> SetBottomDim -> LayerSetup -> Forward`，
 *    验**参数确实被解析并往下传**。
 *
 * ② 不是可有可无的：第一版只有 ①，变异测试把 `Forward` 传下去的 pad
 * 改回全 0（也就是原始缺陷），**门禁依然全绿** ——
 * 因为缺陷恰恰在层里，① 压根没经过层。
 * 与 AGENTS.md「阳性对照要换一个变异位置再问一次」同源：
 * 同一个变异落在**判据覆盖维度之外**时，门禁看不见。
 *
 * 覆盖：NCHWC1/4/8 × {MAX, AVG} × {C = A, C = A+2} × 5 组 pad × 2 段
 *     = 120 个用例；pad 含 0（对照）、对称、非对称两种方向、不对称且不等。
 *
 * 独立参考本身也栽过两次，都是**参考错、库对**（详见 ref_pool 里那两条注释）：
 * AVG 除数写错（恒定 kH*kW vs 落在补齐区里的格数）、有效范围写错
 * （原图 [0,H) vs 补齐区 [-pT, H+pB)）。两次的症状都是"库算错了"。
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
#include "ZQCNN/ZQ_CNN_Layer_NCHWC.h"

#define RES_FILE "/tmp/zq_poolpad_res.txt"

enum { SEG_UTIL = 0, SEG_LAYER = 1 };
static const char* g_seg[2] = { "前向", "层  " };

struct Case {
    int align, N, H, W, C, kH, kW, sH, sW, pT, pB, pL, pR, maxpool, padtype, seg;
};

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    // 全为正且互不相同：MAX 的「补 0」与「补 -inf」在正数上等价，
    // 而全负输入才能暴露补 0 的问题 —— AVG 那一档承担这个区分。
    return (float)((int)(x % 2001) + 1) * 0.001f;
}

// 明文定义的独立参考：不经过任何被测代码。
static void ref_pool(const std::vector<float>& in, const Case& c,
                     const std::vector<float>& got, long& n_ok, long& n_bad, double& worst)
{
    const int N = c.N, H = c.H, W = c.W, C = c.C;
    const int need_H = (int)ceil((float)(H + c.pT + c.pB - c.kH) / c.sH + 1);
    const int need_W = (int)ceil((float)(W + c.pL + c.pR - c.kW) / c.sW + 1);
    if (need_H <= 0 || need_W <= 0) return;
    for (int n = 0; n < N; n++)
        for (int ch = 0; ch < C; ch++)
            for (int oh = 0; oh < need_H; oh++)
                for (int ow = 0; ow < need_W; ow++) {
                    double sum = 0.0; int cnt = 0; bool first = true; double mx = 0.0;
                    for (int kh = 0; kh < c.kH; kh++) {
                        // 有效范围是**补齐区** [-pT, H+pB)，不是原图 [0, H)。
                        // 第一版写成 [0, H)，对称 padding 的 oh=0 就少算一格、18 例红 ——
                        // 那是**参考自己**的错，症状与「库算错了」一模一样。
                        const int ih = oh * c.sH - c.pT + kh;
                        if (ih < -c.pT || ih >= H + c.pB) continue;
                        for (int kw = 0; kw < c.kW; kw++) {
                            const int iw = ow * c.sW - c.pL + kw;
                            if (iw < -c.pL || iw >= W + c.pR) continue;
                            cnt++;
                            const double v = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                ? in[(((size_t)n * C + ch) * H + ih) * W + iw] : 0.0;
                            sum += v;
                            if (first || v > mx) { mx = v; first = false; }
                        }
                    }
                    if (cnt == 0) continue;
                    // AVG 除以**落在补齐区里的格数**，不是恒定的 kH*kW ——
                    // 最后一个窗口在补齐区右边只够 2 格时除以 2。
                    // 第一版恒除 kH*kW，6 个用例红（同样是我的错）。
                    const double want = c.maxpool ? mx : sum / (double)cnt;
                    const double gv = got[(((size_t)n * C + ch) * need_H + oh) * need_W + ow];
                    const double d = fabs(gv - want);
                    if (d > worst) worst = d;
                    if (d > 1e-4 * (fabs(want) + 1.0)) n_bad++; else n_ok++;
                }
}

template <class TEN>
static void run_one(const Case& c)
{
    const int N = c.N, H = c.H, W = c.W, C = c.C;
    std::vector<float> in((size_t)N * C * H * W);
    for (size_t i = 0; i < in.size(); i++) in[i] = val(1, (int)i);

    std::vector<float> got;
    if (c.seg == SEG_UTIL) {
        const int need_H = (int)ceil((float)(H + c.pT + c.pB - c.kH) / c.sH + 1);
        const int need_W = (int)ceil((float)(W + c.pL + c.pR - c.kW) / c.sW + 1);
        if (need_H <= 0 || need_W <= 0) return;
        TEN ti, to;
        if (!ti.ChangeSize(N, H, W, C, 0, 0)) return;
        if (!ti.ConvertFromCompactNCHW(&in[0], N, C, H, W)) return;
        if (!to.ChangeSize(N, need_H, need_W, C, 0, 0)) return;
        if (c.maxpool)
            ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::MaxPooling(ti, to, c.kH, c.kW, c.sH, c.sW,
                                                           c.pT, c.pB, c.pL, c.pR, false);
        else
            ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::AVGPooling(ti, to, c.kH, c.kW, c.sH, c.sW,
                                                          c.pT, c.pB, c.pL, c.pR, false);
        got.assign((size_t)N * C * need_H * need_W, -12345.0f);
        to.ConvertToCompactNCHW(&got[0]);
    } else {
        // ---- 驱动层本身：ReadParam -> SetBottomDim -> LayerSetup -> Forward ----
        ZQ::ZQ_CNN_Layer_NCHWC_Pooling<TEN>* L = new ZQ::ZQ_CNN_Layer_NCHWC_Pooling<TEN>();
        char line[256];
        snprintf(line, sizeof(line),
                 "Pooling name=p1 bottom=data top=p1 pool=%s kernel_H=%d kernel_W=%d "
                 "stride_H=%d stride_W=%d pad_H_top=%d pad_H_bottom=%d pad_W_left=%d pad_W_right=%d",
                 c.maxpool ? "MAX" : "AVG", c.kH, c.kW, c.sH, c.sW, c.pT, c.pB, c.pL, c.pR);
        if (!L->ReadParam(std::string(line))) { delete L; return; }
        if (!L->SetBottomDim(C, H, W)) { delete L; return; }
        int top_C = 0, top_H = 0, top_W = 0;
        L->GetTopDim(top_C, top_H, top_W);
        if (top_H <= 0 || top_W <= 0) { delete L; return; }

        TEN bi, to;
        if (!bi.ChangeSize(N, H, W, C, 0, 0)) { delete L; return; }
        if (!bi.ConvertFromCompactNCHW(&in[0], N, C, H, W)) { delete L; return; }
        if (!to.ChangeSize(N, top_H, top_W, C, 0, 0)) { delete L; return; }
        std::vector<TEN*> bv(1, &bi), tv(1, &to);
        if (!L->Forward(&bv, &tv)) { delete L; return; }
        delete L;
        got.assign((size_t)N * C * top_H * top_W, -12345.0f);
        to.ConvertToCompactNCHW(&got[0]);
        // 层的 GetTopDim 必须与参考的 need_* 一致，否则形状本身就对不上
        const int want_H = (int)ceil((float)(H + c.pT + c.pB - c.kH) / c.sH + 1);
        const int want_W = (int)ceil((float)(W + c.pL + c.pR - c.kW) / c.sW + 1);
        if (top_H != want_H || top_W != want_W) {
            FILE* f = fopen(RES_FILE, "w");
            if (f) { fprintf(f, "0 1 0.000000e+00\n"); fclose(f); }
            return;
        }
    }

    long n_ok = 0, n_bad = 0; double worst = 0.0;
    ref_pool(in, c, got, n_ok, n_bad, worst);
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %.6e\n", n_ok, n_bad, worst); fclose(f); }
}

typedef void (*RUNNER)(const Case&);
static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(const Case& c, RUNNER r)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) { zq_child_silence_stderr(); r(c); _exit(0); }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0; double worst = 0; int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%ld %ld %lf", &ok, &bad, &worst) == 3); fclose(f); }
    char tag[128];
    snprintf(tag, sizeof(tag), "nchwc%d %s %s C=%-3d k=%dx%d s=%dx%d pad=%d,%d,%d,%d",
             c.align, g_seg[c.seg], c.maxpool ? "MAX" : "AVG", c.C, c.kH, c.kW, c.sH, c.sW,
             c.pT, c.pB, c.pL, c.pR);
    if (!have) { g_crash++; printf("  %-64s 没跑完\n", tag); return; }
    if (WIFSIGNALED(st)) { g_crash++; printf("  %-64s CRASH(%d)\n", tag, WTERMSIG(st)); return; }
    if (bad > 0) { g_bad++; printf("  %-64s FAIL %ld/%ld 格不同, 最大差 %.3e\n", tag, bad, ok + bad, worst); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC 池化的带 padding 行为：零填充后池化（明文定义）\n");
    printf("两段覆盖：前向（内核 + padding）与 层（参数是否真的传下去）\n");
    printf("AVG 除以窗口落在补齐区里的格数；pad 覆盖 0/对称/非对称两种方向/不等\n\n");

    RUNNER r[3] = { &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC1>,
                    &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC4>,
                    &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC8> };

    static const int PADS[5][4] = {
        { 0, 0, 0, 0 },   // 对照
        { 1, 1, 1, 1 },   // 对称
        { 0, 1, 0, 1 },   // 非对称（VALID 会产生这种）
        { 1, 0, 1, 0 },   // 非对称（反方向）
        { 2, 1, 2, 1 },   // 不对称且不等
    };
    for (int ai = 0; ai < 3; ai++) {
        const int A = (ai == 0) ? 1 : (ai == 1 ? 4 : 8);
        const int o0 = g_case, b0 = g_bad, x0 = g_crash;
        for (int seg = 0; seg < 2; seg++)
            for (int mp = 0; mp < 2; mp++)
                for (int m = 0; m < 2; m++)
                    for (int pi = 0; pi < 5; pi++) {
                        Case c; memset(&c, 0, sizeof(c));
                        c.align = A; c.seg = seg; c.maxpool = mp;
                        c.N = 2; c.H = 7; c.W = 9; c.C = (m ? A + 2 : A);
                        c.kH = 3; c.kW = 3; c.sH = 2; c.sW = 2;
                        c.pT = PADS[pi][0]; c.pB = PADS[pi][1];
                        c.pL = PADS[pi][2]; c.pR = PADS[pi][3];
                        one(c, r[ai]);
                    }
        printf("  nchwc%d  %d 个用例：对 %d，错 %d，崩/未跑 %d\n",
               A, g_case - o0, (g_case - o0) - (g_bad - b0) - (g_crash - x0), g_bad - b0, g_crash - x0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩/未跑 %d\n", g_case, g_ok, g_bad, g_crash);
    return (g_bad || g_crash) ? 1 : 0;
}