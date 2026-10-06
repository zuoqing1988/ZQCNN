/* NCHWC 与 NCHW 的池化**逐位对拍**（附录 F2，待登记进回归）
 *
 * 契约：同一份数据、同一个池化参数，NCHWC 的结果必须与 NCHW **逐位相同**。
 *
 * 为什么这条契约成立、而别的算子不一定
 * ------------------------------------
 * NCHW 与 NCHWC 是同一套数学的两种布局，NCHW 是**参考**。
 * 这也是本文件「同仓的两份实现互为对照」那条规则的直接应用：
 * 一份对一份错，就不是设计取舍。
 *
 * 现在要抓的是
 * ------------
 * `ZQ_CNN_Layer_NCHWC_Pooling::ReadParam` **解析了** `pad`（写进 pad_H/pad_W），
 * 而 `Forward` 调的 `MaxPooling(..., kernel_H, kernel_W, stride_H, stride_W, global_pool)`
 * **签名里根本没有 pad** —— 于是模型文件里写 `pad 1` 会**加载成功、然后静默按无 pad 计算**。
 *
 * 另外 NCHWC 那一族**完全没有 pad_type**（`grep -c pad_type ZQ_CNN_Layer_NCHWC.h` = 0），
 * 所以 `pad_type=SAME` 会落进 "unknown para" 分支、只打一行警告。
 * 而随仓有 **220 个** pad_type=SAME 层（Pose-zq 147 / det5-112-gray 36 /
 * headposegaze-112-gray 37）—— 把它们转成 NCHWC 就是整网算错。
 *
 * 判据用**逐位相同**而不是后向误差：两侧跑的是同一份数学，
 * 任何差异都是布局/参数传递问题，浮点误差不该出现。
 * 形状不一致时**直接判失败并报两个形状**，不要拿它去比数值。
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "ZQCNN/ZQ_CNN_Tensor4D.h"
#include "ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h"
#include "ZQCNN/ZQ_CNN_Forward_SSEUtils.h"
#include "ZQCNN/ZQ_CNN_Forward_SSEUtils_NCHWC.h"

#define RES_FILE "/tmp/zq_poolpar_res.txt"

struct Case { int align, N, H, W, C, kH, kW, sH, sW, pT, pB, pL, pR, maxpool; };

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

template <class TEN>
static void run_one(const Case& c)
{
    const int N = c.N, H = c.H, W = c.W, C = c.C;
    std::vector<float> in((size_t)N * C * H * W);
    for (size_t i = 0; i < in.size(); i++) in[i] = val(1, (int)i);

    // ---- NCHW 参考（注意：MaxPooling 会**就地**给 input 做 Padding，
    //      所以每个输出都必须用一份全新的输入张量）----
    // `ZQ_CNN_Tensor4D` 是**抽象类**（一堆纯虚函数），必须挑一个具体变体。
    // 这里用 Align128bit：`Align256bit` 的 SIMD 路径要求缓冲区 32 字节对齐，
    // 而 std::vector 只给 16 —— 那会是**探针自己的坑**（附录 IW.13）。
    typedef ZQ::ZQ_CNN_Tensor4D_NHW_C_Align128bit NCHWT;
    NCHWT ref;
    if (!ref.ChangeSize(N, H, W, C, 0, 0)) return;
    if (!ref.ConvertFromCompactNCHW(&in[0], N, C, H, W)) return;
    NCHWT refout;
    if (c.maxpool)
        ZQ::ZQ_CNN_Forward_SSEUtils::MaxPooling(ref, refout, c.kH, c.kW, c.sH, c.sW,
                                                c.pT, c.pB, c.pL, c.pR, false);
    else
        ZQ::ZQ_CNN_Forward_SSEUtils::AVGPooling(ref, refout, c.kH, c.kW, c.sH, c.sW,
                                                c.pT, c.pB, c.pL, c.pR, false);

    // ---- NCHWC 被测 ----
    TEN ti, to;
    if (!ti.ChangeSize(N, H, W, C, 0, 0)) return;
    if (!ti.ConvertFromCompactNCHW(&in[0], N, C, H, W)) return;
    const int oH = refout.GetH(), oW = refout.GetW();
    if (oH <= 0 || oW <= 0) return;
    if (!to.ChangeSize(N, oH, oW, C, 0, 0)) return;
    // NCHWC 的签名里**没有 pad**，只能按"库里实际能表达的东西"调用 ——
    // 这正是本探针要暴露的差距。
    if (c.maxpool)
        ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::MaxPooling(ti, to, c.kH, c.kW, c.sH, c.sW, false);
    else
        ZQ::ZQ_CNN_Forward_SSEUtils_NCHWC::AVGPooling(ti, to, c.kH, c.kW, c.sH, c.sW, false);

    if (to.GetH() != oH || to.GetW() != oW) {
        // 形状不一致：单独一种失败，别混进"数值不同"那一桶。
        FILE* f = fopen(RES_FILE, "w");
        if (f) { fprintf(f, "SHAPE %d %d %d %d\n", oH, oW, to.GetH(), to.GetW()); fclose(f); }
        return;
    }

    std::vector<float> a((size_t)N * C * oH * oW), b(a.size());
    refout.ConvertToCompactNCHW(&a[0]);
    to.ConvertToCompactNCHW(&b[0]);
    long n_ok = 0, n_bad = 0; double worst = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > worst) worst = d;
        if (d > 0) n_bad++; else n_ok++;
    }
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "VAL %ld %ld %.6e\n", n_ok, n_bad, worst); fclose(f); }
}

typedef void (*RUNNER)(const Case&);

static int g_case = 0, g_ok = 0, g_bad = 0, g_shape = 0, g_crash = 0;

static void one(const Case& c, RUNNER r)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) { r(c); _exit(0); }
    int st = 0; waitpid(pid, &st, 0);
    char kind[8] = { 0 };
    long ok = 0, bad = 0, w2 = 0, w3 = 0; double worst = 0; int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) {
        // SHAPE 行是 `SHAPE wantH wantW gotH gotW`（4 个数），VAL 行是
        // `VAL ok bad worst`（2 个数 + 1 个浮点）—— **两种格式不一样**，
        // 所以先按 token 读 kind，再按 kind 决定后面读几个。
        if (fscanf(f, "%7s", kind) == 1) {
            if (strcmp(kind, "SHAPE") == 0)
                have = (fscanf(f, "%ld %ld %ld %ld", &ok, &bad, &w2, &w3) == 4);
            else
                have = (fscanf(f, "%ld %ld %lf", &ok, &bad, &worst) == 3);
        }
        fclose(f);
    }
    char tag[160];
    snprintf(tag, sizeof(tag), "nchwc%d %s k=%dx%d s=%dx%d pad=%d,%d,%d,%d in=%dx%d",
             c.align, c.maxpool ? "MAX" : "AVG", c.kH, c.kW, c.sH, c.sW,
             c.pT, c.pB, c.pL, c.pR, c.H, c.W);
    if (!have) { g_crash++; printf("  %-72s 没跑完\n", tag); return; }
    if (strcmp(kind, "SHAPE") == 0) {
        g_shape++;
        printf("  %-72s 形状不同 want(%ld,%ld) got(%ld,%ld)\n", tag, ok, bad, w2, w3);
        return;
    }
    if (bad > 0) { g_bad++; printf("  %-72s FAIL %ld/%ld 格不同, 最大差 %.6e\n", tag, bad, ok + bad, worst); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC vs NCHW 池化逐位对拍（NCHW 是参考）\n");
    printf("pad=0 的用例是**对照**（两侧本来就该一样）；pad!=0 的用例是**判据**\n\n");

    RUNNER r[3] = { &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC1>,
                    &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC4>,
                    &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC8> };

    for (int ai = 0; ai < 3; ai++) {
        const int A = (ai == 0) ? 1 : (ai == 1 ? 4 : 8);
        const int o0 = g_case, b0 = g_bad, s0 = g_shape;
        for (int mp = 0; mp < 2; mp++)
            for (int m = 0; m < 2; m++) {
                // C = A 是对照，pad 恒 0；
                // C = A+2 让 dst_slice>=1，pad!=0 的差异才不会被别的东西掩盖。
                Case c; memset(&c, 0, sizeof(c));
                c.align = A; c.maxpool = mp; c.N = 2; c.H = 7; c.W = 9; c.C = (m ? A + 2 : A);
                c.kH = 3; c.kW = 3; c.sH = 2; c.sW = 2;
                c.pT = c.pB = c.pL = c.pR = 0;
                one(c, r[ai]);                                   // 对照：pad = 0
                c.pT = c.pB = c.pL = c.pR = 1;
                one(c, r[ai]);                                   // 判据：pad = 1
                c.pT = c.pL = 0; c.pB = c.pR = 1;
                one(c, r[ai]);                                   // 判据：非对称（VALID/SAME 会产生）
            }
        printf("  nchwc%d  %d 个用例：全对 %d，数值不同 %d，形状不同 %d，崩 %d\n",
               A, g_case - o0, (g_case - o0) - (g_bad - b0) - (g_shape - s0),
               g_bad - b0, g_shape - s0, g_crash);
    }
    printf("\n共 %d 个用例：全对 %d，数值不同 %d，形状不同 %d，崩 %d\n",
           g_case, g_ok, g_bad, g_shape, g_crash);
    return (g_bad || g_shape || g_crash) ? 1 : 0;
}