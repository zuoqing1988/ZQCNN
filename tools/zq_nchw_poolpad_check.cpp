/* NCHW 池化「带 padding」与明文定义的逐位对拍 —— 附录 DO
 *
 * 判据：NCHW 的 `MaxPooling` / `AVGPooling` 在 `pad != 0` 时，
 * 必须等于「把输入零填充到 (H+pt+pb) x (W+pl+pr)，再按 kernel/stride 池化」。
 *
 * 已测到的根因（不是推演，是量出来的）
 * ------------------------------------
 * `ZQ_CNN_Forward_SSEUtils::MaxPooling` 在函数开头读
 *     int in_pixStep = input.GetPixelStep();
 *     int in_widthStep = input.GetWidthStep();
 * 之后才进 padding 分支：
 *     input.Padding(pad_W_left, pad_W_right, pad_H_top, pad_H_bottom, 0);
 *     const float* in_data = input.GetFirstPixelPtr()
 *                          - pad_H_top*in_widthStep - pad_W_left*in_pixStep;
 * 而 `Padding` **会重建张量**（realW 4 -> 6），于是 `widthStep` 变了：
 *
 *     == Padding 之前 ==   pixelStep=4  widthStep=16  borderW=0
 *     == Padding 之后 ==   pixelStep=4  widthStep=24  borderW=1
 *
 * `in_data` 用的是**旧的那个 16**，正确的应该是 24。
 * 4x4 / C=4 / pad=1 实测：正确起点相对新 firstPixelData 是 **-28** float，
 * 现有代码给的是 **-20** float，差 8 float（= 新旧 widthStep 之差）。
 *
 * 只要 pad != 0 **且** Padding 改变了步长（realW 变化 ⇒ widthStep 变化），
 * 窗口起点就是错的。
 *
 * 覆盖：k{2,3} x s{1,2,3} x pad{对称 0/1/2、非对称 (0,1)/(1,0)/(1,2)} x {MAX, AVG}，
 * 输入 5x5~11x11、C = align 与 align+2（后者让"补齐区不是整数个通道"也覆盖到）。
 * 形状不一致**单独记一类**，不混进"数值不同"。
 */
#include "zq_check_child.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "ZQCNN/ZQ_CNN_Tensor4D.h"
#include "ZQCNN/ZQ_CNN_Forward_SSEUtils.h"

#define RES_FILE "/tmp/zq_poolpad_res.txt"

struct Case {
    int H, W, C, k, s, pT, pB, pL, pR, maxpool;
};

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    // 有正有负：MAX 在"补 0"与"补 -inf"下对**全负**输入才会分道扬镳
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

static void run_one(const Case& c)
{
    const int N = 1;
    std::vector<float> in((size_t)N * c.C * c.H * c.W);
    for (size_t i = 0; i < in.size(); i++) in[i] = val(1, (int)i);

    const int pH = c.H + c.pT + c.pB;
    const int pW = c.W + c.pL + c.pR;
    // **尺寸约定必须用 ZQCNN 自己的那个**（附录 DE.1 已经查清）：
    //     out = ceil((in + pad_top + pad_bottom - kernel) / stride) + 1
    // 注意那个 **`+ 1` 在 ceil 里面**，不是 Caffe 的 ceil((in-k)/s) + 1。
    // 第一版我按 Caffe 写，于是**连 pad=0 的对照组都全错**（in=8 k=3 s=3：
    // 本仓库给 3、标准式给 2）—— 症状是"392 个形状不同"，真实原因全是参考自己。
    const int oH = (int)ceil((float)(c.H + c.pT + c.pB - c.k) / c.s + 1);
    const int oW = (int)ceil((float)(c.W + c.pL + c.pR - c.k) / c.s + 1);
    if (oH <= 0 || oW <= 0) return;

    ZQ::ZQ_CNN_Tensor4D_NHW_C_Align128bit t, out;
    if (!t.ChangeSize(N, c.H, c.W, c.C, 0, 0)) return;
    if (!t.ConvertFromCompactNCHW(&in[0], N, c.C, c.H, c.W)) return;
    if (c.maxpool)
        ZQ::ZQ_CNN_Forward_SSEUtils::MaxPooling(t, out, c.k, c.k, c.s, c.s,
                                               c.pT, c.pB, c.pL, c.pR, false);
    else
        ZQ::ZQ_CNN_Forward_SSEUtils::AVGPooling(t, out, c.k, c.k, c.s, c.s,
                                               c.pT, c.pB, c.pL, c.pR, false);

    if (out.GetH() != oH || out.GetW() != oW) {
        FILE* f = fopen(RES_FILE, "w");
        if (f) { fprintf(f, "SHAPE %d %d %d %d\n", oH, oW, out.GetH(), out.GetW()); fclose(f); }
        return;
    }

    std::vector<float> got((size_t)N * c.C * oH * oW);
    out.ConvertToCompactNCHW(&got[0]);

    long n_ok = 0, n_bad = 0; double worst = 0.0;
    for (int ch = 0; ch < c.C; ch++)
        for (int oh = 0; oh < oH; oh++)
            for (int ow = 0; ow < oW; ow++) {
                double sum = 0.0; int cnt = 0;
                bool first = true; double mx = 0.0;
                // **窗口起点要换算到"补齐区下标"再比**：
                //     原图行 ih  <->  补齐区行 ih + pad_top
                //     原图窗口起点 oh*S - pad_top  ->  补齐区下标 oh*S
                // 第一版直接拿 `oh*S - pT` 去和 [0, pH) 比，**等于把 pad 减了两次**
                // （坐标混用），于是参考自己给出 0 0 0 / 0 6 8 / 0 14 16 这种
                // 明显不对的东西，还被我当成"库错了"记了半年。
                // 这一版用 ih（补齐区下标）与 okh（补齐区内第几行）分开算。
                for (int kh = 0; kh < c.k; kh++) {
                    const int ih = oh * c.s + kh;          // 补齐区下标
                    if (ih < 0 || ih >= pH) continue;
                    for (int kw = 0; kw < c.k; kw++) {
                        const int iw = ow * c.s + kw;      // 补齐区下标
                        if (iw < 0 || iw >= pW) continue;
                        // 落在原图内取值，落在补齐区（=0）取 0
                        const bool in_orig = (ih >= c.pT && ih < c.pT + c.H &&
                                             iw >= c.pL && iw < c.pL + c.W);
                        const double v = in_orig
                            ? in[((size_t)ch * c.H + (ih - c.pT)) * c.W + (iw - c.pL)]
                            : 0.0;
                        sum += v; cnt++;
                        if (first || v > mx) { mx = v; first = false; }
                    }
                }
                if (cnt == 0) continue;
                const double want = c.maxpool ? mx : sum / (double)cnt;
                const double g = got[((size_t)ch * oH + oh) * oW + ow];
                const double d = fabs(g - want);
                if (d > worst) worst = d;
                if (d > 1e-5 * (fabs(want) + 1.0)) n_bad++; else n_ok++;
            }
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "VAL %ld %ld %.6e\n", n_ok, n_bad, worst); fclose(f); }
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_shape = 0, g_crash = 0;

static void one(const Case& c)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) { zq_child_silence_stderr(); run_one(c); _exit(0); }
    int st = 0; waitpid(pid, &st, 0);
    char kind[8] = { 0 };
    long a = 0, b = 0, d = 0, e = 0; double worst = 0; int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) {
        have = (fscanf(f, "%7s %ld %ld %ld %lf", kind, &a, &b, &d, &worst) >= 2);
        fclose(f);
    }
    char tag[128];
    snprintf(tag, sizeof(tag), "%s k=%d s=%d pad=%d,%d,%d,%d %dx%d C=%d",
             c.maxpool ? "MAX" : "AVG", c.k, c.s, c.pT, c.pB, c.pL, c.pR, c.H, c.W, c.C);
    if (!have) { g_crash++; printf("  %-56s 没跑完\n", tag); return; }
    if (WIFSIGNALED(st)) { g_crash++; printf("  %-56s CRASH(%d)\n", tag, WTERMSIG(st)); return; }
    if (strcmp(kind, "SHAPE") == 0) {
        g_shape++;
        printf("  %-56s 形状不同 want(%ld,%ld) got(%ld,%ld)\n", tag, a, b, d, e);
        return;
    }
    if (b > 0) { g_bad++; printf("  %-56s FAIL %ld/%ld 格不同, 最大差 %.3e\n", tag, b, a + b, worst); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHW 池化带 padding：与「零填充后池化」的明文定义逐位对拍\n\n");

    static const int PADS[6][4] = {
        { 0, 0, 0, 0 },   // 对照：pad=0 应当全对
        { 1, 1, 1, 1 },   // 对称
        { 2, 2, 2, 2 },   // 对称、更大（>1 时 realW 变化更大）
        { 0, 1, 0, 1 },   // 非对称（VALID 会产生这种）
        { 1, 0, 1, 0 },   // 非对称、反方向
        { 1, 2, 1, 2 },   // 不对称且不等
    };
    static const int KS[2] = { 2, 3 };
    static const int SS[3] = { 1, 2, 3 };
    static const int HS[3] = { 5, 8, 11 };

    for (int mp = 0; mp < 2; mp++)
        for (int ki = 0; ki < 2; ki++)
            for (int si = 0; si < 3; si++)
                for (int hi = 0; hi < 3; hi++)
                    for (int ci = 0; ci < 2; ci++)
                        for (int pi = 0; pi < 6; pi++) {
                            Case c;
                            memset(&c, 0, sizeof(c));
                            c.maxpool = mp; c.k = KS[ki]; c.s = SS[si];
                            c.H = HS[hi]; c.W = HS[(hi + 1) % 3];
                            c.C = ci ? 6 : 4;      // 4 = align 本身；6 让补齐区不是整数个通道
                            c.pT = PADS[pi][0]; c.pB = PADS[pi][1];
                            c.pL = PADS[pi][2]; c.pR = PADS[pi][3];
                            one(c);
                        }

    printf("\n共 %d 个用例：全对 %d，数值不同 %d，形状不同 %d，崩/未跑 %d\n",
           g_case, g_ok, g_bad, g_shape, g_crash);
    return (g_bad || g_shape || g_crash) ? 1 : 0;
}