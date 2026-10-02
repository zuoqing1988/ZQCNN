/* NCHWC softmax 门禁 —— 附录 CK
 *
 * 覆盖面：5 个入口（符号表用 nm 核实）
 *   zq_cnn_softmax_nchwc1_C / _H / _W
 *   zq_cnn_softmax_nchwc4_C
 *   zq_cnn_softmax_nchwc8_C
 *
 * 语义（从 `zq_cnn_softmax_nchwc_raw.h` 与 .c 里的两个手写变体读出来）：
 *   沿指定轴做标准 softmax，**就地**：
 *       max_val = max(该轴上的值)
 *       v = exp(v - max_val);  sum = Σ v;  v = v / sum
 *   `_C` 沿通道轴、`_H` 沿 H、`_W` 沿 W。
 *
 * 顺带记一个"看着像 bug 其实不是"的写法（CK.3）：
 *   尾循环 `for (; c < in_C; c++, slice_ptr++)` 里的 `slice_ptr++`
 *   看上去该是 `+= in_sliceStep`，但主循环退出时 slice_ptr 已经停在
 *   正确位置上，尾循环**先读后加**、加完就丢弃，所以是对的。
 *
 * 沿用 CB/CE/CF/CG/CH/CI/CJ：名字写全走函数指针表、后向误差 + 逐格统计、
 * 每用例 fork 子进程并**显式判"没读到结果文件"= 失败**（CJ.4）、
 * 用真实的 ZQ_CNN_Tensor4D_NCHWC{1,4,8}。
 */
#include "zq_check_child.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h"
#include "ZQCNN/layers_nchwc/zq_cnn_softmax_nchwc.h"

typedef void (*FN)(float* d, int N, int H, int W, int C, int ws, int ss, int is);

enum { AX_C = 0, AX_H = 1, AX_W = 2 };
static const char* g_axis_name[3] = { "C", "H", "W" };

struct Entry { FN fn; int axis; int align; };

// 5 行平铺，**每个内核名写全**
static const Entry g_entries[5] = {
  { zq_cnn_softmax_nchwc1_C, AX_C, 1 },
  { zq_cnn_softmax_nchwc1_H, AX_H, 1 },
  { zq_cnn_softmax_nchwc1_W, AX_W, 1 },
  { zq_cnn_softmax_nchwc4_C, AX_C, 4 },
  { zq_cnn_softmax_nchwc8_C, AX_C, 8 },
};
static const int N_ENTRY = 5;

#define RES_FILE "/tmp/zq_sm_res.txt"
static const double TOL = 1e-5;

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

struct Case { int entry, N, H, W, C; };

typedef void (*RUNNER)(const Case&);
template <class TEN>
static void run_one(const Case& c)
{
    const Entry& e = g_entries[c.entry];
    const int A = TEN().GetAlignSize();
    const int N = c.N, H = c.H, W = c.W, C = c.C;

    std::vector<float> in((size_t)N * C * H * W);
    for (size_t i = 0; i < in.size(); i++) in[i] = val(1, (int)i);
    TEN t;
    if (!t.ChangeSize(N, H, W, C, 0, 0)) return;
    if (!t.ConvertFromCompactNCHW(&in[0], N, C, H, W)) return;
    const int ws = t.GetWidthStep(), ss = t.GetSliceStep(), is = t.GetImageStep();

    e.fn(t.GetFirstPixelPtr(), N, H, W, C, ws, ss, is);

    // ---- 参考：沿指定轴做 softmax ----
    // 沿轴外的两个维度要**各自独立**地遍历：_C 时是 (h,w)，_H 时是 (c,w)，_W 时是 (c,h)
    const int AL = (e.axis == AX_C) ? C : (e.axis == AX_H) ? H : W;
    const int O1 = (e.axis == AX_C) ? H : C;          // 第一个非轴维度的大小
    const int O2 = (e.axis == AX_C) ? W : W;          // 第二个非轴维度：_C/_H 都是 W，_W 是 H
    const int O2s = (e.axis == AX_W) ? H : W;
    long n_ok = 0, n_bad = 0; double worst = 0.0;
    for (int n = 0; n < N; n++)
        for (int o1 = 0; o1 < O1; o1++)
            for (int o2 = 0; o2 < O2s; o2++) {
                double v[128];
                if (AL > 128) return;
                double mx = -1e300, sum = 0.0;
                for (int i = 0; i < AL; i++) {
                    int ic, ih, iw;
                    if (e.axis == AX_C)      { ic = i;  ih = o1; iw = o2; }
                    else if (e.axis == AX_H) { ic = o1; ih = i;  iw = o2; }
                    else                    { ic = o1; ih = o2; iw = i;  }
                    v[i] = in[(((size_t)n * C + ic) * H + ih) * W + iw];
                    if (v[i] > mx) mx = v[i];
                }
                for (int i = 0; i < AL; i++) { v[i] = exp(v[i] - mx); sum += v[i]; }
                for (int i = 0; i < AL; i++) {
                    int ic, ih, iw;
                    if (e.axis == AX_C)      { ic = i;  ih = o1; iw = o2; }
                    else if (e.axis == AX_H) { ic = o1; ih = i;  iw = o2; }
                    else                    { ic = o1; ih = o2; iw = i;  }
                    double y = v[i] / sum;
                    double got = t.GetFirstPixelPtr()[n * is + (ic / A) * ss
                                                      + ih * ws + iw * A + (ic % A)];
                    // softmax 的输出是概率，用它自己当尺度
                    double den = (y > 1e-6) ? y : 1.0;
                    double be = fabs(got - y) / den;
                    if (be > TOL) n_bad++; else n_ok++;
                    if (be > worst) worst = be;
                }
            }
    (void)O2;
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %.6e\n", n_ok, n_bad, worst); fclose(f); }
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(const Case& c, RUNNER r)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) {
        zq_child_silence_stderr();
        r(c);
        _exit(0);
    }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0; double worst = 0;
    int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%ld %ld %lf", &ok, &bad, &worst) == 3); fclose(f); }
    char nm[56], tag[64];
    snprintf(nm, sizeof(nm), "nchwc%d softmax_%s", g_entries[c.entry].align, g_axis_name[g_entries[c.entry].axis]);
    snprintf(tag, sizeof(tag), "N=%d %dx%d C=%d", c.N, c.H, c.W, c.C);
    if (!have) { g_crash++; printf("  %-24s %s  没跑完（子进程没写结果文件，退出码 %d）\n", nm, tag, WEXITSTATUS(st)); return; }
    if (WIFSIGNALED(st)) { g_crash++; printf("  %-24s %s  CRASH(信号 %d)\n", nm, tag, WTERMSIG(st)); return; }
    if (bad > 0) { g_bad++; printf("  %-24s %s  FAIL %ld/%ld 格错, 最差 %.3e\n", nm, tag, bad, ok + bad, worst); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC softmax：5 个入口（沿 C / H / W 三个轴）\n");
    printf("内核名全部写全、走函数指针表（nm 核实）；用真实 ZQ_CNN_Tensor4D_NCHWC{1,4,8}\n");
    printf("判据：softmax 输出是概率，用它自己当尺度；逐格统计\n");
    printf("C 取「是 align 倍数」与「不是 align 倍数」两种，都跑（对齐组的尾循环要走到）\n\n");

    RUNNER r[3] = { &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC1>,
                    &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC4>,
                    &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC8> };
    for (int e = 0; e < N_ENTRY; e++) {
        const int ai = (g_entries[e].align == 1) ? 0 : (g_entries[e].align == 4 ? 1 : 2);
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        const int A = g_entries[e].align;
        // 0: C 恰好是 align 的倍数（走不到尾循环）；1: 不是（尾循环必须被走到）
        for (int j = 0; j < 2; j++) {
            Case c; memset(&c, 0, sizeof(c));
            c.entry = e; c.N = (j ? 1 : 2); c.H = 5; c.W = 7;
            c.C = (j ? A + 3 : A * 2);
            one(c, r[ai]);
        }
        Case c2; memset(&c2, 0, sizeof(c2));
        c2.entry = e; c2.N = 1; c2.H = 3; c2.W = 4; c2.C = 1;    // C=1：全部落在尾循环里
        one(c2, r[ai]);
        printf("  nchwc%d softmax_%-2s          %d 个用例：对 %d，错 %d，崩 %d\n",
               A, g_axis_name[g_entries[e].axis], g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
