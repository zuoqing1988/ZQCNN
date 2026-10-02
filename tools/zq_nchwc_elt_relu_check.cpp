/* NCHWC relu + eltwise 门禁 —— 附录 CH
 *
 * 为什么要有这个门禁
 * ------------------
 * 附录 CG 收尾时列的空白里，NCHWC 的 relu / eltwise 排在 pooling 之前：
 * 两者都是**逐元素**运算，语义没有歧义（不像 batchnormscale 那种逐通道
 * 归一化数学），所以参考实现几乎不可能写错 —— 也就意味着门禁一旦变红，
 * 结论可信。
 *
 * 覆盖面：relu(3) + eltwise sum / max / mul / sum_with_weight(12) = **15 个入口**
 *         （符号表已用 `nm` 核实）
 *
 * 语义（逐条从源码读出来的，不是猜的）：
 *   relu(data, ..., slope)   就地；`slope == 0` 时 out = max(0, x)，
 *                             否则 out = slope*min(0,x) + max(0,x)
 *                             —— raw.h:23 那个 `if (slope == 0)` 是唯一的分派
 *   eltwise_sum              out = Σ in[i]
 *   eltwise_max              out = max_i in[i]
 *   eltwise_mul              out = Π in[i]
 *   eltwise_sum_with_weight  out = Σ weight[i] * in[i]
 *                             weight 是**每张输入一个标量**（raw 里
 *                             `zq_mm_set1_ps(weight[i])` 广播），不是逐通道
 *
 * 沿用 CB/CE/CF/CG 已验证过的做法：名字写全走函数指针表、后向误差 + 逐格统计、
 * 每用例 fork 子进程、用真实的 ZQ_CNN_Tensor4D_NCHWC{1,4,8} 类。
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
#include "ZQCNN/layers_nchwc/zq_cnn_relu_nchwc.h"
#include "ZQCNN/layers_nchwc/zq_cnn_eltwise_nchwc.h"

typedef void (*FN_RELU)(float* data, int N, int H, int W, int C,
                         int ws, int ss, int is, float slope);
typedef void (*FN_ELT)(int num, const float** ins, int N, int H, int W, int C,
                       const int* ws_arr, const int* ss_arr, const int* is_arr,
                       float* out, int ows, int oss, int ois);
typedef void (*FN_ELTW)(int num, const float** ins, const float* weight,
                        int N, int H, int W, int C,
                        const int* ws_arr, const int* ss_arr, const int* is_arr,
                        float* out, int ows, int oss, int ois);

enum { E_RELU = 0, E_SUM = 1, E_MAX = 2, E_MUL = 3, E_SUMW = 4, E_COUNT };
static const char* g_kind_name[E_COUNT] = { "relu", "eltwise_sum", "eltwise_max",
                                             "eltwise_mul", "eltwise_sum_with_weight" };

struct Entry { void* fn; int kind; int align; };

// 15 行平铺，**每个内核名写全**
static const Entry g_entries[15] = {
  { (void*)zq_cnn_relu_nchwc1, E_RELU, 1 },
  { (void*)zq_cnn_relu_nchwc4, E_RELU, 4 },
  { (void*)zq_cnn_relu_nchwc8, E_RELU, 8 },

  { (void*)zq_cnn_eltwise_sum_nchwc1, E_SUM, 1 },
  { (void*)zq_cnn_eltwise_sum_nchwc4, E_SUM, 4 },
  { (void*)zq_cnn_eltwise_sum_nchwc8, E_SUM, 8 },
  { (void*)zq_cnn_eltwise_max_nchwc1, E_MAX, 1 },
  { (void*)zq_cnn_eltwise_max_nchwc4, E_MAX, 4 },
  { (void*)zq_cnn_eltwise_max_nchwc8, E_MAX, 8 },
  { (void*)zq_cnn_eltwise_mul_nchwc1, E_MUL, 1 },
  { (void*)zq_cnn_eltwise_mul_nchwc4, E_MUL, 4 },
  { (void*)zq_cnn_eltwise_mul_nchwc8, E_MUL, 8 },
  { (void*)zq_cnn_eltwise_sum_with_weight_nchwc1, E_SUMW, 1 },
  { (void*)zq_cnn_eltwise_sum_with_weight_nchwc4, E_SUMW, 4 },
  { (void*)zq_cnn_eltwise_sum_with_weight_nchwc8, E_SUMW, 8 },
};
static const int N_ENTRY = 15;

#define RES_FILE "/tmp/zq_er_res.txt"
static const double TOL = 1e-6;

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

struct Case { int entry, N, H, W, C, num_in, slope_sel; };

typedef void (*RUNNER)(const Case&);
template <class TEN>
static void run_one(const Case& c)
{
    const Entry& e = g_entries[c.entry];
    const int A = TEN().GetAlignSize();
    const int N = c.N, H = c.H, W = c.W, C = c.C;

    if (e.kind == E_RELU) {
        const float slope = (c.slope_sel == 0) ? 0.0f : 0.125f;   // 两个分支都要走
        std::vector<float> in((size_t)N * C * H * W);
        for (size_t i = 0; i < in.size(); i++) in[i] = val(1, (int)i);
        TEN t;
        if (!t.ChangeSize(N, H, W, C, 0, 0)) return;
        if (!t.ConvertFromCompactNCHW(&in[0], N, C, H, W)) return;
        float* p = t.GetFirstPixelPtr();
        ((FN_RELU)e.fn)(p, N, H, W, C, t.GetWidthStep(), t.GetSliceStep(), t.GetImageStep(), slope);

        long n_ok = 0, n_bad = 0; double worst = 0.0;
        for (int n = 0; n < N; n++)
            for (int ch = 0; ch < C; ch++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++) {
                        double x = in[(((size_t)n * C + ch) * H + h) * W + w];
                        double y = (slope == 0) ? (x > 0 ? x : 0.0) : (x > 0 ? x : slope * x);
                        double got = p[n * t.GetImageStep() + (ch / A) * t.GetSliceStep()
                                       + h * t.GetWidthStep() + w * A + (ch % A)];
                        double den = fabs(y) > 1.0 ? fabs(y) : 1.0;
                        double be = fabs(got - y) / den;
                        if (be > TOL) n_bad++; else n_ok++;
                        if (be > worst) worst = be;
                    }
        FILE* f = fopen(RES_FILE, "w");
        if (f) { fprintf(f, "%ld %ld %.6e\n", n_ok, n_bad, worst); fclose(f); }
        return;
    }

    // ---- eltwise：多张输入 ----
    const int T = c.num_in;
    std::vector<std::vector<float> > hv(T);
    std::vector<TEN> tt(T);
    std::vector<const float*> ptr(T);
    std::vector<int> ws(T), ss(T), is(T);
    for (int t = 0; t < T; t++) {
        hv[t].resize((size_t)N * C * H * W);
        for (size_t i = 0; i < hv[t].size(); i++) hv[t][i] = val(10 + t, (int)i);
        if (!tt[t].ChangeSize(N, H, W, C, 0, 0)) return;
        if (!tt[t].ConvertFromCompactNCHW(&hv[t][0], N, C, H, W)) return;
        ptr[t] = tt[t].GetFirstPixelPtr();
        ws[t] = tt[t].GetWidthStep(); ss[t] = tt[t].GetSliceStep(); is[t] = tt[t].GetImageStep();
    }
    TEN to;
    if (!to.ChangeSize(N, H, W, C, 0, 0)) return;
    std::vector<float> zero((size_t)N * to.GetImageStep(), 0.0f);
    memcpy(to.GetFirstPixelPtr(), &zero[0], zero.size() * sizeof(float));
    float* op = to.GetFirstPixelPtr();

    float wbuf[8];
    for (int t = 0; t < T; t++) wbuf[t] = 0.5f + 0.25f * t;

    if (e.kind == E_SUMW)
        ((FN_ELTW)e.fn)(T, &ptr[0], wbuf, N, H, W, C, &ws[0], &ss[0], &is[0],
                        op, to.GetWidthStep(), to.GetSliceStep(), to.GetImageStep());
    else
        ((FN_ELT)e.fn)(T, &ptr[0], N, H, W, C, &ws[0], &ss[0], &is[0],
                       op, to.GetWidthStep(), to.GetSliceStep(), to.GetImageStep());

    long n_ok = 0, n_bad = 0; double worst = 0.0;
    for (int n = 0; n < N; n++)
        for (int ch = 0; ch < C; ch++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++) {
                    double y;
                    if (e.kind == E_SUM) {
                        y = 0; for (int t = 0; t < T; t++) y += hv[t][(((size_t)n * C + ch) * H + h) * W + w];
                    } else if (e.kind == E_MAX) {
                        y = hv[0][(((size_t)n * C + ch) * H + h) * W + w];
                        for (int t = 1; t < T; t++) { double v = hv[t][(((size_t)n * C + ch) * H + h) * W + w]; if (v > y) y = v; }
                    } else if (e.kind == E_MUL) {
                        y = 1; for (int t = 0; t < T; t++) y *= hv[t][(((size_t)n * C + ch) * H + h) * W + w];
                    } else {   // E_SUMW
                        y = 0; for (int t = 0; t < T; t++) y += wbuf[t] * hv[t][(((size_t)n * C + ch) * H + h) * W + w];
                    }
                    double got = op[n * to.GetImageStep() + (ch / A) * to.GetSliceStep()
                                   + h * to.GetWidthStep() + w * A + (ch % A)];
                    double den = fabs(y) > 1.0 ? fabs(y) : 1.0;
                    double be = fabs(got - y) / den;
                    if (be > TOL) n_bad++; else n_ok++;
                    if (be > worst) worst = be;
                }
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
    char nm[72], tag[72];
    snprintf(nm, sizeof(nm), "nchwc%d %s", g_entries[c.entry].align, g_kind_name[g_entries[c.entry].kind]);
    if (g_entries[c.entry].kind == E_RELU)
        snprintf(tag, sizeof(tag), "N=%d %dx%d C=%d slope=%s", c.N, c.H, c.W, c.C, c.slope_sel ? "0.125" : "0");
    else
        snprintf(tag, sizeof(tag), "N=%d %dx%d C=%d T=%d", c.N, c.H, c.W, c.C, c.num_in);
    // **结果文件缺失 / 读不出来 = 这个用例没跑完，必须判失败。**
    // ASan 撞上 SEGV 时默认走 Die() -> _exit(1)，**不发信号**，
    // 于是 WIFSIGNALED 为假、退出码也不是 3 —— 缺了这道判断就会把
    // 一个段错误当成"通过"。附录 CJ.4 抓出来的，四个门禁统一补上。
    if (!have) { g_crash++; printf("  没跑完（子进程没写结果文件，退出码 %d）\n", WEXITSTATUS(st)); return; }
    if (WIFSIGNALED(st)) { g_crash++; printf("  %-32s %s  CRASH\n", nm, tag); return; }
    if (bad > 0) { g_bad++; printf("  %-32s %s  FAIL %ld/%ld 格错, 最差 %.3e\n", nm, tag, bad, ok + bad, worst); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC relu + eltwise：5 个动作 x 3 种对齐 = 15 个入口（附录 CH）\n");
    printf("内核名全部写全、走函数指针表；用真实 ZQ_CNN_Tensor4D_NCHWC{1,4,8}\n");
    printf("判据：逐元素后向误差（无归约，阈值 1e-6），逐格统计\n\n");

    RUNNER r[3] = { &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC1>,
                    &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC4>,
                    &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC8> };
    for (int e = 0; e < N_ENTRY; e++) {
        const int ai = (g_entries[e].align == 1) ? 0 : (g_entries[e].align == 4 ? 1 : 2);
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        const int A = g_entries[e].align;
        if (g_entries[e].kind == E_RELU) {
            for (int sl = 0; sl < 2; sl++) {          // slope==0 与 slope!=0 两条分支
                Case c; memset(&c, 0, sizeof(c));
                c.entry = e; c.N = 1; c.H = 5; c.W = 13; c.C = A + 2; c.slope_sel = sl;
                one(c, r[ai]);
            }
            Case c2; memset(&c2, 0, sizeof(c2));
            c2.entry = e; c2.N = 2; c2.H = 3; c2.W = 8; c2.C = A; c2.slope_sel = 1;
            one(c2, r[ai]);
        } else {
            for (int T = 2; T <= 4; T++) {           // 输入张量数：2/3/4
                Case c; memset(&c, 0, sizeof(c));
                c.entry = e; c.N = 1; c.H = 5; c.W = 7; c.C = A + 2; c.num_in = T;
                one(c, r[ai]);
            }
            Case c2; memset(&c2, 0, sizeof(c2));
            c2.entry = e; c2.N = 2; c2.H = 3; c2.W = 8; c2.C = A; c2.num_in = 3;
            one(c2, r[ai]);
        }
        printf("  nchwc%d %-24s  %d 个用例：对 %d，错 %d，崩 %d\n",
               A, g_kind_name[g_entries[e].kind], g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
