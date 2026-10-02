/* NCHWC（layers_nchwc）激活层门禁：addbias / prelu / addbias_prelu —— 附录 CG
 *
 * 为什么要有这个门禁
 * ------------------
 * 附录 CF 指出 NCHWC 这一族里 depthwise 完全没测过（补了，CF）。
 * 剩下的空白里，**激活层是紧挨着卷积的那一块**：
 * `zq_cnn_conv_no_padding_gemm_nchwc*_*_with_bias_prelu` 三个变体内部直接调
 * `zq_mm_fmadd_ps(slope_v, min(0,x), max(0,x))` —— 也就是 prelu。
 * 它一旦错，CB 修好的那 16+ 个卷积入口会**一起错**，
 * 而目前没有任何独立的门禁盯它。
 *
 * 覆盖面：`addbias` / `prelu` / `prelu_sure_slope_lessthan1` /
 *         `addbias_prelu` / `addbias_prelu_sure_slope_lessthan1`
 *         各 × NCHWC1/4/8 = **15 个入口**（已用 `nm` 核实过符号表，
 *         头里那两个宏别名 `zq_cnn_prelu_nchwc` / `..._sure_slope_lessthan1`
 *         只是 include 时的重命名，真实定义就是下面这 15 个）
 *
 * 这三个内核都是**就地**（in & out）运算，所以门禁先把输入拷进张量、
 * 跑完再与"用原始输入算出的参考值"逐格比。
 *
 * 沿用 CB / CE / CF 已验证过的做法
 * -------------------------------
 *  · 内核名**写全**走函数指针表，不做字符串拼接（CA.3）
 *  · 后向误差 + **逐格统计**（CA.5）
 *  · 每用例 fork 一个子进程，子进程 stderr 接 /dev/null、只写结果文件
 *  · 用真实的 ZQ_CNN_Tensor4D_NCHWC1/4/8 类分配与填充
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
#include "ZQCNN/layers_nchwc/zq_cnn_prelu_nchwc.h"
#include "ZQCNN/layers_nchwc/zq_cnn_addbias_nchwc.h"

// 9 参数：最后一个指针是 slope（prelu）或 bias（addbias），由 kind 区分
typedef void (*FN9)(float* data, int N, int H, int W, int C,
                     int ws, int ss, int is, const float* tail);
// 10 参数：bias + slope
typedef void (*FN10)(float* data, int N, int H, int W, int C,
                      int ws, int ss, int is, const float* bias, const float* slope);

enum { K_ADDBIAS = 0, K_PRELU = 1, K_PRELU_SURE = 2, K_ADDBIAS_PRELU = 3, K_ADDBIAS_PRELU_SURE = 4 };
static const char* g_kind_name[5] = {
    "addbias", "prelu", "prelu_sure", "addbias_prelu", "addbias_prelu_sure"
};

struct Entry { void* fn9; void* fn10; int kind; int align; };

// 15 行平铺，**每个内核名写全**（不拼接、不用宏生成）
static const Entry g_entries[15] = {
  { (void*)zq_cnn_addbias_nchwc1, 0, K_ADDBIAS, 1 },
  { (void*)zq_cnn_addbias_nchwc4, 0, K_ADDBIAS, 4 },
  { (void*)zq_cnn_addbias_nchwc8, 0, K_ADDBIAS, 8 },

  { (void*)zq_cnn_prelu_nchwc1, 0, K_PRELU, 1 },
  { (void*)zq_cnn_prelu_nchwc4, 0, K_PRELU, 4 },
  { (void*)zq_cnn_prelu_nchwc8, 0, K_PRELU, 8 },

  { (void*)zq_cnn_prelu_nchwc1_sure_slope_lessthan1, 0, K_PRELU_SURE, 1 },
  { (void*)zq_cnn_prelu_nchwc4_sure_slope_lessthan1, 0, K_PRELU_SURE, 4 },
  { (void*)zq_cnn_prelu_nchwc8_sure_slope_lessthan1, 0, K_PRELU_SURE, 8 },

  { 0, (void*)zq_cnn_addbias_prelu_nchwc1, K_ADDBIAS_PRELU, 1 },
  { 0, (void*)zq_cnn_addbias_prelu_nchwc4, K_ADDBIAS_PRELU, 4 },
  { 0, (void*)zq_cnn_addbias_prelu_nchwc8, K_ADDBIAS_PRELU, 8 },

  { 0, (void*)zq_cnn_addbias_prelu_nchwc1_sure_slope_lessthan1, K_ADDBIAS_PRELU_SURE, 1 },
  { 0, (void*)zq_cnn_addbias_prelu_nchwc4_sure_slope_lessthan1, K_ADDBIAS_PRELU_SURE, 4 },
  { 0, (void*)zq_cnn_addbias_prelu_nchwc8_sure_slope_lessthan1, K_ADDBIAS_PRELU_SURE, 8 },
};
static const int N_ENTRY = 15;

#define RES_FILE "/tmp/zq_act_res.txt"
static const double TOL = 1e-6;   // 逐元素加法/乘法，没有归约，阈值可以更严

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
    const bool has_bias   = (e.kind == K_ADDBIAS || e.kind == K_ADDBIAS_PRELU || e.kind == K_ADDBIAS_PRELU_SURE);
    const bool has_slope  = (e.kind != K_ADDBIAS);

    std::vector<float> in((size_t)N * C * H * W);
    // bias / slope 要按**补齐后的通道数**开，不是按 align ——
    // 内核是 `slope_v = load_ps(slope + c)` 每次读 align 个 float，
    // c 走到最后一个不满的组时会读过界；C = align+2 的用例正好踩到。
    // （第一版按 A 开，于是通道 align/align+1 的 slope 是 0，
    //   参考值和内核"恰好一致"，门禁看起来全绿 —— 变异测试抓出来的。）
    const int paddedC = (C + A - 1) / A * A;
    // 长度按补齐后的通道数（CG.4），并且**必须 32 字节对齐**：
    // align=8 那一族用 `zq_mm_load_ps` = `_mm256_load_ps`，要求 32 字节对齐，
    // `std::vector<float>` 只给 16（附录 CJ.4）。
    std::vector<float> bv_m(paddedC + 8), sl_m(paddedC + 8);
    float* bv = (float*)(((size_t)bv_m.data() + 31) / 32 * 32);
    float* sl = (float*)(((size_t)sl_m.data() + 31) / 32 * 32);
    for (size_t i = 0; i < in.size(); i++) in[i] = val(1, (int)i);
    for (int k = 0; k < paddedC; k++) {
        bv[k] = (k < C) ? val(3, k) * 0.5f : 0.0f;
        sl[k] = (k < C) ? 0.1f + 0.01f * (k % 7) : 0.0f;
    }

    TEN t;
    if (!t.ChangeSize(N, H, W, C, 0, 0)) return;
    if (!t.ConvertFromCompactNCHW(&in[0], N, C, H, W)) return;
    float* p = t.GetFirstPixelPtr();
    const int ws = t.GetWidthStep(), ss = t.GetSliceStep(), is = t.GetImageStep();

    const Entry& e2 = g_entries[c.entry];
    if (e2.fn9)
        ((FN9)e2.fn9)(p, N, H, W, C, ws, ss, is, has_bias ? bv : sl);
    else
        ((FN10)e2.fn10)(p, N, H, W, C, ws, ss, is, bv, sl);

    // ---- 逐格统计 ----
    long n_ok = 0, n_bad = 0; double worst = 0.0;
    for (int n = 0; n < N; n++)
        for (int ch = 0; ch < C; ch++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++) {
                    double x = in[(((size_t)n * C + ch) * H + h) * W + w];
                    double y = has_bias ? x + bv[ch] : x;
                    if (has_slope) y = (y >= 0) ? y : y * sl[ch];   // max(x, a*x)，a <= 1
                    double got = p[n * is + (ch / A) * ss + h * ws + w * A + (ch % A)];
                    double den = fabs(y) > 1.0 ? fabs(y) : 1.0;
                    double be = fabs(got - y) / den;
                    if (be > TOL) n_bad++; else n_ok++;
                    if (be > worst) worst = be;
                }
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%d %ld %ld %.6e\n", (int)(N * C * H * W), n_ok, n_bad, worst); fclose(f); }
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
    long ok = 0, bad = 0; int tot = 0; double worst = 0;
    int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%d %ld %ld %lf", &tot, &ok, &bad, &worst) == 4); fclose(f); }
    char nm[64], tag[64];
    snprintf(nm, sizeof(nm), "nchwc%d %s", g_entries[c.entry].align, g_kind_name[g_entries[c.entry].kind]);
    snprintf(tag, sizeof(tag), "N=%d %dx%d C=%d", c.N, c.H, c.W, c.C);
    // **结果文件缺失 / 读不出来 = 这个用例没跑完，必须判失败。**
    // ASan 撞上 SEGV 时默认走 Die() -> _exit(1)，**不发信号**，
    // 于是 WIFSIGNALED 为假、退出码也不是 3 —— 缺了这道判断就会把
    // 一个段错误当成"通过"。附录 CJ.4 抓出来的，四个门禁统一补上。
    if (!have) { g_crash++; printf("  没跑完（子进程没写结果文件，退出码 %d）\n", WEXITSTATUS(st)); return; }
    if (WIFSIGNALED(st)) { g_crash++; printf("  %-28s %s  CRASH\n", nm, tag); return; }
    if (bad > 0) { g_bad++; printf("  %-28s %s  FAIL %ld/%ld 格错, 最差 %.3e\n", nm, tag, bad, ok + bad, worst); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC 激活层：5 个动作 x 3 种对齐 = 15 个入口（附录 CG）\n");
    printf("就地运算；内核名全部写全、走函数指针表；用真实 ZQ_CNN_Tensor4D_NCHWC{1,4,8}\n");
    printf("判据：逐元素后向误差（无归约，阈值 1e-6），逐格统计\n");
    printf("W 覆盖 %%4 的四个分支，内核是按 in_W%%4==0/1/2/3 分派实现的\n\n");

    RUNNER r[3] = { &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC1>,
                    &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC4>,
                    &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC8> };
    for (int e = 0; e < N_ENTRY; e++) {
        const int ai = (g_entries[e].align == 1) ? 0 : (g_entries[e].align == 4 ? 1 : 2);
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        // W 取 4 的四种余数，覆盖内核自己的 in_W%4==0/1/2/3 分派
        for (int wsel = 0; wsel < 4; wsel++) {
            Case c; memset(&c, 0, sizeof(c));
            c.entry = e; c.N = 1; c.H = 5; c.W = 12 + wsel; c.C = g_entries[e].align + 2;  // C 不是 align 倍数
            one(c, r[ai]);
        }
        Case c2; memset(&c2, 0, sizeof(c2));
        c2.entry = e; c2.N = 2; c2.H = 3; c2.W = 8; c2.C = g_entries[e].align;          // 整对齐 + N=2
        one(c2, r[ai]);
        printf("  nchwc%d %-18s  %d 个用例：对 %d，错 %d，崩 %d\n",
               g_entries[e].align, g_kind_name[g_entries[e].kind],
               g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    printf("共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
