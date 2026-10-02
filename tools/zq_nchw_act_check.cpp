/* NCHW（layers_c）激活与归一化层门禁 —— 附录 CO
 *
 * 为什么是这一批
 * --------------
 * 附录 CN 查到的那处缺陷（remap 的 SIMD 版横向插值用 sy 而不是 sx），
 * 它的**结构性前置条件**是：同一个功能同时存在「.c 里手写的标量实现」与
 * 「_raw.h 里模板化的 SIMD 实现」，两份一旦分叉就可能一份对一份错。
 * 扫 `layers_c/` 全部 22 个家族，**18 个具备这个结构**（附录 CP）。
 *
 * 这一批是其中语义最确定、参考实现最不容易写错的一批：
 *   relu(6)  relu6(3)  prelu(3)  prelu_sure(3)
 *   addbias_prelu(3)  addbias_prelu_sure(3)  addbias(3)  dropout(3)
 *   softmax(5)  batchnorm_b_a(3)  batchnorm_mean_var(3)  scale(3)
 *   batchnormscale_mean_var_scale_bias(3)
 * 合计 **38 个真实符号**（`nm` 核实）。
 *
 * 语义（逐条从 `layers_c/zq_cnn_*_32f_align_c_raw.h` 读出来的，不是按名字推的）
 * ---------------------------------------------------------------------------
 *  relu(data,…,slope)          就地；`slope == 0` 时 out = max(0,x)，
 *                               否则 out = slope*min(0,x) + max(0,x)
 *                               （raw.h:115 那个 `if (slope == 0)` 是唯一分派）
 *  relu6(data,…)               就地；out = min(6, max(0, x))
 *  prelu(data,…,slope)         就地；out = max(0,x) + slope*min(0,x)
 *  prelu_sure_…                就地；out = max(x, slope*x)  —— slope<=1 时与 prelu 等价
 *  addbias_prelu(…,bias,slope) 就地；out = prelu(x + bias[c], slope[c])
 *  addbias(data,…,bias)        就地；out = x + bias[c]
 *  dropout(data,…,ratio)       就地；`scale = 1-ratio`，
 *                               **`scale == 1.0f` 时直接 return（整段跳过）**，
 *                               否则 out = x * scale
 *  softmax_*_C/H/W             就地，沿指定轴做标准 softmax
 *  batchnorm_b_a(…,b,a)        就地；out = **x*b[c] + a[c]**（b 是乘数、a 是加数，
 *                               与名字的直觉相反 —— 附录 CJ.2 记过一次）
 *  batchnorm_mean_var(…,m,v,e) 就地；out = (x - m[c]) / sqrt(max(v[c]+e, 1e-32))
 *  scale(…,scale,bias)         就地；`bias != NULL` 时 x*scale[c] + bias[c]，
 *                               否则只 x*scale[c]（两个分支都测）
 *  batchnormscale_mean_var_scale_bias(…)
 *                               b[c] = scale[c]/sqrt(max(v[c]+e,1e-32))
 *                               a[c] = bias[c] - m[c]*b[c]，out = x*b + a
 *
 * NCHW 布局：`offset(n,c,h,w) = n*sliceStep + h*widthStep + w*pixelStep + c`
 * （AGENTS.md「NCHW 与 NCHWC 的步长语义」）
 *
 * 沿用 CB~CN：名字写全走函数指针表、逐格统计、每用例 fork 子进程并
 * **显式判"没读到结果文件" = 失败**（CJ.4）。
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "ZQCNN/layers_c/zq_cnn_relu_32f_align_c.h"
#include "ZQCNN/layers_c/zq_cnn_prelu_32f_align_c.h"
#include "ZQCNN/layers_c/zq_cnn_addbias_32f_align_c.h"
#include "ZQCNN/layers_c/zq_cnn_dropout_32f_align_c.h"
#include "ZQCNN/layers_c/zq_cnn_softmax_32f_align_c.h"
#include "ZQCNN/layers_c/zq_cnn_batchnormscale_32f_align_c.h"

typedef void (*F8)(float* d, int N, int H, int W, int C, int ps, int ws, int ss);
typedef void (*F9V)(float* d, int N, int H, int W, int C, int ps, int ws, int ss, float v);
typedef void (*F9P)(float* d, int N, int H, int W, int C, int ps, int ws, int ss, const float* p);
typedef void (*F2P)(float* d, int N, int H, int W, int C, int ps, int ws, int ss,
                    const float* p, const float* q);
typedef void (*F11)(float* d, int N, int H, int W, int C, int ps, int ws, int ss,
                    const float* m, const float* v, float eps);
typedef void (*F13)(float* d, int N, int H, int W, int C, int ps, int ws, int ss,
                    const float* m, const float* v, const float* s, const float* b, float eps);

enum { A_RELU = 0, A_RELU6, A_PRELU, A_PRELU_SURE, A_ADDBIAS_PRELU, A_ADDBIAS_PRELU_SURE,
       A_ADDBIAS, A_DROPOUT, A_SOFTMAX_C, A_SOFTMAX_H, A_SOFTMAX_W,
       A_BN_BA, A_BN_MV, A_SCALE, A_BNS_MVSB, A_COUNT };
static const char* g_name[A_COUNT] = {
    "relu", "relu6", "prelu", "prelu_sure", "addbias_prelu", "addbias_prelu_sure",
    "addbias", "dropout", "softmax_C", "softmax_H", "softmax_W",
    "batchnorm_b_a", "batchnorm_mean_var", "scale", "batchnormscale_mv_sb" };

struct Entry { void* fn; int kind; int align; };

// 38 行平铺，**每个内核名写全**
static const Entry g_entries[39] = {
  { (void*)zq_cnn_relu_32f_align0, A_RELU, 1 },
  { (void*)zq_cnn_relu_32f_align128bit, A_RELU, 4 },
  { (void*)zq_cnn_relu_32f_align256bit, A_RELU, 8 },
  { (void*)zq_cnn_relu6_32f_align0, A_RELU6, 1 },
  { (void*)zq_cnn_relu6_32f_align128bit, A_RELU6, 4 },
  { (void*)zq_cnn_relu6_32f_align256bit, A_RELU6, 8 },
  { (void*)zq_cnn_prelu_32f_align0, A_PRELU, 1 },
  { (void*)zq_cnn_prelu_32f_align128bit, A_PRELU, 4 },
  { (void*)zq_cnn_prelu_32f_align256bit, A_PRELU, 8 },
  { (void*)zq_cnn_prelu_32f_align128bit_sure_slope_lessthan1, A_PRELU_SURE, 4 },
  { (void*)zq_cnn_prelu_32f_align256bit_sure_slope_lessthan1, A_PRELU_SURE, 8 },
  { (void*)zq_cnn_addbias_prelu_32f_align0, A_ADDBIAS_PRELU, 1 },
  { (void*)zq_cnn_addbias_prelu_32f_align128bit, A_ADDBIAS_PRELU, 4 },
  { (void*)zq_cnn_addbias_prelu_32f_align256bit, A_ADDBIAS_PRELU, 8 },
  { (void*)zq_cnn_addbias_prelu_32f_align128bit_sure_slope_lessthan1, A_ADDBIAS_PRELU_SURE, 4 },
  { (void*)zq_cnn_addbias_prelu_32f_align256bit_sure_slope_lessthan1, A_ADDBIAS_PRELU_SURE, 8 },
  { (void*)zq_cnn_addbias_32f_align0, A_ADDBIAS, 1 },
  { (void*)zq_cnn_addbias_32f_align128bit, A_ADDBIAS, 4 },
  { (void*)zq_cnn_addbias_32f_align256bit, A_ADDBIAS, 8 },
  { (void*)zq_cnn_dropout_32f_align0, A_DROPOUT, 1 },
  { (void*)zq_cnn_dropout_32f_align128bit, A_DROPOUT, 4 },
  { (void*)zq_cnn_dropout_32f_align256bit, A_DROPOUT, 8 },
  { (void*)zq_cnn_softmax_32f_align0_C, A_SOFTMAX_C, 1 },
  { (void*)zq_cnn_softmax_32f_align0_H, A_SOFTMAX_H, 1 },
  { (void*)zq_cnn_softmax_32f_align0_W, A_SOFTMAX_W, 1 },
  { (void*)zq_cnn_softmax_32f_align128bit_C, A_SOFTMAX_C, 4 },
  { (void*)zq_cnn_softmax_32f_align256bit_C, A_SOFTMAX_C, 8 },
  { (void*)zq_cnn_batchnorm_32f_b_a_align0, A_BN_BA, 1 },
  { (void*)zq_cnn_batchnorm_32f_b_a_align128bit, A_BN_BA, 4 },
  { (void*)zq_cnn_batchnorm_32f_b_a_align256bit, A_BN_BA, 8 },
  { (void*)zq_cnn_batchnorm_32f_mean_var_align0, A_BN_MV, 1 },
  { (void*)zq_cnn_batchnorm_32f_mean_var_align128bit, A_BN_MV, 4 },
  { (void*)zq_cnn_batchnorm_32f_mean_var_align256bit, A_BN_MV, 8 },
  { (void*)zq_cnn_scale_32f_align0, A_SCALE, 1 },
  { (void*)zq_cnn_scale_32f_align128bit, A_SCALE, 4 },
  { (void*)zq_cnn_scale_32f_align256bit, A_SCALE, 8 },
  { (void*)zq_cnn_batchnormscale_32f_mean_var_scale_bias_align0, A_BNS_MVSB, 1 },
  { (void*)zq_cnn_batchnormscale_32f_mean_var_scale_bias_align128bit, A_BNS_MVSB, 4 },
  { (void*)zq_cnn_batchnormscale_32f_mean_var_scale_bias_align256bit, A_BNS_MVSB, 8 },
};
static const int N_ENTRY = 39;

#define RES_FILE "/tmp/zq_nact_res.txt"
static const double TOL = 1e-5;
static const double FLOAT_EPS_FOR_DIV = 1e-32;

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

struct Case { int entry, C, variant; };

static void run_one(const Case& c)
{
    const Entry& e = g_entries[c.entry];
    const int A = e.align;
    const int N = 2, H = 5, W = 7, C = c.C;
    // NCHW 的 pixelStep **就是 C**（C 是通道数，没有 NCHWC 那种"补齐到 align"）。
    // 第一版写成 ps = A，于是只有 C==A 的用例恰好对上，其余全红 —— 又是一次假红
    // （附录 CE.9 / CN.3 记过同一类：形状给错，红是假的）。
    const int ps = C, ws = ps * W, ss = ws * H;
    const float eps = 1e-5f;

    // 逐通道数组按**补齐后的通道数**开，且 32 字节对齐
    // （长度：CG.4 的教训；对齐：CJ.4 的教训，align=8 用的是 _mm256_load_ps）
    const int paddedC = (C + A - 1) / A * A;
    std::vector<float> m1(paddedC + 8), m2(paddedC + 8), m3(paddedC + 8);
    float* pv = (float*)(((size_t)m1.data() + 31) / 32 * 32);
    float* qv = (float*)(((size_t)m2.data() + 31) / 32 * 32);
    float* rv = (float*)(((size_t)m3.data() + 31) / 32 * 32);
    for (int k = 0; k < paddedC; k++) {
        if (k < C) {
            pv[k] = val(3, k) * 0.5f;                 // slope / bias / b
            qv[k] = 0.1f + 0.01f * (k % 7);          // bias / a
            rv[k] = 0.5f + 0.01f * (k % 17);         // var
        } else { pv[k] = 0; qv[k] = 0; rv[k] = 1; }
    }

    // **输入也按 stride 布局填**，于是"参考值"与"内核输出"用的是**同一套下标**。
    // 第一版把输入按紧凑 (n,c,h,w) 填、参考值用紧凑式、got 用 stride 式 ——
    // 两套下标混用，NCHW 这边 ps==C 但 w 与 c 的位置不同，错位 130 个用例全红
    // （连 relu 这种纯拷贝都错，本身就说明是门禁的错）。
    // 教训与 CN.6 的"无效用例"同源：**判据用的下标必须和被测方用的下标是同一个。**
    const size_t n_elem = (size_t)N * C * H * W;
    std::vector<float> in(n_elem);
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    in[(size_t)n * ss + h * ws + w * ps + c] = val(1, (int)(((size_t)n * C + c) * H + h) * W + w);

    // 主数据缓冲区**同样要 32 字节对齐** —— align=8 那一族用的是
    // `zq_mm_load_ps` = `_mm256_load_ps`，它要求 32 字节。
    // `std::vector<float>` 只给 16，第一版直接 `&buf[0]` 当成
    // "逐元素运算对齐无所谓"，于是 align=8 全线错/崩。
    // （附录 CJ.4 已经把这条写进 AGENTS.md 了，这里又踩了一次 ——
    //   当时给**逐通道数组**做了对齐，忘了**主缓冲区**。）
    std::vector<float> buf_store(n_elem + 8);
    float* d = (float*)(((size_t)buf_store.data() + 31) / 32 * 32);
    for (size_t i = 0; i < n_elem; i++) d[i] = in[i];   // 就地运算，in 留作参考

    const float relu_slope = (c.variant == 0) ? 0.0f : 0.125f;   // slope==0 与 slope!=0 两条分支
    const float drop_ratio  = (c.variant == 0) ? 0.0f : 0.25f;   // scale==1 提前返回 与 正常缩放
    const bool   no_bias    = (c.variant == 1) && (e.kind == A_SCALE);   // scale 的 bias==NULL 分支

    switch (e.kind) {
    case A_RELU:   ((F9V)e.fn)(d, N, H, W, C, ps, ws, ss, relu_slope); break;
    case A_RELU6:  ((F8)e.fn)(d, N, H, W, C, ps, ws, ss); break;
    case A_PRELU:
    case A_PRELU_SURE: ((F9P)e.fn)(d, N, H, W, C, ps, ws, ss, pv); break;
    case A_ADDBIAS_PRELU:
    case A_ADDBIAS_PRELU_SURE: ((F2P)e.fn)(d, N, H, W, C, ps, ws, ss, pv, qv); break;
    case A_ADDBIAS: ((F9P)e.fn)(d, N, H, W, C, ps, ws, ss, pv); break;
    case A_DROPOUT: ((F9V)e.fn)(d, N, H, W, C, ps, ws, ss, drop_ratio); break;
    case A_SOFTMAX_C: case A_SOFTMAX_H: case A_SOFTMAX_W:
        ((F8)e.fn)(d, N, H, W, C, ps, ws, ss); break;
    case A_BN_BA:  ((F2P)e.fn)(d, N, H, W, C, ps, ws, ss, pv, qv); break;
    case A_BN_MV:  ((F11)e.fn)(d, N, H, W, C, ps, ws, ss, pv, rv, eps); break;
    case A_SCALE:  ((F2P)e.fn)(d, N, H, W, C, ps, ws, ss, pv, no_bias ? 0 : qv); break;
    default:       ((F13)e.fn)(d, N, H, W, C, ps, ws, ss, pv, rv, pv, qv, eps); break;
    }

    // ---- 参考 ----
    long n_ok = 0, n_bad = 0; double worst = 0.0;
    const int AX = (e.kind == A_SOFTMAX_C) ? C : (e.kind == A_SOFTMAX_H ? H : W);
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++) {
                    const double x = in[(size_t)n * ss + h * ws + w * ps + c];   // 同一套下标
                    const double got = d[((size_t)n * ss + h * ws + w * ps) + c];
                    double y;
                    switch (e.kind) {
                    case A_RELU: y = (relu_slope == 0) ? (x > 0 ? x : 0.0)
                                  : (x > 0 ? x : relu_slope * x); break;
                    case A_RELU6: { double t = x > 0 ? x : 0.0; y = t > 6.0 ? 6.0 : t; break; }
                    case A_PRELU: y = (x > 0) ? x : pv[c] * x; break;
                    case A_PRELU_SURE: y = (x > pv[c] * x) ? x : pv[c] * x; break;
                    case A_ADDBIAS_PRELU: case A_ADDBIAS_PRELU_SURE: {
                        double t = x + pv[c];
                        y = (t > 0) ? t : qv[c] * t; break; }
                    case A_ADDBIAS: y = x + pv[c]; break;
                    case A_DROPOUT: y = (drop_ratio == 0) ? x : x * (1.0f - drop_ratio); break;
                    case A_BN_BA: y = x * pv[c] + qv[c]; break;
                    case A_BN_MV: y = (x - pv[c]) / sqrt(rv[c] + eps); break;
                    case A_SCALE: y = no_bias ? x * pv[c] : x * pv[c] + qv[c]; break;
                    case A_BNS_MVSB: {
                        double s = sqrt(rv[c] + eps);
                        if (s < FLOAT_EPS_FOR_DIV) s = FLOAT_EPS_FOR_DIV;
                        y = (x - pv[c]) * pv[c] / s + qv[c]; break; }
                    default: {   // softmax：沿 AX 做
                        int AL = AX;
                        double v[64];
                        if (AL > 64) return;
                        double mx = -1e300, sum = 0.0;
                        for (int i = 0; i < AL; i++) {
                            int ic = c, ih = h, iw = w;
                            if (e.kind == A_SOFTMAX_C)      { ic = i; ih = h; iw = w; }
                            else if (e.kind == A_SOFTMAX_H) { ic = c; ih = i; iw = w; }
                            else                           { ic = c; ih = h; iw = i; }
                            v[i] = in[(size_t)n * ss + ih * ws + iw * ps + ic];   // 同一套下标
                            if (v[i] > mx) mx = v[i];
                        }
                        for (int i = 0; i < AL; i++) { v[i] = exp(v[i] - mx); sum += v[i]; }
                        y = (e.kind == A_SOFTMAX_C) ? v[c] / sum
                          : (e.kind == A_SOFTMAX_H) ? v[h] / sum : v[w] / sum;
                        break; }
                    }
                    double den = (fabs(y) > 1.0) ? fabs(y) : 1.0;
                    if (e.kind == A_BN_MV || e.kind == A_BNS_MVSB) den *= sqrt(rv[c] + eps);
                    double be = fabs(got - y) / den;
                    if (be > TOL) n_bad++; else n_ok++;
                    if (be > worst) worst = be;
                }
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %.6e\n", n_ok, n_bad, worst); fclose(f); }
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(const Case& c)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) {
        FILE* dn = freopen("/dev/null", "w", stderr); (void)dn;
        run_one(c);
        _exit(0);
    }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0; double worst = 0;
    int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%ld %ld %lf", &ok, &bad, &worst) == 3); fclose(f); }
    char nm[48], tag[40];
    snprintf(nm, sizeof(nm), "align%d %s", g_entries[c.entry].align, g_name[g_entries[c.entry].kind]);
    snprintf(tag, sizeof(tag), "C=%d %s", c.C, c.variant ? "变体1" : "变体0");
    if (!have) { g_crash++; printf("  %-34s %-16s  没跑完（退出码 %d）\n", nm, tag, WEXITSTATUS(st)); return; }
    if (WIFSIGNALED(st)) { g_crash++; printf("  %-34s %-16s  CRASH(信号 %d)\n", nm, tag, WTERMSIG(st)); return; }
    if (bad > 0) { g_bad++; printf("  %-34s %-16s  FAIL %ld/%ld 格错, 最差 %.3e\n", nm, tag, bad, ok + bad, worst); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHW 激活与归一化：39 个真实符号（nm 核实）\n");
    printf("  relu / relu6 / prelu(+sure) / addbias_prelu(+sure) / addbias / dropout\n");
    printf("  softmax(_C/_H/_W) / batchnorm_b_a / batchnorm_mean_var / scale / batchnormscale_mv_sb\n");
    printf("内核名全部写全、走函数指针表；判据：逐元素后向误差，逐格统计\n");
    printf("带分支的都跑两个分支：relu 的 slope==0、dropout 的 scale==1 提前返回、scale 的 bias==NULL\n");
    printf("逐通道数组按补齐后长度开、且 32 字节对齐（附录 CG.4 / CJ.4 的两条教训）\n\n");

    for (int e = 0; e < N_ENTRY; e++) {
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        const int A = g_entries[e].align;
        // 两个维度分开跑：形状（C=A / C=2A）与分支（relu 的 slope==0、
        // dropout 的 scale==1 提前返回、scale 的 bias==NULL）。
        // **C 必须是 A 的倍数**：NCHW 的 pixelStep == C（C 就是通道数，没有补齐），
        // 给 ps=A 配 C=A+2 是**非法张量形状**，内核跨像素读写，出来的红是假的
        // —— 附录 CE.9 记过同一个坑。
        for (int cs = 0; cs < 2; cs++)
            for (int v = 0; v < 2; v++) {
                Case c; memset(&c, 0, sizeof(c));
                c.entry = e; c.variant = v;
                c.C = (cs == 0) ? A : A * 2;
                one(c);
            }
        printf("  align%d %-24s  %d 个用例：对 %d，错 %d，崩 %d\n",
               A, g_name[g_entries[e].kind], g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
