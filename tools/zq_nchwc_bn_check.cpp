/* NCHWC batchnormscale 门禁 —— 附录 CJ
 *
 * 覆盖面：scale / batchnorm_b_a / batchnorm_mean_var /
 *         batchnormscale_mean_var_scale_bias 各 × NCHWC1/4/8 = **12 个入口**
 *
 * 语义（逐条从 `zq_cnn_batchnormscale_nchwc_raw.h` 抄下来的，**参数名与直觉相反**）
 * ---------------------------------------------------------------------------
 *  scale_nchwc(data,…, scale, bias)          就地
 *      bias != NULL  ->  out = x*scale[c] + bias[c]
 *      bias == NULL  ->  out = x*scale[c]                （两个分支都要测）
 *
 *  batchnorm_b_a_nchwc(data,…, b_data, a_data)         就地
 *      out = x*b_data[c] + a_data[c]
 *      **b 乘、a 加** —— 参数名 b 在前、a 在后，代码是 fmadd(x, b_vec, a_vec)，
 *      也就是**第一个参数是乘数、第二个是加数**，与名字的直觉相反
 *
 *  batchnorm_mean_var_nchwc(data,…, mean, var, eps)    就地
 *      b[c] = 1 / sqrt(max(var[c] + eps, 1e-32))
 *      a[c] = -mean[c] * b[c]
 *      out = x*b + a = (x - mean[c]) / sqrt(var[c]+eps)
 *      **它算完 a/b 之后直接调 batchnorm_b_a_nchwc**（raw.h:123），
 *      所以 b_a 的任何问题会在两处一起出现
 *
 *  batchnormscale_mean_var_scale_bias_nchwc(data,…, mean, var, scale, bias, eps) 就地
 *      b[c] = scale[c] / sqrt(max(var[c] + eps, 1e-32))
 *      a[c] = bias[c] - mean[c] * b[c]
 *      out = x*b + a = (x - mean[c])*scale[c]/sqrt(var[c]+eps) + bias[c]
 *
 * `FLOAT_EPS_FOR_DIV = 1e-32`（zq_cnn_batchnormscale_nchwc.c:40）。
 * 用例里的 var 取正常正数，那道钳位不会触发。
 *
 * 沿用 CB/CE/CF/CG/CH/CI：名字写全走函数指针表、后向误差 + 逐格统计、
 * 每用例 fork 子进程、用真实的 ZQ_CNN_Tensor4D_NCHWC{1,4,8}。
 * **逐通道数组一律按 paddedC = ceil(C/align)*align 开**（CG.4 的教训）。
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h"
#include "ZQCNN/layers_nchwc/zq_cnn_batchnormscale_nchwc.h"

typedef void (*FN_SCALE)(float* d, int N, int H, int W, int C,
                         int ws, int ss, int is, const float* scale, const float* bias);
typedef void (*FN_BA)(float* d, int N, int H, int W, int C,
                      int ws, int ss, int is, const float* b, const float* a);
typedef void (*FN_MV)(float* d, int N, int H, int W, int C,
                      int ws, int ss, int is, const float* mean, const float* var, float eps);
typedef void (*FN_MVSB)(float* d, int N, int H, int W, int C,
                        int ws, int ss, int is, const float* mean, const float* var,
                        const float* scale, const float* bias, float eps);

enum { B_SCALE = 0, B_BA = 1, B_MV = 2, B_MVSB = 3 };
static const char* g_kind_name[4] = { "scale", "batchnorm_b_a", "batchnorm_mean_var",
                                      "batchnormscale_mean_var_scale_bias" };

struct Entry { void* fn; int kind; int align; };

// 12 行平铺，**每个内核名写全**（不拼接）
static const Entry g_entries[12] = {
  { (void*)zq_cnn_scale_nchwc1, B_SCALE, 1 },
  { (void*)zq_cnn_scale_nchwc4, B_SCALE, 4 },
  { (void*)zq_cnn_scale_nchwc8, B_SCALE, 8 },
  { (void*)zq_cnn_batchnorm_b_a_nchwc1, B_BA, 1 },
  { (void*)zq_cnn_batchnorm_b_a_nchwc4, B_BA, 4 },
  { (void*)zq_cnn_batchnorm_b_a_nchwc8, B_BA, 8 },
  { (void*)zq_cnn_batchnorm_mean_var_nchwc1, B_MV, 1 },
  { (void*)zq_cnn_batchnorm_mean_var_nchwc4, B_MV, 4 },
  { (void*)zq_cnn_batchnorm_mean_var_nchwc8, B_MV, 8 },
  { (void*)zq_cnn_batchnormscale_mean_var_scale_bias_nchwc1, B_MVSB, 1 },
  { (void*)zq_cnn_batchnormscale_mean_var_scale_bias_nchwc4, B_MVSB, 4 },
  { (void*)zq_cnn_batchnormscale_mean_var_scale_bias_nchwc8, B_MVSB, 8 },
};
static const int N_ENTRY = 12;

#define RES_FILE "/tmp/zq_bn_res.txt"
static const double TOL = 1e-5;
static const double FLOAT_EPS_FOR_DIV = 1e-32;

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

struct Case { int entry, N, H, W, C, no_bias; };

typedef void (*RUNNER)(const Case&);
template <class TEN>
static void run_one(const Case& c)
{
    const Entry& e = g_entries[c.entry];
    const int A = TEN().GetAlignSize();
    const int N = c.N, H = c.H, W = c.W, C = c.C;
    const float eps = 1e-5f;

    // 逐通道数组按**补齐后的通道数**开（CG.4 的教训：按 align 开会让越界的组读到 0）
    //
    // 而且必须**32 字节对齐**：align=8 那一族用的是 `zq_mm_load_ps` = `_mm256_load_ps`，
    // 它要求 32 字节对齐，`std::vector<float>` 只给 16 —— 对齐不满足直接 SIGSEGV。
    // （AGENTS.md「ZQ_GEMM 的调用方契约」第 2 条就是这条；
    //   第一版探针就是栽在这里，报出来的是 SEGV 而不是数值错。）
    // 手法：多分配 8 个 float 用来把首地址推到 32 的倍数上。
    const int paddedC = (C + A - 1) / A * A;
    std::vector<float> mean_m(paddedC + 8), var_m(paddedC + 8), sc_m(paddedC + 8), bi_m(paddedC + 8);
    float* mean_v = (float*)(((size_t)mean_m.data() + 31) / 32 * 32);
    float* var_v  = (float*)(((size_t)var_m.data()  + 31) / 32 * 32);
    float* sc_v   = (float*)(((size_t)sc_m.data()   + 31) / 32 * 32);
    float* bi_v   = (float*)(((size_t)bi_m.data()   + 31) / 32 * 32);
    for (int k = 0; k < paddedC; k++) {
        if (k < C) {
            mean_v[k] = val(3, k) * 0.5f;
            var_v[k]  = 0.5f + 0.01f * (k % 17);      // 正常正数，FLOAT_EPS_FOR_DIV 钳位不触发
            sc_v[k]   = 0.8f + 0.02f * (k % 11);
            bi_v[k]   = val(4, k) * 0.25f;
        } else { mean_v[k] = 0; var_v[k] = 1; sc_v[k] = 1; bi_v[k] = 0; }
    }

    std::vector<float> in((size_t)N * C * H * W);
    for (size_t i = 0; i < in.size(); i++) in[i] = val(1, (int)i);
    TEN t;
    if (!t.ChangeSize(N, H, W, C, 0, 0)) return;
    if (!t.ConvertFromCompactNCHW(&in[0], N, C, H, W)) return;
    float* p = t.GetFirstPixelPtr();
    const int ws = t.GetWidthStep(), ss = t.GetSliceStep(), is = t.GetImageStep();

    switch (e.kind) {
    case B_SCALE:
        ((FN_SCALE)e.fn)(p, N, H, W, C, ws, ss, is, &sc_v[0], c.no_bias ? 0 : &bi_v[0]);
        break;
    case B_BA:
        // b 是**乘数**、a 是**加数**（与参数名的直觉相反，见文件头注释）
        ((FN_BA)e.fn)(p, N, H, W, C, ws, ss, is, &sc_v[0], &bi_v[0]);
        break;
    case B_MV:
        ((FN_MV)e.fn)(p, N, H, W, C, ws, ss, is, &mean_v[0], &var_v[0], eps);
        break;
    default:
        ((FN_MVSB)e.fn)(p, N, H, W, C, ws, ss, is, &mean_v[0], &var_v[0], &sc_v[0], &bi_v[0], eps);
        break;
    }

    long n_ok = 0, n_bad = 0; double worst = 0.0;
    for (int n = 0; n < N; n++)
        for (int ch = 0; ch < C; ch++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++) {
                    double x = in[(((size_t)n * C + ch) * H + h) * W + w];
                    double y;
                    if (e.kind == B_SCALE)      y = c.no_bias ? x * sc_v[ch] : x * sc_v[ch] + bi_v[ch];
                    else if (e.kind == B_BA)   y = x * sc_v[ch] + bi_v[ch];
                    else if (e.kind == B_MV)   y = (x - mean_v[ch]) / sqrt(var_v[ch] + eps);
                    else {
                        double s = sqrt(var_v[ch] + eps);
                        if (s < FLOAT_EPS_FOR_DIV) s = FLOAT_EPS_FOR_DIV;
                        y = (x - mean_v[ch]) * sc_v[ch] / s + bi_v[ch];
                    }
                    double got = p[n * is + (ch / A) * ss + h * ws + w * A + (ch % A)];
                    // 尺度跟着 scale 走，用它归一化才有意义
                    double den = fabs(y) > 1.0 ? fabs(y) : 1.0;
                    if (e.kind == B_MV) { double s2 = sqrt(var_v[ch] + eps); den *= s2; }
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
        FILE* dn = freopen("/dev/null", "w", stderr); (void)dn;
        r(c);
        _exit(0);
    }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0; double worst = 0;
    int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%ld %ld %lf", &ok, &bad, &worst) == 3); fclose(f); }
    char nm[80], tag[80];
    snprintf(nm, sizeof(nm), "nchwc%d %s", g_entries[c.entry].align, g_kind_name[g_entries[c.entry].kind]);
    snprintf(tag, sizeof(tag), "N=%d %dx%d C=%d%s", c.N, c.H, c.W, c.C,
             (g_entries[c.entry].kind == B_SCALE && c.no_bias) ? " (无 bias 分支)" : "");
    if (WIFSIGNALED(st)) { g_crash++; printf("  %-42s %s  CRASH(信号 %d)\n", nm, tag, WTERMSIG(st)); return; }
    // **结果文件缺失 / 读不出来 = 这个用例没跑完，必须判失败。**
    // ASan 撞上 SEGV 时默认走 `Die()` -> `_exit(1)`，**不发信号**，
    // 于是 WIFSIGNALED 为假、退出码 1 也不是 3 —— 第一版就掉进这个洞，
    // 把一个段错误当成了"通过"（附录 CJ.4）。这里显式判 `have`。
    if (!have) { g_crash++; printf("  %-42s %s  没跑完（退出码 %d%s）\n", nm, tag,
                                   WEXITSTATUS(st), WIFSIGNALED(st) ? "" : "，ASan 撞 SEGV 时就是这个"); return; }
    if (bad > 0) { g_bad++; printf("  %-42s %s  FAIL %ld/%ld 格错, 最差 %.3e\n", nm, tag, bad, ok + bad, worst); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC batchnormscale：4 个动作 x 3 种对齐 = 12 个入口（附录 CJ）\n");
    printf("内核名全部写全、走函数指针表；用真实 ZQ_CNN_Tensor4D_NCHWC{1,4,8}\n");
    printf("判据：逐元素后向误差（无归约），逐格统计\n");
    printf("**batchnorm_b_a 的第一个参数是乘数、第二个是加数**（与名字直觉相反）\n");
    printf("scale 的 bias==NULL 与 bias!=NULL 两个分支都测\n\n");

    RUNNER r[3] = { &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC1>,
                    &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC4>,
                    &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC8> };
    for (int e = 0; e < N_ENTRY; e++) {
        const int ai = (g_entries[e].align == 1) ? 0 : (g_entries[e].align == 4 ? 1 : 2);
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        const int A = g_entries[e].align;
        if (g_entries[e].kind == B_SCALE) {
            for (int nb = 0; nb < 2; nb++) {          // bias == NULL / != NULL
                for (int j = 0; j < 2; j++) {
                    Case c; memset(&c, 0, sizeof(c));
                    c.entry = e; c.N = (j ? 2 : 1); c.H = 3; c.W = 5;
                    c.C = (j ? A : A + 2); c.no_bias = nb;
                    one(c, r[ai]);
                }
            }
        } else {
            for (int j = 0; j < 2; j++) {
                Case c; memset(&c, 0, sizeof(c));
                c.entry = e; c.N = (j ? 2 : 1); c.H = 3; c.W = 5;
                c.C = (j ? A : A + 2);
                one(c, r[ai]);
            }
        }
        printf("  nchwc%d %-38s  %d 个用例：对 %d，错 %d，崩 %d\n",
               A, g_kind_name[g_entries[e].kind], g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
