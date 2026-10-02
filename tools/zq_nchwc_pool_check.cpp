/* NCHWC pooling 门禁 —— 附录 CI
 *
 * 为什么要有这个门禁
 * ------------------
 * pooling 是每个 CNN 都会用到的层，而 NCHWC 这一族（24 个入口）之前
 * **一道门禁都没有**（NCHW 那一族有 `zq_pool_check`）。附录 CE/CF/CG/CH
 * 连着四轮都印证了同一件事：**同一个功能的两个变体，测了的那份全对、
 * 没测的那份全错**。
 *
 * 覆盖面：24 个入口（`nm` 核实，不是按头里的声明数 —— 头里有重复声明）
 *   avg/max pooling × nodivided/suredivided × nchwc1/4/8
 *   suredivided 另有 general / kernel2x2 / kernel3x3 三种
 *
 * 语义（逐条从 `zq_cnn_pooling_nchwc_raw.h` 读出来的，名字有歧义，代码没有）
 * ------------------------------------------------------------------
 *   `final_kH = min(kernel_H, in_H - (out_H-1)*stride_H)`
 *   `final_kW = min(kernel_W, in_W - (out_W-1)*stride_W)`
 *
 *   nodivided_general —— **有**四条边界分支（行 525 起那个 avg 版本）：
 *       内部    除以 kernel_H*kernel_W
 *       末列    除以 kernel_H*final_kW
 *       末行    除以 final_kH*kernel_W
 *       角上    除以 final_kH*final_kW
 *   suredivided_* —— **没有**边界分支（行 360 那个 avg 版本只有一个循环）：
 *       一律除以 kernel_H*kernel_W
 *       **契约**：每个窗口都必须放得下，即
 *                 (in_H - kernel_H) % stride_H == 0 且 out_H = (in_H-kernel_H)/stride_H + 1
 *                 否则会越界读。分派器用
 *                 `suredivided = (in_H+pad-kernel_H)%stride_H==0 && (…)` 保证这一点
 *                 （ZQ_CNN_Forward_SSEUtils.h:1604）
 *
 *   `suredivided_kernel2x2` / `_kernel3x3` —— **把 2×2 / 3×3 写死了**
 *       （每行固定两次/三次 load、行数也固定），却仍用 `1/(kernel_H*kernel_W)` 做除数。
 *       传别的 kernel 会静默算错。**分派器有守卫**：
 *       `if (kernel_H==2 && kernel_W==2) … else if (3&&3) … else general`
 *       （ZQ_CNN_Forward_SSEUtils_NCHWC.cpp:3638）—— 与附录 CC.5 的
 *       `same_pixstep_kernel1x1` 同一形态：契约在调用侧强制，内核内部不重复校验。
 *       所以**本门禁也必须按契约喂**，见下面每入口的形状。
 *
 *   max pooling 不做除法，所以 nodivided / suredivided 的区别**只在边界处理**。
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h"
#include "ZQCNN/layers_nchwc/zq_cnn_pooling_nchwc.h"

typedef void (*FN)(const float* in_data, int in_N, int in_H, int in_W, int in_C,
                   int in_ws, int in_ss, int in_is,
                   int kernel_H, int kernel_W, int stride_H, int stride_W,
                   float* out_data, int out_N, int out_H, int out_W, int out_C,
                   int out_ws, int out_ss, int out_is);

enum { P_AVG = 0, P_MAX = 1 };
static const char* g_pool_name[2] = { "avg", "max" };

struct Entry { FN fn; int pool; int divided; int align; int fixk; };
// fixk: 0 = general（kernel 由用例给），2 = 写死 2x2，3 = 写死 3x3

// 24 行平铺，**每个内核名写全**
static const Entry g_entries[24] = {
  { zq_cnn_avgpooling_nopadding_nodivided_nchwc1_general,  P_AVG, 0, 1, 0 },
  { zq_cnn_avgpooling_nopadding_nodivided_nchwc4_general,  P_AVG, 0, 4, 0 },
  { zq_cnn_avgpooling_nopadding_nodivided_nchwc8_general,  P_AVG, 0, 8, 0 },

  { zq_cnn_avgpooling_nopadding_suredivided_nchwc1_general,  P_AVG, 1, 1, 0 },
  { zq_cnn_avgpooling_nopadding_suredivided_nchwc1_kernel2x2, P_AVG, 1, 1, 2 },
  { zq_cnn_avgpooling_nopadding_suredivided_nchwc1_kernel3x3, P_AVG, 1, 1, 3 },
  { zq_cnn_avgpooling_nopadding_suredivided_nchwc4_general,  P_AVG, 1, 4, 0 },
  { zq_cnn_avgpooling_nopadding_suredivided_nchwc4_kernel2x2, P_AVG, 1, 4, 2 },
  { zq_cnn_avgpooling_nopadding_suredivided_nchwc4_kernel3x3, P_AVG, 1, 4, 3 },
  { zq_cnn_avgpooling_nopadding_suredivided_nchwc8_general,  P_AVG, 1, 8, 0 },
  { zq_cnn_avgpooling_nopadding_suredivided_nchwc8_kernel2x2, P_AVG, 1, 8, 2 },
  { zq_cnn_avgpooling_nopadding_suredivided_nchwc8_kernel3x3, P_AVG, 1, 8, 3 },

  { zq_cnn_maxpooling_nopadding_nodivided_nchwc1_general,  P_MAX, 0, 1, 0 },
  { zq_cnn_maxpooling_nopadding_nodivided_nchwc4_general,  P_MAX, 0, 4, 0 },
  { zq_cnn_maxpooling_nopadding_nodivided_nchwc8_general,  P_MAX, 0, 8, 0 },

  { zq_cnn_maxpooling_nopadding_suredivided_nchwc1_general,  P_MAX, 1, 1, 0 },
  { zq_cnn_maxpooling_nopadding_suredivided_nchwc1_kernel2x2, P_MAX, 1, 1, 2 },
  { zq_cnn_maxpooling_nopadding_suredivided_nchwc1_kernel3x3, P_MAX, 1, 1, 3 },
  { zq_cnn_maxpooling_nopadding_suredivided_nchwc4_general,  P_MAX, 1, 4, 0 },
  { zq_cnn_maxpooling_nopadding_suredivided_nchwc4_kernel2x2, P_MAX, 1, 4, 2 },
  { zq_cnn_maxpooling_nopadding_suredivided_nchwc4_kernel3x3, P_MAX, 1, 4, 3 },
  { zq_cnn_maxpooling_nopadding_suredivided_nchwc8_general,  P_MAX, 1, 8, 0 },
  { zq_cnn_maxpooling_nopadding_suredivided_nchwc8_kernel2x2, P_MAX, 1, 8, 2 },
  { zq_cnn_maxpooling_nopadding_suredivided_nchwc8_kernel3x3, P_MAX, 1, 8, 3 },
};
static const int N_ENTRY = 24;

#define RES_FILE "/tmp/zq_pool8_res.txt"
static const double TOL = 1e-5;

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

struct Case { int entry, N, H, W, C, kH, kW, S; };

typedef void (*RUNNER)(const Case&);
template <class TEN>
static void run_one(const Case& c)
{
    const Entry& e = g_entries[c.entry];
    const int A = TEN().GetAlignSize();
    const int N = c.N, H = c.H, W = c.W, C = c.C, kH = c.kH, kW = c.kW, S = c.S;

    // out 的尺寸**由本入口的契约决定**，不是随手给的：
    //   suredivided -> 每个窗口都必须放得下，out = (in-k)/S + 1 且 (in-k)%S==0
    //   nodivided  -> 用 ceil 形状，**刻意让最后一个窗口被裁**，从而走 final_kH/kW 那几条分支
    int oH, oW;
    if (e.divided) { oH = (H - kH) / S + 1; oW = (W - kW) / S + 1; }
    else { oH = (H - kH + S - 1) / S + 1; oW = (W - kW + S - 1) / S + 1; }
    if (oH <= 0 || oW <= 0) return;

    std::vector<float> in((size_t)N * C * H * W);
    for (size_t i = 0; i < in.size(); i++) in[i] = val(1, (int)i);

    TEN ti, to;
    if (!ti.ChangeSize(N, H, W, C, 0, 0)) return;
    if (!to.ChangeSize(N, oH, oW, C, 0, 0)) return;
    if (!ti.ConvertFromCompactNCHW(&in[0], N, C, H, W)) return;

    ((FN)e.fn)(ti.GetFirstPixelPtr(), N, H, W, C,
               ti.GetWidthStep(), ti.GetSliceStep(), ti.GetImageStep(),
               kH, kW, S, S,
               to.GetFirstPixelPtr(), N, oH, oW, C,
               to.GetWidthStep(), to.GetSliceStep(), to.GetImageStep());

    const int ows = to.GetWidthStep(), oss = to.GetSliceStep(), ois = to.GetImageStep();
    long n_ok = 0, n_bad = 0; double worst = 0.0;
    for (int n = 0; n < N; n++)
        for (int ch = 0; ch < C; ch++)
            for (int oh = 0; oh < oH; oh++)
                for (int ow = 0; ow < oW; ow++) {
                    // nodivided 在边界收窄窗口；suredivided 一律用满 kernel_H*kernel_W
                    const int fh = e.divided ? kH : (kH < H - oh * S ? kH : H - oh * S);
                    const int fw = e.divided ? kW : (kW < W - ow * S ? kW : W - ow * S);
                    double sum = 0.0, best = 0.0, sc = 0.0;
                    bool first = true;
                    for (int a = 0; a < fh; a++)
                        for (int b = 0; b < fw; b++) {
                            double v = in[(((size_t)n * C + ch) * H + (oh * S + a)) * W + (ow * S + b)];
                            sum += v; sc += v * v;
                            if (first || v > best) { best = v; first = false; }
                        }
                    double y;
                    if (e.pool == P_MAX) y = best;
                    else y = sum / (double)(e.divided ? (kH * kW) : (fh * fw));
                    double got = to.GetFirstPixelPtr()[n * ois + (ch / A) * oss
                                                     + oh * ows + ow * A + (ch % A)];
                    double den = sqrt(sc); if (den < 1e-30) den = 1.0;
                    double be = fabs(got - y) / den;
                    if (be > TOL) n_bad++; else n_ok++;
                    if (be > worst) worst = be;
                }
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%d %d %ld %ld %.6e\n", oH, oW, n_ok, n_bad, worst); fclose(f); }
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
    long ok = 0, bad = 0; int oh = 0, ow = 0; double worst = 0;
    int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%d %d %ld %ld %lf", &oh, &ow, &ok, &bad, &worst) == 5); fclose(f); }
    char nm[80], tag[80];
    snprintf(nm, sizeof(nm), "nchwc%d %s %s%s", g_entries[c.entry].align, g_pool_name[g_entries[c.entry].pool],
             g_entries[c.entry].divided ? "suredivided" : "nodivided",
             g_entries[c.entry].fixk ? (g_entries[c.entry].fixk == 2 ? "/k2x2" : "/k3x3") : "/general");
    snprintf(tag, sizeof(tag), "N=%d %dx%d C=%d k=%dx%d s=%d -> %dx%d", c.N, c.H, c.W, c.C, c.kH, c.kW, c.S, oh, ow);
    // **结果文件缺失 / 读不出来 = 这个用例没跑完，必须判失败。**
    // ASan 撞上 SEGV 时默认走 Die() -> _exit(1)，**不发信号**，
    // 于是 WIFSIGNALED 为假、退出码也不是 3 —— 缺了这道判断就会把
    // 一个段错误当成"通过"。附录 CJ.4 抓出来的，四个门禁统一补上。
    if (!have) { g_crash++; printf("  没跑完（子进程没写结果文件，退出码 %d）\n", WEXITSTATUS(st)); return; }
    if (WIFSIGNALED(st)) { g_crash++; printf("  %-38s %s  CRASH\n", nm, tag); return; }
    if (bad > 0) { g_bad++; printf("  %-38s %s  FAIL %ld/%ld 格错, 最差 %.3e\n", nm, tag, bad, ok + bad, worst); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC pooling：24 个入口（avg/max × nodivided/suredivided × NCHWC1/4/8）\n");
    printf("内核名全部写全、走函数指针表（nm 核实，不是按头里的声明数 —— 头里有重复声明）\n");
    printf("用真实 ZQ_CNN_Tensor4D_NCHWC{1,4,8}；判据：后向误差 + 逐格统计\n");
    printf("**每个入口只喂符合它自己契约的形状**：\n");
    printf("  suredivided  -> (in-k)%%s==0、out=(in-k)/s+1（窗口总能放下，否则越界读）\n");
    printf("  nodivided   -> out 用 ceil 形状，刻意让最后一个窗口被裁（走 final_kH/final_kW 分支）\n");
    printf("  k2x2/k3x3   -> 只喂 2x2 / 3x3（内核把窗口写死了，分派器也有守卫）\n\n");

    RUNNER r[3] = { &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC1>,
                    &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC4>,
                    &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC8> };
    // (kH, kW, stride) 组合：(in-k)%s==0 的那一批供 suredivided 用
    static const int KK[][3] = { {2,2,2}, {3,3,3}, {2,2,1}, {4,4,2}, {3,3,1} };
    for (int e = 0; e < N_ENTRY; e++) {
        const int ai = (g_entries[e].align == 1) ? 0 : (g_entries[e].align == 4 ? 1 : 2);
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        const int A = g_entries[e].align;
        if (g_entries[e].fixk) {
            const int k = g_entries[e].fixk;
            // 写死窗口的特化：kernel 固定 2x2 / 3x3，用 stride==kernel 让 (in-k)%s==0
            for (int m = 0; m < 2; m++) {
                Case c; memset(&c, 0, sizeof(c));
                c.entry = e; c.N = (m ? 2 : 1); c.H = 4 * k + 2; c.W = 4 * k + 2;
                c.C = A; c.kH = k; c.kW = k; c.S = k;
                one(c, r[ai]);
            }
        } else {
            for (int j = 0; j < 5; j++) {
                Case c; memset(&c, 0, sizeof(c));
                c.entry = e; c.N = (j % 2 ? 2 : 1);
                c.kH = KK[j][0]; c.kW = KK[j][1]; c.S = KK[j][2];
                // 让 (H-kH)%S==0 且 (W-kW)%S==0，suredivided 才合法
                c.H = ((c.kH + 5 * c.S - 1) / c.S) * c.S + c.kH;
                c.W = ((c.kW + 3 * c.S - 1) / c.S) * c.S + c.kW;
                c.C = A;
                one(c, r[ai]);
            }
            Case c2; memset(&c2, 0, sizeof(c2));
            c2.entry = e; c2.N = 1; c2.H = 11; c2.W = 9; c2.C = A + 2; c2.kH = 3; c2.kW = 3; c2.S = 2;
            one(c2, r[ai]);
            if (!g_entries[e].divided) {
                // **刻意让 (in-k) % s != 0**，这样最后一个窗口会被裁，
                // nodivided 的 final_kH / final_kW 那四条分支才真的被执行。
                // 上面所有形状都是 (in-k)%s==0 的（suredivided 的契约要求），
                // 对 nodivided 来说 final_kH 恒等于 kernel_H —— 边界分支一次都走不到。
                for (int j = 0; j < 3; j++) {
                    Case c3; memset(&c3, 0, sizeof(c3));
                    c3.entry = e; c3.N = (j == 1 ? 2 : 1);
                    c3.kH = KK[j][0]; c3.kW = KK[j][1]; c3.S = KK[j][2];
                    c3.C = A;
                    // 让余数非 0：kH/S 整、kW 整（能放进 S 的整数倍再加一点余数）
                    c3.H = c3.kH + (c3.kH / c3.S + 1) * c3.S + 1;
                    c3.W = c3.kW + (c3.kW / c3.S + 1) * c3.S + 1;
                    one(c3, r[ai]);
                }
            }
        }
        printf("  nchwc%d %s %s%s  %d 个用例：对 %d，错 %d，崩 %d\n",
               A, g_pool_name[g_entries[e].pool], g_entries[e].divided ? "suredivided" : "nodivided",
               g_entries[e].fixk ? (g_entries[e].fixk == 2 ? "/k2x2" : "/k3x3") : "/general",
               g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
