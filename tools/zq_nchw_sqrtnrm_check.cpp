/* NCHW（layers_c）sqrt / normalize 门禁 —— 附录 CS
 *
 * 覆盖面：6 个 32f 真实符号（nm 核实）
 *   zq_cnn_sqrt_32f_align0                                       1 个
 *   zq_cnn_normalize_32f_align0                                   1 个（across_spatial 运行时参数）
 *   zq_cnn_normalize_across_spatial_32f_align{128,256}bit        2 个
 *   zq_cnn_normalize_not_across_spatial_32f_align{128,256}bit    2 个
 * 全部在 x86 分派器 ZQ_CNN_Forward_SSEUtils.cpp 里被引用，**是活的**。
 *
 * 语义（逐条从源码读出来的）
 * --------------------------
 *  sqrt(data, …)        就地；out = sqrt(x)
 *
 *  normalize 是 **L2 归一化、不减均值**（不是 batchnorm 那种）：
 *      across_spatial == 1   每张图一个尺度：对每个 n，
 *          s = 1 / sqrt( Σ_{h,w,c} x² + eps )
 *      across_spatial == 0   每个像素一个尺度：对每个 (n,h,w)，
 *          s = 1 / sqrt( Σ_c x² + eps )
 *      然后 out[c] = x[c] * s * scale[c]；channel_shared 时用 scale[0]
 *
 * 上一轮起草这道门禁时**自己的参考下标就写错了**（把三个分支的索引混着用），
 * 于是没有提交。CR.5 记了这件事。本版把三个参考写成**各自独立、
 * 一次只算一种布局**的小函数，不做任何跨分支复用。
 *
 * 沿用 CB~CQ：名字写全走函数指针表、逐格统计、每用例 fork 子进程并
 * **显式判"没读到结果文件" = 失败**（CJ.4）、缓冲区 32 字节对齐（CJ.4 / CP.5）。
 */
#include "zq_check_child.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "ZQCNN/layers_c/zq_cnn_sqrt_32f_align_c.h"
#include "ZQCNN/layers_c/zq_cnn_normalize_32f_align_c.h"

typedef void (*F_SQRT)(float* d, int N, int H, int W, int C, int ps, int ws, int ss);
// align0 版：across_spatial / channel_shared 都是运行时参数（12 参）
typedef void (*F_NRM0)(int across_spatial, int channel_shared, float* in, const float* scale,
                       int N, int H, int W, int C, int ps, int ws, int ss, float eps);
// 特化版：across_spatial 已经写死在函数名里（11 参）
typedef void (*F_NRMA)(int channel_shared, float* in, const float* scale,
                       int N, int H, int W, int C, int ps, int ws, int ss, float eps);

enum { K_SQRT = 0, K_NRM_ACROSS = 1, K_NRM_NOTACROSS = 2 };
struct Entry { void* fn; int kind; int align; };

static const Entry g_entries[] = {
  { (void*)zq_cnn_sqrt_32f_align0, K_SQRT, 1 },
  { (void*)zq_cnn_normalize_across_spatial_32f_align128bit, K_NRM_ACROSS, 4 },
  { (void*)zq_cnn_normalize_across_spatial_32f_align256bit, K_NRM_ACROSS, 8 },
  { (void*)zq_cnn_normalize_not_across_spatial_32f_align128bit, K_NRM_NOTACROSS, 4 },
  { (void*)zq_cnn_normalize_not_across_spatial_32f_align256bit, K_NRM_NOTACROSS, 8 },
  { (void*)zq_cnn_normalize_32f_align0, K_NRM_ACROSS, 1 },   // across_spatial = 1
  // 同一个函数再注册一次，走 across_spatial = 0（**附录 CR 修的就是这一支**）。
  // 少了这一行，CR 的修复没有任何门禁直接覆盖 —— 见附录 CS.6。
  { (void*)zq_cnn_normalize_32f_align0, K_NRM_NOTACROSS, 1 },
};
static const char* g_name[] = {
  "zq_cnn_sqrt_32f_align0",
  "zq_cnn_normalize_across_spatial_32f_align128bit",
  "zq_cnn_normalize_across_spatial_32f_align256bit",
  "zq_cnn_normalize_not_across_spatial_32f_align128bit",
  "zq_cnn_normalize_not_across_spatial_32f_align256bit",
  "zq_cnn_normalize_32f_align0[across_spatial=1]",
  "zq_cnn_normalize_32f_align0[across_spatial=0]",
};
static const int N_ENTRY = 7;

#define RES_FILE "/tmp/zq_sn_res.txt"
static const double TOL = 1e-5;

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
    const int N = 2, H = 5, W = 7, C = c.C;
    const int ps = C, ws = ps * W, ss = ws * H;      // NCHW：pixelStep 就是 C
    const float eps = 1e-3f;
    const size_t n = (size_t)N * ss;

    // 三个缓冲区都 32 字节对齐（CJ.4 / CP.5）
    std::vector<float> in_m(n + 8), ref_m(n + 8), sc_m((size_t)C + 8);
    float* in  = (float*)(((size_t)in_m.data()  + 31) / 32 * 32);
    float* ref = (float*)(((size_t)ref_m.data() + 31) / 32 * 32);
    float* sc  = (float*)(((size_t)sc_m.data()  + 31) / 32 * 32);

    // variant 0：普通数据；variant 1：**全零** —— 那正是 CR 修掉的 eps 缺失会炸的场景
    for (size_t i = 0; i < n; i++)
        in[i] = (c.variant == 1) ? 0.0f
              : (e.kind == K_SQRT ? fabsf(val(1, (int)i)) : val(1, (int)i));
    for (int k = 0; k < C; k++) sc[k] = 0.5f + 0.1f * (k % 5);
    const int shared = c.variant ? 1 : 0;

    if (e.kind == K_SQRT) {
        for (size_t i = 0; i < n; i++) ref[i] = sqrt((double)in[i]);
        ((F_SQRT)e.fn)(in, N, H, W, C, ps, ws, ss);
    } else if (e.kind == K_NRM_ACROSS) {
        // 每张图一个尺度：Σ_{h,w,c} x²
        for (int nn = 0; nn < N; nn++) {
            double sum = 0.0;
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    for (int k = 0; k < C; k++) {
                        const double v = in[(size_t)nn * ss + h * ws + w * ps + k];
                        sum += v * v;
                    }
            const double s = 1.0 / sqrt(sum + eps);
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    for (int k = 0; k < C; k++) {
                        const size_t o = (size_t)nn * ss + h * ws + w * ps + k;
                        ref[o] = (float)(in[o] * s * (shared ? sc[0] : sc[k]));
                    }
        }
        if (e.align == 1) ((F_NRM0)e.fn)(1, shared, in, sc, N, H, W, C, ps, ws, ss, eps);
        else                ((F_NRMA)e.fn)(shared, in, sc, N, H, W, C, ps, ws, ss, eps);
    } else {
        // 每个像素一个尺度：Σ_c x²
        for (int nn = 0; nn < N; nn++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++) {
                    double sum = 0.0;
                    for (int k = 0; k < C; k++) {
                        const double v = in[(size_t)nn * ss + h * ws + w * ps + k];
                        sum += v * v;
                    }
                    const double s = 1.0 / sqrt(sum + eps);
                    for (int k = 0; k < C; k++) {
                        const size_t o = (size_t)nn * ss + h * ws + w * ps + k;
                        ref[o] = (float)(in[o] * s * (shared ? sc[0] : sc[k]));
                    }
                }
        if (e.align == 1) ((F_NRM0)e.fn)(0, shared, in, sc, N, H, W, C, ps, ws, ss, eps);
        else                ((F_NRMA)e.fn)(shared, in, sc, N, H, W, C, ps, ws, ss, eps);
    }

    long bad = 0; double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double v = fabs((double)ref[i]);
        const double den = v > 1.0 ? v : 1.0;
        double be = fabs((double)in[i] - (double)ref[i]) / den;
        if (!(be == be)) { bad++; if (1.0 > worst) worst = 1.0; continue; }   // NaN 直接算错
        if (be > TOL) bad++;
        if (be > worst) worst = be;
    }
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %.6e\n", (long)n - bad, bad, worst); fclose(f); }
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(const Case& c)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) {
        zq_child_silence_stderr();
        run_one(c);
        _exit(0);
    }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0; double worst = 0;
    int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%ld %ld %lf", &ok, &bad, &worst) == 3); fclose(f); }
    char tag[80];
    snprintf(tag, sizeof(tag), "C=%d %s", c.C, c.variant ? "全零/channel_shared" : "普通");
    if (!have) { g_crash++; printf("  %-52s %s  没跑完（退出码 %d）\n", g_name[c.entry], tag, WEXITSTATUS(st)); return; }
    if (WIFSIGNALED(st)) { g_crash++; printf("  %-52s %s  CRASH(信号 %d)\n", g_name[c.entry], tag, WTERMSIG(st)); return; }
    if (bad > 0) { g_bad++; printf("  %-52s %s  FAIL %ld/%ld 格错, 最差 %.3e\n", g_name[c.entry], tag, bad, ok + bad, worst); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHW sqrt / normalize：%d 个 32f 入口全测\n", N_ENTRY);
    printf("内核名全部写全、走函数指针表；判据：逐格后向误差，逐格统计\n");
    printf("normalize 的每个入口都跑两次：普通数据 + **全零输入**\n");
    printf("  （全零正是附录 CR 修掉的 eps 缺失会算出 NaN 的那个场景）\n");
    printf("C 取 align 与 2*align 两种；缓冲区 32 字节对齐（CJ.4 / CP.5）\n\n");

    for (int e = 0; e < N_ENTRY; e++) {
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        const int A = g_entries[e].align;
        for (int v = 0; v < 2; v++)
            for (int cc = 0; cc < 2; cc++) {
                Case c; memset(&c, 0, sizeof(c));
                c.entry = e; c.variant = v; c.C = (cc ? A * 2 : A);
                one(c);
            }
        printf("  %-52s  %d 个用例：对 %d，错 %d，崩 %d\n",
               g_name[e], g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
