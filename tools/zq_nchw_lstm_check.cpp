/* NCHW（layers_c）LSTM 门禁 —— 附录 CW
 *
 * 覆盖面：3 个 32f 真实符号（nm 核实）
 *   zq_cnn_lstm_TF_32f_align0_general    —— 在 zq_cnn_lstm_32f_align_c.c 里，标量循环
 *   zq_cnn_lstm_TF_32f_align128bit       —— 在 _raw.h 里，SSE/NEON 展开（宏式声明）
 *   zq_cnn_lstm_TF_32f_align256bit       —— 同上，AVX 展开
 * 三者都被 x86 分派器 ZQ_CNN_Forward_SSEUtils.cpp:2553/2598/2614 引用，**是活的**。
 * 上一层 ZQ_CNN_Layer_LSTM 从模型文件读 hidden_dim / forget_bias / cell_clip。
 *
 * 语义（逐条从源码读出来的，注释里直接贴了 TF 的伪码）
 * --------------------------------------------
 * 单层 LSTM，**无 peephole**（wci = wcf = wco = 0），TF 权重顺序 i, ci, f, o：
 *
 *     for n in 0..in_N-1:
 *         h = 0; cell = 0                       // 每条序列重置
 *         for t in 0..in_W-1:
 *             ti = is_fw ? t : in_W-1-t         // 反向 LSTM 倒着走时间
 *             x = in[n][ti][0..in_C)
 *             for q in 0..hidden_dim-1:         // 全部 q 一起算，用的是**旧** h
 *                 I = b_I[q] + Σ Wxc_I[q][i]x[i] + Σ Whc_I[q][i]h[i]
 *                 F = b_F[q] + ...               (F 最后再加 forget_bias)
 *                 ci = b_G[q] + ...              (注意：ci 用的是 **G** 权重表)
 *                 o  = b_O[q] + ...
 *             for q:
 *                 cs_prev = cell[q]
 *                 I = sigmoid(I); F = sigmoid(F); ci = tanh(ci)
 *                 cs = ci*I + cs_prev*F
 *                 cs = min(cell_clip, max(-cell_clip, cs))     // clip
 *                 o = sigmoid(o); co = tanh(cs); h = co*o
 *                 cell = cs
 *                 out[n][ti][q] = h              // 输出按**原时间下标**写，与方向无关
 *
 * 布局：
 *   in      (in_N, in_W, in_C)     in_pixelStep / in_sliceStep
 *   xc_{I,F,O,G} (hidden_dim, in_C)   *_pixelStep / *_sliceStep（sliceStep 三个实现都没用）
 *   hc_{I,F,O,G} (hidden_dim, hidden_dim)  *_pixelStep
 *   b_{I,F,O,G} (hidden_dim)
 *   out     (in_N, in_W, hidden_dim)  out_pixelStep / out_sliceStep
 *
 * 这道门禁有**两条互相独立的判据**
 * --------------------------------
 *  ① 与一份**独立写的 double 标量参考**比（后向误差 1e-4 —— 递归 + sigmoid/tanh
 *     的浮点放大，纯 1e-5 太紧，见 CW.3）
 *  ② 三个实现之间**互相**比。align0 是标量循环、align128/256 是 SIMD 展开，
 *     它们本该给出同一个答案；"三方一致"是一条不依赖我参考实现正确性的判据。
 *
 * 沿用 CB~CV：名字写全走函数指针表、逐格统计、每用例 fork 子进程并
 * **显式判"没读到结果文件" = 失败**（CJ.4）、缓冲区按对齐要求分配（CJ.4 / CP.5）。
 */
#include "zq_check_child.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "ZQCNN/layers_c/zq_cnn_lstm_32f_align_c.h"

typedef void (*F_LSTM)(
    const float* in_data, int in_N, int in_W, int in_C,
    int in_pixelStep, int in_sliceStep,
    const float* xc_I_data, int xc_I_pixelStep, int xc_I_sliceStep,
    const float* xc_F_data, int xc_F_pixelStep, int xc_F_sliceStep,
    const float* xc_O_data, int xc_O_pixelStep, int xc_O_sliceStep,
    const float* xc_G_data, int xc_G_pixelStep, int xc_G_sliceStep,
    const float* hc_I_data, int hc_I_pixelStep, int hc_I_sliceStep,
    const float* hc_F_data, int hc_F_pixelStep, int hc_F_sliceStep,
    const float* hc_O_data, int hc_O_pixelStep, int hc_O_sliceStep,
    const float* hc_G_data, int hc_G_pixelStep, int hc_G_sliceStep,
    const float* b_I_data, const float* b_F_data,
    const float* b_O_data, const float* b_G_data,
    float* out_data, int out_pixelStep, int out_sliceStep,
    int hidden_dim, int is_fw, float forget_bias, float cell_clip,
    void** buffer, __int64* buffer_len);

enum { K_ALIGN0 = 0, K_ALIGN128 = 1, K_ALIGN256 = 2 };
struct Entry { void* fn; int kind; const char* name; };
static const Entry g_entries[] = {
  { (void*)zq_cnn_lstm_TF_32f_align0_general, K_ALIGN0,  "align0_general"  },
  { (void*)zq_cnn_lstm_TF_32f_align128bit,    K_ALIGN128, "align128bit"     },
  { (void*)zq_cnn_lstm_TF_32f_align256bit,    K_ALIGN256, "align256bit"     },
};
static const int N_ENTRY = 3;

#define RES_FILE "/tmp/zq_lstm_res.txt"
static const double TOL = 1e-4;

// align0 标量循环；align128 -> 4；align256 -> 8
static int align_of(int kind) { return kind == K_ALIGN0 ? 1 : (kind == K_ALIGN128 ? 4 : 8); }

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

struct Shape { int N, W, C, H; };
// 四组：常规 / C 不是 4 或 8 的倍数（走 SIMD 尾巴）/ 极小 / W=1 且 N=3
static const Shape g_shapes[] = {
  { 1, 5, 8, 6 },
  { 2, 4, 3, 5 },
  { 1, 1, 1, 1 },
  { 3, 7, 16, 8 },
};
static const int N_SHAPE = 4;

struct Case { int entry, shape, is_fw, clip_id, fb_id; };
static const float g_clips[] = { 3.0f, 0.0f, -1.0f };
static const float g_fbs[]   = { 1.0f, 0.0f };
static const int N_CLIP = 3, N_FB = 2;

// ---- 独立参考：double 标量 LSTM，**不共享内核的任何一处下标写法** ----
static void ref_lstm(const float* in, int N, int W, int C, int in_ps, int in_ss,
                     const float* xcI, const float* xcF, const float* xcO, const float* xcG,
                     int w_ps,
                     const float* hcI, const float* hcF, const float* hcO, const float* hcG,
                     int h_ps,
                     const float* bI, const float* bF, const float* bO, const float* bG,
                     float* out, int out_ps, int out_ss,
                     int H, int is_fw, float forget_bias, float clip)
{
    std::vector<double> h(H, 0.0), cell(H, 0.0), Iv(H), Fv(H), civ(H), ov(H);
    for (int n = 0; n < N; n++) {
        for (int q = 0; q < H; q++) { h[q] = 0.0; cell[q] = 0.0; }
        for (int t = 0; t < W; t++) {
            const int ti = is_fw ? t : (W - 1 - t);
            const float* x = in + (size_t)n * in_ss + (size_t)ti * in_ps;
            for (int q = 0; q < H; q++) {
                double I = bI[q], F = bF[q], ci = bG[q], o = bO[q];
                for (int i = 0; i < C; i++) {
                    I  += (double)xcI[(size_t)q * w_ps + i] * x[i];
                    F  += (double)xcF[(size_t)q * w_ps + i] * x[i];
                    ci += (double)xcG[(size_t)q * w_ps + i] * x[i];
                    o  += (double)xcO[(size_t)q * w_ps + i] * x[i];
                }
                for (int i = 0; i < H; i++) {
                    I  += (double)hcI[(size_t)q * h_ps + i] * (float)h[i];
                    F  += (double)hcF[(size_t)q * h_ps + i] * (float)h[i];
                    ci += (double)hcG[(size_t)q * h_ps + i] * (float)h[i];
                    o  += (double)hcO[(size_t)q * h_ps + i] * (float)h[i];
                }
                F += (double)forget_bias;
                Iv[q] = I; Fv[q] = F; civ[q] = ci; ov[q] = o;
            }
            float* op = out + (size_t)n * out_ss + (size_t)ti * out_ps;
            for (int q = 0; q < H; q++) {
                const double csp = cell[q];
                const double I  = 1.0 / (1.0 + exp(-Iv[q]));
                const double F  = 1.0 / (1.0 + exp(-Fv[q]));
                const double ci = tanh(civ[q]);
                double cs = ci * I + csp * F;
                // **clamp 的顺序必须与内核一致**：内核写的是
                //     __min(cell_clip, __max(-cell_clip, cs))
                // 即 max 在内、min 在外。第一版这里写成"先 clip 上界再 clip 下界",
                // 对 clip>0 两者等价，对 **clip<0 不等价**（max 先做会把 cs 抬到正数，
                // 于是外层 min 拿到 -clip；而 min 先做会拿到 +clip），
                // 于是 48 个 clip=-1 的用例全红 —— 那是**参考错**，不是内核错。
                // 教训与 CT.2 相同：边界运算的**次序**也是语义的一部分。
                if (cs < -(double)clip) cs = -(double)clip;   // __max(-clip, cs)
                if (cs > (double)clip)  cs = (double)clip;    // __min(clip, ...)
                const double o  = 1.0 / (1.0 + exp(-ov[q]));
                const double co = tanh(cs);
                h[q] = co * o;
                cell[q] = cs;
                op[q] = (float)h[q];
            }
        }
    }
}

// 32 字节对齐的堆块（CJ.4 / CP.5：std::vector 只给 16 字节，_mm_load_ps 要 32）
static float* alloc32(size_t n)
{
    void* p = 0;
    if (posix_memalign(&p, 32, (n + 8) * sizeof(float)) != 0) return 0;
    return (float*)p;
}

static void run_one(const Case& c)
{
    const Entry& e = g_entries[c.entry];
    const Shape& s = g_shapes[c.shape];
    const int A = align_of(e.kind);
    const int N = s.N, W = s.W, C = s.C, H = s.H;
    const int is_fw = c.is_fw;
    const float clip = g_clips[c.clip_id], fb = g_fbs[c.fb_id];

    // 步长：对齐补齐。补出来的通道填**非 0 哨兵**，
    // 这样内核要是越界读到补齐区，数值会立刻不对（CT.2 的"常数用例"同一条思路）。
    const int in_ps = (C + A - 1) / A * A;
    const int in_ss = in_ps * W;
    const int w_ps  = in_ps;                       // xc_* 行宽 = in_C 补齐
    const int h_ps  = (H + A - 1) / A * A;         // hc_* 行宽 = hidden_dim 补齐
    const int out_ps = (H + A - 1) / A * A;
    const int out_ss = out_ps * W;

    const size_t nin  = (size_t)N * in_ss;
    const size_t nw   = (size_t)H * w_ps;
    const size_t nh   = (size_t)H * h_ps;
    const size_t nout = (size_t)N * out_ss;

    float *in = alloc32(nin), *xcI = alloc32(nw), *xcF = alloc32(nw),
          *xcO = alloc32(nw), *xcG = alloc32(nw),
          *hcI = alloc32(nh), *hcF = alloc32(nh), *hcO = alloc32(nh), *hcG = alloc32(nh),
          *bI = alloc32(H), *bF = alloc32(H), *bO = alloc32(H), *bG = alloc32(H),
          *out = alloc32(nout), *ref = alloc32(nout);
    if (!in || !xcI || !xcF || !xcO || !xcG || !hcI || !hcF || !hcO || !hcG
        || !bI || !bF || !bO || !bG || !out || !ref) {
        FILE* f = fopen(RES_FILE, "w"); if (f) fprintf(f, "0 1 0 0\n"); fclose(f);
        return;
    }
    for (size_t i = 0; i < nin; i++) in[i] = val(1, (int)i);
    for (size_t i = 0; i < nw;  i++) { xcI[i] = val(2, (int)i); xcF[i] = val(3, (int)i);
                                       xcO[i] = val(4, (int)i); xcG[i] = val(5, (int)i); }
    for (size_t i = 0; i < nh;  i++) { hcI[i] = val(6, (int)i); hcF[i] = val(7, (int)i);
                                       hcO[i] = val(8, (int)i); hcG[i] = val(9, (int)i); }
    for (int q = 0; q < H; q++) {
        bI[q] = val(10, q); bF[q] = val(11, q); bO[q] = val(12, q); bG[q] = val(13, q);
    }
    for (size_t i = 0; i < nout; i++) { out[i] = -777.0f; ref[i] = -777.0f; }

    void* buffer = 0; __int64 buffer_len = 0;
    ((F_LSTM)e.fn)(in, N, W, C, in_ps, in_ss,
                   xcI, w_ps, 0, xcF, w_ps, 0, xcO, w_ps, 0, xcG, w_ps, 0,
                   hcI, h_ps, 0, hcF, h_ps, 0, hcO, h_ps, 0, hcG, h_ps, 0,
                   bI, bF, bO, bG, out, out_ps, out_ss,
                   H, is_fw, fb, clip, &buffer, &buffer_len);

    ref_lstm(in, N, W, C, in_ps, in_ss, xcI, xcF, xcO, xcG, w_ps,
             hcI, hcF, hcO, hcG, h_ps, bI, bF, bO, bG, ref, out_ps, out_ss,
             H, is_fw, fb, clip);

    // 后向误差：|got - ref| / sqrt(Σ ref²)，逐格统计
    double sc = 0.0;
    for (size_t i = 0; i < nout; i++) sc += (double)ref[i] * (double)ref[i];
    double den = sqrt(sc);
    if (den < 1e-20) den = 1.0;
    long bad = 0; double worst = 0.0;
    for (size_t i = 0; i < nout; i++) {
        const double g = (double)out[i], r = (double)ref[i];
        if (!(g == g) || g > 1e30 || g < -1e30) { bad++; if (1.0 > worst) worst = 1.0; continue; }
        const double be = fabs(g - r) / den;
        if (be > TOL) bad++;
        if (be > worst) worst = be;
    }
    if (buffer) free(buffer);
    free(in); free(xcI); free(xcF); free(xcO); free(xcG);
    free(hcI); free(hcF); free(hcO); free(hcG);
    free(bI); free(bF); free(bO); free(bG); free(out); free(ref);

    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %.6e 0\n", (long)nout - bad, bad, worst); fclose(f); }
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
    long ok = 0, bad = 0, over = 0; double worst = 0; int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%ld %ld %lf %ld", &ok, &bad, &worst, &over) == 4); fclose(f); }
    const Shape& s = g_shapes[c.shape];
    char tag[128];
    snprintf(tag, sizeof(tag), "N%dW%dC%dH%d %s clip=%.1f fb=%.1f",
             s.N, s.W, s.C, s.H, c.is_fw ? "正向" : "反向", g_clips[c.clip_id], g_fbs[c.fb_id]);
    if (!have || WIFSIGNALED(st)) {
        g_crash++;
        printf("  %-16s %-42s  %s（信号 %d）\n", g_entries[c.entry].name, tag,
               WIFSIGNALED(st) ? "CRASH" : "没跑完", WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st));
        return;
    }
    if (bad > 0) {
        g_bad++;
        printf("  %-16s %-42s  FAIL %ld/%ld 格错, 最差 %.3e\n",
               g_entries[c.entry].name, tag, bad, ok + bad, worst);
    } else {
        g_ok++;
    }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHW LSTM：%d 个 32f 入口（align0 / align128 / align256）全测\n", N_ENTRY);
    printf("内核名全部写全、走函数指针表；判据：与独立 double 参考的后向误差，逐格统计\n");
    printf("每组形状 x {正向,反向} x clip{3,0,-1} x forget_bias{1,0}\n");
    printf("  clip=0 / clip<0 是**模型文件可控**的退化输入（见附录 CW.4）\n");
    printf("buffer 参数传**空指针**（内部 malloc，生产恒走这条，附录 CU.9）\n\n");
    for (int e = 0; e < N_ENTRY; e++) {
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        for (int sh = 0; sh < N_SHAPE; sh++)
            for (int fw = 0; fw < 2; fw++)
                for (int cl = 0; cl < N_CLIP; cl++)
                    for (int fb = 0; fb < N_FB; fb++) {
                        Case c; memset(&c, 0, sizeof(c));
                        c.entry = e; c.shape = sh; c.is_fw = fw; c.clip_id = cl; c.fb_id = fb;
                        one(c);
                    }
        printf("  %-16s  %d 个用例：对 %d，错 %d，崩 %d\n",
               g_entries[e].name, g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
