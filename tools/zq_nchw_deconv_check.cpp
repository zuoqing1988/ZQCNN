/* NCHW（layers_c）deconvolution 的**越界读**门禁 —— 附录 CX
 *
 * 为什么这道门禁**不做数值比对**（这是本附录最要紧的一条决定）
 * ----------------------------------------------------------
 * 我先按「转置卷积的 gather 形式」写了参考实现，跑出来 45/46 全红。
 * 追下去发现**不是参考错，是这三份实现的语义本身互相不吻合**，而且：
 *
 *  1) 内核里的关系式是 `oh = i*stride_H - pad_top + kh`（转置卷积），
 *     解出 `i = (need_in_h_idx + kh) / stride_H`，`need = oh - pad_top`。✓
 *  2) 但**读 filter 时用的是未翻转的 `kh`**
 *     （`cur_filter_pix_ptr = ... + kh*filter_widthStep + kw*filter_alignPixelStep`）。
 *     转置卷积的 gather 形式在 `oh = i*S - p + k` 下要按 `k` 取；
 *     而这个式子展开成 `oh = i*S + (fH-1-p) - k'` 时要按 `fH-1-k'` 取。
 *     两者不是同一个东西。
 *  3) 分派器算输出尺寸用的是
 *     `need_H = (in_H-1)*S + 1 - (filter_H-1)*d - 1 + (pt+pb) + 1`
 *     —— 形式上又和内核里的关系式对不上。
 *
 * **关键事实**：`grep -i deconv ZQCNN/ZQ_CNN_Layer.h` 为空，
 * `model/` 下也没有任何模型用到它。也就是说：
 *
 *   · 这条路**有完整的一层 → wrapper → 内核 的调用链**（`ZQ_CNN_Net.h:433` 注册了 `DeConvolution` 层类型，
 *     `ZQ_CNN_Layer_DeConvolution` 调 `DeConvolutionWithBiasPReLU` 等四个 wrapper）。
 *     **2026-10-02 我在这里写过「仓内零调用方」——那是错的**：
 *     当时那条 grep **没有加 `-i`**，类名里的 `DeConv` 大小写不匹配，命令静默返回空。
 *     正确的事实是：**随仓库发布的模型里没有一个用它**（`grep -rlin deconv model/` 为空），
 *     但**模型文件可以打开这条路**。见附录 DA.3。
 *   · **没有任何参考实现**可以用来判定"哪种语义才是它想要的"
 *
 * 没有随仓库发布的参考模型 + 三份说法互不吻合 ⇒ **判定不了意图**。
 * 在这种情况下写数值门禁，只能是"拿我猜的语义去判它"，红了说明我猜错、
 * 绿了说明我刚好猜对，两种都没有信息量。所以**不发这道数值门禁**。
 *
 * 这道门禁只断言**一件与语义无关、无歧义**的事：
 *
 *   **无论采用哪种语义，输入下标都必须落在 [0, in_H-1] / [0, in_W-1]。**
 *
 * 附录 CX.2 证明原来的循环上界会让 `real_in_h_idx` 取到 `in_H`，
 * ASan 报 `heap-buffer-overflow READ`。这不是语义分歧的问题：
 * **读到自己缓冲之外，在任何解释下都是越界。**
 * 同仓的 gemm 孪生实现 `zq_cnn_deconvolution_gemm_32f_align_c_raw.h:187`
 * 写的就是 `if (real_in_h_idx < 0 || real_in_h_idx >= in_H)` —— 有守卫。
 *
 * 覆盖面：3 个 `general` 入口（align0 在 .c，align128/align256 在 _raw.h）
 * × 3 组会触发越界的形状 × 2 组不触发的形状。判据：**不许崩、不许 ASan 报**。
 *
 * 沿用 CB~CW：名字写全走函数指针表、每用例 fork 子进程并
 * **显式判"没读到结果文件" = 失败**（CJ.4）、缓冲按对齐要求分配。
 */
#include "zq_check_child.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "ZQCNN/layers_c/zq_cnn_deconvolution_32f_align_c.h"

typedef void (*FN)(
    const float* in_tensor4D_data,
    int in_N, int in_H, int in_W, int in_C,
    int in_alignPixelStep, int in_widthStep, int in_SliceStep,
    const float* filters_data,
    int filter_N, int filter_H, int filter_W, int filter_C,
    int filter_alignPixelStep, int filter_widthStep, int filter_SliceStep,
    int stride_H, int stride_W, int dilation_H, int dilation_W,
    float* out_tensor4D_data,
    int out_N, int out_H, int out_W, int out_C,
    int out_alignPixelStep, int out_alignWidthStep, int out_alignSliceStep,
    int pad_top, int pad_bottom, int pad_left, int pad_right);

enum { K_GEN = 0 };
struct Entry { void* fn; int kind; int align; const char* name; };
static const Entry g_entries[] = {
  { (void*)zq_cnn_deconv_with_padding_32f_align0_general,      K_GEN, 1, "align0_general" },
  { (void*)zq_cnn_deconv_with_padding_32f_align128bit_general, K_GEN, 4, "align128bit_general" },
  { (void*)zq_cnn_deconv_with_padding_32f_align256bit_general, K_GEN, 8, "align256bit_general" },
};
static const int N_ENTRY = 3;

#define RES_FILE "/tmp/zq_decoob_res.txt"

// 形状：(N, in_H, in_W, in_C, filter_N, filter_H, filter_W, stride, pad)
// trigger=1 的三组是附录 CX.2 的越界触发形状：filter 比 (in-1)*S + pad 还大
struct Shape { int N, H, W, C, fN, fH, fW, S, pad, trigger; };
static const Shape g_shapes[] = {
  { 1, 1, 1, 2, 2, 4, 4, 2, 1, 1 },   // in 1x1 + fil 4x4 + s2 + pad1  -> 越界
  { 1, 1, 3, 2, 2, 4, 2, 2, 1, 1 },   // 只在 H 方向触发
  { 1, 2, 2, 2, 2, 3, 3, 2, 2, 1 },   // pad2 + fil3 + s2，边界档
  { 1, 6, 7, 4, 8, 3, 3, 1, 0, 0 },   // 常规，不触发
  { 2, 5, 5, 8, 4, 2, 2, 2, 0, 0 },   // 常规 2x2/s2，不触发
};
static const int N_SHAPE = 5;

// **精确分配，不多给一个字节。**
// 第一版写的是 `posix_memalign(&p, 32, (n + 8) * sizeof(float))` —— 那 8 个 float
// 是为了做 32 字节对齐的余量，结果**恰好把越界读到的那一格藏进了合法内存**：
// 门禁因此在"修复被回退掉"的情况下依然全绿，是一次标准的假绿。
//
// posix_memalign 本身保证返回指针按 alignment 对齐（哪怕只请求 4 字节），
// 所以**精确请求字节数**与**32 字节对齐**并不冲突；
// 而 ASan 的 posix_memalign 拦截器记录的是**请求字节数**，
// 越界一个 float 必落进右红区。这才是"越界一个元素即被抓到"的正确姿势。
static float* alloc32(size_t n)
{
    void* p = 0;
    if (posix_memalign(&p, 32, n * sizeof(float)) != 0) return 0;
    return (float*)p;
}

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

struct Case { int entry, shape; };

static void run_one(const Case& c)
{
    const Entry& e = g_entries[c.entry];
    const Shape& s = g_shapes[c.shape];
    const int A = e.align;
    const int N = s.N, H = s.H, W = s.W, C = s.C;
    const int fN = s.fN, fH = s.fH, fW = s.fW, S = s.S, P = s.pad;

    const int in_ps = (C + A - 1) / A * A,  in_ws = in_ps * W,  in_ss = in_ws * H;
    const int f_ps  = (C + A - 1) / A * A,  f_ws  = f_ps * fW,  f_ss  = f_ws * fH;
    const int o_ps  = (fN + A - 1) / A * A;
    // 输出高度给足：让 (oh - pad_top) 能取到 [-(H-1)*S, (H-1)*S] 这一整段，
    // 越界的 kh 才有出现的机会。
    const int oH = (H - 1) * S + fH + 2 * P + 1;
    const int oW = (W - 1) * S + fW + 2 * P + 1;
    const int o_ws = o_ps * oW, o_ss = o_ws * oH;

    const size_t nin = (size_t)N * in_ss, nf = (size_t)fN * f_ss, nout = (size_t)N * o_ss;
    // **输入缓冲严格只给 N*in_sliceStep 个 float**：任何 real_in_*_idx 越界
    // 都会撞到 ASan 的右红区。（`std::vector` 的 operator new 是按请求字节数
    // 下毒红的，所以越界一个元素必报，不是"越界几个 float 才报"。）
    float *in = alloc32(nin), *fil = alloc32(nf), *out = alloc32(nout);
    if (!in || !fil || !out) { FILE* f = fopen(RES_FILE, "w"); if (f) fprintf(f, "0 1 0 0\n"); fclose(f); return; }
    for (size_t i = 0; i < nin; i++) in[i] = val(1, (int)i);
    for (size_t i = 0; i < nf;  i++) fil[i] = val(2, (int)i);
    for (size_t i = 0; i < nout; i++) out[i] = -777.0f;

    ((FN)e.fn)(in, N, H, W, C, in_ps, in_ws, in_ss,
               fil, fN, fH, fW, C, f_ps, f_ws, f_ss,
               S, S, 1, 1, out, N, oH, oW, fN, o_ps, o_ws, o_ss,
               P, P, P, P);

    // 不做数值比对（见文件头），只确认输出里没有 NaN/Inf
    long bad = 0;
    for (size_t i = 0; i < nout; i++) {
        const float g = out[i];
        if (!(g == g) || g > 1e30f || g < -1e30f) bad++;
    }
    free(in); free(fil); free(out);

    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %.6e 0\n", (long)nout - bad, bad, 0.0); fclose(f); }
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
    // 格式串与实参逐个对齐：10 个 %d 对 10 个实参（第一版多写两个，实参整体错位）
    snprintf(tag, sizeof(tag), "N=%d in=%dx%d C=%d fil=%dx%d K=%d s=%d pad=%d%s",
             s.N, s.H, s.W, s.C, s.fH, s.fW, s.fN, s.S, s.pad,
             s.trigger ? "  [越界触发形状]" : "");
    if (!have || WIFSIGNALED(st)) {
        g_crash++;
        printf("  %-20s %-44s  %s（信号 %d）\n", g_entries[c.entry].name, tag,
               WIFSIGNALED(st) ? "CRASH —— 越界读" : "没跑完（ASan 报错并 _exit）",
               WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st));
        return;
    }
    if (bad > 0) {
        g_bad++;
        printf("  %-20s %-44s  FAIL %ld/%ld 格是 NaN/Inf\n",
               g_entries[c.entry].name, tag, bad, ok + bad);
    } else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHW deconvolution general：**越界读**门禁（附录 CX）\n");
    printf("覆盖面：%d 个 general 入口（k2s2 那一族是 2x2/s2 的特化，另有 gemm 孪生，\n", N_ENTRY);
    printf("  它的 real_in_*_idx 守卫是显式写出来的，不在这道门禁的范围里）\n");
    printf("判据只有一条：**不许读到自己输入缓冲之外**（ASan）。**不做数值比对** ——\n");
    printf("  这几份实现的语义互相不吻合、且没有参考模型，判定不了意图（见文件头）\n");
    printf("输入缓冲严格只给 N*in_sliceStep 个 float（posix_memalign 精确字节数），\n");
    printf("  越界一个元素即落进 ASan 的右红区\n");
#if defined(__SANITIZE_ADDRESS__)
    printf("本 TU 开着 ASan；实现 TU 由 run_zqlib_checks.py 的 $SAN 保证同样带 ASan\n");
    printf("  **--no-asan 跑这道门禁时判据会退化成「不许崩」，越界读查不出来** ——\n");
    printf("  那种模式下这一栏的绿**不代表**没有越界读。\n");
#else
    printf("!! 本 TU **没有** ASan：判据退化成「不许崩」，**越界读查不出来**。\n");
#endif
    printf("\n");
    for (int e = 0; e < N_ENTRY; e++) {
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        for (int sh = 0; sh < N_SHAPE; sh++) { Case c; memset(&c, 0, sizeof(c)); c.entry = e; c.shape = sh; one(c); }
        printf("  %-20s  %d 个用例：对 %d，错 %d，崩 %d\n",
               g_entries[e].name, g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
