/* NCHW（layers_c）no_padding 卷积门禁 —— 附录 CE
 *
 * 为什么要有这个门禁
 * ------------------
 * `ZQCNN/layers_c/zq_cnn_convolution_gemm_32f_align_c*.{c,h}` 是 **NCHW 卷积**，
 * 也就是 x86（Windows + Linux）上的**主生产路径**（NCHWC 只是可选的 SIMD 变体）。
 * 仓库里 21 个 `zq_*_check.cpp` 覆盖了 NCHWC 卷积（`zq_nchwc_conv*`）、
 * NCHWC innerproduct、NCHW 的 eltwise / pooling / lrn，
 * **唯独没有测 NCHW 卷积本身**。
 * 而附录 CB 刚在 NCHWC 那一族的相邻两个函数之间找出三处独立缺陷 ——
 * 同一个族的另一个成员出问题，一点也不奇怪。
 *
 * 设计上照抄 CB 那两道门禁已经验证过的做法
 * ----------------------------------------
 *  · 内核名在数组里**写全**、走函数指针表，**不做任何字符串拼接**
 *    （CA.3：宏拼接会把签名写错变成静默传错；写错名字会**编译报错**）
 *  · 判据用**后向误差** `|got-exp| / sqrt(sum(a^2 f^2))`，阈值 1e-5
 *  · **逐格统计**（多少格对 / 多少格错 / 最差是多少），不拿"最差格"当结论（CA.5）
 *  · 每个用例 fork 一个子进程；子进程 stderr 接 /dev/null、只写结果文件
 *    —— ASan 的报告走 stderr，会把父进程 stdout 那一行拦腰截断（BN.5）
 *
 * extern "C" 声明是**照抄 `zq_cnn_convolution_gemm_32f_align_c.h` 连参数名一起抄**的：
 * C 链接不检查 arity，少写一个参数只会让实参整体错位（AGENTS.md 专门有一条）。
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "layers_c/zq_cnn_convolution_gemm_32f_align_c.h"

typedef void (*FN)(
    const float* in_tensor4D_data,
    int in_N, int in_H, int in_W, int in_C,
    int in_pixelStep, int in_widthStep, int in_sliceStep,
    const float* filters_data,
    int filter_N, int filter_H, int filter_W, int filter_C,
    int filter_pixelStep, int filter_widthStep, int filter_sliceStep,
    int stride_H, int stride_W,
    int dilation_H, int dilation_W,
    float* out_tensor4D_data,
    int out_N, int out_H, int out_W, int out_C,
    int out_pixelStep, int out_widthStep, int out_sliceStep,
    void** buffer, __int64* buffer_len);

// ---- 契约 ----
// align : 0 = 不补齐（pixelStep 必须恰好等于 C）；4 = 补到 4 的倍数；8 = 补到 8 的倍数
// kind  : 0 = 通用 / 1 = kernel1x1 / 2 = 要求 in_C==4 / 3 = batch(N>1) / 4 = C3(in_C<=4)
struct Entry {
    FN         fn;
    const char* name;
    int        align;
    int        kind;
};

// **名字全部写全**，不做拼接 —— 写错会编译报错，不会静默传错
static const Entry g_entries[] = {
    { zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep,
      "align128bit_same_pixstep", 4, 0 },
    { zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_kernel1x1,
      "align128bit_same_pixstep_kernel1x1", 4, 1 },
    { zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_C4,
      "align128bit_same_pixstep_C4", 4, 2 },
    { zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_batch,
      "align128bit_same_pixstep_batch", 4, 3 },
    { zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep,
      "align256bit_same_pixstep", 8, 0 },
    { zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep_kernel1x1,
      "align256bit_same_pixstep_kernel1x1", 8, 1 },
    { zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep_C4,
      "align256bit_same_pixstep_C4", 8, 2 },
    { zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep_batch,
      "align256bit_same_pixstep_batch", 8, 3 },
    { zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep,
      "align0_same_or_notsame_pixstep", 0, 0 },
    { zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep,
      "align128bit_same_or_notsame_pixstep", 4, 0 },
    { zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep,
      "align256bit_same_or_notsame_pixstep", 8, 0 },
    { zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_C3,
      "align128bit_same_or_notsame_pixstep_C3", 4, 4 },
    { zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep_C3,
      "align256bit_same_or_notsame_pixstep_C3", 8, 4 },
    { zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep_batch,
      "align0_same_or_notsame_pixstep_batch", 0, 3 },
    { zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_batch,
      "align128bit_same_or_notsame_pixstep_batch", 4, 3 },
    { zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep_batch,
      "align256bit_same_or_notsame_pixstep_batch", 8, 3 },
};
static const int N_ENTRY = (int)(sizeof(g_entries) / sizeof(g_entries[0]));

#define RES_FILE "/tmp/zq_nchw_conv_res.txt"
static const double TOL = 1e-5;

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

struct Case {
    int entry, N, H, W, C, K, fH, fW, stride, dil, diff_pixstep;
    int in_pixStep, f_pixStep;     // run_case 回填，只为了把用例行打全
    int buf_mode;                  // 0 = 传空指针（生产恒走这条）/ 1 = 调用方给缓冲
    long acc_ok, acc_bad;          // 两种模式累计
    double acc_worst;
};

// 逐格比对：后向误差 |got - ref| / sqrt(Σ a²f²)，**逐格统计**（附录 CA.5）
static int check_out(Case& c, const std::vector<float>& in,
                     const std::vector<float>& flt, const std::vector<float>& out,
                     int N, int C, int K, int fH, int fW, int S, int D, int oH, int oW,
                     int in_pixStep, int in_widthStep, int in_sliceStep,
                     int f_pixStep, int f_widthStep, int f_sliceStep,
                     int out_pixStep, int out_widthStep, int out_sliceStep)
{
    long n_ok = 0, n_bad = 0;
    double worst = 0.0;
    for (int n = 0; n < N; n++)
        for (int oh = 0; oh < oH; oh++)
            for (int ow = 0; ow < oW; ow++)
                for (int k = 0; k < K; k++) {
                    double sum = 0, sc = 0;
                    for (int cc = 0; cc < C; cc++)
                        for (int fh = 0; fh < fH; fh++)
                            for (int fw = 0; fw < fW; fw++) {
                                double a = in[(size_t)n * in_sliceStep + (oh * S + fh * D) * in_widthStep + (ow * S + fw * D) * in_pixStep + cc];
                                double f = flt[(size_t)k * f_sliceStep + fh * f_widthStep + fw * f_pixStep + cc];
                                sum += a * f; sc += a * a * f * f;
                            }
                    double got = out[(size_t)n * out_sliceStep + oh * out_widthStep + ow * out_pixStep + k];
                    double den = sqrt(sc); if (den < 1e-30) den = 1.0;
                    double be = fabs(got - sum) / den;
                    if (be > TOL) n_bad++; else n_ok++;
                    if (be > worst) worst = be;
                }
    c.acc_ok += n_ok; c.acc_bad += n_bad;
    if (worst > c.acc_worst) c.acc_worst = worst;
    return 0;
}

static int run_case(Case& c)
{
    const Entry& e = g_entries[c.entry];
    const int N = c.N, H = c.H, W = c.W, C = c.C, K = c.K;
    const int fH = c.fH, fW = c.fW, S = c.stride, D = c.dil;
    const int effH = (fH - 1) * D + 1, effW = (fW - 1) * D + 1;
    if (H < effH || W < effW) return 2;
    const int oH = (H - effH) / S + 1, oW = (W - effW) / S + 1;
    if (oH <= 0 || oW <= 0) return 2;

    // ---- 通道补齐 ----
    int in_pixStep = C;
    if (e.align == 4) in_pixStep = (C + 3) / 4 * 4;
    if (e.align == 8) in_pixStep = (C + 7) / 8 * 8;
    int f_pixStep = in_pixStep;
    if (c.diff_pixstep) {
        // same_or_notsame 那一族的头注释写着 "in_pixStep can be different with
        // filter_pixStep"，所以要**真的**造出一个不同的值去喂它。
        // 第一版写的是"把 f_pixStep 也补到 4 的倍数"，对 C=8 算出来还是 8，
        // 等于根本没测到这一支 —— 现在直接 +4，并把它打进用例行里，
        // 免得又出现"以为测了、其实没测"。
        f_pixStep = in_pixStep + 4;
    }
    int out_pixStep = K;
    if (e.align == 4) out_pixStep = (K + 3) / 4 * 4;
    if (e.align == 8) out_pixStep = (K + 7) / 8 * 8;

    const int in_widthStep = in_pixStep * W, in_sliceStep = in_widthStep * H;    const int f_widthStep = f_pixStep * fW,  f_sliceStep = f_widthStep * fH;
    const int out_widthStep = out_pixStep * oW, out_sliceStep = out_widthStep * oH;

    // ---- 填数据（紧凑 NCHW：偏移 n*slice + h*width + w*pixel + c）----
    std::vector<float> in((size_t)N * in_sliceStep, 7.0f);   // 补齐通道填非 0 的哨兵
    std::vector<float> flt((size_t)K * f_sliceStep, 5.0f);
    std::vector<float> out((size_t)N * out_sliceStep, -12345.0f);
    for (int n = 0; n < N; n++)
        for (int cc = 0; cc < C; cc++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    in[(size_t)n * in_sliceStep + h * in_widthStep + w * in_pixStep + cc] = val(1, ((n * C + cc) * H + h) * W + w);
    for (int k = 0; k < K; k++)
        for (int cc = 0; cc < C; cc++)
            for (int fh = 0; fh < fH; fh++)
                for (int fw = 0; fw < fW; fw++)
                    flt[(size_t)k * f_sliceStep + fh * f_widthStep + fw * f_pixStep + cc] = val(2, ((k * C + cc) * fH + fh) * fW + fw);

    c.in_pixStep = in_pixStep; c.f_pixStep = f_pixStep;

    // **两种 buffer 模式都要跑**（附录 CU.9）。
    //
    // 库的判据是 `if (buffer == 0)` —— 判的是**那个 void** 本身是不是空指针，
    // 不是 `*buffer`。而生产侧 ZQ_CNN_Layer.h:308 是
    //     void** tmp_buffer = use_buffer ? buffer : 0;
    // `use_buffer` 构造函数里初始化为 false，全仓没有一处置 true，
    // 所以**生产恒定传 0**，恒定走"内部 malloc"那条分支。
    //
    // 这道门禁原来只传 `&buffer`，也就是"调用方给缓冲"那条 —— **生产从不走的那条**。
    // 于是"内部 malloc"模式一直没有任何数值门禁，
    // 而它恰好是附录 CU.6 那 7 处释放非自有内存、CU.7 那 2 处静默无输出
    // 所在的分支。**绿色覆盖错了地方。**
    for (int mode = 0; mode < 2; mode++) {
        for (size_t i = 0; i < out.size(); i++) out[i] = -12345.0f;
        void* buffer = 0;
        __int64 buffer_len = 0;
        void** parg = 0;                  // mode 0：空指针 -> 内部 malloc
        if (mode == 1) {                  // mode 1：调用方给缓冲
            buffer = _aligned_malloc(32, 32);
            if (!buffer) return 2;
            parg = &buffer;
        }
        e.fn(&in[0], N, H, W, C, in_pixStep, in_widthStep, in_sliceStep,
             &flt[0], K, fH, fW, C, f_pixStep, f_widthStep, f_sliceStep,
             S, S, D, D,
             &out[0], N, oH, oW, K, out_pixStep, out_widthStep, out_sliceStep,
             parg, mode ? &buffer_len : 0);
        if (mode == 1 && buffer) _aligned_free(buffer);
        c.buf_mode = mode;
        // 注意判据方向：check_out **返回 0 表示成功**（不是指针），
        // 写成 `if (!check_out(...)) return 2;` 会变成"成功就报 SETUP 失败" ——
        // 第一版就栽在这里，56 个用例里 43 个报 SETUP 失败，看起来像库坏了。
        if (check_out(c, in, flt, out, N, C, K, fH, fW, S, D, oH, oW,
                      in_pixStep, in_widthStep, in_sliceStep,
                      f_pixStep, f_widthStep, f_sliceStep,
                      out_pixStep, out_widthStep, out_sliceStep) != 0)
            return 2;
    }

    // ---- 结果文件 ----
    // pixStep 也在结果文件里：子进程填的 c.in_pixStep 父进程看不到（fork 之后
    // 地址空间是分开的），想让用例行可自证"这一格到底喂了什么"就得传回来。
    // 两种 buffer 模式的格子数**合并**统计，所以格数是原来的两倍。
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%d %d %ld %ld %.6e\n", in_pixStep, f_pixStep,
                     c.acc_ok, c.acc_bad, c.acc_worst); fclose(f); }
    return 0;
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(Case& c)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) {
        FILE* dn = freopen("/dev/null", "w", stderr); (void)dn;
        int r = run_case(c);
        _exit(r == 2 ? 3 : 0);
    }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0; double worst = 0;
    int ips = -1, fps = -1;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { if (fscanf(f, "%d %d %ld %ld %lf", &ips, &fps, &ok, &bad, &worst) != 5) { ok = bad = 0; } fclose(f); }
    const char* nm = g_entries[c.entry].name;
    char tag[160];
    snprintf(tag, sizeof(tag), "N=%d %dx%d C=%d K=%d f=%dx%d s=%d d=%d pix=%d/%d",
             c.N, c.H, c.W, c.C, c.K, c.fH, c.fW, c.stride, c.dil, ips, fps);
    if (WIFSIGNALED(st)) { g_crash++; printf("  %-38s %s  CRASH\n", nm, tag); return; }
    if (WEXITSTATUS(st) == 3) { g_crash++; printf("  %-38s SETUP 失败\n", nm); return; }
    if (bad > 0) { g_bad++; printf("  %-38s %s  FAIL %ld/%ld 格错, 最差 %.3e\n",
                                    nm, tag, bad, ok + bad, worst); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHW（layers_c）no_padding 卷积：%d 个入口 x 多组形状（附录 CE）\n", N_ENTRY);
    printf("内核名全部写全、走函数指针表；判据：后向误差，逐格统计\n");
    printf("每个入口只喂**符合它自己契约**的形状（对齐宽度 / C4 / C3 / batch）\n");
    printf("每个用例**跑两遍**：传空指针（内部 malloc）与传 &buffer（调用方给缓冲）\n");
    printf("  —— 生产恒走前一条（use_buffer 恒 false，见附录 CU.9）；\n");
    printf("     原来只跑后一条，也就是**生产从不走的那条分支**\n\n");

    for (int e = 0; e < N_ENTRY; e++) {
        const Entry& en = g_entries[e];
        printf("%s  (align=%d, kind=%d)\n", en.name, en.align, en.kind);
        int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        Case c; memset(&c, 0, sizeof(c));
        c.entry = e; c.N = 1; c.H = 20; c.W = 20; c.C = 8; c.K = 8;
        c.fH = 3; c.fW = 3; c.stride = 1; c.dil = 1; c.diff_pixstep = 0;
        switch (en.kind) {
        case 1: c.fH = 1; c.fW = 1; break;                       // kernel1x1
        case 2: c.C = 4; break;                                  // 要求 in_C == 4
        case 4: c.C = 3; break;                                  // C3：in_C <= 4
        case 3: c.N = 2; break;                                  // batch
        default: break;
        }
        one(c);
        c.stride = 2; one(c); c.stride = 1;
        c.K = 4; one(c); c.K = 8;
        // same_or_notsame 那一族：再试一次 in/filter 的 pixelStep 真的不同
        int ndiff = 0;
        if (strstr(en.name, "same_or_notsame")) { c.diff_pixstep = 1; one(c); ndiff = 1; }
        printf("  -> %d 个用例：对 %d，错 %d，崩 %d%s\n",
               g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0,
               ndiff ? "（含 1 个 in/filter pixelStep 不同的）" : "");
        printf("\n");
    }
    printf("共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
