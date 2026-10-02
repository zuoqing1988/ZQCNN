// zq_innerproduct_check.cpp —— zq_cnn_innerproduct_gemm_32f_align*_same_pixstep_batch 回归
//
// 起因（audit_k3_20261001.md 附录 BC / BI）
// ------------------------------------------
// 附录 BC 那次失败，是因为**测试用的形状根本不是生产会用的形状**。
// 唯一的生产调用点在 ZQ_CNN_Forward_SSEUtils.cpp:2417：
//
//     if (out_N >= 16 && filter_N >= 16 && in_pixStep == filter_pixStep)
//         zq_cnn_innerproduct_gemm_32f_align128bit_same_pixstep_batch(
//             in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep,
//             filter_data, filter_N, filter_pixStep, filter_widthStep, filter_sliceStep,
//             out_data, out_N, filter_N,
//             out_sliceStep, out_sliceStep, out_sliceStep, buffer, buffer_len);
//
// 三个关键约定（BC 里我全猜错了）：
//
// ① `out_N >= 16 && filter_N >= 16` —— 生产**只**在这种形状下走 GEMM，
//    其余形状走 `zq_cnn_innerproduct_32f_align*_noborder`。第一版测试用
//    N=1 / filter_N=1..17，等于在生产永远不会出现的形状上调它。
//
// ② **`filter_sliceStep` 是 K = H*W*C（逻辑长度），不是 K*Fpad。**
//    它同时被当作 sgemm 的 `ldb`：`matrix_B_cols = filter_N`、
//    `matrix_A_cols = matrix_A_cols = filter_sliceStep`、`Bt` 的 `ldb` 也是它。
//    第一版传 `K * Fpad`，于是 `lda` 大了 8 倍、im2col 只填了 1/8 的列，
//    sgemm 会读到未初始化的内存。
//
// ③ 三个 out 步长**传的是同一个 `out_sliceStep`**（out 是 [N,1,1,filter_N]），
//    所以 `need_allocate_tmp_out` 取决于 `out_sliceStep != filter_N`。
//
// 覆盖：N ∈ {16,17,20} × filter_N ∈ {16,17,24,33} × (H,W,C) 三组
// × {align128bit, align256bit} × {内部 malloc, 复用 buffer}。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// 不要 `#include` 那个 .c —— g++ 不让（`matrix_A = *buffer;` 是 void* -> float*），
// .c 交给 gcc 编（见 tools/run_zqlib_checks.py 的 EXTRA_*）。

// 声明**照抄 ZQCNN/layers_c/zq_cnn_innerproduct_gemm_32f_align_c.h**，
// 带参数名 —— C 链接不检查 arity，参数少写一个不会报错、只会让实参整体错位
// （附录 BC.4 踩过；g++ 报的 "invalid conversion from void** to int" 恰恰
//  是在说"声明比调用点多了一个 int"，而我当时的第一反应是反的）。
extern "C" {
void zq_cnn_innerproduct_gemm_32f_align128bit_same_pixstep_batch(
    const float* in_tensor4D_data,
    int in_N, int in_H, int in_W, int in_C,
    int in_pixelStep, int in_widthStep, int in_sliceStep,
    const float* filters_data,
    int filter_N,
    int filter_pixelStep, int filter_widthStep, int filter_sliceStep,
    float* out_tensor4D_data,
    int out_N, int out_C,
    int out_pixelStep, int out_widthStep, int out_sliceStep,
    void** buffer, long long* buffer_len);
void zq_cnn_innerproduct_gemm_32f_align256bit_same_pixstep_batch(
    const float* in_tensor4D_data,
    int in_N, int in_H, int in_W, int in_C,
    int in_pixelStep, int in_widthStep, int in_sliceStep,
    const float* filters_data,
    int filter_N,
    int filter_pixelStep, int filter_widthStep, int filter_sliceStep,
    float* out_tensor4D_data,
    int out_N, int out_C,
    int out_pixelStep, int out_widthStep, int out_sliceStep,
    void** buffer, long long* buffer_len);
}

static int g_fail = 0;

template <class F>
static void run(const char* tag, F fn, int align, bool use_buffer,
                int N, int H, int W, int C, int filter_N)
{
    int K = H * W * C;                       // 约定 ②：filter_sliceStep == K
    int in_pixStep = C;                      // 约定 ①：in_pixStep == filter_pixStep
    int in_widthStep = in_pixStep * W;
    int in_sliceStep = in_widthStep * H;
    int out_sliceStep = filter_N;             // 约定 ③：三个 out 步长相同
    (void)align;

    std::vector<float> in((size_t)N * in_sliceStep, 0.f);
    for (size_t i = 0; i < in.size(); i++) {
        int v = (int)((i * 37) % 61) - 30;   // 必须先转 int 再减，否则 size_t 下溢
        in[i] = (float)v * 0.01f;
    }
    std::vector<float> flt((size_t)K * filter_N, 0.f);
    for (size_t i = 0; i < flt.size(); i++) {
        int v = (int)((i * 23 + 7) % 41) - 20;
        flt[i] = (float)v * 0.01f;
    }
    std::vector<float> out((size_t)N * out_sliceStep, -12345.f);

    void* buf = 0;
    long long buf_len = 0;
    if (use_buffer) { buf = 0; buf_len = 0; }   // ZQ_CNN_Net::Buffer 的初始状态

    fn(&in[0], N, H, W, C, in_pixStep, in_widthStep, in_sliceStep,
       &flt[0], filter_N, in_pixStep, in_widthStep, K,
       &out[0], N, filter_N,
       out_sliceStep, out_sliceStep, out_sliceStep,
       use_buffer ? &buf : 0, use_buffer ? &buf_len : 0);

    // 对拍时**两种 filter 布局都试**：
    //   A. filters[k*filter_N + f]  —— k 优先（与 sgemm 的 ldb=filter_N 读法一致）
    //   B. filters[f*K + k]          —— f 优先（与 NCHW 张量 [filter_N][H][W][C] 一致）
    // 到底哪个对，是本测试要判的核心问题 —— 判错了会把"参考实现错"报成"内核错"。
    double worstA = 0, worstB = 0;
    for (int n = 0; n < N; n++)
        for (int f = 0; f < filter_N; f++) {
            double accA = 0, accB = 0;
            for (int k = 0; k < K; k++) {
                double iv = (double)in[(size_t)n * in_sliceStep + k];
                accA += iv * (double)flt[(size_t)k * filter_N + f];   // A: k 优先
                accB += iv * (double)flt[(size_t)f * K + k];           // B: f 优先
            }
            size_t ooff = (size_t)n * out_sliceStep + f;
            double a = fabs((double)out[ooff] - accA) / (fabs(accA) + 1.0);
            double b = fabs((double)out[ooff] - accB) / (fabs(accB) + 1.0);
            if (a > worstA) worstA = a;
            if (b > worstB) worstB = b;
        }
    // 只有"某个布局能对上"才算通过；对不上就报两个数，让人一眼看出内核读的是哪种布局。
    bool ok = (worstA < 1e-4) || (worstB < 1e-4);
    const char* which = (worstA < 1e-4) ? "k优先[k][F]" : (worstB < 1e-4 ? "f优先[F][K]" : "**都不对**");
    if (!ok) g_fail++;
    printf("  %-8s %-7s align=%d N=%2d %dx%dx%d F=%2d  A(k优先)=%.2e B(f优先)=%.2e  %s %s\n",
           tag, use_buffer ? "buf" : "malloc", align, N, H, W, C, filter_N,
           worstA, worstB, ok ? "ok" : "FAIL", which);
    fflush(stdout);
    // buffer 路径的**所有权在调用方**：内核把它存进 *buffer 就不再管了
    // （生产里那是 ZQ_CNN_Net::Buffer::data，随 net 一起活着）。
    // 测试里 buf 是个局部变量，不 free 的话 LeakSanitizer 会在退出时报一堆
    // 假泄漏 —— 第一版就踩了这个，84 KB × 36。
    if (use_buffer && buf != 0)
        free(buf);
}

int main()
{
    printf("innerproduct_gemm same_pixstep_batch 回归（附录 BI）\n");
    printf("ASan 会在越界时直接 abort\n");
    printf("生产调用点的约定：out_N>=16 && filter_N>=16 && in_pixStep==filter_pixStep\n\n");

    static const int Ns[] = { 16, 17, 20 };
    static const int Fs[] = { 16, 17, 24, 33 };
    struct S { int H, W, C; };
    static const S shapes[] = { { 2, 3, 8 }, { 3, 3, 4 }, { 1, 1, 16 } };

    for (int ni = 0; ni < 3; ni++)
        for (int fi = 0; fi < 4; fi++)
            for (int si = 0; si < 3; si++) {
                const S& s = shapes[si];
                int N = Ns[ni], F = Fs[fi];
                run("a128", zq_cnn_innerproduct_gemm_32f_align128bit_same_pixstep_batch,
                    4, false, N, s.H, s.W, s.C, F);
                run("a128", zq_cnn_innerproduct_gemm_32f_align128bit_same_pixstep_batch,
                    4, true, N, s.H, s.W, s.C, F);
                run("a256", zq_cnn_innerproduct_gemm_32f_align256bit_same_pixstep_batch,
                    8, false, N, s.H, s.W, s.C, F);
                run("a256", zq_cnn_innerproduct_gemm_32f_align256bit_same_pixstep_batch,
                    8, true, N, s.H, s.W, s.C, F);
            }

    if (g_fail) { printf("\n%d 条对拍失败\n", g_fail); return 1; }
    printf("\n全部通过（无越界、数值与标量参考一致）\n");
    return 0;
}
