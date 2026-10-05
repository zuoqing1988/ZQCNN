// LSTM_TF 内核的**独立参考实现**比对（附录 IX）
//
// 为什么现在才写
// --------------
// `LSTM_TF` 是 36 种层类型里**最后一个 UNUSED**（没有任何随仓模型跑得到），
// 此前只有 `SamplesZQCNN/SampleLSTMTFCalib` 这个**标定装置**（附录 IN）：
// 它回答"哪个张量能改变输出"，**不回答"算得对不对"** —— 没有参考实现，
// 也就没有任何一个数字可以判成败。IN.5 当时留下两条未解释的观察：
//   * `fw_b_G` 单独点亮，输出全 0
//   * `fw_b_O=5` 让 t>=1 变大，但 t=0 不变
//
// 语义（从 `zq_cnn_lstm_32f_align_c_raw.h` 与它引用的 TF 1.9 伪码逐字读出来）
// ------------------------------------------------------------------------
// 对每个 n（batch）和每个 q（hidden 维）：
//     I  = b_I + x_t·Wxc_I + h_{t-1}·Whc_I
//     F  = b_F + x_t·Wxc_F + h_{t-1}·Whc_F + forget_bias
//     o  = b_O + x_t·Wxc_O + h_{t-1}·Whc_O
//     ci = b_G + x_t·Wxc_G + h_{t-1}·Whc_G
//     i  = sigmoid(I);  f = sigmoid(F);  ci = tanh(ci)
//     cs = ci*i + cs_{t-1}*f;  cs = clip(cs, cell_clip)
//     o  = sigmoid(o);  co = tanh(cs);  h = co*o
//     out[t][q] = h
// n 维开始时 h = 0、cs_{t-1} = 0；时间顺序由 is_fw 决定（反向是 t = W-1 .. 0），
// 但**输出仍然写在各自的 ti 位置**（不是按处理顺序重排）。
//
// **注意 `ci` 用的是 `b_G`、`o` 用的是 `b_O`**，而内核的形参顺序是
// `b_I, b_F, b_O, b_G`（`zq_cnn_lstm_32f_align_c.h:64`）——
// 权重文件也是这个顺序（`LoadBinary_NCHW` 里依次读
// xc_I, xc_F, xc_O, xc_G, hc_I, hc_F, hc_O, hc_G, b_I, b_F, b_O, b_G）。
// 第一版按 TF 论文的 i/f/c/o 顺序写成 `b_I, b_F, b_G, b_O`，
// 编译直接报"invalid conversion from 'const float*' to 'int'" ——
// 形参个数对不上，**编译器替我挡住了**，比运行时对不上强得多。
//
// 覆盖
// ----
// 10 组：(C, hidden, W, N) x 正/反向 x 「共用 buffer」复用分支 x 很紧的 cell_clip。
// 复用分支此前零覆盖：`*buffer_len` 够大时内核**不重新分配**，直接把一块切成 9 段
// （h/cell/cs/I/F/cs_prev/ci/co/o），这里连跑两遍把两条路都走一遍。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include "zq_cnn_lstm_32f_align_c.h"

static int g_fail = 0;

// 64 字节对齐缓冲（附录 IW.13：`_mm256_load_ps` 要求 32 字节对齐；
// 自己造缓冲喂 SIMD 内核时必须显式对齐，且 ASan 与 UBSan 两条轴都要跑）
struct AlignedBuf {
    float* base = 0;
    float* p = 0;
    size_t n = 0;
    void alloc(size_t count)
    {
        free_raw();
        n = count;
        base = (float*)malloc(n * sizeof(float) + 64);
        if (!base) { printf("  malloc 失败\n"); exit(2); }
        size_t m = (size_t)((uintptr_t)base % 64);
        p = base + (m ? (64 - m) / sizeof(float) : 0);
        memset(p, 0, n * sizeof(float));
    }
    void free_raw() { if (base) free(base); base = 0; p = 0; n = 0; }
    float& operator[](size_t i) { return p[i]; }
};

static float rnd(unsigned& s)
{
    s = s * 1664525u + 1013904223u;
    return (float)((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

static double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

// 12 块权重，顺序与内核形参**逐字一致**：
//   0..3 : xc_I, xc_F, xc_O, xc_G   —— [hidden][in_C]
//   4..7 : hc_I, hc_F, hc_O, hc_G   —— [hidden][hidden]
//   8..11: b_I,  b_F,  b_O,  b_G    —— [hidden]
struct Weights {
    std::vector<float> blk[12];
    int hidden, in_C;
    const float& w(int b, int q, int c) const { return blk[b][(size_t)q * (b < 4 ? in_C : hidden) + c]; }
    const float& bias(int b, int q) const { return blk[b][q]; }
};

static void fill_weights(Weights& W, int hidden, int in_C, unsigned& s)
{
    W.hidden = hidden; W.in_C = in_C;
    for (int b = 0; b < 12; b++) {
        const int K = (b < 4) ? in_C : (b < 8 ? hidden : 1);
        W.blk[b].resize((size_t)hidden * K);
        for (size_t i = 0; i < W.blk[b].size(); i++) W.blk[b][i] = rnd(s) * 0.5f;
    }
}

static void ref_lstm(const std::vector<float>& in, const Weights& W,
                     int N, int in_C, int W_len, bool is_fw,
                     float forget_bias, float cell_clip,
                     std::vector<float>& out)
{
    const int hidden = W.hidden;
    out.assign((size_t)N * W_len * hidden, 0.0f);
    std::vector<double> h(hidden), cs(hidden), I(hidden), F(hidden), O(hidden), CI(hidden);
    for (int n = 0; n < N; n++) {
        for (int q = 0; q < hidden; q++) { h[q] = 0.0; cs[q] = 0.0; }
        for (int step = 0; step < W_len; step++) {
            const int ti = is_fw ? step : W_len - 1 - step;
            const float* x = &in[((size_t)n * W_len + ti) * in_C];
            for (int q = 0; q < hidden; q++) {
                I[q]  = W.bias(8,  q);      // b_I
                F[q]  = W.bias(9,  q);      // b_F
                O[q]  = W.bias(10, q);      // b_O
                CI[q] = W.bias(11, q);      // b_G -> ci
                for (int c = 0; c < in_C; c++) {
                    const double xv = x[c];
                    I[q]  += (double)W.w(0, q, c) * xv;
                    F[q]  += (double)W.w(1, q, c) * xv;
                    O[q]  += (double)W.w(2, q, c) * xv;
                    CI[q] += (double)W.w(3, q, c) * xv;
                }
                for (int c = 0; c < hidden; c++) {
                    const double hv = h[c];
                    I[q]  += (double)W.w(4,  q, c) * hv;
                    F[q]  += (double)W.w(5,  q, c) * hv;
                    O[q]  += (double)W.w(6,  q, c) * hv;
                    CI[q] += (double)W.w(7,  q, c) * hv;
                }
                F[q] += (double)forget_bias;
            }
            for (int q = 0; q < hidden; q++) {
                const double cs_prev = cs[q];
                const double i  = sigmoid(I[q]);
                const double f  = sigmoid(F[q]);
                const double ci = tanh(CI[q]);
                double cur = ci * i + cs_prev * f;
                if (cur >  (double)cell_clip) cur =  (double)cell_clip;
                if (cur < -(double)cell_clip) cur = -(double)cell_clip;
                const double o  = sigmoid(O[q]);
                const double co = tanh(cur);
                cs[q] = cur;
                h[q] = co * o;
                out[((size_t)n * W_len + ti) * hidden + q] = (float)h[q];
            }
        }
    }
}

struct Block { AlignedBuf buf; int pixStep; int widStep; int sliStep; };

// 权重块按 [hidden][K] 摆，pixelStep 补到 8 的倍数（内核按 `q*pixelStep + i` 取）。
// **偏置块不能这么摆**：内核取的是 `b_G_data[q]` —— **步长 1，连续**。
// 这一点对应层里 `fw_b_G->ChangeSize(1,1,1,hidden_dim)`：N=H=W=1、C=hidden_dim，
// `GetFirstPixelPtr()` 就是一个**连续**的 hidden_dim 长数组。
// 第一版把偏置也按 pixelStep=8 摆（与权重块共用同一个 pack），
// 于是 q=1 读到的是填充里的 0 —— 输出变成正数而参考是负数，
// 差得很远但**看起来像"库算错了"**（附录 IX.2）。
static void pack_w(const std::vector<float>& src, int hidden, int K, Block& b)
{
    b.pixStep  = (K + 7) / 8 * 8;
    b.widStep  = b.pixStep;            // H = W = 1
    b.sliStep  = b.pixStep;
    b.buf.alloc((size_t)b.pixStep * hidden);
    for (int q = 0; q < hidden; q++)
        for (int c = 0; c < K; c++)
            b.buf[(size_t)q * b.pixStep + c] = src[(size_t)q * K + c];
}

static void pack_b(const std::vector<float>& src, int hidden, Block& b)
{
    b.pixStep = b.widStep = b.sliStep = 1;   // 连续
    b.buf.alloc((size_t)hidden);
    for (int q = 0; q < hidden; q++) b.buf[q] = src[q];
}

static void run(int in_C, int hidden, int W_len, int N, bool is_fw,
                float forget_bias, float cell_clip, bool share_buffer)
{
    unsigned s = 20262001u + (unsigned)(in_C * 131 + hidden * 17 + W_len * 7 + (is_fw ? 3 : 11));
    std::vector<float> in((size_t)N * W_len * in_C);
    for (size_t i = 0; i < in.size(); i++) in[i] = rnd(s);
    Weights W;
    fill_weights(W, hidden, in_C, s);

    const int in_pix = (in_C + 7) / 8 * 8;
    const int out_pix = (hidden + 7) / 8 * 8;
    AlignedBuf inbuf, outbuf;
    inbuf.alloc((size_t)in_pix * W_len * N);
    outbuf.alloc((size_t)out_pix * W_len * N);
    for (int n = 0; n < N; n++)
        for (int t = 0; t < W_len; t++)
            for (int c = 0; c < in_C; c++)
                inbuf[((size_t)n * W_len + t) * in_pix + c] = in[((size_t)n * W_len + t) * in_C + c];
    // 输出先填 -777：任何没被写到的格子都会露出来
    for (size_t i = 0; i < outbuf.n; i++) outbuf[i] = -777.0f;

    Block b[8];
    for (int i = 0; i < 8; i++) pack_w(W.blk[i], hidden, i < 4 ? in_C : hidden, b[i]);
    Block bb[4];
    for (int i = 0; i < 4; i++) pack_b(W.blk[8 + i], hidden, bb[i]);

    // 内核的形参是 `void** buffer`，`*buffer` 才是**数据指针** ——
    // 和 `ZQ_CNN_Net::Forward` 里 `layers[i]->buffer = &(_buffer.data)` 一样。
    // 第一版多包了一层（`buffer` 存 `&shared`，再传 `&buffer`），
    // 于是 `*buffer` 变成了**我那个局部变量的地址**，
    // ASan 立刻报「stack-buffer-overflow，写在 shared 之后」——
    // 判据还是对的，只是**装置的接线错了**（附录 IW.14）。
    void* shared = 0;
    __int64 buffer_len = 0;
    if (share_buffer)
    {
        shared = _aligned_malloc(1024 * 1024, 32);
        buffer_len = 1024 * 1024;
    }

    const int passes = share_buffer ? 2 : 1;
    for (int pass = 0; pass < passes; pass++) {
        zq_cnn_lstm_TF_32f_align256bit(
            &inbuf[0], N, W_len, in_C, in_pix, in_pix * W_len,
            &b[0].buf[0], b[0].pixStep, b[0].pixStep,   // xc_I
            &b[1].buf[0], b[1].pixStep, b[1].pixStep,   // xc_F
            &b[2].buf[0], b[2].pixStep, b[2].pixStep,   // xc_O
            &b[3].buf[0], b[3].pixStep, b[3].pixStep,   // xc_G
            &b[4].buf[0], b[4].pixStep, b[4].pixStep,   // hc_I
            &b[5].buf[0], b[5].pixStep, b[5].pixStep,   // hc_F
            &b[6].buf[0], b[6].pixStep, b[6].pixStep,   // hc_O
            &b[7].buf[0], b[7].pixStep, b[7].pixStep,   // hc_G
            &bb[0].buf[0], &bb[1].buf[0], &bb[2].buf[0], &bb[3].buf[0],   // b_I,b_F,b_O,b_G
            &outbuf[0], out_pix, out_pix * W_len,
            hidden, is_fw ? 1 : 0, forget_bias, cell_clip, &shared, &buffer_len);
    }

    std::vector<float> want;
    ref_lstm(in, W, N, in_C, W_len, is_fw, forget_bias, cell_clip, want);

    double worst = 0; long wi = -1; double wg = 0.0, we = 0.0;
    bool unwritten = false;
    for (int n = 0; n < N; n++)
        for (int t = 0; t < W_len; t++)
            for (int q = 0; q < hidden; q++) {
                const float g = outbuf[((size_t)n * W_len + t) * out_pix + q];
                const float e = want[((size_t)n * W_len + t) * hidden + q];
                if (g == -777.0f) unwritten = true;
                const double d = fabs((double)g - (double)e);
                if (d > worst) {
                    worst = d;
                    wi = (long)(((size_t)n * W_len + t) * hidden + q);
                    wg = (double)g; we = (double)e;
                }
            }
    printf("  %-3s C=%-2d hidden=%-2d W=%-2d N=%d fb=%.1f clip=%.1f%s  最大绝对偏差 %-12.6g %s\n",
           is_fw ? "fw" : "bw", in_C, hidden, W_len, N, forget_bias, cell_clip,
           share_buffer ? " [共用buffer x2]" : "", worst,
           (worst > 1e-5 || unwritten) ? "**不符**" : "OK");
    if (worst > 1e-5 || unwritten) {
        printf("        最差下标 %ld = (n=%d,t=%d,q=%d)：库 %.6f / 参考 %.6f%s\n",
               wi, (int)(wi / (W_len * hidden)),
               (int)((wi / hidden) % W_len), (int)(wi % hidden), wg, we,
               unwritten ? "  **有没被写到的格子**" : "");
        g_fail++;
    }

    if (shared) _aligned_free(shared);
    for (int i = 0; i < 4; i++) bb[i].buf.free_raw();
    for (int i = 0; i < 8; i++) b[i].buf.free_raw();
    inbuf.free_raw();
    outbuf.free_raw();
}

int main()
{
    printf("=== LSTM_TF 内核 vs 独立参考实现 ===\n");
    run(1, 1, 4, 1, true,  1.0f, 3.0f, false);
    run(1, 1, 4, 1, false, 1.0f, 3.0f, false);
    run(3, 2, 5, 2, true,  1.0f, 3.0f, false);
    run(3, 2, 5, 2, false, 1.0f, 3.0f, false);
    run(4, 4, 7, 1, true,  1.0f, 3.0f, false);
    run(4, 4, 7, 1, false, 1.0f, 3.0f, false);
    run(8, 3, 6, 2, true,  1.0f, 3.0f, false);
    run(5, 1, 3, 1, false, 0.0f, 3.0f, false);
    run(8, 8, 4, 1, true,  1.0f, 0.5f, false);   // cell_clip 很紧 -> 走 clip 分支
    run(4, 4, 6, 1, true,  1.0f, 3.0f, true);    // 共用 buffer（复用分支）
    run(4, 4, 6, 1, false, 1.0f, 3.0f, true);
    if (g_fail) { printf("LSTM CHECK FAILED (%d 处)\n", g_fail); return 1; }
    printf("LSTM CHECK OK\n");
    return 0;
}
