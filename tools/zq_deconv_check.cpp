// DeConvolution **内核**语义装置（附录 IW）
//
// 为什么单独探内核，不走 `ZQ_CNN_Net`
// ------------------------------------
// `SampleUnusedLayerProbe` 里那个装置（IW.1）走完整前向链：参数解析、SetBottomDim、
// GetTopDim、张量对齐、四路分派全在里面，给不出"是哪一层出的错"。
// 这里把范围收到**只剩内核**：自己分配带对齐填充的裸缓冲，自己给
// pixelStep / widthStep / sliceStep，直接调三条 general 内核
// （align0 / align128bit / align256bit）。
//
// 两个装置
// --------
// (1) **索引映射反解**：输入只点亮一个格子 `in[ic0][ih0][iw0]=1`，权重只点亮**一个内存槽**
//     `fdata[oc*f_sliStep + kh*f_widStep + kw*f_pixStep + ic]=1`。于是输出里恰好一个格子
//     = 1，位置 (oc,oh,ow) 反解出 p 的四元组。第一版按 `0..NP-1` 逐个点亮，
//     而 filter 的 pixelStep 是 8 不是 C —— 真实槽位只在特定偏移上，
//     于是「只点亮 7 个 / 54」看着像 bug，其实是**我数错了要点的位置**。
// (2) **逐元素比对**：整幅随机权重 + 明文参考。
//
// 参考语义（由 (1) 的实测读出来，不是猜的）：
//     out[oc][oh][ow] = sum_{ic,kh,kw} w[oc][kh][kw][ic] * in[ic][ih][iw]
//     ih = (oh - pad_top + kh)/stride_H，iw = (ow - pad_left + kw)/stride_W
//     两者都必须整除并落在 [0, in_H) / [0, in_W)
//
// **注意 `w[oc][kh][kw][ic]` 是「张量里」的顺序，不是「权重文件里」的顺序。**
// 文件里是 (num_output, in_channels, kH, kW) —— Caffe / MXNet 的权重 blob；
// `LoadBinary_NCHW` 调 `ConvertFromCompactNCHW(data, OC, IC, KH, KW)`
// 才把它摆成张量的 `[oc][kh][kw][ic]`。两者下标顺序不同，
// 混淆过一次（附录 IW.9），代价是 7 组假阳性。
// 本文件直接调内核、自己摆张量，所以**不碰**那个转换，按张量顺序写就对了。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include "zq_cnn_deconvolution_32f_align_c.h"

typedef void (*Kern)(const float*, int, int, int, int, int, int, int,
                     const float*, int, int, int, int, int, int, int,
                     int, int, int, int,
                     float*, int, int, int, int, int, int, int,
                     int, int, int, int);

static int g_fail = 0;

// 64 字节对齐的 float 缓冲。
//
// **必须对齐**：align256bit 那一路用的是 `_mm256_load_ps`，它要求 32 字节对齐
// （不对齐在 x86 上是 #GP，硬件直接杀进程）。第一版这里用普通 `std::vector<float>`，
// ASan 全绿（ASan 不管对齐），**UBSan 那一轴一跑就报**
//     zq_cnn_deconvolution_32f_align_c_raw.h:129 misaligned address
// 这正是本门禁自己踩的坑：判据里"没有越界读"这一条靠 ASan，
// 而"没有未对齐 SIMD 访问"这一条**只有 UBSan 看得见**（附录 IW.13）。
// 库里的张量由 `ZQ_CNN_Tensor4D_*_Align*` 保证对齐，本装置绕过了那一层，所以得自己补上。
struct AlignedBuf {
    float* base = 0;      // malloc 回来的原始地址
    float* p = 0;         // 对齐后的地址
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
    void zero() { memset(p, 0, n * sizeof(float)); }
    void free_raw() { if (base) free(base); base = 0; p = 0; n = 0; }
    float& operator[](size_t i) { return p[i]; }
};

static void alloc_buf(AlignedBuf& d, int C, int W, int H, int align,
                      int& pix, int& wid, int& sli)
{
    pix = (align > 0) ? (C + align - 1) / align * align : C;
    wid = pix * W;
    sli = wid * H;
    d.alloc((size_t)sli);
}

static float rnd(unsigned& s)
{
    s = s * 1664525u + 1013904223u;
    return (float)((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

// 明文参考
static void ref_deconv(const std::vector<float>& in, const std::vector<float>& w,
                       int C, int H, int W, int OC, int KH, int KW,
                       int strideH, int strideW, int padT, int padL,
                       int outH, int outW, std::vector<float>& out)
{
    out.assign((size_t)OC * outH * outW, 0.0f);
    for (int oc = 0; oc < OC; oc++)
        for (int oh = 0; oh < outH; oh++)
            for (int ow = 0; ow < outW; ow++) {
                double acc = 0;
                for (int kh = 0; kh < KH; kh++) {
                    int nh = oh - padT + kh;
                    if (nh < 0 || (nh % strideH) != 0) continue;
                    int ih = nh / strideH;
                    if (ih >= H) continue;
                    for (int kw = 0; kw < KW; kw++) {
                        int nw = ow - padL + kw;
                        if (nw < 0 || (nw % strideW) != 0) continue;
                        int iw = nw / strideW;
                        if (iw >= W) continue;
                        for (int ic = 0; ic < C; ic++)
                            acc += (double)w[(((size_t)oc * KH + kh) * KW + kw) * C + ic]
                                 * (double)in[(((size_t)ic * H) + ih) * W + iw];
                    }
                }
                out[((size_t)oc * outH + oh) * outW + ow] = (float)acc;
            }
}

// 把 compact NCHW 的 [oc][kh][kw][ic] 铺进带对齐填充的 filter 缓冲
static void pack_filter(const std::vector<float>& w, int OC, int KH, int KW, int C,
                        int f_pix, int f_wid, int f_sli, AlignedBuf& dst)
{
    for (int oc = 0; oc < OC; oc++)
        for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++)
                for (int ic = 0; ic < C; ic++)
                    dst[(size_t)oc * f_sli + kh * f_wid + kw * f_pix + ic] =
                        w[(((size_t)oc * KH + kh) * KW + kw) * C + ic];
}

static void unpack_out(AlignedBuf& buf, int OC, int outH, int outW,
                       int o_pix, int o_wid, std::vector<float>& out)
{
    out.assign((size_t)OC * outH * outW, 0.0f);
    for (int oc = 0; oc < OC; oc++)
        for (int y = 0; y < outH; y++)
            for (int x = 0; x < outW; x++)
                out[((size_t)oc * outH + y) * outW + x] =
                    buf[(size_t)y * o_wid + (size_t)x * o_pix + oc];
}

struct Case { int C, H, W, OC, KH, KW, stride, padT, padL; };

static void one(Kern kern, const char* kname, int align, const Case& c, double& worstOut)
{
    const int outH = (c.H - 1) * c.stride + 1 - ((c.KH - 1) + 1) + 2 * c.padT + 1;
    const int outW = (c.W - 1) * c.stride + 1 - ((c.KW - 1) + 1) + 2 * c.padL + 1;
    if (outH <= 0 || outW <= 0) return;

    const int f_pix = (align > 0) ? (c.C + align - 1) / align * align : c.C;
    const int f_wid = f_pix * c.KW;
    const int f_sli = f_wid * c.KH;
    AlignedBuf fbuf;
    fbuf.alloc((size_t)f_sli * c.OC);

    unsigned s = 20262020u + (unsigned)(c.C * 131 + c.KH * 17 + c.stride);
    std::vector<float> in((size_t)c.C * c.H * c.W), w((size_t)c.OC * c.KH * c.KW * c.C);
    for (size_t i = 0; i < in.size(); i++) in[i] = rnd(s);
    for (size_t i = 0; i < w.size(); i++) w[i] = rnd(s);

    int i_pix, i_wid, i_sli, o_pix, o_wid, o_sli;
    AlignedBuf ibuf, obuf;
    alloc_buf(ibuf, c.C, c.W, c.H, align, i_pix, i_wid, i_sli);
    alloc_buf(obuf, c.OC, outW, outH, align, o_pix, o_wid, o_sli);
    for (int ic = 0; ic < c.C; ic++)
        for (int y = 0; y < c.H; y++)
            for (int x = 0; x < c.W; x++)
                ibuf[(size_t)y * i_wid + (size_t)x * i_pix + ic] = in[((size_t)ic * c.H + y) * c.W + x];
    pack_filter(w, c.OC, c.KH, c.KW, c.C, f_pix, f_wid, f_sli, fbuf);

    kern(&ibuf[0], 1, c.H, c.W, c.C, i_pix, i_wid, i_sli,
         &fbuf[0], c.OC, c.KH, c.KW, c.C, f_pix, f_wid, f_sli, c.stride, c.stride, 1, 1,
         &obuf[0], 1, outH, outW, c.OC, o_pix, o_wid, o_sli, c.padT, c.padT, c.padL, c.padL);

    std::vector<float> got, want;
    unpack_out(obuf, c.OC, outH, outW, o_pix, o_wid, got);
    ref_deconv(in, w, c.C, c.H, c.W, c.OC, c.KH, c.KW, c.stride, c.stride, c.padT, c.padL,
               outH, outW, want);
    double worst = 0; long wi = -1;
    for (size_t i = 0; i < want.size(); i++) {
        double d = fabs((double)got[i] - want[i]);
        if (d > worst) { worst = d; wi = (long)i; }
    }
    if (worst > worstOut) worstOut = worst;
    printf("  %-9s C=%-2d OC=%-2d k=%dx%d s=%d pad=%d HxW=%dx%d -> %dx%d  "
           "最大绝对偏差 %-12.6g %s\n",
           kname, c.C, c.OC, c.KH, c.KW, c.stride, c.padT, c.H, c.W, outH, outW, worst,
           (worst > 1e-4) ? "**不符**" : "OK");
    if (worst > 1e-4) {
        printf("        最差下标 %ld = (oc=%d,oh=%d,ow=%d)：库 %.6f / 参考 %.6f\n",
               wi, (int)(wi / (outH * outW)), (int)((wi / outW) % outH), (int)(wi % outW),
               wi >= 0 ? got[wi] : 0.0, wi >= 0 ? want[wi] : 0.0);
        g_fail++;
    }
    // LSan 也管泄漏（run_zqlib_checks.py 的 B 组是 ASan+LSan）：
    // 这三块是 malloc 出来的，不还回去整道门禁会以"泄漏"判红。
    fbuf.free_raw();
    ibuf.free_raw();
    obuf.free_raw();
}

int main()
{
    static const Case CASES[] = {
        { 1, 3, 3, 1, 3, 3, 1, 0, 0 },
        { 2, 5, 5, 2, 3, 3, 1, 0, 0 },
        { 2, 5, 5, 2, 3, 3, 1, 1, 1 },
        { 2, 7, 7, 1, 3, 3, 1, 0, 0 },
        { 3, 4, 4, 3, 3, 3, 1, 1, 1 },
        { 4, 5, 5, 2, 1, 1, 1, 0, 0 },
        { 4, 4, 4, 2, 3, 3, 2, 1, 1 },
        { 8, 4, 4, 2, 3, 3, 1, 0, 0 },
        { 8, 4, 4, 2, 3, 3, 1, 1, 1 },
        { 4, 6, 6, 3, 3, 3, 2, 1, 1 },
        { 8, 6, 6, 4, 2, 2, 2, 0, 0 },
    };
    const size_t n = sizeof(CASES) / sizeof(CASES[0]);

    double w0 = 0, w128 = 0, w256 = 0;
    printf("=== 三条 general 内核逐元素比对 ===\n");
    for (size_t i = 0; i < n; i++) {
        one(zq_cnn_deconv_with_padding_32f_align0_general, "align0", 0, CASES[i], w0);
        one(zq_cnn_deconv_with_padding_32f_align128bit_general, "align128", 4, CASES[i], w128);
        one(zq_cnn_deconv_with_padding_32f_align256bit_general, "align256", 8, CASES[i], w256);
    }
    printf("最大偏差汇总：align0 %.3g / align128 %.3g / align256 %.3g\n", w0, w128, w256);

    if (g_fail) { printf("DECONV KERNEL PROBE FAILED (%d 处)\n", g_fail); return 1; }
    printf("DECONV KERNEL PROBE OK\n");
    return 0;
}
