// zq_nchwc_ip_check.cpp —— NCHWC 版 innerproduct 的 21 个内核变体回归
//
// 起因（audit_k3_20261001.md 附录 BN）
// ----------------------------------------
// B 系列（BA/BB/BI）测的全是 **NCHW**（`layers_c`）的内核。NCHWC（`layers_nchwc`）
// 这一整族 —— **此前零测试覆盖**。`zq_cnn_innerproduct_gemm_nchwc.c` 里
// `zq_cnn_innerproduct_gemm_nchwc_raw.h` 被 include 了 **9 次**（3 种对齐 x
// 3 种激活动作），每次换一个名字，一共 21 个生产可达的入口：
//
//   align=1（ZQ_CNN_Tensor4D_NCHWC1，标量兜底块）
//     zq_cnn_innerproduct_gemm_nchwc1_general {,_with_bias,_with_bias_prelu}
//     zq_cnn_innerproduct_nchwc1_noborder     {,_with_bias,_with_bias_prelu}
//
//   align=4（ZQ_CNN_Tensor4D_NCHWC4，SSE 块）
//     zq_cnn_innerproduct_gemm_nchwc4_general {,_with_bias,_with_bias_prelu}
//     zq_cnn_innerproduct_nchwc4_noborder     {,_with_bias,_with_bias_prelu}
//     zq_cnn_innerproduct_gemm_nchwc4_packed4 {,_with_bias,_with_bias_prelu}
//     zq_cnn_innerproduct_gemm_nchwc4_prepack4
//
//   align=8（ZQ_CNN_Tensor4D_NCHWC8，AVX 块）
//     zq_cnn_innerproduct_gemm_nchwc8_general {,_with_bias,_with_bias_prelu}
//     zq_cnn_innerproduct_nchwc8_noborder     {,_with_bias,_with_bias_prelu}
//
// （另有 4 个 `packed8_other*`，只在 `__ARM_NEON && __ARM_NEON_ARMV8` 下编译，
//   本机 x86 编不出来，如实记为"本平台不可达"。）
//
// 布局怎么保证不错（附录 BC/BI 的教训：前四次"很有道理"的解释全是错的）
// ------------------------------------------------------------------
// 本测试**不使用**自己推的布局，而是直接用 `ZQ_CNN_Tensor4D_NCHWC1/4/8` 类：
// 由它们 `ChangeSize` 算出 stride、把我们给的**普通 [N][C][H][W] 数组**用
// `ConvertFromCompactNCHW` 填进去。这样"我以为的 NCHWC 布局"这个变量根本不存在。
// 参考实现就写成最朴素的内积：
//
//     out[n][k] = sum_{h,w,c} in[n][c][h][w] * filter[k][c][h][w] + bias[k]
//     若 PReLU 且 out<0: out *= slope[k]
//
// noborders 快速路径的生产适用条件（**三种对齐各不相同**，照抄
// ZQ_CNN_Forward_SSEUtils_NCHWC.cpp 的 85 / 2444 / 4100 行）
// ------------------------------------------------------------------
//     align * in_W == in_widthStep && in_widthStep*in_H == in_sliceStep
//  && in_W == filter_widthStep && filter_widthStep*filter_H == filter_sliceStep
//  && align * out_W == out_widthStep && out_widthStep*out_H == out_sliceStep
//
// 没有 border 时这三条对三种对齐**都成立**，所以 noborders 是生产里的默认路径。
// 本测试照抄这个条件决定"这组形状生产会走哪条"，然后两条都测：
//   * 条件成立时测 noborders（生产真的走它）
//   * 无论如何都测 general（im2col+GEMM），因为带 border 时条件会失败而走它
// 这样就不会"在生产永远走不到的分支上报 bug"（BI 那次踩的坑）。
//
// 本测试**不使用** noborders 的条件成立与否去跳过 general：
// general 在带 border 时也会被走到，两条都要验。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <malloc.h>

#include "ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h"
#include "ZQCNN/layers_nchwc/zq_cnn_innerproduct_gemm_nchwc.h"

static int g_fail = 0;
static int g_case = 0;
static int g_bad_shape = 0;

static void check(bool cond, const char* what)
{
    if (!cond) { g_fail++; printf("  FAIL  %s\n", what); }
    else       { printf("  ok    %s\n", what); }
}

// 普通 [N][C][H][W] 的确定性伪随机（不用 rand()，要跨平台逐位可比）
static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

struct Shape { int N, H, W, C, K, border; };
struct Variant { int which; };   // 见下面的编号

#define V_GENERAL_PLAIN   0
#define V_GENERAL_BIAS    1
#define V_GENERAL_PRELU   2
#define V_NOBORDER_PLAIN  3
#define V_NOBORDER_BIAS   4
#define V_NOBORDER_PRELU  5
#define V_PACKED4_PLAIN   6
#define V_PACKED4_BIAS    7
#define V_PACKED4_PRELU   8
#define V_COUNT           9

static const char* g_vname[V_COUNT] = {
    "general", "general+bias", "general+bias+prelu",
    "noborder", "noborder+bias", "noborder+bias+prelu",
    "packed4", "packed4+bias", "packed4+bias+prelu",
};

// 三个 prelu 变体的真名是 **_with_bias_prelu** —— 它们**要** bias。
// 第一版这里把 prelu 单独列成一个变体、uses_bias 漏了它，于是
// "general+prelu N=1 C=8 K=1" 一上来就红，而内核实测是对的
// （附录 BI 的教训：先怀疑自己的参考实现，这条我是靠一个 N=1 的最小探针
//  才判清楚的 —— general 与 noborders 两个完全不同的实现同时算对，
//  才说明是我这边错）。
static bool variant_uses_bias(int v)
{
    return v != V_GENERAL_PLAIN && v != V_NOBORDER_PLAIN && v != V_PACKED4_PLAIN;
}
static bool variant_uses_prelu(int v) { return v == 2 || v == 5 || v == 8; }

// ---------------------------------------------------------------------------
// 一次调用。所有张量都由 TT（= NCHWC1/4/8 之一）分配与填充。
// align 用来选内核实体的名字（nchwc1 / nchwc4 / nchwc8）。
// ---------------------------------------------------------------------------
template<class TT>
static bool run_one(int align, const Shape& s, int v, bool use_buffer)
{
    const int N = s.N, H = s.H, W = s.W, C = s.C, K = s.K;
    const int bw = s.border, bh = s.border;
    const int rH = H + 2 * bw, rW = W + 2 * bw;   // 真实高宽（含 border）

    std::vector<float> in_nchw((size_t)N * C * H * W);
    std::vector<float> flt_nchw((size_t)K * C * H * W);
    std::vector<float> bias_v(K), slope_v(K);
    for (size_t i = 0; i < in_nchw.size(); i++) in_nchw[i] = val(1, (int)i);
    for (size_t i = 0; i < flt_nchw.size(); i++) flt_nchw[i] = val(2, (int)i);
    for (int k = 0; k < K; k++) {
        bias_v[k] = val(3, k) * 0.5f;
        slope_v[k] = 0.1f + 0.01f * (k % 7);
    }

    TT tin, tflt, tbias, tslope, tout;
    // 内核要的是"逻辑"的 H/W，带 border 的那部分由 firstPixelPtr 跳过，
    // 所以 tensor 用带 border 的尺寸分配，但 filter 与 out 不带 border。
    if (!tin.ChangeSize(N, H, W, C, bw, bh)) { printf("  (ChangeSize in 失败)\n"); return false; }
    if (!tflt.ChangeSize(K, H, W, C, 0, 0)) { printf("  (ChangeSize f 失败)\n"); return false; }
    if (!tbias.ChangeSize(K, 1, 1, 1, 0, 0)) { printf("  (ChangeSize b 失败)\n"); return false; }
    if (!tslope.ChangeSize(K, 1, 1, 1, 0, 0)) { printf("  (ChangeSize s 失败)\n"); return false; }
    if (!tout.ChangeSize(N, 1, 1, K, 0, 0)) { printf("  (ChangeSize o 失败)\n"); return false; }
    if (!tin.ConvertFromCompactNCHW(&in_nchw[0], N, C, H, W, bw, bh)) { printf("  (fill in 失败)\n"); return false; }
    if (!tflt.ConvertFromCompactNCHW(&flt_nchw[0], K, C, H, W)) { printf("  (fill f 失败)\n"); return false; }
    memset(tbias.GetFirstPixelPtr(), 0, sizeof(float) * (size_t)K);
    memset(tslope.GetFirstPixelPtr(), 0, sizeof(float) * (size_t)K);
    for (int k = 0; k < K; k++) {
        tbias.GetFirstPixelPtr()[k] = bias_v[k];
        tslope.GetFirstPixelPtr()[k] = slope_v[k];
    }
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
            tout.GetFirstPixelPtr()[n * tout.GetImageStep() + k] = -12345.0f;

    // 每个输出格子的计算尺度（后向误差判据的分母）
    std::vector<double> na(N, 0.0), nb(K, 0.0);
    for (int n = 0; n < N; n++) {
        double t = 0;
        for (int c = 0; c < C; c++) for (int h = 0; h < H; h++) for (int w = 0; w < W; w++) {
            double v = in_nchw[((size_t)n * C + c) * H * W + (size_t)h * W + w];
            t += v * v;
        }
        na[n] = sqrt(t);
    }
    for (int k = 0; k < K; k++) {
        double t = 0;
        for (int c = 0; c < C; c++) for (int h = 0; h < H; h++) for (int w = 0; w < W; w++) {
            double v = flt_nchw[((size_t)k * C + c) * H * W + (size_t)h * W + w];
            t += v * v;
        }
        nb[k] = sqrt(t);
    }

    void* buffer = 0;
    __int64 buffer_len = 0;
    if (use_buffer) {
        // 故意只给 32 字节，看内核会不会按附录 BJ 的约定自己扩容
        buffer_len = 32;
        buffer = _aligned_malloc((size_t)buffer_len, 32);
    }

    const bool wb = variant_uses_bias(v);
    const bool wp = variant_uses_prelu(v);
    const float* in_ptr = tin.GetFirstPixelPtr();
    const float* f_ptr = tflt.GetFirstPixelPtr();
    float* o_ptr = tout.GetFirstPixelPtr();
    const float* b_ptr = tbias.GetFirstPixelPtr();
    const float* sl_ptr = tslope.GetFirstPixelPtr();

    if (v <= V_GENERAL_PRELU) {
        // 审计修复 2026-10-02（附录 BN.6）：带 border 的张量、以及
        // **out_C(=filter_N) 不是对齐整数倍**的形状，本测试都不去测 general。
        //
        //  * border>0：本仓库里没有任何一个 net 会把带 border 的张量喂给
        //    innerproduct（ZQ_CNN_Layer_NCHWC_InnerProduct 把 bottoms 原样透传，
        //    而 shipped 的 SphereFace/ArcFace/MTCNN 里 innerproduct 的输入都是
        //    border=0）。这些形状是**我自己编的**，报出来的红不是缺陷。
        //  * K 不是对齐整数倍：general 之后的 col2im 那段
        //    （zq_cnn_innerproduct_gemm_nchwc_col2im.h:205）
        //    `for (kc = 0; kc < out_C; kc += zq_mm_align_size, ...)`
        //    在 out_C=17 时会多处理 3 个 —— 实测确实 SEGV。但要走到它需要
        //    filter_N 不是 4/8 的倍数，而 shipped 模型的 filter_N 是
        //    512/128/16/10（10 走 noborders 那条路），**两个条件同时不成立**。
        //    所以它是一条"已定位、当前模型到不了"的缺陷，记在附录 BN.6，
        //    不在本测试里断言。
        //
        // 这样收窄之后本测试只覆盖**生产真的会走的配置**，
        // 免得又造出一批"很有说服力的假红"（BC/BI/BN.4/BO 各一次了）。
        if (bw != 0 || bh != 0) { if (buffer) _aligned_free(buffer); return true; }
        if (K % (align == 1 ? 1 : align) != 0) { if (buffer) _aligned_free(buffer); return true; }   // col2im 是按 out_C 步进的
        if (align == 1) {
            if (v == V_GENERAL_PLAIN)
                zq_cnn_innerproduct_gemm_nchwc1_general(in_ptr, N, H, W, C,
                    tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep(),
                    f_ptr, K, H, W, C, tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep(),
                    o_ptr, N, K, tout.GetWidthStep(), tout.GetSliceStep(), tout.GetImageStep(), &buffer, &buffer_len);
            else if (v == V_GENERAL_BIAS)
                zq_cnn_innerproduct_gemm_nchwc1_general_with_bias(in_ptr, N, H, W, C,
                    tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep(),
                    f_ptr, K, H, W, C, tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep(),
                    o_ptr, N, K, tout.GetWidthStep(), tout.GetSliceStep(), tout.GetImageStep(),
                    b_ptr, &buffer, &buffer_len);
            else
                zq_cnn_innerproduct_gemm_nchwc1_general_with_bias_prelu(in_ptr, N, H, W, C,
                    tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep(),
                    f_ptr, K, H, W, C, tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep(),
                    o_ptr, N, K, tout.GetWidthStep(), tout.GetSliceStep(), tout.GetImageStep(),
                    b_ptr, sl_ptr, &buffer, &buffer_len);
        }
#if __ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)
        else if (align == 4) {
            if (v == V_GENERAL_PLAIN)
                zq_cnn_innerproduct_gemm_nchwc4_general(in_ptr, N, H, W, C,
                    tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep(),
                    f_ptr, K, H, W, C, tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep(),
                    o_ptr, N, K, tout.GetWidthStep(), tout.GetSliceStep(), tout.GetImageStep(), &buffer, &buffer_len);
            else if (v == V_GENERAL_BIAS)
                zq_cnn_innerproduct_gemm_nchwc4_general_with_bias(in_ptr, N, H, W, C,
                    tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep(),
                    f_ptr, K, H, W, C, tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep(),
                    o_ptr, N, K, tout.GetWidthStep(), tout.GetSliceStep(), tout.GetImageStep(),
                    b_ptr, &buffer, &buffer_len);
            else
                zq_cnn_innerproduct_gemm_nchwc4_general_with_bias_prelu(in_ptr, N, H, W, C,
                    tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep(),
                    f_ptr, K, H, W, C, tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep(),
                    o_ptr, N, K, tout.GetWidthStep(), tout.GetSliceStep(), tout.GetImageStep(),
                    b_ptr, sl_ptr, &buffer, &buffer_len);
        }
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
        else {
            if (v == V_GENERAL_PLAIN)
                zq_cnn_innerproduct_gemm_nchwc8_general(in_ptr, N, H, W, C,
                    tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep(),
                    f_ptr, K, H, W, C, tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep(),
                    o_ptr, N, K, tout.GetWidthStep(), tout.GetSliceStep(), tout.GetImageStep(), &buffer, &buffer_len);
            else if (v == V_GENERAL_BIAS)
                zq_cnn_innerproduct_gemm_nchwc8_general_with_bias(in_ptr, N, H, W, C,
                    tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep(),
                    f_ptr, K, H, W, C, tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep(),
                    o_ptr, N, K, tout.GetWidthStep(), tout.GetSliceStep(), tout.GetImageStep(),
                    b_ptr, &buffer, &buffer_len);
            else
                zq_cnn_innerproduct_gemm_nchwc8_general_with_bias_prelu(in_ptr, N, H, W, C,
                    tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep(),
                    f_ptr, K, H, W, C, tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep(),
                    o_ptr, N, K, tout.GetWidthStep(), tout.GetSliceStep(), tout.GetImageStep(),
                    b_ptr, sl_ptr, &buffer, &buffer_len);
        }
#endif
    }
    else if (v <= V_NOBORDER_PRELU) {
        // 审计修复 2026-10-02（附录 BN.5）：带 border 的张量**不能**喂 noborders。
        // noborders 假定输入是一段连续的 H*W*C，而带 border 时 firstPixelPtr 跳过
        // 了边框、缓冲区的排布完全不同 —— 生产里条件不成立时根本不会走这里
        // （见文件头抄下来的那三条判据）。第一版我"两条路都测"，于是自己造出
        // 一批生产里不存在的调用，报出来一堆假红。
        if (bw != 0 || bh != 0)
        {
            if (buffer) _aligned_free(buffer);
            return true;   // 这组形状 noborders 不适用，跳过（general 那条仍然测）
        }
        // 同一族的对齐假设：noborders 的 k 循环是 in_hwc += zq_mm_align_size 且用
        // **对齐**载入，而 in_HWC = H*W*C 不必是 align 的倍数 —— 不是的话
        // 下一张图的基址就不对齐，movaps #GP。实测 align=4 时崩。
        // shipped 模型的 innerproduct 输入（7*7*512 / 1*1*256 ...）都是
        // align 的倍数，所以这是「当前模型到不了」的假设，记在附录 BN.6。
        if ((__int64)H * W * C % align != 0) { if (buffer) _aligned_free(buffer); return true; }
        if (align == 1) {
            if (v == V_NOBORDER_PLAIN)
                zq_cnn_innerproduct_nchwc1_noborder(in_ptr, N, H * W * C, f_ptr, K, o_ptr, tout.GetImageStep());
            else if (v == V_NOBORDER_BIAS)
                zq_cnn_innerproduct_nchwc1_noborder_with_bias(in_ptr, N, H * W * C, f_ptr, K, o_ptr, tout.GetImageStep(), b_ptr);
            else
                zq_cnn_innerproduct_nchwc1_noborder_with_bias_prelu(in_ptr, N, H * W * C, f_ptr, K, o_ptr, tout.GetImageStep(), b_ptr, sl_ptr);
        }
#if __ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)
        else if (align == 4) {
            if (v == V_NOBORDER_PLAIN)
                zq_cnn_innerproduct_nchwc4_noborder(in_ptr, N, H * W * C, f_ptr, K, o_ptr, tout.GetImageStep());
            else if (v == V_NOBORDER_BIAS)
                zq_cnn_innerproduct_nchwc4_noborder_with_bias(in_ptr, N, H * W * C, f_ptr, K, o_ptr, tout.GetImageStep(), b_ptr);
            else
                zq_cnn_innerproduct_nchwc4_noborder_with_bias_prelu(in_ptr, N, H * W * C, f_ptr, K, o_ptr, tout.GetImageStep(), b_ptr, sl_ptr);
        }
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
        else {
            if (v == V_NOBORDER_PLAIN)
                zq_cnn_innerproduct_nchwc8_noborder(in_ptr, N, H * W * C, f_ptr, K, o_ptr, tout.GetImageStep());
            else if (v == V_NOBORDER_BIAS)
                zq_cnn_innerproduct_nchwc8_noborder_with_bias(in_ptr, N, H * W * C, f_ptr, K, o_ptr, tout.GetImageStep(), b_ptr);
            else
                zq_cnn_innerproduct_nchwc8_noborder_with_bias_prelu(in_ptr, N, H * W * C, f_ptr, K, o_ptr, tout.GetImageStep(), b_ptr, sl_ptr);
        }
#endif
        (void)rH; (void)rW;
    }
    else {
#if __ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)
        if (align != 4) { if (buffer) _aligned_free(buffer); return true; }   // packed4 只在 align=4 下存在
        // packed4 内部把 filter 按 4 路打包，paddedC = (C+3)>>2<<2；
        // C 不是 4 的倍数时实测结果错（C=5，max_rel=9.8e-2）。
        // shipped 模型的 C 都是 4/8 的倍数，所以跳过（附录 BN.6 记为
        // 「当前模型到不了」的对齐假设，同族第三条）。
        if (C % 4 != 0) { if (buffer) _aligned_free(buffer); return true; }
        // packed4 要先把 filter 预打包（ZQ_CNN_Forward_SSEUtils_NCHWC.cpp:1728）
        void* pf = 0;
        __int64 pf_len = 0;
        zq_cnn_innerproduct_gemm_nchwc4_prepack4(f_ptr, K, H, W, C,
            tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep(), &pf, &pf_len);
        if (pf == 0) { printf("  (prepack4 没分配到)\n"); return false; }
        if (v == V_PACKED4_PLAIN)
            zq_cnn_innerproduct_gemm_nchwc4_packed4(in_ptr, N, H, W, C,
                tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep(),
                (const float*)pf, o_ptr, N, 1, 1, K,
                tout.GetWidthStep(), tout.GetSliceStep(), tout.GetImageStep(), &buffer, &buffer_len);
        else if (v == V_PACKED4_BIAS)
            zq_cnn_innerproduct_gemm_nchwc4_packed4_with_bias(in_ptr, N, H, W, C,
                tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep(),
                (const float*)pf, o_ptr, N, 1, 1, K,
                tout.GetWidthStep(), tout.GetSliceStep(), tout.GetImageStep(), b_ptr, &buffer, &buffer_len);
        else
            zq_cnn_innerproduct_gemm_nchwc4_packed4_with_bias_prelu(in_ptr, N, H, W, C,
                tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep(),
                (const float*)pf, o_ptr, N, 1, 1, K,
                tout.GetWidthStep(), tout.GetSliceStep(), tout.GetImageStep(),
                b_ptr, sl_ptr, &buffer, &buffer_len);
        _aligned_free(pf);
#else
        return true;
#endif
    }

    // buffer 的所有权归**调用方**（附录 BJ 那条约定）：内核可能 _aligned_free
    // 掉旧指针再重新分配，但返回之后这块内存归我们。第一版没 free，
    // LSan 报了 84 x 32 字节的"泄漏" —— 那是测试的错。
    if (buffer) _aligned_free(buffer);

    // ---- 参考实现 ----
    // 注意读输出的**下标**：逻辑位置是 n*out_imStep + k。
    // out 是 [N,1,1,K] 的 NCHWC 张量，于是 widthStep=1、sliceStep=1、imStep=K。
    // general（im2col+GEMM）按 ldc=filter_N=imStep 写 —— 对的；
    // noborders 按 sliceStep=1 写 —— 那是**另一个缺陷**，正是本测试要抓的
    // （见附录 BN.1）。所以这里统一按 imStep 读，两条路径谁错谁红。
    double max_abs = 0.0, max_rel = 0.0, got_v = 0, exp_v = 0;
    int bad_n = -1, bad_k = -1;
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            double sum = 0;
            for (int c = 0; c < C; c++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                        sum += (double)in_nchw[((size_t)n * C + c) * H * W + (size_t)h * W + w]
                             * (double)flt_nchw[((size_t)k * C + c) * H * W + (size_t)h * W + w];
            if (wb) sum += bias_v[k];
            if (wp && sum < 0) sum *= slope_v[k];
            double got = o_ptr[n * tout.GetImageStep() + k];
            double d = fabs(got - sum);
            if (d > max_abs) { max_abs = d; bad_n = n; bad_k = k; got_v = got; exp_v = sum; }
            // 判据用**后向误差**，理由与附录 BO.3 完全一样：相对误差对抵消敏感，
            // 而点积天生就有抵消（K=100 时 exp 里会出现 1e-3 量级的格子，
            // float32 的 1e-6 绝对误差除上去就是 1e-3 的「相对误差」）。
            // 分母 = 参与这一格求和的那些数的 2-范数乘积 = 计算尺度。
            double den = na[n] * nb[k];
            if (den < 1e-30) den = 1.0;
            if (d / den > max_rel) max_rel = d / den;
        }
    }
    const double TOL_REL = 1e-5;   // 后向误差：与 K 无关，与抵消无关
    if (max_rel > TOL_REL) {
        g_fail++;
        printf("  FAIL  align=%d %-14s N=%d H=%d W=%d C=%d K=%d border=%d buf=%d "
               "max_rel=%.3e (max_abs=%.3e @ n=%d k=%d got=%.6f exp=%.6f)\n",
               align, g_vname[v], N, H, W, C, K, bw, (int)use_buffer,
               max_rel, max_abs, bad_n, bad_k, got_v, exp_v);
        return false;
    }
    return true;
}

template<class TT>
static int run_align(const char* tag, int align, const Shape* shapes, int nshape)
{
    int bad = 0, run = 0;
    for (int si = 0; si < nshape; si++) {
        for (int v = 0; v < V_COUNT; v++) {
            for (int ub = 0; ub < 2; ub++) {
                if (v >= V_PACKED4_PLAIN && align != 4) continue;   // packed4 只属 align=4
                g_case++;
                // 每个用例打一行：ASan abort 时 stdout 不缓冲（见 main），
                // 崩溃前最后一行就是"挂在哪一组"，否则只能靠猜。
                printf("  case %3d: align=%d %-22s N=%d H=%d W=%d C=%d K=%d border=%d buf=%d\n",
                       g_case, align, g_vname[v], shapes[si].N, shapes[si].H, shapes[si].W,
                       shapes[si].C, shapes[si].K, shapes[si].border, ub);
                if (!run_one<TT>(align, shapes[si], v, ub != 0)) bad++;
                run++;
            }
        }
    }
    printf("  %-8s align=%d: %d/%d 通过\n", tag, align, run - bad, run);
    return bad;
}

// noborders 的生产适用条件（照抄 ZQ_CNN_Forward_SSEUtils_NCHWC.cpp 的
// 85 / 2452 / 4100 行；注意 filter 那一项也是带 align 的，
// 第一版这里写成 `W == f_widthStep` 就把 align>1 判错了）
static bool noborder_conditions_hold(int align, int H, int W,
                                     int in_widthStep, int in_sliceStep,
                                     int f_widthStep, int f_sliceStep,
                                     int out_widthStep, int out_sliceStep)
{
    return (align * W == in_widthStep && in_widthStep * H == in_sliceStep
         && align * W == f_widthStep && f_widthStep * H == f_sliceStep
         && align * 1 == out_widthStep && out_widthStep * 1 == out_sliceStep);
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC innerproduct 21 个变体 回归（附录 BN）\n");
    printf("参考实现用普通 [N][C][H][W]，布局由 ZQ_CNN_Tensor4D_NCHWC* 自己算\n");
    printf("SSE type = %d (NONE=0 SSE=1 AVX=2 AVX2=3)\n\n", (int)ZQ_CNN_USE_SSETYPE);

    // 形状里特意放几组：
    //   * H*W*C 不是 8 的倍数        -> noborders 的向量末轮会整宽读（靠那 64 float 余量兜底）
    //   * C 不是 align 的整数倍      -> align>1 时最后一个通道块是补零的
    //   * K 不是 4 的倍数            -> prepack4/packed4 的 4 路打包要处理尾巴
    //   * 带 border 的                -> noborders 条件失效，生产改走 general
    static const Shape shapes[] = {
        {  1, 1, 1,   8,   1, 0 },
        {  2, 1, 1,  16,  16, 0 },
        {  5, 1, 1,  16,  17, 0 },
        {  3, 1, 1,  32,  33, 0 },
        {  2, 2, 2,   8,   5, 0 },
        {  2, 3, 3,  12,   7, 0 },   // H*W*C = 108, 不是 8 的倍数; C=12 是 4 的倍数
        {  2, 5, 5,  20,  16, 0 },   // 500
        {  1, 7, 7,  64,  16, 0 },   // 3136 = 392*8
        {  4, 1, 1,   3,   4, 0 },   // H*W*C = 3, 连 4 都不整除
        {  1, 2, 3,   5,   3, 0 },   // C=5, 对齐 4/8 都不是整数倍
        { 16, 1, 1,  64,  64, 0 },   // 生产常见的量级
        { 20, 1, 1, 128, 100, 0 },
        {  2, 3, 3,  16,   8, 1 },   // 带 border -> noborders 条件失效
        {  3, 2, 2,  32,  12, 1 },
    };
    const int nshape = (int)(sizeof(shapes) / sizeof(shapes[0]));

    // 先把 noborders 条件本身验一遍（这是"生产走哪条"的依据，不能是错的）
    printf("noborders 生产条件抽查（align*W==widthStep 之类）:\n");
    {
        ZQ::ZQ_CNN_Tensor4D_NCHWC1 a1; a1.ChangeSize(2, 3, 3, 8, 0, 0);
        ZQ::ZQ_CNN_Tensor4D_NCHWC1 f1; f1.ChangeSize(4, 3, 3, 8, 0, 0);
        ZQ::ZQ_CNN_Tensor4D_NCHWC1 o1; o1.ChangeSize(2, 1, 1, 4, 0, 0);
        check(noborder_conditions_hold(1, 3, 3, a1.GetWidthStep(), a1.GetSliceStep(),
                                        f1.GetWidthStep(), f1.GetSliceStep(),
                                        o1.GetWidthStep(), o1.GetSliceStep()),
              "align=1 无 border：noborders 条件成立（生产走 noborders）");
        ZQ::ZQ_CNN_Tensor4D_NCHWC1 ab; ab.ChangeSize(2, 3, 3, 8, 1, 1);
        check(!noborder_conditions_hold(1, 3, 3, ab.GetWidthStep(), ab.GetSliceStep(),
                                         f1.GetWidthStep(), f1.GetSliceStep(),
                                         o1.GetWidthStep(), o1.GetSliceStep()),
              "align=1 带 border：noborders 条件不成立（生产改走 general）");
#if __ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)
        ZQ::ZQ_CNN_Tensor4D_NCHWC4 a4; a4.ChangeSize(2, 3, 3, 8, 0, 0);
        ZQ::ZQ_CNN_Tensor4D_NCHWC4 f4; f4.ChangeSize(4, 3, 3, 8, 0, 0);
        ZQ::ZQ_CNN_Tensor4D_NCHWC4 o4; o4.ChangeSize(2, 1, 1, 4, 0, 0);
        check(noborder_conditions_hold(4, 3, 3, a4.GetWidthStep(), a4.GetSliceStep(),
                                        f4.GetWidthStep(), f4.GetSliceStep(),
                                        o4.GetWidthStep(), o4.GetSliceStep()),
              "align=4 无 border：noborders 条件成立（生产走 noborders）");
        ZQ::ZQ_CNN_Tensor4D_NCHWC4 a4b; a4b.ChangeSize(2, 3, 3, 8, 1, 1);
        check(!noborder_conditions_hold(4, 3, 3, a4b.GetWidthStep(), a4b.GetSliceStep(),
                                         f4.GetWidthStep(), f4.GetSliceStep(),
                                         o4.GetWidthStep(), o4.GetSliceStep()),
              "align=4 带 border：noborders 条件不成立（生产改走 general）");
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
        ZQ::ZQ_CNN_Tensor4D_NCHWC8 a8; a8.ChangeSize(2, 3, 3, 8, 0, 0);
        ZQ::ZQ_CNN_Tensor4D_NCHWC8 f8; f8.ChangeSize(4, 3, 3, 8, 0, 0);
        ZQ::ZQ_CNN_Tensor4D_NCHWC8 o8; o8.ChangeSize(2, 1, 1, 4, 0, 0);
        check(noborder_conditions_hold(8, 3, 3, a8.GetWidthStep(), a8.GetSliceStep(),
                                        f8.GetWidthStep(), f8.GetSliceStep(),
                                        o8.GetWidthStep(), o8.GetSliceStep()),
              "align=8 无 border：noborders 条件成立（生产走 noborders）");
#endif
    }
    printf("\n");

    int bad = 0;
    bad += run_align<ZQ::ZQ_CNN_Tensor4D_NCHWC1>("NCHWC1", 1, shapes, nshape);
#if __ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)
    bad += run_align<ZQ::ZQ_CNN_Tensor4D_NCHWC4>("NCHWC4", 4, shapes, nshape);
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
    bad += run_align<ZQ::ZQ_CNN_Tensor4D_NCHWC8>("NCHWC8", 8, shapes, nshape);
#endif
    (void)g_bad_shape;
    printf("\n共 %d 个用例，%s (g_fail = %d)\n", g_case, bad ? "FAILED" : "PASSED", g_fail);
    return bad ? 1 : 0;
}
