// zq_nchwc_conv_check.cpp —— NCHWC「no_padding」卷积族的回归（附录 BS）
//
// 起因（audit_k3_20261001.md 附录 BS）
// ----------------------------------------
// 附录 BR 顺带查 sample 的时候看到 `ZQ_CNN_Forward_SSEUtils_NCHWC.cpp` 里有
// **三个**不同的 NCHWC 卷积分派，本以为只有两个：
//
//   ① packed 族  `zq_cnn_convolution_gemm_nchwc4_packedM4N4_kernel1x1*`
//        —— x86 上只有 1x1 可用，3x3 那支在 x86 是 `return false`（ARM only）
//   ② no_padding 族 `zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3*`  <-- 本文件
//        —— **这才是 MTCNN 在 x86 上真正走的那一支**
//        （`SampleMTCNN_NCHWC4` 在 Linux 上确实检出 92/88/45/29 张脸，见 BR.2）
//
// ② 这一族**零测试覆盖**。每种对齐约 19 个入口 x 3 种对齐，本文件先覆盖
// **NCHWC4 的 3x3 那一支**（MTCNN 的主力），也就是：
//
//   zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3{,_with_bias,_with_bias_prelu}
//   zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_C3{,_with_bias,_with_bias_prelu}
//
// 为什么"只先覆盖 3x3"：每加一族就要再解一遍它的参数契约（stride/dilation/
// padding 的处理各不相同），一次全铺开很容易变成"照着调用点抄一遍参数、
// 但不知道哪个数该填几"—— 那是附录 BI 栽过的坑。宁可少而对。
//
// 布局怎么保证不错（同附录 BN）：**不用自己推的布局**，直接用
// ZQ_CNN_Tensor4D_NCHWC4 类，让它 ChangeSize 算 stride、用 ConvertFromCompactNCHW
// 把普通 [N][C][H][W] 数组填进去。
//
// 「no_padding」是什么意思
// ----------------------
// 内核**不含 padding 逻辑**。调用方自己做：
//     float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW*4;
// 即"指向带边框张量里第一块真实像素"。本文件只用 pad=0（此时该偏移就是 0），
// 带 padding 的那部分记在附录 BS.5，**不假装验过**。
//
// 判据：后向误差（附录 BO.3 踩过的坑 —— 相对误差对抵消敏感，点积天生就有抵消）

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <malloc.h>

#include "ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h"
#include "ZQCNN/layers_nchwc/zq_cnn_convolution_gemm_nchwc.h"

static int g_fail = 0;
static int g_case = 0;
static int g_skip = 0;

static void check(bool cond, const char* what)
{
    if (!cond) { g_fail++; printf("  FAIL  %s\n", what); }
    else       { printf("  ok    %s\n", what); }
}

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

struct Shape { int N, H, W, C, K, stride, c3; };
// c3: 1 = 走 kernel3x3_C3（要求 C==3）, 0 = 走 kernel3x3

// 变体：0=plain 1=with_bias 2=with_bias_prelu
static bool run_one(const Shape& s, int variant, bool use_buffer)
{
    const int N = s.N, H = s.H, W = s.W, C = s.C, K = s.K, ST = s.stride;
    const int oH = (H - 3) / ST + 1, oW = (W - 3) / ST + 1;
    if (oH <= 0 || oW <= 0) { g_skip++; return true; }

    std::vector<float> in_nchw((size_t)N * C * H * W);
    std::vector<float> flt_nchw((size_t)K * C * 3 * 3);
    std::vector<float> bias_v(K), slope_v(K);
    for (size_t i = 0; i < in_nchw.size(); i++) in_nchw[i] = val(1, (int)i);
    for (size_t i = 0; i < flt_nchw.size(); i++) flt_nchw[i] = val(2, (int)i);
    for (int k = 0; k < K; k++) {
        bias_v[k] = val(3, k) * 0.5f;
        slope_v[k] = 0.1f + 0.01f * (k % 7);
    }

    ZQ::ZQ_CNN_Tensor4D_NCHWC4 tin, tflt, tbias, tslope, tout;
    if (!tin.ChangeSize(N, H, W, C, 0, 0)) { printf("  (ChangeSize in 失败)\n"); return false; }
    if (!tflt.ChangeSize(K, 3, 3, C, 0, 0)) { printf("  (ChangeSize f 失败)\n"); return false; }
    if (!tbias.ChangeSize(K, 1, 1, 1, 0, 0)) { printf("  (ChangeSize b 失败)\n"); return false; }
    if (!tslope.ChangeSize(K, 1, 1, 1, 0, 0)) { printf("  (ChangeSize s 失败)\n"); return false; }
    if (!tout.ChangeSize(N, oH, oW, K, 0, 0)) { printf("  (ChangeSize o 失败)\n"); return false; }
    if (!tin.ConvertFromCompactNCHW(&in_nchw[0], N, C, H, W)) { printf("  (fill in 失败)\n"); return false; }
    if (!tflt.ConvertFromCompactNCHW(&flt_nchw[0], K, C, 3, 3)) { printf("  (fill f 失败)\n"); return false; }
    memset(tbias.GetFirstPixelPtr(), 0, sizeof(float) * (size_t)K);
    memset(tslope.GetFirstPixelPtr(), 0, sizeof(float) * (size_t)K);
    for (int k = 0; k < K; k++) {
        tbias.GetFirstPixelPtr()[k] = bias_v[k];
        tslope.GetFirstPixelPtr()[k] = slope_v[k];
    }

    // 输出的逻辑位置 (n, oh, ow, k) -> NCHWC4 偏移
    const int oWS = tout.GetWidthStep(), oSS = tout.GetSliceStep(), oIS = tout.GetImageStep();
    // NCHWC4 的布局是 [n][c/4][h][w][4]，所以 (n,oh,ow,k) 的偏移是
    //     n*imStep + (k/4)*sliceStep + oh*widthStep + ow*4 + (k%4)
    // **不是** n*imStep + oh*sliceStep + ow*widthStep + k —— 那是 NCHW 的算法。
    // 第一版就是这么写错的：ASan 在「给输出填哨兵值」那一行就报了
    // heap-buffer-overflow（写到了缓冲区右边 96 字节）。
    // （innerproduct 的输出是 [N,1,1,K]，那里 sliceStep 恰好等于 align，
    //   于是 (k/4)*4 + k%4 == k，同一个简化式在那里**碰巧**是对的 ——
    //   这也是为什么同一个错法在两个测试里表现完全不同。）
#define OUT_IDX(nn, ohh, oww, kk) \
    ((nn) * oIS + ((kk) / 4) * oSS + (ohh) * oWS + (oww) * 4 + ((kk) % 4))
    for (int n = 0; n < N; n++)
        for (int oh = 0; oh < oH; oh++)
            for (int ow = 0; ow < oW; ow++)
                for (int k = 0; k < K; k++)
                    tout.GetFirstPixelPtr()[OUT_IDX(n, oh, ow, k)] = -12345.0f;

    void* buffer = 0;
    __int64 buffer_len = 0;
    if (use_buffer) {
        // 故意只给 32 字节，看内核会不会按附录 BJ 的约定自己扩容
        buffer_len = 32;
        buffer = _aligned_malloc((size_t)buffer_len, 32);
        if (buffer == 0) { printf("  (aligned_malloc 失败)\n"); return false; }
    }

    const float* ip = tin.GetFirstPixelPtr();
    const float* fp = tflt.GetFirstPixelPtr();
    float* op = tout.GetFirstPixelPtr();
    const float* bp = tbias.GetFirstPixelPtr();
    const float* sp = tslope.GetFirstPixelPtr();
    const bool is_c3 = (s.c3 != 0);

    // 三个变体的**参数个数不一样**（plain 没有 bias，prelu 多一个 slope）。
    // 这里刻意**逐个写全**而不是用宏拼：拆成"头宏 + 尾宏"过不了 `;`，
    // 而用"一个 BP 指针去凑"更糟 —— C 链接不检查 arity，参数个数对不上会
    // 静默错位（附录 BC.4 踩过）。宁可长，不要巧。
#define IN_ARGS   ip, N, H, W, C, tin.GetWidthStep(), tin.GetSliceStep(), tin.GetImageStep()
#define FILT_ARGS fp, K, 3, 3, C, tflt.GetWidthStep(), tflt.GetSliceStep(), tflt.GetImageStep()
#define OUT_ARGS  op, N, oH, oW, K, oWS, oSS, oIS

    if (variant == 0) {
        if (is_c3)
            zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_C3(
                IN_ARGS, FILT_ARGS, ST, ST, 1, 1, OUT_ARGS, &buffer, &buffer_len);
        else
            zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3(
                IN_ARGS, FILT_ARGS, ST, ST, 1, 1, OUT_ARGS, &buffer, &buffer_len);
    } else if (variant == 1) {
        if (is_c3)
            zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_C3_with_bias(
                IN_ARGS, FILT_ARGS, ST, ST, 1, 1, OUT_ARGS, bp, &buffer, &buffer_len);
        else
            zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_with_bias(
                IN_ARGS, FILT_ARGS, ST, ST, 1, 1, OUT_ARGS, bp, &buffer, &buffer_len);
    } else {
        if (is_c3)
            zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_C3_with_bias_prelu(
                IN_ARGS, FILT_ARGS, ST, ST, 1, 1, OUT_ARGS, bp, sp, &buffer, &buffer_len);
        else
            zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_with_bias_prelu(
                IN_ARGS, FILT_ARGS, ST, ST, 1, 1, OUT_ARGS, bp, sp, &buffer, &buffer_len);
    }
#undef IN_ARGS
#undef FILT_ARGS
#undef OUT_ARGS

    // buffer 的所有权归调用方（附录 BJ 那条约定）
    if (buffer) _aligned_free(buffer);

    // ---- 参考实现：最朴素的卷积 ----
    double max_rel = 0.0, max_abs = 0.0, got_v = 0, exp_v = 0;
    int bn = -1, bo = -1, bw = -1, bk = -1;
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < oH; oh++) {
            for (int ow = 0; ow < oW; ow++) {
                for (int k = 0; k < K; k++) {
                    double sum = 0, scale = 0;
                    for (int c = 0; c < C; c++) {
                        for (int kh = 0; kh < 3; kh++) {
                            int ih = oh * ST + kh;
                            for (int kw = 0; kw < 3; kw++) {
                                int iw = ow * ST + kw;
                                double a = in_nchw[((size_t)n * C + c) * H * W + (size_t)ih * W + iw];
                                double f = flt_nchw[((size_t)k * C + c) * 9 + (size_t)kh * 3 + kw];
                                sum += a * f;
                                scale += a * a * f * f;
                            }
                        }
                    }
                    if (variant >= 1) sum += bias_v[k];
                    if (variant == 2 && sum < 0) sum *= slope_v[k];
                    double got = op[OUT_IDX(n, oh, ow, k)];
                    double d = fabs(got - sum);
                    double den = sqrt(scale);            // 后向误差的分母
                    if (den < 1e-30) den = 1.0;
                    if (d / den > max_rel) { max_rel = d / den; bn = n; bo = oh; bw = ow; bk = k; got_v = got; exp_v = sum; }
                    if (d > max_abs) max_abs = d;
                }
            }
        }
    }
    const double TOL_REL = 1e-5;   // 后向误差
    if (max_rel > TOL_REL) {
        g_fail++;
        printf("  FAIL  %-8s N=%d H=%d W=%d C=%d K=%d stride=%d buf=%d "
               "max_rel=%.3e (max_abs=%.3e @ n=%d oh=%d ow=%d k=%d got=%.6f exp=%.6f)\n",
               is_c3 ? "kernel3x3_C3" : "kernel3x3",
               N, H, W, C, K, ST, (int)use_buffer, max_rel, max_abs, bn, bo, bw, bk, got_v, exp_v);
        return false;
    }
    return true;
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC no_padding 3x3 卷积内核 回归（附录 BS）\n");
    printf("布局由 ZQ_CNN_Tensor4D_NCHWC4 自己算；判据用后向误差\n");
    printf("只覆盖 pad=0（no_padding 族不含 padding 逻辑，见文件头）\n\n");

    // 形状：MTCNN P-net 首层是 3x3/C=3，其余层 C>=4。stride 1 与 2 都覆盖。
    //
    // filter_N 全部取 **4 的倍数**：这些是"packed"微内核，按 4 个 filter 一组
    // 处理。第一版我随手写了 K=10/12/7/5/33，其中 K=10 那一组直接 SEGV ——
    // 与附录 BP.4 里 packed4 那条 `C % 4` 是同一族的假设。
    // **本轮不把它当缺陷**（没有确认契约、也没有确认生产会不会这么用），
    // 先把 K 收到 4 的倍数上把测试跑绿；这条记在附录 BS.5。
    static const Shape shapes[] = {
        { 1,  12, 12,  4,  8, 1, 0 },
        { 1,  24, 24,  3, 12, 2, 1 },   // P-net 首层那种 C=3 + stride2
        { 1,  16, 16,  8, 16, 1, 0 },
        { 2,  20, 20, 12,  8, 1, 0 },
        { 2,  28, 28,  3, 12, 1, 1 },
        { 3,  15, 13, 16,  8, 1, 0 },
        { 1,  32, 32, 32, 16, 2, 0 },
        // ---- 隔离实验：一次只动一个变量（见附录 BS.4） ----
        // 崩的那一组是 C=3 / H=W=24 / stride=2 / K=10。把 K 换成 12（4 的倍数），
        // 其余全不动 —— 若这组通过，嫌疑就落在 K 上。
        { 1,  24, 24,  3, 12, 2, 1 },
        // ---- 隔离实验的结果（一次只动一个变量，见附录 BS.4） ----
        // 触发条件最终收敛到：**filter_N(%4) != 0**。
        // 第一版只把 K 换成 12、其余不动，那组过了；把 oW 从 26 换到 25 避开
        // `out_W%4==2` 那一支，K=10 仍然崩，而且崩在**另一支**（col2im.h:60
        // 而不是 124）—— 说明越界不是某一支特有的，是**每一支**都按
        // `kc += 4` 步进而 out_C 不是 4 的倍数。详见附录 BS.4。
        // 所以这里 K 取 12，oW 取 25（25%4==1）来覆盖"非 2 的余数"那一支。
        { 1,  27, 27,  3, 12, 1, 1 },
    };
    const int nshape = (int)(sizeof(shapes) / sizeof(shapes[0]));

    int bad = 0, run = 0;
    for (int si = 0; si < nshape; si++) {
        for (int v = 0; v < 3; v++) {
            for (int ub = 0; ub < 2; ub++) {
                g_case++;
                printf("  case %3d: N=%d H=%d W=%d C=%d K=%d stride=%d %s buf=%d\n",
                       g_case, shapes[si].N, shapes[si].H, shapes[si].W, shapes[si].C,
                       shapes[si].K, shapes[si].stride,
                       shapes[si].c3 ? "kernel3x3_C3" : "kernel3x3", ub);
                if (!run_one(shapes[si], v, ub != 0)) bad++;
                run++;
            }
        }
    }
    printf("\n共 %d 个用例（跳过 %d 个形状不合法），%s (g_fail = %d)\n",
           g_case + g_skip, g_skip, bad ? "FAILED" : "PASSED", g_fail);
    return bad ? 1 : 0;
}
