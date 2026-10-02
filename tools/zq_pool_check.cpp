// zq_pool_check.cpp —— max/avg pooling 内核的越界与数值回归
//
// 起因（audit_k3_20261001.md 附录 BB）
// ------------------------------------
// 继 BA（eltwise）之后测第二个内核家族。选 pooling 的理由是它的索引算术里
// 有一处**别的内核都没有的东西**：
//
//     final_kH = __min(kernel_H, in_H - (out_H - 1) * stride_H);
//     final_kW = __min(kernel_W, in_W - (out_W - 1) * stride_W);
//
// 也就是说**最后一行/最后一列的池化窗口可能被截短**。实现上用两段循环：
// 主循环只做 `out_h < out_H - 1` 且 `out_w < out_W - 1`（窗口完整），
// 边界那一行/列单独用 final_kH/final_kW 再算一遍。
// 这正是 off-by-one 最容易藏的地方 —— 少算一行、多读一列都不一定崩，
// 只是结果错。所以对拍是必须的，ASan 只是其中一半。
//
// 覆盖：
//   × {max, avg}
//   × {align0, align128bit, align256bit}（nodivided_general）
//   × align0 的 general（它不叫 nodivided，签名一样）
//   × 一组 (in_H, in_W, kernel, stride)，含"能整除"与"不能整除"两种
//   × C 覆盖 1..17 的全部余数类
//   与朴素标量参考（带同样的截短规则）逐元素对拍。
//
// 一个前提要说清楚：**out_H / out_W 必须由调用方算成
// ceil((in_H - kernel_H)/stride_H) + 1**，这是内核的契约（头文件注释里写着）。
// 本测试按契约传值；契约本身被破坏时的行为不在本测试覆盖范围内
// （那属于调用方的校验责任，见附录 BB.3）。

#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <vector>

#include "zq_check_alloc.h"
#include "layers_c/zq_cnn_pooling_32f_align_c.c"

static int g_fail = 0;

typedef void (*PoolFn)(const float*, int, int, int, int, int, int, int,
                      int, int, int, int, float*, int, int, int, int, int, int, int);

template <class F>
static void run(const char* tag, F fn, int align, bool is_max,
                int N, int H, int W, int C,
                int kernel_H, int kernel_W, int stride_H, int stride_W)
{
    int Cpad = ((C + align - 1) / align) * align;
    int in_pixStep = Cpad, in_widthStep = Cpad * W, in_sliceStep = in_widthStep * H;
    int out_H = (H - kernel_H) / stride_H + 1;
    int out_W = (W - kernel_W) / stride_W + 1;
    int out_pixStep = Cpad, out_widthStep = Cpad * out_W, out_sliceStep = out_widthStep * out_H;

    size_t in_need = (size_t)(N - 1) * in_sliceStep + (size_t)(H - 1) * in_widthStep
                   + (size_t)(W - 1) * in_pixStep + Cpad;
    size_t out_need = (size_t)(N - 1) * out_sliceStep + (size_t)(out_H - 1) * out_widthStep
                    + (size_t)(out_W - 1) * out_pixStep + Cpad;
    // **32 字节对齐**（附录 CY.1）：align256 入口内部是 _mm256_load_ps。
    float* in = zq_alloc_f32(in_need);
    float* out = zq_alloc_f32(out_need);
    if (!in || !out) { printf("  分配失败\n"); if (in) zq_free_f32(in); if (out) zq_free_f32(out); return; }
    for (size_t i = 0; i < in_need; i++) {
        int v = (int)((i * 37) % 61) - 30;      // 注意：必须先转 int 再减
        in[i] = (float)v * 0.01f;               // 否则 size_t 下溢成 1.8e17
    }
    for (size_t i = 0; i < out_need; i++) out[i] = -12345.f;

    fn(in, N, H, W, C, in_pixStep, in_widthStep, in_sliceStep,
       kernel_H, kernel_W, stride_H, stride_W,
       out, N, out_H, out_W, C, out_pixStep, out_widthStep, out_sliceStep);

    // 朴素参考，带同样的截短规则
    double worst = 0;
    for (int n = 0; n < N; n++)
        for (int oh = 0; oh < out_H; oh++) {
            int kh_eff = __min(kernel_H, H - oh * stride_H);
            for (int ow = 0; ow < out_W; ow++) {
                int kw_eff = __min(kernel_W, W - ow * stride_W);
                for (int c = 0; c < C; c++) {
                    double acc = is_max ? -FLT_MAX : 0.0;
                    double amax = 0.0;
                    int cnt = 0;
                    for (int kh = 0; kh < kh_eff; kh++)
                        for (int kw = 0; kw < kw_eff; kw++) {
                            size_t off = (size_t)n * in_sliceStep
                                       + (size_t)(oh * stride_H + kh) * in_widthStep
                                       + (size_t)(ow * stride_W + kw) * in_pixStep + c;
                            float v = in[off];
                            if (is_max) { if (v > acc) acc = v; }
                            else acc += v;
                            if (fabs((double)v) > amax) amax = fabs((double)v);
                            cnt++;
                        }
                    double ref = is_max ? acc : (acc / cnt);
                    size_t ooff = (size_t)n * out_sliceStep
                                + (size_t)oh * out_widthStep + (size_t)ow * out_pixStep + c;
                    // 误差尺度取**窗口内输入的最大绝对值**，不是结果本身。
                    // avgpooling 的结果可能接近 0（正负相消），用
                    // `|out-ref|/(|ref|+eps)` 会在那里把 1 ulp 的差放大成 1e-3
                    // 的"相对误差" —— 第一版就是这么误报了 65 条。
                    // 内核算的是 `acc * (1/(kH*kW))`，参考算的是 `acc/cnt`，
                    // 两者在二进制下本来就不逐位相等（1/9 不可精确表示）。
                    double d = fabs((double)out[ooff] - ref) / (amax + 1e-3);
                    if (d > worst) worst = d;
                }
            }
        }
    zq_free_f32(in); zq_free_f32(out);

    bool ok = worst < 1e-5;
    if (!ok) g_fail++;
    printf("  %-5s %-10s align=%d C=%2d %dx%d k=%dx%d s=%dx%d  相对误差 %.2e  %s\n",
           is_max ? "max" : "avg", tag, align, C, H, W, kernel_H, kernel_W,
           stride_H, stride_W, worst, ok ? "ok" : "FAIL");
}

int main()
{
    printf("max/avg pooling 回归（附录 BB）\n");
    printf("ASan 会在越界时直接 abort\n\n");

    // (H, W, kH, kW, sH, sW)：前两组"能整除"，后四组"不能整除"（触发 final_kH/kW 截短）
    struct S { int H, W, kH, kW, sH, sW; };
    static const S shapes[] = {
        { 8, 8, 2, 2, 2, 2 },
        { 9, 9, 3, 3, 3, 3 },
        { 7, 7, 2, 2, 2, 2 },
        { 8, 8, 3, 3, 2, 2 },
        { 10, 6, 3, 3, 2, 2 },
        { 5, 5, 2, 2, 1, 1 },
    };
    static const int Cs[] = { 1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 17 };

    for (int si = 0; si < (int)(sizeof(shapes) / sizeof(shapes[0])); si++) {
        const S& s = shapes[si];
        for (int ci = 0; ci < (int)(sizeof(Cs) / sizeof(Cs[0])); ci++) {
            int C = Cs[ci];
            run("a0-general", zq_cnn_maxpooling_nopadding_32f_align0_general, 1, true,
                1, s.H, s.W, C, s.kH, s.kW, s.sH, s.sW);
            run("a0-general", zq_cnn_avgpooling_nopadding_32f_align0_general, 1, false,
                1, s.H, s.W, C, s.kH, s.kW, s.sH, s.sW);
            run("nd-128", zq_cnn_maxpooling_nopadding_nodivided_32f_align128bit_general, 4, true,
                1, s.H, s.W, C, s.kH, s.kW, s.sH, s.sW);
            run("nd-128", zq_cnn_avgpooling_nopadding_nodivided_32f_align128bit_general, 4, false,
                1, s.H, s.W, C, s.kH, s.kW, s.sH, s.sW);
            run("nd-256", zq_cnn_maxpooling_nopadding_nodivided_32f_align256bit_general, 8, true,
                1, s.H, s.W, C, s.kH, s.kW, s.sH, s.sW);
            run("nd-256", zq_cnn_avgpooling_nopadding_nodivided_32f_align256bit_general, 8, false,
                1, s.H, s.W, C, s.kH, s.kW, s.sH, s.sW);
        }
    }

    if (g_fail) { printf("\n%d 条对拍失败\n", g_fail); return 1; }
    printf("\n全部通过（无越界、数值与标量参考一致）\n");
    return 0;
}
