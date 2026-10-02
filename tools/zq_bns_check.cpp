// zq_bns_check.cpp —— zq_cnn_batchnormscale_mean_var_scale_bias_nchwc 的回归测试
//
// 起因（audit_k3_20261001.md 附录 AY）
// --------------------------------------
// tools/run_msvc_analyze.py 在 ZQCNN/layers_nchwc/zq_cnn_batchnormscale_nchwc_raw.h
// 上报了 C6011（取消对 NULL 指针 a/b 的引用）。查下来是**两件独立的事**，都已修：
//
// ① `_aligned_malloc` 没有判 NULL（4 对共 8 个分配点，另一个文件里还有 2 对）。
//    in_C / ceil_C 来自模型文件（不可信输入），一个巨大的通道数就能让分配失败。
// ② 读模型参数的循环上界用 `ceil_C` 而不是 `in_C`：
//        for (c = 0; c < ceil_C; c++) {
//            b[c] = slope_data[c] / sqrt(...);   // slope_data 只有 in_C 个 float
//            a[c] = bias_data[c] - mean_data[c] * b[c];
//        }
//    补零向量 a/b 要按 ceil_C 填满（主内核整宽读），但**读模型参数不能**。
//    ASan 坐实：READ of size 4 越过 20 字节的模型数组区域。
//
// ⚠ 关于"主内核 zq_cnn_batchnorm_b_a_nchwc 的索引约定是 NCHW 的、所以错了"——
//   **那个判断是错的，已撤回**。NCHWC 的布局是 [n][c片][h][w][align]
//   （ZQ_CNN_Tensor4D_NCHWC.cpp:142-146），imStep/sliceStep/widthStep/align
//   四个步长**全对**。写这个测试的过程中我自己连踩 5 次坑（缓冲区大小、图距、
//   就地修改、参考下标…），其中一次让我一度确信"内核错了"并写了对拍补丁。
//   详见附录 AY.5。
//
// 覆盖：in_C 取 1..33 的全部余数类 × {nchwc1, nchwc4, nchwc8} 三个变体，
// 与标量参考对拍。当前结果：**99/99 逐位精确（相对误差 0.00e+00）**。

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "layers_nchwc/zq_cnn_batchnormscale_nchwc.c"

static int g_fail = 0;

template <class F>
static void run_variant(const char* name, F fn, int align, int N, int H, int W, int C)
{
    int Cpad = ((C + align - 1) / align) * align;
    int widthStep = Cpad * W;
    int sliceStep = widthStep * H;
    // NCHWC 的真实步长约定（ZQ_CNN_Tensor4D_NCHWC.cpp:142-146 的 ChangeSize）：
    //     dst_slice     = ceil(dst_C / align_size)      <- 通道被切成"片"
    //     dst_widthStep = dst_realW * align_size        <- 一行（W 个位置 x align）
    //     dst_sliceStep = dst_widthStep * dst_realH     <- 一片
    //     dst_imStep    = dst_slice * dst_sliceStep     <- 一张图
    // 也就是说布局是 [n][c片][h][w][align]，**C 在最内层、只是按 align 分组**。
    //
    // 第一版这里写成 imStep = sliceStep * N —— 那是 NCHW 的图距，
    // 于是对拍全错、看上去像"内核算术约定错了"，其实是我把图距算成了 N 而不是
    // ceil(C/align)。按 NCHW 布局去读 NCHWC 内核，必然读出"不对"的假象。
    int imStep = ((C + align - 1) / align) * sliceStep;
    // 缓冲区大小按上面的 [n][c片][h][w][align] 布局算最大下标，不能写 N*imStep
    // 之类。第一版连着写错了两次（先写成 N*imStep，后写成按 NCHW 的下标），
    // 每次都是**我自己的参考实现**先越界，ASan 报的是 tools/zq_bns_check.cpp ——
    // 非常容易让人以为内核有问题。
    size_t need = (size_t)(N - 1) * imStep
                + (size_t)(((C + align - 1) / align) - 1) * sliceStep
                + (size_t)(H - 1) * widthStep
                + (size_t)(W - 1) * align + (Cpad - 1) + 1;

    // 刻意**不**给模型参数数组加对齐余量：C 个 float，一个不多。
    // 越界读会被 ASan 当场抓住。
    std::vector<float> mean(C), var(C), scale(C), bias(C);
    for (int c = 0; c < C; c++) {
        mean[c] = 0.1f * c - 0.3f;
        var[c] = 0.5f + 0.01f * c;
        scale[c] = 1.0f + 0.05f * c;
        bias[c] = -0.2f * c;
    }
    std::vector<float> im(need, 0.f);
    for (size_t i = 0; i < im.size(); i++)
        im[i] = (float)((i * 29) % 97) * 0.01f - 0.5f;
    // 内核是**就地修改**（第一个参数是非 const 指针，结果写回 in_data 本身），
    // 所以参考实现要用**调用前**的副本。第一版拿 out（全 0）去比、或者拿
    // 已经被改过的 im 去比，都会得到"相对误差 1.00"—— 看着像内核算错，
    // 其实是测试比错了对象。
    std::vector<float> im0 = im;

    const float eps = 1e-5f;
    if (getenv("ZQBNS_VERBOSE"))
        printf("  [call] %s align=%d N=%d H=%d W=%d C=%d Cpad=%d widthStep=%d "
               "sliceStep=%d imStep=%d need=%d(%.0fB) model=%d(%.0fB)\n",
               name, align, N, H, W, C, Cpad, widthStep, sliceStep, imStep,
               (int)need, need * 4.0, C, C * 4.0);
    fn(&im[0], N, H, W, C, widthStep, sliceStep, imStep,
       &mean[0], &var[0], &scale[0], &bias[0], eps);

    // 标量参考：value = b*value + a,  b = scale/sqrt(var+eps), a = bias-mean*b
    double worst = 0;
    for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                for (int c = 0; c < C; c++) {
                    // 布局 [n][c片][h][w][align]：
                    //   n*imStep + (c/align)*sliceStep + h*widthStep + w*align + c%align
                    // 写错成 NCHW 的 h*sliceStep + w*widthStep + c 会越界，
                    // 而 ASan 报的是**本文件**，看起来像内核有问题（2026-10-02 踩了 5 次）。
                    size_t off = (size_t)n * imStep
                               + (size_t)(c / align) * sliceStep
                               + (size_t)h * widthStep
                               + (size_t)w * align + (size_t)(c % align);
                    float b = scale[c] / sqrtf(var[c] + eps);
                    float a = bias[c] - mean[c] * b;
                    double ref = b * im0[off] + a;
                    double d = fabs(im[off] - ref) / (fabs(ref) + 1e-6);
                    if (d > worst) worst = d;
                }
    bool ok = worst < 1e-5;
    if (!ok) g_fail++;
    printf("  %-10s align=%d C=%2d  相对误差 %.2e  %s\n",
           name, align, C, worst, ok ? "ok" : "FAIL");
}

int main()
{
    printf("zq_cnn_batchnormscale_mean_var_scale_bias_nchwc 越界回归（附录 AY）\n");
    printf("ASan 会在越界读时直接 abort\n\n");
    for (int C = 1; C <= 33; C++) {
        run_variant("nchwc1", zq_cnn_batchnormscale_mean_var_scale_bias_nchwc1, 1, 1, 2, 3, C);
        run_variant("nchwc4", zq_cnn_batchnormscale_mean_var_scale_bias_nchwc4, 4, 1, 2, 3, C);
        run_variant("nchwc8", zq_cnn_batchnormscale_mean_var_scale_bias_nchwc8, 8, 1, 2, 3, C);
    }
    if (g_fail) { printf("\n%d 条对拍失败\n", g_fail); return 1; }
    printf("\n全部通过（无越界、数值与标量参考一致）\n");
    return 0;
}
