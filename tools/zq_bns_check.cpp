// zq_bns_check.cpp —— zq_cnn_batchnormscale_mean_var_scale_bias_nchwc 的缺陷钉子
//
// 起因（audit_k3_20261001.md 附录 AY）
// --------------------------------------
// tools/run_msvc_analyze.py 在 ZQCNN/layers_nchwc/zq_cnn_batchnormscale_nchwc_raw.h
// 上报了 C6011（取消对 NULL 指针 a/b 的引用）。查下来是**三件独立的事**：
//
// ① `_aligned_malloc` 没有判 NULL（已修，见该文件里的注释）。
// ② 循环上界用 `ceil_C` 而不是 `in_C`，读模型参数数组时越界：
//
//        int ceil_C = (in_C + zq_mm_align_size - 1)/zq_mm_align_size*zq_mm_align_size;
//        for (c = 0; c < ceil_C; c++) {
//            b[c] = slope_data[c] / sqrt(...);   // slope_data 只有 in_C 个 float
//            a[c] = bias_data[c] - mean_data[c] * b[c];
//        }
//
//    补零向量 a/b 要按 ceil_C 算（主内核整宽读），但**读模型参数不能**。已修。
//
// ③ 主内核 zq_cnn_batchnorm_b_a_nchwc 的索引约定是**从 NCHW 版改过来但没改完**：
//    `c` 循环用 `slice_ptr += in_sliceStep` 跨**整张图**的步长，而 NCHWC 布局里
//    C 是**最内层**维度、通道连续。ASan 实测（in_C=3, H=2, W=3）读到了缓冲区外。
//    **这一条不修** —— 修它等于把这个文件整个重写，而它没有调用方（见下）。
//
// ⚠ ③ 之所以不修：这个内核是**死代码**。全仓除本测试外，没有任何地方引用
//   zq_cnn_batchnormscale_mean_var_scale_bias_nchwc*。真正在跑的 NCHWC 路径是
//   ZQ_CNN_Forward_SSEUtils_NCHWC.h:23 的 BatchNormScaleBias_Compute_b_a，
//   它**自己有一份标量 C++ 实现**，循环上界就是 C、没有越界。
//   文件本身仍被 CMake 的 file(GLOB layers_nchwc/*.c) 编进库。
//   保留本测试是为了把已修的 ①② 钉住：将来有人把这个文件接上（或只当"示例"抄）
//   不会把 bug 一起抄过去。
//
// 覆盖：in_C 取 1..33 的全部余数类 × {nchwc1, nchwc4, nchwc8} 三个变体。
// ①② 不再触发；③ 仍会触发并 abort —— **这是已知未修项，不是回归**。

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
    int imStep = sliceStep * N;
    // 缓冲区大小不是 N*imStep：NCHWC 的布局是 [n][h][w][c]，
    // 最后一个元素的下标是 (N-1)*imStep + (H-1)*sliceStep + (W-1)*widthStep + Cpad-1。
    // 写成 N*imStep 会**少分配**一大截，于是连**我自己的参考实现**都会越界 ——
    // ASan 报出来的是 tools/zq_bns_check.cpp 而不是内核，很有误导性（2026-10-02）。
    size_t need = (size_t)(N - 1) * imStep + (size_t)(H - 1) * sliceStep
                + (size_t)(W - 1) * widthStep + Cpad;

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
    std::vector<float> out(need, 0.f);

    const float eps = 1e-5f;
    if (getenv("ZQBNS_VERBOSE"))
        printf("  [call] %s align=%d N=%d H=%d W=%d C=%d Cpad=%d widthStep=%d "
               "sliceStep=%d imStep=%d need=%d(%.0fB) model=%d(%.0fB)\n",
               name, align, N, H, W, C, Cpad, widthStep, sliceStep, imStep,
               (int)need, need * 4.0, C, C * 4.0);
    fprintf(stderr, "CASE %s align=%d C=%d need=%d model=%d\n",
            name, align, C, (int)need, C);
    fn(&im[0], N, H, W, C, widthStep, sliceStep, imStep,
       &mean[0], &var[0], &scale[0], &bias[0], eps);

    // 标量参考：value = b*value + a,  b = scale/sqrt(var+eps), a = bias-mean*b
    double worst = 0;
    for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                for (int c = 0; c < C; c++) {
                    size_t off = (size_t)n * imStep + (size_t)h * sliceStep
                               + (size_t)w * widthStep + c;
                    float b = scale[c] / sqrtf(var[c] + eps);
                    float a = bias[c] - mean[c] * b;
                    double ref = b * im[off] + a;
                    double d = fabs(out[off] - ref) / (fabs(ref) + 1e-6);
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
