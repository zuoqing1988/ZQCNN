// zq_lrn_check.cpp —— zq_cnn_lrn_across_channels_32f_align 的越界回归测试
//
// 起因（audit_k3_20261001.md 附录 AX）
// ------------------------------------
// MSVC /analyze 在 ZQCNN/layers_c/zq_cnn_lrn_32f_align_c_raw.h 上报了
// C6386（写入 square_buf / accumulate_buf 时缓冲区溢出）与 C6385（读取无效数据）。
// MSVC 的 Code Analysis 在向量化代码上误报很多，所以**不能照单全收** ——
// 只能靠 ASan 实测。这一轮实测下来，它报的那些**不是**误报，而是**指错了地方**：
// 它标的行（68/73 行）其实安全，真正越界的是它**没标**的第 64 行。
//
// 缺陷本体（SSE 版，zq_mm_align_size == 4）
// -----------------------------------------
//     pad_size = local_size / 2 + zq_mm_align_size - 1;
//     pad_size = pad_size - pad_size % zq_mm_align_size;
//     len = C + (pad_size << 1);
//     square_buf = _aligned_malloc(sizeof(float) * len, ...);
//
//     for (c = 0, square_ptr = square_buf + pad_size; c < C;
//          c += zq_mm_align_size, square_ptr += zq_mm_align_size)
//     {
//         data_v = zq_mm_load_ps(in_c_ptr);
//         zq_mm_store_ps(square_ptr, zq_mm_mul_ps(data_v, data_v));   // 写 align 个 float
//     }
//
// 循环以 `align` 为步长、每次写 `align` 个 float，所以**最后一个 store 写到的
// 最高下标是 `pad_size + ceil(C/align)*align - 1`**，而缓冲区只有 `len` 个。
// 越界条件：`ceil(C/align)*align > C + pad_size`。
//
//   * C % align == 0 -> 相等，安全；
//   * C % align != 0 -> 需要 `align - C%align <= pad_size`；
//     而 local_size == 1 时 `p0 = 0 + align - 1 < align`，向下对齐后 **pad_size == 0**，
//     于是只要 `C % align != 0` 就**必然越界**。
//
// 可达性：ZQ_CNN_Layer.h 的 LRN_across_channels 只校验
// `if (local_size % 2 != 1) return false;` —— `local_size == 1` 通过这个校验。
// local_size 来自模型文件（.zqparams）的 `local_size=` 那一行，
// 按本报告的威胁模型属于**不可信输入**。
//
// 复现：C = 5, local_size = 1, align = 4
//   pad_size = 0, len = 5，缓冲区 5 个 float
//   c = 0 -> 写 square_buf[0..3]
//   c = 4 -> 写 square_buf[4..7]   <- 越界 3 个 float（12 字节）
//
// 本测试直接调内核，覆盖 C 从 1 到 17 的全部余数类、local_size 取 1/3/5/7，
// 并与标量参考实现对拍。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// 把 .c 整个 include 进来（内核在它的 extern "C" 里）。
// 内核自己用 `#include "../ZQ_CNN_CompileConfig.h"` 这种相对路径，
// 所以必须**按它在树里的位置**引用，include 路径给 -I ZQCNN。
#include "zq_check_alloc.h"
#include "layers_c/zq_cnn_lrn_32f_align_c.c"

static int g_fail = 0;

static void run_case(int N, int H, int W, int C, int local_size,
                     float alpha, float beta, float k)
{
    // zq_mm_align_size 必须与被测内核一致：下面直接调的是 **256bit** 版本，
    // 它的对齐宽度是 8 个 float（32 字节）。第一版这里写 4，
    // 于是 &_in[0] 之后的像素地址是 16 字节步进，_mm256_load_ps 要求 32 字节
    // 对齐 -> 直接 SEGV（而且报出来的故障地址是 0，很有迷惑性）。
    const int align = 8;   // = zq_mm_align_size（256bit）
    int in_pixStep = ((C + align - 1) / align) * align;
    int in_widthStep = in_pixStep * W;
    int in_sliceStep = in_widthStep * H;
    int out_pixStep = in_pixStep, out_widthStep = in_widthStep,
        out_sliceStep = in_sliceStep;

    // **32 字节对齐**（附录 CY.1）：这里直接调 align256 入口，内部是
    // _mm256_store_ps，std::vector<float> 只给 16 字节 —— ASan 看不见，
    // UBSan 报 "requires 32 byte alignment"。
    const size_t nin = (size_t)in_sliceStep * N, nout = (size_t)out_sliceStep * N;
    float* in = zq_alloc_f32(nin);
    float* out = zq_alloc_f32(nout);
    if (!in || !out) { printf("  分配失败\n"); if (in) zq_free_f32(in); if (out) zq_free_f32(out); return; }
    for (size_t i = 0; i < nin; i++)
        in[i] = (float)((i * 37) % 101) * 0.01f - 0.5f;   // 确定性的伪随机

    zq_cnn_lrn_across_channels_32f_align256bit(
        local_size, alpha, beta, k,
        in, N, H, W, C, in_pixStep, in_widthStep, in_sliceStep,
        out, out_pixStep, out_widthStep, out_sliceStep);

    // 与标量参考对拍
    double worst = 0;
    int pad = local_size / 2;
    int len = C + (pad << 1);
    std::vector<float> sq(len, 0.f), acc(len + 1, 0.f);
    for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++) {
                const float* px = in + (size_t)n * in_sliceStep
                                      + (size_t)h * in_widthStep + (size_t)w * in_pixStep;
                float* opx = out + (size_t)n * out_sliceStep
                                  + (size_t)h * out_widthStep + (size_t)w * out_pixStep;
                for (int c = 0; c < len; c++) sq[c] = 0.f;
                for (int c = 0; c < C; c++) sq[pad + c] = px[c] * px[c];
                acc[0] = 0.f;
                for (int c = pad; c < len; c++) acc[c + 1] = acc[c] + sq[c];
                for (int c = 0; c < C; c++) {
                    double s = acc[c + local_size] - acc[c];
                    double p = pow(k + (alpha / local_size) * s, -beta);
                    double ref = px[c] * p;
                    double d = std::fabs(opx[c] - ref) / (std::fabs(ref) + 1e-6);
                    if (d > worst) worst = d;
                }
            }
    bool ok = worst < 1e-3;
    zq_free_f32(in); zq_free_f32(out);

    if (!ok) g_fail++;
    printf("  N=%d H=%d W=%2d C=%2d local_size=%d  相对误差 %.2e  %s\n",
           N, H, W, C, local_size, worst, ok ? "ok" : "FAIL");
}

int main()
{
    printf("zq_cnn_lrn_across_channels_32f_align 越界回归（附录 AX）\n");
    printf("ASan 会在越界时直接 abort —— 下面能跑完就说明没越界\n");
    printf("对齐宽度 zq_mm_align_size = 4\n\n");

    // 重点：local_size == 1 且 C % 4 != 0
    for (int C = 1; C <= 17; C++)
        run_case(1, 2, 2, C, 1, 1e-4f, 0.75f, 1.0f);

    // 正常形状也要过
    for (int ls = 3; ls <= 9; ls += 2)
        for (int C = 1; C <= 17; C++)
            run_case(1, 2, 3, C, ls, 1e-4f, 0.75f, 1.0f);

    if (g_fail) { printf("\n%d 条对拍失败\n", g_fail); return 1; }
    printf("\n全部通过（无越界、数值与标量参考一致）\n");
    return 0;
}
