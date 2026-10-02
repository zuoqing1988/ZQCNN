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

// ---- 入口表（附录 DC）----------------------------------------------
// 三个 32f 入口都是**活的**（`ZQ_CNN_Forward_SSEUtils.h` 的 LRN_across_channels
// 包装器会按 align_mode 路由到其中之一）。
// **这道门禁原来只测 align256 那一个** —— align0 与 align128bit 两条入口
// 零覆盖，而 LRN 又是"没有任何 shipped 模型会跑到"的那一类（附录 DB.2），
// 于是那两个符号实际上**只被编译、不被执行、更没有被比对过**。
//
// 名字全部写全、走函数指针表，不做字符串拼接（附录 CA.3）。
typedef void (*F_LRN)(
    int local_size, float alpha, float beta, float k,
    const float* in_tensor4D_data,
    int N, int H, int W, int C,
    int in_pixelStep, int in_widthStep, int in_sliceStep,
    float* out_tensor4D_data,
    int out_pixStep, int out_widthStep, int out_sliceStep);

enum { K_A0 = 0, K_A128 = 1, K_A256 = 2 };
struct Entry { F_LRN fn; int align; const char* name; };
static const Entry g_entries[] = {
  { zq_cnn_lrn_across_channels_32f_align0,     1, "align0"     },
  { zq_cnn_lrn_across_channels_32f_align128bit,4, "align128bit"},
  { zq_cnn_lrn_across_channels_32f_align256bit,8, "align256bit"},
};
static const int N_ENTRY = 3;
static int g_entry = K_A256;   // main 里按入口逐个跑

static void run_case(int N, int H, int W, int C, int local_size,
                     float alpha, float beta, float k)
{
    // zq_mm_align_size：**按入口取**（align0 -> 1 / align128 -> 4 / align256 -> 8）。
    // 第一版这里写死 8，于是只测了 align256 一个入口。
    //
    // **但要说清楚：这个宽度实测下来并不是"承重"的。** 变异测试把三个入口
    // 全部按 8 补齐，门禁**依然全绿** —— 因为 align0 的标量循环只按 C 走，
    // 补齐区被完全忽略；而两个 SIMD 入口的向量化内层循环上界也是 C、尾巴走标量。
    // 所以按入口取宽度是**卫生**（让喂进去的布局符合该入口的约定），
    // 不是**正确性**所系。
    //
    // 反过来说，这道门禁**不能**验证"align0 入口会不会误读补齐区" ——
    // 因为参考实现用的是同一个 pixStep，两者一起错就一起对。
    // 记在附录 DC.3，别把这次变异测试的"没红"当成"宽度无关紧要"的证据。
    const int align = g_entries[g_entry].align;
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

    g_entries[g_entry].fn(
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
    printf("zq_cnn_lrn_across_channels_32f_align 回归（附录 AX + DC）\n");
    printf("ASan 会在越界时直接 abort —— 下面能跑完就说明没越界\n");
    printf("**三个 32f 入口都测**（原来只测 align256，align0/align128bit 零覆盖）\n\n");

    for (g_entry = 0; g_entry < N_ENTRY; g_entry++) {
        const int c0 = g_fail;
        printf("== %s（zq_mm_align_size = %d）==\n",
               g_entries[g_entry].name, g_entries[g_entry].align);
        // 重点：local_size == 1 且 C 不是对齐宽度的倍数
        for (int C = 1; C <= 17; C++)
            run_case(1, 2, 2, C, 1, 1e-4f, 0.75f, 1.0f);
        // 正常形状也要过
        for (int ls = 3; ls <= 9; ls += 2)
            for (int C = 1; C <= 17; C++)
                run_case(1, 2, 3, C, ls, 1e-4f, 0.75f, 1.0f);
        printf("   -> %s\n\n", g_fail == c0 ? "全部通过" : "有失败");
    }

    if (g_fail) { printf("\n%d 条对拍失败\n", g_fail); return 1; }
    printf("\n三个入口全部通过（无越界、数值与标量参考一致）\n");
    return 0;
}
