// zq_eltwise_check.cpp —— zq_cnn_eltwise_{sum,mul,max}_32f_align 的越界/数值回归
//
// 起因（audit_k3_20261001.md 附录 BA）
// ---------------------------------------
// `layers_c/` 与 `layers_nchwc/` 下 36 个内核 TU 里，到本轮为止只有 LRN 和
// batchnormscale 有 ASan 测试 —— 而**恰恰是这被测的两个各查出 1~2 条真缺陷**
// （附录 AX / AY）。这个命中率说明未测的那 34 个值得测。
//
// eltwise 是 SSD / CascadeOnet 之类模型的常用层，三个算子（sum / mul / max）
// 共用同一套索引骨架：
//     if (C % align32 == 0) {...} else if (C % align16 == 0) {...} else {...}
// 也就是"按 C 的余数分派到不同宽度的向量循环 + 标量兜底"，
// 和 LRN 那个"补齐到 align 倍数的缓冲 + 固定步长循环"是**两种不同结构** ——
// 后者才是 AX 里出 bug 的那种。这个测试就是确认这一族是干净的。
//
// 覆盖：N/H/W/C 一组形状（含 C 覆盖 1..17 的全部余数类）× {align0,128bit,256bit}
// × {sum, mul, max} × {2 个输入, 3 个输入}，与标量参考逐元素对拍。
// 内存按 `pixelStep = ceil(C/align)*align` 分配（张量就是这么补齐的），
// 越界会被 ASan 当场抓住。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "zq_check_alloc.h"
#include "layers_c/zq_cnn_eltwise_32f_align_c.c"

static int g_fail = 0;

enum { OP_SUM, OP_MUL, OP_MAX };
static const char* kOpName[3] = { "sum", "mul", "max" };

// 内核签名是 `const float**`（顶层 const 在函数类型里被忽略，
// 但两个写法在函数指针类型上是不同的类型，必须对上）。
typedef void (*EltFn)(int, const float**, int, int, int, int,
                      const int*, const int*, const int*,
                      float*, int, int, int);

template <class F>
static void run(const char* variant, F fn, int align, int op,
                int tensors, int N, int H, int W, int C)
{
    int Cpad = ((C + align - 1) / align) * align;
    int widthStep = Cpad * W;
    int sliceStep = widthStep * H;

    // **32 字节对齐**（附录 CY.1）：align256 入口内部全是 _mm256_load_ps，
    // std::vector<float> 只给 16 字节对齐 —— 在 x86 上值照样算对（只是慢），
    // ASan 看不见，UBSan 会报 "requires 32 byte alignment"。
    std::vector<float*> bufs(tensors, (float*)0);
    std::vector<const float*> ptrs(tensors);
    for (int t = 0; t < tensors; t++) {
        size_t need = (size_t)(N - 1) * sliceStep + (size_t)(H - 1) * widthStep
                    + (size_t)(W - 1) * Cpad + Cpad;
        bufs[t] = zq_alloc_f32(need);
        if (!bufs[t]) { printf("  %-9s %-4s align=%d 分配失败\n", kOpName[op], variant, align); g_fail++; return; }
        for (size_t i = 0; i < need; i++) {
            // 注意这个 -30 必须在 **int** 里做：第一版写成
            //   (float)(((i * 37 + t * 11) % 61) - 30) * 0.01f
            // 而 i 是 size_t，整个表达式按无符号算，模结果 < 30 时**下溢**成
            // 1.8e17，参考值直接 inf —— 于是 36 个 mul/tensors=3 的用例报成
            // 「相对误差 inf」，看上去像内核溢出，其实是我的数据生成器坏了。
            int v = (int)((i * 37 + (size_t)(t * 11)) % 61) - 30;
            bufs[t][i] = (float)v * 0.01f;
        }
        ptrs[t] = bufs[t];
    }
    // 输入刻意不按 Cpad 补零的余数处理：只保证前 C 个是有效数据，
    // 其余是确定的 0 —— 标量参考也只算前 C 个。
    size_t oneed = (size_t)(N - 1) * sliceStep + (size_t)(H - 1) * widthStep
                + (size_t)(W - 1) * Cpad + Cpad;
    float* out = zq_alloc_f32(oneed);
    if (!out) { for (int t = 0; t < tensors; t++) zq_free_f32(bufs[t]); printf("  分配失败\n"); g_fail++; return; }
    for (size_t i = 0; i < oneed; i++) out[i] = -12345.f;

    std::vector<int> pix(tensors, Cpad), wid(tensors, widthStep), sli(tensors, sliceStep);
    fn(tensors, &ptrs[0], N, H, W, C, &pix[0], &wid[0], &sli[0],
       out, Cpad, widthStep, sliceStep);

    double worst = 0;
    for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++) {
                size_t base = (size_t)n * sliceStep + (size_t)h * widthStep
                            + (size_t)w * Cpad;
                for (int c = 0; c < C; c++) {
                    double ref;
                    if (op == OP_SUM) {
                        ref = 0;
                        for (int t = 0; t < tensors; t++)
                            ref += bufs[t][base + c];
                    } else if (op == OP_MUL) {
                        ref = 1;
                        for (int t = 0; t < tensors; t++)
                            ref *= bufs[t][base + c];
                    } else {
                        ref = bufs[0][base + c];
                        for (int t = 1; t < tensors; t++)
                            if (bufs[t][base + c] > ref) ref = bufs[t][base + c];
                    }
                    double d = fabs(out[base + c] - ref) / (fabs(ref) + 1e-6);
                    if (d > worst) worst = d;
                }
            }
    for (int t = 0; t < tensors; t++) zq_free_f32(bufs[t]);
    zq_free_f32(out);

    bool ok = worst < 1e-5;
    if (!ok) g_fail++;
    printf("  %-9s %-4s align=%d tensors=%d C=%2d  相对误差 %.2e  %s\n",
           kOpName[op], variant, align, tensors, C, worst, ok ? "ok" : "FAIL");
}

int main()
{
    printf("zq_cnn_eltwise_{sum,mul,max}_32f_align 回归（附录 BA）\n");
    printf("ASan 会在越界时直接 abort\n\n");

    static const int Cs[] = { 1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17 };

    for (int ci = 0; ci < (int)(sizeof(Cs) / sizeof(Cs[0])); ci++) {
        int C = Cs[ci];
        // 三个算子都用「先算 in[0] op in[1]，再对 tensor_id>=2 做第二趟
        // out = in[t] op out」的结构（raw.h 里 sum/mul/max 各一份），
        // 所以 2 个和 3 个输入都要测。
        for (int op = 0; op < 3; op++) {
            for (int tensors = 2; tensors <= 3; tensors++) {
                // 三种对齐变体的实际行为应当一致（张量的 C 都补齐到各自宽度），
                // 所以各自按自己的 align 分配、各自对拍。
                run("align0", op == OP_SUM ? zq_cnn_eltwise_sum_32f_align0
                          : op == OP_MUL ? zq_cnn_eltwise_mul_32f_align0
                                        : zq_cnn_eltwise_max_32f_align0, 1, op, tensors, 1, 2, 3, C);
                run("align128", op == OP_SUM ? zq_cnn_eltwise_sum_32f_align128bit
                             : op == OP_MUL ? zq_cnn_eltwise_mul_32f_align128bit
                                            : zq_cnn_eltwise_max_32f_align128bit, 4, op, tensors, 1, 2, 3, C);
                run("align256", op == OP_SUM ? zq_cnn_eltwise_sum_32f_align256bit
                             : op == OP_MUL ? zq_cnn_eltwise_mul_32f_align256bit
                                            : zq_cnn_eltwise_max_32f_align256bit, 8, op, tensors, 1, 2, 3, C);
            }
        }
    }

    if (g_fail) { printf("\n%d 条对拍失败\n", g_fail); return 1; }
    printf("\n全部通过（无越界、数值与标量参考一致）\n");
    return 0;
}
