// zq_resize_align_check.cpp —— `ResizeBilinearRect` 三种对齐的**一致性**（附录 ID）
//
// 背景：审计报告 H12 把它记成「同一调用在不同对齐的 tensor 上返回值/结果不同」。
// 那是**真的**，但一直只有一句话、没有判据，于是既不能证伪也不能回归。
// 这里把它变成两条**可判**的断言：
//
//   1. **rect 在界内时，三种对齐必须给出逐位相同的结果。**
//      这是真正的契约 —— 对同一个算法，三套 SIMD 实现不能有分歧。
//      任何一条退化路线（`Align128bit` 独有的夹取逻辑、AVX 与 SSE 的舍入）
//      都会在这里露出来。
//
//   2. **rect 越界时**：`Align0` / `Align256bit` 直接 `return false`，
//      `Align128bit` 则把 rect 夹到 `[-border, 尺寸-1+border]` 再算。
//      这里断言：**显式传入那个被夹过的 rect，结果与越界调用逐位相同** ——
//      也就是「夹取」这件事真的只是夹取，没有夹完再做别的。
//
// 为什么 Align128bit 要夹、另外两个不夹
// ------------------------------------
// `ZQ_CNN_Tensor4D.cpp:1108` 那段注释写得很清楚：MTCNN 的 NMS 检测框
// **不做边界裁剪**（`ZQ_CNN_MTCNN*.h` 里 16 处边界检查被注释掉了），
// 所以 Align128bit 那条路**真的会拿到越界 rect**；
// 不夹就是货真价实的堆越界读（实测纵向超出 34 像素）。
// 至于 Align0 / Align256bit 为什么还是硬拒：MTCNN 的张量本来就是
// `Align128bit`（`ConvertFromBGR` 用 `ChangeSize(1,H,W,3,1,1)`），
// 那两条路当前**在生产里走不到**。分歧是**潜在的**，不是活跃的。
// 这里不改行为，只把它**钉住**。
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include "ZQCNN/ZQ_CNN_Tensor4D.h"

static int g_fail = 0;

static void make_src(ZQ::ZQ_CNN_Tensor4D& s, int C, int H, int W, int border)
{
    if (!s.ChangeSize(1, H, W, C, border, border)) { printf("src 分配失败\n"); exit(2); }
    float* p = s.GetFirstPixelPtr();
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                p[(size_t)y * s.GetWidthStep() + (size_t)x * s.GetPixelStep() + c] =
                    (float)((c * 37 + y * 13 + x * 7) % 61) * 0.5f + 0.25f;
}

static bool run(ZQ::ZQ_CNN_Tensor4D& src, ZQ::ZQ_CNN_Tensor4D& dst,
                int dst_W, int dst_H, int off_x, int off_y, int rw, int rh,
                std::vector<float>& out)
{
    if (!dst.ChangeSize(1, dst_H, dst_W, src.GetC(), 0, 0)) return false;
    bool ok = src.ResizeBilinearRect(dst, dst_W, dst_H, 0, 0, off_x, off_y, rw, rh,
                                     ZQ::ZQ_CNN_Tensor4D::SAMPLE_ALIGN_CENTER);
    out.assign((size_t)dst_H * dst_W * src.GetC(), 0.0f);
    dst.ConvertToCompactNCHW(&out[0]);
    return ok;
}

static double max_diff(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size()) return 1e30;
    double m = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main()
{
    printf("=== ResizeBilinearRect：三种对齐的一致性 ===\n");
    const int C = 4, H = 12, W = 12, BORDER = 1;
    ZQ::ZQ_CNN_Tensor4D_NHW_C_Align0 s0, d0;
    ZQ::ZQ_CNN_Tensor4D_NHW_C_Align128bit s128, d128;
    ZQ::ZQ_CNN_Tensor4D_NHW_C_Align256bit s256, d256;
    make_src(s0, C, H, W, BORDER);
    make_src(s128, C, H, W, BORDER);
    make_src(s256, C, H, W, BORDER);

    struct Case { int dst_W, dst_H, off_x, off_y, rw, rh; };
    // ① 同尺寸（走 ROI 分支）；② 缩小（走 safeborder）；③ 放大且贴左边（触发 can_call_safeborder=false）
    static const Case IN[] = {
        {  6,  6, 1, 1,  6,  6 },
        {  5,  4, 2, 3,  7,  8 },
        { 20, 18, 0, 0, 12, 12 },
        { 20, 18, 3, 4,  9,  7 },
    };

    for (size_t i = 0; i < sizeof(IN) / sizeof(IN[0]); i++) {
        const Case& c = IN[i];
        std::vector<float> o0, o128, o256;
        bool k0 = run(s0, d0, c.dst_W, c.dst_H, c.off_x, c.off_y, c.rw, c.rh, o0);
        bool k128 = run(s128, d128, c.dst_W, c.dst_H, c.off_x, c.off_y, c.rw, c.rh, o128);
        bool k256 = run(s256, d256, c.dst_W, c.dst_H, c.off_x, c.off_y, c.rw, c.rh, o256);
        printf("  in-bounds #%zu dst=%dx%d rect=(%d,%d,%d,%d) -> ret a0=%d a128=%d a256=%d\n",
               i, c.dst_W, c.dst_H, c.off_x, c.off_y, c.rw, c.rh, k0, k128, k256);
        if (!(k0 && k128 && k256)) { printf("  **FAIL** 界内 rect 三种对齐都应当成功\n"); g_fail++; continue; }
        double e1 = max_diff(o0, o128), e2 = max_diff(o0, o256);
        if (e1 > 1e-6 || e2 > 1e-6) {
            printf("  **FAIL** 界内 rect 三种对齐结果不一致：a0 vs a128 = %.6g，a0 vs a256 = %.6g\n",
                   e1, e2);
            g_fail++;
        } else {
            printf("  %-6s 界内 rect 三种对齐逐位一致（最大偏差 %.3g）\n", "ok", e2);
        }
    }

    // 越界 rect：Align128bit 夹取，另外两个硬拒
    {
        const int dst_W = 20, dst_H = 18;
        std::vector<float> oob, clamped;
        bool k128 = run(s128, d128, dst_W, dst_H, -6, 2, 12, 9, oob);
        // 按 Align128bit 的夹取规则自己算一遍
        const int hi_x = W - 1 + BORDER, hi_y = H - 1 + BORDER;
        int ox = -6, oy = 2, rw = 12, rh = 9;
        if (ox < -BORDER) ox = -BORDER;
        if (oy < -BORDER) oy = -BORDER;
        if (ox + rw - 1 > hi_x) rw = hi_x - ox + 1;
        if (oy + rh - 1 > hi_y) rh = hi_y - oy + 1;
        bool kc = run(s128, d128, dst_W, dst_H, ox, oy, rw, rh, clamped);
        // **各自一个输出向量**：run() 无论 resize 成功与否都会把 dst 拷出来，
        // 共用一个向量的话，后两次失败的调用会把前一次的结果**覆盖掉** ——
        // 第一版就是这么写的，于是「夹取 != 显式传夹过的 rect」是个假阳性。
        std::vector<float> junk0, junk256;
        bool k0 = run(s0, d0, dst_W, dst_H, -6, 2, 12, 9, junk0);
        bool k256 = run(s256, d256, dst_W, dst_H, -6, 2, 12, 9, junk256);

        printf("  out-of-bounds rect=(-6,2,12,9) -> a0=%d a128=%d a256=%d（预期 0/1/0）\n",
               k0, k128, k256);
        if (!(k0 == false && k256 == false && k128 == true)) {
            printf("  **FAIL** 越界 rect 的返回值与已知契约不符\n"); g_fail++;
        } else {
            printf("  %-6s 越界 rect 的返回值符合契约：Align0/Align256bit 硬拒，Align128bit 夹取\n", "ok");
        }
        double e = kc ? max_diff(oob, clamped) : 1e30;
        if (e > 0.0) {
            printf("  **FAIL** 夹取后应与「显式传夹过的 rect」逐位相同，偏差 %.6g\n", e);
            g_fail++;
        } else {
            printf("  %-6s 夹取 == 显式传夹过的 rect（逐位相同）\n", "ok");
        }
    }

    if (g_fail) { printf("RESIZE ALIGN CHECK FAILED (%d 处)\n", g_fail); return 1; }
    printf("RESIZE ALIGN CHECK OK\n");
    return 0;
}
