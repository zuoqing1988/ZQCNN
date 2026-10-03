/* `ZQ_CNN_Tensor4D_NCHWC` **自己那批方法**的门禁 —— 附录 EA
 *
 * 为什么这道门禁存在
 * ------------------
 * NCHWC 张量类被 **9 道门禁**当数据容器用（`ChangeSize` + 取首指针），
 * 但它**自己的方法**一个都没有门禁：
 *   ConvertFromCompactNCHW / ConvertToCompactNCHW / ConvertFromBGR /
 *   ConvertFromGray / ConvertToBGR / Permute_NCHW / Flatten_NCHW /
 *   Reshape_NCHW / SaveToFile，以及三个子类各自的 Padding / ROI / CopyData / Resize*
 *
 * 这与附录 DX.6 在基类 `ZQ_CNN_Tensor4D` 上发现的缺口是同一个形状，
 * 而基类那批方法在两轮里出了**两条真缺陷**（DY.2 的越界读、DY.9 的 border 传反）。
 *
 * 判据原则（沿用 DY / DZ，不重复论证）
 * ------------------------------------
 * 1. **恒等变换必须逐格不变** —— 形状没变，任何正确实现都不许动一个数。
 *    DY.1 正是靠这条把"`i_c` 漏乘 `in_PixelStep`"这个**误判**否掉的。
 * 2. **参考不依赖实现的公式**。Reshape / Permute / Flatten 的实现都走
 *    "转成 compact NCHW -> 在 compact 上算 -> 转回来"，所以参考**不能也走 compact** ——
 *    那是拿实现对照实现。参考改成：**按布局公式自己取出 compact**，
 *    然后在 compact 上用独立的下标算术算出期望，再与输出的 compact 比。
 * 3. **短 shape + 一个 `-1`** 是 DY.2 越界读的触发条件，必须常驻。
 * 4. **`C < 3` 时 `ConvertToBGR` 必须被拒**（DZ.1；NCHWC 这份拷贝也修了）。
 *
 * 布局（`ZQ_CNN_Tensor4D_NCHWC.h` 的 `ChangeSize` / `ConvertToCompactNCHW`）：
 *   元素 (n,c,h,w) 在  firstPixelData + n*imageStep + c*sliceStep
 *                                        + h*widthStep  + w*align_size
 *   注意 NCHWC 侧 **imStep 才是"一张图"、sliceStep 是"一个通道"**
 *   —— 与 NCHW 侧恰好相反（AGENTS.md 有专门一节记这条）。
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "zq_check_child.h"
#include "ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h"

using namespace ZQ;

#define RES_FILE "/tmp/zq_nchwctensor_res.txt"

enum { OP_COMPACT_RT = 0, OP_BGR_RT, OP_TOBGR_REJECT, OP_GRAY, OP_RESHAPE, OP_PERMUTE, OP_FLATTEN };
static const char* g_opname[] = {
    "compact往返", "BGR往返", "toBGR拒C<3", "灰度", "Reshape", "Permute", "Flatten"
};
static const char* g_note[] = {
    "", "搭建失败", "**返回值与期望相反**", "**形状不对**",
    "**数据错**", "**应当被拒却收下了**", ""
};

struct Case {
    int op;
    int N, C, H, W;
    int s[5];        // reshape 的 shape，只前 slen 个有效
    int slen;
    int ax0, ax1;    // Flatten 用
    int order[4];    // Permute 用
    int expect_ok;
};

static const Case g_cases[] = {
  // ---- compact NCHW 往返：恒等 ----
  { OP_COMPACT_RT, 1, 3, 2, 2, {0,0,0,0}, 0, 0, 0, {0,1,2,3}, 1 },
  { OP_COMPACT_RT, 2, 5, 2, 3, {0,0,0,0}, 0, 0, 0, {0,1,2,3}, 1 },   // C 不是 align 的倍数
  { OP_COMPACT_RT, 1, 8, 1, 1, {0,0,0,0}, 0, 0, 0, {0,1,2,3}, 1 },
  // ---- BGR 往返（逐字节精确）----
  { OP_BGR_RT, 1, 3, 2, 3, {0,0,0,0}, 0, 0, 0, {0,1,2,3}, 1 },
  { OP_BGR_RT, 1, 3, 5, 5, {0,0,0,0}, 0, 0, 0, {0,1,2,3}, 1 },
  // ---- C<3 必须被拒（DZ.1）----
  { OP_TOBGR_REJECT, 1, 1, 2, 2, {0,0,0,0}, 0, 0, 0, {0,1,2,3}, 0 },
  { OP_TOBGR_REJECT, 1, 2, 2, 2, {0,0,0,0}, 0, 0, 0, {0,1,2,3}, 0 },
  // ---- 灰度：指针是 ++（步长 1），一行只取前 W 个字节 ----
  { OP_GRAY, 1, 1, 2, 3, {0,0,0,0}, 0, 0, 0, {0,1,2,3}, 1 },
  { OP_GRAY, 1, 1, 4, 4, {0,0,0,0}, 0, 0, 0, {0,1,2,3}, 1 },
  // ---- Reshape：恒等 ----
  { OP_RESHAPE, 1, 3, 2, 2, {1,3,2,2}, 4, 0, 0, {0,1,2,3}, 1 },
  { OP_RESHAPE, 2, 5, 2, 3, {2,5,2,3}, 4, 0, 0, {0,1,2,3}, 1 },
  { OP_RESHAPE, 1, 8, 1, 1, {1,8,1,1}, 4, 0, 0, {0,1,2,3}, 1 },
  // ---- Reshape：跨形状 ----
  { OP_RESHAPE, 1, 2, 3, 4, { 24,1,1,1 }, 4, 0, 0, {0,1,2,3}, 1 },
  { OP_RESHAPE, 1, 2, 3, 4, {  2,3,4,1 }, 4, 0, 0, {0,1,2,3}, 1 },
  { OP_RESHAPE, 1, 2, 3, 4, {  6,4,1,1 }, 4, 0, 0, {0,1,2,3}, 1 },
  // ---- Reshape：一个 -1（4 元素）----
  { OP_RESHAPE, 1, 2, 3, 4, {  2,-1,4,1 }, 4, 0, 0, {0,1,2,3}, 1 },
  { OP_RESHAPE, 1, 2, 3, 4, {  1,2,-1,1 }, 4, 0, 0, {0,1,2,3}, 1 },
  // ---- Reshape：**短 shape + -1** = DY.2 越界读的触发条件，必须常驻 ----
  { OP_RESHAPE, 1, 4, 2, 3, { -1,0,0,0 }, 1, 0, 0, {0,1,2,3}, 1 },
  { OP_RESHAPE, 1, 4, 2, 3, {  2,-1,0,0 }, 2, 0, 0, {0,1,2,3}, 1 },
  { OP_RESHAPE, 1, 4, 2, 3, {  2,3,-1,0 }, 3, 0, 0, {0,1,2,3}, 1 },
  // ---- Reshape：0 = 沿用输入的该维 ----
  { OP_RESHAPE, 2, 3, 4, 5, {  0,0,0,0 }, 4, 0, 0, {0,1,2,3}, 1 },
  { OP_RESHAPE, 2, 3, 4, 5, {  5,0,0,2 }, 4, 0, 0, {0,1,2,3}, 1 },
  // ---- Reshape：应当被拒 ----
  { OP_RESHAPE, 1, 2, 3, 4, { 1,1,1,1,1 }, 5, 0, 0, {0,1,2,3}, 0 },
  { OP_RESHAPE, 1, 2, 3, 4, { 1,-1,-1,1 }, 4, 0, 0, {0,1,2,3}, 0 },
  { OP_RESHAPE, 1, 2, 3, 4, { 1,2,3,5 }, 4, 0, 0, {0,1,2,3}, 0 },
  { OP_RESHAPE, 1, 2, 3, 4, { 5,-1,1,1 }, 4, 0, 0, {0,1,2,3}, 0 },
  { OP_RESHAPE, 1, 2, 3, 4, { 5,-1 }, 2, 0, 0, {0,1,2,3}, 0 },
  // ---- Permute：恒等 order ----
  { OP_PERMUTE, 1, 3, 2, 2, {0,0,0,0}, 0, 0, 0, {0,1,2,3}, 1 },
  { OP_PERMUTE, 2, 5, 2, 3, {0,0,0,0}, 0, 0, 0, {0,1,2,3}, 1 },
  // ---- Permute：非平凡置换 ----
  { OP_PERMUTE, 1, 3, 2, 4, {0,0,0,0}, 0, 0, 0, {0,2,3,1}, 1 },   // N,C,H,W -> N,H,W,C
  { OP_PERMUTE, 1, 2, 3, 4, {0,0,0,0}, 0, 0, 0, {3,2,1,0}, 1 },   // 完全倒序
  { OP_PERMUTE, 1, 2, 3, 4, {0,0,0,0}, 0, 0, 0, {0,3,1,2}, 1 },
  // ---- Permute：order 不合法必须被拒 ----
  { OP_PERMUTE, 1, 2, 3, 4, {0,0,0,0}, 0, 0, 0, {0,0,1,2}, 0 },
  { OP_PERMUTE, 1, 2, 3, 4, {0,0,0,0}, 0, 0, 0, {0,1,2,2}, 0 },
  { OP_PERMUTE, 1, 2, 3, 4, {0,0,0,0}, 0, 0, 0, {0,1,2,5}, 0 },
  // ---- Flatten：恒等 (0,0) 与跨形状 ----
  { OP_FLATTEN, 1, 3, 2, 4, {0,0,0,0}, 0, 0, 0, {0,1,2,3}, 1 },
  { OP_FLATTEN, 1, 3, 2, 4, {0,0,0,0}, 0, 0, 1, {0,1,2,3}, 1 },
  { OP_FLATTEN, 1, 3, 2, 4, {0,0,0,0}, 0, 0, 2, {0,1,2,3}, 1 },
  { OP_FLATTEN, 1, 3, 2, 4, {0,0,0,0}, 0, 0, 3, {0,1,2,3}, 1 },
  { OP_FLATTEN, 1, 3, 2, 4, {0,0,0,0}, 0, 1, 3, {0,1,2,3}, 1 },
  { OP_FLATTEN, 1, 3, 2, 4, {0,0,0,0}, 0, 2, 3, {0,1,2,3}, 1 },
};
static const int N_CASE = (int)(sizeof(g_cases) / sizeof(g_cases[0]));

// 元素的线性偏移。**NCHWC 侧 imStep 是"一张图"、sliceStep 是"一组 align_size 个通道"** ——
// 与 NCHW 侧相反，写错就是附录 BN.2 那一族缺陷。
//
// **第一版写成了 `c*sliceStep`，被 ASan 打在我自己的 fill_unique 上**：
// `sliceStep` 步进的是**一个通道组**（align_size 个通道），
// 通道在组内是连续的 —— 所以是 `(c/align)*sliceStep + (c%align)`。
// 这正是 AGENTS.md 那条「ASan 报的栈顶在测试文件里时，先怀疑测试」。
static inline size_t nchwc_off(int n, int c, int h, int w,
                               int imStep, int sliceStep, int widthStep, int align)
{
    return (size_t)n * imStep + (size_t)(c / align) * sliceStep
         + (size_t)h * widthStep + (size_t)w * align + (size_t)(c % align);
}

template <class T>
static void fill_unique(T& t)
{
    const int N = t.GetN(), C = t.GetC(), H = t.GetH(), W = t.GetW();
    const int im = t.GetImageStep(), sl = t.GetSliceStep(), ws = t.GetWidthStep();
    const int al = t.GetAlignSize();
    float* p = t.GetFirstPixelPtr();
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    p[nchwc_off(n, c, h, w, im, sl, ws, al)] =
                        (float)(((n * 37 + c * 11 + h * 5 + w) % 89) + 1) * 0.25f;
}

// 独立取出 compact NCHW（按布局公式，不走实现的循环）
template <class T>
static void my_compact(const T& t, std::vector<float>& out)
{
    const int N = t.GetN(), C = t.GetC(), H = t.GetH(), W = t.GetW();
    const int im = t.GetImageStep(), sl = t.GetSliceStep(), ws = t.GetWidthStep();
    const int al = t.GetAlignSize();
    const float* p = t.GetFirstPixelPtr();
    out.assign((size_t)N * C * H * W, 0.f);
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    out[((size_t)n * C + c) * H * W + (size_t)h * W + w] =
                        p[nchwc_off(n, c, h, w, im, sl, ws, al)];
}

template <class T>
static void run_one(const Case& c)
{
    T t, o;
    long bad = 0, first = -1; int note = 0;

    if (c.op == OP_TOBGR_REJECT) {
        if (!t.ChangeSize(1, c.H, c.W, c.C, 0, 0)) { note = 1; bad++; }
        else {
            const int wsz = c.W * 3;
            std::vector<unsigned char> img((size_t)c.H * wsz, 0xAB);
            if (t.ConvertToBGR(&img[0], c.W, c.H, wsz, 0)) { note = 5; bad++; }  // 不拒就是越界读
        }
    } else if (c.op == OP_COMPACT_RT) {
        if (!t.ChangeSize(c.N, c.H, c.W, c.C, 0, 0)) { note = 1; bad++; }
        else {
            fill_unique(t);
            std::vector<float> a, impl, b;
            my_compact(t, a);                       // 独立提取（按布局公式）
            impl.assign(a.size(), 0.f);
            t.ConvertToCompactNCHW(&impl[0]);       // 实现的提取
            for (size_t i = 0; i < a.size() && bad < 4; i++)
                if (a[i] != impl[i]) { if (first < 0) first = (long)i; bad++; note = 4; }
            if (!bad) {
                // 再走回来：compact -> 张量，必须与原张量逐格一致
                if (!o.ConvertFromCompactNCHW(&impl[0], c.N, c.C, c.H, c.W, 0, 0)) { note = 2; bad++; }
                else {
                    my_compact(o, b);
                    if (b.size() != a.size()) { note = 3; bad++; }
                    else for (size_t i = 0; i < a.size() && bad < 4; i++)
                        if (a[i] != b[i]) { if (first < 0) first = (long)i; bad++; note = 4; }
                }
            }
        }
    } else if (c.op == OP_BGR_RT) {
        const int wsz = c.W * 3;
        std::vector<unsigned char> img((size_t)c.H * wsz);
        for (int h = 0; h < c.H; h++)
            for (int w = 0; w < c.W; w++)
                for (int k = 0; k < 3; k++)
                    img[(size_t)h * wsz + w * 3 + k] = (unsigned char)((h * 7 + w * 3 + k * 53) & 0xFF);
        if (!t.ConvertFromBGR(&img[0], c.W, c.H, wsz)) { note = 2; bad++; }
        else if (t.GetC() != 3) { note = 3; bad++; }
        else {
            std::vector<unsigned char> back((size_t)c.H * wsz, 0xAB);
            if (!t.ConvertToBGR(&back[0], c.W, c.H, wsz, 0)) { note = 2; bad++; }
            else for (int h = 0; h < c.H && bad < 4; h++)
                for (int w = 0; w < c.W && bad < 4; w++)
                    for (int k = 0; k < 3 && bad < 4; k++) {
                        const size_t i = (size_t)h * wsz + w * 3 + k;
                        if (back[i] != img[i]) { if (first < 0) first = (long)i; bad++; }
                    }
            if (bad) note = 4;
        }
    } else if (c.op == OP_GRAY) {
        // 输入按 BGR 布局给，但 ConvertFromGray 的指针是 `++`（步长 1）：
        // 一行只取前 W 个字节。参考必须与之一致（附录 DZ.3 第 ① 条）。
        const int wsz = c.W * 3;
        std::vector<unsigned char> img((size_t)c.H * wsz);
        for (int h = 0; h < c.H; h++)
            for (int w = 0; w < c.W * 3; w++)
                img[(size_t)h * wsz + w] = (unsigned char)((h * 11 + w * 5) & 0xFF);
        if (!t.ConvertFromGray(&img[0], c.W, c.H, wsz)) { note = 2; bad++; }
        else if (t.GetC() != 1) { note = 3; bad++; }
        else {
            const float mv = 127.5f, sc = 0.0078125f;
            const int im = t.GetImageStep(), sl = t.GetSliceStep(), ws = t.GetWidthStep();
            const int al = t.GetAlignSize();
            const float* p = t.GetFirstPixelPtr();
            for (int h = 0; h < c.H && bad < 4; h++)
                for (int w = 0; w < c.W && bad < 4; w++) {
                    const double want = ((double)img[(size_t)h * wsz + w] - mv) * sc;
                    const double got = (double)p[nchwc_off(0, 0, h, w, im, sl, ws, al)];
                    if (fabs(got - want) > 1e-6) {
                        if (first < 0) first = (long)h * 1000 + w; bad++; note = 4;
                    }
                }
        }
    } else if (c.op == OP_RESHAPE) {
        if (!t.ChangeSize(c.N, c.H, c.W, c.C, 0, 0)) { note = 1; bad++; }
        else {
            fill_unique(t);
            std::vector<int> shape;
            for (int i = 0; i < c.slen; i++) shape.push_back(c.s[i]);   // capacity == size
            const bool r = t.Reshape_NCHW(o, shape, 1);
            if (c.expect_ok == 0) {
                if (r) { note = 5; bad++; }
            } else if (!r) { note = 2; bad++; }
            else {
                std::vector<float> a, b;
                my_compact(t, a);
                my_compact(o, b);
                if (a.size() != b.size()) { note = 3; bad++; }
                else for (size_t i = 0; i < a.size(); i++)
                    if (a[i] != b[i]) { if (first < 0) first = (long)i; bad++; }
                if (bad) note = 4;
            }
        }
    } else if (c.op == OP_PERMUTE) {
        if (!t.ChangeSize(c.N, c.H, c.W, c.C, 0, 0)) { note = 1; bad++; }
        else {
            fill_unique(t);
            const int ord[4] = { c.order[0], c.order[1], c.order[2], c.order[3] };
            const bool r = t.Permute_NCHW(o, ord, 1);
            if (c.expect_ok == 0) {
                if (r) { note = 5; bad++; }
            } else if (!r) { note = 2; bad++; }
            else {
                const int inD[4] = { c.N, c.C, c.H, c.W };
                int outD[4];
                for (int i = 0; i < 4; i++) outD[i] = inD[ord[i]];
                if (o.GetN() != outD[0] || o.GetC() != outD[1] || o.GetH() != outD[2] || o.GetW() != outD[3]) {
                    note = 3; bad++;
                } else {
                    // 独立参考：输出 compact 的线性下标 -> 解成 (a0,a1,a2,a3)
                    // -> 第 i 个输出轴取输入的第 ord[i] 个轴 -> 输入 compact 下标
                    std::vector<float> a, b;
                    my_compact(t, a);
                    my_compact(o, b);
                    const int st[4] = { outD[1] * outD[2] * outD[3], outD[2] * outD[3], outD[3], 1 };
                    const int ist[4] = { inD[1] * inD[2] * inD[3], inD[2] * inD[3], inD[3], 1 };
                    for (int j = 0; j < (int)b.size() && bad < 4; j++) {
                        int k = j, coord[4];
                        for (int i = 0; i < 4; i++) { coord[i] = k / st[i]; k %= st[i]; }
                        int src = 0;
                        for (int i = 0; i < 4; i++) src += coord[i] * ist[ord[i]];
                        if (b[j] != a[src]) { if (first < 0) first = j; bad++; }
                    }
                    if (bad) note = 4;
                }
            }
        }
    } else if (c.op == OP_FLATTEN) {
        if (!t.ChangeSize(c.N, c.H, c.W, c.C, 0, 0)) { note = 1; bad++; }
        else {
            fill_unique(t);
            const bool r = t.Flatten_NCHW(o, c.ax0, c.ax1, 1);
            if (!r) { note = 2; bad++; }
            else {
                // 独立算期望的输出尺寸：shape = [old_dim[0..ax0-1], prod(ax0..ax1), old_dim[ax1+1..3]]
                int sh[4] = {0, 0, 0, 0};
                const int oldD[4] = { c.N, c.C, c.H, c.W };
                int p = 0, f = 1;
                for (int i = 0; i < c.ax0 && p < 4; i++) sh[p++] = oldD[i];
                for (int i = c.ax0; i <= c.ax1; i++) f *= oldD[i];
                if (p < 4) sh[p++] = f;
                for (int i = c.ax1 + 1; i < 4 && p < 4; i++) sh[p++] = oldD[i];
                while (p < 4) sh[p++] = 1;
                if (o.GetN() != sh[0] || o.GetC() != sh[1] || o.GetH() != sh[2] || o.GetW() != sh[3]) {
                    note = 3; bad++;
                } else {
                    std::vector<float> a, b;
                    my_compact(t, a);
                    my_compact(o, b);
                    for (size_t i = 0; i < a.size() && bad < 4; i++)
                        if (a[i] != b[i]) { if (first < 0) first = (long)i; bad++; }
                    if (bad) note = 4;
                }
            }
        }
    }

    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %d %ld\n", 1L - bad, bad, note, first); fclose(f); }
}

static const char* g_kind_name[] = { "NCHWC1", "NCHWC4", "NCHWC8" };
static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

template <class T>
static void run_all(const char* name)
{
    const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
    for (int i = 0; i < N_CASE; i++) {
        const Case& c = g_cases[i];
        g_case++;
        remove(RES_FILE);
        pid_t pid = fork();
        if (pid == 0) { zq_child_silence_stderr(); run_one<T>(c); _exit(0); }
        int st = 0; waitpid(pid, &st, 0);
        long ok = 0, bad = 0, first = -1; int note = 0, have = 0;
        FILE* f = fopen(RES_FILE, "r");
        if (f) { have = (fscanf(f, "%ld %ld %d %ld", &ok, &bad, &note, &first) == 4); fclose(f); }
        char tag[128];
        if (c.op == OP_RESHAPE) {
            char sh[48]; int p = 0; sh[0] = 0;
            for (int i2 = 0; i2 < c.slen && i2 < 5; i2++)
                p += snprintf(sh + p, sizeof(sh) - p, "%s%d", i2 ? "," : "", c.s[i2]);
            snprintf(tag, sizeof(tag), "%-10s N%dC%dH%dW%d -> {%s}/%d%s",
                     g_opname[c.op], c.N, c.C, c.H, c.W, sh, c.slen, c.expect_ok ? "" : " (应拒)");
        } else if (c.op == OP_PERMUTE) {
            snprintf(tag, sizeof(tag), "%-10s N%dC%dH%dW%d order{%d,%d,%d,%d}%s",
                     g_opname[c.op], c.N, c.C, c.H, c.W,
                     c.order[0], c.order[1], c.order[2], c.order[3], c.expect_ok ? "" : " (应拒)");
        } else if (c.op == OP_FLATTEN) {
            snprintf(tag, sizeof(tag), "%-10s N%dC%dH%dW%d Flatten(%d,%d)",
                     g_opname[c.op], c.N, c.C, c.H, c.W, c.ax0, c.ax1);
        } else {
            snprintf(tag, sizeof(tag), "%-10s N%dC%dH%dW%d%s",
                     g_opname[c.op], c.N, c.C, c.H, c.W, c.expect_ok ? "" : " (应拒)");
        }
        if (!have) {
            g_crash++;
            printf("  %-8s %-46s  没跑完%s\n", name, tag,
                   WIFSIGNALED(st) ? "（子进程被信号杀掉）" : "（结果文件读不出来）");
            continue;
        }
        if (WIFSIGNALED(st)) {
            g_crash++;
            printf("  %-8s %-46s  CRASH（信号 %d）\n", name, tag, WTERMSIG(st));
            continue;
        }
        if (bad > 0) {
            g_bad++;
            printf("  %-8s %-46s  %s %ld 项", name, tag, g_note[note], bad);
            if (note == 4) printf("，首个错在第 %ld 个", first);
            printf("\n");
        } else g_ok++;
    }
    printf("  %-8s %d 个用例：对 %d，错 %d，崩 %d\n", name, g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("ZQ_CNN_Tensor4D_NCHWC 自有方法门禁（附录 EA）\n");
    printf("为什么有这道门禁：该类被 9 道门禁当**数据容器**用（ChangeSize + 取指针），\n");
    printf("  但它自己的 Convert 族 / Permute / Flatten / Reshape **一个门禁都没有** ——\n");
    printf("  与附录 DX.6 在基类上发现的缺口同一个形状，而基类那批两轮出了两条真缺陷\n");
    printf("  （DY.2 越界读、DY.9 border 传反）。\n");
    printf("判据：恒等变换逐格不变；跨形状按**独立下标算术**对拍（不走实现的 compact 路径）；\n");
    printf("      短 shape + -1 常驻（DY.2 触发条件）；C<3 时 ConvertToBGR 必须被拒（DZ.1）。\n");
    printf("布局：NCHWC 侧 imStep 才是\"一张图\"、sliceStep 是\"一个通道\"，与 NCHW 侧相反。\n\n");
    run_all<ZQ_CNN_Tensor4D_NCHWC1>(g_kind_name[0]);
    run_all<ZQ_CNN_Tensor4D_NCHWC4>(g_kind_name[1]);
    run_all<ZQ_CNN_Tensor4D_NCHWC8>(g_kind_name[2]);
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
