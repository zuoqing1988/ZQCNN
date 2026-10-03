/* `ZQ_CNN_Tensor4D::Reshape_NCHW` / `Flatten_NCHW` 的形状 + 数值门禁 —— 附录 DY
 *
 * 为什么这道门禁存在（附录 DX 的覆盖探针排出来的下一位）
 * ------------------------------------------------------
 * `Reshape_NCHW` 51 行、`Flatten_NCHW` 16 行，此前**零门禁**。
 * 探针按"行数 × 指针算术密度"排序时它排在 Tile / ROI 之后，
 * 而 Tile/ROI 这族已经连续出了三条缺陷（附录 DD.9 / DX）。
 *
 * 它已经出了一条：附录 DY.2 的 `Reshape_NCHW_get_size` 越界读。
 *
 * 与别道门禁的形态差别
 * --------------------
 * 同 `zq_tile`（附录 DC.6）：逻辑写在 `ZQ_CNN_Tensor4D` 的成员函数里，
 * 绕不开真实张量对象，所以要编 `ZQ_CNN_Tensor4D.cpp` + resize 内核。
 *
 * 判据
 * ----
 * 1. **恒等 reshape 必须逐格不变**。这是最强的判据：形状完全没变，
 *    任何正确实现都不许动一个数。第一版就是靠它把
 *    "`i_c` 漏乘 `in_PixelStep`"这个**误判**给否掉的（附录 DY.1）——
 *    输入是 NHWC 对齐布局，内存里 `c` 本来就是最内层，`w*ps + c` 才是对的。
 * 2. **跨形状 reshape**：`out[n,c,h,w] = in[把 out 的 NCHW 线性下标
 *    解码回输入得到的 (n,c,h,w)]`。参考是独立写的，不是抄实现的循环。
 * 3. **`shape` 可以短于 4**：调用方给的 vector 就是这么长
 *    （`ZQ_CNN_Forward_SSEUtils.h::Reshape` 原样透传），
 *    短 shape + 一个 `-1` 就是附录 DY.2 那条越界读的触发条件。
 * 4. **`0` 的语义是"沿用输入的该维"**，不是"填 1"。
 * 5. 被拒的组合（5 维、两个 `-1`、乘积对不上、除不尽）必须返回 false，
 *    且**输出缓冲一个字节都不许被写**。
 * 6. 输出尺寸必须等于表里手算的 `eN/eC/eH/eW` —— 把"算形状"和"搬数据"
 *    拆成两个独立判据，否则形状算错会被数据判据掩盖。
 *
 * **只有 ASan 能抓到 DY.2 那条**（越界读的字节恰好落在 `new_dim[i]==1`
 * 的空操作上，非 ASan 下结果完全正确）。所以本门禁的 OOB 判据依赖 sanitizer 轴，
 * 这点必须写在门禁自己脸上，不能让人以为跑绿了就等于没越界。
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "zq_check_child.h"
#include "ZQCNN/ZQ_CNN_Tensor4D.h"
// **同源拷贝 #2**：ZQ_CNN_Tensor4D_NCHWC.h 里有一份逐字相同的
// Reshape_NCHW_get_size（附录 DY.2 一起修了）。它的类名与 guard 都不同，
// 所以能进同一个 TU —— `Reshape_NCHW_get_size` 是 **static** 成员，
// 不需要实例化任何子类，直接静态调用即可。
// 之前我以为这个头文件在 Linux 上编不了（里面有 `__int64` / `_aligned_free`
// 这类 MSVC 名），**实测能编**（-fsyntax-only 干净通过）—— 又一次"凭印象下结论"。
#include "ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h"
using namespace ZQ;

#define RES_FILE "/tmp/zq_reshape_res.txt"

// 输入每格填一个**唯一**的值（base 32，n<8 / c,h,w<32 -> 最大 262143，
// 远小于 2^24，float 里**精确可表示**，所以可以逐格精确比对而不是比容差）。
// 定位"数据被写错位置"这类缺陷的杀手锏就是它：输出值能反查它等于输入第几格。
static float val(int n, int c, int h, int w)
{
    return (float)(n * 32768 + c * 1024 + h * 32 + w);
}

enum { K_A0 = 0, K_A128 = 1, K_A256 = 2 };
static const char* g_kind_name[] = { "align0", "align128bit", "align256bit" };

static ZQ_CNN_Tensor4D* make(int kind)
{
    if (kind == K_A0) return new ZQ_CNN_Tensor4D_NHW_C_Align0();
    if (kind == K_A128) return new ZQ_CNN_Tensor4D_NHW_C_Align128bit();
    return new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
}

enum { OP_RESHAPE = 0, OP_FLATTEN = 1 };
struct Case {
    int kind, N, C, H, W;
    int op;
    int s[5];       // reshape 的 shape，只前 slen 个有效（s[5] 是为了装"应当被拒"的 5 维用例）
    int slen;
    int ax0, ax1;   // Flatten 用
    int expect_ok;
    int eN, eC, eH, eW;   // 手算的输出尺寸
};

// (N, C, H, W, op, shape[4], slen, ax0, ax1, expect_ok, eN, eC, eH, eW)
static const Case g_cases[] = {
  // ---- A. 恒等：形状完全不变，任何正确实现都必须逐格不变 ----
  { 0, 1, 1, 2, 3, OP_RESHAPE, { 1, 1, 2, 3 }, 4, 0, 0, 1, 1, 1, 2, 3 },
  { 0, 1, 3, 2, 2, OP_RESHAPE, { 1, 3, 2, 2 }, 4, 0, 0, 1, 1, 3, 2, 2 },
  { 0, 2, 4, 3, 5, OP_RESHAPE, { 2, 4, 3, 5 }, 4, 0, 0, 1, 2, 4, 3, 5 },
  { 0, 1, 8, 2, 2, OP_RESHAPE, { 1, 8, 2, 2 }, 4, 0, 0, 1, 1, 8, 2, 2 },
  { 0, 1, 5, 1, 1, OP_RESHAPE, { 1, 5, 1, 1 }, 4, 0, 0, 1, 1, 5, 1, 1 },
  { 0, 1,16, 1, 1, OP_RESHAPE, { 1,16, 1, 1 }, 4, 0, 0, 1, 1,16, 1, 1 },
  { 0, 1, 3, 5, 7, OP_RESHAPE, { 1, 3, 5, 7 }, 4, 0, 0, 1, 1, 3, 5, 7 },
  // ---- B. shape[i]==0 的语义是"沿用输入的该维" ----
  // 全 0 = 沿用全部（恒等）；部分 0 = 只沿用那几维，其余显式给。
  // 特意挑**结果与输入不同形**的组合（{5,0,0,2} -> 5x3x4x2）：
  // 全 0 那条即使把 0 当成"填 1"也会红，而这条两者都过不了 —— 才能区分。
  { 0, 2, 3, 4, 5, OP_RESHAPE, { 0, 0, 0, 0 }, 4, 0, 0, 1, 2, 3, 4, 5 },
  { 0, 2, 3, 4, 5, OP_RESHAPE, { 2, 0, 0, 5 }, 4, 0, 0, 1, 2, 3, 4, 5 },
  { 0, 2, 3, 4, 5, OP_RESHAPE, { 5, 0, 0, 2 }, 4, 0, 0, 1, 5, 3, 4, 2 },
  { 0, 2, 3, 4, 5, OP_RESHAPE, { 0, 0, 5, 4 }, 4, 0, 0, 1, 2, 3, 5, 4 },
  // ---- C. 跨形状（count 恒为 24）----
  { 0, 1, 2, 3, 4, OP_RESHAPE, { 24, 1, 1, 1 }, 4, 0, 0, 1, 24, 1, 1, 1 },
  { 0, 1, 2, 3, 4, OP_RESHAPE, {  2,12, 1, 1 }, 4, 0, 0, 1,  2,12, 1, 1 },
  { 0, 1, 2, 3, 4, OP_RESHAPE, {  2, 3, 4, 1 }, 4, 0, 0, 1,  2, 3, 4, 1 },
  { 0, 1, 2, 3, 4, OP_RESHAPE, {  6, 4, 1, 1 }, 4, 0, 0, 1,  6, 4, 1, 1 },
  { 0, 1, 2, 3, 4, OP_RESHAPE, {  4, 3, 2, 1 }, 4, 0, 0, 1,  4, 3, 2, 1 },
  { 0, 2, 3, 2, 2, OP_RESHAPE, {  1, 2, 3, 4 }, 4, 0, 0, 1,  1, 2, 3, 4 },
  // ---- D. 一个 -1（4 元素 shape，unknown_num==1 分支的正常形态）----
  { 0, 1, 2, 3, 4, OP_RESHAPE, { -1, 2, 3, 4 }, 4, 0, 0, 1,  1, 2, 3, 4 },
  { 0, 1, 2, 3, 4, OP_RESHAPE, {  2,-1, 4, 1 }, 4, 0, 0, 1,  2, 3, 4, 1 },
  { 0, 1, 2, 3, 4, OP_RESHAPE, {  1, 2,-1, 1 }, 4, 0, 0, 1,  1, 2,12, 1 },
  { 0, 1, 2, 3, 4, OP_RESHAPE, {  1, 1, 1,-1 }, 4, 0, 0, 1,  1, 1, 1,24 },
  // ---- E. **短 shape + 一个 -1** = 附录 DY.2 那条越界读的触发条件 ----
  //     vector 的 capacity 正好等于 size，于是 shape[shape_dim..3] 落在块外。
  { 0, 1, 4, 2, 3, OP_RESHAPE, { -1, 0, 0, 0 }, 1, 0, 0, 1, 24, 1, 1, 1 },
  { 0, 1, 4, 2, 3, OP_RESHAPE, {  2,-1, 0, 0 }, 2, 0, 0, 1,  2,12, 1, 1 },
  { 0, 1, 4, 2, 3, OP_RESHAPE, {  1,-1, 0, 0 }, 2, 0, 0, 1,  1,24, 1, 1 },
  { 0, 1, 4, 2, 3, OP_RESHAPE, {  2, 3,-1, 0 }, 3, 0, 0, 1,  2, 3, 4, 1 },
  { 0, 1, 4, 2, 3, OP_RESHAPE, {  6, 4,-1, 0 }, 3, 0, 0, 1,  6, 4, 1, 1 },
  { 0, 1, 4, 2, 3, OP_RESHAPE, {  2, 6,-1, 0 }, 3, 0, 0, 1,  2, 6, 2, 1 },
  // ---- F. 短 shape 但没有 -1（Flatten 实际产出的形态，unknown_num==0）----
  { 0, 1, 2, 3, 4, OP_RESHAPE, { 24, 0, 0, 0 }, 1, 0, 0, 1, 24, 1, 1, 1 },
  { 0, 1, 2, 3, 4, OP_RESHAPE, {  4, 6, 0, 0 }, 2, 0, 0, 1,  4, 6, 1, 1 },
  { 0, 1, 2, 3, 4, OP_RESHAPE, {  6, 4, 0, 0 }, 2, 0, 0, 1,  6, 4, 1, 1 },
  { 0, 1, 2, 3, 4, OP_RESHAPE, {  2, 3, 4, 0 }, 3, 0, 0, 1,  2, 3, 4, 1 },
  // ---- G. Flatten_NCHW：它会真的构造出 1/2/3 元素的 shape ----
  //  start_axis=0,end_axis=0 -> [N,C,H,W]   4 元素，恒等
  { 0, 1, 2, 3, 4, OP_FLATTEN, { 0,0,0,0 }, 0, 0, 0, 1,  1, 2, 3, 4 },
  { 0, 2, 3, 2, 2, OP_FLATTEN, { 0,0,0,0 }, 0, 0, 0, 1,  2, 3, 2, 2 },
  //  start_axis=0,end_axis=1 -> [N*C, H, W]   3 元素
  { 0, 1, 2, 3, 4, OP_FLATTEN, { 0,0,0,0 }, 0, 0, 1, 1,  2, 3, 4, 1 },
  //  start_axis=0,end_axis=2 -> [N*C*H, W]   2 元素
  { 0, 1, 2, 3, 4, OP_FLATTEN, { 0,0,0,0 }, 0, 0, 2, 1,  6, 4, 1, 1 },
  //  start_axis=0,end_axis=3 -> [N*C*H*W]     1 元素
  { 0, 1, 2, 3, 4, OP_FLATTEN, { 0,0,0,0 }, 0, 0, 3, 1, 24, 1, 1, 1 },
  //  start_axis=1,end_axis=3 -> [N, C*H*W]   2 元素
  { 0, 1, 2, 3, 4, OP_FLATTEN, { 0,0,0,0 }, 0, 1, 3, 1,  1,24, 1, 1 },
  { 0, 2, 3, 2, 2, OP_FLATTEN, { 0,0,0,0 }, 0, 1, 3, 1,  2,12, 1, 1 },
  //  start_axis=2,end_axis=2 -> [N, C, H, W]  恒等（H 的乘积就是 H 自己）
  { 0, 1, 2, 3, 4, OP_FLATTEN, { 0,0,0,0 }, 0, 2, 2, 1,  1, 2, 3, 4 },
  //  start_axis=2,end_axis=3 -> [N, C, H*W]   3 元素
  { 0, 1, 2, 3, 4, OP_FLATTEN, { 0,0,0,0 }, 0, 2, 3, 1,  1, 2,12, 1 },
  { 0, 2, 3, 2, 2, OP_FLATTEN, { 0,0,0,0 }, 0, 2, 3, 1,  2, 3, 4, 1 },
  // **这一条我第一版写成"应当被拒"，是表算错了**：
  // {1,2,3,0} -> new_dim = {1,2,3,in_W=4} = 24 == count，**本来就成立**。
  // 我当时在注释里都写出了"-> 96"，却没发现 in_W 本身就是 4。
  // 留在成立组里，注释记下误判，免得下次有人"顺手"再判一次。
  { 0, 1, 2, 3, 4, OP_RESHAPE, { 1,2,3,0 }, 4, 0, 0, 1, 1, 2, 3, 4 },
  // ---- H. 应当被拒 ----
  { 0, 1, 2, 3, 4, OP_RESHAPE, { 1,1,1,1,1 }, 5, 0, 0, 0, 0, 0, 0, 0 },  // 超过 4 维
  { 0, 1, 2, 3, 4, OP_RESHAPE, { 1,-1,-1,1 }, 4, 0, 0, 0, 0, 0, 0, 0 },  // 两个 -1
  { 0, 1, 2, 3, 4, OP_RESHAPE, { 1,1,1,1 }, 4, 0, 0, 0, 0, 0, 0, 0 },     // 乘积 1 != 24
  { 0, 1, 2, 3, 4, OP_RESHAPE, { 1,2,3,5 }, 4, 0, 0, 0, 0, 0, 0, 0 },     // 乘积 30 != 24
  { 0, 1, 2, 3, 4, OP_RESHAPE, { 5,-1,1,1 }, 4, 0, 0, 0, 0, 0, 0, 0 },     // 24 除不尽 5
  { 0, 1, 2, 3, 4, OP_RESHAPE, { 5,-1 }, 2, 0, 0, 0, 0, 0, 0, 0 },         // 短 shape 也除不尽
  { 0, 1, 2, 3, 4, OP_RESHAPE, { 1,-1,-1 }, 3, 0, 0, 0, 0, 0, 0, 0 },     // 短 shape 两个 -1
  { 0, 1, 2, 3, 4, OP_RESHAPE, {24,0,0,0 }, 4, 0, 0, 0, 0, 0, 0, 0 },     // 0 沿用输入维 -> 24*2*3*4=576
};
static const int N_CASE = (int)(sizeof(g_cases) / sizeof(g_cases[0]));

// 结果文件：ok bad worst note first oN oC oH oW
// note: 0 无事 / 1 搭建失败 / 2 返回值与期望相反 / 3 输出尺寸算错
//       / 4 数据错 / 5 应当被拒却收下了 / 6 应当被拒却写了输出
static const char* g_note[] = {
    "", "搭建失败", "**返回值与期望相反**", "**输出尺寸算错**",
    "**数据错**", "**应当被拒却收下了**", "**应当被拒却写了输出**"
};

static void write_res(long ok, long bad, double worst, int note, long first, int oN, int oC, int oH, int oW)
{
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %.6f %d %ld %d %d %d %d\n", ok, bad, worst, note, first, oN, oC, oH, oW); fclose(f); }
}

static void run_case(const Case& c)
{
    ZQ_CNN_Tensor4D* in = make(c.kind);
    ZQ_CNN_Tensor4D* out = make(c.kind);
    long bad = 0, first = -1; int note = 0;
    if (!in || !out
        || !in->ChangeSize(c.N, c.H, c.W, c.C, 0, 0)
        || !out->ChangeSize(1, 1, 1, 1, 0, 0)) {
        write_res(0, 1, 0.0, 1, -1, 0, 0, 0, 0);
        delete in; delete out; return;
    }

    const int in_ss = in->GetSliceStep(), in_ws = in->GetWidthStep(), in_ps = in->GetPixelStep();
    float* ip = in->GetFirstPixelPtr();
    // 先把**整个 slice（含补齐区）**刷成哨兵，再填真值。
    // 补齐区保持 -31337：万一实现越界读到补齐区，那一格必然对不上期望值。
    for (int i = 0; i < c.N * in_ss; i++) ip[i] = -31337.0f;
    for (int n = 0; n < c.N; n++)
        for (int ch = 0; ch < c.C; ch++)
            for (int h = 0; h < c.H; h++)
                for (int w = 0; w < c.W; w++)
                    ip[(size_t)n * in_ss + (size_t)h * in_ws + (size_t)w * in_ps + ch] = val(n, ch, h, w);

    // 输出缓冲的快照，用来判"被拒时有没有被动过"。
    // **可读长度必须按张量自己的 N 算**（此刻只有 1x1x1x1）—— 附录 DD 的教训：
    // 门禁自己越界读的症状和被测代码坏了长得一模一样。
    const int cap = out->GetSliceStep() * out->GetN();
    std::vector<float> before(cap, 0.0f);
    for (int i = 0; i < cap; i++) before[i] = out->GetFirstPixelPtr()[i];

    bool r;
    if (c.op == OP_FLATTEN) {
        r = in->Flatten_NCHW(*out, c.ax0, c.ax1, 1);
    } else {
        // **只用 push_back 构造**：capacity 正好等于 size，
        // 于是 shape[shape_dim..3] 正好落在分配块之外，ASan 才抓得到。
        // 如果谁把这里改成 reserve(4) / 用 4 元素定长数组，判据会**静默失效**
        // （不越界了，也就不报了）—— 这正是附录 DA.2「grep 静默失败」的同款陷阱。
        std::vector<int> shape;
        for (int i = 0; i < c.slen; i++) shape.push_back(c.s[i]);
        r = in->Reshape_NCHW(*out, shape, 1);
    }
    // **调用之后必须重新取首指针**：ChangeSize 是 free + malloc，缓存的指针随即悬空。
    float* op = out->GetFirstPixelPtr();

    if (c.expect_ok == 0) {
        if (r) { bad++; note = 5; }
        const int lim = (out->GetSliceStep() * out->GetN() < cap) ? out->GetSliceStep() * out->GetN() : cap;
        for (int i = 0; i < lim; i++)
            if (op[i] != before[i]) { if (!note) note = 6; bad++; break; }
        write_res(1L - bad, bad, 0.0, note, -1, out->GetN(), out->GetC(), out->GetH(), out->GetW());
    } else if (!r) {
        write_res(0, 1, 0.0, 2, -1, 0, 0, 0, 0);
    } else {
        const int oN = out->GetN(), oC = out->GetC(), oH = out->GetH(), oW = out->GetW();
        if (oN != c.eN || oC != c.eC || oH != c.eH || oW != c.eW) {
            write_res(0, 1, 0.0, 3, -1, oN, oC, oH, oW);
        } else {
            const int o_ps = out->GetPixelStep(), o_ws = out->GetWidthStep(), o_ss = out->GetSliceStep();
            const int inCHW = c.C * c.H * c.W, inHW = c.H * c.W;
            for (int on = 0; on < oN; on++)
                for (int oc = 0; oc < oC; oc++)
                    for (int oh = 0; oh < oH; oh++)
                        for (int ow = 0; ow < oW; ow++) {
                            // 独立参考：out 的 **NCHW 线性下标**解码回输入的 (n,c,h,w)。
                            // 这与实现里 idx 逐层 ++ 的顺序一致，但不是抄它的循环。
                            int k = ((on * oC + oc) * oH + oh) * oW + ow;
                            const int rn = k / inCHW; k %= inCHW;
                            const int rc = k / inHW;  k %= inHW;
                            const int rh = k / c.W;   const int rw = k % c.W;
                            const double e = (double)val(rn, rc, rh, rw);
                            const double g = (double)op[(size_t)on * o_ss + (size_t)oh * o_ws
                                                        + (size_t)ow * o_ps + oc];
                            if (!(g == e)) {                     // 值都是精确整数，直接比相等
                                if (first < 0) first = rn * 1000000L + rc * 10000L + rh * 100L + rw;
                                bad++;
                            }
                        }
            write_res(1L - bad, bad, 0.0, bad ? 4 : 0, first, oN, oC, oH, oW);
        }
    }
    delete in; delete out;
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

// ---------------------------------------------------------------------
// 第二阶段：同源拷贝 #2（ZQ_CNN_Tensor4D_NCHWC）
// ---------------------------------------------------------------------
// 只跑**形状**这一半：那一版的 Reshape_NCHW 是非静态成员，要跑就得实例化
// 一个具体的 NCHWC 子类（这个头文件里 concrete 类在别处），成本高；
// 而缺陷在 **static 的 get_size** 里，静态调用就能全覆盖，
// 且 4 个输出参数是引用，形状对不对直接读得到。
#define RES_FILE_N "/tmp/zq_reshape_res_n.txt"

static void run_case_nchwc(const Case& c)
{
    if (c.op != OP_RESHAPE) return;          // Flatten 用例不产生 shape，跳过
    // 同样只用 push_back，让 capacity == size，越界读才落在块外
    std::vector<int> shape;
    for (int i = 0; i < c.slen; i++) shape.push_back(c.s[i]);
    int a = -1, b = -1, d = -1, e = -1;
    const bool r = ZQ_CNN_Tensor4D_NCHWC::Reshape_NCHW_get_size(shape, c.N, c.C, c.H, c.W, a, b, d, e);
    int note = 0;
    if (c.expect_ok == 0) {
        if (r) note = 5;
    } else if (!r) note = 2;
    else if (a != c.eN || b != c.eC || d != c.eH || e != c.eW) note = 3;
    FILE* f = fopen(RES_FILE_N, "w");
    if (f) { fprintf(f, "%d %d %d %d %d %d\n", note, a, b, d, e, r ? 1 : 0); fclose(f); }
}

static void one_nchwc(const Case& c)
{
    if (c.op != OP_RESHAPE) return;
    g_case++;
    remove(RES_FILE_N);
    pid_t pid = fork();
    if (pid == 0) { zq_child_silence_stderr(); run_case_nchwc(c); _exit(0); }
    int st = 0; waitpid(pid, &st, 0);
    int note = 0, a = 0, b = 0, d = 0, e = 0, rr = 0, have = 0;
    FILE* f = fopen(RES_FILE_N, "r");
    if (f) { have = (fscanf(f, "%d %d %d %d %d %d", &note, &a, &b, &d, &e, &rr) == 6); fclose(f); }
    char tag[192], sh[64]; int p = 0; sh[0] = 0;
    for (int i = 0; i < c.slen && i < 5; i++)
        p += snprintf(sh + p, sizeof(sh) - p, "%s%d", i ? "," : "", c.s[i]);
    snprintf(tag, sizeof(tag), "NCHWC N%dC%dH%dW%d -> {%s}/%d%s",
             c.N, c.C, c.H, c.W, sh, c.slen, c.expect_ok ? "" : " 应当被拒");
    if (!have || WIFSIGNALED(st)) {
        g_crash++;
        printf("  %-12s %-38s  没跑完%s\n", "NCHWC拷贝", tag,
               WIFSIGNALED(st) ? "（子进程被信号杀掉）" : "");
        return;
    }
    if (note == 0) { g_ok++; return; }
    g_bad++;
    printf("  %-12s %-38s  %s", "NCHWC拷贝", tag, g_note[note]);
    if (note == 3) printf("（得到 %dx%dx%dx%d，应为 %dx%dx%dx%d）", a, b, d, e, c.eN, c.eC, c.eH, c.eW);
    printf("\n");
}

static void one(const Case& c)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) { zq_child_silence_stderr(); run_case(c); _exit(0); }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0, first = -1; double worst = 0; int note = 0, have = 0;
    int oN = 0, oC = 0, oH = 0, oW = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%ld %ld %lf %d %ld %d %d %d %d",
                            &ok, &bad, &worst, &note, &first, &oN, &oC, &oH, &oW) == 9); fclose(f); }
    char tag[192];
    if (c.op == OP_FLATTEN)
        snprintf(tag, sizeof(tag), "N%dC%dH%dW%d Flatten(%d,%d)%s",
                 c.N, c.C, c.H, c.W, c.ax0, c.ax1, c.expect_ok ? "" : " 应当被拒");
    else {
        char sh[64]; int p = 0; sh[0] = 0;
        for (int i = 0; i < c.slen && i < 5; i++)
            p += snprintf(sh + p, sizeof(sh) - p, "%s%d", i ? "," : "", c.s[i]);
        snprintf(tag, sizeof(tag), "N%dC%dH%dW%d -> {%s}/%d%s",
                 c.N, c.C, c.H, c.W, sh, c.slen, c.expect_ok ? "" : " 应当被拒");
    }
    // 判据 CJ.4：结果文件读不出来 = 失败，不是"跳过"
    if (!have) {
        g_crash++;
        printf("  %-12s %-38s  没跑完%s\n", g_kind_name[c.kind], tag,
               WIFSIGNALED(st) ? "（子进程被信号杀掉）" : "");
        return;
    }
    if (WIFSIGNALED(st)) {
        g_crash++;
        printf("  %-12s %-38s  CRASH（信号 %d）\n", g_kind_name[c.kind], tag, WTERMSIG(st));
        return;
    }
    if (note == 1) { g_crash++; printf("  %-12s %-38s  搭建失败\n", g_kind_name[c.kind], tag); return; }
    if (bad > 0) {
        g_bad++;
        // FAIL 行必须带细节（附录 CA.5）：只打 FAIL 的话"参考错"和"内核错"长得一样
        printf("  %-12s %-38s  %s %ld 项", g_kind_name[c.kind], tag, g_note[note], bad);
        if (note == 3)
            printf("（得到 %dx%dx%dx%d，应为 %dx%dx%dx%d）", oN, oC, oH, oW, c.eN, c.eC, c.eH, c.eW);
        else if (note == 4)
            printf("，首个错格应等于输入的 (n,c,h,w)=(%ld,%ld,%ld,%ld)",
                   first / 1000000, (first / 10000) % 100, (first / 100) % 100, first % 100);
        printf("\n");
    } else g_ok++;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("ZQ_CNN_Tensor4D::Reshape_NCHW / Flatten_NCHW 门禁（附录 DY）\n");
    printf("用真实的张量对象测：这两个函数写在成员函数里，绕不开（附录 DC.6）\n");
    printf("判据：恒等 reshape 逐格不变；跨形状对拍；输出尺寸对表；\n");
    printf("      0 = 沿用输入维；-1 可推；短 shape 合法；5 维/双 -1/乘积错必须被拒；\n");
    printf("      被拒时输出缓冲不许被写\n");
    printf("三种张量子类都跑（实现只有一份，但步长来自具体子类）\n");
    printf("另跑一遍 NCHWC 那份同源拷贝的**静态** get_size（附录 DY.2 一起修的那份）\n");
    printf("**注意：附录 DY.2 的越界读只有 ASan 能抓到**（越界字节落在 new_dim[i]==1 的空操作上，\n");
    printf("  非 sanitizer 下结果完全正确）。本门禁在 ASan 轴下才是完整判据。\n\n");
    for (int k = 0; k < 3; k++) {
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        for (int i = 0; i < N_CASE; i++) { Case c = g_cases[i]; c.kind = k; one(c); }
        printf("  %-12s %d 个用例：对 %d，错 %d，崩 %d\n",
               g_kind_name[k], g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    {
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        for (int i = 0; i < N_CASE; i++) one_nchwc(g_cases[i]);
        printf("  %-12s %d 个用例：对 %d，错 %d，崩 %d（静态 get_size，只查形状）\n",
               "NCHWC拷贝", g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
