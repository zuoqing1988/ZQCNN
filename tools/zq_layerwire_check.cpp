/* UNUSED 层类型的**接线**门禁 —— 附录 EC
 *
 * 为什么需要这道门禁
 * ------------------
 * 可达性门禁（附录 C7）说：36 种层类型里 **15 种是 UNUSED** ——
 * 不会被任何随仓库发布的模型跑到。而报告开头的威胁模型写明
 * **模型文件是不可信输入**，所以这 15 种是攻击者构造模型时**可以点到**的代码。
 *
 * 它们此前既没有门禁、也没有 sample 跑过。已有的 43 道门禁测的是**内核**
 * （`layers_c/*.c` 与 `ZQ_CNN_Forward_SSEUtils` 的小写辅助函数），
 * **测不到"层有没有把参数接对"** —— 而"层"和"内核"是两份实现，
 * 正是 AGENTS.md 第 13 条「同仓的两份实现互为对照」当中的应用对象。
 *
 * 设计：桩下在**小写内部辅助函数**那一层
 * ------------------------------------
 * `ZQ_CNN_Forward_SSEUtils` 的**公开包装方法**（`ScalarOperation_*` / `Reduction*` /
 * `LRN_across_channels`）全部是**头文件里的 inline 实现**，
 * 定义在 .cpp 里的是**小写辅助函数** `_scalaroperation_* / _reduction_* / _lrn_*`。
 *
 * 于是：层 ->（inline 包装，真实）-> 我的桩（只记录，不计算）。
 * 桩记下三样东西，正好就是"接线"的全部：
 *   in_data  / out_data  落在**哪个张量的缓冲**里
 *   scalar / axis / local_size  有没有被层偷偷改
 *
 * **为什么不编 Forward_SSEUtils.cpp**：一个 TU 引用半个库
 * （addbias / prelu / avgpooling / batchnorm / conv / conv_gemm …），
 * 最后拖进 `zq_cnn_convolution_gemm_32f_align_c.c`（**单编 5 分钟以上**），
 * 只能挂进 SLOW 集合。为一个"接线"测试付这个代价不划算。
 *
 * **为什么不用"跑一遍真内核再对拍"**：那样"层和内核一起算错"会被掩盖过去 ——
 * 而这正是本门禁要防的那类错。桩与实现**不同源**。
 *
 * 判据
 * ----
 * 1. **输入指针必须落在 bottoms[0] 的缓冲里**
 * 2. **输出指针必须落在 tops[0] 的缓冲里**（就地做那一支允许 out 落在 bottom 里，
 *    因为 top==bottom 时本来就是同一块）
 * 3. **标量 / axis / local_size 必须原样传下去**
 * 4. **非法参数必须在到达内核之前被层拒掉**（LRN 的偶数或 0 `local_size`、
 *    Reduction 的越界 `axis`）—— 这一条是**层的职责**
 * 5. **bottom 的内容绝不能被改**
 *
 * 明确不在范围内：内核自身的数值语义，以及不走小写辅助函数的层
 * （`Sqrt` / `Scale` / `ScaleWithBias` 是纯头文件 inline，没有可下的桩点）。
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
#include "ZQCNN/ZQ_CNN_Tensor4D.h"
#include "ZQCNN/ZQ_CNN_Layer.h"

using namespace ZQ;

// ---------------------------------------------------------------------
// 记录桩：只记参数，不做任何计算
// ---------------------------------------------------------------------
// **调的是哪个 helper 本身就是"接线"的一部分** —— 比如 DIV 被层改写成
// `Mul(1/scalar)`：改对了是"等价变换"，改错了就是"算错但看不出来"，
// 所以必须把 helper 的身份一起记下来。
enum { H_MUL = 1, H_ADD, H_MAX, H_MIN, H_POW, H_RDIV, H_RMINUS, H_REDSUM, H_REDMEAN, H_LRN };
static const char* g_helper[] = { "?", "Mul", "Add", "Max", "Min", "Pow", "Rdiv", "Rminus",
                                  "ReductionSum", "ReductionMean", "LRN" };
struct Rec {
    const float* in;
    float* out;
    float f1, f2;
    int i1, i2;
    int in_place;
    int helper;
};
static Rec g_rec[8];
static int g_nrec = 0;
static int g_helper_id = 0;     // 每个桩在 rec_call 之前设一下自己是谁

static void rec_call(const float* in, float* out, float f1, float f2, int i1, int i2, int inplace)
{
    if (g_nrec < 8) {
        g_rec[g_nrec].in = in; g_rec[g_nrec].out = out;
        g_rec[g_nrec].f1 = f1; g_rec[g_nrec].f2 = f2;
        g_rec[g_nrec].i1 = i1; g_rec[g_nrec].i2 = i2;
        g_rec[g_nrec].in_place = inplace;
        g_rec[g_nrec].helper = g_helper_id;
        g_nrec++;
    }
}

#define ZQ_STUB_HELPER(NAME, HID) \
void ZQ_CNN_Forward_SSEUtils::NAME(int align_mode, float scalar, const float* in_data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep, \
    float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep) \
{ (void)align_mode;(void)N;(void)H;(void)W;(void)C;(void)pixStep;(void)widthStep;(void)sliceStep; \
  (void)out_pixStep;(void)out_widthStep;(void)out_sliceStep; \
  g_helper_id = HID; rec_call(in_data, out_data, scalar, 0, 0, 0, 0); } \
void ZQ_CNN_Forward_SSEUtils::NAME(int align_mode, float scalar, float* data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep) \
{ (void)align_mode;(void)N;(void)H;(void)W;(void)C;(void)pixStep;(void)widthStep;(void)sliceStep; \
  g_helper_id = HID; rec_call(data, data, scalar, 0, 0, 0, 1); }

ZQ_STUB_HELPER(_scalaroperation_mul, H_MUL)
ZQ_STUB_HELPER(_scalaroperation_add, H_ADD)
ZQ_STUB_HELPER(_scalaroperation_max, H_MAX)
ZQ_STUB_HELPER(_scalaroperation_min, H_MIN)
ZQ_STUB_HELPER(_scalaroperation_pow, H_POW)
ZQ_STUB_HELPER(_scalaroperation_rdiv, H_RDIV)
ZQ_STUB_HELPER(_scalaroperation_rminus, H_RMINUS)
#undef ZQ_STUB_HELPER

#define ZQ_STUB_RED(NAME, HID) \
void ZQ_CNN_Forward_SSEUtils::NAME(int align_mode, const float* in_data, int N, int H, int W, int C, int axis, bool keepdims, \
    int pixStep, int widthStep, int sliceStep, float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep) \
{ (void)align_mode;(void)N;(void)H;(void)W;(void)C;(void)pixStep;(void)widthStep;(void)sliceStep; \
  (void)out_pixStep;(void)out_widthStep;(void)out_sliceStep; \
  g_helper_id = HID; rec_call(in_data, out_data, 0, 0, axis, keepdims ? 1 : 0, 0); }
ZQ_STUB_RED(_reduction_sum, H_REDSUM)
ZQ_STUB_RED(_reduction_mean, H_REDMEAN)
#undef ZQ_STUB_RED

void ZQ_CNN_Forward_SSEUtils::_lrn_across_channels(int align_mode, int local_size, float alpha, float beta, float k,
    const float* in_data, int N, int H, int W, int C, int pixStep, int widthStep, int sliceStep,
    float* out_data, int out_pixStep, int out_widthStep, int out_sliceStep)
{ (void)align_mode;(void)N;(void)H;(void)W;(void)C;(void)pixStep;(void)widthStep;(void)sliceStep;
  (void)beta;(void)out_pixStep;(void)out_widthStep;(void)out_sliceStep;
  g_helper_id = H_LRN; rec_call(in_data, out_data, alpha, k, local_size, 0, 0); }

#define RES_FILE "/tmp/zq_layerwire_res.txt"

enum { K_A0 = 0, K_A128 = 1, K_A256 = 2 };
static const char* g_kind_name[] = { "align0", "align128bit", "align256bit" };

static ZQ_CNN_Tensor4D* make_t(int kind)
{
    if (kind == K_A0) return new ZQ_CNN_Tensor4D_NHW_C_Align0();
    if (kind == K_A128) return new ZQ_CNN_Tensor4D_NHW_C_Align128bit();
    return new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
}

enum { OP_SCALAR = 0, OP_REDUCTION, OP_LRN, OP_SQUEEZE, OP_COPY };
static const char* g_opname[] = { "ScalarOp", "Reduction", "LRN", "Squeeze", "Copy" };
// note: 0 无事 / 1 搭建失败 / 2 返回值与期望相反 / 3 该调内核却一次都没调
//       / 4 输入指针不在 bottom 里 / 5 输出指针不在 top 里 / 6 参数被层改了
//       / 7 非法参数却仍然放行 / 8 bottom 的内容被改了
static const char* g_note[] = {
    "", "搭建失败", "**返回值与期望相反**", "**该调内核却一次都没调**",
    "**输入指针不在 bottoms[0] 里**", "**输出指针不在 tops[0] 里**", "**参数被层改了**",
    "**非法参数却仍然放行**", "**bottom 的内容被改了**"
};

struct Case {
    int op;
    int N, C, H, W;
    int a;              // ScalarOp 的 operation / LRN 的 local_size / Reduction 的 axis
    float f;            // 标量
    int expect_kernel;  // 1 = 应当调到内核；0 = 应当被层在到达内核之前拒掉
};

static float val(int n, int c, int h, int w)
{
    return (float)(n * 32768 + c * 1024 + h * 32 + w) * 0.001f + 1.0f;
}

// 指针是否落在 [base, base + span) 里
static bool in_span(const float* p, const float* base, int span)
{
    return p >= base && p < base + span;
}

static void run_case(const Case& c, int kind)
{
    g_nrec = 0;
    memset(g_rec, 0, sizeof(g_rec));
    ZQ_CNN_Tensor4D* bottom = make_t(kind);
    ZQ_CNN_Tensor4D* top = make_t(kind);
    long bad = 0, first = -1; int note = 0;

    if (!bottom->ChangeSize(c.N, c.H, c.W, c.C, 0, 0)) { note = 1; bad++; }
    else if (!top->ChangeSize(1, 1, 1, 1, 0, 0)) { note = 1; bad++; }
    else {
        const int bss = bottom->GetSliceStep(), bps = bottom->GetPixelStep(), bws = bottom->GetWidthStep();
        float* bp = bottom->GetFirstPixelPtr();
        for (int i = 0; i < c.N * bss; i++) bp[i] = -31337.0f;   // 补齐区哨兵
        for (int n = 0; n < c.N; n++)
            for (int ch = 0; ch < c.C; ch++)
                for (int h = 0; h < c.H; h++)
                    for (int w = 0; w < c.W; w++)
                        bp[(size_t)n * bss + (size_t)h * bws + (size_t)w * bps + ch] = val(n, ch, h, w);
        const int bspan = c.N * bss;

        std::vector<ZQ_CNN_Tensor4D*> vb, vt;
        vb.push_back(bottom); vt.push_back(top);
        bool r = false;

        if (c.op == OP_SCALAR) {
            ZQ_CNN_Layer_ScalarOperation* L = new ZQ_CNN_Layer_ScalarOperation();
            L->operation = c.a; L->scalar = c.f;
            r = L->Forward(&vb, &vt);
            delete L;
        } else if (c.op == OP_REDUCTION) {
            ZQ_CNN_Layer_Reduction* L = new ZQ_CNN_Layer_Reduction();
            L->axis = c.a; L->keepdims = true;
            L->operation = ZQ_CNN_Layer_Reduction::REDUCTION_SUM;
            r = L->Forward(&vb, &vt);
            delete L;
        } else if (c.op == OP_LRN) {
            ZQ_CNN_Layer_LRN* L = new ZQ_CNN_Layer_LRN();
            L->local_size = c.a; L->alpha = 1.0f; L->beta = 0.5f; L->k = 2.0f;
            L->operation = ZQ_CNN_Layer_LRN::LRN_ACROSS_CHANNELS;
            r = L->Forward(&vb, &vt);
            delete L;
        } else if (c.op == OP_SQUEEZE) {
            ZQ_CNN_Layer_Squeeze* L = new ZQ_CNN_Layer_Squeeze();
            L->dim.push_back(1);
            L->SetBottomDim(c.C, c.H, c.W);
            int tC, tH, tW; L->GetTopDim(tC, tH, tW);
            top->SetShape(c.N, tC, tH, tW);
            r = L->Forward(&vb, &vt);
            delete L;
        } else {
            ZQ_CNN_Layer_Copy* L = new ZQ_CNN_Layer_Copy();
            r = L->Forward(&vb, &vt);
            delete L;
        }

        if (c.op == OP_SQUEEZE || c.op == OP_COPY) {
            // 这两个层**根本不调内核**（只 CopyData）。钉住它们"是恒等桩"这个契约：
            // top 逐格等于 bottom、bottom 未被改、形状不变。
            // "dim 被读进来却完全不用"是**未记录的契约** ——
            // 有人真去实现 squeeze 时这里会红，从而强制他同时想清楚 GetTopDim。
            if (!r) { note = 2; bad++; }
            else {
                if (top->GetN() != c.N || top->GetC() != c.C || top->GetH() != c.H || top->GetW() != c.W) {
                    note = 3; bad++;
                } else {
                    const int tss = top->GetSliceStep(), tps = top->GetPixelStep(), tws = top->GetWidthStep();
                    const float* tp = top->GetFirstPixelPtr();
                    for (int n = 0; n < c.N && bad < 4; n++)
                        for (int ch = 0; ch < c.C && bad < 4; ch++)
                            for (int h = 0; h < c.H && bad < 4; h++)
                                for (int w = 0; w < c.W && bad < 4; w++) {
                                    const size_t o = (size_t)n * tss + (size_t)h * tws + (size_t)w * tps + ch;
                                    if (tp[o] != val(n, ch, h, w)) {
                                        if (first < 0) first = (long)o;
                                        if (!note) note = 3; bad++;
                                    }
                                }
                }
            }
        } else if (!c.expect_kernel) {
            // 判据 4：非法参数必须在**到达内核之前**被层拒掉
            if (g_nrec != 0) { note = 7; bad++; }
            else if (r) { note = 2; bad++; }
        } else if (!r) { note = 2; bad++; }
        else if (g_nrec == 0) { note = 3; bad++; }
        else {
            const float* tb = bottom->GetFirstPixelPtr();
            const float* tt = top->GetFirstPixelPtr();
            const int tspan = top->GetN() * top->GetSliceStep();
            for (int i = 0; i < g_nrec && bad < 4; i++) {
                const Rec& R = g_rec[i];
                // 判据 1：输入指针必须落在 bottom 里
                if (!in_span(R.in, tb, bspan)) { if (!note) note = 4; bad++; }
                // 判据 2：输出指针必须落在 top 里；就地做那一支允许落在 bottom 里
                if (!in_span(R.out, tt, tspan) && !in_span(R.out, tb, bspan)) {
                    if (!note) note = 5; bad++;
                }
                // 判据 3：参数原样传下去 —— **但 DIV / MINUS 是例外，因为层
                // 故意把它们改写成另一个 helper**：
                //     DIV   -> Mul(1.0f/scalar)
                //     MINUS -> Add(-scalar)
                // 这是**等价变换，改对了才对**。第一版判"标量必须原样传下去"，
                // 于是 DIV / MINUS 两例红，看起来像接线错 —— 那是门禁判据错了。
                // 现在改成"落到哪个 helper + 变换后的标量是多少"，把映射本身钉住。
                if (c.op == OP_SCALAR) {
                    int exp_helper = H_MUL;
                    float exp_scalar = c.f;
                    if (c.a == 1)      { exp_helper = H_MUL;  exp_scalar = 1.0f / c.f; }   // DIV
                    else if (c.a == 2) { exp_helper = H_ADD;  exp_scalar = c.f; }           // ADD
                    else if (c.a == 3) { exp_helper = H_ADD;  exp_scalar = -c.f; }          // MINUS
                    else if (c.a == 4) { exp_helper = H_MAX;  exp_scalar = c.f; }           // MAX
                    else if (c.a == 5) { exp_helper = H_MIN;  exp_scalar = c.f; }           // MIN
                    else if (c.a == 6) { exp_helper = H_POW;  exp_scalar = c.f; }           // POW
                    else if (c.a == 7) { exp_helper = H_RDIV; exp_scalar = c.f; }           // RDIV
                    else if (c.a == 8) { exp_helper = H_RMINUS; exp_scalar = c.f; }         // RMINUS
                    if (R.helper != exp_helper) { if (!note) note = 6; bad++; }
                    if (R.f1 != exp_scalar) { if (!note) note = 6; bad++; }
                }
                if (c.op == OP_LRN && R.helper != H_LRN) { if (!note) note = 6; bad++; }
                if (c.op == OP_LRN && R.i1 != c.a) { if (!note) note = 6; bad++; }
                if (c.op == OP_REDUCTION) {
                    if (R.helper != H_REDSUM) { if (!note) note = 6; bad++; }
                    if (R.i1 != c.a) { if (!note) note = 6; bad++; }
                }
            }
            // 判据 5：bottom 一个字节都不许被改
            for (int n = 0; n < c.N && bad < 4; n++)
                for (int ch = 0; ch < c.C && bad < 4; ch++)
                    for (int h = 0; h < c.H && bad < 4; h++)
                        for (int w = 0; w < c.W && bad < 4; w++)
                            if (bp[(size_t)n * bss + (size_t)h * bws + (size_t)w * bps + ch] != val(n, ch, h, w)) {
                                if (first < 0) first = ((n * c.C + ch) * c.H + h) * c.W + w;
                                if (!note) note = 8; bad++;
                            }
        }
    }
    delete bottom; delete top;
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %d %ld\n", 1L - bad, bad, note, first); fclose(f); }
}

static const Case g_cases[] = {
  // ScalarOperation：9 个操作。判据 3 是"标量原样传下去"
  { OP_SCALAR, 1, 3, 2, 2, 0, 2.0f, 1 },    // MUL
  { OP_SCALAR, 1, 3, 2, 2, 1, 2.0f, 1 },    // DIV
  { OP_SCALAR, 1, 3, 2, 2, 2, 2.0f, 1 },    // ADD
  { OP_SCALAR, 1, 3, 2, 2, 3, 2.0f, 1 },    // MINUS
  { OP_SCALAR, 1, 3, 2, 2, 4, 2.0f, 1 },    // MAX
  { OP_SCALAR, 1, 3, 2, 2, 5, 2.0f, 1 },    // MIN
  { OP_SCALAR, 1, 3, 2, 2, 6, 2.0f, 1 },    // POW
  { OP_SCALAR, 1, 3, 2, 2, 7, 2.0f, 1 },    // RDIV
  { OP_SCALAR, 1, 3, 2, 2, 8, 2.0f, 1 },    // RMINUS
  { OP_SCALAR, 2, 5, 2, 3, 0, 0.5f, 1 },
  { OP_SCALAR, 1, 3, 2, 2, 99, 2.0f, 0 },   // 未知 operation：层必须自己拒
  // Reduction：axis 原样传下去；越界必须在到达内核之前被拒
  { OP_REDUCTION, 1, 4, 2, 2, 0, 0.f, 1 },
  { OP_REDUCTION, 1, 4, 2, 2, 1, 0.f, 1 },
  { OP_REDUCTION, 1, 4, 2, 2, 2, 0.f, 1 },
  { OP_REDUCTION, 1, 4, 2, 2, 3, 0.f, 1 },
  { OP_REDUCTION, 1, 4, 2, 2, 4, 0.f, 0 },
  { OP_REDUCTION, 1, 4, 2, 2, -1, 0.f, 0 },
  // LRN：local_size 原样传下去；非法值必须在到达内核之前被拒
  { OP_LRN, 1, 8, 4, 4, 3, 0.f, 1 },
  { OP_LRN, 1, 8, 4, 4, 5, 0.f, 1 },
  { OP_LRN, 1, 8, 4, 4, 0, 0.f, 0 },    // 0：前几轮已被包装里的 `% 2 != 1` 守住
  { OP_LRN, 1, 8, 4, 4, 4, 0.f, 0 },    // 偶数
  { OP_LRN, 1, 8, 4, 4, -3, 0.f, 0 },   // 负数
  // Squeeze / Copy：恒等桩（只 CopyData，不调内核）
  { OP_SQUEEZE, 1, 3, 2, 2, 0, 0.f, 1 },
  { OP_COPY,    1, 3, 2, 2, 0, 0.f, 1 },
  { OP_COPY,    2, 5, 2, 3, 0, 0.f, 1 },
};
static const int N_CASE = (int)(sizeof(g_cases) / sizeof(g_cases[0]));

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(const Case& c, int kind)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) { zq_child_silence_stderr(); run_case(c, kind); _exit(0); }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0, first = -1; int note = 0, have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%ld %ld %d %ld", &ok, &bad, &note, &first) == 4); fclose(f); }
    char tag[128];
    snprintf(tag, sizeof(tag), "%-10s N%dC%dH%dW%d p=%d s=%.2f%s",
             g_opname[c.op], c.N, c.C, c.H, c.W, c.a, c.f, c.expect_kernel ? "" : " (应拒)");
    if (!have) {
        g_crash++;
        printf("  %-12s %-42s  没跑完%s\n", g_kind_name[kind], tag,
               WIFSIGNALED(st) ? "（子进程被信号杀）" : "（结果文件读不出来）");
        return;
    }
    if (WIFSIGNALED(st)) {
        g_crash++;
        printf("  %-12s %-42s  CRASH（信号 %d）\n", g_kind_name[kind], tag, WTERMSIG(st));
        return;
    }
    if (bad > 0) {
        g_bad++;
        printf("  %-12s %-42s  %s %ld 项", g_kind_name[kind], tag, g_note[note], bad);
        if (note == 8 || note == 3) printf("，首个错格线性号 %ld", first);
        printf("\n");
    } else g_ok++;
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("UNUSED 层类型的接线门禁（附录 EC）\n");
    printf("为什么有这道门禁：36 种层类型里 15 种 UNUSED（没有随仓库模型会跑到），\n");
    printf("  而威胁模型写明**模型文件是不可信输入** —— 它们是攻击者构造模型时能点到的代码。\n");
    printf("  已有 43 道门禁测的是**内核**，测不到\"层有没有把参数接对\"。\n");
    printf("设计：桩下在 ZQ_CNN_Forward_SSEUtils 的**小写内部辅助函数**那一层\n");
    printf("  （公开包装方法是头文件 inline，定义在 .cpp 里的是 _scalaroperation_* /\n");
    printf("   _reduction_* / _lrn_*）。层 -> inline 包装 -> 我的桩（只记录不计算）。\n");
    printf("  ① 不用编 Forward_SSEUtils.cpp —— 它一个 TU 引用半个库，会拖进\n");
    printf("     单编 5 分钟以上的 conv GEMM，只能挂进 SLOW 集合；\n");
    printf("  ② 不用\"跑真内核再对拍\" —— 那样\"层和内核一起算错\"会被掩盖过去。\n");
    printf("判据：① 输入指针必须落在 bottoms[0] 里 ② 输出必须落在 tops[0] 里（就地做除外）\n");
    printf("      ③ 标量/axis/local_size 原样传下 ④ 非法参数必须在到达内核之前被层拒掉\n");
    printf("      ⑤ bottom 一个字节都不许被改\n\n");
    for (int k = 0; k < 3; k++) {
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        for (int i = 0; i < N_CASE; i++) one(g_cases[i], k);
        printf("  %-12s %d 个用例：对 %d，错 %d，崩 %d\n",
               g_kind_name[k], g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
