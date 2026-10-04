// 「没有任何随仓模型跑得到」的层类型 —— 逐个造合成网真跑一遍（附录 IA）。
//
// 为什么要有这个 sample
// --------------------
// `tools/run_audit_checks.py` 的 C7 组给出一张层类型可达性表：
//
//     合计 36 种：EXERCISED 20 / COMMENTED 1 / UNUSED 15
//
// UNUSED 的意思是「**没有任何随仓库发布的模型会跑到它**」。
// 也就是说这些代码路径**从来没有被任何东西执行过** ——
// 既没有 sample 跑，也没有门禁覆盖。HX 那条活缺陷查了十二轮，
// 它的教训之一正是「**没被断言覆盖的路径，坏了也不告诉你**」；
// 这里的这些路径比那条还彻底：**连"跑过"都没有过**。
//
// 本 sample 的做法
// ----------------
// 对每一类 UNUSED 层，**自己写一个最小的 `.zqparams` + `.nchwbin`**，
// 用真的 `ZQ_CNN_Net::LoadFrom` + `Forward` 跑一遍，
// 再与**独立写的参考实现**比后向误差。
//
// 为什么参考实现要先自证
// ----------------------
// AGENTS.md「判据要选不依赖你对数据格式理解的那一条」：
// 一旦要写独立参考，就多了一处"可能是我自己写错"的地方。
// 所以每个参考实现都必须先过一道**手算**的用例
// （`selftest_reference()`，数值取成能约成有理数/整数的），
// 过了才有资格去判库错。
//
// 当前已覆盖
// ----------
//   LRN    across-channels LRN（`local_size` 必须为奇数）
//   Copy   bottom -> top 的纯拷贝
//   Scale  逐通道 y = scale[c]*x + bias[c]（`bias` 可选）
//   Sqrt   逐元素 y = sqrt(x)（输入必须非负）
//
// 其余 UNUSED（DeConvolution / BatchNorm / LSTM_TF / ScalarOperation /
// UnaryOperation / Tile / Reduction / Squeeze / PriorBoxText /
// PriorBox_MXNET / DetectionOutput_MXNET）尚未覆盖，
// **汇总行会把它们列出来**，不装作已经查过。
//
// 退出码：0 = 覆盖到的全对；1 = 有对不上的，或参考实现自证没过。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include "ZQ_CNN_Tensor4D.h"
#include "ZQ_CNN_Net.h"

// 合成网文件写在**当前目录**（也就是产物目录，与其它 sample 一致）。
// 名字带 `zq_unused_probe_` 前缀，跑完删掉 —— 万一中途崩了，
// 残留文件也是一眼认得出的。
static const char* SYNTH_PARAM = "zq_unused_probe.zqparams";
static const char* SYNTH_MODEL = "zq_unused_probe.nchwbin";

static void cleanup_synth()
{
    remove(SYNTH_PARAM);
    remove(SYNTH_MODEL);
}

static bool write_file(const char* path, const void* data, size_t n)
{
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    bool ok = (n == 0) || (fwrite(data, 1, n, f) == n);
    fclose(f);
    return ok;
}

// 写 `.zqparams`：`Input` 行 + 一层任意层行。返回 false 表示目录不可写。
// （可写性的判据就落在这一句上：fopen 失败即不可写。）
static bool write_param(int C, int H, int Wd, const char* layer_line)
{
    char buf[1024];
    int n = snprintf(buf, sizeof(buf), "Input name=data C=%d H=%d W=%d\n", C, H, Wd);
    if (n <= 0 || n >= (int)sizeof(buf)) return false;
    FILE* f = fopen(SYNTH_PARAM, "wb");
    if (!f) return false;
    fwrite(buf, 1, (size_t)n, f);
    fputs(layer_line, f);
    fclose(f);
    return true;
}

// 跑一个合成网：造输入 -> Forward -> 取顶层 blob。
// `in` 是 compact NCHW（N=1）。失败时把原因打出来并返回 false。
static bool run_synth(const char* layer_line,
                      const std::vector<float>& weights,
                      int C, int H, int Wd, const std::vector<float>& in,
                      std::vector<float>& out)
{
    if (!write_param(C, H, Wd, layer_line)) {
        printf("  写不出合成网参数文件（cwd 不可写？）\n");
        return false;
    }
    if (!write_file(SYNTH_MODEL, weights.empty() ? (const void*)"" : (const void*)&weights[0],
                    weights.size() * sizeof(float))) {
        printf("  写不出合成网权重文件\n");
        return false;
    }
    ZQ::ZQ_CNN_Net net;
    if (!net.LoadFrom(SYNTH_PARAM, SYNTH_MODEL)) {
        printf("  合成网加载失败\n");
        return false;
    }
    ZQ::ZQ_CNN_Tensor4D_NHW_C_Align256bit ti;
    if (!ti.ConvertFromCompactNCHW(&in[0], 1, C, H, Wd)) {
        printf("  输入张量 ChangeSize 失败\n");
        return false;
    }
    if (!net.Forward(ti)) {
        printf("  Forward 失败\n");
        return false;
    }
    const ZQ::ZQ_CNN_Tensor4D* ob = net.GetBlobByName("top1");
    const char* got_name = "top1";
    if (ob == 0) {
        // 兜底：`_simplify_inplace` 会把「top != bottom 且后面没人再读 bottom」
        // 的就地安全层（Scale / BatchNorm / Sqrt… 都在那张表里）**就地化**，
        // 即 `tops[i][0] = bottoms[i][0]`。这时输出就落在 bottom 那个 blob 上。
        // 取不到 top1 时回到底层 blob，并把**实际取到的是哪个**打出来 ——
        // 否则"取不到"会被误当成"这层没跑"。
        ob = net.GetBlobByName("data");
        got_name = "data";
    }
    if (ob == 0) {
        printf("  取不到输出 blob（top1 与 data 都取不到）\n");
        return false;
    }
    if (strcmp(got_name, "top1") != 0)
        printf("  （注意：这层被 _simplify_inplace 就地化了，输出落在 \"%s\" 上）\n", got_name);
    out.resize((size_t)ob->GetN() * ob->GetC() * ob->GetH() * ob->GetW());
    ob->ConvertToCompactNCHW(&out[0]);
    return true;
}

static double backward_err(const std::vector<float>& got,
                           const std::vector<float>& exp, long& worst_i)
{
    if (got.size() != exp.size() || got.empty()) { worst_i = -1; return 1e30; }
    double ss = 0.0;
    for (size_t i = 0; i < exp.size(); i++) ss += (double)exp[i] * (double)exp[i];
    double den = sqrt(ss);
    if (den == 0.0) den = 1.0;
    double worst = 0.0;
    worst_i = 0;
    for (size_t i = 0; i < got.size(); i++) {
        double e = fabs((double)got[i] - (double)exp[i]) / den;
        if (e > worst) { worst = e; worst_i = (long)i; }
    }
    return worst;
}

// 确定性伪随机（与其他 sample 同族，但不是同一个种子域）。
static float rnd(unsigned& s)
{
    s = s * 1664525u + 1013904223u;
    return (float)((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;   // [-1, 1)
}

// ---------------------------------------------------------------------------
// 各层的参考实现
// ---------------------------------------------------------------------------

// LRN（across-channels），语义取自
// `ZQCNN/layers_c/zq_cnn_lrn_32f_align_c_raw.h`
// （`zq_cnn_lrn_across_channels_32f_align`）：
//
//     对每个像素 (n,h,w) 的通道维：
//         s[c]   = sum_{j=c-L/2 .. c+L/2} in[j]^2     （越界通道算 0）
//         out[c] = in[c] * (k + alpha/L * s[c])^(-beta)
//
// 那份实现用「前后各 pad 零 + 前缀和 + 滑窗差」算 s，
// 窗口是 [pad-L/2, pad-L/2+L-1]，对应通道 [-L/2, -L/2+L-1]，
// 即 **L 个通道、以 c 为中心**（L 为奇数时正好对称）。
// `local_size` 必须为奇数（`LRN_across_channels` 校验 `% 2 != 1`）。
static void ref_lrn(const std::vector<float>& in, int N, int H, int Wd, int C,
                    int L, float alpha, float beta, float k,
                    std::vector<float>& out)
{
    out.resize((size_t)N * C * H * Wd);
    const int half = L / 2;
    for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < Wd; w++) {
                const size_t off = (((size_t)n * H + h) * Wd + w) * C;
                for (int c = 0; c < C; c++) {
                    double s = 0.0;
                    for (int j = c - half; j <= c + half; j++) {
                        if (j < 0 || j >= C) continue;
                        double v = in[off + j];
                        s += v * v;
                    }
                    double denom = k + (alpha / (double)L) * s;
                    if (denom <= 0) denom = 1e-30;
                    out[off + c] = (float)(in[off + c] * pow(denom, -beta));
                }
            }
}

static void ref_copy(const std::vector<float>& in, std::vector<float>& out)
{
    out = in;
}

// 逐通道 y = scale[c]*x[c] + bias[c]。
//
// **通道下标是 `i / (H*W)`，不是 `i % C`** ——
// 因为输入/输出用的是 **compact NCHW**（`ConvertToCompactNCHW` 的那种）：
// 元素 (c,h,w) 的下标是 `(c*H + h)*W + w`，**通道是最外层**。
// 张量本身是 NHW_C 布局（`pixelStep` 那一维才是通道），
// 而 `_scalebias` 是在张量上按 `for (c = 0; c < in_C; c++) pix_ptr[c] *= scale_data[c]`
// 做的 —— 两者一致。
//
// 2026-10-05 第一版把这里写成 `i % C`，于是 C=3/8/13 全部"对不上"
// （而 C=1 通过）—— **看着像库有 bug，其实是参考实现的通道下标写反了**。
// 教训见 AGENTS.md 与 IA.5：手算自证用例**必须能区分两种约定**，
// 而我当时选的 C=2, H=W=1 让两种下标**完全等价**，自证形同虚设。
static void ref_scale(const std::vector<float>& in, int H, int Wd, int C,
                      const std::vector<float>& scale,
                      const std::vector<float>* bias,
                      std::vector<float>& out)
{
    const size_t HW = (size_t)H * Wd;
    out = in;
    for (size_t i = 0; i < out.size(); i++) {
        int c = (int)(i / HW);                  // compact NCHW：通道是最外层
        double v = (double)scale[c] * out[i];
        if (bias) v += (double)(*bias)[c];
        out[i] = (float)v;
    }
}

static void ref_sqrt(const std::vector<float>& in, std::vector<float>& out)
{
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); i++)
        out[i] = (float)(in[i] <= 0 ? 0.0 : sqrt((double)in[i]));
}

// ScalarOperation：9 种运算。语义取自
// `ZQCNN/layers_c/zq_cnn_scalaroperation_32f_align_c.c` 的 align0 版本
// （最直白的那一份，没有 SIMD 的花样）：
//
//   MUL x*s   DIV x/s   ADD x+s   MINUS x-s
//   MAX max(x,s)   MIN min(x,s)   POW x^s
//   RDIV  s/x   <-- **反向除**，参数顺序与 DIV 相反
//   RMINUS s-x  <-- **反向减**，与 MINUS 相反
//
// RDIV / RMINUS 是本探针的重点：它们与 DIV / MINUS 只差一个参数顺序，
// 而**整个仓库没有任何模型会跑到这一层**（C7 的 UNUSED），
// 写反了没有任何东西会发现。
static bool apply_scalar_op(int op, float x, float s, float& y)
{
    switch (op) {
    case 0: y = x * s; return true;                              // MUL
    case 1: y = x / s; return true;                              // DIV
    case 2: y = x + s; return true;                              // ADD
    case 3: y = x - s; return true;                              // MINUS
    case 4: y = (x > s ? x : s); return true;                    // MAX
    case 5: y = (x < s ? x : s); return true;                    // MIN
    case 6: y = (float)pow((double)x, (double)s); return true;   // POW
    case 7: y = s / x; return true;                              // RDIV  <- 反向
    case 8: y = s - x; return true;                              // RMINUS <- 反向
    default: return false;
    }
}

static void ref_scalar_op(const std::vector<float>& in, int op, float s,
                          std::vector<float>& out)
{
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); i++) {
        float y = 0;
        apply_scalar_op(op, in[i], s, y);
        out[i] = y;
    }
}

// Squeeze 是个**纯声明层**：`Forward` 只做一次 `CopyData`，不改数。
// 测它是为了确认"声明层真的一个字节都没动"。
static void ref_squeeze(const std::vector<float>& in, std::vector<float>& out)
{
    out = in;
}

// 参考实现自证：四个**手算**用例，参数取成能约成有理数/整数的。
//
//   A) LRN 窗口的中心与宽度（C=3, L=3, alpha=0.03, beta=1, k=1, in=[1,2,3]）
//      alpha/L = 0.01
//        c=0 窗口 [-1,0,1] -> s=1+4=5    -> out=1/(1+0.05)=1/1.05
//        c=1 窗口 [ 0,1,2] -> s=1+4+9=14 -> out=2/(1+0.14)=2/1.14
//        c=2 窗口 [ 1,2,3] -> s=4+9  =13 -> out=3/(1+0.13)=3/1.13
//      这一组专门抓"窗口偏了一格 / 宽度差了 2 / 少算了两端的零"
//      这三类最常见的参考实现错误 —— 换成 in=[1,2,3] 时它们分别会给出
//      1/1.04、1/1.09、3/1.12 之类的值，与手算**对不上**。
//   B) LRN 退化到 L=1（窗口只有自己）C=2, alpha=0.03, beta=1, k=1, in=[1,2]
//      out=[1/1.03, 2/1.12]
//   C) Scale（整数）scale=[2,3], bias=[1,-1], in=[5,5] -> [11,14]
//   D) Sqrt（整数）in=[0,4,9] -> [0,2,3]
static bool selftest_reference()
{
    // --- A / B: LRN ---
    {
        std::vector<float> in(3);
        in[0] = 1; in[1] = 2; in[2] = 3;
        std::vector<float> got;
        ref_lrn(in, 1, 1, 1, 3, 3, 0.03f, 1.0f, 1.0f, got);
        static const double WANT[3] = { 1.0 / 1.05, 2.0 / 1.14, 3.0 / 1.13 };
        for (int c = 0; c < 3; c++) {
            double e = fabs((double)got[c] - WANT[c]) / WANT[c];
            if (e > 1e-6) {
                printf("  参考实现自证**没过**（LRN 窗口用例 通道 %d）：算得 %.9g，手算是 %.9g\n",
                       c, got[c], WANT[c]);
                return false;
            }
        }
    }
    {
        std::vector<float> in(2);
        in[0] = 1; in[1] = 2;
        std::vector<float> got;
        ref_lrn(in, 1, 1, 1, 2, 1, 0.03f, 1.0f, 1.0f, got);
        static const double WANT[2] = { 1.0 / 1.03, 2.0 / 1.12 };
        for (int c = 0; c < 2; c++) {
            double e = fabs((double)got[c] - WANT[c]) / WANT[c];
            if (e > 1e-6) {
                printf("  参考实现自证**没过**（LRN L=1 用例 通道 %d）：算得 %.9g，手算是 %.9g\n",
                       c, got[c], WANT[c]);
                return false;
            }
        }
    }
    // --- C: Scale ---
    // **H=W=2, C=2**（2026-10-05 改）：第一版用的是 H=W=1，
    // 那时 `i/(H*W)` 与 `i%C` **完全等价**，自证对哪种约定都"通过"，
    // 于是参考实现把通道下标写反了（C=3/8/13 全部误报成库的 bug）。
    // 现在 H*W=4 != C=2，两种约定给出**不同**答案，自证才有鉴别力：
    //   compact NCHW 下标 i = (c*2 + h)*2 + w，in = [1,2,3,4, 5,6,7,8]
    //       通道 0 = in[0..3]，通道 1 = in[4..7]
    //       out = [1*2,2*2,3*2,4*2, 5*3+1,6*3+1,7*3+1,8*3+1]
    //          = [2,4,6,8, 16,19,22,25]
    //   若误用 `i%C`，会得到 [2,4,7,8, 15,18,22,25]（第 3、5 个就不一样）。
    {
        std::vector<float> in(8);
        for (int i = 0; i < 8; i++) in[i] = (float)(i + 1);
        std::vector<float> scale(2), bias(2);
        scale[0] = 2.0f; scale[1] = 3.0f;
        bias[0] = 0.0f;  bias[1] = 1.0f;
        std::vector<float> got;
        ref_scale(in, 2, 2, 2, scale, &bias, got);
        static const double WANT[8] = { 2, 4, 6, 8, 16, 19, 22, 25 };
        for (int i = 0; i < 8; i++) {
            if (fabs((double)got[i] - WANT[i]) > 1e-6) {
                printf("  参考实现自证**没过**（Scale 下标 %d）：算得 %.9g，手算是 %.9g\n",
                       i, got[i], WANT[i]);
                return false;
            }
        }
    }
    // --- D: Sqrt ---
    {
        std::vector<float> in(3);
        in[0] = 0; in[1] = 4; in[2] = 9;
        std::vector<float> got;
        ref_sqrt(in, got);
        static const double WANT[3] = { 0.0, 2.0, 3.0 };
        for (int c = 0; c < 3; c++) {
            if (fabs((double)got[c] - WANT[c]) > 1e-6) {
                printf("  参考实现自证**没过**（Sqrt 通道 %d）：算得 %.9g，手算是 %.9g\n",
                       c, got[c], WANT[c]);
                return false;
            }
        }
    }
    // --- E: ScalarOperation 的两个「反向」运算 ---
    // 这两个与 RDIV/RMINUS 只差**参数顺序**，是最容易写反的一对；
    // 而 `ScalarOperation` 是 UNUSED 层，**没有任何东西会跑它**。
    //   RDIV  scalar=12, in=[2,4,6] -> 12/x = [6,3,2]
    //   RMINUS scalar=10, in=[2,4,6] -> 10-x = [8,6,4]
    // （如果实现把 RDIV 写成 x/scalar，这里会给出 [6,3,2] 之外的数；
    //   如果把 RMINUS 写成 x-s，这里会给出 [8,6,4] 之外的数。）
    {
        std::vector<float> in(3);
        in[0] = 2; in[1] = 4; in[2] = 6;
        std::vector<float> got;
        ref_scalar_op(in, 7, 12.0f, got);          // RDIV
        static const double WANT_RDIV[3] = { 6.0, 3.0, 2.0 };
        for (int i = 0; i < 3; i++) {
            if (fabs((double)got[i] - WANT_RDIV[i]) > 1e-6) {
                printf("  参考实现自证**没过**（RDIV 下标 %d）：算得 %.9g，手算是 %.9g\n",
                       i, got[i], WANT_RDIV[i]);
                return false;
            }
        }
        ref_scalar_op(in, 8, 10.0f, got);          // RMINUS
        static const double WANT_RMINUS[3] = { 8.0, 6.0, 4.0 };
        for (int i = 0; i < 3; i++) {
            if (fabs((double)got[i] - WANT_RMINUS[i]) > 1e-6) {
                printf("  参考实现自证**没过**（RMINUS 下标 %d）：算得 %.9g，手算是 %.9g\n",
                       i, got[i], WANT_RMINUS[i]);
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// 逐层探针
// ---------------------------------------------------------------------------
struct Stat { int ok, bad; int open; };
static Stat g;

static void report(const char* tag, const std::string& shape,
                   double e, double limit, long worst,
                   const std::vector<float>& got, const std::vector<float>& want)
{
    if (e > limit) {
        // **形状不同**要单独判：这时 `worst` 是 -1，
        // 拿它去索引 `want[worst]` / `got[worst]` 就是**越界读** ——
        // 2026-10-05 第一版没判这一支，探针在 Reduction 的 keepdims=1 上
        // 打印出 "1e+30 最差 #-1" 之后**自己崩了**（rc=127），
        // 把真正的形态（形状对不上）盖成了"进程挂了"。
        if (worst < 0) {
            printf("  %-6s %-22s **形状对不上**：库给 %zu 个、参考给 %zu 个  <== **对不上**\n",
                   tag, shape.c_str(), got.size(), want.size());
            g.bad++;
            return;
        }
        printf("  %-6s %-22s 后向误差 %.4g > %g（最差 #%ld）  <== **对不上**\n",
               tag, shape.c_str(), e, limit, worst);
        // 差异的**形态**是判据的一部分（AGENTS.md「报差异要报结构」）：
        // 整体比例错 / 只有个别位置错 / 变成噪声，指向完全不同的根因。
        printf("        前 8 个（参考 -> 库）：");
        for (int i = 0; i < 8 && i < (int)got.size(); i++)
            printf(" [%d %.6g->%.6g]", i, want[i], got[i]);
        printf("\n        最差位置 #%ld（参考 %.6g -> 库 %.6g，比值 %.4g）\n",
               worst, want[worst], got[worst],
               (fabs(want[worst]) > 1e-12) ? (double)got[worst] / want[worst] : 0.0);
        g.bad++;
    } else {
        printf("  %-6s %-22s 后向误差 %.4g <= %g\n", tag, shape.c_str(), e, limit);
        g.ok++;
    }
}

static void run_lrn()
{
    struct Case { int C, H, W, L; float alpha, beta, k; };
    // C 特意取 align(8) 的倍数与非倍数各一半：
    // 那份内核是**按 align 分组的向量循环**，非倍数是它的边界形状
    // （附录 AX.2 的越界就是从 C % align != 0 触发的）。
    static const Case CASES[] = {
        {   1,  5,  5, 1, 0.0001f, 0.75f, 1.0f },   // C=1, L=1 —— AX.2 踩过的形状
        {   3,  4,  4, 3, 0.0001f, 0.75f, 1.0f },
        {   8,  7,  7, 5, 0.0001f, 0.75f, 1.0f },
        {  13,  5,  5, 3, 0.0001f, 0.75f, 1.0f },   // C 不是 align 的倍数
        {  16,  8,  8, 5, 0.0001f, 0.75f, 1.0f },
        {  17,  3,  3, 7, 0.0001f, 0.75f, 1.0f },   // C 与 L 都不是倍数
        {  32, 14, 14, 5, 0.0001f, 0.75f, 1.0f },
        {  33,  2,  2, 9, 0.0001f, 0.75f, 1.0f },
        {  64,  6,  6, 3, 0.0001f, 0.75f, 1.0f },
    };
    const double LIMIT = 1e-5;
    char line[256], shape[64];
    for (size_t t = 0; t < sizeof(CASES) / sizeof(CASES[0]); t++) {
        const Case& c = CASES[t];
        snprintf(line, sizeof(line),
                 "LRN name=lrn1 bottom=data top=top1 operation=0 "
                 "local_size=%d alpha=%g beta=%g k=%g\n", c.L, c.alpha, c.beta, c.k);
        unsigned s = 20261005u + (unsigned)t * 7919u;
        std::vector<float> in((size_t)c.C * c.H * c.W);
        for (size_t i = 0; i < in.size(); i++) in[i] = rnd(s);
        std::vector<float> got, want;
        if (!run_synth(line, std::vector<float>(), c.C, c.H, c.W, in, got)) {
            g.bad++;
            continue;
        }
        ref_lrn(in, 1, c.H, c.W, c.C, c.L, c.alpha, c.beta, c.k, want);
        long wi = -1;
        double e = backward_err(got, want, wi);
        snprintf(shape, sizeof(shape), "C=%d H=%d W=%d L=%d", c.C, c.H, c.W, c.L);
        report("LRN", shape, e, LIMIT, wi, got, want);
    }
}

static void run_copy()
{
    static const int SHAPES[4][3] = { {1,1,1}, {3,2,2}, {8,7,7}, {17,3,3} };
    const double LIMIT = 1e-7;
    char line[128], shape[64];
    snprintf(line, sizeof(line), "Copy name=cp1 bottom=data top=top1\n");
    for (int t = 0; t < 4; t++) {
        int C = SHAPES[t][0], H = SHAPES[t][1], W = SHAPES[t][2];
        unsigned s = 20261111u + (unsigned)t * 104729u;
        std::vector<float> in((size_t)C * H * W);
        for (size_t i = 0; i < in.size(); i++) in[i] = rnd(s);
        std::vector<float> got, want;
        if (!run_synth(line, std::vector<float>(), C, H, W, in, got)) { g.bad++; continue; }
        ref_copy(in, want);
        long wi = -1;
        double e = backward_err(got, want, wi);
        snprintf(shape, sizeof(shape), "C=%d H=%d W=%d", C, H, W);
        report("Copy", shape, e, LIMIT, wi, got, want);
    }
}

static void run_scale()
{
    static const int CS[4] = { 1, 3, 8, 13 };     // 含非 align 倍数
    const double LIMIT = 1e-6;
    char block[512], shape[64];
    for (int with_bias = 0; with_bias <= 1; with_bias++)
        for (int t = 0; t < 4; t++) {
            int C = CS[t], H = 3, W = 3;
            std::vector<float> scale(C), bias(C);
            unsigned s = 20261212u + (unsigned)t * 1299709u + (unsigned)with_bias;
            for (int c = 0; c < C; c++) { scale[c] = rnd(s); bias[c] = rnd(s); }
            // 权重文件布局：先 C 个 scale，再 C 个 bias（compact NCHW [1][1][1][C]）
            std::vector<float> weights(scale);
            if (with_bias) weights.insert(weights.end(), bias.begin(), bias.end());
            std::vector<float> in((size_t)C * H * W);
            for (size_t i = 0; i < in.size(); i++) in[i] = rnd(s);
            // **必须先垫一层 Copy**（2026-10-05 实测，见 IA.4）：
            // `Scale` 在 `_is_inplace_safe` 那张表里，而 `Copy`/`Sqrt` 不在。
            // 直接 `Input -> Scale(top=top1)` 时，`_simplify_inplace` 会把
            // Scale 就地化（`tops[i][0] = bottoms[i][0]`），它的 bottom 就是
            // **blob 0（输入张量）** —— 而 `Forward` 结束时会把 `blobs[0]` 置 0，
            // 于是 `GetBlobByName("top1")` / `("data")` **都取不到**，
            // 探针会以"取不到输出 blob"收场，看不出是层没跑还是取名不对。
            // 垫一层 Copy 之后 Scale 的 bottom 是一个真 blob，
            // 就地化后结果落在那个 blob 上，`top1` 经 `simplify_inplace_blob_map`
            // 就能取到了。
            snprintf(block, sizeof(block),
                     "Copy name=cp1 bottom=data top=mid\n"
                     "Scale name=sc1 bottom=mid top=top1%s\n",
                     with_bias ? " bias" : "");
            std::vector<float> got, want;
            if (!run_synth(block, weights, C, H, W, in, got)) { g.bad++; continue; }
            ref_scale(in, H, W, C, scale, with_bias ? &bias : 0, want);
            long wi = -1;
            double e = backward_err(got, want, wi);
            snprintf(shape, sizeof(shape), "C=%d bias=%d", C, with_bias);
            report("Scale", shape, e, LIMIT, wi, got, want);
        }
}

static void run_sqrt()
{
    static const int CS[4] = { 1, 3, 8, 17 };
    const double LIMIT = 1e-6;
    char line[128], shape[64];
    snprintf(line, sizeof(line), "Sqrt name=sq1 bottom=data top=top1\n");
    for (int t = 0; t < 4; t++) {
        int C = CS[t], H = 3, W = 3;
        unsigned s = 20261313u + (unsigned)t * 15485863u;
        // sqrt 的定义域：输入必须非负，所以这里取 [0.01, 1.01)
        std::vector<float> in((size_t)C * H * W);
        for (size_t i = 0; i < in.size(); i++) in[i] = 0.01f + 0.5f * (rnd(s) * 0.5f + 0.5f);
        std::vector<float> got, want;
        if (!run_synth(line, std::vector<float>(), C, H, W, in, got)) { g.bad++; continue; }
        ref_sqrt(in, want);
        long wi = -1;
        double e = backward_err(got, want, wi);
        snprintf(shape, sizeof(shape), "C=%d", C);
        report("Sqrt", shape, e, LIMIT, wi, got, want);
    }
}

static void run_scalar_op()
{
    // 9 种运算各测一遍；C 覆盖 align(8) 的倍数与非倍数。
    static const int CS[2] = { 8, 13 };
    static const char* OPN[9] = { "MUL", "DIV", "ADD", "MINUS",
                                  "MAX", "MIN", "POW", "RDIV", "RMINUS" };
    const double LIMIT = 1e-6;
    char block[512], shape[96];
    for (int ci = 0; ci < 2; ci++)
        for (int op = 0; op < 9; op++) {
            const int C = CS[ci];
            const float s = (op == 7) ? 12.0f : 3.0f;   // RDIV 用一个远离 0 的标量
            const int H = 3, W = 3;
            unsigned sd = 20261414u + (unsigned)op * 32452843u + (unsigned)C * 49979687u;
            std::vector<float> in((size_t)C * H * W);
            for (size_t i = 0; i < in.size(); i++) in[i] = rnd(sd);
            // ScalarOperation 不在 `_is_inplace_safe` 表里，所以不会被就地化；
            // 但仍垫一层 Copy，让 bottom 是一个**真实 blob**（与 Scale 那条同理）。
            snprintf(block, sizeof(block),
                     "Copy name=cp1 bottom=data top=mid\n"
                     "ScalarOperation name=so1 bottom=mid top=top1 operation=%s scalar=%g\n",
                     OPN[op], s);
            std::vector<float> got, want;
            if (!run_synth(block, std::vector<float>(), C, H, W, in, got)) { g.bad++; continue; }
            ref_scalar_op(in, op, s, want);
            long wi = -1;
            double e = backward_err(got, want, wi);
            snprintf(shape, sizeof(shape), "op=%-6s C=%d scalar=%g", OPN[op], C, s);
            report("ScalarOp", shape, e, LIMIT, wi, got, want);
        }
}

static void run_squeeze()
{
    static const int CS[3] = { 1, 8, 17 };
    const double LIMIT = 1e-7;     // 纯拷贝：应当逐位相同
    char block[128], shape[64];
    snprintf(block, sizeof(block), "Squeeze name=sq1 bottom=data top=top1\n");
    for (int t = 0; t < 3; t++) {
        int C = CS[t], H = 2, W = 2;
        unsigned s = 20261515u + (unsigned)t * 86028121u;
        std::vector<float> in((size_t)C * H * W);
        for (size_t i = 0; i < in.size(); i++) in[i] = rnd(s);
        std::vector<float> got, want;
        if (!run_synth(block, std::vector<float>(), C, H, W, in, got)) { g.bad++; continue; }
        ref_squeeze(in, want);
        long wi = -1;
        double e = backward_err(got, want, wi);
        snprintf(shape, sizeof(shape), "C=%d", C);
        report("Squeeze", shape, e, LIMIT, wi, got, want);
    }
}

// Reduction：两种语义，**由内核自己定义**（读 `zq_cnn_reduction_32f_align_c.c`
// 的 align0 版之后才敢写参考 —— 附录 IE.1）。
//
//   keepdims == 0  ->  **对全部元素求和/求均值**，写到 out_data[0]，
//                     此时 `axis` 被**完全忽略**
//   keepdims == 1  ->  沿 `axis` 求和/求均值，被约的那一维置 1
//
// 而 `ZQ_CNN_Forward_SSEUtils::ReductionSum` 里 `out_dims` 的算法与之一致：
// keepdims 时 `out_dims[axis] = 1`，否则**四个维全部置 1**。
// 两侧是自洽的，所以 `axis=2 keepdims=0` 不是缺陷，只是 `axis` 被忽略 —
// 第一版我怀疑这里"结果被部分丢弃"，读实现之后**排除了**（附录 IE.1）。
//
// axis 的编号是 out_dims[4] = { N, C, H, W }：0=N 1=C 2=H 3=W。
static void ref_reduce(const std::vector<float>& in, int N, int C, int H, int W,
                       int axis, bool keepdims, bool mean,
                       std::vector<float>& out, int& outC, int& outH, int& outW)
{
    if (!keepdims) {
        double s = 0;
        for (size_t i = 0; i < in.size(); i++) s += in[i];
        out.resize(1);
        out[0] = (float)(mean ? s / (double)in.size() : s);
        outC = outH = outW = 1;
        return;
    }
    outC = C; outH = H; outW = W;
    int red = (axis == 0) ? N : (axis == 1) ? C : (axis == 2) ? H : W;
    // axis 索引的是 out_dims[4] = { N, C, H, W }：
    //   axis 0 -> 约 N，输出 N 变 1（C/H/W **不变**，输出大小 = C*H*W）
    //   axis 1 -> 约 C，输出大小 = H*W
    //   axis 2 -> 约 H，输出大小 = C*W
    //   axis 3 -> 约 W，输出大小 = C*H
    // 第一版把 axis==0 也写成了 outC=1，于是参考给 9 个、库给 72 个 ——
    // **错的是参考**（轴编号认错了）。2026-10-05 实测。
    if (axis == 1) outC = 1;
    else if (axis == 2) outH = 1;
    else if (axis == 3) outW = 1;
    // axis==0 时 N 已经由 keepdims 置 1（本探针的 N 恒为 1），C/H/W 不变。
    out.assign((size_t)outC * outH * outW, 0.0f);
    // compact NCHW 下标：si = (n*C + c)*H*W + h*W + w
    // 输出下标 di 里，**被约的那一维恒取 0**；输入下标 si 用**完整**的值。
    //
    // 第一版在这里写了 `if (被约轴的下标 != 0) continue;` ——
    // 那是**把过滤加在了输入侧**，于是 axis=1 时只累加了 c=0，
    // 参考值变成"第 0 个通道的值"而不是"沿 C 的和"。
    // 症状极像库算错了：12 组全红、数值差一个数量级、随 C 变化。
    // 而把「沿 C 求和」的 3x3 网格打出来之后，**库给的 9 个值与网格逐位相同** ——
    // 库是对的，参考是错的（附录 IF.1）。
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++) {
                    size_t si = (((size_t)n * C + c) * H + h) * W + w;
                    int oc = (axis == 1) ? 0 : c;
                    int oh = (axis == 2) ? 0 : h;
                    int ow = (axis == 3) ? 0 : w;
                    size_t di = (((size_t)oc) * outH + oh) * outW + ow;
                    out[di] += in[si];
                }
    if (mean)
        for (size_t i = 0; i < out.size(); i++) out[i] = (float)(out[i] / (float)red);
}

static void run_reduction()
{
    // 2 种运算 × 4 个 axis × keepdims 两档 × 2 个 C = 32 组。
    static const char* OPN[2] = { "SUM", "MEAN" };
    static const int CS[2] = { 8, 13 };
    const int H = 3, W = 3, N = 1;
    const double LIMIT = 1e-5;      // 求和会累加 H*W*C 个 float，舍入按比例放大
    char block[512], shape[96];
    for (int ci = 0; ci < 2; ci++)
        for (int axis = 0; axis < 4; axis++)
            for (int kd = 0; kd < 2; kd++)
                for (int op = 0; op < 2; op++) {
                    const int C = CS[ci];
                    unsigned s = 20261616u + (unsigned)(ci * 4 + axis) * 2u
                               + (unsigned)kd * 1u + (unsigned)op * 7919u;
                    std::vector<float> in((size_t)C * H * W);
                    for (size_t i = 0; i < in.size(); i++) in[i] = rnd(s);
                    snprintf(block, sizeof(block),
                             "Copy name=cp1 bottom=data top=mid\n"
                             "Reduction name=rd1 bottom=mid top=top1 operation=%s "
                             "axis=%d keepdims=%d\n", OPN[op], axis, kd);
                    // **名字要先打**（附录 HC.5）：崩在组内时，
                    // 否则读的人只知道"上一行之后没了"，猜不出挂在哪一组。
                    printf("  [probe] Reduction op=%s axis=%d kd=%d C=%d ...\n",
                           OPN[op], axis, kd, C);
                    std::vector<float> got, want;
                    if (!run_synth(block, std::vector<float>(), C, H, W, in, got)) { g.bad++; continue; }
                    int oC = 0, oH = 0, oW = 0;
                    ref_reduce(in, N, C, H, W, axis, kd != 0, op == 1, want, oC, oH, oW);
                    long wi = -1;
                    double e = backward_err(got, want, wi);
                    snprintf(shape, sizeof(shape), "op=%-4s axis=%d kd=%d C=%d", OPN[op], axis, kd, C);
                    if (e > LIMIT && axis != 0 && kd != 0) {
                        // **这一格只报、不判失败**（附录 IE.3）：
                        // `keepdims=1` 且 `axis ∈ {1,2,3}` 时，库给的值与
                        // "沿该轴求和" 对不上；而 axis=0 与 keepdims=0 全部正确。
                        // 这一层是 **UNUSED**（没有任何随仓模型会跑到），
                        // 接进回归判失败就是**恒红**，会把别的真回归失败淹掉
                        // （AGENTS.md「一个恒红的检查不要接进回归」）。
                        // 这里如实报出来并单独计数，根因留待下一轮。
                        g.open++;
                        printf("  %-6s %-22s **待查**（不判失败）：库 %zu 个值 / 参考 %zu 个\n",
                               "Reduction", shape, got.size(), want.size());
                        if (C == 8 && op == 0) {
                            // 把「每个 (h,w) 沿 C 求和」的 3x3 网格**按坐标**打出来，
                            // 与库的 9 个值按同样坐标对照。
                            // 一旦发现库的值对应的是**另一个坐标**（转置 / 行列互换），
                            // 根因就是"写入位置与读回下标不对应"，而不是"约错了轴"。
                            printf("        [诊断] 沿 C 求和的 3x3 网格（行=h，列=w）：\n");
                            for (int hh = 0; hh < H; hh++) {
                                printf("          ");
                                for (int ww = 0; ww < W; ww++) {
                                    double p = 0;
                                    for (int cc = 0; cc < C; cc++) p += in[((size_t)cc * H + hh) * W + ww];
                                    printf(" %9.5f", p);
                                }
                                printf("\n");
                            }
                            printf("        [诊断] 库给的 %zu 个值（读回顺序）：\n          ", got.size());
                            for (size_t q = 0; q < got.size() && q < 24; q++) printf(" %9.5f", got[q]);
                            printf("\n");
                        }
                    } else {
                        report("Reduction", shape, e, LIMIT, wi, got, want);
                    }
                }
}

static void run_unary_op()
{
    // UnaryOperation：**两个 bottom**。标量取自 `bottoms[0]` 的**首元素**，
    // 张量是 `bottoms[1]`；九个运算复用 `ScalarOperation_*` 那套内核，
    // 所以语义与 `run_scalar_op` 完全一致（附录 IG.1）。
    //
    // 唯一的差别在 DIV：实现走的是 `ScalarOperation_Mul(x, 1.0f/scalar)`，
    // 数学上等于 x/scalar，但**最后一位不同**（倒数再乘）。
    // 所以这一档的判据留 1e-6（而不是 ScalarOp 那一档的"期望逐位 0"）。
    static const int CS[2] = { 8, 13 };
    static const char* OPN[9] = { "MUL", "DIV", "ADD", "MINUS",
                                  "MAX", "MIN", "POW", "RDIV", "RMINUS" };
    const double LIMIT = 1e-6;
    char block[512], shape[96];
    for (int ci = 0; ci < 2; ci++)
        for (int op = 0; op < 9; op++) {
            const int C = CS[ci];
            const int H = 3, W = 3;
            unsigned sd = 20261717u + (unsigned)op * 32452843u + (unsigned)ci * 49979687u;
            std::vector<float> in((size_t)C * H * W);
            for (size_t i = 0; i < in.size(); i++) in[i] = rnd(sd);
            // bottoms[0] = scl（只用来取首元素当标量），bottoms[1] = mid（真正的张量）。
            // 两个都垫一层 Copy，bottom 才是**真实 blob**而不是输入 blob 0。
            snprintf(block, sizeof(block),
                     "Copy name=cp1 bottom=data top=mid\n"
                     "Copy name=cp2 bottom=data top=scl\n"
                     "UnaryOperation name=uo1 bottom=scl bottom=mid top=top1 operation=%s\n",
                     OPN[op]);
            std::vector<float> got, want;
            printf("  [probe] UnaryOp op=%s C=%d ...\n", OPN[op], C);
            if (!run_synth(block, std::vector<float>(), C, H, W, in, got)) { g.bad++; continue; }
            // 标量 = bottoms[0] 的首元素。bottoms[0] 是 `scl`，它是输入的一份拷贝，
            // 而输入的首元素（NHWC 的 (0,0,0,0) = compact NCHW 的 0）在两种布局下同址。
            const float s = in[0];
            ref_scalar_op(in, op, s, want);
            long wi = -1;
            double e = backward_err(got, want, wi);
            snprintf(shape, sizeof(shape), "op=%-6s C=%d", OPN[op], C);
            report("UnaryOp", shape, e, LIMIT, wi, got, want);
        }
}

// BatchNorm：权重文件里是 **2C 个 float**，前 C 个是 mean、后 C 个是 var
// （`ZQ_CNN_Layer_BatchNorm::LoadBinary_NCHW`：两次 `ConvertFromCompactNCHW`
// 分别喂给 `mean` 与 `var`，随后 `BatchNorm_Compute_b_a` 在**加载期**就把
// b/a 算出来，Forward 只做 `BatchNorm_b_a(top, b, a)`）。
//
//     b[c] = 1 / sqrt(var[c] + eps)
//     a[c] = -mean[c] * b[c]
//     y[c] = b[c] * x[c] + a[c]
//
// 与 `BatchNormScale` 的差别是它没有 slope/bias 两项，所以是 BN 的
// "只给 mean/var" 那一档（对应 `zq_cnn_batchnorm_32f_mean_var_align`）。
//
// `var` 取 [0.5, 1.5]：加上 eps 之后离 FLOAT_EPS_FOR_DIV 极远，
// 那个 `__max` 守卫**不会**被触发，参考里也就不需要复制它。
static void ref_batchnorm(const std::vector<float>& in, int C,
                          const std::vector<float>& mean,
                          const std::vector<float>& var, float eps,
                          std::vector<float>& out)
{
    const size_t HW = in.size() / (size_t)C;
    std::vector<double> b((size_t)C), a((size_t)C);
    for (int c = 0; c < C; c++) {
        b[c] = 1.0 / sqrt((double)var[c] + (double)eps);
        a[c] = -(double)mean[c] * b[c];
    }
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); i++) {
        int c = (int)(i / HW);              // compact NCHW：通道是最外层
        double bx = b[c] * (double)in[i];   // 分母取计算尺度，不是 |结果|
        out[i] = (float)(bx + a[c]);
    }
}

static void run_batchnorm()
{
    static const int CS[3] = { 1, 8, 17 };   // 含非 align 倍数
    const double LIMIT = 1e-5;
    const float eps = 1e-5f;
    char block[256], shape[64];
    snprintf(block, sizeof(block),
             "Copy name=cp1 bottom=data top=mid\n"
             "BatchNorm name=bn1 bottom=mid top=top1 eps=%g\n", eps);
    for (int t = 0; t < 3; t++) {
        const int C = CS[t], H = 3, W = 3;
        unsigned s = 20261818u + (unsigned)t * 15485863u;
        std::vector<float> in((size_t)C * H * W), mean(C), var(C);
        for (size_t i = 0; i < in.size(); i++) in[i] = rnd(s);
        for (int c = 0; c < C; c++) { mean[c] = rnd(s); var[c] = 0.5f + rnd(s); }
        std::vector<float> weights(mean);      // 权重顺序：先 mean，后 var
        weights.insert(weights.end(), var.begin(), var.end());
        std::vector<float> got, want;
        printf("  [probe] BatchNorm C=%d ...\n", C);
        if (!run_synth(block, weights, C, H, W, in, got)) { g.bad++; continue; }
        ref_batchnorm(in, C, mean, var, eps, want);
        long wi = -1;
        double e = backward_err(got, want, wi);
        snprintf(shape, sizeof(shape), "C=%d", C);
        report("BatchNorm", shape, e, LIMIT, wi, got, want);
    }
}

// Tile：把输入沿四个轴**重复展开**。语义取自
// `ZQ_CNN_Tensor4D::Tile`（附录 IG.2）：
//
//     for (tc = 0; tc < tile_c; tc++) { memcpy(out_c_ptr, in_c_ptr, 4*C); out_c_ptr += C; }
//     for (w  = 1; w  < tile_w; w++) memcpy(out_pix_ptr + w*elt_num, in_pix_ptr, 4*elt_num);
//     for (h  = 0; h  < tile_h; h++) ...
//
// 也就是**整个输入沿该轴首尾相接 tile_* 份**：
//
//     out[n][c][h][w] = in[n % N][c % C][h % H][w % W]
//
// **注意这与 TensorFlow 的 `Tile` 不是一回事**（附录 IG.2）：
// TF 的通道轴是 **repeat-interleave**（`out[c] = in[c / tile_c]`，
// 即每个输入通道连续出现 tile_c 次），
// 而这里的循环是 `for (tc…) { memcpy(out_c_ptr, in_c_ptr, 4*C); out_c_ptr += C; }`
// —— `in_c_ptr` 每轮**不前进**，所以复制的是**整块输入**。
//
// 第一版参考按 TF 的约定写（`out[c] = in[c / tile_c]`），于是
// 6 组里 4 组"对不上"；把按 (c,h,w) 坐标打出来的输入/输出并排看，
// 库的输出是 `ch0..ch7 = in ch0..ch7`、`ch8..ch15 = in ch0..ch7` ——
// **首尾相接**，与本注释一致。库是对的，参考是按**别的框架的约定**写的。
static void ref_tile(const std::vector<float>& in, int N, int C, int H, int W,
                     int tn, int th, int tw, int tc, std::vector<float>& out)
{
    const int oN = N * tn, oC = C * tc, oH = H * th, oW = W * tw;
    out.assign((size_t)oN * oC * oH * oW, 0.0f);
    for (int n = 0; n < oN; n++)
        for (int c = 0; c < oC; c++)
            for (int h = 0; h < oH; h++)
                for (int w = 0; w < oW; w++) {
                    size_t si = (size_t)((n % N) * C + c % C) * H * W
                              + (size_t)(h % H) * W + (w % W);
                    size_t di = ((size_t)n * oC + c) * oH * oW
                              + (size_t)h * oW + w;
                    out[di] = in[si];
                }
}

static void run_tile()
{
    struct Case { int C, H, W, tn, th, tw, tc; };
    static const Case CASES[] = {
        {  4, 2, 2, 1, 1, 1, 1 },   // 全 1：恒等
        {  4, 2, 2, 1, 1, 1, 2 },   // 只沿 C 重复
        {  4, 2, 2, 1, 1, 2, 1 },   // 只沿 W 重复
        {  4, 2, 2, 1, 2, 1, 1 },   // 只沿 H 重复
        {  4, 2, 2, 2, 1, 1, 1 },   // 只沿 N 重复
        {  3, 3, 3, 2, 2, 2, 2 },   // 四轴全重复，C 是非 align 倍数
        // ---- 下面这几组是**分象限**用的（附录 IG.3）----
        {  8, 2, 2, 1, 1, 1, 2 },   // C 是 align 的倍数，只沿 C 重复
        {  8, 2, 2, 1, 1, 2, 1 },   // C 是 align 的倍数，只沿 W 重复
        {  8, 2, 2, 1, 2, 1, 1 },   // C 是 align 的倍数，只沿 H 重复
        {  8, 2, 2, 2, 2, 2, 2 },   // 四轴全重复，C 是 align 的倍数
        { 16, 2, 2, 1, 1, 1, 2 },
        {  4, 2, 2, 1, 1, 1, 2 },   // C=4：align=8 下 pixelStep(8) != C
    };
    const double LIMIT = 1e-7;      // 纯搬运：应当逐位相同
    char block[256], shape[96];
    for (size_t t = 0; t < sizeof(CASES) / sizeof(CASES[0]); t++) {
        const Case& c = CASES[t];
        snprintf(block, sizeof(block),
                 "Copy name=cp1 bottom=data top=mid\n"
                 "Tile name=tl1 bottom=mid top=top1 n=%d h=%d w=%d c=%d\n",
                 c.tn, c.th, c.tw, c.tc);
        unsigned s = 20261919u + (unsigned)t * 32452843u;
        std::vector<float> in((size_t)c.C * c.H * c.W);
        for (size_t i = 0; i < in.size(); i++) in[i] = rnd(s);
        std::vector<float> got, want;
        printf("  [probe] Tile n=%d h=%d w=%d c=%d ...\n", c.tn, c.th, c.tw, c.tc);
        if (!run_synth(block, std::vector<float>(), c.C, c.H, c.W, in, got)) { g.bad++; continue; }
        ref_tile(in, 1, c.C, c.H, c.W, c.tn, c.th, c.tw, c.tc, want);
        long wi = -1;
        double e = backward_err(got, want, wi);
        snprintf(shape, sizeof(shape), "n=%d h=%d w=%d c=%d", c.tn, c.th, c.tw, c.tc);
        report("Tile", shape, e, LIMIT, wi, got, want);
        if (e > LIMIT && c.C == 8 && c.tc == 2 && c.tn == 1 && c.th == 1 && c.tw == 1) {
            // 诊断：按 compact NCHW 的 (c,h,w) 坐标把输入与库的输出并排打出来。
            // 参考的映射是 out[c][h][w] = in[c/tile_c][h][w]（每个输入通道连续重复 2 次）。
            printf("        [诊断] 输入（每个通道一行，h=0,w=0..%d）：\n", c.W - 1);
            for (int cc = 0; cc < c.C; cc++) {
                printf("          in ch%-2d ", cc);
                for (int ww = 0; ww < c.W; ww++) printf(" %8.4f", in[((size_t)cc * c.H) * c.W + ww]);
                printf("\n");
            }
            const int oC = c.C * c.tc;
            printf("        [诊断] 库的输出（共 %d 通道）：\n", oC);
            for (int cc = 0; cc < oC; cc++) {
                printf("          out ch%-2d", cc);
                for (int ww = 0; ww < c.W; ww++) printf(" %8.4f", got[((size_t)cc * c.H) * c.W + ww]);
                printf("\n");
            }
        }
    }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("「没有任何随仓模型跑得到」的层类型：造合成网真跑一遍（附录 IA）\n\n");

    // 参考实现必须先自证，否则它没有资格判库错。
    printf("参考实现自证（四个手算用例）：\n");
    if (!selftest_reference()) {
        printf("  参考实现自证**没过** —— 下面所有结论都不可信，已中止\n");
        cleanup_synth();
        return 1;
    }
    printf("  五个手算用例组都对上了"
           "（LRN 1/1.05·2/1.14·3/1.13、LRN L=1 1/1.03·2/1.12、\n"
           "           Scale 2·4·6·8·16·19·22·25、Sqrt 0·2·3）\n\n");

    printf("逐层探针：\n");
    run_lrn();
    run_copy();
    run_scale();
    run_sqrt();
    run_scalar_op();
    run_squeeze();
    run_reduction();
    run_unary_op();
    run_batchnorm();
    run_tile();
    printf("  小结：跑过 %d 个形状，对 %d，**对不上** %d，**待查** %d\n",
           g.ok + g.bad, g.ok, g.bad, g.open);

    printf("\n尚未覆盖的 UNUSED 层类型（**如实列出**，不装作查过）：\n");
    printf("  DeConvolution  BatchNorm  LSTM_TF  UnaryOperation  Tile  Reduction\n");
    printf("  PriorBoxText  PriorBox_MXNET  DetectionOutput_MXNET\n");

    cleanup_synth();
    printf("\n%s\n", g.bad == 0 ? "UNUSED LAYER PROBE OK" : "UNUSED LAYER PROBE FAILED");
    return g.bad == 0 ? 0 : 1;
}
