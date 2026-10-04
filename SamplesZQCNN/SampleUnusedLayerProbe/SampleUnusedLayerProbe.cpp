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
    return true;
}

// ---------------------------------------------------------------------------
// 逐层探针
// ---------------------------------------------------------------------------
struct Stat { int ok, bad; };
static Stat g;

static void report(const char* tag, const std::string& shape,
                   double e, double limit, long worst,
                   const std::vector<float>& got, const std::vector<float>& want)
{
    if (e > limit) {
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
    printf("  四个手算用例都对上了"
           "（LRN 1/1.05·2/1.14·3/1.13、LRN L=1 1/1.03·2/1.12、\n"
           "           Scale 2·4·6·8·16·19·22·25、Sqrt 0·2·3）\n\n");

    printf("逐层探针：\n");
    run_lrn();
    run_copy();
    run_scale();
    run_sqrt();
    printf("  小结：跑过 %d 个形状，对 %d，**对不上** %d\n", g.ok + g.bad, g.ok, g.bad);

    printf("\n尚未覆盖的 UNUSED 层类型（**如实列出**，不装作查过）：\n");
    printf("  DeConvolution  BatchNorm  LSTM_TF  ScalarOperation  UnaryOperation\n");
    printf("  Tile  Reduction  Squeeze  PriorBoxText  PriorBox_MXNET  DetectionOutput_MXNET\n");

    cleanup_synth();
    printf("\n%s\n", g.bad == 0 ? "UNUSED LAYER PROBE OK" : "UNUSED LAYER PROBE FAILED");
    return g.bad == 0 ? 0 : 1;
}
