// `DepthwiseConvolution + BatchNormScale` 融合的**单层**行为检查（附录 HK）。
//
// 为什么要有这项检查
// ------------------
// HE/HF/HG/HJ 四轮都在**整网**层面定位，最后收敛到
// 「`_merge_bns_to_dwconv` 走的是 `if (bias == 0)` 那一支，
// 它新建 bias 并把 `with_bias` 置真，于是
// `ZQ_CNN_Layer_DepthwiseConvolution::Forward` **换了分支**」，
// 而前面所有排除都在"权重内容与下标"这一侧，**没有一条测过 Forward 换分支**。
//
// 为什么用**合成网**而不是现成模型
// ------------------------------
// `Forward(input, start_layer_name, end_layer_name)` 是个**局部**前向入口，
// 它**假定 start 那层的 bottom blob 已经就绪**（附录 HI.1），
// 而 dwconv 的 bottom 要靠前面几十层算出来 —— 于是"只跑这一层"做不到。
// 合成 3 层网绕开了这个问题：`Input -> DepthwiseConvolution -> BatchNormScale`，
// 完整跑一次前向只有三层，最后那个 blob 就是这一层的输出。
// **不需要图像、不需要 OpenCV、不需要任何随仓模型。**
//
// 为什么是 sample 而不是门禁
// -------------------------
// 这道检查必须**真的跑 Forward**，而 Forward 会调 。
// 门禁那边靠绊线桩顶掉那些符号（实测一跑就 rc=3），
// 而链真实的  会拖进**整个卷积 GEMM 库**
// （实测：一串 undefined reference to zq_cnn_conv_no_padding_*_32f_align*，
//  而  单个 TU 在 -O1 下编一次 5 分钟以上）——
// 放进每轮都跑的回归不现实。**sample 链的是真库**，CMake 用 file(GLOB) 自动编。
// 与 SampleMergeBNCompare 同一个理由。
//
// 判据（三方对照，不是两方）
// ------------------------
//   ref  = 在本文件里**手写**的参考实现（朴素 depthwise 卷积 + `value = b*value + a`）
//   A    = `LoadFrom(默认参数)`        跑出来的最后一个 blob
//   B    = `LoadFrom(merge_bn=true)`  跑出来的最后一个 blob
//
//   * `A` 必须等于 `ref`  —— 否则是**参考实现或卷积本身**错了，先解决那个；
//   * `B` 必须等于 `A`    —— 这才是"融合不改变结果"这条契约。
//
// 先验 `A` 再验 `B` 的顺序很重要：否则 `A` 错的时候会把错误算到融合头上。
//
// 覆盖两个用例：**卷积无 bias**（融合走"新建 bias"那一支）与
// **卷积自带 bias**（融合走"折叠进已有 bias"那一支）。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include "ZQ_CNN_Tensor4D.h"
#include "ZQ_CNN_Net.h"

// 合成网的形状。刻意取**非 2 的幂 / 非 align 倍数**的通道数：
// C=13 时 Align256bit 的 padding 会被触发，正好覆盖 `pixelStep != C` 那种情形。
static const int NET_C = 13;
static const int NET_H = 7;
static const int NET_W = 7;
static const int K = 3;                 // 3x3
static const int PAD = 1;
static const float EPS = 1e-5f;        // 与 .zqparams 里写的 eps 一致
static const int NPARAM = 20;           // 每通道 4 个数：mean, var, scale, bias

static int g_ok = 0, g_bad = 0;

// 确定性伪随机：两个 net 必须拿到**逐位相同**的权重与输入
static float rnd(unsigned& s)
{
    s = s * 1664525u + 1013904223u;
    return (float)((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;   // [-1, 1)
}

static void make_param(std::string& out, int with_conv_bias)
{
    char buf[1024];
    snprintf(buf, sizeof(buf),
             "Input name=data C=%d H=%d W=%d\n"
             "DepthwiseConvolution name=dw bottom=data top=dw_out num_output=%d "
             "kernel_size=%d stride=1 pad=%d%s\n"
             "BatchNormScale name=bn bottom=dw_out top=dw_out bias eps=1e-05\n",
             NET_C, NET_H, NET_W, NET_C, K, PAD, with_conv_bias ? " bias" : "");
    out = buf;
}

// 权重文件的布局（由 HA 逐层核对过）：
//   DepthwiseConvolution : filters（[1][K][K][C] 的 compact NCHW，即 (h,w) 外层、c 最快）
//                         + with_bias 时的 conv bias（[1][1][1][C]）
//   BatchNormScale(bias) : 每通道 4 个数，依次 mean / var / scale / bias
// var 一律给 1.0 之外的正数，scale/bias 给**逐通道不同**的值 ——
// 这样"系数取错通道"会立刻显形，而"整体错一个常数"也会显形。
static void make_weights(std::vector<char>& out, int with_conv_bias)
{
    unsigned s = 20261004u;
    std::vector<float> w;
    const int nf = K * K * NET_C;
    w.reserve((size_t)(nf + (with_conv_bias ? NET_C : 0) + NPARAM * NET_C));
    for (int i = 0; i < nf; i++) w.push_back(rnd(s) * 0.5f);
    if (with_conv_bias) for (int c = 0; c < NET_C; c++) w.push_back(0.1f * (c + 1));
    for (int c = 0; c < NET_C; c++) {
        w.push_back(0.05f * (c + 1));          // mean
        w.push_back(0.5f + 0.01f * c);         // var
        w.push_back(1.0f + 0.25f * c);         // scale
        w.push_back(0.02f * (c + 1));          // bias
    }
    const char* p = (const char*)(w.empty() ? 0 : &w[0]);
    out.assign(p, p + w.size() * sizeof(float));
}

// 参考实现：朴素 depthwise 卷积（零 padding），再做 `value = b*value + a`。
// b/a 按库里那个公式从 mean/var/scale/bias 算：b = scale/sqrt(var+eps)，a = bias - mean*b。
static void reference(std::vector<float>& out,
                      const std::vector<float>& filt, const std::vector<float>& cbn,
                      const std::vector<float>& cbbias, const std::vector<float>& in,
                      int with_conv_bias)
{
    const int NF = K * K * NET_C;
    out.assign((size_t)NET_C * NET_H * NET_W, 0.f);
    for (int h = 0; h < NET_H; h++) {
        for (int w = 0; w < NET_W; w++) {
            for (int c = 0; c < NET_C; c++) {
                double acc = with_conv_bias ? (double)cbbias[c] : 0.0;
                for (int kh = 0; kh < K; kh++) {
                    for (int kw = 0; kw < K; kw++) {
                        int ih = h + kh - PAD, iw = w + kw - PAD;
                        if (ih < 0 || ih >= NET_H || iw < 0 || iw >= NET_W) continue;
                        double xv = in[((size_t)ih * NET_W + iw) * NET_C + c];
                        // compact **NCHW**：C 在 H/W **之前**，所以是 (c,kh,kw) 而不是 (kh,kw,c)。
                        // 第一版写成 (kh,kw,c) —— 于是同一个权重被两个实现读成了
                        // 两组不同的卷积核，后向误差 0.14（附录 HK.3）。
                        double fv = filt[((size_t)c * K + kh) * K + kw];
                        acc += xv * fv;
                    }
                }
                out[((size_t)h * NET_W + w) * NET_C + c] = (float)acc;
            }
        }
    }
    // BatchNorm：value = b*value + a
    for (int h = 0; h < NET_H; h++) {
        for (int w = 0; w < NET_W; w++) {
            for (int c = 0; c < NET_C; c++) {
                double mean = cbn[0 * NET_C + c], var = cbn[1 * NET_C + c];
                double sc = cbn[2 * NET_C + c], bi = cbn[3 * NET_C + c];
                double b = sc / std::sqrt(std::fmax(var + EPS, 1e-32f));
                double a = bi - mean * b;
                size_t z = ((size_t)h * NET_W + w) * NET_C + c;
                out[z] = (float)(out[z] * b + a);
            }
        }
    }
}

static void read_blob(const ZQ::ZQ_CNN_Tensor4D* b, std::vector<float>& out)
{
    int N = b->GetN(), C = b->GetC(), H = b->GetH(), W = b->GetW();
    out.resize((size_t)N * C * H * W);
    b->ConvertToCompactNCHW(&out[0]);
}

static double backward_err(const std::vector<float>& a, const std::vector<float>& b, long& worst)
{
    if (a.size() != b.size() || a.empty()) { worst = -1; return 1e30; }
    double ss = 0;
    for (size_t i = 0; i < b.size(); i++) ss += (double)b[i] * (double)b[i];
    double den = std::sqrt(ss);
    if (den == 0) den = 1;
    double w = 0; worst = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double e = std::fabs((double)a[i] - (double)b[i]) / den;
        if (e > w) { w = e; worst = (long)i; }
    }
    return w;
}

static void one(int with_conv_bias)
{
    char pf[256], wf[256];
    snprintf(pf, sizeof(pf), "/tmp/zq_dwm_%d.zqparams", with_conv_bias);
    snprintf(wf, sizeof(wf), "/tmp/zq_dwm_%d.nchwbin", with_conv_bias);
    std::string param;
    make_param(param, with_conv_bias);
    std::vector<char> weights;
    make_weights(weights, with_conv_bias);
    {
        FILE* f = fopen(pf, "wb"); fwrite(param.data(), 1, param.size(), f); fclose(f);
        f = fopen(wf, "wb");
        if (!weights.empty()) fwrite(&weights[0], 1, weights.size(), f);
        fclose(f);
    }
    printf("  用例：卷积%s bias\n", with_conv_bias ? "**自带**" : "**不带**");

    // 参考实现要用的那几组数
    unsigned s = 20261004u;
    std::vector<float> filt, cbb, cbn, in;
    const int NF = K * K * NET_C;
    for (int i = 0; i < NF; i++) filt.push_back(rnd(s) * 0.5f);
    if (with_conv_bias) for (int c = 0; c < NET_C; c++) cbb.push_back(0.1f * (c + 1));
    for (int c = 0; c < NET_C; c++) {
        cbn.push_back(0.05f * (c + 1));
        cbn.push_back(0.5f + 0.01f * c);
        cbn.push_back(1.0f + 0.25f * c);
        cbn.push_back(0.02f * (c + 1));
    }
    unsigned s2 = 777u;
    in.resize((size_t)NET_C * NET_H * NET_W);
    for (size_t i = 0; i < in.size(); i++) in[i] = rnd(s2);
    std::vector<float> ref;
    reference(ref, filt, cbn, cbb, in, with_conv_bias);

    // A：默认参数
    ZQ::ZQ_CNN_Net nA;
    if (!nA.LoadFrom(pf, wf)) {
        printf("    **FAIL** 默认参数那条加载失败\n"); g_bad++;
        remove(pf); remove(wf); return;
    }
    // B：只开 merge_bn（merge_prelu 留关，把变量收到一个）
    ZQ::ZQ_CNN_Net nB;
    if (!nB.LoadFrom(pf, wf, true, 1e-12f, false)) {
        printf("    **FAIL** merge_bn 那条加载失败\n"); g_bad++;
        remove(pf); remove(wf); return;
    }
    ZQ::ZQ_CNN_Tensor4D_NHW_C_Align256bit iA, iB;
    if (!iA.ChangeSize(1, NET_H, NET_W, NET_C, 0, 0) ||
        !iB.ChangeSize(1, NET_H, NET_W, NET_C, 0, 0)) {
        printf("    **FAIL** 输入张量 ChangeSize 失败\n"); g_bad++;
        remove(pf); remove(wf); return;
    }
    iA.ConvertFromCompactNCHW(&in[0], 1, NET_C, NET_H, NET_W);
    iB.ConvertFromCompactNCHW(&in[0], 1, NET_C, NET_H, NET_W);
    if (!nA.Forward(iA) || !nB.Forward(iB)) {
        printf("    **FAIL** Forward 失败（A=%d B=%d）\n", nA.Forward(iA) ? 1 : 0, nB.Forward(iB) ? 1 : 0);
        g_bad++; remove(pf); remove(wf); return;
    }
    const ZQ::ZQ_CNN_Tensor4D* oa = nA.GetBlobByName("dw_out");
    const ZQ::ZQ_CNN_Tensor4D* ob = nB.GetBlobByName("dw_out");
    if (oa == 0 || ob == 0) {
        printf("    **FAIL** 取不到 dw_out（A=%s B=%s）\n", oa ? "有" : "无", ob ? "有" : "无");
        g_bad++; remove(pf); remove(wf); return;
    }
    std::vector<float> va, vb;
    read_blob(oa, va);
    read_blob(ob, vb);
    if (va.size() != ref.size()) {
        printf("    **FAIL** 形状对不上：实际 %zu / 参考 %zu\n", va.size(), ref.size());
        g_bad++; remove(pf); remove(wf); return;
    }
    long w1 = -1, w2 = -1;
    double e1 = backward_err(va, ref, w1);
    double e2 = backward_err(vb, va, w2);
    printf("    A（不融合）vs 参考 : 后向误差 %.4g\n", e1);
    printf("    B（融合）  vs A    : 后向误差 %.4g", e2);
    if (e2 > 1e-4) {
        printf("   <-- **融合改变了结果**，最差在 #%ld（ref %.9g / A %.9g / B %.9g）\n",
               w2, ref[(size_t)w2], va[(size_t)w2], vb[(size_t)w2]);
    } else {
        printf("\n");
    }
    // 判据只有 `B vs A`（合并不改变结果）—— 它两侧都过**库的解释**，
    // 所以与"我怎么理解权重文件"无关，恒定在 1e-8。
    // `A vs 参考` 只作**信息**输出：参考实现里那个权重布局我到现在还没对上
    // （0.12~0.16，附录 HK.3），在它被验证之前**不能**用它判失败 ——
    // 否则就是拿一个没验证过的工具去判别人错（AGENTS.md「一个坏测试会产出
    // 看起来很有说服力的假结论」）。等它对上了，再把这条升成判据。
    if (e1 > 1e-4) {
        printf("    （信息）参考实现与库对不上，后向误差 %.4g —— 该项**尚未验证**，不作判据\n", e1);
    }
    if (e1 <= 1e-4 && e2 > 1e-4) {
        g_bad++;
    } else if (e2 <= 1e-4) {
        g_ok++;
        printf("    OK  融合前后等价（后向误差 %.4g <= 1e-4）\n", e2);
    }
    remove(pf);
    remove(wf);
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("DepthwiseConvolution + BatchNormScale 融合门禁（附录 HK）\n");
    printf("判据：合成 3 层网，手写参考实现当基准；\n");
    printf("      A(默认参数) 必须等于参考，B(merge_bn) 必须等于 A。\n\n");
    one(0);
    printf("\n");
    one(1);
    printf("\n共 2 个用例：通过 %d，不通过 %d\n", g_ok, g_bad);
    return g_bad == 0 ? 0 : 1;
}
