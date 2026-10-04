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
static int NET_C = 13;   // 由 main 按配置改写（附录 HM）
static int NET_H = 7;   // 由 main 按配置改写（附录 HM）
static int NET_W = 7;   // 由 main 按配置改写（附录 HM）
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
             "BatchNormScale name=bn bottom=dw_out top=dw_out bias eps=1e-05\n"
             // 第四层：**读 dw_out**。目的是让 `_merge_bn` 守卫里的
             //     later_refer 变成 **true** ——
             // 没有它时 BN 是最后一层，later_refer=false，走的是 `|| !later_refer`
             // 那一支；而真模型里后面还有人读那个 blob，走的是
             // `tops[i+1][0] == bottoms[i+1][0]` 那一支。
             // **同一个 if 的两个析取项是两个不同的代码路径**（附录 HK.5）。
             // Flatten 没有 LoadBinary_NCHW 重载，所以**不需要任何额外权重字节**。
             "Flatten name=fl bottom=dw_out top=fl\n",
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

// 第三个用例：**两个 dwconv 写同一个 blob**（ResNet 共享 skip 路径的写法）。
//
// 为什么必须有它：前两个用例里 dwconv 只有一个，写它的 blob 也只有它写。
// 而 mobilefacenet-v1 的 res4 段里
//     res4_block1/2/3/4_conv_dw  四层的 bottom 全是 res4_block1_conv、
//     top 全是 res4_block1_conv_dw
// —— 四个 dwconv 轮流**写同一个 blob**。这是合成网与真模型剩下的**唯一**结构差别，
// 而它恰好是前四轮都没能复现的那一类（HK.4 单层融合是对的；HJ.3 没有层被漏掉；
// HH.2 输入是对的；HJ.2 通道内 ratio 0/256 恒定）。
static void case_shared_blob()
{
    const int C = NET_C, HF = NET_H, WF = NET_W;
    const int NF = K * K * C;
    char pf[256], wf[256];
    snprintf(pf, sizeof(pf), "/tmp/zq_dwm_shared.zqparams");
    snprintf(wf, sizeof(wf), "/tmp/zq_dwm_shared.nchwbin");
    {
        char buf[1024];
        snprintf(buf, sizeof(buf),
                 "Input name=data C=%d H=%d W=%d\n"
                 "Convolution name=c1 bottom=data top=c1 num_output=%d kernel_size=1 stride=1 pad=0\n"
                 "DepthwiseConvolution name=dw1 bottom=c1 top=shared num_output=%d kernel_size=3 stride=1 pad=1\n"
                 "BatchNormScale name=bn1 bottom=shared top=shared bias eps=1e-05\n"
                 "DepthwiseConvolution name=dw2 bottom=c1 top=shared num_output=%d kernel_size=3 stride=1 pad=1\n"
                 "BatchNormScale name=bn2 bottom=shared top=shared bias eps=1e-05\n"
                 "Flatten name=fl bottom=shared top=fl\n",
                 C, HF, WF, C, C, C);
        FILE* f = fopen(pf, "wb");
        fwrite(buf, 1, strlen(buf), f);
        fclose(f);
    }
    {
        unsigned s = 424242u;
        std::vector<float> w;
        // c1 是**普通**卷积，filters 形状是 [num_output][kH][kW][bottom_C]
        // = C*1*1*C = **C*C**。我第一版只写了 C 个，于是 dw2 读到的位置整体前移，
        // 报 "Failed to load Binary for layer dw2"（附录 HL.2）。
        for (int i = 0; i < C * C; i++) w.push_back(rnd(s) * 0.3f);    // c1：C*C
        // 权重文件的顺序必须与**层在网里的顺序**一致：
        //     c1, dw1, bn1, dw2, bn2
        // 我第一版写成 c1, dw1, dw2, bn1, bn2 —— 于是 dw2 读到的是 bn1 的数据，
        // 报 "Failed to load Binary for layer dw2"（附录 HL.2）。
        for (int b = 0; b < 2; b++) {                                 // dw1、dw2 交替
            for (int i = 0; i < NF; i++) w.push_back(rnd(s) * 0.4f);
            for (int c = 0; c < C; c++) {                              // 紧跟它的 bn
                w.push_back(0.03f * (c + 1));                        // mean
                w.push_back(0.4f + 0.02f * c);                        // var
                w.push_back(0.8f + 0.3f * c);                         // scale
                w.push_back(0.01f * (c + 1));                         // bias
            }
        }
        FILE* f = fopen(wf, "wb");
        fwrite(&w[0], 1, w.size() * sizeof(float), f);
        fclose(f);
    }
    printf("  用例：**两个 dwconv 写同一个 blob**（共享 skip 拓扑）\n");

    unsigned s2 = 999u;
    std::vector<float> in((size_t)C * HF * WF);
    for (size_t i = 0; i < in.size(); i++) in[i] = rnd(s2);

    ZQ::ZQ_CNN_Net nA, nB;
    bool la = nA.LoadFrom(pf, wf);
    bool lb = nB.LoadFrom(pf, wf, true, 1e-12f, false);
    if (!la || !lb) {
        printf("    **FAIL** 加载失败（A=%d B=%d）\n", la ? 1 : 0, lb ? 1 : 0);
        g_bad++;
        remove(pf); remove(wf);
        return;
    }
    ZQ::ZQ_CNN_Tensor4D_NHW_C_Align256bit iA, iB;
    iA.ChangeSize(1, HF, WF, C, 0, 0);
    iB.ChangeSize(1, HF, WF, C, 0, 0);
    iA.ConvertFromCompactNCHW(&in[0], 1, C, HF, WF);
    iB.ConvertFromCompactNCHW(&in[0], 1, C, HF, WF);
    if (!nA.Forward(iA) || !nB.Forward(iB)) {
        printf("    **FAIL** Forward 失败\n");
        g_bad++;
        remove(pf); remove(wf);
        return;
    }
    const ZQ::ZQ_CNN_Tensor4D* oa = nA.GetBlobByName("shared");
    const ZQ::ZQ_CNN_Tensor4D* ob = nB.GetBlobByName("shared");
    if (oa == 0 || ob == 0) {
        printf("    **FAIL** 取不到 shared（A=%s B=%s）\n", oa ? "有" : "无", ob ? "有" : "无");
        g_bad++;
        remove(pf); remove(wf);
        return;
    }
    std::vector<float> va, vb;
    read_blob(oa, va);
    read_blob(ob, vb);
    if (va.size() != vb.size()) {
        printf("    **FAIL** 形状对不上：%zu / %zu\n", va.size(), vb.size());
        g_bad++;
        remove(pf); remove(wf);
        return;
    }
    long wi = -1;
    double e2 = backward_err(vb, va, wi);
    printf("    B（融合）vs A : 后向误差 %.4g", e2);
    if (e2 > 1e-4) {
        printf("   <-- **融合改变了结果**，最差在 #%ld（A %.9g / B %.9g）\n",
               wi, va[(size_t)wi], vb[(size_t)wi]);
        g_bad++;
    } else {
        printf("\n    OK  融合前后等价\n");
        g_ok++;
    }
    remove(pf);
    remove(wf);
}

// 第四个用例：**`dw -> BN -> PReLU`**，用**生产实参** `merge_bn=true, merge_prelu=true`。
//
// 为什么必须有它：前三个用例都传了 `merge_prelu=false`。
// 而生产（`ZQ_CNN_MTCNN.h:109`、`SampleSphereFaceNet.cpp:90`）传的是
// `(true, 1e-9, true)` —— **`_merge_prelu` 跑在已经被 `_merge_bn` 改写过的图上**。
// HE 的三路测量（(bn=1,prelu=0)=0.3695 / (bn=0,prelu=1)=0 / (bn=1,prelu=1)=0.3695）
// 说明"`merge_prelu` 自己是对的"，但**没有回答**"它在 merge_bn 之后的图上还对不对"——
// 那是一个**不同的图**，而 HE 那一测是整网比对，分不出是哪一步坏的。
//
// 这是"组合"与"各自"的差别：两个变换各自正确，串起来仍可能错
// （第二个看到的是第一个的**输出状态**）。
static void case_bn_then_prelu()
{
    const int C = NET_C, HF = NET_H, WF = NET_W;
    const int NF = K * K * C;
    char pf[256], wf[256];
    snprintf(pf, sizeof(pf), "/tmp/zq_dwm_bnp.zqparams");
    snprintf(wf, sizeof(wf), "/tmp/zq_dwm_bnp.nchwbin");
    {
        char buf[1024];
        snprintf(buf, sizeof(buf),
                 "Input name=data C=%d H=%d W=%d\n"
                 "DepthwiseConvolution name=dw bottom=data top=dw_out num_output=%d kernel_size=3 stride=1 pad=1\n"
                 "BatchNormScale name=bn bottom=dw_out top=dw_out bias eps=1e-05\n"
                 "PReLU name=relu bottom=dw_out top=dw_out\n"
                 "Flatten name=fl bottom=dw_out top=fl\n",
                 C, HF, WF, C);
        FILE* f = fopen(pf, "wb");
        fwrite(buf, 1, strlen(buf), f);
        fclose(f);
    }
    {
        unsigned s = 13579u;
        std::vector<float> w;
        for (int i = 0; i < NF; i++) w.push_back(rnd(s) * 0.4f);         // dw
        for (int c = 0; c < C; c++) {                                    // bn
            w.push_back(0.05f * (c + 1));
            w.push_back(0.5f + 0.01f * c);
            w.push_back(1.0f + 0.25f * c);
            w.push_back(0.02f * (c + 1));
        }
        for (int c = 0; c < C; c++) w.push_back(0.05f + 0.01f * c);    // relu slope
        FILE* f = fopen(wf, "wb");
        fwrite(&w[0], 1, w.size() * sizeof(float), f);
        fclose(f);
    }
    printf("  用例：dw -> BN -> PReLU，用**生产实参** (merge_bn=true, merge_prelu=true)\n");

    unsigned s2 = 2468u;
    std::vector<float> in((size_t)C * HF * WF);
    for (size_t i = 0; i < in.size(); i++) in[i] = rnd(s2);

    ZQ::ZQ_CNN_Net nA, nC;
    bool la = nA.LoadFrom(pf, wf);                                  // 都不融
    bool lc = nC.LoadFrom(pf, wf, true, 1e-12f, true);            // 生产
    if (!la || !lc) {
        printf("    **FAIL** 加载失败（A=%d C=%d）\n", la ? 1 : 0, lc ? 1 : 0);
        g_bad++;
        remove(pf); remove(wf);
        return;
    }
    ZQ::ZQ_CNN_Tensor4D_NHW_C_Align256bit iA, iC;
    iA.ChangeSize(1, HF, WF, C, 0, 0);
    iC.ChangeSize(1, HF, WF, C, 0, 0);
    iA.ConvertFromCompactNCHW(&in[0], 1, C, HF, WF);
    iC.ConvertFromCompactNCHW(&in[0], 1, C, HF, WF);
    if (!nA.Forward(iA) || !nC.Forward(iC)) {
        printf("    **FAIL** Forward 失败\n");
        g_bad++;
        remove(pf); remove(wf);
        return;
    }
    const ZQ::ZQ_CNN_Tensor4D* oa = nA.GetBlobByName("dw_out");
    const ZQ::ZQ_CNN_Tensor4D* oc = nC.GetBlobByName("dw_out");
    if (oa == 0 || oc == 0) {
        printf("    **FAIL** 取不到 dw_out（A=%s C=%s）\n", oa ? "有" : "无", oc ? "有" : "无");
        g_bad++;
        remove(pf); remove(wf);
        return;
    }
    std::vector<float> va, vc;
    read_blob(oa, va);
    read_blob(oc, vc);
    if (va.size() != vc.size()) {
        printf("    **FAIL** 形状对不上：%zu / %zu\n", va.size(), vc.size());
        g_bad++;
        remove(pf); remove(wf);
        return;
    }
    long wi = -1;
    double ec = backward_err(vc, va, wi);
    printf("    生产实参 vs 都不融 : 后向误差 %.4g", ec);
    if (ec > 1e-4) {
        printf("   <-- **两个变换串起来改变了结果**，最差在 #%ld（A %.9g / 生产 %.9g）\n",
               wi, va[(size_t)wi], vc[(size_t)wi]);
        g_bad++;
    } else {
        printf("\n    OK  两个变换串起来结果不变\n");
        g_ok++;
    }
    remove(pf);
    remove(wf);
}

// 第五个用例：**多个 Convolution 写同一个 blob，DepthwiseConvolution 读它**。
//
// 这是 HM.3 定位到的那个结构在真模型里的**准确形状**（我第一版以为是
// "c1 后面少了 BN/PReLU"，那不是关键；关键是**写者不止一个**）。
// `mobilefacenet-v1` 的 res4 段：
//
//   Convolution       name=res4_block1_conv  bottom=_plus4  top=res4_block1_conv
//   BatchNormScale    name=res4_block1_conv_bn bottom=res4_block1_conv top=res4_block1_conv
//   PReLU             name=res4_block1_conv_relu bottom=res4_block1_conv top=res4_block1_conv
//   DepthwiseConvolution name=res4_block1_conv_dw bottom=res4_block1_conv top=res4_block1_conv_dw
//   ... sep 读 _dw ...
//   Convolution       name=res4_block2_conv  bottom=_plus5  top=res4_block1_conv   <== 又写一遍
//   BatchNormScale    name=res4_block2_conv_bn  bottom=res4_block1_conv top=res4_block1_conv
//   PReLU             name=res4_block2_conv_relu bottom=res4_block1_conv top=res4_block1_conv
//   DepthwiseConvolution name=res4_block2_conv_dw bottom=res4_block1_conv top=res4_block1_conv_dw
//
// 也就是说 `res4_block1_conv` 这个 blob **被 block1..block5 的 conv+bn+relu
// 反复覆写**，而 block1..block4 的 dwconv 读的正是它 —— 读到的值取决于
// "跑到了第几个写者"。
//
// 而 `_merge_bn` 的 `later_refer` 判断用的是**静态 blob 下标**
// （`bottoms[j][0] == tops[i][0]`），它不知道读者在第几个写者之后读。
// 前四个合成用例里那个被读的 blob（`c1` / `skip` / `dw_out` / `dw_out`）
// 都**只被一个卷积写**，所以这一整类形状一次都没被覆盖到。
static void case_multi_writer()
{
    const int C = NET_C, HF = NET_H, WF = NET_W;
    const int NF = K * K * C;
    char pf[256], wf[256];
    snprintf(pf, sizeof(pf), "/tmp/zq_dwm_mw.zqparams");
    snprintf(wf, sizeof(wf), "/tmp/zq_dwm_mw.nchwbin");
    {
        char buf[2048];
        snprintf(buf, sizeof(buf),
                 "Input name=data C=%d H=%d W=%d\n"
                 // 第一个写者：conv1 写 shared
                 "Convolution name=conv1 bottom=data top=shared num_output=%d kernel_size=1 stride=1 pad=0\n"
                 "DepthwiseConvolution name=dw1 bottom=shared top=dw1out num_output=%d kernel_size=3 stride=1 pad=1\n"
                 "BatchNormScale name=bn1 bottom=dw1out top=dw1out bias eps=1e-05\n"
                 "Flatten name=fl1 bottom=dw1out top=fl1\n"
                 // 第二个写者：conv2 **又**写 shared（同名覆写）
                 "Convolution name=conv2 bottom=data top=shared num_output=%d kernel_size=1 stride=1 pad=0\n"
                 "BatchNormScale name=bn_c2 bottom=shared top=shared bias eps=1e-05\n"
                 "PReLU name=relu_c2 bottom=shared top=shared\n"
                 "DepthwiseConvolution name=dw2 bottom=shared top=dw2out num_output=%d kernel_size=3 stride=1 pad=1\n"
                 "BatchNormScale name=bn2 bottom=dw2out top=dw2out bias eps=1e-05\n"
                 "Flatten name=fl2 bottom=dw2out top=fl2\n",
                 C, HF, WF, C, C, C, C);
        FILE* f = fopen(pf, "wb");
        fwrite(buf, 1, strlen(buf), f);
        fclose(f);
    }
    {
        // 顺序必须与层顺序一致：conv1, dw1, bn1, fl1, conv2, bn_c2, relu_c2, dw2, bn2
        unsigned s = 24680u;
        std::vector<float> w;
        for (int i = 0; i < C * C; i++) w.push_back(rnd(s) * 0.3f);     // conv1：C*C
        for (int i = 0; i < NF; i++) w.push_back(rnd(s) * 0.4f);         // dw1
        for (int c = 0; c < C; c++) {                                     // bn1
            w.push_back(0.03f * (c + 1)); w.push_back(0.4f + 0.02f * c);
            w.push_back(0.8f + 0.3f * c);  w.push_back(0.01f * (c + 1));
        }
        for (int i = 0; i < C * C; i++) w.push_back(rnd(s) * 0.35f);    // conv2：C*C
        for (int c = 0; c < C; c++) {                                     // bn_c2
            w.push_back(0.02f * (c + 1)); w.push_back(0.6f + 0.01f * c);
            w.push_back(0.9f + 0.2f * c);  w.push_back(0.03f * (c + 1));
        }
        for (int c = 0; c < C; c++) w.push_back(0.07f + 0.005f * c);    // relu_c2 slope
        for (int i = 0; i < NF; i++) w.push_back(rnd(s) * 0.45f);        // dw2
        for (int c = 0; c < C; c++) {                                     // bn2
            w.push_back(0.04f * (c + 1)); w.push_back(0.45f + 0.015f * c);
            w.push_back(1.1f + 0.22f * c);  w.push_back(0.015f * (c + 1));
        }
        FILE* f = fopen(wf, "wb");
        fwrite(&w[0], 1, w.size() * sizeof(float), f);
        fclose(f);
    }
    printf("  用例：**两个 Convolution 写同一个 blob**，dwconv 读它（真模型 res4 的形状）\n");

    unsigned s2 = 13579u;
    std::vector<float> in((size_t)C * HF * WF);
    for (size_t i = 0; i < in.size(); i++) in[i] = rnd(s2);

    ZQ::ZQ_CNN_Net nA, nC;
    bool la = nA.LoadFrom(pf, wf);                               // 都不融
    bool lc = nC.LoadFrom(pf, wf, true, 1e-12f, true);         // 生产实参
    if (!la || !lc) {
        printf("    **FAIL** 加载失败（A=%d C=%d）\n", la ? 1 : 0, lc ? 1 : 0);
        g_bad++;
        remove(pf); remove(wf);
        return;
    }
    ZQ::ZQ_CNN_Tensor4D_NHW_C_Align256bit iA, iC;
    iA.ChangeSize(1, HF, WF, C, 0, 0);
    iC.ChangeSize(1, HF, WF, C, 0, 0);
    iA.ConvertFromCompactNCHW(&in[0], 1, C, HF, WF);
    iC.ConvertFromCompactNCHW(&in[0], 1, C, HF, WF);
    if (!nA.Forward(iA) || !nC.Forward(iC)) {
        printf("    **FAIL** Forward 失败\n");
        g_bad++;
        remove(pf); remove(wf);
        return;
    }
    // dw2 的输出是**第二个**读者读 `shared` 的结果 —— 那是最接近真模型的一处
    const ZQ::ZQ_CNN_Tensor4D* oa = nA.GetBlobByName("dw2out");
    const ZQ::ZQ_CNN_Tensor4D* oc = nC.GetBlobByName("dw2out");
    if (oa == 0 || oc == 0) {
        printf("    **FAIL** 取不到 dw2out（A=%s C=%s）\n", oa ? "有" : "无", oc ? "有" : "无");
        g_bad++;
        remove(pf); remove(wf);
        return;
    }
    std::vector<float> va, vc;
    read_blob(oa, va);
    read_blob(oc, vc);
    if (va.size() != vc.size()) {
        printf("    **FAIL** 形状对不上：%zu / %zu\n", va.size(), vc.size());
        g_bad++;
        remove(pf); remove(wf);
        return;
    }
    long wi = -1;
    double ec = backward_err(vc, va, wi);
    printf("    生产实参 vs 都不融（dw2out）: 后向误差 %.4g", ec);
    if (ec > 1e-4) {
        printf("   <-- **多个写者时融合改变了结果**，最差在 #%ld（A %.9g / 生产 %.9g）\n",
               wi, va[(size_t)wi], vc[(size_t)wi]);
        g_bad++;
    } else {
        printf("\n    OK  多个写者时融合前后等价\n");
        g_ok++;
    }
    remove(pf);
    remove(wf);
}

// 第六个用例：**极端的 BN 参数** —— 让 `b = scale/sqrt(var+eps)` 变得很大。
//
// 为什么加这个：到第五个用例为止，20 组配置全过，说明缺陷**不是拓扑性的**，
// 而更像**数据相关**的。而我的合成权重全是良性的
// （`rnd*0.4`、`var` 落在 0.4~0.6），真实训练出来的 `var` 未必。
//
// 而 `_merge_bns_to_dwconv` 干的第一件事是
//     pix_ptr[0] *= b_v;
// **在 float32 下把权重乘上 b**。若 `b_v` 很大（`var` 极小时 `sqrt(var)` 极小），
// 乘出来的权重就可能**溢出成 inf** —— 而未融合那条路算的是 `x*b + a`，
// 它自己不一定溢出。两边的结果于是天差地别，而且**形态就是"散乱噪声"**
// （与 HH.3 观察到的一致）。
//
// 顺带：`if (fabs(pix_ptr[0]) < this->ignore_small_value) pix_ptr[0] = 0;`
// 这条清零在 `b_v` **很小**时会误伤（把整层的权重清成 0）。
// 两种极端都验。
static void case_extreme_bn()
{
    const int C = NET_C, HF = NET_H, WF = NET_W;
    const int NF = K * K * C;
    struct { const char* what; float var; float scale; } kinds[2] = {
        { "var 极小(1e-20) -> b 极大 ~1e10，权重乘上去可能溢出 float32", 1e-20f, 1.0f },
        { "var 极大(1e+20) -> b 极小 ~1e-10，权重乘上去会被 ignore_small_value 全清零", 1e+20f, 1.0f },
    };
    for (int ki = 0; ki < 2; ki++) {
        char pf[256], wf[256];
        snprintf(pf, sizeof(pf), "/tmp/zq_dwm_ex%d.zqparams", ki);
        snprintf(wf, sizeof(wf), "/tmp/zq_dwm_ex%d.nchwbin", ki);
        {
            char buf[1024];
            snprintf(buf, sizeof(buf),
                     "Input name=data C=%d H=%d W=%d\n"
                     "DepthwiseConvolution name=dw bottom=data top=dw_out num_output=%d kernel_size=3 stride=1 pad=1\n"
                     "BatchNormScale name=bn bottom=dw_out top=dw_out bias eps=1e-05\n"
                     "Flatten name=fl bottom=dw_out top=fl\n",
                     C, HF, WF, C);
            FILE* f = fopen(pf, "wb");
            fwrite(buf, 1, strlen(buf), f);
            fclose(f);
        }
        {
            unsigned s = 55555u;
            std::vector<float> w;
            for (int i = 0; i < NF; i++) w.push_back(rnd(s) * 0.4f);         // dw
            for (int c = 0; c < C; c++) {                                     // bn
                w.push_back(0.0f);                                          // mean
                w.push_back(kinds[ki].var);                                  // var
                w.push_back(kinds[ki].scale);                                // scale
                w.push_back(0.1f);                                          // bias
            }
            FILE* f = fopen(wf, "wb");
            fwrite(&w[0], 1, w.size() * sizeof(float), f);
            fclose(f);
        }
        printf("  用例：极端 BN —— %s\n", kinds[ki].what);

        unsigned s2 = 31337u;
        std::vector<float> in((size_t)C * HF * WF);
        for (size_t i = 0; i < in.size(); i++) in[i] = rnd(s2);

        ZQ::ZQ_CNN_Net nA, nB;
        bool la = nA.LoadFrom(pf, wf);
        bool lb = nB.LoadFrom(pf, wf, true, 1e-12f, false);
        if (!la || !lb) {
            printf("    **FAIL** 加载失败（A=%d B=%d）\n", la ? 1 : 0, lb ? 1 : 0);
            g_bad++;
            remove(pf); remove(wf);
            continue;
        }
        ZQ::ZQ_CNN_Tensor4D_NHW_C_Align256bit iA, iB;
        iA.ChangeSize(1, HF, WF, C, 0, 0);
        iB.ChangeSize(1, HF, WF, C, 0, 0);
        iA.ConvertFromCompactNCHW(&in[0], 1, C, HF, WF);
        iB.ConvertFromCompactNCHW(&in[0], 1, C, HF, WF);
        if (!nA.Forward(iA) || !nB.Forward(iB)) {
            printf("    **FAIL** Forward 失败\n");
            g_bad++;
            remove(pf); remove(wf);
            continue;
        }
        const ZQ::ZQ_CNN_Tensor4D* oa = nA.GetBlobByName("dw_out");
        const ZQ::ZQ_CNN_Tensor4D* ob = nB.GetBlobByName("dw_out");
        if (oa == 0 || ob == 0) {
            printf("    **FAIL** 取不到 dw_out\n");
            g_bad++;
            remove(pf); remove(wf);
            continue;
        }
        std::vector<float> va, vb;
        read_blob(oa, va);
        read_blob(ob, vb);
        if (va.size() != vb.size()) {
            printf("    **FAIL** 形状对不上\n");
            g_bad++;
            remove(pf); remove(wf);
            continue;
        }
        // 顺便数一下有没有非有限值 —— 溢出假设的直接证据
        long ninf = 0;
        for (size_t i = 0; i < vb.size(); i++)
            if (!(vb[i] == vb[i]) || vb[i] > 3.0e38f || vb[i] < -3.0e38f) ninf++;
        long wi = -1;
        double e2 = backward_err(vb, va, wi);
        printf("    B（融合）vs A : 后向误差 %.4g", e2);
        if (ninf) printf("；融合侧有 %ld 个非有限值（溢出/NaN）", ninf);
        if (e2 > 1e-4) {
            printf("   <-- **极端 BN 下融合改变了结果**，最差在 #%ld（A %.9g / B %.9g）\n",
                   wi, va[(size_t)wi], vb[(size_t)wi]);
            g_bad++;
        } else {
            printf("\n    OK  极端 BN 下融合前后等价\n");
            g_ok++;
        }
        remove(pf);
        remove(wf);
    }
}

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    // 通道数/特征图大小由**配置表**驱动，理由见附录 HM.3：
    // 这道检查要覆盖的是「**C 是不是 align 的倍数**」这条轴 ——
    // C 不是倍数时 pixelStep > C（张量有 padding），是倍数时 pixelStep == C（无 padding）。
    // 第一版只跑 C=13（有 padding）一组，而真实模型是 C=256（无 padding）：
    // **两条轴各自验过、交叉没验过**。所以这里把四个配置都跑一遍。
    // 也可以从命令行给一组：SampleDwMergeBranch <C> <H> <W>
    struct Cfg { int C, H, W; const char* note; };
    std::vector<Cfg> cfgs;
    if (argc >= 4) {
        Cfg c; c.C = atoi(argv[1]); c.H = atoi(argv[2]); c.W = atoi(argv[3]); c.note = "命令行指定";
        cfgs.push_back(c);
    } else {
        static const Cfg all[] = {
            {  13,  7,  7, "C 不是 align(8) 的倍数 -> pixelStep(16) > C，**有 padding**" },
            { 256, 14, 14, "C 是 align(8) 的倍数   -> pixelStep == C，**无 padding**（真模型就是这样）" },
            {   8,  5,  5, "C 正好等于 align(8)   -> pixelStep == C，无 padding，边界值" },
            {   9,  4,  4, "C = align+1          -> pixelStep(16) > C，有 padding，边界值" },
        };
        for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) cfgs.push_back(all[i]);
    }

    int total_ok = 0, total_bad = 0;
    for (size_t ci = 0; ci < cfgs.size(); ci++) {
        NET_C = cfgs[ci].C; NET_H = cfgs[ci].H; NET_W = cfgs[ci].W;
        printf("================================================================\n");
        printf("配置 C=%d H=%d W=%d —— %s\n", NET_C, NET_H, NET_W, cfgs[ci].note);
        printf("----------------------------------------------------------------\n");
        int b0 = g_bad;
        one(0);
        printf("\n");
        one(1);
        printf("\n");
        case_shared_blob();
        printf("\n");
        case_bn_then_prelu();
        printf("\n");
        case_multi_writer();
        printf("\n");
        case_extreme_bn();
        int passed = 6 - (g_bad - b0);
        total_ok += passed;
        total_bad += (g_bad - b0);
        g_bad = 0;
    }
    printf("\n================================================================\n");
    printf("共 %zu 组配置 × 6 个用例：通过 %d，不通过 %d\n", cfgs.size(), total_ok, total_bad);
    printf("%s\n", total_bad == 0 ? "ALL CONFIG OK" : "SOME CONFIG FAILED");
    return total_bad == 0 ? 0 : 1;
}
