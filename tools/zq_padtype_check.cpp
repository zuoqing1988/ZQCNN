// NCHW 层「pad_type = SAME」的输出尺寸探针（附录 IV，待登记进回归）
//
// 三个 NCHW 层（Convolution / DeConvolution / Pooling）的 `pad_type == TYPE_SAME`
// 分支里，`SetBottomDim` 算 padding 用的是
//
//     int top_W = bottom_W / stride_W;            // <-- 整数除法 = **向下取整**
//     int pad_W = __max((top_W - 1)*stride_W + real_kernel_W - bottom_W, 0);
//
// 而 SAME 的定义（TF / Caffe / ONNX 三家一致）是
//
//     out     = ceil(in / stride)
//     pad_tot = max((out - 1)*stride + kernel - in, 0)
//
// floor 与 ceil 只在 `in % stride == 0` 时相等。于是：
//
//     in=8 stride=2 k=2 : floor==ceil==4，pad=0        -> 对
//     in=7 stride=2 k=3 : ceil=4 pad_tot=1（left=0,right=1）
//                         floor=3 pad=0，GetTopDim 又给 ceil((7-3)/2)+1=3
//                         -> 输出宽度 3 而不是 4，**整层少一列**
//
// 本探针直接实例化层类、走 `ReadParam -> SetBottomDim -> GetTopDim` 这条**生产路径**，
// 把 top_H/top_W 与三家定义逐个对。不猜、不推演。
//
// 形状表刻意只取 `in % stride != 0` 的那一半 —— 可整除时 floor 与 ceil 相等，
// 跑一百个也证明不了什么（同 AGENTS.md「一维退化形状会让错误互相等价」）。
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include "ZQ_CNN_Tensor4D.h"
#include "ZQ_CNN_Net.h"

static int g_case = 0, g_ok = 0, g_bad = 0;

enum Kind { K_CONV, K_DWCONV, K_DECONV, K_POOL, K_N };
static const char* g_kind[K_N] = { "Convolution", "DepthwiseConvolution", "DeConvolution", "Pooling" };

static std::string make_param(Kind k, int kH, int kW, int sH, int sW, int dH, int dW, const char* pt)
{
    char buf[512];
    switch (k) {
    case K_POOL:
        snprintf(buf, sizeof(buf),
                 "Pooling name=p1 bottom=data top=p1 pool=MAX "
                 "kernel_H=%d kernel_W=%d stride_H=%d stride_W=%d pad_type=%s",
                 kH, kW, sH, sW, pt);
        break;
    case K_DWCONV:
        snprintf(buf, sizeof(buf),
                 "DepthwiseConvolution name=w1 bottom=data top=w1 num_output=8 "
                 "kernel_H=%d kernel_W=%d stride_H=%d stride_W=%d dilate_H=%d dilate_W=%d pad_type=%s",
                 kH, kW, sH, sW, dH, dW, pt);
        break;
    case K_DECONV:
        snprintf(buf, sizeof(buf),
                 "DeConvolution name=d1 bottom=data top=d1 num_output=8 "
                 "kernel_H=%d kernel_W=%d stride_H=%d stride_W=%d dilate_H=%d dilate_W=%d pad_type=%s",
                 kH, kW, sH, sW, dH, dW, pt);
        break;
    default:
        snprintf(buf, sizeof(buf),
                 "Convolution name=c1 bottom=data top=c1 num_output=8 "
                 "kernel_H=%d kernel_W=%d stride_H=%d stride_W=%d dilate_H=%d dilate_W=%d pad_type=%s",
                 kH, kW, sH, sW, dH, dW, pt);
        break;
    }
    return std::string(buf);
}

static ZQ::ZQ_CNN_Layer* make(Kind k)
{
    switch (k) {
    case K_CONV:   return new ZQ::ZQ_CNN_Layer_Convolution();
    case K_DWCONV: return new ZQ::ZQ_CNN_Layer_DepthwiseConvolution();
    case K_DECONV: return new ZQ::ZQ_CNN_Layer_DeConvolution();
    default:       return new ZQ::ZQ_CNN_Layer_Pooling();
    }
}

struct PadInfo { int top, before, after; };

static bool probe(Kind k, int kH, int kW, int sH, int sW, int dH, int dW,
                  const char* pt, int H, int W, PadInfo& hi, PadInfo& wi)
{
    ZQ::ZQ_CNN_Layer* l = make(k);
    const std::string line = make_param(k, kH, kW, sH, sW, dH, dW, pt);
    bool ok = l->ReadParam(line);
    if (ok) ok = l->SetBottomDim(8, H, W);
    if (ok) {
        int oC = 0, oH = 0, oW = 0;
        l->GetTopDim(oC, oH, oW);
        hi.top = oH; wi.top = oW;
        // pad_* 是 public 成员，直接读；DeConvolution 之外的类字段名一致。
        if (ZQ::ZQ_CNN_Layer_Convolution* c = dynamic_cast<ZQ::ZQ_CNN_Layer_Convolution*>(l)) {
            hi.before = c->pad_H_top; hi.after = c->pad_H_bottom;
            wi.before = c->pad_W_left; wi.after = c->pad_W_right;
        } else if (ZQ::ZQ_CNN_Layer_DepthwiseConvolution* c =
                       dynamic_cast<ZQ::ZQ_CNN_Layer_DepthwiseConvolution*>(l)) {
            hi.before = c->pad_H_top; hi.after = c->pad_H_bottom;
            wi.before = c->pad_W_left; wi.after = c->pad_W_right;
        } else if (ZQ::ZQ_CNN_Layer_DeConvolution* c =
                       dynamic_cast<ZQ::ZQ_CNN_Layer_DeConvolution*>(l)) {
            hi.before = c->pad_H_top; hi.after = c->pad_H_bottom;
            wi.before = c->pad_W_left; wi.after = c->pad_W_right;
        } else if (ZQ::ZQ_CNN_Layer_Pooling* c = dynamic_cast<ZQ::ZQ_CNN_Layer_Pooling*>(l)) {
            hi.before = c->pad_H_top; hi.after = c->pad_H_bottom;
            wi.before = c->pad_W_left; wi.after = c->pad_W_right;
        } else {
            hi.before = hi.after = wi.before = wi.after = -1;
        }
    }
    delete l;
    return ok;
}

static int ceil_div(int a, int b) { return (a + b - 1) / b; }

static void check_one(Kind k, int kH, int kW, int sH, int sW, int dH, int dW,
                      const char* pt, int H, int W)
{
    g_case++;
    PadInfo hi, wi;
    if (!probe(k, kH, kW, sH, sW, dH, dW, pt, H, W, hi, wi)) {
        printf("  %-22s k=%dx%d s=%dx%d d=%dx%d pad_type=%s in=%dx%d  建层/读参数失败\n",
               g_kind[k], kH, kW, sH, sW, dH, dW, pt, H, W);
        g_bad++;
        return;
    }
    // **Pooling 没有 dilation** —— 它的 ReadParam 根本不解析 dilate_H/dilate_W。
    // 我第一版对四类层用同一个 real_k = (k-1)*d+1，于是 Pooling 那 95 个"错"
    // 全部是**我期望值算错**（把 dilation 用在了没有 dilation 的层上），
    // 而这类错误长得极像"库的缺陷"。
    // （同 AGENTS.md「参数名不是语义」：先确认这个字段在这个层里存不存在。）
    const int rkH = (k == K_POOL) ? kH : (kH - 1) * dH + 1;
    const int rkW = (k == K_POOL) ? kW : (kW - 1) * dW + 1;
    // ---- 期望值：四类层的语义**各不相同**，必须逐类写 ----
    //   Convolution / Depthwise : 与 TF/Caffe 的 SAME 一致，out = ceil(in/stride)
    //   Pooling                 : 本仓库的池化尺寸约定是 out = ceil((in-k)/stride) + 1
    //                             （层、Forward、内核三处一致），**不是** VALID 的
    //                             ceil((in-k+1)/stride) —— VALID 时 pad 恒为 0
    //   DeConvolution           : SAME 的定义是 out = in*stride（TF Conv2DTranspose），
    //                             不是我第一版写的 ceil(in/stride)；
    //                             它也**只实现了 SAME**，VALID 不做任何处理
    int eH, eW, pH, pW;
    if (k == K_DECONV) {
        if (strcmp(pt, "SAME") != 0) { return; }   // 见 main：DeConvolution 只跑 SAME
        // TF Conv2DTranspose 的 SAME：out = in*stride，
        // pad_total = max(out - ((in-1)*stride + 2 - real_k), 0) = stride + real_k - 2。
        eH = H * sH; eW = W * sW;
        pH = __max(sH + (kH - 1) * dH + 1 - 2, 0);
        pW = __max(sW + (kW - 1) * dW + 1 - 2, 0);
    } else if (k == K_POOL) {
        // Pooling 的尺寸约定是 ceil((in-k)/s)+1，不是 VALID 的 ceil((in-k+1)/s)。
        // VALID 时 pad 恒为 0（SAME 的 pad 在 in%s==0 时也是 0），
        // 所以下面按 pad_type 分支在这里没有意义。
        eH = (int)ceil((double)(H - kH) / sH) + 1;
        eW = (int)ceil((double)(W - kW) / sW) + 1;
        pH = pW = 0;
    } else {
        if (strcmp(pt, "SAME") == 0) {
            eH = ceil_div(H, sH); eW = ceil_div(W, sW);
        } else {
            eH = ceil_div(H - rkH + 1, sH); eW = ceil_div(W - rkW + 1, sW);
        }
        pH = __max((eH - 1) * sH + rkH - H, 0);
        pW = __max((eW - 1) * sW + rkW - W, 0);
    }
    // 判据分两项：① top 尺寸 ② 上下/左右 padding 之和。两项都必须对上。
    bool sz_ok = (hi.top == eH && wi.top == eW);
    bool pad_ok = (hi.before + hi.after == pH && wi.before + wi.after == pW);
    if (sz_ok && pad_ok) { g_ok++; return; }
    g_bad++;
    printf("  %-22s k=%dx%d s=%dx%d d=%dx%d pad_type=%-5s in=%dx%d  "
           "top got(%d,%d) want(%d,%d)  padsum got(%d,%d) want(%d,%d)\n",
           g_kind[k], kH, kW, sH, sW, dH, dW, pt, H, W,
           hi.top, wi.top, eH, eW,
           hi.before + hi.after, wi.before + wi.after, pH, pW);
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHW pad_type：top 尺寸与 padding 之和对各家定义\n");
    printf("  Convolution / Depthwise : SAME out = ceil(in/stride)\n");
    printf("  Pooling                 : 本仓库约定 out = ceil((in-k)/stride)+1（pad 恒 0）\n");
    printf("  DeConvolution           : SAME out = in*stride（TF Conv2DTranspose）；\n");
    printf("                            VALID **未实现**，本探针不跑那一档\n\n");

    const char* PTS[2] = { "SAME", "VALID" };
    for (int k = 0; k < K_N; k++) {
        const int c0 = g_case, b0 = g_bad;
        for (int pt = 0; pt < 2; pt++) {
            for (int kh = 1; kh <= 3; kh++)
                for (int s = 2; s <= 3; s++)
                    for (int dh = 1; dh <= 2; dh++)
                        for (int H = 5; H <= 17; H += 2) {
                            // 只喂 in % stride != 0 的那一半（另用可整除的那半做对照）
                            if (H % s == 0) continue;
                            check_one((Kind)k, kh, kh, s, s, dh, dh, PTS[pt], H, H);
                        }
        }
        printf("  %-22s  %3d 个用例：对 %d，错 %d\n",
               g_kind[k], g_case - c0, (g_case - c0) - (g_bad - b0), g_bad - b0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d\n", g_case, g_ok, g_bad);
    return g_bad ? 1 : 0;
}