// DeConvolution **层**的权重 -> (输出格子, 输入格子) 线性映射反解（附录 IW.6）
//
// 前面两步已经把范围收窄了：
//   * `tools/zq_deconv_kernel_probe.cpp`：三条 general 内核（align0 / align128 / align256）
//     逐元素全对，ASan 干净 —— **内核没问题**。
//   * 走完整 `ZQ_CNN_Net` 的层级对照：几乎**每个**形状都对不上（C=1 的 3x3 除外）。
// 所以差异出在「层怎么把权重摆进张量 / 怎么把张量交给内核」这一段。
//
// 这次不猜，直接把这段线性映射**读出来**
// ------------------------------------------------------------
// 这一层对 `w` 是线性的，而 `in` 取成**两两可区分**的小整数：
//     in[ic][ih][iw] = (1 + ic*100 + ih*10 + iw) * 0.01
// 于是「某个输出格子的值 / 0.01 向上取整」就直接告诉我是哪个输入格子。
// 再把权重一个个置 1（`w[p] = 1`，其余 0），跑一遍，记下全部输出：
//     p  ->  {(oc,oh,ow) : (ic,ih,iw)}
// 这就是完整的映射表。四元组 (oc,kh,kw,ic) 对应哪个 p、以及它读到哪个 (ih,iw)，
// 都在表里 —— 不再需要任何关于布局的假设。
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include "ZQ_CNN_Tensor4D.h"
#include "ZQ_CNN_Net.h"
#include "zq_cnn_deconvolution_32f_align_c.h"

static const char* PARAM = "zq_deconv_layer_probe.zqparams";
static const char* MODEL  = "zq_deconv_layer_probe.nchwbin";

static void write_file(const char* path, const void* d, size_t n)
{
    FILE* f = fopen(path, "wb");
    if (!f) { printf("  写不出 %s\n", path); exit(2); }
    if (n) fwrite(d, 1, n, f);
    fclose(f);
}

static const float UNIT = 0.01f;

static int decode_code(float v, int C, int H, int W, int& ic, int& ih, int& iw)
{
    int code = (int)floor((double)v / UNIT + 0.5) - 1;
    if (code <= 0) return 0;
    ic = code / 100;
    ih = (code % 100) / 10;
    iw = code % 10;
    if (ic >= C || ih >= H || iw >= W) return 0;
    return code;
}

static bool run_layer(const char* padtype, int C, int H, int W, int OC, int KH, int KW,
                      int S, const std::vector<float>& in, const std::vector<float>& w,
                      std::vector<float>& got, int& oH, int& oW)
{
    char line[512], text[1024];
    snprintf(line, sizeof(line),
             "DeConvolution name=dc1 bottom=data top=top1 num_output=%d "
             "kernel_H=%d kernel_W=%d stride_H=%d stride_W=%d pad_type=%s\n",
             OC, KH, KW, S, S, padtype);
    snprintf(text, sizeof(text), "Input name=data C=%d H=%d W=%d\n%s", C, H, W, line);
    write_file(PARAM, text, strlen(text));
    write_file(MODEL, w.empty() ? (const void*)"" : (const void*)&w[0], w.size() * sizeof(float));
    ZQ::ZQ_CNN_Net net;
    if (!net.LoadFrom(PARAM, MODEL)) { printf("  加载失败\n"); return false; }
    ZQ::ZQ_CNN_Tensor4D_NHW_C_Align256bit ti;
    if (!ti.ConvertFromCompactNCHW(&in[0], 1, C, H, W)) { printf("  输入失败\n"); return false; }
    if (!net.Forward(ti)) { printf("  Forward 失败\n"); return false; }
    const ZQ::ZQ_CNN_Tensor4D* ob = net.GetBlobByName("top1");
    if (!ob) ob = net.GetBlobByName("data");
    if (!ob) { printf("  取不到输出\n"); return false; }
    oH = ob->GetH(); oW = ob->GetW();
    got.assign((size_t)ob->GetN() * ob->GetC() * oH * oW, 0.0f);
    ob->ConvertToCompactNCHW(&got[0]);
    return true;
}

int main(int argc, char** argv)
{
    int C = (argc > 1) ? atoi(argv[1]) : 2;
    int H = (argc > 2) ? atoi(argv[2]) : 5;
    int W = (argc > 3) ? atoi(argv[3]) : 5;
    int OC = (argc > 4) ? atoi(argv[4]) : 2;
    int KH = (argc > 5) ? atoi(argv[5]) : 3;
    int KW = KH;
    int S = (argc > 6) ? atoi(argv[6]) : 1;
    const char* padtype = (argc > 7) ? argv[7] : "VALID";

    const int NP = OC * KH * KW * C;
    std::vector<float> zero_in((size_t)C * H * W, 0.0f);
    std::vector<float> in((size_t)C * H * W);
    for (int ic = 0; ic < C; ic++)
        for (int ih = 0; ih < H; ih++)
            for (int iw = 0; iw < W; iw++)
                in[((size_t)ic * H + ih) * W + iw] = (float)(1 + ic * 100 + ih * 10 + iw) * UNIT;

    int oH = 0, oW = 0;
    // 注意：权重文件**长度必须是 NP**，不能是 0 ——
    // `LoadBinary_NCHW` 先 `dst_len = num_output*kernel_H*kernel_W*bottom_C`，
    // 再按 dst_len 读；文件短了直接 `Failed to load Binary for layer dc1`。
    std::vector<float> probe((size_t)NP, 0.0f), got;

    // ---- 先单跑一次「输入全 0」，在别的调用之前 ----
    // 上一版把它排在 54 次映射扫描之后，输出是一份**陈旧内存**（两次不同输入
    // 给出一模一样的数），根子就在这里：顺序一变结论就变，说明那份数根本不是
    // 这一层算出来的（附录 IW.7）。
    {
        unsigned sd = 7u;
        std::vector<float> w0((size_t)NP);
        for (size_t i = 0; i < w0.size(); i++) { sd = sd * 1664525u + 1013904223u; w0[i] = (float)((sd >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
        if (!run_layer(padtype, C, H, W, OC, KH, KW, S, zero_in, w0, got, oH, oW)) return 2;
        double mx = 0;
        for (size_t i = 0; i < got.size(); i++) { double a = fabs((double)got[i]); if (a > mx) mx = a; }
        printf("=== 第一次跑（输入全 0）：输出 %dx%d，最大绝对值 %.6g %s\n", oH, oW, mx,
               mx > 1e-6 ? "**非 0 —— 输入为 0 却有输出**" : "OK");
        if (oW <= 8)
            for (int oc = 0; oc < OC; oc++)
                for (int oh = 0; oh < oH; oh++) {
                    printf("      oc=%d oh=%d:", oc, oh);
                    for (int ow = 0; ow < oW; ow++) printf(" %10.5f", got[((size_t)oc * oH + oh) * oW + ow]);
                    printf("\n");
                }
    }

    if (!run_layer(padtype, C, H, W, OC, KH, KW, S, in, probe, got, oH, oW)) return 2;
    printf("=== DeConv 层级映射反解：pad=%s C=%d HxW=%dx%d OC=%d k=%dx%d s=%d -> 库输出 %dx%d ===\n",
           padtype, C, H, W, OC, KH, KW, S, oH, oW);


    for (int p = 0; p < NP; p++) {
        std::fill(probe.begin(), probe.end(), 0.0f);
        probe[p] = 1.0f;
        if (!run_layer(padtype, C, H, W, OC, KH, KW, S, in, probe, got, oH, oW)) return 2;
        if ((int)got.size() != OC * oH * oW) {
            printf("  p=%2d 输出尺寸变成 %d（期望 %d）\n", p, (int)got.size(), OC * oH * oW);
            continue;
        }
        printf("  w[%2d] ->", p);
        for (int oc = 0; oc < OC; oc++)
            for (int y = 0; y < oH; y++)
                for (int x = 0; x < oW; x++) {
                    float v = got[((size_t)oc * oH + y) * oW + x];
                    if (fabs(v) < 0.4 * UNIT) continue;
                    int ic = -1, ih = -1, iw = -1;
                    int code = decode_code(v, C, H, W, ic, ih, iw);
                    if (code == 0) printf(" (%d,%d,%d)=%.3f?", oc, y, x, v);
                    else printf(" (%d,%d,%d)<-(ic=%d,ih=%d,iw=%d)", oc, y, x, ic, ih, iw);
                }
        printf("\n");
    }

    // -----------------------------------------------------------------
    // 2) 逐元素比对。三份输入各做一遍：可区分编码的 in、随机 in、**全 0 的 in**。
    //    这一层对 w 线性、对 in 也线性，所以三份都该对得上。
    //    特别地：**输入全 0 时输出也必须全 0** —— 一旦不为 0，
    //    就证明存在一个「与输入无关的加项」（bias / 读到了填充里的脏数据），
    //    而那**不是**卷积语义，是实打实多出来的东西。
    // -----------------------------------------------------------------
    // pad_type=SAME 时 pad 由 SetBottomDim 算：**top = bottom*stride**，
    // 不是普通 SAME（stride=2 时 top 变成 2H、pad 变成 kernel）。
    int padT = 0, padB = 0, padL = 0, padR = 0;
    if (strcmp(padtype, "SAME") == 0) {
        const int pH = (H * S - 1 + KH) - (H - 1) * S - 1;
        const int pW = (W * S - 1 + KW) - (W - 1) * S - 1;
        padT = (int)ceil(pH / 2.0);
        padB = pH - padT;
        padL = (int)ceil(pW / 2.0);
        padR = pW - padL;
    }
    for (int pass = 0; pass < 3; pass++) {
        unsigned sd = 20262020u;
        std::vector<float> q_in((size_t)C * H * W), q_w((size_t)NP);
        for (size_t i = 0; i < q_in.size(); i++) { sd = sd * 1664525u + 1013904223u; q_in[i] = (float)((sd >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
        for (size_t i = 0; i < q_w.size(); i++) { sd = sd * 1664525u + 1013904223u; q_w[i] = (float)((sd >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
        const std::vector<float>& use_in = (pass == 0) ? in : (pass == 1) ? q_in : zero_in;
        if (!run_layer(padtype, C, H, W, OC, KH, KW, S, use_in, q_w, got, oH, oW)) return 2;
        std::vector<float> want((size_t)OC * oH * oW, 0.0f);
        for (int oc = 0; oc < OC; oc++)
            for (int oh = 0; oh < oH; oh++)
                for (int ow = 0; ow < oW; ow++) {
                    double acc = 0;
                    for (int kh = 0; kh < KH; kh++) {
                        int nh = oh - padT + kh;
                        if (nh < 0 || nh % S) continue;
                        int ih = nh / S;
                        if (ih >= H) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int nw = ow - padL + kw;
                            if (nw < 0 || nw % S) continue;
                            int iw = nw / S;
                            if (iw >= W) continue;
                            for (int ic = 0; ic < C; ic++)
                                acc += (double)q_w[(((size_t)oc * C + ic) * KH + kh) * KW + kw]
                                     * (double)use_in[(((size_t)ic * H) + ih) * W + iw];
                        }
                    }
                    want[((size_t)oc * oH + oh) * oW + ow] = (float)acc;
                }
        double worst = 0; long wi = -1;
        for (size_t i = 0; i < want.size(); i++) {
            double d = fabs((double)got[i] - want[i]);
            if (d > worst) { worst = d; wi = (long)i; }
        }
        printf("逐元素比对（%s输入）：输出 %dx%d，最大绝对偏差 %.6g"
               "（最差 (oc=%d,oh=%d,ow=%d)：库 %.6f / 参考 %.6f）\n",
               (pass == 0) ? "可区分编码" : (pass == 1) ? "随机" : "全 0", oH, oW, worst,
               (int)(wi / (oH * oW)), (int)((wi / oW) % oH), (int)(wi % oW),
               wi >= 0 ? got[wi] : 0.0, wi >= 0 ? want[wi] : 0.0);
        if (oW <= 6 && worst > 1e-4) {
            for (int oc = 0; oc < OC; oc++)
                for (int oh = 0; oh < oH; oh++) {
                    printf("        oc=%d oh=%d 库:", oc, oh);
                    for (int ow = 0; ow < oW; ow++) printf(" %8.3f", got[((size_t)oc * oH + oh) * oW + ow]);
                    printf("\n               参:");
                    for (int ow = 0; ow < oW; ow++) printf(" %8.3f", want[((size_t)oc * oH + oh) * oW + ow]);
                    printf("\n");
                }
        }
    }
    remove(PARAM); remove(MODEL);

    // -----------------------------------------------------------------
    // 3) 全 1 用例。in 全 1、w 全 1 时，VALID/stride1/无 pad 下每个输出的值
    //    应该是**同一个数** = (参与的 kh,kw 个数) x C = 9 x C。
    //    这个数不需要任何参考实现就能手算，**装置配错也不会骗人** ——
    //    前面那版把「零输入」那一趟的统计打在 run_layer **之前**，
    //    读到的是上一趟的缓冲，于是「输入全 0 也有输出」这个结论是假的（附录 IW.7）。
    // -----------------------------------------------------------------
    {
        std::vector<float> one_in((size_t)C * H * W, 1.0f), one_w((size_t)NP, 1.0f);
        if (!run_layer(padtype, C, H, W, OC, KH, KW, S, one_in, one_w, got, oH, oW)) return 2;
        double mx = 0, mn = 1e30;
        for (size_t i = 0; i < got.size(); i++) {
            double a = (double)got[i];
            if (a > mx) mx = a;
            if (a < mn) mn = a;
        }
        printf("全 1 用例：输出 %dx%d，最小 %.6g / 最大 %.6g", oH, oW, mn, mx);
        for (int oh = 0; oh < oH && oh <= 2; oh++)
            for (int ow = 0; ow < oW; ow++) {
                int taps = 0;
                for (int kh = 0; kh < KH; kh++) { if (oh + kh < H) taps++; }
                (void)taps;
            }
        printf("；角上 (0,0) 的理论值 = 9 x C = %d\n", 9 * C);
    }

    // -----------------------------------------------------------------
    // 4) **完全绕开层**：拿同样的输入/权重，自己按层实际用的三个 step
    //    （in pix=8 wid=40 sli=200 / f pix=8 wid=24 sli=72 / out pix=8 wid=24 sli=72）
    //    调 align128bit 内核，和层给的输出逐元素比。
    //    这一步把「层传下去的东西对不对」和「内核算得对不对」彻底分开：
    //    两者对不上 -> 层传错了；两者一致 -> 内核在这个组合下有问题。
    // -----------------------------------------------------------------
    {
        unsigned sd = 20262020u;
        std::vector<float> r_in((size_t)C * H * W), r_w((size_t)NP);
        for (size_t i = 0; i < r_in.size(); i++) { sd = sd * 1664525u + 1013904223u; r_in[i] = (float)((sd >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
        for (size_t i = 0; i < r_w.size(); i++) { sd = sd * 1664525u + 1013904223u; r_w[i] = (float)((sd >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
        if (!run_layer(padtype, C, H, W, OC, KH, KW, S, r_in, r_w, got, oH, oW)) return 2;

        const int i_pix = 8, i_wid = 8 * W,  i_sli = 8 * W * H;
        const int f_pix = 8, f_wid = 8 * KW, f_sli = 8 * KW * KH;
        const int o_pix = 8, o_wid = 8 * oW, o_sli = 8 * oW * oH;
        std::vector<float> ib((size_t)i_sli, 0.0f), fb((size_t)f_sli * OC, 0.0f), ob((size_t)o_sli, -777.0f);
        for (int ic = 0; ic < C; ic++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    ib[(size_t)y * i_wid + (size_t)x * i_pix + ic] = r_in[((size_t)ic * H + y) * W + x];
        for (int oc = 0; oc < OC; oc++)
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++)
                    for (int ic = 0; ic < C; ic++)
                        fb[(size_t)oc * f_sli + kh * f_wid + kw * f_pix + ic] =
                            r_w[(((size_t)oc * C + ic) * KH + kh) * KW + kw];
        zq_cnn_deconv_with_padding_32f_align128bit_general(
            &ib[0], 1, H, W, C, i_pix, i_wid, i_sli,
            &fb[0], OC, KH, KW, C, f_pix, f_wid, f_sli, S, S, 1, 1,
            &ob[0], 1, oH, oW, OC, o_pix, o_wid, o_sli, padT, padB, padL, padR);
        double worst = 0; long wi = -1;
        for (int oc = 0; oc < OC; oc++)
            for (int y = 0; y < oH; y++)
                for (int x = 0; x < oW; x++) {
                    float kv = got[((size_t)oc * oH + y) * oW + x];
                    float kd = ob[(size_t)y * o_wid + (size_t)x * o_pix + oc];
                    double d = fabs((double)kv - kd);
                    if (d > worst) { worst = d; wi = ((size_t)oc * oH + y) * oW + x; }
                }
        printf("层 vs 直接调 align128bit 内核（同样的 step）：最大绝对偏差 %.6g", worst);
        if (worst > 1e-4) {
            int oc = (int)(wi / (oH * oW)), oh = (int)((wi / oW) % oH), ow = (int)(wi % oW);
            printf("（最差 (oc=%d,oh=%d,ow=%d)：层 %.6f / 内核 %.6f）",
                   oc, oh, ow, got[wi], ob[(size_t)oh * o_wid + (size_t)ow * o_pix + oc]);
        }
        printf("\n");
        if (worst > 1e-4) {
            printf("      我这边 r_w[0..11] =");
            for (int i = 0; i < 12; i++) printf(" %g", r_w[i]);
            printf("\n      我这边 r_in[0..5] =");
            for (int i = 0; i < 6; i++) printf(" %g", r_in[i]);
            printf("\n      我这边 r_in[25..29] =");
            for (int i = 25; i < 30; i++) printf(" %g", r_in[i]);
            printf("\n      我这边的 in [0..7] =");
            for (int i = 0; i < 8; i++) printf(" %g", ib[i]);
            printf("\n      我这边的 f  [0..7] =");
            for (int i = 0; i < 8; i++) printf(" %g", fb[i]);
            printf("\n      我这边 f[24..31] =");
            for (int i = 24; i < 32; i++) printf(" %g", fb[i]);
            printf("\n      我这边 f[72..79] =");
            for (int i = 72; i < 80; i++) printf(" %g", fb[i]);
            printf("\n      内核 out[0..7]  =");
            for (int i = 0; i < 8; i++) printf(" %g", ob[i]);
            printf("\n      内核 out[8..15] =");
            for (int i = 8; i < 16; i++) printf(" %g", ob[i]);
            printf("\n");
        }
    }

    remove(PARAM); remove(MODEL);
    printf("DECONV LAYER PROBE DONE\n");
    return 0;
}
