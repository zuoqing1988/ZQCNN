// `LSTM_TF` 的**标定装置**（附录 IN）—— 36 种层类型里最后一个零覆盖项。
//
// 为什么是"装置"而不是"再写一份参考实现"
// --------------------------------------
// 这是一个**沿时间步递推**的层（`input.GetW()` 就是时间长度），
// 四个门 I/F/O/G 加上 hidden-to-hidden 与 bias，一共 **16 个权重张量**。
// 手写参考只要有一处与实现不同，症状就与"库算错了"**完全一样** ——
// 本文件 HR 已经为同一种情况给过结论：
// 「为『某个映射/约定是什么』猜了两次以上，就改成设计一个能把它测出来的装置」。
//
// 装置的做法（HR 同款）
// -------------------
// 把权重与输入都填成**互不相同**的值，于是输出的每一个数
// **唯一地**确定"哪一块权重、哪一个时间步、经了哪条通路"。
// `hidden_dim = 1`、`input C = 1` 时每一块权重只有 **1 个 float**，
// 于是 12 块（fw 向，type=0）逐一被单独点亮，读输出就知道它接到哪个门。
//
// 契约（读 `ZQ_CNN_Layer_LSTM_TF::LayerSetup` / `GetTopDim` 得到）
// --------------------------------------------------------------
//   fw_xc_{I,F,O,G}  : [hidden_dim, 1, 1, bottom_C]     x -> gate
//   fw_hc_{I,F,O,G}  : [hidden_dim, 1, 1, hidden_dim]  h -> gate
//   fw_b_{I,F,O,G}   : [hidden_dim, 1, 1, 1]           偏置
//   out = [N, hidden*(fw?1:0 + bw?1:0), 1, bottom_W]   （top_H = 1，top_W = 时间）
//   type=0 -> 只走前向；type=1 -> 只走后向；其它 -> 两个方向都走
//
// 权重文件的读取顺序（`LoadBinary_NCHW`）就是上面这个顺序，
// 每块 `N*H*W*C` 个 float、`ConvertFromCompactNCHW` —— 本装置里全是 1 个 float，
// 所以**块与块之间无法用长度区分**，只能靠顺序。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include "ZQ_CNN_Tensor4D.h"
#include "ZQ_CNN_Net.h"

#if defined(_WIN32)
#define MODEL_DIR "model"
#else
#define MODEL_DIR "../../model"
#endif

static const char* SYNTH_PARAM = "zq_lstm_calib.zqparams";
static const char* SYNTH_MODEL = "zq_lstm_calib.nchwbin";

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

// 12 块前向权重，顺序与 `LoadBinary_NCHW` 一致
static const char* FW_BLOCKS[12] = {
    "fw_xc_I", "fw_xc_F", "fw_xc_O", "fw_xc_G",
    "fw_hc_I", "fw_hc_F", "fw_hc_O", "fw_hc_G",
    "fw_b_I",  "fw_b_F",  "fw_b_O",  "fw_b_G",
};

// hidden=1、C=1 时每块都只有 1 个 float —— 12 块 = 12 个 float。
// `lit` 是要点亮的块号（-1 = 全 0，用来看"全零"是什么样子）。
static bool run_one(int lit, int T, float value, std::vector<float>& out)
{
    std::vector<float> w(12, 0.0f);
    if (lit >= 0) w[lit] = value;
    char block[256];
    snprintf(block, sizeof(block),
             "Input name=data C=1 H=1 W=%d\n"
             "LSTM_TF name=lstm1 bottom=data top=seq hidden_dim=1 type=0 "
             "forget_bias=0 cell_clip=10\n",
             T);
    if (!write_file(SYNTH_PARAM, block, strlen(block))) return false;
    if (!write_file(SYNTH_MODEL, &w[0], w.size() * sizeof(float))) return false;

    ZQ::ZQ_CNN_Net net;
    if (!net.LoadFrom(SYNTH_PARAM, SYNTH_MODEL)) {
        printf("    %-10s 合成网加载失败\n", FW_BLOCKS[lit < 0 ? 0 : lit]);
        return false;
    }
    // 输入：只有 t=0 是 1，其余 0 —— 让"第 0 步的输入"唯一可辨
    std::vector<float> in((size_t)T, 0.0f);
    in[0] = 1.0f;
    ZQ::ZQ_CNN_Tensor4D_NHW_C_Align256bit ti;
    if (!ti.ConvertFromCompactNCHW(&in[0], 1, 1, 1, T)) return false;
    if (!net.Forward(ti)) {
        printf("    %-10s Forward 失败\n", FW_BLOCKS[lit < 0 ? 0 : lit]);
        return false;
    }
    const ZQ::ZQ_CNN_Tensor4D* ob = net.GetBlobByName("seq");
    if (ob == 0) {
        printf("    %-10s 取不到输出 blob seq\n", FW_BLOCKS[lit < 0 ? 0 : lit]);
        return false;
    }
    out.resize((size_t)ob->GetN() * ob->GetC() * ob->GetH() * ob->GetW());
    ob->ConvertToCompactNCHW(&out[0]);
    return true;
}

// 同时点亮两块（其余 0）：用来验证两个门是不是**相乘**地作用。
static bool run_two(int b1, int b2, int T, float value, std::vector<float>& out)
{
    std::vector<float> w(12, 0.0f);
    w[b1] = value;
    w[b2] = value;
    char block[256];
    snprintf(block, sizeof(block),
             "Input name=data C=1 H=1 W=%d\n"
             "LSTM_TF name=lstm1 bottom=data top=seq hidden_dim=1 type=0 "
             "forget_bias=0 cell_clip=10\n",
             T);
    if (!write_file(SYNTH_PARAM, block, strlen(block))) return false;
    if (!write_file(SYNTH_MODEL, &w[0], w.size() * sizeof(float))) return false;
    ZQ::ZQ_CNN_Net net;
    if (!net.LoadFrom(SYNTH_PARAM, SYNTH_MODEL)) return false;
    std::vector<float> in((size_t)T, 0.0f);
    in[0] = 1.0f;
    ZQ::ZQ_CNN_Tensor4D_NHW_C_Align256bit ti;
    if (!ti.ConvertFromCompactNCHW(&in[0], 1, 1, 1, T)) return false;
    if (!net.Forward(ti)) return false;
    const ZQ::ZQ_CNN_Tensor4D* ob = net.GetBlobByName("seq");
    if (ob == 0) return false;
    out.resize((size_t)ob->GetN() * ob->GetC() * ob->GetH() * ob->GetW());
    ob->ConvertToCompactNCHW(&out[0]);
    return true;
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("LSTM_TF 标定装置（附录 IN）\n");
    printf("hidden_dim=1 / C=1 / type=0（只前向）/ 输入只有 t=0 为 1\n");
    printf("逐块点亮，每块只有 1 个 float —— 输出即「这块接到哪个门」的唯一答案\n\n");

    const int T = 4;
    std::vector<float> out;

    printf("  基线（12 块全 0）：\n");
    if (run_one(-1, T, 0.0f, out)) {
        printf("    输出 %zu 个：", out.size());
        for (size_t i = 0; i < out.size() && i < 16; i++) printf(" %.6f", out[i]);
        printf("\n");
    }

    printf("\n  逐块点亮（值 = 5，sigmoid/tanh 都饱和到可分辨的一端）：\n");
    for (int b = 0; b < 12; b++) {
        if (!run_one(b, T, 5.0f, out)) continue;
        printf("    %-8s 输出 %zu 个：", FW_BLOCKS[b], out.size());
        for (size_t i = 0; i < out.size() && i < 16; i++) printf(" %.6f", out[i]);
        printf("\n");
    }

    printf("\n  组合（G + O 同时点亮，验证两个门是不是相乘）：\n");
    if (run_two(3, 11, T, 5.0f, out)) {
        printf("    %-8s 输出 %zu 个：", "xc_G+b_O", out.size());
        for (size_t i = 0; i < out.size() && i < 16; i++) printf(" %.6f", out[i]);
        printf("\n");
    }
    printf("\n  对照（只有 G，值 5）：\n");
    if (run_two(3, 3, T, 5.0f, out)) {
        printf("    %-8s 输出 %zu 个：", "xc_G", out.size());
        for (size_t i = 0; i < out.size() && i < 16; i++) printf(" %.6f", out[i]);
        printf("\n");
    }

    cleanup_synth();
    printf("\nLSTM_TF CALIB DONE\n");
    return 0;
}
