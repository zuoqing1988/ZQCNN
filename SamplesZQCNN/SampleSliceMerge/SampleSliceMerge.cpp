// 用**真实权重**的合成网跑 `merge_bn` 对照（附录 HV）。
//
// 背景：HE.4 找到的那条活缺陷 —— `merge_bn` 在生产路径上把
// `mobilefacenet-v1` 的输出改了 0.37（后向误差），而 HH.3 把范围锁死在
// 产出 `res4_block1_conv_dw` 那个 blob 的 dwconv 层上。
// 之后 HN/HP 排除了 11 类原因，HN.4 指出**缺的是真实数据**；
// HU 把那一层（连同它后面的 5 层）的**真实权重**从 `.nchwbin` 里精确切了出来。
//
// 本 sample 就是把那 5 层 + 一个合成的 `Input` 组成一个 6 层的小网，
// 用**真实权重**跑 `merge_bn` 对照。两种结果都很有用：
//   * **复现**（后向误差大）=> 缺陷落进了一个 6 层的最小用例里；
//   * **不复现**（后向误差 ~1e-8）=> 说明缺陷还需要那 70 多层的前置上下文，
//     而这一段（dwconv + BN + PReLU + conv + BN）的融合**本身是对的**。
//
// **这正是本 sample 要回答的问题**，所以两种结果都不是"失败"，
// 但**必须把结论显式打出来** —— 一个不表态的检查等于没有。
//
// 输入由 `tools/slice_model_weights.py` 生成（附录 HU）：
//   cmake-out-*/Release/.zqslice/slice.zqparams   单段（5 层）
//   cmake-out-*/Release/.zqslice/slice.nchwbin
//   cmake-out-*/Release/.zqslice/multi.zqparams   四段共享同一 blob（20 层）
//   cmake-out-*/Release/.zqslice/multi.nchwbin
// **multi 优先**：HV.3 那张表的最后一格（共享拓扑 + 真实权重）就是这一段，
// 单段在 HV.2 里已经测过且是对的，只当回退。层定义是**逐字照搬真模型**的，
// 所以四个 dwconv 的 bottom/top 与 `model/mobilefacenet-v1.zqparams`
// 的 72/81/90/99 行完全一样（都在写 `res4_block1_conv_dw`）——
// "共享拓扑"这个变量是真的，不是手工搭的假拓扑。
// 文件不在就**明确报"没生成"并退出非 0**，不空跑。
//
// 判据沿用 HE.2：`B(merge_bn) vs A(默认参数)`，两侧都过库的解释，
// 与权重布局无关。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "ZQ_CNN_Tensor4D.h"
#include "ZQ_CNN_Net.h"

// **两处都找，且全用正斜杠**。
//   * 两处：`slice_model_weights.py` 把切片写在 `cmake-out-unix-x64/...`，
//     Windows 产物在另一棵树下面。2026-10-04 实测：只写死一处，
//     Windows 侧就报"找不到"（报得对、退出码也对，但那一侧等于没验）。
//   * 正斜杠：Windows 的文件 API 本来就接受正斜杠，而**反斜杠在本会话的
//     shell heredoc 里会被吃掉**（我为此把路径拼错过三次：
//     `.zqslice\slice.zqparams` 变成 `.zqsliceslice.zqparams`）。
//     既然躲不开，就**一个反斜杠都不写**。
// 注意：本 sample **是从产物目录里跑的**（与其它 sample 一样），
// 所以候选路径是**相对产物目录**的，不是相对仓库根。
// 2026-10-04 实测：第一个候选我写成 `cmake-out-win32-x64/.../.zqslice`，
// 而 cwd 已经是产物目录，于是拼成了 `<产物>/cmake-out-win32-x64/...` —— 找不到。
// Windows 侧正确的是**直接** `.zqslice`。
static const char* SLICE_DIRS[] = {
#if defined(_WIN32)
    ".zqslice",
    "../../cmake-out-unix-x64/Release/.zqslice",
    "../../cmake-out-win32-x64/release/Release/.zqslice",
#else
    ".zqslice",
    "../../cmake-out-unix-x64/Release/.zqslice",
    "../../cmake-out-win32-x64/release/Release/.zqslice",
#endif
};
static char P[512] = "";
static char W[512] = "";
static const char* STEM = "";
// 段名：multi 优先（HV.3 里唯一没测过的那格），slice 回退。
// 2026-10-05 实测：`multi.zqparams` 里最后那个 `top=` 是
// `res4_block4_conv_sep`，与 slice 的 `res4_block1_conv_sep` **不是同一个 blob**，
// 所以输出名字不能写死 —— 见 `read_last_top()`。
static const char* STEMS[] = { "multi", "slice" };
static bool find_slice()
{
    for (size_t s = 0; s < sizeof(STEMS) / sizeof(STEMS[0]); s++) {
        for (size_t i = 0; i < sizeof(SLICE_DIRS) / sizeof(SLICE_DIRS[0]); i++) {
            char cand[512];
            snprintf(cand, sizeof(cand), "%s/%s.zqparams", SLICE_DIRS[i], STEMS[s]);
            FILE* f = fopen(cand, "rb");
            if (!f) continue;
            fclose(f);
            snprintf(P, sizeof(P), "%s", cand);
            snprintf(W, sizeof(W), "%s/%s.nchwbin", SLICE_DIRS[i], STEMS[s]);
            STEM = STEMS[s];
            return true;
        }
    }
    return false;
}
// 取 .zqparams 里**最后一个** `top=`：那就是这个合成网最后一个 blob 的名字。
// 为什么不能用写死的 `res4_block1_conv_sep`：单段网它确实是最后一个（HV.2 验过），
// 但 multi 网的最后一段是 block4，末行 `top=res4_block4_conv_sep`，
// 写死就会在 `GetBlobByName` 上拿到空指针，然后打印一个把 bug 掩盖掉的 FAIL。
// 2026-10-05 因此踩过一次：判据本身是对的，取名字取错了。
static bool read_last_top(char* out, size_t n)
{
    out[0] = 0;
    FILE* f = fopen(P, "rb");
    if (!f) return false;
    char line[1024];
    bool got = false;
    while (fgets(line, sizeof(line), f)) {
        char* t = strstr(line, "top=");
        if (!t) continue;
        t += 4;
        char* e = t;
        while (*e && *e != ' ' && *e != '\t' && *e != '\r' && *e != '\n') e++;
        size_t len = (size_t)(e - t);
        if (len == 0 || len + 1 > n) continue;
        memcpy(out, t, len);
        out[len] = 0;
        got = true;
    }
    fclose(f);
    return got;
}

static const double LIMIT = 1e-4;

// 从 .zqparams 的 Input 行取形状（合成网是工具生成的，形状必然在那里）
static bool read_shape(int& C, int& H, int& Wd)
{
    FILE* f = fopen(P, "rb");
    if (!f) return false;
    char line[1024];
    bool got = false;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "Input", 5) == 0) {
            got = (sscanf(line, "Input name=%*s C=%d H=%d W=%d", &C, &H, &Wd) == 3);
            break;
        }
    }
    fclose(f);
    return got;
}

static void read_blob(const ZQ::ZQ_CNN_Tensor4D* b, std::vector<float>& out)
{
    int N = b->GetN(), C = b->GetC(), H = b->GetH(), Wd = b->GetW();
    out.resize((size_t)N * C * H * Wd);
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

int main()
{
    printf("真实权重合成网：merge_bn 对照（附录 HV / HW）\n");
    printf("权重片段来自 tools/slice_model_weights.py 从 model/*.nchwbin 精确切出。\n\n");

    int C = 0, H = 0, Wd = 0;
    if (!find_slice()) {
        printf("  **FAIL** 所有候选下都没找到切出来的权重：\n");
        for (size_t s = 0; s < sizeof(STEMS) / sizeof(STEMS[0]); s++)
            for (size_t i = 0; i < sizeof(SLICE_DIRS) / sizeof(SLICE_DIRS[0]); i++)
                printf("        试过 %s/%s.zqparams\n", SLICE_DIRS[i], STEMS[s]);
        printf("        先生成（multi 优先）：\n"
               "          python tools/slice_model_weights.py --multi"
               " model/mobilefacenet-v1.zqparams res4_block1_conv_dw res4_block2_conv_dw"
               " res4_block3_conv_dw res4_block4_conv_dw\n"
               "          python tools/slice_model_weights.py"
               " model/mobilefacenet-v1.zqparams res4_block1_conv_dw\n");
        return 1;
    }
    char OUT[256] = "";
    if (!read_last_top(OUT, sizeof(OUT))) {
        printf("  **FAIL** %s 里读不出最后一行的 top=\n", P);
        return 1;
    }
    printf("  段名：%s\n", STEM);
    printf("  权重片段：%s\n", P);
    printf("  末层输出 blob：%s\n", OUT);
    if (!read_shape(C, H, Wd)) {
        printf("  **FAIL** %s 里读不出 Input 的形状\n", P);
        return 1;
    }
    printf("  合成网形状 C=%d H=%d W=%d\n", C, H, Wd);

    std::vector<float> in((size_t)C * H * Wd);
    unsigned s = 20261004u;
    for (size_t i = 0; i < in.size(); i++) {
        s = s * 1664525u + 1013904223u;
        // 取值范围收一点，避免激活后溢出（这里要的是"有效数字"，不是"极端值"）
        in[i] = (float)((s >> 9) & 0xFFFF) / 32768.0f - 0.5f;
    }

    ZQ::ZQ_CNN_Net nA, nB;
    if (!nA.LoadFrom(P, W)) { printf("  **FAIL** 不融合那条加载失败\n"); return 1; }
    // 生产实参：merge_bn=true, ignore_small_value=1e-9, merge_prelu=true
    if (!nB.LoadFrom(P, W, true, 1e-9f, true)) { printf("  **FAIL** 融合那条加载失败\n"); return 1; }

    ZQ::ZQ_CNN_Tensor4D_NHW_C_Align256bit iA, iB;
    if (!iA.ChangeSize(1, H, Wd, C, 0, 0) || !iB.ChangeSize(1, H, Wd, C, 0, 0)) {
        printf("  **FAIL** 输入张量 ChangeSize 失败\n");
        return 1;
    }
    iA.ConvertFromCompactNCHW(&in[0], 1, C, H, Wd);
    iB.ConvertFromCompactNCHW(&in[0], 1, C, H, Wd);
    if (!nA.Forward(iA) || !nB.Forward(iB)) {
        printf("  **FAIL** Forward 失败\n");
        return 1;
    }
    const ZQ::ZQ_CNN_Tensor4D* oa = nA.GetBlobByName(OUT);
    const ZQ::ZQ_CNN_Tensor4D* ob = nB.GetBlobByName(OUT);
    if (oa == 0 || ob == 0) {
        printf("  **FAIL** 取不到末层输出 %s（A=%s B=%s）\n",
               OUT, oa ? "有" : "无", ob ? "有" : "无");
        return 1;
    }
    std::vector<float> va, vb;
    read_blob(oa, va);
    read_blob(ob, vb);
    long wi = -1;
    double e = backward_err(vb, va, wi);
    printf("  输出 %s：%zu 个 float\n", OUT, va.size());
    printf("  B(merge_bn) vs A(默认) 后向误差 %.4g", e);
    if (e > LIMIT) {
        printf("   <== **复现了**（最差 #%ld）\n", wi);
        printf("\nSLICE MERGE: REPRODUCED（真实权重 + 共享 blob 就够）\n");
        return 1;
    }
    printf("   <= %g\n", LIMIT);
    // 注意：这段结论**已被附录 HX 推翻**，别再照着它往下查。
    // 共享拓扑本身是对的（两个平台都是 7.205e-09），
    // 而 `mobilefacenet-v1` 那 0.37 的根因是 `res4_block5` 那一对
    // **dwconv.top != BN.bottom** —— 见 `tools/check_bn_prelu_pairing.py`。
    printf("\nSLICE MERGE: NOT REPRODUCED —— 段 `%s` 的融合**本身是对的**。\n", STEM);
    if (strcmp(STEM, "multi") == 0)
        printf("  即 HV.3 那张表的最后一格（共享拓扑 + 真实权重）测下来是**对**的。\n"
               "  真正的根因见附录 HX：block5 的 dwconv 写了一个**没人读**的 blob，\n"
               "  而它的 BN 读的是上一个 block 留下的 blob（接线错，不是算术错）。\n");
    else
        printf("  缺陷还需要 mobilefacenet-v1 前那 70 多层的上下文。\n");
    return 0;
}
