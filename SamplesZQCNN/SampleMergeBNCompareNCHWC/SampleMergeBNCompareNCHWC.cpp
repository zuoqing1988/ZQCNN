// 「融合 vs 不融合」在 **NCHWC** 那条路径上的前向对照（附录 HY）。
//
// 为什么要有这个 sample
// --------------------
// 附录 HX 把 `merge_bn` 改坏了 `mobilefacenet-v1` 的那条活缺陷定位并修掉了，
// 而修的时候发现**同一段守卫在仓库里有三份**（各 5 处）：
//
//   ZQCNN/ZQ_CNN_Net.h                        主文件（NCHW 张量）
//   ZQCNN/ZQ_CNN_Net_NCHWC.h                  **生产代码**
//   ZQCNN_to_MNN/converter/source/ZQ_CNN_Net.h 转换器那份拷贝（编不了）
//
// 主文件那份修完立刻有 `SampleMergeBNCompare` 盯着（17 个模型全过）。
// **NCHWC 那份没有任何东西盯着**，而它**确实走生产路径**：
//
//   ZQCNN/ZQ_CNN_MTCNN_NCHWC.h:109-113
//       pnet[i].LoadFrom(pnet_param, pnet_model, true, 1e-9, true)
//       && rnet[i].LoadFrom(rnet_param, rnet_model, true, 1e-9, true)
//       && onet[i].LoadFrom(onet_param, onet_model, true, 1e-9, true);
//
// `SampleMTCNN_NCHWC4` 确实跑的就是它，但那个 sample **只看检出张数** ——
// 而"融合算错"的典型症状恰恰是**检出数不变、框变歪**（附录 GS 已经记过一次：
// 模型被悄悄弄坏的形态不是崩溃，是安静地给出错误结果）。
// 所以"它在回归里"**不等于**"它的融合结果被验证过"。
//
// 本 sample 就是那个验证：同一条 NCHWC 路径、同一份确定性输入，
// 两条加载方式各跑一次 `Forward`，比最后一个 blob 的**后向误差**。
//
// 判据沿用 HE.2（不重新发明）：`B(merge) vs A(默认)`，
// 两侧都过库的解释，与权重布局无关；阈值 1e-4（float32 下折叠的舍入累积）。
//
// 退出码：0 = 全部在阈值内；1 = 至少一个超阈值，或**一个模型都没跑成**
// （"没检查"不是"通过" —— 2026-10-04 在 NCHW 那份上栽过一次，
//  见 `SampleMergeBNCompare.cpp` 里 MODEL_DIR 的注释）。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include "ZQ_CNN_Tensor4D.h"
#include "ZQ_CNN_Net_NCHWC.h"

// Windows 产物的 model/ 是指向仓库根 model/ 的**目录联接**，所以 Windows 侧
// 用 "model"，Linux 侧用 "../../model"（sample 一律从产物目录跑）。
#if defined(_WIN32)
#define MODEL_DIR "model"
#else
#define MODEL_DIR "../../model"
#endif

static const double BACKWARD_ERR_LIMIT = 1e-4;
static const float PROD_IGNORE_SMALL = 1e-9f;

static bool file_exists(const std::string& p)
{
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

// 确定性伪随机：两个 net 必须拿到**逐位相同**的输入。
static float rnd(unsigned& s)
{
    s = s * 1664525u + 1013904223u;
    return (float)((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

static bool parse_param(const std::string& zp, int& C, int& H, int& W, std::string& top)
{
    FILE* f = fopen(zp.c_str(), "rb");
    if (!f) return false;
    char line[4096];
    std::string last;
    C = H = W = 0;
    while (fgets(line, sizeof(line), f)) {
        std::string s(line);
        while (!s.empty() && (s[s.size() - 1] == '\n' || s[s.size() - 1] == '\r')) s.erase(s.size() - 1);
        if (s.empty() || s[0] == '#') continue;
        if (s.compare(0, 5, "Input") == 0)
            sscanf(s.c_str(), "Input name=%*s C=%d H=%d W=%d", &C, &H, &W);
        last = s;
    }
    fclose(f);
    if (last.empty()) return false;
    size_t t = last.find("top=");
    if (t == std::string::npos) return false;
    size_t e = t + 4;
    while (e < last.size() && last[e] != ' ' && last[e] != '\t') e++;
    top = last.substr(t + 4, e - (t + 4));
    return C > 0 && H > 0 && W > 0 && !top.empty();
}

// 模板而不是写死 `const ZQ_CNN_Tensor4D*`：
// `ZQ_CNN_Net_NCHWC<Tensor4D>::GetBlobByName` 返回的是 **`const Tensor4D*`**，
// 也就是 `const ZQ_CNN_Tensor4D_NCHWC4*`，不是基类指针 ——
// 写成基类参数会得到 `error C2440: 无法将 "const Tensor4D *" 转换为 ...`
//（2026-10-05 第一版就写成基类，当场编不过。）
// 这里只需要 `GetN/C/H/W` 与 `ConvertToCompactNCHW`，两者都在基类上，
// 所以模板参数约束成「至少有这五个成员」即可。
template <class T>
static void read_blob(const T* b, std::vector<float>& out)
{
    int N = b->GetN(), C = b->GetC(), H = b->GetH(), W = b->GetW();
    out.resize((size_t)N * C * H * W);
    b->ConvertToCompactNCHW(&out[0]);
}

static double backward_err(const std::vector<float>& got,
                           const std::vector<float>& exp,
                           long& worst_i)
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

// 随仓的模型名硬编码在这里（同 NCHW 那份的理由：扫目录要用 <dirent.h>，
// 那是 POSIX 的，而这个 sample **双平台都要编**）。
static const char* MODELS[] = {
    "det1-dw20-fast", "det1-dw20-plus",
    "det2-dw24-fast", "det2-dw24-p0",   "det2-dw24-plus",
    "det3-dw48-fast", "det3-dw48-p0",   "det3-dw48-plus",
    "det4-dw48-v2n",  "det4-dw48-v2s",  "det4-dw64-v3s",
    "det5-dw64-v3s",  "det5-dw96-v2s",  "det5-dw96-v2t",  "det5-dw96-v3s",
    "det5-dw112",
    "mobilefacenet-v1",
};

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC 路径：融合 vs 不融合 前向对照（附录 HY）\n");
    printf("判据：同一输入两条路径加载后跑 Forward，最后一个 blob 的后向误差 <= %g\n",
           BACKWARD_ERR_LIMIT);
    printf("      融合实参 = 生产实参 (merge_bn=true, ignore_small_value=%g, merge_prelu=true)\n\n",
           PROD_IGNORE_SMALL);

    int ok = 0, bad = 0, skip = 0;
    const size_t nmodels = sizeof(MODELS) / sizeof(MODELS[0]);
    for (size_t i = 0; i < nmodels; i++) {
        const char* base = MODELS[i];
        std::string zp = std::string(MODEL_DIR) + "/" + base + ".zqparams";
        std::string mp = std::string(MODEL_DIR) + "/" + base + ".nchwbin";
        if (!file_exists(zp) || !file_exists(mp)) {
            printf("%-22s SKIP (仓库里没有配套的 .zqparams/.nchwbin)\n", base);
            skip++;
            continue;
        }
        int C = 0, H = 0, W = 0;
        std::string top;
        if (!parse_param(zp, C, H, W, top)) {
            printf("%-22s SKIP (取不到 Input 的 C/H/W 或最后一层的 top —— "
                   "Forward 需要显式形状)\n", base);
            skip++;
            continue;
        }

        unsigned seed = 12345u;
        std::vector<float> in((size_t)C * H * W);
        for (size_t k = 0; k < in.size(); k++) in[k] = rnd(seed);

        // 基线：默认参数（merge_bn=false, merge_prelu=false）
        ZQ::ZQ_CNN_Net_NCHWC<ZQ::ZQ_CNN_Tensor4D_NCHWC4> nA;
        if (!nA.LoadFrom(zp, mp)) {
            printf("%-22s SKIP (NCHWC4 这条路径加载不了这个模型 —— "
                   "不是缺陷，是这个 net 不支持它的某些层)\n", base);
            skip++;
            continue;
        }
        ZQ::ZQ_CNN_Tensor4D_NCHWC4 inA;
        inA.ConvertFromCompactNCHW(&in[0], 1, C, H, W);
        if (!nA.Forward(inA)) {
            printf("%-22s SKIP (基线 Forward 失败)\n", base);
            skip++;
            continue;
        }
        const ZQ::ZQ_CNN_Tensor4D_NCHWC4* oa = nA.GetBlobByName(top);
        if (oa == 0) {
            printf("%-22s SKIP (基线取不到输出 blob \"%s\")\n", base, top.c_str());
            skip++;
            continue;
        }
        std::vector<float> va;
        read_blob(oa, va);

        // 三种融合组合分别跑一遍，判据只看生产那一档；
        // 分开跑是为了能指出**是哪一个 merge** 改坏了结果。
        static const bool CFG[3][2] = { { true, false }, { false, true }, { true, true } };
        static const char* CFGN[3] = { "bn only", "prelu only", "bn+prelu (prod)" };
        double errs[3] = { -1.0, -1.0, -1.0 };
        int ngot = 0;
        for (int ci = 0; ci < 3; ci++) {
            ZQ::ZQ_CNN_Net_NCHWC<ZQ::ZQ_CNN_Tensor4D_NCHWC4> nX;
            if (!nX.LoadFrom(zp, mp, CFG[ci][0], PROD_IGNORE_SMALL, CFG[ci][1])) continue;
            ZQ::ZQ_CNN_Tensor4D_NCHWC4 inX;
            inX.ConvertFromCompactNCHW(&in[0], 1, C, H, W);
            if (!nX.Forward(inX)) continue;
            const ZQ::ZQ_CNN_Tensor4D_NCHWC4* ox = nX.GetBlobByName(top);
            if (ox == 0) continue;
            std::vector<float> vx;
            read_blob(ox, vx);
            long wi = -1;
            errs[ci] = backward_err(vx, va, wi);
            ngot++;
        }
        if (ngot < 3) {
            printf("%-22s BAD  三种组合里有 %d 种加载/Forward/取 blob 失败\n",
                   base, 3 - ngot);
            bad++;
            continue;
        }
        double prod = errs[2];
        if (prod > BACKWARD_ERR_LIMIT) {
            const char* culprit = "无（单看最后一档）";
            if (errs[0] > BACKWARD_ERR_LIMIT) culprit = "merge_bn";
            else if (errs[1] > BACKWARD_ERR_LIMIT) culprit = "merge_prelu";
            else culprit = "两者之一（需再细分）";
            printf("%-22s BAD  后向误差 %.4g > %g（bn only %.4g / prelu only %.4g）"
                   " -> %s 是元凶，top=%s\n", base, prod, BACKWARD_ERR_LIMIT,
                   errs[0], errs[1], culprit, top.c_str());
            bad++;
        } else {
            printf("%-22s OK   后向误差 %.4g（bn only %.4g / prelu only %.4g），"
                   "输出 %zu 个 float，top=%s\n", base, prod, errs[0], errs[1],
                   va.size(), top.c_str());
            ok++;
        }
    }
    printf("\n共 %zu 个模型：跑过 %d（通过 %d，超阈值 %d），跳过 %d\n",
           nmodels, ok + bad, ok, bad, skip);
    // **一个都没跑过 = 失败**，不是通过。理由同 NCHW 那份：
    // "没检查"与"检查通过"在输出上曾经一模一样。
    if (ok + bad == 0) {
        printf("NCHWC MERGE COMPARE FAILED（一个模型都没跑过 —— 这不是通过）\n");
        return 1;
    }
    printf("%s\n", bad == 0 ? "NCHWC MERGE COMPARE OK" : "NCHWC MERGE COMPARE FAILED");
    return bad == 0 ? 0 : 1;
}
