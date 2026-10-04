// 「融合」与「不融合」两条加载路径的**前向输出对照**（附录 HE）。
//
// 为什么要有这个 sample
// --------------------
// `_merge_bn`（`ZQ_CNN_Net.h:1538`，132 行）与 `_merge_prelu`（1670，96 行）
// 是**会删层 + 重连 blob** 的代码。而：
//
//   * **每一道门禁**都用 `LoadFrom` 的默认参数（`merge_bn=false`、`merge_prelu=false`），
//     也就是说这两条路径**零行为覆盖**；
//   * 而生产恰恰走的是它们（`ZQ_CNN_MTCNN.h:109`）：
//         pnet[i].LoadFrom(pnet_param, pnet_model, true, 1e-9, true)
//
// 融合的**唯一**目的就是「结果不变、快一点」—— 结果变了就是缺陷。
//
// 为什么是 sample 而不是门禁
// -------------------------
// 这道对照必须**真的跑 Forward**，而 `Forward` 会调遍 `ZQ_CNN_Forward_SSEUtils`
// 里的每一个卷积辅助函数。门禁那边靠"绊线桩"顶掉那些符号（见
// `tools/zq_net_fwd_tripwires.h`），那样一跑 Forward 桩就响（rc=3）。
// 而链真实的 `ZQ_CNN_Forward_SSEUtils.cpp` 会拖进整个 GEMM 内核库
// （`layers_c/zq_cnn_convolution_gemm_32f_align_c.c` 单个 TU 在 -O1 下编一次
// 5 分钟以上），放进每轮都跑的回归里不现实。
// sample 链的是**真库**，CMake 用 `file(GLOB)` 自动编，不需要额外配置。
//
// 判据
// ----
// 同一份确定性输入，分别用两条路径加载，跑 `Forward`，比较最后一个 blob。
// 误差用**后向误差**而不是相对误差（AGENTS.md「GEMM 的判据必须用后向误差」）：
//     err = max_i |got_i - exp_i| / ||exp||_2
// 相对误差在结果被抵消到很小的地方会报出"完全不对"，
// 而那只是 float32 的固有性质（见附录 BN 那条）。
//
// 退出码：0 = 全部在阈值内；1 = 至少一个超阈值。
// 输出里**每行自带模型名与判据数值**（AGENTS.md：通用工具的输出要自带它作用于谁）。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include "ZQ_CNN_Tensor4D.h"
#include "ZQ_CNN_Net.h"

// Windows 产物的 model/ 是指向仓库根 model/ 的**目录联接**（见 build-with-cmake.md:15），
// 所以 Windows 侧要用 "model" 而不是 "../../model" —— 与 SampleMTCNN 等一致。
// 2026-10-04 实测：第一版两边都写 ../../model，Windows 侧一个模型都没找到，
// 却报 "MERGE COMPARE OK / rc=0"（跑过 0、跳过 17）—— **典型的"没检查"被当成"通过"**。
#if defined(_WIN32)
#define MODEL_DIR "model"
#else
#define MODEL_DIR "../../model"
#endif

// 融合只是**重排 + 乘一个常数**，float32 下 1e-4 的后向误差已经很宽松。
// 若这一行要调，先问"它到底该有多准"（把阈值调到能容下真缺陷，就等于没有判据）。
static const double BACKWARD_ERR_LIMIT = 1e-4;

// 逐 blob 扫描用的阈值，比判据严 3~4 个数量级。
// 可用环境变量 ZQ_BLOB_LIMIT 覆盖（设 1e-9 可以看逐位差异的分布）。
static double blob_scan_limit()
{
    const char* e = getenv("ZQ_BLOB_LIMIT");
    return e ? atof(e) : 1e-7;
}
#define BLOB_SCAN_LIMIT blob_scan_limit()

// 生产实参（ZQ_CNN_MTCNN.h:109）：merge_bn=true, ignore_small_value=1e-9, merge_prelu=true
static const float PROD_IGNORE_SMALL = 1e-9f;

// 确定性伪随机：两个 net 必须拿到**逐位相同**的输入，
// 否则比的就不是"融合改没改结果"而是"输入变了"。
static float rnd(unsigned& s)
{
    s = s * 1664525u + 1013904223u;
    return (float)((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

// 收集 .zqparams 里所有 top= 的 blob 名，**按文件顺序**（对绝大多数模型就是拓扑序）。
// 逐 blob 比对靠它 —— 因为 ZQ_CNN_Net 没有公开"列出所有 blob 名"的接口。
static std::vector<std::string> collect_blob_names(const std::string& zp)
{
    std::vector<std::string> out;
    FILE* f = fopen(zp.c_str(), "rb");
    if (!f) return out;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        std::string s(line);
        while (!s.empty() && (s[s.size()-1]=='\n' || s[s.size()-1]=='\r')) s.erase(s.size()-1);
        if (s.empty() || s[0]=='#') continue;
        size_t p = s.find("top=");
        if (p == std::string::npos) continue;
        size_t e = p + 4;
        while (e < s.size() && s[e] != ' ' && s[e] != '\t') e++;
        std::string nm = s.substr(p + 4, e - (p + 4));
        if (!nm.empty()) out.push_back(nm);
    }
    fclose(f);
    return out;
}

// 返回**最后一个** top= 该 blob 的层名。
// 为什么不能取第一个：mobilefacenet-v1 的 res4 段里
// res4_block1/2/3/4_conv_dw **四层都写同一个 blob**（共享 skip 路径），
// 所以"这个 blob 的值"取决于**停在哪个写者**。
// 2026-10-04 实测：我第一版用"停在 block1"的值当分母，
// 而那是**另一个卷积**的输出 —— 拿它算比值，结论直接作废
// （详见附录 HJ.3）。
static std::string last_writer_of(const std::string& zp, const std::string& blob)
{
    FILE* f = fopen(zp.c_str(), "rb");
    if (!f) return "";
    char line[4096];
    std::string best;
    while (fgets(line, sizeof(line), f)) {
        std::string s(line);
        while (!s.empty() && (s[s.size()-1]=='\n' || s[s.size()-1]=='\n')) s.erase(s.size()-1);
        if (s.empty() || s[0]=='#') continue;
        size_t n = s.find("name=");
        size_t t = s.find("top=");
        if (n == std::string::npos || t == std::string::npos) continue;
        if (n > t) continue;                       // name 必须写在 top 前面
        size_t ne = n + 5; while (ne < s.size() && s[ne] != ' ' && s[ne] != '	') ne++;
        size_t te = t + 4; while (te < s.size() && s[te] != ' ' && s[te] != '	') te++;
        if (s.substr(t + 4, te - (t + 4)) == blob) best = s.substr(n + 5, ne - (n + 5));
    }
    fclose(f);
    return best;
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
        if (s.compare(0, 5, "Input") == 0) {
            sscanf(s.c_str(), "Input name=%*s C=%d H=%d W=%d", &C, &H, &W);
        }
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

static void fill_input(ZQ::ZQ_CNN_Tensor4D_NHW_C_Align256bit& t,
                       int C, int H, int W, unsigned seed)
{
    unsigned s = seed;
    int n = C * H * W;
    std::vector<float> v((size_t)n);
    for (int i = 0; i < n; i++) v[(size_t)i] = rnd(s);
    t.ChangeSize(1, H, W, C, 0, 0);
    t.ConvertFromCompactNCHW(&v[0], 1, C, H, W);
}

static void read_blob(const ZQ::ZQ_CNN_Tensor4D* b, std::vector<float>& out)
{
    int N = b->GetN(), C = b->GetC(), H = b->GetH(), W = b->GetW();
    out.resize((size_t)N * C * H * W);
    b->ConvertToCompactNCHW(&out[0]);
}

static double backward_err(const std::vector<float>& a, const std::vector<float>& b, long& worst_i)
{
    if (a.size() != b.size() || a.empty()) { worst_i = -1; return 1e30; }
    double ss = 0.0;
    for (size_t i = 0; i < a.size(); i++) ss += (double)b[i] * (double)b[i];
    double den = sqrt(ss);
    if (den == 0.0) den = 1.0;
    double worst = 0.0;
    worst_i = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double e = fabs((double)a[i] - (double)b[i]) / den;
        if (e > worst) { worst = e; worst_i = (long)i; }
    }
    return worst;
}

// 随仓的模型名硬编码在这里，而不是扫目录 ——
// 扫目录要用 <dirent.h> / access()，那是 POSIX 的，而这个 sample
// **双平台都要编**（AGENTS.md「不要依赖 MSVC 的传递包含」那条的同源问题：
// 依赖一个只有一侧有的头，症状是"Linux 编得过、Windows 编不过"）。
// 存在性用 fopen 试，跨平台。
static const char* MODELS[] = {
    "det1-dw20-fast", "det1-dw20-plus",
    "det2-dw24-fast", "det2-dw24-p0",   "det2-dw24-plus",
    "det3-dw48-fast", "det3-dw48-p0",   "det3-dw48-plus",
    "det4-dw48-v2n",  "det4-dw48-v2s",  "det4-dw64-v3s",
    "det5-dw64-v3s",  "det5-dw96-v2s",  "det5-dw96-v2t",  "det5-dw96-v3s",
    "det5-dw112",
    "mobilefacenet-v1",
};

static bool file_exists(const std::string& p)
{
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("融合 vs 不融合 前向对照（附录 HE）\n");
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
            printf("%-34s SKIP (仓库里没有配套的 .zqparams/.nchwbin)\n", base);
            skip++;
            continue;
        }
        int C, H, W;
        std::string top;
        if (!parse_param(zp, C, H, W, top)) {
            printf("%-34s SKIP (取不到 Input 形状或最后一层的 top)\n", base);
            skip++;
            continue;
        }
        // 基线：默认参数（merge_bn=false, merge_prelu=false）
        ZQ::ZQ_CNN_Net nA;
        if (!nA.LoadFrom(zp, mp)) {
            printf("%-34s BAD  不融合那条加载失败\n", base);
            bad++;
            continue;
        }
        ZQ::ZQ_CNN_Tensor4D_NHW_C_Align256bit inA;
        fill_input(inA, C, H, W, 12345u);
        if (!nA.Forward(inA)) {
            printf("%-34s BAD  不融合那条 Forward 失败\n", base);
            bad++;
            continue;
        }
        const ZQ::ZQ_CNN_Tensor4D* oa = nA.GetBlobByName(top);
        if (oa == 0) {
            printf("%-34s BAD  基线取不到输出 blob \"%s\"\n", base, top.c_str());
            bad++;
            continue;
        }
        std::vector<float> va;
        read_blob(oa, va);
        // 三种融合组合**分别**跑一遍，判据只看生产那一档。
        // 分开跑是为了能**定位到具体是哪一个 merge 改坏了结果** ——
        // 2026-10-04 实测：mobilefacenet-v1 在生产实参下后向误差 0.37，
        // 而只报"融合 vs 不融合"的话只知道"坏了"，不知道"哪个 merge 坏的"。
        // 组合表：
        //   (bn=0, prelu=0) 基线
        //   (bn=1, prelu=0) 只融 BN
        //   (bn=0, prelu=1) 只融 PReLU
        //   (bn=1, prelu=1) 生产实参
        static const bool CFG[3][2] = { { true, false }, { false, true }, { true, true } };
        static const char* CFGN[3] = { "bn only", "prelu only", "bn+prelu (prod)" };
        double errs[3];
        int ngot = 0;
        // 生产那一档（ci == 2）的输出留下来，供下面的"差异结构"用
        std::vector<float> vp;
        for (int ci = 0; ci < 3; ci++) {
            ZQ::ZQ_CNN_Net nX;
            if (!nX.LoadFrom(zp, mp, CFG[ci][0], PROD_IGNORE_SMALL, CFG[ci][1])) {
                errs[ci] = -1.0;
                continue;
            }
            ZQ::ZQ_CNN_Tensor4D_NHW_C_Align256bit inX;
            fill_input(inX, C, H, W, 12345u);
            if (!nX.Forward(inX)) { errs[ci] = -1.0; continue; }
            const ZQ::ZQ_CNN_Tensor4D* ox = nX.GetBlobByName(top);
            if (ox == 0) { errs[ci] = -1.0; continue; }
            std::vector<float> vx;
            read_blob(ox, vx);
            long wi = -1;
            errs[ci] = backward_err(va, vx, wi);
            if (ci == 2) vp = vx;
            ngot++;
        }
        if (ngot < 3) {
            printf("%-34s BAD  三种组合里有 %d 种加载/Forward/取 blob 失败\n", base, 3 - ngot);
            bad++;
            continue;
        }
        double prod = errs[2];
        // 定位用的一行：哪一档先坏
        const char* culprit = "无（单看最后一档）";
        if (prod > BACKWARD_ERR_LIMIT) {
            if (errs[0] > BACKWARD_ERR_LIMIT) culprit = "merge_bn";
            else if (errs[1] > BACKWARD_ERR_LIMIT) culprit = "merge_prelu";
            else culprit = "两者之一（需再细分）";
        }
        if (prod > BACKWARD_ERR_LIMIT) {
            printf("%-34s BAD  后向误差 %.4g > %g（bn only %.4g / prelu only %.4g）"
                   " -> %s 是元凶，top=%s\n", base, prod, BACKWARD_ERR_LIMIT,
                   errs[0], errs[1], culprit, top.c_str());
            // 差异**结构**：全对 / 只错一部分 / 按同一比例错。
            // 这三种形态对应完全不同的根因，所以要打出来而不是只报一个最大值
            // （AGENTS.md「一个最差格不能用来概括整体」）。
            {
                double ss = 0.0;
                for (size_t q = 0; q < va.size(); q++) ss += (double)va[q] * (double)va[q];
                double den = sqrt(ss);
                if (den == 0.0) den = 1.0;
                long ndiff_big = 0, shown = 0;
                double worst_ratio = 0.0;
                int worst_ratio_i = -1;
                printf("      输出 %zu 个，|diff| > 1e-3*||exp|| 的有：", va.size());
                for (size_t q = 0; q < va.size(); q++) {
                    double d = fabs((double)va[q] - (double)vp[q]) / den;
                    if (d <= 1e-3) continue;
                    ndiff_big++;
                    if (fabs(va[q]) > 1e-8) {
                        double rr = (double)vp[q] / (double)va[q];
                        if (fabs(rr) > fabs(worst_ratio)) { worst_ratio = rr; worst_ratio_i = (int)q; }
                    }
                    if (shown < 8) { printf(" [#%d %.6g->%.6g]", (int)q, va[q], vp[q]); shown++; }
                }
                printf("\n      差异大的 = %ld / %zu", ndiff_big, va.size());
                if (worst_ratio_i >= 0) printf("；比值偏离最大的 #%d 比值 %.6g", worst_ratio_i, worst_ratio);
                printf("\n");
            }
            // ---- 逐 blob 扫描：定位**第一个**分歧的 blob ----
            //
            // 为什么必须扫：只比最后一个 blob 的话，误差经过后面几十层传播，
            // 已经看不出"是哪一层开始错的"。逐个 blob 比就能直接指出
            // "第一个对不上的 blob 是哪个" —— 那就是出问题的那一层的输出。
            //
            // 判据：同一个 blob 名字在两条路上的内容，按后向误差比。
            // **形状不同**也算分歧（那说明融合改了张量形状，比数值错更严重）。
            {
                ZQ::ZQ_CNN_Net nP;
                if (nP.LoadFrom(zp, mp, true, PROD_IGNORE_SMALL, true)) {
                    ZQ::ZQ_CNN_Tensor4D_NHW_C_Align256bit inP;
                    fill_input(inP, C, H, W, 12345u);
                    if (nP.Forward(inP)) {
                        std::vector<std::string> names = collect_blob_names(zp);
                        int nbad_blob = 0, first_bad_blob = -1;
                        std::string first_name;
                        for (size_t q = 0; q < names.size(); q++) {
                            const ZQ::ZQ_CNN_Tensor4D* ba = nA.GetBlobByName(names[q]);
                            const ZQ::ZQ_CNN_Tensor4D* bp = nP.GetBlobByName(names[q]);
                            if (ba == 0 || bp == 0) continue;   // 融合后被删掉的 blob，跳过
                            std::vector<float> fa_, fp_;
                            read_blob(ba, fa_);
                            read_blob(bp, fp_);
                            if (fa_.size() != fp_.size()) {
                                if (first_bad_blob < 0) { first_bad_blob = (int)q; first_name = names[q]; }
                                nbad_blob++;
                                continue;
                            }
                            long wi = -1;
                            double e = backward_err(fa_, fp_, wi);
                            // 逐 blob 扫描用**比判据严得多**的阈值（附录 HM.4）：
                            // 判据是 1e-4，而这里用 BLOB_SCAN_LIMIT。
                            // 理由：1e-4 下的"正确"只说明"差得小"，
                            // 而后面几十层里一次小的偏差可能被放大 ——
                            // 所以要问的是"前 70 个 blob 里有没有**已经**不一样的"。
                            if (e > BLOB_SCAN_LIMIT) {
                                if (first_bad_blob < 0) { first_bad_blob = (int)q; first_name = names[q]; }
                                nbad_blob++;
                                if (nbad_blob <= 60) {
                                    printf("      blob[%2d] %-28s 后向误差 %.4g（%zu 个 float）\n",
                                           (int)q, names[q].c_str(), e, fa_.size());
                                }
                                // 第一个分歧的 blob：**逐元素**摆出前几个。
                                // 形态决定根因：整体错 / 只有个别位置错 /
                                // 第 i 个位置的值等于别处的未融合值（= 错位）。
                                if (q == (size_t)first_bad_blob && fp_.size() == fa_.size()) {
                                    double ss = 0.0;
                                    for (size_t z = 0; z < fa_.size(); z++) ss += (double)fa_[z] * (double)fa_[z];
                                    double dn = sqrt(ss);
                                    if (dn == 0.0) dn = 1.0;
                                    int nbad = 0;
                                    for (size_t z = 0; z < fa_.size(); z++)
                                        if (fabs((double)fa_[z] - (double)fp_[z]) / dn > 1e-3) nbad++;
                                    printf("        形状 N=%d H=%d W=%d C=%d，共 %zu 个 float，差异大的 %d 个\n",
                                           ba->GetN(), ba->GetH(), ba->GetW(), ba->GetC(), fa_.size(), nbad);
                                    printf("        前 10 个（未融合 -> 融合）：");
                                    for (int z = 0; z < 10 && (size_t)z < fa_.size(); z++)
                                        printf(" [%d %.6g->%.6g]", z, fa_[z], fp_[z]);
                                    printf("\n");
                                    printf("        前 10 个（未融合 -> 融合）：");
                                    for (int z = 0; z < 10 && (size_t)z < fa_.size(); z++)
                                        printf(" [%d %.6g->%.6g]", z, fa_[z], fp_[z]);
                                    printf("\n");
                                    // ---- 逐**通道**（附录 HP.1）----
                                    //
                                    // compact NCHW  (n,c,h,w) 的下标是
                                    //     (c*H + h)*W + w
                                    // 所以通道 c 的元素是**等距**的，每隔 H*W。
                                    //
                                    // 这一层切分把两种根因分开：
                                    //   * **只有部分通道**超阈值 => 映射/接线问题
                                    //     （某几个通道拿到了别人的系数，或某一层没被写）；
                                    //   * **全部通道**都超阈值 => 这一层**整体**算错
                                    //     （读错了输入、或权重整体不对）。
                                    // 两种指向完全不同的修法，一次测量就能分开。
                                    {
                                        const int NN2 = ba->GetN(), HH2 = ba->GetH(),
                                                  WW2 = ba->GetW(), CC2 = ba->GetC();
                                        const long HW2 = (long)HH2 * WW2;
                                        int nbad_ch = 0, worst_ch = -1;
                                        double worst_ch_err = 0;
                                        int printed = 0;
                                        for (int c = 0; c < CC2; c++) {
                                            // 这个通道自己的尺度（分母用未融合那一侧）
                                            double ss = 0.0;
                                            for (long k = 0; k < NN2 * HW2; k++) {
                                                double v = fa_[(size_t)k * CC2 + c];
                                                ss += v * v;
                                            }
                                            double den = sqrt(ss);
                                            if (den == 0.0) den = 1.0;
                                            double wc = 0.0;
                                            for (long k = 0; k < NN2 * HW2; k++) {
                                                size_t z = (size_t)k * CC2 + c;
                                                double e = fabs((double)fa_[z] - (double)fp_[z]) / den;
                                                if (e > wc) wc = e;
                                            }
                                            if (wc > 1e-3) {
                                                nbad_ch++;
                                                if (wc > worst_ch_err) { worst_ch_err = wc; worst_ch = c; }
                                                if (printed < 6) {
                                                    printf("        通道 %4d 后向误差 %.4g\n", c, wc);
                                                    printed++;
                                                }
                                            }
                                        }
                                        printf("        逐通道：%d / %d 个通道超 1e-3；最差通道 #%d 误差 %.4g\n",
                                               nbad_ch, CC2, worst_ch, worst_ch_err);
                                        printf("        => %s\n",
                                               nbad_ch == 0 ? "全部通道都在阈值内"
                                           : (nbad_ch * 2 < CC2
                                              ? "**只有少数通道**超阈值 => 映射/接线问题"
                                              : "**多数通道**都超阈值 => 这一层整体算错（读错输入/权重整体不对）"));
                                    }

                                    // ---- 决定性的那一次测量（附录 HJ.2）----
                                    //
                                    // 融合把每个输出通道 c 的权重整体乘上 `b[c]`，
                                    // 所以**融合后的输出必然等于 "BN 之前的输出 × 逐通道常数"**：
                                    // 同一个通道 c 内，所有 (h,w) 位置上的
                                    //     ratio = 融合值 / BN 前的值
                                    // 应当是**同一个常数**。
                                    //
                                    // 于是这一个比值就把两类根因分开：
                                    //   * 通道内 ratio **恒定** => 这一层的计算是自洽的，
                                    //     错的是"那个常数取错了"（系数/映射）；
                                    //   * 通道内 ratio **乱跳** => 融合后的这一层
                                    //     **不是** "同一批权重 × 同一份输入" 算出来的，
                                    //     也就是它读到的输入或权重根本不是那一份。
                                    //
                                    // "BN 之前的值"用**公开的局部前向**取：
                                    //   Forward(in, "data", <dwconv 层名>)
                                    // 起点是 Input 层（名字就是 .zqparams 里的 name=data），
                                    // 所以它会把 dwconv 跑完并**停在那里**，
                                    // 而 dwconv 的 BN 层还没跑 ——
                                    // 这正是 BN 之前的值。
                                    {
                                        const int NN = ba->GetN(), HH = ba->GetH(),
                                                  WW = ba->GetW(), CC = ba->GetC();
                                        // 找出产出这个 blob 的那一层：从 blob 名反查
                                        // 用不到内部结构，所以直接用 .zqparams 的层顺序：
                                        // 第一个 bottom/top 命中该 blob 的 DepthwiseConvolution。
                                        ZQ::ZQ_CNN_Net nPre;
                                        if (nPre.LoadFrom(zp, mp)) {
                                            ZQ::ZQ_CNN_Tensor4D_NHW_C_Align256bit inPre;
                                            fill_input(inPre, C, H, W, 12345u);
                                            std::string lastw = last_writer_of(zp, first_name);
                                            printf("        [该 blob 的最后写者是 %s]\n",
                                                   lastw.empty() ? "?" : lastw.c_str());
                                            if (!lastw.empty() && nPre.Forward(inPre, "data", lastw)) {
                                                const ZQ::ZQ_CNN_Tensor4D* bpre =
                                                    nPre.GetBlobByName(first_name);
                                                if (bpre != 0 && bpre->GetC() == CC) {
                                                    std::vector<float> pre;
                                                    read_blob(bpre, pre);
                                                    if (pre.size() == fp_.size()) {
                                                        // 每个通道内 ratio 的极差
                                                        double worst_spread = 0.0;
                                                        int worst_ch = -1;
                                                        int nconst = 0;
                                                        for (int c = 0; c < CC; c++) {
                                                            double lo = 0, hi = 0;
                                                            int cnt = 0;
                                                            for (int k = 0; k < NN * HH * WW; k++) {
                                                                size_t z = (size_t)k * CC + c;
                                                                if (fabs(pre[z]) < 1e-6f) continue;
                                                                double r = (double)fp_[z] / (double)pre[z];
                                                                if (cnt == 0) { lo = hi = r; }
                                                                else { if (r < lo) lo = r; if (r > hi) hi = r; }
                                                                cnt++;
                                                            }
                                                            if (cnt < 2) continue;
                                                            double spread = (hi - lo) / (fabs(hi) + 1e-12);
                                                            if (spread < 1e-3) nconst++;
                                                            if (spread > worst_spread) { worst_spread = spread; worst_ch = c; }
                                                        }
                                                        printf("        通道内 ratio 恒定(<1e-3 极差)的通道: %d / %d\n",
                                                               nconst, CC);
                                                        printf("        极差最大的通道 #%d，相对极差 %.4g\n",
                                                               worst_ch, worst_spread);
                                                        printf("        => %s\n",
                                                               nconst > CC / 2
                                                                ? "这一层的计算是自洽的，错的是那个逐通道系数（系数/映射）"
                                                                : "融合后的这一层**不是**同一批权重×同一份输入算出来的（读错了输入或权重）");
                                                        // 第二个决定性测量：这个 blob 被
                                                        // res4_block1/2/3/4 **四个 dwconv 写**（共享 skip）。
                                                        // 所以"最终内容"正常应该是**最后一个写者**的。
                                                        // 若"融合后的最终值"等于"只跑完 block1 就停的值"，
                                                        // 就说明融合后的网**没有再让 block2/3/4 写它** ——
                                                        // 即某一层被漏掉了，而不是某一层算错了。
                                                        {
                                                            ZQ::ZQ_CNN_Net nB1;
                                                            if (nB1.LoadFrom(zp, mp)) {
                                                                ZQ::ZQ_CNN_Tensor4D_NHW_C_Align256bit inB1;
                                                                fill_input(inB1, C, H, W, 12345u);
                                                                if (!lastw.empty() && nB1.Forward(inB1, "data", lastw)) {
                                                                    const ZQ::ZQ_CNN_Tensor4D* b1 =
                                                                        nB1.GetBlobByName(first_name);
                                                                    if (b1 != 0) {
                                                                        std::vector<float> v1;
                                                                        read_blob(b1, v1);
                                                                        if (v1.size() == fp_.size()) {
                                                                            long wi1 = -1;
                                                                            double e1 = backward_err(v1, fp_, wi1);
                                                                            printf("        对照：只跑完 %s 的 dw+BN 就停，其值 vs 融合后最终值，后向误差 %.4g%s\n",
                                                                                   first_name.c_str(), e1,
                                                                                   e1 < 1e-4 ? "  <== **完全相同** => 融合后的网漏写了后面几个 dwconv" : "");
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        printf("      逐 blob 扫描：%zu 个 blob 里 %d 个对不上；**第一个**是 #%d \"%s\"\n",
                               names.size(), nbad_blob, first_bad_blob, first_name.c_str());
                    } else {
                        printf("      逐 blob 扫描：生产那一档 Forward 失败\n");
                    }
                } else {
                    printf("      逐 blob 扫描：生产那一档加载失败\n");
                }
            }
            bad++;
        } else {
            printf("%-34s OK   后向误差 %.4g（bn only %.4g / prelu only %.4g），"
                   "输出 %zu 个 float，top=%s\n", base, prod, errs[0], errs[1], va.size(), top.c_str());
            ok++;
        }
    }
    printf("\n共 %zu 个模型：跑过 %d（通过 %d，超阈值 %d），跳过 %d\n",
           nmodels, ok + bad, ok, bad, skip);
    // **一个都没跑过 = 失败**，不是通过。
    // 2026-10-04 实测踩到：Windows 侧的 MODEL_DIR 写错，一个模型都没找到，
    // 于是 `bad == 0` 成立、报 "MERGE COMPARE OK"、rc=0 ——
    // 而"没检查"与"检查通过"在输出上一模一样。
    // 这与本文件「没有找到 .zqparams —— 同样是"没检查"，不是通过」是同一条，
    // 而我第一版在门禁里写了、搬成 sample 时**漏掉了**。
    if (ok + bad == 0) {
        printf("MERGE COMPARE FAILED（一个模型都没跑过 —— 这不是通过）\n");
        return 1;
    }
    printf("%s\n", bad == 0 ? "MERGE COMPARE OK" : "MERGE COMPARE FAILED");
    return bad == 0 ? 0 : 1;
}
