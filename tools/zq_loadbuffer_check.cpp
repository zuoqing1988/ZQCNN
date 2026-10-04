// `LoadFromBuffer` 路径的行为门禁（附录 HA）。
//
// 为什么要有这道门禁
// ------------------
// `ZQ_CNN_Net` 有三条加载路径，`LoadFrom`（文件）那道有两道行为门禁
// （`zq_concat_alias` / `zq_model_params` / `zq_weight_tail`），
// 而 **`LoadFromBuffer` 一道都没有**：36 个 `LoadBinary_NCHW(buffer,…)` 重载
// 全在门禁覆盖之外，只有 `SampleMTCNNLoadFromCode` 一个 Linux sample 顺带跑到过。
//
// 它与文件路径**不等价**，而且差在两处：
//   1. 文件路径的"字节不够"靠 `fread_s` 的短读检测；buffer 路径靠
//      `LoadBinary_NCHW` 自己判 `buffer_len`。**漏判一个，就是堆越界读。**
//   2. `const char*&` 形参看着像会推进调用方的指针，实际全链都按值传
//      （附录 HA.3）—— 这条"安全"依赖一个没有任何注释的约定。
//
// 本门禁量三件事：
//   (A) 随仓 23 对模型走 `LoadFromBuffer` 必须**全部成功** —— 从"零覆盖"变成
//       "有基线"。
//   (B) 把权重 buffer **截短**必须让 `LoadFromBuffer` 返回 false，
//       而不是读越界。**这条是真正的安全属性**：ASan 会在真读过界时报出来，
//       所以判据不必自己去测"有没有读越界"，只要"该失败时没失败"就会红。
//   (C) `ZQ_CNN_Layer_Scale::LoadBinary_NCHW` 单独打一遍 —— 它是
//       36 个 buffer 重载里**唯一**一个不查 `buffer_len` 的（附录 HA.1），
//       而随仓 27 个 `.zqparams` 一个 `Scale` 层都没有，所以 (A)/(B) 永远碰不到它。
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dirent.h>
#include "ZQCNN/ZQ_CNN_Tensor4D.h"
#include "ZQCNN/ZQ_CNN_Net.h"
#include "ZQCNN/ZQ_CNN_Layer.h"
// 链接期需要的那批 ZQ_CNN_Forward_SSEUtils 符号，用现成的**绊线桩**顶上
// （与 zq_model_params_check.cpp 同一套）。
#include "zq_net_fwd_tripwires.h"
#include "zq_concat_getsize_real.h"

#define MODEL_DIR "/mnt/d/ZQCNN/model"

// Normalize / Scale 的 scale 张量是 [1,1,1,C]，C 随便取个 64
#define NORM_C 64

static int g_ok = 0, g_bad = 0;
static int skipped = 0;

static bool read_file(const std::string& p, std::vector<char>& out)
{
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return false; }
    out.resize((size_t)n);
    size_t got = fread(&out[0], 1, (size_t)n, f);
    fclose(f);
    return got == (size_t)n;
}

// (A) 完整 buffer 必须加载成功
static void case_full(const char* base, const std::string& zp, const std::string& mp)
{
    std::vector<char> pb, mb;
    if (!read_file(zp, pb) || !read_file(mp, mb)) {
        printf("  %-34s **FAIL** 读文件失败\n", base);
        g_bad++;
        return;
    }
    const char* pp = pb.empty() ? "" : &pb[0];
    const char* mp0 = mb.empty() ? "" : &mb[0];
    ZQ::ZQ_CNN_Net net;
    if (!net.LoadFromBuffer(pp, (__int64)pb.size(), mp0, (__int64)mb.size())) {
        printf("  %-34s **FAIL** 完整 buffer 加载失败（%zu 参数字节 / %zu 权重字节）\n",
               base, pb.size(), mb.size());
        g_bad++;
        return;
    }
    g_ok++;
    printf("  %-34s OK  (A) 完整 buffer 加载成功\n", base);
}

// (B) 权重 buffer 截短 -> 必须失败。
//     截掉的比例扫一串：轻微截断最容易"看起来还在跑"，大幅截断更容易命中别的守卫。
static void case_truncated(const char* base, const std::string& zp, const std::string& mp)
{
    // 越界读就发生在第一个元素上：MobileNetSSD_deploy 的权重少 1 字节
    // -> Convolution 的 bias memcpy 读过界 -> ASan heap-buffer-overflow。
    static const int cuts[] = {1, 4, 16, 64};
    std::vector<char> pb;
    if (!read_file(zp, pb)) { printf("  %-34s **FAIL** 读参数文件失败\n", base); g_bad++; return; }
    std::vector<char> mb;
    if (!read_file(mp, mb)) { printf("  %-34s **FAIL** 读权重文件失败\n", base); g_bad++; return; }
    int bad = 0;
    for (size_t ci = 0; ci < sizeof(cuts) / sizeof(cuts[0]); ci++) {
        size_t cut = (size_t)cuts[ci];
        if (mb.size() <= cut) continue;
        std::vector<char> shortbuf(mb.begin(), mb.end() - cut);
        const char* pp = pb.empty() ? "" : &pb[0];
        const char* m0 = shortbuf.empty() ? "" : &shortbuf[0];
        ZQ::ZQ_CNN_Net net;
        if (net.LoadFromBuffer(pp, (__int64)pb.size(), m0, (__int64)shortbuf.size())) {
            printf("  %-34s **FAIL** 权重少 %d 字节仍然加载成功 —— \"字节不够\"没被检查\n",
                   base, (int)cut);
            bad++;
        }
    }
    if (bad == 0) {
        g_ok++;
        printf("  %-34s OK  (B) 少 1/4/16/64 字节全部被拒\n", base);
    } else {
        g_bad += bad;
    }
}

// (C) 随仓 27 个 .zqparams 里**一次都没出现**、却各自实现了
//     `LoadBinary_NCHW(buffer,…)` 的层，**逐个直接打**。
//
//     为什么必须逐个打：附录 HA.1 的三处缺陷全在这些层里，而
//     随仓模型一个都不含它们 —— 所以 (A)/(B) 那种"拿真模型截断"的测法
//     **结构上**永远碰不到，只能直接构造层对象。
//     （2026-10-04 实测：我第一版把 Normalize 的那处写成了"Scale"，
//       而 `ZQ_CNN_Layer_Scale` 本来两个分支都有守卫 —— 门禁于是对着
//       一个安全类打，全绿。**变异测试立刻抓到了**：把 Normalize 的守卫
//       删掉重跑，门禁仍然 PASS。判据必须能区分"有守卫"和"没守卫"。）
//
//     三个层的"读完整个 buffer 需要多少字节"是各自算出来的，不是抄来的：
//       Normalize     scale = [1,1,1,channel_shared?1:C]  -> C float
//       Scale         scale = [1,1,1,channel_shared?1:C]  -> C float
//       DeConvolution filters = [num_output,kH,kW,bottom_C] + (with_bias)
//                        bias   = [1,1,1,num_output]        -> 两段
struct LayerSpec {
    const char* name;
    ZQ::ZQ_CNN_Layer* (*make)();
    size_t need;
};

static ZQ::ZQ_CNN_Layer* make_normalize()
{
    ZQ::ZQ_CNN_Layer_Normalize* l = new ZQ::ZQ_CNN_Layer_Normalize();
    l->channel_shared = false;                    // 默认就是 false，写出来是为了让读者看见
    if (!l->SetBottomDim(NORM_C, 1, 1)) { delete l; return 0; }
    return l;
}
static ZQ::ZQ_CNN_Layer* make_scale()
{
    ZQ::ZQ_CNN_Layer_Scale* l = new ZQ::ZQ_CNN_Layer_Scale();
    if (!l->SetBottomDim(NORM_C, 1, 1)) { delete l; return 0; }
    return l;
}
static ZQ::ZQ_CNN_Layer* make_deconv()
{
    ZQ::ZQ_CNN_Layer_DeConvolution* l = new ZQ::ZQ_CNN_Layer_DeConvolution();
    l->num_output = 8; l->kernel_H = 3; l->kernel_W = 3; l->with_bias = true;
    if (!l->SetBottomDim(4, 8, 8)) { delete l; return 0; }
    return l;
}

static void case_layers()
{
    // DeConvolution: filters [8][3][3][4] = 288 float, bias [1,1,1][8] = 8 float
    const size_t deconv_need = (size_t)(8 * 3 * 3 * 4 + 8) * sizeof(float);
    LayerSpec specs[] = {
        { "ZQ_CNN_Layer_Normalize",     make_normalize, (size_t)NORM_C * sizeof(float) },
        { "ZQ_CNN_Layer_Scale",         make_scale,     (size_t)NORM_C * sizeof(float) },
        { "ZQ_CNN_Layer_DeConvolution", make_deconv,    deconv_need },
    };
    for (size_t si = 0; si < sizeof(specs) / sizeof(specs[0]); si++) {
        const LayerSpec& sp = specs[si];
        int bad = 0;
        // 正向：给足 need 字节必须成功，且 readed == need。
        // **先跑正向**：守卫写反（该拒却收下）时，正向会先报出来，
        // 不用等到 ASan 那边才有话说。
        {
            ZQ::ZQ_CNN_Layer* l = sp.make();
            if (!l) {
                printf("  %-34s **FAIL** 构造失败\n", sp.name);
                g_bad++;
                continue;
            }
            char* buf = (char*)malloc(sp.need);
            memset(buf, 0, sp.need);
            __int64 readed = 0;
            bool ok = l->LoadBinary_NCHW(buf, (__int64)sp.need, readed);
            if (!ok) {
                printf("  %-34s **FAIL** 字节恰好够（%zu）却返回失败\n", sp.name, sp.need);
                bad++;
            } else if (readed != (__int64)sp.need) {
                printf("  %-34s **FAIL** 字节够时 readed=%lld，应为 %zu\n",
                       sp.name, (long long)readed, sp.need);
                bad++;
            }
            free(buf);
            delete l;
        }
        // 反向：给得太少必须被拒。**短多少要同时从两头量**：
        // 只用 {0,1,4,16,64} 这种"离 0 很近"的长度，只能探到**第一段**
        // （filters / mean / var）的守卫 —— 后面几段一个都碰不到。
        // 2026-10-04 实测：DeConvolution 去掉 bias 那段的守卫后门禁**照样绿**，
        // 因为 filters 段的 1152 字节早就把短 buffer 挡回去了。
        // 真正有鉴别力的是 need-1 / need-4 / need-16 这种**从末尾缺口**的。
        size_t shorts[9];
        shorts[0] = 0; shorts[1] = 1; shorts[2] = 4; shorts[3] = 16; shorts[4] = 64;
        shorts[5] = sp.need - 64; shorts[6] = sp.need - 16;
        shorts[7] = sp.need - 4;  shorts[8] = sp.need - 1;
        for (size_t k = 0; k < sizeof(shorts) / sizeof(shorts[0]); k++) {
            size_t len = shorts[k];
            if (len >= sp.need) continue;
            // 同一段里可能有重复（need 很小的时候），去重
            bool dup = false;
            for (size_t q = 0; q < k; q++) if (shorts[q] == len) dup = true;
            if (dup) continue;
            ZQ::ZQ_CNN_Layer* l = sp.make();
            if (!l) { printf("  %-34s **FAIL** 构造失败\n", sp.name); g_bad++; break; }
            char* buf = (char*)malloc(len ? len : 1);
            memset(buf, 0, len ? len : 1);
            __int64 readed = 0;
            bool ok = l->LoadBinary_NCHW(buf, (__int64)len, readed);
            if (ok) {
                printf("  %-34s **FAIL** buffer 只有 %zu/%zu 字节却返回成功"
                       "（readed=%lld）—— \"字节不够\"没被检查\n",
                       sp.name, len, sp.need, (long long)readed);
                bad++;
            }
            free(buf);
            delete l;
        }
        if (bad == 0) {
            g_ok++;
            printf("  %-34s OK  (C) 给足 %zu 字节成功；短 0/1/4/16/64 与"
                   "末尾缺 64/16/4/1 字节全被拒\n", sp.name, sp.need);
        } else {
            g_bad += bad;
        }
    }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("LoadFromBuffer 路径门禁（附录 HA）\n");
    printf("判据：(A) 23 对随仓模型走 buffer 路径全部成功；\n");
    printf("      (B) 权重少 1/4/16/64 字节必须全部被拒；\n");
    printf("      (C) 随仓模型一次都没用到的 Normalize / Scale / DeConvolution，\n");
    printf("          它们的 buffer 重载必须查 buffer_len。\n\n");

    case_layers();
    printf("\n");

    DIR* d = opendir(MODEL_DIR);
    if (!d) { printf("  打开 %s 失败 —— 门禁不能在没有模型的情况下报通过\n", MODEL_DIR); return 1; }
    std::vector<std::string> params;
    struct dirent* e;
    while ((e = readdir(d)) != NULL) {
        std::string n = e->d_name;
        if (n.size() > 9 && n.compare(n.size() - 9, 9, ".zqparams") == 0)
            params.push_back(std::string(MODEL_DIR) + "/" + n);
    }
    closedir(d);
    if (params.empty()) { printf("  没有找到 .zqparams —— 同样是\"没检查\"，不是通过\n"); return 1; }

    for (size_t i = 0; i < params.size(); i++) {
        std::string zp = params[i];
        std::string mp = zp.substr(0, zp.size() - 9) + ".nchwbin";
        const char* base = strrchr(params[i].c_str(), '/');
        base = base ? base + 1 : params[i].c_str();
        if (access(mp.c_str(), R_OK) != 0) {
            printf("  %-34s 跳过（仓库里没有配套 .nchwbin）\n", base);
            skipped++;
            continue;
        }
        case_full(base, zp, mp);
        case_truncated(base, zp, mp);
    }
    // 汇总行**不许出现字面的 `FAIL`**（AGENTS.md「写检查类工具」第 6 条）：
    // run_zqlib_checks.py 的 ASan 判据是 `grep -cE 'FAIL' <tag>.out`，
    // 写在汇总里会让**一个都没失败**也被判成"1 条断言失败"。
    // 2026-10-04 实测踩了：这一行第一版写的是 "OK %d，FAIL %d"。
    // 真失败由逐项那几行 `**FAIL**` 报，那几行本来就只在真失败时出现。
    printf("\n共 %d 个 .zqparams：跑过 %d 项（通过 %d，被拒 %d），"
           "无配套权重跳过 %d\n",
           (int)params.size(), g_ok + g_bad, g_ok, g_bad, skipped);
    return g_bad == 0 ? 0 : 1;
}
