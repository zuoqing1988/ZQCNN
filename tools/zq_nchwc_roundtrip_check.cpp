// NCHWC 那条加载/回存路径的往返门禁（附录 HC）。
//
// 缺口：`zq_weight_tail` / `zq_weight_roundtrip` / `zq_loadbuffer` 三道门禁
// **全部只跑 `ZQ_CNN_Net`（NCHW 那一侧）**。而 `ZQ_CNN_Net_NCHWC<Tensor4D>`
// 是**一份独立的模板实现**，它有自己的一套 `LoadBinary_NCHW` /
// `SaveBinary_NCHW`（`ZQ_CNN_Layer_NCHWC.h` 里 11 处）与自己的 `_prepack()`。
// 它的往返**零覆盖**。
//
// 为什么这值得单独一道：随仓 27 个 `.zqparams` 里有 **20 个**只用了
// NCHWC 认的那 10 种层类型，也就是说 NCHWC 侧**能加载 20 个模型**，
// 而生产里只跑了 4 个（`SampleMTCNN_NCHWC4` 用 det1/det2/det3/det5-dw64-v3s）。
// 剩下 16 个能不能加载、加载后回存还对不对，**没人知道**。
//
// 量的三件事（与 NCHW 侧同口径，好并排看）：
//   (A) LoadFrom -> SaveModel 的长度必须等于 .nchwbin 长度
//   (B) 内容必须与原文件逐字节相同；不同的那些输入值必须**全部**满足
//       fabs(v) < 1e-12（NCHW 侧在 HB 里量过是 20 个逐字节相同、
//       3 个可由该阈值解释；NCHWC 侧第一次量）
//   (C) load->save->load->save 必须幂等
//
// 顺带产出一条**信息**（不判失败）：哪些模型 NCHWC **根本加载不了**。
// NCHWC 只认 10 种层类型（NCHW 认 36 种），所以"加载不了"里
// 既有"合法地含了 NCHWC 不支持的层"，也有"真出了毛病"——
// 这两类必须分开报，不能混成一句"失败"。
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <dirent.h>
#include "ZQCNN/ZQ_CNN_Tensor4D.h"
#include "ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h"
#include "ZQCNN/ZQ_CNN_Net_NCHWC.h"
#include "zq_net_fwd_tripwires.h"
#include "zq_concat_getsize_real.h"

#define MODEL_DIR "/mnt/d/ZQCNN/model"

static int g_ok = 0, g_bad = 0, g_skip = 0, g_same = 0, g_diff = 0, g_unsupported = 0;

// `zq_net_fwd_tripwires.h` 对「加载期**合法**被调用」的那一族发的是**记录型**桩：
// 打一行 `LOADTIME: <名>` 到 stdout 然后返回 true。
// 这里数它 —— **必须断言它出现过**，否则那个桩就是个静默 no-op，
// 而"Prepack 这一步被悄悄跳过"正是本文件头坚持要防的那类假绿。
// 计数由 zq_net_fwd_tripwires.h 里的 zq_net_loadtime_calls() 提供
#define ZQ_LOADTIME_CALLS() zq_net_loadtime_calls()

struct DiffStat {
    long ndiff, nbad, first_bad;
    float bad_a, bad_b;
    long na, nb;
};

static DiffStat cmp_files(const std::string& a, const char* b, double thresh)
{
    DiffStat st;
    st.ndiff = 0; st.nbad = 0; st.first_bad = -1;
    st.bad_a = st.bad_b = 0.f;
    st.na = st.nb = 0;
    FILE* fa = fopen(a.c_str(), "rb");
    FILE* fb = fopen(b, "rb");
    if (!fa || !fb) {
        if (fa) fclose(fa);
        if (fb) fclose(fb);
        st.nbad = 1; st.first_bad = -2;
        return st;
    }
    fseek(fa, 0, SEEK_END); long na = ftell(fa);
    fseek(fb, 0, SEEK_END); long nb = ftell(fb);
    st.na = na; st.nb = nb;
    if (na != nb) { fclose(fa); fclose(fb); st.nbad = 1; st.first_bad = -2; return st; }
    rewind(fa); rewind(fb);
    std::vector<char> ba((size_t)na), bb((size_t)nb);
    if (na > 0) {
        if (fread(&ba[0], 1, (size_t)na, fa) != (size_t)na) { fclose(fa); fclose(fb); st.nbad = 1; st.first_bad = -2; return st; }
        if (fread(&bb[0], 1, (size_t)nb, fb) != (size_t)nb) { fclose(fa); fclose(fb); st.nbad = 1; st.first_bad = -2; return st; }
    }
    fclose(fa); fclose(fb);
    long nflo = na / 4;
    int* pa = (int*)&ba[0];
    int* pb = (int*)&bb[0];
    float* va_ = (float*)&ba[0];
    for (long i = 0; i < nflo; i++) {
        if (pa[i] == pb[i]) continue;
        st.ndiff++;
        float va = va_[i];
        if (fabs(va) >= thresh) {
            st.nbad++;
            if (st.first_bad < 0) {
                st.first_bad = i;
                st.bad_a = va;
                memcpy(&st.bad_b, &pb[i], 4);
            }
        }
    }
    return st;
}

static void one(const char* base, const std::string& zp, const std::string& mp)
{
    char rt1[512], rt2[512];
    snprintf(rt1, sizeof(rt1), "/tmp/zq_nchwc_rt1_%s.nchwbin", base);
    snprintf(rt2, sizeof(rt2), "/tmp/zq_nchwc_rt2_%s.nchwbin", base);

    ZQ::ZQ_CNN_Net_NCHWC<ZQ::ZQ_CNN_Tensor4D_NCHWC4> n1;
    // 本模型 .zqparams 里有没有 InnerProduct 层 —— 决定要不要期望 LOADTIME 行
    FILE* pf = fopen(zp.c_str(), "rb");
    bool has_ip = false;
    if (pf) {
        char line[1024];
        while (fgets(line, sizeof(line), pf)) {
            if (strncmp(line, "InnerProduct", 12) == 0) { has_ip = true; break; }
        }
        fclose(pf);
    }
    int loadtime_before = ZQ_LOADTIME_CALLS();
    // **先打模型名再 LoadFrom**。LoadFrom 会在 stdout 上刷一串
    // `warning: unknown para …`，而如果模型名打在它**之后**，
    // 那些警告读起来就属于**上一个**模型 ——
    // 2026-10-04 实测：我据此得出"det4-dw64-v3s / det5-dw96-v3s 加载成功却带 pad_type
    // 警告"，而那两个文件里**根本没有 pad_type**（只有 Pose-zq / det5-112-gray /
    // headposegaze 有）。按名字归属到下一个模型之后，结论反过来了。
    // 这与 AGENTS.md「通用工具的输出里不要写死某一次的具体描述」是同一条：
    // 输出要**自带它作用于谁**，且归属要无歧义。
    printf(">>> %s\n", base);
    fflush(NULL);
    if (!n1.LoadFrom(zp, mp)) {
        // 分不清是"合法地含了 NCHWC 不支持的层"还是"真出了毛病"，
        // 统一记成"不支持"，并在汇总里报出层名需要人工看。
        // **不判失败** —— 判据是"能加载的那些往返必须对"，
        // 不是"所有模型都必须能加载"。
        g_unsupported++;
        printf("  %-34s NCHWC 加载不了（可能含它不认的层类型，见下方汇总）\n", base);
        return;
    }
    if (!n1.SaveModel(rt1)) {
        printf("  %-34s **FAIL** 回存 1 失败\n", base);
        g_bad++;
        return;
    }
    // 含 InnerProduct 层的模型，NCHWC 的 LoadFrom 一定会走 `_prepack()` ->
    // `Prepack()` -> `InnerProductPrePack`。那个符号在本门禁里是**记录型桩**
    // （不中止、只打一行 LOADTIME），所以**必须在这里断言它响过** ——
    // 否则桩就是个静默 no-op，而"Prepack 被悄悄跳过"没人会发现。
    if (has_ip && ZQ_LOADTIME_CALLS() == loadtime_before) {
        printf("  %-34s **FAIL** .zqparams 里有 InnerProduct 层，但一次 LOADTIME 都没有"
               " —— Prepack 那一步没走到（记录型桩被静默跳过了？）\n", base);
        g_bad++;
        remove(rt1); remove(rt2);
        return;
    }
    ZQ::ZQ_CNN_Net_NCHWC<ZQ::ZQ_CNN_Tensor4D_NCHWC4> n2;
    if (!n2.LoadFrom(zp, rt1)) {
        printf("  %-34s **FAIL** 回存出来的文件 NCHWC 加载不了（**回存不可再入**）\n", base);
        g_bad++;
        remove(rt1); remove(rt2);
        return;
    }
    if (!n2.SaveModel(rt2)) {
        printf("  %-34s **FAIL** 回存 2 失败\n", base);
        g_bad++;
        remove(rt1); remove(rt2);
        return;
    }

    int bad = 0;
    DiffStat da = cmp_files(mp, rt1, 1e-12);
    if (da.nbad != 0) {
        if (da.first_bad == -2) {
            printf("  %-34s **FAIL** (A) 长度对不上：原文件 %ld 字节 / NCHWC 回存 %ld 字节"
                   "（差 %ld）\n", base, da.na, da.nb, da.nb - da.na);
        } else {
            printf("  %-34s **FAIL** (A) 有 %ld 个 float 对不上且**无法**用 ignore_small_value"
                   " 解释，首个在 #%ld（%.9g vs %.9g）\n",
                   base, da.nbad, da.first_bad, da.bad_a, da.bad_b);
        }
        bad++;
    } else if (da.ndiff == 0) {
        g_same++;
    } else {
        g_diff++;
        printf("  %-34s (A) 有 %ld 个 float 与原文件不同，**全部** fabs(v) < 1e-12\n",
               base, da.ndiff);
    }
    DiffStat db = cmp_files(rt1, rt2, 0.0);
    if (db.nbad != 0) {
        printf("  %-34s **FAIL** (C) load->save->load->save 不幂等：%ld 个 float 对不上，"
               "首个在 #%ld（%.9g vs %.9g）\n",
               base, db.nbad, db.first_bad, db.bad_a, db.bad_b);
        bad++;
    }
    if (bad == 0) {
        g_ok++;
        printf("  %-34s OK  (A) %s；(C) 幂等\n", base,
               da.ndiff == 0 ? "与原文件逐字节相同" : "差异可由 ignore_small_value 解释");
    } else {
        g_bad += bad;
    }
    remove(rt1);
    remove(rt2);
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC 权重往返门禁（附录 HC）\n");
    printf("判据：对 NCHWC **能加载**的模型，LoadFrom->SaveModel 的长度与内容\n");
    printf("      必须与 .nchwbin 一致（内容差异只允许 fabs(v) < 1e-12 的清零），\n");
    printf("      且 load->save->load->save 幂等。\n");
    printf("不判失败：NCHWC 加载不了的模型（NCHWC 只认 10 种层类型，NCHW 认 36 种）——\n");
    printf("      但要**报出来**，不能混进「通过」里。\n\n");

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
            g_skip++;
            continue;
        }
        one(base, zp, mp);
    }
    printf("\n共 %d 个 .zqparams：NCHWC 加载成功 %d"
           "（往返正确 %d，被拒 %d），加载不了 %d，无配套权重跳过 %d\n",
           (int)params.size(), g_ok + g_bad, g_ok, g_bad, g_unsupported, g_skip);
    printf("内容分类：与原文件**逐字节相同** %d 个；"
           "有差异但**全部**可由 ignore_small_value 解释 %d 个\n", g_same, g_diff);
    return g_bad == 0 ? 0 : 1;
}
