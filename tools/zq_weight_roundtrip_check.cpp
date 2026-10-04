// 权重**内容**往返契约（附录 HB）。
//
// 缺口：`zq_weight_tail` 证的是「`LoadFrom` 读进来的东西，`SaveModel` 能
// 写成同样**长**」——**只比长度**。而 `SaveBinary_NCHW` 与
// `LoadBinary_NCHW` 是**两份独立实现**：如果它们对布局的理解不一致
// （本仓库历史上最常见的一类错：NCHW 的 `sliceStep` 与 NCHWC 的 `imStep`
// 同名反义，见 AGENTS.md「NCHW 与 NCHWC 的步长语义」），
// 长度照样相等、**内容已经错了**。
//
// 量的三件事，对 23 对随仓模型逐个做：
//   (A) rt1（load 完再 save）与原文件 mp **逐字节相同**
//   (B) rt2（把 rt1 再 load 一次再 save）与 rt1 **逐字节相同**（幂等）
//   (C) 记下 (A) 不成立的那些，并逐字节找出**第一处不同**在第几个字节、
//       两边各是什么 float —— 因为"不一样"没有信息量，"第 12345 字节、
//       1.0e-12 vs 0"才有。
//
// (A) 有一个**正当**的不成立理由：`LoadBinary_NCHW` 会把
// `fabs(v) < ignore_small_value` 的权重清零（默认 1e-12），
// 而 `SaveBinary_NCHW` 如实写出 0。所以 (A) 不成立**不一定**是缺陷。
// (B) 没有任何这样的豁免 —— 它是**硬判据**。
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
#include "zq_net_fwd_tripwires.h"
#include "zq_concat_getsize_real.h"

#define MODEL_DIR "/mnt/d/ZQCNN/model"

static int g_ok = 0, g_bad = 0, g_same = 0, g_diff = 0, g_skip = 0;

// 逐字节比较。**不是**"找到第一处不同就返回" ——
// `ignore_small_value` 的清零是**逐元素**的，输出第 i 个 float 只由输入第 i 个
// float 决定（长度相同，不会移位），所以可以逐个位置独立判断：
//     凡是不同的位置，输入值都必须满足 fabs(v) < 1e-12（默认阈值）。
// 只查第一处是不够的 —— 第一处合法不代表第二处也合法。
struct DiffStat {
    long ndiff;        // 不同的 float 个数
    long nbad;         // 其中**无法用 ignore_small_value 解释**的个数
    long first_bad;    // 第一个无法解释的位置
    float bad_a, bad_b;// 那两边的值
    float max_cleared; // 被 ignore_small_value 清零的那些里，原值绝对值的最大者
};

static DiffStat cmp_files(const std::string& a, const char* b, double thresh)
{
    DiffStat st;
    // ndiff / nbad 是**计数**，初值 0；first_bad 是"位置"，
    // 用 -1 表示"还没找到"。第一版把三个一起初始化成 -1，
    // 于是 nbad 恒为 -1、判据 `nbad != 0` 恒真 ——
    // **23 个真通过的模型全被报成对不上**。
    // 教训见 AGENTS.md「标签/计数要说它实际数的是什么」：一个哨兵值
    // 被复用到计数上，症状是"全部失败"，而失败信息还带着
    // "首个在 #-1"这种一眼荒谬的输出 —— 看到荒谬的数字要立刻怀疑代码。
    st.ndiff = 0;
    st.nbad = 0;
    st.first_bad = -1;
    st.bad_a = st.bad_b = 0.f;
    st.max_cleared = 0.f;
    FILE* fa = fopen(a.c_str(), "rb");
    FILE* fb = fopen(b, "rb");
    if (!fa || !fb) {
        if (fa) fclose(fa);
        if (fb) fclose(fb);
        printf("        打不开：%s / %s\n", a.c_str(), b);
        st.nbad = 1; st.first_bad = -2;
        return st;
    }
    fseek(fa, 0, SEEK_END); long na = ftell(fa);
    fseek(fb, 0, SEEK_END); long nb = ftell(fb);
    if (na != nb) {
        printf("        长度不同：%ld vs %ld\n", na, nb);
        fclose(fa); fclose(fb);
        st.nbad = 1; st.first_bad = -2;
        return st;
    }
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
    float* fa_ = (float*)&ba[0];
    for (long i = 0; i < nflo; i++) {
        if (pa[i] == pb[i]) continue;
        st.ndiff++;
        float va = fa_[i];
        if (fabs(va) > st.max_cleared) st.max_cleared = fabs(va);
        if (fabs(va) >= thresh) {
            // **不能**用 ignore_small_value 解释 —— 这是真的对不上
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
    snprintf(rt1, sizeof(rt1), "/tmp/zq_rt1_%s.nchwbin", base);
    snprintf(rt2, sizeof(rt2), "/tmp/zq_rt2_%s.nchwbin", base);

    // 第一趟：原文件 -> load -> save
    ZQ::ZQ_CNN_Net n1;
    if (!n1.LoadFrom(zp, mp.c_str())) {
        printf("  %-34s **FAIL** 原文件加载失败\n", base);
        g_bad++;
        return;
    }
    if (!n1.SaveModel(rt1)) {
        printf("  %-34s **FAIL** 回存 1 失败\n", base);
        g_bad++;
        return;
    }
    // 第二趟：rt1 -> load -> save
    ZQ::ZQ_CNN_Net n2;
    if (!n2.LoadFrom(zp.c_str(), rt1)) {
        printf("  %-34s **FAIL** 回存出来的文件加载失败（**回存不可再入**）\n", base);
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
    // (B) 硬判据：load/save 这一对必须是幂等的。
    //     注意：这里**不能**给 ignore_small_value 豁免 ——
    //     第 2 趟是从 rt1 读的，rt1 已经被清过零了，
    //     再清一次是幂等的，所以任何差异都是真的对不上。
    DiffStat db = cmp_files(rt1, rt2, 0.0);
    if (db.nbad != 0) {
        printf("  %-34s **FAIL** (B) load->save->load->save 不幂等：%ld 个 float 对不上，"
               "首个在 #%ld（%.9g vs %.9g）\n", base, db.nbad, db.first_bad,
               db.bad_a, db.bad_b);
        bad++;
    }
    // (A) load->save 必须等于原文件，**除非**差异能被 ignore_small_value 解释：
    //     凡是不同的位置，输入值都要满足 fabs(v) < 1e-12（LoadFrom 的默认阈值）。
    //     这是硬判据 —— "不一样"本身不算结论，"不一样的那些是不是都小于阈值"才算。
    DiffStat da = cmp_files(mp, rt1, 1e-12);
    if (da.nbad != 0) {
        printf("  %-34s **FAIL** (A) 回存与原文件有 %ld 个 float 对不上且**无法**用"
               " ignore_small_value 解释，首个在 #%ld（%.9g vs %.9g）\n",
               base, da.nbad, da.first_bad, da.bad_a, da.bad_b);
        bad++;
    } else if (da.ndiff == 0) {
        g_same++;
    } else {
        g_diff++;
        printf("  %-34s (A) 有 %ld 个 float 与原文件不同，**全部**满足 fabs(v) < 1e-12"
               "（清零的正当效果），其中原值最大者 |v| = %.3g\n", base, da.ndiff, da.max_cleared);
    }
    if (bad == 0) {
        g_ok++;
        printf("  %-34s OK  (B) 幂等；(A) %s\n", base,
               da.ndiff == 0 ? "与原文件逐字节相同"
                             : "差异全部可由 ignore_small_value 解释");
    } else {
        g_bad += bad;
    }
    remove(rt1);
    remove(rt2);
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("权重内容往返门禁（附录 HB）\n");
    printf("判据：(B) load->save->load->save 必须幂等（硬判据）。\n");
    printf("      (A) load->save 与原文件逐字节相同（参考判据，\n");
    printf("          允许因 ignore_small_value 清零而不同，只统计不判失败）。\n\n");

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
    printf("\n共 %d 个 .zqparams：跑过 %d（通过 %d，被拒 %d），跳过 %d\n",
           (int)params.size(), g_ok + g_bad, g_ok, g_bad, g_skip);
    printf("(A) 分类：与原文件**逐字节相同** %d 个；"
           "有差异但**全部**可由 ignore_small_value 解释 %d 个\n", g_same, g_diff);
    return g_bad == 0 ? 0 : 1;
}
