/* 剩下三个 UNUSED 层的 ReadParam 接线门禁 —— 附录 GE
 *
 * 为什么是这三个
 * --------------
 * 36 种层类型里有 15 种 UNUSED（没有随仓库模型会跑到）。前几轮把其中 11 种
 * 接进了 `zq_layerwire`；剩下 4 种里 `DeConvolution` 已经被
 * `zq_convparam` 覆盖（附录 EO.7 扩到 49 例时含 `C_DECONV`），
 * 于是只剩这三个：`LSTM_TF` / `PriorBoxText` / `DetectionOutput_MXNET`。
 *
 * 之前记的是"需要双模式桩（绊线 + 记录桩）所以不做"（附录 EY.4）。
 * 复看之后那个理由**只对 `Forward` 成立**：
 * 这三处的守卫**全在 `ReadParam` 里**，而 `ReadParam` 根本不碰 Forward ——
 * 与 EY/EZ 一样，构造层对象 + 喂一行参数就够了。
 *
 * 判据（每一处都是"模型文件能控制的东西"）
 * -----------------------------------------
 *   LSTM_TF              return has_hidden_dim && has_type && has_bottom && has_top && has_name
 *   PriorBoxText         !has_bottom || bottom_names.size() != 2 || !has_top || !has_name
 *                        （PriorBoxText **完全继承**基类 ZQ_CNN_Layer_PriorBox 的 ReadParam）
 *   DetectionOutput_MXNET !has_bottom || bottom_names.size() != 3 || !has_top || !has_name
 *
 * 最有价值的是**个数**那几条：`bottom_names` 的大小直接来自 .zqparams 里
 * 有几个 `bottom=`，而下游 `LayerSetup` 硬取 `(*bottoms)[0..2]` ——
 * 少一个就是越界读。
 *
 * 形态：与 `zq_layerwire` 同构，但**不需要**任何 Forward 桩，
 * 所以只依赖 `ZQ_CNN_Tensor4D.cpp` + resize 内核 + `zq_net_fwd_tripwires.h`。
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "zq_check_child.h"
#include "ZQCNN/ZQ_CNN_Tensor4D.h"
#include "ZQCNN/ZQ_CNN_Layer.h"
#include "zq_net_fwd_tripwires.h"

using namespace ZQ;

#define RES_FILE "/tmp/zq_unusedlayers_res.txt"

enum { L_LSTM = 0, L_PRIORBOXTEXT, L_DETOUTPUT_MXNET, L_NCLS };
static const char* g_cls[] = { "LSTM_TF", "PriorBoxText", "DetectionOutput_MXNET" };

// 每个类的"完整合法参数行"与若干"应当被拒"的变体
struct Case {
    int cls;
    std::string line;   // 完整的一行 .zqparams
    int expect;         // 1 = ReadParam 应当放行，0 = 应当拒绝
    const char* why;    // 期望被拒的原因（写进报告，便于核对）
};

static std::string lstm(const char* extra)
{
    std::string s = "LSTM_TF name=l bottom=data top=out hidden_dim=32 type=fw";
    if (extra && *extra) { s += " "; s += extra; }
    return s;
}
static std::string pbt(const char* extra)
{
    std::string s = "PriorBox name=p bottom=feat bottom=img top=out "
                    "min_size=30 flip=1 clip=0 variance=0.1 variance=0.1 variance=0.2 variance=0.2";
    if (extra && *extra) { s += " "; s += extra; }
    return s;
}
static std::string dm(const char* extra)
{
    std::string s = "DetectionOutput_MXNET name=d bottom=loc bottom=conf bottom=prior "
                    "top=out variance=0.1 variance=0.1 variance=0.2 variance=0.2 nms_threshold=0.4 "
                    "nms_top_k=100 keep_top_k=100 confidence_threshold=0.01";
    if (extra && *extra) { s += " "; s += extra; }
    return s;
}

static Case* build_cases(int& n)
{
    static Case v[64];
    n = 0;
// 宏参数**不能**与结构体字段同名。第一版写成
//     #define ADD(cls, line, expect, why)   v[n].cls = (cls); ...
// 预处理器会把 `v[n].cls` 里的 `cls` **也**替换成实参，于是变成
// `v[n].L_LSTM = ...` -> "'struct Case' has no member named 'L_LSTM'"。
#define ADD(KCLS, KLINE, KEXP, KWHY) \
    do { v[n].cls = (KCLS); v[n].line = (KLINE); v[n].expect = (KEXP); \
         v[n].why = (KWHY); n++; } while (0)

    // ---- LSTM_TF：hidden_dim / type / bottom / top / name 五项 ----
    ADD(L_LSTM, "LSTM_TF name=l bottom=data top=out hidden_dim=32 type=fw", 1, "完整合法");
    ADD(L_LSTM, "LSTM_TF name=l bottom=data top=out type=fw", 0, "缺 hidden_dim");
    ADD(L_LSTM, "LSTM_TF name=l bottom=data top=out hidden_dim=32", 0, "缺 type");
    ADD(L_LSTM, "LSTM_TF name=l top=out hidden_dim=32 type=fw", 0, "缺 bottom");
    ADD(L_LSTM, "LSTM_TF name=l bottom=data hidden_dim=32 type=fw", 0, "缺 top");
    ADD(L_LSTM, "LSTM_TF bottom=data top=out hidden_dim=32 type=fw", 0, "缺 name");

    // ---- PriorBoxText（继承 PriorBox 的 ReadParam）：**恰好 2 个 bottom** ----
    ADD(L_PRIORBOXTEXT, pbt(0), 1, "完整合法");
    ADD(L_PRIORBOXTEXT, "PriorBoxText name=p bottom=feat top=out min_size=30 "
                        "flip=1 clip=0 variance=0.1 variance=0.1 variance=0.2 variance=0.2", 0, "只有 1 个 bottom");
    ADD(L_PRIORBOXTEXT, "PriorBoxText name=p bottom=a bottom=b bottom=c top=out "
                        "min_size=30 flip=1 clip=0 variance=0.1 variance=0.1 variance=0.2 variance=0.2",
        0, "3 个 bottom");
    ADD(L_PRIORBOXTEXT, "PriorBoxText name=p bottom=feat bottom=img min_size=30 "
                        "flip=1 clip=0 variance=0.1 variance=0.1 variance=0.2 variance=0.2", 0, "缺 top");
    ADD(L_PRIORBOXTEXT, "PriorBoxText bottom=feat bottom=img top=out min_size=30 "
                        "flip=1 clip=0 variance=0.1 variance=0.1 variance=0.2 variance=0.2", 0, "缺 name");
    ADD(L_PRIORBOXTEXT, "PriorBoxText name=p bottom=feat bottom=img top=out "
                        "flip=1 clip=0 variance=0.1 variance=0.1 variance=0.2 variance=0.2", 0, "缺 min_size");

    // ---- DetectionOutput_MXNET：**恰好 3 个 bottom** ----
    ADD(L_DETOUTPUT_MXNET, dm(0), 1, "完整合法");
    ADD(L_DETOUTPUT_MXNET, "DetectionOutput_MXNET name=d bottom=loc bottom=conf top=out "
                           "variance=0.1 variance=0.1 variance=0.2 variance=0.2", 0, "只有 2 个 bottom");
    ADD(L_DETOUTPUT_MXNET, "DetectionOutput_MXNET name=d bottom=loc bottom=conf "
                           "top=out variance=0.1", 0, "只有 1 个 variance");
    ADD(L_DETOUTPUT_MXNET, "DetectionOutput_MXNET name=d bottom=loc bottom=conf "
                           "bottom=prior variance=0.1 variance=0.1 variance=0.2 variance=0.2", 0, "缺 top");
    ADD(L_DETOUTPUT_MXNET, "DetectionOutput_MXNET bottom=loc bottom=conf bottom=prior "
                           "top=out variance=0.1 variance=0.1 variance=0.2 variance=0.2", 0, "缺 name");

    // ---- LSTM_TF 的 hidden_dim 也要看它是否接受负数/0 ----
    ADD(L_LSTM, lstm(0), 1, "hidden_dim=32 正常");
#undef ADD
    return v;
}

static ZQ_CNN_Layer* make_layer(int cls)
{
    if (cls == L_LSTM)         return new ZQ_CNN_Layer_LSTM_TF();
    if (cls == L_PRIORBOXTEXT)  return new ZQ_CNN_Layer_PriorBoxText();
    return new ZQ_CNN_Layer_DetectionOutput_MXNET();
}

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    int ncase = 0;
    Case* cases = build_cases(ncase);
    int idx = (argc > 1) ? atoi(argv[1]) : -1;

    if (idx >= 0) {
        if (idx >= ncase) return 0;
        ZQ_CNN_Layer* l = make_layer(cases[idx].cls);
        bool ok = l->ReadParam(cases[idx].line);
        delete l;
        fflush(NULL);
        FILE* f = fopen(RES_FILE, "a");
        if (f) { fprintf(f, "%d %d\n", idx, ok ? 1 : 0); fclose(f); }
        return 0;
    }

    printf("三个 UNUSED 层的 ReadParam 门禁（附录 GE）\n");
    printf("判据：%d 个用例，全部来自模型文件能控制的那些字段。\n", ncase);
    printf("      最有价值的是 bottom/top **个数** —— 下游 LayerSetup 硬取 [0..2]，\n");
    printf("      少一个就是越界读。\n\n");

    int ok = 0, bad = 0, crash = 0;
    for (int i = 0; i < ncase; i++) {
        remove(RES_FILE);
        pid_t pid = fork();
        if (pid == 0) {
            zq_child_silence_stderr();
            // 子的 stdout 单独收一个文件：ReadParam 的失败消息全走 stdout，
            // 而父进程在自己打印**之后**才 waitpid，于是那些消息会串到
            // **下一个**用例那一行 —— 附录 FD.4 踩过同样的错位。
            char outf[64];
            snprintf(outf, sizeof(outf), "/tmp/zq_unusedlayers_out_%d.txt", i);
            freopen(outf, "w", stdout);
            int r = 0;
            {
                ZQ_CNN_Layer* l = make_layer(cases[i].cls);
                r = l->ReadParam(cases[i].line) ? 1 : 0;
                delete l;
            }
            fflush(NULL);
            FILE* f = fopen(RES_FILE, "a");
            if (f) { fprintf(f, "%d %d\n", i, r); fclose(f); }
            _exit(0);
        }
        int st = 0; waitpid(pid, &st, 0);

        int got = -1, res = -1, have = 0;
        FILE* f = fopen(RES_FILE, "r");
        if (f) { have = (fscanf(f, "%d %d", &got, &res) == 2); fclose(f); }
        int tripped = (WIFEXITED(st) && WEXITSTATUS(st) == 3);

        if (!have || WIFSIGNALED(st) || tripped ||
            (WIFEXITED(st) && WEXITSTATUS(st) != 0)) {
            crash++;
            printf("  %-22s %-30s 没跑完%s\n", g_cls[cases[i].cls],
                   cases[i].why, tripped ? "（撞上绊线：ReadParam 不该调 Forward）" : "");
            continue;
        }
        if (res != cases[i].expect) {
            bad++;
            printf("  %-22s %-30s %s：实际%s\n", g_cls[cases[i].cls],
                   cases[i].why, cases[i].expect ? "应放行" : "应拒",
                   res ? "放行" : "拒绝");
            // 失败时把这一行参数**原样**打出来：判据错了还是参数行写错了，
            // 看一眼输入就能分清（附录 EZ.4 的 `dims=` vs `dim=` 就是这么定位的）。
            printf("        输入: %s\n", cases[i].line.c_str());
            {
                char outf[64];
                snprintf(outf, sizeof(outf), "/tmp/zq_unusedlayers_out_%d.txt", i);
                FILE* of = fopen(outf, "r");
                if (of) {
                    char lb[512];
                    while (fgets(lb, sizeof(lb), of)) printf("        | %s", lb);
                    fclose(of);
                }
            }
        } else {
            ok++;
            printf("  %-22s %-30s %s\n", g_cls[cases[i].cls], cases[i].why,
                   res ? "放行" : "拒绝");
        }
    }
    printf("\n共 %d 个用例：对 %d，错 %d，崩/撞线 %d\n", ncase, ok, bad, crash);
    // 汇总行里**不出现字面 "FAIL"**（附录 FD.5：harness 的 ASan 判据是
    // `grep -cE 'FAIL' <tag>.out`，把含该字的行数当"断言失败数"）。
    return (bad || crash) ? 1 : 0;
}
