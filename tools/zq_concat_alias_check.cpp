/* Concat 的 top/bottom **跨下标别名** —— 附录 EN
 *
 * 缺陷
 * ----
 * `ZQ_CNN_Net::_check_connect` 有一道就地守卫，但它只比**同一下标**
 * （`tops[i][j] == bottoms[i][j]`）。于是这种模型能过：
 *
 *     Concat  name=c  bottom=A  bottom=B  top=B
 *
 * 而 `ZQ_CNN_Forward_SSEUtils::_concat_NCHW` 把 inputs 收成**指针**，
 * 随后才 `output.ChangeSize(out_N, out_H, out_W, out_C, 0, 0)`：
 *
 *   1. output 就是 B，所以 B 被**就地扩容**成 C = out_C，内容被 Reset 清零；
 *   2. 拷贝循环用 `in_C = valid_inputs[i]->GetC()` 取**扩容后**的 C；
 *   3. 每个像素写 out_C 个 float，最后一个像素越出整块分配 —— 越界量
 *      = 排在 B 前面那个输入的 C 个 float。
 *
 * ASan 实证（tools/zq_concat_alias_probe.cpp，逐字照抄那个拷贝循环）：
 * 3 种对齐 x 8 组形状，别名配置 46/48 报 heap 越界，独立 top 的对照组 24/24 干净。
 *
 * 为什么这道门禁测的是**真代码**
 * ------------------------------
 * 复现探针照抄了循环，它证明的是"那段循环在别名下会越界"；
 * **不是**"仓库里那段代码会越界"。要把断言钉在被测代码上，
 * 只能走 `ZQ_CNN_Net::LoadFrom` 的真 `_check_connect`。
 * 这也顺带把"守卫有没有被绕过去"一起钉住。
 *
 * 判据
 * ----
 * 每个用例都是"写一份 .zqparams + 一个空 .nchwbin，然后 LoadFrom"：
 *   - 别名（top == 某个 bottom，跨下标）必须**加载失败**；
 *   - 良性对照（top 是独立 blob）必须**加载成功**。
 * 良性对照是关键 —— 只测"别名被拒"的话，把 `return false` 无脑写在
 * LoadFrom 开头也能全绿（附录 CA.1：先问"我的门禁有没有可能不干活"）。
 *
 * 形态：不编 ZQ_CNN_Forward_SSEUtils.cpp（附录 EC.1 的理由，它会拖进
 * 单编 5 分钟以上的 conv GEMM）。这里要 Link 的只有 ZQ_CNN_Net.h 里
 * `_check_connect` 之前**根本走不到**的那些符号，所以桩越少越好。
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
#include "ZQCNN/ZQ_CNN_Net.h"
// ZQ_CNN_Net 是普通类、Forward() 就定义在头里，所以 include 它就把**每一个**层的
// Forward 都拖成未定义符号。这些是**绊线**桩：被调到就说明门禁走到了不该走的
// 代码，打名字 + _exit(3)，而不是静默返回。
#include "zq_net_fwd_tripwires.h"

using namespace ZQ;

// ---------------------------------------------------------------------
// 唯一一个**不**做成绊线的：_concat_NCHW_get_size
// ---------------------------------------------------------------------
// `ZQ_CNN_Layer_Concat::LayerSetup` 在**加载期**就要算输出形状，走的正是它，
// 所以它是被合法调到的 —— 做成绊线的话，两个良性对照用例会直接 rc=3。
// 第一版就是这么写的，红了还一度以为是守卫没修好。
//
// 逐字照抄的真实现在 tools/zq_concat_getsize_real.h（门禁与模型加载探针共用一份），
// 为什么它必须给真实现、以及它带来的同步义务，都写在那份文件的头注释里。
//
// 特别值得注意：**含越界写的那个拷贝循环 _concat_NCHW 本身仍然是绊线**。
// 也就是说，万一哪天守卫被绕过去、别名模型真的被加载成功，
// 门禁会立刻在 _concat_NCHW 上炸掉，而不是"算出一堆垃圾还报全绿"。
// 这比"只看 LoadFrom 的返回值"多一道保险。
#include "zq_concat_getsize_real.h"


#define RES_FILE "/tmp/zq_concat_alias_res.txt"
#define PARAM_FILE "/tmp/zq_concat_alias_net.zqparams"
#define BINARY_FILE "/tmp/zq_concat_alias_net.nchwbin"

// 36 种层类型里，只有这些在 .zqparams 解析时不需要外部权重。
// ReLU 在 _is_inplace_safe 名单里（top 可以和 bottom 同名），
// 用它把 data 分成 A / B 两个独立 blob，再让 Concat 去引用它们。
static const char* NET_HEAD =
    "Input\t\t\tname=data C=3 H=4 W=4\n"
    "ReLU\t\t\t\tname=r1 bottom=data top=A\n"
    "ReLU\t\t\t\tname=r2 bottom=data top=B\n";

struct Case {
    int alias_slot;   // -1 = top 是独立 blob（良性对照）
                      //  0 = top=A（别名 bottoms[0]）
                      //  1 = top=B（别名 bottoms[1]）—— **原来漏掉的那一种**
    int alias_len;    // 0 = `top=B`（单 top）  1 = `top=AB`（双 top，top[0] 别名）
};

static bool run_case(const Case& c)
{
    std::string param = NET_HEAD;
    if (c.alias_len == 0) {
        if (c.alias_slot == 0)       param += "Concat name=c axis=1 bottom=A bottom=B top=A\n";
        else if (c.alias_slot == 1)  param += "Concat name=c axis=1 bottom=A bottom=B top=B\n";
        else                         param += "Concat name=c axis=1 bottom=A bottom=B top=C\n";
    } else {
        if (c.alias_slot == 0)       param += "Concat name=c axis=1 bottom=A bottom=B top=A top=C\n";
        else if (c.alias_slot == 1)  param += "Concat name=c axis=1 bottom=A bottom=B top=C top=B\n";
        else                         param += "Concat name=c axis=1 bottom=A bottom=B top=C top=D\n";
    }
    // Concat 不是最后一层：补一个 sink，让图是连通的
    param += "ReLU\t\t\t\tname=r3 bottom=";
    param += (c.alias_len == 0) ? (c.alias_slot < 0 ? "C" : (c.alias_slot == 0 ? "A" : "B"))
                                : "C";
    param += " top=out\n";

    FILE* fp = fopen(PARAM_FILE, "wb");
    if (!fp) return false;
    fwrite(param.c_str(), 1, param.size(), fp);
    fclose(fp);
    // 权重文件：本用例的层都不读权重，但 _load_model_file 会**无条件** fopen
    fp = fopen(BINARY_FILE, "wb");
    if (!fp) return false;
    fclose(fp);

    ZQ_CNN_Net net;
    bool loaded = net.LoadFrom(PARAM_FILE, BINARY_FILE);
    return loaded;
}

static const Case g_cases[] = {
    { -1, 0 },   // 0 良性：top=C 独立
    {  0, 0 },   // 1 别名 tops[0]==bottoms[0]（**原来就挡得住**的）
    {  1, 0 },   // 2 别名 tops[0]==bottoms[1]（**原来放行**的那一种）
    { -1, 1 },   // 3 良性：双 top，都独立
    {  0, 1 },   // 4 双 top，tops[0]==bottoms[0]
    {  1, 1 },   // 5 双 top，tops[1]==bottoms[1]（同一下标，原来也挡得住）
};
static const char* g_expect[] = { "应放行", "应拒", "应拒", "应放行", "应拒", "应拒" };
static const int N_CASE = (int)(sizeof(g_cases) / sizeof(g_cases[0]));

// 子进程：跑一个用例，把"加载成功?"写进结果文件
static int child(int idx)
{
    bool loaded = run_case(g_cases[idx]);
    FILE* fp = fopen(RES_FILE, "a");
    if (fp) { fprintf(fp, "%d %d\n", idx, loaded ? 1 : 0); fclose(fp); }
    return 0;
}

static const char* g_alias_desc[] = {
    "top=C（独立）",
    "top=A（= bottoms[0]）",
    "top=B（= bottoms[1]，跨下标）",
    "top=C top=D（两个都独立）",
    "top=A top=C（tops[0]=bottoms[0]）",
    "top=C top=B（tops[1]=bottoms[1]）",
};

// 父进程：每个用例 fork 一个子进程
static int drive()
{
    int ok = 0, bad = 0, crash = 0;
    for (int i = 0; i < N_CASE; i++) {
        remove(RES_FILE);
        pid_t pid = fork();
        if (pid == 0) { zq_child_silence_stderr(); _exit(child(i)); }
        int st = 0; waitpid(pid, &st, 0);

        int got = -1, loaded = -1, have = 0;
        FILE* f = fopen(RES_FILE, "r");
        if (f) { have = (fscanf(f, "%d %d", &got, &loaded) == 2); fclose(f); }

        // rc=3 = 撞上绊线（ZQ_CNN_Forward_SSEUtils 的某个函数被调到了）
        int tripped = (WIFEXITED(st) && WEXITSTATUS(st) == 3);

        if (!have || WIFSIGNALED(st) || tripped) {
            crash++;
            printf("  用例 %d %-28s  %s%s\n", i, g_alias_desc[i],
                   !have ? "结果文件读不出来" : (tripped ? "**撞上绊线**" : "子进程被信号杀"),
                   tripped ? "（走到了本不该走的 Forward）" : "");
            continue;
        }
        int want = (strcmp(g_expect[i], "应拒") == 0) ? 0 : 1;
        if (loaded != want) {
            bad++;
            printf("  用例 %d %-28s  %s：实际%s，%s\n", i, g_alias_desc[i], g_expect[i],
                   loaded ? "放行" : "拒绝", loaded ? "**必须拒绝**" : "**必须放行**");
        } else {
            ok++;
            printf("  用例 %d %-28s  %s（%s）\n", i, g_alias_desc[i], g_expect[i],
                   loaded ? "已放行" : "已拒绝");
        }
    }
    printf("\n共 %d 个用例：对 %d，错 %d，崩/撞线 %d\n", N_CASE, ok, bad, crash);
    return (bad || crash) ? 1 : 0;
}

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    // 带下标 = 只跑那一个用例（变异测试用）；不带 = 跑全套。
    //
    // **第一版只有子进程那半边**，直接跑打印 0/0，harness 照样判"通过" ——
    // 一道什么都不做的门禁是绿的。所以父进程驱动是必须的，不是锦上添花。
    if (argc > 1) {
        int idx = atoi(argv[1]);
        if (idx < 0 || idx >= N_CASE) return 0;
        return child(idx);
    }
    printf("Concat 的 top/bottom 跨下标别名门禁（附录 EN）\n");
    printf("判据：别名模型必须在 _check_connect 处被拒；独立 top 的对照必须能加载。\n");
    printf("      45 个 ZQ_CNN_Forward_SSEUtils 辅助函数是**绊线**，被调到就红。\n\n");
    return drive();
}
