/* 卷积 kernel/dilate 整数溢出守卫的门禁 —— 附录 EO.7（EM 的门禁，EM.5 记的"没有"）
 *
 * 被测的守卫
 * ----------
 * `ZQ_CNN_Layer_Convolution` / `DepthwiseConvolution` / `DeConvolution` 三个类的
 * `ReadParam` 里各有一段（`ZQCNN/ZQ_CNN_Layer.h:629 / 1222 / 1840`）：
 *
 *     if ((__int64)dilate_H * (kernel_H - 1) + 1 > 0x7FFFFFFF
 *         || (__int64)dilate_W * (kernel_W - 1) + 1 > 0x7FFFFFFF) { ...; return false; }
 *
 * kernel/dilate 都来自**模型文件**（不可信输入）。不守的话，
 * `(kernel-1)*dilate` 这个 int 乘法溢出**回绕成正数**，
 * 于是绕过了下游的 `top_H <= 0` 检查，top_H 算成 0 -> 零尺寸张量 ->
 * `firstPixelData = 0` -> 空指针解引用。
 *
 * 判据：乘积**放得进 int** 就放行，放不进才拒。
 * 这一点必须钉死 —— 见下面"最容易写错的期望"。
 *
 * EM.5 曾记"要拖 2215 个符号的卷积内核图，拖不进快速门禁"，
 * **那个测量是错的**：`new ZQ_CNN_Layer_Convolution()` 实际只拖出 **6 个**
 * 未定义符号，且全是 ZQ_CNN_Forward_SSEUtils 的辅助函数 ——
 * 正是 tools/zq_net_fwd_tripwires.h（44 个绊线）覆盖的那一族。
 * 所以这道门禁可以进**快速通道**，不必挂 SLOW。
 *
 * 绊线的作用：`ReadParam` 不会调任何 Forward，被调到就说明门禁走到了不该走的代码。
 *
 * 最容易写错的期望
 * ----------------
 * `kernel=2e9, dilate=1` 的乘积是 2e9，**放得进 int**，所以**必须放行**。
 * 我第一版把它期望成"拒绝"（"2e9 这么大的 kernel 肯定不对"）—— 那不是溢出条件，
 * 按直觉写期望会把一个正确的守卫判成错的（EM.3 踩过，本门禁把它钉成用例 3）。
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

#define RES_FILE "/tmp/zq_convparam_res.txt"

enum { C_CONV = 0, C_DWCONV, C_DECONV };
static const char* g_cls[] = { "Convolution", "DepthwiseConvolution", "DeConvolution" };

struct Case {
    int cls;
    int kernel;
    int dilate;     // 0 = 不写 dilate 参数（走默认值 1）
    int stride;     // 0 = 不写 stride 参数（走默认值 1）；负数 = 写出这个值
    int expect;     // 1 = ReadParam 应当放行，0 = 应当拒绝
};

static const Case g_cases[] = {
    // ---- 常规形状：三个类各来一发，必须照旧放行 ----
    { C_CONV,    3, 1, 1, 1 },
    { C_CONV,    1, 1, 1, 1 },
    { C_CONV,    2, 1, 1, 1 },
    { C_DWCONV,  3, 1, 1, 1 },
    { C_DECONV,  3, 1, 1, 1 },
    // ---- 边界：乘积刚好放得进 int ----
    // 2e9 * (2e9-1) 会溢出；2e9 * 1 = 2e9 < INT_MAX，必须**放行**
    { C_CONV,    2000000000, 1, 1, 1 },
    { C_DWCONV,  2000000000, 1, 1, 1 },
    { C_DECONV,  2000000000, 1, 1, 1 },
    // ---- 边界：乘积越界，三个类都必须拒绝 ----
    { C_CONV,    2000000000, 2, 1, 0 },   // 4e9
    { C_DWCONV,  2000000000, 2, 1, 0 },
    { C_DECONV,  2000000000, 2, 1, 0 },
    { C_CONV,    1000000000, 3, 1, 0 },   // 3e9
    // ---- 常规 dilate / stride：不能被守卫误杀 ----
    { C_CONV,    3, 2, 1, 1 },
    { C_CONV,    3, 4, 1, 1 },
    { C_DWCONV,  3, 2, 1, 1 },
    { C_CONV,    3, 1, 2, 1 },
    { C_DWCONV,  3, 1, 2, 1 },
    { C_DECONV,  3, 1, 2, 1 },
    // ---- stride == 0 / 负数：**整数除零 -> SIGFPE，进程直接死**（附录 EY.1）----
    // 这一族比上面的溢出更狠：溢出的后果是数据错，除零的后果是整个进程没了。
    // 守卫是 `kernel_H <= 0 || ... || stride_H <= 0 || stride_W <= 0`。
    { C_CONV,    3, 1, 0, 0 },
    { C_DWCONV,  3, 1, 0, 0 },
    { C_DECONV,  3, 1, 0, 0 },
    { C_CONV,    3, 1, -1, 0 },
    { C_DWCONV,  3, 1, -1, 0 },
    { C_DECONV,  3, 1, -1, 0 },
    // ---- kernel == 0 / 负数：同一道守卫的另一半 ----
    { C_CONV,    0, 1, 1, 0 },
    { C_DWCONV,  0, 1, 1, 0 },
    { C_DECONV,  0, 1, 1, 0 },
    { C_CONV,   -1, 1, 1, 0 },
    { C_DECONV, -1, 1, 1, 0 },
    // ---- dilate == 0：`(kernel-1)*0+1 = 1` 不溢出，但退化到 1 像素核 ----
    { C_CONV,    3, 0, 1, 0 },
    { C_DECONV,  3, 0, 1, 0 },
};
static const int N_CASE = (int)(sizeof(g_cases) / sizeof(g_cases[0]));

static ZQ_CNN_Layer* make_layer(int cls)
{
    if (cls == C_CONV)   return new ZQ_CNN_Layer_Convolution();
    if (cls == C_DWCONV) return new ZQ_CNN_Layer_DepthwiseConvolution();
    return new ZQ_CNN_Layer_DeConvolution();
}

static std::string param_line(const Case& c)
{
    // stride / dilate 的默认值都是 1，所以**只有不等于 1 时才写进参数行** ——
    // 写成"非 0 才写"的话 dilate=0 和 stride=0 这两个最要紧的用例根本表达不出来
    // （第一版就是这么写的，于是 `stride=0` 那几条永远走的是默认值 1）。
    char buf[256];
    snprintf(buf, sizeof(buf),
             "Convolution name=c bottom=data top=out num_output=8 "
             "kernel_size=%d stride=%d pad=1", c.kernel, c.stride);
    std::string s(buf);
    if (c.dilate != 1) {
        char d[32];
        snprintf(d, sizeof(d), " dilate=%d", c.dilate);
        s += d;
    }
    return s;
}

static int child(int idx)
{
    const Case& c = g_cases[idx];
    ZQ_CNN_Layer* l = make_layer(c.cls);
    std::string line = param_line(c);
    bool ok = l->ReadParam(line);
    delete l;
    FILE* f = fopen(RES_FILE, "a");
    if (f) { fprintf(f, "%d %d\n", idx, ok ? 1 : 0); fclose(f); }
    return 0;
}

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc > 1) {
        int idx = atoi(argv[1]);
        if (idx < 0 || idx >= N_CASE) return 0;
        return child(idx);
    }
    printf("卷积 kernel/dilate 整数溢出守卫门禁（附录 EO.7）\n");
    printf("判据：((kernel-1)*dilate+1) 放得进 int 就放行，放不进才拒。\n");
    printf("      kernel=2e9 dilate=1 的乘积是 2e9，**必须放行**（用例 6~8）。\n");
    printf("      44 个 ZQ_CNN_Forward_SSEUtils 辅助函数是绊线，被调到就红。\n\n");

    int ok = 0, bad = 0, crash = 0;
    for (int i = 0; i < N_CASE; i++) {
        const Case& c = g_cases[i];
        remove(RES_FILE);
        pid_t pid = fork();
        if (pid == 0) { zq_child_silence_stderr(); _exit(child(i)); }
        int st = 0; waitpid(pid, &st, 0);

        int got = -1, loaded = -1, have = 0;
        FILE* f = fopen(RES_FILE, "r");
        if (f) { have = (fscanf(f, "%d %d", &got, &loaded) == 2); fclose(f); }
        int tripped = (WIFEXITED(st) && WEXITSTATUS(st) == 3);

        char tag[160];
        snprintf(tag, sizeof(tag), "%-22s kernel=%-11d dilate=%-3d stride=%d",
                 g_cls[c.cls], c.kernel, c.dilate, c.stride);

        if (!have || WIFSIGNALED(st) || tripped) {
            crash++;
            printf("  %-52s %s%s\n", tag,
                   !have ? "结果文件读不出来"
                         : (tripped ? "**撞上绊线**" : "子进程被信号杀"),
                   tripped ? "（ReadParam 不该调 Forward）" : "");
            continue;
        }
        if (loaded != c.expect) {
            bad++;
            printf("  %-52s %s：实际%s\n", tag,
                   c.expect ? "应放行" : "应拒", loaded ? "放行" : "拒绝");
        } else {
            ok++;
            printf("  %-52s %s\n", tag, c.expect ? "放行" : "拒绝");
        }
    }
    printf("\n共 %d 个用例：对 %d，错 %d，崩/撞线 %d\n", N_CASE, ok, bad, crash);
    return (bad || crash) ? 1 : 0;
}
