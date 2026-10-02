/* `ZQ_CNN_Tensor4D::ROI` 的**边界检查**门禁 —— 附录 DX
 *
 * 为什么这道门禁只有"拒绝"一类判据（附录 DX.2）
 * ------------------------------------------------
 * `ROI` 的拷贝与 border memset 我逐项核过，**都是对的**；
 * 有问题的是开头那一行边界检查：
 *
 *     if (off_x < 0 || off_y < 0 || off_x + width > W || off_y + height > H)
 *         return false;
 *
 * `off_x + width` 是 **int 加法**。`off_x` 来自 MTCNN 的 P-net 检测框输出
 * （`ZQ_CNN_MTCNN.h:615 / 621` 等处），**是数据/模型可控的**。
 * off_x 足够大时加法**回绕成负数** ⇒ `> W` 不成立 ⇒ **边界检查被整条绕过**，
 * 紧接着 `src_slice_ptr = GetFirstPixelPtr() + off_y*widthStep + off_x*pixelStep`
 * 就是一次**越界读**。
 *
 * UBSan 坐实（修之前）：
 *     ZQ_CNN_Tensor4D.h:74:40: runtime error: signed integer overflow:
 *         2147483645 + 8 cannot be represented in type 'int'
 *
 * **判据只有一条：这些输入必须被拒（返回 false），且不许有任何 sanitizer 报告。**
 * 为什么不做数值比对：`ROI` 的拷贝逻辑我核过是对的（CX.3 记着"正常值全绿"），
 * 而**错的那一条是"该不该拒"**——对"该被拒的输入"做数值比对是没有意义的，
 * 进程能不能活着、报不报 sanitizer 才是判据。
 * 正常值的数值比对留给 `zq_tile_check.cpp` 那种逐格门禁（见 DX.1）。
 *
 * 本门禁**必须用真实张量对象**（与 zq_tile 同理，见附录 DD.2），
 * 因此要编 `ZQ_CNN_Tensor4D.cpp`（约 3 秒）与 resize/remap 内核（链接期需要）。
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <unistd.h>
#include <sys/wait.h>
#include "ZQCNN/ZQ_CNN_Tensor4D.h"

using namespace ZQ;

#define RES_FILE "/tmp/zq_roi_res.txt"

struct Case {
    const char* what;
    int srcH, srcW, srcC;
    int off_x, off_y, width, height;
    int borderH, borderW;
    int expect_ok;        // 1 = 应当成功；0 = 应当被拒
};

// 正常值：整图 / 子块 / 贴边 / 单像素
static const Case g_cases[] = {
  { "\u6574\u56fe\uff08\u504f\u79fb 0\uff0c\u5c3a\u5bf8 = \u539f\u56fe\uff09", 8, 8, 3, 0, 0, 8, 8, 0, 0, 1 },
  { "\u5b50\u5757\uff08\u504f\u79fb 2,2\uff09",                          8, 8, 3, 2, 2, 3, 3, 0, 0, 1 },
  { "\u8d34\u8fb9\uff08\u504f\u79fb 5\uff0c\u5bbd 3\uff0c\u6070\u597d\u5230\u53f3\u8fb9\uff09", 8, 8, 3, 5, 0, 3, 8, 0, 0, 1 },
  { "\u5355\u50cf\u7d20",                                              8, 8, 3, 7, 7, 1, 1, 0, 0, 1 },
  { "\u5e26 border\uff08\u9876/\u5de6\u5404 1\uff09",                     8, 8, 3, 1, 1, 4, 4, 1, 1, 1 },
  // ---- \u5e94\u5f53\u88ab\u62d2\u7684 ----
  { "\u6b63\u5e38\u8d8a\u754c\uff08\u504f\u79fb 4 + \u5bbd 8 > 8\uff09",          8, 8, 3, 4, 0, 8, 8, 0, 0, 0 },
  { "\u504f\u79fb\u8d8a\u754c\uff08off_x = W\uff09",                        8, 8, 3, 8, 0, 1, 1, 0, 0, 0 },
  { "\u8d1f width\uff08\u539f\u6765\u4f1a\u6f0f\u8fc7\u68c0\u67e5\uff09",          8, 8, 3, 0, 0, -1, 4, 0, 0, 0 },
  { "\u8d1f height",                                                  8, 8, 3, 0, 0, 4, -1, 0, 0, 0 },
  // **\u672c\u6b21\u4fee\u7684\u90a3\u4e2a**\uff1aint \u52a0\u6cd5\u56de\u7ed5
  { "**int \u52a0\u6cd5\u56de\u7ed5** off_x = INT_MAX-2",                 8, 8, 3, INT_MAX - 2, 0, 8, 8, 0, 0, 0 },
  { "**int \u52a0\u6cd5\u56de\u7ed5** off_y = INT_MAX-2",                 8, 8, 3, 0, INT_MAX - 2, 8, 8, 0, 0, 0 },
  { "**int \u52a0\u6cd5\u56de\u7ed5** off_x = INT_MAX, width = INT_MAX",  8, 8, 3, INT_MAX, 0, INT_MAX, 8, 0, 0, 0 },
};
static const int N_CASE = (int)(sizeof(g_cases) / sizeof(g_cases[0]));

static void run_one(const Case& c)
{
    ZQ_CNN_Tensor4D_NHW_C_Align0* src = new ZQ_CNN_Tensor4D_NHW_C_Align0();
    ZQ_CNN_Tensor4D_NHW_C_Align0* dst = new ZQ_CNN_Tensor4D_NHW_C_Align0();
    long bad = 0;
    if (!src->ChangeSize(1, c.srcH, c.srcW, c.srcC, 0, 0)) bad++;
    else if (!dst->ChangeSize(1, 1, 1, 1, 0, 0)) bad++;
    else {
        float* sp = src->GetFirstPixelPtr();
        const int ss = src->GetSliceStep();
        for (int i = 0; i < ss; i++) sp[i] = (float)(i % 97) * 0.01f;
        const bool r = src->ROI(*dst, c.off_x, c.off_y, c.width, c.height, c.borderH, c.borderW);
        if (r != (c.expect_ok != 0)) bad++;
    }
    delete src; delete dst;
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %.6e 0\n", 1L - bad, bad, 0.0); fclose(f); }
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(const Case& c)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) {
        FILE* dn = freopen("/dev/null", "w", stderr); (void)dn;
        run_one(c);
        _exit(0);
    }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0, over = 0; double worst = 0; int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%ld %ld %lf %ld", &ok, &bad, &worst, &over) == 4); fclose(f); }
    if (!have || WIFSIGNALED(st)) {
        g_crash++;
        printf("  %-46s  %s\uff08\u4fe1\u53f7 %d\uff09\n", c.what,
               WIFSIGNALED(st) ? "CRASH" : "\u6ca1\u8dd1\u5b8c\uff08ASan/UBSan \u62a5\u9519\u5e76 _exit\uff09",
               WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st));
        return;
    }
    if (bad > 0) { g_bad++; printf("  %-46s  FAIL\uff08%s\uff09\n", c.what,
                                     c.expect_ok ? "\u5e94\u8be5\u6210\u529f\u5374\u88ab\u62d2" : "\u5e94\u8be5\u88ab\u62d2\u5374\u6536\u4e0b"); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("ZQ_CNN_Tensor4D::ROI \u8fb9\u754c\u68c0\u67e5\u95e8\u7981\uff08\u9644\u5f55 DX\uff09\n");
    printf("\u5224\u636e\u53ea\u6709\u4e00\u6761\uff1a**\u8be5\u62d2\u7684\u5fc5\u987b\u88ab\u62d2\uff0c\u4e14\u4e0d\u8bb8\u6709\u4efb\u4f55 sanitizer \u62a5\u544a**\u3002\n");
    printf("  \u5173\u952e\u7528\u4f8b\uff1a`off_x + width` \u662f **int \u52a0\u6cd5**\uff0c\u800c off_x \u6765\u81ea MTCNN \u7684\u68c0\u6d4b\u6846\u8f93\u51fa\uff08\u6570\u636e/\u6a21\u578b\u53ef\u63a7\uff09\uff0c\n");
    printf("  \u56de\u7ed5\u6210\u8d1f\u6570\u540e `> W` \u4e0d\u6210\u7acb \u2192 **\u8fb9\u754c\u68c0\u67e5\u88ab\u6574\u6761\u7ed5\u8fc7** \u2192 \u8d8a\u754c\u8bfb\u3002\n");
    printf("  UBSan \u5750\u5b9e\uff08\u4fee\u4e4b\u524d\uff09\uff1aZQ_CNN_Tensor4D.h:74:40: signed integer overflow: 2147483645 + 8\n\n");
    for (int i = 0; i < N_CASE; i++) one(g_cases[i]);
    printf("\n\u5171 %d \u4e2a\u7528\u4f8b\uff1a\u5168\u5bf9 %d\uff0c\u6709\u9519 %d\uff0c\u5d29\u6e83/\u642d\u5efa\u5931\u8d25 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**\u6bcf\u4e00\u9879\u5728\u4e0b\u7ed3\u8bba\u4e4b\u524d\u90fd\u8981\u5148\u7528\u72ec\u7acb\u590d\u73b0\u5bf9\u4e00\u904d**\uff08\u9644\u5f55 CA.3\uff09\u3002\n");
    return (g_bad || g_crash) ? 1 : 0;
}
