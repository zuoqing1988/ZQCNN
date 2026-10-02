/* tools/ 共用：子进程里怎么处置 stderr —— 附录 CZ
 *
 * 为什么需要它
 * ------------
 * 几乎每道门禁都是"父进程 fork 一个子进程跑一个用例，子进程只把统计写进结果文件"。
 * 原来子进程把 stderr 接到 `/dev/null`，理由是（附录 BN.5）：
 * sanitizer 的报告走 stderr，会把父进程 stdout 那一行**拦腰截断** ——
 * 报告里第一行就是 `==12345==ERROR: AddressSanitizer: ...`，
 * 正好顶在父进程要打印结果的那一行中间。
 *
 * 那个理由是对的，但代价是：
 *
 *   **sanitizer 的报告本身也一并看不见了。**
 *
 * 附录 CY.4 撞上的就是这个：20 道 fork 型门禁里，`zq_nchw_resize` 在 UBSan 下
 * 报的是 `FAIL (rc=1)` 而不是"1 条 sanitizer 报错" —— 因为 UBSan 的消息
 * 被子进程自己吞了。判据仍然是对的（结果文件读不出来 = 失败，附录 CJ.4），
 * 但"知道它失败了、不知道它为什么失败"，要查只能把二进制单独跑一遍。
 * ASan 下更糟：`bad-free` / `heap-buffer-overflow` 同样看不见。
 *
 * 做法
 * ----
 * 1) 子进程把 stderr 重定向到**一个由 harness 通过环境变量指定的文件**
 *    （`ZQ_CHILD_ERR`），而不是 /dev/null；
 * 2) harness 在判定某个门禁失败时，把那个文件的前若干行打出来。
 *
 * 用环境变量而不是在门禁里写死路径，是因为门禁不知道自己的 tag，
 * 而 harness 知道（它就是按 tag 组织的）。
 * 环境变量没设时**退回 /dev/null**，所以单独手工编译运行一个门禁时行为不变。
 *
 * 用法
 * ----
 *     #include "zq_check_child.h"
 *     ...fork 之后、子进程里：
 *         zq_child_silence_stderr();
 *     ...跑用例...
 */
#ifndef ZQ_CHECK_CHILD_H_
#define ZQ_CHECK_CHILD_H_

#include <cstdio>
#include <cstdlib>

// 子进程里调一次：把 stderr 重定向到 $ZQ_CHILD_ERR（没设就 /dev/null）。
// 返回值忽略 —— 重定向失败不该让用例失败。
static inline void zq_child_silence_stderr()
{
    const char* p = getenv("ZQ_CHILD_ERR");
    if (p == 0 || p[0] == 0) p = "/dev/null";
    FILE* f = freopen(p, "w", stderr);
    (void)f;
}

#endif
