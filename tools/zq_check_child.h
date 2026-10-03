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
 * **必须是追加（"a"）而不是截断（"w"）** —— 附录 DY.4 又补了第三条
 * ---------------------------------------------------------------
 * 一道 fork 型门禁会 fork **几十个子进程**（`zq_reshape` 50 个），
 * 而它们**共用同一个 `ZQ_CHILD_ERR` 路径**。原来是 `"w"`：
 * 每个子进程一 `freopen` 就把文件截断，于是**排在崩溃用例后面的用例
 * 会把崩溃用例的 sanitizer 报告擦掉**。
 *
 * 症状极具欺骗性：门禁确实红了（崩溃用例的结果文件读不出来 → rc=1，
 * 判据 CJ.4 生效），但 harness 打印的"子进程 sanitizer 报告"一栏
 * **一个字都没有**，看上去像"根本没有 sanitizer 报告"。
 * —— 附录 DA.2「看起来正常的输出是最危险的失败模式」的又一例：
 *    这次不是 grep 静默失败，是**机制本身被后一个用例擦掉了**。
 *
 * 而且它只在**子进程崩在中途**时才发作：崩在最后一个用例时报告恰好还在。
 * 所以第一版（和后来的几次）跑出来都是"正常"的，压根看不出有问题 ——
 * 是这次拿 `zq_reshape` 做变异测试、崩溃用例正好排在第 20~22 个才暴露。
 *
 * 改成 `"a"`：所有子进程往同一份报告**顺次追加**，崩溃用例的诊断留到最后。
 * 跨轮不会累积，因为 harness 每轮开头 `rm -rf $WDIR`（`WDIR=/tmp/zqchecks`）。
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
    FILE* f = freopen(p, "a", stderr);   // **追加**，理由见文件头"附录 DY.4"
    (void)f;
}

#endif
