# CHANGELOG 2026-10-03


## 新增/变更：附录 CZ —— 让 fork 型门禁的 sanitizer 报告不再被吞掉（附录 CY.7 待办 1）

### 变更文件

* `tools/zq_check_child.h`（新增）—— `zq_child_silence_stderr()`
* 20 个门禁：`FILE* dn = freopen("/dev/null", "w", stderr);` → `zq_child_silence_stderr();`
* `tools/run_zqlib_checks.py` —— 跑每个 tag 时带上 `ZQ_CHILD_ERR`；
  判定失败时把子进程 stderr 的前 24 行打出来；**顺带修掉 UBSan 分支一个真 bug**
* `audit_k3_20261001.md` 新增附录 CZ

**`ZQCNN/` 下没有改动任何生产代码。**

### 为什么

20 道门禁是"父进程 fork 子进程跑一个用例"的形状，子进程把 stderr 接到 `/dev/null`。
理由（附录 BN.5）是对的：sanitizer 报告会把父进程 stdout 那一行**拦腰截断**。
但 `/dev/null` 把**报告本身**也一并扔了 ——
现象是"知道门禁失败了、不知道它为什么失败"，要查只能把二进制单独手工跑一遍。

附录 CY.4 撞上的就是这个：`zq_nchw_resize` 在 UBSan 下报的是 `FAIL (rc=1)`
而不是"1 条 sanitizer 报错"；另外四道（bns / eltwise / lrn / pool）是**进程内**跑的，
消息才进得了 `.out`。**同一个问题，只因门禁形状不同就表现成两种完全不同的现象。**
判据一直是对的（没读到结果文件 = 失败，附录 CJ.4），坏的只是**可诊断性**。

做法：子进程把 stderr 重定向到 `$ZQ_CHILD_ERR`（harness 指定的 per-gate 文件），
harness 判定失败时把前 24 行打出来。环境变量没设时退回 `/dev/null`，
所以手工单独编译运行一个门禁时行为不变。

### 验证：故障注入

在子进程里 `fprintf(stderr, ...)` 一行然后 `_exit(9)`，harness 输出：

```
zq_nchw_reduction                  FAIL (rc=1)
---- zq_nchw_reduction 子进程 sanitizer 报告（前 24 行）----
   TEMP-Fault-Injection: this line must reach the harness
---- 报告结束 ----
0/1 通过
```

端到端通了：子进程死掉之后**不用再手工跑二进制**就知道它写过什么。

### 自己踩的四个坑（都是"看起来机制坏了，其实是注入写错了"）

1. WSL 侧文件生成了、本地目录却空的 —— 复制命令走
   `subprocess.run('wsl ... bash -lc "..."', shell=True)`，要过 **cmd.exe**，
   里面的 `;` `*` `2>/dev/null` 全被 cmd 先解释一遍。
   **必须走 `run_wsl`（脚本经 stdin 送进 bash）**。
2. 复制了但仍不打 —— 复制块写在"打印失败详情"的循环**之后**。
   回归通常是"跑一次就去看"，看到的正好是文件还没到的那一轮。**必须放在循环之前**。
3. 子进程文件 0 字节 —— `fprintf` 到 stderr 是**缓冲**的，而 `_exit()` **不刷新** stdio。
   要 `fflush`。sanitizer 的报告走 `write(2)`，不受这条影响。
4. 门禁仍然全绿 —— 注入里那行注释中的字面 `\n` 让 `_exit(9);` **被 `//` 吞掉了**，
   子进程根本没退出。

第 4 条最值得记：**"我以为我改的东西生效了"和"它真的生效了"是两件事**，
而这一类偏差**不会让编译失败、也不会让测试变红** ——
它只是让你测了一个不存在的行为。

### 顺带修掉的一个真 bug

UBSan 分支的运行命令模板里，实参个数从 7 变成了 8（加 `ZQ_CHILD_ERR` 时多写了一个
`tag`），`--ubsan` 一跑就 `TypeError: not all arguments converted`。
ASan 分支参数是对的，所以**只有 UBSan 那条轴坏了** ——
而 ASan 是默认模式，于是这个 bug 在常规回归里完全看不见。

> 这正是 CZ.2 说的那个现象的另一个版本：**同一件事在两种门禁形状下表现完全不同**。
> 一条轴坏了而默认轴是好的，只有真的去跑那条轴才会发现。

### 撤掉注入之后

* ASan：`30/30 通过`，EXIT=0
* UBSan：`30/30 通过`，EXIT=0
* 编码检查 OK、换行检查 OK（顺带修掉 `tools/zq_nchwc_conv_check.cpp` 的一处 mixed-EOL，
  是批量打补丁时插进去的一行 LF）

### 剩下的待办（CY.7-2，本轮没做）

把"门禁不得用 `std::vector<float>` 喂 align256 入口"做成一条**常驻源码检查**
（现在靠 UBSan 轴被动发现；没被 UBSan 覆盖到的 align256 门禁仍可能带这个缺陷）。
