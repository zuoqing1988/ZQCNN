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


## 新增/变更：附录 DA —— 更正 CX 的一个错误结论，顺着查出两条真缺陷

### 变更文件

* `ZQCNN/ZQ_CNN_Layer.h` —— 1 行（`GetTopDim` 的 `top_W` 用了 `real_kernel_H`）
* `audit_k3_20261001.md` 新增附录 DA；**并在 CX 的两处就地标注更正**
* `docs-changelogs/CHANGELOG_2026-10-02.md` 的 CX 段就地加更正标注
* `tools/zq_nchw_deconv_check.cpp` 文件头更正

### 我错在哪：一次静默返回空的 grep

上一轮我执行的是

    grep -rn "deconvolution\|deconv" --include=*.cpp --include=*.h ZQCNN/

而层类名是 **`ZQ_CNN_Layer_DeConvolution`** —— `DeConvolution` 的前六个字母
`D-e-C-o-n-v` 只有**加 `-i`** 才会匹配小写的 `deconv`。命令于是**返回空且不报错**。
我把"空"当成了"不存在"，写下「仓内零调用方」，并据此判定这一整条路是死代码。

补上 `-i`：

    1408:  class ZQ_CNN_Layer_DeConvolution : public ZQ_CNN_Layer
    1465:      DeConvolutionWithBiasPReLU(...)     1488: DeConvolutionWithBias(...)
    1514:      DeConvolutionWithPReLU(...)         1537: DeConvolution(...)

而且 `ZQ_CNN_Net.h:433` **把这个层类型注册进了模型解析器**
（`type: "DeConvolution"` 就能 new 出这个层）。

**正确的事实**：

| | CX 的说法（错） | 事实 |
|---|---|---|
| 层 → wrapper → 内核 | 不存在 | **完整接通**，4 个 wrapper 全被调用 |
| 模型文件能否打开这条路 | 不能 | **能** |
| 随仓库发布的模型里有没有用 | —— | **没有**（`grep -rlin deconv model/` 为空） |
| CX.2 修的 `end_kh` 越界读 | 死代码里的问题 | **模型可达**（形状触发即可） |
| CX.6 记的 `dilation` 被忽略 | 公开 API 陷阱 | **模型可达 + 模型可控的静默错误结果** |

CX.3「不发数值门禁」这个决定本身不变，理由从"零调用方"换成
"**没有随仓库发布的参考模型** + 四份说法互不吻合"，后一条仍然成立。

> 这是本会话"工具静默失败 → 结论错 → 真缺陷被一起藏起来"的最贵一次：
> 代价 ≈ 一整轮，而且差一步就漏掉下面两条缺陷。
> **教训：grep 的空结果不是证据，除非能说出它为什么空。**

### 缺陷一：`GetTopDim` 的 `top_W` 用了 `real_kernel_H`（已修）

    int real_kernel_W = (kernel_W - 1)*dilate_W + 1;
    int real_kernel_H = (kernel_H - 1)*dilate_H + 1;
    top_H = __max(0,(bottom_H - 1)*stride_H + 1 - real_kernel_H + (pad_H_top + pad_H_bottom) + 1);
    top_W = __max(0,(bottom_W - 1)*stride_W + 1 - real_kernel_H + (pad_W_left + pad_W_right) + 1);
                                                       ^^^^^^^^^^^^^^^ 应该是 real_kernel_W

同文件 A/B：`ZQ_CNN_Layer_Convolution::GetTopDim` 的同名公式是对的
（`top_H` 用 `kernel_H`、`top_W` 用 `kernel_W`）。
另一个指纹：`real_kernel_W` **声明了却一次没用**。

后果：`kernel_H != kernel_W` 时 top 张量的初始宽度按高度算；模型可控
（`kernel_H`/`kernel_W` 都是裸 atoi，无校验）。不构成越界写（wrapper 随后会
ChangeSize，见下条），但**输出尺寸是错的**。已修。

### 缺陷二：`output_H` / `output_W` 被无条件改掉（判定不修）

模型里可以写 `output_H` / `output_W`；层里 `atoi`、校验 `> 0`、并在 `GetTopDim` 里
**确实生效**，顶层张量就是按它分配的。
但 `ZQ_CNN_Forward_SSEUtils.h:298-302`（`DeconvolutionWithBiasPReLU`，
另外三个 wrapper 同一形状）紧接着按公式算出 `need_H/need_W`，
`if (out_H != need_H || ...) output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0)`
—— **无条件改回去**。于是这两个参数净效果是**零**：模型里写了、层里用了、
wrapper 里扔掉。

**判定不修**：哪个才对无法判定（模型显式指定应被尊重，还是历史遗留），
两种改法都是在猜；改成"尊重"要动 4 个 wrapper 的签名，改成"删参数"会让
可能正在写它的模型直接报错。

这也是 CX.3「判定不了意图」的更有力证据：这条路上一共有**四份**关于输出尺寸的说法：

1. `GetTopDim`（转置卷积形状）
2. `GetTopDim` 的 `output_H/W`（模型显式指定）
3. wrapper 的 `need_H/need_W`（**卷积**形状）
4. 内核内部关系式 `oh = i*S - pad + kh`（转置卷积）

**今天实际生效的是 4 + 3**（3 的长度截断了 4 的范围），1 和 2 都会被 3 覆盖掉。
1 和 4 说转置卷积、3 说卷积 —— 本来就对不上。

### 顺带实测：`-Wunused-variable` 这条轴信噪比很差，不加成常驻门禁

`real_kernel_W` 本来会被 `-Wunused-variable` 指出来，而 `warn_sweep_src.py`
把它关掉了。实测 43 个 TU 打开它之后：79 条 `t2` / 79 条 `t1` / 73 条 `t5` ……
滤掉 `t1..t5`（计时）、`d1..d3`、`*_sliceStepN`（各对齐变体）这些结构性噪声，
剩下的绝大多数还是「没用的形参」「没用的临时量」，真正像 `real_kernel_W`
（**成对计算里只有一个被用**）的**只有一条**，而那条我已经靠读代码找到了。

**结论：不加成常驻门禁。** 400 条里只有 1 条有信号的轴，天天跑只会训练大家
忽略它 —— 这正是 `warn_sweep_src.py` 当初关掉它的理由，而那个决定是对的。
记录实测数字作为"为什么不开这条轴"的依据。
