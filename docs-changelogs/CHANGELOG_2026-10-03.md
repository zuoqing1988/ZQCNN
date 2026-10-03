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


## 新增/变更：附录 DB —— 把「某条路径有没有被用到」变成一条可复算的门禁

### 变更文件

* `tools/reachability_probe.py`（新增）—— 层类型可达性探针
* `tools/reachability_baseline.txt`（新增）—— 36 行基线
* `tools/run_audit_checks.py` 新增 `--reachability`（C7 组，约 1 秒）
* `audit_k3_20261001.md` 新增附录 DB

**`ZQCNN/` 下没有改动任何生产代码。**

### 为什么要有这个工具

附录 DA.2 那次错判的代价 ≈ 一整轮，根因是**一次手敲 grep 的静默失败**：
命令没加 `-i`，类名 `DeConvolution` 的大小写不匹配，**返回空且不报错**，
我把"空"当成了"不存在"。本工具把那条教训变成机制：跑一次拿到一张
**可复算**的表，不再靠记忆、不靠手敲 grep。

### 实测结果

    注册表：ZQCNN/ZQ_CNN_Net.h，共 36 种
    扫描：model/*.zqparams，共 27 个
    合计 36 种：EXERCISED 20 / COMMENTED 1 / UNUSED 15

**15 个 UNUSED**：`DeConvolution` / `BatchNorm` / `Scale` / `Copy` / `LSTM_TF` /
`ScalarOperation` / `UnaryOperation` / `Sqrt` / `Tile` / `Reduction` / `LRN` /
`Squeeze` / `PriorBoxText` / `PriorBox_MXNET` / `DetectionOutput_MXNET`

**1 个 COMMENTED**：`Dropout`（只出现在 `#Dropout` 这样的注掉行里）

### 三态而不是两态，理由

`model/*.zqparams` 里有一堆 `#Convolution`、`#Dropout` 这样**被 `#` 注掉**的行。
`grep "Dropout" model/*.zqparams` 一定命中 —— 但那行**根本不参与前向**。
只看"有没有搜到"，会把"看着在用"和"真的在用"混成一谈。

`Dropout` 是活例子：它是全表唯一那个 `COMMENTED`，
而它恰好是本审计已修过一处**双重缩放**（附录 CO）的层 ——
那条缺陷所在的路，**从一开始就没有任何随仓库发布的模型能跑到**。

### 这张表能/不能回答什么

**能**：「这条路径会不会被任何随仓库发布的模型跑到」——可复算。
**不能**：跑起来对不对、跑没跑到那条分支（一个 Convolution 被 27 个模型用到，
不代表每种 stride/dilation/pad 组合都被覆盖）。

写可达性结论时，**「模型里没搜到」不再是证据**；要拿这张表。

### 门禁化

    python tools/reachability_probe.py                            # 看表
    python tools/reachability_probe.py --unused                    # 只看没被跑到的
    python tools/run_audit_checks.py --reachability                # C7 组，约 1 秒

基线 36 行 `<层类型>\t<状态>\t<未注释命中数>\t<被注释命中数>`；
`--check-baseline` 在**状态**变化时失败，命中数变化只提示不失败。

### 工具自己的一条设计约束

> **「没有搜到」和「不存在」是两件事。**
> 本工具在"注册表解析出 0 个层类型"时**直接返回 1 并打印「正则失效了」**，
> 而不是报「ZQCNN 没有层」——这正是 DA.2 里我踩的那个坑的形状：
> 工具静默地给出一个看起来合理的答案。
> 同一条纪律在附录 CA.3、CJ.4、CX.5 里各出现过一次，这里是第四次。

### 未完成的部分（如实记下）

**「这 16 种无模型跑到的层各自有没有门禁覆盖」这份交叉表本轮没做对。**
第一版用正则按层类型名去门禁源码里搜，结果是错的：
`LSTM_TF` 在门禁里写作 `zq_cnn_lstm_TF_32f`、`Reduction` 写作
`zq_cnn_reduction_32f`、`Sqrt` 写作 `zq_cnn_sqrt_32f`，按层名搜一个都搜不到；
反过来 `LRN` 在 `zq_eltwise_check.cpp` 里出现纯属巧合。
要做得靠**显式的「层类型 → 内核入口前缀」映射表**，而不是正则。

本会话已能确认有门禁覆盖的至少包括：`DeConvolution`（CX）、`LSTM_TF`（CW）、
`Reduction`（CT）、`Sqrt`（CS）、`ScalarOperation`（CQ）、`Dropout`（CO）、
`LRN`（既有 `zq_lrn` 门禁）。**其余 9 种未逐条核对** ——
下一块该做的是把那张映射表补出来，从而得到「既没有模型、又没有门禁」的真正清单。


## 新增/变更：附录 DC —— 把「既无模型、又无门禁」挖出来，第一条是 LRN 的两条入口

### 变更文件

* `tools/zq_lrn_check.cpp` —— 改成入口表，**三个 32f 入口都测**（原来只测 align256）
* `audit_k3_20261001.md` 新增附录 DC

**`ZQCNN/` 下没有改动任何生产代码。**

### 三跳追踪

附录 DB 给了"哪些层没被 shipped 模型跑到"的表，但"这些层有没有门禁"这一半
本轮用正则没做对（`LSTM_TF` 在门禁里写作 `zq_cnn_lstm_TF_32f_32f`，一个都搜不到）。
本轮用**三跳追踪**走通：

    ZQ_CNN_Layer_LRN 的 Forward
       └─ ZQ_CNN_Forward_SSEUtils::LRN_across_channels(...)   （.h 里的公开包装器）
            └─ _lrn_across_channels(...)                       （.cpp 里的私有辅助）
                 └─ zq_cnn_lrn_across_channels_32f_align{0,128bit,256bit}

**三跳都必要**：`.h` 的公开包装器**不直接调内核**，它调 `.cpp` 的私有辅助；
只做两跳的话每个包装器都返回"没有内核"，而**返回空看起来和"确实没有内核"
一模一样** —— 正是附录 DA.2 那个形状。

抽取器带**自检**：任何包装器抽不出内核符号时明确打印
"**这说明抽取器仍有漏洞，不是'这些包装器没有内核'**"，工具不许用"空"冒充"没有"。
（自检还真抓到了我一个 bug：`PRIV_RE` 里已经有一个 `_`，我又传了带 `_` 的名字，
拼成 `::__reduction_sum`，一个都匹配不上。）

### 结果

24 个包装器里 18 个抽出了内核，逐一对应到门禁。
**"既无模型、又无门禁"的第一条**：

`ZQ_CNN_Layer_LRN` 没有任何 shipped 模型用到（附录 DB.2），
而 `tools/zq_lrn_check.cpp` 把 `const int align = 8;` **写死**、只调
`zq_cnn_lrn_across_channels_32f_align256bit`。于是
**`zq_cnn_lrn_across_channels_32f_align0` 与 `..._align128bit`
这两个真实符号只被编译、从不被执行、更没有被比对过。**

### 补上：三个入口都测

改成入口表（名字写全、走函数指针表），`align` 按入口取（1 / 4 / 8），
三个入口各跑一遍原来的形状集合（`local_size == 1` 且 C 从 1 到 17 扫一遍，
加上 `local_size` 3..9）。

* ASan：`三个入口全部通过`
* UBSan：`三个入口全部通过`

**没有查出新缺陷** —— 这是一个诚实的"补覆盖、没出彩"的结果，
和 CX（补覆盖同时出两条缺陷）不一样，如实记下。

### 一次变异测试，以及它推翻掉的一个说法

我先在代码注释里写：「align0 若也按 8 补齐，值会错但不会崩」。
**变异测试（三个入口全部写死 `align = 8`）之后门禁依然全绿** —— **那句话是错的**：
align0 的标量循环只按 C 走、补齐区被完全忽略；两个 SIMD 入口的向量化内层
上界也是 C、尾巴走标量。所以按入口取宽度是**卫生**、**不是正确性所系**。

注释已按实测结论改写，并补上另一半：这道门禁**不能**验证"align0 入口会不会
误读补齐区"，因为参考实现用的是同一个 pixStep，两者一起错就一起对。
**别把这次变异测试的"没红"当成"宽度无关紧要"的证据，它只说明这条变异是无效变异。**

> 与附录 CX.5 的两次假绿同源：**"没红"有三种原因——没测到、变异无效、真的无关。
> 必须区分。**

### 剩下的清单

| 包装器 | 状态 |
|---|---|
| `PriorBoxText` / `PriorBox_MXNET` / `DetectionOuput_MXNET` | 无模型、无门禁；逻辑在 .cpp 私有辅助里，**抽取器到第三跳就断了**，不是"已确认无缺陷" |
| `Tile` | 无模型、无门禁；走 `ZQ_CNN_Tensor4D::Tile`，附录 BF 修的整数回绕就在这条路上，**修复后没有门禁** |
| `BatchNorm_Compute_b_a` | 逻辑内联在 .h 里，现有内核级门禁（CO）到不了 |
| `Copy` / `Squeeze` | 无模型，且层的 Forward 里**没有** `ZQ_CNN_Forward_SSEUtils::` 调用 |

**下一块该做**：把 `Tile` / `Squeeze` / `Copy` / 三个 MXNET 路径补上覆盖。
它们要么走张量类、要么逻辑内联在头文件里 —— **现有那道"内核级门禁"的天花板
就到不了那里**，这是门禁形态本身要扩的地方，不是补几个用例能解决的。


## 新增/变更：附录 DD —— `Tile` 门禁，查出 W 方向复制缺陷，并给"门禁形态"立了个先例

### 变更文件

* `tools/zq_tile_check.cpp`（新增）—— `ZQ_CNN_Tensor4D::Tile` 门禁（45 个用例）
* `tools/run_zqlib_checks.py` 登记 `zq_tile`（四处）
* `ZQCNN/ZQ_CNN_Tensor4D.h` —— 三个扩展循环的外层上界与步长
* `audit_k3_20261001.md` 新增附录 DD

### 门禁形态：第一次必须编 `ZQ_CNN_Tensor4D.cpp`

其余 30 来道门禁都直接调 `zq_cnn_*` 内核、用紧凑下标自己填 `std::vector`、
不经过张量类（附录 CE 立下的规矩）。但 `Tile` 的逻辑写在 `ZQ_CNN_Tensor4D` 的
**虚函数**里（全文只有一处定义，没有 align0/128/256 三个变体），经
`ZQ_CNN_Forward_SSEUtils::Tile` 直接转发 —— **要测它必须用真实张量对象**。

成本实测：`ZQ_CNN_Tensor4D.cpp` 带 ASan 编译一次约 **3 秒**，于是这道门禁能进
**每次**回归。链接期还需要 `zq_cnn_resize_32f_align_c.c`（`ResizeNearest` /
`Remap` 方法要调它们）—— 第一版按名字猜成 resize / remap 两个文件，
第二个**根本不存在**；remap 那几个符号在**同一个** resize 源文件里。

覆盖面：3 个张量子类（align0 / align128bit / align256bit）× 15 个用例 = 45 个。
`Tile` 只有一份实现，但 **`out_*` 步长来自具体子类**，所以三个都要跑。

### 缺陷：三个扩展循环的外层上界与步长都写错了

| 方向 | 原来 | 应当 |
|---|---|---|
| Tile W | `n < tile_n` / `h < tile_h` | `n < N` / `h < H` |
| Tile H | 只复制**一个输入行高的块**（步长 `out.widthStep*H`），且只做第 0 行 | 逐个**输入行**、步长 `H*out.widthStep`、复制**一行** |
| Tile N | `int elt_num = out.sliceStep*N;`（一个**步长**），只复制第 0 个 slice | 遍历全部 N 个输入 slice，目标 `n + nn*N`（与 H/W 一致的取模语义） |

实测 `N=1,C=3,H=4,W=5, tile=1x1x2x1`：**45 格错**，第一个错在 (n=0,h=1,w=5,c=0)；
而 H 方向、C 方向、N 方向单独测全部 0 错。

**为什么以前没人发现**：`tile_w > 1` 且 `H > tile_h` 才触发；`tile_h == 1` 时
恰好只复制第 0 行（那时就是错的），而 `H == 1` 时又恰好覆盖全部 ——
**两种情况都不错**。旧用例里既没有 `tile_w > 1`、也没有 `H > tile_h` 的组合。

**可达性**：`Tile` 在 `model/` 里 UNUSED，但 `ZQ_CNN_Net.h` 注册了这个层类型，
模型文件可以打开这条路 —— **模型可达，但没有 shipped 模型会踩到**。

### 门禁抓到了我自己改错的两个地方

修 DD.3 时我连着改错两次，**两次都是门禁当场抓住的**：

1. **Tile H 的步长**写成 `hh*out.widthStep`（步长 1）。输入第 h 行应映射到输出的
   h, h+H, h+2H, ...；步长 1 会与上一个 h 的输出区重叠并覆盖。门禁立刻把原本
   通过的 `1x3x1x1` 变成 150 格错。
2. **Tile N 的目标步长**写成 `out.sliceStep`（交错），N=2/tile_n=2 时会把 slice 1
   写两遍、slice 3 永远不写。门禁把失败从 1 个用例扩到 3 个。

改对之后 **45 -> 42 全对**。这比缺陷本身更值得记：**门禁不只抓被测代码，
它同样抓改代码的人**；没有那道"逐格 + 报第一个错在哪"的门禁，这两个错会直接进主干。

### 还没定位的一个组合（如实记下，已从门禁里标出）

`N=2, C=5, H=2, W=3, tile=2x1x1x2` 仍错 180 格，第一个错在 (n=0,h=1,w=0,c=0)。
**它在修复之前就是红的**（当时 9 个失败里就有它），所以**不是我改出来的**；
我的修改把失败从 9 个降到 3 个。已排除：N=1/C=3 的四个方向单独测全对、
N=2/tile_n=2 单独测全对；只有 **N>1 与 C=5 同时出现**才复现，而 C=5 是唯一一个
既不是 4 也不是 8 的倍数的通道数 —— 怀疑与 `out.pixelStep` 的补齐有关，
但**没有证据，不下结论**。已把这个组合从门禁里**标出来并注明原因**
（在 `g_cases[]` 里就写着），而不是悄悄删掉或改判据。

### 门禁自己踩的三个坑

1. **门禁自己越界读**：`out` 此刻只有 1x1x1x1，我按 `sliceStep*4` 去读它
   => 45 个用例全崩，症状看起来像 Tile 坏了。
2. **缓存了会被重新分配的指针**：`op = out->GetFirstPixelPtr()` 写在调用 `Tile`
   **之前**，而 `Tile` 内部的 `ChangeSize` 是 free + malloc => `op` 悬空，
   ASan 报 heap-use-after-free（freed by ZQ_CNN_Tensor4D.cpp:218）。
   **这正是我这些轮次一直在审的"所有权"错误（附录 CU.9），出现在我自己的门禁里。**
3. **第一版参考把四个方向都写成"整除"**，应该是**取模**。15 个用例红，
   **红的形状还很规整**（凡 H/W/C 任一方向 tile>1 就红），看着极像真缺陷 ——
   把前 12 个输出格和期望打出来一看，**Tile 给的就是对的**。
   **形状一致并不等于结论正确。**

> 三次都是同一句话：**门禁坏了和被测代码坏了，输出长得一模一样。**
> 这已经是本会话第五次遇到（CU.8.1、CX.5、CV.3、CU.7、这里）。

### 现在的状态

ASan 42/42、UBSan 42/42（不含 DD.5 那一组）。附录 BF 修的**整数回绕**用例已被钉住
（`C=3, tile_c=0x55555556` 必须被拒）；`tile_* <= 0`、乘积超 `0x7FFFFFFF`、
被拒时输出不被写 —— 都已被钉住。

**下一块的第一件事**：把 DD.5 那一组定位掉，然后放回门禁。


## 新增/变更：附录 DD.8 —— 更正 DD.5 的结论：触发条件是 `N>1`，与通道数无关

### 变更文件

* `tools/zq_tile_check.cpp` —— 把 `N>1` 的组合放回用例表，并加 `KNOWN_FAIL` 机制
* `audit_k3_20261001.md` 新增附录 DD.8；**并在 DD.5 就地标注结论被更正**

**`ZQCNN/` 下没有改动任何生产代码。**

### DD.5 的判断是错的

DD.5 里我写"只有 N>1 **与 C=5 同时出现**才复现，C=5 是唯一一个既不是 4
也不是 8 的倍数的通道数"。**逐维二分推翻了这个判断**（N=2, C=5, H=2, W=3）：

| 打开的 tile 维度 | 错格数 |
|---|---|
| 全 1（恒等） | **0** |
| 只开 tile_c = 2 | **90** |
| 只开 tile_n = 2 | 0 |
| 只开 tile_h = 2 | **60** |
| 只开 tile_w = 2 | **90** |

换通道数（tile_n=2, tile_c=2）：C=4 错 144 / C=6 错 216 / C=8 错 288。

**结论：只要 `N > 1` 且任一 tile 方向不是 1 就错，与 C 是不是 5 完全无关。**
C=4/6/8 错得更多只是因为格子总数更大。

**第一版之所以只看到一个用例红**：用例表里 **N>1 的组合只有那一个**，
其余 14 个全是 N=1，而 N=1 时四个方向**全对**。
**用例表的形状决定了能看到多少缺陷** —— 这与附录 CV.5
「N=1 的那一档让 `n < 1` 与 `n < N` 等价」是同一条规律，
只不过那一次它是**保护**、这一次它是**遮蔽**。

### 已经排除的

* `nm` 确认全仓只有一份 `Tile`（`ZQ_CNN_Tensor4D.h:240`，弱符号），
  不存在"编译进去的不是我读的那份"
* 手工重编的 `t4dx.o` 与 harness 编的 `zq_t4d.o` 行为一致（排除"探针链了旧对象"）
* Tile C 循环（286~307 行）逐行读过，逻辑本身正确
* ASan 没有任何报告 ⇒ 这不是越界，纯粹是**写到了错误的位置 / 该写的地方没写**

### 观测到的现象（N=2, C=5, H=2, W=3, tile=2x1x1x2）

    in  ps=5  ws=15  ss=30      out ps=10 ws=30 ss=60   (out 4x10x2x3)

    out[n=0,h=0][0][0] = 0.001 = ip[ 0]   （期望 in[0,0] = ip[ 0]）  ✓
    out[n=0,h=1][0][0] = 0.031 = ip[30]   （期望 in[0,1] = ip[15]）  ✗ 偏了一个 in.widthStep
    out[n=1,h=0][0][0] = 0.046 = ip[45]   （期望 in[1,0] = ip[30]）  ✗ 同样偏一个 in.widthStep
    out[n=1,h=1][0][0] = 0.000            （期望 in[1,1] = ip[45]）  ✗ **该格从未被写**

前两个偏移量都**恰好等于一个 `in.widthStep`（15）**，
第三格的值在整个输入里找不到 ⇒ **它从没被任何一步写过**。

### 现在的处理：KNOWN_FAIL

放在 `g_known[]` 里：默认跳过、**每轮打印**，加 `--known-fail` 可以跑
（会红 3 个，三种对齐子类各一）。门禁当前 42/42 全绿。

两个极端都不可接受：放进默认回归 ⇒ 门禁恒红 ⇒ 有人会去"修"它
（或更糟：习惯性地忽略红门禁）；从用例表里删掉 ⇒ 缺陷到期清零、没人知道曾经存在。

**下一块的第一件事**：按"偏一个 in.widthStep + 有一格从未被写"这两条线索定位。


## 新增/变更：附录 DD.9 —— `Tile` 的 `N>1` 缺陷定位并修复（一个 token）

### 变更文件

* `ZQCNN/ZQ_CNN_Tensor4D.h` —— `Tile C` 的 `n` 循环增量（一个 token）+ 原因注释
* `tools/zq_tile_check.cpp` —— `g_known[]` 清空（机制保留）
* `audit_k3_20261001.md` 新增附录 DD.9

### 缺陷

    n++, in_slice_ptr += sliceStep, out_slice_ptr += sliceStep)
                                          ^^^^^^^^^^^^^ 用的是**输入**的 sliceStep
    应当是 out.sliceStep

`in_slice_ptr += sliceStep` 用 `this->sliceStep`（输入张量自己的）—— **正确**；
`out_slice_ptr += sliceStep` 同样用 `this->sliceStep`，但它推进的是**输出**张量的指针 ——
错在**张量**选错了，不是步长本身。

**为什么"全 tile=1"时看不出来**：`out` 的尺寸是 `N*tile_n, H*tile_h, W*tile_w, C*tile_c`，
全为 1 时 `out` 的形状与 `in` 完全相同、两个 `sliceStep` 恰好相等 ⇒ **完全没有可观测后果**。
任何一个 tile 方向 > 1，`out.sliceStep` 就更大，`n` 从 1 开始的每个 slice 都**整体前移**，
而输出张量的尾部**永远不被写**。

这解释了 DD.8 的每一条观测：`out[0][0]` 正确（n=0 时增量还没执行）；
`out[0][1]` 拿到的是 `in[1][0]` 的数据；最后一个 slice 的最后一行**从未被写**。

修一个词之后：**45/45 全绿**（ASan 与 UBSan 各一遍），含之前 KNOWN_FAIL 的 3 个。

### 我上一轮为什么没能从源码看出来

**我读错了。** 那一行的原文是

    288:  n++, in_slice_ptr += sliceStep, out_slice_ptr += sliceStep)

我把它读成了 `out_slice_ptr += out.sliceStep`（"两个 sliceStep 长得像，
带 `out.` 的那个应该是输出"），于是连续三轮**拿着一个不存在的源码去推理行为**，
怎么推都对不上 —— 而"怎么都对不上"本身早就在提示我：**我读的那份不是真正在跑的那份。**

> 这是"**读代码不核对原文**"的第四次（前面三次：CT.2 读 `end_kh` 时、DD.3 里读
> Tile H 循环时、以及本附录之前这一次）。
> 每次代价都是整轮。
> **教训：涉及两个相似标识符的行，必须逐字符对照原始输出，不能凭印象转写。**
> DA.2 记过"参数名不是语义"，这里补一条：**符号名不是原文。**

### 定位路径（可复用）

1. **逐维二分**（DD.8）：把"一个组合红"变成"一类缺陷"
2. **逐维隔离**：`tile_n=1`（摘掉 N 方向）、`C=1`、`H=1`、`W=1`、全 1。
   结论：**跟 tile 维度无关，只要 tile != 全 1 就错；全 1 就对**
3. **反查值来源**：给输入每个元素填**唯一**的值（`(i+1)*0.001`），
   再从输出反查"它等于输入的第几个"。得到 `[0][0]→0  [0][1]→30  [1][0]→45  [1][1]→无`
4. **按偏移量反推**：0 正确、其余都偏了**恰好一个 `sliceStep`** ⇒ 指针推进用了错的步长

第 3 步是关键：**唯一值 + 反查**把"值错了"变成"值从哪来错了"，
这套手法对任何"数据被写错位置"的缺陷都适用。

### 现在的状态

* ASan 45/45、UBSan 45/45，全绿
* `g_known[]` 清空，**KNOWN_FAIL 机制保留**（下次再出现"已知未定位"的缺陷时直接放进去，
  而不用去动 `g_cases[]` —— 那会让缺陷到期清零）
* 可达性同前：`Tile` 在 `model/` 里 UNUSED，但 `ZQ_CNN_Net.h` 注册了这个层类型，**模型可达**

### 顺带记一条（未修，如实记下）

`ZQ_CNN_Layer_Tile::GetTopDim`：`top_C = bottom_C*tile_c` 是**无检查的 int 乘法**
（`tile_c` 是裸 atoi，与附录 BF 同类）；而且它**没有 `top_N`** ——
**`tile_n` 是死参数**：层不算它，`Tile` 却按 `N*tile_n` 去 `ChangeSize`，
模型写 `tile_n=2` 时顶层张量会被撑大一倍。

**判定不修**：改成"禁止 tile_n>1"或"让 GetTopDim 也算 N"都是猜；
`Tile` 的语义现在已经自洽（`out[r] = in[r % N]`），真正该问的是
**模型转换器会不会写出 `tile_n`** —— 那是仓外的问题，仓内没有证据。

## 新增/变更：附录 DX —— Tensor4D 基类覆盖探针 + `ROI` 边界检查的整数溢出（已修）

### 变更文件

* `ZQCNN/ZQ_CNN_Tensor4D.h` —— `ROI` 的边界检查（3 行）
* `tools/zq_roi_check.cpp`（新增）—— `ROI` 边界检查门禁（12 个用例）
* `tools/tensor4d_cov_probe.py`（新增）—— 基类方法覆盖探针
* `tools/run_zqlib_checks.py` 登记 `zq_roi`（四处）
* `audit_k3_20261001.md` 新增附录 DX

### 缺陷：`off_x + width` 是 int 加法，会回绕

    if (off_x < 0 || off_y < 0 || off_x + width > W || off_y + height > H)

`off_x + width` 是 **int 加法**。而 `off_x` 来自 MTCNN 的 P-net 检测框输出
（`ZQ_CNN_MTCNN.h:615 / 621` 等 12 处 `ROI(...)` 调用），**是数据/模型可控的**。
off_x 足够大时加法**回绕成负数** ⇒ `> W` 不成立 ⇒ **边界检查被整条绕过**，
紧接着 `src_slice_ptr = GetFirstPixelPtr() + off_y*widthStep + off_x*pixelStep`
就是一次**越界读**。

UBSan 坐实（修之前）：

    ZQ_CNN_Tensor4D.h:74:40: runtime error: signed integer overflow:
        2147483645 + 8 cannot be represented in type 'int'

修法（全程不产生可能溢出的加法）：

    if (off_x < 0 || off_y < 0 || width < 0 || height < 0)  return false;
    if (off_x > W || off_y > H)                                return false;
    if (width > W - off_x || height > H - off_y)               return false;

顺带把 `width < 0` / `height < 0` 也拒掉 —— 原来负的 width/height 会让
`off_x + width` 变小而**漏过**检查。

`ROI` 的其余部分逐项核过是**对的**：主拷贝循环按 pixelStep / dstPixelStep 分别推进、
补齐区 memset 为 0、三段 border memcpy 的起点与长度都能对上（右边 border 那段的
`-2*borderW*pixelStep + (h+1)*widthStep` 恒等于 `h*widthStep + width*pixelStep`，
因为 `widthStep = (width+2*borderW)*pixelStep`）。
**有问题的是"该不该拒"这一条，不是"算得对不对"。**

### 门禁 `zq_roi_check.cpp`（12 个用例）

判据只有一条：**这些输入必须被拒（返回 false），且不许有任何 sanitizer 报告。**
为什么不做数值比对：对"该被拒的输入"做数值比对没有意义 ——
进程能不能活着、报不报 sanitizer 才是判据。正常值的数值比对留给逐格门禁。

用例：正常值 5 个（整图 / 子块 / 贴边右边界恰好 / 单像素 / 带 border 顶左各 1）；
应被拒 7 个（正常越界、off_x == W、负 width、负 height、三个 int 加法回绕）。

变异测试（把修复回退成原来那一行）：

    **int 加法回绕** off_x = INT_MAX-2   没跑完（ASan/UBSan 报错并 _exit）（信号 1）
    **int 加法回绕** off_y = INT_MAX-2   没跑完（ASan/UBSan 报错并 _exit）（信号 1）
    共 12 个用例：全对 10，有错 0，崩溃/搭建失败 2

恢复修复后：ASan 12/12、UBSan 12/12 全过。注意"全对 10、有错 0"而失败记在
"崩溃"栏 —— 这正是附录 CJ.4 那条"结果文件读不出来 = 失败"守卫在起作用。

### 覆盖探针：这一整块的缺口有多大

`Tile` 查出两条真缺陷，说明**整个 `ZQ_CNN_Tensor4D` 基类**值得查 —— 而这一整块
（900 行头文件里几十个带循环、带 memcpy、带指针算术的虚函数）**此前零覆盖**。

新增 `tools/tensor4d_cov_probe.py`，列出基类里**有实体的方法**及
**是否被任何门禁碰过**，**只排优先级不下结论**：

    有实体的方法 21 个，其中 19 个**没有任何门禁碰过**（占实体行数 504/655 = 77%）
    其中实体 >= 5 行的「无门禁」方法：16 个

按优先级：ConvertColor_BGR2GRAY 57 / **ROI 53（本附录已查已修）** /
Reshape_NCHW 51 / ConvertFromBGR 36 / ConvertFromBGR2GRAY 35 / ConvertFromGray 34 /
SaveToFile 34 / Permute_NCHW 32 / ConvertToBGR 27 / FlipY 24 / FlipX 23 /
AddScalar 19 / MulScalar 19 / CopyData 18 / ConvertToCompactNCHW 17 / Flatten_NCHW 15。

### 探针自己出的一个错，以及它为什么值得记

第一版把 `if` / `while` 当成了**方法名**（`DEF_RE` 的返回类型那段正则太宽，
控制流关键字也匹配得上），于是表里出现了 `if` 这一行、"有实体的方法 30 个" ——
**一个工具自己输出了不存在的名字**。

**工具输出一个不存在的名字，比不输出更坏** —— 它会让人以为基类里有 9 个叫 `if`
的方法。这与 DA.2（grep 静默失败）、CU.8.1 / CX.5（门禁假绿）、
DD.9.4（我读错符号名）同源：**"看起来正常的输出"是最危险的失败模式**。
已加 `KEYWORDS` 过滤，并把工具自己的结论改成**从数据里算**而不是硬编码。

### 下一步

按探针优先级，下一个是 `Reshape_NCHW`（51 行）与 `Permute_NCHW`（32 行）——
两者都是**改形状**的，与 `Tile` / `ROI` 同属"指针算术 + 形状变换"这一族，
而这一族已经连续出了三条缺陷。


---

## 变更：附录 DY —— Reshape / Flatten 族（一次误判、一处真缺陷、两个自造的 harness 缺陷）

### 变更文件

| 文件 | 性质 |
|---|---|
| `ZQCNN/ZQ_CNN_Tensor4D.h` | **修生产缺陷**（越界读） |
| `ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h` | 同源拷贝，同样修 |
| `ZQCNN_to_MNN/converter/source/ZQ_CNN_Tensor4D.h` | 同源拷贝，同样修 |
| `tools/zq_reshape_check.cpp` | **新建门禁**（150 例） |
| `tools/run_zqlib_checks.py` | 登记门禁（四处）+ 修 `cp -n` 缺陷 |
| `tools/zq_check_child.h` | `"w"` → `"a"`，修报告被后一个用例擦掉 |
| `audit_k3_20261001.md` | 追加附录 DY |

### 缺陷 1（已修，生产代码）：`Reshape_NCHW_get_size` 越界读 `shape[i]`

循环上界写的是常量 `4`，而 `shape` 是调用方给的 `std::vector<int>`、**允许短于 4**：
`shape.size()==2` 时读 `shape[2]` / `shape[3]` 落在分配块之外。

ASan 坐实（`heap-buffer-overflow`，READ of size 4，修复前 `ZQ_CNN_Tensor4D.h:799`）：

    ==359977==ERROR: AddressSanitizer: heap-buffer-overflow ... READ of size 4
        #0 ZQ::ZQ_CNN_Tensor4D::Reshape_NCHW_get_size(...) ZQCNN/ZQ_CNN_Tensor4D.h:799
        #1 ZQ::ZQ_CNN_Tensor4D::Reshape_NCHW(...)            ZQCNN/ZQ_CNN_Tensor4D.h:822
    0x602000000058 is located 0 bytes to the right of 8-byte region

**修法**：`i < 4` → `i < shape_dim`。循环前已把 `i >= shape_dim` 的 `new_dim[i]` 全设成 1，
而 `total % 1` 恒真、`total /= 1` 是空操作 —— **那一段本来就是纯空操作，改上界不改变任何结果**。
三份同源拷贝一并修（主库 803 / NCHWC 490 / MNN converter 511，现行号）。

**可达性（不夸大）**：`Reshape` 层（`ZQ_CNN_Layer.h:8472`）与 `Flatten_NCHW` 都打不到
—— 前者总把 shape 补到 4 且全为正数，后者构造的 shape 也全为正数，都走 `unknown_num==0` 分支。
**可达的是公开 API** `ZQ_CNN_Forward_SSEUtils.h::Reshape`（调用方的 vector 原样透传）
与直接调 `ZQ_CNN_Tensor4D::Reshape_NCHW`。
所以这是**公开 API 上的越界读**，不是能让 shipped 模型跑错的那一类。
后果轻微（读到的字节只参与空操作）但确属 UB，修它一个 token。

### 缺陷 2（已修，我自己的 harness）：子进程 sanitizer 报告被擦掉

两个叠在一起的缺陷，都在 `tools/`（附录 CZ 那套机制）里：

1. **`tools/zq_check_child.h` 用 `"w"` 截断**。一道 fork 型门禁会 fork 几十个子进程
   （`zq_reshape` 50 个），它们**共用同一个 `ZQ_CHILD_ERR` 路径**，
   每个子进程一启动就截断，**排在崩溃用例后面的用例把崩溃用例的报告擦掉**。
   只在崩在中途时发作，崩在最后一个用例时报告恰好还在 —— 所以之前每次跑出来都"正常"。
2. **`tools/run_zqlib_checks.py` 用 `cp -n` 且本地 `%TEMP%/zqchild/` 从不清空**。
   第一轮拉回来的空报告一直挡在后面，后面每轮即使拉到真报告也覆盖不掉。
   WSL 侧 `$WDIR` 每轮开头 `rm -rf *`，所以旧的只存在于本地镜像目录里。

**症状极具欺骗性**：门禁确实红了（崩溃用例结果文件读不出来 → rc=1，判据 CJ.4 生效），
但报告栏永远空白，看上去像"根本没有 sanitizer 报告"。

**不改变任何门禁的判定** —— 子进程崩了 rc 就是 1，与 stderr 去哪无关。
只影响可诊断性。修好后整条链端到端可用：门禁红 → ASan 报告自动浮出 →
精确指到 `ZQ_CNN_Tensor4D.h:803` 并带完整栈。

### 新增门禁

`tools/zq_reshape_check.cpp` —— `Reshape_NCHW` / `Flatten_NCHW` 的形状 + 数值门禁，
**150 例全绿**（50 用例 × 3 个张量子类 + 42 个 NCHWC 用例）。
形态同 `zq_tile` / `zq_roi`（这两个函数写在成员函数里，绕不开真实张量对象）。

判据：恒等 reshape 逐格不变 / 跨形状对拍 / 输出尺寸对表 / `0`=沿用输入维 /
短 shape 合法 / 5 维·双 `-1`·乘积错必须被拒且输出缓冲不许被写 / 三种子类 /
另跑 NCHWC 那份同源拷贝的**静态** `get_size`。

门禁自己在脸上一条警告：**缺陷 1 只有 ASan 能抓到**（越界字节落在空操作上，
非 sanitizer 下结果完全正确），所以本门禁在 ASan 轴下才是完整判据。

**变异测试**：把 `i < shape_dim` 回退成 `i < 4`，门禁如期变红 ——
9 崩 = 3 个"短 shape + `-1`"用例 × 3 个子类，**其余 141 例仍全对**，
说明变异精准命中触发条件而非无差别破坏。

### 一条被实测否掉的"缺陷"（重要，不要再犯）

我曾把 `Reshape_NCHW` 里的 `i_c + i_w*in_PixelStep` 判成"漏乘 `i_c*in_PixelStep`"，
拿同仓 `ConvertFromCompactNCHW` 当对照物。**对照物选错了** ——
两者内存布局不同（NHWC 里通道是最内层，`w*ps + c` 本来就对；
compact NCHW 里通道在最外层才要乘步长）。用恒等 reshape 实测：四组 C=1/3/4/8 全部逐格不变，
**代码是对的，不改**。已转"已核对无缺陷"清单。

新记的经验：**拿同仓另一个函数当对照物之前，必须先确认它们的输入布局一致** ——
名字像、语义像，不代表内存布局像。

### 门禁表第一版有 5 个用例是红的，逐个复核后全部是**我的表算错了**

| 红项 | 我写的期望 | 真相 |
|---|---|---|
| `{0,1,0,1}` | 应当成立 | `2*1*4*1=8`，count 是 120 |
| `{2,3}/2` | 应当成立 | `2*3=6` |
| `Flatten(2,2)` | 1×2×12×1 | 把 H 乘成 H 自己，是**恒等** |
| `{1,2,3,0}` | 应当被拒 | `1*2*3*in_W(=4)=24=count`，**本来就成立** |
| `{0,0,0,3}` | 应当成立 | `1*2*3*3=18≠24` |

第 4 条最说明问题：**注释里都写了"→ 96"，却没发现 `in_W` 本身就是 4** ——
注释和用例算的是两套数。**不逐条复核就会把"我算错了"当成"产品有 5 个 bug"写进报告。**
该用例已留在成立组，并在注释里记下误判。

### 附带核对无缺陷（本轮）

`ConvertFromBGR` / `ConvertFromBGR2GRAY` / `ConvertColor_BGR2GRAY` 三处的 border 清零。
右侧 memset 起点写成 `firstPixelData - pixelStep*(borderW<<1) + widthStep*(h+1)`，
乍看像差了一个 borderW，但恒等式 `widthStep = pixelStep*(W + 2*borderW)` 使它
**恰好等于** `firstPixelData + widthStep*h + pixelStep*W`，正是右边界起点。
该恒等式对三个子类都成立（`widthStep = pixelStep*realW`，只是 `pixelStep`
分别是 `C` / `ceil(C/4)*4` / `ceil(C/8)*8`）。
**第一遍我把这个恒等式算错了、得出"错位一个 borderW"，重推才对。**

### 注意事项 / 未完成

- **MNN converter 那份拷贝覆盖不到**：它的头文件保护符（`_ZQ_CNN_TENSOR_4D_H_`）
  与类名都与主库完全相同，同一个 TU 里会静默地用第一份；实测它**在 Linux 上能编能调**，
  补覆盖需要给 harness 加"一门禁多源文件"的能力。本次**记为未完成，不假装覆盖了**。
- **NCHWC 头文件我原先判"Linux 上编不了"（`__int64` / `_aligned_free`）是错的**，
  实测 `-fsyntax-only` 干净通过。已加静态调用覆盖。
- **下一条线索尚未定性**：`ChangeSize` 的第 5/6 形参是 `(borderW, borderH)`，
  而 `ROI` 的形参是 `(dst_borderH, dst_borderW)`（**H 在前**）、
  `ResizeBilinearRect` 是 `(dst_borderW, dst_borderH)`（**W 在前**）——
  同一套 API 里两套相反约定，且 `ROI` / `ResizeBilinearRect` / `ConvertColor_BGR2GRAY`
  内部调 `ChangeSize` 时都按 `(H, W)` 传。已实测确认 `ChangeSize(1,4,5,3,2,3)` 的
  `widthStep == ps*(W+2*2)`（第 5 形参确实是 borderW）。**待补门禁再下结论。**


---

## 变更：附录 DY.7 / DY.8 / DY.9 —— 非对称 border 越界写（51 处）+ 两个 harness 缺陷

> **本条更正上一条 changelog 与报告 DY.7 的结论。**
> DY.7 写的是"borderW/borderH 顺序不一致是埋雷不是活雷、不改代码" —— **那是错的**，
> 错在只看了语义、没核对 memset 的边界。实际情况是**堆越界写**。

### 缺陷 3（已修，生产代码，51 处）：非对称 border ⇒ 堆越界写

`ChangeSize` 的第 5/6 形参是 `(borderW, borderH)`，而 `ROI` / `ResizeBilinearRect` /
`ConvertColor_BGR2GRAY` 内部都这么调：

    dst.ChangeSize(N, H, W, C, dst_borderH, dst_borderW)   // ← W/H 传反

于是张量**按转置后的 border 分配**，紧接着的 border 清零却**按未转置的形参名**
（`dstPixelStep*dst_borderW` 表水平、`dstWidthStep*dst_borderH` 表垂直）——
两边对不上，越界写。

ASan 坐实（三处，规律一致）：

    ==367469==ERROR: AddressSanitizer: heap-buffer-overflow ... WRITE of size 360
        #2 ZQ::ZQ_CNN_Tensor4D::ROI(...) ZQCNN/ZQ_CNN_Tensor4D.h:129
    0x617000000a80 is located 0 bytes to the right of 768-byte region

| 入口 | 对称 (1,1) | **borderH > borderW** | borderW > borderH |
|---|---|---|---|
| `ROI` | 正常 | **OVERFLOW** `Tensor4D.h:129` | 正常 |
| `ConvertColor_BGR2GRAY` | 正常 | **OVERFLOW** `Tensor4D.h:514` | 正常 |
| `ResizeBilinearRect` | 正常 | **OVERFLOW** `Tensor4D.cpp:447` | 正常 |

**溢出条件统一是 `dst_borderH > dst_borderW`。**

**修法**：改成 `ChangeSize(..., dst_borderW, dst_borderH)`，共 **51 处**：

| 文件 | 处数 |
|---|---|
| `ZQCNN/ZQ_CNN_Tensor4D.cpp` | 32 |
| `ZQCNN/ZQ_CNN_Tensor4D.h` | 3 |
| `ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp` | 15 |
| `ZQCNN_to_MNN/converter/source/ZQ_CNN_Tensor4D.h` | 1 |

覆盖 `ROI` / `ResizeBilinear` / `ResizeBilinearRect`（标量与 vector 两个重载）/
`ResizeNearest` / `Remap` / `ConvertColor_BGR2GRAY`，跨 align0/128/256 三个子类。

**改这个方向的理由**：三处的 memset 写法完全一致（`dstPixelStep*dst_borderW` 表水平、
`dstWidthStep*dst_borderH` 表垂直），**与 `ChangeSize` 的 `(borderW, borderH)` 一致**，
所以"按形参名分配"才是自洽的那一边；改完之后 `GetBorderW()` 也终于与形参名对上了。

**零行为变更（实测）**：全仓 25 处 `ROI` 全传 `(0,0)`、`ResizeBilinear*` 全传 `(0,0)`/`(-1,-1)`、
`ConvertColor_BGR2GRAY` 全传 `(1,1)` —— 全部对称。
对称用例结果与修复前**逐字节一致**（`(1,1)` 修复前后都是 `widthStep=18, sliceStep=108`）。

**自动化改代码的两个坑（都在 dry-run 阶段抓住，没进版本库）**：
1. 第一版脚本**把函数声明也匹配上了**，真应用下去就是改签名 —— 加"实参里不许有类型关键字"的护栏。
2. 第二版**判断条件写反了**，把本来正确的调用点当成要改的。
   应用后又机器核对：diff 是 **51 删 / 51 增，0 行不是纯 `borderW`/`borderH` 互换**。

### 门禁：`zq_roi` 扩成两段（19 → 29 例）

判据从 1 条加到 3 条：① 该拒的必须被拒（DX 原有）② `GetBorderW()/GetBorderH()`
必须等于**同名形参**（钉死 DY.9；没有它，"把两个参数交换回去"照样能过）
③ **整块 dst 逐格核对**：数据区 = 源 ROI，**border 一圈必须是 0**。

**四处变异测试全部被抓住**：ROI / ConvertColor / ResizeBilinearRect 的 vector 重载 /
标量重载。回退后分别出现"ASan 崩"（H>W 方向）与"GetBorderW/H 对不上"（W>H 方向）两类红。

**"没红"的第二种原因又撞上一次**：第一次变异打在标量重载（`.cpp:262`）时门禁**没红** ——
不是"与它无关"，是**门禁调的是 vector 重载、那个点没被覆盖**。补了标量重载用例后同一变异就红。
（附录 DA.2 立的规矩：没红必须区分**没测到 / 变异无效 / 真无关**，默认怀疑自己没测到。）

**覆盖边界（如实说）**：51 处里门禁直接打到的是 `ROI` / `ConvertColor_BGR2GRAY` /
`ResizeBilinearRect`（两个重载）在 align0 上的行为；其余变体（`ResizeNearest` / `Remap` /
align128 / align256 / NCHWC 那份 / MNN converter 那份）**靠"51 处是同一个机械替换 +
机器核对 diff 全是纯互换"来保证一致，没有逐点门禁**。

### 缺陷 4（已修，harness）：EXTRA_SOURCES 编译失败被报成链接错误

全量 33 道门禁跑出过一次 `zq_nchw_depthwise` 的
`BUILD FAIL: g++: error: /tmp/zqchecks/zq_dwnchw.o: No such file or directory`。
排查：单独编该源**成功**、单跑该门禁**PASS**、完整重跑 **33/33 通过** —— 没复现。
但**"失败时报不出真因"这件事本身是确定的缺陷**：`EXTRA_SOURCES` 的 gcc 原来是裸命令，
stderr 进总输出、`.o` 不生成、链接时才炸，报表只给链接器的话。

已改成把额外编译并进同一条判定，失败时把**所有**相关日志里的第一条 `error`/`fatal` 拼进消息；
**一条都没命中时退回日志第一行**（编译器报的未必含这两个词，否则消息为空 = 什么都没报）。
变异测试：把 `zq_reshape` 的一条额外源改成不存在的文件 ——
改前报 `g++: error: .../zq_reshape_bogus.o: No such file`（指向链接器），
改后报 `gcc: error: .../NO_SUCH_FILE_bogus.c: No such file`（**真因**）。

### 我自己在这道门禁上踩的三个坑（都是"门禁坏了长得像代码坏了"）

1. `(size_t)h * ws` 对负 `h` 回绕成 ~1.8e19 → 19 个用例**全部**报 ASan 越界，而被测代码没动。
2. `fscanf` 的 `%95[^\n]` 遇空串返回 3 而不是 4 → 每个用例被判"没跑完"，
   还被打上 **"sanitizer 报错"** 的标签，而 stderr 文件是 **0 字节**。
   **没有证据就断言原因，会把排查方向带偏。**
3. heredoc 里的 `\uXXXX` 改含中文的文件 → 截断汉字、写出非法 UTF-8，整个文件读不出来。

### 变更文件

| 文件 | 性质 |
|---|---|
| `ZQCNN/ZQ_CNN_Tensor4D.{h,cpp}` | **修生产缺陷**（51 处中的一部分） |
| `ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp` | 同上（15 处） |
| `ZQCNN_to_MNN/converter/source/ZQ_CNN_Tensor4D.h` | 同上（1 处） |
| `tools/zq_roi_check.cpp` | 门禁扩成两段（19→29 例）+ 改用共享 `zq_child_silence_stderr()` |
| `tools/run_zqlib_checks.py` | EXTRA_SOURCES 失败可见化 + 空消息退回第一行 |
| `audit_k3_20261001.md` | 追加 DY.7 / DY.8 / DY.9 |


---

## 变更：附录 DZ —— Convert 族：一处越界读 + 新门禁 zq_convert + harness 修复自带 bug

### 缺陷 5（已修，生产代码）：`ConvertToBGR` 未校验 `C >= 3`

函数无条件读 `cur_pix[0]` / `cur_pix[1]` / `cur_pix[2]`，而守卫只校验了 `W` / `H` / `n_id`。
`C=1` 时读 `cur_pix[2]` 就是越界。ASan 坐实：

    ERROR: AddressSanitizer: heap-buffer-overflow ... READ of size 4
        #0 ZQ::ZQ_CNN_Tensor4D::ConvertToBGR(...) ZQCNN/ZQ_CNN_Tensor4D.h:630
    0x606000000060 is located 0 bytes to the right of 64-byte region

**一个必须写下来的细节：只有 align0 会炸。** align128/align256 的 `pixelStep` 被补到 4/8，
读 `cur_pix[1]/[2]` 仍落在同一个像素内，所以不越界 ——
**"另外两个变体没报"不等于"没问题"，只代表那处内存恰好还在。**

修法：`if (C < 3) return false;`。三份同源拷贝一并修
（`ZQCNN/ZQ_CNN_Tensor4D.h:608` / `ZQ_CNN_Tensor4D_NCHWC.h:268` /
`ZQCNN_to_MNN/converter/source/ZQ_CNN_Tensor4D.h:317`）。

该函数**全仓零调用点**，但按 AGENTS.md 已修订的判据
（"是不是内存安全问题"优先于"生产可不可达"），越界读必须堵。

### 新门禁 `zq_convert`（54 例，全绿）

判据用**精确往返**，参考值**不依赖任何一条实现公式**。
第一版想逐格写 `期望值 = (b - 127.5f) * 0.0078125f`，写到一半发现那是在
**把实现的公式抄一遍**（抄错了门禁和实现一起错，看着全绿其实什么都没验）。
改用：`ConvertFromBGR` 与 `ConvertToBGR` 互为逆运算且**逐位精确**
（b=0/100/128/255 四点验算见报告），判据就是"喂进去的字节必须原样回来"；
反向把张量填成 `b/127.5f - 1.0f`，必须吐回 `b`。

覆盖 `ConvertFromBGR`↔`ConvertToBGR`、`ConvertFromGray`、`ConvertFromBGR2GRAY`、
`ConvertToCompactNCHW`↔`ConvertFromCompactNCHW`、`ConvertColor_BGR2GRAY`（含 border 一圈为 0）、
**`C < 3` 必须被拒**；三种张量子类各跑一遍。
变异测试：去掉守卫 → 红（align0 三例被 ASan 打死 + align128/256 六例"应拒却收下"）。

### 已读并核对无缺陷（本轮）

`ConvertFromBGR` 36 / `ConvertFromBGR2GRAY` 35 / `ConvertFromGray` 34 / `SaveToFile` 34 /
`CopyData` 18 / `FlipX` 23 / `FlipY` 24 / `Permute_NCHW` 32 / `AddScalar` 19 / `MulScalar` 19。
`Permute_NCHW` 的步数分解每轮 `idx %= new_steps[j]` 后已天然落在下一维范围，**不需要额外取模**。

### 门禁自己踩的五个坑（全部是"门禁坏了长得像代码坏了"）

第一版 18 例红，逐条独立复核后 **9 例是真缺陷、9 例是我门禁自己的错**：

1. 灰度参考按三通道算，但 `ConvertFromGray` 的指针是 `gray_pix++`（**步长 1**），
   而 `ConvertFromBGR2GRAY` 是 `bgr_pix += 3`。**名字像、参数像，语义完全不同**
   —— 附录 DY.1 的同一条教训，在同一个文件里隔 130 行又撞一次。
2. `ConvertColor_BGR2GRAY` 的数据检查读的是**源**张量而不是目标张量。
3. compact 往返按整条 slice 比对（含补齐区），而补齐区本该是 0。
4. **"整张图是不是常数"这个判据方向写反了**：写成"只要有一个像素与首像素不同就 `bad++`"，
   而那恰恰是正常情况。于是 align128/256 上 4 例全红、align0 恰好全同反而"过了" ——
   **一个判据错误伪造出"某些变体有问题"的信号**。当时若急着"修被测代码"，
   就会去改一个没有问题的内核（与附录 AY.5 的"恒定相对误差 1.00"同类）。
5. `fabs` 忘了 `#include <cmath>`。

### 我给 harness 做的 DY.8 修复，**自己带了两个 bug**

写完 DY.8 紧接着写新门禁，第一次 BUILD FAIL 消息**又是空的** —— DY.8 那次改动自己带的：

1. **日志名后缀重复**：`2> %s.build.log` 格式串已带后缀，我又传了一个以 `.build.log`
   结尾的名字 → 实际写出 `zq_convert.build.log.build.log`，取内容时读到的是空文件。
2. **命令替换里 `||` 绑错位置**：`$(cat … | grep -m1 … | tr -d '\r' || head -1 …)`
   的 `||` 绑在管道最后一条命令上，退出码取自 `tr`（恒 0），`head -1` 永不执行。
   改成显式赋值 `M=$(…grep…); [ -n "$M" ] || M=$(…head -1…)`。

两个都是**变异测试**抓出来的，不是读代码看出来的 ——
读代码时这两个 bug 看起来都很合理。变异验证（两条路径都试了）：

    BUILD FAIL: gcc: error: /mnt/d/ZQCNN/ZQCNN/NO_SUCH_bogus2.c: No such file or directory
    BUILD FAIL: /mnt/d/ZQCNN/tools/zq_convert_check.cpp:54:1: error: expected unqualified-id before 'this'

> **新规矩：修复本身要单独验一次。**
> DY.5（报告被擦掉）、DY.8（失败报成链接错误）、DZ.4（修复自带 bug）
> 三个都是"改了之后**以为**好了、其实没有"的连续实例，
> 全部靠**故意弄坏**发现，不是靠读代码发现。已写进 AGENTS.md。

### 变更文件

| 文件 | 性质 |
|---|---|
| `ZQCNN/ZQ_CNN_Tensor4D.h` | **修生产缺陷**（`ConvertToBGR` 加 `C<3` 守卫） |
| `ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h` | 同源拷贝，同样修 |
| `ZQCNN_to_MNN/converter/source/ZQ_CNN_Tensor4D.h` | 同源拷贝，同样修 |
| `tools/zq_convert_check.cpp` | **新建门禁**（54 例） |
| `tools/run_zqlib_checks.py` | 登记门禁（四处）+ 修 DY.8 遗留的两个 bug |
| `audit_k3_20261001.md` | 追加附录 DZ |


---

## 变更：附录 EA —— NCHWC 张量类自有方法的门禁从零建起来（本轮未在该类查出新缺陷）

### 新门禁 `zq_nchwc_tensor`（123 例，全绿 + 变异测试通过）

`tools/zq_nchwc_tensor_check.cpp`，模板遍历 `NCHWC1/4/8` 三个子类，41 用例 × 3 子类。

**缺口**：`ZQ_CNN_Tensor4D_NCHWC` 被 9 道门禁引用（act / bn / conv / conv8 / depthwise /
elt_relu / ip / pool / resize），但那些**只把它当数据容器**用（`ChangeSize` + 取首指针）。
它自己的 Convert 族 / Permute / Flatten / Reshape **一个门禁都没有** ——
与附录 DX.6 在基类上发现的缺口同一个形状，而基类那批方法两轮里出了两条真缺陷
（DY.2 越界读、DY.9 border 传反）。

> 覆盖探针的局限：**"有门禁引用这个类"不等于"这个类的方法被测过"**。
> 只按"文件被引用"统计会把这块报成"已覆盖"。

**判据原则**
1. 恒等变换必须逐格不变（DY.1 靠这条否掉过一个误判）
2. **参考不依赖实现的路径** —— Reshape/Permute/Flatten 的实现都走
   "转 compact NCHW -> 在 compact 上算 -> 转回来"，参考若也走 compact 就是拿实现对照实现。
   参考改成按布局公式自己取 compact，再在 compact 上用独立下标算术算期望
3. **短 shape + `-1`**（DY.2 触发条件）必须常驻
4. **`C < 3` 时 `ConvertToBGR` 必须被拒**（DZ.1）

**变异测试有鉴别力**：把 `ZQ_CNN_Tensor4D_NCHWC.h` 里 DY.2 的修复回退
（`i < shape_dim` → `i < 4`），门禁变红且 ASan 正确指到
`ZQ::ZQ_CNN_Tensor4D_NCHWC::Reshape_NCHW` 的 heap-buffer-overflow；恢复后全绿。

### 门禁自己踩的坑：NCHWC 的 `sliceStep` 不是"一个通道"

ASan 第一次就把 `fill_unique` 打穿了，栈顶在测试文件里（AGENTS.md：先怀疑测试）。
真因：元素偏移我写成 `n*imStep + c*sliceStep + h*widthStep + w*align`，
但 NCHWC 的 `sliceStep` 步进的是**一个通道组（`align_size` 个通道）**，
正确的是 `(c/align)*sliceStep + (c%align)`。

这与 AGENTS.md「NCHW 与 NCHWC 的步长语义」一节、附录 BN.2 是**同一条知识的第三个体现**，
而我刚写完那段文档就又踩了一次。三处层次不同：BN.2 是内核用错、AGENTS.md 记成文档、
EA 是**门禁自己用错** —— 门禁用错比内核用错更隐蔽，
因为参照系本身就是错的时，本该抓到的缺陷永远抓不到。

### 本轮在该类未查出新缺陷（如实说边界）

门禁 123 例全绿 + 变异有鉴别力，**但这只等于"这批方法在这些输入下与独立参考一致"**。
未覆盖：三个子类的 `Padding` / `ROI` / `CopyData` / `Resize*` / `Swap` / `ShrinkToFit`；
非对称 border 组合（`Reshape_NCHW` 固定传 0,0 调 ChangeSize）；
`SaveToFile`；`ConvertFromBGR` 的非默认 `mean_val` / `scale`。

其中**非对称 border 最值得下一轮补**：DY.9 修的 51 处里有 15 处在
`ZQ_CNN_Tensor4D_NCHWC.cpp`，目前只有机器核对 diff 保证一致，**没有逐点门禁**。

### 变更文件

| 文件 | 性质 |
|---|---|
| `tools/zq_nchwc_tensor_check.cpp` | **新建门禁**（123 例） |
| `tools/run_zqlib_checks.py` | 登记门禁（四处），含 NCHWC resize 内核的链接依赖 |
| `audit_k3_20261001.md` | 追加附录 EA |


### 补：NCHWC border 路径的逐点门禁（EA.5 / EA.6）

`zq_nchwc_tensor` 加第二段，**156 例**（52 × 3 子类）全绿，
补上 EA.4 点名的最大缺口 —— DY.9 修的 15 处 NCHWC 传参此前只有机器核对 diff 保证一致。

覆盖三个子类的 `ResizeBilinearRect`（**标量与 vector 两个重载**）、`ROI`、`CopyData`，
边框取 `(1,1)` / `(1,3)` / `(3,1)`，其中两种非对称。
判据与 `zq_roi` 第二段同源：① 返回 true ② `GetBorderW()/GetBorderH()` 必须等于**同名形参**
③ border 一圈必须是 0。注意 NCHWC 里**两套相反约定并存**：
`ResizeBilinear*` 是 `(dst_borderW, dst_borderH)`，`ROI` 是 `(dst_borderH, dst_borderW)`。

**两处变异测试，逐子类验证鉴别力**：

| 回退的点 | 门禁反应 |
|---|---|
| `NCHWC1::ResizeBilinearRect` 的 `ChangeSize` 传参 | 红：**只有 NCHWC1** 红（1 崩 + 1 几何不符），NCHWC4/8 仍 52/52 |
| `NCHWC4::ResizeBilinearRect` 的 `ChangeSize` 传参 | 红：**只有 NCHWC4** 红，NCHWC1/8 不受影响 |

**逐子类验证是必须的**：第一次变异只改"第一处出现的位置"，
红的是 NCHWC4 而不是 NCHWC1 —— 因为 NCHWC1 那处用的是另一种写法
（`__max(0,dst_borderW)` 无空格），字面量不同。
**"打中一个子类"不等于"三个子类都有鉴别力"** —— 与 DY.4「没红的三种原因」同条，
只是方向反过来：**红了也不等于全面覆盖**。

> 可复用手法：想知道"哪些站点归哪个函数"，用正则按
> `bool ZQ_CNN_Tensor4D_NCHWC(\d)::(\w+)\(` 的出现位置切段，
> 再把每个 `ChangeSize(...)` 站点映射到它前面最近的函数头。
> 我第一次就是靠"猜第一处属于谁"猜错了，改成脚本映射之后一眼看清。

**门禁自己踩的第二个坑**：`CopyData` 恒等用例第一版写成 `t.CopyData(t)` ——
源和目的是同一个对象，而 `CopyData` 内部会 `ChangeSize`，6 例全红。
真因在门禁。改成 `o.CopyData(t)` 后全绿。
自拷贝本身是不是缺陷**没查**（三个子类的 `ChangeSize` 在参数完全相同时会提前返回、
不重分配，所以大概率安全，但没证据就不下结论），记为未查。


---

## 变更：附录 EB —— 并发运行会互相摧毁（harness 第四个"失败时报不出真因"）

### 现象与定位

最终回归跑到 B 阶段「ZQlib 独立回归测试 x10」时报 `zq_bns  BUILD FAIL:`（消息为空），
而 `zq_bns` **单跑 PASS**。

定位过程：
1. 单跑 `run_zqlib_checks.py zq_bns` → PASS
2. `run_audit_checks.py:100` 显示 B 阶段**就是把本脚本当子进程调起**
3. 我在这期间自己手工跑了好几次 `run_zqlib_checks.py`（查新门禁）
4. 两者共用固定 `WDIR=/tmp/zqchecks`，脚本开头 `cd $WDIR && rm -rf *`
   ⇒ **一次运行正在链接时，另一次把它刚编出来的 `.o` 和日志全删了**

### 机制坐实（先复现再修）

改回共享目录复刻场景：全量 33 道跑到一半插一次单门禁

    -> zq_nchw_act  BUILD FAIL: g++: error: /tmp/zqchecks/zq_nact_relu.o: No such file or directory
    -> 34/35 通过, FULL_EXIT=1

报出来的是**链接错误**而非真因 —— 日志也被对方那次 `rm -rf *` 删了，
`cat | grep error` 什么也没 grep 到。与附录 DY.8 同一机制的两个出口。

**第一次尝试没复现**：两个单门禁并发跑（各 ~10 秒）都 PASS，窗口太窄。
必须"全量 + 中途插一次"才撞得上 —— **"没复现"不能当成"不是这个原因"**（与 DY.4 同条）。

### 修法与前后对照

`WDIR` 改成每轮唯一（`/tmp/zqchecks_<pid>_<时间戳>`），本地镜像目录同理；
跑完删掉**本轮自己的** `$WDIR`（`rm -rf /tmp/zqchecks*` 是错的，会删掉正在跑的别的运行）。

顺带修掉**修法本身带出的两个 bug**（今天第三次"修复自带 bug"）：
1. `cd $WDIR && rm -rf * && mkdir -p $WDIR` —— `cd` 到**还不存在**的目录会失败，
   `&&` 把后面的 `mkdir` 短路掉，于是所有编译写到不存在的路径上，
   **每道门禁都 BUILD FAIL 且消息为空**。改成 `mkdir -p $WDIR && cd $WDIR && rm -rf ./*`。
2. 拉回报告时源目录写死 `/tmp/zqchecks` —— 会把**别的运行**的报告也一起拷进来。

同一场景前后对照：

| | 结果 |
|---|---|
| 共享目录（修复前） | `zq_nchw_act BUILD FAIL`，**34/35 通过，EXIT=1** |
| 唯一目录（修复后） | **35/35 通过，EXIT=0** |

### 为什么单独立附录

本会话 harness 已出过四个"失败时报不出真因"的缺陷：
DY.5（子进程共用 stderr 文件被截断 / `cp -n` + 本地目录不清）、
DY.8（额外源失败报成链接错误）、**EB.1（并发运行互删工作目录）**。
共同点：**报出来的东西比真相更像真相**。
判据 CJ.4 让它们全表现为"门禁红"，于是人会去查**被测代码** ——
而真因在"用来测它的东西"里。

> 可直接照做的规矩：**跑回归期间不要并行跑同一套 harness 的任何一部分。**
> 光靠"记得别并发"不可靠 —— 正确做法是让 harness 本身对并发免疫。

### 变更文件

| 文件 | 性质 |
|---|---|
| `tools/run_zqlib_checks.py` | WDIR/镜像目录按运行唯一 + 先 mkdir 再 cd + 只删自己的目录 |
| `audit_k3_20261001.md` | 追加附录 EB |


---

## 最终回归结果（2026-10-03，附录 DY / DZ / EA / EB 全部落地后）

一次**干净**的完整回归（全程未并行跑任何 harness —— 附录 EB.1 的教训）：

    python tools/run_audit_checks.py --with-build --warn-sweep --src-sweep \
        --bounds-sweep --ubsan-sweep --reachability --msvc-asan
    python tools/run_zqlib_checks.py --with-slow

| | 结果 |
|---|---|
| 审计阶段（16 个 A* 门禁 + 6 条扫描轴 + 双平台构建 + sample 回归 + MSVC ASan） | **AUDIT_EXIT=0**，全阶段 OK |
| 慢门禁 | **43/43 通过，SLOW_EXIT=0**，0 个 FAIL |

含本轮新建/扩充的五道门禁，全部 PASS：

| 门禁 | 用例数 | 覆盖 |
|---|---|---|
| `zq_reshape` | 150 | `Reshape_NCHW` / `Flatten_NCHW` + NCHWC 同源拷贝的静态 `get_size` |
| `zq_roi` | 29 | `ROI` 的边界检查 + **非对称 border 几何**（含 3 个子类的 `ResizeBilinearRect` / `ConvertColor_BGR2GRAY`） |
| `zq_convert` | 54 | Convert 族（精确往返判据）+ `C<3` 必须被拒 |
| `zq_nchwc_tensor` | 156 | NCHWC 张量类自有方法（Convert 族 / Permute / Flatten / Reshape）+ **子类 border 路径逐点覆盖** |
| `zq_tile` | 45 | `Tile`（附录 DD，本轮未改动，作为回归基线保留） |

### 一次"不算数"的回归，以及它换来的东西

第一次跑最终回归时审计阶段报了 `zq_bns  BUILD FAIL`（消息为空），而 `zq_bns` 单跑 PASS。
追下去发现是**我自己**造成的：审计 harness 的 B 阶段就是把 `run_zqlib_checks.py`
当子进程调起，而我在它跑到 B 阶段时自己也跑了好几次单门禁 ——
两者共用固定的 `/tmp/zqchecks` 且开头 `rm -rf *`，**互相删对方的 `.o` 和日志**。

这条查成了附录 **EB**，并且**先复现、再修、再对照**：

| | 结果 |
|---|---|
| 共享目录（修复前） | `zq_nchw_act BUILD FAIL`，**34/35 通过，EXIT=1** |
| 唯一目录（修复后） | **35/35 通过，EXIT=0** |

**这就是"不算数"的那次回归的价值** —— 如果当时把 `zq_bns` 当成真缺陷去查，
会浪费大量时间在一条根本不存在的问题上；而如果当时**随手当噪声放过**，
这个 harness 缺陷会一直留着，下次再坑一次。

### 本轮提交

| commit | 内容 |
|---|---|
| `cb71d35` | 附录 DY：Reshape 误判推翻 + 越界读已修 + `zq_reshape` 门禁 + harness 两处修好 |
| `67200a7` | 附录 DY.9：非对称 border 堆越界写已修 51 处 + `zq_roi` 扩到 29 例 + 四处变异测试 |
| `ff0044f` | AGENTS.md：补 2026-10-03 的四条硬规矩 |
| `d0b4644` | 附录 DZ：`ConvertToBGR` 的 `C<3` 越界读已修 + `zq_convert` 门禁 + harness 修复自带 bug |
| `c51a2a6` | 附录 EA：NCHWC 张量类自有方法门禁（123 例） |
| `1995771` | 附录 EA 补：NCHWC border 路径逐点门禁（156 例）+ 逐子类变异测试 |
| `2fd22b3` | 附录 EB：harness 并发互删工作目录已修 + 机制坐实 |
| `98660c8` | 审计报告：加"结论更正索引"章节 |
