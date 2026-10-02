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
