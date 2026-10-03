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


---

## 变更：附录 EC —— UNUSED 层类型的接线门禁（75 例）

### 切片来源

威胁模型写明**模型文件是不可信输入**，而可达性门禁说 36 种层类型里 **15 种 UNUSED**
（不会被随仓库发布的模型跑到）。两者放在一起：**UNUSED 不等于不可达** ——
攻击者构造的 `.zqparams` 可以引用它们。这 15 种此前既没有门禁、也没有 sample 跑过；
而已有的 43 道门禁测的是**内核**，测不到"层有没有把参数接对"。

### 先逐个读了 6 个 UNUSED 层：都已被加固，或是自洽的桩

- `LRN`：`local_size % 2 != 1` 已守住 0 / 偶数 / 负数
- `Reduction`：`axis < 0 || axis >= 4` 已守住（我早前修的 CT 那条）
- `PriorBox` / `PriorBox_MXNET`：`step_w <= 0` 退回 `1.0f/layer_width`，
  除零那族（BD/BE/BF）已收口
- `Squeeze` / `Copy`：**自洽的桩**（只 `CopyData`，`dim` 读了不用，
  但 `GetTopDim` 也原样返回，声明与行为一致）—— 这正解释了它们为什么 UNUSED
- `ScalarOperation`：`DIV` 走 `Mul(1.0f/scalar)`，`scalar == 0` 得 inf；
  浮点除零不崩，数值可疑但不是内存问题

"低垂果实已经摘干净"本身值得记：前几轮的加固覆盖面比报告统计表看起来更广。

### 新门禁 `zq_layerwire`（75 例，全绿 + 变异通过）

第一版想编 `ZQ_CNN_Forward_SSEUtils.cpp` 让层调真内核，**结果链接期缺半个库**
（addbias / prelu / avgpooling / batchnorm / conv / conv_gemm …），
最后拖进 `zq_cnn_convolution_gemm_32f_align_c.c`（单编 5 分钟以上），只能挂进 SLOW。

**改成把桩下在更下面一层**：`ZQ_CNN_Forward_SSEUtils` 的公开包装方法
（`ScalarOperation_*` / `Reduction*` / `LRN_across_channels`）全部是**头文件 inline**，
定义在 .cpp 里的是**小写内部辅助函数** `_scalaroperation_*` / `_reduction_*` / `_lrn_*`。
于是：层 →（inline 包装，真实）→ **我的桩（只记录不计算）**。

两个好处：① 不用拖半个库；② 门禁**不可能与实现同源**（桩里没有一行计算），
而"跑一遍真内核再对拍"会被"层和内核一起算错"掩盖过去 —— 那正是要防的错。

判据五条：输入指针必须在 `bottoms[0]` 缓冲里；输出必须在 `tops[0]` 里（就地做除外）；
**落到哪个 helper** 与**变换后的标量**要对；非法参数必须在**到达内核之前**被层拒掉；
bottom 一个字节都不许被改。

**变异测试**：把 `SCALAR_DIV` 的两处 `1.0f/scalar` 改回 `scalar`（DIV 退化成 MUL）→ 红；
恢复后全绿且 `git diff --numstat ZQCNN/ZQ_CNN_Layer.h` 为空。门禁有鉴别力。

### 门禁自己踩的三个坑（每一个都长得像"被测代码坏了"）

1. **判据"标量必须原样传下去"是错的** —— 层对 `DIV` 故意改写成 `Mul(1.0f/scalar)`、
   对 `MINUS` 改写成 `Add(-scalar)`，那是**等价变换，改对了才对**。6 例红，真因在判据。
   改成"落到哪个 helper + 变换后的标量"后全绿，而且判据**更强** ——
   它把操作映射本身也钉住了。
2. **桩下错了层**：`ZQ_CNN_Forward_SSEUtils` 是**类**不是命名空间；
   改对之后又给公开包装写类外定义（它们已是头文件 inline）→ `redefinition`。
   两次都是看编译器第一行报错定位的。
3. **我自己引入的 harness 改动毁掉了诊断证据** —— EB 里加的"跑完删掉本轮工作目录"
   在门禁编译失败时连 `*.build.log` 一起删了。已改成**只在没有构建失败时清理**。
   **清理本身会毁掉证据**，与 DY.5 同一类。

### 覆盖边界（如实记下）

在范围内：`ScalarOperation`（9 op + 未知）、`Reduction`（4 合法 + 2 非法 axis）、
`LRN`（2 合法 + 3 非法 `local_size`）、`Squeeze`、`Copy`。

不在范围内：`Sqrt`/`Scale`/`ScaleWithBias`（纯头文件 inline，没有可下的桩点）；
另外 9 个 UNUSED 层（大多需要 net 提供权重 blob，直接驱动要额外搭架子）；
12 个 EXERCISED 层的接线 —— sample 能发现"结果不对"，但发现不了
"用了错误的张量而结果恰好一样"（就地做 vs 拷贝到 top 在 bottom==top 时完全等价）。


---

## 变更：附录 ED —— 零尺寸张量空指针解引用（已修）+ 一次差点改坏随仓库模型的修复（已撤回）

### 缺陷 6（已修）：`UnaryOperation` 无条件解引用零尺寸张量的首指针

`ZQ_CNN_Layer_UnaryOperation` **名不副实** —— 它要**两个 bottom**，
第二个是逐元素张量，第一个只被读成**一个标量**：
`float scalar = (*bottoms)[0]->GetFirstPixelPtr()[0];`
而 `ReadParam` / `LayerSetup` **只查指针非空、不查尺寸**。

**关键事实（实测）**：`ChangeSize` 对**任意一维为 0** 的张量**返回成功**，
并把 `firstPixelData` 置成 **0**：

    C=0 / H=0 / W=0 / N=0   ChangeSize(...) -> 1   GetFirstPixelPtr() = (nil)

于是"零尺寸的第一个 bottom"就是**空指针解引用**。门禁里两个用例当场复现（子进程崩）。

修法：解引用之前先确认四个维度都为正。

### 差点改坏随仓库模型的一次修复（**已撤回**）

顺藤摸瓜看到 `ZQ_CNN_Layer_Input` 的构造函数是 `H(0), W(0), C(3)`，
而 `ReadParam` 的返回值只看 `has_C && has_name` —— **H/W 完全不要求**。
当时推断"模型写 `Input C=3 name=n` 就会造出零尺寸张量"，于是**改了这一行**。

**然后按 AGENTS.md「改完要抓基线再对照」数了一下影响面**：

    省略 H 或 W 的 Input 行：1 条，涉及 1 个模型
       model\det1.zqparams        Input 		name=data  C=3

**这会改坏一个随仓库模型。** 继续查才发现更根本的：`MTCNN::SetPara` 只设包装器自己的
width/height、**不写 net 的 Input 层**；真正的图像是通过 `ConvertFromBGR` 写进 blob 0 的，
而那会重新 `ChangeSize`。**零尺寸状态是瞬态的，中间没人解引用它**
（`Input::Forward` 是空的 `return true`，Image 在 Forward 之前才写进去）。

**所以"缺陷 2"不存在，修复已撤回。** 只保留 `C > 0`：`C` 是必填项、
`C=0`/`C<0` 无任何合法用途、随仓库模型全是 `C=1` 或 `C=3`。

**这一条比修掉的那个缺陷更有价值**：

> **看到"未校验的参数"不等于"缺陷"。** 判断依据必须是三样：
> ① 下游兜住没有 ② **调用方依赖这个默认值没有**
> ③ 有没有一份"同仓的第二处实现"给出的相反约定。
> `has_H_val` 在 net 里只被用来"有 InnerProduct 才强制" —— 这就是 ② 的直接反证，
> 而我是在**改完之后**才去查的。顺序应该反过来：**先查影响面，再改。**

已补进 AGENTS.md 的「修改生产代码前先数影响面」。

### 门禁现在钉的是**真实契约**

撤回之后，`Input` 的用例改成钉**实际成立**的那份：
`Input C=3 name=n`（省略 H/W）**必须仍被接受**（`det1.zqparams` 依赖它）、
`C=0` / `C<0` 必须被拒。**将来谁把 H/W 变必填，这道门禁会红** ——
把差点犯的错误变成永久防线。

### 两处变异测试

| 回退 | 门禁反应 |
|---|---|
| 去掉 `UnaryOperation` 的尺寸守卫 | 红（子进程崩，空指针解引用） |
| 去掉 `Input` 的 `C > 0` | 红 |

恢复后全绿。

### 这一轮我自己的三次操作失误（都当场抓住）

1. Python 按行范围替换切多了，留下残码（`L was not declared in this scope`）。
   编译器只说"标识符没声明"、不说是哪来的 —— 直接看那几行才定位得到。
2. **heredoc 里的 `\\n` 变成真换行**，写出断行的 `fprintf` 字符串。
   AGENTS.md 早就写了"改源码和写脚本一律用 Write/Edit 工具落盘"——
   **这次又用 heredoc 改 C++，同一条规则第二次被违反**。
3. 门禁的 `Input` 参数行一开始写成 `Input top=t bottom=b C=3`，
   而这个层**不接受 `top`/`bottom` 键**，日志里刷 `unknown para top`。
   **测试用例必须按真实语法写**，否则测的是另一条路径。


---

## 变更：附录 EE —— ZQ_CNN_Tensor4D 的就地运算门禁（48 例）

从覆盖探针看到的真实缺口（EC 之后重跑 `tools/tensor4d_cov_probe.py`）：

| | DX 时 | EE 之前 |
|---|---|---|
| 有实体但**无门禁**的方法 | 19 / 21 | **8 / 21** |
| 占实体行数 | 504/655 = 77% | **129/684 = 19%** |

剩下实体 >= 5 行的是 `SaveToFile 34` / `FlipY 24` / `FlipX 23` / `AddScalar 19` /
`MulScalar 19`。`SaveToFile` 每次回归都写文件、对拍还要跟磁盘状态较劲，
不适合进默认门禁；**剩下这四个是纯就地运算、无外部依赖、判据能写得很硬** —— 本轮收掉。

### 新门禁 `zq_tensorop`（48 例，全绿 + 两处变异通过）

16 用例 × 3 个张量子类。四条判据：

1. 逐格对拍，参考是**自己按步长公式取的整块缓冲**，不复用实现的循环
   （`FlipX: out[n][h][w][c] == in[n][h][W-1-w][c]`，其余三个同理）
2. **做两次回到原样**（FlipX/FlipY 是对合；Add 用 `-s`、Mul 用 `1/s` 抵消）——
   抓"只翻了数据区的一半"这类错
3. **border 一圈必须原封不动**：这四个方法都只遍历 `n<H, w<W, c<C`，
   也就是**只动数据区**。这是有意的（padding 不该被翻/被改），钉住它，
   免得以后有人"顺手"把循环边界写大
4. **align128/align256 的补齐区也必须原封不动**

**两处变异都通过**：
- `FlipX` 的 `W-1-w` 改成 `W-w`（越界一格）→ 红
- `AddScalar` 的通道上界 `C` 改成 `C+1`（会动补齐区）→ 红

第二个变异专门验判据 3/4 —— 如果"补齐区必须原封不动"这条没写，
`C -> C+1` 就会被放过去。**判据要对着"最可能被顺手改坏的那一处"设计。**

### 这道门禁上我自己连踩三次坐标/符号坑

1. `(size_t)h * widthStep` 对负 `h` 回绕成 ~1.8e19 -> 48 例全报 ASan 越界，
   而被测代码一个字节没动。**本会话第二次踩同一个坑**（第一次在 zq_roi_check.cpp）。
2. 快照按 `firstPixelData` 起算、只申请 `N*sliceStep`，可 border 那一圈
   （负偏移）根本落在向量之外 —— **参照系本身就是错的**。
3. 换到快照坐标系时把 `base_off` 的符号搞反 -> 48 例全崩。

**①和③的症状完全不同（"全崩" vs "全数据错"），根因却是同一个** ——
同一个根因会伪装成不同症状，不能靠症状认根因。
做法：把坐标换算收敛成 `off()` / `base_off()` / `bidx()` 三个函数，
每个注释写清坐标系原点和符号。已补进 AGENTS.md。

### 顺带再记一次覆盖探针的局限

探针统计的是"**这个方法名在哪些门禁里出现过**"。
`ConvertFromCompactNCHW` 显示被 12 道门禁碰过 —— 但那些门禁只是**调用它**，
没有一条断言它的数值正确。**"被调用"与"被验证"是两件事。**


---

## 记录：EE 之后的全量非慢门禁结果

新增 `zq_tensorop`（48 例）后，全量非慢门禁由 36 增至 **37 道，全部通过**。

慢门禁（`zq_gemm_shape` / `zq_nchwc_conv` / `zq_nchw_conv` / `zq_innerproduct` /
`zq_facedb*` 那几个重头）当时仍在后台跑，其结果只覆盖到 **EC** 状态；
**ED 对 `ZQ_CNN_Layer.h` 的生产改动需要另跑一次 `--with-build` 的双平台构建
+ sample 回归**来验证 —— 未跑之前不算验过。


---

## 变更：附录 EF —— ZQlibFaceID 文件读入计数缺上界（同仓已有一处正确写法被漏掉）

### 切片来源：整个 ZQlibFaceID 的覆盖缺口

| | 数量 |
|---|---|
| `ZQlibFaceID/*.h` | 29 |
| **其中没有任何门禁提到** | **26** |
| 仅前 18 个就合计 | ~5585 行 |

有门禁的只有 3 个（都在 `zq_facedb` / `zq_facedb2` 那两道慢门禁里）。
这个缺口比张量类那次大得多；多数是应用层封装（识别器、视频聚类、人脸库管理），
需要 net 提供权重 blob 才能驱动，不是一轮能收完的量。
**本轮只做一件有明确判据的事：把"文件读入的计数驱动分配"这一族系统性过一遍。**

### 一条系统性扫描，以及它**第一次报错了**

威胁模型里"人脸库文件（.imgfeat 等）"是不可信输入，而 `ZQlibFaceID` 里有大量
`fread(&count, sizeof(int), 1, in)` 之后直接 `resize(count)` 的写法。
扫描找出所有"由单次 fread 读入的 int 驱动 resize/reserve/new[]"的点并判上界。

**第一次跑出来的结论是错的**：它把 `ZQ_FaceGroup.h:44` 判成"完全没查"，
而第 45 行就有 `num >= 0 && num < 1000000`。原因是我的正则只认 `var > N`，
**漏掉了 `var < N` 这种上界写法**。改正后（`<`/`>` 都认）：13 处里**只有 1 处**缺上界。

> 扫描给出的"权威结论"（带表格、带计数、看着可信）**是错的**，
> 错因是**正则漏了一种写法**。凡是靠正则判"有没有上界"的工具，
> **两种写法都要认** —— 否则"0 处有问题"和"N 处有问题"都不可信。

### 缺陷（已修）：`num = 0x7FFFFFFF` 时申请 8 GB，无 catch → terminate

`ZQ_FaceClusterImagesForVideo.h:170` 只挡 `num < 0`，紧接着
`offset.resize(num)` / `length.resize(num)`。

**同仓已有正确写法**：`ZQ_FaceContainerForVideo.h:84-100` 面对**完全同一个问题**
（`key_num` 驱动 `frames.resize()`），那里的注释把失效模式写得很清楚：
"key_num 来自不可信文件, 只挡负数不够: 0x7FFFFFFF 会让 frames.resize() 直接 OOM
(未捕获的 bad_alloc -> terminate)。用剩余文件长度做上界交叉校验"。
**一份修了一处没修 —— 按 AGENTS.md 第 13 条，这不是设计取舍，是遗漏。**

**失效机制（隔离复现，不复现那个类本身）**：
- 不限内存（Linux 默认 overcommit）：`resize` 成功（8 GB 虚拟），随后 `fread` 失败 → 返回 false
- `ulimit -v 1GB`（无 overcommit / cgroup / 32 位进程）：**抛出 `std::bad_alloc`**

而这条加载路径**没有 catch**，异常一路冒到 `std::terminate()` → abort。
**内存受限的部署上，4 字节的文件就能让进程崩掉。**

**修法与 `ZQ_FaceContainerForVideo.h` 逐字一致**：用剩余文件长度做上界交叉校验
（每个条目至少要装下自己的 `offset` 与 `length` 两个 int，共 8 字节）。

### 顺带清掉一处重复代码

`ZQ_FaceContainerForVideo.h` 里那个 `if (rest_len > 0 && key_num*4 > rest_len)`
**连续出现了两遍**，是前一轮修复被应用了两次留下的。已去掉，逻辑不变。

### 验证级别（如实说，不夸大）

`ZQ_FaceClusterImagesForVideo.h:8` 是 `#include <opencv2\opencv.hpp>`（**反斜杠**），
**在 Linux 上编不过**；而且全仓**没有任何文件 include 这个头**（连 sample 都没有），
所以 Windows 构建也不会编译它 —— 改动前后都**没有现成门禁覆盖**。

按 AGENTS.md 第 5 条补了最小编译验证：
- 取巧：Linux 允许文件名含反斜杠，造一个名字就叫 `opencv2\opencv.hpp` 的转发头；
  再加一个只提供 `__int64` / `__min` / `__max` 的垫片（**不进仓库**）。
- **改动前后各编一遍**（改动前取 `git show HEAD:` 的版本作对照）：**两边都 rc=0、无输出**。
- 尝试行为验证：**链接缺 OpenCV 符号，做不了**。

| | |
|---|---|
| 编译验证（改动前/后对照） | 通过 |
| 失效机制隔离复现 | 通过（两种内存条件都实测） |
| **端到端跑这个类** | **做不到** —— Linux 链接缺 OpenCV；Windows 侧无 sample 编译它 |
| 回归保护 | **无门禁** |

**所以这一条是"编译验证 + 机制验证"，不是"行为验证"。**

### 方法论：grep 到可疑读入点后要连着读后面 20 行

我一度以为 `ZQ_FaceClustersForVideo.h:106` 的 `video_frames` 也是缺陷
（`fread` 之后完全没有校验）。读完整个函数才发现它只被**写**、从不当循环上界或下标 ——
实际用的是单独读入且有 `0..10000000` 边界的 `fr_num`。
另一次我 grep 到 `ZQ_FaceContainerForVideo.h:78` 的 `key_num < 0` 就判"只挡负数"，
而守卫在 10 行之后、被一段注释隔开 —— **差点把一处已经修好的地方报成缺陷。**


---

## 变更：附录 EG —— ZQlibFaceID 里两个头在 Linux 上根本编不过（已修）

### 直接命中目标的一条：跨平台可编译性

报告开头的目标是"确保 windows 和 linux 都能完全跑通"。
`ZQlibFaceID` 里有两个头的 include 让它们**在 Linux 上永远编不过**：

    #include <opencv2\opencv.hpp>        // ← 反斜杠

gcc/clang 不会把 `opencv2\opencv.hpp` 当目录分隔（只有 MSVC 会）。

**全仓扫描反斜杠 include** 结果分两类：

| 位置 | 数量 | 有没有平台守卫 |
|---|---|---|
| `SamplesZQlibFaceID/*/*.cpp` 的 `<openblas\cblas.h>` / `<mkl\mkl.h>` | 数十处 | **有** —— 全在 `#if defined(_WIN32)` 块里，Linux 构建看不到，**无害** |
| `ZQlibFaceID/ZQ_FaceClusterImagesForVideo.h:8` | 1 | **没有** |
| `ZQlibFaceID/ZQ_FaceIDPrecisionEvaluation.h:8` | 1 | **没有** |

**只有头文件里那两处是真问题。** 也顺带解释了 EF.0 的覆盖缺口：
`ZQlibFaceID` 26/29 个头没有门禁，**其中一部分不是"没人写门禁"，
而是"根本编不过、写不了"**。

> 顺便记一次**扫描工具自己报错**（今天第 N 次）：我用 Python 写的那版扫描报 **0 处**，
> 而直接 grep 找到几十处 —— 正则里的反斜杠在 `python -c "..."`（shell 双引号）里被转义吃掉了。
> **同一件事 grep 对、Python 错**，而 Python 那版还带着表格和计数，看着更权威。
> 凡是"正则很复杂"的扫描，**必须用 grep 交叉验一次**。

### 修法（已修，两处各 8 行）

除反斜杠外，这两个头还有第二道障碍：用了 `__min` / `__max` / `__int64`
（自身 10 处，且它 include 的第三方头 `ZQ_MathBase.h` 内部也在用）。

**`ZQCNN/ZQ_CNN_CompileConfig.h:97/101/105` 已经为非 MSVC 提供了这三个的可移植定义**
（`#ifndef` 包着，MSVC 上不覆盖原生版本）。所以修法是**在两个头的最前面**补上
`#include "ZQ_CNN_CompileConfig.h"`，再把反斜杠换成正斜杠。

### 验证

| | |
|---|---|
| **Linux/gcc 编译两个头** | **通过**（改动前报 `__min` / `__int64` 未声明） |
| **反向对照** | 去掉新增那行 include → 重新出现 error，证明这一步必要 |
| **Windows/MSVC 编译** | **本机测不了** —— `jpeglib.h` 在整台机器上都不存在，而这两个头 include 了 `ZQ_JpegEncoder.h`（无条件 `#include "jpeglib.h"`）。Linux 侧是靠系统的 libjpeg-dev 才通过的 |
| **会不会破坏 Windows 构建** | **不会** —— **全仓没有任何 .cpp include 这两个头**（连 sample 都没有），它们根本不在 Windows 的编译图里 |

**所以这一条是"Linux 侧修好并验证 + Windows 侧论证不受影响"，不是"双平台都实测过"。**

### 顺带确认：改动过程中又踩了 AGENTS.md 已有的两个坑

1. **Python 批量改写源码忘了补行尾**（AGENTS.md 行尾规则第 6 条）：插入的 7 行是 LF，
   混进 CRLF 文件，`check_line_endings.py` 报 `mixed-EOL(260 CRLF/7 LF)`。
   已 `--fix` 并复验（267 CRLF / 0 LF）。
   这条规则**当天已经踩过一次**，说明光"记得"不够 ——
   改完必须**立刻跑 `check_line_endings.py`**，而不是等到提交前。
2. **bat 文件用 heredoc 写成 LF**，`cmd` 直接不认
   （`系统找不到指定的路径` / `'cl' 不是内部或外部命令`）。
   改成用 Python 显式写 CRLF 才跑通。

### 这一条对覆盖缺口的意义

修完之后这两个头**可以进 Linux 编译图了** —— 也就有了给它们写门禁的前提。
EF 里那条"编译验证 + 机制验证、但没有行为验证"的状态，现在可以往"有门禁"推进：
`ZQ_FaceClusterImagesForVideo::LoadFromFile` 的 `num` 上界修复**从此可以被门禁覆盖**。
还差两步：① 链接需要 OpenCV/jpeg 的 .so（本机只有 Windows 的 .lib）；
② 该头没有被任何 .cpp include，要写门禁得由门禁自己 include（可行）。

### EG 补记：实测确认「给 `ZQ_FaceClusterImagesForVideo` 写门禁」在本机不可行

EG.5 说"还差链接 OpenCV/jpeg 的 .so"，这一轮实测确认：

    g++ ... dbg_faceclu.cpp -o fcl -ljpeg
    /usr/bin/ld: cvstd.hpp:648: undefined reference to `cv::String::deallocate()'
                         cvstd.hpp:656: undefined reference to `cv::String::deallocate()'
    collect2: error: ld returned 1 exit status

`libjpeg.so` 在 Linux 上是有的（`/usr/lib/x86_64-linux-gnu/libjpeg.so.8`），
缺的只有 **OpenCV 的 `.so`** —— 仓库里只有 Windows 的 `3rdparty/opencv/build/*.lib`。
缺的符号是 `cv::Mat` 成员的内联析构被发射出来的，**绕不过去**（除非自己在门禁里
伪造一个 `cv::String::deallocate()`，那是拿假实现换绿灯，不做）。

**结论**：EG 的修复解锁了这两个头在 Linux 上的**可编译性**，
但**不足以**支撑行为门禁。EF.2 那条 `num` 上界修复目前仍是
「编译验证 + 失效机制隔离验证」，**没有行为门禁**。如实记着，不假装覆盖了。

---

## 变更：附录 EH —— ZQlibFaceID 可编译性门禁（29 个"从来没被编译过"的头）

### 缺口

| | |
|---|---|
| `ZQlibFaceID/*.h` | 29 个 |
| **没有任何门禁提到** | **26 个** |
| 不在 Windows 构建里 / 不在 Linux 构建里 | 都是 |

**这个目录的绝大部分代码从来没被任何编译器检查过。** EG 那两个"Linux 上编不过"的头
就是这么撞出来的 —— 纯靠运气。`tools/probe_zqlib_headers.py` 已经覆盖了
`3rdparty/include/ZQlib/`（143 个头，118 个可编译），但没覆盖 `ZQlibFaceID/`。

### 新门禁 `tools/probe_faceid_headers.py`

逐个头在 Linux 上 `g++ -fsyntax-only` 编一遍，分四类：
`OK` / `NEEDS_LIB`（缺外部依赖，**不是缺陷**）/ `MSVC_ONLY`（MSVC 专有写法，**是缺陷**）/
`BROKEN`（其它编译错误，**是缺陷**）。
`--save-baseline` / `--check-baseline`：任何一个头从 OK 变成非 OK 就退出 1。
已接进 `run_audit_checks.py` 的 **C1** 组（与 C 组的 ZQlib 探针并列）。

### 当前结果

    OK 22 / NEEDS_LIB 7 / MSVC_ONLY 0 / BROKEN 0 / UNKNOWN 0   （共 29）

7 个 NEEDS_LIB 被外部 SDK 挡住（ncnn 的 `nn`、SeetaFace 的 `seeta`）。
**0 MSVC_ONLY / 0 BROKEN** —— EG 修好之后，凡是没被外部 SDK 挡住的 ZQlibFaceID 头
现在都能在 Linux 上编过。

### 变异测试（有鉴别力）

| | OK 数 | 那两个头 |
|---|---|---|
| 修复后（当前） | **22** | `OK` |
| 回退到 EG 之前 | **20** | `NEEDS_LIB (opencv2\`)` |

> **第一次做这个变异测试时对照组取错了**：`HEAD` 是只改 changelog 的提交、
> `HEAD~1` 才是 EG 修复那个提交，"回退"根本没回退，探针照样报 OK 22。
> 第二次补了一行验证（打印各 revision 的 `CompileConfig` 出现次数，确认 `HEAD~2`
> 才是修复前）才拿到 20 vs 22。
> **对照组必须先证明自己与实验组不同**，否则"没变化"什么都不能说明。

### 写这个工具时我自己犯了五次错，三个会让结论方向相反

1. include 路径漏了头文件自己所在的目录（`ZQlibFaceID/`）→ 29 个头全报"找不到"，
   而那张表看着像"全军覆没"；
2. `printf` 里的 `\n` 经过"Python -> 文件 -> 喂给 bash"两层被吃掉变成真换行；
3. `tr -d '\r'` 被吃掉成 `tr -d ''`（空参数）；
4. **结果解析 `if len(parts) < 4: continue`，而失败时只有 3 段** ——
   所有编译失败的头被静默丢掉，表上显示"探针没回来说这一条"，
   差点被读成"这些头没问题"；
5. **取 `head -1` 拿到的是 gcc 的 `In file included from ...` 引导行**，
   真正的 error 在第 2 行 —— 这会把"缺外部 SDK"误判成 `BROKEN`（= 真缺陷），
   **方向正好反了**。

> 通则：**一个"分类/统计"型工具，在被信任之前必须先证明它能把一个已知样本分对类。**
> 这与 AGENTS.md 早就写着的「"一条都没报"的检查工具必须自带自测」是同一条 ——
> 那次说的是"漏报"，这次是"**误分类**"，而且误分类方向可以任意。

### 能做什么、不能做什么（如实说）

**能做**：锁住"跨平台可编译性"不再退步（EG 那两个头修好之后不会再坏）。

**不能做**：
- 不检查数值/内存安全。`ZQlibFaceID` 里 7 个 NEEDS_LIB 的头与全部 22 个 OK 的头，
  **至今没有任何行为门禁**（EF.2 那条 `num` 上界修复也是）；
- 只覆盖"能不能编过"。能编过不代表对；
- 那 7 个 NEEDS_LIB 的头**连编译都没被检查**，要检查得先把 ncnn / SeetaFace 接进来。

**所以：`ZQlibFaceID` 的行为覆盖缺口一步都没缩小**，
这一轮只是把"能不能编过"这一维从"完全没人管"变成"有门禁盯着"。

---

## 变更：附录 EI —— ZQlibFaceID 第一道能真正在 Linux 上跑的行为门禁

### 为什么只能是这两个类

EH 修完之后 22 个头能在 Linux 上**编过**，但**"能编过"不等于"能链接"** ——
本机只有 Windows 的 OpenCV / ncnn / SeetaFace 库。多数头链不过；
`ZQ_FaceGroup.h` / `ZQ_FaceSearchTarget.h` 只依赖
`ZQ_FaceFeature` / `ZQ_CNN_BBox` / `<vector>` / `<stdio.h>`，**不需要任何外部库**，
是 `ZQlibFaceID` 里唯一能真正跑行为门禁的地方 ——
而它们恰好是"人脸库文件不可信"威胁模型下的解析入口。

### 新门禁 `zq_facegroup`（12 例，全绿）

判据：① 往返一致（`WriteToFile` → `LoadFromFile` 逐字段比对）
② 恶意 `feat_dim` / `num` 被拒（守卫 `0<=feat_dim<65535`、`0<=num<1000000`，边界内外都打）
③ 截断文件被拒
④ **失败时文件流位置必须回到调用前**（`LoadFromFile` 开头记 `pos`、失败时 `fseek(pos)`；
调用方靠它才能在一个流里连续解析多个 group）—— **这一条只有测试钉得住**：
读到 `fseek` 不等于它在所有失败路径上都执行了
⑤ `num==0` 空组必须成功

### 变异测试：精准命中判据 4

删掉失败路径里的那两行 `fseek`（**只删这两行**，其余解析逻辑不动）：

    共 12 个用例：全对 11，有错 1，崩溃/搭建失败 0
      ?    **失败后文件流位置没回退**

**恰好一条**、消息正是判据 4 —— 门禁钉的是**那个契约**，不是"这一带随便坏点都能发现"。

> 变异测试本身的两个坑：第一版替换范围**太大**（把整段解析逻辑都折叠了），
> 门禁照样变红，但那只证明"能检测这一带的任何破坏"；
> 第二版因为 `#ifdef _WIN64` 那行**顶格没有缩进**而模式匹配 0 处。
> **变异要"小到只碰被验证的那一处"。**

### 写这道门禁时我自己犯了三个错（都是"门禁坏了长得像代码坏了"）

1. **双 free**：`ZQ_FaceFeature` 析构函数会 `free(pData)`，我的 `free_group` 又手动 free
   了一遍 —— 子进程直接崩，父进程只看到"结果文件读不出来"；
2. **文件头写错**：`with_box` 我按 `sizeof(int)` 写，代码用的是 **`sizeof(bool)`（1 字节）**
   ⇒ 文件头错位，后面读出来的 `feat_dim` 全是垃圾（日志里 `feat_dim = 318899360` 就是证据）；
3. **期望反了**：我把 `num = 1000000` 标成"恰好在界内、应当被收下"，
   而守卫是 `num < 1000000`，**1000000 本来就该被拒** ——
   **差点把一处正确的守卫报成缺陷**（与 EF 那次同类：我把实现当成了规格）。

### 顺带修掉我在附录 EB 引入、却没验到的两个 harness 缺陷

EB 把工作目录改成每轮唯一（`/tmp/zqchecks_<pid>_<ts>`），**但我漏改了两处仍写死
`/tmp/zqchecks/` 的路径**：

| 位置 | 后果 |
|---|---|
| `ce = '/tmp/zqchecks/' + tag + '.child.err'` | 子进程 stderr 落到没人读的目录 |
| `cat /tmp/zqchecks/<tag>.out` | **失败明细永远读不到** —— 输出只剩一行标题后面空白 |

而且"跑完清理 WDIR"又踩了一次顺序坑：清理只判 `not build_fail`，
而**门禁运行失败（不是构建失败）时 `build_fail` 仍为空**，
于是目录被删掉、紧接着的明细 `cat` 读到空。
已改成**清理放在最后、且只在 `nfail == 0 且 build_fail` 为空时**才做。

> 这是本会话自己写进 AGENTS.md 的「修复本身要单独验一次」的第一个实例：
> EB 当时做了变异测试（验的是 BUILD FAIL 那条路径），**没验明细这条路径**，
> 两个漏网的地方又活了十几轮。**要逐条路径都问一遍"这个改动影响到哪些地方"。**

### 覆盖边界（如实说）

覆盖了：`ZQ_FaceGroup` / `WithBox` / `WithoutBox` 的往返、恶意参数拒绝、截断拒绝、
**流位置回退契约**。
没覆盖：`ZQ_FaceSearchTarget` 那一层（守卫读代码看着对但没门禁钉住）；
`ZQ_FaceContainerForVideo` / `ZQ_FaceClustersForVideo` / `ZQ_FaceDatabase*` / `ZQ_FaceRecognizer*`
—— 链不过（外部 SDK），仍只有编译验证。
**`ZQlibFaceID` 的行为覆盖缺口依然远大于零**，这一轮只是把**唯一能测的两个类**测起来了。

验证：全量非慢门禁 **38/38 通过**（新增 zq_facegroup）。

---

## 变更：附录 EI.7 —— 补上 ZQ_FaceSearchTarget 那一层（19 例）

EI.6 自己点名的缺口，这一轮补上：`ZQ_FaceSearchTarget::SaveToFile` / `LoadFromFile`
的往返、`num == 0` 空表、三个恶意 `num`、**截断后 `targets` 必须被清空**、文件不存在。

**两份实现的 `num` 上界差一，必须分别读**：

| 类 | 守卫 | `num = 1000000` |
|---|---|---|
| `ZQ_FaceGroup` | `num >= 0 && num < 1000000` | **拒绝** |
| `ZQ_FaceSearchTarget` | `num < 0 \|\| num > 1000000` | **接受**（在界内） |

我第一版按 `ZQ_FaceGroup` 的口径给 `ZQ_FaceSearchTarget` 写期望 ——
这是 EI.4 第 3 条同一个错误的**反向版本**：上次是"把实现当规格"，
这次是**把其中一份实现的边界套到另一份上**。

**变异测试，以及一处"没红"的诚实交代**

| 变异 | 结果 |
|---|---|
| 去掉 `num < 0` 检查 | **红**：`ST num=-1` 被信号打死（`resize(-1)`） |
| `num` 上界 `1000000` -> `2000000` | **没红** —— 而这是**正确的** |

第二个变异没红，说明**我的用例对 `num` 的上界没有鉴别力**：
那个文件只写了 `num`、后面没有任何内容，所以无论守卫松紧，
"截断"都会让 `LoadFromFile` 返回 false —— 判据分不出是守卫拒的还是截断拒的。
要让上界可判，文件得有足够内容满足 `num` 个 target（数 GB，不可行）。
**所以 `num` 上界这条判据目前是"看起来覆盖了、其实没有"** —— 如实记下。

**本轮我自己第四次"门禁坏了"**：ST 分支开头就 `fclose(f)`，
后面 `wr_int(f, num)` 往**已关闭的 `FILE*`** 写 —— 4 个用例子进程全崩。
隔离复现（单独跑 `num=-1` 的库调用）证明**库本身完全正常**，是门禁的锅。
这已经是同一个门禁文件上的第四个了（双 free / sizeof(bool) / 期望反 / fclose 顺序），
**每一个的症状都长得像"被测代码坏了"**。

验证：全量非慢门禁 **38/38 通过**。

---

## 变更：附录 EJ —— zq_layerwire 扩到 123 例（Sqrt / Tile / Scale 三个 UNUSED 层）

EC 覆盖了 7 个 UNUSED 层；这一轮补上**依赖是 public 成员、可直接驱动**的三个：

| 层 | 接线风险点 |
|---|---|
| `Sqrt` | 应当**先 CopyData(bottom→top) 再就地开方 top**，不是就地开方 bottom |
| `Tile` | `tile_n`/`tile_h`/`tile_w`/`tile_c` **四个参数挨着**，传错一个整体错位 |
| `Scale` / `ScaleWithBias` | 应当作用在 **top**（copy 之后）而不是 bottom；scale/bias 是**逐通道**张量 |

判据仍是"逐格对拍 + **独立公式** + bottom 一个字节都不许被改 + 输出形状符合声明"。

**变异测试：把 Tile 转发里的 `tile_n` 与 `tile_h` 对调** → 恰好 3 条 Tile 用例变红。

> **第一次做这个变异时它没红**，原因值得记：我的 Tile 用例是
> `tile_n = tile_h = tile_w = a` —— **三个倍数相等**，把 n 和 h 换掉结果一模一样。
> **"变异没红"的第一种原因又出现了：用例取值让这处变异不可观测。**
> 改成 `n = a, h = a+1, w = 1`（互不相同）之后，同一个变异立刻被抓住。

**本轮我自己又犯了两个"门禁坏了"的错**（症状依旧是"被测代码坏了"）：

1. **双 free（第三次撞上这一类）**：`ZQ_CNN_Layer_Scale` 析构里
   `if (scale) delete scale; if (bias) delete bias;` —— **层接管了这两个张量**，
   我又手动 delete 一遍。ASan 报 heap-use-after-free，读点落在我自己的判断代码上。
2. **桩的语义没对齐**：`Scale`（不带 bias）调的也是 `_scalebias`，
   但 **bias 传的是 NULL**（`ZQ_CNN_Forward_SSEUtils.h:2221`），
   我的桩无条件解引用 `bias[c]` => 不带 bias 的那两条直接崩。
   另外我第一版把非 Tile 三个层的输出倍数写成 `? c.a : c.N`，形状判据恒假。

**一个顺带的收获**：`Scale` 这条崩了之后，是 **EI 里刚修好的"失败明细路径"**
把子进程的 ASan 报告完整打了出来（栈顶直接指向我自己的第 306 行）——
那两处路径在修好之前，这条只会显示成"没跑完"。**同一个修复的第二次兑现**。

**覆盖边界**：`Scale`/`Sqrt`/`Tile` 现在有门禁；
`DeConvolution` / `BatchNorm` / `LSTM_TF` / `PriorBoxText` / `DetectionOutput_MXNET`
仍只有编译验证（需要 net 权重 blob 或更多初始化）。

验证：全量非慢门禁 **38/38 通过**。

---

## 变更：附录 EK —— zq_layerwire 再扩：BatchNorm（132 例）

`ZQ_CNN_Layer_BatchNorm` 的 `b` / `a` 是 public 张量，helper 是 `_batchnorm_b_a`
（语义 `value = b*value + a`，逐通道），可以直接驱动。
判据：逐格对拍（b=2, a=0.5 => 期望 x*2+0.5）+ bottom 未被改 + **b==0 时层必须自己拒**。

**变异测试：把 `BatchNorm_b_a` 的 `b` 与 `a` 对调** -> 每种张量变体 2 条用例变红。

> **消歧**：这段字面量在 `ZQ_CNN_Layer.h` 里有**两处** ——
> `ZQ_CNN_Layer_BatchNormScale`（2077）和 `ZQ_CNN_Layer_BatchNorm`（2431）。
> `assert count == 1` 抛错是对的；用脚本把每处映射到它**最近的 class**，
> 确认第二处才是 BatchNorm 之后才动手。
> **"同字面量多处"必须先消歧再替换**，否则很容易改到**另一个类**去。

**又一次"判据错"，以及一次"消息文案误导"**

1. `BN_NOB`（b==0 应被拒）红了报"返回值与期望相反" —— **层的行为是对的**：
   它 `if (b == 0 || a == 0) return false;`（2426 行）确实拒了。
   真因是**我新增的这组分支根本没看 `expect_kernel` 字段**。
2. 这一组分支复用旧分支的 note 文案，数据对拍不过时显示成
   **"输入指针不在 bottoms[0] 里"** —— 驴唇不对马嘴。已新增 note=9
   `**数据错（逐格对拍不过）**`，并把形状不符改成 note=6（参数被层改传），
   因为**参数顺序传错**才是这一组真正要抓的东西。

**这两条合起来是同一个病**：新增判据时只顾着"会不会红"，没顾着
"红了之后那句话说的是不是真的"。**判据的文案也是判据的一部分** ——
它会直接决定下一个读日志的人往哪个方向查。

**为什么 DeConvolution 不做**：它的 Forward 要 filters / bias / prelu_slope 三个张量
加十几个 int 参数，而要钉住"参数有没有传错顺序"就得**把整个反卷积内核重新实现一遍**
（或链接单编 5 分钟以上的 conv GEMM TU）。**为一个接线测试付这个代价不划算**，
且报告里已记了它的几条已知问题，记为**明确不做**并说明理由。

**UNUSED 层覆盖现状**（15 个）：已覆盖 11 个
（ScalarOperation / Reduction / LRN / Squeeze / Copy / UnaryOperation / Input /
Sqrt / Tile / Scale / BatchNorm），未覆盖 4 个
（DeConvolution / LSTM_TF / PriorBoxText / DetectionOutput_MXNET）。

验证：全量非慢门禁 **38/38 通过**。

---

## 变更：附录 EL —— 模型权重加载面（72 个 LoadBinary_NCHW）的扫描与一个负结果

### 为什么查这里

威胁模型把**模型文件（.nchwbin）**列为不可信输入，而它的解析入口
`ZQ_CNN_Layer::LoadBinary_NCHW` 在 `ZQ_CNN_Layer.h` 里有 **72 个实现**、**40 处 fread**，
而**门禁对它们零覆盖**。把 EF 那次的一次性脚本做成工具
`tools/check_filecount_bounds.py`（扫"文件读入的 int 驱动内存分配、有没有上界"这一族）。

### 结论：`ZQ_CNN_Layer.h` 里这一族缺陷**不存在**（负结果，但有用）

`LoadBinary_NCHW` 的形状是：

    int dst_len = filters->GetN() * filters->GetH() * filters->GetW() * filters->GetC();
    if (dst_len <= 0) return false;
    std::vector<float> nchw_raw(dst_len);
    if (dst_len != fread_s(&nchw_raw[0], dst_len*sizeof(float), sizeof(float), dst_len, in))

**它不从文件里读长度** —— 长度来自**已分配好的张量的维度**。
全文件统计：`fread(&X, sizeof(int), ...)` **0 处**；`int dst_len =` 63 处；
`if (dst_len <= 0)` 117 处。
所以**模型文件无法直接控制分配大小**（只能通过控制维度间接影响）。

### 但查出两件值得记的事

1. **`dst_len` 没有任何显式上界**（`grep -cE "dst_len > [0-9]"` = 0），只有 `<= 0`。
   间接上界来自 `ZQ_CNN_Tensor4D::ChangeSize`：`dst_tensor_raw_size > 0x7FFFFFFF`
   时 `return false`（`ZQ_CNN_Tensor4D.cpp:186-188`），于是单层权重被限在 ~2 GB。
   **"没有显式上界、只有别处兜着"比"没有上界"更难查** ——
   读 `LoadBinary_NCHW` 本身完全看不出这个约束在哪。
2. **`ZQ_CNN_Net.h` 里 `catch(` 出现 0 次**，而层加载点
   （`ZQ_CNN_Net.h:1136` / `1185`）直接调 `LoadBinary_NCHW`。
   真撞上 `bad_alloc` 时异常一路冒到 `std::terminate()`，与 EF 那条同一种失效模式。

**这两条都不作为缺陷修**（理由写进报告）：
第 1 条的兜底是 `ChangeSize` 的既有契约，改它会影响所有张量分配；
第 2 条要真正防住得引入"全局内存预算"，那是**设计变更**而不是补一个守卫。
记为"模型文件很大 ⇒ 进程可能 abort"这一**已知边界**。

### 工具本身：写了六版才对，每一次都是"看起来很权威的错误输出"

| # | 症状 | 真因 |
|---|---|---|
| 1 | 跑 ZQCNN 报"0 处" | `re.search` 少了 `ctx` 参数；拿已知阳性样本一验就暴露 |
| 2 | 把 `num < 0 \|\| num > 1000000` 判成"没上界" | 模板里的 `\b%s` **从未被 `% var` 替换**，匹配的是字面量 `%s` |
| 3 | 把 `num < 0` 判成"上界 0 = 有界" | `< 0` 是**负数检查**不是上界 —— **恰好把 EF 修掉的那处报成 OK** |
| 4 | 漏判"守卫写在十几行注释之后" | 上下文用了固定 20 行窗口 |
| 5 | 匹配到**隔壁函数**的 resize | 找分配也用了整个函数 |
| 6 | 删掉守卫后**仍然报 OK** | 匹配的是"提到了 rest_len 这个词"，而不是"rest_len 真的参与了判断" |

第 3、6 条最危险：它们让工具对它**被造出来抓的那类缺陷完全失明**，
而输出还带着表格和计数，看着比真相更像真相。

> **固化**：一个"分类/统计"型工具，**在被信任之前必须先证明它能把一个已知样本分对类** ——
> 而且**必须用"变异"去证**，不是跑一遍看输出顺眼。
> 第 6 条就是"跑一遍看着对、其实没验"：我第一次变异只删了 `if`、留下了计算 rest_len
> 的那段代码，工具照样报 OK；**是那次不完整的变异把工具的漏洞又藏了一轮**，
> 直到把**整块**删掉才暴露。**验证工具的变异，也必须先确认它自己变异到位了。**

**工具定位：分诊辅助，不进回归门禁**（有已知局限：只看一种形状，窗口是启发式的）。
真正的门禁仍然是行为门禁（`zq_facegroup` 那一类）。

---

## 变更：附录 EM —— 卷积 `dilate * (kernel - 1)` 整数溢出（模型文件可控，已定位未修）

### 缺陷

`ZQ_CNN_Layer.h:1287-1288`（`ZQ_CNN_Layer_Convolution::GetTopDim`）：

    int dilate_filter_H = dilate_H * (kernel_H - 1) + 1;

`kernel_H/W` 与 `dilate_H/W` 都由 `ReadParam` 从**模型文件**直接 `atoi` 读入
（`kernel_size=N` 同时设 kernel_H/W，`dilate=N` 同时设 dilate_H/W）。
唯一值域校验是上一轮加的 BE.2 那条，**只挡 `<= 0`、没有上界**，
于是 `kernel_size=2000000000 dilate=2000000000` 会被放行。

### 实测：表达式确实溢出，且回绕成**正数**

同式同类型（`int`）单独跑，编译器直接给了 `[-Woverflow]`：

    warning: integer overflow in expression of type 'int' results in '643460096' [-Woverflow]
    filt = 643460097        （原始 int 溢出）

**回绕成正数**这一点很要紧：它绕过了下游的负值检查 ——
`top_H = max(0, floor((8 - 643460097)/1) + 1) = 0`。

### 可能的连锁（只到"自洽"，没有端到端跑通）

`top_H = 0` ⇒ 零尺寸张量 ⇒ 附录 **ED.1 实测过**：`ChangeSize` 对零尺寸**返回成功**
并把 `firstPixelData` 置 **0** ⇒ 任何解引用它的层（如 `ZQ_CNN_Layer_UnaryOperation`
的 `GetFirstPixelPtr()[0]`）就是**空指针解引用**。

**诚实标注**：① 溢出**已实测**；② `top_H = 0` 是**算出来的**；
③ 从"零尺寸张量"到"空指针解引用"依赖 ED.1 的实测事实，
**我没有把这条链端到端跑通**（要构造完整 .zqparams + .nchwbin 且真的走到解引用那层）。

### 为什么这一轮没有修、也没有建门禁

1. **加值域上界**会改变 `ReadParam` 对现有模型的行为。按 ED.2 的规矩，
   动手前必须先数影响面 —— 这一轮没来得及数，**所以不擅自改**
   （ED.2 那次就是因为没先数，差点改坏 `det1.zqparams`）。
2. **建门禁**：`new ZQ_CNN_Layer_Convolution()` 会发射**整个虚表**，
   于是 `Forward` 引用的 `_convolution_nopadding` / `_addbias` / `_addbias_prelu` / `_prelu`
   全变成未定义符号；它们都住在 `ZQ_CNN_Forward_SSEUtils.cpp`（**单编 5 分钟以上**），
   拖进来会让 `zq_layerwire` 从快速门禁变成慢速门禁。
   试过在门禁里给这 4 个写空实现，签名对不上、反复两轮没成 ——
   **及时收手**，`git checkout` 复原到已验证的 132 例，**不留半成品**。

**记为"已定位、未修"，并写明为什么不在这一轮修。**
下一轮正确做法：① 先数现有模型里 kernel/dilate 的取值分布；
② 门禁单独开一个慢速通道（与 zq_nchw_conv 同批）建这个用例。

验证：`zq_layerwire` 132 例仍全绿，工作树干净（本轮最终只改了报告与 changelog）。

---

## 变更：附录 EM.3 —— 更正归属 + 修掉 16 处的整数溢出族（3 个 ReadParam 各加一个守卫）

### 更正 EM 的一处归属错误

EM 里我把 `ZQ_CNN_Layer.h:1287` 写成 `ZQ_CNN_Layer_Convolution::GetTopDim` —— **错的**。
类边界是 `Convolution` 262–870 / `DepthwiseConvolution` 871–1418 / `DeConvolution` 1419+，
**1287 属于 `DepthwiseConvolution`**。我刚读完 Convolution 就顺手归给了它 ——
**附录 DD.9.4「把一段代码归错符号」的又一次，而且是在我自己刚写的上一条附录里**。

### 按「全仓枚举同类站点」重扫：16 处、3 个类

| 类 | 行号 | 处数 |
|---|---|---|
| `ZQ_CNN_Layer_Convolution` | 682/683/695/696/713/714 | 6 |
| `ZQ_CNN_Layer_DepthwiseConvolution` | 1256/1257/1269/1270/1287/1288 | 6 |
| `ZQ_CNN_Layer_DeConvolution` | 1857/1858/1875/1876 | 4 |

### 修法：按"乘积必须放得进 int"来卡

    if ((__int64)dilate_H * (kernel_H - 1) + 1 > 0x7FFFFFFF
        || (__int64)dilate_W * (kernel_W - 1) + 1 > 0x7FFFFFFF)
    { ...; return false; }

**为什么不拍一个任意的系数上限**（比如"kernel 不许超过 100"）：
**乘积检查本身就是溢出条件**，不需要另设阈值 ——
`k=2e9, d=1` 的乘积是 2e9，放得进 int，就**不该拒**；
按系数卡会把一批合法（虽然罕见）配置误杀。

### 动手前的影响面统计（ED.2 的规矩）

    扫了 728 条卷积行
    kernel_size 分布 {1:67, 2:2, 3:32}，最大 3；dilate **从未出现**（全默认 1）
    会被新守卫拒的：**0 条**

现有模型离溢出差 **7.16 亿倍**，对 36 个随仓库模型**零影响**。

### 端到端验证（含一个我自己的期望错误）

| 输入 | 乘积 | ReadParam | 期望 | |
|---|---|---|---|---|
| `kernel_size=3` | — | 接受 | 接受 | OK |
| `k=2e9 d=1` | 2e9 < MAX | 接受 | **拒绝** | **不符 —— 是我期望错了** |
| `k=2e9 d=2` | 4e9 > MAX | 拒绝 | 拒绝 | OK |
| `k=1e9 d=3` | 3e9 > MAX | 拒绝 | 拒绝 | OK |

第二行值得单独记：我第一版把它期望成"拒绝"（"2e9 这么大的 kernel 肯定不对"），
**但它并不溢出**，守卫放行是**正确**的 —— 又一次"我把直觉当规格"。

**可复用的探针技巧**：`new ZQ_CNN_Layer_Convolution()` 会发射整个虚表，
把 4 个内核辅助函数变成未定义符号，而真实实现要拖进 **2215 个符号**的卷积内核图。
**从 `ZQ_CNN_Forward_SSEUtils.h` 的声明文本直接生成空实现**，签名必然一致：
正则抓 `static void NAME(...)` 整段 → 转成类外定义。
**我手写这 4 个签名对了三轮都没成，从声明生成一次就成了。**

### 门禁：这一条仍然没有

上面是一次性探针，不是门禁 —— 2215 个符号的卷积内核图拖不进快速门禁。
**记为"已修 + 已端到端验证 + 无门禁"**；要建就放进慢速通道（与 zq_nchw_conv 同批）。

验证：全量非慢门禁 **38/38 通过**；`zq_layerwire` 132 例全绿。

---

## 变更：附录 EN —— Concat 的 top/bottom **跨下标别名** = 堆越界写（修 + 门禁）

### 这个开放问题挂了好几轮，结论是"真缺陷"而不是"已知边界"

前面几轮把 `CopyData` 的自拷贝记成"语义上是抹掉数据、内存安全，所以不算缺陷"。
查完调用链之后发现**另一半**是真越界：

`ZQ_CNN_Forward_SSEUtils::_concat_NCHW`（`ZQCNN/ZQ_CNN_Forward_SSEUtils.cpp:4951`）
把 inputs 收成**指针**，随后才 `output.ChangeSize(...)`：

1. `output` 恰好就是某个 `valid_inputs[i]` 时，那个输入被**就地扩容**成 `C = out_C`，
   内容被 `Reset()` 清零；
2. 拷贝循环用 `in_C = valid_inputs[i]->GetC()` 取**扩容后**的 C；
3. 每个像素写 `out_C` 个 float，最后一个像素**越出整块分配**，越界量 =
   排在它前面那个输入的 C 个 float。

### 为什么没被挡住：守卫只比了**同一下标**

`ZQ_CNN_Net::_check_connect` 有一道就地守卫（附录 DY 那一轮加的），但它写的是

    for (int j = 0; j < top_names.size() && j < bottoms[i].size(); j++)
        if (tops[i][j] == bottoms[i][j]) ...

于是 `bottom=A bottom=B top=B` 被放行 —— j=0 比的是 B 与 A。
**别名的两个方向都会越界**，`top=A` 同样中招（只是越界量变成后面那个输入的 C）。

### ASan 实证（`tools/zq_concat_alias_probe.cpp`）

逐字照抄那个拷贝循环 + 真实 `ZQ_CNN_Tensor4D`，3 种对齐 x 8 组形状：

| 配置 | 崩溃数 |
|---|---|
| top 别名第 0 / 第 1 个输入 | **46 / 48** |
| top 是独立张量（对照） | **0 / 24** |

报告原文：`memcpy-param-overlap` + `0 bytes to the right of 32-byte region`。
没崩的那 2 例是溢出量正好落在对齐填充里 —— **不是"那两种配置是安全的"**。

### 修法（两处，缺一不可）

| 位置 | 作用 |
|---|---|
| `ZQCNN/ZQ_CNN_Forward_SSEUtils.cpp:4970` | `_concat_NCHW` 里加 `output` 与 `valid_inputs` 的逐个比对，撞上就 `return false`。这是**兜底**，护住直接调库 API 的人。 |
| `ZQCNN/ZQ_CNN_Net.h:1381` / `ZQCNN/ZQ_CNN_Net_NCHWC.h:806` | 守卫从"同一下标"改成"**比全部组合**"。这是**主修**，护住模型文件这条攻击面。 |

`ZQ_CNN_Net_NCHWC.h` 里那份同源守卫**必须同步改** —— 两份 Net 是各自独立的拷贝，
只改一份必然漂移（`tools/probe_inplace_topbottom.py` 的注释里已经写了这条"改一处必须改两处"）。

### 动手前的影响面统计（ED.2 的规矩）

    扫 28 个 .zqparams（build 目录里的副本已排除）、8880 个非 Input 层
    现有守卫（同一下标）命中：0
    加强后的守卫（任意 top 命中本层任意 bottom）命中：**0**

所以加强守卫**不拒任何真实模型**。据此才敢改。

### 门禁：新的 `zq_concat_alias`（6 例，含变异验证）

走**真**的 `ZQ_CNN_Net::LoadFrom`，不是照抄的循环 —— 照抄只能证明"那段代码会越界"，
证明不了"仓库里那段代码会越界"。

| 用例 | 内容 | 期望 |
|---|---|---|
| 0 | `top=C` 独立 | 放行 |
| 1 | `top=A`（= bottoms[0]） | 拒 |
| 2 | `top=B`（= **bottoms[1]，跨下标**） | **拒 —— 原来放行的那一种** |
| 3 | `top=C top=D` 都独立 | 放行 |
| 4 | `top=A top=C` | 拒 |
| 5 | `top=C top=B` | 拒 |

**对照用例 0/3 是必须的**：只测"别名被拒"的话，在 `LoadFrom` 开头无脑
`return false` 也能全绿。

**变异验证**（把内层循环改回只比 `k == j`，大括号平衡的改法）：

    变异体：harness RC=1，输出 "用例 2 ... 实际放行，**必须拒绝**"
    修复后：harness RC=0，6/6

### 附带产出：`tools/zq_net_fwd_tripwires.h`（44 个绊线桩，由脚本生成）

`ZQ_CNN_Net` 是**普通类**、`Forward()` 定义在头里，所以门禁 include 它就把
**45 个** `ZQ_CNN_Forward_SSEUtils` 辅助函数拖成未定义符号。不编那个
Forward_SSEUtils.cpp（会拖进单编 5 分钟以上的 conv GEMM，附录 EC.1）就得自己定义。

**做成绊线而不是空函数**：空桩是静默 no-op。哪天守卫被挪晚、某个 Forward 真被调到，
空桩会把数据丢掉然后**照样返回 true**，门禁报"全绿"而其实什么都没验。
绊线桩打名字 + `_exit(3)`，跑到就一定红。

`tools/gen_net_fwd_tripwires.py` 从**链接器的未定义符号表**生成（`--check` 可查过期），
踩了四个坑，每个都记在脚本注释里：

1. 符号集要用**独立探针 TU**（`zq_net_symprobe.cpp`）采集 —— 门禁自己已 include 生成物，
   链接是通的，采不到符号（鸡生蛋）；
2. 探针必须**引用 `LoadFrom`**：只引用 `Forward` 不够，层是虚函数、走虚表不需要定义，
   虚表也不发射，探针会**直接链接成功**、一个符号都没有。真正发射 36 个虚表的是
   `_load_param_file` 里那条 `if (名字=="X") new ZQ_CNN_Layer_X(); else if ...` 长链；
3. `c++filt` **不打印非模板函数的返回类型**，补 `void` 会把 `_concat_NCHW`（真返回 `bool`）
   写成另一个函数，链接报 "no declaration matches"。改成从
   `ZQ_CNN_Forward_SSEUtils.h` 的声明里解析真实返回类型；
4. 过滤排除项时不能用 `sig.split("::")[-1]` —— 参数里的 `std::vector<ZQ::ZQ_CNN_Tensor4D*>`
   自带 `::`，会把函数名切碎。

**排除项 1 个**：`_concat_NCHW_get_size` 不做绊线，因为
`ZQ_CNN_Layer_Concat::LayerSetup` 在**加载期**就要调它算输出形状 ——
做成绊线的话两个良性对照直接 rc=3。第一版就是这么写的，红了还一度以为是守卫没修好。
门禁里逐字照抄了真实现。**含越界写的 `_concat_NCHW` 本身仍然是绊线**：
守卫哪天被绕过去，别名模型会被当场炸掉，而不是"算出一堆垃圾还报全绿"。

### 一个负结果：MNN 转换器那份不用改

`ZQCNN_to_MNN/converter/source/ZQ_CNN_Net.h` 里的 `_check_connect` **连就地守卫都没有**
（比另外两份旧）。查了它的 `ZQ_CNN_Layer.h`：`Forward` 出现 0 次 —— 那是个
**只做图翻译、不跑前向**的快照，`_check_connect` 走不到那个拷贝循环。
所以这是**一致性漂移，不是安全缺陷**，不改（改了也只是让三份拷贝看起来一致，
而它们本来就不该一致 —— 那一份是另一个工具的私有快照）。

### 验证

- `zq_concat_alias` 6/6，harness RC=0；变异体 RC=1
- 全量非慢门禁 **39/39 通过**（新增 1 道）
- `check_text_encoding.py` / `check_line_endings.py` 均 OK

---

## 变更：附录 EO —— EM.3 自己写的注释里有 18 处**形近字乱码**，并把它变成门禁

### 发现

给 EN 做收尾时读 `ZQCNN/ZQ_CNN_Layer.h` 的卷积守卫，无意中看到：

    // kernel/dilate 部来自**模型文件**（不可信输入），两者都取 2e9 时这个 int 乘法
    // **溂出**（gcc 实测 -Woverflow，回绵成 643460097 这个**正数**，
    // 于是绕过下游的贤值检查，top_H 被算成 0 -> 零尺寸张量 ->
    // firstPixelData = 0 -> 空指针解参看，见附录 ED.1。）。
    // 这里**按"乘量必须放得进 int"来卡，而不拍一个任意的系数上限——
    // 乘积检查本身就是溂出条件，不需要另设闲值。
    // ... 离溂出差 7.16 亿倍

**这是我在附录 EM.3 里自己写的注释。** 六个词各被换成了一个形近的生僻字：

| 应为 | 写成 | 码位 |
|---|---|---|
| 溢出 | 溂出 | U+6E82 |
| 回绕 | 回绵 | U+7EF5 |
| 负值 | 贤值 | U+8D24 |
| 乘积 | 乘量 | U+4E58 U+91CF |
| 阈值 | 闲值 | U+95F2 U+503C |
| 解引用 | 解参看 | U+89E3 U+53C2 U+770B |

**三份拷贝（Convolution / DepthwiseConvolution / DeConvolution）、18 处**，
就在每天都在读的核心文件里。

### 为什么所有既有门禁都没抓到

`check_text_encoding.py` 的三类判定都**结构性地查不出**这一类：

- 不是非法 UTF-8 —— 字节完全合法；
- 不是 U+FFFD —— 那是"解码失败"留下的替换符，而这里是**解码成功但字选错了**；
- 罕见字清单里**确实有**它（全文只出现 3 次，`<= rarity` 阈值内），
  但那一栏的性质是"**人工核对用**"，不判失败。

而 `AGENTS.md` 里早就写着"一个常用字被换成形近的生僻字"是典型乱码特征 ——
**规则写了，工具没实现**。

### 修法：第 4 类 —— 已知形近字黑名单（自动判定，命中即失败）

    python tools/check_text_encoding.py --no-rare   # 修好后 RC=0
    # 注入 1 处坏字后                              # RC=1
    ZQCNN/ZQ_CNN_Layer.h: 形近字乱码: '溂' 应为 '溢出'（1 处）

黑名单只收**实际踩到过**的字，窄但零误报。写它的时候踩了四个坑，逐个记下来：

1. **黑名单会扫到自己**。文件里直接写字面量，第一版实测 **7 条报错全是自指**。
   改成 Python 的 `\uXXXX` 转义。
2. **不能收"会被文档引述"的字**。2026-10-02 那个被换的「不」不能进黑名单：
   `audit_k3_20261001.md` 与 `CHANGELOG_2026-10-02.md` 里**合法地引用了那次事件**
   （原文照抄），一进黑名单就是 **6 处误报**。
   教训：能被引述的说明已经有人盯着，不缺这一道自动检查。
3. **码位必须由 `ord()` 算，不能手打**。我把 U+6E82 手写成了 **U+6E42**，
   于是黑名单里那个键**根本不是那个字**，整条检查**静默失效**：
   探针把坏字注回去了，门禁还是绿的。
   **一个"检查别人有没有写错字"的检查，自己写错了一个码位而无人察觉** ——
   这是本会话最讽刺的一处，也是为什么必须做变异验证而不是"跑一遍绿了就信"。
4. **heredoc 和 Edit 都会把 `\uXXXX` 解释掉**，两次都改不动那一行。
   最后靠"写成脚本文件再执行"绕开（Write 写字面量，不吃转义）。

### 顺带修掉的 18 处

`ZQCNN/ZQ_CNN_Layer.h` 三份相同注释全部改正，另外把两个标点/措辞问题一并理顺
（`部来自` -> `同样来自`、句尾 `。）。` -> `）。`、
`绕过下游的贤值检查` -> `绕过了下游的 top_H <= 0 检查`）。

**注释只改文字，代码一行没动** —— 守卫逻辑与 EM.3 提交时完全一致。

### 验证

- `check_text_encoding.py --no-rare`：RC=0 / `OK: 697 text files`
- 变异（注入 1 处坏字）：**RC=1**，报出文件与期望字
- `check_line_endings.py`：`line endings OK`
- C6 UBSan 全量：**39/39 通过**（EM 生产改动 + EN 新门禁）

### EO.7 EM 的门禁建起来了 + **更正 EM.5 的一个错误结论**

EM.5 记的是"真实实现要拖 **2215 个符号**的卷积内核图，拖不进快速门禁"，
所以 EM 被记成"已修 + 已端到端验证 + **无门禁**"。

**那个测量是错的**：实测 `new ZQ_CNN_Layer_Convolution()`（连带
Depthwise / DeConvolution）的未定义符号只有 **6 个**，全是
`ZQ_CNN_Forward_SSEUtils` 的辅助函数 —— 正是 `tools/zq_net_fwd_tripwires.h`
（44 个绊线）覆盖的那一族。于是门禁可以直接进**快速通道**。

**2215 是怎么来的**：EM.3 的探针做法是对的（从声明文本生成空实现），
但我当时顺手 `nm` 了**整个静态库的符号表**，把"库里有多少符号"
当成了"链接器还差多少符号"。**一个差了两个数量级的测量错误，
就足以把"能做"记成"做不了"**，而且它进报告之后**后面几轮都直接引用，没人重测**。

教训与 DY.5 / DY.8 同族但更隐蔽：那几次是**消息为空**，
这次是**数字本身错得合理**，以至于没人怀疑。

**门禁 `zq_convparam`（15 例）**，两个方向的变异验证：

| 变异 | 结果 |
|---|---|
| 3 个守卫改成 `if (false && ...)` | RC=1，**恰好 4 个溢出用例红** |
| 守卫换成**拍脑袋的系数上限** `kernel>1000` | RC=1，**恰好 3 个 `kernel=2e9 dilate=1` 红** |

第二个变异才是价值所在：它拦住的是**"看起来更严格、实际会误杀合法配置"的错修法**。
EM.3 选"按乘积卡"而不选"按系数卡"的理由写在注释里，但**没有任何东西钉住这个选择**。

### 变更文件

- `tools/zq_convparam_check.cpp`（新，15 例）
- `tools/run_zqlib_checks.py`（4 张表各加一行）

### 验证

- `zq_convparam` 15/15；两个方向的变异体都是 RC=1
- 全量非慢门禁 **40/40 通过**（新增 1 道）

### EO.8 `_simplify_inplace` 那个"可疑点"是**承重设计**，差点被我加守卫打死

顺着 EN 的思路查：`_is_inplace_safe` 名单里的层**本来就被允许**拿自己的输入当输出。
那 `_simplify_inplace()` 里那句只改下标 0 的

    tops[i][0] = bottoms[i][0];

看起来像个该补守卫的洞 —— `ReLU bottom=A bottom=B top=C` 里声明的 top C 会被静默丢弃。

**动手前先数**（`tools/probe_inplace_simplify.py`）：

    扫 27 个不同的 .zqparams
    inplace-safe 层共 1229 个
      声明了多于一个 bottom 的：  0
      声明了 top != bottoms[0] 的： **638**

**638 个真实层**都写着 `bottom=X top=Y`（X≠Y）。
所以 `tops[i][0] = bottoms[i][0]` **不是 bug，是这套库的核心优化** ——
把 ReLU / BatchNormScale 变成真正的就地运算，省一次缓冲区分配；
`simplify_inplace_blob_map` 负责让 `GetBlobByName(top名字)` 还指得到那个 buffer。

**不但不该加守卫，加了会一次拒掉 638 个真实层。**

ED.2「动手前先数现有模型的取值分布」第三次直接救下一场事故
（前两次：EO.1 的 dilate 守卫、EN 的跨下标守卫）。

顺带一个探针自身的教训：这门探针第一版把 638 条全打出来，**82 KB**，
顶爆输出上限、结论被埋在中间。已改成只打前 10 条并报总数。

### 变更文件

- `tools/probe_inplace_simplify.py`（新，只读诊断，不进回归）

---

## 变更：附录 EP —— 层类的 `bottoms[k]` 越界读（一个**已修**的族 + 一个两次判错的探针）

### 假设

`ZQ_CNN_Layer.h` 到处是这个惯用法：

    if (... || bottoms->size() == 0 || ... || (*bottoms)[0] == 0) return false;

它只拒掉**空**的 bottoms，然后去读 `(*bottoms)[1]` ——
对任何需要两个 bottom 的层，**`(*bottoms)[1] == 0` 这个判断本身就是那次越界读**。

`ZQ_CNN_Layer_PriorBox` 里已经写着这条注释（上一轮的我加的，`git log -S` 查到是 3c6bb9c）：

    // 本层要用 (*bottoms)[1] 拿图像尺寸, 守卫只判 size()==0 的话, 直接
    // 构造层对象并只传 1 个 bottom 就是越界读。

**所以这族问题已经修完了。**

### 探针两次判错（比结论更值得记）

为了确认别的类没有同样的问题，写了 `tools/probe_bottom_count.py` 扫 36 个层类。
它**错了两次**，两次都是同一类错误：

| 版本 | 判据 | 结果 | 错在哪 |
|---|---|---|---|
| v1 | 只认 `size() >= N` 算守卫 | 4 个类全报 "**NO GUARD**" | 这个文件**实际用的写法是 `< N`**（`size() < 3` 才继续 = 保护 3 个）。4 个类**全都有**正确守卫。 |
| v2 | 认全 6 种形式，但把 `< N` 算成"只保护 N-1" | 3 个类报 "**NO GUARD**" | **语义写反**：守卫在条件成立时 return false，所以 `size() < 3` 意味着 size ≥ 3 才继续 —— 保护的正是 3 个。 |
| v3 | 修正语义 | **0 个缺守卫** | — |

**一个对正确代码一律报警的探针，比没有探针更糟** —— 会让人去"修"已经修好的地方。
v2 错得尤其彻底：形式认对了、语义反了，而"形式对、语义反"恰好是那种
**看起来已经 work** 的错。

配了变异验证：把 `PriorBox` 那道 `size() < 2` 退回成 `== 0`，

| | 退出码 | 输出 |
|---|---|---|
| 变异体 | **1** | 精确报出 `ZQ_CNN_Layer_PriorBox ... **NO GUARD**` |
| 还原 | **0** | `classes without a sufficient guard: 0` |

### 结论

- 需要 2 个以上 bottom 的层共 **4 个**：`DetectionOutput`(3)、
  `DetectionOutput_MXNET`(3)、`PriorBox`(2)、`UnaryOperation`(2)，
  **全部已有充分守卫**，无缺陷。
- 其余 32 个类只索引 `bottoms[0]`，`size()==0` 的守卫已经够。
- `PriorBoxText` 继承 `PriorBox`，共用那道守卫。
- `PriorBoxText::Forward` 里那个 `vector<T*>` → `vector<const T*>` 的强制转型
  **记为已知技术债**：标准上是 UB，但布局一致、10 个真实 PriorBox 层全走正常路径，
  不是可利用的缺陷，不动。

**这一节没修任何东西** —— 记下来是因为"查过、确认已修、并且探针的两次误报"
本身有价值：下一个人不必重查，也不必再写一个会误报的探针。

### 变更文件

- `tools/probe_bottom_count.py`（新，只读诊断，不进回归）
- `audit_k3_20261001.md`（附录 EP）

---

## 变更：附录 EQ —— 把 EM 那一族扫干净（探针上了两次变异才可信）

### 探针第一版报 0 命中，而那是**假的**

`tools/probe_param_products.py` 专门扫"`dilate * (kernel-1)` 这个模型值 int 乘法溢出"族。
第一次跑输出 0 命中，看起来是"这一族已清干净"。

**但今天已经写错过两个检测器（EP.2 的 bottoms 计数两次判错、EL 的文件计数改了六版），
所以这个 0 一文不值 —— 除非先证明它能抓到已知样本。**

拿 EM 修复**之前**的写法（去掉 3 处 `(__int64)`）当阳性对照：

| | 探针输出 |
|---|---|
| 修复前（阳性对照） | **0 命中** ← 探针坏了 |
| 修复后 | 0 命中 |

原因：正则写成 `VAR * VAR`，而 EM 的实际形状是 `dilate_H * (kernel_H - 1)` ——
右操作数是**带括号的表达式**，不是裸变量。

### 改完还是漏了一半

放宽右操作数后阳性对照能抓到了，但 `(kernel_W - 1)*dilate_W`
（模型变量在**右**边，`ZQ_CNN_Layer.h:1310/1311`）仍然漏。

**两次错都是同一类：正则的方向性。** v1 看不见右括号表达式，v2 看不见左括号表达式，
而且**两半看起来都很合理**。

最终改成**无方向的成对扫描**：取一行里所有模型变量的位置，
任意两个之间若只隔着 `*`（且无 `= ; , < > ! + /`）即命中。

### 两次误报也一并修掉

| 误报 | 原因 | 修法 |
|---|---|---|
| `if ((__int64)dilate_H * ...` 6 条 | 已经加宽了 | 加宽正则没算结尾的 `)` —— EM 的守卫恰好是 `(__int64)` |
| 3 条注释行 | 注释里提到，不是计算 | 跳过 `//` 开头的行 |

### 结果：16 条，与 EM 记的清单逐类数量完全吻合

| 类 | 本探针 | EM.3 |
|---|---|---|
| Convolution | 6 | 6 |
| DepthwiseConvolution | 6 | 6 |
| DeConvolution | 4 | 4 |

**没有第 17 处。**

### 这 16 处不需要各自再加守卫

`LoadFrom -> _load_param_file`（对**每一个**层调 `ReadParam`，EM 的 3 个守卫在这里）
`-> _check_connect -> _load_model_file -> SetUp`（逐层 `LayerSetup` ->
`SetBottomDim`/`GetTopDim`）。任何一层 `ReadParam` 返回 false 就 `_clear()` 并让
`LoadFrom` 失败，所以那 16 处**永远看不到会溢出的组合**。

EQ 只是把"下游确实只有这 16 处、且都在守卫之后"**从推断变成了计数**。

### 两条方法论

1. **"扫出来 0 命中"不是结论，是"我还不知道探针坏没坏"。**
   今天第三次为这件事交学费。**sweep 必须先在已知样本上证明自己能命中。**
2. **正则在扫代码时几乎总是有方向性**，两版各漏一半且都很合理。
   凡是扫语法模式的工具，成对扫描比正则更不容易漏。

### 变更文件

- `tools/probe_param_products.py`（新，只读诊断，不进回归；头注释里带阳性对照的记法）
- `audit_k3_20261001.md`（附录 EQ）

---

## 变更：附录 ER —— 模型值作除数 / 移位量（两族都干净，探针带两个阳性对照）

### 除数

`tools/probe_div_shift.py` 扫"模型值当除数"（stride==0 -> x86 idiv -> SIGFPE 进程死）：

    18 个除数站点，6 个 (变量, 类) 组合，**全部 zero-guarded**
    分布：Convolution / DepthwiseConvolution / Pooling，各 6（stride_H、stride_W 各 3）

**`Pooling` 是 EM 那次没提到的**，但它同样有 `stride == 0` 守卫、同样安全。
记下来是因为"EM 提过 7 个卷积 wrapper"很容易让人以为只有卷积有除法。

### 移位量

| | 探针输出 |
|---|---|
| 注入 `1 << kernel_H` | `shifts : 1`，精确指出 1329 DepthwiseConvolution::GetTopDim |
| 还原 | `shifts : 0` |

**0 处真命中** —— `ZQ_CNN_Layer.h` 里不存在以模型值为移位量的位运算。

### 三个误报族，每一个都让答案隐形

共同点：不是"报多了"，而是**信号被噪声埋掉，结论变成"没有"**。

| 误报 | 为什么分不清 | 修法 |
|---|---|---|
| `<< kernel_H` 当成移位 | `<<` 既是移位也是流插入 | 按**语句**（累积到 `;`）判断，不按行 |
| 逐 token 也不行 | 链式插入里 `kernel_H` 前的 `<<` 前面是字符串字面量 | 放宽成"这条语句里出现过 cout/cerr/clog" |
| 逐行也不行 | `std::cout` 在**上一行**，`<<` 在下一行 | 语句跨行累积 |
| 守卫"算存在" | 守卫可能在别的层里 | 查找范围限定在同一 class（弱于"同一函数"，已写进头注释） |

第 2、3 条尤其值得记：**两次修法各自都"看起来对"**，
一次只管同一行内、一次只管同一 token，而真实代码是**跨行 + 链式**的，
两个盲区正好各漏一半。

### 变更文件

- `tools/probe_div_shift.py`（新，只读诊断，不进回归；带 `--selftest` 阳性对照）
- `audit_k3_20261001.md`（附录 ER）

---

## 变更：附录 ES —— ZQlib 的 Linux 可编译性（修 3 类真缺陷 + 挖出 1 个 harness 缺陷）

基线：**121 OK / 13 BROKEN / 8 NEEDS_LIB / 1 MSVC_ONLY**（143 个头）。
用户目标是"windows 和 linux 都能完全跑通"，这 22 个非 OK 的头是硬指标。

### ES.1 `__min` / `__max`：56 个头、381 处，ZQlib 侧一个定义都没有

这两个是 **MSVC 内建**，gcc 没有。`ZQCNN/ZQ_CNN_CompileConfig.h:100-106`
**早就有**可移植定义 —— 同一作者、同一写法，**只是 ZQlib 侧从来没有**。

**为什么 49 个用了它的头还能编过**：

| 上下文 | gcc 的反应 |
|---|---|
| 非模板函数里用 | 只当"隐式函数声明"，**能编过** |
| **模板**函数里用 | 两阶段查找 → **硬错误** |

所以正确说法是"**在模板里用了才编不过**"。真正卡住 4 个头：
`ZQ_CameraCalibrationMulti` / `ZQ_MultiCamCalibration` / `ZQ_OpticalFlow` /
`ZQ_StereoRectify`。

新增 `3rdparty/include/ZQlib/ZQ_CompileConfig.h`（与 ZQCNN 那份逐字一致），
挂在 `ZQ_MathBase.h` —— 它是这 4 个头**传递包含闭包的公共祖先**里最底层的一个。

### ES.2 `ZQ_OpticalFlow.h` 同一变量声明两次 —— 哪个编译器都编不过

    3017:  int nPixels = width*height;
    3024:  int nPixels = occ.npixels();      <-- 同作用域重复声明
    3032:  for (int i = 0; i < nPixels; i++)

同一作用域重声明在任何标准 C++ 下都是硬错误，**MSVC 也编不过**。
没被发现是因为全仓只有 `ZQ_StereoRectify.h` include 它，而它又不在任何构建里 ——
**一个孤立头从来没有被任何编译器看过**。删掉第二句（两值本来就相等）。

### ES.3 模板成员函数的**类外定义**里写了 `static`（`[class.mfct]` 违反）

`ZQ_CameraCalibrationMulti.h:428` 与 `ZQ_MultiCamCalibration.h:432`。
**注意别改错地方**：类**内**的 `template<class T> static bool f();` 是**合法**的，
要删的只有类外定义那 2 处（全仓符合此形状的 8 处里，6 处合法）。

### ES.4 harness 缺陷：探针用**固定**工作目录 `/tmp/zqprobe` 且开头 `rm -rf`

修完上面三处重跑，**BROKEN 从 12 变成 103**。第一反应是"我改坏了" ——
单独重编"新坏"的头，**通过**。91 个全是假的，且**错误信息是空的**
（`grep error:` 在已被删掉的 `.err` 上找不到东西）。

原因：完整审计回归**内部会调这个脚本**，我同时手工跑了一次，两边互删文件。
与附录 EB.1 同一个毛病，**当时只修了 `run_zqlib_checks.py`，漏了这个探针**。
已改成 `pid + 时间戳` 的唯一目录，且不删别人的目录。

> 空错误消息本身就是最强的信号：真的编译错误不会没有消息。

### ES.5 `ZQ_ObjLoader.h`：格式串丢失 + `strncpy_s` 参数顺序错

    printf(buf, "something wrong:%s:%d\n", __FILE__, __LINE__);   // buf 成了格式串
    assert(buf);                                                    // 断言恒真的东西
    strncpy_s(buf, argv + i, len);                                  // 第 2 个参数应是缓冲区大小

`buf` 是全零数组 ⇒ 实际什么也不打印；`strncpy_s` 传错位置 ⇒ **MSVC 上同样编不过**。
两处（740 / 791）都改了。

### ES.6 一个**不修**的：`ZQ_Calibration.h` 引用了已被删除的 API

24 处调用 `ZQ_Rodrigues_r2R_fun` / `_jac`，而现存的只有 `ZQ_Rodrigues_r2R`，
且**返回 `void`**（调用方按 `bool` 用）。不是改名能了事，语义也变了。
而这个头**全仓没有任何人 include**。判定为陈旧死代码，不修。

### ES.7 负面结论：不需要给 `*_s` 做 shim

28 处 `*_s` 里 **21 处已在 `#if defined(_WIN32)` 里**（全是 `fopen_s`），
真正没守卫的只有 7 处，集中在 3 个**本来就因别的原因**非 OK 的头
（log4cplus / MSVC-only / GBK）。**写 shim 不会让任何一个头变成 OK** —— 不做。

### 变更文件

- `3rdparty/include/ZQlib/ZQ_CompileConfig.h`（新）
- `3rdparty/include/ZQlib/ZQ_MathBase.h`（挂上配置头）
- `3rdparty/include/ZQlib/ZQ_OpticalFlow.h`（删重复声明）
- `3rdparty/include/ZQlib/ZQ_CameraCalibrationMulti.h`（删类外 static）
- `3rdparty/include/ZQlib/ZQ_MultiCamCalibration.h`（同上）
- `3rdparty/include/ZQlib/ZQ_ObjLoader.h`（格式串 + strncpy_s 两处 × 2）
- `tools/probe_zqlib_headers.py`（工作目录唯一化）

### 实测结果

    探针（tools/probe_zqlib_headers.py，143 个头）
                    修前      修后
      OK             121   ->  126   (+5)
      BROKEN          13   ->    9   (-4)
      NEEDS_LIB        8   ->    8
      MSVC_ONLY        1   ->    1

修好的 4 个：`ZQ_CameraCalibrationMulti` / `ZQ_MultiCamCalibration` /
`ZQ_OpticalFlow` / `ZQ_StereoRectify`（第 5 个 OK 增量是 `ZQ_ImageProcessing.h`，
它经由 `__min` 修复被动受益）。

**剩下 9 个 BROKEN 全部有交代**：

| 头 | 原因 | 处置 |
|---|---|---|
| `ZQ_WinSock*`（7 个） | 依赖 `winsock2.h` —— **设计上就是 Windows-only** | 不修 |
| `ZQ_GLSLShader.h` | 缺 `<GL/glew.h>`（环境依赖未装） | 不修 |
| `ZQ_Calibration.h` | 引用已被删除的 `ZQ_Rodrigues_r2R_fun`（死代码，见 ES.6） | 不修 |

所以 ZQlib 侧**真正需要在 Linux 上修的编译问题已经清零**。

### 注意事项

- 新增的 `ZQ_CompileConfig.h` 必须先于任何用到 `__min`/`__max` 的头被包含；
  现在挂在 `ZQ_MathBase.h` 上，它是那 4 个头传递闭包里最底层的一个。
  将来若新增 ZQlib 头并绕过 `ZQ_MathBase.h`，**需要再挂一次**。
- `#ifndef` 保护是必需的：MSVC 下 `__min`/`__max` 是编译器内建，重复定义会报错。

---

## 变更：附录 ET —— 文件级可达性门禁（"没有任何编译器看过这个文件"变成一条 diff）

### 动机

ES.2 那个 bug（`ZQ_OpticalFlow.h` 同作用域重复声明，**任何编译器都编不过**）
之所以活下来：全仓只有 `ZQ_StereoRectify.h` include 它，而后者不在任何构建里 ——
**离任何构建两跳**。`reachability_probe.py` 已有，但它答的是"层类型有没有被模型跑到"，
不是"文件有没有被任何编译器编过"。

### 探针三次判错

| 版本 | 报出"从未编译" | 错在哪 |
|---|---|---|
| v1 | **410**（全部） | 源文件绝对路径 vs 入口点相对路径，两个集合**永不相交** |
| v2 | 338 | `file(GLOB .../*.c)` 没展开 —— ZQCNN/CMakeLists.txt:5-7 **正是用 GLOB 列内核 TU** |
| v3 | 338 | GLOB 模式**没加引号**（`${CMAKE_CURRENT_LIST_DIR}/*.cpp`），只认引号内的 |
| v4 | 140 | samples 用自定义宏 `SUBDIRLIST` + foreach 嵌套 GLOB ⇒ 改用**标注过的近似** |
| v5 | **7** | — |

第一次那个错最值得记：**它报"410 个文件全部从未被编译"**，一个荒谬的数字，
我却又往下走了两版才反应过来。**荒谬的结果本身就是信号** ——
与 ES.4 里"空错误消息"同类。

### 结构性假阳性单列

`3rdparty/include/ZQlib/` 下 **120 个头**永远不在闭包里（头文件库，没有自己的 TU），
但 `probe_zqlib_headers.py` **逐个给它们生成最小 TU 编过**。
所以探针把它们单列成 `[zqlib]`，不计入基线 —— 混进去就是 120 条噪声。

### 剩下的 7 个

| 文件 | 判定 |
|---|---|
| `ZQCNN/ZQ_CNN_MTCNN_ncnn.h` | 需要 ncnn |
| `.../zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_8x4.h`（81 行） | **死代码片段**：BOM 开头，第一行 `#if WITH_BIAS`，没有函数签名 |
| `.../zq_cnn_convolution_gemm_nchwc_kernel1x1_neon_raw.h`（1711 行） | **死代码片段**：完整 static 函数但引用外部 `Mat`；ARM NEON 路径 |
| `ZQlibFaceID/ZQ_Face{ClusterImagesForVideo,ClustersForVideo,ContainerForVideo,Extractor}.h` | **主库**的 4 个头，从未被任何编译器类型检查过 |

最后 4 个最值得记：它们是**主库**的头。
`ZQ_FaceExtractor.h` 正是那 4 个 UNUSED 层的调用方 ——
**它们能接线的前提，恰好是一个从未被编译过的头**。

两个 NCHWC 内核头合计 **1792 行**从未被编过；内部大概率也有问题（ES.2 先例），
但**不修** —— 复活死代码等于新增功能，不在审计范围内。

### 变异验证

| | 结果 |
|---|---|
| 造一个没人 include 的文件 | `+ ZQCNN/zq_orphan_control.h`，`1 new`，**RC=1** |
| 删掉 | `baseline OK: 7 entries unchanged`，**RC=0** |

（第一次变异选错（注释掉 `add_subdirectory(ZQCNN)`）—— 探针**正确地**报无变化，
因为 samples 仍 include ZQCNN 头。**选错变异目标不是探针的错**。）

内置阳性对照 `--selftest`：`ZQ_OpticalFlow.h` / `ZQ_StereoRectify.h` 必须判**不可达**，
`ZQ_CNN_Tensor4D.h` 必须判**可达**。

### 变更文件

- `tools/probe_file_reachability.py`（新，带 `--selftest` / `--save-baseline` / `--check-baseline`）
- `tools/file_reach_baseline.txt`（新，7 条）
- `tools/run_audit_checks.py`（挂进回归，组名 **C5**；`C4` 已被主工程 HIGH 桶门禁占用）

---

## 变更：附录 EU —— C1 门禁把 3 个头**分错类**了（它们不缺库，缺的是 MSVC 内建）

### 起点

ET 的文件级可达性门禁报出 `ZQlibFaceID` 的 4 个头不在任何构建里，
其中 `ZQ_FaceExtractor.h` 正是那 4 个 UNUSED 层的调用方。
去 C1 基线里查，3 个被归成 `NEEDS_LIB`（"缺外部库"）—— **归类不对**。

### 三个真实问题

| 头 | 真实原因 | 症状 |
|---|---|---|
| `ZQ_FaceContainerForVideo.h:87` | 用了 `__int64`，include 链上没有 `ZQ_CNN_CompileConfig.h` | `'__int64' was not declared` |
| `ZQ_FaceExtractor.h:55` | 用了 `__min` | `'__min' was not declared` |
| `ZQ_FaceClustersForVideo.h:185` | 形参是 `ZQ_FaceRecognizer&`，**但没 include 它** | `'ZQ_FaceRecognizer' has not been declared` |
| `ZQ_FaceClustersForVideo.h:315` | 用了 `FLT_MAX`，**没 include `<cfloat>`** | `'FLT_MAX' was not declared` |

前两个与附录 ES.1 同一族；后两个是"靠调用方碰巧先 include 过才编得过"。

### 修法

- `ZQCNN/ZQ_CNN_BBox.h` 加 `#include "ZQ_CNN_CompileConfig.h"` ——
  它是 ZQlibFaceID 那一侧的**公共祖先**（`ZQ_FaceGroup.h` / `ZQ_FaceDetector.h` 都 include 它），
  挂一次覆盖整条链。配置头自带 guard、三个宏都是 `#ifndef` 保护，
  MSVC 下本就是内建 ⇒ **对 Windows 侧零影响**。
- `ZQ_FaceClustersForVideo.h` 补 `#include "ZQ_FaceRecognizer.h"` 与 `#include <cfloat>`。

### 实测结果

    ZQ_FaceClusterImagesForVideo : 0 error
    ZQ_FaceClustersForVideo      : 0 error   (修前 201 -> 4 -> 1 -> 0)
    ZQ_FaceContainerForVideo     : 0 error   (修前 4)
    ZQ_FaceExtractor             : 0 error   (修前 1)

### 这一节真正学到的

**"NEEDS_LIB" 是个会骗人的桶。** 它的本意是"缺 jpeglib/OpenCV，装上就能编"，
但它也**接住了**"缺 `__int64` / `__min` / `FLT_MAX` / 漏 include"这几类
**根本不需要装任何东西**的问题。后果不是多报几条，而是**"装个库就好了"这个
判断被顺手接受了**。

便宜的强化判据：**一个头如果只 include 了本仓的头和标准头，却被归成 NEEDS_LIB，
那它一定不是真的缺库。** 本节 3 个头全部符合（依赖的 ZQ_FaceFeature.h /
ZQ_Kmeans.h / ZQ_MathBase.h 都是仓内 ZQlib 头）。

### 变更文件

- `ZQCNN/ZQ_CNN_BBox.h`（引入编译配置）
- `ZQlibFaceID/ZQ_FaceClustersForVideo.h`（补两个 include）

### EU.6 更严重的：C1 分类器让**两个桶从来没被填过**

查"为什么分错类"时发现了更要紧的东西。`classify()` 是**拿错误消息整行做子串匹配**，
而 `NEEDS_LIB` 列表里有一项 `'nn'`；错误消息开头就是文件路径：

    /mnt/d/ZQCNN/ZQlibFaceID/ZQ_FaceExtractor.h:55: error: '__min' was ...
                                       ^^^^^^ 小写后含 "cnn"，含 "nn"

**每一条错误消息都命中 `'nn'`。** 直调分类器实测：

| 输入 | 旧结果 | 应该是 |
|---|---|---|
| `/mnt/d/ZQCNN/.../Z.h:55: error: '__min' ...` | `NEEDS_LIB nn` | `MSVC_ONLY` |
| `/home/u/Z.h:55: error: '__min' ...` | `MSVC_ONLY __min` | `MSVC_ONLY` |

同一个错误，只因路径里有没有 `ZQCNN` 就分成两类。

**后果**：四个分类桶里 `MSVC_ONLY` 与 `BROKEN` 自门禁建立（6f91624，附录 EH）
以来**从来没被填过** —— 基线里 7 个非 OK **全部**是 `NEEDS_LIB`、零 MSVC_ONLY、
零 BROKEN，就是它失效的证据。
`ZQ_ObjLoader.h` 用 `strncpy_s`（在 MSVC_ONLY 名单里）却被报成 NEEDS_LIB，
我一开始就觉得奇怪 —— 原来只是被 `'nn'` 先截胡了。

> **一个从不变化的分类结果本身就该被怀疑。**

#### 修法（三步，缺一不可）

1. `NEEDS_LIB` 里 `'nn'` 换成 `'nn/'`、`'nnapi'`；
2. **不再拿整行匹配**：只对 `fatal error: <头名>: No such file` 里的那个头名匹配
   —— 它才是编译器真正找不到的东西；
3. **没有这个形状就完全不进 NEEDS_LIB 分支**。编译能走到"用了 `__int64`"
   这种错误，说明所有头都找到了，再谈"缺库"没有意义。

> 第 3 条是必要的：只做第 2 条并保留"没匹配到就用整行"的回退时，
> 六个测试错了三个 —— 路径里的 `cnn/` 仍然含 `nn/`。

#### 分类器现在有自己的单元测试

`--selftest`，9 个用例，固化每一类（MSVC_ONLY×2 / BROKEN×2 / NEEDS_LIB×5）。
**分类器自己不会失败，只会安静地把所有东西归进同一个桶**，
所以它和被它分类的对象一样需要门禁。已挂进回归（组名 C1b）。

### 变更文件（EU.6）

- `tools/probe_faceid_headers.py`（分类器三步修 + `--selftest` 9 例）
- `tools/run_audit_checks.py`（新增 C1b 组）

### EU.7 修好分类器后**浮出 4 个此前不可见的头**

| | 修前 | 修后 |
|---|---|---|
| OK | 22 | **25** |
| NEEDS_LIB | 7 | 0 |
| MSVC_ONLY | 0 | 0 |
| BROKEN | 0 | **4** |

3 个是 EU.1~EU.4 修好的（升到 OK）。但有 4 个从 NEEDS_LIB **掉到 BROKEN** ——
逐个看第一条错误后发现它们同样是缺外部库，只是名单没收录：

| 头 | 第一条错误 |
|---|---|
| `ZQ_FaceDetectorLibFaceDetect.h` | `facedetect-dll.h` |
| `ZQ_FaceRecognizerArcFaceMiniCaffe.h` | `caffe/caffe.hpp` |
| `ZQ_FaceRecognizerSphereFaceMiniCaffe.h` | `caffe/caffe.hpp` |
| `ZQ_FaceRecognizerSeetaFace.h` | `face_identification.h` |

已补进 `NEEDS_LIB`：**修好一个检测器，才看得见另一个检测器的缺口** ——
名单不全本身是 bug，但在 `NEEDS_LIB` 永远命中的前提下它毫无表现。

### EU.8 自测数据里的隐藏字符

自测那条 seata 用例一直报 BROKEN，而把**同样的文本**手工构造传给 `classify()`
却得到 NEEDS_LIB，逐项对比到 `msg_a == msg_b -> False`
（两者 print 完全一样、needle 一样、命中列表一样）。

分类器两边都对，**是测试数据里有一个隐藏字符**。换成更短的 `seata/face.h`，
自测现在 **9/9 全过**。

> 今天"形近字替换"（EO）、"heredoc 吃转义"、"Edit 剥掉前导空白"已经是第三类
> **文本在传输途中被悄悄改掉**的问题。共同特征：**输出看着对，
> 只有做等值比较才暴露**。

### 完整回归

`python tools/run_audit_checks.py --with-build --warn-sweep --src-sweep
--bounds-sweep --ubsan-sweep --reachability --msvc-asan`
→ **ALL CHECKS PASSED（AUDIT_EXIT=0，34 段全过）**。

---

## 变更：附录 EV —— 两份 `ZQ_CNN_CompileConfig.h` 共用同一个 include guard，而取值不同

### 怎么发现的

给 EU 的 `ZQ_CNN_BBox.h` 加 `#include "ZQ_CNN_CompileConfig.h"` 之后，
顺手查这个核心头会不会和别处冲突 —— 结果有 5 个宏在**两份同名头**里都被定义：

| 文件 | `ZQ_CNN_USE_MKL_GEMM` |
|---|---|
| `ZQCNN/ZQ_CNN_CompileConfig.h:27` | **0** |
| `ZQCNN_to_MNN/converter/source/ZQ_CNN_CompileConfig.h:38` | **1** |

主仓那份的注释写明 0 是必须的：

> 默认 0：ZQCNN 内部并不调用 cblas_*，开着它只会让示例程序链上 mklml.lib，
> 于是没装 MKL 运行库的机器上所有 exe 都起不来。

### 为什么是隐患

include guard **按 TU 生效**，两份用同一个 guard 名，于是某个 TU 若同时
include 两份：**先到的赢，后到的被整个静默跳过**，
`ZQ_CNN_USE_MKL_GEMM` 取 0 还是 1 **取决于 include 顺序**，且没有任何提示。

**实测全仓没有任何 TU 同时 include 两份**（逐文件收集出现过的配置头路径，
跨路径交集为空）—— 所以是**潜在隐患**而非现网缺陷。

### 修法

只把 MNN 那份的 guard 改名为 `_ZQ_CNN_MNN_CONVERTER_COMPILE_CONFIG_H_`。
改 guard 名是**行为等价**的，只是让两份不再互相遮蔽。

**两份的 MKL 取值都保持原样** —— 转换器是另一个工具、另一套依赖，
它要 1 可能是对的；把它的值改成 0 去"和主仓一致"是**用一致性换正确性**。
要消除的是"静默互相遮蔽"，不是"两份内容不同"。

> 纪律：**"两份同名头"本身就该是个问题**，哪怕当前内容一致 ——
> 一致只是巧合，不是约束。

### 变更文件

- `ZQCNN_to_MNN/converter/source/ZQ_CNN_CompileConfig.h`（guard 改名 + 说明）

---

## 变更：附录 EW —— ET 报出的"从未被编译"里藏着一个**真的编不过**的头

### 不可达 ≠ 编不过

ET 的基线里 7 个文件"任何构建都不编、也没有别的探针单独编过"。
ET 只回答"**有没有构建会编它**"，所以对"依赖其实在仓库里、只是没人 include"的那些无能为力。

手工核依赖：`ZQCNN/ZQ_CNN_MTCNN_ncnn.h` 要 `net.h`，
而 `3rdparty/include/ncnn/net.h` **就在仓库里**（`libncnn.a` 也在）——"编不过"不成立。

### 编出来的结果：真缺陷

    stl_uninitialized.h:127:72: error: static assertion failed:
        result type must be constructible from value type of input range
      required from 'std::vector<ncnn::Net>::resize(...)'
      ZQCNN/ZQ_CNN_MTCNN_ncnn.h:371:26:   required from here

根因：`ncnn::Net` 的**拷贝构造是 private**（`ncnn/net.h:156`）且无移动构造，
而 `vector::resize(n)` 扩容要移动/拷贝已有元素。单独验证过这个区分：

    std::vector<ncnn::Net> a(n);          // 编得过：只要可默认构造
    std::vector<ncnn::Net> b; b.resize(n); // 编不过：扩容要可移动/可拷贝

**这个头在 gcc 上编不过，而全仓没有任何 TU include 它** —— 两个事实叠加，
所以它从建库至今没被任何编译器看过。

### 修法

6 处 `resize(thread_num)` 换成"用带尺寸的临时对象赋值"：
vector 的**移动赋值在分配器相同时只交换内部指针**，不需要元素可移动；
带尺寸的构造只需要可默认构造（`ncnn::Net` 有 public 的 `Net();`）。语义与 resize 相同。

变异验证：退回 `resize` → error 1；改回 → 0。修后该头 `-fsyntax-only` **0 error**。

### 给 ET 加 `--syntax-only`：把"不可达"再拆一层

不可达的文件逐个单独编一遍，结果：

    OK    ZQCNN/ZQ_CNN_MTCNN_ncnn.h
    OK    ZQlibFaceID/ZQ_Face{ClusterImagesForVideo,ClustersForVideo,ContainerForVideo,Extractor}.h
    FRAG  .../zq_cnn_convolution_gemm_nchwc_kernel1x1_neon_raw.h
          ^ ARM NEON 片段：完整 static 函数但引用外部 Mat 类
    FRAG  .../zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_8x4.h
          ^ 81 行片段：BOM 开头、第一行 #if WITH_BIAS，没有函数签名
    -> 5/5 编得过；2 个是已知片段（不是独立头）

两个片段单列 `FRAG` 并各带理由，不报成 FAIL ——
否则门禁天天红在那两个**已知**项上，真正的回归就被埋掉。

顺带修掉实现里的一个坑：翻译单元走 **`g++ -x c++ -` 从 stdin 喂**。
第一版在 Windows 侧写 `/tmp/x.cpp` 再传路径 —— Git Bash 的 `/tmp` 是 `D:\tmp`，
WSL 里没有这个文件，于是每个文件都返回 "No such file or directory"、整列 FAIL。

### ET 的 7 条现在全部有交代

| 文件 | 交代 |
|---|---|
| `ZQCNN/ZQ_CNN_MTCNN_ncnn.h` | **真缺陷，已修**，现 0 error |
| `ZQlibFaceID/ZQ_Face{...}4 个` | 附录 EU 已修，现在都编得过 |
| 两个 NCHWC 内核片段（1792 行） | 死代码，标 FRAG，不复活 |

### 变更文件

- `ZQCNN/ZQ_CNN_MTCNN_ncnn.h`（6 处 resize -> 带尺寸赋值）
- `tools/probe_file_reachability.py`（`--syntax-only` + `NOT_A_HEADER` 片段白名单 + ncnn 等 include 根）

---

## 变更：附录 EX —— MNN 转换器**分叉**出去的那份 ZQCNN 头，落后三处已修的守卫

### 怎么找到的

给 EV 修完 `ZQCNN_to_MNN` 那份配置头的 include guard 后，顺手查这个目录在不在构建里：

    顶层 CMakeLists.txt 的 add_subdirectory 没有 ZQCNN_to_MNN
    ZQCNN_to_MNN/converter/CMakelists.txt 从来没被 configure 过
    原因：转换器本体第一行 #include "MNN_generated.h"，而 MNN 框架不在仓库里

于是 `converter/source/` 的**七个头、10304 行从来没有被任何编译器看过** ——
与 ES.2、EW 同源，规模更大。

### 漂移量化（tools/probe_mnn_fork_drift.py）

| 头 | 主树 | 分叉 |
|---|---|---|
| `ZQ_CNN_BBox.h` | 178 | 132 |
| `ZQ_CNN_BBoxUtils.h` | 760 | 728 |
| `ZQ_CNN_Forward_SSEUtils.h` | 3386 | **172**（纯声明的桩） |
| `ZQ_CNN_Layer.h` | 10542 | 6346 |
| `ZQ_CNN_Net.h` | 1934 | 1692 |
| `ZQ_CNN_Tensor4D.h` | 1044 | 839 |

分叉**落后 4 处**已诊断并修好的改动：

| # | 缺什么 | 后果 |
|---|---|---|
| 1 | `ZQ_CNN_BBox.h` 没 include 编译配置 | `__min`/`__max`/`__int64` 是 MSVC 内建，gcc 下无定义 → `ZQ_CNN_BBoxUtils.h` **10 个编译错误** |
| 2 | `ZQ_CNN_Net.h` **完全没有就地守卫** | `Concat bottom=A bottom=B top=B` 一路走到底 → **堆越界写**（附录 EN 那一族） |
| 3 | 卷积 `ReadParam` 没有 `stride==0` 守卫 | `GetTopDim:492` 整数除零 → **SIGFPE** |
| 4 | 同一处没有 kernel/dilate 溢出守卫 | `(kernel_H-1)*dilate_H` 溢出 → top_H=0 → 空指针解引用（附录 EM 那一族） |

> 第 3、4 条是把分叉的 `GetTopDim` 逐行看过才发现的 ——
> 探针只能告诉你"少了某个修复的文本"，**"分叉自己有哪些缺陷"要另外查**。
> 分叉的 `ReadParam` 结尾只有 `return has_num_output && has_kernelH && …`，
> **连 `kernel_H <= 0` 都没有**，比主树 EM 修复前还老一代。

### 全部镜像过去

- `ZQ_CNN_BBox.h` 加 `#include "ZQ_CNN_CompileConfig.h"`（与主树 EU.2 同一处）
- `ZQ_CNN_Net.h` 的 `_check_connect` 补就地守卫，**含"比全部组合"**，
  名单与该文件 `_simplify_inplace` 那段一致
- `ZQ_CNN_Layer.h` 的**两个**卷积类（`Convolution` / `DepthwiseConvolution`，
  它们的 `ReadParam` 收尾**逐字节相同**）各补 `stride==0` 与 kernel/dilate 溢出守卫

修后：分叉 7 个头 `gcc -fsyntax-only` **全部 0 error**，漂移项 **3 → 0**。

### 门禁 probe_mnn_fork.py（挂进回归，组名 C5b）

两件事：① 逐头 `-fsyntax-only` 编一遍；② 断言那几处守卫**还在** ——
后者防"将来从主树同步时又悄悄丢掉"。

阳性对照每次都跑，**三条都验**：主树 BBox 有配置 include / 好的头编得过 /
不存在的头编不过（第三条验"探针能不能看出失败"，只有全过的对照不够）。

| 变异 | 结果 |
|---|---|
| 就地守卫退回"只比同一下标" | **RC=1**，精确报 GUARD MISSING |
| 两处溢出守卫改成 `if (false && …)` | **RC=1** |
| 去掉 `ZQ_CNN_BBox.h` 的配置头 include | **RC=1**，报 COMPILE FAIL |
| 全部还原 | **RC=0** |

### 为什么不顺手做版本同步

分叉与主树还有大量**正常的**版本差（`Forward_SSEUtils.h` 3386 vs 172 行的
纯声明桩、`Layer.h` 差 4000 行）。那是分叉该有的样子 ——
它服务于一个只需要图结构、不需要内核实现的转换器。
**只镜像"已诊断为缺陷、且本仓库另一处已经修好"的那几条** ——
和 AGENTS.md 里「两份同名头内容不同是正常的，但互相静默遮蔽不是」同一条原则。

### 变更文件

- `ZQCNN_to_MNN/converter/source/ZQ_CNN_BBox.h`（引入编译配置）
- `ZQCNN_to_MNN/converter/source/ZQ_CNN_Net.h`（就地守卫，比全部组合）
- `ZQCNN_to_MNN/converter/source/ZQ_CNN_Layer.h`（两处卷积守卫）
- `tools/probe_mnn_fork.py`（新门禁，含阳性对照）
- `tools/probe_mnn_fork_drift.py`（新，漂移量化）
- `tools/run_audit_checks.py`（新增 C5b 组）

---

## 变更：附录 EY —— 把 `stride == 0` 那一族钉进 `zq_convparam`（SIGFPE 比溢出更狠）

### 为什么单独拎出来

EM 修的是 `dilate * (kernel - 1)` 溢出，后果是**数据错**。
同一道守卫的另一半 —— `stride == 0` —— 后果是 `x / stride` 的**整数除零**：
x86 `idiv` 直接陷阱，**进程 SIGFPE 当场死掉**。

`stride <= 0` 就在**同一行 if 里**：

    if (kernel_H <= 0 || kernel_W <= 0 || stride_H <= 0 || stride_W <= 0
        || dilate_H <= 0 || dilate_W <= 0)

"在同一个 if 里"说明它们是同一轮加的，只是我给其中一半写了用例。

### 门禁扩到 31 例（原来 15）

新增 16 例：三个卷积类 × {`stride=0`, `stride=-1`, `kernel=0`, `kernel=-1`, `dilate=0`}，
外加"正常 `stride=2` 不能被误杀"的正例。

> 顺带修掉一个**自造的无效用例**：`param_line()` 原来写 `if (c.dilate) { 写 dilate }` ——
> 于是 `dilate=0` **根本表达不出来**，而它恰好是最该测的值之一。
> 改成"不等于默认值 1 时才写"（stride 同理），0 与负数才都进得来。
> 与 AGENTS.md「无效的测试用例和无效的变异一样有害」同一件事：
> **被验证的那一处在当前输入下够不着，用例就是废的。**

### 两个方向的变异验证

| 变异 | 结果 |
|---|---|
| 3 处 `stride_H <= 0 \|\| stride_W <= 0` 改成 `<= -1000000` | **RC=1**，**恰好 6 条** stride 用例红（3 类 × {0,-1}），其余 25 条仍绿 |
| 3 处 `kernel_H <= 0 \|\| kernel_W <= 0` 同样改 | **RC=1**，**恰好 5 条** kernel 用例红 |
| 全部还原 | **RC=0**，31/31 |

变异写法选 `<= -1000000` 而不是删掉那一项 —— 删掉会让整行少一个操作数、
**根本编不过**，"变异体没红"又变成"变异体没编出来"。

### 剩下 3 个 UNUSED 层记为未做

`LSTM_TF` / `PriorBoxText` / `DetectionOutput_MXNET` 的 `Forward` 调的是
`ZQ_CNN_Forward_SSEUtils` 的**私有**辅助函数，而那 44 个正是给它们准备的**绊线** ——
要测"层有没有把参数接对"就得换成**记录桩**，即再开一个 `zq_layerwire` 那样的例外
（第二份照抄实现 + 一份需要同步的例外清单）。

按 AGENTS.md「推不动就把排除了什么记下来」：**记为未做**。
理由是它需要双模式桩，而收益是覆盖三个**没有任何随仓库模型会用**的层；
相比之下 EY.1 那一行就在三个**在用**的类里、后果是**进程直接死**，
先做它是正确的排序。

### 变更文件

- `tools/zq_convparam_check.cpp`（15 -> 31 例；`param_line` 改为"非默认值才写"）

---

## 变更：附录 EZ —— 把 EY 那个模式系统化：三个 `ReadParam` 守卫正确但零用例

### 把"同一道守卫只测了一半"变成一次扫描

`tools/probe_readparam_coverage.py` 对每个层类列出 `ReadParam` 里的数值守卫，
报出哪些分支在任何门禁源码里都没出现过。

> **探针自己错了一次**：它把 `Reduction` 的 `axis` 报成"未覆盖"，
> 而 `zq_layerwire_check.cpp:583-584` 明明有 `axis=4`/`axis=-1` 两条期望拒绝的用例。
> 原因是我用"**标识符在门禁源码里出现过没有**"当判据 ——
> 门禁的用例表把 axis 存在 `c.a` 里，根本不出现 `axis` 这个词。
> **这个判据"未覆盖"不可信、"覆盖"才可能**，所以输出改成"候选"，逐条手工核实。
> 这是同一个坑（EP.2 / EQ）的第三次。

### 逐条手工核实

| 类 | 守卫 | 结论 |
|---|---|---|
| `Reduction` | `axis < 0 \|\| axis > 3` | **已有门禁**，探针误报 |
| `Pooling` | `!global_pool && (kernel_H<=0 \|\| … \|\| stride_W<=0)`（`:3945`） | 守卫正确、**零用例** |
| `Softmax` | `return axis >= 0 && axis <= 3 && …`（`:8500`） | 守卫正确、**零用例** |
| `Reshape` | `valid_num_axes=false`（`:8495`）→ `return valid_axis && valid_num_axes && …`（`:8506`） | 守卫正确、**零用例** |

三个都**不是缺陷**，都是**覆盖缺口**。

### 补进 zq_convparam（31 -> 49 例，零额外构建成本）

这三个类的 `Forward` 走的是 `ZQ_CNN_Forward_SSEUtils` 的**私有**辅助函数（那 44 个**绊线**），
所以只测 `ReadParam` —— 门禁只需要"参数能不能被拒"，碰不到 `Forward`，
也就不需要把绊线换成记录桩。复用的就是 `zq_convparam` 已编好的那批对象。

最有价值的是 `Pooling` 的**正例**：

    { C_POOLING, 0, 1, 1, 1, 0, 1 },   // global_pool=1 + kernel=0：必须放行
    { C_POOLING, 0, 1, 0, 1, 0, 1 },   // global_pool=1 + stride=0：必须放行

全局池化时 kernel/stride 本来就没有意义，守卫的 `!global_pool` 那一半是**豁免**。
若被"顺手简化"成无条件拒绝，**合法的 global_pool 模型会被全拒掉**，
而这条改写**不会有任何现有门禁报错**（根本没测过它）。

### 我自己错了两次，都是"把直觉当规格"

| 我写的期望 | 实际 | 真相 |
|---|---|---|
| `num_axes=4` 是合法上界 | 被拒 | 守卫是 `num_axes >= shape.size()`，`dim` 有 4 项 ⇒ 合法上界是 **3** |
| 参数名写 `dims=4,1,8,8` | 解析器不认 | 参数名是**重复的 `dim=`**（见 `model/*.zqparams`） |

第二条尤其值得记：`dims` 这个名字**是我按 "dimensions" 想出来的**，
看见"实际拒绝"还一度以为是代码有 bug，查真实模型才发现名字就不对 ——
**参数名不能按含义编，要从真实数据里抄**。

### 三个类的变异验证

| 变异 | 结果 |
|---|---|
| `Pooling` 的 `!global_pool` 改成 `true` | **RC=1**，**恰好 2 条** global_pool 正例红 |
| `Softmax` 的 `axis <= 3` 改成 `<= 9` | **RC=1**，**恰好 1 条**（axis=4）红 |
| `Reshape` 的 `num_axes < -1` 改成 `< -99` | **RC=1**，**恰好 1 条**（num_axes=-2）红 |
| 全部还原 | **RC=0**，49/49 |

第一条的"恰好 2 条"证明**豁免路径**也被钉住了，而不只是拒绝路径 ——
这正是"简化守卫"最容易踩的那一侧。

### 变更文件

- `tools/probe_readparam_coverage.py`（新，只读诊断；输出措辞改为"候选"）
- `tools/zq_convparam_check.cpp`（31 -> 49 例）

---

## 变更：附录 FA —— 把"条件化校验守卫"扫一遍（全仓只有 1 处）

### 为什么这类守卫要单独立一节

EZ 在 `Pooling::ReadParam` 发现的形状：

    if (!global_pool
        && (kernel_H <= 0 || kernel_W <= 0 || stride_H <= 0 || stride_W <= 0))
    { ...; return false; }

`!global_pool` 是**豁免**。关键在它是**单边风险**：
无条件守卫写错 ⇒ 合法模型被拒（sample 回归立刻发现）；
**条件化守卫被"顺手简化"掉 ⇒ 同样误拒合法模型，而没有任何现有门禁会报错**，
因为从来没人驱动过豁免路径。

### 结论：全仓只有 1 处

    ZQCNN/ZQ_CNN_Layer.h:3945  ZQ_CNN_Layer_Pooling  豁免条件 !global_pool

`ZQ_CNN_Net.h` / `ZQ_CNN_Net_NCHWC.h` / `ZQlibFaceID/*.h` 全扫过，**零处**。
这类风险已被 EZ 完全覆盖。

### 探针自己也错了两次（第四次同一个坑）

| 版本 | 问题 |
|---|---|
| v1 | 正则要求 `!cond` 后同一行必须有 `)` 或 `&&`，而 Pooling 那处**换行了** ⇒ 整条不匹配，**报 0 处** —— 正是它专为之写的形态 |
| v2 | 修好匹配后报 1 处 Pooling + **5 处 `ZQ_FaceDatabaseMaker` 误报**（它们的"比较"来自 `FindFace(..., 60, 0.709, ...)` 的实参） |
| v3 | 收紧成"整条条件只由比较/标识符/算术构成" ⇒ **恰好 1 处** |

今天这是第四次为"扫出来 0 命中"交学费（EP.2 两次、EQ 一次、FA 一次）。
写进 AGENTS.md 的规则仍然对，但**光写规则不够 —— 写完工具立刻做变异**。

**变异验证**：`!global_pool` 改成 `true` ⇒ 探针报 0 处；还原 ⇒ 报 1 处。
同一变异下 `zq_convparam` RC=1（EZ.5 已验）。

### 变更文件

- `tools/probe_exempt_guards.py`（新，只读诊断）

---

## 变更：附录 FB —— 补上 `ZQ_CNN_Net_NCHWC` 的行为门禁（EN 那处镜像改动一直是零覆盖）

### 缺口

EN 修就地守卫时在**两份** Net 上各改一份（`ZQ_CNN_Net.h` 与
`ZQ_CNN_Net_NCHWC.h`，各自独立的拷贝），但覆盖极不对等：

| 文件 | 覆盖 |
|---|---|
| `ZQCNN/ZQ_CNN_Net.h` | `zq_concat_alias` 走真的 `LoadFrom` 验，6 例 |
| `ZQCNN/ZQ_CNN_Net_NCHWC.h` | **只有编译覆盖**，行为上零门禁 |

**EN 那处镜像改动从来没有被任何东西验证过。** 与 ES.2 同形但更隐蔽 ——
它**编得过**，所以"能编过"这道轴也照不到。

### 门禁 zq_nchwc_net（5 例）

走真的 `ZQ_CNN_Net_NCHWC<T>::LoadFrom`，`T` 取 NCHWC1/4/8 **三种对齐变体**
（与 `SampleLnet106.cpp:41-48` 一致；守卫写在模板里，三种都得跑）。

| 用例 | 模型 | 期望 |
|---|---|---|
| 0 | Convolution → ReLU → Pooling，独立 blob | 放行 |
| 1 | `ReLU bottom=A top=A`（**就地安全层的豁免**） | **放行** |
| 2 | `Convolution bottom=data top=data`（同下标别名） | 拒 |
| 3 | `Eltwise bottom=A bottom=B top=B`（跨下标别名） | 拒 |
| 4 | 四段链，各 blob 独立 | 放行 |

用例 1 是 FA 那条"单边风险"的对应物：守卫若被"顺手"收紧成"一律不许 top==bottom"，
它会红 —— 而收紧的人不会想到自己拒掉了所有真实的就地 ReLU。

### 两个方向的变异验证

| 变异 | 结果 |
|---|---|
| 守卫退回"只比同一下标" | **RC=1**，**恰好用例 3** 红 |
| 去掉 `_is_inplace_safe` 豁免 | **RC=1**，**恰好用例 1** 红 |
| 全部还原 | **RC=0**，5/5 |

各只红 1 条 ⇒ 两个方向都被独立钉住（只测拒绝侧的抓不住"被收紧"）。

### 顺带把绊线桩 44 -> 104

NCHWC 那条 Net 走的是**另一个类** `ZQ_CNN_Forward_SSEUtils_NCHWC`，多 60 个符号。
生成器原来把类名**写死**，泛化成"按符号自己带的类名分组"，踩了三处：

1. 返回类型表按**名字**索引 ⇒ `ReLU` 在两个类里返回类型不同（`void` vs `bool`）
   ⇒ `SystemExit("return type mismatch for ReLU")`。改成按 `(类名, 函数名)`。
2. 只解析 NCHW 那个头 ⇒ `InnerProductPrePack` 这种只在 NCHWC 头里的重载找不到返回类型。
3. 前导没 include NCHWC 那个头 ⇒ 生成的头自己编不过，症状是**三道门禁一起 `0/1`**
   而不是某一个报错 —— 又一次"坏掉的不是被测对象，是用来测它的东西"。

探针 TU 同步扩了（实例化三种对齐变体），符号集仍**自动发现**；`--check` 与探针一致。

### 门禁自己踩的坑：两处类/参数名搞错，症状都是"应放行、实际拒绝"

1. **Pooling 用错类**：NCHWC 的 `ZQ_CNN_Layer_NCHWC_Pooling`
   （`ZQ_CNN_Layer_NCHWC.h:1979`）认 `kernel_size`/`stride`/`pad`，
   而**主库**的 `ZQ_CNN_Layer_Pooling` 认 `pool=`/`kernel_H=`/`pad_type=`。
   我按主库那份写，每个 key 都被报 `unknown para`。
2. **权重文件必须够长**：`dst_len = 4*3*3*3 = 108` 个 float；
   写空文件 ⇒ `Failed to load Binary` ⇒ `LoadFrom` false ⇒ **合法对照显示成"被拒"**。

> 三个症状共同点：**都在"守卫坏了"这个方向上伪装**。与 EZ.4 的 `dim=`/`dims=` 同一类。

### 变更文件

- `tools/zq_nchwc_net_check.cpp`（新门禁，5 例 × 3 种对齐变体）
- `tools/zq_net_fwd_tripwires.h`（44 -> 104 桩，重新生成）
- `tools/gen_net_fwd_tripwires.py`（按类名分组生成）
- `tools/zq_net_symprobe.cpp`（实例化 NCHWC 三变体）
- `tools/run_zqlib_checks.py`（`zq_nchwc_net` 接进 4 张表）

---

## 变更：附录 FC —— 把用户那条「sample 的 GUI 调用一律注释掉」变成门禁（C7）

### 规则一直成立，但没有任何东西在守

用户 2026-10-01 明确要求（AGENTS.md「示例程序规则」第一条）：
所有 Sample 里的 `cv::namedWindow` / `cv::imshow` / `cv::waitKey` 一律注释掉。

核现状：**71 个 sample 源文件、98 处 GUI 调用，全部已注释** —— 规则成立。
但它只是一条**文字规则**：谁加一行 `imshow`，Linux sample 回归就挂住或在无显示器
机器上直接失败，**症状离原因很远**。

### 判据：问编译器，别手写注释解析器

1. `g++ -E`（保留 linemarker）去掉注释；
2. 按 `# <line> "<file>"` 把每行归属到来源文件，**只看该 sample 自己的行** ——
   否则 OpenCV `highgui.hpp` 自己的 `imshow`/`waitKey` 声明会让每个 sample 都中；
3. 这样的行上出现活的 GUI 调用就是命中。

### 这个检查自己差点又一次静默全绿（当天第五次）

| 版本 | 症状 | 根因 |
|---|---|---|
| v1 | 73 个全"通过" | `-I` 含**不存在**的 `opencv4` 目录 ⇒ g++ 报错 |
| v1 修 | 加"零输出就报错"守卫，仍全绿 | g++ 失败时**也会先吐 6 行** ⇒ 守卫没响 |
| v2 | 真报错了（`ZQ_CNN_Net.h: No such file`） | `to_wsl(i[2:])` 把 **`-I` 前缀一起剥了**，目录变成裸参数被当成输入文件 |
| v3 | 4 分钟 | 每个 sample 一次 `wsl` 启动（73 次）⇒ 改成一次 WSL 调用编完全部 |
| v3 修 | `TypeError: expected str, not int` | `hits` 键写成下标，崩在报出任何结论**之前** |

> 第二行最值得记：**我加的守卫（"零输出就报错"）挡不住真实情况** ——
> 判据必须是 g++ 自己的**退出码 / stderr**。
> 第三行是 AGENTS.md「不要把 Windows 路径丢给 WSL 的 bash」的变体，
> **而我是在写了那条规则的同一个文件里踩的**。

### 变异验证

把 `SampleHeatMap.cpp:141` 的 `// waitKey(0);` 换成活的 `imshow+waitKey`：

| | 结果 |
|---|---|
| 变异体 | **RC=1**，报 `FAIL .../SampleHeatMap.cpp (1 处)`，并**逐字引用那一行** |
| 还原 | **RC=0**，`全部 71 个 sample：无未注释的 GUI 调用，且每一个都成功预处理过` |

内建阳性对照两半都验：注释掉的调用不得被报、活的调用必须被报。

### 放在慢组 + 组名用 C7

要对 71 个 sample 各跑一次 `g++ -E`，**实测约 4 分钟**，不进默认通道。
组名用 **C7**（`C4`/`C5`/`C6` 已被主流程占用 —— C4 那次撞名是我自己犯的）。

### 变更文件

- `tools/check_no_gui_calls.py`（新，带 `--selftest`）
- `tools/run_audit_checks.py`（新增 C7 组）

---

## 变更：附录 FD —— 把「每个随仓库模型都还能加载」变成门禁

### 缺口

EN 改的正是**模型加载路径**。改完只做了两件**一次性**的事：
`probe_inplace_topbottom.py`（静态扫 28 个 .zqparams / 8880 层，命中 0）
与一次性的 `zq_model_load_probe.cpp`（27 个模型跑一遍）。
**两个都不是门禁** —— 以后谁再动守卫，没有东西会告诉他某个随仓库模型被拒了。

而"误杀一个真实模型"正是这个改动最坏的后果（EN.5 就是为此才做了那次静态统计）。

### 不查权重的做法

权重 66 MB / 27 个 `.nchwbin`，进不了快速通道。要守的只是 EN 改的那一段：

    LoadFrom -> _load_param_file（各层 ReadParam）
             -> _check_connect（连通性 + 就地守卫）
             -> _load_model_file   <-- 66 MB，不查

给一个**故意不存在**的权重路径，LoadFrom 会把前两步跑完、只在第三步失败并打印
`failed to open`。于是判据是**失败消息**而不是返回值（两种失败都返回 false）：

| 输出里有 | 含义 |
|---|---|
| `failed to open` | 参数与连通性全过，只差权重 -> 通过 |
| `unknown blob` / `changes shape but declares top == bottom` / `missing ` / `invalid conv params` / `conv kernel/dilate overflow` … | **被守卫拒了** -> 失败 |
| 都没有 | 卡在别的阶段 = **没验到**，同样算失败 |

一个子进程一个模型，某个模型崩了不连累其余 26 个。

### 结果与变异

| | 结果 |
|---|---|
| 27 个随仓库模型 | **27 OK / 0 FAIL / 0 崩** |
| 放一个 `Concat bottom=A bottom=B top=B` 的模型 | **RC=1**，精确报 `被守卫拒绝: changes shape but declares top == bottom`，`共 28 个：OK 27，FAIL 1` |
| 删掉 | 回到 27/27 |

即 EN 那个守卫**没有拒掉任何随仓库模型**，而**确实**能拒掉真的别名模型。

### 门禁自己踩的坑：`_exit()` 不刷缓冲

第一版是 **27 个全 FAIL、且无任何拒绝标记**。逐层看：每条其实都打印了
`failed to open`，但父进程读到空串。

原因：父进程用 `_exit(child(...))` 收子进程，`_exit` **不跑 atexit、不刷缓冲**；
子进程 stdout 被 `freopen` 重定向到**文件** ⇒ 全缓冲 ⇒ 那行卡在缓冲里随进程消失。
补 `fflush(NULL); std::cout.flush();` 后 27/27 全过。

> 与 FC.3「零输出守卫挡不住真实情况」同族：缓冲/退出路径只有真跑一遍才暴露。

### 变更文件

- `tools/zq_model_params_check.cpp`（新门禁）
- `tools/run_zqlib_checks.py`（`zq_model_params` 接进 4 张表）

### FD.5 门禁被 harness 判成"1 条断言失败"，而它其实 27/0/0 全过

第一次进全量回归，`zq_model_params` 报 `FAIL (1 条 sanitizer 报错)`，
而它**一个模型都没拒绝**。

根因在 harness 的 ASan 判据（`run_zqlib_checks.py:866`）：

    echo "R|tag|$?|$(grep -cE 'FAIL' tag.out)|0"

它把输出里**含字面 `FAIL` 的行数**当作"断言失败数"，而我的汇总行**总是**打印
`共 27 个模型：OK 27，FAIL 0，崩 0` ⇒ 27 通过 / 0 拒绝也被算成"1 条断言失败"。

**不是我的门禁有 bug，是我没遵守那个约定**：其它门禁只在**真失败**时打 `FAIL`，
汇总行一律是"共 N 个：对 X，错 Y"。改掉措辞即通过。

> 与 DY.5（"消息为空"把排查方向带偏）同族：**门禁的输出格式本身就参与判定**。
> `run_zqlib_checks.py` 与各门禁之间有一份**不成文的输出格式约定**，
> 既没写进 AGENTS.md 也没有任何检查。

### FD.6 顺带补上"子进程非 0 退出"的判断

ASan 撞致命错误走 `Die()` → `_exit(1)`，**不发信号** ⇒ `WIFSIGNALED` 为假。
只判 `WIFSIGNALED` 的话，一次 sanitizer 崩溃会被记成"通过"。
加上 `WIFEXITED(st) && WEXITSTATUS(st) != 0` 才兜得住。

---

## 变更：附录 GE —— 剩下 3 个 UNUSED 层补上线（之前记的"做不了"只对 Forward 成立）

### 复核 EY.4 的"未做"理由

EY.4 记「这 3 个 UNUSED 层记为未做，理由是需要绊线 + 记录桩的双模式桩」。
复看之后，那条理由**只对 `Forward` 成立**：

- 这三处的守卫**全部在 `ReadParam` 里**；
- `ReadParam` 不碰 `Forward`，所以完全用不到记录桩 ——
  和 EY/EZ 一样，"构造层对象 + 喂一行参数"就够了。

那条"未做"的理由**把范围放大了**：它成立的那部分（Forward 接线）
本来就不在"守卫有没有被测"这个问题里。

顺带更正：EY.4 说"4 个 UNUSED 层"含 `DeConvolution`，但它早已被
`zq_convparam` 覆盖（EO.7 扩到 49 例时含 `C_DECONV`）。真正剩下的就是这三个。

### 门禁 zq_unusedlayers（18 例）

| 类 | 守卫 | 用例 |
|---|---|---|
| `LSTM_TF` | `has_hidden_dim && has_type && has_bottom && has_top && has_name` | 完整合法 + 逐项缺 5 个 |
| `PriorBoxText` | `!has_bottom \|\| bottom_names.size() != 2 \|\| !has_top \|\| !has_name`（**完全继承** `PriorBox::ReadParam`） | 完整合法 + 1/3 个 bottom + 缺 top/name/min_size |
| `DetectionOutput_MXNET` | `!has_bottom \|\| bottom_names.size() != 3 \|\| !has_top \|\| !has_name` | 完整合法 + 2 个 bottom + 1 个 variance + 缺 top/name |

最有价值的是**个数**那几条：`bottom_names` 的大小直接来自 .zqparams 里有几个
`bottom=`，而下游 `LayerSetup` 硬取 `(*bottoms)[0..2]` —— **少一个就是越界读**。

### 三个方向的变异验证

| 变异 | 结果 |
|---|---|
| `LSTM_TF` 去掉 `has_hidden_dim &&` | **RC=1**，**恰好 1 条**红 |
| `DetectionOutput_MXNET` 去掉 `bottom_names.size() != 3` | **RC=1**，**恰好 2 条**红 |
| `PriorBox` 去掉 `bottom_names.size() != 2` | **RC=1**，**恰好 2 条**红 |
| 全部还原 | **RC=0**，18/18 |

### 门禁自己踩的坑：期望值写错，靠"打印子进程消息"一眼定位

第一版 18 例错 1：`PriorBoxText` 的"完整合法"被**静默拒绝**。
给子进程 stdout 加**按用例分文件**的收集后，一行看到：

    输入: PriorBox ... variance=0.1 variance=0.2
    | Layer p must provide 4 variance

**我写了 2 个 `variance`，而 `_setup()` 要求恰好 4 个**（真实模型也是 4 个）。
是我的数据错了，代码是对的。

两件事值得分开记：

1. **"子进程的消息串到下一个用例"**：父进程在自己打印**之后**才 `waitpid`，
   子进程的消息出现在**下一行**。第一版我据此以为那例也报了
   `must have 2 bottoms`，差点去查一个不存在的解析问题 —— FD.4 踩过同一种错位。
2. **"把子进程的消息原样打出来"是把 20 分钟的猜变成一眼看到的关键**。
   与附录 CA「失败信息只说观察到的事实」是同一道理的正面用法：
   **让门禁自己把证据摆出来**，比让读报告的人推断便宜得多。

### 变更文件

- `tools/zq_unusedlayers_check.cpp`（新门禁，18 例）
- `tools/run_zqlib_checks.py`（接进 4 张表）
---

## 变更：附录 GF —— 27 个模型解析阶段零警告，并把它变成判据

### 先量现状

FD 那道门禁只判"模型有没有被守卫拒掉"，于是有一个它**看不到**的情况：
`.zqparams` 里参数名**打错**，解析器不认、打一行 `warning: unknown para`、
然后**静默走默认值** —— 模型照样"加载成功"，但那个参数根本没生效。
后果在推理上表现为"精度略差"，在日志里只是一行 warning。

把 27 个模型的解析输出全抓出来：**零** warning / missing / invalid / unknown para。
说明这批模型文件本身是干净的。

### 但"干净"只是今天干净 —— 变成判据

FD 增加一条：**解析阶段的 warning 必须为零**。

| 情况 | 旧判据 | 新判据 |
|---|---|---|
| 必需参数打错（`kernel_size`->`kenerl_size`） | 抓到（`missing`） | 同样抓到 |
| **可选**参数打错（`bias`->`bais`） | **抓不到**（模型正常加载） | **抓到** |

变异验证（都在 `model/det1.zqparams` 上，改完即还原）：

| 变异 | 结果 |
|---|---|
| `kernel_size=` -> `kenerl_size=` | **RC=1**，`被守卫拒绝: missing` |
| ` bias` -> ` bais`（**可选**） | **RC=1**，`解析阶段有警告（仍会加载成功，但参数可能没生效）` |
| 全部还原 | **RC=0**，27/27 |

第二条正是这条判据存在的理由：**旧判据对它完全无感**。

### 顺带修掉一个真问题：zq_concat_getsize_real.h 不是自包含的

用**另一个**编译顺序（只 include 它、不先 include `ZQ_CNN_Net.h`）编译时报：

    zq_concat_getsize_real.h:39:48: error: invalid use of incomplete type
      'class ZQ::ZQ_CNN_Forward_SSEUtils'

那个头只**前置声明**了 `ZQ_CNN_Forward_SSEUtils` 就去**定义它的成员函数** ——
定义成员需要完整类型。先前两个消费者都碰巧先 include 了 `ZQ_CNN_Net.h`，
把它间接带进来了。已补 `#include "ZQ_CNN_Forward_SSEUtils.h"`。

> 与 FA 那条"探针的正向对照必须取自**它专为之写的那个形态**"同族：
> **一个头在"恰好没踩到"的调用顺序下能用，不等于它是自包含的。**
> 我写这份头时是照 `zq_concat_alias_check.cpp` 的顺序写的，那个顺序下能编 ——
> 于是"能用"被当成了"对"。

### 变更文件

- `tools/zq_model_params_check.cpp`（新增"解析阶段 warning 必须为零"判据）
- `tools/zq_concat_getsize_real.h`（补 include，变自包含）
---

## 变更：附录 GG —— sample 回归只覆盖 44 个里的 8 个；外加我自己一次测错的退出码

### 覆盖率

| | 数量 |
|---|---|
| `SamplesZQCNN/` 下的 sample 目录 | **44** |
| Linux 回归跑的 / Windows 回归跑的 | **8** / **6** |
| 产物目录里实际存在的 Linux 可执行文件 | 54 |

「windows 和 linux 都能完全跑通」目前由 **8 / 44** 个 sample 支撑。

### 抽 12 个「看起来不需要外部资源」的实测

| sample | 输出 | 类别 |
|---|---|---|
| `CompareWithOpenBLAS` / `SampleMatMulNEON` / `SampleMatMulNEON_FP16` | 平台桩 | 桩 |
| `SampleFacialNet` | `failed to open file model/FacialNet.zqparam` / `failed to load net` | 要 Model Zoo 权重；**退出码 1，正确** |
| `SampleGEPB` / `SampleMatMul` / `example_for_very_high_gflops` | 基准输出 | 真跑 |
| `SampleSSDDetectorPytorch` / `model2code` / `swapRGBandBGR` / `testImageProcessing` | 打印用法 | 需参数 |
| `testWinoF2233` | **无输出** | 真跑，但会被 `NOOUT` 判失败 |

真正「能无条件跑」的只有三个基准 + 一个无输出的 `testWinoF2233`。
**处置**：本轮不扩回归列表（三个是基准会拖慢回归；`testWinoF2233` 无输出会被 `NOOUT` 判失败），
而是把覆盖缺口的量化事实记下来。

### 我自己测错了一次退出码，而且写进了报告

这一节原来的标题是「含一个 rc=0 但实际失败的实例」，
结论是 `SampleFacialNet` 打印失败却退出 0，并据此说
AGENTS.md 那条「退出码 0 只证明没崩」在真实 sample 上被证伪。

**那个结论是错的。** 根因是**我的测量**：

    out=$(timeout 90 ./$e 2>&1 | head -2 | tr '
' ' '); rc=$?

`$?` 取的是**管道最后一个命令（`tr`）的退出码**，恒为 0，
和被测程序毫无关系。

发现它是因为去查「为什么源码写着 `return EXIT_FAILURE;` 却退出 0」，
查到 `EXIT_FAILURE` 没被重定义、二进制比源码新、二进制里确实有那个路径 ——
一切正常，于是回头裸跑，三种写法三次都是 **1**。

> 这一条比原结论更有价值：**测量与源码矛盾时，先怀疑测量。**
> 我看到 `rc=0`、又看到源码里明明白白的 `return EXIT_FAILURE;`，
> 却把矛盾解释成「库有 bug」并写进报告 ——
> 而 AGENTS.md 恰好有一条现成规则：
> 「**没有证据就不要在失败信息里断言原因**」。
> 这次不是"没有证据"，是**证据本身错的**，而我拿它当证据用了。
>
> 推论：**凡"测出来的"和"读得出来的代码"冲突，先把测量重做一遍** ——
> 尤其当那个结果刚好支持一个我已经想好的结论时。

### 顺带查出的一个真缺陷：SampleFacialNet 的模型名拼错了

`SamplesZQCNN/SampleFacialNet/SampleFacialNet.cpp:42` 写的是
`"model/FacialNet.zqparam"` —— 扩展名**少一个 `s`**。
仓库里 27 个模型全是 `.zqparams`（289 处引用无一例外），
而 `model/FacialNet.*` 属 Model Zoo、不在仓库里，所以本地两种写法都失败，
**这个错字一直没被暴露**；下 Model Zoo 的人拿到手也会直接失败。已改正。

### 变更文件

- `SamplesZQCNN/SampleFacialNet/SampleFacialNet.cpp`（`.zqparam` -> `.zqparams`）
- `audit_k3_20261001.md` / `docs-changelogs/CHANGELOG_2026-10-03.md`（附录 GG 更正）
