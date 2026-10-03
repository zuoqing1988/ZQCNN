# CHANGELOG 2026-10-04

## 变更：附录 GN —— 两个平台的默认值算出来的数不是同一个数；顺手把 MSVC /analyze 接进回归


### GN.1 两个平台的默认档不同，位模式全不一样

`ZQCNN/ZQ_CNN_CompileConfig.h`：Windows 默认 **AVX2**、Linux 默认 **AVX**。
后者还决定 `FMADD128/256`（`>= AVX2` 才开）。

同一个固定种子的输入，分别链两档的 ZQ_GEMM，对结果 C 的原始字节做 FNV-1a：

| 形状 | SSETYPE=2 (AVX) | SSETYPE=3 (AVX2) | max\|C\| |
|---|---|---|---|
| 64x64x64 | de181db28a681f0d | ebe41ebc67128e2c | 9.90054（相同） |
| 128x128x128 | 9f2e43c1f0c4acc7 | a5bd9a315a076c08 | 15.4255（相同） |
| 256x256x8 | 1d71d618285b31c5 | 6984acf241a8a374 | 4.63772（相同） |
| 16x16x32 | aad105774abf7e13 | 28fd1cacb67fc091 | 4.75747（相同） |
| 64x64x8192 | 5584286bd594593a | 09260d5b21324a44 | 114.072（相同） |
| 384x384x384 | 2471f8eef424b7c8 | a67a8625135f7694 | 28.043（相同） |

**6/6 形状位模式都不同，而 max|C| 到 6 位有效数字完全相同** —— 差 1 个 ULP。

判据用位模式而不是误差：差 1 ULP 用误差判据会被浮点噪声淹没，
而对**人脸识别**，同一张图在两台机器上算出的 embedding 差最后一位
是可能翻转 1:1 判定的那一类差异。

原因不止 FMA：两档选中的内核集合本身就不同
（`zq_gemm_32f_auto.c` 里 `#if >= AVX` 与 `#if >= SSE` 是两套派发）。
准确说法是"两档产出不同的浮点结果"，FMA 是其中一个可辨识因素。

### GN.2 原来的档位差异没有换来任何兼容性收益

| 位置 | 内容 | 看 SSETYPE 吗 |
|---|---|---|
| 根 CMakeLists.txt:113 | `add_compile_options(-mavx2 -mfma)` | **不看** |
| 根 CMakeLists.txt:118 | `add_compile_options(/utf-8 /arch:AVX2)` | **不看** |

两个平台**本来就都无条件发 AVX2+FMA**。`ZQ_CNN_USE_SSETYPE`
只决定哪些代码路径被编进来，**不管制编译器能发什么指令**。
把 Linux 设在 AVX 挡不住任何老机器，只换来跨平台不可复现 + 用不上 AVX2 内核。

### GN.3 性能上是赚的：AVX2 相对 AVX 中位数 122%

25 个形状、空载机器、三轮交错（两档交替跑抵消"跑久了变慢"），
每形状取三轮里最好的一轮：

* 形状数 25，**中位数 122%**、平均 134%、最好 265%（512x512x512、
  1000x1000x512）、最差 80%（384x128x384）
* 比值 < 100% 的只有 4 个：3x3x3 90%、192x192x192 91%、
  384x128x384 80%、512x512x1 81%

**不是全面更快** —— 192x192x192 与 384x128x384 这两个"不是 2 的幂"的
实尺寸上 AVX2 反而更慢。但中位数与均值明显为正。

### GN.4 改了什么

`ZQ_CNN_CompileConfig.h` 的 Linux 分支：默认档 AVX -> AVX2。
现在被 `#ifndef` 包着（GL.1 加的），`cmake -DZQ_CNN_USE_SSETYPE=1|2` 仍能回退。

**这次改动是被门禁拦下来的**：C9 在改完之后立刻报"6 组期望全红"
（期望 SSETYPE=2、实际 SSETYPE=3），然后才改的期望表。
一条拦住了默认行为变更的门禁，比一条永远绿的门禁有用得多。

顺带把 C9 第 6 组从 `-DZQ_CNN_USE_SSETYPE=3`（默认值已是 3，变成恒真样例）
改成 `-DZQ_CNN_USE_SSETYPE=1`（验 FMADD 跟着关）。

**没有同步到 ZQCNN_to_MNN 的那份 CompileConfig**（它 Linux 侧仍是 AVX）：
那是个不进任何构建的分叉，改默认值既无法验证也没有收益；
按 EX 的规矩（只镜像已诊断为缺陷且主仓已修的守卫），
默认值这类**行为变更**不镜像。这条差异是有意留下的。

### GN.5 MSVC /analyze：一直存在，却从没在回归里跑过

`tools/run_msvc_analyze.py` 是附录 AW 写的，一直可用，但：
* AW 那一轮只跑了 **7 个 TU**，现在扫 **46 个**；
* 从没接进 `run_audit_checks.py`（报告 3792 行记作"尚未接"）。

全量结果（已排除 SDK 噪声）：25 个 `(TU, 告警号, 文件)` 组合、180 条。

| 告警号 | 条数 | 是什么 |
|---|---|---|
| C6246 | 50 | 局部变量遮蔽 |
| C6386 | 45 | 缓冲区溢出 |
| C6385 | 33 | 缓冲区越界 |
| C6326 | 5 | 常量与常量比较 |
| **C6011** | **4** | **解引用可能为 NULL 的指针** |
| C6235 | 5 | 恒真条件 |
| C6387 | 3 | 无效参数 |

真正值得看的两类：

1. **4 条 C6011** 在 `zq_cnn_batchnormscale_32f_align_c.c` 270/271/304/305：
   `a = malloc(...)`、`b = malloc(...)` 之后**立刻 `b[c] = ...`**，没判空。
   与报告 H16 那一族同源。严重性低（`in_C*4` 分配失败意味着进程已没内存），
   但确实是"未判空就解引用"。
2. **5 条 C6235** 就是 AW.1 记的那两个 `if (1 || ...)` 恒真条件，
   代码里有大段注释说明是**有意留的** —— 既有结论，不是新发现。

C6385/C6386/C6246 集中在 `*_raw.h` 生成代码里，绝大多数是 SIMD 对齐加载
**设计内**的越界读（靠 K 的对齐整除保证安全）。按「静态分析器是筛子，
动态实测才是判据」，这批不逐条改，只进基线拦新增。

基线键是 `(TU, 告警号, 文件名)` + **计数**，**不含行号** ——
行号会因无关编辑整体平移（check_filecount_bounds.py 的第一版键就栽在这），
存计数则"同一文件多一处同类发现"也抓得到。

#### 正向对照，以及它**没有**证明的那一半

在扫到的 TU 里注入真实形态的缺陷（`malloc` 未判空就 `p[0] = 1`）：
退出码 0 -> 1，报 `MORE zq_cnn_lstm_32f_align_c ... 0 -> 1`；还原后回 0。

**但抓到的是 C4312（未被引用的局部函数）而不是 C6011**，试了两次
（保留 / 去掉 `static`）都是 C4312。

所以这次对照**证明的是**"扫到的 TU 里出现新发现会被抓到"，
**没有**证明"C6011 会被抓到"。C6011 确实在基线里（4 条），
说明分析器在这份代码上会报这一类；但"它会报"与"我的注入会触发它"是两回事。

> 今天第四次"对照的判别力比看起来弱"。记下来不是因为对照失败，
> 而是因为**把一个只证明了 A 的对照当成证明了 A+B 来汇报，
> 是审计里最容易被后来人继承的错误。**

### 变更文件

- `ZQCNN/ZQ_CNN_CompileConfig.h`（Linux 默认档 AVX -> AVX2）
- `tools/check_blas_config.py`（期望表更新 + 第 6 组改成显式 SSE）
- `tools/run_msvc_analyze.py`（加 `--save-baseline` / `--check-baseline`）
- `tools/msvc_analyze_baseline.txt`（新增，25 组合 / 180 条）
- `tools/run_audit_checks.py`（接进 C11，慢组）
- `audit_k3_20261001.md`（追加附录 GN）

### 注意事项

- **性能数字来自一台机器、一个编译器**（WSL gcc 9.4，25 形状，不含 MKL 对比）。
- 位模式只比了 6 个形状、且只比 ZQ_GEMM 的一层 ——
  **没有**端到端比"同一模型在两个平台的 embedding"。
- **ARM/NEON 路径本机无法编译**（WSL 里没有 `arm-linux-gnueabihf-gcc`、
  也没有 clang），ARM 侧只做到"宏取值"（C9）与"四档 x86"（C10）层面，
  **NEON 分支的代码一行都没编过**。而仓库根的 `build.sh` 正是构建
  armeabi-v7a 的。


## 变更：附录 GO —— 手写汇编内核不在推理路径上（两个平台的构建产物里都量过）

### 起因

GN 那个"汇编对 MKL 98%"的数字。既然汇编这么接近 MKL，
那它**在哪儿被调用**就该问一句 —— 而这个问题读代码看不出来。

### 全仓只有三个地方调用汇编入口

```
SamplesZQBLAS/SampleGEMMCompare.cpp:296, 300
SamplesZQGEMM/SampleGEMMAsmCompare.cpp:44, 141
tools/zq_gemm_oob_check.c:43
```

两个是**对比用 sample**，一个是**审计探针**。**ZQCNN 库自己一次都没调。**

### 在实际构建产物上确证（不靠"读代码应该不会"）

Linux（v11 的 D2 构建目录）：

```
$ nm -u /tmp/zqb2/ZQCNN/libZQCNN.a | grep -c auto_asm
0
$ nm -u /tmp/zqb2/ZQCNN/libZQCNN.a | grep zq_gemm
U zq_gemm_32f_AnoTrans_Btrans_auto     <-- 全部指向 intrinsic 派发器
U zq_gemm_32f_align0_AnoTrans_Btrans
$ nm --defined-only /tmp/zqb2/ZQ_GEMM/libZQ_GEMM.a | grep auto
T zq_gemm_32f_AnoTrans_Btrans_auto_asm  <-- 定义了
T zq_gemm_32f_AnoTrans_Btrans_auto     <-- 也定义了
（逐个 .o 扫：libZQ_GEMM.a 里 0 个对象引用 _auto_asm）
```

Windows：

```
$ dumpbin /symbols build_x64/ZQCNN/Release/ZQCNN.lib | grep -c auto_asm
0
```

**两个平台都是 0**，而且连 `libZQ_GEMM.a` 内部都没有调用者 ——
汇编入口是被 sample 从外面拉进去用的。

### 这意味着什么

* ZQCNN 卷积/内积的 GEMM 走 `zq_cblas_sgemm`
  → `zq_gemm_32f_AnoTrans_Btrans_auto`（**intrinsic 派发器**）
* 手写汇编内核只能被两个对比 sample 和一个审计探针调到
* **没有运行时开关**：`ZQ_GEMM_ISA` / `zq_gemm_32f_asm_isa_usable`
  管的是"汇编路径内部要不要回落 intrinsic"，不是"要不要走汇编"

所以"汇编达到 MKL 的 98%"这件事，**与 ZQCNN 的推理速度无关**。

这不是缺陷，是**一个需要所有者决定的事实**：汇编路径可能是**刻意**不进主路径
（README 把 `SamplesZQGEMM` 描述成"对比程序"），但"刻意的"与"忘了接上"
在仓库里长得一模一样 —— **代码里没有任何一处注释说明"汇编暂不进主路径"**，
而 `ZQ_GEMM/CMakeLists.txt:8` 读起来像是**在用**。

按 AGENTS.md「推不动就把排除了什么记下来」，**本轮不做代码改动**。
真要接，最小改动是在 `zq_gemm_32f_auto.c` 的派发器里加一层
"先试 `_auto_asm`、失败回落 intrinsic"（汇编文件里已有
`zq_gemm_32f_asm_isa_usable` 这套探测，落地点是现成的），
但那是**新增功能**，且会改变默认推理路径的性能与数值结果 ——
得先逐形状验证并由所有者拍板。

### 顺带把 Linux 侧 asm/MKL 在新默认档下重测

`3rdparty/mkl_runtime/linux/libmkl_rt.so.2` 在，所以能测。
`SampleGEMMCompare`（v11 D2 用新默认档 AVX2 编的）64 形状：

| | 之前记录（AVX） | 现在（AVX2） |
|---|---|---|
| asm/MKL 中位 | 99% | **98%** |
| asm/MKL 平均 | — | 143% |
| 最好 / 最差 | — | 1347%（1x1x1）/ 47%（1024x1x1） |
| < 90% / < 60% 的形状 | — | 22 个 / 2 个 |
| asm/intrinsic 中位 | — | 1.38×（最大 25.94×） |
| 最大 err(asm) | — | 1.5e-05 |

**比例基本没变（99% -> 98%，噪声内）**，这与 GN 不矛盾 ——
两者量的**不是同一个函数**：

| | 量的是 | SSETYPE 改动的结果 |
|---|---|---|
| GN 的 25 形状微基准 | `zq_gemm_32f_AnoTrans_Btrans_auto`（**intrinsic 派发器**，库真正在跑的那条） | **中位 +122%** |
| 这张 64 形状表 | `..._auto_asm`（汇编入口，**库不用**） | 不变 |

原因对得上：汇编内核选不选、要不要 FMA 判的是 `ZQA_IMPL`（`SSETYPE >= AVX`）
与 `ZQA_HAVE_FMA`（`defined(__FMA__)`），**两者在 AVX 与 AVX2 下都是真** ——
那一档的改动对汇编路径没有任何影响。

这同时把 GN 那句话说准了：**默认值改动的收益落在库真正在跑的那条 intrinsic
派发器上（+22% 中位），不是落在汇编内核上。**

### 变更文件

- `audit_k3_20261001.md`（追加附录 GO）
- `docs-changelogs/CHANGELOG_2026-10-04.md`（本节）

### 注意事项

- 汇编 vs MKL 的 98% 是**单线程**口径（sample 默认强制 MKL 单线程，
  `SampleGEMMCompare --mt` 可以不强制）。多线程口径未测。
- Linux 侧 64 形状表是**单次读数**（每个形状内已取多轮最好的一轮），
  跨机器不可直接比。


## 变更：附录 GO.5 / GO.6 —— 把"库内无人引用"变成常驻类别，并找到第二例

### 新类别

`tools/check_dead_tu.py` 原来只判"**有没有人**引用这个 TU 的外部符号"，
于是对汇编内核那个 TU 它答"有人用"（两个 sample 确实在调），
**"库自己一次都没调过"被完全吞掉**。新增一节输出：
**库内无人引用的 TU**（全部外部符号都只被 `Samples*` / `tools*` 引用）。
当前 2 条：

```
layers_nchwc/zq_cnn_addbias_nchwc.c     3/3
math/zq_gemm_32f_align_c_asm.c          8/8
```

### 三次返工，两次是我的错

| 版本 | 做法 | 结果 |
|---|---|---|
| v1 | 库内引用 = "在库内源文件里 grep 到符号名" | **假阳性**：`zq_gemm_32f_auto.c` 被报成"库内零引用"，而 libZQCNN.a 上明明有 `U zq_gemm_32f_AnoTrans_Btrans_auto`。库里的引用是**经宏**进来的（`zq_cblas_sgemm(...)` 才展开成那个名字），源码文本里一次都不出现 |
| v2 | 加链接器视角（库 .o 的未定义符号算"库在用"） | 仍假阳性 —— `nm --undefined-only` 的输出是 **2 段**（`         U name`，无地址列），而解析要求 ≥3 段，46 个 .o 解析出 **0 个**符号 |
| v3 | 两种格式都认 + 库内 .o 按**目标文件路径**判 | 46 个 .o / 581 个未定义符号，类别从虚假的 40 条收敛到 2 条 |

外加一次"过滤条件用错字段"：给类别加"只统计本身是库源的 TU"时用了
`is_library_src(r['tu'])`，而 `tu` 也是不带 `ZQCNN/` 前缀的部分路径 ——
**两条真结果被自己滤掉，类别变成 0 条**。

> 四次返工的共同形状：**每一版都"看起来在跑"**（有输出、有类别、有数字），
> 错的是**判据本身**。工具在跑 ≠ 工具在说真话。

### 判别性对照：第一次、第二次都没成立

| 尝试 | 注入 | 结果 |
|---|---|---|
| 1 | 库内文件加一行 `_auto_asm` 的**声明** | 无变化 —— 声明不产生未定义符号引用 |
| 2 | 在 `zq_gemm_32f_auto.c` 函数体里塞 `static` 一次性调用 | build rc=2（括号嵌套把函数体搞坏） |
| 3 | 同一文件**末尾**加一个自包含的非 static 探针函数，真调一次 | build rc=0，**类别 2 -> 1**（汇编 TU 消失）；还原后回到 2 |

> AGENTS.md 那条"选错了变异目标，不是探针的错"今天又撞了一次 ——
> 而且这次**连续两次**选错，第三次才对。

### GO.6 第二例：NCHWC 的 AddBias 专用内核同样没接线

顺着新类别查第二条：

* `zq_cnn_addbias_nchwc{1,4,8}` 只出现在自己的 `.c`、声明它的 `.h`，
  以及一个审计门禁 `tools/zq_nchwc_act_check.cpp`；
* `ZQ_CNN_Layer_AddBias::Forward`（`ZQCNN/ZQ_CNN_Layer.h:3007`）走的是
  `ZQ_CNN_Forward_SSEUtils::AddBias`，不是这三个内核；
* `ZQ_CNN_Forward_SSEUtils_NCHWC.cpp` 里有三个变体的 `AddBiasPReLU`，
  **但没有 `AddBias`** —— NCHWC 张量的 AddBias 落回基类通用实现。

不一定是缺陷（通用实现走张量虚函数，NCHWC 张量可能自己就够快），
但同样**没有任何注释说明"这个专用内核暂不接"**。

这是**第二个**"写了但没接线"的例子，所以这一类值得单独立一条：
接线还是删掉，是所有者的决定，不是审计能替做的。

### 变更文件

- `tools/check_dead_tu.py`（新增"库内无人引用的 TU"一节、
  `undefined_symbols()`、`is_library_obj()`；修 `nm` 两段输出的解析）
- `audit_k3_20261001.md`（追加 GO.5 / GO.6）
- `docs-changelogs/CHANGELOG_2026-10-04.md`（本节）

### 注意事项

- 本工具**没有**基线模式，也没接进回归 —— 它是个诊断工具，
  输出给人看。这一类目前靠"谁想起来跑"驱动，不是常驻门禁。
- 它要求 Linux 构建目录里有 .o（默认 `/tmp/zqb2`），没有就退出。
  完整回归的 D2 段会在 `/tmp/zqb2` 留下构建产物，所以回归跑完之后
  可以直接用它。


## 变更：附录 GP —— ARM/NEON 分支：36 个 TU、一行都没被任何编译器看过

### 缺口

| | |
|---|---|
| 带 `#if __ARM_NEON` 分支的 TU | **36 个**（ZQ_GEMM/math 1 + ZQCNN/layers_c 17 + layers_nchwc 18） |
| NEON 区域里的调用点 | **1606 处**，21 个不同名字 |
| 本机 ARM 交叉工具链 | **没有**（`arm-linux-gnueabihf-gcc` / `aarch64-linux-gnu-gcc` / `clang` 全无） |
| 仓库自带的 ARM 构建脚本 | **有** —— 根目录 `build.sh` 就是 armeabi-v7a |

即：**仓库带着一条从未被验证过的构建路径**，而它正是移动端部署要用的那条。
GM 已经证明这条轴上"没人看"会漏东西（`zq_gemm_32f_asm_core_m6n8` 缺前置声明，
只在 SSETYPE=0/1 出现，默认两档连 warning 都没有）。

### 做法

`tools/arm_neon.h`（桩）+ `tools/check_neon_branch.py`（驱动）：

1. 扫出所有 NEON 区域用到的 `v*q_*`，与桩比对 —— **桩缺哪个名字门禁直接喊**。
   完整性由门禁自己保证，不靠人手清单。**第一次跑就喊出一个：`vdivq_f16`。**
2. 量一次"NEON 区域里 `__asm__` 的数量必须是 0"（方案成立的前提，每次都量）。
   实测 0 个。
3. gcc `-fsyntax-only -DZQ_CNN_USE_ARM_NEON`，`-I tools/` 最前，逐个编那 36 个 TU。

### 三次返工，两次是我的错，第三次改对了

| 版本 | 做法 | 结果 |
|---|---|---|
| v1 | 桩叫 `arm_neon_stub.h`；`#define ...(...) 0`；`.h` 也收 | 46 个"失败"：一半 `arm_neon.h` 找不到、一半"把 `*_raw.h` 当 TU 编" |
| v2 | 改名 + 只收 `.c/.cpp` + 逐个名字定义 | 36 个全部编过（0 失败）—— **但什么都没验** |
| v3 | 桩改成 `((void)sizeof((__VA_ARGS__, 0)), 0)` | 两个对照都成立 |

v2 绿得毫无破绽，而它**什么都没验**：对照 B 注入
`vst1q_f32(q, zq_gp_never_declared)`（NEON 分支里用一个未声明的变量），
**编译通过** —— `#define ...(...) 0` 把参数整个丢掉，
`zq_gp_never_declared` 压根没进 token 流。而 NEON 代码的绝大部分正好在实参里
（`vfmaq_laneq_f32` 一个名字 888 处）。

> **"全绿"和"什么都没验"在输出上一模一样**，而且这次是我自己写的桩。

### 两个正向对照

| 对照 | 注入 | 结果 |
|---|---|---|
| A | NEON 内在函数名拼错（vaddq_f32 -> vaddq_f33） | 退出码 1（走**桩完整性**那条路） |
| B | NEON 分支里用一个未声明的变量 | 退出码 1（编译报"编不过 1 个" + 裸跑失败即失败） |

两个注入都挂在 `#if __ARM_NEON` 包起来的探针函数上 ——
注到 x86 那侧的话默认回归早就会红，而门禁要证明的是
**它能看见 NEON 分支里的问题**。还原后退出码回 0。

顺带修掉一个真 bug：**裸跑时"有文件编不过"却不影响退出码**。
第一版只在 `--check-baseline` 分支里把失败变成问题，于是对照 B
"被抓到"（打印了）却又"没被抓住"（rc=0）。
与 GK.2 那条元门禁同源：**失败必须落在退出码上，打印出来不算。**

### 它证明什么、不证明什么

**能**：NEON 分支能被解析；其中的普通 C（下标算术、打包循环、尾部处理）
过真正的类型检查；每个 NEON 内在函数名都在桩里，拼错会立刻报；
内在函数**实参里的表达式**也被检查（靠 sizeof，见上面 v3）。

**不能**（写下来是为了不让它被当成"ARM 路径验过了"）：
桩把向量类型全 typedef 成 `float`，所以**不验 NEON 的类型**；
所有内在函数返回 0，所以**不验语义**（对齐/饱和/舍入都验不了）；
`__ARM_NEON_FP16` 那段**仍未覆盖**。

### 现状

`tools/neon_branch_baseline.txt` **是空的** —— 36 个 TU 的 NEON 分支现在全部能解析。
门禁以 **C12**（慢组，约 1 分钟）接进回归。

实测 C5 可达性基线**不受影响**（`7 entries unchanged`）：
那个探针的扫描范围是"从 CMakeLists 列出的 TU 沿 `#include` 传递闭包"，
`tools/` 下的门禁文件既不是入口也不在任何闭包里。
（我写报告时预测"需要重存基线"，**实测是错的，已按实测改。）

### 变更文件

- `tools/arm_neon.h`（新增，NEON 桩）
- `tools/check_neon_branch.py`（新增，C12 门禁）
- `tools/neon_branch_baseline.txt`（新增，空基线）
- `tools/run_audit_checks.py`（接进 C12，慢组）
- `audit_k3_20261001.md`（追加附录 GP）


## 变更：附录 GQ —— `SIMD_ARCH_TYPE=arm64-fp16` 是一个「存在但不工作」的 CMake 选项

GP 把 ARM/NEON 的 f32 档编过了，并明确留下 FP16 档没覆盖。补上之后证明：
那一档从来不能工作，且有**三个互相独立**的缺陷。

### 缺陷一：`float16_t` 全仓从未被定义

被 **45 个文件**用到，而 `grep -rn "float16_t" | grep -E "typedef|define"`
命中的全是 `#define zq_base_type float16_t` 这种**使用**。
实测 FP16 宏组合下 36 个 TU 里 **23 个编不过**，全部同一个错：
`error: unknown type name 'float16_t'`。

修：在 `ZQCNN/ZQ_CNN_CompileConfig.h` 加受控 typedef。
为什么是 `__fp16`：ACLE 只保证 `__fp16` / `_Float16`，
`float16_t` 是 GCC 12+ / Clang 14+ 才有的**内建类型名**，
在那些编译器上再 typedef 会与关键字冲突，所以默认只给老编译器补，
可用 `ZQ_CNN_FLOAT16_T_BUILTIN` 覆盖。
**本机没有 ARM 工具链，"补完数值正确"没有证据**；这里只保证能编过。

### 缺陷二：15 处「通用实现与 FP16 实现同时被编进来」

7 个文件里 15 个函数/类型被定义两次：通用实现**无守卫**，
FP16 实现包在 `#if __ARM_NEON_FP16` 里。

修了 6 个文件、11 处。定位不靠猜：
- 第二个定义 = gcc 报的 `error: redefinition of` 行号；
- 第一个定义 = gcc 紧跟着的 `note: previous definition ... was here` 行号；
- 函数结尾用花括号配对（跳过注释与字面量）。

改完**双向验**（FP16 档与 f32 档都 0 失败）。这一步当场抓到我自己的错：
第一个自动区间把 `zq_gemm_32f_align_c.c` 的 128..537 行整块包进去，
而那里面有 f32 需要的 `zq_base_type` / `zq_mm_*`，**f32 档立刻红 462 个 error**。
双向验不是形式主义。

### 缺陷三：2 处 `padK` 未声明（同 H9 那一族）—— 故意没修

```c
#if __ARM_NEON
#if !__ARM_NEON_FP16
    int padK = (K + 3) / 4 * 4;
#endif                       <-- 没有 #else
#else
   /* x86 分支是三档齐全的 #if/#elif/#else */
#endif
...
int matrix_A_cols = padK;   <-- 无条件使用
```

同一个变量、同一个文件，两侧写法不一致。

**故意不补那个 `#else`**：补了能编过，但 padK 的取值只能猜
（与上面 f32 那行一致是唯一自洽选择），而我无法在 ARM 硬件上验证。
把一个响亮的编译失败换成可能静默算错的数值，是更坏的结果。

### 修完之后 FP16 档还剩 6/36 编不过，全部进基线并逐条写明理由

`tools/neon_fp16_baseline.txt`（回归 C12b 组盯着）：
`zq_gemm_32f_align_c.c`（重复定义在 #define 宏块内部，自动区间会伤到 f32，
要正确修必须人读那个宏块）、两个 `padK`、`zq_base_type` 落空、
一个语法错误、一个**桩的限制**（`ZQA_NEON_STUB_ANY` 展开成
`((void)sizeof(...), 0)`，代码把内在函数结果当可调用对象用时报的）。

所以现状是：从「23/36 编不过」变成「6/36 编不过，且每条写清为什么现在不修」。
**不是修好了，是从"没人知道"变成"有据可查且被盯着"。**

### 文档

`build-with-cmake.md` 的 arm 段加警告：**这个选项存在，但它不工作，
在它修好之前不要用。** 根 CMakeLists.txt:95 确实实现了它，
而那段文档之前只写了 arm 与 arm64。

### 变更文件

- `ZQCNN/ZQ_CNN_CompileConfig.h`（float16_t 的受控 typedef）
- `ZQCNN/layers_c/` 6 个 .c（FP16 与通用实现互斥的守卫，11 处）
- `tools/check_neon_branch.py`（--fp16 档；错误原因抓取从 head -3 改成扫整个日志）
- `tools/arm_neon.h`（__fp16 桩）
- `tools/neon_fp16_baseline.txt`（新增，6 条，逐条带理由）
- `tools/run_audit_checks.py`（接进 C12b，慢组）
- `audit_k3_20261001.md`（追加附录 GQ）、`build-with-cmake.md`

### 注意事项

- 6 个被改的 .c 文件原本是 CRLF，我的脚本用 `\n` 写入造成 mixed-EOL，
  已由 `check_line_endings.py --fix` 修回（`line endings OK`）。
- 本附录所有结论都来自**解析**层面（桩 + gcc -fsyntax-only），
  **没有一条来自真实 ARM 硬件上的运行**。
