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
