# AGENTS.md

## 变更日志规则（必须严格遵守）

写变更日志到 `docs-changelogs/CHANGELOG_YYYY-MM-DD.md` 时：

1. **先检查当天日期的日志文件是否已存在**（用 `ls docs-changelogs/` 或直接 Read）。
2. **已存在 → 只追加，不覆盖**：必须先 Read 读出现有内容，把新章节追加到文件末尾（Edit 在末尾锚点插入，或 Write 时用"现有内容 + 新章节"的完整内容）。严禁直接 Write 覆盖导致已有章节丢失。
3. **不存在 → 新建**：以 `# CHANGELOG YYYY-MM-DD` 开头。
4. **每次写日志前先执行 `date '+%Y-%m-%d'` 确认当天日期**，再决定写入哪个文件——不得凭会话惯性沿用上一次写日志的日期。会话可能跨天或间隔数日继续（教训实例：2026-08-10 的实测记录曾误写入 CHANGELOG_2026-08-07.md，事后迁移订正）。
5. 章节格式沿用既有风格：`## 新增/变更：标题` + `### 变更文件` / `### 实测结果` / `### 注意事项`。
6. 写完日志后发现他人（或前一个任务）又加了内容时，合并保留双方章节，不得整文件重写丢弃。

## 构建规则

1. **CMake 是唯一构建入口**：仓库已于 2026-10-01 移除全部 `.sln/.vcxproj*`（停留在 VS2015/v140、SDK 8.1，文件列表早已与源码脱节）。不要重新引入 MSVC 工程文件。
2. **双平台必须都编译通过**：改动 C/C++ 后要分别验证 Windows（`cmake --build build_x64 --config Release`）与 Linux（`wsl -d Ubuntu-20.04` + gcc 9，`/tmp/zqb` 下 cmake + make）。MSVC 能过不代表 gcc 能过。
3. **不要依赖 MSVC 的传递包含**：MSVC 会顺带引入 `<cfloat>` `<cmath>` 等，gcc 不会。新写的代码要显式 `#include` 用到的标准头（已踩坑：FLT_MAX 在 gcc 下报未声明）。
4. `ZQ_GEMM/CMakeLists.txt`、`ZQCNN/CMakeLists.txt`、`SamplesZQCNN/CMakeLists.txt` 都用 `file(GLOB ...)`，**新增 .c/.cpp 文件无需改 CMake，但需要重新 configure 才生效**。

## 提交规则

1. 阶段性成果就 commit（构建修复、审计修复、文档、报告各自成次）。
2. **不要 push**，推送由用户自己决定。
3. 一个 commit 只做一件事，提交信息用中文写清楚改了什么、为什么。

## 示例程序规则

1. 所有 Sample 中的 `cv::namedWindow` / `cv::imshow` / `cv::waitKey` **一律注释掉**（无头环境与自动化验证会阻塞；用户 2026-10-01 明确要求）。
2. 跑示例前先确认没有等待按键的调用。


## 汇编/低层代码规则

1. **MSVC x64 不支持函数体内联汇编**：`__asm { }` 和 `__declspec(naked)` 在 x64 目标上都会编译失败（2026-10-01 实测 MSVC 14.35 报 C2143/C4235）。Windows 侧的手写汇编必须走独立 `.asm` 文件（CMake 里 `enable_language(ASM_MASM)` + `.asm`），GCC/Clang 侧才用 `__asm__ volatile` 内联汇编。写跨平台低层代码前先想清楚这两条路径。
2. `3rdparty/lib/libncnn.a` 是 clang 编译的，引用 `__exp_finite`/`__log_finite` 等 compiler-rt 符号，用 gcc 链接时由 `ZQCNN/math/zq_libm_compat.c` 补齐，不要删。
3. 基准测试程序里 MKL / OpenBLAS 一律**运行时动态加载**（`LoadLibrary`/`dlopen`），不引入链接期依赖；MKL 运行时放在 `3rdparty/mkl_runtime/`（已 gitignore），Linux 下 `libmkl_rt.so.2`、Windows 下 `mkl_rt.3.dll`。

## 行尾与跨平台编译规则

1. **内核头文件（`*_raw.h`）在仓库里必须是纯 LF**。这些文件里全是跨行宏（行尾 `\` 续行），一旦被提交成 `\`+CR+CR+LF，MSVC 能容忍而 **gcc 的行拼接会失效**——宏在第一行就被截断，表现为"文件作用域出现未声明标识符"之类的编译错误；更糟的是增量构建会**沿用旧的目标文件**，让 Linux 侧跑出一个和源码完全对不上的旧二进制。`.gitattributes` 已经给这些文件标了 `eol=lf`。
2. 改完内核头文件后，**Linux 侧要确认目标文件真的被重新编译**（看 `make` 输出里有没有 `Building C object ...raw...`），不要只看 `make` 的返回码。
3. 本机 `core.autocrlf=true`，提交时 git 会把工作区的 CRLF 归一成 LF 存进仓库——所以**"git diff 干净"不代表工作区行尾干净**，而工作区才是编译器真正读的东西。
4. **改完代码跑一次 `python tools/check_line_endings.py`**。它查三类问题：multi-CR（`\r\r\n`，会让 `\r` 并进 `#include`/`#ifndef` 的预处理符 token，是 UB）、lone-CR、以及**同一文件里 CRLF 与裸 LF 混用**。`--fix` 可以自动规范化。纯 LF 文件（`*_raw.h`）不会被误判。
5. **用 Python 批量改写源码时必须自己保证行尾**：`open(p,'rb').read().split(b'\n')` 再 `b'\n'.join(...)` 这种写法，**新插入的行不带 `\r`**，工作区立刻变成 CRLF/LF 混用。正确做法是插入时补 `b'\r'`，或改完立刻跑 `--fix`。2026-10-01 批量修 MTCNN 时就是这么踩到的。
6. **不要凭"以前的修复报告写了什么"来判断覆盖面**。第四/五轮声称边框清零已覆盖 `ROI`，实际只改了 `Resize*`；`NCHWC1/4/8` 整个系列一处没改。收口一类缺陷时要**全仓枚举同类站点**（`grep` 出所有出现位置逐个核对），而不是只信上一轮的清单。

## 配置宏的唯一真相

1. `ZQ_CNN_USE_ZQ_GEMM` / `ZQ_CNN_USE_BLAS_GEMM` / `ZQ_CNN_USE_BOTH_BLAS_ZQ_GEMM` / `ZQ_CNN_USE_MKL_GEMM` / `ZQ_CNN_USE_SSETYPE` **在根 CMakeLists.txt 与 `ZQCNN/ZQ_CNN_CompileConfig.h` 里各有一套**。改任何一边都要同步另一边，否则会出现"链了 MKL 却仍用 ZQ_GEMM 算"这种静默走错实现。
2. CMake 里判断 `BLAS_TYPE` 一律用 `string(TOLOWER ...)` + `STREQUAL` 精确比较，**不要用 `MATCHES`**（子串匹配会让 `openblas_zq_gemm` 被 `openblas` 抢先命中、默认的 `ZQ_GEMM` 匹配不到 `zq_gemm`）。这一点已经踩过坑。
3. `BLAS_TYPE` 的分支必须放在 `SIMD_ARCH_TYPE` 判断**之外**，否则 x86 上 `-DBLAS_TYPE` 形同虚设。
4. 给头文件里的配置宏加 `#ifndef` 保护之前，先想清楚命令行 `-D` 与头文件的优先级，否则改了一处会静默改变另一处的行为。

## ZQCNN 里容易看反的 API 约定

1. `ZQ_CNN_Tensor4D::ResizeBilinear/ResizeNearest/...` 是 **const** 方法，形如 `input.ResizeBilinear(dst, W, H, ...)` 里 **`input` 是源、`dst` 是目的地**。同理 `src.ROI(dst, ...)`。看错方向会把"并发写同一个对象"误判成数据竞争（我在这上面误报过一次），也会把修复做反。
2. `LoadFrom/LoadFromFile` 失败后对象可能处于半更新状态；重复调用 `Init()` 的安全性要看具体实现，别假设。
3. 报告"修之前先确认方向"：本项目里"看起来是 bug"的地方有一半是 API 约定与直觉相反（见上面两条，以及 `zq_cnn_eltwise_*` 里"增量加到 `*_im_ptr` 还是 `*_slice_ptr`"这类必须逐元素推演的地方）。

## 三个张量变体对"越界 rect"的策略是互相冲突的（Align128bit 的那个是 MTCNN 赖以工作的）

`ZQ_CNN_Tensor4D_NHW_C_Align0` / `_Align128bit` / `_Align256bit` 的
`ResizeBilinearRect` / `ResizeNearestRect`，越界检查写在三个不同位置：

| 变体 | `if (越界)` 的分支体 |
|---|---|
| Align0 | `return false;` |
| Align256bit | `return false;` |
| **Align128bit** | **`<正常 resize, 不报错>`** —— 也就是说越界 rect **被接受**并照常做 resize |

而 `Align128bit::ResizeBilinearRect` 那个"正常 resize"分支**不含**
`can_call_safeborder` 优化，也不走 ROI 快捷路径；不含它的 `else` 分支才含。

**这不是笔误能直接改掉的**：全部 5 个 MTCNN 变体的 `Find()` 入参都是
`ZQ_CNN_Tensor4D_NHW_C_Align128bit& input`，并且它们**故意**传入越界 rect ——
`ZQ_CNN_MTCNN*.h` 里 16 处

```cpp
if (/*off_x < 0 || off_x + rect_w > width || off_y < 0 || off_y + rect_h > height ||*/ ...)
```

的边界检查是被**注释掉的**（检测框不做图像边界裁剪，靠 resize 内核自己去读
border）。把 Align128bit 的守卫改成 `return false` 之后，
`SampleCascadeOnet` / `SampleCascadeOnet_Interface` 立刻 `Find()` 返回 false
（一张脸都检不出），实测确认过。

2026-10-01 的处理：只把 **`ResizeNearestRect`**（标量版，全仓零调用方）
的守卫对齐成 `return false`；**`ResizeBilinearRect` 保持原样**，并在
`audit_k3_20261001.md` 附录 N 记为"已知的不一致，改它必须先修 MTCNN 的 rect
越界"。**不要单独改 `Align128bit::ResizeBilinearRect` 的守卫。**

## ZQ_GEMM 的数据布局（写内核前必须先确认，否则结果全错且不崩）

`zq_gemm_32f_AnoTrans_Btrans_*` 的语义是：

```
C[i][j] = sum_k A[i*lda + k] * Bt[j*ldb + k]
A  : M x K 行主序      Bt : N x K 行主序（注意是 N x K！）      C : M x N 行主序
```

**唯一在 A 和 Bt 里都连续的维度是 K。** C 沿 N 连续，但 `Bt[j][k] = Bt[j*ldb + k]`
意味着相邻两列在内存里差 `ldb` 个 float，**不是相邻的**。

所以"沿 N 方向一次取 8 列、乘上 A 的一个标量、写出 8 个 float"这种写法
（`vbroadcastss (A[k])` + `vfmadd231ps (Bt + n*ldb)` + 一次 32B store）**是错的**：
那条 `vfmadd231ps` 的内存操作数读的是 `Bt[n*ldb + 0 .. n*ldb + 7]`，即同一列的
K 元素加上后面几列的头部，8 个结果里有 7 个是错的。

这正是 2026-10-01 我给 `K < 8` 写小 K 行内核时踩的：数值明显不对但**不崩溃**，
所以只靠"跑通了没"发现不了，必须逐个尺寸对比 intrinsic 的结果。
要用 N 方向向量化只有两条路：
① 把 B 打包成 N 方向的连续面板（packing，多一趟 N*K 的读写）；
② 用 `vgatherdps` 按 ldb 跨步收集。
K 方向向量化（现有 m2n4 / m1n8 / m1n1 微内核走的路）在任何 K 下都成立，
只是 K 很小时水平归约的固定开销占比过大。

改完 GEMM 内核**必须**用 `SamplesZQGEMM/SampleGEMMAsmCompare` 和
`SamplesZQBLAS/SampleGEMMCompare` 逐尺寸核对，它会把 |intrinsic - asm| 打出来；
`SampleGEMMCompare` 还会把 `asm/MKL` 比例打出来。

## NCHW 与 NCHWC 的步长语义（最容易搞反的一处）

`ZQ_CNN_Tensor4D`(NCHW) 里：

```
pixelStep  = C（可能按 ALIGN 对齐补到 4/8）   // 一个"像素"= 一个 C 向量
widthStep  = pixelStep * realW                // 一行
sliceStep  = widthStep  * realH               // 一"张图"（注意：不是"一个通道"!)
buffer     = sliceStep * N
```

所以元素 `(n, c, h, w)` 的偏移是 `n*sliceStep + c*1 + h*widthStep + w*pixelStep`——**c 维的步长是 1，不是 sliceStep**。`Reshape_NCHW` 里 `out_c_ptr++` / `i_c` 这么写是对的；我曾按"slice = 通道"的直觉改成 `*sliceStep`，结果 `SampleSSD` 在模型加载阶段直接段错误，已回退。

而 `ZQ_CNN_Tensor4D_NCHWC` 是另一套：那边 `imStep` 才是一张图，`sliceStep` 是通道步长。同样的变量名在两套布局里含义相反，改任一侧前务必先看 `ChangeSize` 里 `dst_tensor_raw_size` 是怎么算的。
