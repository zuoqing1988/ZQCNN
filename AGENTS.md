# AGENTS.md

## 变更日志规则（必须严格遵守）

写变更日志到 `docs-changelogs/CHANGELOG_YYYY-MM-DD.md` 时：

1. **先检查当天日期的日志文件是否已存在**（用 `ls docs-changelogs/` 或直接 Read）。
2. **已存在 → 只追加，不覆盖**：必须先 Read 读出现有内容，把新章节追加到文件末尾（Edit 在末尾锚点插入，或 Write 时用"现有内容 + 新章节"的完整内容）。严禁直接 Write 覆盖导致已有章节丢失。
3. **不存在 → 新建**：以 `# CHANGELOG YYYY-MM-DD` 开头。
4. **每次在任何带日期的产物里写日期前，先执行 `date '+%Y-%m-%d'` 确认当天日期**，再决定写入哪个文件——不得凭会话惯性沿用上一次写的日期。会话可能跨天或间隔数日继续。
   - **适用范围不止 changelog**：审计报告、`reports/` 下的实测报告、工具文件里的注释、commit message，只要写了日期都要用当天这个。
   - **不要凭感觉写日期。** 2026-10-01 就因为凭感觉，把 `2026-10-01` 误写成了 `2026-10-21`，混进了本文第 9 条和 `tools/sweep_samples_linux.sh` 的注释里。本条规则原本只约束 changelog，于是漏了过去 —— 现已补全适用范围。
   - 历史教训实例：2026-08-10 的实测记录曾误写入 CHANGELOG_2026-08-07.md，事后迁移订正。
5. 章节格式沿用既有风格：`## 新增/变更：标题` + `### 变更文件` / `### 实测结果` / `### 注意事项`。
6. 写完日志后发现他人（或前一个任务）又加了内容时，合并保留双方章节，不得整文件重写丢弃。

## 构建规则

1. **CMake 是唯一构建入口**：仓库已于 2026-10-01 移除全部 `.sln/.vcxproj*`（停留在 VS2015/v140、SDK 8.1，文件列表早已与源码脱节）。不要重新引入 MSVC 工程文件。
2. **双平台必须都编译通过**：改动 C/C++ 后要分别验证 Windows（`cmake --build build_x64 --config Release`）与 Linux（`wsl -d Ubuntu-20.04` + gcc 9，`/tmp/zqb` 下 cmake + make）。MSVC 能过不代表 gcc 能过。
3. **不要依赖 MSVC 的传递包含**：MSVC 会顺带引入 `<cfloat>` `<cmath>` 等，gcc 不会。新写的代码要显式 `#include` 用到的标准头（已踩坑：FLT_MAX 在 gcc 下报未声明）。
4. `ZQ_GEMM/CMakeLists.txt`、`ZQCNN/CMakeLists.txt`、`SamplesZQCNN/CMakeLists.txt` 都用 `file(GLOB ...)`，**新增 .c/.cpp 文件无需改 CMake，但需要重新 configure 才生效**。
5. **不要改无法在本仓库编译验证的第三方头**。`3rdparty/include/ZQlib/ZQ_JpegDecoder.h` /
   `ZQ_JpegEncoder.h` 既缺 `jpeglib.h` 也缺 `jpeg.lib`，全仓零 `#include` 引用点。
   2026-10-01 第三轮给它套 `setjmp/longjmp` 时把 `Decode_with_allocated` 里指向
   **调用方缓冲区**的 `pDst` 换成了内部变量 `dst_buf`（该函数语义就是不分配缓冲区，
   `dst_buf` 恒为 0），下一行 `memcpy(point, ...)` 首行即崩，longjmp 分支还会把
   调用方的指针清零 —— **一次加固反而造出了一个新 bug，而且编译不出来**。
   这类头要动，先补一个最小编译验证（哪怕 stub 一个 `jpeglib.h`）。
   > **2026-10-02 更新**：上面这条的前提（很多第三方头没法编译验证）**已经不成立**。
   > `python tools/probe_zqlib_headers.py` 会把 143 个 ZQlib 头逐个单独编译一遍 ——
   > **83 个能独立编译**，真正需要 Windows/MFC/OpenCV 的只有 6 个。
   > 写新的第三方头测试时先跑一遍这个探测器，别凭印象判断。
6. **全部检查有一个统一入口**：`python tools/run_audit_checks.py`
   - **全量回归用 `--all`**（2026-10-06 补，附录 IQ）：
     `python tools/run_audit_checks.py --all`
     它一次打开全部 9 个 opt-in 开关：`--with-build` / `--msvc-probe` /
     `--warn-sweep` / `--src-sweep` / `--reachability` / `--ubsan-sweep` /
     `--bounds-sweep` / `--msvc-asan` / `--with-slow`。
     **以前那条"全量回归"命令在仓库里根本不存在** —— 只在这里逐条列出开关，
     从没把它们组合过，于是每加一个开关就要记得往一条口口相传的命令里补，
     漏了没人知道（`--with-slow` 就这么烂掉过，附录 IK）。
     `C19` 门禁盯着这件事：新加一个 `store_true` 开关却忘了写进
     `OPT_IN_FLAGS`，门禁立刻红。
     **`--all` 不含 `--quick`（它减少覆盖）与 `--ubsan`（它换口径）** ——
     要 UBSan 那一轴请显式 `--all --ubsan`。
   - 默认：文本卫生（A1/A2/A3/A4）+ 第三方头库的 10 组 ASan 测试 + ZQlib 可编译性门禁（约 2.5 分钟）
   - `--quick`：跳过可编译性门禁。**注意这一条现在不准了**（2026-10-06 更正）：
     它只跳过 C（可编译性），**B 组仍然全跑**，而 B 组已从 58 涨到 **63** 道门禁
     （其中 `zq_padtype` / `zq_nchw_poolpad` / `zq_nchwc_poolpad` / `zq_nchwc_padreject`
     四道各要编整个内核族），实测 `--quick` 已经**跑不完 600 秒**（会超时，不是失败）。
     **只想跑 A 组源码门禁时直接逐个调**（`python tools/check_xxx.py`），别用 `--quick`。
   - `--ubsan`：B 组换成 `-fsanitize=undefined` 口径
   - `--msvc-asan`：加上 Windows 侧 MSVC ASan 那 10 组
   - `--msvc-probe`：加上 MSVC `cl /Zs` 逐头语法检查
   - `--warn-sweep`：加上 gcc `-Wall -Wextra` 的 HIGH 桶门禁（慢，约 2 分钟）
   - `--with-build`：再加上**双平台全量构建 + 关键 sample 回归**（Windows cmake
     与 WSL gcc 各一遍，两边各跑 6 个 sample；sample 必须在**产物目录**里跑，
     从仓库根跑只会打一行 `empty image`，看着像跑过了其实什么都没验）
   - `--with-slow`：把 `run_zqlib_checks.py` 的 7 个「编译慢」测试也带上。
     **不传它，默认通道里一条 GEMM 调度的用例都没有**（2026-10-06 补，附录 IK）：
     `zq_gemm_32f_AnoTrans_Btrans_auto` 是 NCHW/NCHWC conv + innerproduct 的
     公共入口，而覆盖它的 6 个测试全在 SLOW 里。改 GEMM 内核或者改卷积/
     innerproduct 走 GEMM 的那条路，**必须带这个开关验证**。
   任何一组失败就退出 1。**改完东西先跑它**，比逐个记命令可靠。

7. **第三方头库有独立回归入口**：`python tools/run_zqlib_checks.py`
   会自动发现 `tools/zq_*_check.cpp`，用 `gcc -O1 -g -fsanitize=address
   -I3rdparty/include/ZQlib` 逐个编译并运行，任何一个非 0 退出就整体失败。
   现在覆盖 **10** 组：`ZQ_BitonicSort` / `ZQ_ImageProcessing` / `ZQ_Kmeans` /
   `ZQ_MergeSort` / `ZQ_QuickSort` / `ZQ_Quaternion`+RBFKernel / `ZQ_Matrix`+Kahansum /
   `ZQ_MathBase`（SVD_Decompose + Cond_by_double_svd），
   以及 KDTree+WeightedMedian+CubicInterpolation+FindLargestSubMatrix、
   Matrix+ScanLinePolygonFill 两个组合。
   **动了 `3rdparty/include/ZQlib/` 下的头就要跑它**（主工程的 sample 回归验不到那里）。
   - `--ubsan` 换成 `-fsanitize=undefined` 再跑一遍（见「sanitizer 的两个坑」一节）。
   - Windows 侧对应物是 `python tools/run_audit_checks.py --msvc-asan`
     （`tools/run_zqlib_checks_msvc.bat`，MSVC + `/fsanitize=address`）。
     **只在一侧跑不算跑过**：`run_zqlib_checks.py` 把编译外包给了 WSL，
     所以它本质是 gcc/Linux 的结果，而本轮改过的头里有 5 个真的链接进了
     Windows 侧 sample（审计报告附录 AP/AS）。
8. **改完 ZQlib 头还要跑可编译性门禁**：
   `python tools/probe_zqlib_headers.py --check-baseline tools/zqlib_probe_baseline.txt`
   把每个头的独立编译结果与基线逐头比对，**任何一个头从 OK 变成非 OK 就退出 1**。
   修好或新增头之后用 `--save-baseline tools/zqlib_probe_baseline.txt` 更新基线。
   这一步抓到的是「自己源码就编不过」那一族缺陷（缺 `typename`、缺 include、
   未声明的变量、调用不存在的成员、命名空间少 `ZQ_` 前缀）—— 它们编不过，
   所以这些头**从来没被编译过**，也从来没被发现。2026-10-02 靠它把可验证的
   ZQlib 头从 81 个提到 118 个（57% → 82%）。它要编译 143 个翻译单元，
   比 `run_zqlib_checks.py` 慢（约 2 分钟），所以单独跑、不进前者。
7. **改完要抓基线再对照**：每次改动前先跑一遍关键 sample 并把输出存成基线
   （`tools/run_sample_regression.sh` + 手工滤掉耗时行），改完逐字节 diff。
   本项目已经有过 3 次"方向判断反了、改完功能反而坏掉"，其中
   `Align128bit::ResizeBilinearRect` 那次只有跑 sample 才能发现
   （见下面「三个张量变体」一节）。**只看编译通过是不够的。**
8. **没有 profiler 时不要按 uop 估算来换算法**。本机 WSL 里没有 `perf`，
   只能用 A/B 实测。实例：2026-10-01 给 ZQ_GEMM 的小 K 路径换算法时，
   我按"每 8 个结果 14 条 uop、理论 22 GF/s"推导出应该大幅提速，
   实测却是 1.3 GF/s（差 16 倍），换完算法还是 −7%。
   **uop 估算只能用来排除明显不划算的方案，不能用来论证一个方案会更快。**
9. **微基准的基线必须和生产版本同源**。2026-10-01 的教训：给小 K 族换内核时，
   微基准里我现写了一个朴素 intrinsics 版的 m2n4 当基线（1.2~2.0 GF/s），
   拿它和生产里的手写汇编 m2n4（13~24 GF/s）比，得出"新内核快 10~17 倍"，
   集成进真实驱动后反而比原版**更差**（`313x32x28` 28% -> 16%）。
   **基线要从生产代码里取，或者直接做 A/B**。同理，别拿自己顺手写的
   "显然更慢"的版本当基线 —— 那样得到的倍数是没有信息量的。
10. **报性能数字之前，先测噪声下限；A/B 必须交替跑**。本机 GEMM 测量的
   噪声下限约 7%（用同一个文件跟它自己比测出来的）。踩过的坑：
   ①「先把 A 跑完再跑 B」会让中间的睿频/温度漂移**系统性偏袒后跑的那个**，
      空对照里 64 个形状有 20 个出现 >3% 的假差异、最大的 10%；
      改成交替跑后降到 7 个、最多 7.3%。当时据此报出的「N 方向分块让
      2048³ 快 22%」**完全是噪声**（见 `audit_k3_20261001.md` 附录 R）。
      **同一个坑第二次咬人**：2026-10-01 测 Linux 加 `-mfma`，先跑完无 fma 的
      3 轮再跑有 fma 的 3 轮，结论是「加了 -mfma 反而慢 23%（中位 0.77×）」；
      改用 `tools/bench_two_binaries.py` 交替跑两个二进制后，结论**完全反过来**：
      中位 **1.11× 快**，37 个形状变快、3 个变慢、最大的 `16x16x16` 快 1.78×。
      方向都能反，不能靠"多跑几轮"解决，只能靠交替。
   ② 手滑把 `cp A B` 写成 `cp A B` 的变体，两个二进制其实是同一个，
      却报出「29 个形状变快」。
   **动手用 `tools/bench_gemm_ab.py`**（已内置交替跑、取最好值、
   8% 默认阈值），不要手写 shell 管道。
11. **微基准的 A 版必须是"完整"的**：K 循环、归约、写 C 一个都不能省。
   2026-10-01 又踩了一次：为了验证 6x8 外积内核，把生产 `m2n4` 的汇编抄进微基准，
   却只跑 1 个 K 块就计时（没有 K 循环 / 19 条归约 / 写 C），而候选版跑完整 K + 写 C。
   两侧口径不同，测出来的倍数没有意义。**要抄就连抄完整。**

## sanitizer 与检查工具的四个坑（2026-10-02 新增）

1. **UBSan 的 `rc` 恒为 0，不能拿它判通过。** UBSan 默认只打一行
   `runtime error:` 然后**继续执行**，所以进程照样返回 0。判据必须是
   `grep -c 'runtime error:'` 的行数。我最初按 `rc == 0` 写，那一栏永远是 0，
   等于什么都没查 —— 而且**看起来全绿**。（ASan 不同：越界/UAF 会直接 abort。）
2. **"一条都没报"的检查工具必须自带自测。** 一个匹配逻辑坏掉的扫描器返回
   "没有命中"，比没有这个工具更危险：它会让人以为这块已经审过了。
   `tools/check_alloc_delete.py --selftest` 有一份内建样本（3 条应命中 + 2 条
   必须不命中），并作为 `run_audit_checks.py` 的 A3 组常驻。
   **改这类工具的匹配逻辑之后必须先跑自测。**
3. **写"按行扫描"的工具时，正则一律用 `finditer` 不用 `search`。**
   `search` 只取一行的第一个匹配，于是「一行里两次 `malloc`」时第二个变量
   根本不登记，它后面的 `delete[]` 永远查不出来 —— 我自己写这个工具时
   在分配和释放两侧各犯了一次。
4. **不要把 Windows 路径丢给 WSL 的 bash。** `D:/ZQCNN/tools/x.h` 会被 bash
   当成一个**相对文件名**：命令照样返回 0，生成的文件里就是那串字面串，
   编译器于是报 "No such file or directory"。如果解析器只 `grep 'warning:'`
   （不看 `error:`），结果就是**整轮扫描一片全绿、实际一条没扫**。
   要转成 `/mnt/d/ZQCNN/...`（`tools/warn_sweep_zqlib.py` 里的 `to_wsl_path`）。
   同理，扫描器要**单独报告"有几个文件根本编不过"**，否则编不过的头在
   "按 -W 分类"的桶里显示为 0 条 —— 那是假绿。
5. **共享一份内容时，先确认消费方式也一致。** 2026-10-02 踩过：为了消除
   两份 MSVC 垫片，把它们改成「一份真实定义 + 一个转发头」，而两个探测工具
   都是把垫片**文本内联**进自己生成的翻译单元 —— 转发头里那句 `#include`
   于是变成真包含、路径在 /tmp 下找不到，**26 个头同时 OK -> BROKEN**，
   而且每条都带完整的 error 行，看着像"改坏了 26 个头"。
   消除重复的两种做法（合并成一份 / 加一层间接）在"内容被内联"这种用法下
   **只有前者是对的**。现在 `zqlib_msvc_shim.h` 是唯一真实定义，
   `warn_sweep_zqlib.py` 里还有一道前置断言禁止垫片含本地 `#include`。
6. **ASan 是逐翻译单元编译期生效的，复用 `.o` 会让 sanitizer 变成瞎子。**
   附录 CB 那次探测越界读时，一开始链接的是从别处复用来的 `.o`
   （没带 `-fsanitize=address` 编的），ASan **什么都没报**，看着像"没有越界"；
   把 `conv.c` / `zq_gemm_32f_align_c.c` 全部带 ASan 重编之后，
   立刻报出 `heap-buffer-overflow READ of size 32`，位置和代码推算分毫不差。
   > **ASan 报不出来的时候，先确认被测的那个 TU 本身插过桩**，
   > 别急着下"这里是干净的"这个结论。
   推论：任何"复用过现成 `.o`"的 sanitizer 探测都**不算数**，
   哪怕它跑完是绿的。顺带：`.c` 必须用 **gcc** 编，`.cpp` 用 **g++** ——
   拿 `g++` 编 `.c` 会先在 `void* -> float*` 上失败（`-fpermissive` 也只给警告）。

## 手写 extern "C" 声明时**照抄头文件、带参数名**，别数参数

C 链接**不检查 arity**。参数少写一个不会报错，只会让**实参整体错位** ——
`buffer` 收到本该给 `out_sliceStep` 的 int、`buffer_len` 收到本该给 `buffer` 的指针、
最后一位落空。2026-10-02 写 innerproduct_gemm 的测试时踩到（附录 BC.4）。

而且 g++ 报的 `invalid conversion from void** to int` 恰恰是在说
"**声明比调用点多了一个 int**" —— 第一反应却是**把声明改少**，方向反了。
把头文件的参数一个个抄过去（连参数名一起），问题自动消失。

顺带：g++ 不让 `#include` 本项目的 `.c`（`matrix_A = *buffer;` 是
`void*` -> `float*`，`-fpermissive` 也只给警告），
所以测内核的测试只能手写声明 —— 这正是上面这条规则会被用到的原因。

## 推不动的时候就把"排除了什么"记下来

`layers_c` 里 `zq_gemm_32f_align_c.c` 单个 TU 在 `-O1` 下编一次要 5 分钟以上，
ASan 更久。2026-10-02 给 `innerproduct_gemm` 写测试时（附录 BC），
每排除一个假设就要重编一轮，预算烧完仍然没定位，于是：

- **删掉那个测试**（一个"每次跑都崩"的测试进回归套件比不放更糟 ——
  至少 `zq_bns` 那种是断言失败、一眼看出是已知项）；
- 把**已排除的假设逐条写进报告**，下一轮从这里起步而不是从零。

**做不完的事如实记成"未完成"，比做成一个半成品更有价值。**

## 一个坏测试会产出**看起来很有说服力的假结论**

这是 2026-10-02 附录 AY.5 用两次错误换来的：

1. 只想确认"这个内核的步长对不对"，写了个对拍测试。
2. 第一次跑：全错，相对误差 1.00。**我据此断定"内核的索引约定是 NCHW 的、错了"**，
   并写了一版重排补丁。
3. 第二次跑：ASan 报越界，栈顶在**测试文件自己**。
4. 一共连修 5 处（缓冲区大小、图距、就地修改、参考下标……）之后，
   99 个用例**逐位精确**通过 —— 内核本来就是对的，补丁被撤回。

**危险的地方在于第 2 步**：一个恒定的"相对误差 1.00"看起来像"结果完全不对"，
而实际上只是我把结果比在了没被写过的 `out` 上（内核是**就地修改**）。
真去改一个没有问题的数值内核，代价比误判严重得多。

推出来的规则：

- **先证明测试本身是对的，再相信它的结论**。最省事的做法：
  写完测试先跑一个**已知答案**的小用例（比如恒等变换、或者一个手算的 2x2），
  它过了才有资格去判别人错。
- **ASan 报的栈顶在测试文件里时，先怀疑测试**。本轮 5 次里 5 次都是。
- **写"某段代码算错了"的补丁之前，先把被调方与调用方的约定两边都核实一遍**。
  这次是"内核的步长"对上了"张量的步长定义"才发现两边都对；
  中间我一度以为 `widthStep = realW * align_size` 是抄错了 NCHW 的公式，
  实际上 NCHWC 按 align 分片之后它**就是对的** ——
  **名字像 NCHW、含义是 NCHWC**。
- **"常量比较"（C6326）在本项目基本都是刻意的 SIMD 宽度分派**：
  `if (zq_mm_align_size >= 4)` 里 `zq_mm_align_size` 是每个变体的编译期常量。
  看到它报"比较两个常量"，去看**有没有 else**，有就是分派、不是缺陷。

## 写"检查类工具"的四条硬规矩（2026-10-02 补，四条都是踩出来的）

这一轮连续写了 5 个检查类工具（`warn_sweep_*` / `check_alloc_delete` /
`check_uninit_members` / `check_param_domain` / `check_div_guard`），
每一个都在**写出后立刻发现自己是错的**。四条教训：

1. **必须自带 `--selfcheck`，而且自测样本里要有一个"故意不合格"的项。**
   `check_param_domain` 的 `find_check` 写成了「格式化后的正则字符串非空吗」
   而不是「匹不匹配」，于是它报告**「全部 211 个参数都已有值域校验」**——
   而 BD/BE/BF 三条缺陷就在这一族里。没有自测的话，这个工具会**绿着**
   挡住后续所有同类缺陷。
   自测样本只放"全部合格"的例子是不够的：**必须放一个该被报出来的**。

2. **"上下文"不能越过函数边界。** 取"上方 8 行"会让函数 A 的守卫
   "罩住"紧随其后的函数 B。向上遇到第一个「行首是 `}` 的行」就截断。

3. **白名单 / 基线文件的路径必须归一化。** 扫描结果走 `os.path.relpath`
   （Windows 上是**反斜杠**），而基线是人手写的（习惯写**正斜杠**）。
   症状是"白名单 0 条"——**看起来像根本没有这个机制**。
   **白名单匹配不上是最危险的一类工具 bug**：它不报错，只让工具退化成
   "什么都不拦"。读入时统一 `replace(chr(92), '/')`。
   推论：**白名单的每一行必须带理由**，否则半年后没人知道为什么放行。

4. **不要自己手写 C++ 注释/语法解析器。** 补"块注释里的字符串不该被当代码"
   时写了个 `while ... break` 的状态机，结果把**真代码**也抹掉了 ——
   命中数从 96 变成 131，**数字对不上就是信号**。
   要解析 C++ 就**用编译器**（`gcc -E` / `cl /Zs`），别写正则状态机。
6. **新写的门禁，汇总行里不要出现字面的 `FAIL`**（2026-10-03 补，附录 FD.5）。
   `tools/run_zqlib_checks.py:866` 的 ASan 判据是 `grep -cE 'FAIL' <tag>.out`
   —— 它把输出里**含 `FAIL` 的行数**当作"断言失败数"。
   现有门禁都遵守「`FAIL` 只在真失败时出现」；我新写的 `zq_model_params`
   汇总行写了 `共 27 个模型：OK 27，FAIL 0，崩 0`，
   于是**一个都没失败也被判成"1 条断言失败"**。
   > 这份**不成文的输出格式约定**没有任何检查、也不在本文件里；
   > 写新门禁时按上面那句办即可。
7. **按名字 grep 出来的候选，必须去看它那一族的实现**，不能凭名字的含义套规则。
   附录 CC 写 `check_c3_guards` 时踩了两次：
   ① **NCHW（`layers_c/`）和 NCHWC（`layers_nchwc/`）两族都叫 `_C3`，含义完全不同** ——
      NCHWC 那一族的 im2col 把通道数 3 硬编码（传 C=4/C=6 出来的是只算前 3 通道的结果，
      **不报错**），必须守 `C == 3`；NCHW 那一族是 `memcpy(dst,src,sizeof(float)*filter_C)`
      的**通用实现**，`_C3` 指的是"小 C 变体"，守的是 `in_C <= 4` / `in_C <= 8`。
      不做族限定就把 3 处正常守卫报成了缺陷 —— **没去手工核实就写进报告了**
   ② `r'\bnchwc'` 匹配不上 `..._gemm_nchwc4_...`，因为**下划线是单词字符**，
      `_` 与 `n` 之间没有词边界。扫出 0 个命中时自测立刻全红
   > 自测样本里除了"必须报"的，**还要放"必须不报"的**（本次放了 3 条 NCHW 负样本）。
8. **「子进程写文件 + 父进程读文件」的门禁，必须显式判「没读到」= 失败**（2026-10-02 补）
   附录 CJ 里五道门禁都写成这样：

   ```c
   if (f) { if (fscanf(...) != N) ok = bad = 0; fclose(f); }
   if (WIFSIGNALED(st)) { g_crash++; …; return; }
   if (bad > 0) { g_bad++; …; } else { g_ok++; }        // <-- 洞在这里
   ```

   **ASan 撞上 SEGV 时默认走 `Die()` → `_exit(1)`，不发信号。**
   于是 `WIFSIGNALED` 为假、退出码也不是约定的值、`bad` 保持 0
   → **一个段错误被记成了"通过"**。
   后果实测：附录 CF 报告的"189 个用例全对"里，**70 个从未真正运行**。

   修法就是补一句显式判断：

   ```c
   int have = 0;
   FILE* f = fopen(RES_FILE, "r");
   if (f) { have = (fscanf(f, "…", …) == N); fclose(f); }
   if (!have) { g_crash++; printf("  没跑完（子进程没写结果文件，退出码 %d）\n", WEXITSTATUS(st)); return; }
   ```

   > 这与上面「崩溃类测试」第 2 条是同一件事的两半：
   > 那条讲"崩溃和算错在 `waitpid` 看来都是 exit code 1，分不开"；
   > **这一条讲"连 exit code 都对不上，因为 ASan 是 `_exit` 而不是发信号"**。
   >
   > **通则：判据里必须区分「读到了但内容不对」和「根本没读到」。**
   > 前者是失败，后者**更**是失败 —— 而默认写法会把后者算成通过。
9. **逐通道数组两个坑要一起防**：**长度**按 `ceil(C/align)*align` 开、
   **对齐**按 32 字节开。两个坑在**同一个数组**上，附录 CG 抓到长度那个、
   CJ 抓到对齐那个。`std::vector<float>` 只给 16 字节对齐，
   而 align=8 那一族用的是 `_mm256_load_ps`（**要求 32**）→ 直接 SIGSEGV。
   手法：多分配 8 个 float，把首地址推到 32 的倍数上
   `float* p = (float*)(((size_t)v.data() + 31) / 32 * 32);`
10. **参数名不是语义**。已踩三次：`batchnorm_b_a(data,…,b_data,a_data)` 的代码是
   `fmadd(x, b_vec, a_vec)`（**b 乘 a 加**）；`eltwise_sum_with_weight` 的 `weight`
   是**每张张量一个标量**、不是逐通道；`nodivided` 才是"按实际窗口收窄"的那个、
   `suredivided` 反而不管边界。**照抄参数名之前先去看 `fmadd`/`store` 那两行。**
11. **「该红的没红」的第一反应是「我的变异是等价的」，不是「库没执行那段」**（2026-10-02 补）
   附录 CK 里我把参考实现的 `exp(v - max)` 改成 `exp(v)`，**15 个用例一个都没变红** ——
   因为 softmax 平移不变：
   `exp(v_i-m)/Σexp(v_j-m) = exp(v_i)/Σexp(v_j)`。
   减 max 纯粹是为数值稳定，**在小输入下两种写法逐位相同**。

   > 门禁"测不到某处代码"**不等于**"那处没被覆盖"，也可能是那处在当前输入下
   > 与其他写法**数学等价**。要真正测到它，得喂**大到会溢出**的输入。
   >
   > 推论也适用于别处：查一个恒等式、查一个 clamp、查一个 max ——
   > 先用代数判断"这个变异在数学上会不会改变结果"，**别靠"跑一遍没变"就下结论**。

12. **看着可疑的 `++`，先确认"加完还有没有被用到"**。附录 CK.3 里 softmax 的对齐尾循环
   写成 `slice_ptr++`（看着漏了 `in_sliceStep`），但主循环退出时指针已停在正确位置、
   尾循环**先读后加**、加完就结束 —— 是对的。**确认，不能靠"看着像"。**
13. **同仓的两份实现互为对照，是判定"笔误还是设计取舍"最快的办法**（2026-10-02 补）
   附录 CL 里 `zq_cnn_resize_nchwc_raw.h` 的边界钳位是
   `x0 = __min(in_W - 1, …)` 但 `y0 = __min(in_H, …)` —— 同一个函数里只差一个 `- 1`。
   而 NCHW 那一族（`layers_c/zq_cnn_resize_32f_align_c_raw.h`）**5 处全是 `in_H - 1`**。
   **同一个功能的两份实现，一份对一份错，就不是设计取舍，是笔误** ——
   这一条比任何静态分析器的判断都硬。

   > 一般化：**找"可对照物"**。找 bug 时先问"同一个功能在别处还有没有第二份实现？
   > 移动平均、resize、边界钳位、`min`/`max` 的偏置 —— 同一份数学被抄两遍时，
   > 抄错的那份几乎一定能被另一份对照出来。

14. **无效的测试用例和无效的变异一样有害**（2026-10-02 补，与第 11 条同源）
   附录 CL.6：cfgB 换了**两版都测不到钳位** ——
   第一版坐标压根没出图；第二版 `w_step = 1` 导致 `sx` 恒为 0，
   而被验证的 `x1` 正是**被 `sx` 加权**的，所以钳不钳都一样。
   两次都是"把想验证的那处变异掉，看会不会红"才暴露的。

   > **让目标代码处在一个"它不影响结果"的位置上，用例就是废的。**
   > 设计用例时要问：**如果被测的那一行被删掉，这个用例会红吗？**
   
6. **扫到 0 个命中时先怀疑匹配逻辑，再下"这里是干净的"这个结论。**
   这条兜底在 `check_c3_guards` 上直接救了一次（坑 #2 的同一个机制）。
7. **新写的数值门禁，必须做一次「变异测试」才算数**（2026-10-02 补，本会话做了两次）
   做法：把**门禁自己**的参考实现改错（只改门禁、不碰被测代码），重跑，看它会不会叫。

   | 门禁 | 变异 | 修门禁前 | 修门禁后 |
   |---|---|---|---|
   | `zq_nchwc_depthwise_check` | 参考的 filter 下标写错 | 66/189 变红 | 66/189（本来就是这个数） |
   | `zq_nchwc_act_check` | 参考的 slope 偏 0.001 | **10/75** | **58/75** |

   第二次那一次是**靠"叫了多少"才发现门禁自己坏了**：
   我把 `bv`/`sl` 按对齐宽度 `A` 开，而用例里 `C = align+2 > A`，
   于是通道 `align`/`align+1` 的 slope 是 0；内核那边
   `load_ps(slope + c)` 在最后一个不满的组上**读过界**读到相邻内存，
   正好也是 0 —— **两边"恰好一致"，门禁全绿，但那 4 组用例根本没在测斜率**。

   > **"叫了多少"本身就是信号。** 只抓到一小部分时，正确的结论不是"库还行"，
   > 而是"我的工具还有几组是空的"。一个门禁要是**在 5 个变体里只抓到 1 个**，
   > 先怀疑那 4 个变体的形状是不是让它变得没有鉴别力。
   > 顺带：**过界的读取是门禁自己的 bug，不是被测代码的** ——
   > 被测代码读过界的前提是调用方给的数组不够长。

**通用判据**：工具报出来的数字如果**说不清为什么是这个数**，
那它多半已经坏了 —— 先去核对总数，再往下看明细。

## 「能编过」不等于「没毛病」

1. 前三十几轮找缺陷靠的是"编不过"这一根轴（`probe_zqlib_headers.py` 问的是
   能不能独立编译）。118 个头都能编过，意味着**编译器早就看见了问题、
   只是默认一声不吭**。要开 `-Wall -Wextra` 才说话：
   `python tools/warn_sweep_zqlib.py`（第三方）/ `warn_sweep_src.py`（主工程）。
   ZQlib 143 个头 4892 行警告，HIGH 桶挖出 5 条真缺陷；主工程 43 个 TU
   HIGH 桶 42 条，挖出 5 处未初始化成员隐患（附录 AT / AU）。
2. **基线只记 HIGH 桶**（`-Wparentheses/-Waddress/-Wnarrowing/-Wreorder/-Wformat=`）。
   MED/LOW 不进基线 —— 一个天天报 3000 条的门禁等于没有门禁。
3. **HIGH 桶也不是判官。** ZQlib 43 条 HIGH 里有 3 条是误报
   （`ZQ_BinaryImageProcessing.h` 的 `&&` 混在 `||` 里，优先级本来就对）；
   主工程 14 处 `-Wparentheses` 同样全是误报。
4. **"-Wmisleading-indentation 把人引到某段代码跟前"本身就有价值** ——
   附录 AT 里最严重的一条（`Cond_by_double_svd` 读错行距、条件数是未初始化
   堆内存）根本不是任何工具报出来的，是追查一条"看着像 bug"的警告时
   顺手把 API 契约读了一遍才发现的。**别因为一条警告被判为误报就跳过它。**
5. **扫主工程时，编译器和宏必须与真实构建一致**，否则会得到一堆假 error：
   `.c` 用 **gcc**、`.cpp` 用 **g++**（`zq_avx_mathfun.c` 的
   `_PS256_CONST_TYPE(sign_mask, int, 0x80000000)` 在 C++11 braced-init 下报
   narrowing，在 C 里完全合法）；加上 `-DZQ_CNN_USE_ZQ_GEMM=1 -mavx2 -mfma -fPIC`。
6. **gcc 不检查"类成员没在构造函数初始化列表里"**
   （`-Wmissing-field-initializers` 只管聚合初始化）。这一类用
   `python tools/check_uninit_members.py` 扫。主工程里查出 5 处真隐患
   （`ZQ_CNN_Layer::buffer` / `sample_type` / `ZQ_CNN_Net::input_C/H/W` …），
   都不是活 bug，但形状和"靠另一个开关撑着的未初始化指针"一样，值得补 `= 0`。

## 写内核 / 写并行代码的新增规则（2026-10-02 补）

1. **向量化循环的缓冲区必须给"最后一次整宽写"留够余量**。典型写法
   `for (c = 0; c < C; c += align) zq_mm_store_ps(p, v);` 一次写 `align` 个
   float，最后一下写到 `ceil(C/align)*align - 1`；缓冲区若只有 `C` 个，
   `C % align != 0` 就越界。`ZQCNN/layers_c/zq_cnn_lrn_32f_align_c_raw.h` 就是
   这样被 `local_size == 1` 触发的（附录 AX）。
2. **OpenMP 里累加到 parallel 区域**外面**声明的变量，必须写
   `reduction(+:...)`**。`ZQ_CNN_MTCNN.h` 的 P-net 漏了，于是两个诊断计数器
   无锁 `+=`，printf 出来的 pre-NMS 候选框数每次运行都不同（附录 AW.6）。
   注意这条的排查顺序：**先看"哪个数字在变、哪个不变"** ——
   不变的那多半在打印前被重新赋值过，变的那才是竞争对象。
3. **测内核的回归测试不能只 `-I3rdparty/include/ZQlib`**：还要
   `-I ZQCNN -I ZQ_GEMM`，而且**主 TU 也要带 `-mavx2 -mfma`**
   （include 了那个 .c，里面的 `_mm256_set1_ps` 是 `always_inline`，
   缺 `-mavx2` 会报 `target specific option mismatch`）；
   `.c` 辅助文件要用 **gcc** 编（g++ 会把 `zq_avx_mathfun.c` 的
   `_PS256_CONST_TYPE(sign_mask, int, 0x80000000)` 判成 narrowing 直接失败）。
   见 `tools/run_zqlib_checks.py` 里的 `EXTRA_*` 四张表。
4. **对齐类崩溃的 ASan 报告里，故障地址可能是 0x000000000000**。
   我的第一版 LRN 测试把 `zq_mm_align_size` 写成 4（实际被测内核是 8），
   于是像素地址 16 字节步进、`_mm256_load_ps` 要求 32 字节对齐 → SIGSEGV，
   报出来的却是 NULL。**看到 0 地址先查对齐，再查空指针。**
5. **cl 的输出是本地代码页（本机 GBK）**。按 utf-8 读会得到一堆 U+FFFD，
   而要匹配的 `warning C6386` 恰好是 ASCII —— 于是统计显示"0 条"，
   看起来像"全部干净"。`tools/run_msvc_analyze.py` 里固定按 gbk 读。
6. **批处理里不要用 `goto` + label 写"扫描全部"的逻辑**
   （`goto`/label 配 `EnableDelayedExpansion` 时 cmd 会开始把 `rem` 注释的
   **片段**当命令执行：2026-10-02 实测报了一串 `'ses' 不是内部或外部命令`）。
   枚举挪到 Python 侧，bat 只负责"给一串文件，逐个编"。

## 静态分析器是筛子，动态实测才是判据

1. **MSVC `/analyze` 的行号不能照单全收。** 附录 AX 那次：它报 `:68`（安全，
   只是零余量）和 `:73`（安全，卡在边界），真正越界的 `:64` 它**没报**；
   修完之后同样的 7 条 C6386 + 6 条 C6385 **依然在报**，而 ASan 证明那里干净。
   —— **判据是"ASan/MSan 跑不跑得出来"，不是"分析器报没报"。**
2. **但它非常适合当"让人去读某段代码"的指路器。** 附录 AX 的堆越界就是
   `/analyze` 把我引到那个文件才查出来的，只是它指错了具体哪一行。
3. gcc 与 MSVC 的检查**覆盖面不同、必须都走**（附录 AR/AT/AU/AW）：
   `-Waddress` 只在 gcc 有，`/analyze` 只在 MSVC 有，恒真条件两边都能报但
   措辞完全不同。

## 提交规则

1. 阶段性成果就 commit（构建修复、审计修复、文档、报告各自成次）。
2. **不要 push**，推送由用户自己决定。
3. 一个 commit 只做一件事，提交信息用中文写清楚改了什么、为什么。
4. **中文 commit message 用 `Write` 落盘再 `git commit -F <文件>`**，
   不要用 `git commit -m "$(cat <<'EOF' … EOF)"`。走 heredoc 时多字节字符会被
   弄坏（本机 2026-10-02 至少两次：提交信息里冒出 U+FFFD，源码批量替换也失配）。
   同 AGENTS.md 行尾一节第 6 条：凡是经过 Bash heredoc 的中文，先验一遍再往下走。
   提交前可以扫一眼：`git log -1 --format=%B | python -c "import sys; print(sys.stdin.buffer.read().count(b'\xef\xbf\xbd'))"`，应为 0。

## 示例程序规则

1. 所有 Sample 中的 `cv::namedWindow` / `cv::imshow` / `cv::waitKey` **一律注释掉**（无头环境与自动化验证会阻塞；用户 2026-10-01 明确要求）。
2. 跑示例前先确认没有等待按键的调用。


## 汇编/低层代码规则

1. **MSVC x64 不支持函数体内联汇编**：`__asm { }` 和 `__declspec(naked)` 在 x64 目标上都会编译失败（2026-10-01 实测 MSVC 14.35 报 C2143/C4235）。Windows 侧的手写汇编必须走独立 `.asm` 文件（CMake 里 `enable_language(ASM_MASM)` + `.asm`），GCC/Clang 侧才用 `__asm__ volatile` 内联汇编。写跨平台低层代码前先想清楚这两条路径。
2. `3rdparty/lib/libncnn.a` 是 clang 编译的，引用 `__exp_finite`/`__log_finite` 等 compiler-rt 符号，用 gcc 链接时由 `ZQCNN/math/zq_libm_compat.c` 补齐，不要删。
3. 基准测试程序里 MKL / OpenBLAS 一律**运行时动态加载**（`LoadLibrary`/`dlopen`），不引入链接期依赖；MKL 运行时放在 `3rdparty/mkl_runtime/`（已 gitignore），Linux 下 `libmkl_rt.so.2`、Windows 下 `mkl_rt.3.dll`。
4. **GCC 的内联汇编不接受 `"r8"(x)` 这种寄存器名约束**（gcc 9.4 实测）：只认
   `rax/rcx/rdx/rbx/rsp/rbp/rsi/rdi` 这 8 个，`"r8"`..`"r15"` 会被当成 `"r"` 加一个
   非法修饰符，报 `matching constraint references invalid operand number`。
   照抄生产代码的写法：**入参全用 `"m"`，进来后自己 `movq`/`movl` 到目标寄存器，
   寄存器名写进 clobber 列表**（见 `zq_gemm_32f_align_c_asm.c` 的 `ZQA_MOV64`/`ZQA_MOV32`）。
5. `vextractf128` 的目标必须写 `%%xmm8`，不能写 `%%ymm8`（后者报
   `operand size mismatch`，紧跟的 `vaddps` 也会报 `register type mismatch`）。
   AT&T 模板里每个寄存器名都要写两个 `%`（`%%ymm0`），写一个会被 GCC 当成操作数引用。
6. **Zen 3（Ryzen 9 5900HX 实测）上 `vbroadcastss ymm, xmm`（寄存器源）比
   `vbroadcastss ymm, m32`（内存源）慢约三个数量级**，而指令数只多 6 条 `movss`。
   写"先把标量读进 xmm 再广播"的优化前先在这台机器上 A/B 一下：
   同一个 6x8 外积内核，内存源 91.2 ns/次（67.4 GF/s），寄存器源慢到跑不完 2000 万次。
   内存源本身就是 1 条 load-port uop，并没有多占端口。
7. **MASM 微内核里绝对不能出现 `rsi` / `rdi` / `rbx` / `rbp` / `r12`-`r15`，
   除非显式 push/pop**。这些在 x64 Windows 是 callee-saved，在 System V (Linux)
   却是 caller-saved —— 用错的后果是 **Linux 全对、Windows 段错误**，
   编译和静态检查都发现不了。2026-10-01 写 `m6n8` 时用了 rsi/rdi，
   Linux 侧 `SampleGEMMAsmCompare` 全过、Windows 侧跑到 `13x11x7` 之后 exit 139。
   可用的只有 `rax rcx rdx r8 r9 r10 r11` 这 7 个；塞不下时优先**把偏移挪到循环
   之后从栈参数重算**（`m6n8` 就是这么做的：循环里只留 ap/bp/K/c 四个）。
8. **打包成"微内核一次吃掉一块"的面板，不要打成"整面板"再按块偏移取**。
   写成 `[K][ncn]` 却按 `Bp + c*8*K` 取块时，只有 `ncn == 8` 才碰巧对 ——
   `16x8x32` 通过而 `32x32x32` 误差 9.07。**这类错误只靠多尺寸对拍能发现**，
   任何新增/改动打包路径都必须过 `SampleGEMMAsmCompare` 的全部尺寸。
9. **汇编里「借寄存器当暂存」是跨内核的耦合**。`ZQA_FMA` 在无 FMA 时借 `ymm15`
   做 `vmulps` 的暂存，而 6x8 内核把 B 向量放在 `ymm15` —— 两者写在同一个文件里
   却互不知情，`-mfma` 一缺席就静默算错（审计报告附录 V）。给某个内核用的
   暂存寄存器必须是**该内核自己的空闲寄存器**（6x8 改用 `ymm7`）。
   更进一步：**编译期分支都要有人走过**。默认构建永远只走「有 FMA」那条，
   另一条（`ZQ_GEMM_ISA=off` 之外还可以靠去掉 `-mfma` 触发）要专门编一次去验。

## 行尾与跨平台编译规则

1. **内核头文件（`*_raw.h`）在仓库里必须是纯 LF**。这些文件里全是跨行宏（行尾 `\` 续行），一旦被提交成 `\`+CR+CR+LF，MSVC 能容忍而 **gcc 的行拼接会失效**——宏在第一行就被截断，表现为"文件作用域出现未声明标识符"之类的编译错误；更糟的是增量构建会**沿用旧的目标文件**，让 Linux 侧跑出一个和源码完全对不上的旧二进制。`.gitattributes` 已经给这些文件标了 `eol=lf`。
2. 改完内核头文件后，**Linux 侧要确认目标文件真的被重新编译**（看 `make` 输出里有没有 `Building C object ...raw...`），不要只看 `make` 的返回码。
3. 本机 `core.autocrlf=true`，提交时 git 会把工作区的 CRLF 归一成 LF 存进仓库——所以**"git diff 干净"不代表工作区行尾干净**，而工作区才是编译器真正读的东西。
4. **改完代码跑一次 `python tools/check_line_endings.py`**。它查三类问题：multi-CR（`\r\r\n`，会让 `\r` 并进 `#include`/`#ifndef` 的预处理符 token，是 UB）、lone-CR、以及**同一文件里 CRLF 与裸 LF 混用**。`--fix` 可以自动规范化。纯 LF 文件（`*_raw.h`）不会被误判。
5. **`tools/run_sample_regression.sh` 只看退出码，不看输出**。改完如果动到了浮点
   语义（换编译选项、换指令、换累加顺序），要**编两次、逐字节 diff sample 输出**。
   2026-10-01 补 `-mfma` 时就是这么查出一个静默算错的 bug（见 `audit_k3_20261001.md`
   附录 V）：两次跑回归都是全 rc=0，但 K≤32 的四个用例误差 6~7。
   另注：sample 必须在**产物目录**（`cmake-out-unix-x64/Release` 或
   `cmake-out-win32-x64/release/Release`）里跑，从仓库根跑只会打一行
   `empty image`，看着像跑过了其实什么都没验。
6. **用 Python 批量改写源码时必须自己保证行尾**：`open(p,'rb').read().split(b'\n')` 再 `b'\n'.join(...)` 这种写法，**新插入的行不带 `\r`**，工作区立刻变成 CRLF/LF 混用。正确做法是插入时补 `b'\r'`，或改完立刻跑 `--fix`。2026-10-01 批量修 MTCNN 时就是这么踩到的。
   同一个坑的另一个变体：**Bash heredoc 里写 `\\n` / `"""` / 中文会被吃掉**，
   所以改源码和写 python 脚本一律用 Write/Edit 工具落盘，不要 `python - <<'PY'`。
7. **不要凭"以前的修复报告写了什么"来判断覆盖面**。第四/五轮声称边框清零已覆盖 `ROI`，实际只改了 `Resize*`；`NCHWC1/4/8` 整个系列一处没改。收口一类缺陷时要**全仓枚举同类站点**（`grep` 出所有出现位置逐个核对），而不是只信上一轮的清单。
8. **改中文注释/文档时不要走有损解码，改完必须跑 `python tools/check_text_encoding.py`**。`core.autocrlf=true` 下用脚本批量改写时，只要有一环用了 `errors='replace'` 再写回，原字节就被永久换成 `EF BF BD`，而且**不报错** —— 只有读那一行时才看到几个黑方块。2026-10-01 在 `reports/ZQ_GEMM_多内核自动选路_设计提案.md`、`docs-changelogs/CHANGELOG_2026-10-01.md` 和 `zq_gemm_32f_align_c_asm_msvc.asm` 各抓到一处（提交前就在仓库里）。该工具还会报严格 UTF-8 解不开的文件，其中 4 个是上游带来的 **GBK 文件**（`ZQ_MFC_Utils.h`、`ZQ_PutTextCN.h`、`ZQ_CNN_FaceCropUtils.h`、`mxnet2caffe.bat`），已在白名单里，不要去"修"它们。
9. **Edit 工具会抹掉文件头的 UTF-8 BOM。** 用 Edit 改一个带 BOM 的文件时，
   第一行的 `EF BB BF` 会被悄悄去掉 —— `git diff` 里表现为第一行被改了一行内容
   （`-﻿#include ...` / `+#include ...`），**混在真正的改动里很容易看漏**
   （2026-10-02 改 `zq_cnn_convolution_gemm_32f_align_c.c` 时就是这样，
   两处目标改动之外还多了一条第一行的改动）。改完先 `git diff` 逐条核对，
   发现自己没打算改第一行就是它；还原用
   `python -c "import io;p=r'...';b=io.open(p,'rb').read();io.open(p,'wb').write(b'\xef\xbb\xbf'+b)"`。

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

### 2026-10-02 补：`[N,1,1,K]` 这种 out 张量上，`sliceStep` 与 `imStep` 差 K 倍

innerproduct 的输出是 `[N,1,1,K]`，于是 NCHWC1 张量上

```
widthStep = 1*align      sliceStep = widthStep*realH = align      imStep = ceil(K/align)*sliceStep
```

**跳到下一张图必须用 `imStep`。** 附录 BN.2 里 12 个 `noborders` 调用点全部传了
`sliceStep`，于是 `N>1` 时相邻两张图的 K 个结果**互相覆盖**、缓冲区尾部根本没被写。

**这个坑最阴的地方：`out_N == 1` 时 `n*sliceStep + k` 恰好就是唯一那张图，
所以所有 sample 回归全绿。** 只有 batch > 1 才暴露。写这类内核时，
自检里一定要有 `N >= 2` 的用例 —— 光有 `N == 1` 等于没测。

参数名要跟着改（`out_sliceStep` -> `out_imStep`）：名字写着 sliceStep 而实际要
imStep，本身就是这个缺陷的成因。注意 `general` 那族的同名参数**确实是** slice
步长，不能一起改。

## 崩溃类测试的三条硬规矩（2026-10-02 补，全是写 `zq_gemm_shape_check` 时踩出来的）

1. **一个用例一个子进程。** ASan 碰到 SEGV 直接 abort 掉整个进程，
   一崩就停的测试只能告诉你"有一个坏了"，没法告诉你"哪些是好的" ——
   而后者恰恰是判断缺陷边界的关键。第一版 304 个用例只跑到第 12 个。
   `fork()` 一次 + 父进程 `waitpid()` 判 `WIFSIGNALED`，崩溃只算该用例失败。
2. **ASan 默认是 `exit(1)` 不是 abort。** 所以"崩溃"和"算错"在 `waitpid` 看来
   都是 exit code 1，分不开（第一版把两种混在一起报了"134 个 FAIL"）。
   要靠这个函数才能分开 ——
   ```cpp
   extern "C" const char* __asan_default_options() { return "abort_on_error=1"; }
   ```
   它在 ASan 初始化**之前**被调用，所以设在这里才有效。
3. **子进程的 stderr 要接 `/dev/null`。** ASan 的报告走 stderr，会把父进程
   stdout 上那一行结果拦腰截断（第一版的网格就是这样变成"每行只有一个 X"的）。

配套：崩溃类测试**每个用例打一行到 stdout**（`setvbuf(_IONBF)`），
这样崩溃前最后一行就是"挂在哪一组"。见附录 BL.8 同一个坑（那时是块缓冲把
全部 ok 行都吞了）。

## 照抄生产代码的分支条件，要连**前面的系数**一起抄（2026-10-02 补）

附录 BN.4：`noborders` 快速路径的生产条件在三种对齐下是**不一样**的 ——
```
align=1:  in_W == in_widthStep  &&  in_W  == filter_widthStep  &&  out_W  == out_widthStep
align=4: 4*in_W == in_widthStep && 4*in_W == filter_widthStep && 4*out_W == out_widthStep
```
我第一版照抄时把 filter 那一项写成了 `W == filter_widthStep`（漏了 `align*`），
于是 align=4/8 的条件检查假红 —— 差点被我当成"又一条真缺陷"报上去。
**照抄条件就是照抄条件，别顺手"化简"掉前面的系数。**

## GEMM 的判据必须用**后向误差**，不能用相对误差（2026-10-02 补，踩了第三次）

写 GEMM 形状测试时我用 `|got-exp| / max(|exp|, 1)`、阈值 2e-4，在 K=1024 上报了一堆
"结果错"：

```
M=1024 N=64 K=1024 max_rel=5.5e-02 @ (m=441,n=11) got=0.000046 exp=0.000043
```

看着像"大形状下静默算错"。**是判据错了。** 1024 个量级 ~0.29 的乘积求和，
中间值在 ±9；某个 (m,n) 上正好抵消到 4.6e-5。float32 只有 7.2 位有效数字，
抵消 5 个数量级之后**绝对**误差自然还有 1e-6 量级 —— 除以 4.6e-5 就是 12%。
**这是 float32 的固有性质，不是内核的错。**

正确判据（GEMM 的标准做法）：

```
err(m,n) = |got - exp| / ( ||A_m||_2 * ||B_n||_2 )
```

分母是"这一格的计算尺度"，与结果被抵消到多小无关。换成它之后，
K=16 那一整张 180 个格子全过，而真正坏掉的形状（K 不是 8 的倍数那一片）
**仍然是崩溃** —— 崩溃与判据无关，所以那条发现不受影响。

连着四次了（附录 BC / BI / BN.4 / BO）：**先确认判据本身是对的，再去解释结果。**
一个"看起来很有说服力"的假结论比没有测试更贵，因为它会被写进报告。

## 写形状/覆盖类门禁时，先看一眼**它自己的形状表落在哪一象限**（2026-10-02 补）

`tools/zq_gemm_oob_check.c` 测 10 个形状，跑了很久、全绿。但：

```
{2,4,64} {2,8,64} {4,4,64} {5,4,32} {8,4,16}
{2,12,64} {3,4,8} {16,4,256} {2,100,128} {7,4,40}
```

**这 10 个的 K 全部是 8 的倍数、N 全部是 4 的倍数** —— 也就是说它的形状表
**恰好只覆盖了安全的那一象限**，结构上不可能发现附录 BO 那条缺陷。
同一个毛病在 `zq_innerproduct_check` 上也有过（BN.3）。

所以：**新写一个门禁时，先确认它的形状表是"代码已经能处理的那几种"，
还是"故意挑了边界上的那几种"。** 后者才有意义。前者只是"跑了很多次全绿"。

## ZQ_GEMM 的两份实现对「对齐」的假设不一样（2026-10-02 补）

* **汇编版** `zq_gemm_32f_align_c_asm.c`：`vmovups` 9 次、`vmovaps` **0 次**。
* **intrinsic 版** `zq_gemm_32f_align_c_raw.h`：`zq_mm_load_ps`（= `_mm_load_ps`
  = 对齐的 `movaps`）**250 次**、`zq_mm_loadu_ps` **0 次** ——
  那个非对齐的宏就在旁边定义着，一个都没用。

后果：intrinsic 版的 k 方向循环**隐含假设 `lda`/`ldb`（=K）是对齐宽度的倍数**。
不满足时 `movaps` #GP -> SIGSEGV（附录 BO.2）。而且 k 循环的上界写的是
`padK = ceil(K/align)*align` 而不是 `K`，紧跟的标量收尾
`for (; k < K; k++)` 在 `K < padK` 时一次都不执行。

**改 ZQ_GEMM 的 K 维循环之前先确认这两点**，否则会以为"只是精度差一点"。

## 批量改代码之后，**重新看一遍被改那几行的控制流**（2026-10-02 补，第五次假绿）

用正则批量给 5 个提前 `return true` 补一句清理，它把语句插到了 `if` 和它的
`return` **之间**：

```c
if (bw != 0 || bh != 0)
    if (buffer) _aligned_free(buffer);
    return true;      // <-- 现在是无条件的
```

结果那一整段测试每个用例都直接返回 —— 全部"跳过"、全部"通过"，
`168/168` 看着完美，实际一条都没跑。是 LeakSanitizer 报的残留泄漏引到那里的。

**编译通过不等于控制流没变；"测试变绿"更不等于。**
`grep -c "_aligned_free(buffer)"` 数出来是 6，看着对 —— **计数不是证据**。

这是本仓库第五次"很有说服力的假绿"（附录 BC / BI / BN.4 / BO.3 / BP.5）。
共同点都是：坏掉的不是被测对象，是**测它的东西**，而它报出来的东西看着比真相更像真相。

## ZQ_GEMM 的调用方契约是**三条**（2026-10-02 补，第 2 条之前谁都没写）

1. `lda`(=K) 要是向量宽度的整数倍 —— 否则 k 方向那族内核 `movaps` #GP
   （附录 BO.2；附录 BP.1 给 dispatcher 入口补了守卫兜住）
2. **缓冲区要 32 字节对齐** —— 256bit 那一族用的是**对齐**载入（`_mm256_load_ps`），
   glibc `malloc` 只给 16
3. `C` 是**覆盖**写（`beta=0`），不是累加

第 2 条的坑特别隐蔽：**ASan 的分配器给 32 字节对齐**，所以所有在 ASan 下写的
测试**永远看不到这一条**，一去掉 ASan（写计时 harness、性能对比）就第一次调用就崩。
写任何"在 ASan 下建、拿去做非 ASan 计时"的临时程序时，缓冲区一律用
`_aligned_malloc(..., 32)`。生产里这些缓冲区本来就是 `_aligned_malloc` 出来的。

## 通用工具的输出里**不要写死某一次的具体描述**（2026-10-02 补）

`tools/ab_time_two_binaries.sh` 的输出曾经是：

```
A (1 || 全走 GEMM) : n=9  median=...
B (守卫恢复)       : n=9  median=...
```

那是某一轮 A/B 的遗留描述，而脚本是**通用工具、被复用过多轮**。
2026-10-02 跑附录 BQ 那一轮时，A 其实是 after、B 是 before ——
照着标签读会把方向整个读反，而数字本身看不出错。

改成直接打印两个二进制的路径。通用工具（计时器、比对器、报告器）的输出
应该**自带它实际作用于谁**，复用的人不会去看脚本、只会看输出。
同一类里还有一条：`grep -c` 数出来"看着对"不算证据（附录 BP.5）。

## 写驱动/计时程序时：无参数调用是一种调用方式（2026-10-02 补）

`if (argc < 2) { usage; return 2; }` —— **无参数时 `argc == 1`**，
所以如果调用方是"不带参数调用"（`ab_time_two_binaries.sh` 就是），
这个守卫会把每一次调用都变成一行 usage，然后外层报"没取到样本 A=0 B=0"。
症状离病因很远：看到的是"没采到数据"，原因是"参数守卫"。
驱动里的形状/tag 参数**一律做成可选**（取编译期默认值）。

## 回归脚本**只看退出码**会把「平台桩」记成通过（2026-10-02 补）

本仓库有一批 sample 是平台桩：打印一句
`./SampleFaceDetectorMTCNN only support windows` / `not support in linux`
然后 `return 0`。`tools/run_sample_regression.sh` 原来写成

    "./$e" >/dev/null 2>&1
    rc=$?

于是历次「Linux sample 8/8 全绿」里，有 **2 个在 Linux 上什么都没做**
（真跑的是 6 个）。**桩本身不是缺陷**，缺陷在回归把它算成了通过 ——
它让「双平台都跑通了」这句话比证据支持的更强。

改法（已落地）：输出留下来，分 `OK` / `STUB` / `NOOUT` / `FAIL`，
`STUB` **报出来但不判失败**，`NOOUT` / `FAIL` 判失败（退出非 0）。
最后一行打「真跑了几个 / 桩几个 / 问题几个」。

**通用教训：退出码为 0 只证明"没崩"，不证明"做了事"。**
凡是"跑一下就算过"的回归，都要问一句：**它输出什么？我看吗？**

## 一个「最差格」不能用来概括整体（2026-10-02 补，第六次假结论）

附录 BZ 报「NCHWC8 的 with_bias 全错、bias 根本没被加上」，证据是测试挑出来的
**最差格**上 `(got-exp)/bias = -1.0000` 精确成立。两个与门禁**没有一行共用代码**的
独立复现给出：**8 个输出通道 x 400 个格子全部与参考值一致**。

这与附录 BO.3 是**同一个错法**：从一大堆数字里挑**一个**（"最差格"、"最大相对误差"）
当结论的证据。区别在于这次**判据是对的**（两个程序用同一个后向误差、同一阈值），
假结论却还在 —— 所以：

> **判据对了不等于测量对了。**
> 要下"某条路径整体如何"的结论，必须**逐格统计**
> （多少格对 / 多少格错 / 错法有几种），不能只报最差的那一格。
> 如果一个结论只在极值格上成立，先假设整体是好的，再去找反证。

## 改测试本身之后，拿「独立复现」对一遍（2026-10-02 补）

附录 CA 里还查出：把 align=8 加进 `zq_nchwc_conv_check` 的那次重写
**同时把 align=4 的带 bias 路径弄坏了** —— 门禁从 6 个失败变成 26 个，
而我当时只看到"新增的 align=8 报了新东西"，没意识到老的那一段也坏了。
那次重写引入了两层新间接：`##ALIGN##` 宏拼接（只能在调用处拼字面量，
模板参数是标识符拼不出来），以及把明细挪到**子进程**里打印
（子进程写 stdout 会把父进程的网格行拦腰截断）。

> **改测试代码 = 改被测对象的行为。**
> 尤其是**新增了宏 / 模板 / 间接层**的时候，改完必须拿一个**与它没有共用代码**的
> 独立复现对一遍，并且**对比改动前后的失败数**（本例 6 -> 26 就是靠这个发现的）。

## 「改了没变化」不等于假设被推翻（2026-10-02 补）

附录 BX 试了五条假设，每条都是"改完跑一遍，看结果变没变"。第 5 条
（gemm 的 K 与 B 的行距不一致）两个方向都试过，**结果一点没变**，
于是被记成"排除"。**它其实是真的** —— `kernel2x2_C3` 同时坏在三处，
另外两处独立地让结果全错，所以改这一处当然看不出差别。

> **一个"改 A 现象不变"的实验，只能说明"A 不是唯一原因"，不能说明"A 不是原因"。**
> 要区分，得先把另外的已知缺陷修掉再重试，或者让 A 的效果**不被 B 掩盖**
> （比如只测 B 失效的那部分输入、或者直接断言中间量而不是最终结果）。

推论：**连续 N 次"改了没用"本身就是信号** —— 优先假设"不止一处"，
而不是"下一处也一样没用"。BX 连着五次落空正是这条规则的缺失造成的。

## 「生产不可达所以不改」这条判据要打补丁（2026-10-02 补）

附录 BX.4 用"生产不可达"（shipped 的两个 2×2 卷积 in_C 是 16/64，都不走 `_C3`）
当理由不修 `kernel2x2_C3`。附录 CB 把三处缺陷全修掉之后回头看，
**那个理由不成立** —— 正确的判据是另外两条：

1. **附近有没有一份可逐行对照的、正确的同类实现？**
   有（`kernel3x3_C3` 就是，同一个文件、同样的展开形状、还带了正确的
   `align>=4 / #else` 分支），"重写整支"的风险就远小于看上去。
2. **缺陷是不是内存安全问题？**
   越界读跟可达性无关。`matrix_A_cols` 用错公式导致按 `ldb=32` 读一块只有
   `16*K` 的缓冲区 —— **不可达的代码也不该留着越界读**。

> 判据从"生产跑不跑得到"改成 **"有没有可对照的正确实现" + "是不是内存安全问题"**。
> 只看可达性，会把带越界的死代码一直留着，而且下次有人扩展调用方就会炸。

## 契约之外的调用会**段错误**而不是返回垃圾值（2026-10-02 补）

`zq_cnn_conv_no_padding_gemm_nchwc{1,4,8}_*` 六支都要求
`filter_N % align == 0`。违反时 `plain`（无 bias）变体**直接 SIGSEGV**，
`with_bias` / `with_bias_prelu` 只是算错：

```
general plain, C=8, K=8 -> 正常返回
general plain, C=8, K=6 -> Segmentation fault (exit 139)
```

库对这些参数**不做任何校验**。所以：**写门禁时，"故意违约"那一档必须标成
「只报告、不判失败」** —— 而且判定代码要真的读那个标记。
附录 CB.4 里 `zq_nchwc_conv8_check` 自己标了"只报告"却在判定处照常 `g_crash++`，
凭空多出 12 个"崩溃"；`zq_nchwc_conv_check` 则先 `ncrash++` 再改成
`info:WRONG`，同一个用例进了两个计数器。**两处都是门禁自己的 bug，
不是被测代码的。**


## 批量改代码 / 判失败原因的四条硬规矩（2026-10-03 补，全是当天踩出来的）

0.5 **门禁里凡是有"负索引"或"两套坐标系"的地方，坐标换算必须收敛成一个函数**
   （2026-10-03 补，当天在 `zq_tensorop` 上一连踩三次、同因不同症）。
   典型场景：对照快照要覆盖 border 那一圈，而 `firstPixelData` 指向**数据区起点**，
   border 在它**之前**（负偏移）。当天三次错法：
   ① `(size_t)h * widthStep` 让负 `h` 回绕成 ~1.8e19 -> 全例 ASan 越界；
   ② 快照按 `firstPixelData` 起算、只申请 `N*sliceStep` -> border 落在向量外；
   ③ 换坐标系时符号搞反 -> 全例崩。
   **①和③的症状完全不同（"全崩" vs "全数据错"），根因却是同一个** ——
   这正说明**同一个根因会伪装成不同症状**，不能靠症状认根因。
   做法：写成 `off()` / `base_off()` / `bidx()` 三个小函数，
   每个注释里写清"这个坐标系的原点在哪、为什么是这个符号"。
   散在循环里内联算，第二次一定还会错。

0. **改一个「由模型文件驱动」的解析点之前，先数影响面，再动手**（2026-10-03 补，
   当天为这条差点把一个随仓库模型改坏）。当时看到
   `ZQ_CNN_Layer_Input` 的构造函数是 `H(0), W(0), C(3)`、而 `ReadParam`
   的返回值只看 `has_C && has_name`（H/W 完全不要求），就推断
   "模型写 `Input C=3 name=n` 会造出零尺寸张量"，于是**先改了再说**。
   按本文件「修改生产代码前先抓基线」那条去数了一下：

       省略 H 或 W 的 Input 行：1 条，涉及 1 个模型
          model\det1.zqparams        Input name=data  C=3

   **这会改坏一个随仓库模型。** 再往下查才发现更根本的：
   `MTCNN::SetPara` 只设包装器自己的 width/height、**不写 net 的 Input 层**；
   真正的图像是通过 `ConvertFromBGR` 写进 blob 0 的、那会重新 `ChangeSize`，
   **零尺寸状态是瞬态的，中间没人解引用它**。于是**撤回**。

   > **看到"未校验的参数"不等于"缺陷"。** 判断依据必须是三样：
   > ① 下游有没有兜住 ② **调用方有没有依赖这个默认值**
   > ③ 有没有一份"同仓的第二处实现"给出的相反约定。
   > 这次 `has_H_val` 在 `ZQ_CNN_Net.h` 里只被用来
   > "**有 InnerProduct 层时才强制**" —— 那就是 ② 的直接反证，
   > 而我是在**改完之后**才去查的。**顺序必须反过来：先查影响面，再改。**
   >
   > 具体做法：改一个解析点之前，先在**现有模型**里统计该参数的实际取值分布
   > （`grep` 一下 `.zqparams` 里那一行的形态就够）。
   > 分布里出现"省略/为 0"且跑得好好的，那就是**有意支持的配置**，
   > 加校验等于改行为。改完再撤回的成本远高于改前查一眼。

1. **脚本批量改源码，必须先 dry-run 逐条打印"改前/改后"，人工核对后再 `--apply`。**
   2026-10-03 用脚本把 `ChangeSize(..., dst_borderH, dst_borderW)` 批量换成 W-first
   （51 处），**两次都在 dry-run 阶段抓到问题，一次都没进版本库**：

   - 第一版**把函数声明也匹配上了**，dry-run 里赫然出现三条
     `ChangeSize(int N, int H, int W, int C, int borderW, int borderH)` 被"交换" ——
     真应用下去就是**把签名改了**。修法：实参里出现类型关键字就判定为声明/定义并跳过。
   - 第二版**判断条件写反了**（把"次位 W、末位 H"这个**正确**形态当成要改的），
     三条本来正确的 `ConvertFromCompactNCHW` 调用差点被改。

   > **改完还要机器核对一次**：把 `git diff` 的增删行两两配对，
   > 断言"0 行不是纯 `borderW`/`borderH` 互换"。这一步比肉眼看 diff 可靠。
   > 同时确认 BOM 与行尾没被动（`git diff --numstat` 的增删行数应当相等）。

2. **同一段字面量出现多次时，`assert count == 1` 会在你确认之前就抛错。**
   `.cpp` 里 10 个 `Resize*` 变体的 `ChangeSize` 那一行**字面完全相同**。
   变异测试想只回退其中一个，`assert d.count(old) == 1` 直接 `AssertionError`，
   脚本提前退出 —— **而"没报错的那次变异"和"没施加的变异"长得一模一样**。
   正确做法：打印 `count`，改**第一处**并把行号一起打出来人工确认是哪一处。

3. **没有证据就不要在失败信息里断言原因。**
   门禁把"结果文件读不出来"一律标成 **"sanitizer 报错"**，而那次子进程 stderr
   文件是 **0 字节**、退出码 0 —— **一条 sanitizer 报告都没有**。
   真正的原因是门禁自己的 `fscanf("%95[^\n]")` 遇到空串返回 3 而不是 4，
   于是每个用例都被判成"没跑完"。**没有证据的断言会把排查方向直接带偏。**
   改法：note 为空时写 `"-"` 之类的哨兵；失败文案只说观察到的事实
   （"结果文件读不出来"），不说推测的原因。

   3b. **测量出来的结果和读出来的代码冲突时，先怀疑测量**（2026-10-03 补，附录 GG.3）。
   上面那条讲的是"没有证据就别下结论"；这一条是它的下一层：
   **证据本身可能是错的，而我拿它当证据用了。**

   当时我测一批 sample 的退出码，脚本是
   ```sh
   out=$(timeout 90 ./$e 2>&1 | head -2 | tr '\n' ' '); rc=$?
   ```
   `$?` 取的是**管道最后一个命令（`tr`）**的退出码，**恒为 0**。
   于是 `SampleFacialNet` 报"rc=0 但打印 failed to load net"，
   我据此写下"AGENTS.md 那条『退出码 0 只证明没崩』在真实 sample 上被证伪"
   并提交了报告。后来去查"为什么源码里明明白白写着 `return EXIT_FAILURE;`"，
   查到没被重定义、二进制比源码新、二进制里确实有那个路径 ——
   一切正常，于是裸跑，三种写法三次都是 **1**。**sample 是对的，测量是错的。**

   > **测退出码时，不要把被测程序放进管道或命令替换里再取 `$?`。**
   > 最稳的写法是先重定向到文件、再单独取 `$?`：
   > ```sh
   > ./prog > /tmp/out 2>&1; rc=$?
   > ```
   >
   > 更一般的一条：**当"测出来的"与"读得出来的代码"矛盾时，
   > 尤其当那个测量结果刚好支持一个你已经想好的结论时 ——
   > 先把测量重做一遍。** 矛盾的解释成本最低的方向永远是"我这边错了"。

4. **门禁 fork 出几十个子进程时，子进程的 stderr 必须用追加（`"a"`）不是截断（`"w"`）。**
   一道门禁 fork 50 个子进程、**共用同一个 `ZQ_CHILD_ERR` 路径**，
   `freopen(p,"w",…)` 让每个子进程一启动就截断 ——
   **排在崩溃用例后面的用例会把崩溃用例的 sanitizer 报告擦掉**。
   症状极具欺骗性：门禁确实红了（结果文件读不出来 → rc=1），但报告栏永远空白，
   看上去像"根本没有 sanitizer 报告"。而且**只在崩在中途时发作**
   （崩在最后一个用例时报告恰好还在），所以之前每次跑出来都"正常"。
   配套：harness 拉回本地时**先删旧文件再 `cp`**，不能用 `cp -n` ——
   本地镜像目录从不清空的话，第一轮的空报告会一直挡在前面。

> 这四条与已有的「批量改代码之后重新看一遍被改那几行的控制流」同源：
> **坏掉的不是被测对象，是"用来测它的东西"，而它报出来的东西看着比真相更像真相。**
> 今天在同一道门禁上一次撞见三种形态：`(size_t)负数` 回绕成 1.8e19（19 例全报越界）、
> `fscanf` 字段数不匹配（每例都判"没跑完"）、`heredoc` 里的 `\uXXXX` 写出非法 UTF-8。

## 扫描/分类类工具的四条（2026-10-03 补，全部来自写 `probe_*` 家族）

今天新写了 5 个探针（文件可达性 / bottoms 计数 / 参数乘积 / 除数移位 /
头文件分类），**每一个都至少错过一次**，而且错法都不重样。

1. **"0 命中"不是结论，是"我还不知道这个探针坏没坏"。**
   阳性对照必须是**这个代码库里真实存在过的缺陷**：把已修好的地方改回去，
   要求探针报出来。附录 EQ 里 `probe_param_products.py` 报 0 命中看起来像
   "这一族已清干净"，而把附录 EM 那三处 `(__int64)` 去掉之后它**依然报 0** ——
   正则写成 `VAR * VAR`，而实际形状是 `dilate_H * (kernel_H - 1)`。
   > 现有的 `--selfcheck` 要求里有"必须不报"的负样本，**今天补的是另一条**：
   > **"必须报"的样本要取自真实代码，而不是手编的理想例子。**

2. **正则扫代码几乎总是有方向性。**
   同一处 `A * (B - 1)` 与 `(B - 1) * A` 要用两条不同的模式才都能匹配上；
   v1 漏了右括号、v2 漏了左括号，**两半看起来都很合理**。
   凡是"扫某一类语法模式"的工具，成对扫描（取一行里所有目标变量的位置，
   任意两个之间只看有没有 `*`）比正则更不容易漏。

3. **分类器不要拿整行做子串匹配。**
   附录 EU.6：`classify()` 拿编译器错误**整行**匹配 `NEEDS_LIB`，
   而 `NEEDS_LIB` 里有一项 `'nn'` —— 错误行开头是文件路径
   `/mnt/d/ZQCNN/...`，小写后含 `cnn`→含 `nn`，于是**每一条**错误都被判成
   "缺外部库"，`MSVC_ONLY` 与 `BROKEN` 两个桶从来没被填过。
   - 只对 `fatal error: <头名>: No such file` 里的**那个头名**匹配；
   - **没有这个形状就完全不进该分支**（能走到"用了 `__int64`"说明所有头都找到了）；
   - 白名单要短：`'nn'` 换成 `'nn/'` 之后**路径里的 `cnn/` 仍然含 `nn/`**。

4. **探针的临时工作目录必须每轮唯一。**
   附录 ES.4：`probe_zqlib_headers.py` 用固定的 `/tmp/zqprobe` 且开头 `rm -rf`，
   与完整回归内部那次调用撞车 → 91 个头凭空报 ERR、**错误信息是空的**。
   症状与"代码突然坏了 91 处"无法区分，靠的是**错误信息为空**这一点。
   改成 `pid + 时间戳`，并且**不删别人的目录**。
   > `run_zqlib_checks.py` 早就因为同一个毛病改过（附录 EB.1），
   > **修一处不够，全仓的探针都要看一遍**。

5. **写完扫描工具立刻做变异，别等回归里第一次踩坑**。
   上面第 1 条是"规则"，但 2026-10-03 那天为它交了**四次**学费
   （EP.2 两次、EQ 一次、FA 一次），四次都是**扫出来 0 命中**，
   而其中三次是探针坏了、代码是干净的。
   FA 那次尤其刺眼：探针叫 `probe_exempt_guards.py`，
   唯一的目标就是那个**跨行**写法的 `if (!global_pool
  && (...))`，
   而它的正则要求同一行内闭合 —— **它连自己要找的东西都找不到**。
   > **判据**：探针的阳性对照必须取自**它专为之写的那个形态**，
   > 而不是"同类里随便一个"。形态对不上时，"0 命中"是最可能的结果，
   > 也是最危险的结果。

**通用的两条信号**（今天各撞一次）：

- **荒谬的数字本身就是信号。** "410 个文件全部从未被编译"、"103 个 BROKEN、
  比修前的 13 还多"——这些数字荒谬到应该立刻停下来，我却各自又往下走了
  2~3 版才反应过来。**看到不对的数字先怀疑工具，再怀疑代码。**
- **一个从不变化的分类结果本身就该被怀疑。** 7 个非 OK **全部**是同一个桶、
  另外三个桶一个样本都没有 —— 那不是"分布"，那是分类器坏了。

## 门禁本身也要有人看着（2026-10-03 补，附录 GK）

`tools/check_filecount_bounds.py` 第 50 行是一句残句（6 空格缩进、没有 `#`，
从一行被截断的注释尾巴里掉下来的），Python 解析期就抛
`SyntaxError: line 50: unexpected indent` —— **那个文件一次都没被执行过**，
而它是附录 EL 的全部依据（报告正文直接引用它）。
**回归全绿。**

1. **写完门禁必须在同一次提交里接进 `run_audit_checks.py`。**
   "写好放着"和"没写"对回归是同一件事。`grep` 查一下自己的门禁名在不在
   入口文件里，这一步十秒钟。
   仓库现在有 `tools/check_gates_runnable.py`（组 C8）兜底：全仓每个 `.py`
   必须严格 UTF-8 且能过 `compile()`，每个 `tools/*.sh` 过 `bash -n`。
   **它放在快组是对的** —— 元门禁比它保护的对象慢，就等于有一样的失效窗口。

2. **测量和读出来的代码冲突时，两边都要怀疑：先怀疑测量，再怀疑渲染。**
   AGENTS.md 已有"先怀疑测量"；附录 GK 补的是另一半 ——
   同一处损伤，**整文件 `Read` 的渲染把一行的 `#` 和前半句吞掉了**，
   看起来比实际多一行残句。`cat -A` 看字节才是真相。
   **错的那一种渲染看起来往往更"完整"。**

3. **判据里"没有捕获组"就等于跳过了捕获组后面的过滤。**
   `check_filecount_bounds.py` 有一条上界模板写成
   `r'\b%s\s*[<>]\s*\w+\s*\|\|'`（无捕获组），于是
   "捕获到的数字够不够大才算真上界"这个过滤整条失效，
   `|| num < 0 ||` 被判成有上界。**删掉真上界，工具一声不吭。**
   > 凡是用"捕获组 + 阈值"表达严格性的判据，都要检查每条分支都有捕获组。

4. **基线的键里不能放任何"会随无关编辑漂移"的东西。**
   同一个门禁的键改了两版：`(文件,行号,…)` 会被无关编辑整体平移搞炸；
   `(文件,变量,分配调用名,…)` 更隐蔽 —— 分配名取自**行窗口**，
   删掉守卫里的一行会让窗口移一位、匹配到**另一个** `resize`，
   于是"判定变差"被报成"新增一个站点、少一个站点"。
   **抓是抓住了，但诊断是错的，而报错的诊断比没有诊断更费时间。**
   最终键是 `(文件, 变量)`，同变量多分配点取最差判定。

5. **写批量转换脚本时，正文用 raw 字符串。**
   往 md 里追加内容、脚本正文里写 `\b%s\s*[<>]\s*(\d+)` 这种正则，
   非 raw 字符串会把 `\b` 解成 **U+0008 退格符**写进文件。
   脚本会报 `SyntaxWarning: invalid escape sequence` —— **看到警告要停下看**，
   我这次是写完才回头看的。
   改完顺手扫一遍控制字符：`ord(c) < 32 and c not in '\n\r\t'`。

6. **给 WSL 脚本的路径必须是 WSL 路径，且目录要先 `mkdir -p`。**
   附录 GL.5：`check_blas_config.py` 把 Python `tempfile.mkdtemp()` 得到的
   **Windows 路径**塞进 bash 的 `cd`，`cd` 失败、而 `set +e` 让它继续跑，
   于是 8 个二进制和 16 个临时 `.txt` **全落在仓库根**。
   症状是"`git status` 里多出一堆 `e0.txt/o0.txt/p0`"，
   离真正的原因（路径分隔 + 目录不存在）隔了两层。
   同理，**临时文件要读回来也得用 WSL 侧的 `cat`** ——
   Windows Python 看不见 `/tmp/...`。
   > 这与本文件已有的「读文件清单用 `git ls-files -z` + 手工 utf-8 解码」
   > 是同一条原则：**跨进程边界时不要依赖默认的路径与编码转换。**

7. **「我验过的组合都绿」不等于「改动是安全的」—— 差着所有我没验的组合。**
   附录 GQ.4（2026-10-04）：我写脚本给 FP16 路径的重复定义批量加
   `#if !(__ARM_NEON && __ARM_NEON_FP16)` 守卫，
   **双向验**（FP16 档 + NEON-f32 档）都通过了，
   结果 v13 的 **Windows 构建**红了：`LNK2019` 三个符号消失。
   根因是我插入的 `#if`/`#endif` **配对到了一个 `#else` 上** ——
   `zq_cnn_lstm_32f_align_c.c:143` 的 `#else` 本来是第 37 行 `#if __ARM_NEON`
   的**非 NEON 分支**，而 x86 唯一走的就是那一支。
   * **验过的组合里没有 x86**：矩阵脚本默认不定义任何平台宏，
     而 ZQ_GEMM 的 SSETYPE 门禁只编那 3 个文件。
   * **教训**：靠脚本自动改生产代码时，判据必须覆盖
     **每一个会被发布的组合**，包含那些"我以为根本不受影响的"
     （这里是平台/架构，不是 SIMD 档位）。
   * 推论：**能自动化的只有"语义上可证等价"的改动**。
     改预处理结构（尤其跨 `#else` 或在 `#define` 宏块里）**不属于这一类** ——
     那需要人读那一段。

8. **不要在回归跑着的时候改生产文件。**
   v13 的 D1 第一次报 LNK2019 时，我先怀疑到自己的 GQ 改动，
   接着用 `git checkout HEAD~1 -- ZQCNN/` 去做对照 ——
   **而那一轮回归正在读同一批文件**，于是它读到的是半改半换的树。
   差点把"并发改文件造成的失败"写成"代码缺陷"。
   对照实验必须在**回归之外**做；要做就先把回归停掉。
   > 症状：同一个缺陷在"回退后"仍然出现，于是看起来像"与我无关"，
   > 而其实那个"回退后"的构建根本没重新编译（增量构建的 lib 是陈的），
   > 或者正被并发改动污染。**两种假象的解药都是同一句：
   > 干净地重来一次，一次只改一个变量。**


## 判据本身的三条（2026-10-04 补，附录 GZ；全是 GZ 这一批踩出来的）

1. **「没看到警告」不能当「没有」的证据 —— 尤其当判据本身就是"打没打警告"。**
   附录 GZ.1 里 `LoadFrom` 只检查"字节不够"、不检查"字节太多"。我写了个探针：
   给 `.nchwbin` 尾部追加 N 字节看 `LoadFrom` 的返回值。四组数据是
   `464 / 468 / 4560 / 66000`（追加量 1 / 4 / 4096 / 65536）——
   每个数都 = 追加量 + **464**，也就是说**基线本身就是 464**。
   而我当时只看了"原文件跑出来没打警告"，就写下"逐字节精确"：
   真相是**我根本没跑基线那一次**，把 464 当成了追加量。

   > 规律：`+基线` 这个常数一眼就该看出来，**看到一列数之间有固定差值，
   > 第一反应要是"有个基线我没测"，而不是"数据还挺整齐"**。
   > 推论：**给"有没有问题"做实验时，必须先跑一遍不加任何干预的原对象**，
   > 并把它的输出单独存下来 —— 否则你分不清"它本来就有的"和"我加进去的"。

2. **比较的两端都要先证明是自己以为的那两个文件。**
   附录 GZ.5 要证明"截短权重不改变行为"，做法是让库从**原始文件**回存一份、
   和截短后的文件 `cmp`。第一次打出了"逐字节相同"——
   但 **Git Bash 的 `/tmp` 和 WSL 的 `/tmp` 不是同一个目录**，
   备份文件 WSL 根本看不见（`failed to open` 就印在输出里），
   而 `cmp` 比的是上一轮扫描**遗留的旧产物**。
   两个条件同时坏掉，结论看着是对的。
   > 判据输入的检查：`rm -f` 掉预期产物 → 显式判"产物存在且非空" → 再比。
   > 这与本文件「Windows 路径丢给 WSL」是同一条原则的另一半：
   > **跨进程边界时不依赖默认的路径转换** —— 不只是"路径写错"，
   > 也包括"这个路径指向的东西不是你以为的那个"。

3. **门禁的列名/标签要说它实际数的是什么。**
   `tools/run_zqlib_checks.py` 生成的判据行里，那一栏叫 `nsan`、
   打印出来写死成"条 sanitizer 报错"。实际两套 sanitizer 模式下
   **两个列是互换的**（ASan 轮数的是 `grep -cE 'FAIL'`）。
   于是 `zq_model_params` 报了两行 `**FAIL**`，被显示成
   `FAIL (rc=1, 2 条 sanitizer 报错)` —— **一条 sanitizer 报告都没有**。
   我为此查了半天才发现门禁语义一直是对的，错的只是标签。
   > 这是本文件「没有证据就不要在失败信息里断言原因」的**上一级**：
   > 那条讲失败文案，这一条讲**汇总层的列名**。汇总层更容易被直接采信，
   > 因为它离数字最近、看着最权威。
   > **凡是把某一栏的失败原因写死在代码里的，改判据时必须一起改。**

## 补齐一处修复时，把**同仓的另一份拷贝**列出来（2026-10-04 补，附录 HA.3）

GZ 给三条加载路径都加了"权重尾部剩余"告警。实际只改了两份：
`ZQ_CNN_Net.h` 的两条 + `ZQ_CNN_Net_NCHWC.h` 的**一条** ——
`ZQ_CNN_Net_NCHWC.h` 的 `_load_model_file` **漏了**，
而 `SampleMTCNN_NCHWC4`（双平台回归里跑着）走的正是那一条。
**编译通过、回归全绿、实际一点作用都没有。**

> 动一份拷贝之前先做一遍"**这份功能在仓库里还有几份实现**"的枚举
> （本文件已有「同仓的两份实现互为对照」那条，这里补的是**收尾**那一半）：
> **改完一处，回头把名单上的其余几处逐个确认过。**
> 名单要在动手**之前**列，不是想起来才列。

## 判据的**粒度**必须比缺陷的粒度细（2026-10-04 补，附录 HA.5）

`Convolution::LoadBinary_NCHW` 的越界读在 **bias 段**，
而 filters 段**有**守卫 —— 于是"这个函数查不查 `buffer_len`"这个判据
**看到"有"就放行**了。

* 第一版扫描器数"36 个重载里有几个体内出现 `buffer_len` 的比较"，
  报「10 个有、26 个没有」，我据此说"只有 3 个有问题"。
* 两处错：① 正则 `buffer_len\s*[<>=]` **漏了不带 `=` 的 `>`**
  （`if (… > buffer_len)` 是这个仓库里的主流写法）；
  ② 判据停在**函数**级，而缺陷在**某一次读**上。
  改成"两次读 `buffer` 之间必须出现一次守卫"之后才露出来。

> 通则：**"这个 X 做了防护吗"这一类判据，只有当防护是 X 级别的才是好判据。**
> 防护在"每次读"上，判据就得按"每次读"数。
> 另外**改细之后仍然有假阳性**（`InnerProduct` 的
> `ConvertFromCompactNCHW((const float*)buffer,…)` 被报成无守卫，
> 其实上面几行的守卫同时覆盖了前面的 `memcpy` 和随后的强传）——
> **静态扫描给的是候选，判官是 ASan / 实跑。**

## 测"分段读取"的长度检查时，短 buffer 必须**两头都量**（2026-10-04 补，附录 HA.4）

新写 `zq_loadbuffer` 时我用 `{0, 1, 4, 16, 64}` 当短 buffer。
它**两度是假绿的**：删掉 `Normalize` 的守卫门禁 PASS（测错对象），
删掉 `DeConvolution` 的守卫**仍然** PASS ——

> 因为 DeConvolution 的第一段（filters）就有 1152 字节，
> 而我的短 buffer 长度**全都远小于 1152**，于是每次都在第一段的守卫那里被拒，
> **压根到不了有洞的那一段**。
> 改成 `{0,1,4,16,64, need-64, need-16, need-4, need-1}` 之后立刻报红。

* **缺陷几乎总在"最后一段"上**（前面的段先读、也先被守卫住），
  所以短 buffer 必须**从末尾也切一刀**。
* 顺带一条：`{1,4,16,64}` **从文件末尾**切的那条，一上来就抓到了
  `Convolution` 的 bias —— **两段方法只有一头是不够的**。
* 推论：**判据失效时先问"我的输入覆盖到缺陷所在的那一段了吗"**，
  而不是"我的判据够不够严"。严不严是另一件事。

## 改 C++ 源码不要走 `python - <<'PY'`（2026-10-04 补，第 N 次踩到）

用 heredoc 把 Python 脚本喂给解释器去改 `.cpp`，结果 C++ 字符串里的 `\n`
被吃成了**真换行**，源码里多出 5 处引号不配对的行，
报错是 `error: missing terminating " character` —— 指向真正的原因之外的地方。

本文件「行尾与跨平台编译规则」第 6 条已经写了"改源码和写 python 脚本一律用
Write/Edit 工具落盘"，这里补的是**症状**：症状是**编译错误**，不是编码问题，
容易往"是不是少了头文件"上找。

> 通用：**批改之后先扫一遍语法完整性**再跑构建 ——
> 我这次是"每行引号数为奇数"一行 Python 查出来的。
> 批改报错时，**先确认批改本身没坏**（`引号配平` / `括号配平` / `行数变化`），
> 再去读编译器的报错。

## 「计数 + 标签」的输出，交上去之前先把总数对一遍（2026-10-04 补）

同一天里犯了**两次同一个错**，而且这个错 AGENTS.md 里已经写着（GZ.3 那条）：

1. `run_zqlib_checks.py` 那一栏叫"条 sanitizer 报错"，实际数的是 `grep -cE 'FAIL'`。
2. `zq_weight_roundtrip` 的汇总行写"其中 %d 个与原文件逐字节相同"，
   而那个计数器在「相同」和「不同」两个分支里**都 ++** —— 它数的是
   "看过几个模型"。输出"其中 **23** 个逐字节相同"，实测是 **20** 个。

**判据（一行就能做）**：凡是"计数 + 标签"的输出，
**先把分项加起来对总数**（`23 = 20 + 3` 对得上吗）。
**对不上就是计数器写错了，不是数据变了。**

> 这类错之所以反复出，是因为它**不会让门禁变红** ——
> 门禁照样 PASS，只是**报出来的数字是错的**，
> 而错数字会被直接抄进报告（我今天就是这么把 23 抄进附录 HB 的，
> 写完报告去看原始数据才发现是 20）。
> 推论：**报告里的每个数字都要能追到它是从哪一行输出抄来的。**

## 计数与"未找到"哨兵不要共用一个初值（2026-10-04 补）

`DiffStat` 的初始化我写成 `st.ndiff = st.nbad = st.first_bad = -1;` ——
本意只是给 `first_bad`（"第一个坏的位置"）一个哨兵，顺手把 `nbad` 也设成 -1。
于是判据 `nbad != 0` **恒真**，23 个真通过的模型全被报成"对不上"。

* 症状极具欺骗性：失败信息是 **`首个在 #-1（0 vs 0）`** ——
  **一个荒谬的数字**。本文件已有"看到荒谬的数字先怀疑工具"，
  这次正是靠荒谬认出根因的。
* 根因是**一个初值承担了两种语义**：「计数为 0」与「位置未找到」。
  这两种东西不该共用 -1。
* 推论：给聚合体做初始化时，**计数成员和哨兵成员分开写**，
  别用一条链式赋值图省事。

## 变异测试的输出本身要能看出差异（2026-10-04 补）

`zq_weight_roundtrip` 的差异打印第一版用 `%g`（6 位有效数字）。
阳性对照是在 save 路径乘 `1.0000001f`，于是打出来是
`2.28729 vs 2.28729` —— **两个数一模一样**。
判据其实**抓到了**（它判的是"是否不同"而不是"差多少"），
但**人读那一行什么都看不出来**，于是它提供的证据约等于零。

> **打印出来的证据看不出区别，本身就是证据没打够。**
> 打印**比较出来的量**（相对误差 `%.3g`、差值的绝对量级），
> 不要只打印两个原始值。至少给 9 位有效数字。

## 输出里"谁的名字打在前面对应哪一段" —— 顺序就是归属（2026-10-04 补，附录 HC.5）

新写的 `zq_nchwc_roundtrip` 在 `LoadFrom` **之后**才打模型名，而 `LoadFrom`
会在 stdout 上刷一串 `warning: unknown para …`。于是那些警告读起来
属于**上一个**模型。我据此写下"det4-dw64-v3s / det5-dw96-v3s 两个模型
加载成功却带 `pad_type` 警告"——而 `grep -l pad_type model/*.zqparams`
只有 3 个文件，**那两个文件里根本没有 `pad_type`**。

* 症状极具欺骗性：**数字是真的、行也是真的**，只是挂在了错的行上；
  挂错行之后结论会从"没有缺陷"翻成"两个模型有缺陷"。
* 改法一行：门禁**先打 `>>> <name>` 再处理**，名字管到下一个名字为止。
* 这是本文件「通用工具的输出里不要写死某一次的具体描述」的**另一半**：
  那一条讲标签不许复用，这一条讲**归属不许有歧义**。
* 推论：凡是要把一段输出**按名字分段**再统计的工具，
  名字必须在**那段输出产生之前**打出来。

## 「某族符号在加载期合法被调用」要用**记录型桩** + 消费方断言它响过（2026-10-04 补）

`zq_net_fwd_tripwires.h` 的文件头写着「任何桩都不该被调到」——
2026-10-04 被证伪：`ZQ_CNN_Net_NCHWC::LoadFrom` 的最后一步 `_prepack()`
会调 `InnerProductPrePack`（`ZQ_CNN_Layer_NCHWC.h:2556`），那是**加载期合法调用**。

三条路，选哪条都要写清代价：

* 并进 `EXCLUDE` —— 代价是**照抄真实实现**，会与生产代码漂移
  （生成器自己就写明"排除项是有代价的"）；
* 链真实的 TU —— 代价是拖进整个内核库（`Forward_SSEUtils_NCHWC.cpp`
  一编就带出几十个 `zq_cnn_*_nchwc8_*`）；
* **记录型桩**（打一行 `LOADTIME:` 并返回 true，不中止）——
  **代价是它会变成静默 no-op**，所以**消费方门禁必须断言它响过**
  （生成器同时暴露一个计数器）。

> 通则：**"这一族例外"要放在例外清单里，不要放在调用点旁边加判断** ——
> 否则下一个同类符号会再加一次判断，而没人说得清清单到底有多长。
> 清单的**每一条都要写清"为什么它在加载期合法"**，并且要有自测
> （这里是生成器的 `--check`）。

## 阳性对照要**换一个变异位置再问一次**，哪怕上一轮刚绿（2026-10-04 补）

同一天里，阳性对照两次抓到的不是"库有问题"、而是"**我刚写的判据不够严**"：

* **HA.4**：新写的 `zq_loadbuffer` 用 `{0,1,4,16,64}` 当短 buffer。
  删掉 `DeConvolution` 的守卫，门禁**照样绿** —— 因为那五个长度全都远小于
  第一段（filters 1152 字节），每次都在第一段的守卫那里被拒，压根到不了有洞的那段。
* **HD.4**：新写的三个变体门禁**只比长度**。在 `ConvertToCompactNCHW` 上打
  1e-7 的内容扰动，`zq_nchwc_roundtrip` 红了 34 条，三个变体门禁**全绿**。

两次形态一样：**判据能通过，但它只覆盖了缺陷的一部分**；
而"通过"和"没验到"在输出上一模一样。

> 上一轮绿说明的是"**当前那个变异**测不出来"，
> 这一轮要问的是"**我这条判据到底测什么**"。
> 所以阳性对照**不能因为上一轮跑绿就省掉** ——
> 要**换一个变异位置**再问一次，而且**变异位置要覆盖判据的每一个维度**：
> 长度 / 内容 / 顺序 / 边界。
>
> 判据："如果这一条判据删掉，会不会有某个变异从它下面溜过去？"
> 想不出那样的变异，说明这条判据不测任何东西。

## 门禁的 `return 0` 不是"生产里没人用"的挡箭牌（2026-10-04 补）

新写的变体门禁第一版 `TRY_VARIANT` 末尾是 `return 0;`（不管 bad 是多少），
注释写"NCHWC1/NCHWC8 生产里没人用"。

* 这与本文件「生产不可达所以不改」那条的**补丁**冲突 ——
  那条讲的是"**要不要修代码**"，判据是"有没有可对照的正确实现"和
  "是不是内存安全问题"，**不看可达性**；
  而"**门禁要不要判失败**"是另一件事，门禁判失败的代价只是让人看一眼。
* 而且"编不过"那种情况本来就由 runner 记成 BUILD FAIL（那是红的），
  所以"只报不算"在 runner 这套机制里**根本实现不了** ——
  那个 `return 0` 实际只是把**运行期**的失败也放过了。

> 推论：**"这条判据会不会让人天天看见红"不是放过的理由**。
> 真会天天红的门禁要问的是"它红得对不对"；不对就修判据，不要关判据。

## 连续两次"看代码看起来是对的"就是停手信号（2026-10-04 补，附录 HE.5）

附录 HE 里我静态推演了两条路，**两次都判错**：
① 怀疑三个 BN 融合函数漏了 bias 项 → 排除（`a` 里已经含了 bias）；
② 怀疑 `_merge_bns_to_dwconv` 把通道索引加错了槽位 → 排除（按步长定义重算是对的）。
两次的错法都是**用直觉代替了代码里写着的定义**：第一次凭"c 是通道，步长该是 sliceStep"，
第二次凭"那个 `+ c` 看起来加错了槽位"。

> 与本文件「改了没变化」的补丁是**同一条**：
> 静态推演连续"排除"两次时，正确的下一步是**换一个更直接的测量**
> （做二分、造探针、跑真实调用），而不是把第三段代码也读一遍。
> 读代码的成本低、看起来在推进，所以特别容易连着读三四段。

## 「解开一段注释 / 恢复它」是**成对**的改动（2026-10-04 补）

`ZQ_CNN_Net.h:1546` 那段是 `/*if (...) { ... } else */if (...)`——
起始的 `/*` 与收尾的 `else */` **必须一起改**。当天我在实验里：
1. 只改起始处 → 忘了改收尾 → `error: unterminated comment`（离原因三层远）；
2. 恢复时只改了起始处 → 又一次 `unterminated comment`。

> 判据：动一段**跨行注释**时，先 `grep` 出**所有**与它配对的
> 起始/收尾标记（这里是 `else */`），**一次全改完**；
> 改完立刻 `grep -c` 确认残留为 0，再去编译。
> 症状（编译错误）离原因（少改了一处）隔了三层，**不要从报错那里往回猜**。

## 一个**恒红**的检查不要接进回归（2026-10-04 补）

`SampleMergeBNCompare` 现在是红的（`mobilefacenet-v1` 的 `merge_bn` 改了输出 0.3695，
根因未定位）。**故意没有**接进 `run_sample_regression.sh` 的执行列表。

理由与本文件「一个坏测试会产出看起来很有说服力的假结论」是**两件事**：
* 那条讲**假绿**（测试坏了却绿）；
* 这一条讲**恒红**——接进去之后每轮回归都红，而根因还没找到，
  于是**别的真回归失败会被这摊红淹掉**。
  与其这样，不如把它留在仓库里当**复现手段**，并把复现命令
  与"根因未定位"这件事**写进报告和 changelog**。

> 判据：**先修，再接**。中间那段用"可复现的 sample + 报告里的复现命令"顶着。
> 但**不能不写** —— 一个"存在于仓库里、没人跑、也没人知道它红"的 sample
> 比没有它更糟：它看起来像覆盖已经补上了。
>
> **2026-10-05 补（附录 HX）："先修再接"还有第三步 —— 修好之后要真的接进去。**
> `SampleMergeBNCompare` 修好当天已经进了 `run_sample_regression.sh`
> 与 `run_audit_checks.py` 的 `WIN_SAMPLES`。只把样本留在仓库里、
> 让下一轮改动**仍然**测不到那条生产路径，等于那一轮的成果没有落地。
> 接进去之后它报的是 "MERGE COMPARE OK"，**不是**静默。
>
> 顺带一条判据：**能不能接进回归，要看它依赖的输入在不在版本库里。**
> `SampleSliceMerge` 依赖 `slice_model_weights.py` **现场切出来**的
> `.zqslice/`（在 `cmake-out-*/Release/` 下、被 gitignore），干净克隆上不存在
> → 它必然 NOOUT/FAIL。**"这条检查有价值"与"这条检查能进回归"是两件事。**

## 「回归全绿」不等于「新增的东西编过了」（2026-10-04 补，附录 HE.8）

`--with-build` 只跑 `cmake --build build_x64 --config Release`，**不重新 configure**。
而 `SamplesZQCNN/CMakeLists.txt` 用 `file(GLOB ...)` ——
所以**新增的 sample / TU 不会进生成的构建系统**。
2026-10-04 实测：v28 回归报 `ALL CHECKS PASSED`，
而 `cmake-out-win32-x64/release/Release/SampleMergeBNCompare.exe` **根本不存在**。

> 本文件「构建规则」第 4 条已经写了"新增文件需要重新 configure 才生效"，
> 但那条的症状是**编不出东西**，这条的症状是**全绿** —— 后者更难发现，
> 因为你不会去看一个你以为"肯定编了"的东西在不在。
>
> **判据：新增 sample / TU 之后，先自己跑一次
> `cmake -S . -B build_x64 -G"Visual Studio 17 2022" -A x64` 再 build，
> 然后 `ls` 一下产物在不在**，再跑回归。

## 把门禁改写成 sample 时，"没检查 ≠ 通过"的守卫要**一起搬**（2026-10-04 补）

`zq_mergebn`（门禁版）里我写了
「没有找到 .zqparams —— 同样是"没检查"，不是通过」；
搬成 `SampleMergeBNCompare` 时**漏掉了**这条，于是 Windows 侧
`MODEL_DIR` 路径写错 → 17 个模型全 SKIP → `bad == 0` → 报 `MERGE COMPARE OK` rc=0。

> 与本文件「一个坏测试会产出看起来很有说服力的假结论」同源，
> 但这里更隐蔽：**守卫本身是对的，只是搬运时掉了**。
>
> **判据：把一个检查从门禁搬到 sample（或反过来）时，
> 逐条对着搬"什么情况下它不算通过"的那些守卫** ——
> 汇总行的字面禁令（不许出现 `FAIL`）、
> "一个都没跑过"、
> "读不到结果文件"、
> 子进程非 0 退出。
> 这些是**判据的一部分**，不是样板文字。

## 二分实验的三个自伤点（2026-10-04 补，附录 HF 一天踩了三个）

1. **开关要加在"整块"上，不能只加在那个函数调用上。**
   只跳过 `_merge_*` 调用而仍执行"删层 + 重连"，坏的是**另一个原因**，
   二分会指错方向 —— 我第一版二分就是这么设计的，写注释时才发现。
2. **`HE_X=`（空值）不等于未设置。** `getenv("HE_X") != 0` 对 `HE_X=""` **成立**，
   于是那一支被跳过，矩阵里四次结果一模一样，**白跑一轮**还不报错。
   > 判据：矩阵里每一行都要有一个**明确不同**的输出；
   > 四行输出一模一样时，先怀疑开关没生效，而不是"结论稳定"。
3. **开关的声明位置就是它的作用域。** 第一版把两个 `static const bool`
   声明在 `_merge_bn` 里、却用在 `_merge_bns_to_dwconv` 里 → 编不过。
   顺带这也是一条免费信息：**编不过就说明作用域不对**，比读代码确认快。

> 通则：**二分实验本身也是一件要验证的实验。**
> 先跑一次"两个开关都关"确认能回到基线（HF.1 第四行 = 0，基线自洽），
> 再看单边。少了这一步，"某一行结果没变"可能只是**开关没生效**。

## 排除到"算术已被证明正确、结果仍然错"时，**问题在调用上下文**，不在函数里（2026-10-04 补）

附录 HE/HF/HG 连续三轮把 `merge_bn` 的缺陷缩小，最后 HG 把
`_merge_bns_to_dwconv` 的**六条**嫌疑全部排除（a/b 未初始化、索引越界、
清零、a/b 反了、偏置折叠没参与、N != 1），而
"跳过整支融合"时误差是 1.095e-07 —— 也就是
**让 BN 层留着就正确，让 merge 替它算就错，而 merge 的算术被证明是对的**。

> 结论只能是：**问题不在那个函数的算术里，而在它被调用的上下文**
> （在哪一对层上被调用、调用之后层图变成了什么样）。
> 静态读代码到这里已经读不出更多了 —— HG 的四次排除全是"读代码 + 算一遍"，
> 再读第五遍大概率还是排除不掉任何东西。
>
> **判据：连续 N 次"排除"之后，如果被测对象是**一段被调用的代码**
> 而不是一整条独立路径，就该换装置**——
> 用 `Forward(input, start_layer, end_layer)` 这类**局部前向入口**把范围缩到一层，
> 再逐通道摆出两条路的输出。**"算术对但结果错"是一个强结论，
> 它排除的是"算术"这一整类原因，而不是某一处代码。**

## 报差异要报**结构**，不要只报最大值（2026-10-04 补，与「一个最差格」同源再推一步）

HG 那一版 sample 只打"最大后向误差 + 最差下标"，看不出形态；
加上「差异大的个数 / 前 8 个 (下标, 未融合值, 融合值) / 比值偏离最大的下标」之后，
一眼看出是 **126/128 全错且比值散乱** —— 而这个形态直接排除了
"整体差一个常数"和"少数通道坏"两种解释，把方向指向"通道错配"。

> 与本文件「一个最差格不能用来概括整体」是同一条；
> 这一条再往前推一步：**差异的形态本身就是判据的一部分**，
> 只报最大值等于把判据扔掉一半。
>
> 通用做法：判据失败时，除了"最差的那个"，再打三样 ——
> **超阈值的个数 / 总数**、**前几个 (下标, 期望值, 实际值)**、
> **比值偏离最大的下标**。这三样几乎总能指出下一刀该往哪儿砍。

## 整网比一个数 → **逐 blob/逐层比**（2026-10-04 补，附录 HH.1）

HE/HF/HG 连续三轮都在**整网**层面比（同一输入、同一最终 blob），
结果是三轮都没能定位到"哪一层开始错"——
因为误差经过后面几十层传播，最后那个数已经**不携带位置信息**了。

改成：两条路各跑一次完整前向，按 `.zqparams` 里 `top=` 的**出现顺序**
逐个 blob 取出来比。HH 一换上这个装置就拿到硬结论：
**前 70 个 blob 逐位正确，第一个分歧点是 `res4_block1_conv_dw`** ——
也就是"融合原理上没错，错的是这一个 blob 及其下游"。

> 与本文件「一个最差格不能用来概括整体」同源，
> 这一条再往前一步：**判据要能指出"位置"，而不只是"差多少"。**
> 通用做法：先用一个粗判据（全网 / 全文件）确认"有问题"，
> 然后**立刻**换一个**能指出位置**的判据（逐 blob / 逐元素 / 逐用例），
> 不要在粗判据上反复加精度。

## 差异的**形态**比差异的**大小**更能指出根因（2026-10-04 补）

HH 里同一个 blob 的差异形态是决定性的：

* 未融合：**平滑上升**的 ~1~2（沿 W 方向，像一张特征图）
* 融合：**~0.1 量级、符号乱跳**的噪声

> 如果错的是"某个系数乘错了 / 某个 `b_v` 取错了"，
> 输出**仍然是平滑的** —— 同样的输入、只是权重差一点，卷积结果必然连续。
> 只有"**读了别的张量**"或"**读了未初始化内存**"才会产出噪声。
>
> 这一条把 HG 那一整轮"读代码 + 算一遍"的排除全部接上了：
> 既然算术已证对而结果错，那就看**结果的形状**——
> 形态直接指向"输入/接线"这一类原因，而不是"算术"。
>
> 判据：拿到一对不一致的输出，**先看它是"平滑但偏了"还是"变成噪声"**。
> 前者查系数，后者查接线与内存。

## 写一个"通用比对"时，优先在**被测对象自己的输出**上比对（2026-10-04 补）

HH 用的判据全部落在库自己的公开接口上：`Forward()` / `GetBlobByName()` /
`SaveModel()`。这条路的价值是**不碰生产代码就能测**，
而且生产代码一动，判据自动跟着变。
代价是它只能覆盖"接口能观测到的东西"。

> 推论：审计一个函数时，**先问"有没有一个公开接口能把它的输出摆出来"**。
> 有就直接在那个接口上建判据，别急着改代码加打印。
> 本轮三次定位（HE 找到活缺陷、HF 二分到函数、HH 定位到 blob）
> **一次都没有改过生产代码**。

## `Forward(input, start_layer_name, end_layer_name)` **不是**"从输入算到 end"（2026-10-04 补）

它的循环是：

```cpp
for (int i = 0; i < layers.size(); i++) {
    if (layers[i]->name == start_layer_name) has_begin = true;
    if (!has_begin) continue;          // <-- start 之前的层**一律不跑**
    ... layers[i]->Forward(...)
    if (layers[i]->name == end_layer_name) { has_end = true; break; }
}
```

**它假定 `start_layer_name` 那层的 bottom blob 已经被填好了。**

> 我第一次看到它时以为"给定一段层名就能从输入重算这一段"，于是设计了一个
> "跑到 `conv_dw` 之前停下、读它的输出 blob"的测法 ——
> 那个测法**根本拿不到东西**（那一层没跑，blob 是未写状态）。
>
> 判据：**用"局部前向"做任何测量之前，先读它的循环体**，
> 确认三件事：起点之前跑不跑、起点那一层跑不跑、终点跑完跑不跑。
> 这三个"跑不跑"决定了你能观测到什么。

## 缩小范围时，**先看输入对不对**，再看算术（2026-10-04 补，附录 HH.7）

拿到"结果变成噪声"时，直觉首选嫌疑是"图接错了"——
而 HH.7 的一条推论把它排除了：出问题那一层的 bottom blob
落在逐 blob 扫描的"**全部正确**"区间里，也就是**输入是对的**。

> 于是嫌疑收缩到"**这一层自己的 Forward**"，
> 具体到 `_merge_bns_to_dwconv` 走的那条
> `if (bias == 0)` 分支 —— 它把 `with_bias` 从 false 改成 true，
> 于是 `Forward` 走的是**另一条分支**。
>
> **前一轮的六条排除全都针对"权重与下标"，没有一条覆盖"Forward 换分支"** ——
> 排除清单本身也有**盲区维度**：按"哪一类原因"列一遍，
> 会发现某一类根本没被覆盖过。
>
> 判据：写排除清单时按**原因类别**列（算术 / 下标 / 内存 / 接线 /
> **控制流分支** / 状态），而不是按"我想到的几个函数"列。
> "控制流分支"这一类最容易被漏，因为它在**被测函数之外**。

## 一个"**多写者**"的 blob，它的值取决于读到它时**跑到了哪一步**（2026-10-04 补，附录 HJ.2）

`mobilefacenet-v1` 的 `res4` 段里，`res4_block1/2/3/4_conv_dw`
**四层都写同一个 blob**（读同一个 skip 源、权重不同）。
所以"用局部前向停在某一层、读那个 blob"拿到的
**是那一层的输出**，而 blob 的**最终内容是最后一个写者的输出**。

我第一版拿"停在 block1"的值当分母去算比值，
等于**拿两个不同的卷积做比值** —— 结论直接作废。
改成停在**最后一个写者**（新增 `last_writer_of(zp, blob)` 从 `.zqparams` 里找）
之后，方向没变但证据完全不同（最大相对极差 78.69 -> 322.6）。

> 判据里**只要用了某个中间张量**，就先确认两件事：
> ① 它被几层写？② 我读到它时，跑到了第几个写者？
> 这是「判据要能指出位置」的**反面**：
> **位置找准了，但位置上的东西是不是同一个，还得再确认一次。**
>
> 推论：**复用中间量做判据时，"取它的时机"是判据的一部分**，
> 不是一个可以随手选的实现细节。

## 测量本身出错时，**先把对象换对了再下结论**（2026-10-04 补）

HJ 的第一次结论（"不是同一批权重 × 同一份输入"）方向是对的，但**证据是错的**
（比错了对象）。如果当时因为"方向对"就收手，报告里会留下一条
**站不住但看起来很有力**的结论。

> 与本文件「先证明测量是对的，再相信它的结论」同源，
> 这条是它的**加强版**：不是"测量错了要重做"，
> 而是"**结论方向对不代表证据对**"——
> 方向对会让人**放松警惕**，比方向错更危险。
>
> 判据：拿到一个结论后问一句"**如果我的测量对象换了，结论会变吗？**"
> 会变 ⇒ 现在的证据撑不住这个结论，得把对象确认清楚。

## 判据要选**不依赖你对数据格式的理解**的那一条（2026-10-04 补，附录 HK.2）

验证"融合 A 与 B 是不是同一个变换"时，我第一版把判据写成
「B 是否等于**手写参考实现**」。结果参考实现与库差 **0.14** ——
而真实结论是两者只差 **1.7e-08**。

原因：**`B vs A` 两侧都过库的解释**，所以它与"我怎么理解权重文件"无关；
而"对比手写参考"依赖我把权重布局、padding 语义、var 的定义**全猜对**。
第一版就猜错了（`(kh,kw,c)` vs compact NCHW 的 `(c,kh,kw)`），换了之后**也没对上**。

> 通则：**一道判据里，凡是需要"我理解数据格式"才能算出来的那部分，
> 优先换成"两边都走被测代码"的那部分。**
> 前者是**独立实现**（能发现两边都错），后者是**同一实现**（只能发现"变了"）。
> 两个都要，但**先后顺序要对**：先用不依赖理解的判据把"变了没有"定下来，
> 再用独立实现去查"对不对"。
>
> 推论：**独立实现没验过之前不许当判据** ——
> 否则就是拿一个没验证过的工具去判别人错，
> 而它报出来的数字看着和真结论一模一样（这与
> 「一个坏测试会产出看起来很有说服力的假结论」同源）。

## 合成用例的**两个分支**要各自覆盖到（2026-10-04 补，附录 HK.5）

`_merge_bn` 的守卫是两个析取项：

```cpp
if (tops[i + 1][0] == bottoms[i + 1][0] || !later_refer) { ...融合... }
```

`Input -> DWConv -> BN` 这个**最自然的**三层合成网，`later_refer` 是 **false**
（BN 是最后一层，没人再读那个 blob），只覆盖了 `|| !later_refer` 那一支。
真模型里（ResNet 共享 skip 路径）`later_refer` 是 **true**，走的是**另一个析取项**。

> 我拿合成网验完"融合是对的"就以为覆盖到了 ——
> 而**同一个 `if` 的两个析取项是两个不同的代码路径**。
>
> 判据：**写最小合成用例时，把被测函数里每个分支条件都列出来，
> 确认合成用例命中了哪几个、还差哪几个。**
> 最容易漏的正是"短路条件的另一半"——
> 因为最自然的用例往往只走到第一个条件为真的那条路。

## 「加载失败」只说明**那个位置的数据**不对，不说明是**那层**写错了（2026-10-04 补，附录 HL.2）

造第三个合成用例时连踩三个坑，**症状全是同一句**：

| 我犯的错 | 报出来的话 |
|---|---|
| 层名 `c1` 当成 blob 名用（`bottom=c1`，而那层的 top 是 `skip`） | `unknown blob c1 in Layer dw1` |
| 权重文件按 c1,dw1,dw2,bn1,bn2 写（网里是 c1,dw1,**bn1**,dw2,**bn2**） | `Failed to load Binary for layer dw2` |
| 普通 Convolution 的 filters 写成 C 个（应是 `[N][kH][kW][C]` = **C×C**） | 同上 |

真实原因在**三个不同的地方**，而症状长得一模一样。

> 与本文件「没有证据就不要断言原因」同源；
> 这一条更具体：**"某层加载失败"是个关于"文件偏移"的报错，
> 不是关于"那层参数"的报错。**
>
> 判据：排这类错时按**数据在文件里的偏移**往前推
> （"X 这层需要几个 float？我写了几个？前面各层写了几个？累计对不对？"），
> **不要按"X 这层的参数对不对"去猜**。
>
> 顺带：**层名与 blob 名在 `.zqparams` 里是两个独立字段**，
> 它们可以取不同的字符串（`name=c1 top=skip` 完全合法），
> 而 `bottom=`/`top=` 引用的**永远是 blob 名**。

## 两条轴**各自**验过 ≠ **交叉**验过（2026-10-04 补，附录 HL.3）

合成网用了 `C=13`（不是 align 的倍数，`pixelStep=16 != C`，**有 padding**）；
而真实模型是 `C=256`（`pixelStep == C`，**无 padding**）。

* HG 在**真实模型**上证明"缩放的下标对每个实际形状都可证正确"——
  那些形状**全是** `pix == C` 的无 padding 情形；
* HK/HL 在合成网上证明"融合是对的"—— 那些形状**全是** `pix != C` 的有 padding 情形。

**两条轴各自验过，交叉没验过**，而缺陷很可能正好在交叉点上。

> 判据：列"已验过的条件"时按**每个维度**列一张表，
> 标出哪些格子验过、哪些没验过。
> 合成用例最容易只覆盖"我挑的那个值"，而真实数据往往落在**另一个**值上。
> 动手改合成用例之前，先做这件事。

## 「阈值意义上的没超过 X」**不是**「逐位相同」—— 写报告时别把两者混用（2026-10-04 补，附录 HM.2）

HH.2 我写了"**前 70 个 blob 逐位正确**，第一个分歧是 #70"。
把扫描阈值从 1e-4 降到 1e-9 之后：

```
逐 blob 扫描：145 个 blob 里 145 个对不上；第一个是 #0 "conv1"
```

**从网络的第一个 blob 起就有差异** —— 那是第一次融合引入的 float 舍入，
完全正常。那句"逐位正确"是**在 1e-4 阈值下说的**，被我写成了"逐位"。

> 差别有多大：#0 到 #69 是一条平缓曲线（2.7e-09 → 1.9e-08），
> 到了 #70 **一步跳到 0.1055**（7 个数量级），#75 又跳回 1.3e-08。
> **如果沿用 1e-4 的阈值，#0–#69 全部"没问题"，而 #70 那个 7 个数量级的突变
> 会被当成"误差在深层放大"** —— 结论会完全反过来。
>
> 判据：**写"逐位/完全正确"之前，先把阈值调到 0（或者 1 ULP）跑一遍**。
> 只要还有非零差异，就不叫"逐位"。
> 推论：**判据阈值与"归因阈值"要分开** —— 前者决定过不过，
> 后者决定"问题从哪一层开始"。这一批之前我把两者设成了同一个 1e-4。

## 误差的**增长曲线**能区分「舍入放大」和「某一层坏了」（2026-10-04 补，附录 HM.3）

* 平缓增长（2.7e-09 → 1.9e-08，跨 70 个 blob）= **float 舍入被逐层放大**，正常；
* **一步之内跳 7 个数量级**，然后下一步又跳回去 = **某一层本身坏了**。

> 关键在**跳完之后又跳回来**：
> 如果只是"越往后越不准"，曲线应该是单调的；
> 出现"突变 -> 回落"说明那一支坏了、**而并行的另一支（skip）仍然是好的**。
>
> 判据：拿到"结果不对"时，**先把误差沿计算图逐层打印出来**，
> 看它是**单调增长**还是**有突变**。
> 单调 -> 查精度/阈值；有突变 -> 直接去那一层。
> 这比"二分查找第一个超阈值的 blob"更早给出答案，
> 因为二分会把"突变"和"缓慢越过阈值"混成同一件事。

## 合成用例失败到某个数量还不复现时，**"复现不出来"本身就是结论**（2026-10-04 补，附录 HN.3）

24 组合成配置（4 组通道数 × 6 个用例）全过之后，
`merge_bn`（depthwise）+ `merge_prelu` 在**我能合成的全部结构与数值条件下**
都被证明是正确的，而生产那条 `mobilefacenet-v1` 仍然差 0.37。

这时候继续加合成用例的**边际收益已经很低**了 ——
因为每加一个，都是在"猜真模型还有什么我没模仿到的特征"。
诚实的做法是把它写成一条**否定性结论**，并把剩下的可能**分类列出**：

1. 依赖**真实数据**的某个我还没覆盖的数值特征（不是"极小/极大"这种极端，
   而是某种中间分布）；
2. 缺陷在**被测函数之外**，只是在与它交互时显形
   （而合成网里那些交互的**规模**太小，不足以触发）。

> 判据：**合成用例加到第 N 个还不复现，就停下来把"排除了什么"写成结论**，
> 而不是继续往下堆。堆到第 50 个还没复现，那 50 个用例的**信息量**
> 已经在第 8 个就取完了，剩下的只是重复。
>
> 推论：否定性结论要**连同"剩下的可能"一起写**，
> 否则读的人会以为"没查出来"等于"没问题"。

## 猜"合成用例缺了什么"时，**先把真模型那一段抄出来读**（2026-10-04 补，附录 HN.1）

HM.4 猜的缺口是"`c1` 后面少了 BN/PReLU"—— **猜错了**。
把真模型的 `.zqparams` 那十几行抄出来读一眼就看到真正的东西：
`res4_block1_conv` 这个 blob **被 block1..block5 的 conv+bn+relu 反复覆写**，
而 block1..block4 的 dwconv 读的正是它。
**"写者不止一个"**才是那个形状的关键，BN/PReLU 写不写都不影响。

> 判据：说"合成用例缺了 X"之前，先把真模型对应的**那十几行**原样读一遍，
> 找出**结构上**（不是参数上）的差别。
> 参数级的差别我几次都猜错过（eps、权重布局、BN 的 bias 项），
> 而**结构级的差别**（几个写者、谁先谁后）才是我每次都漏的那一个。

## 排根因时先问「**有没有一个环境变量/开关能直接排掉一整类**」（2026-10-04 补，附录 HP.3）

我为了排"是不是 use-after-free"，先造了 7 个合成用例、扫了 N=1..48 的规模 ——
一共十几次运行，**全都没复现**，于是差一步就去编整个库的 ASan。

**一个环境变量就把它排掉了**：

```
不开                        后向误差 0.3695  [0 1.15908->0.493928] …
MALLOC_PERTURB_=42（0x2A）  后向误差 0.3695  [0 1.15908->0.493928] …
MALLOC_PERTURB_=170（0xAA） 后向误差 0.3695
```

`MALLOC_PERTURB_` 会把**被释放的内存填成指定字节**。
如果读到的是悬空指针，**打开扰动结果必然变**；它没变 ⇒ 不是 use-after-free。

> 判据：排一类根因之前，先问一句
> **「有没有一个开关/环境变量，能让这一类根因'自曝'？」**
> - use-after-free / 未初始化读 -> `MALLOC_PERTURB_`
> - 越界读写 -> ASan（能定位到**哪一行**，不只是"有没有"）
> - 有符号溢出 / 除零 -> UBSan
> - 浮点精度 -> `-ffloat-store`、不同优化等级
> - 并发 -> `TSAN_OPTIONS`、把 `OMP_NUM_THREADS` 设为 1
>
> **这类开关的性价比远高于"再造几个用例"** ——
> 用例要靠"猜它长什么样"触发，而开关是**无条件生效**的。
> 本附录之前我连造七个案例如果为零，而环境变量一次就给出了确定答案。

## 悬空与"读到了另一份有效数据"要分开（2026-10-04 补）

`MALLOC_PERTURB_` 不改变结果 ⇒ 读到的是**有效、确定、但不是它自己的**数据。

> 两者症状**很像**（都是"输出一片乱"），修法完全不同：
> 悬空 = **生命周期**问题（谁提前释放了）；读到另一份 = **指向**问题（连错对象）。
>
> 判据：看到"输出一片乱"先问 **"这堆乱值每次跑都一样吗？"**
> * 每次都一样（且扰动不变）⇒ **确定性的错误指向**，不是内存失效；
> * 每次不一样 ⇒ 才有内存/未初始化的嫌疑。
> 这一条**比跑 ASan 便宜得多**，而且能先把方向定死。

## 调查编译器行为时，**先确认用的就是项目实际用的那个编译器**（2026-10-04 补，附录 HQ.4）

我调查 C4819 时用的是 `cl.exe` 路径里**第一个** —— `14.16.27023`（VS2017 工具集），
而项目实际用 **VS2022（`14.35.32215`）**。两者的行为不一样，
我据此得出的一系列"结论"（"加 BOM 能解决"、"挪到 include 之后能解决"）
在真实构建里**一条都不成立**。

> 判据：查编译期行为（warning、codepage、`/utf-8`、`/W`、优化等级）
> 之前，先确认两件事：
> 1. **工具集版本** —— `ls -d ".../VC/Tools/MSVC/"*` 把所有版本列出来，
>    再从 `build/CMakeCache.txt` 或构建日志里确认**实际用的是哪个**
>    （`ls | head -1` 取到的不一定是项目在用的那个）；
> 2. **旗标** —— `add_compile_options` 只影响它**之后**创建的目标，
>    所以"根 CMakeLists 里有这个旗标"不等于"我编的这个目标带着它"。
>
> 推论：**"直编 cl 的结果"与"构建系统的结果"对不上时，先怀疑旗标/工具集，
> 不要急着把直编的结果当结论写进报告** —— 我这一次两样都栽了。

## **控制台不是 UTF-8 时，别用它的输出去判断"文件里有没有某个字符"（2026-10-04 补）

我写了一句 Python 打印"这个文件含不含中文"，控制台是 GBK、输出成了乱码，
我看着乱码**以为答案是"有"** —— 而真实答案是 `SampleMTCNN.cpp` 的
非 ASCII 字符种类 **0**（它根本没有中文）。
于是我整个调查的前提就是错的。

> 判据：脚本要输出**非 ASCII 字符**时，**用 ASCII 输出结论**
> （比如 `print('nonascii', n)` 或 `print('U+%04X' % ord(ch))`），
> **不要打印字符本身**。
> 需要看内容就把结果写进文件再用 Read 工具看。
> 与本文件「cl 的输出是本地代码页」是同一条：凡是要**判断内容的输出**，
> 先确认它经过的通道是不是无损的。

## 同一件事猜三次就该换成"测"（2026-10-04 补，附录 HR）

参考实现与库对不上时，我逐个猜可疑参数，**连错三次**：
`eps` 没写进 `.zqparams`、权重布局是 `(kh,kw,c)` 还是 `(c,kh,kw)`、
padding 是补 0 还是 clamp。**每次都有道理，每次都不对。**

第四次改成**搭一个标定装置**：给权重每个位置填**唯一值**（`i+1`）、
用**单点输入**去探测，从输出反解索引映射。
于是"布局是哪种"从"我觉得是 A"变成
**"A 命中 271 个样本里的 1 个，B 命中 0 个，两个都不对"**。

> 判据：**为"某个映射/约定是什么"猜了两次以上，就改成设计一个能把它测出来的装置。**
> 典型装置：
> - 给每个位置填**唯一值**，让"哪个值出现在哪"成为可数的事实；
> - 用**单点/单通道/单个用例**去探测，把多变量问题降成一维；
> - 把未知量变成**匹配数**而不是判断题。
>
> **可数 > 可辩。** 而且要标出**哪几组结果可信** ——
> 我第一版在 `C=13`（有 padding）上标定，那组结果**从一开始就不该采信**，
> 却在结论里用了。判据自己也要标可信度。

> 另一个附带收获：装置建好之后，**"没定出来"本身也是结论** ——
> 它把"要查什么"从"从头搭"变成"直接接着跑"。

## 拿到干净结果之后**顺手加的增强**，坏掉了就一起回滚（2026-10-04 补，附录 HS.3）

`SampleMTCNNThreadSweep` 第一版跑出来是干净的（4 种 thread_num 检出逐位相同）。
我顺手加了"每个 thread_num 重复 8 次"来提高对 race 的灵敏度 ——
理由完全正确（race 非确定性，跑一次碰不到不等于没有），
但那一版出现了 **`bad++` 了却没有任何 FAIL 打印**。

我没有继续调它，而是**回滚到能跑对的那一版**。

> 判据：**"增强后的检查跑不红"不等于"增强有用"** ——
> 一个连自己都对不上的判据，比没有判据更糟，因为它会**制造假绿**。
> 拿到干净结果之后动它，就要准备好**回滚的手段**
> （新文件要先 `git add` 再改，或者把原版留成可重建的脚本）。
>
> 而"为什么回滚"必须写进报告和样本的头注释 ——
> 否则下一个人会以为"这里本来就只跑一次"，把那个不足当成设计。

## `system()` 在 Windows 上**不要自己加引号**（2026-10-04 补，附录 HT.4）

MSVC 的 `system()` 已经把整条命令包在**一层引号**里交给 `cmd.exe /c`。
你在字符串里再写 `\"`，cmd 收到的是**字面的引号**，
于是把 `"D:\...\exe"` 整个当成命令名：
    '\"D:\ZQCNN\...\SampleMTCNNThreadSweep.exe\"' 不是内部或外部命令
**rc=1**，而且因为命令里带了 `>/dev/null 2>&1`，**错误信息被自己吞掉**，
现象是"子进程全部没启动"而看不到原因。

两个配套的坑：
* **`argv[0]` 不可靠**：从 Git Bash 用 `./foo.exe` 启动时 `argv[0]` 就是
  `./foo.exe`，而 **cmd 不认 `./`**（Linux 没事，POSIX shell 认）。
  Windows 侧用 `GetModuleFileNameA` 取绝对路径。
* **别用 `>/dev/null 2>&1` 吞掉错误**：先把错误显出来，再决定要不要吞。

> 修法：Windows 侧**不加引号**，并对**含空格的路径显式拒绝**
> （报"请从不含空格的目录运行"），而不是静默跑出一条错的命令。
>
> 附带一条更有价值的：**父进程那个"子进程没写出结果文件"的守卫是对的** ——
> 它把"全都没启动"报成了**失败**而不是**通过**。
> 这个守卫在第一次跑（HS）时没有，第二次（HT）就有了，而它立刻
> 把一个"静默全灭"的假绿变成了可见的红。**父进程必须能观测到子进程的失败。**

## 跨边界（Windows Python <-> WSL）要**写进工具里**，光写进 AGENTS.md 不够（2026-10-04 补）

同一天在"Windows Python 与 WSL 的路径不是同一个"这条上栽了**三次**：
  * 探针 exe 放 `tempfile.gettempdir()` -> 每次都"探针还没编"（WSL 写的文件 Windows 看不见）
  * 临时切片文件放 Windows temp -> 探针（WSL 里）读不到，`consumed()` 返回 None
  * `subprocess.run([exe, ...])` -> `[WinError 193] %1 不是有效的 Win32 应用程序`
    （探针是 Linux ELF，Windows Python 不能原生执行）
> 前两次的教训**上一轮已经写进 AGENTS.md**（HQ.3），
> 结果今天又栽了两次。**"写进 AGENTS.md"不等于"下次不会再犯"** ——
> 真正管用的是把它变成**工具里的固定做法**：
>   * 跨边界双方都要碰的文件，放**仓库内**的构建产物目录
>     （本项目 `.gitignore` 已有 `cmake-out*/`，不会进版本库）；
>   * 要在另一边执行的二进制，一律**经 `wsl --` 启动**，不做"能不能原生跑"的判断；
>   * 路径转换收敛到一个 `to_wsl()` 函数，**不要**在每个调用点各写一遍。
>
> 判据：**凡是"这个文件/这个进程要在另一边被看到"的，
> 它的位置与启动方式就是工具的一部分**，不是调用方的事。

## **反斜杠不要放进 shell heredoc**（2026-10-04 补，当天连栽四次）

`cat >> f << 'EOF'` 与 `python - << 'PY'` 里写 `\n`、`\s`、`\`，落到文件里会变成
**真换行**或被吃掉一个字符：

| 我写的 | 文件里实际是 | 症状 |
|---|---|---|
| `printf("...\n")` | `printf("..."` + 换行 + `")` | `error: missing terminating " character` |
| `"%s\slice.zqparams"` | `"%s\slice.zqparams"` 的 `\s` 被吃掉 | 拼出 `.zqsliceslice.zqparams`，找不到文件 |

**一天栽了四次**（`\n` 三次、`\s` 一次），而且**每一次的症状都指向别处**
（引号不配对、路径拼接），没有一次直接说"你的反斜杠被吃了"。

> 判据：**要写含反斜杠的内容，一律用 `Write` / `Edit` 工具落盘，
> 不要经 shell heredoc。** 路径**全用正斜杠**（Windows 的文件 API 本来就接受），
> 这样连一个反斜杠都不用写。
> 批改之后先 `grep` 一遍有没有 `\s` `\` 之类被吃剩的碎片。

## 相对路径要连"**cwd 在哪**"一起写进注释（2026-10-05 补，附录 HV.5）

本项目的 sample **一律从产物目录里跑**（`cmake-out-*/Release/`），
所以代码里"相对路径"的基准是**产物目录**，不是仓库根。
我把候选目录写成 `cmake-out-win32-x64/release/Release/.zqslice`，
而 cwd 已经是产物目录，于是拼成 `<产物>/cmake-out-win32-x64/...` —— 找不到。

> 这与本文件「sample 必须在产物目录里跑」是同一条，但**后果不同**：
> 那条说的是"跑在哪"，这条说的是"**代码里那些相对路径按哪个基准算**"。
>
> 判据：凡是代码里出现相对路径，旁边**必须**有一行注释写明
> "相对产物目录 / 相对仓库根"，否则下一个人（和下一个我）一定会按错的那个算。

## 「折/融合/重写」类代码：守卫要按**语义前提**重推，不能只验证算术（2026-10-05 补，附录 HX）

`ZQ_CNN_Net.h` 的 `_merge_bn` / `_merge_prelu` 把「卷积 + 紧随其后的 BN/PReLU」
折成一层。折了 11 轮都定位不到的活缺陷（`mobilefacenet-v1` 生产实参下输出被改
**0.37**）最后是这么落地的：

```cpp
// 原来只有这一个条件：
if (tops[i + 1][0] == bottoms[i + 1][0] || !later_refer)   // 折
// 加上前提之后：
if (tops[i][0] == bottoms[i + 1][0] &&                      // <== 缺的就是这一条
    (tops[i + 1][0] == bottoms[i + 1][0] || !later_refer))
```

原来的条件问的是「**这个卷积的输出后面还有没有人要**」，
而真正要问的是「**后一层吃的是不是这个卷积的输出**」。
`mobilefacenet-v1.zqparams` 第 108/109 行写的是：

```
DepthwiseConvolution  name=res4_block5_conv_dw    bottom=res4_block1_conv   top=res4_block5_conv_dw
BatchNormScale       name=res4_block5_conv_dw_bn bottom=res4_block1_conv_dw top=res4_block1_conv_dw
```

于是 block5 的 BN 读的是**上一个 block 留下的** blob，block5 的 dwconv 输出
**压根没人读**（死层）。原图算「别人的值 × 逐通道系数」，
融合后算「这个卷积自己的输出 × 逐通道系数」—— 两个数差了 0.37。

修完：`0.3695 -> 2.92e-07`，17 个模型全过（`SampleMergeBNCompare`）。

三条可迁移的：

1. **优化/折叠/重写的前向代码，守卫条件要按语义重新推一遍**，
   不能因为「算术被证明是对的」就认为条件也是对的。
   `later_refer` 那条守卫是**充分**的直觉、不是**必要**的前提；
   少的那一条恰好是唯一能区分两种拓扑的东西。
   > 与本文件「排除到算术已被证明正确、结果仍然错」是同一条，
   > 但**收尾那一步不同**：那一轮我写的是「问题在调用上下文」，
   > 而这次收尾的方式是**把「上下文」写成一条新的前提**，
   > 写进 `if` 里，而不是停在"我知道它在上下文里"。
2. **造合成用例时，要把真模型里相邻层对的 `(top, bottom)` 组合全部列出来。**
   十一轮二分全是负结果，根因是每一个合成网都只有 `conv.top == BN.bottom`
   这一种拓扑 —— 而真模型里那一格**恰好是唯一一处不一样的**。
   > 与本文件「合成用例的两个分支要各自覆盖到」「两条轴各自验过 ≠ 交叉验过」
   > 同源；这里补的是第三个维度：**不只是"条件"的分支，还有"接线"的数据**。
   > 判据：造合成用例前，先把真模型对应那几行**逐行抄出来**，
   > 对每一对相邻层记下 `(上一层的 top, 下一层的 bottom)`，
   > **去重之后逐格覆盖**。列出来常常只有 2~3 格，而真值往往落在第 2 格上。
3. **模型文件本身可能是错的，库不能"顺手把它修好"。**
   block5 的 dwconv 是死层、`BN.bottom` 写错 blob 名 —— 那是**模型**的缺陷
   （本仓库不拥有那个模型，改它会改变所有人的 embedding）。
   库的契约是「融合不改变结果」，所以正确做法是**不折**。
   > 这条决定了修法的方向：不是去改 `.zqparams`，
   > 也不是去"让融合也复现模型的 bug"。
   > **优化路径的契约是对齐到未优化路径，不是对齐到"更合理"的语义。**
   > 判据：改优化/融合类代码前，先问「**它有没有可能比原路径更'对'？**」
   > 会的话，那就是要修的是它，不是原路径。

配套门禁：`tools/check_bn_prelu_pairing.py`（扫全仓 `.zqparams` 里
「后一层读的 blob ≠ 前一层写的 blob」的相邻层对，基线外判失败）。
**写这类门禁时正面记一条已知项**（随仓 1 处），比"全仓 0 处所以这条路没风险"
要诚实得多 —— 0 处只说明这条路上**没人走过**。

4. **改守卫之前先 `grep -c later_refer`，把同仓拷贝一次列全。**
   这一段守卫有**三份**：`ZQCNN/ZQ_CNN_Net.h`（主）、
   `ZQCNN/ZQ_CNN_Net_NCHWC.h`（**生产代码**，`SampleMTCNN_NCHWC4` 走它）、
   `ZQCNN_to_MNN/converter/source/ZQ_CNN_Net.h`（附录 U 那份拷贝）。
   各 5 处 = **15 处**，改完逐文件核对（5/5/5，旧的写法 0 处）。
   > 只改主文件的话，**NCHWC 那条路不会立刻变红** ——
   > 它在双平台回归里跑着，但没人比对它的融合前后输出，
   > 所以它会**继续静默改结果**。这与本文件「门禁的 `return 0` 不是挡箭牌」
   > 是同一件事：**没被断言覆盖的路径，坏了也不告诉你。**
   > 顺带一份**没法验证**的拷贝要**如实写"没验证"**：
   > `ZQCNN_to_MNN` 那个转换器要 MNN SDK 的四个头（仓库里没有），
   > 本机只能 `g++ -fsyntax-only` 单独编那个头 ——
   > **"这个头能编过"不等于"这个转换器能编过"。**
5. **每一份拷贝都要有**自己的**对照，不能只给主文件写一个**（2026-10-05 补，附录 HY）
   `SampleMergeBNCompareNCHWC` 就是给第二份拷贝补的那个对照，
   `SampleMTCNN_NCHWC4` **只断言检出张数**、不断言融合结果。
   > 判据：修完拷贝之后问一句「**这条路径的结果，现在被谁断言过？**」
   > 答不上来就还没修完 —— **补上缺陷 ≠ 覆盖住缺陷**。
   > 而且新写的判据要**做一次变异测试**：
   > 把那份拷贝退回修复前，确认它**变红**、且**只有该红的那个模型红**。
   > 变异前后两份数字不必相等（布局不同、舍入不同），
   > 但**方向与量级必须一致**；差一个数量级就要回头查是不是另有一处。
6. **「跳过」必须能区分成因，模型清单必须与磁盘对得上**（2026-10-05 补，附录 HZ）
   两个融合对照 sample 的列表只写了 17 个模型，而 `model/` 下有 **27** 个 ——
   差的那 6 个**从来没被验过**，而汇总行只体现为一个数字。
   补进去之后还有两个仍然被跳，原因是**判据自己**按位置解析 Input 行：
   `MobileNetSSD_deploy` 写的是 `H=300 W=300 C=3`，而 `sscanf` 写的是
   `"... C=%d H=%d W=%d"`。
   > **一份"模型清单"要能从磁盘枚举出来才谈得上完整** ——
   > 手写列表与目录不一致时，**差异部分永远是零覆盖，而且看不出来**。
   >
   > 汇总行里的一个"跳过 N 个"，读的人分不清是
   > **模型本来不适用**、**权重文件不在**、还是**我的解析器没覆盖它的写法**。
   > 三种成因三种修法，**在输出里必须分开写**。
   > 判据：`共 N 个：跑过 M，跳过 K` 旁边要能回答
   > **"跳过的每一个是因为什么"**；答不上来就补上再交。
7. **手算自证用例必须能区分你要防的那个错误**（2026-10-05 补，附录 IA.5）
   我给 `Scale` 的参考实现写了手算自证，选的是 `C=2, H=W=1` ——
   此时 `i/(H*W)`（compact NCHW 的正确通道下标）与 `i%C`（我写错的）
   **完全等价**，于是自证对**两种约定都通过**，形同虚设。
   于是参考里通道下标写反了，`C=3/8/13` 全部误报成"库有 bug"。
   > 判据：**每个自证用例都要能回答「它能区分哪两种约定？」**
   > 答不上来就是无效用例。
   > **一维退化形状（`H=W=1`、`C=1`、`N=1` 同时取）会让大量
   > "下标 / 步长 / 布局"类错误互相等价**，于是全绿。
   > 改成 `C=2, H=W=2`（`H*W=4 != C`）之后两种约定才给出不同答案。
   >
   > 这与「写形状/覆盖类门禁时先看一眼形状表落在哪一象限」是**同一条**：
   > 那一条讲**被测代码**的象限，这一条讲**自证用例**的象限。
8. **探针的每个信号只允许对应一个原因**（2026-10-05 补，附录 IB.5）
   越界读探针第一版用裸 `malloc` 分配，align=8 的组直接 SEGV
   （对齐载入要求 32 字节），于是"对齐不够"和"越界读"报成同一句
   「越界/崩溃」，拿到报告**没法用**。
   改用 `posix_memalign(32, C*4)`：既对齐、又**恰好 C 个 float**，红区紧贴末尾。
   > 判据：写完探针先问「**这一条红了，可能是哪几种原因？**」
   > 超过一种，就去把别的可能性消掉（或者分开成不同的组、不同的行）。
   > 配套：子进程 stderr **不要接 `/dev/null`** ——
   > 用仓库共用的 `zq_check_child.h::zq_child_silence_stderr()`，
   > 失败时 harness 会把 `ZQ_CHILD_ERR` 开头打出来（附录 CZ）。





9. **「某个层大面积对不上」时，先怀疑自己的参考**（2026-10-05 补，附录 IF.3）
   这一族里连着三次"参考实现写错、库是对的"：

   | 附录 | 参考错在哪 | 症状 |
   |---|---|---|
   | IA.4 | compact NCHW 的通道下标写成 `i % C`（应是 `i / (H*W)`） | C=3/8/13 全红，C=1 通过 |
   | IE.4 | `axis` 编号：`axis==0` 也写成 `outC=1`（`axis 0` 约的是 **N**） | 形状对不上（9 vs 72） |
   | IF.2 | 约轴的过滤加在**输入侧**而不是输出侧 | 12 组全红、差一个数量级 |

   三次的症状都**极像"库有缺陷"**：数字大、随形状变、每组都报。
   而库每一次都是对的。
   > 判据：**同一族里"一部分组合全对、另一部分全错"时，更像下标/约定问题，
   > 而不是内核整体算错。** 先怀疑自己那一族还没验过。
   >
   > 两条配套手段：
   > ① **把中间量按坐标摆出来**（IF.1）——
   > `Reduction` 只打一维序列时看不出"换了坐标还是换了算法"，
   > 按 `(h,w)` 排成 3x3 网格之后，库的值与网格**逐位相同**，一眼结案。
   > 这是本文件「报差异要报结构」的实践版：**结构也包括"这个值属于哪个坐标"**。
   > ② **让自证用例能区分你要防的那个错误**（见上面第 7 条）。
10. **写独立参考前，先确认契约写在代码的哪里 —— 不要按别的框架的层名约定写**（2026-10-05 补，附录 IG.2）
    本仓库的 `Tile` 沿各轴是**整块首尾相接** tile_* 份：
    `out[c] = in[c % C]`；而 **TensorFlow 的 `Tile` 在通道轴是 repeat-interleave**：
    `out[c] = in[c / tile_c]`。我第一版按 TF 的约定写参考，6 组里 4 组"对不上"，
    又差点写成"库算错了"。
    > 与第 9 条同源，但**更隐蔽**：参考不是**写错了**，
    > 而是**按另一个框架的约定写的** —— 它在你**自己重推一遍**的时候完全合理。
    >
    > 判据：**层名相同、语义不同的情形比"写错下标"更难发现。**
    > 写参考之前问一句「**这个层名在本仓库里的语义，是代码里哪几行定义的？**」
    > 而不是「这个层在别的框架里通常是什么意思」。
    > 确认的办法就是把**输入与输出按坐标并排打出来**（IF.1 / IG.2 各一次），
    > 让"它是首尾相接还是交错重复"变成**一眼可见的事实**。
11. **"覆盖了哪些"的集合要从**源码**推出来，不要手写名单**（2026-10-05 补，附录 II.2）
    手写的覆盖名单与实际内容不一致时，**差异部分永远是假覆盖**，
    而且外观与"真的覆盖了"一模一样（附录 HZ 那份 17 个模型的清单漏了 10 个，
    就是同一种错误）。
    > 可推的就推：`reachability_probe.py` 的 `PROBED` 一档是用一条正则从
    > `SampleUnusedLayerProbe.cpp` 里抓 `"<层类型> name=..."` 得到的，
    > 探针加一层，表自动跟着变。
    >
    > 配套：**依赖源不存在时要显式告警**，而不是静默退化成"什么都没覆盖" ——
    > 这里的做法是返回 `None`、打印告警、并让基线 diff 变红。
    > **一个"覆盖表"比它描述的事实更乐观，比更悲观安全得多。**
12. **参数的**键写法**与**类型**也是契约的一部分**（2026-10-05 补，附录 IJ.2）
    `PriorBox_MXNET` 上我连栽两次，两次的症状（"形状对不上"、"值差一点"）
    都**很像"映射写错了"**，而真实原因只是：
    * `size=30 59.1` —— `59.1` 是**裸 token**，`ReadParam` 报
      `unknown para 59.1` 并**只**收下 `30`。多值要写成 `size=30 size=59.1`。
    * `step_w=0.1` —— `step*` 是用 **`atoi`** 读的，所以被读成 `0`，
      从而走"由层尺寸推"那一支。多值要写成整数或 `0`。
    > 与「参数名不是语义」同族，但更隐蔽：**类型与写法只写在 `ReadParam` 的那几行里**，
    > 而读参数时**没人会去看那几行**。
    >
    > 判据：写合成用例时，**每个参数的"值怎么写"都要从 `ReadParam` 确认一遍**，
    > 尤其是「多值怎么分隔」和「这个键是 int 还是 float」。
    > 症状识别：**只有"本来就写了默认值/0"的那一组精确通过**，
    > 是"类型约定错了"的强信号 —— 映射写错的话不会挑出这种分组。
13. **「某处没出现」只能支持「我还没看到」，不能支持「它不存在」**（2026-10-05 补，附录 IO.3）
    我读了 `PriorBoxText` 里 ratio 循环的前半段，没看到 `flip`，
    就写下"`flip` 被解析并存进成员，但这个生成器一次都没用它"。
    装置一扫出来：**每个 ratio 发 4 个框**（而不是 2 个），
    多出来的那一对正是翻转框的形状 —— `flip` **大概率是起作用的**。
    > 与「参数名不是语义」不同：那条讲**名字骗人**，这条讲
    > **"我读的那几行里没出现"被我当成了"全局没出现"**。
    >
    > 判据：**判断"某个参数/分支/字段是否被使用"，必须覆盖它的整个作用域**
    > （整个函数、整个循环、所有 early-return 路径），
    > 或者干脆用装置**测**出来。读一半就下结论，在循环体里尤其危险 ——
    > 同一个 `for` 的后半段正好是它在用的地方。
14. **装置的旋钮要一项一项扫，公式才会从"差一点"变成"完全对上"**（2026-10-05 补，附录 IP.2）
    `PriorBoxText` 的 prior 个数我先猜成 `2m(1+M+r)`，实测每一行都差 `2mr` ——
    差得很规整，于是**不是公式错，是装置里有个旋钮一直拧在 1**：
    参数里写着 `flip=1`，而公式是 `flip=0` 那一档的。
    把 `flip` 做成装置的旋钮、跑两档之后，两个公式**各自完整对上 12 行**。
    > 判据：**"每一行都差同一个量"是装置配错的强信号**，
    > 因为随机错误不会这么规整。看到这种形态，先去翻装置的**参数**，
    > 不要先怀疑被测对象。
    >
    > 顺带：这也说明**参考实现与装置的旋钮必须一一对应** ——
    > 参考里少读一个参数，症状是"差一点点"，而不是"明显不对"。
15. **读 `Forward`/`ReadParam` 的前半段，就得不出"别的分支会怎么处理"**（2026-10-05 补，附录 IS.2）
    同一个 session 里被装置打掉三次：
      * IO.3  "`flip` 被解析但没用"      -> 实际每个 ratio 翻转让个数翻倍
      * IP.4  "`num_valid_ratios` 只算 >1" -> 实际是 `r == 1` 不算、且 `r` 与 `1/r` 同组
      * IS.2  "负 min_size 写成 (-x)*img_w" -> 实际**直接拒载**，报 must be positive
    三次的共同点：**我读的是函数的前半段，而结论依赖的是另一条分支**。
    > 与「某处没出现只支持我还没看到」（第 13 条）是同一族，但更具体：
    > **分支型的语义，只能靠把每个分支都跑一遍来定。**
    >
    > 判据：判断"某个参数在某条路径上会怎样"时，
    > **先把那个参数能走到的每条分支都列出来**（早退、跳过、拒绝…），
    > 再决定用装置覆盖哪几条 —— 而不是读到我关心的那段就下结论。
16. **一次实验"没让数字变好"不能否定"假设本身是错的"**（2026-10-05 补，附录 IT.2）
    我试过"max 框的 `sqrt(min*max)` 用原始 float 而不是 `(int)` 之后的那份"，
    整体后向误差只从 0.06326 变到 0.06323 —— 于是记成"那个假设也被打掉"。
    后来把 `60.7` / `59.1` 放进**单格探针**（让每一项单独放大到可辨认），
    读出来是 `sqrt(30×60)/2 = 21.2132` 与 `sqrt(59×111)/2 = 40.4629`
    —— 也就是说**我那次改动方向本来就是错的**，而它"看起来没造成变化"。
    > 与本文件「'改了没变化'不等于假设被推翻」是**同一条的两面**：
    > 那一条讲"不能因为 A 不够就否定 A"；这一条讲
    > **"混在整体误差里的变化"看不见单个假设的对错**。
    >
    > 判据：**要否定一个假设，得看它单独作用时的读数** ——
    > 把那一项在一个能把其他项都固定住的配置里单独放大，
    > 而不是看它混在总体误差里的那一点点变化。
17. **"把同一件事在一段代码里做齐"是一次替换，不是一次决定**（2026-10-05 补，附录 IV.2）
    我给参考实现加 `clip` 时用脚本批量替换，**只匹配上了 `size` 那一支的缩进**，
    `max` / `ratio` / 翻转三个分支仍然绕过 clip ——
    而 `size` 的半宽恰好小到"clip 前后完全一样"，于是**三个漏掉的分支藏了好几轮**。
    > 判据：**批量改完按"应当改动的地方数"核对**，而不是看"有没有报错"。
    > 这里应当是 `grep -c 'pbt_emit'` == 分支数（4），我当时只看了编译通过。
    >
    > 推论：**改动要能被计数**。凡是"这几处都一样"的重构，
    > 改完立刻数一遍出现次数 —— 少一处就是漏一处，
    > 而"少的那一处"往往正好是**最不容易被现有用例覆盖**的那一支。
18. **「每个假设都只解释一部分」先怀疑判别形状落在退化形状上**（2026-10-05 补，附录 IW.9）
    DeConvolution 的权重布局，我猜了两轮：
    OC-major `[oc][kh][kw][ic]` 只解释得通 k=1x1 那一组，
    C-major `[ic][kh][kw][oc]` 只解释得通 C=1 那一组，两轮都"各对一半"。
    真布局是**第三种**：文件里是 `(num_output, in_channels, kH, kW)`
    （Caffe / MXNet 的权重 blob），`ConvertFromCompactNCHW` 才把它摆成张量的
    `[oc][kh][kw][ic]`；三者**长度都一样**，所以加载与长度检查全都正常。
    回头看，"只对一半"的那两组恰恰是两个**退化形状**：
    `C==1` 时 `[oc][1][kh][kw]` 与 `[oc][kh][kw][1]` 同序，
    `kH==kW==1` 时 `[oc][ic][1][1]` 与 `[oc][1][1][ic]` 同序 —— 怎么挑都过。
    > 判据：**当两个（或三个）假设各自只解释一部分时，
    > 先算一遍"哪些形状能让候选 A、B 变得不可区分"**，
    > 确认自己的判别形状不在那个集合里，再去猜第三种可能。
    > 换句话说：**"每个假设都解释一部分"更可能是"我的尺子坏了"，
    > 而不是"世界比我以为的复杂"。**
    >
    > 推论（比本条更常用）：**挑判别形状时，先自己证明这一组能把候选区分开** ——
    > 随手挑一个形状，然后看它能不能把候选答案算出两个不同的数。
    > 算出来一样 = 这一组白挑，比没挑还糟（它会给出虚假的信心）。
19. **把作用域一层层剥掉：内核 → 层 → 文件**（2026-10-05 补，附录 IW.6/IW.7）
    定位"层和参考对不上"时，我在同一次运行里同时做了：权重逐个置 1 的映射反解、
    整幅随机权重的逐元素比对、零输入比对、全 1 比对。
    结果"零输入也有输出" —— **但那条统计打在 `run_layer` 调用之前**，
    读到的是上一趟的缓冲，于是"存在与输入无关的加项"这个结论完全是假的。
    改成"先把零输入那一趟放在所有别的调用之前"，它就是干净的 0。
    > 判据：**每一项读数都要能说出它是哪一次运行的**。
    > 一旦某个数字来自"上一轮留下的东西"，整条推理链都要重来。
    > 结构性做法：把想比的量**在同一次运行里算完再打印**，别跨调用攒。
    >
    > 有效的分层顺序（本轮实测管用）：
    > 1. **内核级**：自己摆张量、直接调内核，比明文参考 —— 排除内核；
    > 2. **层 vs 内核**：用层**实际下发的**那组 step 手工摆一份再直接调内核，
    >    与层的输出比 —— 排除"层传错了"；
    > 3. **文件**：剩下只可能是"文件里的字节顺序与我以为的不一样"。
    > 第 2 步是关键：**同样数据、同样 step，两条路径必须给出同一个数**。
20. **恒红的项要么定性、要么删掉，"待查"不是垃圾桶**（2026-10-05 补，附录 IW.12）
    `SampleUnusedLayerProbe` 的汇总里长期挂着一条
    `PriorBoxText mn=-24 **待查**（合成网加载失败）`。
    根子是把**"加载失败"**和**"数值对不上"**记成了同一条：
    实际上 `_setup()` 里明确写着 `if (min_sizes[i] <= 0) ... must be positive; return false;`
    —— 这是**契约如此**，属于"该拒就拒"。
    > 判据：一条检查项长期红着，会把**整栏红**这件事本身变成常态，
    > 于是后来的人不再看它 —— **恒红项是门禁里最贵的一种**（它把整个门禁的信号清零）。
    > 每轮收尾时问一遍："这些待查，有几条是**契约如此**？"
    > 是契约的改成断言"它确实这样"，不是契约的要么查，要么删。
    >
    > 推论：**"只报不判失败"只适用于"根因未定位且需要继续盯"的情况**，
    > 一旦定性了，就要从"待查"挪到"断言通过/断言失败"，否则它就白挂了。
21. **诊断打印要打**最差的下标**，不是第 0 个**（2026-10-05 补，附录 IW.11）
    待查那一栏原来打的是 `fabs(got[0]-want[0])`，还标成"最大相对偏差"。
    真正出问题的是最差下标 82（3.637742 vs 0.321523），而第 0 个几乎一样 ——
    **每次看这行都以为没问题**，白查一轮。
    > 判据：**报告差异的那一行，必须能让人直接跳到出问题的那一格**。
    > 已经算出 `worst_i` 了还打 `got[0]`，等于把工具白写。
22. **绕过库的对齐层时，"未对齐 SIMD 访问"只有 UBSan 看得见**（2026-10-05 补，附录 IW.13）
    新写的内核门禁自己用 `std::vector<float>` 摆缓冲喂 `_mm256_load_ps` 入口。
    ASan+LSan **全绿**（ASan 不管对齐），UBSan 那一轴一跑就报
    `zq_cnn_deconvolution_32f_align_c_raw.h:129 misaligned address`。
    库自己的张量由 `ZQ_CNN_Tensor4D_*_Align*` 保证对齐，所以这不是库的缺陷，
    但它说明一件事：**门禁里"没有越界读"靠 ASan，"没有未对齐访问"只有 UBSan 看得见**，
    两条轴缺一条就会漏掉一整类问题。
    > 判据：任何**自己造缓冲**去喂 SIMD 内核的门禁，
    > 缓冲都要显式对齐到 32/64 字节，并在 **ASan 与 UBSan 两条轴上都跑**。
23. **阴性结论也要落盘；扫描器被自己的输出绊倒时，先修扫描器**（2026-10-05 补，附录 IX.10 / IX.11）
    这一轮把「一个副本有守卫、孪生副本没有」这个形状挖到底，中途栽了三次，
    三次**都是扫描器自己的问题**，不是被测代码的问题：
      * 门禁的说明文字里写着 `` `*buffer = _aligned_malloc(...)` ``，
        匹配前没剥行注释 -> 门禁把自己的注释当成一处分配点报出来；
      * 判空的作用域划在 `else { }` 那一层花括号里 ->
        `if (c) x = p; else { x = _aligned_malloc(...); } if (c) y = q; else {...}`
        这种写法里判空写在**外层**块末行，于是正确的写法被报成漏的；
      * 比对 NCHW / NCHWC 两族分派守卫时，把「`return false;` 前面那一行」当条件，
        而 NCHWC 那份里 `return false;` 前面多一个**空行**（被格式化工具拆过），
        于是 4 个函数被报成「NCHWC 没有守卫」，打开一看两边逐字相同。
    而误报的代价是**门禁恒红** —— 按第 20 条，恒红等于没有门禁。
    > 判据：**一个自动判据刚写出来时，它自己的误报比它抓到的缺陷更值得先处理**。
    > 每加一条规则，立刻用**三种**输入验它：合格的、故意不合格的、
    > 以及一份**真实代码**（真实的格式最容易被规则误伤）。
    >
    > 推论：扫完一遍「没发现任何问题」时，**把这个阴性结论写进审计报告**，
    > 连同"差的在哪、为什么不是缺陷"一起写。否则下一个人会把同样的扫描重做一遍，
    > 而且**很可能得出同样的错误结论**（因为当时的扫描器确实是错的，
    > 而当时没人发现）。

24. **观测手段不能破坏被测环境** —— 给子进程设 `RLIMIT_AS` / `RLIMIT_DATA` 这类**地址空间或内存上限**，
    与 AddressSanitizer **不兼容**：ASan 启动时预留了 128TB 影子地址空间，且它自己后续每一次
    `mmap` 都要过这个上限。实测把上限设成 1/8/16/24/40/64/96/128/200 GB，子进程一律只剩
    一行 `ERROR: Failed to mmap`，门禁报「没跑完」——看上去像被测代码崩了，其实是观测手段
    先把被测环境弄坏了。要限制大额分配，改量 **RSS 增量**（`getrusage` 前后 `ru_maxrss` 之差），
    或者干脆把用例规模调到「修好时分配 0、修坏时分配几百 MB」这个可判的量级。
    —— 附录 CA.3 的第 6 次。
25. **门禁的桩/替身必须落在被测代码真实使用的路径上**。给 OpenCV 写桩时，桩 `imread` 造的文件名
    必须和被测头**自己拼出来**的路径逐字符一致（`folder + "/" + name + "/" + name + "_%04i.jpg"`）。
    我第一版图省事写成 `img_%03d.jpg` 放在根目录下，结果「一张都读不到」，
    于是「正常路径」用例实际跑的是「全失败路径」，门禁照样红/绿，结论完全是假的。
    —— 附录 CA.3 的第 7 次：**先证明观测手段走到了那条路径，再看结论。**
26. **门禁红了 ≠ 代码坏了；门禁绿了 ≠ 门禁扫到了东西。** 两个信号本身都需要证据。
    本会话已经因为「观测手段自己制造/销毁了信号」栽了 **7 次**，逐条记在这里：
    1) 桩造的文件路径与被测代码拼的不一致 -> 假阴性（第 25 条）；
    2) RLIMIT_AS 弄坏 ASan -> 假阴性（第 24 条）；
    3) 门禁的 `DEFAULT_FILES` 里**根本没有**被扫的文件 -> 全程一条规则都没执行；
    4) 判据用了「往后看 N 个字符」的窗口，被注释长度撑破 -> 假阴性（一天内两次）；
    5) 判据正则里**漏了 re.M**、或写了源码里从来不存在的那种形态 -> **恒假** = 等于没判；
    6) 判据按 basename 分流，而两份拷贝**同名不同路径** -> 互相串台；
    7) `gcc ... 2>&1 | head -N` —— 告警一多 `head` 先退出，**gcc 收 SIGPIPE 以 141 退出**，
       `${PIPESTATUS[0]}` 拿到的正是 141，于是「告警很多的头」被误报成「编不过」
       （实测把两个本来自包含的头报成了 FAIL）。
    **对策**：① 每条新判据都配一条**变异测试**（故意破坏它，门禁必须变红并点名）；
    ② 门禁自测里必须有**阴性对照**（不该报的形态）；
    ③ 取编译/测试的 rc 时**先把输出落盘再截断**，不要用管道取 PIPESTATUS；
    ④ 判据尽量认「结构性标志」而不是变量名/固定窗口；
    ⑤ 门禁报「全过」时，**确认它列出的文件数与预期一致**。
27. **改一个被多个 sample include 的头，要编全量而不是抽查。**
    v66 的 D1 组抓到我自己用 `re.subn(lambda m: NEW, ...)` 引入的 13 行字面量 ``：
    lambda 形式下反向引用 `` **不会展开**。当时我只编了 MTCNN 相关的几个 target，
    而 `SamplesZQlibFaceID` 也 include 同一个头 —— 是全量构建把它抓出来的。
    抽查能覆盖「我以为改的地方」，覆盖不到「被这个头牵连的别处」。

28. **测试数据里如果有"两个本该不同的量恒相等"，那一维就是盲区**（2026-10-06 补，附录 IU.1/IU.2）
    这一批两条缺陷都藏在同一个形状里：被测代码把 `imageStep` 写成了 `sliceStep`，
    而**任何一种常见测试数据都让这两个值相等**：
    * `N == 1` 时图像循环只跑一圈，推进量乘 0，两种写法等价；
    * `C` 是 align 的整数倍时，`dst_imStep = ceil(dst_C/align) * dst_sliceStep`
      里的 `ceil` 恒为 1，两个 step **数值相同**。
    两层遮蔽同时成立 ⇒ 门禁全绿、sample 全绿、sanitizer 全绿（因为
    `sliceStep <= imageStep`，写错的地址仍在缓冲区里，**不越界、不崩**），
    而第 2 张及以后的图整片算错。
    > 判据：**给一个含"步进/偏移/指针推进"的门禁挑形状时，
    > 先把被测代码里那几对"可能相等也可能不相等的量"列出来**，
    > 然后逐对确认用例把它们拆开了。表格式最省事：
    > `| 维度 | 用例取值 | 被测代码里哪两个量因此分家 |`。
    > 空着的行就是零覆盖。
    >
    > 推论：这一族缺陷**只能靠形状覆盖发现**，静态分析器看不见
    > （类型完全合法、只是取值不对），sanitizer 也看不见（不越界）。
    > 所以它必须配一条**源码门禁**，不能只靠跑测试。
29. **正则的结束符必须覆盖这类语句的**真实**收尾字符**（2026-10-06 补，附录 IU）
    `tools/check_imstep_guard.py` 第一版把匹配写成 `\b(\w+_im_ptr)\s*\+=\s*([^;]+);`，
    而这些语句绝大多数出现在 **`for` 的第三个子句**里、以 `)` 收尾 ——
    于是一条都没匹配上，门禁**永远绿**。
    是靠「把两个真实站点改回错误写法、要求门禁变红」当场打出来的。
    > 这是本文件第 26 条清单之外的**第八种**"门禁绿了 ≠ 扫到了东西"，
    > 形态是**结束符写窄了**。
    >
    > 判据：写"扫某一族语句"的正则时，**先 `grep` 出三行真实样本**
    > （一行正常的、一行跨行的、一行收尾字符不一样的），
    > 再决定结束符怎么写；并在 `--selfcheck` 里放一条**从真实文件抄来的**正例。
30. **一次只改一个变量，且改完立刻确认门禁"还能红"**（2026-10-06 补，附录 IU）
    做 IU 的变异测试时第一版脚本抛了异常，异常点在 **`finally` 之前** ——
    于是**被变异的源文件留在磁盘上没还原**，紧接着下一次跑就在读一份
    半改的树。如果当时没注意，会把"我刚才的变异"写成"代码本来就有的问题"。
    > 判据：**变异测试脚本里的还原必须放在 `finally` 里**，
    > 哪怕只是 `try/finally` + 一次写回。
    > 跑完再 `grep -c` 一次真实站点确认形态已回到正确写法，才算结束。
31. **回归跑着的时候，连"新增文件"也不能碰 —— 先确认回归入口有没有 glob**（2026-10-06 补）
    本条是第 8 条（不要在回归跑着的时候改生产文件）的补充，而踩的是同一个坑的**另一半**：
    我在 v68 正在跑 `run_zqlib_checks.py` 的时候新建了
    `tools/zq_nchwc_batch_check.cpp`，而那个脚本第 921 行是
    `sorted(glob.glob(os.path.join(HERE, 'zq_*_check.cpp')))` ——
    **新文件立刻被当成一道新的检查项**，而它还没登记 EXTRA_SOURCES，
    于是链接失败、v68 报一道"我没听说过"的检查项挂了。
    症状与"我引入了一个编译错误"几乎一样，而真实原因在**文件名**上。
    > 判据：**动仓库里任何文件之前，先问「正在跑的那个进程会不会 glob 到它」。**
    > 常见的自动发现入口：文件名的 glob、目录的 glob、
    > CMake 的 `file(GLOB)`、测试框架的自动用例发现。
    >
    > 临时手段：把文件先起一个**不会被 glob 到**的名字（`tools/_probe_*.cpp`），
    > 跑通、登记进 EXTRA_SOURCES 之后再改成正式名。
    > 别在回归跑着的时候改那个入口脚本本身 —— Python 已经把整个文件读进内存了，
    > 于是"改了没生效"和"改了生效了"两种情况会同时出现。
32. **不要同时跑两个 MSBuild**（2026-10-06 补）—— 它们共享同一份 intermediates，
    会互相把 `.obj` 锁住，报出来的 14 条全是
    `fatal error C1083: Permission denied ... .obj` / `LNK1104: 无法打开文件 ...obj`，
    **一条都不指向源码**，而汇总行里"最后几行成功链接"看着像构建基本没问题。
    我自己踩的：为了省时间同时起了两个 `cmake --build build_x64`，后来又杀掉其中一个，
    剩下的那个立刻在 14 个 target 上报 Permission denied。
    > **判据：看到成片的 `C1083` / `LNK1104` + `Permission denied` / `无法打开文件`，
    > 第一反应是"有别的构建在跑"，不是"代码坏了"。**
    > 先确认没有第二个构建进程，再重跑一次 —— 一次只改一个变量。
    >
    > 与本文件第 8 条同源：**观测手段自己制造出来的失败，会长得极像代码缺陷。**
    > 而这次更隐蔽的地方在于**汇总行**：末尾几行是 `xxx.exe` 成功链接，
    > 看上去"大部分是好的"，实际那 14 个 target 的 `.obj` 是坏的。
33. **改一个 public static 函数的签名之前，先找出仓库里所有"手抄该签名"的地方**（2026-10-06 补）
    附录 DF 给 NCHWC 的六份池化加了 4 个 pad 形参、并把 `input` 从 `const&` 改成 `&`，
    于是 v70 的 C6 组**同时打掉 12 道门禁**：

        tools/zq_net_fwd_tripwires.h:233:6: error: no declaration matches
        'void ZQ_CNN_Forward_SSEUtils_NCHWC::AVGPooling(const ZQ_CNN_Tensor4D_NCHWC1&,
         ZQ_CNN_Tensor4D_NCHWC1&, int, int, int, int, bool)'

    原因：`tools/zq_net_fwd_tripwires.h` 里有 6 个**打桩定义**（AVG/Max × NCHWC1/4/8），
    它们是**逐字抄**真实签名的，连 `const&` 都抄了。
    一处手抄的签名 = 12 道门禁同时 BUILD FAIL，而报错指向的是那个头文件、
    **不指向你刚改的那个 .h**。
    > 与本文件「手写 extern "C" 声明时照抄头文件、带参数名」同源，
    > 只是这里连**引用/非引用**都抄了。
    >
    > 判据：改签名前
    > ```sh
    > grep -rl "函数名" --include=*.h --include=*.cpp .
    > ```
    > 把每一处**手抄的定义/声明**都找出来一并改。
    > **报错信息里出现"no declaration matches"且指向一个你没改过的文件时，
    > 第一反应是"有人抄了我的签名"**，而不是"那个文件坏了"。
    >
    > 顺带：这类桩文件是**跨门禁共享**的，它一坏就同时坏一片 ——
    > 所以它不在任何单道门禁的覆盖范围内，只能靠全量构建抓到。
34. **错误/说明消息的措辞会撞上门禁的启发式**（2026-10-06 补，一天内两次）
    * 附录 DJ：`tools/run_sample_regression.sh` 的 `STUB_RE='only support|not support|...'`
      把 `SampleMergeBNCompareNCHWC` 的一次**真加载失败**判成了平台桩
      —— 因为我那句拒载消息里写了 `does not support para`。
      STUB 是**不判失败**的，于是一道真失败被门禁藏了起来。
    * 附录 DN：`check_div_guard.py` 把任何 `/<标识符>` 当成"除以模型参数"，
      而我的消息里写了 `symmetric pad / pad_H / pad_W` —— 两处被报成"待查"。

    两次的共同点：**消息字符串被当成了代码**。
    > 判据：新增/修改任何 `printf` / `std::cout` 的**错误信息**之后，
    > 问一句「这段文字里有没有会被现有门禁当代码的形状」——
    > 至少包括 `/xxx`、`not support`、`unknown`、`duplicate` 这几类。
    > 全仓门禁（`python tools/run_audit_checks.py --quick`）是最省事的验证。
    >
    > 两次的修法都是**改措辞**而不是**放宽判据**：
    > 放宽判据会让门禁变弱，而"措辞撞上"是**低频、需要时立刻能改**的事。
    > 真要让扫描器跳过字符串字面量是更好的长期做法，
    > 但那要先给它配**阳性对照**（往字符串里塞一个真除法，看它还报不报），
    > 不能只为了让它变绿就改。
35. **"变异之后门禁还是绿的"要先确认变异生效，再讨论门禁**（2026-10-06 补，附录 DO）
    这一天里这个坑出现三次，三次形态不同：
    * 一次只把三行里的**一行**换成旧值 —— 变成"一半新一半旧"的状态，
      而那个状态恰好也全对，于是我据此把**修好的代码撤了**。
      真正的修前状态要**三行一起**退回才测得出来（72/432 对 -> 432/432）。
    * 一次只加了一句 `(void)keys;` 而函数照样 `return true`。
    * 一次门禁压根没经过**缺陷发生的那一层**（DF 的第一版门禁直接调前向函数，
      而缺陷在层里没往下传 pad）。

    > 判据：**数替换处数**。变异脚本应当断言
    > "目标形态出现 N 处"且 N 与预期一致，然后才跑门禁。
    > 更快的一招：**先跑一遍变异版、把输出和基线版并排看** ——
    > 两边一模一样就是变异没生效，不必再猜。
    >
    > 推论：**"没有差异"这个观测结果，本身也需要被验证**。
    > 它和"判据没鉴别力"在输出上一模一样。
36. **自测用例本身可能和被测对象共错 —— 判据必须落在真实文件上的变异**（2026-10-06 补，附录 IJ）
    这是第 35 条的第四次复现，而且这次踩的是**自测**。
    `probe_platform_divergence.py` 的 12 条自测用例全绿、真实树 0 命中，
    但在真实文件里植入一个未圈住的 `fopen_s`，**门禁照样绿**。

    原因：那条"include guard + win32"的自测用例里**同时含一个 win32 层**，
    正好把它本该抓的 bug 盖住了 —— 用例和被测代码共享同一个错误前提，
    于是永远一起对。

    > 判据：**在真实文件里做变异，数命中数**，而不是只看自测是否全绿。
    > 自测能挡住**崩溃**，挡不住**共错的判据**。
    > 写自测用例时，刻意让"必须报错"的那条路径**只经过被判定的机制**，
    > 不要顺手把别的东西也放进去当上下文。

    顺带一条同源的：逐行做源码扫描时，**跨行的块注释**必须单独处理。
    逐行 `strip` 只能消掉"开闭在同一行"的 `/* */`，
    于是 20 行注释块里的 6 处 `sprintf_s`/`fopen_s` 全被当成真代码报出来。
    消注释时要按行替换成**等长空格**，否则行号全错、报错定位会指到别的文件。

37. **一条自称"这里是坏的"的注释能长期错，往往是因为验证它的测试没跑**
    （2026-10-06 补，附录 IK）
    `tools/zq_gemm_shape_check.cpp` 里写着"崩溃/结果错的形状**现在就是坏的**，
    所以这个测试当前应当是红的"，而附录 BP 早就修好了，实测 PASS。
    它错这么久不是因为没人写，而是那个测试挂在 `SLOW` 里、**默认不跑**，
    而且全仓没有任何地方传过 `--with-slow`。

    > 判据：注释里凡是**断言当前状态**的（现在会崩 / 现在是红的 / 在 SKIP 里 /
    > 还没修），要么指向一个**会跑的**门禁，要么删掉。
    > "改之前是什么样"写进审计报告，别留在源码里当现状描述。
    > 顺带：排除测试的理由里那些**具体数字**要定期复核 ——
    > 这一条写的是">5 分钟"，实测 150 秒，差了两个数量级。

38. **门禁"存在"不等于门禁"会跑"**（2026-10-06 补，附录 IK）
    6 个 GEMM 调度调用点的测试一直都在仓库里，只是没人调用它们。
    加测试很容易，把它接进**统一入口**、并且**确认不需要额外开关就会跑**，同样是活。

    > 判据：新加测试时，除了能在 `run_audit_checks.py` 里 grep 到，
    > 还要**不带任何可选开关**跑一遍，确认它确实在默认集合里。

39. **说"某个测试编译慢"之前，先看它是不是在重复编译同一个 TU**（2026-10-06 补，附录 IK）
    剩下 7 个慢测试各自把 `zq_gemm_32f_align_c.c` 编到**各自的文件名**
    （命令、头路径、sanitizer 档位全同，只有输出名不同），整轮里同一个 TU
    被编 6 次以上 —— 这才是"慢"的真因，不是文件大。

    > 判据：仓库里已有正确做法 —— `zq_nchwc_conv` / `zq_nchwc_conv8`
    > 是"把 .o 换个名字落一份"来共享的。推广即可，别急着加缓存。
    > 推广时的判据见第 40 条 —— 推广本身差点把整轮门禁打挂。

40. **共享/复用产物时，判据是「整条命令逐字相同」，不是「同一个源文件」**
    （2026-10-06 补，附录 IL）
    把 EXTRA_SOURCES 里重复编译收敛成"编一次、其余 cp"时，第一版按**源文件名**
    共享。而 `zq_gemm_shape` 那条编译命令里**没有 `$SAN`**、别家有 ——
    于是把一份**没插桩**的 `.o` cp 给了 sanitizer 门禁：它们照样全绿，
    但**越界读一个元素也没人报**，正是 2026-10-02 栽过的那一跤。

    > 判据：键取"**抹掉输出名之后的整条命令**"，旗标自然进键。
    > 改完立刻确认：同一个 TU 应该产出**几个**对象？（插桩/不插桩 = 2 个，
    > 不是 1 个）

    配套一条：**共享只允许新增一份产物，绝不能动任何人已有的那个文件名。**
    第二版把"第一次那条"的输出**改名**成规范名，于是第一个用到它的门禁链接时
    找不到自己的 `.o`，整轮 **44/71、27 个 BUILD FAIL**，报错全在链接期 ——
    就是 DY.8 那个"错误出现在错误的地方"。

    > 改共享/去重这类**批量改动**时，先做一个**静态验证**再跑完整回归：
    > 分别生成改动前后两版脚本，比对它们**产出的文件名集合**，
    > 确认**丢失 0 个**。这一步几秒钟，而它比"跑一遍看看"早发现问题 ——
    > 那 27 个 BUILD FAIL 是一次 15 分钟的回归才发现的。

41. **改动 batch 构建脚本时，"跑一遍"之前先看生成物**（2026-10-06 补，附录 IL）
    `run_zqlib_checks.py` 是把 shell 脚本**生成出来**再交给 WSL 跑的。
    所以可以在**完全不跑 WSL** 的前提下断言生成物的形状：

    ```python
    cap = []
    m.run_wsl = lambda s, *a, **k: (cap.append(s), '')[1]   # 截住脚本，不执行
    ```

    > 判据：改这类脚本先截获生成物做静态断言（文件名集合、命令条数、
    > 旗标出现位置），**再**花时间跑真实回归。

42. **给一个"专治某类缺陷"的门禁补上插桩，先用变异量一量它到底买到什么**
    （2026-10-06 补，附录 IM）
    `zq_gemm_shape` 是 GEMM 的 (M,N,K) 形状安全图，测的正是越界读，
    而它的 `EXTRA_SOURCES` 里**没有 `$SAN`** —— 被测内核从来没被插桩。

    我当时的判断是"不插桩就等于这个测试对它要抓的那类缺陷是瞎的"。
    **实测把这个判断推翻了**：在 fallback 里植入一处越界读（`k < K` 改成
    `k <= K`）之后，**不插桩的那版也报出来了**（997 崩溃 + 927 结果错），
    只是靠垃圾值、时灵时不灵。插桩版是 1083 崩溃 + **0 结果错** ——
    第一个越界就 abort，确定性地全报出来。

    > 判据：买到的可能是**确定性**而不是"从看不见到看得见"。
    > 这两者价值不同，写报告时**不要混为一谈** —— 我就是先把它们混着说了，
    > 写进注释里才发现不对。**结论要按实测写，不按预期写。**
    >
    > 附带一条：去重（附录 IL）省掉的是**编译**，省不掉**运行时插桩开销**。
    > "去重之后加插桩几乎免费"这句话在**单跑**时是错的（150s -> 698s），
    > 在**整轮**里才接近免费（因为那份 .o 本来就要编）。

43. **补插桩要把"范围"查全，别只修第一个撞见的**（2026-10-06 补，附录 IM）
    以为是 `zq_gemm_shape` 一个测试缺 `$SAN`，按 `EXTRA_SOURCES` 全量一查，
    是 **6 个 tag、30 条命令**都没有。这 6 个恰好就是附录 IK 里
    "GEMM 调度的全部调用点"。

    > 判据：这类性质（"这条 EXTRA_SOURCES 带不带 `$SAN`"）先**全量列一张表**，
    > 再决定改哪几条 —— 不要边查边改。表比叙述更能暴露"原来是 6 个不是 1 个"。

44. **改 `EXTRA_SOURCES` 只能按 tag 块定位，**不能**按字面量全局替换**
    （2026-10-06 补，附录 IN）
    连栽两次：先是按整条字面量替换 —— 而一个编译命令在源码上是**两行隐式
    拼接**（`'gcc ... '` 紧跟 `'src -o out.o'`），逻辑上一条、源码里不连续，
    `src.count(...)` 恒为 0；改全局替换更糟 —— 全文件 **64** 处编译行不带
    `$SAN`，要改的只有 **27** 处，全局替换会顺手改掉另外 37 处**不属于目标
    tag** 的命令。

    > 判据：同一个编译前缀（`gcc -O1 -g -mavx2 ...`）被几十个 tag 共用，
    > 这是这份文件的结构决定的。改某个 tag 就**按 `'tag': [` 的行界切块、
    > 只在块内逐行替换**，并**断言"改动行数 == 预期"**。
    > 断言不只是防手滑 —— 中途脚本自己就有个"正则没匹配上仍去取 `m.group(1)`"
    > 的 bug，是断言兜住的。

45. **同一次提交里把改完的整轮回归结果也报出来**（2026-10-06 补，附录 IN）
    插桩这类"会让 sanitizer 真正开始看被测代码"的改动，
    **最可能的结果就是冒出潜伏缺陷**。所以它必须配一次完整回归，
    而且结论要分两种写清楚：
    * 全绿 -> 这是**阴性结论**，同样要写（"ASan 下 conv/innerproduct/GEMM 内核干净"）；
    * 红了 -> 那才是本轮的主要产出，要接着查到底。

    别把"回归跑过了"当成流程走完；**跑出来的结果本身才是结论**。

46. **「全量改某个属性」之前，先去**另一个地方**确认有没有反向声明**
    （2026-10-06 补，附录 IO）
    把 `EXTRA_SOURCES` 里的编译行全量补上 `$SAN`，规则本身是对的，
    但它**有例外**，而例外写在**另一个字典**里：
    `zq_nchw_conv_free` 在 `EXTRA_CXXFLAGS`（980 行开外）带着
    `-fno-sanitize=address`（它要自己接管 `free`，与 ASan 运行时冲突，附录 CU.9）。
    于是编译行插桩、链接行不插桩 -> `collect2: error: ld returned 1 exit status`，
    全量补完第一轮是 **70/71**。

    > 判据：凡是"全量给 X 加/删 Y"，先 `grep` 一遍**仓库里所有声明 Y 的地方**。
    > 规则的正确形式往往带豁免：
    > 「每条都带 `$SAN`，**除非**该 tag 在 `EXTRA_CXXFLAGS` 里显式
    > `-fno-sanitize`」。**例外和规则必须写在同一个地方**，否则下一次全量改还会踩。

47. **BUILD FAIL 的消息为空/无用时，去截获 harness 生成的命令**（2026-10-06 补，附录 IO）
    这一条里门禁只报了 `collect2: error: ld returned 1 exit status`，
    真正有用的信息（`-fsanitize=address ... -fno-sanitize=address`）没打出来。
    **手工复现却"成功"了** —— 因为我照自己理解拼的链接行，
    而 harness 拼的那条不一样。

    > 判据：手工复现**成功**而门禁**失败**，说明你复现的不是同一个东西 ——
    > 别再猜，去看**实际发给 shell 的命令**（第 41 条的截获生成物），
    > 再一字不差地复现。"我手工跑是好的"在这种情况下是**误导性证据**。

48. **"少写一个属性不会让任何测试变红"的性质，必须单独落成门禁**（2026-10-06 补，附录 IP）
    手工补插桩花了三轮（IM 补 1 个 tag、IN 补 5 个、IO 补完剩下 18 个）。
    而 `EXTRA_SOURCES` 里少写一个 `$SAN` **不会让任何测试变红** ——
    只是"被测实现 TU 没被插桩"这件事**悄无声息**地回来了。

    > 判据：凡是"某个**配置/编译属性**决定了一类缺陷能不能被发现"，
    > 就给它配一道门禁，哪怕当前是全对。
    > 门的形状：**规则 + 豁免 + 两个方向都断言**（豁免项既要"可以不满足规则"，
    > 也要"不许偷偷满足规则"），再配真实文件上的变异测试。

49. **变异脚本自己失败时，门禁的"绿"是没有意义的**（2026-10-06 补，附录 IP）
    做 C18 的第一次变异时，我拿带 `-fopenmp` 的串去改 `zq_bns`，
    而它的编译行没有 `-fopenmp`，`str.index` 抛了 `ValueError` ——
    **变异根本没发生**，而门禁当然还是绿的。

    > 判据：变异脚本必须**自己断言"目标形态出现了 N 处"**再跑门禁，
    > 这一条又出现在附录 IO 和 IP 里，说明它不是"偶尔忘了"而是**默认要做的动作**。
    > 脚本抛异常时**先看异常**，别把"没报错"当成"变异成功"。

50. **"全量回归"必须是一条**仓库里存在的命令**，不能靠口口相传**（2026-10-06 补，附录 IQ）
    9 个 opt-in 开关各自管一整块覆盖，而"把它们全打开"的那条命令
    **从来没被写下来过** —— 于是每加一个开关就要记得往一条传口令的命令里补，
    漏了没人知道。`--with-slow` 就这么烂掉好几天。

    > 判据：加完开关就写 `--all`，并配门禁断言"每个 `store_true` 开关
    > 都已在 `OPT_IN_FLAGS` 或 `NON_OPT_IN_FLAGS` 里显式分类"。
    > **排除项也要写理由**（`--quick` 减少覆盖、`--ubsan` 换口径），
    > 不写理由的排除项下次就变成了"忘了"。

51. **同一门禁里两套命名（flag 名 / dest 名）必须先归一再比**（2026-10-06 补，附录 IQ）
    `OPT_IN_FLAGS` 存的是 argparse 的 dest（`with_build`），
    正则从 `add_argument('--with-build', ...)` 扫出来的是 flag 名。
    直接跨两套命名比，结果是 12 个开关全部误报。

    > 判据：**第一版全红时，先怀疑判据，再怀疑表。**
    > 尤其当"表"是人手写的、"判据"是刚写的时候。
    > 这次是**自测先抓住的**（6 条里错 3 条）—— 又一次证明自测不是装饰。

52. **覆盖验证要比"命令"，不能只比"组名"**（2026-10-06 补，附录 IR）
    验证 `--all` 与"逐个开关的并集"等价时，第一版只比组名 ——
    而 `--with-slow` 这种开关**不新增任何组**，只是给已有的 B 组命令追加一个
    参数，按名字看它在"开"与"不开"之间**完全一样**。于是算出
    「并集 8 / `--all` 8 / 一致」，看着通过，其实什么都没验到。

    > 判据：凡是"开关/配置改变的是**命令**而不是**组列表**"，
    > 覆盖验证必须比**命令元组**。比名字的版本在这种开关上是**恒真的**。

53. **判据站不住就砍掉，不要放宽阈值凑绿**（2026-10-06 补，附录 IR）
    加了一条"`run_group` 的名字必须逐字出现在 `failed.append` 里"，
    在真实树上报 4 处，全是**有意**的（D1/D2/D3 由父组统一记账、B2 标签写得短）。
    改成"允许前缀关系"能消掉这 4 处，但那样就打不出 `优化期`/`优化后`
    那种真打错字的情况 —— 换来的是一个看起来更严、其实更糊的判据。

    > 判据：**误报率和漏报率一起看**。一条会误报的规则不如没有 ——
    > 它训练人忽略门禁。砍掉它，并把"为什么砍"写进文件头注释。

54. **给门禁编号加唯一性门禁，别靠"我记得避开"**（2026-10-06 补，附录 IR）
    源码注释里本来写着"C4 那次撞名是我自己犯的……这里直接避开"，
    结果 C7 又撞上了。**注释挡不住编号撞车。**

    > 而且这道门禁**当场抓到了作者自己**（把 GUI 门禁改叫 C20 的下一行，
    > 就把「编号唯一」门禁也起名叫 C20）。这比"门禁有用"的论证有说服力。

55. **编号撞车时改一处会把另一处也顶掉，顺移要一次做完**（2026-10-06 补，附录 IR）
    C7->C21、C5->C22、C5->C23 是**互相依赖**的（原来 C21/C22 是空位，
    改完第一个就占掉了第二个的位置）。分多次改会中途出现新的撞名，
    每次都得重跑一遍门禁才知道有没有撞干净。

    > 判据：编号重命名**一次改完再跑门禁**，不要改一个验一次。

56. **管道后面接 `; echo $?`，拿到的是 `tail` 的退出码**（2026-10-07 补，附录 IS）
    `python run_audit_checks.py --all 2>&1 | tail -60; rc=$?` 打出了
    `ALL RC=0`，而同一份输出里明明写着 `1 CHECK GROUP(S) FAILED`。
    `$?` 是**管道最后一个命令**（`tail`）的退出码。

    > 判据：要退出码就别接管道；非要接管道，用 `set -o pipefail`，
    > 或写成 `... > log 2>&1; rc=$?; tail -60 log`。
    > **判据要看"被测程序"的退出码，不是看管道尾巴的。**

57. **"把测试提进默认通道"经常直接暴露既有缺陷**（2026-10-07 补，附录 IS）
    `zq_gemm_shape` 用 `std::vector<float>` 喂给要求 32 字节对齐的 AVX GEMM 内核，
    **一直在做 UB**，ASan 那一轴上是绿的（ASan 不查对齐）。
    附录 IK 把它从 `SLOW` 提进默认通道之后，UBSan 那一轴（C6）**第一次**跑到它，
    当场变红。

    > 判据：ASan 与 UBSan 查的是**不同的类**（ASan 查越界/释放后使用，
    > UBSan 查对齐/溢出/移位/别名）。一个测试"绿"只说明它在那**一条轴**上绿。
    > 提覆盖时优先挑"换轴会不会响"的测试。

58. **测试喂给库的数据必须满足库的契约，而契约往往只写在调用方**（2026-10-07 补，附录 IS）
    查 `_aligned_malloc(..., 32)` 才知道 GEMM 要求 32 字节对齐 ——
    库的契约**没有任何注释声明**，只能从调用方的分配方式反推。

    > 判据：写新测试/新 sample 调某个加速内核前，
    > 先 `grep` 一遍**真实调用方怎么分配缓冲**，照着分配。
    > 不要图省事用 `std::vector` —— 它只给 `alignof(max_align_t)`（x86-64 上 16）。

59. **"有没有跑过"要按**轴**问，不按"跑没跑"问**（2026-10-07 补，附录 IV）
    本仓的并行代码集中在 MTCNN（约 30 个 `#pragma omp parallel for`），
    而仓内所有 sample 的 `thread_num` 都被夹成 1 —— 这批循环
    **一次都没执行过**。但"回归跑过很多遍"这件事本身是成立的。

    > 判据：问三个问题 —— **哪一条轴**跑的？跑的是**哪条路径**？
    > 被夹成单线程的参数、被注掉的参考分支、被 `#if` 关掉的分支，
    > 都让"跑过"变成空话。附录 II 挖出六个 MTCNN 问题，靠的正是这一问。

60. **新增一条 sanitizer 轴之前，先用最小探针验"链接得起来吗"**
    （2026-10-07 补，附录 IV）
    本想加 TSan 轴，写了 4 线程写同一格的最小探针，一链接就是
    `cannot find libtsan_preinit.o` —— 这台 WSL 的 gcc-9 缺 TSan 运行时。
    **在没有装任何东西之前，"能不能跑"是一个五分钟就能证伪的前提。**

    > 判据：新开一条轴的顺序是
    > ① 最小探针链接通过 → ② 探针**确实报出**预期的 race → ③ 才接进门禁。
    > 第 ② 步不能省：能链接 ≠ 能检出（附录 IJ 那道门禁的教训反过来）。
    > 装不上的依赖**如实记成环境限制**，不要写成"已覆盖"。
