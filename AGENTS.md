# AGENTS.md

> **核心规则**（每次会话必读）。全部经验细则（检查工具写法、内核/并行、
> 调试方法论、编号教训 1~92 等）已迁移至 **`AGENTS_LESSONS.md`** ——
> 排查具体问题、写门禁/内核/测试之前，先去读那边的对应小节。

---

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


---

## 编码与行尾红线（静默损坏类，提交前必查）

1. **四件套**：`python tools/check_text_encoding.py` + `check_line_endings.py`
   + `check_stmt_joins.py` + `check_gates_runnable.py`，提交前必须全绿。
2. **含转义的脚本一律用 Write 工具落盘再执行** —— shell heredoc 会吃掉
   反斜杠（`\n` 变真换行），详见经验细则「反斜杠不要放进 shell heredoc」。
3. **Edit 工具会抹掉文件头的 UTF-8 BOM**；改带 BOM 的文件（如
   `layers_c/*_raw.h`）后用 git diff 确认 BOM 还在。
4. **Python 批量改写源码必须自己保行尾、保编码**：新插入的行不带 `\r`，
   绝不走 `errors='replace'`（会把原字节永久换成 EF BF BD 且不报错）。
   改中文注释/文档后必须跑 `check_text_encoding.py`。
5. **`*_raw.h` 内核头必须是纯 LF**（`.gitattributes` 已标 `eol=lf`）：
   行尾宏被 CRLF 截断时 MSVC 能忍、gcc 静默拼接失效。改完跑
   `check_line_endings.py`。

---

## 全量回归与工具入口

- **全量回归唯一入口**：`python tools/run_audit_checks.py --all`（约 100 分钟）。
- **新改动必须纳入全量**；回归跑着期间不要改树。
- **ZQlib 独立回归**：`python tools/run_zqlib_checks.py`（必须从 Windows Python 调）。
- **报告/附录完整性**：`python tools/build_audit_index.py --check`（重号即失败）。

---

## 经验细则索引（在 AGENTS_LESSONS.md）

- 写检查类工具 / 门禁判据 →「写\"检查类工具\"的四条硬规矩」「判据本身的三条」
- 写 GEMM 内核 / SIMD →「写内核 / 写并行代码」「ZQ_GEMM 的数据布局」
- NCHW / NCHWC 步长与张量语义 →「NCHW 与 NCHWC 的步长语义」等 4 节
- 调试数值对不上 → 2026-10-04 一整批（逐层比对 / 差异形态 / 合成用例）
- 批量改写 / 孪生副本 →「批量改代码的四条硬规矩」（第 85/91/92 条编号教训）
- 编号教训 1~92：按日期排序的完整清单，在本文件末尾。

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

