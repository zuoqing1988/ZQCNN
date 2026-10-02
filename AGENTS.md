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
   - 默认：文本卫生（A1/A2/A3/A4）+ 第三方头库的 10 组 ASan 测试 + ZQlib 可编译性门禁（约 2.5 分钟）
   - `--quick`：跳过可编译性门禁（约 40 秒）
   - `--ubsan`：B 组换成 `-fsanitize=undefined` 口径
   - `--msvc-asan`：加上 Windows 侧 MSVC ASan 那 10 组
   - `--msvc-probe`：加上 MSVC `cl /Zs` 逐头语法检查
   - `--warn-sweep`：加上 gcc `-Wall -Wextra` 的 HIGH 桶门禁（慢，约 2 分钟）
   - `--with-build`：再加上**双平台全量构建 + 关键 sample 回归**（Windows cmake
     与 WSL gcc 各一遍，两边各跑 6 个 sample；sample 必须在**产物目录**里跑，
     从仓库根跑只会打一行 `empty image`，看着像跑过了其实什么都没验）
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
