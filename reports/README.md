# reports

实测报告。审计类文档在仓库根的 `audit_k3_20261001.md`（未放在这里，因为它被
各轮 commit 信息和 `AGENTS.md` 大量引用）。

| 文件 | 内容 |
|---|---|
| `ZQ_GEMM_多内核自动选路_设计提案.md` | 多种 kernel + 按形状自动选路的设计提案：现有哪些变体、决策该看什么、不同机器如何靠**运行时自测**做到自适应，以及为什么本项目的自测判据必须用中位数（噪声单边）|
| `ZQCNN_上层优化手段汇总.md` | GEMM 之外的上层优化：图优化（in-place 重定向 / BN·PReLU 折叠，**含实测：MTCNN 上看不出收益**）、NCHW/NCHWC 数据布局、内存分配、算子与 SIMD 内核、并行策略，以及一节明确的**空白点**（无预取、无 cache 分块、算子级无 A/B 实测）|
| `ZQ_GEMM_汇编内核性能对比.md` | 手写汇编 GEMM 内核 vs 仓库原有 intrinsic 版 vs Intel MKL 的三方对比。64 个形状 × 多轮实测（Linux / Windows 各测），含按形状分类的汇总、逐形状明细、以及**测量方法与噪声下限**的说明。**第〇节是最新数字**（补 `-mfma` + 6×8 N 方向内核之后：Linux 99% / Windows 98% of MKL），并列了这一轮实测过但**没有采纳**的四条路线 |
| `ZQ_GEMM_多内核自动选路_设计提案.md` | 多种 kernel + 按形状自动选路的设计提案：现有哪些变体、决策该看什么、不同机器如何靠**运行时自测**做到自适应，以及为什么本项目的自测判据必须用中位数（噪声单边）。第〇节的 6×8 内核已经按这份提案落进了生产分发 |

## 复现

```bash
# Windows（需先 cmake --build build_x64 --config Release），从仓库根运行
#   以便找到 3rdparty/mkl_runtime/win/mkl_rt.3.dll
./cmake-out-win32-x64/release/Release/SampleGEMMCompare.exe

# Linux
wsl -d Ubuntu-20.04 -- bash -c "cd /mnt/d/ZQCNN && MKL_THREADING_LAYER=SEQUENTIAL <zbench>"

# 正确性（汇编 vs intrinsic，18 个针对性用例）
./cmake-out-win32-x64/release/Release/SampleGEMMAsmCompare.exe

# 多轮取中位数后汇总 asm/MKL（本文第〇节的表格就是这么来的）
python tools/gemm_mkl_ratio.py -n 7
python tools/gemm_mkl_ratio.py --exe "D:/ZQCNN/cmake-out-win32-x64/release/Release/SampleGEMMCompare.exe" -n 7

# 两个现成二进制之间的 A/B（**交替**跑、逐形状取中位、8% 噪声阈值）
MSYS_NO_PATHCONV=1 python tools/bench_two_binaries.py /tmp/a/SampleGEMMCompare /tmp/b/SampleGEMMCompare -n 5

# 改 ZQ_GEMM 源码时：在临时目录编两个版本对比（不动工作区）
python tools/bench_gemm_ab.py A.c B.c --replace zq_gemm_32f_align_c_asm.c
```

**读数字前请先看文档第五节**：本机 GEMM 读数的噪声下限约 7%（空对照测得），
小于该幅度的差异不算结论。**而且这个下限是在小/中形状上测的** —— 512³ 以上的
形状实测漂移接近 10%，别拿 7% 去判读 `1024³`/`2048³`。

改完汇编/GEMM 代码后跑这两个回归：

```bash
python tools/check_line_endings.py      # multi-CR / lone-CR / CRLF+LF 混用
python tools/check_text_encoding.py     # UTF-8 有损解码残留（U+FFFD）
```

改了 `3rdparty/include/ZQlib/` 下的头还要跑这两个（主工程的 sample 回归验不到那里）：

```bash
python tools/run_zqlib_checks.py        # 8 个 ZQlib 独立回归测试，ASan + LSan

# 可编译性门禁（慢，约 2 分钟，编译 143 个翻译单元）：
# 任何一个头从 OK 变成编不过就退出 1。修好/新增头之后更新基线。
python tools/probe_zqlib_headers.py --check-baseline tools/zqlib_probe_baseline.txt
python tools/probe_zqlib_headers.py --save-baseline tools/zqlib_probe_baseline.txt
```

## 第三方头库这一轮的结论（2026-10-02）

原来的审计结论是「`3rdparty/include/ZQlib/` 是第三方头、无法在本仓库编译验证，
改动风险大，所以不修」。**这个前提不成立**，后来被逐条推翻：

| | 数字 |
|---|---|
| ZQlib 头总数 | 143 |
| **能独立编译**（所以**能**验证） | **118** |
| 确实需要 `windows.h` / MFC / OpenCV / GL（本机测不了） | 9 |
| 缺兄弟头或依赖（taucs 等）/ 需要重写 | 16 |

起点是「能独立编译的只有 81 个」（可验证覆盖率 57%）。到本轮结束是
**118 个（82%）**，因为其中一批是**「自己源码就编不过」**的零成本修复
（缺 `typename`、缺 include、未声明的变量、调用不存在的成员、命名空间少 `ZQ_` 前缀）——
这类问题编不过就意味着这些头**从来没被编译过**，于是也从来没被发现。

在能独立编译的那批上写了 8 个 ASan + LeakSanitizer 回归测试
（`tools/zq_*_check.cpp`），**翻出 13 条真缺陷并全部修复**，详见
`audit_k3_20261001.md` 附录 W~AJ。其中影响最实际的三条：

- `ZQ_ImageProcessing.h` 的 **3×3 中值滤波结果完全不对**（`Sort_decend_3elements`
  的中间一步照抄了第一步、从未排过第 3 个元素；`MedianFilter33_1channel` 的第二列
  写进了 `col[0]`，`col[1]` **从未被赋值**，读的是未初始化的栈内存）。
  它有活的调用方 `ZQ_FindCorners.h:1562-1563`。
- `ZQ_KDTree.h` 的 `_recursive_ann_fix_radius_search` 叶节点循环**没有上限检查**，
  半径内点数超过 `k` 就写穿调用方缓冲 —— **用全合法的入参就能触发**。
- `ZQ_MergeSort.h` / `ZQ_Kmeans.h` 只靠 MSVC 的传递 include 才能编过，
  libstdc++ 下直接失败（前者缺 `<vector>`、后者缺 `<math.h>`）。

**一条比任何单条缺陷都重要的元发现**：「无法验证」是会自我实现的结论 ——
因为没法验证所以不改，不改就继续没法验证。以后再遇到「第三方头 / 死代码 /
跑不通所以不修」的说法，先花五分钟确认它到底能不能单独编译。

