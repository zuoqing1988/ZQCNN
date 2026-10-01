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

