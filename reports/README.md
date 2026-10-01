# reports

实测报告。审计类文档在仓库根的 `audit_k3_20261001.md`（未放在这里，因为它被
各轮 commit 信息和 `AGENTS.md` 大量引用）。

| 文件 | 内容 |
|---|---|
| `ZQ_GEMM_多内核自动选路_设计提案.md` | 多种 kernel + 按形状自动选路的设计提案：现有哪些变体、决策该看什么、不同机器如何靠**运行时自测**做到自适应，以及为什么本项目的自测判据必须用中位数（噪声单边）|
| `ZQCNN_上层优化手段汇总.md` | GEMM 之外的上层优化：图优化（in-place 重定向 / BN·PReLU 折叠，**含实测：MTCNN 上看不出收益**）、NCHW/NCHWC 数据布局、内存分配、算子与 SIMD 内核、并行策略，以及一节明确的**空白点**（无预取、无 cache 分块、算子级无 A/B 实测）|
| `ZQ_GEMM_汇编内核性能对比.md` | 手写汇编 GEMM 内核 vs 仓库原有 intrinsic 版 vs Intel MKL 的三方对比。64 个形状 × 10 轮实测（Linux 5 轮 + Windows 5 轮），含按形状分类的汇总、逐形状明细、以及**测量方法与噪声下限**的说明 |

## 复现

```bash
# Windows（需先 cmake --build build_x64 --config Release），从仓库根运行
#   以便找到 3rdparty/mkl_runtime/win/mkl_rt.3.dll
./cmake-out-win32-x64/release/Release/SampleGEMMCompare.exe

# Linux
wsl -d Ubuntu-20.04 -- bash -c "cd /mnt/d/ZQCNN && MKL_THREADING_LAYER=SEQUENTIAL <zbench>"

# 正确性（汇编 vs intrinsic，18 个针对性用例）
./cmake-out-win32-x64/release/Release/SampleGEMMAsmCompare.exe

# 两个内核版本之间的 A/B（自动交替跑、取最大、8% 噪声阈值）
python tools/bench_gemm_ab.py A.c B.c --replace zq_gemm_32f_align_c_asm.c
```

**读数字前请先看文档第五节**：本机 GEMM 读数的噪声下限约 7%（空对照测得），
小于该幅度的差异不算结论。
