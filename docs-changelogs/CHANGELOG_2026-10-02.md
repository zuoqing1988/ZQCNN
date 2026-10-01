# CHANGELOG 2026-10-02

## 新增/变更：第十一轮审计 —— 修掉自己刚加的 6x8 内核里一个静默算错的 bug

### 变更文件
- `ZQ_GEMM/math/zq_gemm_32f_align_c_asm.c`
  - 新增 `ZQA_FMA7` 宏：6x8 内核专用的 FMA 宏，无 FMA 时用 `ymm7` 而不是 `ymm15`
  - 6x8 内核的 6 条 FMA 全部改用 `ZQA_FMA7`，clobber 列表补 `xmm7`
- `audit_k3_20261001.md`：新增**附录 V**（第十一轮）
- `AGENTS.md`：行尾一节补第 5 条（只看退出码不够 + sample 要在产物目录跑）、
  第 6 条补 heredoc 变体；汇编一节补第 9 条（借寄存器当暂存是跨内核耦合）

### 问题

`ZQA_FMA` 在没有 FMA 时展开成两条指令，借 **`ymm15`** 当 `vmulps` 的暂存：

```
vmulps %ymm15, %ymm8, %ymm15     <-- 借 ymm15
vaddps %ymm15, %ymm0, %ymm0
```

而 2026-10-01 新加的 6x8 外积内核把 **B 向量放在 `ymm15`**，每个 k 循环要用它做
6 次 FMA。于是 `-mfma` 缺席时，第 1 条 FMA 的 `vmulps` 把 B 向量冲掉，后 5 条全错 ——
**不崩溃、不越界，只是结果错**。

### 实测结果

Linux 去掉 `-mfma` 编译，`SampleGEMMAsmCompare`：

```
   17    19    27 |    6.136e+00 ...   <== FAIL
   16     8    32 |    5.180e+00 ...   <== FAIL
   32    32    32 |    5.709e+00 ...   <== FAIL
  313    32    28 |    7.365e+00 ...   <== FAIL
result: FAIL (4 case(s) failed)
```

四个失败用例的 K 是 27/32/32/28，**全在 `ZQA_NDIR_MAX_K = 32` 里** ——
只有走新路径的形状会错，其余 14 个用例照常通过。这种「只错一部分」的形态比全错
更难发现。

修复：6x8 内核改用 `ZQA_FMA7`（无 FMA 时暂存用该内核空闲的 `ymm7`）。
同一套「不带 -mfma」的构建修复后 `PASS (0 case(s) failed)`，worst 3.8e-06。

### 是怎么发现的

不是靠读代码，是靠**编两次、把 sample 输出逐字节 diff**：
`tools/run_sample_regression.sh` 只看退出码，两次跑都是全 rc=0，但输出内容不一样。
顺带确认了 `-mfma` **不改变任何推理结果**：MTCNN / MTCNN_NCHWC4 / SSD /
CascadeOnet / CascadeOnet_Interface / FaceDetectorMTCNN / MTCNNLoadFromCode
七份输出在两个构建之间逐字节相同（滤掉耗时行后）。

另外发现一个一次性脚本的坑：**sample 必须在产物目录
`cmake-out-unix-x64/Release` 里跑**（CMake 把 `model/` 和 `data/` 联接到了那里），
从仓库根跑只会打一行 `empty image` —— 看着像跑过了，其实什么都没验。
`run_sample_regression.sh` 一直是对的。

### 验证
- Windows Release 全量构建 0 error，`SampleGEMMAsmCompare` PASS（worst 1.9e-06）
- Linux `C_FLAGS = -O3 -DNDEBUG -fPIC -Ofast -ffast-math -mavx2 -mfma`，
  `SampleGEMMAsmCompare` PASS（worst 3.8e-06）
- Linux 去掉 `-mfma` 的构建也 PASS（worst 3.8e-06）—— **这条分支现在有人走过了**
- 两平台 sample 回归全 rc=0
- `tools/check_line_endings.py` / `tools/check_text_encoding.py` 均 OK

### 注意事项
1. 其余 5 个微内核（m2n4 / m1n8 / m1n4 / m4n1 / m1n1）只用 ymm0-ymm13，
   `ymm15` 对它们空闲，所以**只有 6x8 撞上了这个坑**；MASM 侧写死
   `vfmadd231ps`，不受影响。
2. 已写进 `AGENTS.md` 汇编一节第 9 条：**编译期分支都要有人走过**。默认构建永远
   只走「有 FMA」那条，去掉 `-mfma` 是唯一能触发另一条的办法。
