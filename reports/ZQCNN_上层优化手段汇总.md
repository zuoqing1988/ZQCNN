# ZQCNN 上层优化手段汇总（GEMM 之外）

- 日期：2026-10-01
- 范围：`ZQCNN/` 的算子层、数据布局、内存、并行、图优化。**不含** `ZQ_GEMM/`
  的手写汇编内核（见 `ZQ_GEMM_汇编内核性能对比.md`）。
- 方法：逐文件阅读 + 脚本交叉检索；关键结论**逐条回查源码验证**，验证不了的
  明确标"待确认"。

> **本文最重要的一条方法论**：下面把「**结构上存在**」和「**实测更快**」
> 严格分开。绝大多数优化属于前者 —— 代码路径确实存在、确实被走到，但我们
> **没有做算子级的 A/B 实测**。唯一有实测数据的是图优化那一节，而它的结论
> 恰恰是"**看不出差别**"。把"有这段代码"当成"有性能收益"是这类项目里最常见的
> 自我误导。

---

## 一、图优化（`ZQ_CNN_Net.h`）

加载期一次性完成，前向时零成本。

| 手段 | 位置 | 做什么 |
|---|---|---|
| `_simplify_inplace()` | `ZQ_CNN_Net.h:121` | 对 ReLU/ReLU6/PReLU/BatchNormScale/BatchNorm/Scale/AddBias 这 7 种**逐元素**层，若其 top blob 之后没人再用，就把 top 重定向到底 blob，等于**整个 blob 被复用**。这是"in-place"在这里的实现方式 —— 靠改图而不是靠算子分支 |
| `_merge_bn()` | `ZQ_CNN_Net.h:1474` | `Convolution+BatchNormScale` / `DepthwiseConvolution+BatchNormScale` 相邻且 BN 输出没被别处引用时，把 BN 的 scale/bias 折进卷积权重，**删掉 BN 层** |
| `_merge_prelu()` | `ZQ_CNN_Net.h:1606` | 把 PReLU 的 slope 折进相邻卷积（`with_prelu`），删掉 PReLU 层 |
| 算子归约 | `ZQ_CNN_Layer.h:7024/7038` | `SCALAR_DIV` 改写成 `Mul(1/scalar)`、`SCALAR_MINUS` 改写成 `Add(-scalar)`，复用更便宜的内核 |

开关在 `LoadFrom(..., bool merge_bn, float ignore_small_value, bool merge_prelu)`
（`ZQ_CNN_Net.h:54`）。`ignore_small_value` 是合并时把极小权重**直接归零**的
阈值 —— 减少后续计算的同时也抹掉数值噪声。

**实际谁开了**：`SampleMTCNN` / `SampleCascadeOnet` 等在 `InitFromBuffer` /
`Init` 里传 `true, 1e-9, true`（如 `ZQ_CNN_MTCNN.h:109-113`）；
`SampleLnet106`、`SampleHeatMap`（第二个网）同样开启。

### 1.1 实测：图优化在 MTCNN 上**看不出端到端收益**

临时加了 `ZQCNN_NO_GRAPH_OPT=1` 开关关掉 `merge_bn`/`merge_prelu`，
`SampleMTCNN` 连测 3 轮：

| | 第 1 轮 | 第 2 轮 | 第 3 轮 |
|---|---|---|---|
| 图优化开（默认） | 23.908 ms | 25.753 ms | 23.160 ms |
| 图优化关 | 24.086 ms | 25.851 ms | 22.670 ms |

**差异完全落在噪声内。**（临时开关已回退，未留在代码里。）

原因不难理解：MTCNN 的三个模型 BN 极少 —— `model/det1-dw20-fast.zqparams` 只有
**1 个** `BatchNormScale`，`model/det3.zqparams` **一个都没有**；PReLU 有 5~6 个。
融合省的是"单独一趟读 + 写特征图"，而融合后这部分工作被并进卷积/激活**已有的
内存遍历**里。对本来就小、且瓶颈在卷积算术的特征图，**省下来的带宽不是瓶颈**。

> **结论**：图融合的价值不能想当然地按"少了一层 = 快"来估。在本仓库自带的
> 这几个模型上它主要是**兼容性/简洁性**收益。要拿到性能，得换一个 BN 密集的
> 网络（例如 ResNet）再测。

---

## 二、数据布局

### 2.1 NCHW 的三个对齐变体

`ZQ_CNN_Tensor4D.h:17` 定义 `ALIGN_0 / ALIGN_128bit / ALIGN_256bit`，
**运行期**由 `ZQ_CNN_Forward_SSEUtils::_xxx(align_mode, ...)` 分派到不同内核。

| 变体 | 通道步长 | 位置 | 含义 |
|---|---|---|---|
| `Align0` | `pixelStep = C` | `ZQ_CNN_Tensor4D.cpp:182` | 零填充，任意 C |
| `Align128bit` | `((C+3)>>2)<<2` | `:888` | C 补到 4 的倍数，保证 4 个 float 连续可整向量读写 |
| `Align256bit` | `((C+7)>>3)<<3` | `:1662` | C 补到 8，配合 AVX 的 `zq_mm_align_size=8` |

宽度自适应：`align_mode = __min(GetAlignType(), dst.GetAlignType())`
（`ZQ_CNN_Tensor4D.cpp:286/1002`）—— **取源和目标里较窄的那个**，
保证所有向量访存都不越界。这是"宽度自适应"而不是分块。

### 2.2 NCHWC1/4/8 三个变体

`ZQ_CNN_Tensor4D_NCHWC.h:581/615/653`，`GetAlignSize()` 返回 1/4/8。
与 NCHW 相反的方向：把通道放在**最内维**，让卷积的 C 维天然连续。

**注意同名反义**（极易搞错，AGENTS.md 里有专门一条）：

```
NCHW  : sliceStep = 一"张图"      (sliceStep = widthStep * H)
NCHWC : sliceStep = 一"通道切片"   (imStep   = 一"张图")
```

NCHWC4/8 的 `IsBorderEnabled()` 恒为 true（`:619/657`）——
**border 直接物化在张量内存里**，所以 SIMD 核不需要逐像素做边界判断。

### 2.3 双平台 SIMD 宽度**不等价**（已实测确认）

`ZQ_CNN_CompileConfig.h:21` Windows 硬编码 `ZQ_CNN_SSETYPE_AVX2`，
而 `:76` Linux 侧只到 `AVX`。两侧能跑的指令集不同，写跨平台性能结论时要留意。
（GEMM 那边已经通过"FMA 开关对齐"修过一次，附录 P/R 有记录。）

---

## 三、内存与分配

### 3.1 全网共享的 im2col 暂存池（最大的一条免分配路径）

`ZQ_CNN_Net.h:45/203-205`：整个 Net 持有一块 `_buffer`，加载时把
`&_buffer.data / &_buffer.len` 交给**每一层**，所有卷积/内积的 im2col 共用同一块
scratch，而不是每层各自 malloc。NCHWC 版同构（`ZQ_CNN_Net_NCHWC.h:47/256`）。
可用 `TurnOnUseBuffer/TurnOffUseBuffer` 关掉（`ZQ_CNN_Net.h:27/51`）。

### 3.2 `ChangeSize` **不是**"长度够就复用"，而是**精确等长**才复用

`ZQ_CNN_Tensor4D.cpp:211`（及 `:917`/`:1691`，NCHWC `:177`/`:1085`）：

```cpp
if (rawDataLen != needed_dst_raw_len) {
    unsigned char* tmp_data = (unsigned char*)malloc(needed_dst_raw_len);
    ...
    memset(tmp_data, 0, needed_dst_raw_len);   // 还会整块清零
}
```

**新张量只要大了一丁点，就 free + malloc + 整块 memset 一遍。**
形状完全相同时直接 `return true` 不碰内存（`:176`/`:882`/`:1656`）。

> 这是一个**现成的优化空间而不是已实现的优化**：改成"容量够就直接用、只清要用的
> 部分"（把 `rawDataLen` 拆成 `cap` 与 `len`），连续变尺寸的层（MTCNN 金字塔、
> SSD 各 scale）每层都能省一次 malloc+整块 memset。**本轮没有实施**，
> 因为它会改变"新分配的区域一定是零"这一隐含契约，而 ZQ_CNN_NCHWC_ALLOC_SLACK
> 那条投机读余量正依赖这个契约 —— 要动必须连带重新验证越界。

### 3.3 对齐分配

`Align128bit` 用 `_aligned_malloc(len, 16)`、`:1693` `Align256bit` 用 32、
`Align0` 用裸 `malloc`；Linux 侧 `ZQ_CNN_CompileConfig.h:108-114` 把
`_aligned_malloc` 映射到 `memalign`。

**NCHWC 特有的 64 字节余量**（`ZQ_CNN_Tensor4D_NCHWC.cpp:5`）：
SIMD 核（尤其 depthwise 3×3）在最后一个输出像素做**投机读**，会越界 16 字节，
所以多分配 64 字节并整块清零兜底。**这是 2026-10-01 本次审计加的加固**，
不是性能优化。

### 3.4 紧凑布局互转代替跨步转置

`Permute` / `Flatten` / `Reshape`（`ZQ_CNN_Tensor4D.h:600-640`、
`ZQ_CNN_Tensor4D_NCHWC.h:360-395`）**不直接做跨步 gather**，而是
"导出紧凑 → 再导入紧凑"，用一段连续 `memcpy` 换掉 O(N) 的 stride-N 访问。

### 3.5 每层 Forward 里的临时 malloc（未优化）

- `LRN`：每次 Forward 三个 `_aligned_malloc`（`layers_c/zq_cnn_lrn_32f_align_c_raw.h:34-36`）
- `BatchNorm`：每次把 mean/var/scale/bias 现场折成 `a/b` 两个 C 长数组（`:24-25`），两次 malloc+free

---

## 四、算子与 SIMD 内核

### 4.1 两套实现，21(NCHW) + 10(NCHWC) 个算子

`layers_c/`（NCHW）与 `layers_nchwc/`（NCHWC）。`layers_c/` 里**没有手写汇编**，
全部是 intrinsics 经 `*_raw.h` 的宏展开生成，每个算子按
NEON / SSE / AVX / 标量回退四路实例化。

几处值得单说的：

- **多路手工展开**（`zq_cnn_eltwise_nchwc_raw.h`、`zq_cnn_depthwise_convolution_32f_align_c_raw.h`）：
  `op_0_4 → op_0_8 → op_0_16 → op_0_32 → op_0_64` 逐级展开，一次循环体处理
  最多 64 个向量，减少循环开销。
- **水平归约用标量宏**（`zq_cnn_softmax_32f_align_c.c:56/93` `zq_final_max_q`/`zq_final_sum_q`），
  因为 intrinsics 没有横向 max/sum。
- **自研多项式 exp/log 替代 libm**（`math/zq_sse_mathfun.h` / `zq_avx_mathfun.h`），
  Softmax 和 LRN 的 `pow` 都走它（`zq_cnn_lrn_32f_align_c_raw.h:82`
  `pow_v = zq_mm_exp_ps(beta * zq_mm_log_ps(sum_v))`）——**用 exp/log 组合把
  整条 `pow` 指令向量化**。

### 4.2 BatchNorm 的参数预折叠（`layers_c/zq_cnn_batchnormscale_32f_align_c_raw.h:24-33`）

把 4 个 C 长参数（mean/var/scale/bias）**折成 2 个** `a/b`：

```
b = slope / sqrt(var + eps)      a = bias - mean*b
```

内核里只剩 `fma(b, x, a)` —— **4 次读 + 一次除法 + 一次减法 → 2 次读 + 1 次 FMA**。
这与 §1 的 `_merge_bn` 是同一件事的两种实现粒度（层间 vs 内核内）。

### 4.3 Eltwise 的首元素分离（`layers_c/zq_cnn_eltwise_32f_align_c_raw.h:20-40`）

`op_sum_0_4_first` 与 `op_sum_0_4` 分开：第一个输入做 `load + store`，
后续输入直接累加到 output —— **省掉 N−1 次中间张量写**。NCHWC 同构。

### 4.4 算子归约与 in-place 内核

`layers_c/zq_cnn_scalaroperation_32f_align_c.h` 有 64 个符号，每个标量运算都
有 `_inplace_` 变体；`ZQ_CNN_Layer.h:7008-7075` 逐个判断
`(*bottoms)[0] == (*tops)[0]` 来选。

12 个层在 `tops[0] == bottoms[0]`（同一张量）时**跳过 `CopyData` 直接原地算**：
BatchNormScale / BatchNorm / Scale / AddBias / PReLU / ReLU / ReLU6 / Softmax /
Dropout / Copy / Normalize / Squeeze（`ZQ_CNN_Layer.h:2006/2360/2609/2895/3094/
3289/3429/6102/6241/6380/7726/8445`）。NCHWC 侧只覆盖了 4 个。

### 4.5 Resize 的 ROI 短路

`ZQ_CNN_Tensor4D.cpp:254-257` 等 6 处：
`if (dst_W == src_rect_w && dst_H == src_rect_h) return ROI(...)` ——
**目标尺寸等于源 ROI 时退化成纯拷贝**，整条双线性插值路径被跳过。
MTCNN 的 1×1 裁剪和边界框 resize 常用到。

`can_call_safeborder`（`:280-284`）：resize rect 贴到图像边界但目标又比 rect 大时，
判定"没有安全 border"，改走逐像素夹边界的 `without_safeborder` 内核。

### 4.6 NCHWC 独有的权重预打包（只在加载时做一次）

`ZQ_CNN_Net_NCHWC.h:1138` 的 `_prepack()` 在 `LoadFrom` 里**紧跟
`_merge_bn()` / `_merge_prelu()` 之后调用一次**，前向时零成本。

- `layers_nchwc/zq_cnn_convolution_gemm_nchwc_prepack4.h:2-100`：把 4 个输出
  通道的滤波器**交织打包**成 K 连续的 B 面板（`packed_B_step = paddedC * 4`）。
  这正是"让 N 方向连续"的那条 packing 路线（GEMM 那边我们自己也走了同一条，
  见 `ZQ_GEMM_汇编内核性能对比.md` §3.3）
- `layers_nchwc/zq_cnn_convolution_gemm_nchwc_col2im.h`：GEMM 输出 C → NCHWC
  图像的 scatter，**通道扩张 + bias + PReLU 全部融在 col2im 循环里**
  （`fmadd(slope, min(0,a), max(0,a))` 就在那一趟）
- `layers_nchwc/zq_cnn_convolution_gemm_nchwc_packed4.h`（5616+ 行）里的
  `_handle_bias_prelu_{1x4,4x4,6x4,8x4}` 把 bias+PReLU 融进 **epilogue**

**注意**：`packed4_handle_bias_prelu_*.h` 被 `packed4.h` 在
`#if __ARM_NEON && __ARM_NEON_ARMV8`（`:312`）内 include，**x86 上是死代码**。

### 4.7 NCHW 侧没有 prepack

`ZQ_CNN_Net.h` 里**没有** `_prepack()` —— NCHW 路径没有权重预打包。
这与 NCHWC 相比是一个明确的能力差距。

---

## 五、并行

### 5.1 算子层**完全单线程**

`layers_c/`、`layers_nchwc/`、`ZQ_CNN_Forward_SSEUtils*`、`ZQ_CNN_Tensor4D*`
里**一个 `#pragma omp parallel` 都没有**。

`#pragma omp` 只出现在三处：

| 位置 | 粒度 | schedule | 线程数来源 |
|---|---|---|---|
| `ZQ_CNN_BBoxUtils.h:97` | NMS 的 IoU 抑制，按候选框数切 | `schedule(static, chunk_size)`，`chunk = ceil(box_num/thread_num)` | 调用方传入的 `thread_num`；`thread_num<=1` 走单线程分支（同时避开 `schedule(static,0)` 的未定义行为） |
| `ZQ_CNN_MTCNN*.h`（每变体 7 处） | PNet 金字塔各 scale 一线程 / RNet、ONet 按候选框分块 | `dynamic,1`（负载不均）/ `static, chunk_size` | `Init(..., thread_num=1)`；`thread_num<1` 置 `force_run_pnet_multithread=true` 并 `__max(1,thread_num)` |
| `layers_nchwc/..._kernel1x1_neon_raw.h`（6 处） | 1×1 卷积的行 | 无（默认 static） | **仅 ARM NEON 生效** |

全仓**没有** `omp_set_num_threads` / `omp_get_max_threads`，
线程数一律由 `num_threads(...)` 显式给出，**不会意外吃满机器**。
`omp_get_wtime()` 的 60+ 处**只是计时**，不是并行。

### 5.2 MTCNN 的并行代价

每个线程一套**完整的 Net 实例**（`ZQ_CNN_MTCNN.h:110-115`），
**权重按线程复制 N 份**。`thread_num` 加大时内存占用与权重加载时间线性增长。
（未量化。）

---

## 六、明确的空白点

这些是**查完确认不存在**的，写在这里以免后人重复查：

| 空白 | 证据 |
|---|---|
| **没有任何预取指令** | `grep -rn "_mm_prefetch\|__builtin_prefetch\|prfch"` 全仓（含 ZQ_GEMM）**零命中** |
| **没有任何针对 L1/L2 的显式 cache 分块** | 唯一的"分块"是 LRN 的 C 维前缀和（`zq_cnn_lrn_32f_align_c_raw.h:32-78`，算法性的，把 O(C·local_size) 降到 O(C)），不是 cache blocking |
| **算子级没有 A/B 实测** | 除 §1.1 的图优化外，relu/eltwise/bn/resize/pooling 都没有"改前改后"的对比数据 |
| **`ConvertFromBGR` / `ConvertColor_BGR2GRAY` 没 SIMD 化** | `ZQ_CNN_Tensor4D.h:339-343` / `:400` 仍是纯标量 `unsigned char → float` 循环，在预处理链上 |
| **非 GEMM 算子的 int8/量化路径** | `layers_c/` 只有 `32f` 与 NEON 的 `16f`（float16）两族，没有 int8 推理路径 |
| **NCHW 侧没有权重预打包** | 见 §4.7 |

---

## 七、需要澄清的一处表象

`layers_c/zq_cnn_depthwise_convolution_32f_align_c.c:64-90` 里
`kernel3x3_C16 / C24 / C32 / C64 / C128 / C256`、`Cdiv16 / Cdiv32`
这些名字**全部是 `#define` 别名**，指向 `..._mul_4` / `..._mul_8` / `..._mul_16`
这三个不同的实现（`:75/79/85`）。也就是说这些符号名本身不生成独立代码，
但它们指向的 `mul_4/8/16` 确实是三个不同展开度的函数。**看符号名会误以为
"每个 C 值一份特化内核"，实际不是。**（`in_C` 仍是运行期参数，
`raw.h:176`。）

另：`ZQ_CNN_Forward_SSEUtils.cpp:300/489/534` 里 AVX 的 2×2 maxpool 特化
是被**注释掉**的（`/*&& 0*//*seems slow*/`）—— 作者实测过更慢，
不是写错。

---

## 八、小结与下一步的候选

| 方向 | 现状 | 潜在收益 | 风险 |
|---|---|---|---|
| `ChangeSize` 容量复用 | 精确等长才复用 | 连续变尺寸的层每省一次 malloc + 整块 memset | 破坏"新分配必为零"的隐含契约，NCHWC 投机读余量依赖它 |
| NCHW 补 prepack | 没有 | NCHW 与 NCHWC 的一个明确能力差 | 需要重写 NCHW 侧 GEMM 前后的数据流 |
| 算子级 A/B 基准 | 没有 | 先有尺子再谈优化 | 低，但需要写基准 |
| ConvertFromBGR SIMD 化 | 纯标量 | 预处理链上一处明确热点 | 低 |
| 预取 / cache 分块 | 完全没有 | 未测 | 需要 profiler（本机 WSL 无 `perf`） |
| 图优化的真实价值 | 在自带模型上测不出 | 需换 BN 密集的模型重测 | 低，但要一个 ResNet 类的模型文件 |

前三项的共同前提都是：**先把测量做出来**。这与 GEMM 那边的结论一致 ——
本项目目前缺的不是一个优化点，而是一套能判断优化是否有效的尺子
（见 `tools/bench_gemm_ab.py` 和 `tools/sweep_samples_win.py`）。
