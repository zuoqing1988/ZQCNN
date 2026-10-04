# CHANGELOG 2026-10-05

## 新增/变更：HV —— 把那 5 层的**真实权重**灌进合成网跑 merge_bn：不复现，嫌疑只剩共享拓扑

### 装置
`tools/slice_model_weights.py` 从 model/mobilefacenet-v1.nchwbin 里精确切出
`res4_block1_conv_dw` 及其后随的 5 层（HU.4）：**147456 字节，36864 个 float，
全非零，值域 ±0.5** —— 训练出来的真实权重。
工具再生成一个**能直接 LoadFrom 的合成网参数**（6 层）：
    Input              name=res4_block1_conv C=256 H=14 W=14
    DepthwiseConvolution res4_block1_conv_dw   bottom=res4_block1_conv top=res4_block1_conv_dw num_output=256 kernel_size=3 stride=1 pad=1
    BatchNormScale      res4_block1_conv_dw_bn bottom=res4_block1_conv_dw top=res4_block1_conv_dw bias
    PReLU               res4_block1_conv_dw_relu bottom=res4_block1_conv_dw top=res4_block1_conv_dw
    Convolution         res4_block1_conv_sep bottom=res4_block1_conv_dw top=res4_block1_conv_sep num_output=128 kernel_size=1 stride=1 pad=0
    BatchNormScale      res4_block1_conv_sep_bn bottom=res4_block1_conv_sep top=res4_block1_conv_sep bias
`Input` 的 **name 必须等于第一层的 bottom**（Input 层的 blob 名就是它的层名），
H/W 取真模型里这一层的实际特征图尺寸 14（附录 HH.3 量到 blob 形状 [1][14][14][256]）。
`SamplesZQCNN/SampleSliceMerge/` 加载两份（默认参数 / 生产实参 (true, 1e-9, true)），
喂**同一份确定性输入**，比最后一个 blob 的后向误差。

### 结果：不复现，而且是**精确**的不复现
    输出 res4_block1_conv_sep：25088 个 float
    B(merge_bn) vs A(默认) 后向误差 8.263e-09   (Linux)
                              7.512e-09   (Windows)
    SLICE MERGE: NOT REPRODUCED
> **这 5 层（dwconv + BN + PReLU + conv + BN）的融合，用真实训练权重，是对的。**
> 也就是说：HE.4 那条 0.37 的缺陷**不在这一段**。

### 嫌疑只剩一样东西：共享 blob 拓扑
HV.2 排除了"真实权重"这个变量，而 HN.1 已经证明
"两个 dwconv 写同一个 blob"这个拓扑配**人工权重**也是对的。
**两者都对，唯独真实模型里错** —— 剩下的差别只有**两者的组合**：
    单个 dwconv+BN（HN.3）      人工权重 ✅1e-08  真实权重 ✅8e-09（本附录）  都对
    两个 dwconv 写同一 blob（HN.1）人工权重 ✅1.8e-08  真实权重 **未测**      <== 唯一没测过的格子
**那唯一没测的格子是：共享 blob 拓扑 + 真实权重。**

### 但它切不出来 —— 这本身是 HN.4 预言的"天花板"
试着用同一个工具切 `res4_block1_conv` 开始的整段共享路径，它在 8 层后停住了：
    切出 8 层：res4_block1_conv … res4_block1_conv_sep_bn
因为下一层 `res4_block2_conv` 的 bottom 是 `_plus5`，而 `_plus5` 由
**被切掉的 Eltwise** 产出 —— 前缀加载失败，工具正确地报出了这个边界。
> **共享 skip 拓扑的依赖一路往上，没有小切片能单独隔离它。**
> 任何包含这个拓扑的子网，都得把前面几十层一起带上。
> 所以那唯一没测的格子**不能靠"切"得到，只能靠"造"** ——
> 而"造"需要一个网，它的**四个 dwconv+BN 用的都是真实权重**。
> 好消息是：**每对 dwconv+BN 的真实权重都能单独切出来**
> （`slice_model_weights.py <层名>` 就是干这个的），
> 所以那个网是**造得出来的**：用 block1~block4 各自真实的 dwconv+BN 权重，
> 拼成"一个 Input + 四个 dwconv+BN 写同一 blob"的合成网。
> **这是下一批该做的那一步，且现在每一步都已经有工具了。**

### 跨平台：又修了三处**路径**问题（一处是我自己的路径理解错）
1. **`.zqslice` 写在哪**：工具只写 `cmake-out-unix-x64/…`，
   而 Windows 的 sample 跑在另一棵树下面 -> Windows 侧"找不到"
   （报得对、rc 也对，但那一侧等于没验）。改成**两棵产物树各放一份**。
2. **反斜杠在 shell heredoc 里会被吃掉**：我把
   `snprintf(..., "%s\slice.zqparams", ...)` 拼进去，落到文件里成了
   `"%s\slice.zqparams"`（`\s`）-> 拼出 `.zqsliceslice.zqparams`。
   **连修三次都栽在同一个地方**（这一整轮 `\n` 变成真换行，也是同一个原因）。
   > 结论：**别在 shell heredoc 里放反斜杠**。要么用 `Write` 工具，
   > 要么**一个反斜杠都不写**（路径全用正斜杠 —— Windows 的文件 API 本来就接受）。
3. **候选路径是相对谁**：本 sample **从产物目录跑**，所以候选路径相对**产物目录**。
   我第一个候选写成相对仓库根的 `cmake-out-win32-x64/...`，
   而 cwd 已经是产物目录，拼出来就成了 `<产物>/cmake-out-win32-x64/...`。
   > 判据：**"相对谁"要与"cwd 在哪"一起写进注释**，
   > 否则下一个读代码的人（和下一个我）会把 cwd 当成仓库根。

### 变更文件
  SamplesZQCNN/SampleSliceMerge/SampleSliceMerge.cpp（新增）。
    找不到权重时**明确报"没生成"并退出非 0**，不空跑
    （这一点在 Windows 上先红过、报得对、rc 也对 —— 守卫是有用的）。
  tools/slice_model_weights.py —— 增加**合成网参数**输出，以及**两棵产物树各放一份**。
  audit_k3_20261001.md（追加 HV）
**无生产代码改动** —— 两个平台均已手工重编 + 实跑（结论一致，rc=0）

## 新增/变更：HX —— `merge_bn` 那条活缺陷**定位并修复**（0.3695 -> 2.9e-07），并落成门禁

### 结论

`SamplesZQCNN/SampleMergeBNCompare` 报的
`mobilefacenet-v1 BAD 后向误差 0.3695（bn only 0.3695 / prelu only 0）`
**这条生产路径缺陷已定位并修好**。修后两个平台都是 17/17 通过：

| 平台 | 修复前 | 修复后 | 汇总行 |
|---|---|---|---|
| Windows (VS2022/AVX2) | 0.3695 BAD | **2.92e-07** | `MERGE COMPARE OK` (rc=0) |
| Linux (gcc 9.4/WSL) | 0.3695 BAD | **2.053e-07** | `MERGE COMPARE OK` (rc=0) |

其余 16 个模型修复前后都在阈值内，**没有一个变差**。

### 根因

`ZQCNN/ZQ_CNN_Net.h` 的 `_merge_bn` / `_merge_prelu` 把「卷积 + 紧随其后的
BN/PReLU」折成一层。原来的守卫只有一条：

```cpp
if (tops[i + 1][0] == bottoms[i + 1][0] || !later_refer)   // 折
```

它问的是「**这个卷积的输出后面还有没有人要**」，
而真正要问的是「**后一层吃的是不是这个卷积的输出**」。
`model/mobilefacenet-v1.zqparams` 第 108/109 行恰好不满足：

```
DepthwiseConvolution  name=res4_block5_conv_dw     bottom=res4_block1_conv     top=res4_block5_conv_dw
BatchNormScale       name=res4_block5_conv_dw_bn  bottom=res4_block1_conv_dw  top=res4_block1_conv_dw
```

于是 block5 的 BN 读的是**上一个 block 留下的** blob，
而 block5 的 dwconv 输出 `res4_block5_conv_dw` **压根没人读**（死层）。
`later_refer` 在这一格恰好答"没有"，于是把一个喂错了输入的 BN 照折不误：

* 原图算「**别人的值** × 逐通道系数」
* 融合后算「**这个卷积自己的输出** × 逐通道系数」

两个数差 0.37。**折叠的算术是对的，错的是"折哪一对层"这个上下文** ——
这正是 HG 那一轮"算术已证对而结果错"排除掉的那一大类。

### 修法

补一条**必要前提**（`_merge_bn` 三处 + `_merge_prelu` 两处，含被注释的
InnerProduct 分支）：

```cpp
if (tops[i][0] == bottoms[i + 1][0] &&
    (tops[i + 1][0] == bottoms[i + 1][0] || !later_refer))
```

不满足就**不折**，语义与未融合路径逐位一致。
`tops` / `bottoms` 存的是 **blob 下标**，所以别名（两个名字映射到同一个下标）
天然被覆盖。

**没有改 `.zqparams`**：block5 的 dwconv 是死层、`BN.bottom` 写错 blob 名，
那是**模型**的缺陷；本仓库不拥有那个模型，改它会改变所有人的 embedding。
库的契约是「融合不改变结果」，所以正确做法是**不折**，
而不是"让融合也复现模型的 bug"。

### 配套门禁（新增）

`tools/check_bn_prelu_pairing.py`（已接进 `run_audit_checks.py` 的 C16 组）：
扫全仓 27 个 `.zqparams` 里**相邻**的
「Convolution/DepthwiseConvolution/InnerProduct + BatchNormScale/PReLU」层对，
凡是 `bottom=` 不是上一层的 `top=` 就报出来。
随仓已知 1 处（就是 mobilefacenet-v1 那一行），记在
`tools/bn_prelu_pairing_baseline.txt` 里**只报信息、不判失败**；
**基线之外**新增的判失败。

> 为什么要门禁化：这条查了 HE~HX 共**十一轮**才定位，
> 而它的形态是"**模型写错一行，库就静默改结果**"——
> 下一次换个模型再写错同样的接线时，应该在**加载之前**就报出来，
> 而不是等到某天输出对不上再从头二分一遍。

### 回归接入

`SampleMergeBNCompare` 之前**故意没有**接进回归
（AGENTS.md「一个恒红的检查不要接进回归」：恒红会把别的真回归失败淹掉）。
现在它绿了，于是：

* 加进 `tools/run_sample_regression.sh` 的 Linux sample 列表；
* 加进 `tools/run_audit_checks.py` 的 `WIN_SAMPLES`。

`SampleSliceMerge` **仍然不接**：它的输入
（`.zqslice/`）在 `cmake-out-*/Release/` 下、被 gitignore 排除，
干净克隆上不存在，接进去必然 NOOUT/FAIL。它是**按需复现手段**。

### 变更文件

* `ZQCNN/ZQ_CNN_Net.h` — `_merge_bn` / `_merge_prelu` 各补一条必要前提；
  两个函数头各加一段说明（含真模型那两行原文与症状）。
* `ZQCNN/ZQ_CNN_Net_NCHWC.h` — **同一处修改的第二份拷贝**（各 5 处）。
  它是生产代码（`ZQ_CNN_Net_NCHWC` / `SampleMTCNN_NCHWC4` 走这一条），
  症状与主文件**完全一样**，所以必须一起修。
* `ZQCNN_to_MNN/converter/source/ZQ_CNN_Net.h` — **第三份拷贝**（各 5 处）。
  整个转换器**不在主构建里**（`ZQCNN_to_MNN/converter/CMakelists.txt` 自成项目，
  且要 `MNN_generated.h` / `addBizCode.hpp` / `optimizer.hpp` / `writeFb.hpp`
  这些**仓库里没有**的 MNN SDK 头，本机编不了），
  所以只用 `g++ -fsyntax-only` 单独验了这个头 —— **干净通过**。
  > 这一条是 AGENTS.md「补齐一处修复时把同仓的另一份拷贝列出来」的第三次应用。
  > 判据：`grep -rn 'later_refer' --include=*.h` 把名单一次列全，
  > 改完再 `grep` 一次确认**每一处**都带上了前提
  > （本轮三份共 15 处，改完逐文件核对：5 / 5 / 5，旧的写法 0 处）。
* `tools/check_bn_prelu_pairing.py` — **新增**门禁（带 `--selftest` 阳性对照
  与 `--rewrite-baseline`）。
* `tools/bn_prelu_pairing_baseline.txt` — **新增**基线（1 条）。
* `tools/run_audit_checks.py` — 挂 C16 组；`WIN_SAMPLES` 加
  `SampleMergeBNCompare.exe`。
* `tools/run_sample_regression.sh` — Linux sample 列表加
  `SampleMergeBNCompare`，并写明 `SampleSliceMerge` 为什么**不**加。
* `SamplesZQCNN/SampleSliceMerge/SampleSliceMerge.cpp` — 支持
  `multi` / `slice` 两段（优先共享拓扑那格），输出 blob 名改成从参数文件
  最后一个 `top=` 读，不再写死；结尾那句已被 HX 推翻的结论**标注为已推翻**。
* `audit_k3_20261001.md`（追加 HW / HX）
* `AGENTS.md`（新增「折/融合/重写类代码的守卫要按语义前提重推」一节；
  并给「恒红的检查不要接进回归」补上"修好之后要真的接进去"这第三步）

### 注意事项

1. **`tops[i][0] == bottoms[i + 1][0]` 是必要条件，不是"更严的优化"。**
   少了它，折叠结果与原图**不是同一个函数**；多了它，只是少折一对层
   （block5 的那一对），性能损失是一层 BN，语义零风险。
2. **`_merge_prelu` 是同一个洞**，随仓 27 个模型一处都没踩到
   （`grep` 出来是 0），但**前提缺失时它和 `_merge_bn` 一样会静默改结果**，
   所以一起补 —— 缺陷不该等到某个模型踩中了才修。
3. **修复后 `mobilefacenet-v1` 的融合结果与未融合结果差 2.9e-07（不是 0）**，
   这是 float32 下 70 多层折叠累积的舍入，与其余 16 个模型同量级。
   判据阈值 1e-4，不要把它当成"应该逐位相同"去要求。
4. **`model/mobilefacenet-v1.zqparams` 第 108/109 行本身仍然是错的**
   （block5 的 dwconv 是个死层，BN 读的是上一个 block 的 blob）。
   这次修的是**库**：让它不再因此改变结果。
   那个模型文件不在本仓库的产出链路上（附录 GU 已量过：27 个 `.zqparams`
   一个都不是仓里那几个 Python 转换器产的），所以**没有动它**。

