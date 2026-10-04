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


## 新增/变更：HY —— NCHWC 那份 `_merge_bn` 的前后对照（**此前零覆盖**，实测真有缺陷）

### 为什么

HX 修 `merge_bn` 时发现同一段守卫在仓库里有**三份**（各 5 处）。
主文件那份修完立刻有 `SampleMergeBNCompare` 盯着，
**NCHWC 那份（`ZQCNN/ZQ_CNN_Net_NCHWC.h`）没有任何东西盯着** ——
而它**确实走生产路径**：

```
ZQCNN/ZQ_CNN_MTCNN_NCHWC.h:109-113
    pnet[i].LoadFrom(pnet_param, pnet_model, true, 1e-9, true)
    && rnet[i].LoadFrom(rnet_param, rnet_model, true, 1e-9, true)
    && onet[i].LoadFrom(onet_param, onet_model, true, 1e-9, true);
```

`SampleMTCNN_NCHWC4` 跑的就是它，但那个 sample **只看检出张数** ——
而"融合算错"的典型症状恰恰是**检出数不变、框变歪**。
**"它在回归里"不等于"它的融合结果被验证过"。**

### 新增 `SamplesZQCNN/SampleMergeBNCompareNCHWC`

`ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC4>` 上做与 NCHW 那份完全相同的对照：
同一份确定性输入（LCG，seed 12345）分别用默认参数与生产实参
（`merge_bn=true, ignore_small_value=1e-9, merge_prelu=true`）加载，
各跑一次 `Forward`，比最后一个 blob 的**后向误差**，阈值 1e-4。
三种组合（bn only / prelu only / 生产）分开跑，为了能指出**是哪一个 merge** 坏的。

实测（两个平台）：

| 平台 | 结果 |
|---|---|
| Windows VS2022/AVX2 | 17 个模型全过（mobilefacenet-v1 = **2.737e-07**），`NCHWC MERGE COMPARE OK` rc=0 |
| Linux gcc 9.4/WSL | 17 个模型全过（mobilefacenet-v1 = **2.628e-07**），`NCHWC MERGE COMPARE OK` rc=0 |

### 变异测试：判据有鉴别力，不是"整体恒红"

把 `ZQ_CNN_Net_NCHWC.h` **单独退回 HX 修复前的版本**（`git show HEAD~1:...`），
只编这一个 sample 再跑：

```
mobilefacenet-v1   BAD  后向误差 0.1874 > 0.0001（bn only 0.1874 / prelu only 0） -> merge_bn 是元凶
共 17 个模型：跑过 17（通过 16，超阈值 1），跳过 0
NCHWC MERGE COMPARE FAILED        （rc 非 0）
```

其余 16 个模型**仍然绿** —— 也就是说这条判据**恰好**只抓那一个模型，
不是"所有模型一起红"的空判据。
（0.1874 与 NCHW 那条路径的 0.3695 同量级但不相等：
两份拷贝用的张量布局不同，舍入不同，**结论方向与量级一致**。）

改回修复版 → 5 处守卫全部还原、0 编译错误、`NCHWC MERGE COMPARE OK` rc=0。

> 这一步同时证明了一件事：**NCHWC 那条路径上确实存在这条活缺陷**，
> 只是**没有任何判据在看它** ——
> 与 AGENTS.md「门禁的 `return 0` 不是"生产里没人用"的挡箭牌」同源：
> **没被断言覆盖的路径，坏了也不告诉你。**

### 回归接入

* `tools/run_sample_regression.sh`（Linux 列表）加 `SampleMergeBNCompareNCHWC`
* `tools/run_audit_checks.py` 的 `WIN_SAMPLES` 加 `SampleMergeBNCompareNCHWC.exe`

它的权重全部来自 `model/`（在版本库里），**不需要**任何现场生成步骤，
所以两条路径都能直接接（与 `SampleSliceMerge` 不同，那条见 HX 的说明）。

### 变更文件

* `SamplesZQCNN/SampleMergeBNCompareNCHWC/SampleMergeBNCompareNCHWC.cpp` — **新增**。
* `tools/run_sample_regression.sh`、`tools/run_audit_checks.py` — 接入上面那个 sample。
* `audit_k3_20261001.md`（追加 HY）
* `AGENTS.md` — 在新加的那一节里补一条"**第二份拷贝要有自己的对照**"。
* **无生产代码改动**；两个平台均已手工重编 + 实跑（rc=0，17/17）

### 注意事项

1. **新增 sample 之后必须重新 `cmake -S . -B build_x64 ...`**，
   否则 `file(GLOB)` 不生效（AGENTS.md「构建规则」第 4 条 / HE.8）。
   本轮 Windows 侧重新 configure 过；Linux 侧 `cmake /mnt/d/ZQCNN -B /tmp/zqb2`。
2. **`ZQ_CNN_Net_NCHWC<T>::GetBlobByName` 返回的是 `const Tensor4D*`**
   （也就是 `const ZQ_CNN_Tensor4D_NCHWC4*`），**不是基类指针** ——
   按 `const ZQ_CNN_Tensor4D*` 接会得到
   `error C2440: 无法将 "const Tensor4D *" 转换为 ...`。
   本 sample 的 `read_blob` 因此写成**模板**（只需要 `GetN/C/H/W` +
   `ConvertToCompactNCHW`，两者都在基类上）。
3. `mobilefacenet-v1` 的 `Input` 行带 `H=112 W=112`，
   而 `ZQ_CNN_Net_NCHWC::Forward` 在**有 InnerProduct 层**时会校验输入形状 ——
   所以本 sample 显式从参数文件读 `C/H/W` 再造输入，
   取不到就**报 SKIP 并说明原因**，不静默跳过。

## 新增/变更：HZ —— 两个融合对照 sample 的模型列表补全到**全部 27 个**，并修掉一个"按位置解析字段"的洞

### 起因

HY 证明了"**没被列进判据的模型就是零覆盖**"。
而两个对照 sample 的模型列表都只列了 17 个 —— `model/` 下实际有 **27** 个
`.zqparams`，于是下面 6 个**有配套权重、却从来没被验过**：

```
MobileNetSSD_deploy    Pose-zq    det5-112-gray
headposegaze-112-gray  libfacedetection    model-face
```

（另有 4 个没有配套 `.nchwbin`：`det1` / `det2` / `det3` /
`mobilefacenet-res2-6-10-2-dim128`。）

### 顺带查出一个"测量本身有洞"

把新模型加进列表后，`MobileNetSSD_deploy` 与 `libfacedetection` 仍然被报成
"取不到 Input 的形状"。原因不是模型，是**判据自己的解析器**：

```
det5-dw112.zqparams      Input name=data C=3   H=112 W=112
MobileNetSSD_deploy      Input name=data H=300 W=300 C=3      <-- C 在最后
libfacedetection         Input name=data H=240 W=320 C=3      <-- 同上
```

而 `parse_param` 用的是

```cpp
sscanf(s.c_str(), "Input name=%*s C=%d H=%d W=%d", &C, &H, &W);   // **按位置**
```

对后两个模型直接返回 0，`C/H/W` 全留在 0。
**"被跳过"在汇总行里只体现为一个数字**，看不出是"这个模型不适用"
还是"我的解析器没覆盖它的写法"。

改成**按键**取（`tok_int(s,"C",C)` / `"H"` / `"W"`），两个 sample 都改。

> 这一条与本文件「门禁的列名/标签要说它实际数的是什么」同源：
> **汇总行里的一个数字，必须能区分开它内部的多种成因**，
> 否则"跳过 4 个"既可能是模型不适配、也可能是判据没覆盖，
> 而读的人只能看到前者。

### 实测（两个平台）

| 判据 | 修复前 | 修复后（列表补全 + 解析修好） |
|---|---|---|
| `SampleMergeBNCompare`（NCHW） | 共 17 个：跑过 17 | 共 **27** 个：跑过 **23**（通过 23），跳过 4 |
| `SampleMergeBNCompareNCHWC` | 共 17 个：跑过 17 | 共 **27** 个：跑过 **17**（通过 17），跳过 10 |

两个平台**逐项一致**：

* NCHW 路径新覆盖的 6 个全部通过：
  `MobileNetSSD_deploy` 0（逐位相同）、`Pose-zq` 1.943e-07、
  `det5-112-gray` 1.026e-07、`headposegaze-112-gray` 1.186e-06、
  `libfacedetection` 0、`model-face` 0；
* NCHWC 路径这 6 个**全部 SKIP**，原因是 `ZQ_CNN_Net_NCHWC` 不支持它们用到的层
  （`unknown layer type: Permute` / `DetectionOutput` 等）——
  **这是覆盖范围的陈述，不是缺陷**，输出里逐条写了原因。

新增覆盖的这 6 个模型里没有一条踩到 HX 那条接线错接
（新门禁 `check_bn_prelu_pairing.py` 扫全仓仍然只有 1 处），
所以是**如实的"这一族模型没问题"**，不是"没查到"。

### 变更文件

* `SamplesZQCNN/SampleMergeBNCompare/SampleMergeBNCompare.cpp` —
  模型列表 17 → 27；`parse_param` 改为按键取 `C/H/W`。
* `SamplesZQCNN/SampleMergeBNCompareNCHWC/SampleMergeBNCompareNCHWC.cpp` — 同上。
* `audit_k3_20261001.md`（追加 HZ）
* `AGENTS.md` — 补一条「**跳过要能区分成因**」。
* **无生产代码改动**；两个平台均已手工重编 + 实跑（rc=0）

### 注意事项

1. **列表补全之后，"跑过 N 个"这个数字才等于"应该验的都验了"。**
   补之前 17/27，补之后 NCHW 23/27（剩下 4 个真的没有权重文件）。
   > 判据：一份"模型清单"要**能从磁盘枚举出来**才能核对完整性；
   > 手写列表与目录不一致时，**差异部分永远是零覆盖**。
2. NCHWC 路径对那 6 个模型 SKIP 是**如实标注**的，不要改成静默跳过 ——
   静默跳过的话，`17/27` 与 `23/27` 两个数字就长得一样，
   读的人会以为 NCHWC 覆盖得少是因为"那些模型不重要"。

## 新增/变更：HX+HY+HZ 三批落地后的完整回归（v39）

`python tools/run_audit_checks.py --with-build --warn-sweep --src-sweep
--bounds-sweep --ubsan-sweep --reachability --msvc-asan`
（双平台全量构建 + 两平台 sample + 全部 sweep + MSVC ASan）

```
ALL CHECKS PASSED        rc=0        FAILED 计数 = 0
```

覆盖到的组：D1/D2 双平台全量构建、D3 Linux sample 回归、
D4 Windows 七个 sample（含新增的 `SampleMergeBNCompare` 与
`SampleMergeBNCompareNCHWC`）、A1~A16 文本/扫描类门禁、
B/B2 ZQlib ASan + MSVC ASan、C/C1/C1b 可编译性、
C3/C4/C5/C5b/C6/C7/C8/C8b/C9/C10~C16（含新增的 C16 BN/PReLU 接线门禁）、
MSVC `/analyze`、ARM/NEON 与 FP16 档解析。

> 记这一条是为了让下一轮能从"当前树是否已经被完整回归验过"开始，
> 而不是重新跑一遍才知道（HX 那批的教训：v35 是在修复**之前**跑的，
> 结果对修后的树没有意义，只能作废重来一次）。

## 新增/变更：IA / IB —— 给 15 类「没有任何随仓模型跑得到」的层类型造合成网，抓出一条**堆越界读**

### 起因

`run_audit_checks.py` 的 C7 可达性门禁给出一张层类型表：

```
合计 36 种：EXERCISED 20 / COMMENTED 1 / UNUSED 15
UNUSED   DeConvolution  BatchNorm  Scale  Copy  LSTM_TF  ScalarOperation
         UnaryOperation  Sqrt  Tile  Reduction  LRN  Squeeze
         PriorBoxText  PriorBox_MXNET  DetectionOutput_MXNET
```

UNUSED 的意思是「**没有任何随仓库发布的模型会跑到它**」——
这些代码路径**从来没有被任何东西执行过**：
既没有 sample 跑，也没有门禁覆盖。
HX 那条活缺陷查了十二轮，教训之一是「**没被断言覆盖的路径，坏了也不告诉你**」；
这些路径比那条还彻底：**连"跑过"都没有过**。

### IA：新增 `SamplesZQCNN/SampleUnusedLayerProbe`

对每一类 UNUSED 层，**自己写一个最小的 `.zqparams` + `.nchwbin`**，
用真的 `ZQ_CNN_Net::LoadFrom` + `Forward` 跑一遍，
再与**独立写的参考实现**比后向误差。

**参考实现必须先自证**：每个参考都先过一道**手算**用例
（数值取成能约成有理数/整数的），过了才有资格去判库错。

本轮覆盖了 4 类（25 个形状）：

| 层 | 形状数 | 阈值 | 结果 |
|---|---|---|---|
| `LRN` | 9 | 1e-5 | 全过（最大 6.0e-6 @ C=3） |
| `Copy` | 4 | 1e-7 | 全过（**逐位 0**） |
| `Scale`（带 bias / 不带） | 8 | 1e-6 | 全过 |
| `Sqrt` | 4 | 1e-6 | 全过 |

两个平台逐项一致（差异只在 1e-8 量级的浮点舍入）：
Windows `UNUSED LAYER PROBE OK` rc=0、Linux 同。

`LRN` 的形状特意一半取 align(8) 的倍数、一半取非倍数
（`C=1/3/8/13/16/17/32/33/64`，`local_size=1/3/5/7/9`）——
那 9 组里 `C=1, L=1` 正是附录 AX.2 越界写踩过的形状，现在测下来是对的。

**尚未覆盖的 11 类在输出里逐个列出**，不装作已经查过。

### IA.5 一次"看着像库有 bug、其实是我参考写错了"的记录

第一版 `Scale` 的参考把通道下标写成 `i % C`，于是 C=3/8/13 **全部报"对不上"**
（后向误差 0.35~0.54），而 C=1 通过。差点把它当成库里的一条缺陷。

真相：输入输出用的是 **compact NCHW**（`ConvertToCompactNCHW` 那种），
元素 (c,h,w) 的下标是 `(c*H + h)*W + w`，**通道是最外层**，应当是 `i / (H*W)`；
张量本身才是 NHW_C 布局（`pixelStep` 那一维是通道）。
而 `_scalebias` 是在张量上做的，两者一致 —— **库是对的**。

> **教训**：我那个"手算自证"用例选的是 `C=2, H=W=1`，
> 此时 `i/(H*W)` 与 `i%C` **完全等价**，于是自证对**两种约定都通过**，形同虚设。
> 自证用例必须**能区分你要防的那个错误**。
> 改成 `C=2, H=W=2`（`H*W=4 != C`）之后，两种约定给出不同答案
> （`[2,4,6,8,16,19,22,25]` vs `[2,4,7,8,15,18,22,25]`），自证才真的有鉴别力。

### IB：`zq_cnn_scale_32f_align` 带 bias 分支的**堆越界读**（已修）

顺着 `Scale` 往下读实现，发现带 bias 的那一支是**无条件整向量**：

```c
for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size, c_ptr += zq_mm_align_size)
{
    scale_vec = zq_mm_load_ps(scale_data + c);   // 读 zq_mm_align_size 个 float
    bias_vec  = zq_mm_load_ps(bias_data + c);    // 同上
    ...
}
```

而 `scale` / `bias` 两个张量都是 `ChangeSize(1,1,1,in_C,0,0)` ——
**只有 `in_C` 个 float**。于是 `in_C % align != 0` 时最后一下读过 C-1。

**同一个函数里不带 bias 的那一支早就修过**，注释就写在旁边
（"in_C 不是 4/8 的倍数: 整向量读会越过 scale_data 的分配, 改走标量"）——
也就是说**只修了一条分支，另一条留着**。

ASan 实测（新增门禁 `tools/zq_scale_check.cpp`）：

```
ERROR: AddressSanitizer: heap-buffer-overflow ... READ of size 32
  #1 zq_cnn_scale_32f_align256bit
     ZQCNN/layers_c/zq_cnn_batchnormscale_32f_align_c_raw.h:127
0 bytes to the right of 12-byte region        <- scale 的 3 个 float
```

修复：带 bias 的那一支也按 `in_C % zq_mm_align_size == 0` 选路 ——
整倍数走向量，否则走标量。**数值结果完全不变**（标量版本来就是同一组乘法加法），
且整倍数那一档（生产里绝大多数）**仍然走向量，没有性能损失**。

变异测试（把文件退回修复前再跑门禁）：

```
align4  C=3  bias   FAIL 越界读/崩溃
align4  C=13 bias   FAIL 越界读/崩溃
align8  C=3  bias   FAIL 越界读/崩溃
（其余 7 组仍 ok —— 判据有鉴别力，不是"全红"）
```

还原后 `zq_scale PASS`。

### 门禁化

* `tools/zq_scale_check.cpp` — **新增**，已挂进 `tools/run_zqlib_checks.py`
  （`EXTRA_LINK` 直链 `layers_c/zq_cnn_batchnormscale_32f_align_c.c`；
  不编那个 5 分钟的 `ZQ_CNN_Forward_SSEUtils.cpp`，附录 EC.1）。
  用 `zq_check_child.h` 的 `zq_child_silence_stderr()`，
  失败时 harness 会把 `ZQ_CHILD_ERR` 的开头打出来（附录 CZ）。
* `SamplesZQCNN/SampleUnusedLayerProbe` — 接入 `run_sample_regression.sh`
  与 `run_audit_checks.py` 的 `WIN_SAMPLES`。

### 变更文件

* `ZQCNN/layers_c/zq_cnn_batchnormscale_32f_align_c_raw.h` — **生产代码**：
  带 bias 分支加 `in_C % align == 0` 判据 + 标量回退。
* `tools/zq_scale_check.cpp` — **新增** sanitizer 门禁。
* `tools/run_zqlib_checks.py` — 加 `zq_scale` 的 `EXTRA_LINK` / `EXTRA_CXXFLAGS`。
* `SamplesZQCNN/SampleUnusedLayerProbe/SampleUnusedLayerProbe.cpp` — **新增**。
* `tools/run_sample_regression.sh`、`tools/run_audit_checks.py` — 接入上面那个 sample。
* `audit_k3_20261001.md`（追加 IA / IB）；`AGENTS.md`（补两条）

### 注意事项

1. **`Scale` 直接当第一层时输出取不到**：`Scale` 在 `_is_inplace_safe` 表里，
   而 `Copy`/`Sqrt` 不在。`Input -> Scale(top=top1)` 会被 `_simplify_inplace`
   就地化，结果落在 **blob 0（输入张量）** 上，而 `Forward` 结束时把
   `blobs[0]` 置 0 —— 于是 `GetBlobByName` **两个名字都取不到**。
   探针因此垫了一层 `Copy`。这属于**行为记录**，不是本次修的缺陷。
2. **`_aligned_malloc` 只有 MSVC 有**，门禁里的对齐分配收敛成
   `alloc_aligned()` 一个函数（Linux 走 `posix_memalign`）。
3. 第一版门禁用**裸 `malloc`** 分配 `scale`/`bias`，
   align=8 的组直接 SEGV（`_mm256_load_ps` 要求 32 字节对齐），
   把"对齐不够"和"越界读"混成了一个信号。改成
   `posix_memalign(32, C*4)` —— 既对齐又**恰好 C 个 float**，红区紧贴末尾。
   > 判据：**探针的信号必须只对应一个原因**。两个原因共用一句
   > 「越界/崩溃」时，报出来的东西没法用（附录 GZ.3 同族）。

## 新增/变更：IA/IB 落地后的完整回归（v40）

```
python tools/run_audit_checks.py --with-build --warn-sweep --src-sweep \n    --bounds-sweep --ubsan-sweep --reachability --msvc-asan
ALL CHECKS PASSED        rc=0        FAILED 计数 = 0
```

这是**自 v39 以来第一次包含生产代码改动**的一轮
（`zq_cnn_batchnormscale_32f_align_c_raw.h` 的堆越界读修复），
所以它验证的是「改了内核之后双平台仍然跑得通、且新门禁在回归里真的会被执行到」。

新加的两条覆盖在回归里都被跑到了：

* `D4 Windows sample SampleUnusedLayerProbe.exe: OK`（WIN_SAMPLES 已接入）
* `B ZQlib 独立回归测试 x10 (ASan+LSan): OK` —— 其中新增的 `zq_scale`
  由 `tools/run_zqlib_checks.py` 自动发现并执行

## 新增/变更：IC —— 再覆盖 2 类 UNUSED 层（`ScalarOperation` 9 种运算 / `Squeeze`），46 个形状全过

### 起因

IA 把 15 类 UNUSED 里最容易的 4 类覆盖了（LRN / Copy / Scale / Sqrt，25 个形状）。
本轮继续覆盖 **2 类**，共新增 **21 个形状**（ScalarOperation 18 + Squeeze 3），
探针累计 **46 个形状**。

### IC.1 `ScalarOperation`：9 种运算逐个测，重点是**两个反向的**

这一层在 C7 可达性表里是 **UNUSED** —— 仓库里**没有任何模型会跑到它**。
它有 9 种运算，其中两个与"正向"的只差**参数顺序**：

| 运算 | 语义（取自 `zq_cnn_scalaroperation_32f_align_c.c` 的 align0 版） |
|---|---|
| `MUL` / `DIV` / `ADD` / `MINUS` | `x*s` / `x/s` / `x+s` / `x-s` |
| `MAX` / `MIN` / `POW` | `max(x,s)` / `min(x,s)` / `x^s` |
| **`RDIV`** | **`s/x`**（反向除） |
| **`RMINUS`** | **`s-x`**（反向减） |

`RDIV` / `RMINUS` 写反了**没有任何东西会发现** —— 唯一能抓到的是这一层自己的单测。
所以参考实现的自证里专门给它们加了**手算用例**：

```
RDIV  scalar=12, in=[2,4,6] -> 12/x = [6,3,2]
RMINUS scalar=10, in=[2,4,6] -> 10-x = [8,6,4]
```

实测：9 种运算 × 2 个 C（8 = align 的倍数、13 = 非倍数）= **18 个形状全过**
（只有 `DIV` 有 1.5e-8 量级的舍入，其余**逐位 0**）。

### IC.2 `Squeeze`：纯声明层，一个字节都不该动

`ZQ_CNN_Layer_Squeeze::Forward` 只做一次 `CopyData`，不参与计算。
测它是确认"声明层真的没偷偷改数"：C = 1 / 8 / 17 三个形状**逐位 0**。

### 变更文件

* `SamplesZQCNN/SampleUnusedLayerProbe/SampleUnusedLayerProbe.cpp` — 增加
  `ref_scalar_op` / `ref_squeeze`、两组手算自证、`run_scalar_op` / `run_squeeze`。
* `audit_k3_20261001.md`（追加 IC）

**无生产代码改动**；两个平台均已手工重编 + 实跑（rc=0，46/46）。

### 注意事项

1. **第一版把 `for (int C = 0; C < 2; C++)` 当成了"遍历 C 列表"**，
   于是 C=0 那一轮造出一个**零尺寸**张量，`ConvertFromCompactNCHW(&in[0], ...)`
   拿到的是空 vector 的 `&in[0]` —— 9 个用例全部失败，
   而输出里那 9 行显示的是 **C=1** 那一轮（全部通过）。
   > 症状极具欺骗性：**失败的那一轮没有形状标签**，
   > 而通过的那一轮有，于是读的人以为"C=1 全部对、另一个 C 挂了"。
   > 修法：`for (int ci = 0; ci < 2; ci++) { const int C = CS[ci]; ... }`。
   > 判据：**循环变量不要与"被遍历出来的值"同名** ——
   > 两者混用时，失败输出的形状标签会指向另一个值。
2. 尚未覆盖的 UNUSED 还有 **9 类**，输出里逐个列出：
   `DeConvolution` / `BatchNorm` / `LSTM_TF` / `UnaryOperation` / `Tile` /
   `Reduction` / `PriorBoxText` / `PriorBox_MXNET` / `DetectionOutput_MXNET`。

## 新增/变更：ID —— 把「同一个 raw 头里剩下的三个内核」也纳入 ASan 门禁（22 组）

### 起因

IB 修的是 `zq_cnn_scale_32f_align` **带 bias 分支**漏了守卫，
而**不带 bias 的分支早就修过** —— 也就是说作者当初就是**只修了一条分支**。
同一个 raw 头里还有三个做同一件事（`y = b*x + a`）的内核，
必须逐个核对 —— 这是 AGENTS.md「补齐一处修复时把同仓的另一份拷贝列出来」
（附录 HA.3）在**同一个 TU 内**的版本。

人工核对结论（`zq_cnn_batchnormscale_32f_align_c_raw.h`）：

| 函数 | 守卫 |
|---|---|
| `zq_cnn_batchnorm_32f_b_a_align` | 有级联守卫：`% (8*align)` -> `% (4*align)` -> `% (2*align)` -> 标量 |
| `zq_cnn_batchnormscale_32f_mean_var_scale_bias_align` | 标量循环写进 `_aligned_malloc(in_C*sizeof(float))` 的**恰好 in_C 个** float，再交给上面那个有守卫的函数 |
| `zq_cnn_batchnorm_32f_mean_var_align` | 同上 |
| `zq_cnn_scale_32f_align`（不带 bias） | 早就改成标量 |
| `zq_cnn_scale_32f_align`（带 bias） | **原来没有守卫** ← IB 修的就是它 |

结论是"另外三个不需要同样的修复" —— 但那是**读代码**得出的。
ID 把这个结论变成**跑出来的**。

### 门禁扩到 22 组

```
zq_scale        10 组  scale（带 bias / 不带 bias），align 1/4/8，C=3/13/8/16
zq_scale(bn)    12 组  b_a 与 mean_var_scale_bias，align 1/4/8，C=3/13/64/65
```

C=64 是 `8*align` 的整数倍（走向量那一档），C=65 是 `+1`（必须落标量那一档），
**两档都要在**。结果 **22 组全过**（ASan 越界 + 数值）。

变异测试（退回 IB 修复前）：**恰好 3 组红**（Scale 带 bias、C=3/13），
其余全绿 —— **包括 bn 部分 12 组仍然全绿**，
证明两节判据各自独立有鉴别力。

### 这一轮踩到的三个坑（都是判据自己的）

1. **子进程用 `_exit` 不冲刷缓冲** → 子进程打的「值对不上」永远看不到，
   父进程只拿到"非 0 退出"、拿不到原因。补 `fflush(stdout)`。
2. **父进程 fork 之前必须 `setvbuf(_IONBF)`**。否则子进程退出前那次 `fflush`
   会把**父进程缓冲里那一整段**也推出去，于是 `zq_scale：10 组…`
   被**重复打印 12 次** —— 与"跑了 12 遍"长得一模一样
   （仓库里既有规矩：附录 BL.8「崩溃类测试每个用例打一行到 stdout」）。
3. **就地内核必须在调用前把输入快照下来**。`b_a` / `mean_var_scale_bias`
   都是**就地**改 `data`，而参考值要**原始 x**。第一版在调用之后才读 `data[c]`，
   于是 `want = b × 已改过的值 + a`，与 `got` 比**必然全错** ——
   实测相对误差 0.31 / 0.69 / 8.9 / 12.2，形态看着像"内核算错了"。
   > 这一条的形态最有欺骗性：数字**很大**、**随 C 变**、**每个 align 都报**。
   > 而它一次都没跑对过内核的正确路径 —— 只是把输出当成了输入。

### 顺带确认：逐元素运算也必须用后向误差

修好第 3 条后仍有一格红：`mean_var_sc_bias align8 C=65`，相对误差 2.737e-05。
`y = b*x + a` 在 `b*x ≈ -a` 时结果抵消到接近 0，除以 `|结果|` 会把 1e-7 的绝对差
放大成 1e-2。改成分母取**这一格的计算尺度** `|b*x| + |a|`
（AGENTS.md「GEMM 的判据必须用后向误差」—— 这条对逐元素运算同样成立），
同一格降到 ~1e-7。

### 变更文件

* `tools/zq_scale_check.cpp` — 扩到 22 组；加 `setvbuf`、输入快照、后向误差；
  头注释改成说明覆盖范围。
* `tools/run_zqlib_checks.py` — `EXTRA_LINK` 注释补充。
* `audit_k3_20261001.md`（追加 ID）

**无生产代码改动**；`zq_scale` 门禁 PASS，变异测试红/绿各一次。

## 新增/变更：IE —— 再覆盖 `Reduction`（32 组）：axis=0 与 keepdims=0 全对，`keepdims=1` 且 axis∈{1,2,3} 对不上（根因未定位）

### IE.1 先读实现再写参考（又一次）

IA.4 里 `Scale` 的参考把 compact NCHW 的通道下标写反了。
这一次**先把 `zq_cnn_reduction_32f_align_c.c` 的 align0 版读完**再写参考，
当场排除一个看似很大的"缺陷"：

    if (keepdims == 0) { 对全部元素求和，写到 out_data[0] }
    else                { switch (axis) { 沿该轴求和 } }

而 `ReductionSum` 里 `out_dims` 的算法与之一致（keepdims 时 `out_dims[axis]=1`，
否则四个维全置 1）。所以 `axis=2 keepdims=0` **不是缺陷** ——
那一档的语义本来就是"约全部"，`axis` 只是被忽略。两侧自洽。

axis 的编号是 `out_dims[4] = { N, C, H, W }`：0=N 1=C 2=H 3=W。

### IE.2 实测：32 组里 20 组对、12 组对不上

SUM / MEAN × axis {0,1,2,3} × keepdims {0,1} × C {8,13}：

| 组合 | 结果 |
|---|---|
| keepdims=0（约全部） | **全对** |
| keepdims=1, axis=0 | **全对**（N 恒为 1，约 N 是恒等） |
| keepdims=1, axis∈{1,2,3} | **12 组全对不上** |

数值形态（C=8、SUM、axis=1）：
    #0 参考 0.952728 -> 库 2.76187
    #1 参考 -0.993439 -> 库 0.0785828
**值的个数是对的**（axis=1 给 9 个 = H*W），不是形状问题；
而且**库给的 9 个值之和仍等于全部元素之和** ——
它确实把每个元素算了恰好一次，只是**分组方式**不同。

已排除：读 `zq_cnn_reduction_sum_32f_align256bit` 的 case 1/2/3 三支，
循环结构与"沿该轴求和"逐行一致。所以差异不在"约哪个轴"，
而在**写到哪里 / 读回来时的下标对应**。**根因未定位**，如实记成待查项。

### IE.3 这一格只报、不判失败

`Reduction` 是 UNUSED 层，把这一格判失败就是**恒红**，
而恒红会把别的真回归失败淹掉。所以单独计成「待查」，行末标注"不判失败"：

    小结：跑过 66 个形状，对 66，**对不上** 0，**待查** 12
    UNUSED LAYER PROBE OK        rc=0（两个平台）

### IE.4 这一轮踩到的两个坑

1. `report()` 在"形状不同"时越界读：`backward_err` 在长度不等时把 worst 置 -1，
   而 `report()` 直接 `want[worst]` -> **探针自己崩了**（rc=127），
   把真正的形态盖成了"进程挂了"。现在那一支单独打印两个长度并 return。
2. `axis` 编号认错（IA.4 同一族）：第一版把 `axis==0` 也写成 `outC = 1`，
   于是参考给 9 个、库给 72 个，报"形状对不上"。

### v41 回归结果（ID 之后）

    ALL CHECKS PASSED        rc=0        FAILED 计数 = 0

### 变更文件

* `SamplesZQCNN/SampleUnusedLayerProbe/SampleUnusedLayerProbe.cpp` —
  新增 `ref_reduce` / `run_reduction`（32 组）、`report()` 的形状分支、
  「待查」计数；名字改为**先打**再跑（附录 HC.5）
* `audit_k3_20261001.md`（追加 IE）
* **无生产代码改动**；两个平台均已手工重编 + 实跑（rc=0，66/66，12 待查）
