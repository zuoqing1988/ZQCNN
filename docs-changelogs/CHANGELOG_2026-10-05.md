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
