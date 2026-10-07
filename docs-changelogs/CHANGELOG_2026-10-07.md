# CHANGELOG 2026-10-07

## 新增：审计 10/1 至今全部改动 —— 修 2 个文档 BUG + 新增 C29（附录 JN）

### 变更文件

    改 audit_k3_20261001.md        重编 21 个重号附录 + 同步 68 处正文引用
                                   补回丢失的附录 JG + 修一处粘连分隔线
    新增 tools/build_audit_index.py
    新增 audit_k3_20261001_index.md     生成的索引（255 附录）
    改 tools/run_audit_checks.py        接入 C29 / C29b，A/C 组门禁 78 -> 80

### 起因与发现

用户要求审计 10/1 至今的全部改动。10/1 至今引擎侧共 86 文件 +10348 行。
发现的两个 BUG 都在审计报告自己身上：

**① 21 个附录编号重号**：IB~IY 各出现两次（21038 起第一批"36 层零覆盖"
批次；23396 起我又从 IB 重新起号；25070 起又从 IJ 重新起号，IM 没重）。
后果与门禁编号撞名同类（附录 IR）——「查附录 IB」有歧义。
修复：位置感知重编号，第二批换空闲号 JN..KH；正文 68 处引用按区间同步替换；
源码注释里的 IX.5/IX.7 等逐一核实过，指向第一批（保留原号），不受影响。

**② 整个附录 JG 丢失**：commit fbe375c 只写了 changelog，
报告里 JF 之后直接是 JH。已按提交信息补回整节。

顺带修了一处排版：JF 末尾分隔线与正文粘在同行（`74 -> 76---`）。

### 防复发

`tools/build_audit_index.py`：
* `--check` 只读报告，重号即退出 1（已接为 C29；C29b 是自测）。
* 不带参数则**生成** `audit_k3_20261001_index.md`（255 附录的
  编号→标题→行号对照，带阴性/门禁标记）。

生成索引时的两个解析坑：
1. `grep '^# '` 会把代码块里的注释当标题 —— 必须跟踪 ``` 围栏；
2. 第一版给 IL 误标「阴性」，根因是**集合语义遇重号**（neg 按编号存 set，
   两个 IL 有一个命中就两个都标）。重编号后自然痊愈。

### 引擎改动抽查结论

对风险最高的几处逐一复核：
* ZQ_CNN_MTCNN.h tensor 重载尺寸校验（附录 IX）—— guard/信息/返回都正确；
* ZQ_CNN_Layer.h 92 处 ConvertFromCompactNCHW 检查（附录 JH）—— 抽查正确；
* 其余各批（ConvertFromBGR 23 处、sample 16 处、4 个 sample 的 recognizer
  Init、Tensor4D_NCHWC 1 处）此前已逐批过编译 + 全量回归（附录 JG/JI/JK/JM）。

**引擎侧没有发现新缺陷。** 10/1 至今的全部改动处于"五次全量回归 0 失败"状态。

### 实测

    build_audit_index.py --selftest   PASS（围栏内 # 不算标题等）
    build_audit_index.py --check      OK: 255 个附录编号唯一      RC=0
    旧报告(21 重号) --check           报出全部重号                  RC=1
    C29 / C29b 接入验证               INTEGRATION OK（总门禁 80）
    check_text_encoding / line_endings / gates_runnable / gate_ids  全绿

---

## 补记：commit↔附录 双向核对（阴性）+ C29 纳入第六次全量复验（附录 JO）

### commit ↔ 附录 双向核对（10/1 至今）

顺着附录 JN 的活继续核对记录完整性：

* 正向：10/1 以来消息以「附录 X」开头的提交共 **139 个**，报告里
  「## 附录 X」标题缺失数 **0**（JG 是最后一个缺口，JN 已补）。
* 反向：消息里没有「附录」字样的 274 个提交，逐条看都是
  AGENTS.md 增补、全量回归验证提交（v71~v77）、旧式命名修复
  （DN/DL/DJ 等，记录在对应编号下）—— 都有归属，无遗漏修复。

**结论：10/1 至今的改动记录是完整的。**

### C29 纳入第六次全量复验

    python tools/run_audit_checks.py --all
        ALL CHECKS PASSED   RC=0   ELAPSED=5804s（约 97 分钟）
        完成的门禁组数 : 100

    D1 / D2 / D3 双平台构建与 sample 回归                OK
    B  ZQlib 独立回归 x10 (ASan+LSan)                     OK（72 个测试）
    C6 ZQCNN 门禁 UBSan 回归                              OK
    C27 / C28 / C29                                        OK

组数 98 -> 100（C29 / C29b）。本会话新增的门禁至此**全部**纳入过全量复验。
累计六次 --all 全量复验（JE / JG / JI / JK / JM / JO），全部 0 失败。

---

## 修复：两处 IoU 分母无守护除法（附录 JP）

### 变更文件

    改 SamplesZQCNN/TrainMTCNNprocessor/TrainMTCNNprocessor.h
    改 ZQCNN/ZQ_CNN_MTCNN_ncnn.h

### 起因

按心跳清单审 TrainMTCNNprocessor.h（1264 行）。整体**阴性**：解析器守卫齐全
（split_num % 4 == 1、len > 0、缓冲区 len+1、_IOU 下标有界），唯一命中的是
_IOU 的分母无守护 —— Union 的 `area1 + area2 - IOU` 与 Min 的 `__min(area1,
area2)` 在退化框（x1==x2）下为 0，0/0 得 NaN。与 ZQ_CNN_BBoxUtils.h 的
_nms（附录 IJ.2 已修同一公式）同一类。

横向扫描（第 76 条）又命中 ZQ_CNN_MTCNN_ncnn.h:152 —— 检测路径上的 ncnn
变体，正是附录 II.6 记的那个「五份 MTCNN 里已知会漏改」的变体。
ZQ_CNN_VideoFaceDetection_Interface.h:279 是注掉的参考代码，不动。

### 修法与实测

两处同型：`float denom = ...; IOU = denom > 0 ? IOU / denom : 0;`
    wsl make -j8                  0 error
    cmake --build build_x64       RC=0

### 注意事项

TrainMTCNNprocessor 吃的是人工标注，实际触发概率极低 —— 修它的理由是
「与 _nms 同公式就该同写法」，不留两种口径。MTCNN_ncnn.h 当前零 sample
include，但它是公共头且在检测路径上、又是已知漏改体质，顺手补上。

---

## 修复：_aligned_malloc 不查返回值 —— 5 个 GEMM sample ~90 处（附录 JQ）

### 变更文件

    改 SamplesZQCNN/CompareWithOpenBLAS/CompareWithOpenBLAS.cpp   5 处守卫
    改 SamplesZQCNN/SampleMatMul/SampleMatMul.cpp                 2 簇 40 分配
    改 SamplesZQCNN/example_for_very_high_gflops/...              6 簇 12 分配
    改 SamplesZQCNN/testWinoF2233/testWinoF2233.cpp               2 簇 10 分配
    改 SamplesZQCNN/SampleGEPB/SampleGEPB.cpp                     1 簇 6 分配

### 起因

附录 IX 修了 layers_c/*.h 的 42 处同类问题，但只扫了库。这轮按心跳清单扫
sample，发现 CompareWithOpenBLAS(17)、SampleMatMul(40)、
example_for_very_high_gflops(12)、testWinoF2233(10)、SampleGEPB(6)
全没查。全是 GEMM 基准，分配完立刻写缓冲，失败即空指针解引用。
跳过 SampleMatMulNEON/FP16（34 处，#if __ARM_NEON 在两平台都编译成空）。

### 脚本翻车实录

第一版锚点从文件尾倒搜 `q = _aligned_malloc(32, 32);` —— _test_gemv 与
_test_gemm 同名同形都有这行，倒搜撞进 _test_gemm（把含 C 的守卫塞进只有
C1/C2 的函数，C2065）。第二版改用簇行号区间+声明提取变量名，40 处一次通过；
第三版批量 28 处一次通过。

### 实测结果

    cmake --build build_x64 --config Release   RC=0（3 轮，0 error）

---

## 补记：JQ 闭环 —— 第七次 --all 全量复验（附录 JR）

### 起因

附录 JQ 改了 5 个 GEMM sample 的 ~90 处分配守卫，当时只做了 Windows 构建
验证（3 轮 RC=0）。按「新改动纳入全量」的纪律，本轮跑完整 --all 闭环。

### 实测结果

    python tools/run_audit_checks.py --all
        ALL CHECKS PASSED   RC=0   ELAPSED≈111 分钟（15:57 → 17:48）
        完成的门禁组数 : 100（与第六次持平）
        FAIL 行数 : 0

    D1 / D2 / D3 双平台构建与 sample 回归                OK
    B  ZQlib 独立回归 x10 (ASan+LSan)                     OK（72 个测试）
    C6 ZQCNN 门禁 UBSan 回归                              OK
    C24 / C26 / C27 / C28 / C29                           OK

累计七次 --all 全量复验（JE / JG / JI / JK / JM / JO / JR），全部 0 失败。

### 同日其余产出

* AGENTS.md 第 91 条：锚点不唯一时「从尾部倒搜」也是一种猜（JQ 脚本
  翻车实录的教训固化；与第 85 条同族不同变体 —— 精确锚点重复）。
* 审计索引再生成：纳入 JP / JQ 两个新附录（254 附录 + 5 轮次标记 = 259 行）。

### 备注

本轮回归期间核实了一个此前未记录的边界：JQ 改的 5 个 sample 含
`<openblas\cblas.h>` 反斜杠 include，Linux 侧本就不构建它们，
所以「双平台完全跑通」对这 5 个文件的验证面 = Windows 构建 + 全量门禁，
没有 Linux 运行时缺口 —— 这是已知边界，不是遗漏。

---

## 新增：横向扫三个分配/输入类 —— 全阴性（附录 JS）

### 起因

JQ 轮（sample 分配守卫）后，把同族类横向推广到未显式扫过的目录/形态。

### 扫描与结论

| 类 | 范围 | 结论 |
|---|---|---|
| `malloc`/`_aligned_malloc` | `SamplesZQlibFaceID/` 23 个示例 | 0 命中（该目录不用裸分配） |
| `realloc` | ZQCNN / ZQ_GEMM / ZQlibFaceID | 0 命中（全仓不用） |
| `fopen` 未判空 | 库 + sample | 全部带 `if (!fp)`；抽查 mxnet2zqcnn 4 处为「失败即 return false」分支 |
| `sscanf`/`fscanf` `%s` 进定长缓冲 | 库目录 | 无命中（grep 命中全是 printf 格式串） |

### 判读

裸分配与文件输入两个大门类已无已知缺口。下轮方向应转语义类
（如 NCHWC pad_type），不再继续扫形态类。

---

## 修复：选录区 7 项旧账清零 —— 4 真 bug + 3 阴性（附录 JT）

### 起因

审计报告「三、低危与正确性问题（选录）」7 个条目无状态标记，逐一代码核实
（行号已漂移，按内容重定位）。

### 修复（4 项）

* **ZQ_FaceRecognizerSeetaFace.h CropImage 恒真**：`pixFmt == RGB || ZQ_PIXEL_FMT_BGR`
  非零枚举常量使条件恒真，灰度图直调该公开虚函数会 3 倍宽行 memcpy 越读并
  返回 true；补 `pixFmt ==`（全仓同型仅此 1 处）。
* **CompareWithOpenBLAS `_test_im2col` 恒假**：`out_w < out_w` 输出从不写回 +
  `out_row_idx` 不自增；修条件并补自增（参考实现照抄即错）。
* **zq_gemm_32f_align_c.c `align128it` 9 行**：错名全仓无定义（潜在链接炸弹）。
  9 行分三块、各叠第二层错：128bit 块仅拼写；**16f 块前缀也错（32f→16f）**；
  **256bit 块宽度也错（128→256）**。第一版全局 replace_all 直接撞 C2084
  （块 2/3 被并到块 1 符号）—— 按**兄弟行**逐块修才对。
* **convertor.py `scales[0]`**：`scales` 全文件未定义，ResizeBilinear 与
  ResizeNearestNeighbor 两分支 num_int==1 时必 NameError；改 `dst_size[0]`。

### 核实阴性（3 项）

softmax_nchwc = 附录 CK 门禁 15/15 覆盖；dropout 首元素×2 = 附录 CO 已修；
bn_nchwc 符号 = 现行 .c/.h 一致且 D4 运行时覆盖。

### 实测

    wsl  make -j8 ZQCNN                          RC=0
    cmake --build build_x64 --config Release -j8  RC=0
    ast.parse(convertor.py) / 映射块重复目标静态检查  OK

### 教训（已固化方向）

一行里可能叠多层错误；修完一层必须重编译，下一层只有撞了才现身。
批量 replace 撞重复定义时，正确响应不是回滚了事，而是**逐块按兄弟行重定目标**。

### 变更文件

    改 ZQlibFaceID/ZQ_FaceRecognizerSeetaFace.h          补 pixFmt ==（1 行）
    改 SamplesZQCNN/CompareWithOpenBLAS/CompareWithOpenBLAS.cpp  恒假循环 + out_row_idx（2 行）
    改 ZQ_GEMM/math/zq_gemm_32f_align_c.c               9 行映射宏分块重定目标
    改 TensorFlow_to_ZQCNN/convertor.py                 scales[0] -> dst_size[0]（2 处）
    改 audit_k3_20261001.md / 索引 / 本 changelog

---

## 修复：testImageProcessing 5 处 malloc 守卫（附录 JT 补记）

### 起因

心跳清单点名 testImageProcessing「没细审」。该文件此前只被 dup-block 工具
当误报源数过，从未按缺陷类审过。本轮细审（1480 行）：

* **malloc 判空 5 处全缺**（`resize_nn_c1/c2/c3/bgra2rgb/c3_arm32`）——
  JQ/BK.1 类在**最后一个目录**的命中；已补守卫（先断言锚点 5 处、
  已有守卫 0 处，再批量插入）。
* 其余类全阴性：malloc/free 配对 5/5、除零不可达（调用点全为
  字面量 640/360）、边界 clamp 齐全、GUI 已注释、imread 判空。

### 实测

    cmake --build build_x64 --target testImageProcessing   RC=0
        （仅存量 C4244 警告，与本改动无关）

### 备注

本修复在第八次全量回归（附录 JU）启动后落地，JU 验证的是 JT commit；
本改动随下一次全量纳入。

---

## 修复：JQ 类全树收口 —— example 5 处 malloc 守卫（附录 JT 补记 2）

### 起因

JT 补记后用「40 行窗口」重扫 SamplesZQCNN 全树裸分配判空。39 个剩余命中
分两类：NEON/FP16 34 处（已知跳过，维持原判）与
example_for_very_high_gflops 5 处真漏（JQ 簇式修复未覆盖的单点：
`test_{4x4x4,4x4x8,4x4x16,8x8x8,8x8x16}_in_cache` 的 `malloc(4*4/8*8*4)`，
C 随即被 `C[0] += final_sum(q)` 写入）。已补守卫（free A/B + return，
断言 4*4×3 + 8*8×2 后批量插入）。

### 实测

    cmake --build build_x64 --target example_for_very_high_gflops   RC=0

**JQ 类在 SamplesZQCNN 全树至此收口**（除有意跳过的 NEON 34 处）。

### 方法论教训

窗口类探测器「窗口内未见守卫」≠「无守卫」：6 行窗口报 75 处（簇守卫误报），
40 窗口剩 39，人工分类出 5 真漏。真值只能来自更大窗口 + 逐簇人工分类。

---

## 补记：JT 闭环 —— 第八次 --all 全量复验（附录 JU）

    python tools/run_audit_checks.py --all
        ALL CHECKS PASSED   RC=0   ELAPSED≈101 分钟
        完成的门禁组数 : 100    FAIL 行数 : 0

累计八次全量复验全部 0 失败。边界：JT 补记/补记2（10 处 malloc 守卫）
在本轮启动后落地，已各自经目标级构建验证（RC=0），随下次全量纳入。

---

## 变更：AGENTS.md 瘦身 —— 核心规则留下，细则迁入 AGENTS_LESSONS.md（用户指令）

### 变更文件

    改 AGENTS.md           3008 行 -> 186 行（只留核心规则）
    新增 AGENTS_LESSONS.md  2878 行（全部经验细则原样迁移）

### 起因

用户指令：「Agents.md 只保留最核心的规则，把次要内容挪到一个新 md 文件里」。

### 做法

- **AGENTS.md 保留**：变更日志规则（6 条）、构建规则（11 条）、编码与行尾红线
  （四件套 / heredoc / BOM / Python 改写行尾 / raw.h 纯 LF）、全量回归与工具入口、
  经验细则索引、提交规则、示例程序规则。
- **AGENTS_LESSONS.md 接收**：sanitizer 坑、检查工具写法、内核/并行、静态分析、
  汇编规则、行尾细节、张量 API 约定、ZQ_GEMM 布局、NCHW/NCHWC 步长、
  2026-10-02~05 全部专题教训、编号教训 1~92 —— 原样复制，未删改。
- **无损校验**：脚本逐行核对原文件每一非空行都出现在两个产物之一，0 丢失。
- 四件套（text_encoding / line_endings / stmt_joins / gates_runnable）全绿。

### 注意事项

今后新增经验规则请追加到 **AGENTS_LESSONS.md** 末尾；
只有每次会话必读的核心流程规则才进 AGENTS.md。

---

## 订正：两处「部分修复」内存泄漏实为已修完（附录 JV）

### 起因

报告选录区标 `zq_cnn_lrn_32f_align_c.c` 与 `zq_cnn_lstm_32f_align_c` 的
内存泄漏为「🔶 部分修复」。逐行核实现状：LSTM 9 个 malloc 失败/成功
双路径全 free（316/323、410-418 行）；LRN 两个变体同样双路径双 free。
**两处均无泄漏，旧标记作废**，选录区内存泄漏一行全 ✅。

---

## 订正：早期高危表的 🔶/❌ 是历史快照，抽查四项全已修（附录 JW）

报告第一轮高危表（L230~L318）残留 18 个未闭环标记，但该表是**第一轮当天
快照**（L1024 有说明）。本轮抽查四项：`slope = NULL`（全库 0 命中）、
detection_output `find()==end()` 解引用（已改 continue）、
SSDDetectorPytorch aspect_ratios 上限（现行 :586-595 有校验）、
loc blob 校验（附录 H6 ✅）——**全部已修**。

**固化口径**：早期表的 🔶/❌ 一律不作未闭环依据；判未闭环只看
后续附录修复记录 + 代码现状。未来心跳轮不必再扫这批历史标记。
