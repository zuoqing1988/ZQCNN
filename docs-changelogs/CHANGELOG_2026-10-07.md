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
