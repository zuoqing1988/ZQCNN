# CHANGELOG 2026-10-01

## 新增/变更：审计报告 ❌ 未修项全部修复 + 前序 🔶 项逐条核实

对应 `audit_k3_20261001.md` 第六章「遗留工作清单」第 1、2 项。提交 `a75a4e8`。

### 变更文件

**本轮新增修复（审计报告标记 ❌ 未修复的全部条目）**

| 文件 | 漏洞 | 修复要点 |
|---|---|---|
| `ZQCNN/ZQ_CNN_Forward_SSEUtils.cpp` | H5 `_prior_box` | 新增一致性守卫：按 `min_sizes`/`max_sizes`/`aspect_ratios` 实际算出 `priors_per_cell`，与来自不可信 `.zqparams` 的 `num_priors` 不符即 `return false`；`min_sizes`/`max_sizes` 必须 1:1 配对（否则 max 分支越界）；`dim` 用 `long long` 计算并做 64 位溢出守卫 |
| `ZQCNN/ZQ_CNN_Forward_SSEUtils.cpp` | H5 `_prior_box_text` | 同上。该变体每个 `min_size` 产出 2 个框（`center_y` 与 `center_y_offset_1`），`priors_per_cell` 已相应 ×2 |
| `ZQCNN/ZQ_CNN_Forward_SSEUtils.cpp` | H5 `_prior_box_MXNET` | 同上，并额外拒绝空 `sizes`（原代码无条件读 `sizes[0]`）；期望计数 = `num_sizes + (num_ratios - 1)` |
| `ZQCNN/ZQ_CNN_SSDDetectorPytorch.cpp` | H6 | `aspect_ratios` 写入固定数组 `SSDSpec::aspect_ratios[16]` 前，用 `sizeof` 计算容量并校验个数，超限返回失败（原为硬编码假设） |
| `ZQCNN/ZQ_CNN_SSDDetectorPytorch.cpp` | H6 | 新增 `loc` blob 校验：元素数须 `>= cls_C*4`；`cls`/`loc` 各维须为正；`H*W*C` 乘法须不溢出 |
| `ZQCNN/ZQ_CNN_Forward_SSEUtils.cpp` | 中危 | `_detection_output_MXNET` 补齐锚点一致性校验：`loc_len`/`prior_len` 须 `>= num_anchors*4`，`conf_len` 须 `>= num_anchors*num_classes`。原代码按 `offset = i*4` 索引，三个张量任一偏小即越界读 |
| `ZQCNN/ZQ_CNN_Forward_SSEUtils.cpp` | 中危 | `else if (slope = NULL)` → `else if (slope == NULL)`。原写法是赋值误用，会把 `slope` 置空并使 prelu 分支永远不可达 |

**本轮核实（审计报告标记 🔶 部分修复，经逐条 git diff 确认无需返工）**

| 项 | 文件 | 核实结论 |
|---|---|---|
| H3 | `ZQCNN/ZQ_CNN_Tensor4D.cpp` | 三处 `ChangeSize` 均提升到 `__int64` 运算 + `0x7FFFFFFF` 溢出守卫 + 负维度拒绝，回写成员时再转 `int` |
| H4 | `ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp` | 同上，NCHWC 特有的 `imStep` 也纳入守卫 |
| H7-H9 | `ZQ_CNN_MTCNN.h` / `_Interface.h` / `_NCHWC.h` | `keypoint_num = __min(106, keyPoint->GetC()/2)` 限幅正确，各 2 处 |
| H10 | `ZQ_CNN_VideoFaceDetection_Interface.h` | 3 处限幅 + 补 `hpg` 元素数 `>= 9` 校验 |
| H11 | `ZQ_CNN_PersonPose.h` | `num_points = __min(18, hm_C)` + `sliceStep < 7` 校验 |
| H12 | `layers_c/zq_cnn_batchnormscale_32f_align_c.c` | 4 处外层循环 `n < in_C` 全部改为 `n < in_N` |
| H13 | 同上 | `bias_data != NULL` 分支正确加上 bias，写反的 else 分支已修正 |
| H14 | `layers_c/zq_cnn_resize_32f_align_c.c` | FP16 段原为 `float* sx = (zq_base_type*)malloc(sizeof(zq_base_type)*out_W)`，指针按 4 字节解引用而只分配 2 字节（确定性 2× 堆溢出），现声明改为 `zq_base_type*`，类型与分配一致；`coord_x`/`x0_f` 亦为 fp16，语义正确 |
| H15 | `layers_nchwc/zq_cnn_eltwise_nchwc_raw.h` | `eltwise_mul` 通道循环增量原误加到 `in_im_ptr`/`out_im_ptr`（破坏外层 n 游标并在 C 循环中越界），现改为 `*_slice_ptr += *_sliceStep` |
| H16 | 同上 | `eltwise_sum` 中 `tensor_id >= 2` 分支的 n 循环游标已从错误的 `out_slice_ptr` 修正为 `out_im_ptr` |
| H21 | `SamplesZQlibFaceID/SampleFaceDatabase*`（4 个变体） | `system("mkdir " + dst)` 改为 `_mkdir()` / `mkdir(path, 0755)` API，彻底消除 shell 注入面 |
| H22 | `SamplesZQCNN/mxnet2zqcnn/mxnet2zqcnn.cpp` | 逐字段 `fread` 返回值校验 + `ndim > 4` 上限 |
| H24 | `SamplesZQCNN/SampleMTCNNfromlist/SampleMTCNNfromlist.cpp` | `sscanf` 加 `%511s` 宽度限制 + 拼接前长度检查，超长路径跳过而非溢出 |
| H25 | `SamplesZQCNN/TrainMTCNNprocessor/TrainMTCNNprocessor.h` | 8 处 `sprintf` 改 `snprintf` 并检查返回值，超长直接报错返回 |
| 中危 | `layers_c/zq_cnn_pooling_32f_align_c.c` | 4 处补 `stride_H <= 0 \|\| stride_W <= 0` 除零守卫 |
| 中危 | `layers_c/zq_cnn_innerproduct_gemm_32f_align_c_raw.h` | `_aligned_free(matrix_C)` 加 `need_allocate_tmp_out` 保护，避免释放调用方张量 |
| 中危 | `ZQCNN_to_MNN/SampleOnet.cpp` | `net`/`session`/`input`/`output`/`shape` 全部判空 |
| 中危 | `SampleDetectOutliersInFaceDatabase.cpp` | Init 失败路径 `delete` 后置 NULL 并 `return`，消除二次释放 |

### 实测结果

**Windows VS2022 + MSBuild 全量构建（`ZQCNN.sln` / Release / x64）**

- 构建前基线：43 个未提交改动文件 → **0 错误**，4692 警告
- 本轮 ❌ 项修复后单独重建 `ZQCNN.vcxproj` → **0 错误**

4692 个警告全部为既有的类型截断类（`C4244`/`C4267`/`C4305`，float↔int、size_t→int、double→float），非本轮引入，也不影响正确性。

关键修复的触发条件说明（均为不可信模型/配置文件可控）：

- H5：恶意 `.zqparams` 中 `num_priors` 写小、`min_sizes` 写大即可让写入循环越界。常规推理不会命中，故为静默高危。
- H14：仅 ARM + `__ARM_NEON_FP16` 构建受影响，Windows/x86 构建不编译该段。
- H21：`person_name` 取自不可信 list 文件，含 `;` `&` `|` 即任意命令执行，是审计中唯一的「不可信文件 → RCE」路径。

### 注意事项

1. `_prior_box` 系列现在对 `num_priors` 与 `min/max/aspect_ratios` 的一致性做**硬校验**，配置不自洽时返回 `false` 而非静默产出错误 prior。这是行为变更：若某些历史模型的 `.zqparams` 本身就存在数量不匹配（靠越界写"蒙对"的），升级后会从「结果可能错误」变为「明确失败」。这是预期的安全收紧。
2. `_prior_box_text` 的 `priors_per_cell` 比 `_prior_box` 多一倍（每 min_size 两条框），修改该逻辑时勿照搬 `_prior_box` 的公式。
3. H6 的容量校验用 `sizeof(SSDSpec::aspect_ratios)/sizeof(float)` 表达，若日后调整数组大小，校验会自动跟随。
4. `_detection_output_MXNET` 新增的锚点一致性校验会拒绝 loc/prior/conf 三者维度不匹配的组合；这类模型原本会在推理中越界读。
5. 本轮未触及 ARM/FP16、NEON、AVX 专用代码路径的**运行时**验证（Windows 构建不编译这些分支），H14/H15/H16 及 ARM 路径修复的正确性依据是代码审查与 Windows 侧编译通过，需在 WSL/ARM 目标上补运行时验证。

## 新增/变更：移除过时 VS 工程文件 + Linux 首个真实编译错误修复 + AGENTS.md 补充规则

对应 `audit_k3_20261001.md` 第六章「遗留工作清单」第 7 项，并启动第 8 项（Linux 构建验证）。

### 变更文件

**删除（208 个文件）**

- `ZQCNN.sln`、`ZQlibFaceID.sln`
- `ZQCNN/*.vcxproj*`、`ZQ_GEMM/*.vcxproj*`、`ZQlibFaceID/*.vcxproj*`
- `SamplesZQCNN/*/*.vcxproj*`、`SamplesZQlibFaceID/*/*.vcxproj*`
- 保留 `3rdparty/` 下的第三方工程文件不动

**修改**

| 文件 | 变更 |
|---|---|
| `ZQCNN/ZQ_CNN_SSDDetectorPytorch.cpp` | 补 `#include <cfloat>`（gcc 9 下 `FLT_MAX` 未声明，Linux 构建首个真实编译错误） |
| `README.md` / `README_en.md` | 3 处「打开 XXX.sln」的历史描述改为「用 CMake 构建 / 示例在 SamplesXXX 目录下」 |
| `build-with-cmake.md` | 开头声明 CMake 为唯一构建入口；Windows 示例从 VS2015(`Visual Studio 14 Win64`) 更新为 VS2022(`Visual Studio 17 2022 -A x64`)；补充产物目录与 OpenCV 回退说明 |
| `AGENTS.md` | 新增「构建规则 / 提交规则 / 示例程序规则」三节：CMake 唯一入口、双平台必须都编译、不得依赖 MSVC 传递包含、file(GLOB) 需重新 configure、阶段性 commit 且不 push、Sample 中 namedWindow/imshow/waitKey 一律注释 |

### 实测结果

- 提交 `74ce619`（删除 + 文档）、本节修复待随下一次构建验证一并提交。
- **Windows**：VS2022 + cmake Release/x64 全量构建 **0 error**，产物 65+ exe（第二批安全修复后已复验）。
- **Linux（WSL Ubuntu-20.04, gcc 9.4, cmake 3.16）**：`cmake /mnt/d/ZQCNN` 配置 **成功**（意外发现该环境存在可用的 OpenCV，故 samples 全部进入构建，未走「跳过 samples」回退路径）；首次 `make` 在 5% 处因 `FLT_MAX` 失败，补 `<cfloat>` 后继续构建中，其余目标 0 错误。

### 注意事项

1. 删除 `.vcxproj` 后，Windows 用户不能再双击 sln 打开工程，必须用 `cmake -S . -B build_x64 -G"Visual Studio 17 2022" -A x64` 生成 IDE 工程（Visual Studio 会把该目录作为解决方案打开）。这是用户明确授权的取舍。
2. WSL 下 OpenCV 存在但 `pkg-config`/`/usr/include/opencv4` 查不到（可能装在非标准前缀或来自自定义 `OpenCVConfig.cmake`），后续若要长期维护 Linux 构建，建议记录其来源。
3. `ZQ_CNN_SSDDetectorPytorch.cpp` 的 `FLT_MAX` 是既有代码而非本轮引入，只是新加的 `_softmax` 路径在 Windows 下不暴露该问题——典型的「MSVC 通过 ≠ gcc 通过」案例，已写入 AGENTS.md 规则。
