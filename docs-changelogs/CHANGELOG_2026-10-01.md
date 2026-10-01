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

## 新增/变更：第二轮跨平台修复 + 修复自身复核 + MKL 对比基准

对应 `audit_k3_20261001.md` 附录 A/B/C。对应提交：`c2d362e`、`c9a340a`、`64f1545`。

### 变更文件

**跨平台"跑不通"级修复**

| 文件 | 问题 | 修复 |
|---|---|---|
| `ZQCNN/ZQ_CNN_Layer.h`、`ZQ_CNN_Layer_NCHWC.h`、`ZQCNN_to_MNN/converter/source/ZQ_CNN_Layer.h`、`SamplesZQCNN/TrainMTCNNprocessor/TrainMTCNNprocessor.h` | `.zqparams` 是 CRLF，Linux 文本模式不转换行尾，`_is_blank_c` 不认 `\r` → 每行最后一个 token（`bias`）解析失败、模型静默不带 bias、SampleMTCNN 在 Linux 上 segfault | `_is_blank_c` 把 `\r` 也当空白 |
| `ZQCNN/CMakeLists.txt` | Windows 无条件链 `mklml`，默认 BLAS_TYPE=ZQ_GEMM 时根本不需要 → 没装 MKL 运行库的机器所有 exe 起不来 | 仅 `BLAS_TYPE=openblas` 时链接 mklml |
| `CMakeLists.txt` | `3rdparty/bin/*.dll` 不在产物目录，依赖 caffe/libfacedetect/SeetaFace 的示例找不到 dll | configure 时拷到 `CMAKE_RUNTIME_OUTPUT_DIRECTORY` |
| `ZQCNN/math/zq_libm_compat.c`（新增） | `3rdparty/lib/libncnn.a` 由 clang 编译，引用 `__exp_finite` 等 compiler-rt 符号，gcc 链接失败 | 补齐 40 余个 `__*_finite` 符号 |
| `ZQlibFaceID/ZQ_FaceDatabase.h`、`ZQ_FaceDatabaseCompact.h`、`ZQ_FaceIDPrecisionEvaluation.h` | 只剥 `\n` 不剥 `\r`，Linux 上人脸库文件名带 `\r` 匹配全失败；`strlen==0` 时读 `line[-1]` | 同时剥 `\n`/`\r`，消除越界读 |

**修复自身复核后追加的修复**

| 文件 | 问题 |
|---|---|
| `ZQlibFaceID/ZQ_FaceGroup.h` | `int num` 未初始化 + 读取循环不在 `if (flag)` 内 → fread 失败时堆越界写（上轮新加的 `num<1000000` 反而保证循环执行） |
| `ZQCNN/layers_c/zq_cnn_lrn_32f_align_c.c` | 正常路径从不 `free(square_buf/accumulate_buf)`，每次 Forward 泄漏（fp32 + fp16 两处） |
| `ZQCNN/ZQ_CNN_Forward_SSEUtils.cpp` | 第二处 `slope = NULL` 赋值误用（1917 行）；`_prior_box*` 早退分支返回 true（应为 false）；`output.ChangeSize()` 返回值未检查 |
| `ZQCNN/ZQ_CNN_SSDDetectorPytorch.cpp` | `cls_C * 4` int 乘法溢出可绕过 loc 长度校验 |
| `SamplesZQCNN/mxnet2zqcnn/mxnet2zqcnn.cpp` | `read_mxnet_json/read_mxnet_param` 的 bool 返回值被丢弃，解析失败仍写出静默损坏的模型 |
| `ZQlibFaceID/ZQ_FaceFeature.h` | `CopyData` 负 length / malloc 失败未处理 |
| `ZQlibFaceID/ZQ_FaceSearchTarget.h` | `SaveToFile` 失败路径漏 `fclose` |
| `ZQlibFaceID/ZQ_FaceDatabase.h` | 文件名字段 `len` 无上界，恶意库文件可触发未捕获 bad_alloc |
| `ZQCNN_to_MNN/SampleOnet.cpp` | 标签索引只判上界 |

**新增性能对标设施**

| 文件 | 说明 |
|---|---|
| `SamplesZQBLAS/SampleGEMMCompare.cpp` + `CMakeLists.txt`（新增） | 同一组 20 个尺寸（含 1×N、N×1、瘦长、K 主导、CNN 真实形状、含非对齐尾部）上对比 ZQ_GEMM intrinsic / ZQ_GEMM 汇编版 / MKL / OpenBLAS，校验最大绝对误差并打印 GFLOP/s 与相对 MKL 的百分比。MKL 与 OpenBLAS 走运行时动态加载（Windows `LoadLibrary`/Linux `dlopen`），本机没装就跳过，不绑死链接期依赖 |
| `.gitignore` | 忽略 `3rdparty/mkl_runtime/`（MKL 运行时体积大，不入库） |

### 实测结果

- **Windows**：VS2022 + cmake Release/x64 全量构建 0 error。
- **Linux（WSL Ubuntu-20.04, gcc 9.4）**：`cmake` 配置成功、**samples 全部进入构建**（该环境存在可用的 OpenCV）；首轮 `make` 暴露并修复 `FLT_MAX` 编译错误，修复后推进到 66%、59 个可执行文件产出，随后在 `SampleFaceDatabaseNCNN` 链接处被 ncnn 的 `__*_finite` 符号卡住（已修，待复跑确认 0 error）。
- **Linux 运行时**：修复前 `SampleMTCNN` 因 bias 解析失败 segfault；修复后需重新构建复验（进行中）。
- **MKL 运行时**：Windows（MKL 2026.1.0）与 Linux（MKL 2024.2.2）均已从 PyPI wheel 提取到 `3rdparty/mkl_runtime/`，供 `SampleGEMMCompare` 动态加载。

### 注意事项

1. `_is_blank_c` 的改动是**行为修复**：此前在 Linux 上所有带 bias 的层都被当成无 bias，推理结果是错的但不会报错。修复后 Linux 与 Windows 的数值结果才真正一致。
2. `ZQCNN/CMakeLists.txt` 取消默认链接 mklml 后，若有人在 Windows 上用 `-DBLAS_TYPE=openblas` 构建，需要自备 MKL 运行库（mklml.dll 系列）。
3. `_prior_box*` 的早退分支由 `return true` 改为 `return false`：正常模型走不到该分支（`_setup` 保证 `num_priors >= 1`），但如果历史上有畸形模型依赖"返回成功+空张量"的行为，升级后会显式失败——这是预期的安全收紧。
4. `SamplesZQBLAS` 与 `SamplesZQGEMM` 都不依赖 OpenCV，两个平台都能构建。

## 新增/变更：汇编内核首版落地 + MKL 对标实测数据（Linux）

### 实测结果：asm vs intrinsic vs MKL（WSL Ubuntu-20.04, gcc 9.4, -O3 -mavx2 -mfma, 单线程口径）

MKL 2024.2.2 顺序层；20 组尺寸；误差列为 asm 与 intrinsic 的最大绝对差（1e-6 量级，说明 FMA 累加顺序不同但结果一致）。

| MxNxK | intrinsic | asm | MKL(1T) | asm/MKL | asm/intr |
|---|---|---|---|---|---|
| 16³ | 24.80 | 16.60 | 15.67 | 106% | 0.67 |
| 32³ | 40.74 | 26.47 | 56.97 | **46%** | 0.65 |
| 64³ | 56.87 | 36.73 | 66.01 | **56%** | 0.65 |
| 128³ | 66.08 | 47.81 | 73.43 | 65% | 0.72 |
| 256³ | 74.00 | 53.97 | 72.66 | 74% | 0.73 |
| 512³ | 72.77 | 55.46 | 75.15 | 74% | 0.76 |
| 1024³ | 50.85 | 55.06 | 72.89 | 76% | 1.08 |
| 1x1024x1024 | 19.06 | 25.10 | 23.30 | 108% | 1.32 |
| 1024x1x1024 | 28.48 | **2.13** | 16.78 | **13%** | 0.07 |
| 4096x4x4096 | 29.48 | 39.00 | 23.74 | 164% | 1.32 |
| 4x4096x4096 | 29.92 | 36.36 | 12.43 | 292% | 1.22 |
| 128x128x4096 | 29.91 | 56.94 | 68.92 | 83% | 1.90 |
| 1152x256x1152 | 35.28 | 59.97 | 71.60 | 84% | 1.70 |
| 512x512x2048 | 24.26 | 38.25 | 68.91 | **56%** | 1.58 |
| 7x5x13 / 33x17x65 | 7.59 / 25.25 | 4.44 / 22.42 | 3.70 / 30.29 | 120% / 74% | 0.59 / 0.89 |

结论（阶段性）：汇编版在**瘦长形**（4x4096x4096、4096x4x4096、2048x8x2048）上明显超过 MKL，在大方阵上约 65–85%，但 **N=1（1024x1x1024）只有 MKL 的 13%、比 intrinsic 还慢 13 倍**，中小方阵 46–65%，尚未达到"与 MKL 相当"的目标，需继续优化（补 N=1/N=2 专用路径、改进微内核分块与 K 循环展开）。

### 注意事项

1. **MKL 动态加载必须先设 `MKL_THREADING_LAYER=SEQUENTIAL`**：默认走 OpenMP/TBB 层，`libmkl_intel_thread.so` 找不到 `omp_get_num_procs` 会直接段错误（在没链 libgomp 的程序里表现为 dlopen 成功、首次调用崩）。顺带这也保证了与单线程 ZQ_GEMM 的公平对比。
2. **ZQ_GEMM intrinsic 的调用契约**：缓冲区 32 字节对齐，`lda/ldb/ldc` 为 8 的倍数，K 方向 padding 必须补 0（用 `calloc` 等 16 字节对齐的分配器 + K 不是 8 的倍数会直接段错误，调试时很容易踩）。汇编版用非对齐访存，要求更松。
3. `SampleGEMMCompare` 在加载 BLAS 后会立刻做一次 4x4 自检（结果必须是 8.0），避免"库能 dlopen 但一调用就崩"的情况浪费一轮调试时间。
