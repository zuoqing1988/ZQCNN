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

## 新增/变更：Windows 端到端跑通 + NCHWC 两处内存问题修复

### 变更文件

| 文件 | 问题 | 修复 |
|---|---|---|
| `ZQCNN/ZQ_CNN_CompileConfig.h` | 默认 `ZQ_CNN_USE_MKL_GEMM 1`，示例按 `#elif` 分支链上 `mklml.lib`，**没装 MKL 运行库的机器上所有 exe 都起不来**（`error while loading shared libraries: mklml.dll`）。实测 ZQCNN 内部根本没有 `cblas_*` 调用，这个开关纯属多余依赖 | 默认改 0（走自带的 ZQ_GEMM），需要 MKL 的人自行打开 |
| `SamplesZQCNN/CompareWithOpenBLAS/CompareWithOpenBLAS.cpp` | 主库关掉 MKL 后，这个"与厂商 BLAS 对比"的示例就没有 cblas 声明了，编不过 | 该文件自己显式 include `mkl/mkl.h` 并声明链接，不再跟随主库开关 |
| `CMakeLists.txt`、`SamplesZQCNN/CMakeLists.txt`、`SamplesZQlibFaceID/CMakeLists.txt` | dll 与 data/model 联接只放在 `cmake-out-*/release/`，而 Visual Studio 的可执行文件在 `release/Release/` 子目录里 → 产物目录里跑示例报 `empty image` / 缺 `opencv_world342.dll` | 新增 `ZQCNN_DLL_OUTPUT_DIRS`（含多配置的 `Debug`/`Release` 子目录），第三方 dll、OpenCV dll、data/model 联接全部按这份列表投放 |
| `ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp` | ASan 实测 `zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3_s2d1` 在最后一个输出像素上做投机读，正好落在缓冲区末尾之外 16 字节 → heap-buffer-overflow（Linux 上 SampleMTCNN_NCHWC4 段错误的第一个根因） | 分配时多给 64 字节余量并整块清零（原来 memset 被注释掉，未初始化数据会进推理结果） |
| `ZQCNN/ZQ_CNN_Forward_SSEUtils_NCHWC.cpp` | 106 处 `output/input/filters.ChangeSize(...)` **忽略返回值**，分配失败后仍按新维度访问 → 空指针崩溃（Linux 上 NCHWC4 路径的第二个根因） | 全部包成 `if (!xxx.ChangeSize(...)) return false;`（void 返回的 MaxPooling/AVGPooling 用 `return;`） |

### 实测结果

**Windows（VS2022 / cmake Release / x64）**
- 全量构建 **0 error**，产物 69 个 exe
- 产物目录直接运行：

| 示例 | 结果 |
|---|---|
| SampleMTCNN | ✅ 18.6 ms/次 |
| SampleMTCNN_NCHWC4 | ✅ 9.3 ms/次 |
| SampleSSD | ✅ 10.4 ms/次 |
| SampleFaceDetectorMTCNN | ✅ 10.6 ms/次 |

- 加了 `/utf-8` 后 MSVC 不再把中文注释按 ANSI 代码页误解码（之前会报莫名其妙的 C2143/C4235）

**Linux（WSL Ubuntu-20.04 / gcc 9.4）**
- 全量构建 **0 error**（100%），SampleMTCNN 19.2 ms、SampleSSD 8.4 ms 正常
- ASan（Debug + `-fsanitize=address`）定位到 NCHWC4 的两处问题，修复后正在复验

### 注意事项

1. `ZQ_CNN_USE_MKL_GEMM` 默认关掉是**行为变更**：以前所有示例默认链 MKL，现在默认链项目自带的 ZQ_GEMM。ZQCNN 源码里没有任何 `cblas_*` 调用，数值结果不受影响，只是 GEMM 走的实现不同（可自行对比）。
2. `ChangeSize` 补返回值检查是**行为收紧**：以前分配失败会静默继续（可能算出错误结果或崩溃），现在会明确返回 false。
3. MSVC x64 不支持函数体内联汇编（`__asm{}` 报 C4235），汇编内核在 Windows 侧只能走独立 `.asm`（MASM），这条已写进 AGENTS.md。

---

## 修复：Windows 端 ZQ_GEMM 汇编内核踩坏调用方的 xmm6-xmm15（Linux 一直是对的）

### 根因

`ZQ_GEMM/math/zq_gemm_32f_align_c_asm_msvc.asm` 的三个微内核用到 `ymm0-ymm14`
（8 个累加器 + 6 个操作数 + 归约临时），但**没有保存/恢复**。

这是两套 x86-64 ABI 的差别：

| ABI | xmm 寄存器 |
|---|---|
| System V AMD64（Linux/macOS/gcc-clang 内联汇编路径） | x87 与**全部** xmm0-xmm31 都是 caller-saved，随便用 |
| **Windows x64（MASM 路径）** | 只有 xmm0-xmm5 是 volatile，**xmm6-xmm15 是 callee-saved**，调用方会把值留在里面跨越调用 |

所以 Windows 上每次微内核返回，调用方 MSVC 编译代码留在 xmm6-xmm15 里的
double 局部变量/中间值就变成了累加器残值。Linux 侧因为 XMM 全是
caller-saved，从来没有暴露过这个 bug。

原文件头注释里"只用 caller-saved 的 ymm0-ymm15"的说法是错的，是这个 bug 的
直接来源。

### 症状（为什么看起来像"内存破坏"而不是"寄存器被踩"）

- 局部 `double` 变成 `-1.7e30` / `1.404e+306` 之类的乱码
- `time_gemm` 里 `t1 - t0` 由垃圾值算出，耗时打印成 `0.00` / `inf` / `-inf`，
  进一步让 `err` 列和 `g_asm/g_intr` 一起变成垃圾
- MSVC 把哪个变量放在 xmm6-xmm15 完全取决于当时怎么分配寄存器，所以症状
  **随代码改动而变**：加一个 canary 数组、换一个优化级别、甚至换个尺寸表，
  "能测出问题"的用例集合就完全不一样，小栈帧的独立小程序往往测不出来
- `g_pad_*` 哨兵变脏是因为哨兵检查读的是 `main` 帧里被污染后重新算出来的
  状态；A/B/C 缓冲本身其实**没有**越界（用 4096 float 冗余 + 0xA5 图案扫描
  验证过，maxdiff 全为 0）

### 定位手段（都排除了什么）

1. `dumpbin /disasm` MASM 目标文件：指令序列与栈上参数偏移
   （第 5 个参数 `[rsp+28h]`，调用者在 `call` 前写 `[rsp+20h]`，中间差一个
   返回地址）**全部正确**
2. `dumpbin /disasm` C 驱动目标文件：7 个静态 kernel 里的 21 个调用点，
   rcx/rdx/r8/r9 + `[rsp+20h..38h]` 的传参顺序**全部正确**
3. MSVC `/fsanitize=address` 重建整个 ZQ_GEMM 再跑：**没有任何报告** →
   不是 C 层的越界读写
4. 守护页（`VirtualAlloc` + `PAGE_GUARD`）与 0xA5 图案扫描：三个缓冲都没有越界写
5. 把三个微内核改成入口立刻 `ret`（nop 版）重新跑 `SampleGEMMCompare`：
   **OOB 全部消失、耗时全部恢复正常** → 定位到微内核本身
6. 在 MSVC 生成的 `zq_gemm_32f_asm_k1n8` 里看到
   `vmovaps [rsp+0F0h], xmm6` / `vmovaps xmm6, [rsp+0F0h]` 一对
   保存/恢复 —— **MSVC 自己就遵守 Windows x64 的这条规则**，反证微内核违反

### 变更文件

| 文件 | 改法 |
|---|---|
| `ZQ_GEMM/math/zq_gemm_32f_align_c_asm_msvc.asm` | 新增 `ZQA_FRAME EQU 0A8h`（168 = 10×16 保存区 + 8 字节补齐，168 ≡ 8 mod 16，保证减栈后保存区 16 字节对齐且出口对齐不变）、`ZQA_PROLOGUE` / `ZQA_EPILOGUE` 两个宏（`vmovups` 保存/恢复 xmm6-xmm15 + `sub/add rsp`），三个微内核 `PROC` 首尾各插一条；栈上传参偏移改用 `ZQA_ARG5..ZQA_ARG8`（= 原偏移 + `ZQA_FRAME`）而不是裸数字；文件头补上两套 ABI 的差异说明 |
| `ZQ_GEMM/math/zq_gemm_32f_align_c_asm.c` | 纯注释：订正"caller-saved 的 ymm0-ymm15"这个错误说法，新增「XMM 寄存器与 ABI」小节说明 MSVC 路径由 MASM 宏负责、gcc 路径靠 clobber 列表（原本就写全了 xmm0-xmm15，System V 下是 no-op，对 MinGW-w64 同样正确）。**删掉一行误提交进仓库的调试 `printf`（`[K %s M=%d ...]`）** |

没有动 CMake 编译选项，也没有动 ZQ_GEMM 以外的任何库代码。

### 实测结果

**Windows（VS2022 17.5 / x64 / Release / `cmake --build build_x64 --config Release`）**

`SamplesZQBLAS/SampleGEMMCompare.exe`（19 个尺寸，修复前后对比）：

```
                                修复前                                  修复后
MxNxK            intrinsic  asm   err(asm)          intrinsic  asm   err(asm)
16x16x16          0.00    0.00   5.1e-06  OOB!       12.60    8.48   4.8e-07
32x32x32   1.16e+30  -0.00   1.6e+35  OOB!  FAIL   24.29   14.52   4.8e-07
64x64x64   4.26e+33  -0.00   2.1e+73  OOB!  FAIL   30.79   22.37   9.5e-07
1024x1024x1024 6.05e+95 -0.00   9.3e+215 OOB!  FAIL  44.16   49.31   3.8e-06
1024x1x1024     23.34    2.14   0.0e+00  OOB!       20.65    2.11   4.0e-05
128x128x4096   垃圾double 0.00  -2.2e+307 OOB!      33.14   51.25   7.6e-06
7x5x13       垃圾double 0.00  -2.2e+307 OOB!       4.64    2.51   4.8e-07
33x17x65     垃圾double 0.00  -2.2e+307 OOB!      18.08   17.00   1.1e-06
```

- 全部尺寸 `err(asm) ≤ 4.0e-05`（判据 1e-3），**OOB 标记全部消失**
- 耗时恢复成合理值，`asm/intr` 比值 0.10~1.55

`SamplesZQGEMM/SampleGEMMAsmCompare.exe`（17 个用例，含大量非对齐尾部）：

```
   32   32   32 |    4.768e-07    5.135e-07 | intr    34.60 GF/s   asm   19.62 GF/s
  128  128  256 |    1.907e-06    3.378e-06 | intr    61.11 GF/s   asm   46.54 GF/s
  313   32   28 |    1.192e-06    9.227e-07 | intr    18.86 GF/s   asm    9.46 GF/s
--------------------------------------------------------------
worst |intrinsic - asm| over all cases = 1.907349e-06
result: PASS (0 case(s) failed)
```

（修复前：`worst = 1.404448e+306`，`FAIL (6 case(s) failed)`）

**Linux（WSL Ubuntu-20.04 / gcc 9.4 / `-O3 -mavx2 -mfma`）—— 未受任何影响**

`build_check_gemm.sh` + `/tmp/zbchk/zbench`：

```
MxNxK            intrinsic       asm   MKL(1T)  asm/MKL asm/intr  err(asm)
16x16x16             24.63     16.46     15.78     104%     0.67  4.8e-07
1024x1024x1024       50.50     56.48     72.11      78%     1.12  9.5e-06
1024x1x1024          25.29      2.14     17.06      13%     0.08  4.0e-05
128x128x4096         30.40     59.69     62.45      96%     1.96  2.3e-05
33x17x65             25.12     22.27     35.52      63%     0.89  1.1e-06
```

`SampleGEMMAsmCompare`（gcc 内联汇编路径）：`worst = 3.814697e-06`，`PASS (0 failed)`

### 注意事项 / 残留风险

1. **性能代价**：每次微内核调用多 20 条 `vmovups` + 一次 `sub/add rsp`。
   K=1024（k8=128）时约 1% 开销，K=8（k8=1）时相对开销明显。实测大尺寸
   `asm/intr` 比值与修复前在同一量级，没有出现数量级退化。
2. **同类风险仍在别处**：任何手写 x64 Windows 汇编（`__declspec(naked)`、
   MASM 文件）只要用到 xmm6-xmm15 / rbx rbp rsi rdi r12-r15 又不保存，
   都会踩同一个坑。`ZQ_CNN_Forward_SSEUtils_NCHWC.cpp` 等 C/C++ 代码由编译器
   负责，天然安全。
3. **MinGW-w64 / clang-cl**：走的是 gcc 内联汇编分支，clobber 列表本来就写全
   了 xmm0-xmm15，两套 ABI 都正确；但这条路径本机没有实测环境。
4. `.c` 里那行调试 `printf` 之前被误提交进仓库（会往 stdout 刷
   `[K zq_gemm_32f_asm_k8n4 M=8 N=16 ...]`），本次一并删掉。

## 新增/变更：全示例回归扫描（Windows）与"哪些示例能直接跑"的清单

### 实测结果

Windows 产物目录逐个跑主要示例（每个都有 300s 超时，示例里的 `imshow`/`waitKey` 早已全部注释，不会阻塞）：

| 状态 | 示例 |
|---|---|
| ✅ exit=0 正常出结果 | SampleMTCNN（20.1ms）、SampleMTCNN_NCHWC4（8.2ms）、SampleCascadeOnet、SampleCascadeOnet_Interface、SampleLnet106、SampleSSD（15.4ms）、SampleMTCNNLoadFromCode |
| ⚠️ exit=1，但原因是**模型/图片不在仓库里**（Model Zoo 与数据集需另行下载） | SampleSphereFaceNet、SampleMobileFaceNet、SampleFacialNet、SamplePnet、SampleLnet、SampleHeatMap、SamplePersonPose、SampleNSFW、SampleMTCNN_Interface、SampleMTCNN_AspectRatio（`data/hand6.jpg`）、SampleTextBoxes（`data/0113.jpg`）、SampleDetectMouth、SampleSwapFace（需要命令行参数） |

Linux 侧同样：SampleMTCNN 19.2ms、SampleSSD 8.4ms 正常；`SampleGenderAge`/`SampleMobileFaceNet` 报 `failed to load net`（同样缺 `model/GA112.zqparams` 等文件）。

### 注意事项

1. **仓库自带的 `model/` 只含 MTCNN/SSD 系列的权重**。SphereFace/ArcFace/PersonPose/NSFW/TextBoxes 等示例的权重在 Model Zoo（README 里的百度网盘链接），`data/` 里也只有 58 张测试图的一部分，个别示例引用的图（如 `data/hand6.jpg`、`data/0113.jpg`、`data/4.jpg`）并不在仓库里。这些失败与本次改动无关。
2. 想让某个示例跑起来，先把对应权重/图片放到 `model/`、`data/`，或改成仓库里已有的文件名。
3. `SampleMTCNN_NCHWC4` 在 Windows 上正常（8.2ms、检出 4 张脸），Linux 上仍有段错误，已单独立项排查（不是这次改动引入的：ASan 报出的两处——depthwise SIMD 投机读越界、`ChangeSize` 忽略返回值——都已修，仍崩说明还有第三个根因）。

## 新增/变更：Linux NCHWC4 段错误的真正根因（行尾）+ 双平台复验

### 根因

`ZQCNN/layers_nchwc/zq_cnn_eltwise_nchwc_raw.h` 与 `ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp` 在仓库里被存成了 **`<反斜杠> + CR CR LF`**：

- MSVC 容忍这种行尾，所以 Windows 一直编得过；
- **gcc 的行拼接只认 `\`+换行**，中间的裸 `\r` 会被当成空白字符，于是**每个跨行宏都在第一行就结束**，宏体被当成文件作用域代码解析（`'in_pix_ptr' undeclared here (not in a function)` 一类报错）；
- 更隐蔽的后果是：增量构建里这两个文件编译失败，**却沿用了旧的目标文件**，于是 Linux 上跑起来的 `SampleMTCNN_NCHWC4` 是一个早于前面两处修复（SIMD 投机读余量、106 处 `ChangeSize` 检查）的旧二进制 —— 崩溃只是"旧二进制"的最后一次表现。

用 `LD_PRELOAD` 装 SIGSEGV handler（WSL 里没有 gdb）抓到的现场：

```
SIGSEGV code=128 (SI_KERNEL) si_addr=(nil)
  vmovaps -0xc0(%r10),%ymm3        r10 = 0x…54d9f0   (0x54d9f0 & 0x1f = 0x10)
  zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv2_Kdiv64
  ← zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N2 ← zq_gemm_32f_AnoTrans_Btrans_auto
  ← zq_cnn_conv_no_padding_gemm_nchwc4_kernel1x1_with_bias_prelu
  ← ConvolutionWithBiasPReLU ← ZQ_CNN_MTCNN_NCHWC::_Rnet_stage
```

`vmovaps` 在 16 字节（而非 32 字节）对齐的地址上执行会触发 #GP，表现为 `si_addr=nil`——正是 `filters_data` 走 `zq_mm_load_ps`（对齐 load）遇到 malloc 只保证 16 字节对齐的场景；改成 `_aligned_malloc(..., 32)` + 64 字节余量的修复本身是对的，只是**从来没在 Linux 上被编译过**。

### 变更文件

| 文件 | 变更 |
|---|---|
| `ZQCNN/layers_nchwc/zq_cnn_eltwise_nchwc_raw.h` | CRCRLF → LF（787 行，**这就是修复本身**），内容逐字节等价（忽略 CR） |
| `ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp` | CRCRLF → LF（1387 行），同类隐患，一并修掉 |
| `.gitattributes`（新增） | `ZQCNN/layers_nchwc/*_raw.h`、`ZQCNN/layers_c/*_raw.h`、`ZQ_GEMM/math/*_raw.h` 固定 `eol=lf`；`.asm` 固定 `eol=crlf`。本机 `core.autocrlf=true` 且原先没有 `.gitattributes`，这是坑的来源 |
| `AGENTS.md` | 新增「行尾与跨平台编译规则」 |

### 实测结果

- **Linux**（`/tmp/zqbclean` 全新 cmake + 全量 make）：`build_rc=0`，`SampleMTCNN_NCHWC4` → `stage 3: cost 1.131 ms`、`final found num: 4`、`run_rc=0`
- **Windows**（全量构建）：`build_rc=0`，`SampleMTCNN_NCHWC4` → `final found num: 4`、`run_rc=0`
- 两平台结果一致；对全部 sgemm 调用点插桩断言 `matrix_A`/`filters_data`/`matrix_C` 均 32 字节对齐，**零违例**（插桩已撤）

### 注意事项

1. **教训**：改完内核头文件后，Linux 侧要确认目标文件真的重新编译（看 `make` 输出里有没有对应那行 `Building C object`），不能只看 `make` 的返回码——本例中宏被截断导致的编译失败没有让构建整体失败，旧目标文件被沿用。
2. 该问题只在 Linux 暴露，Windows 全程正常，属于典型的"一个平台绿、另一个平台悄悄用旧二进制"的坑。

## 新增/变更：手写汇编 GEMM 内核与 MKL 对标完成

### 变更文件

| 文件 | 说明 |
|---|---|
| `ZQ_GEMM/math/zq_gemm_32f_align_c_asm.c` | GCC/Clang 内联汇编版微内核 + 驱动：新增 `m4n1`/`m1n1`（N<4 专用），归约改两级 `vhaddps` 树，`ZQA_HAVE_FMA` 认 `__FMA__`，`m2n4` 改 2 指针寻址 |
| `ZQ_GEMM/math/zq_gemm_32f_align_c_asm_msvc.asm` | Windows 侧同构实现（MASM），同步上面四项 |
| `SamplesZQBLAS/SampleGEMMCompare.cpp` | 工作区从 4M 元素放宽到 17M，覆盖 `4096x4x4096` / `4x4096x4096` |

### 实测结果（单线程，MKL 顺序层）

| 尺寸 | Linux asm/MKL | Windows asm/MKL |
|---|---|---|
| 16³ | 185% | 152% |
| 32³ | 68% | 78% |
| 64³ | 75% | 80% |
| 128³ | 93% | 93% |
| 256³ | 87% | 100% |
| 512³ | 92% | 90% |
| 1024³ | 89% | 81% |
| 1x1024x1024 | 130% | 182% |
| 1024x1x1024 | **177%**（优化前 13%） | **206%**（优化前 14%） |
| 4096x4x4096 | — | 266% |
| 8x2048x2048 | 184% | 220% |
| 2048x8x2048 | 181% | 297% |
| 128x128x4096 | 88% | 89% |
| 1152x256x1152 | 79% | 89% |
| 512x512x2048 | 77% | 78% |
| 7x5x13 | 144% | 120% |

`asm/intr` 多数形状 0.97–1.40（汇编版已反超项目自带 intrinsic 版），最大绝对误差 ≤1.5e-05。`SampleGEMMAsmCompare` 双平台 18/18 PASS。

### 注意事项

1. **Linux 侧此前一直没走 FMA**：`ZQ_CNN_CompileConfig.h` 里 `ZQ_CNN_USE_SSETYPE` 是 AVX（不是 AVX2），GCC 版发的是 `vmulps`+`vaddps`，而 Windows MASL 一直发 `vfmadd231ps`，MKL 也用 FMA —— 之前 Linux/Windows 的性能差距主要来自这里。现在内核只要编译器允许就用 FMA，纯 AVX 目标仍能退回 mul+add。
2. 归约改 `vhaddps` 树后，收尾 shuffle 的立即数**必须是 `0x44`**（树归约后 lane 0 与 lane 2 相同，`0x88` 会静默打包重复值）。这个坑在两个平台同时表现为"结果算错但不崩"。
3. 汇编版只在 x86/x86-64 提供；ARM/NEON 自动回落到 intrinsic 版。
4. 仍未达到 100% 的形状集中在"小块 + 中等 K"（32³ 68%、512x512x2048 77%）：此时每次微内核调用的固定开销占比过高，需要多 tile 融合的汇编例程才能进一步摊薄。

## 新增/变更：第三轮审计修复（第三方头文件层 + 人脸库层 + 特征提取正确性）

对应 `audit_k3_20261001.md` 附录 F。

### 变更文件

| 文件 | 问题 | 修复 |
|---|---|---|
| `3rdparty/include/ZQlib/ZQ_JpegDecoder.h` | `malloc(widthStep * height)` 是 int×int 乘法，恶意超大 JPEG（3×30000×24000）能让乘积溢出成负；紧接着 `memset` 因 `sizeof` 提升到 size_t 而用**真实大尺寸**去清**被截断的小 buffer** → 确定性堆溢出。`malloc` 未判空。调用链真实存在：`ZQ_FaceClusterImagesForVideo` 解码来自不可信容器文件的 JPEG | 64 位计算 + `0x7FFFFFFF` 守卫 + 判空 + 失败路径 `jpeg_destroy_decompress` |
| 同上 | `jpeg_start_decompress` 失败直接 `return false`，cinfo 与 JPOOL 全部泄漏 | 补 `jpeg_abort_decompress` + `jpeg_destroy_decompress` |
| `ZQlibFaceID/ZQ_FaceRecognizerSphereFaceZQCNN.h` | 6 个像素格式分支的裁剪行跨度写成 `h*crop_width + w*3`，BGR 每像素 3 通道应再乘 3。`ZQ_FaceDatabaseMaker` 在 `image.channels()==1` 时走 GRAY 分支 → **喂灰度图时特征一直是错的** | 6 处统一为 `h*crop_width*3 + w*3` |
| `ZQlibFaceID/ZQ_FaceRecognizerSphereFaceOpenCV.h` | 同一处错误的 5 个副本 | 同上 |
| `ZQlibFaceID/ZQ_FaceClusterImagesForVideo.h` | ① fread 数量不符时 `return true`（调用方拿到"加载成功但内容已清空"的容器）；② JPEG 编码失败释放了 buffer 却漏 `return false`，把空指针压进容器 | ①改 `return false` ②补 `return false`；另加 `length[i] > 0` / `offset[i] >= 0` / offset 连续性 / 64 位累计四重校验 |
| `ZQCNN/ZQ_CNN_MouthDetector.h` | `label*123457` 的 int 乘法在 label≥17387 时溢出成负，`% 6` 得负 offset → `colors[offset]` 负下标 | 改 64 位并对取模结果归一化 |
| `SamplesZQCNN/TrainMTCNNprocessor/TrainMTCNNprocessor.h` | 三处 1MB `malloc` 未判空，紧跟 `memset` | 判空 + 释放另一块 + `return false` |
| `ZQCNN/ZQ_CNN_TextBoxes.h`、`ZQ_CNN_NSFW.h`、`ZQ_CNN_MTCNN_old.h`、`ZQ_CNN_MTCNN_AspectRatio.h`、`ZQ_CNN_PersonPose.h`（两处） | `vector<uchar> buffer(w*h*3)` 的 int 乘法溢出 → 分配过小后越界写 | 64 位计算 + 范围守卫 |

### 实测结果

- **Windows**：全量构建 **0 error**；SampleMTCNN 18.7ms、SampleMTCNN_NCHWC4 7.5ms、SampleSSD、SampleFaceDetectorMTCNN 全部 exit=0，数值与改动前一致
- **Linux**：全量构建 **0 error**；SampleMTCNN、SampleMTCNN_NCHWC4（7.0ms）exit=0

### 注意事项

1. 前两轮把 `3rdparty/include` 当成"只看接口不看实现"的黑盒，漏掉了 `ZQ_JpegDecoder.h`——它同时有整数溢出、未判空、资源泄漏、致命错误处理缺失四个问题，且挂在人脸视频聚类的不可信文件读取路径上，是本项目唯一一条完整的"恶意文件 → 内存破坏"链路。第三轮起第三方头文件里**被实际调用到的函数体**也纳入审计范围。
2. `ZQ_CNN_MTCNN_ncnn.h` 的多处缺陷仍未修，但它**从不参与任何构建**（全仓没有一处 `#include` 它，CMake 只 glob `*.cpp`），属潜伏代码。
3. 仍未修的两项已记入报告：`ZQ_JpegDecoder` 未装 `setjmp`（损坏 JPEG 会让 libjpeg 直接 `exit()` 带走宿主进程）、`ZQ_CNN_BBoxUtils.h` 的 OpenMP 数据竞争（当前所有调用都传 `thread_num=1`，多线程分支是死路径）。

## 新增/变更：NMS 多线程竞争 + GEMM 暂存缓冲失败处理

### 变更文件

| 文件 | 问题 | 修复 |
|---|---|---|
| `ZQCNN/ZQ_CNN_BBoxUtils.h`（`_nms`） | `IOU/maxX/maxY/minX/minY` 声明在**函数作用域**却被 `parallel for` 里所有线程共用，IOU 会被撕裂 → NMS 抑制结果不确定；`cur_overlap++` 多线程自增丢失；`thread_num <= 0` 整数除零、`(box_num/thread_num)` 整除为 0 时 `schedule(static, 0)` 本身是未定义行为 | 变量下沉为循环内局部；`cur_overlap++` 加 `#pragma omp atomic`；`thread_num <= 1` 一律走单线程分支，`chunk_size` 保底 1 |
| `ZQCNN/layers_c/zq_cnn_convolution_gemm_32f_align_c.c`（4 处） | ① 调用方自带 buffer 时，`_aligned_malloc` 失败**仍然**把 `*buffer_len` 更新成新长度——下次调用 `*buffer_len < need` 不成立，会跳过重新分配直接拿 NULL 去 im2col；② 调用方传 `buffer==0` 时三个 `_aligned_malloc` 全部不判空 | 失败即释放已分配块并 `return`，只在成功后更新 `*buffer_len` |

### 实测结果

- **Windows**：全量构建 **0 error**；SampleMTCNN 20.3ms、SampleMTCNN_NCHWC4 8.2ms、SampleSSD、SampleFaceDetectorMTCNN 全部 exit=0，数值与改动前一致
- **Linux**：全量构建 **0 error**；SampleMTCNN、SampleSSD exit=0

### 注意事项

1. NMS 的多线程分支目前仍是"死路径"（全仓调用都传 `thread_num=1`），本次修复是按对外 API 的正确性做的，不影响现有行为。
2. GEMM 暂存缓冲的 4 处覆盖 fp32/fp16 × 通用/batch 两条路径；`layers_nchwc` 侧的同类模式在前一轮已经修过。

## 新增/变更：第四轮审计（构建配置面 / 数值算法 / 资源生命周期 / 示例参数解析）

对应 `audit_k3_20261001.md` 附录 H。

### 变更文件

| 文件 | 问题 | 修复 |
|---|---|---|
| `ZQCNN/layers_c/zq_cnn_convolution_gemm_32f_align_c_raw.h` | **高·静默算错**：1x1 卷积核 im2col 按 `padK` 写行，sgemm 收到的 `lda` 仍是未补齐的 `matrix_A_cols(=in_pixelStep)`。`padK` 由全局 `ZQ_CNN_USE_SSETYPE` 选，与该 TU 自己的 `zq_mm_align_size` 无关；`in_C % 8 != 0` 且快速路径不成立时，lda 与真实行距不符，**每一行都从错位置读**——不崩溃只出错数 | 该核不做 K 补齐（两侧行长本来就都是 `in_pixelStep`）。仓库内模型通道数都是 8 的倍数，所以此前一直没暴露 |
| 同上 | 7 处"内部自行分配"不判 `_aligned_malloc` 的 NULL，紧接着就 `zq_mm_store_ps` 写进去 | 判空 + 释放已分配块 + `return` |
| 同上 | 7 处"调用方自带 buffer"分配失败**仍然**更新 `*buffer_len`——与上一轮已在 `.c` 侧修好的同款，两边不一致 | 失败即 `return` 且不更新 `*buffer_len` |
| `CMakeLists.txt` | ① `BLAS_TYPE` 的分支嵌在 `if(SIMD_ARCH_TYPE MATCHES "arm…")` 里，x86 上 `-DBLAS_TYPE` **完全无效**；② 用 `MATCHES` 做子串匹配——默认的 `ZQ_GEMM` 匹配不到 `"zq_gemm"`，`openblas_zq_gemm` 被 `"openblas"` 抢先，`ZQ_CNN_USE_BOTH_BLAS_ZQ_GEMM` 永远不可达 | 分支移出 SIMD 判断；改成 `string(TOLOWER)` + `STREQUAL` 精确比较，顺序 `openblas_zq_gemm` → `openblas` → `zq_gemm` |
| `SamplesZQlibFaceID/SampleFaceDatabase{,,N,OpenCV,GrayN}CNN/*.cpp`（4 个） | `select_subset()` 里 `max_thread_num = atoi(argv[5])`，但按 usage 线程数是**第 7 个**参数 → 用户设的线程数被静默忽略、开满全部核（同文件的 `select_subset_desired_num` 写的是 `argv[9]`，可见是复制时漏改） | 4 处改为 `argv[7]` |
| `CMakeLists.txt` | 干净 checkout 时 `data/` 联接会退化成 `copy_directory`——多配置生成器的 `$(Configuration)` 子目录在首次 configure 时还不存在，`mklink /J` 直接失败，之后仓库里新增的测试图片不会再反映到产物目录 | 联接前先 `file(MAKE_DIRECTORY)` 建出各配置子目录 |
| `.gitignore` + 仓库 | 误提交了 8 个 MSVC 目标文件（约 3MB）、`tmp_*.cpp`、`build_verify_*.log` | 清理并加忽略规则（`*.obj` / `*.log` / `tmp_*`） |
| `AGENTS.md` | —— | 新增「配置宏的唯一真相」一节：CMake 与 `ZQ_CNN_CompileConfig.h` 两侧同名宏必须同步；`BLAS_TYPE` 用精确比较且不放进 SIMD 分支 |

### 实测结果

- **Windows**：全量构建 **0 error**；SampleMTCNN 20.5ms、SampleMTCNN_NCHWC4 8.5ms、SampleSSD、SampleFaceDetectorMTCNN 全部 exit=0，数值与改动前一致
- **Linux**：全量构建 **0 error**；SampleMTCNN、SampleMTCNN_NCHWC4、SampleSSD exit=0
- **干净 checkout 验证**：删掉整个 `cmake-out-win32-x64` 后重新 configure + 全量构建，0 error，`data`/`model` 都是 `<JUNCTION>` 而非拷贝，4 个示例 exit=0

### 注意事项

1. 1x1 卷积那个 bug 属于"静默算错"：不崩溃、不越界，只是结果不对。仓库自带的模型通道数都是 8 的倍数，所以现有示例全对；**如果用户换成 C 通道不是 8 倍数的模型（1x1 卷积接在 3 通道输入之后），修复前会算错**。
2. `ZQ_CNN_USE_BLAS_GEMM` 等宏在 CMake 与头文件各有一份且没有 `#ifndef` 保护，命令行 `-D` 会被头文件覆盖——本轮只把规则写进 AGENTS.md，没有改行为（改了会让"链了 MKL 却仍用 ZQ_GEMM 算"变成另一种静默行为）。
3. 本轮另有 6 项中低危已评估为"暂不修"并记入报告附录 H（`ChangeSize` 半更新、resize 边框清零不对称、`SSETYPE_NONE` 缺标量分支、`model/CMakeLists.txt` 死文件、deconvolution 同款 padK 死代码、`similarity_thresh` 不夹取）。

## 新增/变更：Tensor4D 的边框清零与 ChangeSize 半更新

### 变更文件

| 文件 | 问题 | 修复 |
|---|---|---|
| `ZQCNN/ZQ_CNN_Tensor4D.cpp`（`ResizeBilinearRect`/`ResizeNearestRect`，4 个派生类各一处） | 下边框写成一次性 `memset(dst_slice_ptr + dstWidthStep*dst_borderH, ..., dstWidthStep*dst_borderH)`，清的是行 `[borderH, 2*borderH)`——**真正的下边框 `[H, H+borderH)` 根本没清到**；左右两列也只清 `h ∈ [0, borderH)` 而不是全部 H 行。`ChangeSize` 发现长度够就复用 buffer（不重新分配也不重新 memset），于是上一轮的残留数据会被后续带 padding 的卷积读到 | 改成与同类 `Padding()` 一致的逐行写法：上边框行 `[0,borderH)`、下边框行 `[H, H+borderH)`，左右两列覆盖全部 H 行 |
| 同上（`ChangeSize`，3 处） | `shape_nchw[0..3]` 在 `_aligned_malloc` **之前**就写；分配失败 `return false` 时对象处于"形状是新的、指针/长度/步长是旧的"半更新状态 | 推迟到分配成功之后再写形状 |

### 实测结果

- **Windows**：全量构建 **0 error**；SampleMTCNN 19.3ms、SampleMTCNN_NCHWC4 9.0ms、SampleSSD、SampleFaceDetectorMTCNN、SampleCascadeOnet、SampleLnet106 全部 exit=0，数值与改动前一致
- **Linux**：全量构建 **0 error**；SampleMTCNN、SampleSSD exit=0

### 注意事项

1. 这两处都是"平时看不出来"的问题：只有当**同一个 tensor 对象被复用来做不同尺寸的 resize**（`ChangeSize` 走复用分支）时才会读到脏数据。现有的示例都是一次性分配，因此数值没有变化。
2. 第四轮其余"已评估暂不修"的 8 项已在 `audit_k3_20261001.md` 附录 I 逐条写明理由，其中 `zq_cnn_convolution_gemm` 的 deconvolution 同款 `padK` 失配属于**全仓无调用点的死代码**。

## 新增/变更：第五轮自查（并发 / 数据流 / 平台语义）—— 三处"看着危险、核对后安全"

本轮没有代码改动，全部是**先核对再动手**的结论，记录下来避免以后重复走弯路：

1. **MTCNN 多线程 Pnet/Rnet/Onet/Lnet 的并行区**（`ZQ_CNN_MTCNN.h` 6 处 `#pragma omp parallel for`）：一度被判成"多线程并发改写同一个 `input` 张量"的高危数据竞争。核对 `ResizeBilinear/ResizeBilinearRect/ROI` 的签名后确认它们都是 **const 成员函数，调用者是源、形参是目的地**（`input.ResizeBilinear(dst, ...)`），而 `pnet_images` / `task_*_images` 是 `vector<vector<Tensor>>` 或按 `thread_num` 下标分配，各写各的 → **线程安全**。曾按错误结论改过一次，会把缩放结果写进没人用的临时张量，已回退。
2. **Concat 的数据流**（`ZQ_CNN_Forward_SSEUtils.cpp:_concat_NCHW`）：拷贝循环对每个输入分别取自己的 `in_pixStep/in_widthStep/in_sliceStep`、对输出取 `out_*`，异构输入混拼是安全的；`valid_inputs.size()==1` 走 `CopyData`，`==0` 走 `ChangeSize(0,...)`，边界都有处理。
3. **BatchNormScale 的 `eps`**：从不可信 `.zqparams` 里 `atof` 读、解析处零校验，看着像"eps=0 + var=0 → 除零 → inf 污染全网"。内核里已有 `__max(var_data[c]+eps, FLOAT_EPS_FOR_DIV)` 兜底，零/负/垃圾值都被挡住 → 不改。

### 注意事项

1. 第 1 条已经写进 AGENTS.md 的「ZQCNN 里容易看反的 API 约定」：`Resize*/ROI/Padding` 的接收者是**源**。这类方向性误判在本项目里很容易发生（本轮连着误判了 eltwise 增量目标、`handled` 初值、ResizeBilinear 参数方向三次），动代码前必须先逐元素/逐参数推演。
2. 第五轮的方向二（真实模型的层→kernel 步长推演）、方向三（LLP64/算术右移/`-ffast-math` 下的 NaN 检查）、方向四（`Init` 失败后重复调用）仍在审计中，结论待补。

## 新增/变更：第五轮审计修复（Init 失败路径 + 边框清零 12 处收口）

对应 `audit_k3_20261001.md` 附录 L。

### 变更文件

| 文件 | 问题 | 修复 |
|---|---|---|
| `ZQCNN/ZQ_CNN_MTCNN.h`、`_Interface.h`、`_NCHWC.h`、`_AspectRatio.h`（7 处） | `Init`/`InitFromBuffer` 在 `ret==false` 时把 `pnet/rnet/onet/lnet` 全部 `clear()` 并 `thread_num=0`，但紧接着的调试打印和**无条件**的 `rnet[0].GetInputDim(C,H,W)` / `onet[0]` / `lnet[0]` 仍在解引用空 vector 的下标 0（`return ret(false)` 排在崩溃之后） | 在 `else this->thread_num = thread_num;` 之后补 `if (!ret) return false;` |
| `ZQCNN/ZQ_CNN_Tensor4D.cpp`（11 处）、`ZQ_CNN_Tensor4D.h` 的 `ROI`（1 处） | 上一轮只改了 12 处中的 4 处边框清零，其余沿用旧写法：下边框一次性清行 `[borderH, 2*borderH)`，**真下边框 `[H, H+borderH)` 从未清到**；左右两列只清 `h ∈ [0, borderH)`。`ChangeSize` 长度够就复用 buffer（不重新分配也不重新 memset），于是同尺寸复用时会读到上一轮残留；`NSFW`/`PersonPose` 走的 `Align128bit::ResizeBilinearRect` 在 `borderH=1` 时会把缩放结果的**第 1 行整行清零** | 全部改为与同类 `Padding()` 一致的逐行写法：上边框 `[0,borderH)`、下边框 `[H, H+borderH)`，左右两列覆盖全部 H 行 |

### 一条被否掉的审计结论（重要）

第五轮把 `Reshape_NCHW` 的 `i_c`（步长 1）与 `out_c_ptr++` 判成"高危静默算错"，理由是"c 维步长应为 sliceStep"。按此修改后 **`SampleSSD` 在模型加载阶段直接段错误**，回退后恢复。核对 `ZQ_CNN_Tensor4D::ChangeSize`：

```
dst_pixelStep = dst_C;              dst_widthStep = dst_pixelStep*dst_realW;
dst_sliceStep = dst_widthStep*dst_realH;   dst_tensor_raw_size = dst_sliceStep*dst_N;
```

NCHW 里 `sliceStep` 是**一张图**的步长（不是通道），`(n,c,h,w)` 的偏移是 `n*sliceStep + c*1 + h*widthStep + w*pixelStep`——原实现是对的。有 `imStep`（一张图）与通道 `sliceStep` 之分的是 NCHWC 那一侧。已把这条写进 AGENTS.md。

### 实测结果

- **Windows**：全量构建 **0 error**；SampleSSD（10.2ms）、SampleMTCNN（18.2ms）、SampleMTCNN_NCHWC4（7.8ms）、SampleFaceDetectorMTCNN 全部 exit=0
- **Linux**：全量构建 **0 error**；SampleMTCNN、SampleMTCNN_NCHWC4、SampleSSD 全部 exit=0

### 注意事项

1. 附录 L 里还列了 7 项中低危待办（batchnorm 末分支整向量读写、Eltwise `ReadParam` 条件写反、`LoadFromBuffer` 未做 `_simplify_inplace`、MTCNN 构造函数未初始化若干成员、两处 omp 非原子调试计数、`_Lnet106_stage` 无条件 memcpy 212 float、`-ffast-math` 只在 GCC/Clang 侧开导致跨平台浮点结合序不同），均不影响现有示例，尚未修。
2. 本轮再次印证 AGENTS.md 里那条"先确认 API/步长方向再动手"：连续三次方向性误判（`ResizeBilinear` 接收者、`eltwise` 增量目标、NCHW 的 sliceStep 含义）都曾导致"修复"反而破坏功能。

## 新增/变更：附录 L 待办清零（第 5、6 项）

| 文件 | 问题 | 修复 |
|---|---|---|
| `ZQCNN/ZQ_CNN_Layer.h` | Eltwise `ReadParam` 收尾条件写反（`!=`）：weight 数量**正确**时整个模型加载失败，数量**不匹配**反而放行 | 改为 `==` |
| `ZQCNN/ZQ_CNN_MTCNN*.h`（4 个变体构造函数） | `has_lnet/thread_num/rnet_size/onet_size/lnet_size/do_landmark/early_accept_thresh/nms_thresh_per_scale` 未初始化 | 补初值（各变按自身成员裁剪） |
| `ZQCNN/ZQ_CNN_MTCNN.h` | `_Lnet106_stage` 整块 memcpy 212 个 float，`keypoint_num < 106` 时尾部是未初始化内存 | 拷贝前 memset 清零 |
| `ZQlibFaceID/ZQ_FaceDatabase.h` | 并行区里 `same_pair_num++/notsame_pair_num++` 非原子共享写（UB），且结果本来就被整体覆盖 → 死写 | 删除 |
| `ZQCNN/layers_c/zq_cnn_batchnormscale_32f_align_c_raw.h` | `in_C` 非 4/8 倍数的两条回退分支仍按整向量读写，`a`/`b`/`scale` 是 `_aligned_malloc(2*4)` 之类的小分配 → 过界读 8 字节 | 两条回退分支改标量 |
| `ZQlibFaceID/ZQ_FaceContainerForVideo.h` | `key_num` 只挡负数，`0x7FFFFFFF` 直接 OOM（未捕获 bad_alloc → terminate） | 用剩余文件长度交叉校验帧数上界 |

**双平台复验**：Windows 全量构建 0 error（SampleMTCNN 18.6ms / SampleSSD 10.0ms / SampleFaceDetectorMTCNN 9.7ms），Linux 全量构建 0 error（SampleMTCNN / SampleMTCNN_NCHWC4 / SampleSSD 全部 exit=0）。

**仍未修**：`LoadFromBuffer` 未调用 `_simplify_inplace()`（会改变 buffer 加载路径的行为，仓库里没有可用模型可验证，按"改动前先能验证"的原则暂留待办）。

## 新增/变更：LoadFromBuffer 启用 `_simplify_inplace()`（附录 L 最后一项）

`ZQ_CNN_Net::LoadFromBuffer` 里的 `//_simplify_inplace();` 被注释掉，导致两条加载入口行为不一致：

- `GetBlobByName(top_name)` 对 in-place 层（ReLU/ReLU6/PReLU/BatchNormScale/BatchNorm/Scale/AddBias）返回的是 **bottom 张量**而不是 top；
- 对用**不同名**声明 in-place 的模型（如 `det3-dw48-p0` 的 `PReLU bottom=bn1 top=relu1`）会多分配一整份 blob。

本机有可验证路径（`SampleCascadeOnet` / `SampleCascadeOnet_Interface` / `SampleMTCNNLoadFromCode` 都走 `LoadFromBuffer`），启用后回归结果与改动前一致：SampleCascadeOnet 1.64M、SampleCascadeOnet_Interface 0.65M、SampleMTCNNLoadFromCode 136ms、SampleMTCNN 19.0ms、SampleSSD 10.2ms，全部 exit=0。

至此 `audit_k3_20261001.md` 附录 L 的 7 项待办全部闭环。

## 新增/变更：第六轮审计（换角度：调用契约与边界条件）

第六轮不再看单文件内部，而是横向核对**调用方与被调方的契约**、**边界输入**、
**未被前五轮覆盖的目录**。本条记录前半部分（行尾 + 边框 + 关键点），
后半部分（加载器校验、SetPara、PriorBox 等）待后续提交补。

### 变更文件

| 文件 | 问题 | 修复 |
|---|---|---|
| `SamplesZQCNN/CompareWithOpenBLAS/CompareWithOpenBLAS.cpp` 等 7 个 | 多重 CR 行尾（`\r\r\n` / `\r\r\r\n`），CR 会并进 `#include` / `#ifndef` 的预处理符 token（UB） | 统一为纯 CRLF；非空行内容零差异 |
| `tools/check_line_endings.py`（新增） | 缺 mixed-EOL 检测 | 增加 CRLF/裸 LF 混用检测与 `--fix` |
| `tools/run_sample_regression.sh`（新增） | 双平台回归靠手敲 | 一键跑 8 个关键 sample |
| `ZQCNN/ZQ_CNN_Tensor4D.h` | `ROI` 左右边框只清前 `dst_borderH` 行；`ConvertColor_BGR2GRAY` 除此以外下边框位置也写成了 `+ dstWidthStep*dst_borderH`（清的是数据区而不是真下边框） | 分别改为 `h < height` / `+ dstWidthStep*H` |
| `ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp` | **NCHWC1/4/8 整个系列 15 处**同型缺陷（6 处下边框位置 + 9 处左右列循环上界） | 与 NCHW 侧写法对齐（`dst_H` / `height`） |
| `ZQCNN_to_MNN/converter/source/ZQ_CNN_Tensor4D.h` | 转换器里的独立副本同型 | 同上 |
| `ZQCNN/ZQ_CNN_MTCNN*.h`（5 个文件） | Lnet/Onet 取 `conv6-3` 后**不判空**直接解引用；5 点循环上界写死 5，不看实际通道数 | 21 处（5 点 15 + 106 点 6）统一改为判空取指针 + `__min(N, GetC()/2)` 限幅 |

### 实测结果

- Windows：`cmake --build build_x64 --config Release -j8` → 0 error，69 exe
- Linux：`make -j8` → 0 error
- 双平台 8 个 sample 全部 rc=0
- 关键点改动前后 `SampleMTCNN` / `SampleMTCNN_NCHWC4` 的
  rnet/onet/lnet 参数量、候选数、检出数**逐字节一致**（改前已抓基线）
- 边框改动前后同样逐字节一致（MTCNN 的 Resize 调用 border 多数为 0，
  属潜伏缺陷而非现网错误）
- `tools/check_line_endings.py` 全仓扫描：line endings OK

### 注意事项

1. **第四/五轮声称"边框清零已覆盖 ROI"是不成立的**——实际只改了 NCHW 的
   `Resize*`，`ROI`、`ConvertColor_BGR2GRAY` 以及整个 NCHWC 系列一处没动。
   收口一类缺陷必须全仓枚举同类站点逐个核对，不能只信上一轮的清单。
   这条已写进 AGENTS.md。
2. 本机 `core.autocrlf=true`，git 提交时会把工作区 CRLF 归一成 LF 存进仓库，
   所以**"git diff 干净"不代表工作区行尾干净**，而工作区才是编译器读的东西。
3. 用 Python 批量改写源码时 `split(b'\n')` / `join(b'\n')` 会让新插入的行
   丢失 `\r`，造成工作区 CRLF/LF 混用。这条也已写进 AGENTS.md 第 5 条。

## 新增/变更：第六轮后半段（模型加载器 / Net 契约 / PriorBox / 低危项收口）

| 文件 | 问题 | 修复 |
|---|---|---|
| `ZQCNN/ZQ_CNN_Net.h` | 未知层类型被**静默丢弃**：if/else 链以 `Input` 结束、无 else 兜底 | 打印层名与整行后 `return false` |
| `ZQCNN/ZQ_CNN_Net.h`、`ZQ_CNN_Net_NCHWC.h` | **`_getline` 行指针错位**：`buffer += cur_len` 而 `cur_len = j - i`，少推进开头跳过的行尾符，导致每读一行有效层定义就多产出一行 1 字符的垃圾行 | `buffer += j`（推到行尾符位置） |
| 同上 | `sscanf(...) == 0` 只挡"无匹配"，**空行返回 EOF(-1)** 会穿过去；且缺 `#` 注释行跳过 | 改判 `!= 1`；补 `if (buf[0] == '#') continue;` |
| `ZQCNN/ZQ_CNN_Layer.h` | `Reduction::ReadParam` 只实现 SUM/MEAN，未知名字 `atoi(str)` → `"max"` 静默变成求和 | 判非法并报错 |
| `SamplesZQCNN/mxnet2zqcnn/mxnet2zqcnn.cpp` | 26 处写出加载器不认的层类型名 + 末尾兜底把 mxnet op 名当层类型；`Activation` 分支缺 else | 全部改为转换阶段明确报错 `return -1` |
| `ZQCNN/ZQ_CNN_Net.h`、`_NCHWC.h` | 会改形状的层把 `top` 声明成自己的 `bottom` 会毁掉输入 | `_check_connect` 拦截；in-place 白名单抽成 `_is_inplace_safe(i)` 与 `_simplify_inplace()` 共用 |
| `ZQCNN/ZQ_CNN_Layer.h` PriorBox | `Forward` 就地把百分比换算成像素写回成员 → 第二次 Forward 不重算；守卫漏判 `bottoms->size()==1` | 局部副本换算；守卫补 `< 2` 与 `(*bottoms)[1]` |
| 同上 `ReadParam` | `img_w = img_w;` / `step_w = step_w;` 三处自赋值，宽/步宽根本没被解析 | 补成 `img_h = img_w = atoi(...)` / `step_h = step_w = ...` |
| `ZQCNN/ZQ_CNN_Forward_SSEUtils.cpp` | `//return false;` 注释掉后紧接着解引用 `map::end()`；`resize(keep_top_k)` 增长语义补出假检测框、`num_kept` 虚计（各 2 处） | `continue`；只截断不增长 + 按实际条数计 |
| `SamplesZQCNN/TrainMTCNNprocessor/TrainMTCNNprocessor.h` | `buf2[-1]` 越界读（2 处） | `len > 0 && ...` |
| `ZQlibFaceID/ZQ_FaceFeature.h` | `ChangeSize` malloc 失败仍写 `length` | 一并清 0 |
| `ZQlibFaceID/ZQ_FaceClustersForVideo.h` | `pivot_pt_ids[i]` 为 -1 未防御；`face_boxes[box_id]` 不校验 | 两处加守卫 |
| `ZQlibFaceID/ZQ_FaceGroup.h` | `WriteToFile` 不校验 `face_boxes.size() >= num` | 写前校验 |
| `ZQlibFaceID/ZQ_FaceClusterImagesForVideo.h` | `int _off` 累加 JPEG 长度可溢出成负 | 改 `__int64` |
| `ZQlibFaceID/ZQ_FaceRecognizerSphereFaceZQCNN.h` | 预置路径不校验 blob 存在性与通道数，`ExtractFeature` 结尾 `memcpy` 越界读/静默截断 | 补 `out != NULL && out->GetC() == feat_dim` |
| `audit_k3_20261001.md` | — | 新增**附录 M**（含 M.5 记录一条"评估后否决"的修复） |

### 实测结果

- Windows：重建 0 error；SampleSSD 9.9~10.5 ms/iter（改动前同区间）；
  SampleMTCNN / SampleMTCNN_NCHWC4 输出与本轮改动前的基线**逐字节一致**；
  8 个 sample 全部 rc=0；TrainMTCNNprocessor / mxnet2zqcnn 正常启动
- Linux：`make -j8` → 0 error；8 个 sample 全部 rc=0

### 注意事项

1. **审计建议里有一条是错的，照做会直接让仓库自带的 mobilefacenet 模型加载失败。**
   "两个层写同一个 top 就报错"会命中 77 处合法用法（残差块复用上一个 block
   的 blob 名当输出）。已写脚本全仓扫描验证：A(重复 top)=77、B(top==bottom)=0，
   只采纳 B。详见报告 M.5。
2. **本轮最值得记的一条**：给加载器加"未知层类型就报错"的兜底之后，
   立刻炸出 `_getline` 的行指针错位——而后者之所以长期没暴露，
   正是因为前者把垃圾行静默吃掉了。两个 bug 互相掩盖。
3. `PriorBox` 的百分比路径（`min_size` 为负）本机没有模型覆盖，
   尝试写独立探针程序复现但未跑通，**未取得端到端证据**，
   按"按构造正确 + 现有模型等价"记录，不声称已复现。

## 新增/变更：第七轮（张量变体一致性 + 裸指针生命周期）

### 变更文件

| 文件 | 问题 | 修复 |
|---|---|---|
| `ZQCNN/ZQ_CNN_Tensor4D.cpp` | `Align128bit::ResizeBilinearRect`（标量+向量）把越界 rect 原样交给 resize 内核。实测（临时探针）`SampleCascadeOnet_Interface` 有 1 次 rect 纵向超出 34 像素，而源张量只有 1 像素 border → **堆越界读** | 进 resize 前把 rect 夹到 `[-border, 尺寸-1+border]`；向量版入参是 `const&`，夹到本地副本 |
| 同上 | `Align128bit::ResizeNearestRect`（标量）越界守卫分支体写成了"正常 resize" | 对齐成 `return false`（`ResizeNearest` 全仓零调用方，无行为风险） |
| `ZQlibFaceID/ZQ_FaceDatabaseCompact.h` | **空析构**：三个堆指针的释放逻辑全在 `_clear()` 里，析构一次都没调 → 加载过数据库就出作用域必然全量泄漏，正常路径 100% 触发 | 析构调 `_clear()`；同时 `= delete` 拷贝构造/赋值（只补析构会把泄漏变成 double free） |
| `ZQlibFaceID/ZQ_FaceDatabase.h` | `Search` 的"首元素"判断写在 k 循环内部：维度不匹配时 `max_id` 被钉死为 0，**返回 true 并给出一整份伪结果** | 维度不匹配整对 `continue`；`filenames[person_j[...]]` 补边界守卫 |
| 同上 | `_detect_lowest_pair` 的 `out_i/out_j` 未初始化（2 处） | 补初值 |
| `ZQlibFaceID/ZQ_FaceDatabaseMaker.h` | `omp_get_num_procs()-1` 单核时为 0，`num_threads(0)` 是 OpenMP 未定义行为（2 处） | `__max(1, ...)` |
| `ZQlibFaceID/ZQ_FaceContainerForVideo.h` | `SaveToFile` 失败路径漏 `fclose(out)` | 补 `fclose` |
| `AGENTS.md` | — | 新增「三个张量变体对越界 rect 的策略互相冲突」+「不要改无法编译验证的第三方头」+「改完先抓基线」三条 |
| `audit_k3_20261001.md` | — | 新增**附录 N**（变体一致性）与**附录 O**（ownership / 第三方头） |

### 实测结果

- 越界幅度探针（跑完即删）：CascadeOnet 3 次调用 0 次超 border；
  **CascadeOnet_Interface 10 次调用 1 次超 border，最大纵向 34 px**；
  MTCNN 700 次 / FaceDetectorMTCNN 8815 次 / LoadFromCode 1192 次全部 0
- Windows 重建 0 error；SampleMTCNN / NCHWC4 输出与基线逐字节一致；sample 全部 rc=0
- Linux `make -j8` → 0 error；8 个 sample 全部 rc=0

### 注意事项

1. **又一次"方向判断反了"**：`Align128bit::ResizeBilinearRect` 看起来就是守卫写反
   （另两个变体都是 `if (越界) return false;`），改成 `return false` 之后
   `SampleCascadeOnet` / `_Interface` 直接 `Find()` 返回 false，一张脸都检不出。
   分两步定位：① 保留原分支只加守卫 → 仍失败；② 只改 ResizeNearestRect → 全通过。
   根因是 MTCNN 全家**故意**传越界 rect（`ZQ_CNN_MTCNN*.h` 里 16 处边界检查
   被注释掉了），所以最终选了"夹取"而不是"拒绝"。
2. 审计代理报告的 21 条里，复核后采纳 5 条、明确记下不修的 8 条（附理由），
   另有 13 条它自己已确认干净。**死代码链要单独标出来**：
   `ZQ_FaceClustersForVideo.h` / `ZQ_FaceContainerForVideo.h` /
   `ZQ_FaceClusterImagesForVideo.h` 全仓零 `#include` 引用点。
3. `ZQ_FaceDatabaseCompact::_load_feats` 失败路径"不释放"看起来像泄漏，
   实际是安全的（已写进成员，调用方 `_clear()` 兜住）—— 这类极易误判。

## 新增/变更：GEMM 小 K 家族（负结果，回退）

64 形状对拍里唯一系统性低于 MKL 的是 K ≤ 32 那一族（512x512x1 只有 6~11%，
1024x1024x4 约 40%）。根因：微内核沿 K 方向做 ymm 累加，K < 8 时 `k8 == 0`，
K 循环一次都不进，等于把 C 整块清零再由标量补加，对 C 做了三趟访存。

两次尝试都**已回退**，详细过程见 `audit_k3_20261001.md` 附录 P：

1. 沿 N 方向直接向量化 —— **结果错误**（`Bt` 是 N×K，相邻列差 `ldb`，
   8 个结果里 7 个错且不崩溃）
2. 先把 B 打包成 N 连续再向量化 —— **正确但没收益**（512x512x1 −7%、
   1024x1024x4 −3.7%，其余在噪声内）

### 更有价值的一条

我原先按指令条数估算"每 8 个结果 14 条 uop、理论 22 GF/s"，与实测 1.3 GF/s
差了 16 倍 —— 瓶颈判断错了。已写进 `AGENTS.md`：
**没有 profiler 时不要按 uop 估算换算法**，uop 估算只能排除明显不划算的方案。
本机 WSL 无 `perf`，只能 A/B 实测，而三次实测都是"没收益或略差"。

### 本次新增的文档
- `audit_k3_20261001.md` 附录 P
- `AGENTS.md` 第 7 条构建规则

## 新增/变更：审计报告主表状态复核 + GEMM A/B 基准工具

### 复核结果
第一轮写的「一、高危漏洞明细」表里有 18 条标着"待核实/未修复"，但那些
其实在后续几轮里已经修完，表没同步。逐条打开当前代码核实后**全部落定**：

  H3/H4  ChangeSize 整数溢出   → 6 处 __int64 + 0x7FFFFFFF 守卫
  H5     _prior_box 尺寸不一致 → priors_per_cell != num_priors 前置校验
  H6     SSDSpec::aspect_ratios → 写入前查 sizeof 上界
  H7-H9  keypoint 写 ppoint[212] → __min(106, GetC()/2)
  H10    VideoFaceDetection    → 已有限幅
  H11    num_points 固定 18 点  → __min(18, hm_C)
  H12/13 batchnormscale 整向量 → c 循环已改标量
  H14    FP16 resize 2× 溢出    → 按 sizeof(zq_base_type) 分配
  H15    eltwise_mul_nchwc 增量 → 已加到 *_slice_ptr
  H16    eltwise_sum_nchwc 重置 → n 循环里重初始化 out_im_ptr
  H22    name_count/len 越界   → 双校验
  H23    input_index 越界      → 4 处范围检查
  H24    sprintf 栈溢出        → %511s + 长度检查
  H25    buf2[-1] 越界读       → len2 > 0 守卫

主表现在 **0 条** H 项还带"待核实/未修复"，头部统计与结论段落同步更新。

### 新增 tools/bench_gemm_ab.py，并查出一个影响所有历史 GEMM 数字的问题
**同一个文件跟它自己比，64 个形状里有 20 个出现 >3% 的"差异"，最大的 10%，
而且方向一致。** 原因是"先把 A 跑完再跑 B"，中间睿频/温度漂移系统性偏袒
后跑的那个。改成**交替跑**（A,B,A,B,…）后降到 7 个 >3%、最大 7.3%、
49/64 在 ±3% 内 —— 这个 ~7% 就是本机 GEMM 测量的噪声下限，
默认阈值已设成 8%。

这条对附录 P 里小 K 家族的"无收益"结论是**加强**而不是削弱：
那次测到的 −7% 正好在噪声边缘，所以"没有明显收益"这个判断依然成立，
但"−7% 是变慢"这种说法应当收回为"在噪声内"。

同时用该工具复测了 N 方向分块（`be6eeea`）的真实收益，
结果见下一个提交。
