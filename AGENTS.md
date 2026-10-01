# AGENTS.md

## 变更日志规则（必须严格遵守）

写变更日志到 `docs-changelogs/CHANGELOG_YYYY-MM-DD.md` 时：

1. **先检查当天日期的日志文件是否已存在**（用 `ls docs-changelogs/` 或直接 Read）。
2. **已存在 → 只追加，不覆盖**：必须先 Read 读出现有内容，把新章节追加到文件末尾（Edit 在末尾锚点插入，或 Write 时用"现有内容 + 新章节"的完整内容）。严禁直接 Write 覆盖导致已有章节丢失。
3. **不存在 → 新建**：以 `# CHANGELOG YYYY-MM-DD` 开头。
4. **每次写日志前先执行 `date '+%Y-%m-%d'` 确认当天日期**，再决定写入哪个文件——不得凭会话惯性沿用上一次写日志的日期。会话可能跨天或间隔数日继续（教训实例：2026-08-10 的实测记录曾误写入 CHANGELOG_2026-08-07.md，事后迁移订正）。
5. 章节格式沿用既有风格：`## 新增/变更：标题` + `### 变更文件` / `### 实测结果` / `### 注意事项`。
6. 写完日志后发现他人（或前一个任务）又加了内容时，合并保留双方章节，不得整文件重写丢弃。

## 构建规则

1. **CMake 是唯一构建入口**：仓库已于 2026-10-01 移除全部 `.sln/.vcxproj*`（停留在 VS2015/v140、SDK 8.1，文件列表早已与源码脱节）。不要重新引入 MSVC 工程文件。
2. **双平台必须都编译通过**：改动 C/C++ 后要分别验证 Windows（`cmake --build build_x64 --config Release`）与 Linux（`wsl -d Ubuntu-20.04` + gcc 9，`/tmp/zqb` 下 cmake + make）。MSVC 能过不代表 gcc 能过。
3. **不要依赖 MSVC 的传递包含**：MSVC 会顺带引入 `<cfloat>` `<cmath>` 等，gcc 不会。新写的代码要显式 `#include` 用到的标准头（已踩坑：FLT_MAX 在 gcc 下报未声明）。
4. `ZQ_GEMM/CMakeLists.txt`、`ZQCNN/CMakeLists.txt`、`SamplesZQCNN/CMakeLists.txt` 都用 `file(GLOB ...)`，**新增 .c/.cpp 文件无需改 CMake，但需要重新 configure 才生效**。

## 提交规则

1. 阶段性成果就 commit（构建修复、审计修复、文档、报告各自成次）。
2. **不要 push**，推送由用户自己决定。
3. 一个 commit 只做一件事，提交信息用中文写清楚改了什么、为什么。

## 示例程序规则

1. 所有 Sample 中的 `cv::namedWindow` / `cv::imshow` / `cv::waitKey` **一律注释掉**（无头环境与自动化验证会阻塞；用户 2026-10-01 明确要求）。
2. 跑示例前先确认没有等待按键的调用。


## 汇编/低层代码规则

1. **MSVC x64 不支持函数体内联汇编**：`__asm { }` 和 `__declspec(naked)` 在 x64 目标上都会编译失败（2026-10-01 实测 MSVC 14.35 报 C2143/C4235）。Windows 侧的手写汇编必须走独立 `.asm` 文件（CMake 里 `enable_language(ASM_MASM)` + `.asm`），GCC/Clang 侧才用 `__asm__ volatile` 内联汇编。写跨平台低层代码前先想清楚这两条路径。
2. `3rdparty/lib/libncnn.a` 是 clang 编译的，引用 `__exp_finite`/`__log_finite` 等 compiler-rt 符号，用 gcc 链接时由 `ZQCNN/math/zq_libm_compat.c` 补齐，不要删。
3. 基准测试程序里 MKL / OpenBLAS 一律**运行时动态加载**（`LoadLibrary`/`dlopen`），不引入链接期依赖；MKL 运行时放在 `3rdparty/mkl_runtime/`（已 gitignore），Linux 下 `libmkl_rt.so.2`、Windows 下 `mkl_rt.3.dll`。
