#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZQCNN 审计这一轮加出来的全部检查，一个入口跑完（**在 Windows 侧跑**）。

    python tools/run_audit_checks.py                # 检查组（下面 A/B/C）
    python tools/run_audit_checks.py --quick        # 跳过慢的可编译性门禁
    python tools/run_audit_checks.py --with-build   # 再加上双平台全量构建 + sample 回归
    python tools/run_audit_checks.py --ubsan        # B 组换成 UBSan 再跑一遍

分组：

D 主工程双平台回归（只在 --with-build 时跑）
    Windows  cmake --build build_x64 --config Release
    Linux    wsl 里的 /tmp/zqb2 make
    两边各跑一遍关键 sample（tools/run_sample_regression.sh 与等价的 exe 调用）

A 文本与配对卫生（秒级）
    tools/check_line_endings.py    multi-CR / lone-CR / CRLF+LF 混用
    tools/check_text_encoding.py   UTF-8 有损解码残留（U+FFFD）
    tools/check_alloc_delete.py    malloc 配 delete[] / new 配 free（--selftest 先自测）

B 第三方头库的独立回归测试（10 组，每组几秒）
    tools/run_zqlib_checks.py        (gcc / WSL，ASan+LSan 或 UBSan)
    tools/run_zqlib_checks_msvc.bat  (MSVC /fsanitize=address / Windows，--msvc-asan 时跑)

C ZQlib 可编译性与警告门禁（慢，各约 2 分钟）
    C   tools/probe_zqlib_headers.py --check-baseline tools/zqlib_probe_baseline.txt
    C2  MSVC 侧头探测                       (--msvc-probe)
    C3  gcc -Wall -Wextra 的 HIGH 桶基线    (--warn-sweep)

**为什么这个脚本必须是 Python 而不是 .sh**
B 和 C 里的两个工具本身是「Windows 侧 Python → 通过 `wsl ... bash -s` 喂脚本 →
在 WSL 里编译」。把它们放进一个 WSL 里的 shell 脚本去调用，会在 WSL 里再起一个
Python，然后那个 Python 想调 `wsl` —— 没有 `subprocess` 模块（那是 Windows 的
标准库）。2026-10-02 实测踩过：`AttributeError: 'module' object has no attribute 'run'`。
"""

from __future__ import print_function

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

GROUPS = [
    ('A1 行尾卫生 (check_line_endings)', ['check_line_endings.py'], False),
    ('A2 编码卫生 (check_text_encoding)', ['check_text_encoding.py'], False),
    # A3 只有 5 秒，但它挡掉的是**整个工具自己变成哑巴**这件事：
    # 改 check_alloc_delete.py 的匹配逻辑之后忘了跑自测，那它返回的「没有命中」
    # 就毫无意义（附录 AT.8 的教训）。
    ('A3 分配/释放配对扫描自测 (check_alloc_delete --selftest)',
     ['check_alloc_delete.py', '--selftest'], False),
    ('A4 malloc/delete 错配扫描 (全仓 706 个源文件)',
     ['check_alloc_delete.py'], False),
    ('A5 未初始化类成员扫描自测 (check_uninit_members --selftest)',
     ['check_uninit_members.py', '--selftest'], False),
    ('A6 未初始化类成员扫描 (ZQCNN/*.h)',
     ['check_uninit_members.py', '--all'], False),
    # A7 是 A5 那类"自测"思路的延续：值域校验基线保证
    # 「已经在 ReadParam 里查过的参数」不会悄悄丢掉守卫。
    # BD/BE/BF 三条缺陷（pooling 除零、卷积 SIGFPE、Tile 堆溢出）都是这一族漏网的实例。
    ('A7 ReadParam 值域校验基线 (check_param_domain)',
     ['check_param_domain.py', '--selfcheck'], False),
    ('A8 ReadParam 值域校验基线比对 (check_param_domain)',
     ['check_param_domain.py', '--check-baseline'], False),
    # A9/A10 是 BE 的"同一类收口"门禁：BE 修了 NCHW 那 7 处 `/ strideH`，
    # 忘了 NCHWC 那 25 处，是 BG 的工具抓出来的。这两个组保证以后不会再漏。
    ('A9 "除以模型参数" 守卫普查自测 (check_div_guard --selfcheck)',
     ['check_div_guard.py', '--selfcheck'], False),
    ('A10 "除以模型参数" 守卫普查 (check_div_guard)',
     ['check_div_guard.py'], False),
    # A11/A12 是 BM 的门禁：BM 修了 ZQlibFaceID/ZQ_FaceRecognizerUtils.h 里
    # 两处没检查返回值的 cv::invert（失败时输出 Mat 是空的 -> 后面空指针解引用）。
    # 同一个动机：修了一处不等于只有这一处，靠人记得普查是靠不住的。
    ('A11 "丢弃 OpenCV bool 返回值" 自测 (check_uncked_cv_return --selfcheck)',
     ['check_uncked_cv_return.py', '--selfcheck'], False),
    ('A12 "丢弃 OpenCV bool 返回值" 普查 (check_uncked_cv_return)',
     ['check_uncked_cv_return.py'], False),
    # A13/A14 是附录 CC 的门禁：NCHWC 那一族的 _C3 内核把通道数 3 **硬编码**进
    # im2col 展开（实测传 C=4 / C=6，输出与 C=3 逐位相同 —— 不报错、不崩溃，
    # 只是安静地只算前 3 个通道，比崩溃更危险），所以每个调用点都得有 C==3 守卫。
    # 现状 29 个调用点全部有守卫；这个门禁保证以后新增调用点时不会漏。
    # **只管 NCHWC 那一族**：NCHW（layers_c）那一族也叫 _C3，但它是通用实现
    # （memcpy 按实际 filter_C），守的是 in_C <= 4 / <= 8，不要求恰好等于 3。
    ('A13 "NCHWC _C3 调用点的 C==3 守卫" 自测 (check_c3_guards --selfcheck)',
     ['check_c3_guards.py', '--selfcheck'], False),
    ('A14 "NCHWC _C3 调用点的 C==3 守卫" 普查 (check_c3_guards)',
     ['check_c3_guards.py'], False),
    # A15/A16 是附录 CM 的门禁：附录 CL 那处**越界读**就是这个形状 ——
    # 同一个函数里 x0 = __min(in_W - 1, …) 而 y0 = __min(in_H, …)，
    # 少减一个 -1 就读到最后一行之后。NCHW 的对应实现写的是 in_H - 1，
    # 同仓 A/B 一次就定性为笔误。现在它是一条常驻门禁。
    ('A15 "同函数内各轴钳位上界一致" 自测 (check_clamp_asymmetry --selfcheck)',
     ['check_clamp_asymmetry.py', '--selfcheck'], False),
    ('A16 "同函数内各轴钳位上界一致" 普查 (check_clamp_asymmetry)',
     ['check_clamp_asymmetry.py'], False),
    # A17/A18 是附录 IX 的门禁，两条源码级判定：
    #   1. `zq_final_sum_q` 横向归约宏在 ARM NEON + `__ARM_NEON_FP16` 那一节
    #      写成 9 项，而 `q` 只有 `q[8]` —— 越界读（IX.1）。
    #      **它只能用源码门禁盖**：那一节在 x86 上根本不参与编译
    #      （`zq_base_type` 是 float，SSE/AVX 两节分别是 4 项 / 8 项，都对），
    #      所以 ASan / UBSan / 跑 sample 全部碰不到它。
    #   2. 被 `zq_mm_store_ps` 写的**裸栈数组**没有对齐属性（IX.3）——
    #      不对齐在 x86 上是 `vmovaps` -> #GP，**换个调用点就崩**；
    #      x86 走的是 store，**ASan 那一轴全绿**，只有 UBSan 看得见。
    ('A17 "SIMD 归约项数 / 栈数组对齐" 自测 (check_mm_safety --selfcheck)',
     ['check_mm_safety.py', '--selfcheck'], False),
    ('A18 "SIMD 归约项数 / 栈数组对齐" 普查 (check_mm_safety)',
     ['check_mm_safety.py'], False),
    # A19/A20 是附录 IX.19 的门禁：`GetTopDim` 要算 `(kernel_H-1)*dilate_H`，
    # 这是 **int** 乘法，两个参数都来自模型文件。乘积回绕成负数 -> top_H 巨大 ->
    # SetShape 的 ChangeSize 失败而 **LayerSetup 不检查它的返回值** ->
    # 零尺寸张量 -> 空指针解引用。主副本 16 处有守卫（EM.3），
    # `ZQCNN/ZQ_CNN_Layer_NCHWC.h` 这一族**一处都没有** ——
    # 「一个副本有守卫、孪生副本没有」的第四次（IH.9 / BE.2 / IX.14 之后）。
    ('A19 "卷积 kernel/dilate 溢出守卫" 自测 (check_conv_overflow_guard --selfcheck)',
     ['check_conv_overflow_guard.py', '--selfcheck'], False),
    ('A20 "卷积 kernel/dilate 溢出守卫" 普查 (check_conv_overflow_guard)',
     ['check_conv_overflow_guard.py'], False),
    # A21/A22 是附录 IE 的门禁：`strcmp(typeid(T).name(), "double")` 在 GCC 上**恒为假**
    # （IEEE ABI 返回 "d"），于是 PCG / TaucsBase / PoissonSolver 等 21 个头里的
    # 分派分支**全部落到 else**（多半是 `return false`）——
    # 也就是「这些算法在 Linux 上一个数都算不出来，还不报错、不崩」。
    # 这是**跨平台行为不一致**，正对着「windows 和 linux 都能完全跑通」这条硬要求。
    ('A21 "strcmp(typeid(T).name()) 判类型" 自测 (check_typeid_name --selfcheck)',
     ['check_typeid_name.py', '--selfcheck'], False),
    ('A22 "strcmp(typeid(T).name()) 判类型" 普查 (check_typeid_name)',
     ['check_typeid_name.py'], False),
    # A23/A24 是附录 II 的门禁。`ZQCNN/ZQ_CNN_MTCNN*.h` 有**五份**逐字拷贝的 MTCNN 实现，
    # 这一轮在它们身上一次挖出**六个**问题，其中三个的共同点是
    # **「只有源码能看见，跑 sample 看不见」**：
    #   · `_Interface` 多线程 lnet：Forward 用 `lnet[thread_id]`，读 blob 却用 `lnet[0]`
    #     —— 仓内 sample 全部 thread_num=0（夹成 1），`lnet[0]==lnet[thread_id]`，
    #     跑一万次也看不出差别；
    #   · 同一段下颌/眼周 29 个点带一个**活的** `* 0.5`，单线程支路是注掉的、
    #     参考实现 `ZQ_CNN_MTCNN.h` 也是注掉的 —— 只有 thread_num>1 才走到；
    #   · 串行支路用 `omp_get_thread_num()` 索引大小**恰好是 thread_num** 的容器
    #     —— 仓内没有调用方把 Find 放进自己的 parallel 区，实测恒返回 0。
    # 另外 `ZQ_CNN_MTCNN_ncnn.h` **漏了**另外四份都有的 `pnet_size/pnet_stride`
    # `__max(1,...)` 夹取（上一轮修四份时漏的）—— 又一次「孪生副本」。
    # 证据：修复前后 4 个 MTCNN sample 的输出**逐字节相同**，
    # 这既说明修得对，也说明 sample 对这六条是瞎的 —— 所以只能源码级判定。
    ('A23 "MTCNN SetPara / 多线程索引一致性" 自测 (check_mtcnn_setpara --selfcheck)',
     ['check_mtcnn_setpara.py', '--selfcheck'], False),
    ('A24 "MTCNN SetPara / 多线程索引一致性" 普查 (check_mtcnn_setpara)',
     ['check_mtcnn_setpara.py'], False),
    # A25/A26 是附录 IJ 的门禁。`ZQCNN/ZQ_CNN_BBoxUtils.h`（759 行）是
    # MTCNN / CascadeOnet / SSD / MXNET-SSD **四条检测线共用的**几何底座，
    # 此前**零行为门禁**，而输入是「网络输出 + 模型文件」，两者都不可信。
    # 这一轮挖出六条，其中三条**不需要跑就能从公式推出矛盾**：
    #   · `_detection_output`（SSD 主路径）从不把 `num_priors` 和三个 blob 的长度对账，
    #     而 num_priors 来自 Layer 从 conf 的 H 推出来的**另一个张量**，
    #     Layer 只校验了 loc 的 C 和 conf 的 C、**没校验 prior 的 C** ——
    #     GetPriorBBoxes 要读 8*num_priors 就是堆越界**读**。
    #     铁证：同一份文件的 `_detection_output_MXNET` 早就有完整守卫。
    #   · `_nms` 的 IoU 混用两套面积口径（交集 +1、area 不带 +1），
    #     12x12 的框算出 **1.42**（>1）、1x1 算出 **-2**（永不抑制）、3x2 分母 **0**（除零）。
    #   · `it->area = (float)(row2 - row1)` 的减法在 int 里先算完再转 float，溢出是 UB。
    ('A25 "BBoxUtils NMS / 解码契约" 自测 (check_bbox_nms --selfcheck)',
     ['check_bbox_nms.py', '--selfcheck'], False),
    ('A26 "BBoxUtils NMS / 解码契约" 普查 (check_bbox_nms)',
     ['check_bbox_nms.py'], False),
    # A27/A28 是附录 IM 的门禁：每个头都必须能**单独**编过。
    # C++ 头文件的头号卫生问题是「用了某类型却没 include 它的定义」——
    # 主工程里每个 TU 都按习惯顺序 include 一堆头，某个头恰好排在提供方**前面**，
    # 于是看起来一切正常。本项目真实踩到的一例：`ZQ_CNN_CascadeOnet_Interface.h:113`
    # 用了具体的 `ZQ_CNN_Net`（只 include 了抽象基类 `ZQ_CNN_Net_Interface.h`），
    # 而 SampleVideoFaceDetection_Interface.cpp 第一行恰好 include 了 ZQ_CNN_Net.h，
    # **顺序正好把它盖住**，所以一直没人发现。
    # 注意它要跑 ~3 分钟（每个头一次 wsl 调用），所以只注册普查、不并进更快的那几组。
    ('A27 "头文件自包含性" 自测 (check_header_selfcontained --selfcheck)',
     ['check_header_selfcontained.py', '--selfcheck'], False),
    ('A28 "头文件自包含性" 普查 (check_header_selfcontained)',
     ['check_header_selfcontained.py'], False),
    # A29/A30 是附录 IN 的门禁：PersonPose / PersonPose2 / MouthDetector / FaceCropUtils
    # 四个头此前**零行为门禁**。本轮挖出 9 条，其中六条是**两个拷贝之间的差异** ——
    # 「单看一个文件是否合规」这种判据会漏：PersonPose.h 因为**已经有**溢出守卫而
    # "通过"，正好掩盖 PersonPose2.h 的缺失。所以判据要写成「两份都要有」。
    # 另注：ZQ_CNN_FaceCropUtils.h 是**已知的 GBK 文件**（上游 MFC 中文界面带来的），
    # 门禁按 gbk 读它。
    ('A29 "姿态/嘴部/人脸裁剪" 自测 (check_pose_mouth --selfcheck)',
     ['check_pose_mouth.py', '--selfcheck'], False),
    ('A30 "姿态/嘴部/人脸裁剪" 普查 (check_pose_mouth)',
     ['check_pose_mouth.py'], False),
    # A31/A32 是附录 IO 的门禁：ZQ_FaceDatabaseMaker / ZQ_FaceDetectorLibFaceDetect。
    # 这一轮的三条缺陷**一条都跑不到**：`MakeDatabase(` / `MakeDatabaseCompact(` 零调用方
    # （四个 SampleFaceDatabase* 只用 *AlreadyCropped 变体，恰好绕开 detectors[id] 那一支），
    # `_auto_detect_database` 的 #else(Linux) 分支从不链接（10 个 include 此头的 sample
    # 全部包在 #if defined(_WIN32) 里），GRAY 分支需要「灰度图 + roi_min_x > 0」
    # 而三个调用点全传 BGR。**回归全绿不代表这些路径验过了** —— 只能靠源码判据。
    ('A31 "人脸库构建 / libfacedetect 封装" 自测 (check_facedb_maker --selfcheck)',
     ['check_facedb_maker.py', '--selfcheck'], False),
    ('A32 "人脸库构建 / libfacedetect 封装" 普查 (check_facedb_maker)',
     ['check_facedb_maker.py'], False),
    # A33/A34 是附录 IQ 的门禁：SSD / CascadeOnet 检测线。
    # 最值得记的一条是 IQ.1 —— `ZQ_CNN_NSFW.h` 的 include guard 写成了
    # `_ZQ_CNN_SSD_H_`，与 ZQ_CNN_SSD.h **完全撞名**。任一 TU 同时 include 两者，
    # 第二个整份被跳过 -> `is not a member of ZQ`。`#pragma once` 救不了：
    # 它按**文件**生效，阻止 NSFW.h 的是那个撞名的 guard。
    ('A33 "SSD / CascadeOnet" 自测 (check_ssd_cascade --selfcheck)',
     ['check_ssd_cascade.py', '--selfcheck'], False),
    ('A34 "SSD / CascadeOnet" 普查 (check_ssd_cascade)',
     ['check_ssd_cascade.py'], False),
    # A35/A36 是附录 IR 的门禁：扫「一条语句后面粘着下一条」这种补丁脚本痕迹。
    # 本轮用 Python 批量改 C++ 时栽了 4 次（少一个换行就把两行粘成一行）。
    # 注意它**不改变语义**（`}` 后接声明仍是两条语句，照样编过），
    # 真正致命的是标识符被截断（编译器能抓）。所以定位是**可读性/一致性**，
    # 并且把仓库原有的 8 处同类写法列进白名单 —— 永远红的规则等于没有规则。
    ('A35 "语句粘连" 自测 (check_stmt_joins --selfcheck)',
     ['check_stmt_joins.py', '--selfcheck'], False),
    ('A36 "语句粘连" 普查 (check_stmt_joins)',
     ['check_stmt_joins.py'], False),
    # A37/A38 是附录 IS 的门禁：NCHWC 检测线的**守卫一致性**。
    # 这一族的**对齐契约是自洽的**（附录 IR 已推导），所以剩下的风险全在
    # 「一份对 N 份错的守卫」上：6 个 pooling 的早退块缺无条件 return（stride==0 ->
    # 整数 idiv 除零 SIGFPE）、packed 重载缺 filter_N != bias_C（内核满宽读 bias，
    # 最后一组读过缓冲末尾）。注意「没有 bias 形参的那两个 packed 重载**不该**有那条守卫」
    # —— 判据必须把「该有的」与「不该有的」分开，否则要么漏要么误报。
    ('A37 "NCHWC 守卫一致性" 自测 (check_nchwc_guards --selfcheck)',
     ['check_nchwc_guards.py', '--selfcheck'], False),
    ('A38 "NCHWC 守卫一致性" 普查 (check_nchwc_guards)',
     ['check_nchwc_guards.py'], False),
    ('B  ZQlib 独立回归测试 x10 (ASan+LSan)', ['run_zqlib_checks.py'], False),
    # 基线路径给**绝对路径**：子进程以 ROOT 为 cwd 运行，而基线文件在 tools/ 下，
    # 相对路径会解析成 <ROOT>/zqlib_probe_baseline.txt 而找不到（2026-10-02 实测）。
    ('C  ZQlib 可编译性门禁',
     ['probe_zqlib_headers.py', '--check-baseline',
      os.path.join(HERE, 'zqlib_probe_baseline.txt')], True),
    # ZQlibFaceID 的姊妹篇（附录 EH）。ZQlibFaceID 整个目录**既不在两个构建里、
    # 也没有任何门禁提到** —— 那 29 个头"从来没被编译过"。
    # 本门禁把"能被外部 SDK 满足的那些"逐个在 Linux 上编一遍，
    # 任何一个头从 OK 变成非 OK 就退出 1。
    # 变异测试确认它有鉴别力：回退附录 EG 的修复 -> OK 22 变 20。
    ('C1 ZQlibFaceID 可编译性门禁',
     ['probe_faceid_headers.py', '--check-baseline',
      os.path.join(HERE, 'faceid_probe_baseline.txt')], True),
    # C1 的**分类器**自测（附录 EU.6）。分类器自己不会失败，只会安静地把所有东西
    # 归进同一个桶 —— `'nn'` 那一条就曾让 MSVC_ONLY 与 BROKEN 两个桶从来没被填过。
    # 分类器和被它分类的对象一样需要门禁。
    ('C1b C1 分类器自测',
     ['probe_faceid_headers.py', '--selftest'], False),
    # 文件级可达性（附录 ET）。C/C1 问的是"这个**头**能不能单独编"，
    # 本门禁问的是另一个问题："这个**文件**有没有被任何构建编过"。
    # 两者互补 —— 附录 ES.2 那个同作用域重复声明，任何编译器都编不过，
    # 而它活下来正是因为 ZQ_OpticalFlow.h 离任何构建都有两跳：
    # **C/C1 会发现它编不过（因为它逐个单独编），
    #   但如果没人去编它，就永远不会有"编不过"这个事件发生。**
    ('C5 文件级可达性门禁（没有任何构建编过的文件）',
     ['probe_file_reachability.py', '--selftest', '--check-baseline',
      os.path.join(HERE, 'file_reach_baseline.txt')], True),
    # C5b（附录 EX）：MNN 转换器**分叉**出去的那份 ZQCNN 头。
    # 顶层 CMake 没有 add_subdirectory(ZQCNN_to_MNN)，转换器本体又要 MNN 的
    # MNN_generated.h，所以那 7 个头**从来没被任何编译器看过** —— 而它们落后主树
    # 三处已修的守卫（__min/__max 无定义、就地守卫整个缺失、卷积 stride/dilate
    # 无守卫）。本门禁逐头编一遍并断言那三处守卫还在，
    # 免得将来从主树同步时又悄悄丢掉。
    ('C5b MNN 转换器分叉头门禁',
     ['probe_mnn_fork.py', '--selftest'], False),
    # C6：用户 2026-10-01 明确要求「所有 sample 的 namedWindow / imshow / waitKey
    # 一律注释掉」（无头环境与自动化验证会阻塞），AGENTS.md 也有。
    # 那条规则一直成立，但**没有任何东西在守** —— 而违反它的症状是
    # Linux sample 回归挂住、或在没有显示器的机器上直接失败，
    # 离"有人加了一行 imshow"这个原因很远。
    #
    # 放在**慢组**：它要对 71 个 sample 各跑一次 `g++ -E`（去掉注释），
    # 实测约 4 分钟 —— 不适合进默认通道。
    # 组名用 **C7**：C4/C5/C6 已经被主流程里的
    # 「C4 主工程 HIGH 桶」「C5 主工程 -O2 优化期告警」「C6 ZQCNN 门禁 UBSan 回归」占用。
    # （C4 那次撞名是我自己犯的，注释里已记；这里直接避开。）
    ('C7 sample 不得含未注释的 GUI 调用（用户指令门禁）',
     ['check_no_gui_calls.py'], True),
    # C8（附录 GK）：**门禁自己必须能跑起来**。
    # `check_filecount_bounds.py` 的第 50 行是一句 6 空格缩进、没有 `#` 的
    # 残句（从一行被截断的注释尾巴里掉下来的），Python 在解析期就抛
    # SyntaxError —— 那个文件**一次都没被执行过**，而它正是附录 EL 的
    # 全部依据（audit_k3_20261001.md:13287 直接把结论建立在它身上）。
    # 回归之所以发现不了，是因为 40 多道门禁**互相不看对方**。
    # 放在快组：只做 compile()，不执行，50 个 .py 一秒内跑完。
    ('C8 门禁自身可解析（每个 .py / .sh）',
     ['check_gates_runnable.py', '--selftest'], False),
    # C8 顺带把 `check_filecount_bounds.py` 接进回归。基线的键是
    # **(文件, 变量)**：不含行号也不含分配调用名 —— 后者会被实测逼出来，
    # 见 UB_TEMPLATES 上面那段注释。
    ('C8b「读入 int -> 分配」站点基线（附录 EL）',
     ['check_filecount_bounds.py', '--check-baseline',
      os.path.join(HERE, 'filecount_baseline.txt')], False),
    # C9（附录 GL）：`-DBLAS_TYPE=...` 到底有没有真的生效。
    # 这个缺陷的性质是「**编得过、值不对**」，所以判据必须是**取值**而不是
    # 能不能编：头文件原来无条件 `#define ZQ_CNN_USE_BLAS_GEMM 0`，
    # 把 CMake 传来的 -D 静默按回去，编译一路绿灯而 `-DBLAS_TYPE=openblas`
    # 实际是个空操作（`build-with-cmake.md:50` 就是这么教的）。
    # 8 组配置排列 + Windows 分支写法，一次 wsl 调用跑完，约 7 秒。
    ('C9 配置宏取值（-DBLAS_TYPE 到底有没有生效）',
     ['check_blas_config.py', '--selftest'], False),
    # C10（附录 GM）：SSETYPE 四档（NONE / SSE / AVX / AVX2）**逐档**编译 + 行为。
    # 这道门禁的由来是两条反直觉的事实：
    #   1. 报告里记的 H4「`ZQ_CNN_SSETYPE_NONE` 编不过」是**错的** ——
    #      四档实测全部编译、链接、后向误差 1e-8。负结果也要钉住，
    #      否则会有人照着去"修"一个不存在的问题。
    #   2. `zq_gemm_32f_asm_core_m6n8` 缺前置声明这个**真缺陷**，
    #      **只在 SSETYPE=0/1 两档存在**；默认那两档（AVX/AVX2）
    #      连 warning 都没有。所以"默认档 sweep 干净"完全不能说明问题。
    # 放在慢组：四档各编 3 个 TU + 跑探针，实测约 2.5 分钟。
    ('C10 SSETYPE 四档逐档编译 + 行为（附录 GM）',
     ['check_ssetype_matrix.py', '--selftest'], True),
    # C11（附录 GN.3）：MSVC /analyze 门禁。
    # 这个工具**一直存在**（附录 AW），但：
    #   * AW 那一轮只跑了 7 个 TU，现在是 46 个 —— 多出来的部分里有
    #     4 条 C6011（malloc 未判空）这类真发现；
    #   * 它**从来没接进过回归**，所以"跑过一次"和"一直在跑"差着十万八千里。
    # 放在慢组：46 个 TU 的 /analyze 实测约 2~4 分钟，且只依赖 MSVC。
    ('C11 MSVC /analyze 基线（附录 AW / GN.3）',
     ['run_msvc_analyze.py', '--check-baseline',
      os.path.join(HERE, 'msvc_analyze_baseline.txt')], True),
    # C12（附录 GP）：ARM/NEON 分支的解析门禁。
    # 36 个 TU 的 `#if __ARM_NEON` 分支在本机**从来没有被任何编译器看过** ——
    # WSL 里没有 arm-linux-gnueabihf-gcc、也没有 clang，而仓库根的
    # `build.sh` 正是构建 armeabi-v7a 的。
    # 用一个 arm_neon.h 桩 + `-DZQ_CNN_USE_ARM_NEON` 让这些分支在 x86 上
    # 至少过一遍**解析**与**类型检查**（实测约 1 分钟，慢组）。
    # 它**不验 NEON 的类型与语义** —— 桩里向量全 typedef 成 float。
    ('C12 ARM/NEON 分支解析（附录 GP）',
     ['check_neon_branch.py', '--check-baseline',
      os.path.join(HERE, 'neon_branch_baseline.txt')], True),
    # C12b（附录 GQ）：同一批 TU 的 **FP16** 档
    # （`SIMD_ARCH_TYPE=arm64-fp16`，根 CMakeLists.txt:95 真实实现了、
    #  build-with-cmake.md 里却没写）。这一档 2026-10-04 之前**从未被编译过**：
    #  实测有 15 处「通用实现与 FP16 实现同时被编进来」的重复定义、
    #  2 处 `padK` 未声明，以及 `float16_t` 这个**全仓从未定义**的类型。
    #  已修的部分与**故意没修**的部分都记在基线里，每条带理由。
    ('C12b ARM/NEON FP16 档（附录 GQ）',
     ['check_neon_branch.py', '--fp16', '--check-baseline',
      os.path.join(HERE, 'neon_fp16_baseline.txt')], True),
    # C13（附录 GT）：data/ 里的图必须**扩展名与内容一致**且能解码。
    # 实测发现 data/mouth0.jpg、data/mouth1.jpg 是 **PNG 内容、.jpg 扩展名** ——
    # OpenCV 按内容嗅探所以当时还能读，但任何按扩展名分派的调用方
    # （IMREAD_JPEG、libjpeg 直连、第三方脚本）都会拿到错的东西。
    # 附录 B-3 记着 libjpeg 没装 setjmp 时畸形图片会让它直接 exit()，
    # 所以"图能不能解"不只是 sample 的事。
    ('C13 data/ 图像扩展名与可解码性（附录 GT）',
     ['check_data_images.py', '--selftest'], True),
    # C14（附录 GU）：`manualExportCaffe/` 的 8 个 MATLAB 导出脚本。
    # 它们写 `.nchwbin`（ZQCNN 权重格式），但本机**没有 MATLAB 也没有 Octave**，
    # 产物也一个都不在 `model/` 里 —— 所以这道门禁只能是一个**下限**：
    # 块配平 / layers 表形状 / 用到的 flag 都被 strcmp 处理过。
    # 类型码与 C++ 枚举是否对得上、权重字节数与 LoadBinary_NCHW 是否一致，
    # 那几条要真跑 MATLAB 才谈得上，门禁输出里明写了。
    ('C14 MATLAB 导出脚本结构（附录 GU）',
     ['check_export_scripts.py', '--selftest'], False),
    # C15（附录 GX）：层类型契约 —— **生产者写的层名，消费者必须认**。
    # 三个生产者（两个 Python 转换器 + 27 个随仓 .zqparams）与消费者
    # （ZQ_CNN_Layer*.h 的 ReadParam）**都在这个仓库里**，
    # 却没有任何东西在核对它们一致。改一边，另一边静默失效，
    # 症状是运行期一句 "load failed"，离原因隔着一整条转换链。
    # 已有的 FD/GH 门禁验的是"能不能解析并连通"，**不验层名在不在表里**。
    ('C15 层类型契约（生产者 vs 消费者）',
     ['check_layer_type_contract.py', '--selftest'], False),
    # C16（附录 HX.3）：「后一层读的 blob ≠ 前一层写的 blob」。
    # `_merge_bn` / `_merge_prelu` 折掉 BN/PReLU 的前提就是"它吃的是这个卷积的输出"。
    # `model/mobilefacenet-v1.zqparams` 第 108/109 行不满足，于是生产实参下
    # 它的输出被改了 0.37（后向误差）—— 而这条查了 HE~HX 共十一轮才定位。
    # 落成门禁的理由：**下一个模型再写出这种接线时，在加载之前就报出来**。
    ('C16 BN/PReLU 接线（附录 HX）',
     ['check_bn_prelu_pairing.py', '--selftest'], False),
]


WIN_BUILD = ['cmake', '--build', 'build_x64', '--config', 'Release']
LINUX_SAMPLES = ('cd /mnt/d/ZQCNN && bash tools/run_sample_regression.sh')

# 关键 sample：两个平台都要过（rc 必须为 0）。
# SampleGEMMAsmCompare 是汇编 vs intrinsic 的对拍，PASS 才算过；
# 其余是推理链路（MTCNN / SSD / CascadeOnet）。
WIN_SAMPLES = ['SampleGEMMAsmCompare.exe', 'SampleMTCNN.exe', 'SampleMTCNN_NCHWC4.exe',
               'SampleSSD.exe', 'SampleCascadeOnet.exe', 'SampleFaceDetectorMTCNN.exe',
               # 附录 HX（2026-10-05）：`merge_bn` / `merge_prelu` 的前向对照。
               # 之前**故意不接**：mobilefacenet-v1 上它恒红（后向误差 0.3695），
               # 恒红的检查会把别的真回归失败淹掉。HX 修掉根因后（17 个模型全过）
               # 它才接进来 —— 它守的是一条**生产路径**上的结果不变性。
               'SampleMergeBNCompare.exe',
               # 附录 HY（2026-10-05）：同一段守卫的**第二份拷贝**
               # （`ZQ_CNN_Net_NCHWC`），同样走生产路径
               # （`ZQ_CNN_MTCNN_NCHWC.h:109`），此前**零覆盖**。
               'SampleMergeBNCompareNCHWC.exe',
               # 附录 IA（2026-10-05）：15 类**没有任何随仓模型跑得到**的层类型，
               # 各自造合成网真跑一遍并与独立参考实现对拍。
               # 它当场抓出了 `zq_cnn_scale_32f_align` 带 bias 分支的堆越界读
               # （附录 IB）。
               'SampleUnusedLayerProbe.exe',
               # 附录 IN（2026-10-05）：`LSTM_TF` 的标定装置 ——
               # 36 种层类型里最后一个零覆盖项。自己写合成权重，不依赖随仓模型。
               'SampleLSTMTFCalib.exe']
WIN_BIN = os.path.join(ROOT, 'cmake-out-win32-x64', 'release', 'Release')

# Windows 侧的检出数下界（附录 GS.3，2026-10-04）。
#
# 为什么要和 Linux 侧一样：GS 那条下界原本**只加在 run_sample_regression.sh 里**，
# 也就是只管 Linux。而 Windows 侧恰恰是 MSVC/AVX2 特有的问题最容易出现的地方 ——
# 只在 Linux 断言检出数，等于把"换个编译器会不会坏"这个问题留在外面。
#
# 实测（2026-10-04，VS2022，仓库自带权重）：Windows 的四个数与 Linux
# **完全一致** —— 10 / 4 / 4 / 3。跨平台可复现性这一条因此也有了实测支撑。
#
# 下界同样取"实测值的一半（至少 1）"；`SampleGEMMAsmCompare.exe` 是纯 GEMM
# 基准，不适用。
#
# `SampleFaceDetectorMTCNN.exe` 2026-10-04 才拿到下界（实测 4 -> 2）：
# 它在 **Linux 上是平台桩**（只打一行 "only support windows"），
# 所以 `run_sample_regression.sh` 那边**刻意不给**下界 ——
# 给了会让 Linux 侧报 NOCOUNT 而红，那是误报。
# **同一个 sample 在两个平台上的适用性不同**，两张表因此不一样。
WIN_DETECT_FLOOR = {
    'SampleMTCNN.exe': 5,
    'SampleMTCNN_NCHWC4.exe': 2,
    'SampleSSD.exe': 2,
    'SampleCascadeOnet.exe': 1,
    'SampleFaceDetectorMTCNN.exe': 2,
    'SampleGEMMAsmCompare.exe': None,
}


def run_group(name, cmd, cwd=None, shell=False):
    print('=' * 74)
    print('### %s' % name)
    # 子进程直接写同一个 fd, 不 flush 的话它的输出会排在父进程缓冲的 print 之前,
    # 读起来是乱的（2026-10-02 实测）
    sys.stdout.flush()
    p = subprocess.run(cmd, cwd=cwd, shell=shell)
    sys.stdout.flush()
    ok = (p.returncode == 0)
    print('--- %s: %s' % (name, 'OK' if ok else 'FAILED (rc=%d)' % p.returncode))
    return ok


def run_build_group():
    ok = True
    ok &= run_group('D1 Windows 全量构建 (VS2022/cmake)',
                    WIN_BUILD, cwd=ROOT)
    ok &= run_group('D2 Linux 全量构建 (gcc/wsl)',
                    'wsl -d Ubuntu-20.04 -- bash -c "cd /tmp/zqb2 && make -j8"')
    ok &= run_group('D3 Linux sample 回归',
                    'wsl -d Ubuntu-20.04 -- bash -c "%s"' % LINUX_SAMPLES)
    for exe in WIN_SAMPLES:
        path = os.path.join(WIN_BIN, exe)
        if not os.path.isfile(path):
            print('--- Windows sample %s: MISSING (%s)' % (exe, path))
            ok = False
            continue
        # 注意: sample 必须在**产物目录**里跑（CMake 把 model/ 和 data/ 联接到了那里），
        # 从仓库根跑只会打一行 empty image，看着像跑过了其实什么都没验。
        #
        # 这里**不用 run_group**：它只看退出码、不接 stdout，
        # 而这里要读 `final found num:` —— 见 WIN_DETECT_FLOOR 的注释（附录 GS.3）。
        floor = WIN_DETECT_FLOOR.get(exe)
        print('=' * 74)
        print('### D4 Windows sample %s' % exe)
        sys.stdout.flush()
        p = subprocess.run([path], cwd=WIN_BIN, capture_output=True)
        out = (p.stdout or b'').decode('utf-8', 'replace') + \
              (p.stderr or b'').decode('utf-8', 'replace')
        sys.stdout.write(out)
        sys.stdout.flush()
        good = (p.returncode == 0)
        why = ''
        if good and floor:
            m = re.search(r'final found num:\s*(\d+)', out)
            if not m:
                good = False
                why = '应当打印 "final found num:"，实际没有（断言被绕过了？）'
            elif int(m.group(1)) < floor:
                good = False
                why = '检出 %s 张 < 下界 %d' % (m.group(1), floor)
            else:
                why = '检出 %s 张（下界 %d）' % (m.group(1), floor)
        print('--- D4 Windows sample %s: %s%s'
              % (exe, 'OK' if good else 'FAILED',
                 ('  ' + why) if why else ''))
        ok &= good
    return ok


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='跳过慢的可编译性门禁（约 2 分钟）')
    ap.add_argument('--with-build', action='store_true',
                    help='额外跑双平台全量构建 + sample 回归（很慢，几分钟）')
    ap.add_argument('--msvc-probe', action='store_true',
                    help='额外用 MSVC 探一遍 ZQlib 头（Windows 侧覆盖，见附录 AR）')
    ap.add_argument('--msvc-asan', action='store_true',
                    help='额外用 MSVC /fsanitize=address 把 9 个 ZQlib 测试在 Windows 上真跑一遍'
                         '（gcc 那套只在 WSL 里跑，见附录 AS）')
    ap.add_argument('--ubsan', action='store_true',
                    help='把 B 组换成 -fsanitize=undefined 再跑一遍（抓 ASan 看不见的'
                         '有符号溢出/移位越界等，见附录 AS.2）')
    ap.add_argument('--warn-sweep', action='store_true',
                    help='额外跑 gcc -Wall -Wextra 的 HIGH 桶门禁（较慢，约 2 分钟，见附录 AT）')
    ap.add_argument('--src-sweep', action='store_true',
                    help='额外扫**主工程** ZQCNN/ 的 43 个 TU 的 HIGH 桶（约 1 分钟，见附录 AU）')
    ap.add_argument('--reachability', action='store_true',
                    help='跑「层类型可达性」门禁（约 1 秒，见附录 DB）：'
                         '把 36 种已注册层类型分成 EXERCISED / COMMENTED / UNUSED，'
                         '并与基线比对。它是附录 DA.2 那次错判的产物 ——'
                         '「某条路径有没有被用到」从此跑一条命令就能复算，'
                         '不再靠手敲 grep（那次就是漏了 -i 而静默返回空）。')
    ap.add_argument('--ubsan-sweep', action='store_true',
                    help='把 A 组门禁用 **UBSan** 再跑一遍（约 3 分钟，见附录 CY）。'
                         'ASan 看不见未对齐 SIMD 访问、有符号溢出、移位越界这类 UB；'
                         '这一轴 2026-10-02 第一次真正跑起来（之前 UBSan 是可恢复的，'
                         'rc 恒为 0，等于没查），首跑就抓出 5 道门禁在用只 16 字节'
                         '对齐的缓冲喂 align256 入口。')
    ap.add_argument('--bounds-sweep', action='store_true',
                    help='额外跑 **-O2 -c** 的优化期告警 HIGH 桶门禁（约 2.5 分钟，见附录 CU）。'
                         '与 --src-sweep 的差别只有一处但是决定性的：--src-sweep 用 '
                         '-fsyntax-only，不出代码不做优化，所以 -Warray-bounds 这类'
                         '**依赖优化器值域传播**的告警在它那条轴上永远不响。')
    args = ap.parse_args()

    failed = []
    if args.with_build:
        if not run_build_group():
            failed.append('D 双平台构建 + sample 回归')

    if args.msvc_probe:
        bat = os.path.join(os.environ.get('TEMP', '.'), 'zqprobe_msvc.bat')
        with open(bat, 'w') as f:
            f.write('@echo off\r\n'
                    'call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community'
                    '\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul 2>&1\r\n'
                    'cd /d %s\r\n'
                    'python tools\\probe_zqlib_headers_msvc.py > "%%TEMP%%\\zqprobe_msvc_out.txt" 2>&1\r\n'
                    % ROOT)
        if not run_group('C2 MSVC 侧 ZQlib 头探测', ['cmd', '/c', bat], cwd=ROOT):
            failed.append('C2 MSVC 侧 ZQlib 头探测')
        out = os.path.join(os.environ.get('TEMP', '.'), 'zqprobe_msvc_out.txt')
        if os.path.isfile(out):
            try:
                sys.stdout.write(open(out, encoding='utf-8', errors='replace').read())
                rows = [l for l in open(out, encoding='utf-8', errors='replace')
                        if l.startswith(('OK', 'BROKEN'))]
                bad = [l for l in rows if l.startswith('BROKEN')]
                print('MSVC: %d 个头, OK %d, BROKEN %d'
                      % (len(rows), len(rows) - len(bad), len(bad)))
            except IOError:
                pass

    if args.warn_sweep:
        if not run_group('C3 gcc -Wall/-Wextra HIGH 桶门禁',
                         [sys.executable, os.path.join(HERE, 'warn_sweep_zqlib.py'),
                          '--check-baseline',
                          os.path.join(HERE, 'zqlib_warn_baseline.txt')],
                         cwd=ROOT):
            failed.append('C3 gcc -Wall/-Wextra HIGH 桶门禁')

    if args.src_sweep:
        if not run_group('C4 主工程 ZQCNN/ 的 HIGH 桶门禁',
                         [sys.executable, os.path.join(HERE, 'warn_sweep_src.py'),
                          '--check-baseline',
                          os.path.join(HERE, 'zqcnn_warn_baseline.txt')],
                         cwd=ROOT):
            failed.append('C4 主工程 ZQCNN/ 的 HIGH 桶门禁')

    if args.reachability:
        if not run_group('C7 层类型可达性门禁（EXERCISED/COMMENTED/UNUSED）',
                         [sys.executable, os.path.join(HERE, 'reachability_probe.py'),
                          '--check-baseline',
                          os.path.join(HERE, 'reachability_baseline.txt')],
                         cwd=ROOT):
            failed.append('C7 层类型可达性门禁')

    if args.ubsan_sweep:
        if not run_group('C6 ZQCNN 门禁 UBSan 回归',
                         [sys.executable, os.path.join(HERE, 'run_zqlib_checks.py'),
                          '--ubsan'], cwd=ROOT):
            failed.append('C6 ZQCNN 门禁 UBSan 回归')

    if args.bounds_sweep:
        if not run_group('C5 主工程 -O2 -c 优化期告警 HIGH 桶门禁',
                         [sys.executable, os.path.join(HERE, 'warn_sweep_bounds.py'),
                          '--check-baseline',
                          os.path.join(HERE, 'zqcnn_bounds_baseline.txt')],
                         cwd=ROOT):
            failed.append('C5 主工程 -O2 -c 优化期告警 HIGH 桶门禁')

    if args.msvc_asan:
        if not run_group('B2 ZQlib 独立回归测试 x10 (MSVC /fsanitize=address)',
                         ['cmd', '/c', os.path.join(HERE, 'run_zqlib_checks_msvc.bat')],
                         cwd=ROOT):
            failed.append('B2 ZQlib 独立回归测试 x10 (MSVC ASan)')

    for name, argv, slow in GROUPS:
        if slow and args.quick:
            print('=' * 74)
            print('### %s：--quick 跳过' % name)
            continue
        cmd = [sys.executable, os.path.join(HERE, argv[0])] + argv[1:]
        if name.startswith('B ') and args.ubsan:
            cmd.append('--ubsan')
        ok = run_group(name, cmd, cwd=ROOT)
        if not ok:
            failed.append(name)

    print('=' * 74)
    if not failed:
        print('ALL CHECKS PASSED')
        return 0
    print('%d CHECK GROUP(S) FAILED:' % len(failed))
    for n in failed:
        print('   %s' % n)
    return 1


if __name__ == '__main__':
    sys.exit(main())
