#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""普查 `_C3` 内核的每个调用点上方有没有 `C == 3` 守卫。

为什么需要这个门禁
------------------
`zq_cnn_conv_*_kernel2x2_C3` / `kernel3x3_C3` 这一族把通道数 **3 硬编码**进了
im2col 展开（`matrix_B_rows` 里的 `* 3`）。实测（审计报告附录 CB.2）：
传 C=4 或 C=6 进去，输出**与 C=3 逐位相同** —— 不报错、不崩溃，
只是安静地只算前 3 个通道。**这比崩溃更危险**，所以每个调用点都必须有守卫。

用法
----
    python tools/check_c3_guards.py              # 扫默认文件
    python tools/check_c3_guards.py --selfcheck  # 先自测（门禁里常驻这一组）
    python tools/check_c3_guards.py <文件路径>    # 扫指定文件

自测样本里**必须有一个故意不合格的项**（AGENTS.md「写检查类工具的四条硬规矩」第 1 条），
否则一个匹配逻辑坏掉的扫描器会绿着挡住后续所有同类缺陷。
"""
from __future__ import print_function

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILES = [
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_Forward_SSEUtils_NCHWC.cpp'),
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_Forward_SSEUtils.cpp'),
]

# 守卫里通道数可能用的名字。**这份清单来自实际代码**：
#   filter_C / in_C / need_C  —— 卷积主分派里的名字
#   C                         —— prepack 那几个函数里局部变量就叫 C
#                              （ZQ_CNN_Forward_SSEUtils_NCHWC.cpp:1705）
# 第一版只写了前三个，于是把上面那处误报成「没有守卫」——
# 而那处我本来手工读过、确认有守卫，正好拿来当自测样本。
# \b 保证 out_C / need_N 这类不会误匹配（下划线是单词字符）。
GUARD_RE = re.compile(r'\b(filter_C|in_C|need_C|filter_channel|C)\s*==\s*3\b')

# 调用点：一个 _C3 内核被**调用**（不是声明、不是定义）
CALL_RE = re.compile(r'^\s*(zq_cnn_\w*_C3(?:_with_bias(?:_prelu)?)?)\s*\(')

# **只管 NCHWC 那一族。**
#
# 两族都叫 "_C3"，含义却不同 —— 这是第一版最大的一个错误：
#   * NCHWC（layers_nchwc/zq_cnn_convolution_gemm_nchwc_raw.h）：
#     im2col 把通道数 3 硬编码进展开（`matrix_B_rows` 里的 `* 3`，
#     实测传 C=4/C=6 输出与 C=3 逐位相同），所以调用点必须有 `C == 3` 守卫
#   * NCHW（layers_c/zq_cnn_convolution_gemm_32f_align_c_raw.h）：
#     那一族是**完全通用的** —— `memcpy(dst, src, sizeof(float)*filter_C)`、
#     `cp_dst_ptr += filter_C`、`padded_len` 按实际 C 推导。
#     它的 "_C3" 指的是**小 C 变体**（调用侧守的是 `in_C <= 4` / `in_C <= 8`），
#     **不要求 C 恰好等于 3**。
# 不加这个限定的话，ZQ_CNN_Forward_SSEUtils.cpp 里那 3 处会被误报成缺守卫。
#
# 注意**不能**写成 r'\bnchwc'：内核名是 `..._gemm_nchwc4_kernel3x3_C3`，
# 下划线是单词字符，`_` 与 `n` 之间**没有**词边界，`\bnchwc` 一条都匹配不上
# （自测第一遍就是这么全 BAD 抓出来的 —— 扫出 0 个调用点时它会直接报
#  "匹配逻辑多半坏了"）。
NCHWC_ONLY = re.compile(r'nchwc')

# 往上看多少行找守卫
LOOKBACK = 12


def scan_text(text, lookback=LOOKBACK):
    """返回 (NCHWC 的 _C3 调用点总数, 没有 C==3 守卫的调用点列表[(行号, 内核名)])。

    NCHW 那一族（`zq_cnn_conv_no_padding_gemm_32f_align*_C3`）**不计入** ——
    它的 im2col 按实际 C 做 memcpy，是通用实现，见 NCHWC_ONLY 的说明。
    """
    lines = text.split('\n')
    total, bad = 0, []
    for i, line in enumerate(lines):
        m = CALL_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        if not NCHWC_ONLY.search(name):
            continue
        total += 1
        ctx = '\n'.join(lines[max(0, i - lookback):i])
        if not GUARD_RE.search(ctx):
            bad.append((i + 1, name))
    return total, bad


SELFCHECK_CASES = [
    # (说明, 源码片段, 期望的"没守卫"数量, 期望扫到的调用点数)
    ('NCHWC 标准写法 filter_C == 3',
     'if (filter_C == 3)\n{\n\tzq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_C3(a, b);\n}\n', 0, 1),
    ('NCHWC 局部变量就叫 C（第一版正则漏掉的正是这一条）',
     'if (C == 3)\n{\n\tzq_cnn_convolution_gemm_nchwc4_prepack8_other_kernel3x3_C3(a, b);\n}\n', 0, 1),
    ('NCHWC in_C == 3 且带后缀 _with_bias_prelu',
     'if (in_C == 3)\n\tzq_cnn_conv_no_padding_gemm_nchwc8_kernel2x2_C3_with_bias_prelu(a, b);\n', 0, 1),
    ('故意不合格：完全没有守卫 —— 必须被抓出来',
     'zq_cnn_conv_no_padding_gemm_nchwc4_kernel2x2_C3(a, b);\n', 1, 1),
    ('故意不合格：守卫是 C == 4（等式方向对但值不对）',
     'if (C == 4)\n{\n\tzq_cnn_conv_no_padding_gemm_nchwc8_kernel3x3_C3(a, b);\n}\n', 1, 1),
    ('故意不合格：out_C == 3 不算守卫（不是 filter_C/in_C）',
     'if (out_C == 3)\n{\n\tzq_cnn_conv_no_padding_gemm_nchwc1_kernel2x2_C3(a, b);\n}\n', 1, 1),
    # 下面三条是 NCHW 那一族：必须**一个都不计入**。
    # 它们守的是 in_C <= 4 / in_C <= 8，im2col 按实际 C 做 memcpy，是通用实现。
    ('NCHW 小 C 变体守的是 in_C <= 4 —— 不该被算成 NCHWC 调用点',
     'if (in_C <= 4)\n{\n\tzq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_C3(a, b);\n}\n', 0, 0),
    ('NCHW 256bit 变体 in_C <= 8 —— 同上',
     'if (in_C <= 8)\n{\n\tzq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep_C3(a, b);\n}\n', 0, 0),
    ('NCHW 连守卫都没有也不该被报 —— 它根本不是 NCHWC 内核',
     'zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_C3(a, b);\n', 0, 0),
]


def selftest():
    ok = True
    for name, src, expect_bad, expect_total in SELFCHECK_CASES:
        total, bad = scan_text(src)
        got_bad, got_total = len(bad), total
        mark = 'OK ' if (got_bad == expect_bad and got_total == expect_total) else 'BAD'
        if mark == 'BAD':
            ok = False
        print('  [%s] %-56s 期望 没守卫 %d/调用点 %d，实得 %d/%d'
              % (mark, name, expect_bad, expect_total, got_bad, got_total))
    return ok


def main(argv):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

    if '--selfcheck' in argv:
        print('check_c3_guards 自测：')
        if not selftest():
            print('**自测没过 —— 匹配逻辑已坏，先修它再谈别的**')
            return 1
        print('自测通过')
        return 0

    files = [a for a in argv[1:] if not a.startswith('-')]
    if not files:
        files = [f for f in DEFAULT_FILES if os.path.exists(f)]

    grand_total, grand_bad = 0, []
    for f in files:
        text = io.open(f, encoding='utf-8').read()
        total, bad = scan_text(text)
        grand_total += total
        grand_bad += [(f, ln, nm) for ln, nm in bad]
        print('%-56s %3d 个 _C3 调用点，%d 个没守卫'
              % (os.path.relpath(f, ROOT), total, len(bad)))
        for ln, nm in bad:
            print('      !! 第 %d 行 %s' % (ln, nm))

    print('合计 %d 个 **NCHWC** _C3 调用点，%d 个缺 C==3 守卫'
          % (grand_total, len(grand_bad)))
    if not grand_total:
        print('**一个调用点都没扫到 —— 匹配逻辑多半坏了**（AGENTS.md 坑 #2）')
        return 1
    if grand_bad:
        print('**_C3 会只算前 3 个通道而不报错，必须逐个补 C==3 守卫**')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
