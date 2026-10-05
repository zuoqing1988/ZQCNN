#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NCHWC 检测线的守卫一致性门禁 —— 附录 IS。

范围：`ZQCNN/ZQ_CNN_Forward_SSEUtils_NCHWC.cpp`。

为什么需要它
------------
这一族的**对齐契约是自洽的**（附录 IR 已推导），所以剩下的风险全在
「**一份对 N 份错**的守卫」上：

* **IS.1 / IR.1** `MaxPooling` / `AVGPooling` 的 6 个（`NCHWC1/4/8` × `Max/Avg`）里，
  `if (need_W <= 0 || need_H <= 0)` 那个早退块**必须有无条件 `return`** ——
  `ChangeSize(0,0,0,0,0,0)` 是**成功**的，少了它就会走到 `(in_H - kernel_H) % stride_H`，
  `stride_H == 0` 时是**整数 idiv 除零 -> SIGFPE**。
  **NCHW 版有**那行（`ZQ_CNN_Forward_SSEUtils.h:1601`），NCHWC 版原来漏了。
* **IS.2 / IR.2** 带 `bias` 形参的 packed 重载**必须**有 `filter_N != bias_C`
  —— 内核按 `zq_mm_load_ps(bias + out_c)` 满宽读 bias，而 bias 只有
  `ceil(bias_C/align)*align` 个 float。**同文件的 unpacked 重载有**（`:2412` 等 4 处）。
  **没有 bias 形参的那两个重载**（`InnerProductWithPReLU` / `ConvolutionWithBiasPReLU`）
  **不该**有这条 —— 它们只有 slope，补了就是编译不过。

「不该有的守卫」和「该有的守卫」必须分开判，否则要么漏、要么误报
（这正是 A9/A31 那些规则的教训）。

用法
----
    python tools/check_nchwc_guards.py            # 扫
    python tools/check_nchwc_guards.py --selfcheck  # 先自测
"""
from __future__ import print_function

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_Forward_SSEUtils_NCHWC.cpp')

# IS.1: `if (need_W <= 0 || need_H <= 0)` 的早退块里，`ChangeSize(0,...)` 之后
# 必须还有一个**不在 if 里面**的 return。
# 条件里**没有分号**（第一版写成 `[^;{]*;\s*\{`，结果一条都匹配不上 ——
# 「判据一条都没匹配」和「判据发现问题」在报告里长得一模一样，这是第 N 次）。
# 条件与 `{` 之间只允许空白与注释（注释已被 strip_comments 变成空格）。
NEED_BLOCK = re.compile(
    r'if\s*\(\s*need_W\s*<=\s*0\s*\|\|\s*need_H\s*<=\s*0\s*\)\s*'
    r'\{(?P<body>[^{}]*)\}', re.S)
CHANGE0 = re.compile(r'if\s*\(\s*!output\.ChangeSize\(0,\s*0,\s*0,\s*0,\s*0,\s*0\)\)')
UNCOND_RETURN = re.compile(r'^\s*return\s*;?\s*$', re.M)

# IS.2: packed 重载
# 形参区：从 `Buffer& packedfilters, int filter_N,` 往后到函数体的 `{`。
PACKED_SIG = re.compile(
    r'const ZQ_CNN_Tensor4D_NCHWC::Buffer&\s*packedfilters\s*,\s*int\s*filter_N\s*,'
    r'(?P<sig>[^;{]*?)\s*\{', re.S)
HAS_BIAS = re.compile(r'&\s*bias\s*,')


def read(p):
    with io.open(p, 'r', encoding='utf-8') as f:
        return f.read()


def strip_comments(text):
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == '\\' else 1
            j = min(j + 1, n)
            out.append(text[i:j])
            i = j
            continue
        if c == '/' and i + 1 < n:
            nx = text[i + 1]
            if nx == '/':
                j = text.find('\n', i)
                j = n if j < 0 else j
                out.append(' ' * (j - i))
                i = j
                continue
            if nx == '*':
                j = text.find('*/', i + 2)
                j = n if j < 0 else j + 2
                out.append(''.join(ch if ch == '\n' else ' ' for ch in text[i:j]))
                i = j
                continue
        out.append(c)
        i += 1
    return ''.join(out)


def scan(t):
    bad, ok = [], 0

    # ---- IS.1 ----
    n_early = 0
    miss = []
    for m in NEED_BLOCK.finditer(t):
        n_early += 1
        body = m.group('body')
        cm = CHANGE0.search(body)
        if not cm:
            continue
        # `ChangeSize(0,...)` 之后必须还有**第二个** return。
        # 注意不能直接搜整段：第一个 return 是在 `if (!ChangeSize(...))` **里面**的，
        # 只剩它时也会被当成「无条件 return」而误判为合格（自测立刻发现）。
        tail = body[cm.end():]
        tail = re.sub(r'^\s*return\s*;?\s*$', '', tail, count=1, flags=re.M)
        if not UNCOND_RETURN.search(tail):
            miss.append(t[:m.start()].count(chr(10)) + 1)
    if n_early == 0:
        bad.append(('IS.1', '一个 `need_W <= 0 || need_H <= 0` 早退块都没找到 —— '
                              '这个文件结构变了，判据需要更新'))
    elif miss:
        bad.append(('IS.1', '第 %s 行的早退块**缺无条件 return** —— '
                    '`ChangeSize(0,0,0,0,0,0)` 是**成功**的，少了它就会走到 '
                    '`(in_H - kernel_H) %% stride_H`，stride_H==0 时是整数 idiv 除零 -> SIGFPE。'
                    'NCHW 版（ZQ_CNN_Forward_SSEUtils.h:1601）有那行。'
                    % ', '.join(str(x) for x in miss)))
    else:
        ok += 1

    # ---- IS.2 ----
    packed = list(PACKED_SIG.finditer(t))
    with_bias = [m for m in packed if HAS_BIAS.search(m.group('sig'))]
    without_bias = [m for m in packed if not HAS_BIAS.search(m.group('sig'))]
    if not packed:
        bad.append(('IS.2', '一个 packed 重载都没找到 —— 判据需要更新'))
    else:
        miss2 = [t[:m.start()].count(chr(10)) + 1 for m in with_bias
                 if 'filter_N != bias_C' not in t[m.end():m.end() + 3000]]
        # 反向：没有 bias 形参的**不该**有那条守卫
        over = [t[:m.start()].count(chr(10)) + 1 for m in without_bias
                if 'filter_N != bias_C' in t[m.end():m.end() + 3000]]
        if miss2:
            bad.append(('IS.2', '第 %s 行的 packed 重载**缺 `filter_N != bias_C`** —— '
                        '内核按 `zq_mm_load_ps(bias + out_c)` 满宽读 bias，而 bias 只有 '
                        'ceil(bias_C/align)*align 个 float，最后一组会读过缓冲末尾。'
                        '同文件的 unpacked 重载有（:2412 等 4 处）。'
                        % ', '.join(str(x) for x in miss2)))
        elif over:
            bad.append(('IS.2', '第 %s 行的 packed 重载**多了** `filter_N != bias_C` —— '
                        '它根本没有 bias 形参（只有 slope），这条守卫是**编译不过**的。'
                        % ', '.join(str(x) for x in over)))
        else:
            ok += 1
    return ok, bad


# ------------------------------------------------------------------ 自测
FULL = """
if (need_W <= 0 || need_H <= 0)
{
    if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
        return;
    return;
}
bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProductWithBias(ZQ_CNN_Tensor4D_NCHWC4& input,
    const ZQ_CNN_Tensor4D_NCHWC::Buffer& packedfilters, int filter_N,
    const ZQ_CNN_Tensor4D_NCHWC4& bias,
    ZQ_CNN_Tensor4D_NCHWC4& output, void** buffer, __int64* buffer_len)
{
    int bias_C = bias.GetC();
    if (in_N <= 0) { return true; }
    if (filter_N != bias_C)
        return false;
    return true;
}
bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProductWithPReLU(ZQ_CNN_Tensor4D_NCHWC4& input,
    const ZQ_CNN_Tensor4D_NCHWC::Buffer& packedfilters, int filter_N,
    const ZQ_CNN_Tensor4D_NCHWC4& slope,
    ZQ_CNN_Tensor4D_NCHWC4& output, void** buffer, __int64* buffer_len)
{
    return true;
}
"""

SELFCHECK = [
    ('全合格', FULL, []),
    ('IS.1 早退块缺无条件 return',
     FULL.replace('        return;\n    return;\n', '        return;\n'), ['IS.1']),
    ('IS.2 packed 缺 bias_C 守卫',
     FULL.replace('    if (filter_N != bias_C)\n        return false;\n', ''), ['IS.2']),
    ('IS.2 没 bias 形参却加了守卫',
     FULL.replace('    ZQ_CNN_Tensor4D_NCHWC4& output, void** buffer, __int64* buffer_len)\n{\n    return true;',
                  '    ZQ_CNN_Tensor4D_NCHWC4& output, void** buffer, __int64* buffer_len)\n{\n'
                  '    if (filter_N != bias_C) return false;\n    return true;'), ['IS.2']),
    ('阴性：一个都没有也报（判据更新信号）',
     'int x = 1;\n', ['IS.1', 'IS.2']),
]


def selfcheck():
    bad = 0
    for name, text, expect in SELFCHECK:
        _, b = scan(strip_comments(text))
        got = sorted(set(c for c, _ in b))
        if got != sorted(set(expect)):
            bad += 1
            print('  [self-MISMATCH] %s' % name)
            print('      expect %s, got %s' % (sorted(expect), got))
            for c, m in b:
                print('        %s: %s' % (c, m))
        else:
            print('  [self-OK]      %-34s %s' % (name, ','.join(got) if got else 'clean'))
    if bad:
        print('selfcheck FAILED: %d / %d mismatch' % (bad, len(SELFCHECK)))
        return 1
    print('selfcheck OK: %d cases, all as expected' % len(SELFCHECK))
    return 0


def main(argv):
    if '--selfcheck' in argv:
        return selfcheck()
    ok, b = scan(strip_comments(read(SRC)))
    if b:
        print('FAIL ZQ_CNN_Forward_SSEUtils_NCHWC.cpp')
        for c, m in b:
            print('       - %s: %s' % (c, m))
        print('合计合格判定 %d 项' % ok)
        return 1
    print('OK   ZQ_CNN_Forward_SSEUtils_NCHWC.cpp（%d 项）' % ok)
    print('NCHWC 守卫一致性: OK')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
