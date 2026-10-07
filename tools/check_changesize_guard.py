#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""门禁：`ChangeSize` 的每一份实现都必须在动手分配之前**拒掉非正尺寸**。

起因（2026-10-07，附录 JL）
------------------------------
调查「负宽/负高会不会一路传到 SIMD 内核」时的结论：

* `ConvertFromBGR(w, h, ...)` 的第一件事就是
  `ChangeSize(1, _height, _width, 3, 1, 1)`；
* `ChangeSize` 的三个 NCHW 实现（`ZQ_CNN_Tensor4D.cpp:174 / :880 / :1654`）
  与三个 NCHWC 实现（`ZQ_CNN_Tensor4D_NCHWC.cpp:134 / :593 / :1054`）
  **都有** `if (dst_N < 0 || dst_H < 0 || dst_W < 0 || dst_C < 0) return false;`。

所以 `ConvertFromBGR(负宽, 负高)` 是**在入口被挡住的**，
不会一路传到内核 —— 附录 IN.6 记的那条链（反向框 -> `rect_w = -2e9` ->
`buffer_size = size_H*size_W*3` 为正 -> `buffer_size <= 0` 挡不住）之所以没炸，
靠的就是这几行。

**但整条防线就压在这 6 行上。** 任何一份实现把那一句删掉，
负尺寸就会一路走到分配与写入，而且**不会有任何测试立刻变红**。

判据
----
1. 每个 `::ChangeSize(int dst_N, int dst_H, int dst_W, int dst_C, ...)` 定义，
   在**函数体前若干行内**必须出现对尺寸的 `< 0` 拒绝；
2. 实现份数必须与 `EXPECTED_IMPLS` 一致 —— 少一份说明有人删了实现，
   多一份说明新加了实现而没照着写。
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCAN_FILES = [
    'ZQCNN/ZQ_CNN_Tensor4D.cpp',
    'ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp',
    'ZQCNN/ZQ_CNN_Tensor4D_NHW_C_Align128bit.h',
    'ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h',
]
EXPECTED_IMPLS = 6          # NCHW 3 + NCHWC 3（2026-10-07 实测）
DEF_RE = re.compile(
    r'::ChangeSize\s*\(\s*int\s+dst_N\s*,\s*int\s+dst_H\s*,\s*int\s+dst_W\s*,'
    r'\s*int\s+dst_C')
# 至少要拒掉 N/H/W/C 四个里的若干个；写成"至少命中一次 < 0 的尺寸守卫"
GUARD_RE = re.compile(r'dst_[NHWC]\s*<\s*0')
LOOKAHEAD = 14              # 函数头之后看多少行（守卫必须在动手之前）


def strip_comments(src):
    out, i, n = [], 0, len(src)
    while i < n:
        if src.startswith('/*', i):
            j = src.find('*/', i + 2)
            j = n if j < 0 else j + 2
            out.append(''.join(c if c == '\n' else ' ' for c in src[i:j]))
            i = j
            continue
        if src.startswith('//', i):
            j = src.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i))
            i = j
            continue
        out.append(src[i])
        i += 1
    return ''.join(out)


def scan_text(text, rel):
    lines = strip_comments(text).split('\n')
    bad = []
    total = 0
    for idx, ln in enumerate(lines):
        if not DEF_RE.search(ln):
            continue
        total += 1
        window = '\n'.join(lines[idx:idx + LOOKAHEAD])
        if not GUARD_RE.search(window):
            head = ln.strip()[:60]
            bad.append((rel, idx + 1, head))
    return total, bad


def collect():
    total, bad = 0, []
    for rel in SCAN_FILES:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            t, b = scan_text(f.read(), rel)
            total += t
            bad += b
    return total, bad


def selftest():
    good = ('bool X::ChangeSize(int dst_N, int dst_H, int dst_W, int dst_C, int bW, int bH)\n'
            '{\n'
            '\tif (dst_N < 0 || dst_H < 0 || dst_W < 0 || dst_C < 0)\n'
            '\t\treturn false;\n'
            '\t// ...\n'
            '\treturn true;\n'
            '}\n')
    noguard = ('bool Y::ChangeSize(int dst_N, int dst_H, int dst_W, int dst_C, int bW, int bH)\n'
               '{\n'
               '\t__int64 sz = (__int64)dst_H * dst_W;\n'
               '\treturn true;\n'
               '}\n')
    late = ('bool Z::ChangeSize(int dst_N, int dst_H, int dst_W, int dst_C, int bW, int bH)\n'
            '{\n'
            + '\tint pad = 0;\n' * 20
            + '\tif (dst_W < 0) return false;\n'
            '\treturn true;\n'
            '}\n')
    cases = [
        ('有负尺寸守卫 -> 合规', scan_text(good, 't.cpp')[0], 1),
        ('有实现但没守卫 -> 必须报', len(scan_text(noguard, 't.cpp')[1]), 1),
        ('守卫写得太晚（前面已经算过 size）-> 必须报',
         len(scan_text(late, 't.cpp')[1]), 1),
    ]
    bad = []
    for name, got, expect in cases:
        ok = got == expect
        print('  [%s] %-44s expect=%d got=%d'
              % ('PASS' if ok else 'FAIL', name, expect, got))
        if not ok:
            bad.append(name)
    if bad:
        print('SELFTEST FAILED: %s' % ', '.join(bad))
        return 1
    print('selftest OK: %d cases' % len(cases))
    return 0


def main():
    if '--selftest' in sys.argv:
        return selftest()
    total, bad = collect()
    print('找到 ChangeSize 实现：%d 份（期望 %d）' % (total, EXPECTED_IMPLS))
    if total != EXPECTED_IMPLS:
        print('  * 实现份数与 EXPECTED_IMPLS 不符：少一份可能有人删了实现，'
              '多一份可能新加了实现而没照着写')
    for rel, ln, head in bad:
        print('  * %s:%d  %s' % (rel, ln, head))
        print('      -> 必须在动手分配/算尺寸之前拒掉 dst_N/H/W/C 的负值')
    if bad or total != EXPECTED_IMPLS:
        print('')
        print('整条「负尺寸不得进入内核」的防线就压在这几行守卫上；')
        print('删掉它不会有任何测试立刻变红。')
        return 1
    print('')
    print('OK: %d 份 ChangeSize 实现都在分配之前拒掉负尺寸' % total)
    return 0


if __name__ == '__main__':
    sys.exit(main())