#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扫「同一个函数里，不同轴的边界钳位上界不一致」—— 附录 CM 的门禁。

为什么需要这个
--------------
附录 CL 查出的那处**越界读**就是这个形状（`zq_cnn_resize_without_safeborder_nchwc*`）：

    x0[w] = __min(in_W - 1, __max(0, x0[w]));    // 上界 in_W - 1   对
    y0    = __min(in_H,     __max(0, y0));       // 上界 in_H       错，漏了 -1

同一个函数里 W / H / C 三条轴访问的合法下标上界**应当是同一个**（都取最后一个合法下标），
少减一个 `- 1` 就会读到最后一行/列**之后**。
NCHW 的对应实现（`zq_cnn_resize_32f_align_c_raw.h`，5 处）写的是 `in_H - 1`，
**同仓 A/B 一次就定性为笔误**（AGENTS.md「同仓的两份实现互为对照」）。

用法
----
    python tools/check_clamp_asymmetry.py              # 扫 ZQCNN/ 下所有 raw 头与 32f_align_c 的 .c
    python tools/check_clamp_asymmetry.py --selfcheck  # 先自测（门禁里常驻这一组）

**自测样本里必须有"一个应该被抓出来的"**（AGENTS.md「写检查类工具的四条硬规矩」第 1 条），
而且最好是**从真实缺陷反推**的 —— 这里用的就是附录 CL 修好之前的那两行。
"""
from __future__ import print_function

import io
import os
import re
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ZQCNN = os.path.join(ROOT, 'ZQCNN')
SUBDIRS = ['layers_nchwc', 'layers_c', 'math', '']

# 一个钳位：__min(in_H - 1, ...) / __max(0, ...) / MIN(in_H - 1, ...)
PAT = re.compile(
    r'(__min|__max|\bMIN|\bMAX|_min|_max)\s*\(\s*'
    r'(in|out|filter|kernel|src|dst)?_?\s*'
    r'([WHCN])\s*(-\s*(\d+))?\s*,')


def scan_text(text):
    """返回 [(文件名占位, 函数名, 钳位名, {轴: 偏移集合})]，只保留**不一致**的那些。"""
    lines = text.split('\n')
    fn = '?'
    byfn = defaultdict(list)
    for i, l in enumerate(lines):
        m = re.match(r'\s*void\s+(zq_cnn_\w+)', l)
        if m:
            fn = m.group(1)
        for mm in PAT.finditer(l):
            axis = mm.group(3)
            off = int(mm.group(5)) if mm.group(5) else 0
            byfn[fn].append((i + 1, axis, off, mm.group(1)))
    out = []
    for f, rs in byfn.items():
        per = defaultdict(dict)
        for ln, ax, off, cl in rs:
            per[cl].setdefault(ax, set()).add(off)
        for cl, axes in per.items():
            vals = {a: tuple(sorted(v)) for a, v in axes.items()}
            if len(vals) >= 2 and len(set(vals.values())) > 1:
                out.append((f, cl, vals))
    return out


def target_files():
    fs = []
    for d in SUBDIRS:
        p = os.path.join(ZQCNN, d) if d else ZQCNN
        if not os.path.isdir(p):
            continue
        for name in sorted(os.listdir(p)):
            if name.endswith('_raw.h') or name.endswith('_32f_align_c.c'):
                fp = os.path.join(p, name)
                if os.path.isfile(fp):
                    fs.append((os.path.join(d, name) if d else name, fp))
    return fs


# 自测样本：(说明, 代码, 期望检出的不对称数)
# 样本 1 与 2 直接取自附录 CL 的真实代码（修好 / 没修好两个版本）
SELFCHECK = [
    ('附录 CL 修复前：y0 漏了 -1（必须被抓出来）',
     'void zq_cnn_demo_f(\n'
     '  int in_W, int in_H)\n'
     '{\n'
     '\tx0[w] = __min(in_W - 1, __max(0, x0[w]));\n'
     '\ty0    = __min(in_H,     __max(0, y0));\n'
     '}\n', 1),
    ('同一处修好之后：两轴都是 -1（必须不再报）',
     'void zq_cnn_demo_g(\n'
     '  int in_W, int in_H)\n'
     '{\n'
     '\tx0[w] = __min(in_W - 1, __max(0, x0[w]));\n'
     '\ty0    = __min(in_H - 1, __max(0, y0));\n'
     '}\n', 0),
    ('故意不合格：y 写成 H - 2（必须被抓出来）',
     'void zq_cnn_demo_h(\n'
     '  int in_W, int in_H)\n'
     '{\n'
     '\tx0[w] = __min(in_W - 1, __max(0, x0[w]));\n'
     '\ty0    = __min(in_H - 2, __max(0, y0));\n'
     '}\n', 1),
    ('只有一条轴 —— 没有可比对象（不该报）',
     'void zq_cnn_demo_i(\n'
     '  int in_W)\n'
     '{\n'
     '\tx0[w] = __min(in_W - 1, __max(0, x0[w]));\n'
     '}\n', 0),
    ('__min 与 __max 是两回事，分别比（不该报）',
     'void zq_cnn_demo_j(\n'
     '  int in_W, int in_H)\n'
     '{\n'
     '\tx0[w] = __min(in_W - 1, __max(0, x0[w]));\n'
     '\ty0    = __min(in_H,     __max(0, y0));\n'
     '}\n', 1),
]


def selftest():
    ok = True
    for name, src, expect in SELFCHECK:
        got = len(scan_text(src))
        mark = 'OK ' if got == expect else 'BAD'
        if got != expect:
            ok = False
        print('  [%s] %-46s 期望 %d 处，实得 %d 处' % (mark, name, expect, got))
    return ok


def main(argv):
    if '--selfcheck' in argv:
        print('check_clamp_asymmetry 自测：')
        if not selftest():
            print('**自测没过 —— 匹配逻辑已坏，先修它再谈别的**')
            return 1
        print('自测通过')
        return 0

    files = target_files()
    total = 0
    for label, fp in files:
        hits = scan_text(io.open(fp, encoding='utf-8-sig').read())
        for fn, cl, vals in hits:
            total += 1
            desc = '  '.join('%s:%s' % (a, ('-' + str(v[0]) if len(v) == 1 and v[0] else str(v)))
                             for a, v in sorted(vals.items()))
            print('!! %-40s %-44s %s' % (label, fn, cl))
            print('       %s' % desc)
    print('扫了 %d 个源文件，报出 %d 处「同一函数内不同轴的钳位上界不一致」'
          % (len(files), total))
    if not files:
        print('**一个文件都没扫到 —— 路径或后缀写错了**')
        return 1
    if total:
        print('**同一函数里 W/H/C 的合法下标上界应当一致；不一致多半是少减了 -1，')
        print('  会读到最后一个元素之后（附录 CL 就是这么越界的）**')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
