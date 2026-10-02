#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""列出 `ZQ_CNN_Tensor4D` 基类里**有实体的方法**，并标出哪些没有任何门禁碰过。

为什么要有这个工具（附录 DX.1）
------------------------------
`ZQ_CNN_Tensor4D` 的基类（`ZQ_CNN_Tensor4D.h` 第 14 ~ 886 行）里有一大批
**带循环、带 memcpy、带指针算术**的虚函数，**整个函数体就写在头文件里**。

这一整块在本次审计之前是**零覆盖**的：
仓库里 30 多道 `zq_*_check.cpp` 没有一道调用它们（它们都直接调 `zq_cnn_*` 内核，
见附录 CE 立下的"不过张量类"的规矩）。

附录 DD 就是这么撞上去的：`Tile` 只是其中**一个**方法，补上门禁之后
**当场查出两条真缺陷**（W 方向的复制上界 + `out_slice_ptr += sliceStep` 用错了张量）。
一个方法两条 ⇒ 这一整块有多值得查，不需要论证。

本工具只做一件事
--------------
把每个方法的**实体行数**与**它是否在任何门禁源码里出现过**列出来，
按"实体越长、越没人碰"排序。**它不下结论，只排优先级。**
实体为 0 的（= `{ return true; }` 这种）直接跳过 —— 那些没有可查的逻辑。

用法
----
    python tools/tensor4d_cov_probe.py              # 全表
    python tools/tensor4d_cov_probe.py --ungated    # 只看没人碰过的
    python tools/tensor4d_cov_probe.py --min-lines 20
"""

from __future__ import print_function

import argparse
import glob
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HDR = os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_Tensor4D.h')
TOOLS = os.path.join(ROOT, 'tools')


def base_class_span(lines):
    """基类是文件里第一个 `class ZQ_CNN_Tensor4D {`，到第一个 `class ... : public ZQ_CNN_Tensor4D` 之前。"""
    start = None
    for i, ln in enumerate(lines):
        if start is None and re.match(r'^\s*class ZQ_CNN_Tensor4D\s*\{?\s*$', ln):
            start = i
        elif start is not None and re.match(r'^\s*class \w+\s*:\s*public ZQ_CNN_Tensor4D', ln):
            return start, i
    return start, len(lines)


# 一行里出现 `Type Name(` 且不是纯声明（结尾是 `;` 或 `{`），视为**定义**的开头
DEF_RE = re.compile(r'^\s*(?:virtual\s+)?(?:static\s+)?[\w:<>\*&\s]+?\b(\w+)\s*\([^;]*\)\s*(const\s*)?\{?\s*$')

# 控制流关键字**不是方法名**。第一版没有这个选择，于是把 `if` / `while`
# 当成了方法名列在"有实体的方法"里。
# 工具自己输出一个不存在的名字，比不输出更坏 —— 这就是本会话一直在打的
# "工具自信地给错答案"。
KEYWORDS = set(['if', 'for', 'while', 'switch', 'return', 'else', 'do',
                'catch', 'sizeof', 'new', 'delete'])


def methods(lines, lo, hi):
    """返回 [(name, 定义起始行号, 实体行数)]。用花括号配平数实体行。"""
    out = []
    i = lo
    while i < hi:
        m = DEF_RE.match(lines[i])
        if not m or lines[i].rstrip().endswith(';'):
            i += 1
            continue
        name = m.group(1)
        if name in KEYWORDS:
            i += 1
            continue
        # 找到这一行之后第一个 '{'
        k = i
        depth = 0
        started = False
        body = 0
        while k < hi:
            depth += lines[k].count('{') - lines[k].count('}')
            if '{' in lines[k]:
                started = True
            if started and k > i:
                body += 1
            if started and depth <= 0:
                break
            k += 1
        if not started:
            i += 1
            continue
        out.append((name, i + 1, body))
        i = k + 1
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument('--ungated', action='store_true', help='只看没有任何门禁碰过的')
    ap.add_argument('--min-lines', type=int, default=1, help='实体少于这么多行的不列')
    args = ap.parse_args()

    if not os.path.isfile(HDR):
        print('找不到 %s' % HDR)
        return 1
    lines = io.open(HDR, encoding='utf-8', errors='replace').read().split('\n')
    lo, hi = base_class_span(lines)
    if lo is None:
        print('解析基类范围失败 —— 正则失效了，不要当成"没有方法"')
        return 1

    gates = {}
    for g in sorted(glob.glob(os.path.join(TOOLS, 'zq_*_check.cpp'))):
        gates[os.path.basename(g)] = io.open(g, encoding='utf-8', errors='replace').read()
    allsrc = '\n'.join(gates.values())

    ms = methods(lines, lo, hi)
    rows = []
    for name, ln, body in ms:
        if body < args.min_lines:
            continue
        hit = sorted(os.path.basename(g) for g, t in gates.items()
                     if re.search(r'\b' + re.escape(name) + r'\s*\(', t))
        rows.append((name, ln, body, hit))

    print('ZQ_CNN_Tensor4D 基类方法覆盖探针（附录 DX）')
    print('  头文件：ZQCNN/ZQ_CNN_Tensor4D.h，第 %d ~ %d 行' % (lo + 1, hi))
    print('  门禁：tools/zq_*_check.cpp，共 %d 个' % len(gates))
    print('  **本工具只排优先级，不下结论**；实体行数越多、越没人碰 ⇒ 越值得查\n')

    shown = [r for r in rows if (not args.ungated or not r[3])]
    shown.sort(key=lambda r: (-r[2], r[0]))
    print('  %-28s %-8s %-6s %s' % ('方法', '定义行', '实体行', '被哪些门禁碰过'))
    print('  ' + '-' * 96)
    for name, ln, body, hit in shown:
        print('  %-28s %-8d %-6d %s' % (name, ln, body, ', '.join(hit) or '**无**'))

    n_ung = len([r for r in rows if not r[3]])
    n_big_ung = len([r for r in rows if not r[3] and r[2] >= 5])
    tot_body = sum(r[2] for r in rows)
    ung_body = sum(r[2] for r in rows if not r[3])
    print('\n  有实体的方法 %d 个，其中 %d 个**没有任何门禁碰过**（占实体行数 %d/%d）'
          % (len(rows), n_ung, ung_body, tot_body))
    print('  其中实体 >= 5 行的「无门禁」方法：%d 个' % n_big_ung)
    tile_row = [r for r in rows if r[0] == 'Tile']
    tile_body = tile_row[0][2] if tile_row else 0
    print('  附录 DD 的教训：一个方法（Tile，%d 行实体）就查出两条真缺陷。' % tile_body)
    return 0


if __name__ == '__main__':
    sys.exit(main())
