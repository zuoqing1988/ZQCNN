#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查 SIMD 横向归约宏 `zq_final_sum_q` 的项数 == 向量 lane 数，且不越 `q[]` 数组。

为什么需要这个门禁
------------------
`ZQCNN/layers_c/zq_cnn_*_32f_align_c.c` 用 `#define` 把同一个 raw 头**反复实例化**，
每次换一套 `zq_mm_load_ps` / `zq_base_type` / `zq_mm_align_size`，横向归约统一写成

    #define zq_final_sum_q (q[0]+q[1]+...+q[N-1])

而 `q` 在各个 raw 头里一律声明成 `ZQ_DECLSPEC_ALIGN32 zq_base_type q[8];`。
**N 必须正好等于这一节的 lane 数**（= 同一节的 `zq_mm_align_size`）：
小了就是丢项，大了就是**越界读**。

2026-10-05 实测抓到的那条（附录 IX.1）就是大了：ARM NEON + `__ARM_NEON_FP16`
那一节的宏写成 9 项 `q[0]..q[8]`，而 `q` 只有 `q[8]` 个元素 —— 越界读，
且 x86 两套构建**永远编不到**那一节（`zq_base_type` 在 x86 上是 `float`，
SSE/AVX 两节分别是 4 项 / 8 项，都对），所以任何运行时门禁都盖不住。
唯一能盖住它的就是这种**读源码本身**的门禁。

判定
----
对每个 `#define zq_final_sum_q (...)`：
  1. 项数 == 该节在生效的 `zq_mm_align_size`（lane 数）
  2. 下标恰好是 0..项数-1（不缺项、不重复）
  3. 项数 <= 该节 `#include` 进去的 raw 头里 `zq_base_type q[N]` 的 N（不许越界）

用法
----
    python tools/check_mm_reduce_terms.py              # 扫默认文件
    python tools/check_mm_reduce_terms.py --selfcheck  # 先自测（门禁里常驻这一组）
    python tools/check_mm_reduce_terms.py <文件路径>    # 扫指定文件
"""
from __future__ import print_function

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS_C = os.path.join(ROOT, 'ZQCNN', 'layers_c')

# 一个 .c 里可能出现的横向归约宏（名字现在只有一个，但按前缀扫，
# 免得以后新增一层又从头踩一遍）
FINAL_SUM_RE = re.compile(r'^\s*#\s*define\s+(zq_\w*final_sum\w*)\s*\((.*)\)\s*$')
ALIGN_RE = re.compile(r'^\s*#\s*define\s+zq_mm_align_size\s+(\d+)\s*$')
QDECL_RE = re.compile(r'zq_base_type\s+q\[(\d+)\]')
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
TERM_RE = re.compile(r'\bq\[(\d+)\]')


def scan_text(text, raw_texts=None):
    """返回 (sections, bad)。

    sections: [(行号, 项数, lane, 下标列表, raw 里的 q 大小或 None)]
    bad:      [(行号, 说明)]
    raw_texts: {raw 头文件名: 文本}，用于查 q 的声明大小
    """
    raw_texts = raw_texts or {}
    sections, bad = [], []
    lines = text.split('\n')
    lane = None
    for i, line in enumerate(lines):
        m = ALIGN_RE.match(line)
        if m:
            lane = int(m.group(1))
        m = FINAL_SUM_RE.match(line)
        if not m:
            continue
        idx = [int(x) for x in TERM_RE.findall(m.group(2))]
        n = len(idx)
        # raw 头：本节之后第一条 include 的 *_raw.h
        qsize = None
        for j in range(i, min(i + 8, len(lines))):
            mi = INCLUDE_RE.match(lines[j])
            if not mi:
                continue
            nm = os.path.basename(mi.group(1))
            rt = raw_texts.get(nm)
            if rt is None:
                qpath = os.path.join(LAYERS_C, nm)
                if os.path.exists(qpath):
                    rt = io.open(qpath, encoding='utf-8').read()
                    raw_texts[nm] = rt
            if rt:
                qs = QDECL_RE.findall(rt)
                if qs:
                    qsize = min(int(x) for x in qs)
            break
        sections.append((i + 1, n, lane, idx, qsize))
        if lane is None:
            bad.append((i + 1, '这一节没有 zq_mm_align_size，无法判定 lane 数'))
            continue
        if n != lane:
            bad.append((i + 1, '项数 %d != lane 数 %d（zq_mm_align_size）' % (n, lane)))
        if sorted(idx) != list(range(n)):
            bad.append((i + 1, '下标不是 0..%d 的连续序列：%s' % (n - 1, idx)))
        if qsize is not None and n > qsize:
            bad.append((i + 1, '项数 %d 越过了 raw 头里的 q[%d]' % (n, qsize)))
    return sections, bad


GOOD = """\
#define zq_mm_align_size 8
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])
#include "zq_demo_raw.h"
"""

# 2026-10-05 那条真缺陷的形状：8 lane 却写了 9 项
OFF_BY_ONE = """\
#define zq_mm_align_size 8
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7]+q[8])
#include "zq_demo_raw.h"
"""

# 另一种形状：项数对，但下标重复（少算一项、重复一项）
DUP_INDEX = """\
#define zq_mm_align_size 8
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7]+q[7])
#include "zq_demo_raw.h"
"""

RAW = "ZQ_DECLSPEC_ALIGN32 zq_base_type q[8];\n"


def selftest():
    ok = True
    # 「多一项」会同时触发 3 条判定中的 2 条：项数 != lane，且下标不是 0..n-1 的连续序列；
    # 再叠上越过 q[] 就是 3 条。这里写的是**实测真值**，不是「我觉得应该是几」。
    cases = [
        ('合格：8 lane / 8 项', GOOD, 1, 0),
        ('**多一项**（IX.1 那条）', OFF_BY_ONE, 1, 2),
        ('下标重复', DUP_INDEX, 1, 3),
    ]
    for name, txt, want_sec, want_bad in cases:
        secs, bad = scan_text(txt, {'zq_demo_raw.h': RAW})
        got = (len(secs), len(bad))
        mark = 'OK ' if got == (want_sec, want_bad) else '**BAD**'
        if got != (want_sec, want_bad):
            ok = False
        print('  %s %-28s 扫到 %d 节 / %d 个问题（期望 %d / %d）'
              % (mark, name, got[0], got[1], want_sec, want_bad))
    # q[] 比 lane 数还小：项数与 lane 数一致，但仍然越界
    tiny_raw = "ZQ_DECLSPEC_ALIGN32 zq_base_type q[4];\n"
    txt = """\
#define zq_mm_align_size 8
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])
#include "zq_demo_raw.h"
"""
    secs, bad = scan_text(txt, {'zq_demo_raw.h': tiny_raw})
    mark = 'OK ' if len(bad) == 1 and 'q[4]' in bad[0][1] else '**BAD**'
    if len(bad) != 1:
        ok = False
    print('  %s %-28s 扫到 %d 个问题（期望 1）' % (mark, '越 q[4]', len(bad)))
    return ok


def main(argv):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

    if '--selfcheck' in argv:
        print('check_mm_reduce_terms 自测：')
        if not selftest():
            print('**自测没过 —— 匹配逻辑已坏，先修它再谈别的**')
            return 1
        print('自测通过')
        return 0

    files = [a for a in argv[1:] if not a.startswith('-')]
    if not files:
        files = [os.path.join(LAYERS_C, f) for f in sorted(os.listdir(LAYERS_C))
                 if f.endswith('.c')]

    raw_texts, total, bad_all = {}, 0, []
    for f in files:
        text = io.open(f, encoding='utf-8').read()
        secs, bad = scan_text(text, raw_texts)
        total += len(secs)
        bad_all += [(f, ln, msg) for ln, msg in bad]
        print('%-56s %3d 处 zq_final_sum_q，%d 处不合格'
              % (os.path.relpath(f, ROOT), len(secs), len(bad)))
        for ln, msg in bad:
            print('      !! 第 %d 行 %s' % (ln, msg))

    print('合计 %d 处 zq_final_sum_q，%d 处不合格' % (total, len(bad_all)))
    if not total:
        print('**一处都没扫到 —— 匹配逻辑多半坏了**（AGENTS.md 坑 #2）')
        return 1
    if bad_all:
        print('**归约项数必须正好等于 lane 数；多一项就是越界读（ARM FP16 上）**')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
