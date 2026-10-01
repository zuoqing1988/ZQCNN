#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""枚举主工程的全部 TU 并跑 MSVC /analyze（tools/msvc_analyze.bat 的驱动）。

为什么要有这一层（audit_k3_20261001.md 附录 AZ）
------------------------------------------------
批处理里写"扫描所有 TU"的逻辑（`goto` + label + `EnableDelayedExpansion`）
在 2026-10-02 直接坏掉：cmd 开始把 `rem` 注释的**片段**当命令执行
（`'ses'` 不是内部或外部命令…）。所以枚举挪到 Python 侧，
批处理只负责"给我一串文件，我逐个编"—— 无聊但它能用。

TU 范围与 ZQCNN/CMakeLists.txt 的 file(GLOB) 一致：
    ZQCNN/*.cpp
    ZQCNN/math/*.c
    ZQCNN/layers_c/*.c
    ZQCNN/layers_nchwc/*.c

结果解析：cl 的输出是**本地代码页**（本机 GBK），必须按 gbk 读，
否则读出来全是乱码而统计结果会显示"0 条"—— 又一个假绿。

用法:
    python tools/run_msvc_analyze.py            # 全部（约 3~5 分钟）
    python tools/run_msvc_analyze.py --list     # 只列 TU
    python tools/run_msvc_analyze.py zq_cnn_lr  # 只跑名字里含这个串的
    python tools/run_msvc_analyze.py --json
"""

from __future__ import print_function

import collections
import glob
import io
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TU_GLOBS = [
    os.path.join(ROOT, 'ZQCNN', '*.cpp'),
    os.path.join(ROOT, 'ZQCNN', 'math', '*.c'),
    os.path.join(ROOT, 'ZQCNN', 'layers_c', '*.c'),
    os.path.join(ROOT, 'ZQCNN', 'layers_nchwc', '*.c'),
]
# cl 的行尾和消息都是**本地代码页**（本机 GBK）。用 utf-8 读会得到一堆 U+FFFD，
# 而正则要匹配的 `warning C6386` 恰好还是 ASCII —— 于是统计会显示"0 条"，
# 看起来像"全部干净"。2026-10-02 实测踩到。
CL_ENCODING = 'gbk'
WARN_RE = re.compile(r'\b(warning|error)\s+(C\d+)')
# Windows SDK 自己的告警（intrin.h 的 C28251 等）与本项目无关
SDK_NOISE = ('intrin.h', 'Program Files', 'include\\')


def collect_tus(flt):
    out = []
    for g in TU_GLOBS:
        for p in sorted(glob.glob(g)):
            if flt and flt not in os.path.basename(p):
                continue
            out.append(p)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    argv = [a for a in sys.argv[1:]]
    as_json = '--json' in argv
    list_only = '--list' in argv
    argv = [a for a in argv if not a.startswith('--')]

    tus = collect_tus(argv[0] if argv else '')
    if list_only:
        for p in tus:
            print(os.path.relpath(p, ROOT))
        return 0
    if not tus:
        print('no TU matches')
        return 1

    print('=' * 74)
    print('MSVC /analyze over %d translation units' % len(tus))
    print('=' * 74)
    sys.stdout.flush()
    cmd = ['cmd', '/c', os.path.join(HERE, 'msvc_analyze.bat')] + tus
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True)
    sys.stdout.write((p.stdout or b'').decode('utf-8', 'replace'))
    sys.stdout.flush()

    tmp = os.environ.get('TEMP', '.')
    by_flag = collections.Counter()
    rows = []
    build_fail = []
    for t in tus:
        stem = os.path.splitext(os.path.basename(t))[0]
        log = os.path.join(tmp, 'zan_%s.log' % stem)
        if not os.path.isfile(log):
            build_fail.append(stem)
            continue
        txt = io.open(log, encoding=CL_ENCODING, errors='replace').read()
        for line in txt.splitlines():
            m = WARN_RE.search(line)
            if not m:
                continue
            loc = re.split(r': (?:warning|error) ', line)[0].strip()
            if any(n in loc for n in SDK_NOISE):
                continue
            by_flag[m.group(2)] += 1
            rows.append({'tu': stem, 'flag': m.group(2),
                         'loc': os.path.basename(loc),
                         'text': line.strip()})

    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    print()
    print('按告警号统计（已排除 Windows SDK 自己的告警）:')
    if not by_flag:
        print('   0 条')
    for k, v in by_flag.most_common():
        print('   %-8s %d' % (k, v))
    if rows:
        print('\n明细（按 TU 分组，同一位置只列一次）:')
        seen = set()
        cur = None
        for r in rows:
            key = (r['tu'], r['flag'], r['loc'])
            if key in seen:
                continue
            seen.add(key)
            if r['tu'] != cur:
                cur = r['tu']
                print('  %s' % cur)
            print('     %-8s %-40s %s' % (r['flag'], r['loc'], r['text'][:70]))
    if build_fail:
        print('\n没有日志文件（编译可能失败）: %s' % ', '.join(build_fail))
    return 0


if __name__ == '__main__':
    sys.exit(main())
