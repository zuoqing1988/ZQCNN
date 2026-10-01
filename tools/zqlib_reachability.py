#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""算出「主工程真正用到 3rdparty/include/ZQlib 里的哪些头」（传递闭包）。

为什么需要
----------
2026-10-02 这一轮改了 `3rdparty/include/ZQlib/` 下 12 个头（合并
ZQ_MergeSort / ZQ_Kmeans / ZQ_QuickSort / ZQ_ImageProcessing / ZQ_KDTree /
ZQ_Quaternion / ZQ_RBFKernel / ZQ_Matrix / ZQ_ScanLinePolygonFill /
ZQ_CameraProjection / ZQ_SparseMatrix / ZQ_TaucsBase）。

直接 grep 只能证明「没有文件直接 include 它们」—— 但 include 是**传递**的，
某个 ZQlib 头可能经由另一条链把它们拖进主工程。所以必须算闭包。

这个工具给出三件事：
  1. 主工程（ZQCNN / ZQlibFaceID / SamplesZQ* / model）里所有
     `#include "..."` 的**传递闭包**里出现的 ZQlib 头集合；
  2. 其中哪些是「被直接 include 的」；
  3. 哪些 ZQlib 头**完全不可达**（改了不会影响任何产物）。

用法:
    python tools/zqlib_reachability.py            # 摘要
    python tools/zqlib_reachability.py --list     # 列出可达的头
"""

from __future__ import print_function

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ZQLIB = os.path.join(ROOT, '3rdparty', 'include', 'ZQlib')

# 主工程的源码根
SRC_ROOTS = ['ZQCNN', 'ZQlibFaceID', 'SamplesZQCNN', 'SamplesZQlibFaceID',
             'SamplesZQBLAS', 'SamplesZQGEMM', 'model']

# 头文件搜索路径（与 CMake 里的 ZQCNN_INCLUDE_DIRS 对齐）
INCLUDE_DIRS = [
    ROOT,
    os.path.join(ROOT, 'ZQCNN'),
    os.path.join(ROOT, 'ZQCNN', 'math'),
    os.path.join(ROOT, 'ZQ_GEMM'),
    os.path.join(ROOT, 'ZQ_GEMM', 'math'),
    os.path.join(ROOT, 'ZQlibFaceID'),
    os.path.join(ROOT, '3rdparty', 'include'),
    ZQLIB,
]

INC_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)

# 本轮（2026-10-02）改过的 ZQlib 头
EDITED = [
    'ZQ_MergeSort.h', 'ZQ_Kmeans.h', 'ZQ_QuickSort.h', 'ZQ_ImageProcessing.h',
    'ZQ_KDTree.h', 'ZQ_Quaternion.h', 'ZQ_RBFKernel.h', 'ZQ_Matrix.h',
    'ZQ_ScanLinePolygonFill.h', 'ZQ_CameraProjection.h', 'ZQ_SparseMatrix.h',
    'ZQ_TaucsBase.h', 'ZQ_CameraCalibration.h', 'ZQ_CameraPoseEstimation.h',
    'ZQ_PoissonSolver.h', 'ZQ_BlendTwoImages3D.h', 'ZQ_GridDeformation3D.h',
    'ZQ_ShapeDeformation.h', 'ZQ_StructureFromTexture.h',
    'ZQ_CompressedImageRaw.h', 'ZQ_MGMRESSolver.h', 'ZQ_StereoMatching.h',
    'ZQ_LazySnapping.h', 'ZQ_SplinePCHIP.h', 'ZQ_MinIndependentSets.h',
    'ZQ_LazySnappingGUI.h',
]


def resolve(name, cur_dir):
    """按 C 的 #include "..." 语义找文件：先看当前文件所在目录，再看 -I 列表。"""
    cands = [os.path.join(cur_dir, name)]
    cands += [os.path.join(d, name) for d in INCLUDE_DIRS]
    for c in cands:
        if os.path.isfile(c):
            return os.path.normpath(c)
    return None


def walk(entry):
    """从 entry 开始做 include 闭包遍历，返回 (可达文件集合, 直接 include 的 ZQlib 头集合)。"""
    seen = set()
    direct = set()
    stack = [entry]
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        try:
            txt = open(f, 'r', encoding='utf-8', errors='replace').read()
        except IOError:
            continue
        d = os.path.dirname(f)
        for m in INC_RE.finditer(txt):
            inc = m.group(1)
            r = resolve(inc, d)
            if r is None:
                continue
            if os.path.dirname(r) == os.path.normpath(ZQLIB):
                direct.add(os.path.basename(r))
            stack.append(r)
    return seen, direct


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    entries = []
    for sr in SRC_ROOTS:
        p = os.path.join(ROOT, sr)
        if not os.path.isdir(p):
            continue
        for dp, _dn, fns in os.walk(p):
            for fn in fns:
                if fn.endswith(('.h', '.cpp', '.c')):
                    entries.append(os.path.join(dp, fn))

    reachable = set()
    direct = set()
    for e in entries:
        _seen, d = walk(os.path.normpath(e))
        reachable |= d
        direct |= d

    allzqlib = set(f for f in os.listdir(ZQLIB) if f.endswith('.h'))
    unreachable = sorted(allzqlib - reachable)

    print('主工程源码文件: %d 个 (%s)' % (len(entries), ', '.join(SRC_ROOTS)))
    print('ZQlib 头总数  : %d' % len(allzqlib))
    print('**可达**     : %d' % len(reachable))
    print('**不可达**   : %d   <- 改了不会进任何产物' % len(unreachable))
    if '--list' in sys.argv:
        print('\n可达的 ZQlib 头:')
        for f in sorted(reachable):
            mark = '*' if f in EDITED else ' '
            print('  %s %s' % (mark, f))
    ed_reach = [f for f in EDITED if f in reachable]
    ed_unreach = [f for f in EDITED if f not in reachable]
    print('\n本轮改过的 %d 个 ZQlib 头:' % len(EDITED))
    print('  可达 %d 个: %s' % (len(ed_reach), ', '.join(sorted(ed_reach)) or '(无)'))
    print('  不可达 %d 个' % len(ed_unreach))
    return 0


if __name__ == '__main__':
    sys.exit(main())
