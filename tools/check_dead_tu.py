#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""找出「编进库了、但全仓零调用」的翻译单元 —— 符号从 **.o 文件**里取，不用正则猜。

为什么要有这个工具（audit_k3_20261001.md 附录 AZ）
------------------------------------------------
2026-10-02 查一个 `/analyze` 告警时顺手 grep 了一下，发现
`ZQCNN/layers_nchwc/zq_cnn_batchnormscale_nchwc.c` **全仓没有任何地方调用** ——
而它仍然被 `ZQCNN/CMakeLists.txt` 的 `file(GLOB .../layers_nchwc/*.c)` 编进库。
于是：每次构建都编它；它里面有两处真缺陷（未判 NULL 的 `_aligned_malloc`、
读模型参数时用了 `ceil_C` 上界），**正因为没人调所以从来没人发现**；
它的主内核索引约定还是从 NCHW 版复制过来没改完的。

这和附录 W 里 ZQlib 那些头是同一类（没人看、没人编译、没人测），
只不过这次在**主工程**里 —— 而主工程被人工精读过十三轮。

**第一版用正则从源码里抽符号，结果是废的**
（2026-10-02 实测：`ZQ_CNN_Tensor4D.cpp` 那个几千行的文件只被认出 5 个符号，
而且 5 个全是局部变量 `t`/`n0`/`r`…）。函数签名、限定名、宏别名、
花括号换行，全都打它一个措手不及。**改用 `nm`**：目标文件里的符号表是
编译器自己写的，没有猜测成分。

判据
----
对每个 .o：
  1. `nm -g --defined-only` 取出**外部可见**的符号（排除 U 开头的未定义项）；
  2. 在全仓源文件（**排除该 .o 所属的 TU 及其配套 .h**）里 grep 这些符号；
  3. 零命中就记一条。

局限（写在 KNOWN_GAPS 里 —— "这个工具说没人用"比不说更危险）
----------------------------------------------------------
* 只看**字符串引用**。函数指针表、通过宏转发、跨 .so/.dll 的调用都可能绕过。
* `static` 函数不出现（那是好事：它们本来就只在 TU 内用）。
* C++ 里 inline/模板函数可能不导出（也算好事：它们在头里，TU 单独看无意义）。
* 只覆盖 ZQCNN/ 与 ZQ_GEMM/。SamplesZQCNN 的 .cpp 是逐个 add_executable 的。

用法:
    python tools/check_dead_tu.py            # 自动找 WSL 构建目录里的 .o
    python tools/check_dead_tu.py --objs <目录>
    python tools/check_dead_tu.py --symbols  # 把零引用的符号名都打出来
    python tools/check_dead_tu.py --json
"""

from __future__ import print_function

import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WSL_DIST = 'Ubuntu-20.04'
DEFAULT_OBJ_DIR = '/tmp/zqb2'

KNOWN_GAPS = [
    '只看字符串引用：函数指针表、通过宏转发、跨 .so/.dll 的调用都可能绕过 —— '
    '报"零引用"时要人眼确认一眼再下结论。',
    '要求 Linux 构建目录里有 .o（默认 /tmp/zqb2）。没有 .o 时本工具直接退出，'
    '**不会**退回"用正则猜符号"那条路 —— 猜出来的东西是废的（2026-10-02 实测）。',
    '只覆盖 ZQCNN/ 与 ZQ_GEMM/。SamplesZQCNN / SamplesZQlibFaceID 的 .cpp '
    '是逐个 add_executable 列出来的，不在范围内。',
]

SKIP_DIRS = {'.git', 'build_x64', 'cmake-out-win32-x64', 'cmake-out-unix-x64',
             '3rdparty', '__pycache__', 'SamplesZQCNN', 'SamplesZQlibFaceID',
             'ZQlibFaceID', 'ZQCNN_to_MNN', 'mobilefacenet-mxnet2caffe-ZQ'}

# .o 路径 -> 源文件相对路径的映射靠它反推：
#   /tmp/zqb2/ZQCNN/CMakeFiles/ZQCNN.dir/layers_c/foo.c.o  ->  ZQCNN/layers_c/foo.c
OBJ_RE = re.compile(r'CMakeFiles/[^/]+\.dir/(?P<rel>.+?)\.o$')


def wsl(script):
    p = subprocess.run('wsl -d %s -- bash -s' % WSL_DIST, shell=True,
                       input=script.encode('utf-8'), capture_output=True)
    return ((p.stdout or b'').decode('utf-8', 'replace')
            + (p.stderr or b'').decode('utf-8', 'replace'))


def find_objs(obj_dir):
    out = wsl("find %s -name '*.o' 2>/dev/null" % obj_dir)
    return sorted(l for l in out.splitlines() if l.endswith('.o'))


def defined_symbols(obj_path):
    out = wsl("nm -g --defined-only '%s' 2>/dev/null" % obj_path)
    syms = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        typ, name = parts[-2], parts[-1]
        if typ.upper() in ('U', 'W'):        # 未定义 / 弱未定义，不是"定义"
            continue
        if typ.upper() != 'T' and typ.upper() != 'D' and typ.upper() != 'B':
            continue
        if name.startswith('_') or name in ('main',):
            continue
        syms.add(name.split('@')[0])          # 去掉 @GLIBC_2.x 之类后缀
    return syms


def collect_source_files():
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = os.path.relpath(dirpath, ROOT)
        if rel.split(os.sep)[0] in SKIP_DIRS:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not d.startswith('.')]
        for fn in filenames:
            if fn.endswith(('.h', '.hpp', '.c', '.cpp', '.cxx', '.inl')):
                out.append(os.path.join(dirpath, fn))
    return out


def read(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except IOError:
        return ''


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    argv = [a for a in sys.argv[1:]]
    as_json = '--json' in argv
    show_syms = '--symbols' in argv
    obj_dir = DEFAULT_OBJ_DIR
    if '--objs' in argv:
        obj_dir = argv[argv.index('--objs') + 1]
    argv = [a for a in argv if not a.startswith('--')]

    objs = find_objs(obj_dir)
    if not objs:
        print('在 %s 下没找到 .o —— 先在 WSL 里 build 一次。' % obj_dir)
        print('（本工具**不会**退回用正则猜符号：2026-10-02 实测那样是废的。）')
        return 1

    srcs = collect_source_files()
    src_cache = {s: read(s) for s in srcs}

    results = []
    for obj in objs:
        m = OBJ_RE.search(obj)
        rel_src = m.group('rel') if m else os.path.basename(obj)
        # 该 TU 自己的 .c/.h 不算"外部引用"
        own = os.path.normcase(os.path.join(ROOT, rel_src))
        stem = os.path.splitext(own)[0]
        others = [s for s in srcs
                  if os.path.normcase(s) != own
                  and not os.path.normcase(s).startswith(stem + '.')
                  and stem not in os.path.normcase(s) + '.h']
        syms = defined_symbols(obj)
        unreferenced = []
        for sym in sorted(syms):
            pat = re.compile(r'\b%s\b' % re.escape(sym))
            if not any(pat.search(src_cache[s]) for s in others):
                unreferenced.append(sym)
        results.append({'tu': rel_src, 'obj': obj,
                        'symbols': sorted(syms),
                        'unreferenced': unreferenced})

    dead = [r for r in results if r['unreferenced']]

    if as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    print('=' * 74)
    print('死 TU 扫描：%d 个 .o / %d 个源文件（对象目录 %s）'
          % (len(objs), len(srcs), obj_dir))
    print('=' * 74)
    if not dead:
        print('没有发现「外部符号全都没人引用」的 TU。')
    else:
        for r in dead:
            print('\n%-52s  %d/%d 个外部符号零引用'
                  % (r['tu'], len(r['unreferenced']), len(r['symbols'])))
            if show_syms:
                for s in r['unreferenced'][:20]:
                    print('       %s' % s)
                if len(r['unreferenced']) > 20:
                    print('       ...（另 %d 个）' % (len(r['unreferenced']) - 20))
    print('\n已知的局限（这些是筛子不是判官，命中之后必须人眼看）：')
    for g in KNOWN_GAPS:
        print('  - %s' % g)
    return 0


if __name__ == '__main__':
    sys.exit(main())
