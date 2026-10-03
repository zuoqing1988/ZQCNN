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
    return _collect_nm(out, defined=True)


def undefined_symbols(obj_path):
    out = wsl("nm -g --undefined-only '%s' 2>/dev/null" % obj_path)
    return _collect_nm(out, defined=False)


def _collect_nm(out, defined):
    syms = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        # **两种输出格式**（第一版只认一种，于是 46 个 .o 解析出 0 个符号）：
        #   定义： 00000000000060b0 T zq_gemm_...      -> 3 段，取 [-2] [-1]
        #   未定义：         U zq_gemm_...             -> **2 段**（没有地址列）
        if len(parts) >= 3:
            typ, name = parts[-2], parts[-1]
        else:
            typ, name = parts[0], parts[1]
        is_undef = typ.upper() in ('U', 'W')
        if is_undef == defined:             # 未定义 / 弱未定义，不是"定义"
            continue
        if defined and typ.upper() not in ('T', 'D', 'B'):
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


# 什么算"库内"：真正会被打进 ZQCNN / ZQ_GEMM 的那几棵树。
# 一切 Samples*、tools/、根目录的 .cpp 都不算 ——
# 它们是消费者，不是生产者。
LIB_PREFIXES = ('ZQCNN/', 'ZQ_GEMM/', '3rdparty/', 'ZQlibFaceID/')


def is_library_src(rel):
    """rel 是**相对仓库根**、带正斜杠的源文件路径。"""
    r = rel.replace('\\', '/')
    return any(r.startswith(p) for p in LIB_PREFIXES)


def is_library_obj(obj_path):
    """obj_path 是构建目录里的**目标文件**路径。

    **不能用 OBJ_RE 的 rel 组来判库内/库外** —— 那个组给出的是
    `.dir` 目录下的**部分**路径（`SampleGEMMCompare.cpp`），
    不带 `ZQCNN/` / `SamplesZQBLAS/` 这一层，于是
    `is_library_src` 全判 False、`lib_objs` 收成 0 个，
    "库有没有用"这一问就悄悄退化成纯文本搜索 ——
    而纯文本搜索在宏引用上是错的（见 main() 里那段注释）。

    目标文件路径本身是带那一层的（`/tmp/zqb2/ZQCNN/CMakeFiles/...`），
    所以按**路径分段**判，不靠正则。
    """
    r = obj_path.replace('\\', '/')
    parts = r.split('/')
    return any(p in LIB_DIRS for p in parts)


# 构建目录里代表库的那几层（与 LIB_PREFIXES 对应，只是形态不同）
LIB_DIRS = {'ZQCNN', 'ZQ_GEMM', 'ZQlibFaceID', '3rdparty'}


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

    # **库引用哪个符号，以链接器的视角为准**（附录 GO.2）。
    #
    # 第一版用"在库内源文件里 grep 那个符号名"来判断，**立刻出了假阳性**：
    # 它把 `ZQ_GEMM/math/zq_gemm_32f_auto.c` 报成"库内零引用"，而
    # GO.2 刚在 `libZQCNN.a` 上量到 `U zq_gemm_32f_AnoTrans_Btrans_auto` ——
    # 库**确实**在用它。原因是库里的引用是**经宏**进来的：
    # `ZQCNN/layers_c/zq_cnn_convolution_gemm_32f_align_c.c` 里写的是
    #     zq_cblas_sgemm(x1,...) -> zq_gemm_32f_AnoTrans_Btrans_auto(...)
    # 那个名字在**源码文本里一次都不出现**，只在预处理之后出现。
    #
    # 所以"库有没有用"必须问编译器/链接器，不能问文本 ——
    # 这正是本文件 KNOWN_GAPS 里"通过宏转发都可能绕过"那条，
    # 只是之前它只被当成"可能漏报"，而这一次它造成了**误报**。
    lib_undef = set()
    lib_objs = []
    for obj in objs:
        if not is_library_obj(obj):
            continue
        lib_objs.append(obj)
        lib_undef |= undefined_symbols(obj)
    print('库内 .o = %d 个，其中出现过的未定义符号 = %d 个'
          % (len(lib_objs), len(lib_undef)))
    if not lib_objs:
        print('!! 一个库内 .o 都没认出来 —— 「库有没有用」这一问会退化成纯文本搜索，')
        print('   而纯文本搜索在**宏引用**上是错的（见上面那段注释）。停下。')
        return 2

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
        # **库内 / 库外**要分开（2026-10-03 补，附录 GO.1）。
        #
        # 原来的判据是"**有没有人**引用这个 TU 的外部符号"。
        # 对汇编内核那个 TU，它答"有人用" —— 因为
        # SamplesZQBLAS/SampleGEMMCompare 和 SamplesZQGEMM/SampleGEMMAsmCompare
        # 确实在调 `zq_gemm_32f_AnoTrans_Btrans_auto_asm`。
        # 于是**"库自己一次都没调过"这个事实被完全吞掉了**。
        #
        # "有没有人用"和"**库**有没有用"是两个问题，而后者才是
        # "这份代码到底在不在生产路径上"的答案。
        inside = [s for s in others if is_library_src(s)]
        outside = [s for s in others if not is_library_src(s)]
        syms = defined_symbols(obj)
        unreferenced = []
        lib_unreferenced = []
        sample_only = []
        for sym in sorted(syms):
            pat = re.compile(r'\b%s\b' % re.escape(sym))
            hit_any = any(pat.search(src_cache[s]) for s in others)
            # 库用没用它：链接器看到了就算（宏展开后的引用也在内），
            # 或者库内源码文本里出现了（函数指针表那种 nm 看不到的情况）。
            hit_lib = (sym in lib_undef
                       or any(pat.search(src_cache[s]) for s in inside))
            if not hit_any:
                unreferenced.append(sym)
            if not hit_lib:
                lib_unreferenced.append(sym)
                if hit_any:
                    sample_only.append(sym)
        results.append({'tu': rel_src, 'obj': obj,
                        # **TU 自己是不是库源，也要按目标文件路径判**：
                        # `tu`（OBJ_RE 的 rel 组）是 `.dir` 下的部分路径
                        # （`math/zq_gemm_32f_align_c_asm.c`），
                        # 拿它去 startswith('ZQ_GEMM/') 必然 False ——
                        # 加上这个过滤之后，汇编内核那一例被自己滤掉了，
                        # 类别从 2 条变成 0 条。**又一次"过滤条件用错了字段"。**
                        'is_lib': is_library_obj(obj),
                        'symbols': sorted(syms),
                        'unreferenced': unreferenced,
                        'lib_unreferenced': lib_unreferenced,
                        'sample_only': sample_only})

    dead = [r for r in results if r['unreferenced']]
    # **库内无人引用**：全部外部符号都只被 sample / 工具引用。
    # **只统计本身是库源的 TU**。第一版没加这个过滤，
    # 于是 `SampleMTCNNLoadFromCode` / `SampleMatMul` 这两个**sample 自己**
    # 混进了"库内无人引用"—— 它们按定义就不该在这个类别里，
    # 出现即噪声（真类别只有 2 条，混进来变 4 条）。
    lib_dead = [r for r in results
                if r['symbols']
                and r["is_lib"]
                and len(r['lib_unreferenced']) == len(r['symbols'])]

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
    # 库内无人引用的 TU（附录 GO.1）—— 与上面那节**分开报**，
    # 因为它们回答的是不同的问题：
    #   上面那节：这个 TU 是不是完全没人要（可以删）；
    #   这一节：这个 TU 有没有人要，但**要它的全是 sample / 工具**
    #          （它不在生产路径上 —— 删不得，但也说明它对库没有贡献）。
    print()
    print('=' * 74)
    print('库内无人引用的 TU（%d 个）：只被 Samples* / tools* 引用'
          % len(lib_dead))
    print('=' * 74)
    if not lib_dead:
        print('没有。')
    for r in lib_dead:
        print('\n%-52s  %d/%d 个外部符号库内零引用（其中 %d 个只被 sample/工具引用）'
              % (r['tu'], len(r['lib_unreferenced']), len(r['symbols']),
                 len(r['sample_only'])))
        if show_syms:
            for s in r['sample_only'][:12]:
                print('       %s' % s)
            if len(r['sample_only']) > 12:
                print('       ...（另 %d 个）' % (len(r['sample_only']) - 12))
    if lib_dead:
        print('\n这一节**不是**说这些 TU 该删 —— 它们被 sample 用着，删了 sample 就编不过。')
        print('它说的是：**这些代码不在 ZQCNN 的推理路径上**。'
              '（汇编内核那一例见 audit_k3_20261001.md 附录 GO。）')

    print('\n已知的局限（这些是筛子不是判官，命中之后必须人眼看）：')
    for g in KNOWN_GAPS:
        print('  - %s' % g)
    return 0


if __name__ == '__main__':
    sys.exit(main())
