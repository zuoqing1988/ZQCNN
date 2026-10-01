#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扫出「malloc 分配、delete[] 释放」这类配对错误的释放点。

为什么要有这个工具（audit_k3_20261001.md 附录 AT.8）
---------------------------------------------------
ZQ_MathBase.h:1166 是这么写的：

    double* val = (double*)malloc(sizeof(double)*row*col);
    ...
    delete[]val;                 // <-- malloc 配 delete[]

这是未定义行为。ASan 报 `alloc-dealloc-mismatch` 并 abort；不开 ASan 时
glibc 常常**一声不吭**地继续跑（同一块内存被还给不同的分配器），崩溃点出现在
完全无关的地方。这类缺陷在人工 review 里极难看见，因为它长得完全像正常代码。

本工具做的是纯文本层面的配对检查：对每个 `delete`/`delete[]`，
看它操作的变量在同一个文件里是不是由 malloc/calloc/realloc/fopen 之类
**C 风格分配**出来的，反之亦然。

**它不是编译器，也不是完整的静态分析。** 已知的局限写在下面的 KNOWN_GAPS 里，
因为「工具说自己查过了」比「工具没查」更危险。

用法:
    python tools/check_alloc_delete.py                # 扫全仓
    python tools/check_alloc_delete.py 3rdparty       # 只扫某个子目录
    python tools/check_alloc_delete.py --json         # 机器可读
"""

from __future__ import print_function

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))

# C 风格分配：结果赋给变量的那一步
C_ALLOC_RE = re.compile(
    r'\b([A-Za-z_]\w*)\s*=\s*[^;]*?\b(?:malloc|calloc|realloc|_alloca)\s*\(')
#
# 末尾的 (?!\[) 不能省：`delete[]values[i]` 释放的是 **values[i]**（那个元素，
# 由 new[] 分配，配对是**对的**），不是 values（那个 malloc 出来的外层数组）。
# 少了这个断言，ZQ_TaucsBase.h 的两处正确写法会被报成 bug —— 而一个天天误报的
# 工具，用两次就没人看了（2026-10-02 实测踩到）。
CPP_FREE_RE = re.compile(r'\bdelete\s*(\[\s*\])?\s*([A-Za-z_]\w*)\s*(?!\[)')
# C 风格释放
C_FREE_RE = re.compile(r'\bfree\s*\(\s*([A-Za-z_]\w*)\s*\)')
# new 表达式
NEW_RE = re.compile(r'\b([A-Za-z_]\w*)\s*=\s*new\b')

# 这些是"合理"的组合，写出来是为了让 KNOWN_GAPS 不是空谈
KNOWN_GAPS = [
    '一个指针可能先 new 后被 C 风格 API 接管（或反过来），本工具按变量名配对，'
    '分不清到底是哪一种 —— 所以它是**筛子**不是判官，命中之后必须人眼看一眼。',
    '指针在成员变量 / 全局 / 跨函数传递时，分配与释放可能不在同一个文件，'
    '本工具只看单文件。',
    '同一变量在一个文件里既 malloc 又 new（先分配再重新赋值）属于正常写法，'
    '本工具会误报。',
]

SKIP_DIRS = {'.git', 'build_x64', 'cmake-out-win32-x64', 'cmake-out-unix-x64',
             '3rdparty/opencv', '__pycache__'}

# 内建自测样本。一个「什么都查不出来」的检查工具比没有这个工具更危险 ——
# 它会让人以为这块已经审过了（附录 AT.8 的教训：真正的 ZQ_MathBase.h:1166 就是
# 被 ASan 撞出来的，而纯文本扫描在那之前从来没报过任何东西，没人知道它到底
# 能不能用）。所以每次改这个工具的匹配逻辑，都必须先跑通这一段。
#
# 期望：命中 f1 / f2 / f3 的 delete 那一行；**不**命中 f3 的 `delete[] s[0]`
# （它配的是 new double[2]，是对的），**不**命中 f4（new + delete[]，是对的）。
SELFTEST_SRC = r'''
void f1(){ char* p = (char*)malloc(10); delete[] p; }          // 应命中
void f2(){ int* q = new int[4]; free(q); }                      // 应命中
void f3(){ double* r = (double*)malloc(8); double** s = (double**)malloc(8);
          s[0] = new double[2]; delete[] s[0]; delete[] s; }    // 只应命中最后那个
void f4(){ char* d = new char[4]; delete[] d; }                 // 不应命中
'''
SELFTEST_EXPECT = {(1, 'p'), (2, 'q'), (4, 's')}


def selftest(tmp_path):
    if not os.path.isdir(tmp_path):
        os.makedirs(tmp_path)
    p = os.path.join(tmp_path, '_alloc_selftest.cpp')
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        f.write(SELFTEST_SRC.lstrip('\n'))
    got = set()
    try:
        for (ln, kind, var, _al, _t) in scan_file(p):
            got.add((ln, var))
    finally:
        os.remove(p)
    missing = SELFTEST_EXPECT - got
    extra = got - SELFTEST_EXPECT
    print('=' * 74)
    print('check_alloc_delete.py 自测')
    print('=' * 74)
    for ln, var in sorted(SELFTEST_EXPECT):
        print('   行 %d 变量 %-4s %s' % (ln, var, '命中' if (ln, var) in got else '**漏了**'))
    for ln, var in sorted(extra):
        print('   行 %d 变量 %-4s **误报**' % (ln, var))
    if missing or extra:
        print('\n自测失败：漏 %d 条，误报 %d 条 —— 这个工具现在**不可信**，先别用它的结论。'
              % (len(missing), len(extra)))
        return 1
    print('\n自测通过：3 条应命中全部命中，0 条误报。')
    return 0


def scan_file(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except IOError:
        return []

    c_alloc = {}     # var -> line no
    cplus_alloc = {} # var -> line no
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith('*') or line.lstrip().startswith('//'):
            continue    # 注释里出现不算
        # 两处都用 finditer：真实代码里 `double* r = malloc(...); double** s = malloc(...)`
        # 写在同一行很常见，只记第一个会让 s 完全不登记，释放处的错配就查不出来
        # （自测里就是靠这一条才暴露的，2026-10-02）。
        for m in C_ALLOC_RE.finditer(line):
            c_alloc.setdefault(m.group(1), i)
        for m in NEW_RE.finditer(line):
            cplus_alloc.setdefault(m.group(1), i)

    hits = []
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith('*') or line.lstrip().startswith('//'):
            continue
        # finditer 而不是 search：一行里可能有两次 delete，
        # 只看第一个会漏掉后面那个 —— 自测里 `s[0]=new double[2]; delete[] s[0];
        # delete[] s;` 就是这么漏的（2026-10-02 实测）。
        for m in CPP_FREE_RE.finditer(line):
            is_array = bool(m.group(1))
            var = m.group(2)
            src = c_alloc.get(var)
            if src is not None and var not in cplus_alloc:
                hits.append((i, 'malloc 后 delete', var, src, line.strip()))
            elif var in cplus_alloc and not is_array and var in c_alloc:
                # new T 之后用 delete（不带 []）—— 多态基类指针时这是对的，
                # 标成 MED 而非 HIGH
                hits.append((i, 'new 后 delete(无[]) + 同名 C 分配', var,
                             cplus_alloc[var], line.strip()))
        for m in C_FREE_RE.finditer(line):
            var = m.group(1)
            if var in cplus_alloc and var not in c_alloc:
                hits.append((i, 'new 后 free', var, cplus_alloc[var], line.strip()))
    return hits


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

    args = [a for a in sys.argv[1:]]
    as_json = '--json' in args
    want_selftest = '--selftest' in args
    args = [a for a in args if not a.startswith('--')]

    if want_selftest:
        return selftest(os.path.join(HERE, '__selftest_tmp__'))

    roots = [os.path.join(ROOT, a) for a in args] or [ROOT]
    findings = []
    scanned = 0
    for r in roots:
        if os.path.isfile(r):
            files = [r]
        else:
            files = []
            for dirpath, dirnames, filenames in os.walk(r):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                for fn in filenames:
                    if fn.endswith(('.h', '.hpp', '.cpp', '.c', '.cxx', '.inl')):
                        files.append(os.path.join(dirpath, fn))
        for p in files:
            scanned += 1
            for hit in scan_file(p):
                findings.append((os.path.relpath(p, ROOT),) + hit)

    if as_json:
        print(json.dumps([
            {'file': f, 'line': ln, 'kind': k, 'var': v, 'alloc_line': al,
             'text': t} for (f, ln, k, v, al, t) in findings], ensure_ascii=False,
                         indent=2))
        return 1 if findings else 0

    print('=' * 74)
    print('malloc/delete 配对扫描：%d 个源文件' % scanned)
    print('=' * 74)
    if not findings:
        print('没有命中。')
    else:
        for f, ln, kind, var, al, text in findings:
            print('%-46s:%-5d %-34s 变量 %s (分配在 :%d)'
                  % (f, ln, kind, var, al))
            print('    %s' % text[:96])
    print('\n已知的局限（这些是筛子不是判官，命中之后必须人眼看）：')
    for g in KNOWN_GAPS:
        print('  - %s' % g)
    return 1 if findings else 0


if __name__ == '__main__':
    sys.exit(main())
