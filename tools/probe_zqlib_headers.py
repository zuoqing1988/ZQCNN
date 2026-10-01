#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""探测 3rdparty/include/ZQlib 下每个头能否**独立编译**（Linux / gcc）。

为什么需要这个工具
------------------
审计报告里反复出现同一类结论：「这是第三方头，改动面大且**无法编译验证**，不修」。
ZQ_MergeSort 那条就是这样被记成「不修」的 —— 直到有人真去数了一下它的 include，
发现只有 4 个标准头，补上 MSVC 的 __int64/__min/__max 就能单独编，
于是「无法验证」这个前提根本不成立（见 audit_k3_20261001.md 附录 W）。

这个工具把「能不能验证」从印象变成一张表：逐个头生成一个只 `#include` 它的
最小翻译单元，用 gcc 编译，把结果分类。

分类
----
  OK        编译通过 —— 可以单独验证，任何修改都能端到端测
  NEEDS_LIB 报缺外部库（jpeglib.h / OpenCV 之类）—— 装上依赖即可
  MSVC_ONLY 报 MSVC 专有关键字（__int64 / _fseeki64 / strcpy_s ...）
             —— 补几个 typedef/macro 就能编，属于可救
  BROKEN    其它编译错误 —— 真有问题，或者依赖链缺失

用法
----
    python tools/probe_zqlib_headers.py            # 全部头
    python tools/probe_zqlib_headers.py ZQ_Kmeans  # 只探名字里含这个串的
"""

from __future__ import print_function

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INC = os.path.join(ROOT, '3rdparty', 'include', 'ZQlib')
WSL_DIST = 'Ubuntu-20.04'

# 补在 #include 之前的兼容层：MSVC 的类型与内建函数，gcc 下没有。
# 注意 _fseeki64 只能给**一种**定义（函数式），给两个会 redefinition 报错。
SHIM = r'''
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
typedef long long __int64;
typedef unsigned long long __uint64;
#ifndef __min
#define __min(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef __max
#define __max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#define _fseeki64(f, o, w) fseeko64((f), (o), (w))
#define _ftelli64(f)       ftello64(f)
#define fopen_s(p, a, m)   (((*(p)) = fopen((a), (m))) == 0 ? 0 : -1)
#define strcpy_s(d, n, s)  strncpy((d), (s), (n))
#define sprintf_s          snprintf
#define _snprintf          snprintf
'''


def run_wsl(script):
    # 必须以**字节**喂给 wsl: text=True + input=str 在 Windows 上会按文本模式
    # 把 \n 翻译成 \r\n, WSL 里的 bash 于是看到 `set +\r` / `cd dir\r`,
    # 直接 `invalid option` + `syntax error: unexpected end of file`，
    # 一行脚本都没跑（2026-10-02 实测踩过）。
    p = subprocess.run('wsl -d %s -- bash -s' % WSL_DIST, shell=True,
                       input=script.encode('utf-8'), capture_output=True)
    return ((p.stdout or b'').decode('utf-8', 'replace')
            + (p.stderr or b'').decode('utf-8', 'replace'))


def classify(msg):
    low = msg.lower()
    for lib in ('jpeglib.h', 'jerror.h', 'png.h', 'zlib.h', 'opencv2/',
                'cuda_runtime.h', 'tbb/', 'omp.h', 'windows.h', 'tchar.h',
                'afx'):
        if lib in low:
            return 'NEEDS_LIB', lib
    for kw in ('__int64', '__uint64', '_fseeki64', '_ftelli64', 'strcpy_s',
               '_sopen', '__declspec', 'fopen_s', 'sprintf_s', '_snprintf',
               '_MSC_VER', '__forceinline', '_stricmp', '__try', 'min/max'):
        if kw in msg:
            return 'MSVC_ONLY', kw
    return 'BROKEN', ''


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    flt = sys.argv[1] if len(sys.argv) > 1 else ''
    headers = sorted(h for h in os.listdir(INC) if h.endswith('.h') and flt in h)
    if not headers:
        print('no header matches %r' % flt)
        return 1

    lines = ['set +e', 'cd /tmp && rm -rf zqprobe && mkdir zqprobe && cd zqprobe']
    for h in headers:
        stem = h[:-2]
        lines.append("cat > %s.cpp <<'PROBE_EOF'\n%s\n#include \"%s\"\n"
                     "int main(){return 0;}\nPROBE_EOF" % (stem, SHIM, h))
        # 成功打 OK, 失败把第一条 error: 打出来 (同一行, 便于解析)
        lines.append(
            "if g++ -fsyntax-only -std=c++11 -I/mnt/d/ZQCNN/3rdparty/include/ZQlib "
            "%s.cpp 2> %s.err; then echo 'R|%s|OK|'; else "
            "echo \"R|%s|ERR|$(grep -m1 error: %s.err | tr -d '\\r')\"; fi"
            % (stem, stem, h, stem, stem))
    lines.append('echo R|__END__|OK|')
    out = run_wsl('\n'.join(lines))

    rows = []
    for line in out.splitlines():
        if not line.startswith('R|'):
            continue
        parts = line.split('|', 3)
        if len(parts) != 4:
            continue
        rows.append((parts[1], parts[2], parts[3]))

    buckets = {'OK': [], 'NEEDS_LIB': [], 'MSVC_ONLY': [], 'BROKEN': []}
    for name, status, msg in rows:
        if status == 'OK':
            buckets['OK'].append((name, ''))
        else:
            c, d = classify(msg)
            buckets[c].append((name, d or msg.strip()[:70]))

    print('=' * 74)
    print('ZQlib header standalone-compile probe (gcc, -std=c++11, Linux)  total=%d'
          % len(headers))
    print('=' * 74)
    print('OK (independently verifiable): %d' % len(buckets['OK']))
    for name, _ in buckets['OK']:
        print('   %s' % name)
    for c in ('NEEDS_LIB', 'MSVC_ONLY', 'BROKEN'):
        items = buckets[c]
        print('\n%s: %d' % (c, len(items)))
        for name, d in items:
            print('   %-40s %s' % (name, d))
    return 0


if __name__ == '__main__':
    sys.exit(main())
