#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 `warn_sweep_zqlib.py` 的那套分桶逻辑搬到**主工程**的翻译单元上。

为什么是同一个工具的另一个前端（audit_k3_20261001.md 附录 AU）
-----------------------------------------------------------
附录 AT 在 143 个**第三方**头上用 `-Wall -Wextra` 挖出 5 条真缺陷。
那些头是「没人看、没人编译、没人测」的重灾区；主工程 `ZQCNN/` 已经被人工精读过
十三轮，边际收益理应更低 —— 但**没人算过**，而「没人算过」正是本报告开头
那条元发现（「无法验证」是会自我实现的结论）的另一种写法。

所以：同一套 HIGH/MED/LOW 分桶、同一套基线门禁，扫主工程的翻译单元。
分桶表与理由都写在 warn_sweep_zqlib.py 顶部，这里不重复，只 import。

扫哪些 TU（与 ZQCNN/CMakeLists.txt 的 file(GLOB) 保持一致）
---------------------------------------------------------
    ZQCNN/*.cpp
    ZQCNN/math/*.c
    ZQCNN/layers_c/*.c
    ZQCNN/layers_nchwc/*.c

include 路径也取自根 CMakeLists.txt 的 ZQCNN_INCLUDE_DIRS：
    ZQ_GEMM / ZQCNN / 3rdparty/include

**注意口径**：`-Wall -Wextra` 下主工程的告警量会比第三方头大一个数量级
（内联汇编、把 const 指针转成非 const、SSE 内建函数返回 __m128 之类），
所以这里**不要**照抄第三方那套「HIGH 桶必须为空」的目标。
先跑一次 `--save-baseline` 把现状记下来，之后只拦**新增**。
基线里每一条都应该是「已判定：不是缺陷」并写明理由。

用法:
    python tools/warn_sweep_src.py                       # HIGH + MED
    python tools/warn_sweep_src.py --bucket HIGH         # 只看 HIGH
    python tools/warn_sweep_src.py --all                 # 连 LOW 一起
    python tools/warn_sweep_src.py ZQ_CNN_Tensor4D      # 只扫名字里含这个串的 TU
    python tools/warn_sweep_src.py --save-baseline  tools/zqcnn_warn_baseline.txt
    python tools/warn_sweep_src.py --check-baseline tools/zqcnn_warn_baseline.txt
"""

from __future__ import print_function

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WSL_DIST = 'Ubuntu-20.04'

# 与 ZQCNN/CMakeLists.txt 的 file(GLOB ...) 完全一致
TU_GLOBS = [
    os.path.join(ROOT, 'ZQCNN', '*.cpp'),
    os.path.join(ROOT, 'ZQCNN', 'math', '*.c'),
    os.path.join(ROOT, 'ZQCNN', 'layers_c', '*.c'),
    os.path.join(ROOT, 'ZQCNN', 'layers_nchwc', '*.c'),
]
INCLUDE_SUBDIRS = ['ZQ_GEMM', 'ZQCNN', os.path.join('3rdparty', 'include')]

sys.path.insert(0, HERE)
from warn_sweep_zqlib import (bucket_of, to_wsl_path, WARN_RE)  # noqa: E402


def run_wsl(script):
    p = subprocess.run('wsl -d %s -- bash -s' % WSL_DIST, shell=True,
                       input=script.encode('utf-8'), capture_output=True)
    return ((p.stdout or b'').decode('utf-8', 'replace')
            + (p.stderr or b'').decode('utf-8', 'replace'))


def collect_tus(flt):
    out = []
    for g in TU_GLOBS:
        d = os.path.dirname(g)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            base = os.path.splitext(name)[0]
            ext = os.path.splitext(name)[1]
            if ext not in ('.c', '.cpp'):
                continue
            if flt and flt not in name:
                continue
            out.append(os.path.join(d, name))
    return out


def build_script(tus):
    incs = ' '.join('-I%s' % to_wsl_path(os.path.join(ROOT, d))
                    for d in INCLUDE_SUBDIRS)
    # **必须与真实构建用同一套宏与 -m 开关**，否则会出现"一堆 TU 编不过"的假警报。
    # 2026-10-02 第一版两样都没给：
    #  ① 少 -DZQ_CNN_USE_ZQ_GEMM=1（根 CMakeLists.txt:15 默认 BLAS_TYPE=ZQ_GEMM），
    #     ZQ_CNN_CompileConfig.h 于是走进另一条分支，zq_lstm_32f_align_c 直接
    #     "invalid conversion"；
    #  ② 少 -mavx2 -mfma（CMakeLists.txt:113）。
    # 数值与 CMakeLists.txt:15/85/113 一致；ARM 那侧不扫（-mfma 不存在）。
    flags = ['-mavx2', '-mfma', '-fPIC', '-DZQ_CNN_USE_ZQ_GEMM=1']
    lines = ['set +e',
             'cd /tmp && rm -rf zqsrcwarn && mkdir zqsrcwarn && cd zqsrcwarn',
             'FLAGS="-Wall -Wextra -Wno-unused-parameter -Wno-write-strings '
             '-Wno-unused-function -Wno-unused-variable '
             '-Wno-unused-but-set-variable %s"' % ' '.join(flags)]
    for p in tus:
        tag = os.path.splitext(os.path.basename(p))[0]
        # **必须按扩展名选编译器**。CMake 里 ZQCNN/math/*.c、layers_c/*.c、
        # layers_nchwc/*.c 是 C 源文件，用 gcc 编；只有 ZQCNN/*.cpp 是 C++。
        # 2026-10-02 第一版对 .c 也用 g++，于是 zq_avx_mathfun.c 报了一堆
        #   error: narrowing conversion of '2147483648' from 'unsigned int' to 'int'
        # —— 那是 `_PS256_CONST_TYPE(sign_mask, int, 0x80000000)` 在 C++11 braced-init
        # 下的收窄检查，**在 C 里完全合法**。真按这条 error 去"修"生产代码，
        # 就会为了迎合一个错误的编译器模式去改一个没有问题的文件。
        if p.endswith('.cpp'):
            cc, std = 'g++', '-std=c++11'
        else:
            cc, std = 'gcc', '-std=gnu11'
        lines.append('%s -fsyntax-only %s $FLAGS %s -I. "%s" 2> %s.warn'
                     % (cc, std, incs, to_wsl_path(p), tag))
    lines.append('for f in *.warn; do echo "@@@FILE@@@ ${f%.warn}"; cat "$f"; done')
    lines.append('echo "@@@FILE@@@ __END__"')
    return '\n'.join(lines)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

    argv = list(sys.argv[1:])
    show_all = '--all' in argv
    argv = [a for a in argv if a != '--all']
    bucket_filter = None
    if '--bucket' in argv:
        i = argv.index('--bucket')
        bucket_filter = argv[i + 1].upper()
        del argv[i:i + 2]
    save_to = None
    if '--save-baseline' in argv:
        i = argv.index('--save-baseline')
        save_to = argv[i + 1]
        del argv[i:i + 2]
    check_against = None
    if '--check-baseline' in argv:
        i = argv.index('--check-baseline')
        check_against = argv[i + 1]
        del argv[i:i + 2]
    argv = [a for a in argv if not a.startswith('--')]

    flt = argv[0] if argv else ''
    tus = collect_tus(flt)
    if not tus:
        print('no translation unit matches %r' % flt)
        return 1

    out = run_wsl(build_script(tus))

    findings = []
    compile_errors = []
    total_warn = 0
    cur = None
    for line in out.splitlines():
        if line.startswith('@@@FILE@@@'):
            cur = line.split(' ', 1)[1].strip()
            continue
        if not line.strip():
            continue
        if ': error:' in line or line.startswith('error:'):
            compile_errors.append((cur or '?', line.strip()))
            continue
        m = WARN_RE.match(line.strip())
        if not m:
            continue
        total_warn += 1
        findings.append((os.path.basename(m.group('file')),
                         m.group('flag') or '?',
                         '%s:%s:%s' % (os.path.basename(m.group('file')),
                                       m.group('line'), m.group('col')),
                         m.group('msg')))

    buckets = {'HIGH': [], 'MED': [], 'LOW': []}
    for hdr, flag, loc, msg in findings:
        buckets[bucket_of(flag)].append((hdr, flag, loc, msg))

    print('=' * 74)
    print('ZQCNN 主工程 gcc -Wall -Wextra warning sweep  total_TU=%d total_warnings=%d'
          % (len(tus), total_warn))
    print('HIGH=%d  MED=%d  LOW=%d' % (len(buckets['HIGH']), len(buckets['MED']),
                                       len(buckets['LOW'])))
    if compile_errors:
        first = {}
        for hdr, msg in compile_errors:
            first.setdefault(hdr, msg)
        print('\n!! %d 个 TU **编不过** —— 它们的告警没被扫到:' % len(first))
        for hdr in sorted(first):
            print('   %-38s %s' % (hdr, first[hdr][:80]))

    def show(which):
        items = buckets[which]
        if not items:
            print('\n%s: 0  ✓' % which)
            return
        print('\n%s: %d' % (which, len(items)))
        seen = set()
        for hdr, flag, loc, msg in items:
            key = (hdr, flag, loc)
            if key in seen:
                continue
            seen.add(key)
            print('   %-38s %-24s %-30s %s' % (hdr, flag, loc, msg[:64]))

    if bucket_filter:
        show(bucket_filter if bucket_filter in buckets else 'HIGH')
    else:
        show('HIGH')
        show('MED')
        if show_all:
            show('LOW')
        else:
            print('\nLOW: %d  （--all 才显示）' % len(buckets['LOW']))

    key_of = lambda t: (t[0], t[1], t[2])

    if save_to:
        high = sorted(set(key_of(t) for t in buckets['HIGH']))
        lines = ['# ZQCNN 主工程 gcc -Wall/-Wextra HIGH 桶基线'
                 '（tools/warn_sweep_src.py --save-baseline 生成）',
                 '# 格式: <文件名>\\t<-W标志>\\t<文件:行:列>',
                 '#',
                 '# 与 ZQlib 的那份**口径不同**：主工程告警量大一个数量级，',
                 '# 所以基线不是空的 —— 每一条都应当是「已判定：不是缺陷」并写明理由。',
                 '# 基线的作用是拦**新增**，不是要求桶为空。']
        for hdr, flag, loc in high:
            lines.append('%s\t%s\t%s' % (hdr, flag, loc))
        with open(save_to, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(lines) + '\n')
        print('\nbaseline written to %s (%d HIGH findings)' % (save_to, len(high)))

    if check_against:
        base = set()
        try:
            with open(check_against, encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) >= 3:
                        base.add(tuple(parts[:3]))
        except IOError as e:
            print('\nERROR: 读不到基线 %s: %s' % (check_against, e))
            return 1
        cur = set(key_of(t) for t in buckets['HIGH'])
        new = sorted(cur - base)
        fixed = sorted(base - cur)
        print('\n=== 与基线 %s 比对 ===' % check_against)
        print('HIGH: 基线 %d 条 -> 现在 %d 条' % (len(base), len(cur)))
        if new:
            print('\nNEW HIGH — 新出现的高信号警告:')
            for h, fl, lo in new:
                print('   %-38s %-24s %s' % (h, fl, lo))
        if fixed:
            print('\nFIXED — 比基线少了:')
            for h, fl, lo in fixed:
                print('   %-38s %-24s %s' % (h, fl, lo))
        if not new and not fixed:
            print('无新增、无消失。')
        return 1 if new else 0

    return 0


if __name__ == '__main__':
    sys.exit(main())
