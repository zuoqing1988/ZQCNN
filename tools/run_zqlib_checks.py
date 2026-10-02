#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跑 tools/ 下所有 zq_*_check.cpp（第三方头库的独立回归测试）。

为什么要这个入口
----------------
附录 W/X/Y/Z/AA 一路下来的做法是：给每个「之前说没法验证」的第三方头写一个
脱离主工程的最小测试，用 ASan + LeakSanitizer 跑。到 AA 为止已经有 4 个，
但每个都要手敲一条编译命令 —— 没人会记得在每次改动后都跑一遍，于是它们会
慢慢变成「写过一次就没再跑过」的死文件。

这个脚本把它们统一起来：
  * 自动发现 tools/zq_*_check.cpp
  * gcc -O1 -g -fsanitize=address -I3rdparty/include/ZQlib
  * 逐个跑，任何一个非 0 退出就整体失败

用法:
    python tools/run_zqlib_checks.py            # 全部
    python tools/run_zqlib_checks.py mergesort  # 只跑名字里含这个串的
    python tools/run_zqlib_checks.py --list     # 只列出来

注意: 这些测试验的是**第三方头库**（3rdparty/include/ZQlib），与 ZQCNN 的
主工程构建无关，所以放在 tools/ 而不是 CMake 里。主工程的验证仍然是
tools/run_sample_regression.sh + tools/check_line_endings.py +
tools/check_text_encoding.py。
"""

import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WSL_DIST = 'Ubuntu-20.04'
INC = '/mnt/d/ZQCNN/3rdparty/include/ZQlib'


def run_wsl(script):
    # 必须喂字节: text=True 会在 Windows 上把 \n 变成 \r\n,
    # bash 于是看到 `set +\r` / `cd dir\r` 直接不跑（见附录 X.5）。
    p = subprocess.run('wsl -d %s -- bash -s' % WSL_DIST, shell=True,
                       input=script.encode('utf-8'), capture_output=True)
    return ((p.stdout or b'').decode('utf-8', 'replace')
            + (p.stderr or b'').decode('utf-8', 'replace'))


# 少数测试不是"纯 ZQlib 头"，还要编主工程的内核 .c。
# 一律用 **gcc** 编 .c（用 g++ 会把 C 的 braced-init 判成 narrowing 直接报错，
# 见附录 AU.2 踩过的坑），链接时再加 -mavx2 -mfma。
EXTRA_SOURCES = {
    'zq_lrn': [
        'gcc -O1 -g -mavx2 -mfma -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/math/zq_sse_mathfun.c -o $WDIR/zq_lrn_sse.o',
        'gcc -O1 -g -mavx2 -mfma -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/math/zq_avx_mathfun.c -o $WDIR/zq_lrn_avx.o',
    ],
    # zq_eltwise：同 zq_bns，两个 math .c 只是因为它们定义了 log/exp 等 SIMD 辅助。
    'zq_eltwise': [
        'gcc -O1 -g -mavx2 -mfma -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/math/zq_sse_mathfun.c -o $WDIR/zq_eltwise_sse.o',
        'gcc -O1 -g -mavx2 -mfma -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/math/zq_avx_mathfun.c -o $WDIR/zq_eltwise_avx.o',
    ],
    # zq_pool：只 include 了内核的 .c，不含 math/zq_*_mathfun.c，所以 EXTRA_SOURCES
    # 是空的 —— 留个空表项是为了"以后要加时知道该加哪儿"。
    'zq_pool': [],
    # zq_innerproduct 要链上 ZQ_GEMM 的三个 TU，其中 zq_gemm_32f_align_c.c
    # 单独一个就要编 5 分钟以上。默认不跑（--with-slow 才跑），理由写在这里。
    'zq_innerproduct': [
        # 内核自己也要编 —— 只链 ZQ_GEMM 那三个会 undefined reference
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_innerproduct_gemm_32f_align_c.c -o $WDIR/zq_ipgemm.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_align_c.c -o $WDIR/zq_gemm_align.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_align_c_asm.c -o $WDIR/zq_gemm_asm.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_auto.c -o $WDIR/zq_gemm_auto.o',
    ],
    # zq_bns 不自动跑（见 SKIP），但点名时要能真的编出来。
    'zq_bns': [
        'gcc -O1 -g -mavx2 -mfma -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/math/zq_sse_mathfun.c -o $WDIR/zq_bns_sse.o',
        'gcc -O1 -g -mavx2 -mfma -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/math/zq_avx_mathfun.c -o $WDIR/zq_bns_avx.o',
    ],
}
EXTRA_LINK = {'zq_innerproduct': ' $WDIR/zq_ipgemm.o $WDIR/zq_gemm_align.o $WDIR/zq_gemm_asm.o $WDIR/zq_gemm_auto.o',
              'zq_lrn': ' $WDIR/zq_lrn_sse.o $WDIR/zq_lrn_avx.o',
              'zq_pool': '',
              'zq_bns': ' $WDIR/zq_bns_sse.o $WDIR/zq_bns_avx.o',
              'zq_eltwise': ' $WDIR/zq_eltwise_sse.o $WDIR/zq_eltwise_avx.o'}
EXTRA_INC = {'zq_innerproduct': ' -I$R/ZQCNN -I$R/ZQ_GEMM',
             'zq_lrn': ' -I$R/ZQCNN -I$R/ZQ_GEMM',
             'zq_pool': ' -I$R/ZQCNN -I$R/ZQ_GEMM',
             'zq_bns': ' -I$R/ZQCNN -I$R/ZQ_GEMM',
             'zq_eltwise': ' -I$R/ZQCNN -I$R/ZQ_GEMM'}
# 测内核的测试自己也 include 了那个 .c，所以**主 TU 也要带 -mavx2 -mfma**，
# 否则 _mm256_set1_ps 这些 always_inline 内建会报
# "target specific option mismatch"（2026-10-02 实测）。
EXTRA_CXXFLAGS = {'zq_innerproduct': ' -mavx2 -mfma -fopenmp',
                   'zq_lrn': ' -mavx2 -mfma',
                   'zq_pool': ' -mavx2 -mfma', 'zq_bns': ' -mavx2 -mfma',
                   'zq_eltwise': ' -mavx2 -mfma'}


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('filter', nargs='?', default='')
    ap.add_argument('--list', action='store_true')
    # 明确不自动跑、但保留在仓库里的测试。
    # 每一个都要写清理由 —— "跑不过所以不跑"和"它是已知的未修项所以不跑"
    # 是两件完全不同的事，混起来就成了"无法验证"那个自我实现的结论（附录 W）。
    # 2026-10-02 一度把 zq_bns 放在这里（理由是「主内核索引约定已知未决」）。
    # 核实之后那个理由不成立：NCHWC 的布局是 [n][c片][h][w][align]，
    # 三个内层步长（imStep / sliceStep / widthStep / align）**全对**。
    # 现在 99 个用例逐位精确（相对误差 0.00e+00），它回到正常回归里。
    SKIP = {}
    ap.add_argument('--with-slow', action='store_true',
                    help='连那些编译特别慢的测试一起跑（zq_innerproduct 要链 ZQ_GEMM 的'
                         '三个 TU，其中 zq_gemm_32f_align_c.c 单个 >5 分钟）')
    ap.add_argument('--no-asan', action='store_true',
                    help='不带 sanitizer 编译（想先确认能不能编过时用）')
    ap.add_argument('--ubsan', action='store_true',
                    help='用 -fsanitize=undefined 代替 address：抓有符号溢出 / 移位越界 / '
                         '空指针解引用 / 未对齐访问等 ASan 看不见的行为（见附录 AS.2）')
    args = ap.parse_args()

    if args.ubsan and args.no_asan:
        print('--ubsan 和 --no-asan 不能同时给')
        return 1
    san = '' if args.no_asan else ('-fsanitize=undefined' if args.ubsan
                                   else '-fsanitize=address')

    srcs = sorted(glob.glob(os.path.join(HERE, 'zq_*_check.cpp')))
    srcs = [s for s in srcs if args.filter in os.path.basename(s)]
    # 除非显式点名（args.filter 命中），否则跳过 SKIP 里的那几个，并**把理由打出来**。
    skipped = []
    kept = []
    SLOW = {'zq_innerproduct': '要链 ZQ_GEMM 的三个 TU，编译 >5 分钟；'
                               '用 --with-slow 才跑'}
    if not args.with_slow:
        skipped_slow = []
        kept2 = []
        for s in srcs:
            tag2 = os.path.splitext(os.path.basename(s))[0][:-6]
            if tag2 in SLOW:
                skipped_slow.append((tag2, SLOW[tag2]))
            else:
                kept2.append(s)
        for tag2, why in skipped_slow:
            print('跳过（慢）%s: %s' % (tag2, why))
        srcs = kept2

    for s in srcs:
        tag = os.path.splitext(os.path.basename(s))[0][:-6]
        if tag in SKIP and tag not in args.filter:
            skipped.append((tag, SKIP[tag]))
        else:
            kept.append(s)
    for tag, why in skipped:
        print('跳过 %s: %s\n' % (tag, why))
    srcs = kept
    if not srcs:
        print('no zq_*_check.cpp matches %r' % args.filter)
        return 1

    print('找到 %d 个测试:' % len(srcs))
    for s in srcs:
        print('   %s' % os.path.basename(s))
    if args.list:
        return 0

    lines = ['set +e',
             'R=/mnt/d/ZQCNN',
             'WDIR=/tmp/zqchecks',
             'cd $WDIR && rm -rf * && mkdir -p $WDIR']
    for s in srcs:
        fname = os.path.basename(s)              # zq_xxx_check.cpp
        stem = fname[:-4]                        # zq_xxx_check
        tag = stem[:-6] if stem.endswith('_check') else stem
        for extra in EXTRA_SOURCES.get(tag, []):
            lines.append(extra)
        lines.append(
            "if g++ -O1 -g %s%s -I%s%s /mnt/d/ZQCNN/tools/%s%s -o %s "
            "2> %s.build.log; then echo 'B|%s|OK|'; else "
            "echo \"B|%s|BUILD_FAIL|$(grep -m1 -i error: %s.build.log | tr -d '\\r')\"; fi"
            % ('' if args.no_asan else san,
               EXTRA_CXXFLAGS.get(tag, ''),
               INC, EXTRA_INC.get(tag, ''), fname, EXTRA_LINK.get(tag, ''), tag,
               tag, tag, tag, tag))
        # 两套 sanitizer 的失败口径不同，分开写：
        #   ASan  -> 断言自己打的 "FAIL" 行数 + 进程非 0（越界/释放后使用会直接 abort）
        #   UBSan -> "runtime error:" 行数。**不要指望 rc**：不加
        #            -fno-sanitize-recover=all 的话 UBSan 只打一行就继续跑，rc 恒为 0，
        #            那一栏永远是 0 等于没查（2026-10-02 实测）。
        if args.ubsan:
            lines.append(
                "if [ -x ./%s ]; then UBSAN_OPTIONS=print_stacktrace=1 ./%s > %s.out 2>&1; "
                "echo \"R|%s|$?|$(grep -c 'runtime error:' %s.out)|"
                "$(grep -cE 'FAIL' %s.out)\"; fi"
                % (tag, tag, tag, tag, tag, tag))
        else:
            lines.append(
                "if [ -x ./%s ]; then ASAN_OPTIONS=detect_leaks=1 ./%s > %s.out 2>&1; "
                "echo \"R|%s|$?|$(grep -cE 'FAIL' %s.out)|0\"; fi"
                % (tag, tag, tag, tag, tag))
    lines.append('echo R|__END__|0|0')
    out = run_wsl('\n'.join(lines))

    build_fail, results = [], []
    for line in out.splitlines():
        if line.startswith('B|'):
            _, name, st, msg = (line.split('|', 3) + [''])[:4]
            if st != 'OK':
                build_fail.append((name, msg.strip()))
        elif line.startswith('R|') and '__END__' not in line:
            parts = line.split('|')
            if len(parts) >= 5:
                results.append((parts[1], parts[2], parts[3], parts[4]))

    print()
    nfail = 0
    for name, rc, nsan, nassert in results:
        ok = (rc == '0' and nsan == '0' and nassert == '0')
        if not ok:
            nfail += 1
        why = []
        if rc != '0':
            why.append('rc=%s' % rc)
        if nsan != '0':
            why.append('%s 条 sanitizer 报错' % nsan)
        if nassert != '0':
            why.append('%s 条断言失败' % nassert)
        print('%-34s %s' % (name, 'PASS' if ok else 'FAIL (%s)' % ', '.join(why)))
    for name, msg in build_fail:
        nfail += 1
        print('%-34s BUILD FAIL: %s' % (name, msg))

    print('\n%d/%d 通过' % (len(results) + len(build_fail) - nfail,
                           len(results) + len(build_fail)))
    if nfail:
        # 把失败的输出打出来，否则只知道失败不知道失败在哪。
        # UBSan 的栈可能落在最后 30 行之外（前面一堆正常运行日志），所以给到 80 行。
        for name, rc, nsan, nassert in results:
            if rc != '0' or nsan != '0' or nassert != '0':
                detail = run_wsl("cat /tmp/zqchecks/%s.out 2>/dev/null | tail -80" % name)
                print('\n===== %s =====\n%s' % (name, detail))
    return 1 if nfail else 0


if __name__ == '__main__':
    sys.exit(main())
