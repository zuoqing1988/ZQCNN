#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跑 N 轮 SampleGEMMCompare, 逐形状取**中位数**, 汇总 asm/MKL 比例。

为什么不能只跑一轮: 本机 GEMM 读数的噪声下限约 7% (同一个二进制跟自己比测出来的,
见 AGENTS.md 第 8 条)。单次读数会把 asm/MKL 的中位数抬高或压低好几个百分点,
拿它写报告等于把噪声当结论。

用法:
    python tools/gemm_mkl_ratio.py                       # Linux, 5 轮
    python tools/gemm_mkl_ratio.py -n 9
    python tools/gemm_mkl_ratio.py --exe <exe 路径>      # Windows 也行(直接跑 exe)
"""
import argparse, os, re, statistics, subprocess, sys

ROW = re.compile(r'^(\d+)x(\d+)x(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+([\d.]+)\s+([\d.eE+-]+)\s*$')


def run(cmd, cwd, env):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', env=env)
    out = {}
    for line in p.stdout.splitlines():
        m = ROW.match(line)
        if m:
            out['%sx%sx%s' % m.group(1, 2, 3)] = {
                'intr': float(m.group(4)), 'asm': float(m.group(5)),
                'mkl': float(m.group(6)), 'ratio': float(m.group(8)[:-1]),
                'ai': float(m.group(9)), 'err': float(m.group(10)),
            }
    return out


def main():
    # 本机 Windows 控制台是 GBK: 不显式改成 utf-8 的话中文标签全是乱码,
    # 读输出比读代码还费劲。
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--rounds', type=int, default=5)
    ap.add_argument('--exe', default=None,
                    help='直接给出的可执行文件路径（Windows）。不给则走 WSL 里的 Linux 版')
    ap.add_argument('--top', type=int, default=15, help='列出最差的几个形状')
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.exe:
        cmd, cwd = [args.exe], repo
        env = dict(os.environ)
    else:
        cmd = ['wsl', '-d', 'Ubuntu-20.04', '--', 'bash', '-c',
               'cd /mnt/d/ZQCNN && export LD_LIBRARY_PATH='
               '/mnt/d/ZQCNN/3rdparty/mkl_runtime/linux && '
               './cmake-out-unix-x64/Release/SampleGEMMCompare']
        cwd = repo
        env = dict(os.environ)

    runs = []
    for i in range(args.rounds):
        d = run(cmd, cwd, env)
        if not d:
            sys.stderr.write('第 %d 轮没有解析出任何形状，先确认二进制能跑\n' % (i + 1))
            return 1
        runs.append(d)
        print('round %d/%d ok' % (i + 1, args.rounds))

    keys = sorted(set(runs[0]) & set(runs[-1]))
    rows = []
    for k in keys:
        med = {f: statistics.median([r[k][f] for r in runs]) for f in
               ('intr', 'asm', 'mkl', 'ratio', 'ai')}
        med['err'] = max(r[k]['err'] for r in runs)
        rows.append((k, med))

    ratios = [m['ratio'] for _, m in rows if 0 < m['ratio'] < 1000]
    ai = [m['ai'] for _, m in rows]
    print('\n形状数 %d   轮数 %d' % (len(rows), args.rounds))
    print('asm/MKL : 中位 %.0f%%   最低 %.0f%%   最高 %.0f%%   >=100%% 的有 %d 个'
          % (statistics.median(ratios), min(ratios), max(ratios),
             sum(1 for x in ratios if x >= 100)))
    print('asm/intr: 中位 %.2fx' % statistics.median(ai))
    print('err(asm): 最大 %.2e' % max(m['err'] for _, m in rows))

    rows.sort(key=lambda t: t[1]['ratio'])
    print('\n最差的 %d 个形状:' % args.top)
    print('%-18s %9s %9s %9s %9s %9s' %
          ('MxNxK', 'intr', 'asm', 'MKL', 'asm/MKL', 'asm/intr'))
    for k, m in rows[:args.top]:
        print('%-18s %9.2f %9.2f %9.2f %8.0f%% %9.2f'
              % (k, m['intr'], m['asm'], m['mkl'], m['ratio'], m['ai']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
