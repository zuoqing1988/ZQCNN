#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交替跑两个 SampleGEMMCompare 二进制, 逐尺寸取中位数对比。

AGENTS.md 第 8 条: 先把 A 跑完再跑 B 会被睿频/温度漂移系统性偏袒后跑的那个
(空对照里 64 个形状有 20 个出现 >3% 的假差异)。本脚本 A/B/A/B 交替。

用法: python tools/bench_two_binaries.py <binA> <binB> [-n 7] [--threshold 8]
两个路径都按 WSL 下的路径理解。
"""
import argparse, os, re, statistics, subprocess, sys

ROW = re.compile(r'^(\d+)x(\d+)x(\d+)\s+([\d.]+)\s+([\d.]+)')


def run(binary, rounds, env_home):
    """返回 {shape: [asm 值 ...]}"""
    per_round = []
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = '/mnt/d/ZQCNN/3rdparty/mkl_runtime/linux'
    for _ in range(rounds):
        p = subprocess.run(['wsl', '-d', 'Ubuntu-20.04', '--', 'bash', '-c',
                            'cd /mnt/d/ZQCNN && %s' % binary],
                           capture_output=True, text=True, encoding='utf-8',
                           errors='replace', env=env)
        d = {}
        for line in p.stdout.splitlines():
            m = ROW.match(line)
            if m:
                d['%sx%sx%s' % m.group(1, 2, 3)] = float(m.group(5))
        if d:
            per_round.append(d)
        else:
            sys.stderr.write('[warn] no rows parsed from %s (rc=%d)\n%s\n'
                             % (binary, p.returncode, p.stderr[:400]))
    return per_round


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bin_a')
    ap.add_argument('bin_b')
    ap.add_argument('-n', '--rounds', type=int, default=7)
    ap.add_argument('--threshold', type=float, default=8.0,
                    help='百分比, 超出才算真差异')
    args = ap.parse_args()

    a_runs, b_runs = [], []
    for i in range(args.rounds):
        a_runs.extend(run(args.bin_a, 1, None))
        b_runs.extend(run(args.bin_b, 1, None))
        print('round %d/%d done' % (i + 1, args.rounds))

    keys = sorted(set(a_runs[0]) & set(b_runs[0]))
    rows = []
    for k in keys:
        a = statistics.median([r[k] for r in a_runs])
        b = statistics.median([r[k] for r in b_runs])
        if a > 0:
            rows.append((b / a, k, a, b))
    rows.sort()
    print('\n%-18s %9s %9s %8s' % ('MxNxK', 'A', 'B', 'B/A'))
    for r, k, a, b in rows:
        flag = ''
        if r > 1 + args.threshold / 100: flag = '  B 快'
        elif r < 1 - args.threshold / 100: flag = '  A 快'
        print('%-18s %9.2f %9.2f %7.2fx%s' % (k, a, b, r, flag))
    rs = [r for r, _, _, _ in rows]
    up = sum(1 for r in rs if r > 1 + args.threshold / 100)
    dn = sum(1 for r in rs if r < 1 - args.threshold / 100)
    print('\n%d 形状  中位 %.2fx  B 更快 %d  A 更快 %d  噪声内 %d  (阈值 %.0f%%)'
          % (len(rs), statistics.median(rs), up, dn, len(rs) - up - dn,
             args.threshold))


if __name__ == '__main__':
    main()
