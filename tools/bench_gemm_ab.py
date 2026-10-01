#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZQ_GEMM 内核的 A/B 基准：编两个版本、**交替**各跑 N 次、逐尺寸取**中位数**对比。

为什么需要这个工具
------------------
2026-10-01 两次给 GEMM 换算法都栽在这里：

* 换完只跑一遍就下结论。同一形状连跑两次能从 45 GF/s 跳到 66 GF/s
  （睿频/boost 抖动），单次读数完全不可信。
* 手写 shell 管道做对比，中间踩过"把 `cp A B` 写成 `cp A A` 于是
  两个二进制其实是同一个"这种错，结论完全无效。
* 没有 profiler（WSL 里没有 `perf`），只能靠反复 A/B。

所以这里把三件事固定下来：编译、交替运行取最好值、只标记超过噪声阈值的差异。
阈值 3% 来自实测：同一二进制重复运行的最大抖动约 7%，但 90% 的形状在 2% 内。

用法
----
    python tools/bench_gemm_ab.py A.c B.c            # 两个源文件
    python tools/bench_gemm_ab.py A.c B.c -n 5       # 多跑几轮
    python tools/bench_gemm_ab.py --from-git HEAD     # A = 当前工作区, B = HEAD

A/B 两个源文件会被分别复制进 WSL 的临时构建目录，**不会动工作区**。
默认编译 ZQ_GEMM/math 下除被比较的那个文件以外的全部 .c，加上
SamplesZQBLAS/SampleGEMMCompare.cpp。
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEMM_DIR = os.path.join(ROOT, 'ZQ_GEMM', 'math')
BENCH_SRC = os.path.join(ROOT, 'SamplesZQBLAS', 'SampleGEMMCompare.cpp')
WSL_DIST = 'Ubuntu-20.04'
CFLAGS = '-O3 -mavx2 -mfma -fopenmp'


def sh(cmd, **kw):
    # 显式指定 utf-8: text=True 会用系统 locale(本机是 GBK),
    # 读 C++ 源码里的中文注释会直接 UnicodeDecodeError。
    kw.setdefault('encoding', 'utf-8')
    kw.setdefault('errors', 'replace')
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def wsl(script, check=True):
    """把脚本喂给 WSL 里的 bash -s。

    不要用 `wsl ... bash -lc "<script>"`：脚本里的 $f / $(...) / *.c 会被
    **外层** shell 先展开一遍。走 stdin 就绕开了这一层引号地狱。
    """
    p = subprocess.run(
        'wsl -d %s -- bash -s' % WSL_DIST,
        shell=True, input=script, capture_output=True, text=True,
        encoding='utf-8', errors='replace')
    if check and p.returncode != 0:
        sys.stderr.write(p.stdout + p.stderr)
        raise SystemExit('wsl failed')
    return p.stdout


def resolve_replace(path):
    """决定这个变体文件替换 GEMM_DIR 下的哪一个文件。

    变体文件必须以**被替换文件的原名**放进构建目录，否则 `for f in *.c`
    会把原文件和变体一起编进去，链接期一堆 multiple definition。
    所以：basename 命中就自己定，否则要求 --replace。
    """
    base = os.path.basename(path)
    if base in os.listdir(GEMM_DIR):
        return base
    return None


def to_wsl_path(p):
    """把 Windows 路径变成 WSL 看得见的 /mnt/<drive>/... 形式。"""
    p = os.path.abspath(p)
    drive = p[0].lower()
    rest = p[2:].replace('\\', '/')
    return '/mnt/%s%s' % (drive, rest)


def build(tag, variant_src, replace_name, root):
    """在 WSL 的 root 下建 gemm-<tag>/，装好源文件并编出 zbench-<tag>。"""
    d = '%s/gemm-%s' % (root, tag)
    out = '%s/zbench-%s' % (root, tag)
    gemm = to_wsl_path(GEMM_DIR)
    # 复制交给 shell 做：root 是 WSL 路径，Python 侧的 os/shutil 碰不到。
    # 关键是把变体装成 replace_name（否则 `for f in *.c` 会把原文件和变体
    # 一起编进去，链接期一堆 multiple definition）。
    script = (
        'set -e; rm -rf {d}; mkdir -p {d}; '
        'for f in {gemm}/*.c {gemm}/*.h; do '
        '  b=$(basename "$f"); [ "$b" = "{rep}" ] && continue; cp "$f" {d}/; done; '
        'cp {var} {d}/{rep}; cd {d}; '
        'for f in *.c; do gcc -c {cf} -I. -I/mnt/d/ZQCNN/ZQCNN "$f" -o "${{f%.c}}.o" 2>/dev/null; done; '
        'g++ {cf} -I. -I/mnt/d/ZQCNN/ZQCNN -o {out} '
        '/mnt/d/ZQCNN/SamplesZQBLAS/SampleGEMMCompare.cpp ./*.o -ldl -lm'
    ).format(d=d, gemm=gemm, rep=replace_name, var=to_wsl_path(variant_src), cf=CFLAGS, out=out)
    wsl(script)
    return out


def run_one(binary, out):
    """跑一轮，把这一轮的读数追加到 out 里。"""
    script = 'cd /mnt/d/ZQCNN && export MKL_THREADING_LAYER=SEQUENTIAL; %s' % binary
    txt = wsl(script, check=False)
    got = {}
    for ln in txt.splitlines():
        f = ln.split()
        if len(f) < 6 or 'x' not in f[0]:
            continue
        try:
            got[f[0]] = float(f[2])
        except ValueError:
            continue
    out.append(got)


def best_of(runs):
    """把多轮读数合并成 {shape: 逐轮中位数}。

    **不要用"取最大"**。GF/s 读数噪声大（本机约 7%），取最大会系统性地
    挑中"那一轮恰好顺风"的样本：同一个文件跟自己比，取最大会报出
    101% 这种数字，看起来像"全面超越"。中位数是无偏的。
    2026-10-01 就是靠这个发现"中位数 95%"其实是 86%。
    """
    import statistics
    per = {}
    for r in runs:
        for k, v in r.items():
            per.setdefault(k, []).append(v)
    out = {}
    for k, vs in per.items():
        if len(vs) >= 2:
            out[k] = statistics.median(vs)
        else:
            out[k] = vs[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('-n', '--runs', type=int, default=5,
                    help='每个版本交替跑几轮，逐形状取**中位数**（默认 5，至少要 3）')
    ap.add_argument('--replace', help='变体文件替换 GEMM_DIR 下的哪个文件'
                                        '（变体文件名与原名相同时可省略）')
    ap.add_argument('--from-git', action='store_true',
                    help='把 B 当作 git ref 取出（配合 A=当前工作区）')
    ap.add_argument('--threshold', type=float, default=8.0,
                    help='超过这个百分比才算有意义的差异（默认 8；空对照测出的噪声下限）')
    args = ap.parse_args()

    if not os.path.exists(BENCH_SRC):
        raise SystemExit('找不到 %s' % BENCH_SRC)

    replace = args.replace or resolve_replace(args.a)
    if replace is None:
        raise SystemExit('变体文件 %s 的名字在 ZQ_GEMM/math 下找不到，'
                         '请用 --replace 指定它替换哪个文件' % args.a)
    if replace not in os.listdir(GEMM_DIR):
        raise SystemExit('ZQ_GEMM/math 下没有 %s' % replace)

    a_src = args.a
    b_src = args.b
    tmp_ref = None
    if args.from_git:
        # 变体文件在仓库里的相对路径（用于 git show），不是构建目录里的名字
        rel = os.path.join('ZQ_GEMM', 'math', replace).replace('\\', '/')
        p = sh('git show %s:%s' % (args.b, rel))
        if p.returncode != 0:
            raise SystemExit(p.stderr)
        tmp_ref = tempfile.NamedTemporaryFile(suffix='.c', delete=False)
        tmp_ref.write(p.stdout.encode('utf-8'))
        tmp_ref.close()
        b_src = tmp_ref.name

    # 构建目录直接放在 WSL 的 /tmp 下：不需要 wslpath 翻译 Windows 临时目录，
    # 也不会因为翻译失败而把产物落在仓库里。每次用唯一子目录，保证干净。
    stamp = str(abs(hash((os.path.abspath(a_src), os.path.abspath(b_src), replace))))[:10]
    root = '/tmp/zqgemmab-%s' % stamp
    wsl('rm -rf %s; mkdir -p %s' % (root, root))
    try:
        bin_a = build('a', a_src, replace, root)
        bin_b = build('b', b_src, replace, root)
        # **必须交替跑**。先把 A 跑完再跑 B 的话, 两个版本之间的睿频/温度
        # 漂移会系统性地偏袒后跑的那个 —— 实测（同一文件自比, 各 5 轮）
        # 有 20/64 个形状出现 >3% 的"差异", 最大的到 10%, 且方向一致。
        ra_runs, rb_runs = [], []
        for i in range(args.runs):
            run_one(bin_a, ra_runs)
            run_one(bin_b, rb_runs)
        ra, rb = best_of(ra_runs), best_of(rb_runs)
    finally:
        wsl('rm -rf %s' % root, check=False)
        if tmp_ref:
            os.unlink(tmp_ref.name)

    keys = [k for k in ra if k in rb]
    if not keys:
        raise SystemExit('两个版本都没有产出可解析的读数')

    def kdim(k):
        return (int(k.split('x')[0]) * int(k.split('x')[1]), int(k.split('x')[2]))

    keys.sort(key=kdim)
    print('%-18s %11s %11s %9s' % ('MxNxK', 'A asm', 'B asm', 'delta'))
    worse = better = 0
    for k in keys:
        d = 100.0 * (ra[k] - rb[k]) / rb[k] if rb[k] else 0.0
        flag = ''
        if d > args.threshold:
            flag, better = '  <== A 更快', better + 1
        elif d < -args.threshold:
            flag, worse = '  <== A 更慢', worse + 1
        print('%-18s %11.2f %11.2f %+8.1f%%%s' % (k, ra[k], rb[k], d, flag))
    print('\n%d 个形状里：A 更快 %d，A 更慢 %d，其余在 ±%g%% 噪声内（共 %d）'
          % (len(keys), better, worse, args.threshold, len(keys) - better - worse))
    return 0


if __name__ == '__main__':
    sys.exit(main())
