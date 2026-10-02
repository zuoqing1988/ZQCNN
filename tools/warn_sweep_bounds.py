#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 gcc **-O2 真出代码**扫主工程，捞「优化器才证明得了」的那一类告警。

为什么要有这个工具（附录 CT.5）
------------------------------
`warn_sweep_src.py` 用的是 `-fsyntax-only`：

    gcc -fsyntax-only -Wall -Wextra ...     # 只做语义分析，**不出代码、不做优化**

而 `-Warray-bounds` 依赖的恰恰是**优化器的值域传播** ——
"这个下标经过前面那个范围检查，只可能是 0..4"这种推断只有优化器做得出。
没有优化，这条告警**永远不会响**。

这不是推测，是实测撞上的：附录 CT.4 里
`ZQ_CNN_Forward_SSEUtils::ReductionSum` 的守卫写成了 `axis > 4`
（`axis` 紧接着索引 `int out_dims[4]`），`axis==4` 越界写一个 int。
在 `-fsyntax-only` 的扫描里**从来没出现过**；换 `-O2 -c` 立刻报：

    ZQ_CNN_Forward_SSEUtils.h:2490: warning: array subscript 4 is above
        array bounds of 'int [4]' [-Warray-bounds]

所以本工具与 `warn_sweep_src.py` 的差别只有一处，但它是**决定性**的：
`-fsyntax-only` -> `-O2 -c`。**用的是同一套宏、同一套 include、同一批 TU**，
所以两边的结果可以直接对比；`-Wmaybe-uninitialized` / `-Wuninitialized`
这类告警也只有在真优化之后才可能出现，同样是这一轴的增量。

代价：`-O2 -c` 比 `-fsyntax-only` 慢一个数量级，个别 TU（`zq_gemm_32f_align_c.c`
之类几千行的手写内核）几分钟起步。因此本工具：

* 每个 TU 有独立的 `timeout`（默认 300 秒），超时的**逐个列出来**，
  绝不静默跳过 —— "跑不过所以不跑"和"它没有告警"是两件完全不同的事
  （附录 W 那条自我实现的结论）。
* 支持 `--budget` 给整轮一个墙钟预算，按**实测耗时从小到大**的顺序编译，
  超预算就停下并把没跑到的列出来。
* 默认按 `--save-baseline` 记基线，回归时用 `--check-baseline` 拦**新增**。

分桶
----
HIGH  优化器能证明是真错的（CT.4 就是靠 HIGH 桶里的一条抓到的）
MED   -Wall/-Wextra 的其余部分 + 可疑但不足以定罪的
LOW   结构性噪声

**为什么不复用 `warn_sweep_zqlib.py` 的分桶表**：那份表里
`-Warray-bounds` 被放在 MED。CT.4 证明它在本项目里抓到过真缺陷，
所以在这个工具里一律进 HIGH。但**不去改共享的那张表** ——
它服务的是第三方头的 `-fsyntax-only` 扫描，那条轴本来就不会产生
`-Warray-bounds`，改它只会让两套工具的基线互相干扰。要不要提升，
是另一次单独的、要说清楚理由的决定。

用法
----
    python tools/warn_sweep_bounds.py                       # 默认档
    python tools/warn_sweep_bounds.py --all                 # 连慢 TU 也跑
    python tools/warn_sweep_bounds.py --budget 1800         # 墙钟预算（秒）
    python tools/warn_sweep_bounds.py --timeout 600         # 单 TU 超时
    python tools/warn_sweep_bounds.py ZQ_CNN_Tensor4D       # 只扫名字里含这个串的
    python tools/warn_sweep_bounds.py --save-baseline tools/zqcnn_bounds_baseline.txt
    python tools/warn_sweep_bounds.py --check-baseline tools/zqcnn_bounds_baseline.txt
"""

from __future__ import print_function

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WSL_DIST = 'Ubuntu-20.04'

# TU 集合与 include 路径**必须**与 warn_sweep_src.py 完全一致，
# 否则两边的结果没法对比，"哪条是 -O2 带来的增量"这个问题就答不了。
sys.path.insert(0, HERE)
from warn_sweep_src import collect_tus, INCLUDE_SUBDIRS  # noqa: E402
from warn_sweep_zqlib import bucket_of, to_wsl_path, WARN_RE  # noqa: E402

# 只在**真优化**之后才可能出现、或本身就是"优化器证明得了"的告警。
# 全部按 gcc 9.4（Wsl Ubuntu-20.04 的版本）能认的写，但**不假定它们一定存在**：
# 脚本开头会先拿编译器逐个探一遍，把不认识的旗标**打出来并剔除**
# （见 build_script 里的 @@@FLAGDROP@@@ 探针）。第一版就栽在这上面：
# `-Wstringop-overread` 是 Clang 才有的，gcc 直接
# `error: unrecognized command line option`，43 个 TU 全编不过。
# 与其手工维护一份"哪个版本支持哪些旗标"的表，不如让编译器自己回答。
OPT_ONLY_FLAGS = [
    '-Warray-bounds=2', '-Wstringop-overflow=2',
    '-Wnull-dereference', '-Wformat-overflow=2', '-Wformat-truncation=2',
    '-Wcalloc-transposed-args', '-Wfree-nonheap-object',
    '-Wduplicated-cond', '-Wduplicated-branches', '-Wlogical-op',
    '-Wmaybe-uninitialized', '-Wuninitialized', '-Wstrict-overflow=2',
]
# 与 warn_sweep_src.py 完全相同的那几个 -Wno-，否则两边的 LOW 桶不可比
NOISE_FLAGS = [
    '-Wno-unused-parameter', '-Wno-write-strings', '-Wno-unused-function',
    '-Wno-unused-variable', '-Wno-unused-but-set-variable',
]

# HIGH：这一轴存在的全部理由
HIGH_FLAGS = {
    '-Warray-bounds': '优化器证明的下标越界 —— 附录 CT.4 就是靠它抓到 `axis > 4`',
    '-Wstringop-overflow': '字符串/内存操作越界（优化器证明的）',
    '-Wstringop-overread': '字符串/内存操作越界读（优化器证明的）',
    '-Wnull-dereference': '优化器证明的解引用必定非空',
    '-Wformat-overflow': '输出会超出目标缓冲的格式串',
    '-Wformat-truncation': '截断后结果与源串不同（拷贝类 API 误用）',
    '-Wcalloc-transposed-args': 'calloc 的实参顺序写反了（经典 off-by-N）',
    '-Wfree-nonheap-object': 'free 了一个不是 malloc 出来的指针',
    '-Wduplicated-cond': '同一条件在同一个 if/else 链里出现两次，其中一支永远进不去',
    '-Wduplicated-branches': '同一分支在同一个 if/else 链里出现两次',
    '-Wlogical-op': '逻辑运算符两侧有副作用/非常量（可能是本该 && 写成 &）',
    '-Wmaybe-uninitialized': '可能未初始化就使用（**只有真优化后才可能报**）',
    '-Wuninitialized': '确定未初始化就使用（**只有真优化后才可能报**）',
}

def bucket_bounds(flag):
    if not flag.startswith('-'):
        flag = '-' + flag
    if flag in HIGH_FLAGS:
        return 'HIGH'
    return bucket_of(flag)


def run_wsl(script, timeout=None):
    p = subprocess.run('wsl -d %s -- bash -s' % WSL_DIST, shell=True,
                       input=script.encode('utf-8'), capture_output=True,
                       timeout=timeout)
    return ((p.stdout or b'').decode('utf-8', 'replace')
            + (p.stderr or b'').decode('utf-8', 'replace'))


def build_script(tus, wd, per_tu_timeout, budget):
    incs = ' '.join('-I%s' % to_wsl_path(os.path.join(ROOT, d))
                    for d in INCLUDE_SUBDIRS)
    # budget==0 表示不限。**必须显式换成一个大数**：bash 里 `[ $EL -gt 0 ]`
    # 在跑过一秒之后恒真，会把所有 TU 都跳掉 ——
    # 这是"0 既像'无限'又像'立即停'、而 bash 只认后者"的老坑。
    if budget <= 0:
        budget = 10 ** 9
    probe = ' '.join(OPT_ONLY_FLAGS + NOISE_FLAGS)
    lines = ['set +e',
             'cd /tmp && rm -rf %s && mkdir %s && cd %s' % (wd, wd, wd),
             # 旗标探针：编译器不认识的 -Wxxx 会直接 `error: unrecognized ...`，
             # 那会让**每个 TU 都编不过**，于是 HIGH=0 MED=0 —— 一个
             # 「全部编不过所以没有告警」的假绿（第一版就撞上了这个）。
             # 探针把认识的旗标收进 $OKFLAGS，不认识的**打出来并剔除**，
             # 这样这条轴「少扫了什么」永远摆在输出里，不会静默。
             'OKFLAGS=""',
             'for f in %s; do' % probe,
             '  if gcc -x c -fsyntax-only $f - < /dev/null > /dev/null 2>&1; '
             'then OKFLAGS="$OKFLAGS $f"; else echo "@@@FLAGDROP@@@ $f"; fi',
             'done',
             'FLAGS="-O2 -Wall -Wextra -mavx2 -mfma -fPIC '
             '-DZQ_CNN_USE_ZQ_GEMM=1$OKFLAGS"',
             'INC="%s"' % incs,
             'echo "@@@LIST@@@"']
    for p in tus:
        lines.append('echo "TU %s"' % to_wsl_path(p))
    lines.append('BUDGET=%d' % budget)
    # 注意：这一行**不走 % 格式化**（没有 % (...)），所以这里必须写单个 %s。
    # 写成 %%s 的话 T0 会被赋成字面量 "%s"，紧接着 $((NOW-T0)) 就炸成
    #   bash: %s: syntax error: operand expected
    # 而错误发生在**后面每一行**的算术展开上，报错行号指向编译行、
    # 真正出问题的是这一行 —— 第一版就是这么查了半天。
    lines.append('T0=$(date +%s)')
    for p in tus:
        tag = os.path.splitext(os.path.basename(p))[0]
        if p.endswith('.cpp'):
            cc, std = 'g++', '-std=c++11'
        else:
            cc, std = 'gcc', '-std=gnu11'
        lines.append(
            'NOW=$(date +%%s); EL=$((NOW-T0)); '
            'if [ $EL -gt $BUDGET ]; then echo "@@@SKIP@@@ %s budget"; '
            'else timeout %d %s %s $FLAGS $INC -c "%s" -o %s.o 2> %s.warn; '
            'RC=$?; NOW=$(date +%%s); echo "@@@TU@@@ %s rc=$RC t=$((NOW-T0))"; fi'
            % (tag, per_tu_timeout, cc, std, to_wsl_path(p), tag, tag, tag))
        lines.append('echo "@@@ENDTU@@@"')
    # 已经在 $wd 里面，所以 glob 是 *.warn 而不是 $wd/*.warn；
    # 再加一道 `[ -e "$f" ] || continue` —— 一个 warn 都没生成时
    # shell 会把字面量 "*.warn" 当文件名传给 cat。
    lines.append('for f in *.warn; do [ -e "$f" ] || continue; '
                 'echo "@@@FILE@@@ ${f%.warn}"; cat "$f"; done')
    lines.append('echo "@@@FILE@@@ __END__"')
    return '\n'.join(lines)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

    argv = list(sys.argv[1:])
    run_all = '--all' in argv
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
    per_tu_timeout = 300
    if '--timeout' in argv:
        i = argv.index('--timeout')
        per_tu_timeout = int(argv[i + 1])
        del argv[i:i + 2]
    budget = 0            # 0 = 不限
    if '--budget' in argv:
        i = argv.index('--budget')
        budget = int(argv[i + 1])
        del argv[i:i + 2]
    argv = [a for a in argv if not a.startswith('--')]

    flt = argv[0] if argv else ''
    # **不设慢 TU 名单。** 起草时照着旧笔记写了三个"几分钟起步"的名字
    # （zq_gemm_32f_align_c / zq_cnn_convolution_gemm_32f_align_c /
    #  zq_cnn_convolution_gemm_nchwc_raw），实测 -O2 编译 43 个 TU 的真实耗时：
    #   最慢 15.81s (zq_cnn_convolution_gemm_nchwc)
    #   次慢 13.74s (zq_cnn_depthwise_convolution_32f_align_c)
    #   7 个 C++ TU 合计约 30s，36 个 C TU 合计约 95s —— 整轮 2 分钟内跑完。
    # 那三个名字里 zq_gemm_32f_align_c 根本不在本工具的扫描集合内
    # （它属于 ZQ_GEMM/math/，不在 ZQCNN 的四个 glob 里），另外两个只要 2 秒。
    # 换句话说，那份名单是**照着旧笔记抄的、没量过的**，属于本报告开头
    # 那条元发现的反面：不是"无法验证"，而是"没验证就写成了结论"。
    # 单 TU 超时机制仍然保留（--timeout，默认 300 秒），
    # 只是不再预置任何"跑不过所以不跑"的名字 —— 真超时了工具会逐个列出来。
    tus = collect_tus(flt)
    if not tus:
        print('no translation unit matches %r' % flt)
        return 1

    wd = 'zqboundswarn'
    print('扫 %d 个 TU，-O2 -Wall -Wextra + %d 条优化期告警，单 TU 超时 %ds，总预算 %s'
          % (len(tus), len(OPT_ONLY_FLAGS), per_tu_timeout,
             ('%ds' % budget) if budget else '不限'))
    sys.stdout.flush()

    t0 = time.time()
    out = run_wsl(build_script(tus, wd, per_tu_timeout, budget))
    wall = time.time() - t0

    # 先解析旗标探针（编译器不认识的旗标会让**每个 TU 都编不过**）
    dropped = []
    for line in out.splitlines():
        if line.startswith('@@@FLAGDROP@@@ '):
            dropped.append(line[len('@@@FLAGDROP@@@ '):].strip())
    if dropped:
        print('编译器不认识的旗标（已剔除，这条轴少扫了它们）: %s'
              % ', '.join(dropped))

    # 再解析 TU 级的 rc / 耗时
    tu_info = {}
    skipped = []
    cur_tu = None
    for line in out.splitlines():
        if line.startswith('@@@LIST@@@'):
            mode = 'list'; continue
        if line.startswith('TU '):
            mode = 'list'; cur_tu = os.path.basename(line[3:]); continue
        if line.startswith('@@@SKIP@@@ '):
            skipped.append(line[len('@@@SKIP@@@ '):]); mode = None; continue
        if line.startswith('@@@TU@@@ '):
            body = line[len('@@@TU@@@ '):]
            parts = body.split()
            tag = parts[0]
            rc = next((p.split('=', 1)[1] for p in parts if p.startswith('rc=')), '?')
            tt = next((p.split('=', 1)[1] for p in parts if p.startswith('t=')), '?')
            tu_info[tag] = (rc, tt)
            mode = None
            continue
        if line.startswith('@@@ENDTU@@@'):
            mode = None; continue
        if line.startswith('@@@FILE@@@'):
            mode = 'file'; continue

    findings = []
    compile_errors = []
    cur = None
    in_file = False
    for line in out.splitlines():
        if line.startswith('@@@FILE@@@'):
            in_file = True
            cur = line.split(' ', 1)[1].strip()
            continue
        if not in_file or not line.strip():
            continue
        if ': error:' in line or line.startswith('error:'):
            compile_errors.append((cur or '?', line.strip()))
            continue
        m = WARN_RE.match(line.strip())
        if not m:
            continue
        findings.append((os.path.basename(m.group('file')),
                         m.group('flag') or '?',
                         '%s:%s:%s' % (os.path.basename(m.group('file')),
                                       m.group('line'), m.group('col')),
                         m.group('msg')))

    buckets = {'HIGH': [], 'MED': [], 'LOW': []}
    for hdr, flag, loc, msg in findings:
        buckets[bucket_bounds(flag)].append((hdr, flag, loc, msg))

    timed_out = sorted(t for t, (rc, _) in tu_info.items() if rc == '124')
    failed = sorted(t for t, (rc, _) in tu_info.items() if rc not in ('0', '124'))

    print('=' * 78)
    print('ZQCNN 主工程 gcc **-O2 -c** 优化期告警 sweep   total_TU=%d  墙钟=%.1fs'
          % (len(tus), wall))
    print('HIGH=%d  MED=%d  LOW=%d' % (len(buckets['HIGH']), len(buckets['MED']),
                                       len(buckets['LOW'])))
    if timed_out:
        print('\n!! %d 个 TU **超时**（单 TU 上限 %ds）—— 它们的告警没被扫到:'
              % (len(timed_out), per_tu_timeout))
        for t in timed_out:
            print('   %s   （加 --timeout %d 或 --all 再试）' % (t, per_tu_timeout * 2))
    if skipped:
        print('\n!! %d 个 TU **没跑**（预算耗尽）:' % len(skipped))
        for s in skipped:
            print('   %s' % s)
    if failed:
        print('\n!! %d 个 TU **编不过** —— 它们的告警没被扫到:' % len(failed))
        for t in failed:
            first = [m for h, m in compile_errors if h.startswith(t)]
            print('   %-38s %s' % (t, (first[0][:80] if first else '?')))
    slowest = sorted(((float(tt), t) for t, (rc, tt) in tu_info.items()
                      if tt not in ('?', 'None')), reverse=True)[:5]
    if slowest:
        print('\n最慢的 5 个 TU: ' + ', '.join('%s %.1fs' % (t, s) for s, t in slowest))

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
            print('   %-38s %-26s %-30s %s' % (hdr, flag, loc, msg[:70]))

    if bucket_filter:
        show(bucket_filter if bucket_filter in buckets else 'HIGH')
    else:
        show('HIGH')
        show('MED')
        if run_all:
            show('LOW')

    key_of = lambda t: (t[0], t[1], t[2])

    if save_to:
        high = sorted(set(key_of(t) for t in buckets['HIGH']))
        lines = ['# ZQCNN 主工程 gcc -O2 -c 优化期告警 HIGH 桶基线',
                 '# （tools/warn_sweep_bounds.py --save-baseline 生成）',
                 '# 格式: <文件名>\\t<-W标志>\\t<文件:行:列>',
                 '#',
                 '# 与 tools/zqcnn_warn_baseline.txt（-fsyntax-only 那条轴）的区别：',
                 '# 那一轴只看得到语义级告警；本轴多出的是**优化器才证明得了**的',
                 '# 那一类，附录 CT.4 的 `axis > 4` 就只在本轴上出现过。',
                 '# 基线的作用是拦**新增**，不是要求桶为空 ——',
                 '# 每一条都应当是「已判定：不是缺陷」并写明理由。']
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
        cur_set = set(key_of(t) for t in buckets['HIGH'])
        new = sorted(cur_set - base)
        fixed = sorted(base - cur_set)
        print('\n=== 与基线 %s 比对 ===' % check_against)
        print('HIGH: 基线 %d 条 -> 现在 %d 条' % (len(base), len(cur_set)))
        if new:
            print('\nNEW HIGH — 新出现的高信号警告:')
            for h, fl, lo in new:
                print('   %-38s %-26s %s' % (h, fl, lo))
        if fixed:
            print('\nFIXED — 比基线少了:')
            for h, fl, lo in fixed:
                print('   %-38s %-26s %s' % (h, fl, lo))
        if not new and not fixed:
            print('无新增、无消失。')
        return 1 if new else 0

    return 0


if __name__ == '__main__':
    sys.exit(main())
