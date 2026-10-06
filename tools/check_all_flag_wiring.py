#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""门禁：`--all` 必须真的覆盖 argparse 里**每一个** opt-in 开关。

起因（2026-10-06，附录 IQ）
-----------------------------
`run_audit_checks.py` 有 9 个 `action='store_true'` 的开关，各自管着一整块
覆盖：双平台构建、MSVC ASan、MSVC 探针、两条 warn 轴、文件可达性、
UBSan 轴、以及那 8 个编译慢的测试。

而"全量回归"这条命令**在仓库里根本不存在** —— AGENTS.md 只逐条列出开关，
从没把它们组合过。于是每加一个开关就要记得往那条口口相传的命令里补一次，
**漏了没人知道**。后果已经发生过：附录 IK 发现 `--with-slow` 从来没有被
任何入口传过，6 个 GEMM 调度调用点测试因此**从未被自动执行过**。

`--all` 把这条命令固化下来；本门禁盯住它别再腐烂 ——
新加一个 store_true 开关却忘了写进 `OPT_IN_FLAGS`，它就永远不会被 `--all`
打开，而"全量回归"照样跑完、照样全绿、照样什么都没多验。
"""
import contextlib
import importlib.util
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, 'run_audit_checks.py')

# argparse 里 store_true 的开关长这样：add_argument('--foo', action='store_true'
STORE_TRUE_RE = re.compile(
    r"add_argument\(\s*'(--[a-z0-9-]+)'\s*,\s*action='store_true'")


def scan_store_true(src):
    return sorted(set(STORE_TRUE_RE.findall(src)))


def to_dest(flag):
    """`--with-build` -> `with_build`，即 argparse 的 dest。

    比对必须在这一层做：OPT_IN_FLAGS 里存的是 **dest**，而正则扫出来的是
    **flag 名**（带两个连字符、用连字符分隔）。直接在两套命名之间比，
    会让每个开关都报"未分类" —— 本门禁第一版就是这么全红的。
    """
    return flag.lstrip('-').replace('-', '_')


def audit(src, opt_in, non_opt_in):
    """返回 (violations, flags)。

    `opt_in` / `non_opt_in` 是**dest 名**（`with_build`）。
    规则：每个 store_true 开关必须在其中之一被显式分类。
    未分类 = 新加的开关忘了决定要不要进 --all —— 这正是要抓的。
    """
    flags = scan_store_true(src)
    dests = {to_dest(f) for f in flags}
    oin = set(opt_in)
    non = set(non_opt_in)
    violations = []
    for f in flags:
        d = to_dest(f)
        if d in oin or d in non:
            continue
        violations.append(
            '%s 是 store_true 开关，但既不在 OPT_IN_FLAGS 也不在 NON_OPT_IN_FLAGS 里 —— '
            '新加的开关必须显式决定要不要进 --all' % f)
    # 反向：OPT_IN_FLAGS 里不能有不存在的开关（改名后忘了更新表）
    for d in sorted(oin):
        if d not in dests:
            violations.append(
                'OPT_IN_FLAGS 里的 %s 在 argparse 里已经不存在（改名了？）—— '
                '--all 会 setattr 一个不存在的属性' % d)
    return violations, flags


def selftest():
    """阳性/阴性对照。"""
    base = ("ap.add_argument('--with-build', action='store_true')\n"
            "ap.add_argument('--quick', action='store_true')\n")
    cases = [
        ('正常：两个开关都被分类',
         base, ['with_build'], {'quick'}, 0),
        ('阳性：新开关忘了分类',
         base + "ap.add_argument('--brand-new', action='store_true')\n",
         ['with_build'], {'quick'}, 1),
        ('阳性：OPT_IN_FLAGS 指向了不存在的开关',
         base, ['with_build', 'typo_flag'], {'quick'}, 1),
        ('阴性：非 store_true 的开关不受管',
         "ap.add_argument('--filter', type=str)\n"
         "ap.add_argument('--limit', default=10)\n",
         [], {}, 0),
        ('阴性：store=False 不是 opt-in 开关',
         "ap.add_argument('--interactive', action='store_false')\n",
         [], {}, 0),
        ('阳性：新增开关在最末尾也照样抓得到',
         base + "\n\n\nap.add_argument('--tail', action='store_true',\n"
               "                help='x')\n",
         ['with_build'], {'quick'}, 1),
    ]
    bad = []
    for name, src, oin, non, expect in cases:
        v, _ = audit(src, oin, non)
        ok = len(v) == expect
        print('  [%s] %-42s expect=%d got=%d'
              % ('PASS' if ok else 'FAIL', name, expect, len(v)))
        if not ok:
            bad.append(name)
    if bad:
        print('SELFTEST FAILED: %s' % ', '.join(bad))
        return 1
    print('selftest OK: %d cases' % len(cases))
    return 0


def dry_groups(argv):
    """跑一遍 main() 但把 run_group / run_build_group 换掉，只收集要跑的组。

    返回 [(组名, 命令元组)]。**命令也要比**：有些开关（--with-slow）
    不新增任何组，只是给已有的 B 组命令**追加一个参数** ——
    只比组名的话这类开关是**完全看不见的**，而"看不见"正是本门禁要防的那类。
    """
    spec = importlib.util.spec_from_file_location('rac_dry', TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seen = []
    mod.run_group = lambda name, cmd, cwd=None, shell=False: (
        seen.append((name, tuple(cmd))), True)[1]
    mod.run_build_group = lambda: (
        seen.append(('D_dual_platform_build', ())), True)[1]
    old_argv = sys.argv
    try:
        sys.argv = ['run_audit_checks.py'] + list(argv)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                mod.main()
            except SystemExit:
                pass
    finally:
        sys.argv = old_argv
    return seen, mod


def verify_coverage():
    """`--all` 必须等价于"每个 opt-in 开关单独打开"的并集。

    比**组名 + 命令**两样。只比组名会漏掉 --with-slow 这类"只改命令不加组"
    的开关 —— 而漏掉正是这个门禁存在的理由。
    """
    allg, allmod = dry_groups(['--all'])
    base, _ = dry_groups([])
    base_set = set(base)
    all_set = set(allg)

    opt_in = ['--' + f.replace('_', '-') for f in allmod.OPT_IN_FLAGS]
    union = set()
    for f in opt_in:
        g, _ = dry_groups([f])
        union |= set(g) - base_set

    violations = []
    missing = sorted(union - all_set)
    extra = sorted(all_set - union - base_set)
    if missing:
        violations.append('--all 漏掉了这些组/命令: %s' % missing)
    if extra:
        violations.append('--all 多跑了这些组/命令: %s' % extra)

    # 单独验证 --with-slow 的效果真的看得见（否则上面那个比较形同虚设）
    slow, _ = dry_groups(['--with-slow'])
    slow_new = set(slow) - base_set
    all_new = set(allg) - base_set
    if not slow_new:
        violations.append(
            '--with-slow 单独跑**没有任何组/命令变化** —— 它只是给 B 组追加参数，'
            '按名字比对时会被漏掉；请确认本门禁比的是命令而不是只比名字')

    print('组/命令：无参数 %d 组，--all %d 组，--all 额外 %d 条'
          % (len(base), len(allg), len(all_new)))
    print('单独开每个开关的并集额外：%d 条' % len(union))
    if violations:
        print('')
        for v in violations:
            print('  * %s' % v)
        return 1
    print('OK: --all 与逐个开关的并集完全一致（组名与命令都比了）')
    return 0


def main():
    if '--selftest' in sys.argv:
        return selftest()
    if '--verify-coverage' in sys.argv:
        return verify_coverage()
    spec = importlib.util.spec_from_file_location('rac_for_allflag', TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with open(TARGET, 'r', encoding='utf-8') as f:
        src = f.read()

    violations, flags = audit(src, mod.OPT_IN_FLAGS, mod.NON_OPT_IN_FLAGS)

    print('argparse 里 store_true 的开关 : %d' % len(flags))
    for f in flags:
        d = to_dest(f)
        if d in mod.OPT_IN_FLAGS:
            tag = '--all 会打开'
        else:
            tag = '有意排除：' + mod.NON_OPT_IN_FLAGS.get(d, '?')
        print('   %-22s %s' % (f, tag))
    if violations:
        print('')
        print('发现 %d 处问题：' % len(violations))
        for v in violations:
            print('  * %s' % v)
        return 1
    print('')
    print('OK: 每个 store_true 开关都已分类，--all 覆盖完整')
    return 0


if __name__ == '__main__':
    sys.exit(main())