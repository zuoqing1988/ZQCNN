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
import importlib.util
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


def main():
    if '--selftest' in sys.argv:
        return selftest()
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