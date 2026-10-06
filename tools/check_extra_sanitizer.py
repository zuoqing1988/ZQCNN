#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""门禁：`EXTRA_SOURCES` 的编译行必须带 `$SAN`，除非该 tag 有意豁免。

为什么需要它（2026-10-06，附录 IM/IN/IO）
------------------------------------------------
`EXTRA_SOURCES` 里每一条 `gcc/g++ ... -c` 就是在编**被测库的实现 TU**。
不带 `$SAN` 时 ASan 只覆盖测试自己的代码 —— 被测代码里的越界读**不会被报**，
只有读到未映射页才会 SEGV；"读超了一两个元素"这种在同一页之内的完全静默。

补插桩这件事手工做了三轮（附录 IM 只补了 1 个 tag、IN 补了 5 个、
IO 补完剩下 18 个才补齐），而它是个**随时可能被下一次编辑破坏**的性质：
少写一个 `$SAN` 不会让任何测试变红，只是让覆盖悄悄少一块。
所以落成门禁。

为什么必须有豁免名单（附录 IO）
------------------------------------------------
`zq_nchw_conv_free` 在 `EXTRA_CXXFLAGS` 里带 `-fno-sanitize=address`：
它要自己接管 `free`，而 ASan 运行时自己也要调 `free`（附录 CU.9）。
给它补 `$SAN` 的后果是**编译行插桩、链接行不插桩**，
`.o` 里的 `__asan_*` 引用没人提供 -> `collect2: error: ld returned 1 exit status`。

问题不在例外存在，而在于**例外写在另一个字典里**（EXTRA_CXXFLAGS 在 980 行
开外，EXTRA_SOURCES 在前 700 行）。本门禁把两边的关系显式断言出来，
下一次谁再加一个有意豁免的 tag，本门禁会要求他在 EXTRA_SOURCES 旁边留证据。
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, 'run_zqlib_checks.py')


def load_module():
    """按文件路径 import，避免依赖 sys.path 里的名字。"""
    spec = importlib.util.spec_from_file_location('rzc_for_san_check', TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def is_compile_line(e):
    """只认「编一个 .o」的命令：gcc/g++ 开头且含 ` -c `。"""
    return (e.startswith('gcc ') or e.startswith('g++ ')) and ' -c ' in e


def check(extra_sources, extra_cxxflags, src_text=''):
    """返回 (violations, stats)。

    两条规则：
      1. 编译行必须带 $SAN，除非该 tag 在 EXTRA_CXXFLAGS 里显式 -fno-sanitize；
      2. 已经在 EXTRA_CXXFLAGS 里豁免的 tag，其编译行**必须不带** $SAN
         —— 否则就是「编译行插桩、链接行不插桩」的那个链接错配。
    """
    opt_out = {t for t, f in extra_cxxflags.items() if 'fno-sanitize' in f}
    violations = []
    stats = {'compile_lines': 0, 'instrumented': 0,
             'tags': len(extra_sources), 'opt_out': sorted(opt_out)}

    for tag in sorted(extra_sources):
        for e in extra_sources[tag]:
            if not is_compile_line(e):
                continue
            stats['compile_lines'] += 1
            has_san = '$SAN' in e
            if has_san:
                stats['instrumented'] += 1
            if tag in opt_out:
                if has_san:
                    violations.append(
                        '%s: 该 tag 在 EXTRA_CXXFLAGS 里带了 -fno-sanitize（有意豁免），'
                        '但它的编译行又带了 $SAN —— 编译行插桩而链接行不插桩，'
                        '会在链接期报 collect2: error: ld returned 1 exit status' % tag)
            elif not has_san:
                violations.append(
                    '%s: 编译行没有 $SAN（被测的实现 TU 没被插桩，ASan 看不到它内部的越界读）'
                    % tag)

    # 豁免必须**看得见**：EXTRA_SOURCES 那一段里要留下文字证据，
    # 否则下一个改 EXTRA_SOURCES 的人看不见例外从哪来。
    if opt_out and src_text:
        for tag in sorted(opt_out):
            if tag not in src_text:
                violations.append(
                    '%s 在 EXTRA_CXXFLAGS 里豁免了 sanitizer，但 EXTRA_SOURCES 文件里'
                    '根本看不到这个 tag —— 例外的证据要写在规则旁边' % tag)
    return violations, stats


def selftest():
    """阳性/阴性对照。没有自测的判据不能证明自己有鉴别力。

    每条用例是 (名字, EXTRA_SOURCES, EXTRA_CXXFLAGS, 期望违规数, src_text)。
    `src_text` 模拟 EXTRA_SOURCES 那个文件的正文 —— 只有"豁免可见性"那条
    需要它，故意只给不含豁免 tag 的内容。
    """
    cases = [
        ('正常：都插桩且无豁免',
         {'a': ['gcc -O1 -g $SAN -mavx2 -c x.c -o x.o']}, {}, 0, 'a'),
        ('正常：豁免的 tag 编译行也不带 $SAN',
         {'a': ['gcc -O1 -g -mavx2 -c x.c -o x.o']},
         {'a': ' -fno-sanitize=address'}, 0, 'a'),
        ('阳性：普通 tag 漏了 $SAN',
         {'a': ['gcc -O1 -g -mavx2 -c x.c -o x.o']}, {}, 1, 'a'),
        ('阳性：豁免的 tag 却带了 $SAN（链接错配）',
         {'a': ['gcc -O1 -g $SAN -mavx2 -c x.c -o x.o']},
         {'a': ' -fno-sanitize=address'}, 1, 'a'),
        ('阳性：g++ 的编译行同样受管',
         {'a': ['g++ -O1 -g -mavx2 -mfma -c x.cpp -o x.o']}, {}, 1, 'a'),
        ('阴性：非编译行（拷贝/探测）不受管',
         {'a': ['if [ -z "$OCV" ]; then echo NOOPENCV; fi']}, {}, 0, 'a'),
        ('阳性：一个 tag 里漏一条也算',
         {'a': ['gcc -O1 -g $SAN -mavx2 -c x.c -o x.o',
                'gcc -O1 -g -mavx2 -c y.c -o y.o']}, {}, 1, 'a'),
        ('阳性：豁免 tag 在 EXTRA_SOURCES 里看不到证据',
         {'b': ['gcc -O1 -g $SAN -mavx2 -c x.c -o x.o']},
         {'a': ' -fno-sanitize=address'}, 1, 'b'),
    ]
    bad = []
    for name, ex, cx, expect, src in cases:
        v, _ = check(ex, cx, src_text=src)
        ok = len(v) == expect
        print('  [%s] %-46s expect=%d got=%d'
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
    mod = load_module()
    with open(TARGET, 'r', encoding='utf-8') as f:
        src_text = f.read()
    violations, stats = check(mod.EXTRA_SOURCES, mod.EXTRA_CXXFLAGS, src_text)

    print('tag 数            : %d' % stats['tags'])
    print('编译行总数        : %d' % stats['compile_lines'])
    print('已插桩($SAN)      : %d' % stats['instrumented'])
    print('有意豁免的 tag    : %s' % (', '.join(stats['opt_out']) or '(无)'))
    if violations:
        print('')
        print('发现 %d 处问题：' % len(violations))
        for v in violations:
            print('  * %s' % v)
        return 1
    print('')
    print('OK: 所有编译行都带 $SAN（豁免名单：%s）'
          % (', '.join(stats['opt_out']) or '无'))
    return 0


if __name__ == '__main__':
    sys.exit(main())