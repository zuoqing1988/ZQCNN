#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""门禁：门禁编号不能撞车。

起因（2026-10-06，附录 IR）
------------------------------
`run_audit_checks.py` 里每个组都带一个编号前缀（A1 / B / C5 / D3 …），
方便在日志和审计报告里指认。2026-10-06 发现**两处撞名**：

    C5  文件级可达性门禁（没有任何构建编过的文件）
    C5  主工程 -O2 -c 优化期告警 HIGH 桶门禁
    C7  sample 不得含未注释的 GUI 调用（用户指令门禁）
    C7  层类型可达性门禁（EXERCISED/COMMENTED/UNUSED）

撞名的后果不是"跑重了"，而是**日志不可读**：门禁红了报一句 `C7 FAILED`，
看的人无从判断是哪一道。而这不是第一次 —— 源码里那段注释本来就写着
"C4 那次撞名是我自己犯的……这里直接避开"，结果 C7 又撞上了。

所以重命名（C7->C21、C5->C22、C5->C23）之后，把"编号唯一"本身变成门禁。

范围只限**编号唯一**这一条。顺带试过再加一条"`run_group` 的名字必须出现在
`failed.append` 里"，**行不通，已放弃**：D1/D2/D3 走的是 `ok &= run_group(...)`
累积、最后由父组 `failed.append('D 双平台构建 + sample 回归')` 统一记账，
本来就不逐条 append；B2 那种 `...(MSVC ASan)` vs `...(MSVC /fsanitize=address)`
也是有意写得短。硬加这条只会得到一堆误报，而判据站不住就该砍掉。
"""
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, 'run_audit_checks.py')
RUN_GROUP_RE = re.compile(r"run_group\(\s*'([^']+)'")


def collect(module, src):
    """返回 {编号: {组名集合}}。

    记的是**组名**而不是"来源"。第一版记来源（GROUPS / run_group），
    结果"两个 GROUPS 撞名"这种**最该抓**的情况反而看不见 ——
    两个条目同属 GROUPS，来源集合大小仍是 1。是自测先抓住的。
    """
    ids = {}
    for name, _, _ in module.GROUPS:
        ids.setdefault(name.split()[0], set()).add(name)
    for mt in RUN_GROUP_RE.finditer(src):
        name = mt.group(1)
        ids.setdefault(name.split()[0], set()).add(name)
    return ids


def audit(module, src):
    """编号撞车 = 同一个编号下出现了**不止一个组名**。"""
    ids = collect(module, src)
    return {k: v for k, v in ids.items() if len(v) > 1}


def selftest():
    """阳性/阴性对照。"""
    class Fake(object):
        pass

    def mk(groups, calls):
        f = Fake()
        f.GROUPS = [(g, [], False) for g in groups]
        src = ''.join("run_group('%s'," % c for c in calls)
        return f, src

    cases = [
        ('正常：编号互不相同', mk(['C5 a', 'C7 b'], ['C22 c']), 0),
        ('阳性：GROUPS 与 run_group 撞名', mk(['C7 a'], ['C7 b']), 1),
        ('阳性：两个 GROUPS 撞名', mk(['C5 a', 'C5 b'], []), 1),
        ('阴性：前缀不同不算撞名', mk(['C5 a', 'C5b b'], []), 0),
        ('阴性：C5b 与 C5 是两个编号', mk(['C5 a'], ['C5b b']), 0),
    ]
    bad = []
    for name, (mod, src), expect in cases:
        got = len(audit(mod, src))
        ok = got == expect
        print('  [%s] %-38s expect=%d got=%d'
              % ('PASS' if ok else 'FAIL', name, expect, got))
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
    spec = importlib.util.spec_from_file_location('rac_for_ids', TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with open(TARGET, 'r', encoding='utf-8') as f:
        src = f.read()

    ids = collect(mod, src)
    dup = audit(mod, src)
    print('门禁编号总数：%d' % len(ids))
    if dup:
        print('')
        print('发现 %d 处编号撞车：' % len(dup))
        for k in sorted(dup):
            print('  * %s 被 %d 个组同时占用：%s'
                  % (k, len(dup[k]), ' / '.join(sorted(dup[k]))))
        return 1
    print('')
    print('OK: 所有门禁编号唯一')
    return 0


if __name__ == '__main__':
    sys.exit(main())