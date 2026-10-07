#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 audit_k3_20261001.md 编成一份可检索的索引。

为什么要它（2026-10-07，附录 JN）
-----------------------------------
那份报告已经 **27465 行 / 260 个附录**。它是这次审计的**记录**，也是交付物 ——
但现在没人能在里面回答「**某某问题在哪一节**」。本工具把
「章节 -> 附录 -> 标题 -> 行号」抽出来，另加两条机器可读的标记：

* `[阴性]`：该附录的结论是**查过但没发现缺陷**（这很重要 —— 它们同样是
  「不用再查一遍」的依据，只看标题看不出来）；
* `[门禁]`：该附录产出了或加固了某道门禁（标题里通常带 Cnn）。

解析上的一个坑（实测踩到）
-------------------------
直接 `grep '^# '` 会把**代码块内部的注释**当成标题 ——
报告里嵌了大段源码，`# Windows 侧再手动一个个跑 exe` 这种行会被误收。
所以本工具**跟踪 ``` 围栏**，围栏内的一切都不当标题。
"""
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPORT = os.path.join(ROOT, 'audit_k3_20261001.md')
INDEX = os.path.join(ROOT, 'audit_k3_20261001_index.md')

FENCE = re.compile(r'^\s*```')
H2 = re.compile(r'^##\s+(.*)$')
APPX = re.compile(r'^附录\s+([A-Z]{1,3}[0-9]*)\s*[：:]\s*(.*)$')


def parse(path):
    """返回 (二级标题列表, 附录列表, 阴性附录集合, 重号列表)。

    每个附录记：编号、标题、起始行号、所属二级章节。
    `重号列表` = [(编号, [行号...])] —— **报告自身的编号也可能撞车**
    （2026-10-07 实测：IB~IY 各出现两次，是 23396 起那批复用了旧号；
    与 run_audit_checks.py 的门禁编号撞名是同一类病，见附录 IR）。
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    in_fence = False
    h2 = []                       # (标题, 行号)
    appx = []                     # (编号, 标题, 行号, 所属 h2)
    neg = set()
    cur_h2 = ''
    cur_appx = None
    by_id = defaultdict(list)
    for i, ln in enumerate(lines, 1):
        if FENCE.match(ln):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = H2.match(ln)
        if not m:
            # 阴性结论的标记：在**当前附录**的正文里
            if cur_appx and ('阴性结论' in ln):
                neg.add(cur_appx[0])
                if os.environ.get('ZQ_INDEX_DEBUG'):
                    print('  [%s] <- 阴性结论来自报告第 %d 行' % (cur_appx[0], i))
            continue
        title = m.group(1).strip()
        cur_h2 = title
        h2.append((title, i))
        ma = APPX.match(title)
        if ma:
            cur_appx = (ma.group(1), ma.group(2).strip(), i, cur_h2)
            appx.append(cur_appx)
            by_id[ma.group(1)].append(i)
        else:
            cur_appx = None
    dups = [(aid, lns) for aid, lns in by_id.items() if len(lns) > 1]
    return h2, appx, neg, dups


def build():
    h2, appx, neg, dups = parse(REPORT)
    out = []
    out.append('# ZQCNN 审计报告索引（自动生成，勿手改）')
    out.append('')
    out.append('由 `python tools/build_audit_index.py` 从 `audit_k3_20261001.md` 生成。')
    out.append('报告本体 %d 行；下面是「章节 -> 附录」的对照表，每条都能直接跳。'
               % sum(1 for _ in open(REPORT, encoding='utf-8')))
    out.append('')
    out.append('标记：`[阴性]` = 该附录查过但**结论是没发现缺陷**（同样是"不必重查"的依据）；'
               '`[门禁]` = 产出或加固了门禁。')
    out.append('')
    out.append('| 附录 | 标题 | 标记 | 报告行 |')
    out.append('|---|---|---|---|')
    for aid, title, line, sec in appx:
        marks = []
        if aid in neg:
            marks.append('阴性')
        if '门禁' in title or 'C1' in title or 'C2' in title:
            marks.append('门禁')
        # 同一编号多次出现（更正/补充）时标出来
        out.append('| %s | %s | %s | %d |'
                   % (aid, title.replace('|', '\\|'),
                      ' '.join(marks), line))
    out.append('')
    out.append('## 章节清单（%d 个二级标题）' % len(h2))
    out.append('')
    for title, line in h2:
        out.append('- L%d  %s' % (line, title))
    out.append('')
    with open(INDEX, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(out) + '\n')
    if dups:
        print('发现 %d 个重号附录（同一编号被多个附录占用）：' % len(dups))
        for aid, lns in sorted(dups, key=lambda x: x[1][0]):
            print('  * %s : %s' % (aid, ' / '.join('L%d' % x for x in lns)))
        print('')
        print('索引仍已生成，但**重号必须修**：给后写的附录换新编号，')
        print('并同步正文里的「附录 X / X.N」引用（附录 JN 的做法可参照）。')
    print('索引已写入 %s：%d 个附录 / %d 个二级标题 / %d 个阴性结论%s'
          % (os.path.basename(INDEX), len(appx), len(h2), len(neg),
             ' / %d 个重号' % len(dups) if dups else ''))
    return 1 if dups else 0


def selftest():
    """自测：围栏内的 '#' 必须被忽略；阴性标记要落在正确的附录上。"""
    import tempfile
    sample = (
        '# 报告\n'
        '```cpp\n'
        '# Windows 侧再手动一个个跑 exe\n'
        '```\n'
        '## 附录 AA：真标题\n'
        '正文。\n'
        '## 附录 BB：另一个\n'
        '这里是**阴性结论**：查过但没问题。\n'
        '## 普通小节\n')
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'r.md')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(sample)
    global REPORT
    old = REPORT
    REPORT = p
    try:
        h2, appx, neg, _d = parse(p)
        ok = True
        # 围栏里的那行不能进 h2
        for t, _ in h2:
            if 'Windows 侧再手动' in t:
                ok = False
        # 围栏之后正文不得算到上一个附录头上（本轮修的正是这个）
        sample2 = ('## 附录 CC：真阴性\n'
                    '```cpp\n'
                    'int x;\n'
                    '```\n'
                    '这里有**阴性结论**。\n')
        p2 = p + '.2'
        with open(p2, 'w', encoding='utf-8') as f:
            f.write(sample2)
        _h, _a, neg2, _d2 = parse(p2)
        if 'CC' not in neg2:
            ok = False
        os.remove(p2)
        # 两个附录都在
        if len(appx) != 2:
            ok = False
        # 阴性只能落在 BB 上（BB 之后的正文里才有那句话）
        if neg != {'BB'}:
            ok = False
        print('  [%s] 围栏内的 # 不算标题 / 附录数对 / 阴性归属正确'
              % ('PASS' if ok else 'FAIL'))
        return 0 if ok else 1
    finally:
        os.remove(p)
        os.rmdir(d)
        REPORT = old


def check():
    """不写索引文件，只判定：重号 -> 退出 1。给回归当门禁用。"""
    _h, appx, _neg, dups = parse(REPORT)
    if dups:
        print('发现 %d 个重号附录：' % len(dups))
        for aid, lns in sorted(dups, key=lambda x: x[1][0]):
            print('  * %s : %s' % (aid, ' / '.join('L%d' % x for x in lns)))
        return 1
    print('OK: %d 个附录编号唯一' % len(appx))
    return 0


def main():
    if '--selftest' in sys.argv:
        return selftest()
    if '--check' in sys.argv:
        return check()
    return build()


if __name__ == '__main__':
    sys.exit(main())