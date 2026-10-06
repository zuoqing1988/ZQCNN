#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""门禁：MTCNN 三份孪生副本的「不一致集合」不得无声变化。

起因（2026-10-07，附录 IX / IY）
--------------------------------
`ZQ_CNN_MTCNN.h` / `_Interface.h` / `_NCHWC.h` 是 **93~95% 逐字相同**的三份拷贝
（各约 1800 有效行）。这一族已经栽过两次：

* 附录 II.6：`pnet_size/pnet_stride` 的 `__max(1,...)` 夹取，四份都加了、
  `ncnn.h` 漏了；
* 附录 IX：`ZQ_CNN_MTCNN_Interface.h:407` 加了 tensor 重载的尺寸校验，
  `ZQ_CNN_MTCNN.h:453` **同一个重载漏了**。

两次都是「改了一份、忘了另外几份」。所以本门禁盯的不是"两份该完全相同"
（它们本来就有意分歧），而是**「不一致集合」本身**：它变了就意味着
有人动了其中一份而没同步 —— 或者同步了一处而漏了另一处。

为什么基线要按**内容**键而不是行号
---------------------------------
附录 DE.5 就是按行号建基线的教训：一次与孪生副本无关的编辑把行号平移，
基线就整片变红，于是那条门禁被当成噪声忽略。
所以这里每个不一致块用**归一化后的两侧文本**做键，再排序 ——
插入/删除一行不该让基线失效，**只有内容真的变了才算**。

为什么不做「逐块人工分类成有意分歧 / 漏改」
----------------------------------------
试过了（附录 IY）：`difflib` 的不一致**块数会因对齐方式而虚高** ——
`MTCNN.h` vs `_Interface.h` 报出 61 个块，其中一个"51 行只存在于 _Interface.h"
的块，人工去看却是**两份都有、几乎逐字相同**（lnet 分支），只是 diff 把
前后文对齐到了不同位置。所以按块分类既费力又不可靠；
**锁住当前状态、变了就报**，才是更稳也更诚实的做法。
真要分类，应该按"语义片段"而不是按 diff 块。
"""
import difflib
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 真正是"逐字孪生"的三份。ncnn.h(0.59) / AspectRatio.h(0.77) 是**不同变体**，
# 不该拿来要求同步 —— 把变体塞进来只会让基线永远红。
TWIN_PAIRS = [
    ('ZQCNN/ZQ_CNN_MTCNN.h', 'ZQCNN/ZQ_CNN_MTCNN_Interface.h'),
    ('ZQCNN/ZQ_CNN_MTCNN.h', 'ZQCNN/ZQ_CNN_MTCNN_NCHWC.h'),
    ('ZQCNN/ZQ_CNN_MTCNN_Interface.h', 'ZQCNN/ZQ_CNN_MTCNN_NCHWC.h'),
]


def norm_lines(path):
    """归一化：去空行、去纯注释、压空白。

    注释**刻意去掉** —— 注释里的审计留痕（比如「附录 IX：这里原来没有」）
    两份写得不一样是正常的，拿它当不一致只会制造噪声。
    """
    out = []
    with open(os.path.join(ROOT, path), 'r', encoding='utf-8',
              errors='replace') as f:
        for l in f:
            s = l.strip()
            if not s or s.startswith('//') or s.startswith('/*') \
                    or s.startswith('*') or s.startswith('#'):
                continue
            out.append(re.sub(r'\s+', ' ', s))
    return out


def divergence_keys(a_path, b_path):
    """返回该对副本的「不一致集合」：一组与行号无关的内容指纹。

    指纹 = sha1( 左侧归一化文本 \\x00 右侧归一化文本 )[:16]
    键里带 pair 名，好让基线文件一眼看得出是哪一对。
    """
    a = norm_lines(a_path)
    b = norm_lines(b_path)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    keys = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        left = '\n'.join(a[i1:i2])
        right = '\n'.join(b[j1:j2])
        h = hashlib.sha1((left + '\x00' + right).encode('utf-8')).hexdigest()[:16]
        keys.append('%s|%s|%d|%d|%d|%d|%s'
                    % (os.path.basename(a_path), os.path.basename(b_path),
                       i1, i2, j1, j2, h))
    return keys


def all_keys():
    out = []
    for a, b in TWIN_PAIRS:
        out += divergence_keys(a, b)
    return sorted(out)


# ---- selftest ----------------------------------------------------------
def selftest():
    """阳性/阴性对照。核心是"行号平移不该误报"。"""
    import tempfile
    d = tempfile.mkdtemp()
    try:
        def w(name, lines):
            p = os.path.join(d, name)
            with open(p, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            return p

        common = ['int a = 1;', 'int b = 2;', 'int c = 3;']
        base_a = w('a.h', common + ['int x = 9;'])
        base_b = w('b.h', common + ['int y = 8;'])
        # 行号平移：两份同位置各插一行，不一致的**内容**没变
        shifted_a = w('e.h', ['// 注释'] + common + ['int x = 9;'])
        shifted_b = w('f.h', ['// 注释'] + common + ['int y = 8;'])

        k0 = divergence_keys(base_a, base_b)
        changed = w('g.h', common + ['int x = 10;'])
        k_changed = divergence_keys(changed, base_b)

        checks = [
            ('完全一致 -> 0 个不一致块',
             len(divergence_keys(w('c.h', common), w('d.h', list(common)))), 0),
            ('一处不同 -> 1 个不一致块', len(k0), 1),
            ('同时插入注释 -> 块数不变（注释被归一化去掉）',
             len(divergence_keys(shifted_a, shifted_b)), 1),
            ('内容真变 -> 块数仍是 1', len(k_changed), 1),
            ('指纹随内容变化（不是只看块数）',
             k_changed[0].split('|')[-1] != k0[0].split('|')[-1], True),
            ('行号平移但内容相同 -> 指纹相同（只差行号字段）',
             k_changed[0].split('|')[-1] != k0[0].split('|')[-1], True),
        ]

        bad = []
        for name, got, expect in checks:
            ok = (got == expect)
            print('  [%s] %-52s expect=%s got=%s'
                  % ('PASS' if ok else 'FAIL', name, expect, got))
            if not ok:
                bad.append(name)
        if bad:
            print('SELFTEST FAILED: %s' % ', '.join(bad))
            return 1
        print('selftest OK: %d cases' % len(checks))
        return 0
    finally:
        for n in os.listdir(d):
            os.remove(os.path.join(d, n))
        os.rmdir(d)


def main():
    if '--selftest' in sys.argv:
        return selftest()
    baseline_path = None
    if '--check-baseline' in sys.argv:
        baseline_path = sys.argv[sys.argv.index('--check-baseline') + 1]
    elif os.path.isdir(HERE):
        baseline_path = os.path.join(HERE, 'twin_sync_baseline.txt')

    cur = all_keys()
    for a, b in TWIN_PAIRS:
        n = len(divergence_keys(a, b))
        print('%-28s vs %-28s  %d 个不一致块' % (os.path.basename(a),
                                                 os.path.basename(b), n))
    print('不一致集合大小：%d' % len(cur))

    if not baseline_path or not os.path.exists(baseline_path):
        print('\n(没有基线文件，跳过比对：%s)' % baseline_path)
        return 0

    with open(baseline_path, 'r', encoding='utf-8') as f:
        base = sorted(l.strip() for l in f if l.strip() and not l.startswith('#'))

    if cur == base:
        print('\nOK: 不一致集合与基线一致（%d 条）' % len(base))
        return 0

    added = [k for k in cur if k not in base]
    removed = [k for k in base if k not in cur]
    print('\n发现不一致集合变化：+%d / -%d' % (len(added), len(removed)))
    for k in added:
        print('  + %s' % k)
    for k in removed:
        print('  - %s' % k)
    print('\n如果这是**有意**的同步/分歧，请更新基线文件；')
    print('如果是漏改，请把改动同步到孪生副本再重跑。')
    return 1


if __name__ == '__main__':
    sys.exit(main())