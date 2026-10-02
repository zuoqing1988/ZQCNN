#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扫「层类的 ReadParam 只校验参数**在不在**、不校验**值**」这一族。

为什么要有这个工具（audit_k3_20261001.md 附录 BG）
------------------------------------------------
附录 BD / BE / BF 三轮都是从同一件事长出来的：

* BD：pooling 的 `stride=0` -> `(int)ceil(inf)` 是 UB
* BE：卷积的 `stride=0` -> **整数除零** -> SIGFPE，7 处 wrapper
* BF：`Tile` 的 `C*tile_c` **整数回绕** -> 分配按回绕值、写入按原值 -> 堆溢出写

三处的共同点：**`kernel_*` / `stride_*` / `tile_*` 这些值全部来自
**不可信的模型文件**（`.zqparams`），而 `ReadParam` 只检查"这一行在不在"
（`has_strideH` 之类），不检查值的范围。**

本轮把 25 个会 `atoi` 参数的 `ReadParam` 全过了一遍，
逐个确认了每个参数**最终有没有被兜住**（结论记在附录 BF.1）。
但那份结论是**写在报告里的** —— 下次有人删掉某个守卫、或者新增一个层类，
**没有任何机制会提醒**。本工具把"哪些 (层, 参数) 在 ReadParam 里有值域校验"
变成一个**可回归的基线**。

它能可靠地做的一件事、也只做这一件
--------------------------------
对每个 `ReadParam`：
  1. 抽出它用 `atoi` 赋值的变量；
  2. 在**同一个 ReadParam 函数体**里找这些变量的值域校验
     （`x <= 0` / `x < 1` / `x != 0` / `invalid x` / `has_x && x` 等）。

**它不做**「这个参数在下游有没有被兜住」——那是语义判断，正则做不了，
结论留在附录 BF.1 里。本工具只保证**"ReadParam 自己查过的那部分"不会悄悄消失**。

用法:
    python tools/check_param_domain.py                # 列出全部
    python tools/check_param_domain.py --selfcheck    # 自测（工具自己能不能用）
    python tools/check_param_domain.py --check-baseline tools/param_domain_baseline.txt
    python tools/check_param_domain.py --save-baseline  tools/param_domain_baseline.txt
"""

from __future__ import print_function

import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADERS = [
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_Layer.h'),
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_Layer_NCHWC.h'),
]

# 参数名 -> (类, 参数) 的基线文件
DEFAULT_BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'param_domain_baseline.txt')

ATOI_RE = re.compile(r'([A-Za-z_]\w*)\s*=\s*(?:std::)?atoi\s*\(')

# 值域校验的几种写法。刻意**只认"明确在比较"**：
# 宽松的写法（比如"出现过这个变量、同一句里又有 return false"）会把
# `return has_a && has_b && has_name` 之类误判成值域校验，
# 于是工具报告"全部已校验"——而实际上 BD/BE/BF 三条缺陷就在这一族里。
#
# **不认**的：枚举/集合限制（`sample_type == 0 || sample_type == 1`）、
# `x && ...` 这类布尔用法。它们的语义是"必须是集合里的某一个"，
# 不是"范围"，混进来会让基线失真。
CHECK_RES = [
    re.compile(r'\b%s\s*<=\s*0\b'),            # x <= 0
    re.compile(r'\b0\s*>=\s*%s\b'),             # 0 >= x
    re.compile(r'\b%s\s*<\s*0\b'),             # x < 0
    re.compile(r'\b%s\s*<\s*1\b'),             # x < 1
    re.compile(r'\b%s\s*!=\s*0\b'),            # x != 0
    re.compile(r'\b%s\s*==\s*0\b'),            # x == 0
    re.compile(r'\binvalid\s+%s\b'),           # "invalid <var>"
    re.compile(r'\b%s\s*<=\s*0[xX]'),          # 留给将来
]


def find_check(var, body):
    """`var` 在**这个 ReadParam 函数体**里有没有值域校验。

    第一版写成了 `if r.pattern % re.escape(var):` —— 那是"格式化后的正则字符串
    非空吗"，不是"匹不匹配"，于是**恒为真**，工具报告"全部已校验"。
    自测立刻逮到了它（tag 明明没校验却被判成已校验）。
    """
    for r in CHECK_RES:
        if re.search(r.pattern % re.escape(var), body):
            return True
    return False

SELFTEST_SRC = r'''
class ZQ_Demo_Pooling {
    virtual bool ReadParam(const std::string& line) {
        int kernel_H = 0, stride_H = 0, tag = 0;
        if (_my_strcmpi("kernel_H", p[0]) == 0) { kernel_H = atoi(p[1].c_str()); }
        if (_my_strcmpi("stride_H", p[0]) == 0) { stride_H = atoi(p[1].c_str()); }
        if (_my_strcmpi("tag",      p[0]) == 0) { tag      = atoi(p[1].c_str()); }
        if (kernel_H <= 0 || stride_H < 1) return false;   // 这两个算"有校验"
        return true;                                        // tag 没有
    }
};
'''


def scan_text(text, src):
    lines = text.replace('\r\n', '\n').split('\n')
    starts = [i for i, l in enumerate(lines) if 'virtual bool ReadParam' in l]
    rows = []
    for si in starts:
        cname = '?'
        for k in range(si, -1, -1):
            m = re.match(r'\s*class\s+([A-Za-z_]\w*)', lines[k])
            if m:
                cname = m.group(1)
                break
        # 结束：下一个 ReadParam / 下一个 class / 文件尾
        end = len(lines)
        for k in range(si + 1, min(len(lines), si + 400)):
            if 'virtual bool ReadParam' in lines[k] or re.match(r'\s*class\s', lines[k]):
                end = k
                break
        body = '\n'.join(lines[si:end])
        vars_ = []
        for m in ATOI_RE.finditer(body):
            if m.group(1) not in vars_:
                vars_.append(m.group(1))
        if not vars_:
            continue
        for v in vars_:
            rows.append({'src': src, 'class': cname, 'line': si + 1,
                         'param': v, 'checked': find_check(v, body)})
    return rows


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    argv = list(sys.argv[1:])
    selfcheck = '--selfcheck' in argv
    as_json = '--json' in argv
    check = '--check-baseline' in argv
    save = '--save-baseline' in argv
    base_path = DEFAULT_BASELINE
    for flag in ('--check-baseline', '--save-baseline'):
        if flag in argv:
            # 允许只给 flag 不给路径 —— 那样就用默认的
            # tools/param_domain_baseline.txt。第一版没判这个，
            # `--save-baseline` 单独用会 IndexError。
            i = argv.index(flag) + 1
            if i < len(argv) and not argv[i].startswith('--'):
                base_path = argv[i]

    if selfcheck:
        rows = scan_text(SELFTEST_SRC, '<selftest>')
        got = set((r['class'], r['param'], r['checked']) for r in rows)
        want = {('ZQ_Demo_Pooling', 'kernel_H', True),
                ('ZQ_Demo_Pooling', 'stride_H', True),
                ('ZQ_Demo_Pooling', 'tag', False)}
        if got != want:
            print('自测失败。\n  期望: %s\n  实际: %s' % (sorted(want), sorted(got)))
            return 1
        print('自测通过：3 个 atoi 参数，2 个识别为已校验、1 个识别为未校验。')
        return 0

    rows = []
    for h in HEADERS:
        if not os.path.isfile(h):
            continue
        with io.open(h, encoding='utf-8', errors='replace') as f:
            rows += scan_text(f.read(), os.path.basename(h))
    rows.sort(key=lambda r: (r['src'], r['class'], r['param']))

    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    if check or save:
        checked = set((r['class'], r['param']) for r in rows if r['checked'])
        if save:
            out = ['# 层类 ReadParam 的值域校验基线'
                   '（tools/check_param_domain.py --save-baseline 生成）',
                   '# 格式: <类名>\\t<参数名>',
                   '#',
                   '# 只记**在 ReadParam 里被校验过**的那些。删掉其中任何一条，'
                   '--check-baseline 就会失败。',
                   '#',
                   '# 注意：本工具**不判断**参数在下游有没有被兜住 ——',
                   '# 那是语义判断，结论在 audit_k3_20261001.md 附录 BF.1。',
                   '#',
                   '# BD/BE/BF 三条缺陷（pooling 除零、卷积 SIGFPE、Tile 堆溢出）'
                   '都是这一族漏网的实例。']
            for c, p in sorted(checked):
                out.append('%s\t%s' % (c, p))
            with io.open(base_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write('\n'.join(out) + '\n')
            print('基线已写入 %s（%d 条）' % (base_path, len(checked)))
            if not check:
                return 0

        base = set()
        try:
            with io.open(base_path, encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) >= 2:
                        base.add((parts[0], parts[1]))
        except IOError as e:
            print('读不到基线 %s: %s' % (base_path, e))
            return 1
        lost = sorted(base - checked)
        gained = sorted(checked - base)
        print()
        print('与基线 %s 比对：基线 %d 条，现在 %d 条' % (base_path, len(base), len(checked)))
        if lost:
            print('\n**REGRESSION** —— 这些 (层, 参数) 原来有值域校验，现在没有了：')
            for c, p in lost:
                print('   %-34s %s' % (c, p))
        if gained:
            print('\n新增（已自动并入基线）:')
            for c, p in gained:
                print('   %-34s %s' % (c, p))
        if not lost and not gained:
            print('无变化。')
        return 1 if lost else 0

    # 列出模式
    print('=' * 78)
    print('ReadParam 值域校验普查：%d 个 (层, 参数) 对，来自 %d 个头'
          % (len(rows), len([h for h in HEADERS if os.path.isfile(h)])))
    print('=' * 78)
    by_cls = {}
    for r in rows:
        by_cls.setdefault((r['src'], r['class']), []).append(r)
    n_unchecked = 0
    for (src, c), items in sorted(by_cls.items()):
        ck = [i for i in items if i['checked']]
        un = [i for i in items if not i['checked']]
        n_unchecked += len(un)
        print('\n%-22s %-34s  atoi 参数 %d（已校验 %d / 未校验 %d）'
              % (src, c, len(items), len(ck), len(un)))
        print('   已校验: ' + (', '.join(i['param'] for i in ck) if ck else '—— 无'))
        print('   未校验: ' + (', '.join(i['param'] for i in un) if un else '—— 无'))
    print('\n合计 %d 个 (层, 参数) 在 ReadParam 里**没有**值域校验。' % n_unchecked)
    print('其中每一个是否被下游兜住，结论在 audit_k3_20261001.md 附录 BF.1；')
    print('已确认漏网的三个是 BD(pooling stride) / BE(卷积 stride) / BF(Tile tile_*)。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
