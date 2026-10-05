#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""算出「ZQCNN 的 36 种层类型里，哪些真的被随仓库发布的模型跑到」。

为什么要有这个工具（附录 DB.1）
------------------------------
2026-10-02 我在附录 CX 里断言「`deconvolution` **仓内零调用方**」，
并据此把一整条「层 → wrapper → 内核」的链路判成死代码。
**那个断言是错的。**

错因是一次 **grep 大小写不匹配的静默失败**：命令是

    grep -rn "deconvolution\\|deconv" --include=*.cpp --include=*.h ZQCNN/

而层类名是 `ZQ_CNN_Layer_DeConvolution` —— `DeConvolution` 的前六个字母是
`D-e-C-o-n-v`，只有**加了 `-i`** 才会匹配小写的 `deconv`。
命令返回空**且不报错**，我把"空"当成了"不存在"。

代价 ≈ 一整轮，而且差一步就漏掉两条真缺陷（附录 DA.3 / DA.4）。

本工具把那条教训变成机制
------------------------
"某个东西有没有被用到"这种判断，从此**不再靠手敲 grep**：
跑一次本工具，拿到的是一张**可复算**的表，而且：

* 匹配**强制忽略大小写**（DA.2 的直接教训，写在输出里提醒）；
* 区分**三种状态**，而不只是"有/没有"：
    - `EXERCISED`  —— 至少一个 `.zqparams` 里有一行**未被注释**地用了它
    - `COMMENTED`  —— 只在被 `#` 注掉的行里出现过（**看着在用、其实禁用**）
    - `UNUSED`     —— 一次都没出现
  第三种区分是手敲 grep 给不出来的，而它恰恰是最容易误导人的：
  一堆 `#Convolution` 会让"grep 到了"和"跑过了"看起来一样。
* `--check-baseline` 把它变成常驻门禁：新增/删除一个模型、或某层从
  `EXERCISED` 变成 `UNUSED`，都会变成一条可见的 diff。

**它不能回答的问题**（如实写清楚，别过度解读这张表）：
它只说"模型文件里写没写"，**不说**"跑起来对不对"、也不说"跑没跑到那条分支"。
一个 `Convolution` 被 27 个模型用到，不代表它的每种 stride/dilation/pad 组合
都被跑到 —— 那是门禁和样例回归各自负责的事。
这张表的用途只有一个：**别再把"没搜到"当成"不存在"。**

用法
----
    python tools/reachability_probe.py                      # 列出全部
    python tools/reachability_probe.py --unused            # 只看没被用到的
    python tools/reachability_probe.py --save-baseline  tools/reachability_baseline.txt
    python tools/reachability_probe.py --check-baseline tools/reachability_baseline.txt
"""

from __future__ import print_function

import argparse
import glob
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET_H = os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_Net.h')
MODEL_DIR = os.path.join(ROOT, 'model')
DEFAULT_BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'reachability_baseline.txt')

# 层类型注册处：_my_strcmpi(&buf[0], "Convolution") == 0
# _my_strcmpi 本身就是大小写不敏感的比较，所以注册表是大小写不敏感的；
# 模型里的类型名同样按大小写不敏感来找 —— 见上面 DA.2 的教训。
REG_RE = re.compile(r'_my_strcmpi\(\s*&buf\[0\]\s*,\s*"([^"]+)"\s*\)')
# .zqparams 的一行：第一段空白是层类型；# 开头是注释掉的
LAYER_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\b')

EXERCISED = 'EXERCISED'
PROBED = 'PROBED'        # 没有任何模型用它，但被 SampleUnusedLayerProbe 造网跑过
COMMENTED = 'COMMENTED'
UNUSED = 'UNUSED'

# 「被探针覆盖」的层类型从**探针源码里推出来**，而不是手写一张名单 ——
# 手写名单与探针内容不一致时，差异部分永远是**假覆盖**（附录 II.1）。
#
# 探针写的合成 `.zqparams` 行一律是 `"<层类型> name=..."` 的形状，
# 所以一条正则就够了：抓到的是"探针**真的在构造**哪些层"。
PROBE_SRC = os.path.join(ROOT, 'SamplesZQCNN', 'SampleUnusedLayerProbe',
                         'SampleUnusedLayerProbe.cpp')
PROBE_LAYER_RE = re.compile(r'"([A-Z][A-Za-z0-9_]*)\s+name=')


def probe_covered(path=PROBE_SRC):
    """探针 sample 覆盖的层类型集合（小写）。文件不在就返回 None。"""
    if not os.path.isfile(path):
        return None
    with io.open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    return set(m.group(1).lower() for m in PROBE_LAYER_RE.finditer(text))


def read_net_types(path):
    with io.open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    out = []
    for m in REG_RE.finditer(text):
        name = m.group(1)
        if name not in out:
            out.append(name)
    return out


def scan_models(model_dir, types):
    """返回 {小写类型: {'EXERCISED': [(模型, 行号)], 'COMMENTED': [...]}}"""
    lower2name = dict((t.lower(), t) for t in types)
    hits = dict((t, {EXERCISED: [], COMMENTED: []}) for t in types)
    files = sorted(glob.glob(os.path.join(model_dir, '*.zqparams')))
    for path in files:
        model = os.path.basename(path)
        with io.open(path, encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f, 1):
                if line.lstrip().startswith('#'):
                    # 注掉的行：**只看它开头那个词**是不是某个层类型
                    body = line.lstrip()[1:]
                    m = LAYER_RE.match(body)
                    key = m.group(1).lower() if m else None
                    if key in lower2name:
                        hits[lower2name[key]][COMMENTED].append((model, i))
                    continue
                m = LAYER_RE.match(line)
                if not m:
                    continue
                key = m.group(1).lower()
                if key in lower2name:
                    hits[lower2name[key]][EXERCISED].append((model, i))
    return hits, files


def status_of(h, probed=False):
    if h[EXERCISED]:
        return EXERCISED
    if h[COMMENTED]:
        return COMMENTED
    if probed:
        return PROBED
    return UNUSED


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument('--unused', action='store_true',
                    help='只列出 EXERCISED/COMMENTED 都为空的层类型')
    ap.add_argument('--save-baseline')
    ap.add_argument('--check-baseline')
    args = ap.parse_args()

    if not os.path.isfile(NET_H):
        print('找不到 %s' % NET_H)
        return 1
    types = read_net_types(NET_H)
    if not types:
        print('从 %s 里没解析出任何层类型 —— 正则失效了，不要当成"没有层"' % NET_H)
        return 1
    hits, files = scan_models(MODEL_DIR, types)
    probed = probe_covered()
    if probed is None:
        print('**注意**：找不到 %s —— 本次**没有** PROBED 这一档，'
              '下面的 UNUSED 里会混入"其实已被探针覆盖"的层。' % PROBE_SRC)
        probed = set()

    print('ZQCNN 层类型可达性（附录 DB）')
    print('  注册表：%s，共 %d 种' % (os.path.relpath(NET_H, ROOT).replace('\\', '/'), len(types)))
    print('  扫描：model/*.zqparams，共 %d 个' % len(files))
    print('  **匹配一律忽略大小写** —— 2026-10-02 那次「零调用方」的错判就是漏了 -i')
    print('    （见附录 DA.2），所以这条规则写死在这里。\n')

    n_ex = n_pr = n_cm = n_un = 0
    rows = []
    for t in types:
        h = hits[t]
        is_probed = (t.lower() in probed)
        st = status_of(h, is_probed)
        n_live = len(h[EXERCISED])
        n_cmt = len(h[COMMENTED])
        if st == EXERCISED:
            n_ex += 1
        elif st == PROBED:
            n_pr += 1
        elif st == COMMENTED:
            n_cm += 1
        else:
            n_un += 1
        where = ''
        if n_live:
            models = sorted(set(m for m, _ in h[EXERCISED]))
            where = '%d 个模型（首个 %s:%d）' % (len(models), models[0], h[EXERCISED][0][1])
        elif st == PROBED:
            where = '**没有任何模型用它**，但 `SampleUnusedLayerProbe` 造合成网真跑过'
        elif n_cmt:
            where = '只出现在**被注掉的行**里（首个 %s:%d）' % (
                h[COMMENTED][0][0], h[COMMENTED][0][1])
        rows.append((t, st, n_live, n_cmt, where))
        if args.unused and st == EXERCISED:
            continue
        print('  %-8s %-22s %s' % (st, t, where))

    print('\n合计 %d 种：EXERCISED %d / PROBED %d / COMMENTED %d / UNUSED %d'
          % (len(types), n_ex, n_pr, n_cm, n_un))
    if n_pr:
        print('PROBED 这 %d 种**没有任何模型会跑到**，但 `SampleUnusedLayerProbe`'
              '给它们造了合成网并与独立参考实现对拍（附录 IA~IH）。' % n_pr)
    if n_un:
        print('UNUSED 与 COMMENTED 这两类的代码路径**不会被任何随仓库发布的模型跑到**，')
        print('  也没有探针覆盖 —— 它们才是真正的零覆盖区。')
        print('  写可达性结论时**不许**拿"模型里没搜到"当证据，要拿本工具的表。')

    if args.save_baseline:
        lines = ['# ZQCNN 层类型可达性基线（tools/reachability_probe.py --save-baseline 生成）',
                 '# 格式: <层类型>\t<状态>\t<未注释命中数>\t<被注释命中数>',
                 '#',
                 '# 状态：EXERCISED（至少一个模型有未注释的层用它）/',
                 '#       COMMENTED（只出现在 # 注掉的行里 —— 看着在用其实禁用）/',
                 '#       PROBED（没有模型用它，但 SampleUnusedLayerProbe 造网跑过）/',
                 '#       UNUSED（一次都没出现，且探针也没覆盖）',
                 '#',
                 '# 基线的作用是让「新增/删除一个模型」「某个层从 EXERCISED 变成 UNUSED」',
                 '# 变成一条可见的 diff，而不是又一次靠记忆的判断。']
        for t, st, nl, nc, _w in rows:
            lines.append('%s\t%s\t%d\t%d' % (t, st, nl, nc))
        with io.open(args.save_baseline, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(lines) + '\n')
        print('\nbaseline written to %s' % args.save_baseline)

    if args.check_baseline:
        base = {}
        with io.open(args.check_baseline, encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                p = line.rstrip('\n').split('\t')
                if len(p) >= 4:
                    base[p[0]] = (p[1], p[2], p[3])
        cur = dict((t, (st, str(nl), str(nc))) for t, st, nl, nc, _w in rows)
        new_t = [t for t in cur if t not in base]
        gone_t = [t for t in base if t not in cur]
        changed = [(t, base[t], cur[t]) for t in cur
                   if t in base and base[t][0] != cur[t][0]]
        count_changed = [(t, base[t][1], cur[t][1]) for t in cur
                         if t in base and base[t][1] != cur[t][1]]
        print('\n=== 与基线 %s 比对 ===' % args.check_baseline)
        print('基线 %d 条 -> 现在 %d 条' % (len(base), len(cur)))
        if new_t:
            print('NEW（基线里没有的层类型）: %s' % ', '.join(new_t))
        if gone_t:
            print('GONE（基线里有、现在解析不到了）: %s' % ', '.join(gone_t))
        if changed:
            print('状态变了:')
            for t, b, c in changed:
                print('   %-24s %s -> %s' % (t, b[0], c[0]))
        if count_changed:
            print('命中数变了（正常：增删了模型）: %d 处' % len(count_changed))
        if not (new_t or gone_t or changed):
            print('无状态变化。')
        return 1 if (new_t or gone_t or changed) else 0

    return 0


if __name__ == '__main__':
    sys.exit(main())
