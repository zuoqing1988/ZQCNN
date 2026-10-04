#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`manualExportCaffe/` 的结构性门禁（附录 GU）。

为什么这道门禁长这样，而不是"语法检查"
--------------------------------------
本机**没有 MATLAB，也没有 Octave**（2026-10-04 实测），仓库里也没有 caffe。
而这 8 个脚本的产物（`.nchwbin`）**一个都不在 `model/` 里** ——
所以它们既不能被动态验证，也没有随仓的对应物可比。

于是这道门禁只能是一个**下限**，不是语法检查。三条判据：

  1. **块结构配平**：`function/if/for/while/switch/try/parfor` 与 `end` 配平，
     且中途不为负；
  2. **`layers` 表能解析**，且每行是 `'名字', 类型码, 'flag'` 的形状；
  3. **用到的 flag 都被同一个文件的 `strcmp(flag,...)` 处理过** ——
     没处理的 flag 会让那一层**按 none 的形状写出去**，权重静默错位。

**它明确不查的**（写下来是为了不让它被当成"MATLAB 脚本验过了"）：
类型码是否与 C++ 侧 `ZQ_CNN_Layer` 的枚举对得上、权重排列是否与
`LoadBinary_NCHW` 期待的字节数一致、caffe API 是否还存在。
那几条要真跑 MATLAB 才谈得上。

一条必须先说的实现纪律
----------------------
**注释行必须先剥掉。** 第一版直接对整行做正则，于是
`export_spherefacenet06bn_...m` 第 5 行的
`%out_file = 'sphereface04bn256.nchwbin';`（**注释掉的**）
被当成了活跃代码，报出"两个脚本写同一个输出名"的假冲突。
这与 C++ 侧"用编译器而不是正则"是同一条道理：注释里的东西不是代码。

用法:
    python tools/check_export_scripts.py
    python tools/check_export_scripts.py --selftest
"""
import io
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DIR = os.path.join(ROOT, 'manualExportCaffe')

# 块开启关键字。`end` 单独收。
OPEN_RE = re.compile(r'^\s*(function|if|for|while|switch|try|parfor)\b')
END_RE = re.compile(r'^\s*end\b')
LAYERS_RE = re.compile(r'layers\s*=\s*\{(.*?)\n\s*\}\s*;', re.S)
ROW_RE = re.compile(r"'([^']+)'\s*,\s*(\d+)\s*,\s*'([^']+)'")
FLAG_BRANCH_RE = re.compile(r"strcmp\(\s*flag\s*,\s*'([^']+)'\s*\)")


def strip_comment(line):
    """MATLAB 的 `%` 是行注释。整行是注释 -> 返回空串。

    只做**整行**判断，不做行内截断：MATLAB 里 `%` 也可能出现在
    字符串里（`sprintf('%d',x)`），行内截断会把字符串截断。
    宁可漏掉行内注释，也不要把字符串改坏 ——
    截断出来的假代码会让"配平"凭空多出/少掉 `end`。
    """
    return '' if line.lstrip().startswith('%') else line


def _rel(path):
    """相对仓库根的路径；**跨盘时退回 basename**。

    阳性对照把脚本复制到 `%TEMP%`（本机是 C:）而仓库在 D:，
    `os.path.relpath` 在跨盘时会抛 `ValueError: path is on mount 'C:',
    start on mount 'D:'` —— 对照自己先崩了，门禁看上去"没抓到"。
    """
    try:
        return os.path.relpath(path, ROOT).replace('\\', '/')
    except ValueError:
        return os.path.basename(path)


def check_file(path):
    """返回问题列表。"""
    rel = _rel(path)
    probs = []
    try:
        text = io.open(path, encoding='utf-8', errors='replace').read()
    except (IOError, OSError) as e:
        return ['%s：读不了 %s' % (rel, e)]
    lines = [strip_comment(l) for l in text.split('\n')]

    # 1) 块配平
    depth = 0
    first_neg = None
    for i, ln in enumerate(lines, 1):
        if OPEN_RE.match(ln):
            depth += 1
        elif END_RE.match(ln):
            depth -= 1
            if depth < 0 and first_neg is None:
                first_neg = i
    if first_neg is not None:
        probs.append('%s：第 %d 行处 end 多出来（深度变负）' % (rel, first_neg))
    if depth != 0:
        probs.append('%s：块没配平，末尾 depth=%d（多了 %d 个未闭合的块）'
                     % (rel, depth, depth))

    # 2) layers 表
    m = LAYERS_RE.search(text)
    if not m:
        probs.append('%s：解析不出 layers 表' % rel)
        return probs
    rows = ROW_RE.findall(m.group(1))
    if not rows:
        probs.append('%s：layers 表解析出 0 行' % rel)
        return probs
    # 表体里出现 ROW_RE 匹配不到的实义行 = 有写坏的行
    body_lines = [l for l in m.group(1).split('\n')
                  if strip_comment(l).strip() and not strip_comment(l).strip().startswith('%')]
    stray = [l.strip() for l in body_lines if not ROW_RE.search(l)]
    if stray:
        probs.append('%s：layers 表里有 %d 行不是 '
                     "'名字', 类型码, 'flag' 的形状：%r" % (rel, len(stray), stray[:2]))

    # 3) flag 是否都被处理
    handled = set(FLAG_BRANCH_RE.findall(text))
    used = set(r[2] for r in rows)
    missing = sorted(used - handled)
    if missing:
        probs.append('%s：用到的 flag %s 没有被 strcmp(flag,...) 处理 —— '
                     '那一层会按 none 的形状写出去' % (rel, missing))
    return probs


def selftest():
    """阳性对照：造三份坏脚本，每份都必须被抓到。

    造法是**复制真实脚本再改一处**，不是手写 —— 手写的样本
    很可能连"坏在哪"都和真实情况不一样（GT 那次就是这么栽的：
    对照依赖了被修掉的文件名，改完缺陷对照自己就失效了）。
    """
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix='zqexp_')
    srcs = [os.path.join(DIR, n) for n in sorted(os.listdir(DIR))
            if n.endswith('.m')]
    if not srcs:
        return False, 'manualExportCaffe/ 下没有 .m'
    # 底本要**同时含**三个变异各自需要的锚点，否则 `replace` 静默不改，
    # 对照就变成"什么都没测却显示通过"。下面每个变异都断言自己生效了。
    base = None
    for p in srcs:
        t = io.open(p, encoding='utf-8', errors='replace').read()
        if "'conv1_1',2,'none';" in t and "'fc7x6'" in t:
            base = p
            break
    if base is None:
        return False, ('没有一个脚本同时含锚点 '
                       "'conv1_1',2,'none'; 与 'fc7x6' —— 对照无从下手")
    text = io.open(base, encoding='utf-8', errors='replace').read()

    def mutate(title, fn):
        body = fn(text)
        if body == text:
            raise AssertionError('变异 %r 没有改变任何内容 —— 对照会变成'
                                 '一个永远为真的断言' % title)
        return (title, body)

    cases = [
        # A 少一个 end
        mutate('少一个 end', lambda t: t.replace('\nend\n', '\n', 1)),
        # B layers 表里放一行坏行
        mutate('layers 表有坏行',
               lambda t: re.sub(r"('conv1_1',2,'none';)", r"\1\n    'oops',;",
                                t, count=1)),
        # C 用一个没被处理的 flag
        mutate('flag 未被处理',
               lambda t: t.replace("'fc7x6'", "'no_such_flag'", 1)),
    ]

    names = []
    caught = set()
    # 底本自己**必须是通过的**，否则"三种都没抓到"可能只是因为对照起点就是坏的
    if check_file(base):
        return False, '底本 %s 自己就没通过，对照起点是坏的' % _rel(base)
    try:
        for i, (title, body) in enumerate(cases):
            p = os.path.join(tmp, 'case%d.m' % i)
            with io.open(p, 'w', encoding='utf-8', newline='\n') as f:
                f.write(body)
            names.append(title)
            if check_file(p):
                caught.add(title)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    missed = [t for t in names if t not in caught]
    if missed:
        return False, '这几种破坏没被抓到：%s' % missed
    return True, ''


def main():
    if '--selftest' in sys.argv:
        ok, why = selftest()
        print('阳性对照：三种人为破坏都被抓到 =', ok)
        if not ok:
            print(why)
            return 2

    if not os.path.isdir(DIR):
        print('manualExportCaffe/ 不存在 —— 门禁失效（不是"通过"）')
        return 2
    files = sorted(f for f in os.listdir(DIR) if f.endswith('.m'))
    if not files:
        print('manualExportCaffe/ 下没有 .m —— 门禁失效（不是"通过"）')
        return 2

    problems = []
    for f in files:
        problems += check_file(os.path.join(DIR, f))
    print('查了 %d 个 .m 导出脚本' % len(files))
    print('  判据：块配平 / layers 表形状 / 用到的 flag 都被处理')
    print('  **不查**：类型码与 C++ 枚举是否对得上、权重字节数是否与 '
          'LoadBinary_NCHW 一致、caffe API 是否还在 —— 那几条要真跑 MATLAB。')
    for p in problems:
        print('  ' + p)
    if problems:
        return 1
    print('全部通过。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
