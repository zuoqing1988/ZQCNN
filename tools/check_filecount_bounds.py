#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扫「文件读入的 int 驱动内存分配」这一族缺陷。

为什么需要它（附录 EL）
----------------------
威胁模型把**模型文件（`.nchwbin`）**列为不可信输入，而它的解析入口
`ZQ_CNN_Layer::LoadBinary_NCHW` 在 `ZQ_CNN_Layer.h` 里有 **72 个实现**、
**40 处 `fread`**，而**门禁对它们零覆盖**。

72 个实现不可能逐个写门禁，但"**读进来的 int 在驱动分配之前有没有上界**"
这个问题是可以机械回答的。附录 EF 就是用这个思路在 `ZQlibFaceID` 的
13 个同类站点上找出 1 处真缺陷（`ZQ_FaceClusterImagesForVideo` 的 `num`
只挡负数 ⇒ `resize(0x7FFFFFFF)` ⇒ 内存受限环境下 `bad_alloc` → terminate）。

判据（两条，缺一不可）
--------------------
1. 找 `fread(&X, sizeof(int), 1, ...)`（以及 `fread(&X, sizeof(int), N, ...)`）
   之后 **~20 行**内由 `X`（或其算术式）驱动的
   `resize` / `reserve` / `new T[...]` / `malloc` / `calloc`。
2. 判断 `X` 在**驱动分配之前**有没有**上界**。上界的写法有多种，
   **两种都要认**（这是 EF 那次误判的直接原因）：
      X > 1000000        X < 1000000        X >= 0 && X < 4096
      X <= N（与常量比较）  rest_len 交叉校验   min/max 夹取

   **只认 `X < 0`（负数检查）不算上界** —— 那是 EF 查出来的那处。

分类
----
  BOUNDED     有上界
  NEG_ONLY    **只挡了负数** —— 这一类就是要报的
  NO_CHECK    完全没有检查
  （带 `X` 与行列信息的，输出里会附上）

用 ``--min-bound`` 调"上界至少要多大才算有意义"（默认 1024）。
"""
import re
import sys
import os

READ_RE = re.compile(r'fread(?:_s)?\s*\(\s*&(\w+)\s*,\s*sizeof\s*\(\s*int\s*\)')
ALLOC_RES = re.compile(r'(\w+)\s*\.\s*(resize|reserve)\s*\(')
ALLOC_NEW = re.compile(r'new\s+\w+\s*\[\s*(\w+)\s*\]')
ALLOC_MAL = re.compile(r'\b(?:malloc|calloc)\s*\(\s*[^,)]*?(?:\(\s*(?:__)?int64\s*\)\s*)?(\w+)\s*\*')

# 上界的各种写法（`<` 和 `>` 都要认 —— EF 那次只认 `>`，把 `num < 1000000` 判成了没上界）。
# **模板里的 %s 必须在 classify() 里用变量填上** ——
# 第一版把 `%s` 原样留着，编译成正则去匹配字面量 "%s"，
# 于是**任何"用比较表达的上界"都识别不出来**，
      **用比较表达的上界**都识别不出来**，
UB_TEMPLATES = [
    r'\b%s\s*>\s*(\d+)',
    r'\b%s\s*<\s*(\d+)',
    r'\b%s\s*>=?\s*\d+\s*&&\s*\w+\s*[<>]=?\s*\d+',
    r'\b%s\s*[<>]=?\s*\d+\s*&&\s*\w+\s*[<>]=?\s*\d+',
    r'\b%s\s*<=?\s*(\w+)',
    r'\b%s\s*[<>]\s*\w+\s*\|\|',
    # 形如 `|| num > 1000000`（**必须带 %s**；
    # 漏掉占位符会让 `tmpl % var` 抛 'not all arguments converted'）
    r'\|\|\s*%s\s*[<>]\s*(\d+)',
    r'%s\s*>\s*0\s*&&\s*\w+\s*<\s*(\d+)',
]
NEG_ONLY = re.compile(r'\b%s\s*<\s*0|\b%s\s*>=\s*0\s*&&(?![^;]*\b%s\s*[<>])')


MIN_BOUND = 16   # 小于这个数的"上界"不算数（`num < 0` 是负数检查，不是上界）


def classify(var, ctx, min_bound=MIN_BOUND):
    """返回 (状态, 说明)。状态 ∈ BOUNDED / NEG_ONLY / NO_CHECK

    **`num < 0` / `num > 0` 不是上界。** 第一版把 `num < 0` 当成"上界 0"判成 BOUNDED，
    于是**恰好把附录 EF 修掉的那处报成 OK** ——
    谁要是把 `rest_len` 守卫删了，工具仍然报 OK，**正好掩盖它被造出来抓的那类缺陷**。
    所以：数字上界必须 >= min_bound 才算数。
    """
    # 明显的上界写法（**每次按当前变量重新编译**）
    for tmpl in UB_TEMPLATES:
        pat = tmpl % re.escape(var)
        m = re.search(pat, ctx)
        if not m:
            continue
        # 带数字捕获的：数字必须 >= min_bound，否则那只是负数/零检查
        if m.lastindex:
            try:
                if int(m.group(1)) < min_bound:
                    continue
            except (ValueError, IndexError):
                pass
        return 'BOUNDED', m.group(0).strip()
    if re.search(r'__min\s*\(\s*%s' % re.escape(var), ctx) or \
       re.search(r'__max\s*\(\s*\d+\s*,\s*%s' % re.escape(var), ctx):
        return 'BOUNDED', '__min/__max 夹取'
    # **必须是 rest_len 与本变量出现在同一行**（真参与了守卫判断），
    # 不能只匹配 rest_len 这个词 —— 第一版只匹配词，
    # 于是把「算 rest_len 的那段代码留下、只删掉 if」这种变异也判成 OK，
    # **工具的验证本身被自己骗过去了**（而那正是它要抓的那类缺陷）。
    joined_same_line = re.search(r'[^\n]*' + re.escape(var) + r'[^\n]*', ctx)
    if (joined_same_line and re.search(r'rest_len|SEEK_END|ftell', joined_same_line.group(0))) \
            or re.search(r'rest_len[^\n]*' + re.escape(var), ctx) \
            or re.search(re.escape(var) + r'[^\n]*rest_len', ctx):
        return 'BOUNDED', '用剩余文件长度交叉校验'
    if re.search(r'\b%s\s*<\s*0' % re.escape(var), ctx) or \
       re.search(r'\b%s\s*>=\s*0' % re.escape(var), ctx):
        return 'NEG_ONLY', '只挡了负数'
    return 'NO_CHECK', '完全没有检查'


def scan(path):
    try:
        text = open(path, encoding='utf-8', errors='replace').read()
    except OSError as e:
        print('读不了 %s: %s' % (path, e))
        return []
    lines = text.split('\n')
    out = []
    for i, ln in enumerate(lines):
        m = READ_RE.search(ln)
        if not m:
            continue
        var = m.group(1)
        # 该读入点所属的函数（往上找最近的形如 `bool XXX::YYY(` 或 `bool YYY(`）
        owner = '?'
        for j in range(i, max(-1, i - 60), -1):
            fm = re.search(r'\b(?:bool|void)\s+(?:ZQ::ZQ_CNN_Layer::)?(\w+)\s*\(', lines[j])
            if fm:
                owner = fm.group(1)
                break
        # **两个关注点用不同的窗口宽度**（第一版把它们混在一起，两头都错）：
        #   · 找**分配点**：窄窗口 20 行。
        #     用整个函数会跨到别的函数里去（`ZQ_FaceDatabase.h` 的 `len` 站点
        #     曾匹配到隔壁函数的 `filenames.resize`）。
        #   · 找**上界守卫**：宽窗口 60 行，且**不做函数边界检测** ——
        #     试过按 `}` 找函数末尾，结果那个 `}` 常常就是守卫**内部**代码块的，
        #     函数被截断在守卫之前，于是"已经修好的"被报成"没修"。
        #     宽窗口宁可多包含一点（假阳性让人多看一眼），也不能漏判。
        near = '\n'.join(lines[max(0, i - 3):i + 20])
        ctx = '\n'.join(lines[max(0, i - 3):i + 60])
        hits = []
        for a in ALLOC_RES.finditer(near):
            hits.append(a.group(0).strip())
        for a in ALLOC_NEW.finditer(near):
            hits.append(a.group(0).strip())
        for a in ALLOC_MAL.finditer(near):
            hits.append(a.group(0).strip())
        if not hits:
            continue
        # 分配里出现的变量必须和读入的变量相关
        relevant = [h for h in hits if var in h or re.search(r'\b%s\b' % re.escape(var), ctx.split(h)[0][-80:])]
        if not relevant:
            continue
        st, why = classify(var, ctx)
        out.append((i + 1, owner, var, st, why, relevant[0]))
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    roots = sys.argv[1:] or ['ZQCNN']
    for r in roots:
        for dp, dn, fn in os.walk(r):
            dn[:] = [d for d in dn if d not in ('.git', 'build')]
            for f in sorted(fn):
                if not f.endswith(('.h', '.cpp', '.c', '.hpp')):
                    continue
                p = os.path.join(dp, f)
                res = scan(p)
                if not res:
                    continue
                bad = [x for x in res if x[3] != 'BOUNDED']
                print('=== %s：%d 个"读入 int -> 分配"站点，其中 **%d 个缺上界** ==='
                      % (p, len(res), len(bad)))
                for (line, owner, var, st, why, alloc) in res:
                    mark = 'OK ' if st == 'BOUNDED' else st
                    print('  %-9s %-28s :%-4d %-14s %-22s %s'
                          % (mark, owner[:28], line, var, why[:22], alloc[:40]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
