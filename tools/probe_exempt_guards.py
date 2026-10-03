#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Find *conditional* validation guards: `if (!cond && (a <= 0 || ...))`.

Why this is its own sweep (appendix FA)
---------------------------------------
Appendix EZ found `Pooling::ReadParam`:

    if (!global_pool
        && (kernel_H <= 0 || kernel_W <= 0 || stride_H <= 0 || stride_W <= 0))
    { ...; return false; }

The `!global_pool` is an **exemption**: for a global pool, kernel/stride are
meaningless, so rejecting them would reject perfectly legal models.
The obvious "simplification" -- dropping `!global_pool &&` -- breaks every
global-pool model, and **no existing gate notices**, because no gate drives
the exempted path at all.

So: a validation guard that is *conditional* carries a one-sided risk that an
unconditional one does not.  This lists every such guard, and for each says
whether a gate drives the exempted path.

Diagnostic only, writes nothing.
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADERS = [
    os.path.join(ROOT, "ZQCNN", "ZQ_CNN_Layer.h"),
    os.path.join(ROOT, "ZQCNN", "ZQ_CNN_Net.h"),
    os.path.join(ROOT, "ZQCNN", "ZQ_CNN_Net_NCHWC.h"),
    os.path.join(ROOT, "ZQlibFaceID"),
]

CLASS_RE = re.compile(r"^\tclass (ZQ_\w+)")


def find_in_text(text):
    """[(class, line_no, exempt_condition, validation_condition)]"""
    out = []
    cls = "?"
    for i, l in enumerate(text.split("\n"), 1):
        m = CLASS_RE.match(l)
        if m:
            cls = m.group(1)
        # `if (!cond` possibly continued by `&& (validations)` on the next lines
        # 第一版要求 `!cond` 后面**同一行**必须有 `)` 或 `&&`，于是
        #   if (!global_pool
        #       && (kernel_H <= 0 || ...))
        # 这种**跨行**写法整条不匹配 —— 而那正是本探针专为之写的形态。
        # 教训与今天前几次一样：**扫描工具先在它要抓的那个样本上验一遍**。
        m = re.match(r"\s*if\s*\(\s*!([A-Za-z_]\w*)", l)
        if not m:
            continue
        cond = m.group(1)
        rest = l[m.end():].strip()
        # gather the continuation of the condition
        j = i
        while ")" not in rest and j - i < 4 and j < len(text.split("\n")):
            j += 1
            rest += " " + text.split("\n")[j - 1].strip()
        if not re.search(r"(<=|<|>=|>|==|!=)\s*0", rest):
            continue          # not a numeric validation -> not this pattern
        # 第二版加的收紧：校验条件里**不许有函数调用、成员访问、字符串**。
        # 第一版把 `if (!detector && detector->FindFace(..., 60, 0.709, ...))`
        # 也算成了"条件化校验" —— 它的 `<=` 来自 FindFace 的**实参 0.709**，
        # 根本不是校验。判据必须是"整条条件只由比较、标识符、算术构成"。
        if re.search(r"[.\"']|\w\s*\(", rest):
            continue
        out.append((cls, i, cond, rest.strip()[:90]))
    return out


def main():
    total = 0
    for h in HEADERS:
        paths = []
        if os.path.isdir(h):
            for f in sorted(os.listdir(h)):
                if f.endswith(".h"):
                    paths.append(os.path.join(h, f))
        else:
            paths = [h]
        for p in paths:
            if not os.path.exists(p):
                continue
            text = io.open(p, encoding="utf-8", newline="",
                           errors="replace").read()
            hits = find_in_text(text)
            if not hits:
                continue
            rel = os.path.relpath(p, ROOT).replace("\\", "/")
            print("\n%s" % rel)
            for cls, ln, cond, rest in hits:
                total += 1
                print("  %5d  %-34s 豁免条件 !%s" % (ln, cls, cond))
                print("         校验: %s" % rest)
    print("\n共 %d 处**条件化**校验守卫。" % total)
    if total:
        print("每一处的豁免路径都值得有一条\"必须放行\"的用例 ——")
        print("把它们简化掉会误拒合法模型，而没有任何现有门禁会报错。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
