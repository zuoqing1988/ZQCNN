#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Which ReadParam guards in ZQ_CNN_Layer.h does **any** gate actually exercise?

Appendix EY was this pattern turned up by hand: the convolution `ReadParam` guard
is one `if` with six disjuncts, and I had only written cases for the
`dilate * (kernel - 1)` overflow half.  The other half — `stride == 0`, whose
consequence is an integer division by zero (SIGFPE, the process dies) — had
**no case at all**, even though it is the same line.

That is worth systematising.  For every layer class, list the numeric guards its
`ReadParam` applies to model-file values, and report which ones no `zq_*_check.cpp`
mentions.  A disjunct nobody drives is not "probably fine" -- it is untested.

Diagnostic only, writes nothing.

Run:
    python tools/probe_readparam_coverage.py
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "ZQCNN", "ZQ_CNN_Layer.h")
TOOLS = os.path.join(ROOT, "tools")

CLASS_RE = re.compile(r"^\tclass (ZQ_CNN_Layer_\w+)")
# `if (A || B || C)` spread over several lines, inside a ReadParam
GUARD_RE = re.compile(r"if\s*\(([^)]*?)\)\s*\n?\s*\{?\s*\n\s*std::cout", re.S)
OR_SPLIT = re.compile(r"\|\|")


def class_spans(lines):
    spans, cur, start = [], None, 0
    for i, l in enumerate(lines):
        m = CLASS_RE.match(l)
        if m:
            if cur:
                spans.append((cur, start, i))
            cur, start = m.group(1), i
    if cur:
        spans.append((cur, start, len(lines)))
    return spans


def readparam_bodies(lines, spans):
    """[(class, ReadParam source text)]"""
    out = []
    for name, a, b in spans:
        body = lines[a:b]
        start = None
        for i, l in enumerate(body):
            if re.search(r"virtual bool ReadParam\s*\(", l):
                start = i
                break
        if start is None:
            out.append((name, None))
            continue
        # ReadParam ends at the first line that is exactly '\t\t}' or '\t}'
        end = len(body)
        for i in range(start + 1, len(body)):
            if body[i].rstrip() in ("\t\t}", "\t}"):
                end = i
                break
        out.append((name, "\n".join(body[start:end])))
    return out


def main():
    lines = io.open(SRC, encoding="utf-8", newline="").read().split("\n")
    spans = class_spans(lines)

    # which identifiers do the gates ever mention?
    gate_text = ""
    for f in sorted(os.listdir(TOOLS)):
        if f.startswith("zq_") and f.endswith("_check.cpp"):
            gate_text += io.open(os.path.join(TOOLS, f), encoding="utf-8",
                                 newline="", errors="replace").read()
    gate_ids = set(re.findall(r"\b([a-z_][a-z0-9_]*_[a-z0-9_]*)\b", gate_text))
    # a disjunct counts as covered only if the gate text mentions the identifier
    # at all -- deliberately coarse, so "covered" can be a false positive but
    # "NOT covered" is trustworthy.
    print("ZQCNN/ZQ_CNN_Layer.h：%d 个层类，门禁源码里出现过的标识符 %d 个\n"
          % (len(spans), len(gate_ids)))

    total_guards = 0
    uncovered = 0
    for name, body in readparam_bodies(lines, spans):
        if body is None:
            continue
        flat = re.sub(r"\s+", " ", body)
        guards = []
        for m in re.finditer(r"if\s*\((.{0,400}?)\)\s*\{?\s*std::cout", flat):
            cond = m.group(1)
            if "||" not in cond:
                continue
            parts = [p.strip() for p in OR_SPLIT.split(cond) if p.strip()]
            # keep only guards about numeric comparisons
            if not all(re.search(r"[<>]=?|==", p) for p in parts):
                continue
            guards.append(parts)
        if not guards:
            continue
        total_guards += len(guards)
        miss = []
        for parts in guards:
            for p in parts:
                ids = re.findall(r"\b([a-z_][a-z0-9_]*)\b", p)
                nums = re.findall(r"\b\d+\b", p)
                if not ids:
                    continue
                # "covered" = every identifier appears somewhere in the gates
                if not all(i in gate_ids for i in ids):
                    miss.append(p)
        uncovered += len(miss)
        flag = "" if not miss else "   ** %d 个分支门禁里没提过 **" % len(miss)
        print("  %-42s 数值守卫 %d 处%s" % (name, len(guards), flag))
        for p in miss[:6]:
            print("        - %s" % p)

    print("\n共 %d 处数值守卫，其中 %d 个分支在任何门禁源码里都没出现过。"
          % (total_guards, uncovered))
    print("「没出现过」是可信的：门禁文本里根本没有那个标识符，")
    print("它就不可能被驱动到（反过来「出现过」可能只是别处提到，不构成覆盖）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
