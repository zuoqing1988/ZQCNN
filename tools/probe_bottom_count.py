#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Which ZQ_CNN_Layer classes read bottoms[k] for k >= 1 without a size() guard?

The idiom used all over ZQ_CNN_Layer.h is

    if (bottoms == 0 || ... || bottoms->size() == 0 || ... || (*bottoms)[0] == 0)
        return false;

-- it rejects an *empty* bottoms, then goes on to read (*bottoms)[1].
For a layer that needs two bottoms, a bottoms of size 1 is already one past
the end, and the `(*bottoms)[1] == 0` test is itself the out-of-bounds read.

This lists every class that indexes bottoms[k], k >= 1, together with the
smallest bottoms->size() that the layer is known to tolerate, so any gap is
explicit.

**First version got this wrong.**  It only recognised `bottoms->size() >= N`
as a guard, so it reported "no guard" for all four classes that index
bottoms[>=1] -- including `ZQ_CNN_Layer_PriorBox`, which has had
`bottoms->size() < 2` in place since commit 3c6bb9c.  A detector that cries
wolf on correct code is worse than no detector, so the predicate below
handles the forms actually used here and the file carries a mutation test
(see tools/check_bottom_guards.py) that removes a real guard and requires
this script to notice.

Diagnostic only, writes nothing.
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "ZQCNN", "ZQ_CNN_Layer.h")

CLASS_RE = re.compile(r"^\tclass (ZQ_CNN_Layer_\w+)")
IDX_RE = re.compile(r"\(\*bottoms\)\[(\d+)\]")

# the four forms actually used in this file, and what each one guarantees
FORMS = [
    (re.compile(r"bottoms->size\(\)\s*>=\s*(\d+)"),      "ge"),
    (re.compile(r"bottoms->size\(\)\s*>\s*(\d+)"),       "gt"),
    (re.compile(r"bottoms->size\(\)\s*<\s*(\d+)"),       "lt"),
    (re.compile(r"bottoms->size\(\)\s*<=\s*(\d+)"),      "le"),
    (re.compile(r"bottoms->size\(\)\s*!=\s*(\d+)"),      "ne"),
    (re.compile(r"bottoms->size\(\)\s*==\s*(\d+)"),      "eq"),
]

# How many bottoms each form lets through.  The guard aborts when the condition
# holds, so we *proceed* with the sizes on the other side:
#   size() >= k  -> proceed with k or more      -> protects k bottoms
#   size() >  k   -> proceed with k+1 or more    -> protects k+1
#   size() <  k   -> abort below k              -> proceed with k or more -> k
#   size() <= k   -> abort up to and incl. k    -> proceed with k+1 or more -> k+1
#   size() != k   -> proceed only when size==k  -> protects k, and only if we
#                                                  need exactly k
#   size() == k   -> same
def tolerated(form, k):
    if form in ("ge", "lt"):
        return k
    if form in ("gt", "le"):
        return k + 1
    if form in ("ne", "eq"):
        return k
    return 0


def main():
    lines = io.open(SRC, encoding="utf-8", newline="").read().split("\n")
    cur, cur_start, out = None, 0, []
    for i, l in enumerate(lines):
        m = CLASS_RE.match(l)
        if m:
            if cur:
                out.append((cur, cur_start, i))
            cur, cur_start = m.group(1), i + 1
    if cur:
        out.append((cur, cur_start, len(lines)))

    rows = []
    for name, a, b in out:
        body = lines[a:b]
        maxidx = -1
        for l in body:
            for mm in IDX_RE.finditer(l):
                maxidx = max(maxidx, int(mm.group(1)))
        if maxidx < 1:
            continue
        need = maxidx + 1
        # the largest bottoms->size() the layer provably rejects nothing above
        safe = 0          # how many bottoms we are sure are handled
        for l in body:
            for rx, form in FORMS:
                for mm in rx.finditer(l):
                    k = int(mm.group(1))
                    if form == "eq" and k != need:
                        # `size()==0` only rejects the empty case
                        continue
                    safe = max(safe, tolerated(form, k))
        rows.append((name, need, safe, safe >= need, b - a))

    print("ZQCNN/ZQ_CNN_Layer.h: %d layer classes, %d of them index bottoms[>=1]\n"
          % (len(out), len(rows)))
    print("%-40s %-10s %-14s %s" % ("class", "needs", "provably safe", "verdict"))
    for name, need, safe, ok, n in sorted(rows, key=lambda r: (r[3], -r[1], r[0])):
        print("%-40s %-10d %-14s %s"
              % (name, need, ("up to %d" % safe) if safe else "none",
                 "guarded" if ok else "**NO GUARD**"))
    gap = [r for r in rows if not r[3]]
    print("\nclasses without a sufficient guard: %d" % len(gap))
    return 1 if gap else 0


if __name__ == "__main__":
    sys.exit(main())
