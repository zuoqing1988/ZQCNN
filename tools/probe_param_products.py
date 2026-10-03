#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Find int products of two model-file-controlled values in ZQ_CNN_Layer.h.

Appendix EM was exactly this shape: `dilate * (kernel - 1)` is an int multiply
of two values that come straight out of .zqparams, and the product overflows long
before either operand looks suspicious.  EM fixed three ReadParam guards; this
sweeps for the rest of the family so the next one is found by a sweep rather
than by luck.

What counts as a hit
--------------------
  - inside a `ReadParam` or `_setup` (model-file derived values live there), or
    inside `GetTopDim` / `SetBottomDim` (where the products are consumed);
  - a `*` between two model-file values, i.e. NOT already widened with a
    `(__int64)` / `long long` cast.

The right operand may be parenthesised, because the shape EM actually has is
`dilate_H * (kernel_H - 1)`.  The first version of this pattern only accepted
a bare identifier and therefore matched **zero** of the three EM sites -- the
sweep reported "clean" on the very family it was written for.  Caught by
mutation testing this probe against the pre-fix form; a "0 hits" result from a
sweep is worth exactly nothing until the sweep has been shown to hit.

The `(__int64)` exclusion matters: EM's fix is exactly such a cast, and a
sweep that flagged its own fix would be reporting noise.

Diagnostic only, writes nothing.  Every hit still has to be read by hand --
see appendix ED.2 for what happened the last time a number was taken at face
value.

How this sweep was validated (appendix EQ)
------------------------------------------
The first version reported **0 hits**, which looked like "this family is
clean".  It was not clean, the probe was broken: with EM's own pre-fix form
(three `(__int64)` casts removed) as a positive control it *still* reported 0.
Widening the right operand to allow `(kernel_H - 1)` then made it miss the
mirror-image form `(kernel_W - 1)*dilate_W` -- so v1 missed one side and v2
missed the other, both for the same reason (regex directionality).

It now reproduces exactly EM's 16 sites, 6/6/4 per class, and finds no 17th.
A "0 hits" from a sweep is worth nothing until the sweep has been shown to
hit; run the positive control before believing it.
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "ZQCNN", "ZQ_CNN_Layer.h")

# int-ish locals/params that ReadParam fills from the .zqparams line
MODEL_VARS = re.compile(
    r"\b(kernel_[HW]|stride_[HW]|pad_[HW]|dilate_[HW]|dilation_[HW]|"
    r"pool_size_[HW]|pool_stride_[HW]|pool_pad_[HW]|global_pool|"
    r"num_output|group|axis|input_dim|hidden_size|layers)\b")

# A hit is: two model-file values separated by a `*`, with no `=`, `;`, `,`
# or comparison between them -- i.e. they are operands of the same product.
#
# Both operand orders have to be caught.  A first version used a regex shaped
# like `VAR * VAR` and it found **zero** of the three EM sites, because the
# shape EM actually has is `dilate_H * (kernel_H - 1)`.  Widening it to allow a
# parenthesised right operand then still missed `(kernel_W - 1)*dilate_W` on the
# other side.  Pairwise scanning is order-agnostic and has no such blind spot.
PAIR_WINDOW = 40


def products_in(line):
    """[(left, '*', right)] for every model-value product on this line."""
    hits = []
    positions = [(m.start(), m.end(), m.group(0))
                 for m in MODEL_VARS.finditer(line)]
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            a_s, a_e, a = positions[i]
            b_s, b_e, b = positions[j]
            if b_s - a_s > PAIR_WINDOW:
                break
            between = line[a_e:b_s]
            if "*" not in between:
                continue
            if any(c in between for c in "=;,"):
                continue
            if any(c in between for c in "<>!+/"):
                continue
            hits.append((a, "*", b, between))
    return hits

FUNC_RE = re.compile(
    r"^\s*(virtual\s+)?(bool|void|int)\s+(\w+)\s*\(")


def main():
    lines = io.open(SRC, encoding="utf-8", newline="").read().split("\n")
    fn = "?"
    cls = "?"
    hits = []
    for i, l in enumerate(lines):
        m = re.match(r"^\tclass (\w+)", l)
        if m:
            cls = m.group(1)
        m = FUNC_RE.match(l)
        if m:
            fn = m.group(3)
        interesting = fn in ("ReadParam", "_setup", "GetTopDim", "SetBottomDim",
                             "GetBottomDim", "GetTopDim_no_bottom")
        if not interesting:
            continue
        # A comment mentions the product; it does not compute it.
        if l.strip().startswith("//"):
            continue
        # already widened?  a cast immediately left of the product disqualifies
        # it.  The trailing ')' matters: EM's own guard is
        #     if ((__int64)dilate_H * (kernel_H - 1) + 1 > 0x7FFFFFFF
        # so the text left of `dilate_H` ends in ')'.
        for a, star, b, between in products_in(l):
            prod_start = l.index(a)
            if re.search(r"(__int64|long long|int64_t|size_t|double|float)\)*\s*$",
                         l[:prod_start]):
                continue
            hits.append((i + 1, cls, fn, a + " * " + b, l.strip()[:100]))

    print("ZQCNN/ZQ_CNN_Layer.h: %d unwidened model-value products in "
          "ReadParam/_setup/GetTopDim/SetBottomDim\n" % len(hits))
    by_fn = {}
    for h in hits:
        by_fn.setdefault((h[1], h[2]), []).append(h)
    for (c, f), hs in sorted(by_fn.items(), key=lambda kv: -len(kv[1])):
        print("\n%s :: %s   (%d)" % (c, f, len(hs)))
        for h in hs[:6]:
            print("   %6d  %-34s | %s" % (h[0], h[3], h[4]))
        if len(hs) > 6:
            print("   ... and %d more" % (len(hs) - 6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
