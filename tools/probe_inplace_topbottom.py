#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Probe: how many real .zqparams models would be rejected if the
_check_connect() in-place guard were strengthened from "tops[i][j]==bottoms[i][j]"
(same index) to "any top name of the layer also appears in its bottom list".

Writes no files, touches no production code. Diagnostic only.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# must stay identical to ZQ_CNN_Net::_is_inplace_safe()
INPLACE_SAFE = {
    "relu", "relu6", "prelu",
    "batchnormscale", "batchnorm", "scale", "addbias",
}

NAME_RE = re.compile(r"\bname\s*=\s*(\S+)")
TOP_RE = re.compile(r"\btop\s*=\s*(\S+)")
BOTTOM_RE = re.compile(r"\bbottom\s*=\s*(\S+)")


def parse_layer(line):
    parts = line.replace("\t", " ").split()
    if not parts:
        return None
    ltype = parts[0].strip()
    m = NAME_RE.search(line)
    name = m.group(1) if m else "?"
    tops = TOP_RE.findall(line)
    bottoms = BOTTOM_RE.findall(line)
    return ltype, name, tops, bottoms


def main():
    files = []
    for base, _dirs, names in os.walk(ROOT):
        if os.sep + ".git" in base:
            continue
        for n in names:
            if n.endswith(".zqparams"):
                files.append(os.path.join(base, n))
    files.sort()

    same_index_hits = []
    any_index_hits = []
    total_layers = 0
    layer_types = {}

    for path in files:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for lineno, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                p = parse_layer(line)
                if p is None:
                    continue
                ltype, name, tops, bottoms = p
                if ltype == "Input":
                    continue
                total_layers += 1
                layer_types[ltype.lower()] = layer_types.get(ltype.lower(), 0) + 1
                if ltype.lower() in INPLACE_SAFE:
                    continue
                n = min(len(tops), len(bottoms))
                for j in range(n):
                    if tops[j] == bottoms[j]:
                        same_index_hits.append(
                            (os.path.relpath(path, ROOT), lineno, ltype, name, tops[j]))
                bset = set(bottoms)
                for j, t in enumerate(tops):
                    if t in bset:
                        any_index_hits.append(
                            (os.path.relpath(path, ROOT), lineno, ltype, name, t, bottoms))

    print("scanned %d .zqparams, %d non-Input layers" % (len(files), total_layers))
    print()
    print("current guard (tops[i][j] == bottoms[i][j], same index): %d hit(s)" % len(same_index_hits))
    for h in same_index_hits:
        print("   %s:%d  %s '%s' top==bottom '%s'" % (h[0], h[1], h[2], h[3], h[4]))
    print()
    print("strengthened guard (any top name also in the same layer's bottoms): %d hit(s)"
          % len(any_index_hits))
    for h in any_index_hits:
        print("   %s:%d  %s '%s' top '%s' bottoms %s" % (h[0], h[1], h[2], h[3], h[4], h[5]))
    print()
    print("non-inplace-safe layer types seen:")
    for t, c in sorted(layer_types.items(), key=lambda kv: -kv[1]):
        mark = "  (inplace-safe)" if t in INPLACE_SAFE else ""
        print("   %-28s %5d%s" % (t, c, mark))
    return 0


if __name__ == "__main__":
    sys.exit(main())
