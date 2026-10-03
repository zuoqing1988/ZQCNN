#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model-file values used as divisors or shift counts in ZQ_CNN_Layer.h.

Two families, both reachable from .zqparams:

  1. `x / stride_W` with stride_W == 0  -> integer division by zero
     (x86 idiv traps: SIGFPE, the process dies).  Appendix EM already found
     seven convolution wrappers for this and added guards; this sweep asks
     whether any *other* divisor is unguarded.
  2. `1 << N` with N >= 32 (or N < 0)   -> undefined behaviour.

**This probe ships with a positive control**, because appendix EQ recorded
twice that a "0 hits" sweep is worthless until it has been shown to hit.  The
control is the `stride_W == 0` guard EM added: if the sweep cannot see that the
divisor is checked, the sweep is broken, not the code.

Three false-positive families were found while building it, all worth
recording because each one made the answer invisible:

  - `<<` is both "bit shift" and "stream insertion".  Every one of the 37
    apparent shifts in this file is `std::cout << value` inside an error
    message.  Deciding per *token* ("is the thing before `<<` a stream?") does
    not work either -- in a chain the `<<` before `kernel_H` is preceded by a
    string literal.
  - the fix for that is to judge per *statement*, but a statement here spans
    several lines and the stream name sits on the **first** one:
        std::cout << "Layer " << name << " conv kernel/dilate overflow: kernel "
            << kernel_H << "x" << kernel_W << ...
    So the statement is accumulated up to its terminating ';'.
  - a `stride_W == 0` guard in *some other* layer does not protect this one, so
    the guard lookup is scoped to the same class.  That is still weaker than
    "same function", which would need real parsing; the limitation is stated
    rather than hidden.

Run:
    python tools/probe_div_shift.py             # sweep only
    python tools/probe_div_shift.py --selftest  # sweep + built-in positive control
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "ZQCNN", "ZQ_CNN_Layer.h")

MODEL_VARS = (
    r"\b(stride_[HW]|pool_stride_[HW]|dilate_[HW]|dilation_[HW]|kernel_[HW]|"
    r"group|num_output|axis|hidden_size|pool_size_[HW])\b")
DIV_RE = re.compile(r"/\s*(" + MODEL_VARS + r")")
SHIFT_RE = re.compile(r"<<\s*(" + MODEL_VARS + r")")

STREAM_RE = re.compile(r"\b(?:std::)?(?:cout|cerr|clog)\b")
FUNC_RE = re.compile(r"^\s*(virtual\s+)?(bool|void|int|float)\s+(\w+)\s*\(")
CLASS_RE = re.compile(r"^\tclass (\w+)")


def read_lines():
    return io.open(SRC, encoding="utf-8", newline="").read().split("\n")


def class_spans(lines):
    """[(class_name, start_index, end_index)] over the whole file."""
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


def scan():
    """Walk the file statement by statement, not line by line."""
    lines = read_lines()
    spans = class_spans(lines)

    def owner(line_no):
        return next((c for c, a, b in spans if a < line_no <= b), "?")

    fn = "?"
    divs, shifts = [], []
    stmt = ""            # accumulated statement text
    stmt_line = 0        # 1-based line where the statement started
    offset = 0           # chars of `stmt` that came from lines before `l`

    for idx, l in enumerate(lines):
        if not stmt:
            stmt_line = idx + 1
            offset = 0
        stmt += " " + l

        m = FUNC_RE.match(l)
        if m:
            fn = m.group(3)
        if not l.strip().startswith("//"):
            base = len(stmt) - len(l)      # position of l inside stmt
            for mm in DIV_RE.finditer(l):
                divs.append((idx + 1, owner(idx + 1), fn,
                             mm.group(1), l.strip()[:96]))
            for mm in SHIFT_RE.finditer(l):
                if STREAM_RE.search(stmt[:base + mm.start()]):
                    continue
                shifts.append((idx + 1, owner(idx + 1), fn,
                               mm.group(1), l.strip()[:96]))

        if ";" in l:
            stmt = ""
    return divs, shifts, spans


def zero_guarded(lines, spans, cls, divisor):
    """Does this class contain a `<divisor> == 0` (or <=0 / >=0) guard?"""
    rx = re.compile(re.escape(divisor) + r"\s*(?:==|<=|>=)\s*0")
    for c, a, b in spans:
        if c == cls:
            for l in lines[a:b]:
                if rx.search(l):
                    return True
    return False


def main():
    divs, shifts, spans = scan()
    lines = read_lines()
    print("ZQCNN/ZQ_CNN_Layer.h")
    print("  model-value divisors  : %d" % len(divs))
    print("  model-value shifts    : %d  (stream insertions excluded)" % len(shifts))

    by = {}
    for (line_no, cls, fn, name, text) in divs:
        by.setdefault((name, cls), []).append((line_no, fn, text))
    print("\ndivisors:")
    unguarded = []
    for (name, cls), v in sorted(by.items()):
        ok = zero_guarded(lines, spans, cls, name)
        if not ok:
            unguarded.append((name, cls))
        print("  / %-16s in %-36s x%-3d %s"
              % (name, cls, len(v), "zero-guarded" if ok else "** NO ZERO GUARD **"))
        for line_no, fn2, text in v[:2]:
            print("        %6d  %s | %s" % (line_no, fn2, text))

    if shifts:
        print("\nshifts:")
        for line_no, cls, fn, name, text in shifts:
            print("  %6d %s::%s | %s" % (line_no, cls, fn, text))
    else:
        print("\nshifts: none (every apparent shift was a stream insertion)")

    if "--selftest" in sys.argv:
        print("\n=== positive control ===")
        if not divs:
            print("  sweep found no divisors at all -> the sweep is broken, not the code")
            return 1
        if unguarded:
            print("  control FAILED for: %s" % ", ".join("%s(%s)" % u for u in unguarded))
            return 1
        print("  control passed: %d divisor/class pairs, all zero-guarded" % len(by))
    return 0


if __name__ == "__main__":
    sys.exit(main())
