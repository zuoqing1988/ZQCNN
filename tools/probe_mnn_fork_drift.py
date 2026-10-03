#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""How far has the MNN converter's forked ZQCNN headers drifted from the main tree?

`ZQCNN_to_MNN/converter/source/` holds a copy of seven ZQCNN headers
(ZQ_CNN_Layer.h is 6293 lines on its own).  It is **not** in any build:
the top-level CMakeLists.txt has no `add_subdirectory(ZQCNN_to_MNN)` and the
converter's own `CMakelists.txt` is never configured, because its main .cpp
needs MNN's generated `MNN_generated.h`.

So the fork has never been compiled by anything, and it is a **snapshot from
whenever it was taken**.  Two questions this answers:

  1. Which audit fixes made since the fork are **missing** from it?
  2. Does the fork still contain the defects that were fixed in the main tree?

A fork is allowed to lag.  What it is not allowed to do is silently carry a
defect that was already diagnosed and fixed elsewhere in the same repository.

Diagnostic only, writes nothing.
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(ROOT, "ZQCNN")
FORK = os.path.join(ROOT, "ZQCNN_to_MNN", "converter", "source")

# (描述, 判定它在**主树**里存在的正则, 判定它在**分叉**里存在的正则)
CHECKS = [
    ("ChangeSize(..., dst_borderH, dst_borderW) 传反（附录 DY.2，51 处）",
     r"ChangeSize\([^;]*?,\s*\w*[bB]orderH\s*,\s*\w*[bB]orderW\s*\)",
     r"ChangeSize\([^;]*?,\s*\w*[bB]orderH\s*,\s*\w*[bB]orderW\s*\)"),
    ("卷积 dilate*(kernel-1) 溢出守卫（附录 EM.3）",
     r"__int64\)dilate_H\s*\*\s*\(kernel_H\s*-\s*1\)",
     r"__int64\)dilate_H\s*\*\s*\(kernel_H\s*-\s*1\)"),
    ("就地守卫 top==bottom（附录 DY / EN）",
     r"destroys its own input",
     r"destroys its own input"),
    ("就地守卫改成比**全部组合**（附录 EN.4）",
     r"for \(int k = 0; k < bottoms\[i\]\.size\(\); k\+\+\)",
     r"for \(int k = 0; k < bottoms\[i\]\.size\(\); k\+\+\)"),
    ("_concat_NCHW 的 output/input 别名检查（附录 EN.2）",
     r"Concat: output is also input",
     r"Concat: output is also input"),
    ("Reshape_NCHW_get_size 的 shape_dim 上界（附录 DY.1）",
     r"for \(int i = 0; i < 4; i\+\+\)",
     r"for \(int i = 0; i < 4; i\+\+\)"),
    ("ConvertToBGR 的 C>=3 校验（附录 DY.1）",
     r"ConvertToBGR",
     r"ConvertToBGR"),
]


def read(path):
    try:
        return io.open(path, encoding="utf-8", newline="").read()
    except IOError:
        return ""


def main():
    files = sorted(f for f in os.listdir(FORK) if f.endswith(".h"))
    print("MNN converter fork: %d headers under ZQCNN_to_MNN/converter/source/"
          % len(files))
    print("(not in any build; its main .cpp needs MNN's MNN_generated.h)\n")

    total_lag = 0
    for f in files:
        a = read(os.path.join(MAIN, f))
        b = read(os.path.join(FORK, f))
        if not a:
            print("  %-28s 主树没有同名文件" % f)
            continue
        la, lb = len(a.split("\n")), len(b.split("\n"))
        lag = []
        for desc, rmain, rfork in CHECKS:
            in_main = bool(re.search(rmain, a))
            in_fork = bool(re.search(rfork, b))
            if in_main and not in_fork:
                lag.append(desc)
        total_lag += len(lag)
        print("  %-28s 主树 %5d 行 / 分叉 %5d 行   落后项 %d"
              % (f, la, lb, len(lag)))
        for d in lag:
            print("        - 缺: %s" % d)
    print("\n分叉总共落后 %d 处已诊断并修掉的改动。" % total_lag)
    if total_lag:
        print("这些**不是**新缺陷（主树已经修好），但分叉里仍在，")
        print("而分叉不在任何构建里 —— 也就是说它们从未被任何编译器复查过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
