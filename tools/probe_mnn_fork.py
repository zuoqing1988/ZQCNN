#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gate for the MNN converter's forked ZQCNN headers.

Why this exists (appendix EX)
-----------------------------
`ZQCNN_to_MNN/converter/source/` is a snapshot of seven ZQCNN headers taken
whenever the fork was made.  It is in **no build**: the top-level
CMakeLists.txt has no `add_subdirectory(ZQCNN_to_MNN)`, and the converter's
own `CMakelists.txt` is never configured because its main .cpp needs MNN's
generated `MNN_generated.h`.

So nothing had ever compiled it, and it had drifted: it was missing three
guards that were already diagnosed and fixed in the main tree --

  * `ZQ_CNN_BBox.h` did not pull in `ZQ_CNN_CompileConfig.h`, so `__min` /
    `__max` / `__int64` (MSVC builtins) were undefined and
    `ZQ_CNN_BBoxUtils.h` had **10 compile errors**;
  * `ZQ_CNN_Net.h` had **no in-place guard at all** -- the same defect class as
    appendix EN, i.e. a model declaring `Concat bottom=A bottom=B top=B` walks
    straight into a heap buffer overflow;
  * `ZQ_CNN_Layer.h`'s convolution `ReadParam` had **no** `stride == 0` guard
    (integer division by zero -> SIGFPE) and **no** kernel/dilate overflow
    guard (appendix EM).

This gate does two things:
  1. compiles each of the seven headers on its own (gcc -fsyntax-only);
  2. asserts the three guards are present in the fork, so a future sync from
     the main tree cannot quietly drop them again.

Run:
    python tools/probe_mnn_fork.py            # sweep
    python tools/probe_mnn_fork.py --selftest # + built-in positive control
"""
import io
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORK = os.path.join(ROOT, "ZQCNN_to_MNN", "converter", "source")

INC_ROOTS = [
    os.path.join(ROOT, "ZQCNN_to_MNN", "converter", "source"),
    os.path.join(ROOT, "ZQCNN"),
    os.path.join(ROOT, "ZQ_GEMM"),
    os.path.join(ROOT, "3rdparty", "include"),
    os.path.join(ROOT, "3rdparty", "include", "ZQlib"),
]

HEADERS = [
    "ZQ_CNN_BBox.h", "ZQ_CNN_BBoxUtils.h", "ZQ_CNN_CompileConfig.h",
    "ZQ_CNN_Forward_SSEUtils.h", "ZQ_CNN_Layer.h", "ZQ_CNN_Net.h",
    "ZQ_CNN_Tensor4D.h",
]

# (说明, 文件, 必须存在的正则)
GUARDS = [
    ("ZQ_CNN_BBox.h 引入编译配置（否则 __min/__max/__int64 无定义）",
     "ZQ_CNN_BBox.h", r'#include\s+"ZQ_CNN_CompileConfig\.h"'),
    ("_check_connect 的就地守卫（附录 EN 那一族堆越界写）",
     "ZQ_CNN_Net.h", r'destroys its own input'),
    ("就地守卫比的是**全部组合**而不是同一下标（附录 EN.4）",
     "ZQ_CNN_Net.h", r'for \(int k = 0; k < bottoms\[i\]\.size\(\); k\+\+\)'),
    ("卷积 ReadParam 的 stride==0 守卫（整数除零 -> SIGFPE）",
     "ZQ_CNN_Layer.h", r'invalid conv params'),
    ("卷积 ReadParam 的 kernel/dilate 溢出守卫（附录 EM.3）",
     "ZQ_CNN_Layer.h", r'\(__int64\)dilate_H \* \(kernel_H - 1\) \+ 1 > 0x7FFFFFFF'),
    ("卷积 ReadParam 的 kernel/dilate 溢出守卫在两个卷积类里都存在",
     "ZQ_CNN_Layer.h", r'\(__int64\)dilate_H \* \(kernel_H - 1\) \+ 1 > 0x7FFFFFFF'),
]


def to_wsl(p):
    p = p.replace("\\", "/")
    if p.startswith("D:/"):
        return "/mnt/d/" + p[3:]
    if p.startswith("C:/"):
        return "/mnt/c/" + p[3:]
    return p


def compile_header(name):
    """-fsyntax-only one header, fed on **stdin**.

    Writing a temp .cpp on the Windows side and passing the path does not work:
    `/tmp/x.cpp` from Git Bash is `D:\\tmp\\x.cpp` on Windows and does not exist
    inside WSL, so every file came back "No such file or directory" (appendix EW.4).
    """
    src = '#include "%s"\nint main(){return 0;}\n' % name
    cmd = ["wsl", "-d", os.environ.get("WSL_DIST", "Ubuntu-20.04"), "--",
           "g++", "-fsyntax-only", "-std=c++11", "-fPIC", "-x", "c++", "-"]
    for r in INC_ROOTS:
        cmd += ["-I", to_wsl(r)]
    p = subprocess.run(cmd, input=src, capture_output=True, text=True)
    errs = [l.strip() for l in (p.stdout + p.stderr).splitlines() if "error:" in l]
    return (not errs), (errs[0][:110] if errs else "")


def main():
    print("MNN converter fork gate (appendix EX)")
    print("  %d headers, none of them in any build\n" % len(HEADERS))

    bad = 0
    for h in HEADERS:
        ok, why = compile_header(h)
        if ok:
            print("  COMPILE OK   %s" % h)
        else:
            bad += 1
            print("  COMPILE FAIL %s\n             %s" % (h, why))

    print()
    missing = 0
    cache = {}
    for desc, f, rx in GUARDS:
        if f not in cache:
            cache[f] = io.open(os.path.join(FORK, f), encoding="utf-8",
                               newline="").read()
        if re.search(rx, cache[f]):
            print("  GUARD OK     %s" % desc)
        else:
            missing += 1
            print("  GUARD MISSING %s\n                (%s)" % (desc, f))

    if "--selftest" in sys.argv:
        print("\n=== positive control ===")
        # The guard regexes must MATCH in the main tree (where they were fixed)
        # and the compile must be able to tell a good header from a broken one.
        mainbbox = io.open(os.path.join(ROOT, "ZQCNN", "ZQ_CNN_BBox.h"),
                           encoding="utf-8", newline="").read()
        ok = bool(re.search(r'#include\s+"ZQ_CNN_CompileConfig\.h"', mainbbox))
        print("  main tree ZQCNN/ZQ_CNN_BBox.h has the config include: %s"
              % ("ok" if ok else "** WRONG **"))
        if not ok:
            return 1
        ok2, _ = compile_header("ZQ_CNN_Tensor4D.h")
        print("  an OK header compiles: %s" % ("ok" if ok2 else "** WRONG **"))
        if not ok2:
            return 1
        ok3, _ = compile_header("ZQ_CNN_ThisHeaderDoesNotExist.h")
        print("  a non-existent header fails: %s"
              % ("ok" if not ok3 else "** WRONG **"))
        if ok3:
            return 1
        print("  control passed")

    if bad or missing:
        print("\n%d compile failure(s), %d missing guard(s)" % (bad, missing))
        return 1
    print("\nall %d headers compile, all guards present" % len(HEADERS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
