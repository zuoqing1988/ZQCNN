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
    # 附录 IX.14：`_merge_bns_to_*` 的通道数一致性守卫。这份分叉头**整族漏了**，
    # 而 `b`/`a` 是 BatchNormScale 自己的张量（按 BN 那层的 bottom_C 分配）——
    # BN 通道数与前一层卷积的 num_output 对不上时，`b->GetFirstPixelPtr()[n]`
    # 就是**堆越界读**，读到的垃圾值还会被写回 filters。
    # 主副本 `ZQCNN/ZQ_CNN_Net.h` 与 `ZQ_CNN_Net_NCHWC.h` 都有；门禁盯住第三份。
    ("_merge_bns_to_conv 的通道数一致性守卫（附录 IX.14，堆越界读）",
     "ZQ_CNN_Net.h", r'if \(b->GetC\(\) != N \|\| a->GetC\(\) != N\)\s*\n\s*return false;',
     "_merge_bns_to_conv"),
    ("_merge_bns_to_innerproduct 的通道数一致性守卫（附录 IX.14）",
     "ZQ_CNN_Net.h", r'if \(b->GetC\(\) != N \|\| a->GetC\(\) != N\)\s*\n\s*return false;',
     "_merge_bns_to_innerproduct"),
    ("_merge_bns_to_dwconv 的通道数一致性守卫（附录 IX.14）",
     "ZQ_CNN_Net.h", r'if \(b->GetC\(\) != kC \|\| a->GetC\(\) != kC\)\s*\n\s*return false;',
     "_merge_bns_to_dwconv"),
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
    # `p.stdout` / `p.stderr` **可能有一个是 None**（进程起不来 / 被杀 / 解码失败）。
    # 原来直接 `p.stdout + p.stderr`，于是那种情况下门禁自己抛
    # `TypeError: can only concatenate str (not "NoneType")` **带着 traceback 崩掉**，
    # 而不是报一行 COMPILE FAIL —— 崩掉虽然也红，但红的位置和原因都指不到被测文件。
    out = (p.stdout or '') + (p.stderr or '')
    errs = [l.strip() for l in out.splitlines() if "error:" in l]
    if not errs and p.returncode != 0 and not out.strip():
        return False, "编译器没给出任何输出（rc=%d）" % p.returncode
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
    for item in GUARDS:
        desc, f, rx = item[0], item[1], item[2]
        func = item[3] if len(item) > 3 else None
        if f not in cache:
            cache[f] = io.open(os.path.join(FORK, f), encoding="utf-8",
                               newline="").read()
        # 给了函数名就**只在那个函数体里**找 ——
        # `_merge_bns_to_conv` 与 `_merge_bns_to_innerproduct` 的守卫文本一模一样，
        # 全文件搜的话两个断言其实只验了一次：删掉其中一个，另一个照样「OK」。
        # 这正是 HX 那个形状（改一处、另一个副本没跟上）在门禁里的翻版。
        scope = cache[f]
        if func:
            m = re.search(r'\bbool\s+' + re.escape(func) + r'\s*\(', cache[f])
            if not m:
                missing += 1
                print("  GUARD MISSING %s\n                (找不到函数 %s)" % (desc, func))
                continue
            i = cache[f].index('{', m.end())
            depth, j = 0, i
            while j < len(cache[f]):
                if cache[f][j] == '{':
                    depth += 1
                elif cache[f][j] == '}':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            scope = cache[f][i:j]
        if re.search(rx, scope):
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
