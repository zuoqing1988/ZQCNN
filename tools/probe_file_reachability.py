#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Which source files does **no compiler ever look at**?

Appendix ES.2 found a real bug precisely because of this: `ZQ_OpticalFlow.h`
declared `int nPixels` twice in the same scope -- ill-formed in *every* standard
C++, so the header could not be compiled by MSVC either.  It survived because
the only file including it was `ZQ_StereoRectify.h`, and *that* header is in no
build target.  Two include hops from any build, and nothing had ever type-checked
it.

`reachability_probe.py` already answers "is this layer type used by a shipped
model".  This one answers the different, complementary question: "is this file
ever compiled at all".

Method
------
Entry points are the translation units the project actually builds:
  - every source listed in the CMakeLists.txt files (samples, libraries);
  - every `tools/zq_*_check.cpp` gate.
From there we walk `#include "..."` transitively (quoted includes resolve
against the including file's directory, then against each -I root).  A file that
never lands in that closure is never parsed by any compiler in this repo.

Diagnostic only.  Run:
    python tools/probe_file_reachability.py
    python tools/probe_file_reachability.py --selftest
"""
import glob
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIRS = ["ZQCNN", "ZQlibFaceID", "ZQ_GEMM", "SamplesZQCNN", "SamplesZQlibFaceID",
               "SamplesZQBLAS", "SamplesZQGEMM", "3rdparty/include/ZQlib",
               "3rdparty/include/ZQlibFaceID"]
SKIP_DIR_PARTS = {".git", "cmake-out-win32-x64", "cmake-out-unix-x64", "build_x64",
                  "build", "__pycache__", "node_modules"}

INC_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)

# -I roots that quoted includes are resolved against (mirrors the build)
INC_ROOTS = [
    ".",
    "ZQCNN",
    "ZQ_GEMM",
    "ZQlibFaceID",
    "3rdparty/include",
    "3rdparty/include/ZQlib",
    "3rdparty/include/ZQlibFaceID",
    "3rdparty/include/opencv4",
    "3rdparty/include/opencv",
]


def norm(p):
    """Normalise to a ROOT-relative, forward-slash path.

    Entry points come out of CMakeLists parsing (relative) while the source
    list comes out of os.walk (absolute).  Comparing one against the other
    without this step made the two sets never intersect, and the probe
    cheerfully reported **all 410 files** as never compiled.
    """
    p = os.path.normpath(p).replace("\\", "/")
    try:
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
    except ValueError:
        return p
    # a path outside ROOT keeps its absolute form (relpath would grow ..s)
    if rel.startswith(".."):
        return p
    return rel


def all_sources():
    out = []
    for d in SOURCE_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames if x not in SKIP_DIR_PARTS]
            for f in filenames:
                if f.endswith((".h", ".hpp", ".c", ".cpp", ".cxx")):
                    # all_sources() walks from ROOT, so make the path
                    # ROOT-relative here too (see norm()).
                    out.append(norm(os.path.join(dirpath, f)))
    return sorted(out)


def entry_points():
    """Sources the build actually compiles: CMake-listed sources + gate .cpp files."""
    eps = set()
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [x for x in dirnames if x not in SKIP_DIR_PARTS]
        for f in filenames:
            if f == "CMakeLists.txt":
                p = os.path.join(dirpath, f)
                try:
                    txt = io.open(p, encoding="utf-8", errors="replace").read()
                except IOError:
                    continue
                for m in re.finditer(r'["\']([^"\']+\.(?:cpp|c|cc|cxx))["\']', txt):
                    cand = norm(os.path.join(dirpath, m.group(1)))
                    if os.path.exists(cand):
                        eps.add(cand)
                # file(GLOB VAR <pattern> ...) -- **the project lists most of its
                # sources this way** (ZQCNN/CMakeLists.txt:5-7 globs
                # layers_c/*.c and layers_nchwc/*.c).  Without expanding GLOB,
                # every kernel TU looked like an entry point that does not
                # exist, and 74 kernel headers came out as "never compiled" --
                # all false, and it buried the real answer.
                for g in re.finditer(r'file\s*\(\s*GLOB\s+\w+\s+([^)]*)\)', txt):
                    # CMake accepts the patterns **unquoted** -- the real files
                    # spell them `${CMAKE_CURRENT_LIST_DIR}/*.cpp` with no
                    # quotes at all -- so match bare whitespace-separated
                    # tokens too, not just quoted strings.  Quoted-only
                    # extraction silently found nothing.
                    args = g.group(1)
                    pats = re.findall(r'["\']([^"\']+)["\']', args)
                    pats += [t for t in args.split()
                             if t and '"' not in t and "'" not in t
                             and ("*" in t or t.endswith((".c", ".cpp", ".cc", ".cxx")))]
                    for pat in pats:
                        # CMake spells the directory as a variable, not a
                        # literal path (ZQCNN/CMakeLists.txt:5-7 all use
                        # ${CMAKE_CURRENT_LIST_DIR}); expand it to the directory
                        # this CMakeLists.txt itself lives in.
                        pat = pat.replace("${CMAKE_CURRENT_LIST_DIR}", dirpath)
                        for hit in glob.glob(norm(os.path.join(dirpath, pat))):
                            if os.path.isfile(hit) and hit.endswith(
                                    (".c", ".cpp", ".cc", ".cxx")):
                                eps.add(norm(hit))
    tools = os.path.join(ROOT, "tools")
    if os.path.isdir(tools):
        for f in os.listdir(tools):
            if f.startswith("zq_") and f.endswith("_check.cpp"):
                eps.add(norm(os.path.join(tools, f)))
    # **Samples 用的是自定义宏，不能靠解析 CMakeLists 拿到它们的源文件。**
    # SamplesZQCNN/CMakeLists.txt:47-53 是
    #     SUBDIRLIST(SAMPLE_SUBDIRS ${CMAKE_CURRENT_LIST_DIR})
    #     foreach(SAMPLE_SUBDIR ${SAMPLE_SUBDIRS})
    #         file(GLOB sample_src ${CMAKE_CURRENT_LIST_DIR}/${SAMPLE_SUBDIR}/*.cpp ...)
    # 要展开它得实现 SUBDIRLIST 并把 foreach 的变量代进去 —— 那是半个 CMake。
    #
    # 这里用一个**明确标注的近似**：每个 sample 目录自成一个可执行目标
    # （`add_executable(${SAMPLE_SUBDIR} ${sample_src})`），
    # 所以 Samples*/**/*.cpp 一律算入口点。
    # 近似只可能**多**收（把没被构建的 sample 也算进来），
    # 不会漏收 —— 而"漏收"才是会产生假阳性的方向。
    for sd in os.listdir(ROOT):
        if not sd.startswith("Samples"):
            continue
        base = os.path.join(ROOT, sd)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames if x not in SKIP_DIR_PARTS]
            for f in filenames:
                if f.endswith((".cpp", ".c", ".cc", ".cxx")):
                    eps.add(norm(os.path.join(dirpath, f)))
    return eps


def resolve(inc, from_file):
    cand = norm(os.path.join(os.path.dirname(from_file), inc))
    if os.path.exists(cand):
        return cand
    for r in INC_ROOTS:
        cand = norm(os.path.join(r, inc))
        if os.path.exists(cand):
            return cand
    return None


def closure(eps):
    seen = set()
    stack = list(eps)
    unresolved = {}
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            continue
        try:
            txt = io.open(p, encoding="utf-8", errors="replace").read()
        except IOError:
            continue
        for m in INC_RE.finditer(txt):
            inc = m.group(1)
            r = resolve(inc, f)
            if r:
                stack.append(r)
            else:
                unresolved.setdefault(os.path.basename(f), set()).add(inc)
    return seen, unresolved


def main():
    eps = entry_points()
    srcs = all_sources()
    reach, unresolved = closure(eps)

    if "--selftest" in sys.argv:
        print("=== positive control ===")
        # ZQ_OpticalFlow.h was proven to be in no build; ZQ_CNN_Tensor4D.h was
        # proven to be in many.  If the walk cannot tell those two apart, the
        # walk is broken and the report below means nothing (appendix EQ).
        ctrl = [
            ("3rdparty/include/ZQlib/ZQ_OpticalFlow.h", False),
            ("3rdparty/include/ZQlib/ZQ_StereoRectify.h", False),
            ("ZQCNN/ZQ_CNN_Tensor4D.h", True),
        ]
        ok = True
        for path, want in ctrl:
            got = path in reach
            mark = "ok" if got == want else "** WRONG **"
            print("  %-46s reachable=%-5s expect=%-5s %s"
                  % (os.path.basename(path), got, want, mark))
            if got != want:
                ok = False
        if not ok:
            print("  control FAILED")
            return 1
        print("  control passed")

    unreached = [s for s in srcs if s not in reach]
    # ZQlib 的头是**头文件库**：它们没有自己的 TU，只有被消费者 include 时才
    # 会被编到，而消费者不是每个都存在。`probe_zqlib_headers.py` 逐个给它们
    # 生成最小 TU 单独编过一遍（基线 144 个头，126 OK / 9 BROKEN），
    # 所以**不要**把它们算进"从未被编译"—— 那会凭空多出 125 条噪声，
    # 把真正的孤儿埋掉。这一类单列出来。
    ZQLIB_PREFIX = "3rdparty/include/ZQlib/"
    zqlib = [s for s in unreached if s.startswith(ZQLIB_PREFIX)]
    others = [s for s in unreached if not s.startswith(ZQLIB_PREFIX)]

    print("\nentry points compiled by the build : %d" % len(eps))
    print("source files under the source dirs : %d" % len(srcs))
    print("reachable from a build entry point : %d" % len(reach & set(srcs)))
    print()
    print("NEVER compiled by any build (%d) -- covered by probe_zqlib_headers.py "
          "instead, listed for completeness:" % len(zqlib))
    for s in zqlib:
        print("   [zqlib] %s" % s)
    print()
    print("NEVER compiled by anything, and NOT covered elsewhere (%d):" % len(others))
    for s in others:
        print("   %s" % s)

    if "--save-baseline" in sys.argv:
        base = sys.argv[sys.argv.index("--save-baseline") + 1]
        with io.open(base, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("# 文件级可达性基线"
                     "（tools/probe_file_reachability.py --save-baseline 生成）\n")
            fh.write("# 格式: <相对路径>\n")
            fh.write("#\n")
            fh.write("# 收录的是「**任何构建都不会编译**、且**也没有别的探针单独编过**"
                     "的文件。\n")
            fh.write("# ZQlib 的头不在此列：它们由 tools/probe_zqlib_headers.py "
                     "逐个生成最小 TU 编译。\n")
            fh.write("#\n")
            fh.write("# 基线的作用：新增/删除一个 sample、或改动 include 结构，"
                     "让某个文件从「有构建」变成\n")
            fh.write("# 「无构建」变成一条可见的 diff —— 附录 ES.2 那个"
                     "重复声明之所以活那么久，\n")
            fh.write("# 就是因为 ZQ_OpticalFlow.h 离任何构建都有两跳。\n")
            for s in others:
                fh.write(s + "\n")
        print("\nbaseline written to %s (%d entries)" % (base, len(others)))
        return 0

    if "--check-baseline" in sys.argv:
        base = sys.argv[sys.argv.index("--check-baseline") + 1]
        want = set()
        for l in io.open(base, encoding="utf-8"):
            l = l.strip()
            if l and not l.startswith("#"):
                want.add(l)
        got = set(others)
        new = sorted(got - want)
        gone = sorted(want - got)
        if new:
            print("\nNEW never-compiled files (a build or an include changed):")
            for s in new:
                print("   + %s" % s)
        if gone:
            print("\nno longer never-compiled:")
            for s in gone:
                print("   - %s" % s)
        if new or gone:
            print("\nbaseline mismatch: %d new, %d gone" % (len(new), len(gone)))
            return 1
        print("\nbaseline OK: %d entries unchanged" % len(want))
    return 0


if __name__ == "__main__":
    sys.exit(main())
