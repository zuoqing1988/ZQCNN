#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""No sample may call a blocking OpenCV GUI function.

The rule
--------
The user asked for this explicitly (2026-10-01, and it is written into
AGENTS.md "示例程序规则"): every `cv::namedWindow` / `cv::imshow` / `cv::waitKey`
in Samples* must be commented out.  A headless run otherwise **hangs** on
`waitKey`, or dies on a missing display -- either way the sample stops being
runnable and the regression that runs it stops being meaningful.

As of 2026-10-03 the rule holds for all 73 sample sources; **nothing enforced
it**, so one `imshow` added to one sample would silently break the Linux sample
regression, with the first symptom being a timeout far away from the edit.

How it decides
--------------
**Ask the compiler, don't hand-roll a comment parser** (AGENTS.md: "不要自己手写
C++ 注释/语法解析器 ... 要解析 C++ 就用编译器"):

  1. `g++ -E` (linemarkers on) preprocesses each sample, which **strips comments**;
  2. the `# <line> "<file>"` markers attribute every output line to its origin
     file, and we only consider lines attributed to the sample itself --
     otherwise OpenCV's own `highgui.hpp` *declarations* (`void imshow(...)`)
     would trip the check on every single sample;
  3. a live `imshow(` / `waitKey(` / `namedWindow(` on such a line is a hit.

All samples are preprocessed in **one** WSL invocation: 73 separate `wsl`
startups cost minutes, which is the mistake `probe_zqlib_headers.py`'s comment
warns about.

Run:
    python tools/check_no_gui_calls.py
    python tools/check_no_gui_calls.py --selftest   # + positive control
"""
import io
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# **不含 ZQCNN_to_MNN**：它要 MNN 的 `ImageProcess.hpp`（附录 EX.1 ——
# 那棵子树要 MNN 框架才能 configure，本机没有），`g++ -E` 必然失败。
# 把一个**编不了**的目录塞进"必须编过"的检查里，只会让它永远红。
SAMPLE_DIRS = ["SamplesZQCNN", "SamplesZQlibFaceID", "SamplesZQBLAS",
               "SamplesZQGEMM"]

# **只保留真实存在的目录。** 给 g++ 一个不存在的 -I 目录，它会直接报错；
# 而"扫了等于没扫"被当成"干净"，就是这道检查最该防的失败模式。
_INC_CANDIDATES = (".", "ZQCNN", "ZQ_GEMM", "ZQ_GEMM/math", "ZQlibFaceID",
                   "3rdparty/include", "3rdparty/include/ZQlib",
                   "3rdparty/include/opencv4", "3rdparty/include/opencv",
                   # SamplesZQBLAS / SamplesZQGEMM 用的是
                   #   #include "zq_gemm_32f_align_c.h"   （不带 math/ 前缀）
                   # 而 ZQCNN 那边是 "math/zq_gemm_32f_align_c.h"，
                   # 所以 ZQ_GEMM 与 ZQ_GEMM/math 都得在包含路径里。
                   "SamplesZQCNN", "SamplesZQBLAS", "SamplesZQGEMM")
INC = ["-I" + os.path.join(ROOT, d) for d in _INC_CANDIDATES
       if os.path.isdir(os.path.join(ROOT, d))]

GUI = re.compile(r"\b(waitKey|namedWindow|imshow)\s*\(")
MARK = re.compile(r'^# \d+ "([^"]+)"')
WSL_DIST = os.environ.get("WSL_DIST", "Ubuntu-20.04")


def to_wsl(p):
    p = p.replace("\\", "/")
    if p.startswith("D:/"):
        return "/mnt/d/" + p[3:]
    if p.startswith("C:/"):
        return "/mnt/c/" + p[3:]
    return p


def inc_args():
    """-I 列表。**前缀不能被 to_wsl 吃掉。**

    `to_wsl(i[2:])` 会连 "-I" 一起剥掉，于是那些目录变成**裸参数**、
    被 g++ 当成输入文件，于是报 "ZQ_CNN_Net.h: No such file or directory"，
    而 73 个 sample 全部"通过"。
    这正是 AGENTS.md 里「不要把 Windows 路径丢给 WSL 的 bash」那个坑的变体，
    而我是在**写了那条规则的同一个文件里**踩的。
    """
    return ["-I" + to_wsl(i[2:]) for i in INC]


def sample_sources():
    out = []
    for d in SAMPLE_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dp, dn, fn in os.walk(base):
            dn[:] = [x for x in dn if x not in ("cmake-out-win32-x64", "build")]
            for f in sorted(fn):
                if f.endswith((".cpp", ".c")):
                    out.append(os.path.join(dp, f))
    return sorted(out)


def attribute_gui(stdout, path):
    """GUI calls among the lines attributed to `path` itself.

    Keyed on the file's **basename**, not on "the path contains Samples" --
    the first version used the latter, and then the positive control (whose
    temp file lives under tools/) could not produce a signal at all.
    """
    me = os.path.basename(path)
    cur, hits = None, []
    for line in (stdout or "").splitlines():
        m = MARK.match(line)
        if m:
            cur = m.group(1)
            continue
        if cur and os.path.basename(cur) == me and GUI.search(line):
            hits.append(line.strip()[:80])
    return hits


def scan_all_v2(paths):
    """Same, but each sample's preprocessed output is tagged so we can split it.

    Emitting `A|<i>` before each file's output and `B|<i>` after makes the
    split unambiguous -- guessing where one file's output ends and the next
    begins is exactly the kind of thing that silently reports "clean".
    """
    lines = ["set +e", "W=/tmp/zq_gui_check_%d" % os.getpid(),
             "rm -rf $W && mkdir -p $W"]
    incs = " ".join("'%s'" % a for a in inc_args())
    for i, p in enumerate(paths):
        lines.append(
            "echo 'A|%d'; "
            "g++ -E -std=c++11 %s '%s' 2> $W/%d.err; echo \"X|%d|$?\"; "
            "echo 'B|%d'" % (i, incs, to_wsl(p), i, i, i))
    lines.append("rm -rf $W")
    r = subprocess.run(["wsl", "-d", WSL_DIST, "--", "bash", "-s"],
                       input="\n".join(lines).encode("utf-8"),
                       capture_output=True)
    out = (r.stdout or b"").decode("utf-8", "replace")

    hits, errs = {}, {}
    cur, buf = None, []
    def flush():
        if cur is not None:
            # 键必须是**路径**，不是下标 —— 第一版写成 `hits[cur]`，
            # 于是 main() 里 `os.path.relpath(p, ...)` 收到一个 int 直接抛
            # TypeError，**在报出任何结论之前**就崩了。
            hits[paths[cur]] = attribute_gui("\n".join(buf), paths[cur])
    for line in out.splitlines():
        if line.startswith("A|"):
            flush()
            cur, buf = int(line[2:]), []
            continue
        if line.startswith("X|"):
            _, i, rc = line.split("|", 2)
            if rc != "0":
                errs[paths[int(i)]] = "g++ -E rc=%s" % rc
            continue
        if line.startswith("B|"):
            continue
        if cur is not None:
            buf.append(line)
    flush()
    return hits, errs


def main():
    srcs = sample_sources()
    print("GUI 调用门禁：%d 个 sample 源文件" % len(srcs))
    print("规则（用户 2026-10-01 明确要求，AGENTS.md 亦有）：")
    print("  Samples* 里的 cv::namedWindow / cv::imshow / cv::waitKey 一律注释掉。\n")
    print("判据：g++ -E 去掉注释后，按 linemarker 只看**该 sample 自己**的行 ——")
    print("      否则 OpenCV highgui.hpp 自己的 imshow/waitKey 声明会让每个 sample 都中。")
    print("      一次 WSL 调用编完全部（73 次 wsl 启动要好几分钟）。\n")

    hits, errs = scan_all_v2(srcs)
    for p, n in sorted(hits.items()):
        if n:
            print("  FAIL %s  (%d 处未注释的 GUI 调用)"
                  % (os.path.relpath(p, ROOT), len(n)))
            for l in n[:3]:
                print("        %s" % l)
    for p, why in sorted(errs.items()):
        print("  CANNOT-CHECK %s  (%s)" % (os.path.relpath(p, ROOT), why))

    if "--selftest" in sys.argv:
        print("\n=== positive control ===")
        probe = os.path.join(ROOT, "tools", "_gui_selftest_tmp.cpp")
        try:
            # a commented-out call must NOT be flagged
            with io.open(probe, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("int f() {\n"
                         "  // cv::imshow(\"x\", 0);\n"
                         "  /* cv::waitKey(0); */\n"
                         "  return 0;\n}\n")
            h, _ = scan_all_v2([probe])
            ok1 = not h.get(probe)
            print("  commented-out call is not flagged: %s"
                  % ("ok" if ok1 else "** WRONG **"))
            # a live one MUST be flagged
            with io.open(probe, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("int f() {\n  cv::imshow(\"x\", 0);\n  return 0;\n}\n")
            h, _ = scan_all_v2([probe])
            ok2 = bool(h.get(probe))
            print("  live call IS flagged: %s" % ("ok" if ok2 else "** WRONG **"))
            if not (ok1 and ok2):
                return 1
            print("  control passed")
        finally:
            try:
                os.remove(probe)
            except OSError:
                pass

    if hits and any(hits.values()):
        print("\n有 sample 含未注释的 GUI 调用 —— 无头环境会阻塞/失败。")
        return 1
    if errs:
        print("\n%d 个 sample **编不了**，无法判定 —— 那不是通过。" % len(errs))
        return 1
    print("\n全部 %d 个 sample：无未注释的 GUI 调用，且每一个都成功预处理过。"
          % len(srcs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
