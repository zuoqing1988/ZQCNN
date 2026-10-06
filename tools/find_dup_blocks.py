#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
找「**同一段代码紧挨着出现两次**」的站点。

起因（2026-10-06，附录 DL）：读 `ZQCNN/ZQ_CNN_PersonPose2.h` 时发现

    // 审计修复（附录 IN.8）：返回值原来被丢弃。          <- 第 1 遍
    ...
    // 审计修复（附录 IN.8）：返回值原来被丢弃。          <- 第 2 遍，一字不差
    ...
    temp_img.ConvertFromBGR(...);                      <- 返回值仍被丢弃（原调用）
    if (!temp_img.ConvertFromBGR(...)) { ... }         <- 修好的那次
    if (!temp_img.ConvertFromBGR(...)) { ... }         <- **同一段又来一遍**

也就是说**某一次脚本批量改写在这个文件上跑了两遍**。
注释、代码都多了一份。后果分三种，从轻到重：

  ① 白白多做一次转换（性能）
  ② 两次 `return`（第二段不可达 —— 无害但会误导读代码的人）
  ③ **两次释放 / 两次自增 / 两次写同一处**，那就不是"多余"而是**行为变了**

判据只看"**相邻且逐字相同**"，不做任何语义判断 ——
本工具只负责**把候选站点全部列出来**，剩下的逐个人工核实。
（这与本文件「阴性结论也要落盘 / 扫到 0 命中先怀疑工具」是同一条纪律的另一面：
**报出来的必须逐条看过，没报出来的也要知道为什么没报出来。**）
"""
import os
import sys
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_DIRS = ["ZQCNN", "ZQ_GEMM", "ZQlibFaceID", "SamplesZQCNN", "SamplesZQlibFaceID",
             "3rdparty", "ZQCNN_to_MNN"]
SKIP_DIR_PARTS = [".git", "build_x64", "cmake-out", "__pycache__"]
MIN_LINES = 4          # 至少 4 行才算"一段"
MAX_LINES = 60
EXTS = (".h", ".c", ".cpp", ".hpp", ".inl")

# **按宏展开成多份的文件必然含大量「相邻且逐字相同」的块** ——
# 同一份 _raw.h 被 include 两次、三次，生成的函数体在源文件里就是背靠背的。
# 第一版跑出来 142 处、20 个文件，其中绝大部分来自这 12 个（实测）：
#   layers_c/*_32f_align_c_raw.h / layers_nchwc/*_raw.h / zq_gemm_32f_align_c.c
#   layers_c/zq_cnn_*_32f_align_c.c / layers_nchwc/zq_cnn_*_nchwc.c
#   SamplesZQCNN/SampleMatMul/*.cpp
# 把它们排除掉之后剩下的，才是"人手写的代码里出现了重复"。
# **排除名单必须写出来并说明理由** —— 白名单/排除项不写理由，
# 下一个人就不知道它是"查过了没问题"还是"忘了查"（AGENTS.md 第 3 条那类坑）。
EXCLUDE = [
    # 宏展开成 align0 / align128bit / align256bit 多个变体的那一族
    "ZQCNN/layers_c/", "ZQCNN/layers_nchwc/",
    "ZQ_GEMM/math/zq_gemm_32f_align_c.c",
    # 内含一份 3000+ 行的生成代码
    "SamplesZQCNN/example_for_very_high_gflops/",
    "SamplesZQCNN/SampleMatMul/",
]


def iter_files():
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames if x not in SKIP_DIR_PARTS]
            for fn in filenames:
                if fn.endswith(EXTS):
                    yield os.path.join(dirpath, fn)


def norm(line):
    """只比"去掉了行尾空白"的文本 —— 缩进不同不算重复。"""
    return line.rstrip()


def find_adjacent_dupes(lines):
    hits = []
    n = len(lines)
    for k in range(MIN_LINES, MAX_LINES + 1):
        i = 0
        while i + 2 * k <= n:
            a = [norm(x) for x in lines[i:i + k]]
            b = [norm(x) for x in lines[i + k:i + 2 * k]]
            if a == b and any(x.strip() for x in a):
                # 避免把同一个更长的重复里的小片段重复报出来
                hits.append((i, k))
                i += 2 * k
            else:
                i += 1
    # 只保留"没有被更长命中覆盖"的那些
    keep = []
    for (i, k) in sorted(hits, key=lambda t: (-t[1], t[0])):
        if any(i >= a and i + k <= a + b for (a, b) in keep):
            continue
        keep.append((i, k))
    return sorted(keep)


def main():
    total = 0
    per_file = []
    skipped = 0
    for path in iter_files():
        rel0 = os.path.relpath(path, ROOT).replace("\\", "/")
        if any(rel0.startswith(x) for x in EXCLUDE):
            skipped += 1
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except (IOError, OSError):
            continue
        if not lines:
            continue
        hits = find_adjacent_dupes(lines)
        if hits:

            per_file.append((rel0, len(lines), hits))
            total += len(hits)
    if not per_file:
        print("OK: 排除 %d 个「按宏展开成多份」的文件后，剩下的人手写代码里" % skipped)
        print("    没找到「相邻且逐字相同」的 >=%d 行代码块。" % MIN_LINES)
        return 0
    print("排除 %d 个「按宏展开成多份」的文件后，发现 %d 处"
          "「相邻且逐字相同」的 >=%d 行代码块（%d 个文件）:\n"
          % (skipped, total, MIN_LINES, len(per_file)))
    for rel, nlines, hits in per_file:
        print("  %s（%d 行）" % (rel, nlines))
        for (i, k) in hits:
            if i + k >= len(lines):
                continue
            first = norm(lines[i]).strip()
            if len(first) > 58:
                first = first[:58] + "..."
            print("     :%-6d %2d 行   %s" % (i + 1, k, first))
    print("\n逐条核实过再下结论 —— 后果可能是「白做一次」也可能是「行为变了」。")
    return 1


if __name__ == "__main__":
    sys.exit(main())