#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
门禁：图像(batch)维的指针步进**不许**用 sliceStep 冒充 imStep。

背景 —— 附录 IU.1 / IU.2（2026-10-06）
------------------------------------------------
NCHWC 的张量布局是 [n][c][h][w]：

    widthStep  走一个像素（w 方向，含对齐）
    sliceStep  走一个**通道片**（c 方向，步长 align）
    imageStep  走完**一张图的全部通道**（n 方向）

所以「遍历 batch 的那个循环」推进指针时必须用 imStep。
写成 sliceStep 时，N=1 完全看不出来；而只要 C 不是 align 的整数倍
（ChangeSize 里 dst_slice = ceil(dst_C/align) >= 2），两个 step 就分家，
第 2 张及以后的图会被读/写到错误的通道片 —— 且因为 sliceStep <= imageStep，
地址仍在缓冲区里，**不越界、不崩、没有任何 sanitizer 会报**，纯静默算错。

实测被这个门禁钉住的两个站点：
    ZQCNN/layers_nchwc/zq_cnn_pooling_nchwc_raw.h
        zq_cnn_avgpooling_nopadding_suredivided_kernel2x2   (IU.1)
    ZQCNN/layers_nchwc/zq_cnn_resize_nchwc_raw.h
        zq_cnn_resize_with_safeborder                       (IU.2)

规则怎么写才不会变成「恒真」/「恒假」
------------------------------------
**不写死变量名**。这里用的是命名约定 `<prefix>_im_ptr` 里的 `im` 段：
凡是推进 image 级指针（名字里带 `_im_ptr`）的语句，步长表达式里
**不允许出现** `<prefix>_sliceStep`。名字可以是 in_/out_/cur_/任意前缀。
（教训来自审计日志里那八次自伤：写死一个源码里根本不存在的变量名，
门禁就永远为真，等于没判。）

同一行里 `in_slice_ptr += in_sliceStep` 是**正确**的 —— 那推进的是通道片，
不是图像。所以只盯 `_im_ptr`。
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 只扫手写内核。3rdparty/ 是第三方，不归我们管。
SCAN_DIRS = [
    os.path.join(ROOT, "ZQCNN", "layers_nchwc"),
    os.path.join(ROOT, "ZQCNN", "layers_c"),
    os.path.join(ROOT, "ZQCNN", "math"),
    os.path.join(ROOT, "ZQ_GEMM", "math"),
]

# `<prefix>_im_ptr += <expr>`  —— 只认 `+=`，因为这才是循环里的推进。
# 结束符要同时接受 `;` 和 `)`：这些语句绝大多数出现在 **for 的第三个子句**里，
# 以 `)` 收尾而不是 `;`。第一版只写了 `;`，于是整条规则**恒真** ——
# 变异测试（tools/_mut_imstep.py）当场把它打出来，那才叫「门禁绿了 ≠ 门禁扫到了东西」。
IM_ADVANCE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*_im_ptr)\s*\+=\s*([^;)]+)[;)]")

# 步长表达式里出现 `<prefix>_sliceStep`（前缀与左边的 im_ptr 对不上也算，
# 例如 in_im_ptr += out_sliceStep）
SLICE_STEP = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*_sliceStep\b")

SKIP_EXT = (".o", ".obj", ".a", ".lib", ".so", ".dll", ".exe")


def iter_files():
    for d in SCAN_DIRS:
        if not os.path.isdir(d):
            continue
        for dirpath, _dirnames, filenames in os.walk(d):
            for fn in filenames:
                if fn.endswith(SKIP_EXT):
                    continue
                if not (fn.endswith((".h", ".c", ".cpp", ".hpp", ".inl"))):
                    continue
                yield os.path.join(dirpath, fn)


def selfcheck():
    """自测：证明这条规则**不是恒真**。

    教训（2026-10-06 实测）：第一版把结束符写成 `[^;]+;`，而这些语句
    绝大多数在 for 的第三个子句里、以 `)` 收尾 —— 于是一条规则都没匹配上，
    门禁永远绿。第一次变异测试（把两个真实站点改回 sliceStep）当场把它打出来。
    所以自测必须证明两件事：
      1) 合法的 imStep 推进**能**被匹配到（规则没瞎）；
      2) sliceStep 推进**会**被报告（规则有效）。
    """
    good = [
        "\t\tn++, in_im_ptr += in_imStep, out_im_ptr += out_imStep)\n",
        "\t\t\tn++, in_im_ptr += in_imStep)\n",
        "\tfor (...) n++, cur_im_ptr += out_imStep;\n",
    ]
    bad = [
        "\t\tn++, in_im_ptr += in_sliceStep, out_im_ptr += out_sliceStep)\n",
        "\t\t\tn++, in_im_ptr += in_sliceStep)\n",
    ]
    fail = 0
    for g in good:
        if not IM_ADVANCE.search(g):
            print("  FAIL 合法形态没被匹配:", g.strip())
            fail += 1
    for b in bad:
        m = IM_ADVANCE.search(b)
        if not (m and SLICE_STEP.search(m.group(2))):
            print("  FAIL 非法形态没被报出:", b.strip())
            fail += 1
    # 反向自检：`in_slice_ptr += in_sliceStep` 是**正确**的（推进通道片），
    # 必须**不**被报出来，否则这条规则会天天误报直到被人加白名单关掉。
    ok = "\t\tc += zq_mm_align_size, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)\n"
    m = IM_ADVANCE.search(ok)
    if m and SLICE_STEP.search(m.group(2)):
        print("  FAIL 通道片的推进被误判:", ok.strip())
        fail += 1
    if fail:
        print("check_imstep_guard --selfcheck: %d 项不过" % fail)
        return 1
    print("check_imstep_guard --selfcheck: OK（3 正例 + 2 反例 + 1 不该报）")
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        return selfcheck()
    hits = []
    scanned = 0
    for path in iter_files():
        scanned += 1
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        # 先把 /* ... */ 注释整段抹掉，避免注释里的示例代码误报
        in_block = False
        for i, raw in enumerate(lines):
            line = raw
            if in_block:
                if "*/" in line:
                    line = line.split("*/", 1)[1]
                    in_block = False
                else:
                    continue
            while "/*" in line:
                if "*/" in line.split("/*", 1)[1]:
                    line = line.split("/*", 1)[0] + line.split("*/", 1)[1]
                else:
                    line = line.split("/*", 1)[0]
                    in_block = True
                    break
            line = line.split("//", 1)[0]
            m = IM_ADVANCE.search(line)
            if not m:
                continue
            var, expr = m.group(1), m.group(2)
            if SLICE_STEP.search(expr):
                hits.append((rel, i + 1, var.strip(), expr.strip()))

    if hits:
        print("FAIL: 图像维指针推进用了 sliceStep（附录 IU.1 / IU.2）")
        for rel, ln, var, expr in hits:
            print("  %s:%d  %s += %s" % (rel, ln, var, expr))
        print("  —— NCHWC 布局 [n][c][h][w]，n 方向必须用 *_imStep")
        return 1
    print("OK: %d 个内核源文件，batch 维指针步进无 sliceStep 冒充（附录 IU.1 / IU.2）"
          % scanned)
    return 0


if __name__ == "__main__":
    sys.exit(main())