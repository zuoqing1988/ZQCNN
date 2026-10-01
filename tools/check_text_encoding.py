#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扫描仓库里的文本文件, 报三类问题:
  1) UTF-8 有损解码残留 (U+FFFD) —— 说明某个字节被替换过, 汉字已经丢了
  2) 严格 UTF-8 解码失败 (非法字节序列)
  3) **罕见汉字清单**（人工核对用，见下面「查不出的那一类」）

为什么需要: `core.autocrlf=true` 下用 Edit/批量脚本改中文注释时, 只要有一环
走了 errors='replace' 的解码再写回, 原字节就被永久换成 EF BF BD, 而且**不会
报错** —— 只在读那一行时看到几个黑方块。2026-10-01 在 reports/、
docs-changelogs/ 和一个 .asm 注释里各抓到一处。

## 查不出的那一类（2026-10-02 才意识到）

前两类只覆盖「非法字节」。还有一类**任何纯结构检查都发现不了**:

    // clock()/clock_t 本来眉不能编过        <-- 「眉」应为「不」

这是**合法的 UTF-8、只是有一个字错了**, 来源是在 python 里手写 `\\xNN` 的
UTF-8 字节时写错了一位。

因此加了第 3 类: 按出现次数把全仓汉字排序列出**罕见字**, 供人眼扫一眼。
它**不能判定「这个字一定错」** —— 用得对的生僻字也会在里面 ——
定位是**筛子**不是**判官**, 作用是把「905 个不同的汉字」收敛到「几百个低频的」,
让肉眼核对从不可能变成几秒。

用法:
    python tools/check_text_encoding.py             # 扫描并报告, 有问题退出码 1
    python tools/check_text_encoding.py --quiet     # 只在有问题时输出
    python tools/check_text_encoding.py --no-rare   # 关掉第 3 类 (只留前两类)
    python tools/check_text_encoding.py --rarity 30 # 罕见字阈值 (默认 3)

二进制文件 (.nchwbin/.onnx/.pb/.jpg/.lib/.dll/...) 靠扩展名和 NUL 字节识别后跳过。
"""

from __future__ import print_function

import collections
import os
import sys

try:
    import unicodedata
except ImportError:            # py2
    import unicodedata

SKIP_EXT = set("""
.bin .nchwbin .onnx .pb .jpg .jpeg .png .bmp .gif .lib .dll .exe .obj .o .a .so
.zip .7z .tar .gz .ico .ttf .mp4 .avi .npy .pkl .model .caffemodel .dat
""".split())

SKIP_DIR = set(""".git build build_x64 cmake-out-unix-x64 cmake-out-win32-x64
3rdparty/node_modules .vs .idea""".split())

# 这几个是**上游带来的 GBK 文件**, 不是损坏: 严格 UTF-8 解不开是它们的正常状态。
# 改成 UTF-8 会破坏 Windows 侧的 MFC 中文界面和 .bat 的编码, 不要动。
# (mxnet2caffe.bat 必须与它调用的脚本编码一致。)
GBK_FILES = set("""
3rdparty/include/ZQlib/ZQ_MFC_Utils.h
3rdparty/include/ZQlib/ZQ_PutTextCN.h
ZQCNN/ZQ_CNN_FaceCropUtils.h
mobilefacenet-mxnet2caffe-ZQ/mxnet2caffe.bat
""".replace("\\", "/").split())


def is_binary(path, head):
    if os.path.splitext(path)[1].lower() in SKIP_EXT:
        return True
    return b"\x00" in head


def is_cjk(ch):
    o = ord(ch)
    return (0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF
            or 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF)


def main():
    # 本机 Windows 控制台是 GBK: 不显式改成 utf-8 的话, 罕见字清单全是乱码,
    # 恰恰在最需要人眼核对的时候看不清 (2026-10-02 实测)。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    quiet = "--quiet" in sys.argv
    want_rare = "--no-rare" not in sys.argv
    rarity = 3
    if "--rarity" in sys.argv:
        try:
            rarity = int(sys.argv[sys.argv.index("--rarity") + 1])
        except (IndexError, ValueError):
            print("--rarity needs an integer")
            return 2
    problems = []
    scanned = 0
    counts = collections.Counter()
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR]
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, ".").replace("\\", "/")
            if rel in GBK_FILES:
                continue
            try:
                raw = open(path, "rb").read()
            except (IOError, OSError):
                continue
            if is_binary(path, raw[:4096]):
                continue
            scanned += 1
            n_fffd = raw.count(b"\xef\xbf\xbd")
            if n_fffd:
                lines = [i + 1 for i, l in enumerate(raw.split(b"\n"))
                         if b"\xef\xbf\xbd" in l]
                problems.append((path, "U+FFFD x%d" % n_fffd, lines))
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as e:
                problems.append((path, "illegal UTF-8: %s" % e, []))
                continue
            if want_rare:
                for ch in text:
                    if is_cjk(ch) and not ch.isspace():
                        counts[ch] += 1

    rare = sorted([(c, n) for c, n in counts.items() if n <= rarity],
                  key=lambda kv: (kv[1], kv[0]))

    if problems:
        for path, why, lines in problems:
            print("%s: %s%s" % (path, why,
                                ("  lines %s" % lines[:8]) if lines else ""))
        print("\n%d problem(s) in %d text files scanned" % (len(problems), scanned))
        return 1

    if want_rare and rare:
        print("OK: %d text files, all strict UTF-8, no U+FFFD" % scanned)
        print("")
        print("=== 人工核对：罕见汉字（出现 <= %d 次，共 %d 个不同字 / %d 次） ==="
              % (rarity, len(rare), sum(n for _c, n in rare)))
        print("这一类**不是**自动判定的错误，只是把 %d 个不同汉字收敛到可疑范围。"
              % len(counts))
        print("乱码的典型特征：一个常用字被换成形近的生僻字（"
              "如 2026-10-02 抓到的「本来<眉>不能编过」，应为「不」）。")
        print("逐个扫一眼，或者只在刚改过中文注释的那几个文件里人工核对：")
        buf = []
        for c, n in rare:
            buf.append(u"%s%d" % (c, n))
        for i in range(0, len(buf), 40):
            print("   " + " ".join(buf[i:i + 40]))
        return 0

    if not quiet:
        if want_rare:
            print("OK: %d text files, all strict UTF-8, no U+FFFD" % scanned)
            print("     汉字 %d 个不同字，罕见字(<=%d 次) %d 个 —— 见上一条说明"
                  % (len(counts), rarity, len(rare)))
        else:
            print("OK: %d text files, all strict UTF-8, no U+FFFD" % scanned)
    return 0


if __name__ == "__main__":
    sys.exit(main())
