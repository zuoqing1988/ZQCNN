#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扫描仓库里的文本文件, 报两类问题:
  1) UTF-8 有损解码残留 (U+FFFD) —— 说明某个字节被替换过, 汉字已经丢了
  2) 严格 UTF-8 解码失败 (非法字节序列)

为什么需要: `core.autocrlf=true` 下用 Edit/批量脚本改中文注释时, 只要有一环
走了 errors='replace' 的解码再写回, 原字节就被永久换成 EF BF BD, 而且**不会
报错** —— 只在读那一行时看到几个黑方块。2026-10-01 在 reports/、
docs-changelogs/ 和一个 .asm 注释里各抓到一处。

用法:
    python tools/check_text_encoding.py           # 扫描并报告, 有问题退出码 1
    python tools/check_text_encoding.py --quiet   # 只在有问题时输出

二进制文件 (.nchwbin/.onnx/.pb/.jpg/.lib/.dll/...) 靠扩展名和 NUL 字节识别后跳过。
"""

from __future__ import print_function

import os
import sys

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


def main():
    quiet = "--quiet" in sys.argv
    problems = []
    scanned = 0
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
                raw.decode("utf-8")
            except UnicodeDecodeError as e:
                problems.append((path, "illegal UTF-8: %s" % e, []))

    if problems:
        for path, why, lines in problems:
            print("%s: %s%s" % (path, why,
                                ("  lines %s" % lines[:8]) if lines else ""))
        print("\n%d problem(s) in %d text files scanned" % (len(problems), scanned))
        return 1
    if not quiet:
        print("OK: %d text files, all strict UTF-8, no U+FFFD" % scanned)
    return 0


if __name__ == "__main__":
    sys.exit(main())
