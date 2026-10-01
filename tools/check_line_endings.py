#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查仓库里被 git 跟踪的文本文件有没有损坏的行尾。

需要检查的原因（都是实际踩过的坑）：

1) **孤立的 CR 落进预处理符 token。**
   形如 ``#include "x.h"\\r\\r\\n`` 的文件里，有两个 CR 紧挨着 LF 前，
   真正紧贴 LF 的只有一个。按 C 标准的翻译阶段 3，``#include`` 行上的
   CR 会并进 header-name token，``#ifndef``/``#define`` 的宏名也会带上 CR。
   MSVC 容忍，gcc/clang 只在部分位置告警，属于 UB。

2) **宏续行失效。**
   行尾 ``\\`` 后接 ``\\r\\r\\n`` 时，反斜杠转义掉的是那个 CR 而不是换行，
   宏在第一行就被截断（详见 .gitattributes 里 *_raw.h 的说明）。

3) **字符串字面量与字符宽度**。
   源码里的 ``"..."\\r\\r\\n`` 字面量是按实际字节编译的，跨平台读到的
   字符串长度不一致。

用法::

    python tools/check_line_endings.py            # 检查全仓库
    python tools/check_line_endings.py --fix      # 顺手修复（只动行尾）

退出码：0 = 干净，1 = 有问题。
"""

import argparse
import os
import re
import subprocess
import sys

# 生成式内核头文件里全是跨行宏，必须保持纯 LF
FORCE_LF_SUFFIXES = ("_raw.h",)
FORCE_LF_GLOBS = ("ZQCNN/layers_nchwc/", "ZQCNN/layers_c/", "ZQ_GEMM/math/")

TEXT_SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".inl", ".inc",
                 ".cu", ".cuh", ".cmake", ".txt", ".py", ".sh", ".md",
                 ".bat", ".yml", ".yaml", ".json", ".asm")

SKIP_DIRS = {".git", "build_x64", "cmake-out-unix-x64", "cmake-out-win32-x64",
             "3rdparty", "node_modules", "__pycache__", "data", "model"}

# \r{2,}\n  : 多个 CR 后跟 LF（CRCRLF 家族）
# \r(?!\n)  : 不紧跟 LF 的孤立 CR（老 Mac 换行，或行中杂散 CR）
RE_MULTI_CR = re.compile(rb"\r{2,}\n")
RE_LONE_CR = re.compile(rb"\r(?!\n)")


def must_be_lf(relpath: str) -> bool:
    if relpath.endswith(FORCE_LF_SUFFIXES):
        return True
    norm = relpath.replace("\\", "/")
    return any(norm.startswith(g) for g in FORCE_LF_GLOBS)


def iter_tracked():
    """优先用 git ls-files；不在 git 仓库里时退化成扫目录。"""
    try:
        out = subprocess.run(["git", "ls-files", "-z"],
                             capture_output=True, check=True).stdout
        for name in out.split(b"\0"):
            if name:
                yield name.decode("utf-8", "surrogateescape")
        return
    except (OSError, subprocess.CalledProcessError):
        pass
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            yield os.path.relpath(os.path.join(root, f), ".")


def scan(data: bytes):
    """返回 (问题类型列表)。"""
    problems = []
    if RE_MULTI_CR.search(data):
        problems.append("multi-CR")
    if RE_LONE_CR.search(data):
        problems.append("lone-CR")
    return problems


def normalize(data: bytes, want_lf: bool) -> bytes:
    data = RE_MULTI_CR.sub(b"\n" if want_lf else b"\r\n", data)
    data = RE_LONE_CR.sub(b"\n" if want_lf else b"\r\n", data)
    if want_lf:
        data = data.replace(b"\r\n", b"\n")
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="就地修复行尾")
    args = ap.parse_args()

    bad = []
    fixed = 0
    for rel in iter_tracked():
        if not rel.lower().endswith(TEXT_SUFFIXES):
            continue
        top = rel.replace("\\", "/").split("/")[0]
        if top in SKIP_DIRS:
            continue
        try:
            with open(rel, "rb") as f:
                data = f.read()
        except OSError:
            continue
        problems = scan(data)
        if not problems:
            continue
        bad.append((rel, ",".join(problems)))
        if args.fix:
            with open(rel, "wb") as f:
                f.write(normalize(data, must_be_lf(rel)))
            fixed += 1

    for rel, why in bad:
        print("%-12s %s" % (why, rel))
    if bad:
        print("\n%d file(s) with broken line endings%s."
              % (len(bad), " (fixed)" if args.fix else ""))
        if not args.fix:
            print("Run: python tools/check_line_endings.py --fix")
        return 0 if args.fix and fixed == len(bad) else 1
    print("line endings OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
