#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""每个头文件都必须能**单独**编过 —— 附录 IM。

为什么要这道门禁
----------------
C++ 头文件的头号卫生问题：某个头用了另一个类型，却**没有 include 它的定义**。
这不会在主工程里暴露，因为主工程的每个 TU 都按「习惯顺序」include 一堆头，
某个头恰好排在提供方**前面**，于是看起来一切正常。

本项目里真实踩到的一例（附录 IM.1）：

    ZQCNN/ZQ_CNN_CascadeOnet_Interface.h:113
        std::vector<ZQ_CNN_Net*> nets;      // 具体类型 ZQ_CNN_Net
    而它只 include 了 ZQ_CNN_Net_Interface.h（里面只有抽象基类 ZQ_CNN_Net_Interface）

    => 这个头**不是自包含的**：按自然顺序 include 它就报
       'ZQ_CNN_Net' was not declared in this scope
    而 SampleVideoFaceDetection_Interface.cpp 恰好第一行是 #include "ZQ_CNN_Net.h"、
    第三行才 include 本头 —— 顺序正好把它盖住，所以**一直没人发现**。
    写 `-fsyntax-only` 检查「这个头能不能单独编过」时当场就炸出来了。

判定
----
对每个头生成一个**只 include 它自己**的 .cpp，编到语法检查（不链接）：

    #include "ZQ_CNN_CompileConfig.h"
    #include "<ZQCNN/ZQ_CNN_CascadeOnet_Interface.h>"

`ZQ_CNN_CompileConfig.h` 排在前面是有意的：它定义 `__max`/`__min`/`__int64`
这些**本项目自己**的可移植别名，是「平台前置」而不是依赖，不算「外部提供方」。
除此之外，**任何**一个头都必须能自给自足。

已知需要外部工具链、本门禁**跳过**的头（它们的 include 是本来就该由调用方提供的）：
带 ncnn / OpenCV / SeetaFace / OpenBLAS / MKL 的那些 —— 本机 WSL 没有对应的
Linux 库或头，编不过不是「不自包含」而是「环境缺」。名单在 SKIP_SUBSTR 里，
**并且这份名单本身会被报告出来**，避免「跳过」变成「藏起来」。

用法
----
    python tools/check_header_selfcontained.py            # 全扫
    python tools/check_header_selfcontained.py --list     # 只列文件
    python tools/check_header_selfcontained.py <文件路径>  # 扫指定文件
    python tools/check_header_selfcontained.py --selfcheck
"""
from __future__ import print_function

import io
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WSL_DIST = 'Ubuntu-20.04'
MNT_ROOT = '/mnt/d/ZQCNN'

# 这些头依赖本机 WSL 没有的第三方工具链，编不过不等于「不自包含」。
# **名单会被打印出来**，不藏着。
SKIP_SUBSTR = [
    'ncnn', 'opencv', 'seetaface', 'SeetaFace', 'openblas', 'mkl',
    'opencv_stub', 'gl_stub',
]

# 需要的前置：只有这个是「平台前置」而不是「依赖」
PREAMBLE = '#include "ZQ_CNN_CompileConfig.h"\n'


def candidate_files():
    out = []
# MNN 转换器那份拷贝（附录 IL）也在扫：它虽然**不在任何构建里**，
    # 但同目录的 ZQ_CNN_Layer.h 上一轮同步过溢出守卫，说明这目录是**半维护**的；
    # 「半维护」最需要的就是这种自动检查。
    for sub in ('ZQCNN', 'ZQlibFaceID', 'ZQ_GEMM', 'ZQCNN_to_MNN/converter/source'):
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.h'):
                continue
            if fn == 'ZQ_CNN_CompileConfig.h':
                continue
            if any(s in fn for s in SKIP_SUBSTR):
                continue
            if any(s in sub for s in SKIP_SUBSTR):
                continue
            out.append(os.path.join(sub, fn))
    return out


def run_wsl(script):
    p = subprocess.run('wsl -d %s -- bash -s' % WSL_DIST, shell=True,
                       input=script.encode('utf-8'), capture_output=True)
    return ((p.stdout or b'').decode('utf-8', 'replace')
            + (p.stderr or b'').decode('utf-8', 'replace'))


# 这些是**本机 WSL 没有的第三方库**，编不过是环境问题不是代码问题。
# 必须和「用了某类型却没 include 它」区分开，否则报告会把两类混在一起。
THIRD_PARTY_MARKERS = [
    'caffe/', 'ncnn/', 'ncnn.h', 'opencv2/', 'opencv.hpp', 'mkl.h', 'mkl_',
    'SeetaFace', 'seetaface', 'ZQlib/', 'cblas.h', 'lapacke.h',
    # Windows-only 的第三方 SDK 头，本机 WSL 上不可能有
    'facedetect-dll.h', 'caffe/', 'opencv2/',
]


def _classify(out):
    m = re.search(r'fatal error:\s*([^:\n]+):\s*No such file or directory', out)
    if m:
        miss = m.group(1).strip()
        for t in THIRD_PARTY_MARKERS:
            if t in miss:
                return 'env', ('缺第三方头 %s（本机 WSL 没装，属环境问题）' % miss)
        return 'not-self-contained', out.replace('fatal error:', '缺头').strip()[:400]
    return 'not-self-contained', out.strip()[:400]


def check_one(rel):
    """返回一个 (状态, 详情)。状态 ∈ ok / not-self-contained / error。"""
    src = ('#include "ZQ_CNN_CompileConfig.h"\n'
           '#include "%s/%s"\n'
           'int main() { return 0; }\n' % (MNT_ROOT, rel.replace(os.sep, '/')))
    script = (
    'MNT=%s\n' % MNT_ROOT +
        'cat > /tmp/_zq_selfcontained.cpp <<\'ZQEOT\'\n'
        + src.replace('\\', '\\\\') +
        'ZQEOT\n'
        'g++ -fsyntax-only -std=c++11 -mavx2 -mfma -fopenmp '
        '-I$MNT -I$MNT/ZQCNN -I$MNT/ZQ_GEMM -I$MNT/ZQlibFaceID '
        '-I$MNT/3rdparty/include -I$MNT/3rdparty/include/ZQlib '
        '/tmp/_zq_selfcontained.cpp > /tmp/_zq_sc.log 2>&1\n'
        'echo "__RC=$?"\n'
        'head -4 /tmp/_zq_sc.log\n'
    )
    # 注意：**不能**写成 `gcc ... 2>&1 | head -N`。
    # 告警一多，head 先退出，gcc 收到 SIGPIPE 以 **141** 退出，
    # 而 PIPESTATUS[0] 拿到的正是这个 141 ——
    # 于是「告警很多的头」被误判成「不是自包含的」。
    # 实测 ZQ_CNN_MouthDetector.h / ZQ_FaceDatabaseMaker.h 两个**本来就自包含**的头
    # 正是这样被报成 FAIL 的。
    # 所以：先把全部输出落盘、再取 rc、最后才 head。
    out = run_wsl(script)
    m = re.search(r'__RC=(-?\d+)', out)
    if not m:
        return 'error', out.strip()[:300]
    rc = int(m.group(1))
    if rc == 0:
        return 'ok', ''
    return _classify(out.replace('__RC=%d' % rc, ''))


SELFCHECK = [
    # (说明, 头内容, 期望 rc)
    ('自给自足的头', 'struct A { int x; };\n', 0),
    ('用了没 include 的类型', 'struct A { B* p; };\n', 1),
    ('include 了提供方就自足',
     'struct B { int y; };\nstruct A { B* p; };\n', 0),
    # 注意：原来这条写的是「用 __max 而不定义」，但 `ZQ_CNN_CompileConfig.h`
    # **本来就定义** __max/__min/__int64（非 MSVC 的可移植别名），
    # 所以那条期望 rc=1 是错的，实测 rc=0 —— 门禁自己的期望写错了。
    # 换成真正缺失的标准库前置：
    ('缺 <vector>', 'struct A { std::vector<int> v; };\n', 1),
]


def selfcheck():
    """用一个 4 例的最小样本自测判定本身。

    重点是**阴性对照**（第 1、3 例）：只测阳性的话，
    「永远判失败」也能全绿。
    """
    bad = 0
    for name, body, expect in SELFCHECK:
        src = ('#include "ZQ_CNN_CompileConfig.h"\n'
               + body
               + 'int main() { return 0; }\n')
        script = (
        'MNT=%s\n' % MNT_ROOT +
            "cat > /tmp/_zq_sc.cpp <<'ZQEOT'\n" + src + 'ZQEOT\n'
            'g++ -fsyntax-only -std=c++11 -I$MNT -I$MNT/ZQCNN -I$MNT/ZQ_GEMM '
            '-I$MNT/3rdparty/include -I$MNT/3rdparty/include/ZQlib '
            '/tmp/_zq_sc.cpp > /tmp/_zq_sc.log 2>&1\n'
            'echo "__RC=$?"\n'
            'head -3 /tmp/_zq_sc.log\n'
        )
        out = run_wsl(script)
        m = re.search(r'__RC=(-?\d+)', out)
        got = int(m.group(1)) if m else -99
        ok = (got == 0) == (expect == 0)
        print('  [%s] %-34s expect rc=%s, got %s'
              % ('self-OK' if ok else 'self-MISMATCH', name, expect, got))
        if not ok:
            bad += 1
    if bad:
        print('selfcheck FAILED: %d / %d mismatch' % (bad, len(SELFCHECK)))
        return 1
    print('selfcheck OK: %d cases' % len(SELFCHECK))
    return 0


def main(argv):
    if '--selfcheck' in argv:
        return selfcheck()
    if '--list' in argv:
        for f in candidate_files():
            print(f)
        return 0
    files = [a for a in argv[1:] if not a.startswith('-')]
    if not files:
        files = candidate_files()
    nbad = 0
    nerr = 0
    nenv = 0
    nok = 0
    for f in files:
        rel = os.path.relpath(f, ROOT)
        st, detail = check_one(rel)
        if st == 'ok':
            nok += 1
        elif st == 'error':
            nerr += 1
            print('ERR  %s' % rel)
            print('       %s' % detail.replace(chr(10), chr(10) + '       '))
        elif st == 'env':
            nenv += 1
            print('ENV  %s  %s' % (rel, detail))
        else:
            nbad += 1
            print('FAIL %s  **不是自包含的**' % rel)
            print('       %s' % detail.replace(chr(10), chr(10) + '       '))
    print('自包含 OK %d / 失败 %d / 环境缺第三方库 %d / 其他错 %d'
          % (nok, nbad, nenv, nerr))
    if nenv:
        print('注意：%d 个是**环境缺第三方库**（ncnn / caffe / opencv / ZQlib），'
              '不是「不自包含」，不判失败。' % nenv)
    if nerr:
        print('注意：%d 个是**其他错**（门禁自己没跑通之类），需要查。' % nerr)
    if nbad:
        print('**%d 个头不是自包含的** —— 用到某类型就得 include 它的定义，'
              '不能指望调用方的 include 顺序' % nbad)
        return 1
    print('头文件自包含性: OK')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
