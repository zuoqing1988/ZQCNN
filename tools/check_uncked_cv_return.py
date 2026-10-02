#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""门禁：**返回 bool 的 OpenCV 调用，用作独立语句时必须检查返回值**。

为什么要有这个（audit_k3_20261001.md 附录 BM）
----------------------------------------------
附录 BM 修了 `ZQlibFaceID/ZQ_FaceRecognizerUtils.h` 里两处
`cv::invert(...)` 没检查返回值。`cv::invert` 对**奇异**矩阵返回 `false`
并把 dst 留成**空 `cv::Mat`**，紧接着的 `tmp.ptr<T>(j)[i]` 就是空指针解引用。

这类形状（全树扫下来只有这一处，但形状本身很典型）：

    cv::invert(transform1, tmp, cv::DECOMP_SVD);      // 返回值丢掉
    ...
    trans.ptr<TmpType>(i)[j] = tmp.ptr<TmpType>(j)[i];   // 空 Mat -> 空指针

和附录 BH 同一个动机：**修了一处，不等于只有这一处**。既然要靠人记得去普查，
不如让普查结果进门禁。

判定的函数清单
--------------
只收**返回值是 bool、且失败时会把某个 OutputArray 留成空**的那几个：

    cv::solve      求不出唯一解 / 失败 -> r 空
    cv::invert     奇异矩阵          -> dst 空
    cv::gemm       (签名上是 bool，但失败时 dst 同样是空的；一并收)

`cv::add` / `cv::warpAffine` 这类返回 `void` 或返回 Mat 引用的**不收** ——
它们不是"忘检查就崩"的形状，收进来只会制造噪声。

怎么算"用了返回值"
------------------
命中条件 = 这一行**以 `cv::xxx(` 开头**（前面只有空白）。
也就是把返回值整个丢掉、当独立语句用。
下面这些都**不算**命中，因为返回值被用了：

    if (!cv::solve(...))          // 命中规则的反面：检查了
    bool ok = cv::invert(...);    // 接住了
    return cv::invert(...);       // 直接返回
    cv::gemm(a, b, c);            // 命中（真的是丢掉）

局限（KNOWN_GAPS）
-----------------
* 只看**单行**开头。写成 `if (x) cv::invert(...); else ...;` 这种一行里
  带条件的会漏判（宁可漏判也不要误判成"有检查"）。
* 不判断"检查了是不是检查对了"（比如 `if (cv::invert(...)) { /* 用反了 */ }`）。
  这类得靠人读。
* 只覆盖源码里出现 `cv::` 限定的写法。`using namespace cv; invert(...)` 这种
  认不出来 —— 本仓库目前没有这种写法，但换个人写就有了。
* **编不了 OpenCV 的平台做不了运行时验证**：本机
  `3rdparty/opencv` 只有 Windows 的 `opencv_world342.lib`，没有 Linux 的 `.so`，
  所以这个缺陷是**读代码 + 与同仓库那份已经加固的 ZQ_CNN_FaceCropUtils.h 对照**
  定位的，没有 ASan 运行时证据。如实记在这里，不假装有。

用法:
    python tools/check_uncked_cv_return.py             # 扫，退出码 1 = 有漏
    python tools/check_uncked_cv_return.py --selfcheck  # 自测
    python tools/check_uncked_cv_return.py --list      # 只列出命中点
"""

from __future__ import print_function

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWLIST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'uncked_cv_return_allowlist.txt')

# 返回 bool 且失败会把 OutputArray 留空的 OpenCV 函数
FUNCS = ('solve', 'invert', 'gemm')

SCAN_DIRS = ('ZQCNN', 'ZQlibFaceID', '3rdparty', 'SamplesZQCNN',
             'SamplesZQlibFaceID', 'tools', 'ZQ_GEMM')
SCAN_EXT = ('.h', '.hpp', '.cpp', '.c', '.cu')

SKIP_DIR_PARTS = ('build', 'cmake-out', '.git', '3rdparty/opencv', '3rdparty/mkl_runtime')

CALL_RE = re.compile(r'^\s*cv::(' + '|'.join(FUNCS) + r')\s*\(')

KNOWN_GAPS = [
    '只认「本行以 cv::xxx( 开头」这一种形状；一行里带条件的写法会漏判。',
    '不判断检查得对不对（invert 的 bool 语义是 true=成功）。',
    '只覆盖 cv:: 限定写法。',
    '没有 ASan 运行时证据：本机没有 Linux 版 OpenCV .so（见模块 docstring）。',
]


def scan_files():
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            rel = os.path.relpath(dirpath, ROOT).replace('\\', '/')
            if any(part in rel for part in SKIP_DIR_PARTS):
                dirnames[:] = []
                continue
            for fn in filenames:
                if fn.endswith(SCAN_EXT):
                    yield os.path.join(dirpath, fn)


def read_text(path):
    """按 utf-8 读；读不了就当二进制跳过（本仓库有 GBK 的头）。"""
    try:
        with io.open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except (UnicodeDecodeError, ValueError):
        return None


def find_hits():
    """返回 [(相对路径, 行号, 行内容, 函数名)]。"""
    hits = []
    for path in scan_files():
        text = read_text(path)
        if text is None:
            continue
        rel = os.path.relpath(path, ROOT).replace('\\', '/')
        for i, line in enumerate(text.splitlines(), 1):
            # 注释行不算
            stripped = line.lstrip()
            if stripped.startswith('//') or stripped.startswith('*'):
                continue
            m = CALL_RE.match(line)
            if m:
                hits.append((rel, i, line.strip(), m.group(1)))
    return hits


def load_allowlist():
    s = set()
    if not os.path.exists(ALLOWLIST):
        return s
    with io.open(ALLOWLIST, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            s.add(line)
    return s


def selfcheck():
    """用构造出来的样例文本验判定逻辑本身，别让这个门禁自己变成"永远绿"。

    两层：
      1) 逐行判定的 11 条（正则层）
      2) **真的往被扫的目录里写一个临时文件**，跑一遍完整的 find_hits，
         确认"文件放进去就会被抓到"（遍历层）。
         只测正则是不够的：一个从不匹配的正则 + 零命中 = 永远绿的门禁。
    """
    cases = [
        ('    cv::invert(a, b, cv::DECOMP_SVD);', True),
        ('\t\tcv::solve(X, U, r, cv::DECOMP_SVD)', True),
        ('  cv::gemm(a, b, c, 1.0);', True),
        ('if (!cv::solve(X, U, r, cv::DECOMP_SVD))', False),
        ('bool ok = cv::invert(a, b, cv::DECOMP_SVD);', False),
        ('\treturn cv::invert(a, b);', False),
        ('// cv::invert(a, b);', False),
        ('    /* cv::invert(a, b); */', False),
        ('    cv::warpAffine(img, crop, transform, size);', False),
        ('    cv::resize(src, dst, size);', False),
        ('    int n = cv::solve(a, b, c);', False),
    ]
    bad = 0
    for line, want in cases:
        got = bool(CALL_RE.match(line))
        if got != want:
            bad += 1
            print('  SELFTEST FAIL  %-52r 期望 %s 实际 %s'
                  % (line, want, got))
    if bad:
        print('selfcheck: %d/%d 判定失败' % (bad, len(cases)))
        return 1
    print('selfcheck: %d 条判定全部符合预期' % len(cases))

    # --- 遍历层：临时文件必须被抓到，且跑完要清掉 ---
    probe = os.path.join(ROOT, 'tools', '_selftest_uncked_cv.cpp')
    body = u"""// 门禁自测用的临时文件，跑完即删
void f()
{
    cv::invert(a, b, cv::DECOMP_SVD);
    cv::solve(X, U, r, cv::DECOMP_SVD);
    if (!cv::gemm(a, b, c, 1.0)) { }
}
"""
    try:
        with io.open(probe, 'w', encoding='utf-8') as f:
            f.write(body)
        hits = [(r, l, t, fn) for (r, l, t, fn) in find_hits()
                if r.endswith('_selftest_uncked_cv.cpp')]
    finally:
        if os.path.exists(probe):
            os.remove(probe)
    if len(hits) != 2:
        print('  SELFTEST FAIL  遍历层：临时文件里放了 2 处，find_hits 报了 %d 处' % len(hits))
        for h in hits:
            print('      %s:%d %s' % h)
        return 1
    if os.path.exists(probe):
        print('  SELFTEST FAIL  临时文件没清掉')
        return 1
    print('selfcheck: 遍历层也通过（临时文件 2 处 -> find_hits 2 处，跑完已清理）')
    return 0


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    if '--selfcheck' in sys.argv:
        return selfcheck()
    if '--gaps' in sys.argv:
        for g in KNOWN_GAPS:
            print('  - ' + g)
        return 0

    hits = find_hits()
    allow = load_allowlist()
    if '--list' in sys.argv:
        for rel, ln, txt, fn in hits:
            mark = ' [allow]' if ('%s:%d' % (rel, ln)) in allow else ''
            print('%s:%d: %s%s' % (rel, ln, txt, mark))
        print('共 %d 处' % len(hits))
        return 0

    real = [(r, l, t, f) for (r, l, t, f) in hits
            if ('%s:%d' % (r, l)) not in allow]
    if not real:
        print('OK: 全部 cv::%s 调用的返回值都被用上了（扫了 %d 处）'
              % ('/'.join(FUNCS), len(hits)))
        return 0
    print('FAIL: %d 处丢弃了返回值的 cv:: 调用（白名单 %d 条）'
          % (len(real), len(allow)))
    for rel, ln, txt, fn in real:
        print('  %s:%d: %s   <-- 返回值是 bool，失败时输出 Mat 会是空的' % (rel, ln, txt))
    print()
    print('确实可以丢的一律写进 tools/uncked_cv_return_allowlist.txt（格式 路径:行号）')
    return 1


if __name__ == '__main__':
    sys.exit(main())
