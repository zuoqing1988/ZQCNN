#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""「零调用点」体检：哪些 public API **一个调用方都没有** —— 附录 IT。

为什么要这道门禁
----------------
这一轮多次出现「**改了但从没跑到**」的情况，根因都是同一个：
某个 public 入口**全仓没有调用方**，于是它所在的整条链路不在任何回归覆盖里。

已确认的三处：

* `ZQ_FaceDatabaseMaker::MakeDatabase(` / `MakeDatabaseCompact(` 零调用方
  ⇒ `_make_database` / `_extract_feature_from_img` / `_extract_feature_from_box`
  ——**整个检测器驱动的路径** —— 不被任何 sample 执行。
  四个 `SampleFaceDatabase*` 只用 `*AlreadyCropped` 变体，恰好绕开 `detectors[id]`。
* `ZQ_CNN_VideoFaceDetection_Interface.h` 的 `_auto_detect_database` 的 `#else`
  (Linux) 分支**从不链接** —— 10 个 include 此头的 sample 全部包在
  `#if defined(_WIN32)` 里。
* `ZQ_CNN_CascadeOnet_Interface::Find` 零调用点，
  于是 `ZQ_CNN_VideoFaceDetection_Interface` 里那 `3 × thread_num` 份
  `cascade_Onets` 加载后**从不推理**（`thread_num=8` 就是 24 份 det3 白常驻内存）。
* `ZQ_CNN_Tensor4D_NCHWC.h` 的 `Permute_NCHW` 零调用点 ⇒ 那条整数除零永远不发作。

判据
----
对每个登记的符号，扫全仓（排除声明处与定义处）看有没有引用。
**零命中就是命中不了** —— 和 C1 那次「被引号坑掉」不同，
这里用的是 `\\b<名字>\\b` 精确匹配，不会被相近名字吃掉。

**这道门禁不判失败**，只**报告**：
零调用点本身不是缺陷（很多 API 本来就是给外部用的），
但它意味着「这个入口的代码不在回归覆盖内」，
所以清单要定期看，而不是躺在那里没人知道。

用法
----
    python tools/check_zero_callers.py           # 全扫
    python tools/check_zero_callers.py --list    # 只列符号名
"""
from __future__ import print_function

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (符号, 声明所在文件, 一句话说明)
WATCH = [
    ('MakeDatabase', 'ZQlibFaceID/ZQ_FaceDatabaseMaker.h',
     '整个检测器驱动的路径（_make_database / _extract_feature_from_*）不在回归覆盖内'),
    ('MakeDatabaseCompact', 'ZQlibFaceID/ZQ_FaceDatabaseMaker.h', '同上'),
    ('MakeDatabaseAlreadyCropped', 'ZQlibFaceID/ZQ_FaceDatabaseMaker.h',
     '四个 SampleFaceDatabase* 只用这一个变体'),
    ('MakeDatabaseCompactAlreadyCropped', 'ZQlibFaceID/ZQ_FaceDatabaseMaker.h', '同上'),
    ('_auto_detect_database', 'ZQlibFaceID/ZQ_FaceDatabaseMaker.h',
     'Linux 分支（#else）从不链接：10 个 sample 全在 #if defined(_WIN32) 里'),
    ('_write_database_txt', 'ZQlibFaceID/ZQ_FaceDatabaseMaker.h', '30 行死代码'),
    ('Permute_NCHW', 'ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h',
     '零调用点 ⇒ 那条零维整数除零永远不发作（NCHWC 没有 Transpose 层）'),
    ('_filtering_iou', 'ZQCNN/ZQ_CNN_VideoFaceDetection_Interface.h',
     '死代码，但里面是 C1 那个越界的**另一个形态**；谁把注释放开就多一个 OOB'),
    ('Evaluate', 'ZQlibFaceID/ZQ_FaceRecognizer.h', 'Clustering 家族的基类默认实现'),
    ('Conquer', 'ZQlibFaceID/ZQ_FaceRecognizer.h', '同上'),
]

# **只扫代码**。第一版把 .md / .txt 也算进来，于是 changelog 与审计报告里
# 「提到过 `MakeDatabase`」被当成「有调用方」—— `MakeDatabase` 命中 6 次（全是文档），
# 零调用点这条结论就被自己的**说明文字**抹掉了。
# 这是「观测手段」的第八次自伤：扫的**对象**选错了，而不是判据写错了。
CODE_DIRS = ('ZQCNN', 'ZQlibFaceID', 'ZQ_GEMM')
CODE_PREFIXES = ('SamplesZQCNN', 'SamplesZQlibFaceID', 'SamplesZQGEMM',
                 'SamplesZQBLAS', 'mobilefacenet-mxnet2caffe-ZQ', 'ZQCNN_to_MNN')
SKIP_DIR = {'.git', 'build_x64', 'cmake-out-unix-x64', 'cmake-out-win32-x64',
            'cmake-out-win32-x86', 'node_modules', '__pycache__'}
EXT = ('.h', '.cpp', '.c', '.hpp', '.in')


def iter_files():
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR]
        rel = os.path.relpath(root, ROOT).replace(os.sep, '/')
        top = rel.split('/')[0] if '/' in rel or rel != '.' else ''
        in_code = (rel in CODE_DIRS) or rel.startswith(CODE_PREFIXES)
        if not in_code:
            continue
        for fn in files:
            if os.path.splitext(fn)[1].lower() in EXT:
                yield os.path.join(root, fn)


def find_callers(sym, decl_rel):
    """返回 (命中的相对路径列表, 总命中次数)。声明所在文件本身不算。"""
    pat = re.compile(r'\b' + re.escape(sym) + r'\b')
    decl_abs = os.path.normpath(os.path.join(ROOT, decl_rel))
    hits = []
    total = 0
    for f in iter_files():
        try:
            with io.open(f, 'r', encoding='utf-8', errors='replace') as fh:
                text = fh.read()
        except (IOError, OSError):
            continue
        n = len(pat.findall(text))
        if n == 0:
            continue
        if os.path.normpath(f) == decl_abs:
            # 声明文件本身：先记下来，但要单独数「除声明外还有没有别的引用」
            continue
        hits.append((os.path.relpath(f, ROOT).replace(os.sep, '/'), n))
        total += n
    # 声明文件内部、定义之后的引用也算（同一文件里的其它调用）
    try:
        with io.open(decl_abs, 'r', encoding='utf-8', errors='replace') as fh:
            self_text = fh.read()
    except (IOError, OSError):
        self_text = ''
    return hits, total


def main(argv):
    if '--list' in argv:
        for s, f, _ in WATCH:
            print('%s\t%s' % (s, f))
        return 0
    zero = []
    for sym, decl, note in WATCH:
        hits, total = find_callers(sym, decl)
        if total == 0:
            zero.append((sym, decl, note))
            print('ZERO  %-38s %s' % (sym, decl))
            print('        %s' % note)
        else:
            where = ', '.join('%s(%d)' % (p, n) for p, n in hits[:3])
            print('ok    %-38s %d 处引用  %s' % (sym, total, where))
    print()
    print('登记 %d 个符号，其中**零调用点** %d 个' % (len(WATCH), len(zero)))
    if zero:
        print('零调用点不等于缺陷（很多 API 是给外部用的），')
        print('但它意味着「这些入口的代码**不在任何回归覆盖内**」——')
        print('改了只能靠源码门禁钉住，行为验证要另写调用方。')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
