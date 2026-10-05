#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扫描「一条语句后面粘着下一条语句」这种补丁脚本留下的痕迹 —— 附录 IR。

为什么需要它
------------
这一轮（附录 IH~IQ）用 Python 脚本批量改 C++ 源码，**四次**因为替换串里少了一个换行
而把两行粘成一行，例如：

    return false;			int C, H, W;     // 粘在一起
    }				const ZQ_CNN_Tensor4D* prob = ...   // 粘在一起

后果分两类，都很难自己看出来：

* **能编过但可读性烂**（`return true; }`）—— 编译器不报错，code review 容易滑过去；
* **编不过**（`}   const X* p = ...` 被解析成一条语句）—— 但**只有编到那个 TU 才发现**。
  而如果「编不到」是因为那段代码已经不在任何构建里，这个错误会**永久留在仓库里**。

**所以这类痕迹必须有自动检查**，而不是靠「下次编全量时顺便发现」。

判据
----
扫 `ZQCNN/*.h`、`ZQlibFaceID/*.h`、`ZQ_GEMM/*.h`、`ZQCNN_to_MNN/**/*.h`，
找**语句结束之后**紧跟（中间只有空白、没有换行）**下一个标识符**的形态：

    <语句>;  [ \\t]{2,}  <标识符>

阈值取 **2 个以上空白字符**：1 个空格的 `} ` / `return true; }` 那类是本仓库
**原有的**书写风格（`ZQ_CNN_Tensor4D.h` / `_NCHWC.h` 里就有好几处），
把它们一起报出来只会让这条规则永远红 —— 而**永远红的规则等于没有规则**
（AGENTS.md 第 20 条）。要动那些原有风格是另一件事、单独做。

用法
----
    python tools/check_stmt_joins.py            # 扫
    python tools/check_stmt_joins.py --selfcheck  # 先自测
"""
from __future__ import print_function

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = ['ZQCNN', 'ZQlibFaceID', 'ZQ_GEMM', os.path.join('ZQCNN_to_MNN', 'converter', 'source')]

# 句子结束符 -> 后面粘了 >=2 个空白再接标识符
JOIN = re.compile(r'(?:^|[;{}])\s*[^\n;{}]*[;{}][ \t]{2,}[A-Za-z_}\\]')
# 注释行、行尾注释、字符串字面量内部的 ; 不算
SKIP_LINE = re.compile(r'^\s*(//|\*|/\*)')


# ---------------------------------------------------------------------------
# 这条检查防的是什么、不是什么
# ---------------------------------------------------------------------------
# 刚写第一版时以为「语句粘连」会改变语义 —— **不会**：
# `}` 后接一条声明、`return false;` 后接一条声明，C++ 都解析成**两条**语句，
# 照样编过。本轮 4 次粘连里，**唯一真的编不过**的那次是
# `int C, H, W;` 被分割脚本吃掉了首字母变成 `nt C, H, W;` ——
# 那是**标识符被截断**，编译器能抓，跟粘连无关。
#
# 所以这条规则的定位是**可读性 / 一致性**，不是正确性：
# 粘连的行在 code review 里极容易滑过去（读起来像一行），而下一轮脚本
# 再往这行里插东西时就更容易连锁出错。
#
# 因为「能编过」，如果把仓库**原有**的同类写法也报出来，这条规则就会**永远红** ——
# 而永远红的规则等于没有规则（AGENTS.md 第 20 条）。所以下面显式列出
# 原有站点的白名单，并写明理由；新引入的粘连一律红。
KNOWN = [
    # (相对路径, 行首片段, 数量) —— 都是本轮之前就有的书写风格，不动
    ('ZQCNN/ZQ_CNN_MTCNN_AspectRatio.h', '}void SetLimit(', 1),
    ('ZQCNN/ZQ_CNN_MTCNN_Interface.h', '}void SetLimit(', 1),
    ('ZQCNN/ZQ_CNN_MTCNN_NCHWC.h', '}void SetLimit(', 1),
    ('ZQCNN/ZQ_CNN_VideoFaceDetection_Interface.h', '}if (IOU > overlap_threshold)', 1),
    # ZQlibFaceID/ZQ_FaceDatabase*.h 的 `score_begin[pp] = s;  s += ...`
    # 是**刻意的列对齐**，不是粘连（两边的空白是为了让 += 对齐）。
    ('ZQlibFaceID/ZQ_FaceDatabase.h', 'score_begin[pp]=s;', 1),
    ('ZQlibFaceID/ZQ_FaceDatabase.h', 'flag_begin[pp]=f;', 1),
    ('ZQlibFaceID/ZQ_FaceDatabaseCompact.h', 'score_begin[pp]=s;', 1),
    ('ZQlibFaceID/ZQ_FaceDatabaseCompact.h', 'flag_begin[pp]=f;', 1),
]


def is_known(rel, frag):
    for k_rel, k_frag, k_cnt in KNOWN:
        if rel.replace(os.sep, '/') == k_rel and frag.replace(' ', '').replace(chr(9), '') \
                .startswith(k_frag.replace(' ', '')):
            return True
    return False



def candidates():
    out = []
    for d in DIRS:
        p = os.path.join(ROOT, d)
        if not os.path.isdir(p):
            continue
        for fn in sorted(os.listdir(p)):
            if fn.endswith('.h'):
                out.append(os.path.join(d, fn))
    return out


def scan_text(text):
    """返回 [(行号, 片段)]；跳过注释行与预处理指令。"""
    bad = []
    lines = text.split(chr(10))
    for i, ln in enumerate(lines):
        if SKIP_LINE.match(ln):
            continue
        if ln.lstrip().startswith('#'):
            continue
        for m in JOIN.finditer(ln):
            frag = ln[m.start():]
            # 去掉行尾注释再判一次：`a;  \t// b` 不是粘连
            frag = re.sub(r'//.*$', '', frag)
            if JOIN.search(frag):
                bad.append((i + 1, frag.strip()[:80]))
                break
    return bad


def selfcheck():
    cases = [
        ('正常：一行一条', 'int a = 1;\nint b = 2;\nreturn a;\n', 0),
        ('粘连：return 后接声明', 'return false;\t\t\tint C, H, W;\n', 1),
        ('粘连：} 后接声明', '\t\t}\t\t\tconst T* p = q;\n', 1),
        ('原有风格：单个空格不算', 'return true; }\n', 0),
        ('注释行不算', '// return false;\t\tint x;\n', 0),
        ('预处理指令不算', '#define X 1;\t\tint y;\n', 0),
        ('行尾注释不算', 'int a;\t\t// b\n', 0),
    ]
    n = 0
    for name, text, expect in cases:
        got = len(scan_text(text))
        ok = got == expect
        print('  [%s] %-28s expect %d, got %d'
              % ('self-OK' if ok else 'self-MISMATCH', name, expect, got))
        if not ok:
            n += 1
    if n:
        print('selfcheck FAILED: %d / %d mismatch' % (n, len(cases)))
        return 1
    print('selfcheck OK: %d cases' % len(cases))
    return 0


def main(argv):
    if '--selfcheck' in argv:
        return selfcheck()
    nbad = 0
    nfile = 0
    for rel in candidates():
        path = os.path.join(ROOT, rel)
        try:
            with io.open(path, 'r', encoding='utf-8-sig') as f:
                text = f.read()
        except (IOError, OSError, UnicodeDecodeError):
            continue
        nfile += 1
        allbad = scan_text(text)
        bad = []
        used = {}
        for ln, frag in allbad:
            k = (rel.replace(os.sep, '/'), re.sub(r'[ \t]+', '', frag))
            hit = False
            for k_rel, k_frag, k_cnt in KNOWN:
                if k[0] == k_rel and k[1].startswith(k_frag.replace(' ', '')):
                    # 配额必须按 (文件, 片段) 计 —— 只按文件计的话，
                    # 同一个文件里的两条白名单会互相吃掉配额（第一条就报成「新增」）。
                    kk = (k_rel, k_frag)
                    n = used.get(kk, 0)
                    if n < k_cnt:
                        used[kk] = n + 1
                        hit = True
                    break
            if not hit:
                bad.append((ln, frag))
        if bad:
            nbad += 1
            print('FAIL %s' % rel.replace(os.sep, '/'))
            for ln, frag in bad[:6]:
                print('       line %d: %s' % (ln, frag))
    print('扫了 %d 个头，%d 个有**新引入**的粘连行' % (nfile, nbad))
    print('（原有站点 %d 处按白名单放行，理由写在文件头）' % sum(c for _, _, c in KNOWN))
    if nbad:
        print('**这些行是补丁脚本丢换行造成的**（本轮栽了 4 次）。')
        print('其中「能编过但可读性烂」的比「编不过」更危险 —— 后者至少会报错。')
        return 1
    print('语句粘连: OK')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
