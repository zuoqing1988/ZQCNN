#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""禁止用 `strcmp(typeid(T).name(), "float"/"double")` 判类型（附录 IE）。

为什么这是**高危**而不是风格问题
------------------------------
`typeid(T).name()` 返回的**字符串是实现定义的**，两个主流 ABI 完全不同：

| 编译器 | `typeid(float).name()` | `typeid(double).name()` |
| --- | --- | --- |
| MSVC | `"float"`  | `"double"` |
| GCC / Clang（Itanium ABI） | `"f"` | `"d"` |

所以
```cpp
if      (strcmp(typeid(T).name(), "float")  == 0) { /* float  */ }
else if (strcmp(typeid(T).name(), "double") == 0) { /* double */ }
else return false;                       // <-- GCC 上 T 是 double，于是走这里
```
在 **GCC 上恒成立地走进 `return false`**：
`PCG` / `PCG_sparse_unsquare` / `PCG_BQP` / `ZQ_taucs_ccs_matrix_time_vec` 等
**每一个函数都立刻返回 false，一个数都算不出来**，而且**不报错、不崩**。

2026-10-06 实测（附录 IE.1）：`ZQ_PCGSolver::PCG` 在 gcc 9.4 上
对任意 SPD 系统 `ret=false, it=-1`，残差 ≈ ||f||∞（一步没走）。
换成 `std::is_same<T, double>::value` 之后残差 1e-16。

这与「同一个调用在两个平台上行为不同」是同一类，而本项目的目标里
「windows 和 linux 都能完全跑通」是硬要求 —— 所以它归**高危**。

正确写法：`std::is_same<T, float>::value` / `std::is_same<T, double>::value`
（需要 `#include <type_traits>`）。它在两个 ABI 上都是**编译期**判定的。

用法
----
    python tools/check_typeid_name.py              # 扫 ZQlib 头
    python tools/check_typeid_name.py --selfcheck  # 先自测
    python tools/check_typeid_name.py <文件路径>    # 扫指定文件

注意：`sprintf("...", typeid(T).name())` 这种**只用来打印**的地方不算缺陷
（打印出来的名字在不同 ABI 上不一样，但不改变行为），本门禁**不报**它。
"""
from __future__ import print_function

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(ROOT, '3rdparty', 'include', 'ZQlib')

# 只匹配**参与条件判断**的那种：后面一定跟着 ==0 / !=0
BAD_RE = re.compile(
    r'strcmp\(\s*typeid\(\s*\w+\s*\)\.name\(\)\s*,\s*"(?:float|double)"\s*\)\s*(?:==|!=)\s*0')
# 只用来打印的（不算缺陷）
PRINT_RE = re.compile(r'sprintf(?:_s)?\s*\([^;]*typeid\(\s*\w+\s*\)\.name\(\)')


def scan_text(text, label):
    """返回 (命中的行号列表, [(行号, 说明)])。"""
    hits, bad = [], []
    for i, line in enumerate(text.split('\n')):
        code = line.split('//')[0]
        if PRINT_RE.search(code):
            continue
        if BAD_RE.search(code):
            hits.append(i + 1)
            bad.append((i + 1, '%s:%d  用 strcmp(typeid(T).name(), ...) 判类型 -> '
                               'GCC 上恒进 else 分支（IEEE ABI 返回 "f"/"d"），'
                               '应改 std::is_same<T, float/double>::value'
                        % (label, i + 1)))
    return hits, bad


GOOD = """\
#include <type_traits>
template<class T> bool f() {
    if (std::is_same<T, float>::value) return true;
    else if (std::is_same<T, double>::value) return true;
    return false;
}
"""

BAD = """\
#include <typeinfo>
#include <string.h>
template<class T> bool f() {
    if (strcmp(typeid(T).name(), "float") == 0) return true;
    else if (strcmp(typeid(T).name(), "double") == 0) return true;
    return false;
}
"""

# 只打印、不判断 —— **不算缺陷**，门禁必须放过它
PRINT_ONLY = """\
template<class T> void f(char* out) {
    sprintf(out, "%s", typeid(T).name());
}
"""


def selftest():
    ok = True
    for name, txt, want_hits in (
            ('已改用 std::is_same（合格）', GOOD, 0),
            ('**strcmp 判类型**（IE 那条）', BAD, 2),
            ('只用来打印（不算缺陷）', PRINT_ONLY, 0)):
        hits, bad = scan_text(txt, '<selftest>')
        mark = 'OK ' if len(hits) == want_hits else '**BAD**'
        if len(hits) != want_hits:
            ok = False
        print('  %s %-30s 命中 %d 处（期望 %d）' % (mark, name, len(hits), want_hits))
    return ok


def main(argv):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

    if '--selfcheck' in argv:
        print('check_typeid_name 自测：')
        if not selftest():
            print('**自测没过 —— 匹配逻辑已坏，先修它再谈别的**')
            return 1
        print('自测通过')
        return 0

    files = [a for a in argv[1:] if not a.startswith('-')]
    if not files:
        if not os.path.isdir(DEFAULT_DIR):
            print('**找不到 %s**' % DEFAULT_DIR)
            return 1
        files = [os.path.join(DEFAULT_DIR, f) for f in sorted(os.listdir(DEFAULT_DIR))
                 if f.endswith('.h')]

    total, bad_all = 0, []
    for f in files:
        text = io.open(f, encoding='utf-8', errors='replace').read()
        hits, bad = scan_text(text, os.path.relpath(f, ROOT))
        total += len(hits)
        bad_all += bad
        if hits:
            print('%-46s %d 处' % (os.path.relpath(f, ROOT), len(hits)))

    for _, msg in bad_all:
        print('      !! %s' % msg)
    print('合计 %d 处用 strcmp(typeid(T).name(), ...) 判类型' % total)
    if not files:
        print('**一个文件都没扫到 —— 匹配逻辑或路径多半坏了**')
        return 1
    if bad_all:
        print('**GCC/Clang 上 typeid(double).name() 是 "d" 而不是 "double"，'
              '这些分支会全部落到 else（多半是 return false）**')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
