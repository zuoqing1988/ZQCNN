#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""门禁：`#pragma omp parallel for` 的循环体里，thread_id 不能是**字面 0**。

起因（2026-10-07，附录 IZ）
----------------------------
`ZQ_CNN_MTCNN*.h` 里有一份**手写维护的不变量**（见
`ZQ_CNN_MTCNN_Interface.h:713-719` 的注释）：

    其余 5 处都在 `#pragma omp parallel for num_threads(thread_num)` 里，
    `thread_id < thread_num` 有保证 —— **只有这一处裸奔**。

也就是说：`int thread_id = 0;` 只有在**不在并行区里**的时候才是对的
（单线程，索引必然是 0）；一旦被挪进
`#pragma omp parallel for` 循环体，所有线程就会共用同一个 scratch 缓冲
和同一个网络对象 —— 数据竞争 + 结果错乱，而且**编译器不会报**。

**这条不变量没有任何东西在守。** 本门禁把它变成机械判定。

为什么这条值得单独一道门禁
------------------------
这一族已经出过两次真实问题：
* 附录 II.1：`Forward` 用 `lnet[thread_id]`、读 blob 却用 `lnet[0]`；
* 附录 II.5：串行支路用 `omp_get_thread_num()` 索引大小恰好是 `thread_num`
  的容器。
两次都因为"仓内 sample 的 thread_num 全被夹成 1"而**跑不出来**（附录 IV）。

本轮踩过的两次坑也记在这里：初稿我以为 `MTCNN.h:772` 与 `_Interface.h:719`
是两处漏改，逐个打开才发现**两处都在 `if (thread_num <= 1)` 分支里**，
`thread_id = 0` 是**正确**的。所以本门禁必须**精确到"是否在并行区内"**，
不能只看"出现了 thread_id = 0"。
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCAN_DIRS = ('ZQCNN', 'ZQ_GEMM', 'ZQlibFaceID', 'SamplesZQCNN', 'SamplesZQlibFaceID')
SKIP_DIRS = {'3rdparty', 'build_x64', 'cmake-out-win32-x64', 'cmake-out-linux-x64'}

PARALLEL_PAT = re.compile(r'#\s*pragma\s+omp\s+parallel')
FOR_PAT = re.compile(r'\bfor\s*\(')
# 并行区里的 thread_id 取值：**字面 0**（或 0u/0L）是错的
BAD_ASSIGN_PAT = re.compile(
    r'\b(?:const\s+)?(?:int|auto)\s+thread_id\s*=\s*(?:0|0u|0U|0l|0L)\s*;')


def scan_file(path):
    """返回 [(行号, 文本)]：位于 omp parallel for 循环体里的字面 0 赋值。"""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.read().split('\n')

    hits = []
    i = 0
    n = len(lines)
    while i < n:
        if not PARALLEL_PAT.search(lines[i]):
            i += 1
            continue
        # 往后找紧跟的 for( 与它的循环体 { ... }
        j = i + 1
        depth = 0
        started = False
        first_line = None
        last_line = None
        while j < n:
            ln = lines[j]
            stripped = ln.strip()
            if not started and FOR_PAT.search(ln):
                started = True
                first_line = j
            if started:
                depth += ln.count('{') - ln.count('}')
                last_line = j
                if depth <= 0 and j > first_line:
                    break
                if depth < 0:
                    break
            j += 1
        if started and first_line is not None and last_line is not None:
            for k in range(first_line, min(last_line + 1, n)):
                if BAD_ASSIGN_PAT.search(lines[k]):
                    hits.append((k + 1, lines[k].strip()))
        i = (last_line + 1) if last_line else i + 1
    return hits


def collect():
    out = []
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames if x not in SKIP_DIRS]
            for fn in filenames:
                if fn.lower().endswith(('.h', '.hpp', '.cpp', '.cc', '.c')):
                    p = os.path.join(dirpath, fn)
                    for ln, txt in scan_file(p):
                        out.append((os.path.relpath(p, ROOT).replace('\\', '/'), ln, txt))
    return sorted(out)


def selftest():
    cases = [
        ('并行区外的 thread_id = 0（合法：单线程）',
         ['void f(){', '  if (thread_num <= 1) {',
          '    for (int i=0;i<n;i++) {', '      int thread_id = 0;', '    }', '  }', '}'],
         0),
        ('并行区里的 thread_id = 0（必须报）',
         ['void f(){', '#pragma omp parallel for num_threads(tn)',
          '  for (int i=0;i<n;i++) {', '    int thread_id = 0;', '  }', '}'],
         1),
        ('并行区里取 omp_get_thread_num（合法）',
         ['void f(){', '#pragma omp parallel for num_threads(tn)',
          '  for (int i=0;i<n;i++) {', '    int thread_id = omp_get_thread_num();', '  }', '}'],
         0),
        ('const int thread_id = 0; 也要报',
         ['void f(){', '#pragma omp parallel for',
          '  for (int i=0;i<n;i++) {', '    const int thread_id = 0;', '  }', '}'],
         1),
        ('嵌套花括号不影响判定',
         ['void f(){', '#pragma omp parallel for schedule(dynamic,1)',
          '  for (int i=0;i<n;i++) {', '    if (a) {', '      int thread_id = 0;', '    }', '  }', '}'],
         1),
        ('pragma 之后的非 for 语句不算并行区',
         ['void f(){', '#pragma omp parallel for num_threads(tn)',
          '  int q = 1;', '}'],
         0),
    ]
    import tempfile
    bad = []
    for name, src, expect in cases:
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, 't.cpp')
            with open(p, 'w', encoding='utf-8') as fh:
                fh.write('\n'.join(src) + '\n')
            got = len(scan_file(p))
        finally:
            os.remove(p)
            os.rmdir(d)
        ok = (got == expect)
        print('  [%s] %-46s expect=%d got=%d'
              % ('PASS' if ok else 'FAIL', name, expect, got))
        if not ok:
            bad.append(name)
    if bad:
        print('SELFTEST FAILED: %s' % ', '.join(bad))
        return 1
    print('selftest OK: %d cases' % len(cases))
    return 0


def main():
    if '--selftest' in sys.argv:
        return selftest()
    hits = collect()
    if hits:
        print('发现 %d 处「并行区里把 thread_id 写成字面 0」：' % len(hits))
        for f, ln, txt in hits:
            print('  * %s:%d  %s' % (f, ln, txt))
        print('')
        print('并行区内所有线程共用同一个 scratch 缓冲/网络对象 -> 数据竞争 + 结果错乱。')
        print('改成 `omp_get_thread_num()`；若这一处**本来就不在并行区**，'
              '说明判定被绕过了，请回来报。')
        return 1
    print('OK: 所有 omp parallel for 循环体里，thread_id 都不是字面 0')
    return 0


if __name__ == '__main__':
    sys.exit(main())