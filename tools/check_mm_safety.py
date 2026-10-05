#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SIMD 内层的两条源码级判定（附录 IX）：

  1. `zq_final_sum_q` 的项数 == 向量 lane 数，且不越 `q[]`；
  2. **裸栈数组不得被 `zq_mm_store_ps` 写**（store 要对齐，x86-64 ABI 不保证）；
  3. **`*buffer = _aligned_malloc(...)` 之后必须查返回值**，
     且必须在更新 `*buffer_len` **之前**（否则下次调用会跳过重新分配，
     拿着 `*buffer == 0` 去算 —— 真正的空指针解引用发生在下一次，不在这一行）。

为什么需要这个门禁
------------------
`ZQCNN/layers_c/zq_cnn_*_32f_align_c.c` 用 `#define` 把同一个 raw 头**反复实例化**，
每次换一套 `zq_mm_load_ps` / `zq_base_type` / `zq_mm_align_size`。

**第 1 条**（IX.1）：横向归约宏统一写成 `#define zq_final_sum_q (q[0]+...+q[N-1])`，
而 `q` 在各个 raw 头里一律声明成 `ZQ_DECLSPEC_ALIGN32 zq_base_type q[8];`。
**N 必须正好等于这一节的 lane 数**（= 同一节的 `zq_mm_align_size`）：
小了是丢项，大了是**越界读**。
2026-10-05 实测抓到的那条就是大了：ARM NEON + `__ARM_NEON_FP16` 那一节写成 9 项
`q[0]..q[8]`，而 `q` 只有 8 个元素 —— 越界读，
且 x86 两套构建**永远编不到**那一节（`zq_base_type` 在 x86 上是 `float`，
SSE/AVX 两节分别是 4 项 / 8 项，都对），所以任何运行时门禁都盖不住。

**第 2 条**（IX.3）：`zq_mm_store_ps` 在 AVX 那一档是 `_mm256_store_ps`，
**要求 32 字节对齐**；x86-64 ABI 只保证栈 16 字节对齐。
`zq_cnn_lstm_32f_align_c_raw.h` 里那四个 `zq_base_type buffer_*[zq_mm_align_size]`
是**裸栈数组**却被它写 —— 平时不崩只是因为栈恰好对上了，换个调用点就可能崩。
同批 raw 头里的 `q[8]` 本来就写了 `ZQ_DECLSPEC_ALIGN32`，那四个漏了。
**ASan 那一轴全绿**（ASan 不管对齐），只有 UBSan 看得见（与附录 IW.13 同一件事）。

判定
----
1. 对每个 `#define zq_final_sum_q (...)`：
   * 项数 == 该节在生效的 `zq_mm_align_size`（lane 数）
   * 下标恰好是 0..项数-1（不缺项、不重复）
   * 项数 <= 该节 `#include` 进去的 raw 头里 `zq_base_type q[N]` 的 N（不许越界）
2. 对每个 `*_raw.h`：凡是**被 `zq_mm_store_ps(...)` 当第一个实参写**的
   `zq_base_type` 数组，声明处必须带对齐属性（`ZQ_DECLSPEC_ALIGN*`）。
3. 对 `ZQCNN/layers_c/` 与 `ZQCNN/layers_nchwc/` 下每个头：
   每个 `*buffer = _aligned_malloc(...)` 之后的 5 行内必须出现 `*buffer == 0` 检查。

用法
----
    python tools/check_mm_safety.py                 # 扫默认文件
    python tools/check_mm_safety.py --selfcheck     # 先自测（门禁里常驻这一组）
    python tools/check_mm_safety.py <文件路径>      # 扫指定文件

自测样本里**必须有故意不合格的项**（AGENTS.md「写检查类工具的四条硬规矩」第 1 条），
否则一个匹配逻辑坏掉的扫描器会绿着挡住后续所有同类缺陷。
"""
from __future__ import print_function

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS_C = os.path.join(ROOT, 'ZQCNN', 'layers_c')
# NCHWC 那一族（layers_nchwc）同样在两个构建里，同样有这三条问题 ——
# 第一次只扫 layers_c 时 14 处漏网全在这一族里（附录 IX.6）。
LAYERS_NCHWC = os.path.join(ROOT, 'ZQCNN', 'layers_nchwc')

# 一个 .c 里可能出现的横向归约宏（名字现在只有一个，但按前缀扫，
# 免得以后新增一层又从头踩一遍）
FINAL_SUM_RE = re.compile(r'^\s*#\s*define\s+(zq_\w*final_sum\w*)\s*\((.*)\)\s*$')
ALIGN_RE = re.compile(r'^\s*#\s*define\s+zq_mm_align_size\s+(\d+)\s*$')
QDECL_RE = re.compile(r'zq_base_type\s+q\[(\d+)\]')
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
TERM_RE = re.compile(r'\bq\[(\d+)\]')


# ---- 判定 2：被 zq_mm_store_ps 写的栈数组必须带对齐属性（附录 IX.3）----
# 只在 *_raw.h 里查：那些文件才是"裸函数体"，数组一定在栈上。
RAW_DECL_RE = re.compile(
    r'^\s*(?P<align>ZQ_DECLSPEC_ALIGN\w+\s+)?zq_base_type\s+(?P<name>\w+)\s*\[')
STORE_RE = re.compile(r'zq_mm_store_ps\s*\(\s*(?P<name>\w+)\s*,')


# ---- 判定 3/4：分配之后必须查返回值（附录 IX.5 / IX.6）----
# 3a（IX.5）：`*buffer = _aligned_malloc(...)` 之后必须查 `*buffer == 0`。
#     这一条单独盯：它的危害**不在这一行**，而在**下一次调用** ——
#     不查就更新 `*buffer_len`，下次进来跳过重新分配，拿着空指针去算。
# 3b（IX.6）：任何 `名字 = _aligned_malloc(...)` 之后 12 行内必须有该名字的判空。
BUF_MALLOC_RE = re.compile(r'\*buffer\s*=\s*(?:\(\s*zq_base_type\s*\*\s*\)\s*)?_aligned_malloc')
ANY_MALLOC_RE = re.compile(
    r'^\s*(?:[\w:\*]+\s+)?(\w+)\s*=\s*(?:\(\s*[\w:\*\s]+\*\s*\)\s*)?_aligned_malloc')
NULL_WIN = 400       # 扫到下一个函数为止的上限（不是窗口大小）
# raw 头里每个函数都从第 0 列开始（`void zq_cnn_xxx(`），用它划函数边界
FUNC_TOP_RE = re.compile(r'^(?:void|static\s+void|int|float)\s+\w+\s*\(')


def _has_null_check(lines, start, var):
    """从分配点往后找，**直到离开它所在的函数**，看有没有该变量的判空。

    为什么按「函数」而不是按「花括号块」划作用域：
    `if (buffer == 0) { if (c) x = p; else { x = _aligned_malloc(...); } if (c) y = q; else { y = _aligned_malloc(...); } }`
    这种写法里，判空写在**外层**那个块的最后一行 —— 按内层块划作用域会在
    `else` 的 `}` 处就停下，把正确的写法报成漏的（第一版就踩了）。
    raw 头里每个函数都是从第 0 列开始的 `void zq_cnn_xxx(...)`，
    所以「下一个第 0 列的定义行」就是函数边界。
    """
    # **0 / NULL 两种写法都算判空** —— `batchnormscale` 那一族写的是 `== NULL`，
    # gemm / lrn 那一族写的是 `== 0`；只认一种就会把另一半报成假的
    # （AGENTS.md 第 20 条：恒红项要么定性、要么删掉）。
    pat = re.compile(r'\b%s\s*==\s*(?:0|NULL|nullptr)\b|!\s*\b%s\b'
                     % (re.escape(var), re.escape(var)))
    for j in range(start, min(len(lines), start + NULL_WIN)):
        if j > start and pat.search(lines[j]):
            return True
        if FUNC_TOP_RE.match(lines[j]):
            break
    return False



def scan_buffer_nullcheck(text):
    """返回 (分配点总数, [(行号, 说明)])。"""
    lines = text.split('\n')
    # **先把行注释去掉再匹配**：门禁自己的说明文字里写着
    # `` `*buffer = _aligned_malloc(...)` ``，不剥掉就会被当成一处分配点报出来
    # （第一版就踩了：扫描器被自己的注释绊了一跤）。
    code = [ln.split('//')[0] for ln in lines]
    total, bad = 0, []
    for i, line in enumerate(code):
        if BUF_MALLOC_RE.search(line):
            total += 1
            win = '\n'.join(lines[i:i + 5])
            if not re.search(r'\*\s*buffer\s*==\s*0|!\s*\(\s*\*\s*buffer\s*\)', win):
                bad.append((i + 1,
                            '*buffer = _aligned_malloc(...) 之后 5 行内没有 *buffer == 0 检查'))
            continue
        m = ANY_MALLOC_RE.match(line)
        if m:
            var = m.group(1)
            total += 1
            if not _has_null_check(lines, i, var):
                bad.append((i + 1, '`%s = _aligned_malloc(...)` 之后 %d 行内没有判空'
                            % (var, NULL_WIN)))
    return total, bad


def scan_stack_align(text, fname=''):
    """返回 (被 store 的数组数, [(行号, 名字, 说明)])。"""
    lines = text.split('\n')
    decl = {}          # 名字 -> (行号, 是否带对齐属性)
    for i, line in enumerate(lines):
        m = RAW_DECL_RE.match(line)
        if m:
            decl.setdefault(m.group('name'), (i + 1, m.group('align') is not None))
    bad = []
    seen = set()
    for i, line in enumerate(lines):
        m = STORE_RE.search(line)
        if not m:
            continue
        nm = m.group('name')
        if nm in seen or nm not in decl:
            continue
        seen.add(nm)
        ln, aligned = decl[nm]
        if not aligned:
            bad.append((ln, nm,
                        'zq_mm_store_ps 要写它（第 %d 行），但声明处没有 ZQ_DECLSPEC_ALIGN' % (i + 1)))
    return len(seen), bad



def scan_text(text, raw_texts=None):
    """返回 (sections, bad)。

    sections: [(行号, 项数, lane, 下标列表, raw 里的 q 大小或 None)]
    bad:      [(行号, 说明)]
    raw_texts: {raw 头文件名: 文本}，用于查 q 的声明大小
    """
    raw_texts = raw_texts or {}
    sections, bad = [], []
    lines = text.split('\n')
    lane = None
    for i, line in enumerate(lines):
        m = ALIGN_RE.match(line)
        if m:
            lane = int(m.group(1))
        m = FINAL_SUM_RE.match(line)
        if not m:
            continue
        idx = [int(x) for x in TERM_RE.findall(m.group(2))]
        n = len(idx)
        # raw 头：本节之后第一条 include 的 *_raw.h
        qsize = None
        for j in range(i, min(i + 8, len(lines))):
            mi = INCLUDE_RE.match(lines[j])
            if not mi:
                continue
            nm = os.path.basename(mi.group(1))
            rt = raw_texts.get(nm)
            if rt is None:
                qpath = os.path.join(LAYERS_C, nm)
                if os.path.exists(qpath):
                    rt = io.open(qpath, encoding='utf-8').read()
                    raw_texts[nm] = rt
            if rt:
                qs = QDECL_RE.findall(rt)
                if qs:
                    qsize = min(int(x) for x in qs)
            break
        sections.append((i + 1, n, lane, idx, qsize))
        if lane is None:
            bad.append((i + 1, '这一节没有 zq_mm_align_size，无法判定 lane 数'))
            continue
        if n != lane:
            bad.append((i + 1, '项数 %d != lane 数 %d（zq_mm_align_size）' % (n, lane)))
        if sorted(idx) != list(range(n)):
            bad.append((i + 1, '下标不是 0..%d 的连续序列：%s' % (n - 1, idx)))
        if qsize is not None and n > qsize:
            bad.append((i + 1, '项数 %d 越过了 raw 头里的 q[%d]' % (n, qsize)))
    return sections, bad


GOOD = """\
#define zq_mm_align_size 8
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])
#include "zq_demo_raw.h"
"""

# 2026-10-05 那条真缺陷的形状：8 lane 却写了 9 项
OFF_BY_ONE = """\
#define zq_mm_align_size 8
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7]+q[8])
#include "zq_demo_raw.h"
"""

# 另一种形状：项数对，但下标重复（少算一项、重复一项）
DUP_INDEX = """\
#define zq_mm_align_size 8
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7]+q[7])
#include "zq_demo_raw.h"
"""

RAW = "ZQ_DECLSPEC_ALIGN32 zq_base_type q[8];\n"


def selftest():
    ok = True
    # 「多一项」会同时触发 3 条判定中的 2 条：项数 != lane，且下标不是 0..n-1 的连续序列；
    # 再叠上越过 q[] 就是 3 条。这里写的是**实测真值**，不是「我觉得应该是几」。
    cases = [
        ('合格：8 lane / 8 项', GOOD, 1, 0),
        ('**多一项**（IX.1 那条）', OFF_BY_ONE, 1, 2),
        ('下标重复', DUP_INDEX, 1, 3),
    ]
    for name, txt, want_sec, want_bad in cases:
        secs, bad = scan_text(txt, {'zq_demo_raw.h': RAW})
        got = (len(secs), len(bad))
        mark = 'OK ' if got == (want_sec, want_bad) else '**BAD**'
        if got != (want_sec, want_bad):
            ok = False
        print('  %s %-28s 扫到 %d 节 / %d 个问题（期望 %d / %d）'
              % (mark, name, got[0], got[1], want_sec, want_bad))
    # q[] 比 lane 数还小：项数与 lane 数一致，但仍然越界
    tiny_raw = "ZQ_DECLSPEC_ALIGN32 zq_base_type q[4];\n"
    txt = """\
#define zq_mm_align_size 8
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])
#include "zq_demo_raw.h"
"""
    secs, bad = scan_text(txt, {'zq_demo_raw.h': tiny_raw})
    mark = 'OK ' if len(bad) == 1 and 'q[4]' in bad[0][1] else '**BAD**'
    if len(bad) != 1:
        ok = False
    print('  %s %-28s 扫到 %d 个问题（期望 1）' % (mark, '越 q[4]', len(bad)))

    # ---- 判定 2 的自测 ----
    # 合格：带对齐属性
    ok_src = """\
ZQ_DECLSPEC_ALIGN32 zq_base_type buffer_I[8];
void f(void) { zq_mm_store_ps(buffer_I, v_I); }
"""
    # **不合格（IX.3 那条真缺陷）**：四个 buffer_* 都是裸栈数组
    bad_src = """\
zq_base_type buffer_I[8];
zq_base_type buffer_F[8];
void f(void) {
    zq_mm_store_ps(buffer_I, v_I);
    zq_mm_store_ps(buffer_F, v_F);
}
"""
    for name, txt, want_n, want_bad in (
            ('栈数组带对齐（合格）', ok_src, 1, 0),
            ('**裸栈数组被 store**（IX.3）', bad_src, 2, 2)):
        n, bad = scan_stack_align(txt)
        got = (n, len(bad))
        mark = 'OK ' if got == (want_n, want_bad) else '**BAD**'
        if got != (want_n, want_bad):
            ok = False
        print('  %s %-28s 扫到 %d 个被 store 的数组 / %d 个问题（期望 %d / %d）'
              % (mark, name, got[0], got[1], want_n, want_bad))

    # ---- 判定 3 的自测 ----
    good_buf = """        *buffer = _aligned_malloc(total_need_buffer_len, 32);
        if (*buffer == 0)
            return;
        *buffer_len = total_need_buffer_len;
"""
    bad_buf = """        *buffer = _aligned_malloc(total_need_buffer_len, 32);
        *buffer_len = total_need_buffer_len;
"""
    good_var = """        h = _aligned_malloc(need_buffer_size, 32);
        cell = _aligned_malloc(need_buffer_size, 32);
        if (h == 0 || cell == 0)
            return;
"""
    bad_var = """        h = _aligned_malloc(need_buffer_size, 32);
        cell = _aligned_malloc(need_buffer_size, 32);
        memset(h, 0, sizeof(float)*hidden_dim);
"""
    for name, txt, want_n, want_bad in (
            ('分配后查了返回值（合格）', good_buf, 1, 0),
            ('**没查返回值**（IX.5）', bad_buf, 1, 1),
            ('裸变量分配后判空（合格）', good_var, 2, 0),
            ('**裸变量分配不判空**（IX.6）', bad_var, 2, 2)):
        n, bad = scan_buffer_nullcheck(txt)
        got = (n, len(bad))
        mark = 'OK ' if got == (want_n, want_bad) else '**BAD**'
        if got != (want_n, want_bad):
            ok = False
        print('  %s %-28s 扫到 %d 个分配点 / %d 个问题（期望 %d / %d）'
              % (mark, name, got[0], got[1], want_n, want_bad))
    return ok


def main(argv):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

    if '--selfcheck' in argv:
        print('check_mm_safety 自测：')
        if not selftest():
            print('**自测没过 —— 匹配逻辑已坏，先修它再谈别的**')
            return 1
        print('自测通过')
        return 0

    files = [a for a in argv[1:] if not a.startswith('-')]
    if not files:
        files = []
        for d in (LAYERS_C, LAYERS_NCHWC):
            if not os.path.isdir(d):
                continue
            files += [os.path.join(d, f) for f in sorted(os.listdir(d))
                      if f.endswith(('.c', '.h'))]

    raw_texts, total, bad_all = {}, 0, []
    store_total, store_bad_all = 0, []
    nullcheck_total, nullcheck_bad = 0, []
    for f in files:
        text = io.open(f, encoding='utf-8').read()
        if f.endswith('.c'):
            secs, bad = scan_text(text, raw_texts)
            total += len(secs)
            bad_all += [(f, ln, msg) for ln, msg in bad]
            if secs:
                print('%-56s %3d 处 zq_final_sum_q，%d 处不合格'
                      % (os.path.relpath(f, ROOT), len(secs), len(bad)))
                for ln, msg in bad:
                    print('      !! 第 %d 行 %s' % (ln, msg))
        bn, bbad = scan_buffer_nullcheck(text)
        if bn:
            store_total += 0
            nullcheck_total += bn
            nullcheck_bad += [(f, ln, msg) for ln, msg in bbad]
        n, sbad = scan_stack_align(text, f)
        if n:
            store_total += n
            store_bad_all += [(f, ln, nm, msg) for ln, nm, msg in sbad]
            print('%-56s %3d 个 zq_mm_store_ps 目标，%d 个没对齐'
                  % (os.path.relpath(f, ROOT), n, len(sbad)))
            for ln, nm, msg in sbad:
                print('      !! 第 %d 行 %s：%s' % (ln, nm, msg))

    print('合计 %d 处 zq_final_sum_q（%d 处不合格）；'
          '%d 个 zq_mm_store_ps 目标（%d 个没对齐）；'
          '%d 个 *_aligned_malloc(*buffer)（%d 个没查返回值）'
          % (total, len(bad_all), store_total, len(store_bad_all),
             nullcheck_total, len(nullcheck_bad)))
    if nullcheck_bad:
        for f, ln, msg in nullcheck_bad:
            print('      !! %s 第 %d 行 %s'
                  % (os.path.relpath(f, ROOT), ln, msg))
    if not total and not store_total and not nullcheck_total:
        print('**一处都没扫到 —— 匹配逻辑多半坏了**（AGENTS.md 坑 #2）')
        return 1
    if bad_all:
        print('**归约项数必须正好等于 lane 数；多一项就是越界读（ARM FP16 上）**')
        return 1
    if store_bad_all:
        print('**被 zq_mm_store_ps 写的栈数组必须声明成 ZQ_DECLSPEC_ALIGN32：'
              '不对齐在 x86 上是 vmovaps -> #GP**')
        return 1
    if nullcheck_bad:
        print('**`*buffer = _aligned_malloc(...)` 必须查返回值：**'
              '不查就更新 `*buffer_len`，下次调用会跳过重新分配、'
              '拿着空指针去算 —— 空指针解引用发生在**下一次**，不在这一行**')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
