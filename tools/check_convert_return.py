#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""门禁：`ConvertFromBGR` 的**返回值**必须被使用。

起因（2026-10-07，附录 JF）
---------------------------
`ZQ_CNN_Tensor4D::ConvertFromBGR` 与 `ConvertFromCompactNCHW` 返回 `bool`（尺寸不符 / 空指针时失败），
而 `ZQ_FaceRecognizerSphereFaceZQCNN::ExtractFeature` 里 **7 处调用全部把
返回值丢掉了**：

    input.ConvertFromBGR(&bgr_buffer[0], crop_width, crop_height, ...);
    break;                       // <- 失败也往下走

失败之后 `net.Forward(input)` 拿的是**上一次留在 input 里的旧数据**，
于是特征是错的，**却不报**。这一族正是本会话反复修的那类：
「静默」换「响亮」。

判据
----
凡是**调用点**（`某个对象. ConvertFromBGR(...)`），
它必须出现在 `if (...)` 里、或者被赋值给一个变量。
**声明与定义不算调用** ——
`ZQ_CNN_Tensor4D.h:395` 的 `virtual bool ConvertFromBGR(...)` 是函数头，
第一版没排除它，把定义当成了漏检的调用点。

范围：ZQCNN / ZQlibFaceID / SamplesZQCNN 的一方代码。
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCAN_DIRS = ('ZQCNN', 'ZQ_GEMM', 'ZQlibFaceID', 'SamplesZQCNN',
             'SamplesZQlibFaceID')
SKIP_DIRS = {'3rdparty', 'build_x64', 'cmake-out-win32-x64',
             'cmake-out-linux-x64'}

# 同一族的**全部** bool 返回方法：必须一起加，且三条正则共用 _M。
# 第一次只改了 DECL_RE，CALL_RE / USED_RE 里还只有 ConvertFromBGR ——
# 于是 ConvertFromCompactNCHW 的 141 个站点**根本没被扫**，
# 真实树报「0 处违规」而自测里那条「丢弃 -> 必须报」是红的。
# 三条正则用同一个 _M 拼，别再各写各的。
# **只覆盖这两个方法**，因为只有它们是「按名字就能确定返回类型」的：
#   ConvertFromBGR / ConvertFromCompactNCHW —— 全仓只有 `ZQ_CNN_Tensor4D`
#   一处声明，返回 bool。
#
# **刻意排除** CopyData / ChangeSize / ConvertFromGray / AddScalar /
# MulScalar：**同名不同返回类型**。实测（2026-10-07）：
#   ZQ_CNN_Tensor4D::CopyData   -> bool
#   ZQ_FaceFeature::CopyData    -> **void**   (ZQlibFaceID/ZQ_FaceFeature.h:42)
#   ZQ_CNN_Tensor4D::ChangeSize -> bool
#   ZQ_FaceFeature::ChangeSize  -> **void**
# 按名字匹配就会给 void 的调用包上 `if (!x.ChangeSize(..))`，直接编不过
# （当时 ZQ_FaceDatabaseMaker.h / ZQ_FaceIDPrecisionEvaluation.h 报了一片
#  'could not convert void to bool'）。
#
# 要覆盖它们必须**按接收者的类**去解析声明的返回类型 —— 那是另一件事。
# 宁可少覆盖，也不做一个会把代码改坏的门禁。
_M = r'(?:ConvertFromBGR|ConvertFromCompactNCHW)'
CALL_RE = re.compile(r'\b(\w+)\s*(?:\.|->)\s*' + _M + r'\s*\(')
# 函数头：`virtual bool ConvertFromBGR(` / `bool CopyData(` …—
# 没有「对象.」，不算调用点
DECL_RE = re.compile(r'^\s*(?:virtual\s+|static\s+|inline\s+)*bool\s*'
                     + _M + r'\s*\(')
USED_RE = re.compile(r'\bif\s*\(|=\s*\w+\s*(?:\.|->)\s*' + _M + r'|'
                     r'!\s*\w+\s*(?:\.|->)\s*' + _M + r'|'
                     r'return\s+\w+\s*(?:\.|->)\s*' + _M)


def strip_comments(src):
    out, i, n = [], 0, len(src)
    while i < n:
        if src.startswith('/*', i):
            j = src.find('*/', i + 2)
            j = n if j < 0 else j + 2
            out.append(''.join(c if c == '\n' else ' ' for c in src[i:j]))
            i = j
            continue
        if src.startswith('//', i):
            j = src.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i))
            i = j
            continue
        out.append(src[i])
        i += 1
    return ''.join(out)


INEXPRESSIBLE_RE = re.compile(
    r'^\s*(?:[\w:]+\s*)?[A-Za-z_]\w*\s*\(\s*\)\s*$'
    r'|^\s*[\w:]+\s*&?\s*operator\s*=')


def enclosing_head(lines, idx):
    """用花括号配平找出 idx 所在函数的**头那一行**（配平，不是正则猜）。"""
    depth = 0
    for k in range(idx, -1, -1):
        depth += lines[k].count('}') - lines[k].count('{')
        if depth < 0:
            for j in range(k, -1, -1):
                if '{' in lines[j]:
                    # 函数头通常在左花括号的**上一行**。
                    if lines[j].strip() == '{' and j > 0:
                        return lines[j - 1]
                    return lines[j]
            return ''
    return ''


def scan_text(text, rel):
    lines = strip_comments(text).split('\n')
    hits = []
    inexpressible = []
    for idx, ln in enumerate(lines):
        if DECL_RE.match(ln):
            continue                      # 函数头，不是调用点
        if not CALL_RE.search(ln):
            continue
        if USED_RE.search(ln):
            continue
        if INEXPRESSIBLE_RE.match(enclosing_head(lines, idx)):
            # **构造函数 / operator= 里没法"返回失败"** —— 它们的返回类型
            # 分别是类本身和引用，`return false;` 是类型错误。
            # 这不是"按严重性放宽"，而是**语言层面无法表达**，所以单列一类
            # **可见但不判失败**，而不是悄悄放过（附录 IO 豁免名单的老教训）。
            inexpressible.append((idx + 1, ln.strip()[:70]))
            continue
        hits.append((idx + 1, ln.strip()[:78]))
    return hits, inexpressible


def collect():
    out = []
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames if x not in SKIP_DIRS]
            for fn in filenames:
                if not fn.lower().endswith(('.h', '.cpp', '.c', '.hpp')):
                    continue
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, ROOT).replace('\\', '/')
                with open(p, 'r', encoding='utf-8', errors='replace') as f:
                    hs, inex = scan_text(f.read(), rel)
                    for ln, txt in hs:
                        out.append((rel, ln, txt))
                    for ln, txt in inex:
                        out.append((rel, ln, 'INEXPRESSIBLE::' + txt))
    return sorted(out)


def selftest():
    cases = [
        ('if (!x.ConvertFromBGR(..)) return false;  -> 合规',
         'if (!input.ConvertFromBGR(p, w, h, s)) { return false; }', 0),
        ('if (input.ConvertFromBGR(..)) ..           -> 合规',
         'if (input.ConvertFromBGR(p, w, h, s)) { ok = 1; }', 0),
        ('bool ok = x.ConvertFromBGR(..);            -> 合规',
         'bool ok = input.ConvertFromBGR(p, w, h, s);', 0),
        ('丢弃返回值                                 -> 必须报',
         'input.ConvertFromBGR(p, w, h, s);\n\t\tbreak;', 1),
        ('函数头 virtual bool ConvertFromBGR(..)    -> 不算调用点',
         'virtual bool ConvertFromBGR(const unsigned char* b, int w, int h, int s,\n'
         '\t\tconst float m = 127.5f)', 0),
        ('函数头 bool ConvertFromBGR(..)            -> 不算调用点',
         'bool ConvertFromBGR(const unsigned char* b, int w, int h, int s)', 0),
        ('注释里的调用不算',
         '// input.ConvertFromBGR(p, w, h, s);', 0),
        ('指针调用 x->ConvertFromBGR 丢弃            -> 必须报',
         'input->ConvertFromBGR(p, w, h, s);', 1),
        ('ConvertFromCompactNCHW 已检查              -> 合规',
         'if (!o->ConvertFromCompactNCHW(d, N, C, H, W)) { return false; }', 0),
        ('ConvertFromCompactNCHW 丢弃                -> 必须报',
         'filters->ConvertFromCompactNCHW(&raw[0], N, C, H, W);\n\t\treturn true;', 1),
        ('ConvertFromCompactNCHW 用 return 返回      -> 合规',
         'return output.ConvertFromCompactNCHW(&b[0], N, C, H, W);', 0),
        ('ConvertFromCompactNCHW 的函数头不算调用点  -> 合规',
         'bool ConvertFromCompactNCHW(const float* d, int N, int C, int H, int W)', 0),
    ]
    bad = []
    for name, src, expect in cases:
        hits, _inex = scan_text(src, 't.cpp')
        got = len(hits)
        ok = (got == expect)
        print('  [%s] %-50s expect=%d got=%d'
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
    raw = collect()
    inex = [h for h in raw if h[2].startswith('INEXPRESSIBLE::')]
    hits = [h for h in raw if not h[2].startswith('INEXPRESSIBLE::')]
    if inex:
        print('注意：%d 处「丢弃返回值」在**构造函数 / operator=** 里，'
              '语言层面无法用返回值表达失败 —— 单列可见，不判失败：' % len(inex))
        for rel, ln, txt in inex:
            print('    %s:%d  %s' % (rel, ln, txt.split('::', 1)[1]))
        print('')
    if hits:
        print('发现 %d 处「ConvertFromBGR 的返回值被丢弃」：' % len(hits))
        for rel, ln, txt in hits:
            print('  * %s:%d  %s' % (rel, ln, txt))
        print('')
        print('失败之后代码会继续往下跑，用的是**上一次留在 input 里的旧数据**，')
        print('于是结果是错的却不报。改成 `if (!x.ConvertFromBGR(..)) return false;`。')
        return 1
    print('OK: 所有 ConvertFromBGR 调用点的返回值都被使用了')
    return 0


if __name__ == '__main__':
    sys.exit(main())