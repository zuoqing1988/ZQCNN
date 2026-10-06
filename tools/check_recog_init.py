#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""门禁：把 recognizer 递进流水线之前，必须先 Init 过。

起因（2026-10-07，附录 JB）
--------------------------
`SamplesZQlibFaceID` 的 5 个 `SampleCropImagesFor*` 里，**只有 SeetaFace 那份**
在把 recognizer 交给 `ZQ_FaceDatabaseMaker` 之前调用了 `Init`：

    // SampleCropImagesForSeetaFace（对）
    if (!recognizers[i].Init(model_file)) { printf(...); return false; }

    // SampleCropImagesForArcFace / ArcFaceFast / SphereFace / SphereFaceFast（错）
    detectors[i].Init();                      // 只初始化了检测器
    ptr_recognizers[i] = &recognizers[i];     // recognizer 裸着进流水线

而 `ZQ_FaceRecognizer::Init` 是**纯虚**（`= 0`），它是"把网络加载进来"这一步；
`ZQ_FaceDatabaseMaker::MakeDatabase` **只检查指针非空，不检查是否已初始化**。
于是未初始化的 recognizer 一路走到特征提取。

静默：本地实测 `SampleCropImagesForSphereFace.exe data/ <out>` 返回 **RC=0**、
产出 0 张图、耗时 0.000000s —— 不报任何错。

为什么值得一道门禁
------------------
ZQlibFaceID 有 **29 个头 + 23 个 sample**，在 Windows 上被编译，
而 `WIN_SAMPLES` 里**一个都没跑**（附录 JB）。也就是说
"样例写错了"这件事没有任何自动化能发现 —— 门禁是这里唯一可行的防线。

判据
----
在每个 sample 源里找「把 recognizer 塞进 `ptr_recognizers[...]`」的位置，
要求**同一个函数里、该位置之前**出现过对同一变量的 `Init(` 调用。
只认 `ptr_` 前缀那类"递进流水线"的写法 —— 普通的局部 `recognizers[i].Init()`
也算，因为那同样是初始化。
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCAN = ('SamplesZQlibFaceID', 'SamplesZQCNN')
SKIP_DIRS = {'cmake-build-debug', 'cmake-build-release', 'build'}

HANDOFF_RE = re.compile(r'\bptr_recognizers\s*\[\s*\w+\s*\]\s*=')


def strip_noise(src):
    """去掉注释与字符串，避免注释里的 Init 被当成真调用。

    **块注释必须逐字符替换成等量字符（保留换行）**：
    第一版用 `re.sub(r'/\\*.*?\\*/', ' ', flags=re.S)`，DOTALL 把整个跨行
    注释压成**一个空格** —— 于是文件里的换行被吃掉了，**行号全错、花括号
    配平也全错**。表现出来就是：唯一写对的那份 sample 也被报成缺陷
    （SeetaFace），而且报出来的行号指向别处。

    判据与附录 IS 那条一样：**按行替换成等长空格**，行号才不漂。
    """
    out = []
    i = 0
    n = len(src)
    while i < n:
        if src.startswith('/*', i):
            j = src.find('*/', i + 2)
            j = n if j < 0 else j + 2
            out.append(''.join(ch if ch == '\n' else ' ' for ch in src[i:j]))
            i = j
            continue
        if src.startswith('//', i):
            j = src.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i))
            i = j
            continue
        ch = src[i]
        if ch in '"\'':
            q = ch
            j = i + 1
            while j < n:
                if src[j] == '\\':
                    j += 2
                    continue
                if src[j] == q:
                    j += 1
                    break
                j += 1
            out.append(' ' * (j - i))
            i = j
            continue
        out.append(ch)
        i += 1
    return ''.join(out)


def func_start(lines, idx):
    """向上找**函数体**的 `{`：取最近一个顶格的 `{`。

    试过两条路，都不行，值得记下来：

    * 「向上找最近的像函数签名的行」—— 猜的，SeetaFace 里 `Init` 与递进分在
      两个独立 for 循环、中间隔着 printf，那一版把 `Init` 落在窗口外，
      于是把**唯一写对的那份**也报成缺陷。
    * 「用花括号配平回溯」—— 也不行：这里要找的是**函数**的 `{`，而 idx 外面
      最近的那个 `{` 是 **for 循环体**的，它由 idx **下方**的 `}` 闭合。
      从 idx 往上走根本看不到那个 `}`，于是配平在 for 的 `{` 处就"配平"了，
      返回的是循环而不是函数。

    最后用的是这个仓的书写习惯：函数体的 `{` 顶格（列 0），for/if 的 `{`
    一定带缩进。直接认顶格，不猜。
    """
    for k in range(idx, -1, -1):
        if lines[k].startswith('{'):
            return k
    return 0


def scan_text(text, path, rel):
    """返回 [(行号, 说明)]：递进流水线前没有 Init 的站点。"""
    text = strip_noise(text)
    lines = text.split('\n')
    hits = []
    for idx, ln in enumerate(lines):
        if not HANDOFF_RE.search(ln):
            continue
        var = re.search(r'ptr_recognizers\s*\[\s*(\w+)\s*\]', ln).group(1)
        # 该行里推出被赋对象的变量名：ptr_recognizers[i] = &recognizers[i];
        m = re.search(r'&\s*(\w+)\s*\[', ln)
        obj = m.group(1) if m else var
        start = func_start(lines, idx)
        window = '\n'.join(lines[start:idx + 1])
        # Init 必须在**递进之前**，且对象是同一个
        init = re.search(r'\b%s\s*\[\s*%s\s*\]\s*\.\s*Init\s*\('
                         % (re.escape(obj), re.escape(var)), window)
        if not init:
            hits.append((idx + 1,
                         '%s: %s 在 Init 之前被递进流水线'
                         % (rel, ln.strip()[:70])))
    return hits


def collect():
    out = []
    for d in SCAN:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames if x not in SKIP_DIRS]
            for fn in filenames:
                if not fn.lower().endswith(('.cpp', '.cc', '.cxx')):
                    continue
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, ROOT).replace('\\', '/')
                with open(p, 'r', encoding='utf-8', errors='replace') as f:
                    out += scan_text(f.read(), p, rel)
    return out


def selftest():
    cases = [
        ('有 Init 才递进 -> 合规',
         'bool F(){\n'
         '  for (int i=0;i<n;i++){\n'
         '    if (!recognizers[i].Init("04bn256")) return false;\n'
         '    ptr_recognizers[i] = &recognizers[i];\n'
         '  }\n'
         '}\n', 0),
        ('没 Init 就递进 -> 必须报',
         'bool F(){\n'
         '  for (int i=0;i<n;i++){\n'
         '    detectors[i].Init();\n'
         '    ptr_recognizers[i] = &recognizers[i];\n'
         '  }\n'
         '}\n', 1),
        ('注释里的 Init 不算数',
         'bool F(){\n'
         '  for (int i=0;i<n;i++){\n'
         '    // recognizers[i].Init("x");\n'
         '    ptr_recognizers[i] = &recognizers[i];\n'
         '  }\n'
         '}\n', 1),
        ('两个不同变量：别把别人的 Init 当成自己的',
         'bool F(){\n'
         '  for (int i=0;i<n;i++){\n'
         '    other[i].Init("x");\n'
         '    ptr_recognizers[i] = &recognizers[i];\n'
         '  }\n'
         '}\n', 1),
    ]
    bad = []
    for name, src, expect in cases:
        got = len(scan_text(src, None, 't.cpp'))
        ok = (got == expect)
        print('  [%s] %-44s expect=%d got=%d'
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
        print('发现 %d 处「recognizer 未 Init 就被递进流水线」：' % len(hits))
        for ln, msg in hits:
            print('  * %s:%d' % (msg.split(':')[0], ln))
            print('      %s' % msg)
        print('')
        print('`ZQ_FaceRecognizer::Init` 是纯虚，它是"把网络加载进来"这一步；')
        print('`MakeDatabase` 只查指针非空、不查是否已初始化。')
        return 1
    print('OK: 所有递进流水线的 recognizer 都先 Init 过')
    return 0


if __name__ == '__main__':
    sys.exit(main())