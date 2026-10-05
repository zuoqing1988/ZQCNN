#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SSD / CascadeOnet 检测线的门禁 —— 附录 IQ。

范围
----
`ZQCNN/ZQ_CNN_SSD.h`、`ZQ_CNN_ZQ_CNN_CascadeOnet.h`、`ZQ_CNN_CascadeOnet_Interface.h`、
`ZQCNN/ZQ_CNN_NSFW.h`。这四个头此前零行为门禁覆盖。

一个**元结论**先说在前面，因为它决定了本门禁的写法：

    `diff -w -B ZQ_CNN_CascadeOnet.h ZQ_CNN_CascadeOnet_Interface.h`
    两个文件的主体**只有 4 处类型替换，零逻辑差异**。

所以「这两份拷贝之间逐条找差异」的答案是 **0 条**。
真正有价值的方向在别处：

* **IQ.1** `ZQ_CNN_NSFW.h` 的 include guard 写成了 `_ZQ_CNN_SSD_H_` —— 与
  `ZQ_CNN_SSD.h` **完全撞名**。任一 TU 同时 include 两者，第二个整份被跳过
  -> `'ZQ_CNN_NSFW' is not a member of 'ZQ'`。`#pragma once` 救不了：
  它按**文件**生效，NSFW.h 的 pragma once 不阻止 NSFW.h 自己被处理，
  阻止它的是那个撞名的 guard。目前无 TU 同时包含两者，但 `ZQ_CNN_MouthDetector.h`
  是公共头，外部使用者加上 NSFW 需求即中招。
* **IQ.3** CascadeOnet 两份副本的 `Find(bgr_img,...)` **缺少 `ZQ_CNN_SSD.h:59`
  已经有的入参守卫**。`ConvertFromBGR` 在 `ChangeSize` 之后**无条件**解引用
  `bgr_pix[0..2]`，所以 `bgr_img == nullptr` 是空指针解引用，
  `_widthStep < _width*3` 是**读调用方图像缓冲区越界**。
  这条是「同仓已有正确样板，照抄即可」的最干净一例。
* **IQ.4** 两份副本都丢弃 `Forward` 的返回值。`ZQ_CNN_Net::Forward` 失败时
  只 printf 然后 return false，**blob 内存原样保留**；于是「曾经成功过、后来失败」
  时读到的是**上一轮的陈旧数据**，`Find` 还**返回 true**。
  `SampleCascadeOnet_Interface.cpp:71` 传 `nIters=10`，同一批 net 连跑 10 轮 ——
  sample 的 `if (!Find(...)) failed;` 抓不到。
* **IQ.5** SSD 的 `output.clear()` 排在**七条** `return false` **之后** ——
  任何一次 Detect 失败，调用方仍读 output 就拿到**上一次成功调用的框**。
* **IQ.6** SSD 的 `if (show_debug_info) net.TurnOnShowDebugInfo();` **只开不开**，
  形参是每次调用的，某次传 true 之后所有 Detect 都刷屏。
* **IQ.2** `mxnet_ssd` 未初始化，而 `Init` 有**两条** return false 排在赋值之前。

用法
----
    python tools/check_ssd_cascade.py              # 扫
    python tools/check_ssd_cascade.py --selfcheck  # 先自测
"""
from __future__ import print_function

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SSD = os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_SSD.h')
NSFW = os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_NSFW.h')
CO = os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_CascadeOnet.h')
COI = os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_CascadeOnet_Interface.h')

ALL = [SSD, NSFW, CO, COI]

# 各头的 include guard 必须**全局唯一**
GUARD_RE = re.compile(r'^\s*#ifndef\s+(\w+)\s*$', re.M)

IQ3_GUARD = re.compile(
    r'bgr_img\s*==\s*0.*?_width\s*<=\s*0.*?_height\s*<=\s*0.*?_widthStep\s*<\s*_width\s*\*\s*3', re.S)
IQ4_FWD = re.compile(r'if\s*\(\s*!\s*nets\[i\]->Forward\(')
IQ4_CLEAR = re.compile(r'results\.clear\(\)\s*;')
# IQ.5 不用「字符数窗口」判位置 —— 第一版用 `[^\n]{0,400}?` 猜 clear 是不是「在开头」，
# 结果**正确**的样本被判成在中后段（自测立刻发现）。
# 「往后看 N 个字符」这个隐含假设，本会话已经栽了四次（A27 / A39 / IN.7 / 这里）。
# 改成：取 Detect 的**函数体**，看 output.clear() 之前有没有 return。
RE_DETECT = re.compile(r'bool\s+Detect\s*\(')
IQ2_INIT = re.compile(r'bool\s+mxnet_ssd\s*=\s*false\s*;')
IQ6_OFF = re.compile(r'else\s*\n\s*net\.TurnOffShowDebugInfo\(\)\s*;')


def _block_after(text, at):
    i = text.find('{', at)
    if i < 0:
        return '', -1
    depth = 0
    j = i
    n = len(text)
    while j < n:
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j
        j += 1
    return text[i + 1:], -1


def read(p):
    with io.open(p, 'r', encoding='utf-8-sig') as f:   # NSFW.h 带 BOM
        return f.read()


def strip_comments(text):
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == '\\' else 1
            j = min(j + 1, n)
            out.append(text[i:j])
            i = j
            continue
        if c == '/' and i + 1 < n:
            nx = text[i + 1]
            if nx == '/':
                j = text.find('\n', i)
                j = n if j < 0 else j
                out.append(' ' * (j - i))
                i = j
                continue
            if nx == '*':
                j = text.find('*/', i + 2)
                j = n if j < 0 else j + 2
                out.append(''.join(ch if ch == '\n' else ' ' for ch in text[i:j]))
                i = j
                continue
        out.append(c)
        i += 1
    return ''.join(out)


def scan_guards(texts):
    """IQ.1：include guard 全局唯一。"""
    bad, ok = [], 0
    seen = {}
    for p in ALL:
        name = os.path.basename(p)
        m = GUARD_RE.search(texts[name])
        if not m:
            bad.append(('IQ.1', '%s 找不到 `#ifndef <GUARD>`' % name))
            continue
        g = m.group(1)
        if g in seen:
            bad.append(('IQ.1', 'include guard `%s` **撞名**：%s 与 %s 用同一个 —— '
                        '任一 TU 同时 include 两者，第二个整份被跳过 -> '
                        '`is not a member of ZQ`。（`#pragma once` 按**文件**生效，'
                        '救不了这个。）' % (g, seen[g], name)))
        else:
            seen[g] = name
    if not bad:
        ok += 1
    return ok, bad


def scan_ssd(t):
    bad, ok = [], 0
    # IQ.2
    if IQ2_INIT.search(t):
        ok += 1
    else:
        bad.append(('IQ.2', '`bool mxnet_ssd;` 无初值，而 `Init` 有**两条** return false '
                    '排在赋值之前；调用方忽略 Init 返回值时 Detect 读的是不确定值（UB）'))
    # IQ.5
    m = RE_DETECT.search(t)
    body, _ = _block_after(t, m.end()) if m else ('', -1)
    ci = body.find('output.clear();')
    ri = body.find('return')
    if m and ci >= 0 and (ri < 0 or ci < ri):
        ok += 1
    elif not m:
        bad.append(('IQ.5', '找不到 Detect'))
    else:
        bad.append(('IQ.5', '`output.clear()` 排在 Detect 体内**第一个 return 之后** '
                    "\u2014\u2014 任何一次 Detect 失败，调用方仍读 output 就拿到"
                    '**上一次成功调用的框**'))
    # IQ.6
    if IQ6_OFF.search(t):
        ok += 1
    else:
        bad.append(('IQ.6', '`if (show_debug_info) net.TurnOnShowDebugInfo();` **只开不开** —— '
                    '形参是每次调用的，某次传 true 之后所有 Detect 都刷屏，'
                    '类里也没有 TurnOff 出口'))
    return ok, bad


def scan_cascade(t, name):
    bad, ok = [], 0
    # IQ.3
    if IQ3_GUARD.search(t):
        ok += 1
    else:
        bad.append(('IQ.3', '%s 的 `Find(bgr_img,...)` **缺少入参守卫**，'
                    '而同仓 `ZQ_CNN_SSD.h:59` 就有。`ConvertFromBGR` 在 ChangeSize 之后'
                    '**无条件**解引用 bgr_pix[0..2]：bgr_img==nullptr 是空指针解引用，'
                    '_widthStep<_width*3 是**读调用方图像缓冲区越界**' % name))
    # IQ.4
    if IQ4_FWD.search(t) and IQ4_CLEAR.search(t):
        ok += 1
    else:
        bad.append(('IQ.4', '%s 丢弃了 `Forward` 的返回值。`ZQ_CNN_Net::Forward` 失败时'
                    '只 printf 然后 return false，**blob 内存原样保留**；'
                    '于是「曾经成功过、后来失败」时读到的是**上一轮的陈旧数据**，'
                    '而 Find 还**返回 true**（sample 的 if(!Find) 抓不到）。'
                    '失败时应 results.clear(); return false;' % name))
    return ok, bad


# ------------------------------------------------------------------ 自测
FULL_SSD = """
#ifndef _ZQ_CNN_SSD_H_
#define _ZQ_CNN_SSD_H_
class ZQ_CNN_SSD {
	bool mxnet_ssd = false;
	bool Detect(std::vector<BBox>& output, const unsigned char* bgr_img, int width, int height, int widthStep, float thresh,
		bool show_debug_info = false)
	{
		output.clear();
		if (bgr_img == 0 || width <= 0 || height <= 0 || widthStep < width * 3)
			return false;
		if (show_debug_info)
			net.TurnOnShowDebugInfo();
		else
			net.TurnOffShowDebugInfo();
		if (mxnet_ssd) { } else { }
		float scale_X = width;
		return true;
	}
};
#endif
"""
FULL_CO = """
#ifndef _ZQ_CNN_CASCADEONET_H_
#define _ZQ_CNN_CASCADEONET_H_
class ZQ_CNN_CascadeOnet {
	bool Find(const unsigned char* bgr_img, int _width, int _height, int _widthStep, std::vector<ZQ_CNN_BBox>& results)
	{
		if (bgr_img == 0 || _width <= 0 || _height <= 0 || _widthStep < _width * 3)
			return false;
		if (!input.ConvertFromBGR(bgr_img, _width, _height, _widthStep))
			return false;
		if (!nets[i]->Forward(*(onet_images[i])))
		{
			results.clear();
			return false;
		}
		return true;
	}
};
#endif
"""

SELFCHECK = [
    ('SSD 全合格', 'ssd', FULL_SSD, []),
    ('IQ.2 mxnet_ssd 无初值', 'ssd', FULL_SSD.replace('bool mxnet_ssd = false;', 'bool mxnet_ssd;'), ['IQ.2']),
    ('IQ.5 clear 排在后面', 'ssd',
     FULL_SSD.replace('output.clear();\n\t\tif (bgr_img == 0', 'if (bgr_img == 0')
     .replace('\t\treturn true;\n\t}\n};', '\t\treturn true;\n\t}\n};'),
     ['IQ.5']),
    ('IQ.6 只开不开', 'ssd', FULL_SSD.replace('\t\telse\n\t\t\tnet.TurnOffShowDebugInfo();\n', ''), ['IQ.6']),
    ('CascadeOnet 全合格', 'cascade', FULL_CO, []),
    ('IQ.3 缺入参守卫', 'cascade',
     FULL_CO.replace('if (bgr_img == 0 || _width <= 0 || _height <= 0 || _widthStep < _width * 3)\n\t\t\treturn false;\n\t\t', ''),
     ['IQ.3']),
    ('IQ.4 丢弃 Forward 返回值', 'cascade',
     FULL_CO.replace('if (!nets[i]->Forward(*(onet_images[i])))\n\t\t{\n\t\t\tresults.clear();\n\t\t\treturn false;\n\t\t}',
                     'nets[i]->Forward(*(onet_images[i]));'),
     ['IQ.4']),
    # IQ.1 的阴性对照：两个不同的 guard
    ('IQ.1 阴性：guard 不撞名', 'guards',
     '#ifndef _ZQ_CNN_SSD_H_\n' + '#ifndef _ZQ_CNN_CASCADEONET_H_\n', []),
    ('IQ.1 guard 撞名', 'guards',
     '#ifndef _ZQ_CNN_SSD_H_\n' + '#ifndef _ZQ_CNN_SSD_H_\n', ['IQ.1']),
]


def selfcheck():
    bad = 0
    for name, kind, text, expect in SELFCHECK:
        t = strip_comments(text)
        if kind == 'ssd':
            r = scan_ssd(t)
        elif kind == 'cascade':
            r = scan_cascade(t, 'X.h')
        else:
            g1, g2 = GUARD_RE.search(t), None
            g2 = GUARD_RE.search(t, g1.end()) if g1 else None
            seen = {}
            r = ([], [])
            for nm, gg in (('a.h', g1), ('b.h', g2)):
                if gg and gg.group(1) in seen:
                    r[1].append(('IQ.1', '撞名'))
                elif gg:
                    seen[gg.group(1)] = nm
            r = (1, []) if not r[1] else r
        got = sorted(set(c for c, _ in r[1]))
        if got != sorted(set(expect)):
            bad += 1
            print('  [self-MISMATCH] %s' % name)
            print('      expect %s, got %s' % (sorted(expect), got))
            for c, m in r[1]:
                print('        %s: %s' % (c, m))
        else:
            print('  [self-OK]      %-30s %s' % (name, ','.join(got) if got else 'clean'))
    if bad:
        print('selfcheck FAILED: %d / %d mismatch' % (bad, len(SELFCHECK)))
        return 1
    print('selfcheck OK: %d cases, all as expected' % len(SELFCHECK))
    return 0


def main(argv):
    if '--selfcheck' in argv:
        return selfcheck()
    texts = {}
    for p in ALL:
        texts[os.path.basename(p)] = read(p)
    total_ok = 0
    failed = False
    # IQ.1
    ok, b = scan_guards(texts)
    total_ok += ok
    if b:
        failed = True
        print('FAIL include guard 唯一性')
        for c, m in b:
            print('       - %s: %s' % (c, m))
    else:
        print('OK   include guard 唯一性')
    # IQ.2/5/6
    ok, b = scan_ssd(strip_comments(texts['ZQ_CNN_SSD.h']))
    total_ok += ok
    if b:
        failed = True
        print('FAIL ZQ_CNN_SSD.h')
        for c, m in b:
            print('       - %s: %s' % (c, m))
    else:
        print('OK   ZQ_CNN_SSD.h')
    # IQ.3/4
    for f in ('ZQ_CNN_CascadeOnet.h', 'ZQ_CNN_CascadeOnet_Interface.h'):
        ok, b = scan_cascade(strip_comments(texts[f]), f)
        total_ok += ok
        if b:
            failed = True
            print('FAIL %s' % f)
            for c, m in b:
                print('       - %s: %s' % (c, m))
        else:
            print('OK   %s' % f)
    print('合计合格判定 %d 项' % total_ok)
    if failed:
        print('**SSD / CascadeOnet 门禁不合格**')
        return 1
    print('SSD / CascadeOnet: OK')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
