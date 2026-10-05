#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""每个「从模型文件读 dilate」的卷积层类，都必须有 `(kernel-1)*dilate+1` 的溢出守卫。

为什么要这道门禁
----------------
`GetTopDim` 里要算 `(kernel_H - 1) * dilate_H`，这是 **int** 乘法。
`kernel_H` / `dilate_H` 都来自**模型文件**（不可信输入）：两个都取 50000 时
乘积 2,499,950,000 超过 `INT_MAX`，gcc 实测回绕成 **-1,795,017,296**（负数），
于是 `bottom_H + pad*2 - 负数 - 1` 变成巨大正数，`top_H` 被算成十几亿，
`SetShape` 的 `ChangeSize` 随后因「raw size > 0x7FFFFFFF」失败 ——
而 `LayerSetup` **不检查 `SetShape` 的返回值**，于是留下一个**零尺寸张量**，
`firstPixelData == 0` -> 空指针解引用（附录 ED.1）。

主副本 `ZQCNN/ZQ_CNN_Layer.h` 里 16 处有守卫（附录 EM.3），
但**另外两份拷贝各只有一半**：
  * `ZQCNN_to_MNN/converter/source/ZQ_CNN_Layer.h` 当初也有（后来同步过去了）；
  * `ZQCNN/ZQ_CNN_Layer_NCHWC.h` **2026-10-05 之前一处都没有**（附录 IX.19）。
这是「一个副本有守卫、孪生副本没有」的第四次（IH.9 / BE.2 / IX.14 之后）。

所以这条判定不看「文件里有没有出现过守卫」，而是**逐个类**看：
**这个类从模型文件读 `dilate`，它自己就必须带这条守卫**。

用法
----
    python tools/check_conv_overflow_guard.py              # 扫默认文件
    python tools/check_conv_overflow_guard.py --selfcheck  # 先自测
    python tools/check_conv_overflow_guard.py <文件路径>    # 扫指定文件

自测样本里**必须有故意不合格的项**（AGENTS.md「写检查类工具的四条硬规矩」第 1 条）。
"""
from __future__ import print_function

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILES = [
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_Layer.h'),
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_Layer_NCHWC.h'),
    os.path.join(ROOT, 'ZQCNN_to_MNN', 'converter', 'source', 'ZQ_CNN_Layer.h'),
]

CLASS_RE = re.compile(r'\n\tclass (ZQ_CNN_Layer_\w+)\s*:\s*public ZQ_CNN_Layer')
# 「这个类从模型文件读 dilate」的判据：ReadParam 里那两种赋值都算
DILATE_FROM_MODEL_RE = re.compile(r'\bdilate(?:_H|_W)?\s*=\s*atoi\(')
GUARD_RE = re.compile(
    r'\(__int64\)dilate_H\s*\*\s*\(kernel_H\s*-\s*1\)\s*\+\s*1\s*>\s*0x7FFFFFFF')


def scan_text(text, label='<text>'):
    """返回 (读到 dilate 的类数, [(类名, 说明)])。"""
    out, bad = [], []
    for m in CLASS_RE.finditer(text):
        name = m.group(1)
        i = text.index('{', m.end())
        depth, j = 0, i
        while j < len(text):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = text[i:j]
        code = '\n'.join(ln.split('//')[0] for ln in body.split('\n'))
        if not DILATE_FROM_MODEL_RE.search(code):
            continue
        out.append(name)
        if not GUARD_RE.search(code):
            bad.append((name, '%s: 从模型文件读 dilate，却没有 (kernel-1)*dilate+1 的 int 溢出守卫'
                        % label))
    return out, bad


GOOD = """
	class ZQ_CNN_Layer_Demo : public ZQ_CNN_Layer
	{
		virtual bool ReadParam(const std::string& line)
		{
			dilate_H = atoi(paras[n][1].c_str());
			if ((__int64)dilate_H * (kernel_H - 1) + 1 > 0x7FFFFFFF)
			{
				return false;
			}
			return true;
		}
	};
"""

BAD = """
	class ZQ_CNN_Layer_Demo : public ZQ_CNN_Layer
	{
		virtual bool ReadParam(const std::string& line)
		{
			dilate_H = atoi(paras[n][1].c_str());
			return true;
		}
	};
"""


def selftest():
    ok = True
    for name, txt, want_n, want_bad in (
            ('有 dilate 也有守卫（合格）', GOOD, 1, 0),
            ('**有 dilate 没有守卫**（IX.19）', BAD, 1, 1)):
        cls, bad = scan_text(txt, '<selftest>')
        got = (len(cls), len(bad))
        mark = 'OK ' if got == (want_n, want_bad) else '**BAD**'
        if got != (want_n, want_bad):
            ok = False
        print('  %s %-28s 扫到 %d 个类 / %d 个问题（期望 %d / %d）'
              % (mark, name, got[0], got[1], want_n, want_bad))
    return ok


def main(argv):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

    if '--selfcheck' in argv:
        print('check_conv_overflow_guard 自测：')
        if not selftest():
            print('**自测没过 —— 匹配逻辑已坏，先修它再谈别的**')
            return 1
        print('自测通过')
        return 0

    files = [a for a in argv[1:] if not a.startswith('-')]
    if not files:
        files = [f for f in DEFAULT_FILES if os.path.isfile(f)]
    if not files:
        print('**一个待扫文件都没找到 —— 匹配逻辑或路径多半坏了**')
        return 1

    total, bad_all = 0, []
    for f in files:
        text = io.open(f, encoding='utf-8', errors='replace').read()
        cls, bad = scan_text(text, os.path.relpath(f, ROOT))
        total += len(cls)
        bad_all += bad
        print('%-52s %2d 个类从模型文件读 dilate，%d 个缺溢出守卫'
              % (os.path.relpath(f, ROOT), len(cls), len(bad)))
        for name, why in bad:
            print('      !! %s：%s' % (name, why))

    print('合计 %d 个类从模型文件读 dilate，%d 个缺 (kernel-1)*dilate+1 的溢出守卫'
          % (total, len(bad_all)))
    if not total:
        print('**一个都没扫到 —— 匹配逻辑多半坏了**（AGENTS.md 坑 #2）')
        return 1
    if bad_all:
        print('**缺这条守卫 -> (kernel-1)*dilate 溢出回绕 -> top_H 巨大 -> '
              'SetShape 失败但没人检查 -> 零尺寸张量 -> 空指针解引用**')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
