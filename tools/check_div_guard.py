#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""门禁：**每一处"除以模型参数"的语句，上方必须有一个 <=0 的守卫**。

为什么要有这个（audit_k3_20261001.md 附录 BH）
---------------------------------------------
附录 BE 修了 `ZQ_CNN_Forward_SSEUtils.h` 里 7 处 `/ strideH`，
但**忘了 NCHWC 那份复制品**里的 25 处 —— 是附录 BG 的新工具
（`check_param_domain.py`）把它抓出来的。

也就是说：**当时没有任何机制会告诉我"还有 25 处一样的东西"**。
本工具就是那个机制。

它扫的范围是**整棵树**（ZQCNN / ZQlibFaceID / SamplesZQCNN / SamplesZQlibFaceID），
不只看 `ZQ_CNN_Layer*.h`，所以它能覆盖"层类之外"的地方。

判定
----
对每一行里出现 `/ <模型参数名>` 的语句，检查**上方 8 行**内是否有
`if (... <= 0 ...)` 之类的守卫。没有就报出来。

局限（KNOWN_GAPS 那一节）
------------------------
* 「上方 8 行」是启发式：守卫写在别处（同一个函数开头）会漏判，
  写在更远处也会漏判。反过来，守卫写了但没 `return`（只是没做事）会被误判成有。
* 只认 `<= 0` / `< 1` 这类**比较**；`assert(x)`、`if (!x)` 认不出来。
* **它不知道那个"参数"是不是真的来自模型文件** —— 名单是写死的
  （见 `PARAM_NAMES`），改模型格式时要同步改这里。

用法:
    python tools/check_div_guard.py                 # 全扫，退出码 1 表示有漏
    python tools/check_div_guard.py --selfcheck     # 自测
    python tools/check_div_guard.py --list         # 只列出所有命中点（含已守卫）
"""

from __future__ import print_function

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
ALLOWLIST = os.path.join(HERE, 'div_guard_allowlist.txt')
TREE = ['ZQCNN', 'ZQlibFaceID', 'SamplesZQCNN', 'SamplesZQlibFaceID']
SKIP_DIRS = {'.git', 'build_x64', 'cmake-out-win32-x64', 'cmake-out-unix-x64',
             '3rdparty', '__pycache__'}

# 会被当成"来自模型文件的参数"的名字
PARAM_NAMES = ['stride', 'strideH', 'strideW', 'kernel_H', 'kernel_W',
               'local_size', 'dilate_H', 'dilate_W', 'pad_H', 'pad_W',
               'tile_n', 'tile_h', 'tile_w', 'tile_c', 'hidden_dim',
               'num_output', 'axis', 'keep_top_k', 'nms_top_k', 'step_h', 'step_w']

DIV_RE = re.compile(r'/\s*(' + '|'.join(re.escape(n) for n in PARAM_NAMES) + r')\b')
# 守卫：同一 if 里出现 <= 0 / < 1 / != 0
GUARD_RE = re.compile(r'if\s*\([^)]*(?:<=\s*0|<\s*1|!=\s*0)[^)]*\)')

SELFTEST = '''
// A: 有守卫
int f1(int in_H, int strideH) {
    if (strideH <= 0 || strideW <= 0)
        return false;
    int need_H = (in_H - 1) / strideH + 1;
    return need_H > 0;
}
// B: 没守卫
int f2(int in_H, int strideH) {
    int need_H = (in_H - 1) / strideH + 1;
    return need_H;
}
// C: 整行注释，不算
// int need_H = (in_H - 1) / strideH + 1;
// D: 行尾注释，不算
int f4(int in_H, int strideH) {
    int a = in_H;  // 除以 strideH 只是一句说明
    return a;
}
'''


BLOCK_OPEN = '/*'
BLOCK_CLOSE = '*/'


def strip_comments(lines):
    """把注释整段换成空（保持行号），返回"去注释后的行"列表。

    第一版只处理 `//` 和"行首是 * 的行"，于是
        /*if (...) std::cout << "... step_h/step_w ...";*/
    这种**块注释里**的字符串被当成了真除法（真扫出来 1 条假警报）。
    后来补的块注释状态机写错了（`while` 里 break 之后又去判断同一行），
    结果把**真代码**也抹掉了 —— 131 处 vs 正确的 96 处，一眼就看得出不对。
    教训见 AGENTS.md：**别手写半吊子的解析器**。
    """
    out = []
    in_block = False
    for line in lines:
        if in_block:
            idx = line.find(BLOCK_CLOSE)
            if idx < 0:
                out.append('')
                continue
            line = line[idx + 2:]
            in_block = False
        while True:
            i = line.find(BLOCK_OPEN)
            if i < 0:
                break
            j = line.find(BLOCK_CLOSE, i + 2)
            if j < 0:
                line = line[:i]
                in_block = True
                break
            line = line[:i] + line[j + 2:]
        k = line.find('//')
        if k >= 0:
            line = line[:k]
        out.append(line)
    return out


def scan_lines(raw_lines, path):
    lines = strip_comments(raw_lines)
    hits = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        m = DIV_RE.search(line)
        if not m:
            continue
        # 上下文只取**同一个函数体内**的向上若干行。
        # 第一版直接取上方 8 行，于是函数 A 的守卫会"罩住"紧随其后的函数 B ——
        # 自测里那个没守卫的 f2 就是这样被误判成有守卫的。
        # 切点：从本行往上，遇到第一个「行首是 } 的行」就停（那通常是上一个函数结束）。
        start = max(0, i - 8)
        for k in range(i - 1, max(-1, i - 9), -1):
            if raw_lines[k].startswith('}'):
                start = k + 1
                break
        ctx = '\n'.join(lines[start:i])
        guarded = bool(GUARD_RE.search(ctx))
        hits.append({'path': path, 'line': i + 1, 'param': m.group(1),
                     'guarded': guarded, 'text': s[:100]})
    return hits


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    argv = list(sys.argv[1:])
    if '--selfcheck' in argv:
        hits = scan_lines(SELFTEST.split('\n'), '<selftest>')
        got = set((h['param'], h['guarded']) for h in hits)
        want = {('strideH', True), ('strideH', False)}
        if got != want:
            print('自测失败。\n  期望: %s\n  实际: %s' % (sorted(want), sorted(got)))
            return 1
        print('自测通过：2 处除法（1 有守卫 / 1 漏网）都识别正确；'
              '整行注释与行尾注释里的都不算。')
        return 0

    all_hits = []
    for t in TREE:
        base = os.path.join(ROOT, t)
        if not os.path.isdir(base):
            continue
        for dp, dn, fn in os.walk(base):
            dn[:] = [d for d in dn if d not in SKIP_DIRS]
            for f in sorted(fn):
                if not f.endswith(('.h', '.cpp', '.c')):
                    continue
                p = os.path.join(dp, f)
                with io.open(p, encoding='utf-8', errors='replace') as fh:
                    all_hits += scan_lines(fh.read().replace('\r\n', '\n').split('\n'),
                                          os.path.relpath(p, ROOT))

    if '--list' in argv:
        for h in all_hits:
            print('%-46s %-5d %-4s %-9s %s'
                  % (h['path'], h['line'], h['param'],
                     'GUARDED' if h['guarded'] else '** NO **', h['text']))
        print('\n共 %d 处' % len(all_hits))
        return 0

    unguarded = [h for h in all_hits if not h['guarded']]

    # 人工确认"守卫在别处"的白名单。**必须连理由一起写** —— 没有理由的白名单
    # 半年后就没人知道为什么放行，那还不如直接删掉。
    allow = {}
    if os.path.isfile(ALLOWLIST):
        with io.open(ALLOWLIST, encoding='utf-8') as fh:
            for line in fh:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 3:
                    # 路径归一化成 "/"：扫描结果走 os.path.relpath（Windows 上是
                    # 反斜杠），而白名单是人手写的（习惯写正斜杠）。不归一化的话
                    # 白名单会**静默地一条都匹配不上**，看起来像"没有白名单"。
                    allow[(parts[0].replace('\\', '/'), parts[1])] = parts[2]
    allow_hits = [h for h in unguarded
                  if (h['path'].replace('\\', '/'), h['param']) in allow]
    real = [h for h in unguarded
            if (h['path'].replace('\\', '/'), h['param']) not in allow]

    if '--save-allowlist' in argv:
        out = ['# tools/check_div_guard.py 的人工白名单',
               '# 格式: <文件路径(相对仓库)>\\t<参数名>\\t<为什么它安全>',
               '#',
               '# 只放"守卫在**别的**函数里、而且已经人工核实过"的情况。',
               '# 本工具只看得到除法所在函数体内上方的 8 行，看不到 InitFromBuffer 那种。',
               '# 没有理由的行不要加 —— 半年后没人知道为什么放行。']
        for (p, param), why in sorted(allow.items()):
            out.append('%s\t%s\t%s' % (p, param, why))
        with io.open(ALLOWLIST, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('\n'.join(out) + '\n')
        print('白名单已写入 %s（%d 条）' % (ALLOWLIST, len(allow)))
        return 0

    print('=' * 78)
    print('"除以模型参数" 守卫普查：%d 处 / 已守卫 %d / 白名单 %d / **待查 %d**'
          % (len(all_hits), len(all_hits) - len(unguarded), len(allow_hits), len(real)))
    print('=' * 78)
    if allow_hits:
        print('\n白名单（守卫在别处，已人工核实）:')
        for h in sorted(allow_hits, key=lambda x: (x['path'], x['line'])):
            key = (h['path'].replace(os.sep, '/'), h['param'])
            print('  %-46s:%-5d /%-8s  %s'
                  % (h['path'], h['line'], h['param'], allow[key]))
    for h in real:
        print('\n  %-46s:%-5d /%s' % (h['path'], h['line'], h['param']))
        print('      %s' % h['text'])
    if real:
        print('\n每一处都要人眼确认：那个参数是不是真的可能为 0、'
              '上游是不是已经保证了它 > 0。')
    elif not real and not allow_hits:
        print('\n全部已守卫。')
    return 1 if real else 0


if __name__ == '__main__':
    sys.exit(main())
