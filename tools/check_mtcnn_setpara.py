#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MTCNN 五个变体的 SetPara / 多线程索引一致性门禁 —— 附录 II。

为什么要这道门禁
----------------
`ZQCNN/ZQ_CNN_MTCNN*.h` 有**五份** MTCNN 实现（`_Interface` / NCHWC / ncnn /
`_AspectRatio` / 主副本）。它们是**逐字拷贝**出来的，所以任何一处修改
都必须五份一起改 —— 这已经是第四次栽在「孪生副本」上（IH.9 / BE.2 / IX.14 /
conv_overflow_guard 那一轮）。

而附录 II 在它们身上一次挖出**六个**问题，其中三个的共同点是
**「只有源码能看见，跑 sample 看不见」**：

  II.1  `_Interface` 多线程 lnet：Forward 用 `lnet[thread_id]`，
        紧接着读 blob 却用 `lnet[0]` —— 0 号线程此刻正在改写它。
        仓内 sample 全部 `thread_num=0`（被 `__max(1,...)` 夹成 1），
        `lnet[0] == lnet[thread_id]`，**跑一万次也看不出差别**。
  II.2  `_Interface` 多线程 lnet：下颌/眼周那 29 个点带一个**活的** `* 0.5`，
        单线程支路同一处是 `/**0.25*/`（注掉的），参考实现 `ZQ_CNN_MTCNN.h`
        同一处是 `/**0.5*/`（也是注掉的）。只有 `thread_num>1` 才走到。
  II.5  `_Interface` 串行支路（`thread_num<=1`）用 `omp_get_thread_num()`
        去索引大小**恰好是 thread_num** 的 `pnet` / `task_pnet_images`。
        仓内没有调用方把 `Find` 放进自己的 parallel 区，实测恒返回 0。
  II.6  `ZQ_CNN_MTCNN_ncnn.h` **漏了**另外四份都有的
        `pnet_size/pnet_stride = __max(1, ...)` 夹取 ——
        上一轮修另外四份时漏了这一份。`pnet_size==0` 时整除 SIGFPE。

另外两个（II.3 / II.4）互相咬合：
  II.3  `mapH/mapW/maps` 只为「通过 `changedH < pnet_size` 过滤的 scale」建，
        是**紧凑下标**；`task_scale_id.push_back(i)` 存的是 `scales` 的**全局下标**；
        消费端又用紧凑下标去取 `scales[i]`。三处混用，只要有一个 scale 被过滤就错位，
        其中 `maps[scale_id][...]` 那处是越界**写**。
  II.4  `SetPara` 的缓存失效条件只比 width/height/scale_factor，
        漏了 `pnet_size` / `min_size` / `special_handle_very_big_face` ——
        而 `scales` 的**生成**恰恰依赖这三个。二次 SetPara 改 pnet_size
        就会用旧 scale 打破 II.3 那个不变量。

**这六条没有一条能靠跑 sample 发现**（4 个 MTCNN sample 的输出在修复前后
逐字节相同 —— 这本身就是"修得对"的证据，但也说明它们对这几条是瞎的）。
所以只能写成**源码级**门禁，和 A17~A22 同一套路。

判定
----
A23  五个变体都必须有 `pnet_size` / `pnet_stride` 的 `__max(1, ...)` 夹取
A24  五个变体的 SetPara 失效条件都必须比较 `old_pnet_size` / `old_min_size` /
     `old_special_big`（也就是必须**先存旧值**再赋值）
A25  五个变体都必须在 `pnet_images.resize(count)` 之后做「剔掉会被过滤的 scale」
     的兜底（II.3 的不变量保证）
A26  `ZQ_CNN_MTCNN_Interface.h` 的 lnet 分支里不得有活的 `* 0.5`
     （`/**...*/` 里的不算）
A27  `ZQ_CNN_MTCNN_Interface.h` 不得出现 `lnet[0].GetBlobByName` 与
     `lnet[thread_id].Forward` 同行区（Forward 后紧跟读 blob 却用 [0]）
A28  `ZQ_CNN_MTCNN_Interface.h` 的 `thread_num <= 1` 分支里不得出现
     `omp_get_thread_num()`

用法
----
    python tools/check_mtcnn_setpara.py              # 扫默认文件
    python tools/check_mtcnn_setpara.py --selfcheck  # 先自测
    python tools/check_mtcnn_setpara.py <文件路径>    # 扫指定文件

自测样本里**必须有故意不合格的项**（AGENTS.md「写检查类工具的四条硬规矩」第 1 条）。
"""
from __future__ import print_function

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILES = [
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_MTCNN.h'),
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_MTCNN_AspectRatio.h'),
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_MTCNN_Interface.h'),
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_MTCNN_NCHWC.h'),
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_MTCNN_ncnn.h'),
]

# ------------------------------------------------------------------ 各条判定
# A23: pnet_size / pnet_stride 必须夹到 >= 1
RE_A23 = re.compile(r'this->pnet_size\s*=\s*__max\(\s*1\s*,\s*pnet_size\s*\)')
RE_A23b = re.compile(r'this->pnet_stride\s*=\s*__max\(\s*1\s*,\s*pnet_stride\s*\)')

# A24: 失效条件里必须比较旧值 —— 旧值的快照也必须在（否则拿新值比新值）
RE_A24_SNAP = re.compile(
    r'int\s+old_pnet_size\s*=\s*this->pnet_size\s*;.*?'
    r'int\s+old_min_size\s*=\s*min_size\s*;.*?'
    r'bool\s+old_special_big\s*=\s*special_handle_very_big_face\s*;',
    re.S)
# 三个比较**分开**判：源码里用的是 `||`（任一参数变了就重建），
# 但写死成 `&&` 会让「换成 && 的等价实现」被误判为不合格 —— 那不是缺陷。
# 这里只要求三个比较**都出现**。
RE_A24_CMP = [
    re.compile(r'old_pnet_size\s*!=\s*this->pnet_size'),
    re.compile(r'old_min_size\s*!=\s*min_size'),
    re.compile(r'old_special_big\s*!=\s*special_handle_very_big_face'),
]

# A25: pnet_images.resize(count) 之后必须有剔掉被过滤 scale 的兜底
RE_A25 = re.compile(
    r'pnet_images\.resize\(count\)\s*;.*?int\s+kept\s*=\s*0\s*;.*?'
    r'changedH\s*<\s*pnet_size\s*\|\|\s*changedW\s*<\s*pnet_size.*?'
    r'scales\[kept\+\+\]\s*=\s*scales\[i\]\s*;.*?scales\.resize\(kept\)\s*;', re.S)

# A26: lnet 分支里不得有**活的** `* 0.5`（注释里的 `/**0.5*/` 不算）
RE_A26_LIVE = re.compile(r'keyPoint_ptr\[[^\]]*\]\s*\*\s*0\.5\s*;')

# A27：`lnet[X].Forward` 之后**最近的下一个** `lnet[Y].GetBlobByName` 必须 X == Y。
#
# 第一版写成「Forward 之后 400 字符内」—— 结果自己写的修复说明注释
# （约 500 字）把那个窗口撑破了，变异测试立刻发现 A27 抓不到 ——
# **窗口大小是这种规则的隐含假设，注释一长就失效。**
# 改成"最近的下一个"就没有窗口，也就不会被注释长度影响。
#
# 下标的字符类**必须包含数字**：第一版写的是 `[A-Za-z_][A-Za-z0-9_]*`，
# 于是 `lnet[0].GetBlobByName` 根本匹配不上 —— 变异测试把 A27 退回成
# `lnet[0]` 之后门禁**照样全绿**。下标既可能是 `thread_id` 也可能是字面量 `0`，
# 写死成标识符就等于把最关键的那个 case 排除在外了。
RE_A27_FWD = re.compile(r'lnet\[([A-Za-z0-9_]+)\]\.Forward\s*\(')
RE_A27_GET = re.compile(r'lnet\[([A-Za-z0-9_]+)\]\.GetBlobByName\s*\(')

# A28: `thread_num <= 1` 分支体内不得有 omp_get_thread_num()
RE_A28_COND = re.compile(r'if\s*\(\s*thread_num\s*<=\s*1\s*\)')

# A29: SetPara 必须把夹取后的 scale_factor 落到**成员** factor
#     （原来那句改的是形参，成员 factor 从构造函数起就一直是 0.709）
RE_A29_BAD = re.compile(r'^\s*scale_factor\s*=\s*__max\(', re.M)
RE_A29_NEW = re.compile(r'const\s+float\s+new_factor\s*=.*?__max\(0\.5f', re.S)
RE_A29_OLD = re.compile(r'const\s+float\s+old_factor\s*=\s*factor\s*;')
RE_A29_ASSIGN = re.compile(r'this->factor\s*=\s*new_factor\s*;')
RE_A29_CMP = re.compile(r'old_factor\s*!=\s*new_factor')

# A30: AspectRatio 的 Pnet **任务循环**里，族判断与派生下标必须用 scale_id。
#
#      第一版写成「全文不许出现 `if (i < ori_num)`」—— 结果误报了 5 处：
#      那些是 `for (int i = 0; i < total_scale_num; i++)` 的**尺度**循环，
#      那里 `i` **就是**尺度下标，`if (i < ori_num)` / `maps[i]` / `mapH[i]`
#      全都正确。真正有问题的只有**任务**循环
#      （`for (int i = 0; i < task_num; i++)` + `int scale_id = task_scale_id[i];`）
#      里那几个：任务表按 scale 顺序追加，`scale_id <= i`，
#      一个 scale 产出多个块时两者就分离了。
#      所以判据必须**锚在 `scale_id = task_scale_id[i]` 上**，不能全文扫。
#      —— 这也是「扫描器过度敏感」和「扫描器过窄」一样有害的又一例。
RE_A30_TASK = re.compile(r'int\s+scale_id\s*=\s*task_scale_id\[i\]\s*;')
RE_A30_BAD = [
    re.compile(r'if\s*\(\s*i\s*<\s*ori_num\s*\)'),
    re.compile(r'else\s+if\s*\(\s*i\s*<\s*ori_num\s*\+'),
    re.compile(r'int\s+j\s*=\s*i\s*-\s*ori_num\s*;'),
    re.compile(r'int\s+k\s*=\s*i\s*-\s*ori_num\s*-\s*xhalf_num\s*;'),
]
A30_WINDOW = 3000   # 任务循环的分派块约 45 行；用窗口而不是配对花括号，够用且不会跨到别的循环

# A31: 空任务守卫必须判**当前槽位**，不能只判外层 vector（外层 size == need_thread_num，恒 >= 1）
RE_A31_BAD = re.compile(r'^\s*if\s*\(\s*task_src_off_x\.size\(\)\s*==\s*0\s*\)', re.M)
RE_A31_GOOD = re.compile(r'task_src_off_x\[pp\]\.size\(\)\s*==\s*0')

# A32: block_end 的终止条件必须用**每行/每列的块数**，不是块总数
RE_A32_BAD = [
    re.compile(r'block_end_w\[bb\]\s*=\s*\(\s*bw\s*==\s*block_num\s*-\s*1\s*\)'),
    re.compile(r'block_end_h\[bb\]\s*=\s*\(\s*bh\s*==\s*block_num\s*-\s*1\s*\)'),
]
RE_A32_GOOD = [
    re.compile(r'block_end_w\[bb\]\s*=\s*\(\s*bw\s*==\s*block_W_num\s*-\s*1\s*\)'),
    re.compile(r'block_end_h\[bb\]\s*=\s*\(\s*bh\s*==\s*block_H_num\s*-\s*1\s*\)'),
]

# A33: block 循环的 pragma 必须带 reduction 子句
RE_A33_PRAGMA = re.compile(
    r'#pragma\s+omp\s+parallel\s+for\s+schedule\(dynamic,\s*chunk_size\)'
    r'[^\n]*\bnum_threads\(thread_num\)')
RE_A33_RED = re.compile(r'reduction\(\s*\+\s*:\s*before_count\s*,\s*after_count\s*\)')


# ---- A34~A38（附录 II.12~II.16）----
# A34: `score` / `location` 的 GetBlobByName 之后必须有判空。
#      GetBlobByName 找不到就返回 0（ZQ_CNN_Net.h:295-300），而 Init 对 blob 名零校验。
#      `keyPoint` 一直**有**判空、`score`/`location` 没有 —— 判据不一致本身就是信号。
RE_A34_DECL = re.compile(
    r'const\s+ZQ_CNN_Tensor4D\w*\*\s*(score|location)\s*=\s*\w+\[\w*\]\.GetBlobByName\(')
A34_WINDOW = 1200   # 紧跟着的判空 + 注释

# A35: Init 的 thread_num 必须有上界（按份复制整模型）
RE_A35_BAD = re.compile(r'^\s*thread_num\s*=\s*__max\(1,\s*thread_num\)\s*;\s*$', re.M)
RE_A35_CAPPED = re.compile(r'if\s*\(\s*thread_num\s*>\s*\d+\s*\)')

# A36: SetPara 入口必须校验 w/h（minside==0 -> scale=+inf -> (int)ceil(inf) 是 UB）
RE_A36 = re.compile(r'if\s*\(\s*w\s*<=\s*0\s*\|\|\s*h\s*<=\s*0\s*\)')

# A37: special_handle_very_big_face 的 tmp_size 循环必须有 count 上界
RE_A37_LOOP = re.compile(
    r'for\s*\(int tmp_size = last_size - 1;[^;]*;\s*tmp_size -= 2\)')
RE_A37_GOOD = re.compile(r'count\s*<\s*\d+')

# A38: 张量版 Find 必须有尺寸守卫（bgr 版本来就有）
# 「哪个 Find」用**纯字符串**判，不用正则 —— 这条规则被自己的正则坑过三轮：
#   1. 返回类型写死成 `bool`，换个返回类型就**静默不执行**（扫不到 != 没问题）；
#   2. heredoc 吃掉一层转义，反斜杠 s 变成字面量；
#   3. 为躲 2 改成拼接，还是不对。
# 字符串判断没有转义层，也就没有这三类问题。
A38_FIND_MARK = 'Find(ZQ_CNN_Tensor4D_Interface&'
RE_A38_GUARD = re.compile(r'input\.GetW\(\)\s*!=\s*width\s*\|\|\s*input\.GetH\(\)\s*!=\s*height')


# ---- A39/A40（附录 II.17/II.18）----
# A39: bgr 入口必须校验像素缓冲本身（指针 / 尺寸 / widthStep）。
#      ConvertFromBGR 是 `bgr_row = BGR_img + h*_widthStep` 再逐像素 `bgr_pix += 3`，
#      所以 nullptr 直接崩、`_widthStep <= 0` 越过缓冲前端、`_widthStep < _width*3`
#      在最后一行越过缓冲末尾 —— 三种都是调用方一个笔误。
#      **不能只判「文件里出现过这个 if」**：有些变体根本没有 bgr 入口。
A39_ENTRY = re.compile(
    r'\b(?:bool|void)\s+(?:Find|_Pnet_stage|Find106)\s*\(\s*const unsigned char\*\s*bgr_img')
A39_GUARD = re.compile(
    r'bgr_img\s*==\s*0.*?_width\s*<=\s*0.*?_widthStep\s*<\s*_width\s*\*\s*3', re.S)
# 每个 bgr 入口**各自**都要有守卫。原来只判「文件里存在一处」——
# 变异测试把 Interface.h 的两处之一改坏，门禁照样全绿。
# 同一函数的两个重载（Find / Find106）形状完全一样，最容易只改一处。

# A40: Rnet/Onet 的 ResizeBilinearRect 失败分支必须先清空该槽的框。
#      原来只有裸 `continue`：这一槽的框一个都没被评过，却仍然 exist=true、
#      score 还是上一阶段的旧分数，随后的汇总把它们带进 NMS 当 hero。
A40_RESIZE = re.compile(r'if \(!input\.ResizeBilinearRect\(task_(?:rnet|onet)_images\[pp\]')
A40_CLEAR = re.compile(r'task_(?:second|third)Bbox\[pp\]\.clear\(\)\s*;')

def _read(path):
    with io.open(path, 'r', encoding='utf-8') as f:
        return f.read()


# 字符串字面量（粗略）：用来在剥注释时不把 "/*" 当注释起点
_RE_STR = re.compile(r'"(?:\\.|[^"\\])*"' + r"|'(?:\\.|[^'\\])*'")


def strip_comments(text):
    """把 C++ 注释替换成等长空格（**保留换行**，行号才不变）。

    为什么必须剥
    ------------
    第一版没剥，于是 A28 匹配到了**我自己写的修复说明注释**里那句
    「这里原来写的是 `omp_get_thread_num()`」—— 门禁被自己的文档绊倒，
    报了一个根本不存在的缺陷。
    而 A26 更是**必须**剥：`/**0.5*/` 是注释（= x1，正确），
    活的 `* 0.5` 是代码（= 错）。不剥就没法用一条正则把两者分开，
    只能靠"数一下前面有没有 `/*`"这种脆办法。
    —— 附录 IX.23 那两条"静默少扫"的正则同一个教训：
      **扫描器要么只看代码，要么明确处理注释，不能含糊。**
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            m = _RE_STR.match(text, i)
            if m:
                out.append(m.group(0))
                i = m.end()
                continue
            out.append(c)
            i += 1
            continue
        if c == '/' and i + 1 < n:
            nxt = text[i + 1]
            if nxt == '/':
                j = text.find('\n', i)
                if j < 0:
                    j = n
                out.append(' ' * (j - i))
                i = j
                continue
            if nxt == '*':
                j = text.find('*/', i + 2)
                j = n if j < 0 else j + 2
                seg = text[i:j]
                out.append(''.join(ch if ch == '\n' else ' ' for ch in seg))
                i = j
                continue
        out.append(c)
        i += 1
    return ''.join(out)


def _block_after(text, pos):
    """从 pos 处的 '{' 开始返回配对花括号的内容（找不到就返回到文件尾）。"""
    i = text.find('{', pos)
    if i < 0:
        return ''
    depth = 0
    j = i
    n = len(text)
    while j < n:
        c = text[j]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return text[i + 1:]


def scan_text(raw, label='<text>'):
    """返回 (合格项数, [(规则号, 说明)])。

    **判定一律在剥掉注释之后的代码上做**（见 strip_comments 的说明），
    但**行号仍取自原文** —— 剥注释保留了换行，所以行号不变。
    """
    text = strip_comments(raw)
    bad = []
    ok = 0
    is_iface = 'MTCNN_Interface' in label

    # A23 / A24 / A25
    if RE_A23.search(text) and RE_A23b.search(text):
        ok += 1
    else:
        bad.append(('A23', 'pnet_size/pnet_stride 没有 __max(1,...) 夹取'
                    '（pnet_size==0 时整数除零 SIGFPE；<0 时 scales 涨到 OOM）'))
    if RE_A24_SNAP.search(text) and all(r.search(text) for r in RE_A24_CMP):
        ok += 1
    else:
        bad.append(('A24', 'SetPara 失效条件没比较 old_pnet_size/old_min_size/'
                    'old_special_big（二次 SetPara 会复用按旧 pnet_size 生成的 scales）'))
    if RE_A25.search(text):
        ok += 1
    else:
        bad.append(('A25', 'pnet_images.resize(count) 之后没有「剔掉会被 pnet_size '
                    '过滤的 scale」兜底（mapH/maps 紧凑下标 vs task_scale_id 全局下标）'))

    # A26 只对 _Interface 有意义（多线程 lnet 那份）
    if is_iface:
        m = RE_A26_LIVE.search(text)
        if m:
            bad.append(('A26', 'lnet 分支里有**活的** `* 0.5`（原文行 %d）—— '
                       '单线程支路与参考实现 ZQ_CNN_MTCNN.h 都是注掉的(=x1)'
                       % (text[:m.start()].count('\n') + 1)))
        else:
            ok += 1

        # A27：Forward 用哪个下标，紧接着读 blob 就必须用同一个
        for m in RE_A27_FWD.finditer(text):
            fwd_idx = m.group(1)
            g = RE_A27_GET.search(text, m.end())
            if g is None:
                continue                       # 这一支后面没有读 blob，不归本条管
            if g.group(1) == fwd_idx:
                ok += 1
            else:
                bad.append(('A27', '`lnet[%s].Forward` 之后最近的读 blob 是 '
                            '`lnet[%s].GetBlobByName`（原文行 %d）—— '
                            '%s 号那份此刻正被别的线程改写'
                            % (fwd_idx, g.group(1),
                               text[:g.start()].count('\n') + 1,
                               '0' if fwd_idx != '0' else fwd_idx)))

        # A28：thread_num<=1 分支体内不得有 omp_get_thread_num()
        for m in RE_A28_COND.finditer(text):
            blk = _block_after(text, m.end())
            if 'omp_get_thread_num' in blk:
                bad.append(('A28', '`thread_num <= 1` 分支体内用了 omp_get_thread_num()'
                           '（行 %d）—— 调用方把 Find 放进自己的 parallel 区时它返回'
                           '**外层**线程号，可 >= thread_num'
                           % (text[:m.start()].count('\n') + 1)))
            else:
                ok += 1

    # ---- A29~A33：五个变体都要过（AspectRatio 的 A30 额外单独判）----
    if RE_A29_BAD.search(text):
        bad.append(('A29', 'SetPara 里还有 `scale_factor = __max(...)` —— '
                    '它改的是**形参**，成员 factor 从构造函数起就一直是 0.709，'
                    '调用方传的 scale_factor 完全无效'))
    elif not (RE_A29_NEW.search(text) and RE_A29_OLD.search(text)
              and RE_A29_ASSIGN.search(text) and RE_A29_CMP.search(text)):
        bad.append(('A29', 'SetPara 没把夹取后的 scale_factor 落到成员 factor'
                    '（缺 new_factor / old_factor / this->factor= / 比较用 old_factor!=new_factor 之一）'))
    else:
        ok += 1

    if 'task_src_off_x' not in text:
        pass                     # 本文件根本没有这套变量（比如 ncnn.h 换了自己的命名），A31 不适用
    elif RE_A31_BAD.search(text):
        bad.append(('A31', '还有裸的 `if (task_src_off_x.size() == 0)` —— 外层 vector 的大小是 '
                    'need_thread_num，**恒 >= 1**，这个守卫永远不成立；'
                    '真正会为空的是当前槽位 task_src_off_x[pp]'))
    elif not RE_A31_GOOD.search(text):
        bad.append(('A31', '一处都没有 `task_src_off_x[pp].size() == 0`（空任务槽位没守卫）'))
    else:
        ok += 1

    bad32 = [r.pattern for r in RE_A32_BAD if r.search(text)]
    if bad32:
        bad.append(('A32', 'block_end 仍用 `block_num - 1`（块总数）而不是 '
                    'block_W_num/block_H_num - 1（每行/每列的块数）—— '
                    '除最后一个块外都扫不到边缘，贴边的脸会漏'))
    elif not all(r.search(text) for r in RE_A32_GOOD):
        bad.append(('A32', 'block_end_w/block_end_h 没有用 block_W_num/block_H_num'))
    else:
        ok += 1

    n_pragma = len(RE_A33_PRAGMA.findall(text))
    n_red = len(RE_A33_RED.findall(text))
    if n_pragma == 0:
        bad.append(('A33', '找不到 block 循环的 `#pragma omp parallel for '
                    'schedule(dynamic, chunk_size) num_threads(thread_num)`'))
    elif n_red < n_pragma:
        bad.append(('A33', 'block 循环的 pragma 有 %d 处、带 reduction 子句的只有 %d 处 —— '
                    'before_count/after_count 无锁 += 是数据竞争' % (n_pragma, n_red)))
    else:
        ok += 1

    # A30 只对 AspectRatio 有意义：它是唯一有 xhalf/yhalf 三族分派的
    is_ar = 'AspectRatio' in label
    if is_ar:
        hits = []
        for m in RE_A30_TASK.finditer(text):
            win = text[m.end():m.end() + A30_WINDOW]
            for r in RE_A30_BAD:
                if r.search(win):
                    ln = text[:m.end() + r.search(win).start()].count(chr(10)) + 1
                    hits.append((r.pattern, ln))
                    break
        if hits:
            bad.append(('A30', 'Pnet **任务**循环里仍用任务下标 `i` 判族/算 j、k（应全部用 scale_id）'
                        '—— k 可为负 => std::vector::operator[](负) 越界 -> 野张量上 .ROI()。'
                        '（尺度循环 `for (i < total_scale_num)` 里的 `i` 是对的，不归本条管。）'
                        + '；'.join('行 %d' % ln for _, ln in hits)))
        else:
            ok += 1

    # ---- A34~A38 ----
    a34 = []
    for m in RE_A34_DECL.finditer(text):
        v = m.group(1)
        if not re.search(v + r'\s*==\s*0', text[m.end():m.end() + A34_WINDOW]):
            a34.append(text[:m.start()].count(chr(10)) + 1)
    if a34:
        bad.append(('A34', '`score` / `location` 的 GetBlobByName 之后没有判空（行 %s）—— '
                    'GetBlobByName 找不到返回 0，下一行 score->GetH() 就是空指针解引用；'
                    'Init 对 blob 名零校验，传个名字不对的模型就踩得到'
                    % ', '.join(str(x) for x in a34)))
    else:
        ok += 1

    if RE_A35_BAD.search(text) and not RE_A35_CAPPED.search(text):
        bad.append(('A35', 'Init 里 thread_num 只夹了**下界**就 pnet.resize(thread_num) 并逐份 '
                    'LoadFrom —— 传 100000 就是 30 万份模型常驻内存'))
    else:
        ok += 1

    if not RE_A36.search(text):
        bad.append(('A36', 'SetPara 入口没有 `w <= 0 || h <= 0` 守卫 —— minside==0 时 '
                    '`scales.push_back(pnet_size/minside)` 是 +inf，'
                    '消费端 `(int)ceil(height*scales[i])` 是 float->int 的 UB'))
    else:
        ok += 1

    a37 = [m.group(0) for m in RE_A37_LOOP.finditer(text) if not RE_A37_GOOD.search(m.group(0))]
    if a37:
        bad.append(('A37', 'special_handle_very_big_face 的 tmp_size 循环没有 count 上界（%d 处）—— '
                    '次数 ~minside/2，20000x20000 的图近 1 万个 scale -> GB 级内存；'
                    'last_size > INT_MAX 时 `int tmp_size = last_size - 1` 本身就是 UB'
                    % len(a37)))
    else:
        ok += 1

    if A38_FIND_MARK in text and not RE_A38_GUARD.search(text):
        bad.append(('A38', '张量版 Find 没有尺寸守卫（bgr 版第一行就有）—— '
                    'scales / pnet_images / width / height 全是按 SetPara 那对尺寸生成的，'
                    '换个尺寸的图进来所有几何全按过期尺寸算'))
    else:
        ok += 1

    # ---- A39/A40 ----
    # 窗口到**下一个入口**为止，不用固定字符数。
    # 固定窗口（试过 2000）会跨进下一个函数体里，于是「这一个入口没守卫、
    # 下一个有」被判成全过 —— 和 A27 那个 400 字符窗口一个毛病。
    entries = list(A39_ENTRY.finditer(text))
    a39 = []
    for k, m in enumerate(entries):
        stop = entries[k + 1].start() if k + 1 < len(entries) else len(text)
        if not A39_GUARD.search(text[m.end():stop]):
            a39.append(text[:m.start()].count(chr(10)) + 1)
    if a39:
        bad.append(('A39', 'bgr 入口（行 %s）没有校验像素缓冲（bgr_img / _width / _height / '
                    '_widthStep < _width*3）—— ConvertFromBGR 是 '
                    '`BGR_img + h*_widthStep` 再逐像素 +3，nullptr 直接崩、'
                    'widthStep<=0 越过前端、widthStep<_width*3 在最后一行越过末尾'
                    % ', '.join(str(x) for x in a39)))
    else:
        ok += 1

    a40 = [m.start() for m in A40_RESIZE.finditer(text)]
    a40_ok = 0
    for st in a40:
        win = text[st:st + 1200]
        if A40_CLEAR.search(win):
            a40_ok += 1
    if a40 and a40_ok < len(a40):
        bad.append(('A40', 'Rnet/Onet 的 ResizeBilinearRect 失败分支有 %d/%d 处是裸 `continue` —— '
                    '该槽的框一个都没被评过，却仍然 exist=true、score 还是上一阶段的旧分数，'
                    '随后的汇总把它们带进 NMS 当 hero 抑制掉真正的框。'
                    '正确写法是先 `task_secondBbox[pp].clear()` / `task_thirdBbox[pp].clear()`。'
                    % (len(a40) - a40_ok, len(a40))))
    else:
        ok += 1

    return ok, bad



# FULL 是一份"什么都齐"的骨架，后面每条自测样本都在它基础上**只破坏一处**。
# 这样每条自测的期望集合都是可推导的，而不是拍脑袋写的。
FULL = r"""
void SetPara(int w, int h, float scale_factor = 0.709) {
	if (w <= 0 || h <= 0) { w = 1; h = 1; }
	const float new_factor = (float)__max(0.5f, __min(0.97f, scale_factor));
	const float old_factor = factor;
	this->factor = new_factor;
	this->pnet_size = __max(1, pnet_size);
	this->pnet_stride = __max(1, pnet_stride);
	int old_pnet_size = this->pnet_size;
	int old_min_size = min_size;
	bool old_special_big = special_handle_very_big_face;
	if (width != w || height != h || old_factor != new_factor
		|| old_pnet_size != this->pnet_size || old_min_size != min_size
		|| old_special_big != special_handle_very_big_face)
	{
		pnet_images.resize(count);
		{
			int kept = 0;
			for (int i = 0; i < (int)scales.size(); i++) {
				int changedH = (int)ceil(height * scales[i]);
				int changedW = (int)ceil(width * scales[i]);
				if (changedH < pnet_size || changedW < pnet_size) continue;
				scales[kept++] = scales[i];
			}
			if (kept != (int)scales.size()) { scales.resize(kept); }
		}
	}
	int block_H_num = 1, block_W_num = 1, block_num = 1;
	int scoreH = 1, scoreW = 1, width_per_block = 1, height_per_block = 1;
	for (int bh = 0; bh < block_H_num; bh++) for (int bw = 0; bw < block_W_num; bw++) {
		int bb = bh * block_W_num + bw;
		block_end_w[bb] = (bw == block_W_num - 1) ? scoreW : ((bw + 1)*width_per_block);
		block_end_h[bb] = (bh == block_H_num - 1) ? scoreH : ((bh + 1)*height_per_block);
	}
	if (task_src_off_x.size() == 0 || task_src_off_x[pp].size() == 0) continue;
#pragma omp parallel for schedule(dynamic, chunk_size) num_threads(thread_num) reduction(+:before_count, after_count)
	for (int bb = 0; bb < block_num; bb++) { before_count += tmp_before_count; after_count += tmp_after_count; }
	if (task_src_off_x.size() == 0 || task_src_off_x[pp].size() == 0) continue;
	for (int tmp_size = last_size - 1; tmp_size >= pnet_size + 1 && count < 2000; tmp_size -= 2) {}
}

void Init(int thread_num) {
	thread_num = __max(1, thread_num);
	if (thread_num > 128) { thread_num = 128; }
	pnet.resize(thread_num);
}

void Find(ZQ_CNN_Tensor4D_Interface& input) {
	if (input.GetW() != width || input.GetH() != height) return false;
	const ZQ_CNN_Tensor4D* score = rnet[0].GetBlobByName("prob1");
	if (score == 0) { continue; }
	int h = score->GetH();
}
"""


def _drop(s, pat, flags=0):
    return re.sub(pat, '', s, flags=flags)


SELFCHECK = [
    # (说明, 文本, **期望触发的规则集合**, 变体类型)
    #
    # 集合比对而不是「有没有报错」：只缺 A23 的样本因为顺带也缺 A24/A25，
    # 在只比红/不红的门禁里会被判成符合预期 —— 门禁自己糊弄自己。
    # 阴性对照那几条同样重要：它们声明的是「**不该**报」，是防过度敏感的护栏。
    # 变体类型显式写在每条上，不靠「文本里有没有 lnet/thread_num」去猜 ——
    # 靠猜的话 A30（只对 AspectRatio 判）永远不会被执行到。

    ('全合格骨架', FULL, [], 'base'),

    ('A23 漏夹取', FULL.replace('this->pnet_size = __max(1, pnet_size);',
                                'this->pnet_size = pnet_size;'), ['A23'], 'base'),
    ('A24 失效条件没比旧值',
     _drop(FULL, r'^\t\t\|\| old_pnet_size.*\n\t\t\|\| old_special_big.*\n', re.M),
     ['A24'], 'base'),
    ('A25 没有不变量兜底',
     _drop(FULL, r'\t\t\{\n\t\t\tint kept = 0;.*?\n\t\t\}\n', re.S), ['A25'], 'base'),
    ('A29 scale_factor 改形参',
     FULL.replace('const float new_factor = (float)__max(0.5f, __min(0.97f, scale_factor));',
                  'scale_factor = __max(0.5, __min(0.97, scale_factor));'), ['A29'], 'base'),
    ('A29 只算 new_factor 不落成员',
     _drop(FULL, r'^\tthis->factor = new_factor;\n', re.M), ['A29'], 'base'),
    ('A31 只判外层 task_src_off_x',
     FULL.replace('if (task_src_off_x.size() == 0 || task_src_off_x[pp].size() == 0) continue;',
                  'if (task_src_off_x.size() == 0) continue;'), ['A31'], 'base'),
    ('A32 block_end 用块总数',
     FULL.replace('(bw == block_W_num - 1)', '(bw == block_num - 1)'), ['A32'], 'base'),
    ('A33 pragma 缺 reduction',
     _drop(FULL, r' reduction\(\+:before_count, after_count\)'), ['A33'], 'base'),

    ('A26 活的 * 0.5',
     FULL + '\nx = col1 + (col2-col1)*keyPoint_ptr[i*step + num * 2] * 0.5;\n',
     ['A26'], 'iface'),
    ('A27 Forward[thread_id] 后读 lnet[0]',
     FULL + '\nlnet[thread_id].Forward(img);\n'
            'const T* keyPoint = lnet[0].GetBlobByName("landmark_fc2/BiasAdd");\n',
     ['A27'], 'iface'),
    ('A28 串行支路里的 omp_get_thread_num',
     FULL + '\nif (thread_num <= 1)\n{\n'
            '    for (int i = 0; i < n; i++) { int thread_id = omp_get_thread_num(); }\n}\n',
     ['A28'], 'iface'),
    ('A30 任务循环用 i 判族',
     FULL + '\nint scale_id = task_scale_id[i];\nif (i < ori_num) { }\n'
            'else if (i < ori_num + xhalf_num) { int j = i - ori_num; }\n'
            'else { int k = i - ori_num - xhalf_num; }\n',
     ['A30'], 'ar'),

    ('A34 score/location 没判空',
     FULL.replace('if (score == 0) { continue; }', '/*removed*/'), ['A34'], 'base'),
    ('A35 thread_num 无上界',
     FULL.replace('if (thread_num > 128) { thread_num = 128; }', ''), ['A35'], 'base'),
    ('A36 SetPara 没校验 w/h',
     FULL.replace('if (w <= 0 || h <= 0) { w = 1; h = 1; }', ''), ['A36'], 'base'),
    ('A37 special_handle 循环无上界',
     FULL.replace(' && count < 2000', ''), ['A37'], 'base'),
    ('A38 张量版 Find 无尺寸守卫',
     FULL.replace('if (input.GetW() != width || input.GetH() != height) return false;', ''),
     ['A38'], 'base'),

    # ---- 阴性对照：以下都**不该**报 ----
    ('阴性：注掉的 0.5',
     FULL + '\nx = (a-b)*keyPoint_ptr[i*step + num * 2]/**0.5*/;\n', [], 'iface'),
    ('阴性：只在注释里出现 omp_get_thread_num 与 * 0.5',
     FULL + '\n// 这里原来写的是 `omp_get_thread_num()`，已改成 const int thread_id = 0;\n'
            '/* 另一处注释：keyPoint_ptr[...] * 0.5 也是注掉的 */\n', [], 'iface'),
    ('阴性：并行区里的 thread_num',
     FULL + '\n#pragma omp parallel for num_threads(thread_num)\n'
            'for (int i = 0; i < n; i++) { int thread_id = omp_get_thread_num(); }\n',
     [], 'iface'),
    ('阴性：xhalf/yhalf 分派全用 scale_id',
     FULL + '\nif (scale_id < ori_num) { }\n'
            'else if (scale_id < ori_num + xhalf_num) { int j = scale_id - ori_num; }\n'
            'else { int k = scale_id - ori_num - xhalf_num; }\n', [], 'ar'),
]


def selfcheck():
    bad_cnt = 0
    for name, text, expect, kind in SELFCHECK:
        label = {'base': 'ZQC_fake_MTCNN.h',
                 'iface': 'ZQC_fake_MTCNN_Interface.h',
                 'ar': 'ZQC_fake_MTCNN_AspectRatio.h'}[kind]
        _, bad = scan_text(text, label)
        got = set(c for c, _ in bad)
        if got != set(expect):
            bad_cnt += 1
            print('  [self-MISMATCH] %s' % name)
            print('      expect %s, got %s' % (sorted(expect) or '[]', sorted(got) or '[]'))
            for c, msg in bad:
                print('        %s: %s' % (c, msg))
        else:
            print('  [self-OK]      %-46s %s'
                  % (name, ','.join(sorted(got)) if got else 'clean'))
    if bad_cnt:
        print('selfcheck FAILED: %d / %d mismatch' % (bad_cnt, len(SELFCHECK)))
        return 1
    print('selfcheck OK: %d cases, all as expected' % len(SELFCHECK))
    return 0


def main(argv):
    if '--selfcheck' in argv:
        return selfcheck()

    files = [a for a in argv[1:] if not a.startswith('-')] or DEFAULT_FILES
    total_ok = 0
    problems = []
    for p in files:
        if not os.path.exists(p):
            print('SKIP %s (not found)' % p)
            continue
        name = os.path.basename(p)
        ok, bad = scan_text(_read(p), name)
        total_ok += ok
        if bad:
            print('FAIL %s' % name)
            for code, msg in bad:
                print('       - %s: %s' % (code, msg))
            problems.append(name)
        else:
            print('OK   %s' % name)
    print('合计合格判定 %d 项' % total_ok)
    if problems:
        print('**%d 个文件不合格**' % len(problems))
        return 1
    print('MTCNN SetPara / 多线程索引一致性: OK')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
