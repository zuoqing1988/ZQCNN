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
    return ok, bad


SELFCHECK = [
    # (说明, 文本, 期望 bad 条数是否 > 0)
    ('全合格的最小样本', '''
void SetPara(int w, int h) {
    int old_pnet_size = this->pnet_size;
    int old_min_size = min_size;
    bool old_special_big = special_handle_very_big_face;
    this->pnet_size = __max(1, pnet_size);
    this->pnet_stride = __max(1, pnet_stride);
    if (width != w || height != h || factor != scale_factor
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
}
''', []),
    # A23 不合格
    ('A23 漏夹取（ncnn.h 修前的样子）', '''
void SetPara(int w, int h) {
    int old_pnet_size = this->pnet_size;
    this->pnet_size = pnet_size;
    this->pnet_stride = pnet_stride;
    pnet_images.resize(count);
    { int kept = 0; for (int i = 0; i < (int)scales.size(); i++) {
        int changedH = 0; int changedW = 0;
        if (changedH < pnet_size || changedW < pnet_size) continue;
        scales[kept++] = scales[i]; } scales.resize(kept); }
}
''', ['A23', 'A24']),
    # A24 不合格
    ('A24 失效条件没比旧值', '''
void SetPara(int w, int h) {
    this->pnet_size = __max(1, pnet_size);
    this->pnet_stride = __max(1, pnet_stride);
    if (width != w || height != h || factor != scale_factor) {
        pnet_images.resize(count);
        { int kept = 0; for (int i = 0; i < (int)scales.size(); i++) {
            int changedH = 0; int changedW = 0;
            if (changedH < pnet_size || changedW < pnet_size) continue;
            scales[kept++] = scales[i]; } scales.resize(kept); }
    }
}
''', ['A24']),
    # A25 不合格
    ('A25 没有不变量兜底', '''
void SetPara(int w, int h) {
    int old_pnet_size = this->pnet_size;
    int old_min_size = min_size;
    bool old_special_big = special_handle_very_big_face;
    this->pnet_size = __max(1, pnet_size);
    this->pnet_stride = __max(1, pnet_stride);
    if (width != w || height != h || factor != scale_factor
        || old_pnet_size != this->pnet_size || old_min_size != min_size
        || old_special_big != special_handle_very_big_face)
    { pnet_images.resize(count); }
}
''', ['A25']),
    # A26 不合格
    ('A26 活的 * 0.5', '''
int thread_num = 1;
x = col1 + (col2-col1)*keyPoint_ptr[i*step + num * 2] * 0.5;
''', ['A23', 'A24', 'A25', 'A26']),
    # A27 不合格
    ('A27 Forward[thread_id] 后读 lnet[0]', '''
int thread_num = 1;
lnet[thread_id].Forward(img);
const T* keyPoint = lnet[0].GetBlobByName("landmark_fc2/BiasAdd");
''', ['A23', 'A24', 'A25', 'A27']),
    # A28 不合格
    ('A28 串行支路里的 omp_get_thread_num', '''
if (thread_num <= 1)
{
    for (int i = 0; i < n; i++) { int thread_id = omp_get_thread_num(); }
}
''', ['A23', 'A24', 'A25', 'A28']),
    # 阴性对照之二：**注释里**出现这两样都**不该**报。
    # 这是第一版真踩到的坑：修复说明里那句「这里原来写的是 `omp_get_thread_num()`」
    # 被 A28 当成了缺陷。加这条是为了让"剥注释"这件事本身也有回归保护。
    ('阴性对照：只出现在注释里', '''
int thread_num = 1;
// 这里原来写的是 `omp_get_thread_num()`，已改成 const int thread_id = 0;
/* 另一处注释：keyPoint_ptr[...] * 0.5 也是注掉的 */
''', ['A23', 'A24', 'A25']),
    # 阴性对照：注释里的 0.5 与不在 thread_num<=1 里的 omp_get_thread_num 都**不该**报
    ('阴性对照：注掉的 0.5 + 并行区里的 thread_num', '''
x = (a-b)*keyPoint_ptr[i*step + num * 2]/**0.5*/;
#pragma omp parallel for num_threads(thread_num)
for (int i = 0; i < n; i++) { int thread_id = omp_get_thread_num(); }
''', ['A23', 'A24', 'A25']),
]


def selfcheck():
    bad_cnt = 0
    for name, text, expect in SELFCHECK:
        label = 'ZQC_fake_MTCNN_Interface.h' if 'lnet' in text or 'thread_num' in text \
            else 'ZQC_fake_MTCNN.h'
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
