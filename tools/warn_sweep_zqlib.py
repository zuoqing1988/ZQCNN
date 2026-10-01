#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 gcc -Wall -Wextra 扫 3rdparty/include/ZQlib 下每个头，把警告按 **类别** 归桶。

为什么要有这个工具
------------------
前面三十轮（附录 W~AR）找缺陷靠的是「**编不过**」这一根轴：probe_zqlib_headers.py
问的是「这个头能不能独立编译」。但 ZQlib 里 118 个头都能编过 —— 也就是说，能编过
的那些头里，编译器**早就看见了问题却一声不吭**。gcc 默认不警告，`-Wall -Wextra`
才会说。

2026-10-02 第一次开这条轴，143 个头共 4892 行警告。按 -W 分类后绝大多数是噪声
（-Wsign-compare 208、-Wcomment 182、-Wwrite-strings 172、-Wunused-* 250），
但**高信号的那几类里藏着 4 条真缺陷**（见附录 AT）：死掉的空指针守卫、构造后丢掉的
异常、跨平台不一致的 printf 格式符、类型宽度错配的 %d。

所以本工具的设计是**按「值得人看」的类别分桶**，而不是「把全部警告倒出来」——
4892 行倒出来等于没做。默认只显示 HIGH_SIGNAL 桶。

分桶
----
HIGH  真缺陷高发，基线里**必须为空**，出现即失败
      -Wparentheses  -Waddress  -Wnarrowing  -Wreorder  -Wformat=
MED   值得扫一眼但不阻塞：-Wswitch（枚举漏 case）、-Wmisleading-indentation
      （多数是排版，但偶尔藏着真错）、-Wignored-qualifiers、-Wparentheses 之外的
      括号类
LOW   纯噪声，不显示：-Wsign-compare -Wcomment -Wwrite-strings -Wunused-*
      -Wdeprecated-declarations -Wunknown-pragmas -Wunused-parameter

用法
----
    python tools/warn_sweep_zqlib.py                      # HIGH + MED
    python tools/warn_sweep_zqlib.py --all                # 连 LOW 一起
    python tools/warn_sweep_zqlib.py --bucket HIGH        # 只看一桶
    python tools/warn_sweep_zqlib.py ZQ_CDT               # 只扫名字里含这个串的头
    python tools/warn_sweep_zqlib.py --save-baseline  tools/zqlib_warn_baseline.txt
    python tools/warn_sweep_zqlib.py --check-baseline tools/zqlib_warn_baseline.txt

**基线里只记 HIGH 桶**。MED/LOW 记进去只会让人懒得看 —— 一个门禁如果天天报 3000 条，
就等于没有门禁。HIGH 是「一条都不许新增」。
"""

from __future__ import print_function

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INC = os.path.join(ROOT, '3rdparty', 'include', 'ZQlib')
WSL_DIST = 'Ubuntu-20.04'

# 与 probe_zqlib_headers.py **共用同一份**垫片，读的是 zqlib_msvc_shim.h 的文本。
# 两个工具都是把垫片**内联**进自己生成的翻译单元（不是 #include 它）——
# 这一点必须两边一致：2026-10-02 先是各写一份（会悄悄漂移），
# 后来改成「一个转发头 + 一个真实定义」，而本工具内联的是转发头的内容，
# 于是那句 #include 变成了真包含、路径在 /tmp 下找不到，
# 26 个头同时「OK -> BROKEN」（见 audit_k3_20261001.md 附录 AV）。
# 结论：垫片**只存文本内容**，不含任何 #include 别的本地文件。
SHIM_SRC = os.path.join(HERE, 'zqlib_msvc_shim.h')

HIGH_FLAGS = {
    '-Wparentheses': '“&&” 混在 “||” 里 / 条件里出现算术 —— 可能是优先级写错',
    '-Waddress': '对**必定非空**的东西取地址再判空 —— 通常是空指针守卫写错了位置',
    '-Wnarrowing': 'braced-init 里隐式收窄（double->float / int->char）',
    '-Wreorder': '构造函数初始化列表顺序与**声明顺序**不一致',
    '-Wformat=': 'printf/scanf 格式串与实参类型不匹配',
}
MED_FLAGS = {
    '-Wswitch': 'switch 漏了 enum 的某些取值',
    '-Wmisleading-indentation': '缩进让人误读控制流（多数是排版，少数是真错）',
    '-Wignored-qualifiers': '限定符被忽略（const/volatile 没起作用）',
    '-Wsequence-point': '同一表达式里多次修改同一对象而无序列点',
    '-Wfloat-equal': '浮点用 == 比较',
    '-Wuninitialized': '可能未初始化就使用',
    '-Wreturn-type': '非 void 函数走到结尾没返回值',
    '-Wtautological-compare': '恒真/恒假的比较（常见于写错的边界条件）',
    '-Warray-bounds': '数组下标越界（编译期已知）',
    '-Wstringop-overflow': '字符串操作越界',
}
# 这几条在第三方头里是**结构性噪声**，出现也不看
IGNORE_FLAGS = {
    '-Wsign-compare', '-Wcomment', '-Wwrite-strings', '-Wunused-variable',
    '-Wunused-but-set-variable', '-Wunused-parameter', '-Wunused-function',
    '-Wdeprecated-declarations', '-Wunknown-pragmas', '-Wreorder-ctor',
    '-Wpedantic', '-Wnon-virtual-dtor', '-Wcast-align', '-Wnoexcept',
}

WARN_RE = re.compile(r'^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+): '
                     r'warning: (?P<msg>.*?)(?: \[-(?P<flag>W[a-z0-9=+-]+)\])?$')


def bucket_of(flag):
    # 正则 `\[-?(W...)\]` 捕获到的是 **不带前导连字符** 的 `Wsign-compare`，
    # 而下面三个字典的键都写成 gcc 文档里的样子 `-Wsign-compare`。早期版本两边
    # 对不上，于是 -Wsign-compare / -Wcomment / -Wunused-* 全被当成"未知"掉进
    # MED 桶，MED 一口气 807 条 —— 一个什么也不筛的桶。补回连字符即可。
    if not flag.startswith('-'):
        flag = '-' + flag
    if flag in IGNORE_FLAGS:
        return 'LOW'
    if flag in HIGH_FLAGS:
        return 'HIGH'
    if flag in MED_FLAGS:
        return 'MED'
    return 'MED'


def run_wsl(script):
    # 必须喂字节: text=True + input=str 在 Windows 上会把 \n 翻成 \r\n,
    # WSL 的 bash 看到 `set +\r` 直接不跑（见附录 X.5 / 探测工具里同一段注释）
    p = subprocess.run('wsl -d %s -- bash -s' % WSL_DIST, shell=True,
                       input=script.encode('utf-8'), capture_output=True)
    return ((p.stdout or b'').decode('utf-8', 'replace')
            + (p.stderr or b'').decode('utf-8'))


def to_wsl_path(winpath):
    """D:\\ZQCNN\\tools\\x.h  ->  /mnt/d/ZQCNN/tools/x.h

    **不能**把 Windows 路径原样丢给 WSL 的 bash：它会安静地把 `D:/...` 当成一个
    相对文件名，printf 照样成功、文件里就是那串字面串，g++ 于是报
    "No such file or directory" 进 .warn —— 而我的正则只挑 `warning:`，
    于是整个扫描**一片绿、实际一条没扫**。这种失败比直接报错危险得多。
    """
    p = winpath.replace('\\', '/')
    if len(p) > 1 and p[1] == ':':
        return '/mnt/' + p[0].lower() + p[2:]
    return p


def build_script(headers):
    inc_wsl = to_wsl_path(INC)
    # 垫片是**内联**进每个 .cpp 的（不是 -I 指向它）。原因见上面 SHIM_SRC 的注释。
    try:
        with open(SHIM_SRC, encoding='utf-8') as f:
            shim = f.read()
    except IOError as e:
        raise SystemExit('读不到 shim %s: %s' % (SHIM_SRC, e))
    # 垫片里如果有以 # 开头的行，在 heredoc 里没问题；但必须保证它自身
    # **不含** #include "..."（本地头），否则路径在 /tmp 下不存在。
    for ln_no, ln in enumerate(shim.splitlines(), 1):
        s = ln.strip()
        if s.startswith('#include "'):
            raise SystemExit(
                'shim %s 第 %d 行是 #include "%s" —— 两个探测工具都是把 shim 的**文本**\n'
                '内联进翻译单元的，本地 #include 在 /tmp 下必然找不到。\n'
                '垫片只许包含标准头。' % (SHIM_SRC, ln_no, s[10:].rstrip('"')))

    hdr_of = lambda h: h
    lines = ['set +e',
             'cd /tmp && rm -rf zqwarn && mkdir zqwarn && cd zqwarn',
             # -Wno-unused-parameter / -Wno-write-strings: 这两条在整个 ZQlib 上是
             # 结构性噪声（-Wwrite-strings 172 条全来自「const char* 字面量赋给 char*」
             # 这种 C 风格习惯），进不了 HIGH 桶，压掉只是为了让输出短。
             'FLAGS="-Wall -Wextra -Wno-unused-parameter -Wno-write-strings"']
    for h in headers:
        stem = h[:-2]
        # 用 cat + 引号 heredoc 写：垫片里有中文注释和 $ 之类，必须完全不展开
        lines.append("cat > %s.cpp <<'ZQ_SHIM_EOF'\n%s\nZQ_SHIM_EOF" % (stem, shim))
        lines.append('printf \'#include "%s"\\nint main(){return 0;}\\n\' >> %s.cpp'
                     % (hdr_of(h), stem))
        lines.append('g++ -fsyntax-only -std=c++11 $FLAGS -I%s -I. %s.cpp 2> %s.warn'
                     % (inc_wsl, stem, stem))
    # 每个 warn 文件前打一个分隔行，Python 侧靠它还原「这条警告属于哪个头」。
    # （不要用 sed 在 bash 里改路径再 grep —— 那样每加一个字段就要多一层转义，
    #   实测很容易把 grep -E 的 \\ 吃掉。分隔行是这里最省事又最不容易错的办法。）
    lines.append('for f in *.warn; do echo "@@@FILE@@@ ${f%.warn}.h"; cat "$f"; done')
    lines.append('echo "@@@FILE@@@ __END__"')
    return '\n'.join(lines)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

    argv = list(sys.argv[1:])
    show_all = '--all' in argv
    argv = [a for a in argv if a != '--all']
    bucket_filter = None
    if '--bucket' in argv:
        i = argv.index('--bucket')
        bucket_filter = argv[i + 1].upper()
        del argv[i:i + 2]
    save_to = None
    if '--save-baseline' in argv:
        i = argv.index('--save-baseline')
        save_to = argv[i + 1]
        del argv[i:i + 2]
    check_against = None
    if '--check-baseline' in argv:
        i = argv.index('--check-baseline')
        check_against = argv[i + 1]
        del argv[i:i + 2]
    argv = [a for a in argv if not a.startswith('--')]

    flt = argv[0] if argv else ''
    headers = sorted(h for h in os.listdir(INC) if h.endswith('.h') and flt in h)
    if not headers:
        print('no header matches %r' % flt)
        return 1

    if not os.path.isfile(SHIM_SRC):
        print('缺少 shim: %s' % SHIM_SRC)
        return 1

    script = build_script(headers)
    out = run_wsl(script)

    # 解析：@@@FILE@@@ 行切换「当前属于哪个头」，其余行按 gcc 的告警格式解析。
    per_header = {}          # header -> {flag: count}
    findings = []            # (header, flag, loc, msg)
    compile_errors = []      # (header, 原文) —— 见下面的「假绿」说明
    total_warn = 0
    cur_hdr = None
    for line in out.splitlines():
        if line.startswith('@@@FILE@@@'):
            cur_hdr = line.split(' ', 1)[1].strip()
            continue
        if not line.strip():
            continue
        if ': error:' in line or line.startswith('error:'):
            # 一个头编不过 => 它的 .warn 里全是 error、没有 warning => 这个头
            # 在本工具眼里是「0 条高信号警告」的**全绿**。那正是假绿：先编过的头
            # 编不过（附录 AG~AJ 里 20 多条就是这样溜过去的），必须在这里喊出来。
            compile_errors.append((cur_hdr or '?', line.strip()))
            continue
        m = WARN_RE.match(line.strip())
        if not m:
            continue
        total_warn += 1
        flag = m.group('flag') or '?'
        hdr = os.path.basename(m.group('file'))
        if cur_hdr and hdr == cur_hdr:
            hdr = cur_hdr
        per_header.setdefault(hdr, {})
        per_header[hdr][flag] = per_header[hdr].get(flag, 0) + 1
        findings.append((hdr, flag,
                         '%s:%s:%s' % (os.path.basename(m.group('file')),
                                       m.group('line'), m.group('col')),
                         m.group('msg')))

    buckets = {'HIGH': [], 'MED': [], 'LOW': []}
    for hdr, flag, loc, msg in findings:
        buckets[bucket_of(flag)].append((hdr, flag, loc, msg))

    print('=' * 74)
    print('ZQlib gcc -Wall -Wextra warning sweep  total_headers=%d total_warnings=%d'
          % (len(headers), total_warn))
    print('HIGH=%d  MED=%d  LOW=%d' % (len(buckets['HIGH']), len(buckets['MED']),
                                       len(buckets['LOW'])))
    if compile_errors:
        # 按**头**去重计数，不是按错误行 —— 一个编不过的头通常一次刷几十行 error。
        # 报 61 看着像"61 个头扫不到"，会让人以为覆盖率比实际差得多，也会让
        # 真正该看的那 9 个被淹没。
        first_err = {}
        for hdr, msg in compile_errors:
            first_err.setdefault(hdr, msg)
        print('\n!! %d 个头**编不过** —— 它们的告警没被扫到，别把下面的 0 当成干净:'
              % len(first_err))
        for hdr in sorted(first_err):
            print('   %-38s %s' % (hdr, first_err[hdr][:80]))

    def show(which):
        items = buckets[which]
        if not items:
            print('\n%s: 0  ✓' % which)
            return
        print('\n%s: %d' % (which, len(items)))
        if which == 'HIGH':
            for k, v in sorted(HIGH_FLAGS.items()):
                print('   %-22s %s' % (k, v))
        seen = set()
        for hdr, flag, loc, msg in items:
            key = (hdr, flag, loc)
            if key in seen:      # 同一头被多次 include 会重复报，逐头扫时不会出现，
                continue         # 但 zqlib 头之间互相 include 会有，留着去重
            seen.add(key)
            print('   %-38s %-24s %-28s %s' % (hdr, flag, loc, msg[:70]))

    if bucket_filter:
        show(bucket_filter if bucket_filter in buckets else 'HIGH')
    else:
        show('HIGH')
        show('MED')
        if show_all:
            show('LOW')
        else:
            print('\nLOW: %d  （--all 才显示）' % len(buckets['LOW']))

    # --- 基线：只记 HIGH ---
    if save_to:
        high = sorted(set((hdr, flag, loc) for hdr, flag, loc, _ in buckets['HIGH']))
        lines = ['# ZQlib gcc -Wall/-Wextra HIGH 桶基线'
                 '（tools/warn_sweep_zqlib.py --save-baseline 生成）',
                 '# 格式: <头名>\\t<-W标志>\\t<文件:行:列>',
                 '#',
                 '# 只记 HIGH 桶。MED/LOW 不进基线 —— 一个天天报 3000 条的门禁等于没有门禁。',
                 '# HIGH 的定义见 tools/warn_sweep_zqlib.py 顶部的 HIGH_FLAGS。',
                 '#',
                 '# 基线应当是**空的**：HIGH 桶里的每一条在 2026-10-02 那轮都已修掉。',
                 '# 将来 nonempty 说明有人引入了新的高信号警告。']
        for hdr, flag, loc in high:
            lines.append('%s\t%s\t%s' % (hdr, flag, loc))
        with open(save_to, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(lines) + '\n')
        print('\nbaseline written to %s (%d HIGH findings)' % (save_to, len(high)))

    if check_against:
        base = set()
        try:
            with open(check_against, encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) >= 3:
                        base.add(tuple(parts[:3]))
        except IOError as e:
            print('\nERROR: 读不到基线 %s: %s' % (check_against, e))
            return 1
        cur = set((hdr, flag, loc) for hdr, flag, loc, _ in buckets['HIGH'])
        new = sorted(cur - base)
        fixed = sorted(base - cur)
        print('\n=== 与基线 %s 比对 ===' % check_against)
        print('HIGH: 基线 %d 条 -> 现在 %d 条' % (len(base), len(cur)))
        if new:
            print('\nNEW HIGH — 新出现的高信号警告:')
            for h, fl, lo in new:
                print('   %-38s %-24s %s' % (h, fl, lo))
        if fixed:
            print('\nFIXED — 比基线少了:')
            for h, fl, lo in fixed:
                print('   %-38s %-24s %s' % (h, fl, lo))
        if not new and not fixed:
            print('无新增、无消失。')
        return 1 if new else 0

    return 0


if __name__ == '__main__':
    sys.exit(main())
