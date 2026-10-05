#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""探测 3rdparty/include/ZQlib 下每个头能否**独立编译**（Linux / gcc）。

为什么需要这个工具
------------------
审计报告里反复出现同一类结论：「这是第三方头，改动面大且**无法编译验证**，不修」。
ZQ_MergeSort 那条就是这样被记成「不修」的 —— 直到有人真去数了一下它的 include，
发现只有 4 个标准头，补上 MSVC 的 __int64/__min/__max 就能单独编，
于是「无法验证」这个前提根本不成立（见 audit_k3_20261001.md 附录 W）。

这个工具把「能不能验证」从印象变成一张表：逐个头生成一个只 `#include` 它的
最小翻译单元，用 gcc 编译，把结果分类。

分类
----
  OK        编译通过 —— 可以单独验证，任何修改都能端到端测
  NEEDS_LIB 报缺外部库（jpeglib.h / OpenCV 之类）—— 装上依赖即可
  MSVC_ONLY 报 MSVC 专有关键字（__int64 / _fseeki64 / strcpy_s ...）
             —— 补几个 typedef/macro 就能编，属于可救
  BROKEN    其它编译错误 —— 真有问题，或者依赖链缺失

用法
----
    python tools/probe_zqlib_headers.py            # 全部头
    python tools/probe_zqlib_headers.py ZQ_Kmeans  # 只探名字里含这个串的
"""

from __future__ import print_function

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INC = os.path.join(ROOT, '3rdparty', 'include', 'ZQlib')
WSL_DIST = 'Ubuntu-20.04'

# 补在 #include 之前的兼容层：MSVC 的类型与内建函数，gcc 下没有。
#
# **只从 zqlib_msvc_shim.h 这一份读**（2026-10-02 教训，见下面那段长注释）。
# 历史上这份垫片散落过三份，其中一份是「转发头」，而本工具是把垫片**文本内联**
# 进探测翻译单元的 —— 于是转发头里那句 #include "zqlib_msvc_shim.h" 变成了
# 一条真的 #include，路径在 /tmp/zqprobe 下不存在，于是 26 个头同时
# 「OK -> BROKEN」。更糟的是它**报得很像真的**（每条都带一条 error 行）。
SHIM_FILE = os.path.join(HERE, 'zqlib_msvc_shim.h')

try:
    with open(SHIM_FILE, encoding='utf-8') as _f:
        SHIM = _f.read()
except IOError as e:
    raise SystemExit('读不到 shim %s: %s' % (SHIM_FILE, e))


def run_wsl(script):
    # 必须以**字节**喂给 wsl: text=True + input=str 在 Windows 上会按文本模式
    # 把 \n 翻译成 \r\n, WSL 里的 bash 于是看到 `set +\r` / `cd dir\r`,
    # 直接 `invalid option` + `syntax error: unexpected end of file`，
    # 一行脚本都没跑（2026-10-02 实测踩过）。
    p = subprocess.run('wsl -d %s -- bash -s' % WSL_DIST, shell=True,
                       input=script.encode('utf-8'), capture_output=True)
    return ((p.stdout or b'').decode('utf-8', 'replace')
            + (p.stderr or b'').decode('utf-8', 'replace'))


def classify(msg):
    low = msg.lower()
    for lib in ('jpeglib.h', 'jerror.h', 'png.h', 'zlib.h', 'opencv2/',
                'cuda_runtime.h', 'tbb/', 'omp.h', 'windows.h', 'tchar.h',
                'afx'):
        if lib in low:
            return 'NEEDS_LIB', lib
    for kw in ('__int64', '__uint64', '_fseeki64', '_ftelli64', 'strcpy_s',
               '_sopen', '__declspec', 'fopen_s', 'sprintf_s', '_snprintf',
               '_MSC_VER', '__forceinline', '_stricmp', '__try', 'min/max'):
        if kw in msg:
            return 'MSVC_ONLY', kw
    return 'BROKEN', ''


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    argv = [a for a in sys.argv[1:]]
    save_to = None
    check_against = None
    if '--save-baseline' in argv:
        i = argv.index('--save-baseline')
        save_to = argv[i + 1]
        del argv[i:i + 2]
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

    # **工作目录必须每轮唯一**（附录 ES.4，与 run_zqlib_checks.py 的 EB.1 同一个毛病）。
    # 原来是固定的 `/tmp/zqprobe` 且开头 `rm -rf`：任何两次并发运行都会互相摧毁 ——
    # 一次正在编译，一次把它的 .cpp / .err 全删了，于是几十个头凭空报 ERR，
    # 而且**错误信息是空的**（`grep error:` 在已被删掉的 .err 上当然找不到），
    # 看起来像"这些头突然全坏了"。
    # 2026-10-03 实测：完整审计回归内部会调本脚本，我同时手工跑了一次，
    # BROKEN 从 12 变成 103，其中 91 个是假的 —— 单独重编每一个都通过。
    #
    # 与 EB.1 一样：**不要在本轮结束时删别人的目录**，所以这里只建不删。
    run_id = '%d_%d' % (os.getpid(), int(time.time()))
    wdir = '/tmp/zqprobe_%s' % run_id
    lines = ['set +e',
             'mkdir -p %s && cd %s && rm -rf ./*' % (wdir, wdir)]
    for h in headers:
        stem = h[:-2]
        lines.append("cat > %s.cpp <<'PROBE_EOF'\n%s\n#include \"%s\"\n"
                     "int main(){return 0;}\nPROBE_EOF" % (stem, SHIM, h))
        # 成功打 OK, 失败把第一条 error: 打出来 (同一行, 便于解析)
        # 注意两个分支**都必须打 h 而不是 stem**: 早先 ERR 分支写的是 stem
        # (不带 .h), 于是「本来 OK、现在编不过」的头会被 --check-baseline
        # 判成「新增」而不是「回退」, 门禁就失效了。2026-10-02 实测踩到。
        lines.append(
            # `tools/gl_stub` 提供一份**只给 -fsyntax-only 用的** GL/glew.h 桩
            # （附录 IZ.1）：`ZQ_GLSLShader.h` 要的只是 OpenGL，不是平台特有的
            # windows.h。加进 include 路径之后它能真的编一遍，于是能验证
            # "这个头自身自足吗"，而不必再接受"本机没装 glew 所以测不了"。
            "if g++ -fsyntax-only -std=c++11 -I/mnt/d/ZQCNN/3rdparty/include/ZQlib "
            "-I/mnt/d/ZQCNN/tools/gl_stub "
            "%s.cpp 2> %s.err; then echo 'R|%s|OK|'; else "
            "echo \"R|%s|ERR|$(grep -m1 error: %s.err | tr -d '\\r')\"; fi"
            % (stem, stem, h, h, stem))
    lines.append('echo R|__END__|OK|')
    out = run_wsl('\n'.join(lines))

    rows = []
    for line in out.splitlines():
        if not line.startswith('R|'):
            continue
        parts = line.split('|', 3)
        if len(parts) != 4:
            continue
        rows.append((parts[1], parts[2], parts[3]))

    buckets = {'OK': [], 'NEEDS_LIB': [], 'MSVC_ONLY': [], 'BROKEN': []}
    for name, status, msg in rows:
        if status == 'OK':
            buckets['OK'].append((name, ''))
        else:
            c, d = classify(msg)
            buckets[c].append((name, d or msg.strip()[:70]))

    print('=' * 74)
    print('ZQlib header standalone-compile probe (gcc, -std=c++11, Linux)  total=%d'
          % len(headers))
    print('=' * 74)
    print('OK (independently verifiable): %d' % len(buckets['OK']))
    for name, _ in buckets['OK']:
        print('   %s' % name)
    for c in ('NEEDS_LIB', 'MSVC_ONLY', 'BROKEN'):
        items = buckets[c]
        print('\n%s: %d' % (c, len(items)))
        for name, d in items:
            print('   %-40s %s' % (name, d))

    # 逐头分类表：name<TAB>CATEGORY<TAB>detail
    table = {}
    for c in ('OK', 'NEEDS_LIB', 'MSVC_ONLY', 'BROKEN'):
        for name, d in buckets[c]:
            table[name] = (c, d)

    if save_to:
        lines = ['# ZQlib 头独立编译分类基线（tools/probe_zqlib_headers.py --save-baseline 生成）',
                 '# 格式: <头名>\\t<CATEGORY>\\t<detail>',
                 '# CATEGORY 取值: OK / NEEDS_LIB / MSVC_ONLY / BROKEN',
                 '#',
                 '# 用途: --check-baseline 会在「OK 变成非 OK」时报失败。',
                 '# 也就是: 将来往 ZQlib 里加了新头、或改了现有头导致某个头编不过了，',
                 '# 这一步能立刻抓到 —— 附录 AG~AJ 里那 20 来条缺陷全是这一类，',
                 '# 而它们之所以长期没被发现，正是因为「编不过」没人看得见。',
                 '#',
                 '# OK 的个数就是「可以被单独验证（因而可以写 ASan 测试）」的个数。']
        for name in sorted(table):
            c, d = table[name]
            lines.append('%s\t%s\t%s' % (name, c, d))
        with open(save_to, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(lines) + '\n')
        print('\nbaseline written to %s (%d headers)' % (save_to, len(table)))

    if check_against:
        base = {}
        try:
            with open(check_against, encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) >= 2:
                        base[parts[0]] = parts[1]
        except IOError as e:
            print('\nERROR: 读不到基线 %s: %s' % (check_against, e))
            return 1
        if not base:
            print('\nERROR: 基线 %s 是空的' % check_against)
            return 1

        regress, improved, added, gone = [], [], [], []
        for name, (c, _d) in table.items():
            if name not in base:
                added.append((name, c))
            elif base[name] == 'OK' and c != 'OK':
                regress.append((name, base[name], c))
            elif base[name] != 'OK' and c == 'OK':
                improved.append(name)
        for name in base:
            if name not in table:
                gone.append(name)

        print('\n=== 与基线 %s 比对 ===' % check_against)
        print('OK: %d -> %d' % (sum(1 for v in base.values() if v == 'OK'),
                                sum(1 for c, _ in table.values() if c == 'OK')))
        if regress:
            print('\nREGRESSION — 这些头本来能编, 现在编不过了:')
            for name, was, now in regress:
                print('   %-40s %s -> %s' % (name, was, now))
        if improved:
            print('\nIMPROVED — 变可验证了 (记得跑 --save-baseline 更新基线):')
            for name in improved:
                print('   %s' % name)
        if added:
            print('\nNEW — 基线里没有的新头 (记得决定它的分类并更新基线):')
            for name, c in added:
                print('   %-40s %s' % (name, c))
        if gone:
            print('\nREMOVED — 基线里有、现在目录里没有了:')
            for name in gone:
                print('   %s' % name)
        if not regress and not added and not gone:
            print('无回退、无新增。')
        return 1 if regress or added or gone else 0

    return 0


if __name__ == '__main__':
    sys.exit(main())
