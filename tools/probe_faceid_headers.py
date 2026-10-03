#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZQlibFaceID 头的**可编译性**探针（Linux/gcc）。

为什么需要它
------------
附录 EG 之前，`ZQlibFaceID/` 的 29 个头里：

* **26 个没有任何门禁提到**；
* 而 `ZQlibFaceID` **既不在 Windows 构建里、也不在 Linux 构建里** ——
  全仓没有任何 `.cpp` include 其中大部分头，**这些代码从来没有被编译过**。

`tools/probe_zqlib_headers.py` 已经对 `3rdparty/include/ZQlib/` 做了同样的事
（118/143 可编译），但**没覆盖 `ZQlibFaceID/`**。
本工具是它的姊妹篇：逐个头在 Linux 上 `-fsyntax-only` 编一遍，把结果分成四类

    OK          编译通过
    NEEDS_LIB   缺外部依赖（jpeglib.h / opencv2/ / windows.h …）——
                **不是本仓库的缺陷**，是环境缺东西
    MSVC_ONLY   用了 MSVC 专有写法（`__int64` / `fopen_s` / `__declspec` …）——
                **是缺陷**：意味着这份头在 Linux 上编不过
    BROKEN      其它编译错误 —— **一定是缺陷**

基线
----
和 `probe_zqlib_headers.py` 一样支持 `--save-baseline` / `--check-baseline`：
**任何一个头从 OK 变成非 OK 就退出 1**。

用 ``--check-baseline`` 时的价值要讲清楚：
它锁的是**状态变化**，不是"绝对正确"。
今天 EG 修好的那两个头，基线里应该是 OK；
而那些"本来就 MSVC_ONLY"的头仍然不是 OK —— **工具不会替我把它们修掉**，
但它会保证**我修好一个，就不会再坏一个**，并且**新引入的跨平台缺陷立刻可见**。

用法::

    python tools/probe_faceid_headers.py            # 跑一遍，打印分类表
    python tools/probe_faceid_headers.py --save-baseline tools/faceid_probe_baseline.txt
    python tools/probe_faceid_headers.py --check-baseline tools/faceid_probe_baseline.txt
    python tools/probe_faceid_headers.py Face       # 只看名字含 Face 的

进回归：``run_audit_checks.py`` 的 C 组（与 ``probe_zqlib_headers.py`` 并列）。
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INC = os.path.join(ROOT, 'ZQlibFaceID')
WSL_DIST = os.environ.get('WSL_DIST', 'Ubuntu-20.04')

# WSL 侧的 include 路径。OpenCV 只在 Windows 上装了 .lib，但**头**是全的
# （3rdparty/opencv/build/include），所以"能不能编过"这件事可以在 Linux 上问。
INCS = [
    '/mnt/d/ZQCNN',
    '/mnt/d/ZQCNN/ZQlibFaceID',      # ← 头文件自己所在目录；少了这一行，**每个头都"找不到"**
    '/mnt/d/ZQCNN/ZQCNN',
    '/mnt/d/ZQCNN/ZQ_GEMM',
    '/mnt/d/ZQCNN/3rdparty/include',
    '/mnt/d/ZQCNN/3rdparty/include/ZQlib',
    '/mnt/d/ZQCNN/3rdparty/opencv/build/include',
]

# 缺这些就是"环境缺东西"，不是本仓库的缺陷
NEEDS_LIB = ('jpeglib.h', 'jerror.h', 'jconfig.h', 'png.h', 'zlib.h',
             'opencv2/', 'opencv2\\', 'cuda_runtime.h', 'tbb/', 'omp.h',
             'windows.h', 'tchar.h', 'afx', 'ncnn', 'seeta', 'nn')

# MSVC 专有写法 —— 出现即意味着"这份头在 Linux 上编不过"
MSVC_ONLY = ('__int64', '__uint64', '_fseeki64', '_ftelli64', 'strcpy_s',
             'strncpy_s', 'sprintf_s', 'vsprintf_s', '_snprintf', 'fopen_s',
             '_sopen', '__declspec', '_MSC_VER', '__forceinline', '_stricmp',
             '_stricmp', '__try', ' _aligned_malloc', '_aligned_free',
             '__min', '__max', 'min<int', 'max<int')


def run_wsl(script):
    # 必须以**字节**喂给 wsl：text=True 在 Windows 上会把 \n 翻成 \r\n，
    # WSL 里的 bash 于是看到 `set +\r`，一行都不跑（probe_zqlib_headers.py 的注释）。
    p = subprocess.run('wsl -d %s -- bash -s' % WSL_DIST, shell=True,
                       input=script.encode('utf-8'), capture_output=True)
    return ((p.stdout or b'').decode('utf-8', 'replace')
            + (p.stderr or b'').decode('utf-8', 'replace'))


def classify(msg):
    low = msg.lower()
    # NEEDS_LIB 优先于 MSVC_ONLY：一个头可能两样都有，
    # 但"缺 jpeglib.h"是环境问题，"用了 __int64"才是本仓库的缺陷 ——
    # 所以先看是不是**被外部依赖挡住了**，挡住了就只报 NEEDS_LIB，
    # 否则会把一堆"其实还轮不到评"的错误误判成缺陷。
    for lib in NEEDS_LIB:
        if lib.lower() in low:
            return 'NEEDS_LIB', lib
    for kw in MSVC_ONLY:
        if kw in msg:
            return 'MSVC_ONLY', kw
    return 'BROKEN', ''


def probe_all(headers):
    """一次 WSL 调用编完所有头，返回 {头名: (原始状态, 首行错误)}。

    探针 .cpp **用两条 echo 写**，不用 printf 里的 `\\n` ——
    它要经过 Python -> 文件 -> 再喂给 bash 两层，实测会被吃掉变成真换行，
    把本该单行的 printf 拆成多行（bash 仍能跑，但脆）。
    另外 `tr -d '\\r'` 那一层也被吃掉成了 `tr -d ''`（空参数）——
    错误信息里带 CR 就会破坏后面的 `|` 切分，所以干脆**不 tr**，交给 Python 侧 strip。
    """
    script = ['set +e', 'T=/tmp/faceid_probe', 'rm -rf $T', 'mkdir -p $T', 'cd $T']
    incs = ' '.join('-I' + p for p in INCS)
    for n, h in enumerate(headers):
        script.append("echo '#include \"%s\"' > p%d.cpp" % (h, n))
        script.append("echo 'int probe_%d(){return 0;}' >> p%d.cpp" % (n, n))
        # **取第一条含 `error:` 的行，不是 `head -1`** —— gcc 的第一行是
        # `In file included from X:1:` 这种引导行，真正的 error 在第 2 行。
        # 用 head -1 会把 `face_identification.h: No such file` 这类
        # **缺外部 SDK** 误判成 BROKEN（= 真缺陷），方向正好反了。
        script.append(
            'if g++ -fsyntax-only -mavx2 -mfma -fopenmp %s p%d.cpp 2> e%d.log; '
            'then echo "R|%s|OK|"; else '
            'echo "R|%s|$(grep -m1 "error:" e%d.log || head -3 e%d.log | tr "\\n" " ")<<END>>"; fi'
            % (incs, n, n, h, h, n, n))
    script.append('echo R|__END__|0|')
    out = run_wsl('\n'.join(script))
    res = {}
    for line in out.splitlines():
        if not line.startswith('R|') or '__END__' in line:
            continue
        parts = line.split('|', 3)
        # 成功时是 `R|头|OK|`（4 段），失败时是 `R|头|首行错误`（**3 段** ——
        # 首行错误里通常没有 `|`）。第一版写 `if len(parts) < 4: continue`，
        # 于是**所有编译失败的头都被静默丢掉**，表上显示成"探针没回来说这一条"，
        # 差点被我当成"这些头没问题"。
        if len(parts) < 3:
            continue
        name, status = parts[1], parts[2]
        first = parts[3] if len(parts) > 3 else ''
        first = first.replace('<<END>>', '').strip()
        res[name] = (status, first)
    return res


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    argv = sys.argv[1:]
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
    flt = next((a for a in argv if not a.startswith('--')), '')

    if not os.path.isdir(INC):
        print('找不到 %s' % INC)
        return 1
    headers = sorted(h for h in os.listdir(INC) if h.endswith('.h') and flt in h)
    if not headers:
        print('没有匹配的头（filter=%r）' % flt)
        return 1

    print('ZQlibFaceID 可编译性探针：%d 个头（Linux/gcc -fsyntax-only）' % len(headers))
    res = probe_all(headers)

    table = []
    for h in headers:
        status, detail = res.get(h, ('UNKNOWN', '探针压根没回这一条'))
        if status == 'OK':
            why = ''
        else:
            # 失败时探针把**错误首行**塞在 status 位（3 段格式没有单独的 detail 字段）。
            # 拿它去分类：缺外部依赖 / MSVC 专有写法 / 真·编译错误。
            # UNKNOWN 必须如实报成 UNKNOWN，不能静默降级 —— "没跑出结果"和"编不过"
            # 是两件事，混在一起就会把工具自己的故障记成仓库的缺陷。
            line = status if status not in ('NEEDS_LIB', 'MSVC_ONLY', 'BROKEN') else detail
            if status == 'UNKNOWN':
                cat, why = classify(line)
                status = 'UNKNOWN'
            else:
                cat, why = classify(line)
                status = cat if cat != 'OK' else 'BROKEN'
            why = why or line
        table.append((h, status, why))

    print('-' * 78)
    for h, status, why in table:
        mark = 'OK         ' if status == 'OK' else status.ljust(11)
        print('  %-40s %s %s' % (h, mark, (why or '')[:44]))
    print('-' * 78)
    n_ok = sum(1 for _, s, _ in table if s == 'OK')
    n_lib = sum(1 for _, s, _ in table if s == 'NEEDS_LIB')
    n_msvc = sum(1 for _, s, _ in table if s == 'MSVC_ONLY')
    n_broken = sum(1 for _, s, _ in table if s == 'BROKEN')
    n_unk = sum(1 for _, s, _ in table if s == 'UNKNOWN')
    print('  OK %d / NEEDS_LIB %d / MSVC_ONLY %d / BROKEN %d / UNKNOWN %d  （共 %d）'
          % (n_ok, n_lib, n_msvc, n_broken, n_unk, len(table)))
    print('  NEEDS_LIB = 环境缺依赖，不是缺陷；MSVC_ONLY 与 BROKEN 是**真缺陷**'
          '（这份头在 Linux 上编不过）')

    if save_to:
        with open(save_to, 'w', encoding='utf-8', newline='\n') as f:
            f.write('# ZQlibFaceID 头可编译性基线（tools/probe_faceid_headers.py 生成）\n')
            f.write('# 格式: <头名>\t<OK|NEEDS_LIB|MSVC_ONLY|BROKEN|UNKNOWN>\n')
            f.write('# 任何一个头从 OK 变成非 OK，--check-baseline 就退出 1。\n')
            for h, status, _ in table:
                f.write('%s\t%s\n' % (h, status))
        print('  基线已写入 %s' % save_to)

    if check_against:
        if not os.path.isfile(check_against):
            print('  基线文件不存在：%s' % check_against)
            return 1
        base = {}
        for line in open(check_against, encoding='utf-8'):
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                base[parts[0]] = parts[1]
        now = {h: s for h, s, _ in table}
        regress, added, removed = [], [], []
        for h in sorted(set(base) | set(now)):
            b, n = base.get(h), now.get(h)
            if b == 'OK' and n != 'OK':
                regress.append((h, b, n))
            elif b is None and n is not None:
                added.append((h, n))
            elif n is None and b is not None:
                removed.append((h, b))
        for h, b, n in regress:
            print('  **回归** %-40s %s -> %s' % (h, b, n))
        for h, b in removed:
            print('  **头不见了** %-36s （基线是 %s）' % (h, b))
        for h, n in added:
            print('  新增 %-40s %s' % (h, n))
        if regress or removed:
            print('\n%d 个头从 OK 变坏 / 消失 —— 这就是回归。' % (len(regress) + len(removed)))
            return 1
        print('  与基线一致：%d 个头，无回归' % len(now))
    return 0


if __name__ == '__main__':
    sys.exit(main())
