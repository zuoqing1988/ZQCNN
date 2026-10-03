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
    # 下面三行是 2026-10-03 补的（附录 GJ）：这三个 SDK 的头**就在仓库里**
    #   3rdparty/include/libfacedetection/facedetect-dll.h
    #   3rdparty/include/mini-caffe/caffe/caffe.hpp
    #   3rdparty/include/SeetaFaceEngine/FaceIdentification/include/face_identification.h
    # 而 ZQlibFaceID 那三个头是**按裸名** include 的（"facedetect-dll.h" 等），
    # 所以探针必须把对应目录**自己**放进包含路径。
    # 缺了它们的后果不是"报 BROKEN"这么温和 —— 而是那 3 个头被归成
    # **NEEDS_LIB（"环境缺库，不是缺陷"）**，于是**真缺陷被当成环境问题放过了**。
    # 与 ES.1（INCLUDE 列表里有不存在的目录 -> g++ 报错 -> 静默全绿）
    # 是同一族：**包含路径的问题会伪装成分类问题**。
    '/mnt/d/ZQCNN/3rdparty/include/libfacedetection',
    '/mnt/d/ZQCNN/3rdparty/include/mini-caffe',
    '/mnt/d/ZQCNN/3rdparty/include/mini-caffe/caffe',
    '/mnt/d/ZQCNN/3rdparty/include/SeetaFaceEngine/FaceIdentification/include',
]

# 缺这些就是"环境缺东西"，不是本仓库的缺陷
#
# **'nn' 这一条曾经把整个门禁废掉**（附录 EU.6）
# ----------------------------------------------------------
# 它本意是认出 ncnn 这类 SDK 的头路径（".../nn/..."），但 `classify()` 是拿
# 错误消息**整行**去子串匹配的，而错误消息开头就是文件路径：
#
#     /mnt/d/ZQCNN/ZQlibFaceID/ZQ_FaceExtractor.h:55: error: '__min' was ...
#                                        ^^^^^^ 小写后含 "cnn"，含 "nn"
#
# 于是**每一条**错误消息都命中 'nn'，被归成 NEEDS_LIB ——
# MSVC_ONLY 与 BROKEN 这两个桶**从来没有被填过**（基线里 7 个非 OK 全是
# NEEDS_LIB 就是证据）。举例：'a very long sentence' 会匹配 'long'。
#
# 实测（分类器直调）：
#     "/mnt/d/ZQCNN/.../Z.h:55: error: '__min' ..."   -> NEEDS_LIB nn   <- 错
#     "/home/u/Z.h:55: error: '__min' ..."           -> MSVC_ONLY __min <- 对
# 同一个错误，只因为路径里有没有 ZQCNN 就分成两类。
#
# 修法：改成能真正命中 SDK 头路径的形式，并且**排除仓库自己的路径**。
NEEDS_LIB = ('jpeglib.h', 'jerror.h', 'jconfig.h', 'png.h', 'zlib.h',
             'opencv2/', 'opencv2\\', 'cuda_runtime.h', 'tbb/', 'omp.h',
             'windows.h', 'tchar.h', 'afx', 'ncnn', 'nn/', 'nnapi', 'seata',
             # The four below only became visible AFTER the classify() fix (appendix
             # EU.7).  While the short 'nn' entry was still there they were
             # short-circuited into NEEDS_LIB together with the real missing-library
             # cases; once that was fixed they fell through to BROKEN, even though what
             # they actually lack is an **external SDK header** -- no different from any
             # other NEEDS_LIB.  First error of each, checked by hand:
             #   ZQ_FaceDetectorLibFaceDetect.h         -> facedetect-dll.h
             #   ZQ_FaceRecognizerArcFaceMiniCaffe.h    -> caffe/caffe.hpp
             #   ZQ_FaceRecognizerSphereFaceMiniCaffe.h -> caffe/caffe.hpp
             #   ZQ_FaceRecognizerSeetaFace.h           -> face_identification.h
             'facedetect', 'caffe/', 'caffe.hpp', 'face_identification.h')

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


MISSING_RE = re.compile(r'fatal error:\s*([^\s:]+):\s*(?:No such file|file not found)',
                        re.I)


def classify(msg):
    # **只拿"缺失的那个头名"去匹配 NEEDS_LIB**，不要拿整行。
    #
    # 拿整行做子串匹配是这个门禁最大的一个坑（附录 EU.6）：错误消息开头就是
    # 文件路径，而路径里带着仓库自己的名字：
    #     /mnt/d/ZQCNN/ZQlibFaceID/ZQ_FaceExtractor.h:55: error: '__min' ...
    #                                        ^^^^^^ 小写后含 "cnn"，含 "nn"
    # 于是**每一条**错误都命中 NEEDS_LIB 里那个 'nn' / 'nn/'，
    # MSVC_ONLY 与 BROKEN 两个桶**从来没有被填过**。
    # 'nn/' 也不行：路径里的 "cnn/" 同样含 "nn/"。
    #
    # "是不是缺外部库"这个问题，正确的信息源只有 `fatal error: X: No such file`
    # 里的那个 X —— 它才是编译器**真正找不到**的东西。
    #
    # **没有这个形状就直接别看 NEEDS_LIB**：编译走到"用了 __int64"这种错误时，
    # 说明所有头都找到了；此时再拿整行去匹配 NEEDS_LIB，只会把路径里的
    # "cnn/" 当成"缺 ncnn"。回退成整行匹配是这一版最初的写法，六个测试里错了三个。
    m = MISSING_RE.search(msg)
    if m:
        needle = m.group(1).lower()
        # NEEDS_LIB 优先于 MSVC_ONLY：一个头可能两样都有，
        # 但"缺 jpeglib.h"是环境问题，"用了 __int64"才是本仓库的缺陷 ——
        # 所以先看是不是**被外部依赖挡住了**，挡住了就只报 NEEDS_LIB，
        # 否则会把一堆"其实还轮不到评"的错误误判成缺陷。
        for lib in NEEDS_LIB:
            if lib.lower() in needle:
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


def selftest():
    """classify() 的单元测试 —— 附录 EU.6。

    为什么分类器必须有自己的测试：它决定"这个头坏了没有、坏在谁头上"，
    而它自己**不会失败**，只会安静地把所有东西归进同一个桶。
    `'nn'` 那一条就是这样让 MSVC_ONLY 与 BROKEN 两个桶**从来没被填过**，
    而基线里"7 个非 OK 全是 NEEDS_LIB"正是它失效的证据 ——
    **一个从不变化的分类结果，本身就该被怀疑**。

    跑法：python tools/probe_faceid_headers.py --selftest
    """
    D = '/mnt/d/ZQCNN/ZQlibFaceID/'
    cases = [
        # (错误消息, 期望分类, 为什么这条重要)
        (D + "ZQ_FaceExtractor.h:55:11: error: '__min' was not declared in this scope",
         'MSVC_ONLY', '路径含 ZQCNN/cnn/ —— 正是把 NEEDS_LIB 误触发的那个串'),
        (D + "ZQ_FaceContainerForVideo.h:87:4: error: '__int64' was not declared in this scope",
         'MSVC_ONLY', '同上，且 __int64 在 MSVC_ONLY 名单里'),
        (D + "ZQ_Foo.h:9:1: error: 'x' was not declared in this scope",
         'BROKEN', '仓内头找不到也是 BROKEN，不是"缺外部库"'),
        (D + "ZQ_Foo.h:1:10: fatal error: ncnn/nn.h: No such file or directory",
         'NEEDS_LIB', '真的缺 ncnn'),
        (D + "ZQ_Foo.h:1:10: fatal error: seata/face.h: No such file or directory",
         'NEEDS_LIB', '真的缺 seeta'),
        (D + "ZQ_Foo.h:1:10: fatal error: jpeglib.h: No such file or directory",
         'NEEDS_LIB', '真的缺 jpeglib'),
        (D + "ZQ_Foo.h:1:10: fatal error: opencv2/core.hpp: No such file or directory",
         'NEEDS_LIB', '真的缺 opencv'),
        (D + "ZQ_Foo.h:1:10: fatal error: windows.h: No such file or directory",
         'NEEDS_LIB', '真的缺 windows.h'),
        (D + "ZQ_Foo.h:1:10: fatal error: ZQ_CNN_BBox.h: No such file or directory",
         'BROKEN', '**仓内**头找不到：缺 -I 路径，是配置问题不是缺库'),
    ]
    bad = 0
    for msg, want, why in cases:
        got = classify(msg)[0]
        if got == want:
            print("  %-10s %s" % (got, why))
        else:
            bad += 1
            print("  %-10s ** WRONG, expected %s **  %s" % (got, want, why))
    print("\nclassify() selftest: %d case(s), %d wrong" % (len(cases), bad))
    return 1 if bad else 0


def main():
    if '--selftest' in sys.argv:
        return selftest()
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
