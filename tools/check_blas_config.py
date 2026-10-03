#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""配置宏取值门禁（附录 GL.5）：`-DBLAS_TYPE=...` 到底有没有真的生效。

为什么需要它
------------
2026-10-03 发现 `build-with-cmake.md:50` 教的 `-DBLAS_TYPE=openblas`
**在任何平台都是空操作**，而且是**两层各自独立**地废掉的：

1. `ZQCNN/ZQ_CNN_CompileConfig.h` 无条件 `#define ZQ_CNN_USE_BLAS_GEMM 0`，
   把 CMake 传来的 `-DZQ_CNN_USE_BLAS_GEMM=1` 静默按回去
   （gcc 报 redefined 警告、MSVC 报 C4005，两边都以头文件为准）；
2. `ZQCNN/CMakeLists.txt` 的 `if(BLAS_TYPE MATCHES "openblas")`
   去链的是 **mklml**，而 `elseif(UNIX)` 分支**根本不链任何 BLAS**。

两层**各自都能编译通过** —— 缺陷不在"编不过"，在"编出来的不是你想要的"。
所以判据不能是能不能编，要**是取到的值**。

判据
----
在 8 组 `-D` 下编译 `tools/zq_blas_config_probe.cpp` 并运行，逐个比对
`ZQ_CNN_USE_BLAS_GEMM / MKL_GEMM / ZQ_GEMM / SSETYPE / FMADD128 / FMADD256`
六个宏的取值。任何一个未定义（探针里的 `#error`）或取值不符预期就退出 1。

另外做一条**结构检查**：Windows 分支（`_WIN32`）里的四个后端宏也必须是
`#ifndef` 包着的。取值检查跑在 gcc/Linux 上，覆盖不到 Windows 分支的
**写法**，而 GL.1 的根因正是"一边改了、一边没改"。

**这个门禁的已知局限**（写下来是因为不写会被当成"Windows 也验过了"）：
宏**取值**只在 Linux/gcc 上实测；Windows 分支只查结构，不查取值。
要在 MSVC 上验取值得把探针接进 `probe_zqlib_headers_msvc.py` 那套 vcvars。

用法:
    python tools/check_blas_config.py
    python tools/check_blas_config.py --selftest
"""
import io
import os
import re
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WSL_DIST = 'Ubuntu-20.04'
PROBE = os.path.join(HERE, 'zq_blas_config_probe.cpp')
CONFIG_H = os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_CompileConfig.h')

# (名字, -D 列表, 期望的六个值)
# SSETYPE: 0=NONE 1=SSE 2=AVX 3=AVX2（ZQ_CNN_SSETYPE_* 的定义值）
# FMADD 由 SSETYPE>=AVX2 推导，所以 SSETYPE=3 时它们才是 1。
#
# **默认值那一行是 2026-10-03（附录 GN）改过的**：Linux 侧从 AVX 改成 AVX2，
# 与 Windows 侧对齐 —— 因为两个平台本来就都无条件发 AVX2 指令
# （CMakeLists.txt:113 / :118 都不看 SSETYPE），AVX 挡不住任何老机器，
# 却让同一个模型在两个平台上算出不同的浮点数（实测 6/6 形状位模式不同）。
# 改动当天这个门禁**如实把它拦了下来**（6 组期望全红），
# 然后才改的期望表 —— 一条"拦住了默认行为变更"的门禁，
# 比一条"永远绿"的门禁有用得多。
CASES = [
    ('默认（x86 Linux，2026-10-03 起是 AVX2）',
     [],
     'BLAS=0 MKL=0 ZQ_GEMM=1 SSETYPE=3 FMADD128=1 FMADD256=1'),
    ('CMake 的 BLAS_TYPE=openblas',
     ['-DZQ_CNN_USE_BLAS_GEMM=1'],
     'BLAS=1 MKL=0 ZQ_GEMM=0 SSETYPE=3 FMADD128=1 FMADD256=1'),
    ('CMake 的 BLAS_TYPE=openblas_zq_gemm（x86 上没人读这个宏）',
     ['-DZQ_CNN_USE_BOTH_BLAS_ZQ_GEMM=1'],
     'BLAS=0 MKL=0 ZQ_GEMM=1 SSETYPE=3 FMADD128=1 FMADD256=1'),
    ('显式 -DZQ_CNN_USE_MKL_GEMM=1',
     ['-DZQ_CNN_USE_MKL_GEMM=1'],
     'BLAS=0 MKL=1 ZQ_GEMM=0 SSETYPE=3 FMADD128=1 FMADD256=1'),
    ('CMake 的 BLAS_TYPE=zq_gemm',
     ['-DZQ_CNN_USE_ZQ_GEMM=1'],
     'BLAS=0 MKL=0 ZQ_GEMM=1 SSETYPE=3 FMADD128=1 FMADD256=1'),
    ('-DZQ_CNN_USE_SSETYPE=1（SSE，FMADD 应当跟着关）',
     ['-DZQ_CNN_USE_SSETYPE=1'],
     'BLAS=0 MKL=0 ZQ_GEMM=1 SSETYPE=1 FMADD128=0 FMADD256=0'),
    ('CMake 的 SIMD_ARCH_TYPE=arm（三个后端开关曾经全部未定义）',
     ['-DZQ_CNN_USE_ARM_NEON'],
     'BLAS=0 MKL=0 ZQ_GEMM=0 SSETYPE=0 FMADD128=0 FMADD256=0'),
    ('ARM + openblas_zq_gemm（两个开关都该是 1）',
     ['-DZQ_CNN_USE_ARM_NEON', '-DZQ_CNN_USE_BOTH_BLAS_ZQ_GEMM=1'],
     'BLAS=1 MKL=0 ZQ_GEMM=1 SSETYPE=0 FMADD128=0 FMADD256=0'),
]

# 必须被 `#ifndef` 兜底的宏（头文件里任何一个少了兜底，本门禁就报）。
GUARDED = ['ZQ_CNN_USE_SSETYPE', 'ZQ_CNN_USE_BLAS_GEMM', 'ZQ_CNN_USE_MKL_GEMM',
           'ZQ_CNN_USE_ZQ_GEMM', 'ZQ_CNN_USE_FMADD128', 'ZQ_CNN_USE_FMADD256']

# Windows 分支的**写法**检查只管这三个 + SSETYPE，**不管 FMADD**。
# FMADD 的取值是**由 SSETYPE 推导**出来的
# （`#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX2`），
# 它本来就该在条件块里，套一层 #ifndef 反而是错的。
# 第一版把六个宏一视同仁地查，立即报了两条假阳性 ——
# 判据"放宽到没有"和"收紧到过头"是同一种错：没问"这条规则为什么存在"。
WIN_GUARDED = ['ZQ_CNN_USE_SSETYPE', 'ZQ_CNN_USE_BLAS_GEMM',
               'ZQ_CNN_USE_MKL_GEMM', 'ZQ_CNN_USE_ZQ_GEMM']


def build_and_run(wdir):
    """一次 wsl 调用把 8 组全编全跑，返回 {case_name: (编译rc, 运行rc, 输出, 错误)}。

    标记里的下标是**用 Python 拼进去**的，不靠 bash 变量 ——
    第一版写的是 `echo "@@RC2_$d=$?"`：Python 的 `%` 把 `p%d` 吃掉之后
    `$d` 留在字符串里，被 bash 展开成**空**（$d 未定义），
    于是 8 组的运行退出码全都收不到，而 Python 那边的正则
    `'@@RC2_(\\d+)=(\\d+)' % idx` 又因为串里没有 `%d` 抛
    "not all arguments converted"。**两个错叠在一起，症状是 TypeError。**
    """
    lines = ['set +e', 'R=/mnt/d/ZQCNN',
             'mkdir -p %s' % wdir, 'cd %s || { echo "@@CDFAIL"; exit 1; }' % wdir]
    for idx, (_name, defs, _exp) in enumerate(CASES):
        lines.append('g++ -O0 -I$R/ZQCNN -I$R/3rdparty/include %s '
                     '$R/tools/zq_blas_config_probe.cpp -o p%d 2>e%d.txt'
                     % (' '.join(defs), idx, idx))
        lines.append('echo "@@RC%d=$?"' % idx)
        # 输出**重定向到文件**再由 sed 加前缀回传：直接在管道里 echo
        # 的话，程序自己的 stdout 会和标记混在一起，解析时要靠"最后一行"。
        lines.append('./p%d > o%d.txt 2>/dev/null; echo "@@RC2_%d=$?"'
                     % (idx, idx, idx))
        lines.append('sed "s/^/@@OUT%d /" o%d.txt 2>/dev/null; true' % (idx, idx))
    script = '\n'.join(lines) + '\n'
    p = subprocess.run('wsl -d %s -- bash -s' % WSL_DIST, shell=True,
                       input=script.encode('utf-8'), capture_output=True)
    text = ((p.stdout or b'') + (p.stderr or b'')).decode('utf-8', 'replace')

    res = {}
    for idx, (name, _defs, _exp) in enumerate(CASES):
        comp = re.search(r'@@RC%d=(\d+)' % idx, text)
        run = re.search(r'@@RC2_%d=(\d+)' % idx, text)
        out = re.search(r'@@OUT%d (.*)' % idx, text)
        err = ''
        try:
            # 错误输出**在 WSL 那一侧**读，不能用 Python 的 open ——
            # wdir 是 /tmp/... 这种 WSL 路径，Windows Python 看不见它。
            raw = subprocess.run(
                'wsl -d %s -- cat %s/e%d.txt' % (WSL_DIST, wdir, idx),
                capture_output=True)
            err = (raw.stdout or b'').decode('utf-8', 'replace')
        except (IOError, OSError):
            pass
        res[name] = (int(comp.group(1)) if comp else -1,
                     int(run.group(1)) if run else -1,
                     (out.group(1).strip() if out else ''),
                     err)
    return res


def check_windows_branch():
    """Windows 分支的四个后端宏必须也是 #ifndef 包着的。

    取值检查跑在 gcc 上，覆盖不到这一段**写法**；
    而 GL.1 的根因正是"Linux 那份改了、Windows 那份没改"。
    """
    src = io.open(CONFIG_H, encoding='utf-8').read()
    m = re.search(r'#if defined\(_WIN32\)(.*?)\n#else', src, re.S)
    if not m:
        return ['ZQ_CNN_CompileConfig.h 里找不到 `#if defined(_WIN32)` 分支']
    win = m.group(1)
    bad = []
    for macro in WIN_GUARDED:
        # 找 `#define <macro>`，看它前面紧邻的是不是 #ifndef <macro>
        for dm in re.finditer(r'^#define\s+%s\b' % macro, win, re.M):
            head = win[:dm.start()].rstrip('\n').split('\n')[-1].strip()
            if head != '#ifndef %s' % macro:
                bad.append('%s 的 #define 没有被 #ifndef 包着（前面是 %r）'
                           % (macro, head or '文件开头'))
    return bad


def compare(name, crc, rrc, out, expect):
    """比对一组的结果，返回问题描述或 None。

    抽成函数是为了让 --selftest 能**直接复用**它：阳性对照改的是
    "喂进来的 expect"，而不是再跑一遍编译 —— 这样测的正是
    "比对逻辑真的会拒绝"，而不是"我又跑了一遍"。
    """
    if crc != 0:
        return '编译失败（退出码 %d）' % crc
    if rrc != 0:
        return '编过了但跑不起来（退出码 %d）' % rrc
    if out != expect:
        return '取值不符：期望 %s，实际 %s' % (expect, out or '(空)')
    return None


def selftest():
    """阳性对照：把某一组的**期望值**改错，门禁必须报出来。

    改期望值而不是改源码 —— 这样不会碰到仓库里的文件，
    而且验证的正是"比对逻辑真的会拒绝"。
    """
    name, _defs, good = CASES[1]        # BLAS=1 那一组
    if 'BLAS=1' not in good:
        return False, '阳性对照构造失败：期望串里没有 BLAS=1'
    bad_expect = good.replace('BLAS=1', 'BLAS=0')
    real_out = good                     # 假装实测就是 good
    if compare(name, 0, 0, real_out, good) is not None:
        return False, '正确期望被误报了 —— 比对逻辑本身有问题'
    if compare(name, 0, 0, real_out, bad_expect) is None:
        return False, '错误期望竟然通过了 —— 这个门禁什么都拦不住'
    if compare(name, 1, 0, '', good) is None:
        return False, '编译失败没被抓到'
    if compare(name, 0, 139, '', good) is None:
        return False, '运行失败没被抓到'
    return True, ''


def main():
    if '--selftest' in sys.argv:
        ok, why = selftest()
        print('阳性对照：故意写错的期望值被识别为不符 =', ok)
        if not ok:
            print('  ', why)
            return 2

    # 临时目录必须在 **WSL 那一侧**建，不能用 Python 的 tempfile：
    # 第一版传的是 Windows 路径（`C:\Users\...\Temp\zqblascfg_xxx`），
    # 脚本里的 `cd <那个路径>` 在 bash 里当然失败，而 `set +e` 让它
    # 继续往下跑 —— 于是 8 个二进制和 16 个 .txt **全落在仓库根**。
    # 症状是"git status 里多出一堆 e0.txt/o0.txt/p0"，
    # 离真正的原因（路径分隔）隔了两层。
    # 目录名带 pid + 时间戳，与 AGENTS.md「探针的临时工作目录必须每轮唯一」
    # 同一条规矩。
    stamp = int(time.time())
    wdir = '/tmp/zqblascfg_%d_%d' % (os.getpid(), stamp)
    script_cleanup = 'rm -rf %s' % wdir
    try:
        res = build_and_run(wdir)
    finally:
        subprocess.run('wsl -d %s -- bash -c "%s"' % (WSL_DIST, script_cleanup),
                       capture_output=True)

    problems = []
    for name, _defs, expect in CASES:
        crc, rrc, out, err = res[name]
        why = compare(name, crc, rrc, out, expect)
        if why is None:
            continue
        if crc != 0 and err.strip():
            why += '\n      ' + err.strip().split('\n')[0]
        problems.append('%s：%s' % (name, why))

    win_bad = check_windows_branch()
    for b in win_bad:
        problems.append('Windows 分支：' + b)

    print('查了 %d 组配置 + Windows 分支写法' % len(CASES))
    for p in problems:
        print('  ' + p)
    if problems:
        print('配置宏与预期不符 —— `-DBLAS_TYPE=...` 可能又变回空操作了。')
        return 1
    print('配置宏取值全部符合预期。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
