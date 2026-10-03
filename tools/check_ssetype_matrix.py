#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SSETYPE 四档的行为门禁（附录 GM）。

背景
----
`ZQ_CNN_USE_SSETYPE` 有四档（NONE / SSE / AVX / AVX2），而全仓库只构建过其中
一档：Windows 默认 AVX2、Linux 默认 AVX（见 `ZQCNN/ZQ_CNN_CompileConfig.h`）。
另外两档**从来没有任何东西编过**，所以
`audit_k3_20261001.md:474` 的 H4「`ZQ_CNN_SSETYPE_NONE` 在 x86 上编不过」
才会看起来像真的 —— 它是**读代码读出来的**，不是测出来的。

2026-10-03 实测：四档**全部**编译通过、链接通过、后向误差都在 1e-8 量级。
H4 是个错的记录，本门禁把它变成**常驻判据**。

判据
----
对 NONE / SSE / AVX / AVX2 四档各做一遍：
  1. 用**真实旗标**编 `ZQ_GEMM/math/*.c`（就 3 个文件，很快）；
  2. 链接并运行 `tools/zq_ssetype_probe.cpp`；
  3. 断言 6 组形状的**后向误差**都在容差内、且 C 里没有 NaN 残留。

为什么必须用真实旗标 `-mavx2 -mfma`
-----------------------------------
根 `CMakeLists.txt:113` 对**所有** gcc x86 构建都加 `-mavx2 -mfma`，
与 SSETYPE 无关 —— SSETYPE 只决定**哪些代码路径被编进来**，
不管制编译器能发什么指令。第一版按档位配 `-mavx` / `-msse4.2`，
于是 AVX 档凭空挂了一个 TU（`zq_avx_mathfun.c` 的 `always_inline` 要求 AVX2），
看起来像仓库的缺陷 —— **那是测试写错**。

已知局限（不写会被当成"全平台都验过了"）
----------------------------------------
只在 Linux/gcc 上验。MSVC 侧那两档没测：MSVC 的默认档是 AVX2，
要把 `ZQ_CNN_USE_SSETYPE` 改成 0/1 才能走到，而那需要改头文件再重编，
不在门禁的能力范围内。

用法:
    python tools/check_ssetype_matrix.py
    python tools/check_ssetype_matrix.py --selftest
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
WSL_ROOT = '/mnt/d/ZQCNN'
GEMM_SRCS = ['ZQ_GEMM/math/zq_gemm_32f_align_c.c',
             'ZQ_GEMM/math/zq_gemm_32f_align_c_asm.c',
             'ZQ_GEMM/math/zq_gemm_32f_auto.c']

# (值, 名字)。旗标**一律相同** —— 见模块 docstring。
LEVELS = [(0, 'NONE'), (1, 'SSE'), (2, 'AVX'), (3, 'AVX2')]
# `-Werror=implicit-function-declaration` 是本门禁的一半价值所在：
# 2026-10-03 在 SSETYPE=0/1 两档查到
# `zq_gemm_32f_asm_core_m6n8`（一个 **static** 函数）缺前置声明。
# 它只在**默认那两档看不到** —— AVX/AVX2 下 ndir 有调用点，
# 编译器在定义处就见过它了。所以"默认档 sweep 是干净的"完全不能说明问题，
# **必须逐档编**。C99 已经不认隐式声明，C23 更是删掉了这个特性，
# 所以这条是**会过期成硬错误**的：`-Werror` 让它现在就红。
COMMON = ('-O2 -mavx2 -mfma -fPIC -Wall -Wextra '
          '-Werror=implicit-function-declaration '
          '-Wno-unused-parameter -Wno-unused-variable '
          '-Wno-unused-function -Wno-unused-but-set-variable '
          '-Wno-write-strings')
INC = ('-I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
       '-I$R/3rdparty/include/openblas')


def build_and_run_all(wdir):
    """一次 wsl 调用跑完四档，返回 {档名: (编译rc, 链接rc, 运行rc, 输出)}。"""
    lines = ['set +e', 'R=%s' % WSL_ROOT, 'mkdir -p %s' % wdir,
             'cd %s || { echo "@@CDFAIL"; exit 1; }' % wdir]
    for v, name in LEVELS:
        for s in GEMM_SRCS:
            b = s.split('/')[-1][:-2]          # 去掉 .c
            lines.append('gcc -c %s -DZQ_CNN_USE_SSETYPE=%d %s $R/%s -o g%d_%s.o '
                         '2>ce%d_%s.txt'
                         % (COMMON, v, INC, s, v, b, v, b))
        # `$?` 取的是**最后一条** gcc 的退出码，而三份源码里任何一份挂了
        # 都不会让另外两份挂 —— 所以逐个记一个码，最后合并。
        # 三份都是 0 才算这一档编过。
        lines.append('rc=0; for t in ce%d_*.txt; do '
                     '[ -s "$t" ] && grep -q "error:" "$t" && rc=1; done; '
                     'echo "@@C%d=$rc"' % (v, v))
        lines.append('g++ -O2 -mavx2 -mfma %s $R/tools/zq_ssetype_probe.cpp '
                     'g%d_*.o -o p%d 2>le%d.txt' % (INC, v, v, v))
        lines.append('echo "@@L%d=$?"' % v)
        # **跑两种环境**（附录 GR）：
        #   默认         -> 走真汇编内核（本机有 AVX2 时）
        #   ZQ_GEMM_ISA=off -> 强制回落，验 `zq_gemm_32f_asm_isa_usable` 那条分支。
        # 源码注释写着后者"用来验证这条路径"，而它**一次都没被跑过**。
        # 两种环境都必须 0 失败。
        for tag, env in (('auto', ''), ('isaoff', 'ZQ_GEMM_ISA=off ')):
            lines.append('%s./p%d > o%d_%s.txt 2>&1; echo "@@R%d_%s=$?"'
                         % (env, v, v, tag, v, tag))
            lines.append('sed "s/^/@@O%d_%s /" o%d_%s.txt 2>/dev/null; true'
                         % (v, tag, v, tag))
    script = '\n'.join(lines) + '\n'
    p = subprocess.run('wsl -d %s -- bash -s' % WSL_DIST, shell=True,
                       input=script.encode('utf-8'), capture_output=True)
    text = ((p.stdout or b'') + (p.stderr or b'')).decode('utf-8', 'replace')
    res = {}
    for v, name in LEVELS:
        def g(pat):
            m = re.search(pat, text)
            return int(m.group(1)) if m else -1
        # sed 给输出的**每一行**都加了 `@@O<v>_<tag> ` 前缀，所以要按前缀收集
        # 全部行，而不是拿一条正则去"截到下一个 @@ 为止" ——
        # 那样只会拿到第一行，探针明明过了 6 组形状，门禁却只看见 1 组。
        res[name] = (g(r'@@C%d=(\d+)' % v), g(r'@@L%d=(\d+)' % v),
                     g(r'@@R%d_auto=(\d+)' % v),
                     g(r'@@R%d_isaoff=(\d+)' % v),
                     [ln.split(' ', 1)[1] for ln in text.split('\n')
                      if ln.startswith('@@O%d_' % v)])
    return res


def selftest():
    """阳性对照：把某一档的运行退出码伪造成失败，主逻辑必须报出来。

    两种环境（默认 / `ZQ_GEMM_ISA=off`）**都要能被抓** ——
    只测一种的话，"另一种环境的退出码被忽略了"就发现不了。
    """
    name, (c, l, r_auto, r_off, out) = list(_FAKE.items())[0]
    ok = True
    # 只把 `ZQ_GEMM_ISA=off` 那一路伪造成失败
    probs = evaluate({name: (c, l, r_auto, 1, out)}, {name: 0})
    if not probs:
        ok = False
    # 只把默认那一路伪造成失败
    if not evaluate({name: (c, l, 1, r_off, out)}, {name: 0}):
        ok = False
    # 编译/链接失败也要能抓
    if not evaluate({name: (1, l, r_auto, r_off, out)}, {name: 0}):
        ok = False
    if not evaluate({name: (c, 1, r_auto, r_off, out)}, {name: 0}):
        ok = False
    # 全部正确时不能误报
    if evaluate({name: (c, l, r_auto, r_off, out)}, {name: 0}):
        ok = False
    # 结论行不对也要能抓
    if not evaluate({name: (c, l, r_auto, r_off, ['本档有超差或 NaN 残留'])},
                    {name: 0}):
        ok = False
    return ok, ''


def evaluate(res, expect_rc):
    """把实测结果转成问题列表。`expect_rc` 正常是 0；自测时故意塞非 0。"""
    problems = []
    for name, (c, l, r_auto, r_off, out) in res.items():
        if c != 0:
            problems.append('%s：编译失败（%d）' % (name, c))
            continue
        if l != 0:
            problems.append('%s：链接失败（%d）' % (name, l))
            continue
        want = expect_rc.get(name, 0)
        # **两种环境都要 0**：默认（真汇编）与 ZQ_GEMM_ISA=off（强制回落）。
        # 漏掉后者的话，`zq_gemm_32f_asm_isa_usable` 那条分支就又没人看了。
        for tag, r in (('默认', r_auto), ('ZQ_GEMM_ISA=off', r_off)):
            if r != want:
                problems.append('%s：%s 环境运行退出码 %d，期望 %d'
                                % (name, tag, r, want))
        if want == 0 and out:
            tails = [ln for ln in out if ln.startswith('本档')]
            if not tails or not all(t.startswith('本档全部在容差内')
                                    for t in tails):
                problems.append('%s：结论行不是「本档全部在容差内」：%r'
                                % (name, tails[-1] if tails else '(没有)'))
    return problems


_FAKE = {}


def main():
    selftest_mode = '--selftest' in sys.argv
    if selftest_mode:
        _FAKE['NONE'] = (0, 0, 0, 0,
                         ['  M=8 N=8 K=12 后向误差=1e-08 NaN残留=否',
                          '本档全部在容差内'])
        if not selftest()[0]:
            print('阳性对照没通过 —— 本门禁可能对任何输入都报「一切正常」')
            return 2
        print('阳性对照：伪造的失败（两种环境/编译/链接/结论行）都被识别 = True')
        _FAKE.clear()

    stamp = int(time.time())
    wdir = '/tmp/zqsset_%d_%d' % (os.getpid(), stamp)
    try:
        res = build_and_run_all(wdir)
    finally:
        subprocess.run('wsl -d %s -- bash -c "rm -rf %s"' % (WSL_DIST, wdir),
                       capture_output=True)

    problems = evaluate(res, {name: 0 for _v, name in LEVELS})
    print('查了 %d 档 SSETYPE（旗标一律 -mavx2 -mfma，与真实构建一致）'
          % len(LEVELS))
    for name, (_c, _l, _ra, _ro, out) in res.items():
        for ln in out:
            print('  [%s] %s' % (name, ln))
    for p in problems:
        print('  ' + p)
    if problems:
        print('SSETYPE 矩阵有问题 —— 别拿"默认值那一档能编"当结论。')
        return 1
    print('四档全部编译、链接、行为都正常。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
