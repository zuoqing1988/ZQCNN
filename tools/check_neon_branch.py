#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ARM/NEON 分支的解析门禁（附录 GP）。

为什么需要它
------------
仓库里 **86 个文件**带 `#if __ARM_NEON` 分支、**1606 处** NEON 调用点，
而**一行都没被任何编译器看过**：

* WSL 里没有 `arm-linux-gnueabihf-gcc`，也没有 clang（2026-10-04 实测）；
* 仓库根的 `build.sh` 正是构建 armeabi-v7a 的 ——
  仓库自己带着一条**从未被验证过**的构建路径。

而 GM 已经证明这条轴上"没人看"会漏掉真缺陷：
`zq_gemm_32f_asm_core_m6n8` 缺前置声明，只在 SSETYPE=0/1 出现，
默认那两档连 warning 都没有。

做法
----
1. 先扫出所有 NEON 区域里用到的 `v*q_*` 名字，与 `tools/arm_neon.h`
   里定义的比对 —— **桩缺哪个名字，本门禁直接喊出来**，
   桩的完整性由门禁自己保证，不靠人手维护的清单；
2. 再量一条**前提**：这些文件的 NEON 区域里 `__asm__` 的出现次数必须是 0。
   有的话 x86 上编不出内联汇编，方案对那个文件就不成立；
3. 然后用 gcc `-fsyntax-only -DZQ_CNN_USE_ARM_NEON`，并把
   `-I tools/` 放在最前面，让 `#include <arm_neon.h>` 命中桩，
   逐个编译上面那 86 个文件。

**它证明不了什么**（写下来是为了不让它被当成"ARM 路径验过了"）：
桩把向量类型全 typedef 成 `float`，所以**不验 NEON 的类型**、
**不验语义**（所有内在函数返回 0）。它能验的是：NEON 分支能被解析、
其中的普通 C 代码过真正的类型检查、以及 NEON 内在函数名没拼错。

用法:
    python tools/check_neon_branch.py
    python tools/check_neon_branch.py --save-baseline  <file>
    python tools/check_neon_branch.py --check-baseline <file>
    python tools/check_neon_branch.py --list
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
STUB = os.path.join(HERE, 'arm_neon.h')

DIRS = ['ZQ_GEMM/math', 'ZQCNN/layers_c', 'ZQCNN/layers_nchwc', 'ZQCNN/math',
        '3rdparty/include/ZQlib']
# 与真实构建一致的旗标（除掉 x86 专属的那些，见 check_gates_runnable 的教训）
COMMON = '-O1 -fsyntax-only -std=gnu11 -Wall -Wextra ' \
         '-Wno-unused-parameter -Wno-unused-variable -Wno-unused-function ' \
         '-Wno-unused-but-set-variable -Wno-write-strings -Wno-unused-label'
INC = '-I$R/tools -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include ' \
      '-I$R/3rdparty/include/openblas'
NEON_RE = re.compile(r'\b(v[a-z0-9_]*q_[a-z0-9_]+)\b')


def wsl(script):
    p = subprocess.run('wsl -d %s -- bash -s' % WSL_DIST, shell=True,
                       input=script.encode('utf-8'), capture_output=True)
    return ((p.stdout or b'') + (p.stderr or b'')).decode('utf-8', 'replace')


def neon_regions(text):
    """所有 __ARM_NEON 区域（去掉 __ARM_NEON_FP16）的文本。

    用**栈**按行扫，而不是正则找区间 —— ET.3 那次 GLOB 配引号的教训：
    "看起来能工作"的正则匹配在预处理指令这种嵌套结构上很容易少算一层。
    """
    out, stack = [], []
    for ln in text.split('\n'):
        s = ln.strip()
        if re.match(r'#\s*if(def)?\b', s):
            stack.append(('__ARM_NEON' in s, '__ARM_NEON_FP16' in s))
            continue
        if re.match(r'#\s*endif\b', s):
            if stack:
                stack.pop()
            continue
        if re.match(r'#\s*(else|elif)\b', s):
            if stack:
                a, b = stack[-1]
                stack[-1] = (a or ('__ARM_NEON' in s), b)
            continue
        if any(a and not b for a, b in stack):
            out.append(ln)
    return '\n'.join(out)


def collect():
    """返回 (待检查文件, 用到的 NEON 名字集合, NEON 区域里有 __asm__ 的文件)"""
    used = set()
    files = []
    asm_files = []
    for d in DIRS:
        full = os.path.join(ROOT, d)
        if not os.path.isdir(full):
            continue
        for fn in sorted(os.listdir(full)):
            # **只收 .c / .cpp**。第一版把 `*.h` 也收进来，于是 46 个文件里
            # 一半报 `unknown type name 'zq_base_type'` ——
            # 那些是 `*_raw.h`，本来要由对应 .c 先定义 zq_base_type / zq_mm_*
            # 再 #include 进来，**单独编它们没有意义**。
            # 把"不是 TU 的东西当 TU 编"和"代码有 bug"分不开，是最费时间的
            # 一类假警报。
            if not fn.endswith(('.c', '.cpp')):
                continue
            p = os.path.join(full, fn)
            try:
                text = io.open(p, encoding='utf-8', errors='replace').read()
            except (IOError, OSError):
                continue
            if '__ARM_NEON' not in text:
                continue
            reg = neon_regions(text)
            if re.search(r'__asm__|\basm\s*\(', reg):
                asm_files.append(os.path.relpath(p, ROOT).replace('\\', '/'))
            for m in NEON_RE.finditer(reg):
                used.add(m.group(1))
            files.append(p)
    return files, used, asm_files


def stub_names():
    txt = io.open(STUB, encoding='utf-8', errors='replace').read()
    return set(re.findall(r'^#define\s+(v[a-z0-9_]+)\b', txt, re.M))


def compile_all(wdir):
    files, used, asm_files = collect()
    lines = ['set +e', 'R=%s' % WSL_ROOT, 'mkdir -p %s' % wdir,
             'cd %s || { echo "@@CDFAIL"; exit 1; }' % wdir]
    for i, p in enumerate(files):
        rel = os.path.relpath(p, ROOT).replace('\\', '/')
        cc = 'g++' if p.endswith('.cpp') else 'gcc'
        std = '-std=c++11' if p.endswith('.cpp') else '-std=gnu11'
        lines.append('%s %s %s -DZQ_CNN_USE_ARM_NEON %s "$R/%s" '
                     '2> e%d.txt; echo "@@RC%d=$?"'
                     % (cc, std, COMMON, INC, rel, i, i))
    script = '\n'.join(lines) + '\n'
    out = wsl(script)
    res = []
    for i, p in enumerate(files):
        rel = os.path.relpath(p, ROOT).replace('\\', '/')
        m = re.search(r'@@RC%d=(\d+)' % i, out)
        rc = int(m.group(1)) if m else -1
        first = ''
        if rc != 0:
            e = wsl('head -3 %s/e%d.txt' % (wdir, i))
            first = ' / '.join(x.strip() for x in e.splitlines()
                               if 'error' in x)[:220]
        res.append((rel, rc, first))
    wsl('rm -rf %s' % wdir)
    return res, used, asm_files, len(files)


def main():
    argv = sys.argv[1:]
    save_to = check_against = None
    for flag in ('--save-baseline', '--check-baseline'):
        if flag in argv:
            i = argv.index(flag)
            v = argv[i + 1]
            if flag == '--save-baseline':
                save_to = v
            else:
                check_against = v
            del argv[i:i + 2]
    list_only = '--list' in argv

    files, used, asm_files = collect()
    if list_only:
        for p in files:
            print(os.path.relpath(p, ROOT).replace('\\', '/'))
        return 0

    have = stub_names()
    missing = sorted(used - have)
    print('带 __ARM_NEON 分支的文件 = %d 个；NEON 区域里用到的名字 = %d 个'
          % (len(files), len(used)))
    print('桩里已定义 = %d 个；**桩缺少** = %d 个' % (len(have), len(missing)))
    if missing:
        for m in missing:
            print('  桩里没有：%s  <-- 补进 tools/arm_neon.h' % m)

    problems = []
    if missing:
        problems.append('桩不完整（%d 个名字）' % len(missing))
    if asm_files:
        problems.append('这 %d 个文件的 NEON 区域里有 __asm__，'
                        'x86 上编不出内联汇编，方案对它们不成立：%s'
                        % (len(asm_files), ', '.join(asm_files)))

    wdir = '/tmp/zqneon_%d_%d' % (os.getpid(), int(time.time()))
    try:
        res, used2, asm2, n = compile_all(wdir)
    finally:
        wsl('rm -rf %s' % wdir)

    bad = [(rel, why) for rel, rc, why in res if rc != 0]
    print('NEON 分支逐文件解析：%d 个文件，编不过 %d 个' % (n, len(bad)))
    # **没给基线时，"有文件编不过"本身就是失败。**
    # 第一版只在 --check-baseline 分支里把失败变成 problems，于是裸跑时
    # 明明报着"编不过 1 个"、退出码却是 0 ——
    # 对照 B 就是这么"被抓到"却又"没被抓住"的。
    # 判据的形状和 GK.2 那条元门禁一样：**失败必须落在退出码上**，
    # 打印出来不算。
    if not check_against and bad:
        problems.append('有 %d 个文件的 NEON 分支编不过'
                        '（没给基线时这本身就是失败）' % len(bad))

    if save_to:
        lines = ['# ARM/NEON 分支解析基线'
                 '（tools/check_neon_branch.py --save-baseline 生成）',
                 '# 格式: <文件>\\t<第一条 error>',
                 '# 本基线只登记「编不过」的文件；每一条都应当是人眼看过的。']
        for rel, why in bad:
            lines.append('%s\t%s' % (rel, why))
        with io.open(save_to, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(lines) + '\n')
        print('baseline written to %s (%d files)' % (save_to, len(bad)))

    if check_against:
        base = set()
        try:
            for line in io.open(check_against, encoding='utf-8'):
                if line.startswith('#') or not line.strip():
                    continue
                base.add(line.split('\t')[0].strip())
        except IOError as e:
            print('ERROR: 读不到基线 %s: %s' % (check_against, e))
            return 1
        cur = set(rel for rel, _ in bad)
        new = sorted(cur - base)
        gone = sorted(base - cur)
        print('=== 与基线 %s 比对 ===' % check_against)
        print('编不过的文件: 基线 %d -> 现在 %d' % (len(base), len(cur)))
        for rel, why in bad:
            if rel in new:
                print('  NEW  %-58s %s' % (rel, why))
            else:
                print('       %-58s %s' % (rel, why))
        for rel in gone:
            print('  FIXED %s' % rel)
        if not new and not gone:
            print('无新增、无消失。')
        if new:
            problems.append('有 %d 个文件新编不过' % len(new))

    for p in problems:
        print('  ' + p)
    if problems:
        return 1
    print('NEON 分支全部能解析（在本门禁能证明的范围内）。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
