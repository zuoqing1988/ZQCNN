#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZQCNN 审计这一轮加出来的全部检查，一个入口跑完（**在 Windows 侧跑**）。

    python tools/run_audit_checks.py                # 检查组（下面 A/B/C）
    python tools/run_audit_checks.py --quick        # 跳过慢的可编译性门禁
    python tools/run_audit_checks.py --with-build   # 再加上双平台全量构建 + sample 回归

分组：

D 主工程双平台回归（只在 --with-build 时跑）
    Windows  cmake --build build_x64 --config Release
    Linux    wsl 里的 /tmp/zqb2 make
    两边各跑一遍关键 sample（tools/run_sample_regression.sh 与等价的 exe 调用）

A 文本卫生（秒级）
    tools/check_line_endings.py    multi-CR / lone-CR / CRLF+LF 混用
    tools/check_text_encoding.py   UTF-8 有损解码残留（U+FFFD）

B 第三方头库的独立回归测试（ASan + LeakSanitizer，9 组，每组几秒）
    tools/run_zqlib_checks.py

C ZQlib 可编译性门禁（慢，约 2 分钟，编译 143 个翻译单元）
    tools/probe_zqlib_headers.py --check-baseline tools/zqlib_probe_baseline.txt

**为什么这个脚本必须是 Python 而不是 .sh**
B 和 C 里的两个工具本身是「Windows 侧 Python → 通过 `wsl ... bash -s` 喂脚本 →
在 WSL 里编译」。把它们放进一个 WSL 里的 shell 脚本去调用，会在 WSL 里再起一个
Python，然后那个 Python 想调 `wsl` —— 没有 `subprocess` 模块（那是 Windows 的
标准库）。2026-10-02 实测踩过：`AttributeError: 'module' object has no attribute 'run'`。
"""

from __future__ import print_function

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

GROUPS = [
    ('A1 行尾卫生 (check_line_endings)', ['check_line_endings.py'], False),
    ('A2 编码卫生 (check_text_encoding)', ['check_text_encoding.py'], False),
    ('B  ZQlib 独立回归测试 x9 (ASan+LSan)', ['run_zqlib_checks.py'], False),
    # 基线路径给**绝对路径**：子进程以 ROOT 为 cwd 运行，而基线文件在 tools/ 下，
    # 相对路径会解析成 <ROOT>/zqlib_probe_baseline.txt 而找不到（2026-10-02 实测）。
    ('C  ZQlib 可编译性门禁',
     ['probe_zqlib_headers.py', '--check-baseline',
      os.path.join(HERE, 'zqlib_probe_baseline.txt')], True),
]


WIN_BUILD = ['cmake', '--build', 'build_x64', '--config', 'Release']
LINUX_SAMPLES = ('cd /mnt/d/ZQCNN && bash tools/run_sample_regression.sh')

# 关键 sample：两个平台都要过（rc 必须为 0）。
# SampleGEMMAsmCompare 是汇编 vs intrinsic 的对拍，PASS 才算过；
# 其余是推理链路（MTCNN / SSD / CascadeOnet）。
WIN_SAMPLES = ['SampleGEMMAsmCompare.exe', 'SampleMTCNN.exe', 'SampleMTCNN_NCHWC4.exe',
               'SampleSSD.exe', 'SampleCascadeOnet.exe', 'SampleFaceDetectorMTCNN.exe']
WIN_BIN = os.path.join(ROOT, 'cmake-out-win32-x64', 'release', 'Release')


def run_group(name, cmd, cwd=None, shell=False):
    print('=' * 74)
    print('### %s' % name)
    # 子进程直接写同一个 fd, 不 flush 的话它的输出会排在父进程缓冲的 print 之前,
    # 读起来是乱的（2026-10-02 实测）
    sys.stdout.flush()
    p = subprocess.run(cmd, cwd=cwd, shell=shell)
    sys.stdout.flush()
    ok = (p.returncode == 0)
    print('--- %s: %s' % (name, 'OK' if ok else 'FAILED (rc=%d)' % p.returncode))
    return ok


def run_build_group():
    ok = True
    ok &= run_group('D1 Windows 全量构建 (VS2022/cmake)',
                    WIN_BUILD, cwd=ROOT)
    ok &= run_group('D2 Linux 全量构建 (gcc/wsl)',
                    'wsl -d Ubuntu-20.04 -- bash -c "cd /tmp/zqb2 && make -j8"')
    ok &= run_group('D3 Linux sample 回归',
                    'wsl -d Ubuntu-20.04 -- bash -c "%s"' % LINUX_SAMPLES)
    for exe in WIN_SAMPLES:
        path = os.path.join(WIN_BIN, exe)
        if not os.path.isfile(path):
            print('--- Windows sample %s: MISSING (%s)' % (exe, path))
            ok = False
            continue
        # 注意: sample 必须在**产物目录**里跑（CMake 把 model/ 和 data/ 联接到了那里），
        # 从仓库根跑只会打一行 empty image，看着像跑过了其实什么都没验。
        ok &= run_group('D4 Windows sample %s' % exe, [path], cwd=WIN_BIN)
    return ok


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='跳过慢的可编译性门禁（约 2 分钟）')
    ap.add_argument('--with-build', action='store_true',
                    help='额外跑双平台全量构建 + sample 回归（很慢，几分钟）')
    ap.add_argument('--msvc-probe', action='store_true',
                    help='额外用 MSVC 探一遍 ZQlib 头（Windows 侧覆盖，见附录 AR）')
    args = ap.parse_args()

    failed = []
    if args.with_build:
        if not run_build_group():
            failed.append('D 双平台构建 + sample 回归')

    if args.msvc_probe:
        bat = os.path.join(os.environ.get('TEMP', '.'), 'zqprobe_msvc.bat')
        with open(bat, 'w') as f:
            f.write('@echo off\r\n'
                    'call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community'
                    '\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul 2>&1\r\n'
                    'cd /d %s\r\n'
                    'python tools\\probe_zqlib_headers_msvc.py > "%%TEMP%%\\zqprobe_msvc_out.txt" 2>&1\r\n'
                    % ROOT)
        if not run_group('C2 MSVC 侧 ZQlib 头探测', ['cmd', '/c', bat], cwd=ROOT):
            failed.append('C2 MSVC 侧 ZQlib 头探测')
        out = os.path.join(os.environ.get('TEMP', '.'), 'zqprobe_msvc_out.txt')
        if os.path.isfile(out):
            try:
                sys.stdout.write(open(out, encoding='utf-8', errors='replace').read())
                rows = [l for l in open(out, encoding='utf-8', errors='replace')
                        if l.startswith(('OK', 'BROKEN'))]
                bad = [l for l in rows if l.startswith('BROKEN')]
                print('MSVC: %d 个头, OK %d, BROKEN %d'
                      % (len(rows), len(rows) - len(bad), len(bad)))
            except IOError:
                pass

    for name, argv, slow in GROUPS:
        if slow and args.quick:
            print('=' * 74)
            print('### %s：--quick 跳过' % name)
            continue
        ok = run_group(name, [sys.executable, os.path.join(HERE, argv[0])] + argv[1:],
                       cwd=ROOT)
        if not ok:
            failed.append(name)

    print('=' * 74)
    if not failed:
        print('ALL CHECKS PASSED')
        return 0
    print('%d CHECK GROUP(S) FAILED:' % len(failed))
    for n in failed:
        print('   %s' % n)
    return 1


if __name__ == '__main__':
    sys.exit(main())
