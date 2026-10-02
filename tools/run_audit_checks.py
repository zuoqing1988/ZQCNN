#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZQCNN 审计这一轮加出来的全部检查，一个入口跑完（**在 Windows 侧跑**）。

    python tools/run_audit_checks.py                # 检查组（下面 A/B/C）
    python tools/run_audit_checks.py --quick        # 跳过慢的可编译性门禁
    python tools/run_audit_checks.py --with-build   # 再加上双平台全量构建 + sample 回归
    python tools/run_audit_checks.py --ubsan        # B 组换成 UBSan 再跑一遍

分组：

D 主工程双平台回归（只在 --with-build 时跑）
    Windows  cmake --build build_x64 --config Release
    Linux    wsl 里的 /tmp/zqb2 make
    两边各跑一遍关键 sample（tools/run_sample_regression.sh 与等价的 exe 调用）

A 文本与配对卫生（秒级）
    tools/check_line_endings.py    multi-CR / lone-CR / CRLF+LF 混用
    tools/check_text_encoding.py   UTF-8 有损解码残留（U+FFFD）
    tools/check_alloc_delete.py    malloc 配 delete[] / new 配 free（--selftest 先自测）

B 第三方头库的独立回归测试（10 组，每组几秒）
    tools/run_zqlib_checks.py        (gcc / WSL，ASan+LSan 或 UBSan)
    tools/run_zqlib_checks_msvc.bat  (MSVC /fsanitize=address / Windows，--msvc-asan 时跑)

C ZQlib 可编译性与警告门禁（慢，各约 2 分钟）
    C   tools/probe_zqlib_headers.py --check-baseline tools/zqlib_probe_baseline.txt
    C2  MSVC 侧头探测                       (--msvc-probe)
    C3  gcc -Wall -Wextra 的 HIGH 桶基线    (--warn-sweep)

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
    # A3 只有 5 秒，但它挡掉的是**整个工具自己变成哑巴**这件事：
    # 改 check_alloc_delete.py 的匹配逻辑之后忘了跑自测，那它返回的「没有命中」
    # 就毫无意义（附录 AT.8 的教训）。
    ('A3 分配/释放配对扫描自测 (check_alloc_delete --selftest)',
     ['check_alloc_delete.py', '--selftest'], False),
    ('A4 malloc/delete 错配扫描 (全仓 706 个源文件)',
     ['check_alloc_delete.py'], False),
    ('A5 未初始化类成员扫描自测 (check_uninit_members --selftest)',
     ['check_uninit_members.py', '--selftest'], False),
    ('A6 未初始化类成员扫描 (ZQCNN/*.h)',
     ['check_uninit_members.py', '--all'], False),
    # A7 是 A5 那类"自测"思路的延续：值域校验基线保证
    # 「已经在 ReadParam 里查过的参数」不会悄悄丢掉守卫。
    # BD/BE/BF 三条缺陷（pooling 除零、卷积 SIGFPE、Tile 堆溢出）都是这一族漏网的实例。
    ('A7 ReadParam 值域校验基线 (check_param_domain)',
     ['check_param_domain.py', '--selfcheck'], False),
    ('A8 ReadParam 值域校验基线比对 (check_param_domain)',
     ['check_param_domain.py', '--check-baseline'], False),
    # A9/A10 是 BE 的"同一类收口"门禁：BE 修了 NCHW 那 7 处 `/ strideH`，
    # 忘了 NCHWC 那 25 处，是 BG 的工具抓出来的。这两个组保证以后不会再漏。
    ('A9 "除以模型参数" 守卫普查自测 (check_div_guard --selfcheck)',
     ['check_div_guard.py', '--selfcheck'], False),
    ('A10 "除以模型参数" 守卫普查 (check_div_guard)',
     ['check_div_guard.py'], False),
    # A11/A12 是 BM 的门禁：BM 修了 ZQlibFaceID/ZQ_FaceRecognizerUtils.h 里
    # 两处没检查返回值的 cv::invert（失败时输出 Mat 是空的 -> 后面空指针解引用）。
    # 同一个动机：修了一处不等于只有这一处，靠人记得普查是靠不住的。
    ('A11 "丢弃 OpenCV bool 返回值" 自测 (check_uncked_cv_return --selfcheck)',
     ['check_uncked_cv_return.py', '--selfcheck'], False),
    ('A12 "丢弃 OpenCV bool 返回值" 普查 (check_uncked_cv_return)',
     ['check_uncked_cv_return.py'], False),
    # A13/A14 是附录 CC 的门禁：NCHWC 那一族的 _C3 内核把通道数 3 **硬编码**进
    # im2col 展开（实测传 C=4 / C=6，输出与 C=3 逐位相同 —— 不报错、不崩溃，
    # 只是安静地只算前 3 个通道，比崩溃更危险），所以每个调用点都得有 C==3 守卫。
    # 现状 29 个调用点全部有守卫；这个门禁保证以后新增调用点时不会漏。
    # **只管 NCHWC 那一族**：NCHW（layers_c）那一族也叫 _C3，但它是通用实现
    # （memcpy 按实际 filter_C），守的是 in_C <= 4 / <= 8，不要求恰好等于 3。
    ('A13 "NCHWC _C3 调用点的 C==3 守卫" 自测 (check_c3_guards --selfcheck)',
     ['check_c3_guards.py', '--selfcheck'], False),
    ('A14 "NCHWC _C3 调用点的 C==3 守卫" 普查 (check_c3_guards)',
     ['check_c3_guards.py'], False),
    ('B  ZQlib 独立回归测试 x10 (ASan+LSan)', ['run_zqlib_checks.py'], False),
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
    ap.add_argument('--msvc-asan', action='store_true',
                    help='额外用 MSVC /fsanitize=address 把 9 个 ZQlib 测试在 Windows 上真跑一遍'
                         '（gcc 那套只在 WSL 里跑，见附录 AS）')
    ap.add_argument('--ubsan', action='store_true',
                    help='把 B 组换成 -fsanitize=undefined 再跑一遍（抓 ASan 看不见的'
                         '有符号溢出/移位越界等，见附录 AS.2）')
    ap.add_argument('--warn-sweep', action='store_true',
                    help='额外跑 gcc -Wall -Wextra 的 HIGH 桶门禁（较慢，约 2 分钟，见附录 AT）')
    ap.add_argument('--src-sweep', action='store_true',
                    help='额外扫**主工程** ZQCNN/ 的 43 个 TU 的 HIGH 桶（约 1 分钟，见附录 AU）')
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

    if args.warn_sweep:
        if not run_group('C3 gcc -Wall/-Wextra HIGH 桶门禁',
                         [sys.executable, os.path.join(HERE, 'warn_sweep_zqlib.py'),
                          '--check-baseline',
                          os.path.join(HERE, 'zqlib_warn_baseline.txt')],
                         cwd=ROOT):
            failed.append('C3 gcc -Wall/-Wextra HIGH 桶门禁')

    if args.src_sweep:
        if not run_group('C4 主工程 ZQCNN/ 的 HIGH 桶门禁',
                         [sys.executable, os.path.join(HERE, 'warn_sweep_src.py'),
                          '--check-baseline',
                          os.path.join(HERE, 'zqcnn_warn_baseline.txt')],
                         cwd=ROOT):
            failed.append('C4 主工程 ZQCNN/ 的 HIGH 桶门禁')

    if args.msvc_asan:
        if not run_group('B2 ZQlib 独立回归测试 x10 (MSVC /fsanitize=address)',
                         ['cmd', '/c', os.path.join(HERE, 'run_zqlib_checks_msvc.bat')],
                         cwd=ROOT):
            failed.append('B2 ZQlib 独立回归测试 x10 (MSVC ASan)')

    for name, argv, slow in GROUPS:
        if slow and args.quick:
            print('=' * 74)
            print('### %s：--quick 跳过' % name)
            continue
        cmd = [sys.executable, os.path.join(HERE, argv[0])] + argv[1:]
        if name.startswith('B ') and args.ubsan:
            cmd.append('--ubsan')
        ok = run_group(name, cmd, cwd=ROOT)
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
