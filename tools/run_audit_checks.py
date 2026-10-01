#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZQCNN 审计这一轮加出来的全部检查，一个入口跑完（**在 Windows 侧跑**）。

    python tools/run_audit_checks.py
    python tools/run_audit_checks.py --quick      # 跳过慢的可编译性门禁

分三组：

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

**主工程（ZQCNN / ZQ_GEMM）的验证不在这里** —— 那是双平台 cmake 全量构建 +
`tools/run_sample_regression.sh` 的事，Windows 侧要单独跑。
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


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='跳过慢的可编译性门禁（约 2 分钟）')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    failed = []
    for name, argv, slow in GROUPS:
        if slow and args.quick:
            print('=' * 74)
            print('### %s：--quick 跳过' % name)
            continue
        print('=' * 74)
        print('### %s' % name)
        # 子进程直接写同一个 fd, 不 flush 的话它的输出会排在父进程缓冲的 print
        # 之前, 读起来是乱的
        sys.stdout.flush()
        p = subprocess.run([sys.executable, os.path.join(HERE, argv[0])] + argv[1:],
                           cwd=ROOT)
        sys.stdout.flush()
        if p.returncode == 0:
            print('--- %s: OK' % name)
        else:
            print('--- %s: FAILED (rc=%d)' % (name, p.returncode))
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
