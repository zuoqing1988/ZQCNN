#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 MSVC 对 3rdparty/include/ZQlib 的每个头做**语法检查**（Windows 侧）。

为什么需要这个
--------------
`tools/probe_zqlib_headers.py` 只用 gcc 探（跑在 WSL 里）。而 2026-10-02 这一轮
改了 ZQlib 下 26 个头，其中 5 个**真的链接进了 Windows 侧的 sample**
（见 `tools/zqlib_reachability.py`）。也就是说：那些改动在 Windows 上
**一次都没有被编译验证过**，更没有跑过（那批 sample 要人脸库/模型，仓库里没有）。

MSVC 与 gcc 的严格程度不同，而且 MSVC 对**反向**的一些东西更宽松：

| 现象 | gcc | MSVC |
|---|---|---|
| 嵌套类重复声明同名模板参数 | 报错 `shadows template parameter` | **放行** |
| 依赖类型缺 `typename` | 报错 `need 'typename'` | **放行** |
| `strcpy_s` / `_fseeki64` 等 | 无声明 | 有声明 |

也就是说 MSVC 放行的那些错，**正是我这一轮修掉的一批**。反过来，我补的
`typename`、`#include <ctime>`、`ZQ_TaucsBase::` 命名空间限定等改动，
也要确认 MSVC 仍然接受。这个脚本就是为了关掉这个口子。

用法（PowerShell / cmd 里先 vcvars64）：
    python tools/probe_zqlib_headers_msvc.py            # 全部头
    python tools/probe_zqlib_headers_msvc.py ZQ_Kmeans  # 只探名字里含这个串的

已知局限
--------
本机只装了 OpenCV 的**源码树**（D:/opencv3.4.2/opencv、D:/opencv4.5.1/opencv），
没有预编译的 include 目录，所以 14 个头会以 `fatal error C1083: cannot open include
file` 结束 —— 它们要 opencv2/ 、jpeglib.h、GL/glew.h、libav*、stdafx.h。
**这 14 个不是自身编译错**，主工程里那些路径是有的。判断「自身编译错」时要把
C1083 排除掉再看。

2026-10-02 的结果：143 个头里 **129 个 MSVC 编译通过，0 个自身编译错**
（排除上面那 14 个外部依赖之后）。
"""

from __future__ import print_function

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INC = os.path.join(ROOT, '3rdparty', 'include', 'ZQlib')
TMP = os.path.join(os.environ.get('TEMP', '/tmp'), 'zqprobe_msvc')

# MSVC 下不需要 shim（__int64/__min/__max/_fseeki64 都是它自己的）
SHIM = '#include <vector>\n#include <string>\n#include <cmath>\n'


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    flt = sys.argv[1] if len(sys.argv) > 1 else ''
    headers = sorted(h for h in os.listdir(INC) if h.endswith('.h') and flt in h)
    if not headers:
        print('no header matches %r' % flt)
        return 1

    if not os.path.isdir(TMP):
        os.makedirs(TMP)
    inc_win = INC

    # 主工程真正用的外部 include 路径（从 build_x64/CMakeCache.txt 的 OpenCV_DIR
    # 推出来，再按常见布局补齐）。不给的话 opencv2/ 那 14 个头会误报成
    # 「缺文件」，把「自身编译错」的真问题淹掉（2026-10-02 实测）。
    extra = []
    cache = os.path.join(ROOT, 'build_x64', 'CMakeCache.txt')
    if os.path.isfile(cache):
        for line in open(cache, encoding='utf-8', errors='replace'):
            if line.startswith('OpenCV_DIR:PATH='):
                p = line.split('=', 1)[1].strip()
                extra += [os.path.join(os.path.dirname(p), 'include')]
    for cand in (r'D:\opencv3.4.2\opencv\include',):
        if os.path.isdir(cand):
            extra.append(cand)
    extra = [e for e in extra if os.path.isdir(e)]

    for h in headers:
        src = os.path.join(TMP, h[:-2] + '.cpp')
        with open(src, 'w', encoding='utf-8', newline='\n') as f:
            f.write(SHIM)
            f.write('#include "%s"\n' % h)
            f.write('int main(){return 0;}\n')
        cmd = ['cl', '/nologo', '/Zs', '/EHsc', '/std:c++14', '/W3', '/I' + inc_win]
        cmd += ['/I' + e for e in extra]
        cmd.append(src)
        p = subprocess.run(
            cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if p.returncode == 0:
            print('OK       %s' % h)
        else:
            err = ''
            for line in (p.stdout or '').splitlines():
                if 'error' in line.lower():
                    err = line.strip()
                    break
            print('BROKEN   %-38s %s' % (h, err or (p.stdout or '').strip()[:70]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
