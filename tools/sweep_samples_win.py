"""逐个运行 Windows 上构建出的 sample，记录**真实的** Windows 退出码。

为什么要用 Python 而不是 bash 的 `$?`：Git Bash 对某些 Windows 程序的
退出码翻译不可靠 —— 实测 `mxnet2zqcnn.exe` 无参运行时打印了正常的 Usage
并由 cmd 报 EXIT=0，但 bash 的 `$?` 却是 127。同一台机器上 127 有两种完全
不同的来源（"进程起不来" vs "bash 翻译不出来"），混用会得出错误结论。

2026-10-01 首次全量扫描的收获：发现 `SampleFaceRecognizerArcFaceOpenCV`
以 0xC0000409 硬崩（OpenCV dnn 的 CV_Error 走 terminate()，调用方的
`if (!Init(...))` 根本没机会执行）。已修。

用法: python tools/sweep_samples_win.py [产物目录]
"""

import os
import subprocess
import sys

DIR = r'D:\ZQCNN\cmake-out-win32-x64\release\Release'
TIMEOUT = 20

exes = sorted(f for f in os.listdir(DIR) if f.lower().endswith('.exe'))
print('共 %d 个 exe\n' % len(exes))

buckets = {}
for e in exes:
    try:
        p = subprocess.run([os.path.join(DIR, e)], capture_output=True, timeout=TIMEOUT,
                           cwd=DIR)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = 'TIMEOUT'
    except OSError as ex:
        rc = 'OSERR:%s' % ex
    buckets.setdefault(str(rc), []).append(e)

print('=== 退出码分布 ===')
for rc in sorted(buckets, key=lambda s: (s == 'TIMEOUT', s)):
    names = buckets[rc]
    tag = {'0': '正常完成', '1': '主动失败(缺数据/缺参数，正常)',
           'TIMEOUT': '超时(长跑基准类)'}.get(rc, '<<< 需要看')
    print('rc=%-8s %2d 个   %s' % (rc, len(names), tag))
    if rc not in ('0', '1', 'TIMEOUT'):
        for n in names:
            print('        ', n)
print()
print('rc=0 :', ', '.join(buckets.get('0', [])))
print()
print('rc=TIMEOUT :', ', '.join(buckets.get('TIMEOUT', [])))
