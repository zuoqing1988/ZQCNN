#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""每一条门禁自己必须先能跑起来 —— 元门禁。

为什么需要它（附录 GK）
----------------------
`tools/check_filecount_bounds.py` 的第 50 行是一句**残句**：
一行 6 空格缩进、没有 `#` 的中文（`      **用比较表达的上界**都识别不出来**，`），
它从一行被截断的注释尾巴里掉下来，正好落在 `UB_TEMPLATES = [` 的上一行。
Python 因此在**解析期**就抛 `SyntaxError: unexpected indent` ——
这个文件从头到尾**一次都没被执行过**。

而它并不小：它是附录 EL 的全部依据，`audit_k3_20261001.md:13287`
直接把结论建立在它身上。**一份报告引用着一个跑不起来的工具，
而回归全绿。**

为什么回归发现不了
----------------
两条原因叠在一起，缺一不可：

1. 它**没有接进** `tools/run_audit_checks.py`；
2. 就算接进去了，`run_group()` 判的是**子进程退出码** ——
   而 `python 一个编不过的脚本` 退出码**就是** 1，会被抓到。
   真正的问题在第 1 条：**没人跑它**。

也就是说，这 40 多道门禁**互相不看对方**。
`check_text_encoding.py` 会发现它写坏了编码（那 4 类门禁里有），
`check_line_endings.py` 也会看它 —— 但**没有任何一道门禁看「你能不能解析」**。
这正是「找没有任何东西在看的地方」这条高产模式的又一次命中。

判据
----
仓库里**每一个**受版本控制的 `.py`：
  1. 能按严格 UTF-8 解码；
  2. 能通过 `compile(src, path, 'exec')`（用当前解释器的语法版本）。

外加：每个 `tools/*.sh` 能通过 `bash -n`。

**只查 `tools/` 是不够的** —— 同一批事故也污染过仓库里非门禁的
Python（`mobilefacenet-mxnet2caffe-ZQ/` 那三个文件是 Python 2 的
`print` 语句，在 Python 3 下同样一次都跑不起来）。
`TensorFlow_to_ZQCNN/convertor.py`、`onnx_to_ZQCNN/onnx2ZQCNN.py`
同样在扫描范围内。

`compile()` 而不是真的执行：执行会把 `import torch` 之类的重依赖
拖进来（这个仓库的转换器脚本都需要 mxnet/caffe），
而**语法层能不能解析**才是这里要回答的问题。

读文件清单用 `git ls-files -z` + 手工按 utf-8 解码：
`text=True` 在中文 Windows 上会用区域编码（GBK）解 git 的输出，
而仓库里有 `reports/ZQCNN_上层优化手段汇总.md` 这类**中文文件名**，
一撞就 `UnicodeDecodeError`（2026-10-03 实测）。
`tools/check_line_endings.py:68` 早就是这个写法，这里沿用。

用法:
    python tools/check_gates_runnable.py
    python tools/check_gates_runnable.py --selftest    # + 阳性对照
"""
import io
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WSL_DIST = 'Ubuntu-20.04'


def git_files(*pathspecs):
    """受版本控制的文件（bytes 模式，见模块 docstring 的中文文件名说明）。"""
    cmd = ['git', '-C', ROOT, 'ls-files', '-z'] + list(pathspecs)
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return [n.decode('utf-8', 'surrogateescape')
            for n in out.split(b'\0') if n]


def check_python(paths):
    """返回 [(relpath, 原因)]"""
    bad = []
    for rel in paths:
        full = os.path.join(ROOT, rel)
        try:
            raw = open(full, 'rb').read()
        except (IOError, OSError) as e:
            bad.append((rel, '读不了: %s' % e))
            continue
        try:
            src = raw.decode('utf-8')
        except UnicodeDecodeError as e:
            bad.append((rel, '不是严格 UTF-8: %s' % e))
            continue
        try:
            compile(src, rel, 'exec')
        except SyntaxError as e:
            bad.append((rel, 'line %s: %s' % (e.lineno, e.msg)))
        except ValueError as e:
            # 源码里有 NUL 之类的，compile 会抛 ValueError 而不是 SyntaxError
            bad.append((rel, 'ValueError: %s' % e))
    return bad


def check_shell(paths):
    """一次 wsl 调用批检所有 .sh（每文件一次 wsl 要几十秒，见 check_no_gui_calls.py）。"""
    if not paths:
        return []
    wsl_paths = ' '.join("/mnt/d/ZQCNN/'%s'" % p.replace("'", "'\\''")
                         for p in paths)
    p = subprocess.run(
        'wsl -d %s -- bash -s' % WSL_DIST, shell=True,
        input=('for f in %s; do bash -n "$f" || echo "BAD $f"; done\n'
               % wsl_paths).encode('utf-8'),
        capture_output=True)
    text = ((p.stdout or b'') + (p.stderr or b'')).decode('utf-8', 'replace')
    return [(ln[4:].strip(), 'bash -n 报语法错')
            for ln in text.splitlines() if ln.startswith('BAD ')]


def selftest():
    """阳性对照：造一个**确实编不过**的 .py，本工具必须报出来。

    没有这一步的话，「扫了 714 个文件、0 个有问题」和
    「一个文件都没扫到」在输出上一模一样 —— 这正是 FC 那次的教训。
    """
    d = tempfile.mkdtemp(prefix='zqgate_')
    good = os.path.join(d, 'good.py')
    bad = os.path.join(d, 'bad.py')
    with io.open(good, 'w', encoding='utf-8') as f:
        f.write('x = 1\nprint(x)\n')
    with io.open(bad, 'w', encoding='utf-8') as f:
        f.write('def f(:\n    pass\n')
    found = check_python([good, bad])
    ok = (not any(r == good for r, _ in found)
          and any(r == bad for r, _ in found))
    print('阳性对照：好文件未被误报=%s，坏文件被报出=%s'
          % (not any(r == good for r, _ in found),
             any(r == bad for r, _ in found)))
    for r in os.listdir(d):
        os.remove(os.path.join(d, r))
    os.rmdir(d)
    return ok


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    selftest_mode = '--selftest' in sys.argv
    pys = git_files('*.py')
    shs = git_files('*.sh')

    bad_py = check_python(pys)
    bad_sh = check_shell(shs)

    print('扫了 %d 个 .py、%d 个 .sh' % (len(pys), len(shs)))
    for rel, why in bad_py:
        print('  编不过 %-52s %s' % (rel, why))
    for rel, why in bad_sh:
        print('  编不过 %-52s %s' % (rel, why))

    if selftest_mode:
        if not selftest():
            print('阳性对照没通过 —— 本工具可能对任何输入都报「一切正常」')
            return 2

    n = len(bad_py) + len(bad_sh)
    if n:
        print('有 %d 个脚本**解析不了**，其中至少一条是门禁 —— 门禁自己跑不起来，'
              '它保护的东西等于没人保护。' % n)
        return 1
    print('全部脚本都能解析。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
