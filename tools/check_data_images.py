#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""data/ 图像门禁（附录 GT）：扩展名必须与内容一致，且必须能解码。

为什么要有它
------------
`data/` 里的图是**回归与示例的输入**，而附录 B-3 记着：
libjpeg 没有装 `setjmp` 时，一张畸形图片会让 libjpeg 直接 `exit()`，
把宿主进程带走。所以"图能不能解"不只是 sample 的事。

2026-10-04 实测发现 `data/mouth0.jpg` / `data/mouth1.jpg`
**是 PNG 内容、`.jpg` 扩展名**。OpenCV 按内容嗅探所以当时仍能读 ——
但任何**按扩展名分派**的调用方（`IMREAD_JPEG`、libjpeg 直连、文档、
第三方脚本）都会拿到错的东西。这类"现在没事、将来出事"的坑最难查。

判据三条，缺一不可：
  1. 魔数认出的格式 == 扩展名声称的格式；
  2. `cv::imread` 返回非空；
  3. **按魔数认出的格式强制解码**也要成功 ——
     这一条专门抓"扩展名骗人、而调用方是按格式来"的情形。

做法
----
编译 `tools/zq_imgcheck.cpp` 后跑一遍。包含路径**只给真实存在的目录** ——
给 g++ 一个不存在的 `-I` 它会直接报错，而"扫了等于没扫"被当成"干净"
正是这道检查最该防的失败模式（同 `check_no_gui_calls.py` 的做法）。

用法:
    python tools/check_data_images.py
    python tools/check_data_images.py --selftest
"""
import io
import os
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
SRC = 'tools/zq_imgcheck.cpp'

# 候选包含路径，**只保留真实存在的**（理由见模块 docstring）。
# 仓库内的相对路径按 Windows 侧判断存在性；`/usr/...` 这种 WSL 路径
# **原样交给编译器**去判 —— 在 Windows 侧 `os.path.isdir('/usr/...')`
# 恒为 False，自己判就等于把 WSL 上真实存在的路径全删掉。
_INC_CANDIDATES = ('3rdparty/opencv/build/include',
                   '3rdparty/include/opencv4', '3rdparty/include/opencv',
                   '/usr/include/opencv4', '/usr/local/include/opencv4')
_INC = []
for d in _INC_CANDIDATES:
    if d.startswith('/'):
        _INC.append('-I' + d)
    elif os.path.isdir(os.path.join(ROOT, d)):
        _INC.append('-I' + os.path.join(ROOT, d))
if not _INC:
    print('!! 找不到任何 OpenCV 包含路径 —— 这道门禁会"扫了等于没扫"')
    print('   候选：' + ' / '.join(_INC_CANDIDATES))
    sys.exit(2)


def wsl(script):
    p = subprocess.run('wsl -d %s -- bash -s' % WSL_DIST, shell=True,
                       input=script.encode('utf-8'), capture_output=True)
    return ((p.stdout or b'') + (p.stderr or b'')).decode('utf-8', 'replace')


def build_and_run(wdir, target=None):
    # **必须是仓库里的绝对路径**：脚本前面 `cd $wdir` 了，
    # 相对路径 'data' 在那里解析不到，第一版就是这么得到
    # "打不开目录 data" + rc=2 的。
    if target is None:
        target = WSL_ROOT + '/data'
    lines = ['set +e', 'R=%s' % WSL_ROOT, 'mkdir -p %s' % wdir,
             'cd %s || { echo "@@CDFAIL"; exit 1; }' % wdir,
             'g++ -O1 -std=c++11 %s $R/%s -o chk $(pkg-config --libs opencv4 2>/dev/null '
             '|| echo "-lopencv_core -lopencv_imgcodecs -lopencv_imgproc") '
             '2> cc.txt' % (' '.join(_INC), SRC),
             'if [ ! -x ./chk ]; then echo "@@NOBUILD"; head -5 cc.txt; exit 0; fi',
             './chk %s; echo "@@RC=$?"' % target]
    out = wsl('\n'.join(lines) + '\n')
    return out


def selftest():
    """阳性对照：造一个「扩展名骗人」的图，门禁必须报出来。

    造法是**复制仓库里真实存在的那张**（内容与扩展名不符的那两张之一），
    改名成 .jpg 之外的名字再改回去 —— 不手编字节，
    免得造出一个"连内容都不对"的样本，测的就不是扩展名那一维了。
    """
    wdir = '/tmp/zqimg_%d_%d' % (os.getpid(), int(time.time()))
    out = wsl('set +e\nR=%s\nD=%s\nmkdir -p $D/bad\n'
              # 先确认那两张 PNG 内容确实在（门禁自己会报它们）
              'cp $R/data/mouth0.jpg $D/bad/x.jpg 2>/dev/null\n'
              'cp $R/data/11.jpg $D/bad/good.jpg 2>/dev/null\n'
              'ls $D/bad\ncd $D\n'
              'g++ -O1 -std=c++11 %s $R/%s -o chk '
              '$(pkg-config --libs opencv4 2>/dev/null '
              '|| echo "-lopencv_core -lopencv_imgcodecs -lopencv_imgproc") 2>cc.txt\n'
              'if [ ! -x ./chk ]; then echo "@@NOBUILD"; head -5 cc.txt; fi\n'
              './chk bad; echo "@@RC=$?"\n'
              # 参数顺序必须**与占位符出现顺序一致**：R、D、_INC、SRC。
              # 第一版写成 (_INC, SRC, WSL_ROOT, wdir) —— 于是
              # `R=-I/usr/include/opencv4`、`D=tools/zq_imgcheck.cpp`，
              # 报出来的是"mkdir: cannot create directory 'tools/zq_imgcheck.cpp'"
              # 这种完全指不到真因的消息。
              % (WSL_ROOT, wdir, ' '.join(_INC), SRC))
    wsl('rm -rf %s' % wdir)
    if '@@NOBUILD' in out:
        # **失败信息必须带出真实输出**。第一版只说"OpenCV 找不到？"，
        # 而真因在 cc.txt 里 —— 报一句自己猜的原因，等于把排查推给下一个人。
        return False, '对照里没编出来，真实输出如下：\n' + out[-800:]
    if '扩展名不符' in out and '@@RC=1' in out:
        return True, ''
    return False, '注入的错标图没被抓到：\n' + out[-400:]


def main():
    if '--selftest' in sys.argv:
        ok, why = selftest()
        print('阳性对照：错标扩展名的图被报出来 =', ok)
        if not ok:
            print(why)
            return 2

    wdir = '/tmp/zqimg_%d_%d' % (os.getpid(), int(time.time()))
    try:
        out = build_and_run(wdir)
    finally:
        wsl('rm -rf %s' % wdir)
    if '@@NOBUILD' in out:
        print('编不出来：\n' + out)
        return 2
    sys.stdout.write(out)
    m = [l for l in out.splitlines() if l.startswith('扫了')]
    if not m:
        print('门禁没有给出汇总行 —— 判据不能只靠"没报错"。')
        return 2
    return 0 if '@@RC=0' in out else 1


if __name__ == '__main__':
    sys.exit(main())
