"""用**截断二分支**定位某一层在 .nchwbin 里的字节区间（附录 HU）。

为什么不用"按层类型推 float 数"：那一版（HU 第一版）差了 37 万字节 ——
`.zqparams` 里**没有** `in_channels`，它是**从 bottom blob 的形状推出来的**，
所以纯静态的形状推断等于把 `SetBottomDim` 重写一遍。

这里换成一个**完全不需要形状推断**的办法：

  * 取 `.zqparams` 的**前 k 层**（Input + 若干层）写成临时参数文件；
  * 把权重文件**截短**到 X 字节；
  * 二分找**最小的、能加载成功的 X** —— 那就是前 k 层消费的字节数；
  * `f(k) - f(k-1)` = 第 k 层自己的字节数，**精确、无假设**。

**它顺带自证**：`f(0) = 0`、`f(全部层) = 文件长度`，
而 GZ 已经证明"LoadFrom 消费完整个文件、一个字节都不剩"（`zq_weight_tail` 门禁），
所以 f 的两端正好卡在已知的两个端点上 —— 差一格就说明哪一步错了。

用法：
    python tools/slice_model_weights.py model/mobilefacenet-v1.zqparams res4_block1_conv_dw
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile

# 探针 exe 放在**仓库内**的构建产物目录，而不是 tempfile.gettempdir()：
# Windows Python 的 tempdir 与 WSL 的 /tmp **不是同一个目录**（附录 HP.2 同一条教训），
# 探针在 WSL 里编出来之后，Windows 侧会"找不到"它，于是每次都重编。
PROBE_REL = os.path.join('cmake-out-unix-x64', 'Release', '.zq_slice_probe')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def to_wsl(p):
    a = os.path.abspath(p)
    return '/mnt/' + a[0].lower() + a[2:].replace(chr(92), '/') if len(a) > 1 and a[1] == ':' else a


def loadable(zqparams, nchwbin):
    """用一个小 C++ 探针 LoadFrom 一次，返回 True/False。"""
    exe = os.path.join(ROOT, PROBE_REL)
    if not os.path.exists(exe):
        print('探针还没编，见 slice_model_weights.py 的 build_probe()', file=sys.stderr)
        raise SystemExit(3)
    # 探针是 **Linux ELF**，Windows Python 不能原生执行它
    # （2026-10-04 实测：直接 subprocess.run 会报
    #  [WinError 193] %1 不是有效的 Win32 应用程序），
    # 所以**每一次探测都要经 wsl 启动**。
    wsl_exe = to_wsl(exe)
    r = subprocess.run(['wsl', '--', wsl_exe, to_wsl(zqparams), to_wsl(nchwbin)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0


def build_probe():
    """编一个只做 LoadFrom 的最小探针。"""
    src = os.path.join(ROOT, 'tools', 'zq_slice_probe.cpp')
    exe = os.path.join(ROOT, PROBE_REL)
    if os.path.exists(exe):
        return exe
    cmd = ['wsl', '--', 'g++', '-O1', '-mavx2', '-mfma', '-fopenmp',
           '-I/mnt/d/ZQCNN', '-I/mnt/d/ZQCNN/ZQCNN', '-I/mnt/d/ZQCNN/ZQ_GEMM',
           '-I/mnt/d/ZQCNN/3rdparty/include', '-I/mnt/d/ZQCNN/tools',
           '/mnt/d/ZQCNN/tools/zq_slice_probe.cpp',
           # 四个实现 TU 都得编，缺一个就一串 undefined reference：
           #   Tensor4D.cpp        —— 基础张量
           #   Tensor4D_NCHWC.cpp  —— BatchNormScale::SetBottomDim 里会
           #                          new ZQ_CNN_Tensor4D_NHW_C_Align256bit()
           #   两族 resize         —— 上面那两个张量类的 Resize* 要
           '/mnt/d/ZQCNN/ZQCNN/ZQ_CNN_Tensor4D.cpp',
           '/mnt/d/ZQCNN/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c',
           # **NCHWC 张量那个 .cpp 也要编**：
           # `new ZQ_CNN_Tensor4D_NHW_C_Align256bit()`，少它就一串 undefined reference
           # （2026-10-04 实测踩过）
           '/mnt/d/ZQCNN/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp',
           # NCHWC 张量那一套 resize 内核也要（两族：Align0/128/256 都在这一个 .c 里）
           '/mnt/d/ZQCNN/ZQCNN/layers_nchwc/zq_cnn_resize_nchwc.c',
           '-o', '/mnt/d/ZQCNN/' + PROBE_REL.replace(chr(92), '/')]
    print('编探针：%s' % ' '.join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit('探针编不过')
    return exe


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    args = sys.argv[1:]
    multi = False
    if args and args[0] == '--multi':
        multi = True
        args = args[1:]
    zp = args[0]
    target = args[1]
    base = zp[:-len('.zqparams')] + '.nchwbin'
    build_probe()

    raw = io.open(zp, encoding='utf-8', errors='replace').read().split('\n')
    lines = [l for l in raw if l.strip() and not l.strip().startswith('#')]
    total_bytes = os.path.getsize(base)
    data = open(base, 'rb').read()

    # 目标层的下标（按"非空非注释行"计）
    tgt = None
    for i, l in enumerate(lines):
        if ('name=%s' % target) in l or ('name=%s ' % target) in l:
            tgt = i
            break
    if tgt is None:
        print('参数文件里找不到层 %s' % target)
        return 1
    print('层 %s 在第 %d 行（共 %d 行）' % (target, tgt, len(lines)))

    # 临时切片文件也放在**仓库内**：探针是 Linux ELF，经 wsl 启动，
    # 它**看不见 Windows 的 temp 目录**（跨边界路径问题，今天第三次踩到）。
    tmp = os.path.join(ROOT, 'cmake-out-unix-x64', 'Release', '.zqslice')
    if not os.path.isdir(tmp):
        os.makedirs(tmp)
    wp = os.path.join(tmp, 'm.zqparams')
    wb = os.path.join(tmp, 'm.nchwbin')

    def f(k):
        """前 k 行 + 截短权重 -> 能否加载。"""
        with io.open(wp, 'w', encoding='utf-8', newline='\n') as f2:
            f2.write('\n'.join(lines[:k]) + '\n')
        return loadable(wp, wb)

    def consumed(k):
        """前 k 行消费的最小字节数。f(0) 视为 0。"""
        if k == 0:
            return 0
        lo, hi = 0, total_bytes + 4
        # 先确认 k 行确实能加载完整个文件，否则这个 k 不可用
        open(wb, 'wb').write(data)
        if not f(k):
            return None
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            open(wb, 'wb').write(data[:mid])
            if f(k):
                hi = mid
            else:
                lo = mid
        return hi

    # f(全部行) 必须 == 文件长度，否则说明"前 k 行"这条路有问题
    f_all = consumed(len(lines))
    print('全部 %d 行消费的字节 = %s（文件 %d）' % (len(lines), f_all, total_bytes))
    if f_all != total_bytes:
        print('**两端对不上** —— 二分结果不可信，先查工具')
        shutil.rmtree(tmp, ignore_errors=True)
        return 1

    for k in range(tgt, len(lines)):
        c = consumed(k + 1)
        if c is None:
            print('前 %d 行加载不完（前一层有悬空引用？）' % (k + 1))
            break
        name = lines[k].split()[0]
        prev = consumed(k)
        nbytes = (c - prev) if prev is not None else None
        if nbytes is not None:
            print('  第 %3d 行 %-24s 累计 %8d 字节，本层 %7d 字节 (%7d float)'
                  % (k, name, c, nbytes, nbytes // 4))
        if target in lines[k] or (tgt < k <= tgt + 2):
            pass
        if k >= tgt + 2:
            break
    # 把目标层**及其后随的那几层**（BN / PReLU）切出来，做成一个
    # 「Input + 那几层」的两三合成网的权重文件 —— 那就是 HU 要的
    # "把真实权重灌进合成网"的原料。
    tgt_k = None
    for k in range(tgt, len(lines)):
        if consumed(k + 1) is not None:
            tgt_k = k
            break
    if tgt_k is None:
        print('没能算出目标层的边界')
        return 1
    # 往后一直收到第一个「不含权重」的层为止（Input / ReLU / Concat …）
    end_k = tgt_k
    while end_k + 1 < len(lines):
        nxt = consumed(end_k + 2)
        cur = consumed(end_k + 1)
        if nxt is None or cur is None:
            break
        kind_next = lines[end_k + 1].split()[0]
        if kind_next in ('ReLU', 'ReLU6', 'Concat', 'Reshape', 'Permute',
                         'Squeeze', 'Flatten', 'Dropout', 'Copy', 'Eltwise',
                         'Input', 'Softmax'):
            break
        end_k += 1
    lo = consumed(tgt_k)
    hi = consumed(end_k + 1)
    out = os.path.join(tmp, 'slice.nchwbin')
    open(out, 'wb').write(data[lo:hi])
    lf = chr(10)
    open(os.path.join(tmp, 'slice_layers.txt'), 'w', encoding='utf-8', newline=lf).write(
        lf.join(lines[tgt_k:end_k + 1]) + lf)
    print()
    print('切出 %d 层：%s' % (end_k - tgt_k + 1, ' | '.join(lines[tgt_k:end_k + 1])))
    print('  权重 %d 字节（原文件 [%d, %d)）-> %s' % (hi - lo, lo, hi, out))
    print('  层定义 -> %s' % os.path.join(tmp, 'slice_layers.txt'))

    # 顺带出一个**能直接 LoadFrom 的合成网参数**（附录 HV）：
    # `Input` 的 **name 必须等于切出来那几层里第一层的 bottom** ——
    # Input 层的 blob 名就是它的层名（真模型里 `Input name=data C=3 ...`，
    # 而后面几层的 bottom 写的就是 data）。所以这里取第一层的 bottom 作为 Input 的 name。
    src_bottom = None
    cin = None
    for tok in lines[tgt_k].replace('\t', ' ').split():
        if tok.startswith('bottom='):
            src_bottom = tok[len('bottom='):]
        elif tok.startswith('num_output='):
            cin = tok[len('num_output='):]
    if src_bottom is None or cin is None:
        print('切出的第一层缺 bottom= 或 num_output=，做不成合成网')
        return 1
    # 通道数：depthwise 的 num_output 就是它的输入通道（一进一出）
    # H/W：真模型里这一层是 14x14（附录 HH.3 量到的 blob 形状 [1][14][14][256]）
    hw = 14
    synth = 'Input name=%s C=%s H=%d W=%d' % (src_bottom, cin, hw, hw)
    sp = os.path.join(tmp, 'slice.zqparams')
    with io.open(sp, 'w', encoding='utf-8', newline=lf) as fh:
        fh.write(synth + lf)
        fh.write(lf.join(lines[tgt_k:end_k + 1]) + lf)
    # **两棵产物树各放一份**：Windows 的 sample 跑在
    # `cmake-out-win32-x64/release/Release/` 下，Linux 的跑在
    # `cmake-out-unix-x64/Release/` 下，跨树用相对路径找是找不准的
    # （2026-10-04 实测：写死相对路径 -> Windows 侧报"找不到"，
    #  报得对、退出码也对，但那一侧等于没验）。所以**各放一份**。
    import shutil as _sh
    win = os.path.join(ROOT, 'cmake-out-win32-x64', 'release', 'Release', '.zqslice')
    if os.path.isdir(os.path.dirname(win)):
        if not os.path.isdir(win):
            os.makedirs(win)
        for _fn in ('slice.nchwbin', 'slice.zqparams', 'slice_layers.txt'):
            _sh.copyfile(os.path.join(tmp, _fn), os.path.join(win, _fn))
        print('  也复制了一份到 %s' % win)
    print('  合成网参数 -> %s' % sp)

    if multi:
        # --multi：把**若干层**的权重按顺序**拼**起来，配一个共享 blob 的合成网。
        #
        # 动机（附录 HW.1）：`res4` 段里 block1..block4 的 dwconv
        #     **四层的 bottom 全是 res4_block1_conv、top 全是 res4_block1_conv_dw**
        # —— 而 `res4_blockN_conv` 的 bottom 又都是 `_plusN`（Eltwise 的输出），
        # 所以**整段共享路径切不出来**（HV.4）。
        # 但**每一对 dwconv+BN 的权重是能单独切出来的**，
        # 而那四对层定义的 blob 名本来就一模一样 ——
        # 于是把四段权重**按顺序拼**起来、层定义**逐字照搬**，
        # 就得到了"一个 Input + 四个 dwconv+BN 写同一 blob"的网，
        # 而且**四个用的都是真实训练权重**。
        #
        # 这是 HV.3 那张表里**唯一没测过的格子**。
        parts = []
        # --multi 时 args[1:] 就是**全部**要拼的层（位置参数 target 也在里面，
        # 它同时也是单段模式的定位目标）。所以命令行里每一层**只传一次**。
        for t in args[1:]:
            ti = None
            for i, l in enumerate(lines):
                if ('name=%s' % t) in l:
                    ti = i
                    break
            if ti is None:
                print('找不到层 %s' % t)
                return 1
            # 往后收，直到遇到第一个"无权重"的层为止（BN 是有权重的，要收进去）
            ei = ti
            while ei + 1 < len(lines):
                kn = lines[ei + 1].split()[0]
                if kn in ('ReLU', 'ReLU6', 'Concat', 'Reshape', 'Permute',
                          'Squeeze', 'Flatten', 'Dropout', 'Copy', 'Eltwise',
                          'Input', 'Softmax'):
                    break
                ei += 1
            lo2, hi2 = consumed(ti), consumed(ei + 1)
            if lo2 is None or hi2 is None:
                print('%s 那段算不出边界' % t)
                return 1
            parts.append((t, ti, ei, lo2, hi2))
            print('  %-24s 行 [%d..%d] 权重 %d 字节 [%d, %d)'
                  % (t, ti, ei, hi2 - lo2, lo2, hi2))
        blob = None
        wbuf = b''
        plist = []
        for t, ti, ei, lo2, hi2 in parts:
            if blob is None:
                for tok in lines[ti].split():
                    if tok.startswith('bottom='):
                        blob = tok[len('bottom='):]
            wbuf += data[lo2:hi2]
            plist.extend(lines[ti:ei + 1])
        wm = os.path.join(tmp, 'multi.nchwbin')
        open(wm, 'wb').write(wbuf)
        wpb = os.path.join(tmp, 'multi.zqparams')
        with io.open(wpb, 'w', encoding='utf-8', newline=lf) as fh:
            fh.write('Input name=%s C=256 H=14 W=14' % blob + lf)
            fh.write(lf.join(plist) + lf)
        print()
        print('  拼出来：%d 段共 %d 字节，%d 层' % (len(parts), len(wbuf), len(plist)))
        print('  共享 blob：bottom=%s，top=%s（四段逐字照搬，所以 blob 名天然一致）'
              % (blob, blob + '_dw'))
        print('  权重 -> %s' % wm)
        print('  参数 -> %s' % wpb)
        for _fn in ('multi.nchwbin', 'multi.zqparams'):
            _sh.copyfile(os.path.join(tmp, _fn), os.path.join(win, _fn))
        return 0
    print('    %s' % synth)
    return 0


if __name__ == '__main__':
    sys.exit(main())
