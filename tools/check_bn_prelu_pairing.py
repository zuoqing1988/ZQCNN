#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""「卷积 → BN / PReLU」接线门禁（附录 HX.3）。

判据只有一条，越具体越好
----------------------
对每个 `.zqparams` 里**相邻**的这两组层：

    Convolution / DepthwiseConvolution / InnerProduct   (下称 conv)
  + BatchNormScale / PReLU                              (下称 act)

要求后一层的 ``bottom=`` **就是**前一层的 ``top=``。不成立就是一处
「后一层读的并不是前一层写的那份数据」。

为什么这条要单独立一道门禁
--------------------------
`_merge_bn` / `_merge_prelu` 会把这两层折成一层，而折的**前提**就是
「后一层吃的就是前一层写的那个 blob」（``ZQ_CNN_Net.h`` 里
``tops[i][0] == bottoms[i+1][0]`` 那一条，附录 HX.2）。
不成立时折叠仍会**静默改变结果**：`model/mobilefacenet-v1.zqparams`
第 108/109 行就是这样写的，于是生产实参下它的输出被改了 **0.37**
（后向误差，附录 HE.2）—— 十一轮二分（HE~HR）都没抓到，直到把
「block5 的 dwconv 写了一个没人读的 blob」这一格补上。

所以这道门禁的作用是：**下一个模型再写出这种接线时，在加载之前就报出来**，
而不是等到某天输出对不上再从头二分。

随仓现状
--------
17 个模型里**只有 1 处**，就是 `mobilefacenet-v1` 的 108/109 行。
那是**模型文件本身**的接线问题（`.nchwbin` 里 block5 的 dwconv 权重
因此一直是死代码），本仓库不拥有那个模型、也不该悄悄改它的语义，
所以按基线记账：基线里的条目**只报信息**，基线外的**判失败**。

用法:
    python tools/check_bn_prelu_pairing.py
    python tools/check_bn_prelu_pairing.py --selftest
    python tools/check_bn_prelu_pairing.py --rewrite-baseline
"""
import io
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, 'model')
BASELINE = os.path.join(HERE, 'bn_prelu_pairing_baseline.txt')

CONV_TYPES = ('Convolution', 'DepthwiseConvolution', 'InnerProduct')
ACT_TYPES = ('BatchNormScale', 'PReLU')
TOKEN_RE = {
    'name': re.compile(r'\bname=(\S+)'),
    'top': re.compile(r'\btop=(\S+)'),
    'bottom': re.compile(r'\bbottom=(\S+)'),
}


def _tok(line, key):
    m = TOKEN_RE[key].search(line)
    return m.group(1) if m else None


def scan_text(text, origin):
    """返回 [(origin, 行号, conv 名, conv.top, act 名, act.bottom), ...]。

    `origin` 只是给人看的来源标签，判据本身不依赖它。
    """
    out = []
    lines = text.split('\n')
    for i in range(len(lines) - 1):
        a, b = lines[i], lines[i + 1]
        ta = a.split()
        tb = b.split()
        if not ta or not tb:
            continue
        if ta[0] not in CONV_TYPES or tb[0] not in ACT_TYPES:
            continue
        ct = _tok(a, 'top')
        cb = _tok(b, 'bottom')
        if ct and cb and ct != cb:
            out.append((origin, i + 1, _tok(a, 'name') or '?',
                        ct, _tok(b, 'name') or '?', cb))
    return out


def scan_models(model_dir):
    if not os.path.isdir(model_dir):
        return None, 0
    found = []
    n = 0
    for fn in sorted(os.listdir(model_dir)):
        if not fn.endswith('.zqparams'):
            continue
        n += 1
        t = io.open(os.path.join(model_dir, fn), encoding='utf-8',
                    errors='replace').read()
        found.extend(scan_text(t, fn))
    return found, n


def key_of(item):
    """基线比对用的身份：模型 + 层名对。行号会随模型改动漂移，不进 key。"""
    return '%s\t%s\t%s' % (item[0], item[2], item[4])


def fmt(item):
    return ('%-28s 行 %-4d %-24s top=%-26s | %-28s bottom=%s'
            % item)


def read_baseline():
    if not os.path.isfile(BASELINE):
        return None
    t = io.open(BASELINE, encoding='utf-8').read()
    # 井号开头的是给人看的说明行，**不能**当条目 ——
    # 2026-10-05 第一版没滤掉，于是 `--rewrite-baseline` 之后紧跟着的一次
    # 正常运行报"基线里有 4 条、只扫到 1 条"，差点让人以为基线坏了。
    return set(x for x in (ln.strip() for ln in t.split('\n'))
               if x and not x.startswith('#'))


def selftest():
    """阳性对照：造一段**接线对不上**的 .zqparams，判据必须抓到它；
    再造一段接线正常的，必须**不**报。只跑"合法"那一路等于没测。"""
    bad = ('Input name=data C=3 H=8 W=8\n'
           'Convolution\tname=c1\tbottom=data\ttop=blobA num_output=4 kernel_size=1\n'
           'BatchNormScale\tname=c1_bn\tbottom=blobB\ttop=blobB bias\n')
    good = ('Input name=data C=3 H=8 W=8\n'
            'Convolution\tname=c1\tbottom=data\ttop=blobA num_output=4 kernel_size=1\n'
            'BatchNormScale\tname=c1_bn\tbottom=blobA\ttop=blobA bias\n')
    if not scan_text(bad, 'x.zqparams'):
        return False, 'bottom 与 top 对不上却没报'
    if scan_text(good, 'x.zqparams'):
        return False, '接线正常却误报了'
    # 非相邻的一对不算：`_merge_bn` 折的是 i / i+1，中间隔了别的层就不是它的对象。
    # （第一版这里写的是 conv / PReLU / BN，而那三行的**前两行本来就是相邻的**，
    #   于是它报了出来 —— 报的是对的，错的是我这个"阴性对照"本身，
    #   差点让我去改判据。阴性对照也得自己过一遍阳性对照。）
    far = ('Convolution\tname=c1\tbottom=data\ttop=blobA num_output=4 kernel_size=1\n'
           'ReLU\tname=r1\tbottom=blobA\ttop=blobA\n'
           'BatchNormScale\tname=c1_bn\tbottom=blobB\ttop=blobB bias\n')
    if scan_text(far, 'x.zqparams'):
        return False, '不相邻的一对被误判成相邻'
    return True, ''


def main():
    if '--selftest' in sys.argv:
        ok, why = selftest()
        print('阳性对照：接线对不上的被抓到、正常的与不相邻的不误报 =', ok)
        if not ok:
            print(why)
            return 2

    found, nmodels = scan_models(MODEL_DIR)
    if found is None:
        print('**FAIL** 找不到 %s —— 一个模型都没扫（这不是"通过"）' % MODEL_DIR)
        return 1

    base = read_baseline()
    if base is None:
        print('**FAIL** 找不到基线文件 %s —— 没有基线就没法区分'
              '"已知的模型问题"和"新引入的问题"' % BASELINE)
        return 1

    print('扫了 %d 个 .zqparams；「后一层读的 blob ≠ 前一层写的 blob」共 %d 处'
          % (nmodels, len(found)))
    for it in found:
        state = '已记账' if key_of(it) in base else '**新增**'
        print('  [%s] %s' % (state, fmt(it)))
    if len(base) > len(found):
        print('  （信息）基线里有 %d 条、现在只扫到 %d 条 —— '
              '可能有人改了模型或基线，需要看一眼' % (len(base), len(found)))

    new = [it for it in found if key_of(it) not in base]
    if new:
        print('\nBN/PReLU 接线门禁 FAILED：%d 处基线之外的错接 —— '
              '`_merge_bn` / `_merge_prelu` 会在这些地方静默改变结果' % len(new))
        print('  修法：要么把 .zqparams 的 bottom= 改成上一层的 top=，'
              '要么确认这是有意为之并 `--rewrite-baseline` 记账。')
        return 1
    print('\nBN/PReLU 接线门禁 OK（%d 处基线内的错接，不判失败）' % len(found))
    return 0


if __name__ == '__main__':
    if '--rewrite-baseline' in sys.argv:
        found, n = scan_models(MODEL_DIR)
        if found is None:
            print('找不到 %s' % MODEL_DIR)
            sys.exit(1)
        with io.open(BASELINE, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('# 「后一层读的 blob ≠ 前一层写的 blob」基线（附录 HX.3）\n')
            fh.write('# 一行一条，格式 = key_of()：<模型>\\t<conv 层名>\\t<act 层名>\n')
            fh.write('# 用 --rewrite-baseline 重写；改动前请先想清楚这是不是缺陷。\n')
            for it in sorted(found, key=key_of):
                fh.write(key_of(it) + '\n')
        print('基线已重写：%d 条 -> %s' % (len(found), BASELINE))
        sys.exit(0)
    sys.exit(main())
