#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""层类型契约门禁（附录 GX）：**生产者写的层名，消费者必须认**。

这个仓库里有三个"生产者"和两个"消费者"：

  生产者            消费者
  ---------------   ------------------------------------------
  *.zqparams       ZQ_CNN_Layer_ReadParam（ZQ_CNN_Layer*.h）
  mobilefacenet-*  caffe（3rdparty/include/mini-caffe）—— 不在本门禁范围

随仓的 27 个 `.zqparams` 是**手写/历史产物**，不是这三个 Python 脚本产的
（附录 GU 已经量过：MATLAB 那 8 个脚本的产物一个都不在 `model/`）。
但它们**共用同一套层名**，所以三者都要在同一个判据里。

为什么要有这道门禁
------------------
两侧**都在这个仓库里**，却没有任何东西在核对它们一致。
C++ 侧改一个层名（`ReLU` -> `ReLU_v2`），或者某个转换器新增一个 ONNX op，
结果都是：**生成出来的模型加载失败**，而症状是运行期一句
"load failed"，离原因隔着一整条转换链 ——
本会话已经为同一种"两侧各自演化"吃过两次亏（MNN 分叉、FP16 路径）。

三条判据
--------
1. **Python 转换器写出的每个层名，C++ 读者必须认**（这是本门禁的主体）；
2. **随仓 27 个 `.zqparams` 里出现的每个层名，C++ 必须认** ——
   已有的 FD/GH 门禁验的是"能不能解析并连通"，**不验"层名在不在表里"**；
3. 统计 C++ 认但**没有生产者**的层名 —— **只报信息，不判失败**：
   很多层（Concat/Permute/Eltwise…）本来就来自手写 .zqparams 或 MATLAB。

用法:
    python tools/check_layer_type_contract.py
    python tools/check_layer_type_contract.py --selftest
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

# 消费者：从两个 Layer 头里取 `_my_strcmpi("X"` 的全部 X
CONSUMER_HEADS = [
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_Layer.h'),
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_Layer_NCHWC.h'),
]
# 生产者：写 `.zqparams` 的 Python
PRODUCERS = [
    os.path.join(ROOT, 'onnx_to_ZQCNN', 'onnx2ZQCNN.py'),
    os.path.join(ROOT, 'TensorFlow_to_ZQCNN', 'convertor.py'),
]
KNOWN_RE = re.compile(r'_my_strcmpi\(\s*"([A-Za-z0-9_]+)"')
# 转换器拼层行的形状：`line = 'Convolution name=...`
EMIT_RE = re.compile(
    r"""(?:line|type|name)\s*=\s*'([A-Z][A-Za-z0-9_]{2,})(?:\s+name=)""")
MODEL_DIR = os.path.join(ROOT, 'model')


def consumer_names():
    out = set()
    for p in CONSUMER_HEADS:
        if not os.path.isfile(p):
            raise SystemExit('找不到消费者头 %s —— 门禁失效（不是"通过"）' % p)
        t = io.open(p, encoding='utf-8', errors='replace').read()
        out |= set(KNOWN_RE.findall(t))
    if not out:
        raise SystemExit('从消费者头里一个层名都没抽到 —— 正则失效了')
    return out


def producer_names():
    out = {}
    for p in PRODUCERS:
        if not os.path.isfile(p):
            raise SystemExit('找不到生产者 %s —— 门禁失效（不是"通过"）' % p)
        t = io.open(p, encoding='utf-8', errors='replace').read()
        out[os.path.basename(p)] = set(EMIT_RE.findall(t))
    return out


def shipped_model_types():
    """随仓 .zqparams 里实际出现的层名。"""
    out = {}
    if not os.path.isdir(MODEL_DIR):
        return out
    for fn in sorted(os.listdir(MODEL_DIR)):
        if not fn.endswith('.zqparams'):
            continue
        t = io.open(os.path.join(MODEL_DIR, fn), encoding='utf-8',
                    errors='replace').read()
        got = set()
        for ln in t.split('\n'):
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            tok = ln.split()
            if tok:
                got.add(tok[0])
        if got:
            out[fn] = got
    return out


def selftest():
    """阳性对照：往"已知集合"里塞一个不存在的层名，比对必须报出来。

    直接复用 `check()` 的比对函数，不复制判据 ——
    复制一遍就会出现"对照过了、真的判据没测到"那种情况。
    """
    known = {'ReLU', 'Convolution'}
    emitted = {'ReLU', 'NoSuchLayer_XYZ'}
    shipped = {'m.zqparams': {'ReLU'}}
    probs = check({'p.py': emitted}, known, shipped)
    if not probs:
        return False, '塞了一个不存在的层名，比对却没报'
    if check({'p.py': {'ReLU'}}, known, shipped):
        return False, '全部合法时却误报了'
    if not check({'p.py': {'ReLU'}}, known, {'m.zqparams': {'Nope_XYZ'}}):
        return False, '随仓 .zqparams 里的非法层名没被抓到'
    return True, ''


def check(producers, known, shipped):
    problems = []
    for fn, names in sorted(producers.items()):
        bad = sorted(n for n in names if n not in known)
        if bad:
            problems.append('%s 写出的层名 C++ 不认：%s' % (fn, bad))
    for fn, names in sorted(shipped.items()):
        bad = sorted(n for n in names if n not in known)
        if bad:
            problems.append('model/%s 里的层名 C++ 不认：%s' % (fn, bad))
    return problems


def main():
    if '--selftest' in sys.argv:
        ok, why = selftest()
        print('阳性对照：不合法的层名被抓到、合法的不误报 =', ok)
        if not ok:
            print(why)
            return 2

    known = consumer_names()
    producers = producer_names()
    shipped = shipped_model_types()

    print('C++ 读者认识的层名 = %d 个' % len(known))
    for fn, names in sorted(producers.items()):
        print('  生产者 %-22s 写出 %2d 个：%s'
              % (fn, len(names), ' '.join(sorted(names))))
    print('随仓 .zqparams = %d 个' % len(shipped))

    probs = check(producers, known, shipped)
    for p in probs:
        print('  ' + p)

    used = set()
    for n in producers.values():
        used |= n
    for n in shipped.values():
        used |= n
    orphans = sorted(known - used)
    print('  （信息）C++ 认但没有任何生产者写的层名 = %d 个 —— '
          '**不判失败**，它们多来自手写 .zqparams' % len(orphans))

    if probs:
        return 1
    print('生产者与消费者的层名契约一致。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
