# -*- coding: utf-8 -*-
"""盘点：哪些 sample 会被构建，但**回归从不执行**（附录 GV，诊断工具，不是门禁）。

**它不建立什么**（2026-10-04 实测）：
静态分类出"资产齐备、不需参数、非平台专属"的 9 个，逐个实跑之后
**能加进回归的是 0 个** —— 4 个是基准（120s 超时）、3 个是平台桩（0 行输出）、
2 个要视频设备。所以"跑起来"不是补覆盖的手段；
到今天为止真正**买到**覆盖的只有 GS 那一招（让能断言的 sample 真的断言）。
静态分类的价值是**划出边界**，不是替代实跑。

三次返工的记录（都写在这一版里）：
  1. 路径以 `\` 结尾，而我只 `rstrip('/')` -> `os.path.basename` 返回空串，
     表格里出现 `err_log.txt` 那种莫名其妙的"sample 名"。
  2. 用 `argv[` 判断"要参数"是**错的**：`printf("%s only support windows",
     argv[0])` 只是取程序名，与要不要传参无关。
     真正的信号是 `argc` 判断。
  3. 上一版**没有断言**结果非空，于是坏输出照样打印成一张"看起来很像结论"
     的表。现在：名字必须非空、必须唯一、必须与磁盘上的目录一一对应。
"""
import io
import os
import re
import glob
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.abspath('.')
SEP = '\\/'          # 两种分隔符都要认


def dirs_of(sample_root):
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, sample_root, '*'))):
        base = os.path.basename(d.rstrip(SEP))
        if os.path.isdir(d) and base and not base.startswith('.'):
            out.append((base, d))
    return out


def read_src(d):
    parts = []
    for pat in ('*.cpp', '*.c', '*.h'):
        for f in glob.glob(os.path.join(d, pat)):
            parts.append(io.open(f, encoding='utf-8', errors='replace').read())
    return '\n'.join(parts)


def missing_assets(s):
    """返回 'model/' 与 'data/' 里都没有的资产文件名。"""
    out = []
    for m in sorted(set(re.findall(
            r'["\']([\w./' + SEP + r'-]+\.(?:zqparams|nchwbin|caffemodel|'
            r'prototxt|txt|bin|dat|jpg|png|idx))["\']', s))):
        base = m.replace('/', SEP).split(SEP)[-1]
        if not (os.path.exists(m)
                or os.path.exists(os.path.join('model', base))
                or os.path.exists(os.path.join('data', base))):
            out.append(base)
    return out


def needs_args(s):
    """真正的判据是 argc 判断，不是 argv[0]。"""
    return bool(re.search(r'\bargc\b\s*(<|>|==|!=|<=|>=)', s)
                or re.search(r'if\s*\(\s*argv\[1\]', s))


def win_only(s):
    return bool(re.search(r'only support windows|not support in linux', s))


def main():
    ran = set()
    for f in ('tools/run_sample_regression.sh', 'tools/run_audit_checks.py'):
        t = io.open(f, encoding='utf-8', errors='replace').read()
        ran |= set(re.findall(r'Sample\w+', t))
    ran = {r[:-4] if r.endswith('.exe') else r for r in ran}

    total = 0
    never = []
    for sample_root in ('SamplesZQCNN', 'SamplesZQlibFaceID',
                        'SamplesZQGEMM', 'SamplesZQBLAS'):
        if not os.path.isdir(sample_root):
            continue
        ds = dirs_of(sample_root)
        names = [n for n, _ in ds]
        if not ds:
            # **平铺目录**（SamplesZQGEMM / SamplesZQBLAS 就是）没有子目录
            print('%s：平铺布局，按源文件数计' % sample_root)
            for f in sorted(glob.glob(os.path.join(ROOT, sample_root, '*'))):
                b = os.path.basename(f)
                if os.path.isfile(f) and b.endswith(('.cpp', '.c')):
                    total += 1
                    mark = '**回归跑**' if b[:-4] in ran else '  回归不跑'
                    if b[:-4] not in ran:
                        never.append((sample_root, b[:-4], '-', '-'))
                    print('  %-36s %s' % (b, mark))
            print()
            continue
        # 断言 3：名字必须非空且唯一
        assert all(names), '%s 下有目录名为空' % sample_root
        assert len(names) == len(set(names)), \
            '%s 下有重名目录：%s' % (sample_root,
                                    [n for n in names if names.count(n) > 1])
        print('=' * 78)
        print('%s：%d 个 sample' % (sample_root, len(ds)))
        print('=' * 78)
        w = max(len(n) for n in names)
        for n, d in ds:
            s = read_src(d)
            miss = missing_assets(s)
            flags = []
            if needs_args(s):
                flags.append('要参数')
            if win_only(s):
                flags.append('WinOnly')
            if n in ran:
                mark = '**回归跑**'
            else:
                mark = '  回归不跑'
                never.append((sample_root, n, ','.join(miss) or '-',
                              ' '.join(flags) or '-'))
            print('  %-*s %-9s %-30s %s'
                  % (w, n, mark, ','.join(miss)[:30] or '-',
                     ' '.join(flags)))
            total += 1
        print()

    print('=' * 78)
    print('合计 %d 个 sample；回归**从不执行**的 %d 个：'
          % (total, len(never)))
    print('=' * 78)
    ready = [x for x in never if x[2] == '-' and x[3] == '-']
    print('其中**静态看**资产齐备、不需参数、非平台专属 = %d 个：'
          % len(ready))
    for r in ready:
        print('   %s/%s' % (r[0], r[1]))
    if ready:
        # 这句话在 2026-10-04 的实测里**被推翻了**：上列 9 个逐个实跑之后，
        # 能加进回归的是 **0 个** —— 4 个是基准（120s 超时）、3 个是平台桩
        # （0 行输出）、2 个要视频设备。
        # 工具**不许**打印一句作者已经证伪的结论：那比不打印更坏，
        # 因为下一个人会拿它当依据。
        print('  注意：这只是**静态**筛选。2026-10-04 逐个实跑之后，'
              '这 %d 个里能加进回归的是 **0 个**' % len(ready))
        print('  （4 个基准 120s 超时 / 3 个平台桩 0 行输出 / 2 个要视频设备）。'
              '详见 audit_k3_20261001.md 附录 GV。')
        print('  "跑起来"不是补覆盖的手段 —— 到今天真正买到覆盖的只有 GS 那一招。')


if __name__ == '__main__':
    main()
