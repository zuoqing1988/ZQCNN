#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扫「类成员没在构造函数初始化列表里」——也就是**未初始化成员**。

为什么要有这个工具（audit_k3_20261001.md 附录 AU）
-------------------------------------------------
`-Wall -Wextra` 抓不到这一类：gcc 不会对「类成员没在初始化列表里」发警告
（`-Wmissing-field-initializers` 只管聚合初始化，不管构造函数）。
本项目 43 个 TU 的 HIGH 桶里 0 条属于这一类，但它在主工程里**真实存在**：

    class ZQ_CNN_Layer {
        void**    buffer;      // <-- 构造函数里没有它
        __int64*  buffer_len;  // <-- 也没有
        bool      use_buffer;  // 有

        ZQ_CNN_Layer() :show_debug_info(false),use_buffer(false),... {}
        ...
        void** tmp_buffer = use_buffer ? buffer : 0;   // 被读了
    };

`use_buffer` 全仓**从来没有被赋成 true**（只在构造函数里赋 false），
所以这条路径今天是死代码，`buffer` 是垃圾指针也没事。
但它的形状和附录 AT.4 那个「`&ot == NULL` 守卫」一模一样：
**读一个没初始化的指针，安全性完全靠另一个从不改变的开关撑着。**
哪天有人为了「开启 buffer 复用」把 `use_buffer` 设成 true，就是野指针写。

本工具按「声明了但没出现在初始化列表里」逐个报出来。
**它不会判断某个成员是否「设计上就该由别处赋值」** ——
ZQCNN 里大量成员是 `LoadParam` 阶段填的。所以输出是**筛子**，
每一条都要人眼确认，判据是：「有没有一条从构造到使用之间必然被赋值的路径」。

用法:
    python tools/check_uninit_members.py                 # 扫 ZQCNN/ZQ_CNN_Layer.h
    python tools/check_uninit_members.py --file <path>
    python tools/check_uninit_members.py --all          # 扫 ZQCNN/ 下所有 .h
    python tools/check_uninit_members.py --json
"""

from __future__ import print_function

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 成员声明：行首（可有缩进）一个类型 + 名字 + 可选的 [] + 可选的 = 初值 + ;
# 刻意保守：只认「一行一个成员」这种最常见的写法，不做完整 C++ 解析。
#
# `= 初值` 那一段**必须是可选的**。第一版把它写成了必需，于是它只匹配
# 「带默认初始化器」的成员 —— 而我们要找的恰恰是**不带**初始化的那些。
# 结果是：指针成员（`void** buffer;` 恰好落在另一条 `[*&]+` 分支上）能报出来，
# 标量成员一条都报不出来，而自测里 `int num;` / `int q;` 正好各漏一条。
MEMBER_RE = re.compile(
    r'^\s*(?P<type>(?:const\s+|static\s+|unsigned\s+|signed\s+|struct\s+|enum\s+)*'
    r'[A-Za-z_][\w:]*(?:\s*<[^;]*?>)?(?:\s*[*&])*)\s+'
    r'(?P<name>[A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*'
    r'(?:=\s*[^;]+)?;\s*$')

# 类 / 结构体。**只匹配到类名**，不要求 '{' 在同一行（见 parse_classes 的注释）
CLASS_RE = re.compile(r'^\s*(?:class|struct)\s+(?P<name>[A-Za-z_]\w*)\b')

CTOR_RE = re.compile(r'^\s*(?P<name>[A-Za-z_]\w*)\s*\([^)]*\)\s*(:|{)')

# 只关心这几类：指针（野指针风险最高）、内置整型/浮点（值不确定）
RISKY = ('*', '&')
PRIMITIVE = {'int', 'unsigned', 'long', 'short', 'char', 'float', 'double',
             'bool', '__int64', '__int32', 'unsigned int', 'unsigned char',
             'size_t', 'size_t*'}

# 这些名字不是数据成员
NOT_MEMBERS = {'return', 'if', 'for', 'while', 'switch', 'delete', 'typedef',
               'operator', 'using', 'template'}

_LINE_COMMENT_RE = re.compile(r'//.*$')


def strip_comment(ln):
    """去掉行尾 // 注释（不算字符串字面量里的 //，本文件里没有那种写法）。

    不做这一步，MEMBER_RE 末尾的 `\\s*;\\s*$` 在**带行尾注释**的成员上一条都
    匹配不上 —— 而带注释恰恰是这个项目最常见的写法，于是自测 4 漏 0，
    真实文件里那些没注释的成员却能报出来，工具看起来"时灵时不灵"。
    """
    return _LINE_COMMENT_RE.sub('', ln)


def parse_classes(text):
    """返回 [(class_name, start_line, end_line, body_lines)]，按大括号配平。"""
    lines = text.split('\n')
    out = []
    i = 0
    while i < len(lines):
        m = CLASS_RE.match(lines[i])
        if not m:
            i += 1
            continue
        # 找到 class 体的起始 '{'。**不能要求它和 class 同一行** ——
        # 本项目（和绝大多数 C++ 代码）都把 '{' 放在下一行：
        #     class ZQ_CNN_Layer
        #     {
        # 第一版 CLASS_RE 写成 `...[^;{]*\{` 单行匹配，于是**一个类都没解析出来**，
        # 工具报"没有命中"、退出码 0 —— 又一个"看起来很绿"的哑巴工具。
        depth = 0
        started = False
        j = i
        while j < len(lines):
            for ch in lines[j]:
                if ch == '{':
                    depth += 1
                    started = True
                elif ch == '}':
                    depth -= 1
            if started and depth == 0:
                break
            j += 1
        out.append((m.group('name'), i + 1, j + 1, lines[i:j + 1]))
        i = j + 1
    return out


def ctor_init_members(body_lines, class_name):
    """找出类内所有构造函数的初始化列表里出现的成员名集合。"""
    names = set()
    in_ctor = False
    depth = 0
    for ln in body_lines:
        stripped = ln.strip()
        if stripped.startswith('*') or stripped.startswith('//'):
            continue
        m = CTOR_RE.match(ln)
        if m and m.group('name') == class_name:
            in_ctor = True
            depth = 0
        if in_ctor:
            for ch in ln:
                if ch in '({':
                    depth += 1
                elif ch in ')}':
                    depth -= 1
            # 初始化列表里的 name(
            for fm in re.finditer(r'([A-Za-z_]\w*)\s*\(', ln):
                names.add(fm.group(1))
            if '{' in ln and depth <= 1 and ':' not in ln.split('{')[0]:
                in_ctor = False
    return names


def scan_text(text, path):
    findings = []
    for cname, start, end, body in parse_classes(text):
        inited = ctor_init_members(body, cname)
        # 判据是「**这个成员在整个类体里从来没有被赋过值**」，
        # 所以要在**全类体**（含所有成员函数体）里找 `name =` / `name = {`。
        # 第一版只认「同一行写成 `int x = 0;`」这种默认初始化器，于是
        # `void setup(){ assigned_later = 0; }` 里的赋值看不见，
        # 自测里就多出一条误报。而 ZQCNN 里绝大多数成员正是在 Set()/Forward()
        # 里赋值的 —— 只认默认初始化器的话这个工具会报出几百条噪声。
        assigned_anywhere = set()
        for ln in body:
            for am in re.finditer(r'\b([A-Za-z_]\w*)\s*=(?!=)', strip_comment(ln)):
                assigned_anywhere.add(am.group(1))
        # **只看类的数据成员区（花括号深度 1）**。
        # 第一版没做这件事，于是把成员函数体里的局部变量也当成成员报出来：
        # `double t1; bool ret; int num;` 一口气几百条，工具立刻变成噪声源
        # —— 而"一个天天误报的检查工具等于没有工具"这条，本轮已经踩过一次
        # （check_alloc_delete.py 的 delete[]values[i]）。
        depth = 0
        for idx, ln in enumerate(body):
            if ln.strip().startswith('*') or ln.strip().startswith('//'):
                continue
            opens = ln.count('{')
            closes = ln.count('}')
            at_member_depth = (depth == 1)
            depth += opens - closes
            if depth < 1:
                continue
            if not at_member_depth:
                continue
            s = ln.strip()
            if s.startswith(('public', 'private', 'protected', 'friend',
                             'typedef', 'using', 'enum', 'struct', 'class',
                             'union', 'template', 'return', 'virtual',
                             'static_assert', 'operator')):
                continue
            # `static const int TYPE_NONE = 0;` 是类常量，不是每实例存储。
            # 不排除它的话会刷出一堆 `static const int` 的假命中
            # （本项目有 40 多条 TYPE_*/ELTWISE_*/... ），而第一版的
            # 「类体内 = 0 就算初始化」那条正则只认**一个**类型词，
            # 认不出 `static const int` 这三个词，于是全漏。
            if re.search(r'\bstatic\b', ln):
                continue
            m = MEMBER_RE.match(strip_comment(ln))
            if not m:
                continue
            name = m.group('name')
            typ = (m.group('type') or '').strip()
            if name in NOT_MEMBERS or name in inited or name in assigned_anywhere:
                continue
            base = typ.replace('const', '').replace('static', '').strip()
            # `std::vector<Tensor4D*>` 里的 '*' 在尖括号中间，不是指针成员；
            # 而且 std:: 容器是**默认构造**的，本来就不需要初始化。
            # 不排除的话 ZQ_CNN_Net::blobs / ZQ_CNN_Net_NCHWC::blobs 会各报一条。
            if 'std::' in typ:
                continue
            is_ptr = any(c in typ for c in RISKY)
            is_prim = base.replace('unsigned', '').replace('signed', '').strip() \
                in {p.replace('unsigned', '').replace('signed', '').strip()
                    for p in PRIMITIVE}
            if not (is_ptr or is_prim):
                continue
            findings.append({
                'file': path, 'class': cname, 'line': start + idx,
                'member': name, 'type': typ,
                'risk': '指针' if is_ptr else '标量',
            })
    return findings


# 内建自测。见 AGENTS.md「sanitizer 与检查工具的四个坑」第 2 条：
# 一个「什么都查不出来」的检查工具比没有这个工具更危险。
# 期望命中 A1(buffer) / A3(t2) / B1(p) 三条；
# **不得**命中 A2(在初始化列表里) / A4(类体内 =0) / A5(成员函数里的局部变量)
# / A6(static const) / A7(函数参数) / A8(非指针非标量的类型)。
# 注意**不能**用 r'''...'''：r 前缀会让源码里的 \t 保持成「反斜杠 + t」两个字符，
# 于是 ^\s* 匹配不上，工具对自测样本一条都报不出来 —— 而自测的意义就是
# 防住这种事。2026-10-02 第一版就是这么写的，自测 4 漏 0。
SELFTEST_SRC = '''
class A {
public:
	void**   buffer;          // 应命中：指针，构造函数没提到它
	int      num;             // 应命中：标量
	int      len;             // 不应命中：在初始化列表里
	int      zeroed = 0;      // 不应命中：默认成员初始化器
	int      assigned_later;  // 不应命中：类体内有 zero = 0 那样的赋值
	static const int KIND = 0;// 不应命中：类常量
	A() :len(0) {}
	void setup() { assigned_later = 0; }
	void f() { int local; double t1; local = 1; }
};

class B {
public:
	float*   p;               // 应命中
	int      q;               // 应命中
};

void g(int arg) { int not_a_member; }
'''
SELFTEST_EXPECT = {('A', 'buffer'), ('A', 'num'), ('B', 'p'), ('B', 'q')}


def selftest():
    got = set((f['class'], f['member'])
              for f in scan_text(SELFTEST_SRC, '<selftest>'))
    missing = SELFTEST_EXPECT - got
    extra = got - SELFTEST_EXPECT
    print('=' * 74)
    print('check_uninit_members.py 自测')
    print('=' * 74)
    for k in sorted(SELFTEST_EXPECT):
        print('   %-24s %s' % ('%s::%s' % k, '命中' if k in got else '**漏了**'))
    for k in sorted(extra):
        print('   %-24s **误报**' % ('%s::%s' % k))
    if missing or extra:
        print('\n自测失败：漏 %d 条，误报 %d 条 —— 这个工具现在**不可信**。'
              % (len(missing), len(extra)))
        return 1
    print('\n自测通过：4 条应命中全部命中，0 条误报。')
    return 0


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    argv = list(sys.argv[1:])
    as_json = '--json' in argv
    scan_all = '--all' in argv
    want_selftest = '--selftest' in argv
    argv = [a for a in argv if not a.startswith('--')]
    if want_selftest:
        return selftest()
    target = None
    if '--file' in argv:
        i = argv.index('--file')
        target = argv[i + 1]
        del argv[i:i + 2]

    files = []
    if target:
        files = [target]
    elif scan_all:
        d = os.path.join(ROOT, 'ZQCNN')
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.h'):
                files.append(os.path.join(d, fn))
    else:
        files = [os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_Layer.h')]

    all_findings = []
    for f in files:
        with open(f, 'r', encoding='utf-8', errors='replace') as fh:
            all_findings += scan_text(fh.read(), os.path.relpath(f, ROOT))

    if as_json:
        print(json.dumps(all_findings, ensure_ascii=False, indent=2))
        return 0

    print('=' * 74)
    print('未初始化类成员扫描：%d 个文件' % len(files))
    print('=' * 74)
    if not all_findings:
        print('没有命中。')
    else:
        by_cls = {}
        for x in all_findings:
            by_cls.setdefault((x['file'], x['class']), []).append(x)
        for (f, c), items in sorted(by_cls.items()):
            ptrs = [i for i in items if i['risk'] == '指针']
            print('\n%s  class %s   (%d 个：指针 %d / 标量 %d)'
                  % (f, c, len(items), len(ptrs), len(items) - len(ptrs)))
            for i in items:
                print('   :%-6d %-24s %-8s %s'
                      % (i['line'], i['type'], i['risk'], i['member']))
    print('\n注意：本工具是**筛子**。ZQCNN 里大量成员是 LoadParam/Forward 阶段填的，')
    print('「没在初始化列表里」不等于「会用未初始化值」。逐条判据：')
    print('  从构造到第一次使用之间，**有没有一条必然给它赋值的路径**。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
