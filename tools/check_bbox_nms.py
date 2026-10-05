#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`ZQ_CNN_BBoxUtils.h` / `ZQ_CNN_Forward_SSEUtils.cpp` 的 NMS 与解码契约门禁 —— 附录 IJ。

为什么要这道门禁
----------------
`ZQ_CNN_BBoxUtils.h`（759 行）此前**零行为门禁**，而它是 MTCNN / CascadeOnet /
SSD / MXNET-SSD **四条检测线共用的**几何底座：`_nms`、`_refine_and_square_bbox`、
`DecodeBBoxes*`、`GetPriorBBoxes`、`JaccardOverlap` 都在这里。
输入是「网络输出 + 模型文件」，两者都不可信。

这一轮（附录 IJ）在里面挖出六条，其中三条是**可证明的**（不需要跑就能从公式推出矛盾）：

  IJ.1  `_detection_output`（SSD 主路径）只判 `len <= 0`，**从不把 `num_priors`
        和三个 blob 的实际长度对账**。而 `num_priors` 来自**另一个张量**（Layer 里
        从 conf 的 H 推出来），Layer 只校验了 loc 的 C 和 conf 的 C，**没校验 prior 的 C**。
        `GetPriorBBoxes` 要读 `8*num_priors` 个 float，`prior_len` 只有 `4*num_priors`
        时就是**堆越界读**。
        判定为笔误的铁证：**同一份文件**的 `_detection_output_MXNET` 早就有完整的
        `num_anchors*4` 对账守卫 —— 一条有、一条没有。

  IJ.2  `_nms` 的 IoU **混用两套面积约定**：交集用「含端点」（`+1`），
        而 `area` 的所有生产点都是「不含 +1」。同一个分母里两套口径，
        于是 IoU 本身不成立：12x12 的框算出 **1.42**（>1）、1x1 退化框算出 **-2**
        （`> threshold` 恒假 -> 永远不被抑制）、3x2 算出分母 **0**（除零 -> +inf -> 误抑制一切）。
        「Min」模式更直接：`IOU / __min(area1, area2)`，零面积框除零得 +inf，
        而 **R-net / O-net 走的就是 Min** -> 一个零面积框抑制掉所有框。
        对照物：同文件 `JaccardOverlap` **内部是自洽的**（交集与 BBoxSize 同一个开关）。

  IJ.4  `it->area = (float)(row2 - row1) * (col2 - col1)` —— 减法在 **int** 里
        先算完再转 float，|row2-row1| > 2^31 就是 signed overflow UB
        （编译器可以假设永不溢出从而删掉后面的检查）。

另外三条是**当前不可达但防护是假的**（判了等于没判），留着是为了「一旦上游变了
不用再查一遍」：

  IJ.3  `_nms` 的 `order`（来自外部传入的 `oriOrder`）只挡 `order < 0`，**不挡上界**；
        而 `ZQ_CNN_OrderScore` 的默认构造是 memset 到 0 —— 「漏填」会**静默指向 0 号框**。
  IJ.5  `DecodeBBoxesAll` 里 `if (find(label) == end()) { /*LOG(FATAL)*/ }` ——
        判了**什么也不做**，下一行照样 `find(label)->second` 解引用 `end()`（UB）。
        这层保护唯一的作用是让读代码的人以为这里被守住了。
        同仓 `ZQ_CNN_Forward_SSEUtils.cpp:5111` 早就是 `continue` 的写法。
  IJ.6  `GetLocPredictions` 的返回值被丢弃；它在 `share_location && num_loc_classes != 1`
        时 return false 且**不 resize**，目前靠下游 `all_loc_preds.size() != num`
        这个**二阶守卫**兜住。

用法
----
    python tools/check_bbox_nms.py              # 扫默认文件
    python tools/check_bbox_nms.py --selfcheck  # 先自测
    python tools/check_bbox_nms.py <文件路径>    # 扫指定文件

自测样本里**必须有故意不合格的项**，且期望写成**规则集合**而不是「红/不红」——
只比红不红的话，一个只缺 A1 的样本因为顺带也缺 A2 会被判成符合预期，
门禁自己糊弄自己。
"""
from __future__ import print_function

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILES = [
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_BBoxUtils.h'),
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_Forward_SSEUtils.cpp'),
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_MTCNN.h'),
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_MTCNN_AspectRatio.h'),
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_MTCNN_Interface.h'),
    os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_MTCNN_NCHWC.h'),
]

# --- BBoxUtils：_nms / area / find-end ---
BBOX_FILE = 'ZQ_CNN_BBoxUtils.h'
# A1: 交集不得用 `+ 1`（那是与 area 不带 +1 混用的那一套）
A1_INTERSECT_PLUS1 = re.compile(r'float\s+w\s*=\s*__max\(\s*minX\s*-\s*maxX\s*\+\s*1\s*,')
# A1': 交集与面积必须同口径 —— 面积就地重算（不直接用 .area 字段做除数）
A1_AREA_INLINE = re.compile(r'float\s+dh1\s*=\s*\(float\)boundingBox\[num\]\.row2')
A1_DENOM_GUARD = re.compile(r'IOU\s*=\s*\(\s*denom\s*>\s*0\s*\)\s*\?')
# A2: area 的 int 减法必须先拓宽
A2_OLD_AREA = re.compile(r'\(\s*float\s*\)\s*\(\s*\w+\s*->\s*row2\s*-\s*\w+\s*->\s*row1\s*\)')
A2_NEW_AREA = re.compile(r'\(\s*float\s*\)\s*\w+\s*->\s*row2\s*-\s*\(\s*float\s*\)\s*\w+\s*->\s*row1\s*\)')
# A3: order 上界
A3_OLD_ORDER = re.compile(r'if\s*\(\s*order\s*<\s*0\s*\)\s*continue\s*;')
A3_NEW_ORDER = re.compile(r'order\s*>=\s*\(\s*int\s*\)\s*boundingBox\.size\(\)')
# A4: find()/end() 的分支体里必须有 continue。
# 第一版用正则 `\{[^{}]*?\}` 匹配分支体 —— 但**合格**的样本里
# 那个体里正好有一个 `continue;`，`}` 后面照样跟着 `find(label)->second`，
# 于是「判了 + continue」和「判了但什么都不做」**两种都匹配**，
# 合格样本被报成不合格。正则分不出这两种，只能把分支体**取出来看**。
A4_IF = re.compile(
    r'find\(label\)\s*==\s*all_loc_preds\[i\]\.end\(\)\s*\)\s*\{')
A4_DEREF = re.compile(r'find\(label\)\s*->\s*second')


def _block_after(text, brace_pos):
    """从 '{' 处开始返回配对花括号**内部**的内容；配不上就返回到文件尾。"""
    i = text.find('{', brace_pos)
    if i < 0:
        return '', i
    depth = 0
    j = i
    n = len(text)
    while j < n:
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j
        j += 1
    return text[i + 1:], n

# --- Forward_SSEUtils：detection_output 的长度对账 + GetLocPredictions 返回值 ---
FWD_FILE = 'ZQ_CNN_Forward_SSEUtils.cpp'
# A5: _detection_output（主路径）必须按 num_priors 对账，prior 要 *2（bbox + variance）
# 守卫的形态是「`num_priors <= 0` 就拒」，不是 `> 0`：
# 一开始写成找 `num_priors > 0`，而源码里从来没有这种写法，
# 于是这条判据**恒假**、等于没判 —— 「扫不到」和「没问题」在报告里长得一样。
A5_NUM_PRIORS = re.compile(r'num_priors\s*<=\s*0')
A5_LOC = re.compile(r'loc_len\s*<\s*\(\s*long long\s*\)\s*num\s*\*\s*needed_anchor\s*\*\s*num_loc_classes')
A5_CONF = re.compile(r'conf_len\s*<\s*\(\s*long long\s*\)\s*num\s*\*\s*num_priors\s*\*\s*num_classes')
A5_PRIOR = re.compile(r'prior_len\s*<\s*needed_anchor\s*\*\s*2LL')
# A6: GetLocPredictions 返回值必须检查
# **必须带 re.M**：`^` 不加 MULTILINE 只匹配整个字符串的开头，
# 而调用点在文件中间 —— 不加的话这条判据**恒假**、等于没判。
# 「扫不到」和「没问题」在报告里长得一模一样，这是第三次栽在这上面。
A6_UNCHECKED = re.compile(r'^\s*ZQ_CNN_BBoxUtils::GetLocPredictions\(', re.M)
A6_CHECKED = re.compile(r'if\s*\(\s*!\s*ZQ_CNN_BBoxUtils::GetLocPredictions\(')


def _read(path):
    with io.open(path, 'r', encoding='utf-8') as f:
        return f.read()


def strip_comments(text):
    """把 C++ 注释替换成等长空格（保留换行，行号才不变）。"""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == '\\' else 1
            j = min(j + 1, n)
            out.append(text[i:j])
            i = j
            continue
        if c == '/' and i + 1 < n:
            nxt = text[i + 1]
            if nxt == '/':
                j = text.find('\n', i)
                j = n if j < 0 else j
                out.append(' ' * (j - i))
                i = j
                continue
            if nxt == '*':
                j = text.find('*/', i + 2)
                j = n if j < 0 else j + 2
                out.append(''.join(ch if ch == '\n' else ' ' for ch in text[i:j]))
                i = j
                continue
        out.append(c)
        i += 1
    return ''.join(out)


def scan_text(raw, name):
    text = strip_comments(raw)
    bad = []
    ok = 0

    if name == BBOX_FILE:
        # 每条规则都带**前置条件**（该构造在本文件里到底存不存在）。
        # 不加前置的后果：第一版把 A1/A2/A3/A4 都写成无条件判定，
        # 于是「只含 _nms 的自测样本」被判成「也缺 find/end 守卫」——
        # **「不适用」被当成了「不满足」**。与附录 A31 同一个教训。
        has_nms = bool(re.search(r'float\s+w\s*=\s*__max\(\s*minX', text))
        has_area = bool(re.search(r'\w+\s*->\s*area\s*=', text))
        has_order = bool(re.search(r'order\s*=\s*bboxScore\.back\(\)\.oriOrder', text))
        has_find = bool(re.search(r'find\(label\)\s*==\s*all_loc_preds\[i\]\.end\(\)', text))

        # A1
        if not has_nms:
            ok += 1
        elif A1_INTERSECT_PLUS1.search(text):
            bad.append(('A1', '`_nms` 的交集还是 `__max(minX - maxX + 1, 0)`（含端点口径），'
                        '而 area 的所有生产点都是不含 +1 的 —— 同一个分母里两套口径，'
                        'IoU 可 >1（12x12 算出 1.42）、可为负（1x1 算出 -2 -> 永不抑制）、'
                        '分母可为 0（3x2 -> 除零 -> +inf -> 误抑制一切）'))
        elif not (A1_AREA_INLINE.search(text) and A1_DENOM_GUARD.search(text)):
            bad.append(('A1', '`_nms` 的面积没有就地重算 / 分母没有兜底 —— '
                        '调用方传的 `area` 可能为 0 或负，而它要拿来做除数；'
                        '「Min」模式下零面积框会除零得 +inf，抑制掉所有框'))
        else:
            ok += 1
        # A2
        if not has_area:
            ok += 1
        elif A2_OLD_AREA.search(text):
            bad.append(('A2', '还有 `(float)(row2 - row1)` 形态：减法在 **int** 里先算完再转 '
                        'float，|row2-row1| > 2^31 是 signed overflow UB '
                        '（编译器可假设永不溢出从而删掉后面的检查）'))
        else:
            ok += 1
        # A3
        if not has_order:
            ok += 1
        elif A3_OLD_ORDER.search(text):
            bad.append(('A3', '`_nms` 的 `order` 还是只挡 `order < 0`，不挡上界 —— '
                        '越界的 oriOrder 会让 `boundingBox[order].exist = false` 越界**写**、'
                        '`boundingBox[order].col1` 越界**读**。'
                        '而 ZQ_CNN_OrderScore 默认构造 memset 到 0，'
                        '「漏填」会**静默指向 0 号框**'))
        elif not A3_NEW_ORDER.search(text):
            bad.append(('A3', '找不到 `order >= (int)boundingBox.size()` 的上界守卫'))
        else:
            ok += 1
        # A4
        if not has_find:
            ok += 1
        else:
            m4 = A4_IF.search(text)
            body, close = _block_after(text, m4.end() - 1) if m4 else ('', 0)
            tail = text[close:close + 400] if m4 else ''
            if m4 and 'continue' not in body and A4_DEREF.search(tail):
                bad.append(('A4', '`DecodeBBoxesAll` 里 `if (find(label) == end()) { ... }` '
                            '判了之后**什么都不做**，下一行照样 `find(label)->second` '
                            '解引用 `end()`（UB）—— 这层保护是假的。应为 `continue`'))
            elif m4 and 'continue' not in body:
                bad.append(('A4', 'find/end 的分支体里没有 continue，形态不认识'))
            else:
                ok += 1
        return ok, bad

    if name == FWD_FILE:
        # A5 / A6 只在 _detection_output 家族里判
        if '_detection_output' in raw:
            miss = []
            if not A5_NUM_PRIORS.search(text):
                miss.append('没有 num_priors > 0')
            if not A5_LOC.search(text):
                miss.append('loc_len 没有按 num*num_priors*4*num_loc_classes 对账')
            if not A5_CONF.search(text):
                miss.append('conf_len 没有按 num*num_priors*num_classes 对账')
            if not A5_PRIOR.search(text):
                miss.append('prior_len 没有按 num_priors*4*2（bbox+variance 两半）对账')
            if miss:
                bad.append(('A5', '`_detection_output` 的 blob 长度对账不完整：'
                            + '；'.join(miss)
                            + ' —— num_priors 来自 Layer 从 conf 的 H 推出来的另一个张量，'
                              '而 Layer 只校验了 loc 的 C 和 conf 的 C，没校验 prior 的 C。'
                              'GetPriorBBoxes 要读 8*num_priors，prior_len 只有 4*num_priors '
                              '时就是堆越界**读**。MXNET 那条路径早就有完整守卫'))
            else:
                ok += 1
        if A6_UNCHECKED.search(text):
            bad.append(('A6', '`GetLocPredictions` 的返回值被丢弃 —— 它在 '
                        '`share_location && num_loc_classes != 1` 时 return false 且'
                        '**不 resize**，目前靠下游 `all_loc_preds.size() != num` 这个'
                        '**二阶守卫**兜住'))
        else:
            ok += 1
        return ok, bad

    return ok, bad


# 一份"什么都齐"的骨架，作为后面每条自测样本的公共前缀。
FULL_NMS = r"""
static void _nms(std::vector<ZQ_CNN_BBox>& boundingBox, std::vector<ZQ_CNN_OrderScore>& bboxScore,
	float overlap_threshold, const char* modelname, int thread_num)
{
	while (bboxScore.size() > 0)
	{
		int order = bboxScore.back().oriOrder;
		bboxScore.pop_back();
		if (order < 0 || order >= (int)boundingBox.size()) continue;
		for (int num = 0; num < (int)boundingBox.size(); num++)
		{
			float maxX = (float)__max(boundingBox[num].col1, boundingBox[order].col1);
			float minX = (float)__min(boundingBox[num].col2, boundingBox[order].col2);
			float w = __max(minX - maxX, 0);
			float h = __max(minY - maxY, 0);
			float IOU = w * h;
			float dh1 = (float)boundingBox[num].row2 - (float)boundingBox[num].row1;
			float dw1 = (float)boundingBox[num].col2 - (float)boundingBox[num].col1;
			float dh2 = (float)boundingBox[order].row2 - (float)boundingBox[order].row1;
			float dw2 = (float)boundingBox[order].col2 - (float)boundingBox[order].col1;
			if (dh1 < 0) dh1 = 0;
			if (dw1 < 0) dw1 = 0;
			if (dh2 < 0) dh2 = 0;
			if (dw2 < 0) dw2 = 0;
			float area1 = dh1 * dw1;
			float area2 = dh2 * dw2;
			if (!modelname.compare("Union"))
			{
				float denom = area1 + area2 - IOU;
				IOU = (denom > 0) ? (IOU / denom) : 0;
			}
			else if (!modelname.compare("Min"))
			{
				float denom = __min(area1, area2);
				IOU = (denom > 0) ? (IOU / denom) : 0;
			}
		}
	}
	it->area = ((float)it->row2 - (float)it->row1) * ((float)it->col2 - (float)it->col1);
}
"""

FULL_DECODE = r"""
static bool DecodeBBoxesAll(const std::vector<ZQ_CNN_LabelBBox>& all_loc_preds)
{
	if (all_loc_preds[i].find(label) == all_loc_preds[i].end())
	{
		continue;
	}
	const std::vector<ZQ_CNN_NormalizedBBox>& label_loc_preds = all_loc_preds[i].find(label)->second;
	return true;
}
"""

FULL_FWD = r"""
bool ZQ_CNN_Forward_SSEUtils::_detection_output(const ZQ_CNN_Tensor4D& loc, const ZQ_CNN_Tensor4D& conf,
	const ZQ_CNN_Tensor4D& prior, int num_priors, int num_loc_classes, int num_classes, bool share_location)
{
	const int num = loc.GetN();
	int loc_len = loc.GetN() * loc.GetH() * loc.GetW() * loc.GetC();
	int conf_len = conf.GetN() * conf.GetH() * conf.GetW() * conf.GetC();
	int prior_len = prior.GetN() * prior.GetH() * prior.GetW() * prior.GetC();
	if (loc_len <= 0 || conf_len <= 0 || prior_len <= 0)
		return false;
	{
		const long long needed_anchor = (long long)num_priors * 4LL;
		if (num_priors <= 0 || num_classes <= 0 || num_loc_classes <= 0
			|| (long long)loc_len < (long long)num * needed_anchor * num_loc_classes
			|| (long long)conf_len < (long long)num * num_priors * num_classes
			|| (long long)prior_len < needed_anchor * 2LL)
		{
			return false;
		}
	}
	if (!ZQ_CNN_BBoxUtils::GetLocPredictions(&loc_data[0], num, num_priors, num_loc_classes, share_location, &all_loc_preds))
		return false;
	return true;
}
"""


SELFCHECK = [
    # (说明, 文本, 期望触发的规则集合, 扫哪个文件)
    ('全合格 NMS', FULL_NMS, [], BBOX_FILE),
    ('A1 交集回到 +1 口径',
     FULL_NMS.replace('float w = __max(minX - maxX, 0);',
                      'float w = __max(minX - maxX + 1, 0);'), ['A1'], BBOX_FILE),
    ('A1 面积不重算 + 分母不兜底',
     FULL_NMS.replace('float dh1 = (float)boundingBox[num].row2 - (float)boundingBox[num].row1;', '')
               .replace('float dw1 = (float)boundingBox[num].col2 - (float)boundingBox[num].col1;', '')
               .replace('float dh2 = (float)boundingBox[order].row2 - (float)boundingBox[order].row1;', '')
               .replace('float dw2 = (float)boundingBox[order].col2 - (float)boundingBox[order].col1;', '')
               .replace('IOU = (denom > 0) ? (IOU / denom) : 0;', 'IOU = IOU / denom;'),
     ['A1'], BBOX_FILE),
    ('A2 area 又变回 int 减法',
     FULL_NMS.replace('it->area = ((float)it->row2 - (float)it->row1) * ((float)it->col2 - (float)it->col1);',
                      'it->area = (float)(it->row2 - it->row1)*(it->col2 - it->col1);'),
     ['A2'], BBOX_FILE),
    ('A3 order 回到只挡下界',
     FULL_NMS.replace('if (order < 0 || order >= (int)boundingBox.size()) continue;',
                      'if (order < 0)continue;'), ['A3'], BBOX_FILE),
    ('全合格 DecodeBBoxesAll', FULL_DECODE, [], BBOX_FILE),
    ('A4 find/end 判了不做',
     FULL_DECODE.replace('\t\tcontinue;\n', '\t\t//LOG(FATAL) << label;\n'),
     ['A4'], BBOX_FILE),
    ('全合格 _detection_output', FULL_FWD, [], FWD_FILE),
    ('A5 prior 只对账一半（漏 *2）',
     FULL_FWD.replace('(long long)prior_len < needed_anchor * 2LL',
                      '(long long)prior_len < needed_anchor'), ['A5'], FWD_FILE),
    ('A5 loc 不对账',
     FULL_FWD.replace('|| (long long)loc_len < (long long)num * needed_anchor * num_loc_classes\n', ''),
     ['A5'], FWD_FILE),
    ('A6 GetLocPredictions 返回值不查',
     FULL_FWD.replace('if (!ZQ_CNN_BBoxUtils::GetLocPredictions(',
                      'ZQ_CNN_BBoxUtils::GetLocPredictions(')
              .replace(' share_location, &all_loc_preds))\n\t\treturn false;',
                      ' share_location, &all_loc_preds);'),
     ['A6'], FWD_FILE),
    # 阴性：注释里出现这些词**不该**报
    ('阴性：只有注释提到 + 1 / order < 0',
     '// 交集原来是 __max(minX - maxX + 1, 0)，area 用的是不含 +1 的口径\n'
     '// if (order < 0)continue;\n', [], BBOX_FILE),
]


def selfcheck():
    bad_cnt = 0
    for name, text, expect, target in SELFCHECK:
        _, bad = scan_text(text, target)
        got = set(c for c, _ in bad)
        if got != set(expect):
            bad_cnt += 1
            print('  [self-MISMATCH] %s' % name)
            print('      expect %s, got %s' % (sorted(expect) or '[]', sorted(got) or '[]'))
            for c, msg in bad:
                print('        %s: %s' % (c, msg))
        else:
            print('  [self-OK]      %-40s %s'
                  % (name, ','.join(sorted(got)) if got else 'clean'))
    if bad_cnt:
        print('selfcheck FAILED: %d / %d mismatch' % (bad_cnt, len(SELFCHECK)))
        return 1
    print('selfcheck OK: %d cases, all as expected' % len(SELFCHECK))
    return 0


def main(argv):
    if '--selfcheck' in argv:
        return selfcheck()
    files = [a for a in argv[1:] if not a.startswith('-')] or DEFAULT_FILES
    total_ok = 0
    problems = []
    for p in files:
        if not os.path.exists(p):
            print('SKIP %s (not found)' % p)
            continue
        name = os.path.basename(p)
        ok, bad = scan_text(_read(p), name)
        total_ok += ok
        if bad:
            print('FAIL %s' % name)
            for code, msg in bad:
                print('       - %s: %s' % (code, msg))
            problems.append(name)
        else:
            print('OK   %s' % name)
    print('合计合格判定 %d 项' % total_ok)
    if problems:
        print('**%d 个文件不合格**' % len(problems))
        return 1
    print('BBoxUtils NMS / 解码契约: OK')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
