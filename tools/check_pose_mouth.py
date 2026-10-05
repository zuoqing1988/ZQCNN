#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""姿态/嘴部/人脸裁剪四个头的门禁 —— 附录 IN。

范围
----
- `ZQCNN/ZQ_CNN_PersonPose.h`   （单人姿态）
- `ZQCNN/ZQ_CNN_PersonPose2.h`  （半/全模式，**与上面是逐字拷贝**）
- `ZQCNN/ZQ_CNN_MouthDetector.h`（嘴部检测，用 MTCNN + SSD）
- `ZQCNN/ZQ_CNN_FaceCropUtils.h`（人脸裁剪几何；**注意这个文件是 GBK 编码**，
  是上游 MFC 中文界面带来的，`check_text_encoding.py` 里有白名单，
  注释里明确写着「改成 UTF-8 会破坏 Windows 侧的中文界面，不要动」）

为什么需要它
------------
这四个头此前**零行为门禁**。本轮（附录 IN）挖出 9 条，其中六条是
**两个拷贝之间的差异** —— 也就是说「单看一个文件是否合规」这种判据会漏：

  IN.1  MouthDetector 的 real_border_x/y 可为**负**（因为 MTCNN 的边界钳位
        是注释掉的，脸贴边是常规输入）-> `cv::Mat(image, rect)` 抛未捕获异常
  IN.2  FaceCropUtils 的 `fill_val` 形参被接受后**丢弃**，硬编码成 0；
        同文件 :61 的另一个重载传的是 `fill_val` —— 一份对一份错
  IN.3  PersonPose.h 的 `points[54]` 只 memset 了 **51** 个 float
        （差第 17 个关键点）；PersonPose2.h 是 42/42 正确
  IN.4  两个头的成员 int 全部**未初始化**，而 Init 的 6 个 return false 都在赋值之前
  IN.5  姿态侧 `pose_ptr = GetBlobByName(...)` 没判空，而**紧邻的 SSD 侧判了**
  IN.6  PersonPose.h 的 `npts` 取自**调用方可控**的 public 字段 `num_points`，
        npts==0 时守卫恒假 -> 产出 `col1=1e9 > col2=-1e9` 的反向框；
        PersonPose2.h 的 npts 由 half_mode 推导，**永远 >= 1**
  IN.7  PersonPose2.h 的 `MapToFull` 跳过 4 个位置，其中 `full[9]` 保留了
        半模式的 `half[9]` -> 紧接着的「没检到脚踝就扩框」分支**永远走不到**
  IN.8  四处 `ConvertFromBGR` + `ResizeBilinear` 返回值被丢弃
        （同文件 :107/:112/:116 对同样的调用**全都检查了**）
  IN.9  PersonPose2.h 的 `size_H*size_W*3` 是**纯 int 算术**，缺
        `(__int64)` + `> 0x7FFFFFFF` 守卫；PersonPose.h 两处都有

所以判据分两类：
  * **单文件判**（IN.1/IN.2/IN.3/IN.4/IN.5/IN.7/IN.8/IN.9）—— 只看这一个文件里
    该有的形态在不在；
  * **两拷贝对照判**（IN.3/IN.6/IN.9）—— 判据写成「两份都要有」，
    否则 PersonPose.h 会因为**已经有**守卫而"通过"，掩盖 PersonPose2.h 的缺失。
"""
from __future__ import print_function

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1 = os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_PersonPose.h')
P2 = os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_PersonPose2.h')
MD = os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_MouthDetector.h')
FC = os.path.join(ROOT, 'ZQCNN', 'ZQ_CNN_FaceCropUtils.h')


def read(path, enc='utf-8'):
    with io.open(path, 'r', encoding=enc) as f:
        return f.read()


def strip_comments(text):
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
            nx = text[i + 1]
            if nx == '/':
                j = text.find('\n', i)
                j = n if j < 0 else j
                out.append(' ' * (j - i))
                i = j
                continue
            if nx == '*':
                j = text.find('*/', i + 2)
                j = n if j < 0 else j + 2
                out.append(''.join(ch if ch == '\n' else ' ' for ch in text[i:j]))
                i = j
                continue
        out.append(c)
        i += 1
    return ''.join(out)


# ------------------------------------------------------------------ 判据
def _p1(t1):
    """PersonPose.h：IN.3 / IN.4 / IN.5 / IN.6 / IN.8 / IN.9 的这一份。"""
    bad, ok = [], 0
    # IN.3: memset 必须写 sizeof(points)，不能写死 51
    if re.search(r'memset\s*\(\s*points\s*,\s*0\s*,\s*sizeof\s*\(\s*float\s*\)\s*\*\s*\d+', t1):
        bad.append(('IN.3', '`points` 的 memset 仍写死元素个数 —— 数组是 float[54]，'
                    '写 51 就漏了第 17 个关键点的 points[51..53]，'
                    '而 `num_points` 仍会告诉调用方「有 18 个点」。'
                    '应写 `sizeof(points)`，以后加字段也不会再错'))
    elif not re.search(r'memset\s*\(\s*points\s*,\s*0\s*,\s*sizeof\s*\(\s*points\s*\)\s*\)', t1):
        bad.append(('IN.3', '找不到 `memset(points, 0, sizeof(points));`'))
    else:
        ok += 1
    # IN.6: npts <= 0 守卫
    if not re.search(r'if\s*\(\s*npts\s*<=\s*0\s*\)\s*continue\s*;', t1):
        bad.append(('IN.6', '`npts` 取自**调用方可控**的 public 字段 `num_points`；'
                    'npts==0 时下面两个守卫与 0 比**恒假** -> 不 erase -> '
                    '直接算出 col1=1e9 > col2=-1e9 的反向框，'
                    '下一帧负宽负高进 ConvertFromBBR。需要 `if (npts <= 0) continue;`'))
    else:
        ok += 1
    # IN.9: 溢出守卫（两份都要有）
    n_ovf = len(re.findall(r'const\s+__int64\s+buffer_size\s*=\s*\(__int64\)\s*size_H', t1))
    if n_ovf < 2:
        bad.append(('IN.9', '`size_H*size_W*3` 的 `(__int64)` + `> 0x7FFFFFFF` 守卫只有 %d 处，'
                    '两个函数各要一处' % n_ovf))
    else:
        ok += 1
    # IN.8: ConvertFromBGR / ResizeBilinear 判空
    n_c = len(re.findall(r'if\s*\(\s*!\s*temp_img\.ConvertFromBGR\(', t1))
    n_r = len(re.findall(r'if\s*\(\s*!\s*temp_img\.ResizeBilinear\(', t1))
    if n_c < 2 or n_r < 2:
        bad.append(('IN.8', 'ConvertFromBGR 判空 %d 处 / ResizeBilinear 判空 %d 处，'
                    '两个函数各要一对（失败时 temp_img 停在**上一次**的尺寸，'
                    '紧接着的 ResizeBilinear 就按陈旧尺寸跑）' % (n_c, n_r)))
    else:
        ok += 1
    # IN.5: pose_ptr 判空
    n = len(re.findall(r'pose_ptr\s*=\s*pose_net\.GetBlobByName\(', t1))
    g = len(re.findall(r'if\s*\(\s*pose_ptr\s*==\s*0\s*\)', t1))
    if n and g < n:
        bad.append(('IN.5', '姿态侧 %d 处 `GetBlobByName` 只有 %d 处判空 —— '
                    '紧邻的 SSD 侧判了，姿态侧没有，是同文件内的不对称' % (n, g)))
    else:
        ok += 1
    # IN.4: 成员 int 初值
    for name in ('ssd_C', 'pose_C', 'pose_npts'):
        m = re.search(r'\bint\s+[A-Za-z0-9_,\s]*\b' + name + r'\b[^;]*;', t1)
        if m and '=' not in m.group(0):
            bad.append(('IN.4', '成员 `%s` 没有初值 —— Init 的 6 个 return false 都发生在 '
                        '`GetInputDim` 赋值之前，调用方忽略 Init 返回值时 Detect '
                        '上来就读它' % name))
            break
    else:
        ok += 1
    return ok, bad


def _block_after(text, at):
    """从 at 处开始找配对花括号，返回**块内**内容与结束位置。

    为什么不用「往后看 N 个字符」的窗口：附录 IN 这轮已经因为窗口被长注释
    撑破栽了**第三次**（A27 的 2000、A39 的 2000、IN.7 的 300）。
    窗口长度是这类规则的隐含假设，注释一长就失效 —— 而且失效方向是
    **假阴性**（看不见缺陷），最难发现。配对花括号没有这个问题。
    """
    i = text.find('{', at)
    if i < 0:
        return '', -1
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
    return text[i + 1:], -1


def _p2(t2):
    """PersonPose2.h：IN.4 / IN.5 / IN.7 / IN.8 / IN.9 的这一份。"""
    bad, ok = [], 0
    # IN.7: MapToFull 的 else 分支
    m = re.search(r'if\s*\(\s*map_id\[i\]\s*>=\s*0\s*\)', t2)
    # else 分支的**体**里必须真的把 other.points[i*3..] 清掉。
    # 两个坑都踩过：① 第一版只判「有没有 else」—— 变异测试把 else 里的 memset
    # 删掉，门禁照样全绿；② 改成「往后看 300 字符」之后，那段修复说明注释
    # （约 500 字）又把窗口撑破了，于是**正确**的代码被报成不合格。
    # 现在用配对花括号取块体，没有窗口。
    zeroes = False
    if m:
        e = re.search(r'\n\s*else\s*\{', t2[m.end():m.end() + 2000])
        if e:
            body, _ = _block_after(t2, m.end() + e.end() - 1)
            zeroes = bool(re.search(r'other\.points\s*\+\s*i\s*\*\s*3', body))
    if m and not zeroes:
        bad.append(('IN.7', '`MapToFull` 的 `map_id[i] < 0` 分支**什么都不写** —— '
                    '`other` 就是原来那个半模式对象，于是 `full[9]` 保留了 '
                    '`half[9]` 并被当成全模式的膝盖，紧接着「没检到脚踝就扩框」'
                    '的分支**永远走不到**，框底被截掉'))
    elif not m:
        bad.append(('IN.7', '找不到 `MapToFull` 的 map_id 分派'))
    else:
        ok += 1
    # IN.9: 溢出守卫
    n_ovf = len(re.findall(r'const\s+__int64\s+buffer_size\s*=\s*\(__int64\)\s*size_H', t2))
    if n_ovf < 1:
        bad.append(('IN.9', '`size_H*size_W*3` 是**纯 int 算术**，缺 `(__int64)` + '
                    '`> 0x7FFFFFFF` 守卫（拷贝 PersonPose.h 两处都有）'))
    else:
        ok += 1
    # IN.8
    n_c = len(re.findall(r'if\s*\(\s*!\s*temp_img\.ConvertFromBGR\(', t2))
    n_r = len(re.findall(r'if\s*\(\s*!\s*temp_img\.ResizeBilinear\(', t2))
    if n_c < 1 or n_r < 2:
        bad.append(('IN.8', 'ConvertFromBGR 判空 %d 处（要 1）/ ResizeBilinear 判空 %d 处'
                    '（两支各一处，要 2）' % (n_c, n_r)))
    else:
        ok += 1
    # IN.5
    n = len(re.findall(r'pose_ptr\s*=\s*pose_(?:half|full)_net\.GetBlobByName\(', t2))
    g = len(re.findall(r'if\s*\(\s*pose_ptr\s*==\s*0\s*\)', t2))
    if n and g < n:
        bad.append(('IN.5', '姿态侧 %d 处 `GetBlobByName` 只有 %d 处判空' % (n, g)))
    else:
        ok += 1
    # IN.4
    for name in ('ssd_C', 'pose_full_C', 'pose_half_C', 'pose_full_npts', 'pose_half_npts'):
        m = re.search(r'\bint\s+[A-Za-z0-9_,\s]*\b' + name + r'\b[^;]*;', t2)
        if m and '=' not in m.group(0):
            bad.append(('IN.4', '成员 `%s` 没有初值' % name))
            break
    else:
        ok += 1
    return ok, bad


def _md(t):
    bad, ok = [], 0
    if not re.search(r'if\s*\(\s*real_border_x\s*<\s*0\s*\)\s*real_border_x\s*=\s*0\s*;', t):
        bad.append(('IN.1', '`real_border_x` 可以是**负数**（`one_face.off_x = col1`，'
                    '而 MTCNN 的 `_refine_and_square_bbox` 边界钳位是**注释掉的**，'
                    '脸贴边是常规输入）-> 传给 `cv::Rect(Point,Point)` 会变成 '
                    '`x < 0` 的 ROI -> `cv::Mat(image, rect)` 抛**未捕获**的 '
                    'cv::Exception；同一个负值还让 `cur_box.col1 < real_border_x` '
                    '这个守卫**恒假**'))
    else:
        ok += 1
    if not re.search(r'if\s*\(\s*real_border_y\s*<\s*0\s*\)\s*real_border_y\s*=\s*0\s*;', t):
        bad.append(('IN.1', '`real_border_y` 同样没有夹到 0'))
    else:
        ok += 1
    return ok, bad


def _fc(t):
    bad, ok = [], 0
    # IN.2: 传 fill_val 而不是字面量 0
    lits = re.findall(r'Remap\s*\(\s*crop\s*,\s*dst_W\s*,\s*dst_H\s*,\s*1\s*,\s*1\s*,'
                      r'\s*map_x\s*,\s*map_y\s*,\s*true\s*,\s*([^)]+)\)', t)
    if not lits:
        bad.append(('IN.2', '找不到 CropImage 里的 Remap 调用'))
    elif any(v.strip() == '0' for v in lits):
        bad.append(('IN.2', '有 Remap 调用把填充值写死成字面量 `0` —— '
                    '形参 `fill_val` 被接受后**丢弃**了。'
                    '同文件另一个重载传的是 `fill_val`，一份对一份错；'
                    '而 VideoFaceDetection_Interface:769 明确传了 -1，'
                    '被静默吞掉'))
    elif not any('fill_val' in v for v in lits):
        bad.append(('IN.2', 'Remap 的填充值既不是 0 也不是 `fill_val`，形态不认识'))
    else:
        ok += 1
    return ok, bad


def scan(target, texts):
    fn = {'ZQ_CNN_PersonPose.h': _p1, 'ZQ_CNN_PersonPose2.h': _p2,
          'ZQ_CNN_MouthDetector.h': _md, 'ZQ_CNN_FaceCropUtils.h': _fc}[target]
    return fn(texts[target])


# ------------------------------------------------------------------ 自测
FULL_P1 = """
class ZQ_CNN_PersonPose {
	int ssd_C = 0, ssd_H = 0, ssd_W = 0;
	int pose_C = 0, pose_H = 0, pose_W = 0;
	int pose_npts = 0;
	void Detect() {
		const ZQ_CNN_Tensor4D* pose_ptr = pose_net.GetBlobByName(pose_out_blob_name);
		if (pose_ptr == 0) { return false; }
		int npts = output[nn].num_points;
		if (npts <= 0) continue;
		const __int64 buffer_size = (__int64)size_H * size_W * 3;
		if (buffer_size <= 0 || buffer_size > 0x7FFFFFFF) return false;
		if (!temp_img.ConvertFromBGR(&buffer[0], size_W, size_H, size_W * 3, 0, 1)) return false;
		if (!temp_img.ResizeBilinear(pose_input, pose_W, pose_H, 0, 0, SAMPLE_ALIGN_CENTER)) return false;
		{ // 第二个函数（不同作用域，同一个变量名）
			const __int64 buffer_size = (__int64)size_H * size_W * 3;
			if (buffer_size <= 0 || buffer_size > 0x7FFFFFFF) return false;
		}
		if (!temp_img.ConvertFromBGR(&buffer[0], size_W, size_H, size_W * 3, 0, 1)) return false;
		if (!temp_img.ResizeBilinear(pose_input, pose_W, pose_H, 0, 0, SAMPLE_ALIGN_CENTER)) return false;
	}
	struct BBox { float points[54]; BBox() { memset(points, 0, sizeof(points)); } };
};
"""

FULL_P2 = """
class ZQ_CNN_PersonPose2 {
	int ssd_C = 0, ssd_H = 0, ssd_W = 0;
	int pose_full_C = 0, pose_full_H = 0, pose_full_W = 0;
	int pose_half_C = 0, pose_half_H = 0, pose_half_W = 0;
	int pose_full_npts = 0;
	int pose_half_npts = 0;
	void Detect() {
		const __int64 buffer_size = (__int64)size_H * size_W * 3;
		if (buffer_size <= 0 || buffer_size > 0x7FFFFFFF) return false;
		if (!temp_img.ConvertFromBGR(&buffer[0], size_W, size_H, size_W * 3, 0, 1)) return false;
		if (!temp_img.ResizeBilinear(pose_input, pose_half_W, pose_half_H, 0, 0, A)) return false;
		if (!temp_img.ResizeBilinear(pose_input, pose_full_W, pose_full_H, 0, 0, A)) return false;
		pose_ptr = pose_half_net.GetBlobByName(x);
		if (pose_ptr == 0) { return false; }
		pose_ptr = pose_full_net.GetBlobByName(y);
		if (pose_ptr == 0) { return false; }
	}
	void MapToFull(BBox& other) {
		for (int i = 0; i < 14; i++) {
			if (map_id[i] >= 0)
				memcpy(other.points + i * 3, points + map_id[i] * 3, sizeof(float) * 3);
			else
			{
				memset(other.points + i * 3, 0, sizeof(float) * 3);
			}
		}
	}
};
"""

FULL_MD = """
class ZQ_CNN_MouthDetector {
	void Detect() {
		int real_border_x = __min(border_x, __min(one_face.off_x, width - one_face.off_x - one_face.width));
		int real_border_y = __min(border_y, __min(one_face.off_y, height - one_face.off_y - one_face.height));
		if (real_border_x < 0) real_border_x = 0;
		if (real_border_y < 0) real_border_y = 0;
	}
};
"""

FULL_FC = """
class ZQ_CNN_FaceCropUtils {
	static bool CropImage(..., float fill_val = 0.0f) {
		if (!img.Remap(crop, dst_W, dst_H, 1, 1, map_x, map_y, true, fill_val)) return false;
	}
};
"""

SELFCHECK = [
    ('PersonPose 全合格', 'ZQ_CNN_PersonPose.h', FULL_P1, []),
    ('IN.3 memset 写死 51',
     'ZQ_CNN_PersonPose.h', FULL_P1.replace('memset(points, 0, sizeof(points))',
                                            'memset(points, 0, sizeof(float) * 51)'), ['IN.3']),
    ('IN.4 成员没初值',
     'ZQ_CNN_PersonPose.h', FULL_P1.replace('int ssd_C = 0, ssd_H = 0, ssd_W = 0;',
                                            'int ssd_C, ssd_H, ssd_W;'), ['IN.4']),
    ('IN.6 缺 npts<=0',
     'ZQ_CNN_PersonPose.h', FULL_P1.replace('if (npts <= 0) continue;', ''), ['IN.6']),
    ('IN.8 只判一半',
     'ZQ_CNN_PersonPose.h',
     FULL_P1.replace('if (!temp_img.ConvertFromBGR(&buffer[0], size_W, size_H, size_W * 3, 0, 1)) return false;',
                     'temp_img.ConvertFromBGR(&buffer[0], size_W, size_H, size_W * 3, 0, 1);', 1),
     ['IN.8']),
    ('PersonPose2 全合格', 'ZQ_CNN_PersonPose2.h', FULL_P2, []),
    ('IN.7 MapToFull 缺 else',
     'ZQ_CNN_PersonPose2.h',
     FULL_P2.replace('			else' + chr(10) + '			{' + chr(10) + '				memset(other.points + i * 3, 0, sizeof(float) * 3);' + chr(10) + '			}', ''),
     ['IN.7']),
    ('IN.9 PersonPose2 缺溢出守卫',
     'ZQ_CNN_PersonPose2.h',
     FULL_P2.replace('const __int64 buffer_size = (__int64)size_H * size_W * 3;',
                     'std::vector<unsigned char> buffer(size_H*size_W * 3, 0);'),
     ['IN.9']),
    ('MouthDetector 全合格', 'ZQ_CNN_MouthDetector.h', FULL_MD, []),
    ('IN.1 缺 real_border 夹取',
     'ZQ_CNN_MouthDetector.h',
     FULL_MD.replace('if (real_border_x < 0) real_border_x = 0;\n\t\tif (real_border_y < 0) real_border_y = 0;\n', ''),
     ['IN.1', 'IN.1']),
    ('FaceCropUtils 全合格', 'ZQ_CNN_FaceCropUtils.h', FULL_FC, []),
    ('IN.2 fill_val 被丢弃',
     'ZQ_CNN_FaceCropUtils.h', FULL_FC.replace('true, fill_val)', 'true, 0)'), ['IN.2']),
]


def selfcheck():
    bad = 0
    for name, target, text, expect in SELFCHECK:
        _, b = scan(target, {target: strip_comments(text)})
        got = sorted(set(c for c, _ in b))
        if got != sorted(set(expect)):
            bad += 1
            print('  [self-MISMATCH] %s' % name)
            print('      expect %s, got %s' % (sorted(expect), got))
            for c, m in b:
                print('        %s: %s' % (c, m))
        else:
            print('  [self-OK]      %-34s %s' % (name, ','.join(got) if got else 'clean'))
    if bad:
        print('selfcheck FAILED: %d / %d mismatch' % (bad, len(SELFCHECK)))
        return 1
    print('selfcheck OK: %d cases, all as expected' % len(SELFCHECK))
    return 0


def main(argv):
    if '--selfcheck' in argv:
        return selfcheck()
    files = [(P1, 'utf-8'), (P2, 'utf-8'), (MD, 'utf-8'), (FC, 'gbk')]
    texts = {}
    for p, enc in files:
        texts[os.path.basename(p)] = strip_comments(read(p, enc))
    total_ok = 0
    failed = []
    for p, enc in files:
        name = os.path.basename(p)
        ok, b = scan(name, texts)
        total_ok += ok
        if b:
            failed.append(name)
            print('FAIL %s  (%s)' % (name, enc))
            for c, m in b:
                print('       - %s: %s' % (c, m))
        else:
            print('OK   %s  (%s)' % (name, enc))
    print('合计合格判定 %d 项' % total_ok)
    if failed:
        print('**%d 个文件不合格**' % len(failed))
        return 1
    print('姿态 / 嘴部 / 人脸裁剪: OK')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
