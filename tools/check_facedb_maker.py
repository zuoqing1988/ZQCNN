#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`ZQ_FaceDatabaseMaker.h` / `ZQ_FaceDetectorLibFaceDetect.h` 门禁 —— 附录 IO。

为什么**必须**是源码门禁
-----------------------
这轮挖出的三条缺陷 —— IO.1 / IO.2 / IO.4 —— **一条都跑不到**：

- `MakeDatabase(` / `MakeDatabaseCompact(` **零调用方**（`grep "MakeDatabase("` 全仓无命中）。
  也就是说 `_make_database` / `_extract_feature_from_img` / `_extract_feature_from_box`
  —— **整个检测器驱动的路径** —— 不被任何 sample 执行。四个 `SampleFaceDatabase*`
  只用 `*AlreadyCropped` 变体，恰好绕开了 `detectors[id]` 那一支。
- `_auto_detect_database` 的 `#else`(Linux) 分支**从不链接** —— 10 个 include
  此头的 sample 全部包在 `#if defined(_WIN32)` 里。
- `ZQ_FaceDetectorLibFaceDetect` 的 GRAY 分支（IO.4）需要「灰度图 + roi_min_x > 0」，
  而三个调用点全传 BGR。

也就是说：**回归全绿不代表这些路径验过了**。按 AGENTS.md
「回归全绿 ≠ 新增的东西编过了」，这一节必须落进 changelog，
而钉住它们只能靠源码判据。

判定
----
IO.1  `ErrorCode err_code` 必须有初值（2 处，两个并行区各一处）
IO.2  `intptr_t lfDir` 必须初始化，且 `_findclose(lfDir)` 必须在 `lfDir != -1l` 守卫内
IO.3  七个像素格式分支里，采样指针**必须都带 `rect_off_x`**（IO.4 的判据
      按「每个分支都带」写，不按「有没有一处带」—— 否则只修一处也会判过）
IO.4  `float center[2] = { ... }` 的 braced-init 里不得有会触发 `-Wnarrowing`
      的 `int * double`（报告 A3；HIGH 桶，warn_sweep 会抓，但先在这里报出来更准）

用法
----
    python tools/check_facedb_maker.py              # 扫默认文件
    python tools/check_facedb_maker.py --selfcheck  # 先自测
"""
from __future__ import print_function

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAKER = os.path.join(ROOT, 'ZQlibFaceID', 'ZQ_FaceDatabaseMaker.h')
LIBFD = os.path.join(ROOT, 'ZQlibFaceID', 'ZQ_FaceDetectorLibFaceDetect.h')


def read(p):
    with io.open(p, 'r', encoding='utf-8') as f:
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


RE_ERRCODE = re.compile(r'^\s*ErrorCode\s+err_code\s*(=|;)', re.M)
RE_LFD_INIT = re.compile(r'\bintptr_t\s+lfDir\s*=\s*-1l\s*;')
RE_CLOSE = re.compile(r'^\s*_findclose\s*\(\s*lfDir\s*\)\s*;', re.M)
RE_CLOSE_GUARD = re.compile(r'if\s*\(\s*lfDir\s*!=\s*-1l\s*\)')
# 每个像素格式分支的采样指针
RE_BRANCH = re.compile(r'case\s+ZQ_PIXEL_FMT_(\w+)\s*:')
RE_SAMPLE = re.compile(r'const\s+unsigned\s+char\s*\*\s*ori_pix_ptr\s*=')


def _block_after(text, at):
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


def scan(name, t):
    bad, ok = [], 0
    # ---- IO.1 ----
    errs = RE_ERRCODE.findall(t)
    if len(errs) >= 2:
        uninit = [m.start() for m in RE_ERRCODE.finditer(t)
                  if m.group(1) == ';']
        if uninit:
            bad.append(('IO.1', '`ErrorCode err_code;` 有 %d 处**没有初值**，'
                        '而 CropImage 失败的分支会把它 push 进 ErrorCodes'
                        '（读未定值是 UB，随后被格式化函数打进 err_log.txt，产生随机错误码）'
                        % len(uninit)))
        else:
            ok += 1
    else:
        bad.append(('IO.1', '只找到 %d 处 `ErrorCode err_code`，预期 2 处（两个并行区各一处）'
                    % len(errs)))

    # ---- IO.2 ----
    if not RE_LFD_INIT.search(t):
        bad.append(('IO.2', '`intptr_t lfDir` **未初始化**；配合那个空 if 体，'
                    '_findfirst 失败时会对 -1 句柄调 _findclose'))
    else:
        ok += 1
    closes = [m.start() for m in RE_CLOSE.finditer(t)]
    unguarded = []
    for st in closes:
        # 往前找最近一个非空、非注释的行
        pre = t[:st]
        ln_start = pre.rfind('\n') + 1
        prev_line = t[:ln_start].rstrip().split('\n')[-1] if ln_start else ''
        if not RE_CLOSE_GUARD.search(prev_line):
            unguarded.append(t[:st].count('\n') + 1)
    if unguarded:
        bad.append(('IO.2', '第 %s 行的 `_findclose(lfDir)` **没有 `lfDir != -1l` 守卫**'
                    % ', '.join(str(x) for x in unguarded)))
    elif closes:
        ok += 1
    else:
        bad.append(('IO.2', '找不到 `_findclose(lfDir)`'))
    # IO.4: -Wnarrowing（报告 A3；它在 ZQ_FaceDatabaseMaker.h，不在 libfacedetect 那个头）
    m = re.search(r'float\s+center\s*\[\s*2\s*\]\s*=\s*\{([^}]*)\}', t)
    if m and re.search(r'\bcols\b[^,}]*\*\s*0\.5(?![0-9a-zA-Z_])'
         r'|\brows\b[^,}]*\*\s*0\.5(?![0-9a-zA-Z_])', m.group(1)):
        bad.append(('IO.4', '`float center[2] = { image.cols*0.5, image.rows*0.5 };` —— '
                    '`int * double` 在 braced-init-list 里窄化成 float，'
                    'gcc 报 `-Wnarrowing`（HIGH 桶，warn_sweep 会抓，'
                    '但这里报出来能顺带指出该改成什么'))
    else:
        ok += 1
    return ok, bad


def scan_libfd(t):
    """IO.3 / IO.4：七个像素格式分支的采样指针。"""
    bad, ok = [], 0
    branches = list(RE_BRANCH.finditer(t))
    missing = []
    for k, m in enumerate(branches):
        stop = branches[k + 1].start() if k + 1 < len(branches) else len(t)
        body, _ = _block_after(t, m.end())
        if not body:
            body = t[m.end():stop]
        sm = RE_SAMPLE.search(body)
        if sm is None:
            continue          # 该分支不做采样（不是我们要管的形态）
        if 'rect_off_x' not in body[sm.start():sm.start() + 400]:
            missing.append(m.group(1))
    if missing:
        bad.append(('IO.3', '像素格式分支 %s 的采样指针**少加了 rect_off_x** —— '
                    '同一个 switch 里另外 6 个分支全都加了，一份对六份错。'
                    '后果：roi_min_x > 0 时 ROI 采样整体左移，检出框和 landmark 全部错位'
                    '（不是内存越界，读仍在界内）' % ', '.join(missing)))
    else:
        ok += 1

    return ok, bad


# ------------------------------------------------------------------ 自测
FULL_MAKER = """
class ZQ_FaceDatabaseMaker {
	void a() {
		ErrorCode err_code = ERR_WARNING;
		intptr_t lfDir = -1l;
		if ((lfDir = _findfirst(dir.c_str(), &fileDir)) == -1l) { }
		else { do { } while (_findnext(lfDir, &fileDir) == 0); }
		if (lfDir != -1l)
			_findclose(lfDir);
	}
	void b() {
		ErrorCode err_code = ERR_WARNING;
		for (int i = 0; i < n; i++) {
			if ((lfDir = _findfirst(dir.c_str(), &fileDir)) == -1l) { }
			else { do { } while (_findnext(lfDir, &fileDir) == 0); }
			if (lfDir != -1l)
				_findclose(lfDir);
		}
	}
};
"""

FULL_LIBFD = """
class ZQ_FaceDetectorLibFaceDetect {
	void f() {
		switch (fmt) {
		case ZQ_PIXEL_FMT_GRAY:
			{ const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x); }
			break;
		case ZQ_PIXEL_FMT_BGR:
			{ const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x)*3; }
			break;
		case ZQ_PIXEL_FMT_RGB:
			{ const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x)*3; }
			break;
		case ZQ_PIXEL_FMT_BGRX:
			{ const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x)*4; }
			break;
		case ZQ_PIXEL_FMT_RGBX:
			{ const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x)*4; }
			break;
		case ZQ_PIXEL_FMT_XBGR:
			{ const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x)*4; }
			break;
		case ZQ_PIXEL_FMT_XRGB:
			{ const unsigned char* ori_pix_ptr = img + (h + rect_off_y)*widthStep + (w + rect_off_x)*4; }
			break;
		}
		float center[2] = { image.cols*0.5f, image.rows*0.5f };
	}
};
"""

FULL_MAKER2 = FULL_MAKER + """
	float center[2] = { image.cols*0.5f,image.rows*0.5f };
"""

SELFCHECK = [
    ('Maker 全合格', 'ZQ_FaceDatabaseMaker.h', FULL_MAKER2, []),
    ('IO.1 err_code 无初值',
     'ZQ_FaceDatabaseMaker.h', FULL_MAKER2.replace('ErrorCode err_code = ERR_WARNING;',
                                                  'ErrorCode err_code;'), ['IO.1', 'IO.1']),
    ('IO.2 lfDir 未初始化',
     'ZQ_FaceDatabaseMaker.h', FULL_MAKER2.replace('intptr_t lfDir = -1l;',
                                                  'intptr_t lfDir;'), ['IO.2']),
    ('IO.2 _findclose 无守卫',
     'ZQ_FaceDatabaseMaker.h',
     FULL_MAKER2.replace('if (lfDir != -1l)\n\t\t\t_findclose(lfDir);', '_findclose(lfDir);'),
     ['IO.2']),
    ('LibFaceDetect 全合格', 'ZQ_FaceDetectorLibFaceDetect.h', FULL_LIBFD, []),
    ('IO.3 GRAY 分支漏 rect_off_x',
     'ZQ_FaceDetectorLibFaceDetect.h',
     FULL_LIBFD.replace('img + (h + rect_off_y)*widthStep + (w + rect_off_x); }',
                       'img + (h + rect_off_y)*widthStep + w; }'), ['IO.3']),
    ('IO.4 narrowing',
     'ZQ_FaceDatabaseMaker.h',
     FULL_MAKER2.replace('image.cols*0.5f,image.rows*0.5f',
                        'image.cols*0.5,image.rows*0.5'),
     ['IO.4']),
]


def selfcheck():
    bad = 0
    for name, target, text, expect in SELFCHECK:
        t = strip_comments(text)
        r = scan(target, t) if target.endswith('FaceDatabaseMaker.h') else scan_libfd(t)
        got = sorted(set(c for c, _ in r[1]))
        if got != sorted(set(expect)):
            bad += 1
            print('  [self-MISMATCH] %s' % name)
            print('      expect %s, got %s' % (sorted(expect), got))
            for c, m in r[1]:
                print('        %s: %s' % (c, m))
        else:
            print('  [self-OK]      %-30s %s' % (name, ','.join(got) if got else 'clean'))
    if bad:
        print('selfcheck FAILED: %d / %d mismatch' % (bad, len(SELFCHECK)))
        return 1
    print('selfcheck OK: %d cases, all as expected' % len(SELFCHECK))
    return 0


def main(argv):
    if '--selfcheck' in argv:
        return selfcheck()
    total_ok = 0
    failed = []
    for p, fn in ((MAKER, scan), (LIBFD, scan_libfd)):
        name = os.path.basename(p)
        t = strip_comments(read(p))
        ok, b = fn(name, t) if fn is scan else fn(t)
        total_ok += ok
        if b:
            failed.append(name)
            print('FAIL %s' % name)
            for c, m in b:
                print('       - %s: %s' % (c, m))
        else:
            print('OK   %s' % name)
    print('合计合格判定 %d 项' % total_ok)
    if failed:
        print('**%d 个文件不合格**' % len(failed))
        return 1
    print('人脸库构建 / libfacedetect 封装: OK')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
