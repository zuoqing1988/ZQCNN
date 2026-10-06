#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Probe: cross-platform portability smells in first-party code.

Part 1: constants that differ between the _WIN32 side and the other side of
an `#if defined(_WIN32) ... #else ...`. One platform computing with a
different constant than the other is a silent portability bug.

Part 2: MSVC-only spellings (`fopen_s`, `strtok_s`, `Sleep`, ...) that are
NOT enclosed by a pure `_WIN32`/`_MSC_VER` preprocessor test. Those would
not exist at all on Linux/gcc.

Deliberately a *probe*, not a gate. Every hit needs a human decision: most
are legitimate, and the point of the probe is to make that decision
explicit rather than by luck.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {'.git', '3rdparty', 'build', 'build_x64', 'build_linux',
             'cmake-build-debug', 'cmake-build-release', 'tools',
             'opencv', 'opencv_src', 'opencv-build', '__pycache__',
             'node_modules', 'cmake-out-win32-x64', 'cmake-out-linux-x64'}

WIN_PAT = re.compile(
    r'^\s*#\s*(?:if|ifdef)\s+(?:defined\s*\(\s*_WIN32\s*\)|_WIN32|_WIN64)\b')
ELSE_PAT = re.compile(r'^\s*#\s*else\s*(?:if)?\b')
ENDIF_PAT = re.compile(r'^\s*#\s*endif\b')

# Numbers worth flagging: anything with a decimal point, or an integer that is
# not 0/1/2 (those show up as incidental loop bounds all over the place).
NUM_PAT = re.compile(r'(?<![\w.])\d+\.\d+(?![\w.])|(?<![\w.])\d+(?![\w.])')
TRIVIAL = {'0', '1', '2'}

# Part 2. `Sleep` is Windows-only by name; `alloca` is the opposite case
# (works everywhere) and is deliberately not listed.
MSVCISM_PAT = re.compile(
    r'\b(?:fopen_s|freopen_s|localtime_s|gmtime_s|strtok_s|sprintf_s|'
    r'strncpy_s|strcpy_s|scanf_s|fscanf_s|sscanf_s|vsprintf_s|_itoa|'
    r'strdup_s|getcwd_s|_getcwd|__try|__except|__declspec|_countof|'
    r'Sleep|SetCurrentDirectory|_chdir|_mkdir|_rmdir|_access|_open|_close|'
    r'_read|_write|_lseek|_tell|_fstat|_fileno|_fdopen|_dup|_commit|'
    r'_getpid|_getcwd|SleepEx)\b')
# Any of these appearing in an enclosing #if makes it a "pure win32" test only
# if the condition is *just* the win32 predicate (no || with a unix branch).
IF_PAT = re.compile(r'^\s*#\s*(if|ifdef|ifndef)\b\s*(.*)$')
WIN_TOKEN = re.compile(r'_WIN32|_WIN64|_MSC_VER')


def strip_block_comments(lines):
    """Blank out /* ... */ spans that run across lines, keeping line numbers.

    `strip_noise` only sees one line at a time, so it cannot tell that a
    six-line debug dump starting with `/*const static int BUF_LEN = 50;`
    is dead code. Left unhandled, that reports six "unguarded MSVC-isms"
    inside a comment. Comment bytes become spaces so that any column
    arithmetic elsewhere still lines up.
    """
    out = []
    in_block = False
    for ln in lines:
        buf = []
        i = 0
        n = len(ln)
        while i < n:
            if in_block:
                if ln.startswith('*/', i):
                    buf.append('  ')
                    i += 2
                    in_block = False
                else:
                    buf.append(' ')
                    i += 1
            else:
                if ln.startswith('/*', i):
                    buf.append('  ')
                    i += 2
                    in_block = True
                else:
                    buf.append(ln[i])
                    i += 1
        out.append(''.join(buf))
    return out


def strip_noise(line):
    """Drop comments and string/char literals so we only see real code."""
    out = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == '"' or c == "'":
            q = c
            i += 1
            while i < n:
                if line[i] == '\\':
                    i += 2
                    continue
                if line[i] == q:
                    i += 1
                    break
                i += 1
            out.append(' ')
            continue
        if c == '/' and i + 1 < n and line[i + 1] == '/':
            break
        if c == '/' and i + 1 < n and line[i + 1] == '*':
            j = line.find('*/', i + 2)
            i = n if j < 0 else j + 2
            out.append(' ')
            continue
        out.append(c)
        i += 1
    return ''.join(out)


def nums_in(lines):
    s = set()
    for ln in lines:
        for m in NUM_PAT.findall(strip_noise(ln)):
            if m not in TRIVIAL:
                s.add(m)
    return s


def walk_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.lower().endswith(('.h', '.hpp', '.cpp', '.cc')):
                yield os.path.join(dirpath, fn)


def win32_stack_map(lines):
    """For each line, True when it sits under a win32-only region.

    Each enclosing #if is classified into one of three kinds, because a
    boolean is not enough to get this right:

      neutral -- the condition does not test a win32 macro at all (an
                 include guard, say). Imposes no constraint either way.
      win     -- every disjunct is a win32 macro, so the level confines
                 code to the Windows build. `#ifndef _WIN32` flips which
                 arm that is, and so does crossing the #else.
      shared  -- the condition ORs a win32 macro with something else,
                 e.g. `#if defined(_WIN32) || defined(__linux__)`. That arm
                 compiles on **both** platforms, so an MSVC-ism in it is
                 unguarded whichever arm we are on.

    Getting this wrong is not academic. The first version of this probe
    required *every* enclosing level to be a win32 test, which marked every
    line of every include-guarded header unguarded and reported 109 false
    positives in correctly-guarded ZQlibFaceID code. The --selftest cases
    pin all three kinds in both directions.
    """
    guarded = [False] * len(lines)
    stack = []  # list of [kind, else_seen, negated]
    for idx, ln in enumerate(lines):
        m = IF_PAT.match(ln)
        if m:
            kind_word, cond = m.group(1), m.group(2)
            neg = (kind_word == 'ifndef')
            disjuncts = re.split(r'\|\|', cond)
            if any(WIN_TOKEN.search(d) for d in disjuncts):
                kind = 'win' if all(WIN_TOKEN.search(d)
                                    for d in disjuncts) else 'shared'
            else:
                kind = 'neutral'
            stack.append([kind, False, neg])
            continue
        if ENDIF_PAT.match(ln):
            if stack:
                stack.pop()
            continue
        if ELSE_PAT.match(ln):
            if stack:
                stack[-1][1] = True
            continue
        # Guarded means: at least one enclosing level confines this line to
        # the Windows build, and no enclosing level makes it shared.
        # An include guard alone does NOT make a line guarded -- treating
        # it as if it did is what made this gate toothless on the real
        # tree: a real unguarded `fopen_s` planted in an include-guarded
        # ZQlibFaceID header passed, because the only enclosing level was
        # that neutral include guard. The --selftest cannot see this (its
        # include-guard case also contains a win32 level); only mutating
        # a real file exposes it, so do that before trusting a green run.
        if any(kind == 'shared' for kind, _, _ in stack):
            guarded[idx] = False
            continue
        guarded[idx] = any(kind == 'win' and (neg == else_seen)
                           for kind, else_seen, neg in stack)
    return guarded


def selftest():
    """Prove the probe still has discriminating power.

    This probe shipped reporting 109 hits that were all false positives:
    the preprocessor-stack map demanded that *every* enclosing #if be a
    win32 test, so every line of every include-guarded header came out
    "unguarded". A green run would have been meaningless. These cases pin
    both directions -- must-flag and must-not-flag -- so that a future edit
    cannot silently cost the check its teeth.

    (Lesson, AGENTS.md: a gate that stays green after a mutation means
    either the gate lost power or the mutation never applied. Count the
    sites, and make the negative control explicit.)
    """
    cases = [
        # (name, source, expected unguarded-hit count)
        ('plain unguarded', [
            'void f(){ FILE* p=0; fopen_s(&p,"a","r"); }',
        ], 1),
        ('win32 arm is guarded', [
            '#if defined(_WIN32)',
            '  fopen_s(&p,"a","r");',
            '#else',
            '  p = fopen("a","r");',
            '#endif',
        ], 0),
        ('else arm is NOT guarded', [
            '#if defined(_WIN32)',
            '  p = fopen("a","r");',
            '#else',
            '  sprintf_s(buf, 8, "%d", 1);',
            '#endif',
        ], 1),
        ('inside an include guard + win32', [
            '#ifndef _ZQ_X_H_',
            '#define _ZQ_X_H_',
            '#if defined(_WIN32)',
            '  _mkdir("d");',
            '#else',
            '  mkdir("d", 0755);',
            '#endif',
            '#endif',
        ], 0),
        ('no space after #if keeps depth sane', [
            '#if(defined(_WIN32))',
            '  fopen_s(&p,"a","r");',
            '#else',
            '  p = fopen("a","r");',
            '#endif',
            'fopen_s(&p,"b","r");',
        ], 1),
        ('shared || branch is not guarded', [
            '#if defined(_WIN32) || defined(__linux__)',
            '  fopen_s(&p,"a","r");',
            '#endif',
        ], 1),
        ('both disjuncts win32 macros = win-only', [
            '#if defined(_WIN32) || defined(_MSC_VER)',
            '  strcpy_s(d,n,s);',
            '#endif',
        ], 0),
        ('#ifndef _WIN32 body is the LINUX arm', [
            '#ifndef _WIN32',
            '  fopen_s(&p,"a","r");',
            '#else',
            '  p = fopen("a","r");',
            '#endif',
        ], 1),
        ('#ifndef _WIN32 else arm is the win32 one', [
            '#ifndef _WIN32',
            '  p = fopen("a","r");',
            '#else',
            '  fopen_s(&p,"a","r");',
            '#endif',
        ], 0),
        ('neutral level does not rescue a shared one', [
            '#ifndef _ZQ_Y_H_',
            '#if defined(_WIN32) || defined(__linux__)',
            '  sscanf_s(l,"%d",&i);',
            '#endif',
            '#endif',
        ], 1),
        ('nested win32 region', [
            '#if defined(_WIN32)',
            '  #if defined(_MSC_VER)',
            '    strcpy_s(d, n, s);',
            '  #endif',
            '#endif',
        ], 0),
        ('commented-out call does not count', [
            '// fopen_s(&p,"a","r");',
            '/* sprintf_s(buf,8,"%d",1); */',
        ], 0),
        ('multi-line block comment does not count', [
            '/*const static int BUF_LEN = 50;',
            'sprintf_s(file, BUF_LEN, "%d_mu.txt", i);',
            'fopen_s(&out, file, "w");',
            'fclose(out);*/',
            'fopen_s(&p,"real","r");',
        ], 1),
        ('code before a block comment still counts', [
            'fopen_s(&p,"real","r");',
            '/* trailing note */',
        ], 1),
        ('block comment inside a win32 arm', [
            '#if defined(_WIN32)',
            '/* note',
            'fopen_s(&p,"a","r");',
            '*/',
            '#else',
            'p = fopen("a","r");',
            '#endif',
        ], 0),
    ]
    failures = []
    for name, src, expect in cases:
        code = [strip_noise(ln)
                for ln in strip_block_comments(src)]
        guarded = win32_stack_map(src)
        got = 0
        for idx, ln in enumerate(code):
            if ln.strip() and MSVCISM_PAT.search(ln) and not guarded[idx]:
                got += 1
        ok = (got == expect)
        print('  [%s] %-38s expect=%d got=%d' %
              ('PASS' if ok else 'FAIL', name, expect, got))
        if not ok:
            failures.append(name)
    if failures:
        print('SELFTEST FAILED: %d case(s): %s' % (len(failures), ', '.join(failures)))
        return 1
    print('selftest OK: %d cases' % len(cases))
    return 0


def main():
    if '--selftest' in sys.argv:
        return selftest()
    part1 = 0
    regions = 0
    part2 = 0
    for path in walk_files():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.read().split('\n')
        except UnicodeDecodeError:
            continue
        rel = os.path.relpath(path, ROOT).replace('\\', '/')

        # ---- Part 2 ----
        code_lines = [strip_noise(ln) for ln in strip_block_comments(lines)]
        guarded = win32_stack_map(lines)
        for idx, ln in enumerate(code_lines):
            code = ln
            if not code.strip():
                continue
            m = MSVCISM_PAT.search(code)
            if m and not guarded[idx]:
                part2 += 1
                print('[unguarded MSVC-ism] %s:%d  %s' % (rel, idx + 1, code.strip()[:110]))
                print('    ---')

        # ---- Part 1 ----
        i = 0
        while i < len(lines):
            if not WIN_PAT.match(lines[i]):
                i += 1
                continue
            # collect until matching #else at this nesting level
            depth = 1
            j = i + 1
            win_part = []
            else_start = None
            else_part = []
            while j < len(lines):
                ln = lines[j]
                if WIN_PAT.match(ln) or re.match(r'^\s*#\s*if\b', ln):
                    depth += 1
                elif ENDIF_PAT.match(ln):
                    depth -= 1
                    if depth == 0:
                        break
                elif depth == 1 and ELSE_PAT.match(ln):
                    else_start = j
                if depth == 1:
                    if else_start is None:
                        win_part.append(ln)
                    else:
                        else_part.append(ln)
                j += 1
            if else_start is not None:
                regions += 1
                w = nums_in(win_part)
                o = nums_in(else_part)
                only_w = sorted(w - o, key=lambda x: (len(x), x))
                only_o = sorted(o - w, key=lambda x: (len(x), x))
                if only_w or only_o:
                    part1 += 1
                    print('%s:%d' % (rel, i + 1))
                    if only_w:
                        print('    _WIN32 only : %s' % ' '.join(only_w))
                    if only_o:
                        print('    other   only: %s' % ' '.join(only_o))
                    print('    ---')
            i = j + 1 if j > i else i + 1
    print('regions with an #else: %d' % regions)
    print('regions with asymmetric constants: %d' % part1)
    print('unguarded MSVC-only spellings: %d' % part2)
    # Only part 2 gates. Part 1 stays informational on purpose: the one hit
    # in the whole tree is `Sleep(10 * 1000)` vs `sleep(10)` in
    # model/benchncnn.cpp, which is *correct* (ms vs seconds) and would
    # otherwise turn a correct branch into a permanent red gate.
    return 1 if part2 else 0


if __name__ == '__main__':
    sys.exit(main())