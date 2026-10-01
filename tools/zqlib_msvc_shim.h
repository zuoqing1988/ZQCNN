/* ZQlib 头的 MSVC→gcc 兼容垫片。
 *
 * ZQlib 的头大量使用 MSVC 专有的类型与内建（__int64 / __min / __max /
 * _fseeki64 / _ftelli64 / fopen_s / strcpy_s / sprintf_s）。MSVC 的 CRT 会
 * 经由其它标准头把它们传递进来，**libstdc++ 不会** —— 于是同一个头在 Windows
 * 编得过、在 Linux 编不过。
 *
 * 每个 zq_*_check.cpp 都在 `#include` 目标 ZQlib 头**之前**包含本文件（顺序不能
 * 反）。tools/zqlib_probe_shim.h 转发到本文件，供 tools/probe_zqlib_headers.py
 * 与 tools/warn_sweep_zqlib.py 使用 —— **本文件是唯一一份真实定义**，改垫片只改这里。
 *
 * 补上垫片之后能被单独编译的 ZQlib 头从 81 涨到 83（ZQ_MergeSort.h 缺 <vector>、
 * ZQ_Kmeans.h 缺 <math.h>，那两处是直接改头本身修的）。
 */
#ifndef _ZQ_TESTS_MSVC_SHIM_H_
#define _ZQ_TESTS_MSVC_SHIM_H_

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#if !defined(_MSC_VER)

typedef long long __int64;
typedef unsigned long long __uint64;

#ifndef __min
#define __min(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef __max
#define __max(a, b) (((a) > (b)) ? (a) : (b))
#endif

/* _fseeki64 只能给**一种**定义（函数式）；同时给对象式和函数式会 redefinition 报错 */
#define _fseeki64(f, o, w) fseeko64((f), (o), (w))
#define _ftelli64(f)       ftello64(f)
#define fopen_s(p, a, m)   (((*(p)) = fopen((a), (m))) == 0 ? 0 : -1)
#define strcpy_s(d, n, s)  strncpy((d), (s), (n))
#define sprintf_s          snprintf
#define _snprintf          snprintf

#endif /* !_MSC_VER */

#endif
