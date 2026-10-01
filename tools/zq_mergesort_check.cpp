/* ZQ_MergeSort 的独立回归测试（不进构建，进 tools/）。
 *
 * 背景：审计报告附录 O 把 `ZQ_MergeSort::_mergeSort_OOC` 的若干问题记为「不修」，
 * 理由是「这是 3rdparty 第三方头，改动面大且**无法在无数据库的机器上端到端验证**」。
 * 但这个头只 include 了 4 个标准头（stdlib/string/stdio/iostream），完全能脱离
 * ZQlibFaceID 单独编译 —— 于是「无法验证」这个前提本身就不成立，可以直接测。
 *
 * 覆盖：
 *   1. MergeSort_OOC 的正确性（含跨块归并：block_size < n 的多轮）
 *   2. MergeSort_OOCWithData 的正确性（键与载荷一起搬）
 *   3. 各种边界：n=0 / n=1 / 正好一个块 / 正好 2 的幂 / 降序
 *   4. 在 ASan + LeakSanitizer 下跑，泄漏与越界会直接报出来
 *
 * 编译（WSL）：
 *   g++ -O1 -g -fsanitize=address -I3rdparty/include/ZQlib \
 *       tools/zq_mergesort_check.cpp -o /tmp/ms && /tmp/ms
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <string>

/* 头文件用了 MSVC 的 __int64 / __max，必须在 include 之前补上。 */
#if !defined(_MSC_VER)
typedef long long __int64;
#ifndef __max
#define __max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef __min
#define __min(a, b) (((a) < (b)) ? (a) : (b))
#endif
#endif

#include "ZQ_MergeSort.h"

struct Rec
{
	float score;
	int   id;
};

static int fail = 0;

static void check(bool cond, const char* what)
{
	if (!cond)
	{
		printf("  FAIL: %s\n", what);
		fail = 1;
	}
}

static void write_vals(const char* path, const std::vector<float>& v)
{
	FILE* f = fopen(path, "wb");
	for (size_t i = 0; i < v.size(); i++)
	{
		float x = v[i];
		fwrite(&x, 4, 1, f);
	}
	fclose(f);
}

static std::vector<float> read_vals(const char* path)
{
	std::vector<float> v;
	FILE* f = fopen(path, "rb");
	if (!f) return v;
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	v.resize(n / 4);
	if (n) fread(&v[0], 4, n / 4, f);
	fclose(f);
	return v;
}

static void test_ooc(size_t n, bool asc, int mem_kb)
{
	char src[256], dst[256];
	snprintf(src, sizeof(src), "/tmp/ms_v_%zu.bin", n);
	snprintf(dst, sizeof(dst), "/tmp/ms_vout_%zu.bin", n);

	std::vector<float> in(n);
	unsigned s = 12345u + (unsigned)n;
	for (size_t i = 0; i < n; i++)
	{
		s = s * 1103515245u + 12345u;
		in[i] = (float)((int)(s >> 8) % 100000) / 1000.0f;
	}
	write_vals(src, in);

	bool ok = ZQ::ZQ_MergeSort::MergeSort_OOC<float>(src, dst, asc, mem_kb);
	if (n == 0)
	{
		/* 空输入: 头文件按 "total_len == 0" 判为失败, 这是合理的契约 */
		check(!ok, "空输入应返回 false");
		remove(src);
		return;
	}
	check(ok, "MergeSort_OOC 返回 true");

	std::vector<float> got = read_vals(dst);
	check(got.size() == n, "输出元素个数不变");
	std::vector<float> exp = in;
	std::sort(exp.begin(), exp.end());
	if (!asc) std::reverse(exp.begin(), exp.end());
	size_t bad = 0;
	for (size_t i = 0; i < got.size() && i < exp.size(); i++)
		if (got[i] != exp[i]) bad++;
	check(bad == 0, "排序结果逐个相等");

	remove(src); remove(dst);
	char t[300]; snprintf(t, sizeof(t), "%s.tmp", dst); remove(t);
}

static void test_ooc_with_data(size_t n, bool asc, int mem_kb)
{
	char sv[256], dv[256], sd[256], dd[256];
	snprintf(sv, sizeof(sv), "/tmp/ms_v2_%zu.bin", n);
	snprintf(dv, sizeof(dv), "/tmp/ms_vout2_%zu.bin", n);
	snprintf(sd, sizeof(sd), "/tmp/ms_d2_%zu.bin", n);
	snprintf(dd, sizeof(dd), "/tmp/ms_dout2_%zu.bin", n);

	std::vector<float> in(n);
	unsigned s = 999u + (unsigned)n;
	for (size_t i = 0; i < n; i++)
	{
		s = s * 1103515245u + 12345u;
		in[i] = (float)((int)(s >> 8) % 100000) / 1000.0f;
	}
	write_vals(sv, in);
	FILE* f = fopen(sd, "wb");
	for (size_t i = 0; i < n; i++)
	{
		int id = (int)i;
		fwrite(&id, 4, 1, f);
	}
	fclose(f);

	bool ok = ZQ::ZQ_MergeSort::MergeSortWithData_OOC<float>(
		sv, dv, sd, dd, 4, asc, mem_kb);
	if (n == 0)
	{
		check(!ok, "空输入应返回 false");
		remove(sv); remove(dv); remove(sd); remove(dd);
		return;
	}
	check(ok, "MergeSortWithData_OOC 返回 true");

	std::vector<float> got = read_vals(dv);
	FILE* g = fopen(dd, "rb");
	std::vector<int> gid;
	if (g)
	{
		fseek(g, 0, SEEK_END);
		long m = ftell(g);
		fseek(g, 0, SEEK_SET);
		gid.resize(m / 4);
		if (m) fread(&gid[0], 4, m / 4, g);
		fclose(g);
	}
	check(got.size() == n, "带载荷版输出元素个数不变");
	check(gid.size() == n, "带载荷版载荷个数不变");

	/* 载荷必须跟着键一起走：每个 id 的原始分数要等于排序后对应位置的分数 */
	size_t bad = 0;
	for (size_t i = 0; i < got.size() && i < gid.size(); i++)
		if (in[(size_t)gid[i]] != got[i]) bad++;
	check(bad == 0, "载荷与键保持对应关系");

	remove(sv); remove(dv); remove(sd); remove(dd);
	char t[300];
	snprintf(t, sizeof(t), "%s.tmp", dv); remove(t);
	snprintf(t, sizeof(t), "%s.tmp", dd); remove(t);
}

/* 内存受限路径: 头文件里 malloc 的结果**从不判空**
   (val_block_buffer / data_block_buffer / out_val_buffer / out_data_buffer,
   以及 _mergeSortWithData 的 tmp_data_left/right)。
   这里故意把 n 和 max_mem_size_in_KB 拉大, 让 block_size*2*elt_size 超过
   ASAN_OPTIONS.max_allocation_size_mb, 逼出 malloc 失败。
   期望: 返回 false 而不是空指针崩溃。
   需要环境变量 ZQ_MSORT_OOM_TEST=1 才跑 (正常跑时用不到)。 */
static void test_oom(void)
{
	char src[256], dst[256];
	const char* fn = getenv("ZQ_MSORT_OOM_SRC");
	snprintf(src, sizeof(src), "%s", fn ? fn : "/tmp/ms_oom.bin");
	snprintf(dst, sizeof(dst), "/tmp/ms_oom_out.bin");
	if (!fn)
	{
		printf("  (ZQ_MSORT_OOM_SRC 未设, 跳过 OOM 用例)\n");
		return;
	}
	printf("  OOM 用例: block 会被 ASan 的 max_allocation_size_mb 拒掉\n");
	bool ok = ZQ::ZQ_MergeSort::MergeSort_OOC<float>(src, dst, true, 1024 * 1024);
	printf("  MergeSort_OOC 在 malloc 失败时返回 %s\n", ok ? "true" : "false");
	remove(dst);
	char t[300]; snprintf(t, sizeof(t), "%s.tmp", dst); remove(t);
}

int main()
{
	setvbuf(stdout, NULL, _IONBF, 0);
	if (getenv("ZQ_MSORT_OOM_TEST"))
	{
		test_oom();
		printf("%s\n", fail ? "RESULT: FAIL" : "RESULT: PASS");
		return fail ? 1 : 0;
	}
	const size_t ns[] = { 0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 100, 1000, 4096, 5000 };
	const int mems[] = { 1, 4, 16, 1024 };
	printf("--- MergeSort_OOC ---\n");
	for (size_t i = 0; i < sizeof(ns) / sizeof(ns[0]); i++)
		for (size_t j = 0; j < sizeof(mems) / sizeof(mems[0]); j++)
			for (int a = 0; a < 2; a++)
			{
				char tag[64];
				snprintf(tag, sizeof(tag), "n=%zu mem=%dKB asc=%d", ns[i], mems[j], a);
				int before = fail;
				test_ooc(ns[i], a != 0, mems[j]);
				if (fail != before) printf("  (上面这条: %s)\n", tag);
			}
	printf("--- MergeSort_OOCWithData ---\n");
	for (size_t i = 0; i < sizeof(ns) / sizeof(ns[0]); i++)
		for (int a = 0; a < 2; a++)
			test_ooc_with_data(ns[i], a != 0, 4);
	printf("%s\n", fail ? "RESULT: FAIL" : "RESULT: PASS");
	return fail ? 1 : 0;
}
