/* ZQ_BitonicSort 的独立回归测试（不进构建，进 tools/）。
 *
 * 背景：附录 Z 在 ZQ_QuickSort 里抓到一处「返回值对、数组坏」的缺陷之后，
 * 继续按附录 X 的 OK 列表逐个重判。BitonicSort 属于同一类风险：
 * 一个**公开的排序算法**，写错了不会崩、只会静默给出错误顺序。
 *
 * 覆盖：
 *   1. Sort 两个重载：所有 2 的幂（1..4096）× 升降序，与 std::sort 逐元素相等
 *   2. 带 idx 重载的下标序列（值互异，顺序唯一）
 *   3. 非 2 的幂必须返回 false（而不是按奇数长度硬跑）
 *   4. len <= 0 / len == 1 的边界：不崩、返回 true
 *   5. Sort_Recursive 带 start_idx 的分段排序
 *   6. ASan + LeakSanitizer 全程无报告
 *
 * 编译（WSL）：
 *   g++ -O1 -g -fsanitize=address -I3rdparty/include/ZQlib \
 *       tools/zq_bitonicsort_check.cpp -o /tmp/bs && /tmp/bs
 */
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <algorithm>
#include "ZQ_BitonicSort.h"

static int fail = 0;

static void check(bool cond, const std::string& what)
{
	if (!cond)
	{
		printf("  FAIL: %s\n", what.c_str());
		fail = 1;
	}
}

/* 互异的值, 打成乱序 —— 值必须有重复就没有唯一的下标序列可断言 */
static std::vector<float> shuffled(int n, unsigned seed)
{
	std::vector<float> v(n);
	for (int i = 0; i < n; i++) v[i] = (float)i;
	unsigned s = seed ? seed : 1u;
	for (int i = n - 1; i > 0; i--)
	{
		s = s * 1103515245u + 12345u;
		int j = (int)((s >> 8) % (unsigned)(i + 1));
		std::swap(v[i], v[j]);
	}
	return v;
}

static void test_sort(int n, bool asc)
{
	char tag[80];
	snprintf(tag, sizeof(tag), "Sort(n=%d, asc=%d) 与 std::sort 逐元素相等", n, asc);
	std::vector<float> v = shuffled(n, 777u + (unsigned)n);
	std::vector<float> got = v;
	bool ok = ZQ::ZQ_BitonicSort::Sort<float>(got.data(), n, asc);
	check(ok, std::string("Sort(n=") + std::to_string(n) + ") 返回 true");
	std::vector<float> exp = v;
	std::sort(exp.begin(), exp.end());
	if (!asc) std::reverse(exp.begin(), exp.end());
	check(got == exp, tag);
}

static void test_sort_with_idx(int n, bool asc)
{
	char tag[96];
	std::vector<float> v = shuffled(n, 999u + (unsigned)n);
	std::vector<float> got = v;
	std::vector<int>  gi(n);
	std::vector<int>  id(n);
	for (int i = 0; i < n; i++) { gi[i] = i; id[i] = i; }
	bool ok = ZQ::ZQ_BitonicSort::Sort<float>(got.data(), gi.data(), n, asc);
	snprintf(tag, sizeof(tag), "Sort 带 idx(n=%d, asc=%d) 返回 true", n, asc);
	check(ok, tag);

	std::vector<int> order(n);
	for (int i = 0; i < n; i++) order[i] = i;
	std::sort(order.begin(), order.end(), [&](int a, int b) {
		return asc ? (v[a] < v[b]) : (v[a] > v[b]);
	});
	bool vals_ok = true, idx_ok = true;
	for (int i = 0; i < n; i++)
	{
		if (got[i] != v[order[i]]) vals_ok = false;
		if (gi[i] != order[i]) idx_ok = false;
	}
	snprintf(tag, sizeof(tag), "Sort 带 idx(n=%d, asc=%d) 值序列正确", n, asc);
	check(vals_ok, tag);
	snprintf(tag, sizeof(tag), "Sort 带 idx(n=%d, asc=%d) 下标序列正确", n, asc);
	check(idx_ok, tag);
}

static void test_non_power_of_two()
{
	const int ns[] = { 3, 5, 6, 7, 9, 10, 12, 100, 1000 };
	for (int i = 0; i < (int)(sizeof(ns) / sizeof(ns[0])); i++)
	{
		std::vector<float> v = shuffled(ns[i], 5u + (unsigned)ns[i]);
		std::vector<float> got = v;
		std::vector<int>  gi(ns[i], 0), id(ns[i], 0);
		char tag[96];
		bool ok = ZQ::ZQ_BitonicSort::Sort<float>(got.data(), ns[i], true);
		snprintf(tag, sizeof(tag), "Sort(n=%d 非2的幂) 返回 false", ns[i]);
		check(!ok, tag);
		ok = ZQ::ZQ_BitonicSort::Sort<float>(got.data(), gi.data(), ns[i], true);
		snprintf(tag, sizeof(tag), "Sort 带 idx(n=%d 非2的幂) 返回 false", ns[i]);
		check(!ok, tag);
		ok = ZQ::ZQ_BitonicSort::Sort_Recursive<float>(got.data(), ns[i], 0, true);
		snprintf(tag, sizeof(tag), "Sort_Recursive(n=%d 非2的幂) 返回 false", ns[i]);
		check(!ok, tag);
	}
}

static void test_edges()
{
	std::vector<float> dummy(8, 1.0f);
	std::vector<int>  di(8, 0);
	for (int n = -2; n <= 1; n++)
	{
		char tag[96];
		bool ok = ZQ::ZQ_BitonicSort::Sort<float>(dummy.data(), n, true);
		snprintf(tag, sizeof(tag), "Sort(len=%d) 不崩且返回 true", n);
		check(ok, tag);
		ok = ZQ::ZQ_BitonicSort::Sort<float>(dummy.data(), di.data(), n, true);
		snprintf(tag, sizeof(tag), "Sort 带 idx(len=%d) 不崩且返回 true", n);
		check(ok, tag);
		ok = ZQ::ZQ_BitonicSort::Sort_Recursive<float>(dummy.data(), n, 0, true);
		snprintf(tag, sizeof(tag), "Sort_Recursive(len=%d) 不崩且返回 true", n);
		check(ok, tag);
	}
}

static void test_recursive_offset(int n, int start)
{
	/* 把 [start, start+n) 这一段单独排好, 段外不动 */
	std::vector<float> v = shuffled(start + n, 31337u + (unsigned)n);
	std::vector<float> exp = v;
	std::sort(exp.begin() + start, exp.begin() + start + n);
	std::vector<float> got = v;
	bool ok = ZQ::ZQ_BitonicSort::Sort_Recursive<float>(got.data(), n, start, true);
	char tag[128];
	snprintf(tag, sizeof(tag),
	         "Sort_Recursive(n=%d, start=%d) 只把 [start, start+n) 排好", n, start);
	check(ok, tag);
	check(got == exp, tag);
}

int main()
{
	setvbuf(stdout, NULL, _IONBF, 0);
	const int ps[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096 };
	printf("--- Sort (2 的幂) ---\n");
	for (int i = 0; i < (int)(sizeof(ps) / sizeof(ps[0])); i++)
		for (int a = 0; a < 2; a++)
		{
			test_sort(ps[i], a != 0);
			test_sort_with_idx(ps[i], a != 0);
		}
	printf("--- 非 2 的幂必须拒绝 ---\n");
	test_non_power_of_two();
	printf("--- 边界 (len <= 0 / len == 1) ---\n");
	test_edges();
	printf("--- Sort_Recursive 带 start_idx ---\n");
	test_recursive_offset(8, 0);
	test_recursive_offset(8, 8);
	test_recursive_offset(16, 16);
	test_recursive_offset(64, 32);

	printf("%s\n", fail ? "RESULT: FAIL" : "RESULT: PASS");
	return fail ? 1 : 0;
}
