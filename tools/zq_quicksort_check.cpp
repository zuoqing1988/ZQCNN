/* ZQ_QuickSort 的独立回归测试（不进构建，进 tools/）。
 *
 * 背景：附录 Y 之后继续按附录 X 的 OK 列表逐个重判。这是第一个翻出来的**活** bug：
 * ZQ_QuickSort.h:247
 *
 *     vals[i] = tmp_val;
 *     vals[i] = tmp_idx;      <-- 应该是 idx[i] = tmp_idx;
 *
 * 后果有两条：
 *   ① vals[i] 被「下标」覆盖掉刚刚写进去的枢轴值 -> 返回的数组是乱的
 *   ② idx[i] 从头到尾没被写过 -> 下标数组是错的
 *
 * 同一个文件里的 _quickSort(vals, idx, ...) 写法是对的
 * (`vals[i] = tmp_val; idx[i] = tmp_idx;`), 所以这是复制粘贴时漏改了一处。
 * 对外入口 FindKthMax(vals, idx, num, k, output, out_idx) 是 public, 直接可达。
 *
 * 编译（WSL）：
 *   g++ -O1 -g -fsanitize=address -I3rdparty/include/ZQlib \
 *       tools/zq_quicksort_check.cpp -o /tmp/qs && /tmp/qs
 */
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <functional>
#include "ZQ_QuickSort.h"

static int fail = 0;

static void check(bool cond, const char* what)
{
	if (!cond)
	{
		printf("  FAIL: %s\n", what);
		fail = 1;
	}
}

static void test_quicksort(int n, bool asc)
{
	std::vector<float> v(n);
	unsigned s = 4242u + (unsigned)n;
	for (int i = 0; i < n; i++)
	{
		s = s * 1103515245u + 12345u;
		v[i] = (float)((int)(s >> 8) % 100000) / 100.0f;
	}
	std::vector<float> got = v;
	ZQ::ZQ_QuickSort::QuickSort<float>(got.data(), n, asc);
	std::vector<float> exp = v;
	std::sort(exp.begin(), exp.end());
	if (!asc) std::reverse(exp.begin(), exp.end());
	check(got == exp, "QuickSort 单参数版结果与 std::sort 一致");
}

static void test_quicksort_with_idx(int n, bool asc)
{
	/* 必须用**互不相同**的值: 值有重复时「排完之后下标该怎么排」本来就不唯一
	   (std::sort 不稳定, 快排也不稳定), 拿它当期望值是在测一个没有定义的东西。
	   这里先造一组互异值再打乱。 */
	std::vector<float> v(n);
	std::vector<int> id(n);
	for (int i = 0; i < n; i++) v[i] = (float)i;
	for (int i = n - 1; i > 0; i--) std::swap(v[i], v[rand() % (i + 1)]);
	for (int i = 0; i < n; i++) id[i] = i;
	std::vector<float> got = v;
	std::vector<int>  gotid = id;
	ZQ::ZQ_QuickSort::QuickSort<float>(got.data(), gotid.data(), n, asc);

	std::vector<int> order(n);
	for (int i = 0; i < n; i++) order[i] = i;
	std::sort(order.begin(), order.end(), [&](int a, int b) {
		return asc ? (v[a] < v[b]) : (v[a] > v[b]);
	});

	bool vals_ok = true, idx_ok = true;
	for (int i = 0; i < n; i++)
	{
		if (got[i] != v[order[i]]) vals_ok = false;
		if (gotid[i] != order[i]) idx_ok = false;
	}
	check(vals_ok, "QuickSort 带 idx 版: 值序列正确");
	check(idx_ok, "QuickSort 带 idx 版: 下标序列正确");
}

/* FindKthMax 会对 vals 就地重排（quickselect 的分区副作用），
   所以只断言「output 是第 k 大」和「out_idx 指向原来那个元素」。 */
static void test_findkthmax(int n, int k)
{
	std::vector<float> v(n);
	std::vector<int> id(n);
	unsigned s = 31337u + (unsigned)n;
	for (int i = 0; i < n; i++)
	{
		s = s * 1103515245u + 12345u;
		v[i] = (float)((int)(s >> 8) % 100000) / 100.0f;
		id[i] = i;
	}
	std::vector<float> exp = v;
	std::sort(exp.begin(), exp.end(), std::greater<float>());

	std::vector<float> a = v;
	float out1 = -1;
	bool ok1 = ZQ::ZQ_QuickSort::FindKthMax<float>(a.data(), n, k, out1);
	char msg[128];
	snprintf(msg, sizeof(msg), "FindKthMax(n=%d,k=%d) 返回 true", n, k);
	check(ok1, msg);
	snprintf(msg, sizeof(msg), "FindKthMax(n=%d,k=%d) 单参数版 output = 第 k 大 (%.4f vs %.4f)",
	         n, k, out1, exp[k]);
	check(out1 == exp[k], msg);

	std::vector<float> b = v;
	std::vector<int> bi = id;
	float out2 = -1;
	int outidx2 = -1;
	bool ok2 = ZQ::ZQ_QuickSort::FindKthMax<float>(b.data(), bi.data(), n, k, out2, outidx2);
	snprintf(msg, sizeof(msg), "FindKthMax(n=%d,k=%d) 带 idx 版返回 true", n, k);
	check(ok2, msg);
	snprintf(msg, sizeof(msg), "FindKthMax(n=%d,k=%d) 带 idx 版 output = 第 k 大 (%.4f vs %.4f)",
	         n, k, out2, exp[k]);
	check(out2 == exp[k], msg);
	snprintf(msg, sizeof(msg),
	         "FindKthMax(n=%d,k=%d) 带 idx 版 out_idx 指向原数组里值为 %.4f 的那个元素 (现为 %d, 值 %.4f)",
	         n, k, exp[k], outidx2, (outidx2 >= 0 && outidx2 < n) ? v[outidx2] : -1.0);
	check(outidx2 >= 0 && outidx2 < n && v[outidx2] == exp[k], msg);

	/* 关键的一条: quickselect 会就地重排 vals, 但重排只能改变顺序,
	   **不能凭空改掉元素**。修之前 ZQ_Kmeans 那种「pivot 槽被写坏」的问题会
	   在这里现形: vals[i] = tmp_idx 把枢轴位置换成了下标值, 数组不再是原数组的
	   一个排列。这一条不依赖返回值, 所以能抓到纯数组损坏。 */
	std::vector<float> a1 = b;          // 就地重排后的结果
	std::vector<float> v0 = v;          // 原数组的升序
	std::sort(a1.begin(), a1.end());
	std::sort(v0.begin(), v0.end());
	snprintf(msg, sizeof(msg),
	         "FindKthMax(n=%d,k=%d) 带 idx 版就地重排后 vals 仍是原数组的一个排列",
	         n, k);
	check(a1 == v0, msg);
}

int main()
{
	setvbuf(stdout, NULL, _IONBF, 0);
	const int ns[] = { 1, 2, 3, 5, 8, 17, 64, 1000 };

	printf("--- QuickSort ---\n");
	for (int i = 0; i < (int)(sizeof(ns) / sizeof(ns[0])); i++)
		for (int a = 0; a < 2; a++)
		{
			test_quicksort(ns[i], a != 0);
			test_quicksort_with_idx(ns[i], a != 0);
		}
	printf("--- FindKthMax ---\n");
	for (int i = 0; i < (int)(sizeof(ns) / sizeof(ns[0])); i++)
		for (int k = 0; k < ns[i] && k < 6; k++)
			test_findkthmax(ns[i], k);

	printf("%s\n", fail ? "RESULT: FAIL" : "RESULT: PASS");
	return fail ? 1 : 0;
}
