/* 第 4 批缺陷的独立回归测试（不进构建，进 tools/）。
 *
 * 覆盖四条（全部来自附录 AB.5 里「代理报告已回读确认、本轮未处理」的那批）：
 *
 *  A. ZQ_KDTree.h:584  BuildKDTree 接受 npts == 0
 *     守卫只有 `npts < 0`，于是 npts==0 通过；随后 _find_min_max(:114) 无条件
 *     读 pts[pts_idx[0]][d]，而 pts_idx 是 new int[0] -> 零长数组越界读。
 *
 *  B. ZQ_KDTree.h:625/641/652/664  四个搜索入口只判 `tree->npts < k`，
 *     **没判 k <= 0**（与 ZQ_Kmeans 同一形状）。k==0 时 _update_search_result
 *     走 `cur_k == k` 分支读 out_dis2[k-1] = out_dis2[-1]。
 *
 *  C. ZQ_KDTree.h:499-509  _recursive_ann_fix_radius_search 的叶节点循环里
 *     `out_idx[cur_k] = ...; cur_k++;` **没有 cur_k < k 检查** ——
 *     半径内的点数超过 k 时直接冲垮调用方的输出缓冲。与 k 的取值无关，
 *     用一个合法的小 k 就能触发。
 *
 *  D. ZQ_WeightedMedian.h:15-46  `num` 从不判 <=0。num==0 时两个循环都不进，
 *     inf_num 保持 0，于是 output = sort_vals[num - inf_num] = sort_vals[0]
 *     读零长数组；num<0 时 new T[-1] 抛未捕获异常。
 *
 *  E. ZQ_CubicInterpolation.h:73-87  ZQ_nCubicInterpolate 只有 n==1 会终止，
 *     n<=0 无限递归（1 << (n-1)*2 还是一次 UB 的移位）。
 *
 *  F. ZQ_FindLargestSubMatrix.h:14-18  `new unsigned int[in_height*in_width]`
 *     是 unsigned x unsigned，乘积在 2^32 处回绕后才拓宽到 size_t；
 *     且 in_height==0 时 (in_height-1)*in_width+w 也回绕成巨大下标。
 *
 * 编译（WSL）：
 *   g++ -O1 -g -fsanitize=address -I3rdparty/include/ZQlib \
 *       tools/zq_batch4_check.cpp -o /tmp/b4 && /tmp/b4
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <cmath>
#include "zqlib_msvc_shim.h"
#include "ZQ_KDTree.h"
#include "ZQ_WeightedMedian.h"
#include "ZQ_CubicInterpolation.h"
#include "ZQ_FindLargestSubMatrix.h"

static int fail = 0;

static void check(bool cond, const char* what)
{
	if (!cond)
	{
		printf("  FAIL: %s\n", what);
		fail = 1;
	}
}

/* --- A/B/C: KDTree --- */
static void test_kdtree()
{
	const int NPTS = 12, NDIM = 2;
	std::vector<double> data(NPTS * NDIM);
	for (int i = 0; i < NPTS; i++)
	{
		data[i * NDIM + 0] = (double)(i % 4);
		data[i * NDIM + 1] = (double)(i / 4);
	}

	ZQ::ZQ_KDTree<double> tree;
	check(tree.BuildKDTree(data.data(), NPTS, NDIM, 3), "BuildKDTree 正常路径返回 true");

	double pt[NDIM] = { 1.5, 1.5 };
	int   oi[NPTS];
	double od[NPTS];

	/* C: 半径覆盖全部 12 个点, 但调用方只给 3 个槽 —— 现在会写穿 */
	int kk = 3;
	check(tree.AnnFixRadiusSearch(pt, 1000.0, kk, oi, od),
	      "AnnFixRadiusSearch 返回 true");

	/* B: k == 0 必须被拒 */
	check(!tree.AnnSearch(pt, 0, oi, od), "AnnSearch(k=0) 应返回 false");
	check(!tree.BruteForceSearch(pt, 0, oi, od), "BruteForceSearch(k=0) 应返回 false");
	check(!tree.AnnSearchWithInitalRadius(pt, 0, oi, od, 1000.0, kk), "AnnSearchWithInitalRadius(k=0) 应返回 false");
	check(!tree.AnnFixRadiusSearch(pt, 1000.0, 0, oi, od), "AnnFixRadiusSearch(k=0) 应返回 false");
	/* k < 0 同理 */
	check(!tree.AnnSearch(pt, -1, oi, od), "AnnSearch(k=-1) 应返回 false");

	/* A: npts == 0 必须被拒 */
	ZQ::ZQ_KDTree<double> empty;
	check(!empty.BuildKDTree(data.data(), 0, NDIM, 3), "BuildKDTree(npts=0) 应返回 false");
	check(!empty.BuildKDTree(data.data(), -1, NDIM, 3), "BuildKDTree(npts=-1) 应返回 false");
}

/* --- D: WeightedMedian --- */
static void test_weighted_median()
{
	const double vals[3]   = { 1.0, 2.0, 3.0 };
	const double wts[3]    = { 1.0, 1.0, 1.0 };
	double out = -1;

	check(ZQ::ZQ_WeightedMedian::FindMedian<double>(vals, wts, 3, out),
	      "FindMedian 正常路径返回 true");
	check(out == 2.0, "FindMedian(1,2,3 等权) 应为 2");

	check(!ZQ::ZQ_WeightedMedian::FindMedian<double>(vals, wts, 0, out),
	      "FindMedian(num=0) 应返回 false");
	check(!ZQ::ZQ_WeightedMedian::FindMedian<double>(vals, wts, -1, out),
	      "FindMedian(num=-1) 应返回 false");
	check(!ZQ::ZQ_WeightedMedian::FindMedian<double>(0, wts, 3, out),
	      "FindMedian(vals=NULL) 应返回 false");
}

/* --- E: CubicInterpolation --- */
static void test_cubic()
{
	/* n<=0 会无限递归; 只跑 n==0, ASan 会报 stack-overflow */
	double p[64];
	for (int i = 0; i < 64; i++) p[i] = (double)i;
	float co[1] = { 0.5f };
	double r = ZQ::ZQ_nCubicInterpolate<double>(0, p, co);
	check(r == r, "ZQ_nCubicInterpolate(n=0) 应返回而不是爆栈");
}

/* --- F: FindLargestSubMatrix --- */
static void test_find_largest_submatrix()
{
	const unsigned W = 4, H = 4;
	std::vector<bool> flag(W * H, false);     /* 用 vector<bool> 的引用语义不行, 换数组 */
	static bool f[16];
	for (unsigned i = 0; i < W * H; i++) f[i] = false;
	int ox = -1, oy = -1, w = -1, h = -1;
	ZQ::ZQ_FindLargestSubMatrix::FindLargestSubMatrix(f, W, H, ox, oy, w, h);
	check(w >= 0 && h >= 0, "全 false 时应给出 0 宽或 0 高的答案而不是崩");

	/* in_height == 0: 下标 (in_height-1)*in_width + w 回绕 */
	ZQ::ZQ_FindLargestSubMatrix::FindLargestSubMatrix(f, W, 0u, ox, oy, w, h);
	check(true, "in_height=0 不应越界写");
}

int main()
{
	setvbuf(stdout, NULL, _IONBF, 0);
	printf("--- KDTree ---\n");
	test_kdtree();
	printf("--- WeightedMedian ---\n");
	test_weighted_median();
	printf("--- CubicInterpolation ---\n");
	test_cubic();
	printf("--- FindLargestSubMatrix ---\n");
	test_find_largest_submatrix();
	printf("%s\n", fail ? "RESULT: FAIL" : "RESULT: PASS");
	return fail ? 1 : 0;
}
