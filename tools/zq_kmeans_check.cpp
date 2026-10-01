/* ZQ_Kmeans 的独立回归测试（不进构建，进 tools/）。
 *
 * 背景：审计报告附录 O 记过一条「`ZQ_Kmeans.h` 的 `k == 0` 往零长数组写 ——
 * 不修：仅经死代码链引用」。附录 X 证明这个头**能独立编译**（补 <math.h> 之后
 * 进了 tools/probe_zqlib_headers.py 的 OK 列表），于是那条「不修」的第二个理由
 * 也不成立，可以直接测。
 *
 * 覆盖：
 *   1. 正常路径：k=1/2/3 的聚类结果与手算/暴力最近中心一致
 *   2. k = 0 与 k < 0：应当返回 false，而不是写零长数组 / 抛未捕获异常
 *   3. 其余入参守卫：nPts<=0 / dim<=0 / k>nPts / 空指针
 *   4. ASan + LeakSanitizer 全程无报告
 *
 * 编译（WSL）：
 *   g++ -O1 -g -fsanitize=address -I3rdparty/include/ZQlib \
 *       tools/zq_kmeans_check.cpp -o /tmp/km && /tmp/km
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <cmath>
#include "ZQ_Kmeans.h"

static int fail = 0;

static void check(bool cond, const char* what)
{
	if (!cond)
	{
		printf("  FAIL: %s\n", what);
		fail = 1;
	}
}

/* 4 个 2 维点，明显分成两组：(0,0),(1,0)  |  (10,10),(11,10) */
static const float PTS[] = { 0,0, 1,0, 10,10, 11,10 };
static const int   NPTS = 4;
static const int   DIM  = 2;

static void normal_case()
{
	for (int k = 1; k <= 3; k++)
	{
		float centers[3 * DIM];
		int   idx[NPTS];
		bool ok = ZQ::ZQ_Kmeans<float>::Kmeans(NPTS, DIM, k, PTS, idx, centers);
		check(ok, "Kmeans 正常路径返回 true");
		if (!ok) continue;

		/* 每个点必须被分到最近的中心 */
		for (int i = 0; i < NPTS; i++)
		{
			double best = 1e30;
			for (int c = 0; c < k; c++)
			{
				double d = 0;
				for (int j = 0; j < DIM; j++)
				{
					double x = PTS[i * DIM + j] - centers[c * DIM + j];
					d += x * x;
				}
				if (d < best) best = d;
			}
			double d0 = 0;
			for (int j = 0; j < DIM; j++)
			{
				double x = PTS[i * DIM + j] - centers[idx[i] * DIM + j];
				d0 += x * x;
			}
			char msg[96];
			snprintf(msg, sizeof(msg),
			         "k=%d: 第 %d 个点被分到的不是最近中心 (%g vs %g)", k, i, d0, best);
			check(d0 <= best + 1e-3, msg);
		}
	}
}

static void bad_k_case()
{
	float centers[3 * DIM];
	int   idx[NPTS];
	char msg[128];

	for (int k = 0; k >= -2; k--)
	{
		bool ok = ZQ::ZQ_Kmeans<float>::Kmeans_with_init(
			NPTS, DIM, k, PTS, PTS, idx, centers);
		snprintf(msg, sizeof(msg), "Kmeans_with_init(k=%d) 应返回 false", k);
		check(!ok, msg);

		ok = ZQ::ZQ_Kmeans<float>::KmeansNormVec_with_init(
			NPTS, DIM, k, PTS, PTS, idx, centers);
		snprintf(msg, sizeof(msg), "KmeansNormVec_with_init(k=%d) 应返回 false", k);
		check(!ok, msg);

		ok = ZQ::ZQ_Kmeans<float>::Kmeans(NPTS, DIM, k, PTS, idx, centers);
		snprintf(msg, sizeof(msg), "Kmeans(k=%d) 应返回 false", k);
		check(!ok, msg);

		ok = ZQ::ZQ_Kmeans<float>::KmeansNormVec(NPTS, DIM, k, PTS, idx, centers);
		snprintf(msg, sizeof(msg), "KmeansNormVec(k=%d) 应返回 false", k);
		check(!ok, msg);
	}

	/* _select_init_center 是 public（头里 //private: 被注释掉了），
	   k > nPts 时 rand() % (nPts - i) 会除以 0 */
	bool ok = ZQ::ZQ_Kmeans<float>::_select_init_center(NPTS, DIM, NPTS + 1, PTS, centers);
	snprintf(msg, sizeof(msg), "_select_init_center(k=nPts+1) 应返回 false 而不是除以 0");
	check(!ok, msg);

	ok = ZQ::ZQ_Kmeans<float>::_select_init_center(NPTS, DIM, 0, PTS, centers);
	snprintf(msg, sizeof(msg), "_select_init_center(k=0) 应返回 false");
	check(!ok, msg);
}

static void other_guards()
{
	float centers[3 * DIM];
	int   idx[NPTS];
	bool ok = ZQ::ZQ_Kmeans<float>::Kmeans(0, DIM, 1, PTS, idx, centers);
	check(!ok, "nPts=0 返回 false");
	ok = ZQ::ZQ_Kmeans<float>::Kmeans(NPTS, 0, 1, PTS, idx, centers);
	check(!ok, "dim=0 返回 false");
	ok = ZQ::ZQ_Kmeans<float>::Kmeans(NPTS, DIM, NPTS + 1, PTS, idx, centers);
	check(!ok, "k>nPts 返回 false");
	ok = ZQ::ZQ_Kmeans<float>::Kmeans(NPTS, DIM, 1, 0, idx, centers);
	check(!ok, "pts=NULL 返回 false");
	ok = ZQ::ZQ_Kmeans<float>::Kmeans(NPTS, DIM, 1, PTS, 0, centers);
	check(!ok, "idx=NULL 返回 false");
	ok = ZQ::ZQ_Kmeans<float>::Kmeans(NPTS, DIM, 1, PTS, idx, 0);
	check(!ok, "out_centers=NULL 返回 false");
}

int main()
{
	setvbuf(stdout, NULL, _IONBF, 0);
	printf("--- 正常路径 ---\n");
	normal_case();
	printf("--- k <= 0 与 _select_init_center 越界 ---\n");
	bad_k_case();
	printf("--- 其它入参守卫 ---\n");
	other_guards();
	printf("%s\n", fail ? "RESULT: FAIL" : "RESULT: PASS");
	return fail ? 1 : 0;
}
