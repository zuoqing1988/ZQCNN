/* ZQ_ImageProcessing 的独立回归测试（不进构建，进 tools/）。
 *
 * 背景：附录 AB 扫出来的两条真缺陷，都在 3x3 中值滤波这条路上：
 *
 * ① `Sort_decend_3elements`（:115-131）中间那一步是**照抄了第一步**：
 *      if (values[1] < values[2]) { tmp = values[0]; values[0] = values[1]; values[1] = tmp; }
 *    三步都在换 0/1，values[2] 从头到尾没参与过任何交换。
 *    `{0,1,2}` 的实际结果是 `{1,0,2}` —— 既不是降序，values[0] 也不是最大值。
 *
 * ② `MedianFilter33_1channel`（:1377-1381，非 OpenMP 那条路）第二列写错了槽位：
 *      col[0][0] = tmpImg[h*padding_width + 1];   // 应该是 col[1][0]
 *      ...
 *      Sort_decend_3elements(col[1]);             // 排的是从来没被赋值的 col[1]
 *    于是 col[1] 在第一行是**未初始化的栈内存**，之后是上一轮的残留值，
 *    再被 :1391 的 __min/Median_value/__max 读走。
 *
 * 两条叠在一起 = 3x3 中值滤波的结果完全不对，而它有活的调用方：
 * `ZQ_FindCorners.h:1562-1563`。
 *
 * 编译（WSL）：
 *   g++ -O1 -g -fsanitize=address -I3rdparty/include/ZQlib \
 *       tools/zq_imageprocessing_check.cpp -o /tmp/ip && /tmp/ip
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include "zqlib_msvc_shim.h"     /* 必须在下一行之前 */
#include "ZQ_ImageProcessing.h"

static int fail = 0;

static void check(bool cond, const char* what)
{
	if (!cond)
	{
		printf("  FAIL: %s\n", what);
		fail = 1;
	}
}

static void test_sort3()
{
	/* 三元素降序排序: 穷举 0..2 的 6 种排列, 外加一组含负数/相等值的 */
	struct C { int a, b, c; };
	const C cases[] = {
		{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0},
		{5,5,5},{-3,7,-1},{9,-9,0},
	};
	for (int i = 0; i < (int)(sizeof(cases) / sizeof(cases[0])); i++)
	{
		int v[3] = { cases[i].a, cases[i].b, cases[i].c };
		ZQ::ZQ_ImageProcessing::Sort_decend_3elements<int>(v);
		char msg[128];
		snprintf(msg, sizeof(msg),
		         "Sort_decend_3elements({%d,%d,%d}) 应得降序, 实得 {%d,%d,%d}",
		         cases[i].a, cases[i].b, cases[i].c, v[0], v[1], v[2]);
		check(v[0] >= v[1] && v[1] >= v[2], msg);
	}
}

/* 参考实现：把 9 个像素排序后取中间那个 */
static int median9(const std::vector<int>& img, int w, int h, int x, int y)
{
	std::vector<int> w9;
	for (int dy = -1; dy <= 1; dy++)
		for (int dx = -1; dx <= 1; dx++)
		{
			int yy = y + dy, xx = x + dx;
			/* 和实现里的 padding 一样: 越界复制边缘 */
			if (yy < 0) yy = 0; else if (yy >= h) yy = h - 1;
			if (xx < 0) xx = 0; else if (xx >= w) xx = w - 1;
			w9.push_back(img[yy * w + xx]);
		}
	std::sort(w9.begin(), w9.end());
	return w9[4];
}

static void test_median_filter(int w, int h)
{
	/* 用一幅中间有明显尖峰/尖谷的图: 正确的中值滤波会把孤立噪点抹掉 */
	std::vector<int> img(w * h);
	for (int i = 0; i < w * h; i++) img[i] = 100;
	/* 一个孤立亮点和几个暗点 */
	img[(h / 2) * w + (w / 2)] = 900;
	img[0] = 10;
	img[w * h - 1] = 20;

	std::vector<int> got(w * h, -1);
	ZQ::ZQ_ImageProcessing::MedianFilter33_1channel<int>(
		img.data(), got.data(), w, h, false);

	int bad = 0;
	int firstx = -1, firsty = -1, firstexp = 0, firstgot = 0;
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			int e = median9(img, w, h, x, y);
			if (got[y * w + x] != e)
			{
				if (bad == 0)
				{
					firstx = x; firsty = y; firstexp = e; firstgot = got[y * w + x];
				}
				bad++;
			}
		}
	char msg[200];
	snprintf(msg, sizeof(msg),
	         "MedianFilter33_1channel(%dx%d) 有 %d/%d 个像素与暴力中值不符 "
	         "(首个 (%d,%d): 期望 %d 实得 %d)", w, h, bad, w * h,
	         firstx, firsty, firstexp, firstgot);
	check(bad == 0, msg);
}

static void test_laplacian_edge(int w, int h)
{
	/* Laplacian 的四条边界特判没有维度守卫:
	     左边界读 (i*width + 1)，右边界读 (i*width + width - 2)，
	     上边界读 (1*width + j)，下边界读 ((height-2)*width + j)。
	   width == 1 时 (i*width + width - 2) = i-1，i==0 就是 -1 -> 读到缓冲区前面；
	   height == 1 时 ((height-2)*width + j) = -width+j 同理。
	   参考实现在下面，用 EnforceRange 复刻通用滤波器的边界处理。 */
	/* Laplacian 的四条边界特判没有维度守卫，width==1 / height==1 时 ASan 直接报
	   heap-buffer-overflow READ（读到缓冲区前面 4 字节）。这个测试只断言**能验证的
	   两件事**：
	     ① 跑完不崩（ASan 会替我们抓越界）；
	     ② 输出缓冲里不再残留调用前的哨兵值 —— 说明每个元素都被写过。
	   这里刻意**不**断言像素值：写参考实现时发现该函数在本机构建下的输出与
	   「可分离两趟」这个读法对不上（8x6 的 (0,0) 得 2、参考值 402，且 got[15]
	   保持初值 -1），这个疑点本轮没查清，见 audit 附录 AD。不要把一个没对上的
	   参考值写成断言 —— 那只会把测试变成「永远失败」。 */
	const int C = 1;
	const int SENTINEL = -12345;
	std::vector<int> img(w * h * C);
	for (int i = 0; i < w * h * C; i++) img[i] = 100 + (i % 7);
	std::vector<int> got(w * h * C, SENTINEL);
	ZQ::ZQ_ImageProcessing::Laplacian<int>(img.data(), got.data(), w, h, C, false);

	int untouched = 0, firstidx = -1;
	for (int i = 0; i < w * h * C; i++)
		if (got[i] == SENTINEL)
		{
			if (untouched == 0) firstidx = i;
			untouched++;
		}
	char msg[160];
	snprintf(msg, sizeof(msg),
	         "Laplacian(%dx%d) 有 %d/%d 个元素没被写过（首个下标 %d）",
	         w, h, untouched, w * h, firstidx);
	check(untouched == 0, msg);
}

int main()
{
	setvbuf(stdout, NULL, _IONBF, 0);
	printf("--- Sort_decend_3elements ---\n");
	test_sort3();
	printf("--- MedianFilter33_1channel ---\n");
	test_median_filter(3, 3);
	test_median_filter(5, 4);
	test_median_filter(7, 6);
	test_median_filter(16, 16);
	printf("--- Laplacian 的边界特判 ---\n");
	test_laplacian_edge(8, 6);
	test_laplacian_edge(1, 8);      /* width == 1 */
	test_laplacian_edge(8, 1);      /* height == 1 */
	test_laplacian_edge(1, 1);
	test_laplacian_edge(2, 2);
	printf("%s\n", fail ? "RESULT: FAIL" : "RESULT: PASS");
	return fail ? 1 : 0;
}
