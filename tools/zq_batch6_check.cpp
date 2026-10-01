/* 第 6 批缺陷的独立回归测试（不进构建，进 tools/）。
 *
 *  A. ZQ_Matrix.h:100-109  `operator=` 没有自赋值保护
 *     ```cpp
 *     if(data) free(data);          // 先释放自己的
 *     nRow = other.nRow;
 *     data = (T*)malloc(...);
 *     memcpy(data, other.data, ...); // 再从 other 拷 —— a = a 时 other.data
 *                                     // 就是刚被 free 掉的那块
 *     ```
 *     `a = a;` = 释放后从已释放内存读（use-after-free read）且值是垃圾。
 *     拷贝构造（:95-97）没这个问题，因为它读 other 时还不拥有 data。
 *
 *  B. ZQ_ScanLinePolygonFill.h:49-65  `ScanLinePolygonFillWithClip` 把
 *     `ClipPolygon` 算出来的 `out_poly` 丢掉了 —— 后面三处（取 minmax、
 *     建边表、扫描填充）全用的是**原始的** `polygon_pts`。于是这个
 *     「WithClip」函数其实一点都没裁剪；而 `FillOneStrokeWithClip`
 *     （唯一调用方，:70）传的 width/height 本来是想用来限制笔画的。
 *
 * 编译（WSL）：
 *   g++ -O1 -g -fsanitize=address -I3rdparty/include/ZQlib \
 *       tools/zq_batch6_check.cpp -o /tmp/b6 && /tmp/b6
 */
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "zqlib_msvc_shim.h"
#include "ZQ_Matrix.h"
#include "ZQ_ScanLinePolygonFill.h"

static int fail = 0;

static void check(bool cond, const char* what)
{
	if (!cond)
	{
		printf("  FAIL: %s\n", what);
		fail = 1;
	}
}

/* --- A: ZQ_Matrix --- */
static void test_matrix_self_assign()
{
	ZQ::ZQ_Matrix<double> a(3, 2);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 2; j++)
			a.SetData(i, j, 100.0 + i * 10 + j);

	/* 先拷一份留底, 自赋值之后内容必须一模一样 */
	ZQ::ZQ_Matrix<double> backup(a);
	a = a;                                  // 自赋值: use-after-free

	bool same = true;
	for (int i = 0; i < 3 && same; i++)
		for (int j = 0; j < 2 && same; j++)
		{
			bool f1 = false, f2 = false;
			double v1 = a.GetData(i, j, f1);
			double v2 = backup.GetData(i, j, f2);
			if (!f1 || !f2 || v1 != v2) same = false;
		}
	check(same, "Matrix: a = a 之后内容应与自赋值前完全相同");
}

static void test_matrix_normal()
{
	ZQ::ZQ_Matrix<double> a(3, 2), b(4, 4);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 2; j++) a.SetData(i, j, i * 10 + j + 0.5);
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++) b.SetData(i, j, -1);
	b = a;
	check(b.GetRowDim() == 3 && b.GetColDim() == 2, "Matrix: 正常赋值后尺寸应跟着变");
	bool f = false;
	check(b.GetData(2, 1, f) == 21.5 && f, "Matrix: 正常赋值后元素应正确");
}

/* --- B: ScanLinePolygonFillWithClip --- */
static void test_scanline_clip()
{
	/* 一个明显超出 [0,width)x[0,height) 的三角形 */
	const int W = 8, H = 8;
	std::vector<ZQ::ZQ_Vec2D> tri;
	tri.push_back(ZQ::ZQ_Vec2D(-5.0f, 4.0f));
	tri.push_back(ZQ::ZQ_Vec2D(12.0f, 4.0f));
	tri.push_back(ZQ::ZQ_Vec2D(4.0f, -5.0f));

	std::vector<ZQ::ZQ_Vec2D> px;
	bool ok = ZQ::ZQ_ScanLinePolygonFill::ScanLinePolygonFillWithClip(tri, W, H, px);
	check(ok, "ScanLinePolygonFillWithClip 返回 true");

	int outside = 0;
	for (size_t i = 0; i < px.size(); i++)
	{
		float x = px[i].x, y = px[i].y;
		if (x < -0.5f || x > W - 0.5f || y < -0.5f || y > H - 0.5f)
			outside++;
	}
	printf("  WithClip 返回 %d 个像素, 其中 %d 个落在 %dx%d 之外\n",
	       (int)px.size(), outside, W, H);
	check(outside == 0, "WithClip 返回的像素应全部落在 width x height 之内");

	/* 完全在图外的多边形: 裁剪后应为空 */
	std::vector<ZQ::ZQ_Vec2D> far;
	far.push_back(ZQ::ZQ_Vec2D(100.0f, 100.0f));
	far.push_back(ZQ::ZQ_Vec2D(110.0f, 100.0f));
	far.push_back(ZQ::ZQ_Vec2D(105.0f, 110.0f));
	std::vector<ZQ::ZQ_Vec2D> px2;
	ZQ::ZQ_ScanLinePolygonFill::ScanLinePolygonFillWithClip(far, W, H, px2);
	printf("  完全在图外的三角形 -> %d 个像素\n", (int)px2.size());
	check(px2.empty(), "完全在图外的多边形裁剪后应为空");
}

int main()
{
	setvbuf(stdout, NULL, _IONBF, 0);
	printf("--- ZQ_Matrix ---\n");
	test_matrix_normal();
	test_matrix_self_assign();
	printf("--- ZQ_ScanLinePolygonFillWithClip ---\n");
	test_scanline_clip();
	printf("%s\n", fail ? "RESULT: FAIL" : "RESULT: PASS");
	return fail ? 1 : 0;
}
