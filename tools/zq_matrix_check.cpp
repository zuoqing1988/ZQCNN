/* ZQ_Matrix / ZQ_Kahansum 的独立回归测试（不进构建，进 tools/）。
 *
 * ZQ_Matrix：本轮修过 `operator=` 的自赋值 use-after-free（审计附录 AE.1）。
 *   那条缺陷**不会崩**（glibc 的 free 只是把块放回空闲链表，紧接着的 malloc 很可能
 *   拿到同一块地址），所以 ASan 也抓不到 —— 只有「自赋值之后内容必须一字不差」
 *   这条断言能抓到。本测试的第一条就是这个。
 *   其余覆盖 GetData/SetData 的越界标志、Transpose、MatrixMul。
 *
 * ZQ_Kahansum：补偿求和的累加器。核心不变式是「累加顺序不同、结果差在 ulp 量级，
 * 而不是差一大截」。
 *
 * 编译（WSL）：
 *   g++ -O1 -g -fsanitize=address -I3rdparty/include/ZQlib \
 *       tools/zq_matrix_check.cpp -o /tmp/mx && /tmp/mx
 */
#include <stdio.h>
#include <cmath>
#include <vector>
#include "zqlib_msvc_shim.h"
#include "ZQ_Matrix.h"
#include "ZQ_Kahansum.h"

static int fail = 0;

static void check(bool cond, const char* what)
{
	if (!cond)
	{
		printf("  FAIL: %s\n", what);
		fail = 1;
	}
}

/* --- ZQ_Matrix --- */
static void test_matrix_core()
{
	const int R = 3, Cc = 4;
	ZQ::ZQ_Matrix<double> a(R, Cc);
	check(a.GetRowDim() == R && a.GetColDim() == Cc, "构造后行列应为 3x4");
	for (int i = 0; i < R; i++)
		for (int j = 0; j < Cc; j++)
			check(a.SetData(i, j, 10.0 + i * 10 + j), "SetData 合法下标应返回 true");

	/* 越界下标：GetData 应把 flag 置 false 并返回一个可忽略的值，
	   SetData 应返回 false —— 绝不能写坏内存。 */
	bool f = true;
	double v = a.GetData(0, 0, f);
	check(f && v == 10.0, "GetData(0,0) 合法");
	f = true;
	a.GetData(-1, 0, f);
	check(!f, "GetData 行 -1 应置 flag=false");
	f = true;
	a.GetData(0, Cc, f);
	check(!f, "GetData 列 nCol 应置 flag=false");
	f = true;
	a.GetData(R, 0, f);
	check(!f, "GetData 行 nRow 应置 flag=false");
	check(!a.SetData(-1, 0, 1.0), "SetData 行 -1 应返回 false");
	check(!a.SetData(0, Cc, 1.0), "SetData 列 nCol 应返回 false");
	/* 越界写之后，原有内容不能被破坏 */
	bool ok = true;
	for (int i = 0; i < R; i++)
		for (int j = 0; j < Cc; j++)
		{
			bool g = false;
			if (a.GetData(i, j, g) != 10.0 + i * 10 + j || !g) ok = false;
		}
	check(ok, "越界 SetData 之后原有内容应保持不变");
}

static void test_matrix_self_assign()
{
	ZQ::ZQ_Matrix<double> a(3, 2);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 2; j++) a.SetData(i, j, 100.0 + i * 10 + j);

	ZQ::ZQ_Matrix<double> backup(a);
	a = a;   /* 修复前: 先 free(data) 再 memcpy(other.data) -> 6 个元素全变垃圾 */

	bool same = true;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 2; j++)
		{
			bool f = false;
			double x = a.GetData(i, j, f), y = backup.GetData(i, j, f);
			if (x != y) same = false;
		}
	check(same, "Matrix: a = a 之后内容应与自赋值前完全相同（曾因先 free 后 memcpy 而变垃圾）");
}

static void test_matrix_ops()
{
	ZQ::ZQ_Matrix<double> a(2, 3), b(3, 2);
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 3; j++) a.SetData(i, j, i + j);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 2; j++) b.SetData(i, j, 1.0 + i * 10 + j);

	ZQ::ZQ_Matrix<double> c(2, 2);   // 默认构造是私有的
	check(ZQ::ZQ_Matrix<double>::MatrixMul(a, b, c), "MatrixMul 维度匹配应返回 true");
	check(c.GetRowDim() == 2 && c.GetColDim() == 2, "MatrixMul 结果应为 2x2");
	/* 手算 a*b */
	bool ok = true;
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
		{
			double e = 0;
			for (int k = 0; k < 3; k++)
			{
				bool f1 = false, f2 = false;
				e += a.GetData(i, k, f1) * b.GetData(k, j, f2);
			}
			bool f = false;
			if (c.GetData(i, j, f) != e) ok = false;
		}
	check(ok, "MatrixMul 元素应与手算一致");

	ZQ::ZQ_Matrix<double> t(a);
	t.Transpose();
	check(t.GetRowDim() == 3 && t.GetColDim() == 2, "Transpose 后行列应互换");
	ok = true;
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 3; j++)
		{
			bool f1 = false, f2 = false;
			if (t.GetData(j, i, f1) != a.GetData(i, j, f2)) ok = false;
		}
	check(ok, "Transpose 后 t[j][i] 应等于 a[i][j]");

	/* 维度不匹配的 MatrixMul 应失败，而不是产生垃圾 */
	ZQ::ZQ_Matrix<double> wrong(2, 2), out(2, 2);
	check(!ZQ::ZQ_Matrix<double>::MatrixMul(a, wrong, out), "维度不匹配的 MatrixMul 应返回 false");
}

/* --- ZQ_Kahansum ---
 *
 * Kahan 补偿求和。核心不变式：补偿求和的结果与朴素累加**在 ulp 量级内一致**，
 * 绝不会差一大截（补偿求和的收益体现在很长的求和序列上，量级差异往往极小，
 * 所以这里只断言「量级一致」并把实际差异打印出来，不去断言一个我没验证过的精度门槛）。
 *
 * 第一版我断言「朴素累加 1e6 次 0.1 的相对误差应 > 1e-6」作为对照组的有效性证明，
 * 实测只有 1.3e-11 —— 我对 0.1 这个特定值的误差估计过于悲观，断言本身错了。
 */
static void test_kahansum()
{
	const int N = 1000000;
	std::vector<double> in(N);
	for (int i = 0; i < N; i++) in[i] = 0.1;

	double naive = 0.0;
	for (int i = 0; i < N; i++) naive += in[i];
	double kahan = ZQ::ZQ_KahanSum<double>(in.data(), N);
	double exact = 0.1 * N;

	printf("  naive=%.10f  kahan=%.10f  exact=%.10f  |naive-exact|=%.3e  |kahan-exact|=%.3e\n",
	       naive, kahan, exact, fabs(naive - exact), fabs(kahan - exact));

	check(fabs(kahan - exact) < fabs(naive - exact) * 1.0001,
	      "Kahan 的误差不应比朴素累加更差");
	check(fabs(kahan - exact) / exact < 1e-9,
	      "Kahan 求 1e6 个 0.1 的相对误差应 < 1e-9" );

	/* n <= 0 时不应越界读；n==0 返回 0 */
	check(ZQ::ZQ_KahanSum<double>(in.data(), 0) == 0.0, "n=0 应返回 0");
	check(ZQ::ZQ_KahanSum<double>(in.data(), 1) == 0.1, "n=1 应返回唯一的元素");
	check(ZQ::ZQ_KahanSum<double>(in.data(), -5) == 0.0, "n<0 时循环不进入, 返回 0");
}

int main()
{
	setvbuf(stdout, NULL, _IONBF, 0);
	printf("--- ZQ_Matrix ---\n");
	test_matrix_core();
	test_matrix_self_assign();
	test_matrix_ops();
	printf("--- ZQ_Kahansum ---\n");
	test_kahansum();
	printf("%s\n", fail ? "RESULT: FAIL" : "RESULT: PASS");
	return fail ? 1 : 0;
}
