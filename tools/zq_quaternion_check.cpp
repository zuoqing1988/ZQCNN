/* ZQ_Quaternion / ZQ_RBFKernel 的独立回归测试（不进构建，进 tools/）。
 *
 * ZQ_Quaternion：本轮修过一个 `operator+=` 里 `w += w.z;` 的自引用笔误
 *   （审计附录 AB.3）。那条缺陷**返回值对不上就发现不了**吗？不，这次是
 *   靠「自赋值/复合赋值后各分量应等于逐分量相加」这条断言抓到的 ——
 *   而如果只测「两个四元数相加」，自引用会让结果差一个 z，断言照样能抓。
 *   这里测的正是复合赋值这条路径。
 *
 * ZQ_RBFKernel：附录 AF 补过 <cmath>（之前 fabs 靠 MSVC 传递 include）。
 *   它的两个核函数都有 flag 输出参数（false 表示参数非法），一并覆盖。
 *
 * 编译（WSL）：
 *   g++ -O1 -g -fsanitize=address -I3rdparty/include/ZQlib \
 *       tools/zq_quaternion_check.cpp -o /tmp/qt && /tmp/qt
 */
#include <stdio.h>
#include <cmath>
#include "zqlib_msvc_shim.h"
#include "ZQ_Quaternion.h"
#include "ZQ_RBFKernel.h"

static int fail = 0;

static void check(bool cond, const char* what)
{
	if (!cond)
	{
		printf("  FAIL: %s\n", what);
		fail = 1;
	}
}

static bool close(double a, double b, double eps = 1e-9)
{
	return fabs(a - b) <= eps * (1.0 + fabs(a) + fabs(b));
}

static void test_quat_basics()
{
	ZQ::ZQ_Quaternion<double> a(1.0, 2.0, 3.0, 4.0);
	ZQ::ZQ_Quaternion<double> b(0.5, -0.5, 1.5, 2.0);

	ZQ::ZQ_Quaternion<double> s = a + b;
	check(s.x == 1.5 && s.y == 1.5 && s.z == 4.5 && s.w == 6.0,
	      "operator+ 应逐分量相加");

	ZQ::ZQ_Quaternion<double> d = a - b;
	check(d.x == 0.5 && d.y == 2.5 && d.z == 1.5 && d.w == 2.0,
	      "operator- 应逐分量相减");

	/* 这条是本轮修的那处：原来是 w += w.z（自引用），应为 w += v.w */
	ZQ::ZQ_Quaternion<double> c = a;
	c += b;
	check(c.x == 1.5 && c.y == 1.5 && c.z == 4.5 && c.w == 6.0,
	      "operator+= 应等于逐分量相加（曾因 w += w.z 自引用而错）");

	ZQ::ZQ_Quaternion<double> m = a;
	m *= 2.0;
	check(m.x == 2.0 && m.y == 4.0 && m.z == 6.0 && m.w == 8.0,
	      "operator*= 应各分量乘以标量");

	check(close(a.Length(), sqrt(1.0 + 4.0 + 9.0 + 16.0)), "Length 应为模长");
	check(close(a.Dot(b), 1.0 * 0.5 + 2.0 * -0.5 + 3.0 * 1.5 + 4.0 * 2.0),
	      "Dot 应为分量积之和");
}

static void test_quat_rot()
{
	/* 四元数 -> 旋转矩阵 -> 四元数，应往返一致（w >= 0 的那一支） */
	const double qs[6][4] = {
		{ 0, 0, 0, 1 },              /* 单位四元数 */
		{ 0, 0, sin(0.3), cos(0.3) },/* 绕 z 转 0.3 rad */
		{ 0, sin(0.2), 0, cos(0.2) },/* 绕 y */
		{ sin(0.25), 0, 0, cos(0.25) },/* 绕 x */
		{ 0.1, 0.2, 0.3, 0.9 },
		{ -0.05, 0.12, -0.07, 0.99 },
	};
	for (int i = 0; i < 6; i++)
	{
		ZQ::ZQ_Quaternion<double> q(qs[i][0], qs[i][1], qs[i][2], qs[i][3]);
		/* Quat2Rot 假定输入是**单位四元数**（它的公式里没有归一化因子），
		   所以先用 Length() 归一化。用例 0~3 本来就是单位四元数，4~5 不是 ——
		   直接喂非单位四元数会算出不正交的 R，那是测试的错不是代码的错。 */
		{
			double len = q.Length();
			q = q * (1.0 / len);
		}
		double R[9];
		ZQ::ZQ_Quaternion<double>::Quat2Rot(q, R);
		/* 旋转矩阵应当是正交的: R * R^T == I */
		double worst = 0.0;
		for (int a = 0; a < 3; a++)
			for (int b = 0; b < 3; b++)
			{
				double s = 0.0;
				for (int k = 0; k < 3; k++) s += R[a * 3 + k] * R[b * 3 + k];
				double e = fabs(s - (a == b ? 1.0 : 0.0));
				if (e > worst) worst = e;
			}
		char msg[96];
		snprintf(msg, sizeof(msg), "用例 %d: Quat2Rot 的 R 应正交, 最大偏差 %.3e", i, worst);
		check(worst < 1e-12, msg);

		ZQ::ZQ_Quaternion<double> back;
		bool ok = ZQ::ZQ_Quaternion<double>::Rot2Quat(R, back);
		snprintf(msg, sizeof(msg), "用例 %d: Rot2Quat 应返回 true", i);
		check(ok, msg);

		/* q 与 -q 表示同一旋转，符号由 Rot2Quat 规范化，这里比较点积 */
		double dp = back.x * q.x + back.y * q.y + back.z * q.z + back.w * q.w;
		snprintf(msg, sizeof(msg),
		         "用例 %d: Rot2Quat(Quat2Rot(q)) 应与 q 同旋转 (dot=%.12f)", i, dp);
		check(fabs(fabs(dp) - 1.0) < 1e-9, msg);
	}
}

static void test_rbf_kernels()
{
	/* 注意 flag 的语义: 它表示「RBF_TYPE 这个分支被识别了」，**不是**
	   「参数在有效范围内」—— 每个 case 里都是无条件 flag = true。
	   我第一版把 flag 当成「参数是否合法」来断言，测出两条假 FAIL。 */
	{
		bool flag = false;
		double v = ZQ::ZQ_RBFKernel::_compact_kernel(flag, ZQ::ZQ_RBFKernel::COMPACT_CPC0, 5.0, 1.0);
		check(flag, "紧支撑核: type 被识别时 flag 应为 true");
		check(v == 0.0, "紧支撑核超出支撑半径 (d=5, r=1) 应返回 0");
	}
	{
		bool flag = false;
		double v = ZQ::ZQ_RBFKernel::_compact_kernel(flag, ZQ::ZQ_RBFKernel::COMPACT_CPC0, 0.0, 1.0);
		check(flag, "紧支撑核: d=0 时 flag 应为 true");
		check(close(v, 1.0), "紧支撑核 CPC0 在 d=0 时应为 (1-0)^2 = 1");
	}
	{
		bool flag = false;
		double v = ZQ::ZQ_RBFKernel::_compact_kernel(flag, ZQ::ZQ_RBFKernel::COMPACT_CPC0, 1.0, 1.0);
		check(close(v, 0.0), "紧支撑核 CPC0 在 d=r 边界应为 0");
	}
	{
		bool flag = false;
		double v = ZQ::ZQ_RBFKernel::_global_kernel(flag, ZQ::ZQ_RBFKernel::GLOBAL_GAUSS, 1.0, 1.0);
		check(flag, "全局核: type 被识别时 flag 应为 true");
		check(v > 0.0 && v < 1.0, "全局核 GAUSS 在 d=sigma=1 时应在 (0,1) 内");
	}
	{
		bool flag = false;
		double v = ZQ::ZQ_RBFKernel::_global_kernel(flag, ZQ::ZQ_RBFKernel::GLOBAL_GAUSS, 0.0, 1.0);
		check(close(v, 1.0), "全局核 GAUSS 在 d=0 时应为 exp(0)=1");
	}
	{
		bool flag = false;
		double v = ZQ::ZQ_RBFKernel::_global_kernel(flag, ZQ::ZQ_RBFKernel::GLOBAL_TPS, 1.0, 1.0);
		check(close(v, 0.0), "全局核 TPS: x^2*log(x) 在 x=1 时应为 0");
	}
	/* 记下来但本轮不改: sigma/radius 没有下界检查。
	   sigma == 0 时 x = fabs(distance/0) 得到 inf（distance>0）或 NaN（distance==0），
	   GLOBAL_TPS 于是返回 inf 而不是报错。不崩，但会污染下游。 */
	{
		bool flag = false;
		double v = ZQ::ZQ_RBFKernel::_global_kernel(flag, ZQ::ZQ_RBFKernel::GLOBAL_TPS, 1.0, 0.0);
		printf("  note: GLOBAL_TPS with sigma=0 returns %g (inf/NaN, no lower-bound check)\n", v);
	}
}

int main()
{
	setvbuf(stdout, NULL, _IONBF, 0);
	printf("--- ZQ_Quaternion ---\n");
	test_quat_basics();
	test_quat_rot();
	printf("--- ZQ_RBFKernel ---\n");
	test_rbf_kernels();
	printf("%s\n", fail ? "RESULT: FAIL" : "RESULT: PASS");
	return fail ? 1 : 0;
}
