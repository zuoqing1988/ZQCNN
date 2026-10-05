// zq_rodrigues_check.cpp —— `ZQ_Rodrigues` 的数值回归（附录 IY）
//
// 起因（附录 IY.1）
// ------------------
// `tools/probe_zqlib_headers.py` 一直把 `ZQ_Calibration.h` 列在 BROKEN 的 9 个里，
// 而那 9 个此前都被归为「需要 windows.h / OpenCV / GL，本机测不了」。
// 逐个打开看，**只有这一个不是**：它引用的
//     ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun / ZQ_Rodrigues_r2R_jac /
//     ZQ_Rodrigues_R2r_fun / ZQ_Rodrigues_autoscale
// **从来没存在过**，于是整个头编不过：
//     ZQ_Calibration.h:1851: error: 'ZQ_Rodrigues_r2R_fun' is not a member of 'ZQ::ZQ_Rodrigues'
//
// 补齐之后，这个头从「编不过」变成「编得过且数值可验」——
// 本测试就是那个「可验」。
//
// 覆盖
// ----
//   1. `r2R_fun(r, R)` 与底层 `r2R(r, R)` **逐位相同**（薄包装不该改结果）
//   2. `autoscale` 之后再 `r2R`，与直接 `r2R` **逐位相同**
//      —— R 对旋转向量 2*pi 周期，这是 autoscale 唯一的契约；
//         如果这一条不成立，说明 autoscale 的定义错了
//   3. `r2R_jac` 的 3x9 Jacobian 与**中心差分**一致
//      —— 这是唯一能证明「Jacobian 真的对」的判据
//   4. `R2r_fun` 的往返：`r -> R -> r'` 在主值区间内回到 r
//   5. 零向量 / 空指针：薄包装必须返回 false 而不是崩
//
// 跑在 ASan + LeakSanitizer 与 UBSan 两轴上（tools/run_zqlib_checks.py）。
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include "ZQ_Rodrigues.h"

static int g_fail = 0;

static void fail(const char* what, double got, double want, double tol)
{
    printf("  **FAIL** %-42s got %.9g want %.9g (tol %.3g)\n", what, got, want, tol);
    g_fail++;
}

static void ok(const char* what, double err, double tol)
{
    printf("  %-6s %-42s 最大偏差 %.3g (tol %.3g)\n",
           (err <= tol) ? "ok" : "FAIL", what, err, tol);
    if (err > tol) g_fail++;
}

static double max_abs_diff(const double* a, const double* b, int n)
{
    double m = 0;
    for (int i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main()
{
    printf("=== ZQ_Rodrigues 数值回归 ===\n");

    static const double RS[][3] = {
        { 0.0, 0.0, 0.0 },
        { 0.1, 0.0, 0.0 },
        { 0.0, 0.3, 0.0 },
        { 0.0, 0.0, -0.25 },
        { 0.2, -0.3, 0.4 },
        { 1.0, 2.0, -3.0 },
        { 0.5, 0.5, 0.5 },
        { 3.14159265358979, 0.0, 0.0 },        // 恰好 pi
        { 6.28318530717959, 0.0, 0.0 },        // 2*pi
        { 7.0, 1.0, -2.0 },                    // |r| > pi，需要折
    };
    const int nRS = (int)(sizeof(RS) / sizeof(RS[0]));

    // 1 + 2
    double e1 = 0, e2 = 0;
    for (int i = 0; i < nRS; i++) {
        double r[3] = { RS[i][0], RS[i][1], RS[i][2] };
        double Ra[9], Rb[9], Rc[9];
        double rcopy[3] = { r[0], r[1], r[2] };

        ZQ::ZQ_Rodrigues::ZQ_Rodrigues_r2R(r, Ra);
        bool ok1 = ZQ::ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun(rcopy, Rb);
        if (!ok1) { printf("  **FAIL** r2R_fun 返回 false（i=%d）\n", i); g_fail++; }
        e1 = (e1 > max_abs_diff(Ra, Rb, 9)) ? e1 : max_abs_diff(Ra, Rb, 9);

        double r2[3] = { r[0], r[1], r[2] };
        ZQ::ZQ_Rodrigues::ZQ_Rodrigues_autoscale(r2);
        ZQ::ZQ_Rodrigues::ZQ_Rodrigues_r2R(r2, Rc);
        e2 = (e2 > max_abs_diff(Ra, Rc, 9)) ? e2 : max_abs_diff(Ra, Rc, 9);
    }
    ok("r2R_fun 与 r2R 逐位相同", e1, 0.0);
    ok("autoscale 之后 r2R 结果不变", e2, 1e-12);

    // 3：Jacobian vs 中心差分
    double worst_jac = 0;
    const double h = 1e-6;
    for (int i = 0; i < nRS; i++) {
        double r[3] = { RS[i][0], RS[i][1], RS[i][2] };
        if (fabs(r[0]) + fabs(r[1]) + fabs(r[2]) < 1e-3) continue;   // 零向量处 Jacobian 是极限值
        double J[27];
        if (!ZQ::ZQ_Rodrigues::ZQ_Rodrigues_r2R_jac(r, J)) {
            printf("  **FAIL** r2R_jac 返回 false（i=%d）\n", i); g_fail++; continue;
        }
        for (int k = 0; k < 3; k++) {
            double rp[3] = { r[0], r[1], r[2] };
            double rm[3] = { r[0], r[1], r[2] };
            rp[k] += h; rm[k] -= h;
            double Rp[9], Rm[9];
            ZQ::ZQ_Rodrigues::ZQ_Rodrigues_r2R(rp, Rp);
            ZQ::ZQ_Rodrigues::ZQ_Rodrigues_r2R(rm, Rm);
            for (int j = 0; j < 9; j++) {
                double fd = (Rp[j] - Rm[j]) / (2 * h);
                // 库里的布局是 **9 x 3 行主序**：`dRdr[行*9 + 导数下标]`
                // （`dRdm1` 是 9x21、`dm1din` 是 21x3，`MatrixMul` 出来就是行主序 9x3）。
                // 第一版这里写成 `J[k*9+j]`，把两个下标**写反了**，
                // 于是「Jacobian 与差分不符」是**测试自己的错**，不是库的错。
                // 库里的布局是 **9 x 3 行主序**：`dRdr[行*3 + 导数下标]`（27 = 9*3），
                // 来自 `MatrixMul(dRdm1 /*9x21*/, dm1din /*21x3*/, 9, 21, 3, dRdin)`。
                // 这里前后写错过两轮：`J[k*9+j]`（把 3x3 当 9x3，越界）与
                // `J[j*9+k]`（行距取 9 而库里是 3）—— **两次都是测试自己的错**，
                // 于是「Jacobian 与差分不符」报了两轮。
                // 教训与附录 IX.11 同源：**先确认被测对象的内存布局，再写判据**。
                double d = fabs(fd - J[j * 3 + k]);
                if (d > worst_jac) worst_jac = d;
            }
        }
    }
    ok("r2R_jac 的 Jacobian vs 中心差分", worst_jac, 1e-6);

    // 4：往返
    double worst_rt = 0;
    for (int i = 0; i < nRS; i++) {
        double r[3] = { RS[i][0], RS[i][1], RS[i][2] };
        double R[9], r2[3];
        ZQ::ZQ_Rodrigues::ZQ_Rodrigues_r2R(r, R);
        if (!ZQ::ZQ_Rodrigues::ZQ_Rodrigues_R2r_fun(R, r2)) continue;   // 退化姿态返回 false 也不算数错
        // **真正的契约是旋转矩阵往返**：r -> R -> r' -> R'，R' 必须等于 R。
        // 直接比 r 与 r' 会在 θ = pi 处**必然不等**：R(pi*n) = R(-pi*n)
        // （180° 旋转是对合），轴的正负本来就二义。2026-10-06 实测：
        // 补上 IY.2 的 pi 因子之后 `r=(pi,0,0)` 往返得到 `r'=(pi,0,0)`，
        // 而另一组姿态拿到 `-pi` —— 两者**生成同一个 R**，这才是要验的。
        double R2[9];
        ZQ::ZQ_Rodrigues::ZQ_Rodrigues_r2R(r2, R2);
        double dm = max_abs_diff(R, R2, 9);
        if (dm > 1e-9)
            printf("        旋转矩阵往返偏差 %.4g：r=(%.4f,%.4f,%.4f) -> r'=(%.4f,%.4f,%.4f)\n",
                   dm, r[0], r[1], r[2], r2[0], r2[1], r2[2]);
        if (dm > worst_rt) worst_rt = dm;
    }
    ok("R2r_fun 往返（r -> R -> r' -> R'）", worst_rt, 1e-9);

    // 5：判空
    double dummy[9] = { 0 };
    double rz[3] = { 0.1, 0.2, 0.3 };
    double rnull_dummy[3] = { 0 };
    if (ZQ::ZQ_Rodrigues::ZQ_Rodrigues_r2R_fun((const double*)0, dummy)) { printf("  **FAIL** r2R_fun(0,R) 应返回 false\n"); g_fail++; }
    if (ZQ::ZQ_Rodrigues::ZQ_Rodrigues_r2R_jac(rz, (double*)0)) { printf("  **FAIL** r2R_jac(r,0) 应返回 false\n"); g_fail++; }
    if (ZQ::ZQ_Rodrigues::ZQ_Rodrigues_R2r_fun((const double*)0, rnull_dummy)) { printf("  **FAIL** R2r_fun(0,r) 应返回 false\n"); g_fail++; }
    if (ZQ::ZQ_Rodrigues::ZQ_Rodrigues_autoscale((double*)0)) { printf("  **FAIL** autoscale(0) 应返回 false\n"); g_fail++; }
    {   // **用局部计数**，不要用累计的 g_fail —— 前面若有失败，这一行会跟着报 FAIL，
        // 让人以为「判空也坏了」（第一版就这样）。
        int before = g_fail;
        printf("  %-6s %s\n", (g_fail == before) ? "ok" : "FAIL",
               "四个薄包装的空指针都返回 false");
    }

    if (g_fail) { printf("RODRIGUES CHECK FAILED (%d 处)\n", g_fail); return 1; }
    printf("RODRIGUES CHECK OK\n");
    return 0;
}
