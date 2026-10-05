// zq_calibration_check.cpp —— `ZQ_Calibration` 的**带 ground truth** 数值回归（附录 IB）
//
// 起因（附录 IY.1）
// ------------------
// `ZQ_Calibration.h` 有 4500 多行，因为引用了四个**从来没存在过**的
// `ZQ_Rodrigues` 成员而**整个头编不过**。补齐之后（IY），它第一次能被编、被跑，
// 于是第一次能验。
//
// 为什么这个头值得验
// ------------------
// 它是仓库里唯一一份**有 ground truth 可构造**的数值算法：
// 投影模型 `proj_no_distortion` 是公开的，而标定入口
// `calib_estimate_no_distortion_with_init` 的参数含义完全由
// `_calib_estimate_no_distortion_func` 写死：
//
//     p = [alpha, beta, u0, v0, <每相机 r(3), t(3)>]
//     A  = [alpha 0 u0; 0 beta v0; 0 0 1]
//     xc = R·X3 + t ;  xd = A·xc
//     X2 = (xd0*xd2/(xd2^2+eps^2), xd1*xd2/(xd2^2+eps^2))
//
// 于是**自己造一套真值**（内参 + 两个相机的外参 + 3D 点），
// 用同一个 `proj_no_distortion` 生成 2D 观测，再喂回标定入口，
// 看它能不能把真值还原回来。X3 是**所有相机共用的那一份**（函数里
// `proj_no_distortion(N,A,R,t,X3,...)` 对每个相机传的是**同一个 X3 指针**），
// X2 才是按相机分段的（`X2[N*2*cc+i]`）—— 这条约定也是从代码读出来的。
//
// 覆盖
// ----
//   1. 初始化 = 真值 -> 必须零残差收敛，且解出的参数与真值一致
//   2. **初始化被扰动**（内参 2%、主点 3 px、外参一个小角度）
//      -> 优化器必须**自己走回真值**。这一条比第 1 条强得多：
//      它同时验了目标函数**和解析 Jacobian**（`_..._jac`）。
//   3. 退化输入：3D 点落在相机后面 / 点数不足 -> 不崩
//
// 跑在 ASan + LeakSanitizer 与 UBSan 两轴上（tools/run_zqlib_checks.py）。
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include "ZQ_Calibration.h"
#include "ZQ_Rodrigues.h"

static int g_fail = 0;

static const int N_PTS = 24;
static const int N_CAMS = 2;

struct Truth {
    double intr[4];                 // alpha, beta, u0, v0
    double rT[N_CAMS * 6];          // r(3), t(3) per camera
};

static void build_truth(Truth& T)
{
    T.intr[0] = 520.0; T.intr[1] = 505.0; T.intr[2] = 320.0; T.intr[3] = 240.0;
    // cam0: 单位旋转、零平移
    T.rT[0] = 0.0; T.rT[1] = 0.0; T.rT[2] = 0.0;
    T.rT[3] = 0.0; T.rT[4] = 0.0; T.rT[5] = 0.0;
    // cam1: 小角度旋转 + 平移
    T.rT[6]  = 0.05; T.rT[7] = -0.08; T.rT[8] = 0.02;
    T.rT[9]  = 0.30; T.rT[10] = -0.15; T.rT[11] = 0.50;
}

// 造一套**两台相机都看得到**的 3D 点（确定性，不用随机）
static void build_points(std::vector<double>& X3)
{
    X3.assign((size_t)N_PTS * 3, 0.0);
    for (int i = 0; i < N_PTS; i++) {
        int a = i % 4, b = (i / 4) % 4, c = i / 16;
        X3[(size_t)i * 3 + 0] = -1.5 + 1.0 * a;
        X3[(size_t)i * 3 + 1] = -0.9 + 0.6 * b;
        X3[(size_t)i * 3 + 2] =  4.0 + 0.5 * c;
    }
}

static void project_all(const Truth& T, const std::vector<double>& X3,
                        std::vector<double>& X2)
{
    double A[9] = { T.intr[0], 0, T.intr[2],
                    0, T.intr[1], T.intr[3],
                    0, 0, 1 };
    X2.assign((size_t)N_CAMS * N_PTS * 2, 0.0);
    for (int cc = 0; cc < N_CAMS; cc++) {
        double R[9], t[3];
        ZQ::ZQ_Rodrigues::ZQ_Rodrigues_r2R(T.rT + cc * 6, R);
        memcpy(t, T.rT + cc * 6 + 3, sizeof(double) * 3);
        ZQ::ZQ_Calibration::proj_no_distortion(N_PTS, A, R, t, &X3[0],
                                               &X2[(size_t)cc * N_PTS * 2], 1e-12);
    }
}

static void report(const char* what, double err, double tol)
{
    printf("  %-6s %-44s 最大偏差 %.4g (tol %.3g)\n",
           (err <= tol) ? "ok" : "FAIL", what, err, tol);
    if (err > tol) g_fail++;
}

static void run_case(const char* name, const Truth& T, double pert_scale,
                     const std::vector<double>& X3, const std::vector<double>& X2)
{
    std::vector<double> intr(T.intr, T.intr + 4);
    std::vector<double> rT(T.rT, T.rT + N_CAMS * 6);
    // 扰动初始化（pert_scale = 0 表示用真值当初值）
    if (pert_scale > 0) {
        intr[0] *= (1.0 + 0.02 * pert_scale);
        intr[1] *= (1.0 - 0.015 * pert_scale);
        intr[2] += 3.0 * pert_scale;
        intr[3] -= 2.5 * pert_scale;
        rT[0] += 0.01 * pert_scale; rT[7] -= 0.012 * pert_scale;
        rT[10] += 0.02 * pert_scale;
    }
    double avg_err = -1.0;
    bool ok = ZQ::ZQ_Calibration::calib_estimate_no_distortion_with_init(
        N_CAMS, N_PTS, &X3[0], &X2[0], 60, &intr[0], &rT[0], avg_err, 1e-12);
    printf("  [%s] 返回 %s，avg_err_square = %.6g\n", name, ok ? "true" : "false", avg_err);
    if (!ok) { printf("  **FAIL** %s：标定入口返回 false\n", name); g_fail++; return; }
    double ei = 0;
    for (int i = 0; i < 4; i++) {
        double d = fabs(intr[i] - T.intr[i]);
        if (d > ei) ei = d;
    }
    double er = 0;
    for (int i = 0; i < N_CAMS * 6; i++) {
        double d = fabs(rT[i] - T.rT[i]);
        if (d > er) er = d;
    }
    report("内参相对真值", ei, 0.05);
    report("外参相对真值", er, 0.01);
    double resid = fabs(avg_err);
    report("重投影残差 avg_err_square", resid, 1e-4);
}

int main()
{
    printf("=== ZQ_Calibration 带 ground truth 的回归 ===\n");
    Truth T;
    build_truth(T);
    std::vector<double> X3, X2;
    build_points(X3);
    project_all(T, X3, X2);

    // 1：真值当初值
    run_case("init=truth", T, 0.0, X3, X2);
    // 2：扰动初值（验目标函数 + 解析 Jacobian）
    run_case("init=perturbed", T, 1.0, X3, X2);
    run_case("init=perturbed x2", T, 2.0, X3, X2);

    // 3：退化输入 —— 3D 点全部落在相机后面
    {
        std::vector<double> X3b(X3), X2b((size_t)N_CAMS * N_PTS * 2, 0.0);
        for (size_t i = 0; i < X3b.size(); i += 3) X3b[i + 2] = -X3b[i + 2] - 10.0;
        std::vector<double> intr(T.intr, T.intr + 4), rT(T.rT, T.rT + N_CAMS * 6);
        double avg_err = -1.0;
        bool ok = ZQ::ZQ_Calibration::calib_estimate_no_distortion_with_init(
            N_CAMS, N_PTS, &X3b[0], &X2b[0], 5, &intr[0], &rT[0], avg_err, 1e-12);
        bool finite = true;
        for (int i = 0; i < 4; i++) if (!(intr[i] == intr[i])) finite = false;
        for (int i = 0; i < N_CAMS * 6; i++) if (!(rT[i] == rT[i])) finite = false;
        printf("  %-6s %s（返回 %s，解全为有限值：%s）\n", (ok || !finite) ? "ok" : "FAIL",
               "3D 点全在相机后面", ok ? "true" : "false", finite ? "是" : "**否**");
        if (!finite) g_fail++;
    }

    // 3b：位姿估计（内参固定、只解 rT）—— 同一个投影模型，构造同样的真值
    {
        double A[9] = { T.intr[0], 0, T.intr[2],
                        0, T.intr[1], T.intr[3],
                        0, 0, 1 };
        double rT[6];
        memcpy(rT, T.rT, sizeof(double) * 6);          // cam0 的真值
        double pert[6];
        memcpy(pert, rT, sizeof(rT));
        pert[0] += 0.02; pert[1] -= 0.015; pert[5] += 0.05;   // 扰动初值
        double avg_err = -1.0;
        bool ok = ZQ::ZQ_Calibration::pose_estimate_no_distortion_with_init(
            N_PTS, &X3[0], &X2[0], 60, T.intr, pert, avg_err, 1e-12);
        double e = 0;
        for (int i = 0; i < 6; i++) {
            double d = fabs(pert[i] - rT[i]);
            if (d > e) e = d;
        }
        printf("  [pose] ret=%s avg_err_square=%.6g\n", ok ? "true" : "false", avg_err);
        report("pose 位姿相对真值", ok ? e : 1e9, 0.01);
        report("pose 重投影残差", fabs(avg_err), 1e-4);
    }

    // 4：点数为 0
    {
        std::vector<double> dummy3(3, 1.0), dummy2(2, 1.0);
        std::vector<double> intr(T.intr, T.intr + 4), rT(T.rT, T.rT + N_CAMS * 6);
        double avg_err = -1.0;
        bool ok = ZQ::ZQ_Calibration::calib_estimate_no_distortion_with_init(
            N_CAMS, 0, &dummy3[0], &dummy2[0], 3, &intr[0], &rT[0], avg_err, 1e-12);
        printf("  %-6s %s（返回 %s）\n", ok ? "ok" : "ok",
               "点数为 0 不崩", ok ? "true" : "false");
    }

    if (g_fail) { printf("CALIBRATION CHECK FAILED (%d 处)\n", g_fail); return 1; }
    printf("CALIBRATION CHECK OK\n");
    return 0;
}
