// zq_mathbase_check.cpp —— ZQ_MathBase::SVD_Decompose / Cond_by_double_svd 回归测试
//
// 起因（audit_k3_20261001.md 附录 AT.5 / AT.6）
// ---------------------------------------------
// tools/warn_sweep_zqlib.py 的 -Wmisleading-indentation 在 ZQ_MathBase.h:764 报了一处：
//
//   if( (k < nrt) && ( e[k] != 0 ) )
//       for(int j = k+1; j < n; j++ ) { ...V 的 Householder 累加... }   // 775 行闭合
//       for(int i = 0; i < n; ++i ) V[i][k] = 0;                        // 777 行
//       V[k][k] = 1;                                                    // 779 行
//
// 777/779 的缩进和 if 里的 for **一样深**，第一眼看着像"零列这一步本该在 if 里"。
// 为了判定这到底算不算 bug，把 SVD_Decompose 的输出约定整个查了一遍 —— 查的过程中
// 发现了**另外**一个真 bug（AT.5），而这个缩进警告本身最后证明是虚惊（AT.6）。
//
// SVD_Decompose 的输出约定（源码里没有任何注释，附录 AT.5 补在这里）
// -------------------------------------------------------------------
//   sdim = min(row, col)
//   Smat : sdim x sdim，行距 **sdim**（1054-1056 行：memset sdim*sdim，
//          Smat[i*sdim+i] = S[i]）
//   Umat : 行距 sdim，写满 max(row,col) x sdim
//   Vmat : 行距 sdim，写满 sdim x sdim
//   row >= col 时 Umat 拿左奇异向量、Vmat 拿右奇异向量；
//   row <  col 时两者**对调**（1059-1095 的两个分支），
//   但最终 A == Umat * diag(Smat) * Vmat^T 这个式子在两种情况下都成立。
//
// AT.5 的 bug：Cond_by_double_svd 用 `S[(N-1)*col + (N-1)]` 取最小奇异值
//   （1141 / 1148 / 1175 / 1182 四处）。行距是 **N == sdim**，不是 col。
//   col == N 时碰巧相等，所以方阵和宽矩阵一直是对的；
//   row < col（高矩阵）时读到的是 memset 之外的**恒为 0** 的位置，
//   于是 is_singular 永远为 true、返回值永远是 1e32。

#include "zqlib_msvc_shim.h"   // 必须在下一行之前（ZQ_MathBase.h 用了 __min）
#include "ZQ_MathBase.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

static int g_fail = 0;
static void CHECK(bool cond, const char* what)
{
    printf("%-64s %s\n", what, cond ? "ok" : "FAIL");
    if (!cond) g_fail++;
}

// A == Umat * diag(Smat) * Vmat^T 的相对重建误差。
// 行距一律用 sdim —— 这正是 AT.5 里被搞错的那个量，写错的话本测试自己也会越界。
static double rel_recon_err(int m, int n, const double* A,
                            const double* U, const double* S, const double* V)
{
    int sdim = m < n ? m : n;
    double se = 0, sn = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double acc = 0;
            for (int k = 0; k < sdim; k++)
                acc += U[i * sdim + k] * S[k * sdim + k] * V[j * sdim + k];
            double d = A[i * n + j] - acc;
            se += d * d;
            sn += A[i * n + j] * A[i * n + j];
        }
    }
    return std::sqrt(se / sn);
}

// ||M^T M - I||_max。M 是 r 行 c 列、行距 c 的行主序。
static double ortho_err(const double* M, int r, int c)
{
    double worst = 0;
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < c; j++) {
            double acc = 0;
            for (int k = 0; k < r; k++) acc += M[k * c + i] * M[k * c + j];
            double want = (i == j) ? 1.0 : 0.0;
            double d = std::fabs(acc - want);
            if (d > worst) worst = d;
        }
    }
    return worst;
}

static void run_svd_case(const char* name, int m, int n, const double* A)
{
    int sdim = m < n ? m : n;
    int before = g_fail;
    printf("\n--- SVD %s (%d x %d, sdim=%d) ---\n", name, m, n, sdim);
    // Umat / Vmat 各给 max*sdim 个元素。行距恒为 sdim，但**写满的范围**在两个分支里
    // 不一样：row>=col 时 Vmat 只写 sdim*sdim，row<col 时 Vmat 要写到 col*sdim。
    // 按 sdim*sdim 分配会在高矩阵上越界 —— 本测试第一版就是这么写的，ASan 直接
    // 报 ZQ_MathBase.h:1084 heap-buffer-overflow。那**不是**库的 bug，是这里按方阵的
    // 直觉分配小了；库内两处调用方给的是 row*row / col*col，够用。
    int cap = (m > n ? m : n) * sdim;
    std::vector<double> U(cap, 0.0), S(sdim * sdim, 0.0), V(cap, 0.0);
    bool ok = ZQ::ZQ_MathBase::SVD_Decompose(A, m, n, &U[0], &S[0], &V[0]);
    CHECK(ok, "SVD_Decompose 返回 true");
    if (!ok) return;

    bool desc = true;
    for (int k = 1; k < sdim; k++)
        if (S[(k - 1) * sdim + (k - 1)] < S[k * sdim + k]) desc = false;
    CHECK(desc, "奇异值降序");

    double e = rel_recon_err(m, n, A, &U[0], &S[0], &V[0]);
    printf("   相对重建误差 = %.3e\n", e);
    CHECK(e < 1e-11, "A == Umat * diag(Smat) * Vmat^T");

    // Umat 是 row x sdim；Vmat 则是 **max(row,col) x sdim**（行距恒为 sdim）：
    //   row >= col 时是 col x col，row < col 时是 col x row。
    // 后者容易看错 —— 按方阵的直觉去查 Vmat^T Vmat 会得到一个假失败。
    double eu = ortho_err(&U[0], m, sdim);
    double ev = ortho_err(&V[0], (m > n ? m : n), sdim);
    printf("   ||U^T U - I||max = %.3e   ||V V^T - I||max = %.3e\n", eu, ev);
    CHECK(eu < 1e-11, "Umat 正交");
    CHECK(ev < 1e-11, "Vmat 正交");

    if (g_fail > before) printf("   ^ 本用例 %d 条断言失败\n", g_fail - before);
}

int main()
{
    printf("ZQ_MathBase 回归测试\n");

    // ---------- 1. 奇异值本身对不对 ----------
    // A^T A = [[66,78,97],[78,93,116],[97,116,145]]
    // numpy.linalg.eigh 给出的 sigma: 17.41250517, 0.87516135, 0.19686652
    {
        static double A[3 * 3] = { 1, 2, 3, 4, 5, 6, 7, 8, 10 };
        std::vector<double> U(9, 0.0), S(9, 0.0), V(9, 0.0);
        ZQ::ZQ_MathBase::SVD_Decompose(A, 3, 3, &U[0], &S[0], &V[0]);
        static const double REF[3] = { 17.41250517, 0.87516135, 0.19686652 };
        double sd = 0;
        for (int i = 0; i < 3; i++)
            sd += std::fabs(S[i * 3 + i] - REF[i]);
        printf("\n--- SVD 奇异值对照 numpy (3x3) ---\n   最大绝对偏差 = %.3e\n", sd);
        CHECK(sd < 1e-6, "奇异值与 numpy.linalg.eigh 一致");

        double maxdev = 0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                double want = (i == j) ? 1.0 : 0.0;
                double d = std::fabs(V[i * 3 + j] - want);
                if (d > maxdev) maxdev = d;
            }
        printf("   max |V - I| = %.3e\n", maxdev);
        CHECK(maxdev > 1e-3, "V 不是单位阵（奇异向量确实算出来了）");
    }

    // ---------- 2. 各种形状 ----------
    {
        static double A[3 * 3] = { 1, 2, 3, 4, 5, 6, 7, 8, 10 };
        run_svd_case("方阵", 3, 3, A);
    }
    {
        static double A[4 * 3] = { 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 15 };
        run_svd_case("宽矩阵 4x3", 4, 3, A);
    }
    {
        // 高矩阵 3x4 —— 正是 AT.5 那个 bug 的触发形状
        static double A[3 * 4] = { 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17 };
        run_svd_case("高矩阵 3x4", 3, 4, A);
    }
    {
        // 秩亏：第 3 行 = 第 1 行 + 第 2 行
        static double A[3 * 3] = { 1, 2, 3, 0, 1, 4, 1, 3, 7 };
        run_svd_case("秩亏 3x3", 3, 3, A);
    }

    // ---------- 3. Cond_by_double_svd（AT.5 的正主） ----------
    // 条件数 = S[0] / S[sdim-1]。对两个形状各测一次：
    //   4x3 宽矩阵 —— col == sdim，旧代码碰巧是对的
    //   3x4 高矩阵 —— sdim != col，旧代码读错位置
    {
        printf("\n--- Cond_by_double_svd ---\n");
        struct Case { const char* name; int row; int col; const double* a; };
        static double W[4 * 3] = { 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 15 };
        static double T[3 * 4] = { 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17 };
        Case cases[2] = { { "宽矩阵 4x3 (col==sdim, 旧代码碰巧对)", 4, 3, W },
                          { "高矩阵 3x4 (sdim!=col, 旧代码读错)", 3, 4, T } };
        for (int ci = 0; ci < 2; ci++) {
            bool succ = false, sing = true;
            double cond = ZQ::ZQ_MathBase::Cond_by_double_svd(cases[ci].a,
                                                             cases[ci].row,
                                                             cases[ci].col,
                                                             succ, sing);
            printf("   %-40s succ=%d is_singular=%d cond=%.6g\n",
                   cases[ci].name, (int)succ, (int)sing, cond);
            CHECK(succ, "  Succ 为真");
            CHECK(!sing, "  非奇异矩阵不该被判成奇异（AT.5）");
            CHECK(cond > 0 && cond < 1e31, "  条件数是有限正数（AT.5）");
        }
        // 秩亏矩阵：S[sdim-1] 真的是 0，这时才该报奇异
        static double D[3 * 3] = { 1, 2, 3, 0, 1, 4, 1, 3, 7 };
        bool succ = false, sing = false;
        double cond = ZQ::ZQ_MathBase::Cond_by_double_svd(D, 3, 3, succ, sing);
        printf("   %-40s succ=%d is_singular=%d cond=%.6g\n",
               "秩亏 3x3 (真奇异)", (int)succ, (int)sing, cond);
        CHECK(sing, "  秩亏矩阵应当被判成奇异");
    }

    if (g_fail) {
        printf("\n%d 条断言失败\n", g_fail);
        return 1;
    }
    printf("\n全部通过\n");
    return 0;
}
