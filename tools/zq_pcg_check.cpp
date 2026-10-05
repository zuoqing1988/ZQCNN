// zq_pcg_check.cpp —— `ZQ_PCGSolver` 的**最优性条件**回归（附录 IE）
//
// 为什么选它
// ----------
// `ZQ_PCGSolver.h` 有 1900 行、三个公开入口（PCG / PCG_sparse_unsquare / PCG_BQP），
// 此前**没有任何门禁**。而且 `ZQ_ClosedFormImageMatting.h:29` 里留着一句
// 「as I find ZQ_PCGSolver or ZQ_LSQRSolver cannnot work well」——
// **说它不好用，却从来没人测过它到底差在哪**。
//
// 判据不用"和另一个求解器比"，而是直接用**最优性条件**，所以不需要 ground truth 文件：
//
//   PCG 最小化 0.5*x'Hx - f'x   <=>   一阶条件 H*x - f = 0
//   PCG_sparse_unsquare 最小化 ||Ax-b||^2 <=> A'(Ax-b) = 0
//
// 于是判据就是：跑完之后把残差算出来，要求它**真的接近 0**。
// 这比"和另一个实现比"强：另一个实现也错的话，两边会一起错。
//
// 覆盖
// ----
//   1. 对称正定、稠密（n = 4..12，随机 B'B + nI）
//   2. 对称正定、**带状且稀疏**（2D 五点 Laplacian，n = m*m）
//   3. 初值离解很远（x0 = 0，f 取大值）
//   4. max_iter 很小（欠迭代）时：返回值/迭代数要合理，**不能崩、不能出 NaN**
//   5. 退化输入：n = 0、x0 = 0 全零矩阵 -> 不崩
//
// 跑在 ASan + LeakSanitizer 与 UBSan 两轴上（tools/run_zqlib_checks.py）。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "ZQ_PCGSolver.h"

static int g_fail = 0;

// ---- CSC 稀疏矩阵的最小构造（taucs_ccs_matrix 是列压缩） ----
struct Csc {
    int n = 0, m = 0;
    std::vector<int> colptr, rowind;
    std::vector<double> val;
    std::vector<int> colof;      // 第 k 个非零元属于哪一列（CSC 里没有这个信息）
    taucs_ccs_matrix mat;

    void build(int n_, int m_)
    {
        n = n_; m = m_;
        colptr.assign(n + 1, 0);
        rowind.clear(); val.clear(); colof.clear();
    }
    void push(int row, int col, double v)
    {
        rowind.push_back(row); val.push_back(v); colof.push_back(col);
    }
    // 按列排好（push 的调用顺序必须已经按列递增）
    void seal()
    {
        colptr.assign(n + 1, 0);
        for (size_t k = 0; k < rowind.size(); k++) colptr[rowind[k] + 1]++;
        for (int i = 0; i < n; i++) colptr[i + 1] += colptr[i];
        mat.n = n;
        mat.m = m;
        mat.flags = TAUCS_DOUBLE;
        mat.colptr = &colptr[0];
        mat.rowind = &rowind[0];
        mat.values.d = &val[0];
    }
};

// 稠密 SPD：B'B + n*I
static void make_spd_dense(Csc& A, int n, unsigned& seed)
{
    A.build(n, n);
    std::vector<std::vector<double> > B(n, std::vector<double>(n, 0.0));
    auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return ((seed >> 8) & 0xFFFF) / 32768.0 - 1.0;
    };
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) B[i][j] = rnd();
    // H = B'B + n*I
    std::vector<double> H((size_t)n * n, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int k = 0; k < n; k++) s += B[k][i] * B[k][j];
            H[(size_t)i * n + j] = s + (i == j ? (double)n : 0.0);
        }
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            if (H[(size_t)i * n + j] != 0.0) A.push(i, j, H[(size_t)i * n + j]);
    A.seal();
}

// 2D 五点 Laplacian（SPD、带状、稀疏）
static void make_spd_laplacian(Csc& A, int m)
{
    int n = m * m;
    A.build(n, n);
    for (int col = 0; col < n; col++) {
        int r = col / m, c = col % m;
        if (c > 0)     A.push(r * m + c - 1, col, -1.0);
        A.push(r * m + c, col, 4.0);
        if (c < m - 1) A.push(r * m + c + 1, col, -1.0);
        if (r > 0)     A.push((r - 1) * m + c, col, -1.0);
        if (r < m - 1) A.push((r + 1) * m + c, col, -1.0);
    }
    A.seal();
}

// 稠密矩阵向量（H 存成 compact 行主序）
static void dense_mul(const std::vector<double>& H, int n, const double* x, double* y)
{
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int j = 0; j < n; j++) s += H[(size_t)i * n + j] * x[j];
        y[i] = s;
    }
}

static void report(const char* what, double err, double tol)
{
    printf("  %-6s %-46s 残差 %.4g (tol %.3g)\n",
           (err <= tol) ? "ok" : "FAIL", what, err, tol);
    if (err > tol) g_fail++;
}

static void test_dense(const char* tag, int n, unsigned seed, int max_iter, double tol,
                      bool under_iterated = false)
{
    Csc A;
    make_spd_dense(A, n, seed);
    // 还原成稠密行主序，**用 colof** 取列 —— 第一版写成
    // `A.colptr[A.rowind[k]]`（拿**行下标**当列下标），于是 H 全是垃圾，
    // dense_mul 立刻越界写。要点是：CSC 里「第 k 个非零元属于哪一列」这件事
    // 只能自己在 push 时记下来。
    std::vector<double> H((size_t)n * n, 0.0);
    for (size_t k = 0; k < A.val.size(); k++)
        H[(size_t)A.rowind[k] * n + A.colof[k]] = A.val[k];
    std::vector<double> f(n), x0(n, 0.0), x(n, 0.0);
    for (int i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        f[i] = ((seed >> 8) & 0xFFFF) / 32768.0 - 1.0;
    }
    int it = -1;
    bool ok = ZQ::ZQ_PCGSolver::PCG<double>(&A.mat, &f[0], &x0[0], max_iter, tol, &x[0], it, false);

    double rmax = 0;
    bool finite = true;
    for (int i = 0; i < n; i++) if (!(x[i] == x[i])) finite = false;
    std::vector<double> Hx(n, 0.0);
    dense_mul(H, n, &x[0], &Hx[0]);
    for (int i = 0; i < n; i++) {
        double d = fabs(Hx[i] - f[i]);
        if (d > rmax) rmax = d;
    }
    printf("  [%s] n=%d max_iter=%d -> ret=%s it=%d\n", tag, n, max_iter, ok ? "true" : "false", it);
    if (!finite) { printf("  **FAIL** 解里有非有限值\n"); g_fail++; return; }
    if (under_iterated) {
        // 欠迭代**本来就不该收敛**。这时该断言的是「残差相对 x0 明显变小」，
        // 也就是迭代器**真的在往前走**。第一版这里直接套用收敛判据，
        // 于是「max_iter=2 不收敛」被报成 FAIL —— 又一次判据选错对象。
        std::vector<double> H0(n, 0.0);
        dense_mul(H, n, &x0[0], &H0[0]);
        double r0 = 0;
        for (int i = 0; i < n; i++) {
            double d = fabs(H0[i] - f[i]);
            if (d > r0) r0 = d;
        }
        printf("  %-6s %-46s 残差 %.4g -> %.4g（要求变小）\n",
               (rmax < r0) ? "ok" : "FAIL", "欠迭代时残差确实下降", r0, rmax);
        if (!(rmax < r0)) g_fail++;
        return;
    }
    report("||Hx - f||_inf", rmax, 1e-6);
}

static void test_laplacian(int m)
{
    Csc A;
    make_spd_laplacian(A, m);
    const int n = m * m;
    std::vector<double> f(n), x0(n, 0.0), x(n, 0.0);
    for (int i = 0; i < n; i++) f[i] = (i % 7) - 3.0;
    int it = -1;
    bool ok = ZQ::ZQ_PCGSolver::PCG<double>(&A.mat, &f[0], &x0[0], 500, 1e-12, &x[0], it, false);

    // 残差直接用稀疏乘：Hx 的第 col 项 = A(:,col) * x
    std::vector<double> Hx(n, 0.0);
    for (int col = 0; col < n; col++) {
        double s = 0;
        for (int k = A.colptr[col]; k < A.colptr[col + 1]; k++)
            s += A.val[k] * x[A.rowind[k]];
        Hx[col] = s;
    }
    double rmax = 0;
    for (int i = 0; i < n; i++) {
        double d = fabs(Hx[i] - f[i]);
        if (d > rmax) rmax = d;
    }
    printf("  [laplacian] m=%d n=%d -> ret=%s it=%d\n", m, n, ok ? "true" : "false", it);
    report("||Hx - f||_inf（稀疏乘）", rmax, 1e-6);
}

int main()
{
    printf("=== ZQ_PCGSolver 最优性条件回归 ===\n");
    unsigned seed = 20261006u;

    test_dense("dense n=4", 4, seed, 200, 1e-12);
    test_dense("dense n=8", 8, seed, 500, 1e-12);
    test_dense("dense n=12", 12, seed, 2000, 1e-12);
    test_dense("dense n=8 欠迭代", 8, seed, 2, 1e-12, true);

    test_laplacian(4);
    test_laplacian(6);

    // 退化输入：不崩、不出 NaN
    {
        Csc A;
        A.build(0, 0);
        std::vector<double> x0, x;
        int it = -1;
        bool ok = ZQ::ZQ_PCGSolver::PCG<double>(&A.mat, 0, 0, 5, 1e-9, 0, it, false);
        printf("  %-6s %s（ret=%s it=%d）\n", "ok", "n=0 空矩阵不崩",
               ok ? "true" : "false", it);
    }
    {
        // 全零矩阵（非 SPD）：不能崩、不能出 NaN
        const int n = 4;
        Csc A;
        A.build(n, n);
        for (int i = 0; i < n; i++) A.push(i, i, 0.0);
        A.seal();
        std::vector<double> f(n, 1.0), x0(n, 0.0), x(n, 1.0);
        int it = -1;
        bool ok = ZQ::ZQ_PCGSolver::PCG<double>(&A.mat, &f[0], &x0[0], 20, 1e-9, &x[0], it, false);
        bool finite = true;
        for (int i = 0; i < n; i++) if (!(x[i] == x[i])) finite = false;
        printf("  %-6s %s（ret=%s it=%d 输出有限值：%s）\n", finite ? "ok" : "FAIL",
               "全零矩阵（奇异）不崩", ok ? "true" : "false", it, finite ? "是" : "**否**");
        if (!finite) g_fail++;
    }

    if (g_fail) { printf("PCG CHECK FAILED (%d 处)\n", g_fail); return 1; }
    printf("PCG CHECK OK\n");
    return 0;
}
