/* NCHW（layers_c）scalaroperation 门禁 —— 附录 CQ
 *
 * 覆盖面：`zq_cnn_scalaroperation_32f_align_c.c` 里 **38 个 32f 真实符号**
 *   7 个运算（add / mul / max / min / rminus / rdiv / pow）
 *   × 2 种形式（out-of-place / in-place）× 3 种对齐
 *   （`pow` 只有 align0 一份）
 * 全部是**逐元素二元运算**，此前零覆盖。
 *
 * 这一族也在 CQ 结构扫描的「手写标量 + SIMD 模板」名单里 ——
 * CN / CO / CP 三处缺陷都出在这个结构上。
 *
 * 七个运算的真实算式（从 `zq_cnn_scalaroperation_32f_align_c.c` 里
 * 那七个 `#define zq_mm_operation_ps` **逐个抄**的，不是按名字推的）：
 *
 *   调用是 zq_mm_operation_ps(a_i, scalar_v)，即 **x = 输入、y = 标量**
 *   add     vaddq_f32(x, y)          out = in + s
 *   mul     vmulq_f32(x, y)          out = in * s
 *   max     vmaxq_f32(x, y)          out = max(in, s)
 *   min     vminq_f32(x, y)          out = min(in, s)
 *   rminus  vsubq_f32(y, x)          out = s - in      <-- **反向**
 *   rdiv    vdivq_f32(y, x)          out = s / in      <-- **反向**
 *   pow     (align0 手写标量)          out = pow(in, s)
 *
 * `r` 前缀是"反向"（reverse），**按名字推会整个搞反** ——
 * 这是本会话第五次「参数名不是语义」（CJ.2 / CH.2 / CI.3 / CP.5 各记过一次）。
 *
 * NCHW 布局：`offset(n,c,h,w) = n*sliceStep + h*widthStep + w*pixelStep + c`，
 * **pixelStep 就是 C**（附录 CO.5 记过我在这里栽过三次）。
 * 缓冲区一律 32 字节对齐（附录 CJ.4 / CP.5）。
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "ZQCNN/layers_c/zq_cnn_scalaroperation_32f_align_c.h"

typedef void (*F13)(float scalar, const float* in, int N, int H, int W, int C,
                    int ps, int ws, int ss, float* out, int ops, int ows, int oss);
typedef void (*F9)(float scalar, float* data, int N, int H, int W, int C,
                   int ps, int ws, int ss);

struct Entry { void* fn; int op; int inplace; int align; };

static const Entry g_entries[] = {
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_add_32f_align0), 0, 0, 1 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_add_32f_align128bit), 0, 0, 4 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_add_32f_align256bit), 0, 0, 8 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_add_inplace_32f_align0), 0, 1, 1 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_add_inplace_32f_align128bit), 0, 1, 4 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_add_inplace_32f_align256bit), 0, 1, 8 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_mul_32f_align0), 1, 0, 1 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_mul_32f_align128bit), 1, 0, 4 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_mul_32f_align256bit), 1, 0, 8 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_mul_inplace_32f_align0), 1, 1, 1 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_mul_inplace_32f_align128bit), 1, 1, 4 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_mul_inplace_32f_align256bit), 1, 1, 8 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_max_32f_align0), 2, 0, 1 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_max_32f_align128bit), 2, 0, 4 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_max_32f_align256bit), 2, 0, 8 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_max_inplace_32f_align0), 2, 1, 1 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_max_inplace_32f_align128bit), 2, 1, 4 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_max_inplace_32f_align256bit), 2, 1, 8 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_min_32f_align0), 3, 0, 1 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_min_32f_align128bit), 3, 0, 4 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_min_32f_align256bit), 3, 0, 8 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_min_inplace_32f_align0), 3, 1, 1 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_min_inplace_32f_align128bit), 3, 1, 4 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_min_inplace_32f_align256bit), 3, 1, 8 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_rminus_32f_align0), 4, 0, 1 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_rminus_32f_align128bit), 4, 0, 4 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_rminus_32f_align256bit), 4, 0, 8 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_rminus_inplace_32f_align0), 4, 1, 1 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_rminus_inplace_32f_align128bit), 4, 1, 4 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_rminus_inplace_32f_align256bit), 4, 1, 8 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_rdiv_32f_align0), 5, 0, 1 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_rdiv_32f_align128bit), 5, 0, 4 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_rdiv_32f_align256bit), 5, 0, 8 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_rdiv_inplace_32f_align0), 5, 1, 1 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_rdiv_inplace_32f_align128bit), 5, 1, 4 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_rdiv_inplace_32f_align256bit), 5, 1, 8 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_pow_32f_align0), 6, 0, 1 },
  { reinterpret_cast<void*>(zq_cnn_scalaroperation_pow_inplace_32f_align0), 6, 1, 1 },
};
static const char* g_name[] = {

  "zq_cnn_scalaroperation_add_32f_align0",
  "zq_cnn_scalaroperation_add_32f_align128bit",
  "zq_cnn_scalaroperation_add_32f_align256bit",
  "zq_cnn_scalaroperation_add_inplace_32f_align0",
  "zq_cnn_scalaroperation_add_inplace_32f_align128bit",
  "zq_cnn_scalaroperation_add_inplace_32f_align256bit",
  "zq_cnn_scalaroperation_mul_32f_align0",
  "zq_cnn_scalaroperation_mul_32f_align128bit",
  "zq_cnn_scalaroperation_mul_32f_align256bit",
  "zq_cnn_scalaroperation_mul_inplace_32f_align0",
  "zq_cnn_scalaroperation_mul_inplace_32f_align128bit",
  "zq_cnn_scalaroperation_mul_inplace_32f_align256bit",
  "zq_cnn_scalaroperation_max_32f_align0",
  "zq_cnn_scalaroperation_max_32f_align128bit",
  "zq_cnn_scalaroperation_max_32f_align256bit",
  "zq_cnn_scalaroperation_max_inplace_32f_align0",
  "zq_cnn_scalaroperation_max_inplace_32f_align128bit",
  "zq_cnn_scalaroperation_max_inplace_32f_align256bit",
  "zq_cnn_scalaroperation_min_32f_align0",
  "zq_cnn_scalaroperation_min_32f_align128bit",
  "zq_cnn_scalaroperation_min_32f_align256bit",
  "zq_cnn_scalaroperation_min_inplace_32f_align0",
  "zq_cnn_scalaroperation_min_inplace_32f_align128bit",
  "zq_cnn_scalaroperation_min_inplace_32f_align256bit",
  "zq_cnn_scalaroperation_rminus_32f_align0",
  "zq_cnn_scalaroperation_rminus_32f_align128bit",
  "zq_cnn_scalaroperation_rminus_32f_align256bit",
  "zq_cnn_scalaroperation_rminus_inplace_32f_align0",
  "zq_cnn_scalaroperation_rminus_inplace_32f_align128bit",
  "zq_cnn_scalaroperation_rminus_inplace_32f_align256bit",
  "zq_cnn_scalaroperation_rdiv_32f_align0",
  "zq_cnn_scalaroperation_rdiv_32f_align128bit",
  "zq_cnn_scalaroperation_rdiv_32f_align256bit",
  "zq_cnn_scalaroperation_rdiv_inplace_32f_align0",
  "zq_cnn_scalaroperation_rdiv_inplace_32f_align128bit",
  "zq_cnn_scalaroperation_rdiv_inplace_32f_align256bit",
  "zq_cnn_scalaroperation_pow_32f_align0",
  "zq_cnn_scalaroperation_pow_inplace_32f_align0",
};
static const int N_ENTRY = (int)(sizeof(g_entries) / sizeof(g_entries[0]));

#define RES_FILE "/tmp/zq_sc_res.txt"
static const double TOL = 1e-5;
static const char* g_op_name[7] = { "add", "mul", "max", "min", "rminus", "rdiv", "pow" };

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

struct Case { int entry, C, variant; };

// x = 输入，y = 标量；逐条对着 .c 里那七个 #define 抄
static double apply_op(int op, double x, double y)
{
    switch (op) {
    case 0: return x + y;                 // add     vaddq(x,y)
    case 1: return x * y;                 // mul     vmulq(x,y)
    case 2: return x > y ? x : y;         // max     vmaxq(x,y)
    case 3: return x < y ? x : y;         // min     vminq(x,y)
    case 4: return y - x;                 // rminus  vsubq(y,x)  **反向**
    case 5: return y / x;                 // rdiv    vdivq(y,x)  **反向**
    default: return pow(x, y);            // pow     (align0 手写标量)
    }
}

static void run_one(const Case& c)
{
    const Entry& e = g_entries[c.entry];
    const int C = c.C, N = 1, H = 5, W = 7;
    // 标量取负数 —— 六个 SIMD 运算里只有 max/min 对符号敏感，
    // rminus/rdiv 更需要非零负数才能暴露"反向"写反
    const float s = (c.variant == 0) ? 0.375f : -0.5f;
    const int ps = C, ws = ps * W, ss = ws * H;

    const size_t n = (size_t)N * ss;
    std::vector<float> in_m(n + 8), out_m(n + 8);
    float* in  = (float*)(((size_t)in_m.data()  + 31) / 32 * 32);
    float* out = (float*)(((size_t)out_m.data() + 31) / 32 * 32);
    for (size_t i = 0; i < n; i++) in[i] = val(1, (int)i);
    for (size_t i = 0; i < n; i++) out[i] = -12345.0f;

    if (e.inplace)
        ((F9)e.fn)(s, in, N, H, W, C, ps, ws, ss);
    else
        ((F13)e.fn)(s, in, N, H, W, C, ps, ws, ss, out, ps, ws, ss);

    const float* res = e.inplace ? in : out;
    long n_ok = 0, n_bad = 0; double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double y = apply_op(e.op, val(1, (int)i), s);
        const double got = res[i];
        double den = fabs(y) > 1.0 ? fabs(y) : 1.0;
        if (e.op == 5) den = fabs(y) > 1e-6 ? fabs(y) : 1.0;   // rdiv：分母敏感
        double be = fabs(got - y) / den;
        if (be > TOL) n_bad++; else n_ok++;
        if (be > worst) worst = be;
    }
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %.6e\n", n_ok, n_bad, worst); fclose(f); }
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(const Case& c)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) {
        FILE* dn = freopen("/dev/null", "w", stderr); (void)dn;
        run_one(c);
        _exit(0);
    }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0; double worst = 0;
    int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%ld %ld %lf", &ok, &bad, &worst) == 3); fclose(f); }
    char tag[96];
    snprintf(tag, sizeof(tag), "C=%d s=%.3f", c.C, c.variant ? -0.5f : 0.375f);
    if (!have) { g_crash++; printf("  %-52s %s  没跑完（退出码 %d）\n", g_name[c.entry], tag, WEXITSTATUS(st)); return; }
    if (WIFSIGNALED(st)) { g_crash++; printf("  %-52s %s  CRASH(信号 %d）\n", g_name[c.entry], tag, WTERMSIG(st)); return; }
    if (bad > 0) { g_bad++; printf("  %-52s %s  FAIL %ld/%ld 格错, 最差 %.3e\n", g_name[c.entry], tag, bad, ok + bad, worst); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHW scalaroperation：%d 个 32f 入口全测（7 运算 x 2 形式 x 3 对齐，pow 只有 align0）\n", N_ENTRY);
    printf("内核名全部写全、走函数指针表；判据：逐元素后向误差，逐格统计\n");
    printf("**rminus = s - in、rdiv = s / in 是反向运算**（.c 里 vsubq_f32(y,x) / vdivq_f32(y,x)）\n");
    printf("标量取 0.375 与 -0.5 两个值：负数才能暴露 rminus/rdiv 写反\n");
    printf("缓冲区 32 字节对齐（附录 CJ.4 / CP.5）\n\n");

    for (int e = 0; e < N_ENTRY; e++) {
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        const int A = g_entries[e].align;
        for (int v = 0; v < 2; v++) {
            for (int cc = 0; cc < 2; cc++) {
                Case c; memset(&c, 0, sizeof(c));
                c.entry = e; c.variant = v;
                c.C = (cc ? A * 2 : A);          // 整对齐 / 两倍
                one(c);
            }
        }
        printf("  %-50s  %d 个用例：对 %d，错 %d，崩 %d\n",
               g_name[e] + strlen("zq_cnn_scalaroperation_"),
               g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
