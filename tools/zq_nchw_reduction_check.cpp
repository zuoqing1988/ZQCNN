/* NCHW（layers_c）reduction 门禁 —— 附录 CT
 *
 * 覆盖面：2 个 32f 真实符号（nm 核实）
 *   zq_cnn_reduction_sum_32f_align0
 *   zq_cnn_reduction_mean_32f_align0
 * 二者都在 x86 分派器 ZQ_CNN_Forward_SSEUtils.cpp 里被引用，**是活的**；
 * 上一层 ZQ_CNN_Layer_Reduction 会读到模型文件里的 axis/keepdims。
 *
 * 语义（逐条从源码读出来的，`axis` 的约定见下）
 * --------------------------------------------
 *  上下文的四元组是 (N, C, H, W)，**axis 索引的是这个顺序**：
 *      axis == 0  约 N    axis == 1  约 C
 *      axis == 2  约 H    axis == 3  约 W
 *  证据是 ZQ_CNN_Forward_SSEUtils::ReductionSum 里的
 *      int out_dims[4] = { N, C, H, W };
 *      if (keepdims) out_dims[axis] = 1;
 *  **不是** `for n { for h { for w { for c` 的循环嵌套顺序 ——
 *  附录 CR.6 的笔记在这一点上是错的，本门禁按源码重读后改正（见 CT.2）。
 *
 *  keepdims != 0：沿 axis 求和/求均值，输出仍是 4 维，只是被约的那一维变成 1。
 *  keepdims == 0：out_dims 四维全设成 1，内核走 keepdims==0 那一支，
 *                  把**全部** N*H*W*C 个元素约成一个标量写进 out[0]。
 *
 *  步长约定（ZQ_CNN_Tensor4D，offset(n,c,h,w) = n*sliceStep + h*widthStep
 *  + w*pixelStep + c，c 连续）：
 *      pixelStep = oc，widthStep = ow*oc，sliceStep = oh*ow*oc
 *  门禁**从输出维度反推**步长，而不是把内核里那几个循环的步长抄一遍 ——
 *  抄一遍等于把自己的实现再抄一次，两个错会长得一模一样。
 *
 * 沿用 CB~CS：名字写全走函数指针表、逐格统计、每用例 fork 子进程并
 * **显式判"没读到结果文件" = 失败**（CJ.4）、缓冲区 32 字节对齐（CJ.4 / CP.5）。
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "ZQCNN/layers_c/zq_cnn_reduction_32f_align_c.h"

// sum / mean 的 14 参签名完全一样，用同一个函数指针类型。
typedef void (*F_RED)(const float* in, int N, int H, int W, int C,
                      int axis, int keepdims,
                      int ps, int ws, int ss,
                      float* out, int ops, int ows, int oss);

enum { K_SUM = 0, K_MEAN = 1 };
struct Entry { void* fn; int kind; const char* name; };

static const Entry g_entries[] = {
  { (void*)zq_cnn_reduction_sum_32f_align0,  K_SUM,  "zq_cnn_reduction_sum_32f_align0"  },
  { (void*)zq_cnn_reduction_mean_32f_align0, K_MEAN, "zq_cnn_reduction_mean_32f_align0" },
};
static const int N_ENTRY = 2;

// 每个函数跑 5 行：keepdims==0 一行 + axis 0..3 四行。
static const int N_ROW = 5;

#define RES_FILE "/tmp/zq_red_res.txt"

// 内核是**顺序 float 累加**，参考是 double 累加。n 个元素顺序累加的误差界
// 约 (n-1)*eps*Σ|x|；本门禁最大 2*3*4*5=120 个、|x|<=0.5，界 ~2e-4。
// 所以判据必须给一个绝对下限，否则最坏情况下会误报（附录 CT.5）。
static const double ATOL = 1e-3;
static const double RTOL = 1e-5;

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 1001) - 500) * 0.001f;   // [-0.5, 0.5]
}

struct Shape { int N, H, W, C; };
// 0: 四维都 >1；1: H=1；2: N=1 且 W=1（被约的那一维长度为 1 的极端情形）
static const Shape g_shapes[] = { {2,3,4,5}, {3,1,5,2}, {1,4,1,3} };
static const int N_SHAPE = 3;

struct Case { int entry, row, axis, keepdims, shape, variant; };

#define GUARD 8
static const float SENTINEL = -12345.0f;

static void run_one(const Case& c)
{
    const Entry& e = g_entries[c.entry];
    const Shape& s = g_shapes[c.shape];
    const int N = s.N, H = s.H, W = s.W, C = s.C;
    const int ps = C, ws = ps * W, ss = ws * H;      // NCHW：pixelStep 就是 C
    const size_t nin = (size_t)N * ss;

    // ext[] 的顺序是 (N, C, H, W) —— 与 axis 的约定一致。
    int ext[4] = { N, C, H, W };
    int od[4]  = { N, C, H, W };
    int red[4] = { 0, 0, 0, 0 };
    if (c.keepdims) { od[c.axis] = 1; red[c.axis] = 1; }
    else             { for (int a = 0; a < 4; a++) { od[a] = 1; red[a] = 1; } }

    const int ops = od[1];                    // pixelStep = oc
    const int ows = od[3] * od[1];            // widthStep = ow*oc
    const int oss = od[2] * od[3] * od[1];    // sliceStep = oh*ow*oc
    const int nout = od[0] * od[1] * od[2] * od[3];

    std::vector<float> in_m(nin + 8), ref_m(nout + 8), out_m(nout + GUARD + 8);
    float* in  = (float*)(((size_t)in_m.data()  + 31) / 32 * 32);
    float* ref = (float*)(((size_t)ref_m.data() + 31) / 32 * 32);
    float* out = (float*)(((size_t)out_m.data() + 31) / 32 * 32);

    // variant 0：普通数据；variant 1：**全 0.25 常数**。
    // 常数那一组的杀伤力在于 mean 必须恒等于 0.25、sum 必须恒等于
    // 0.25*count —— 约数算错（拿 C 当分母、漏乘 weight…）立刻整行红，
    // 而普通数据上分母错了只会带来一点点相对误差。
    for (size_t i = 0; i < nin; i++)
        in[i] = (c.variant == 1) ? 0.25f : val(1, (int)i);
    for (int i = 0; i < nout + GUARD; i++) out[i] = SENTINEL;
    memset(ref, 0, sizeof(float) * (size_t)nout);

    // 参考：遍历**输出空间**，被约掉的那些轴在内层取满范围。
    int ra[4], nr = 0;
    for (int a = 0; a < 4; a++) if (red[a]) ra[nr++] = a;
    // mean 的约数：keepdims!=0 时是 1/被约的那一维，keepdims==0 时是 1/(N*C*H*W)。
    // 两种写法内核里不一样（一个是 weight，一个是 sum / (N*H*W*C)），
    // 参考这边也分开写，免得"两个不等价的写法恰好凑出来"这种假绿。
    const int kd = c.keepdims ? c.axis : 0;    // keepdims==0 时 axis 被内核忽略，传 0
    const float div = c.keepdims ? (float)ext[kd] : (float)(N * H * W * C);
    for (int n = 0; n < od[0]; n++)    for (int n = 0; n < od[0]; n++)
        for (int k = 0; k < od[1]; k++)
            for (int h = 0; h < od[2]; h++)
                for (int w = 0; w < od[3]; w++) {
                    int coord[4] = { n, k, h, w };
                    int rk[4] = { 0, 0, 0, 0 };
                    double sum = 0.0;
                    for (;;) {
                        int cd[4] = { coord[0], coord[1], coord[2], coord[3] };
                        for (int i = 0; i < nr; i++) cd[ra[i]] = rk[i];
                        sum += in[(size_t)cd[0] * ss + (size_t)cd[2] * ws
                                  + (size_t)cd[3] * ps + cd[1]];
                        int i = nr - 1;
                        for (; i >= 0; i--) { if (++rk[i] < ext[ra[i]]) break; rk[i] = 0; }
                        if (i < 0) break;
                    }
                    const size_t o = (size_t)n * oss + (size_t)h * ows
                                   + (size_t)w * ops + k;
                    ref[o] = (e.kind != K_MEAN) ? (float)sum
                          : (c.keepdims)       ? (float)(sum * (double)(1.0f / div))
                                               : (float)(sum / (double)div);
                }

    ((F_RED)e.fn)(in, N, H, W, C, kd, c.keepdims, ps, ws, ss, out, ops, ows, oss);

    long bad = 0; double worst = 0.0;
    for (int i = 0; i < nout; i++) {
        const double g = (double)out[i], r = (double)ref[i];
        if (!(g == g) || !(g == g && g < 1e30 && g > -1e30)) { bad++; if (1.0 > worst) worst = 1.0; continue; }
        const double lim = ATOL + RTOL * (r < 0 ? -r : r);
        const double err = (g > r ? g - r : r - g);
        if (err > lim) bad++;
        const double rel = err / (1.0 + (r < 0 ? -r : r));
        if (rel > worst) worst = rel;
    }
    // 越界写守卫：内核只该写前 nout 个，后面 GUARD 个必须还是哨兵
    long over = 0;
    for (int i = 0; i < GUARD; i++)
        if (out[nout + i] != SENTINEL) over++;

    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %.6e %ld\n", (long)nout - bad - over, bad + over, worst, over); fclose(f); }
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static const char* row_name(int e, int row)
{
    static char buf[128];
    const Entry& en = g_entries[e];
    if (row == 0) snprintf(buf, sizeof(buf), "%s[keepdims=0]", en.name);
    else          snprintf(buf, sizeof(buf), "%s[axis=%d]", en.name, row - 1);
    return buf;
}

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
    long ok = 0, bad = 0, over = 0; double worst = 0;
    int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%ld %ld %lf %ld", &ok, &bad, &worst, &over) == 4); fclose(f); }
    char tag[96];
    const Shape& s = g_shapes[c.shape];
    snprintf(tag, sizeof(tag), "N%dC%dH%dW%d %s", s.N, s.C, s.H, s.W,
             c.variant ? "常数0.25" : "普通");
    if (!have) { g_crash++; printf("  %-46s %s  没跑完（退出码 %d）\n", row_name(c.entry, c.row), tag, WEXITSTATUS(st)); return; }
    if (WIFSIGNALED(st)) { g_crash++; printf("  %-46s %s  CRASH(信号 %d)\n", row_name(c.entry, c.row), tag, WTERMSIG(st)); return; }
    if (bad > 0) {
        g_bad++;
        printf("  %-46s %s  FAIL %ld/%ld 格错, 最差 %.3e%s\n",
               row_name(c.entry, c.row), tag, bad, ok + bad, worst,
               over ? "  [含越界写]" : "");
    } else {
        g_ok++;
    }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHW reduction：%d 个 32f 入口 x %d 行 = %d 行全测\n", N_ENTRY, N_ROW, N_ENTRY * N_ROW);
    printf("内核名全部写全、走函数指针表；判据：逐格绝对+相对误差，逐格统计\n");
    printf("每行跑 3 种形状 x 2 种数据（普通 / 全 0.25 常数）\n");
    printf("  常数那一组让 mean 恒等于 0.25、sum 恒等于 0.25*count ——\n");
    printf("  **约数算错会整行红**，普通数据上分母错了只差一点点\n");
    printf("输出缓冲后面放哨兵，**越界写会被抓出来**\n\n");

    for (int e = 0; e < N_ENTRY; e++) {
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        for (int row = 0; row < N_ROW; row++) {
            const int r0 = g_case, a0 = g_ok, b1 = g_bad, x1 = g_crash;
            const int axis = row - 1;              // row 0 = keepdims==0
            const int kd = (row == 0) ? 0 : 1;
            for (int sh = 0; sh < N_SHAPE; sh++)
                for (int v = 0; v < 2; v++) {
                    Case c; memset(&c, 0, sizeof(c));
                    c.entry = e; c.row = row; c.axis = axis; c.keepdims = kd;
                    c.shape = sh; c.variant = v;
                    one(c);
                }
            printf("    %-44s  %d 个用例：对 %d，错 %d，崩 %d\n",
                   row_name(e, row), g_case - r0, g_ok - a0, g_bad - b1, g_crash - x1);
        }
        printf("  %-46s  %d 个用例：对 %d，错 %d，崩 %d\n",
               g_entries[e].name, g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
