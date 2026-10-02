/* NCHW（layers_c）no_padding GEMM 的**内存归属**门禁 —— 附录 CU
 *
 * 这道门禁只问一件事：**失败路径释放的内存，是不是这个函数自己分配的。**
 *
 * 缺陷（附录 CU.1 / CU.2）
 * -----------------------
 * `zq_cnn_convolution_gemm_32f_align_c_raw.h` 里 7 个 no_padding gemm 函数，
 * 错误处理都是这个形状：
 *
 *     if (buffer == 0)
 *     {
 *         matrix_A = _aligned_malloc(...);
 *         if (need_allocate_tmp_out) matrix_C = _allocated_malloc(...);
 *         if (matrix_A == 0 || matrix_Bt == 0 || (need_allocate_tmp_out && matrix_C == 0))
 *         {
 *             if (matrix_A) _aligned_free(matrix_A);
 *             if (matrix_Bt) _aligned_free(matrix_Bt);   // ← 从来不是自己分配的
 *             if (matrix_C) _aligned_free(matrix_C);     // ← !need_allocate_tmp_out 时是 out_tensor4D_data
 *             return;
 *         }
 *     }
 *
 * `matrix_Bt` 在这几个函数里是 `const zq_base_type* matrix_Bt = filters_data;`
 * —— **调用方滤波器数据的别名**，从头到尾没被赋过 malloc 指针，只当只读输入用。
 * 把它 free 掉 = 释放一块本函数既没有所有权、也不知道大小的内存。
 * 同理 `!need_allocate_tmp_out` 时 `matrix_C = out_tensor4D_data`，
 * 那是**调用方的输出张量**。
 *
 * 第二条（CU.2）：`..._same_or_notsame_pixstep{,_batch}` 把「先分配再判空」
 * 改成了「先判空再分配」，于是 `need_allocate_matrix_Bt` 为真时判空恒真、
 * 函数必然直接 return（详见附录 CU.2）。
 *
 * 同文件里正确写法就在几十行外（`..._same_or_notsame_pixstep_C3`）：
 * **先分配、再判空**，释放时带上各自的 need_allocate_* 守卫。
 *
 * 怎么测
 * ------
 * **两个拦截器，都是链接期的，一个字节的生产代码都不用改：**
 *
 *  1) `memalign` —— 模拟 OOM。`ZQ_CNN_CompileConfig.h:109` 把
 *     `_aligned_malloc(x,y)` 定义成 `memalign(y,x)`，所以拦 memalign
 *     就等于拦分配。
 *  2) `free` —— 记录「谁被释放了」。
 *
 * 关键细节：**free 拦截器在 ASan 下不能用**。ASan 自己的运行时会调 free，
 * 我们在 main 之前就把 free 抢过来，dlsym 还没初始化好，一调用就段错误
 * （这个坑踩了两次，附录 CU.9）。所以这道门禁**显式关掉 ASan**
 * （run_zqlib_checks.py 里给本 tag 加 `-fno-sanitize=address`），
 * 改用「记录指针」而不是「让 ASan 去报」—— 后者依赖 sanitizer 的行为，
 * 前者只依赖「这个指针有没有被传进 free」，**确定性得多**。
 *
 * 命中规则：把 `filters_data` / `out_tensor4D_data` 指向两块**我们自己 malloc 的
 * 堆块**，并在 free 拦截器里登记它们的地址。库要是把其中任何一块 free 掉，
 * 拦截器记下来并**不再往下转发**（转发就是真的破坏堆，测试不该那样干）。
 *
 * 三组用例：
 *   G1  登记 filters_data，强制第一次 memalign 失败 -> 期望**没被 free**
 *   G2  登记 out_tensor4D_data（need_allocate_tmp_out=0），同样强制失败
 *       -> 期望**没被 free**，也没被写
 *   G3  in_pixelStep < filter_pixelStep、buffer==0、正常分配
 *       -> 期望输出**真的被算了**。oracle 用**同一个函数的 buffer!=0 分支**
 *       （那条本来就没坏），不用手写参考实现 —— 又是一次同仓 A/B。
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/wait.h>

// ---- 拦截器 1：memalign，用来模拟 OOM ----
static int  g_memalign_calls = 0;
static int  g_fail_at = -1;          // -1 = 永不失败
static void* (*g_real_memalign)(size_t, size_t) = 0;

extern "C" void* memalign(size_t alignment, size_t size) throw()
{
    const int n = g_memalign_calls++;
    if (n == g_fail_at)
        return 0;
    if (!g_real_memalign)
        g_real_memalign = (void* (*)(size_t, size_t))dlsym(RTLD_NEXT, "memalign");
    return g_real_memalign ? g_real_memalign(alignment, size) : 0;
}

// ---- 拦截器 2：free，记录"谁被释放了" ----
static void* g_watch[4] = {0, 0, 0, 0};   // 被登记的"调用方拥有"的块
static int   g_watch_freed[4] = {0, 0, 0, 0};
static void  (*g_real_free)(void*) = 0;

extern "C" void free(void* p) throw()
{
    for (int i = 0; i < 4; i++) {
        if (g_watch[i] && p == g_watch[i]) {
            g_watch_freed[i] = 1;
            return;                      // **不再往下转发**：真转发就是破坏堆
        }
    }
    if (!g_real_free)
        g_real_free = (void (*)(void*))dlsym(RTLD_NEXT, "free");
    if (g_real_free) g_real_free(p);
}

// ---- GEMM 桩：把那个 >5 分钟的 zq_gemm_32f_align_c.c 从门禁里摘掉 ----------
// 这道门禁只问「释放的是不是自己的内存」「函数到底跑没跑」。
// 数值正确性由 tools/zq_nchw_conv_check.cpp 负责（那道门禁才需要真 GEMM）。
// 这里桩掉 zq_gemm_32f_align_c.c 唯一提供的两个符号：
//   zq_gemm_32f_AnoTrans_Btrans_auto   —— 分派器
//   zq_gemm_32f_align0_AnoTrans_Btrans —— align0 内核（被 notsame 族引用）
// 桩里写一个由输入决定的确定性图案，于是「GEMM 被调过」与
// 「函数提前 return、输出原封不动」在结果上必然可分。
// 效果：门禁从「编一次 7 分钟」变成几秒，因此可以进**每次**回归，
// 而不是像 zq_nchw_conv 那样挂在 --with-slow 后面。
static void stub_fill(int M, int N, int K, const float* A, int lda,
                      const float* Bt, int ldb, float* C, int ldc)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++)
                acc += (double)A[(size_t)i * lda + k] * (double)Bt[(size_t)j * ldb + k];
            C[(size_t)i * ldc + j] = (float)acc;
        }
}
extern "C" void zq_gemm_32f_AnoTrans_Btrans_auto(int M, int N, int K,
    const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{ stub_fill(M, N, K, A, lda, Bt, ldb, C, ldc); }
extern "C" void zq_gemm_32f_align0_AnoTrans_Btrans(int M, int N, int K,
    const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{ stub_fill(M, N, K, A, lda, Bt, ldb, C, ldc); }

#include "layers_c/zq_cnn_convolution_gemm_32f_align_c.h"

typedef void (*FN)(
    const float* in_tensor4D_data,
    int in_N, int in_H, int in_W, int in_C,
    int in_pixelStep, int in_widthStep, int in_sliceStep,
    const float* filters_data,
    int filter_N, int filter_H, int filter_W, int filter_C,
    int filter_pixelStep, int filter_widthStep, int filter_sliceStep,
    int stride_H, int stride_W,
    int dilation_H, int dilation_W,
    float* out_tensor4D_data,
    int out_N, int out_H, int out_W, int out_C,
    int out_pixelStep, int out_widthStep, int out_sliceStep,
    void** buffer, __int64* buffer_len);

struct Entry { FN fn; const char* name; };

// 名字全部写全，不做字符串拼接（附录 CA.3）
static const Entry g_entries[] = {
  { zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep,          "align128bit_same_pixstep" },
  { zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_kernel1x1,"align128bit_same_pixstep_kernel1x1" },
  { zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_C4,       "align128bit_same_pixstep_C4" },
  { zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_batch,    "align128bit_same_pixstep_batch" },
  { zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep,       "align128bit_same_or_notsame_pixstep" },
  { zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_C3,    "align128bit_same_or_notsame_pixstep_C3" },
  { zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_batch, "align128bit_same_or_notsame_pixstep_batch" },
};
static const int N_ENTRY = 7;
static const int ENTRY_NOTSAME = 4;   // 同上表第 5 个起才是 notsame 族

#define RES_FILE "/tmp/zq_convfree_res.txt"
static const double TOL = 1e-5;

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

enum { G1 = 1, G2 = 2, G3 = 3 };
struct Case { int group, entry; };

// G1/G2 用的形状：N=1,C=4,H=6,W=7，3x3，filter_N=5
static void run_fail_path(const Case& c, int which)   // which: 0 = fil, 1 = out
{
    const int N = 1, H = 6, W = 7, C = 4, fH = 3, fW = 3, fN = 5;
    const int in_ps = C, in_ws = in_ps * W, in_ss = in_ws * H;
    const int f_ps = C, f_ws = f_ps * fW, f_ss = f_ws * fH;
    const int oH = H - fH + 1, oW = W - fW + 1;
    // need_allocate_tmp_out = 0：out 的步长要与 filter_N 一致
    const int o_ps = fN, o_ws = o_ps * oW, o_ss = o_ws * oH;
    const size_t nin = (size_t)N * in_ss, nfil = (size_t)fN * f_ss, nout = (size_t)N * o_ss;

    std::vector<float> in_m(nin + 8);
    float* in = (float*)(((size_t)in_m.data() + 31) / 32 * 32);
    for (size_t i = 0; i < nin; i++) in[i] = val(1, (int)i);

    // **两个"调用方拥有的"缓冲都用 malloc 分配** —— 这才是生产里的真实情形
    // （packed filters 是跨多层复用的堆块，输出是调用方的张量）。
    float* fil = (float*)malloc(nfil * sizeof(float));
    float* out = (float*)malloc(nout * sizeof(float));
    if (!fil || !out) { FILE* f = fopen(RES_FILE, "w"); if (f) fprintf(f, "0 1 0 0\n"); fclose(f); return; }
    for (size_t i = 0; i < nfil; i++) fil[i] = val(2, (int)i);
    for (size_t i = 0; i < nout; i++) out[i] = -777.0f;

    // 两块都登记：不管库 free 的是哪一个，都记下来
    g_watch[0] = fil; g_watch[1] = out; g_watch[2] = 0; g_watch[3] = 0;
    g_watch_freed[0] = g_watch_freed[1] = g_watch_freed[2] = g_watch_freed[3] = 0;

    g_memalign_calls = 0;
    g_fail_at = 0;                  // 第一次分配就失败

    // **buffer 参数必须传【空指针】，不是 &buf。**
    // 这就是第一版门禁"没有牙齿"的真正原因：库里的判据是 `if (buffer == 0)`
    // —— 判的是**那个 void** 本身是不是空指针，而不是 `*buffer`。
    // 传 &buf（非空）会走"调用方给缓冲"那条分支，内部 malloc 分支一次都没跑到，
    // 于是错误路径自然不会触发，门禁永远全绿。
    // 代价是排查了整整一轮才定位到：现象是"错误路径明明进不去却也不报错"，
    // 而 `free(*buffer)` 传进来的那个 `(nil)` 才是唯一露出来的线索。
    void** buffer_arg = 0;          // 刻意是**空指针** -> 走内部 malloc 分支

    ((FN)g_entries[c.entry].fn)(in, N, H, W, C, in_ps, in_ws, in_ss,
        fil, fN, fH, fW, C, f_ps, f_ws, f_ss, 1, 1, 1, 1,
        out, N, oH, oW, fN, o_ps, o_ws, o_ss, buffer_arg, 0);
    g_fail_at = -1;

    // 判据：这两块内存一块都不该被 free
    long bad = 0;
    if (g_watch_freed[0]) bad++;
    if (g_watch_freed[1]) bad++;
    // 另外 out 也不该被写（分配就失败了）
    long written = 0;
    if (which == 1)
        for (size_t i = 0; i < nout; i++) if (out[i] != -777.0f) written++;

    g_watch[0] = g_watch[1] = 0;    // 解除登记，下面这两次 free 才是干净的
    free(fil); free(out);

    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %.6e %ld\n", 2L - bad, bad, 0.0, written); fclose(f); }
}

// G3：need_allocate_matrix_Bt = 1（in_pixelStep < filter_pixelStep），buffer==0。
// oracle = 同一个函数的 buffer!=0 分支（它本来就没坏）。
static void run_notsame(const Case& c)
{
    const int N = 1, H = 8, W = 9, C = 8, fH = 3, fW = 3, fN = 4;
    const int in_ps = 4;                       // **故意小于 filter_pixelStep**
    const int in_ws = in_ps * W, in_ss = in_ws * H;
    const int f_ps = C, f_ws = f_ps * fW, f_ss = f_ws * fH;
    const int oH = H - fH + 1, oW = W - fW + 1;
    const int o_ps = fN, o_ws = o_ps * oW, o_ss = o_ws * oH;
    const size_t nin = (size_t)N * in_ss, nfil = (size_t)fN * f_ss, nout = (size_t)N * o_ss;

    std::vector<float> in_m(nin + 8), fil_m(nfil + 8),
                      a_m(nout + 8), b_m(nout + 8);
    float* in  = (float*)(((size_t)in_m.data()  + 31) / 32 * 32);
    float* fil = (float*)(((size_t)fil_m.data() + 31) / 32 * 32);
    float* a   = (float*)(((size_t)a_m.data()   + 31) / 32 * 32);
    float* b   = (float*)(((size_t)b_m.data()   + 31) / 32 * 32);
    for (size_t i = 0; i < nin; i++)  in[i]  = val(1, (int)i);
    for (size_t i = 0; i < nfil; i++) fil[i] = val(2, (int)i);
    for (size_t i = 0; i < nout; i++) { a[i] = -777.0f; b[i] = -777.0f; }

    g_watch[0] = g_watch[1] = g_watch[2] = g_watch[3] = 0;
    g_watch_freed[0] = g_watch_freed[1] = g_watch_freed[2] = g_watch_freed[3] = 0;
    g_fail_at = -1;

    // 支路 1：buffer 是**空指针**（内部 malloc）—— 这就是被修的那条
    void** buffer_arg = 0;
    ((FN)g_entries[c.entry].fn)(in, N, H, W, C, in_ps, in_ws, in_ss,
        fil, fN, fH, fW, C, f_ps, f_ws, f_ss, 1, 1, 1, 1,
        a, N, oH, oW, fN, o_ps, o_ws, o_ss, buffer_arg, 0);

    // 支路 2：buffer != 0（调用方给缓冲）—— oracle
    void* buf2 = 0; __int64 buf2_len = 0;
    ((FN)g_entries[c.entry].fn)(in, N, H, W, C, in_ps, in_ws, in_ss,
        fil, fN, fH, fW, C, f_ps, f_ws, f_ss, 1, 1, 1, 1,
        b, N, oH, oW, fN, o_ps, o_ws, o_ss, &buf2, &buf2_len);

    long bad = 0; double worst = 0.0;
    for (size_t i = 0; i < nout; i++) {
        const double d = fabs((double)a[i] - (double)b[i]);
        const double den = 1.0 + fabs((double)b[i]);
        if (d / den > TOL) bad++;
        if (d / den > worst) worst = d / den;
    }
    free(buf2);
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %.6e 0\n", (long)nout - bad, bad, worst); fclose(f); }
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(const Case& c)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) {
        FILE* dn = freopen("/dev/null", "w", stderr); (void)dn;
        if (c.group == G3) run_notsame(c);
        else                 run_fail_path(c, c.group == G1 ? 0 : 1);
        _exit(0);
    }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0, over = 0; double worst = 0; int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%ld %ld %lf %ld", &ok, &bad, &worst, &over) == 4); fclose(f); }
    const char* tag = c.group == G1 ? "OOM 时不得 free(filters_data)"
                     : c.group == G2 ? "OOM 时不得 free(out_tensor4D_data)"
                                     : "in_pixelStep<filter_pixelStep 必须真算";
    if (!have || WIFSIGNALED(st)) {
        g_crash++;
        printf("  %-44s %-34s  %s（信号 %d）\n", g_entries[c.entry].name, tag,
               WIFSIGNALED(st) ? "CRASH" : "没跑完",
               WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st));
        return;
    }
    if (bad > 0) {
        g_bad++;
        printf("  %-44s %-34s  FAIL %ld/%ld 项越权%s%s\n",
               g_entries[c.entry].name, tag, bad, ok + bad,
               over ? "，且输出被写了 " : "", over ? "" : "");
        if (over) printf("  %-44s %-34s        （另有 %ld 格输出被写了，分配失败时不该写）\n",
                         g_entries[c.entry].name, tag, over);
    } else {
        g_ok++;
    }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHW no_padding gemm：%d 个入口 x 3 组归属检查\n", N_ENTRY);
    printf("G1/G2 用链接期 memalign 拦截模拟 OOM，用 free 拦截记录\"谁被释放了\"\n");
    printf("    —— **这道门禁显式关掉 ASan**：ASan 运行时自己也要调 free，\n");
    printf("       在它初始化完成前抢走 free 会段错误（附录 CU.9）\n");
    printf("G3 的 oracle 是**同一个函数的 buffer!=0 分支**（那条本来就没坏）\n\n");
    for (int e = 0; e < N_ENTRY; e++) {
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        one((Case){ G1, e }); one((Case){ G2, e });
        if (e >= ENTRY_NOTSAME) one((Case){ G3, e });
        printf("  %-44s  %d 个用例：对 %d，错 %d，崩 %d\n",
               g_entries[e].name, g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
