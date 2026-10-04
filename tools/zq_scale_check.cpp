// `zq_cnn_batchnormscale_32f_align_c.c` 整个 TU 的越界/数值门禁（附录 IB / ID）。
//
// 它覆盖 TU 里的**四个**逐通道内核（都做 `y = b*x + a` 这一件事）：
//   zq_cnn_scale_32f_align                          （附录 IB 修过带 bias 那支）
//   zq_cnn_batchnorm_32f_b_a_align{0,128bit,256bit}
//   zq_cnn_batchnormscale_32f_mean_var_scale_bias_align{0,128bit,256bit}
//
// 为什么是整个 TU 而不是只查 Scale：IB 修的正是"**带 bias 分支**漏了守卫、
// 而**不带 bias 的分支早就修过**"这件事 ——
// 也就是说，作者当初就是**只修了一条分支**。
// ID 把同 TU 的另外三个函数也拉进来，是为了确认
// **它们不需要同样的修复**，而这个结论应该是**跑出来的**，不是读出来的。
//
// **失败口径**：ASan 轮数的是输出里含 `FAIL` 的行数（附录 GZ.3），
// 所以真崩的组必须打 `FAIL`，而全通过时的汇总行**不许**出现字面 `FAIL`。
//
// 缺陷本体
// --------
// `ZQCNN/layers_c/zq_cnn_batchnormscale_32f_align_c_raw.h` 里的
// `zq_cnn_scale_32f_align`，**带 bias** 的那一支是：
//
//     for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size, c_ptr += zq_mm_align_size)
//     {
//         scale_vec = zq_mm_load_ps(scale_data + c);   // 读 zq_mm_align_size 个 float
//         bias_vec  = zq_mm_load_ps(bias_data + c);    // 同上
//         zq_mm_store_ps(c_ptr, ...);
//     }
//
// `scale` / `bias` 两个张量都是 `ChangeSize(1, 1, 1, C, 0, 0)` ——
// **只有 C 个 float**。于是 `in_C % align != 0` 时最后一下
// `zq_mm_load_ps(scale_data + c)` 会读过 C-1。
//
// 而**不带 bias** 的那一支**已经被修过**（同一文件里就写着）：
//
//     /* in_C 不是 4/8 的倍数: 整向量读会越过 scale_data 的分配, 改走标量 */
//     for (c = 0; c < in_C; c++)
//         pix_ptr[c] = pix_ptr[c] * scale_data[c];
//
// 也就是说：**只修了一条分支，另一条留着** —— 与 AGENTS.md
// 「补齐一处修复时把同仓的另一份拷贝列出来」（附录 HA.3）同源。
//
// 为什么是独立的门禁而不是塞进 SampleUnusedLayerProbe
// --------------------------------------------------
// `ZQ_CNN_Forward_SSEUtils.cpp` **一个 TU 引用半个库、-O1 编一次 5 分钟以上**
// （附录 EC.1 记过），所以现成的 sanitizer 门禁都**不编它**（用绊线桩）。
// 但这一层的实现本身在 `zq_cnn_batchnormscale_32f_align_c.c` 里 ——
// 一个很小的 TU，直接链它就行。
//
// 判据
// ----
// ASan 抓越界**读**（不是"值对不对"）：`scale` / `bias` 用
// **恰好 C 个 float 的对齐块**分配（`posix_memalign(32, C*4)`）——
// 对齐是为了让 `_mm256_load_ps` 那条路**先能跑起来**（第一版用裸 `malloc`，
// align=8 的组直接 SEGV，把"对齐不够"和"越界读"两种故障混成了一个信号）；
// 而"恰好 C 个 float"保证 ASan 的红区紧贴在 `scale[C-1]` 后面，
// 一读过界立刻 `heap-buffer-overflow READ`。
// 每组一个子进程（与仓库其它崩溃类门禁同款，附录 CJ），
// 崩溃只算该组失败，不影响后面的组。
#include "zq_check_child.h"
#include <cstdio>
#include <unistd.h>
#include <sys/wait.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#if defined(_WIN32)
#include <malloc.h>
#endif

// `_aligned_malloc` 只有 MSVC 有；Linux 侧用 `posix_memalign`。
// 收敛成一个函数，**不要**在每个调用点各写一遍跨平台分支
// （AGENTS.md「跨边界要写进工具里」）。
static float* alloc_aligned(size_t bytes, size_t align)
{
#if defined(_WIN32)
    return (float*)_aligned_malloc(bytes, align);
#else
    void* p = 0;
    if (posix_memalign(&p, align, bytes) != 0) return 0;
    return (float*)p;
#endif
}
static void free_aligned(float* p)
{
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

extern "C" {
void zq_cnn_scale_32f_align0(float* in_data, int in_N, int in_H, int in_W, int in_C,
                             int in_pixStep, int in_widthStep, int in_sliceStep,
                             const float* scale, const float* bias);
void zq_cnn_scale_32f_align128bit(float* in_data, int in_N, int in_H, int in_W, int in_C,
                                  int in_pixStep, int in_widthStep, int in_sliceStep,
                                  const float* scale, const float* bias);
void zq_cnn_scale_32f_align256bit(float* in_data, int in_N, int in_H, int in_W, int in_C,
                                  int in_pixStep, int in_widthStep, int in_sliceStep,
                                  const float* scale, const float* bias);
void zq_cnn_batchnorm_32f_b_a_align0(float* in_data, int in_N, int in_H, int in_W, int in_C,
        int in_pixStep, int in_widthStep, int in_sliceStep, const float* b, const float* a);
void zq_cnn_batchnorm_32f_b_a_align128bit(float* in_data, int in_N, int in_H, int in_W, int in_C,
        int in_pixStep, int in_widthStep, int in_sliceStep, const float* b, const float* a);
void zq_cnn_batchnorm_32f_b_a_align256bit(float* in_data, int in_N, int in_H, int in_W, int in_C,
        int in_pixStep, int in_widthStep, int in_sliceStep, const float* b, const float* a);
void zq_cnn_batchnormscale_32f_mean_var_scale_bias_align0(float* in_data, int in_N, int in_H, int in_W,
        int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
        const float* mean, const float* var, const float* slope, const float* bias, float eps);
void zq_cnn_batchnormscale_32f_mean_var_scale_bias_align128bit(float* in_data, int in_N, int in_H, int in_W,
        int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
        const float* mean, const float* var, const float* slope, const float* bias, float eps);
void zq_cnn_batchnormscale_32f_mean_var_scale_bias_align256bit(float* in_data, int in_N, int in_H, int in_W,
        int in_C, int in_pixStep, int in_widthStep, int in_sliceStep,
        const float* mean, const float* var, const float* slope, const float* bias, float eps);
}

// ASan 撞 SEGV 时默认走 Die() -> _exit(1)，**不发信号**，
// 所以父进程不能只看 WIFSIGNALED（附录 CJ.2）。
extern "C" const char* __asan_default_options() { return "abort_on_error=1"; }

struct Group { const char* name; int align; int C; int with_bias; };

static void run_group(const Group& g)
{
    const int H = 2, W = 2, N = 1;
    // 输出像素按对齐宽度铺开（NHW_C 布局：pixelStep = ceil(C/align)*align）
    const int pixStep = ((g.C + g.align - 1) / g.align) * g.align;
    const int widthStep = pixStep * W;
    const int sliceStep = widthStep * H;
    float* data = alloc_aligned(sizeof(float) * sliceStep, 32);
    // **精确分配** C 个 float：ASan 的红区就紧贴在 scale[C-1] 后面
    float* scale = alloc_aligned(sizeof(float) * g.C, 32);
    float* bias = alloc_aligned(sizeof(float) * g.C, 32);
    if (!data || !scale || !bias) { printf("alloc failed\n"); exit(2); }
    for (int i = 0; i < sliceStep; i++) data[i] = (float)(i % 17) * 0.25f - 2.0f;
    for (int c = 0; c < g.C; c++) { scale[c] = 1.0f + 0.01f * c; bias[c] = -0.5f * c; }

    switch (g.align) {
    case 1:  zq_cnn_scale_32f_align0(data, N, H, W, g.C, pixStep, widthStep, sliceStep,
                                      scale, g.with_bias ? bias : 0); break;
    case 4:  zq_cnn_scale_32f_align128bit(data, N, H, W, g.C, pixStep, widthStep, sliceStep,
                                          scale, g.with_bias ? bias : 0); break;
    default: zq_cnn_scale_32f_align256bit(data, N, H, W, g.C, pixStep, widthStep, sliceStep,
                                          scale, g.with_bias ? bias : 0); break;
    }
    // 值也要看一眼：越过 C 之后的那些填充 lane 会带着 scale/bias 的越界值，
    // 算出来 NaN 就说明越界读到的不是可解释的数。
    double acc = 0;
    for (int c = 0; c < g.C; c++) acc += data[c];
    if (acc != acc) printf("  %-22s 算出了 NaN\n", g.name);
    free_aligned(data);
    free_aligned(scale);
    free_aligned(bias);
}

// ---------------------------------------------------------------------------
// 同族第二部分：`zq_cnn_batchnorm_32f_b_a_align` 与
// `zq_cnn_batchnormscale_32f_mean_var_scale_bias_align`
// ---------------------------------------------------------------------------
//
// 为什么把它们也拉进来（2026-10-05，附录 ID）
// ------------------------------------------------
// IB 修的是 `zq_cnn_scale_32f_align` **带 bias 分支**的整向量越界读。
// 同一个 raw 头里还有三个函数，做的是同一件事（逐通道 y = b*x + a），
// 所以**必须逐个核对**是不是也有同款问题 —— 否则就是
// 「补齐一处修复时把同仓的另一份拷贝列出来」（附录 HA.3）漏了另一半。
//
// 人工核对结论（ID.1）：
//   zq_cnn_batchnorm_32f_b_a_align              **有**级联守卫：
//       % (8*align) -> % (4*align) -> % (2*align) -> **标量**，四档
//   两个 mean_var 系列                        先用**标量**循环把 a/b 算进
//       `_aligned_malloc(in_C*sizeof(float))` 分配的**恰好 in_C 个** float 里，
//       再交给上面那个有守卫的函数
//   zq_cnn_scale_32f_align（不带 bias 那支）  早就改成标量
//   zq_cnn_scale_32f_align（带 bias）         **原来没有守卫** <- IB 修的就是它
//
// 人工核对是"读代码"，**这条门禁把它变成"跑出来"**：
// 每个函数都在 ASan 下跑 C ∈ {3, 13, 64, 65}（非倍数 / 2 的倍数 /
// 8*align 的倍数 / 8*align+1），既查越界也查**值**。
struct BnGroup {
    const char* name;
    int align;
    int C;
    int kind;            // 0 = b_a          1 = mean_var_scale_bias
};

// 确定性的伪随机：两个 net / 两次调用必须拿到逐位相同的数据
static float brnd(unsigned& s)
{
    s = s * 1664525u + 1013904223u;
    return (float)((s >> 8) & 0xFFFF) / 32768.0f;      // [0, 1)
}

static void run_bn_group(const BnGroup& g)
{
    const int H = 2, W = 2, N = 1;
    const int pixStep = ((g.C + g.align - 1) / g.align) * g.align;
    const int widthStep = pixStep * W;
    const int sliceStep = widthStep * H;
    float* data = alloc_aligned(sizeof(float) * sliceStep, 32);
    // **恰好 C 个 float** 的对齐块：ASan 的红区紧贴在 [C-1] 后面，
    // 整向量读一旦越界立刻被抓；同时满足对齐载入的要求。
    float* p0 = alloc_aligned(sizeof(float) * g.C, 32);
    float* p1 = alloc_aligned(sizeof(float) * g.C, 32);
    float* p2 = alloc_aligned(sizeof(float) * g.C, 32);
    float* p3 = alloc_aligned(sizeof(float) * g.C, 32);
    if (!data || !p0 || !p1 || !p2 || !p3) { printf("alloc failed\n"); fflush(stdout); _exit(3); }
    unsigned s = 777u + (unsigned)g.C * 131u + (unsigned)g.align * 17u + (unsigned)g.kind;
    for (int i = 0; i < sliceStep; i++) data[i] = brnd(s) - 0.5f;
    for (int c = 0; c < g.C; c++) { p0[c] = brnd(s); p1[c] = brnd(s); }
    for (int c = 0; c < g.C; c++) { p2[c] = brnd(s); p3[c] = brnd(s); }

    const float eps = 1e-5f;
    // 参考值：mean_var_scale_bias 的定义写在那个 raw 头的文件头注释里
    //     a = bias - slope * mean / sqrt(var+eps)
    //     b = slope / sqrt(var+eps)
    //     y = b * x + a
    // var 取 [0.5, 1.5]，加上 eps 之后离 FLOAT_EPS_FOR_DIV 极远，
    // 所以那个 __max 守卫**不会被触发** —— 参考里也就不需要复制它。
    // **必须在调内核之前**把输入快照下来：这两个内核都是**就地**改 data，
    // 而参考值要的是**原始 x**。第一版在调用之后才读 data[c]，
    // 于是 want = b*已改过的值 + a，与 got 比自然全错
    // （实测相对误差 0.31 / 0.69 / 8.9 / 12.2，形态看着像"内核算错了"，
    //   实际是判据自己把被测对象的输出当成了它的输入）。
    std::vector<float> orig((size_t)g.C);
    for (int c = 0; c < g.C; c++) orig[c] = data[c];

    std::vector<float> ba((size_t)g.C), aa((size_t)g.C);
    for (int c = 0; c < g.C; c++) {
        float bb, aa_v;
        if (g.kind == 0) {           // 直接给 b / a
            bb = p0[c]; aa_v = p1[c];
        } else {                     // mean=p0, var=p1, slope=p2, bias=p3
            bb = p2[c] / sqrt(p1[c] + eps);
            aa_v = p3[c] - p0[c] * bb;
        }
        ba[c] = bb; aa[c] = aa_v;
    }

    switch (g.align) {
    case 1:
        if (g.kind == 0) zq_cnn_batchnorm_32f_b_a_align0(data, N, H, W, g.C, pixStep, widthStep, sliceStep, p0, p1);
        else zq_cnn_batchnormscale_32f_mean_var_scale_bias_align0(data, N, H, W, g.C, pixStep, widthStep, sliceStep, p0, p1, p2, p3, eps);
        break;
    case 4:
        if (g.kind == 0) zq_cnn_batchnorm_32f_b_a_align128bit(data, N, H, W, g.C, pixStep, widthStep, sliceStep, p0, p1);
        else zq_cnn_batchnormscale_32f_mean_var_scale_bias_align128bit(data, N, H, W, g.C, pixStep, widthStep, sliceStep, p0, p1, p2, p3, eps);
        break;
    default:
        if (g.kind == 0) zq_cnn_batchnorm_32f_b_a_align256bit(data, N, H, W, g.C, pixStep, widthStep, sliceStep, p0, p1);
        else zq_cnn_batchnormscale_32f_mean_var_scale_bias_align256bit(data, N, H, W, g.C, pixStep, widthStep, sliceStep, p0, p1, p2, p3, eps);
        break;
    }

    // 值也要查：只查越界的话，"整向量读到合法但错误的地址"这一类查不出来。
    // 逐通道比 b*x+a（C 之外的填充 lane 不参与）。
    double worst = 0;
    for (int c = 0; c < g.C; c++) {
        // 判据用**后向误差**：分母是"这一格的计算尺度" |b*x| + |a|，
        // 而不是 |结果|。y = b*x + a 在 b*x ≈ -a 时结果抵消到接近 0，
        // 除以它会把 1e-7 的绝对差放大成 1e-2 —— 那不是内核的错，
        // 是 float32 的固有性质（AGENTS.md「GEMM 的判据必须用后向误差」，
        // 这条对逐元素运算同样成立）。
        // 第一版用 |got-want| / (|want|+1e-6)，C=65 那一组报 2.737e-05，
        // 换成后向误差后同一格是 ~1e-7。
        double bx = (double)ba[c] * (double)orig[c];
        double want = bx + (double)aa[c];
        double got = (double)data[c];
        double scale = fabs(bx) + fabs((double)aa[c]) + 1e-6;
        double e = fabs(got - want) / scale;
        if (e > worst) worst = e;
    }
    if (worst > 1e-5) {
        printf("  %-30s 值对不上：最大相对误差 %.4g\n", g.name, worst);
        fflush(stdout);   // _exit 不冲刷缓冲，不加这行这行永远看不到
        _exit(4);        // 非 0 退出 -> 父进程判 FAIL
    }

    free_aligned(data);
    free_aligned(p0); free_aligned(p1); free_aligned(p2); free_aligned(p3);
}

int main()
{
    // 崩溃类门禁的固定做法（附录 BL.8）：**逐用例即时输出**。
    // 不加这一行的话，子进程退出前那次 fflush 会把父进程缓冲里
    // 那一整段也推出去，于是同一个汇总行被重复打印 12 次。
    setvbuf(stdout, 0, _IONBF, 0);
    static const Group GROUPS[] = {
        { "align1  C=3  no-bias", 1,  3, 0 },
        { "align1  C=3  bias   ", 1,  3, 1 },
        { "align4  C=3  no-bias", 4,  3, 0 },
        { "align4  C=3  bias   ", 4,  3, 1 },   // 修复前：heap-buffer-overflow
        { "align4  C=13 no-bias", 4, 13, 0 },
        { "align4  C=13 bias   ", 4, 13, 1 },   // 修复前：heap-buffer-overflow
        { "align8  C=3  no-bias", 8,  3, 0 },
        { "align8  C=3  bias   ", 8,  3, 1 },   // 修复前：heap-buffer-overflow
        { "align8  C=8  bias   ", 8,  8, 1 },   // C 是 align 的倍数：本来就不该红
        { "align8  C=16 bias   ", 8, 16, 1 },   // 同上
    };
    const int n = (int)(sizeof(GROUPS) / sizeof(GROUPS[0]));
    int bad = 0;
    for (int i = 0; i < n; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            // stderr 交给 $ZQ_CHILD_ERR（没设就 /dev/null）——
            // harness 在判失败时会把它的开头打出来（附录 CZ）。
            zq_child_silence_stderr();
            run_group(GROUPS[i]);
            fflush(stdout);
            _exit(0);
        }
        int st = 0;
        waitpid(pid, &st, 0);
        // ASan 撞 SEGV 默认走 Die() -> _exit(1)，**不发信号**，
        // 所以不能只看 WIFSIGNALED（附录 CJ.2）。
        int crashed = (WIFSIGNALED(st) || (WIFEXITED(st) && WEXITSTATUS(st) != 0));
        if (crashed) bad++;
        printf("  %-22s %s\n", GROUPS[i].name, crashed ? "FAIL 越界读/崩溃" : "ok");
    }
    // 汇总行**不许**出现字面的 FAIL —— harness 的 ASan 判据是
    // `grep -cE 'FAIL'`，全通过时也不该被算成失败（附录 FD.5）。
    printf("zq_scale：%d 组，越界/崩溃 %d 组\n", n, bad);

    // ---- 同族第二部分：BatchNorm / BatchNormScale 的四个入口 ----
    static const BnGroup BN_GROUPS[] = {
        { "b_a              align1  C=3 ", 1,  3, 0 },
        { "b_a              align4  C=3 ", 4,  3, 0 },
        { "b_a              align8  C=3 ", 8,  3, 0 },
        { "b_a              align8  C=13", 8, 13, 0 },
        { "b_a              align8  C=64", 8, 64, 0 },   // 8*align 的倍数：走向量那一档
        { "b_a              align8  C=65", 8, 65, 0 },   // +1：必须落到标量那一档
        { "mean_var_sc_bias align1  C=3 ", 1,  3, 1 },
        { "mean_var_sc_bias align4  C=3 ", 4,  3, 1 },
        { "mean_var_sc_bias align8  C=3 ", 8,  3, 1 },
        { "mean_var_sc_bias align8  C=13", 8, 13, 1 },
        { "mean_var_sc_bias align8  C=64", 8, 64, 1 },
        { "mean_var_sc_bias align8  C=65", 8, 65, 1 },
    };
    const int nbn = (int)(sizeof(BN_GROUPS) / sizeof(BN_GROUPS[0]));
    int nbad = 0;
    for (int i = 0; i < nbn; i++) {
        char errpath[512];
        snprintf(errpath, sizeof(errpath), "/tmp/zq_scale_bn_err_%d.txt", (int)getpid());
        pid_t pid = fork();
        if (pid == 0) {
            fclose(stderr);
            FILE* ef = freopen(errpath, "w", stderr);
            (void)ef;
            run_bn_group(BN_GROUPS[i]);
            fflush(stdout);
            _exit(0);
        }
        int st = 0;
        waitpid(pid, &st, 0);
        int crashed = (WIFSIGNALED(st) || (WIFEXITED(st) && WEXITSTATUS(st) != 0));
        if (crashed) nbad++;
        printf("  %-30s %s\n", BN_GROUPS[i].name, crashed ? "FAIL 越界/值对不上" : "ok");
        if (crashed) {
            FILE* ef = fopen(errpath, "rb");
            if (ef) {
                char buf[2048];
                size_t nr = fread(buf, 1, sizeof(buf) - 1, ef);
                buf[nr] = 0;
                fclose(ef);
                if (nr) {
                    printf("      --- 该组的报告（前 1000 字节）---\n");
                    fputs(buf, stdout);
                    printf("      --------------------------------\n");
                }
            }
        }
        remove(errpath);
    }
    printf("zq_scale(bn 部分)：%d 组，越界/值对不上 %d 组\n", nbn, nbad);

    return (bad == 0 && nbad == 0) ? 0 : 1;
}
