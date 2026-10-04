// `zq_cnn_scale_32f_align*` 的**越界读**门禁（附录 IB）。
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

int main()
{
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
    return bad == 0 ? 0 : 1;
}
