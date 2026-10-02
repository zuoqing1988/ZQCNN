/* tools/ 共用：对齐分配的小工具 —— 附录 CY
 *
 * 为什么需要它
 * ------------
 * `std::vector<float>` 只保证 **16 字节**对齐（`operator new` 的实现决定的），
 * 而 ZQCNN 的 align256 内核全是 `_mm256_load_ps` / `_mm256_store_ps`，
 * 要求 **32 字节**对齐。
 *
 * 后果：**任何拿 `std::vector<float>` 去喂 align256 入口的门禁，都会让内核
 * 执行未对齐 SIMD 访问。** 在 x86 上这不会崩、值也算对（只是慢一点），
 * 所以 ASan 完全看不见；而 UBSan 会报
 * `load of misaligned address ... for type '__m256', which requires 32 byte alignment`。
 *
 * 2026-10-02 第一次跑 UBSan 轴（附录 CY.1）时，30 个门禁里有 5 个因此报红：
 *   zq_bns / zq_eltwise / zq_lrn / zq_pool   —— 未对齐 load/store
 *   zq_nchw_resize                            —— 同上，但子进程把 stderr 吞了，看不到消息
 * 全部是**门禁侧的分配缺陷**，不是生产缺陷（生产给 align256 入口的是
 * `ZQ_CNN_Tensor4D_NHW_C_Align256bit`，它自己按 32 字节分配）。
 *
 * 但"是门禁的问题"这个结论必须**由实验坐实**，不能靠断言 ——
 * 所以这批门禁改成 32 字节对齐之后重跑 UBSan：如果未对齐告警全部消失，
 * 结论才成立（附录 CY.2）。
 *
 * 用法
 * ----
 *     #include "zq_check_alloc.h"
 *     float* p = zq_alloc_f32(n);      // n 个 float，32 字节对齐
 *     ... 用 p[i] ...
 *     zq_free_f32(p);
 *
 * **注意**：要精确分配，不要多给余量。判据是"越界即被抓"的门禁里，
 * 多分配的字节会把越界那一格藏进合法内存（附录 CX.5）。
 * 需要"对齐但可越界一点"的地方，自己在调用处 `realloc`/多申请。
 */
#ifndef ZQ_CHECK_ALLOC_H_
#define ZQ_CHECK_ALLOC_H_

#include <stdlib.h>

// n 个 float，返回 32 字节对齐的块；失败返回 0
static inline float* zq_alloc_f32(size_t n)
{
    void* p = 0;
    if (n == 0) n = 1;                 // malloc(0) 的返回值语义各平台不一
    if (posix_memalign(&p, 32, n * sizeof(float)) != 0) return 0;
    return (float*)p;
}

static inline void zq_free_f32(float* p) { free(p); }

#endif
