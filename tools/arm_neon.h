/* ARM NEON 桩 —— **只为了在 x86 上把 `__ARM_NEON` 分支解析一遍**（附录 GP）。
 *
 * 为什么需要它
 * ------------
 * 仓库里 **86 个文件**带 `#if __ARM_NEON` 分支，**1606 处** NEON 调用点，
 * 而**一行都没被任何编译器看过**：
 *   * WSL 里没有 `arm-linux-gnueabihf-gcc`，也没有 clang（2026-10-04 实测）；
 *   * 仓库根的 `build.sh` 正是构建 armeabi-v7a 的 —— 也就是说
 *     仓库自己带着一条**从未被验证过**的构建路径。
 *
 * 而 GM 已经证明过这条轴上"没人看"会漏掉真缺陷：
 * `zq_gemm_32f_asm_core_m6n8` 缺前置声明，只在 SSETYPE=0/1 出现，
 * 默认那两档连 warning 都没有。
 *
 * 这份桩能证明什么、不能证明什么
 * ----------------------------
 * **能**（本门禁的全部承诺）：
 *   * NEON 分支能被**解析**（括号、大括号、预处理嵌套、局部变量）；
 *   * NEON 分支里用到的每个普通 C 代码（下标算术、打包循环、尾部处理）
 *     会经过**真正的类型检查**；
 *   * 每个 NEON 内在函数名都在桩里 —— 拼错会立刻报"未声明"。
 *
 * **不能**（写下来是为了不让它被当成"ARM 路径验过了"）：
 *   * **不验 NEON 的类型**。下面把 `float32x4_t` / `float16x8_t` 都
 *     typedef 成 `float` —— 一切运算都合法，于是"把 f16 的向量当 f32 用"
 *     这类错**抓不到**。要抓它需要真正的 `arm_neon.h`。
 *   * **不验语义**。NEON 内在函数的返回全是 0，
 *     所以任何依赖运行结果的判断（对齐、饱和、舍入）都验不了。
 *   * **不验 x86 上根本编不出来的部分**。实测那 86 个文件里，
 *     NEON 区域出现 `__asm__` 的有 **0 个** —— 这是这个方案能成立的前提，
 *     `tools/check_neon_branch.py` 每次都会重新量这一条。
 *
 * 桩的完整性由谁保证
 * ------------------
 * 不是由我手写一张"应该有哪些"的清单保证，而是**由代码实际用到的名字反推**：
 * `check_neon_branch.py` 先扫出所有 NEON 区域里用到的 `v*q_*` 名字，
 * 凡是桩里没有的，它**直接把名字列出来**并要求补 —— 门禁自己会喊。
 * （第一次跑就喊出一个：`vdivq_f16`。）
 *
 * **为什么这个文件就叫 arm_neon.h**：代码里写的是 `#include <arm_neon.h>`，
 * 桩不叫这个名字就完全不会被命中 —— 第一版叫 `arm_neon_stub.h`，
 * 86 个文件里 46 个报 "arm_neon.h: No such file or directory"。
 * 安全性：`grep` 过全部 CMakeLists，**没有任何构建把 tools/ 加进包含路径**，
 * 所以它不会在任何真实构建里遮蔽系统/交叉工具链的 arm_neon.h。
 */
#ifndef ZQ_ARM_NEON_STUB_H_
#define ZQ_ARM_NEON_STUB_H_

/* 刻意**不是**结构体：`typedef float` 让向量之间的 + - * / 与标量全部合法，
   于是"桩不支持某个写法"不会伪装成"代码有 bug"。
   代价就是上面写的"不验类型" —— 这是一个明确的取舍，不是疏忽。 */
/* `__fp16` 在真实 ARM 工具链上是**编译器内建的关键字**（ACLE 保证），
   x86 上没有，所以这里补一个。
   它必须**排在 ZQ_CNN_CompileConfig.h 之前**可用 ——
   头文件里那句 `typedef __fp16 float16_t;` 就在配置头里。
   所以门禁用 gcc 的 `-include tools/arm_neon.h` 把桩提前，
   而不是靠 `#include <arm_neon.h>` 的自然顺序（那在 .c 里，晚于配置头）。 */
typedef float __fp16;

typedef float float32x4_t;
typedef float float16x8_t;
typedef float int32x4_t;
typedef float uint32x4_t;
typedef float int16x4_t;
typedef float uint16x4_t;
typedef float uint8x8_t;
typedef float int64x2_t;
typedef float uint64x2_t;
typedef float float32x2_t;
typedef float int32x2_t;

/* **每个实参都要过一遍 `sizeof`。**
 *
 * 第一版是 `#define ZQA_NEON_STUB_ANY(...) 0` —— 宏把参数**整个丢掉**。
 * 后果实测出来了：注入 `vst1q_f32(q, zq_gp_never_declared)` **编译通过**，
 * 因为 `zq_gp_never_declared` 压根没进 token 流。
 * 也就是说"内在函数实参里的表达式"整片都逃过了类型检查 ——
 * 而 NEON 代码的绝大部分正好在那里（`vfmaq_laneq_f32` 一个名字就 888 处）。
 *
 * `sizeof((__VA_ARGS__, 0))` 把每个实参拉进一个逗号表达式：
 * 名字拼错、未声明、不完整类型都会在这里变成编译错误；
 * 代价只是**不求值**（我们本来就只做 `-fsyntax-only`）。
 *
 * 局限仍然在：**类型对不对**不查（向量全 typedef 成 float），
 * 只有"这个名字存不存在、这个表达式成不成立"被查。 */
#define ZQA_NEON_STUB_ANY(...) ((void)sizeof((__VA_ARGS__, 0)), 0)

/* --- f32（绝大多数调用点） --- */
#define vaddq_f32            ZQA_NEON_STUB_ANY
#define vsubq_f32            ZQA_NEON_STUB_ANY
#define vmulq_f32            ZQA_NEON_STUB_ANY
#define vdivq_f32            ZQA_NEON_STUB_ANY
#define vmaxq_f32            ZQA_NEON_STUB_ANY
#define vminq_f32            ZQA_NEON_STUB_ANY
#define vfmaq_f32            ZQA_NEON_STUB_ANY
#define vaddvq_f32           ZQA_NEON_STUB_ANY
#define vld1q_f32            ZQA_NEON_STUB_ANY
#define vst1q_f32            ZQA_NEON_STUB_ANY
#define vmulq_laneq_f32      ZQA_NEON_STUB_ANY
#define vfmaq_laneq_f32      ZQA_NEON_STUB_ANY
#define vdupq_n_f32          ZQA_NEON_STUB_ANY
#define vdupq_lane_f32       ZQA_NEON_STUB_ANY
#define vgetq_lane_f32       ZQA_NEON_STUB_ANY
#define vsetq_lane_f32       ZQA_NEON_STUB_ANY
#define vreinterpretq_f32_s32 ZQA_NEON_STUB_ANY
#define vreinterpretq_s32_f32 ZQA_NEON_STUB_ANY
#define vcgtq_f32            ZQA_NEON_STUB_ANY
#define vcltq_f32            ZQA_NEON_STUB_ANY
#define vceqq_f32            ZQA_NEON_STUB_ANY
#define vbslq_f32            ZQA_NEON_STUB_ANY
#define vrev64q_f32          ZQA_NEON_STUB_ANY
#define vcombine_f32         ZQA_NEON_STUB_ANY

/* --- f16（`__ARM_NEON_FP16` 才会走到，但有些文件无条件用了） --- */
#define vaddq_f16            ZQA_NEON_STUB_ANY
#define vsubq_f16            ZQA_NEON_STUB_ANY
#define vmulq_f16            ZQA_NEON_STUB_ANY
#define vminq_f16            ZQA_NEON_STUB_ANY
#define vmaxq_f16            ZQA_NEON_STUB_ANY
#define vfmaq_f16            ZQA_NEON_STUB_ANY
#define vaddq_f16x2          ZQA_NEON_STUB_ANY
#define vadd1q_f16           ZQA_NEON_STUB_ANY
#define vdivq_f16            ZQA_NEON_STUB_ANY
#define vld1q_f16            ZQA_NEON_STUB_ANY
#define vst1q_f16            ZQA_NEON_STUB_ANY
#define vdupq_n_f16          ZQA_NEON_STUB_ANY

/* --- 其它标量/向量搬运用 --- */
#define vgetq_lane_s32       ZQA_NEON_STUB_ANY
#define vsetq_lane_s32       ZQA_NEON_STUB_ANY
#define vreinterpretq_s32_u64 ZQA_NEON_STUB_ANY
#define vreinterpretq_u64_s32 ZQA_NEON_STUB_ANY
#define vld1q_u8             ZQA_NEON_STUB_ANY
#define vst1q_s32            ZQA_NEON_STUB_ANY
#define vmovl_s8             ZQA_NEON_STUB_ANY
#define vshlq_n_s32          ZQA_NEON_STUB_ANY

#endif /* ZQ_ARM_NEON_STUB_H_ */
