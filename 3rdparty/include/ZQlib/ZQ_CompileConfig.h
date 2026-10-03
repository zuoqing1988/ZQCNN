/* ZQlib 的可移植性配置头 —— 附录 ES
 *
 * 为什么需要它
 * ------------
 * ZQlib 里有 56 个头、**381 处**用了 `__min` / `__max`。这两个名字是
 * **MSVC 的内建**，gcc / clang 根本没有。
 *
 * 为什么 121 个"能单独编译"的头里有 49 个用了它却还能过：
 * gcc 对**未声明的标识符**在非模板上下文里只当"隐式函数声明"（`-fpermissive`
 * 才降级成警告），而在**模板**里是两阶段查找 —— 只要实参不依赖模板参数，
 * 就是一个**硬错误**。所以：
 *
 *     非模板函数里用 __max  -> 编得过（带隐式声明）
 *     模板函数里用 __max    -> 编不过
 *
 * 于是"用了 __max 就编不过"是错的，"在模板里用了才编不过"才对。
 * 实际卡住的是这 4 个头：
 *     ZQ_CameraCalibrationMulti.h / ZQ_MultiCamCalibration.h
 *     ZQ_OpticalFlow.h / ZQ_StereoRectify.h（后者 include 前者）
 *
 * 修法与 `ZQCNN/ZQ_CNN_CompileConfig.h:100-106` **完全一致**（同一个作者、
 * 同一个写法），那边早就有，只是 ZQlib 侧从来没有：
 *
 *     #ifndef __min
 *     #define __min(a,b) ((a)<(b)?(a):(b))
 *     #endif
 *
 * `#ifndef` 是必需的：MSVC 下这两个名字**是**编译器内建，
 * 重复定义会报错/警告。
 *
 * 注意这两个宏会把实参**求值两次**（`f(a++), f(b++)` 会出错）。
 * 这是 MSVC 原版的既有行为，此处保持一致，不引入新的行为差异。
 * 真要修得改成 inline 函数，但那是**行为改变**，不在本次审计范围内。
 */
#ifndef ZQ_COMPILE_CONFIG_H_
#define ZQ_COMPILE_CONFIG_H_

// MSVC 之外的编译器补上这两个内建。
// MSVC 下它们由编译器提供，所以必须用 #ifndef 保护。
#ifndef __min
#define __min(a,b) ((a)<(b)?(a):(b))
#endif

#ifndef __max
#define __max(a,b) ((a)>(b)?(a):(b))
#endif

// ZQlib 里有 `unsigned long` / `size_t` 混用的地方，MSVC 的这两个内建
// 在 gcc 下同样不存在。ZQCNN 那份配置头里还有 `__int64` 与 `_aligned_malloc`
// 的等价定义；ZQlib 目前没有用到那两个，所以这里不抄 ——
// **只抄真正需要的**，免得留一个没人验证过的定义。
#endif // ZQ_COMPILE_CONFIG_H_
