/*
 * zq_libm_compat.c
 *
 * 3rdparty/lib/libncnn.a 是用 clang + OpenMP 编的，它引用了 clang compiler-rt 里的
 * __*_finite 系列数学函数（__exp_finite / __log_finite / ...）。这些符号不在
 * glibc 的 libm 里，用 gcc 链接 libncnn.a 时会报 undefined reference。
 *
 * 这里按 clang 的语义补一份实现：_finite 版本与标准函数唯一的区别是
 * NaN/Inf 输入时不设置 errno，对推理结果没有任何影响。
 *
 * 注意 libncnn.a 里的调用点是 OpenMP SIMD 区域，需要的是 __exp_finite 的
 * **向量化体**（符号名形如 _ZGVbN2v___exp_finite），所以本文件用
 * `#pragma omp declare simd` 声明，让 gcc 同样生成一份；
 * 因此本文件必须带 -fopenmp 编译（见 ZQCNN/CMakeLists.txt）。
 */

#include <math.h>

#if defined(_OPENMP)
#  pragma omp declare simd
#  define ZQ_SHIM
#else
#  define ZQ_SHIM
#endif

#if !defined(_WIN32) && !defined(__APPLE__)

ZQ_SHIM double __exp_finite(double x) { return exp(x); }
ZQ_SHIM float  __expf_finite(float x) { return expf(x); }
ZQ_SHIM double __log_finite(double x) { return log(x); }
ZQ_SHIM float  __logf_finite(float x) { return logf(x); }
ZQ_SHIM double __log2_finite(double x) { return log2(x); }
ZQ_SHIM float  __log2f_finite(float x) { return log2f(x); }
ZQ_SHIM double __log10_finite(double x) { return log10(x); }
ZQ_SHIM float  __log10f_finite(float x) { return log10f(x); }
ZQ_SHIM double __pow_finite(double x, double y) { return pow(x, y); }
ZQ_SHIM float  __powf_finite(float x, float y) { return powf(x, y); }
ZQ_SHIM double __sin_finite(double x) { return sin(x); }
ZQ_SHIM float  __sinf_finite(float x) { return sinf(x); }
ZQ_SHIM double __cos_finite(double x) { return cos(x); }
ZQ_SHIM float  __cosf_finite(float x) { return cosf(x); }
ZQ_SHIM double __tan_finite(double x) { return tan(x); }
ZQ_SHIM float  __tanf_finite(float x) { return tanf(x); }
ZQ_SHIM double __asin_finite(double x) { return asin(x); }
ZQ_SHIM float  __asinf_finite(float x) { return asinf(x); }
ZQ_SHIM double __acos_finite(double x) { return acos(x); }
ZQ_SHIM float  __acosf_finite(float x) { return acosf(x); }
ZQ_SHIM double __atan_finite(double x) { return atan(x); }
ZQ_SHIM float  __atanf_finite(float x) { return atanf(x); }
ZQ_SHIM double __atan2_finite(double x, double y) { return atan2(x, y); }
ZQ_SHIM float  __atan2f_finite(float x, float y) { return atan2f(x, y); }
ZQ_SHIM double __sinh_finite(double x) { return sinh(x); }
ZQ_SHIM float  __sinhf_finite(float x) { return sinhf(x); }
ZQ_SHIM double __cosh_finite(double x) { return cosh(x); }
ZQ_SHIM float  __coshf_finite(float x) { return coshf(x); }
ZQ_SHIM double __tanh_finite(double x) { return tanh(x); }
ZQ_SHIM float  __tanhf_finite(float x) { return tanhf(x); }
ZQ_SHIM double __fabs_finite(double x) { return fabs(x); }
ZQ_SHIM float  __fabsf_finite(float x) { return fabsf(x); }
ZQ_SHIM double __fmod_finite(double x, double y) { return fmod(x, y); }
ZQ_SHIM float  __fmodf_finite(float x, float y) { return fmodf(x, y); }
ZQ_SHIM double __round_finite(double x) { return round(x); }
ZQ_SHIM float  __roundf_finite(float x) { return roundf(x); }
ZQ_SHIM double __trunc_finite(double x) { return trunc(x); }
ZQ_SHIM float  __truncf_finite(float x) { return truncf(x); }
ZQ_SHIM double __floor_finite(double x) { return floor(x); }
ZQ_SHIM float  __floorf_finite(float x) { return floorf(x); }
ZQ_SHIM double __ceil_finite(double x) { return ceil(x); }
ZQ_SHIM float  __ceilf_finite(float x) { return ceilf(x); }
ZQ_SHIM double __cbrt_finite(double x) { return cbrt(x); }
ZQ_SHIM float  __cbrtf_finite(float x) { return cbrtf(x); }
ZQ_SHIM double __hypot_finite(double x, double y) { return hypot(x, y); }
ZQ_SHIM float  __hypotf_finite(float x, float y) { return hypotf(x, y); }
ZQ_SHIM double __expm1_finite(double x) { return expm1(x); }
ZQ_SHIM float  __expm1f_finite(float x) { return expm1f(x); }
ZQ_SHIM double __log1p_finite(double x) { return log1p(x); }
ZQ_SHIM float  __log1pf_finite(float x) { return log1pf(x); }

#endif
