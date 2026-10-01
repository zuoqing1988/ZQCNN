/*
 * zq_libm_compat.c
 *
 * 3rdparty/lib/libncnn.a 是用 clang 编译的，它引用了 clang compiler-rt 里的
 * __*_finite 系列数学函数（__exp_finite / __log_finite / ...）。这些符号不在
 * glibc 的 libm 里，用 gcc 链接 libncnn.a 时会报 undefined reference。
 *
 * 这里按 clang 的语义补一份实现：_finite 版本与标准函数唯一的区别是
 * NaN/Inf 输入时不设置 errno，对推理结果没有任何影响。
 */

#include <math.h>

#if !defined(_WIN32) && !defined(__APPLE__)

double __exp_finite(double x) { return exp(x); }
float  __expf_finite(float x) { return expf(x); }
double __log_finite(double x) { return log(x); }
float  __logf_finite(float x) { return logf(x); }
double __log2_finite(double x) { return log2(x); }
float  __log2f_finite(float x) { return log2f(x); }
double __log10_finite(double x) { return log10(x); }
float  __log10f_finite(float x) { return log10f(x); }
double __pow_finite(double x, double y) { return pow(x, y); }
float  __powf_finite(float x, float y) { return powf(x, y); }
double __sin_finite(double x) { return sin(x); }
float  __sinf_finite(float x) { return sinf(x); }
double __cos_finite(double x) { return cos(x); }
float  __cosf_finite(float x) { return cosf(x); }
double __tan_finite(double x) { return tan(x); }
float  __tanf_finite(float x) { return tanf(x); }
double __asin_finite(double x) { return asin(x); }
float  __asinf_finite(float x) { return asinf(x); }
double __acos_finite(double x) { return acos(x); }
float  __acosf_finite(float x) { return acosf(x); }
double __atan_finite(double x) { return atan(x); }
float  __atanf_finite(float x) { return atanf(x); }
double __atan2_finite(double x, double y) { return atan2(x, y); }
float  __atan2f_finite(float x, float y) { return atan2f(x, y); }
double __sinh_finite(double x) { return sinh(x); }
float  __sinhf_finite(float x) { return sinhf(x); }
double __cosh_finite(double x) { return cosh(x); }
float  __coshf_finite(float x) { return coshf(x); }
double __tanh_finite(double x) { return tanh(x); }
float  __tanhf_finite(float x) { return tanhf(x); }
double __fabs_finite(double x) { return fabs(x); }
float  __fabsf_finite(float x) { return fabsf(x); }
double __fmod_finite(double x, double y) { return fmod(x, y); }
float  __fmodf_finite(float x, float y) { return fmodf(x, y); }
double __round_finite(double x) { return round(x); }
float  __roundf_finite(float x) { return roundf(x); }
double __trunc_finite(double x) { return trunc(x); }
float  __truncf_finite(float x) { return truncf(x); }
double __floor_finite(double x) { return floor(x); }
float  __floorf_finite(float x) { return floorf(x); }
double __ceil_finite(double x) { return ceil(x); }
float  __ceilf_finite(float x) { return ceilf(x); }
double __cbrt_finite(double x) { return cbrt(x); }
float  __cbrtf_finite(float x) { return cbrtf(x); }
double __hypot_finite(double x, double y) { return hypot(x, y); }
float  __hypotf_finite(float x, float y) { return hypotf(x, y); }
double __expm1_finite(double x) { return expm1(x); }
float  __expm1f_finite(float x) { return expm1f(x); }
double __log1p_finite(double x) { return log1p(x); }
float  __log1pf_finite(float x) { return log1pf(x); }

#endif
