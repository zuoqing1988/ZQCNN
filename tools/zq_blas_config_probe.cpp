// 配置宏取值探针（附录 GL.5）。
//
// 由 `tools/check_blas_config.py` 在若干组 -D 下编译并运行它，
// 断言每个后端开关的**最终取值**。两个用途：
//
//   1. `#error` 掉任何"未定义"的开关。`#if` 里未定义等价于 0，
//      被**取值**时（printf / 赋值 / 常量表达式）却直接编不过 ——
//      2026-10-03 实测：`-DZQ_CNN_USE_ARM_NEON` 之前就会在这里炸。
//   2. 打印出来，让脚本比对。**只看编译过不过是不够的**：
//      头文件原来把命令行的 `-DZQ_CNN_USE_BLAS_GEMM=1` 静默按回 0，
//      编译是**成功**的 —— 错的是值。
//
// 这里的宏清单**必须**与 ZQCNN/ZQ_CNN_CompileConfig.h 里被 `#ifndef` 兜底的
// 那六个一一对应；头文件里多一个开关而这里没查，就等于那个开关没人看。
#include "ZQ_CNN_CompileConfig.h"
#include <cstdio>

#ifndef ZQ_CNN_USE_BLAS_GEMM
#error "ZQ_CNN_USE_BLAS_GEMM is not defined"
#endif
#ifndef ZQ_CNN_USE_MKL_GEMM
#error "ZQ_CNN_USE_MKL_GEMM is not defined"
#endif
#ifndef ZQ_CNN_USE_ZQ_GEMM
#error "ZQ_CNN_USE_ZQ_GEMM is not defined"
#endif
#ifndef ZQ_CNN_USE_SSETYPE
#error "ZQ_CNN_USE_SSETYPE is not defined"
#endif
#ifndef ZQ_CNN_USE_FMADD128
#error "ZQ_CNN_USE_FMADD128 is not defined"
#endif
#ifndef ZQ_CNN_USE_FMADD256
#error "ZQ_CNN_USE_FMADD256 is not defined"
#endif

int main() {
    printf("BLAS=%d MKL=%d ZQ_GEMM=%d SSETYPE=%d FMADD128=%d FMADD256=%d\n",
           ZQ_CNN_USE_BLAS_GEMM, ZQ_CNN_USE_MKL_GEMM, ZQ_CNN_USE_ZQ_GEMM,
           ZQ_CNN_USE_SSETYPE, ZQ_CNN_USE_FMADD128, ZQ_CNN_USE_FMADD256);
    return 0;
}
