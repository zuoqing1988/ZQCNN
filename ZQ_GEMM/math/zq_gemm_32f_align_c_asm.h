/*
 * zq_gemm_32f_align_c_asm.h
 *
 * ZQ_GEMM 手写内联汇编 (inline assembly) 内核 —— 与 intrinsic 版
 * (zq_gemm_32f_align_c.c + zq_gemm_32f_align_c_raw.h) 完全并列、互不干扰,
 * 用于逐函数 A/B 对比性能与数值。
 *
 * 语义与 intrinsic 版严格一致:
 *   C = A * Bt   (C 被覆盖写入, 不是累加)
 *   A : M x K, 行主序, lda >= K
 *   Bt: N x K, 行主序, ldb >= K     (注意是 N x K, 不是 K x N)
 *   C : M x N, 行主序, ldc >= N
 *
 * 与 intrinsic 版的唯一契约差异:
 *   本文件的 AVX 实现只读取 [0, K) 范围内的数据 (K 不是 8 的倍数时,
 *   剩余 1~7 个元素用 C 标量循环补齐), 因此不像 intrinsic 版那样
 *   要求 A / Bt 的行尾补零到 SIMD 宽度的整数倍。intrinsic 版仍然需要,
 *   两边对比时按 intrinsic 版的约定分配缓冲即可。
 *
 * 命名约定: 与 intrinsic 版同名 + "_asm" 后缀, 例如
 *   zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N4_asm
 * 对应
 *   zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N4
 *
 * 平台支持:
 *   x86-64 + MSVC          : 函数体内 __asm { } (Intel 语法)
 *   x86-64 + GCC / Clang   : 函数体内 __asm__ volatile (AT&T 语法)
 *   其他平台 (ARM/NEON、无 AVX) : 这些函数退化为直接调用 intrinsic 版
 *                              zq_gemm_32f_AnoTrans_Btrans_auto, 符号始终存在
 */

#ifndef _ZQ_GEMM_32F_ALIGN_C_ASM_H_
#define _ZQ_GEMM_32F_ALIGN_C_ASM_H_

#include "ZQ_CNN_CompileConfig.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N4_asm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N8_asm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N4_asm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N8_asm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N4_asm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N8_asm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_N4_asm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

	/* 与 zq_gemm_32f_AnoTrans_Btrans_auto 同签名, 同语义 */
	void zq_gemm_32f_AnoTrans_Btrans_auto_asm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif /* _ZQ_GEMM_32F_ALIGN_C_ASM_H_ */
