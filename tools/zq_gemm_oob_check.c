/* ASan 回归测试：GEMM 的 C 写回不得越界。
 *
 * 背景（2026-10-01）：m2n4 / m1n4 微内核的归约收尾 vshufps 0x44 只把 4 个结果
 * 放在低 4 个 lane，高 4 个 lane 里是重复值，而出口是一条 vmovups（16 字节）——
 * 每 4 列块多写 12 字节，越过 C 那一行的末尾；最后一行就是越过 C 缓冲区末尾。
 * 独立微基准里表现为 glibc 的 `munmap_chunk(): invalid pointer`。
 *
 * 这个测试把 C 放在 ASan 的堆上，越界写会直接被标成 heap-buffer-overflow。
 * 用法（在 WSL 下）：
 *   g++ -O1 -g -fsanitize=address -mavx2 -mfma -I. -I../ZQCNN -I../ZQ_GEMM \
 *       zq_gemm_oob_check.c ../ZQ_GEMM/math/*.c -o /tmp/oob && /tmp/oob
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "zq_gemm_32f_align_c_asm.h"
#include "zq_gemm_32f_align_c.h"

int main(void)
{
	/* N 取 4 的倍数, 让 m2n4/m1n4 的最后一个列块正好贴着行尾 */
	const int cases[][3] = {
		{2, 4, 64}, {2, 8, 64}, {4, 4, 64}, {5, 4, 32}, {8, 4, 16},
		{2, 12, 64}, {3, 4, 8}, {16, 4, 256}, {2, 100, 128}, {7, 4, 40},
	};
	int bad = 0;
	for (int ci = 0; ci < (int)(sizeof(cases) / sizeof(cases[0])); ci++)
	{
		const int M = cases[ci][0], N = cases[ci][1], K = cases[ci][2];
		const size_t asz = (size_t)M * K, bsz = (size_t)N * K, csz = (size_t)M * N;
		float* A = (float*)malloc(asz * sizeof(float));
		float* Bt = (float*)malloc(bsz * sizeof(float));
		float* C = (float*)malloc(csz * sizeof(float));
		float* Ref = (float*)malloc(csz * sizeof(float));
		for (size_t i = 0; i < asz; i++) A[i] = (float)((i % 7)) * 0.25f - 0.5f;
		for (size_t i = 0; i < bsz; i++) Bt[i] = (float)((i % 5)) * 0.5f - 1.0f;
		for (size_t i = 0; i < csz; i++) { C[i] = 0.f; Ref[i] = 0.f; }

		zq_gemm_32f_AnoTrans_Btrans_auto(M, N, K, A, K, Bt, K, Ref, N);
		/* 精确对齐到 4 字节的最小分配: 越界写会踩到 ASan redzone */
		free(Ref);
		Ref = NULL;
		zq_gemm_32f_AnoTrans_Btrans_auto_asm(M, N, K, A, K, Bt, K, C, N);

		/* 越界写没被发现的话, 至少要确认结果本身对 */
		double worst = 0;
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
			{
				double r = 0;
				for (int k = 0; k < K; k++)
					r += (double)A[(size_t)i * K + k] * (double)Bt[(size_t)j * K + k];
				double d = fabs(r - C[(size_t)i * N + j]);
				if (d > worst) worst = d;
			}
		printf("%2dx%2dx%3d  worst |ref-asm| = %.3e%s\n",
			M, N, K, worst, worst > 1e-3 ? "   <== WRONG" : "");
		if (worst > 1e-3) bad++;
		free(A); free(Bt); free(C);
	}
	printf("%s\n", bad ? "RESULT: FAIL" : "RESULT: PASS");
	return bad ? 1 : 0;
}
