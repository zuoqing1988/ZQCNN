#include "zq_gemm_32f_align_c.h"

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

#if __ARM_NEON 
#define SWAP_A_Bt \
	if ((long long)M*N < 0.1*((long long)M*N*K) && M + 8 < N) \
	{ \
		swap = 1; \
		A = oldB; \
		Bt = oldA; \
		lda = old_ldb; \
		ldb = old_lda; \
		M = old_N; \
		N = old_M; \
		ldc = N; \
		C = _aligned_malloc((size_t)M*N * sizeof(float), 32); \
		if (C == 0) \
		{ \
			swap = 0; \
			A = oldA; \
			Bt = oldB; \
			lda = old_lda; \
			ldb = old_ldb; \
			M = old_M; \
			N = old_N; \
			ldc = old_ldc; \
		} \
	}

#define SWAP_C \
	if (swap == 1) \
	{ \
		for (n = 0; n < N; n++) \
		{ \
			for (m = 0; m < M; m++) \
			{ \
				old_C[n*old_ldc + m] = C[m*ldc + n]; \
			} \
		} \
		_aligned_free(C); \
	}

	int zq_gemm_32f_AnoTrans_Btrans_special(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
	{
		const float* oldA = A, *oldB = Bt;
		float* old_C = C;
		int old_lda = lda, old_ldb = ldb, old_ldc = ldc, old_M = M, old_N = N;
		int m, n;
		int swap = 0;
		int handled = 0;

#if __ARM_NEON_ARMV8
		if (K == 16)
		{
			if ((M == 576 && N == 32) //det3-dw48-fast
				|| (M == 121 && N == 32) //det2-dw24-fast
				)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
		else if (K == 24)
		{
			if ((M >= 1 && N == 2) //det1-dw20-fast
				|| (M >= 1 && N == 4) //det1-dw20-fast
				)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N8(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
		else if (K == 28)
		{
			if (
				(M == 3136 && N == 64) //mobilefacenet
				|| (M == 8836 && N == 32) //det5-dw96-v2s
				|| (M % 2304 == 0 && N == 16) //det3-dw48-fast
				|| (M % 484 == 0 && N == 16) //det2-dw24-fast
				|| (M >= 2500 && N == 8) //det1-dw20-fast
				)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
		else if (K == 32)
		{
			if ((M == 8649 && N == 32) //det5-dw96-v2s
				|| (M == 2116 && N == 64) //det5-dw96-v2s
				|| (M == 144 && N == 64) //det3-dw48-fast
				|| (M == 25 && N == 64) //det2-dw24-fast
				)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
		else if (K == 64)
		{
			if (
				//(M == 3136 && N == 128) //mobilefacenet
				//||
				(M == 3136 && N == 64) //mobilefacenet-res2-6-10-2
				|| (M == 2025 && N == 64) //det5-dw96-v2s
				|| (M == 484 && N == 64) //det5-dw96-v2s
				|| (M == 441 && N == 64) //det5-dw96-v2s
				|| (M == 25 && N == 64) //det3-dw48-fast
				|| (M == 9 && N == 128) //det3-dw48-fast && det2-dw24-fast
				)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
		else if (K == 128)
		{
			if ((M == 16 && N == 256) //det5-dw96-v2s
				|| (M >= 1 && N == 2) //det3-dw48-fast && det2-dw24-fast
				|| (M >= 1 && N == 4) //det3-dw48-fast && det2-dw24-fast
				)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
		else if (K == 256)
		{
			if ((M == 9 && N == 256) //det5-dw96-v2s
				|| (M == 1 && N == 212) //det5-dw96-v2s
				)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
		else if (K == 512)
		{
			if ((M == 1 && (N == 128 || N == 256 || N == 512)) ////mobilefacenet & mobilefacenet-res2-6-10-2
				|| (M == 49 && N == 512) //mobilefacenet-res2-6-10-2
				)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
#endif
		return handled;
	}

	
#if __ARM_NEON_ARMV8
	void zq_gemm_32f_AnoTrans_Btrans_auto(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
	{
		const float* oldA = A, *oldB = Bt;
		float* old_C = C;
		int old_lda = lda, old_ldb = ldb, old_ldc = ldc, old_M = M, old_N = N;
		int m, n;
		int swap = 0;
		int handled = 0;
		if (K == 8)
		{
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
			SWAP_C;
		}
		else if (K == 16)
		{
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
			SWAP_C;
		}
		else if (K == 24)
		{
			if (N == 8 || N == 16)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N8(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
			else
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
		else if (K == 27) //3*3*3
		{
			if (N <= 16)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
			else if (N <= 128)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
		else if (K == 28)
		{
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
			SWAP_C;
		}
		else if (K == 32)
		{

			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
			SWAP_C;
		}
		else if (K == 64)
		{
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
			SWAP_C;
		}
		else if (K == 72) // 3*3*8
		{
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
			SWAP_C;
		}

		//back up methods
		if (handled == 0)
		{
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
			SWAP_C;
		}
	}

#else // not ARMV8
	
	void zq_gemm_32f_AnoTrans_Btrans_auto(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
	{
		const float* oldA = A, *oldB = Bt;
		float* old_C = C;
		int old_lda = lda, old_ldb = ldb, old_ldc = ldc, old_M = M, old_N = N;
		int m, n;
		int swap = 0;
		int handled = 0;
		if (K == 16)
		{
			if (N >= 8)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
		else if (K == 27) //3*3*3
		{
			if (N <= 16)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
			else if (N <= 128)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
		else if (K == 28)
		{
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
			SWAP_C;
		}
		else if (K == 32)
		{
			if (N >= 8)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
		else if (K == 64)
		{
			if (N >= 8)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
		else if (K == 72) // 3*3*8
		{
			if (N <= 64)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
			else if (N <= 128)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
		else if (K == 128)
		{
			if (N >= 256)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
		else if (K == 144) // 3*3*16
		{
			if (N <= 32)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
			else if (N <= 128)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
		else if (K == 256)
		{
			if (N >= 256)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
		else if (K == 512)
		{
			if (N >= 256)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}
		else if (K == 1024)
		{
			if (N >= 256)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
				SWAP_C;
			}
		}

		//back up methods
		if (handled == 0)
		{

			if (K <= 64)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;

			}


			SWAP_C;
		}
	}
#endif// __ARM_NEON_ARMV8
	
#else // not __ARM_NEON

	/* ------------------------------------------------------------------
	 * 附录 BO：K 方向那一族内核**隐含假设 lda(=K) 是向量宽度的整数倍**。
	 *
	 *   ① k 方向循环全程用 zq_mm_load_ps（= _mm_load_ps = 对齐的 movaps）。
	 *      `lda = K`，K 不是 4/8 的倍数时，第 m 行的基址 A + m*lda 就不对齐，
	 *      movaps 遇到不对齐的地址会 #GP -> SIGSEGV。
	 *      raw 头里 zq_mm_load_ps 出现 250 次、zq_mm_loadu_ps 0 次 ——
	 *      那个非对齐的宏就在旁边定义着，一个都没用。
	 *   ② 循环上界写的是 padK = ceil(K/align)*align 而不是 K，紧跟的标量收尾
	 *      `for (; k < K; k++)` 在 K < padK 时一次都不执行 —— 就算对齐碰巧成立，
	 *      也会多读 padK-K 个元素（中间行读到下一行的数据进和里，最后一行读到缓冲区外）。
	 *
	 * 同仓库的汇编版 `zq_gemm_32f_align_c_asm.c` 全程 vmovups、零个 vmovaps，
	 * 所以对任意 lda/ldb 都安全 —— 同一份数学，一份假设了对齐、一份没假设。
	 *
	 * 实测（附录 BO.6，3240 个形状）：x86(AVX2) 下
	 *     K % 8 == 0 -> 1800 个格子零崩溃、零错值（含 M=3136/N=512 的 production 形状）
	 *     K % 8 != 0 -> 784 个崩溃（K=17/27/33/108/150；K=24/28 靠专用内核幸免）
	 *
	 * 所以在入口把 K 不是向量宽度整数倍的形状挡到下面这条朴素路径上。
	 * **这不会造成性能回退**：K%align==0 的那一片现在全是好的，一条都碰不到；
	 * 被改道的那一片现在**全是崩溃**，不存在"原来更快"这回事。
	 * 判别式必须用**各档自己的 align**（AVX 档 8 / SSE 档 4），写死 8 会在
	 * 只编到 SSE 的配置下拦掉一批本来安全的形状。
	 * ------------------------------------------------------------------ */
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#define ZQ_GEMM_K_ALIGN 8
#elif ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#define ZQ_GEMM_K_ALIGN 4
#else
#define ZQ_GEMM_K_ALIGN 1
#endif

	static void zq_gemm_32f_AnoTrans_Btrans_fallback(int M, int N, int K,
		const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
	{
		/* C[m][n] = sum_k A[m*lda+k] * Bt[n*ldb+k]
		   语义与上面那些内核一致：**覆盖**写 C（调用点都是 cblas 的 beta=0），
		   不是累加，所以不需要先清零。
		   累加用 double：这条路现在只能被"原本会崩"的形状走到，正确性优先；
		   顺带它比 SIMD 那些树形归约还准一点。真要是 3x3x3 卷积跑到很热，
		   这里再上分块/向量化（并配一轮 MKL 对标），不是先猜着优化。 */
		int m, n, k;
		if (M <= 0 || N <= 0)
			return;
		for (n = 0; n < N; n++)
		{
			const float* bn = Bt + (size_t)n * ldb;
			for (m = 0; m < M; m++)
			{
				const float* a = A + (size_t)m * lda;
				double s = 0;
				for (k = 0; k < K; k++)
					s += (double)a[k] * (double)bn[k];
				C[(size_t)m * ldc + n] = (float)s;
			}
		}
	}

	void zq_gemm_32f_AnoTrans_Btrans_auto(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
	{
		const float* oldA = A, *oldB = Bt;
		float* old_C = C;
		int old_lda = lda, old_ldb = ldb, old_ldc = ldc, old_M = M, old_N = N;
		int m, n;
		int swap = 0;
		/* 附录 BO：必须在下面那个 SWAP_A_Bt 之前判 —— 换了 A/Bt 之后
		   lda/ldb 互换了，判据要按原始的 K 走才说得清（K 换了也没用，
		   swap 不改变 K）。 */
		if (K % ZQ_GEMM_K_ALIGN != 0)
		{
			zq_gemm_32f_AnoTrans_Btrans_fallback(M, N, K, A, lda, Bt, ldb, C, ldc);
			return;
		}
		if ((long long)M*N < 0.1*((long long)M*N*K) && M + 8 < N)
		{
			swap = 1;
			A = oldB;
			Bt = oldA;
			lda = old_ldb;
			ldb = old_lda;
			M = old_N;
			N = old_M;
			ldc = N;
			C = _aligned_malloc((size_t)M*N * sizeof(float), 32);
			if (C == 0)
			{
				swap = 0;
				A = oldA;
				Bt = oldB;
				lda = old_lda;
				ldb = old_ldb;
				M = old_M;
				N = old_N;
				ldc = old_ldc;
			}
		}
		int handled = 0;


#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX

		if (K == 16)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 27) //3*3*3
		{
			if (N <= 16)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else if (N <= 128)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 28)
		{
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
		}
		else if (K == 32)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 64)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 72) // 3*3*8
		{
			if (N <= 64)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else if (N <= 128)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 128)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 144) // 3*3*16
		{
			if (N <= 32)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else if (N <= 128)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 256)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 512)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 1024)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}

		//back up methods
		if (handled == 0)
		{
			if (K <= 64)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else
			{
				zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}

#elif ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		if (K == 16)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 27) //3*3*3
		{
			if (N <= 16)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else if (N <= 128)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 28)
		{
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
		}
		else if (K == 32)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 64)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 72) // 3*3*8
		{
			if (N <= 64)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else if (N <= 128)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 128)
		{
			if (N >= 256)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 144) // 3*3*16
		{
			if (N <= 32)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else if (N <= 128)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 256)
		{
			if (N >= 256)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 512)
		{
			if (N >= 256)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 1024)
		{
			if (N >= 256)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}

		//back up methods
		if (handled == 0)
		{
			if (K <= 64)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
#else
		zq_gemm_32f_align0_AnoTrans_Btrans(M, N, K, A, lda, Bt, ldb, C, ldc);
#endif

		if (swap == 1)
		{
			for (n = 0; n < N; n++)
			{
				for (m = 0; m < M; m++)
				{
					old_C[n*old_ldc + m] = C[m*ldc + n];
				}
			}
			_aligned_free(C);
		}
	}


#endif

#if defined(__cplusplus) || defined(c_plusplus) 
	}
#endif