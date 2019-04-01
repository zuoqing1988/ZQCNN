#include "zq_gemm_32f_align_c.h"

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

#if __ARM_NEON 
#define SWAP_A_Bt \
	if (M*N < 0.1*(M*N*K) && M + 8 < N) \
	{ \
		swap = 1; \
		A = oldB; \
		Bt = oldA; \
		lda = old_ldb; \
		ldb = old_lda; \
		M = old_N; \
		N = old_M; \
		ldc = N; \
		C = _aligned_malloc(M*N * sizeof(float), 32); \
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

	void zq_gemm_32f_AnoTrans_Btrans_auto(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
	{
		const float* oldA = A, *oldB = Bt;
		float* old_C = C;
		int old_lda = lda, old_ldb = ldb, old_ldc = ldc, old_M = M, old_N = N;
		int m, n;
		int swap = 0;
		if (M*N < 0.1*(M*N*K) && M + 8 < N)
		{
			swap = 1;
			A = oldB;
			Bt = oldA;
			lda = old_ldb;
			ldb = old_lda;
			M = old_N;
			N = old_M;
			ldc = N;
			C = _aligned_malloc(M*N * sizeof(float), 32);
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