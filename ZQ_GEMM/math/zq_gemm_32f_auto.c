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

	void zq_gemm_32f_AnoTrans_Btrans_auto(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
	{
		const float* oldA = A, *oldB = Bt;
		float* old_C = C;
		int old_lda = lda, old_ldb = ldb, old_ldc = ldc, old_M = M, old_N = N;
		int m, n;
		int swap = 0;
		int handled = 0;

#if __ARM_NEON_ARMV8
		if (K == 8)
		{
#if ZQ_CNN_USE_BLAS_GEMM
#else
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
#endif
		}
		else if (K == 16)
		{
#if ZQ_CNN_USE_BLAS_GEMM
			if (N == 32 || N == 64)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
#else
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
#endif
		}
		else if (K == 24)
		{
			if (N == 8 || N == 16)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N8(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 27) //3*3*3
		{
			if (N <= 16)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else if (N <= 128)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 28)
		{
#if ZQ_CNN_USE_BLAS_GEMM
			if (N == 8 || N == 16 || N == 32)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
#else
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
#endif
		}
		else if (K == 32)
		{
#if ZQ_CNN_USE_BLAS_GEMM
			if (N == 32)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
#else
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
#endif
		}
		else if (K == 64)
		{
#if ZQ_CNN_USE_BLAS_GEMM
			if (N == 32)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
#else
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
#endif
		}
		else if (K == 72) // 3*3*8
		{
#if ZQ_CNN_USE_BLAS_GEMM
#else
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
#endif
		}
		else if (K == 128)
		{
#if ZQ_CNN_USE_BLAS_GEMM
#else
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
#endif
		}
		else if (K == 144) // 3*3*16
		{
#if ZQ_CNN_USE_BLAS_GEMM
#else
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
#endif
		}
		else if (K == 256)
		{
#if ZQ_CNN_USE_BLAS_GEMM
#else
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
#endif
		}
		else if (K == 512)
		{
#if ZQ_CNN_USE_BLAS_GEMM
#else
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
#endif
		}
		else if (K == 1024)
		{
#if ZQ_CNN_USE_BLAS_GEMM
#else
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
#endif
		}

		//back up methods
		if (handled == 0)
		{
#if ZQ_CNN_USE_BLAS_GEMM
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, A, lda, Bt, ldb, 0, C, ldc);
#else
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
#endif//ZQ_CNN_USE_BLAS_GEMM
		}

#else // not ARMV8

		if (K == 16)
		{
			if (N >= 8)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 27) //3*3*3
		{
			if (N <= 16)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else if (N <= 128)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 28)
		{
			SWAP_A_Bt;
			zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
			handled = 1;
		}
		else if (K == 32)
		{
			if (N >= 8)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 64)
		{
			if (N >= 8)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 72) // 3*3*8
		{
			if (N <= 64)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else if (N <= 128)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 128)
		{
			if (N >= 256)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 144) // 3*3*16
		{
			if (N <= 32)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else if (N <= 128)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 256)
		{
			if (N >= 256)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 512)
		{
			if (N >= 256)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 1024)
		{
			if (N >= 256)
			{
				SWAP_A_Bt;
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}

		//back up methods
		if (handled == 0)
		{
#if ZQ_CNN_USE_BLAS_GEMM
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, A, lda, Bt, ldb, 0, C, ldc);
#else
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
#endif
		}
#endif// __ARM_NEON_ARMV8


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