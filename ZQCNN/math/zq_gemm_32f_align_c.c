#include "..\ZQ_CNN_CompileConfig.h"
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#include <mmintrin.h> //MMX  
#include <xmmintrin.h> //SSE(include mmintrin.h)  
#include <emmintrin.h> //SSE2(include xmmintrin.h)  
#include <pmmintrin.h> //SSE3(include emmintrin.h)  
#include <tmmintrin.h>//SSSE3(include pmmintrin.h)  
#include <smmintrin.h>//SSE4.1(include tmmintrin.h)  
#include <nmmintrin.h>//SSE4.2(include smmintrin.h)  
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#include <wmmintrin.h>//AES(include nmmintrin.h)  
#include <immintrin.h>//AVX(include wmmintrin.h)  
#include <intrin.h>//(include immintrin.h)  
#endif
#include "zq_gemm_32f_align_c.h"

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	void zq_gemm_32f_AnoTrans_Btrans_auto(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
	{
		int handled = 0;
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		
		if (K == 16)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 27) //3*3*3
		{
			if (N <= 16)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else if(N <= 128)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 32)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 64)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 72) // 3*3*8
		{
			if (N <= 64)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else if (N <= 128)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 128)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align256bit_AnoTrans_Btrans_M4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 144) // 3*3*16
		{
			if (N <= 32)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else if (N <= 128)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 256)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align256bit_AnoTrans_Btrans_M4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 512)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align256bit_AnoTrans_Btrans_M4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 1024)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align256bit_AnoTrans_Btrans_M4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}

		//back up methods
		if (handled == 0)
		{
			if (K <= 64)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else
			{
				zq_gemm_32f_align256bit_AnoTrans_Btrans_M4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		
#elif ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
		if (K == 16)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 27) //3*3*3
		{
			if (N <= 16)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else if (N <= 128)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 32)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 64)
		{
			if (N >= 8)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 72) // 3*3*8
		{
			if (N <= 64)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else if (N <= 128)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 128)
		{
			if (N >= 256)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 144) // 3*3*16
		{
			if (N <= 32)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M1(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else if (N <= 128)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 256)
		{
			if (N >= 256)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 512)
		{
			if (N >= 256)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
		else if (K == 1024)
		{
			if (N >= 256)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}

		//back up methods
		if (handled == 0)
		{
			if (K <= 64)
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M2(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
			else
			{
				zq_gemm_32f_align128bit_AnoTrans_Btrans_M4(M, N, K, A, lda, Bt, ldb, C, ldc);
				handled = 1;
			}
		}
#else
		zq_gemm_32f_align0_AnoTrans_Btrans(M, N, K, A, lda, Bt, ldb, C, ldc);
#endif
	}

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#define zq_mm_load_ps _mm_load_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_set1_ps _mm_set1_ps
#define zq_mm_setzero_ps _mm_setzero_ps
#define zq_mm_type __m128
#define zq_mm_align_size 4
#define zq_mm_align_size2 8
#define zq_mm_align_size3 12
#define zq_mm_align_size4 16
#define zq_mm_align_size5 20
#define zq_mm_align_size6 24
#define zq_mm_align_size7 28
#define zq_mm_align_size8 32
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3])
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps _mm_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm_add_ps(_mm_mul_ps(A, B), C)
#endif
#define zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNgeneral zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNgeneral 
#define zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNgeneral zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNgeneral 
#define zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNgeneral zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_caseNgeneral 
#define zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign8 zq_gemm_32f_align128it_AnoTrans_Btrans_M1_caseNdiv4_Kdiv32
#define zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign8 zq_gemm_32f_align128it_AnoTrans_Btrans_M2_caseNdiv4_Kdiv32
#define zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign8 zq_gemm_32f_align128it_AnoTrans_Btrans_M4_caseNdiv4_Kdiv32
#define zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign4 zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv4_Kdiv16
#define zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign4 zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv4_Kdiv16
#define zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign4 zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_caseNdiv4_Kdiv16
#define zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_Kgeneral zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv4_Kgeneral
#define zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_Kgeneral zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv4_Kgeneral
#define zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_Kgeneral zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_caseNdiv4_Kgeneral
#define zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign8 zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv8_Kdiv32
#define zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign8 zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv8_Kdiv32
#define zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign4 zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv8_Kdiv16
#define zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign4 zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv8_Kdiv16
#define zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_Kgeneral zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv8_Kgeneral
#define zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_Kgeneral zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv8_Kgeneral
#define zq_gemm_32f_align_AnoTrans_Btrans_M1 zq_gemm_32f_align128bit_AnoTrans_Btrans_M1
#define zq_gemm_32f_align_AnoTrans_Btrans_M2 zq_gemm_32f_align128bit_AnoTrans_Btrans_M2
#define zq_gemm_32f_align_AnoTrans_Btrans_M4 zq_gemm_32f_align128bit_AnoTrans_Btrans_M4
#include "zq_gemm_32f_align_c_raw.h"

#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_set1_ps
#undef zq_mm_setzero_ps
#undef zq_mm_type
#undef zq_mm_align_size
#undef zq_mm_align_size2
#undef zq_mm_align_size3
#undef zq_mm_align_size4
#undef zq_mm_align_size5
#undef zq_mm_align_size6
#undef zq_mm_align_size7
#undef zq_mm_align_size8
#undef zq_final_sum_q
#undef zq_mm_fmadd_ps
#undef zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNgeneral
#undef zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNgeneral
#undef zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNgeneral
#undef zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign8
#undef zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign8
#undef zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign8
#undef zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign4
#undef zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign4
#undef zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign4
#undef zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_Kgeneral
#undef zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_Kgeneral
#undef zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_Kgeneral
#undef zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign8
#undef zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign8
#undef zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign4
#undef zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign4
#undef zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_Kgeneral
#undef zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_Kgeneral
#undef zq_gemm_32f_align_AnoTrans_Btrans_M1
#undef zq_gemm_32f_align_AnoTrans_Btrans_M2
#undef zq_gemm_32f_align_AnoTrans_Btrans_M4
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX

#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_set1_ps _mm256_set1_ps
#define zq_mm_setzero_ps _mm256_setzero_ps
#define zq_mm_type __m256
#define zq_mm_align_size 8
#define zq_mm_align_size2 16
#define zq_mm_align_size3 24
#define zq_mm_align_size4 32
#define zq_mm_align_size5 40
#define zq_mm_align_size6 48
#define zq_mm_align_size7 56
#define zq_mm_align_size8 64
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])
#if ZQ_CNN_USE_FMADD256
#define zq_mm_fmadd_ps _mm256_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#endif
#define zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNgeneral zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNgeneral 
#define zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNgeneral zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNgeneral 
#define zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNgeneral zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNgeneral 
#define zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign8 zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv4_Kdiv64
#define zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign8 zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv4_Kdiv64
#define zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign8 zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv4_Kdiv64
#define zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign4 zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv4_Kdiv32
#define zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign4 zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv4_Kdiv32
#define zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign4 zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv4_Kdiv32
#define zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_Kgeneral zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv4_Kgeneral
#define zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_Kgeneral zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv4_Kgeneral
#define zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_Kgeneral zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv4_Kgeneral
#define zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign8 zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv8_Kdiv64
#define zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign8 zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv8_Kdiv64
#define zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign4 zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv8_Kdiv32
#define zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign4 zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv8_Kdiv32
#define zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_Kgeneral zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv8_Kgeneral
#define zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_Kgeneral zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv8_Kgeneral
#define zq_gemm_32f_align_AnoTrans_Btrans_M1 zq_gemm_32f_align256bit_AnoTrans_Btrans_M1
#define zq_gemm_32f_align_AnoTrans_Btrans_M2 zq_gemm_32f_align256bit_AnoTrans_Btrans_M2
#define zq_gemm_32f_align_AnoTrans_Btrans_M4 zq_gemm_32f_align256bit_AnoTrans_Btrans_M4
#include "zq_gemm_32f_align_c_raw.h"

#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_set1_ps
#undef zq_mm_setzero_ps
#undef zq_mm_type
#undef zq_mm_align_size
#undef zq_mm_align_size2
#undef zq_mm_align_size3
#undef zq_mm_align_size4
#undef zq_mm_align_size5
#undef zq_mm_align_size6
#undef zq_mm_align_size7
#undef zq_mm_align_size8
#undef zq_final_sum_q
#undef zq_mm_fmadd_ps
#undef zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNgeneral
#undef zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNgeneral
#undef zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNgeneral
#undef zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign8
#undef zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign8
#undef zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign8
#undef zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign4
#undef zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign4
#undef zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign4
#undef zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_Kgeneral
#undef zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_Kgeneral
#undef zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_Kgeneral
#undef zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign8
#undef zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign8
#undef zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign4
#undef zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign4
#undef zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_Kgeneral
#undef zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_Kgeneral
#undef zq_gemm_32f_align_AnoTrans_Btrans_M1
#undef zq_gemm_32f_align_AnoTrans_Btrans_M2
#undef zq_gemm_32f_align_AnoTrans_Btrans_M4
#endif
	
	void zq_gemm_32f_align0_AnoTrans_Btrans(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
	{
		const float* Aptr, *A_c_ptr, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
		const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
		float* Cptr, *C_c_ptr;
		int m, n, k;
		float sum1, sum2, sum3, sum4, a_val;
		Aptr = A;
		Cptr = C;
		for (m = 0; m < M; m++, Aptr += lda, Cptr += ldc)
		{
			Bptr = Bt;
			C_c_ptr = Cptr;
			for (n = 0; n < N - 3; n += 4, Bptr += ldb)
			{
				sum1 = 0;
				sum2 = 0;
				sum3 = 0;
				sum4 = 0;
				Bptr1 = Bptr;
				Bptr2 = Bptr1 + ldb;
				Bptr3 = Bptr2 + ldb;
				Bptr4 = Bptr3 + ldb;
				for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < K - 4 + 1;
					k += 4)
				{
					a_val = *(A_c_ptr++);
					sum1 += a_val*(*(B_c_ptr1++));
					sum2 += a_val*(*(B_c_ptr2++));
					sum3 += a_val*(*(B_c_ptr3++));
					sum4 += a_val*(*(B_c_ptr4++));
					a_val = *(A_c_ptr++);
					sum1 += a_val*(*(B_c_ptr1++));
					sum2 += a_val*(*(B_c_ptr2++));
					sum3 += a_val*(*(B_c_ptr3++));
					sum4 += a_val*(*(B_c_ptr4++));
					a_val = *(A_c_ptr++);
					sum1 += a_val*(*(B_c_ptr1++));
					sum2 += a_val*(*(B_c_ptr2++));
					sum3 += a_val*(*(B_c_ptr3++));
					sum4 += a_val*(*(B_c_ptr4++));
					a_val = *(A_c_ptr++);
					sum1 += a_val*(*(B_c_ptr1++));
					sum2 += a_val*(*(B_c_ptr2++));
					sum3 += a_val*(*(B_c_ptr3++));
					sum4 += a_val*(*(B_c_ptr4++));
				}
				for (; k < K;k ++)
				{
					a_val = *(A_c_ptr++);
					sum1 += a_val*(*(B_c_ptr1++));
					sum2 += a_val*(*(B_c_ptr2++));
					sum3 += a_val*(*(B_c_ptr3++));
					sum4 += a_val*(*(B_c_ptr4++));
				}
				
				*(C_c_ptr++) = sum1;
				*(C_c_ptr++) = sum2;
				*(C_c_ptr++) = sum3;
				*(C_c_ptr++) = sum4;
			}
			for (; n < N; n++, Bptr += ldb)
			{
				sum1 = 0;
				Bptr1 = Bptr;
				for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1;
					k < K - 4 + 1;k += 4)
				{
					a_val = *(A_c_ptr++);
					sum1 += a_val*(*(B_c_ptr1++));
					a_val = *(A_c_ptr++);
					sum1 += a_val*(*(B_c_ptr1++));
					a_val = *(A_c_ptr++);
					sum1 += a_val*(*(B_c_ptr1++));
					a_val = *(A_c_ptr++);
					sum1 += a_val*(*(B_c_ptr1++));
				}
				for (; k < K;k ++)
				{
					a_val = *(A_c_ptr++);
					sum1 += a_val*(*(B_c_ptr1++));
				}
				*(C_c_ptr++) = sum1;
			}
		}
	}

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
