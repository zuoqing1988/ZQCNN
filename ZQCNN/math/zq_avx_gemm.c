#include "..\ZQ_CNN_CompileConfig.h"
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#include <immintrin.h>
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

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
#define zq_final_sum_q1 (q1[0]+q1[1]+q1[2]+q1[3]+q1[4]+q1[5]+q1[6]+q1[7])
#define zq_final_sum_q2 (q2[0]+q2[1]+q2[2]+q2[3]+q2[4]+q2[5]+q2[6]+q2[7])
#define zq_final_sum_q3 (q3[0]+q3[1]+q3[2]+q3[3]+q3[4]+q3[5]+q3[6]+q3[7])
#define zq_final_sum_q4 (q4[0]+q4[1]+q4[2]+q4[3]+q4[4]+q4[5]+q4[6]+q4[7])
#define zq_final_sum_q5 (q5[0]+q5[1]+q5[2]+q5[3]+q5[4]+q5[5]+q5[6]+q5[7])
#define zq_final_sum_q6 (q6[0]+q6[1]+q6[2]+q6[3]+q6[4]+q6[5]+q6[6]+q6[7])
#define zq_final_sum_q7 (q7[0]+q7[1]+q7[2]+q7[3]+q7[4]+q7[5]+q7[6]+q7[7])
#define zq_final_sum_q8 (q8[0]+q8[1]+q8[2]+q8[3]+q8[4]+q8[5]+q8[6]+q8[7])
#if ZQ_CNN_USE_FMADD256
#define zq_mm_fmadd_ps _mm256_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#endif

#include "zq_gemm_align32f_raw.h"

	/*C = A*B ,
	A: M * K,
	Bt: N * K,
	K % 4 == 0
	*/
	void zq_avx_gemm_align32f_AnoTrans_Btrans(const float* A, const float* Bt, float* C, int M, int N, int K)
	{
		if (N % 8 == 0)
		{
			zq_gemm_align32f_AnoTrans_Btrans_caseNdiv8(A, Bt, C, M, N, K);
			return;
		}
		else if (N % 4 == 0)
		{
			zq_gemm_align32f_AnoTrans_Btrans_caseNdiv4(A, Bt, C, M, N, K);
			return;
		}
		else
		{
			zq_gemm_align32f_AnoTrans_Btrans_caseNgeneral(A, Bt, C, M, N, K);
			return;
		}
	}


#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif

#endif