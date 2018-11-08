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

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

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
#define zq_final_sum_q1 (q1[0]+q1[1]+q1[2]+q1[3])
#define zq_final_sum_q2 (q2[0]+q2[1]+q2[2]+q2[3])
#define zq_final_sum_q3 (q3[0]+q3[1]+q3[2]+q3[3])
#define zq_final_sum_q4 (q4[0]+q4[1]+q4[2]+q4[3])
#define zq_final_sum_q5 (q5[0]+q5[1]+q5[2]+q5[3])
#define zq_final_sum_q6 (q6[0]+q6[1]+q6[2]+q6[3])
#define zq_final_sum_q7 (q7[0]+q7[1]+q7[2]+q7[3])
#define zq_final_sum_q8 (q8[0]+q8[1]+q8[2]+q8[3])
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps _mm_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm_add_ps(_mm_mul_ps(A, B), C)
#endif
#define zq_gemm_32f_align_AnoTrans_Btrans_caseNgeneral zq_gemm_32f_align128bit_AnoTrans_Btrans_caseNgeneral
#define zq_gemm_32f_align_AnoTrans_Btrans_caseNdiv4 zq_gemm_32f_align128bit_AnoTrans_Btrans_caseNdiv4
#define zq_gemm_32f_align_AnoTrans_Btrans_caseNdiv8 zq_gemm_32f_align128bit_AnoTrans_Btrans_caseNdiv8
#define zq_gemm_32f_align_AnoTrans_Btrans zq_gemm_32f_align128bit_AnoTrans_Btrans
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
#undef zq_final_sum_q1
#undef zq_final_sum_q2
#undef zq_final_sum_q3
#undef zq_final_sum_q4
#undef zq_final_sum_q5
#undef zq_final_sum_q6
#undef zq_final_sum_q7
#undef zq_final_sum_q8
#undef zq_mm_fmadd_ps
#undef zq_gemm_32f_align_AnoTrans_Btrans_caseNgeneral
#undef zq_gemm_32f_align_AnoTrans_Btrans_caseNdiv4
#undef zq_gemm_32f_align_AnoTrans_Btrans_caseNdiv8
#undef zq_gemm_32f_align_AnoTrans_Btrans
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
#define zq_gemm_32f_align_AnoTrans_Btrans_caseNgeneral zq_gemm_32f_align256bit_AnoTrans_Btrans_caseNgeneral 
#define zq_gemm_32f_align_AnoTrans_Btrans_caseNdiv4 zq_gemm_32f_align256bit_AnoTrans_Btrans_caseNdiv4
#define zq_gemm_32f_align_AnoTrans_Btrans_caseNdiv8 zq_gemm_32f_align256bit_AnoTrans_Btrans_caseNdiv8
#define zq_gemm_32f_align_AnoTrans_Btrans zq_gemm_32f_align256bit_AnoTrans_Btrans
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
#undef zq_final_sum_q1
#undef zq_final_sum_q2
#undef zq_final_sum_q3
#undef zq_final_sum_q4
#undef zq_final_sum_q5
#undef zq_final_sum_q6
#undef zq_final_sum_q7
#undef zq_final_sum_q8
#undef zq_mm_fmadd_ps
#undef zq_gemm_32f_align_AnoTrans_Btrans_caseNgeneral
#undef zq_gemm_32f_align_AnoTrans_Btrans_caseNdiv4
#undef zq_gemm_32f_align_AnoTrans_Btrans_caseNdiv8
#undef zq_gemm_32f_align_AnoTrans_Btrans

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
