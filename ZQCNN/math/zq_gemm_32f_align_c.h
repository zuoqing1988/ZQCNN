#ifndef _ZQ_SSE_MATHFUN_H_
#define _ZQ_SSE_MATHFUN_H_
#include "../ZQ_CNN_CompileConfig.h"


#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	void zq_gemm_32f_AnoTrans_Btrans_auto(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

	void zq_gemm_32f_align0_AnoTrans_Btrans(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	/*C = A*B ,
	A: M * K,
	Bt: N * K,
	K % 4 == 0
	*/
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N1(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N2(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N4(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N8(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N1_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N2_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N4_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_N8_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv1_Kdiv16(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv1_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv1_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv1_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv1_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv2_Kdiv16(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv2_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv2_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv2_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv2_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv4_Kdiv16(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv4_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv4_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv4_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv4_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv8_Kdiv16(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv8_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv8_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv8_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M1_caseNdiv8_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N1(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N2(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N4(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N8(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N1_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N2_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N4_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N8_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv1_Kdiv16(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv1_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv1_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv1_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv1_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv2_Kdiv16(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv2_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv2_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv2_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv2_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv4_Kdiv16(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv4_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv4_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv4_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv4_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv8_Kdiv16(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv8_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv8_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv8_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_caseNdiv8_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N2(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N1_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N2_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_N4_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_caseNdiv1_Kdiv16(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_caseNdiv1_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_caseNdiv1_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_caseNdiv1_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_caseNdiv1_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_caseNdiv2_Kdiv16(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_caseNdiv2_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_caseNdiv2_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_caseNdiv2_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_caseNdiv2_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_caseNdiv4_Kdiv16(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_caseNdiv4_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_caseNdiv4_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_caseNdiv4_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M4_caseNdiv4_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_N1(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_N2(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_N1_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_N2_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_caseNdiv1_Kdiv16(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_caseNdiv1_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_caseNdiv1_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_caseNdiv1_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_caseNdiv1_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_caseNdiv2_Kdiv16(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_caseNdiv2_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_caseNdiv2_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_caseNdiv2_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M8_caseNdiv2_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M16_N1(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M16_N1_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M16_caseNdiv1_Kdiv16(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M16_caseNdiv1_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M16_caseNdiv1_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M16_caseNdiv1_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_M16_caseNdiv1_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	/*C = A*B ,
	A: M * K,
	Bt: N * K,
	K % 8 == 0
	*/
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N1(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N2(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N4(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N8(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N1_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N2_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N4_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N8_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv1_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv1_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv1_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv1_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv1_Kdiv512(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv2_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv2_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv2_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv2_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv2_Kdiv512(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv4_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv4_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv4_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv4_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv4_Kdiv512(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv8_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv8_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv8_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv8_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_caseNdiv8_Kdiv512(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N1(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N2(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N4(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N8(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N1_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N2_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N4_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N8_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv1_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv1_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv1_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv1_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv1_Kdiv512(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv2_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv2_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv2_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv2_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv2_Kdiv512(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv4_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv4_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv4_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv4_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv4_Kdiv512(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv8_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv8_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv8_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv8_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv8_Kdiv512(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N1(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N2(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N4(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N1_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N2_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N4_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv1_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv1_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv1_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv1_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv1_Kdiv512(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv2_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv2_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv2_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv2_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv2_Kdiv512(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv4_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv4_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv4_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv4_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_caseNdiv4_Kdiv512(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_N1(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_N2(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_N1_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_N2_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_caseNdiv1_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_caseNdiv1_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_caseNdiv1_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_caseNdiv1_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_caseNdiv1_Kdiv512(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_caseNdiv2_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_caseNdiv2_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_caseNdiv2_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_caseNdiv2_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_caseNdiv2_Kdiv512(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M16_N1(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M16_N1_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_caseNdiv1_Kdiv32(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_caseNdiv1_Kdiv64(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_caseNdiv1_Kdiv128(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_caseNdiv1_Kdiv256(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_caseNdiv1_Kdiv512(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	
#endif

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif