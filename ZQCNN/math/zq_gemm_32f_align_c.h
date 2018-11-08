#ifndef _ZQ_SSE_MATHFUN_H_
#define _ZQ_SSE_MATHFUN_H_
#include "..\ZQ_CNN_CompileConfig.h"


#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	void zq_gemm_32f_align0_AnoTrans_Btrans(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	/*C = A*B ,
	A: M * K,
	Bt: N * K,
	K % 4 == 0
	*/
	void zq_gemm_32f_align128bit_AnoTrans_Btrans(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

	void zq_gemm_32f_align128bit_AnoTrans_Btrans_caseNgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_caseNdiv4(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align128bit_AnoTrans_Btrans_caseNdiv8(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	/*C = A*B ,
	A: M * K,
	Bt: N * K,
	K % 8 == 0
	*/
	void zq_gemm_32f_align256bit_AnoTrans_Btrans(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

	void zq_gemm_32f_align256bit_AnoTrans_Btrans_caseNgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_caseNdiv4(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
	void zq_gemm_32f_align256bit_AnoTrans_Btrans_caseNdiv8(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);
#endif

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif