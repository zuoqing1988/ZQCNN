#ifndef _ZQ_SSE_MATHFUN_H_
#define _ZQ_SSE_MATHFUN_H_
#include "..\ZQ_CNN_CompileConfig.h"
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	/*C = A*B ,
	A: M * K,
	Bt: N * K,
	K % 8 == 0
	*/
	void zq_avx_gemm_align32f_AnoTrans_Btrans(const float* A, const float* Bt, float* C, int M, int N, int K);

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif
#endif