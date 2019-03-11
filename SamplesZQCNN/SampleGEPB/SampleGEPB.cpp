#include "ZQ_CNN_CompileConfig.h"
#if __ARM_NEON
#include <arm_neon.h>
#else

#if defined(__GNUC__)
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#include <smmintrin.h>
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#include <x86intrin.h>
#endif
#elif defined(_WIN32)
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
#endif

#endif // __ARM_NEON

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#if __ARM_NEON
#define zq_mm_load_ps vld1q_f32
#define zq_mm_broadcast_ss vld1q_dup_f32
#define zq_mm_store_ps vst1q_f32
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps(A, B, C) vfmaq_f32(C, A, B)
#else
#define zq_mm_fmadd_ps(A, B, C) vaddq_f32(vmulq_f32(A, B), C)
#endif
#define zq_mm_type float32x4_t
#define zq_mm_align_size 4
#define zq_mm_align_size2 8
#define zq_mm_align_size3 12
#define zq_mm_align_size4 16
#define zq_mm_align_size5 20
#define zq_mm_align_size6 24
#define zq_mm_align_size7 28
#define zq_mm_align_size8 32
#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_broadcast_ss _mm256_broadcast_ss
#define zq_mm_store_ps _mm256_store_ps
#if ZQ_CNN_USE_FMADD256
#define zq_mm_fmadd_ps _mm256_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#endif
#define zq_mm_type __m256
#define zq_mm_align_size 8
#define zq_mm_align_size2 16
#define zq_mm_align_size3 24
#define zq_mm_align_size4 32
#define zq_mm_align_size5 40
#define zq_mm_align_size6 48
#define zq_mm_align_size7 56
#define zq_mm_align_size8 64
#elif ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#define zq_mm_load_ps _mm_load_ps
#define zq_mm_broadcast_ss _mm_broadcast_ss
#define zq_mm_store_ps _mm_store_ps
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps _mm_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm_add_ps(_mm_mul_ps(A, B), C)
#endif
#define zq_mm_type __m128
#define zq_mm_align_size 4
#define zq_mm_align_size2 8
#define zq_mm_align_size3 12
#define zq_mm_align_size4 16
#define zq_mm_align_size5 20
#define zq_mm_align_size6 24
#define zq_mm_align_size7 28
#define zq_mm_align_size8 32
#endif
#endif

int align_ceil(int num, int align)
{
	return num + (align - (num % align)) % align;
}

template<const int M>
void inner_kernel_MxALIGN2_template(int K, const float *packA, const float *packB, float *c, int ldc)
{
	const float *aptr = packA;
	const float *bptr = packB;
	float *cptr = c;
	register zq_mm_type va;
	register zq_mm_type vb0, vb1;
	register zq_mm_type vc00, vc10, vc20, vc30, vc40, vc50, vc60, vc70;
	register zq_mm_type vc01, vc11, vc21, vc31, vc41, vc51, vc61, vc71;

	vc00 = zq_mm_load_ps(cptr);
	vc01 = zq_mm_load_ps(cptr + zq_mm_align_size);
	cptr += ldc;
	if (M > 1)
	{
		vc10 = zq_mm_load_ps(cptr);
		vc11 = zq_mm_load_ps(cptr + zq_mm_align_size);
		cptr += ldc;
	}
	if (M > 2)
	{
		vc20 = zq_mm_load_ps(cptr);
		vc21 = zq_mm_load_ps(cptr + zq_mm_align_size);
		cptr += ldc;
	}
	if (M > 3)
	{
		vc30 = zq_mm_load_ps(cptr);
		vc31 = zq_mm_load_ps(cptr + zq_mm_align_size);
		cptr += ldc;
	}
	if (M > 4)
	{
		vc40 = zq_mm_load_ps(cptr);
		vc41 = zq_mm_load_ps(cptr + zq_mm_align_size);
		cptr += ldc;
	}
	if (M > 5)
	{
		vc50 = zq_mm_load_ps(cptr);
		vc51 = zq_mm_load_ps(cptr + zq_mm_align_size);
		cptr += ldc;
	}
	if (M > 6)
	{
		vc60 = zq_mm_load_ps(cptr);
		vc61 = zq_mm_load_ps(cptr + zq_mm_align_size);
		cptr += ldc;
	}
	if (M > 7)
	{
		vc70 = zq_mm_load_ps(cptr);
		vc71 = zq_mm_load_ps(cptr + zq_mm_align_size);
	}
	vb0 = zq_mm_load_ps(bptr);
	vb1 = zq_mm_load_ps(bptr + zq_mm_align_size);
	for (int p = 0; p < (K - 1); ++p)
	{
		va = zq_mm_broadcast_ss(aptr);
		vc00 = zq_mm_fmadd_ps(vb0, va, vc00);
		vc01 = zq_mm_fmadd_ps(vb1, va, vc01);

		if (M > 1)
		{
			va = zq_mm_broadcast_ss(aptr + 1);
			vc10 = zq_mm_fmadd_ps(vb0, va, vc10);
			vc11 = zq_mm_fmadd_ps(vb1, va, vc11);
		}

		if (M > 2)
		{
			va = zq_mm_broadcast_ss(aptr + 2);
			vc20 = zq_mm_fmadd_ps(vb0, va, vc20);
			vc21 = zq_mm_fmadd_ps(vb1, va, vc21);
		}

		if (M > 3)
		{
			va = zq_mm_broadcast_ss(aptr + 3);
			vc30 = zq_mm_fmadd_ps(vb0, va, vc30);
			vc31 = zq_mm_fmadd_ps(vb1, va, vc31);
		}

		if (M > 4)
		{
			va = zq_mm_broadcast_ss(aptr + 4);
			vc40 = zq_mm_fmadd_ps(vb0, va, vc40);
			vc41 = zq_mm_fmadd_ps(vb1, va, vc41);
		}

		if (M > 5)
		{
			va = zq_mm_broadcast_ss(aptr + 5);
			vc50 = zq_mm_fmadd_ps(vb0, va, vc50);
			vc51 = zq_mm_fmadd_ps(vb1, va, vc51);
		}

		if (M > 6)
		{
			va = zq_mm_broadcast_ss(aptr + 6);
			vc60 = zq_mm_fmadd_ps(vb0, va, vc60);
			vc61 = zq_mm_fmadd_ps(vb1, va, vc61);
		}

		if (M > 7)
		{
			va = zq_mm_broadcast_ss(aptr + 7);
			vc70 = zq_mm_fmadd_ps(vb0, va, vc70);
			vc71 = zq_mm_fmadd_ps(vb1, va, vc71);
		}

		aptr += M;
		bptr += zq_mm_align_size2;
		vb0 = zq_mm_load_ps(bptr);
		vb1 = zq_mm_load_ps(bptr + zq_mm_align_size);
	}
	cptr = c;
	va = zq_mm_broadcast_ss(aptr);
	vc00 = zq_mm_fmadd_ps(vb0, va, vc00);
	vc01 = zq_mm_fmadd_ps(vb1, va, vc01);
	zq_mm_store_ps(cptr, vc00);
	zq_mm_store_ps(cptr + zq_mm_align_size, vc01);
	if (M > 1)
	{
		va = zq_mm_broadcast_ss(aptr + 1);
		vc10 = zq_mm_fmadd_ps(vb0, va, vc10);
		vc11 = zq_mm_fmadd_ps(vb1, va, vc11);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc10);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc11);
	}
	if (M > 2)
	{
		va = zq_mm_broadcast_ss(aptr + 2);
		vc20 = zq_mm_fmadd_ps(vb0, va, vc20);
		vc21 = zq_mm_fmadd_ps(vb1, va, vc21);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc20);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc21);
	}
	if (M > 3)
	{
		va = zq_mm_broadcast_ss(aptr + 3);
		vc30 = zq_mm_fmadd_ps(vb0, va, vc30);
		vc31 = zq_mm_fmadd_ps(vb1, va, vc31);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc30);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc31);
	}
	if (M > 4)
	{
		va = zq_mm_broadcast_ss(aptr + 4);
		vc40 = zq_mm_fmadd_ps(vb0, va, vc40);
		vc41 = zq_mm_fmadd_ps(vb1, va, vc41);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc40);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc41);
	}
	if (M > 5)
	{
		va = zq_mm_broadcast_ss(aptr + 5);
		vc50 = zq_mm_fmadd_ps(vb0, va, vc50);
		vc51 = zq_mm_fmadd_ps(vb1, va, vc51);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc50);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc51);
	}
	if (M > 6)
	{
		va = zq_mm_broadcast_ss(aptr + 6);
		vc60 = zq_mm_fmadd_ps(vb0, va, vc60);
		vc61 = zq_mm_fmadd_ps(vb1, va, vc61);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc60);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc61);
	}
	if (M > 7)
	{
		va = zq_mm_broadcast_ss(aptr + 7);
		vc70 = zq_mm_fmadd_ps(vb0, va, vc70);
		vc71 = zq_mm_fmadd_ps(vb1, va, vc71);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc70);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc71);
	}
}

template<const int M>
void inner_kernel_MxALIGN3_template(int K, const float *packA, const float *packB, float *c, int ldc)
{
	const float *aptr = packA;
	const float *bptr = packB;
	float *cptr = c;
	register zq_mm_type va;
	register zq_mm_type vb0, vb1, vb2;
	register zq_mm_type vc00, vc10, vc20, vc30, vc40, vc50, vc60, vc70;
	register zq_mm_type vc01, vc11, vc21, vc31, vc41, vc51, vc61, vc71;
	register zq_mm_type vc02, vc12, vc22, vc32, vc42, vc52, vc62, vc72;

	vc00 = zq_mm_load_ps(cptr);
	vc01 = zq_mm_load_ps(cptr + zq_mm_align_size);
	vc02 = zq_mm_load_ps(cptr + zq_mm_align_size2);
	cptr += ldc;
	if (M > 1)
	{
		vc10 = zq_mm_load_ps(cptr);
		vc11 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc12 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		cptr += ldc;
	}
	if (M > 2)
	{
		vc20 = zq_mm_load_ps(cptr);
		vc21 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc22 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		cptr += ldc;
	}
	if (M > 3)
	{
		vc30 = zq_mm_load_ps(cptr);
		vc31 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc32 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		cptr += ldc;
	}
	if (M > 4)
	{
		vc40 = zq_mm_load_ps(cptr);
		vc41 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc42 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		cptr += ldc;
	}
	if (M > 5)
	{
		vc50 = zq_mm_load_ps(cptr);
		vc51 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc52 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		cptr += ldc;
	}
	if (M > 6)
	{
		vc60 = zq_mm_load_ps(cptr);
		vc61 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc62 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		cptr += ldc;
	}
	if (M > 7)
	{
		vc70 = zq_mm_load_ps(cptr);
		vc71 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc72 = zq_mm_load_ps(cptr + zq_mm_align_size2);
	}
	vb0 = zq_mm_load_ps(bptr);
	vb1 = zq_mm_load_ps(bptr + zq_mm_align_size);
	vb2 = zq_mm_load_ps(bptr + zq_mm_align_size2);
	for (int p = 0; p < (K - 1); ++p)
	{
		va = zq_mm_broadcast_ss(aptr);
		vc00 = zq_mm_fmadd_ps(vb0, va, vc00);
		vc01 = zq_mm_fmadd_ps(vb1, va, vc01);
		vc02 = zq_mm_fmadd_ps(vb2, va, vc02);

		if (M > 1)
		{
			va = zq_mm_broadcast_ss(aptr + 1);
			vc10 = zq_mm_fmadd_ps(vb0, va, vc10);
			vc11 = zq_mm_fmadd_ps(vb1, va, vc11);
			vc12 = zq_mm_fmadd_ps(vb2, va, vc12);
		}

		if (M > 2)
		{
			va = zq_mm_broadcast_ss(aptr + 2);
			vc20 = zq_mm_fmadd_ps(vb0, va, vc20);
			vc21 = zq_mm_fmadd_ps(vb1, va, vc21);
			vc22 = zq_mm_fmadd_ps(vb2, va, vc22);
		}

		if (M > 3)
		{
			va = zq_mm_broadcast_ss(aptr + 3);
			vc30 = zq_mm_fmadd_ps(vb0, va, vc30);
			vc31 = zq_mm_fmadd_ps(vb1, va, vc31);
			vc32 = zq_mm_fmadd_ps(vb2, va, vc32);
		}

		if (M > 4)
		{
			va = zq_mm_broadcast_ss(aptr + 4);
			vc40 = zq_mm_fmadd_ps(vb0, va, vc40);
			vc41 = zq_mm_fmadd_ps(vb1, va, vc41);
			vc42 = zq_mm_fmadd_ps(vb2, va, vc42);
		}

		if (M > 5)
		{
			va = zq_mm_broadcast_ss(aptr + 5);
			vc50 = zq_mm_fmadd_ps(vb0, va, vc50);
			vc51 = zq_mm_fmadd_ps(vb1, va, vc51);
			vc52 = zq_mm_fmadd_ps(vb2, va, vc52);
		}

		if (M > 6)
		{
			va = zq_mm_broadcast_ss(aptr + 6);
			vc60 = zq_mm_fmadd_ps(vb0, va, vc60);
			vc61 = zq_mm_fmadd_ps(vb1, va, vc61);
			vc62 = zq_mm_fmadd_ps(vb2, va, vc62);
		}

		if (M > 7)
		{
			va = zq_mm_broadcast_ss(aptr + 7);
			vc70 = zq_mm_fmadd_ps(vb0, va, vc70);
			vc71 = zq_mm_fmadd_ps(vb1, va, vc71);
			vc72 = zq_mm_fmadd_ps(vb2, va, vc72);
		}
		bptr += zq_mm_align_size3;
		aptr += M;
		vb0 = zq_mm_load_ps(bptr);
		vb1 = zq_mm_load_ps(bptr + zq_mm_align_size);
		vb2 = zq_mm_load_ps(bptr + zq_mm_align_size2);

	}
	cptr = c;
	va = zq_mm_broadcast_ss(aptr);
	vc00 = zq_mm_fmadd_ps(vb0, va, vc00);
	vc01 = zq_mm_fmadd_ps(vb1, va, vc01);
	vc02 = zq_mm_fmadd_ps(vb2, va, vc02);
	zq_mm_store_ps(cptr, vc00);
	zq_mm_store_ps(cptr + zq_mm_align_size, vc01);
	zq_mm_store_ps(cptr + zq_mm_align_size2, vc02);
	if (M > 1)
	{
		va = zq_mm_broadcast_ss(aptr + 1);
		vc10 = zq_mm_fmadd_ps(vb0, va, vc10);
		vc11 = zq_mm_fmadd_ps(vb1, va, vc11);
		vc12 = zq_mm_fmadd_ps(vb2, va, vc12);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc10);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc11);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc12);
	}
	if (M > 2)
	{
		va = zq_mm_broadcast_ss(aptr + 2);
		vc20 = zq_mm_fmadd_ps(vb0, va, vc20);
		vc21 = zq_mm_fmadd_ps(vb1, va, vc21);
		vc22 = zq_mm_fmadd_ps(vb2, va, vc22);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc20);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc21);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc22);
	}
	if (M > 3)
	{
		va = zq_mm_broadcast_ss(aptr + 3);
		vc30 = zq_mm_fmadd_ps(vb0, va, vc30);
		vc31 = zq_mm_fmadd_ps(vb1, va, vc31);
		vc32 = zq_mm_fmadd_ps(vb2, va, vc32);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc30);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc31);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc32);
	}
	if (M > 4)
	{
		va = zq_mm_broadcast_ss(aptr + 4);
		vc40 = zq_mm_fmadd_ps(vb0, va, vc40);
		vc41 = zq_mm_fmadd_ps(vb1, va, vc41);
		vc42 = zq_mm_fmadd_ps(vb2, va, vc42);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc40);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc41);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc42);
	}
	if (M > 5)
	{
		va = zq_mm_broadcast_ss(aptr + 5);
		vc50 = zq_mm_fmadd_ps(vb0, va, vc50);
		vc51 = zq_mm_fmadd_ps(vb1, va, vc51);
		vc52 = zq_mm_fmadd_ps(vb2, va, vc52);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc50);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc51);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc52);
	}
	if (M > 6)
	{
		va = zq_mm_broadcast_ss(aptr + 6);
		vc60 = zq_mm_fmadd_ps(vb0, va, vc60);
		vc61 = zq_mm_fmadd_ps(vb1, va, vc61);
		vc62 = zq_mm_fmadd_ps(vb2, va, vc62);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc60);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc61);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc62);
	}
	if (M > 7)
	{
		va = zq_mm_broadcast_ss(aptr + 7);
		vc70 = zq_mm_fmadd_ps(vb0, va, vc70);
		vc71 = zq_mm_fmadd_ps(vb1, va, vc71);
		vc72 = zq_mm_fmadd_ps(vb2, va, vc72);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc70);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc71);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc72);
	}
}

template<const int M>
void inner_kernel_MxALIGN4_template(int K, const float *packA, const float *packB, float *c, int ldc)
{
	const float *aptr = packA;
	const float *bptr = packB;
	float *cptr = c;
	register zq_mm_type va;
	register zq_mm_type vb0, vb1, vb2, vb3;
	register zq_mm_type vc00, vc10, vc20, vc30, vc40, vc50, vc60, vc70;
	register zq_mm_type vc01, vc11, vc21, vc31, vc41, vc51, vc61, vc71;
	register zq_mm_type vc02, vc12, vc22, vc32, vc42, vc52, vc62, vc72;
	register zq_mm_type vc03, vc13, vc23, vc33, vc43, vc53, vc63, vc73;

	vc00 = zq_mm_load_ps(cptr);
	vc01 = zq_mm_load_ps(cptr + zq_mm_align_size);
	vc02 = zq_mm_load_ps(cptr + zq_mm_align_size2);
	vc03 = zq_mm_load_ps(cptr + zq_mm_align_size3);
	cptr += ldc;
	if (M > 1)
	{
		vc10 = zq_mm_load_ps(cptr);
		vc11 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc12 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		vc13 = zq_mm_load_ps(cptr + zq_mm_align_size3);
		cptr += ldc;
	}
	if (M > 2)
	{
		vc20 = zq_mm_load_ps(cptr);
		vc21 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc22 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		vc23 = zq_mm_load_ps(cptr + zq_mm_align_size3);
		cptr += ldc;
	}
	if (M > 3)
	{
		vc30 = zq_mm_load_ps(cptr);
		vc31 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc32 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		vc33 = zq_mm_load_ps(cptr + zq_mm_align_size3);
		cptr += ldc;
	}
	if (M > 4)
	{
		vc40 = zq_mm_load_ps(cptr);
		vc41 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc42 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		vc43 = zq_mm_load_ps(cptr + zq_mm_align_size3);
		cptr += ldc;
	}
	if (M > 5)
	{
		vc50 = zq_mm_load_ps(cptr);
		vc51 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc52 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		vc53 = zq_mm_load_ps(cptr + zq_mm_align_size3);
		cptr += ldc;
	}
	if (M > 6)
	{
		vc60 = zq_mm_load_ps(cptr);
		vc61 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc62 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		vc63 = zq_mm_load_ps(cptr + zq_mm_align_size3);
		cptr += ldc;
	}
	if (M > 7)
	{
		vc70 = zq_mm_load_ps(cptr);
		vc71 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc72 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		vc73 = zq_mm_load_ps(cptr + zq_mm_align_size3);
	}
	vb0 = zq_mm_load_ps(bptr);
	vb1 = zq_mm_load_ps(bptr + zq_mm_align_size);
	vb2 = zq_mm_load_ps(bptr + zq_mm_align_size2);
	vb3 = zq_mm_load_ps(bptr + zq_mm_align_size3);
	for (int p = 0; p < (K - 1); ++p)
	{
		va = zq_mm_broadcast_ss(aptr);
		vc00 = zq_mm_fmadd_ps(vb0, va, vc00);
		vc01 = zq_mm_fmadd_ps(vb1, va, vc01);
		vc02 = zq_mm_fmadd_ps(vb2, va, vc02);
		vc03 = zq_mm_fmadd_ps(vb3, va, vc03);

		if (M > 1)
		{
			va = zq_mm_broadcast_ss(aptr + 1);
			vc10 = zq_mm_fmadd_ps(vb0, va, vc10);
			vc11 = zq_mm_fmadd_ps(vb1, va, vc11);
			vc12 = zq_mm_fmadd_ps(vb2, va, vc12);
			vc13 = zq_mm_fmadd_ps(vb3, va, vc13);
		}

		if (M > 2)
		{
			va = zq_mm_broadcast_ss(aptr + 2);
			vc20 = zq_mm_fmadd_ps(vb0, va, vc20);
			vc21 = zq_mm_fmadd_ps(vb1, va, vc21);
			vc22 = zq_mm_fmadd_ps(vb2, va, vc22);
			vc23 = zq_mm_fmadd_ps(vb3, va, vc23);
		}

		if (M > 3)
		{
			va = zq_mm_broadcast_ss(aptr + 3);
			vc30 = zq_mm_fmadd_ps(vb0, va, vc30);
			vc31 = zq_mm_fmadd_ps(vb1, va, vc31);
			vc32 = zq_mm_fmadd_ps(vb2, va, vc32);
			vc33 = zq_mm_fmadd_ps(vb3, va, vc33);
		}

		if (M > 4)
		{
			va = zq_mm_broadcast_ss(aptr + 4);
			vc40 = zq_mm_fmadd_ps(vb0, va, vc40);
			vc41 = zq_mm_fmadd_ps(vb1, va, vc41);
			vc42 = zq_mm_fmadd_ps(vb2, va, vc42);
			vc43 = zq_mm_fmadd_ps(vb3, va, vc43);
		}

		if (M > 5)
		{
			va = zq_mm_broadcast_ss(aptr + 5);
			vc50 = zq_mm_fmadd_ps(vb0, va, vc50);
			vc51 = zq_mm_fmadd_ps(vb1, va, vc51);
			vc52 = zq_mm_fmadd_ps(vb2, va, vc52);
			vc53 = zq_mm_fmadd_ps(vb3, va, vc53);
		}

		if (M > 6)
		{
			va = zq_mm_broadcast_ss(aptr + 6);
			vc60 = zq_mm_fmadd_ps(vb0, va, vc60);
			vc61 = zq_mm_fmadd_ps(vb1, va, vc61);
			vc62 = zq_mm_fmadd_ps(vb2, va, vc62);
			vc63 = zq_mm_fmadd_ps(vb3, va, vc63);
		}

		if (M > 7)
		{
			va = zq_mm_broadcast_ss(aptr + 7);
			vc70 = zq_mm_fmadd_ps(vb0, va, vc70);
			vc71 = zq_mm_fmadd_ps(vb1, va, vc71);
			vc72 = zq_mm_fmadd_ps(vb2, va, vc72);
			vc73 = zq_mm_fmadd_ps(vb3, va, vc73);
		}

		bptr += zq_mm_align_size4;
		aptr += M;
		vb0 = zq_mm_load_ps(bptr);
		vb1 = zq_mm_load_ps(bptr + zq_mm_align_size);
		vb2 = zq_mm_load_ps(bptr + zq_mm_align_size2);
		vb3 = zq_mm_load_ps(bptr + zq_mm_align_size3);

	}
	cptr = c;
	va = zq_mm_broadcast_ss(aptr);
	vc00 = zq_mm_fmadd_ps(vb0, va, vc00);
	vc01 = zq_mm_fmadd_ps(vb1, va, vc01);
	vc02 = zq_mm_fmadd_ps(vb2, va, vc02);
	vc03 = zq_mm_fmadd_ps(vb3, va, vc03);
	zq_mm_store_ps(cptr, vc00);
	zq_mm_store_ps(cptr + zq_mm_align_size, vc01);
	zq_mm_store_ps(cptr + zq_mm_align_size2, vc02);
	zq_mm_store_ps(cptr + zq_mm_align_size3, vc03);
	if (M > 1)
	{
		va = zq_mm_broadcast_ss(aptr + 1);
		vc10 = zq_mm_fmadd_ps(vb0, va, vc10);
		vc11 = zq_mm_fmadd_ps(vb1, va, vc11);
		vc12 = zq_mm_fmadd_ps(vb2, va, vc12);
		vc13 = zq_mm_fmadd_ps(vb3, va, vc13);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc10);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc11);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc12);
		zq_mm_store_ps(cptr + zq_mm_align_size3, vc13);
	}
	if (M > 2)
	{
		va = zq_mm_broadcast_ss(aptr + 2);
		vc20 = zq_mm_fmadd_ps(vb0, va, vc20);
		vc21 = zq_mm_fmadd_ps(vb1, va, vc21);
		vc22 = zq_mm_fmadd_ps(vb2, va, vc22);
		vc23 = zq_mm_fmadd_ps(vb3, va, vc23);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc20);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc21);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc22);
		zq_mm_store_ps(cptr + zq_mm_align_size3, vc23);
	}
	if (M > 3)
	{
		va = zq_mm_broadcast_ss(aptr + 3);
		vc30 = zq_mm_fmadd_ps(vb0, va, vc30);
		vc31 = zq_mm_fmadd_ps(vb1, va, vc31);
		vc32 = zq_mm_fmadd_ps(vb2, va, vc32);
		vc33 = zq_mm_fmadd_ps(vb3, va, vc33);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc30);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc31);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc32);
		zq_mm_store_ps(cptr + zq_mm_align_size3, vc33);
	}
	if (M > 4)
	{
		va = zq_mm_broadcast_ss(aptr + 4);
		vc40 = zq_mm_fmadd_ps(vb0, va, vc40);
		vc41 = zq_mm_fmadd_ps(vb1, va, vc41);
		vc42 = zq_mm_fmadd_ps(vb2, va, vc42);
		vc43 = zq_mm_fmadd_ps(vb3, va, vc43);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc40);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc41);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc42);
		zq_mm_store_ps(cptr + zq_mm_align_size3, vc43);
	}
	if (M > 5)
	{
		va = zq_mm_broadcast_ss(aptr + 5);
		vc50 = zq_mm_fmadd_ps(vb0, va, vc50);
		vc51 = zq_mm_fmadd_ps(vb1, va, vc51);
		vc52 = zq_mm_fmadd_ps(vb2, va, vc52);
		vc53 = zq_mm_fmadd_ps(vb3, va, vc53);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc50);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc51);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc52);
		zq_mm_store_ps(cptr + zq_mm_align_size3, vc53);
	}
	if (M > 6)
	{
		va = zq_mm_broadcast_ss(aptr + 6);
		vc60 = zq_mm_fmadd_ps(vb0, va, vc60);
		vc61 = zq_mm_fmadd_ps(vb1, va, vc61);
		vc62 = zq_mm_fmadd_ps(vb2, va, vc62);
		vc63 = zq_mm_fmadd_ps(vb3, va, vc63);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc60);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc61);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc62);
		zq_mm_store_ps(cptr + zq_mm_align_size3, vc63);
	}
	if (M > 7)
	{
		va = zq_mm_broadcast_ss(aptr + 7);
		vc70 = zq_mm_fmadd_ps(vb0, va, vc70);
		vc71 = zq_mm_fmadd_ps(vb1, va, vc71);
		vc72 = zq_mm_fmadd_ps(vb2, va, vc72);
		vc73 = zq_mm_fmadd_ps(vb3, va, vc73);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc70);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc71);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc72);
		zq_mm_store_ps(cptr + zq_mm_align_size3, vc73);
	}
}

template<const int M>
void inner_kernel_MxALIGN8_template(int K, const float *packA, const float *packB, float *c, int ldc)
{
	const float *aptr = packA;
	const float *bptr = packB;
	float *cptr = c;
	register zq_mm_type va;
	register zq_mm_type vb0, vb1, vb2, vb3, vb4, vb5, vb6, vb7;
	register zq_mm_type vc00, vc10, vc20, vc30, vc40, vc50, vc60, vc70;
	register zq_mm_type vc01, vc11, vc21, vc31, vc41, vc51, vc61, vc71;
	register zq_mm_type vc02, vc12, vc22, vc32, vc42, vc52, vc62, vc72;
	register zq_mm_type vc03, vc13, vc23, vc33, vc43, vc53, vc63, vc73;
	register zq_mm_type vc04, vc14, vc24, vc34, vc44, vc54, vc64, vc74;
	register zq_mm_type vc05, vc15, vc25, vc35, vc45, vc55, vc65, vc75;
	register zq_mm_type vc06, vc16, vc26, vc36, vc46, vc56, vc66, vc76;
	register zq_mm_type vc07, vc17, vc27, vc37, vc47, vc57, vc67, vc77;

	vc00 = zq_mm_load_ps(cptr);
	vc01 = zq_mm_load_ps(cptr + zq_mm_align_size);
	vc02 = zq_mm_load_ps(cptr + zq_mm_align_size2);
	vc03 = zq_mm_load_ps(cptr + zq_mm_align_size3);
	vc04 = zq_mm_load_ps(cptr + zq_mm_align_size4);
	vc05 = zq_mm_load_ps(cptr + zq_mm_align_size5);
	vc06 = zq_mm_load_ps(cptr + zq_mm_align_size6);
	vc07 = zq_mm_load_ps(cptr + zq_mm_align_size7);
	cptr += ldc;
	if (M > 1)
	{
		vc10 = zq_mm_load_ps(cptr);
		vc11 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc12 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		vc13 = zq_mm_load_ps(cptr + zq_mm_align_size3);
		vc14 = zq_mm_load_ps(cptr + zq_mm_align_size4);
		vc15 = zq_mm_load_ps(cptr + zq_mm_align_size5);
		vc16 = zq_mm_load_ps(cptr + zq_mm_align_size6);
		vc17 = zq_mm_load_ps(cptr + zq_mm_align_size7);
		cptr += ldc;
	}
	if (M > 2)
	{
		vc20 = zq_mm_load_ps(cptr);
		vc21 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc22 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		vc23 = zq_mm_load_ps(cptr + zq_mm_align_size3);
		vc24 = zq_mm_load_ps(cptr + zq_mm_align_size4);
		vc25 = zq_mm_load_ps(cptr + zq_mm_align_size5);
		vc26 = zq_mm_load_ps(cptr + zq_mm_align_size6);
		vc27 = zq_mm_load_ps(cptr + zq_mm_align_size7);
		cptr += ldc;
	}
	if (M > 3)
	{
		vc30 = zq_mm_load_ps(cptr);
		vc31 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc32 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		vc33 = zq_mm_load_ps(cptr + zq_mm_align_size3);
		vc34 = zq_mm_load_ps(cptr + zq_mm_align_size4);
		vc35 = zq_mm_load_ps(cptr + zq_mm_align_size5);
		vc36 = zq_mm_load_ps(cptr + zq_mm_align_size6);
		vc37 = zq_mm_load_ps(cptr + zq_mm_align_size7);
		cptr += ldc;
	}
	if (M > 4)
	{
		vc40 = zq_mm_load_ps(cptr);
		vc41 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc42 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		vc43 = zq_mm_load_ps(cptr + zq_mm_align_size3);
		vc44 = zq_mm_load_ps(cptr + zq_mm_align_size4);
		vc45 = zq_mm_load_ps(cptr + zq_mm_align_size5);
		vc46 = zq_mm_load_ps(cptr + zq_mm_align_size6);
		vc47 = zq_mm_load_ps(cptr + zq_mm_align_size7);
		cptr += ldc;
	}
	if (M > 5)
	{
		vc50 = zq_mm_load_ps(cptr);
		vc51 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc52 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		vc53 = zq_mm_load_ps(cptr + zq_mm_align_size3);
		vc54 = zq_mm_load_ps(cptr + zq_mm_align_size4);
		vc55 = zq_mm_load_ps(cptr + zq_mm_align_size5);
		vc56 = zq_mm_load_ps(cptr + zq_mm_align_size6);
		vc57 = zq_mm_load_ps(cptr + zq_mm_align_size7);
		cptr += ldc;
	}
	if (M > 6)
	{
		vc60 = zq_mm_load_ps(cptr);
		vc61 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc62 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		vc63 = zq_mm_load_ps(cptr + zq_mm_align_size3);
		vc64 = zq_mm_load_ps(cptr + zq_mm_align_size4);
		vc65 = zq_mm_load_ps(cptr + zq_mm_align_size5);
		vc66 = zq_mm_load_ps(cptr + zq_mm_align_size6);
		vc67 = zq_mm_load_ps(cptr + zq_mm_align_size7);
		cptr += ldc;
	}
	if (M > 7)
	{
		vc70 = zq_mm_load_ps(cptr);
		vc71 = zq_mm_load_ps(cptr + zq_mm_align_size);
		vc72 = zq_mm_load_ps(cptr + zq_mm_align_size2);
		vc73 = zq_mm_load_ps(cptr + zq_mm_align_size3);
		vc74 = zq_mm_load_ps(cptr + zq_mm_align_size4);
		vc75 = zq_mm_load_ps(cptr + zq_mm_align_size5);
		vc76 = zq_mm_load_ps(cptr + zq_mm_align_size6);
		vc77 = zq_mm_load_ps(cptr + zq_mm_align_size7);
	}
	vb0 = zq_mm_load_ps(bptr);
	vb1 = zq_mm_load_ps(bptr + zq_mm_align_size);
	vb2 = zq_mm_load_ps(bptr + zq_mm_align_size2);
	vb3 = zq_mm_load_ps(bptr + zq_mm_align_size3);
	vb4 = zq_mm_load_ps(bptr + zq_mm_align_size4);
	vb5 = zq_mm_load_ps(bptr + zq_mm_align_size5);
	vb6 = zq_mm_load_ps(bptr + zq_mm_align_size6);
	vb7 = zq_mm_load_ps(bptr + zq_mm_align_size7);

	for (int p = 0; p < (K - 1); ++p)
	{
		va = zq_mm_broadcast_ss(aptr);
		vc00 = zq_mm_fmadd_ps(vb0, va, vc00);
		vc01 = zq_mm_fmadd_ps(vb1, va, vc01);
		vc02 = zq_mm_fmadd_ps(vb2, va, vc02);
		vc03 = zq_mm_fmadd_ps(vb3, va, vc03);
		vc04 = zq_mm_fmadd_ps(vb4, va, vc04);
		vc05 = zq_mm_fmadd_ps(vb5, va, vc05);
		vc06 = zq_mm_fmadd_ps(vb6, va, vc06);
		vc07 = zq_mm_fmadd_ps(vb7, va, vc07);

		if (M > 1)
		{
			va = zq_mm_broadcast_ss(aptr + 1);
			vc10 = zq_mm_fmadd_ps(vb0, va, vc10);
			vc11 = zq_mm_fmadd_ps(vb1, va, vc11);
			vc12 = zq_mm_fmadd_ps(vb2, va, vc12);
			vc13 = zq_mm_fmadd_ps(vb3, va, vc13);
			vc14 = zq_mm_fmadd_ps(vb4, va, vc14);
			vc15 = zq_mm_fmadd_ps(vb5, va, vc15);
			vc16 = zq_mm_fmadd_ps(vb6, va, vc16);
			vc17 = zq_mm_fmadd_ps(vb7, va, vc17);
		}

		if (M > 2)
		{
			va = zq_mm_broadcast_ss(aptr + 2);
			vc20 = zq_mm_fmadd_ps(vb0, va, vc20);
			vc21 = zq_mm_fmadd_ps(vb1, va, vc21);
			vc22 = zq_mm_fmadd_ps(vb2, va, vc22);
			vc23 = zq_mm_fmadd_ps(vb3, va, vc23);
			vc24 = zq_mm_fmadd_ps(vb4, va, vc24);
			vc25 = zq_mm_fmadd_ps(vb5, va, vc25);
			vc26 = zq_mm_fmadd_ps(vb6, va, vc26);
			vc27 = zq_mm_fmadd_ps(vb7, va, vc27);
		}

		if (M > 3)
		{
			va = zq_mm_broadcast_ss(aptr + 3);
			vc30 = zq_mm_fmadd_ps(vb0, va, vc30);
			vc31 = zq_mm_fmadd_ps(vb1, va, vc31);
			vc32 = zq_mm_fmadd_ps(vb2, va, vc32);
			vc33 = zq_mm_fmadd_ps(vb3, va, vc33);
			vc34 = zq_mm_fmadd_ps(vb4, va, vc34);
			vc35 = zq_mm_fmadd_ps(vb5, va, vc35);
			vc36 = zq_mm_fmadd_ps(vb6, va, vc36);
			vc37 = zq_mm_fmadd_ps(vb7, va, vc37);
		}

		if (M > 4)
		{
			va = zq_mm_broadcast_ss(aptr + 4);
			vc40 = zq_mm_fmadd_ps(vb0, va, vc40);
			vc41 = zq_mm_fmadd_ps(vb1, va, vc41);
			vc42 = zq_mm_fmadd_ps(vb2, va, vc42);
			vc43 = zq_mm_fmadd_ps(vb3, va, vc43);
			vc44 = zq_mm_fmadd_ps(vb4, va, vc44);
			vc45 = zq_mm_fmadd_ps(vb5, va, vc45);
			vc46 = zq_mm_fmadd_ps(vb6, va, vc46);
			vc47 = zq_mm_fmadd_ps(vb7, va, vc47);
		}

		if (M > 5)
		{
			va = zq_mm_broadcast_ss(aptr + 5);
			vc50 = zq_mm_fmadd_ps(vb0, va, vc50);
			vc51 = zq_mm_fmadd_ps(vb1, va, vc51);
			vc52 = zq_mm_fmadd_ps(vb2, va, vc52);
			vc53 = zq_mm_fmadd_ps(vb3, va, vc53);
			vc54 = zq_mm_fmadd_ps(vb4, va, vc54);
			vc55 = zq_mm_fmadd_ps(vb5, va, vc55);
			vc56 = zq_mm_fmadd_ps(vb6, va, vc56);
			vc57 = zq_mm_fmadd_ps(vb7, va, vc57);
		}

		if (M > 6)
		{
			va = zq_mm_broadcast_ss(aptr + 6);
			vc60 = zq_mm_fmadd_ps(vb0, va, vc60);
			vc61 = zq_mm_fmadd_ps(vb1, va, vc61);
			vc62 = zq_mm_fmadd_ps(vb2, va, vc62);
			vc63 = zq_mm_fmadd_ps(vb3, va, vc63);
			vc64 = zq_mm_fmadd_ps(vb4, va, vc64);
			vc65 = zq_mm_fmadd_ps(vb5, va, vc65);
			vc66 = zq_mm_fmadd_ps(vb6, va, vc66);
			vc67 = zq_mm_fmadd_ps(vb7, va, vc67);
		}

		if (M > 7)
		{
			va = zq_mm_broadcast_ss(aptr + 7);
			vc70 = zq_mm_fmadd_ps(vb0, va, vc70);
			vc71 = zq_mm_fmadd_ps(vb1, va, vc71);
			vc72 = zq_mm_fmadd_ps(vb2, va, vc72);
			vc73 = zq_mm_fmadd_ps(vb3, va, vc73);
			vc74 = zq_mm_fmadd_ps(vb4, va, vc74);
			vc75 = zq_mm_fmadd_ps(vb5, va, vc75);
			vc76 = zq_mm_fmadd_ps(vb6, va, vc76);
			vc77 = zq_mm_fmadd_ps(vb7, va, vc77);
		}

		bptr += zq_mm_align_size8;
		aptr += M;
		vb0 = zq_mm_load_ps(bptr);
		vb1 = zq_mm_load_ps(bptr + zq_mm_align_size);
		vb2 = zq_mm_load_ps(bptr + zq_mm_align_size2);
		vb3 = zq_mm_load_ps(bptr + zq_mm_align_size3);
		vb4 = zq_mm_load_ps(bptr + zq_mm_align_size4);
		vb5 = zq_mm_load_ps(bptr + zq_mm_align_size5);
		vb6 = zq_mm_load_ps(bptr + zq_mm_align_size6);
		vb7 = zq_mm_load_ps(bptr + zq_mm_align_size7);

	}
	cptr = c;
	va = zq_mm_broadcast_ss(aptr);
	vc00 = zq_mm_fmadd_ps(vb0, va, vc00);
	vc01 = zq_mm_fmadd_ps(vb1, va, vc01);
	vc02 = zq_mm_fmadd_ps(vb2, va, vc02);
	vc03 = zq_mm_fmadd_ps(vb3, va, vc03);
	vc04 = zq_mm_fmadd_ps(vb4, va, vc04);
	vc05 = zq_mm_fmadd_ps(vb5, va, vc05);
	vc06 = zq_mm_fmadd_ps(vb6, va, vc06);
	vc07 = zq_mm_fmadd_ps(vb7, va, vc07);
	zq_mm_store_ps(cptr, vc00);
	zq_mm_store_ps(cptr + zq_mm_align_size, vc01);
	zq_mm_store_ps(cptr + zq_mm_align_size2, vc02);
	zq_mm_store_ps(cptr + zq_mm_align_size3, vc03);
	zq_mm_store_ps(cptr + zq_mm_align_size4, vc04);
	zq_mm_store_ps(cptr + zq_mm_align_size5, vc05);
	zq_mm_store_ps(cptr + zq_mm_align_size6, vc06);
	zq_mm_store_ps(cptr + zq_mm_align_size7, vc07);
	if (M > 1)
	{
		va = zq_mm_broadcast_ss(aptr + 1);
		vc10 = zq_mm_fmadd_ps(vb0, va, vc10);
		vc11 = zq_mm_fmadd_ps(vb1, va, vc11);
		vc12 = zq_mm_fmadd_ps(vb2, va, vc12);
		vc13 = zq_mm_fmadd_ps(vb3, va, vc13);
		vc14 = zq_mm_fmadd_ps(vb4, va, vc14);
		vc15 = zq_mm_fmadd_ps(vb5, va, vc15);
		vc16 = zq_mm_fmadd_ps(vb6, va, vc16);
		vc17 = zq_mm_fmadd_ps(vb7, va, vc17);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc10);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc11);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc12);
		zq_mm_store_ps(cptr + zq_mm_align_size3, vc13);
		zq_mm_store_ps(cptr + zq_mm_align_size4, vc14);
		zq_mm_store_ps(cptr + zq_mm_align_size5, vc15);
		zq_mm_store_ps(cptr + zq_mm_align_size6, vc16);
		zq_mm_store_ps(cptr + zq_mm_align_size7, vc17);
	}
	if (M > 2)
	{
		va = zq_mm_broadcast_ss(aptr + 2);
		vc20 = zq_mm_fmadd_ps(vb0, va, vc20);
		vc21 = zq_mm_fmadd_ps(vb1, va, vc21);
		vc22 = zq_mm_fmadd_ps(vb2, va, vc22);
		vc23 = zq_mm_fmadd_ps(vb3, va, vc23);
		vc24 = zq_mm_fmadd_ps(vb4, va, vc24);
		vc25 = zq_mm_fmadd_ps(vb5, va, vc25);
		vc26 = zq_mm_fmadd_ps(vb6, va, vc26);
		vc27 = zq_mm_fmadd_ps(vb7, va, vc27);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc20);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc21);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc22);
		zq_mm_store_ps(cptr + zq_mm_align_size3, vc23);
		zq_mm_store_ps(cptr + zq_mm_align_size4, vc24);
		zq_mm_store_ps(cptr + zq_mm_align_size5, vc25);
		zq_mm_store_ps(cptr + zq_mm_align_size6, vc26);
		zq_mm_store_ps(cptr + zq_mm_align_size7, vc27);
	}
	if (M > 3)
	{
		va = zq_mm_broadcast_ss(aptr + 3);
		vc30 = zq_mm_fmadd_ps(vb0, va, vc30);
		vc31 = zq_mm_fmadd_ps(vb1, va, vc31);
		vc32 = zq_mm_fmadd_ps(vb2, va, vc32);
		vc33 = zq_mm_fmadd_ps(vb3, va, vc33);
		vc34 = zq_mm_fmadd_ps(vb4, va, vc34);
		vc35 = zq_mm_fmadd_ps(vb5, va, vc35);
		vc36 = zq_mm_fmadd_ps(vb6, va, vc36);
		vc37 = zq_mm_fmadd_ps(vb7, va, vc37);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc30);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc31);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc32);
		zq_mm_store_ps(cptr + zq_mm_align_size3, vc33);
		zq_mm_store_ps(cptr + zq_mm_align_size4, vc34);
		zq_mm_store_ps(cptr + zq_mm_align_size5, vc35);
		zq_mm_store_ps(cptr + zq_mm_align_size6, vc36);
		zq_mm_store_ps(cptr + zq_mm_align_size7, vc37);
	}
	if (M > 4)
	{
		va = zq_mm_broadcast_ss(aptr + 4);
		vc40 = zq_mm_fmadd_ps(vb0, va, vc40);
		vc41 = zq_mm_fmadd_ps(vb1, va, vc41);
		vc42 = zq_mm_fmadd_ps(vb2, va, vc42);
		vc43 = zq_mm_fmadd_ps(vb3, va, vc43);
		vc44 = zq_mm_fmadd_ps(vb4, va, vc44);
		vc45 = zq_mm_fmadd_ps(vb5, va, vc45);
		vc46 = zq_mm_fmadd_ps(vb6, va, vc46);
		vc47 = zq_mm_fmadd_ps(vb7, va, vc47);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc40);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc41);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc42);
		zq_mm_store_ps(cptr + zq_mm_align_size3, vc43);
		zq_mm_store_ps(cptr + zq_mm_align_size4, vc44);
		zq_mm_store_ps(cptr + zq_mm_align_size5, vc45);
		zq_mm_store_ps(cptr + zq_mm_align_size6, vc46);
		zq_mm_store_ps(cptr + zq_mm_align_size7, vc47);
	}
	if (M > 5)
	{
		va = zq_mm_broadcast_ss(aptr + 5);
		vc50 = zq_mm_fmadd_ps(vb0, va, vc50);
		vc51 = zq_mm_fmadd_ps(vb1, va, vc51);
		vc52 = zq_mm_fmadd_ps(vb2, va, vc52);
		vc53 = zq_mm_fmadd_ps(vb3, va, vc53);
		vc54 = zq_mm_fmadd_ps(vb4, va, vc54);
		vc55 = zq_mm_fmadd_ps(vb5, va, vc55);
		vc56 = zq_mm_fmadd_ps(vb6, va, vc56);
		vc57 = zq_mm_fmadd_ps(vb7, va, vc57);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc50);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc51);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc52);
		zq_mm_store_ps(cptr + zq_mm_align_size3, vc53);
		zq_mm_store_ps(cptr + zq_mm_align_size4, vc54);
		zq_mm_store_ps(cptr + zq_mm_align_size5, vc55);
		zq_mm_store_ps(cptr + zq_mm_align_size6, vc56);
		zq_mm_store_ps(cptr + zq_mm_align_size7, vc57);
	}
	if (M > 6)
	{
		va = zq_mm_broadcast_ss(aptr + 6);
		vc60 = zq_mm_fmadd_ps(vb0, va, vc60);
		vc61 = zq_mm_fmadd_ps(vb1, va, vc61);
		vc62 = zq_mm_fmadd_ps(vb2, va, vc62);
		vc63 = zq_mm_fmadd_ps(vb3, va, vc63);
		vc64 = zq_mm_fmadd_ps(vb4, va, vc64);
		vc65 = zq_mm_fmadd_ps(vb5, va, vc65);
		vc66 = zq_mm_fmadd_ps(vb6, va, vc66);
		vc67 = zq_mm_fmadd_ps(vb7, va, vc67);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc60);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc61);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc62);
		zq_mm_store_ps(cptr + zq_mm_align_size3, vc63);
		zq_mm_store_ps(cptr + zq_mm_align_size4, vc64);
		zq_mm_store_ps(cptr + zq_mm_align_size5, vc65);
		zq_mm_store_ps(cptr + zq_mm_align_size6, vc66);
		zq_mm_store_ps(cptr + zq_mm_align_size7, vc67);
	}
	if (M > 7)
	{
		va = zq_mm_broadcast_ss(aptr + 7);
		vc70 = zq_mm_fmadd_ps(vb0, va, vc70);
		vc71 = zq_mm_fmadd_ps(vb1, va, vc71);
		vc72 = zq_mm_fmadd_ps(vb2, va, vc72);
		vc73 = zq_mm_fmadd_ps(vb3, va, vc73);
		vc74 = zq_mm_fmadd_ps(vb4, va, vc74);
		vc75 = zq_mm_fmadd_ps(vb5, va, vc75);
		vc76 = zq_mm_fmadd_ps(vb6, va, vc76);
		vc77 = zq_mm_fmadd_ps(vb7, va, vc77);
		cptr += ldc;
		zq_mm_store_ps(cptr, vc70);
		zq_mm_store_ps(cptr + zq_mm_align_size, vc71);
		zq_mm_store_ps(cptr + zq_mm_align_size2, vc72);
		zq_mm_store_ps(cptr + zq_mm_align_size3, vc73);
		zq_mm_store_ps(cptr + zq_mm_align_size4, vc74);
		zq_mm_store_ps(cptr + zq_mm_align_size5, vc75);
		zq_mm_store_ps(cptr + zq_mm_align_size6, vc76);
		zq_mm_store_ps(cptr + zq_mm_align_size7, vc77);
	}
}

typedef void(*kernel)(int, const float *, const float *, float *, int);

template<const int COL_BATCH>
kernel get_kernel_MxCOL(int k)
{
	if (COL_BATCH == zq_mm_align_size2)
	{
		switch (k)
		{
		case 1:
			return inner_kernel_MxALIGN2_template<1>;
			break;
		case 2:
			return inner_kernel_MxALIGN2_template<2>;
			break;
		case 3:
			return inner_kernel_MxALIGN2_template<3>;
			break;
		case 4:
			return inner_kernel_MxALIGN2_template<4>;
			break;
		case 5:
			return inner_kernel_MxALIGN2_template<5>;
			break;
		case 6:
			return inner_kernel_MxALIGN2_template<6>;
			break;
		case 7:
			return inner_kernel_MxALIGN2_template<7>;
			break;
		case 0:case 8:
			return inner_kernel_MxALIGN2_template<8>;
			break;
		}
	}
	else if (COL_BATCH == zq_mm_align_size3)
	{
		switch (k)
		{
		case 1:
			return inner_kernel_MxALIGN3_template<1>;
			break;
		case 2:
			return inner_kernel_MxALIGN3_template<2>;
			break;
		case 3:
			return inner_kernel_MxALIGN3_template<3>;
			break;
		case 4:
			return inner_kernel_MxALIGN3_template<4>;
			break;
		case 5:
			return inner_kernel_MxALIGN3_template<5>;
			break;
		case 6:
			return inner_kernel_MxALIGN3_template<6>;
			break;
		case 7:
			return inner_kernel_MxALIGN3_template<7>;
			break;
		case 0:case 8:
			return inner_kernel_MxALIGN3_template<8>;
			break;
		}
	}
	else if (COL_BATCH == zq_mm_align_size4)
	{
		switch (k)
		{
		case 1:
			return inner_kernel_MxALIGN4_template<1>;
			break;
		case 2:
			return inner_kernel_MxALIGN4_template<2>;
			break;
		case 3:
			return inner_kernel_MxALIGN4_template<3>;
			break;
		case 4:
			return inner_kernel_MxALIGN4_template<4>;
			break;
		case 5:
			return inner_kernel_MxALIGN4_template<5>;
			break;
		case 6:
			return inner_kernel_MxALIGN4_template<6>;
			break;
		case 7:
			return inner_kernel_MxALIGN4_template<7>;
			break;
		case 0:case 8:
			return inner_kernel_MxALIGN4_template<8>;
			break;
		}
	}
	else if (COL_BATCH == zq_mm_align_size8)
	{
		switch (k)
		{
		case 1:
			return inner_kernel_MxALIGN8_template<1>;
			break;
		case 2:
			return inner_kernel_MxALIGN8_template<2>;
			break;
		case 3:
			return inner_kernel_MxALIGN8_template<3>;
			break;
		case 4:
			return inner_kernel_MxALIGN8_template<4>;
			break;
		case 5:
			return inner_kernel_MxALIGN8_template<5>;
			break;
		case 6:
			return inner_kernel_MxALIGN8_template<6>;
			break;
		case 7:
			return inner_kernel_MxALIGN8_template<7>;
			break;
		case 0:case 8:
			return inner_kernel_MxALIGN8_template<8>;
			break;
		}
	}

	return NULL;
}

template<const int ROW_BATCH, const int COL_BATCH>
void compute_block(int M, int nc, int kc, const float* packA, const float* packB, float* loadC, float *C, int ldc, kernel rest_kernel)
{
	//printf("ROW_BATCH=%d, COL_BATCH=%d\n", ROW_BATCH, COL_BATCH);
	kernel full_kernel = get_kernel_MxCOL<COL_BATCH>(ROW_BATCH);
	int nc_ceil = align_ceil(nc, COL_BATCH);
	int nc_floor = nc - nc % zq_mm_align_size;
	for (int i = 0; i < M - M % ROW_BATCH; i += ROW_BATCH)
	{
		//Load C into cache
		float* rC = C + i * ldc;
		for (int m = 0; m < ROW_BATCH; ++m)
		{
			float* pC = rC + m * ldc;
			float* pL = loadC + m * nc_ceil;
			for (int n = 0; n < nc_floor; n += zq_mm_align_size)
			{
				zq_mm_store_ps(pL + n, zq_mm_load_ps(pC + n));
			}
			for (int n = nc_floor; n < nc; ++n)
			{
				pL[n] = pC[n];
			}
		}
		for (int j = 0; j < nc_ceil; j += COL_BATCH)
		{
			float* pC = loadC + j;
			const float* pA = packA + i * kc;
			const float* pB = packB + j * kc;
			full_kernel(kc, pA, pB, pC, nc_ceil);
		}
		//Write Results
		for (int m = 0; m < ROW_BATCH; ++m)
		{
			float* pC = rC + m * ldc;
			float* pL = loadC + m * nc_ceil;
			for (int n = 0; n < nc_floor; n += zq_mm_align_size)
			{
				zq_mm_type vec = zq_mm_load_ps(pL + n);

				zq_mm_store_ps(pC + n, vec);
			}
			//Last column batch.
			for (int n = nc_floor; n < nc; ++n)
			{
				float l = pL[n];
				pC[n] = l;
			}
		}
	}
	int m_len = M % ROW_BATCH;
	if (m_len)
	{
		int i = M - M % ROW_BATCH;
		//Load C into cache
		float* rC = C + i * ldc;
		for (int m = 0; m < m_len; ++m)
		{
			float* pC = rC + m * ldc;
			float* pL = loadC + m * nc_ceil;
			for (int n = 0; n < nc_floor; n += zq_mm_align_size)
			{
				zq_mm_store_ps(pL + n, zq_mm_load_ps(pC + n));
			}
			for (int n = nc_floor; n < nc; ++n)
			{
				pL[n] = pC[n];
			}
		}
		for (int j = 0; j < nc_ceil; j += COL_BATCH)
		{
			float* pC = loadC + j;
			const float* pA = packA + i * kc;
			const float* pB = packB + j * kc;
			rest_kernel(kc, pA, pB, pC, nc_ceil);
		}
		//Write Results
		for (int m = 0; m < m_len; ++m)
		{
			float* pC = rC + m * ldc;
			float* pL = loadC + m * nc_ceil;
			for (int n = 0; n < nc_floor; n += zq_mm_align_size)
			{
				zq_mm_type vec = zq_mm_load_ps(pL + n);

				zq_mm_store_ps(pC + n, vec);
			}
			//Last column batch.
			for (int n = nc_floor; n < nc; ++n)
			{
				float l = pL[n];
				pC[n] = l;
			}
		}
	}
}

template<const int ROW_BATCH>
void packed_sgemm_init(int M, int K, int kc, float* packA, const float* A, int lda)
{
	for (int p = 0; p < K; p += kc)
	{
		float* pPack = packA + (p / kc) * M * kc;
		for (int i = 0; i < M; i += ROW_BATCH)
		{
			int k_len = kc;
			int j_len = ROW_BATCH;
			if (M - i < ROW_BATCH)
			{
				j_len = M - i;
			}
			const float* pA = A + i * lda + p;
			if (K - p < kc)
				k_len = K - p;
			//Every ROW_BATCH rows are batched together.
			for (int k = 0; k < k_len; ++k)
			{
				for (int j = 0; j < j_len; ++j)
				{
					pPack[j] = pA[j * lda];
				}
				pPack += j_len;
				pA++;
			}
		}
	}
}

template<const int COL_BATCH>
void pack_B_avx(int kc, int nc, float* packB, const float* B, int ldb)
{
	int nc_floor = nc - nc % COL_BATCH;
	for (int k = 0; k < kc; ++k)
	{
		const float* pB = B + k * ldb;
		for (int j = 0; j < nc_floor; j += COL_BATCH)
		{
			float* pPack = packB + (j / COL_BATCH) * kc * COL_BATCH + k * COL_BATCH;
			zq_mm_store_ps(pPack, zq_mm_load_ps(pB));
			zq_mm_store_ps(pPack + zq_mm_align_size, zq_mm_load_ps(pB + zq_mm_align_size));
			if (COL_BATCH > zq_mm_align_size2)
				zq_mm_store_ps(pPack + zq_mm_align_size2, zq_mm_load_ps(pB + zq_mm_align_size2));
			if (COL_BATCH > zq_mm_align_size3)
				zq_mm_store_ps(pPack + zq_mm_align_size3, zq_mm_load_ps(pB + zq_mm_align_size3));
			pB += COL_BATCH;
		}
		if (nc_floor < nc)
		{
			int j = nc_floor;
			int n_len = nc - nc_floor;
			float* pPack = packB + (j / COL_BATCH) * kc * COL_BATCH + k * COL_BATCH;
			for (int i = 0; i < n_len; ++i)
			{
				pPack[i] = pB[i];
			}
		}
	}
}

template <const int ROW_BATCH, const int COL_BATCH>
void packed_sgemm(int M, int N, int K, const float *packA, const float *b, int ldb, float *c, int ldc, int nc, int kc, float* pack_array)
{
	kernel rest_kernel = get_kernel_MxCOL<COL_BATCH>(M%ROW_BATCH);
	for (int i = 0; i < M; ++i)
	{
		memset(c + ldc * i, 0, sizeof(float) * N);
	}

	int M_align = align_ceil(M, ROW_BATCH);
	int N_align = align_ceil(N, COL_BATCH);

	int NBlocks = (N_align + nc - 1) / nc;
	int KBlocks = (K + kc - 1) / kc;

	float* packB = pack_array;
	float* loadC = pack_array + kc * nc;
	//printf("loadC %x %d\n", loadC, ((size_t) loadC) % 32);

	//Our GEMM is implemented in GEPB fashion, as the operands are row-major
	int k_len = kc;
	int n_len = nc;
	for (int kt = 0; kt < KBlocks - 1; ++kt)
	{
		for (int nt = 0; nt < NBlocks; ++nt)
		{
			const float* pA = packA + kt * kc * M;
			const float* pB = b + kt * kc * ldb + nt * nc;
			float* pC = c + nt * nc;
			if (nt == NBlocks - 1)
				n_len = N - nt * nc;
			else
				n_len = nc;
			memset(packB, 0, sizeof(float) * kc * nc);
			pack_B_avx<COL_BATCH>(k_len, n_len, packB, pB, N);
			compute_block<ROW_BATCH, COL_BATCH>(M, n_len, k_len, pA, packB, loadC, pC, ldc, rest_kernel);
		}
	}
	{
		int kt = KBlocks - 1;
		k_len = (K - kt * kc);
		for (int nt = 0; nt < NBlocks; ++nt)
		{
			const float* pA = packA + kt * kc * M;
			const float* pB = b + kt * kc * ldb + nt * nc;
			float* pC = c + nt * nc;
			if (nt == NBlocks - 1)
				n_len = N - nt * nc;
			else
				n_len = nc;
			//I'm going to pack B in here.
			memset(packB, 0, sizeof(float) * kc * nc);
			pack_B_avx<COL_BATCH>(k_len, n_len, packB, pB, N);
			compute_block<ROW_BATCH, COL_BATCH>(M, n_len, k_len, pA, packB, loadC, pC, ldc, rest_kernel);
		}
	}
}


bool check_value(int M, int N, const float* C1, int ldc1, const float* C2, int ldc2, float thresh, bool show)
{
	int m, n;
	const float* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	float v1, v2;
	bool ret = true;
	for (m = 0, Cptr1 = C1, Cptr2 = C2; m < M; m++, Cptr1 += ldc1, Cptr2 += ldc2)
	{
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n++)
		{
			v1 = *C_c_ptr1;
			v2 = *C_c_ptr2;
			float scale = __max(fabs(v1), fabs(v2));
			float real_thresh = __max(thresh, thresh*scale);
			if (fabs(v1 - v2) > real_thresh)
			{
				if (show)
					printf("%d,%d = %f %f\n", m, n, v1, v2);
				ret = false;
			}
			C_c_ptr1++;
			C_c_ptr2++;
		}
	}
	return ret;
}


void MatMul0_AB(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc)
{
	int m, n, k;
	float sum;
	const float* A_row_ptr, *B_row_ptr;
	float *C_row_ptr;
	const float* A_c_ptr, *B_c_ptr;
	for (m = 0, A_row_ptr = A, C_row_ptr = C; m < M; m++, A_row_ptr += lda, C_row_ptr += ldc)
	{
		for (n = 0, B_c_ptr = B; n < N; n++, B_c_ptr++)
		{
			sum = 0;
			for (k = 0, A_c_ptr = A_row_ptr, B_row_ptr = B_c_ptr; k < K; k++, B_row_ptr += ldb)
				sum += (*(A_c_ptr++)) * (*B_row_ptr);
			C_row_ptr[n] = sum;
		}
	}
}

template<int ROW_BATCH, int COL_BATCH>
void MatMul_GEPB(int M, int N, int K, const float* packA, const float* B, int ldb, float* C, int ldc, int nc, int kc, float* pack_array)
{
	packed_sgemm<ROW_BATCH, COL_BATCH>(M, N, K, packA, B, N, C, N, nc, kc, pack_array);
}

template<int ROW_BATCH, int COL_BATCH>
int test(int M, int N, int K, int nIters, float thresh, bool show, int nc = 96, int kc = 256)
{
	printf("BATCH: ROW = %d, COL= %d\n", ROW_BATCH, COL_BATCH);
	float* A = (float*)_aligned_malloc(M*K * sizeof(float), 32);
	float* packA = (float*)_aligned_malloc(M*K * sizeof(float), 32);
	float* pack_array = (float*)_aligned_malloc((nc*kc + M*(N + 32)) * sizeof(float), 32);
	//memset(pack_array, 0, sizeof((nc*kc + M*(N + 32)) * sizeof(float)));
	float* B = (float*)_aligned_malloc(K*N * sizeof(float), 32);
	float* C0 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C1 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	for (int i = 0; i < M*K; i++)
	{
		A[i] = rand() % 10001 / 5000.0f - 1.0f;
	}
	for (int i = 0; i < K*N; i++)
	{
		B[i] = rand() % 10001 / 5000.0f - 1.0f;
	}
	packed_sgemm_init<ROW_BATCH>(M, K, kc, packA, A, K);
	int nIter0 = __max(1, nIters / 10);
	double mul_count0 = (double)nIter0*M*N*K / (1024 * 1024 * 1024);
	double mul_count = (double)nIters*M*N*K / (1024 * 1024 * 1024);
	double time = 0;
	double t1 = omp_get_wtime();
	for (int i = 0; i < nIter0; i++)
		MatMul0_AB(M, N, K, A, K, B, N, C0, N);
	double t2 = omp_get_wtime();
	time = t2 - t1;
	printf("%d x %d x %d, cost = %.3f s, naive gflops = %.3f\n", M, N, K, time, mul_count0 / time);

	for (int i = 0; i < nIters; i++)
		MatMul_GEPB<ROW_BATCH, COL_BATCH>(M, N, K, packA, B, N, C1, N, nc, kc, pack_array);
	double t3 = omp_get_wtime();
	time = t3 - t2;
	printf("%d x %d x %d, cost = %.3f s, GEPB gflops = %.3f\n", M, N, K, time, mul_count / time);

	printf("check C0 C1 = %s\n", check_value(M, N, C0, N, C1, N, thresh, show) ? "True" : "False");

	_aligned_free(A);
	_aligned_free(B);
	_aligned_free(packA);
	_aligned_free(pack_array);
	_aligned_free(C0);
	_aligned_free(C1);
	return 0;
}

int main()
{
#if __ARM_NEON
	const int ALIGN2 = 8, ALIGN3 = 12, ALIGN4 = 16, ALIGN8 = 32;
#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	const int ALIGN2 = 16, ALIGN3 = 24, ALIGN4 = 32, ALIGN8 = 64;
#elif ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	const int ALIGN2 = 8, ALIGN3 = 12, ALIGN4 = 16, ALIGN8 = 32;
#endif
#endif
	
	/*while (1)
	{
		int M = 8 + rand() % 1000;
		int K = 8 + rand() % 100 * 8;
		int N = 8 + rand() % 100 * 8;
		test<8, ALIGN2>(M, N, K, 1, 1e-4, true);
		test<8, ALIGN3>(M, N, K, 1, 1e-4, true);
		test<8, ALIGN4>(M, N, K, 1, 1e-4, true);
		test<6, ALIGN2>(M, N, K, 1, 1e-4, true);
		test<6, ALIGN3>(M, N, K, 1, 1e-4, true);
		test<6, ALIGN4>(M, N, K, 1, 1e-4, true);
		test<4, ALIGN2>(M, N, K, 1, 1e-4, true);
		test<4, ALIGN3>(M, N, K, 1, 1e-4, true);
		test<4, ALIGN4>(M, N, K, 1, 1e-4, true);
		test<2, ALIGN2>(M, N, K, 1, 1e-4, true);
		test<2, ALIGN3>(M, N, K, 1, 1e-4, true);
		test<2, ALIGN4>(M, N, K, 1, 1e-4, true);
	}*/

	
	test<8, ALIGN2>(64, 64, 64, 50000, 1e-5, false);
	test<8, ALIGN3>(64, 64, 64, 50000, 1e-5, false);
	test<8, ALIGN4>(64, 64, 64, 50000, 1e-5, false);
	test<8, ALIGN8>(64, 64, 64, 50000, 1e-5, false);
	test<6, ALIGN2>(64, 64, 64, 50000, 1e-5, false);
	test<6, ALIGN3>(64, 64, 64, 50000, 1e-5, false);
	test<6, ALIGN4>(64, 64, 64, 50000, 1e-5, false);
	test<6, ALIGN8>(64, 64, 64, 50000, 1e-5, false);
	test<4, ALIGN2>(64, 64, 64, 50000, 1e-5, false);
	test<4, ALIGN3>(64, 64, 64, 50000, 1e-5, false);
	test<4, ALIGN4>(64, 64, 64, 50000, 1e-5, false);
	test<4, ALIGN8>(64, 64, 64, 50000, 1e-5, false);
	test<2, ALIGN2>(64, 64, 64, 50000, 1e-5, false);
	test<2, ALIGN3>(64, 64, 64, 50000, 1e-5, false);
	test<2, ALIGN4>(64, 64, 64, 50000, 1e-5, false);
	test<2, ALIGN8>(64, 64, 64, 50000, 1e-5, false);

	test<8, ALIGN2>(128, 128, 128, 8000, 1e-5, false);
	test<8, ALIGN3>(128, 128, 128, 8000, 1e-5, false);
	test<8, ALIGN4>(128, 128, 128, 8000, 1e-5, false);
	test<8, ALIGN8>(128, 128, 128, 8000, 1e-5, false);
	test<6, ALIGN2>(128, 128, 128, 8000, 1e-5, false);
	test<6, ALIGN3>(128, 128, 128, 8000, 1e-5, false);
	test<6, ALIGN4>(128, 128, 128, 8000, 1e-5, false);
	test<6, ALIGN8>(128, 128, 128, 8000, 1e-5, false);
	test<4, ALIGN2>(128, 128, 128, 8000, 1e-5, false);
	test<4, ALIGN3>(128, 128, 128, 8000, 1e-5, false);
	test<4, ALIGN4>(128, 128, 128, 8000, 1e-5, false);
	test<4, ALIGN8>(128, 128, 128, 8000, 1e-5, false);
	test<2, ALIGN2>(128, 128, 128, 8000, 1e-5, false);
	test<2, ALIGN3>(128, 128, 128, 8000, 1e-5, false);
	test<2, ALIGN4>(128, 128, 128, 8000, 1e-5, false);
	test<2, ALIGN8>(128, 128, 128, 8000, 1e-5, false);

	test<8, ALIGN2>(256, 256, 256, 1000, 1e-5, false);
	test<8, ALIGN3>(256, 256, 256, 1000, 1e-5, false);
	test<8, ALIGN4>(256, 256, 256, 1000, 1e-5, false);
	test<8, ALIGN8>(256, 256, 256, 1000, 1e-5, false);
	test<6, ALIGN2>(256, 256, 256, 1000, 1e-5, false);
	test<6, ALIGN3>(256, 256, 256, 1000, 1e-5, false);
	test<6, ALIGN4>(256, 256, 256, 1000, 1e-5, false);
	test<6, ALIGN8>(256, 256, 256, 1000, 1e-5, false);
	test<4, ALIGN2>(256, 256, 256, 1000, 1e-5, false);
	test<4, ALIGN3>(256, 256, 256, 1000, 1e-5, false);
	test<4, ALIGN4>(256, 256, 256, 1000, 1e-5, false);
	test<4, ALIGN8>(256, 256, 256, 1000, 1e-5, false);
	test<2, ALIGN2>(256, 256, 256, 1000, 1e-5, false);
	test<2, ALIGN3>(256, 256, 256, 1000, 1e-5, false);
	test<2, ALIGN4>(256, 256, 256, 1000, 1e-5, false);
	test<2, ALIGN8>(256, 256, 256, 1000, 1e-5, false);

	test<8, ALIGN2>(512, 512, 512, 125, 1e-5, false);
	test<8, ALIGN3>(512, 512, 512, 125, 1e-5, false);
	test<8, ALIGN4>(512, 512, 512, 125, 1e-5, false);
	test<8, ALIGN4>(512, 512, 512, 125, 1e-5, false);
	test<6, ALIGN2>(512, 512, 512, 125, 1e-5, false);
	test<6, ALIGN3>(512, 512, 512, 125, 1e-5, false);
	test<6, ALIGN4>(512, 512, 512, 125, 1e-5, false);
	test<6, ALIGN8>(512, 512, 512, 125, 1e-5, false);
	test<4, ALIGN2>(512, 512, 512, 125, 1e-5, false);
	test<4, ALIGN3>(512, 512, 512, 125, 1e-5, false);
	test<4, ALIGN4>(512, 512, 512, 125, 1e-5, false);
	test<4, ALIGN8>(512, 512, 512, 125, 1e-5, false);
	test<2, ALIGN2>(512, 512, 512, 125, 1e-5, false);
	test<2, ALIGN3>(512, 512, 512, 125, 1e-5, false);
	test<2, ALIGN4>(512, 512, 512, 125, 1e-5, false);
	test<2, ALIGN8>(512, 512, 512, 125, 1e-5, false);

	test<8, ALIGN2>(1024, 1024, 1024, 16, 1e-4, false);
	test<8, ALIGN3>(1024, 1024, 1024, 16, 1e-4, false);
	test<8, ALIGN4>(1024, 1024, 1024, 16, 1e-4, false);
	test<8, ALIGN8>(1024, 1024, 1024, 16, 1e-4, false);
	test<6, ALIGN2>(1024, 1024, 1024, 16, 1e-4, false);
	test<6, ALIGN3>(1024, 1024, 1024, 16, 1e-4, false);
	test<6, ALIGN4>(1024, 1024, 1024, 16, 1e-4, false);
	test<6, ALIGN8>(1024, 1024, 1024, 16, 1e-4, false);
	test<4, ALIGN2>(1024, 1024, 1024, 16, 1e-4, false);
	test<4, ALIGN3>(1024, 1024, 1024, 16, 1e-4, false);
	test<4, ALIGN4>(1024, 1024, 1024, 16, 1e-4, false);
	test<4, ALIGN8>(1024, 1024, 1024, 16, 1e-4, false);
	test<2, ALIGN2>(1024, 1024, 1024, 16, 1e-4, false);
	test<2, ALIGN3>(1024, 1024, 1024, 16, 1e-4, false);
	test<2, ALIGN4>(1024, 1024, 1024, 16, 1e-4, false);
	test<2, ALIGN8>(1024, 1024, 1024, 16, 1e-4, false);


	return 0;
}