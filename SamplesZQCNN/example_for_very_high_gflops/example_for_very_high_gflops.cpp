#include <stdio.h>
#include <time.h>
#define ZQ_CNN_SSETYPE_SSE 1
#define ZQ_CNN_SSETYPE_AVX 2
#define ZQ_CNN_SSETYPE_AVX2 3

#if defined(_WIN32)
#include <intrin.h>//(include immintrin.h)
#define ZQ_CNN_USE_SSETYPE ZQ_CNN_SSETYPE_AVX2
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
#else
#define ZQ_CNN_USE_SSETYPE ZQ_CNN_SSETYPE_AVX
#include <x86intrin.h>
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX2
#define zq_mm_fmadd_ps _mm256_fmadd_ps
#elif ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#define zq_mm_fmadd_ps(A,B,C) _mm256_add_ps(_mm256_mul_ps(A,B),C)
#else
#define zq_mm_fmadd_ps(A,B,C) _mm_add_ps(_mm_mul_ps(A,B),C)
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#define final_sum(q) (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])
#define zq_mm_set1_ps _mm256_set1_ps
#define zq_mm_setzero_ps _mm256_setzero_ps
#define zq_mm_add_ps _mm256_add_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_type __m256
#define num_per_op 8
#else
#define final_sum(q) (q[0]+q[1]+q[2]+q[3])
#define zq_mm_set1_ps _mm_set1_ps
#define zq_mm_setzero_ps _mm_setzero_ps
#define zq_mm_add_ps _mm_add_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_type __m128
#define num_per_op 4
#endif

void example_for_very_high_gflops();

int main()
{
	example_for_very_high_gflops();
	example_for_very_high_gflops();
	example_for_very_high_gflops();
	return 0;
}

void example_for_very_high_gflops()
{
	int nIters = 50;
	register zq_mm_type a1 = zq_mm_set1_ps(1), a2 = zq_mm_set1_ps(1), a3 = zq_mm_set1_ps(1), a4 = zq_mm_set1_ps(1);
	register zq_mm_type a5 = zq_mm_set1_ps(1), a6 = zq_mm_set1_ps(1), a7 = zq_mm_set1_ps(1), a8 = zq_mm_set1_ps(1);
	register zq_mm_type b1 = zq_mm_set1_ps(1), b2 = zq_mm_set1_ps(1), b3 = zq_mm_set1_ps(1), b4 = zq_mm_set1_ps(1);
	register zq_mm_type b5 = zq_mm_set1_ps(1), b6 = zq_mm_set1_ps(1), b7 = zq_mm_set1_ps(1), b8 = zq_mm_set1_ps(1);
	register zq_mm_type sum1 = zq_mm_setzero_ps();
	register zq_mm_type sum2 = zq_mm_setzero_ps();
	register zq_mm_type sum3 = zq_mm_setzero_ps();
	register zq_mm_type sum4 = zq_mm_setzero_ps();
	register zq_mm_type sum5 = zq_mm_setzero_ps();
	register zq_mm_type sum6 = zq_mm_setzero_ps();
	register zq_mm_type sum7 = zq_mm_setzero_ps();
	register zq_mm_type sum8 = zq_mm_setzero_ps();
	register zq_mm_type sum1_ = zq_mm_setzero_ps();
	register zq_mm_type sum2_ = zq_mm_setzero_ps();
	register zq_mm_type sum3_ = zq_mm_setzero_ps();
	register zq_mm_type sum4_ = zq_mm_setzero_ps();
	register zq_mm_type sum5_ = zq_mm_setzero_ps();
	register zq_mm_type sum6_ = zq_mm_setzero_ps();
	register zq_mm_type sum7_ = zq_mm_setzero_ps();
	register zq_mm_type sum8_ = zq_mm_setzero_ps();
	register zq_mm_type suma = zq_mm_setzero_ps();
	register zq_mm_type sumb = zq_mm_setzero_ps();
#if defined(_WIN32)
	_declspec(align(32)) float q[8];
#else
	__attribute__((aligned(32))) float q[8];
#endif
	const int M = 10000, N = 64 * 1000;
	clock_t t1 = clock();
	int i, j, k;
	for (i = 0; i < nIters; i++)
	{
		suma = zq_mm_setzero_ps();
		for (j = 0; j < M; j++)
		{
			sum1 = zq_mm_setzero_ps();
			sum2 = zq_mm_setzero_ps();
			sum3 = zq_mm_setzero_ps();
			sum4 = zq_mm_setzero_ps();
			sum5 = zq_mm_setzero_ps();
			sum6 = zq_mm_setzero_ps();
			sum7 = zq_mm_setzero_ps();
			sum8 = zq_mm_setzero_ps();
			for (k = 0; k < N; k += 64)
			{
				sum1_ = zq_mm_fmadd_ps(a1, b1, sum1);
				sum2_ = zq_mm_fmadd_ps(a2, b2, sum2);
				sum3_ = zq_mm_fmadd_ps(a3, b3, sum3);
				sum4_ = zq_mm_fmadd_ps(a4, b4, sum4);
				sum5_ = zq_mm_fmadd_ps(a5, b5, sum5);
				sum6_ = zq_mm_fmadd_ps(a6, b6, sum6);
				sum7_ = zq_mm_fmadd_ps(a7, b7, sum7);
				sum8_ = zq_mm_fmadd_ps(a8, b8, sum8);
				sum1 = zq_mm_fmadd_ps(a1, b1, sum1_);
				sum2 = zq_mm_fmadd_ps(a2, b2, sum2_);
				sum3 = zq_mm_fmadd_ps(a3, b3, sum3_);
				sum4 = zq_mm_fmadd_ps(a4, b4, sum4_);
				sum5 = zq_mm_fmadd_ps(a5, b5, sum5_);
				sum6 = zq_mm_fmadd_ps(a6, b6, sum6_);
				sum7 = zq_mm_fmadd_ps(a7, b7, sum7_);
				sum8 = zq_mm_fmadd_ps(a8, b8, sum8_);
				sum1_ = zq_mm_fmadd_ps(a1, b1, sum1);
				sum2_ = zq_mm_fmadd_ps(a2, b2, sum2);
				sum3_ = zq_mm_fmadd_ps(a3, b3, sum3);
				sum4_ = zq_mm_fmadd_ps(a4, b4, sum4);
				sum5_ = zq_mm_fmadd_ps(a5, b5, sum5);
				sum6_ = zq_mm_fmadd_ps(a6, b6, sum6);
				sum7_ = zq_mm_fmadd_ps(a7, b7, sum7);
				sum8_ = zq_mm_fmadd_ps(a8, b8, sum8);
				sum1 = zq_mm_fmadd_ps(a1, b1, sum1_);
				sum2 = zq_mm_fmadd_ps(a2, b2, sum2_);
				sum3 = zq_mm_fmadd_ps(a3, b3, sum3_);
				sum4 = zq_mm_fmadd_ps(a4, b4, sum4_);
				sum5 = zq_mm_fmadd_ps(a5, b5, sum5_);
				sum6 = zq_mm_fmadd_ps(a6, b6, sum6_);
				sum7 = zq_mm_fmadd_ps(a7, b7, sum7_);
				sum8 = zq_mm_fmadd_ps(a8, b8, sum8_);
				sum1_ = zq_mm_fmadd_ps(a1, b1, sum1);
				sum2_ = zq_mm_fmadd_ps(a2, b2, sum2);
				sum3_ = zq_mm_fmadd_ps(a3, b3, sum3);
				sum4_ = zq_mm_fmadd_ps(a4, b4, sum4);
				sum5_ = zq_mm_fmadd_ps(a5, b5, sum5);
				sum6_ = zq_mm_fmadd_ps(a6, b6, sum6);
				sum7_ = zq_mm_fmadd_ps(a7, b7, sum7);
				sum8_ = zq_mm_fmadd_ps(a8, b8, sum8);
				sum1 = zq_mm_fmadd_ps(a1, b1, sum1_);
				sum2 = zq_mm_fmadd_ps(a2, b2, sum2_);
				sum3 = zq_mm_fmadd_ps(a3, b3, sum3_);
				sum4 = zq_mm_fmadd_ps(a4, b4, sum4_);
				sum5 = zq_mm_fmadd_ps(a5, b5, sum5_);
				sum6 = zq_mm_fmadd_ps(a6, b6, sum6_);
				sum7 = zq_mm_fmadd_ps(a7, b7, sum7_);
				sum8 = zq_mm_fmadd_ps(a8, b8, sum8_);
				sum1_ = zq_mm_fmadd_ps(a1, b1, sum1);
				sum2_ = zq_mm_fmadd_ps(a2, b2, sum2);
				sum3_ = zq_mm_fmadd_ps(a3, b3, sum3);
				sum4_ = zq_mm_fmadd_ps(a4, b4, sum4);
				sum5_ = zq_mm_fmadd_ps(a5, b5, sum5);
				sum6_ = zq_mm_fmadd_ps(a6, b6, sum6);
				sum7_ = zq_mm_fmadd_ps(a7, b7, sum7);
				sum8_ = zq_mm_fmadd_ps(a8, b8, sum8);
				sum1 = zq_mm_fmadd_ps(a1, b1, sum1_);
				sum2 = zq_mm_fmadd_ps(a2, b2, sum2_);
				sum3 = zq_mm_fmadd_ps(a3, b3, sum3_);
				sum4 = zq_mm_fmadd_ps(a4, b4, sum4_);
				sum5 = zq_mm_fmadd_ps(a5, b5, sum5_);
				sum6 = zq_mm_fmadd_ps(a6, b6, sum6_);
				sum7 = zq_mm_fmadd_ps(a7, b7, sum7_);
				sum8 = zq_mm_fmadd_ps(a8, b8, sum8_);

			}

			suma = zq_mm_add_ps(suma, sum1);
			suma = zq_mm_add_ps(suma, sum2);
			suma = zq_mm_add_ps(suma, sum3);
			suma = zq_mm_add_ps(suma, sum4);
			suma = zq_mm_add_ps(suma, sum5);
			suma = zq_mm_add_ps(suma, sum6);
			suma = zq_mm_add_ps(suma, sum7);
			suma = zq_mm_add_ps(suma, sum8);
		}
		sumb = zq_mm_add_ps(sumb, suma);
	}
	clock_t t2 = clock();
	double time = (t2 - t1) / 1000.0;
	double mul_count = (double)nIters*M*N * num_per_op / (1024.0*1024.0*1024.0);
	printf("mul_count = %.3f G, time = %.3f s, gflops = %.3f\n", mul_count, time, mul_count / time);
	printf("i,j,k=%d,%d,%d\n", i, j, k);
	zq_mm_store_ps(q, sum1);
	printf("%e %e %e %e %e %e %e %e\n", q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
	zq_mm_store_ps(q, suma);
	printf("%e %e %e %e %e %e %e %e\n", q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
	zq_mm_store_ps(q, sumb);
	printf("%e %e %e %e %e %e %e %e\n", q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
}