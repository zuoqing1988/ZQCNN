#include <stdio.h>
#include <intrin.h>//(include immintrin.h)
#include <time.h>
#define zq_mm_fmadd_ps _mm256_fmadd_ps
//#define zq_mm_fmadd_ps(A,B,C) _mm256_add_ps(_mm256_mul_ps(A,B),C)
#define final_sum(q) (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])
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
	register __m256 a1 = _mm256_set1_ps(1), a2 = _mm256_set1_ps(1), a3 = _mm256_set1_ps(1), a4 = _mm256_set1_ps(1);
	register __m256 a5 = _mm256_set1_ps(1), a6 = _mm256_set1_ps(1), a7 = _mm256_set1_ps(1), a8 = _mm256_set1_ps(1);
	register __m256 b1 = _mm256_set1_ps(1), b2 = _mm256_set1_ps(1), b3 = _mm256_set1_ps(1), b4 = _mm256_set1_ps(1);
	register __m256 b5 = _mm256_set1_ps(1), b6 = _mm256_set1_ps(1), b7 = _mm256_set1_ps(1), b8 = _mm256_set1_ps(1);
	register __m256 sum1 = _mm256_setzero_ps();
	register __m256 sum2 = _mm256_setzero_ps();
	register __m256 sum3 = _mm256_setzero_ps();
	register __m256 sum4 = _mm256_setzero_ps();
	register __m256 sum5 = _mm256_setzero_ps();
	register __m256 sum6 = _mm256_setzero_ps();
	register __m256 sum7 = _mm256_setzero_ps();
	register __m256 sum8 = _mm256_setzero_ps();
	register __m256 sum1_ = _mm256_setzero_ps();
	register __m256 sum2_ = _mm256_setzero_ps();
	register __m256 sum3_ = _mm256_setzero_ps();
	register __m256 sum4_ = _mm256_setzero_ps();
	register __m256 sum5_ = _mm256_setzero_ps();
	register __m256 sum6_ = _mm256_setzero_ps();
	register __m256 sum7_ = _mm256_setzero_ps();
	register __m256 sum8_ = _mm256_setzero_ps();
	register __m256 suma = _mm256_setzero_ps();
	register __m256 sumb = _mm256_setzero_ps();
	_declspec(align(32)) float q[8];
	const int M = 10000, N = 64 * 1000;
	clock_t t1 = clock();
	int i, j, k;
	for (i = 0; i < nIters; i++)
	{
		suma = _mm256_setzero_ps();
		for (j = 0; j < M; j++)
		{
			sum1 = _mm256_setzero_ps();
			sum2 = _mm256_setzero_ps();
			sum3 = _mm256_setzero_ps();
			sum4 = _mm256_setzero_ps();
			sum5 = _mm256_setzero_ps();
			sum6 = _mm256_setzero_ps();
			sum7 = _mm256_setzero_ps();
			sum8 = _mm256_setzero_ps();
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

			suma = _mm256_add_ps(suma, sum1);
			suma = _mm256_add_ps(suma, sum2);
			suma = _mm256_add_ps(suma, sum3);
			suma = _mm256_add_ps(suma, sum4);
			suma = _mm256_add_ps(suma, sum5);
			suma = _mm256_add_ps(suma, sum6);
			suma = _mm256_add_ps(suma, sum7);
			suma = _mm256_add_ps(suma, sum8);
		}
		sumb = _mm256_add_ps(sumb, suma);
	}
	clock_t t2 = clock();
	double time = (t2 - t1) / 1000.0;
	double mul_count = (double)nIters*M*N * 8 / (1024.0*1024.0*1024.0);
	printf("mul_count = %.3f G, time = %.3f s, gflops = %.3f\n", mul_count, time, mul_count / time);
	printf("i,j,k=%d,%d,%d\n", i, j, k);
	_mm256_store_ps(q, sum1);
	printf("%e %e %e %e %e %e %e %e\n", q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
	_mm256_store_ps(q, suma);
	printf("%e %e %e %e %e %e %e %e\n", q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
	_mm256_store_ps(q, sumb);
	printf("%e %e %e %e %e %e %e %e\n", q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
}