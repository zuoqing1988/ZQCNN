#if !defined(_WIN32)
#include "ZQ_CNN_CompileConfig.h"
#endif


#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <time.h>

#if !__ARM_NEON

#if defined(_WIN32)
#include <intrin.h>//(include immintrin.h)
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
#define zq_mm_mul_ps _mm256_mul_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_type __m256
#define num_per_op 8
#define num_per_op2 16
#define num_per_op3 24
#define num_per_op4 32
#else
#define final_sum(q) (q[0]+q[1]+q[2]+q[3])
#define zq_mm_set1_ps _mm_set1_ps
#define zq_mm_setzero_ps _mm_setzero_ps
#define zq_mm_add_ps _mm_add_ps
#define zq_mm_mul_ps _mm_mul_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_load_ps _mm_load_ps
#define zq_mm_type __m128
#define num_per_op 4
#define num_per_op2 8
#define num_per_op3 12
#define num_per_op4 16
#endif

#else
#include <arm_neon.h>
#define final_sum(q) (q[0]+q[1]+q[2]+q[3])
#define zq_mm_store_ps vst1q_f32
#define zq_mm_load_ps vld1q_f32
#define zq_mm_add_ps vaddq_f32
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps(A, B, C) vfmaq_f32(C, A, B)
#else
#define zq_mm_fmadd_ps(A, B, C) vaddq_f32(vmulq_f32(A, B), C)
#endif
#define zq_mm_mul_ps vmulq_f32
#define zq_mm_setzero_ps() vdupq_n_f32(0)
#define zq_mm_set1_ps vdupq_n_f32
#define zq_mm_type float32x4_t
#define num_per_op 4
#define num_per_op2 8
#define num_per_op3 12
#define num_per_op4 16
#endif

void example_for_very_high_gflops();
void test_memcpy(int type,bool in_cache);
void test_4x4x4_in_cache();
void test_4x4x8_in_cache();
void test_4x4x16_in_cache();
int main()
{
	test_4x4x4_in_cache();
	test_4x4x8_in_cache();
	test_4x4x16_in_cache();
	example_for_very_high_gflops();
	example_for_very_high_gflops();
	example_for_very_high_gflops();
	test_memcpy(0, false);
	test_memcpy(1, false);
	test_memcpy(2, false);
	test_memcpy(0, true);
	test_memcpy(1, true);
	test_memcpy(2, true);
	return 0;
}

void example_for_very_high_gflops()
{
#if __ARM_NEON
	int nIters = 20;
#else
	int nIters = 50;
#endif
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
	ZQ_DECLSPEC_ALIGN32 float q[8];
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
#if defined(_WIN32)
	double time = (t2 - t1) / 1000.0;
#else
	double time = (t2 - t1) / 1000000.0;
#endif
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

void test_memcpy(int type, bool in_cache)
{
	int nIters = 3;
	clock_t t1, t2;
	register zq_mm_type vec0,vec1,vec2,vec3;
	int M = 1000, N = 1024*1024*num_per_op;
	if (in_cache)
	{
		M *= 1024*128;
		N /= 1024*128;
	}
	float* src = (float*)_aligned_malloc(N*sizeof(float),32);
	float* dst = (float*)_aligned_malloc(N*sizeof(float),32);
	float* cur_src_ptr, *cur_dst_ptr;
	for(int i = 0;i < N;i++)
		src[i] = i;
	if (type == 0)
	{
		printf("test load & store %s\n", in_cache ? "in cache":"not in cache");
		for (int i = 0; i < nIters; i++)
		{
			t1 = clock();
			for (int j = 0; j < M; j++)
			{
				cur_src_ptr = src;
				cur_dst_ptr = dst;
				for (int k = 0; k < N; k += num_per_op4)
				{
					vec0 = zq_mm_load_ps(cur_src_ptr);
					zq_mm_store_ps(cur_dst_ptr, vec0);
					vec1 = zq_mm_load_ps(cur_src_ptr + num_per_op);
					zq_mm_store_ps(cur_dst_ptr + num_per_op, vec1);
					vec2 = zq_mm_load_ps(cur_src_ptr + num_per_op2);
					zq_mm_store_ps(cur_dst_ptr + num_per_op2, vec2);
					vec3 = zq_mm_load_ps(cur_src_ptr + num_per_op3);
					zq_mm_store_ps(cur_dst_ptr + num_per_op3, vec3);
					cur_src_ptr += num_per_op4;
					cur_dst_ptr += num_per_op4;
				}
			}
			t2 = clock();
#if defined(_WIN32)
			double time = (t2 - t1)*0.001;
#else
			double time = (t2 - t1)*0.000001;
#endif
			double gflops = (double)M*N / (1024.0*1024.0*1024.0) / time;
			printf("time=%.3f s, gflops=%.3f\n", time, gflops);
		}
	}
	else if (type == 1)
	{
		printf("test only load (store to same addr) %s\n", in_cache ? "in cache" : "not in cache");
		for (int i = 0; i < nIters; i++)
		{
			t1 = clock();
			for (int j = 0; j < M; j++)
			{
				cur_src_ptr = src;
				cur_dst_ptr = dst;
				for (int k = 0; k < N; k += num_per_op4)
				{
					vec0 = zq_mm_load_ps(cur_src_ptr);
					zq_mm_store_ps(cur_dst_ptr, vec0);
					vec1 = zq_mm_load_ps(cur_src_ptr+num_per_op);
					zq_mm_store_ps(cur_dst_ptr + num_per_op, vec1);
					vec2 = zq_mm_load_ps(cur_src_ptr+num_per_op2);
					zq_mm_store_ps(cur_dst_ptr + num_per_op2, vec2);
					vec3 = zq_mm_load_ps(cur_src_ptr+num_per_op3);
					zq_mm_store_ps(cur_dst_ptr + num_per_op3, vec3);
					cur_src_ptr += num_per_op4;
				}
			}
			t2 = clock();
#if defined(_WIN32)
			double time = (t2 - t1)*0.001;
#else
			double time = (t2 - t1)*0.000001;
#endif
			double gflops = (double)M*N / (1024.0*1024.0*1024.0) / time;
			printf("time=%.3f s, gflops=%.3f\n", time, gflops);
		}
	}
	else
	{
		printf("test only store (load from same addr) %s\n", in_cache ? "in cache" : "not in cache");
		for (int i = 0; i < nIters; i++)
		{
			t1 = clock();
			for (int j = 0; j < M; j++)
			{
				cur_src_ptr = src;
				cur_dst_ptr = dst;
				for (int k = 0; k < N; k += num_per_op4)
				{
					vec0 = zq_mm_load_ps(cur_src_ptr);
					zq_mm_store_ps(cur_dst_ptr, vec0);
					vec1 = zq_mm_load_ps(cur_src_ptr + num_per_op);
					zq_mm_store_ps(cur_dst_ptr + num_per_op, vec1);
					vec2 = zq_mm_load_ps(cur_src_ptr + num_per_op2);
					zq_mm_store_ps(cur_dst_ptr + num_per_op2, vec2);
					vec3 = zq_mm_load_ps(cur_src_ptr + num_per_op3);
					zq_mm_store_ps(cur_dst_ptr + num_per_op3, vec3);
					cur_dst_ptr += num_per_op4;
				}
			}
			t2 = clock();
#if defined(_WIN32)
			double time = (t2 - t1)*0.001;
#else
			double time = (t2 - t1)*0.000001;
#endif
			double gflops = (double)M*N / (1024.0*1024.0*1024.0) / time;
			printf("time=%.3f s, gflops=%.3f\n", time, gflops);
		}
	}
	_aligned_free(src);
	_aligned_free(dst);
}

void test_4x4x4_in_cache()
{
	float* A = (float*)_aligned_malloc(4 * 4 * num_per_op * sizeof(float), num_per_op * 4);
	float* B = (float*)_aligned_malloc(4 * 4 * num_per_op * sizeof(float), num_per_op * 4);
	float* C = (float*)malloc(4 * 4 * sizeof(float));
	int i, j, k;
	for (i = 0; i < 4 * 4 * num_per_op; i++)
	{
		A[i] = i;
		B[i] = 1;
	}
	memset(C, 0, sizeof(float) * 16);
#if defined(_WIN32)
	_declspec(align(32)) float q[8];
#else
	ZQ_DECLSPEC_ALIGN32 float q[8];
#endif
	register zq_mm_type a0, a1, a2, a3, b0, b1, b2, b3;
	register zq_mm_type c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33;
	int nIters = 5;
	int M = 10 * 1024 * 1024;
	printf("test 4x4x%d\n", num_per_op * 4);
	for(int i = 0;i < nIters;i++)
	{ 
		clock_t t1 = clock();
		for (j = 0; j < M; j++)
		{
			a0 = zq_mm_load_ps(A);
			a1 = zq_mm_load_ps(A + 4 * num_per_op);
			a2 = zq_mm_load_ps(A + 8 * num_per_op);
			a3 = zq_mm_load_ps(A + 12 * num_per_op);
			b0 = zq_mm_load_ps(B);
			b1 = zq_mm_load_ps(B + 4 * num_per_op);
			b2 = zq_mm_load_ps(B + 8 * num_per_op);
			b3 = zq_mm_load_ps(B + 12 * num_per_op);
			c00 = zq_mm_mul_ps(a0, b0);
			c01 = zq_mm_mul_ps(a0, b1);
			c02 = zq_mm_mul_ps(a0, b2);
			c03 = zq_mm_mul_ps(a0, b3);
			c10 = zq_mm_mul_ps(a1, b0);
			c11 = zq_mm_mul_ps(a1, b1);
			c12 = zq_mm_mul_ps(a1, b2);
			c13 = zq_mm_mul_ps(a1, b3);
			c20 = zq_mm_mul_ps(a2, b0);
			c21 = zq_mm_mul_ps(a2, b1);
			c22 = zq_mm_mul_ps(a2, b2);
			c23 = zq_mm_mul_ps(a2, b3);
			c30 = zq_mm_mul_ps(a3, b0);
			c31 = zq_mm_mul_ps(a3, b1);
			c32 = zq_mm_mul_ps(a3, b2);
			c33 = zq_mm_mul_ps(a3, b3);
			a0 = zq_mm_load_ps(A + num_per_op);
			a1 = zq_mm_load_ps(A + num_per_op + 4 * num_per_op);
			a2 = zq_mm_load_ps(A + num_per_op + 8 * num_per_op);
			a3 = zq_mm_load_ps(A + num_per_op + 12 * num_per_op);
			b0 = zq_mm_load_ps(B + num_per_op);
			b1 = zq_mm_load_ps(B + num_per_op + 4 * num_per_op);
			b2 = zq_mm_load_ps(B + num_per_op + 8 * num_per_op);
			b3 = zq_mm_load_ps(B + num_per_op + 12 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + num_per_op2);
			a1 = zq_mm_load_ps(A + num_per_op2 + 4 * num_per_op);
			a2 = zq_mm_load_ps(A + num_per_op2 + 8 * num_per_op);
			a3 = zq_mm_load_ps(A + num_per_op2 + 12 * num_per_op);
			b0 = zq_mm_load_ps(B + num_per_op2);
			b1 = zq_mm_load_ps(B + num_per_op2 + 4 * num_per_op);
			b2 = zq_mm_load_ps(B + num_per_op2 + 8 * num_per_op);
			b3 = zq_mm_load_ps(B + num_per_op2 + 12 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + num_per_op3);
			a1 = zq_mm_load_ps(A + num_per_op3 + 4 * num_per_op);
			a2 = zq_mm_load_ps(A + num_per_op3 + 8 * num_per_op);
			a3 = zq_mm_load_ps(A + num_per_op3 + 12 * num_per_op);
			b0 = zq_mm_load_ps(B + num_per_op3);
			b1 = zq_mm_load_ps(B + num_per_op3 + 4 * num_per_op);
			b2 = zq_mm_load_ps(B + num_per_op3 + 8 * num_per_op);
			b3 = zq_mm_load_ps(B + num_per_op3 + 12 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			zq_mm_store_ps(q, c00);
			C[0] += final_sum(q);
			zq_mm_store_ps(q, c01);
			C[1] += final_sum(q);
			zq_mm_store_ps(q, c02);
			C[2] += final_sum(q);
			zq_mm_store_ps(q, c03);
			C[3] += final_sum(q);
			zq_mm_store_ps(q, c10);
			C[4] += final_sum(q);
			zq_mm_store_ps(q, c11);
			C[5] += final_sum(q);
			zq_mm_store_ps(q, c12);
			C[6] += final_sum(q);
			zq_mm_store_ps(q, c13);
			C[7] += final_sum(q);
			zq_mm_store_ps(q, c20);
			C[8] += final_sum(q);
			zq_mm_store_ps(q, c21);
			C[9] += final_sum(q);
			zq_mm_store_ps(q, c22);
			C[10] += final_sum(q);
			zq_mm_store_ps(q, c23);
			C[11] += final_sum(q);
			zq_mm_store_ps(q, c30);
			C[12] += final_sum(q);
			zq_mm_store_ps(q, c31);
			C[13] += final_sum(q);
			zq_mm_store_ps(q, c32);
			C[14] += final_sum(q);
			zq_mm_store_ps(q, c33);
			C[15] += final_sum(q);
		}
		clock_t t2 = clock();
#if defined(_WIN32)
		double time = (t2 - t1)*0.001;
#else
		double time = (t2 - t1)*0.000001;
#endif
		double gflops = 4.0*4.0*4.0*num_per_op*M / (1024.0*1024.0*1024.0)/time;
		printf("time = %.3f s, gflops = %.3f\n", time, gflops);
		for (k = 0; k < 1; k++)
			printf("%.1e ", C[k]);
		printf("\n");
	}
	_aligned_free(A);
	_aligned_free(B);
	free(C);
}

void test_4x4x8_in_cache()
{
	float* A = (float*)_aligned_malloc(4 * 8 * num_per_op * sizeof(float), num_per_op * 8);
	float* B = (float*)_aligned_malloc(4 * 8 * num_per_op * sizeof(float), num_per_op * 8);
	float* C = (float*)malloc(4 * 4 * sizeof(float));
	int i, j, k;
	for (i = 0; i < 4 * 8 * num_per_op; i++)
	{
		A[i] = i;
		B[i] = 1;
	}
	memset(C, 0, sizeof(float) * 16);
#if defined(_WIN32)
	_declspec(align(32)) float q[8];
#else
	ZQ_DECLSPEC_ALIGN32 float q[8];
#endif
	register zq_mm_type a0, a1, a2, a3, b0, b1, b2, b3;
	register zq_mm_type c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33;
	int nIters = 5;
	int M = 2 * 1024 * 1024;
	printf("test 4x4x%d\n", num_per_op * 8);
	for (int i = 0; i < nIters; i++)
	{
		clock_t t1 = clock();
		for (j = 0; j < M; j++)
		{
			a0 = zq_mm_load_ps(A);
			a1 = zq_mm_load_ps(A + 8 * num_per_op);
			a2 = zq_mm_load_ps(A + 16 * num_per_op);
			a3 = zq_mm_load_ps(A + 24 * num_per_op);
			b0 = zq_mm_load_ps(B);
			b1 = zq_mm_load_ps(B + 8 * num_per_op);
			b2 = zq_mm_load_ps(B + 16 * num_per_op);
			b3 = zq_mm_load_ps(B + 24 * num_per_op);
			c00 = zq_mm_mul_ps(a0, b0);
			c01 = zq_mm_mul_ps(a0, b1);
			c02 = zq_mm_mul_ps(a0, b2);
			c03 = zq_mm_mul_ps(a0, b3);
			c10 = zq_mm_mul_ps(a1, b0);
			c11 = zq_mm_mul_ps(a1, b1);
			c12 = zq_mm_mul_ps(a1, b2);
			c13 = zq_mm_mul_ps(a1, b3);
			c20 = zq_mm_mul_ps(a2, b0);
			c21 = zq_mm_mul_ps(a2, b1);
			c22 = zq_mm_mul_ps(a2, b2);
			c23 = zq_mm_mul_ps(a2, b3);
			c30 = zq_mm_mul_ps(a3, b0);
			c31 = zq_mm_mul_ps(a3, b1);
			c32 = zq_mm_mul_ps(a3, b2);
			c33 = zq_mm_mul_ps(a3, b3);
			a0 = zq_mm_load_ps(A + 1 * num_per_op);
			a1 = zq_mm_load_ps(A + 9 * num_per_op);
			a2 = zq_mm_load_ps(A + 17 * num_per_op);
			a3 = zq_mm_load_ps(A + 25 * num_per_op);
			b0 = zq_mm_load_ps(B + 1 * num_per_op);
			b1 = zq_mm_load_ps(B + 9 * num_per_op);
			b2 = zq_mm_load_ps(B + 17 * num_per_op);
			b3 = zq_mm_load_ps(B + 25 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 2 * num_per_op);
			a1 = zq_mm_load_ps(A + 10 * num_per_op);
			a2 = zq_mm_load_ps(A + 18 * num_per_op);
			a3 = zq_mm_load_ps(A + 26 * num_per_op);
			b0 = zq_mm_load_ps(B + 2 * num_per_op);
			b1 = zq_mm_load_ps(B + 10 * num_per_op);
			b2 = zq_mm_load_ps(B + 18 * num_per_op);
			b3 = zq_mm_load_ps(B + 26 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 3 * num_per_op);
			a1 = zq_mm_load_ps(A + 11 * num_per_op);
			a2 = zq_mm_load_ps(A + 19 * num_per_op);
			a3 = zq_mm_load_ps(A + 27 * num_per_op);
			b0 = zq_mm_load_ps(B + 3 * num_per_op);
			b1 = zq_mm_load_ps(B + 11 * num_per_op);
			b2 = zq_mm_load_ps(B + 19 * num_per_op);
			b3 = zq_mm_load_ps(B + 27 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 4 * num_per_op);
			a1 = zq_mm_load_ps(A + 12 * num_per_op);
			a2 = zq_mm_load_ps(A + 20 * num_per_op);
			a3 = zq_mm_load_ps(A + 28 * num_per_op);
			b0 = zq_mm_load_ps(B + 4 * num_per_op);
			b1 = zq_mm_load_ps(B + 12 * num_per_op);
			b2 = zq_mm_load_ps(B + 20 * num_per_op);
			b3 = zq_mm_load_ps(B + 28 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 5 * num_per_op);
			a1 = zq_mm_load_ps(A + 13 * num_per_op);
			a2 = zq_mm_load_ps(A + 21 * num_per_op);
			a3 = zq_mm_load_ps(A + 29 * num_per_op);
			b0 = zq_mm_load_ps(B + 5 * num_per_op);
			b1 = zq_mm_load_ps(B + 13 * num_per_op);
			b2 = zq_mm_load_ps(B + 21 * num_per_op);
			b3 = zq_mm_load_ps(B + 29 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 6 * num_per_op);
			a1 = zq_mm_load_ps(A + 14 * num_per_op);
			a2 = zq_mm_load_ps(A + 22 * num_per_op);
			a3 = zq_mm_load_ps(A + 30 * num_per_op);
			b0 = zq_mm_load_ps(B + 6 * num_per_op);
			b1 = zq_mm_load_ps(B + 14 * num_per_op);
			b2 = zq_mm_load_ps(B + 22 * num_per_op);
			b3 = zq_mm_load_ps(B + 30 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 7 * num_per_op);
			a1 = zq_mm_load_ps(A + 15 * num_per_op);
			a2 = zq_mm_load_ps(A + 23 * num_per_op);
			a3 = zq_mm_load_ps(A + 31 * num_per_op);
			b0 = zq_mm_load_ps(B + 7 * num_per_op);
			b1 = zq_mm_load_ps(B + 15 * num_per_op);
			b2 = zq_mm_load_ps(B + 23 * num_per_op);
			b3 = zq_mm_load_ps(B + 31 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			zq_mm_store_ps(q, c00);
			C[0] += final_sum(q);
			zq_mm_store_ps(q, c01);
			C[1] += final_sum(q);
			zq_mm_store_ps(q, c02);
			C[2] += final_sum(q);
			zq_mm_store_ps(q, c03);
			C[3] += final_sum(q);
			zq_mm_store_ps(q, c10);
			C[4] += final_sum(q);
			zq_mm_store_ps(q, c11);
			C[5] += final_sum(q);
			zq_mm_store_ps(q, c12);
			C[6] += final_sum(q);
			zq_mm_store_ps(q, c13);
			C[7] += final_sum(q);
			zq_mm_store_ps(q, c20);
			C[8] += final_sum(q);
			zq_mm_store_ps(q, c21);
			C[9] += final_sum(q);
			zq_mm_store_ps(q, c22);
			C[10] += final_sum(q);
			zq_mm_store_ps(q, c23);
			C[11] += final_sum(q);
			zq_mm_store_ps(q, c30);
			C[12] += final_sum(q);
			zq_mm_store_ps(q, c31);
			C[13] += final_sum(q);
			zq_mm_store_ps(q, c32);
			C[14] += final_sum(q);
			zq_mm_store_ps(q, c33);
			C[15] += final_sum(q);
		}
		clock_t t2 = clock();
#if defined(_WIN32)
		double time = (t2 - t1)*0.001;
#else
		double time = (t2 - t1)*0.000001;
#endif
		double gflops = 4.0*4.0*8.0*num_per_op*M / (1024.0*1024.0*1024.0) / time;
		printf("time = %.3f s, gflops = %.3f\n", time, gflops);
		for (k = 0; k < 1; k++)
			printf("%.1e ", C[k]);
		printf("\n");
	}
	_aligned_free(A);
	_aligned_free(B);
	free(C);
}

void test_4x4x16_in_cache()
{
	float* A = (float*)_aligned_malloc(4 * 16 * num_per_op * sizeof(float), num_per_op * 16);
	float* B = (float*)_aligned_malloc(4 * 16 * num_per_op * sizeof(float), num_per_op * 16);
	float* C = (float*)malloc(4 * 4 * sizeof(float));
	int i, j, k;
	for (i = 0; i < 4 * 16 * num_per_op; i++)
	{
		A[i] = i;
		B[i] = 1;
	}
	memset(C, 0, sizeof(float) * 16);
#if defined(_WIN32)
	_declspec(align(32)) float q[8];
#else
	ZQ_DECLSPEC_ALIGN32 float q[8];
#endif
	register zq_mm_type a0, a1, a2, a3, b0, b1, b2, b3;
	register zq_mm_type c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33;
	int nIters = 5;
	int M = 2 * 1024 * 1024;
	printf("test 4x4x%d\n", num_per_op * 16);
	for (int i = 0; i < nIters; i++)
	{
		clock_t t1 = clock();
		for (j = 0; j < M; j++)
		{
			a0 = zq_mm_load_ps(A);
			a1 = zq_mm_load_ps(A + 16 * num_per_op);
			a2 = zq_mm_load_ps(A + 32 * num_per_op);
			a3 = zq_mm_load_ps(A + 48 * num_per_op);
			b0 = zq_mm_load_ps(B);
			b1 = zq_mm_load_ps(B + 16 * num_per_op);
			b2 = zq_mm_load_ps(B + 32 * num_per_op);
			b3 = zq_mm_load_ps(B + 48 * num_per_op);
			c00 = zq_mm_mul_ps(a0, b0);
			c01 = zq_mm_mul_ps(a0, b1);
			c02 = zq_mm_mul_ps(a0, b2);
			c03 = zq_mm_mul_ps(a0, b3);
			c10 = zq_mm_mul_ps(a1, b0);
			c11 = zq_mm_mul_ps(a1, b1);
			c12 = zq_mm_mul_ps(a1, b2);
			c13 = zq_mm_mul_ps(a1, b3);
			c20 = zq_mm_mul_ps(a2, b0);
			c21 = zq_mm_mul_ps(a2, b1);
			c22 = zq_mm_mul_ps(a2, b2);
			c23 = zq_mm_mul_ps(a2, b3);
			c30 = zq_mm_mul_ps(a3, b0);
			c31 = zq_mm_mul_ps(a3, b1);
			c32 = zq_mm_mul_ps(a3, b2);
			c33 = zq_mm_mul_ps(a3, b3);
			a0 = zq_mm_load_ps(A + 1 * num_per_op);
			a1 = zq_mm_load_ps(A + 17 * num_per_op);
			a2 = zq_mm_load_ps(A + 33 * num_per_op);
			a3 = zq_mm_load_ps(A + 49 * num_per_op);
			b0 = zq_mm_load_ps(B + 1 * num_per_op);
			b1 = zq_mm_load_ps(B + 17 * num_per_op);
			b2 = zq_mm_load_ps(B + 33 * num_per_op);
			b3 = zq_mm_load_ps(B + 49 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 2 * num_per_op);
			a1 = zq_mm_load_ps(A + 18 * num_per_op);
			a2 = zq_mm_load_ps(A + 34 * num_per_op);
			a3 = zq_mm_load_ps(A + 50 * num_per_op);
			b0 = zq_mm_load_ps(B + 2 * num_per_op);
			b1 = zq_mm_load_ps(B + 18 * num_per_op);
			b2 = zq_mm_load_ps(B + 34 * num_per_op);
			b3 = zq_mm_load_ps(B + 50 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 3 * num_per_op);
			a1 = zq_mm_load_ps(A + 19 * num_per_op);
			a2 = zq_mm_load_ps(A + 35 * num_per_op);
			a3 = zq_mm_load_ps(A + 51 * num_per_op);
			b0 = zq_mm_load_ps(B + 3 * num_per_op);
			b1 = zq_mm_load_ps(B + 19 * num_per_op);
			b2 = zq_mm_load_ps(B + 35 * num_per_op);
			b3 = zq_mm_load_ps(B + 51 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 4 * num_per_op);
			a1 = zq_mm_load_ps(A + 20 * num_per_op);
			a2 = zq_mm_load_ps(A + 36 * num_per_op);
			a3 = zq_mm_load_ps(A + 52 * num_per_op);
			b0 = zq_mm_load_ps(B + 4 * num_per_op);
			b1 = zq_mm_load_ps(B + 20 * num_per_op);
			b2 = zq_mm_load_ps(B + 36 * num_per_op);
			b3 = zq_mm_load_ps(B + 52 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 5 * num_per_op);
			a1 = zq_mm_load_ps(A + 21 * num_per_op);
			a2 = zq_mm_load_ps(A + 37 * num_per_op);
			a3 = zq_mm_load_ps(A + 53 * num_per_op);
			b0 = zq_mm_load_ps(B + 5 * num_per_op);
			b1 = zq_mm_load_ps(B + 21 * num_per_op);
			b2 = zq_mm_load_ps(B + 37 * num_per_op);
			b3 = zq_mm_load_ps(B + 53 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 6 * num_per_op);
			a1 = zq_mm_load_ps(A + 22 * num_per_op);
			a2 = zq_mm_load_ps(A + 38 * num_per_op);
			a3 = zq_mm_load_ps(A + 54 * num_per_op);
			b0 = zq_mm_load_ps(B + 6 * num_per_op);
			b1 = zq_mm_load_ps(B + 22 * num_per_op);
			b2 = zq_mm_load_ps(B + 38 * num_per_op);
			b3 = zq_mm_load_ps(B + 54 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 7 * num_per_op);
			a1 = zq_mm_load_ps(A + 23 * num_per_op);
			a2 = zq_mm_load_ps(A + 39 * num_per_op);
			a3 = zq_mm_load_ps(A + 55 * num_per_op);
			b0 = zq_mm_load_ps(B + 7 * num_per_op);
			b1 = zq_mm_load_ps(B + 23 * num_per_op);
			b2 = zq_mm_load_ps(B + 39 * num_per_op);
			b3 = zq_mm_load_ps(B + 55 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 8 * num_per_op);
			a1 = zq_mm_load_ps(A + 24 * num_per_op);
			a2 = zq_mm_load_ps(A + 40 * num_per_op);
			a3 = zq_mm_load_ps(A + 56 * num_per_op);
			b0 = zq_mm_load_ps(B + 8 * num_per_op);
			b1 = zq_mm_load_ps(B + 24 * num_per_op);
			b2 = zq_mm_load_ps(B + 40 * num_per_op);
			b3 = zq_mm_load_ps(B + 56 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 9 * num_per_op);
			a1 = zq_mm_load_ps(A + 25 * num_per_op);
			a2 = zq_mm_load_ps(A + 41 * num_per_op);
			a3 = zq_mm_load_ps(A + 57 * num_per_op);
			b0 = zq_mm_load_ps(B + 9 * num_per_op);
			b1 = zq_mm_load_ps(B + 25 * num_per_op);
			b2 = zq_mm_load_ps(B + 41 * num_per_op);
			b3 = zq_mm_load_ps(B + 57 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 10 * num_per_op);
			a1 = zq_mm_load_ps(A + 26 * num_per_op);
			a2 = zq_mm_load_ps(A + 42 * num_per_op);
			a3 = zq_mm_load_ps(A + 58 * num_per_op);
			b0 = zq_mm_load_ps(B + 10 * num_per_op);
			b1 = zq_mm_load_ps(B + 26 * num_per_op);
			b2 = zq_mm_load_ps(B + 42 * num_per_op);
			b3 = zq_mm_load_ps(B + 58 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 11 * num_per_op);
			a1 = zq_mm_load_ps(A + 27 * num_per_op);
			a2 = zq_mm_load_ps(A + 43 * num_per_op);
			a3 = zq_mm_load_ps(A + 59 * num_per_op);
			b0 = zq_mm_load_ps(B + 11 * num_per_op);
			b1 = zq_mm_load_ps(B + 27 * num_per_op);
			b2 = zq_mm_load_ps(B + 43 * num_per_op);
			b3 = zq_mm_load_ps(B + 59 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 12 * num_per_op);
			a1 = zq_mm_load_ps(A + 28 * num_per_op);
			a2 = zq_mm_load_ps(A + 44 * num_per_op);
			a3 = zq_mm_load_ps(A + 60 * num_per_op);
			b0 = zq_mm_load_ps(B + 12 * num_per_op);
			b1 = zq_mm_load_ps(B + 28 * num_per_op);
			b2 = zq_mm_load_ps(B + 44 * num_per_op);
			b3 = zq_mm_load_ps(B + 60 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 13 * num_per_op);
			a1 = zq_mm_load_ps(A + 29 * num_per_op);
			a2 = zq_mm_load_ps(A + 45 * num_per_op);
			a3 = zq_mm_load_ps(A + 61 * num_per_op);
			b0 = zq_mm_load_ps(B + 13 * num_per_op);
			b1 = zq_mm_load_ps(B + 29 * num_per_op);
			b2 = zq_mm_load_ps(B + 45 * num_per_op);
			b3 = zq_mm_load_ps(B + 61 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 14 * num_per_op);
			a1 = zq_mm_load_ps(A + 30 * num_per_op);
			a2 = zq_mm_load_ps(A + 46 * num_per_op);
			a3 = zq_mm_load_ps(A + 62 * num_per_op);
			b0 = zq_mm_load_ps(B + 14 * num_per_op);
			b1 = zq_mm_load_ps(B + 30 * num_per_op);
			b2 = zq_mm_load_ps(B + 46 * num_per_op);
			b3 = zq_mm_load_ps(B + 62 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			a0 = zq_mm_load_ps(A + 15 * num_per_op);
			a1 = zq_mm_load_ps(A + 31 * num_per_op);
			a2 = zq_mm_load_ps(A + 47 * num_per_op);
			a3 = zq_mm_load_ps(A + 63 * num_per_op);
			b0 = zq_mm_load_ps(B + 15 * num_per_op);
			b1 = zq_mm_load_ps(B + 31 * num_per_op);
			b2 = zq_mm_load_ps(B + 47 * num_per_op);
			b3 = zq_mm_load_ps(B + 63 * num_per_op);
			c00 = zq_mm_fmadd_ps(a0, b0, c00);
			c01 = zq_mm_fmadd_ps(a0, b1, c01);
			c02 = zq_mm_fmadd_ps(a0, b2, c02);
			c03 = zq_mm_fmadd_ps(a0, b3, c03);
			c10 = zq_mm_fmadd_ps(a1, b0, c10);
			c11 = zq_mm_fmadd_ps(a1, b1, c11);
			c12 = zq_mm_fmadd_ps(a1, b2, c12);
			c13 = zq_mm_fmadd_ps(a1, b3, c13);
			c20 = zq_mm_fmadd_ps(a2, b0, c20);
			c21 = zq_mm_fmadd_ps(a2, b1, c21);
			c22 = zq_mm_fmadd_ps(a2, b2, c22);
			c23 = zq_mm_fmadd_ps(a2, b3, c23);
			c30 = zq_mm_fmadd_ps(a3, b0, c30);
			c31 = zq_mm_fmadd_ps(a3, b1, c31);
			c32 = zq_mm_fmadd_ps(a3, b2, c32);
			c33 = zq_mm_fmadd_ps(a3, b3, c33);
			zq_mm_store_ps(q, c00);
			C[0] += final_sum(q);
			zq_mm_store_ps(q, c01);
			C[1] += final_sum(q);
			zq_mm_store_ps(q, c02);
			C[2] += final_sum(q);
			zq_mm_store_ps(q, c03);
			C[3] += final_sum(q);
			zq_mm_store_ps(q, c10);
			C[4] += final_sum(q);
			zq_mm_store_ps(q, c11);
			C[5] += final_sum(q);
			zq_mm_store_ps(q, c12);
			C[6] += final_sum(q);
			zq_mm_store_ps(q, c13);
			C[7] += final_sum(q);
			zq_mm_store_ps(q, c20);
			C[8] += final_sum(q);
			zq_mm_store_ps(q, c21);
			C[9] += final_sum(q);
			zq_mm_store_ps(q, c22);
			C[10] += final_sum(q);
			zq_mm_store_ps(q, c23);
			C[11] += final_sum(q);
			zq_mm_store_ps(q, c30);
			C[12] += final_sum(q);
			zq_mm_store_ps(q, c31);
			C[13] += final_sum(q);
			zq_mm_store_ps(q, c32);
			C[14] += final_sum(q);
			zq_mm_store_ps(q, c33);
			C[15] += final_sum(q);
		}
		clock_t t2 = clock();
#if defined(_WIN32)
		double time = (t2 - t1)*0.001;
#else
		double time = (t2 - t1)*0.000001;
#endif
		double gflops = 4.0*4.0*16.0*num_per_op*M / (1024.0*1024.0*1024.0)/time;
		printf("time = %.3f s, gflops = %.3f\n", time, gflops);
		for (k = 0; k < 1; k++)
			printf("%.1e ", C[k]);
		printf("\n");
	}
	_aligned_free(A);
	_aligned_free(B);
	free(C);
}

