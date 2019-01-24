#if defined(_WIN32)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ZQ_CNN_CompileConfig.h"

#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps _mm_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm_add_ps(_mm_mul_ps(A, B), C)
#endif

#if ZQ_CNN_USE_FMADD256
#define zq_mm256_fmadd_ps _mm256_fmadd_ps
#else
#define zq_mm256_fmadd_ps(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#endif



#define sum_sse_q (q[0]+q[1]+q[2]+q[3])
#define sum_avx_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])

void MatMul0_ABt(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	int m, n, k;
	float sum;
	const float* A_row_ptr, *B_row_ptr;
	float *C_row_ptr;
	const float* A_c_ptr, *B_c_ptr;
	for (m = 0, A_row_ptr = A, C_row_ptr = C; m < M; m++, A_row_ptr += lda, C_row_ptr += ldc)
	{
		for (n = 0, B_row_ptr = Bt; n < N; n++, B_row_ptr += ldb)
		{
			sum = 0;
			for (k = 0, A_c_ptr = A_row_ptr, B_c_ptr = B_row_ptr; k < K; k++)
				sum += (*(A_c_ptr++)) * (*(B_c_ptr++));
			C_row_ptr[n] = sum;
		}
	}
}

void MatMul1_ABt(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	int m, n, k;
	int ldb4 = ldb * 4;
	float sum0, sum1, sum2, sum3;
	float a_val;
	const float* A_row_ptr, *B_row_ptr0, *B_row_ptr1, *B_row_ptr2, *B_row_ptr3;
	float *C_row_ptr, *C_c_ptr;
	const float* A_c_ptr, *B_c_ptr0, *B_c_ptr1, *B_c_ptr2, *B_c_ptr3;

	for (m = 0, A_row_ptr = A, C_row_ptr = C; m < M; m++, A_row_ptr += lda, C_row_ptr += ldc)
	{
		B_row_ptr0 = Bt;
		B_row_ptr1 = B_row_ptr0 + ldb;
		B_row_ptr2 = B_row_ptr1 + ldb;
		B_row_ptr3 = B_row_ptr2 + ldb;
		C_c_ptr = C_row_ptr;
		for (n = 0; n < N - 4; n += 4, B_row_ptr0 += ldb4, B_row_ptr1 += ldb4, B_row_ptr2 += ldb4, B_row_ptr3 += ldb4)
		{
			sum0 = 0;
			sum1 = 0;
			sum2 = 0;
			sum3 = 0;
			for (k = 0, A_c_ptr = A_row_ptr, B_c_ptr0 = B_row_ptr0,
				B_c_ptr1 = B_row_ptr1, B_c_ptr2 = B_row_ptr2, B_c_ptr3 = B_row_ptr3;
				k < K; k++)
			{
				a_val = *(A_c_ptr++);
				sum0 += a_val * (*(B_c_ptr0++));
				sum1 += a_val * (*(B_c_ptr1++));
				sum2 += a_val * (*(B_c_ptr2++));
				sum3 += a_val * (*(B_c_ptr3++));
			}

			*(C_c_ptr++) = sum0;
			*(C_c_ptr++) = sum1;
			*(C_c_ptr++) = sum2;
			*(C_c_ptr++) = sum3;
		}
		for (; n < N; n++, B_row_ptr0 += ldb)
		{
			sum0 = 0;
			for (k = 0, A_c_ptr = A_row_ptr, B_c_ptr0 = B_row_ptr0;
				k < K; k++)
			{
				a_val = *(A_c_ptr++);
				sum0 += a_val * (*(B_c_ptr0++));
			}

			*(C_c_ptr++) = sum0;
		}
	}
}

void MatMul2_ABt(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	int m, n, k;
	int ldb4 = ldb * 4;
	register float sum0, sum1, sum2, sum3;
	register float a_val;
	const float* A_row_ptr, *B_row_ptr0, *B_row_ptr1, *B_row_ptr2, *B_row_ptr3;
	float *C_row_ptr, *C_c_ptr;
	const float* A_c_ptr, *B_c_ptr0, *B_c_ptr1, *B_c_ptr2, *B_c_ptr3;

	for (m = 0, A_row_ptr = A, C_row_ptr = C; m < M; m++, A_row_ptr += lda, C_row_ptr += ldc)
	{
		B_row_ptr0 = Bt;
		B_row_ptr1 = B_row_ptr0 + ldb;
		B_row_ptr2 = B_row_ptr1 + ldb;
		B_row_ptr3 = B_row_ptr2 + ldb;
		C_c_ptr = C_row_ptr;
		for (n = 0; n < N - 4; n += 4, B_row_ptr0 += ldb4, B_row_ptr1 += ldb4, B_row_ptr2 += ldb4, B_row_ptr3 += ldb4)
		{
			sum0 = 0;
			sum1 = 0;
			sum2 = 0;
			sum3 = 0;
			for (k = 0, A_c_ptr = A_row_ptr, B_c_ptr0 = B_row_ptr0,
				B_c_ptr1 = B_row_ptr1, B_c_ptr2 = B_row_ptr2, B_c_ptr3 = B_row_ptr3;
				k < K; k++)
			{
				a_val = *(A_c_ptr++);
				sum0 += a_val * (*(B_c_ptr0++));
				sum1 += a_val * (*(B_c_ptr1++));
				sum2 += a_val * (*(B_c_ptr2++));
				sum3 += a_val * (*(B_c_ptr3++));
			}

			*(C_c_ptr++) = sum0;
			*(C_c_ptr++) = sum1;
			*(C_c_ptr++) = sum2;
			*(C_c_ptr++) = sum3;
		}
		for (; n < N; n++, B_row_ptr0+=ldb)
		{
			sum0 = 0;
			sum1 = 0;
			sum2 = 0;
			sum3 = 0;
			for (k = 0, A_c_ptr = A_row_ptr, B_c_ptr0 = B_row_ptr0;
				k < K; k++)
			{
				a_val = *(A_c_ptr++);
				sum0 += a_val * (*(B_c_ptr0++));
			}

			*(C_c_ptr++) = sum0;
		}
	}
}

void MatMul3_ABt(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	int m, n, k;
	int ldb4 = ldb * 4;
	register float sum0, sum1, sum2, sum3;
	register float a_val;
	const float* A_row_ptr, *B_row_ptr0, *B_row_ptr1, *B_row_ptr2, *B_row_ptr3;
	float *C_row_ptr, *C_c_ptr;
	const float* A_c_ptr, *B_c_ptr0, *B_c_ptr1, *B_c_ptr2, *B_c_ptr3;

	for (m = 0, A_row_ptr = A, C_row_ptr = C; m < M; m++, A_row_ptr += lda, C_row_ptr += ldc)
	{
		B_row_ptr0 = Bt;
		B_row_ptr1 = B_row_ptr0 + ldb;
		B_row_ptr2 = B_row_ptr1 + ldb;
		B_row_ptr3 = B_row_ptr2 + ldb;
		C_c_ptr = C_row_ptr;
		for (n = 0; n < N - 4; n += 4, B_row_ptr0 += ldb4, B_row_ptr1 += ldb4, B_row_ptr2 += ldb4, B_row_ptr3 += ldb4)
		{
			sum0 = 0;
			sum1 = 0;
			sum2 = 0;
			sum3 = 0;
			for (k = 0, A_c_ptr = A_row_ptr, B_c_ptr0 = B_row_ptr0,
				B_c_ptr1 = B_row_ptr1, B_c_ptr2 = B_row_ptr2, B_c_ptr3 = B_row_ptr3;
				k < K - 4; k += 4)
			{
				a_val = *(A_c_ptr++);
				sum0 += a_val * (*(B_c_ptr0++));
				sum1 += a_val * (*(B_c_ptr1++));
				sum2 += a_val * (*(B_c_ptr2++));
				sum3 += a_val * (*(B_c_ptr3++));
				a_val = *(A_c_ptr++);
				sum0 += a_val * (*(B_c_ptr0++));
				sum1 += a_val * (*(B_c_ptr1++));
				sum2 += a_val * (*(B_c_ptr2++));
				sum3 += a_val * (*(B_c_ptr3++));
				a_val = *(A_c_ptr++);
				sum0 += a_val * (*(B_c_ptr0++));
				sum1 += a_val * (*(B_c_ptr1++));
				sum2 += a_val * (*(B_c_ptr2++));
				sum3 += a_val * (*(B_c_ptr3++));
				a_val = *(A_c_ptr++);
				sum0 += a_val * (*(B_c_ptr0++));
				sum1 += a_val * (*(B_c_ptr1++));
				sum2 += a_val * (*(B_c_ptr2++));
				sum3 += a_val * (*(B_c_ptr3++));
			}

			for (; k < K; k++)
			{
				a_val = *(A_c_ptr++);
				sum0 += a_val * (*(B_c_ptr0++));
				sum1 += a_val * (*(B_c_ptr1++));
				sum2 += a_val * (*(B_c_ptr2++));
				sum3 += a_val * (*(B_c_ptr3++));
			}

			*(C_c_ptr++) = sum0;
			*(C_c_ptr++) = sum1;
			*(C_c_ptr++) = sum2;
			*(C_c_ptr++) = sum3;
		}
		for (; n < N; n++, B_row_ptr0+=ldb)
		{
			sum0 = 0;
			sum1 = 0;
			sum2 = 0;
			sum3 = 0;
			for (k = 0, A_c_ptr = A_row_ptr, B_c_ptr0 = B_row_ptr0;
				k < K - 4; k += 4)
			{
				a_val = *(A_c_ptr++);
				sum0 += a_val * (*(B_c_ptr0++));
				a_val = *(A_c_ptr++);
				sum0 += a_val * (*(B_c_ptr0++));
				a_val = *(A_c_ptr++);
				sum0 += a_val * (*(B_c_ptr0++));
				a_val = *(A_c_ptr++);
				sum0 += a_val * (*(B_c_ptr0++));
			}
			for (; k < K; k++)
			{
				a_val = *(A_c_ptr++);
				sum0 += a_val * (*(B_c_ptr0++));
			}
			*(C_c_ptr++) = sum0;
		}
	}
}

void MatMul4_ABt(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	int m, n, k;
	int lda4 = lda * 4;
	int ldb4 = ldb * 4;
	int ldc4 = ldc * 4;
	register float sum00, sum01, sum02, sum03;
	register float sum10, sum11, sum12, sum13;
	register float sum20, sum21, sum22, sum23;
	register float sum30, sum31, sum32, sum33;
	const float *A_row_ptr0, *A_row_ptr1, *A_row_ptr2, *A_row_ptr3;
	const float *B_row_ptr0, *B_row_ptr1, *B_row_ptr2, *B_row_ptr3;
	float *C_row_ptr0, *C_row_ptr1, *C_row_ptr2, *C_row_ptr3;
	const float* A_c_ptr0, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3;
	const float* B_c_ptr0, *B_c_ptr1, *B_c_ptr2, *B_c_ptr3;
	float* C_c_ptr0, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3;

	A_row_ptr0 = A;
	A_row_ptr1 = A_row_ptr0 + lda;
	A_row_ptr2 = A_row_ptr1 + lda;
	A_row_ptr3 = A_row_ptr2 + lda;
	C_row_ptr0 = C;
	C_row_ptr1 = C_row_ptr0 + ldc;
	C_row_ptr2 = C_row_ptr1 + ldc;
	C_row_ptr3 = C_row_ptr2 + ldc;

	for (m = 0; m < M - 4; m += 4,
		A_row_ptr0 += lda4, A_row_ptr1 += lda4, A_row_ptr2 += lda4, A_row_ptr3 += lda4,
		C_row_ptr0 += ldc4, C_row_ptr1 += ldc4, C_row_ptr2 += ldc4, C_row_ptr3 += ldc4)
	{
		B_row_ptr0 = Bt;
		B_row_ptr1 = B_row_ptr0 + ldb;
		B_row_ptr2 = B_row_ptr1 + ldb;
		B_row_ptr3 = B_row_ptr2 + ldb;
		C_c_ptr0 = C_row_ptr0;
		C_c_ptr1 = C_row_ptr1;
		C_c_ptr2 = C_row_ptr2;
		C_c_ptr3 = C_row_ptr3;
		for (n = 0; n < N - 4; n += 4, B_row_ptr0 += ldb4, B_row_ptr1 += ldb4, B_row_ptr2 += ldb4, B_row_ptr3 += ldb4)
		{
			sum00 = 0;	sum01 = 0;	sum02 = 0;	sum03 = 0;
			sum10 = 0;	sum11 = 0;	sum12 = 0;	sum13 = 0;
			sum20 = 0;	sum21 = 0;	sum22 = 0;	sum23 = 0;
			sum30 = 0;	sum31 = 0;	sum32 = 0;	sum33 = 0;
			for (k = 0, A_c_ptr0 = A_row_ptr0, A_c_ptr1 = A_row_ptr1, A_c_ptr2 = A_row_ptr2, A_c_ptr3 = A_row_ptr3,
				B_c_ptr0 = B_row_ptr0, B_c_ptr1 = B_row_ptr1, B_c_ptr2 = B_row_ptr2, B_c_ptr3 = B_row_ptr3;
				k < K; k++)
			{
				sum00 += *(A_c_ptr0) * (*B_c_ptr0);
				sum01 += *(A_c_ptr0) * (*B_c_ptr1);
				sum02 += *(A_c_ptr0) * (*B_c_ptr2);
				sum03 += *(A_c_ptr0++) * (*B_c_ptr3);
				sum10 += *(A_c_ptr1) * (*B_c_ptr0);
				sum11 += *(A_c_ptr1) * (*B_c_ptr1);
				sum12 += *(A_c_ptr1) * (*B_c_ptr2);
				sum13 += *(A_c_ptr1++) * (*B_c_ptr3);
				sum20 += *(A_c_ptr2) * (*B_c_ptr0);
				sum21 += *(A_c_ptr2) * (*B_c_ptr1);
				sum22 += *(A_c_ptr2) * (*B_c_ptr2);
				sum23 += *(A_c_ptr2++) * (*B_c_ptr3);
				sum30 += *(A_c_ptr3) * (*(B_c_ptr0++));
				sum31 += *(A_c_ptr3) * (*(B_c_ptr1++));
				sum32 += *(A_c_ptr3) * (*(B_c_ptr2++));
				sum33 += *(A_c_ptr3++) * (*(B_c_ptr3++));
			}

			*(C_c_ptr0++) = sum00;
			*(C_c_ptr0++) = sum01;
			*(C_c_ptr0++) = sum02;
			*(C_c_ptr0++) = sum03;
			*(C_c_ptr1++) = sum10;
			*(C_c_ptr1++) = sum11;
			*(C_c_ptr1++) = sum12;
			*(C_c_ptr1++) = sum13;
			*(C_c_ptr2++) = sum20;
			*(C_c_ptr2++) = sum21;
			*(C_c_ptr2++) = sum22;
			*(C_c_ptr2++) = sum23;
			*(C_c_ptr3++) = sum30;
			*(C_c_ptr3++) = sum31;
			*(C_c_ptr3++) = sum32;
			*(C_c_ptr3++) = sum33;
		}
		for (; n < N; n++, B_row_ptr0+=ldb)
		{
			sum00 = 0;	sum10 = 0;	sum20 = 0;	sum30 = 0;
			for (k = 0, A_c_ptr0 = A_row_ptr0, A_c_ptr1 = A_row_ptr1, A_c_ptr2 = A_row_ptr2, A_c_ptr3 = A_row_ptr3,
				B_c_ptr0 = B_row_ptr0;
				k < K; k++)
			{
				sum00 += *(A_c_ptr0++) * (*B_c_ptr0);
				sum10 += *(A_c_ptr1++) * (*B_c_ptr0);
				sum20 += *(A_c_ptr2++) * (*B_c_ptr0);
				sum30 += *(A_c_ptr3++) * (*(B_c_ptr0++));
			}

			*(C_c_ptr0++) = sum00;
			*(C_c_ptr1++) = sum10;
			*(C_c_ptr2++) = sum20;
			*(C_c_ptr3++) = sum30;
		}
	}

	for (; m < M; m++, A_row_ptr0 += lda, C_row_ptr0 += ldc)
	{
		B_row_ptr0 = Bt;
		B_row_ptr1 = B_row_ptr0 + ldb;
		B_row_ptr2 = B_row_ptr1 + ldb;
		B_row_ptr3 = B_row_ptr2 + ldb;
		C_c_ptr0 = C_row_ptr0;
		for (n = 0; n < N - 4; n += 4, B_row_ptr0 += ldb4, B_row_ptr1 += ldb4, B_row_ptr2 += ldb4, B_row_ptr3 += ldb4)
		{
			sum00 = 0;	sum01 = 0;	sum02 = 0;	sum03 = 0;
			for (k = 0, A_c_ptr0 = A_row_ptr0,
				B_c_ptr0 = B_row_ptr0, B_c_ptr1 = B_row_ptr1, B_c_ptr2 = B_row_ptr2, B_c_ptr3 = B_row_ptr3;
				k < K; k++)
			{
				sum00 += *(A_c_ptr0) * (*(B_c_ptr0++));
				sum01 += *(A_c_ptr0) * (*(B_c_ptr1++));
				sum02 += *(A_c_ptr0) * (*(B_c_ptr2++));
				sum03 += *(A_c_ptr0++) * (*(B_c_ptr3++));
			}

			*(C_c_ptr0++) = sum00;
			*(C_c_ptr0++) = sum01;
			*(C_c_ptr0++) = sum02;
			*(C_c_ptr0++) = sum03;
		}
		for (; n < N; n++,B_row_ptr0+=ldb)
		{
			sum00 = 0;
			for (k = 0, A_c_ptr0 = A_row_ptr0,
				B_c_ptr0 = B_row_ptr0;
				k < K; k++)
			{
				sum00 += *(A_c_ptr0++) * (*(B_c_ptr0++));
			}

			*(C_c_ptr0++) = sum00;
		}
	}
}

void MatMul5_ABt(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	int m, n, k;
	int lda4 = lda * 4;
	int ldb4 = ldb * 4;
	int ldc4 = ldc * 4;
	register __m128 a0, a1, a2, a3;
	register __m128 b0, b1, b2, b3;
	register __m128 sum00, sum01, sum02, sum03;
	register __m128 sum10, sum11, sum12, sum13;
	register __m128 sum20, sum21, sum22, sum23;
	register __m128 sum30, sum31, sum32, sum33;
	ZQ_DECLSPEC_ALIGN16 float q[4];
	const float *A_row_ptr0, *A_row_ptr1, *A_row_ptr2, *A_row_ptr3;
	const float *B_row_ptr0, *B_row_ptr1, *B_row_ptr2, *B_row_ptr3;
	float *C_row_ptr0, *C_row_ptr1, *C_row_ptr2, *C_row_ptr3;
	const float* A_c_ptr0, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3;
	const float* B_c_ptr0, *B_c_ptr1, *B_c_ptr2, *B_c_ptr3;
	float* C_c_ptr0, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3;
	const int mm_align_size = 4;
	A_row_ptr0 = A;
	A_row_ptr1 = A_row_ptr0 + lda;
	A_row_ptr2 = A_row_ptr1 + lda;
	A_row_ptr3 = A_row_ptr2 + lda;
	C_row_ptr0 = C;
	C_row_ptr1 = C_row_ptr0 + ldc;
	C_row_ptr2 = C_row_ptr1 + ldc;
	C_row_ptr3 = C_row_ptr2 + ldc;

	for (m = 0; m < M - 4; m += 4,
		A_row_ptr0 += lda4, A_row_ptr1 += lda4, A_row_ptr2 += lda4, A_row_ptr3 += lda4,
		C_row_ptr0 += ldc4, C_row_ptr1 += ldc4, C_row_ptr2 += ldc4, C_row_ptr3 += ldc4)
	{
		B_row_ptr0 = Bt;
		B_row_ptr1 = B_row_ptr0 + ldb;
		B_row_ptr2 = B_row_ptr1 + ldb;
		B_row_ptr3 = B_row_ptr2 + ldb;
		C_c_ptr0 = C_row_ptr0;
		C_c_ptr1 = C_row_ptr1;
		C_c_ptr2 = C_row_ptr2;
		C_c_ptr3 = C_row_ptr3;
		for (n = 0; n < N - 4; n += 4, B_row_ptr0 += ldb4, B_row_ptr1 += ldb4, B_row_ptr2 += ldb4, B_row_ptr3 += ldb4)
		{
			sum00 = _mm_setzero_ps();	sum01 = _mm_setzero_ps();	sum02 = _mm_setzero_ps();	sum03 = _mm_setzero_ps();
			sum10 = _mm_setzero_ps();	sum11 = _mm_setzero_ps();	sum12 = _mm_setzero_ps();	sum13 = _mm_setzero_ps();
			sum20 = _mm_setzero_ps();	sum21 = _mm_setzero_ps();	sum22 = _mm_setzero_ps();	sum23 = _mm_setzero_ps();
			sum30 = _mm_setzero_ps();	sum31 = _mm_setzero_ps();	sum32 = _mm_setzero_ps();	sum33 = _mm_setzero_ps();
			for (k = 0, A_c_ptr0 = A_row_ptr0, A_c_ptr1 = A_row_ptr1, A_c_ptr2 = A_row_ptr2, A_c_ptr3 = A_row_ptr3,
				B_c_ptr0 = B_row_ptr0, B_c_ptr1 = B_row_ptr1, B_c_ptr2 = B_row_ptr2, B_c_ptr3 = B_row_ptr3;
				k < K - mm_align_size; k += mm_align_size)
			{
				a0 = _mm_load_ps(A_c_ptr0);
				a1 = _mm_load_ps(A_c_ptr1);
				a2 = _mm_load_ps(A_c_ptr2);
				a3 = _mm_load_ps(A_c_ptr3);
				b0 = _mm_load_ps(B_c_ptr0);
				b1 = _mm_load_ps(B_c_ptr1);
				b2 = _mm_load_ps(B_c_ptr2);
				b3 = _mm_load_ps(B_c_ptr3);
				sum00 = zq_mm_fmadd_ps(a0, b0, sum00);
				sum01 = zq_mm_fmadd_ps(a0, b1, sum01);
				sum02 = zq_mm_fmadd_ps(a0, b2, sum02);
				sum03 = zq_mm_fmadd_ps(a0, b3, sum03);
				sum10 = zq_mm_fmadd_ps(a1, b0, sum10);
				sum11 = zq_mm_fmadd_ps(a1, b1, sum11);
				sum12 = zq_mm_fmadd_ps(a1, b2, sum12);
				sum13 = zq_mm_fmadd_ps(a1, b3, sum13);
				sum20 = zq_mm_fmadd_ps(a2, b0, sum20);
				sum21 = zq_mm_fmadd_ps(a2, b1, sum21);
				sum22 = zq_mm_fmadd_ps(a2, b2, sum22);
				sum23 = zq_mm_fmadd_ps(a2, b3, sum23);
				sum30 = zq_mm_fmadd_ps(a3, b0, sum30);
				sum31 = zq_mm_fmadd_ps(a3, b1, sum31);
				sum32 = zq_mm_fmadd_ps(a3, b2, sum32);
				sum33 = zq_mm_fmadd_ps(a3, b3, sum33);
				A_c_ptr0 += mm_align_size;
				A_c_ptr1 += mm_align_size;
				A_c_ptr2 += mm_align_size;
				A_c_ptr3 += mm_align_size;
				B_c_ptr0 += mm_align_size;
				B_c_ptr1 += mm_align_size;
				B_c_ptr2 += mm_align_size;
				B_c_ptr3 += mm_align_size;
			}
			_mm_store_ps(q, sum00);
			*(C_c_ptr0) = sum_sse_q;
			_mm_store_ps(q, sum01);
			*(C_c_ptr0 + 1) = sum_sse_q;
			_mm_store_ps(q, sum02);
			*(C_c_ptr0 + 2) = sum_sse_q;
			_mm_store_ps(q, sum03);
			*(C_c_ptr0 + 3) = sum_sse_q;
			_mm_store_ps(q, sum10);
			*(C_c_ptr1) = sum_sse_q;
			_mm_store_ps(q, sum11);
			*(C_c_ptr1 + 1) = sum_sse_q;
			_mm_store_ps(q, sum12);
			*(C_c_ptr1 + 2) = sum_sse_q;
			_mm_store_ps(q, sum13);
			*(C_c_ptr1 + 3) = sum_sse_q;
			_mm_store_ps(q, sum20);
			*(C_c_ptr2) = sum_sse_q;
			_mm_store_ps(q, sum21);
			*(C_c_ptr2 + 1) = sum_sse_q;
			_mm_store_ps(q, sum22);
			*(C_c_ptr2 + 2) = sum_sse_q;
			_mm_store_ps(q, sum23);
			*(C_c_ptr2 + 3) = sum_sse_q;
			_mm_store_ps(q, sum30);
			*(C_c_ptr3) = sum_sse_q;
			_mm_store_ps(q, sum31);
			*(C_c_ptr3 + 1) = sum_sse_q;
			_mm_store_ps(q, sum32);
			*(C_c_ptr3 + 2) = sum_sse_q;
			_mm_store_ps(q, sum33);
			*(C_c_ptr3 + 3) = sum_sse_q;
			for (; k < K; k++)
			{
				*(C_c_ptr0) += *(A_c_ptr0) * (*B_c_ptr0);
				*(C_c_ptr0 + 1) += *(A_c_ptr0) * (*B_c_ptr1);
				*(C_c_ptr0 + 2) += *(A_c_ptr0) * (*B_c_ptr2);
				*(C_c_ptr0 + 3) += *(A_c_ptr0++) * (*B_c_ptr3);
				*(C_c_ptr1) += *(A_c_ptr1) * (*B_c_ptr0);
				*(C_c_ptr1 + 1) += *(A_c_ptr1) * (*B_c_ptr1);
				*(C_c_ptr1 + 2) += *(A_c_ptr1) * (*B_c_ptr2);
				*(C_c_ptr1 + 3) += *(A_c_ptr1++) * (*B_c_ptr3);
				*(C_c_ptr2) += *(A_c_ptr2) * (*B_c_ptr0);
				*(C_c_ptr2 + 1) += *(A_c_ptr2) * (*B_c_ptr1);
				*(C_c_ptr2 + 2) += *(A_c_ptr2) * (*B_c_ptr2);
				*(C_c_ptr2 + 3) += *(A_c_ptr2++) * (*B_c_ptr3);
				*(C_c_ptr3) += *(A_c_ptr3) * (*(B_c_ptr0++));
				*(C_c_ptr3 + 1) += *(A_c_ptr3) * (*(B_c_ptr1++));
				*(C_c_ptr3 + 2) += *(A_c_ptr3) * (*(B_c_ptr2++));
				*(C_c_ptr3 + 3) += *(A_c_ptr3++) * (*(B_c_ptr3++));
			}
			C_c_ptr0 += 4;
			C_c_ptr1 += 4;
			C_c_ptr2 += 4;
			C_c_ptr3 += 4;
		}
		for (; n < N; n++, B_row_ptr0 += ldb)
		{
			sum00 = _mm_setzero_ps();
			sum10 = _mm_setzero_ps();
			sum20 = _mm_setzero_ps();
			sum30 = _mm_setzero_ps();
			for (k = 0, A_c_ptr0 = A_row_ptr0, A_c_ptr1 = A_row_ptr1, A_c_ptr2 = A_row_ptr2, A_c_ptr3 = A_row_ptr3,
				B_c_ptr0 = B_row_ptr0;
				k < K - mm_align_size; k += mm_align_size)
			{
				a0 = _mm_load_ps(A_c_ptr0);
				a1 = _mm_load_ps(A_c_ptr1);
				a2 = _mm_load_ps(A_c_ptr2);
				a3 = _mm_load_ps(A_c_ptr3);
				b0 = _mm_load_ps(B_c_ptr0);
				sum00 = zq_mm_fmadd_ps(a0, b0, sum00);
				sum10 = zq_mm_fmadd_ps(a1, b0, sum10);
				sum20 = zq_mm_fmadd_ps(a2, b0, sum20);
				sum30 = zq_mm_fmadd_ps(a3, b0, sum30);
				A_c_ptr0 += mm_align_size;
				A_c_ptr1 += mm_align_size;
				A_c_ptr2 += mm_align_size;
				A_c_ptr3 += mm_align_size;
				B_c_ptr0 += mm_align_size;
			}
			_mm_store_ps(q, sum00);
			*(C_c_ptr0) = sum_sse_q;
			_mm_store_ps(q, sum10);
			*(C_c_ptr1) = sum_sse_q;
			_mm_store_ps(q, sum20);
			*(C_c_ptr2) = sum_sse_q;
			_mm_store_ps(q, sum30);
			*(C_c_ptr3) = sum_sse_q;

			for (; k < K; k++)
			{
				*(C_c_ptr0) += *(A_c_ptr0++) * (*B_c_ptr0);
				*(C_c_ptr1) += *(A_c_ptr1++) * (*B_c_ptr0);
				*(C_c_ptr2) += *(A_c_ptr2++) * (*B_c_ptr0);
				*(C_c_ptr3) += *(A_c_ptr3++) * (*(B_c_ptr0++));
			}
			C_c_ptr0++;
			C_c_ptr1++;
			C_c_ptr2++;
			C_c_ptr3++;
		}
	}

	for (; m < M; m++, A_row_ptr0 += lda, C_row_ptr0 += ldc)
	{
		B_row_ptr0 = Bt;
		B_row_ptr1 = B_row_ptr0 + ldb;
		B_row_ptr2 = B_row_ptr1 + ldb;
		B_row_ptr3 = B_row_ptr2 + ldb;
		C_c_ptr0 = C_row_ptr0;
		for (n = 0; n < N - 4; n += 4, B_row_ptr0 += ldb4, B_row_ptr1 += ldb4, B_row_ptr2 += ldb4, B_row_ptr3 += ldb4)
		{
			sum00 = _mm_setzero_ps();	sum01 = _mm_setzero_ps();	sum02 = _mm_setzero_ps();	sum03 = _mm_setzero_ps();
			for (k = 0, A_c_ptr0 = A_row_ptr0,
				B_c_ptr0 = B_row_ptr0, B_c_ptr1 = B_row_ptr1, B_c_ptr2 = B_row_ptr2, B_c_ptr3 = B_row_ptr3;
				k < K - mm_align_size; k += mm_align_size)
			{
				a0 = _mm_load_ps(A_c_ptr0);
				b0 = _mm_load_ps(B_c_ptr0);
				b1 = _mm_load_ps(B_c_ptr1);
				b2 = _mm_load_ps(B_c_ptr2);
				b3 = _mm_load_ps(B_c_ptr3);
				sum00 = zq_mm_fmadd_ps(a0, b0, sum00);
				sum01 = zq_mm_fmadd_ps(a0, b1, sum01);
				sum02 = zq_mm_fmadd_ps(a0, b2, sum02);
				sum03 = zq_mm_fmadd_ps(a0, b3, sum03);

				A_c_ptr0 += mm_align_size;
				B_c_ptr0 += mm_align_size;
				B_c_ptr1 += mm_align_size;
				B_c_ptr2 += mm_align_size;
				B_c_ptr3 += mm_align_size;
			}
			_mm_store_ps(q, sum00);
			*(C_c_ptr0) = sum_sse_q;
			_mm_store_ps(q, sum01);
			*(C_c_ptr0 + 1) = sum_sse_q;
			_mm_store_ps(q, sum02);
			*(C_c_ptr0 + 2) = sum_sse_q;
			_mm_store_ps(q, sum03);
			*(C_c_ptr0 + 3) = sum_sse_q;

			for (; k < K; k++)
			{
				*(C_c_ptr0) += *(A_c_ptr0) * (*(B_c_ptr0++));
				*(C_c_ptr0 + 1) += *(A_c_ptr0) * (*(B_c_ptr1++));
				*(C_c_ptr0 + 2) += *(A_c_ptr0) * (*(B_c_ptr2++));
				*(C_c_ptr0 + 3) += *(A_c_ptr0++) * (*(B_c_ptr3++));
			}
			C_c_ptr0 += 4;
		}
		for (; n < N; n++,B_row_ptr0+=ldb)
		{
			sum00 = _mm_setzero_ps();

			for (k = 0, A_c_ptr0 = A_row_ptr0,
				B_c_ptr0 = B_row_ptr0;
				k < K - mm_align_size; k += mm_align_size)
			{
				a0 = _mm_load_ps(A_c_ptr0);
				b0 = _mm_load_ps(B_c_ptr0);
				sum00 = zq_mm_fmadd_ps(a0, b0, sum00);
				A_c_ptr0 += mm_align_size;
				B_c_ptr0 += mm_align_size;
			}
			_mm_store_ps(q, sum00);
			*(C_c_ptr0) = sum_sse_q;

			for (; k < K; k++)
			{
				*(C_c_ptr0) += *(A_c_ptr0++) * (*(B_c_ptr0++));
			}
			C_c_ptr0++;
		}
	}
}


void MatMul6_ABt(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	int m, n, k;
	int lda4 = lda * 4;
	int ldb4 = ldb * 4;
	int ldc4 = ldc * 4;
	register __m256 a0, a1, a2, a3;
	register __m256 b0, b1, b2, b3;
	register __m256 sum00, sum01, sum02, sum03;
	register __m256 sum10, sum11, sum12, sum13;
	register __m256 sum20, sum21, sum22, sum23;
	register __m256 sum30, sum31, sum32, sum33;
	ZQ_DECLSPEC_ALIGN32 float q[8];
	const float *A_row_ptr0, *A_row_ptr1, *A_row_ptr2, *A_row_ptr3;
	const float *B_row_ptr0, *B_row_ptr1, *B_row_ptr2, *B_row_ptr3;
	float *C_row_ptr0, *C_row_ptr1, *C_row_ptr2, *C_row_ptr3;
	const float* A_c_ptr0, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3;
	const float* B_c_ptr0, *B_c_ptr1, *B_c_ptr2, *B_c_ptr3;
	float* C_c_ptr0, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3;
	const int mm_align_size = 8;
	A_row_ptr0 = A;
	A_row_ptr1 = A_row_ptr0 + lda;
	A_row_ptr2 = A_row_ptr1 + lda;
	A_row_ptr3 = A_row_ptr2 + lda;
	C_row_ptr0 = C;
	C_row_ptr1 = C_row_ptr0 + ldc;
	C_row_ptr2 = C_row_ptr1 + ldc;
	C_row_ptr3 = C_row_ptr2 + ldc;

	for (m = 0; m < M - 4; m += 4,
		A_row_ptr0 += lda4, A_row_ptr1 += lda4, A_row_ptr2 += lda4, A_row_ptr3 += lda4,
		C_row_ptr0 += ldc4, C_row_ptr1 += ldc4, C_row_ptr2 += ldc4, C_row_ptr3 += ldc4)
	{
		B_row_ptr0 = Bt;
		B_row_ptr1 = B_row_ptr0 + ldb;
		B_row_ptr2 = B_row_ptr1 + ldb;
		B_row_ptr3 = B_row_ptr2 + ldb;
		C_c_ptr0 = C_row_ptr0;
		C_c_ptr1 = C_row_ptr1;
		C_c_ptr2 = C_row_ptr2;
		C_c_ptr3 = C_row_ptr3;
		for (n = 0; n < N - 4; n += 4, B_row_ptr0 += ldb4, B_row_ptr1 += ldb4, B_row_ptr2 += ldb4, B_row_ptr3 += ldb4)
		{
			sum00 = _mm256_setzero_ps();	sum01 = _mm256_setzero_ps();	sum02 = _mm256_setzero_ps();	sum03 = _mm256_setzero_ps();
			sum10 = _mm256_setzero_ps();	sum11 = _mm256_setzero_ps();	sum12 = _mm256_setzero_ps();	sum13 = _mm256_setzero_ps();
			sum20 = _mm256_setzero_ps();	sum21 = _mm256_setzero_ps();	sum22 = _mm256_setzero_ps();	sum23 = _mm256_setzero_ps();
			sum30 = _mm256_setzero_ps();	sum31 = _mm256_setzero_ps();	sum32 = _mm256_setzero_ps();	sum33 = _mm256_setzero_ps();
			for (k = 0, A_c_ptr0 = A_row_ptr0, A_c_ptr1 = A_row_ptr1, A_c_ptr2 = A_row_ptr2, A_c_ptr3 = A_row_ptr3,
				B_c_ptr0 = B_row_ptr0, B_c_ptr1 = B_row_ptr1, B_c_ptr2 = B_row_ptr2, B_c_ptr3 = B_row_ptr3;
				k < K - mm_align_size; k += mm_align_size)
			{
				a0 = _mm256_load_ps(A_c_ptr0);
				a1 = _mm256_load_ps(A_c_ptr1);
				a2 = _mm256_load_ps(A_c_ptr2);
				a3 = _mm256_load_ps(A_c_ptr3);
				b0 = _mm256_load_ps(B_c_ptr0);
				b1 = _mm256_load_ps(B_c_ptr1);
				b2 = _mm256_load_ps(B_c_ptr2);
				b3 = _mm256_load_ps(B_c_ptr3);
				sum00 = zq_mm256_fmadd_ps(a0, b0, sum00);
				sum01 = zq_mm256_fmadd_ps(a0, b1, sum01);
				sum02 = zq_mm256_fmadd_ps(a0, b2, sum02);
				sum03 = zq_mm256_fmadd_ps(a0, b3, sum03);
				sum10 = zq_mm256_fmadd_ps(a1, b0, sum10);
				sum11 = zq_mm256_fmadd_ps(a1, b1, sum11);
				sum12 = zq_mm256_fmadd_ps(a1, b2, sum12);
				sum13 = zq_mm256_fmadd_ps(a1, b3, sum13);
				sum20 = zq_mm256_fmadd_ps(a2, b0, sum20);
				sum21 = zq_mm256_fmadd_ps(a2, b1, sum21);
				sum22 = zq_mm256_fmadd_ps(a2, b2, sum22);
				sum23 = zq_mm256_fmadd_ps(a2, b3, sum23);
				sum30 = zq_mm256_fmadd_ps(a3, b0, sum30);
				sum31 = zq_mm256_fmadd_ps(a3, b1, sum31);
				sum32 = zq_mm256_fmadd_ps(a3, b2, sum32);
				sum33 = zq_mm256_fmadd_ps(a3, b3, sum33);
				A_c_ptr0 += mm_align_size;
				A_c_ptr1 += mm_align_size;
				A_c_ptr2 += mm_align_size;
				A_c_ptr3 += mm_align_size;
				B_c_ptr0 += mm_align_size;
				B_c_ptr1 += mm_align_size;
				B_c_ptr2 += mm_align_size;
				B_c_ptr3 += mm_align_size;
			}
			_mm256_store_ps(q, sum00);
			*(C_c_ptr0) = sum_avx_q;
			_mm256_store_ps(q, sum01);
			*(C_c_ptr0 + 1) = sum_avx_q;
			_mm256_store_ps(q, sum02);
			*(C_c_ptr0 + 2) = sum_avx_q;
			_mm256_store_ps(q, sum03);
			*(C_c_ptr0 + 3) = sum_avx_q;
			_mm256_store_ps(q, sum10);
			*(C_c_ptr1) = sum_avx_q;
			_mm256_store_ps(q, sum11);
			*(C_c_ptr1 + 1) = sum_avx_q;
			_mm256_store_ps(q, sum12);
			*(C_c_ptr1 + 2) = sum_avx_q;
			_mm256_store_ps(q, sum13);
			*(C_c_ptr1 + 3) = sum_avx_q;
			_mm256_store_ps(q, sum20);
			*(C_c_ptr2) = sum_avx_q;
			_mm256_store_ps(q, sum21);
			*(C_c_ptr2 + 1) = sum_avx_q;
			_mm256_store_ps(q, sum22);
			*(C_c_ptr2 + 2) = sum_avx_q;
			_mm256_store_ps(q, sum23);
			*(C_c_ptr2 + 3) = sum_avx_q;
			_mm256_store_ps(q, sum30);
			*(C_c_ptr3) = sum_avx_q;
			_mm256_store_ps(q, sum31);
			*(C_c_ptr3 + 1) = sum_avx_q;
			_mm256_store_ps(q, sum32);
			*(C_c_ptr3 + 2) = sum_avx_q;
			_mm256_store_ps(q, sum33);
			*(C_c_ptr3 + 3) = sum_avx_q;
			for (; k < K; k++)
			{
				*(C_c_ptr0) += *(A_c_ptr0) * (*B_c_ptr0);
				*(C_c_ptr0 + 1) += *(A_c_ptr0) * (*B_c_ptr1);
				*(C_c_ptr0 + 2) += *(A_c_ptr0) * (*B_c_ptr2);
				*(C_c_ptr0 + 3) += *(A_c_ptr0++) * (*B_c_ptr3);
				*(C_c_ptr1) += *(A_c_ptr1) * (*B_c_ptr0);
				*(C_c_ptr1 + 1) += *(A_c_ptr1) * (*B_c_ptr1);
				*(C_c_ptr1 + 2) += *(A_c_ptr1) * (*B_c_ptr2);
				*(C_c_ptr1 + 3) += *(A_c_ptr1++) * (*B_c_ptr3);
				*(C_c_ptr2) += *(A_c_ptr2) * (*B_c_ptr0);
				*(C_c_ptr2 + 1) += *(A_c_ptr2) * (*B_c_ptr1);
				*(C_c_ptr2 + 2) += *(A_c_ptr2) * (*B_c_ptr2);
				*(C_c_ptr2 + 3) += *(A_c_ptr2++) * (*B_c_ptr3);
				*(C_c_ptr3) += *(A_c_ptr3) * (*(B_c_ptr0++));
				*(C_c_ptr3 + 1) += *(A_c_ptr3) * (*(B_c_ptr1++));
				*(C_c_ptr3 + 2) += *(A_c_ptr3) * (*(B_c_ptr2++));
				*(C_c_ptr3 + 3) += *(A_c_ptr3++) * (*(B_c_ptr3++));
			}
			C_c_ptr0 += 4;
			C_c_ptr1 += 4;
			C_c_ptr2 += 4;
			C_c_ptr3 += 4;
		}
		for (; n < N; n++, B_row_ptr0 += ldb)
		{
			sum00 = _mm256_setzero_ps();
			sum10 = _mm256_setzero_ps();
			sum20 = _mm256_setzero_ps();
			sum30 = _mm256_setzero_ps();
			for (k = 0, A_c_ptr0 = A_row_ptr0, A_c_ptr1 = A_row_ptr1, A_c_ptr2 = A_row_ptr2, A_c_ptr3 = A_row_ptr3,
				B_c_ptr0 = B_row_ptr0;
				k < K - mm_align_size; k += mm_align_size)
			{
				a0 = _mm256_load_ps(A_c_ptr0);
				a1 = _mm256_load_ps(A_c_ptr1);
				a2 = _mm256_load_ps(A_c_ptr2);
				a3 = _mm256_load_ps(A_c_ptr3);
				b0 = _mm256_load_ps(B_c_ptr0);
				sum00 = zq_mm256_fmadd_ps(a0, b0, sum00);
				sum10 = zq_mm256_fmadd_ps(a1, b0, sum10);
				sum20 = zq_mm256_fmadd_ps(a2, b0, sum20);
				sum30 = zq_mm256_fmadd_ps(a3, b0, sum30);
				A_c_ptr0 += mm_align_size;
				A_c_ptr1 += mm_align_size;
				A_c_ptr2 += mm_align_size;
				A_c_ptr3 += mm_align_size;
				B_c_ptr0 += mm_align_size;
			}
			_mm256_store_ps(q, sum00);
			*(C_c_ptr0) = sum_avx_q;
			_mm256_store_ps(q, sum10);
			*(C_c_ptr1) = sum_avx_q;
			_mm256_store_ps(q, sum20);
			*(C_c_ptr2) = sum_avx_q;
			_mm256_store_ps(q, sum30);
			*(C_c_ptr3) = sum_avx_q;

			for (; k < K; k++)
			{
				*(C_c_ptr0) += *(A_c_ptr0++) * (*B_c_ptr0);
				*(C_c_ptr1) += *(A_c_ptr1++) * (*B_c_ptr0);
				*(C_c_ptr2) += *(A_c_ptr2++) * (*B_c_ptr0);
				*(C_c_ptr3) += *(A_c_ptr3++) * (*(B_c_ptr0++));
			}
			C_c_ptr0++;
			C_c_ptr1++;
			C_c_ptr2++;
			C_c_ptr3++;
		}
	}

	for (; m < M; m++, A_row_ptr0 += lda, C_row_ptr0 += ldc)
	{
		B_row_ptr0 = Bt;
		B_row_ptr1 = B_row_ptr0 + ldb;
		B_row_ptr2 = B_row_ptr1 + ldb;
		B_row_ptr3 = B_row_ptr2 + ldb;
		C_c_ptr0 = C_row_ptr0;
		for (n = 0; n < N - 4; n += 4, B_row_ptr0 += ldb4, B_row_ptr1 += ldb4, B_row_ptr2 += ldb4, B_row_ptr3 += ldb4)
		{
			sum00 = _mm256_setzero_ps();	sum01 = _mm256_setzero_ps();	sum02 = _mm256_setzero_ps();	sum03 = _mm256_setzero_ps();
			for (k = 0, A_c_ptr0 = A_row_ptr0,
				B_c_ptr0 = B_row_ptr0, B_c_ptr1 = B_row_ptr1, B_c_ptr2 = B_row_ptr2, B_c_ptr3 = B_row_ptr3;
				k < K - mm_align_size; k += mm_align_size)
			{
				a0 = _mm256_load_ps(A_c_ptr0);
				b0 = _mm256_load_ps(B_c_ptr0);
				b1 = _mm256_load_ps(B_c_ptr1);
				b2 = _mm256_load_ps(B_c_ptr2);
				b3 = _mm256_load_ps(B_c_ptr3);
				sum00 = zq_mm256_fmadd_ps(a0, b0, sum00);
				sum01 = zq_mm256_fmadd_ps(a0, b1, sum01);
				sum02 = zq_mm256_fmadd_ps(a0, b2, sum02);
				sum03 = zq_mm256_fmadd_ps(a0, b3, sum03);

				A_c_ptr0 += mm_align_size;
				B_c_ptr0 += mm_align_size;
				B_c_ptr1 += mm_align_size;
				B_c_ptr2 += mm_align_size;
				B_c_ptr3 += mm_align_size;
			}
			_mm256_store_ps(q, sum00);
			*(C_c_ptr0) = sum_avx_q;
			_mm256_store_ps(q, sum01);
			*(C_c_ptr0 + 1) = sum_avx_q;
			_mm256_store_ps(q, sum02);
			*(C_c_ptr0 + 2) = sum_avx_q;
			_mm256_store_ps(q, sum03);
			*(C_c_ptr0 + 3) = sum_avx_q;

			for (; k < K; k++)
			{
				*(C_c_ptr0) += *(A_c_ptr0) * (*(B_c_ptr0++));
				*(C_c_ptr0 + 1) += *(A_c_ptr0) * (*(B_c_ptr1++));
				*(C_c_ptr0 + 2) += *(A_c_ptr0) * (*(B_c_ptr2++));
				*(C_c_ptr0 + 3) += *(A_c_ptr0++) * (*(B_c_ptr3++));
			}
			C_c_ptr0 += 4;
		}
		for (; n < N; n++, B_row_ptr0 += ldb)
		{
			sum00 = _mm256_setzero_ps();

			for (k = 0, A_c_ptr0 = A_row_ptr0,
				B_c_ptr0 = B_row_ptr0;
				k < K - mm_align_size; k += mm_align_size)
			{
				a0 = _mm256_load_ps(A_c_ptr0);
				b0 = _mm256_load_ps(B_c_ptr0);
				sum00 = zq_mm256_fmadd_ps(a0, b0, sum00);
				A_c_ptr0 += mm_align_size;
				B_c_ptr0 += mm_align_size;
			}
			_mm256_store_ps(q, sum00);
			*(C_c_ptr0) = sum_avx_q;

			for (; k < K; k++)
			{
				*(C_c_ptr0) += *(A_c_ptr0++) * (*(B_c_ptr0++));
			}
			C_c_ptr0++;
		}
	}
}