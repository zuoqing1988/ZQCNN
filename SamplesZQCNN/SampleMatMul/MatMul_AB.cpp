#if defined(_WIN32)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define sum_sse_q (q[0]+q[1]+q[2]+q[3])
#define sum_avx_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])

void MatMul0_AB(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc)
{
	int m, n, k;
	float sum;
	const float* A_row_ptr, *B_row_ptr;
	float *C_row_ptr;
	const float* A_c_ptr, *B_c_ptr;
	for (m = 0, A_row_ptr = A, C_row_ptr = C; m < M; m++, A_row_ptr += lda, C_row_ptr += ldc)
	{
		for (n = 0, B_c_ptr = B; n < N; n++, B_c_ptr ++)
		{
			sum = 0;
			for (k = 0, A_c_ptr = A_row_ptr, B_row_ptr = B_c_ptr; k < K; k++, B_row_ptr+=ldb)
				sum += (*(A_c_ptr++)) * (*B_row_ptr);
			C_row_ptr[n] = sum;
		}
	}
}

void MatMul1_AB(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc)
{
	int m, n, k;
	int ldb2 = ldb + ldb;
	int ldb3 = ldb2 + ldb;
	int ldb4 = ldb3 + ldb;
	float sum0, sum1, sum2, sum3;
	float a_val;
	const float* A_row_ptr, *B_row_ptr;
	float *C_row_ptr, *C_c_ptr;
	const float* A_c_ptr, *B_c_ptr;

	for (n = 0, B_c_ptr = B, C_c_ptr = C; n < N - 3; n+=4, B_c_ptr+=4, C_c_ptr+=4)
	{
		for (m = 0, A_row_ptr = A, C_row_ptr = C_c_ptr;
			m < M;
			m++, A_row_ptr += lda, C_row_ptr += ldc)
		{
			sum0 = 0; sum1 = 0; sum2 = 0; sum3 = 0;
			for (k = 0, A_c_ptr = A_row_ptr, B_row_ptr = B_c_ptr; 
				k < K; 
				k++, A_c_ptr++, B_row_ptr+=ldb)
			{
				a_val = *A_c_ptr;
				sum0 += a_val * (*(B_row_ptr));
				sum1 += a_val * (*(B_row_ptr +1));
				sum2 += a_val * (*(B_row_ptr +2));
				sum3 += a_val * (*(B_row_ptr +3));
			}
			*C_row_ptr = sum0;
			*(C_row_ptr + 1) = sum1;
			*(C_row_ptr + 2) = sum2;
			*(C_row_ptr + 3) = sum3;
		}
	}
	for (; n < N; n++, C_c_ptr ++, B_c_ptr++)
	{
		for (m = 0, A_row_ptr = A, C_row_ptr = C_c_ptr;
			m < M;
			m++, A_row_ptr += lda, C_row_ptr += ldc)
		{
			sum0 = 0;
			for (k = 0, A_c_ptr = A_row_ptr, B_row_ptr = B_c_ptr;
				k < K;
				k++, A_c_ptr++, B_row_ptr += ldb)
			{
				a_val = *A_c_ptr;
				sum0 += a_val * (*(B_row_ptr));
			}
			*C_row_ptr = sum0;
		}
	}
}


void MatMul2_AB(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc)
{
	int m, n, k;
	int ldb2 = ldb + ldb;
	int ldb3 = ldb2 + ldb;
	int ldb4 = ldb3 + ldb;
	register float sum0, sum1, sum2, sum3;
	register float a_val;
	const float* A_row_ptr, *B_row_ptr;
	float *C_row_ptr, *C_c_ptr;
	const float* A_c_ptr, *B_c_ptr;

	for (n = 0, B_c_ptr = B, C_c_ptr = C; n < N - 3; n += 4, B_c_ptr += 4, C_c_ptr += 4)
	{
		for (m = 0, A_row_ptr = A, C_row_ptr = C_c_ptr;
			m < M;
			m++, A_row_ptr += lda, C_row_ptr += ldc)
		{
			sum0 = 0; sum1 = 0; sum2 = 0; sum3 = 0;
			for (k = 0, A_c_ptr = A_row_ptr, B_row_ptr = B_c_ptr;
				k < K;
				k++, A_c_ptr++, B_row_ptr += ldb)
			{
				a_val = *A_c_ptr;
				sum0 += a_val * (*(B_row_ptr));
				sum1 += a_val * (*(B_row_ptr + 1));
				sum2 += a_val * (*(B_row_ptr + 2));
				sum3 += a_val * (*(B_row_ptr + 3));
			}
			*C_row_ptr = sum0;
			*(C_row_ptr + 1) = sum1;
			*(C_row_ptr + 2) = sum2;
			*(C_row_ptr + 3) = sum3;
		}
	}
	for (; n < N; n++, C_c_ptr++, B_c_ptr++)
	{
		for (m = 0, A_row_ptr = A, C_row_ptr = C_c_ptr;
			m < M;
			m++, A_row_ptr += lda, C_row_ptr += ldc)
		{
			sum0 = 0;
			for (k = 0, A_c_ptr = A_row_ptr, B_row_ptr = B_c_ptr;
				k < K;
				k++, A_c_ptr++, B_row_ptr += ldb)
			{
				a_val = *A_c_ptr;
				sum0 += a_val * (*(B_row_ptr));
			}
			*C_row_ptr = sum0;
		}
	}
}


void MatMul3_AB(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc)
{
	int m, n, k;
	int lda2 = lda + lda;
	int lda3 = lda2 + lda;
	int lda4 = lda3 + lda;
	int ldb2 = ldb + ldb;
	int ldb3 = ldb2 + ldb;
	int ldb4 = ldb3 + ldb;
	int ldc2 = ldc + ldc;
	int ldc3 = ldc2 + ldc;
	int ldc4 = ldc3 + ldc;
	register float sum0, sum1, sum2, sum3;
	register float a_mk;
	const float* A_row_ptr, *B_row_ptr;
	float *C_row_ptr, *C_c_ptr;
	const float* A_c_ptr, *B_c_ptr;

	for (n = 0, B_c_ptr = B, C_c_ptr = C; n < N - 3; n += 4, B_c_ptr += 4, C_c_ptr += 4)
	{
		for (m = 0, A_row_ptr = A, C_row_ptr = C_c_ptr;
			m < M;
			m++, A_row_ptr += lda, C_row_ptr += ldc)
		{
			sum0 = 0; sum1 = 0; sum2 = 0; sum3 = 0;
			for (k = 0, A_c_ptr = A_row_ptr, B_row_ptr = B_c_ptr;
				k < K-3;
				k+=4)
			{
				a_mk = *(A_c_ptr++);//a(m,k)
				sum0 += a_mk * (*(B_row_ptr));//b(k,n)
				sum1 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum2 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum3 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				B_row_ptr += ldb;
				a_mk = *(A_c_ptr++);//a(m,k)
				sum0 += a_mk * (*(B_row_ptr));//b(k,n)
				sum1 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum2 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum3 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				B_row_ptr += ldb;
				a_mk = *(A_c_ptr++);//a(m,k)
				sum0 += a_mk * (*(B_row_ptr));//b(k,n)
				sum1 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum2 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum3 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				B_row_ptr += ldb;
				a_mk = *(A_c_ptr++);//a(m,k)
				sum0 += a_mk * (*(B_row_ptr));//b(k,n)
				sum1 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum2 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum3 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				B_row_ptr += ldb;
			}
			for (;k < K; k++,B_row_ptr += ldb)
			{
				a_mk = *(A_c_ptr++);//a(m,k)
				sum0 += a_mk * (*(B_row_ptr));//b(k,n)
				sum1 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum2 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum3 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
			}
			*C_row_ptr = sum0;
			*(C_row_ptr + 1) = sum1;
			*(C_row_ptr + 2) = sum2;
			*(C_row_ptr + 3) = sum3;
		}
	}
	for (; n < N; n++, C_c_ptr++, B_c_ptr++)
	{
		for (m = 0, A_row_ptr = A, C_row_ptr = C_c_ptr;
			m < M;
			m++, A_row_ptr += lda, C_row_ptr += ldc)
		{
			sum0 = 0; sum1 = 0; sum2 = 0; sum3 = 0;
			for (k = 0, A_c_ptr = A_row_ptr, B_row_ptr = B_c_ptr;
				k < K - 3;
				k += 4)
			{
				a_mk = *(A_c_ptr++);//a(m,k)
				sum0 += a_mk * (*(B_row_ptr));//b(k,n)
				B_row_ptr += ldb;
				a_mk = *(A_c_ptr++);//a(m,k)
				sum0 += a_mk * (*(B_row_ptr));//b(k,n)
				B_row_ptr += ldb;
				a_mk = *(A_c_ptr++);//a(m,k)
				sum0 += a_mk * (*(B_row_ptr));//b(k,n)
				B_row_ptr += ldb;
				a_mk = *(A_c_ptr++);//a(m,k)
				sum0 += a_mk * (*(B_row_ptr));//b(k,n)
				B_row_ptr += ldb;
			}
			for (; k < K; k++, B_row_ptr += ldb)
			{
				a_mk = *(A_c_ptr++);//a(m,k)
				sum0 += a_mk * (*(B_row_ptr));//b(k,n)
			}
			*C_row_ptr = sum0;
		}
	}
}


void MatMul4_AB(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc)
{
	int m, n, k;
	int lda2 = lda + lda;
	int lda3 = lda2 + lda;
	int lda4 = lda3 + lda;
	int ldb2 = ldb + ldb;
	int ldb3 = ldb2 + ldb;
	int ldb4 = ldb3 + ldb;
	int ldc2 = ldc + ldc;
	int ldc3 = ldc2 + ldc;
	int ldc4 = ldc3 + ldc;
	register float sum00, sum01, sum02, sum03;
	register float sum10, sum11, sum12, sum13;
	register float sum20, sum21, sum22, sum23;
	register float sum30, sum31, sum32, sum33;
	register float a_mk;
	const float* A_row_ptr0, *A_row_ptr1, *A_row_ptr2, *A_row_ptr3, *B_row_ptr;
	float *C_row_ptr0, *C_row_ptr1, *C_row_ptr2, *C_row_ptr3;
	float *C_c_ptr;
	const float* A_c_ptr0, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *B_c_ptr;

	for (n = 0, B_c_ptr = B, C_c_ptr = C; n < N - 3; n += 4, B_c_ptr += 4, C_c_ptr += 4)
	{
		A_row_ptr0 = A;
		A_row_ptr1 = A_row_ptr0 + lda;
		A_row_ptr2 = A_row_ptr1 + lda;
		A_row_ptr3 = A_row_ptr2 + lda;
		C_row_ptr0 = C_c_ptr;
		C_row_ptr1 = C_row_ptr0 + ldc;
		C_row_ptr2 = C_row_ptr1 + ldc;
		C_row_ptr3 = C_row_ptr2 + ldc;

		for (m = 0;
			m < M-3;
			m+=4, A_row_ptr0 += lda4, A_row_ptr1 += lda4, A_row_ptr2 += lda4, A_row_ptr3 += lda4,
			C_row_ptr0 += ldc4, C_row_ptr1 += ldc4, C_row_ptr2 += ldc4, C_row_ptr3 += ldc4)
		{
			sum00 = 0; sum01 = 0; sum02 = 0; sum03 = 0;
			sum10 = 0; sum11 = 0; sum12 = 0; sum13 = 0;
			sum20 = 0; sum21 = 0; sum22 = 0; sum23 = 0;
			sum30 = 0; sum31 = 0; sum32 = 0; sum33 = 0;
			for (k = 0, A_c_ptr0 = A_row_ptr0, A_c_ptr1 = A_row_ptr1, A_c_ptr2 = A_row_ptr2, A_c_ptr3 = A_row_ptr3,
				B_row_ptr = B_c_ptr;
				k < K - 3;
				k += 4)
			{
				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				sum01 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum02 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum03 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				a_mk = *(A_c_ptr1++);//a(m,k)
				sum10 += a_mk * (*(B_row_ptr));//b(k,n)
				sum11 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum12 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum13 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				a_mk = *(A_c_ptr2++);//a(m,k)
				sum20 += a_mk * (*(B_row_ptr));//b(k,n)
				sum21 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum22 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum23 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				a_mk = *(A_c_ptr3++);//a(m,k)
				sum30 += a_mk * (*(B_row_ptr));//b(k,n)
				sum31 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum32 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum33 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				B_row_ptr += ldb;
				
				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				sum01 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum02 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum03 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				a_mk = *(A_c_ptr1++);//a(m,k)
				sum10 += a_mk * (*(B_row_ptr));//b(k,n)
				sum11 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum12 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum13 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				a_mk = *(A_c_ptr2++);//a(m,k)
				sum20 += a_mk * (*(B_row_ptr));//b(k,n)
				sum21 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum22 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum23 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				a_mk = *(A_c_ptr3++);//a(m,k)
				sum30 += a_mk * (*(B_row_ptr));//b(k,n)
				sum31 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum32 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum33 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				B_row_ptr += ldb;

				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				sum01 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum02 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum03 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				a_mk = *(A_c_ptr1++);//a(m,k)
				sum10 += a_mk * (*(B_row_ptr));//b(k,n)
				sum11 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum12 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum13 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				a_mk = *(A_c_ptr2++);//a(m,k)
				sum20 += a_mk * (*(B_row_ptr));//b(k,n)
				sum21 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum22 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum23 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				a_mk = *(A_c_ptr3++);//a(m,k)
				sum30 += a_mk * (*(B_row_ptr));//b(k,n)
				sum31 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum32 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum33 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				B_row_ptr += ldb;

				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				sum01 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum02 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum03 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				a_mk = *(A_c_ptr1++);//a(m,k)
				sum10 += a_mk * (*(B_row_ptr));//b(k,n)
				sum11 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum12 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum13 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				a_mk = *(A_c_ptr2++);//a(m,k)
				sum20 += a_mk * (*(B_row_ptr));//b(k,n)
				sum21 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum22 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum23 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				a_mk = *(A_c_ptr3++);//a(m,k)
				sum30 += a_mk * (*(B_row_ptr));//b(k,n)
				sum31 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum32 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum33 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				B_row_ptr += ldb;
			}
			for (; k < K; k++, B_row_ptr += ldb)
			{
				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				sum01 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum02 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum03 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				a_mk = *(A_c_ptr1++);//a(m,k)
				sum10 += a_mk * (*(B_row_ptr));//b(k,n)
				sum11 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum12 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum13 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				a_mk = *(A_c_ptr2++);//a(m,k)
				sum20 += a_mk * (*(B_row_ptr));//b(k,n)
				sum21 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum22 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum23 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				a_mk = *(A_c_ptr3++);//a(m,k)
				sum30 += a_mk * (*(B_row_ptr));//b(k,n)
				sum31 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum32 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum33 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
			}
			*C_row_ptr0 = sum00;
			*(C_row_ptr0 + 1) = sum01;
			*(C_row_ptr0 + 2) = sum02;
			*(C_row_ptr0 + 3) = sum03;
			*C_row_ptr1 = sum10;
			*(C_row_ptr1 + 1) = sum11;
			*(C_row_ptr1 + 2) = sum12;
			*(C_row_ptr1 + 3) = sum13;
			*C_row_ptr2 = sum20;
			*(C_row_ptr2 + 1) = sum21;
			*(C_row_ptr2 + 2) = sum22;
			*(C_row_ptr2 + 3) = sum23;
			*C_row_ptr3 = sum30;
			*(C_row_ptr3 + 1) = sum31;
			*(C_row_ptr3 + 2) = sum32;
			*(C_row_ptr3 + 3) = sum33;
		}
		for (;m < M;
			m ++, A_row_ptr0 += lda, C_row_ptr0 += ldc)
		{
			sum00 = 0; sum01 = 0; sum02 = 0; sum03 = 0;
			for (k = 0, A_c_ptr0 = A_row_ptr0,
				B_row_ptr = B_c_ptr;
				k < K - 3;
				k += 4)
			{
				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				sum01 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum02 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum03 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				B_row_ptr += ldb;

				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				sum01 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum02 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum03 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				B_row_ptr += ldb;

				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				sum01 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum02 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum03 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				B_row_ptr += ldb;

				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				sum01 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum02 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum03 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
				B_row_ptr += ldb;
			}
			for (; k < K; k++, B_row_ptr += ldb)
			{
				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				sum01 += a_mk * (*(B_row_ptr + 1));//b(k,n+1)
				sum02 += a_mk * (*(B_row_ptr + 2));//b(k,n+2)
				sum03 += a_mk * (*(B_row_ptr + 3));//b(k,n+3)
			}
			*C_row_ptr0 = sum00;
			*(C_row_ptr0 + 1) = sum01;
			*(C_row_ptr0 + 2) = sum02;
			*(C_row_ptr0 + 3) = sum03;
		}
	}
	for (; n < N; n ++, B_c_ptr ++, C_c_ptr ++)
	{
		A_row_ptr0 = A;
		A_row_ptr1 = A_row_ptr0 + lda;
		A_row_ptr2 = A_row_ptr1 + lda;
		A_row_ptr3 = A_row_ptr2 + lda;
		C_row_ptr0 = C_c_ptr;
		C_row_ptr1 = C_row_ptr0 + ldc;
		C_row_ptr2 = C_row_ptr1 + ldc;
		C_row_ptr3 = C_row_ptr2 + ldc;

		for (m = 0;
			m < M - 3;
			m += 4, A_row_ptr0 += lda4, A_row_ptr1 += lda4, A_row_ptr2 += lda4, A_row_ptr3 += lda4,
			C_row_ptr0 += ldc4, C_row_ptr1 += ldc4, C_row_ptr2 += ldc4, C_row_ptr3 += ldc4)
		{
			sum00 = 0;
			sum10 = 0;
			sum20 = 0;
			sum30 = 0;
			for (k = 0, A_c_ptr0 = A_row_ptr0, A_c_ptr1 = A_row_ptr1, A_c_ptr2 = A_row_ptr2, A_c_ptr3 = A_row_ptr3,
				B_row_ptr = B_c_ptr;
				k < K - 3;
				k += 4)
			{
				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				a_mk = *(A_c_ptr1++);//a(m,k)
				sum10 += a_mk * (*(B_row_ptr));//b(k,n)
				a_mk = *(A_c_ptr2++);//a(m,k)
				sum20 += a_mk * (*(B_row_ptr));//b(k,n)
				a_mk = *(A_c_ptr3++);//a(m,k)
				sum30 += a_mk * (*(B_row_ptr));//b(k,n)
				B_row_ptr += ldb;

				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				a_mk = *(A_c_ptr1++);//a(m,k)
				sum10 += a_mk * (*(B_row_ptr));//b(k,n)
				a_mk = *(A_c_ptr2++);//a(m,k)
				sum20 += a_mk * (*(B_row_ptr));//b(k,n)
				a_mk = *(A_c_ptr3++);//a(m,k)
				sum30 += a_mk * (*(B_row_ptr));//b(k,n)
				B_row_ptr += ldb;

				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				a_mk = *(A_c_ptr1++);//a(m,k)
				sum10 += a_mk * (*(B_row_ptr));//b(k,n)
				a_mk = *(A_c_ptr2++);//a(m,k)
				sum20 += a_mk * (*(B_row_ptr));//b(k,n)
				a_mk = *(A_c_ptr3++);//a(m,k)
				sum30 += a_mk * (*(B_row_ptr));//b(k,n)
				B_row_ptr += ldb;

				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				a_mk = *(A_c_ptr1++);//a(m,k)
				sum10 += a_mk * (*(B_row_ptr));//b(k,n)
				a_mk = *(A_c_ptr2++);//a(m,k)
				sum20 += a_mk * (*(B_row_ptr));//b(k,n)
				a_mk = *(A_c_ptr3++);//a(m,k)
				sum30 += a_mk * (*(B_row_ptr));//b(k,n)
				B_row_ptr += ldb;
			}
			for (; k < K; k++, B_row_ptr += ldb)
			{
				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				a_mk = *(A_c_ptr1++);//a(m,k)
				sum10 += a_mk * (*(B_row_ptr));//b(k,n)
				a_mk = *(A_c_ptr2++);//a(m,k)
				sum20 += a_mk * (*(B_row_ptr));//b(k,n)
				a_mk = *(A_c_ptr3++);//a(m,k)
				sum30 += a_mk * (*(B_row_ptr));//b(k,n)
			}
			*C_row_ptr0 = sum00;
			*C_row_ptr1 = sum10;
			*C_row_ptr2 = sum20;
			*C_row_ptr3 = sum30;
		}
		for (; m < M;
			m++, A_row_ptr0 += lda, C_row_ptr0 += ldc)
		{
			sum00 = 0;
			for (k = 0, A_c_ptr0 = A_row_ptr0,
				B_row_ptr = B_c_ptr;
				k < K - 3;
				k += 4)
			{
				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				B_row_ptr += ldb;

				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				B_row_ptr += ldb;

				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				B_row_ptr += ldb;

				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
				B_row_ptr += ldb;
			}
			for (; k < K; k++, B_row_ptr += ldb)
			{
				a_mk = *(A_c_ptr0++);//a(m,k)
				sum00 += a_mk * (*(B_row_ptr));//b(k,n)
			}
			*C_row_ptr0 = sum00;
		}
	}
}
//
//void MatMul5_AB(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc)
//{
//	int m, n, k;
//	int lda2 = lda + lda;
//	int lda3 = lda2 + lda;
//	int lda4 = lda3 + lda;
//	int ldb2 = ldb + ldb;
//	int ldb3 = ldb2 + ldb;
//	int ldb4 = ldb3 + ldb;
//	int ldc2 = ldc + ldc;
//	int ldc3 = ldc2 + ldc;
//	int ldc4 = ldc3 + ldc;
//	register __m128 sum00;
//	register __m128 sum10;
//	register __m128 sum20;
//	register __m128 sum30;
//	register __m128 a_mk,b_kn0,b_kn1,b_kn2,b_kn3;
//	const float* A_row_ptr0, *A_row_ptr1, *A_row_ptr2, *A_row_ptr3, *B_row_ptr;
//	float *C_row_ptr0, *C_row_ptr1, *C_row_ptr2, *C_row_ptr3;
//	float *C_c_ptr;
//	const float* A_c_ptr0, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *B_c_ptr;
//	const int mm_align_size = 4, mm_align_size2 = 8, mm_align_size3 = 12, mm_align_size4 = 16;
//	for (n = 0, B_c_ptr = B, C_c_ptr = C; n < N - 3; n += 4, B_c_ptr += 4, C_c_ptr += 4)
//	{
//		A_row_ptr0 = A;
//		A_row_ptr1 = A_row_ptr0 + lda;
//		A_row_ptr2 = A_row_ptr1 + lda;
//		A_row_ptr3 = A_row_ptr2 + lda;
//		C_row_ptr0 = C_c_ptr;
//		C_row_ptr1 = C_row_ptr0 + ldc;
//		C_row_ptr2 = C_row_ptr1 + ldc;
//		C_row_ptr3 = C_row_ptr2 + ldc;
//
//		for (m = 0;
//			m < M - 3;
//			m += 4, A_row_ptr0 += lda4, A_row_ptr1 += lda4, A_row_ptr2 += lda4, A_row_ptr3 += lda4,
//			C_row_ptr0 += ldc4, C_row_ptr1 += ldc4, C_row_ptr2 += ldc4, C_row_ptr3 += ldc4)
//		{
//			sum00 = _mm_setzero_ps();
//			sum10 = _mm_setzero_ps();
//			sum20 = _mm_setzero_ps();
//			sum30 = _mm_setzero_ps();
//			for (k = 0, A_c_ptr0 = A_row_ptr0, A_c_ptr1 = A_row_ptr1, A_c_ptr2 = A_row_ptr2, A_c_ptr3 = A_row_ptr3,
//				B_row_ptr = B_c_ptr;
//				k < K - mm_align_size4;
//				k += mm_align_size4)
//			{
//				b_kn0 = _mm_load_ps(B_row_ptr);
//				b_kn1 = _mm_load_ps(B_row_ptr + mm_align_size);
//				b_kn2 = _mm_load_ps(B_row_ptr + mm_align_size2);
//				b_kn3 = _mm_load_ps(B_row_ptr + mm_align_size3);
//				a_mk = _mm_load_ps(A_c_ptr0);//a(m,k)
//				A_c_ptr0 += mm_align_size;
//				sum00 = _mm_fmadd_ps(a_mk, b_kn0, sum00);
//				sum01 = _mm_fmadd_ps(a_mk, b_kn1, sum01);
//				sum02 = _mm_fmadd_ps(a_mk, b_kn2, sum02);
//				sum03 = _mm_fmadd_ps(a_mk, b_kn3, sum03);
//				a_mk = _mm_load_ps(A_c_ptr1);//a(m,k)
//				A_c_ptr1 += mm_align_size;
//				sum10 = _mm_fmadd_ps(a_mk, b_kn0, sum10);
//				sum11 = _mm_fmadd_ps(a_mk, b_kn1, sum11);
//				sum12 = _mm_fmadd_ps(a_mk, b_kn2, sum12);
//				sum13 = _mm_fmadd_ps(a_mk, b_kn3, sum13);
//				a_mk = _mm_load_ps(A_c_ptr2);//a(m,k)
//				A_c_ptr2 += mm_align_size;
//				sum20 = _mm_fmadd_ps(a_mk, b_kn0, sum20);
//				sum21 = _mm_fmadd_ps(a_mk, b_kn1, sum21);
//				sum22 = _mm_fmadd_ps(a_mk, b_kn2, sum22);
//				sum23 = _mm_fmadd_ps(a_mk, b_kn3, sum23);
//				a_mk = _mm_load_ps(A_c_ptr3);//a(m,k)
//				A_c_ptr3 += mm_align_size;
//				sum30 = _mm_fmadd_ps(a_mk, b_kn0, sum30);
//				sum31 = _mm_fmadd_ps(a_mk, b_kn1, sum31);
//				sum32 = _mm_fmadd_ps(a_mk, b_kn2, sum32);
//				sum33 = _mm_fmadd_ps(a_mk, b_kn3, sum33);
//				B_row_ptr += ldb;
//				
//				b_kn0 = _mm_load_ps(B_row_ptr);
//				b_kn1 = _mm_load_ps(B_row_ptr + mm_align_size);
//				b_kn2 = _mm_load_ps(B_row_ptr + mm_align_size2);
//				b_kn3 = _mm_load_ps(B_row_ptr + mm_align_size3);
//				a_mk = _mm_load_ps(A_c_ptr0);//a(m,k)
//				A_c_ptr0 += mm_align_size;
//				sum00 = _mm_fmadd_ps(a_mk, b_kn0, sum00);
//				sum01 = _mm_fmadd_ps(a_mk, b_kn1, sum01);
//				sum02 = _mm_fmadd_ps(a_mk, b_kn2, sum02);
//				sum03 = _mm_fmadd_ps(a_mk, b_kn3, sum03);
//				a_mk = _mm_load_ps(A_c_ptr1);//a(m,k)
//				A_c_ptr1 += mm_align_size;
//				sum10 = _mm_fmadd_ps(a_mk, b_kn0, sum10);
//				sum11 = _mm_fmadd_ps(a_mk, b_kn1, sum11);
//				sum12 = _mm_fmadd_ps(a_mk, b_kn2, sum12);
//				sum13 = _mm_fmadd_ps(a_mk, b_kn3, sum13);
//				a_mk = _mm_load_ps(A_c_ptr2);//a(m,k)
//				A_c_ptr2 += mm_align_size;
//				sum20 = _mm_fmadd_ps(a_mk, b_kn0, sum20);
//				sum21 = _mm_fmadd_ps(a_mk, b_kn1, sum21);
//				sum22 = _mm_fmadd_ps(a_mk, b_kn2, sum22);
//				sum23 = _mm_fmadd_ps(a_mk, b_kn3, sum23);
//				a_mk = _mm_load_ps(A_c_ptr3);//a(m,k)
//				A_c_ptr3 += mm_align_size;
//				sum30 = _mm_fmadd_ps(a_mk, b_kn0, sum30);
//				sum31 = _mm_fmadd_ps(a_mk, b_kn1, sum31);
//				sum32 = _mm_fmadd_ps(a_mk, b_kn2, sum32);
//				sum33 = _mm_fmadd_ps(a_mk, b_kn3, sum33);
//				B_row_ptr += ldb;
//
//				b_kn0 = _mm_load_ps(B_row_ptr);
//				b_kn1 = _mm_load_ps(B_row_ptr + mm_align_size);
//				b_kn2 = _mm_load_ps(B_row_ptr + mm_align_size2);
//				b_kn3 = _mm_load_ps(B_row_ptr + mm_align_size3);
//				a_mk = _mm_load_ps(A_c_ptr0);//a(m,k)
//				A_c_ptr0 += mm_align_size;
//				sum00 = _mm_fmadd_ps(a_mk, b_kn0, sum00);
//				sum01 = _mm_fmadd_ps(a_mk, b_kn1, sum01);
//				sum02 = _mm_fmadd_ps(a_mk, b_kn2, sum02);
//				sum03 = _mm_fmadd_ps(a_mk, b_kn3, sum03);
//				a_mk = _mm_load_ps(A_c_ptr1);//a(m,k)
//				A_c_ptr1 += mm_align_size;
//				sum10 = _mm_fmadd_ps(a_mk, b_kn0, sum10);
//				sum11 = _mm_fmadd_ps(a_mk, b_kn1, sum11);
//				sum12 = _mm_fmadd_ps(a_mk, b_kn2, sum12);
//				sum13 = _mm_fmadd_ps(a_mk, b_kn3, sum13);
//				a_mk = _mm_load_ps(A_c_ptr2);//a(m,k)
//				A_c_ptr2 += mm_align_size;
//				sum20 = _mm_fmadd_ps(a_mk, b_kn0, sum20);
//				sum21 = _mm_fmadd_ps(a_mk, b_kn1, sum21);
//				sum22 = _mm_fmadd_ps(a_mk, b_kn2, sum22);
//				sum23 = _mm_fmadd_ps(a_mk, b_kn3, sum23);
//				a_mk = _mm_load_ps(A_c_ptr3);//a(m,k)
//				A_c_ptr3 += mm_align_size;
//				sum30 = _mm_fmadd_ps(a_mk, b_kn0, sum30);
//				sum31 = _mm_fmadd_ps(a_mk, b_kn1, sum31);
//				sum32 = _mm_fmadd_ps(a_mk, b_kn2, sum32);
//				sum33 = _mm_fmadd_ps(a_mk, b_kn3, sum33);
//				B_row_ptr += ldb;
//
//				b_kn0 = _mm_load_ps(B_row_ptr);
//				b_kn1 = _mm_load_ps(B_row_ptr + mm_align_size);
//				b_kn2 = _mm_load_ps(B_row_ptr + mm_align_size2);
//				b_kn3 = _mm_load_ps(B_row_ptr + mm_align_size3);
//				a_mk = _mm_load_ps(A_c_ptr0);//a(m,k)
//				A_c_ptr0 += mm_align_size;
//				sum00 = _mm_fmadd_ps(a_mk, b_kn0, sum00);
//				sum01 = _mm_fmadd_ps(a_mk, b_kn1, sum01);
//				sum02 = _mm_fmadd_ps(a_mk, b_kn2, sum02);
//				sum03 = _mm_fmadd_ps(a_mk, b_kn3, sum03);
//				a_mk = _mm_load_ps(A_c_ptr1);//a(m,k)
//				A_c_ptr1 += mm_align_size;
//				sum10 = _mm_fmadd_ps(a_mk, b_kn0, sum10);
//				sum11 = _mm_fmadd_ps(a_mk, b_kn1, sum11);
//				sum12 = _mm_fmadd_ps(a_mk, b_kn2, sum12);
//				sum13 = _mm_fmadd_ps(a_mk, b_kn3, sum13);
//				a_mk = _mm_load_ps(A_c_ptr2);//a(m,k)
//				A_c_ptr2 += mm_align_size;
//				sum20 = _mm_fmadd_ps(a_mk, b_kn0, sum20);
//				sum21 = _mm_fmadd_ps(a_mk, b_kn1, sum21);
//				sum22 = _mm_fmadd_ps(a_mk, b_kn2, sum22);
//				sum23 = _mm_fmadd_ps(a_mk, b_kn3, sum23);
//				a_mk = _mm_load_ps(A_c_ptr3);//a(m,k)
//				A_c_ptr3 += mm_align_size;
//				sum30 = _mm_fmadd_ps(a_mk, b_kn0, sum30);
//				sum31 = _mm_fmadd_ps(a_mk, b_kn1, sum31);
//				sum32 = _mm_fmadd_ps(a_mk, b_kn2, sum32);
//				sum33 = _mm_fmadd_ps(a_mk, b_kn3, sum33);
//				B_row_ptr += ldb;
//			}
//			for (; k < K; k++, B_row_ptr += ldb)
//			{
//				b_kn0 = _mm_load_ps(B_row_ptr);
//				b_kn1 = _mm_load_ps(B_row_ptr + mm_align_size);
//				b_kn2 = _mm_load_ps(B_row_ptr + mm_align_size2);
//				b_kn3 = _mm_load_ps(B_row_ptr + mm_align_size3);
//				a_mk = _mm_load_ps(A_c_ptr0);//a(m,k)
//				A_c_ptr0 += mm_align_size;
//				sum00 = _mm_fmadd_ps(a_mk, b_kn0, sum00);
//				sum01 = _mm_fmadd_ps(a_mk, b_kn1, sum01);
//				sum02 = _mm_fmadd_ps(a_mk, b_kn2, sum02);
//				sum03 = _mm_fmadd_ps(a_mk, b_kn3, sum03);
//				a_mk = _mm_load_ps(A_c_ptr1);//a(m,k)
//				A_c_ptr1 += mm_align_size;
//				sum10 = _mm_fmadd_ps(a_mk, b_kn0, sum10);
//				sum11 = _mm_fmadd_ps(a_mk, b_kn1, sum11);
//				sum12 = _mm_fmadd_ps(a_mk, b_kn2, sum12);
//				sum13 = _mm_fmadd_ps(a_mk, b_kn3, sum13);
//				a_mk = _mm_load_ps(A_c_ptr2);//a(m,k)
//				A_c_ptr2 += mm_align_size;
//				sum20 = _mm_fmadd_ps(a_mk, b_kn0, sum20);
//				sum21 = _mm_fmadd_ps(a_mk, b_kn1, sum21);
//				sum22 = _mm_fmadd_ps(a_mk, b_kn2, sum22);
//				sum23 = _mm_fmadd_ps(a_mk, b_kn3, sum23);
//				a_mk = _mm_load_ps(A_c_ptr3);//a(m,k)
//				A_c_ptr3 += mm_align_size;
//				sum30 = _mm_fmadd_ps(a_mk, b_kn0, sum30);
//				sum31 = _mm_fmadd_ps(a_mk, b_kn1, sum31);
//				sum32 = _mm_fmadd_ps(a_mk, b_kn2, sum32);
//				sum33 = _mm_fmadd_ps(a_mk, b_kn3, sum33);
//			}
//			_mm_store_ps(C_row_ptr0, sum00);
//			_mm_store_ps(C_row_ptr0+mm_align_size, sum01);
//			_mm_store_ps(C_row_ptr0+mm_align_size2, sum02);
//			_mm_store_ps(C_row_ptr0+mm_align_size3, sum03);
//			_mm_store_ps(C_row_ptr1, sum10);
//			_mm_store_ps(C_row_ptr1 + mm_align_size, sum11);
//			_mm_store_ps(C_row_ptr1 + mm_align_size2, sum12);
//			_mm_store_ps(C_row_ptr1 + mm_align_size3, sum13);
//			_mm_store_ps(C_row_ptr2, sum20);
//			_mm_store_ps(C_row_ptr2 + mm_align_size, sum21);
//			_mm_store_ps(C_row_ptr2 + mm_align_size2, sum22);
//			_mm_store_ps(C_row_ptr2 + mm_align_size3, sum23);
//			_mm_store_ps(C_row_ptr3, sum30);
//			_mm_store_ps(C_row_ptr3 + mm_align_size, sum31);
//			_mm_store_ps(C_row_ptr3 + mm_align_size2, sum32);
//			_mm_store_ps(C_row_ptr3 + mm_align_size3, sum33);
//		}
//		for (; m < M;
//			m++, A_row_ptr0 += lda, C_row_ptr0 += ldc)
//		{
//			sum00 = _mm_setzero_ps(); sum01 = _mm_setzero_ps(); sum02 = _mm_setzero_ps(); sum03 = _mm_setzero_ps();
//			for (k = 0, A_c_ptr0 = A_row_ptr0,
//				B_row_ptr = B_c_ptr;
//				k < K - mm_align_size4;
//				k += mm_align_size4)
//			{
//				b_kn0 = _mm_load_ps(B_row_ptr);
//				b_kn1 = _mm_load_ps(B_row_ptr + mm_align_size);
//				b_kn2 = _mm_load_ps(B_row_ptr + mm_align_size2);
//				b_kn3 = _mm_load_ps(B_row_ptr + mm_align_size3);
//				a_mk = _mm_load_ps(A_c_ptr0);//a(m,k)
//				A_c_ptr0 += mm_align_size;
//				sum00 = _mm_fmadd_ps(a_mk, b_kn0, sum00);
//				sum01 = _mm_fmadd_ps(a_mk, b_kn1, sum01);
//				sum02 = _mm_fmadd_ps(a_mk, b_kn2, sum02);
//				sum03 = _mm_fmadd_ps(a_mk, b_kn3, sum03);
//				B_row_ptr += ldb;
//
//				b_kn0 = _mm_load_ps(B_row_ptr);
//				b_kn1 = _mm_load_ps(B_row_ptr + mm_align_size);
//				b_kn2 = _mm_load_ps(B_row_ptr + mm_align_size2);
//				b_kn3 = _mm_load_ps(B_row_ptr + mm_align_size3);
//				a_mk = _mm_load_ps(A_c_ptr0);//a(m,k)
//				A_c_ptr0 += mm_align_size;
//				sum00 = _mm_fmadd_ps(a_mk, b_kn0, sum00);
//				sum01 = _mm_fmadd_ps(a_mk, b_kn1, sum01);
//				sum02 = _mm_fmadd_ps(a_mk, b_kn2, sum02);
//				sum03 = _mm_fmadd_ps(a_mk, b_kn3, sum03);
//				B_row_ptr += ldb;
//
//				b_kn0 = _mm_load_ps(B_row_ptr);
//				b_kn1 = _mm_load_ps(B_row_ptr + mm_align_size);
//				b_kn2 = _mm_load_ps(B_row_ptr + mm_align_size2);
//				b_kn3 = _mm_load_ps(B_row_ptr + mm_align_size3);
//				a_mk = _mm_load_ps(A_c_ptr0);//a(m,k)
//				A_c_ptr0 += mm_align_size;
//				sum00 = _mm_fmadd_ps(a_mk, b_kn0, sum00);
//				sum01 = _mm_fmadd_ps(a_mk, b_kn1, sum01);
//				sum02 = _mm_fmadd_ps(a_mk, b_kn2, sum02);
//				sum03 = _mm_fmadd_ps(a_mk, b_kn3, sum03);
//				B_row_ptr += ldb;
//
//				b_kn0 = _mm_load_ps(B_row_ptr);
//				b_kn1 = _mm_load_ps(B_row_ptr + mm_align_size);
//				b_kn2 = _mm_load_ps(B_row_ptr + mm_align_size2);
//				b_kn3 = _mm_load_ps(B_row_ptr + mm_align_size3);
//				a_mk = _mm_load_ps(A_c_ptr0);//a(m,k)
//				A_c_ptr0 += mm_align_size;
//				sum00 = _mm_fmadd_ps(a_mk, b_kn0, sum00);
//				sum01 = _mm_fmadd_ps(a_mk, b_kn1, sum01);
//				sum02 = _mm_fmadd_ps(a_mk, b_kn2, sum02);
//				sum03 = _mm_fmadd_ps(a_mk, b_kn3, sum03);
//				B_row_ptr += ldb;
//			}
//			_mm_store_ps(C_row_ptr0, sum00);
//			_mm_store_ps(C_row_ptr0 + mm_align_size, sum01);
//			_mm_store_ps(C_row_ptr0 + mm_align_size2, sum02);
//			_mm_store_ps(C_row_ptr0 + mm_align_size3, sum03);
//			for (; k < K; k++, B_row_ptr += ldb)
//			{
//				*C_row_ptr0 += *(A_c_ptr0++) * (*(B_row_ptr));//b(k,n)
//			}
//		}
//	}
//	for (; n < N; n++, B_c_ptr++, C_c_ptr++)
//	{
//		A_row_ptr0 = A;
//		A_row_ptr1 = A_row_ptr0 + lda;
//		A_row_ptr2 = A_row_ptr1 + lda;
//		A_row_ptr3 = A_row_ptr2 + lda;
//		C_row_ptr0 = C_c_ptr;
//		C_row_ptr1 = C_row_ptr0 + ldc;
//		C_row_ptr2 = C_row_ptr1 + ldc;
//		C_row_ptr3 = C_row_ptr2 + ldc;
//		
//		for (m = 0;
//			m < M - 3;
//			m += 4, A_row_ptr0 += lda4, A_row_ptr1 += lda4, A_row_ptr2 += lda4, A_row_ptr3 += lda4,
//			C_row_ptr0 += ldc4, C_row_ptr1 += ldc4, C_row_ptr2 += ldc4, C_row_ptr3 += ldc4)
//		{
//			sum00 = _mm_setzero_ps();
//			sum10 = _mm_setzero_ps();
//			sum20 = _mm_setzero_ps();
//			sum30 = _mm_setzero_ps();
//			for (k = 0, A_c_ptr0 = A_row_ptr0, A_c_ptr1 = A_row_ptr1, A_c_ptr2 = A_row_ptr2, A_c_ptr3 = A_row_ptr3,
//				B_row_ptr = B_c_ptr;
//				k < K - mm_align_size4;
//				k += mm_align_size4)
//			{
//				b_kn0 = _mm_load_ps(B_row_ptr);
//				a_mk = _mm_load_ps(A_c_ptr0);//a(m,k)
//				A_c_ptr0 += mm_align_size;
//				sum00 = _mm_fmadd_ps(a_mk, b_kn0, sum00);
//				a_mk = _mm_load_ps(A_c_ptr1);//a(m,k)
//				A_c_ptr1 += mm_align_size;
//				sum10 = _mm_fmadd_ps(a_mk, b_kn0, sum10);
//				a_mk = _mm_load_ps(A_c_ptr2);//a(m,k)
//				A_c_ptr2 += mm_align_size;
//				sum20 = _mm_fmadd_ps(a_mk, b_kn0, sum20);
//				a_mk = _mm_load_ps(A_c_ptr3);//a(m,k)
//				A_c_ptr3 += mm_align_size;
//				sum30 = _mm_fmadd_ps(a_mk, b_kn0, sum30);
//				B_row_ptr += ldb;
//
//				b_kn0 = _mm_load_ps(B_row_ptr);
//				a_mk = _mm_load_ps(A_c_ptr0);//a(m,k)
//				A_c_ptr0 += mm_align_size;
//				sum00 = _mm_fmadd_ps(a_mk, b_kn0, sum00);
//				a_mk = _mm_load_ps(A_c_ptr1);//a(m,k)
//				A_c_ptr1 += mm_align_size;
//				sum10 = _mm_fmadd_ps(a_mk, b_kn0, sum10);
//				a_mk = _mm_load_ps(A_c_ptr2);//a(m,k)
//				A_c_ptr2 += mm_align_size;
//				sum20 = _mm_fmadd_ps(a_mk, b_kn0, sum20);
//				a_mk = _mm_load_ps(A_c_ptr3);//a(m,k)
//				A_c_ptr3 += mm_align_size;
//				sum30 = _mm_fmadd_ps(a_mk, b_kn0, sum30);
//				B_row_ptr += ldb;
//
//				b_kn0 = _mm_load_ps(B_row_ptr);
//				a_mk = _mm_load_ps(A_c_ptr0);//a(m,k)
//				A_c_ptr0 += mm_align_size;
//				sum00 = _mm_fmadd_ps(a_mk, b_kn0, sum00);
//				a_mk = _mm_load_ps(A_c_ptr1);//a(m,k)
//				A_c_ptr1 += mm_align_size;
//				sum10 = _mm_fmadd_ps(a_mk, b_kn0, sum10);
//				a_mk = _mm_load_ps(A_c_ptr2);//a(m,k)
//				A_c_ptr2 += mm_align_size;
//				sum20 = _mm_fmadd_ps(a_mk, b_kn0, sum20);
//				a_mk = _mm_load_ps(A_c_ptr3);//a(m,k)
//				A_c_ptr3 += mm_align_size;
//				sum30 = _mm_fmadd_ps(a_mk, b_kn0, sum30);
//				B_row_ptr += ldb;
//
//				b_kn0 = _mm_load_ps(B_row_ptr);
//				a_mk = _mm_load_ps(A_c_ptr0);//a(m,k)
//				A_c_ptr0 += mm_align_size;
//				sum00 = _mm_fmadd_ps(a_mk, b_kn0, sum00);
//				a_mk = _mm_load_ps(A_c_ptr1);//a(m,k)
//				A_c_ptr1 += mm_align_size;
//				sum10 = _mm_fmadd_ps(a_mk, b_kn0, sum10);
//				a_mk = _mm_load_ps(A_c_ptr2);//a(m,k)
//				A_c_ptr2 += mm_align_size;
//				sum20 = _mm_fmadd_ps(a_mk, b_kn0, sum20);
//				a_mk = _mm_load_ps(A_c_ptr3);//a(m,k)
//				A_c_ptr3 += mm_align_size;
//				sum30 = _mm_fmadd_ps(a_mk, b_kn0, sum30);
//				B_row_ptr += ldb;
//			}
//			for (; k < K; k++, B_row_ptr += ldb)
//			{
//				b_kn0 = _mm_load_ps(B_row_ptr);
//				a_mk = _mm_load_ps(A_c_ptr0);//a(m,k)
//				A_c_ptr0 += mm_align_size;
//				sum00 = _mm_fmadd_ps(a_mk, b_kn0, sum00);
//				a_mk = _mm_load_ps(A_c_ptr1);//a(m,k)
//				A_c_ptr1 += mm_align_size;
//				sum10 = _mm_fmadd_ps(a_mk, b_kn0, sum10);
//				a_mk = _mm_load_ps(A_c_ptr2);//a(m,k)
//				A_c_ptr2 += mm_align_size;
//				sum20 = _mm_fmadd_ps(a_mk, b_kn0, sum20);
//				a_mk = _mm_load_ps(A_c_ptr3);//a(m,k)
//				A_c_ptr3 += mm_align_size;
//				sum30 = _mm_fmadd_ps(a_mk, b_kn0, sum30);
//			}
//			_mm_store_ps(C_row_ptr0, sum00);
//			_mm_store_ps(C_row_ptr1, sum10);
//			_mm_store_ps(C_row_ptr2, sum20);
//			_mm_store_ps(C_row_ptr3, sum30);
//			
//		}
//		for (; m < M;
//			m++, A_row_ptr0 += lda, C_row_ptr0 += ldc)
//		{
//			sum00 = _mm_setzero_ps();
//			for (k = 0, A_c_ptr0 = A_row_ptr0,
//				B_row_ptr = B_c_ptr;
//				k < K - mm_align_size4;
//				k += mm_align_size4)
//			{
//				b_kn0 = _mm_load_ps(B_row_ptr);
//				a_mk = _mm_load_ps(A_c_ptr0);//a(m,k)
//				A_c_ptr0 += mm_align_size;
//				sum00 = _mm_fmadd_ps(a_mk, b_kn0, sum00);
//				B_row_ptr += ldb;
//
//				b_kn0 = _mm_load_ps(B_row_ptr);
//				a_mk = _mm_load_ps(A_c_ptr0);//a(m,k)
//				A_c_ptr0 += mm_align_size;
//				sum00 = _mm_fmadd_ps(a_mk, b_kn0, sum00);
//				B_row_ptr += ldb;
//
//				b_kn0 = _mm_load_ps(B_row_ptr);
//				a_mk = _mm_load_ps(A_c_ptr0);//a(m,k)
//				A_c_ptr0 += mm_align_size;
//				sum00 = _mm_fmadd_ps(a_mk, b_kn0, sum00);
//				B_row_ptr += ldb;
//
//				b_kn0 = _mm_load_ps(B_row_ptr);
//				a_mk = _mm_load_ps(A_c_ptr0);//a(m,k)
//				A_c_ptr0 += mm_align_size;
//				sum00 = _mm_fmadd_ps(a_mk, b_kn0, sum00);
//				B_row_ptr += ldb;
//			}
//			_mm_store_ps(C_row_ptr0, sum00);
//			for (; k < K; k++, B_row_ptr += ldb)
//			{
//				*C_row_ptr0 += *(A_c_ptr0++) * (*(B_row_ptr));//b(k,n)
//			}
//		}
//	}
//}


/*void MatMul6_ABt(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)

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
				sum00 = _mm256_fmadd_ps(a0, b0, sum00);
				sum01 = _mm256_fmadd_ps(a0, b1, sum01);
				sum02 = _mm256_fmadd_ps(a0, b2, sum02);
				sum03 = _mm256_fmadd_ps(a0, b3, sum03);
				sum10 = _mm256_fmadd_ps(a1, b0, sum10);
				sum11 = _mm256_fmadd_ps(a1, b1, sum11);
				sum12 = _mm256_fmadd_ps(a1, b2, sum12);
				sum13 = _mm256_fmadd_ps(a1, b3, sum13);
				sum20 = _mm256_fmadd_ps(a2, b0, sum20);
				sum21 = _mm256_fmadd_ps(a2, b1, sum21);
				sum22 = _mm256_fmadd_ps(a2, b2, sum22);
				sum23 = _mm256_fmadd_ps(a2, b3, sum23);
				sum30 = _mm256_fmadd_ps(a3, b0, sum30);
				sum31 = _mm256_fmadd_ps(a3, b1, sum31);
				sum32 = _mm256_fmadd_ps(a3, b2, sum32);
				sum33 = _mm256_fmadd_ps(a3, b3, sum33);
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
				sum00 = _mm256_fmadd_ps(a0, b0, sum00);
				sum10 = _mm256_fmadd_ps(a1, b0, sum10);
				sum20 = _mm256_fmadd_ps(a2, b0, sum20);
				sum30 = _mm256_fmadd_ps(a3, b0, sum30);
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
			*(C_c_ptr1) = sum_avx_q;
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
				sum00 = _mm256_fmadd_ps(a0, b0, sum00);
				sum01 = _mm256_fmadd_ps(a0, b1, sum01);
				sum02 = _mm256_fmadd_ps(a0, b2, sum02);
				sum03 = _mm256_fmadd_ps(a0, b3, sum03);

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
				*(C_c_ptr0) += *(A_c_ptr0) * (*B_c_ptr0);
				*(C_c_ptr0 + 1) += *(A_c_ptr0) * (*B_c_ptr1);
				*(C_c_ptr0 + 2) += *(A_c_ptr0) * (*B_c_ptr2);
				*(C_c_ptr0 + 3) += *(A_c_ptr0++) * (*B_c_ptr3);
			}
			C_c_ptr0 += 4;
		}
		for (; n < N; n++)
		{
			sum00 = _mm256_setzero_ps();

			for (k = 0, A_c_ptr0 = A_row_ptr0,
				B_c_ptr0 = B_row_ptr0;
				k < K - mm_align_size; k += mm_align_size)
			{
				a0 = _mm256_load_ps(A_c_ptr0);
				b0 = _mm256_load_ps(B_c_ptr0);
				sum00 = _mm256_fmadd_ps(a0, b0, sum00);
				A_c_ptr0 += mm_align_size;
				B_c_ptr0 += mm_align_size;
			}
			_mm256_store_ps(q, sum00);
			*(C_c_ptr0) = sum_sse_q;

			for (; k < K; k++)
			{
				*(C_c_ptr0) += *(A_c_ptr0++) * (*(B_c_ptr0++));
			}
			C_c_ptr0++;
		}
	}
}

*/