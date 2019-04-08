//#define __ARM_NEON 1
//#define __aarch64__ 1
#include "mat.h"
#include "convolution_1x1.h"
using namespace ncnn;
void MatMul0_AB(int M, int N, int K, const float* A, const float* B, float* C);
void transpose(float* C, int M, int N);
bool check_value(int M, int N, const float* C1, const float* C2, float thresh, bool show);
int main()
{
	int M = 57*57, N = 128, K = 128;
	Mat A, B, packB, C, bias;
	size_t eltsize = 4;
	A.create(1, M, K, eltsize,0);
	B.create(1, 1, K*N, eltsize,0);
	C.create(1,M, N, eltsize,0);
	bias.create(1, 1, N, eltsize,0);
	float* A_ptr = (float*)(A.data);
	float* B_ptr = (float*)(B.data);
	float* C_ptr = (float*)(C.data);
	float* bias_ptr = (float*)(bias.data);
	float* C2 = (float*)malloc(M*N * sizeof(float));
	
	for (int i = 0; i < M*K; i++)
	{
		A_ptr[i] = i/M;
	}
	for (int i = 0; i < N*K; i++)
	{
		B_ptr[i] = 1;
	}
	memset(bias_ptr, 0, sizeof(float) * N);
	conv1x1s1_sgemm_transform_kernel_neon(B, packB, K, N);
	
	conv1x1s1_sgemm_neon(A, C, packB, bias);

	MatMul0_AB(N, M, K, B_ptr, A_ptr, C2);
	transpose(C2,M,N);
	bool flag = check_value(M, N, C_ptr, C2, 1e-5, true);
	printf("%s\n", flag ? "True" : "False");
	free(C2);
	return 0;
}


bool check_value(int M, int N, const float* C1, const float* C2, float thresh, bool show)
{
	int ldc1 = N;
	int ldc2 = N;
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

void transpose(float* C, int M, int N)
{
	float* tmp = (float*)malloc(M*N * sizeof(float));
	for (int m = 0; m < M; m++)
	{
		for (int n = 0; n < N; n++)
			tmp[m*N + n] = C[n*M + m];
	}
	memcpy(C, tmp, sizeof(float)*M*N);
	free(tmp);
}
void MatMul0_AB(int M, int N, int K, const float* A, const float* B, float* C)
{
	int lda = K;
	int ldb = N;
	int ldc = N;
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