#include <mmintrin.h> //MMX  
#include <xmmintrin.h> //SSE(include mmintrin.h)  
#include <emmintrin.h> //SSE2(include xmmintrin.h)  
#include <pmmintrin.h> //SSE3(include emmintrin.h)  
#include <tmmintrin.h>//SSSE3(include pmmintrin.h)  
#include <smmintrin.h>//SSE4.1(include tmmintrin.h)  
#include <nmmintrin.h>//SSE4.2(include smmintrin.h)  
#include <wmmintrin.h>//AES(include nmmintrin.h)  
#include <immintrin.h>//AVX(include wmmintrin.h)  
#include <intrin.h>//(include immintrin.h)  
#include <malloc.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>
#include <stdio.h>
#include <float.h>
#include "ZQ_CNN_Tensor4D.h"

using namespace ZQ;

double _test_gemm_value()
{
	float A[] = {
		1,2,3,
		4,5,6
	};
	float BT[] = {
		2, 1,3,
		4,5,6
	};
	float C[4];
	for (int it = 0; it < 10; it++)
	{
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 2, 2, 3, 1, A, 3, BT, 3, 0, C, 2);
		printf("%f %f %f %f\n", C[0], C[1], C[2], C[3]);
	}
	return 0;

}
double _test_gemv(int M, int N, int K, int iters = 1000)
{
	/*
	K = (K + 63) >> 6 << 6;
	N = (N + 7) >> 3 << 3;*/
	float* A = (float*)_aligned_malloc(M* K * sizeof(float), 32);
	float* B = (float*)_aligned_malloc(K*N * sizeof(float), 32);
	float* C = (float*)_aligned_malloc(M* N * sizeof(float), 32);
	float* q = (float*)_aligned_malloc(32, 32);


	for (int i = 0; i < M*K; i++)
	{
		//A[i] = rand() % 10001 / 5000.0f - 1.0f;
		A[i] = i;
	}
	for (int i = 0; i < K*N; i++)
	{
		B[i] = rand() % 10001 / 5000.0f - 1.0f;
		B[i] = i;
	}
	double t1 = omp_get_wtime(), t2, mul_count, gflops;
	
	//if (K % 64 == 0)
	//{
	//	for (int it = 0; it < iters; it++)
	//	{
	//		for (int m = 0; m < M; m++)
	//		{
	//			for (int n = 0; n < N; n++)
	//			{
	//				__m256 sum_vec = _mm256_setzero_ps();
	//				float* Aptr = A + m*K;
	//				float* Bptr = B + n*K;
	//				for (int k = 0; k < K; k += 64, Aptr += 64, Bptr += 64)
	//				{

	//					sum_vec = _mm256_fmadd_ps(_mm256_load_ps(Aptr), _mm256_load_ps(Bptr), sum_vec);
	//					sum_vec = _mm256_fmadd_ps(_mm256_load_ps(Aptr + 8), _mm256_load_ps(Bptr + 8), sum_vec);
	//					sum_vec = _mm256_fmadd_ps(_mm256_load_ps(Aptr + 16), _mm256_load_ps(Bptr + 16), sum_vec);
	//					sum_vec = _mm256_fmadd_ps(_mm256_load_ps(Aptr + 24), _mm256_load_ps(Bptr + 24), sum_vec);
	//					sum_vec = _mm256_fmadd_ps(_mm256_load_ps(Aptr + 32), _mm256_load_ps(Bptr + 32), sum_vec);
	//					sum_vec = _mm256_fmadd_ps(_mm256_load_ps(Aptr + 40), _mm256_load_ps(Bptr + 40), sum_vec);
	//					sum_vec = _mm256_fmadd_ps(_mm256_load_ps(Aptr + 48), _mm256_load_ps(Bptr + 48), sum_vec);
	//					sum_vec = _mm256_fmadd_ps(_mm256_load_ps(Aptr + 56), _mm256_load_ps(Bptr + 56), sum_vec);
	//				}
	//				_mm256_store_ps(q, sum_vec);
	//				_mm_store_ps(q, _mm_add_ps(_mm_load_ps(q), _mm_load_ps(q + 4)));
	//				C[m*N+n] = q[0] + q[1] + q[2] + q[3];

	//			}
	//		}
	//	}
	//	//printf("C[0] = %f\n", C[0]);
	//	t2 = omp_get_wtime();
	//	mul_count = (double)M*N*K*iters;
	//	gflops = mul_count / (1 << 30) / (t2 - t1);
	//	printf("%d x %d x %d * %d = %.3e, time = %.3f s, my gflops = %.3f\n", M, N, K, iters, mul_count, t2 - t1, gflops);
	//}



	t1 = omp_get_wtime();
	for (int i = 0; i < iters; i++)
	{
		for (int m = 0; m < M; m++)
			cblas_sgemv(CblasRowMajor, CblasNoTrans, N, K, 1.0, B, K, A + m*K, 1, 0, C + m*N, 1);
	}
	//printf("C[0] = %f\n", C[0]);
	t2 = omp_get_wtime();
	mul_count = (double)M*N*K*iters;
	gflops = mul_count / (1 << 30) / (t2 - t1);

	printf("%d x %d x %d * %d = %.3e, time = %.3f s, gemv gflops = %.3f\n", M, N, K, iters, mul_count, t2 - t1, gflops);

	_aligned_free(A);
	_aligned_free(B);
	_aligned_free(C);
	_aligned_free(q);
	return (t2 - t1) / iters;
}
double _test_gemm(int M, int N, int K, int iters = 1000)
{
	/*M = (M + 7) >> 3 << 3;
	K = (K + 63) >> 6 << 6;
	N = (N + 7) >> 3 << 3;*/
	float* A = (float*)_aligned_malloc(M*K * sizeof(float), 32);
	float* B = (float*)_aligned_malloc(K*N * sizeof(float), 32);
	float* C = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* q = (float*)_aligned_malloc(32, 32);


	for (int i = 0; i < M*K; i++)
		A[i] = rand() % 10001 / 5000.0f - 1.0f;
	for (int i = 0; i < K*N; i++)
		B[i] = rand() % 10001 / 5000.0f - 1.0f;
	double t1 = omp_get_wtime(), t2, mul_count, gflops;
	double time1 = FLT_MAX;
	if (K % 64 == 0)
	{
		if (M == 1)
		{
			for (int it = 0; it < iters; it++)
			{
				float* Aptr = A;
				float* A_c_ptr;
				float* Bptr = B;
				float* B_c_ptr;
				int n, k;
				for (n = 0; n < N; n++, Bptr+=K)
				{
					__m256 sum_vec = _mm256_setzero_ps();
					
					for (k = 0,A_c_ptr = Aptr, B_c_ptr = Bptr; k < K; k += 64, A_c_ptr += 64, B_c_ptr += 64)
					{

						sum_vec = _mm256_fmadd_ps(_mm256_load_ps(A_c_ptr), _mm256_load_ps(B_c_ptr), sum_vec);
						sum_vec = _mm256_fmadd_ps(_mm256_load_ps(A_c_ptr + 8), _mm256_load_ps(B_c_ptr + 8), sum_vec);
						sum_vec = _mm256_fmadd_ps(_mm256_load_ps(A_c_ptr + 16), _mm256_load_ps(B_c_ptr + 16), sum_vec);
						sum_vec = _mm256_fmadd_ps(_mm256_load_ps(A_c_ptr + 24), _mm256_load_ps(B_c_ptr + 24), sum_vec);
						sum_vec = _mm256_fmadd_ps(_mm256_load_ps(A_c_ptr + 32), _mm256_load_ps(B_c_ptr + 32), sum_vec);
						sum_vec = _mm256_fmadd_ps(_mm256_load_ps(A_c_ptr + 40), _mm256_load_ps(B_c_ptr + 40), sum_vec);
						sum_vec = _mm256_fmadd_ps(_mm256_load_ps(A_c_ptr + 48), _mm256_load_ps(B_c_ptr + 48), sum_vec);
						sum_vec = _mm256_fmadd_ps(_mm256_load_ps(A_c_ptr + 56), _mm256_load_ps(B_c_ptr + 56), sum_vec);
					}
					_mm256_store_ps(q, sum_vec);
					_mm_store_ps(q, _mm_add_ps(_mm_load_ps(q), _mm_load_ps(q + 4)));
					C[n] = q[0] + q[1] + q[2] + q[3];
				}

			}
		}
		else
		{
			for (int it = 0; it < iters; it++)
			{
				float* Aptr = A;
				float* A_c_ptr;
				float* Cptr = C;
				for (int m = 0; m < M; m++, Aptr+=K)
				{
					float* Bptr = B;
					float* B_c_ptr;
					int n, k;
					for (n = 0; n < N; n++, Bptr+=K)
					{
						__m256 sum_vec = _mm256_setzero_ps();
						
						for (k = 0,A_c_ptr = Aptr, B_c_ptr = Bptr; k < K; k += 64, A_c_ptr += 64, B_c_ptr += 64)
						{

							sum_vec = _mm256_fmadd_ps(_mm256_load_ps(A_c_ptr), _mm256_load_ps(B_c_ptr), sum_vec);
							sum_vec = _mm256_fmadd_ps(_mm256_load_ps(A_c_ptr + 8), _mm256_load_ps(B_c_ptr + 8), sum_vec);
							sum_vec = _mm256_fmadd_ps(_mm256_load_ps(A_c_ptr + 16), _mm256_load_ps(B_c_ptr + 16), sum_vec);
							sum_vec = _mm256_fmadd_ps(_mm256_load_ps(A_c_ptr + 24), _mm256_load_ps(B_c_ptr + 24), sum_vec);
							sum_vec = _mm256_fmadd_ps(_mm256_load_ps(A_c_ptr + 32), _mm256_load_ps(B_c_ptr + 32), sum_vec);
							sum_vec = _mm256_fmadd_ps(_mm256_load_ps(A_c_ptr + 40), _mm256_load_ps(B_c_ptr + 40), sum_vec);
							sum_vec = _mm256_fmadd_ps(_mm256_load_ps(A_c_ptr + 48), _mm256_load_ps(B_c_ptr + 48), sum_vec);
							sum_vec = _mm256_fmadd_ps(_mm256_load_ps(A_c_ptr + 56), _mm256_load_ps(B_c_ptr + 56), sum_vec);
						}
						_mm256_store_ps(q, sum_vec);
						_mm_store_ps(q, _mm_add_ps(_mm_load_ps(q), _mm_load_ps(q + 4)));
						*(Cptr++) = q[0] + q[1] + q[2] + q[3];
					}
				}

			}
		}
		t2 = omp_get_wtime();
		time1 = t2 - t1;
		mul_count = (double)M*N*K*iters;
		gflops = mul_count / (1 << 30) / (t2 - t1);
		//printf("C[0] = %f\n", C[0]);
		printf("%d x %d x %d * %d = %.3e, time = %.3f s, my gflops = %.3f\n", M, N, K, iters, mul_count, time1, gflops);

	}
	else if (K % 32 == 0)
	{
		for (int it = 0; it < iters; it++)
		{
			for (int m = 0; m < M; m++)
			{
				for (int n = 0; n < N; n++)
				{
					__m256 sum_vec = _mm256_setzero_ps();
					float* Aptr = A + m*K;
					float* Bptr = B + n*K;
					for (int k = 0; k < K; k += 32, Aptr += 32, Bptr += 32)
					{

						sum_vec = _mm256_fmadd_ps(_mm256_load_ps(Aptr), _mm256_load_ps(Bptr), sum_vec);
						sum_vec = _mm256_fmadd_ps(_mm256_load_ps(Aptr + 8), _mm256_load_ps(Bptr + 8), sum_vec);
						sum_vec = _mm256_fmadd_ps(_mm256_load_ps(Aptr + 16), _mm256_load_ps(Bptr + 16), sum_vec);
						sum_vec = _mm256_fmadd_ps(_mm256_load_ps(Aptr + 24), _mm256_load_ps(Bptr + 24), sum_vec);
						
					}
					_mm256_store_ps(q, sum_vec);
					_mm_store_ps(q, _mm_add_ps(_mm_load_ps(q), _mm_load_ps(q + 4)));
					C[m*N + n] = q[0] + q[1] + q[2] + q[3];
				}
			}

		}
		t2 = omp_get_wtime();
		time1 = t2 - t1;
		mul_count = (double)M*N*K*iters;
		gflops = mul_count / (1 << 30) / (t2 - t1);
		//printf("C[0] = %f\n", C[0]);
		printf("%d x %d x %d * %d = %.3e, time = %.3f s, my gflops = %.3f\n", M, N, K, iters, mul_count, time1, gflops);
	}
	else if (K % 16 == 0)
	{
		for (int it = 0; it < iters; it++)
		{
			for (int m = 0; m < M; m++)
			{
				for (int n = 0; n < N; n++)
				{
					__m256 sum_vec = _mm256_setzero_ps();
					float* Aptr = A + m*K;
					float* Bptr = B + n*K;
					for (int k = 0; k < K; k += 16, Aptr += 16, Bptr += 16)
					{
						sum_vec = _mm256_fmadd_ps(_mm256_load_ps(Aptr), _mm256_load_ps(Bptr), sum_vec);
						sum_vec = _mm256_fmadd_ps(_mm256_load_ps(Aptr+8), _mm256_load_ps(Bptr+8), sum_vec);
					}
					_mm256_store_ps(q, sum_vec);
					_mm_store_ps(q, _mm_add_ps(_mm_load_ps(q), _mm_load_ps(q + 4)));
					C[m*N + n] = q[0] + q[1] + q[2] + q[3];
				}
			}

		}
		t2 = omp_get_wtime();
		time1 = t2 - t1;
		mul_count = (double)M*N*K*iters;
		gflops = mul_count / (1 << 30) / (t2 - t1);
		//printf("C[0] = %f\n", C[0]);
		printf("%d x %d x %d * %d = %.3e, time = %.3f s, my gflops = %.3f\n", M, N, K, iters, mul_count, time1, gflops);
	}
	else if (K % 8 == 0)
	{
		for (int it = 0; it < iters; it++)
		{
			for (int m = 0; m < M; m++)
			{
				for (int n = 0; n < N; n++)
				{
					__m256 sum_vec = _mm256_setzero_ps();
					float* Aptr = A + m*K;
					float* Bptr = B + n*K;
					for (int k = 0; k < K; k += 8, Aptr += 8, Bptr += 8)
					{
						sum_vec = _mm256_fmadd_ps(_mm256_load_ps(Aptr), _mm256_load_ps(Bptr), sum_vec);
					}
					_mm256_store_ps(q, sum_vec);
					_mm_store_ps(q, _mm_add_ps(_mm_load_ps(q), _mm_load_ps(q + 4)));
					C[m*N + n] = q[0] + q[1] + q[2] + q[3];
				}
			}

		}
		t2 = omp_get_wtime();
		time1 = t2 - t1;
		mul_count = (double)M*N*K*iters;
		gflops = mul_count / (1 << 30) / (t2 - t1);
		//printf("C[0] = %f\n", C[0]);
		printf("%d x %d x %d * %d = %.3e, time = %.3f s, my gflops = %.3f\n", M, N, K, iters, mul_count, time1, gflops);
	}
	else if (K % 4 == 0)
	{
		for (int it = 0; it < iters; it++)
		{
			for (int m = 0; m < M; m++)
			{
				for (int n = 0; n < N; n++)
				{
					__m128 sum_vec = _mm_setzero_ps();
					float* Aptr = A + m*K;
					float* Bptr = B + n*K;
					for (int k = 0; k < K; k += 4, Aptr += 4, Bptr += 4)
					{
						sum_vec = _mm_fmadd_ps(_mm_load_ps(Aptr), _mm_load_ps(Bptr), sum_vec);
					}
					_mm_store_ps(q, sum_vec);
					C[m*N + n] = q[0] + q[1] + q[2] + q[3];
				}
			}

		}
		t2 = omp_get_wtime();
		time1 = t2 - t1;
		mul_count = (double)M*N*K*iters;
		gflops = mul_count / (1 << 30) / (t2 - t1);
		//printf("C[0] = %f\n", C[0]);
		printf("%d x %d x %d * %d = %.3e, time = %.3f s, my gflops = %.3f\n", M, N, K, iters, mul_count, time1, gflops);
	}



	t1 = omp_get_wtime();
	for (int i = 0; i < iters; i++)
	{
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, A, K, B, K, 0, C, N);
		
	}
	//printf("C[0] = %f\n", C[0]);
	t2 = omp_get_wtime();
	mul_count = (double)M*N*K*iters;
	gflops = mul_count / (1 << 30) / (t2 - t1);
	double  time2 = t2 - t1;
	printf("%d x %d x %d * %d = %.3e, time = %.3f s, gemm gflops = %.3f\n", M, N, K, iters, mul_count, time2, gflops);

	_aligned_free(A);
	_aligned_free(B);
	_aligned_free(C);
	_aligned_free(q);


	return __min(time1, time2) / iters;
}

float _test_im2col(int in_H, int in_W, int filter_N, int filter_C, int stride_H, int stride_W, int iters = 1000)
{
	ZQ_CNN_Tensor4D_NHW_C_Align256bit input, filters, output;
	int in_N = 1, in_C = filter_C;
	int filter_H = 3, filter_W = 3;
	int pad_H = 1, pad_W = 1;
	int out_N = in_N;
	int out_H = (in_H - filter_H + pad_H * 2) / stride_H + 1;
	int out_W = (in_W - filter_W + pad_W * 2) / stride_W + 1;
	int out_C = filter_N;

	input.ChangeSize(in_N, in_H, in_W, in_C, pad_H, pad_W);
	filters.ChangeSize(filter_N, filter_H, filter_W, filter_C, 0, 0);
	output.ChangeSize(out_N, out_H, out_W, out_C, 0, 0);

	
	double t1 = omp_get_wtime();
	float* matrix_C = (float*)malloc(out_H*out_W*filter_N * sizeof(float));
	for (int it = 0; it < iters; it++)
	{
		int in_widthStep = input.GetWidthStep();
		int in_widthStep_mul_stride_H = in_widthStep*stride_H;
		int in_pixelStep = input.GetPixelStep();
		int in_pixelStep_mul_stride_W = in_pixelStep*stride_W;
		int filter_pixStep = filters.GetPixelStep();
		int filter_pixStep_mul_filter_W = filter_pixStep*filter_W;
		int filter_SliceStep = filters.GetSliceStep();
		int out_pixStep = output.GetPixelStep();
		int out_widthStep = output.GetWidthStep();
		int matrix_A_cols = filter_SliceStep;
		int matrix_A_rows = out_H*out_W;
		int matrix_B_cols = filter_N;
		int matrix_B_rows = filter_SliceStep;

		const float* in_firstPixel = input.GetFirstPixelPtr() - pad_H*in_widthStep - pad_W*in_pixelStep;
		float* out_firstPixel = output.GetFirstPixelPtr();
		float* matrix_A = (float*)_aligned_malloc(matrix_A_rows*matrix_A_cols * sizeof(float), 32);
		const float* in_row_ptr, *in_pix_ptr, *cur_in_row_ptr;
		int out_h, out_w, kh;
		float* matrix_A_row_ptr = matrix_A, *matrix_A_col_ptr;
		for (out_h = 0, in_row_ptr = in_firstPixel; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
			{

				for (kh = 0, cur_in_row_ptr = in_pix_ptr, matrix_A_col_ptr = matrix_A_row_ptr;
					kh < filter_H;
					kh++, cur_in_row_ptr += in_widthStep, matrix_A_col_ptr += filter_pixStep_mul_filter_W)
					memcpy(matrix_A_col_ptr, cur_in_row_ptr, sizeof(float)*filter_pixStep_mul_filter_W);
			}
		}
		const float* matrix_B = filters.GetFirstPixelPtr();

		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
			matrix_B, matrix_A_cols, 0, matrix_C, matrix_B_cols);
		int out_row_idx = 0;
		for (out_h = 0; out_h < out_H; out_h++)
		{
			for (out_w = 0; out_w < out_w; out_w++)
			{
				memcpy(out_firstPixel + out_h*out_widthStep + out_w*out_pixStep, matrix_C + out_row_idx*matrix_B_cols, sizeof(float)*matrix_B_cols);
			}
		}
		_aligned_free(matrix_A);

	}


	free(matrix_C);
	double t2 = omp_get_wtime();
	double mul_count = (double)out_H*out_W*filter_N*filter_H*filter_W*filter_C*iters;
	double gflops = mul_count / (1 << 30) / (t2 - t1);

	printf("%dx%d * %dx%dx%dx%d * %d = %.3e, time = %.3f s, gemm gflops = %.3f\n", out_H, out_W, filter_N, filter_H, filter_W, filter_C, iters, mul_count, t2 - t1, gflops);

	return (t2 - t1) / iters;

}

int main()
{
	openblas_set_num_threads(1);
	double total_sum = 0;
	//_test_gemm_value();
	/*total_sum += 1*_test_im2col(112, 96, 64, 3, 2, 2,1000);
	total_sum += 2 * _test_im2col(56, 48, 64, 64, 1, 1,1000);
	total_sum+=1*_test_im2col(56, 48, 128, 64, 2, 2,1000);
	total_sum+=4*_test_im2col(28, 24, 128, 128, 1, 1,1000);
	total_sum += 1*_test_im2col(28, 24, 256, 128, 2, 2,1000);
	total_sum += 8*_test_im2col(14, 12, 256, 256, 1, 1,1000);
	total_sum += 1 * _test_im2col(14, 12, 512, 256, 2, 2,1000);
	total_sum += 2 * _test_im2col(7, 6, 512, 512, 1, 1,1000);
	total_sum += _test_gemm(1 * 1, 512, 7 * 6 * 512,1000);
	printf("total: %.3f\n", total_sum);*/

	//compare gemm the spherefacenet04
	_test_gemm(56 * 48, 64, 3 * 3 * 3,1000);
	_test_gemm(56 * 48, 64, 3 * 3 * 4,1000);
	_test_gemm(28 * 24, 128, 3 * 3 * 64,1000);
	_test_gemm(14 * 12, 256, 3 * 3 * 128,1000);
	_test_gemm(7 * 6, 512, 3 * 3 * 256,1000);
	_test_gemm(1 * 1, 512, 7 * 6 * 512,1000);

	//compare gemv
	/*_test_gemv(56 * 48, 64, 3 * 3 * 3,1000);
	_test_gemv(56 * 48, 64, 3 * 3 * 4, 1000);
	_test_gemv(28 * 24, 128, 3 * 3 * 64,1000);
	_test_gemv(14 * 12, 256, 3 * 3 * 128,1000);
	_test_gemv(7 * 6, 512, 3 * 3 * 256,1000);*/
	_test_gemv(1 * 1, 512, 7 * 6 * 512,1000);

	//compare  MTCNN Pnet
	_test_gemm(214 * 382, 10, 3 * 3 * 3, 1000);
	_test_gemm(214 * 382, 10, 3 * 3 * 4, 1000);
	_test_gemm(105 * 189, 16, 3 * 3 * 10, 1000);
	_test_gemm(105 * 189, 16, 3 * 3 * 16, 1000);
	_test_gemm(103 * 187, 32, 3 * 3 * 16, 1000);
	_test_gemm(103 * 187, 2, 1 * 1 * 32, 1000);
	_test_gemv(103 * 187, 2, 1 * 1 * 32, 1000);
	_test_gemm(103 * 187, 4, 1 * 1 * 32, 1000);
	_test_gemv(103 * 187, 4, 1 * 1 * 32, 1000);
	

	//other tests
	/*_test_gemm(92, 128, 3 * 3 * 64);
	_test_gemm(256,  2, 6 * 7 * 512);
	_test_gemm(256, 4, 6 * 7 * 512);
	_test_gemm(256, 10, 6 * 7 * 512);
	_test_gemm(1, 1024 * 512, 3 * 3 * 64);
	_test_gemm(56*48, 64, 3*3*64);
	_test_gemm(56 * 48+1, 64+1, 3 * 3 * 64+1);
	_test_gemm(28*24, 128, 3 * 3 * 128);
	_test_gemm(28 * 24+1, 128+1, 3 * 3 * 128+1);
	_test_gemm(14 * 12, 256, 3 * 3 * 256);
	_test_gemm(14 * 12+1, 256+1, 3 * 3 * 256+1);
	_test_gemm(7 * 6, 512, 3 * 3 * 512);
	_test_gemm(7 * 6+1, 512+1, 3 * 3 * 512+1);
	_test_gemm(512, 512, 3 * 3 * 512);
	_test_gemm(513, 512 + 1, 3 * 3 * 512 + 1);
	_test_gemm(1024, 1024, 1024);
	_test_gemm(1025, 1025, 1025);*/
	return EXIT_SUCCESS;
}
