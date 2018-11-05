#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "..\ZQ_CNN_CompileConfig.h"
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
#if ZQ_CNN_USE_BLAS_GEMM
#include <cblas.h>

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_C4 zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_C4
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_omp zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_omp
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_batch zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_batch
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_batch_omp zq_cnn_conv_no_padding_gemm_32f_align128bit_same_pixstep_batch_omp
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_omp zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_omp
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_C3 zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_C3
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_C3_omp zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_C3_omp
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_batch zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_batch
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_batch_omp zq_cnn_conv_no_padding_gemm_32f_align128bit_same_or_notsame_pixstep_batch_omp
#define zq_mm_load_ps _mm_load_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_add_ps _mm_add_ps
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps _mm_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm_add_ps(_mm_mul_ps(A, B), C)
#endif
#define zq_mm_mul_ps _mm_mul_ps
#define zq_mm_setzero_ps _mm_setzero_ps
#define zq_mm_type __m128
#define zq_mm_align_size 4
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFF0
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3])

#include "zq_cnn_convolution_gemm_32f_align_c_raw.h"


#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_C4
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_omp
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_batch
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_batch_omp
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_omp
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_C3
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_C3_omp
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_batch
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_batch_omp
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
#undef zq_mm_type
#undef zq_mm_align_size
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q

#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX

#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_C4 zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep_C4
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_omp zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep_omp
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_batch zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep_batch
#define zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_batch_omp zq_cnn_conv_no_padding_gemm_32f_align256bit_same_pixstep_batch_omp
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_omp zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep_omp
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_C3 zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep_C3
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_C3_omp zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep_C3_omp
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_batch zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep_batch
#define zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_batch_omp zq_cnn_conv_no_padding_gemm_32f_align256bit_same_or_notsame_pixstep_batch_omp
#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_add_ps _mm256_add_ps
#if ZQ_CNN_USE_FMADD256
#define zq_mm_fmadd_ps _mm256_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#endif
#define zq_mm_mul_ps _mm256_mul_ps
#define zq_mm_setzero_ps _mm256_setzero_ps
#define zq_mm_type __m256
#define zq_mm_align_size 8
#define zq_mm_bitor_longlong 0xFFFFFFFFFFFFFFE0
#define zq_final_sum_q (q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7])

#include "zq_cnn_convolution_gemm_32f_align_c_raw.h"


#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_C4
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_omp
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_batch
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_batch_omp
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_omp
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_C3
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_C3_omp
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_batch
#undef zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_batch_omp
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_mul_ps
#undef zq_mm_setzero_ps
#undef zq_mm_type
#undef zq_mm_align_size
#undef zq_mm_bitor_longlong
#undef zq_final_sum_q
#endif

	/*in_pixStep can be different with filter_pixStep,
	and the aligned channels should be set to zero*/
	void zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,
		int filter_H,
		int filter_W,
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		int dilation_H,
		int dilation_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		void** buffer,
		__int64 *buffer_len
	)
	{
		/************** image to col **************/
		int in_widthStep_mul_stride_H = in_widthStep*stride_H;
		int in_pixelStep_mul_stride_W = in_pixelStep*stride_W;
		int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
		int filter_pixStep_mul_filter_W = filter_pixelStep*filter_W;
		int matrix_A_cols = filter_H*filter_W*in_C;
		int matrix_A_rows = out_H*out_W;
		int matrix_B_cols = filter_N;
		int matrix_B_rows = filter_H*filter_W*filter_C;
		__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(float) + 31) / 32 * 32;
		__int64 need_B_buffer_len_align32 = (matrix_B_rows*matrix_B_cols * sizeof(float) + 31) / 32 * 32;
		__int64 need_C_buffer_len_align32 = 0;
		__int64 total_need_buffer_len;
		float* matrix_A = 0;
		float* matrix_Bt = 0;
		const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *cur_in_row_ptr,*cur_in_pix_ptr,*filter_slice_ptr,*filter_row_ptr,*filter_pix_ptr;
		int out_n, out_h, out_w, kn, kh, kw;
		float* matrix_A_row_ptr, *matrix_A_col_ptr, *cp_dst_ptr;
		float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		float* matrix_C = 0, *matrix_C_row_ptr;
		int out_row_idx;
		int out_HW = out_H*out_W;
		double t1, t2, t3, t4, t5;
		int need_allocate_tmp_out;
		t1 = omp_get_wtime();
		need_allocate_tmp_out = (out_pixelStep != filter_N) || (out_pixelStep*out_W != out_widthStep) || (out_widthStep*out_H != out_sliceStep);
		if (need_allocate_tmp_out)
		{
			need_C_buffer_len_align32 = (out_HW*filter_N * sizeof(float) + 31) / 32 * 32;
		}
		else
		{
			matrix_C = out_tensor4D_data;
		}
		
		total_need_buffer_len = need_A_buffer_len_align32 + need_B_buffer_len_align32 + need_C_buffer_len_align32;
		if (buffer == 0)
		{
			matrix_A = _aligned_malloc(need_A_buffer_len_align32, 32);
			matrix_Bt = _aligned_malloc(need_B_buffer_len_align32, 32);
			if (need_allocate_tmp_out)
				matrix_C = _aligned_malloc(need_C_buffer_len_align32, 32);
		}
		else
		{
			if (*buffer_len < total_need_buffer_len)
			{
				_aligned_free(*buffer);
				*buffer = _aligned_malloc(total_need_buffer_len, 32);
				*buffer_len = total_need_buffer_len;
			}
			matrix_A = *buffer;
			matrix_Bt = (float*)((char*)(*buffer) + need_A_buffer_len_align32);
			if (need_allocate_tmp_out)
				matrix_C = (float*)((char*)(*buffer) + need_A_buffer_len_align32 + need_B_buffer_len_align32);
		}
		

		cp_dst_ptr = matrix_Bt;
		for (kn = 0,filter_slice_ptr = filters_data; kn < filter_N; kn++,filter_slice_ptr+=filter_sliceStep)
		{
			for (kh = 0, filter_row_ptr = filter_slice_ptr; kh < filter_H; kh++, filter_row_ptr += filter_widthStep)
			{
				for (kw = 0, filter_pix_ptr = filter_row_ptr; kw < filter_W; kw++, filter_pix_ptr += filter_pixelStep)
				{
					memcpy(cp_dst_ptr, filter_pix_ptr, sizeof(float)*in_C);
					cp_dst_ptr += in_C;
				}
			}
		}

		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			t2 = omp_get_wtime();
			matrix_A_row_ptr = matrix_A;
			for (out_h = 0, in_row_ptr = in_slice_ptr; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
				{
					matrix_A_col_ptr = matrix_A_row_ptr;
					for (kh = 0, cur_in_row_ptr = in_pix_ptr;kh < filter_H;	kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep)
					{
						for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < filter_W; kw++, cur_in_pix_ptr += dilate_W_mul_in_pixStep)
						{
							memcpy(matrix_A_col_ptr, cur_in_pix_ptr, sizeof(float)*in_C);
							matrix_A_col_ptr += in_C;
						}
					}
					matrix_A_row_ptr += matrix_A_cols;
				}
			}
			t3 = omp_get_wtime();
			/*gemm*/
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
				matrix_Bt, matrix_A_cols, 0.0f, matrix_C, matrix_B_cols);
			t4 = omp_get_wtime();
			if (need_allocate_tmp_out)
			{
				/*   col2im      */
				out_row_idx = 0;
				matrix_C_row_ptr = matrix_C;
				for (out_h = 0, out_row_ptr = out_slice_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
				{
					for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W; out_w++, out_pix_ptr += out_pixelStep)
					{
						memcpy(out_pix_ptr, matrix_C_row_ptr, sizeof(float)*matrix_B_cols);
						matrix_C_row_ptr += matrix_B_cols;
					}
				}
			}
			else
				matrix_C += out_sliceStep;
			t5 = omp_get_wtime();
		}
		if (buffer == 0)
		{
			_aligned_free(matrix_A);
			_aligned_free(matrix_Bt);
			if (need_allocate_tmp_out)
				_aligned_free(matrix_C);
		}
		if (filter_H == 3 && filter_W == 3 && filter_C == 3)
		{
			printf("total: %.3f ms, alloc %.3f ms, makeA: %.3f ms, gemm: %.3f ms, copy_C: %.3f ms\n", 1000 * (t5 - t1), 1000 * (t2 - t1),
				1000 * (t3 - t2), 1000 * (t4 - t3), 1000 * (t5 - t4));
		}
	}

	/*in_pixStep can be different with filter_pixStep,
	and the aligned channels should be set to zero*/
	void zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep_omp(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,
		int filter_H,
		int filter_W,
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		int dilation_H,
		int dilation_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		int thread_count
	)
	{
		/************** image to col **************/
		int in_widthStep_mul_stride_H = in_widthStep*stride_H;
		int in_pixelStep_mul_stride_W = in_pixelStep*stride_W;
		int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
		int filter_pixStep_mul_filter_W = filter_pixelStep*filter_W;
		int matrix_A_cols = filter_H*filter_W*in_C;
		int matrix_A_rows = out_H*out_W;
		int matrix_B_cols = filter_N;
		int matrix_B_rows = filter_H*filter_W*filter_C;
		float* matrix_A = (float*)_aligned_malloc(matrix_A_rows*matrix_A_cols * sizeof(float), 32);
		float* matrix_Bt = (float*)_aligned_malloc(matrix_B_rows*matrix_B_cols * sizeof(float), 32);
		const float* in_slice_ptr, *filter_slice_ptr, *filter_row_ptr, *filter_pix_ptr;
		int out_n, out_h, out_w, kn, kh, kw;
		float* cp_dst_ptr;
		float* out_slice_ptr;
		float* matrix_C;
		int out_HW = out_H*out_W;
		double t1, t2, t3, t4, t5;
		int need_allocate_tmp_out;
		

		int chunk_size = (out_HW + thread_count - 1) / thread_count;
		int gemm_per_row = (matrix_A_rows + thread_count - 1) / thread_count;
		int cur_row_num;
		int* in_offsets = (int*)malloc(out_HW * sizeof(int));
		int* matA_offsets = (int*)malloc(out_HW * sizeof(int));
		int* out_offsets = (int*)malloc(out_HW * sizeof(int));
		int* matC_offsets = (int*)malloc(out_HW * sizeof(int));
		int idx = 0;
		for (out_h = 0; out_h < out_H; out_h++)
		{
			for (out_w = 0; out_w < out_W; out_w++)
			{
				in_offsets[idx] = out_h*in_widthStep_mul_stride_H + out_w*in_pixelStep_mul_stride_W;
				matA_offsets[idx] = (out_h*out_W + out_w)*matrix_A_cols;
				out_offsets[idx] = out_h*out_widthStep + out_w*out_pixelStep;
				matC_offsets[idx] = (out_h*out_W + out_w)*matrix_B_cols;
				idx++;
			}
		}


		t1 = omp_get_wtime();
		need_allocate_tmp_out = (out_pixelStep != filter_N) || (out_pixelStep*out_W != out_widthStep) || (out_widthStep*out_H != out_sliceStep);
		if (need_allocate_tmp_out)
		{
			matrix_C = (float*)_aligned_malloc(out_HW*filter_N * sizeof(float), 32);
		}
		else
		{
			matrix_C = out_tensor4D_data;
		}

		cp_dst_ptr = matrix_Bt;
		for (kn = 0, filter_slice_ptr = filters_data; kn < filter_N; kn++, filter_slice_ptr += filter_sliceStep)
		{
			for (kh = 0, filter_row_ptr = filter_slice_ptr; kh < filter_H; kh++, filter_row_ptr += filter_widthStep)
			{
				for (kw = 0, filter_pix_ptr = filter_row_ptr; kw < filter_W; kw++, filter_pix_ptr += filter_pixelStep)
				{
					memcpy(cp_dst_ptr, filter_pix_ptr, sizeof(float)*in_C);
					cp_dst_ptr += in_C;
				}
			}
		}

		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			t2 = omp_get_wtime();
			
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
			for (idx = 0; idx < out_HW; idx++)
			{
				const float* in_pix_ptr, *cur_in_row_ptr, *cur_in_pix_ptr;
				float* matrix_A_row_ptr, *matrix_A_col_ptr;
				in_pix_ptr = in_slice_ptr + in_offsets[idx];
				matrix_A_row_ptr = matrix_A + matA_offsets[idx];
				matrix_A_col_ptr = matrix_A_row_ptr;
				for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < filter_H; kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep)
				{
					for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < filter_W; kw++, cur_in_pix_ptr += dilate_W_mul_in_pixStep)
					{
						memcpy(matrix_A_col_ptr, cur_in_pix_ptr, sizeof(float)*in_C);
						matrix_A_col_ptr += in_C;
					}
				}
			}

			t3 = omp_get_wtime();
			/*gemm*/
			if (0 && openblas_get_num_threads() == 1 && gemm_per_row >= 4)
			{
#pragma omp parallel for schedule(static) num_threads(thread_count)
				for (idx = 0; idx < thread_count; idx++)
				{
					cur_row_num = __min(matrix_A_rows, gemm_per_row*(idx + 1)) - gemm_per_row*idx;
					if (cur_row_num > 0)
					{
						cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, cur_row_num, matrix_B_cols, matrix_A_cols, 1,
							matrix_A + gemm_per_row*idx*matrix_A_cols, matrix_A_cols,
							matrix_Bt, matrix_A_cols, 0.0f, matrix_C + gemm_per_row*idx*matrix_B_cols, matrix_B_cols);
					}
				}
			}
			else
			{
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
					matrix_Bt, matrix_A_cols, 0.0f, matrix_C, matrix_B_cols);
			}
			
			t4 = omp_get_wtime();
			if (need_allocate_tmp_out)
			{
				/*   col2im      */
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
				for (idx = 0; idx < out_HW; idx++)
				{
					float* out_pix_ptr = out_slice_ptr + out_offsets[idx];
					float* matrix_C_row_ptr = matrix_C + matC_offsets[idx];
					memcpy(out_pix_ptr, matrix_C_row_ptr, sizeof(float)*matrix_B_cols);
				}
			}
			else
				matrix_C += out_sliceStep;
			t5 = omp_get_wtime();
		}

		if (need_allocate_tmp_out)
			_aligned_free(matrix_C);
		_aligned_free(matrix_A);
		_aligned_free(matrix_Bt);
		free(in_offsets);
		free(out_offsets);
		free(matA_offsets);
		free(matC_offsets);
		if (filter_H == 3 && filter_W == 3 && filter_C == 3)
		{
			printf("total: %.3f ms, alloc %.3f ms, makeA: %.3f ms, gemm: %.3f ms, copy_C: %.3f ms\n", 1000 * (t5 - t1), 1000 * (t2 - t1),
				1000 * (t3 - t2), 1000 * (t4 - t3), 1000 * (t5 - t4));
		}
	}

	/*in_pixStep can be different with filter_pixStep,
	and the aligned channels should be set to zero*/
	void zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep_batch(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,
		int filter_H,
		int filter_W,
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		int dilation_H,
		int dilation_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		void** buffer,
		__int64 *buffer_len
	)
	{
		/************** image to col **************/
		int in_widthStep_mul_stride_H = in_widthStep*stride_H;
		int in_pixelStep_mul_stride_W = in_pixelStep*stride_W;
		int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
		int filter_pixStep_mul_filter_W = filter_pixelStep*filter_W;
		int matrix_A_cols = in_C;
		int matrix_A_rows = out_N*out_H*out_W;
		int matrix_B_cols = filter_N;
		int matrix_B_rows = filter_H*filter_W*filter_C;
		__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(float) + 31) / 32 * 32;
		__int64 need_B_buffer_len_align32 = (matrix_B_rows*matrix_B_cols * sizeof(float) + 31) / 32 * 32;
		__int64 need_C_buffer_len_align32 = 0;
		__int64 total_need_buffer_len;
		float* matrix_A = 0;
		float* matrix_Bt = 0;
		const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *cur_in_row_ptr, *cur_in_pix_ptr, *filter_slice_ptr, *filter_row_ptr, *filter_pix_ptr;
		int out_n, out_h, out_w, kn, kh, kw;
		float* matrix_A_row_ptr, *matrix_A_col_ptr, *cp_dst_ptr;
		float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
		float* matrix_C = 0, *matrix_C_row_ptr;
		int out_row_idx;
		int out_NHW = out_N*out_H*out_W;
		double t1, t2, t3, t4, t5;
		t1 = omp_get_wtime();
		int need_allocate_tmp_out = (out_pixelStep != filter_N) || (out_pixelStep*out_W != out_widthStep) || (out_widthStep*out_H != out_sliceStep);
		if (need_allocate_tmp_out)
		{
			need_C_buffer_len_align32 = (out_NHW*filter_N * sizeof(float) + 31) / 32 * 32;
		}
		else
		{
			matrix_C = out_tensor4D_data;
		}

		total_need_buffer_len = need_A_buffer_len_align32 + need_B_buffer_len_align32 + need_C_buffer_len_align32;
		if (buffer == 0)
		{
			matrix_A = _aligned_malloc(need_A_buffer_len_align32, 32);
			matrix_Bt = _aligned_malloc(need_B_buffer_len_align32, 32);
			if (need_allocate_tmp_out)
				matrix_C = _aligned_malloc(need_C_buffer_len_align32, 32);
		}
		else
		{
			if (*buffer_len < total_need_buffer_len)
			{
				_aligned_free(*buffer);
				*buffer = _aligned_malloc(total_need_buffer_len, 32);
				*buffer_len = total_need_buffer_len;
			}
			matrix_A = *buffer;
			matrix_Bt = (float*)((char*)(*buffer) + need_A_buffer_len_align32);
			if (need_allocate_tmp_out)
				matrix_C = (float*)((char*)(*buffer) + need_A_buffer_len_align32 + need_B_buffer_len_align32);
		}
		t2 = omp_get_wtime();

		cp_dst_ptr = matrix_Bt;
		for (kn = 0, filter_slice_ptr = filters_data; kn < filter_N; kn++, filter_slice_ptr += filter_sliceStep)
		{
			for (kh = 0, filter_row_ptr = filter_slice_ptr; kh < filter_H; kh++, filter_row_ptr += filter_widthStep)
			{
				for (kw = 0, filter_pix_ptr = filter_row_ptr; kw < filter_W; kw++, filter_pix_ptr += filter_pixelStep)
				{
					memcpy(cp_dst_ptr, filter_pix_ptr, sizeof(float)*in_C);
					cp_dst_ptr += in_C;
				}
			}
		}

		matrix_A_row_ptr = matrix_A;
		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
				{
					for (kh = 0, cur_in_row_ptr = in_pix_ptr, matrix_A_col_ptr = matrix_A_row_ptr;
						kh < filter_H;
						kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep)
					{
						for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < filter_W; kw++, cur_in_pix_ptr += dilate_W_mul_in_pixStep)
						{
							memcpy(matrix_A_col_ptr, cur_in_pix_ptr, sizeof(float)*in_C);
							matrix_A_col_ptr += in_C;
						}
					}
					matrix_A_row_ptr += matrix_A_cols;
				}
			}
		}

		t3 = omp_get_wtime();
		/*gemm*/
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
			matrix_Bt, matrix_A_cols, 0.0f, matrix_C, matrix_B_cols);
		t4 = omp_get_wtime();
		if (need_allocate_tmp_out)
		{
			/*   col2im      */
			out_row_idx = 0;
			matrix_C_row_ptr = matrix_C;
			for (out_n = 0, out_slice_ptr = out_tensor4D_data; out_n < out_N; out_n++)
			{
				for (out_h = 0, out_row_ptr = out_slice_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
				{
					for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W; out_w++, out_pix_ptr += out_pixelStep)
					{
						memcpy(out_pix_ptr, matrix_C_row_ptr, sizeof(float)*matrix_B_cols);
						matrix_C_row_ptr += matrix_B_cols;
					}
				}
			}
		}
		else
			matrix_C += out_sliceStep;
		t5 = omp_get_wtime();

		if (buffer == 0)
		{
			_aligned_free(matrix_A);
			_aligned_free(matrix_Bt);
			if (need_allocate_tmp_out)
				_aligned_free(matrix_C);
		}
		if (filter_H == 3 && filter_W == 3 && filter_C == 3)
		{
			printf("total: %.3f ms, alloc %.3f ms, makeA: %.3f ms, gemm: %.3f ms, copy_C: %.3f ms\n", 1000 * (t5 - t1), 1000 * (t2 - t1),
				1000 * (t3 - t2), 1000 * (t4 - t3), 1000 * (t5 - t4));
		}
	}

	/*in_pixStep can be different with filter_pixStep,
	and the aligned channels should be set to zero*/
	void zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep_batch_omp(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		const float* filters_data,
		int filter_N,
		int filter_H,
		int filter_W,
		int filter_C, // must be in_C
		int filter_pixelStep,
		int filter_widthStep,
		int filter_sliceStep,
		int stride_H,
		int stride_W,
		int dilation_H,
		int dilation_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep,
		int thread_count
	)
	{
		/************** image to col **************/
		int in_widthStep_mul_stride_H = in_widthStep*stride_H;
		int in_pixelStep_mul_stride_W = in_pixelStep*stride_W;
		int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
		int filter_pixStep_mul_filter_W = filter_pixelStep*filter_W;
		int matrix_A_cols = in_C;
		int matrix_A_rows = out_N*out_H*out_W;
		int matrix_B_cols = filter_N;
		int matrix_B_rows = filter_H*filter_W*filter_C;
		float* matrix_A = (float*)_aligned_malloc(matrix_A_rows*matrix_A_cols * sizeof(float), 32);
		float* matrix_Bt = (float*)_aligned_malloc(matrix_B_rows*matrix_B_cols * sizeof(float), 32);
		const float* filter_slice_ptr, *filter_row_ptr, *filter_pix_ptr;
		int out_n, out_h, out_w, kn, kh, kw;
		float* cp_dst_ptr;
		float* matrix_C;
		int out_HW = out_H*out_W;
		int out_NHW = out_N*out_HW;
		double t1, t2, t3, t4, t5; 
		int need_allocate_tmp_out;

		int chunk_size = (out_NHW + thread_count - 1) / thread_count;
		int gemm_per_row = (matrix_A_rows + thread_count - 1) / thread_count;
		int cur_row_num;
		int* in_offsets = (int*)malloc(out_NHW * sizeof(int));
		int* matA_offsets = (int*)malloc(out_NHW * sizeof(int));
		int* out_offsets = (int*)malloc(out_NHW * sizeof(int));
		int* matC_offsets = (int*)malloc(out_NHW * sizeof(int));
		int idx = 0;
		for (out_n = 0; out_n < out_N; out_n++)
		{
			for (out_h = 0; out_h < out_H; out_h++)
			{
				for (out_w = 0; out_w < out_W; out_w++)
				{
					in_offsets[idx] = out_n*in_sliceStep + out_h*in_widthStep_mul_stride_H + out_w*in_pixelStep_mul_stride_W;
					matA_offsets[idx] = (out_n*out_HW + out_h*out_W + out_w)*matrix_A_cols;
					out_offsets[idx] = out_n*out_sliceStep + out_h*out_widthStep + out_w*out_pixelStep;
					matC_offsets[idx] = (out_n*out_HW + out_h*out_W + out_w)*matrix_B_cols;
					idx++;
				}
			}
		}

		t1 = omp_get_wtime();
		need_allocate_tmp_out = (out_pixelStep != filter_N) || (out_pixelStep*out_W != out_widthStep) || (out_widthStep*out_H != out_sliceStep);
		if (need_allocate_tmp_out)
		{
			matrix_C = (float*)_aligned_malloc(out_HW*filter_N * sizeof(float), 32);
		}
		else
		{
			matrix_C = out_tensor4D_data;
		}
		t2 = omp_get_wtime();

		cp_dst_ptr = matrix_Bt;
		for (kn = 0, filter_slice_ptr = filters_data; kn < filter_N; kn++, filter_slice_ptr += filter_sliceStep)
		{
			for (kh = 0, filter_row_ptr = filter_slice_ptr; kh < filter_H; kh++, filter_row_ptr += filter_widthStep)
			{
				for (kw = 0, filter_pix_ptr = filter_row_ptr; kw < filter_W; kw++, filter_pix_ptr += filter_pixelStep)
				{
					memcpy(cp_dst_ptr, filter_pix_ptr, sizeof(float)*in_C);
					cp_dst_ptr += in_C;
				}
			}
		}

#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
		for (idx = 0; idx < out_NHW; idx++)
		{
			const float* in_pix_ptr, *cur_in_row_ptr, *cur_in_pix_ptr;
			float* matrix_A_row_ptr, *matrix_A_col_ptr;
			in_pix_ptr = in_tensor4D_data + in_offsets[idx];
			matrix_A_row_ptr = matrix_A + matA_offsets[idx];
			for (kh = 0, cur_in_row_ptr = in_pix_ptr, matrix_A_col_ptr = matrix_A_row_ptr;
				kh < filter_H;
				kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep)
			{
				for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < filter_W; kw++, cur_in_pix_ptr += dilate_W_mul_in_pixStep)
				{
					memcpy(matrix_A_col_ptr, cur_in_pix_ptr, sizeof(float)*in_C);
					matrix_A_col_ptr += in_C;
				}
			}
		}

		t3 = omp_get_wtime();
		/*gemm*/
		if (0 && openblas_get_num_threads() == 1 && gemm_per_row >= 4)
		{
#pragma omp parallel for schedule(static) num_threads(thread_count)
			for (idx = 0; idx < thread_count; idx++)
			{
				cur_row_num = __min(matrix_A_rows, gemm_per_row*(idx + 1)) - gemm_per_row*idx;
				if (cur_row_num > 0)
				{
					cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, cur_row_num, matrix_B_cols, matrix_A_cols, 1,
						matrix_A + gemm_per_row*idx*matrix_A_cols, matrix_A_cols,
						matrix_Bt, matrix_A_cols, 0.0f, matrix_C + gemm_per_row*idx*matrix_B_cols, matrix_B_cols);
				}
			}
		}
		else
		{
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
				matrix_Bt, matrix_A_cols, 0.0f, matrix_C, matrix_B_cols);
		}
		t4 = omp_get_wtime();
		if (need_allocate_tmp_out)
		{
			/*   col2im      */
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
			for (idx = 0; idx < out_NHW; idx++)
			{
				float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];
				float* matrix_C_row_ptr = matrix_C + matC_offsets[idx];
				memcpy(out_pix_ptr, matrix_C_row_ptr, sizeof(float)*matrix_B_cols);
			}
		}
		else
			matrix_C += out_sliceStep;
		t5 = omp_get_wtime();

		if (need_allocate_tmp_out)
			_aligned_free(matrix_C);
		_aligned_free(matrix_A);
		_aligned_free(matrix_Bt);
		free(in_offsets);
		free(out_offsets);
		free(matA_offsets);
		free(matC_offsets);
		if (filter_H == 3 && filter_W == 3 && filter_C == 3)
		{
			printf("total: %.3f ms, alloc %.3f ms, makeA: %.3f ms, gemm: %.3f ms, copy_C: %.3f ms\n", 1000 * (t5 - t1), 1000 * (t2 - t1),
				1000 * (t3 - t2), 1000 * (t4 - t3), 1000 * (t5 - t4));
		}
	}

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif

#endif