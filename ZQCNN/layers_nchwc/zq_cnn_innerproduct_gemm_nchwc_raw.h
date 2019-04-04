/*in_pixStep can be different with filter_pixStep,
and the aligned channels should be set to zero*/
void zq_cnn_innerproduct_gemm_nchwc_general(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* filters_data,
	int filter_N,
	int filter_H,
	int filter_W,
	int filter_C, // must be in_C
	int filter_widthStep,
	int filter_sliceStep,
	int filter_imStep,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_C,	// must be filter_N
	int out_widthStep,
	int out_sliceStep,
	int out_imStep,
#if WITH_BIAS
	const zq_base_type* bias,
#endif
#if WITH_PRELU
	const zq_base_type* slope,
#endif
	void** buffer,
	__int64* buffer_len
)
{
	/************** image to col **************/
	int align_C = (in_C + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	int filter_pixStep_mul_filter_W = zq_mm_align_size*filter_W;
	int in_sliceStep2 = in_sliceStep * 2;
	int in_sliceStep3 = in_sliceStep * 3;
	int in_sliceStep4 = in_sliceStep * 4;
	int in_sliceStep5 = in_sliceStep * 5;
	int in_sliceStep6 = in_sliceStep * 6;
	int in_sliceStep7 = in_sliceStep * 7;
	int in_sliceStep8 = in_sliceStep * 8;
	int out_sliceStep2 = out_sliceStep * 2;
	int out_sliceStep3 = out_sliceStep * 3;
	int out_sliceStep4 = out_sliceStep * 4;
	int out_sliceStep5 = out_sliceStep * 5;
	int out_sliceStep6 = out_sliceStep * 6;
	int out_sliceStep7 = out_sliceStep * 7;
	int out_sliceStep8 = out_sliceStep * 8;
	int matrix_A_cols = filter_H*filter_W*align_C;
	int matrix_A_rows = out_N;
	int matrix_B_cols = filter_N;
	int matrix_B_rows = filter_H*filter_W*align_C;
	int matrix_B_cols2 = matrix_B_cols * 2;
	int matrix_B_cols3 = matrix_B_cols * 3;
	int matrix_B_cols4 = matrix_B_cols * 4;
	int out_NHW = out_N;
	int compactA = in_imStep == in_H*in_W*align_C;
	int compactC = out_imStep == filter_N;
	zq_base_type* matrix_A = 0;
	const zq_base_type* matrix_Bt = filters_data;
	const zq_base_type *in_im_ptr, *in_slice_ptr, *cur_in_row_ptr, *cur_in_pix_ptr;
	int out_n, out_h, out_w, kn, kh, kw, kc, i;
	zq_base_type* matrix_A_row_ptr, *matrix_A_col_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_im_ptr;
	zq_base_type* out_slice_ptr0, *out_slice_ptr1, *out_slice_ptr2, *out_slice_ptr3, *out_slice_ptr4, *out_slice_ptr5, *out_slice_ptr6, *out_slice_ptr7;
	zq_base_type* matrix_C = 0, *matrix_C_row_ptr, *matrix_C_col_ptr;
#if WITH_BIAS
	register zq_mm_type bias_v;
	const zq_base_type* bias_ptr;
	register zq_mm_type bias_v0, bias_v1, bias_v2, bias_v3, bias_v4, bias_v5, bias_v6, bias_v7;
#endif
#if WITH_PRELU
	const zq_base_type* slope_ptr;
	register zq_mm_type slope_v;
	register zq_mm_type b0, b1, b2, b3, b4, b5, b6, b7;
	register zq_mm_type c0, c1, c2, c3, c4, c5, c6, c7;
	register zq_mm_type slope_v0, slope_v1, slope_v2, slope_v3, slope_v4, slope_v5, slope_v6, slope_v7;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif
	register zq_mm_type a0, a1, a2, a3, a4, a5, a6, a7;
	float val, val1, val2;
	__int64 need_A_buffer_len_align32 = compactA ? 0 : (matrix_A_rows*matrix_A_cols * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 need_C_buffer_len_align32 = compactC ? 0 : (out_NHW*filter_N * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 total_need_buffer_len;

	double t1, t2, t3, t4, t5;
	t1 = omp_get_wtime();
	total_need_buffer_len = need_A_buffer_len_align32 + need_C_buffer_len_align32;
	if (buffer == 0)
	{
		if (compactA)
			matrix_A = (zq_base_type*)in_tensor4D_data;
		else
		{
			matrix_A = (zq_base_type*)_aligned_malloc(need_A_buffer_len_align32, 32);
		}
		if (compactC)
			matrix_C = out_tensor4D_data;
		else
		{
			matrix_C = (zq_base_type*)_aligned_malloc(need_C_buffer_len_align32, 32);
		}
	}
	else
	{
		if (*buffer_len < total_need_buffer_len)
		{
			_aligned_free(*buffer);
			*buffer = _aligned_malloc(total_need_buffer_len, 32);
			*buffer_len = total_need_buffer_len;
		}
		if (compactA)
			matrix_A = (zq_base_type*)in_tensor4D_data;
		else
			matrix_A = *buffer;
		if (compactC)
			matrix_C = out_tensor4D_data;
		else
			matrix_C = (zq_base_type*)((char*)(*buffer) + need_A_buffer_len_align32);
	}
	t2 = omp_get_wtime();

	if (compactA == 0)
	{
		matrix_A_row_ptr = matrix_A;
		for (out_n = 0, in_im_ptr = in_tensor4D_data;
			out_n < out_N;
			out_n++, in_im_ptr += in_imStep)
		{
			matrix_A_col_ptr = matrix_A_row_ptr;
			for (kh = 0, cur_in_row_ptr = in_im_ptr;
				kh < filter_H;
				kh++, cur_in_row_ptr += in_widthStep)
			{
				for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < filter_W; kw++, cur_in_pix_ptr += zq_mm_align_size)
				{
					for (kc = 0, in_slice_ptr = cur_in_pix_ptr; kc < in_C;
						kc += zq_mm_align_size, in_slice_ptr += in_sliceStep)
					{
						zq_mm_store_ps(matrix_A_col_ptr, zq_mm_load_ps(in_slice_ptr));
						matrix_A_col_ptr += zq_mm_align_size;
					}
				}
			}
			matrix_A_row_ptr += matrix_A_cols;
		}
	}
	t3 = omp_get_wtime();
	/*gemm*/
#if __ARM_NEON && ZQ_CNN_USE_ZQ_GEMM && ZQ_CNN_USE_BLAS_GEMM
	if (0 == zq_gemm_32f_AnoTrans_Btrans_special(matrix_A_rows, matrix_B_cols, matrix_A_cols, matrix_A, matrix_A_cols,
		matrix_Bt, matrix_A_cols, matrix_C, matrix_B_cols))
	{
		zq_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
			matrix_Bt, matrix_A_cols, 0.0f, matrix_C, matrix_B_cols);
	}
#else
	zq_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
		matrix_Bt, matrix_A_cols, 0.0f, matrix_C, matrix_B_cols);
#endif
	t4 = omp_get_wtime();

	/*   col2im      */
	if (compactC)
	{
#if WITH_BIAS || WITH_PRELU
#include "zq_cnn_innerproduct_gemm_nchwc_col2im.h"
#endif
	}
	else
	{
#include "zq_cnn_innerproduct_gemm_nchwc_col2im.h"
	}
	t5 = omp_get_wtime();

	if (buffer == 0)
	{
		if(compactA == 0)
			_aligned_free(matrix_A);
		if(compactC == 0)
			_aligned_free(matrix_C);
	}
}


void zq_cnn_innerproduct_nchwc_noborder(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_HWC,
	const zq_base_type* filters_data,
	int filter_N,
	zq_base_type* out_tensor4D_data,
	int out_sliceStep
#if WITH_BIAS
	,const zq_base_type* bias
#endif
#if WITH_PRELU
	,const zq_base_type* slope
#endif
)
{
	register zq_mm_type sum_vec;
	zq_base_type result[zq_mm_align_size << 2];
	zq_base_type* q = (zq_base_type*)(((long long)result + (zq_mm_align_size << 2) - 1) & zq_mm_bitor_longlong);
	int out_n, out_c, in_hwc;
	const zq_base_type* in_slice_ptr;
	const zq_base_type* filter_slice_ptr, *filter_c_ptr;
	zq_base_type* out_slice_ptr, *out_c_ptr;
	const zq_base_type* cur_in_c_ptr;

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < in_N; out_n++, in_slice_ptr += in_HWC, out_slice_ptr += out_sliceStep)
	{
		for (out_c = 0, out_c_ptr = out_slice_ptr, filter_slice_ptr = filters_data;
			out_c < filter_N;
			out_c++, out_c_ptr++, filter_slice_ptr += in_HWC)
		{
#if WITH_BIAS
			sum_vec = zq_mm_set1_ps(bias[out_c]);
#else
			sum_vec = zq_mm_setzero_ps();
#endif
			for (in_hwc = 0, cur_in_c_ptr = in_slice_ptr, filter_c_ptr = filter_slice_ptr;
				in_hwc < in_HWC;
				in_hwc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, filter_c_ptr += zq_mm_align_size)
			{
				sum_vec = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(filter_c_ptr), sum_vec);
			}

			zq_mm_store_ps(q, sum_vec);
			*out_c_ptr = zq_final_sum_q;
#if WITH_PRELU
			if(*out_c_ptr < 0)
				*out_c_ptr *= slope[out_c];
#endif
		}
	}
}

