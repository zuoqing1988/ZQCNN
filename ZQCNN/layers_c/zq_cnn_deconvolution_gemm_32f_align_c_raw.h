
void zq_cnn_deconv_with_padding_gemm_32f_k2s2(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
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
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep,
	int pad_top, //pad_top must <= 1
	int pad_bottom,
	int pad_left, //pad_left must <= 1
	int pad_right,
	void** buffer,
	__int64* buffer_len
)
{
	/************** image to col **************/
	int matrix_A_cols = in_pixelStep;
	int matrix_A_rows = in_H*in_W;
	int matrix_B_cols = filter_N;
	int matrix_B_rows = filter_pixelStep;
#if __ARM_NEON
#if !__ARM_NEON_FP16
	int padK = (matrix_A_cols + 3) >> 2 << 2;
#endif
#else
#if ZQ_CNN_USE_SSETYPE > ZQ_CNN_SSETYPE_AVX
	int padK = (matrix_A_cols + 7) >> 3 << 3;
#elif ZQ_CNN_USE_SSETYPE > ZQ_CNN_SSETYPE_SSE
	int padK = (matrix_A_cols + 3) >> 2 << 2;
#else
	int padK = matrix_A_cols;
#endif
#endif

	__int64 need_A_buffer_len_align32 = 0;
	__int64 need_B_buffer_len_align32 = (matrix_B_cols*matrix_B_rows * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 need_C_buffer_len_align32 = (matrix_A_rows*filter_N * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 total_need_buffer_len;
	int filter_pixelStep2 = filter_pixelStep << 2;
	int filter_pixelStep3 = filter_pixelStep + filter_pixelStep2;
	int need_allocate_A = 0;
	zq_base_type* matrix_A = 0,*matrix_Bt0 = 0,*matrix_Bt1 = 0,*matrix_Bt2 = 0,*matrix_Bt3 = 0;
	zq_base_type* matrix_C[2][2] = { 0 };
	const zq_base_type* in_slice_ptr,*in_row_ptr,*in_pix_ptr, *cp_src_ptr, *cur_in_row_ptr,*filter_slice_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *cp_dst_ptr, *matrix_A_row_ptr, *matrix_A_col_ptr;
	int out_n,out_h,out_w,out_c,in_h,in_w,pp,fn,need_in_h_idx,need_in_w_idx;
	int kh, kw, real_in_h_idx, real_in_w_idx;

	need_allocate_A = (in_pixelStep * in_W != in_widthStep) || (in_widthStep*in_H != in_sliceStep);
	if (need_allocate_A)
		need_A_buffer_len_align32 = (matrix_A_rows*padK * sizeof(zq_base_type) + 31) / 32 * 32;
	total_need_buffer_len = need_A_buffer_len_align32 + need_B_buffer_len_align32*4 + need_C_buffer_len_align32*4;
	if (buffer == 0)
	{
		if(need_allocate_A)
			matrix_A = (zq_base_type*)_aligned_malloc(need_A_buffer_len_align32, 32);
		matrix_Bt0 = (zq_base_type*)_aligned_malloc(need_B_buffer_len_align32, 32);
		matrix_Bt1 = (zq_base_type*)_aligned_malloc(need_B_buffer_len_align32, 32);
		matrix_Bt2 = (zq_base_type*)_aligned_malloc(need_B_buffer_len_align32, 32);
		matrix_Bt3 = (zq_base_type*)_aligned_malloc(need_B_buffer_len_align32, 32);
		matrix_C[0][0] = (zq_base_type*)_aligned_malloc(need_C_buffer_len_align32, 32);
		matrix_C[0][1] = (zq_base_type*)_aligned_malloc(need_C_buffer_len_align32, 32);
		matrix_C[1][0] = (zq_base_type*)_aligned_malloc(need_C_buffer_len_align32, 32);
		matrix_C[1][1] = (zq_base_type*)_aligned_malloc(need_C_buffer_len_align32, 32);
	}
	else
	{
		if (*buffer_len < total_need_buffer_len)
		{
			_aligned_free(*buffer);
			*buffer = _aligned_malloc(total_need_buffer_len, 32);
			*buffer_len = total_need_buffer_len;
		}
		matrix_A = (zq_base_type*)(*buffer);
		matrix_Bt0 = (zq_base_type*)(((char*)matrix_A) + need_A_buffer_len_align32);
		matrix_Bt1 = (zq_base_type*)(((char*)matrix_Bt0) + need_B_buffer_len_align32);
		matrix_Bt2 = (zq_base_type*)(((char*)matrix_Bt1) + need_B_buffer_len_align32);
		matrix_Bt3 = (zq_base_type*)(((char*)matrix_Bt2) + need_B_buffer_len_align32);
		matrix_C[0][0] = (zq_base_type*)(((char*)matrix_Bt3) + need_B_buffer_len_align32);
		matrix_C[0][1] = (zq_base_type*)(((char*)matrix_C[0][0]) + need_C_buffer_len_align32);
		matrix_C[1][0] = (zq_base_type*)(((char*)matrix_C[0][1]) + need_C_buffer_len_align32);
		matrix_C[1][1] = (zq_base_type*)(((char*)matrix_C[1][0]) + need_C_buffer_len_align32);
	}

	for (fn = 0, filter_slice_ptr = filters_data; fn < filter_N; fn++, filter_slice_ptr += filter_sliceStep)
	{
		memcpy(matrix_Bt0 + fn*filter_pixelStep, filter_slice_ptr, sizeof(zq_base_type)*filter_pixelStep);
		memcpy(matrix_Bt1 + fn*filter_pixelStep, filter_slice_ptr + filter_pixelStep, sizeof(zq_base_type)*filter_pixelStep);
		memcpy(matrix_Bt2 + fn*filter_pixelStep, filter_slice_ptr + filter_widthStep, sizeof(zq_base_type)*filter_pixelStep);
		memcpy(matrix_Bt3 + fn*filter_pixelStep, filter_slice_ptr + filter_widthStep+filter_pixelStep, sizeof(zq_base_type)*filter_pixelStep);
	}

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		if (need_allocate_A)
		{
			matrix_A_row_ptr = matrix_A;

			for (in_h = 0, in_row_ptr = in_slice_ptr; in_h < in_H; in_h++, in_row_ptr += in_widthStep)
			{
				for (in_w = 0, in_pix_ptr = in_row_ptr; in_w < in_W; in_w++, in_pix_ptr += in_pixelStep)
				{
					cur_in_row_ptr = in_pix_ptr;
					matrix_A_col_ptr = matrix_A_row_ptr;
					for (pp = 0, cp_src_ptr = cur_in_row_ptr, cp_dst_ptr = matrix_A_col_ptr;
						pp < filter_pixelStep;
						pp += zq_mm_align_size, cp_src_ptr += zq_mm_align_size, cp_dst_ptr += zq_mm_align_size)
						zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(cp_src_ptr));
					for (; pp < padK; pp++, cp_dst_ptr++)
						*cp_dst_ptr = 0;
					matrix_A_row_ptr += padK;
				}
			}
		}
		else
			matrix_A = in_slice_ptr;

		/*gemm*/
#if __ARM_NEON && ZQ_CNN_USE_ZQ_GEMM && ZQ_CNN_USE_BLAS_GEMM
		if (0 == zq_gemm_32f_AnoTrans_Btrans_special(matrix_A_rows, matrix_B_cols, matrix_A_cols, matrix_A, matrix_A_cols,
			matrix_Bt0, matrix_A_cols, matrix_C[0][0], matrix_B_cols))
		{
			zq_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
				matrix_Bt0, matrix_A_cols, 0.0f, matrix_C[0][0], matrix_B_cols);
		}
		if (0 == zq_gemm_32f_AnoTrans_Btrans_special(matrix_A_rows, matrix_B_cols, matrix_A_cols, matrix_A, matrix_A_cols,
			matrix_Bt1, matrix_A_cols, matrix_C[0][1], matrix_B_cols))
		{
			zq_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
				matrix_Bt1, matrix_A_cols, 0.0f, matrix_C[0][1], matrix_B_cols);
		}
		if (0 == zq_gemm_32f_AnoTrans_Btrans_special(matrix_A_rows, matrix_B_cols, matrix_A_cols, matrix_A, matrix_A_cols,
			matrix_Bt2, matrix_A_cols, matrix_C[1][0], matrix_B_cols))
		{
			zq_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
				matrix_Bt2, matrix_A_cols, 0.0f, matrix_C[1][0], matrix_B_cols);
		}
		if (0 == zq_gemm_32f_AnoTrans_Btrans_special(matrix_A_rows, matrix_B_cols, matrix_A_cols, matrix_A, matrix_A_cols,
			matrix_Bt3, matrix_A_cols, matrix_C[1][1], matrix_B_cols))
		{
			zq_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
				matrix_Bt3, matrix_A_cols, 0.0f, matrix_C[1][1], matrix_B_cols);
		}
#else
		zq_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
			matrix_Bt0, filter_pixelStep, 0.0f, matrix_C[0][0], matrix_B_cols);
		zq_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
			matrix_Bt1, filter_pixelStep, 0.0f, matrix_C[0][1], matrix_B_cols);
		zq_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
			matrix_Bt2, filter_pixelStep, 0.0f, matrix_C[1][0], matrix_B_cols);
		zq_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
			matrix_Bt3, filter_pixelStep, 0.0f, matrix_C[1][1], matrix_B_cols);
#endif

		need_in_h_idx = -pad_top;
		for (out_h = 0, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, out_row_ptr += out_widthStep, need_in_h_idx++)
		{
			kh = need_in_h_idx & 1;
			real_in_h_idx = (need_in_h_idx + kh) >> 1;
			if (real_in_h_idx < 0 || real_in_h_idx >= in_H)
			{
				memset(out_row_ptr, 0, sizeof(zq_base_type)*out_pixelStep*out_W);
				continue;
			}
			need_in_w_idx = -pad_left;
			for (out_w = 0, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, out_pix_ptr += out_pixelStep, need_in_w_idx++)
			{
				kw = need_in_w_idx & 1;
				real_in_w_idx = (need_in_w_idx + kw) >> 1;
				if (real_in_w_idx >= 0 && real_in_w_idx < in_W)
				{
					memcpy(out_pix_ptr, matrix_C[kh][kw] + (real_in_h_idx*in_W+real_in_w_idx)*filter_N, sizeof(zq_base_type)*filter_N);
					for (out_c = filter_N; out_c < out_pixelStep; out_c++)
						out_pix_ptr[out_c] = 0;
				}
				else
					memset(out_pix_ptr, 0, sizeof(zq_base_type)*out_pixelStep);
			}
		}
	}

	if (buffer == 0)
	{
		if(need_allocate_A)
			_aligned_free(matrix_A);
		
		_aligned_free(matrix_Bt0);
		_aligned_free(matrix_Bt1);
		_aligned_free(matrix_Bt2);
		_aligned_free(matrix_Bt3);
		_aligned_free(matrix_C[0][0]);
		_aligned_free(matrix_C[0][1]);
		_aligned_free(matrix_C[1][0]);
		_aligned_free(matrix_C[1][1]);
	}
}

