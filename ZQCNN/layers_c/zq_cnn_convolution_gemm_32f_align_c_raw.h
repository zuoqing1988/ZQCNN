/*in_pixStep must be equal to filter_pixStep,
and the aligned channels should be set to zero*/
void zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep(
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
	__int64* buffer_len
)
{
	/************** image to col **************/
	int in_widthStep_mul_stride_H = in_widthStep*stride_H;
	int in_pixelStep_mul_stride_W = in_pixelStep*stride_W;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
	int filter_pixStep_mul_filter_W = filter_pixelStep*filter_W;
	int matrix_A_cols = filter_sliceStep;
	int matrix_A_rows = out_H*out_W;
	int matrix_B_cols = filter_N;
	int matrix_B_rows = filter_sliceStep;
	__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols*sizeof(float) + 31) / 32 * 32;
	__int64 need_C_buffer_len_align32 = 0;
	__int64 total_need_buffer_len;
	float* matrix_A = 0;
	const float* in_row_ptr, *in_pix_ptr, *cur_in_row_ptr, *cur_in_pix_ptr;
	int out_n, out_h, out_w, kh, kw, pp;
	float* matrix_A_row_ptr, *matrix_A_col_ptr, *cp_dst_ptr;
	const float* cp_src_ptr;
	float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
	const float* in_slice_ptr;
	const float* matrix_Bt = filters_data;
	float* matrix_C = 0, *matrix_C_row_ptr;
	int out_row_idx;
	int out_HW = out_H*out_W;
	double t1, t2, t3, t4, t5, t6, alloc_time, make_A_time = 0, gemm_time = 0;
	t1 = omp_get_wtime();
	int need_allocate_tmp_out = (out_pixelStep != filter_N) || (out_pixelStep*out_W != out_widthStep) || (out_widthStep*out_H != out_sliceStep);
	if (need_allocate_tmp_out)
	{
		need_C_buffer_len_align32 = (out_HW*filter_N * sizeof(float) + 31) / 32 * 32;
	}
	else
	{
		need_C_buffer_len_align32 = 0;
		matrix_C = out_tensor4D_data;
	}
	total_need_buffer_len = need_A_buffer_len_align32 + need_C_buffer_len_align32;
	if (buffer == 0)
	{
		matrix_A = _aligned_malloc(need_A_buffer_len_align32, 32);
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
		matrix_A = (float*)(*buffer);
		if(need_allocate_tmp_out)
			matrix_C = (float*)((char*)(*buffer) + need_A_buffer_len_align32);
	}

	t2 = omp_get_wtime();
	alloc_time = t2 - t1;
	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		t3 = omp_get_wtime();
		matrix_A_row_ptr = matrix_A;
		if (dilation_W == 1)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
				{
					for (kh = 0, cur_in_row_ptr = in_pix_ptr, matrix_A_col_ptr = matrix_A_row_ptr;
						kh < filter_H;
						kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep, matrix_A_col_ptr += filter_pixStep_mul_filter_W)
					{
						//memcpy(matrix_A_col_ptr, cur_in_row_ptr, sizeof(float)*filter_pixStep_mul_filter_W);
						for (pp = 0, cp_src_ptr = cur_in_row_ptr, cp_dst_ptr = matrix_A_col_ptr;
							pp < filter_pixStep_mul_filter_W;
							pp += zq_mm_align_size, cp_src_ptr += zq_mm_align_size, cp_dst_ptr += zq_mm_align_size)
							zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(cp_src_ptr));

					}
					matrix_A_row_ptr += matrix_A_cols;
				}
			}
		}
		else
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
				{
					for (kh = 0, cur_in_row_ptr = in_pix_ptr, matrix_A_col_ptr = matrix_A_row_ptr;
						kh < filter_H;
						kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep, matrix_A_col_ptr += filter_pixStep_mul_filter_W)
					{
						cp_dst_ptr = matrix_A_col_ptr;
						for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < filter_W; kw++, cur_in_pix_ptr += dilate_W_mul_in_pixStep)
						{
							for (pp = 0, cp_src_ptr = cur_in_pix_ptr;
								pp < filter_C;
								pp += zq_mm_align_size, cp_src_ptr += zq_mm_align_size, cp_dst_ptr += zq_mm_align_size)
								zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(cp_src_ptr));
						}

					}
					matrix_A_row_ptr += matrix_A_cols;
				}
			}
		}
		
		t4 = omp_get_wtime();
		make_A_time += t4 - t3;
		/*gemm*/
		zq_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
			matrix_Bt, matrix_A_cols, 0.0f, matrix_C, matrix_B_cols);
		t5 = omp_get_wtime();
		gemm_time += t5 - t4;
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

	}

	if (buffer == 0)
	{
		_aligned_free(matrix_A);
		if (need_allocate_tmp_out)
			_aligned_free(matrix_C);
	}
	t6 = omp_get_wtime();
	/*if (filter_H == 1 && filter_W == 1 && filter_C == 4)
	{
		printf("gemm_same_pixstep total: %.3f ms, alloc %.3f ms, makeA: %.3f ms, gemm: %.3f ms\n", 1000 * (t6 - t1), 1000 * alloc_time,
			1000 * make_A_time, 1000 * gemm_time);
	}*/
}

/*in_pixStep must be equal to filter_pixStep,
and the aligned channels should be set to zero*/
void zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_C4(
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
	__int64* buffer_len
)
{
	/************** image to col **************/
	int in_widthStep_mul_stride_H = in_widthStep*stride_H;
	int in_pixelStep_mul_stride_W = in_pixelStep*stride_W;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
	int filter_pixStep_mul_filter_W = filter_pixelStep*filter_W;
	int filter_C_mul_filter_W = filter_C*filter_W;
	int matrix_A_cols = filter_sliceStep;
	int matrix_A_rows = out_H*out_W;
	int matrix_B_cols = filter_N;
	int matrix_B_rows = filter_sliceStep;
	__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(float) + 31) / 32 * 32;
	__int64 need_C_buffer_len_align32 = 0;
	__int64 total_need_buffer_len;
	float* matrix_A = 0;
	const float* in_row_ptr, *in_pix_ptr, *cur_in_row_ptr, *cur_in_pix_ptr;
	int out_n, out_h, out_w, kh, kw, pp;
	float* matrix_A_row_ptr, *matrix_A_col_ptr, *cp_dst_ptr;
	const float* cp_src_ptr;
	float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
	const float* in_slice_ptr;
	const float* matrix_Bt = filters_data;
	float* matrix_C = 0, *matrix_C_row_ptr = 0;
	int out_row_idx;
	int out_HW = out_H*out_W;
	double t1, t2, t3, t4, t5, t6, alloc_time, make_A_time = 0, gemm_time = 0;
	t1 = omp_get_wtime();
	int need_allocate_tmp_out = (out_pixelStep != filter_N) || (out_pixelStep*out_W != out_widthStep) || (out_widthStep*out_H != out_sliceStep);
	if (need_allocate_tmp_out)
	{
		need_C_buffer_len_align32 = (out_HW*filter_N * sizeof(float) + 31) / 32 * 32;
	}
	else
	{
		need_C_buffer_len_align32 = 0;
		matrix_C = out_tensor4D_data;
	}
	total_need_buffer_len = need_A_buffer_len_align32 + need_C_buffer_len_align32;
	if (buffer == 0)
	{
		matrix_A = _aligned_malloc(need_A_buffer_len_align32, 32);
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
		matrix_A = (float*)(*buffer);
		if (need_allocate_tmp_out)
			matrix_C = (float*)((char*)(*buffer) + need_A_buffer_len_align32);
	}


	t2 = omp_get_wtime();
	alloc_time = t2 - t1;
	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		t3 = omp_get_wtime();
		matrix_A_row_ptr = matrix_A;
		if (dilation_W == 1)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
				{
					for (kh = 0, cur_in_row_ptr = in_pix_ptr, matrix_A_col_ptr = matrix_A_row_ptr;
						kh < filter_H;
						kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep, matrix_A_col_ptr += filter_C_mul_filter_W)
					{
						for (pp = 0, cp_src_ptr = cur_in_row_ptr, cp_dst_ptr = matrix_A_col_ptr;
							pp < filter_pixStep_mul_filter_W;
							pp += filter_pixelStep, cp_src_ptr += filter_pixelStep, cp_dst_ptr += 4)
							zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(cp_src_ptr));

					}
					matrix_A_row_ptr += matrix_A_cols;
				}
			}
		}
		else
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
				{
					for (kh = 0, cur_in_row_ptr = in_pix_ptr, matrix_A_col_ptr = matrix_A_row_ptr;
						kh < filter_H;
						kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep, matrix_A_col_ptr += filter_C_mul_filter_W)
					{
						cp_dst_ptr = matrix_A_col_ptr;
						for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < filter_W; kw++, cur_in_pix_ptr += dilate_W_mul_in_pixStep)
						{
							zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(cur_in_pix_ptr));
							cp_dst_ptr += 4;
						}

					}
					matrix_A_row_ptr += matrix_A_cols;
				}
			}
		}

		t4 = omp_get_wtime();
		make_A_time += t4 - t3;
		/*gemm*/

		zq_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
			matrix_Bt, matrix_A_cols, 0.0f, matrix_C, matrix_B_cols);

		t5 = omp_get_wtime();
		gemm_time += t5 - t4;
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

	}

	if(buffer == 0)
	{ 
		_aligned_free(matrix_A);
		_aligned_free(matrix_C);
	}
	t6 = omp_get_wtime();
	/*if (filter_H == 1 && filter_W == 1 && filter_C == 4)
	{
	printf("gemm_same_pixstep total: %.3f ms, alloc %.3f ms, makeA: %.3f ms, gemm: %.3f ms\n", 1000 * (t6 - t1), 1000 * alloc_time,
	1000 * make_A_time, 1000 * gemm_time);
	}*/
}


/*in_pixStep must be equal to filter_pixStep,
and the aligned channels should be set to zero*/
void zq_cnn_conv_no_padding_gemm_32f_align_same_pixstep_batch(
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
	__int64* buffer_len
)
{
	/************** image to col **************/
	int in_widthStep_mul_stride_H = in_widthStep*stride_H;
	int in_pixelStep_mul_stride_W = in_pixelStep*stride_W;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
	int filter_pixStep_mul_filter_W = filter_pixelStep*filter_W;
	int matrix_A_cols = filter_sliceStep;
	int matrix_A_rows = out_N*out_H*out_W;
	int matrix_B_cols = filter_N;
	int matrix_B_rows = filter_sliceStep;
	__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(float) + 31) / 32 * 32;
	__int64 need_C_buffer_len_align32 = 0;
	__int64 total_need_buffer_len;
	float* matrix_A = 0;
	const float* in_row_ptr, *in_pix_ptr, *cur_in_row_ptr, *cur_in_pix_ptr, *cp_src_ptr;
	float *cp_dst_ptr;
	int out_n, out_h, out_w, kh, kw, pp;
	float* matrix_A_row_ptr, *matrix_A_col_ptr;
	float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
	const float* in_slice_ptr;
	const float* matrix_Bt = filters_data;
	float* matrix_C = 0, *matrix_C_row_ptr;
	int out_row_idx;
	int out_HW = out_H*out_W;
	int out_NHW = out_HW*out_N;
	double t1, t2, t3, t4, t5;
	int need_allocate_tmp_out;
	t1 = omp_get_wtime();
	need_allocate_tmp_out = (out_pixelStep != filter_N) || (out_pixelStep*out_W != out_widthStep) || (out_widthStep*out_H != out_sliceStep);
	if (need_allocate_tmp_out)
	{
		need_C_buffer_len_align32 = (out_NHW*filter_N * sizeof(float) + 31) / 32 * 32;
	}
	else
	{
		need_C_buffer_len_align32 = 0;
		matrix_C = out_tensor4D_data;
	}
	total_need_buffer_len = need_A_buffer_len_align32 + need_C_buffer_len_align32;
	if (buffer == 0)
	{
		matrix_A = _aligned_malloc(need_A_buffer_len_align32, 32);
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
		matrix_A = (float*)(*buffer);
		if (need_allocate_tmp_out)
			matrix_C = (float*)((char*)(*buffer) + need_A_buffer_len_align32);
	}


	t2 = omp_get_wtime();
	matrix_A_row_ptr = matrix_A;
	if (dilation_W == 1)
	{
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
						kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep, matrix_A_col_ptr += filter_pixStep_mul_filter_W)
					{
						//memcpy(matrix_A_col_ptr, cur_in_row_ptr, sizeof(float)*filter_pixStep_mul_filter_W);
						for (pp = 0, cp_src_ptr = cur_in_row_ptr, cp_dst_ptr = matrix_A_col_ptr;
							pp < filter_pixStep_mul_filter_W;
							pp += zq_mm_align_size, cp_src_ptr += zq_mm_align_size, cp_dst_ptr += zq_mm_align_size)
							zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(cp_src_ptr));
					}

					matrix_A_row_ptr += matrix_A_cols;
				}
			}
		}
	}
	else
	{
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
						kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep, matrix_A_col_ptr += filter_pixStep_mul_filter_W)
					{
						cp_dst_ptr = matrix_A_col_ptr;
						for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr;
							kw < filter_W;
							kw++, cur_in_pix_ptr += dilate_W_mul_in_pixStep)
						{
							for (pp = 0, cp_src_ptr = cur_in_pix_ptr;
								pp < filter_C;
								pp += zq_mm_align_size, cp_src_ptr += zq_mm_align_size, cp_dst_ptr += zq_mm_align_size)
								zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(cp_src_ptr));
						}
					}

					matrix_A_row_ptr += matrix_A_cols;
				}
			}
		}
	}
	

	t3 = omp_get_wtime();
	/*gemm*/
	zq_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
		matrix_Bt, matrix_A_cols, 0, matrix_C, matrix_B_cols);
	t4 = omp_get_wtime();


	if (need_allocate_tmp_out)
	{
		/*   col2im      */
		out_row_idx = 0;
		matrix_C_row_ptr = matrix_C;
		for (out_n = 0, out_slice_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_slice_ptr += out_sliceStep)
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
	if (buffer == 0)
	{
		_aligned_free(matrix_A);
		_aligned_free(matrix_C);
	}
	t5 = omp_get_wtime();
	//if (filter_H == 3 && filter_W == 3 && filter_C == 3)
	/*{
		printf("gemm_same_pixstep_batch total: %.3f ms, alloc %.3f ms, makeA: %.3f ms, gemm: %.3f ms, copy_C: %.3f ms\n", 1000 * (t5 - t1), 1000 * (t2 - t1),
			1000 * (t3 - t2), 1000 * (t4 - t3), 1000 * (t5 - t4));
	}*/
}


/*in_pixStep must be equal to filter_pixStep,
and the aligned channels should be set to zero*/
void zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep(
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
	__int64* buffer_len
)
{
	/************** image to col **************/
	int common_align_pixStep = __min(in_pixelStep, filter_pixelStep);
	int in_widthStep_mul_stride_H = in_widthStep*stride_H;
	int in_pixelStep_mul_stride_W = in_pixelStep*stride_W;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
	int common_pixStep_mul_filter_W = common_align_pixStep*filter_W;
	int matrix_A_cols = filter_H*filter_W*common_align_pixStep;
	int matrix_A_rows = out_H*out_W;
	int matrix_B_cols = filter_N;
	int matrix_B_rows = filter_H*filter_W*common_align_pixStep;
	__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(float) + 31) / 32 * 32;
	__int64 need_B_buffer_len_align32 = 0;
	__int64 need_C_buffer_len_align32 = 0;
	__int64 total_need_buffer_len;
	float* matrix_A = 0;
	float* matrix_Bt = 0;
	const float* in_row_ptr, *in_pix_ptr, *cur_in_row_ptr, *cur_in_pix_ptr, *filter_slice_ptr, *filter_row_ptr, *filter_pix_ptr;
	int out_n, out_h, out_w, kn, kh, kw, pp;
	float* matrix_A_row_ptr, *matrix_A_col_ptr, *cp_dst_ptr;
	const float *cp_src_ptr;
	float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
	const float* in_slice_ptr;
	
	float* matrix_C = 0, *matrix_C_row_ptr;
	int out_row_idx;
	int out_HW = out_H*out_W;
	double t1, t2, t3, t4, t5, alloc_time, make_A_time =0, gemm_time = 0;
	int need_allocate_tmp_out, need_allocate_matrix_Bt;
	t1 = omp_get_wtime();
	need_allocate_tmp_out = (out_pixelStep != filter_N) || (out_pixelStep*out_W != out_widthStep) || (out_widthStep*out_H != out_sliceStep);
	need_allocate_matrix_Bt = common_align_pixStep != filter_pixelStep;
	if (need_allocate_tmp_out)
	{
		need_C_buffer_len_align32 = (out_HW*filter_N * sizeof(float) + 31) / 32 * 32;
	}
	else
	{
		matrix_C = out_tensor4D_data;
	}
	if (need_allocate_matrix_Bt)
	{
		need_B_buffer_len_align32 = (matrix_B_rows*matrix_B_cols * sizeof(float) + 31) / 32 * 32;
	}
	else
	{
		matrix_Bt = (float*)filters_data;
	}
	total_need_buffer_len = need_A_buffer_len_align32 + need_B_buffer_len_align32 + need_C_buffer_len_align32;
	if (buffer == 0)
	{
		matrix_A = _aligned_malloc(need_A_buffer_len_align32, 32);
		if (need_allocate_matrix_Bt)
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
		if (need_allocate_matrix_Bt)
			matrix_Bt = (float*)((char*)(*buffer) + need_A_buffer_len_align32);
		if (need_allocate_tmp_out)
			matrix_C = (float*)((char*)(*buffer) + need_A_buffer_len_align32 + need_B_buffer_len_align32);
	}

	if (need_allocate_matrix_Bt)
	{
		cp_dst_ptr = matrix_Bt;
		for (kn = 0, filter_slice_ptr = filters_data; kn < filter_N; kn++, filter_slice_ptr += filter_sliceStep)
		{
			for (kh = 0, filter_row_ptr = filter_slice_ptr; kh < filter_H; kh++, filter_row_ptr += filter_widthStep)
			{
				for (kw = 0, filter_pix_ptr = filter_row_ptr; kw < filter_W; kw++, filter_pix_ptr += filter_pixelStep)
				{
					for (pp = 0, cp_src_ptr = filter_pix_ptr; pp < common_align_pixStep; pp += zq_mm_align_size, cp_src_ptr += zq_mm_align_size)
					{
						zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(cp_src_ptr));
						cp_dst_ptr += zq_mm_align_size;
					}
				}
			}
		}
	}
	
	t2 = omp_get_wtime();
	alloc_time = t2 - t1;
	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		t3 = omp_get_wtime();
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
						for (pp = 0, cp_src_ptr = cur_in_pix_ptr; pp < common_align_pixStep;
							pp += zq_mm_align_size, cp_src_ptr += zq_mm_align_size)
						{
							zq_mm_store_ps(matrix_A_col_ptr, zq_mm_load_ps(cp_src_ptr));
							matrix_A_col_ptr += zq_mm_align_size;
						}
					}
					

				}
				matrix_A_row_ptr += matrix_A_cols;
			}
		}
		t4 = omp_get_wtime();
		make_A_time += t4 - t3;
		/*gemm*/
		zq_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
			matrix_Bt, matrix_A_cols, 0.0f, matrix_C, matrix_B_cols);
		t5 = omp_get_wtime();
		gemm_time += t5 - t4;
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
		
	}

	if (buffer == 0)
	{
		_aligned_free(matrix_A);
		if (need_allocate_matrix_Bt)
			_aligned_free(matrix_Bt);
		if (need_allocate_tmp_out)
			_aligned_free(matrix_C);
	}
	t5 = omp_get_wtime();
	//if (filter_H == 3 && filter_W == 3 && filter_C == 3)
	/*{
		printf("gemm_nosame_pixstep total: %.3f ms, alloc %.3f ms, makeA: %.3f ms, gemm: %.3f ms\n", 1000 * (t5 - t1), 1000 * alloc_time,
			1000 * make_A_time, 1000 * gemm_time);
	}*/
}

/*in_pixStep can be different with filter_pixStep,
and the aligned channels should be set to zero*/
void zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_C3(
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
	__int64* buffer_len
)
{
	/************** image to col **************/
	int K = filter_H*filter_W*filter_C;
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	int padK = (K + 7) / 8*8;
#elif ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	int padK = (K + 3) / 4 * 4;
#else
	int padK = K;
#endif
	int in_widthStep_mul_stride_H = in_widthStep*stride_H;
	int in_pixelStep_mul_stride_W = in_pixelStep*stride_W;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
	int matrix_A_cols = padK;
	int matrix_A_rows = out_H*out_W;
	int matrix_B_cols = filter_N;
	int matrix_B_rows = padK;
	__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(float) + 31) / 32 * 32;
	__int64 need_B_buffer_len_align32 = 0;
	__int64 need_C_buffer_len_align32 = 0;
	__int64 total_need_buffer_len;
	float* matrix_A = 0;
	float* matrix_Bt = 0;
	const float* in_row_ptr, *in_pix_ptr, *cur_in_row_ptr, *cur_in_pix_ptr, *filter_slice_ptr, *filter_row_ptr, *filter_pix_ptr;
	int out_n, out_h, out_w, kn, kh, kw;
	float* matrix_A_row_ptr, *matrix_A_col_ptr, *matrix_Bt_row_ptr, *cp_dst_ptr;
	float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
	const float* in_slice_ptr;

	float* matrix_C = 0, *matrix_C_row_ptr;
	int out_row_idx;
	int out_HW = out_H*out_W;
	double t1, t2, t3, t4, t5, alloc_time, make_A_time = 0, gemm_time = 0;
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
	need_B_buffer_len_align32 = (matrix_B_rows*matrix_B_cols * sizeof(float) + 31) / 32 * 32;
	total_need_buffer_len = need_A_buffer_len_align32 + need_B_buffer_len_align32 + need_C_buffer_len_align32;
	if (buffer == 0)
	{
		matrix_A = _aligned_malloc(need_A_buffer_len_align32, 32);
		_aligned_malloc(need_B_buffer_len_align32, 32);
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

	matrix_Bt_row_ptr = matrix_Bt;
	for (kn = 0, filter_slice_ptr = filters_data; kn < filter_N; kn++, filter_slice_ptr += filter_sliceStep)
	{
		cp_dst_ptr = matrix_Bt_row_ptr;
		for (kh = 0, filter_row_ptr = filter_slice_ptr; kh < filter_H; kh++, filter_row_ptr += filter_widthStep)
		{
			for (kw = 0, filter_pix_ptr = filter_row_ptr; kw < filter_W; kw++, filter_pix_ptr += filter_pixelStep)
			{
				memcpy(cp_dst_ptr, filter_pix_ptr, sizeof(float)*filter_C);
				cp_dst_ptr += filter_C;
			}
		}
		matrix_Bt_row_ptr += matrix_B_rows;
	}
	

	t2 = omp_get_wtime();
	alloc_time = t2 - t1;
	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		t3 = omp_get_wtime();
		matrix_A_row_ptr = matrix_A;
		for (out_h = 0, in_row_ptr = in_slice_ptr; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
			{
				matrix_A_col_ptr = matrix_A_row_ptr;
				for (kh = 0, cur_in_row_ptr = in_pix_ptr; kh < filter_H; kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep)
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
		t4 = omp_get_wtime();
		make_A_time += t4 - t3;
		/*gemm*/
		zq_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, K, 1, matrix_A, matrix_A_cols,
			matrix_Bt, matrix_A_cols, 0.0f, matrix_C, matrix_B_cols);
		t5 = omp_get_wtime();
		gemm_time += t5 - t4;
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

	}

	if (buffer == 0)
	{
		_aligned_free(matrix_A);
		_aligned_free(matrix_Bt);
		if (need_allocate_tmp_out)
			_aligned_free(matrix_C);
	}
	t5 = omp_get_wtime();
	//if (filter_H == 3 && filter_W == 3 && filter_C == 3)
	/*{
	printf("gemm_nosame_pixstep total: %.3f ms, alloc %.3f ms, makeA: %.3f ms, gemm: %.3f ms\n", 1000 * (t5 - t1), 1000 * alloc_time,
	1000 * make_A_time, 1000 * gemm_time);
	}*/
}




/*in_pixStep can be different with filter_pixStep,
and the aligned channels should be set to zero*/
void zq_cnn_conv_no_padding_gemm_32f_align_same_or_notsame_pixstep_batch(
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
	__int64* buffer_len
)
{
	/************** image to col **************/
	int common_align_pixStep = __min(in_pixelStep, filter_pixelStep);
	int in_widthStep_mul_stride_H = in_widthStep*stride_H;
	int in_pixelStep_mul_stride_W = in_pixelStep*stride_W;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
	int filter_pixStep_mul_filter_W = filter_pixelStep*filter_W;
	int matrix_A_cols = filter_H*filter_W*common_align_pixStep;
	int matrix_A_rows = out_N*out_H*out_W;
	int matrix_B_cols = filter_N;
	int matrix_B_rows = filter_H*filter_W*common_align_pixStep;
	__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(float) + 31) / 32 * 32;
	__int64 need_B_buffer_len_align32 = 0;
	__int64 need_C_buffer_len_align32 = 0;
	__int64 total_need_buffer_len;
	float* matrix_A = 0;
	float* matrix_Bt = 0;
	const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *cur_in_row_ptr, *cur_in_pix_ptr, *filter_slice_ptr, *filter_row_ptr, *filter_pix_ptr;
	int out_n, out_h, out_w, kn, kh, kw, pp;
	float* matrix_A_row_ptr, *matrix_A_col_ptr, *cp_dst_ptr;
	const float* cp_src_ptr;
	float* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
	float* matrix_C = 0, *matrix_C_row_ptr;
	int out_row_idx;
	int out_NHW = out_N*out_H*out_W;
	double t1, t2, t3, t4, t5;
	int need_allocate_matrix_Bt, need_allocate_tmp_out;
	t1 = omp_get_wtime();
	need_allocate_matrix_Bt = common_align_pixStep != filter_pixelStep;
	need_allocate_tmp_out = (out_pixelStep != filter_N) || (out_pixelStep*out_W != out_widthStep) || (out_widthStep*out_H != out_sliceStep);
	if (need_allocate_tmp_out)
	{
		need_C_buffer_len_align32 = (out_NHW*filter_N * sizeof(float) + 31) / 32 * 32;
	}
	else
	{
		matrix_C = out_tensor4D_data;
	}
	if (need_allocate_matrix_Bt)
	{
		need_B_buffer_len_align32 = (matrix_B_rows*matrix_B_cols * sizeof(float) + 31) / 32 * 32;
	}
	else
	{
		matrix_Bt = (float*)filters_data;
	}
	total_need_buffer_len = need_A_buffer_len_align32 + need_B_buffer_len_align32 + need_C_buffer_len_align32;
	if (buffer == 0)
	{
		matrix_A = _aligned_malloc(need_A_buffer_len_align32, 32);
		if (need_allocate_matrix_Bt)
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
		if (need_allocate_matrix_Bt)
			matrix_Bt = (float*)((char*)(*buffer) + need_A_buffer_len_align32);
		if (need_allocate_tmp_out)
			matrix_C = (float*)((char*)(*buffer) + need_A_buffer_len_align32 + need_B_buffer_len_align32);
	}
	if (need_allocate_matrix_Bt)
	{
		cp_dst_ptr = matrix_Bt;
		for (kn = 0, filter_slice_ptr = filters_data; kn < filter_N; kn++, filter_slice_ptr += filter_sliceStep)
		{
			for (kh = 0, filter_row_ptr = filter_slice_ptr; kh < filter_H; kh++, filter_row_ptr += filter_widthStep)
			{
				for (kw = 0, filter_pix_ptr = filter_row_ptr; kw < filter_W; kw++, filter_pix_ptr += filter_pixelStep)
				{
					for (pp = 0, cp_src_ptr = filter_pix_ptr; pp < common_align_pixStep; pp += zq_mm_align_size, cp_src_ptr += zq_mm_align_size)
					{
						zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(cp_src_ptr));
						cp_dst_ptr += zq_mm_align_size;
					}
				}
			}
		}
	}

	t2 = omp_get_wtime();


	matrix_A_row_ptr = matrix_A;
	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (out_h = 0, in_row_ptr = in_slice_ptr; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
			{
				matrix_A_col_ptr = matrix_A_row_ptr;
				for (kh = 0, cur_in_row_ptr = in_pix_ptr;
					kh < filter_H;
					kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep)
				{
					for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr; kw < filter_W; kw++, cur_in_pix_ptr += dilate_W_mul_in_pixStep)
					{
						for (pp = 0, cp_src_ptr = cur_in_pix_ptr; pp < common_align_pixStep;
							pp += zq_mm_align_size, cp_src_ptr += zq_mm_align_size)
						{
							zq_mm_store_ps(matrix_A_col_ptr, zq_mm_load_ps(cp_src_ptr));
							matrix_A_col_ptr += zq_mm_align_size;
						}
					}
				}
				matrix_A_row_ptr += matrix_A_cols;
			}
		}
	}

	t3 = omp_get_wtime();
	/*gemm*/
	zq_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
		matrix_Bt, matrix_A_cols, 0.0f, matrix_C, matrix_B_cols);
	t4 = omp_get_wtime();
	if (need_allocate_tmp_out)
	{
		/*   col2im      */
		out_row_idx = 0;
		matrix_C_row_ptr = matrix_C;
		for (out_n = 0, out_slice_ptr = out_tensor4D_data; out_n < out_N; out_n++,out_slice_ptr+=out_sliceStep)
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
		if (need_allocate_matrix_Bt)
			_aligned_free(matrix_Bt);
		if (need_allocate_tmp_out)
			_aligned_free(matrix_C);
	}
	//if (filter_H == 3 && filter_W == 3 && filter_C == 3)
	/*{
		printf("gemm_same_pixstep_batch total: %.3f ms, alloc %.3f ms, makeA: %.3f ms, gemm: %.3f ms, copy_C: %.3f ms\n", 1000 * (t5 - t1), 1000 * (t2 - t1),
			1000 * (t3 - t2), 1000 * (t4 - t3), 1000 * (t5 - t4));
	}*/
}

