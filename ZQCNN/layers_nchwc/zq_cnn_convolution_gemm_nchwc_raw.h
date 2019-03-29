/*in_pixStep can be different with filter_pixStep,
and the aligned channels should be set to zero*/
void zq_cnn_conv_no_padding_gemm_nchwc_general(
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
	int stride_H,
	int stride_W,
	int dilation_H,
	int dilation_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_widthStep,
	int out_sliceStep,
	int out_imStep,
	void** buffer,
	__int64* buffer_len
)
{
	/************** image to col **************/
	int align_C = (in_C + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	int in_widthStep_mul_stride_H = in_widthStep*stride_H;
	int in_pixelStep_mul_stride_W = zq_mm_align_size*stride_W;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*zq_mm_align_size;
	int filter_pixStep_mul_filter_W = zq_mm_align_size*filter_W;
	int matrix_A_cols = filter_H*filter_W*align_C;
	int matrix_A_rows = out_N*out_H*out_W;
	int matrix_B_cols = filter_N;
	int matrix_B_rows = filter_H*filter_W*align_C;
	int out_NHW = out_N*out_H*out_W;
	__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 need_B_buffer_len_align32 = (matrix_B_rows*filter_N * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 need_C_buffer_len_align32 = need_C_buffer_len_align32 = (out_NHW*filter_N * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 total_need_buffer_len;
	zq_base_type* matrix_A = 0;
	zq_base_type* matrix_Bt = 0;
	const zq_base_type *in_im_ptr, *in_slice_ptr, *in_row_ptr, *in_pix_ptr, *cur_in_row_ptr, *cur_in_pix_ptr;
	const zq_base_type *filter_im_ptr, *filter_slice_ptr, *filter_row_ptr, *filter_pix_ptr;
	int out_n, out_h, out_w, kn, kh, kw, kc, i;
	zq_base_type* matrix_A_row_ptr, *matrix_A_col_ptr, *cp_dst_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_im_ptr;
	zq_base_type* matrix_C = 0, *matrix_C_row_ptr;
	int out_row_idx;
	
	double t1, t2, t3, t4, t5;
	t1 = omp_get_wtime();
	total_need_buffer_len = need_A_buffer_len_align32 + need_B_buffer_len_align32 + need_C_buffer_len_align32;
	if (buffer == 0)
	{
		matrix_A = (zq_base_type*)_aligned_malloc(need_A_buffer_len_align32, 32);
		matrix_Bt = (zq_base_type*)_aligned_malloc(need_B_buffer_len_align32, 32);
		matrix_C = (zq_base_type*)_aligned_malloc(need_C_buffer_len_align32, 32);
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
		matrix_Bt = (zq_base_type*)((char*)(*buffer) + need_A_buffer_len_align32);
		matrix_C = (zq_base_type*)((char*)(*buffer) + need_A_buffer_len_align32 + need_B_buffer_len_align32);
	}
	
	cp_dst_ptr = matrix_Bt;
	for (kn = 0, filter_im_ptr = filters_data; kn < filter_N; kn++, filter_im_ptr += filter_imStep)
	{
		for (kh = 0, filter_row_ptr = filter_im_ptr; kh < filter_H; kh++, filter_row_ptr += filter_widthStep)
		{
			for (kw = 0, filter_pix_ptr = filter_row_ptr; kw < filter_W; kw++, filter_pix_ptr += zq_mm_align_size)
			{
				for (kc = 0, filter_slice_ptr = filter_pix_ptr; kc < filter_C; kc += zq_mm_align_size, filter_slice_ptr += filter_sliceStep)
				{
					zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(filter_slice_ptr));
					cp_dst_ptr += zq_mm_align_size;
				}
			}
		}
	}

	t2 = omp_get_wtime();


	matrix_A_row_ptr = matrix_A;
	for (out_n = 0, in_im_ptr = in_tensor4D_data;
		out_n < out_N;
		out_n++, in_im_ptr += in_imStep)
	{
		for (out_h = 0, in_row_ptr = in_im_ptr; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
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
	out_row_idx = 0;
	matrix_C_row_ptr = matrix_C;
	for (out_n = 0, out_im_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_im_ptr += out_imStep)
	{
		for (out_h = 0, out_row_ptr = out_im_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W; out_w++, out_pix_ptr += zq_mm_align_size)
			{
				for (kc = 0, out_slice_ptr = out_pix_ptr; kc < out_C - zq_mm_align_size; kc += zq_mm_align_size, out_slice_ptr += out_sliceStep)
					zq_mm_store_ps(out_slice_ptr, zq_mm_loadu_ps(matrix_C_row_ptr + kc));
				for (i = 0; i + kc < out_C; i++)
					out_slice_ptr[i] = matrix_C_row_ptr[i + kc];
				matrix_C_row_ptr += matrix_B_cols;
			}
		}
	}
	t5 = omp_get_wtime();

	if (buffer == 0)
	{
		_aligned_free(matrix_A);
		_aligned_free(matrix_Bt);
		_aligned_free(matrix_C);
	}
	//if (filter_H == 3 && filter_W == 3 && filter_C == 3)
	/*{
	printf("gemm_same_pixstep_batch total: %.3f ms, alloc %.3f ms, makeA: %.3f ms, gemm: %.3f ms, copy_C: %.3f ms\n", 1000 * (t5 - t1), 1000 * (t2 - t1),
	1000 * (t3 - t2), 1000 * (t4 - t3), 1000 * (t5 - t4));
	}*/
}

/*in_pixStep can be different with filter_pixStep,
and the aligned channels should be set to zero*/
void zq_cnn_conv_no_padding_gemm_nchwc_kernel1x1(
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
	int stride_H,
	int stride_W,
	int dilation_H,
	int dilation_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_widthStep,
	int out_sliceStep,
	int out_imStep,
	void** buffer,
	__int64* buffer_len
)
{
	/************** image to col **************/
	int align_C = (in_C + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	int in_widthStep_mul_stride_H = in_widthStep*stride_H;
	int in_pixelStep_mul_stride_W = zq_mm_align_size*stride_W;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*zq_mm_align_size;
	int filter_pixStep_mul_filter_W = zq_mm_align_size*filter_W;
	int matrix_A_cols = filter_H*filter_W*align_C;
	int matrix_A_rows = out_N*out_H*out_W;
	int matrix_B_cols = filter_N;
	int matrix_B_rows = filter_H*filter_W*align_C;
	int out_NHW = out_N*out_H*out_W;
	__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 need_C_buffer_len_align32 = need_C_buffer_len_align32 = (out_NHW*filter_N * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 total_need_buffer_len;
	zq_base_type* matrix_A = 0;
	const zq_base_type *in_im_ptr, *in_slice_ptr, *in_row_ptr, *in_pix_ptr;
	int out_n, out_h, out_w, kc, i;
	zq_base_type* matrix_A_row_ptr, *matrix_A_col_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_im_ptr;
	zq_base_type* matrix_C = 0, *matrix_C_row_ptr;
	int out_row_idx;

	double t1, t2, t3, t4, t5;
	t1 = omp_get_wtime();
	total_need_buffer_len = need_A_buffer_len_align32 + need_C_buffer_len_align32;
	if (buffer == 0)
	{
		matrix_A = (zq_base_type*)_aligned_malloc(need_A_buffer_len_align32, 32);
		matrix_C = (zq_base_type*)_aligned_malloc(need_C_buffer_len_align32, 32);
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
		matrix_C = (zq_base_type*)((char*)(*buffer) + need_A_buffer_len_align32);
	}

	t2 = omp_get_wtime();


	matrix_A_row_ptr = matrix_A;
	for (out_n = 0, in_im_ptr = in_tensor4D_data;
		out_n < out_N;
		out_n++, in_im_ptr += in_imStep)
	{
		for (out_h = 0, in_row_ptr = in_im_ptr; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
			{
				matrix_A_col_ptr = matrix_A_row_ptr;
				for (kc = 0, in_slice_ptr = in_pix_ptr; kc < in_C;
					kc += zq_mm_align_size, in_slice_ptr += in_sliceStep)
				{
					zq_mm_store_ps(matrix_A_col_ptr, zq_mm_load_ps(in_slice_ptr));
					matrix_A_col_ptr += zq_mm_align_size;
				}
				matrix_A_row_ptr += matrix_A_cols;
			}
		}
	}

	t3 = omp_get_wtime();
	/*gemm*/
#if __ARM_NEON && ZQ_CNN_USE_ZQ_GEMM && ZQ_CNN_USE_BLAS_GEMM
	if (0 == zq_gemm_32f_AnoTrans_Btrans_special(matrix_A_rows, matrix_B_cols, matrix_A_cols, matrix_A, matrix_A_cols,
		filters_data, matrix_A_cols, matrix_C, matrix_B_cols))
	{
		zq_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
			filters_data, matrix_A_cols, 0.0f, matrix_C, matrix_B_cols);
	}
#else
	zq_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
		filters_data, matrix_A_cols, 0.0f, matrix_C, matrix_B_cols);
#endif
	t4 = omp_get_wtime();

	/*   col2im      */
	out_row_idx = 0;
	matrix_C_row_ptr = matrix_C;
	for (out_n = 0, out_im_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_im_ptr += out_imStep)
	{
		for (out_h = 0, out_row_ptr = out_im_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W; out_w++, out_pix_ptr += zq_mm_align_size)
			{
				for (kc = 0, out_slice_ptr = out_pix_ptr; kc < out_C - zq_mm_align_size; kc += zq_mm_align_size, out_slice_ptr += out_sliceStep)
					zq_mm_store_ps(out_slice_ptr, zq_mm_loadu_ps(matrix_C_row_ptr + kc));
				for (i = 0; i + kc < out_C; i++)
					out_slice_ptr[i] = matrix_C_row_ptr[i + kc];
				matrix_C_row_ptr += matrix_B_cols;
			}
		}
	}
	t5 = omp_get_wtime();

	if (buffer == 0)
	{
		_aligned_free(matrix_A);
		_aligned_free(matrix_C);
	}
}

/*in_pixStep can be different with filter_pixStep,
and the aligned channels should be set to zero*/
void zq_cnn_conv_no_padding_gemm_nchwc_kernel2x2(
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
	int stride_H,
	int stride_W,
	int dilation_H,
	int dilation_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_widthStep,
	int out_sliceStep,
	int out_imStep,
	void** buffer,
	__int64* buffer_len
)
{
	/************** image to col **************/
	int align_C = (in_C + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	int in_widthStep_mul_stride_H = in_widthStep*stride_H;
	int in_pixelStep_mul_stride_W = zq_mm_align_size*stride_W;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*zq_mm_align_size;
	int filter_pixStep_mul_filter_W = zq_mm_align_size*filter_W;
	int matrix_A_cols = filter_H*filter_W*align_C;
	int matrix_A_rows = out_N*out_H*out_W;
	int matrix_B_cols = filter_N;
	int matrix_B_rows = filter_H*filter_W*align_C;
	int out_NHW = out_N*out_H*out_W;
	__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 need_B_buffer_len_align32 = (matrix_B_rows*filter_N * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 need_C_buffer_len_align32 = need_C_buffer_len_align32 = (out_NHW*filter_N * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 total_need_buffer_len;
	zq_base_type* matrix_A = 0;
	zq_base_type* matrix_Bt = 0;
	const zq_base_type *in_im_ptr, *in_slice_ptr, *in_row_ptr, *in_pix_ptr, *cur_in_row_ptr, *cur_in_pix_ptr;
	const zq_base_type *filter_im_ptr, *filter_slice_ptr, *filter_row_ptr, *filter_pix_ptr;
	int out_n, out_h, out_w, kn, kc, i;
	zq_base_type* matrix_A_row_ptr, *matrix_A_col_ptr, *cp_dst_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_im_ptr;
	zq_base_type* matrix_C = 0, *matrix_C_row_ptr;
	int out_row_idx;

	double t1, t2, t3, t4, t5;
	t1 = omp_get_wtime();
	total_need_buffer_len = need_A_buffer_len_align32 + need_B_buffer_len_align32 + need_C_buffer_len_align32;
	if (buffer == 0)
	{
		matrix_A = (zq_base_type*)_aligned_malloc(need_A_buffer_len_align32, 32);
		matrix_Bt = (zq_base_type*)_aligned_malloc(need_B_buffer_len_align32, 32);
		matrix_C = (zq_base_type*)_aligned_malloc(need_C_buffer_len_align32, 32);
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
		matrix_Bt = (zq_base_type*)((char*)(*buffer) + need_A_buffer_len_align32);
		matrix_C = (zq_base_type*)((char*)(*buffer) + need_A_buffer_len_align32 + need_B_buffer_len_align32);
	}

	cp_dst_ptr = matrix_Bt;
	for (kn = 0, filter_im_ptr = filters_data; kn < filter_N; kn++, filter_im_ptr += filter_imStep)
	{
		filter_row_ptr = filter_im_ptr;
		filter_pix_ptr = filter_row_ptr;
		for (kc = 0, filter_slice_ptr = filter_pix_ptr; kc < filter_C; kc += zq_mm_align_size, filter_slice_ptr += filter_sliceStep)
		{
			zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(filter_slice_ptr));
			cp_dst_ptr += zq_mm_align_size;
		}
		filter_pix_ptr += zq_mm_align_size;
		for (kc = 0, filter_slice_ptr = filter_pix_ptr; kc < filter_C; kc += zq_mm_align_size, filter_slice_ptr += filter_sliceStep)
		{
			zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(filter_slice_ptr));
			cp_dst_ptr += zq_mm_align_size;
		}
		filter_row_ptr += filter_widthStep;
		filter_pix_ptr = filter_row_ptr;
		for (kc = 0, filter_slice_ptr = filter_pix_ptr; kc < filter_C; kc += zq_mm_align_size, filter_slice_ptr += filter_sliceStep)
		{
			zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(filter_slice_ptr));
			cp_dst_ptr += zq_mm_align_size;
		}
		filter_pix_ptr += zq_mm_align_size;
		for (kc = 0, filter_slice_ptr = filter_pix_ptr; kc < filter_C; kc += zq_mm_align_size, filter_slice_ptr += filter_sliceStep)
		{
			zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(filter_slice_ptr));
			cp_dst_ptr += zq_mm_align_size;
		}
	}

	t2 = omp_get_wtime();


	matrix_A_row_ptr = matrix_A;
	for (out_n = 0, in_im_ptr = in_tensor4D_data;
		out_n < out_N;
		out_n++, in_im_ptr += in_imStep)
	{
		for (out_h = 0, in_row_ptr = in_im_ptr; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
			{
				matrix_A_col_ptr = matrix_A_row_ptr;
				cur_in_row_ptr = in_pix_ptr;
				cur_in_pix_ptr = cur_in_row_ptr;
				for (kc = 0, in_slice_ptr = cur_in_pix_ptr; kc < in_C;
					kc += zq_mm_align_size, in_slice_ptr += in_sliceStep)
				{
					zq_mm_store_ps(matrix_A_col_ptr, zq_mm_load_ps(in_slice_ptr));
					matrix_A_col_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;
				for (kc = 0, in_slice_ptr = cur_in_pix_ptr; kc < in_C;
					kc += zq_mm_align_size, in_slice_ptr += in_sliceStep)
				{
					zq_mm_store_ps(matrix_A_col_ptr, zq_mm_load_ps(in_slice_ptr));
					matrix_A_col_ptr += zq_mm_align_size;
				}
				cur_in_row_ptr += dilate_H_mul_in_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr;
				for (kc = 0, in_slice_ptr = cur_in_pix_ptr; kc < in_C;
					kc += zq_mm_align_size, in_slice_ptr += in_sliceStep)
				{
					zq_mm_store_ps(matrix_A_col_ptr, zq_mm_load_ps(in_slice_ptr));
					matrix_A_col_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;
				for (kc = 0, in_slice_ptr = cur_in_pix_ptr; kc < in_C;
					kc += zq_mm_align_size, in_slice_ptr += in_sliceStep)
				{
					zq_mm_store_ps(matrix_A_col_ptr, zq_mm_load_ps(in_slice_ptr));
					matrix_A_col_ptr += zq_mm_align_size;
				}
				matrix_A_row_ptr += matrix_A_cols;
			}
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
	out_row_idx = 0;
	matrix_C_row_ptr = matrix_C;
	for (out_n = 0, out_im_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_im_ptr += out_imStep)
	{
		for (out_h = 0, out_row_ptr = out_im_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W; out_w++, out_pix_ptr += zq_mm_align_size)
			{
				for (kc = 0, out_slice_ptr = out_pix_ptr; kc < out_C - zq_mm_align_size; kc += zq_mm_align_size, out_slice_ptr += out_sliceStep)
					zq_mm_store_ps(out_slice_ptr, zq_mm_loadu_ps(matrix_C_row_ptr + kc));
				for (i = 0; i + kc < out_C; i++)
					out_slice_ptr[i] = matrix_C_row_ptr[i + kc];
				matrix_C_row_ptr += matrix_B_cols;
			}
		}
	}
	t5 = omp_get_wtime();

	if (buffer == 0)
	{
		_aligned_free(matrix_A);
		_aligned_free(matrix_Bt);
		_aligned_free(matrix_C);
	}
}

/*in_pixStep can be different with filter_pixStep,
and the aligned channels should be set to zero*/
void zq_cnn_conv_no_padding_gemm_nchwc_kernel3x3(
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
	int stride_H,
	int stride_W,
	int dilation_H,
	int dilation_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_widthStep,
	int out_sliceStep,
	int out_imStep,
	void** buffer,
	__int64* buffer_len
)
{
	/************** image to col **************/
	int align_C = (in_C + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	int in_widthStep_mul_stride_H = in_widthStep*stride_H;
	int in_pixelStep_mul_stride_W = zq_mm_align_size*stride_W;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*zq_mm_align_size;
	int filter_pixStep_mul_filter_W = zq_mm_align_size*filter_W;
	int matrix_A_cols = filter_H*filter_W*align_C;
	int matrix_A_rows = out_N*out_H*out_W;
	int matrix_B_cols = filter_N;
	int matrix_B_rows = filter_H*filter_W*align_C;
	int out_NHW = out_N*out_H*out_W;
	__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 need_B_buffer_len_align32 = (matrix_B_rows*filter_N * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 need_C_buffer_len_align32 = need_C_buffer_len_align32 = (out_NHW*filter_N * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 total_need_buffer_len;
	zq_base_type* matrix_A = 0;
	zq_base_type* matrix_Bt = 0;
	const zq_base_type *in_im_ptr, *in_slice_ptr, *in_row_ptr, *in_pix_ptr, *cur_in_row_ptr, *cur_in_pix_ptr;
	const zq_base_type *filter_im_ptr, *filter_slice_ptr, *filter_row_ptr, *filter_pix_ptr;
	int out_n, out_h, out_w, kn, kc, i;
	zq_base_type* matrix_A_row_ptr, *matrix_A_col_ptr, *cp_dst_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_im_ptr;
	zq_base_type* matrix_C = 0, *matrix_C_row_ptr;
	int out_row_idx;

	double t1, t2, t3, t4, t5;
	t1 = omp_get_wtime();
	total_need_buffer_len = need_A_buffer_len_align32 + need_B_buffer_len_align32 + need_C_buffer_len_align32;
	if (buffer == 0)
	{
		matrix_A = (zq_base_type*)_aligned_malloc(need_A_buffer_len_align32, 32);
		matrix_Bt = (zq_base_type*)_aligned_malloc(need_B_buffer_len_align32, 32);
		matrix_C = (zq_base_type*)_aligned_malloc(need_C_buffer_len_align32, 32);
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
		matrix_Bt = (zq_base_type*)((char*)(*buffer) + need_A_buffer_len_align32);
		matrix_C = (zq_base_type*)((char*)(*buffer) + need_A_buffer_len_align32 + need_B_buffer_len_align32);
	}

	cp_dst_ptr = matrix_Bt;
	for (kn = 0, filter_im_ptr = filters_data; kn < filter_N; kn++, filter_im_ptr += filter_imStep)
	{
		filter_row_ptr = filter_im_ptr;
		filter_pix_ptr = filter_row_ptr;
		for (kc = 0, filter_slice_ptr = filter_pix_ptr; kc < filter_C; kc += zq_mm_align_size, filter_slice_ptr += filter_sliceStep)
		{
			zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(filter_slice_ptr));
			cp_dst_ptr += zq_mm_align_size;
		}
		filter_pix_ptr += zq_mm_align_size;
		for (kc = 0, filter_slice_ptr = filter_pix_ptr; kc < filter_C; kc += zq_mm_align_size, filter_slice_ptr += filter_sliceStep)
		{
			zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(filter_slice_ptr));
			cp_dst_ptr += zq_mm_align_size;
		}
		filter_pix_ptr += zq_mm_align_size;
		for (kc = 0, filter_slice_ptr = filter_pix_ptr; kc < filter_C; kc += zq_mm_align_size, filter_slice_ptr += filter_sliceStep)
		{
			zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(filter_slice_ptr));
			cp_dst_ptr += zq_mm_align_size;
		}
		filter_row_ptr += filter_widthStep;
		filter_pix_ptr = filter_row_ptr;
		for (kc = 0, filter_slice_ptr = filter_pix_ptr; kc < filter_C; kc += zq_mm_align_size, filter_slice_ptr += filter_sliceStep)
		{
			zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(filter_slice_ptr));
			cp_dst_ptr += zq_mm_align_size;
		}
		filter_pix_ptr += zq_mm_align_size;
		for (kc = 0, filter_slice_ptr = filter_pix_ptr; kc < filter_C; kc += zq_mm_align_size, filter_slice_ptr += filter_sliceStep)
		{
			zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(filter_slice_ptr));
			cp_dst_ptr += zq_mm_align_size;
		}
		filter_pix_ptr += zq_mm_align_size;
		for (kc = 0, filter_slice_ptr = filter_pix_ptr; kc < filter_C; kc += zq_mm_align_size, filter_slice_ptr += filter_sliceStep)
		{
			zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(filter_slice_ptr));
			cp_dst_ptr += zq_mm_align_size;
		}
		filter_row_ptr += filter_widthStep;
		filter_pix_ptr = filter_row_ptr;
		for (kc = 0, filter_slice_ptr = filter_pix_ptr; kc < filter_C; kc += zq_mm_align_size, filter_slice_ptr += filter_sliceStep)
		{
			zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(filter_slice_ptr));
			cp_dst_ptr += zq_mm_align_size;
		}
		filter_pix_ptr += zq_mm_align_size;
		for (kc = 0, filter_slice_ptr = filter_pix_ptr; kc < filter_C; kc += zq_mm_align_size, filter_slice_ptr += filter_sliceStep)
		{
			zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(filter_slice_ptr));
			cp_dst_ptr += zq_mm_align_size;
		}
		filter_pix_ptr += zq_mm_align_size;
		for (kc = 0, filter_slice_ptr = filter_pix_ptr; kc < filter_C; kc += zq_mm_align_size, filter_slice_ptr += filter_sliceStep)
		{
			zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(filter_slice_ptr));
			cp_dst_ptr += zq_mm_align_size;
		}
	}

	t2 = omp_get_wtime();


	matrix_A_row_ptr = matrix_A;
	for (out_n = 0, in_im_ptr = in_tensor4D_data;
		out_n < out_N;
		out_n++, in_im_ptr += in_imStep)
	{
		for (out_h = 0, in_row_ptr = in_im_ptr; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
			{
				matrix_A_col_ptr = matrix_A_row_ptr;
				cur_in_row_ptr = in_pix_ptr;
				cur_in_pix_ptr = cur_in_row_ptr;
				for (kc = 0, in_slice_ptr = cur_in_pix_ptr; kc < in_C;
					kc += zq_mm_align_size, in_slice_ptr += in_sliceStep)
				{
					zq_mm_store_ps(matrix_A_col_ptr, zq_mm_load_ps(in_slice_ptr));
					matrix_A_col_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;
				for (kc = 0, in_slice_ptr = cur_in_pix_ptr; kc < in_C;
					kc += zq_mm_align_size, in_slice_ptr += in_sliceStep)
				{
					zq_mm_store_ps(matrix_A_col_ptr, zq_mm_load_ps(in_slice_ptr));
					matrix_A_col_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;
				for (kc = 0, in_slice_ptr = cur_in_pix_ptr; kc < in_C;
					kc += zq_mm_align_size, in_slice_ptr += in_sliceStep)
				{
					zq_mm_store_ps(matrix_A_col_ptr, zq_mm_load_ps(in_slice_ptr));
					matrix_A_col_ptr += zq_mm_align_size;
				}
				cur_in_row_ptr += dilate_H_mul_in_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr;
				for (kc = 0, in_slice_ptr = cur_in_pix_ptr; kc < in_C;
					kc += zq_mm_align_size, in_slice_ptr += in_sliceStep)
				{
					zq_mm_store_ps(matrix_A_col_ptr, zq_mm_load_ps(in_slice_ptr));
					matrix_A_col_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;
				for (kc = 0, in_slice_ptr = cur_in_pix_ptr; kc < in_C;
					kc += zq_mm_align_size, in_slice_ptr += in_sliceStep)
				{
					zq_mm_store_ps(matrix_A_col_ptr, zq_mm_load_ps(in_slice_ptr));
					matrix_A_col_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;
				for (kc = 0, in_slice_ptr = cur_in_pix_ptr; kc < in_C;
					kc += zq_mm_align_size, in_slice_ptr += in_sliceStep)
				{
					zq_mm_store_ps(matrix_A_col_ptr, zq_mm_load_ps(in_slice_ptr));
					matrix_A_col_ptr += zq_mm_align_size;
				}
				cur_in_row_ptr += dilate_H_mul_in_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr;
				for (kc = 0, in_slice_ptr = cur_in_pix_ptr; kc < in_C;
					kc += zq_mm_align_size, in_slice_ptr += in_sliceStep)
				{
					zq_mm_store_ps(matrix_A_col_ptr, zq_mm_load_ps(in_slice_ptr));
					matrix_A_col_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;
				for (kc = 0, in_slice_ptr = cur_in_pix_ptr; kc < in_C;
					kc += zq_mm_align_size, in_slice_ptr += in_sliceStep)
				{
					zq_mm_store_ps(matrix_A_col_ptr, zq_mm_load_ps(in_slice_ptr));
					matrix_A_col_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;
				for (kc = 0, in_slice_ptr = cur_in_pix_ptr; kc < in_C;
					kc += zq_mm_align_size, in_slice_ptr += in_sliceStep)
				{
					zq_mm_store_ps(matrix_A_col_ptr, zq_mm_load_ps(in_slice_ptr));
					matrix_A_col_ptr += zq_mm_align_size;
				}
				matrix_A_row_ptr += matrix_A_cols;
			}
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
	out_row_idx = 0;
	matrix_C_row_ptr = matrix_C;
	for (out_n = 0, out_im_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_im_ptr += out_imStep)
	{
		for (out_h = 0, out_row_ptr = out_im_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W; out_w++, out_pix_ptr += zq_mm_align_size)
			{
				for (kc = 0, out_slice_ptr = out_pix_ptr; kc < out_C - zq_mm_align_size; kc += zq_mm_align_size, out_slice_ptr += out_sliceStep)
					zq_mm_store_ps(out_slice_ptr, zq_mm_loadu_ps(matrix_C_row_ptr + kc));
				for (i = 0; i + kc < out_C; i++)
					out_slice_ptr[i] = matrix_C_row_ptr[i + kc];
				matrix_C_row_ptr += matrix_B_cols;
			}
		}
	}
	t5 = omp_get_wtime();

	if (buffer == 0)
	{
		_aligned_free(matrix_A);
		_aligned_free(matrix_Bt);
		_aligned_free(matrix_C);
	}
}

/*in_pixStep can be different with filter_pixStep,
and the aligned channels should be set to zero*/
void zq_cnn_conv_no_padding_gemm_nchwc_kernel2x2_C3(
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
	int stride_H,
	int stride_W,
	int dilation_H,
	int dilation_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_widthStep,
	int out_sliceStep,
	int out_imStep,
	void** buffer,
	__int64* buffer_len
)
{
	/************** image to col **************/
	int align_C = (in_C + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	int in_widthStep_mul_stride_H = in_widthStep*stride_H;
	int in_pixelStep_mul_stride_W = zq_mm_align_size*stride_W;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*zq_mm_align_size;
	int filter_pixStep_mul_filter_W = zq_mm_align_size*filter_W;
	int matrix_A_cols = filter_H*filter_W*align_C;
	int matrix_A_rows = out_N*out_H*out_W;
	int matrix_B_cols = filter_N;
	int matrix_B_rows = (filter_H*filter_W * 3 + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	int out_NHW = out_N*out_H*out_W;
	__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 need_B_buffer_len_align32 = (matrix_B_rows*filter_N * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 need_C_buffer_len_align32 = need_C_buffer_len_align32 = (out_NHW*filter_N * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 total_need_buffer_len;
	zq_base_type* matrix_A = 0;
	zq_base_type* matrix_Bt = 0;
	const zq_base_type *in_im_ptr, *in_row_ptr, *in_pix_ptr, *cur_in_row_ptr, *cur_in_pix_ptr;
	const zq_base_type *filter_im_ptr, *filter_row_ptr, *filter_pix_ptr;
	int out_n, out_h, out_w, kn, kc, i;
	zq_base_type* matrix_A_row_ptr, *matrix_A_col_ptr, *cp_dst_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_im_ptr;
	zq_base_type* matrix_C = 0, *matrix_C_row_ptr;
	int out_row_idx;

	double t1, t2, t3, t4, t5;
	t1 = omp_get_wtime();
	total_need_buffer_len = need_A_buffer_len_align32 + need_B_buffer_len_align32 + need_C_buffer_len_align32;
	if (buffer == 0)
	{
		matrix_A = (zq_base_type*)_aligned_malloc(need_A_buffer_len_align32, 32);
		matrix_Bt = (zq_base_type*)_aligned_malloc(need_B_buffer_len_align32, 32);
		matrix_C = (zq_base_type*)_aligned_malloc(need_C_buffer_len_align32, 32);
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
		matrix_Bt = (zq_base_type*)((char*)(*buffer) + need_A_buffer_len_align32);
		matrix_C = (zq_base_type*)((char*)(*buffer) + need_A_buffer_len_align32 + need_B_buffer_len_align32);
	}

	cp_dst_ptr = matrix_Bt;
	for (kn = 0, filter_im_ptr = filters_data; kn < filter_N; kn++, filter_im_ptr += filter_imStep)
	{
		filter_row_ptr = filter_im_ptr;
		filter_pix_ptr = filter_row_ptr;
		cp_dst_ptr[0] = filter_pix_ptr[0];
		cp_dst_ptr[1] = filter_pix_ptr[1];
		cp_dst_ptr[2] = filter_pix_ptr[2];
		filter_pix_ptr += zq_mm_align_size;
		cp_dst_ptr[3] = filter_pix_ptr[0];
		cp_dst_ptr[4] = filter_pix_ptr[1];
		cp_dst_ptr[5] = filter_pix_ptr[2];
		
		filter_row_ptr += filter_widthStep;
		filter_pix_ptr = filter_row_ptr;
		cp_dst_ptr[6] = filter_pix_ptr[0];
		cp_dst_ptr[7] = filter_pix_ptr[1];
		cp_dst_ptr[8] = filter_pix_ptr[2];
		filter_pix_ptr += zq_mm_align_size;
		cp_dst_ptr[9] = filter_pix_ptr[0];
		cp_dst_ptr[10] = filter_pix_ptr[1];
		cp_dst_ptr[11] = filter_pix_ptr[2];

		for (kc = 12; kc < matrix_B_rows; kc++)
			cp_dst_ptr[kc] = 0;
	}

	t2 = omp_get_wtime();


	matrix_A_row_ptr = matrix_A;
	for (out_n = 0, in_im_ptr = in_tensor4D_data;
		out_n < out_N;
		out_n++, in_im_ptr += in_imStep)
	{
		for (out_h = 0, in_row_ptr = in_im_ptr; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
			{
				matrix_A_col_ptr = matrix_A_row_ptr;
				cur_in_row_ptr = in_pix_ptr;
				cur_in_pix_ptr = cur_in_row_ptr;
				matrix_A_col_ptr[0] = cur_in_pix_ptr[0];
				matrix_A_col_ptr[1] = cur_in_pix_ptr[1];
				matrix_A_col_ptr[2] = cur_in_pix_ptr[2];
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;
				matrix_A_col_ptr[3] = cur_in_pix_ptr[0];
				matrix_A_col_ptr[4] = cur_in_pix_ptr[1];
				matrix_A_col_ptr[5] = cur_in_pix_ptr[2];
				cur_in_row_ptr += dilate_H_mul_in_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr;
				matrix_A_col_ptr[6] = cur_in_pix_ptr[0];
				matrix_A_col_ptr[7] = cur_in_pix_ptr[1];
				matrix_A_col_ptr[8] = cur_in_pix_ptr[2];
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;
				matrix_A_col_ptr[9] = cur_in_pix_ptr[0];
				matrix_A_col_ptr[10] = cur_in_pix_ptr[1];
				matrix_A_col_ptr[11] = cur_in_pix_ptr[2];
				for (kc = 12; kc < matrix_B_rows; kc++)
					matrix_A_col_ptr[kc] = 0;
				matrix_A_row_ptr += matrix_A_cols;
			}
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
	out_row_idx = 0;
	matrix_C_row_ptr = matrix_C;
	for (out_n = 0, out_im_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_im_ptr += out_imStep)
	{
		for (out_h = 0, out_row_ptr = out_im_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W; out_w++, out_pix_ptr += zq_mm_align_size)
			{
				for (kc = 0, out_slice_ptr = out_pix_ptr; kc < out_C - zq_mm_align_size; kc += zq_mm_align_size, out_slice_ptr += out_sliceStep)
					zq_mm_store_ps(out_slice_ptr, zq_mm_loadu_ps(matrix_C_row_ptr + kc));
				for (i = 0; i + kc < out_C; i++)
					out_slice_ptr[i] = matrix_C_row_ptr[i + kc];
				matrix_C_row_ptr += matrix_B_cols;
			}
		}
	}
	t5 = omp_get_wtime();

	if (buffer == 0)
	{
		_aligned_free(matrix_A);
		_aligned_free(matrix_Bt);
		_aligned_free(matrix_C);
	}
}

/*in_pixStep can be different with filter_pixStep,
and the aligned channels should be set to zero*/
void zq_cnn_conv_no_padding_gemm_nchwc_kernel3x3_C3(
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
	int stride_H,
	int stride_W,
	int dilation_H,
	int dilation_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_widthStep,
	int out_sliceStep,
	int out_imStep,
	void** buffer,
	__int64* buffer_len
)
{
	/************** image to col **************/
	int align_C = (in_C + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	int in_widthStep_mul_stride_H = in_widthStep*stride_H;
	int in_pixelStep_mul_stride_W = zq_mm_align_size*stride_W;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*zq_mm_align_size;
	int filter_pixStep_mul_filter_W = zq_mm_align_size*filter_W;
	int matrix_A_cols = filter_H*filter_W*align_C;
	int matrix_A_rows = out_N*out_H*out_W;
	int matrix_B_cols = filter_N;
	int matrix_B_rows = (filter_H*filter_W*3+zq_mm_align_size-1)/zq_mm_align_size*zq_mm_align_size;
	int out_NHW = out_N*out_H*out_W;
	__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 need_B_buffer_len_align32 = (matrix_B_rows*filter_N * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 need_C_buffer_len_align32 = need_C_buffer_len_align32 = (out_NHW*filter_N * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 total_need_buffer_len;
	zq_base_type* matrix_A = 0;
	zq_base_type* matrix_Bt = 0;
	const zq_base_type *in_im_ptr, *in_row_ptr, *in_pix_ptr, *cur_in_row_ptr, *cur_in_pix_ptr;
	const zq_base_type *filter_im_ptr, *filter_row_ptr, *filter_pix_ptr;
	int out_n, out_h, out_w, kn, kc, i;
	zq_base_type* matrix_A_row_ptr, *matrix_A_col_ptr, *cp_dst_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_im_ptr;
	zq_base_type* matrix_C = 0, *matrix_C_row_ptr;
	int out_row_idx;

	double t1, t2, t3, t4, t5;
	t1 = omp_get_wtime();
	total_need_buffer_len = need_A_buffer_len_align32 + need_B_buffer_len_align32 + need_C_buffer_len_align32;
	if (buffer == 0)
	{
		matrix_A = (zq_base_type*)_aligned_malloc(need_A_buffer_len_align32, 32);
		matrix_Bt = (zq_base_type*)_aligned_malloc(need_B_buffer_len_align32, 32);
		matrix_C = (zq_base_type*)_aligned_malloc(need_C_buffer_len_align32, 32);
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
		matrix_Bt = (zq_base_type*)((char*)(*buffer) + need_A_buffer_len_align32);
		matrix_C = (zq_base_type*)((char*)(*buffer) + need_A_buffer_len_align32 + need_B_buffer_len_align32);
	}
	
	cp_dst_ptr = matrix_Bt;
	for (kn = 0, filter_im_ptr = filters_data; kn < filter_N; kn++, filter_im_ptr += filter_imStep)
	{
		filter_row_ptr = filter_im_ptr;
		filter_pix_ptr = filter_row_ptr;
		cp_dst_ptr[0] = filter_pix_ptr[0];
		cp_dst_ptr[1] = filter_pix_ptr[1];
		cp_dst_ptr[2] = filter_pix_ptr[2];
		filter_pix_ptr += zq_mm_align_size;
		cp_dst_ptr[3] = filter_pix_ptr[0];
		cp_dst_ptr[4] = filter_pix_ptr[1];
		cp_dst_ptr[5] = filter_pix_ptr[2];
		filter_pix_ptr += zq_mm_align_size;
		cp_dst_ptr[6] = filter_pix_ptr[0];
		cp_dst_ptr[7] = filter_pix_ptr[1];
		cp_dst_ptr[8] = filter_pix_ptr[2];
		
		filter_row_ptr += filter_widthStep;
		filter_pix_ptr = filter_row_ptr;
		cp_dst_ptr[9] = filter_pix_ptr[0];
		cp_dst_ptr[10] = filter_pix_ptr[1];
		cp_dst_ptr[11] = filter_pix_ptr[2];
		filter_pix_ptr += zq_mm_align_size;
		cp_dst_ptr[12] = filter_pix_ptr[0];
		cp_dst_ptr[13] = filter_pix_ptr[1];
		cp_dst_ptr[14] = filter_pix_ptr[2];
		filter_pix_ptr += zq_mm_align_size;
		cp_dst_ptr[15] = filter_pix_ptr[0];
		cp_dst_ptr[16] = filter_pix_ptr[1];
		cp_dst_ptr[17] = filter_pix_ptr[2];

		filter_row_ptr += filter_widthStep;
		filter_pix_ptr = filter_row_ptr;
		cp_dst_ptr[18] = filter_pix_ptr[0];
		cp_dst_ptr[19] = filter_pix_ptr[1];
		cp_dst_ptr[20] = filter_pix_ptr[2];
		filter_pix_ptr += zq_mm_align_size;
		cp_dst_ptr[21] = filter_pix_ptr[0];
		cp_dst_ptr[22] = filter_pix_ptr[1];
		cp_dst_ptr[23] = filter_pix_ptr[2];
		filter_pix_ptr += zq_mm_align_size;
		cp_dst_ptr[24] = filter_pix_ptr[0];
		cp_dst_ptr[25] = filter_pix_ptr[1];
		cp_dst_ptr[26] = filter_pix_ptr[2];
		
		for (kc = 27; kc < matrix_B_rows; kc++)
			cp_dst_ptr[kc] = 0;
	}

	t2 = omp_get_wtime();


	matrix_A_row_ptr = matrix_A;
	for (out_n = 0, in_im_ptr = in_tensor4D_data;
		out_n < out_N;
		out_n++, in_im_ptr += in_imStep)
	{
		for (out_h = 0, in_row_ptr = in_im_ptr; out_h < out_H; out_h++, in_row_ptr += in_widthStep_mul_stride_H)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr; out_w < out_W; out_w++, in_pix_ptr += in_pixelStep_mul_stride_W)
			{
				matrix_A_col_ptr = matrix_A_row_ptr;
				cur_in_row_ptr = in_pix_ptr;
				cur_in_pix_ptr = cur_in_row_ptr;
				matrix_A_col_ptr[0] = cur_in_pix_ptr[0];
				matrix_A_col_ptr[1] = cur_in_pix_ptr[1];
				matrix_A_col_ptr[2] = cur_in_pix_ptr[2];
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;
				matrix_A_col_ptr[3] = cur_in_pix_ptr[0];
				matrix_A_col_ptr[4] = cur_in_pix_ptr[1];
				matrix_A_col_ptr[5] = cur_in_pix_ptr[2];
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;
				matrix_A_col_ptr[6] = cur_in_pix_ptr[0];
				matrix_A_col_ptr[7] = cur_in_pix_ptr[1];
				matrix_A_col_ptr[8] = cur_in_pix_ptr[2];
				cur_in_row_ptr += dilate_H_mul_in_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr;
				matrix_A_col_ptr[9] = cur_in_pix_ptr[0];
				matrix_A_col_ptr[10] = cur_in_pix_ptr[1];
				matrix_A_col_ptr[11] = cur_in_pix_ptr[2];
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;
				matrix_A_col_ptr[12] = cur_in_pix_ptr[0];
				matrix_A_col_ptr[13] = cur_in_pix_ptr[1];
				matrix_A_col_ptr[14] = cur_in_pix_ptr[2];
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;
				matrix_A_col_ptr[15] = cur_in_pix_ptr[0];
				matrix_A_col_ptr[16] = cur_in_pix_ptr[1];
				matrix_A_col_ptr[17] = cur_in_pix_ptr[2];
				cur_in_row_ptr += dilate_H_mul_in_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr;
				matrix_A_col_ptr[18] = cur_in_pix_ptr[0];
				matrix_A_col_ptr[19] = cur_in_pix_ptr[1];
				matrix_A_col_ptr[20] = cur_in_pix_ptr[2];
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;
				matrix_A_col_ptr[21] = cur_in_pix_ptr[0];
				matrix_A_col_ptr[22] = cur_in_pix_ptr[1];
				matrix_A_col_ptr[23] = cur_in_pix_ptr[2];
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;
				matrix_A_col_ptr[24] = cur_in_pix_ptr[0];
				matrix_A_col_ptr[25] = cur_in_pix_ptr[1];
				matrix_A_col_ptr[26] = cur_in_pix_ptr[2];
				for (kc = 27; kc < matrix_B_rows; kc++)
					matrix_A_col_ptr[kc] = 0;
				matrix_A_row_ptr += matrix_A_cols;
			}
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
	out_row_idx = 0;
	matrix_C_row_ptr = matrix_C;
	for (out_n = 0, out_im_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_im_ptr += out_imStep)
	{
		for (out_h = 0, out_row_ptr = out_im_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W; out_w++, out_pix_ptr += zq_mm_align_size)
			{
				for (kc = 0, out_slice_ptr = out_pix_ptr; kc < out_C - zq_mm_align_size; kc += zq_mm_align_size, out_slice_ptr += out_sliceStep)
					zq_mm_store_ps(out_slice_ptr, zq_mm_loadu_ps(matrix_C_row_ptr + kc));
				for (i = 0; i + kc < out_C; i++)
					out_slice_ptr[i] = matrix_C_row_ptr[i + kc];
				matrix_C_row_ptr += matrix_B_cols;
			}
		}
	}
	t5 = omp_get_wtime();

	if (buffer == 0)
	{
		_aligned_free(matrix_A);
		_aligned_free(matrix_Bt);
		_aligned_free(matrix_C);
	}
}