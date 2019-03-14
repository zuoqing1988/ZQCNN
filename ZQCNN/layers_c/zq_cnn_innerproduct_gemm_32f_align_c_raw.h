/*in_pixStep must be equal to filter_pixStep,
and the aligned channels should be set to zero*/
void zq_cnn_innerproduct_gemm_32f_align_same_pixstep(
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
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_C,	// must be filter_N
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep,
	void** buffer,
	__int64 *buffer_len
)
{
	/************** image to col **************/
	int filter_pixStep_mul_filter_W = in_C*in_W;
	int matrix_A_cols = filter_sliceStep;
	int matrix_A_rows = 1;
	int matrix_B_cols = filter_N;
	int matrix_B_rows = filter_sliceStep;
	__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 need_B_buffer_len_align32 = 0;
	__int64 need_C_buffer_len_align32 = 0;
	__int64 total_need_buffer_len;
	zq_base_type* matrix_A = 0;
	const zq_base_type *cur_in_row_ptr;
	int out_n, kh, pp;
	zq_base_type* matrix_A_row_ptr, *matrix_A_col_ptr, *cp_dst_ptr;
	const zq_base_type* cp_src_ptr;
	zq_base_type* out_slice_ptr;
	const zq_base_type* in_slice_ptr;
	const zq_base_type* matrix_Bt = filters_data;
	zq_base_type* matrix_C = 0, *matrix_C_row_ptr;
	int out_row_idx;
	double t1, t2, t3, t4, t5;
	t1 = omp_get_wtime();
	int need_allocate_tmp_out = (out_pixelStep != filter_N) || (out_pixelStep != out_widthStep) || (out_widthStep != out_sliceStep);
	if (need_allocate_tmp_out)
	{
		need_C_buffer_len_align32 = (filter_N * sizeof(zq_base_type) + 31) / 32 * 32;
	}
	else
	{
		matrix_C = out_tensor4D_data;
	}
	total_need_buffer_len = need_A_buffer_len_align32 + need_B_buffer_len_align32 + need_C_buffer_len_align32;
	if (buffer == 0)
	{
		matrix_A = (zq_base_type*)_aligned_malloc(need_A_buffer_len_align32, 32);
		if (need_allocate_tmp_out)
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
		if (need_allocate_tmp_out)
			matrix_C = (zq_base_type*)((char*)(*buffer) + need_A_buffer_len_align32);
	}

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		t2 = omp_get_wtime();
		matrix_A_row_ptr = matrix_A;

		for (kh = 0, cur_in_row_ptr = in_slice_ptr, matrix_A_col_ptr = matrix_A_row_ptr;
			kh < in_H;
			kh++, cur_in_row_ptr += in_widthStep, matrix_A_col_ptr += filter_pixStep_mul_filter_W)
		{
			//memcpy(matrix_A_col_ptr, cur_in_row_ptr, sizeof(zq_base_type)*filter_pixStep_mul_filter_W);
			for (pp = 0, cp_src_ptr = cur_in_row_ptr, cp_dst_ptr = matrix_A_col_ptr;
				pp < filter_pixStep_mul_filter_W;
				pp += zq_mm_align_size, cp_src_ptr += zq_mm_align_size, cp_dst_ptr += zq_mm_align_size)
				zq_mm_store_ps(cp_dst_ptr, zq_mm_load_ps(cp_src_ptr));
		}

		t3 = omp_get_wtime();
		/*gemm*/
		zq_cblas_sgemm(zq_CblasRowMajor, zq_CblasNoTrans, zq_CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
			matrix_Bt, matrix_A_cols, 0.0f, matrix_C, matrix_B_cols);
		t4 = omp_get_wtime();
		if (need_allocate_tmp_out)
		{
			/*   col2im      */
			out_row_idx = 0;
			matrix_C_row_ptr = matrix_C;
			memcpy(out_tensor4D_data, matrix_C_row_ptr, sizeof(zq_base_type)*matrix_B_cols);
		}
		else
			matrix_C += out_sliceStep;
		t5 = omp_get_wtime();
	}

	if (buffer == 0)
	{
		_aligned_free(matrix_A);
		if (need_allocate_tmp_out)
			_aligned_free(matrix_C);
	}
	//printf("total: %.3f ms, alloc %.3f ms, makeA: %.3f ms, gemm: %.3f ms, copy_C: %.3f ms\n", 1000 * (t5 - t1), 1000 * (t2 - t1), 
	//	1000 * (t3 - t2), 1000 * (t4 - t3), 1000 * (t5 - t4));
}

/*in_pixStep must be equal to filter_pixStep,
and the aligned channels should be set to zero*/
void zq_cnn_innerproduct_gemm_32f_align_same_pixstep_batch(
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
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_C,	// must be filter_N
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep,
	void** buffer,
	__int64 *buffer_len
)
{
	/************** image to col **************/
	int filter_pixStep_mul_filter_W = filter_pixelStep*in_W;
	int matrix_A_cols = filter_sliceStep;
	int matrix_A_rows = out_N;
	int matrix_B_cols = filter_N;
	int matrix_B_rows = filter_sliceStep;
	__int64 need_A_buffer_len_align32 = (matrix_A_rows*matrix_A_cols * sizeof(zq_base_type) + 31) / 32 * 32;
	__int64 need_B_buffer_len_align32 = 0;
	__int64 need_C_buffer_len_align32 = 0;
	__int64 total_need_buffer_len;
	zq_base_type* matrix_A = 0;
	const zq_base_type* cur_in_row_ptr;
	int out_n, kh;
	zq_base_type* matrix_A_row_ptr, *matrix_A_col_ptr;
	zq_base_type* out_slice_ptr;
	const zq_base_type* in_slice_ptr;
	const zq_base_type* matrix_Bt = filters_data;
	zq_base_type* matrix_C = 0, *matrix_C_row_ptr = 0;
	int out_row_idx;
	int out_NHW = out_N;
	double t1, t2, t3, t4, t5, alloc_time, make_A_time, gemm_time;
	int need_allocate_tmp_out = (out_pixelStep != filter_N) || (out_pixelStep != out_widthStep) || (out_widthStep != out_sliceStep);

	t1 = omp_get_wtime();
	if (need_allocate_tmp_out)
	{
		need_C_buffer_len_align32 = (out_N*filter_N * sizeof(zq_base_type) + 31) / 32 * 32;
	}
	else
	{
		matrix_C = out_tensor4D_data;
	}
	total_need_buffer_len = need_A_buffer_len_align32 + need_B_buffer_len_align32 + need_C_buffer_len_align32;
	if (buffer == 0)
	{
		matrix_A = (zq_base_type*)_aligned_malloc(need_A_buffer_len_align32,32);
		if (need_allocate_tmp_out)
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
		if (need_allocate_tmp_out)
			matrix_C = (zq_base_type*)((char*)(*buffer) + need_A_buffer_len_align32);
	}
	t2 = omp_get_wtime();
	alloc_time = t2 - t1;

	matrix_A_row_ptr = matrix_A;
	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (kh = 0, cur_in_row_ptr = in_slice_ptr, matrix_A_col_ptr = matrix_A_row_ptr;
			kh < in_H;
			kh++, cur_in_row_ptr += in_widthStep, matrix_A_col_ptr += filter_pixStep_mul_filter_W)
		{
			memcpy(matrix_A_col_ptr, cur_in_row_ptr, sizeof(zq_base_type)*filter_pixStep_mul_filter_W);
		}
			
		matrix_A_row_ptr += matrix_A_cols;
	}

	t3 = omp_get_wtime();
	make_A_time = t3 - t2;
	/*gemm*/
#if __ARM_NEON && ZQ_CNN_USE_ZQ_GEMM && ZQ_CNN_USE_BLAS_GEMM
	if (0 == zq_gemm_32f_AnoTrans_Btrans_special(matrix_A_rows, matrix_B_cols, matrix_A_cols, matrix_A, matrix_A_cols,
		matrix_Bt, matrix_A_cols, matrix_C, matrix_B_cols))
	{
		zq_cblas_sgemm(zq_CblasRowMajor, zq_CblasNoTrans, zq_CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
			matrix_Bt, matrix_A_cols, 0, matrix_C, matrix_B_cols);
	}
#else
	zq_cblas_sgemm(zq_CblasRowMajor, zq_CblasNoTrans, zq_CblasTrans, matrix_A_rows, matrix_B_cols, matrix_A_cols, 1, matrix_A, matrix_A_cols,
		matrix_Bt, matrix_A_cols, 0, matrix_C, matrix_B_cols);
#endif
	t4 = omp_get_wtime();
	gemm_time = t4 - t3;


	if (need_allocate_tmp_out)
	{
		/*   col2im      */
		out_row_idx = 0;
		matrix_C_row_ptr = matrix_C;
		for (out_n = 0, out_slice_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_slice_ptr += out_sliceStep)
		{
			memcpy(out_slice_ptr, matrix_C_row_ptr, sizeof(zq_base_type)*matrix_B_cols);
			matrix_C_row_ptr += matrix_B_cols;
		}
	}
	if (buffer == 0)
	{
		_aligned_free(matrix_A);
		_aligned_free(matrix_C);
	}
	t5 = omp_get_wtime();

	//printf("innnerproduct_same_pixstep_batch total: %.3f ms, alloc %.3f ms, makeA: %.3f ms, gemm: %.3f ms\n", 1000 * (t5 - t1), 1000 * alloc_time, 
	//	1000 * make_A_time, 1000 *gemm_time);
}
