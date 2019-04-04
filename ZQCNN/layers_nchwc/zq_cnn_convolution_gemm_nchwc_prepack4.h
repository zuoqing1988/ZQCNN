/*zq_mm_align_size must be 4*/
void zq_cnn_convolution_gemm_nchwc_prepack4_kernel1x1(
	const zq_base_type* filters_data,
	int N,
	int H,
	int W,
	int C,
	int filter_widthStep,
	int filter_sliceStep,
	int filter_imStep,
	void** buffer,
	__int64* buffer_len
)
{
	int div4_size = (N + 3) >> 2;
	int paddedC = (C + 3) >> 2 << 2;
	int packed_B_step = paddedC * 4;
	int i,ii,c;
	__int64 need_buffer_size = (__int64)packed_B_step*div4_size * sizeof(zq_base_type);
	zq_base_type* B_buffer,*dst_ptr;
	const zq_base_type* src_ptr0, *src_ptr1, *src_ptr2, *src_ptr3;
	if (*buffer_len < need_buffer_size)
	{
		if(*buffer != 0)
			_aligned_free(*buffer);
		*buffer = _aligned_malloc(need_buffer_size,32);
		*buffer_len = need_buffer_size;
	}
	B_buffer = (zq_base_type*)(*buffer);
	memset(B_buffer, 0, need_buffer_size);
	for (i = 0; i < div4_size-1; i++)
	{
		dst_ptr = B_buffer + packed_B_step*i;
		src_ptr0 = filters_data + (i * 4)*filter_imStep;
		src_ptr1 = filters_data + (i * 4+1)*filter_imStep;
		src_ptr2 = filters_data + (i * 4+2)*filter_imStep;
		src_ptr3 = filters_data + (i * 4+3)*filter_imStep;
		for (c = 0; c < C; c += zq_mm_align_size)
		{
			zq_mm_store_ps(dst_ptr, zq_mm_load_ps(src_ptr0));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size, zq_mm_load_ps(src_ptr1));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size2, zq_mm_load_ps(src_ptr2));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size3, zq_mm_load_ps(src_ptr3));
			dst_ptr += zq_mm_align_size4;
			src_ptr0 += filter_sliceStep;
			src_ptr1 += filter_sliceStep;
			src_ptr2 += filter_sliceStep;
			src_ptr3 += filter_sliceStep;
		}
	}

	for (i = div4_size * 4 - 4; i < N; i++)
	{
		ii = i % 4;
		dst_ptr = B_buffer + packed_B_step*(div4_size - 1) + zq_mm_align_size*ii;
		src_ptr0 = filters_data + i *filter_imStep;
		for (c = 0; c < C; c += zq_mm_align_size)
		{
			zq_mm_store_ps(dst_ptr, zq_mm_load_ps(src_ptr0));
			dst_ptr += zq_mm_align_size4;
			src_ptr0 += filter_sliceStep;
		}
	}
}