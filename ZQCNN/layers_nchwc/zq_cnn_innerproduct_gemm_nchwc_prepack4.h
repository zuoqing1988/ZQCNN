/*zq_mm_align_size must be 4*/
void zq_cnn_innerproduct_gemm_nchwc_prepack4(
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
	int packed_B_step = H*W*paddedC * 4;
	int i, ii, c, h, w;
	__int64 need_buffer_size = (__int64)packed_B_step*div4_size * sizeof(zq_base_type);
	zq_base_type* B_buffer, *dst_ptr;
	const zq_base_type* im_ptr0, *im_ptr1, *im_ptr2, *im_ptr3;
	const zq_base_type* slice_ptr0, *slice_ptr1, *slice_ptr2, *slice_ptr3;
	const zq_base_type* row_ptr0, *row_ptr1, *row_ptr2, *row_ptr3;
	const zq_base_type* pix_ptr0, *pix_ptr1, *pix_ptr2, *pix_ptr3;
	if (*buffer_len < need_buffer_size)
	{
		if (*buffer != 0)
			_aligned_free(*buffer);
		*buffer = _aligned_malloc(need_buffer_size, 32);
		*buffer_len = need_buffer_size;
	}
	B_buffer = (zq_base_type*)(*buffer);
	memset(B_buffer, 0, need_buffer_size);
	for (i = 0; i < div4_size - 1; i++)
	{
		dst_ptr = B_buffer + packed_B_step*i;
		im_ptr0 = filters_data + (i * 4)*filter_imStep;
		im_ptr1 = filters_data + (i * 4 + 1)*filter_imStep;
		im_ptr2 = filters_data + (i * 4 + 2)*filter_imStep;
		im_ptr3 = filters_data + (i * 4 + 3)*filter_imStep;
		slice_ptr0 = im_ptr0;
		slice_ptr1 = im_ptr1;
		slice_ptr2 = im_ptr2;
		slice_ptr3 = im_ptr3;
		for (c = 0; c < C; c += zq_mm_align_size)
		{
			row_ptr0 = slice_ptr0;
			row_ptr1 = slice_ptr1;
			row_ptr2 = slice_ptr2;
			row_ptr3 = slice_ptr3;
			for (h = 0; h < H; h++)
			{
				pix_ptr0 = row_ptr0;
				pix_ptr1 = row_ptr1;
				pix_ptr2 = row_ptr2;
				pix_ptr3 = row_ptr3;
				for (w = 0; w < H; w++)
				{
					zq_mm_store_ps(dst_ptr, zq_mm_load_ps(pix_ptr0));
					zq_mm_store_ps(dst_ptr + zq_mm_align_size, zq_mm_load_ps(pix_ptr1));
					zq_mm_store_ps(dst_ptr + zq_mm_align_size2, zq_mm_load_ps(pix_ptr2));
					zq_mm_store_ps(dst_ptr + zq_mm_align_size3, zq_mm_load_ps(pix_ptr3));
					dst_ptr += zq_mm_align_size4;
					pix_ptr0 += zq_mm_align_size;
					pix_ptr1 += zq_mm_align_size;
					pix_ptr2 += zq_mm_align_size;
					pix_ptr3 += zq_mm_align_size;
				}
				row_ptr0 += filter_widthStep;
				row_ptr1 += filter_widthStep;
				row_ptr2 += filter_widthStep;
				row_ptr3 += filter_widthStep;
			}
			
			slice_ptr0 += filter_sliceStep;
			slice_ptr1 += filter_sliceStep;
			slice_ptr2 += filter_sliceStep;
			slice_ptr3 += filter_sliceStep;
		}
	}

	for (i = div4_size * 4 - 4; i < N; i++)
	{
		ii = i % 4;
		dst_ptr = B_buffer + packed_B_step*(div4_size - 1) + zq_mm_align_size*ii;
		im_ptr0 = filters_data + i *filter_imStep;
		slice_ptr0 = im_ptr0;
		for (c = 0; c < C; c += zq_mm_align_size)
		{
			row_ptr0 = slice_ptr0;
			for (h = 0; h < H; h++)
			{
				pix_ptr0 = row_ptr0;
				for (w = 0; w < H; w++)
				{
					zq_mm_store_ps(dst_ptr, zq_mm_load_ps(pix_ptr0));
					dst_ptr += zq_mm_align_size4;
					pix_ptr0 += zq_mm_align_size;
				}
				row_ptr0 += filter_widthStep;
			}
			slice_ptr0 += filter_sliceStep;
		}
	}
}