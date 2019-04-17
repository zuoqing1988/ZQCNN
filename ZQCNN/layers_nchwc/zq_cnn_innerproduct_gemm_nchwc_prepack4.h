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

#if __ARM_NEON && __ARM_NEON_ARMV8

/*zq_mm_align_size must be 4*/
void zq_cnn_innerproduct_gemm_nchwc_prepack8_other(
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
	int alignN = (N + 3) >> 2 << 2;
	int div8_num = alignN >> 3;
	int div4_num = (alignN - (div8_num << 3)) >> 2;
	int all_pack_num = div8_num + div4_num;
	int alignC = (C + 3) >> 2 << 2;
	int packed_B_step = H*W*alignC * 8;
	int i, start, ii, c, h, w;
	__int64 need_buffer_size = (__int64)packed_B_step*all_pack_num * sizeof(zq_base_type);
	zq_base_type* B_buffer, *dst_ptr;
	const zq_base_type* im_ptr0, *im_ptr1, *im_ptr2, *im_ptr3, *im_ptr4, *im_ptr5, *im_ptr6, *im_ptr7;
	const zq_base_type* slice_ptr0, *slice_ptr1, *slice_ptr2, *slice_ptr3, *slice_ptr4, *slice_ptr5, *slice_ptr6, *slice_ptr7;
	const zq_base_type* row_ptr0, *row_ptr1, *row_ptr2, *row_ptr3, *row_ptr4, *row_ptr5, *row_ptr6, *row_ptr7;
	const zq_base_type* pix_ptr0, *pix_ptr1, *pix_ptr2, *pix_ptr3, *pix_ptr4, *pix_ptr5, *pix_ptr6, *pix_ptr7;
	if (*buffer_len < need_buffer_size)
	{
		if (*buffer != 0)
			_aligned_free(*buffer);
		*buffer = _aligned_malloc(need_buffer_size, 32);
		*buffer_len = need_buffer_size;
	}
	B_buffer = (zq_base_type*)(*buffer);
	memset(B_buffer, 0, need_buffer_size);
	for (i = 0; i < div8_num; i++)
	{
		dst_ptr = B_buffer + packed_B_step*i;
		im_ptr0 = filters_data + (i * 8)*filter_imStep;
		im_ptr1 = filters_data + (i * 8 + 1)*filter_imStep;
		im_ptr2 = filters_data + (i * 8 + 2)*filter_imStep;
		im_ptr3 = filters_data + (i * 8 + 3)*filter_imStep;
		im_ptr4 = filters_data + (i * 8 + 4)*filter_imStep;
		im_ptr5 = filters_data + (i * 8 + 5)*filter_imStep;
		im_ptr6 = filters_data + (i * 8 + 6)*filter_imStep;
		im_ptr7 = filters_data + (i * 8 + 7)*filter_imStep;
		slice_ptr0 = im_ptr0;
		slice_ptr1 = im_ptr1;
		slice_ptr2 = im_ptr2;
		slice_ptr3 = im_ptr3;
		slice_ptr4 = im_ptr4;
		slice_ptr5 = im_ptr5;
		slice_ptr6 = im_ptr6;
		slice_ptr7 = im_ptr7;
		for (c = 0; c < C; c += zq_mm_align_size)
		{
			row_ptr0 = slice_ptr0;
			row_ptr1 = slice_ptr1;
			row_ptr2 = slice_ptr2;
			row_ptr3 = slice_ptr3;
			row_ptr4 = slice_ptr4;
			row_ptr5 = slice_ptr5;
			row_ptr6 = slice_ptr6;
			row_ptr7 = slice_ptr7;
			for (h = 0; h < H; h++)
			{
				pix_ptr0 = row_ptr0;
				pix_ptr1 = row_ptr1;
				pix_ptr2 = row_ptr2;
				pix_ptr3 = row_ptr3;
				pix_ptr4 = row_ptr4;
				pix_ptr5 = row_ptr5;
				pix_ptr6 = row_ptr6;
				pix_ptr7 = row_ptr7;
				for (w = 0; w < H; w++)
				{
					for (ii = 0; ii < zq_mm_align_size; ii++)
					{
						*(dst_ptr++) = pix_ptr0[ii];
						*(dst_ptr++) = pix_ptr1[ii];
						*(dst_ptr++) = pix_ptr2[ii];
						*(dst_ptr++) = pix_ptr3[ii];
						*(dst_ptr++) = pix_ptr4[ii];
						*(dst_ptr++) = pix_ptr5[ii];
						*(dst_ptr++) = pix_ptr6[ii];
						*(dst_ptr++) = pix_ptr7[ii];
					}
					pix_ptr0 += zq_mm_align_size;
					pix_ptr1 += zq_mm_align_size;
					pix_ptr2 += zq_mm_align_size;
					pix_ptr3 += zq_mm_align_size;
					pix_ptr4 += zq_mm_align_size;
					pix_ptr5 += zq_mm_align_size;
					pix_ptr6 += zq_mm_align_size;
					pix_ptr7 += zq_mm_align_size;
				}
				row_ptr0 += filter_widthStep;
				row_ptr1 += filter_widthStep;
				row_ptr2 += filter_widthStep;
				row_ptr3 += filter_widthStep;
				row_ptr4 += filter_widthStep;
				row_ptr5 += filter_widthStep;
				row_ptr6 += filter_widthStep;
				row_ptr7 += filter_widthStep;
			}

			slice_ptr0 += filter_sliceStep;
			slice_ptr1 += filter_sliceStep;
			slice_ptr2 += filter_sliceStep;
			slice_ptr3 += filter_sliceStep;
			slice_ptr4 += filter_sliceStep;
			slice_ptr5 += filter_sliceStep;
			slice_ptr6 += filter_sliceStep;
			slice_ptr7 += filter_sliceStep;
		}
	}

	for (i = 0; i < div4_num; i++)
	{
		dst_ptr = B_buffer + packed_B_step*(div8_num + i);
		start = (div8_num << 3) + (i << 2);
		im_ptr0 = filters_data + start*filter_imStep;
		im_ptr1 = (start + 1 >= N) ? 0 : (filters_data + (start + 1)*filter_imStep);
		im_ptr2 = (start + 2 >= N) ? 0 : (filters_data + (start + 2)*filter_imStep);
		im_ptr3 = (start + 3 >= N) ? 0 : (filters_data + (start + 3)*filter_imStep);
		slice_ptr0 = im_ptr0;
		slice_ptr1 = im_ptr1;
		slice_ptr2 = im_ptr2;
		slice_ptr3 = im_ptr3;
		for (c = 0; c < C; c += zq_mm_align_size)
		{
			row_ptr0 = slice_ptr0;
			row_ptr1 = slice_ptr1 == 0 ? 0 : slice_ptr1;
			row_ptr2 = slice_ptr2 == 0 ? 0 : slice_ptr2;
			row_ptr3 = slice_ptr3 == 0 ? 0 : slice_ptr3;
			for (h = 0; h < H; h++)
			{
				pix_ptr0 = row_ptr0;
				pix_ptr1 = row_ptr1 == 0 ? 0 : row_ptr1;
				pix_ptr2 = row_ptr2 == 0 ? 0 : row_ptr2;
				pix_ptr3 = row_ptr3 == 0 ? 0 : row_ptr3;
				for (w = 0; w < H; w++)
				{
					for (ii = 0; ii < zq_mm_align_size; ii++)
					{
						*(dst_ptr++) = pix_ptr0[ii];
						*(dst_ptr++) = pix_ptr1 == 0 ? 0 : pix_ptr1[ii];
						*(dst_ptr++) = pix_ptr2 == 0 ? 0 : pix_ptr2[ii];
						*(dst_ptr++) = pix_ptr3 == 0 ? 0 : pix_ptr3[ii];
					}
					pix_ptr0 += zq_mm_align_size;
					if(pix_ptr1 != 0)
						pix_ptr1 += zq_mm_align_size;
					if(pix_ptr2 != 0)
						pix_ptr2 += zq_mm_align_size;
					if(pix_ptr3 != 0)
						pix_ptr3 += zq_mm_align_size;
				}
				row_ptr0 += filter_widthStep;
				if(row_ptr1 != 0)
					row_ptr1 += filter_widthStep;
				if(row_ptr2 != 0)
					row_ptr2 += filter_widthStep;
				if(row_ptr3 != 0)
					row_ptr3 += filter_widthStep;
			}

			slice_ptr0 += filter_sliceStep;
			if(slice_ptr1 != 0)
				slice_ptr1 += filter_sliceStep;
			if(slice_ptr2 != 0)
				slice_ptr2 += filter_sliceStep;
			if(slice_ptr3 != 0)
				slice_ptr3 += filter_sliceStep;
		}
	}
}

#endif