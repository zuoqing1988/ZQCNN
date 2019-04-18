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

#if __ARM_NEON && __ARM_NEON_ARMV8

/*zq_mm_align_size must be 4*/
void zq_cnn_convolution_gemm_nchwc_prepack8_other_kernel1x1(
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
	int packed_B_step = alignC * 8;
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
			for (ii = 0; ii < zq_mm_align_size; ii++)
			{
				*(dst_ptr++) = slice_ptr0[ii];
				*(dst_ptr++) = slice_ptr1[ii];
				*(dst_ptr++) = slice_ptr2[ii];
				*(dst_ptr++) = slice_ptr3[ii];
				*(dst_ptr++) = slice_ptr4[ii];
				*(dst_ptr++) = slice_ptr5[ii];
				*(dst_ptr++) = slice_ptr6[ii];
				*(dst_ptr++) = slice_ptr7[ii];
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

			for (ii = 0; ii < zq_mm_align_size; ii++)
			{
				*(dst_ptr++) = slice_ptr0[ii];
				*(dst_ptr++) = slice_ptr1 == 0 ? 0 : slice_ptr1[ii];
				*(dst_ptr++) = slice_ptr2 == 0 ? 0 : slice_ptr2[ii];
				*(dst_ptr++) = slice_ptr3 == 0 ? 0 : slice_ptr3[ii];
			}

			slice_ptr0 += filter_sliceStep;
			if (slice_ptr1 != 0)
				slice_ptr1 += filter_sliceStep;
			if (slice_ptr2 != 0)
				slice_ptr2 += filter_sliceStep;
			if (slice_ptr3 != 0)
				slice_ptr3 += filter_sliceStep;
		}
	}
}

#endif

/*zq_mm_align_size must be 4*/
void zq_cnn_convolution_gemm_nchwc_prepack4_kernel3x3_C3C4(
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
	int paddedC = 36;
	int packed_B_step = paddedC * 4;
	int i, ii, h, w;
	__int64 need_buffer_size = (__int64)packed_B_step*div4_size * sizeof(zq_base_type);
	zq_base_type* B_buffer, *dst_ptr;
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
		row_ptr0 = filters_data + (i * 4)*filter_imStep;
		row_ptr1 = filters_data + (i * 4 + 1)*filter_imStep;
		row_ptr2 = filters_data + (i * 4 + 2)*filter_imStep;
		row_ptr3 = filters_data + (i * 4 + 3)*filter_imStep;
		for (h = 0; h < 3; h++)
		{
			pix_ptr0 = row_ptr0;
			pix_ptr1 = row_ptr1;
			pix_ptr2 = row_ptr2;
			pix_ptr3 = row_ptr3;
			for (w = 0; w < 3; w++)
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
	}

	for (i = div4_size * 4 - 4; i < N; i++)
	{
		ii = i % 4;
		dst_ptr = B_buffer + packed_B_step*(div4_size - 1) + zq_mm_align_size*ii;
		row_ptr0 = filters_data + i *filter_imStep;
		for (h = 0; h < 3; h++)
		{
			pix_ptr0 = row_ptr0;
			for (w = 0; w < 3; w++)
			{
				zq_mm_store_ps(dst_ptr, zq_mm_load_ps(pix_ptr0));
				dst_ptr += zq_mm_align_size4;
				pix_ptr0 += zq_mm_align_size;
			}
			row_ptr0 += filter_widthStep;
		}
	}
}

#if __ARM_NEON && __ARM_NEON_ARMV8

/*zq_mm_align_size must be 4*/
void zq_cnn_convolution_gemm_nchwc_prepack8_other_kernel3x3_C3(
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
	int alignC = 36;
	int packed_B_step = alignC * 8;
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
		for (h = 0; h < 3; h++)
		{
			row_ptr0 = im_ptr0 + h*filter_widthStep;
			row_ptr1 = im_ptr1 + h*filter_widthStep;
			row_ptr2 = im_ptr2 + h*filter_widthStep;
			row_ptr3 = im_ptr3 + h*filter_widthStep;
			row_ptr4 = im_ptr4 + h*filter_widthStep;
			row_ptr5 = im_ptr5 + h*filter_widthStep;
			row_ptr6 = im_ptr6 + h*filter_widthStep;
			row_ptr7 = im_ptr7 + h*filter_widthStep;
			for (w = 0; w < 3; w++)
			{
				pix_ptr0 = row_ptr0 + w*zq_mm_align_size;
				pix_ptr1 = row_ptr1 + w*zq_mm_align_size;
				pix_ptr2 = row_ptr2 + w*zq_mm_align_size;
				pix_ptr3 = row_ptr3 + w*zq_mm_align_size;
				pix_ptr4 = row_ptr4 + w*zq_mm_align_size;
				pix_ptr5 = row_ptr5 + w*zq_mm_align_size;
				pix_ptr6 = row_ptr6 + w*zq_mm_align_size;
				pix_ptr7 = row_ptr7 + w*zq_mm_align_size;
				for (ii = 0; ii < 3; ii++)
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
			}
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
		for (h = 0; h < 3; h++)
		{
			row_ptr0 = im_ptr0 + h*filter_widthStep;
			row_ptr1 = im_ptr1 == 0 ? 0 : im_ptr1 + h*filter_widthStep;
			row_ptr2 = im_ptr2 == 0 ? 0 : im_ptr2 + h*filter_widthStep;
			row_ptr3 = im_ptr3 == 0 ? 0 : im_ptr3 + h*filter_widthStep;
			for (w = 0; w < 3; w++)
			{
				pix_ptr0 = row_ptr0 + w*zq_mm_align_size;
				pix_ptr1 = row_ptr1 == 0 ? 0 : row_ptr1 + w*zq_mm_align_size;
				pix_ptr2 = row_ptr2 == 0 ? 0 : row_ptr2 + w*zq_mm_align_size;
				pix_ptr3 = row_ptr3 == 0 ? 0 : row_ptr3 + w*zq_mm_align_size;
				for (ii = 0; ii < 3; ii++)
				{
					*(dst_ptr++) = pix_ptr0[ii];
					*(dst_ptr++) = pix_ptr1 == 0 ? 0 : pix_ptr1[ii];
					*(dst_ptr++) = pix_ptr2 == 0 ? 0 : pix_ptr2[ii];
					*(dst_ptr++) = pix_ptr3 == 0 ? 0 : pix_ptr3[ii];
				}
			}
		}
	}
}

#endif