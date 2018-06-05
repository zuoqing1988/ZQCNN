
void zq_cnn_addbias_32f_align(
	float* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* bias_data
)
{
	//zq_mm_type bias_v;
	int n, h, w, c;
	float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
	const float* bias_ptr;

#if 1

	if (in_C%zq_mm_align_size_mul_32 == 0)
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr,bias_ptr =bias_data; c < in_C; c += zq_mm_align_size_mul_32)
					{
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;

					}
				}
			}
		}
	}
	else if (in_C%zq_mm_align_size_mul_16 == 0)
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr, bias_ptr = bias_data; c < in_C; c += zq_mm_align_size_mul_16)
					{
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;

					}
				}
			}
		}
	}
	else if (in_C%zq_mm_align_size_mul_8 == 0)
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr, bias_ptr = bias_data; c < in_C; c += zq_mm_align_size_mul_8)
					{
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
						c_ptr += zq_mm_align_size; bias_ptr += zq_mm_align_size;

					}
				}
			}
		}
	}
	else
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr, bias_ptr = bias_data; c < in_C; 
						c += zq_mm_align_size, c_ptr += zq_mm_align_size,bias_ptr+=zq_mm_align_size)
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
				}
			}
		}
	}

#else
	for (c = 0, c_ptr = in_tensor4D_data; c < in_C; c += zq_mm_align_size, c_ptr += zq_mm_align_size)
	{
		bias_v = zq_mm_load_ps(bias_data + c);
		for (n = 0, slice_ptr = c_ptr; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					zq_mm_store_ps(pix_ptr, zq_mm_add_ps(zq_mm_load_ps(pix_ptr), bias_v));
				}
			}
		}
	}
#endif

}
