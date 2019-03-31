/*
y = max(0,x)+a*min(0,x)
*/
void zq_cnn_prelu_nchwc(
	zq_base_type* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
#if WITH_BIAS
	const zq_base_type* bias,
#endif
	const zq_base_type* slope
)
{
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	register zq_mm_type c0, c1, c2, c3;
	int n, h, w, c;
	register zq_mm_type slope_v;
#if WITH_BIAS
	register zq_mm_type bias_v;
#endif
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *im_ptr;
	register zq_mm_type zero_v = zq_mm_setzero_ps();

	if (in_W % 4 == 0)
	{
		for (n = 0, im_ptr = in_tensor4D_data; n < in_N; n++, im_ptr += in_imStep)
		{
			for (c = 0, slice_ptr = im_ptr; c < in_C; c += zq_mm_align_size, slice_ptr += in_sliceStep)
			{
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + c);
#endif
				slope_v = zq_mm_load_ps(slope + c);
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w+=4, pix_ptr += zq_mm_align_size4)
					{
						a0 = zq_mm_load_ps(pix_ptr);
						a1 = zq_mm_load_ps(pix_ptr + zq_mm_align_size);
						a2 = zq_mm_load_ps(pix_ptr + zq_mm_align_size2);
						a3 = zq_mm_load_ps(pix_ptr + zq_mm_align_size3);
#if WITH_BIAS
						a0 = zq_mm_add_ps(a0, bias_v);
						a1 = zq_mm_add_ps(a1, bias_v);
						a2 = zq_mm_add_ps(a2, bias_v);
						a3 = zq_mm_add_ps(a3, bias_v);
#endif
						b0 = zq_mm_min_ps(zero_v, a0);
						b1 = zq_mm_min_ps(zero_v, a1);
						b2 = zq_mm_min_ps(zero_v, a2);
						b3 = zq_mm_min_ps(zero_v, a3);
						c0 = zq_mm_max_ps(zero_v, a0);
						c1 = zq_mm_max_ps(zero_v, a1);
						c2 = zq_mm_max_ps(zero_v, a2);
						c3 = zq_mm_max_ps(zero_v, a3);
						zq_mm_store_ps(pix_ptr, zq_mm_fmadd_ps(slope_v, b0, c0));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(slope_v, b1, c1));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size2, zq_mm_fmadd_ps(slope_v, b2, c2));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size3, zq_mm_fmadd_ps(slope_v, b3, c3));
					}
				}
			}
		}
	}
	else if (in_W % 4 == 1)
	{
		for (n = 0, im_ptr = in_tensor4D_data; n < in_N; n++, im_ptr += in_imStep)
		{
			for (c = 0, slice_ptr = im_ptr; c < in_C; c += zq_mm_align_size, slice_ptr += in_sliceStep)
			{
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + c);
#endif
				slope_v = zq_mm_load_ps(slope + c);
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W-1; w += 4, pix_ptr += zq_mm_align_size4)
					{
						a0 = zq_mm_load_ps(pix_ptr);
						a1 = zq_mm_load_ps(pix_ptr + zq_mm_align_size);
						a2 = zq_mm_load_ps(pix_ptr + zq_mm_align_size2);
						a3 = zq_mm_load_ps(pix_ptr + zq_mm_align_size3);
#if WITH_BIAS
						a0 = zq_mm_add_ps(a0, bias_v);
						a1 = zq_mm_add_ps(a1, bias_v);
						a2 = zq_mm_add_ps(a2, bias_v);
						a3 = zq_mm_add_ps(a3, bias_v);
#endif
						b0 = zq_mm_min_ps(zero_v, a0);
						b1 = zq_mm_min_ps(zero_v, a1);
						b2 = zq_mm_min_ps(zero_v, a2);
						b3 = zq_mm_min_ps(zero_v, a3);
						c0 = zq_mm_max_ps(zero_v, a0);
						c1 = zq_mm_max_ps(zero_v, a1);
						c2 = zq_mm_max_ps(zero_v, a2);
						c3 = zq_mm_max_ps(zero_v, a3);
						zq_mm_store_ps(pix_ptr, zq_mm_fmadd_ps(slope_v, b0, c0));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(slope_v, b1, c1));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size2, zq_mm_fmadd_ps(slope_v, b2, c2));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size3, zq_mm_fmadd_ps(slope_v, b3, c3));
					}
					a0 = zq_mm_load_ps(pix_ptr);
#if WITH_BIAS
					a0 = zq_mm_add_ps(a0, bias_v);
#endif
					b0 = zq_mm_min_ps(zero_v, a0);
					c0 = zq_mm_max_ps(zero_v, a0);
					zq_mm_store_ps(pix_ptr, zq_mm_fmadd_ps(slope_v, b0, c0));
				}
			}
		}
	}
	else if (in_W % 4 == 2)
	{
		for (n = 0, im_ptr = in_tensor4D_data; n < in_N; n++, im_ptr += in_imStep)
		{
			for (c = 0, slice_ptr = im_ptr; c < in_C; c += zq_mm_align_size, slice_ptr += in_sliceStep)
			{
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + c);
#endif
				slope_v = zq_mm_load_ps(slope + c);
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W - 2; w += 4, pix_ptr += zq_mm_align_size4)
					{
						a0 = zq_mm_load_ps(pix_ptr);
						a1 = zq_mm_load_ps(pix_ptr + zq_mm_align_size);
						a2 = zq_mm_load_ps(pix_ptr + zq_mm_align_size2);
						a3 = zq_mm_load_ps(pix_ptr + zq_mm_align_size3);
#if WITH_BIAS
						a0 = zq_mm_add_ps(a0, bias_v);
						a1 = zq_mm_add_ps(a1, bias_v);
						a2 = zq_mm_add_ps(a2, bias_v);
						a3 = zq_mm_add_ps(a3, bias_v);
#endif
						b0 = zq_mm_min_ps(zero_v, a0);
						b1 = zq_mm_min_ps(zero_v, a1);
						b2 = zq_mm_min_ps(zero_v, a2);
						b3 = zq_mm_min_ps(zero_v, a3);
						c0 = zq_mm_max_ps(zero_v, a0);
						c1 = zq_mm_max_ps(zero_v, a1);
						c2 = zq_mm_max_ps(zero_v, a2);
						c3 = zq_mm_max_ps(zero_v, a3);
						zq_mm_store_ps(pix_ptr, zq_mm_fmadd_ps(slope_v, b0, c0));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(slope_v, b1, c1));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size2, zq_mm_fmadd_ps(slope_v, b2, c2));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size3, zq_mm_fmadd_ps(slope_v, b3, c3));
					}
					a0 = zq_mm_load_ps(pix_ptr);
					a1 = zq_mm_load_ps(pix_ptr + zq_mm_align_size);
#if WITH_BIAS
					a0 = zq_mm_add_ps(a0, bias_v);
					a1 = zq_mm_add_ps(a1, bias_v);
#endif
					b0 = zq_mm_min_ps(zero_v, a0);
					b1 = zq_mm_min_ps(zero_v, a1);
					c0 = zq_mm_max_ps(zero_v, a0);
					c1 = zq_mm_max_ps(zero_v, a1);
					zq_mm_store_ps(pix_ptr, zq_mm_fmadd_ps(slope_v, b0, c0));
					zq_mm_store_ps(pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(slope_v, b1, c1));
				}
			}
		}
	}
	else //if (in_W % 4 == 3)
	{
		for (n = 0, im_ptr = in_tensor4D_data; n < in_N; n++, im_ptr += in_imStep)
		{
			for (c = 0, slice_ptr = im_ptr; c < in_C; c += zq_mm_align_size, slice_ptr += in_sliceStep)
			{
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + c);
#endif
				slope_v = zq_mm_load_ps(slope + c);
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W - 3; w += 4, pix_ptr += zq_mm_align_size4)
					{
						a0 = zq_mm_load_ps(pix_ptr);
						a1 = zq_mm_load_ps(pix_ptr + zq_mm_align_size);
						a2 = zq_mm_load_ps(pix_ptr + zq_mm_align_size2);
						a3 = zq_mm_load_ps(pix_ptr + zq_mm_align_size3);
#if WITH_BIAS
						a0 = zq_mm_add_ps(a0, bias_v);
						a1 = zq_mm_add_ps(a1, bias_v);
						a2 = zq_mm_add_ps(a2, bias_v);
						a3 = zq_mm_add_ps(a3, bias_v);
#endif
						b0 = zq_mm_min_ps(zero_v, a0);
						b1 = zq_mm_min_ps(zero_v, a1);
						b2 = zq_mm_min_ps(zero_v, a2);
						b3 = zq_mm_min_ps(zero_v, a3);
						c0 = zq_mm_max_ps(zero_v, a0);
						c1 = zq_mm_max_ps(zero_v, a1);
						c2 = zq_mm_max_ps(zero_v, a2);
						c3 = zq_mm_max_ps(zero_v, a3);
						zq_mm_store_ps(pix_ptr, zq_mm_fmadd_ps(slope_v, b0, c0));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(slope_v, b1, c1));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size2, zq_mm_fmadd_ps(slope_v, b2, c2));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size3, zq_mm_fmadd_ps(slope_v, b3, c3));
					}
					a0 = zq_mm_load_ps(pix_ptr);
					a1 = zq_mm_load_ps(pix_ptr + zq_mm_align_size);
					a2 = zq_mm_load_ps(pix_ptr + zq_mm_align_size2);
#if WITH_BIAS
					a0 = zq_mm_add_ps(a0, bias_v);
					a1 = zq_mm_add_ps(a1, bias_v);
					a2 = zq_mm_add_ps(a2, bias_v);
#endif
					b0 = zq_mm_min_ps(zero_v, a0);
					b1 = zq_mm_min_ps(zero_v, a1);
					b2 = zq_mm_min_ps(zero_v, a2);
					c0 = zq_mm_max_ps(zero_v, a0);
					c1 = zq_mm_max_ps(zero_v, a1);
					c2 = zq_mm_max_ps(zero_v, a2);
					zq_mm_store_ps(pix_ptr, zq_mm_fmadd_ps(slope_v, b0, c0));
					zq_mm_store_ps(pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(slope_v, b1, c1));
					zq_mm_store_ps(pix_ptr + zq_mm_align_size2, zq_mm_fmadd_ps(slope_v, b2, c2));
				}
			}
		}
	}
	
}

void zq_cnn_prelu_nchwc_sure_slope_lessthan1(
	zq_base_type* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
#if WITH_BIAS
	const zq_base_type* bias,
#endif
	const zq_base_type* slope
)
{
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *im_ptr;
	register zq_mm_type slope_v;
#if WITH_BIAS
	register zq_mm_type bias_v;
#endif
	
	if (in_W % 4 == 0)
	{
		for (n = 0, im_ptr = in_tensor4D_data; n < in_N; n++, im_ptr += in_imStep)
		{
			for (c = 0, slice_ptr = im_ptr; c < in_C; c += zq_mm_align_size, slice_ptr += in_sliceStep)
			{
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + c);
#endif
				slope_v = zq_mm_load_ps(slope + c);
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w+=4, pix_ptr += zq_mm_align_size4)
					{
						a0 = zq_mm_load_ps(pix_ptr);
						a1 = zq_mm_load_ps(pix_ptr + zq_mm_align_size);
						a2 = zq_mm_load_ps(pix_ptr + zq_mm_align_size2);
						a3 = zq_mm_load_ps(pix_ptr + zq_mm_align_size3);
#if WITH_BIAS
						a0 = zq_mm_add_ps(a0, bias_v);
						a1 = zq_mm_add_ps(a1, bias_v);
						a2 = zq_mm_add_ps(a2, bias_v);
						a3 = zq_mm_add_ps(a3, bias_v);
#endif
						b0 = zq_mm_mul_ps(a0, slope_v);
						b1 = zq_mm_mul_ps(a1, slope_v);
						b2 = zq_mm_mul_ps(a2, slope_v);
						b3 = zq_mm_mul_ps(a3, slope_v);
						zq_mm_store_ps(pix_ptr, zq_mm_max_ps(a0, b0));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size, zq_mm_max_ps(a1, b1));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size2, zq_mm_max_ps(a2, b2));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size3, zq_mm_max_ps(a3, b3));
					}
				}
			}
		}
	}
	else if (in_W % 4 == 1)
	{
		for (n = 0, im_ptr = in_tensor4D_data; n < in_N; n++, im_ptr += in_imStep)
		{
			for (c = 0, slice_ptr = im_ptr; c < in_C; c += zq_mm_align_size, slice_ptr += in_sliceStep)
			{
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + c);
#endif
				slope_v = zq_mm_load_ps(slope + c);
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W-1; w += 4, pix_ptr += zq_mm_align_size4)
					{
						a0 = zq_mm_load_ps(pix_ptr);
						a1 = zq_mm_load_ps(pix_ptr + zq_mm_align_size);
						a2 = zq_mm_load_ps(pix_ptr + zq_mm_align_size2);
						a3 = zq_mm_load_ps(pix_ptr + zq_mm_align_size3);
#if WITH_BIAS
						a0 = zq_mm_add_ps(a0, bias_v);
						a1 = zq_mm_add_ps(a1, bias_v);
						a2 = zq_mm_add_ps(a2, bias_v);
						a3 = zq_mm_add_ps(a3, bias_v);
#endif
						b0 = zq_mm_mul_ps(a0, slope_v);
						b1 = zq_mm_mul_ps(a1, slope_v);
						b2 = zq_mm_mul_ps(a2, slope_v);
						b3 = zq_mm_mul_ps(a3, slope_v);
						zq_mm_store_ps(pix_ptr, zq_mm_max_ps(a0, b0));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size, zq_mm_max_ps(a1, b1));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size2, zq_mm_max_ps(a2, b2));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size3, zq_mm_max_ps(a3, b3));
					}
					a0 = zq_mm_load_ps(pix_ptr);
#if WITH_BIAS
					a0 = zq_mm_add_ps(a0, bias_v);
#endif
					b0 = zq_mm_mul_ps(a0, slope_v);
					zq_mm_store_ps(pix_ptr, zq_mm_max_ps(a0, b0));
				}
			}
		}
	}
	else if (in_W % 4 == 2)
	{
		for (n = 0, im_ptr = in_tensor4D_data; n < in_N; n++, im_ptr += in_imStep)
		{
			for (c = 0, slice_ptr = im_ptr; c < in_C; c += zq_mm_align_size, slice_ptr += in_sliceStep)
			{
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + c);
#endif
				slope_v = zq_mm_load_ps(slope + c);
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W-2; w += 4, pix_ptr += zq_mm_align_size4)
					{
						a0 = zq_mm_load_ps(pix_ptr);
						a1 = zq_mm_load_ps(pix_ptr + zq_mm_align_size);
						a2 = zq_mm_load_ps(pix_ptr + zq_mm_align_size2);
						a3 = zq_mm_load_ps(pix_ptr + zq_mm_align_size3);
#if WITH_BIAS
						a0 = zq_mm_add_ps(a0, bias_v);
						a1 = zq_mm_add_ps(a1, bias_v);
						a2 = zq_mm_add_ps(a2, bias_v);
						a3 = zq_mm_add_ps(a3, bias_v);
#endif
						b0 = zq_mm_mul_ps(a0, slope_v);
						b1 = zq_mm_mul_ps(a1, slope_v);
						b2 = zq_mm_mul_ps(a2, slope_v);
						b3 = zq_mm_mul_ps(a3, slope_v);
						zq_mm_store_ps(pix_ptr, zq_mm_max_ps(a0, b0));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size, zq_mm_max_ps(a1, b1));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size2, zq_mm_max_ps(a2, b2));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size3, zq_mm_max_ps(a3, b3));
					}
					a0 = zq_mm_load_ps(pix_ptr);
					a1 = zq_mm_load_ps(pix_ptr + zq_mm_align_size);
#if WITH_BIAS
					a0 = zq_mm_add_ps(a0, bias_v);
					a1 = zq_mm_add_ps(a1, bias_v);
#endif
					b0 = zq_mm_mul_ps(a0, slope_v);
					b1 = zq_mm_mul_ps(a1, slope_v);
					zq_mm_store_ps(pix_ptr, zq_mm_max_ps(a0, b0));
					zq_mm_store_ps(pix_ptr + zq_mm_align_size, zq_mm_max_ps(a1, b1));
				}
			}
		}
	}
	else if (in_W % 4 == 3)
	{
		for (n = 0, im_ptr = in_tensor4D_data; n < in_N; n++, im_ptr += in_imStep)
		{
			for (c = 0, slice_ptr = im_ptr; c < in_C; c += zq_mm_align_size, slice_ptr += in_sliceStep)
			{
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + c);
#endif
				slope_v = zq_mm_load_ps(slope + c);
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W-3; w += 4, pix_ptr += zq_mm_align_size4)
					{
						a0 = zq_mm_load_ps(pix_ptr);
						a1 = zq_mm_load_ps(pix_ptr + zq_mm_align_size);
						a2 = zq_mm_load_ps(pix_ptr + zq_mm_align_size2);
						a3 = zq_mm_load_ps(pix_ptr + zq_mm_align_size3);
#if WITH_BIAS
						a0 = zq_mm_add_ps(a0, bias_v);
						a1 = zq_mm_add_ps(a1, bias_v);
						a2 = zq_mm_add_ps(a2, bias_v);
						a3 = zq_mm_add_ps(a3, bias_v);
#endif
						b0 = zq_mm_mul_ps(a0, slope_v);
						b1 = zq_mm_mul_ps(a1, slope_v);
						b2 = zq_mm_mul_ps(a2, slope_v);
						b3 = zq_mm_mul_ps(a3, slope_v);
						zq_mm_store_ps(pix_ptr, zq_mm_max_ps(a0, b0));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size, zq_mm_max_ps(a1, b1));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size2, zq_mm_max_ps(a2, b2));
						zq_mm_store_ps(pix_ptr + zq_mm_align_size3, zq_mm_max_ps(a3, b3));
					}
					a0 = zq_mm_load_ps(pix_ptr);
					a1 = zq_mm_load_ps(pix_ptr + zq_mm_align_size);
					a2 = zq_mm_load_ps(pix_ptr + zq_mm_align_size2);
#if WITH_BIAS
					a0 = zq_mm_add_ps(a0, bias_v);
					a1 = zq_mm_add_ps(a1, bias_v);
					a2 = zq_mm_add_ps(a2, bias_v);
#endif
					b0 = zq_mm_mul_ps(a0, slope_v);
					b1 = zq_mm_mul_ps(a1, slope_v);
					b2 = zq_mm_mul_ps(a2, slope_v);
					zq_mm_store_ps(pix_ptr, zq_mm_max_ps(a0, b0));
					zq_mm_store_ps(pix_ptr + zq_mm_align_size, zq_mm_max_ps(a1, b1));
					zq_mm_store_ps(pix_ptr + zq_mm_align_size2, zq_mm_max_ps(a2, b2));
				}
			}
		}
	}
	
}