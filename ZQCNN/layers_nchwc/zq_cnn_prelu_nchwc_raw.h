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
	register zq_mm_type value_v;
	int n, h, w, c;
	register zq_mm_type slope_v;
#if WITH_BIAS
	register zq_mm_type bias_v;
#endif
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *im_ptr;
	register zq_mm_type zero_v = zq_mm_setzero_ps();


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
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += zq_mm_align_size)
				{
					value_v = zq_mm_load_ps(pix_ptr);
#if WITH_BIAS
					value_v = zq_mm_add_ps(value_v, bias_v);
#endif
					zq_mm_store_ps(pix_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zero_v, value_v), zq_mm_max_ps(zero_v, value_v)));
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
	register zq_mm_type data_v;
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *im_ptr;
	register zq_mm_type slope_v;
#if WITH_BIAS
	register zq_mm_type bias_v;
#endif
	

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
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += zq_mm_align_size)
				{
					data_v = zq_mm_load_ps(pix_ptr);
#if WITH_BIAS
					data_v = zq_mm_add_ps(data_v, bias_v);
#endif
					zq_mm_store_ps(pix_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, slope_v)));
				}
			}
		}
	}
}