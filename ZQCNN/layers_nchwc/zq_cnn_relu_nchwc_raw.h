/*
y = slope*min(0, x) + max(0, x)
*/
void zq_cnn_relu_nchwc(
	zq_base_type* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	zq_base_type slope
)
{

	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *im_ptr;
	register zq_mm_type a0;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
	register zq_mm_type slope_v = zq_mm_set1_ps(slope);

	if (slope == 0)
	{
		for (n = 0, im_ptr = in_tensor4D_data; n < in_N; n++, im_ptr += in_imStep)
		{
			for (c = 0, slice_ptr = im_ptr; c < in_C; c += zq_mm_align_size, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += zq_mm_align_size)
					{
						zq_mm_store_ps(pix_ptr, zq_mm_max_ps(zero_v, zq_mm_load_ps(pix_ptr)));
					}
				}
			}
		}
	}
	else
	{
		for (n = 0, im_ptr = in_tensor4D_data; n < in_N; n++, im_ptr += in_imStep)
		{
			for (c = 0, slice_ptr = im_ptr; c < in_C; c += zq_mm_align_size, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += zq_mm_align_size)
					{
						a0 = zq_mm_load_ps(pix_ptr);
						zq_mm_store_ps(pix_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zero_v, a0), zq_mm_max_ps(zero_v, a0)));
					}
				}
			}
		}
	}
}