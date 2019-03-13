void zq_cnn_dropout_32f_align(
	zq_base_type* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type dropout_ratio
)
{
	zq_base_type scale = 1.0f - dropout_ratio;
	register zq_mm_type scale_vec;
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;

	if (scale == 1.0f)
	{
		return;
	}
	else
	{
		scale_vec = zq_mm_set1_ps(scale);
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr; c < in_C; c+=zq_mm_align_size, c_ptr+=zq_mm_align_size)
					{
						zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), scale_vec));
						*c_ptr *= scale;
					}
				}
			}
		}
	}
}